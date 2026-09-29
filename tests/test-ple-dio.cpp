// [TAG_FN_PLE_DIRECT_IO] rows read by llama_ple_dio (unbuffered, from the file) against the same rows read through a
// memory mapping of the file (the path the model uses without LLAMA_PLE_DIRECT_IO), on a real GGUF. CPU and disk only.
//
//   test-ple-dio [-m model.gguf | first-shard-00001-of-0000N.gguf] [-t tensor] [-n random-ids] [--seed S] [--qd N]
//
// Default: the Flash-Next file A and per_layer_token_embd.weight (iq4_nl [160, 320,001,536], 90 B rows).
// Checks, each with the raw bytes (memcmp) and the dequantized floats (same to_float as the CPU GET_ROWS, bitwise):
//   1. edge rows (first, last), rows that straddle a sector boundary and random rows, cache off, one row per call
//   2. decode-sized calls (16 and 48 rows) with repeated rows, cache off
//   3. one call larger than the queue depth (the in-flight window refills), cache off
//   4. the same calls twice with the LRU on: the second pass must be served from the LRU
//   5. queue depth 1 (strictly one read at a time)
//   6. throughput by queue depth (printed, not checked)
//   7. llama_ple_dio_copy (LLAMA_PLE_DIO_FILE) of ~3.3 MB of rows: rows read from the copy vs the mapping
// The mapping is touched only for the tested rows (a few MB of pages). Built, not registered with ctest (needs the file).

#include "llama-ple-dio.h"

#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <fcntl.h>
#    include <sys/mman.h>
#    include <sys/stat.h>
#    include <unistd.h>
#endif

static const char * DEFAULT_MODEL = "D:/Projects/LocalAI/models/Qwen3.8-Flash-Next-UD-Q4_K_XL-MTP-00001-of-00005.gguf";

struct table_loc {
    std::string path;
    uint64_t    offset = 0;
    ggml_type   type   = GGML_TYPE_F32;
    int64_t     ne0    = 0;
    int64_t     n_rows = 0;
    size_t      rb     = 0;
};

static bool find_in_file(const std::string & path, const std::string & name, table_loc & loc, int & n_split) {
    ggml_context * ctx = nullptr;
    gguf_init_params ip = { /*.no_alloc =*/ true, /*.ctx =*/ &ctx };
    gguf_context * g = gguf_init_from_file(path.c_str(), ip);
    if (!g) {
        fprintf(stderr, "cannot read GGUF %s\n", path.c_str());
        return false;
    }
    const int64_t kid = gguf_find_key(g, "split.count");
    n_split = kid >= 0 ? (int) gguf_get_val_u16(g, kid) : 1;
    bool found = false;
    const int64_t tid = gguf_find_tensor(g, name.c_str());
    if (tid >= 0) {
        const ggml_tensor * t = ggml_get_tensor(ctx, name.c_str());
        loc.path   = path;
        loc.offset = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, tid);
        loc.type   = t->type;
        loc.ne0    = t->ne[0];
        loc.n_rows = t->ne[1];
        loc.rb     = t->nb[1];
        found = t->ne[2] == 1 && t->ne[3] == 1;
    }
    gguf_free(g);
    ggml_free(ctx);
    return found;
}

// read-only mapping of a whole file; only the touched pages are read in
struct file_map {
    const uint8_t * addr = nullptr;
    uint64_t        size = 0;
#if defined(_WIN32)
    HANDLE h = INVALID_HANDLE_VALUE, hm = nullptr;
    bool open(const std::string & path) {
        const int wn = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
        std::wstring w(wn, L'\0');
        MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, &w[0], wn);
        h = CreateFileW(w.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) {
            return false;
        }
        LARGE_INTEGER sz;
        GetFileSizeEx(h, &sz);
        size = (uint64_t) sz.QuadPart;
        hm = CreateFileMappingW(h, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!hm) {
            return false;
        }
        addr = (const uint8_t *) MapViewOfFile(hm, FILE_MAP_READ, 0, 0, 0);
        return addr != nullptr;
    }
    ~file_map() {
        if (addr) { UnmapViewOfFile(addr); }
        if (hm) { CloseHandle(hm); }
        if (h != INVALID_HANDLE_VALUE) { CloseHandle(h); }
    }
#else
    int fd = -1;
    bool open(const std::string & path) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            return false;
        }
        struct stat sb {};
        fstat(fd, &sb);
        size = (uint64_t) sb.st_size;
        void * p = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) {
            return false;
        }
        addr = (const uint8_t *) p;
        return true;
    }
    ~file_map() {
        if (addr) { munmap((void *) addr, size); }
        if (fd >= 0) { close(fd); }
    }
#endif
};

static int g_fail = 0;

static void log_line(const char * line) {
    fputs(line, stdout);
}

// one read_rows call per batch; compares bytes and floats against the mapping
static void check(llama_ple_dio & dio, const table_loc & loc, const file_map & map, const std::vector<int32_t> & ids,
        size_t batch, const char * what) {
    const ggml_to_float_t to_float = ggml_get_type_traits(loc.type)->to_float;
    std::vector<uint8_t> got;
    std::vector<float>   f_dio(loc.ne0), f_map(loc.ne0);
    int bad_bytes = 0, bad_floats = 0;
    for (size_t b0 = 0; b0 < ids.size(); b0 += batch) {
        const size_t n = std::min(batch, ids.size() - b0);
        got.assign(n * loc.rb, 0xCD);
        dio.read_rows(ids.data() + b0, (int64_t) n, got.data(), (int64_t) std::max<size_t>(1, n / 16), nullptr);
        for (size_t i = 0; i < n; ++i) {
            const int32_t   r   = ids[b0 + i];
            const uint8_t * ref = map.addr + loc.offset + (uint64_t) r * loc.rb;
            const uint8_t * dd  = got.data() + i * loc.rb;
            if (memcmp(ref, dd, loc.rb) != 0) {
                if (bad_bytes++ < 5) {
                    fprintf(stderr, "  %s: row %d bytes differ\n", what, r);
                }
                continue;
            }
            if (loc.type == GGML_TYPE_F32) {
                memcpy(f_dio.data(), dd, loc.rb);
                memcpy(f_map.data(), ref, loc.rb);
            } else {
                to_float(dd, f_dio.data(), loc.ne0);
                to_float(ref, f_map.data(), loc.ne0);
            }
            if (memcmp(f_dio.data(), f_map.data(), loc.ne0 * sizeof(float)) != 0) {
                bad_floats++;
            }
        }
    }
    const bool ok = bad_bytes == 0 && bad_floats == 0;
    printf("  %-44s %5zu rows, calls of %4zu: %s (%d byte mismatches, %d float mismatches)\n", what, ids.size(), batch,
            ok ? "OK" : "FAIL", bad_bytes, bad_floats);
    g_fail += ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    std::string model  = DEFAULT_MODEL;
    std::string tensor = "per_layer_token_embd.weight";
    int         n_rand = 512;
    unsigned    seed   = 1234;
    int         qd     = 64;
    bool        probe  = false; // only the throughput probe: no mapping, no compare
    std::string copy_dir = "E:/turbot-gates/flashnext/ple";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); }
            return argv[++i];
        };
        if (a == "-m") { model = next(); }
        else if (a == "-t") { tensor = next(); }
        else if (a == "-n") { n_rand = atoi(next()); }
        else if (a == "--seed") { seed = (unsigned) atoi(next()); }
        else if (a == "--qd") { qd = atoi(next()); }
        else if (a == "--probe-only") { probe = true; }
        else if (a == "--copy-dir") { copy_dir = next(); }
        else { fprintf(stderr, "usage: %s [-m model.gguf] [-t tensor] [-n random-ids] [--seed S] [--qd N] [--probe-only] [--copy-dir D]\n", argv[0]); return 2; }
    }

    // locate the tensor in the file or in the shards of a split
    table_loc loc;
    int n_split = 1;
    bool found = find_in_file(model, tensor, loc, n_split);
    if (!found && n_split > 1) {
        std::vector<char> buf(4096);
        if (!llama_split_prefix(buf.data(), buf.size(), model.c_str(), 0, n_split)) {
            fprintf(stderr, "%s is not the first shard of a %d-way split\n", model.c_str(), n_split);
            return 2;
        }
        const std::string prefix = buf.data();
        for (int s = 1; s < n_split && !found; ++s) {
            llama_split_path(buf.data(), buf.size(), prefix.c_str(), s, n_split);
            int dummy = 0;
            found = find_in_file(buf.data(), tensor, loc, dummy);
        }
    }
    if (!found) {
        fprintf(stderr, "tensor %s not found in %s\n", tensor.c_str(), model.c_str());
        return 2;
    }
    if (loc.type != GGML_TYPE_F32 && ggml_get_type_traits(loc.type)->to_float == nullptr) {
        fprintf(stderr, "type %s has no to_float\n", ggml_type_name(loc.type));
        return 2;
    }
    printf("table %s: %s [%" PRId64 ", %" PRId64 "], %zu B rows, in %s @ %" PRIu64 "\n", tensor.c_str(),
            ggml_type_name(loc.type), loc.ne0, loc.n_rows, loc.rb, loc.path.c_str(), loc.offset);

    file_map map;
    if (!probe && !map.open(loc.path)) {
        fprintf(stderr, "cannot map %s\n", loc.path.c_str());
        return 2;
    }

    llama_ple_dio_params p;
    p.path        = loc.path;
    p.offset      = loc.offset;
    p.row_bytes   = loc.rb;
    p.n_rows      = loc.n_rows;
    p.queue_depth = qd;
    p.stats_every = 0;
    p.log         = log_line;

    auto open_dio = [&](size_t cache_bytes, int depth) {
        llama_ple_dio_params q = p;
        q.cache_bytes = cache_bytes;
        q.queue_depth = depth;
        std::string err;
        auto d = llama_ple_dio::open(q, err);
        if (!d) {
            fprintf(stderr, "llama_ple_dio::open failed: %s\n", err.c_str());
            exit(1);
        }
        return d;
    };

    // the rows under test
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int64_t> any_row(0, loc.n_rows - 1);
    std::vector<int32_t> edge = { 0, 1, (int32_t) (loc.n_rows - 2), (int32_t) (loc.n_rows - 1) };
    std::vector<int32_t> straddle; // rows that hold the last byte before a 4096 boundary and the first after it
    while ((int) straddle.size() < 64) {
        const uint64_t off = loc.offset + (uint64_t) any_row(rng) * loc.rb;
        const uint64_t B   = (off + 4095) & ~(uint64_t) 4095;
        if (B <= loc.offset || B >= loc.offset + (uint64_t) loc.n_rows * loc.rb) {
            continue;
        }
        const int64_t r = (int64_t) ((B - 1 - loc.offset) / loc.rb);
        if (loc.offset + (uint64_t) r * loc.rb + loc.rb > B) {
            straddle.push_back((int32_t) r);
        }
    }
    std::vector<int32_t> rnd(n_rand);
    for (auto & r : rnd) {
        r = (int32_t) any_row(rng);
    }
    // decode-shaped: 16 rows per token, some n-grams repeated (the EOS-cut windows hash to the same rows)
    std::vector<int32_t> dec;
    for (int i = 0; i < 48 * 4; ++i) {
        const int32_t r = i % 7 == 3 ? dec[i / 2] : (int32_t) any_row(rng);
        dec.push_back(r);
    }

    std::vector<int32_t> all = edge;
    all.insert(all.end(), straddle.begin(), straddle.end());
    all.insert(all.end(), rnd.begin(), rnd.end());

    // 1-3: cache off
    if (!probe) {
        auto dio = open_dio(0, qd);
        printf("direct reads: %s, %zu B aligned, queue depth %d\n", dio->direct() ? "unbuffered" : "pread + DONTNEED",
                dio->align(), dio->params().queue_depth);
        check(*dio, loc, map, edge,     1,  "1. edge rows, one per call");
        check(*dio, loc, map, straddle, 1,  "1. sector-straddling rows, one per call");
        check(*dio, loc, map, rnd,      1,  "1. random rows, one per call");
        check(*dio, loc, map, dec,      16, "2. decode calls (16 rows, repeats)");
        check(*dio, loc, map, dec,      48, "2. decode calls (48 rows, repeats)");
        check(*dio, loc, map, all,      all.size(), "3. one call > queue depth");
        const auto s = dio->stats();
        printf("  stats: %s\n", dio->stats_line().c_str());
        if (s.hits != 0 || s.fallbacks != 0) {
            printf("  FAIL: cache off but %llu hits, %llu fallbacks\n", (unsigned long long) s.hits, (unsigned long long) s.fallbacks);
            g_fail++;
        }
    }
    // 4: LRU on, every row read twice
    if (!probe) {
        auto dio = open_dio((size_t) 64 << 20, qd);
        check(*dio, loc, map, all, 48, "4. LRU 64 MiB, first pass");
        const auto s1 = dio->stats();
        check(*dio, loc, map, all, 48, "4. LRU 64 MiB, second pass");
        const auto s2 = dio->stats();
        const uint64_t hits2  = s2.hits - s1.hits;
        const uint64_t reads2 = s2.reads - s1.reads;
        printf("  second pass: %llu LRU hits, %llu reads\n", (unsigned long long) hits2, (unsigned long long) reads2);
        if (reads2 != 0 || hits2 == 0) {
            printf("  FAIL: the second pass should be served from the LRU\n");
            g_fail++;
        }
        printf("  stats: %s\n", dio->stats_line().c_str());
    }
    // 4b: an LRU smaller than the working set: evictions must keep the table consistent
    if (!probe) {
        auto dio = open_dio(64 * loc.rb, qd);
        check(*dio, loc, map, all, 16, "4b. LRU of 64 rows (evicting)");
        check(*dio, loc, map, all, 16, "4b. LRU of 64 rows, again");
    }
    // 5: one read at a time
    if (!probe) {
        auto dio = open_dio(0, 1);
        check(*dio, loc, map, dec, 48, "5. queue depth 1");
        printf("  stats: %s\n", dio->stats_line().c_str());
    }

    // 6: throughput by queue depth (timing only, not checked): 1024 distinct random rows in one call
    {
        std::vector<int32_t> ids(1024);
        for (auto & r : ids) {
            r = (int32_t) any_row(rng);
        }
        std::vector<uint8_t> buf(ids.size() * loc.rb);
        for (const int depth : { 1, 8, 64, 256 }) {
            auto dio = open_dio(0, depth);
            dio->read_rows(ids.data(), (int64_t) ids.size(), buf.data(), 64, nullptr);
            const auto s = dio->stats();
            printf("  6. queue depth %3d: %zu rows in %.1f ms = %.0f reads/s, %.1f us/read, %.1f us to issue, %llu done inline\n",
                    depth, ids.size(), s.call_us/1000.0, s.reads/(s.call_us*1e-6), s.reads ? s.read_us/s.reads : 0.0,
                    s.reads ? s.issue_us/s.reads : 0.0, (unsigned long long) s.inline_done);
        }
    }

    // 7: the copy that LLAMA_PLE_DIO_FILE reads: unaligned start, several 1 MiB chunks, a partial last sector
    if (!probe) {
        const int64_t     r0  = loc.n_rows / 3 + 1;
        const int64_t     nr  = std::min<int64_t>(loc.n_rows - r0, (int64_t) ((3330000 + loc.rb - 1) / loc.rb));
        const std::string dst = copy_dir + "/test-ple-dio-copy.bin";
        std::string err;
        if (!llama_ple_dio_copy(loc.path, loc.offset + (uint64_t) r0 * loc.rb, (uint64_t) nr * loc.rb, dst, 1 << 20, err, log_line)) {
            printf("  7. copy: FAIL (%s)\n", err.c_str());
            g_fail++;
        } else {
            const int64_t sz = llama_ple_dio_file_size(dst);
            llama_ple_dio_params q = p;
            q.path   = dst;
            q.offset = 0;
            q.n_rows = nr;
            auto cd = llama_ple_dio::open(q, err);
            if (sz != nr * (int64_t) loc.rb || !cd) {
                printf("  7. copy: FAIL (size %lld, expected %lld; %s)\n", (long long) sz, (long long) (nr * (int64_t) loc.rb), err.c_str());
                g_fail++;
            } else {
                std::vector<int32_t> ids = { 0, 1, (int32_t) (nr - 2), (int32_t) (nr - 1) };
                std::uniform_int_distribution<int64_t> in_copy(0, nr - 1);
                for (int i = 0; i < 256; ++i) {
                    ids.push_back((int32_t) in_copy(rng));
                }
                std::vector<uint8_t> got(ids.size() * loc.rb);
                cd->read_rows(ids.data(), (int64_t) ids.size(), got.data(), 1, nullptr);
                int bad = 0;
                for (size_t i = 0; i < ids.size(); ++i) {
                    const uint8_t * ref = map.addr + loc.offset + (uint64_t) (r0 + ids[i]) * loc.rb;
                    bad += memcmp(ref, got.data() + i * loc.rb, loc.rb) != 0;
                }
                printf("  %-44s %5zu rows of a %lld-row copy: %s (%d byte mismatches)\n", "7. unmapped copy vs mapping",
                        ids.size(), (long long) nr, bad == 0 ? "OK" : "FAIL", bad);
                g_fail += bad == 0 ? 0 : 1;
            }
            cd.reset();
            std::remove(dst.c_str());
        }
    }

    printf("%s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
