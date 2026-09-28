// moe-cpu-bench: the CPU expert split of a MoE layer, measured on memory that does not stay in cache.
// [TAG_FN_CPU_MOE_BENCH] Flash-Next WP-CPU CP-0.
//
// The split is the one llama-graph builds for a layer whose experts live in host memory:
//   up = MUL_MAT_ID(up_exps, x, ids), gate = MUL_MAT_ID(gate_exps, x, ids), par = swiglu(gate, up),
//   out = MUL_MAT_ID(down_exps, par, ids)
// at Qwen3.8-Flash-Next shapes by default (n_embd 2560, n_ff 640, 512 experts, 10 used) for T = 1, 2, 3, 4, 8 tokens.
//
// Weights:
//   --random (default): a pool of at least --pool-gib GiB of random blocks, cut into layer instances. Every timed
//                       call takes the next instance and new random ids, so the experts come from DRAM.
//   --gguf FILE:        the real blk.N.ffn_{up,gate,down}_exps of one model shard, mapped read-only (file pages,
//                       no private copy). Layers are grouped by their three weight types.
// Per T it reports the split time and GiB/s over the expert bytes the call reads (distinct routed experts), the same
// for each matrix alone, the fixed cost of a split (every expert skipped with the src[3] table, so only the
// quantization, routing and barriers remain) and a streaming-read roofline taken with the same threads.
// --check compares each CPU expert path switch with all switches off on the same data; --ab NAME=V[,NAME=V] runs
// paired, alternating rounds of the current switches (A) and the same plus NAME=V (B). --pool N also runs the split
// through the persistent CPU MoE worker pool (ggml_cpu_moe_run, N workers including the caller), as the GPU/CPU
// doorbell would, and --check then compares it with the graph too.
// CPU only: it links ggml-base and ggml-cpu, never a GPU backend.

#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
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

namespace {

struct bench_params {
    std::string      gguf_path;          // empty: random pool
    ggml_type        type_up   = GGML_TYPE_Q4_K;
    ggml_type        type_gate = GGML_TYPE_Q4_K;
    ggml_type        type_down = GGML_TYPE_Q5_1;
    bool             all_types = false;  // random mode: every gate/up x down type pair of the plan
    double           pool_gib  = 8.0;
    int64_t          n_embd    = 2560;
    int64_t          n_ff      = 640;
    int              n_expert  = 512;
    int              n_used    = 10;
    std::vector<int> tokens    = { 1, 2, 3, 4, 8 };
    int              n_threads = 16;
    uint64_t         cpumask   = 0;      // 0: no mask
    bool             strict    = false;
    int              poll      = 50;
    int              prio      = 0;      // enum ggml_sched_priority
    int              iters     = 200;
    int              warmup    = 20;
    int              rounds    = 6;      // --ab
    bool             check     = false;
    bool             per_op    = true;
    int              pool_threads = 0;   // --pool: 0 = off
    int              pool_spin_us = 1000;
    std::vector<std::pair<int, int>> ab; // switch, value
};

// the fp16 scale fields at the start of a block (all of them must hold normal numbers in random blocks)
int leading_fp16_fields(ggml_type t) {
    switch (t) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS: return 1;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:   return 2;
        default:               return -1;
    }
}

ggml_type parse_type(const char * s) {
    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        const char * name = ggml_type_name((ggml_type) i);
        if (name && strcmp(name, s) == 0) {
            return (ggml_type) i;
        }
    }
    fprintf(stderr, "unknown type %s\n", s);
    exit(1);
}

int parse_switch(const char * name) {
    for (int i = 0; i < GGML_CPU_FN_SWITCH_COUNT; i++) {
        if (strcmp(ggml_cpu_fn_switch_env((ggml_cpu_fn_switch) i), name) == 0) {
            return i;
        }
    }
    fprintf(stderr, "unknown switch %s (use the environment variable name, e.g. GGML_CPU_MMID_MR)\n", name);
    exit(1);
}

std::vector<int> parse_int_list(const char * s) {
    std::vector<int> r;
    std::string cur;
    for (const char * p = s; ; p++) {
        if (*p == ',' || *p == 0) {
            if (!cur.empty()) {
                r.push_back(atoi(cur.c_str()));
            }
            cur.clear();
            if (*p == 0) {
                break;
            }
        } else {
            cur += *p;
        }
    }
    return r;
}

void usage(const char * argv0) {
    printf("usage: %s [options]\n", argv0);
    printf("  --random               random weight pool (default)\n");
    printf("  --gguf FILE            real experts of a model shard, mapped read-only\n");
    printf("  --pool-gib F           random pool size (default 8)\n");
    printf("  --up T --gate T --down T   weight types in random mode (default q4_K q4_K q5_1)\n");
    printf("  --all                  random mode: gate/up q4_K, q5_K x down q5_1, q8_0, iq4_nl\n");
    printf("  --n-embd N --n-ff N --n-expert N --n-used N   shapes (default 2560 640 512 10)\n");
    printf("  --tokens 1,2,3,4,8     token counts\n");
    printf("  --threads N            (default 16)\n");
    printf("  --cpumask HEX          e.g. 0x55555555 (one thread per physical core on SMT pairs)\n");
    printf("  --strict 0|1           one CPU per thread from the mask\n");
    printf("  --poll N               threadpool polling level (not used by OpenMP builds)\n");
    printf("  --prio N               -1 low, 0 normal, 1 medium, 2 high, 3 realtime\n");
    printf("  --iters N --warmup N   timed / untimed calls per measurement (default 200 / 20)\n");
    printf("  --no-per-op            skip the single-matrix timings\n");
    printf("  --check                every switch against all switches off, same data\n");
    printf("  --ab NAME=V[,NAME=V]   paired alternating rounds: current switches vs these added\n");
    printf("  --rounds N             rounds for --ab (default 6)\n");
    printf("  --pool N               also run the split on the CPU MoE worker pool with N workers (mask from --cpumask)\n");
    printf("  --pool-spin-us N       pool workers spin this long before sleeping (default 1000)\n");
    printf("switches (environment variables, read at start): ");
    for (int i = 0; i < GGML_CPU_FN_SWITCH_COUNT; i++) {
        printf("%s%s", i ? ", " : "", ggml_cpu_fn_switch_env((ggml_cpu_fn_switch) i));
    }
    printf("\n");
}

bench_params parse_args(int argc, char ** argv) {
    bench_params p;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", a.c_str());
                exit(1);
            }
            return argv[++i];
        };
        if (a == "--random") {
            p.gguf_path.clear();
        } else if (a == "--gguf") {
            p.gguf_path = next();
        } else if (a == "--pool-gib") {
            p.pool_gib = atof(next());
        } else if (a == "--up") {
            p.type_up = parse_type(next());
        } else if (a == "--gate") {
            p.type_gate = parse_type(next());
        } else if (a == "--down") {
            p.type_down = parse_type(next());
        } else if (a == "--all") {
            p.all_types = true;
        } else if (a == "--n-embd") {
            p.n_embd = atoll(next());
        } else if (a == "--n-ff") {
            p.n_ff = atoll(next());
        } else if (a == "--n-expert") {
            p.n_expert = atoi(next());
        } else if (a == "--n-used") {
            p.n_used = atoi(next());
        } else if (a == "--tokens") {
            p.tokens = parse_int_list(next());
        } else if (a == "--threads" || a == "-t") {
            p.n_threads = atoi(next());
        } else if (a == "--cpumask") {
            p.cpumask = strtoull(next(), nullptr, 16);
        } else if (a == "--strict") {
            p.strict = atoi(next()) != 0;
        } else if (a == "--poll") {
            p.poll = atoi(next());
        } else if (a == "--prio") {
            p.prio = atoi(next());
        } else if (a == "--iters") {
            p.iters = atoi(next());
        } else if (a == "--warmup") {
            p.warmup = atoi(next());
        } else if (a == "--rounds") {
            p.rounds = atoi(next());
        } else if (a == "--no-per-op") {
            p.per_op = false;
        } else if (a == "--check") {
            p.check = true;
        } else if (a == "--pool") {
            p.pool_threads = atoi(next());
        } else if (a == "--pool-spin-us") {
            p.pool_spin_us = atoi(next());
        } else if (a == "--ab") {
            std::string list = next();
            size_t pos = 0;
            while (pos < list.size()) {
                size_t end = list.find(',', pos);
                if (end == std::string::npos) {
                    end = list.size();
                }
                const std::string item = list.substr(pos, end - pos);
                const size_t eq = item.find('=');
                if (eq == std::string::npos) {
                    fprintf(stderr, "--ab expects NAME=V, got %s\n", item.c_str());
                    exit(1);
                }
                p.ab.emplace_back(parse_switch(item.substr(0, eq).c_str()), atoi(item.c_str() + eq + 1));
                pos = end + 1;
            }
        } else if (a == "-h" || a == "--help") {
            usage(argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "unknown argument %s\n", a.c_str());
            usage(argv[0]);
            exit(1);
        }
    }
    if (p.n_threads < 1 || p.n_used < 1 || p.n_used > p.n_expert || p.iters < 1 || p.tokens.empty()) {
        fprintf(stderr, "invalid arguments\n");
        exit(1);
    }
    return p;
}

// ---- weights ----

struct layer_ptrs {
    const uint8_t * up;
    const uint8_t * gate;
    const uint8_t * down;
    int             il;  // model layer (gguf mode) or instance index
};

struct weight_set {
    ggml_type up_t, gate_t, down_t;
    int64_t   n_embd, n_ff;
    int       n_expert;
    std::vector<layer_ptrs> layers;
    std::vector<std::pair<const uint8_t *, size_t>> regions; // for the read roofline
};

size_t expert_bytes(ggml_type t, int64_t ne0, int64_t ne1) {
    return ggml_row_size(t, ne0) * (size_t) ne1;
}

// random blocks: random bytes, then the leading fp16 fields set to small normal values
void fill_random_blocks(uint8_t * p, size_t nbytes, ggml_type t, uint64_t seed) {
    const size_t bsize = ggml_type_size(t);
    const int    nhalf = leading_fp16_fields(t);
    uint64_t s = seed * 0x9E3779B97F4A7C15ull + 1;
    size_t i = 0;
    for (; i + 8 <= nbytes; i += 8) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        memcpy(p + i, &s, 8);
    }
    for (; i < nbytes; i++) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        p[i] = (uint8_t) s;
    }
    const ggml_fp16_t h0 = ggml_fp32_to_fp16(0.0123f);
    const ggml_fp16_t h1 = ggml_fp32_to_fp16(0.0045f);
    for (size_t b = 0; b + bsize <= nbytes; b += bsize) {
        for (int h = 0; h < nhalf; h++) {
            memcpy(p + b + h * sizeof(ggml_fp16_t), h % 2 ? &h1 : &h0, sizeof(ggml_fp16_t));
        }
    }
}

struct random_pool {
    uint8_t * data  = nullptr;
    size_t    bytes = 0;
    ~random_pool() { free(data); }
};

bool make_random_weights(const bench_params & p, ggml_type up_t, ggml_type gate_t, ggml_type down_t, random_pool & pool, weight_set & ws) {
    for (ggml_type t : { up_t, gate_t, down_t }) {
        if (leading_fp16_fields(t) < 0) {
            fprintf(stderr, "type %s: random blocks not supported (scales are not at the block start)\n", ggml_type_name(t));
            return false;
        }
    }
    if (p.n_embd % ggml_blck_size(up_t) || p.n_embd % ggml_blck_size(gate_t) || p.n_ff % ggml_blck_size(down_t)) {
        fprintf(stderr, "types %s/%s/%s do not divide the rows (n_embd %" PRId64 ", n_ff %" PRId64 ")\n",
                ggml_type_name(up_t), ggml_type_name(gate_t), ggml_type_name(down_t), p.n_embd, p.n_ff);
        return false;
    }
    ws.up_t = up_t; ws.gate_t = gate_t; ws.down_t = down_t;
    ws.n_embd = p.n_embd; ws.n_ff = p.n_ff; ws.n_expert = p.n_expert;

    const size_t b_up   = expert_bytes(up_t,   p.n_embd, p.n_ff) * p.n_expert;
    const size_t b_gate = expert_bytes(gate_t, p.n_embd, p.n_ff) * p.n_expert;
    const size_t b_down = expert_bytes(down_t, p.n_ff, p.n_embd) * p.n_expert;
    const size_t b_layer = GGML_PAD(b_up, 64) + GGML_PAD(b_gate, 64) + GGML_PAD(b_down, 64);
    const int n_layers = std::max(1, (int) std::ceil(p.pool_gib * 1073741824.0 / (double) b_layer));

    pool.bytes = b_layer * n_layers;
    pool.data  = (uint8_t *) malloc(pool.bytes + 64);
    if (!pool.data) {
        fprintf(stderr, "failed to allocate %.2f GiB\n", pool.bytes / 1073741824.0);
        return false;
    }
    uint8_t * base = (uint8_t *) GGML_PAD((uintptr_t) pool.data, 64);

    // fill in parallel, each tensor with its own block type
    struct job { uint8_t * p; size_t n; ggml_type t; };
    std::vector<job> jobs;
    for (int l = 0; l < n_layers; l++) {
        uint8_t * lp = base + (size_t) l * b_layer;
        layer_ptrs lw = { lp, lp + GGML_PAD(b_up, 64), lp + GGML_PAD(b_up, 64) + GGML_PAD(b_gate, 64), l };
        ws.layers.push_back(lw);
        jobs.push_back({ (uint8_t *) lw.up,   b_up,   up_t });
        jobs.push_back({ (uint8_t *) lw.gate, b_gate, gate_t });
        jobs.push_back({ (uint8_t *) lw.down, b_down, down_t });
    }
    ws.regions.emplace_back(base, pool.bytes);

    std::vector<std::thread> th;
    const int nt = std::max(1, std::min(p.n_threads, 32));
    for (int t = 0; t < nt; t++) {
        th.emplace_back([&, t]() {
            for (size_t j = 0; j < jobs.size(); j++) {
                // split every tensor over the threads at block boundaries
                const size_t bsize = ggml_type_size(jobs[j].t);
                const size_t nblk  = jobs[j].n / bsize;
                const size_t b0 = nblk * t / nt;
                const size_t b1 = nblk * (t + 1) / nt;
                fill_random_blocks(jobs[j].p + b0 * bsize, (b1 - b0) * bsize, jobs[j].t, (uint64_t) j * 1000003u + t);
            }
        });
    }
    for (auto & x : th) {
        x.join();
    }
    printf("random pool: %d layer instances x %.1f MiB = %.2f GiB (up %s, gate %s, down %s)\n", n_layers,
           b_layer / 1048576.0, pool.bytes / 1073741824.0, ggml_type_name(up_t), ggml_type_name(gate_t), ggml_type_name(down_t));
    return true;
}

struct mapped_file {
    const uint8_t * base = nullptr;
    size_t          size = 0;
#ifdef _WIN32
    HANDLE hf = INVALID_HANDLE_VALUE;
    HANDLE hm = nullptr;
#else
    int fd = -1;
#endif
    bool open(const char * path) {
#ifdef _WIN32
        hf = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (hf == INVALID_HANDLE_VALUE) {
            return false;
        }
        LARGE_INTEGER sz;
        if (!GetFileSizeEx(hf, &sz)) {
            return false;
        }
        size = (size_t) sz.QuadPart;
        hm = CreateFileMappingA(hf, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!hm) {
            return false;
        }
        base = (const uint8_t *) MapViewOfFile(hm, FILE_MAP_READ, 0, 0, 0);
        return base != nullptr;
#else
        fd = ::open(path, O_RDONLY);
        if (fd < 0) {
            return false;
        }
        struct stat st;
        if (fstat(fd, &st) != 0) {
            return false;
        }
        size = (size_t) st.st_size;
        void * m = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        if (m == MAP_FAILED) {
            return false;
        }
        base = (const uint8_t *) m;
        return true;
#endif
    }
    ~mapped_file() {
#ifdef _WIN32
        if (base) {
            UnmapViewOfFile(base);
        }
        if (hm) {
            CloseHandle(hm);
        }
        if (hf != INVALID_HANDLE_VALUE) {
            CloseHandle(hf);
        }
#else
        if (base) {
            munmap((void *) base, size);
        }
        if (fd >= 0) {
            close(fd);
        }
#endif
    }
};

// the expert tensors of a shard, grouped by their (up, gate, down) types
bool load_gguf_weights(const char * path, mapped_file & mf, std::vector<weight_set> & sets) {
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * g = gguf_init_from_file(path, gp);
    if (!g) {
        fprintf(stderr, "failed to read %s\n", path);
        return false;
    }
    if (!mf.open(path)) {
        fprintf(stderr, "failed to map %s\n", path);
        gguf_free(g);
        return false;
    }
    const size_t data_off = gguf_get_data_offset(g);
    for (int il = 0; il < 1024; il++) {
        char nu[128], ng[128], nd[128];
        snprintf(nu, sizeof(nu), "blk.%d.ffn_up_exps.weight",   il);
        snprintf(ng, sizeof(ng), "blk.%d.ffn_gate_exps.weight", il);
        snprintf(nd, sizeof(nd), "blk.%d.ffn_down_exps.weight", il);
        const int64_t iu = gguf_find_tensor(g, nu);
        const int64_t ig = gguf_find_tensor(g, ng);
        const int64_t id = gguf_find_tensor(g, nd);
        if (iu < 0 || ig < 0 || id < 0) {
            continue;
        }
        const int64_t * neu = gguf_get_tensor_ne(g, iu);
        const int64_t * neg = gguf_get_tensor_ne(g, ig);
        const int64_t * ned = gguf_get_tensor_ne(g, id);
        if (neu[0] != neg[0] || neu[1] != neg[1] || neu[2] != neg[2] || ned[0] != neu[1] || ned[1] != neu[0] || ned[2] != neu[2]) {
            fprintf(stderr, "layer %d: unexpected expert shapes, skipped\n", il);
            continue;
        }
        const ggml_type tu = gguf_get_tensor_type(g, iu);
        const ggml_type tg = gguf_get_tensor_type(g, ig);
        const ggml_type td = gguf_get_tensor_type(g, id);
        const uint8_t * pu = mf.base + data_off + gguf_get_tensor_offset(g, iu);
        const uint8_t * pg = mf.base + data_off + gguf_get_tensor_offset(g, ig);
        const uint8_t * pd = mf.base + data_off + gguf_get_tensor_offset(g, id);
        const size_t su = gguf_get_tensor_size(g, iu);
        const size_t sg = gguf_get_tensor_size(g, ig);
        const size_t sd = gguf_get_tensor_size(g, id);
        if (pd + sd > mf.base + mf.size || pu + su > mf.base + mf.size || pg + sg > mf.base + mf.size) {
            fprintf(stderr, "layer %d: tensor data past the end of the file, skipped\n", il);
            continue;
        }
        weight_set * ws = nullptr;
        for (auto & s : sets) {
            if (s.up_t == tu && s.gate_t == tg && s.down_t == td) {
                ws = &s;
            }
        }
        if (!ws) {
            sets.emplace_back();
            ws = &sets.back();
            ws->up_t = tu; ws->gate_t = tg; ws->down_t = td;
            ws->n_embd = neu[0]; ws->n_ff = neu[1]; ws->n_expert = (int) neu[2];
        }
        ws->layers.push_back({ pu, pg, pd, il });
        ws->regions.emplace_back(pu, su);
        ws->regions.emplace_back(pg, sg);
        ws->regions.emplace_back(pd, sd);
    }
    gguf_free(g);
    if (sets.empty()) {
        fprintf(stderr, "%s holds no complete blk.N.ffn_{up,gate,down}_exps set\n", path);
        return false;
    }
    return true;
}

// ---- read roofline: every thread of the pool streams its share of the regions ----

struct read_job {
    const std::vector<std::pair<const uint8_t *, size_t>> * regions;
};

void read_regions(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    GGML_UNUSED(a);
    const read_job * job = (const read_job *) userdata;
    uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
    for (const auto & r : *job->regions) {
        const size_t n64 = r.second / 8;
        const size_t i0  = n64 * ith / nth;
        const size_t i1  = n64 * (ith + 1) / nth;
        const uint64_t * p = (const uint64_t *) r.first;
        size_t i = i0;
        for (; i + 8 <= i1; i += 8) {
            uint64_t v[8];
            memcpy(v, p + i, sizeof(v));
            a0 += v[0]; a1 ^= v[1]; a2 += v[2]; a3 ^= v[3];
            a4 += v[4]; a5 ^= v[5]; a6 += v[6]; a7 ^= v[7];
        }
        for (; i < i1; i++) {
            uint64_t v;
            memcpy(&v, p + i, sizeof(v));
            a0 += v;
        }
    }
    ((uint64_t *) dst->data)[ith] = a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7;
}

// ---- graphs ----

struct run_ctx {
    const bench_params & p;
    ggml_threadpool *    tp;
    ggml_cpu_moe_pool *  pool = nullptr;
    std::vector<uint8_t> work;

    explicit run_ctx(const bench_params & p_) : p(p_), tp(nullptr) {}

    double compute(ggml_cgraph * g) {
        ggml_cplan plan = ggml_graph_plan(g, p.n_threads, tp);
        if (plan.work_size > work.size()) {
            work.resize(plan.work_size);
        }
        plan.work_data = work.data();
        const int64_t t0 = ggml_time_us();
        const ggml_status st = ggml_graph_compute(g, &plan);
        const int64_t t1 = ggml_time_us();
        if (st != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "graph compute failed\n");
            exit(1);
        }
        return (double) (t1 - t0);
    }
};

struct split_graphs {
    int T = 0;
    ggml_context * ctx_w = nullptr; // weight headers (no data)
    ggml_context * ctx   = nullptr; // activations and graphs
    ggml_tensor * w_up = nullptr, * w_gate = nullptr, * w_down = nullptr;
    ggml_tensor * x = nullptr, * ids = nullptr, * tbl = nullptr, * par_in = nullptr;
    ggml_tensor * out = nullptr;
    ggml_cgraph * g_split = nullptr, * g_fixed = nullptr, * g_up = nullptr, * g_gate = nullptr, * g_down = nullptr;
    std::vector<float> pool_out; // [n_embd, n_used, T], graph layout

    ~split_graphs() {
        if (ctx)   { ggml_free(ctx); }
        if (ctx_w) { ggml_free(ctx_w); }
    }

    void build(const weight_set & ws, int n_used, int T_) {
        T = T_;
        {
            ggml_init_params ip = { ggml_tensor_overhead() * 8, nullptr, true };
            ctx_w = ggml_init(ip);
        }
        w_up   = ggml_new_tensor_3d(ctx_w, ws.up_t,   ws.n_embd, ws.n_ff,   ws.n_expert);
        w_gate = ggml_new_tensor_3d(ctx_w, ws.gate_t, ws.n_embd, ws.n_ff,   ws.n_expert);
        w_down = ggml_new_tensor_3d(ctx_w, ws.down_t, ws.n_ff,   ws.n_embd, ws.n_expert);
        ggml_set_name(w_up, "ffn_up_exps");
        ggml_set_name(w_gate, "ffn_gate_exps");
        ggml_set_name(w_down, "ffn_down_exps");

        // x, par_in, 9 [n_ff, n_used, T] and 3 [n_embd, n_used, T] results, ids, tbl, padding
        const size_t act = sizeof(float) * (size_t) (ws.n_embd * T + 10 * ws.n_ff * n_used * T + 3 * ws.n_embd * n_used * T)
                         + sizeof(int32_t) * (size_t) (n_used * T + ws.n_expert) + (1u << 20);
        ggml_init_params ip = { act + ggml_tensor_overhead() * 64 + ggml_graph_overhead() * 5, nullptr, false };
        ctx = ggml_init(ip);

        x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ws.n_embd, 1, T);
        ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, T);
        tbl = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ws.n_expert);
        par_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ws.n_ff, n_used, T);

        // the split, in llama-graph's order
        ggml_tensor * up   = ggml_mul_mat_id(ctx, w_up,   x, ids);
        ggml_tensor * gate = ggml_mul_mat_id(ctx, w_gate, x, ids);
        ggml_tensor * par  = ggml_swiglu_split(ctx, gate, up);
        out = ggml_mul_mat_id(ctx, w_down, par, ids);
        g_split = ggml_new_graph(ctx);
        ggml_build_forward_expand(g_split, out);

        // the same with every expert marked "on the device": only the fixed work remains
        ggml_tensor * fup   = ggml_mul_mat_id(ctx, w_up,   x, ids);
        ggml_tensor * fgate = ggml_mul_mat_id(ctx, w_gate, x, ids);
        ggml_tensor * fpar  = ggml_swiglu_split(ctx, fgate, fup);
        ggml_tensor * fout  = ggml_mul_mat_id(ctx, w_down, fpar, ids);
        for (ggml_tensor * t : { fup, fgate, fout }) {
            t->src[3]       = tbl;
            t->op_params[0] = ws.n_expert; // "not on the device"; tbl holds slot 0 for every expert
        }
        g_fixed = ggml_new_graph(ctx);
        ggml_build_forward_expand(g_fixed, fout);

        g_up = ggml_new_graph(ctx);
        ggml_build_forward_expand(g_up, ggml_mul_mat_id(ctx, w_up, x, ids));
        g_gate = ggml_new_graph(ctx);
        ggml_build_forward_expand(g_gate, ggml_mul_mat_id(ctx, w_gate, x, ids));
        g_down = ggml_new_graph(ctx);
        ggml_build_forward_expand(g_down, ggml_mul_mat_id(ctx, w_down, par_in, ids));

        std::mt19937 rng(1234 + T);
        std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
        for (int64_t i = 0; i < ggml_nelements(x); i++) {
            ((float *) x->data)[i] = uni(rng);
        }
        for (int64_t i = 0; i < ggml_nelements(par_in); i++) {
            ((float *) par_in->data)[i] = 0.1f * uni(rng);
        }
        for (int e = 0; e < ws.n_expert; e++) {
            ((int32_t *) tbl->data)[e] = 0;
        }
        pool_out.assign((size_t) ws.n_embd * n_used * T, 0.0f);
    }

    // the split on the worker pool; fixed: every expert skipped through the table
    double run_pool(ggml_cpu_moe_pool * pool, int n_expert, int n_used, bool fixed) {
        ggml_cpu_moe_layer layer = { w_up, w_gate, w_down, fixed ? (const int32_t *) tbl->data : nullptr, n_expert };
        ggml_cpu_moe_job job = { &layer, T, n_used, (const float *) x->data, (const int32_t *) ids->data, nullptr, pool_out.data() };
        const int64_t t0 = ggml_time_us();
        if (ggml_cpu_moe_run(pool, &job) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "ggml_cpu_moe_run refused the job (types %s/%s/%s)\n", ggml_type_name(w_up->type), ggml_type_name(w_gate->type), ggml_type_name(w_down->type));
            exit(1);
        }
        return (double) (ggml_time_us() - t0);
    }

    void set_layer(const layer_ptrs & l) {
        w_up->data   = (void *) l.up;
        w_gate->data = (void *) l.gate;
        w_down->data = (void *) l.down;
    }

    // new random top-n_used ids per token (distinct within a token); returns the number of distinct experts
    int set_random_ids(std::mt19937 & rng, int n_expert, int n_used) {
        std::vector<int32_t> perm(n_expert);
        std::vector<char> seen(n_expert, 0);
        int n_distinct = 0;
        int32_t * d = (int32_t *) ids->data;
        for (int t = 0; t < T; t++) {
            for (int e = 0; e < n_expert; e++) {
                perm[e] = e;
            }
            for (int k = 0; k < n_used; k++) {
                std::uniform_int_distribution<int> pick(k, n_expert - 1);
                std::swap(perm[k], perm[pick(rng)]);
                d[t * n_used + k] = perm[k];
                if (!seen[perm[k]]) {
                    seen[perm[k]] = 1;
                    n_distinct++;
                }
            }
        }
        return n_distinct;
    }
};

struct stat_acc {
    std::vector<double> us;
    double bytes = 0.0;
    double time  = 0.0;
    void add(double t_us, double b) { us.push_back(t_us); bytes += b; time += t_us; }
    double median() const {
        if (us.empty()) {
            return 0.0;
        }
        std::vector<double> v = us;
        std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
        return v[v.size() / 2];
    }
    double mean() const { return us.empty() ? 0.0 : time / us.size(); }
    double gibs() const { return time > 0 ? bytes / (time * 1e-6) / 1073741824.0 : 0.0; }
};

enum class which { split, fixed, up, gate, down, pool, pool_fixed };

stat_acc measure(run_ctx & rc, split_graphs & sg, const weight_set & ws, which w, int iters, int warmup, std::mt19937 & rng, size_t & layer_rr) {
    const size_t e_up   = expert_bytes(ws.up_t,   ws.n_embd, ws.n_ff);
    const size_t e_gate = expert_bytes(ws.gate_t, ws.n_embd, ws.n_ff);
    const size_t e_down = expert_bytes(ws.down_t, ws.n_ff,   ws.n_embd);
    ggml_cgraph * g = w == which::split ? sg.g_split : w == which::fixed ? sg.g_fixed :
                      w == which::up    ? sg.g_up    : w == which::gate  ? sg.g_gate  : sg.g_down;
    stat_acc st;
    for (int it = 0; it < warmup + iters; it++) {
        sg.set_layer(ws.layers[layer_rr++ % ws.layers.size()]);
        const int nd = sg.set_random_ids(rng, ws.n_expert, rc.p.n_used);
        const double t = w == which::pool || w == which::pool_fixed
            ? sg.run_pool(rc.pool, ws.n_expert, rc.p.n_used, w == which::pool_fixed)
            : rc.compute(g);
        double b = 0.0;
        switch (w) {
            case which::pool:
            case which::split: b = (double) nd * (e_up + e_gate + e_down); break;
            case which::pool_fixed:
            case which::fixed: b = 0.0; break;
            case which::up:    b = (double) nd * e_up;   break;
            case which::gate:  b = (double) nd * e_gate; break;
            case which::down:  b = (double) nd * e_down; break;
        }
        if (it >= warmup) {
            st.add(t, b);
        }
    }
    return st;
}

double read_roofline(run_ctx & rc, const weight_set & ws, int passes) {
    ggml_init_params ip = { ggml_tensor_overhead() * 4 + ggml_graph_overhead() + sizeof(uint64_t) * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2 * GGML_MAX_N_THREADS);
    read_job job = { &ws.regions };
    ggml_tensor * r = ggml_map_custom1(ctx, a, read_regions, GGML_N_TASKS_MAX, &job);
    ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, r);
    size_t bytes = 0;
    for (const auto & reg : ws.regions) {
        bytes += reg.second;
    }
    rc.compute(g); // first pass: page-in
    double t = 0.0;
    for (int i = 0; i < passes; i++) {
        t += rc.compute(g);
    }
    ggml_free(ctx);
    return (double) bytes * passes / (t * 1e-6) / 1073741824.0;
}

void set_switches(const std::vector<int> & v) {
    for (int i = 0; i < GGML_CPU_FN_SWITCH_COUNT; i++) {
        ggml_cpu_fn_set_switch((ggml_cpu_fn_switch) i, v[i]);
    }
}

std::vector<int> get_switches() {
    std::vector<int> v(GGML_CPU_FN_SWITCH_COUNT);
    for (int i = 0; i < GGML_CPU_FN_SWITCH_COUNT; i++) {
        v[i] = ggml_cpu_fn_get_switch((ggml_cpu_fn_switch) i);
    }
    return v;
}

std::string switches_str(const std::vector<int> & v) {
    std::string s;
    for (int i = 0; i < GGML_CPU_FN_SWITCH_COUNT; i++) {
        if (v[i] != 0) {
            s += (s.empty() ? "" : " ") + std::string(ggml_cpu_fn_switch_env((ggml_cpu_fn_switch) i)) + "=" + std::to_string(v[i]);
        }
    }
    return s.empty() ? "all off" : s;
}

// every switch (and GGML_CPU_MMID_MR=2) against all switches off, on one layer and one set of ids
int run_check(run_ctx & rc, split_graphs & sg, const weight_set & ws) {
    const std::vector<int> saved = get_switches();
    std::vector<int> base(GGML_CPU_FN_SWITCH_COUNT, 0);
    std::mt19937 rng(99);
    sg.set_layer(ws.layers[0]);
    sg.set_random_ids(rng, ws.n_expert, rc.p.n_used);

    set_switches(base);
    rc.compute(sg.g_split);
    const std::vector<float> ref((float *) sg.out->data, (float *) sg.out->data + ggml_nelements(sg.out));

    std::vector<std::vector<int>> variants;
    for (int i = 0; i < GGML_CPU_FN_SWITCH_COUNT; i++) {
        std::vector<int> v = base;
        v[i] = 1;
        variants.push_back(v);
        if (i == GGML_CPU_FN_MMID_MR) {
            v[i] = 2;
            variants.push_back(v);
        }
    }
    std::vector<int> all(GGML_CPU_FN_SWITCH_COUNT, 1);
    variants.push_back(all);

    int n_bad = 0;
    for (const auto & v : variants) {
        set_switches(v);
        rc.compute(sg.g_split);
        const float * o = (const float *) sg.out->data;
        size_t n_diff = 0;
        double max_abs = 0.0, max_rel = 0.0;
        for (size_t i = 0; i < ref.size(); i++) {
            if (memcmp(&o[i], &ref[i], sizeof(float)) != 0) {
                n_diff++;
                const double d = std::fabs((double) o[i] - ref[i]);
                max_abs = std::max(max_abs, d);
                max_rel = std::max(max_rel, d / std::max(1e-30, std::fabs((double) ref[i])));
            }
        }
        printf("  check T=%d %-40s %s", sg.T, switches_str(v).c_str(), n_diff == 0 ? "bitwise equal" : "DIFFERS");
        if (n_diff) {
            printf(": %zu of %zu values, max abs %.3g, max rel %.3g", n_diff, ref.size(), max_abs, max_rel);
            n_bad++;
        }
        printf("\n");
    }
    if (rc.pool) {
        set_switches(base);
        sg.run_pool(rc.pool, ws.n_expert, rc.p.n_used, false);
        size_t n_diff = 0;
        for (size_t i = 0; i < ref.size(); i++) {
            n_diff += memcmp(&sg.pool_out[i], &ref[i], sizeof(float)) != 0;
        }
        printf("  check T=%d %-40s %s\n", sg.T, "worker pool (ggml_cpu_moe_run)", n_diff == 0 ? "bitwise equal" : "DIFFERS");
        n_bad += n_diff != 0;
    }
    set_switches(saved);
    return n_bad;
}

void run_set(run_ctx & rc, const weight_set & ws, int & n_bad) {
    printf("\n== up %s, gate %s, down %s: %zu layer%s, n_embd %" PRId64 ", n_ff %" PRId64 ", %d experts, %d used\n",
           ggml_type_name(ws.up_t), ggml_type_name(ws.gate_t), ggml_type_name(ws.down_t), ws.layers.size(),
           ws.layers.size() == 1 ? "" : "s", ws.n_embd, ws.n_ff, ws.n_expert, rc.p.n_used);
    printf("   expert bytes: up %.1f KiB, gate %.1f KiB, down %.1f KiB\n",
           expert_bytes(ws.up_t, ws.n_embd, ws.n_ff) / 1024.0, expert_bytes(ws.gate_t, ws.n_embd, ws.n_ff) / 1024.0,
           expert_bytes(ws.down_t, ws.n_ff, ws.n_embd) / 1024.0);

    printf("   read roofline (same threads): %.1f GiB/s\n", read_roofline(rc, ws, 2));

    std::mt19937 rng(42);
    size_t layer_rr = 0;
    for (int T : rc.p.tokens) {
        split_graphs sg;
        sg.build(ws, rc.p.n_used, T);

        if (rc.p.check) {
            n_bad += run_check(rc, sg, ws);
        }

        if (!rc.p.ab.empty()) {
            const std::vector<int> a_sw = get_switches();
            std::vector<int> b_sw = a_sw;
            for (const auto & kv : rc.p.ab) {
                b_sw[kv.first] = kv.second;
            }
            std::vector<double> ma, mb;
            for (int r = 0; r < rc.p.rounds; r++) {
                set_switches(a_sw);
                ma.push_back(measure(rc, sg, ws, which::split, rc.p.iters, rc.p.warmup, rng, layer_rr).median());
                set_switches(b_sw);
                mb.push_back(measure(rc, sg, ws, which::split, rc.p.iters, rc.p.warmup, rng, layer_rr).median());
            }
            set_switches(a_sw);
            std::vector<double> sa = ma, sb = mb;
            std::sort(sa.begin(), sa.end());
            std::sort(sb.begin(), sb.end());
            const double a_med = sa[sa.size() / 2];
            const double b_med = sb[sb.size() / 2];
            printf("   T=%d A/B split us (median of %d paired rounds): A [%s] %.1f  B [%s] %.1f  (%+.2f %%)\n", T, rc.p.rounds,
                   switches_str(a_sw).c_str(), a_med, switches_str(b_sw).c_str(), b_med, 100.0 * (b_med - a_med) / a_med);
            printf("        rounds A:");
            for (double v : ma) { printf(" %.1f", v); }
            printf("   B:");
            for (double v : mb) { printf(" %.1f", v); }
            printf("\n");
            continue;
        }

        const stat_acc s_split = measure(rc, sg, ws, which::split, rc.p.iters, rc.p.warmup, rng, layer_rr);
        const stat_acc s_fixed = measure(rc, sg, ws, which::fixed, rc.p.iters, rc.p.warmup, rng, layer_rr);
        printf("   T=%d split: %8.1f us median %8.1f mean  %6.1f GiB/s   fixed (all experts skipped): %6.1f us\n",
               T, s_split.median(), s_split.mean(), s_split.gibs(), s_fixed.median());
        if (rc.pool) {
            const stat_acc s_pool  = measure(rc, sg, ws, which::pool,       rc.p.iters, rc.p.warmup, rng, layer_rr);
            const stat_acc s_pfix  = measure(rc, sg, ws, which::pool_fixed, rc.p.iters, rc.p.warmup, rng, layer_rr);
            printf("        pool (%d workers): %8.1f us median %8.1f mean  %6.1f GiB/s   fixed (all experts skipped): %6.1f us\n",
                   rc.p.pool_threads, s_pool.median(), s_pool.mean(), s_pool.gibs(), s_pfix.median());
        }
        if (rc.p.per_op) {
            const stat_acc s_up   = measure(rc, sg, ws, which::up,   rc.p.iters, rc.p.warmup, rng, layer_rr);
            const stat_acc s_gate = measure(rc, sg, ws, which::gate, rc.p.iters, rc.p.warmup, rng, layer_rr);
            const stat_acc s_down = measure(rc, sg, ws, which::down, rc.p.iters, rc.p.warmup, rng, layer_rr);
            printf("        up %-6s %7.1f us %6.1f GiB/s   gate %-6s %7.1f us %6.1f GiB/s   down %-6s %7.1f us %6.1f GiB/s\n",
                   ggml_type_name(ws.up_t), s_up.median(), s_up.gibs(), ggml_type_name(ws.gate_t), s_gate.median(), s_gate.gibs(),
                   ggml_type_name(ws.down_t), s_down.median(), s_down.gibs());
        }
    }
}

} // namespace

int main(int argc, char ** argv) {
    const bench_params p = parse_args(argc, argv);

    ggml_time_init();
    ggml_cpu_init();

    ggml_threadpool_params tpp;
    ggml_threadpool_params_init(&tpp, p.n_threads);
    if (p.cpumask) {
        for (int i = 0; i < 64 && i < GGML_MAX_N_THREADS; i++) {
            tpp.cpumask[i] = (p.cpumask >> i) & 1;
        }
    }
    tpp.strict_cpu = p.strict;
    tpp.poll       = (uint32_t) std::max(0, p.poll);
    tpp.prio       = (ggml_sched_priority) p.prio;

    run_ctx rc(p);
    rc.tp = ggml_threadpool_new(&tpp);
    if (!rc.tp) {
        fprintf(stderr, "failed to create the threadpool\n");
        return 1;
    }

    if (p.pool_threads > 0) {
        ggml_cpu_moe_pool_params pp = ggml_cpu_moe_pool_params_default(p.pool_threads);
        for (int i = 0; i < 64 && i < GGML_MAX_N_THREADS; i++) {
            pp.cpumask[i] = (p.cpumask >> i) & 1;
        }
        pp.prio    = p.prio;
        pp.spin_us = p.pool_spin_us;
        rc.pool = ggml_cpu_moe_pool_new(&pp);
        if (!rc.pool) {
            fprintf(stderr, "failed to create the CPU MoE pool\n");
            return 1;
        }
    }

    printf("moe-cpu-bench: threads %d, cpumask 0x%" PRIx64 ", strict %d, poll %d, prio %d, pool %d, avx512 %d, avx512_vnni %d, avx512_bf16 %d\n",
           p.n_threads, p.cpumask, (int) p.strict, p.poll, p.prio, p.pool_threads, ggml_cpu_has_avx512(), ggml_cpu_has_avx512_vnni(), ggml_cpu_has_avx512_bf16());
    printf("switches: %s\n", switches_str(get_switches()).c_str());

    int n_bad = 0;
    if (!p.gguf_path.empty()) {
        mapped_file mf;
        std::vector<weight_set> sets;
        if (!load_gguf_weights(p.gguf_path.c_str(), mf, sets)) {
            ggml_threadpool_free(rc.tp);
            return 1;
        }
        for (const auto & ws : sets) {
            run_set(rc, ws, n_bad);
        }
    } else {
        std::vector<std::pair<ggml_type, ggml_type>> pairs;
        if (p.all_types) {
            for (ggml_type gu : { GGML_TYPE_Q4_K, GGML_TYPE_Q5_K }) {
                for (ggml_type dn : { GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL }) {
                    pairs.emplace_back(gu, dn);
                }
            }
        } else {
            pairs.emplace_back(p.type_gate, p.type_down);
        }
        for (const auto & pr : pairs) {
            random_pool pool;
            weight_set ws;
            const ggml_type up_t = p.all_types ? pr.first : p.type_up;
            if (!make_random_weights(p, up_t, pr.first, pr.second, pool, ws)) {
                ggml_threadpool_free(rc.tp);
                return 1;
            }
            run_set(rc, ws, n_bad);
        }
    }

    ggml_threadpool_free(rc.tp);
    ggml_cpu_moe_pool_free(rc.pool);

    if (p.check) {
        printf("\ncheck: %d switch configuration%s not bitwise equal to all switches off\n", n_bad, n_bad == 1 ? "" : "s");
    }
    return n_bad ? 2 : 0;
}
