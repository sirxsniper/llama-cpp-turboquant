#pragma once

// [TAG_R41_PP_SYSKEEP] [TAG_R41_PP_CONVKEEP] the kept files of the conversation store (server-keepctx.cpp): the states of
// system blocks (LLAMA_FN_R41_SYSKEEP=2) and of conversations (LLAMA_FN_R41_CONVKEEP=1) that stay on disk after the server
// ends. This header holds what does not need a model: the first sector of a kept file and its checks, the sum of the
// stored bytes, the fingerprint that ties a file to the weights, the chat template, the settings and the build, and the
// folder rules (an unfinished, damaged or foreign file is deleted, the oldest files go beyond a count and a size).
// Header only, no llama dependency: tests/test-fn-keptfile.cpp runs it on files in a temporary folder.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

static constexpr size_t   KEPT_SECTOR  = 4096;
static constexpr uint32_t KEPT_CKPT    = 32;  // checkpoints a kept file can describe
static constexpr uint32_t KEPT_VERSION = 2;   // 2: body_sum

// the first sector of a kept file. It is written last, so a file whose writing stopped has no valid header. Then: the
// tokens (int32) at tok_off, the drafter's carry at carry_off, the stored bytes of the entry at base; each part at a
// sector offset.
struct kept_header {
    char     magic[8];      // "R41KEPT1"
    uint32_t version;       // KEPT_VERSION
    uint32_t n_ckpt;        // checkpoints (at most KEPT_CKPT)
    uint32_t kind;          // 1 a system block, 2 a conversation
    uint32_t conv_rerender;
    int64_t  conv_hits;
    int32_t  conv_sys_end;
    int32_t  conv_n_prompt;
    uint64_t fp;            // the fingerprint of the server that wrote it (kept_fingerprint)
    uint64_t sys_hash;      // the hash of the tokens
    int64_t  n_tokens;
    uint64_t size_tgt;
    uint64_t size_dft;
    uint64_t n_bytes;       // the stored bytes from base
    uint64_t base;
    uint64_t tok_off;
    uint64_t carry_off;
    uint64_t carry_bytes;
    struct {
        int64_t  n_tokens;
        int32_t  id_task;
        int32_t  pos_min;
        int32_t  pos_max;
        int32_t  pad;
        uint64_t size_tgt;
        uint64_t size_dft;
        uint64_t size_spec;
    } ck[KEPT_CKPT];
    uint64_t body_sum;      // kept_sum of the n_bytes stored bytes
    uint64_t check;         // kept_fnv of every byte above
};
static_assert(sizeof(kept_header) <= KEPT_SECTOR, "the header is one sector");

static inline size_t kept_round_up(size_t n) {
    return (n + KEPT_SECTOR - 1) / KEPT_SECTOR * KEPT_SECTOR;
}

// FNV-1a
static inline uint64_t kept_fnv(uint64_t h, const void * data, size_t n) {
    const uint8_t * p = (const uint8_t *) data;
    for (size_t i = 0; i < n; ++i) {
        h = (h ^ p[i]) * 0x100000001b3ull;
    }
    return h;
}

static constexpr uint64_t KEPT_FNV_SEED = 0xcbf29ce484222325ull;

static inline uint64_t kept_header_check(const kept_header & h) {
    return kept_fnv(KEPT_FNV_SEED, &h, offsetof(kept_header, check));
}

// the magic, the version and the check of a filled header
static inline void kept_header_seal(kept_header & h) {
    std::memcpy(h.magic, "R41KEPT1", 8);
    h.version = KEPT_VERSION;
    h.check   = kept_header_check(h);
}

// the sum of the stored bytes: four lanes of 64-bit words, fed in pieces of any size (the result does not depend on how
// the bytes are cut into pieces). It tells a damaged file from a whole one; it is not a cryptographic hash.
struct kept_sum {
    uint64_t v[4]  = { 0x60ea27eeadc0b5d6ull, 0xc2b2ae3d27d4eb4full, 0, 0x61c8864e7a143579ull };
    uint8_t  buf[32] = {};
    size_t   n_buf = 0;
    uint64_t total = 0;

    static uint64_t rotl(uint64_t x, int r) {
        return (x << r) | (x >> (64 - r));
    }
    static uint64_t lane(uint64_t acc, uint64_t w) {
        acc += w * 0xc2b2ae3d27d4eb4full;
        acc  = rotl(acc, 31);
        return acc * 0x9e3779b185ebca87ull;
    }

    void add(const void * data, size_t n) {
        const uint8_t * p = (const uint8_t *) data;
        total += n;
        if (n_buf > 0) {
            const size_t k = std::min(n, sizeof(buf) - n_buf);
            std::memcpy(buf + n_buf, p, k);
            n_buf += k; p += k; n -= k;
            if (n_buf < sizeof(buf)) {
                return;
            }
            uint64_t w[4];
            std::memcpy(w, buf, sizeof(w));
            for (int i = 0; i < 4; ++i) {
                v[i] = lane(v[i], w[i]);
            }
            n_buf = 0;
        }
        uint64_t a = v[0], b = v[1], c = v[2], d = v[3];
        while (n >= 32) {
            uint64_t w[4];
            std::memcpy(w, p, sizeof(w));
            a = lane(a, w[0]);
            b = lane(b, w[1]);
            c = lane(c, w[2]);
            d = lane(d, w[3]);
            p += 32; n -= 32;
        }
        v[0] = a; v[1] = b; v[2] = c; v[3] = d;
        if (n > 0) {
            std::memcpy(buf, p, n);
            n_buf = n;
        }
    }

    uint64_t done() const {
        uint64_t h = rotl(v[0], 1) + rotl(v[1], 7) + rotl(v[2], 12) + rotl(v[3], 18);
        h ^= total * 0x9e3779b185ebca87ull;
        for (size_t i = 0; i < n_buf; ++i) {
            h = rotl((h ^ buf[i]) * 0x100000001b3ull, 11);
        }
        h ^= h >> 33;
        h *= 0xc2b2ae3d27d4eb4full;
        h ^= h >> 29;
        h *= 0x165667b19e3779f9ull;
        h ^= h >> 32;
        return h;
    }
};

// what a header says about its file of file_size bytes: false = unfinished, damaged, another version or not a kept file
// of this kind (the stored bytes are checked apart, with body_sum)
static inline bool kept_header_ok(const kept_header & h, uint32_t kind, uint64_t file_size) {
    if (std::memcmp(h.magic, "R41KEPT1", 8) != 0 || h.version != KEPT_VERSION || h.check != kept_header_check(h) || h.kind != kind) {
        return false;
    }
    if (h.n_tokens <= 0 || h.n_tokens > ((int64_t) 1 << 28) || h.n_ckpt > KEPT_CKPT || h.size_tgt == 0 ||
        h.carry_bytes > file_size || h.n_bytes > file_size) {
        return false;
    }
    const uint64_t tok_bytes = (uint64_t) h.n_tokens * sizeof(int32_t);
    if (h.tok_off != KEPT_SECTOR || h.carry_off != h.tok_off + kept_round_up((size_t) tok_bytes) ||
        h.base != h.carry_off + kept_round_up((size_t) h.carry_bytes) || file_size < h.base + kept_round_up((size_t) h.n_bytes)) {
        return false;
    }
    uint64_t ck_bytes = 0;
    for (uint32_t k = 0; k < h.n_ckpt; ++k) {
        ck_bytes += h.ck[k].size_tgt + h.ck[k].size_dft + h.ck[k].size_spec;
    }
    return h.size_tgt + h.size_dft + ck_bytes == h.n_bytes;
}

// what a kept file depends on: a server reads back only the files written with the same weights, chat template, settings,
// build and engine environment
struct kept_identity {
    std::vector<std::string> weights;  // one line per weights file (name, size, write time) and the model's own description
    std::string              tmpl;     // the chat template text in use
    std::string              settings; // context, slots, batch, cache types, draft settings
    std::string              build;
    std::vector<std::string> env;      // the engine variables, sorted
};

static inline uint64_t kept_fingerprint(const kept_identity & id) {
    uint64_t h = KEPT_FNV_SEED;
    auto mix = [&](const std::string & s) {
        h = kept_fnv(h, s.data(), s.size());
        h = (h ^ 0xffu) * 0x100000001b3ull;
    };
    mix("weights " + std::to_string(id.weights.size()));
    for (const auto & s : id.weights) {
        mix(s);
    }
    mix("template " + std::to_string(id.tmpl.size()));
    mix(id.tmpl);
    mix(id.settings);
    mix(id.build);
    mix("env " + std::to_string(id.env.size()));
    for (const auto & s : id.env) {
        mix(s);
    }
    return h;
}

// one kept file of a folder
struct kept_file {
    std::filesystem::path           path;
    std::filesystem::file_time_type t;
    uint64_t                        size = 0;
    bool                            busy = false; // it could not be opened (another server writes it): left alone
    kept_header                     h;            // its header (kept_scan_dir: a good one, unless busy)
};

// the files <prefix>*.kcs of a folder, the oldest first
static inline std::vector<kept_file> kept_list(const std::string & dir, const char * prefix) {
    std::vector<kept_file> files;
    std::error_code ec;
    for (const auto & de : std::filesystem::directory_iterator(dir, ec)) {
        const std::string name = de.path().filename().string();
        if (name.rfind(prefix, 0) == 0 && de.path().extension() == ".kcs") {
            std::error_code e1, e2;
            const auto t  = std::filesystem::last_write_time(de.path(), e1);
            const auto sz = std::filesystem::file_size(de.path(), e2);
            if (!e1 && !e2) {
                kept_file f;
                f.path = de.path();
                f.t    = t;
                f.size = (uint64_t) sz;
                std::memset(&f.h, 0, sizeof(f.h));
                files.push_back(std::move(f));
            }
        }
    }
    std::sort(files.begin(), files.end(), [](const kept_file & a, const kept_file & b) {
        return a.t != b.t ? a.t < b.t : a.path.filename().string() < b.path.filename().string();
    });
    return files;
}

// the header of a file; opened: false when the file cannot be opened at all
static inline bool kept_read_header(const std::filesystem::path & p, kept_header & h, bool & opened) {
    std::memset(&h, 0, sizeof(h));
    std::ifstream in(p, std::ios::binary);
    opened = in.is_open();
    if (!opened) {
        return false;
    }
    in.read((char *) &h, sizeof(h));
    return (size_t) in.gcount() == sizeof(h);
}

// the files to delete (indexes into files, the oldest first) so that one more file of need bytes fits within max_files
// files and max_bytes bytes; n_new 0: only what is there has to fit. A busy file is never chosen.
static inline std::vector<size_t> kept_plan_room(const std::vector<kept_file> & files, int32_t max_files, uint64_t max_bytes,
                                                 int32_t n_new, uint64_t need) {
    std::vector<size_t> out;
    uint64_t total = need;
    int64_t  count = n_new;
    for (const auto & f : files) {
        total += f.size;
        count++;
    }
    for (size_t i = 0; i < files.size() && (count > (int64_t) max_files || total > max_bytes); ++i) {
        if (files[i].busy) {
            continue;
        }
        total -= files[i].size;
        count--;
        out.push_back(i);
    }
    return out;
}

struct kept_scan_result {
    std::vector<kept_file> files; // what is left in the folder, the oldest first: a good header each (or busy)
    int n_bad  = 0;               // unfinished, damaged or foreign: deleted
    int n_old  = 0;               // the oldest, deleted beyond the count or the size
    int n_busy = 0;               // could not be opened: left alone
};

// the kept files <prefix>*.kcs of one kind in a folder: a file without a good header (unfinished, damaged, another
// version, not one of ours) is deleted, then the oldest files are deleted until at most max_files files and max_bytes
// bytes are left. The stored bytes are not read here (body_sum is the caller's check when it reads a file back).
static inline kept_scan_result kept_scan_dir(const std::string & dir, const char * prefix, uint32_t kind, int32_t max_files,
                                             uint64_t max_bytes) {
    kept_scan_result r;
    for (auto & f : kept_list(dir, prefix)) {
        bool opened = false;
        const bool got = kept_read_header(f.path, f.h, opened);
        if (!opened) {
            f.busy = true;
            r.n_busy++;
            r.files.push_back(std::move(f));
            continue;
        }
        if (!got || !kept_header_ok(f.h, kind, f.size)) {
            std::error_code ec_rm;
            std::filesystem::remove(f.path, ec_rm);
            r.n_bad++;
            continue;
        }
        r.files.push_back(std::move(f));
    }
    const std::vector<size_t> out = kept_plan_room(r.files, max_files, max_bytes, 0, 0);
    for (auto it = out.rbegin(); it != out.rend(); ++it) {
        std::error_code ec_rm;
        std::filesystem::remove(r.files[*it].path, ec_rm);
        r.files.erase(r.files.begin() + (std::ptrdiff_t) *it);
        r.n_old++;
    }
    return r;
}

// the sum of the stored bytes of a kept file, read with the file cache (the server reads them through its own pipe)
static inline bool kept_body_sum_file(const std::filesystem::path & p, const kept_header & h, uint64_t & sum) {
    std::ifstream in(p, std::ios::binary);
    if (!in.is_open()) {
        return false;
    }
    in.seekg((std::streamoff) h.base);
    std::vector<char> buf(1u << 20);
    kept_sum s;
    uint64_t left = h.n_bytes;
    while (left > 0) {
        const size_t k = (size_t) std::min<uint64_t>(buf.size(), left);
        in.read(buf.data(), (std::streamsize) k);
        if ((size_t) in.gcount() != k) {
            return false;
        }
        s.add(buf.data(), k);
        left -= k;
    }
    sum = s.done();
    return true;
}
