// Host simulation of a state restore whose lend map is refused. The lend code of src/llama-kv-cache.cpp (the namespace
// block, the API block with kvl_update) and llama_kv_cache::state_clear(seq_id, strm, sinfo) are included VERBATIM
// (extract.py); only ggml / CUDA calls are stubbed. A fake VMM buffer tracks every 16-byte unit: unmapped, mapped with
// unknown bytes (a fresh physical page), or mapped and zeroed. Maps and unmaps go in granules of 2 MiB (an assumption of
// this simulation, not a value from a log). One stubbed llama_clear_tensor_data call covers one contiguous cell range of
// one K or V tensor (the real one splits it into 1 MiB tensor_set calls); a call that reaches an unmapped unit is what
// aborts on CUDA.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <set>
#include <utility>
#include <vector>

#define GGML_ASSERT(x) do { if (!(x)) { printf("assert %s\n", #x); exit(2); } } while (0)

typedef int ggml_type;
struct ggml_tensor { int64_t ne[4]; size_t nb[4]; ggml_type type; size_t nbytes; };
static size_t ggml_nbytes(const ggml_tensor * t) { return t->nbytes; }
static size_t ggml_row_size(ggml_type, int64_t n) { return (size_t) n; } // one byte per element
static size_t ggml_type_size(ggml_type) { return 1; }

enum gstate { G_UNMAPPED = 0, G_DIRTY = 1, G_ZERO = 2 };
constexpr size_t UNIT = 16; // rows (400, 272 B) and granules are multiples of it
struct fake_buf {
    size_t gran = 0;
    std::vector<uint8_t> u;  // gstate of every 16-byte unit
    long long budget = -1;   // bytes it may still map (-1: no limit)
};
typedef fake_buf * ggml_backend_buffer_t;

static std::map<const ggml_tensor *, std::pair<fake_buf *, size_t>> g_where; // tensor -> buffer, offset
static size_t g_bad = 0;      // stubbed write calls that reach an unmapped unit
static size_t g_cleared = 0;  // bytes state_clear wrote

static void touch(const ggml_tensor * t, size_t off, size_t size, bool count_clear) {
    auto w = g_where.at(t);
    fake_buf * b = w.first;
    const size_t a = w.second + off, e = a + size;
    bool bad = false;
    for (size_t o = a / UNIT; o < (e + UNIT - 1) / UNIT; ++o) {
        bad = bad || b->u[o] == G_UNMAPPED;
        if (b->u[o] != G_UNMAPPED) {
            b->u[o] = G_ZERO;
        }
    }
    g_bad += bad ? 1 : 0;
    if (count_clear) {
        g_cleared += size;
    }
}
static void ggml_backend_tensor_memset(ggml_tensor * t, int, size_t off, size_t size) { touch(t, off, size, false); }
static void llama_clear_tensor_data(ggml_tensor * t, size_t off, size_t size) { touch(t, off, size, true); }
static void ggml_backend_buffer_init_tensor(ggml_backend_buffer_t, ggml_tensor *) {}
static int64_t ggml_time_us() { return 0; }
#define LLAMA_LOG_INFO(...)  do {} while (0)
#define LLAMA_LOG_WARN(...)  do {} while (0)
#define LLAMA_LOG_ERROR(...) do { if (getenv("SIM_V")) printf(__VA_ARGS__); } while (0)
// [v0.32.0] the zero-commit glue: off as on Linux; SIM_ZC=1: inside a growth's scope a chunk takes a parked handle of three
// chunks (at most the region's room), as R40-VGUARD-3 C3 does; outside a scope (the initial map) nothing changes
static bool g_zc_scope = false;
static int llama_fn_zc_open() { const char * e = getenv("SIM_ZC"); g_zc_scope = e && e[0] == '1'; return g_zc_scope ? 1 : 0; }
static void llama_fn_zc_close(int id) { if (id > 0) g_zc_scope = false; }
static size_t llama_fn_zc_fit(size_t need, size_t room) { return g_zc_scope ? std::min(room, 3 * need) : 0; }
static bool llama_fn_pin_active() { return false; }
static bool llama_fn_dxgi_query(int, int, uint64_t *, uint64_t *) { return false; }
static uint64_t llama_fn_pin_precheck_margin() { return 0; }
static bool llama_fn_pin_on_oom(size_t) { return false; }
enum { LLAMA_FN_PIN_NEED_KV_GROW = 0 };
struct llama_model {};
static void llama_fn_pin_note_need(const llama_model *, int, size_t) {}

typedef int32_t llama_seq_id;
typedef std::vector<uint32_t> idx_vec_t;

struct cells_t {
    uint32_t umax = 0;
    uint32_t n    = 0;
    uint32_t used_max_p1() const { return umax; }
    uint32_t size() const { return n; }
};

struct hparams_t { uint32_t n_embd_v_gqa(uint32_t) const { return 400; } };

struct llama_kv_cache {
    struct slot_info {
        uint32_t s0 = 0, s1 = 0;
        std::vector<llama_seq_id> strm;
        std::vector<idx_vec_t>    idxs;
        uint32_t head() const { return idxs[0][0]; }
        void resize(size_t n) { strm.resize(n); idxs.resize(n); }
        size_t size() const { return idxs[0].size(); }
        bool empty() const { return idxs.empty(); }
        bool is_contiguous() const {
            for (size_t i = 1; i < idxs[0].size(); ++i) {
                if (idxs[0][i] != idxs[0][i - 1] + 1) {
                    return false;
                }
            }
            return true;
        }
    };
    struct layer_t {
        uint32_t il = 0;
        std::vector<ggml_tensor *> k_stream, v_stream;
    };
    llama_model model;
    hparams_t hparams;
    bool fn_kvgrow = true;
    bool v_trans = false;
    uint32_t n_stream = 1;
    std::vector<cells_t> v_cells = std::vector<cells_t>(1);
    std::vector<layer_t> layers;
    bool     kvl_update(uint32_t need) const;
    uint32_t kvl_need() const;
    bool     kvl_owns(ggml_backend_buffer_t buf) const;
    void     kvl_clear() const;
    uint32_t kvl_cells() const; // present after the fix only
    void     seq_rm(llama_seq_id, int, int) {}
    void     clear(bool) {}
    void     state_clear(llama_seq_id seq_id, uint32_t strm, const slot_info & sinfo);
};

namespace {
#include "ns.inc"
}
#include "api.inc"
#include "state_clear.inc"

static bool fmap(ggml_backend_buffer_t b, size_t off, size_t size) {
    if (b->budget >= 0 && (long long) size > b->budget) {
        return false;
    }
    std::fill(b->u.begin() + off / UNIT, b->u.begin() + (off + size) / UNIT, (uint8_t) G_DIRTY); // a new page: old bytes
    if (b->budget >= 0) {
        b->budget -= size;
    }
    return true;
}
static bool funmap(ggml_backend_buffer_t b, size_t off, size_t size) {
    std::fill(b->u.begin() + off / UNIT, b->u.begin() + (off + size) / UNIT, (uint8_t) G_UNMAPPED);
    if (b->budget >= 0) {
        b->budget += size;
    }
    return true;
}

// a lend like kvl_alloc's for `nl` layers (K and V of `row` bytes per cell, or K only), in g_kvls[slot]
static fake_buf * make_lend(int slot, llama_kv_cache & kv, int nl, bool has_v, size_t row, uint32_t kv_size, uint32_t init,
                            uint32_t chunk, size_t gran, std::vector<ggml_tensor> & ts) {
    kvl_cur = &g_kvls[slot];
    auto * b = new fake_buf();
    b->gran = gran;
    const int nreg = nl * (has_v ? 2 : 1);
    ts.assign(nreg, ggml_tensor{});
    std::vector<kvl_region> regs;
    size_t off = 0;
    for (int i = 0; i < nreg; ++i) {
        off = kvl_round_up(off, gran) + gran;
        ts[i].ne[0] = (int64_t) row; ts[i].ne[1] = kv_size; ts[i].nb[1] = row; ts[i].nbytes = (size_t) kv_size * row;
        kvl_region r; r.t = &ts[i]; r.start = off; r.row = row; r.alloc = ts[i].nbytes;
        regs.push_back(r);
        g_where[&ts[i]] = { b, off };
        off += ts[i].nbytes;
    }
    const size_t total = kvl_round_up(off, gran) + gran;
    b->u.assign(total / UNIT, (uint8_t) G_UNMAPPED);
    for (size_t i = 0; i < regs.size(); ++i) {
        regs[i].ext = (i + 1 < regs.size() ? regs[i + 1].start : total) - regs[i].start;
    }
    kv.layers.clear();
    for (int l = 0; l < nl; ++l) {
        llama_kv_cache::layer_t L;
        L.il = l;
        L.k_stream.push_back(&ts[has_v ? 2*l : l]);
        L.v_stream.push_back(has_v ? &ts[2*l + 1] : nullptr);
        kv.layers.push_back(L);
    }
    kv.v_cells[0].n = kv_size;
    g_kvl = kvl_state();
    g_kvl.buf = b; g_kvl.gran = gran; g_kvl.map = fmap; g_kvl.unmap = funmap; g_kvl.regs = regs;
    g_kvl.kv_size = kv_size; g_kvl.init_cells = init; g_kvl.chunk = chunk;
    if (!kvl_map_to(init)) { printf("init map failed\n"); exit(1); }
    g_kvl.kv = &kv;
    g_kvl.init_bytes = kvl_mapped_total();
    for (const auto & r : g_kvl.regs) g_kvl.full_bytes += kvl_bytes_for(r, kv_size);
    return b;
}

static size_t g_out = 0; // bytes the fake hot set has out
static size_t rel(const void *, size_t extra) { (void) extra; return g_out; } // the hot set has nothing to give
static size_t res(const void *, size_t extra) { g_out = extra; return g_out; }

// rows of the cells `c` (below `lim`) of every region of lend `s` that are mapped and hold old bytes
static size_t dirty_in(const kvl_state & s, const idx_vec_t & c, uint32_t lim) {
    size_t n = 0;
    for (const auto & r : s.regs) {
        for (const uint32_t i : c) {
            if (i >= lim) {
                continue;
            }
            bool d = false;
            for (size_t o = (r.start + (size_t) i * r.row) / UNIT; o < (r.start + (size_t) (i + 1) * r.row) / UNIT; ++o) {
                d = d || s.buf->u[o] == G_DIRTY;
            }
            n += d ? 1 : 0;
        }
    }
    return n;
}

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { ++g_fail; printf("  CHECK FAILED: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

// which = 0: the attention lend's map is refused; 1: the attention grows, the indexer's map is refused
static void run(const char * tag, int which, uint32_t kv_size, uint32_t init, uint32_t head, uint32_t count, bool contiguous,
                long long budget) {
    for (auto & s : g_kvls) s = kvl_state();
    g_where.clear();
    llama_kv_cache attn, idx;
    static std::vector<ggml_tensor> ta, ti;
    fake_buf * ba = make_lend(0, attn, 12, true, 400, kv_size, init, 16384, 2u << 20, ta);
    fake_buf * bi = make_lend(1, idx, 12, false, 272, kv_size, init, 16384, 2u << 20, ti);
    llama_kv_lend_set_hooks(rel, res, nullptr);
    g_out = 0;
    llama_kv_cache & kv = which == 0 ? attn : idx;
    fake_buf * bf = which == 0 ? ba : bi;
    kvl_state & ks = g_kvls[which];
    const uint32_t cells0 = ks.cells;
    // the mapped cells held another sequence's data (free now, not cleared: seq_rm does not clear). A sequence only uses
    // cells below `cells`; the restore below writes no data, as its kvl_update fails before state_read_data
    for (auto & r : ks.regs) {
        std::fill(bf->u.begin() + r.start / UNIT, bf->u.begin() + (r.start + (size_t) cells0 * r.row) / UNIT, (uint8_t) G_DIRTY);
    }
    // state_read_meta placed the cells
    llama_kv_cache::slot_info si;
    si.resize(1);
    for (uint32_t i = 0; i < count; ++i) si.idxs[0].push_back(contiguous ? head + i : head + 2*i);
    const uint32_t last = si.idxs[0].back();
    kv.v_cells[0].umax = last + 1;
    bf->budget = budget;
    g_bad = 0; g_cleared = 0;
    const bool ok = kv.kvl_update(kv.kvl_need());
    if (!ok) {
        kv.state_clear(0, 0, si);
    }
    const uint32_t cells1 = ks.cells;
    const size_t bad = g_bad;
    // the placed cells that have memory: zeroed by state_clear (or by the map that gave them memory)
    const size_t dirty_low = dirty_in(ks, si.idxs[0], cells1);
    // a later sequence that needs those cells again: they are mapped (VRAM back) and must hold zeros, not old bytes
    bf->budget = -1;
    kv.v_cells[0].umax = last + 1;
    g_bad = 0;
    const bool ok2 = kv.kvl_update(kv.kvl_need());
    const size_t dirty_all = dirty_in(ks, si.idxs[0], UINT32_MAX);
    printf("%-40s kv %u, cells %u -> %u, restore of %u cells in [%u,%u]: kvl_update %s; state_clear wrote %zu MiB, "
           "%zu clear range(s) reaching unmapped memory; placed rows with memory holding old bytes: %zu; after a later grow (%s): %zu\n",
           tag, kv_size, cells0, cells1, count, head, last, ok ? "ok" : "FAILED", g_cleared >> 20, bad, dirty_low,
           ok2 ? "ok" : "failed", dirty_all);
    CHECK(bad == 0, "%s: %zu clear range(s) reaching unmapped memory", tag, bad);
    if (!ok) {
        CHECK(dirty_low == 0, "%s: placed cells with memory left with old bytes", tag);
        CHECK(dirty_all == 0, "%s: placed cells hold old bytes after the later grow", tag);
    }
}

int main() {
    const long long MiB = 1 << 20;
    run("control: grow succeeds",                    0, 262144, 20971,      0, 100000, true, -1);
    run("control: inside the mapped prefix",         0, 262144, 20971,      0,  10000, true,  0);
    run("attention map refused at once",             0, 262144, 20971,      0, 100000, true,  0);
    run("attention map refused midway",              0, 262144, 20971,      0, 100000, true, 200 * MiB);
    run("512K, refused midway",                      0, 524288, 20971,  30000, 200000, true, 600 * MiB);
    run("512K, scattered cells, refused midway",     0, 524288, 20971,  10000,  60000, false, 300 * MiB);
    run("indexer map refused (attention grew)",      1, 262144, 23130,      0, 100000, true,  0);
    printf("sim checks failed: %d\n", g_fail);
    return g_fail != 0;
}
