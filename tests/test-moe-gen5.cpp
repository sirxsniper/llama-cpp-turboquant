// [TAG_MOE_DMA_SHARE] [TAG_MOE_PREFETCH] [TAG_FN_PREFILL_STREAM] CPU unit test of the gen5 paths (src/llama-moe-gen5.h).
//
// The "device" is the CPU backend: banks and rings are host memory and every copy is a memcpy, so the test checks the
// plan / fence / gate / release logic, the tables, the ring and the bank bookkeeping, the streamer queue and the graph
// shapes, not the CUDA streams (the GPU test plan covers those). Each graph computes the plain MUL_MAT_ID chain and the
// gen5 chain on the same inputs; the results must be bitwise equal (+0 and -0 count as equal), because every expert is
// computed by exactly one path with the same CPU kernel. No model, no GPU. Exit code 0 iff every check passes.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llama.h"

#include "../src/llama-moe-gen5.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int g_fail   = 0;
static int g_checks = 0;

#define TCHECK(cond, ...)                                        \
    do {                                                         \
        ++g_checks;                                              \
        if (!(cond)) {                                           \
            ++g_fail;                                            \
            fprintf(stderr, "FAIL %s:%d: ", __func__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                        \
            fprintf(stderr, "\n");                               \
        }                                                        \
    } while (0)

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

static const char * k_env[] = {
    "LLAMA_MOE_DMA_SHARE", "LLAMA_MOE_PREFETCH", "LLAMA_MOE_DMA_SLOTS", "LLAMA_MOE_DMA_ADMIT", "LLAMA_MOE_DMA_FILL_MIB",
    "LLAMA_MOE_DMA_RING_MIB", "LLAMA_MOE_PREFETCH_SLOTS", "LLAMA_MOE_PREFETCH_FILL", "LLAMA_MOE_PREFETCH_K",
    "LLAMA_MOE_DMA_FILL_THREADS", "LLAMA_MOE_DMA_SYNC", "LLAMA_MOE_DMA_INLINE", "LLAMA_MOE_DMA_HYST", "LLAMA_MOE_DMA_PROFILE",
    "LLAMA_MOE_HOT_PROFILE", "LLAMA_MOE_DMA_STATS", "LLAMA_PREFILL_STREAM_MIN", "LLAMA_PREFILL_STREAM_BUFS",
    "LLAMA_PREFILL_STREAM_WRAP", "LLAMA_PREFILL_STREAM_CHUNK_MIB", "LLAMA_PREFILL_STREAM_RING_MIB",
    "LLAMA_PREFILL_STREAM_THREADS", "LLAMA_PREFILL_STREAM_STATS",
};

static void reset_env() {
    for (const char * n : k_env) {
        set_env(n, nullptr);
    }
}

//
// a small MoE model: 4 host layers with the three type groups of Qwen3.8-Flash-Next UD-Q4_K_XL, two with a hot set
//

static const int64_t N_EMBD   = 256;
static const int64_t N_FF     = 128;
static const int64_t N_EXPERT = 64;
static const int64_t N_USED   = 6;
static const int     N_LAYER  = 4;
static const int     N_HOT    = 12;

struct t_layer {
    int il = 0;
    ggml_type types[3];
    ggml_tensor * w[3] = { nullptr, nullptr, nullptr }; // up, gate, down: host weights
    bool hot = false;
    std::vector<int32_t> hot_ids;
    ggml_tensor * hw[3] = { nullptr, nullptr, nullptr }; // hot slots [.., N_HOT + 1]
    ggml_tensor * hot_tbl = nullptr;                     // I32 [1, N_EXPERT]: slot or N_HOT
};

struct t_model {
    ggml_backend_t        cpu = nullptr;
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<t_layer>  layers;
};

static void fill_quant(ggml_tensor * t, std::mt19937 & rng) {
    const int64_t n_per_row = t->ne[0];
    const int64_t nrows     = t->ne[1]*t->ne[2];
    std::vector<float> f(n_per_row*nrows);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (auto & v : f) {
        v = u(rng);
    }
    std::vector<uint8_t> q(ggml_nbytes(t));
    ggml_quantize_chunk(t->type, f.data(), q.data(), 0, nrows, n_per_row, nullptr);
    ggml_backend_tensor_set(t, q.data(), 0, q.size());
}

static bool build_model(t_model & m) {
    m.cpu = ggml_backend_cpu_init();
    if (!m.cpu) {
        return false;
    }
    ggml_backend_cpu_set_n_threads(m.cpu, 4);
    const ggml_type groups[N_LAYER][3] = {
        { GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_1 },
        { GGML_TYPE_Q5_K, GGML_TYPE_Q5_K, GGML_TYPE_Q8_0 },
        { GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, GGML_TYPE_Q8_0 },
        { GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_1 },
    };
    ggml_init_params ip = { ggml_tensor_overhead()*64, nullptr, true };
    m.ctx = ggml_init(ip);
    m.layers.resize(N_LAYER);
    for (int l = 0; l < N_LAYER; ++l) {
        t_layer & L = m.layers[l];
        L.il  = l;
        L.hot = (l % 2) == 1;
        for (int k = 0; k < 3; ++k) {
            L.types[k] = groups[l][k];
        }
        L.w[0] = ggml_new_tensor_3d(m.ctx, L.types[0], N_EMBD, N_FF, N_EXPERT);
        L.w[1] = ggml_new_tensor_3d(m.ctx, L.types[1], N_EMBD, N_FF, N_EXPERT);
        L.w[2] = ggml_new_tensor_3d(m.ctx, L.types[2], N_FF, N_EMBD, N_EXPERT);
        if (L.hot) {
            L.hw[0]   = ggml_new_tensor_3d(m.ctx, L.types[0], N_EMBD, N_FF, N_HOT + 1);
            L.hw[1]   = ggml_new_tensor_3d(m.ctx, L.types[1], N_EMBD, N_FF, N_HOT + 1);
            L.hw[2]   = ggml_new_tensor_3d(m.ctx, L.types[2], N_FF, N_EMBD, N_HOT + 1);
            L.hot_tbl = ggml_new_tensor_2d(m.ctx, GGML_TYPE_I32, 1, N_EXPERT);
        }
    }
    m.buf = ggml_backend_alloc_ctx_tensors(m.ctx, m.cpu);
    if (!m.buf) {
        return false;
    }
    ggml_backend_buffer_clear(m.buf, 0);
    std::mt19937 rng(42);
    for (auto & L : m.layers) {
        for (int k = 0; k < 3; ++k) {
            fill_quant(L.w[k], rng);
        }
        if (L.hot) {
            std::vector<int32_t> perm(N_EXPERT);
            for (int i = 0; i < N_EXPERT; ++i) {
                perm[i] = i;
            }
            std::shuffle(perm.begin(), perm.end(), rng);
            L.hot_ids.assign(perm.begin(), perm.begin() + N_HOT);
            std::vector<int32_t> tbl(N_EXPERT, N_HOT);
            for (int s = 0; s < N_HOT; ++s) {
                tbl[L.hot_ids[s]] = s;
                for (int k = 0; k < 3; ++k) {
                    memcpy((char *) L.hw[k]->data + (size_t) s*L.hw[k]->nb[2],
                           (const char *) L.w[k]->data + (size_t) L.hot_ids[s]*L.w[k]->nb[2], L.w[k]->nb[2]);
                }
            }
            memcpy(L.hot_tbl->data, tbl.data(), tbl.size()*sizeof(int32_t));
        }
    }
    return true;
}

static void free_model(t_model & m) {
    if (m.buf) { ggml_backend_buffer_free(m.buf); }
    if (m.ctx) { ggml_free(m.ctx); }
    if (m.cpu) { ggml_backend_free(m.cpu); }
}

static std::vector<llama_moe_gen5_layer_desc> descs(const t_model & m) {
    std::vector<llama_moe_gen5_layer_desc> d;
    for (const auto & L : m.layers) {
        llama_moe_gen5_layer_desc x;
        x.il   = L.il;
        x.up   = L.w[0];
        x.gate = L.w[1];
        x.down = L.w[2];
        d.push_back(x);
    }
    return d;
}

static llama_moe_gen5_device cpu_device(const t_model & m) {
    llama_moe_gen5_device d;
    d.dev     = ggml_backend_get_device(m.cpu);
    d.buft    = ggml_backend_cpu_buffer_type();
    d.compute = m.cpu;
    return d;
}

// up/gate/swiglu/down over w with ids (x: [n_embd, 1, T]); skip: src[3] table with miss value
static ggml_tensor * chain(ggml_context * ctx, ggml_tensor * const w[3], ggml_tensor * x, ggml_tensor * ids,
                           ggml_tensor * skip, int32_t miss) {
    ggml_tensor * up   = ggml_mul_mat_id(ctx, w[0], x, ids);
    ggml_tensor * gate = ggml_mul_mat_id(ctx, w[1], x, ids);
    if (skip) {
        up->src[3]   = skip;
        gate->src[3] = skip;
        up->op_params[0]   = miss;
        gate->op_params[0] = miss;
    }
    ggml_tensor * act  = ggml_swiglu_split(ctx, gate, up);
    ggml_tensor * down = ggml_mul_mat_id(ctx, w[2], act, ids);
    if (skip) {
        down->src[3] = skip;
        down->op_params[0] = miss;
    }
    return down;
}

static ggml_tensor * remap(ggml_context * ctx, ggml_tensor * table, ggml_tensor * ids) {
    const int64_t n_used = ids->ne[0];
    const int64_t T      = ids->ne[1];
    ggml_tensor * flat = ggml_reshape_1d(ctx, ggml_cont(ctx, ids), n_used*T);
    return ggml_reshape_2d(ctx, ggml_get_rows(ctx, table, flat), n_used, T);
}

static bool same(const ggml_tensor * a, const ggml_tensor * b, const char * what, int layer) {
    const int64_t n = ggml_nelements(a);
    const float * x = (const float *) a->data;
    const float * y = (const float *) b->data;
    int64_t bad = 0;
    int64_t first = -1;
    for (int64_t i = 0; i < n; ++i) {
        if (!(x[i] == y[i]) || std::isnan(x[i])) {
            if (first < 0) {
                first = i;
            }
            bad++;
        }
    }
    if (bad) {
        fprintf(stderr, "  %s layer %d: %lld of %lld values differ, first at %lld: %.9g vs %.9g\n", what, layer,
                (long long) bad, (long long) n, (long long) first, x[first], y[first]);
    }
    return bad == 0;
}

struct graph_run {
    ggml_context * ctx = nullptr;
    ggml_cgraph *  gf  = nullptr;
    ggml_gallocr_t ga  = nullptr;
    ~graph_run() {
        if (ga)  { ggml_gallocr_free(ga); }
        if (ctx) { ggml_free(ctx); }
    }
    void init() {
        ggml_init_params ip = { ggml_tensor_overhead()*4096 + ggml_graph_overhead_custom(4096, false), nullptr, true };
        ctx = ggml_init(ip);
        gf  = ggml_new_graph_custom(ctx, 4096, false);
    }
    bool alloc() {
        ga = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
        return ggml_gallocr_alloc_graph(ga, gf);
    }
};

static void set_i32(ggml_tensor * t, const std::vector<int32_t> & v) {
    ggml_backend_tensor_set(t, v.data(), 0, v.size()*sizeof(int32_t));
}

static std::vector<int32_t> route(std::mt19937 & rng, int T, const std::vector<int32_t> & work) {
    std::vector<int32_t> ids(N_USED*T);
    std::uniform_int_distribution<int> ue(0, (int) N_EXPERT - 1);
    std::uniform_int_distribution<int> uw(0, (int) work.size() - 1);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    for (int t = 0; t < T; ++t) {
        std::vector<int32_t> row;
        while ((int64_t) row.size() < N_USED) {
            const int32_t e = u01(rng) < 0.8f ? work[uw(rng)] : ue(rng);
            if (std::find(row.begin(), row.end(), e) == row.end()) {
                row.push_back(e);
            }
        }
        std::copy(row.begin(), row.end(), ids.begin() + t*N_USED);
    }
    return ids;
}

//
// DMA share + prefetch
//

struct dma_cfg {
    const char * name;
    const char * share;
    bool         prefetch;
    const char * slots;
    bool         inline_issue;
    bool         sync;
    bool         profile; // warm start from a routing profile
};

// a moeprof v1 file that ranks every expert of every layer
static std::string write_profile() {
    const char * dir = getenv("TEMP");
    std::string path = std::string(dir ? dir : ".") + "/test-moe-gen5.moeprof";
    FILE * f = fopen(path.c_str(), "w");
    if (!f) {
        return "";
    }
    fprintf(f, "moeprof v1 n_layer=%d n_expert=%lld tokens=1000 source=test\n", N_LAYER, (long long) N_EXPERT);
    for (int l = 0; l < N_LAYER; ++l) {
        fprintf(f, "decode_union %d", l);
        for (int e = 0; e < N_EXPERT; ++e) {
            fprintf(f, " %d", (int) ((e*37 + l*11) % N_EXPERT) + 1);
        }
        fprintf(f, "\n");
    }
    fclose(f);
    return path;
}

static void test_dma(t_model & m, const dma_cfg & c, int n_steps) {
    reset_env();
    if (c.share) {
        set_env("LLAMA_MOE_DMA_SHARE", c.share);
    }
    if (c.prefetch) {
        set_env("LLAMA_MOE_PREFETCH", "1");
    }
    set_env("LLAMA_MOE_DMA_SLOTS", c.slots);
    set_env("LLAMA_MOE_DMA_ADMIT", "1/8");
    if (c.profile) {
        set_env("LLAMA_MOE_DMA_ADMIT", nullptr);
    }
    set_env("LLAMA_MOE_DMA_FILL_MIB", "64");
    set_env("LLAMA_MOE_DMA_RING_MIB", "2");
    set_env("LLAMA_MOE_PREFETCH_SLOTS", "3");
    set_env("LLAMA_MOE_PREFETCH_FILL", "2");
    set_env("LLAMA_MOE_DMA_FILL_THREADS", "2");
    if (c.inline_issue) {
        set_env("LLAMA_MOE_DMA_INLINE", "1");
    }
    if (c.sync) {
        set_env("LLAMA_MOE_DMA_SYNC", "1");
    }
    std::string prof;
    if (c.profile) {
        prof = write_profile();
        TCHECK(!prof.empty(), "[%s] profile written", c.name);
        set_env("LLAMA_MOE_DMA_PROFILE", prof.c_str());
        set_env("LLAMA_MOE_DMA_ADMIT", "8/8"); // admission effectively off: the ring is the profile's
    }
    static int owner_tag = 0;
    const void * owner = &owner_tag;
    const bool ok = llama_moe_dma_init_layers(descs(m), cpu_device(m), owner);
    TCHECK(ok, "[%s] init", c.name);
    if (!ok) {
        return;
    }
    TCHECK(llama_moe_dma_lookup(nullptr, m.layers[0].w[0], LLAMA_MOE_DMA_MAX_T + 1) == nullptr, "[%s] no DMA above %d tokens", c.name, LLAMA_MOE_DMA_MAX_T);
    TCHECK(llama_moe_dma_lookup(nullptr, m.layers[0].w[1], 1) == nullptr, "[%s] lookup is keyed by the up tensor", c.name);
    TCHECK(llama_moe_dma_pred_k() == (c.prefetch ? 16 : 0), "[%s] pred_k", c.name);

    std::mt19937 rng(7);
    std::vector<std::vector<int32_t>> work(N_LAYER);
    for (auto & w : work) {
        for (int i = 0; i < 16; ++i) {
            w.push_back((int32_t) (rng() % N_EXPERT));
        }
    }
    bool all_same = true;
    for (int step = 0; step < n_steps; ++step) {
        const int T = 1 + step % LLAMA_MOE_DMA_MAX_T;
        if (step % 16 == 15) { // the working set drifts
            for (auto & w : work) {
                w[rng() % w.size()] = (int32_t) (rng() % N_EXPERT);
            }
        }
        graph_run g;
        g.init();
        ggml_tensor * x = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, N_EMBD, T);
        ggml_set_input(x);
        ggml_tensor * xin = ggml_reshape_3d(g.ctx, x, N_EMBD, 1, T);
        std::vector<ggml_tensor *> ids(N_LAYER), pred(N_LAYER, nullptr), out_ref(N_LAYER), out_g5(N_LAYER);
        for (int l = 0; l < N_LAYER; ++l) {
            ids[l] = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, N_USED, T);
            ggml_set_input(ids[l]);
            if (c.prefetch && l + 1 < N_LAYER) {
                pred[l] = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, 16, T);
                ggml_set_input(pred[l]);
            }
        }
        for (int l = 0; l < N_LAYER; ++l) {
            t_layer & L = m.layers[l];
            out_ref[l] = chain(g.ctx, L.w, xin, ids[l], nullptr, 0);
            ggml_set_output(out_ref[l]);
            ggml_build_forward_expand(g.gf, out_ref[l]);

            const llama_moe_dma_view * v = llama_moe_dma_lookup(nullptr, L.w[0], T);
            TCHECK(v != nullptr, "[%s] lookup layer %d", c.name, l);
            if (!v) {
                return;
            }
            ggml_tensor * tbl = llama_moe_dma_build_plan(g.ctx, v, ids[l], pred[l], L.hot ? L.hot_tbl : nullptr, N_HOT);
            ggml_tensor * hot_down = nullptr;
            if (L.hot) { // built before the CPU chain, as build_moe_ffn does
                hot_down = chain(g.ctx, L.hw, xin, remap(g.ctx, L.hot_tbl, ids[l]), nullptr, 0);
                ggml_build_forward_expand(g.gf, hot_down);
            }
            ggml_tensor * cpu_down = chain(g.ctx, L.w, xin, ids[l], tbl, 0);
            llama_moe_dma_build_fence(g.ctx, g.gf, v, cpu_down);
            ggml_tensor * merged = cpu_down;
            if (hot_down) {
                merged = ggml_add(g.ctx, merged, hot_down);
            }
            ggml_tensor * bw[3] = { v->up, v->gate, v->down };
            merged = ggml_add(g.ctx, merged, chain(g.ctx, bw, xin, remap(g.ctx, v->dev_table, ids[l]), nullptr, 0));
            out_g5[l] = merged;
            ggml_set_output(out_g5[l]);
            ggml_build_forward_expand(g.gf, out_g5[l]);
        }
        if (!g.alloc()) {
            TCHECK(false, "[%s] graph alloc", c.name);
            break;
        }
        std::vector<float> xv(N_EMBD*T);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        for (auto & v : xv) {
            v = u(rng);
        }
        ggml_backend_tensor_set(x, xv.data(), 0, xv.size()*sizeof(float));
        std::vector<std::vector<int32_t>> route_ids(N_LAYER);
        for (int l = 0; l < N_LAYER; ++l) {
            route_ids[l] = route(rng, T, work[l]);
            set_i32(ids[l], route_ids[l]);
        }
        for (int l = 0; l + 1 < N_LAYER; ++l) {
            if (!pred[l]) {
                continue;
            }
            // the lookahead: the next layer's true ids first (like a router with good recall), then noise
            std::vector<int32_t> p(16*T);
            for (int t = 0; t < T; ++t) {
                for (int r = 0; r < 16; ++r) {
                    p[t*16 + r] = r < N_USED && (r + t + step) % 3 != 0 ? route_ids[l + 1][t*N_USED + r] : (int32_t) (rng() % N_EXPERT);
                }
            }
            set_i32(pred[l], p);
        }
        if (ggml_backend_graph_compute(m.cpu, g.gf) != GGML_STATUS_SUCCESS) {
            TCHECK(false, "[%s] compute", c.name);
            break;
        }
        for (int l = 0; l < N_LAYER; ++l) {
            const bool s = same(out_ref[l], out_g5[l], c.name, l);
            all_same = all_same && s;
            TCHECK(s, "[%s] step %d T %d layer %d: gen5 chain == plain chain", c.name, step, T, l);
        }
        llama_moe_dma_step(owner);
        llama_moe_dma_wait_idle(); // the fills of this step are done before the next one (repeatable)
    }
    const llama_moe_dma_counters k = llama_moe_dma_get_counters();
    printf("  [%s] %llu steps, %llu layer-steps: cold %llu, ring-ready %llu, on-demand %llu, prefetched %llu, prefetch hits %llu, "
           "fills %llu (%llu urgent), pred recall %.2f, share %.2f, bitwise %s\n", c.name,
           (unsigned long long) k.steps, (unsigned long long) k.layer_steps, (unsigned long long) k.cold,
           (unsigned long long) k.ring_ready, (unsigned long long) k.dma_on_demand, (unsigned long long) k.prefetched,
           (unsigned long long) k.dma_staged, (unsigned long long) k.fills, (unsigned long long) k.fill_urgent,
           k.cold ? (double) k.pred_cold/k.cold : 0.0, k.share, all_same ? "equal" : "DIFFERENT");
    TCHECK(k.steps == (uint64_t) n_steps, "[%s] steps counted", c.name);
    TCHECK(k.fills > 0, "[%s] the ring was filled", c.name);
    if (c.share && strcmp(c.share, "0") != 0) {
        TCHECK(k.dma_on_demand > 0, "[%s] experts went over the DMA path", c.name);
    }
    if (c.prefetch) {
        TCHECK(k.prefetched > 0, "[%s] prefetch copied experts", c.name);
        TCHECK(k.dma_staged > 0, "[%s] prefetched experts were used", c.name);
        TCHECK(k.fill_urgent > 0, "[%s] predictions requested ring fills", c.name);
    }
    llama_moe_dma_free(owner);
    TCHECK(llama_moe_dma_lookup(nullptr, m.layers[0].w[0], 1) == nullptr, "[%s] freed", c.name);
    if (!prof.empty()) {
        remove(prof.c_str());
    }
}

//
// prefill stream
//

static void test_pfs(t_model & m, int n_bufs, bool wrap, bool partial) {
    reset_env();
    char nb[8];
    snprintf(nb, sizeof(nb), "%d", n_bufs);
    set_env("LLAMA_PREFILL_STREAM_BUFS", nb);
    set_env("LLAMA_PREFILL_STREAM_WRAP", wrap ? "1" : "0");
    set_env("LLAMA_PREFILL_STREAM_MIN", "16");
    set_env("LLAMA_PREFILL_STREAM_CHUNK_MIB", "1");
    set_env("LLAMA_PREFILL_STREAM_RING_MIB", "2");
    set_env("LLAMA_PREFILL_STREAM_THREADS", "3");
    char name[64];
    snprintf(name, sizeof(name), "pfs bufs %d wrap %d%s", n_bufs, wrap ? 1 : 0, partial ? " partial" : "");
    static int owner_tag = 0;
    const void * owner = &owner_tag;
    const bool ok = llama_prefill_stream_init_layers(descs(m), cpu_device(m), owner);
    TCHECK(ok, "[%s] init", name);
    if (!ok) {
        return;
    }
    TCHECK(llama_prefill_stream_lookup(nullptr, m.layers[0].w[0], 15) == nullptr, "[%s] not below the minimum", name);

    std::mt19937 rng(11 + n_bufs);
    const int n_ub = 4;
    bool all_same = true;
    for (int ub = 0; ub < n_ub; ++ub) {
        const int  T      = 40 + 8*ub;
        const bool cut    = partial && ub == 1; // an ubatch that stops after the gate of layer 1 (abort)
        const int  n_used_layers = cut ? 2 : N_LAYER;
        graph_run g;
        g.init();
        ggml_tensor * x = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, N_EMBD, T);
        ggml_set_input(x);
        ggml_tensor * xin = ggml_reshape_3d(g.ctx, x, N_EMBD, 1, T);
        std::vector<ggml_tensor *> ids(N_LAYER), out_ref(N_LAYER), out_s(N_LAYER, nullptr);
        for (int l = 0; l < n_used_layers; ++l) {
            ids[l] = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, N_USED, T);
            ggml_set_input(ids[l]);
        }
        for (int l = 0; l < n_used_layers; ++l) {
            t_layer & L = m.layers[l];
            out_ref[l] = chain(g.ctx, L.w, xin, ids[l], nullptr, 0);
            ggml_set_output(out_ref[l]);
            ggml_build_forward_expand(g.gf, out_ref[l]);

            const llama_pfs_view * v = llama_prefill_stream_lookup(nullptr, L.w[0], T);
            TCHECK(v != nullptr, "[%s] lookup layer %d", name, l);
            if (!v) {
                return;
            }
            llama_prefill_stream_build_gate(g.ctx, g.gf, v, x);
            ggml_tensor * bw[3] = { v->up, v->gate, v->down };
            ggml_tensor * down = chain(g.ctx, bw, xin, ids[l], nullptr, 0);
            if (!(cut && l == 1)) {
                llama_prefill_stream_build_release(g.ctx, g.gf, v, down);
            }
            out_s[l] = down;
            ggml_set_output(out_s[l]);
            ggml_build_forward_expand(g.gf, out_s[l]);
        }
        if (!g.alloc()) {
            TCHECK(false, "[%s] graph alloc", name);
            break;
        }
        std::vector<float> xv(N_EMBD*T);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        for (auto & v : xv) {
            v = u(rng);
        }
        ggml_backend_tensor_set(x, xv.data(), 0, xv.size()*sizeof(float));
        std::vector<int32_t> all(N_EXPERT);
        for (int i = 0; i < N_EXPERT; ++i) {
            all[i] = i;
        }
        for (int l = 0; l < n_used_layers; ++l) {
            set_i32(ids[l], route(rng, T, all));
        }
        if (ggml_backend_graph_compute(m.cpu, g.gf) != GGML_STATUS_SUCCESS) {
            TCHECK(false, "[%s] compute", name);
            break;
        }
        for (int l = 0; l < n_used_layers; ++l) {
            const bool s = same(out_ref[l], out_s[l], name, l);
            all_same = all_same && s;
            TCHECK(s, "[%s] ubatch %d layer %d: bank chain == host chain", name, ub, l);
        }
    }
    const llama_pfs_counters k = llama_prefill_stream_get_counters();
    printf("  [%s] %llu ubatches, %llu layer copies, %.1f MiB, %llu reused, gates waited %.1f ms, bitwise %s\n", name,
           (unsigned long long) k.ubatches, (unsigned long long) k.jobs, k.bytes/1048576.0, (unsigned long long) k.reused,
           k.wait_s*1e3, all_same ? "equal" : "DIFFERENT");
    TCHECK(k.ubatches == (uint64_t) n_ub, "[%s] ubatches counted", name);
    TCHECK(k.jobs >= (uint64_t) (N_LAYER*(n_ub - (partial ? 1 : 0))), "[%s] every layer was copied", name);
    if (wrap && !partial) {
        TCHECK(k.reused >= (uint64_t) (n_ub - 1), "[%s] the wrap copy of layer 0 was used", name);
    }
    llama_prefill_stream_free(owner);
    TCHECK(llama_prefill_stream_lookup(nullptr, m.layers[0].w[0], 64) == nullptr, "[%s] freed", name);
}

int main() {
    t_model m;
    if (!build_model(m)) {
        fprintf(stderr, "cannot build the test model\n");
        return 1;
    }

    printf("test-moe-gen5: DMA share + prefetch\n");
    const dma_cfg cfgs[] = {
        { "share 1 + prefetch",        "1",     true,  "4", false, false, false },
        { "share 1 + prefetch, sync",  "1",     true,  "4", false, true,  false },
        { "prefetch only (share 0)",   nullptr, true,  "4", false, false, false },
        { "share 0.5",                 "0.5",   false, "4", false, false, false },
        { "share auto, inline issue",  "auto",  true,  "4", true,  false, false },
        { "share 1, one slot",         "1",     true,  "1", false, false, false },
        { "share 1, profile ring",     "1",     false, "4", false, false, true  },
    };
    for (const auto & c : cfgs) {
        test_dma(m, c, 48);
    }

    printf("test-moe-gen5: prefill stream\n");
    for (int nb = 1; nb <= 3; ++nb) {
        for (int wrap = 0; wrap <= 1; ++wrap) {
            test_pfs(m, nb, wrap != 0, false);
        }
        test_pfs(m, nb, true, true);
    }

    reset_env();
    free_model(m);
    printf("test-moe-gen5: %d checks, %d errors%s\n", g_checks, g_fail, g_fail ? "" : " - PASSED");
    return g_fail == 0 ? 0 : 1;
}
