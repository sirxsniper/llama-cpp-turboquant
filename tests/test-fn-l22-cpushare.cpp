// [TAG_FN_L22_CPUSHARE] CPU unit test of round 22 package P16 (src/llama-fn-l22-p16.h): the plan helpers, the share op's
// math, and share graphs on the prefill stream with the CPU backend as the device (test-moe-gen5's pattern: banks are host
// memory, copies are memcpy). The device MUL_MAT_IDs here are the CPU ones, so a wrong skip (a bank without an expert the
// chain reads) shows as wrong values. No model, no GPU. Exit code 0 iff every check passes.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llama.h"

#include "../src/llama-moe-gen5.h"
#include "../src/llama-fn-l22.h"
#include "../src/llama-fn-l22-p16.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
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

//
// 1. pick
//

static std::vector<int32_t> make_ids(int T, int k, const std::vector<std::vector<int>> & rows_of) {
    // rows_of[t] = the experts of token t (k of them)
    std::vector<int32_t> ids((size_t) T*k);
    for (int t = 0; t < T; ++t) {
        for (int s = 0; s < k; ++s) {
            ids[(size_t) t*k + s] = rows_of[t][s];
        }
    }
    return ids;
}

static void test_pick() {
    // 16 experts, k 2, 12 tokens. Rows: e0 6, e1 1, e2 2, e3 3, e4 0, e5 4, e6 1, e7 1, e8 0, e9 6 (sum 24 = 12 x 2)
    const int ne = 16, k = 2, T = 12;
    std::vector<std::vector<int>> r = {
        { 0, 1 }, { 0, 2 }, { 0, 2 }, { 0, 3 }, { 0, 3 }, { 0, 3 },
        { 9, 5 }, { 9, 5 }, { 9, 5 }, { 9, 5 }, { 9, 6 }, { 9, 7 },
    };
    const std::vector<int32_t> ids = make_ids(T, k, r);
    std::vector<uint8_t>  late(ne, 1);
    late[0] = 0; // e0 early
    std::vector<uint64_t> bytes(ne, 1000);
    bytes[6] = 2000; // e6 before e1 / e7 at equal rows (bytes desc)
    llama_fn_cps_pick_in in;
    in.n_expert = ne;
    in.n_used   = k;
    in.n_tokens = T;
    in.ids      = ids.data();
    in.late     = late.data();
    in.bytes    = bytes.data();
    in.rows_max = 3;
    in.k_fixed  = 100;
    in.cap      = 64;
    llama_fn_cps_pick_out o;
    llama_fn_cps_pick(in, o);
    TCHECK(o.rows[0] == 6 && o.rows[1] == 1 && o.rows[3] == 3 && o.rows[4] == 0 && o.rows[9] == 6, "rows counted");
    // CPU: late, rows 1..3: e1, e2, e3, e6, e7
    const std::vector<int32_t> want_cpu = { 1, 2, 3, 6, 7 };
    TCHECK(o.cpu == want_cpu, "CPU set (got %zu experts)", o.cpu.size());
    TCHECK(o.skip[4] && o.skip[8] && o.skip[10] && o.skip[15], "unrouted late experts skipped");
    TCHECK(!o.skip[0] && !o.skip[5] && !o.skip[9], "early / heavy experts copied");
    TCHECK(o.skip[1] && o.skip[7] && o.table[1] == ne && o.table[7] == ne && o.table[5] == 5 && o.table[0] == 0, "table / skip of the CPU set");
    // tokens of e1 (0), e2 (1, 2), e3 (3, 4, 5), e6 (10), e7 (11)
    const std::vector<int32_t> want_tok = { 0, 1, 2, 3, 4, 5, 10, 11 };
    TCHECK(o.tok == want_tok, "CPU tokens ascending");
    TCHECK(o.n_pairs == 8, "pairs %d", o.n_pairs);
    TCHECK(o.skip_cpu == 5*1000 + 1000 && o.skip_unr == 1000ull*(16 - 8), "skip bytes cpu %llu unr %llu",
           (unsigned long long) o.skip_cpu, (unsigned long long) o.skip_unr);

    // order: rows asc, bytes desc, id asc -> e6, e1, e7, e2, e3
    in.k_fixed = 2;
    llama_fn_cps_pick(in, o);
    TCHECK(o.cpu.size() == 2 && o.cpu[0] == 1 && o.cpu[1] == 6, "k 2 takes e6 and e1");
    in.k_fixed = 0;
    llama_fn_cps_pick(in, o);
    TCHECK(o.cpu.empty() && o.skip[4] && !o.skip[1], "k 0: skip only");

    // cap: 3 distinct tokens
    in.k_fixed = 100;
    in.cap     = 3;
    llama_fn_cps_pick(in, o);
    TCHECK(o.tok.size() <= 3 && o.cut_cap > 0, "cap: %zu tokens, %d cut", o.tok.size(), o.cut_cap);
    in.cap = 64;

    // candidates and accept
    std::vector<uint8_t> cand(ne, 1);
    cand[2] = 0;
    in.cand   = cand.data();
    in.accept = [](int e) { return e != 3; };
    llama_fn_cps_pick(in, o);
    TCHECK(o.cpu == std::vector<int32_t>({ 1, 6, 7 }) && o.rejected == 1 && !o.skip[2] && !o.skip[3], "cand / accept");
    in.cand   = nullptr;
    in.accept = nullptr;

    // no late phase: nothing skipped, no CPU
    in.late = nullptr;
    llama_fn_cps_pick(in, o);
    TCHECK(o.cpu.empty() && std::count(o.skip.begin(), o.skip.end(), 1) == 0 && o.rows[0] == 6, "no late phase");
    in.late = late.data();

    // budget: copy of the late routed bytes vs CPU time
    in.k_fixed = -1;
    in.c_e = 1000.0;
    in.c_r = 10.0;
    in.dma_bpu = 1000.0; // late routed bytes 1000 x (e1 e2 e3 e5 e6 e7 e9 ...) ~ 8000 -> 8 us of copy
    llama_fn_cps_pick(in, o);
    TCHECK(o.cpu.empty(), "budget: a slow CPU takes nothing");
    in.c_e = 0.1;
    in.c_r = 0.01;
    in.dma_bpu = 1.0;
    llama_fn_cps_pick(in, o);
    TCHECK(o.cpu.size() == 5, "budget: a fast CPU takes every candidate (%zu)", o.cpu.size());

    // same input -> same output
    llama_fn_cps_pick_out o2;
    llama_fn_cps_pick(in, o2);
    TCHECK(o2.cpu == o.cpu && o2.skip == o.skip && o2.tok == o.tok && o2.table == o.table, "deterministic");
}

//
// 2. late
//

static void test_late() {
    const int ne = 8;
    std::vector<float>    share = { 0.01f, 0.5f, 0.02f, 0.0f, 0.04f, 0.2f, 0.03f, 0.01f };
    std::vector<uint8_t>  hot(ne, 0);
    std::vector<uint64_t> bytes(ne, 100);
    hot[7] = 1;
    llama_fn_cps_late_in in;
    in.n_expert  = ne;
    in.n_tokens  = 400;
    in.share     = share.data();
    in.hot       = hot.data();
    in.bytes     = bytes.data();
    in.late_rows = 16; // predicted rows: 4, 200, 8, 0, 16, 80, 12, 4 (hot)
    std::vector<uint8_t> late(ne, 9);
    uint64_t b = llama_fn_cps_late(in, late.data());
    TCHECK(late == std::vector<uint8_t>({ 1, 0, 1, 1, 1, 0, 1, 0 }) && b == 500, "late by predicted rows, hot excluded");
    in.byte_cap = 250; // lowest predicted first: e3 (0), e0 (4), then e2 (8) would be 300
    b = llama_fn_cps_late(in, late.data());
    TCHECK(late == std::vector<uint8_t>({ 1, 0, 0, 1, 0, 0, 0, 0 }) && b == 200, "byte cap keeps the lowest predicted");
    in.byte_cap  = UINT64_MAX;
    in.late_rows = 0;
    b = llama_fn_cps_late(in, late.data());
    TCHECK(b == 0 && std::count(late.begin(), late.end(), 1) == 0, "late rows 0: none");
}

//
// 5. budget model
//

static void test_rls() {
    llama_fn_cps_rls r;
    std::mt19937 rng(5);
    std::uniform_int_distribution<int> ue(1, 40), up(1, 300);
    std::normal_distribution<double> nz(0.0, 5.0);
    for (int i = 0; i < 200; ++i) {
        const double x0 = ue(rng), x1 = up(rng);
        llama_fn_cps_rls_update(r, x0, x1, 50.0*x0 + 2.0*x1 + nz(rng));
    }
    TCHECK(std::fabs(r.th[0] - 50.0) < 5.0 && std::fabs(r.th[1] - 2.0) < 0.3, "RLS converges: %.2f %.3f", r.th[0], r.th[1]);
    // the rates move: forgetting follows
    for (int i = 0; i < 200; ++i) {
        const double x0 = ue(rng), x1 = up(rng);
        llama_fn_cps_rls_update(r, x0, x1, 120.0*x0 + 1.0*x1);
    }
    TCHECK(std::fabs(r.th[0] - 120.0) < 6.0 && std::fabs(r.th[1] - 1.0) < 0.2, "RLS follows a change: %.2f %.3f", r.th[0], r.th[1]);
    for (int i = 0; i < 50; ++i) {
        llama_fn_cps_rls_update(r, 1.0, 1.0, 1e12);
    }
    TCHECK(r.th[0] <= 5000.0 && r.th[1] <= 500.0 && r.th[0] >= 1.0 && r.th[1] >= 0.05, "RLS bounded");
    llama_fn_cps_rls_update(r, 1.0, 1.0, NAN);
    TCHECK(std::isfinite(r.th[0]) && std::isfinite(r.th[1]), "RLS ignores NaN");
}

//
// a small MoE model: 4 host layers with the type groups of Qwen3.8-Flash-Next UD (32-block down types differ per layer)
//

static const int64_t N_EMBD   = 512;
static const int64_t N_FF     = 160;
static const int64_t N_EXPERT = 32;
static const int64_t N_USED   = 4;
static const int     N_LAYER  = 4;

struct t_layer {
    int il = 0;
    ggml_type types[3];
    ggml_tensor * w[3] = { nullptr, nullptr, nullptr };
};

struct t_model {
    ggml_backend_t        cpu = nullptr;
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<t_layer>  layers;
    std::vector<uint8_t>  mirror; // a copy of the weights: the mock "pinned copy" the CPU share reads
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
        { GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, GGML_TYPE_IQ4_NL },
        { GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, GGML_TYPE_Q8_0 },
    };
    ggml_init_params ip = { ggml_tensor_overhead()*64, nullptr, true };
    m.ctx = ggml_init(ip);
    m.layers.resize(N_LAYER);
    for (int l = 0; l < N_LAYER; ++l) {
        t_layer & L = m.layers[l];
        L.il = l;
        for (int k = 0; k < 3; ++k) {
            L.types[k] = groups[l][k];
        }
        L.w[0] = ggml_new_tensor_3d(m.ctx, L.types[0], N_EMBD, N_FF, N_EXPERT);
        L.w[1] = ggml_new_tensor_3d(m.ctx, L.types[1], N_EMBD, N_FF, N_EXPERT);
        L.w[2] = ggml_new_tensor_3d(m.ctx, L.types[2], N_FF, N_EMBD, N_EXPERT);
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
    }
    const uint8_t * base = (const uint8_t *) ggml_backend_buffer_get_base(m.buf);
    m.mirror.assign(base, base + ggml_backend_buffer_get_size(m.buf));
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

// routing: n_hot "heavy" experts take most rows, the rest few or none; pool limits the experts used at all
static void route(std::mt19937 & rng, int T, int pool, std::vector<int32_t> & ids, std::vector<float> & w) {
    ids.assign((size_t) T*N_USED, 0);
    w.assign((size_t) T*N_USED, 0.0f);
    std::uniform_int_distribution<int> ue(0, pool - 1);
    std::uniform_int_distribution<int> uh(0, 5);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    for (int t = 0; t < T; ++t) {
        std::vector<int32_t> row;
        while ((int64_t) row.size() < N_USED) {
            const int32_t e = u01(rng) < 0.6f ? uh(rng) : ue(rng);
            if (std::find(row.begin(), row.end(), e) == row.end()) {
                row.push_back(e);
            }
        }
        float sum = 0.0f;
        for (int s = 0; s < N_USED; ++s) {
            ids[(size_t) t*N_USED + s] = row[s];
            w[(size_t) t*N_USED + s]   = 0.1f + u01(rng);
            sum += w[(size_t) t*N_USED + s];
        }
        for (int s = 0; s < N_USED; ++s) {
            w[(size_t) t*N_USED + s] /= sum;
        }
    }
}

//
// 3. the share op's math
//

static void test_share_math(t_model & m) {
    std::mt19937 rng(77);
    const int T = 300;
    std::vector<float> x((size_t) T*N_EMBD);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (auto & v : x) {
        v = u(rng);
    }
    for (int l = 0; l < N_LAYER; ++l) {
        t_layer & L = m.layers[l];
        std::vector<int32_t> ids;
        std::vector<float>   w;
        route(rng, T, (int) N_EXPERT, ids, w);
        std::vector<uint8_t> cpu_m(N_EXPERT, 0);
        for (int e = 6; e < 20; ++e) {
            cpu_m[e] = 1;
        }
        llama_fn_cps_ref_in in = { L.w[0], L.w[1], L.w[2], x.data(), w.data(), ids.data(), (int) N_USED, T };
        std::vector<int32_t> tok, tok3, tok8, tokv;
        std::vector<float> c1, c3, c8, cv, ref;
        const int cap = T;
        const bool ok1 = llama_fn_cps_share_test(in, cpu_m, 1, cap, true, tok, c1);
        const bool ok3 = llama_fn_cps_share_test(in, cpu_m, 3, cap, true, tok3, c3);
        const bool ok8 = llama_fn_cps_share_test(in, cpu_m, 8, cap, true, tok8, c8);
        const bool okv = llama_fn_cps_share_test(in, cpu_m, 4, cap, false, tokv, cv);
        const bool okr = llama_fn_cps_reference(in, cpu_m, tok, ref);
        TCHECK(ok1 && ok3 && ok8 && okv && okr, "layer %d: share op runs", l);
        if (!(ok1 && ok3 && ok8 && okv && okr)) {
            continue;
        }
        TCHECK(tok == tok3 && tok == tok8 && tok == tokv && !tok.empty(), "layer %d: same tokens", l);
        const size_t n = tok.size()*N_EMBD;
        TCHECK(memcmp(c1.data(), c3.data(), c1.size()*sizeof(float)) == 0 &&
               memcmp(c1.data(), c8.data(), c1.size()*sizeof(float)) == 0, "layer %d: nth 1 / 3 / 8 bitwise equal", l);
        TCHECK(memcmp(c1.data(), ref.data(), n*sizeof(float)) == 0, "layer %d: bitwise equal to the vec_dot reference", l);
        TCHECK(memcmp(cv.data(), ref.data(), n*sizeof(float)) == 0, "layer %d: vec_dot path bitwise equal to the reference", l);
        // padding columns repeat token tok[0]'s column
        bool pad_ok = true;
        for (int j = (int) tok.size(); j < cap; ++j) {
            pad_ok = pad_ok && memcmp(c1.data() + (size_t) j*N_EMBD, c1.data(), N_EMBD*sizeof(float)) == 0;
        }
        TCHECK(pad_ok, "layer %d: padding columns = column 0", l);

        // vs the ggml CPU chain: MUL_MAT_IDs that skip the non-CPU experts (src[3] table), swiglu, weights, slot sum
        ggml_init_params ip = { ggml_tensor_overhead()*256 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_tensor * xt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD, T);
        ggml_tensor * it = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_USED, T);
        ggml_tensor * wt = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, N_USED, T);
        ggml_tensor * tb = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, N_EXPERT);
        ggml_tensor * xin = ggml_reshape_3d(ctx, xt, N_EMBD, 1, T);
        ggml_tensor * up = ggml_mul_mat_id(ctx, L.w[0], xin, it);
        ggml_tensor * gt = ggml_mul_mat_id(ctx, L.w[1], xin, it);
        up->src[3] = tb; gt->src[3] = tb;
        up->op_params[0] = 0; gt->op_params[0] = 0;
        ggml_tensor * act = ggml_swiglu_split(ctx, gt, up);
        ggml_tensor * dn = ggml_mul_mat_id(ctx, L.w[2], act, it);
        dn->src[3] = tb;
        dn->op_params[0] = 0;
        ggml_tensor * ex = ggml_mul(ctx, dn, wt);
        ggml_tensor * sum = nullptr;
        for (int s = 0; s < N_USED; ++s) {
            ggml_tensor * v = ggml_view_2d(ctx, ex, N_EMBD, T, ex->nb[2], s*ex->nb[1]);
            sum = sum ? ggml_add(ctx, sum, v) : v;
        }
        ggml_build_forward_expand(gf, sum);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
        if (ggml_gallocr_alloc_graph(ga, gf)) {
            ggml_backend_tensor_set(xt, x.data(), 0, x.size()*sizeof(float));
            ggml_backend_tensor_set(it, ids.data(), 0, ids.size()*sizeof(int32_t));
            ggml_backend_tensor_set(wt, w.data(), 0, w.size()*sizeof(float));
            std::vector<int32_t> tbv(N_EXPERT);
            for (int e = 0; e < N_EXPERT; ++e) {
                tbv[e] = cpu_m[e] ? 0 : 1; // 0 = computed (op_params[0] = 0)
            }
            ggml_backend_tensor_set(tb, tbv.data(), 0, tbv.size()*sizeof(int32_t));
            ggml_backend_graph_compute(m.cpu, gf);
            std::vector<float> g((size_t) N_EMBD*T);
            ggml_backend_tensor_get(sum, g.data(), 0, g.size()*sizeof(float));
            double max_ref = 0.0, max_d = 0.0;
            for (size_t j = 0; j < tok.size(); ++j) {
                for (int64_t i = 0; i < N_EMBD; ++i) {
                    const double a = g[(size_t) tok[j]*N_EMBD + i];
                    const double b = c1[j*N_EMBD + i];
                    max_ref = std::max(max_ref, std::fabs(a));
                    max_d   = std::max(max_d, std::fabs(a - b));
                }
            }
            printf("  share op layer %d (%s/%s): %zu tokens, max |ggml chain - op| %.3g of %.3g\n", l, ggml_type_name(L.types[0]),
                   ggml_type_name(L.types[2]), tok.size(), max_d, max_ref);
            TCHECK(max_d <= 2e-3*max_ref, "layer %d: share op vs the ggml CPU chain (%.3g of %.3g)", l, max_d, max_ref);
        } else {
            TCHECK(false, "chain graph alloc");
        }
        ggml_gallocr_free(ga);
        ggml_free(ctx);
    }
}

//
// 4. share graphs on the stream
//

// mock pins: every expert range of the weights is "pinned" as the mirror copy (one ticket per query), except the ranges
// whose offset hits the drop rule: a copy that holds only half of the range (the CPU share drops that expert)
static std::atomic<int64_t> g_tk_out{0};
static std::atomic<int32_t> g_tk_next{0};
static t_model *            g_mock_model = nullptr;
static int                  g_drop_mod   = 0;

static bool mock_query(const void * p, size_t n, const void ** src, size_t * len, int32_t * ticket) {
    const uint8_t * base = (const uint8_t *) ggml_backend_buffer_get_base(g_mock_model->buf);
    const size_t    size = ggml_backend_buffer_get_size(g_mock_model->buf);
    const uint8_t * q    = (const uint8_t *) p;
    if (q < base || q + n > base + size) {
        *src = p;
        *len = n;
        *ticket = -1;
        return true;
    }
    const size_t off = (size_t) (q - base);
    *src    = g_mock_model->mirror.data() + off;
    const uint64_t h = ((uint64_t) (off/64)*2654435761ull) >> 16; // spread: expert offsets are multiples of nb[2]
    *len    = (g_drop_mod > 0 && h % (uint64_t) g_drop_mod == 1) ? n/2 : n;
    *ticket = g_tk_next.fetch_add(1);
    g_tk_out.fetch_add(1);
    return true;
}

static void mock_release(int32_t t) {
    if (t >= 0) {
        g_tk_out.fetch_sub(1);
    }
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

// up/gate/swiglu/down over w with ids (x: [n_embd, 1, T])
static ggml_tensor * chain(ggml_context * ctx, ggml_tensor * const w[3], ggml_tensor * x, ggml_tensor * ids) {
    ggml_tensor * up   = ggml_mul_mat_id(ctx, w[0], x, ids);
    ggml_tensor * gate = ggml_mul_mat_id(ctx, w[1], x, ids);
    ggml_tensor * act  = ggml_swiglu_split(ctx, gate, up);
    return ggml_mul_mat_id(ctx, w[2], act, ids);
}

static ggml_tensor * slot_sum(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * experts, ggml_tensor * w) {
    ggml_tensor * ex = ggml_mul(ctx, experts, w);
    ggml_build_forward_expand(gf, ex);
    ggml_tensor * sum = nullptr;
    for (int s = 0; s < N_USED; ++s) {
        ggml_tensor * v = ggml_view_2d(ctx, ex, ex->ne[0], ex->ne[2], ex->nb[2], s*ex->nb[1]);
        ggml_build_forward_expand(gf, v);
        sum = sum ? ggml_add(ctx, sum, v) : v;
        ggml_build_forward_expand(gf, sum);
    }
    return sum;
}

struct cmp_res {
    double max_d   = 0.0;
    double max_ref = 0.0;
    bool   bitwise = true;
};

static cmp_res compare(const ggml_tensor * a, const ggml_tensor * b) {
    cmp_res r;
    const int64_t n = ggml_nelements(a);
    const float * x = (const float *) a->data;
    const float * y = (const float *) b->data;
    for (int64_t i = 0; i < n; ++i) {
        if (!(x[i] == y[i]) || std::isnan(y[i])) {
            r.bitwise = false;
        }
        r.max_ref = std::max(r.max_ref, (double) std::fabs(x[i]));
        r.max_d   = std::max(r.max_d, std::isnan(y[i]) ? 1e30 : (double) std::fabs(x[i] - y[i]));
    }
    return r;
}

struct ub_plan {
    int  T;
    int  n_layers;     // layers in the graph
    bool release_last; // false: the graph is cut (no release of its last layer)
    bool share;        // build the share nodes (else plain gates)
    int  pool;         // experts the routing uses
};

struct scen {
    const char * name;
    int  mode;
    int  k_fixed;
    int  bufs;
    bool wrap;
    bool pins;
    int  drop_mod;
    bool exact;          // bitwise equal to the host chain
    std::vector<ub_plan> ubs;
};

static void run_scen(t_model & m, const scen & sc) {
    set_env("LLAMA_PREFILL_STREAM_BUFS", std::to_string(sc.bufs).c_str());
    set_env("LLAMA_PREFILL_STREAM_WRAP", sc.wrap ? "1" : "0");
    set_env("LLAMA_PREFILL_STREAM_MIN", "16");
    set_env("LLAMA_PREFILL_STREAM_CHUNK_MIB", "1");
    set_env("LLAMA_PREFILL_STREAM_RING_MIB", "2");
    set_env("LLAMA_PREFILL_STREAM_THREADS", "3");
    static int owner_tag = 0;
    const void * owner = &owner_tag;
    g_mock_model = &m;
    g_drop_mod   = sc.drop_mod;
    g_tk_out     = 0;
    llama_fn_cpushare_test_pins(sc.pins ? 1 : 0, mock_query, mock_release);
    llama_fn_cps_cfg cfg;
    cfg.mode      = sc.mode;
    cfg.rows      = 8;
    cfg.max_t     = 3072;
    cfg.cap       = 512;
    cfg.k_fixed   = sc.k_fixed;
    cfg.wait_ms   = 2000;
    cfg.late_rows = 16;
    cfg.early_kib = 0;
    cfg.nowait    = 0; // [TAG_R35_ROUTED] these scenarios check the v0.27.0 wait rule (LLAMA_R35_CPS_NOWAIT=0)
    TCHECK(llama_fn_cpushare_configure_cfg(cfg, nullptr, owner) == sc.mode, "[%s] configure", sc.name);
    TCHECK(llama_fn_cpushare_zero_expert() == (sc.mode == 1), "[%s] zero expert iff mode 1", sc.name);
    const bool ok = llama_prefill_stream_init_layers(descs(m), cpu_device(m), owner);
    TCHECK(ok, "[%s] stream init", sc.name);
    if (!ok) {
        llama_fn_cpushare_free(owner);
        return;
    }
    std::mt19937 rng(1234);
    bool all_ok = true;
    double worst = 0.0;
    for (size_t ub = 0; ub < sc.ubs.size(); ++ub) {
        const ub_plan & u = sc.ubs[ub];
        const int T = u.T;
        graph_run g;
        g.init();
        ggml_tensor * x = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, N_EMBD, T);
        ggml_set_input(x);
        ggml_tensor * xin = ggml_reshape_3d(g.ctx, x, N_EMBD, 1, T);
        std::vector<ggml_tensor *> ids(N_LAYER, nullptr), wts(N_LAYER, nullptr), out_ref(N_LAYER, nullptr), out_s(N_LAYER, nullptr);
        for (int l = 0; l < u.n_layers; ++l) {
            ids[l] = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, N_USED, T);
            wts[l] = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, 1, N_USED, T);
            ggml_set_input(ids[l]);
            ggml_set_input(wts[l]);
        }
        int n_built = 0;
        for (int l = 0; l < u.n_layers; ++l) {
            t_layer & L = m.layers[l];
            out_ref[l] = slot_sum(g.ctx, g.gf, chain(g.ctx, L.w, xin, ids[l]), wts[l]);
            ggml_set_output(out_ref[l]);
            ggml_build_forward_expand(g.gf, out_ref[l]);

            const llama_pfs_view * v = llama_prefill_stream_lookup(nullptr, L.w[0], T);
            TCHECK(v != nullptr, "[%s] lookup layer %d", sc.name, l);
            if (!v) {
                break;
            }
            llama_fn_cpushare_io io = { x, ids[l], wts[l], L.w[0], L.w[1], L.w[2], v, v->up, v->gate, v->down };
            bool planned = false;
            if (u.share) {
                planned = llama_fn_cpushare_build(g.ctx, g.gf, nullptr, m.cpu, l, &io, true);
                n_built += planned ? 1 : 0;
            }
            if (!io.gate_built) {
                llama_prefill_stream_build_gate(g.ctx, g.gf, v, x);
            }
            ggml_tensor * bw[3] = { io.up_mm ? io.up_mm : v->up, io.gate_mm ? io.gate_mm : v->gate, io.down_mm ? io.down_mm : v->down };
            ggml_tensor * down = chain(g.ctx, bw, xin, io.ids ? io.ids : ids[l]);
            if (!(!u.release_last && l == u.n_layers - 1)) {
                llama_prefill_stream_build_release(g.ctx, g.gf, v, down);
            }
            ggml_tensor * moe = slot_sum(g.ctx, g.gf, down, wts[l]);
            if (planned) {
                moe = llama_fn_cpushare_build_finish(g.ctx, nullptr, &io, moe);
            }
            out_s[l] = moe;
            ggml_set_output(out_s[l]);
            ggml_build_forward_expand(g.gf, out_s[l]);
        }
        if (u.share) {
            TCHECK(n_built == u.n_layers, "[%s] ub %zu: every layer has share nodes (%d)", sc.name, ub, n_built);
        }
        if (!g.alloc()) {
            TCHECK(false, "[%s] graph alloc", sc.name);
            break;
        }
        std::vector<float> xv((size_t) N_EMBD*T);
        std::uniform_real_distribution<float> ux(-1.0f, 1.0f);
        for (auto & v : xv) {
            v = ux(rng);
        }
        ggml_backend_tensor_set(x, xv.data(), 0, xv.size()*sizeof(float));
        for (int l = 0; l < u.n_layers; ++l) {
            std::vector<int32_t> iv;
            std::vector<float>   wv;
            route(rng, T, u.pool, iv, wv);
            ggml_backend_tensor_set(ids[l], iv.data(), 0, iv.size()*sizeof(int32_t));
            ggml_backend_tensor_set(wts[l], wv.data(), 0, wv.size()*sizeof(float));
        }
        if (ggml_backend_graph_compute(m.cpu, g.gf) != GGML_STATUS_SUCCESS) {
            TCHECK(false, "[%s] compute", sc.name);
            break;
        }
        for (int l = 0; l < u.n_layers; ++l) {
            if (!out_s[l]) {
                continue;
            }
            const cmp_res r = compare(out_ref[l], out_s[l]);
            const double rel = r.max_ref > 0 ? r.max_d/r.max_ref : r.max_d;
            worst = std::max(worst, rel);
            if (sc.exact) {
                TCHECK(r.bitwise, "[%s] ub %zu layer %d: bitwise equal to the host chain (max |d| %.3g)", sc.name, ub, l, r.max_d);
                all_ok = all_ok && r.bitwise;
            } else {
                TCHECK(rel <= 2e-3, "[%s] ub %zu layer %d: close to the host chain (rel %.3g)", sc.name, ub, l, rel);
                all_ok = all_ok && rel <= 2e-3;
            }
        }
        if (u.release_last && u.n_layers == N_LAYER) {
            TCHECK(g_tk_out.load() == 0, "[%s] ub %zu: pin tickets balanced (%lld out)", sc.name, ub, (long long) g_tk_out.load());
        }
    }
    const llama_pfs_share_counters c = llama_prefill_stream_share_counters();
    const llama_fn_cps_sums s = llama_fn_cpushare_take();
    printf("  [%s] share jobs %llu, skipped %.2f MiB, late copied %.2f MiB, timeouts %llu, invalidated %llu, plans %llu, "
           "CPU experts %llu (pairs %llu, tokens %llu, dropped %llu), worst rel %.3g%s\n", sc.name,
           (unsigned long long) c.jobs, c.skip_bytes/1048576.0, c.late_bytes/1048576.0, (unsigned long long) c.timeouts,
           (unsigned long long) c.invalidated, (unsigned long long) s.layers, (unsigned long long) s.cpu_experts,
           (unsigned long long) s.cpu_pairs, (unsigned long long) s.cpu_tokens, (unsigned long long) s.dropped, worst,
           all_ok ? "" : " - FAILED");
    bool any_share = false;
    for (const auto & u : sc.ubs) {
        any_share = any_share || u.share;
    }
    if (any_share) {
        TCHECK(c.jobs > 0, "[%s] share jobs ran", sc.name);
        TCHECK(c.skip_bytes > 0, "[%s] the stream skipped late bytes", sc.name);
        TCHECK(c.timeouts == 0, "[%s] no plan wait timed out", sc.name);
    }
    if (sc.mode == 1 && sc.pins && sc.k_fixed > 0 && any_share) {
        TCHECK(s.cpu_experts > 0 && s.cpu_pairs > 0, "[%s] the CPU computed experts", sc.name);
        if (sc.drop_mod > 0) {
            TCHECK(s.dropped > 0, "[%s] partial copies were dropped", sc.name);
        }
    }
    if (sc.mode == 1 && (!sc.pins || sc.k_fixed == 0)) {
        TCHECK(s.cpu_experts == 0, "[%s] no CPU experts without pins / with k 0", sc.name);
    }
    llama_prefill_stream_free(owner);
    llama_fn_cpushare_free(owner);
    TCHECK(g_tk_out.load() == 0, "[%s] pin tickets balanced after free (%lld out)", sc.name, (long long) g_tk_out.load());
    TCHECK(llama_fn_cpushare_cfg() == nullptr, "[%s] freed", sc.name);
    llama_fn_cpushare_test_pins(-1, nullptr, nullptr);
}

// (g) a share ubatch whose plans never come: every share job waits its bound, then copies everything (exact)
// (h) a free while the streamer waits for a plan returns at once
struct begin_ud {
    std::vector<uint8_t> late;
};

static void begin_op(ggml_tensor * dst, int ith, int nth, void * ud) {
    (void) dst; (void) nth;
    if (ith != 0) {
        return;
    }
    const begin_ud * b = (const begin_ud *) ud;
    llama_prefill_stream_share_begin(b->late.data(), (int) N_EXPERT);
}

static void test_no_plan(t_model & m, bool free_while_waiting) {
    const char * name = free_while_waiting ? "free while waiting" : "no plan -> timeout";
    set_env("LLAMA_PREFILL_STREAM_BUFS", "2");
    set_env("LLAMA_PREFILL_STREAM_WRAP", "1");
    set_env("LLAMA_PREFILL_STREAM_MIN", "16");
    set_env("LLAMA_PREFILL_STREAM_CHUNK_MIB", "1");
    set_env("LLAMA_PREFILL_STREAM_RING_MIB", "2");
    set_env("LLAMA_PREFILL_STREAM_THREADS", "3");
    static int owner_tag2 = 0;
    const void * owner = &owner_tag2;
    llama_fn_cps_cfg cfg;
    cfg.mode    = 2;
    cfg.wait_ms = free_while_waiting ? 60000 : 30;
    cfg.nowait  = 0; // [TAG_R35_ROUTED] the v0.27.0 wait rule (a late layer times out)
    TCHECK(llama_fn_cpushare_configure_cfg(cfg, nullptr, owner) == 2, "[%s] configure", name);
    if (!llama_prefill_stream_init_layers(descs(m), cpu_device(m), owner)) {
        TCHECK(false, "[%s] stream init", name);
        llama_fn_cpushare_free(owner);
        return;
    }
    begin_ud bu;
    bu.late.assign((size_t) N_LAYER*N_EXPERT, 0);
    for (int l = 1; l < N_LAYER; ++l) {
        for (int e = 0; e < N_EXPERT; e += 2) {
            bu.late[(size_t) l*N_EXPERT + e] = 1;
        }
    }
    const int T = 48;
    graph_run g;
    g.init();
    ggml_tensor * x = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, N_EMBD, T);
    ggml_set_input(x);
    ggml_tensor * xin = ggml_reshape_3d(g.ctx, x, N_EMBD, 1, T);
    ggml_tensor * args[1] = { ggml_view_1d(g.ctx, x, 1, 0) };
    ggml_tensor * bn = ggml_custom_4d(g.ctx, GGML_TYPE_F32, 1, 1, 1, 1, args, 1, begin_op, 1, &bu);
    ggml_build_forward_expand(g.gf, bn);
    std::vector<ggml_tensor *> ids(N_LAYER), out_ref(N_LAYER), out_s(N_LAYER);
    const int n_layers = free_while_waiting ? 0 : N_LAYER;
    for (int l = 0; l < n_layers; ++l) {
        ids[l] = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, N_USED, T);
        ggml_set_input(ids[l]);
        t_layer & L = m.layers[l];
        out_ref[l] = chain(g.ctx, L.w, xin, ids[l]);
        ggml_set_output(out_ref[l]);
        ggml_build_forward_expand(g.gf, out_ref[l]);
        const llama_pfs_view * v = llama_prefill_stream_lookup(nullptr, L.w[0], T);
        llama_prefill_stream_build_gate_share(g.ctx, g.gf, v, bn);
        ggml_tensor * bw[3] = { v->up, v->gate, v->down };
        out_s[l] = chain(g.ctx, bw, xin, ids[l]);
        llama_prefill_stream_build_release(g.ctx, g.gf, v, out_s[l]);
        ggml_set_output(out_s[l]);
        ggml_build_forward_expand(g.gf, out_s[l]);
    }
    TCHECK(g.alloc(), "[%s] alloc", name);
    std::mt19937 rng(9);
    std::vector<float> xv((size_t) N_EMBD*T);
    std::uniform_real_distribution<float> ux(-1.0f, 1.0f);
    for (auto & v : xv) {
        v = ux(rng);
    }
    ggml_backend_tensor_set(x, xv.data(), 0, xv.size()*sizeof(float));
    for (int l = 0; l < n_layers; ++l) {
        std::vector<int32_t> iv;
        std::vector<float>   wv;
        route(rng, T, (int) N_EXPERT, iv, wv);
        ggml_backend_tensor_set(ids[l], iv.data(), 0, iv.size()*sizeof(int32_t));
    }
    TCHECK(ggml_backend_graph_compute(m.cpu, g.gf) == GGML_STATUS_SUCCESS, "[%s] compute", name);
    for (int l = 0; l < n_layers; ++l) {
        const cmp_res r = compare(out_ref[l], out_s[l]);
        TCHECK(r.bitwise, "[%s] layer %d: bitwise equal (max |d| %.3g)", name, l, r.max_d);
    }
    if (!free_while_waiting) {
        const llama_pfs_share_counters c = llama_prefill_stream_share_counters();
        printf("  [%s] share jobs %llu, timeouts %llu, plan wait %.1f ms, skipped %.2f MiB\n", name, (unsigned long long) c.jobs,
               (unsigned long long) c.timeouts, c.plan_wait_ms, c.skip_bytes/1048576.0);
        TCHECK(c.timeouts == (uint64_t) (N_LAYER - 1) && c.skip_bytes == 0, "[%s] every late layer timed out, nothing skipped", name);
    } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(200)); // the streamer reaches job 1's plan wait
    }
    const auto t0 = std::chrono::steady_clock::now();
    llama_prefill_stream_free(owner);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    printf("  [%s] free took %.1f ms\n", name, ms);
    TCHECK(ms < 5000.0, "[%s] free returned (%.0f ms)", name, ms);
    llama_fn_cpushare_free(owner);
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::thread([]() {
        std::this_thread::sleep_for(std::chrono::seconds(600));
        fprintf(stderr, "test-fn-l22-cpushare: TIMEOUT after 600 s (a handshake hangs)\n");
        fflush(stderr);
        std::_Exit(3);
    }).detach();
    const char * k_env[] = { "LLAMA_PREFILL_STREAM_MIN", "LLAMA_PREFILL_STREAM_BUFS", "LLAMA_PREFILL_STREAM_WRAP",
        "LLAMA_PREFILL_STREAM_CHUNK_MIB", "LLAMA_PREFILL_STREAM_RING_MIB", "LLAMA_PREFILL_STREAM_THREADS",
        "LLAMA_PREFILL_STREAM_STATS", "LLAMA_PREFILL_STREAM_LEND", "LLAMA_PREFILL_STREAM_D2D", "LLAMA_PREFILL_STREAM_AHEAD" };
    for (const char * n : k_env) {
        set_env(n, nullptr);
    }

    printf("test-fn-l22-cpushare: unset\n");
    TCHECK(llama_fn_cpushare_cfg() == nullptr && !llama_fn_cpushare_zero_expert(), "nothing configured");
    {
        llama_fn_cps_cfg off;
        TCHECK(llama_fn_cpushare_configure_cfg(off, nullptr, &off) == 0 && llama_fn_cpushare_cfg() == nullptr, "mode 0 configures nothing");
        llama_fn_cpushare_io io = {};
        TCHECK(!llama_fn_cpushare_build(nullptr, nullptr, nullptr, nullptr, 0, &io, true) && !io.gate_built &&
               llama_fn_cpushare_build_finish(nullptr, nullptr, &io, nullptr) == nullptr, "no share without a configuration");
    }

    printf("test-fn-l22-cpushare: pick / late / budget\n");
    test_pick();
    test_late();
    test_rls();

    t_model m;
    if (!build_model(m)) {
        fprintf(stderr, "cannot build the test model\n");
        return 1;
    }

    printf("test-fn-l22-cpushare: share op math\n");
    test_share_math(m);

    printf("test-fn-l22-cpushare: share graphs on the stream\n");
    const int Ts = 48;
    std::vector<ub_plan> base_ubs = {
        { Ts, N_LAYER, true, true, (int) N_EXPERT },
        { Ts + 8, N_LAYER, true, true, 20 },
        { Ts, N_LAYER, true, true, (int) N_EXPERT },
        { Ts + 16, N_LAYER, true, true, 12 },
    };
    // (e) BUFS = N_LAYER keeps every layer in its bank across ubatches: a partial bank of a share ubatch must not serve the
    // next ubatch, which routes the experts the share ubatch skipped
    std::vector<ub_plan> keep_ubs = {
        { Ts, N_LAYER, true, true, 10 },
        { Ts, N_LAYER, true, false, (int) N_EXPERT },
        { Ts, N_LAYER, true, true, 10 },
        { Ts, N_LAYER, true, true, (int) N_EXPERT },
    };
    // (f) aborted share ubatches (graph cut after 1 or 2 layers, no release of the last one), then full ones
    std::vector<ub_plan> cut_ubs = {
        { Ts, N_LAYER, true, true, (int) N_EXPERT },
        { Ts, 2, false, true, 16 },
        { Ts, N_LAYER, true, true, (int) N_EXPERT },
        { Ts, 1, false, true, 16 },
        { Ts, 3, false, true, 8 },
        { Ts, N_LAYER, true, true, (int) N_EXPERT },
        { Ts, N_LAYER, true, false, (int) N_EXPERT },
    };
    const scen scens[] = {
        { "mode 2 skip only",            2,  -1, 2, true,  false, 0, true,  base_ubs },
        { "mode 2, 3 banks, no wrap",    2,  -1, 3, false, false, 0, true,  base_ubs },
        { "mode 1, k 0",                 1,   0, 2, true,  true,  0, true,  base_ubs },
        { "mode 1, no pins",             1,   6, 2, true,  false, 0, true,  base_ubs },
        { "mode 1 + pins, k 6",          1,   6, 2, true,  true,  0, false, base_ubs },
        { "mode 1 + pins, k 40, drops",  1,  40, 2, true,  true,  5, false, base_ubs },
        { "mode 1 + pins, budget",       1,  -1, 2, true,  true,  0, false, base_ubs },
        { "mode 1 + pins, 1 bank",       1,  40, 1, true,  true,  0, false, base_ubs },
        { "(e) partial banks, mode 2",   2,  -1, N_LAYER, true, false, 0, true, keep_ubs },
        { "(e) partial banks, mode 1",   1,  40, N_LAYER, true, true,  0, false, keep_ubs },
        { "(f) cut ubatches, mode 1",    1,  40, 2, true,  true,  0, false, cut_ubs },
        { "(f) cut ubatches, mode 2",    2,  -1, 2, false, false, 0, true,  cut_ubs },
    };
    for (const auto & sc : scens) {
        run_scen(m, sc);
    }

    printf("test-fn-l22-cpushare: plan waits\n");
    test_no_plan(m, false);
    test_no_plan(m, true);

    for (const char * n : k_env) {
        set_env(n, nullptr);
    }
    free_model(m);
    printf("test-fn-l22-cpushare: %d checks, %d errors%s\n", g_checks, g_fail, g_fail ? "" : " - PASSED");
    return g_fail == 0 ? 0 : 1;
}
