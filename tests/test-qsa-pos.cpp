// [TAG_FN_R4_QSA_POS] ggml_qsa_mask (the QSA attention mask from the positional vectors) against the explicit path of
// qwen4exp's build_qsa_sel, on the CPU backend, bit for bit.
//
// The explicit path (src/models/qwen4exp.cpp, kept as the default): an explicit [n_kv, n_q] f16 KQ mask from the
// positions (the rule of llama_kv_cache::set_input_kq_mask for one stream: visible iff the cell holds a position of the
// query's sequence that is <= the query's), an all -inf [n_kv + n_sel, n_q] row block, zeros scattered at the selected
// cells (dead slots - picked pools that score -inf and n_kv sentinels - into their own dump rows), the first n_kv
// columns viewed and the KQ mask added. The ops below are the ones build_qsa_sel builds, in the same order.
//
// Cases: one sequence and two (the two-row vectors), live scores of every kind (positive, +0, -0, -inf), padded pools
// (n_kv), tail cells missing (n_kv) or out of range, duplicates, a budget wider than n_kv, and query chunks (views of
// q_pos, sel and live with row strides, as LLAMA_QSA_POS_CHUNK builds them).

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

struct qsa_case {
    int64_t n_kv;
    int64_t n_q;
    int64_t n_top;  // picked pools per query
    int64_t kpool;  // cells per pool
    bool    ms;     // two sequences
    int     seed;
};

struct qsa_data {
    std::vector<int32_t> kv_pos;   // [n_kv] (+ [n_kv] sets)
    std::vector<int32_t> q_pos;    // [n_q]  (+ [n_q] bits)
    std::vector<int32_t> sel_pool; // [kpool*n_top, n_q]
    std::vector<int32_t> tail;     // [kpool - 1, n_q]
    std::vector<float>   score;    // [n_top, n_q]
    std::vector<ggml_fp16_t> kq;   // [n_kv, n_q] explicit mask
};

qsa_data make_data(const qsa_case & c) {
    std::mt19937 rng(c.seed);
    qsa_data d;
    const int rows = c.ms ? 2 : 1;
    d.kv_pos.assign(c.n_kv*rows, 0);
    int32_t p[2] = { 0, 0 };
    const int64_t off = c.n_kv/5;
    for (int64_t j = 0; j < c.n_kv; ++j) {
        int32_t pos = -1;
        int32_t set = 0;
        if (j >= off && (j - off) % 7 != 6) {
            const int sq = c.ms ? (int) ((j/5) % 2) : 0;
            pos = p[sq]++;
            set = 1 << sq;
            if (c.ms && (j % 13) == 0) {
                set = 3; // a cell shared by both sequences (seq_cp)
            }
        }
        d.kv_pos[j] = pos;
        if (c.ms) {
            d.kv_pos[c.n_kv + j] = set;
        }
    }
    d.q_pos.assign(c.n_q*rows, 0);
    for (int64_t i = 0; i < c.n_q; ++i) {
        const int sq = c.ms ? (int) (i % 2) : 0;
        const int32_t last = std::max<int32_t>(0, p[sq] - 1);
        d.q_pos[i] = std::max<int32_t>(0, last - (int32_t) ((c.n_q - 1 - i)/rows));
        if (c.ms) {
            d.q_pos[c.n_q + i] = 1 << sq;
        }
    }
    // the explicit mask, as set_input_kq_mask writes it for one stream
    d.kq.assign(c.n_kv*c.n_q, ggml_fp32_to_fp16(-INFINITY));
    for (int64_t i = 0; i < c.n_q; ++i) {
        for (int64_t j = 0; j < c.n_kv; ++j) {
            const int32_t kp = d.kv_pos[j];
            bool vis = kp >= 0 && kp <= d.q_pos[i];
            if (c.ms) {
                vis = vis && (((uint32_t) d.kv_pos[c.n_kv + j]) & (uint32_t) d.q_pos[c.n_q + i]) != 0;
            }
            if (vis) {
                d.kq[i*c.n_kv + j] = ggml_fp32_to_fp16(0.0f);
            }
        }
    }
    const int64_t n_blk = (c.n_kv + c.kpool - 1)/c.kpool;
    d.sel_pool.assign(c.kpool*c.n_top*c.n_q, 0);
    d.tail.assign((c.kpool - 1)*c.n_q, 0);
    d.score.assign(c.n_top*c.n_q, 0.0f);
    for (int64_t i = 0; i < c.n_q; ++i) {
        for (int64_t b = 0; b < c.n_top; ++b) {
            // a block in range, sometimes a repeated one (a duplicate selection), sometimes a padded one
            const int64_t blk = (int64_t) (rng() % (uint32_t) (n_blk + 1));
            const bool pad = (rng() % 9) == 0;
            for (int64_t m = 0; m < c.kpool; ++m) {
                const int64_t cell = blk*c.kpool + m;
                d.sel_pool[(i*c.n_top + b)*c.kpool + m] = pad || cell >= c.n_kv ? (int32_t) c.n_kv : (int32_t) cell;
            }
            const uint32_t r = rng() % 8;
            d.score[i*c.n_top + b] = r == 0 ? -INFINITY : r == 1 ? 0.0f : r == 2 ? -0.0f : (float) (rng() % 5000)/100.0f;
        }
        for (int64_t m = 0; m < c.kpool - 1; ++m) {
            // the incomplete tail: a cell, or the n_kv sentinel of a missing one
            const bool miss = (rng() % 3) == 0;
            d.tail[i*(c.kpool - 1) + m] = miss ? (int32_t) c.n_kv : (int32_t) (rng() % (uint32_t) c.n_kv);
        }
    }
    return d;
}

ggml_tensor * new_i32(ggml_context * ctx, int64_t ne0, int64_t ne1, const std::vector<int32_t> & v) {
    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, ne0, ne1);
    memcpy(t->data, v.data(), ggml_nbytes(t));
    return t;
}

// build_qsa_sel's explicit path (src/models/qwen4exp.cpp), op for op
ggml_tensor * build_old(ggml_context * ctx0, const qsa_case & c, ggml_tensor * sel_pool, ggml_tensor * tail,
        ggml_tensor * top_score, ggml_tensor * kq_mask) {
    const int64_t n_tokens   = c.n_q;
    const int64_t kpool      = c.kpool;
    const int64_t n_top_pool = c.n_top;
    const int64_t n_kv       = c.n_kv;

    ggml_tensor * sel_idx = ggml_concat(ctx0, sel_pool, tail, 0);
    const int64_t n_sel = sel_idx->ne[0];

    ggml_tensor * mask_all = ggml_new_tensor_4d(ctx0, kq_mask->type, n_kv + n_sel, 1, 1, 1);
    mask_all = ggml_fill(ctx0, mask_all, -INFINITY);
    mask_all = ggml_repeat_4d(ctx0, mask_all, n_kv + n_sel, n_tokens, 1, 1);
    mask_all = ggml_reshape_3d(ctx0, mask_all, 1, n_kv + n_sel, n_tokens);

    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, kq_mask->type, n_sel, 1, 1, 1);
    zeros = ggml_fill(ctx0, zeros, 0.0f);
    zeros = ggml_repeat_4d(ctx0, zeros, n_sel, n_tokens, 1, 1);
    zeros = ggml_reshape_3d(ctx0, zeros, 1, n_sel, n_tokens);

    ggml_tensor * live_pool = ggml_clamp(ctx0, ggml_scale_bias(ctx0, top_score, 1.0f, 1.0f), 0.0f, 1.0f);
    live_pool = ggml_reshape_2d(ctx0, ggml_repeat_4d(ctx0, live_pool, kpool, n_top_pool, n_tokens, 1), kpool*n_top_pool, n_tokens);
    ggml_tensor * live_tail = ggml_cast(ctx0, tail, GGML_TYPE_F32);
    live_tail = ggml_clamp(ctx0, ggml_scale_bias(ctx0, live_tail, -1.0f, (float) n_kv), 0.0f, 1.0f);
    ggml_tensor * live = ggml_concat(ctx0, live_pool, live_tail, 0);

    ggml_tensor * dump  = ggml_scale_bias(ctx0, ggml_cumsum(ctx0, ggml_fill(ctx0, live, 1.0f)), 1.0f, (float) (n_kv - 1));
    ggml_tensor * idx_f = ggml_cast(ctx0, sel_idx, GGML_TYPE_F32);
    idx_f   = ggml_add(ctx0, ggml_mul(ctx0, ggml_sub(ctx0, idx_f, dump), live), dump);
    sel_idx = ggml_cast(ctx0, idx_f, GGML_TYPE_I32);

    ggml_tensor * sel = ggml_set_rows(ctx0, mask_all, zeros, ggml_reshape_3d(ctx0, sel_idx, n_sel, n_tokens, 1));

    const size_t row = sel->nb[2];
    sel = ggml_view_4d(ctx0, sel, n_kv, kq_mask->ne[1], kq_mask->ne[2], kq_mask->ne[3],
            row, row*kq_mask->ne[1], row*kq_mask->ne[1]*kq_mask->ne[2], 0);
    return ggml_add(ctx0, sel, kq_mask);
}

int run_case(const qsa_case & c, int n_threads) {
    const qsa_data d = make_data(c);
    const int64_t n_sel = c.kpool*c.n_top + c.kpool - 1;
    const int64_t rows  = c.ms ? 2 : 1;

    // ~[n_kv + n_sel, n_q] f16 several times, plus small tensors
    const size_t mem = (size_t) (c.n_kv + n_sel)*c.n_q*2*12 + (size_t) n_sel*c.n_q*4*24 + (size_t) c.n_kv*c.n_q*8 + (64u << 20);
    ggml_init_params ip = { mem, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        fprintf(stderr, "ggml_init failed\n");
        return 1;
    }

    ggml_tensor * kv_pos   = new_i32(ctx, c.n_kv, rows, d.kv_pos);
    ggml_tensor * q_pos    = new_i32(ctx, c.n_q, rows, d.q_pos);
    ggml_tensor * sel_pool = new_i32(ctx, c.kpool*c.n_top, c.n_q, d.sel_pool);
    ggml_tensor * tail     = new_i32(ctx, c.kpool - 1, c.n_q, d.tail);
    ggml_tensor * score    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, c.n_top, c.n_q);
    memcpy(score->data, d.score.data(), ggml_nbytes(score));
    ggml_tensor * kq_mask  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, c.n_kv, c.n_q);
    memcpy(kq_mask->data, d.kq.data(), ggml_nbytes(kq_mask));

    ggml_tensor * ref = build_old(ctx, c, sel_pool, tail, score, kq_mask);

    // the new op on the same selection (the concatenated indices) and the picked scores [n_top, n_q]
    ggml_tensor * sel_idx = ggml_concat(ctx, sel_pool, tail, 0);
    ggml_tensor * live    = ggml_reshape_2d(ctx, score, c.n_top, c.n_q);
    ggml_tensor * out     = ggml_qsa_mask(ctx, kv_pos, q_pos, sel_idx, live, (int32_t) c.kpool);

    // query chunks: views with row strides, as build_attn_qsa makes them (3 chunks)
    std::vector<ggml_tensor *> outs_c;
    std::vector<int64_t>       c0s;
    const int64_t chunk = std::max<int64_t>(1, (c.n_q + 2)/3);
    for (int64_t c0 = 0; c0 < c.n_q; c0 += chunk) {
        const int64_t nc = std::min(chunk, c.n_q - c0);
        ggml_tensor * qp_c = ggml_view_2d(ctx, q_pos, nc, q_pos->ne[1], q_pos->nb[1], c0*q_pos->nb[0]);
        ggml_tensor * sl_c = ggml_view_2d(ctx, sel_idx, sel_idx->ne[0], nc, sel_idx->nb[1], c0*sel_idx->nb[1]);
        ggml_tensor * lv_c = ggml_view_2d(ctx, live, live->ne[0], nc, live->nb[1], c0*live->nb[1]);
        outs_c.push_back(ggml_qsa_mask(ctx, kv_pos, qp_c, sl_c, lv_c, (int32_t) c.kpool));
        c0s.push_back(c0);
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(gf, ref);
    ggml_build_forward_expand(gf, out);
    for (ggml_tensor * t : outs_c) {
        ggml_build_forward_expand(gf, t);
    }
    if (ggml_graph_compute_with_ctx(ctx, gf, n_threads) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed\n");
        ggml_free(ctx);
        return 1;
    }

    int64_t n_bad = 0;
    int64_t n_bad_c = 0;
    int64_t n_zero = 0;
    for (int64_t i = 0; i < c.n_q; ++i) {
        const ggml_fp16_t * r = (const ggml_fp16_t *) ((const char *) ref->data + i*ref->nb[1]);
        const ggml_fp16_t * o = (const ggml_fp16_t *) ((const char *) out->data + i*out->nb[1]);
        size_t k = 0;
        while (k + 1 < c0s.size() && c0s[k + 1] <= i) {
            k++;
        }
        const ggml_fp16_t * oc = (const ggml_fp16_t *) ((const char *) outs_c[k]->data + (i - c0s[k])*outs_c[k]->nb[1]);
        for (int64_t j = 0; j < c.n_kv; ++j) {
            const bool ok  = memcmp(&r[j], &o[j],  sizeof(ggml_fp16_t)) == 0;
            const bool okc = memcmp(&r[j], &oc[j], sizeof(ggml_fp16_t)) == 0;
            if (!ok && n_bad < 5) {
                fprintf(stderr, "  mismatch q %lld cell %lld: explicit %g, ggml_qsa_mask %g\n", (long long) i, (long long) j,
                        ggml_fp16_to_fp32(r[j]), ggml_fp16_to_fp32(o[j]));
            }
            n_bad   += !ok;
            n_bad_c += !okc;
            n_zero  += ggml_fp16_to_fp32(r[j]) == 0.0f;
        }
    }
    printf("qsa_mask n_kv %6lld n_q %4lld pools %4lld x %lld %s: %lld visible, %lld mismatches, %lld in the chunks: %s\n",
            (long long) c.n_kv, (long long) c.n_q, (long long) c.n_top, (long long) c.kpool, c.ms ? "2 seq" : "1 seq",
            (long long) n_zero, (long long) n_bad, (long long) n_bad_c, n_bad == 0 && n_bad_c == 0 && n_zero > 0 ? "OK" : "FAIL");
    ggml_free(ctx);
    return n_bad == 0 && n_bad_c == 0 && n_zero > 0 ? 0 : 1;
}

} // namespace

int main(int argc, char ** argv) {
    const int n_threads = argc > 1 ? std::max(1, atoi(argv[1])) : 4;
    ggml_cpu_init();

    const qsa_case cases[] = {
        { 256,    1,   8,   4, false, 1 },
        { 256,    7,   8,   4, true,  2 },
        { 1024,   33,  64,  4, false, 3 },
        { 1000,   16,  32,  8, true,  4 },   // n_kv not a multiple of the pool
        { 4096,   64,  512, 4, false, 5 },   // the Flash-Next budget (512 pools of 4 + 3 tail cells)
        { 4096,   64,  512, 4, true,  6 },
        { 2048,   4,   600, 4, false, 7 },   // a budget wider than the cache (every pool picked, fillers -inf)
        { 16384,  130, 512, 4, false, 8 },
    };
    int n_fail = 0;
    for (const qsa_case & c : cases) {
        n_fail += run_case(c, n_threads);
    }
    printf("%s: %d case(s) failed\n", n_fail == 0 ? "PASS" : "FAIL", n_fail);
    return n_fail == 0 ? 0 : 1;
}
