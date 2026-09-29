// [TAG_FN_QSA_FUSED] The fused qwen4exp QSA indexer ops (ggml_qsa_score + ggml_qsa_topk) against the unfused graph of
// models/qwen4exp.cpp build_qsa_top_k (get_rows, member sums, scale, rms_norm, mul, rope_multi, mul_mat, relu, head
// sum, block bias, cell expand, mask add, top_k), both on the CPU backend, on a cache laid out as set_input_qsa lays
// it out: full blocks of 4 cells, the incomplete tail on the spare block (bias 1e9), unused block slots at -inf.
//
//   (a) block scores: the fused op equals the graph up to float rounding (relative 1e-4)
//   (b) selection: the fused top-k equals the graph's top-k set; a cell may differ only where its value is within
//       float rounding of the k-th value (a near-tie), and every tail cell a query sees is selected
//   (c) the same with the per-cell bias form (no block bias), two streams, and a 70-query batch
//
// No GPU, no model. Exit code 0 iff every check passes.
//
//   test-qsa-fused

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iterator>
#include <random>
#include <vector>

static int g_fail   = 0;
static int g_checks = 0;

#define QCHECK(cond, ...)                                                   \
    do {                                                                    \
        ++g_checks;                                                         \
        if (!(cond)) {                                                      \
            ++g_fail;                                                       \
            if (g_fail <= 100) {                                            \
                fprintf(stderr, "FAIL %s:%d: ", __func__, __LINE__);        \
                fprintf(stderr, __VA_ARGS__);                               \
                fprintf(stderr, "\n");                                      \
            }                                                               \
        }                                                                   \
    } while (0)

struct qsa_case {
    const char * name;
    int64_t      n_pos;      // tokens in the cache (cells 0 .. n_pos-1 hold positions 0 .. n_pos-1)
    int64_t      n_kv;       // cell window (a multiple of 256)
    int64_t      n_tps;      // queries per stream, at the last positions
    int64_t      n_stream;
    bool         blk_bias;   // block bias + f16 mask (the deployed form), else the per-cell f32 bias
};

static void run_case(const qsa_case & c) {
    const int64_t D        = 128;
    const int64_t H        = 4;
    const int64_t r        = 4;
    const int64_t n_kv     = c.n_kv;
    const int64_t n_blocks = (n_kv + r - 1)/r;
    const int64_t n_tps    = c.n_tps;
    const int64_t n_stream = c.n_stream;
    const int64_t width    = std::min<int64_t>(n_kv, 2048 + r - 1);

    const int   n_rot       = 64;
    int         sections[4] = { 11, 11, 10, 0 };
    const int   mode        = GGML_ROPE_TYPE_IMROPE;
    const float eps         = 1e-6f;

    const size_t mem = (size_t) 256*1024*1024;
    ggml_init_params ip = { mem, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    QCHECK(ctx != nullptr, "%s: ggml_init", c.name);
    if (!ctx) {
        return;
    }

    std::mt19937 rng(77 + (uint32_t) (c.n_pos + n_tps*31 + n_stream));

    // inputs
    ggml_tensor * k_all     = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, D, n_kv, n_stream);
    ggml_tensor * blk_cells = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, r*n_blocks, n_stream);
    ggml_tensor * cell_blk  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_kv, n_stream);
    ggml_tensor * blk_pos   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4*n_blocks*n_stream);
    ggml_tensor * norm_w    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
    ggml_tensor * q         = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, H, n_tps*n_stream);
    ggml_tensor * kq_mask   = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, n_tps, 1, n_stream);
    ggml_tensor * bias      = c.blk_bias ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_blocks, n_tps, n_stream)
                                         : ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_kv, n_tps, n_stream);

    {
        std::uniform_real_distribution<float> du(-1.0f, 1.0f);
        std::vector<float> kf(D*n_kv*n_stream);
        for (auto & x : kf) { x = du(rng); }
        ggml_quantize_chunk(GGML_TYPE_Q8_0, kf.data(), k_all->data, 0, n_kv*n_stream, D, nullptr);

        float * w = (float *) norm_w->data;
        for (int64_t e = 0; e < D; ++e) { w[e] = 0.5f + 0.5f*(du(rng) + 1.0f); }
        float * qd = (float *) q->data;
        for (int64_t i = 0; i < ggml_nelements(q); ++i) { qd[i] = du(rng); }
    }

    // set_input_qsa's layout for one sequence per stream: full groups are blocks 0 .. n_full-1 (bid order = position
    // order), the incomplete tail maps to the spare block n_full, every other slot is unused
    const int64_t n_full = c.n_pos / r;
    const int64_t dead   = n_full < n_blocks ? n_full : n_blocks - 1;
    {
        int32_t * bc = (int32_t *) blk_cells->data;
        int32_t * cb = (int32_t *) cell_blk->data;
        int32_t * bp = (int32_t *) blk_pos->data;
        std::fill(bc, bc + r*n_blocks*n_stream, 0);
        std::fill(bp, bp + 4*n_blocks*n_stream, 0);
        for (int64_t s = 0; s < n_stream; ++s) {
            for (int64_t b = 0; b < n_full; ++b) {
                for (int64_t i = 0; i < r; ++i) {
                    bc[s*r*n_blocks + r*b + i] = (int32_t) (r*b + i);
                }
                for (int sec = 0; sec < 4; ++sec) {
                    bp[sec*(n_blocks*n_stream) + s*n_blocks + b] = (int32_t) (b*r);
                }
            }
            for (int64_t j = 0; j < n_kv; ++j) {
                const bool pooled = j < c.n_pos && j/r < n_full;
                cb[s*n_kv + j] = (int32_t) (pooled ? j/r : dead);
            }
        }

        ggml_fp16_t * m  = (ggml_fp16_t *) kq_mask->data;
        float       * bd = (float *) bias->data;
        for (int64_t s = 0; s < n_stream; ++s) {
            for (int64_t t = 0; t < n_tps; ++t) {
                const int64_t qp         = c.n_pos - n_tps + t;
                const int64_t tail_start = (qp + 1)/r*r;
                for (int64_t j = 0; j < n_kv; ++j) {
                    const bool vis = j < c.n_pos && j <= qp;
                    m[(s*n_tps + t)*n_kv + j] = ggml_fp32_to_fp16(vis ? 0.0f : -INFINITY);
                }
                if (c.blk_bias) {
                    float * row = bd + (s*n_tps + t)*n_blocks;
                    for (int64_t b = 0; b < n_blocks; ++b) {
                        row[b] = b >= n_full ? -INFINITY : (b*r >= tail_start ? 1e9f : 0.0f);
                    }
                    if (n_full < n_blocks) {
                        row[dead] = 1e9f;
                    }
                } else {
                    float * row = bd + (s*n_tps + t)*n_kv;
                    for (int64_t j = 0; j < n_kv; ++j) {
                        float v = -INFINITY;
                        if (j < c.n_pos && j <= qp) {
                            v = j >= tail_start ? 1e9f : (j/r < n_full ? 0.0f : -INFINITY);
                        }
                        row[j] = v;
                    }
                }
            }
        }
    }

    // the unfused graph (models/qwen4exp.cpp build_qsa_top_k)
    ggml_tensor * members = ggml_get_rows(ctx, k_all, blk_cells);
    members = ggml_reshape_4d(ctx, members, D, r, n_blocks, n_stream);
    ggml_tensor * pooled = nullptr;
    for (int64_t i = 0; i < r; ++i) {
        ggml_tensor * slice = ggml_cont(ctx, ggml_view_3d(ctx, members, D, n_blocks, n_stream, members->nb[2], members->nb[3], i*members->nb[1]));
        pooled = pooled ? ggml_add(ctx, pooled, slice) : slice;
    }
    pooled = ggml_scale(ctx, pooled, 1.0f/(float) r);
    pooled = ggml_reshape_3d(ctx, pooled, D, n_blocks*n_stream, 1);
    pooled = ggml_mul(ctx, ggml_rms_norm(ctx, pooled, eps), norm_w);
    pooled = ggml_reshape_3d(ctx, pooled, D, 1, n_blocks*n_stream);
    pooled = ggml_rope_multi(ctx, pooled, blk_pos, nullptr, n_rot, sections, mode, 262144, 1e7f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    pooled = ggml_reshape_3d(ctx, pooled, D, n_blocks, n_stream);

    ggml_tensor * sc = ggml_mul_mat(ctx, pooled, ggml_reshape_3d(ctx, q, D, H*n_tps, n_stream));
    sc = ggml_reshape_4d(ctx, sc, n_blocks, H, n_tps, n_stream);
    sc = ggml_relu(ctx, sc);
    ggml_tensor * summed = nullptr;
    for (int64_t h = 0; h < H; ++h) {
        ggml_tensor * slice = ggml_view_3d(ctx, sc, n_blocks, n_tps, n_stream, sc->nb[2], sc->nb[3], h*sc->nb[1]);
        summed = summed ? ggml_add(ctx, summed, slice) : ggml_cont(ctx, slice);
    }
    ggml_tensor * score_u = c.blk_bias ? ggml_add(ctx, summed, bias) : summed;
    ggml_tensor * expanded = ggml_get_rows(ctx, ggml_cont(ctx, ggml_permute(ctx, score_u, 1, 0, 2, 3)), cell_blk);
    expanded = ggml_cont(ctx, ggml_permute(ctx, expanded, 1, 0, 2, 3));
    if (c.blk_bias) {
        expanded = ggml_add(ctx, expanded, ggml_reshape_3d(ctx, ggml_cast(ctx, kq_mask, GGML_TYPE_F32), n_kv, n_tps, n_stream));
    } else {
        expanded = ggml_add(ctx, expanded, bias);
    }
    ggml_tensor * top_u = ggml_top_k(ctx, expanded, (int) width);

    // the fused ops
    ggml_tensor * score_f = ggml_qsa_score(ctx, k_all, blk_cells, blk_pos, norm_w, q, c.blk_bias ? bias : nullptr, (int) r, eps,
            n_rot, sections, mode, 262144, 1e7f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    ggml_tensor * top_f = ggml_qsa_topk(ctx, score_f, cell_blk, c.blk_bias ? kq_mask : bias, (int) width);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(gf, top_u);
    ggml_build_forward_expand(gf, expanded);
    ggml_build_forward_expand(gf, top_f);
    const ggml_status st = ggml_graph_compute_with_ctx(ctx, gf, 4);
    QCHECK(st == GGML_STATUS_SUCCESS, "%s: compute failed", c.name);

    // (a) block scores
    {
        double max_rel = 0.0;
        int64_t n_inf_diff = 0;
        for (int64_t i = 0; i < ggml_nelements(score_u); ++i) {
            const float u = ((const float *) score_u->data)[i];
            const float f = ((const float *) score_f->data)[i];
            if (std::isinf(u) || std::isinf(f)) {
                n_inf_diff += u != f;
                continue;
            }
            max_rel = std::max(max_rel, (double) std::fabs(u - f) / std::max(1.0, (double) std::fabs(u)));
        }
        QCHECK(n_inf_diff == 0, "%s: %" PRId64 " block scores differ in their -inf entries", c.name, n_inf_diff);
        QCHECK(max_rel < 1e-4, "%s: block scores differ, max relative error %.3g", c.name, max_rel);
        printf("  %-24s scores max rel err %.3g\n", c.name, max_rel);
    }

    // (b) selection
    {
        const float * ev = (const float *) expanded->data;
        int64_t n_mismatch = 0;
        int64_t n_bad      = 0;
        int64_t n_tail_bad = 0;
        for (int64_t s = 0; s < n_stream; ++s) {
            for (int64_t t = 0; t < n_tps; ++t) {
                const int64_t row = s*n_tps + t;
                const float * v   = ev + row*n_kv;

                std::vector<float> sorted(v, v + n_kv);
                std::nth_element(sorted.begin(), sorted.begin() + (width - 1), sorted.end(), std::greater<float>());
                const float T = sorted[width - 1];
                const float tol = std::isfinite(T) ? 1e-4f*std::max(1.0f, std::fabs(T)) : 0.0f;

                std::vector<int32_t> a((const int32_t *) top_u->data + row*width, (const int32_t *) top_u->data + (row + 1)*width);
                std::vector<int32_t> b((const int32_t *) top_f->data + row*width, (const int32_t *) top_f->data + (row + 1)*width);
                std::sort(a.begin(), a.end());
                std::sort(b.begin(), b.end());
                std::vector<int32_t> only;
                std::set_symmetric_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(only));
                n_mismatch += (int64_t) only.size();
                for (const int32_t cell : only) {
                    // a -inf filler may be any masked cell; a finite one must sit at the cut
                    if (std::isfinite(v[cell]) && std::fabs(v[cell] - T) > tol) {
                        ++n_bad;
                    }
                }

                // every visible tail cell (spare block, bias 1e9) is selected
                const int64_t qp = c.n_pos - n_tps + t;
                for (int64_t j = (c.n_pos/r)*r; j < c.n_pos && j <= qp; ++j) {
                    n_tail_bad += !std::binary_search(b.begin(), b.end(), (int32_t) j);
                }
            }
        }
        QCHECK(n_bad == 0, "%s: %" PRId64 " selected cells differ away from the k-th value", c.name, n_bad);
        QCHECK(n_tail_bad == 0, "%s: %" PRId64 " visible tail cells not selected", c.name, n_tail_bad);
        printf("  %-24s selection: %" PRId64 " cells differ, all at the cut (near-ties or -inf fillers)\n", c.name, n_mismatch);
    }

    ggml_free(ctx);
}

int main() {
    ggml_cpu_init();

    const qsa_case cases[] = {
        { "decode 9001",           9001,  9216,  1, 1, true  },
        { "verify 3 of 30002",    30002, 30208,  3, 1, true  },
        { "short 1500",            1500,  1536,  2, 1, true  },   // fewer cells than the budget: fillers
        { "two streams",           6003,  6144,  2, 2, true  },
        { "per-cell bias",         8190,  8192,  3, 1, false },
        { "prefill 70",           12345, 12544, 70, 1, true  },
    };

    for (const qsa_case & c : cases) {
        run_case(c);
    }

    printf("test-qsa-fused: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
