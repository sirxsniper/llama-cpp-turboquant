#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn.cuh"
#include "fattn-turbot-decl.cuh"   // [TAG_TURBOT] host declarations only, never turbot-tables.cuh
#include "turbot-set-rows.cuh"     // [TAG_TURBOT_ANY_ROUTE] ggml_cuda_turbot_any_on, host declarations only
#include "ggml-turbot.h"

#include <set>

// [TAG_TURBOT] every D=256 case call passes through here so a turbot K/V reaches its own kernel instances
// [TAG_TURBOT_ANY_D128] and every D=128 one (unless GGML_CUDA_FA_TURBOT_D128 is 0): the D=128 f16 ladder selects exactly
// the 16 pairs of fattn-turbot-decl.cuh. A turbot K/V only gets here when ggml_cuda_get_best_fattn_kernel admitted it.
template <int DKQ, int DV, int ncols1, int ncols2>
static void ggml_cuda_fattn_mma_case_dispatch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if constexpr (((DKQ == 256 && DV == 256) || (GGML_CUDA_FA_TURBOT_D128 && DKQ == 128 && DV == 128)) && ncols2 <= 8) {
        if (ggml_turbot_is_type(dst->src[1]->type)) {
            ggml_cuda_flash_attn_ext_turbot_case<DKQ, DV, ncols1, ncols2>(ctx, dst);
            return;
        }
    }
    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, ncols1, ncols2>(ctx, dst);
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// one list per group of ncols1 queries: a column is selected if any query of the group can see it
__launch_bounds__(256, 1)
static __global__ void flash_attn_mask_to_sparse_indices(
        const half * mask_ptr, int32_t * indices_ptr, int32_t * counts_ptr, const int ne30, const int n_queries,
        const int ncols1, const int n_kv_max, const int64_t s31, const int64_t s33) {
    ggml_cuda_pdl_sync();

    constexpr int values_per_lane = 8;
    const int tid      = threadIdx.x;
    const int warp     = tid / WARP_SIZE;
    const int lane     = tid % WARP_SIZE;
    const int sequence = blockIdx.y;
    const int group    = blockIdx.x;

    const int q0 = group*ncols1;
    const int q1 = min(q0 + ncols1, n_queries);

    const half * mask = mask_ptr + sequence*s33 + q0*s31;
    int32_t * indices = indices_ptr + (int64_t(sequence)*gridDim.x + group)*n_kv_max;

    __shared__ int warp_offsets[256/WARP_SIZE];
    __shared__ int row_count;
    __shared__ int chunk_count;

    if (tid == 0) {
        row_count = 0;
    }
    __syncthreads();

    for (int i0 = 0; i0 < ne30; i0 += blockDim.x*values_per_lane) {
        uint32_t selected_warp[values_per_lane];
        int warp_count = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int i = i0 + (warp*values_per_lane + item)*WARP_SIZE + lane;
            bool selected = false;
            for (int q = 0; q < q1 - q0 && !selected; ++q) {
                selected = i < ne30 && isfinite(__half2float(mask[q*s31 + i]));
            }
            selected_warp[item] = __ballot_sync(0xFFFFFFFF, selected);
            warp_count += __popc(selected_warp[item]);
        }

        if (lane == 0) {
            warp_offsets[warp] = warp_count;
        }
        __syncthreads();

        if (tid == 0) {
            int offset = 0;
#pragma unroll
            for (int iw = 0; iw < 256/WARP_SIZE; ++iw) {
                const int count = warp_offsets[iw];
                warp_offsets[iw] = offset;
                offset += count;
            }
            chunk_count = offset;
        }
        __syncthreads();

        const uint32_t lane_mask = lane == 0 ? 0 : (1u << lane) - 1;
        int warp_item_offset = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int i = i0 + (warp*values_per_lane + item)*WARP_SIZE + lane;
            const int dst = row_count + warp_offsets[warp] + warp_item_offset + __popc(selected_warp[item] & lane_mask);
            if ((selected_warp[item] & (uint32_t(1) << lane)) && dst < n_kv_max) {
                indices[dst] = i;
            }
            warp_item_offset += __popc(selected_warp[item]);
        }
        __syncthreads();

        if (tid == 0) {
            row_count += chunk_count;
        }
        __syncthreads();
    }

    const int count = min(row_count, n_kv_max);
    for (int i = count + tid; i < n_kv_max; i += blockDim.x) {
        indices[i] = -1;
    }
    if (tid == 0) {
        counts_ptr[int64_t(sequence)*gridDim.x + group] = count;
    }
    __syncthreads();

    // the dependent grid reads indices, signal once the row is complete
    ggml_cuda_pdl_lc();
}
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

void ggml_cuda_flash_attn_ext_compact_mask(
        const ggml_tensor * mask, int32_t * indices, int32_t * counts, int32_t n_queries, int32_t ncols1, int32_t n_kv_max, cudaStream_t stream) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(mask, indices, counts, n_queries, ncols1, n_kv_max, stream);
    GGML_ABORT("sparse flash attention is only supported on NVIDIA CUDA");
#else
    const int64_t s31 = mask->nb[1] / sizeof(half);
    const int64_t s33 = mask->nb[3] / sizeof(half);
    const dim3 blocks_num((n_queries + ncols1 - 1)/ncols1, mask->ne[3], 1);
    const dim3 block_dim(256, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params(blocks_num, block_dim, 0, stream);
    ggml_cuda_kernel_launch(flash_attn_mask_to_sparse_indices, launch_params,
        (const half *) mask->data, indices, counts, int(mask->ne[0]), n_queries, ncols1, n_kv_max, s31, s33);
    CUDA_CHECK(cudaGetLastError());
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

bool ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(const int cc, const ggml_tensor * dst, const int ncols1) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(cc, dst, ncols1);
    return false;
#else
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);

    const int64_t n_gather = (ncols1 == 1 ? Q->ne[1] : ncols1) * (int64_t) n_kv_max;

    // [TAG_SYNC_SPARSE_TURBOT] a turbot K/V has its own kernel with no sparse gather, and the sparse branch in
    // ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1 calls the f16 case directly (past the turbot dispatch). The
    // turbot graph passes n_kv_max = 0, so this only guards an API caller that sets n_kv_max on a turbot FA.
    return GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) && !ggml_turbot_is_type(K->type) &&
        mask != nullptr && n_kv_max > 0 && max_bias == 0.0f && logit_softcap == 0.0f &&
        mask->ne[0] == K->ne[1] && mask->ne[1] >= Q->ne[1] && mask->ne[2] == 1 &&
        K->ne[1] >= std::max<int64_t>(4096, 2*n_gather);
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

// [TAG_TURBOT_Q2_ROUTE] kill switch for the Q <= 2 turbot route below: TURBOT_Q2_ROUTE=0 sends Q = 2 back to the <2,8>
// instance (Q = 1 then takes [TAG_TURBOT_Q1_ROUTE] as before) for the A/B on one binary. Default on.
static bool turbot_q2_route_on() {
    static const bool on = [] {
        const char * e = getenv("TURBOT_Q2_ROUTE");
        return !(e && e[0] == '0');
    }();
    return on;
}

template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

    // [TAG_TURBOT_Q2_ROUTE] turbot only: Q <= 2 runs on the <4,8> instance, as Q = 1 already does ([TAG_TURBOT_Q1_ROUTE]).
    // Q = 2 is one slot verifying a 1-token draft, or two slots decoding one token each (unified KV: one FA call with
    // ne[1] = 2); it used to fall through to <2,8>. Nothing else changes: the kernel zero-fills the missing Q columns and
    // skips their output (and the fixup kernels skip rows >= ne01), and launch_fattn_turbot takes its row-wrapping
    // "turbot<ncols1> wrap" KV bounds scan for Q % ncols1 != 0 (the positional scan wraps with fastmodulo).
    if constexpr (DKQ == 256 && DV == 256 && ncols2 == 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 2 && ggml_turbot_is_type(dst->src[1]->type) && turbot_q2_route_on()) {
            ggml_cuda_fattn_mma_case_dispatch<DKQ, DV, 4, ncols2>(ctx, dst);
            return;
        }
    }

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if constexpr (ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, 1, ncols2)) {
        if (ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, 1)) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 1, ncols2>(ctx, dst);
            return;
        }
    }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
            // [TAG_TURBOT_Q1_ROUTE] turbot only: a single query runs on the <4,8> instance, not <1,8>. At the deployed
            // shape <4,8> computes 4x the queries at nb 4 and is still faster than <1,8> at nb 1 (B0, 131K 261 vs 483
            // us/op), so the extra columns cost less than the <1,8> kernel shape. The kernel zero-fills the missing Q
            // columns and skips their output; launch_fattn_turbot scans only the real mask row for the KV bounds.
            // With [TAG_TURBOT_Q2_ROUTE] on (default) Q = 1 already left above; this is the TURBOT_Q2_ROUTE=0 path.
            if constexpr (DKQ == 256 && DV == 256 && ncols2 == 8) {
                if (Q->ne[1] == 1 && ggml_turbot_is_type(dst->src[1]->type)) {
                    ggml_cuda_fattn_mma_case_dispatch<DKQ, DV, 4, ncols2>(ctx, dst);
                    return;
                }
            }
            ggml_cuda_fattn_mma_case_dispatch<DKQ, DV, 8/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if constexpr (ncols2 <= 16) {
        if (Q->ne[1] <= 16/ncols2) {
            ggml_cuda_fattn_mma_case_dispatch<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if (Q->ne[1] <= 32/ncols2 || (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING) ||
            (GGML_CUDA_CC_IS_AMD(cc) && DKQ > 256)) {
        ggml_cuda_fattn_mma_case_dispatch<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    // [TAG_FA_NCOLS_128] One more tier above 64, D=256 only. Each output tile rereads the
    // whole KV region, so doubling the tile width halves the passes. Gated on Q being wide
    // enough to actually fill it, otherwise the wider tile just wastes columns the way
    // Q=5 does against ncols1=8. FA_NCOLS128=0 disables it for A/B.
    if constexpr (DKQ == 256 && DV == 256 && ncols2 <= 8) {
        static const bool ncols128_on = [] {
            const char * e = getenv("FA_NCOLS128");
            return !(e && e[0] == '0');
        }();
        if (ncols128_on && Q->ne[1] >= 128/ncols2 && turing_mma_available(cc)) {
            ggml_cuda_fattn_mma_case_dispatch<DKQ, DV, 128/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    ggml_cuda_fattn_mma_case_dispatch<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
}

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];   // [TAG_FA_POS_MASK] a positional mask counts as a mask for routing
    const bool has_mask = mask != nullptr || dst->src[5] != nullptr;

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // Edge cases like no mask, ALiBi, unpadded K/V, or misaligned addresses for large data transfers
    //     are put into the template specialization without GQA optimizations.
    bool use_gqa_opt = has_mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // On Volta the GQA optimizations aren't as impactful vs. minimizing wasted compute:
    if (cc == GGML_CUDA_CC_VOLTA) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if constexpr (DKQ <= 256) {
            if (use_gqa_opt && gqa_ratio % 2 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
                return;
            }

            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
            return;
        } else {
            GGML_ABORT("fatal error");
        }
    }

    // [TAG_FA_NCOLS2_PROBE] ncols2 is the GQA packing factor: how many query heads share
    // one K/V tile read. The ladder below rounds UP, so a gqa_ratio that is not a power
    // of two packs slots it cannot fill - gqa_ratio 6 takes the >4 branch and packs 8,
    // computing 8 columns for 6 real heads (33% extra attention work). Packing smaller
    // is exact but re-reads K/V more often, so which wins is a measurement, not a
    // derivation. FA_NCOLS2=<1|2|4|8> forces it for that measurement.
    {
        static const int forced = [] {
            const char * e = getenv("FA_NCOLS2");
            const int v = e ? atoi(e) : 0;
            return (v == 1 || v == 2 || v == 4 || v == 8) ? v : 0;
        }();
        if (forced && use_gqa_opt) {
            switch (forced) {
                case 8: ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst); return;
                case 4: ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst); return;
                case 2: ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst); return;
                default: ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst); return;
            }
        }
    }

    // On RDNA it is preferable to minimize wasted compute vs. duplicate I/O for the mask.
    if (amd_wmma_available(cc)) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 2 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
            return;
        }
    }

    // [TAG_FA_NCOLS2_QWIDTH]
    // ncols2 is the GQA packing factor: how many query heads share one K/V tile read.
    // ntiles_z_gqa = ceil(gqa_ratio / ncols2) is how many times the kernel re-reads the
    // whole K/V region, so at gqa_ratio 6 the exact-divisor ladder below picks 2 and
    // reads the cache three times. Rounding up to 8 reads it once, at the cost of
    // computing eight head slots for six real heads.
    //
    // Which is right depends on the Q width, and the boundary is sharp. Measured with
    // test-backend-ops perf, D=256, GQA 6:1, turbo4 K and V, as us/run:
    //
    //           kv=32768          kv=131072         kv=245760
    //   nb   divisor roundup   divisor roundup   divisor roundup
    //    1     62.53   62.79    197.40  198.02    339.82  340.49   (VEC, unaffected)
    //    2    159.44  161.52    570.09  573.67   1031.46 1033.35   (VEC, unaffected)
    //    4    131.55   49.78    874.56  213.33   1632.41  379.17   <- -62% to -77%
    //    8     95.51   80.83    523.78  344.64    950.33  597.21   <- -15% to -37%
    //   12    119.95  129.98    590.50  612.91   1085.07 1125.78      +4% to +8%
    //   16    121.90  134.24    600.88  621.55   1088.72 1131.98      +3% to +10%
    //   32    200.37  247.23    946.29 1201.18   1702.72 2202.90     +23% to +29%
    //
    // Round-up wins only for nb 4..8 and loses from 12 up, because ncols1 is capped at
    // 64/ncols2: past 8 the kernel starts re-tiling over Q as well, so it pays the
    // wasted head slots AND more than one pass. Below 4 the VEC kernel takes the call
    // and this never runs. So the useful window is exactly a speculative verification
    // batch, and the gate is the largest width measured to win, not a round number.
    //
    // End to end on the server, Qwen3.8-27B-UD-Q5_K_XL, turbo4 KV, DFlash2 n_max 7
    // (Q = 8), greedy so draft acceptance is comparable:
    //
    //     depth    tok/s before   tok/s after      ms/step
    //     32768        79.50         135.03      38.04 -> 34.22
    //    131072        56.81          92.30      60.58 -> 43.25
    //    245760        39.35          63.41      86.67 -> 53.32
    //
    // FA_NCOLS2_MAXQ overrides the gate; 0 disables the rule entirely.
    if (use_gqa_opt) {
        static const int narrow_q_max = [] {
            const char * e = getenv("FA_NCOLS2_MAXQ");
            const int v = e ? atoi(e) : -1;
            return (v >= 0 && v <= 4096) ? v : 8;    // 0 disables the narrow-Q rule
        }();
        if (Q->ne[1] <= narrow_q_max) {
            // smallest power of two >= gqa_ratio, capped at 8 (the instantiated ladder).
            int n2 = 1;
            while (n2 < gqa_ratio && n2 < 8) {
                n2 *= 2;
            }
            switch (n2) {
                case 8: ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst); return;
                case 4: ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst); return;
                case 2: ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst); return;
                default: break;   // gqa_ratio 1: no packing possible, fall through
            }
        }
    }

    // [TAG_FA_NCOLS2_DIVISOR]
    // ncols2 is the GQA packing factor: how many query heads share one K/V tile read.
    // This ladder used to round UP (`> 4` -> 8), so a gqa_ratio that is not a power of
    // two packed slots it could not fill: gqa_ratio 6 took the `> 4` branch and computed
    // 8 columns for 6 real heads - 33% of the attention work thrown away, every tile.
    //
    // Packing the largest power of two that DIVIDES gqa_ratio is exact. It costs more
    // K/V tile reads (6/2 = 3 passes instead of 1), and the reasonable guess is that at
    // long context the extra reads outweigh the wasted compute. Measured, they do not:
    //
    //   Qwen3.8-27B (gqa_ratio 6), RTX 5090, pp512, turbo4 KV, r=2
    //     ncols2   d131072    d245760
    //          8   1047.04     596.80   <- rounding up (was the default)
    //          4   1005.79     607.95
    //          2   1223.53     746.21   <- exact divisor: +16.9% / +25.0%
    //
    // Powers of two are unaffected (8 divides 8, 4 divides 4), so this only changes
    // behaviour for ratios like 6, 12 or 5 that the old ladder over-packed.
    //
    // FA_NCOLS2=<1|2|4|8> overrides it for measurement.
    if (use_gqa_opt) {
        if (gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }
        if (gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }
        if (gqa_ratio % 2 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
            return;
        }
    }

    if constexpr (DKQ <= 256) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
    } else {
        GGML_ABORT("fatal error");
    }
}

static void ggml_cuda_flash_attn_ext_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];   // [TAG_FA_POS_MASK] a positional mask counts as a mask for routing
    const bool has_mask = mask != nullptr || dst->src[5] != nullptr;

    switch (Q->ne[0]) {
        case 64:
            GGML_ASSERT(V->ne[0] == 64);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 64,  64>(ctx, dst);
            break;
        case 80:
            GGML_ASSERT(V->ne[0] == 80);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 80,  80>(ctx, dst);
            break;
        case 96:
            GGML_ASSERT(V->ne[0] == 96);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 96,  96>(ctx, dst);
            break;
        case 112:
            GGML_ASSERT(V->ne[0] == 112);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<112, 112>(ctx, dst);
            break;
        case 128:
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<128, 128>(ctx, dst);
            break;
        case 192: {
            // MiMo-V2.5 / V2.5-Pro / V2-Flash: gqa_ratio is 8 (SWA) or 16 (full attn)
            GGML_ASSERT(V->ne[0] == 128);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));
            const bool use_gqa_opt = has_mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);
            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128, 16>(ctx, dst);
            } else {
                GGML_ASSERT(gqa_ratio % 8 == 0);
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128,  8>(ctx, dst);
            }
        } break;
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>(ctx, dst);
            break;
        case 320:
            // For Mistral Small 4, go straight to the ncols1 switch (ncols2=32-only build).
            GGML_ASSERT(V->ne[0] == 256);
            {
                float max_bias = 0.0f;
                memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

                const bool use_gqa_opt = has_mask && max_bias == 0.0f;
                GGML_ASSERT(use_gqa_opt);
                GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
                const int gqa_ratio = Q->ne[2] / K->ne[2];
                GGML_ASSERT(gqa_ratio % 32 == 0);

                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<320, 256, 32>(ctx, dst);
            }
            break;
        case 512:
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<512, 512>(ctx, dst);
            break;
        case 576: {
            // For Deepseek, go straight to the ncols1 switch to avoid compiling unnecessary kernels.
            GGML_ASSERT(V->ne[0] == 512);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

            const bool use_gqa_opt = has_mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);

            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio == 20) { // GLM 4.7 Flash
                if (cc >= GGML_CUDA_CC_DGX_SPARK) {
                    if (Q->ne[1] <= 8) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_BLACKWELL) {
                    if (Q->ne[1] <= 4 && K->ne[1] >= 65536) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 4) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    if (Q->ne[1] <= 4) {
                        if (K->ne[1] <= 16384) {
                            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                            break;
                        }
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 32>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                // Volta:
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
            } else if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512,  4>(ctx, dst);
            }
        } break;
        case 640: {
            // Padded turbo KV cache for GLM-4.7 Flash (K head_dim=576 zero-padded to 640).
            // D=640 shared memory (Q storage = ncols*(DKQ/2+4)*4) exceeds hardware limit at ncols1>=4.
            // Cap at ncols1=2 (ncols=32): Q=32*324*4=41KB + KV≈37KB = ~78KB total.
            GGML_ASSERT(V->ne[0] == 512);
            if (Q->ne[1] <= 1) {
                ggml_cuda_flash_attn_ext_mma_f16_case<640, 512, 1, 16>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_mma_f16_case<640, 512, 2, 16>(ctx, dst);
            }
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

#define FATTN_VEC_CASE(D, type_K_case, type_V_case)                                                                                \
    if constexpr (GGML_CUDA_FA_##type_K_case##_##type_V_case) {                                                                    \
        const bool type_K_okay = type_K == GGML_TYPE_##type_K_case || (type_K == GGML_TYPE_F32 && GGML_TYPE_##type_K_case == GGML_TYPE_F16); \
        const bool type_V_okay = type_V == GGML_TYPE_##type_V_case || (type_V == GGML_TYPE_F32 && GGML_TYPE_##type_V_case == GGML_TYPE_F16); \
        if (head_size == (D) && type_K_okay && type_V_okay) {                                                                      \
            return ggml_cuda_flash_attn_ext_vec_case<D, GGML_TYPE_##type_K_case, GGML_TYPE_##type_V_case>;                         \
        }                                                                                                                          \
    }                                                                                                                              \

#define FATTN_VEC_CASES_ALL_D(type_K_case, type_V_case) \
    FATTN_VEC_CASE( 64, type_K_case, type_V_case)       \
    FATTN_VEC_CASE(128, type_K_case, type_V_case)       \
    FATTN_VEC_CASE(256, type_K_case, type_V_case)       \

// [TAG_SYNC_FA_TURBO_VEC_CASES] The turbo VEC instances are always compiled (ggml-cuda/CMakeLists.txt appends every
// fattn-vec-instance-*turbo*.cu), so their cases are NOT gated by a GGML_CUDA_FA_<K>_<V> define and never take the
// f16 fallback. Exact type match: a turbo type has no F32 alias.
#define FATTN_VEC_CASE_TURBO(D, type_K_case, type_V_case)                                                  \
    if (head_size == (D) && type_K == GGML_TYPE_##type_K_case && type_V == GGML_TYPE_##type_V_case) {      \
        return ggml_cuda_flash_attn_ext_vec_case<D, GGML_TYPE_##type_K_case, GGML_TYPE_##type_V_case>;     \
    }                                                                                                      \

#define FATTN_VEC_CASES_TURBO_ALL_D(type_K_case, type_V_case) \
    FATTN_VEC_CASE_TURBO( 64, type_K_case, type_V_case)       \
    FATTN_VEC_CASE_TURBO(128, type_K_case, type_V_case)       \
    FATTN_VEC_CASE_TURBO(256, type_K_case, type_V_case)       \

// [TAG_TURBO4P] turbo4p is instantiated for D 128 and 256 only, so it needs a macro that
// does not reach for a D=64 instance that deliberately does not exist. A turbo4p head must
// be a whole number of 128-element WHT groups, and at D=64 two heads would share a norm.
#define FATTN_VEC_CASES_TURBO4P_D(type_K_case, type_V_case) \
    FATTN_VEC_CASE_TURBO(128, type_K_case, type_V_case)     \
    FATTN_VEC_CASE_TURBO(256, type_K_case, type_V_case)     \

typedef void (* fattn_vec_case_t)(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Vector kernel for the given head size and K/V types, nullptr if its template instance was not compiled:
static fattn_vec_case_t ggml_cuda_get_fattn_vec_case(const int64_t head_size, const ggml_type type_K, const ggml_type type_V) {
    FATTN_VEC_CASES_ALL_D(F16,  F16)
    FATTN_VEC_CASES_ALL_D(Q4_0, F16)
    FATTN_VEC_CASES_ALL_D(Q4_1, F16)
    FATTN_VEC_CASES_ALL_D(Q5_0, F16)
    FATTN_VEC_CASES_ALL_D(Q5_1, F16)
    FATTN_VEC_CASES_ALL_D(Q8_0, F16)
    FATTN_VEC_CASES_ALL_D(BF16, F16)

    FATTN_VEC_CASES_ALL_D(F16,  Q4_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q4_0)

    FATTN_VEC_CASES_ALL_D(F16,  Q4_1)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(BF16, Q4_1)

    FATTN_VEC_CASES_ALL_D(F16,  Q5_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q5_0)

    FATTN_VEC_CASES_ALL_D(F16,  Q5_1)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(BF16, Q5_1)

    FATTN_VEC_CASES_ALL_D(F16,  Q8_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q8_0)

    FATTN_VEC_CASES_ALL_D(F16,  BF16)
    FATTN_VEC_CASES_ALL_D(Q4_0, BF16)
    FATTN_VEC_CASES_ALL_D(Q4_1, BF16)
    FATTN_VEC_CASES_ALL_D(Q5_0, BF16)
    FATTN_VEC_CASES_ALL_D(Q5_1, BF16)
    FATTN_VEC_CASES_ALL_D(Q8_0, BF16)
    FATTN_VEC_CASES_ALL_D(BF16, BF16)

    // TurboQuant3 KV cache types (always enabled)
    FATTN_VEC_CASES_TURBO_ALL_D(TURBO3_0, TURBO3_0)

    // Mixed turbo3/q8_0 KV cache types
    FATTN_VEC_CASES_TURBO_ALL_D(TURBO3_0, Q8_0)
    FATTN_VEC_CASES_TURBO_ALL_D(Q8_0,     TURBO3_0)

    // TurboQuant2 KV cache types (always enabled)
    FATTN_VEC_CASES_TURBO_ALL_D(TURBO2_0, TURBO2_0)

    // Mixed turbo2/q8_0 KV cache types
    FATTN_VEC_CASES_TURBO_ALL_D(TURBO2_0, Q8_0)
    FATTN_VEC_CASES_TURBO_ALL_D(Q8_0,     TURBO2_0)

    // Mixed turbo3/turbo2 KV cache types
    FATTN_VEC_CASES_TURBO_ALL_D(TURBO3_0, TURBO2_0)
    FATTN_VEC_CASES_TURBO_ALL_D(TURBO2_0, TURBO3_0)

    // TurboQuant4 KV cache types (always enabled)
    FATTN_VEC_CASES_TURBO_ALL_D(TURBO4_0, TURBO4_0)

    // Mixed turbo4/q8_0 KV cache types
    FATTN_VEC_CASES_TURBO_ALL_D(TURBO4_0, Q8_0)
    FATTN_VEC_CASES_TURBO_ALL_D(Q8_0,     TURBO4_0)

    // Mixed turbo4/turbo3 KV cache types
    FATTN_VEC_CASES_TURBO_ALL_D(TURBO4_0, TURBO3_0)
    FATTN_VEC_CASES_TURBO_ALL_D(TURBO3_0, TURBO4_0)

    // Mixed turbo4/turbo2 KV cache types
    FATTN_VEC_CASES_TURBO_ALL_D(TURBO4_0, TURBO2_0)
    FATTN_VEC_CASES_TURBO_ALL_D(TURBO2_0, TURBO4_0)

    // [TAG_TURBO4P] turbo4p, the split-plane repack of turbo4 (always enabled)
    FATTN_VEC_CASES_TURBO4P_D(TURBO4P_0, TURBO4P_0)

    // Mixed turbo4p/q8_0 KV cache types
    FATTN_VEC_CASES_TURBO4P_D(TURBO4P_0, Q8_0)
    FATTN_VEC_CASES_TURBO4P_D(Q8_0,      TURBO4P_0)

    // Mixed turbo4p/turbo4 KV cache types. The two layouts hold identical values, so an
    // asymmetric cache costs no accuracy and lets K and V be repacked independently.
    FATTN_VEC_CASES_TURBO4P_D(TURBO4P_0, TURBO4_0)
    FATTN_VEC_CASES_TURBO4P_D(TURBO4_0,  TURBO4P_0)

    // [TAG_TURBO5P] turbo5p: same D set as turbo4p (whole WHT groups per head)
    FATTN_VEC_CASES_TURBO4P_D(TURBO5P_0, TURBO5P_0)
    FATTN_VEC_CASES_TURBO4P_D(TURBO5P_0, Q8_0)
    FATTN_VEC_CASES_TURBO4P_D(Q8_0,      TURBO5P_0)
    // [TAG_TURBO5P512]
    FATTN_VEC_CASES_TURBO4P_D(TURBO5P512_0, TURBO5P512_0)
    FATTN_VEC_CASES_TURBO4P_D(TURBO5P512_0, Q8_0)
    FATTN_VEC_CASES_TURBO4P_D(Q8_0,         TURBO5P512_0)

    return nullptr;
}

static void ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    fattn_vec_case_t vec_case = ggml_cuda_get_fattn_vec_case(Q->ne[0], K->type, V->type);
    if (vec_case == nullptr) {
        static bool warned = false;
        if (!warned) {
            GGML_LOG_WARN("%s: no FlashAttention vector kernel compiled for K/V types %s-%s, converting K and V to f16 instead (slow). "
                "Add \"%s-%s\" to GGML_CUDA_FA_QUANTS to compile it.\n",
                __func__, ggml_type_name(K->type), ggml_type_name(V->type), ggml_type_name(K->type), ggml_type_name(V->type));
            warned = true;
        }
        vec_case = ggml_cuda_get_fattn_vec_case(Q->ne[0], GGML_TYPE_F16, GGML_TYPE_F16);
    }
    GGML_ASSERT(vec_case != nullptr);
    vec_case(ctx, dst);
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE    =   0,
    BEST_FATTN_KERNEL_TILE    = 200,
    BEST_FATTN_KERNEL_VEC     = 100,
    BEST_FATTN_KERNEL_MMA_F16 = 400,
};

// K/V types for which there is a vector kernel template instance, other kernels convert these to f16:
static bool ggml_cuda_fattn_kv_type_supported(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_TURBO2_0:
        case GGML_TYPE_TURBO3_0:
        case GGML_TYPE_TURBO4_0:
        case GGML_TYPE_TURBO4P_0:
        case GGML_TYPE_TURBO5P_0:
        case GGML_TYPE_TURBO5P512_0:
            return true;
        default:
            return false;
    }
}

// [TAG_FA_VEC_Q2] Whether a two-token batch should leave the VEC kernel for MMA.
// VEC has no GQA packing at ncols1=2, so it pays gqa_ratio passes over the cache where
// MMA now pays one. Only worth it when there is more than one head to share.
// FA_VEC_Q2_MMA=0 restores the old routing for A/B.
static bool vec_q2_to_mma(int gqa_ratio) {
    static const int forced = [] {
        const char * e = getenv("FA_VEC_Q2_MMA");
        return e ? atoi(e) : -1;
    }();
    if (forced == 0) {
        return false;
    }
    return gqa_ratio >= 2;
}

static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(const int device, const ggml_tensor * dst) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED(device); GGML_UNUSED(dst);
    return BEST_FATTN_KERNEL_NONE;
#endif// FLASH_ATTN_AVAILABLE

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];   // [TAG_FA_POS_MASK] a positional mask counts as a mask for routing
    const bool has_mask = mask != nullptr || dst->src[5] != nullptr;

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // The effective batch size for the kernel can be increased by gqa_ratio.
    // The kernel versions without this optimization are also used for ALiBi, if there is no mask, or if the KV cache is not padded,
    bool gqa_opt_applies = gqa_ratio >= 2 && has_mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt_applies = false;
                break;
            }
        }
    }

    const int cc = ggml_cuda_info().devices[device].cc;

    switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 192:
            if (V->ne[0] != 128 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 8 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 320:
            if (V->ne[0] != 256 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 32 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 512:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
        case 640:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

    // [TAG_TURBOT] turbot K/V is read natively by its own MMA instances at every Q width, Q = 1 included. There is no
    // VEC or TILE instance and no F16 conversion (a turbot row cannot decode without the plan), so anything the MMA
    // turbot kernel cannot take is refused here, before any of the turbo routing below can see the type.
    if (ggml_turbot_is_type(K->type) || ggml_turbot_is_type(V->type)) {
        bool ok = ggml_turbot_is_type(K->type) && ggml_turbot_is_type(V->type) &&
                  Q->ne[0] == 256 && K->ne[0] == 256 && V->ne[0] == 256 && K->ne[2] == 4 && V->ne[2] == 4 &&
                  dst->src[7] != nullptr && dst->src[8] != nullptr &&
                  (!mask || mask->ne[2] == 1) && Q->ne[3] == 1 && turing_mma_available(cc);
        // [TAG_TURBOT_ANY_ROUTE] Any other KV geometry the op params name (docs/turbot/SPEC.md, ggml_turbot_geom_*):
        // D = 256 with 4, 2 or 1 KV heads, D = 128 with 8, 4 or 2 (one to four 256-value runs per row). Evaluated only
        // when the Qwen predicate above failed, so the 4 x 256 route is untouched; GGML_TURBOT_ANY=0 turns it off.
        if (!ok && ggml_cuda_turbot_any_on() && ggml_turbot_is_type(K->type) && ggml_turbot_is_type(V->type)) {
            ggml_turbot_op_params tp;
            if (ggml_turbot_op_params_get(dst, &tp) && ggml_turbot_geom_valid(tp.flags)) {
                const int64_t hd = ggml_turbot_geom_head_dim(tp.flags);
                const int64_t nh = ggml_turbot_geom_n_head(tp.flags);
                const bool d_ok = hd == 256 || (hd == 128 && GGML_CUDA_FA_TURBOT_D128);
                ok = d_ok && Q->ne[0] == hd && K->ne[0] == hd && V->ne[0] == hd && K->ne[2] == nh && V->ne[2] == nh &&
                     dst->src[7] != nullptr && dst->src[8] != nullptr &&
                     (!mask || mask->ne[2] == 1) && Q->ne[3] == 1 && turing_mma_available(cc);
            }
        }
        return ok ? BEST_FATTN_KERNEL_MMA_F16 : BEST_FATTN_KERNEL_NONE;
    }

    // [TAG_SYNC_FA_MIXED_KV] The old "#ifndef GGML_CUDA_FA_ALL_QUANTS: K->type != V->type only for turbo/q8_0
    // pairs" gate is gone with the macro. Mixed K/V types are admitted as upstream does; the turbo pairs that have
    // a VEC instance resolve through the ungated FATTN_VEC_CASE_TURBO entries in ggml_cuda_get_fattn_vec_case, any
    // other pair takes upstream's f16 fallback (every turbo type has a to_fp16 converter) instead of aborting.

    if (!ggml_cuda_fattn_kv_type_supported(K->type) || !ggml_cuda_fattn_kv_type_supported(V->type)) {
        return BEST_FATTN_KERNEL_NONE;
    }

    // turbo2/turbo3/turbo4 VEC kernels are only instantiated for D in {64, 128, 256}.
    if (K->type == GGML_TYPE_TURBO2_0 || K->type == GGML_TYPE_TURBO3_0 || K->type == GGML_TYPE_TURBO4_0) {
        if (K->ne[0] % 64 != 0) {
            return BEST_FATTN_KERNEL_NONE;
        }
    }

    // [TAG_TURBO4P] turbo4p carries a stricter geometry than "D is a multiple of 64",
    // because one block holds 1024 elements rather than 128 and therefore spans heads:
    //   - the head must be a whole number of 128-element WHT groups, or two heads would
    //     share a norm and one rotation would cover both;
    //   - the head must fit inside one block, or reaching it would need two pointers
    //     rather than a block base plus an element offset;
    //   - n_embd_k_gqa (= ne[0]*ne[2] for the FA view) must be a whole number of blocks,
    //     or a KV position would start mid-block and the position stride would stop being
    //     a whole number of blocks.
    // Qwen3.8-27B is 4 kv heads x 256 = 1024 exactly, i.e. one block per position.
    {
        // [TAG_TURBO5P512] the same three conditions against whichever block length the type
        // actually uses: the head must be whole WHT groups, a block must not split a head, and the
        // full row (head_dim x n_head_kv) must be an exact number of blocks. turbo5p512 exists
        // precisely because a 512-element row fails the third test against a 1024-element block.
        auto turbo_blk_geometry_ok = [](const ggml_tensor * t, const int64_t QK) {
            return t->ne[0] % QK_TURBO4P_GROUP == 0 &&
                   QK % t->ne[0] == 0 &&
                   (t->ne[0] * t->ne[2]) % QK == 0;
        };
        auto turbo_blk_len = [](ggml_type ty) -> int64_t {
            return ty == GGML_TYPE_TURBO5P512_0 ? QK_TURBO5P512 : QK_TURBO4P;
        };
        auto is_split_plane = [](ggml_type ty) {
            return ty == GGML_TYPE_TURBO4P_0 || ty == GGML_TYPE_TURBO5P_0 || ty == GGML_TYPE_TURBO5P512_0;
        };
        if (is_split_plane(K->type) && !turbo_blk_geometry_ok(K, turbo_blk_len(K->type))) {
            return BEST_FATTN_KERNEL_NONE;
        }
        if (is_split_plane(V->type) && !turbo_blk_geometry_ok(V, turbo_blk_len(V->type))) {
            return BEST_FATTN_KERNEL_NONE;
        }
        // Only the pairs that have a VEC instance. The mixed-type check above admits any
        // two turbo types, but turbo4p is instantiated against itself, q8_0 and turbo4 only
        // - turbo2 and turbo3 still do a per-element divergent constant-memory lookup, so
        // pairing them with turbo4p would be a measured LOSS anyway (K=turbo4 / V=turbo3 at
        // d131072: 101.07 -> 88.08 t/s). Refusing here beats aborting in the dispatch.
        if (is_split_plane(K->type) || is_split_plane(V->type)) {
            // [TAG_TURBO5P512] turbo5p512 is instantiated against itself and q8_0 only, and it
            // must NOT pair with turbo5p: the two carry different block lengths, and a single
            // kernel reads one geometry for both planes.
            const bool k512 = K->type == GGML_TYPE_TURBO5P512_0;
            const bool v512 = V->type == GGML_TYPE_TURBO5P512_0;
            if (k512 != v512 && (is_split_plane(K->type) && is_split_plane(V->type))) {
                return BEST_FATTN_KERNEL_NONE;
            }
            auto turbo4p_pairable = [](ggml_type t) {
                return t == GGML_TYPE_TURBO4P_0 || t == GGML_TYPE_TURBO5P_0 ||
                       t == GGML_TYPE_TURBO5P512_0 || t == GGML_TYPE_Q8_0 || t == GGML_TYPE_TURBO4_0;
            };
            if (!turbo4p_pairable(K->type) || !turbo4p_pairable(V->type)) {
                return BEST_FATTN_KERNEL_NONE;
            }
        }
    }

    if (mask && mask->ne[2] != 1) {
        return BEST_FATTN_KERNEL_NONE;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:
    // 192 satisfies % 64 == 0 but has no vec instance (DKQ != DV); force it onto the MMA path.
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && Q->ne[0] != 192 && K->ne[1] % FATTN_KQ_STRIDE == 0;

    // If Turing tensor cores are available, use them:
    if (turing_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                // the sparse gather exists only in the MMA kernel: (DKQ, DV, 1, 8) with GQA > 4
                const bool sparse_decode = gqa_opt_applies && gqa_ratio > 4 &&
                    ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(K->ne[0], V->ne[0], 1, 8) &&
                    ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, 1);
                if (!sparse_decode && cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 &&
                        !(gqa_ratio > 4 && (Q->ne[0] >= 256 || K->ne[1] >= 8192))) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            } else {
                // Quantized K/V goes to MMA, which needs an F16 copy of the cache.
                // That conversion used to be the dominant cost for turbo types because
                // turbo4_dequant_element did a divergent __constant__ lookup per element
                // (TURBO_CENTROIDS_4BIT[idx]); constant memory serialises when lanes read
                // different addresses. Fixed in convert.cu with a warp-cooperative kernel
                // (each lane holds one centroid, __shfl_sync broadcasts it).
                //
                // Measured RTX 5090, Qwen3.8-27B, ~90K ctx, DFlash2 n8:
                //   turbo4 MMA before conversion fix   45.2 tok/s
                //   turbo4 MMA after  conversion fix   70.8 tok/s   (26.3 GB)
                //   turbo4 VEC native                  46.6 tok/s
                //   q8_0   MMA                         69.7 tok/s   (29.4 GB)
                // So MMA is the right target once the conversion is not pathological;
                // routing turbo to VEC for wide Q was a workaround and is not needed.
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    // [TAG_TURBO_FA_ROUTING]
                    //
                    // Turbo K/V now go to VEC at ANY depth for narrow Q. VEC reads the
                    // quantized cache natively; MMA cannot, so it materialises the WHOLE
                    // cache as F16 on every call (fattn-common.cuh, to_fp16 over
                    // ggml_nelements(K)) - a decode -> re-encode -> decode round trip paid
                    // once per layer per token.
                    //
                    // This used to route to MMA past 4096 because VEC's per-cell centroid
                    // lookup dominated (the old note here cited MMA 70.8 vs VEC 46.6 at
                    // ~90K). Two fixes inverted that: a PRMT-based 4-at-a-time centroid
                    // gather in the VEC KQ dot (replacing a ~12-deep dependent select
                    // chain) and coalesced stores in the turbo4 dequant kernels.
                    //
                    // Re-measured, RTX 5090, Qwen3.8-27B-UD-Q4_K_XL, llama-bench tg64 r=3:
                    //   depth        MMA+convert   VEC native
                    //        0        66.11         66.36
                    //    65536        47.61         53.87   (+13%)
                    //   131072        35.50         44.40   (+25%)
                    //   245760        24.57         33.73   (+37%)
                    // VEC wins at every depth and ties at 0, so the depth cutoff is gone.
                    //
                    // Only narrow Q is affected: the VEC kernel is instantiated for
                    // cols_per_block 1 and 2 only (fattn-vec.cuh), so a speculative batch
                    // (Q->ne[1] == n_draft+1) still takes MMA exactly as before.
                    //
                    // TURBO_FA_MMA=1 restores the old depth-based MMA routing for A/B.
                    const bool turbo_K = K->type == GGML_TYPE_TURBO2_0 ||
                                         K->type == GGML_TYPE_TURBO3_0 ||
                                         K->type == GGML_TYPE_TURBO4_0 ||
                                         K->type == GGML_TYPE_TURBO4P_0;
                    // Cached: this sits inside the per-FA-op kernel selection, so an
                    // uncached getenv here is a locked CRT lookup on every attention op.
                    static const bool want_mma = [] {
                        const char * e = getenv("TURBO_FA_MMA");
                        return e && e[0] == '1';
                    }();

                    if (turbo_K && want_mma && K->ne[1] >= 4096) {
                        // fall through to MMA (previous behaviour)
                    } else if (Q->ne[1] == 1 || (Q->ne[1] <= 2 && !vec_q2_to_mma(gqa_ratio))) {
                        // [TAG_FA_VEC_Q2] VEC is instantiated for cols_per_block 1 and 2
                        // only, and its two-column instance is <ncols1=2, ncols2=1> - no
                        // GQA packing at all, so it reads the cache gqa_ratio times. At
                        // Q=1 the kernel packs six ways and reads it once, so the two are
                        // not comparable and the old `<= 2` lumped them together.
                        //
                        // Measured, D=256, GQA 6:1, turbo4, us/run:
                        //            nb=1    nb=2    nb=4
                        //   kv 32768  62.5   159.6    49.7
                        //   kv 131072 196.9  568.9   212.4
                        //   kv 245760 339.0 1030.2   377.7
                        //
                        // Two tokens cost ~2.7x four tokens, because four goes to MMA and
                        // reads the cache once while two stays on VEC and reads it six
                        // times. Send Q=2 to MMA as well when the group is wide enough for
                        // that to matter. Forcing a WIDER Q onto VEC is still wrong - it
                        // tiles into ceil(ncols/2) passes, measured 80 -> 12 t/s at 247K.
                        return BEST_FATTN_KERNEL_VEC;
                    }
                } else {
                    if (Q->ne[1] == 1) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            }
            if (!gqa_opt_applies && Q->ne[1] == 1) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    const int ncols2_max = Q->ne[0] == 320 ? 32 : ((Q->ne[0] == 576 || Q->ne[0] == 640 || Q->ne[0] == 192) ? 16 : 8);
    int gqa_ratio_eff = 1;
    while (gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
        gqa_ratio_eff *= 2;
    }

    if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel && Q->ne[1] * gqa_ratio_eff <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        if (Q->ne[1] * gqa_ratio_eff <= 16) {
            return BEST_FATTN_KERNEL_TILE; // On Volta tensor cores are only faster for sufficiently large matrices.
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // AMD MFMA needs a certain minimum batch size to outscale the tile kernel for large head sizes.
    // Note: Q->ne[0] <= 256 already excludes turbo4 head-dim 640 from the AMD path.
    if ((amd_mfma_available(cc) && Q->ne[0] <= 256) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if ((Q->ne[0] <= 64 && Q->ne[1] * gqa_ratio_eff > 8)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 128 && Q->ne[1] * gqa_ratio_eff > 16)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 256 && Q->ne[1] * gqa_ratio_eff > 64)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
    }

    // AMD WMMA is faster than the tile kernel if the wide tiles with high arithmetic intensity can be utilized.
    if ((amd_wmma_available(cc) && gqa_opt_applies && Q->ne[0] <= 256) && Q->ne[0] != 40 && Q->ne[0] != 72 &&
            Q->ne[1] * gqa_ratio_eff > (Q->ne[0] <= 128 ? 8 : 16)) {
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (Q->ne[1] == 1) {
                if (!gqa_opt_applies) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        } else {
            if (Q->ne[1] <= 2) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    return BEST_FATTN_KERNEL_TILE;
}

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst) {
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_EXT);

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    GGML_ASSERT(K != nullptr);
    GGML_ASSERT(V != nullptr);

    const best_fattn_kernel kernel = ggml_cuda_get_best_fattn_kernel(device, dst);

    bool need_f16_K = false;
    bool need_f16_V = false;

    // [TAG_TURBOT] A turbot cache is always read natively by its own MMA instances and has no F16 conversion at all,
    // so no F16 scratch is ever reserved for it (SPEC decision 10), whatever the kernel choice above says.
    if (ggml_turbot_is_type(K->type)) {
        const ggml_cuda_flash_attn_ext_f16_extra_data turbot_extra =
            ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, need_f16_K, need_f16_V);
        return turbot_extra.end - (uintptr_t) dst->data;
    }

    switch (kernel) {
        case BEST_FATTN_KERNEL_TILE:
            need_f16_K = true;
            need_f16_V = true;
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            // [TAG_TURBO_NATIVE_PREDICATE] DELIBERATE OVER-RESERVATION. Do not "fix" this.
            //
            // When the MMA kernel reads a turbo cache natively it passes need_f16_K/V = false to
            // launch_fattn, so the scratch reserved here is never written: ggml_nelements(K) +
            // ggml_nelements(V) halves, about 890 MiB of compute buffer at 256K, provably dead.
            // Reclaiming it with
            //     need_f16_K = !ggml_cuda_fattn_turbo_reads_native(dst, true);
            // works and frees exactly that much (CUDA0 compute buffer 1834 -> 945 MiB), but it
            // COSTS THROUGHPUT ON EVERY KV TYPE. Bisected against clean HEAD in one session,
            // pp2048 / tg64 @ d131072, r=3:
            //
            //                              turbo4p pp    turbo4p tg    q8_0 pp     q8_0 tg
            //   HEAD                        1419.46        44.46       1406.85      41.46
            //   these changes + reclaim     1378.51        42.71       1350.85      39.78   -4%
            //   these changes, no reclaim   1450.04        44.77       1423.59      41.69   +2%
            //
            // Shrinking the buffer moves every allocation that follows it, and q8_0 - which never
            // enters the turbo predicate at all - loses the same 4%, so the cost is allocation
            // placement rather than anything in the FA path. 890 MiB is not worth 4% here.
            //
            // Revisit only WITH a measurement: the memory is real and would buy roughly one more
            // slot at 256K, so if VRAM ever becomes the binding constraint the trade may flip.
            //
            // [TAG_TURBO_FA_RECLAIM] Default is now the reclaim path: a natively-read turbo cache reserves
            // no F16 scratch (about 1 GiB at 256K on Qwen3.8-27B). TURBO_FA_RECLAIM=0 restores the
            // over-reservation if the placement cost above ever shows up again.
            {
                static const bool reclaim = [] {
                    const char * e = getenv("TURBO_FA_RECLAIM");
                    return !(e && e[0] == '0');
                }();
                need_f16_K = reclaim ? !ggml_cuda_fattn_turbo_reads_native(dst, true) : true;
                need_f16_V = need_f16_K;
            }
            break;
        case BEST_FATTN_KERNEL_VEC: {
            const bool f16_fallback = ggml_cuda_get_fattn_vec_case(Q->ne[0], K->type, V->type) == nullptr;
            need_f16_K = K->type == GGML_TYPE_F32 || f16_fallback;
            need_f16_V = V->type == GGML_TYPE_F32 || f16_fallback;
        } break;
        case BEST_FATTN_KERNEL_NONE:
            break;
    }

    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, need_f16_K, need_f16_V);

    return f16_extra.end - (uintptr_t) dst->data;
}

// [TAG_4C_POSMASK_MS] Explicit F16 mask from a multi-sequence positional one (fattn-common.cuh): one row per query,
// 0 where the cell is visible and -INF elsewhere, the values llama_kv_cache::set_input_kq_mask writes on the host.
// For the kernels that do not read sequence sets (VEC, TILE) and for every kernel under GGML_CUDA_FA_POS_MS=0.
// Rows n_q .. n_rows-1 are padding, all -INF (see ggml_cuda_flash_attn_ext_pos_ms_explicit); grid y strides the rows.
static __global__ void flash_attn_pos_ms_to_mask(
        const int32_t * __restrict__ kv_pos, const int32_t * __restrict__ kv_seq,
        const int32_t * __restrict__ q_pos,  const int32_t * __restrict__ q_seq,
        half * __restrict__ mask, const int n_kv, const int n_q, const int n_rows, const int64_t s1) {
    ggml_cuda_pdl_sync();
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_kv) {
        return;
    }
    const int      kp = kv_pos[i];
    const uint32_t ks = (uint32_t) kv_seq[i];
    for (int j = blockIdx.y; j < n_rows; j += gridDim.y) {
        const bool vis = j < n_q && kp >= 0 && kp <= q_pos[j] && (ks & (uint32_t) q_seq[j]) != 0;
        mask[(int64_t) j*s1 + i] = __float2half(vis ? 0.0f : -INFINITY);
    }
}

// [TAG_4C_POSMASK_MS] GGML_CUDA_FA_POS_MS=0: the MMA kernels no longer read sequence sets either; every multi-sequence
// positional mask becomes an explicit mask on the GPU (the host still saves the fill, the upload and the compute buffer).
// A/B for the kernel side on its own; LLAMA_KQ_MASK_POS_MS=0 turns the whole path off on the host.
static bool ggml_cuda_fa_pos_ms_native_on() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_FA_POS_MS");
        return !(e && e[0] == '0');
    }();
    return on;
}

// [TAG_4C_POSMASK_MS] Runs dst with the equivalent explicit mask: the same node with src[3] = the materialised mask and
// no positional vectors, so kernel selection, scratch and launch are exactly those of an explicit-mask FA. Rows are
// padded to 16 bytes so the GQA-packing stride test the router applies to a mask passes, as it does with no mask.
// The buffer holds n_q rounded up to 64 rows, the padding all -INF: the mask-based KV bounds scan of launch_fattn
// (flash_attn_mask_to_KV_max) reads ncols1 whole rows per query tile, i.e. up to GGML_PAD(n_q, ncols1) rows, and
// ncols1 <= 64 divides 64. A pool buffer of exactly n_q rows would let that scan read past its end (at 262K cells one
// row is 512 KiB, so possibly past the pool mapping). -INF rows never stop a tile from being skipped, so the bounds, and
// every kernel (they index rows modulo n_q), are unchanged.
static void ggml_cuda_flash_attn_ext_pos_ms_explicit(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q      = dst->src[0];
    const ggml_tensor * K      = dst->src[1];
    const ggml_tensor * kv_pos = dst->src[5];
    const ggml_tensor * q_pos  = dst->src[6];
    GGML_ASSERT(dst->src[3] == nullptr && kv_pos->ne[1] == 2 && q_pos->ne[1] == 2);
    GGML_ASSERT(ggml_is_contiguous(kv_pos) && ggml_is_contiguous(q_pos));

    const int64_t n_kv   = K->ne[1];
    const int64_t n_q    = Q->ne[1];
    const int64_t n_rows = GGML_PAD(n_q, 64);
    GGML_ASSERT(kv_pos->ne[0] == n_kv && q_pos->ne[0] == n_q && n_rows <= INT32_MAX && n_kv <= INT32_MAX);
    const int64_t s1 = GGML_PAD(n_kv, 8);

    ggml_cuda_pool_alloc<half> mask_buf(ctx.pool(), s1*n_rows);
    {
        const dim3 block_dim(256, 1, 1);
        const dim3 block_nums((unsigned) ((n_kv + 255) / 256), (unsigned) std::min<int64_t>(n_rows, 65535), 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dim, 0, ctx.stream());
        ggml_cuda_kernel_launch(flash_attn_pos_ms_to_mask, launch_params,
            (const int32_t *) kv_pos->data, (const int32_t *) ((const char *) kv_pos->data + kv_pos->nb[1]),
            (const int32_t *) q_pos->data,  (const int32_t *) ((const char *) q_pos->data  + q_pos->nb[1]),
            mask_buf.ptr, (int) n_kv, (int) n_q, (int) n_rows, s1);
    }

    ggml_tensor mask = {};
    mask.type  = GGML_TYPE_F16;
    mask.ne[0] = n_kv;
    mask.ne[1] = n_q;
    mask.ne[2] = 1;
    mask.ne[3] = 1;
    mask.nb[0] = sizeof(half);
    mask.nb[1] = s1*sizeof(half);
    mask.nb[2] = mask.nb[1]*n_rows;
    mask.nb[3] = mask.nb[2];
    mask.data  = mask_buf.ptr;

    ggml_tensor d = *dst;
    d.src[3] = &mask;
    d.src[5] = nullptr;
    d.src[6] = nullptr;
    ggml_cuda_flash_attn_ext(ctx, &d);
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    // Diagnostic: report ONCE per (kernel, K type, decode/prefill) which FA kernel runs.
    // Decode with a turbo KV cache collapses linearly with context (23 -> 4 tok/s from
    // 0.5K to 80K on Qwen3.8-Flash-Next) while q8_0 stays flat, pointing at the MMA path's
    // full-cache F16 materialisation (fattn-common.cuh: to_fp16 over ggml_nelements(K) on
    // EVERY call). This says which path each type actually takes. TURBO_PATH_PROBE=0 silences.
    // Selected ONCE. This used to run here for the probe and again in the switch below,
    // so every attention op paid two full passes of a function that queries device info
    // and walks four tensors by four dimensions.
    const best_fattn_kernel kprobe = ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst);
    // [TAG_4C_POSMASK_MS] Only the MMA kernels (f16/turbo and turbot) read the sequence sets of a multi-sequence
    // positional mask. Any other choice would see positions alone and attend across sequences, so it gets the
    // equivalent explicit mask instead (and so does every kernel under GGML_CUDA_FA_POS_MS=0).
    if (ggml_cuda_fattn_pos_rows(dst) == 2 && (kprobe != BEST_FATTN_KERNEL_MMA_F16 || !ggml_cuda_fa_pos_ms_native_on())) {
        ggml_cuda_flash_attn_ext_pos_ms_explicit(ctx, dst);
        return;
    }
    {
        static std::set<int> seen;
        const ggml_tensor * Kp = dst->src[1];
        const ggml_tensor * Qp = dst->src[0];
        // Key on the ACTUAL Q width, not a narrow/wide bit. The old key collapsed every
        // Q->ne[1] > 2 call into a single entry, so a speculative batch and a prefill
        // ubatch were indistinguishable - which is what hid whether speculative decode
        // clears the turbo_ok Q <= 32 gate in fattn-mma-f16.cuh.
        const int qw  = (int) Qp->ne[1];
        const int key = ((int) kprobe << 20) | ((int) Kp->type << 12) | (qw < 4095 ? qw : 4095);
        // Opt-IN (was opt-out, so release builds paid the getenv, the std::set lookup and
        // a second full ggml_cuda_get_best_fattn_kernel on every attention op).
        static const bool probe_on = [] {
            const char * e = getenv("TURBO_PATH_PROBE");
            return e && e[0] == '1';
        }();
        if (probe_on && seen.insert(key).second) {
            const char * kn = kprobe == BEST_FATTN_KERNEL_VEC     ? "VEC (reads KV natively)"        :
                              kprobe == BEST_FATTN_KERNEL_MMA_F16 ? "MMA_F16 (full-cache F16 copy)" :
                              kprobe == BEST_FATTN_KERNEL_TILE    ? "TILE (full-cache F16 copy)"    : "NONE";
            // Mirrors [TAG_TURBO_MMA_NATIVE] in fattn-mma-f16.cuh. If this reports
            // native=0 on an MMA_F16 call with a turbo4 cache, that call is paying a
            // full-cache F16 materialisation (to_fp16 over ggml_nelements(K)).
            // [TAG_TURBO5P512_MMA] asks the launch predicate itself. The hand copy that was here
            // knew only turbo4 and turbo4p, so it reported native=0 for every turbo5p and
            // turbo5p512 call.
            const bool native = ggml_cuda_fattn_turbo_reads_native(dst);
            fprintf(stderr, "turbo-probe: FA kernel = %s | K=%s Q->ne[1]=%d n_kv=%d kq_stride_ok=%d gqa=%d native=%d\n",
                    kn, ggml_type_name(Kp->type), (int) Qp->ne[1], (int) Kp->ne[1],
                    (int) (Kp->ne[1] % FATTN_KQ_STRIDE == 0),
                    (int) (Qp->ne[2] / Kp->ne[2]),
                    kprobe == BEST_FATTN_KERNEL_MMA_F16 ? (native ? 1 : 0) : -1);
            fflush(stderr);
        }
    }
    switch (kprobe) {
        case BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("fatal error");
        case BEST_FATTN_KERNEL_TILE:
            ggml_cuda_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC:
            ggml_cuda_flash_attn_ext_vec(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;
    }
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    // [TAG_FA_POS_MASK] positional mask: I32 cell/query positions, no explicit mask, single stream
    if (dst->src[5] != nullptr) {
        if (dst->src[3] != nullptr || dst->src[6] == nullptr || dst->src[5]->type != GGML_TYPE_I32 || dst->src[6]->type != GGML_TYPE_I32 ||
            dst->src[0]->ne[3] != 1 || !ggml_is_contiguous(dst->src[5]) || !ggml_is_contiguous(dst->src[6])) {
            return false;
        }
        // [TAG_4C_POSMASK_MS] one row (positions) or two (positions + sequence sets), the same on both vectors
        const ggml_tensor * kvp = dst->src[5];
        const ggml_tensor * qp  = dst->src[6];
        if (kvp->ne[1] != qp->ne[1] || (kvp->ne[1] != 1 && kvp->ne[1] != 2) || kvp->ne[2] != 1 || kvp->ne[3] != 1 ||
            qp->ne[2] != 1 || qp->ne[3] != 1) {
            return false;
        }
    }
    // [TAG_TURBOT] a turbot K/V needs its young pool, its granule table and valid layout params (SPEC 5.2, 7.6); the
    // routing rule itself lives in ggml_cuda_get_best_fattn_kernel
    if (ggml_turbot_is_type(dst->src[1]->type) || ggml_turbot_is_type(dst->src[2]->type)) {
        const ggml_tensor * pool = dst->src[7];
        const ggml_tensor * gtab = dst->src[8];
        if (pool == nullptr || gtab == nullptr || pool->type != GGML_TYPE_I8 || gtab->type != GGML_TYPE_I32 ||
                !ggml_is_contiguous(gtab) || gtab->ne[0]*GGML_TURBOT_GRANULE < dst->src[1]->ne[1]) {
            return false;
        }
        ggml_turbot_op_params tp;
        ggml_turbot_layer     tl;
        if (!ggml_turbot_op_params_get(dst, &tp) || tp.side != GGML_TURBOT_SIDE_BOTH || !ggml_turbot_layer_from_op_params(&tp, &tl)) {
            return false;
        }
        if (!ggml_turbot_is_type(dst->src[1]->type) || !ggml_turbot_is_type(dst->src[2]->type) ||
                ggml_turbot_s_of_type(dst->src[1]->type) != tl.k.s || ggml_turbot_s_of_type(dst->src[2]->type) != tl.v.s ||
                pool->ne[0] != (int64_t) tl.pool_row_bytes || pool->nb[0] != 1 ||
                dst->src[1]->nb[1] != (size_t) tl.k.base_row_bytes || dst->src[2]->nb[1] != (size_t) tl.v.base_row_bytes) {
            return false;
        }
        // [TAG_TURBOT_ANY_ROUTE] the K/V views must have the geometry the op params name (flags 0 is 4 x 256, the only
        // one GGML_TURBOT_ANY=0 accepts). The nb[1] == base_row_bytes cell stride check above holds for every geometry.
        if (!ggml_turbot_geom_valid(tp.flags) || (tp.flags != 0 && !ggml_cuda_turbot_any_on())) {
            return false;
        }
        const int64_t geom_hd = ggml_turbot_geom_head_dim(tp.flags);
        const int64_t geom_nh = ggml_turbot_geom_n_head(tp.flags);
        if (dst->src[0]->ne[0] != geom_hd || dst->src[1]->ne[0] != geom_hd || dst->src[2]->ne[0] != geom_hd ||
                dst->src[1]->ne[2] != geom_nh || dst->src[2]->ne[2] != geom_nh) {
            return false;
        }
    }
    return ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}

// [TAG_TURBOT_ANY_RESOLVE] The device answer of the KV resolver (turbot-set-rows.cuh): the shape-only part of the routing
// above, i.e. what ggml_cuda_get_best_fattn_kernel and ggml_cuda_flash_attn_ext_supported accept for a turbot layer of
// that geometry, and what ggml_cuda_turbot_set_rows_supported accepts for its rows. 4 x 256 (flags 0) needs Turing+ MMA
// exactly as the Qwen predicate does; the other geometries also need GGML_TURBOT_ANY on and, at head dim 128, the D = 128
// instances (GGML_CUDA_FA_TURBOT_D128).
bool ggml_cuda_turbot_geometry_supported(const int device, const int head_dim, const int n_head_kv) {
    const int flags = ggml_turbot_geom_flags(head_dim, n_head_kv);
    if (flags < 0) {
        return false;
    }
    if (flags != 0 && !ggml_cuda_turbot_any_on()) {
        return false;
    }
    if (head_dim == 128 && !GGML_CUDA_FA_TURBOT_D128) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[device].cc;
    return turing_mma_available(cc);
}
