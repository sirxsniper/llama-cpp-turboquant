#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn.cuh"

#include <set>

template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 8/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if constexpr (ncols2 <= 16) {
        if (Q->ne[1] <= 16/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if (Q->ne[1] <= 32/ncols2 || (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING) ||
            (GGML_CUDA_CC_IS_AMD(cc) && DKQ > 256)) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
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
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 128/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
}

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // Edge cases like no mask, ALiBi, unpadded K/V, or misaligned addresses for large data transfers
    //     are put into the template specialization without GQA optimizations.
    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
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
    const ggml_tensor * mask = dst->src[3];

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
            const bool use_gqa_opt = mask && max_bias == 0.0f;
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

                const bool use_gqa_opt = mask && max_bias == 0.0f;
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

            const bool use_gqa_opt = mask && max_bias == 0.0f;
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

#define FATTN_VEC_CASE(D, type_K, type_V)                                                                        \
    {                                                                                                            \
        const bool type_K_okay = K->type == (type_K) || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16); \
        const bool type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16); \
        if (Q->ne[0] == (D) && type_K_okay && type_V_okay) {                                                     \
            ggml_cuda_flash_attn_ext_vec_case<D, type_K, type_V>(ctx, dst);                                      \
            return;                                                                                              \
        }                                                                                                        \
    }                                                                                                            \

#define FATTN_VEC_CASES_ALL_D(type_K, type_V) \
    FATTN_VEC_CASE( 64, type_K, type_V)       \
    FATTN_VEC_CASE(128, type_K, type_V)       \
    FATTN_VEC_CASE(256, type_K, type_V)       \

static void ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#ifdef GGML_CUDA_FA_ALL_QUANTS
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_F16)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q4_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q4_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q5_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q5_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q8_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_BF16)
#else
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_BF16)
#endif // GGML_CUDA_FA_ALL_QUANTS

    // TurboQuant3 KV cache types (always enabled)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0)

    // Mixed turbo3/q8_0 KV cache types
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0,     GGML_TYPE_TURBO3_0)

    // TurboQuant2 KV cache types (always enabled)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0)

    // Mixed turbo2/q8_0 KV cache types
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO2_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0,     GGML_TYPE_TURBO2_0)

    // Mixed turbo3/turbo2 KV cache types
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0)

    // TurboQuant4 KV cache types (always enabled)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0)

    // Mixed turbo4/q8_0 KV cache types
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0,     GGML_TYPE_TURBO4_0)

    // Mixed turbo4/turbo3 KV cache types
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0)

    // Mixed turbo4/turbo2 KV cache types
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO4_0)

    GGML_ABORT("fatal error");
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE    =   0,
    BEST_FATTN_KERNEL_TILE    = 200,
    BEST_FATTN_KERNEL_VEC     = 100,
    BEST_FATTN_KERNEL_MMA_F16 = 400,
};

static bool ggml_cuda_fattn_kv_type_supported(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
            return true;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
#ifndef GGML_CUDA_FA_ALL_QUANTS
            return false;
#endif // GGML_CUDA_FA_ALL_QUANTS
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_BF16:
        case GGML_TYPE_TURBO2_0:
        case GGML_TYPE_TURBO3_0:
        case GGML_TYPE_TURBO4_0:
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
    const ggml_tensor * mask  = dst->src[3];

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // The effective batch size for the kernel can be increased by gqa_ratio.
    // The kernel versions without this optimization are also used for ALiBi, if there is no mask, or if the KV cache is not padded,
    bool gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
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

#ifndef GGML_CUDA_FA_ALL_QUANTS
    if (K->type != V->type) {
        // Allow mixed turbo KV types (any combination of turbo2, turbo3, q8_0)
        auto is_turbo = [](ggml_type t) {
            return t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0 || t == GGML_TYPE_Q8_0;
        };
        if (!is_turbo(K->type) || !is_turbo(V->type)) {
            return BEST_FATTN_KERNEL_NONE;
        }
    }
#endif // GGML_CUDA_FA_ALL_QUANTS

    if (!ggml_cuda_fattn_kv_type_supported(K->type) || !ggml_cuda_fattn_kv_type_supported(V->type)) {
        return BEST_FATTN_KERNEL_NONE;
    }

    // turbo2/turbo3/turbo4 VEC kernels are only instantiated for D in {64, 128, 256}.
    if (K->type == GGML_TYPE_TURBO2_0 || K->type == GGML_TYPE_TURBO3_0 || K->type == GGML_TYPE_TURBO4_0) {
        if (K->ne[0] % 64 != 0) {
            return BEST_FATTN_KERNEL_NONE;
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
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 && !(gqa_ratio > 4 && K->ne[1] >= 8192)) {
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
                                         K->type == GGML_TYPE_TURBO4_0;
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

    // AMD WMMA is always faster than the tile kernel if the full tile width of 16 can be utilized.
    if ((amd_wmma_available(cc) && gqa_opt_applies && Q->ne[0] <= 128) && Q->ne[0] != 40 && Q->ne[0] != 72 && Q->ne[1] * gqa_ratio_eff > 8) {
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

    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    GGML_ASSERT(K != nullptr);
    GGML_ASSERT(V != nullptr);

    const best_fattn_kernel kernel = ggml_cuda_get_best_fattn_kernel(device, dst);

    bool need_f16_K = false;
    bool need_f16_V = false;

    switch (kernel) {
        case BEST_FATTN_KERNEL_TILE:
        case BEST_FATTN_KERNEL_MMA_F16:
            need_f16_K = true;
            need_f16_V = true;
            break;
        case BEST_FATTN_KERNEL_VEC:
            need_f16_K = K->type == GGML_TYPE_F32;
            need_f16_V = V->type == GGML_TYPE_F32;
            break;
        case BEST_FATTN_KERNEL_NONE:
            break;
    }

    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, need_f16_K, need_f16_V);

    return f16_extra.end - (uintptr_t) dst->data;
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
            const ggml_tensor * Vp = dst->src[2];
            const char * nenv = getenv("TURBO_MMA_NATIVE");
            const char * nmq  = getenv("TURBO_MMA_NATIVE_MAXQ");
            const int maxq_env = nmq ? atoi(nmq) : 0;
            const int maxq = (maxq_env >= 1 && maxq_env <= 4096) ? maxq_env : 32;
            const bool native = !(nenv && nenv[0] == '0') &&
                                Qp->ne[0] == 256 && Vp && Vp->ne[0] == 256 &&
                                Qp->ne[1] <= maxq &&
                                Kp->type == GGML_TYPE_TURBO4_0 && Vp->type == GGML_TYPE_TURBO4_0;
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
    return ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}
