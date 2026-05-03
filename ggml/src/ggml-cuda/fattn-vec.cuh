#include "common.cuh"
#include "fattn-common.cuh"

static int ggml_cuda_fattn_vec_get_nthreads_host(const int cc) {
    return 128;
    GGML_UNUSED(cc);
}

static constexpr __device__ int ggml_cuda_fattn_vec_get_nthreads_device() {
    return 128;
}

// Currently llvm with the amdgcn target does not support unrolling loops
// that contain a break that can not be resolved at compile time.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
template<int D, int ncols, ggml_type type_K, ggml_type type_V, bool use_logit_softcap> // D == head size
// PERF: bumped min_blocks_per_sm 2 → 4. With nthreads_V=16 fix the VKQ register
// footprint per thread halved (4 entries vs 8), freeing room for higher occupancy.
// Memory-latency-bound at depth — more in-flight blocks hide DRAM/L2 stalls better.
__launch_bounds__(ggml_cuda_fattn_vec_get_nthreads_device(), 4)
static __global__ void flash_attn_ext_vec(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
#ifdef FLASH_ATTN_AVAILABLE

    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(D == 128 || D == 256)) {
        GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_head_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        NO_DEVICE_CODE;
        return;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;
    // Will be bounded below to fit Q_reg sizing (D/2/nthreads_KQ) once nthreads_KQ is known.

#ifdef GGML_USE_HIP
#ifdef RDNA
    constexpr int nthreads_KQ_q = 2;
#else
    constexpr int nthreads_KQ_q = 4;
#endif // RDNA
    constexpr int nthreads_V_q  = (D/4 < 32 ? D/4 : 32);
#else
    constexpr int nthreads_KQ_q = (D/4 < 32 ? D/4 : 32);
    constexpr int nthreads_V_q  = (D/4 < 32 ? D/4 : 32);
#endif // GGML_USE_HIP

    constexpr int nthreads    = ggml_cuda_fattn_vec_get_nthreads_device();
    // turbo K uses float-Q path (vec_dot_fattn_vec_KQ_turbo*_0 ignores Q_q8) but should
    // get q8's nthreads_KQ_q (=32 for D=128) for parallelism. Tested int8/__dp4a path
    // proved slower at depth (constant-memory serialization on divergent lookups).
    constexpr bool type_K_is_turbo = (type_K == GGML_TYPE_TURBO3_0 || type_K == GGML_TYPE_TURBO2_0 || type_K == GGML_TYPE_TURBO4_0);
    constexpr bool K_is_unquantized = (type_K == GGML_TYPE_F16 || type_K == GGML_TYPE_BF16 || type_K_is_turbo);
    constexpr bool V_is_unquantized = (type_V == GGML_TYPE_F16 || type_V == GGML_TYPE_BF16 || type_V == GGML_TYPE_TURBO3_0 || type_V == GGML_TYPE_TURBO2_0 || type_V == GGML_TYPE_TURBO4_0);
    // PERF (turbo K alignment fix attempt): turbo K uses 16 threads/K (not 32) so
    // cpy_ne=4 fits and byte_base=tid*4 is uniformly 4-byte aligned → single LDG.E.32
    // qs load instead of <2,2> short load. Two 16-thread groups per warp process 2 K
    // positions in parallel per i_KQ_0 iter. Trade-off: doubles per-thread Q_reg size.
    constexpr int nthreads_KQ = type_K_is_turbo ? (nthreads_KQ_q/2) : (K_is_unquantized ? 128 / cpy_nb : nthreads_KQ_q);
    // PERF V: turbo previously routed through nthreads_V=nthreads_V_q (=32 for D=128),
    // giving V_cols_per_iter = WARP_SIZE/nthreads_V = 1 — only ONE V position per warp
    // iteration. f16/bf16 use nthreads_V=128/cpy_nb=8 → V_cols_per_iter=4, processing
    // 4 V positions per warp iter. Route turbo through the same V dispatch as f16/bf16
    // (with V_rows_per_thread=2*cpy_ne and ne=8 dequant support added in fattn-common.cuh).
    constexpr bool type_V_is_turbo = (type_V == GGML_TYPE_TURBO3_0 || type_V == GGML_TYPE_TURBO2_0 || type_V == GGML_TYPE_TURBO4_0);
    // PERF (turbo V correctness + speed): use 16 threads/V (not 8) so each lane holds
    // ONE centroid (matches turbo4's 16-entry table). Eliminates the broken half-pair
    // shfl pattern (~50% wrong-half lookups) AND the half2-packing overhead — single
    // shfl per element with CORRECT semantics. V_cols_per_iter halves to 2, but the
    // total V positions per warp per outer K step is unchanged (just more iters of k0).
    constexpr int nthreads_V  = type_V_is_turbo ? 16 : (V_is_unquantized ? 128 / cpy_nb : nthreads_V_q);

    static_assert(WARP_SIZE % nthreads_KQ == 0, "bad nthreads_K");
    static_assert(WARP_SIZE % nthreads_V  == 0, "bad nthreads_V");

    // CORRECTNESS: Q_reg is sized [(D/2)/nthreads_KQ]; the Q load + K KQ-dot loops
    // both use cpy_ne as inner stride. When nthreads_KQ * cpy_ne > D/2 (e.g. turbo
    // with nthreads_KQ=32, cpy_ne=4, D=128 → 128 > 64), the loop writes/reads OOB
    // into Q_reg (sized 2) at indices 0..3. Cap cpy_ne_KQ to fit per-thread element
    // count exactly: D/(2*nthreads_KQ).
    constexpr int cpy_ne_KQ = (nthreads_KQ * cpy_ne > D/2) ? D/(2*nthreads_KQ) : cpy_ne;
    static_assert(cpy_ne_KQ >= 1, "cpy_ne_KQ must be at least 1");

    // V_rows_per_thread = 2*cpy_ne for ALL V_is_unquantized types. Turbo previously was
    // capped to 4 (a leftover from when nthreads_V was 32 — 32*8/2 = 128 > D/2 = 64
    // would have OOB'd). After my V dispatch fix that makes nthreads_V = 8 for turbo,
    // 8*8/2 = 32 fits cleanly into D/2 = 64 (loop runs 2 iters, covering full D=128).
    // VKQ register array sized D/2/nthreads_V = 8 — all 8 indices used, no OOB.
    // Bumps V dequant per-thread element count from 4 → 8, doubling V throughput.
    constexpr int V_rows_per_thread = V_is_unquantized ? 2*cpy_ne : 4;
    (void)type_V_is_turbo;
    constexpr int V_cols_per_iter   = WARP_SIZE / nthreads_V;

    constexpr vec_dot_KQ_t vec_dot_KQ = get_vec_dot_KQ<type_K, D, nthreads_KQ>();
    constexpr bool Q_q8_1 = !K_is_unquantized;
#ifdef V_DOT2_F32_F16_AVAILABLE
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, half,  V_rows_per_thread>();
#else
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, float, V_rows_per_thread>();
#endif // V_DOT2_F32_F16_AVAILABLE

    const int ic0 = blockIdx.x * ncols; // Index of the Q/QKV column to work on.

    const int sequence = blockIdx.z / ne02;
    const int head = blockIdx.z - sequence*ne02;
    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    Q += nb03*sequence + nb02* head              + nb01*ic0;
    K += nb13*sequence + nb12*(head / gqa_ratio);
    V += nb23*sequence + nb22*(head / gqa_ratio);

    const half * maskh  = (const half  *) (mask + nb33*(sequence % ne33) + nb31*ic0);

    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    static_assert(D % (2*WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");
    constexpr int nwarps = nthreads / WARP_SIZE;
    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < nthreads);

    constexpr int ne_KQ      = ncols*D;
    constexpr int ne_combine = nwarps*V_cols_per_iter*D;
#ifdef V_DOT2_F32_F16_AVAILABLE
    half2            VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};
    __shared__ half   KQ[ne_KQ > ne_combine ? ne_KQ : ne_combine];
#else
    float2           VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};
    __shared__ float  KQ[ne_KQ > ne_combine ? ne_KQ : ne_combine];
#endif // V_DOT2_F32_F16_AVAILABLE

    // Sparse V: skip V dequant for positions with negligible attention weights.
    // 5e-3 — skip positions with weight < 0.5%. After softmax at depth 50K, peak
    // attention concentrates on a few positions; the long tail averages 1/N=2e-5 each
    // so most are below 5e-3. Aggressive but quality risk small (cumulative skipped
    // mass is bounded by how many positions sit in the 5e-3..0 range, typically <30%).
    // Sparse-V is fully disabled (preserves full attention quality at long context).
    // The constexpr enable flag below lets the compiler dead-code-eliminate both the
    // per-K-position dominated check and the sparse_v_threshold conversion. Previously
    // we set threshold=-1.0f which made the check always return false, but the compare
    // and branch still executed in the kernel — measurable overhead at 50K positions.
    constexpr bool sparse_v_enabled = false;
    constexpr float sparse_v_threshold_f = 0.0f;
#ifdef V_DOT2_F32_F16_AVAILABLE
    const     half  sparse_v_threshold_h = __float2half(sparse_v_threshold_f);
#endif
    (void)sparse_v_threshold_f; (void)sparse_v_enabled;

    float KQ_max[ncols];
    float KQ_sum[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ_max[j] = -FLT_MAX/2.0f;
        KQ_sum[j] = 0.0f;
    }

    // Convert Q to float2 (f16 K) or q8_1 (quantized K) and store in registers:
#ifdef V_DOT2_F32_F16_AVAILABLE
    half2  Q_reg[ncols][(D/2)/nthreads_KQ]; // Will be initialized completely.
#else
    __align__(16) float2 Q_reg[ncols][(D/2)/nthreads_KQ] = {{{0.0f, 0.0f}}}; // May be only partially initialized.
#endif // V_DOT2_F32_F16_AVAILABLE
    int    Q_i32[ncols][1 > D/(sizeof(int)*nthreads_KQ) ? 1 : D/(sizeof(int)*nthreads_KQ)];
    float2  Q_ds[ncols][1 > D/(sizeof(int)*nthreads_KQ) ? 1 : D/(sizeof(int)*nthreads_KQ)];
    if constexpr (Q_q8_1) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            int    * tmp_q_i32 = (int    *) &KQ[j*D];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));

            // Set memory to zero if out of bounds:
            if (ncols > 1 && ic0 + j >= int(ne01.z)) {
#pragma unroll
                for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += WARP_SIZE) {
                    const int i = i0 + threadIdx.x;

                    if (i0 + WARP_SIZE <= int(D/sizeof(int)) || i < int(D/sizeof(int))) {
                        tmp_q_i32[i] = 0;
                    }
                }
                if (threadIdx.x < D/QK8_1) {
                    tmp_q_ds[threadIdx.x] = make_float2(0.0f, 0.0f);
                }
            } else {
                const float * Q_f = (const float *) (Q + j*nb01);
                constexpr int nthreads_quantize = D/sizeof(int) < WARP_SIZE ? D/sizeof(int) : WARP_SIZE;
#pragma unroll
                for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_quantize) {
                    quantize_q8_1_to_shared<float2, nthreads_quantize>
                        (Q_f + i0*sizeof(int), scale, tmp_q_i32 + i0, tmp_q_ds + i0/QI8_1);
                }
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            int    * tmp_q_i32 = (int    *) &KQ[j*D];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));

#pragma unroll
            for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_KQ) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ);

                Q_i32[j][i0/nthreads_KQ] = tmp_q_i32[i];
                Q_ds[j][i0/nthreads_KQ]  = tmp_q_ds[i/QI8_1];
            }
        }

        __syncthreads();
    } else {
#ifdef V_DOT2_F32_F16_AVAILABLE
        const half2 scale_h2 = make_half2(scale, scale);
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2 * Q_j = (const float2 *) (Q + j*nb01);
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne_KQ) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ)*cpy_ne_KQ;

                __align__(16) float2 tmp[cpy_ne_KQ] = {{0.0f, 0.0f}};
                if (ncols == 1 || ic0 + j < int(ne01.z)) {
                    if constexpr (cpy_ne_KQ >= 2) {
                        ggml_cuda_memcpy_1<cpy_ne_KQ*4>(tmp,                 &Q_j[i]);
                        if constexpr (cpy_ne_KQ >= 4) {
                            ggml_cuda_memcpy_1<cpy_ne_KQ*4>(tmp + cpy_ne_KQ/2, &Q_j[i + cpy_ne_KQ/2]);
                        }
                    } else {
                        tmp[0] = Q_j[i];
                    }
                }
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne_KQ; ++i1) {
                    Q_reg[j][i0/nthreads_KQ + i1] = make_half2(tmp[i1].x, tmp[i1].y);
                }
            }
#pragma unroll
            for (int k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k] *= scale_h2;
            }
        }
#else
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2 * Q_j = (const float2 *) (Q + j*nb01);
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne_KQ) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ)*cpy_ne_KQ;
                if (ncols == 1 || ic0 + j < int(ne01.z)) {
                    if constexpr (cpy_ne_KQ >= 2) {
                        ggml_cuda_memcpy_1<cpy_ne_KQ*4>(&Q_reg[j][i0/nthreads_KQ], &Q_j[i]);
                        if constexpr (cpy_ne_KQ >= 4) {
                            ggml_cuda_memcpy_1<cpy_ne_KQ*4>(&Q_reg[j][i0/nthreads_KQ + cpy_ne_KQ/2], &Q_j[i + cpy_ne_KQ/2]);
                        }
                    } else {
                        Q_reg[j][i0/nthreads_KQ] = Q_j[i];
                    }
                }
            }
#pragma unroll
            for (int k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k].x *= scale;
                Q_reg[j][k].y *= scale;
            }
        }
#endif // V_DOT2_F32_F16_AVAILABLE
    }

    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
    K     += blockIdx.y*nthreads * nb11;
    V     += blockIdx.y*nthreads * nb21;
    maskh += blockIdx.y*nthreads;
    for (int k_VKQ_0 = blockIdx.y*nthreads; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*nthreads,
             // Increment pointers after each loop:
             K += gridDim.y*nthreads*nb11, V += gridDim.y*nthreads*nb21, maskh += gridDim.y*nthreads) {

        // Calculate KQ tile and keep track of new maximum KQ values:
        float KQ_reg[ncols]; // KQ in registers.

        float KQ_max_new[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            KQ_max_new[j] = KQ_max[j];
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
            const int i_KQ = threadIdx.y*WARP_SIZE + (nthreads_KQ == WARP_SIZE ? 0 : (threadIdx.x & ~(nthreads_KQ-1))) + i_KQ_0;

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                float sum = vec_dot_KQ(K + i_KQ*nb11, Q_reg[j], Q_i32[j], Q_ds[j]);
                sum = warp_reduce_sum<nthreads_KQ>(sum);

                if (use_logit_softcap) {
                    sum = logit_softcap*tanhf(sum);
                }

                if (mask && (ncols == 1 || ic0 + j < int(ne01.z))) {
                    sum += slope*__half2float(maskh[j*ne11 + i_KQ]);
                }

                KQ_max_new[j] = fmaxf(KQ_max_new[j], sum + FATTN_KQ_MAX_OFFSET);

                if ((nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ) == uint32_t(i_KQ_0)) {
                    KQ_reg[j] = sum;
                }
            }
        }

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
#pragma unroll
            for (int offset = nthreads_KQ; offset < WARP_SIZE; offset <<= 1) {
                KQ_max_new[j] = fmaxf(KQ_max_new[j], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[j], offset, WARP_SIZE));
            }
            // PERF (depth-flattening): KQ_max grows monotonically and stabilizes after the
            // first few high-attention K positions are seen. After that, KQ_max_new == KQ_max
            // and KQ_max_scale = exp(0) = 1 — multiplying every prior VKQ by 1.0f for
            // thousands of remaining K positions is wasted work that scales with depth.
            // Branch around the rescale when the max didn't change (the common case at depth).
            const bool max_changed = (KQ_max_new[j] != KQ_max[j]);
            const float KQ_max_scale = max_changed ? expf(KQ_max[j] - KQ_max_new[j]) : 1.0f;
            KQ_max[j] = KQ_max_new[j];

            KQ_reg[j] = expf(KQ_reg[j] - KQ_max[j]);
            KQ_sum[j] = max_changed ? (KQ_sum[j]*KQ_max_scale + KQ_reg[j]) : (KQ_sum[j] + KQ_reg[j]);
            KQ[j*nthreads + tid] = KQ_reg[j];

            if (max_changed) {
#ifdef V_DOT2_F32_F16_AVAILABLE
                const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
                for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                    VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
                }
#else
#pragma unroll
                for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                    VKQ[j][i_VKQ_0/nthreads_V].x *= KQ_max_scale;
                    VKQ[j][i_VKQ_0/nthreads_V].y *= KQ_max_scale;
                }
#endif // V_DOT2_F32_F16_AVAILABLE
            }
        }

#ifndef GGML_USE_HIP
        __syncwarp();
#endif // GGML_USE_HIP

#pragma unroll
        for (int k0 = 0; k0 < WARP_SIZE; k0 += V_cols_per_iter) {
            const int k = threadIdx.y*WARP_SIZE + k0 + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V);

#ifdef V_DOT2_F32_F16_AVAILABLE
            half2 KQ_k[ncols];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                KQ_k[j] = __half2half2(KQ[j*nthreads + k]);
            }

            // Sparse V: skip V dequant if all attention weights for this position are negligible
            // (constexpr-gated so the entire check compiles away when disabled).
            if constexpr (sparse_v_enabled) {
                bool dominated = true;
#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    if (__hgt(__low2half(KQ_k[j]), sparse_v_threshold_h)) { dominated = false; break; }
                }
                if (dominated) { continue; }
            }

#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                half2 tmp[V_rows_per_thread/2];
                if constexpr (type_V == GGML_TYPE_BF16) {
                    float2 tmp_f[V_rows_per_thread/2];
                    dequantize_V(V + k*nb21, tmp_f,
                        2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread);
#pragma unroll
                    for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
                        tmp[i_VKQ_1] = __float22half2_rn(tmp_f[i_VKQ_1]);
                    }
                } else {
                    dequantize_V(V + k*nb21, tmp,
                        2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread);
                }
#pragma unroll
                for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1] += tmp[i_VKQ_1]*KQ_k[j];
                    }
                }
            }
#else
            float KQ_k[ncols];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                KQ_k[j] = KQ[j*nthreads + k];
            }

            // Sparse V: constexpr-gated, fully eliminated when disabled.
            if constexpr (sparse_v_enabled) {
                bool dominated = true;
#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    if (KQ_k[j] >= sparse_v_threshold_f) { dominated = false; break; }
                }
                if (dominated) { continue; }
            }

#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                float2 tmp[V_rows_per_thread/2];
                dequantize_V(V + k*nb21, tmp,
                    2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread);
#pragma unroll
                for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].x += tmp[i_VKQ_1].x*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].y += tmp[i_VKQ_1].y*KQ_k[j];
                    }
                }
            }
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    if (sinks && blockIdx.y == 0) {
        const float sink = ((const float *) sinks)[head];

#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            const float kqmax_new_j = fmaxf(sink, KQ_max[j]);
            const float KQ_max_scale = expf(KQ_max[j] - kqmax_new_j);
            KQ_max[j] = kqmax_new_j;

            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + (threadIdx.x == 0 ? expf(sink - KQ_max[j]) : 0.0f);

#ifdef V_DOT2_F32_F16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y *= KQ_max_scale;
            }
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    __shared__ float KQ_max_shared[ncols][WARP_SIZE];
    __shared__ float KQ_sum_shared[ncols][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.y == 0) {
            KQ_max_shared[j][threadIdx.x] = -FLT_MAX/2.0f;
            KQ_sum_shared[j][threadIdx.x] = 0.0f;
        }
    }

    __syncthreads();

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.x == 0) {
            KQ_max_shared[j][threadIdx.y] = KQ_max[j];
        }
    }
    __syncthreads();

#pragma unroll
    for (int j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        if (ncols > 1 && ic0 + j_VKQ >= int(ne01.z)) {
            break;
        }

        float kqmax_new = KQ_max_shared[j_VKQ][threadIdx.x];
        kqmax_new = warp_reduce_max(kqmax_new);
        const float kqmax_scale = expf(KQ_max[j_VKQ] - kqmax_new);
        KQ_max[j_VKQ] = kqmax_new;

#ifdef V_DOT2_F32_F16_AVAILABLE
        half2 * VKQ_tmp = (half2 *) KQ + threadIdx.y*(V_cols_per_iter*D/2)
            + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V)*(D/2);

        const half2 kqmax_scale_h2 = make_half2(kqmax_scale, kqmax_scale);
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V] *= kqmax_scale_h2;
        }
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            const int i_VKQ = i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*(V_rows_per_thread/2);

            ggml_cuda_memcpy_1<V_rows_per_thread*sizeof(half)>(VKQ_tmp + i_VKQ, &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
        }
#else
        float2 * VKQ_tmp = (float2 *) KQ + threadIdx.y*(V_cols_per_iter*D/2)
            + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V)*(D/2);

#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].x *= kqmax_scale;
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].y *= kqmax_scale;
        }
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            const int i_VKQ = i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*(V_rows_per_thread/2);

            ggml_cuda_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ,                       &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
            ggml_cuda_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ + V_rows_per_thread/4, &VKQ[j_VKQ][i_VKQ_0/nthreads_V + V_rows_per_thread/4]);
        }
#endif // V_DOT2_F32_F16_AVAILABLE

        KQ_sum[j_VKQ] *= kqmax_scale;
        KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);
        if (threadIdx.x == 0) {
            KQ_sum_shared[j_VKQ][threadIdx.y] = KQ_sum[j_VKQ];
        }

        __syncthreads();

        if (nthreads <= D || tid < D) {
            KQ_sum[j_VKQ] = KQ_sum_shared[j_VKQ][threadIdx.x];
            KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);

#pragma unroll
            for (int i0 = 0; i0 < D; i0 += nthreads) {
                float dst_val = 0;
#pragma unroll
                for (int w = 0; w < nwarps; ++w) {
#pragma unroll
                    for (int v = 0; v < V_cols_per_iter; ++v) {
                        dst_val += float(KQ[w*V_cols_per_iter*D + v*D + i0 + tid]);
                    }
                }
                if (gridDim.y == 1) {
                    dst_val /= KQ_sum[j_VKQ];
                }
                dst[(((sequence*int(ne01.z) + ic0 + j_VKQ)*ne02 + head)*gridDim.y + blockIdx.y)*D + i0 + tid] = dst_val;
            }
        }

        if (j_VKQ < ncols-1) {
            __syncthreads();
        }

    }

    if (gridDim.y != 1 && tid < ncols && (ncols == 1 || ic0 + tid < int(ne01.z))) {
        dst_meta[((sequence*int(ne01.z) + ic0 + tid)*ne02 + head)*gridDim.y + blockIdx.y] = make_float2(KQ_max[tid], KQ_sum[tid]);
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

template <int D, int cols_per_block, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
void ggml_cuda_flash_attn_ext_vec_case_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const int nthreads = ggml_cuda_fattn_vec_get_nthreads_host(cc);
    const int nwarps   = nthreads / WARP_SIZE;
    fattn_kernel_t fattn_kernel = flash_attn_ext_vec<D, cols_per_block, type_K, type_V, use_logit_softcap>;
    const bool need_f16_K = type_K == GGML_TYPE_F16;
    const bool need_f16_V = type_V == GGML_TYPE_F16;
    constexpr size_t nbytes_shared = 0;
    launch_fattn<D, cols_per_block, 1>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, false);
}

template <int D, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (Q->ne[1] == 1) {
        constexpr int cols_per_block = 1;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    constexpr int cols_per_block = 2;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    } else {
        constexpr bool use_logit_softcap = true;
        ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    }
}

#define DECL_FATTN_VEC_CASE(D, type_K, type_V)                              \
    template void ggml_cuda_flash_attn_ext_vec_case                         \
    <D, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

#define EXTERN_DECL_FATTN_VEC_CASES(D, type_K)             \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_F16);  \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_1); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_1); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q8_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_BF16); \

EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_BF16)

EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_BF16)

EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_BF16)

// TurboQuant3 — turbo3 K + turbo3 V (KV cache uses same type)
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0);

// Mixed turbo3/q8_0 KV cache types
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0);

extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_0);

// TurboQuant2 -- turbo2 K + turbo2 V
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0);

// Mixed turbo2/q8_0 KV cache types
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO2_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO2_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO2_0, GGML_TYPE_Q8_0);

extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_Q8_0, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_Q8_0, GGML_TYPE_TURBO2_0);

// Mixed turbo3/turbo2 KV cache types
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO2_0);

extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0);

// TurboQuant4 — turbo4 K + turbo4 V (KV cache uses same type)
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0);

// Mixed turbo4/q8_0 KV cache types
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0);

extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0);

// Mixed turbo4/turbo3 KV cache types
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0);

extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0);

// Mixed turbo4/turbo2 KV cache types
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0);

extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO4_0);
