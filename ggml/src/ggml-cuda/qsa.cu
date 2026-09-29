#include "qsa.cuh"

#include <algorithm>
#include <cstdint>
#include <cstring>

// [TAG_FN_QSA_FUSED] The qwen4exp QSA indexer (models/qwen4exp.cpp build_qsa_top_k) as two kernels.
//
// k_qsa_score: one warp per key block. It pools the r member rows of the raw key cache (member order, then 1/r), applies
// the rms_norm of norm.cu (rms_norm_f32<256>: the same per-warp sums and the same second-level sum, so the same bits),
// the norm weight and the rope_multi rotation of rope.cu, and scores the block against every query head of its query
// chunk (relu, heads summed in order, plus the block bias). The pooled key never leaves the warp: the unfused graph
// wrote the gathered rows ([D, n_kv] f32), four slices, their sums, the norm and the rope out to memory for every
// block on every token. Only the head dots differ from the graph (cuBLAS / mmf order), by float rounding.
//
// k_qsa_topk: v[c] = score[cell_blk[c]] + mask[c] is computed on the fly in every pass (no [n_kv, n_tokens] tensor),
// then the radix select of top-k.cu k_top_k_select (four 8-bit passes on an order-preserving key, an index select for
// ties on the k-th value, one compaction pass with one atomic per warp). A long row is split over up to 16 blocks, each
// selecting the top k of its chunk, and one block selects the top k of those candidates. The result is the top-k set
// with ties to the lowest cells, in compaction order.

// ------------------------------------------------------------------------------------------------------------------
// GGML_OP_QSA_SCORE

#define QSA_SCORE_NWARPS 8

struct qsa_rope_args {
    int   n_dims;
    int   sections[4];
    int   is_imrope;
    float theta_scale;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float corr0;
    float corr1;
};

// rope.cu rope_yarn_ramp / rope_yarn<forward = true>, the same expressions
static __device__ float qsa_rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

static __device__ void qsa_rope_yarn(
        const float theta_extrap, const float freq_scale, const float corr0, const float corr1, const int i0, const float ext_factor,
        float mscale, float & cos_theta, float & sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = qsa_rope_yarn_ramp(corr0, corr1, i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    cos_theta = cosf(theta) * mscale;
    sin_theta = sinf(theta) * mscale;
}

// element e of a key row, as get_rows dequantizes it (dequantize_q8_0: float(q) * float(d))
template <ggml_type type>
static __device__ __forceinline__ float qsa_load_k(const char * __restrict__ row, const int e) {
    if constexpr (type == GGML_TYPE_F32) {
        return ((const float *) row)[e];
    } else if constexpr (type == GGML_TYPE_F16) {
        return __half2float(((const half *) row)[e]);
    } else if constexpr (type == GGML_TYPE_BF16) {
        return __bfloat162float(((const nv_bfloat16 *) row)[e]);
    } else {
        static_assert(type == GGML_TYPE_Q8_0, "qsa: key type");
        const block_q8_0 * b = (const block_q8_0 *) row + e/QK8_0;
        const float d = __half2float(b->d);
        float v = b->qs[e % QK8_0];
        v *= d;
        return v;
    }
}

template <ggml_type type>
__launch_bounds__(QSA_SCORE_NWARPS*WARP_SIZE, 1)
static __global__ void k_qsa_score(
        const char    * __restrict__ k,
        const int32_t * __restrict__ blk_cells,
        const int32_t * __restrict__ blk_pos,
        const float   * __restrict__ norm_w,
        const float   * __restrict__ q,
        const float   * __restrict__ bias,
        float         * __restrict__ dst,
        const int D, const int r, const int n_blocks, const int n_tps, const int n_head, const int tq, const int max_pairs,
        const int64_t nbk1, const int64_t nbk2, const int64_t s_cells,
        const int64_t s_q1, const int64_t s_q2, const int64_t s_b1, const int64_t s_b2, const int64_t s_d1, const int64_t s_d2,
        const float inv_r, const float eps, const qsa_rope_args ra) {
    const int lane     = threadIdx.x % WARP_SIZE;
    const int warp     = threadIdx.x / WARP_SIZE;
    const int s        = blockIdx.z;
    const int n_stream = gridDim.z;
    const int t0       = blockIdx.y*tq;
    const int nt       = min(tq, n_tps - t0);
    const int np       = nt*n_head;
    const int npl      = D / WARP_SIZE;

    extern __shared__ float qsa_smem[];
    float * q_s   = qsa_smem;                     // [max_pairs][D + 1], pair p = token t0 + p / n_head, head p % n_head
    float * key_s = q_s + max_pairs*(D + 1);      // [nwarps][D]
    float * rel_s = key_s + QSA_SCORE_NWARPS*D;   // [nwarps][max_pairs]
    float * pow_s = rel_s + QSA_SCORE_NWARPS*max_pairs;   // [D/2]: theta_scale^i, the powf of rope.cu, once per block

    const int half_dims = ra.n_dims/2;

    for (int i = threadIdx.x; i < np*D; i += blockDim.x) {
        const int p = i / D;
        const int e = i - p*D;
        const int t = p / n_head;
        const int h = p - t*n_head;
        q_s[p*(D + 1) + e] = q[(int64_t) (s*n_tps + t0 + t)*s_q2 + (int64_t) h*s_q1 + e];
    }
    for (int i = threadIdx.x; i < half_dims; i += blockDim.x) {
        const int iw = 2*i;
        pow_s[i] = powf(ra.theta_scale, iw / 2.0f);
    }
    __syncthreads();

    float * key = key_s + warp*D;
    float * rel = rel_s + warp*max_pairs;

    const int64_t ne2       = (int64_t) n_blocks*n_stream;
    const int     sect_dims = ra.sections[0] + ra.sections[1] + ra.sections[2] + ra.sections[3];
    const int     sec_w     = ra.sections[1] + ra.sections[0];

    for (int b = blockIdx.x*QSA_SCORE_NWARPS + warp; b < n_blocks; b += gridDim.x*QSA_SCORE_NWARPS) {
        const int32_t * cells = blk_cells + s*s_cells + (int64_t) r*b;
        const char    * kbase = k + s*nbk2;

        // pooled key: lane holds elements lane + 32*j (rows of the unfused graph: slice 0, then + slice 1, ..., then * 1/r)
        float x[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            x[j] = 0.0f;
        }
        for (int i = 0; i < r; ++i) {
            const char * row = kbase + (int64_t) cells[i]*nbk1;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                if (j < npl) {
                    const float v = qsa_load_k<type>(row, lane + WARP_SIZE*j);
                    x[j] = i == 0 ? v : x[j] + v;
                }
            }
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            x[j] = x[j]*inv_r;
        }

        // rms_norm_f32<256>: thread 32j + lane sums element 32j + lane, each warp reduces, then one warp sums the warp sums
        float v = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (j < npl) {
                const float sj = warp_reduce_sum(x[j]*x[j]);
                v = lane == j ? sj : v;
            }
        }
        const float tot   = warp_reduce_sum(v);
        const float mean  = tot / D;
        const float scale = rsqrtf(mean + eps);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (j < npl) {
                key[lane + WARP_SIZE*j] = scale * x[j] * norm_w[lane + WARP_SIZE*j];
            }
        }
        __syncwarp();

        // rope_multi on channels [0, n_dims): pair (i, i + n_dims/2), angle from the block's positions
        const int64_t i2 = (int64_t) s*n_blocks + b;
        for (int i = lane; i < half_dims; i += WARP_SIZE) {
            const int iw     = 2*i;
            const int sector = (iw / 2) % sect_dims;

            float theta_base = 0.0;
            if (ra.is_imrope) {
                if (sector % 3 == 1 && sector < 3 * ra.sections[1]) {
                    theta_base = blk_pos[i2 + ne2 * 1] * pow_s[i];
                } else if (sector % 3 == 2 && sector < 3 * ra.sections[2]) {
                    theta_base = blk_pos[i2 + ne2 * 2] * pow_s[i];
                } else if (sector % 3 == 0 && sector < 3 * ra.sections[0]) {
                    theta_base = blk_pos[i2] * pow_s[i];
                } else {
                    theta_base = blk_pos[i2 + ne2 * 3] * pow_s[i];
                }
            } else {
                if (sector < ra.sections[0]) {
                    theta_base = blk_pos[i2] * pow_s[i];
                } else if (sector >= ra.sections[0] && sector < sec_w) {
                    theta_base = blk_pos[i2 + ne2 * 1] * pow_s[i];
                } else if (sector >= sec_w && sector < sec_w + ra.sections[2]) {
                    theta_base = blk_pos[i2 + ne2 * 2] * pow_s[i];
                } else if (sector >= sec_w + ra.sections[2]) {
                    theta_base = blk_pos[i2 + ne2 * 3] * pow_s[i];
                }
            }

            const float freq_factor = 1.0f;

            float cos_theta;
            float sin_theta;
            qsa_rope_yarn(theta_base/freq_factor, ra.freq_scale, ra.corr0, ra.corr1, iw, ra.ext_factor, ra.attn_factor, cos_theta, sin_theta);

            const float x0 = key[i];
            const float x1 = key[i + half_dims];

            key[i]             = x0*cos_theta - x1*sin_theta;
            key[i + half_dims] = x0*sin_theta + x1*cos_theta;
        }
        __syncwarp();

        // relu(q . key) per (token, head), then the heads of each token in order, plus the block bias. Up to 32 pairs
        // (decode, MTP verify) every lane takes D/32 elements of each dot and the warp sums them; above, a lane per pair.
        if (np <= WARP_SIZE) {
            float kr[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                kr[j] = j < npl ? key[lane + WARP_SIZE*j] : 0.0f;
            }
            for (int p = 0; p < np; ++p) {
                const float * qp = q_s + p*(D + 1) + lane;
                float d = 0.0f;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    if (j < npl) {
                        d += qp[WARP_SIZE*j]*kr[j];
                    }
                }
                d = warp_reduce_sum(d);
                if (lane == 0) {
                    rel[p] = fmaxf(d, 0.0f);
                }
            }
        } else {
            for (int p = lane; p < np; p += WARP_SIZE) {
                const float * qp = q_s + p*(D + 1);
                float d = 0.0f;
                for (int e = 0; e < D; ++e) {
                    d += qp[e]*key[e];
                }
                rel[p] = fmaxf(d, 0.0f);
            }
        }
        __syncwarp();
        for (int t = lane; t < nt; t += WARP_SIZE) {
            float summed = rel[t*n_head];
            for (int h = 1; h < n_head; ++h) {
                summed += rel[t*n_head + h];
            }
            if (bias) {
                summed += bias[s*s_b2 + (int64_t) (t0 + t)*s_b1 + b];
            }
            dst[s*s_d2 + (int64_t) (t0 + t)*s_d1 + b] = summed;
        }
        __syncwarp();
    }
}

static int qsa_score_max_pairs(const int64_t D) {
    return D <= 128 ? 64 : 32;
}

void ggml_cuda_op_qsa_score(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * k         = dst->src[0];
    const ggml_tensor * blk_cells = dst->src[1];
    const ggml_tensor * blk_pos   = dst->src[2];
    const ggml_tensor * norm_w    = dst->src[3];
    const ggml_tensor * q         = dst->src[4];
    const ggml_tensor * bias      = dst->src[5];

    const int32_t * op = (const int32_t *) dst->op_params;

    const int r          = op[0];
    const int n_dims     = op[2];
    const int mode       = op[3];
    const int n_ctx_orig = op[4];

    float eps, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    memcpy(&eps,         op +  1, sizeof(float));
    memcpy(&freq_base,   op +  5, sizeof(float));
    memcpy(&freq_scale,  op +  6, sizeof(float));
    memcpy(&ext_factor,  op +  7, sizeof(float));
    memcpy(&attn_factor, op +  8, sizeof(float));
    memcpy(&beta_fast,   op +  9, sizeof(float));
    memcpy(&beta_slow,   op + 10, sizeof(float));

    qsa_rope_args ra;
    memcpy(ra.sections, op + 11, sizeof(int)*4);
    ra.n_dims      = n_dims;
    ra.is_imrope   = mode == GGML_ROPE_TYPE_IMROPE ? 1 : 0;
    ra.theta_scale = powf(freq_base, -2.0f/n_dims);
    ra.freq_scale  = freq_scale;
    ra.ext_factor  = ext_factor;
    ra.attn_factor = attn_factor;
    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
    ra.corr0 = corr_dims[0];
    ra.corr1 = corr_dims[1];

    const int     D        = (int) k->ne[0];
    const int64_t n_stream = k->ne[2];
    const int     n_blocks = (int) dst->ne[0];
    const int     n_tps    = (int) dst->ne[1];
    const int     n_head   = (int) q->ne[1];

    GGML_ASSERT(D % WARP_SIZE == 0 && D <= 8*WARP_SIZE);
    GGML_ASSERT(ra.sections[0] + ra.sections[1] + ra.sections[2] + ra.sections[3] > 0);

    const int max_pairs = qsa_score_max_pairs(D);
    GGML_ASSERT(n_head <= max_pairs);
    const int tq = std::max(1, std::min(n_tps, max_pairs / n_head));
    const int ny = (n_tps + tq - 1) / tq;

    const int nsm     = ggml_cuda_info().devices[ctx.device].nsm;
    const int nx_need = (n_blocks + QSA_SCORE_NWARPS - 1) / QSA_SCORE_NWARPS;
    const int nx_fill = std::max<int>(1, (int) ((4*nsm + ny*n_stream - 1) / (ny*n_stream)));
    const int nx      = std::max(1, std::min(nx_need, nx_fill));

    const size_t smem = (size_t) (max_pairs*(D + 1) + QSA_SCORE_NWARPS*D + QSA_SCORE_NWARPS*max_pairs + D/2) * sizeof(float);
    GGML_ASSERT(smem <= 48*1024);

    const dim3 grid(nx, ny, (unsigned) n_stream);
    const dim3 block(QSA_SCORE_NWARPS*WARP_SIZE, 1, 1);

    const int64_t s_cells = blk_cells->nb[1] / sizeof(int32_t);
    const int64_t s_q1    = q->nb[1] / sizeof(float);
    const int64_t s_q2    = q->nb[2] / sizeof(float);
    const int64_t s_b1    = bias ? (int64_t) (bias->nb[1] / sizeof(float)) : 0;
    const int64_t s_b2    = bias ? (int64_t) (bias->nb[2] / sizeof(float)) : 0;
    const int64_t s_d1    = dst->nb[1] / sizeof(float);
    const int64_t s_d2    = dst->nb[2] / sizeof(float);
    const float   inv_r   = 1.0f/(float) r;

    cudaStream_t stream = ctx.stream();

#define QSA_SCORE_LAUNCH(T)                                                                                            \
    k_qsa_score<T><<<grid, block, smem, stream>>>(                                                                      \
        (const char *) k->data, (const int32_t *) blk_cells->data, (const int32_t *) blk_pos->data,                    \
        (const float *) norm_w->data, (const float *) q->data, bias ? (const float *) bias->data : nullptr,           \
        (float *) dst->data, D, r, n_blocks, n_tps, n_head, tq, max_pairs,                                              \
        (int64_t) k->nb[1], (int64_t) k->nb[2], s_cells, s_q1, s_q2, s_b1, s_b2, s_d1, s_d2, inv_r, eps, ra)

    switch (k->type) {
        case GGML_TYPE_F32:  QSA_SCORE_LAUNCH(GGML_TYPE_F32);  break;
        case GGML_TYPE_F16:  QSA_SCORE_LAUNCH(GGML_TYPE_F16);  break;
        case GGML_TYPE_BF16: QSA_SCORE_LAUNCH(GGML_TYPE_BF16); break;
        case GGML_TYPE_Q8_0: QSA_SCORE_LAUNCH(GGML_TYPE_Q8_0); break;
        default: GGML_ABORT("qsa_score: key type %s", ggml_type_name(k->type));
    }
#undef QSA_SCORE_LAUNCH
    CUDA_CHECK(cudaGetLastError());
}

// ------------------------------------------------------------------------------------------------------------------
// GGML_OP_QSA_TOPK

#define QSA_TOPK_THREADS    1024
#define QSA_TOPK_HSTRIDE    257    // one private 256-bin histogram per lane, 32 banks apart (top-k.cu [TAG_TOPK_SELECT_WARPHIST])
#define QSA_TOPK_MAX_CHUNKS 16

static __device__ __forceinline__ uint32_t qsa_topk_key(const float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// f(key, cell) for every cell in [c0, c1) of a row, key of v = score[cell_blk[c]] + mask[c]; four cells per step when
// the row (and c0) is aligned
template <bool mask_f16, bool vec4, typename F>
static __device__ __forceinline__ void qsa_topk_for_cells(
        const float * __restrict__ sc, const int32_t * __restrict__ cb, const half * __restrict__ mh, const float * __restrict__ mf,
        const int c0, const int c1, F && f) {
    const int tid = threadIdx.x;
    if constexpr (vec4) {
        const int i40 = c0/4;
        const int i41 = c1/4;
        for (int i4 = i40 + tid; i4 < i41; i4 += QSA_TOPK_THREADS) {
            const int4 bb = ((const int4 *) cb)[i4];
            float m0, m1, m2, m3;
            if constexpr (mask_f16) {
                const uint2 raw = ((const uint2 *) mh)[i4];
                half2 lo;
                half2 hi;
                memcpy(&lo, &raw.x, sizeof(lo));
                memcpy(&hi, &raw.y, sizeof(hi));
                m0 = __low2float(lo);
                m1 = __high2float(lo);
                m2 = __low2float(hi);
                m3 = __high2float(hi);
            } else {
                const float4 mm = ((const float4 *) mf)[i4];
                m0 = mm.x;
                m1 = mm.y;
                m2 = mm.z;
                m3 = mm.w;
            }
            const int c = 4*i4;
            f(qsa_topk_key(sc[bb.x] + m0), c + 0);
            f(qsa_topk_key(sc[bb.y] + m1), c + 1);
            f(qsa_topk_key(sc[bb.z] + m2), c + 2);
            f(qsa_topk_key(sc[bb.w] + m3), c + 3);
        }
    } else {
        for (int c = c0 + tid; c < c1; c += QSA_TOPK_THREADS) {
            const float m = mask_f16 ? __half2float(mh[c]) : mf[c];
            f(qsa_topk_key(sc[cb[c]] + m), c);
        }
    }
}

// f(key, cell) for n candidates of a row (the merge stage)
template <typename F>
static __device__ __forceinline__ void qsa_topk_for_cand(
        const uint32_t * __restrict__ ckey, const int32_t * __restrict__ cidx, const int n, F && f) {
    for (int i = threadIdx.x; i < n; i += QSA_TOPK_THREADS) {
        f(ckey[i], cidx[i]);
    }
}

struct qsa_topk_smem {
    int      lhist[WARP_SIZE*QSA_TOPK_HSTRIDE];
    int      hist[256];
    uint32_t prefix;
    int      need;
    int      eq;
    int      count;
};

// The radix select of top-k.cu k_top_k_select over the elements for_each visits (a key and a cell each), for one block.
// Calls out(pos, key, cell) for exactly k of them (pos < k, any order): every key above the k-th largest key, then the
// lowest cells among the keys equal to it. Every thread of the block must call this.
template <typename ForEach, typename Out>
static __device__ __forceinline__ void qsa_topk_select(qsa_topk_smem & sm, const int k, ForEach && for_each, Out && out) {
    const int tid  = threadIdx.x;
    const int lane = tid % WARP_SIZE;

    int * mine = sm.lhist + lane*QSA_TOPK_HSTRIDE;

    const auto clear = [&]() {
        for (int b = tid; b < WARP_SIZE*QSA_TOPK_HSTRIDE; b += QSA_TOPK_THREADS) {
            sm.lhist[b] = 0;
        }
        __syncthreads();
    };
    const auto collect = [&]() {
        __syncthreads();
        if (tid < 256) {
            int acc = 0;
#pragma unroll
            for (int l = 0; l < WARP_SIZE; ++l) {
                acc += sm.lhist[l*QSA_TOPK_HSTRIDE + tid];
            }
            sm.hist[tid] = acc;
        }
        __syncthreads();
    };

    // 1) value radix select, MSB first: `prefix` ends as the k-th largest key, sm.eq the number of keys equal to it
    uint32_t prefix = 0;
    uint32_t pmask  = 0;
    int      need   = k;
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8*pass;
        clear();
        for_each([&](const uint32_t key, const int c) {
            GGML_UNUSED(c);
            if ((key & pmask) == prefix) {
                atomicAdd(mine + ((key >> shift) & 0xFFu), 1);
            }
        });
        collect();
        if (tid == 0) {
            int acc = 0;
            int b   = 255;
            for (; b > 0; --b) {
                const int cnt = sm.hist[b];
                if (acc + cnt >= need) {
                    break;
                }
                acc += cnt;
            }
            sm.need   = need - acc;
            sm.prefix = prefix | ((uint32_t) b << shift);
            sm.eq     = sm.hist[b];
        }
        __syncthreads();
        need    = sm.need;
        prefix  = sm.prefix;
        pmask  |= 0xFFu << shift;
        __syncthreads();
    }

    // 2) ties on the k-th key: keep the `need` lowest cells among them
    uint32_t idx_thr = 0xFFFFFFFFu;
    if (sm.eq > need) {
        uint32_t iprefix = 0;
        uint32_t imask   = 0;
        int      ineed   = need;
        for (int pass = 0; pass < 4; ++pass) {
            const int shift = 24 - 8*pass;
            clear();
            for_each([&](const uint32_t key, const int c) {
                if (key == prefix && ((uint32_t) c & imask) == iprefix) {
                    atomicAdd(mine + (((uint32_t) c >> shift) & 0xFFu), 1);
                }
            });
            collect();
            if (tid == 0) {
                int acc = 0;
                int b   = 0;
                for (; b < 255; ++b) {
                    const int cnt = sm.hist[b];
                    if (acc + cnt >= ineed) {
                        break;
                    }
                    acc += cnt;
                }
                sm.need   = ineed - acc;
                sm.prefix = iprefix | ((uint32_t) b << shift);
            }
            __syncthreads();
            ineed    = sm.need;
            iprefix  = sm.prefix;
            imask   |= 0xFFu << shift;
            __syncthreads();
        }
        idx_thr = iprefix;
    }

    // 3) compaction of exactly k elements, one shared atomic per warp
    if (tid == 0) {
        sm.count = 0;
    }
    __syncthreads();
    for_each([&](const uint32_t key, const int c) {
        const bool     hit  = key > prefix || (key == prefix && (uint32_t) c <= idx_thr);
        const uint32_t act  = __activemask();
        const uint32_t bal  = __ballot_sync(act, hit);
        const int      lead = __ffs(act) - 1;
        int base = 0;
        if (lane == lead) {
            base = atomicAdd(&sm.count, __popc(bal));
        }
        base = __shfl_sync(act, base, lead);
        if (hit) {
            const int pos = base + __popc(bal & ((1u << lane) - 1u));
            if (pos < k) {
                out(pos, key, c);
            }
        }
    });
}

struct qsa_topk_row_args {
    const float   * score;
    const int32_t * cell_blk;
    const void    * mask;
    int             n_kv;
    int             n_tps;
    int64_t         s_sc1;
    int64_t         s_sc2;
    int64_t         s_cb1;
    int64_t         s_m1;
    int64_t         s_ms;
};

// one block per query row: the whole row in one select
template <bool mask_f16, bool vec4>
__launch_bounds__(QSA_TOPK_THREADS, 1)
static __global__ void k_qsa_topk(const qsa_topk_row_args a, int * __restrict__ dst, const int k, const int64_t s_d1, const int64_t s_d2) {
    const int s = blockIdx.x / a.n_tps;
    const int t = blockIdx.x - s*a.n_tps;

    const float   * sc  = a.score + s*a.s_sc2 + (int64_t) t*a.s_sc1;
    const int32_t * cb  = a.cell_blk + s*a.s_cb1;
    const half    * mh  = mask_f16 ? (const half  *) a.mask + s*a.s_ms + (int64_t) t*a.s_m1 : nullptr;
    const float   * mf  = mask_f16 ? nullptr : (const float *) a.mask + s*a.s_ms + (int64_t) t*a.s_m1;
    int           * out = dst + s*s_d2 + (int64_t) t*s_d1;

    __shared__ qsa_topk_smem sm;

    qsa_topk_select(sm, k,
        [&](auto && f) { qsa_topk_for_cells<mask_f16, vec4>(sc, cb, mh, mf, 0, a.n_kv, f); },
        [&](const int pos, const uint32_t key, const int c) { GGML_UNUSED(key); out[pos] = c; });
}

// [TAG_FN_QSA_FUSED] split select for long rows: stage 1 cuts each row into n_chunks chunks and selects the top k_c =
// min(k, chunk) of each chunk under the same order (key descending, cell ascending); stage 2 selects the top k of the
// row's candidates. Every element of the row's top k is in its chunk's top k, so the set is the one-block set.
template <bool mask_f16, bool vec4>
__launch_bounds__(QSA_TOPK_THREADS, 1)
static __global__ void k_qsa_topk_chunk(const qsa_topk_row_args a, uint32_t * __restrict__ ckey, int32_t * __restrict__ cidx,
        const int k, const int n_chunks) {
    const int g   = blockIdx.x;
    const int row = blockIdx.y;
    const int s   = row / a.n_tps;
    const int t   = row - s*a.n_tps;

    const int c0 = (int) (((int64_t) g*a.n_kv/n_chunks) & ~3LL);
    const int c1 = g == n_chunks - 1 ? a.n_kv : (int) (((int64_t) (g + 1)*a.n_kv/n_chunks) & ~3LL);
    const int kc = min(k, c1 - c0);

    const float   * sc = a.score + s*a.s_sc2 + (int64_t) t*a.s_sc1;
    const int32_t * cb = a.cell_blk + s*a.s_cb1;
    const half    * mh = mask_f16 ? (const half  *) a.mask + s*a.s_ms + (int64_t) t*a.s_m1 : nullptr;
    const float   * mf = mask_f16 ? nullptr : (const float *) a.mask + s*a.s_ms + (int64_t) t*a.s_m1;

    uint32_t * ok = ckey + ((int64_t) row*n_chunks + g)*k;
    int32_t  * oi = cidx + ((int64_t) row*n_chunks + g)*k;

    __shared__ qsa_topk_smem sm;

    qsa_topk_select(sm, kc,
        [&](auto && f) { qsa_topk_for_cells<mask_f16, vec4>(sc, cb, mh, mf, c0, c1, f); },
        [&](const int pos, const uint32_t key, const int c) { ok[pos] = key; oi[pos] = c; });
}

__launch_bounds__(QSA_TOPK_THREADS, 1)
static __global__ void k_qsa_topk_merge(const uint32_t * __restrict__ ckey, const int32_t * __restrict__ cidx,
        int * __restrict__ dst, const int k, const int n_kv, const int n_chunks, const int n_tps, const int64_t s_d1, const int64_t s_d2) {
    const int row = blockIdx.x;
    const int s   = row / n_tps;
    const int t   = row - s*n_tps;

    __shared__ qsa_topk_smem sm;

    // chunk g holds min(k, chunk length) candidates at the front of its k slots
    const uint32_t * rk = ckey + (int64_t) row*n_chunks*k;
    const int32_t  * ri = cidx + (int64_t) row*n_chunks*k;
    int * out = dst + s*s_d2 + (int64_t) t*s_d1;

    qsa_topk_select(sm, k,
        [&](auto && f) {
            for (int g = 0; g < n_chunks; ++g) {
                const int c0 = (int) (((int64_t) g*n_kv/n_chunks) & ~3LL);
                const int c1 = g == n_chunks - 1 ? n_kv : (int) (((int64_t) (g + 1)*n_kv/n_chunks) & ~3LL);
                qsa_topk_for_cand(rk + (int64_t) g*k, ri + (int64_t) g*k, min(k, c1 - c0), f);
            }
        },
        [&](const int pos, const uint32_t key, const int c) { GGML_UNUSED(key); out[pos] = c; });
}

// chunks per row for the split select: at least 4k cells each, at most QSA_TOPK_MAX_CHUNKS; 1 = one block per row.
// GGML_CUDA_QSA_TOPK_SPLIT=0 forces one block per row.
static int qsa_topk_n_chunks(const int n_kv, const int k) {
    static const bool enabled = [] {
        const char * e = getenv("GGML_CUDA_QSA_TOPK_SPLIT");
        return !(e && e[0] == '0');
    }();
    if (!enabled) {
        return 1;
    }
    const int64_t n = (int64_t) n_kv / (4*(int64_t) k);
    return (int) std::max<int64_t>(1, std::min<int64_t>(QSA_TOPK_MAX_CHUNKS, n));
}

void ggml_cuda_op_qsa_topk(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * score    = dst->src[0];
    const ggml_tensor * cell_blk = dst->src[1];
    const ggml_tensor * mask     = dst->src[2];

    const int     k        = (int) dst->ne[0];
    const int     n_tps    = (int) dst->ne[1];
    const int64_t n_stream = dst->ne[2];
    const int     n_kv     = (int) cell_blk->ne[0];

    const bool mask_f16 = mask->type == GGML_TYPE_F16;

    qsa_topk_row_args a;
    a.score    = (const float *) score->data;
    a.cell_blk = (const int32_t *) cell_blk->data;
    a.mask     = mask->data;
    a.n_kv     = n_kv;
    a.n_tps    = n_tps;
    a.s_sc1    = score->nb[1] / sizeof(float);
    a.s_sc2    = score->nb[2] / sizeof(float);
    a.s_cb1    = cell_blk->nb[1] / sizeof(int32_t);
    a.s_m1     = mask->nb[1] / ggml_type_size(mask->type);
    a.s_ms     = (mask_f16 ? mask->nb[3] : mask->nb[2]) / ggml_type_size(mask->type);

    const int64_t s_d1 = dst->nb[1] / sizeof(int32_t);
    const int64_t s_d2 = dst->nb[2] / sizeof(int32_t);

    // int4 / uint2 / float4 loads need every row start aligned: the base pointers and the row strides
    const size_t mask_vec = mask_f16 ? 8 : 16;
    const bool vec4 = n_kv % 4 == 0 &&
        ((uintptr_t) cell_blk->data) % 16 == 0 && (a.s_cb1*sizeof(int32_t)) % 16 == 0 &&
        ((uintptr_t) mask->data) % mask_vec == 0 && (mask->nb[1] % mask_vec) == 0 &&
        ((mask_f16 ? mask->nb[3] : mask->nb[2]) % mask_vec) == 0;

    const unsigned n_rows   = (unsigned) (n_tps*n_stream);
    const int      n_chunks = qsa_topk_n_chunks(n_kv, k);
    cudaStream_t   stream   = ctx.stream();

    if (n_chunks == 1) {
#define QSA_TOPK_LAUNCH(MF16, V4) \
        k_qsa_topk<MF16, V4><<<n_rows, QSA_TOPK_THREADS, 0, stream>>>(a, (int *) dst->data, k, s_d1, s_d2)
        if (mask_f16) {
            if (vec4) { QSA_TOPK_LAUNCH(true,  true); } else { QSA_TOPK_LAUNCH(true,  false); }
        } else {
            if (vec4) { QSA_TOPK_LAUNCH(false, true); } else { QSA_TOPK_LAUNCH(false, false); }
        }
#undef QSA_TOPK_LAUNCH
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    ggml_cuda_pool_alloc<uint32_t> ckey(ctx.pool(), (size_t) n_rows*n_chunks*k);
    ggml_cuda_pool_alloc<int32_t>  cidx(ctx.pool(), (size_t) n_rows*n_chunks*k);

    const dim3 grid1((unsigned) n_chunks, n_rows, 1);
#define QSA_TOPK_CHUNK_LAUNCH(MF16, V4) \
    k_qsa_topk_chunk<MF16, V4><<<grid1, QSA_TOPK_THREADS, 0, stream>>>(a, ckey.get(), cidx.get(), k, n_chunks)
    if (mask_f16) {
        if (vec4) { QSA_TOPK_CHUNK_LAUNCH(true,  true); } else { QSA_TOPK_CHUNK_LAUNCH(true,  false); }
    } else {
        if (vec4) { QSA_TOPK_CHUNK_LAUNCH(false, true); } else { QSA_TOPK_CHUNK_LAUNCH(false, false); }
    }
#undef QSA_TOPK_CHUNK_LAUNCH
    CUDA_CHECK(cudaGetLastError());

    k_qsa_topk_merge<<<n_rows, QSA_TOPK_THREADS, 0, stream>>>(ckey.get(), cidx.get(), (int *) dst->data, k, n_kv, n_chunks, n_tps, s_d1, s_d2);
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_qsa_supported(const ggml_tensor * op) {
    if (op->op == GGML_OP_QSA_SCORE) {
        const ggml_tensor * k = op->src[0];
        const ggml_tensor * q = op->src[4];
        const bool k_ok = k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_BF16 || k->type == GGML_TYPE_Q8_0;
        return k_ok && k->ne[0] % WARP_SIZE == 0 && k->ne[0] <= 8*WARP_SIZE && q->ne[1] <= qsa_score_max_pairs(k->ne[0]) &&
            op->ne[0] <= INT32_MAX && k->ne[1] <= INT32_MAX;
    }
    if (op->op == GGML_OP_QSA_TOPK) {
        const ggml_tensor * mask = op->src[2];
        return (mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32) && op->src[1]->ne[0] <= INT32_MAX &&
            op->ne[0] <= op->src[1]->ne[0];
    }
    return false;
}
