// [TAG_FN_L4_HC] Flash-Next (qwen4exp) lever round 4: an hc mixer in three launches, see fn-l4-hc.cuh.

#include "fn-l4-hc.cuh"
#include "vecdotq.cuh"
#include "unary.cuh"
#include "fn-l3.cuh"

// GGML_CUDA_FN_L4_TRACE=1: a stderr line per step of a matched mixer, a device sync after each launch (debugging only)
static bool fn_l4_trace() {
    static const bool on = getenv("GGML_CUDA_FN_L4_TRACE") != nullptr;
    return on;
}

static void fn_l4_trace_step(const char * what) {
    if (!fn_l4_trace()) {
        return;
    }
    const cudaError_t e = cudaDeviceSynchronize();
    fprintf(stderr, "fn_l4 trace: %s -> %s\n", what, cudaGetErrorString(e));
    fflush(stderr);
}

bool ggml_cuda_fn_l4_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_FN_L4");
        return !(e && e[0] == '0');
    }();
    return on;
}

// quantize_q8_1 (quantize.cu) of one q8_1 block held by a warp, lane l holding element l: the same operations in the
// same order, so the same bytes. Every lane of the warp must call it.
static __device__ __forceinline__ void fn_l4_q8_1_store(const float xi, block_q8_1 * yb, const int lane) {
    float amax = fabsf(xi);
    float sum  = xi;

    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    yb->qs[lane] = q;

    if (lane > 0) {
        return;
    }

    yb->ds = make_half2(d, sum);
}

// ---------------------------------------------------------------------------------------------------------------------
// K1: [the combine ->] grouped rms_norm * gamma -> xn and its q8_1 copy. Block (row h, channel t) of rms_norm_f32's
// launch (rows of n_embd, the block size it takes for n_embd), the combine of dsv4_hc_post_f32 (no comb) with the weight
// of scale_f32 -> op_sigmoid -> scale_f32, the norm of rms_norm_f32<block_size, do_multiply>.
// ---------------------------------------------------------------------------------------------------------------------

struct fn_l4_norm_args {
    const float * bo;  int64_t sb0, sb1;         // block_out [n_embd, T]
    const float * bo2; int64_t sb20, sb21;       // with an ADD before the combine: block_out = bo + bo2 (op_add)
    const float * res; int64_t sr0, sr1, sr2;    // residual [n_embd, hc, T]
    const float * inj; int     inj_hc;            // raw inject [hc, T], contiguous
    float         s1, b1, s2, b2;                 // the two SCALE ops
    float *       post; int64_t sp0, sp1, sp2;   // the combine's output
    const float * x;   int64_t sx1, sx2;         // rms_norm's input without a combine (column stride 1)
    const float * w;   int64_t sw1;              // gamma [n_embd, hc] (column stride 1)
    float         eps;
    float *       xn;                             // [n_embd, hc, T] contiguous
    block_q8_1 *  xq;  int     bpr;               // q8_1 rows of hc*n_embd, bpr blocks per row
    int           ncols;                          // n_embd
};

#define FN_L4_NORM_MAXV 4

template <int block_size, bool has_post>
__launch_bounds__(block_size, 1)
static __global__ void k_fn_l4_hc_norm(const fn_l4_norm_args a) {
    ggml_cuda_pdl_lc();
    const int nrows   = gridDim.x;
    const int row     = blockIdx.x;
    const int channel = blockIdx.y;
    const int tid     = threadIdx.x;
    const int ncols   = a.ncols;

    float r[FN_L4_NORM_MAXV];

    ggml_cuda_pdl_sync();
    if constexpr (has_post) {
        // scale_f32, op_sigmoid, scale_f32 on element (row, channel) of the inject
        const float t  = a.s1 * a.inj[row + channel*a.inj_hc] + a.b1;
        const float sg = 1.0f / (1.0f + expf(-t));
        const float pw = a.s2 * sg + a.b2;
#pragma unroll
        for (int k = 0; k < FN_L4_NORM_MAXV; ++k) {
            const int col = tid + k*block_size;
            if (col < ncols) {
                float xb = a.bo[col*a.sb0 + channel*a.sb1];
                if (a.bo2) {
                    xb = xb + a.bo2[col*a.sb20 + channel*a.sb21];
                }
                float sum = xb * pw;
                sum += a.res[col*a.sr0 + row*a.sr1 + channel*a.sr2];
                a.post[col*a.sp0 + row*a.sp1 + channel*a.sp2] = sum;
                r[k] = sum;
            }
        }
    } else {
        const float * x = a.x + channel*a.sx2 + row*a.sx1;
#pragma unroll
        for (int k = 0; k < FN_L4_NORM_MAXV; ++k) {
            const int col = tid + k*block_size;
            if (col < ncols) {
                r[k] = x[col];
            }
        }
    }

    float tmp = 0.0f; // partial sum for thread in warp
#pragma unroll
    for (int k = 0; k < FN_L4_NORM_MAXV; ++k) {
        const int col = tid + k*block_size;
        if (col < ncols) {
            tmp += r[k] * r[k];
        }
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + a.eps);

    const float * w  = a.w + row*a.sw1;
    float *       xn = a.xn + ((channel*nrows + row)*ncols);
    block_q8_1 *  yq = a.xq + (int64_t) channel*a.bpr;
#pragma unroll
    for (int k = 0; k < FN_L4_NORM_MAXV; ++k) {
        const int col = tid + k*block_size;
        if (col < ncols) {
            const float v = scale * r[k] * w[col];
            xn[col] = v;
            fn_l4_q8_1_store(v, yq + (row*ncols + col)/QK8_1, tid % WARP_SIZE);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// K2: blocks [0, n_inj_blocks): the inject, warp vw of mul_mat_vec_f's block (block size inj_bs) per warp, its sums
// reduced over the warp and stored; blocks [n_inj_blocks, ...): row r of the down mat-vec as mul_mat_vec_q<Q8_0,
// ncols_dst> computes it (4 warps, the same K loop and reduction; the rows per block do not change a row's sums).
// ---------------------------------------------------------------------------------------------------------------------

struct fn_l4_down_args {
    const void *       wd; int stride_row_x; int ncols_x; int nrows; // q8_0 [ncols_x, nrows]
    const block_q8_1 * xq; int stride_col_y;
    float *            lo;                                           // [T][nrows]
    const float *      wi; int stride_row_i;                         // f32 [ncols_x, inj_rows]
    const float *      xn; int stride_col_xn;                        // f32 [ncols_x, T]
    float *            part; int inj_rows; int inj_bs; int n_inj_blocks;
};

#define FN_L4_DOWN_AHEAD 10

template <int ncols_dst>
__launch_bounds__(4*WARP_SIZE, 1)
static __global__ void k_fn_l4_hc_down(const fn_l4_down_args a) {
    ggml_cuda_pdl_lc();
    const int lane = threadIdx.x;

    if ((int) blockIdx.x < a.n_inj_blocks) {
        const int nvw  = a.inj_bs / WARP_SIZE;
        const int bpir = (nvw + 3) / 4;
        const int irow = blockIdx.x / bpir;
        const int vw   = (blockIdx.x % bpir)*4 + threadIdx.y;
        if (vw >= nvw) {
            return;
        }
        ggml_cuda_pdl_sync();

        const int bs            = a.inj_bs;
        const int ncols2        = a.ncols_x / 2;
        const int stride_col_y2 = a.stride_col_xn / 2;
        const float2 * x2 = (const float2 *) (a.wi + int64_t(irow)*a.stride_row_i);
        const float2 * y2 = (const float2 *) a.xn;

        float sumf[ncols_dst] = {0.0f};

        constexpr int ahead = 4;
        int col2 = vw*WARP_SIZE + lane;
        for (; col2 + (ahead - 1)*bs < ncols2; col2 += ahead*bs) {
            float2 tmpx[ahead];
            float2 tmpy[ahead][ncols_dst];
#pragma unroll
            for (int u = 0; u < ahead; ++u) {
                tmpx[u] = x2[col2 + u*bs];
            }
#pragma unroll
            for (int u = 0; u < ahead; ++u) {
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    tmpy[u][j] = y2[j*stride_col_y2 + col2 + u*bs];
                }
            }
#pragma unroll
            for (int u = 0; u < ahead; ++u) {
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    ggml_cuda_mad(sumf[j], tmpx[u].x, tmpy[u][j].x);
                    ggml_cuda_mad(sumf[j], tmpx[u].y, tmpy[u][j].y);
                }
            }
        }
        for (; col2 < ncols2; col2 += bs) {
            const float2 tmpx = x2[col2];
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const float2 tmpy = y2[j*stride_col_y2 + col2];
                ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
                ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);
            }
        }

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            sumf[j] = warp_reduce_sum<WARP_SIZE>(sumf[j]);
        }
        if (lane == 0) {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                a.part[(int64_t(irow)*nvw + vw)*ncols_dst + j] = sumf[j];
            }
        }
        return;
    }

    constexpr int qk     = QK8_0;
    constexpr int qi     = QI8_0;
    constexpr int vdr    = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int nwarps = 4;
    constexpr int blocks_per_iter = vdr * nwarps*WARP_SIZE / qi;

    const int tid  = WARP_SIZE*threadIdx.y + threadIdx.x;
    const int row0 = blockIdx.x - a.n_inj_blocks;
    const int blocks_per_row_x = a.ncols_x / qk;

    ggml_cuda_pdl_sync();

    float tmp[ncols_dst] = {0.0f};

    const block_q8_1 * y = a.xq;
    const int kbx_offset = row0*a.stride_row_x;

    // the K loop FN_L4_DOWN_AHEAD iterations per pass, each guarded, so that their loads issue together (the hc down row
    // of 10240 is one pass); kbx still runs in the original order, so each thread's sum is the same
    const int kqs = vdr * (tid % (qi/vdr));
    for (int kbx0 = tid / (qi/vdr); kbx0 < blocks_per_row_x; kbx0 += FN_L4_DOWN_AHEAD*blocks_per_iter) {
#pragma unroll
        for (int u = 0; u < FN_L4_DOWN_AHEAD; ++u) {
            const int kbx = kbx0 + u*blocks_per_iter;
            if (kbx < blocks_per_row_x) {
                const int kby = kbx * (qk/QK8_1);
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    tmp[j] += vec_dot_q8_0_q8_1(a.wd, &y[j*a.stride_col_y + kby], kbx_offset + kbx, kqs);
                }
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1][ncols_dst][WARP_SIZE];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            tmp_shared[threadIdx.y-1][j][threadIdx.x] = tmp[j];
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int l = 0; l < nwarps-1; ++l) {
            tmp[j] += tmp_shared[l][j][threadIdx.x];
        }
        tmp[j] = warp_reduce_sum<WARP_SIZE>(tmp[j]);
        if (threadIdx.x == 0) {
            a.lo[j*a.nrows + row0] = tmp[j];
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// K3: block b < n_main: the 32 columns [32b, 32b + 32) of hc_pre's output for every token. Prologue: scale_f32 ->
// op_silu -> quantize_q8_1 of the low rank into shared memory. Rows {h*n_embd + 32b + i}: the up mat-vec as
// mul_mat_vec_q<Q8_0, ncols_dst> computes a row of at most 32 q8_0 blocks (lane l adds the terms of lane l of warps
// 0..3 of its block in that order, as mul_mat_vec_q_fn_l3_smk), one warp per rows_per_warp rows. Epilogue:
// dsv4_hc_pre_f32<true> and the q8_1 copy. The last block (with an inject): mul_mat_vec_f's final reduction of the
// inject's per-warp sums.
// ---------------------------------------------------------------------------------------------------------------------

struct fn_l4_up_args {
    const float * lo; int lr; float s_lo, b_lo;  // [T][lr], the SCALE before the silu
    const void *  wu; int stride_row_x;          // q8_0 [lr, hc*n_embd]
    const float * xn; int n_embd; int hc;        // [n_embd, hc, T] contiguous
    float *       mixed; int64_t sd0, sd1; float scale;
    block_q8_1 *  mq; int mq_bpr;                // q8_1 copy of mixed (nullptr: none)
    const float * part; float * inj; int64_t inj_s1; int inj_rows; int nvw;
    int           n_main;
    float         zero;                          // 0.0f, a parameter so that no add of a zero partial sum is folded away
};

#define FN_L4_HC 4

template <int ncols_dst, int rows_per_warp>
__launch_bounds__(WARP_SIZE*(WARP_SIZE*FN_L4_HC/rows_per_warp), 1)
static __global__ void k_fn_l4_hc_up(const fn_l4_up_args a) {
    constexpr int nw   = WARP_SIZE*FN_L4_HC/rows_per_warp;
    constexpr int qk   = QK8_0;
    constexpr int qi   = QI8_0;
    constexpr int vdr  = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int nwarps_mmvq = 4;

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;

    __shared__ block_q8_1 s_y[ncols_dst][WARP_SIZE];
    __shared__ float      s_gate[WARP_SIZE*FN_L4_HC][ncols_dst];

    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();

    if ((int) blockIdx.x >= a.n_main) {
        // the inject: buf_iw of mul_mat_vec_f holds the warp sums, zeros after them; warp 0 reduces it
        if (warp == 0) {
            for (int r = 0; r < a.inj_rows; ++r) {
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    float v = lane < a.nvw ? a.part[(int64_t(r)*a.nvw + lane)*ncols_dst + j] : 0.0f;
                    v = warp_reduce_sum<WARP_SIZE>(v);
                    if (lane == 0) {
                        a.inj[j*a.inj_s1 + r] = v;
                    }
                }
            }
        }
        return;
    }

    // the low rank as q8_1 (one block per warp and pass)
    const int nb = a.lr / QK8_1;
    for (int idx = warp; idx < ncols_dst*nb; idx += nw) {
        const int j  = idx / nb;
        const int kb = idx - j*nb;
        const float t = a.s_lo * a.lo[j*a.lr + kb*QK8_1 + lane] + a.b_lo;
        const float v = ggml_cuda_op_silu_single(t);
        fn_l4_q8_1_store(v, &s_y[j][kb], lane);
    }
    __syncthreads();

    const int i0b = blockIdx.x*WARP_SIZE;
    const int blocks_per_row_x = a.lr / qk;
#pragma unroll
    for (int rr = 0; rr < rows_per_warp; ++rr) {
        const int rl  = warp*rows_per_warp + rr;
        const int h   = rl / WARP_SIZE;
        const int i   = rl % WARP_SIZE;
        const int row = h*a.n_embd + i0b + i;
        const int kbx_offset = row*a.stride_row_x;

        float tmp[ncols_dst];
#pragma unroll
        for (int w = 0; w < nwarps_mmvq; ++w) {
            const int tid = WARP_SIZE*w + lane;
            const int kbx = tid / (qi/vdr);
            float part[ncols_dst];
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                part[j] = a.zero;
            }
            if (kbx < blocks_per_row_x) {
                const int kby = kbx * (qk/QK8_1);
                const int kqs = vdr * (tid % (qi/vdr));
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    part[j] += vec_dot_q8_0_q8_1(a.wu, &s_y[j][kby], kbx_offset + kbx, kqs);
                }
            }
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                tmp[j] = w == 0 ? part[j] : tmp[j] + part[j];
            }
        }
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            tmp[j] = warp_reduce_sum<WARP_SIZE>(tmp[j]);
        }
        if (lane == 0) {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                s_gate[rl][j] = tmp[j];
            }
        }
    }
    __syncthreads();

    if (warp >= ncols_dst) {
        return;
    }
    const int j  = warp;
    const int i0 = i0b + lane;
    float sum = 0.0f;
    for (int64_t ih = 0; ih < a.hc; ++ih) {
        const float xv = a.xn[i0 + ih*a.n_embd + j*(int64_t) a.hc*a.n_embd];
        float wv;
        wv = 1.0f / (1.0f + expf(-s_gate[ih*WARP_SIZE + lane][j]));
        sum += xv * wv;
    }
    const float v = a.scale * sum;
    a.mixed[i0*a.sd0 + j*a.sd1] = v;
    if (a.mq) {
        fn_l4_q8_1_store(v, a.mq + (int64_t) j*a.mq_bpr + i0/QK8_1, lane);
    }
}

// ---------------------------------------------------------------------------------------------------------------------

static float fn_l4_op_f32(const ggml_tensor * t, int slot) {
    float v;
    memcpy(&v, (const float *) t->op_params + slot, sizeof(float));
    return v;
}

void ggml_cuda_fn_l4_hc_run(ggml_backend_cuda_context & ctx, const ggml_cuda_fn_l4_hc_chain & c, ggml_cgraph * cgraph,
        int i_first, int i_last, void (*plain_mm)(ggml_backend_cuda_context & ctx, ggml_tensor * dst)) {
    const ggml_tensor * xn = c.mul;
    const int64_t n_embd = xn->ne[0];
    const int64_t hc     = xn->ne[1];
    const int64_t T      = xn->ne[2];
    const int64_t hc_dim = n_embd*hc;
    const ggml_tensor * wd = c.down->src[0];
    const ggml_tensor * wu = c.up->src[0];
    const int64_t lr = wd->ne[1];
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(hc == FN_L4_HC && T >= 1 && T <= 4);
    if (fn_l4_trace()) {
        fprintf(stderr, "fn_l4 trace: run n_embd %lld hc %lld T %lld lr %lld post %d inject %d fused %d bs %d q8 %d nodes %d..%d\n",
                (long long) n_embd, (long long) hc, (long long) T, (long long) wd->ne[1], c.s1 != nullptr, c.inject != nullptr,
                c.inject_fused, c.inject_bs, c.q8, i_first, i_last);
        fflush(stderr);
    }

    const int bpr = (int) (hc_dim / QK8_1);
    ggml_cuda_pool_alloc<block_q8_1> xq(ctx.pool(), (size_t) (T*bpr));
    ggml_cuda_pool_alloc<float>      lo(ctx.pool(), (size_t) (T*lr));

    int nvw = 0;
    int inj_rows = 0;
    int n_inj_blocks = 0;
    const bool inj_fused = c.inject != nullptr && c.inject_fused;
    if (inj_fused) {
        nvw          = c.inject_bs / WARP_SIZE;
        inj_rows     = (int) c.inject->src[0]->ne[1];
        n_inj_blocks = inj_rows * ((nvw + 3)/4);
    }
    ggml_cuda_pool_alloc<float> part(ctx.pool(), (size_t) std::max<int64_t>(1, (int64_t) inj_rows*nvw*T));

    // K1
    {
        fn_l4_norm_args a = {};
        const ggml_tensor * w = c.mul->src[0] == c.rms ? c.mul->src[1] : c.mul->src[0];
        a.w   = (const float *) w->data;
        a.sw1 = w->nb[1]/sizeof(float);
        memcpy(&a.eps, c.rms->op_params, sizeof(float));
        a.xn    = (float *) c.mul->data;
        a.xq    = xq.get();
        a.bpr   = bpr;
        a.ncols = (int) n_embd;
        const bool has_post = c.s1 != nullptr;
        if (has_post) {
            const ggml_tensor * bo  = c.add ? c.add->src[0] : c.post->src[0];
            const ggml_tensor * res = c.post->src[1];
            a.bo  = (const float *) bo->data;
            a.sb0 = bo->nb[0]/sizeof(float);
            a.sb1 = bo->nb[1]/sizeof(float);
            if (c.add) {
                const ggml_tensor * bo2 = c.add->src[1];
                a.bo2  = (const float *) bo2->data;
                a.sb20 = bo2->nb[0]/sizeof(float);
                a.sb21 = bo2->nb[1]/sizeof(float);
            }
            a.res = (const float *) res->data;
            a.sr0 = res->nb[0]/sizeof(float);
            a.sr1 = res->nb[1]/sizeof(float);
            a.sr2 = res->nb[2]/sizeof(float);
            a.inj    = (const float *) c.s1->src[0]->data;
            a.inj_hc = (int) hc;
            a.s1 = fn_l4_op_f32(c.s1, 0);
            a.b1 = fn_l4_op_f32(c.s1, 1);
            a.s2 = fn_l4_op_f32(c.s2, 0);
            a.b2 = fn_l4_op_f32(c.s2, 1);
            a.post = (float *) c.post->data;
            a.sp0  = c.post->nb[0]/sizeof(float);
            a.sp1  = c.post->nb[1]/sizeof(float);
            a.sp2  = c.post->nb[2]/sizeof(float);
        } else {
            const ggml_tensor * x = c.rms->src[0];
            a.x   = (const float *) x->data;
            a.sx1 = x->nb[1]/sizeof(float);
            a.sx2 = x->nb[2]/sizeof(float);
        }
        const dim3 grid((unsigned) hc, (unsigned) T, 1);
        if (n_embd < 1024) {
            const ggml_cuda_kernel_launch_params lp(grid, dim3(256, 1, 1), 32*sizeof(float), stream);
            if (has_post) {
                ggml_cuda_kernel_launch(k_fn_l4_hc_norm<256, true>, lp, a);
            } else {
                ggml_cuda_kernel_launch(k_fn_l4_hc_norm<256, false>, lp, a);
            }
        } else {
            const ggml_cuda_kernel_launch_params lp(grid, dim3(1024, 1, 1), 32*sizeof(float), stream);
            if (has_post) {
                ggml_cuda_kernel_launch(k_fn_l4_hc_norm<1024, true>, lp, a);
            } else {
                ggml_cuda_kernel_launch(k_fn_l4_hc_norm<1024, false>, lp, a);
            }
        }
    }

    fn_l4_trace_step("K1");

    // K2
    {
        fn_l4_down_args a = {};
        a.wd           = wd->data;
        a.stride_row_x = (int) (wd->nb[1]/ggml_type_size(wd->type));
        a.ncols_x      = (int) hc_dim;
        a.nrows        = (int) lr;
        a.xq           = xq.get();
        a.stride_col_y = bpr;
        a.lo           = lo.get();
        if (inj_fused) {
            const ggml_tensor * wi = c.inject->src[0];
            a.wi            = (const float *) wi->data;
            a.stride_row_i  = (int) (wi->nb[1]/sizeof(float));
            a.xn            = (const float *) c.inject->src[1]->data;
            a.stride_col_xn = (int) (c.inject->src[1]->nb[1]/sizeof(float));
            a.part          = part.get();
            a.inj_rows      = inj_rows;
            a.inj_bs        = c.inject_bs;
            a.n_inj_blocks  = n_inj_blocks;
        }
        const ggml_cuda_kernel_launch_params lp(dim3((unsigned) (n_inj_blocks + lr), 1, 1), dim3(WARP_SIZE, 4, 1), 0, stream);
        switch (T) {
            case 1: ggml_cuda_kernel_launch(k_fn_l4_hc_down<1>, lp, a); break;
            case 2: ggml_cuda_kernel_launch(k_fn_l4_hc_down<2>, lp, a); break;
            case 3: ggml_cuda_kernel_launch(k_fn_l4_hc_down<3>, lp, a); break;
            case 4: ggml_cuda_kernel_launch(k_fn_l4_hc_down<4>, lp, a); break;
            default: GGML_ABORT("fatal error");
        }
    }

    fn_l4_trace_step("K2");

    // an inject mul_mat_vec_f would not compute: the backend's own op, between the two launches that read xn
    if (c.inject != nullptr && !inj_fused) {
        plain_mm(ctx, c.inject);
    }

    // K3
    {
        fn_l4_up_args a = {};
        a.lo   = lo.get();
        a.lr   = (int) lr;
        a.s_lo = fn_l4_op_f32(c.lo_s, 0);
        a.b_lo = fn_l4_op_f32(c.lo_s, 1);
        a.wu           = wu->data;
        a.stride_row_x = (int) (wu->nb[1]/ggml_type_size(wu->type));
        a.xn     = (const float *) c.mul->data;
        a.n_embd = (int) n_embd;
        a.hc     = (int) hc;
        a.mixed  = (float *) c.pre->data;
        a.sd0    = c.pre->nb[0]/sizeof(float);
        a.sd1    = c.pre->nb[1]/sizeof(float);
        a.scale  = fn_l4_op_f32(c.pre, 0);
        a.n_main = (int) (n_embd/WARP_SIZE);
        a.zero   = 0.0f;
        if (inj_fused) {
            a.part     = part.get();
            a.inj      = (float *) c.inject->data;
            a.inj_s1   = c.inject->nb[1]/sizeof(float);
            a.inj_rows = inj_rows;
            a.nvw      = nvw;
        }
        if (c.q8) {
            ggml_cuda_fn_l3_q8_dst q8;
            if (ggml_cuda_mmvq_q8_produce(ctx, c.pre, q8, /*padding_ok =*/ false)) {
                if (q8.ne10 != n_embd || q8.nrows != T || q8.bpr*QK8_1 != n_embd) {
                    // a reader with other rows: no copy (the cache must not claim one)
                    ctx.mmvq_q8.reset();
                } else {
                    a.mq     = q8.y;
                    a.mq_bpr = (int) q8.bpr;
                    // the nodes of this launch group write no bytes of the copy's source but pre's own
                    ctx.mmvq_q8.group   = cgraph->nodes + i_first;
                    ctx.mmvq_q8.n_group = i_last - i_first + 1;
                }
            }
        }
        constexpr int rpw = 4;
        const ggml_cuda_kernel_launch_params lp(dim3((unsigned) (a.n_main + (inj_fused ? 1 : 0)), 1, 1),
                dim3(WARP_SIZE, WARP_SIZE*FN_L4_HC/rpw, 1), 0, stream);
        switch (T) {
            case 1: ggml_cuda_kernel_launch(k_fn_l4_hc_up<1, rpw>, lp, a); break;
            case 2: ggml_cuda_kernel_launch(k_fn_l4_hc_up<2, rpw>, lp, a); break;
            case 3: ggml_cuda_kernel_launch(k_fn_l4_hc_up<3, rpw>, lp, a); break;
            case 4: ggml_cuda_kernel_launch(k_fn_l4_hc_up<4, rpw>, lp, a); break;
            default: GGML_ABORT("fatal error");
        }
    }
    fn_l4_trace_step("K3");
}
