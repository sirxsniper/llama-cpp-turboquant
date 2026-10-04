// [TAG_FN_R4_QSA_POS] GGML_OP_QSA_MASK: the attention mask of a QSA layer from the positional vectors and the indexer's
// selection (ggml_qsa_mask, ggml.h). It replaces the explicit [n_kv, n_q] KQ mask input, the -INF fill of
// [n_kv + n_sel, n_q], the scatter and the add of the explicit path with one [n_kv, n_q] f16 output.
//
// Grid (cell chunks, queries). A block owns the cells [c0, c1) of one query row: it fills them with -INF, then (after a
// barrier, so every fill of the block is visible) walks the query's selection and writes 0 at the selected cells of
// its range that are live and visible. No two blocks write the same element; two slots of one block that select the
// same cell write the same value.

#include "qsa-mask.cuh"

#include <cmath>

#define QSA_MASK_CHUNK   8192 // cells per block (16 KiB of f16)
#define QSA_MASK_THREADS 256

static __global__ void k_qsa_mask(
        const int32_t * __restrict__ kv_pos, const int32_t * __restrict__ kv_seq,
        const char    * __restrict__ q_pos,  const int64_t q_nb0, const int64_t q_nb1, const int ms,
        const char    * __restrict__ sel,    const int64_t sel_nb0, const int64_t sel_nb1, const int n_sel,
        const char    * __restrict__ live,   const int64_t live_nb0, const int64_t live_nb1, const int n_lsel, const int group,
        half * __restrict__ dst, const int64_t dst_nb1, const int n_kv) {
    const int iq = blockIdx.y;
    const int c0 = blockIdx.x*QSA_MASK_CHUNK;
    const int c1 = min(n_kv, c0 + QSA_MASK_CHUNK);

    half * row = (half *) ((char *) dst + iq*dst_nb1);

    const half ninf = __float2half(-INFINITY);
    if ((c0 & 1) == 0 && (c1 & 1) == 0 && (((uintptr_t) row) & 3) == 0) {
        const half2 ninf2 = make_half2(ninf, ninf);
        half2 * row2 = (half2 *) row;
        for (int c = c0/2 + threadIdx.x; c < c1/2; c += blockDim.x) {
            row2[c] = ninf2;
        }
    } else {
        for (int c = c0 + threadIdx.x; c < c1; c += blockDim.x) {
            row[c] = ninf;
        }
    }

    __syncthreads();

    const int32_t  qp = *(const int32_t *) (q_pos + iq*q_nb0);
    const uint32_t qs = ms ? *(const uint32_t *) (q_pos + q_nb1 + iq*q_nb0) : 0u;

    const char * srow = sel + iq*sel_nb1;
    const char * lrow = live ? live + iq*live_nb1 : nullptr;

    for (int s = threadIdx.x; s < n_sel; s += blockDim.x) {
        const int32_t c = *(const int32_t *) (srow + s*sel_nb0);
        if (c < c0 || c >= c1) {
            continue; // another block's range, or no cell (padded block, missing tail cell)
        }
        if (s < n_lsel) {
            const float l = *(const float *) (lrow + (s/group)*live_nb0);
            if (!(l > -INFINITY)) {
                continue; // a pool the indexer could not see
            }
        }
        const int32_t kp = kv_pos[c];
        if (kp < 0 || kp > qp) {
            continue;
        }
        if (ms && (((uint32_t) kv_seq[c]) & qs) == 0u) {
            continue;
        }
        row[c] = __float2half(0.0f);
    }
}

bool ggml_cuda_qsa_mask_supported(const ggml_tensor * op) {
    const ggml_tensor * kv_pos = op->src[0];
    const ggml_tensor * q_pos  = op->src[1];
    const ggml_tensor * sel    = op->src[2];
    const ggml_tensor * live   = op->src[3];
    return op->type == GGML_TYPE_F16 && kv_pos->type == GGML_TYPE_I32 && q_pos->type == GGML_TYPE_I32 &&
        sel->type == GGML_TYPE_I32 && (live == nullptr || live->type == GGML_TYPE_F32) &&
        ggml_is_contiguous_rows(op) && op->ne[0] <= INT32_MAX && op->ne[1] <= 65535 && sel->ne[0] <= INT32_MAX;
}

void ggml_cuda_op_qsa_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * kv_pos = dst->src[0];
    const ggml_tensor * q_pos  = dst->src[1];
    const ggml_tensor * sel    = dst->src[2];
    const ggml_tensor * live   = dst->src[3];

    const int64_t n_kv  = dst->ne[0];
    const int64_t n_q   = dst->ne[1];
    const int32_t group = ggml_get_op_params_i32(dst, 0);
    const int     ms    = kv_pos->ne[1] == 2 ? 1 : 0;

    GGML_ASSERT(n_q <= 65535 && n_kv <= INT32_MAX);
    GGML_ASSERT(!live || group >= 1);

    const int n_lsel = live ? (int) (live->ne[0]*group) : 0;

    const int32_t * kvp = (const int32_t *) kv_pos->data;
    const int32_t * kvs = ms ? (const int32_t *) ((const char *) kv_pos->data + kv_pos->nb[1]) : nullptr;

    const dim3 grid((unsigned) ((n_kv + QSA_MASK_CHUNK - 1)/QSA_MASK_CHUNK), (unsigned) n_q, 1);
    if (grid.x == 0 || grid.y == 0) {
        return;
    }
    k_qsa_mask<<<grid, QSA_MASK_THREADS, 0, ctx.stream()>>>(kvp, kvs,
            (const char *) q_pos->data, q_pos->nb[0], q_pos->nb[1], ms,
            (const char *) sel->data, sel->nb[0], sel->nb[1], (int) sel->ne[0],
            live ? (const char *) live->data : nullptr, live ? live->nb[0] : 0, live ? live->nb[1] : 0, n_lsel, group > 0 ? group : 1,
            (half *) dst->data, dst->nb[1], (int) n_kv);
    CUDA_CHECK(cudaGetLastError());
}
