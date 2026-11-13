// [TAG_QSA_POOL_RANGE] GGML_OP_KPOOL_MASK: the pool mask rows of the lightning indexer from one range per query
// (ggml_kpool_mask, ggml.h). It replaces the explicit [n_pool, n_batch] f16 mask input with the same bytes, written on
// the device: -INF, 0 over the visible pools [first, first + count), -INF.
//
// Grid (pool chunks, queries). A block owns the pools [c0, c1) of one query row. No two blocks write the same element.

#include "kpool-mask.cuh"

#include <cmath>

#define KPOOL_MASK_CHUNK   8192 // pools per block (16 KiB of f16)
#define KPOOL_MASK_THREADS 256

static __global__ void k_kpool_mask(
        const char * __restrict__ range, const int64_t range_nb1, const int row0,
        half * __restrict__ dst, const int64_t dst_nb1, const int n_pool) {
    const int ir = blockIdx.y;
    const int c0 = blockIdx.x*KPOOL_MASK_CHUNK;
    const int c1 = min(n_pool, c0 + KPOOL_MASK_CHUNK);

    const int32_t * rg = (const int32_t *) (range + (int64_t) (row0 + ir)*range_nb1);
    const int p0 = min(max(rg[0], 0), n_pool);
    const int p1 = p0 + min(max(rg[1], 0), n_pool - p0);

    half * row = (half *) ((char *) dst + ir*dst_nb1);

    const half ninf = __float2half(-INFINITY);
    const half zero = __float2half(0.0f);
    for (int c = c0 + threadIdx.x; c < c1; c += blockDim.x) {
        row[c] = c >= p0 && c < p1 ? zero : ninf;
    }
}

bool ggml_cuda_kpool_mask_supported(const ggml_tensor * op) {
    const ggml_tensor * range = op->src[0];
    return op->type == GGML_TYPE_F16 && range->type == GGML_TYPE_I32 && range->ne[0] == 2 &&
        range->nb[0] == sizeof(int32_t) && ggml_is_contiguous_rows(op) && op->ne[0] <= INT32_MAX && op->ne[1] <= 65535;
}

void ggml_cuda_op_kpool_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * range = dst->src[0];

    const int64_t n_pool = dst->ne[0];
    const int64_t n_rows = dst->ne[1];
    const int32_t row0   = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(n_rows <= 65535 && n_pool <= INT32_MAX);
    GGML_ASSERT(row0 >= 0 && row0 + n_rows <= range->ne[1]);

    const dim3 grid((unsigned) ((n_pool + KPOOL_MASK_CHUNK - 1)/KPOOL_MASK_CHUNK), (unsigned) n_rows, 1);
    if (grid.x == 0 || grid.y == 0) {
        return;
    }
    k_kpool_mask<<<grid, KPOOL_MASK_THREADS, 0, ctx.stream()>>>(
            (const char *) range->data, range->nb[1], row0, (half *) dst->data, dst->nb[1], (int) n_pool);
    CUDA_CHECK(cudaGetLastError());
}
