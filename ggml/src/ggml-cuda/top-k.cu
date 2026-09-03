#include "argsort.cuh"
#include "top-k.cuh"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

// [TAG_TOPK_SELECT] Radix-select top-k for small k: one block per row, four 8-bit passes over a
// monotone key to find the exact k-th largest value, an index radix-select for ties on that value,
// then one compaction pass and a rank sort of the k survivors. Output order is the same as a
// stable descending sort truncated to k (values descending, ties by ascending index), which is
// what the CUB path produced. Replaces a full 248K-vocab segmented radix sort per row for the
// DFlash selector (k = 16): about six passes over the row instead of a full sort.
#define TOPK_SELECT_MAX_K   64
#define TOPK_SELECT_THREADS 1024

static __device__ __forceinline__ uint32_t topk_select_key(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

template <int k_max>
static __global__ void __launch_bounds__(TOPK_SELECT_THREADS)
k_top_k_select(const float * __restrict__ x, int * __restrict__ dst, const int ncols, const int k) {
    const int tid = threadIdx.x;
    x   += (size_t) blockIdx.x * ncols;
    dst += (size_t) blockIdx.x * k;

    __shared__ int      hist[256];
    __shared__ uint32_t s_prefix;
    __shared__ int      s_need;
    __shared__ int      s_eq;
    __shared__ int      s_count;
    __shared__ uint32_t s_keys[k_max];
    __shared__ int      s_idx[k_max];

    // 1) value radix-select, MSB first: after 4 passes `prefix` is the exact key of the k-th largest
    uint32_t prefix = 0, pmask = 0;
    int need = k;
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8*pass;
        if (tid < 256) { hist[tid] = 0; }
        __syncthreads();
        for (int i = tid; i < ncols; i += TOPK_SELECT_THREADS) {
            const uint32_t key = topk_select_key(x[i]);
            if ((key & pmask) == prefix) { atomicAdd(&hist[(key >> shift) & 0xFFu], 1); }
        }
        __syncthreads();
        if (tid == 0) {
            int acc = 0, b = 255;
            for (; b > 0; --b) { const int c = hist[b]; if (acc + c >= need) { break; } acc += c; }
            s_need = need - acc; s_prefix = prefix | ((uint32_t) b << shift);
        }
        __syncthreads();
        need = s_need; prefix = s_prefix; pmask |= (0xFFu << shift);
        __syncthreads();
    }

    // 2) ties on the threshold value: keep the `need` smallest indices among them
    if (tid == 0) { s_eq = 0; }
    __syncthreads();
    int eq = 0;
    for (int i = tid; i < ncols; i += TOPK_SELECT_THREADS) { eq += (topk_select_key(x[i]) == prefix) ? 1 : 0; }
    eq = warp_reduce_sum(eq);
    if ((tid % WARP_SIZE) == 0) { atomicAdd(&s_eq, eq); }
    __syncthreads();
    uint32_t idx_thr = 0xFFFFFFFFu;
    if (s_eq > need) {
        uint32_t iprefix = 0, imask = 0;
        int ineed = need;
        for (int pass = 0; pass < 4; ++pass) {
            const int shift = 24 - 8*pass;
            if (tid < 256) { hist[tid] = 0; }
            __syncthreads();
            for (int i = tid; i < ncols; i += TOPK_SELECT_THREADS) {
                if (topk_select_key(x[i]) == prefix && (((uint32_t) i & imask) == iprefix)) { atomicAdd(&hist[((uint32_t) i >> shift) & 0xFFu], 1); }
            }
            __syncthreads();
            if (tid == 0) {
                int acc = 0, b = 0;
                for (; b < 255; ++b) { const int c = hist[b]; if (acc + c >= ineed) { break; } acc += c; }
                s_need = ineed - acc; s_prefix = iprefix | ((uint32_t) b << shift);
            }
            __syncthreads();
            ineed = s_need; iprefix = s_prefix; imask |= (0xFFu << shift);
            __syncthreads();
        }
        idx_thr = iprefix;
    }

    // 3) compaction of the exactly-k survivors, then a rank sort (k <= k_max)
    if (tid == 0) { s_count = 0; }
    __syncthreads();
    for (int i = tid; i < ncols; i += TOPK_SELECT_THREADS) {
        const uint32_t key = topk_select_key(x[i]);
        if (key > prefix || (key == prefix && (uint32_t) i <= idx_thr)) {
            const int pos = atomicAdd(&s_count, 1);
            if (pos < k_max) { s_keys[pos] = key; s_idx[pos] = i; }
        }
    }
    __syncthreads();
    const int n_found = min(s_count, k);
    if (tid < n_found) {
        const uint32_t mk = s_keys[tid];
        const int      mi = s_idx[tid];
        int rank = 0;
        for (int j = 0; j < n_found; ++j) {
            const uint32_t kj = s_keys[j];
            const int      ij = s_idx[j];
            rank += (kj > mk || (kj == mk && ij < mi)) ? 1 : 0;
        }
        dst[rank] = mi;
    }
}

static void top_k_select_cuda(const float * x, int * dst, const int64_t ncols, const int64_t nrows, const int64_t k, cudaStream_t stream) {
    k_top_k_select<TOPK_SELECT_MAX_K><<<(unsigned) nrows, TOPK_SELECT_THREADS, 0, stream>>>(x, dst, (int) ncols, (int) k);
}


void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();

    // [TAG_TOPK_SELECT] small k over a wide row: exact radix-select instead of a full sort
    if (k <= TOPK_SELECT_MAX_K && ncols >= 2048 && ncols <= INT32_MAX) {
        top_k_select_cuda(src0_d, dst_d, ncols, nrows, k, stream);
        return;
    }
#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        if (use_bitonic) {
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src0_d += ncols * iter_nrows;
        dst_d  += k     * iter_nrows;
    }
#else                             // GGML_CUDA_USE_CUB
    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
    int *                     tmp_dst = temp_dst_alloc.get();
    argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                 cudaMemcpyDeviceToDevice, stream));
#endif
}
