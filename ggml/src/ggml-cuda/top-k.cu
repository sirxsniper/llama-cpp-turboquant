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

// [TAG_TOPK_SELECT_WARPHIST] Histogram passes use one private 256-bin histogram per LANE (32 of
// them, stride 257 so the 32 lanes fall into 32 different banks). A warp's 32 atomics therefore go
// to 32 distinct addresses and never serialize against each other; only same-lane atomics from
// different warps meet at the atomic unit. Logits put almost every key into two or three top-byte
// bins, so the single shared histogram this replaces serialized ~150k same-address atomics per pass
// (95 us per row on a 152k-entry vocabulary). Rows are read as float4 when aligned so each thread
// keeps several loads in flight, and the tie count comes straight out of the last radix pass.
// Same selection and output order as before.
#define TOPK_SELECT_NLANE   WARP_SIZE
#define TOPK_SELECT_HSTRIDE 257

// value_mode: bin = byte `shift` of the key, for keys matching (key & pmask) == prefix.
// index mode: bin = byte `shift` of the index, for keys equal to `prefix` whose index matches (i & imask) == iprefix.
// All TOPK_SELECT_THREADS threads call this; the result is left in hist[256].
template <bool value_mode, bool vec4>
static __device__ __forceinline__ void topk_hist_pass(
        const float * __restrict__ x, const int ncols,
        const uint32_t pmask, const uint32_t prefix, const uint32_t imask, const uint32_t iprefix, const int shift,
        int * __restrict__ lhist, int * __restrict__ hist) {
    const int tid  = threadIdx.x;
    int * mine = lhist + (tid % WARP_SIZE) * TOPK_SELECT_HSTRIDE;
    for (int b = tid; b < TOPK_SELECT_NLANE * TOPK_SELECT_HSTRIDE; b += TOPK_SELECT_THREADS) { lhist[b] = 0; }
    __syncthreads();
    const auto add = [&](const uint32_t key, const uint32_t i) {
        if constexpr (value_mode) {
            if ((key & pmask) == prefix) { atomicAdd(mine + ((key >> shift) & 0xFFu), 1); }
        } else {
            if (key == prefix && ((i & imask) == iprefix)) { atomicAdd(mine + ((i >> shift) & 0xFFu), 1); }
        }
    };
    if constexpr (vec4) {
        const float4 * x4 = (const float4 *) x;
        const int n4 = ncols / 4;
        const auto add4 = [&](const float4 v, const int i4) {
            const uint32_t i0 = 4u * (uint32_t) i4;
            add(topk_select_key(v.x), i0);
            add(topk_select_key(v.y), i0 + 1u);
            add(topk_select_key(v.z), i0 + 2u);
            add(topk_select_key(v.w), i0 + 3u);
        };
        int i4 = tid;
        // four independent loads in flight per thread before any histogram work
        for (; i4 + 3 * TOPK_SELECT_THREADS < n4; i4 += 4 * TOPK_SELECT_THREADS) {
            const float4 v0 = x4[i4];
            const float4 v1 = x4[i4 +     TOPK_SELECT_THREADS];
            const float4 v2 = x4[i4 + 2 * TOPK_SELECT_THREADS];
            const float4 v3 = x4[i4 + 3 * TOPK_SELECT_THREADS];
            add4(v0, i4);
            add4(v1, i4 +     TOPK_SELECT_THREADS);
            add4(v2, i4 + 2 * TOPK_SELECT_THREADS);
            add4(v3, i4 + 3 * TOPK_SELECT_THREADS);
        }
        for (; i4 < n4; i4 += TOPK_SELECT_THREADS) {
            add4(x4[i4], i4);
        }
    } else {
        for (int i = tid; i < ncols; i += TOPK_SELECT_THREADS) { add(topk_select_key(x[i]), (uint32_t) i); }
    }
    __syncthreads();
    if (tid < 256) {
        int s = 0;
#pragma unroll
        for (int l = 0; l < TOPK_SELECT_NLANE; ++l) { s += lhist[l * TOPK_SELECT_HSTRIDE + tid]; }
        hist[tid] = s;
    }
    __syncthreads();
}

template <int k_max, bool vec4, bool ordered>
static __global__ void __launch_bounds__(TOPK_SELECT_THREADS)
k_top_k_select(const float * __restrict__ x, int * __restrict__ dst, const int ncols, const int k) {
    const int tid = threadIdx.x;
    x   += (size_t) blockIdx.x * ncols;
    dst += (size_t) blockIdx.x * k;

    __shared__ int      lhist[TOPK_SELECT_NLANE * TOPK_SELECT_HSTRIDE];
    __shared__ int      hist[256];
    __shared__ uint32_t s_prefix;
    __shared__ int      s_need;
    __shared__ int      s_eq;
    __shared__ int      s_count;
    __shared__ uint32_t s_keys[k_max];
    __shared__ int      s_idx[k_max];

    // 1) value radix-select, MSB first: after 4 passes `prefix` is the exact key of the k-th largest,
    //    and the selected bin of the last pass counts the keys equal to it (the ties)
    uint32_t prefix = 0, pmask = 0;
    int need = k;
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8*pass;
        topk_hist_pass<true, vec4>(x, ncols, pmask, prefix, 0u, 0u, shift, lhist, hist);
        if (tid == 0) {
            int acc = 0, b = 255;
            for (; b > 0; --b) { const int c = hist[b]; if (acc + c >= need) { break; } acc += c; }
            s_need = need - acc; s_prefix = prefix | ((uint32_t) b << shift); s_eq = hist[b];
        }
        __syncthreads();
        need = s_need; prefix = s_prefix; pmask |= (0xFFu << shift);
        __syncthreads();
    }

    // 2) ties on the threshold value: keep the `need` smallest indices among them
    uint32_t idx_thr = 0xFFFFFFFFu;
    if (s_eq > need) {
        uint32_t iprefix = 0, imask = 0;
        int ineed = need;
        for (int pass = 0; pass < 4; ++pass) {
            const int shift = 24 - 8*pass;
            topk_hist_pass<false, vec4>(x, ncols, 0u, prefix, imask, iprefix, shift, lhist, hist);
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
    // [TAG_TOPK_WARP_COMPACT] ONE atomicAdd per warp, not one per survivor. The original did
    // atomicAdd(&s_count, 1) for every kept element, which is fine at k = 16 (the DFlash selector
    // this kernel was written for) but serialises catastrophically at k = 2051: every survivor
    // contends on the same shared address, and at decode there is a single row, so a single block
    // on a single SM eats all of it. Measured -56% decode before this, +60% prefill after the
    // unordered path alone. The warp computes its own ballot, the leader reserves the whole run,
    // and each lane derives its slot from a popcount of the lanes below it.
    const auto keep = [&](const float v, const int i) {
        const uint32_t key = topk_select_key(v);
        const bool     hit = (key > prefix || (key == prefix && (uint32_t) i <= idx_thr));
        const uint32_t act = __activemask();          // tail loops leave partial warps
        const uint32_t bal = __ballot_sync(act, hit);
        const int     lead = __ffs(act) - 1;          // lowest ACTIVE lane owns the atomic
        int base = 0;
        if ((int) (threadIdx.x & 31) == lead) { base = atomicAdd(&s_count, __popc(bal)); }
        base = __shfl_sync(act, base, lead);
        if (hit) {
            const int pos = base + __popc(bal & ((1u << (threadIdx.x & 31)) - 1));
            if constexpr (ordered) {
                if (pos < k_max) { s_keys[pos] = key; s_idx[pos] = i; }
            } else {
                // [TAG_TOPK_UNORDERED] no staging and no rank sort: the survivors ARE the answer,
                // so write them out as they are found. This is what lifts the k <= k_max cap.
                if (pos < k) { dst[pos] = i; }
            }
        }
    };
    if constexpr (vec4) {
        const float4 * x4 = (const float4 *) x;
        const int n4 = ncols / 4;
        int i4 = tid;
        for (; i4 + 3 * TOPK_SELECT_THREADS < n4; i4 += 4 * TOPK_SELECT_THREADS) {
            const float4 v0 = x4[i4];
            const float4 v1 = x4[i4 +     TOPK_SELECT_THREADS];
            const float4 v2 = x4[i4 + 2 * TOPK_SELECT_THREADS];
            const float4 v3 = x4[i4 + 3 * TOPK_SELECT_THREADS];
            const float4 vs[4] = { v0, v1, v2, v3 };
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int b = 4 * (i4 + u * TOPK_SELECT_THREADS);
                keep(vs[u].x, b); keep(vs[u].y, b + 1); keep(vs[u].z, b + 2); keep(vs[u].w, b + 3);
            }
        }
        for (; i4 < n4; i4 += TOPK_SELECT_THREADS) {
            const float4 v = x4[i4];
            const int b = 4 * i4;
            keep(v.x, b); keep(v.y, b + 1); keep(v.z, b + 2); keep(v.w, b + 3);
        }
    } else {
        for (int i = tid; i < ncols; i += TOPK_SELECT_THREADS) { keep(x[i], i); }
    }
    __syncthreads();
    if constexpr (ordered) {
        // rank sort of the k survivors; note this indexes by tid, so it is only valid while
        // k <= TOPK_SELECT_THREADS as well as k <= k_max.
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
}

// [TAG_SYNC_TOPK] the fork's launcher must stay outside upstream's HIP-only radix block below.
static void top_k_select_cuda(const float * x, int * dst, const int64_t ncols, const int64_t nrows, const int64_t k, const bool ordered, cudaStream_t stream) {
    // float4 row reads need 16-byte aligned rows: base pointer and row pitch both multiples of 16
    const bool vec4 = (ncols % 4 == 0) && (((uintptr_t) x) % 16 == 0);
    // [TAG_TOPK_UNORDERED] k_max is only the size of the rank-sort staging arrays, so the unordered
    // instantiations ask for 1 element rather than TOPK_SELECT_MAX_K and keep their shared memory
    // to the histograms alone - that is what lets k run into the thousands.
    if (ordered) {
        if (vec4) {
            k_top_k_select<TOPK_SELECT_MAX_K, true,  true><<<(unsigned) nrows, TOPK_SELECT_THREADS, 0, stream>>>(x, dst, (int) ncols, (int) k);
        } else {
            k_top_k_select<TOPK_SELECT_MAX_K, false, true><<<(unsigned) nrows, TOPK_SELECT_THREADS, 0, stream>>>(x, dst, (int) ncols, (int) k);
        }
    } else {
        if (vec4) {
            k_top_k_select<1, true,  false><<<(unsigned) nrows, TOPK_SELECT_THREADS, 0, stream>>>(x, dst, (int) ncols, (int) k);
        } else {
            k_top_k_select<1, false, false><<<(unsigned) nrows, TOPK_SELECT_THREADS, 0, stream>>>(x, dst, (int) ncols, (int) k);
        }
    }
}

#if !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)

static __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

struct top_k_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
    int greater_count;
    int equal_count;
};

static __global__ void top_k_radix_init(top_k_radix_state * states, int nrows, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row] = {0, 0, k, 0, 0};
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    __shared__ int histogram[NBINS];

    histogram[tid] = 0;
    __syncthreads();

    const top_k_radix_state state = states[row];
    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_select(
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];

    int count = 0;
    for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
        const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
        count += block_histograms[offset + tid];
    }
    histogram[tid] = count;
    __syncthreads();

    if (tid == 0) {
        top_k_radix_state state = states[row];
        int bin = NBINS - 1;
        while (bin > 0 && histogram[bin] < state.rank) {
            state.rank -= histogram[bin--];
        }
        state.prefix |= (uint32_t) bin << shift;
        state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
        states[row] = state;
    }
}

static __global__ void top_k_radix_reset_counters(top_k_radix_state * states, int nrows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row].greater_count = 0;
        states[row].equal_count = 0;
    }
}

template<int BLOCK_SIZE>
static __global__ void top_k_radix_gather(
        const float * __restrict__ src,
        int * __restrict__ dst,
        top_k_radix_state * __restrict__ states,
        int ncols,
        int k,
        int blocks_per_row) {
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    top_k_radix_state * state = &states[row];

    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if (key > state->prefix) {
            const int pos = atomicAdd(&state->greater_count, 1);
            row_dst[pos] = col;
        } else if (key == state->prefix) {
            const int pos = atomicAdd(&state->equal_count, 1);
            if (pos < state->rank) {
                row_dst[k - state->rank + pos] = col;
            }
        }
    }
}

static void top_k_radix_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = std::min((ncols + 1023) / 1024, 64);

    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) nrows * blocks_per_row * NBINS);
    top_k_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();

    top_k_radix_init<<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows, k);

    const dim3 row_grid(blocks_per_row * nrows);
    for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
        top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, states, histograms, ncols, blocks_per_row, shift);
        top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
            <<<nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
    }

    top_k_radix_reset_counters
        <<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows);
    top_k_radix_gather<BLOCK_SIZE>
        <<<row_grid, BLOCK_SIZE, 0, stream>>>(
            src, dst, states, ncols, k, blocks_per_row);
}

#endif // !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)

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
    // [TAG_TOPK_UNORDERED] ... and when the caller does not want the order, ANY k qualifies: the
    // k <= TOPK_SELECT_MAX_K ceiling only ever bounded the rank sort, which is now skipped. This is
    // the sparse-attention indexer case (k ~ 2051 over the whole KV row), which otherwise fell all
    // the way through to a full segmented argsort and discarded 99.2% of the result.
    // [TAG_TOPK_UNORDERED] kill switch: TURBO_TOPK_ORDERED=1 forces the old fully-ordered
    // behaviour, so the unordered path can be A/B'd against it in ONE binary (greedy output must
    // be token-identical, since only the ORDER of the selected set changes) and disabled in
    // production without a rebuild if it ever misbehaves.
    static const bool topk_force_ordered = getenv("TURBO_TOPK_ORDERED") != nullptr;
    const bool topk_ordered = topk_force_ordered || dst->op_params[0] == 0;
    if ((k <= TOPK_SELECT_MAX_K || !topk_ordered) && ncols >= 2048 && ncols <= INT32_MAX) {
        top_k_select_cuda(src0_d, dst_d, ncols, nrows, k, topk_ordered, stream);
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
#if defined(GGML_USE_HIP)
    if (ncols > 1024) {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
    } else {
#endif // defined(GGML_USE_HIP)
        ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
        int *                     tmp_dst = temp_dst_alloc.get();
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                     cudaMemcpyDeviceToDevice, stream));
#if defined(GGML_USE_HIP)
    }
#endif // defined(GGML_USE_HIP)
#endif
}
