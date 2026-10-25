#include "common.cuh"
#include "mmid.cuh"

// To reduce shared memory use, store "it" and "iex_used" with 22/10 bits each.
struct mm_ids_helper_store {
    uint32_t data;

    __device__ mm_ids_helper_store(const uint32_t it, const uint32_t iex_used) {
        data = (it & 0x003FFFFF) | (iex_used << 22);
    }

    __device__ uint32_t it() const {
        return data & 0x003FFFFF;
    }

    __device__ uint32_t iex_used() const {
        return data >> 22;
    }
};
static_assert(sizeof(mm_ids_helper_store) == 4, "unexpected size for mm_ids_helper_store");

// the generic path passes 0, which needs no padding since it never groups lanes by token
// [TAG_FN_L21_DUPID] sum over the lanes of a width-wide segment (warp_reduce_sum<width> reduces the whole warp on Ampere+)
template <int width>
static __device__ __forceinline__ int mm_ids_seg_sum(int x) {
    if constexpr (width == 1) {
        return x;
    } else if (width == ggml_cuda_get_physical_warp_size()) {
        return warp_reduce_sum<width>(x);
    } else {
#pragma unroll
        for (int offset = width/2; offset > 0; offset >>= 1) {
            x += __shfl_xor_sync(0xFFFFFFFF, x, offset, width);
        }
        return x;
    }
}

// [TAG_FN_L21_DUPID] inclusive prefix sum over the lanes of the warp
static __device__ __forceinline__ int mm_ids_warp_incl_scan(int x) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
#pragma unroll
    for (int offset = 1; offset < warp_size; offset <<= 1) {
        const int t = __shfl_up_sync(0xFFFFFFFF, x, offset, warp_size);
        if (threadIdx.x >= static_cast<unsigned int>(offset)) {
            x += t;
        }
    }
    return x;
}

template <int n> struct mm_ids_pow2 { static constexpr int value = 2*mm_ids_pow2<(n + 1)/2>::value; };
template <>      struct mm_ids_pow2<1> { static constexpr int value = 1; };
template <>      struct mm_ids_pow2<0> { static constexpr int value = 1; };

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The upper and lower bounds for the ith expert in the compact src1 tensor are stored in expert_bounds[i:i+1].
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    extern __shared__ char data_mm_ids_helper[];
    mm_ids_helper_store * store = (mm_ids_helper_store *) data_mm_ids_helper;

    int nex_prev   = 0; // Number of columns for experts with a lower index.
    int it_compact = 0; // Running index for the compact slice of this expert.
    int dup        = 0; // [TAG_FN_L21_DUPID] a token of this lane used the expert in more than one slot

    if constexpr (n_expert_used_template == 0) {
        // Generic implementation:
        for (int it = 0; it < n_tokens; ++it) {
            int iex_used = -1; // The index at which the expert is used, if any.
            int n_uses   = 0;  // [TAG_FN_L21_DUPID] how often this lane's slots use it
            for (int iex = threadIdx.x; iex < n_expert_used; iex += warp_size) {
                const int expert_used = ids[it*si1 + iex];
                nex_prev += expert_used < expert;
                if (expert_used == expert) {
                    iex_used = iex;
                    n_uses++;
                }
            }

            const int n_uses_it = warp_reduce_sum<warp_size>(n_uses);
            dup |= n_uses_it > 1;

            // [TAG_FN_L21_DUPID] a repeated expert takes the dup pass, which does not read the store (no racing writes)
            if (iex_used != -1 && n_uses_it == 1) {
                store[it_compact] = mm_ids_helper_store(it, iex_used);
            }
            if (n_uses_it > 0) {
                it_compact++;
            }
        }
    } else {
        // Implementation optimized for specific numbers of experts used:
        // a warp holds a whole number of token slots, so the slot count is padded to a power of 2
        constexpr int neu_padded = mm_ids_pow2<n_expert_used_template>::value;
        static_assert(neu_padded <= warp_size && warp_size % neu_padded == 0, "bad n_expert_used");
        for (int it0 = 0; it0 < n_tokens; it0 += warp_size/neu_padded) {
            const int it = it0 + threadIdx.x / neu_padded;

            const int iex = threadIdx.x % neu_padded; // The index at which the expert is used, if any.
            const int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                ids[it*si1 + iex] : INT_MAX;
            const int iex_used = expert_used == expert ? iex : -1;
            nex_prev += expert_used < expert;

            // Whether the threads at this token position have used the expert:
            // [TAG_FN_L21_DUPID] counted, so a token that names the expert in several slots is seen (the dup pass below)
            const int n_uses_it = mm_ids_seg_sum<neu_padded>(iex_used != -1);
            const int it_compact_add_self = n_uses_it > 0;
            dup |= n_uses_it > 1;

            // Do a scan over threads at lower token positions in warp to get the correct index for writing data:
            int it_compact_add_lower = 0;
#pragma unroll
            for (int offset = neu_padded; offset < warp_size; offset += neu_padded) {
                const int tmp = __shfl_up_sync(0xFFFFFFFF, it_compact_add_self, offset, warp_size);
                if (threadIdx.x >= static_cast<unsigned int>(offset)) {
                    it_compact_add_lower += tmp;
                }
            }

            if (iex_used != -1 && n_uses_it == 1) { // [TAG_FN_L21_DUPID] see the generic path
                store[it_compact + it_compact_add_lower] = mm_ids_helper_store(it, iex_used);
            }

            // The thread with the highest index in the warp always has the sum over the whole warp, use it to increment all threads:
            it_compact += __shfl_sync(0xFFFFFFFF, it_compact_add_lower + it_compact_add_self, warp_size - 1, warp_size);
        }
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);
    ggml_cuda_syncwarp();

    // [TAG_FN_L21_DUPID] some token named this expert in more than one slot (upstream #29998's case; ids the caller remaps,
    // e.g. every miss to one zero slot, repeat within a row): the pass above kept one compact row per token, so the other
    // slots' dst rows were never written and their ids_src1 entries (the quantizer's scatter map) were never set (wrong
    // results; an illegal address in the scatter). A second pass gives every (token, slot) that uses the expert its own
    // compact row, in token-major, slot-minor order. It runs only for such experts; without repeats nothing changes.
    if (warp_reduce_any<warp_size>(dup)) {
        int n_rows = 0;
        auto put = [&](const int pos, const int it, const int iex) {
            ids_dst[nex_prev + pos] = it*n_expert_used + iex;
            if (write_inverse) {
                ids_src1[it*n_expert_used + iex] = nex_prev + pos;
            } else {
                ids_src1[nex_prev + pos] = it*sis1 + iex % nchannels_y;
            }
        };
        if constexpr (n_expert_used_template == 0) {
            for (int it = 0; it < n_tokens; ++it) {
                int c = 0;
                for (int iex = threadIdx.x; iex < n_expert_used; iex += warp_size) {
                    c += ids[it*si1 + iex] == expert;
                }
                const int incl = mm_ids_warp_incl_scan(c);
                int pos = n_rows + incl - c;
                for (int iex = threadIdx.x; iex < n_expert_used; iex += warp_size) {
                    if (ids[it*si1 + iex] == expert) {
                        put(pos++, it, iex);
                    }
                }
                n_rows += __shfl_sync(0xFFFFFFFF, incl, warp_size - 1, warp_size);
            }
        } else {
            constexpr int neu_padded = mm_ids_pow2<n_expert_used_template>::value; // [TAG_SYNC_1025] upstream #29940 scope
            for (int it0 = 0; it0 < n_tokens; it0 += warp_size/neu_padded) {
                const int it  = it0 + threadIdx.x / neu_padded;
                const int iex = threadIdx.x % neu_padded;
                const int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                    ids[it*si1 + iex] : INT_MAX;
                const int u    = expert_used == expert;
                const int incl = mm_ids_warp_incl_scan(u);
                if (u) {
                    put(n_rows + incl - 1, it, iex);
                }
                n_rows += __shfl_sync(0xFFFFFFFF, incl, warp_size - 1, warp_size);
            }
        }
        it_compact = n_rows;
    } else
    for (int itc = threadIdx.x; itc < it_compact; itc += warp_size) {
        const mm_ids_helper_store store_it = store[itc];
        const int it       = store_it.it();
        const int iex_used = store_it.iex_used();
        ids_dst[nex_prev + itc] = it*n_expert_used + iex_used;
        // ids_src1 holds the forward map, or the inverse map (token slot -> compact row) for quant dedup
        if (write_inverse) {
            ids_src1[it*n_expert_used + iex_used] = nex_prev + itc;
        } else {
            ids_src1[nex_prev + itc] = it*sis1 + iex_used % nchannels_y;
        }
    }

    if (threadIdx.x != 0) {
        return;
    }

    expert_bounds[expert] = nex_prev;

    if (expert < static_cast<int>(gridDim.x) - 1) {
        return;
    }

    expert_bounds[gridDim.x] = nex_prev + it_compact;
}

template <int n_expert_used_template>
static void launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    GGML_ASSERT(n_tokens          < (1 << 22) && "too few bits in mm_ids_helper_store");
    GGML_ASSERT(n_expert_used_var < (1 << 10) && "too few bits in mm_ids_helper_store");

    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    CUDA_SET_SHARED_MEMORY_LIMIT(mm_ids_helper<n_expert_used_template>, smpbo);

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    const size_t nbytes_shared = n_tokens*sizeof(mm_ids_helper_store);
    GGML_ASSERT(nbytes_shared <= smpbo);
    mm_ids_helper<n_expert_used_template><<<num_blocks, block_size, nbytes_shared, stream>>>
        (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1, write_inverse);
}

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    switch (n_expert_used) {
        case  2:
            launch_mm_ids_helper< 2>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  4:
            launch_mm_ids_helper< 4>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  6:
            launch_mm_ids_helper< 6>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  8:
            launch_mm_ids_helper< 8>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 10:
            launch_mm_ids_helper<10>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 16:
            launch_mm_ids_helper<16>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 32:
            launch_mm_ids_helper<32>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        default:
            launch_mm_ids_helper< 0>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
    }
}
