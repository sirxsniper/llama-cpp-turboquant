#pragma once
// [TAG_FN_L24_TOPK_TIES] the compare-exchange rule of the bitonic argsort (argsort.cu), shared with its CPU test
// (tests/test-fn-l24-topk.cpp). Equal values order by index (the lower index first, in both directions): the bitonic
// network then gives the one sorted order a stable sort gives, the order the CUB radix sort and the radix top-k select
// (lowest index on ties) give for wider rows. Without ties (and with NaNs, which compare unequal) the rule is the old
// value test, so those rows sort exactly as before.

#if defined(__CUDACC__)
#    define GGML_FN_BITONIC_HD __host__ __device__
#else
#    define GGML_FN_BITONIC_HD
#endif

// up: the pair lies in a half sorted in the requested direction ((col & k) == 0); ic / ij: the indices at col / col ^ j
template <bool asc>
static inline GGML_FN_BITONIC_HD bool ggml_fn_bitonic_swap(float xc, float xj, int ic, int ij, bool up) {
    if (up) {
        return (asc ? xc > xj : xc < xj) || (xc == xj && ic > ij);
    }
    return (asc ? xc < xj : xc > xj) || (xc == xj && ic < ij);
}
