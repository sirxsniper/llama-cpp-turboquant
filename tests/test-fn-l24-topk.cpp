// [TAG_FN_L24_TOPK_TIES] the bitonic argsort rule (ggml/src/ggml-cuda/fn-bitonic-rule.h) on the host: the CUDA kernel's
// network (k_argsort_f32_i32: padded to a power of two, one compare-exchange per (k, j) stage) gives the stable order on
// rows with ties (equal values by index, as the CUB sort and the radix top-k), and the old order on rows without ties.
// CPU only, no model.
#include "../ggml/src/ggml-cuda/fn-bitonic-rule.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

static int g_fail   = 0;
static int g_checks = 0;

#define TCHECK(cond, ...)                      \
    do {                                       \
        g_checks++;                            \
        if (!(cond)) {                         \
            g_fail++;                          \
            fprintf(stderr, "FAIL: ");         \
            fprintf(stderr, __VA_ARGS__);      \
            fprintf(stderr, "\n");             \
        }                                      \
    } while (0)

// the kernel's network; old = the value test alone (before the rule)
template <bool asc>
static std::vector<int> bitonic(const std::vector<float> & x, bool old) {
    const int ncols = (int) x.size();
    int pad = 1;
    while (pad < ncols) {
        pad *= 2;
    }
    std::vector<int> d(pad);
    std::iota(d.begin(), d.end(), 0);
    for (int k = 2; k <= pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            for (int col = 0; col < pad; ++col) {
                const int ixj = col ^ j;
                if (ixj <= col) {
                    continue;
                }
                const bool up = (col & k) == 0;
                bool swap;
                if (up) {
                    swap = d[col] >= ncols || (d[ixj] < ncols && (old ? (asc ? x[d[col]] > x[d[ixj]] : x[d[col]] < x[d[ixj]])
                                                                  : ggml_fn_bitonic_swap<asc>(x[d[col]], x[d[ixj]], d[col], d[ixj], true)));
                } else {
                    swap = d[ixj] >= ncols || (d[col] < ncols && (old ? (asc ? x[d[col]] < x[d[ixj]] : x[d[col]] > x[d[ixj]])
                                                                  : ggml_fn_bitonic_swap<asc>(x[d[col]], x[d[ixj]], d[col], d[ixj], false)));
                }
                if (swap) {
                    std::swap(d[col], d[ixj]);
                }
            }
        }
    }
    d.resize(ncols);
    return d;
}

template <bool asc>
static std::vector<int> stable_ref(const std::vector<float> & x) {
    std::vector<int> d(x.size());
    std::iota(d.begin(), d.end(), 0);
    std::stable_sort(d.begin(), d.end(), [&](int a, int b) { return asc ? x[a] < x[b] : x[a] > x[b]; });
    return d;
}

int main() {
    std::mt19937 rng(1234);
    for (int ncols : { 1, 2, 3, 7, 64, 100, 380, 512, 513, 1000, 1024 }) {
        // ties: many exact zeros (the QSA indexer's rectified scores) and repeated values
        std::vector<float> t(ncols);
        for (int i = 0; i < ncols; ++i) {
            const int r = (int) (rng() % 10);
            t[i] = r < 6 ? 0.0f : (r < 8 ? 1.5f : (float) (rng() % 1000) / 7.0f);
        }
        TCHECK(bitonic<false>(t, false) == stable_ref<false>(t), "desc with ties, ncols %d: the stable order", ncols);
        TCHECK(bitonic<true>(t, false) == stable_ref<true>(t), "asc with ties, ncols %d: the stable order", ncols);
        // -inf ties (masked pools) next to zeros
        std::vector<float> m(t);
        for (int i = 0; i < ncols; i += 3) {
            m[i] = -INFINITY;
        }
        TCHECK(bitonic<false>(m, false) == stable_ref<false>(m), "desc with -inf ties, ncols %d", ncols);
        // no ties: the same permutation as the old rule
        std::vector<float> u(ncols);
        for (int i = 0; i < ncols; ++i) {
            u[i] = (float) i * 0.37f - (float) (ncols / 2);
        }
        std::shuffle(u.begin(), u.end(), rng);
        TCHECK(bitonic<false>(u, false) == bitonic<false>(u, true), "desc without ties, ncols %d: unchanged", ncols);
        TCHECK(bitonic<true>(u, false) == bitonic<true>(u, true), "asc without ties, ncols %d: unchanged", ncols);
        TCHECK(bitonic<false>(u, false) == stable_ref<false>(u), "desc without ties, ncols %d: sorted", ncols);
    }
    // a row with NaNs: the rule adds nothing (NaN compares unequal), so the network does what it did
    std::vector<float> n = { 3.0f, NAN, 1.0f, 2.0f, NAN, 0.5f, 4.0f };
    TCHECK(bitonic<false>(n, false) == bitonic<false>(n, true), "NaN row unchanged");
    printf("test-fn-l24-topk: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
