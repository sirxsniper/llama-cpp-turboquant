// [TAG_FN_L4_HOST_TOPK] common_sampler_fn_l4_topk_row against what the sampler chain does with the full candidate array:
// the logit-bias sampler sets the suppressed ids to -inf, then the top-k sampler std::partial_sort's the K largest to the
// front. The fast path must return exactly those K entries (ids, logits, order) or refuse (false); a refusal is only
// allowed when the order is not unique (equal values among the K or at the K-th place), on a NaN, or with < K values.

#include "sampling.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static int n_fail = 0;

static void check(bool ok, const char * what, int iter) {
    if (!ok) {
        printf("FAIL: %s (case %d)\n", what, iter);
        n_fail++;
    }
}

// the reference: set_logits + logit-bias (-inf) + top-k's partial sort
static std::vector<llama_token_data> ref_topk(const std::vector<float> & x, int K, const std::vector<llama_token> & excl) {
    std::vector<llama_token_data> cur(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        cur[i] = llama_token_data{ (llama_token) i, x[i], 0.0f };
    }
    for (llama_token t : excl) {
        cur[t].logit += -INFINITY;
    }
    const int k = std::min<int>(K, (int) cur.size());
    std::partial_sort(cur.begin(), cur.begin() + k, cur.end(),
            [](const llama_token_data & a, const llama_token_data & b) { return a.logit > b.logit; });
    cur.resize(k);
    return cur;
}

// a refusal must have a reason in the data
static bool refusal_justified(const std::vector<float> & x, int K, const std::vector<llama_token> & excl) {
    std::vector<float> v;
    for (size_t i = 0; i < x.size(); ++i) {
        if (std::isnan(x[i])) {
            return true;
        }
        if (!std::binary_search(excl.begin(), excl.end(), (llama_token) i)) {
            v.push_back(x[i]);
        }
    }
    if ((int) v.size() < K) {
        return true;
    }
    std::sort(v.begin(), v.end(), [](float a, float b) { return a > b; });
    if (v[K - 1] == -INFINITY) {
        return true;
    }
    for (int j = 1; j < K; ++j) {
        if (v[j - 1] == v[j]) {
            return true;
        }
    }
    return (int) v.size() > K && v[K] == v[K - 1];
}

int main() {
    std::mt19937 rng(1234);
    const int sizes[] = { 1, 5, 20, 33, 100, 1000, 4097, 248320 };
    const int ks[]    = { 1, 2, 10, 20, 40, 128 };
    int n_cases = 0, n_fast = 0;
    for (int it = 0; it < 4000; ++it) {
        const int n = sizes[it % 8];
        const int K = ks[(it / 8) % 6];
        const int mode = (it / 48) % 6; // 0 normal, 1 coarse (ties), 2 NaN, 3 excluded ids, 4 -inf heavy, 5 all equal
        std::vector<float> x(n);
        std::normal_distribution<float> nd(0.0f, 4.0f);
        for (auto & v : x) {
            v = nd(rng);
            if (mode == 1) {
                v = std::round(v * 2.0f) / 2.0f;
            }
            if (mode == 4 && (rng() % 4 != 0)) {
                v = -INFINITY;
            }
            if (mode == 5) {
                v = 1.5f;
            }
        }
        if (mode == 2 && n > 3) {
            x[rng() % n] = NAN;
        }
        std::vector<llama_token> excl;
        if (mode == 3) {
            // the largest values among the suppressed
            std::vector<int> idx(n);
            for (int i = 0; i < n; ++i) {
                idx[i] = i;
            }
            std::sort(idx.begin(), idx.end(), [&](int a, int b) { return x[a] > x[b]; });
            for (int i = 0; i < std::min(n, 3); ++i) {
                excl.push_back(idx[i * 2 < n ? i * 2 : i]);
            }
            std::sort(excl.begin(), excl.end());
            excl.erase(std::unique(excl.begin(), excl.end()), excl.end());
        }
        std::vector<llama_token_data> fast;
        const bool ok = common_sampler_fn_l4_topk_row(x.data(), n, K, excl, fast);
        n_cases++;
        if (ok) {
            n_fast++;
            const auto ref = ref_topk(x, K, excl);
            bool same = ref.size() == fast.size();
            for (size_t j = 0; same && j < ref.size(); ++j) {
                same = ref[j].id == fast[j].id && ref[j].logit == fast[j].logit && fast[j].p == 0.0f;
            }
            check(same, "fast result differs from the partial sort", it);
        } else {
            check(refusal_justified(x, K, excl), "refused without a tie, NaN or too few values", it);
        }
    }
    printf("%d cases, %d on the fast path, %d failures\n", n_cases, n_fast, n_fail);

    // timing at the Flash-Next vocabulary (248320), K 20: the fast path vs set_logits + partial sort
    {
        const int n = 248320, K = 20;
        std::vector<float> x(n);
        std::normal_distribution<float> nd(0.0f, 4.0f);
        for (auto & v : x) {
            v = nd(rng);
        }
        std::vector<llama_token> excl;
        std::vector<llama_token_data> out;
        std::vector<llama_token_data> cur(n);
        double t_fast = 1e30, t_ref = 1e30;
        for (int r = 0; r < 20; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            const bool ok = common_sampler_fn_l4_topk_row(x.data(), n, K, excl, out);
            auto t1 = std::chrono::steady_clock::now();
            for (int i = 0; i < n; ++i) {
                cur[i] = llama_token_data{ i, x[i], 0.0f };
            }
            std::partial_sort(cur.begin(), cur.begin() + K, cur.end(),
                    [](const llama_token_data & a, const llama_token_data & b) { return a.logit > b.logit; });
            auto t2 = std::chrono::steady_clock::now();
            check(ok && out[0].id == cur[0].id && out[K - 1].id == cur[K - 1].id, "timing case", r);
            t_fast = std::min(t_fast, std::chrono::duration<double, std::micro>(t1 - t0).count());
            t_ref  = std::min(t_ref,  std::chrono::duration<double, std::micro>(t2 - t1).count());
        }
        printf("n_vocab %d, K %d: fast %.1f us, set_logits + partial sort %.1f us (best of 20)\n", n, K, t_fast, t_ref);
    }
    printf("%s\n", n_fail == 0 ? "PASS" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}
