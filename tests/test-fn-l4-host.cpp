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

    // [TAG_R35_HOSTCHAIN] common_sampler_r35_topk_cands from the row's exact top m (value desc, index asc: what ggml_top_k
    // selects), shuffled: when it decides, the row's fast path must decide too and give the same entries; the device gives
    // no candidates for a row with a NaN or an inf (its chunk sums are not finite), so those rows are not asked
    {
        int n_c = 0, n_c_fast = 0, n_c_ref_fast = 0;
        const int ms[] = { 64, 32, 21 };
        for (int it = 0; it < 3150; ++it) {
            const int n = it % 3 == 0 ? 248320 : (it % 3 == 1 ? 4097 : 300);
            const int K = ks[(it / 3) % 5];               // 1, 2, 10, 20, 40
            const int m = ms[(it / 15) % 3];
            const int mode = (it / 45) % 7; // 0 normal, 1 coarse (ties), 2 excluded ids, 3 all equal, 4 ties at the m-th place,
                                            // 5 very coarse, 6 many excluded among the best
            std::vector<float> x(n);
            std::normal_distribution<float> nd(0.0f, 4.0f);
            for (auto & v : x) {
                v = nd(rng);
                if (mode == 1) {
                    v = std::round(v * 2.0f) / 2.0f;
                }
                if (mode == 5) {
                    v = std::round(v / 4.0f);
                }
                if (mode == 3) {
                    v = 1.5f;
                }
            }
            std::vector<int> idx(n);
            for (int i = 0; i < n; ++i) {
                idx[i] = i;
            }
            std::sort(idx.begin(), idx.end(), [&](int a, int b) { return x[a] > x[b] || (x[a] == x[b] && a < b); });
            if (mode == 4 && n > m + 2 && K + 1 < m) {
                // the K-th best equals the m-th best: undecidable from the candidates
                const float v = x[idx[K - 1]];
                for (int j = K - 1; j <= m + 1; ++j) {
                    x[idx[j]] = v;
                }
                std::sort(idx.begin(), idx.end(), [&](int a, int b) { return x[a] > x[b] || (x[a] == x[b] && a < b); });
            }
            std::vector<llama_token> excl;
            if (mode == 2 || mode == 6) {
                const int ne = mode == 2 ? 3 : std::min(m - 1, K + 8);
                for (int i = 0; i < ne && i < n; ++i) {
                    excl.push_back(idx[mode == 2 ? std::min(n - 1, i * 2) : i]);
                }
                std::sort(excl.begin(), excl.end());
                excl.erase(std::unique(excl.begin(), excl.end()), excl.end());
            }
            if (m > n) {
                continue;
            }
            std::vector<float>       cv(m);
            std::vector<llama_token> ci(m);
            std::vector<int>         perm(m);
            for (int j = 0; j < m; ++j) {
                perm[j] = j;
            }
            std::shuffle(perm.begin(), perm.end(), rng);
            for (int j = 0; j < m; ++j) {
                ci[j] = idx[perm[j]];
                cv[j] = x[ci[j]];
            }
            std::vector<llama_token_data> c_out, r_out;
            const bool c_ok = common_sampler_r35_topk_cands(cv.data(), ci.data(), m, K, excl, c_out);
            const bool r_ok = common_sampler_fn_l4_topk_row(x.data(), n, K, excl, r_out);
            n_c++;
            n_c_fast     += c_ok ? 1 : 0;
            n_c_ref_fast += r_ok ? 1 : 0;
            if (c_ok) {
                bool same = r_ok && r_out.size() == c_out.size();
                for (size_t j = 0; same && j < r_out.size(); ++j) {
                    same = r_out[j].id == c_out[j].id && r_out[j].logit == c_out[j].logit && c_out[j].p == 0.0f;
                }
                check(same, "r35 candidates decided a row the row path decides differently (or refuses)", it);
                const auto ref = ref_topk(x, K, excl);
                bool same_ref = ref.size() == c_out.size();
                for (size_t j = 0; same_ref && j < ref.size(); ++j) {
                    same_ref = ref[j].id == c_out[j].id && ref[j].logit == c_out[j].logit;
                }
                check(same_ref, "r35 candidates differ from the partial sort", it);
            } else if (mode == 0 && r_ok && K < m - 1) {
                check(false, "r35 candidates refused a continuous row with room", it);
            }
        }
        // NaN among the candidates refuses
        {
            float v[4] = { 3.0f, NAN, 1.0f, 0.5f };
            llama_token t[4] = { 7, 8, 9, 10 };
            std::vector<llama_token_data> o;
            check(!common_sampler_r35_topk_cands(v, t, 4, 2, {}, o), "r35 candidates with a NaN decided", -1);
        }
        printf("r35 candidates: %d rows, %d decided from the candidates, %d by the row path, %d failures so far\n",
                n_c, n_c_fast, n_c_ref_fast, n_fail);
    }

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
