#include "sampling.h"
#include "speculative-block.h" // [TAG_FN_L3_MTP_BLOCK]

#include "common.h"
#include "fit.h"
#include "log.h"
#include "reasoning-budget.h"
#include "speculative.h"

#include "ggml.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <emmintrin.h>
#define FN_L4_TOPK_SSE2 1
#endif

#include "../src/llama-ext.h" // [TAG_FN_L4_HOST_TOPK] llama_fn_l4_host_flag

// [TAG_FN_L4_HOST_TOPK] the K largest logits of a row (ids in excl skipped) in strictly descending order, i.e. exactly
// what std::partial_sort leaves at the front of the full candidate array. False when that order is not unique (equal
// values among the K or at the K-th place), on a NaN, or when fewer than K candidates exist: the caller then takes
// the full-vocabulary path, so the result never differs from it.
bool common_sampler_fn_l4_topk_row(const float * x, int n, int K, const std::vector<llama_token> & excl,
        std::vector<llama_token_data> & out) {
    if (K <= 0 || K > 128 || n < K) {
        return false;
    }
    llama_token_data top[128];
    int   m        = 0;
    float thr      = -INFINITY; // top[K-1].logit once m == K
    float tie      = -INFINITY; // the largest value dropped while equal to or below the K-th of that time
    bool  have_tie = false;
    bool  nan      = false;

    auto is_excl = [&](int id) {
        return !excl.empty() && std::binary_search(excl.begin(), excl.end(), (llama_token) id);
    };
    auto note_tie = [&](float v) {
        if (!have_tie || v > tie) {
            tie      = v;
            have_tie = true;
        }
    };
    auto consider = [&](int i, float v) {
        if (v != v) {
            nan = true;
            return;
        }
        if (m == K && !(v > thr)) {
            if (v == thr && !is_excl(i)) {
                note_tie(v);
            }
            return;
        }
        if (is_excl(i)) {
            return;
        }
        if (m == K) {
            note_tie(top[K - 1].logit);
            m--;
        }
        int j = m;
        while (j > 0 && top[j - 1].logit < v) {
            top[j] = top[j - 1];
            j--;
        }
        top[j] = llama_token_data{ (llama_token) i, v, 0.0f };
        m++;
        if (m == K) {
            thr = top[K - 1].logit;
        }
    };

    int i = 0;
    for (; i < n && m < K; ++i) {
        consider(i, x[i]);
    }
#if defined(FN_L4_TOPK_SSE2)
    __m128 vt = _mm_set1_ps(thr);
    for (; i + 16 <= n; i += 16) {
        const __m128 a = _mm_loadu_ps(x + i);
        const __m128 b = _mm_loadu_ps(x + i + 4);
        const __m128 c = _mm_loadu_ps(x + i + 8);
        const __m128 d = _mm_loadu_ps(x + i + 12);
        // at or above the K-th, or unordered (NaN)
        const int mask = _mm_movemask_ps(_mm_or_ps(_mm_cmpge_ps(a, vt), _mm_cmpunord_ps(a, a))) |
                         _mm_movemask_ps(_mm_or_ps(_mm_cmpge_ps(b, vt), _mm_cmpunord_ps(b, b))) << 4 |
                         _mm_movemask_ps(_mm_or_ps(_mm_cmpge_ps(c, vt), _mm_cmpunord_ps(c, c))) << 8 |
                         _mm_movemask_ps(_mm_or_ps(_mm_cmpge_ps(d, vt), _mm_cmpunord_ps(d, d))) << 12;
        if (mask != 0) {
            for (int k = 0; k < 16; ++k) {
                if (mask & (1 << k)) {
                    consider(i + k, x[i + k]);
                }
            }
            vt = _mm_set1_ps(thr);
        }
    }
#endif
    for (; i < n; ++i) {
        consider(i, x[i]);
    }

    if (nan || m < K || top[K - 1].logit == -INFINITY) {
        return false;
    }
    for (int j = 1; j < K; ++j) {
        if (!(top[j - 1].logit > top[j].logit)) {
            return false;
        }
    }
    if (have_tie && tie == top[K - 1].logit) {
        return false;
    }
    out.assign(top, top + K);
    return true;
}

// [TAG_R28_PENFAST] LLAMA_R28_PENFAST: 1 (default) the fast candidate path with penalties / a logit bias, 0 off (v0.26.3)
static bool r28_penfast_enabled() {
    static const bool on = [] {
        const char * e = std::getenv("LLAMA_R28_PENFAST");
        return !(e != nullptr && e[0] != 0 && std::atoi(e) == 0);
    }();
    return on;
}

// [TAG_R29_GRAMFAST] LLAMA_R29_GRAMFAST: 1 (default) the candidate-set path with a grammar / a reasoning budget, 0 off (v0.26.4)
static bool r29_gramfast_enabled() {
    static const bool on = [] {
        const char * e = std::getenv("LLAMA_R29_GRAMFAST");
        return !(e != nullptr && e[0] != 0 && std::atoi(e) == 0);
    }();
    return on;
}

// [TAG_R28_PENFAST] the most moved tokens (penalty windows + biased tokens) the fast path takes; penalty_last_n is 64 by default
static constexpr size_t R28_PENFAST_MAX_MOVED = 1024;

// [TAG_R28_PENFAST] see sampling.h
common_penfast_cfg common_sampler_penfast_cfg(const common_params_sampling & params, struct llama_sampler * chain,
        const llama_token * suppress, int32_t n_suppress, int32_t n_vocab) {
    // mirostat builds no top-k; token probabilities to report read more than the chain leaves; top_k <= 128: the
    // std::partial_sort branch of the top-k sampler and the limit of common_sampler_fn_l4_topk_row
    if (chain == nullptr || params.mirostat != 0 || params.n_probs != 0 || params.top_k < 1 || params.top_k > 128 ||
            n_vocab <= params.top_k) {
        return common_penfast_cfg();
    }

    // the logit bias as common_sampler_init merges it (user biases + suppress tokens). A token listed twice gets only its
    // first bias once the candidates are not in id order (llama_sampler_logit_bias_apply): full path.
    std::vector<llama_token> fixed;
    fixed.reserve(params.logit_bias.size() + (size_t) std::max(n_suppress, 0));
    for (const auto & lb : params.logit_bias) {
        fixed.push_back(lb.token);
    }
    for (int32_t i = 0; i < n_suppress; ++i) {
        fixed.push_back(suppress[i]);
    }
    std::sort(fixed.begin(), fixed.end());
    if (std::adjacent_find(fixed.begin(), fixed.end()) != fixed.end()) {
        return common_penfast_cfg();
    }
    // ids outside the vocabulary match no candidate in either pass
    fixed.erase(std::remove_if(fixed.begin(), fixed.end(), [n_vocab](llama_token t) { return t < 0 || t >= n_vocab; }),
            fixed.end());
    if (fixed.size() > R28_PENFAST_MAX_MOVED) {
        return common_penfast_cfg();
    }

    // the samplers before the first top-k: the logit bias, penalties, and samplers that are no-ops for these parameters
    // (llama_sampler_init_empty names them "?..."). Anything else before it (DRY, top-n-sigma, typical, XTC,
    // temperature, top-p, min-p, infill) keeps the full path; after it every sampler sees identical candidates.
    std::vector<int32_t> i_pen;
    const int32_t n = llama_sampler_chain_n(chain);
    for (int32_t i = 0; i < n; ++i) {
        const char * nm = llama_sampler_name(llama_sampler_chain_get(chain, i));
        std::string s = nm != nullptr ? nm : "";
        if (!s.empty() && (s[0] == '+' || s[0] == '-')) {
            s.erase(0, 1); // the backend's support marker
        }
        if (s == "top-k") {
            common_penfast_cfg cfg;
            cfg.top_k   = params.top_k;
            cfg.i_top_k = i;
            cfg.i_pen   = std::move(i_pen);
            cfg.fixed   = std::move(fixed);
            return cfg;
        }
        if (s == "penalties") {
            i_pen.push_back(i);
            continue;
        }
        if (s == "logit-bias" || s == "?penalties" || s == "?dry" || s == "?top-n-sigma") {
            continue;
        }
        break;
    }
    return common_penfast_cfg();
}

// [TAG_R28_PENFAST] the tokens the samplers before the top-k may move: the biased ones and every penalty window; sorted,
// unique, in the vocabulary. False: more than R28_PENFAST_MAX_MOVED of them or a window that cannot be read
static bool penfast_moved(const common_penfast_cfg & cfg, struct llama_sampler * chain, int n_vocab,
        std::vector<llama_token> & excl) {
    excl.assign(cfg.fixed.begin(), cfg.fixed.end());
    for (const int32_t i : cfg.i_pen) {
        const llama_sampler * pen = llama_sampler_chain_get(chain, i);
        const int32_t nw = llama_sampler_penalties_window(pen, nullptr, 0);
        if (nw < 0 || excl.size() + (size_t) nw > R28_PENFAST_MAX_MOVED) {
            return false;
        }
        const size_t at = excl.size();
        excl.resize(at + (size_t) nw);
        if (nw > 0 && llama_sampler_penalties_window(pen, excl.data() + at, nw) != nw) {
            return false;
        }
    }
    std::sort(excl.begin(), excl.end());
    excl.erase(std::unique(excl.begin(), excl.end()), excl.end());
    excl.erase(std::remove_if(excl.begin(), excl.end(), [n_vocab](llama_token t) { return t < 0 || t >= n_vocab; }),
            excl.end());
    return true;
}

// [TAG_R28_PENFAST] see sampling.h. Exactness: the samplers before the top-k move only the tokens in excl, so every token
// outside the candidate set keeps its raw logit, which common_sampler_fn_l4_topk_row leaves strictly below the K-th of
// the K best raw logits outside excl; those K are in the set, unmoved. The chain's top-k set over the full row is then
// inside the set, and when the K best logits of the set are strictly ordered and strictly above the rest of it, the
// top-k sampler leaves the same K candidates in the same order in both passes; every sampler after it (top-p, min-p,
// temperature, dist with its RNG) then sees identical input.
bool common_sampler_penfast_apply(const common_penfast_cfg & cfg, struct llama_sampler * chain, const float * logits,
        int n_vocab, bool after_top_k, std::vector<llama_token> & excl, std::vector<llama_token_data> & cur,
        llama_token_data_array & cur_p) {
    if (cfg.top_k <= 0 || cfg.i_top_k < 0 || chain == nullptr || logits == nullptr || n_vocab <= cfg.top_k) {
        return false;
    }

    // the tokens the samplers before the top-k may move: the biased ones and every penalty window
    if (!penfast_moved(cfg, chain, n_vocab, excl)) {
        return false;
    }

    // the K best raw logits outside excl, strictly descending, no tie at the K-th place, no NaN anywhere in the row
    if (!common_sampler_fn_l4_topk_row(logits, n_vocab, cfg.top_k, excl, cur)) {
        return false;
    }
    // a +inf raw logit: the full path (where the NaN guard of common_sampler_sample sees the row)
    if (std::isinf(cur[0].logit)) {
        return false;
    }
    cur.reserve(cur.size() + excl.size());
    for (const llama_token t : excl) {
        const float v = logits[t];
        if (std::isnan(v) || (std::isinf(v) && v > 0.0f)) {
            return false;
        }
        cur.push_back(llama_token_data{ t, v, 0.0f });
    }
    cur_p = { cur.data(), cur.size(), -1, false };

    // the logit bias, the penalties and the no-ops before the top-k; their apply changes no state
    llama_sampler_chain_apply_range(chain, &cur_p, 0, cfg.i_top_k);

    // the top-k order over the set is unique: the K best strictly descending, the K-th strictly above the next
    const size_t K = (size_t) cfg.top_k;
    std::vector<float> v(cur_p.size);
    for (size_t i = 0; i < cur_p.size; ++i) {
        v[i] = cur_p.data[i].logit;
        if (std::isnan(v[i])) {
            return false;
        }
    }
    const size_t m = std::min(K + 1, v.size());
    std::partial_sort(v.begin(), v.begin() + (std::ptrdiff_t) m, v.end(), std::greater<float>());
    for (size_t j = 1; j < m; ++j) {
        if (!(v[j - 1] > v[j])) {
            return false;
        }
    }

    llama_sampler_chain_apply_range(chain, &cur_p, cfg.i_top_k, after_top_k ? cfg.i_top_k + 1 : INT32_MAX);
    return true;
}

// [TAG_R29_GRAMFAST] the widths of the two candidate sets: 4 x top_k (at least 128), then the v0.25.5 pre-trim width
static size_t r29_width(int32_t top_k, int pass) {
    return pass == 0 ? std::max<size_t>(128, 4 * (size_t) top_k) : std::max<size_t>(1024, 32 * (size_t) top_k);
}

// [TAG_R29_GRAMFAST] every token outside excl (sorted) whose raw logit is at least theta, the M-th largest of them, in no
// order. All ties at theta are taken, so every other token outside excl lies strictly below theta. False on a NaN or +inf
// anywhere in the row, with fewer than M tokens outside excl, theta == -inf, or more than 4 x M tokens at or above theta.
static bool r29_select_row(const float * x, int n, size_t M, const std::vector<llama_token> & excl,
        std::vector<llama_token_data> & out, float & theta) {
    out.clear();
    if (M == 0 || n <= 0 || (size_t) n < M) {
        return false;
    }
    // a min-heap of the M best so far, front() the smallest
    const auto heap_cmp = [](const llama_token_data & a, const llama_token_data & b) { return a.logit > b.logit; };
    out.reserve(M);
    float  thr      = -INFINITY; // front().logit once the heap is full
    bool   full     = false;
    float  tie      = -INFINITY; // the largest value outside excl dropped while equal to or below the M-th of that time
    bool   have_tie = false;
    bool   bad      = false;
    size_t ie       = 0;         // excl walks along with the ascending ids

    auto note_tie = [&](float v) {
        if (!have_tie || v > tie) {
            tie      = v;
            have_tie = true;
        }
    };
    auto consider = [&](int i, float v) {
        if (v != v || v == INFINITY) {
            bad = true;
            return;
        }
        if (full && v < thr) {
            return;
        }
        while (ie < excl.size() && excl[ie] < i) {
            ++ie;
        }
        if (ie < excl.size() && excl[ie] == i) {
            return;
        }
        if (!full) {
            out.push_back(llama_token_data{ (llama_token) i, v, 0.0f });
            std::push_heap(out.begin(), out.end(), heap_cmp);
            if (out.size() == M) {
                full = true;
                thr  = out.front().logit;
            }
            return;
        }
        if (v == thr) {
            note_tie(v);
            return;
        }
        // v > thr: the smallest kept one drops out
        note_tie(out.front().logit);
        std::pop_heap(out.begin(), out.end(), heap_cmp);
        out.back() = llama_token_data{ (llama_token) i, v, 0.0f };
        std::push_heap(out.begin(), out.end(), heap_cmp);
        thr = out.front().logit;
    };

    int i = 0;
    for (; i < n && !full && !bad; ++i) {
        consider(i, x[i]);
    }
#if defined(FN_L4_TOPK_SSE2)
    __m128 vt = _mm_set1_ps(thr);
    for (; i + 16 <= n && !bad; i += 16) {
        const __m128 a = _mm_loadu_ps(x + i);
        const __m128 b = _mm_loadu_ps(x + i + 4);
        const __m128 c = _mm_loadu_ps(x + i + 8);
        const __m128 d = _mm_loadu_ps(x + i + 12);
        // at or above the M-th, or unordered (NaN)
        const int mask = _mm_movemask_ps(_mm_or_ps(_mm_cmpge_ps(a, vt), _mm_cmpunord_ps(a, a))) |
                         _mm_movemask_ps(_mm_or_ps(_mm_cmpge_ps(b, vt), _mm_cmpunord_ps(b, b))) << 4 |
                         _mm_movemask_ps(_mm_or_ps(_mm_cmpge_ps(c, vt), _mm_cmpunord_ps(c, c))) << 8 |
                         _mm_movemask_ps(_mm_or_ps(_mm_cmpge_ps(d, vt), _mm_cmpunord_ps(d, d))) << 12;
        if (mask != 0) {
            for (int k = 0; k < 16; ++k) {
                if (mask & (1 << k)) {
                    consider(i + k, x[i + k]);
                }
            }
            vt = _mm_set1_ps(thr);
        }
    }
#endif
    for (; i < n && !bad; ++i) {
        consider(i, x[i]);
    }
    if (bad || !full || thr == -INFINITY) {
        return false;
    }
    theta = thr;
    if (have_tie && tie == theta) {
        // more tokens at theta than the heap holds: take them all
        out.clear();
        ie = 0;
        for (int j = 0; j < n; ++j) {
            if (!(x[j] >= theta)) {
                continue;
            }
            while (ie < excl.size() && excl[ie] < j) {
                ++ie;
            }
            if (ie < excl.size() && excl[ie] == j) {
                continue;
            }
            if (out.size() >= 4 * M) {
                return false;
            }
            out.push_back(llama_token_data{ (llama_token) j, x[j], 0.0f });
        }
    }
    return true;
}

// [TAG_R29_GRAMFAST] the chain's top-k over the whole row keeps the same tokens in the same order as over cur_p: the K best
// of cur_p strictly descending, the K-th strictly above the next one of cur_p and at or above theta (every token outside
// cur_p lies strictly below theta, or at -inf after the grammar)
static bool r29_top_k_exact(const llama_token_data_array & cur_p, size_t K, float theta, std::vector<float> & v) {
    if (K == 0 || cur_p.size < K) {
        return false;
    }
    v.resize(cur_p.size);
    for (size_t i = 0; i < cur_p.size; ++i) {
        v[i] = cur_p.data[i].logit;
        if (std::isnan(v[i])) {
            return false;
        }
    }
    const size_t m = std::min(K + 1, v.size());
    std::partial_sort(v.begin(), v.begin() + (std::ptrdiff_t) m, v.end(), std::greater<float>());
    for (size_t j = 1; j < m; ++j) {
        if (!(v[j - 1] > v[j])) {
            return false;
        }
    }
    return v[K - 1] >= theta;
}

static size_t r29_count_finite(const llama_token_data * d, size_t n) {
    size_t c = 0;
    for (size_t i = 0; i < n; ++i) {
        c += d[i].logit != -INFINITY ? 1 : 0;
    }
    return c;
}

// [TAG_R29_GRAMFAST] see sampling.h. The candidate array is [first set outside E | E | rest of the second set]: the second
// width appends its extra tokens and runs the grammar over them only, the samplers before the top-k move no token outside E.
bool common_sampler_gramfast_apply(const common_penfast_cfg & cfg, struct llama_sampler * chain, struct llama_sampler * grmr,
        const float * logits, int n_vocab, bool after_top_k, common_gramfast_scratch & sc,
        std::vector<llama_token_data> & cur, llama_token_data_array & cur_p) {
    sc.width  = 0;
    sc.n_fin  = SIZE_MAX;
    sc.n_pass = 0;
    if (cfg.top_k <= 0 || cfg.i_top_k < 0 || chain == nullptr || logits == nullptr || n_vocab <= cfg.top_k) {
        return false;
    }
    if (!penfast_moved(cfg, chain, n_vocab, sc.excl)) {
        return false;
    }
    const size_t K  = (size_t) cfg.top_k;
    const size_t ne = sc.excl.size();
    const size_t w1 = r29_width(cfg.top_k, 0);
    const size_t w2 = r29_width(cfg.top_k, 1);
    // a set of more than half the tokens outside E saves little
    if ((size_t) n_vocab < ne || 2 * w2 > (size_t) n_vocab - ne) {
        return false;
    }

    float theta2 = -INFINITY;
    if (!r29_select_row(logits, n_vocab, w2, sc.excl, sc.top, theta2)) {
        return false;
    }
    // the first set: the w1 best of them, ties at its theta all taken (every other token outside E lies below it)
    const size_t nt = sc.top.size();
    sc.v.resize(nt);
    for (size_t i = 0; i < nt; ++i) {
        sc.v[i] = sc.top[i].logit;
    }
    std::nth_element(sc.v.begin(), sc.v.begin() + (std::ptrdiff_t) (w1 - 1), sc.v.end(), std::greater<float>());
    const float theta1 = sc.v[w1 - 1];
    const size_t n1 = (size_t) (std::partition(sc.top.begin(), sc.top.end(),
            [theta1](const llama_token_data & d) { return d.logit >= theta1; }) - sc.top.begin());

    cur.resize(nt + ne);
    std::copy(sc.top.begin(), sc.top.begin() + (std::ptrdiff_t) n1, cur.begin());
    for (size_t j = 0; j < ne; ++j) {
        const llama_token t = sc.excl[j];
        cur[n1 + j] = llama_token_data{ t, logits[t], 0.0f }; // finite or -inf: r29_select_row saw the whole row
    }
    std::copy(sc.top.begin() + (std::ptrdiff_t) n1, sc.top.end(), cur.begin() + (std::ptrdiff_t) (n1 + ne));

    // the first set: the grammar, then the logit bias, the penalties and the no-ops before the top-k (no state changes)
    cur_p = { cur.data(), n1 + ne, -1, false };
    llama_sampler_apply(grmr, &cur_p);
    size_t n_fin = r29_count_finite(cur_p.data, cur_p.size);
    sc.width = nt > n1 ? w1 : w2; // the first set is the second one when its theta is theta2
    sc.n_fin = n_fin;
    llama_sampler_chain_apply_range(chain, &cur_p, 0, cfg.i_top_k);

    int pass = 0;
    if (r29_top_k_exact(cur_p, K, theta1, sc.v)) {
        pass = 1;
    } else if (nt > n1) {
        // the second set: the grammar over the appended tokens only
        llama_token_data_array rest = { cur.data() + n1 + ne, nt - n1, -1, false };
        llama_sampler_apply(grmr, &rest);
        n_fin += r29_count_finite(rest.data, rest.size);
        sc.width = w2;
        sc.n_fin = n_fin;
        cur_p = { cur.data(), nt + ne, -1, false };
        if (r29_top_k_exact(cur_p, K, theta2, sc.v)) {
            pass = 2;
        }
    }
    if (pass == 0) {
        return false;
    }
    sc.n_pass = pass;
    llama_sampler_chain_apply_range(chain, &cur_p, cfg.i_top_k, after_top_k ? cfg.i_top_k + 1 : INT32_MAX);
    return true;
}

// [TAG_R29_GRAMFAST] see sampling.h: the steps of common_sampler_sample, each over a candidate set
common_gramfast_result common_sampler_gramfast_sample(const common_penfast_cfg & cfg, struct llama_sampler * chain,
        struct llama_sampler * grmr, bool grammar_first, bool use_grammar, const float * logits, int n_vocab,
        common_gramfast_scratch & sc, std::vector<llama_token_data> & cur, llama_token_data_array & cur_p, llama_token * id) {
    sc.width  = 0;
    sc.n_fin  = SIZE_MAX;
    sc.n_pass = 0;
    const bool constrain = use_grammar && grmr != nullptr;
    if (grammar_first && constrain) {
        if (!common_sampler_gramfast_apply(cfg, chain, grmr, logits, n_vocab, false, sc, cur, cur_p)) {
            return COMMON_GRAMFAST_NONE;
        }
        *id = cur_p.data[cur_p.selected].id;
        return COMMON_GRAMFAST_DONE;
    }

    // the chain alone, then the sampled token against the grammar
    if (!common_sampler_penfast_apply(cfg, chain, logits, n_vocab, false, sc.excl, cur, cur_p)) {
        return COMMON_GRAMFAST_NONE;
    }
    *id = cur_p.data[cur_p.selected].id;
    if (grammar_first || !constrain) {
        return COMMON_GRAMFAST_DONE;
    }
    {
        llama_token_data       single     = { *id, 1.0f, 0.0f };
        llama_token_data_array single_arr = { &single, 1, -1, false };
        llama_sampler_apply(grmr, &single_arr);
        if (single.logit != -INFINITY) {
            return COMMON_GRAMFAST_DONE;
        }
    }

    // rejected: the grammar first, then the chain again
    if (common_sampler_gramfast_apply(cfg, chain, grmr, logits, n_vocab, false, sc, cur, cur_p)) {
        *id = cur_p.data[cur_p.selected].id;
        return COMMON_GRAMFAST_DONE;
    }
    return COMMON_GRAMFAST_RESAMPLE;
}

// the ring buffer works similarly to std::deque, but with a fixed capacity
// TODO: deduplicate with llama-impl.h
template<typename T>
struct ring_buffer {
    ring_buffer(size_t cap) : capacity(cap), data(cap) {}

    T & front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    const T & front() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    T & back() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    const T & back() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    void push_back(const T & value) {
        if (sz == capacity) {
            // advance the start when buffer is full
            first = (first + 1) % capacity;
        } else {
            sz++;
        }
        data[pos] = value;
        pos = (pos + 1) % capacity;
    }

    T pop_front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        T value = data[first];
        first = (first + 1) % capacity;
        sz--;
        return value;
    }

    const T & rat(size_t i) const {
        if (i >= sz) {
            throw std::runtime_error("ring buffer: index out of bounds");
        }
        return data[(first + sz - i - 1) % capacity];
    }

    std::vector<T> to_vector() const {
        std::vector<T> result;
        result.reserve(sz);
        for (size_t i = 0; i < sz; i++) {
            result.push_back(data[(first + i) % capacity]);
        }
        return result;
    }

    void clear() {
        // here only reset the status of the buffer
        sz = 0;
        first = 0;
        pos = 0;
    }

    bool empty() const {
        return sz == 0;
    }

    size_t size() const {
        return sz;
    }

    size_t capacity = 0;
    size_t sz = 0;
    size_t first = 0;
    size_t pos = 0;
    std::vector<T> data;
};

struct common_sampler {
    common_params_sampling params;

    struct llama_sampler * grmr;
    struct llama_sampler * rbudget;
    struct llama_sampler * chain;

    // top-k pre-trim used by the grammar passes (see common_sampler_sample). null when
    // common_sampler_pretrim_cfg finds no exact trim for these parameters, which keeps
    // the original full-vocabulary behaviour. [TAG_V0261_PRETRIM]
    struct llama_sampler * grmr_pretrim;

    ring_buffer<llama_token> prev;

    std::vector<llama_token_data> cur;

    llama_token_data_array cur_p;

    // the verifier's random stream, for the fork's maximal coupling and upstream #27694's rejection sampling alike;
    // independent of the draft, or the target distribution is not preserved. Reseeded from speculative_seed on
    // reset() so a request is reproducible. [TAG_SYNC_1004] the fork's speculative_rng and upstream's rng are one
    uint32_t speculative_seed;
    std::mt19937 rng;

    void reset() {
        prev.clear();

        llama_sampler_reset(chain);
        rng.seed(speculative_seed);
    }

    void set_logits(struct llama_context * ctx, int idx) {
        const float *       sampled_probs  = llama_get_sampled_probs_ith     (ctx, idx);
        const float *       sampled_logits = llama_get_sampled_logits_ith    (ctx, idx);
        const llama_token * sampled_ids    = llama_get_sampled_candidates_ith(ctx, idx);

        const llama_model * model = llama_get_model(ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);

        const int n_vocab = llama_vocab_n_tokens(vocab);

        // [TAG_BS_CANDIDATE_GUARD] The three backend-sampled arrays are fetched independently and
        // are not guaranteed to correspond for a given row - get_sampled_candidates_ith falls back
        // to token_ids_full_vocab when this row has no candidate data, while the probs array stays
        // top-k sized. Taking the probs count as the bound for all three then reads token ids out
        // of an array that holds something else, and the whole candidate set comes back as float
        // bit patterns (0x3F800000 = 1.0f, 0xFF800000 = -inf). That surfaces much later as
        // "invalid vector subscript" from the vocab lookup and kills the request with an HTTP 500.
        //
        // Agree on the shortest count, verify every id is a real token, and fall back to plain
        // full-vocab logits if anything is off. Correctness first; the fallback costs one position.
        const bool bs_ok = [&]() -> bool {
            if (!sampled_probs || !sampled_ids || !sampled_logits) {
                return false;
            }

            const uint32_t n_p = llama_get_sampled_probs_count_ith(ctx, idx);
            const uint32_t n_c = llama_get_sampled_candidates_count_ith(ctx, idx);
            const uint32_t n_l = llama_get_sampled_logits_count_ith(ctx, idx);

            if (n_p == 0 || n_c < n_p || n_l < n_p) {
                return false;
            }

            for (uint32_t i = 0; i < n_p; ++i) {
                if (sampled_ids[i] < 0 || sampled_ids[i] >= n_vocab) {
                    return false;
                }
            }

            return true;
        }();

        if (sampled_probs && !bs_ok) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                LOG_WRN("[TAG_BS_CANDIDATE_GUARD] backend-sampled candidates are inconsistent for "
                        "output %d (probs=%u candidates=%u logits=%u) - falling back to full-vocab "
                        "logits. This is logged once.\n", idx,
                        (unsigned) llama_get_sampled_probs_count_ith(ctx, idx),
                        (unsigned) llama_get_sampled_candidates_count_ith(ctx, idx),
                        (unsigned) llama_get_sampled_logits_count_ith(ctx, idx));
            }
        }

        if (bs_ok) {
            const uint32_t sampled_probs_count = llama_get_sampled_probs_count_ith(ctx, idx);
            cur.resize(sampled_probs_count);
            for (uint32_t i = 0; i < sampled_probs_count; ++i) {
                cur[i] = llama_token_data{sampled_ids[i], sampled_logits[i], sampled_probs[i]};
            }
        } else if (sampled_logits && !sampled_probs) {
            const uint32_t sampled_logits_count = llama_get_sampled_logits_count_ith(ctx, idx);
            const uint32_t n_c                  = llama_get_sampled_candidates_count_ith(ctx, idx);

            bool ids_ok = sampled_ids != nullptr && n_c >= sampled_logits_count;
            for (uint32_t i = 0; ids_ok && i < sampled_logits_count; ++i) {
                ids_ok = sampled_ids[i] >= 0 && sampled_ids[i] < n_vocab;
            }

            if (ids_ok) {
                cur.resize(sampled_logits_count);
                for (uint32_t i = 0; i < sampled_logits_count; i++) {
                    cur[i] = llama_token_data{sampled_ids[i], sampled_logits[i], 0.0f};
                }
            } else {
                const auto * logits = llama_get_logits_ith(ctx, idx);
                GGML_ASSERT(logits != nullptr);
                cur.resize(n_vocab);
                for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                    cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
                }
            }
        } else {
            const auto * logits = llama_get_logits_ith(ctx, idx);
            GGML_ASSERT(logits != nullptr);
            cur.resize(n_vocab);
            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
            }
        }

        cur_p = { cur.data(), cur.size(), -1, false };
    }

    // [TAG_FN_L4_HOST_TOPK] the candidate array the chain's top-k leaves, built straight from the row (no full-vocabulary
    // array, no partial sort over it); false: not exact here, the caller uses set_logits(). Only for a chain whose
    // samplers before top-k change nothing (common_sampler_init sets fast_k), and only for plain logits rows.
    bool set_logits_fast(struct llama_context * ctx, int idx) {
        if (llama_get_sampled_probs_ith(ctx, idx) != nullptr || llama_get_sampled_logits_ith(ctx, idx) != nullptr ||
                llama_get_sampled_token_ith(ctx, idx) != LLAMA_TOKEN_NULL) {
            return false;
        }
        const float * logits = llama_get_logits_ith(ctx, idx);
        if (logits == nullptr) {
            return false;
        }
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
        if (!common_sampler_fn_l4_topk_row(logits, n_vocab, fast_k, fast_excl, cur)) {
            return false;
        }
        // the state the chain's top-k leaves: the K largest, sorted
        cur_p = { cur.data(), cur.size(), -1, true };
        return true;
    }

    // [TAG_R28_PENFAST] the fast candidate path with penalties / a logit bias, the chain included. True: cur_p is the
    // chain's result (the caller does not apply it again). False: nothing with state ran; set_logits() + the chain.
    bool sample_penfast(struct llama_context * ctx, int idx) {
        if (llama_get_sampled_probs_ith(ctx, idx) != nullptr || llama_get_sampled_logits_ith(ctx, idx) != nullptr ||
                llama_get_sampled_token_ith(ctx, idx) != LLAMA_TOKEN_NULL) {
            return false;
        }
        const float * logits = llama_get_logits_ith(ctx, idx);
        if (logits == nullptr) {
            return false;
        }
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
        return common_sampler_penfast_apply(penfast, chain, logits, n_vocab, false, penfast_excl, cur, cur_p);
    }

    common_time_meas tm() {
        return common_time_meas(t_total_us, params.no_perf);
    }

    mutable int64_t t_total_us = 0;

    // [TAG_FN_L3_MTP_BLOCK] block verification's scratch sampler (SPEC_MTP_BLOCK_VERIFY on qwen4exp): a clone made at the
    // first block step, this sampler's state copied into it at every later one (its candidate buffer stays allocated)
    struct common_sampler * blk = nullptr;

    // [TAG_FN_L4_HOST_TOPK] LLAMA_FN_L4_HOST_TOPK (qwen4exp): K of the fast candidate path, 0 = off; the ids the
    // logit-bias sampler sets to -inf (sorted)
    int                      fast_k = 0;
    std::vector<llama_token> fast_excl;

    // [TAG_R28_PENFAST] the fast candidate path with penalties / a logit bias (top_k 0: off) and its scratch
    // [TAG_R29_GRAMFAST] with a grammar and/or a reasoning budget penfast drives the candidate-set path instead
    common_penfast_cfg       penfast;
    std::vector<llama_token> penfast_excl;

    // [TAG_R29_GRAMFAST] the candidate-set path's scratch, and what its last run saw
    common_gramfast_scratch  gramfast_sc;

    // [TAG_FN_L21_NANGUARD] rows seen with NaN / +inf logits since the last take (last: the aggregate inits above)
    uint32_t n_nonfinite = 0;

    // [TAG_V0261_PRETRIM] the wider second trim (null: none), the trim rule, and per-position scratch
    struct llama_sampler *   grmr_pretrim_wide = nullptr;
    common_pretrim_cfg       pretrim;
    std::vector<llama_token> pretrim_hist;
    std::vector<llama_token> pretrim_touched;
    size_t                   pretrim_bound = 0;
};

std::string common_params_sampling::print() const {
    char result[1024];

    snprintf(result, sizeof(result),
            "\trepeat_last_n = %d, repeat_penalty = %.3f, frequency_penalty = %.3f, presence_penalty = %.3f\n"
            "\tdry_multiplier = %.3f, dry_base = %.3f, dry_allowed_length = %d, dry_penalty_last_n = %d\n"
            "\ttop_k = %d, top_p = %.3f, min_p = %.3f, xtc_probability = %.3f, xtc_threshold = %.3f, typical_p = %.3f, top_n_sigma = %.3f, temp = %.3f\n"
            "\tmirostat = %d, mirostat_lr = %.3f, mirostat_ent = %.3f, adaptive_target = %.3f, adaptive_decay = %.3f",
            penalty_last_n, penalty_repeat, penalty_freq, penalty_present,
            dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n,
            top_k, top_p, min_p, xtc_probability, xtc_threshold, typ_p, top_n_sigma, temp,
            mirostat, mirostat_eta, mirostat_tau, adaptive_target, adaptive_decay);

    return std::string(result);
}

// [TAG_BS_LAZY_GRAMMAR] declared in src/llama-ext.h (the fork's staging API, not in llama.h)
LLAMA_API bool llama_sampler_grammar_awaiting_trigger(const struct llama_sampler * smpl);
LLAMA_API void llama_sampler_chain_backend_detach(struct llama_sampler * smpl);

void common_sampler_backend_detach(struct common_sampler * gsmpl, struct llama_context * ctx, llama_seq_id seq_id) {
    llama_set_sampler(ctx, seq_id, nullptr);
    if (gsmpl != nullptr) {
        llama_sampler_chain_backend_detach(gsmpl->chain);
    }
}

static bool bs_lazy_grammar_enabled() {
    static const bool on = [] {
        const char * e = std::getenv("TURBO_BS_LAZY_GRAMMAR");
        return e == nullptr || std::atoi(e) != 0;
    }();
    return on;
}

// [TAG_UPSTREAM_29797] LLAMA_GREEDY_TEMP0=0 restores dist as the last sampler of every mirostat-0 chain
static bool greedy_temp0_enabled() {
    static const bool on = [] {
        const char * e = std::getenv("LLAMA_GREEDY_TEMP0");
        return e == nullptr || std::atoi(e) != 0;
    }();
    return on;
}

bool common_sampler_backend_ok(const struct common_sampler * gsmpl) {
    if (gsmpl == nullptr || (!gsmpl->grmr && !gsmpl->rbudget)) {
        return true;
    }
    if (!bs_lazy_grammar_enabled()) {
        return false;
    }
    if (gsmpl->grmr && !llama_sampler_grammar_awaiting_trigger(gsmpl->grmr)) {
        return false;   // non-lazy, llguidance, or already triggered: the grammar constrains sampling now
    }
    if (gsmpl->rbudget && common_reasoning_budget_get_state(gsmpl->rbudget) == REASONING_BUDGET_FORCING) {
        return false;   // the budget is overwriting logits with its forced end sequence
    }
    return true;
}

// [TAG_V0261_PRETRIM] LLAMA_V0261_PRETRIM: 1 (default) the exact trim rule, 0 the PR #1 gate, 2 the v0.25.5 gate
static int v0261_pretrim_mode() {
    static const int mode = [] {
        const char * e = std::getenv("LLAMA_V0261_PRETRIM");
        if (e == nullptr || e[0] == '\0') {
            return 1;
        }
        const int v = std::atoi(e);
        return v == 0 || v == 2 ? v : 1;
    }();
    return mode;
}

// [TAG_V0261_PRETRIM] LLAMA_V0261_PRETRIM_NARROW=1: the second trim also runs after a first trim that kept fewer valid
// tokens than the top-k (c9b6c373c); default 0: the full pass right away there, as v0.25.5
static bool v0261_pretrim_narrow() {
    static const bool on = [] {
        const char * e = std::getenv("LLAMA_V0261_PRETRIM_NARROW");
        return e != nullptr && std::atoi(e) == 1;
    }();
    return on;
}

// the chain's first top-k sees the raw logits: every sampler before it is a no-op for these parameters (penalties,
// dry, top-n-sigma disabled; no user logit bias; no mirostat), apart from the model's suppress tokens
static bool common_sampler_first_top_k_raw(const common_params_sampling & params) {
    if (params.mirostat == 0 && params.logit_bias.empty() && params.top_k >= 1 &&
            (params.penalty_last_n == 0 ||
             (params.penalty_repeat == 1.0f && params.penalty_freq == 0.0f && params.penalty_present == 0.0f)) &&
            (params.dry_multiplier == 0.0f || params.dry_base < 1.0f || params.dry_penalty_last_n == 0) &&
            params.top_n_sigma <= 0.0f) {
        for (const auto t : params.samplers) {
            if (t == COMMON_SAMPLER_TYPE_TOP_K) {
                return true;
            }
            if (t != COMMON_SAMPLER_TYPE_PENALTIES && t != COMMON_SAMPLER_TYPE_DRY && t != COMMON_SAMPLER_TYPE_TOP_N_SIGMA) {
                break;
            }
        }
    }
    return false;
}

// [TAG_V0261_PRETRIM] see sampling.h. Mode 1: the logit bias, penalties and DRY may run before the chain's top-k as long
// as they only lower logits; the need grows by the tokens they can touch, and so does the width of the second trim.
common_pretrim_cfg common_sampler_pretrim_cfg(const common_params_sampling & params, const llama_token * suppress,
        int32_t n_suppress, int32_t n_vocab, int mode) {
    common_pretrim_cfg cfg;
    if (params.top_k <= 0) {
        return cfg;
    }
    // the v0.25.5 width, 32 x top_k and at least 1024
    const size_t width = (size_t) std::min<int64_t>(std::max<int64_t>((int64_t) params.top_k * 32, 1024), INT32_MAX);

    if (mode == 2) {
        cfg.top_k = params.top_k;
        cfg.width = width;
        return cfg;
    }
    if (mode == 0) {
        if (common_sampler_first_top_k_raw(params)) {
            cfg.top_k   = params.top_k;
            cfg.width   = width;
            cfg.n_extra = (size_t) std::max(n_suppress, 0);
        }
        return cfg;
    }

    // a trim as wide as the vocabulary saves nothing
    if (params.mirostat != 0 || width >= (size_t) std::max(n_vocab, 0)) {
        return cfg;
    }

    // the samplers before the first top-k: only penalties, DRY and an inactive top-n-sigma
    bool pen_before = false;
    bool dry_before = false;
    bool have_top_k = false;
    for (const auto t : params.samplers) {
        if (t == COMMON_SAMPLER_TYPE_TOP_K) {
            have_top_k = true;
            break;
        }
        if (t == COMMON_SAMPLER_TYPE_PENALTIES) {
            pen_before = true;
        } else if (t == COMMON_SAMPLER_TYPE_DRY) {
            dry_before = true;
        } else if (t != COMMON_SAMPLER_TYPE_TOP_N_SIGMA || !(params.top_n_sigma <= 0.0f)) {
            return cfg;
        }
    }
    if (!have_top_k) {
        return cfg;
    }

    // active as llama_sampler_init_penalties / llama_sampler_init_dry decide it (a negative window counts as 0)
    int32_t last_n = 0;
    const bool pen_on = std::max(params.penalty_last_n, 0) != 0 &&
        !(params.penalty_repeat == 1.0f && params.penalty_freq == 0.0f && params.penalty_present == 0.0f);
    if (pen_before && pen_on) {
        // repeat < 1 or a negative frequency / presence penalty raises logits
        if (!(params.penalty_repeat >= 1.0f && params.penalty_freq >= 0.0f && params.penalty_present >= 0.0f)) {
            return cfg;
        }
        last_n = params.penalty_last_n;
    }
    const bool dry_on = params.dry_multiplier != 0.0f && params.dry_base >= 1.0f && std::max(params.dry_penalty_last_n, 0) != 0;
    if (dry_before && dry_on) {
        if (!(params.dry_multiplier > 0.0f)) {
            return cfg;
        }
        last_n = std::max(last_n, params.dry_penalty_last_n);
    }

    // a positive (or NaN) bias raises a token from below the trim; a token listed twice gets only its first bias once
    // the candidates are no longer in id order (llama_sampler_logit_bias_apply), so both keep the full pass.
    // [TAG_V0261_PRETRIM] a zero bias counts as a listing too: {t, 0} before {t, -x} (a user zero bias before the
    // ignore_eos -inf or a suppress token's -inf) gives t only the +0 in the trimmed pass and the -x in the full pass.
    // Only the negative ones can lower a logit, so only they join the touched set.
    std::vector<llama_token> fixed;
    std::vector<llama_token> listed;
    for (const auto & lb : params.logit_bias) {
        if (!(lb.bias <= 0.0f)) {
            return cfg;
        }
        listed.push_back(lb.token);
        if (lb.bias < 0.0f) {
            fixed.push_back(lb.token);
        }
    }
    for (int32_t i = 0; i < n_suppress; ++i) {
        listed.push_back(suppress[i]);
        fixed.push_back(suppress[i]);
    }
    std::sort(listed.begin(), listed.end());
    if (std::adjacent_find(listed.begin(), listed.end()) != listed.end()) {
        return cfg;
    }
    std::sort(fixed.begin(), fixed.end());

    cfg.top_k  = params.top_k;
    cfg.width  = width;
    cfg.last_n = last_n;
    cfg.fixed  = std::move(fixed);

    // the second trim: the v0.25.5 rule (32 x the valid tokens needed) for the largest need, at most 1/8 of the vocabulary
    const int64_t need_max = (int64_t) cfg.top_k + (int64_t) cfg.fixed.size() + (int64_t) last_n;
    const int64_t wide     = std::min<int64_t>(std::min<int64_t>(need_max * 32, (int64_t) n_vocab / 8), INT32_MAX);
    cfg.width_max = wide > (int64_t) width ? (size_t) wide : 0;

    return cfg;
}

// [TAG_V0261_PRETRIM] the penalties and DRY lower only tokens of their own window, the last last_n accepted tokens
size_t common_sampler_pretrim_touched(const common_pretrim_cfg & cfg, const std::vector<llama_token> & hist,
        bool hist_all, std::vector<llama_token> & touched) {
    touched.assign(cfg.fixed.begin(), cfg.fixed.end());
    size_t n_bound = 0;
    if (cfg.last_n > 0) {
        const size_t w = (size_t) cfg.last_n;
        const size_t n = std::min(w, hist.size());
        touched.insert(touched.end(), hist.end() - (std::ptrdiff_t) n, hist.end());
        if (w > hist.size() && !hist_all) {
            n_bound = w - hist.size();
        }
        std::sort(touched.begin(), touched.end());
        touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
    }
    return n_bound;
}

// [TAG_V0261_PRETRIM] every token outside the trim has a raw logit at or below every token inside it, and the samplers
// before the chain's top-k only lower logits, so top_k valid tokens inside it that they leave alone keep every token
// outside it out of the chain's top-k set. The scan stops at the need (the trim leaves the best first, so usually
// within the first few dozen); n_valid gets the finite candidates, the full count only when it returns false.
bool common_sampler_pretrim_exact(const common_pretrim_cfg & cfg, const llama_token_data_array * cur_p,
        const std::vector<llama_token> & touched, size_t n_bound, size_t * n_valid) {
    // without a top-k nothing passes: the scan then counts every finite candidate
    const size_t need = cfg.top_k > 0 ? (size_t) cfg.top_k + cfg.n_extra + n_bound : SIZE_MAX;
    size_t n_ok  = 0;
    size_t n_fin = 0;
    for (size_t i = 0; i < cur_p->size; ++i) {
        if (cur_p->data[i].logit == -INFINITY) {
            continue;
        }
        ++n_fin;
        if (!touched.empty() && std::binary_search(touched.begin(), touched.end(), cur_p->data[i].id)) {
            continue;
        }
        if (++n_ok >= need) {
            if (n_valid) {
                *n_valid = n_fin;
            }
            return true;
        }
    }
    if (n_valid) {
        *n_valid = n_fin;
    }
    return false;
}

struct common_sampler * common_sampler_init(
        const struct llama_model * model,
        struct common_params_sampling & params) {
    if (!std::isfinite(params.penalty_repeat) ||
        params.penalty_repeat <= 0.0f ||
        !std::isfinite(1.0f/params.penalty_repeat)) {
        throw std::invalid_argument("penalty_repeat must be finite and greater than 0");
    }
    if (!std::isfinite(params.penalty_freq)) {
        throw std::invalid_argument("penalty_freq must be finite");
    }
    if (!std::isfinite(params.penalty_present)) {
        throw std::invalid_argument("penalty_present must be finite");
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_sampler_chain_params lparams = llama_sampler_chain_default_params();

    lparams.no_perf = params.no_perf;

    llama_sampler * grmr = nullptr;
    llama_sampler * rbudget = nullptr;
    llama_sampler * chain = llama_sampler_chain_init(lparams);

    std::vector<llama_sampler *> samplers;

    const std::string & grammar_str = common_grammar_value(params.grammar);
    if (grammar_str.compare(0, 11, "%llguidance") == 0) {
#ifdef LLAMA_USE_LLGUIDANCE
        grmr = llama_sampler_init_llg(vocab, "lark", grammar_str.c_str());
#else
        throw std::runtime_error("failed to parse grammar: llguidance is not enabled");
#endif // LLAMA_USE_LLGUIDANCE
    } else {
        std::vector<std::string> trigger_patterns;
        std::vector<llama_token> trigger_tokens;
        for (const auto & trigger : params.grammar_triggers) {
            switch (trigger.type) {
                case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
                {
                    const auto & word = trigger.value;
                    trigger_patterns.push_back(regex_escape(word));
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
                {
                    trigger_patterns.push_back(trigger.value);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL:
                {
                    const auto & pattern = trigger.value;
                    std::string anchored = "^$";
                    if (!pattern.empty()) {
                        anchored = (pattern.front() != '^' ? "^" : "")
                            + pattern
                            + (pattern.back() != '$' ? "$" : "");
                    }
                    trigger_patterns.push_back(anchored);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
                {
                    const auto token = trigger.token;
                    trigger_tokens.push_back(token);
                    break;
                }
                default:
                    GGML_ASSERT(false && "unknown trigger type");
            }
        }

        std::vector<const char *> trigger_patterns_c;
        trigger_patterns_c.reserve(trigger_patterns.size());
        for (const auto & regex : trigger_patterns) {
            trigger_patterns_c.push_back(regex.c_str());
        }

        if (!grammar_str.empty()) {
             if (params.grammar_lazy) {
                 grmr = llama_sampler_init_grammar_lazy_patterns(vocab, grammar_str.c_str(), "root",
                         trigger_patterns_c.data(), trigger_patterns_c.size(),
                         trigger_tokens.data(), trigger_tokens.size());
             } else {
                 grmr = llama_sampler_init_grammar(vocab, grammar_str.c_str(), "root");
             }
        }
    }
    if (!grmr && !grammar_str.empty()) {
        throw std::runtime_error("failed to parse grammar");
    }

    // Compute prefill tokens from the generation prompt
    std::vector<llama_token> prefill_tokens;
    if (!params.generation_prompt.empty()) {
        GGML_ASSERT(vocab != nullptr);
        auto tokens = common_tokenize(vocab, params.generation_prompt, false, true);
        for (size_t i = 0; i < tokens.size(); i++) {
            std::string piece = common_token_to_piece(vocab, tokens[i], true);
            if (i == 0 && std::isspace(piece[0]) && !std::isspace(params.generation_prompt[0])) {
                // Some tokenizers will add a space before the first special token, need to exclude
                continue;
            }
            LOG_DBG("%s: prefill token: %d = %s\n", __func__, tokens[i], piece.c_str());
            prefill_tokens.push_back(tokens[i]);
        }
    }

    // Feed generation prompt tokens to the grammar sampler so it advances past
    // tokens the template already placed in the prompt.
    // Only applies to output-format and tool-call grammars; user-supplied grammars must not be prefilled.
    if (grmr && !params.grammar_lazy && common_grammar_needs_prefill(params.grammar)) {
        try {
            for (const auto & token : prefill_tokens) {
                llama_sampler_accept(grmr, token);
                LOG_DBG("%s: grammar accepted prefill token (%d)\n", __func__, token);
            }
        } catch (std::exception &e) {
            LOG_ERR("%s: error initializing grammar sampler for grammar:\n%s\n\nGeneration prompt:\n'%s'\n", __func__,
                common_grammar_value(params.grammar).c_str(), params.generation_prompt.c_str());
            throw e;
        }
    }

    // reasoning budget sampler (skip when budget is unlimited unless a lazy grammar is active, which needs rbudget for thinking-block suppression)
    if (!params.reasoning_budget_start.empty() && !params.reasoning_budget_end.empty() && (params.grammar_lazy || params.reasoning_budget_tokens >= 0 || params.reasoning_control)) {
        rbudget = common_reasoning_budget_init(
            vocab,
            {params.reasoning_budget_start},
            params.reasoning_budget_end,
            params.reasoning_budget_forced,
            params.reasoning_budget_tokens < 0 ? INT_MAX : params.reasoning_budget_tokens);

        for (const auto & token : prefill_tokens) {
            llama_sampler_accept(rbudget, token);
            LOG_DBG("%s: reasoning-budget accepted prefill token (%d)\n", __func__, token);
        }
    }

    // logit bias: user biases + model suppress tokens (-INFINITY)
    {
        std::vector<llama_logit_bias> merged = params.logit_bias;

        int32_t n_suppress = 0;
        const llama_token * suppress = llama_vocab_get_suppress_tokens(vocab, &n_suppress);
        for (int32_t i = 0; i < n_suppress; ++i) {
            merged.push_back({ suppress[i], -INFINITY });
        }

        if (!merged.empty()) {
            samplers.push_back(llama_sampler_init_logit_bias(llama_vocab_n_tokens(vocab), merged.size(), merged.data()));
        }
    }

    if (params.mirostat == 0) {

        bool use_adaptive_p = false; // see below

        for (const auto & cnstr : params.samplers) {
            switch (cnstr) {
                case COMMON_SAMPLER_TYPE_DRY:
                    {
                        std::vector<const char *> c_breakers;
                        c_breakers.reserve(params.dry_sequence_breakers.size());
                        for (const auto & str : params.dry_sequence_breakers) {
                            c_breakers.push_back(str.c_str());
                        }
                        samplers.push_back(llama_sampler_init_dry(vocab, params.dry_multiplier, params.dry_base, params.dry_allowed_length, params.dry_penalty_last_n, c_breakers.data(), c_breakers.size()));
                    }
                    break;
                case COMMON_SAMPLER_TYPE_TOP_K:
                    samplers.push_back(llama_sampler_init_top_k(params.top_k));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_P:
                    samplers.push_back(llama_sampler_init_top_p(params.top_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:
                    samplers.push_back(llama_sampler_init_top_n_sigma(params.top_n_sigma));
                    break;
                case COMMON_SAMPLER_TYPE_MIN_P:
                    samplers.push_back(llama_sampler_init_min_p(params.min_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_XTC:
                    samplers.push_back(llama_sampler_init_xtc(params.xtc_probability, params.xtc_threshold, params.min_keep, params.seed));
                    break;
                case COMMON_SAMPLER_TYPE_TYPICAL_P:
                    samplers.push_back(llama_sampler_init_typical(params.typ_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TEMPERATURE:
                    samplers.push_back(llama_sampler_init_temp_ext(params.temp, params.dynatemp_range, params.dynatemp_exponent));
                    break;
                case COMMON_SAMPLER_TYPE_INFILL:
                    samplers.push_back(llama_sampler_init_infill(vocab));
                    break;
                case COMMON_SAMPLER_TYPE_PENALTIES:
                    samplers.push_back(llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), params.penalty_last_n, params.penalty_repeat, params.penalty_freq, params.penalty_present));
                    break;
                case COMMON_SAMPLER_TYPE_ADAPTIVE_P:
                    // the `adaptive-p` sampler is like `dist` and `mirostat` in that it selects
                    // a single token, so we will add `dist` at the end of the chain by default,
                    // unless the user specifically included `adaptive-p`. we set this flag here
                    // so we know to add the sampler at the very end.
                    use_adaptive_p = true;
                    break;
                default:
                    GGML_ASSERT(false && "unknown sampler type");
            }
        }
        if (use_adaptive_p) {
            // only if user explicitly included adaptive-p sampler
            samplers.push_back(llama_sampler_init_adaptive_p(params.adaptive_target, params.adaptive_decay, params.seed));
        } else {
            // default: sample from distribution
            // [TAG_UPSTREAM_29797] when the chain leaves only one possible token (temperature 0 last, or top-k 1 last),
            // greedy selects the same token as dist and does not pay for it (on the backend: soft_max, cumsum and a
            // synchronous upload of the random number per output row). Keep dist when callers request probabilities.
            // Fork: the top-k 1 case also needs temp <= 0. At temp > 0 the speculative verifier (coupling, rejection,
            // block) reads p, and only dist sets p = 1 for the single candidate.
            const bool greedy = greedy_temp0_enabled() && params.n_probs == 0 && !params.samplers.empty() &&
                ((params.samplers.back() == COMMON_SAMPLER_TYPE_TEMPERATURE && params.temp == 0.0f && params.dynatemp_range == 0.0f) ||
                 (params.samplers.back() == COMMON_SAMPLER_TYPE_TOP_K && params.top_k == 1 && params.temp <= 0.0f));
            samplers.push_back(greedy ? llama_sampler_init_greedy() : llama_sampler_init_dist(params.seed));
        }
    } else if (params.mirostat == 1) {
        samplers.push_back(llama_sampler_init_temp(params.temp));
        samplers.push_back(llama_sampler_init_mirostat(llama_vocab_n_tokens(vocab), params.seed, params.mirostat_tau, params.mirostat_eta, 100));
    } else if (params.mirostat == 2) {
        samplers.push_back(llama_sampler_init_temp(params.temp));
        samplers.push_back(llama_sampler_init_mirostat_v2(params.seed, params.mirostat_tau, params.mirostat_eta));
    } else {
        GGML_ASSERT(false && "unknown mirostat version");
    }

    for (auto * smpl : samplers) {
        llama_sampler_chain_add(chain, smpl);
    }

    // [TAG_BS_LAZY_GRAMMAR] A lazy grammar that has not triggered, and a reasoning budget that is not forcing, change
    // nothing about the distribution (both apply() calls are no-ops), so they no longer cost the request its backend
    // sampler. Tool-bearing agent requests always carry both. The server re-checks common_sampler_backend_ok() after
    // every accepted token and drops to CPU sampling for the slot the moment either one becomes active.
    const bool bs_until_active = bs_lazy_grammar_enabled() &&
        (!grmr    || (params.grammar_lazy && llama_sampler_grammar_awaiting_trigger(grmr))) &&
        (!rbudget || common_reasoning_budget_get_state(rbudget) != REASONING_BUDGET_FORCING);

    if (grmr && params.backend_sampling && !bs_until_active) {
        LOG_WRN("%s: backend sampling is not compatible with grammar, disabling\n", __func__);

        params.backend_sampling = false;
    }

    if (rbudget && params.backend_sampling && !bs_until_active) {
        LOG_WRN("%s: backend sampling is not compatible with reasoning budget, disabling\n", __func__);

        params.backend_sampling = false;
    }

    if ((grmr || rbudget) && params.backend_sampling) {
        LOG_DBG("%s: backend sampling kept until the lazy grammar triggers or the reasoning budget forces\n", __func__);
    }

    const bool first_top_k = common_sampler_first_top_k_raw(params);

    // Pre-trim for the grammar passes. Sized well above params.top_k so the
    // trimmed set almost always still contains enough grammar-valid candidates, which is the
    // condition under which the trim is exact rather than an approximation.
    // [TAG_V0261_PRETRIM] the rule and the widths: common_sampler_pretrim_cfg
    common_pretrim_cfg     pretrim_cfg;
    struct llama_sampler * pretrim      = nullptr;
    struct llama_sampler * pretrim_wide = nullptr;
    if (grmr) {
        int32_t n_suppress = 0;
        const llama_token * suppress = llama_vocab_get_suppress_tokens(vocab, &n_suppress);
        pretrim_cfg = common_sampler_pretrim_cfg(params, suppress, n_suppress, llama_vocab_n_tokens(vocab), v0261_pretrim_mode());
        if (pretrim_cfg.top_k > 0) {
            pretrim = llama_sampler_init_top_k((int32_t) pretrim_cfg.width);
            if (pretrim_cfg.width_max > 0) {
                pretrim_wide = llama_sampler_init_top_k((int32_t) pretrim_cfg.width_max);
            }
        }
    }

    // [TAG_FN_L4_HOST_TOPK] the fast candidate path: the chain's first top-k sees the raw logits, no grammar, no
    // reasoning budget, no token probabilities to report, 1 <= top_k <= 128 (the std::partial_sort branch of the top-k sampler)
    int fast_k = 0;
    std::vector<llama_token> fast_excl;
    if (llama_fn_l4_host_flag(model, "LLAMA_FN_L4_HOST_TOPK") && first_top_k && grmr == nullptr &&
            rbudget == nullptr && params.n_probs == 0 && params.top_k <= 128) {
        fast_k = params.top_k;
        int32_t n_suppress = 0;
        const llama_token * suppress = llama_vocab_get_suppress_tokens(vocab, &n_suppress);
        fast_excl.assign(suppress, suppress + n_suppress);
        std::sort(fast_excl.begin(), fast_excl.end());
    }

    // [TAG_R28_PENFAST] penalties (Kyro's presence penalty 1.5) or a user logit bias switch the path above off: the full
    // 248K array, the penalties over all of it and a partial sort per sample (srv.sample 1.7 ms vs 0.1 ms on Flash-Next).
    // This path is exact (common_sampler_penfast_apply), so all models; no grammar, no reasoning budget.
    // [TAG_R29_GRAMFAST] with a grammar and/or a reasoning budget the same configuration drives the candidate-set path of
    // common_sampler_sample (tool requests: the grammar over ~150 tokens instead of 248K); LLAMA_R29_GRAMFAST=0: v0.26.4
    common_penfast_cfg penfast;
    if (fast_k == 0 && r28_penfast_enabled() && ((grmr == nullptr && rbudget == nullptr) || r29_gramfast_enabled())) {
        int32_t n_suppress = 0;
        const llama_token * suppress = llama_vocab_get_suppress_tokens(vocab, &n_suppress);
        penfast = common_sampler_penfast_cfg(params, chain, suppress, n_suppress, llama_vocab_n_tokens(vocab));
    }

    // Keep verifier randomness independent from both target and draft sampling.
    const uint32_t speculative_seed = llama_sampler_get_seed(chain) ^ 0x9e3779b9U;
    auto * result = new common_sampler {
        /* .params  = */ params,
        /* .grmr    = */ grmr,
        /* .rbudget = */ rbudget,
        /* .chain   = */ chain,
        /* .grmr_pretrim = */ pretrim,
        /* .prev    = */ ring_buffer<llama_token>(std::max(32, params.n_prev)),
        /* .cur     = */ {},
        /* .cur_p   = */ {},
        /* .speculative_seed = */ speculative_seed,
        /* .rng     = */ std::mt19937(speculative_seed),
    };
    result->fast_k    = fast_k; // [TAG_FN_L4_HOST_TOPK]
    result->fast_excl = std::move(fast_excl);
    result->penfast   = std::move(penfast); // [TAG_R28_PENFAST]
    result->grmr_pretrim_wide = pretrim_wide; // [TAG_V0261_PRETRIM]
    result->pretrim           = std::move(pretrim_cfg);

    return result;
}

void common_sampler_free(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return;
    }

    llama_sampler_free(gsmpl->grmr);
    llama_sampler_free(gsmpl->rbudget);
    llama_sampler_free(gsmpl->chain);
    llama_sampler_free(gsmpl->grmr_pretrim);
    llama_sampler_free(gsmpl->grmr_pretrim_wide); // [TAG_V0261_PRETRIM]

    if (gsmpl->blk) {
        common_sampler_free(gsmpl->blk); // [TAG_FN_L3_MTP_BLOCK]
    }

    delete gsmpl;
}

static bool grammar_should_apply(struct common_sampler * gsmpl) {
    if (!gsmpl->grmr) {
        return false;
    }
    if (!gsmpl->rbudget) {
        return true;
    }
    if (gsmpl->params.grammar_lazy) {
        // if grammar is lazy, only apply when reasoning budget is not active
        const auto state = common_reasoning_budget_get_state(gsmpl->rbudget);
        return state == REASONING_BUDGET_IDLE || state == REASONING_BUDGET_DONE;
    }
    return true;
}

void common_sampler_accept(struct common_sampler * gsmpl, llama_token token, bool is_generated) {
    if (!gsmpl) {
        return;
    }

    const auto tm = gsmpl->tm();

    // grammar_should_apply() checks the reasoning budget state, so calculate this before we accept
    const auto accept_grammar = is_generated && grammar_should_apply(gsmpl);

    if (gsmpl->rbudget && is_generated) {
        llama_sampler_accept(gsmpl->rbudget, token);

        // if done, replay end sequence which may contain a grammar trigger
        const bool is_done = common_reasoning_budget_get_state(gsmpl->rbudget) == REASONING_BUDGET_DONE;
        if (gsmpl->grmr && !accept_grammar && is_done) {
            const llama_tokens * end_seq = common_reasoning_budget_get_end_match(gsmpl->rbudget);
            if (end_seq) {
                for (const llama_token end_token : *end_seq) {
                    llama_sampler_accept(gsmpl->grmr, end_token);
                }
            }
        }
    }

    if (gsmpl->grmr && accept_grammar) {
        llama_sampler_accept(gsmpl->grmr, token);
    }

    llama_sampler_accept(gsmpl->chain, token);

    gsmpl->prev.push_back(token);
}

uint32_t common_sampler_take_nonfinite(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return 0;
    }
    const uint32_t n = gsmpl->n_nonfinite;
    gsmpl->n_nonfinite = 0;
    return n;
}

void common_sampler_reset(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return;
    }

    gsmpl->reset();
}

struct common_sampler * common_sampler_clone(common_sampler * gsmpl) {
    // [TAG_SAMPLER_CLONE_CURP] cur_p.data is a raw pointer into `cur`. The clone gets its own copy
    // of `cur` below, so copying cur_p verbatim would leave the clone pointing into the ORIGINAL
    // sampler's vector - dangling the moment that vector is reallocated or the original is freed.
    // common_sampler_copy already re-points for exactly this reason (see below); clone never did.
    common_sampler * res = new common_sampler {
        /* .params  = */ gsmpl->params,
        /* .grmr    = */ llama_sampler_clone(gsmpl->grmr),
        /* .rbudget = */ llama_sampler_clone(gsmpl->rbudget),
        /* .chain   = */ llama_sampler_clone(gsmpl->chain),
        /* .grmr_pretrim = */ gsmpl->grmr_pretrim ? llama_sampler_clone(gsmpl->grmr_pretrim) : nullptr,
        /* .prev    = */ gsmpl->prev,
        /* .cur     = */ gsmpl->cur,
        /* .cur_p   = */ gsmpl->cur_p,
        /* .speculative_seed = */ gsmpl->speculative_seed,
        /* .rng     = */ gsmpl->rng,
    };

    res->cur_p.data = gsmpl->cur_p.data ? res->cur.data() : nullptr; // re-point to the clone's buffer
    res->fast_k     = gsmpl->fast_k;    // [TAG_FN_L4_HOST_TOPK]
    res->fast_excl  = gsmpl->fast_excl;
    res->penfast    = gsmpl->penfast;   // [TAG_R28_PENFAST]
    res->grmr_pretrim_wide = gsmpl->grmr_pretrim_wide ? llama_sampler_clone(gsmpl->grmr_pretrim_wide) : nullptr; // [TAG_V0261_PRETRIM]
    res->pretrim           = gsmpl->pretrim;

    return res;
}

void common_sampler_copy(const common_sampler * src, common_sampler * dst) {
    if (!src || !dst || src == dst) {
        return;
    }

    GGML_ASSERT((src->grmr == nullptr) == (dst->grmr == nullptr));
    GGML_ASSERT((src->rbudget == nullptr) == (dst->rbudget == nullptr));

    llama_sampler_copy(src->grmr,    dst->grmr);
    llama_sampler_copy(src->rbudget, dst->rbudget);
    llama_sampler_copy(src->chain,   dst->chain);

    dst->params     = src->params;
    dst->prev       = src->prev;
    dst->cur        = src->cur;
    dst->cur_p      = src->cur_p;
    dst->cur_p.data = src->cur_p.data ? dst->cur.data() : nullptr; // re-point to dst's buffer
    dst->speculative_seed = src->speculative_seed;
    dst->rng        = src->rng;
    dst->t_total_us = src->t_total_us;
    dst->fast_k     = src->fast_k;    // [TAG_FN_L4_HOST_TOPK]
    dst->fast_excl  = src->fast_excl;
    dst->penfast    = src->penfast;   // [TAG_R28_PENFAST]
}

void common_perf_print(const struct llama_context * ctx, const struct common_sampler * gsmpl) {
    // TODO: measure grammar performance

    const double t_sampling_ms = gsmpl ? 1e-3*gsmpl->t_total_us : 0;

    llama_perf_sampler_data data_smpl;
    llama_perf_context_data data_ctx;

    memset(&data_smpl, 0, sizeof(data_smpl));
    memset(&data_ctx,  0, sizeof(data_ctx));

    if (gsmpl) {
        auto & data = data_smpl;

        data = llama_perf_sampler(gsmpl->chain);

        // note: the sampling time includes the samplers time + extra time spent in common/sampling
        LOG_INF("%s:    sampling time = %10.2f ms\n", __func__, t_sampling_ms);
        LOG_INF("%s:    samplers time = %10.2f ms / %5d tokens\n", __func__, data.t_sample_ms, data.n_sample);
    }

    if (ctx) {
        auto & data = data_ctx;

        data = llama_perf_context(ctx);

        const double t_end_ms = 1e-3 * ggml_time_us();

        const double t_total_ms = t_end_ms - data.t_start_ms;
        const double t_unacc_ms = t_total_ms - (t_sampling_ms + data.t_p_eval_ms + data.t_eval_ms);
        const double t_unacc_pc = 100.0 * t_unacc_ms /  t_total_ms;

        LOG_INF("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
        LOG_INF("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
        LOG_INF("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
                __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
        LOG_INF("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
        LOG_INF("%s: unaccounted time = %10.2f ms / %5.1f %%      (total - sampling - prompt eval - eval) / (total)\n", __func__, t_unacc_ms, t_unacc_pc);
        LOG_INF("%s:    graphs reused = %10d\n", __func__, data.n_reused);
        LOG_INF("%s:   graph reallocs = %10d    / %5d        (same size) / (changed size)\n", __func__, data.n_reallocs_same, data.n_reallocs_diff);

        common_memory_breakdown_print(ctx);
    }
}

struct llama_sampler * common_sampler_get(const struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return nullptr;
    }

    return gsmpl->chain;
}

// [TAG_V0261_PRETRIM] the tokens the chain may lower at this position, from the accepted history (prev holds every
// token since the last reset while it is not full); false when no trim can keep enough of the others
static bool common_sampler_pretrim_prepare(struct common_sampler * gsmpl) {
    const auto & cfg  = gsmpl->pretrim;
    const auto & prev = gsmpl->prev;
    auto & hist = gsmpl->pretrim_hist;

    const size_t n = std::min((size_t) std::max(cfg.last_n, 0), prev.size());
    hist.resize(n);
    for (size_t i = 0; i < n; ++i) {
        hist[n - 1 - i] = prev.rat(i);
    }
    const bool hist_all = n == prev.size() && prev.size() < prev.capacity;

    gsmpl->pretrim_bound = common_sampler_pretrim_touched(cfg, hist, hist_all, gsmpl->pretrim_touched);

    return (size_t) cfg.top_k + cfg.n_extra + gsmpl->pretrim_bound <= std::max(cfg.width, cfg.width_max);
}

// [TAG_V0261_PRETRIM] the grammar over the top raw candidates only, the wider trim second. cur_p: the full set after the
// reasoning budget. True: exact, cur_p holds the trimmed set with the grammar applied. False: cur_p is spent.
static bool common_sampler_pretrim_grammar(struct common_sampler * gsmpl, struct llama_context * ctx, int idx) {
    const auto & cfg = gsmpl->pretrim;
    auto & cur_p = gsmpl->cur_p;

    const size_t need = (size_t) cfg.top_k + cfg.n_extra + gsmpl->pretrim_bound;
    bool spent = false;
    for (int pass = 0; pass < 2; ++pass) {
        struct llama_sampler * trim = pass == 0 ? gsmpl->grmr_pretrim : gsmpl->grmr_pretrim_wide;
        if (trim == nullptr || need > (pass == 0 ? cfg.width : cfg.width_max)) {
            continue;
        }
        if (spent) {
            gsmpl->set_logits(ctx, idx);
            llama_sampler_apply(gsmpl->rbudget, &cur_p);
        }
        spent = true;

        llama_sampler_apply(trim,        &cur_p);
        llama_sampler_apply(gsmpl->grmr, &cur_p);

        // the trim leaves the candidates sorted; a lazy grammar that has not triggered keeps that flag and so does the
        // logit bias, and the chain's top-k would then cut without sorting again
        cur_p.sorted = false;

        size_t n_valid = 0;
        if (common_sampler_pretrim_exact(cfg, &cur_p, gsmpl->pretrim_touched, gsmpl->pretrim_bound, &n_valid)) {
            return true;
        }
        // [TAG_V0261_PRETRIM] fewer valid tokens than the top-k at all: a narrow grammar position (a name, a closing tag,
        // a bool / enum value) that the wider trim does not fill either, and where v0.25.5 also took the full pass. The
        // second trim is for a shortfall from the touched tokens only. LLAMA_V0261_PRETRIM_NARROW=1: run it there too.
        if (pass == 0 && n_valid < (size_t) cfg.top_k + cfg.n_extra && !v0261_pretrim_narrow()) {
            break;
        }
    }
    return false;
}

// [TAG_R29_GRAMFAST] the candidate-set path of common_sampler_sample (penfast is set with a grammar / a reasoning budget only
// under LLAMA_R29_GRAMFAST): plain logits rows, and a reasoning budget that does not force (its apply is then a no-op)
static common_gramfast_result common_sampler_sample_gramfast(struct common_sampler * gsmpl, struct llama_context * ctx,
        int idx, bool grammar_first, llama_token * id) {
    auto & sc = gsmpl->gramfast_sc;
    sc.width  = 0;
    sc.n_fin  = SIZE_MAX;
    sc.n_pass = 0;
    if (llama_get_sampled_probs_ith(ctx, idx) != nullptr || llama_get_sampled_logits_ith(ctx, idx) != nullptr ||
            llama_get_sampled_token_ith(ctx, idx) != LLAMA_TOKEN_NULL) {
        return COMMON_GRAMFAST_NONE;
    }
    if (gsmpl->rbudget != nullptr && common_reasoning_budget_get_state(gsmpl->rbudget) == REASONING_BUDGET_FORCING) {
        return COMMON_GRAMFAST_NONE;
    }
    const float * logits = llama_get_logits_ith(ctx, idx);
    if (logits == nullptr) {
        return COMMON_GRAMFAST_NONE;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    return common_sampler_gramfast_sample(gsmpl->penfast, gsmpl->chain, gsmpl->grmr, grammar_first,
            grammar_should_apply(gsmpl), logits, n_vocab, sc, gsmpl->cur, gsmpl->cur_p, id);
}

// [TAG_R29_GRAMFAST] true when the pre-trim's first trim would run and keep fewer valid tokens than the top-k, so that it
// breaks to the full pass: the candidate-set path's widest set held every token of that trim (its width is at least the
// trim's, its theta at most the trim's) and kept fewer finite ones after the grammar. Then the full pass right away.
static bool common_sampler_gramfast_narrow(struct common_sampler * gsmpl) {
    const auto & sc  = gsmpl->gramfast_sc;
    const auto & cfg = gsmpl->pretrim;
    if (gsmpl->grmr_pretrim == nullptr || cfg.top_k <= 0 || sc.n_fin == SIZE_MAX || sc.width < cfg.width ||
            sc.n_fin >= (size_t) cfg.top_k + cfg.n_extra || v0261_pretrim_narrow()) {
        return false;
    }
    return common_sampler_pretrim_prepare(gsmpl) && (size_t) cfg.top_k + cfg.n_extra + gsmpl->pretrim_bound <= cfg.width;
}

// resampling:
// the sampled token violated the grammar, so sample again with the grammar applied first.
//
// llama_grammar_apply_impl is O(n_vocab): it decodes every candidate and walks it against
// the grammar stacks. Running that over the full ~151K-token vocabulary on every rejection
// is the dominant cost of tool-call generation under stochastic sampling - measured on
// Qwen3.8-27B, tool arguments stream at 8-17 chunks/s with default sampling versus 60-148
// under greedy, where the top token is nearly always grammar-valid so this path never runs.
//
// Pre-trimming to the top-K candidates by logit makes it ~150x cheaper, and it is EXACT
// rather than an approximation whenever at least params.top_k grammar-valid candidates
// survive the trim: every token outside the top-K has a lower logit than every token inside
// it, so the chain's own top-k selects the same set either way and every subsequent sampler
// (top_p / min_p / temp / dist) sees an identical candidate set and an identical
// normalisation. When fewer survive that guarantee does not hold, so we redo the full pass.
// [TAG_V0261_PRETRIM] with samplers that lower logits before the chain's top-k: common_sampler_pretrim_exact.
// [TAG_R29_GRAMFAST] skip_pretrim: common_sampler_gramfast_narrow() said the pre-trim takes the full pass anyway
static llama_token common_sampler_resample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool skip_pretrim) {
    auto & grmr    = gsmpl->grmr;
    auto & rbudget = gsmpl->rbudget;
    auto & chain   = gsmpl->chain;
    auto & cur_p   = gsmpl->cur_p;

    bool resampled = false;

    if (!skip_pretrim && gsmpl->grmr_pretrim && grammar_should_apply(gsmpl) && common_sampler_pretrim_prepare(gsmpl)) {
        gsmpl->set_logits(ctx, idx);

        llama_sampler_apply(rbudget, &cur_p);

        // keep the top-K by logit, grammar over K, not n_vocab
        if (common_sampler_pretrim_grammar(gsmpl, ctx, idx)) {
            llama_sampler_apply(chain, &cur_p);
            resampled = true;
        }
    }

    if (!resampled) {
        gsmpl->set_logits(ctx, idx);

        llama_sampler_apply(rbudget,  &cur_p);

        if (grammar_should_apply(gsmpl)) {
            llama_sampler_apply(grmr,  &cur_p);
        }

        llama_sampler_apply(chain, &cur_p);
    }

    GGML_ASSERT(cur_p.selected != -1 && "no selected token during sampling - check your sampling configuration");

    return cur_p.data[cur_p.selected].id;
}

llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first) {
    llama_synchronize(ctx);

    // start measuring sampling time after the llama_context synchronization in order to not measure any ongoing async operations
    const auto tm = gsmpl->tm();

    llama_token id = LLAMA_TOKEN_NULL;

    auto & grmr  = gsmpl->grmr;
    auto & rbudget = gsmpl->rbudget;
    auto & chain = gsmpl->chain;
    auto & cur_p = gsmpl->cur_p; // initialized by set_logits

    // [TAG_R29_GRAMFAST] a grammar and/or a reasoning budget: the steps below over candidate sets when that is exact. NONE:
    // nothing with state ran, the full path below; RESAMPLE: the chain sampled once, the grammar rejected the token.
    bool skip_pretrim = false;
    if (gsmpl->penfast.top_k > 0 && (grmr != nullptr || rbudget != nullptr)) {
        const common_gramfast_result r = common_sampler_sample_gramfast(gsmpl, ctx, idx, grammar_first, &id);
        if (r == COMMON_GRAMFAST_DONE) {
            return id;
        }
        skip_pretrim = common_sampler_gramfast_narrow(gsmpl);
        if (r == COMMON_GRAMFAST_RESAMPLE) {
            return common_sampler_resample(gsmpl, ctx, idx, skip_pretrim);
        }
    }

    // [TAG_FN_L4_HOST_TOPK] the K candidates straight from the row when that is exact, else the full array
    // [TAG_R28_PENFAST] with penalties / a logit bias: the whole chain over the candidate set when that is exact
    bool chain_done = false;
    if (gsmpl->fast_k > 0 && gsmpl->set_logits_fast(ctx, idx)) {
        // cur_p: the state the chain's top-k leaves
    } else if (gsmpl->penfast.top_k > 0 && grmr == nullptr && rbudget == nullptr && gsmpl->sample_penfast(ctx, idx)) {
        chain_done = true;
    } else {
        gsmpl->set_logits(ctx, idx);
    }

    // [TAG_FN_L21_NANGUARD] LLAMA_FN_L21_NANGUARD (default 1): a NaN hidden state makes every logit of the row NaN (the l19
    // collapse sampled token 0, "!!!!"), so 9 candidates (first, last, 7 strided) are enough to see it; -inf is a legal
    // masked logit, so only NaN and +inf count. The caller takes the count (common_sampler_take_nonfinite) and reacts.
    {
        static const bool l21_nanguard = [] { const char * e = getenv("LLAMA_FN_L21_NANGUARD"); return !(e && e[0] && atoi(e) == 0); }();
        // [TAG_R28_PENFAST] chain_done: the row has no NaN and no +inf (checked), the full path counts nothing either
        if (l21_nanguard && cur_p.size > 0 && !chain_done) {
            const size_t n = cur_p.size;
            bool bad = false;
            for (int k = 0; k < 9 && !bad; ++k) {
                const float v = cur_p.data[k == 8 ? n - 1 : (n * (size_t) k) / 8].logit;
                bad = std::isnan(v) || (std::isinf(v) && v > 0.0f);
            }
            gsmpl->n_nonfinite += bad ? 1 : 0;
        }
    }

    // Check if a backend sampler has already sampled a token in which case we
    // return that token id directly.
    {
        id = llama_get_sampled_token_ith(ctx, idx);

        if (id != LLAMA_TOKEN_NULL) {
            LOG_DBG("%s: Backend sampler selected token: '%d'. Will not run any CPU samplers\n", __func__, id);

            // [TAG_BS_LAZY_GRAMMAR] a backend-sampled token is only exact while common_sampler_backend_ok() holds; the
            // accept loops stop at the token that ends it and the server detaches the backend sampler right after
            GGML_ASSERT(common_sampler_backend_ok(gsmpl) && "backend-sampled token with an active grammar or a forcing reasoning budget");

            for (size_t i = 0; i < cur_p.size; ++i) {
                if (cur_p.data[i].id == id) {
                    cur_p.selected = i;
                    break;
                }
            }

            return id;
        }
    }

    // apply reasoning budget first
    llama_sampler_apply(rbudget, &cur_p);

    if (grammar_first && grammar_should_apply(gsmpl)) {
        // [TAG_GRAMMAR_FIRST_PRETRIM] Trim to the top-K by logit before walking the
        // grammar, for the same reason the rejection path below does.
        //
        // llama_grammar_apply_impl is O(n_vocab): it decodes every candidate and walks it
        // against the grammar stacks. grammar_first makes that run on EVERY sampled
        // position rather than only on a rejection - and the speculative residual-sampling
        // path passes grammar_first = true once per draft position, so a single decode step
        // pays it up to n_draft+1 times. With a 248K vocabulary that is the dominant cost
        // of streaming a tool call's arguments.
        //
        // Measured, Qwen3.8-27B UD-Q5_K_XL, DFlash2 n_max 7, tool-call arguments streaming:
        //   stochastic sampling   20-25 ms/chunk   40-50 tok/s
        //   greedy (no residual)   5.3-6.3 ms/chunk 159-187 tok/s
        // Greedy is unaffected because it takes the non-residual overload, which leaves
        // grammar_first false and only grammar-checks the single sampled token.
        //
        // The trim is exact under the same argument used below: every token outside the
        // top-K has a lower logit than every token inside it, so once at least top_k
        // grammar-valid candidates survive, the chain's own top_k selects exactly the set
        // it would have selected from the full vocabulary. When fewer survive, that
        // guarantee does not hold and we redo the full pass.
        // [TAG_V0261_PRETRIM] the samplers before the chain's top_k may lower logits: the survivors they can touch do
        // not count (common_sampler_pretrim_exact), and a wider second trim runs before the full pass.
        // [TAG_R29_GRAMFAST] skip_pretrim: common_sampler_gramfast_narrow() said the pre-trim takes the full pass anyway
        bool grammar_applied = false;

        if (!skip_pretrim && gsmpl->grmr_pretrim && common_sampler_pretrim_prepare(gsmpl)) {
            grammar_applied = common_sampler_pretrim_grammar(gsmpl, ctx, idx);

            if (!grammar_applied) {
                // not enough survived to guarantee exactness - restore and do it properly
                gsmpl->set_logits(ctx, idx);
                llama_sampler_apply(rbudget, &cur_p);
            }
        }

        if (!grammar_applied) {
            llama_sampler_apply(grmr, &cur_p);
        }
    }

    if (!chain_done) { // [TAG_R28_PENFAST]
        llama_sampler_apply(chain, &cur_p);
    }

    id = cur_p.data[cur_p.selected].id;

    if (grammar_first || !grammar_should_apply(gsmpl)) {
        return id;
    }

    // check if it the sampled token fits the grammar (grammar-based rejection sampling)
    {
        llama_token_data       single_token_data       = { id, 1.0f, 0.0f };
        llama_token_data_array single_token_data_array = { &single_token_data, 1, -1, false };

        llama_sampler_apply(grmr, &single_token_data_array);

        const bool is_valid = single_token_data_array.data[0].logit != -INFINITY;
        if (is_valid) {
            return id;
        }
    }

    // resampling: the sampled token violated the grammar, sample again with the grammar applied first
    return common_sampler_resample(gsmpl, ctx, idx, skip_pretrim);
}

// [TAG_BS_LAZY_GRAMMAR] true when the backend sampler touched this output row. A fully offloaded chain leaves a sampled
// token. A chain that is only partly offloadable (typical_p, xtc, top_n_sigma or adaptive-p after top_k, or penalties
// with several outputs per sequence) samples no token but still leaves filtered logits or probs for the CPU to finish.
// Either way the row was prepared on the GPU without any constraint that became active on an earlier row of the pass.
// Keying the stop on the sampled token alone missed the partial case: the rows after a grammar trigger were then
// sampled from GPU-filtered candidates (inexact, and all -inf when no survivor was grammar-valid). Found in review.
static bool bs_row_on_backend(struct llama_context * ctx, int idx) {
    return llama_get_sampled_token_ith (ctx, idx) != LLAMA_TOKEN_NULL ||
           llama_get_sampled_logits_ith(ctx, idx) != nullptr ||
           llama_get_sampled_probs_ith (ctx, idx) != nullptr;
}

// [TAG_BS_LAZY_GRAMMAR] see sampling.h
bool common_sampler_backend_row_stale(const struct common_sampler * gsmpl, struct llama_context * ctx, int idx) {
    return bs_row_on_backend(ctx, idx) && !common_sampler_backend_ok(gsmpl);
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first) {
    GGML_ASSERT(idxs.size() == draft.size() + 1 && "idxs.size() must be draft.size() + 1");

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));

    size_t i = 0;
    for (; i < draft.size(); i++) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);
        common_sampler_eog_trace(gsmpl, ctx, id, draft[i] == id ? "match" : "target", -1.0f); // [TAG_FN_R2_EOG_TRACE]

        // do not accept draft tokens after an EOG - they are not output but would stay in the context
        // on replay the last token is from the target and can be EOG, so a trailing EOG is still accepted
        if (draft[i] != id || (llama_vocab_is_eog(vocab, id) && i + 1 < draft.size())) {
            break;
        }

        // [TAG_BS_LAZY_GRAMMAR] every row of this step went through the backend sampler in one pass. If this accepted
        // token triggered the lazy grammar (or started reasoning-budget forcing), the next rows were prepared without
        // that constraint: stop here, exactly like a rejection at the next position.
        if (bs_row_on_backend(ctx, idxs[i + 1]) && !common_sampler_backend_ok(gsmpl)) {
            break;
        }
    }

    if (i == draft.size()) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);
        common_sampler_eog_trace(gsmpl, ctx, id, "bonus", -1.0f); // [TAG_FN_R2_EOG_TRACE]
    }

    return result;
}

std::vector<llama_token> common_sampler_sample_and_accept_n(
        struct common_sampler * gsmpl,
        struct llama_context * ctx,
        const std::vector<int> & idxs,
        const llama_tokens & draft,
        const std::vector<common_speculative_token_dist> & dists,
        bool grammar_first) {
    GGML_ASSERT(idxs.size() == draft.size() + 1);
    GGML_ASSERT(dists.size() == draft.size());

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));

    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    size_t i = 0;
    for (; i < draft.size(); ++i) {
        // Residual sampling needs the target distribution after every constraint.
        const llama_token fallback = common_sampler_sample(gsmpl, ctx, idxs[i], true);
        const auto & q = dists[i];
        GGML_ASSERT(q.ids.size() == q.probs.size());

        std::unordered_map<llama_token, float> q_probs;
        q_probs.reserve(q.ids.size());
        for (size_t j = 0; j < q.ids.size(); ++j) {
            q_probs[q.ids[j]] += q.probs[j];
        }
        const auto q_prob = [&](llama_token id) {
            const auto it = q_probs.find(id);
            return it == q_probs.end() ? 0.0f : it->second;
        };

        auto * p = common_sampler_get_candidates(gsmpl, false);
        float p_draft = 0.0f;
        const float q_draft = q_prob(draft[i]);
        for (size_t j = 0; j < p->size; ++j) {
            if (p->data[j].id == draft[i]) {
                p_draft = p->data[j].p;
                break;
            }
        }

        // strict: uniform() can return 0, and a token the target gives p == 0 must never be accepted
        if (q_draft > 0.0f && uniform(gsmpl->rng) * q_draft < p_draft) {
            common_sampler_accept(gsmpl, draft[i], true);
            result.push_back(draft[i]);
            common_sampler_eog_trace(gsmpl, ctx, draft[i], "couple-accept", q_draft); // [TAG_FN_R2_EOG_TRACE]
            // [TAG_SYNC_1004] upstream #29638's stop, as in the greedy overload: no draft token is accepted after an
            // EOG - it would not be output but would stay in the context
            if (llama_vocab_is_eog(vocab, draft[i]) && i + 1 < draft.size()) {
                break;
            }
            // [TAG_BS_LAZY_GRAMMAR] same stop as the greedy overload: the rows after this token went through the
            // backend sampler without the constraint this token just activated
            if (bs_row_on_backend(ctx, idxs[i + 1]) && !common_sampler_backend_ok(gsmpl)) {
                break;
            }
            continue;
        }

        std::vector<float> residual(p->size);
        float residual_sum = 0.0f;
        for (size_t j = 0; j < p->size; ++j) {
            residual[j] = std::max(0.0f, p->data[j].p - q_prob(p->data[j].id));
            residual_sum += residual[j];
        }

        llama_token id = fallback;
        size_t       id_idx = SIZE_MAX;   // [TAG_SAMPLER_CLONE_CURP] which branch produced `id`
        if (residual_sum > 0.0f) {
            std::discrete_distribution<size_t> sample(residual.begin(), residual.end());
            id_idx = sample(gsmpl->rng);
            id = p->data[id_idx].id;
        }

        // [TAG_SAMPLER_CLONE_CURP] An out-of-range id here is a candidate-buffer problem, not a
        // model output: llama_token_data is {int32 id; float logit; float p;}, so reading the wrong
        // field yields float bit patterns (0x3F800000 = 1.0f, 0xFF800000 = -inf). Say which branch
        // and what the candidate array looked like, rather than letting it surface 12 frames later
        // as "invalid vector subscript" from the vocab lookup.
        {
            const int n_vocab_dbg = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
            if (id < 0 || id >= n_vocab_dbg) {
                LOG_ERR("[TAG_SAMPLER_CLONE_CURP] residual sampling produced id %d (0x%08X), outside "
                        "[0, %d): branch=%s, p->size=%zu, p->sorted=%d, residual_sum=%g, idx=%zu, "
                        "fallback=%d, draft_pos=%zu/%zu\n",
                        (int) id, (unsigned) id, n_vocab_dbg,
                        id_idx == SIZE_MAX ? "fallback" : "residual",
                        (size_t) p->size, (int) p->sorted, residual_sum,
                        id_idx, (int) fallback, i, draft.size());
            }
        }

        common_sampler_accept(gsmpl, id, true);
        result.push_back(id);
        common_sampler_eog_trace(gsmpl, ctx, id, id_idx == SIZE_MAX ? "couple-fallback" : "couple-residual", q_prob(id)); // [TAG_FN_R2_EOG_TRACE]
        break;
    }

    if (i == draft.size()) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);
        common_sampler_accept(gsmpl, id, true);
        result.push_back(id);
        common_sampler_eog_trace(gsmpl, ctx, id, "bonus", -1.0f); // [TAG_FN_R2_EOG_TRACE]
    }

    return result;
}

static float prob_of(const llama_token_data * data, size_t n, llama_token id) {
    for (size_t k = 0; k < n; ++k) {
        if (data[k].id == id) {
            return data[k].p;
        }
    }
    return 0.0f;
}

// [TAG_FN_R2_EOG_TRACE] LLAMA_EOG_TRACE=1: one log line per end-of-generation token a sampler emits - how it was chosen,
// its probability in the target's candidate array (cur_p after the chain), the draft's probability, the top candidates
void common_sampler_eog_trace(const struct common_sampler * gsmpl, const struct llama_context * ctx, llama_token id,
        const char * path, float q_x) {
    static const bool on = [] {
        const char * e = getenv("LLAMA_EOG_TRACE");
        return e && e[0] == '1';
    }();
    if (!on || gsmpl == nullptr) {
        return;
    }
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    if (!llama_vocab_is_eog(vocab, id)) {
        return;
    }
    const auto & cp = gsmpl->cur_p;
    std::vector<llama_token_data> top(cp.data, cp.data + cp.size);
    std::sort(top.begin(), top.end(), [](const llama_token_data & a, const llama_token_data & b) { return a.p > b.p; });
    std::string s;
    for (size_t k = 0; k < top.size() && k < 8; ++k) {
        s += string_format("%s%d:%.4f", k ? " " : "", top[k].id, top[k].p);
    }
    float p_sum = 0.0f;
    for (const auto & e : top) {
        p_sum += e.p;
    }
    LOG_WRN("[TAG_FN_R2_EOG_TRACE] EOG %d via %s: p_target %.6f q_draft %.6f, %zu candidates (p sum %.4f), top: %s\n",
            id, path, prob_of(cp.data, cp.size, id), q_x, (size_t) cp.size, p_sum, s.c_str());
}

// Accept a drafted token with probability min(1, p/q), else draw from norm(max(0, p - q)).
// Preserves the target distribution exactly, and accepts more often than matching does when the
// draft samples instead of taking its argmax.
std::vector<llama_token> common_sampler_sample_and_accept_n_rejection(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, const std::vector<std::vector<llama_token_data>> & draft_q, bool grammar_first) {
    GGML_ASSERT(idxs.size()    == draft.size() + 1 && "idxs.size() must be draft.size() + 1");
    GGML_ASSERT(draft_q.size() == draft.size() && "draft_q must have one entry per draft token");

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    // draws come from the sampler's own stream, so they stay independent of what was drafted
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    std::vector<llama_token_data> residual;

    std::vector<llama_token_data> cand; // candidate array masked by the grammar, if there is one

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));

    size_t i = 0;
    for (; i < draft.size(); i++) {
        // leaves the target distribution in the candidate array
        const llama_token id_tgt = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        const auto * cur_p = common_sampler_get_candidates(gsmpl, true);
        const auto & q     = draft_q[i];

        const bool masked = !grammar_first && grammar_should_apply(gsmpl);
        if (masked) {
            cand.assign(cur_p->data, cur_p->data + cur_p->size);
            llama_token_data_array arr = { cand.data(), cand.size(), -1, false };
            llama_sampler_apply(gsmpl->grmr, &arr);
        }

        // a candidate the grammar rejects carries no probability, whatever the target thinks
        auto p_raw = [&](size_t k) {
            return masked && cand[k].logit == -INFINITY ? 0.0f : cur_p->data[k].p;
        };

        // masking drops probability mass, so rescale what is left or the residual is over-weighted
        float p_sum = 0.0f;
        if (masked) {
            for (size_t k = 0; k < cur_p->size; ++k) {
                p_sum += p_raw(k);
            }
        }

        const float p_norm = masked && p_sum > 0.0f ? 1.0f/p_sum : 1.0f;

        auto p_of = [&](size_t k) {
            return p_raw(k)*p_norm;
        };

        // q_x is never 0 for a token the draft produced, but guard the divide
        const float q_x = prob_of(q.data(), q.size(), draft[i]);

        float p_x = 0.0f;
        for (size_t k = 0; k < cur_p->size; ++k) {
            if (cur_p->data[k].id == draft[i]) {
                p_x = p_of(k);
                break;
            }
        }

        if (q_x > 0.0f && (p_x >= q_x || uni(gsmpl->rng) < p_x / q_x)) {
            common_sampler_accept(gsmpl, draft[i], true);
            result.push_back(draft[i]);
            common_sampler_eog_trace(gsmpl, ctx, draft[i], "reject-accept", q_x); // [TAG_FN_R2_EOG_TRACE]
            // [TAG_SYNC_1004] upstream #29638's stop, as in the greedy overload: no draft token after an accepted EOG
            if (llama_vocab_is_eog(vocab, draft[i]) && i + 1 < draft.size()) {
                break;
            }
            // [TAG_BS_LAZY_GRAMMAR] same stop as the other accept loops: the rows after this token went through the
            // backend sampler without the constraint this token just activated
            if (bs_row_on_backend(ctx, idxs[i + 1]) && !common_sampler_backend_ok(gsmpl)) {
                break;
            }
            continue;
        }

        // rejected: tokens outside q's support keep all of p
        residual.clear();
        float sum = 0.0f;
        for (size_t k = 0; k < cur_p->size; ++k) {
            const float r = p_of(k) - prob_of(q.data(), q.size(), cur_p->data[k].id);
            if (r > 0.0f) {
                residual.push_back({ cur_p->data[k].id, 0.0f, r });
                sum += r;
            }
        }

        llama_token id = id_tgt;
        if (sum > 0.0f) {
            float u = uni(gsmpl->rng) * sum;
            id = residual.back().id;
            for (const auto & e : residual) {
                u -= e.p;
                if (u <= 0.0f) {
                    id = e.id;
                    break;
                }
            }
        }

        common_sampler_accept(gsmpl, id, true);
        result.push_back(id);
        common_sampler_eog_trace(gsmpl, ctx, id, sum > 0.0f ? "reject-residual" : "reject-fallback",
                prob_of(q.data(), q.size(), id)); // [TAG_FN_R2_EOG_TRACE]

        break;
    }

    if (i == draft.size()) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);
        common_sampler_eog_trace(gsmpl, ctx, id, "bonus", -1.0f); // [TAG_FN_R2_EOG_TRACE]
    }

    return result;
}

// [TAG_FN_L3_MTP_BLOCK] LLAMA_EOG_TRACE=1 for the block rule: the row's own target distribution (the sampler's candidate
// array holds another row's when the block rule decides)
static void block_eog_trace(const struct llama_context * ctx, llama_token id, const char * path, float q_x,
        const std::vector<llama_token_data> & p) {
    static const bool on = [] {
        const char * e = getenv("LLAMA_EOG_TRACE");
        return e && e[0] == '1';
    }();
    if (!on || !llama_vocab_is_eog(llama_model_get_vocab(llama_get_model(ctx)), id)) {
        return;
    }
    std::vector<llama_token_data> top(p);
    std::sort(top.begin(), top.end(), [](const llama_token_data & a, const llama_token_data & b) { return a.p > b.p; });
    std::string s;
    float p_sum = 0.0f;
    for (size_t k = 0; k < top.size(); ++k) {
        if (k < 8) {
            s += string_format("%s%d:%.4f", k ? " " : "", top[k].id, top[k].p);
        }
        p_sum += top[k].p;
    }
    LOG_WRN("[TAG_FN_R2_EOG_TRACE] EOG %d via %s: p_target %.6f q_draft %.6f, %zu candidates (p sum %.4f), top: %s\n",
            id, path, prob_of(p.data(), p.size(), id), q_x, p.size(), p_sum, s.c_str());
}

// [TAG_FN_L3_MTP_BLOCK] block verification of a sampled draft (common/speculative-block.h). An empty result means it does
// not apply here (fewer than 2 drafts, a grammar or a reasoning budget, a row the backend sampler touched) and the caller
// takes the per-token rule. A scratch copy of the sampler walks the drafts to get the target's distribution at each row
// (the penalties see the same history as with the per-token rule); the real sampler then accepts the kept drafts and the
// token after them. The scratch sampler and the buffers are kept between steps: a step allocates nothing (a clone per
// step copied the 248K-entry candidate buffer into fresh pages, ~0.5 ms on Windows).
// [TAG_SPEC_DFL_BLOCK] dfl: a sampled draft with dists (DFlash2, draft-mtp with TURBO_MTP_DISTS) under SPEC_DFL_BLOCK_VERIFY
static std::vector<llama_token> block_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx,
        const std::vector<int> & idxs, const llama_tokens & draft, const std::vector<std::vector<llama_token_data>> & draft_q, bool dfl) {
    GGML_ASSERT(idxs.size()    == draft.size() + 1 && "idxs.size() must be draft.size() + 1");
    GGML_ASSERT(draft_q.size() == draft.size() && "draft_q must have one entry per draft token");

    const int gamma = (int) draft.size();
    if (gamma < 2 || gsmpl->grmr || gsmpl->rbudget) {
        return {};
    }
    const auto & sp = gsmpl->params;
    // [TAG_SPEC_DFL_BLOCK] samplers that move with their own draws (mirostat, adaptive-p, an active XTC) stay on the per-token
    // rule: the scratch copy replays their streams every step
    if (dfl && (sp.mirostat != 0 ||
            std::find(sp.samplers.begin(), sp.samplers.end(), COMMON_SAMPLER_TYPE_ADAPTIVE_P) != sp.samplers.end() ||
            (sp.xtc_probability > 0.0f && sp.xtc_threshold <= 0.5f &&
             std::find(sp.samplers.begin(), sp.samplers.end(), COMMON_SAMPLER_TYPE_XTC) != sp.samplers.end()))) {
        return {};
    }
    // [TAG_SPEC_DFL_BLOCK] a block the backend sampler sampled whole: each row's probs are the target's distribution there
    // (what the per-token rule reads), exact only while the chain keeps no history (no penalties, no DRY)
    const bool bs = dfl && bs_row_on_backend(ctx, idxs[0]) &&
        (sp.penalty_last_n == 0 || (sp.penalty_repeat == 1.0f && sp.penalty_freq == 0.0f && sp.penalty_present == 0.0f)) &&
        (sp.dry_multiplier == 0.0f || sp.dry_base < 1.0f || sp.dry_penalty_last_n == 0);
    for (int i = 0; i <= gamma; ++i) {
        if (bs ? llama_get_sampled_token_ith(ctx, idxs[i]) == LLAMA_TOKEN_NULL : bs_row_on_backend(ctx, idxs[i])) {
            return {};
        }
    }

    // the scratch sampler: a clone the first time, then the chain's state (penalty counts, RNG, ...) and the history
    // copied in; its candidate buffer stays allocated
    if (bs) {
        // [TAG_SPEC_DFL_BLOCK] no CPU chain runs on these rows: they are read through the sampler itself, nothing accepted
    } else if (gsmpl->blk == nullptr) {
        gsmpl->blk = common_sampler_clone(gsmpl);
    } else {
        llama_sampler_copy(gsmpl->chain, gsmpl->blk->chain);
        gsmpl->blk->prev = gsmpl->prev;
    }
    common_sampler * s = bs ? gsmpl : gsmpl->blk;

    // the target's distribution at rows 0..gamma-1, each after the drafts before it
    static thread_local std::vector<std::vector<llama_token_data>> p;
    if (p.size() < (size_t) gamma) {
        p.resize((size_t) gamma);
    }
    const uint32_t nf0 = s->n_nonfinite;
    for (int i = 0; i < gamma; ++i) {
        common_sampler_sample(s, ctx, idxs[i], false);
        const auto * cur_p = common_sampler_get_candidates(s, false);
        if (dfl && cur_p->size > COMMON_SPEC_BLOCK_MAX_CAND) {
            s->n_nonfinite = nf0;
            return {}; // [TAG_V0262_BLOCKVERIFY] a long row (no top_k in the chain): the per-token rule, no full-vocab copies
        }
        p[i].assign(cur_p->data, cur_p->data + cur_p->size);
        if (dfl && s->n_nonfinite != nf0) {
            s->n_nonfinite = nf0;
            return {}; // [TAG_SPEC_DFL_BLOCK] a non-finite row: the per-token rule decides and counts the rows it reads
        }
        if (bs) {
            // [TAG_SPEC_DFL_BLOCK] a row whose backend arrays did not line up came back as plain logits (p = 0); NaN fails too
            double sum = 0.0;
            for (const auto & e : p[i]) {
                sum += e.p;
            }
            if (!(std::fabs(sum - 1.0) <= 1e-3)) {
                return {};
            }
            continue;
        }
        common_sampler_accept(s, draft[i], true);
    }

    static thread_local std::vector<common_spec_block_dist> pd, qd;
    pd.resize((size_t) gamma);
    qd.resize((size_t) gamma);
    for (int i = 0; i < gamma; ++i) {
        pd[i] = { p[i].data(),       p[i].size() };
        qd[i] = { draft_q[i].data(), draft_q[i].size() };
    }
    static thread_local std::vector<double> w, h;
    common_spec_block_weights(pd.data(), qd.data(), draft.data(), gamma, w, h);

    // draws from the sampler's own stream, independent of what was drafted
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    static thread_local std::vector<float> u;
    u.resize((size_t) gamma);
    for (float & x : u) {
        x = uni(gsmpl->rng);
    }
    const int tau = common_spec_block_tau(h, gamma, u.data());

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));

    std::vector<llama_token> result;
    result.reserve((size_t) tau + 1);
    for (int i = 0; i < tau; ++i) {
        common_sampler_accept(gsmpl, draft[i], true);
        result.push_back(draft[i]);
        block_eog_trace(ctx, draft[i], "block-accept", (float) common_spec_block_prob(qd[i], draft[i]), p[i]); // [TAG_FN_R2_EOG_TRACE]
        if (llama_vocab_is_eog(vocab, draft[i])) {
            return result; // nothing after an accepted end of generation
        }
    }

    llama_token id = LLAMA_TOKEN_NULL;
    if (tau == gamma) {
        id = common_sampler_sample(gsmpl, ctx, idxs[gamma], false);
        common_sampler_eog_trace(gsmpl, ctx, id, "block-bonus", -1.0f); // [TAG_FN_R2_EOG_TRACE]
    } else {
        // max(w_tau * p_tau - q_tau, 0); empty only through rounding (then p_tau itself)
        static thread_local std::vector<llama_token_data> r;
        const double sum = common_spec_block_excess(w[tau], pd[tau], qd[tau], &r);
        const std::vector<llama_token_data> & from = sum > 0.0 ? r : p[tau];
        double tot = 0.0;
        for (const auto & e : from) {
            tot += e.p;
        }
        double x = (double) uni(gsmpl->rng) * tot;
        id = from.empty() ? LLAMA_TOKEN_NULL : from.back().id;
        for (const auto & e : from) {
            x -= e.p;
            if (x <= 0.0) {
                id = e.id;
                break;
            }
        }
        if (id == LLAMA_TOKEN_NULL) {
            // no candidate at all at this row (cannot happen with a sampled row): the row's own sample
            id = common_sampler_sample(gsmpl, ctx, idxs[tau], false);
        }
        block_eog_trace(ctx, id, "block-residual", (float) common_spec_block_prob(qd[tau], id), p[tau]); // [TAG_FN_R2_EOG_TRACE]
    }
    common_sampler_accept(gsmpl, id, true);
    result.push_back(id);

    return result;
}

std::vector<llama_token> common_sampler_sample_and_accept_n_block(struct common_sampler * gsmpl, struct llama_context * ctx,
        const std::vector<int> & idxs, const llama_tokens & draft, const std::vector<std::vector<llama_token_data>> & draft_q) {
    return block_accept_n(gsmpl, ctx, idxs, draft, draft_q, false);
}

// [TAG_SPEC_DFL_BLOCK] q per draft token from its dists entry, a token listed twice summed as the per-token rule sums it
std::vector<llama_token> common_sampler_sample_and_accept_n_block(struct common_sampler * gsmpl, struct llama_context * ctx,
        const std::vector<int> & idxs, const llama_tokens & draft, const std::vector<common_speculative_token_dist> & dists) {
    GGML_ASSERT(dists.size() == draft.size() && "dists must have one entry per draft token");

    // [TAG_V0262_BLOCKVERIFY] the refusals that need no row first; top_k <= 0 leaves full-vocabulary rows, and a long
    // proposal makes the excess sums |p| x |q|: the per-token rule
    if (draft.size() < 2 || gsmpl->grmr || gsmpl->rbudget || gsmpl->params.top_k <= 0) {
        return {};
    }
    for (const auto & d : dists) {
        if (d.ids.size() > COMMON_SPEC_BLOCK_MAX_CAND) {
            return {};
        }
    }

    static thread_local std::vector<std::vector<llama_token_data>> q;
    q.resize(dists.size());
    for (size_t i = 0; i < dists.size(); ++i) {
        const auto & d = dists[i];
        GGML_ASSERT(d.ids.size() == d.probs.size());
        common_spec_block_q_from_ids(d.ids.data(), d.probs.data(), d.ids.size(), q[i]);
    }

    return block_accept_n(gsmpl, ctx, idxs, draft, q, true);
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const llama_tokens & draft, bool grammar_first) {
    std::vector<int> idxs(draft.size() + 1);
    for (size_t i = 0; i < idxs.size(); ++i) {
        idxs[i] = i;
    }

    return common_sampler_sample_and_accept_n(gsmpl, ctx, idxs, draft, grammar_first);
}

std::vector<llama_token> common_sampler_sample_and_accept_n(
        struct common_sampler * gsmpl,
        struct llama_context * ctx,
        const llama_tokens & draft,
        const std::vector<common_speculative_token_dist> & dists,
        bool grammar_first) {
    std::vector<int> idxs(draft.size() + 1);
    for (size_t i = 0; i < idxs.size(); ++i) {
        idxs[i] = i;
    }
    return common_sampler_sample_and_accept_n(gsmpl, ctx, idxs, draft, dists, grammar_first);
}

uint32_t common_sampler_get_seed(const struct common_sampler * gsmpl) {
    return llama_sampler_get_seed(gsmpl->chain);
}

bool common_sampler_reasoning_budget_force(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return false;
    }

    return common_reasoning_budget_force(gsmpl->rbudget);
}

// helpers

llama_token_data_array * common_sampler_get_candidates(struct common_sampler * gsmpl, bool do_sort) {
    const auto tm = gsmpl->tm();

    auto * res = &gsmpl->cur_p;

    if (do_sort && !res->sorted) {
        // remember the selected token before sorting
        const llama_token id = res->data[res->selected].id;

        std::sort(res->data, res->data + res->size, [](const llama_token_data & a, const llama_token_data & b) {
            return a.p > b.p;
        });

        // restore the selected token after sorting
        for (size_t i = 0; i < res->size; ++i) {
            if (res->data[i].id == id) {
                res->selected = i;
                break;
            }
        }

        res->sorted = true;
    }

    return res;
}

llama_token common_sampler_last(const struct common_sampler * gsmpl) {
    return gsmpl->prev.rat(0);
}

std::string common_sampler_print(const struct common_sampler * gsmpl) {
    std::string result = "logits ";

    for (int i = 0; i < llama_sampler_chain_n(gsmpl->chain); i++) {
        const auto * smpl = llama_sampler_chain_get(gsmpl->chain, i);
        result += std::string("-> ");
        result += std::string(llama_sampler_name(smpl)) + " ";
    }

    return result;
}

std::string common_sampler_prev_str(common_sampler * gsmpl, llama_context * ctx_main, int n) {
    n = std::min(n, (int) gsmpl->prev.size());

    if (n <= 0) {
        return "";
    }

    std::string result;
    result.reserve(8*n); // 8 is the average length of a token [citation needed], TODO: compute this from the vocab

    for (int i = n - 1; i >= 0; i--) {
        const llama_token id = gsmpl->prev.rat(i);

        GGML_ASSERT(id != LLAMA_TOKEN_NULL && "null token in the sampling history - should not happen");

        result += common_token_to_piece(ctx_main, id);
    }

    return result;
}

char common_sampler_type_to_chr(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:         return 'd';
        case COMMON_SAMPLER_TYPE_TOP_K:       return 'k';
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return 'y';
        case COMMON_SAMPLER_TYPE_TOP_P:       return 'p';
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: return 's';
        case COMMON_SAMPLER_TYPE_MIN_P:       return 'm';
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return 't';
        case COMMON_SAMPLER_TYPE_XTC:         return 'x';
        case COMMON_SAMPLER_TYPE_INFILL:      return 'i';
        case COMMON_SAMPLER_TYPE_PENALTIES:   return 'e';
        case COMMON_SAMPLER_TYPE_ADAPTIVE_P:  return 'a';
        default : return '?';
    }
}

std::string common_sampler_type_to_str(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:         return "dry";
        case COMMON_SAMPLER_TYPE_TOP_K:       return "top_k";
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return "typ_p";
        case COMMON_SAMPLER_TYPE_TOP_P:       return "top_p";
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: return "top_n_sigma";
        case COMMON_SAMPLER_TYPE_MIN_P:       return "min_p";
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return "temperature";
        case COMMON_SAMPLER_TYPE_XTC:         return "xtc";
        case COMMON_SAMPLER_TYPE_INFILL:      return "infill";
        case COMMON_SAMPLER_TYPE_PENALTIES:   return "penalties";
        case COMMON_SAMPLER_TYPE_ADAPTIVE_P:  return "adaptive_p";
        default : return "";
    }
}

std::vector<common_sampler_type> common_sampler_types_from_names(const std::vector<std::string> & names) {
    // sampler names can be written multiple ways; generate aliases from canonical names
    static const auto sampler_name_map = []{
        // canonical sampler name mapping
        std::unordered_map<std::string, common_sampler_type> canonical_name_map {
            { "dry",         COMMON_SAMPLER_TYPE_DRY         },
            { "top_k",       COMMON_SAMPLER_TYPE_TOP_K       },
            { "top_p",       COMMON_SAMPLER_TYPE_TOP_P       },
            { "top_n_sigma", COMMON_SAMPLER_TYPE_TOP_N_SIGMA },
            { "typ_p",       COMMON_SAMPLER_TYPE_TYPICAL_P   },
            { "min_p",       COMMON_SAMPLER_TYPE_MIN_P       },
            { "temperature", COMMON_SAMPLER_TYPE_TEMPERATURE },
            { "xtc",         COMMON_SAMPLER_TYPE_XTC         },
            { "infill",      COMMON_SAMPLER_TYPE_INFILL      },
            { "penalties",   COMMON_SAMPLER_TYPE_PENALTIES   },
            { "adaptive_p",  COMMON_SAMPLER_TYPE_ADAPTIVE_P  }
        };
        std::unordered_map<std::string, common_sampler_type> alias_name_map;
        for (const auto & entry : canonical_name_map) {
            const std::string & canonical = entry.first;
            if (canonical.find('_') == std::string::npos) {
                continue;
            }
            // kebab-case: "top-k", "min-p", etc.
            {
                std::string kebab_case = canonical;
                std::replace(kebab_case.begin(), kebab_case.end(), '_', '-');
                alias_name_map.insert({kebab_case, entry.second});
            }
            // no dash: "topk", "minp", etc.
            {
                std::string no_dash = canonical;
                no_dash.erase(std::remove(no_dash.begin(), no_dash.end(), '_'), no_dash.end());
                alias_name_map.insert({no_dash, entry.second});
            }
        }
        // misc. aliases
        alias_name_map.insert({"nucleus", COMMON_SAMPLER_TYPE_TOP_P});
        alias_name_map.insert({"temp",    COMMON_SAMPLER_TYPE_TEMPERATURE});
        alias_name_map.insert({"typ",     COMMON_SAMPLER_TYPE_TYPICAL_P});
        // include aliases + canonical names in the complete mapping
        alias_name_map.merge(canonical_name_map);
        return alias_name_map;
    }();

    std::vector<common_sampler_type> samplers;
    samplers.reserve(names.size());

    for (const auto & name : names) {
        std::string name_lower = name;
        std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
        auto sampler = sampler_name_map.find(name_lower);
        if (sampler != sampler_name_map.end()) {
            samplers.push_back(sampler->second);
            continue;
        }
        LOG_WRN("%s: unable to match sampler by name '%s'\n", __func__, name_lower.c_str());
    }

    return samplers;
}

std::vector<common_sampler_type> common_sampler_types_from_chars(const std::string & chars) {
    std::unordered_map<char, common_sampler_type> sampler_name_map = {
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_DRY),         COMMON_SAMPLER_TYPE_DRY },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_K),       COMMON_SAMPLER_TYPE_TOP_K },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TYPICAL_P),   COMMON_SAMPLER_TYPE_TYPICAL_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_P),       COMMON_SAMPLER_TYPE_TOP_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_N_SIGMA), COMMON_SAMPLER_TYPE_TOP_N_SIGMA },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_MIN_P),       COMMON_SAMPLER_TYPE_MIN_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TEMPERATURE), COMMON_SAMPLER_TYPE_TEMPERATURE },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_XTC),         COMMON_SAMPLER_TYPE_XTC },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_INFILL),      COMMON_SAMPLER_TYPE_INFILL },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_PENALTIES),   COMMON_SAMPLER_TYPE_PENALTIES },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_ADAPTIVE_P),  COMMON_SAMPLER_TYPE_ADAPTIVE_P },
    };

    std::vector<common_sampler_type> samplers;
    samplers.reserve(chars.size());

    for (const auto & c : chars) {
        const auto sampler = sampler_name_map.find(c);
        if (sampler != sampler_name_map.end()) {
            samplers.push_back(sampler->second);
        } else {
            LOG_WRN("%s: unable to match sampler by char '%c'\n", __func__, c);
        }
    }

    return samplers;
}

void common_sampling_add_preserved_tokens(common_params_sampling & sampling, const llama_vocab * vocab, const std::vector<std::string> & tokens) {
    GGML_ASSERT(vocab != nullptr);
    for (const auto & t : tokens) {
        auto ids = common_tokenize(vocab, t, false, true);
        if (ids.size() == 1) {
            sampling.preserved_tokens.insert(ids[0]);
        }
    }
}

void common_sampling_add_grammar_triggers(common_params_sampling & sampling, const llama_vocab * vocab, std::vector<common_grammar_trigger> triggers) {
    GGML_ASSERT(vocab != nullptr);
    for (auto & trigger : triggers) {
        if (trigger.type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
            const auto & word = trigger.value;
            auto ids = common_tokenize(vocab, word, false, true);
            if (ids.size() == 1) {
                auto token = ids[0];
                if (std::find(sampling.preserved_tokens.begin(), sampling.preserved_tokens.end(), (llama_token) token) == sampling.preserved_tokens.end()) {
                    throw std::runtime_error("Grammar trigger word should be marked as preserved token: " + word);
                }
                common_grammar_trigger token_trigger;
                token_trigger.type  = COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN;
                token_trigger.value = word;
                token_trigger.token = token;
                sampling.grammar_triggers.push_back(std::move(token_trigger));
            } else {
                sampling.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, word});
            }
        } else {
            sampling.grammar_triggers.push_back(std::move(trigger));
        }
    }
    if (sampling.grammar_lazy && sampling.grammar_triggers.empty()) {
        throw std::runtime_error("Error: no triggers set for lazy grammar!");
    }
}
