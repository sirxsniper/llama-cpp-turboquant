// [TAG_R28_PENFAST] the fast candidate path with penalties (common_sampler_penfast_apply, common/sampling.cpp) against the
// full path, with the real libllama samplers in the chain common_sampler_init builds (logit bias with user biases and
// suppress tokens, penalties, DRY, top-n-sigma, top-k, typical, top-p, min-p, XTC, temperature, dist). Random logits over
// 20000..248000 tokens, a random accepted history (window 64 and others, repeats, mostly among the best raw tokens),
// presence / frequency / repeat penalties incl. Kyro's (presence 1.5, top_k 20, top_p 0.95, min_p 0, temp 0.6), raising
// penalties and positive biases. Whenever the fast path runs: the candidates right after the top-k (ids, logits, order),
// the whole result of the chain (ids, logits, probabilities, selected) and the sampled token over several steps with the
// same seed equal the full path; when it refuses, the caller's fallback (the full chain) gives the same as well. Also: the
// fast path runs in nearly every untied case, ties inside the top-k refuse, raised window tokens make the top-k, and the
// configuration falls back for mirostat, n_probs, top_k outside 1..128, DRY / top-n-sigma / top-p / temperature before
// the top-k and a token listed twice in the logit bias. The grammar, the reasoning budget, backend-sampled rows and
// LLAMA_R28_PENFAST=0 are checked by common_sampler_init / common_sampler_sample themselves (they need a model).
// CPU only, no model.

#include "sampling.h"
#include "../src/llama-ext.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <vector>

static int n_fail = 0;

static void check(bool ok, const char * what, int it) {
    if (!ok) {
        printf("FAIL: %s (case %d)\n", what, it);
        n_fail++;
    }
}

// the chain common_sampler_init builds from p (infill and adaptive-p are not used here)
static llama_sampler * make_chain(const common_params_sampling & p, const std::vector<llama_token> & suppress, int n_vocab) {
    llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    std::vector<llama_logit_bias> merged = p.logit_bias;
    for (const llama_token t : suppress) {
        merged.push_back({ t, -INFINITY });
    }
    if (!merged.empty()) {
        llama_sampler_chain_add(chain, llama_sampler_init_logit_bias(n_vocab, (int32_t) merged.size(), merged.data()));
    }
    if (p.mirostat != 0) {
        llama_sampler_chain_add(chain, llama_sampler_init_temp(p.temp));
        llama_sampler_chain_add(chain, llama_sampler_init_mirostat_v2(p.seed, p.mirostat_tau, p.mirostat_eta));
        return chain;
    }
    for (const auto t : p.samplers) {
        switch (t) {
            case COMMON_SAMPLER_TYPE_PENALTIES:
                llama_sampler_chain_add(chain, llama_sampler_init_penalties(n_vocab, p.penalty_last_n, p.penalty_repeat,
                        p.penalty_freq, p.penalty_present));
                break;
            case COMMON_SAMPLER_TYPE_DRY:
                llama_sampler_chain_add(chain, llama_sampler_init_dry(nullptr, p.dry_multiplier, p.dry_base,
                        p.dry_allowed_length, p.dry_penalty_last_n, nullptr, 0));
                break;
            case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:
                llama_sampler_chain_add(chain, llama_sampler_init_top_n_sigma(p.top_n_sigma));
                break;
            case COMMON_SAMPLER_TYPE_TOP_K:
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(p.top_k));
                break;
            case COMMON_SAMPLER_TYPE_TYPICAL_P:
                llama_sampler_chain_add(chain, llama_sampler_init_typical(p.typ_p, p.min_keep));
                break;
            case COMMON_SAMPLER_TYPE_TOP_P:
                llama_sampler_chain_add(chain, llama_sampler_init_top_p(p.top_p, p.min_keep));
                break;
            case COMMON_SAMPLER_TYPE_MIN_P:
                llama_sampler_chain_add(chain, llama_sampler_init_min_p(p.min_p, p.min_keep));
                break;
            case COMMON_SAMPLER_TYPE_XTC:
                llama_sampler_chain_add(chain, llama_sampler_init_xtc(p.xtc_probability, p.xtc_threshold, p.min_keep, p.seed));
                break;
            case COMMON_SAMPLER_TYPE_TEMPERATURE:
                llama_sampler_chain_add(chain, llama_sampler_init_temp_ext(p.temp, p.dynatemp_range, p.dynatemp_exponent));
                break;
            default:
                break;
        }
    }
    llama_sampler_chain_add(chain, llama_sampler_init_dist(p.seed));
    return chain;
}

static void raw_array(const std::vector<float> & x, std::vector<llama_token_data> & cur, llama_token_data_array & a) {
    cur.resize(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        cur[i] = llama_token_data{ (llama_token) i, x[i], 0.0f };
    }
    a = { cur.data(), cur.size(), -1, false };
}

// bit-identical: size, selected, sorted, and per candidate the id, the logit and (with_p) the probability
static bool same_exact(const llama_token_data_array & a, const llama_token_data_array & b, bool with_p) {
    if (a.size != b.size || a.selected != b.selected || a.sorted != b.sorted) {
        return false;
    }
    for (size_t i = 0; i < a.size; ++i) {
        if (a.data[i].id != b.data[i].id || std::memcmp(&a.data[i].logit, &b.data[i].logit, sizeof(float)) != 0) {
            return false;
        }
        if (with_p && std::memcmp(&a.data[i].p, &b.data[i].p, sizeof(float)) != 0) {
            return false;
        }
    }
    return true;
}

// one sampled position on both chains: the state after the top-k, then the whole chain (fast path or the caller's
// fallback), the sampled token accepted into both. Returns whether the fast path ran (-1: a mismatch).
struct pair_state {
    llama_sampler * full = nullptr;
    llama_sampler * fast = nullptr;
    common_penfast_cfg cfg;
    std::vector<llama_token>      excl;
    std::vector<llama_token_data> cur_full;
    std::vector<llama_token_data> cur_fast;
};

static int step(pair_state & s, const std::vector<float> & x, int it, llama_token * sampled) {
    const int n_vocab = (int) x.size();
    llama_token_data_array a_full;
    llama_token_data_array a_fast;

    // the state right after the chain's top-k: the full path runs the samplers [0, i_top_k] one by one
    raw_array(x, s.cur_full, a_full);
    for (int32_t i = 0; i <= s.cfg.i_top_k; ++i) {
        llama_sampler_apply(llama_sampler_chain_get(s.full, i), &a_full);
    }
    const bool fast_k = common_sampler_penfast_apply(s.cfg, s.fast, x.data(), n_vocab, true, s.excl, s.cur_fast, a_fast);
    if (fast_k) {
        check(same_exact(a_full, a_fast, false), "candidates after the top-k differ", it);
        check(a_fast.size == (size_t) s.cfg.top_k, "top-k size", it);
    }

    // the whole chain; on a refusal the caller runs the full chain (common_sampler_sample: set_logits + chain)
    raw_array(x, s.cur_full, a_full);
    llama_sampler_apply(s.full, &a_full);
    const bool fast = common_sampler_penfast_apply(s.cfg, s.fast, x.data(), n_vocab, false, s.excl, s.cur_fast, a_fast);
    if (!fast) {
        raw_array(x, s.cur_fast, a_fast);
        llama_sampler_apply(s.fast, &a_fast);
    }
    check(fast == fast_k, "the fast path decided differently at the same state", it);
    if (!same_exact(a_full, a_fast, true)) {
        check(false, fast ? "chain result differs (fast path)" : "chain result differs (fallback)", it);
        return -1;
    }
    if (a_full.selected < 0 || a_full.selected >= (int64_t) a_full.size) {
        check(false, "no selected token", it);
        return -1;
    }
    const llama_token id_full = a_full.data[a_full.selected].id;
    const llama_token id_fast = a_fast.data[a_fast.selected].id;
    check(id_full == id_fast, "sampled token differs", it);
    llama_sampler_accept(s.full, id_full);
    llama_sampler_accept(s.fast, id_fast);
    if (sampled) {
        *sampled = id_full;
    }
    return fast ? 1 : 0;
}

static std::vector<llama_token> top_ids(const std::vector<float> & x, int n) {
    std::vector<llama_token> ids(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        ids[i] = (llama_token) i;
    }
    n = std::min(n, (int) ids.size());
    std::partial_sort(ids.begin(), ids.begin() + n, ids.end(), [&](llama_token a, llama_token b) { return x[a] > x[b]; });
    ids.resize(n);
    return ids;
}

static common_params_sampling kyro_params() {
    common_params_sampling p;
    p.top_k            = 20;
    p.top_p            = 0.95f;
    p.min_p            = 0.0f;
    p.temp             = 0.6f;
    p.penalty_last_n   = 64;
    p.penalty_repeat   = 1.0f;
    p.penalty_freq     = 0.0f;
    p.penalty_present  = 1.5f;
    p.seed             = 1234;
    return p;
}

static bool cfg_on(const common_params_sampling & p, const std::vector<llama_token> & suppress, int n_vocab) {
    llama_sampler * chain = make_chain(p, suppress, n_vocab);
    const common_penfast_cfg cfg = common_sampler_penfast_cfg(p, chain, suppress.data(), (int32_t) suppress.size(), n_vocab);
    llama_sampler_free(chain);
    return cfg.top_k > 0;
}

static void test_cfg() {
    const int n_vocab = 32000;
    const std::vector<llama_token> none;
    const std::vector<llama_token> supp = { 5, 77 };
    int it = 9000;

    check(cfg_on(kyro_params(), none, n_vocab), "cfg: Kyro's parameters", it++);
    check(cfg_on(kyro_params(), supp, n_vocab), "cfg: Kyro's parameters + suppress tokens", it++);
    {
        common_params_sampling p = kyro_params();
        p.penalty_repeat = 0.8f; p.penalty_freq = -0.5f; p.penalty_present = -1.0f; // raising penalties
        check(cfg_on(p, none, n_vocab), "cfg: raising penalties", it++);
    }
    {
        common_params_sampling p = kyro_params();
        p.logit_bias = { { 9, 4.0f }, { 10, -INFINITY } };
        check(cfg_on(p, supp, n_vocab), "cfg: user biases", it++);
        p.logit_bias.push_back({ 9, 0.0f });
        check(!cfg_on(p, supp, n_vocab), "cfg: a token listed twice", it++);
        p.logit_bias = { { 77, -1.0f } };
        check(!cfg_on(p, supp, n_vocab), "cfg: a user bias on a suppress token", it++);
    }
    {
        common_params_sampling p = kyro_params();
        p.dry_multiplier = 0.8f; p.dry_base = 1.75f; p.dry_allowed_length = 2; p.dry_penalty_last_n = 64;
        check(!cfg_on(p, none, n_vocab), "cfg: DRY before the top-k", it++);
        p.samplers = { COMMON_SAMPLER_TYPE_PENALTIES, COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_DRY,
                       COMMON_SAMPLER_TYPE_TOP_P, COMMON_SAMPLER_TYPE_TEMPERATURE };
        check(cfg_on(p, none, n_vocab), "cfg: DRY after the top-k", it++);
    }
    {
        common_params_sampling p = kyro_params();
        p.top_n_sigma = 1.0f;
        check(!cfg_on(p, none, n_vocab), "cfg: top-n-sigma before the top-k", it++);
    }
    {
        common_params_sampling p = kyro_params();
        p.mirostat = 2;
        check(!cfg_on(p, none, n_vocab), "cfg: mirostat", it++);
    }
    {
        common_params_sampling p = kyro_params();
        p.n_probs = 5;
        check(!cfg_on(p, none, n_vocab), "cfg: n_probs", it++);
    }
    {
        common_params_sampling p = kyro_params();
        p.top_k = 0;
        check(!cfg_on(p, none, n_vocab), "cfg: top_k 0", it++);
        p.top_k = 129;
        check(!cfg_on(p, none, n_vocab), "cfg: top_k 129", it++);
        p.top_k = 128;
        check(cfg_on(p, none, n_vocab), "cfg: top_k 128", it++);
    }
    {
        common_params_sampling p = kyro_params();
        p.samplers = { COMMON_SAMPLER_TYPE_PENALTIES, COMMON_SAMPLER_TYPE_TOP_P, COMMON_SAMPLER_TYPE_TOP_K,
                       COMMON_SAMPLER_TYPE_TEMPERATURE };
        check(!cfg_on(p, none, n_vocab), "cfg: top-p before the top-k", it++);
        p.samplers = { COMMON_SAMPLER_TYPE_TEMPERATURE, COMMON_SAMPLER_TYPE_TOP_K };
        check(!cfg_on(p, none, n_vocab), "cfg: temperature before the top-k", it++);
        p.samplers = { COMMON_SAMPLER_TYPE_PENALTIES, COMMON_SAMPLER_TYPE_TOP_P, COMMON_SAMPLER_TYPE_TEMPERATURE };
        check(!cfg_on(p, none, n_vocab), "cfg: no top-k", it++);
    }
}

// hand-made rows: a tie inside the top-k refuses, a raised window token makes the top-k, the window cap refuses
static void test_cases() {
    const int n_vocab = 20000;
    const int K = 20;
    int it = 9500;

    // a tie at the K-th place among unmoved tokens: refused, the fallback still equals the full path
    {
        common_params_sampling p = kyro_params();
        std::vector<float> x(n_vocab);
        std::mt19937 g(7);
        std::uniform_real_distribution<float> lo(-10.0f, -5.0f);
        for (auto & v : x) v = lo(g);
        for (int j = 0; j < K - 1; ++j) x[100 + j] = 20.0f - (float) j * 0.5f;
        x[300] = 5.0f;
        x[301] = 5.0f;
        pair_state s;
        s.full = make_chain(p, {}, n_vocab);
        s.fast = make_chain(p, {}, n_vocab);
        s.cfg  = common_sampler_penfast_cfg(p, s.fast, nullptr, 0, n_vocab);
        check(step(s, x, it, nullptr) == 0, "tie at the K-th place not refused", it);
        it++;
        llama_sampler_free(s.full);
        llama_sampler_free(s.fast);
    }
    // a penalised window token lands on an unmoved token's logit inside the top-k: refused
    {
        common_params_sampling p = kyro_params();
        std::vector<float> x(n_vocab);
        std::mt19937 g(8);
        std::uniform_real_distribution<float> lo(-10.0f, -5.0f);
        for (auto & v : x) v = lo(g);
        for (int j = 0; j < K; ++j) x[100 + j] = 20.0f - (float) j * 0.75f;
        x[500] = 10.0f; // in the window: 10.0 - 1.5 = 8.5
        x[501] = 8.5f;  // unmoved
        pair_state s;
        s.full = make_chain(p, {}, n_vocab);
        s.fast = make_chain(p, {}, n_vocab);
        llama_sampler_accept(s.full, 500);
        llama_sampler_accept(s.fast, 500);
        s.cfg = common_sampler_penfast_cfg(p, s.fast, nullptr, 0, n_vocab);
        check(step(s, x, it, nullptr) == 0, "penalised tie inside the top-k not refused", it);
        it++;
        llama_sampler_free(s.full);
        llama_sampler_free(s.fast);
    }
    // a raising repeat penalty lifts a window token from far below the top-k into it; a positive bias does the same
    {
        common_params_sampling p = kyro_params();
        p.penalty_repeat  = 0.5f; // logit > 0: divided by 0.5
        p.penalty_present = 0.0f;
        p.logit_bias = { { 700, 30.0f } };
        std::vector<float> x(n_vocab);
        std::mt19937 g(9);
        std::uniform_real_distribution<float> lo(-10.0f, -5.0f);
        for (auto & v : x) v = lo(g);
        for (int j = 0; j < K + 10; ++j) x[100 + j] = 9.0f - (float) j * 0.125f;
        x[600] = 4.75f;  // rank > K raw; raised to 9.5, the best
        x[700] = -7.0f;  // biased to 23.0
        pair_state s;
        s.full = make_chain(p, {}, n_vocab);
        s.fast = make_chain(p, {}, n_vocab);
        llama_sampler_accept(s.full, 600);
        llama_sampler_accept(s.fast, 600);
        s.cfg = common_sampler_penfast_cfg(p, s.fast, nullptr, 0, n_vocab);
        check(s.cfg.top_k == K, "raising case: cfg off", it);
        std::vector<llama_token_data> cur;
        std::vector<llama_token> excl;
        llama_token_data_array a;
        const bool ok = common_sampler_penfast_apply(s.cfg, s.fast, x.data(), n_vocab, true, excl, cur, a);
        check(ok, "raising case refused", it);
        bool has600 = false;
        bool has700 = false;
        for (size_t i = 0; ok && i < a.size; ++i) {
            has600 |= a.data[i].id == 600;
            has700 |= a.data[i].id == 700;
        }
        check(!ok || (has600 && has700 && a.data[0].id == 700 && a.data[1].id == 600), "raised tokens not at the top", it);
        check(step(s, x, it, nullptr) == 1, "raising case: fast path did not run", it);
        it++;
        llama_sampler_free(s.full);
        llama_sampler_free(s.fast);
    }
    // more moved tokens than the cap: refused, the fallback equals the full path
    {
        common_params_sampling p = kyro_params();
        p.penalty_last_n = 3000;
        std::vector<float> x(n_vocab);
        std::mt19937 g(10);
        std::normal_distribution<float> nd(0.0f, 3.0f);
        for (auto & v : x) v = nd(g);
        pair_state s;
        s.full = make_chain(p, {}, n_vocab);
        s.fast = make_chain(p, {}, n_vocab);
        for (int j = 0; j < 2000; ++j) {
            llama_sampler_accept(s.full, (llama_token) (j * 7));
            llama_sampler_accept(s.fast, (llama_token) (j * 7));
        }
        s.cfg = common_sampler_penfast_cfg(p, s.fast, nullptr, 0, n_vocab);
        check(llama_sampler_penalties_window(llama_sampler_chain_get(s.fast, s.cfg.i_pen.empty() ? 0 : s.cfg.i_pen[0]), nullptr, 0) == 2000,
                "window count", it);
        check(step(s, x, it, nullptr) == 0, "window above the cap not refused", it);
        it++;
        llama_sampler_free(s.full);
        llama_sampler_free(s.fast);
    }
}

int main() {
    test_cfg();
    test_cases();

    int n_steps = 0, n_fast = 0, n_untied = 0, n_untied_fast = 0;
    const int n_cases = 120;
    const int n_step  = 4;
    for (int it = 0; it < n_cases; ++it) {
        std::mt19937 g(1000u + (uint32_t) it);
        std::uniform_real_distribution<float> u01(0.0f, 1.0f);

        const int n_vocab = it % 4 == 0 ? 248000 : it % 4 == 1 ? 20000 : 20000 + (int) (u01(g) * 228000.0f);
        const bool tied   = it % 5 == 4; // logits on a coarse grid: many ties

        common_params_sampling p = kyro_params();
        p.seed = 5000u + (uint32_t) it;
        const int variant = it % 6;
        if (variant == 1) {        // random lowering penalties
            p.penalty_repeat = 1.0f + u01(g) * 0.5f; p.penalty_freq = u01(g); p.penalty_present = u01(g) * 2.0f;
        } else if (variant == 2) { // raising penalties
            p.penalty_repeat = 0.7f + u01(g) * 0.3f; p.penalty_freq = -u01(g) * 0.5f; p.penalty_present = -u01(g);
        } else if (variant == 3) { // other sampling parameters
            p.top_k = 1 + (int) (u01(g) * 127.0f); p.top_p = 0.5f + u01(g) * 0.5f; p.min_p = 0.05f;
            p.temp = 0.2f + u01(g) * 1.3f; p.penalty_last_n = 16 + (int) (u01(g) * 240.0f);
            p.penalty_repeat = 1.1f; p.penalty_freq = 0.2f;
        } else if (variant == 4) { // active DRY after the top-k
            p.samplers = { COMMON_SAMPLER_TYPE_PENALTIES, COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_DRY,
                           COMMON_SAMPLER_TYPE_TOP_P, COMMON_SAMPLER_TYPE_MIN_P, COMMON_SAMPLER_TYPE_TEMPERATURE };
            p.dry_multiplier = 0.8f; p.dry_base = 1.75f; p.dry_allowed_length = 2; p.dry_penalty_last_n = 64;
        }
        // variant 0, 5: Kyro's

        std::vector<float> x(n_vocab);
        std::normal_distribution<float> nd(0.0f, 3.0f);
        auto fill = [&]() {
            for (auto & v : x) {
                v = nd(g);
                if (tied) {
                    v = std::round(v * 4.0f) / 4.0f;
                }
            }
        };
        fill();
        const std::vector<llama_token> best = top_ids(x, 64);

        std::vector<llama_token> suppress;
        if (it % 3 == 0) {
            suppress = { best[(size_t) (it % 7)], (llama_token) (u01(g) * (n_vocab - 1)) };
            if (suppress[0] == suppress[1]) {
                suppress.pop_back();
            }
        }
        if (it % 4 == 2) { // user biases: up, down, -inf, on a best token and on random ones
            std::set<llama_token> used(suppress.begin(), suppress.end());
            const llama_token cand[4] = { best[3], (llama_token) (u01(g) * (n_vocab - 1)),
                                          (llama_token) (u01(g) * (n_vocab - 1)), best[10] };
            const float bias[4] = { -2.5f, 12.0f, -INFINITY, 1.25f };
            for (int j = 0; j < 4; ++j) {
                if (used.insert(cand[j]).second) {
                    p.logit_bias.push_back({ cand[j], bias[j] });
                }
            }
        }

        pair_state s;
        s.full = make_chain(p, suppress, n_vocab);
        s.fast = make_chain(p, suppress, n_vocab);
        s.cfg  = common_sampler_penfast_cfg(p, s.fast, suppress.data(), (int32_t) suppress.size(), n_vocab);
        check(s.cfg.top_k == p.top_k, "cfg off for a supported configuration", it);
        if (s.cfg.top_k <= 0) {
            llama_sampler_free(s.full);
            llama_sampler_free(s.fast);
            continue;
        }

        // the accepted history: mostly the best raw tokens, with repeats
        std::vector<llama_token> history;
        const int n_hist = (int) (u01(g) * 300.0f);
        for (int j = 0; j < n_hist; ++j) {
            const llama_token t = u01(g) < 0.6f ? best[(size_t) (u01(g) * 40.0f)] : (llama_token) (u01(g) * (n_vocab - 1));
            history.push_back(t);
            llama_sampler_accept(s.full, t);
            llama_sampler_accept(s.fast, t);
        }

        for (int k = 0; k < n_step; ++k) {
            if (k > 0) {
                fill();
            }
            // the window the fast path reads is the last penalty_last_n accepted tokens
            if (!s.cfg.i_pen.empty()) {
                const llama_sampler * pen = llama_sampler_chain_get(s.fast, s.cfg.i_pen[0]);
                const int32_t nw = llama_sampler_penalties_window(pen, nullptr, 0);
                std::vector<llama_token> w((size_t) std::max(nw, 0));
                llama_sampler_penalties_window(pen, w.data(), nw);
                std::sort(w.begin(), w.end());
                const size_t n_last = std::min(history.size(), (size_t) p.penalty_last_n);
                std::vector<llama_token> want(history.end() - (std::ptrdiff_t) n_last, history.end());
                std::sort(want.begin(), want.end());
                want.erase(std::unique(want.begin(), want.end()), want.end());
                check(w == want, "penalty window differs from the history", it);
            }
            llama_token id = LLAMA_TOKEN_NULL;
            const int r = step(s, x, it, &id);
            if (r < 0) {
                break;
            }
            history.push_back(id);
            n_steps++;
            n_fast += r;
            if (!tied) {
                n_untied++;
                n_untied_fast += r;
            }
        }
        llama_sampler_free(s.full);
        llama_sampler_free(s.fast);
    }

    printf("test-r28-penfast: %d steps, fast path %d (untied %d of %d)\n", n_steps, n_fast, n_untied_fast, n_untied);
    check(n_untied > 0 && n_untied_fast * 100 >= n_untied * 95, "the fast path ran in fewer than 95% of the untied steps", -1);
    check(n_fast < n_steps, "the tied cases never refused", -1);

    if (n_fail > 0) {
        printf("test-r28-penfast: %d FAILED\n", n_fail);
        return 1;
    }
    printf("test-r28-penfast: OK\n");
    return 0;
}
