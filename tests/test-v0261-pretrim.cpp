// [TAG_V0261_PRETRIM] the exact grammar pre-trim rule (common/sampling.cpp) against the full pass. Random logits over a
// few thousand tokens, presence / frequency / repeat penalties and DRY on a token history, negative biases and suppress
// tokens, and a random grammar mask. The real samplers run the chain as common_sampler_init builds it up to its top-k:
// whenever common_sampler_pretrim_exact accepts a trim, the top-k set (ids and logits) equals the full pass; when it
// refuses, the count of untouched valid tokens is short. Also: the touched set covers every token the samplers moved,
// the v0.25.5 gate gets cases wrong that the new gate refuses, the second wider trim, and the gates of the three modes.
// A token listed twice in the logit bias, a zero bias first, gets a different logit in the trimmed pass: refused.
// CPU only, no model.

#include "sampling.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

static int n_fail = 0;

static void check(bool ok, const char * what, int it) {
    if (!ok) {
        printf("FAIL: %s (case %d)\n", what, it);
        n_fail++;
    }
}

// the chain common_sampler_init builds, up to its first top-k: logit bias (user biases + suppress tokens), then the
// penalties and DRY in the order of params.samplers; with_top_k false: without the top-k (the logits it sees)
static llama_sampler * make_chain(const common_params_sampling & p, const std::vector<llama_token> & suppress,
        int n_vocab, bool with_top_k) {
    llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    std::vector<llama_logit_bias> merged = p.logit_bias;
    for (const llama_token t : suppress) {
        merged.push_back({ t, -INFINITY });
    }
    if (!merged.empty()) {
        llama_sampler_chain_add(chain, llama_sampler_init_logit_bias(n_vocab, (int32_t) merged.size(), merged.data()));
    }
    for (const auto t : p.samplers) {
        if (t == COMMON_SAMPLER_TYPE_PENALTIES) {
            llama_sampler_chain_add(chain, llama_sampler_init_penalties(n_vocab, p.penalty_last_n, p.penalty_repeat,
                    p.penalty_freq, p.penalty_present));
        } else if (t == COMMON_SAMPLER_TYPE_DRY) {
            llama_sampler_chain_add(chain, llama_sampler_init_dry(nullptr, p.dry_multiplier, p.dry_base,
                    p.dry_allowed_length, p.dry_penalty_last_n, nullptr, 0));
        } else if (t == COMMON_SAMPLER_TYPE_TOP_K) {
            if (with_top_k) {
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(p.top_k));
            }
            break;
        }
    }
    return chain;
}

static std::vector<llama_token_data> raw_array(const std::vector<float> & x) {
    std::vector<llama_token_data> cur(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        cur[i] = llama_token_data{ (llama_token) i, x[i], 0.0f };
    }
    return cur;
}

// the grammar's effect on the candidates: invalid ones get -inf
static void apply_mask(llama_token_data_array * a, const std::vector<char> & valid) {
    for (size_t i = 0; i < a->size; ++i) {
        if (!valid[a->data[i].id]) {
            a->data[i].logit = -INFINITY;
        }
    }
    a->sorted = false;
}

static std::vector<llama_token_data> result_of(const llama_token_data_array & a) {
    return std::vector<llama_token_data>(a.data, a.data + a.size);
}

// the same set: compared in one order (logit, then id), so equal logits inside the set do not count as a difference
static bool same_result(std::vector<llama_token_data> a, std::vector<llama_token_data> b) {
    if (a.size() != b.size()) {
        return false;
    }
    const auto by_logit = [](const llama_token_data & u, const llama_token_data & v) {
        return u.logit > v.logit || (u.logit == v.logit && u.id < v.id);
    };
    std::sort(a.begin(), a.end(), by_logit);
    std::sort(b.begin(), b.end(), by_logit);
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].id != b[i].id || a[i].logit != b[i].logit) {
            return false;
        }
    }
    return true;
}

// the trimmed path of common_sampler_pretrim_grammar for one width: trim, grammar, sorted cleared
static std::vector<llama_token_data> trimmed(const std::vector<float> & x, size_t width, const std::vector<char> & valid) {
    std::vector<llama_token_data> cur = raw_array(x);
    llama_token_data_array a = { cur.data(), cur.size(), -1, false };
    llama_sampler * trim = llama_sampler_init_top_k((int32_t) width);
    llama_sampler_apply(trim, &a);
    llama_sampler_free(trim);
    apply_mask(&a, valid);
    cur.resize(a.size);
    return cur;
}

static size_t count_ok(const std::vector<llama_token_data> & cur, const std::vector<llama_token> & touched) {
    size_t n = 0;
    for (const auto & d : cur) {
        if (d.logit != -INFINITY && !std::binary_search(touched.begin(), touched.end(), d.id)) {
            n++;
        }
    }
    return n;
}

static size_t count_valid(const std::vector<llama_token_data> & cur) {
    size_t n = 0;
    for (const auto & d : cur) {
        n += d.logit != -INFINITY ? 1 : 0;
    }
    return n;
}

// what common_sampler_pretrim_prepare passes: prev (capacity cap) after the history, the last last_n of it
static void prev_window(const std::vector<llama_token> & history, size_t cap, int32_t last_n,
        std::vector<llama_token> & hist, bool & hist_all) {
    const size_t prev_size = std::min(history.size(), cap);
    const size_t n = std::min((size_t) std::max(last_n, 0), prev_size);
    hist.assign(history.end() - (std::ptrdiff_t) n, history.end());
    hist_all = n == prev_size && prev_size < cap;
}

struct stats {
    int n_cases       = 0;
    int n_exact1      = 0;
    int n_exact2      = 0;
    int n_refused     = 0;
    int n_old_wrong   = 0; // v0.25.5 gate passed on the first trim, the result differed from the full pass
    int n_bound_cases = 0;
    int n_skip2       = 0; // [TAG_V0261_PRETRIM] first trim kept fewer than top_k valid tokens: no second trim
    int n_skip2_lost  = 0; // of those, the second trim would have been exact (speed only, the full pass is exact too)
};

// one case: the full pass against both trims; returns false when the parameters disable the trim
static bool run_case(const common_params_sampling & p, const std::vector<float> & x, const std::vector<char> & valid,
        const std::vector<llama_token> & suppress, const std::vector<llama_token> & history, size_t cap, int it,
        stats & st) {
    const int n_vocab = (int) x.size();
    const common_pretrim_cfg cfg = common_sampler_pretrim_cfg(p, suppress.data(), (int32_t) suppress.size(), n_vocab, 1);
    if (cfg.top_k <= 0) {
        return false;
    }
    st.n_cases++;

    llama_sampler * chain = make_chain(p, suppress, n_vocab, true);
    llama_sampler * pre   = make_chain(p, suppress, n_vocab, false);
    for (const llama_token t : history) {
        llama_sampler_accept(chain, t);
        llama_sampler_accept(pre, t);
    }

    std::vector<llama_token> hist;
    bool hist_all = false;
    prev_window(history, cap, cfg.last_n, hist, hist_all);
    std::vector<llama_token> touched;
    const size_t n_bound = common_sampler_pretrim_touched(cfg, hist, hist_all, touched);
    st.n_bound_cases += n_bound > 0 ? 1 : 0;
    check(std::is_sorted(touched.begin(), touched.end()) &&
          std::adjacent_find(touched.begin(), touched.end()) == touched.end(), "touched not sorted and unique", it);

    // every token the samplers before the top-k moved is in touched, or within the bound
    {
        std::vector<llama_token_data> cur = raw_array(x);
        llama_token_data_array a = { cur.data(), cur.size(), -1, false };
        llama_sampler_apply(pre, &a);
        size_t n_out = 0;
        for (size_t i = 0; i < cur.size(); ++i) {
            const bool moved = !(cur[i].logit == x[cur[i].id]);
            check(!moved || cur[i].logit < x[cur[i].id], "a sampler before the top-k raised a logit", it);
            if (moved && !std::binary_search(touched.begin(), touched.end(), cur[i].id)) {
                n_out++;
            }
        }
        check(n_out <= n_bound, "a moved token is outside touched and the bound", it);
    }

    // the full pass
    std::vector<llama_token_data> full;
    {
        std::vector<llama_token_data> cur = raw_array(x);
        llama_token_data_array a = { cur.data(), cur.size(), -1, false };
        apply_mask(&a, valid);
        llama_sampler_apply(chain, &a);
        full = result_of(a);
    }

    const size_t need = (size_t) cfg.top_k + cfg.n_extra + n_bound;
    bool done = false;
    for (int pass = 0; pass < 2 && !done; ++pass) {
        const size_t width = pass == 0 ? cfg.width : cfg.width_max;
        if (width == 0) {
            continue;
        }
        std::vector<llama_token_data> cur = trimmed(x, width, valid);
        llama_token_data_array a = { cur.data(), cur.size(), -1, false };
        size_t n_valid = SIZE_MAX;
        const bool exact = common_sampler_pretrim_exact(cfg, &a, touched, n_bound, &n_valid);
        const size_t n_ok = count_ok(cur, touched);
        check(exact == (n_ok >= need), "exact does not match the count of untouched valid tokens", it);
        // the scan stops at the need: the valid count is complete only on a refusal
        check(exact ? n_valid <= count_valid(cur) : n_valid == count_valid(cur), "n_valid of the exact check", it);
        // the coarse form of the rule (n_valid >= top_k + every token that may be touched) always passes the gate
        if (count_valid(cur) >= need + touched.size()) {
            check(exact, "refused although n_valid >= top_k + n_touch", it);
        }
        const bool old_gate = pass == 0 && count_valid(cur) >= (size_t) cfg.top_k;
        if (exact || old_gate) {
            llama_sampler_apply(chain, &a);
            const bool same = same_result(result_of(a), full);
            if (exact) {
                check(same, pass == 0 ? "first trim accepted, top-k set differs from the full pass" :
                                        "second trim accepted, top-k set differs from the full pass", it);
                (pass == 0 ? st.n_exact1 : st.n_exact2)++;
                done = true;
            }
            if (old_gate && !same) {
                st.n_old_wrong++;
                check(!exact, "new gate accepted a wrong result", it);
            }
        }
        // [TAG_V0261_PRETRIM] common_sampler_pretrim_grammar: the second trim runs only when the first kept at least
        // top_k valid tokens (LLAMA_V0261_PRETRIM_NARROW unset)
        if (pass == 0 && !exact && n_valid < (size_t) cfg.top_k + cfg.n_extra && cfg.width_max > 0) {
            st.n_skip2++;
            std::vector<llama_token_data> cur2 = trimmed(x, cfg.width_max, valid);
            llama_token_data_array a2 = { cur2.data(), cur2.size(), -1, false };
            st.n_skip2_lost += common_sampler_pretrim_exact(cfg, &a2, touched, n_bound) ? 1 : 0;
            break;
        }
    }
    st.n_refused += done ? 0 : 1;

    llama_sampler_free(chain);
    llama_sampler_free(pre);
    return true;
}

static common_params_sampling base_params() {
    common_params_sampling p;
    p.top_k           = 20;
    p.penalty_last_n  = 64;
    p.penalty_repeat  = 1.0f;
    p.penalty_freq    = 0.0f;
    p.penalty_present = 1.5f;
    p.dry_multiplier  = 0.0f;
    p.top_n_sigma     = -1.0f;
    p.mirostat        = 0;
    return p;
}

static void test_gates() {
    const int n_vocab = 248320;

    // Kyro's Qwen request: presence 1.5 over the default 64-token window, top_k 20
    {
        const auto p = base_params();
        auto c = common_sampler_pretrim_cfg(p, nullptr, 0, n_vocab, 1);
        check(c.top_k == 20 && c.width == 1024u && c.last_n == 64 && c.n_extra == 0u && c.fixed.empty(), "mode 1 Kyro cfg", 0);
        check(c.width_max == 32u * (20 + 64), "mode 1 Kyro second width", 0);
        c = common_sampler_pretrim_cfg(p, nullptr, 0, n_vocab, 0);
        check(c.top_k == 0, "mode 0 keeps the PR gate (no trim with a penalty)", 0);
        c = common_sampler_pretrim_cfg(p, nullptr, 0, n_vocab, 2);
        check(c.top_k == 20 && c.width == 1024u && c.width_max == 0u && c.n_extra == 0u, "mode 2 keeps the v0.25.5 gate", 0);
    }
    // no penalty: mode 0 trims with the suppress tokens added to the need
    {
        auto p = base_params();
        p.penalty_present = 0.0f;
        const llama_token sup[2] = { 7, 3 };
        auto c = common_sampler_pretrim_cfg(p, sup, 2, n_vocab, 0);
        check(c.top_k == 20 && c.n_extra == 2u && c.width_max == 0u, "mode 0 without penalties", 1);
        c = common_sampler_pretrim_cfg(p, sup, 2, n_vocab, 1);
        check(c.top_k == 20 && c.last_n == 0 && c.fixed == std::vector<llama_token>({ 3, 7 }) && c.width_max == 0u,
                "mode 1 without penalties: suppress tokens fixed, 32 x 22 < 1024", 1);
    }
    // what disables mode 1
    {
        struct { const char * what; void (*edit)(common_params_sampling &); } cases[] = {
            { "positive bias",        [](common_params_sampling & p) { p.logit_bias.push_back({ 5, 2.0f }); } },
            { "NaN bias",             [](common_params_sampling & p) { p.logit_bias.push_back({ 5, NAN }); } },
            { "duplicate bias",       [](common_params_sampling & p) { p.logit_bias.push_back({ 5, -1.0f }); p.logit_bias.push_back({ 5, -2.0f }); } },
            // [TAG_V0261_PRETRIM] a zero bias still counts as a listing (test_zero_bias_duplicate)
            { "zero bias, then -inf", [](common_params_sampling & p) { p.logit_bias.push_back({ 5, 0.0f }); p.logit_bias.push_back({ 5, -INFINITY }); } },
            { "zero bias, then -2",   [](common_params_sampling & p) { p.logit_bias.push_back({ 5, 0.0f }); p.logit_bias.push_back({ 5, -2.0f }); } },
            { "-2, then zero bias",   [](common_params_sampling & p) { p.logit_bias.push_back({ 5, -2.0f }); p.logit_bias.push_back({ 5, 0.0f }); } },
            { "zero bias twice",      [](common_params_sampling & p) { p.logit_bias.push_back({ 5, 0.0f }); p.logit_bias.push_back({ 5, -0.0f }); } },
            { "repeat penalty < 1",   [](common_params_sampling & p) { p.penalty_repeat = 0.9f; } },
            { "negative frequency",   [](common_params_sampling & p) { p.penalty_freq = -0.1f; } },
            { "negative presence",    [](common_params_sampling & p) { p.penalty_present = -1.0f; } },
            { "negative DRY",         [](common_params_sampling & p) { p.dry_multiplier = -0.5f; } },
            { "top-n-sigma",          [](common_params_sampling & p) { p.top_n_sigma = 1.0f; } },
            { "mirostat",             [](common_params_sampling & p) { p.mirostat = 2; } },
            { "no top-k",             [](common_params_sampling & p) { p.top_k = 0; } },
            { "temperature first",    [](common_params_sampling & p) { p.samplers = { COMMON_SAMPLER_TYPE_TEMPERATURE, COMMON_SAMPLER_TYPE_TOP_K }; } },
            { "top-k not in chain",   [](common_params_sampling & p) { p.samplers = { COMMON_SAMPLER_TYPE_PENALTIES, COMMON_SAMPLER_TYPE_TOP_P }; } },
        };
        int i = 10;
        for (const auto & c : cases) {
            auto p = base_params();
            c.edit(p);
            const auto cfg = common_sampler_pretrim_cfg(p, nullptr, 0, n_vocab, 1);
            if (cfg.top_k != 0) {
                printf("FAIL: mode 1 trims with %s\n", c.what);
                n_fail++;
            }
            i++;
        }
        // a user bias on a suppress token is a duplicate too
        auto p = base_params();
        p.logit_bias.push_back({ 9, -3.0f });
        const llama_token sup[1] = { 9 };
        check(common_sampler_pretrim_cfg(p, sup, 1, n_vocab, 1).top_k == 0, "bias on a suppress token", i);
        // a zero user bias on a suppress token: the merged list puts the suppress -inf after it
        p = base_params();
        p.logit_bias.push_back({ 9, 0.0f });
        check(common_sampler_pretrim_cfg(p, sup, 1, n_vocab, 1).top_k == 0, "zero bias on a suppress token", i);
        // a trim as wide as the vocabulary
        check(common_sampler_pretrim_cfg(base_params(), nullptr, 0, 1000, 1).top_k == 0, "width >= n_vocab", i + 1);
    }
    // what keeps mode 1 on
    {
        auto p = base_params();
        p.logit_bias.push_back({ 5, -INFINITY });
        p.logit_bias.push_back({ 6, -2.0f });
        p.logit_bias.push_back({ 8, 0.0f });
        p.penalty_repeat = 1.1f;
        p.penalty_freq   = 0.3f;
        p.dry_multiplier = 0.8f;
        p.dry_penalty_last_n = 128;
        p.top_n_sigma = 1.0f;
        p.samplers = { COMMON_SAMPLER_TYPE_PENALTIES, COMMON_SAMPLER_TYPE_DRY, COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TOP_N_SIGMA };
        const auto c = common_sampler_pretrim_cfg(p, nullptr, 0, n_vocab, 1);
        check(c.top_k == 20 && c.last_n == 128 && c.fixed == std::vector<llama_token>({ 5, 6 }), "negative biases, DRY, top-n-sigma after top-k", 30);
        check(c.width_max == 32u * (20 + 2 + 128), "second width with biases and the DRY window", 30);
    }
    // a negative window is 0 in libllama: the penalties do nothing
    {
        auto p = base_params();
        p.penalty_last_n = -1;
        const auto c = common_sampler_pretrim_cfg(p, nullptr, 0, n_vocab, 1);
        check(c.top_k == 20 && c.last_n == 0, "penalty_last_n -1", 31);
    }
    // the history bound
    {
        const auto c = common_sampler_pretrim_cfg(base_params(), nullptr, 0, n_vocab, 1);
        const std::vector<llama_token> hist = { 4, 4, 9, 1 };
        std::vector<llama_token> touched;
        check(common_sampler_pretrim_touched(c, hist, false, touched) == 60u && touched == std::vector<llama_token>({ 1, 4, 9 }),
                "window longer than the known history: bound", 32);
        check(common_sampler_pretrim_touched(c, hist, true, touched) == 0u, "whole history known: no bound", 33);
    }
}

// fixed cases: the penalties push the valid top of the first trim below tokens past it
static void test_adversarial() {
    for (const int n_vocab : { 3000, 40000 }) {
        auto p = base_params();
        p.top_k = 5;
        p.penalty_present = 50.0f;
        std::vector<float> x(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            x[i] = -0.001f * (float) i;
        }
        // valid: the 10 best and everything from rank 1500 on; the 10 best were all used
        std::vector<char> valid(n_vocab, 0);
        std::vector<llama_token> history;
        for (int i = 0; i < 10; ++i) {
            valid[i] = 1;
            history.push_back(i);
        }
        for (int i = 1500; i < n_vocab; ++i) {
            valid[i] = 1;
        }
        stats st;
        const bool on = run_case(p, x, valid, {}, history, 64, 100 + n_vocab, st);
        check(on, "adversarial case disabled", n_vocab);
        check(st.n_old_wrong == 1, "the v0.25.5 gate accepts the first trim and gets it wrong", n_vocab);
        check(st.n_exact1 == 0, "the new gate refuses the first trim", n_vocab);
        if (n_vocab == 40000) {
            check(st.n_exact2 == 1, "the second trim (2208) is exact", n_vocab);
        } else {
            check(st.n_refused == 1, "no second trim at 3000 tokens: full pass", n_vocab);
        }
        check(st.n_skip2 == 0, "10 valid tokens in the first trim: the second trim runs", n_vocab);
    }
    // [TAG_V0261_PRETRIM] a narrow position: the first trim keeps 3 valid tokens, fewer than top_k 5. The second trim
    // would be exact here, but such positions (names, closing tags, enum values) almost never fill it: full pass
    {
        const int n_vocab = 40000;
        auto p = base_params();
        p.top_k = 5;
        std::vector<float> x(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            x[i] = -0.001f * (float) i;
        }
        std::vector<char> valid(n_vocab, 0);
        for (int i = 0; i < 3; ++i) {
            valid[i] = 1;
        }
        for (int i = 1500; i < n_vocab; ++i) {
            valid[i] = 1;
        }
        stats st;
        check(run_case(p, x, valid, {}, {}, 64, 150, st), "narrow case disabled", 150);
        check(st.n_skip2 == 1 && st.n_skip2_lost == 1 && st.n_refused == 1 && st.n_exact1 == 0 && st.n_exact2 == 0,
                "narrow first trim: no second trim, full pass", 150);
    }
}

// [TAG_V0261_PRETRIM] the case the zero-bias listing check refuses: {t, 0} then {t, -inf} (a user zero bias before the
// ignore_eos -inf). The full array is in id order, so libllama's logit bias applies both entries to t; the trimmed array
// is not, so both go to its search loop, which applies the first match only and t keeps its raw logit there
static void test_zero_bias_duplicate() {
    const int n_vocab = 3000;
    const llama_token t = 2000; // past the trim width, so not in place in the trimmed array either
    auto p = base_params();
    p.top_k = 5;
    std::vector<float> x(n_vocab);
    for (int i = 0; i < n_vocab; ++i) {
        x[i] = -0.001f * (float) i;
    }
    x[t] = 1.0f; // the best raw token
    const std::vector<char> valid(n_vocab, 1);

    const auto full_of = [&](const common_params_sampling & q) {
        llama_sampler * chain = make_chain(q, {}, n_vocab, true);
        std::vector<llama_token_data> cur = raw_array(x);
        llama_token_data_array a = { cur.data(), cur.size(), -1, false };
        apply_mask(&a, valid);
        llama_sampler_apply(chain, &a);
        llama_sampler_free(chain);
        return result_of(a);
    };
    const auto trimmed_of = [&](const common_params_sampling & q, size_t width) {
        llama_sampler * chain = make_chain(q, {}, n_vocab, true);
        std::vector<llama_token_data> cur = trimmed(x, width, valid);
        llama_token_data_array a = { cur.data(), cur.size(), -1, false };
        llama_sampler_apply(chain, &a);
        llama_sampler_free(chain);
        return result_of(a);
    };
    const auto has = [](const std::vector<llama_token_data> & r, llama_token id) {
        for (const auto & d : r) {
            if (d.id == id && d.logit != -INFINITY) {
                return true;
            }
        }
        return false;
    };

    // the -inf alone: trimmed, exact and the same as the full pass
    {
        auto q = p;
        q.logit_bias = { { t, -INFINITY } };
        const auto cfg = common_sampler_pretrim_cfg(q, nullptr, 0, n_vocab, 1);
        check(cfg.top_k == 5 && cfg.fixed == std::vector<llama_token>({ t }), "lone -inf bias trims", 200);
        const auto tr = trimmed_of(q, cfg.width);
        check(!has(full_of(q), t) && same_result(tr, full_of(q)), "lone -inf bias: trimmed equals full", 200);
    }
    // zero first: the trimmed pass would keep t (the ban lost), so the cfg must refuse
    {
        auto q = p;
        q.logit_bias = { { t, 0.0f }, { t, -INFINITY } };
        const auto tr = trimmed_of(q, 1024);
        check(!has(full_of(q), t), "zero then -inf: the full pass bans t", 201);
        check(has(tr, t), "zero then -inf: the trimmed pass keeps t (libllama search path)", 201);
        check(common_sampler_pretrim_cfg(q, nullptr, 0, n_vocab, 1).top_k == 0, "zero then -inf: refused", 201);
    }
    // the same through a suppress token: the user's zero bias comes before the appended suppress -inf
    {
        auto q = p;
        q.logit_bias = { { t, 0.0f } };
        const llama_token sup[1] = { t };
        check(common_sampler_pretrim_cfg(q, sup, 1, n_vocab, 1).top_k == 0, "zero bias on a suppress token: refused", 202);
    }
    // a lone zero bias changes nothing: trimmed and exact
    {
        auto q = p;
        q.logit_bias = { { t, 0.0f } };
        const auto cfg = common_sampler_pretrim_cfg(q, nullptr, 0, n_vocab, 1);
        check(cfg.top_k == 5 && cfg.fixed.empty(), "lone zero bias trims, touches nothing", 203);
        check(has(full_of(q), t) && same_result(trimmed_of(q, cfg.width), full_of(q)), "lone zero bias: trimmed equals full", 203);
    }
}

int main() {
    test_gates();
    test_adversarial();
    test_zero_bias_duplicate();

    std::mt19937 rng(20261026);
    const int   sizes[]    = { 3000, 4096, 12000, 20000 };
    const int   ks[]       = { 1, 5, 20, 40 };
    const float masks[]    = { 0.005f, 0.02f, 0.08f, 0.4f, 1.0f };
    const int   pen_ns[]   = { 0, 16, 64, 128 };
    const float presents[] = { 0.0f, 0.5f, 1.5f, 6.0f };
    const float freqs[]    = { 0.0f, 0.3f };
    const float repeats[]  = { 1.0f, 1.1f, 1.5f };
    const float drys[]     = { 0.0f, 0.8f, 3.0f };
    const int   dry_ns[]   = { 32, 64, 256 };
    const size_t caps[]    = { 32, 64 };

    stats st;
    int n_disabled = 0;
    for (int it = 0; it < 2500; ++it) {
        const int n = sizes[rng() % 4];
        common_params_sampling p = base_params();
        p.top_k              = ks[rng() % 4];
        p.penalty_last_n     = pen_ns[rng() % 4];
        p.penalty_present    = presents[rng() % 4];
        p.penalty_freq       = freqs[rng() % 2];
        p.penalty_repeat     = repeats[rng() % 3];
        p.dry_multiplier     = drys[rng() % 3];
        p.dry_base           = 1.75f;
        p.dry_allowed_length = 2;
        p.dry_penalty_last_n = dry_ns[rng() % 3];

        std::normal_distribution<float> nd(0.0f, 3.0f);
        std::vector<float> x(n);
        for (auto & v : x) {
            v = nd(rng);
        }
        std::vector<int> order(n);
        for (int i = 0; i < n; ++i) {
            order[i] = i;
        }
        std::partial_sort(order.begin(), order.begin() + 300, order.end(), [&](int a, int b) { return x[a] > x[b]; });

        // negative and zero biases and suppress tokens among the best raw tokens (now and then the same token twice)
        std::vector<llama_token> suppress;
        const int n_bias = (int) (rng() % 6);
        for (int i = 0; i < n_bias; ++i) {
            const unsigned r = rng() % 6;
            const float b = r < 2 ? -INFINITY : r == 2 ? 0.0f : -(float) (1 + rng() % 10);
            p.logit_bias.push_back({ order[rng() % 100], b });
        }
        const int n_sup = (int) (rng() % 4);
        for (int i = 0; i < n_sup; ++i) {
            suppress.push_back(order[rng() % 50]);
        }

        // a history drawn mostly from the best raw tokens, with copied runs for DRY
        std::vector<llama_token> history;
        const int lens[] = { 0, 5, 40, 64, 100, 300 };
        const int len = lens[rng() % 6];
        while ((int) history.size() < len) {
            if (history.size() > 8 && rng() % 4 == 0) {
                const size_t run = 3 + rng() % 6;
                const size_t from = rng() % (history.size() - run);
                for (size_t j = 0; j < run && (int) history.size() < len; ++j) {
                    const llama_token t = history[from + j];
                    history.push_back(t);
                }
            } else {
                history.push_back(rng() % 10 < 7 ? order[rng() % 200] : (llama_token) (rng() % n));
            }
        }

        // the grammar: valid with probability q, the best raw tokens invalid more often
        const float q = masks[rng() % 5];
        std::vector<char> valid(n);
        std::uniform_real_distribution<float> ud(0.0f, 1.0f);
        for (int i = 0; i < n; ++i) {
            valid[i] = ud(rng) < q;
        }
        for (int i = 0; i < 40; ++i) {
            if (rng() % 2) {
                valid[order[i]] = 0;
            }
        }

        if (!run_case(p, x, valid, suppress, history, caps[rng() % 2], it, st)) {
            n_disabled++;
        }
    }

    printf("%d cases (%d disabled): first trim exact %d, second trim exact %d, full pass %d, history bound in %d; "
           "v0.25.5 gate wrong in %d; second trim skipped (narrow) %d, of which it would have been exact %d\n",
           st.n_cases, n_disabled, st.n_exact1, st.n_exact2, st.n_refused, st.n_bound_cases, st.n_old_wrong,
           st.n_skip2, st.n_skip2_lost);
    check(st.n_exact1 > 0 && st.n_refused > 0, "both outcomes taken", -1);
    printf("%s\n", n_fail == 0 ? "PASS" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}
