// [TAG_R29_GRAMFAST] the candidate-set path with a grammar (common_sampler_gramfast_apply / common_sampler_gramfast_sample,
// common/sampling.cpp) against the full path, with the real libllama samplers in the chain common_sampler_init builds.
// Part 1, no model: a stand-in grammar (a validity mask, as test-v0261-pretrim stands in for the grammar) over 20000..248000
// tokens, masks from dense to a handful of valid tokens, Kyro's sampling (presence 1.5, top_k 20, top_p 0.95, min_p 0,
// temp 0.6) and other penalties, biases and suppress tokens, rows that put the valid tokens high or below many invalid ones.
// Part 2, argv[1] a vocab-only GGUF (ctest passes models/ggml-vocab-qwen35.gguf, the 248K vocabulary of Qwen3.5..3.8): a real
// GBNF tool-call grammar (a JSON object with name and arguments), driven by accepted prefixes into string, number,
// structural and array positions, and a lazy grammar that waits for its trigger. Whenever the fast path decides, the
// candidates right after the top-k, the whole chain result (ids, logits, probabilities, selected) and the sampled token over
// several steps with the same seeds equal the full path (the grammar over the whole vocabulary); when it refuses, the
// caller's fallback gives the same. Both flows of common_sampler_sample run (grammar first; sample, check, resample). Also:
// the second width, ties at theta, NaN / +inf / masked rows refused, and a narrow refusal (fewer valid tokens than top_k in
// the widest set) is one where the v0.26.1 pre-trim's first trim keeps too few as well (common_sampler_gramfast_narrow).
// LLAMA_R29_GRAMFAST=0, the reasoning budget and backend-sampled rows are checked by common_sampler_sample (they need a model).

#include "sampling.h"
#include "../src/llama-ext.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <string>
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

static llama_token sel_id(const llama_token_data_array & a) {
    return a.selected >= 0 && a.selected < (int64_t) a.size ? a.data[a.selected].id : LLAMA_TOKEN_NULL;
}

// the stand-in grammar: a token whose valid entry is 0 gets -inf, as llama_grammar_apply_impl does with a rejected one
struct mask_grammar {
    const std::vector<char> * valid = nullptr;
};

static const char * mask_grammar_name(const llama_sampler * /*smpl*/) {
    return "mask-grammar";
}

static void mask_grammar_apply(llama_sampler * smpl, llama_token_data_array * cur_p) {
    const auto * m = (const mask_grammar *) smpl->ctx;
    for (size_t i = 0; i < cur_p->size; ++i) {
        const llama_token t = cur_p->data[i].id;
        if (t < 0 || (size_t) t >= m->valid->size() || !(*m->valid)[(size_t) t]) {
            cur_p->data[i].logit = -INFINITY;
        }
    }
    cur_p->sorted = false;
}

static llama_sampler_i make_mask_iface() {
    llama_sampler_i iface = {};
    iface.name  = mask_grammar_name;
    iface.apply = mask_grammar_apply;
    return iface;
}

static llama_sampler_i g_mask_iface = make_mask_iface();

// the full path of common_sampler_sample over a plain row: the grammar over the whole vocabulary (v0.26.4 without the
// pre-trim, which only shortens it). a ends as the chain's result.
static llama_token full_sample(llama_sampler * chain, llama_sampler * grmr, bool grammar_first, bool use_grammar,
        const std::vector<float> & x, std::vector<llama_token_data> & cur, llama_token_data_array & a) {
    const bool constrain = use_grammar && grmr != nullptr;
    raw_array(x, cur, a);
    if (grammar_first && constrain) {
        llama_sampler_apply(grmr, &a);
    }
    llama_sampler_apply(chain, &a);
    const llama_token id = sel_id(a);
    if (grammar_first || !constrain || id == LLAMA_TOKEN_NULL) {
        return id;
    }
    llama_token_data       single     = { id, 1.0f, 0.0f };
    llama_token_data_array single_arr = { &single, 1, -1, false };
    llama_sampler_apply(grmr, &single_arr);
    if (single.logit != -INFINITY) {
        return id;
    }
    raw_array(x, cur, a);
    llama_sampler_apply(grmr, &a);
    llama_sampler_apply(chain, &a);
    return sel_id(a);
}

// the fast path as common_sampler_sample takes it, the caller's fallbacks included
static llama_token fast_sample(const common_penfast_cfg & cfg, llama_sampler * chain, llama_sampler * grmr,
        bool grammar_first, bool use_grammar, const std::vector<float> & x, common_gramfast_scratch & sc,
        std::vector<llama_token_data> & cur, llama_token_data_array & a, common_gramfast_result & r) {
    llama_token id = LLAMA_TOKEN_NULL;
    r = common_sampler_gramfast_sample(cfg, chain, grmr, grammar_first, use_grammar, x.data(), (int) x.size(), sc, cur, a, &id);
    if (r == COMMON_GRAMFAST_NONE) {
        return full_sample(chain, grmr, grammar_first, use_grammar, x, cur, a);
    }
    if (r == COMMON_GRAMFAST_RESAMPLE) {
        raw_array(x, cur, a);
        llama_sampler_apply(grmr, &a);
        llama_sampler_apply(chain, &a);
        return sel_id(a);
    }
    return id;
}

// the v0.26.1 pre-trim's first trim over the raw row, the grammar applied: its finite candidates
static size_t pretrim_first_valid(const std::vector<float> & x, size_t width, llama_sampler * grmr) {
    std::vector<llama_token_data> cur;
    llama_token_data_array a;
    raw_array(x, cur, a);
    llama_sampler * trim = llama_sampler_init_top_k((int32_t) width);
    llama_sampler_apply(trim, &a);
    llama_sampler_free(trim);
    llama_sampler_apply(grmr, &a);
    size_t n = 0;
    for (size_t i = 0; i < a.size; ++i) {
        n += a.data[i].logit != -INFINITY ? 1 : 0;
    }
    return n;
}

struct pair_state {
    llama_sampler * full_chain = nullptr;
    llama_sampler * fast_chain = nullptr;
    llama_sampler * full_grmr  = nullptr;
    llama_sampler * fast_grmr  = nullptr;
    common_penfast_cfg            cfg;
    common_gramfast_scratch       sc;
    std::vector<llama_token_data> cur_full;
    std::vector<llama_token_data> cur_fast;

    void free_all() {
        llama_sampler_free(full_chain);
        llama_sampler_free(fast_chain);
        llama_sampler_free(full_grmr);
        llama_sampler_free(fast_grmr);
        full_chain = nullptr;
        fast_chain = nullptr;
        full_grmr  = nullptr;
        fast_grmr  = nullptr;
    }
};

struct stats {
    int n_steps         = 0;
    int n_pass[3]       = { 0, 0, 0 }; // the direct call: refused / exact at the first width / at the second
    int n_flow[3]       = { 0, 0, 0 }; // the flow: NONE, DONE, RESAMPLE
    int n_fast_resample = 0;           // DONE through the fast resample after a rejected token
    int n_narrow        = 0;           // refusals with fewer finite candidates than top_k in the widest set
    int n_dense         = 0;           // dense untied steps (stand-in only) and how often the direct call decided there
    int n_dense_fast    = 0;
};

// one position on both sides: the state right after the chain's top-k (the full path: the grammar, then the samplers
// [0, i_top_k] one by one), then the flow of common_sampler_sample on both; the token goes into both chains, and into both
// grammars with accept_grammar. False on a mismatch.
static bool step(pair_state & s, const std::vector<float> & x, bool grammar_first, bool use_grammar, bool accept_grammar,
        const common_pretrim_cfg * pcfg, int it, stats & st, llama_token * out, common_gramfast_result * r_out = nullptr,
        bool * fast_k_out = nullptr) {
    const int n_vocab = (int) x.size();
    llama_token_data_array a_full;
    llama_token_data_array a_fast;
    bool ok = true;

    raw_array(x, s.cur_full, a_full);
    llama_sampler_apply(s.full_grmr, &a_full);
    for (int32_t i = 0; i <= s.cfg.i_top_k; ++i) {
        llama_sampler_apply(llama_sampler_chain_get(s.full_chain, i), &a_full);
    }
    const bool fast_k = common_sampler_gramfast_apply(s.cfg, s.fast_chain, s.fast_grmr, x.data(), n_vocab, true, s.sc,
            s.cur_fast, a_fast);
    if (fast_k_out) {
        *fast_k_out = fast_k;
    }
    st.n_pass[fast_k ? s.sc.n_pass : 0]++;
    check(!fast_k || s.sc.n_pass == 1 || s.sc.n_pass == 2, "n_pass out of range", it);
    if (fast_k) {
        if (!same_exact(a_full, a_fast, false)) {
            check(false, "candidates after the top-k differ", it);
            ok = false;
        }
        check(a_fast.size == (size_t) s.cfg.top_k, "top-k size", it);
    } else if (pcfg != nullptr && pcfg->top_k > 0 && s.sc.n_fin != SIZE_MAX && s.sc.width >= pcfg->width &&
            s.sc.n_fin < (size_t) pcfg->top_k + pcfg->n_extra) {
        // a narrow refusal: the pre-trim's first trim keeps too few valid tokens too, it takes the full pass as well
        st.n_narrow++;
        check(pretrim_first_valid(x, pcfg->width, s.full_grmr) < (size_t) pcfg->top_k + pcfg->n_extra,
                "narrow refusal where the pre-trim would not take the full pass", it);
    }

    const llama_token id_full = full_sample(s.full_chain, s.full_grmr, grammar_first, use_grammar, x, s.cur_full, a_full);
    common_gramfast_result r = COMMON_GRAMFAST_NONE;
    const llama_token id_fast = fast_sample(s.cfg, s.fast_chain, s.fast_grmr, grammar_first, use_grammar, x, s.sc,
            s.cur_fast, a_fast, r);
    if (r_out) {
        *r_out = r;
    }
    st.n_flow[(int) r]++;
    if (r == COMMON_GRAMFAST_DONE && !grammar_first && use_grammar && s.sc.n_pass > 0) {
        st.n_fast_resample++;
    }
    if (!same_exact(a_full, a_fast, true)) {
        check(false, r == COMMON_GRAMFAST_DONE ? "chain result differs (fast path)" : "chain result differs (fallback)", it);
        ok = false;
    }
    check(id_full == id_fast, "sampled token differs", it);
    check(id_full != LLAMA_TOKEN_NULL, "no selected token", it);
    if (!ok || id_full != id_fast || id_full == LLAMA_TOKEN_NULL) {
        return false;
    }

    llama_sampler_accept(s.full_chain, id_full);
    llama_sampler_accept(s.fast_chain, id_fast);
    if (accept_grammar) {
        llama_sampler_accept(s.full_grmr, id_full);
        llama_sampler_accept(s.fast_grmr, id_fast);
    }
    st.n_steps++;
    if (out) {
        *out = id_full;
    }
    return true;
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

static void make_pair(pair_state & s, const common_params_sampling & p, const std::vector<llama_token> & suppress,
        int n_vocab, llama_sampler * full_grmr, llama_sampler * fast_grmr) {
    s.full_chain = make_chain(p, suppress, n_vocab);
    s.fast_chain = make_chain(p, suppress, n_vocab);
    s.full_grmr  = full_grmr;
    s.fast_grmr  = fast_grmr;
    s.cfg = common_sampler_penfast_cfg(p, s.fast_chain, suppress.empty() ? nullptr : suppress.data(),
            (int32_t) suppress.size(), n_vocab);
}

// the window of the chain's penalties sampler (empty without one)
static std::vector<llama_token> window_of(const pair_state & s) {
    std::vector<llama_token> w;
    if (s.cfg.i_pen.empty()) {
        return w;
    }
    const llama_sampler * pen = llama_sampler_chain_get(s.fast_chain, s.cfg.i_pen[0]);
    const int32_t nw = llama_sampler_penalties_window(pen, nullptr, 0);
    if (nw > 0) {
        w.resize((size_t) nw);
        llama_sampler_penalties_window(pen, w.data(), nw);
    }
    return w;
}

// raises some tokens of the row: 1 valid ones to the top (and a few invalid), 2 valid ones below many invalid ones,
// 3 penalty-window tokens and some valid ones; 0 nothing
static void boost(std::vector<float> & x, const std::vector<char> & valid, const std::vector<llama_token> & window, int mode,
        std::mt19937 & g) {
    if (mode == 0) {
        return;
    }
    std::vector<llama_token> vt;
    std::vector<llama_token> iv;
    for (size_t i = 0; i < x.size(); ++i) {
        (valid[i] ? vt : iv).push_back((llama_token) i);
    }
    auto pick = [&](const std::vector<llama_token> & from, int n, float add) {
        if (from.empty()) {
            return;
        }
        std::uniform_int_distribution<size_t> u(0, from.size() - 1);
        for (int j = 0; j < n; ++j) {
            x[(size_t) from[u(g)]] += add;
        }
    };
    if (mode == 1) {
        pick(vt, 40, 9.0f);
        pick(iv, 10, 9.0f);
    } else if (mode == 2) {
        pick(iv, 300, 12.0f);
        pick(vt, 30, 7.0f);
    } else {
        for (size_t j = 0; j < window.size() && j < 8; ++j) {
            x[(size_t) window[j]] += 10.0f;
        }
        pick(vt, 20, 8.0f);
    }
}

// dens 0: 97 % valid, 1: 50 %, 2: 10 %, 3: 1 %, 4: 8 tokens
static void make_mask(std::vector<char> & valid, int dens, std::mt19937 & g) {
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    if (dens == 4) {
        std::fill(valid.begin(), valid.end(), (char) 0);
        std::uniform_int_distribution<size_t> u(0, valid.size() - 1);
        for (int j = 0; j < 8; ++j) {
            valid[u(g)] = 1;
        }
        return;
    }
    const float d = dens == 0 ? 0.97f : dens == 1 ? 0.5f : dens == 2 ? 0.1f : 0.01f;
    for (auto & c : valid) {
        c = u01(g) < d ? 1 : 0;
    }
}

// hand-made rows: refusals, ties at theta, a penalised tie, the second width, a narrow position, a fast resample
static void test_cases(stats & st) {
    const int n_vocab = 30000;
    const int K = 20;
    int it = 9000;
    std::vector<char> valid((size_t) n_vocab, 1);
    mask_grammar mg;
    mg.valid = &valid;
    const common_params_sampling p = kyro_params();
    const common_pretrim_cfg pcfg = common_sampler_pretrim_cfg(p, nullptr, 0, n_vocab, 1);
    check(pcfg.top_k == K && pcfg.width == 1024, "pre-trim cfg for Kyro's parameters", it);

    auto fresh = [&](pair_state & s) {
        make_pair(s, p, {}, n_vocab, llama_sampler_init(&g_mask_iface, &mg), llama_sampler_init(&g_mask_iface, &mg));
        check(s.cfg.top_k == K, "cfg off", it);
    };
    auto noise = [&](std::vector<float> & x, uint32_t seed, float mean, float sd) {
        std::mt19937 g(seed);
        std::normal_distribution<float> nd(mean, sd);
        x.resize((size_t) n_vocab);
        for (auto & v : x) {
            v = nd(g);
        }
    };

    // NaN, +inf and a row with fewer finite tokens than the second width: refused, nothing ran
    {
        pair_state s;
        fresh(s);
        std::vector<float> x;
        llama_token_data_array a;
        noise(x, 1, 0.0f, 2.0f);
        x[123] = NAN;
        check(!common_sampler_gramfast_apply(s.cfg, s.fast_chain, s.fast_grmr, x.data(), n_vocab, false, s.sc, s.cur_fast, a),
                "NaN row not refused", it);
        x[123] = INFINITY;
        check(!common_sampler_gramfast_apply(s.cfg, s.fast_chain, s.fast_grmr, x.data(), n_vocab, false, s.sc, s.cur_fast, a),
                "+inf row not refused", it);
        std::fill(x.begin(), x.end(), -INFINITY);
        for (int j = 0; j < 100; ++j) {
            x[(size_t) (j * 37)] = 5.0f - 0.01f * (float) j;
        }
        check(!common_sampler_gramfast_apply(s.cfg, s.fast_chain, s.fast_grmr, x.data(), n_vocab, false, s.sc, s.cur_fast, a),
                "masked row not refused", it);
        llama_token id = LLAMA_TOKEN_NULL;
        check(step(s, x, true, true, false, &pcfg, it, st, &id), "masked row: fallback differs", it);
        s.free_all();
        it++;
    }
    // 3000 tokens tie at both thetas: all taken, the top-k above them decides at the first width
    {
        pair_state s;
        fresh(s);
        std::vector<float> x;
        noise(x, 2, -5.0f, 1.0f);
        for (int j = 0; j < 25; ++j) {
            x[(size_t) (1000 + j)] = 30.0f - (float) j;
        }
        for (int j = 0; j < 3000; ++j) {
            x[(size_t) (5000 + j)] = 1.0f;
            valid[(size_t) (5000 + j)] = (char) (j % 2);
        }
        // after_top_k: only samplers without state run, so the step below starts from the same state
        llama_token_data_array a;
        check(common_sampler_gramfast_apply(s.cfg, s.fast_chain, s.fast_grmr, x.data(), n_vocab, true, s.sc, s.cur_fast, a) &&
                s.sc.n_pass == 1 && s.sc.top.size() == 3025, "ties at theta: not all ties taken", it);
        bool fast_k = false;
        llama_token id = LLAMA_TOKEN_NULL;
        check(step(s, x, true, true, false, &pcfg, it, st, &id, nullptr, &fast_k), "ties at theta: result differs", it);
        check(fast_k, "ties at theta: refused", it);
        // more ties than 4 x the second width: refused, the fallback equals the full path
        for (int j = 0; j < 5000; ++j) {
            x[(size_t) (5000 + j)] = 1.0f;
        }
        check(step(s, x, true, true, false, &pcfg, it, st, &id, nullptr, &fast_k), "many ties: result differs", it);
        check(!fast_k, "many ties: not refused", it);
        std::fill(valid.begin(), valid.end(), (char) 1);
        s.free_all();
        it++;
    }
    // a penalised window token lands on an unmoved token's logit inside the top-k: refused
    {
        pair_state s;
        fresh(s);
        std::vector<float> x;
        noise(x, 3, -8.0f, 1.0f);
        for (int j = 0; j < K; ++j) {
            x[(size_t) (100 + j)] = 20.0f - (float) j * 0.75f;
        }
        x[500] = 10.0f; // in the window: 10.0 - 1.5 = 8.5
        x[501] = 8.5f;  // unmoved
        llama_sampler_accept(s.full_chain, 500);
        llama_sampler_accept(s.fast_chain, 500);
        bool fast_k = true;
        llama_token id = LLAMA_TOKEN_NULL;
        check(step(s, x, true, true, false, &pcfg, it, st, &id, nullptr, &fast_k), "penalised tie: result differs", it);
        check(!fast_k, "penalised tie inside the top-k not refused", it);
        s.free_all();
        it++;
    }
    // the 300 best raw tokens are invalid, 50 valid ones follow: the second width, in both flows
    {
        for (int flow = 0; flow < 2; ++flow) {
            pair_state s;
            fresh(s);
            std::vector<float> x;
            noise(x, 4, 0.0f, 2.0f);
            for (int j = 0; j < 300; ++j) {
                x[(size_t) (100 + j)] = 20.0f + 0.01f * (float) j;
                valid[(size_t) (100 + j)] = 0;
            }
            for (int j = 0; j < 50; ++j) {
                x[(size_t) (1000 + j)] = 12.0f - 0.1f * (float) j;
            }
            bool fast_k = false;
            common_gramfast_result r = COMMON_GRAMFAST_NONE;
            llama_token id = LLAMA_TOKEN_NULL;
            check(step(s, x, flow == 0, true, false, &pcfg, it, st, &id, &r, &fast_k), "second width: result differs", it);
            check(fast_k && r == COMMON_GRAMFAST_DONE && s.sc.n_pass == 2, "second width: not decided there", it);
            check(id >= 1000 && id < 1050, "second width: sampled token outside the valid ones", it);
            std::fill(valid.begin(), valid.end(), (char) 1);
            s.free_all();
            it++;
        }
    }
    // 6 valid tokens in the vocabulary: refused as narrow, in both flows
    {
        for (int flow = 0; flow < 2; ++flow) {
            pair_state s;
            fresh(s);
            std::vector<float> x;
            noise(x, 5, 0.0f, 2.0f);
            std::fill(valid.begin(), valid.end(), (char) 0);
            const llama_token six[6] = { 7, 4000, 9999, 15000, 22222, 29999 };
            for (const llama_token t : six) {
                valid[(size_t) t] = 1;
            }
            const int n_narrow0 = st.n_narrow;
            common_gramfast_result r = COMMON_GRAMFAST_DONE;
            llama_token id = LLAMA_TOKEN_NULL;
            check(step(s, x, flow == 0, true, false, &pcfg, it, st, &id, &r), "narrow: result differs", it);
            check(st.n_narrow == n_narrow0 + 1, "narrow: not seen as narrow", it);
            check(r == (flow == 0 ? COMMON_GRAMFAST_NONE : COMMON_GRAMFAST_RESAMPLE) || r == COMMON_GRAMFAST_DONE,
                    "narrow: flow result", it);
            check(std::find(six, six + 6, id) != six + 6, "narrow: sampled token not valid", it);
            std::fill(valid.begin(), valid.end(), (char) 1);
            s.free_all();
            it++;
        }
    }
    // the chain's own token is invalid with p ~ 1: rejected, the fast resample decides at the first width
    {
        pair_state s;
        fresh(s);
        std::vector<float> x;
        noise(x, 6, 0.0f, 2.0f);
        x[7] = 40.0f;
        valid[7] = 0;
        common_gramfast_result r = COMMON_GRAMFAST_NONE;
        llama_token id = LLAMA_TOKEN_NULL;
        check(step(s, x, false, true, false, &pcfg, it, st, &id, &r), "fast resample: result differs", it);
        check(r == COMMON_GRAMFAST_DONE && s.sc.n_pass == 1 && id != 7, "fast resample: not decided there", it);
        // without the grammar (grammar_should_apply false) the chain alone decides, invalid tokens included
        x[8] = 60.0f;
        valid[8] = 0;
        check(step(s, x, false, false, false, &pcfg, it, st, &id, &r), "no grammar: result differs", it);
        check(r == COMMON_GRAMFAST_DONE && s.sc.n_pass == 0 && id == 8, "no grammar: not decided by the chain alone", it);
        std::fill(valid.begin(), valid.end(), (char) 1);
        s.free_all();
        it++;
    }
}

// part 1: random rows, masks, parameters and histories with the stand-in grammar
static void test_random(stats & st) {
    const int n_cases = 150;
    const int n_step  = 4;
    for (int it = 0; it < n_cases; ++it) {
        std::mt19937 g(2000u + (uint32_t) it);
        std::uniform_real_distribution<float> u01(0.0f, 1.0f);

        const int  n_vocab = it % 4 == 0 ? 248000 : it % 4 == 1 ? 20000 : 20000 + (int) (u01(g) * 228000.0f);
        const bool tied    = it % 7 == 6; // logits on a coarse grid: many ties
        const int  dens    = it % 5;

        common_params_sampling p = kyro_params();
        p.seed = 7000u + (uint32_t) it;
        const int variant = it % 6;
        if (variant == 1) {        // random lowering penalties
            p.penalty_repeat = 1.0f + u01(g) * 0.5f; p.penalty_freq = u01(g); p.penalty_present = u01(g) * 2.0f;
        } else if (variant == 2) { // raising penalties
            p.penalty_repeat = 0.7f + u01(g) * 0.3f; p.penalty_freq = -u01(g) * 0.5f; p.penalty_present = -u01(g);
        } else if (variant == 3) { // other sampling parameters
            p.top_k = 1 + (int) (u01(g) * 127.0f); p.top_p = 0.5f + u01(g) * 0.5f; p.min_p = 0.05f;
            p.temp = 0.2f + u01(g) * 1.3f; p.penalty_last_n = 16 + (int) (u01(g) * 240.0f);
            p.penalty_repeat = 1.1f; p.penalty_freq = 0.2f;
        } else if (variant == 4) { // no penalties: the chain keeps a no-op in their place
            p.penalty_present = 0.0f;
        }
        // variant 0, 5: Kyro's

        std::vector<float> x((size_t) n_vocab);
        std::normal_distribution<float> nd(0.0f, 2.0f);
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
            suppress = { best[(size_t) (it % 7)], (llama_token) (u01(g) * (float) (n_vocab - 1)) };
            if (suppress[0] == suppress[1]) {
                suppress.pop_back();
            }
        }
        if (it % 4 == 2) { // user biases: up, down, -inf, on a best token and on random ones
            std::set<llama_token> used(suppress.begin(), suppress.end());
            const llama_token cand[4] = { best[3], (llama_token) (u01(g) * (float) (n_vocab - 1)),
                                          (llama_token) (u01(g) * (float) (n_vocab - 1)), best[10] };
            const float bias[4] = { -2.5f, 12.0f, -INFINITY, 1.25f };
            for (int j = 0; j < 4; ++j) {
                if (used.insert(cand[j]).second) {
                    p.logit_bias.push_back({ cand[j], bias[j] });
                }
            }
        }

        std::vector<char> valid((size_t) n_vocab, 1);
        mask_grammar mg;
        mg.valid = &valid;
        pair_state s;
        make_pair(s, p, suppress, n_vocab, llama_sampler_init(&g_mask_iface, &mg), llama_sampler_init(&g_mask_iface, &mg));
        check(s.cfg.top_k == p.top_k, "cfg off for a supported configuration", it);
        if (s.cfg.top_k <= 0) {
            s.free_all();
            continue;
        }
        const common_pretrim_cfg pcfg = common_sampler_pretrim_cfg(p, suppress.empty() ? nullptr : suppress.data(),
                (int32_t) suppress.size(), n_vocab, 1);

        // the accepted history: mostly the best raw tokens, with repeats
        const int n_hist = (int) (u01(g) * 300.0f);
        for (int j = 0; j < n_hist; ++j) {
            const llama_token t = u01(g) < 0.6f ? best[(size_t) (u01(g) * 40.0f)] : (llama_token) (u01(g) * (float) (n_vocab - 1));
            llama_sampler_accept(s.full_chain, t);
            llama_sampler_accept(s.fast_chain, t);
        }

        for (int k = 0; k < n_step; ++k) {
            if (k > 0) {
                fill();
            }
            make_mask(valid, dens, g);
            const int mode = (it + k) % 4;
            boost(x, valid, window_of(s), mode, g);
            const bool grammar_first = (it + k) % 2 == 0;
            const bool use_grammar   = it % 9 != 8;
            const int  n_pass0       = st.n_pass[0];
            llama_token id = LLAMA_TOKEN_NULL;
            if (!step(s, x, grammar_first, use_grammar, false, &pcfg, it, st, &id)) {
                break;
            }
            if (!tied && dens <= 1 && mode != 2) {
                st.n_dense++;
                st.n_dense_fast += st.n_pass[0] == n_pass0 ? 1 : 0;
            }
        }
        s.free_all();
    }
}

// part 2: the real grammar of a tool call over a real vocabulary
static const char * k_tool_gbnf =
    R"g(root   ::= "{" ws "\"name\"" ws ":" ws string "," ws "\"arguments\"" ws ":" ws object "}" ws)g" "\n"
    R"g(value  ::= object | array | string | number | ("true" | "false" | "null") ws)g" "\n"
    R"g(object ::= "{" ws ( string ":" ws value ( "," ws string ":" ws value )* )? "}" ws)g" "\n"
    R"g(array  ::= "[" ws ( value ( "," ws value )* )? "]" ws)g" "\n"
    R"g(string ::= "\"" ( [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4}) )* "\"" ws)g" "\n"
    R"g(number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9] [1-9]{0,15})? ws)g" "\n"
    R"g(ws     ::= | " " | "\n" [ \t]{0,20})g" "\n";

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    std::vector<llama_token> toks(text.size() + 8);
    int32_t n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), false, false);
    if (n < 0) {
        toks.resize((size_t) -n);
        n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), false, false);
    }
    toks.resize(n > 0 ? (size_t) n : 0);
    return toks;
}

static void test_real(const char * vocab_path, stats & st) {
    llama_model_params mparams = llama_model_default_params();
    mparams.vocab_only = true;
    llama_model * model = llama_model_load_from_file(vocab_path, mparams);
    if (model == nullptr) {
        check(false, "could not load the vocabulary", -2);
        return;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    const char * prefixes[] = {
        "",
        "{\"name\": \"get_wea",
        "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"",
        "{\"name\": \"get_weather\", \"arguments\": {\"days\": 1",
        "{\"name\": \"write_file\", \"arguments\": {\"content\": \"Hello world, this is a longer text with ",
        "{\"name\": \"x\", \"arguments\": {\"list\": [",
    };
    const int n_prefix = (int) (sizeof(prefixes) / sizeof(prefixes[0]));
    const int n_step   = 6;
    const char * never[] = { "\\[\\[never-in-random-text\\]\\]([\\s\\S]*)" };

    int run = 0;
    for (int pi = 0; pi <= n_prefix; ++pi) {
        const bool lazy = pi == n_prefix; // a lazy grammar waiting for its trigger, no prefix
        for (int gf = 0; gf < 2; ++gf, ++run) {
            const int it = 20000 + run;
            std::mt19937 g(31000u + (uint32_t) run);

            common_params_sampling p = kyro_params();
            p.seed = 9100u + (uint32_t) run;
            std::vector<llama_token> suppress;
            if (run % 3 == 1) {
                suppress = { 11, 12345 };
                p.logit_bias = { { 220, -3.0f }, { 1000, 6.0f } };
            } else if (run % 3 == 2) {
                p.penalty_present = 0.0f;
            }

            llama_sampler * gfull = lazy ? llama_sampler_init_grammar_lazy_patterns(vocab, k_tool_gbnf, "root", never, 1, nullptr, 0)
                                         : llama_sampler_init_grammar(vocab, k_tool_gbnf, "root");
            llama_sampler * gfast = lazy ? llama_sampler_init_grammar_lazy_patterns(vocab, k_tool_gbnf, "root", never, 1, nullptr, 0)
                                         : llama_sampler_init_grammar(vocab, k_tool_gbnf, "root");
            if (gfull == nullptr || gfast == nullptr) {
                check(false, "the grammar does not parse", it);
                llama_sampler_free(gfull);
                llama_sampler_free(gfast);
                continue;
            }
            pair_state s;
            make_pair(s, p, suppress, n_vocab, gfull, gfast);
            check(s.cfg.top_k == p.top_k, "cfg off", it);
            if (s.cfg.top_k <= 0) {
                s.free_all();
                continue;
            }
            const common_pretrim_cfg pcfg = common_sampler_pretrim_cfg(p, suppress.empty() ? nullptr : suppress.data(),
                    (int32_t) suppress.size(), n_vocab, 1);

            // the prefix as accepted tokens: into both chains and both grammars
            if (!lazy) {
                for (const llama_token t : tokenize(vocab, prefixes[pi])) {
                    llama_sampler_accept(s.full_chain, t);
                    llama_sampler_accept(s.fast_chain, t);
                    llama_sampler_accept(s.full_grmr, t);
                    llama_sampler_accept(s.fast_grmr, t);
                }
            }

            std::vector<float> x((size_t) n_vocab);
            std::vector<char> valid((size_t) n_vocab);
            std::vector<llama_token_data> cur;
            std::normal_distribution<float> nd(0.0f, 2.0f);
            for (int k = 0; k < n_step; ++k) {
                for (auto & v : x) {
                    v = nd(g);
                }
                // what the grammar accepts at this position (its apply keeps no state)
                llama_token_data_array a;
                raw_array(x, cur, a);
                llama_sampler_apply(s.full_grmr, &a);
                for (size_t i = 0; i < a.size; ++i) {
                    valid[(size_t) a.data[i].id] = a.data[i].logit != -INFINITY ? 1 : 0;
                }
                boost(x, valid, window_of(s), (run + k) % 4, g);

                const bool grammar_first = gf == 0;
                llama_token id = LLAMA_TOKEN_NULL;
                if (!step(s, x, grammar_first, true, true, &pcfg, it, st, &id)) {
                    break;
                }
                if (lazy) {
                    check(llama_sampler_grammar_awaiting_trigger(s.fast_grmr), "the lazy grammar triggered", it);
                }
                if (llama_vocab_is_eog(vocab, id)) {
                    break;
                }
            }
            s.free_all();
        }
    }
    llama_model_free(model);
}

int main(int argc, char ** argv) {
    stats st1;
    test_cases(st1);
    test_random(st1);
    printf("test-r29-gramfast: stand-in grammar: %d steps, direct call refused %d / width 1 %d / width 2 %d, flow none %d / "
           "done %d / resample %d, fast resamples %d, narrow refusals %d, dense untied fast %d of %d\n",
           st1.n_steps, st1.n_pass[0], st1.n_pass[1], st1.n_pass[2], st1.n_flow[0], st1.n_flow[1], st1.n_flow[2],
           st1.n_fast_resample, st1.n_narrow, st1.n_dense_fast, st1.n_dense);
    check(st1.n_pass[1] > 0 && st1.n_pass[2] > 0, "a width never decided", -1);
    check(st1.n_fast_resample > 0, "the fast resample never decided", -1);
    check(st1.n_narrow > 0, "no narrow refusal seen", -1);
    check(st1.n_flow[0] > 0 && st1.n_flow[2] > 0, "the fallbacks never ran", -1);
    check(st1.n_dense > 0 && st1.n_dense_fast * 100 >= st1.n_dense * 90, "the fast path decided fewer than 90% of the dense untied steps", -1);

    if (argc > 1) {
        llama_backend_init();
        stats st2;
        test_real(argv[1], st2);
        llama_backend_free();
        printf("test-r29-gramfast: real grammar: %d steps, direct call refused %d / width 1 %d / width 2 %d, flow none %d / "
               "done %d / resample %d, fast resamples %d, narrow refusals %d\n",
               st2.n_steps, st2.n_pass[0], st2.n_pass[1], st2.n_pass[2], st2.n_flow[0], st2.n_flow[1], st2.n_flow[2],
               st2.n_fast_resample, st2.n_narrow);
        check(st2.n_steps >= 40, "the real grammar ran too few steps", -2);
        check(st2.n_pass[1] > 0, "the real grammar never decided at the first width", -2);
        check(st2.n_flow[1] > 0, "the real grammar flow never decided fast", -2);
    } else {
        printf("test-r29-gramfast: no vocabulary given, the real grammar part is skipped\n");
    }

    if (n_fail > 0) {
        printf("test-r29-gramfast: %d FAILED\n", n_fail);
        return 1;
    }
    printf("test-r29-gramfast: OK\n");
    return 0;
}
