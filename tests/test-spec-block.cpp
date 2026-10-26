// [TAG_FN_L3_MTP_BLOCK] CPU unit test of block verification (common/speculative-block.h).
//
// 1. Exactness by enumeration: a 5-token vocabulary, random target and draft conditionals per prefix, 1-3 drafts. Over
//    every draft sequence (its probability under the drafter), every kept length (h_t * prod_{j > t} (1 - h_j)) and every
//    correction token, the joint distribution of the first two output tokens (the second from the target when only one
//    is output) must equal the target's. The expected kept length must be at least the per-token rule's.
// 2. The draw: tau from uniforms against the exact distribution (Monte Carlo), and fixed cases.
// 3. Edge cases: one draft = the per-token rule, draft = target keeps everything, a draft outside the target's support.
// 4. [TAG_V0262_BLOCKVERIFY] Drafts stopped early by a rule on the prefix keep the target's output; a stop on the drawn
//    token's own probability is shown to skew it.
// 5. [TAG_V0262_BLOCKVERIFY] Proposals given as dists: duplicate ids are summed, delta proposals keep the target's output.
// No model, no GPU. Exit code 0 iff every check passes.

#include "../common/speculative-block.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <utility>
#include <vector>

static int g_fail   = 0;
static int g_checks = 0;

#define TCHECK(cond, ...)                                                   \
    do {                                                                    \
        ++g_checks;                                                         \
        if (!(cond)) {                                                      \
            ++g_fail;                                                       \
            fprintf(stderr, "FAIL %s:%d: ", __func__, __LINE__);            \
            fprintf(stderr, __VA_ARGS__);                                   \
            fprintf(stderr, "\n");                                          \
        }                                                                   \
    } while (0)

static const int V = 5;

typedef std::vector<llama_token_data> dist_t;

// a distribution over V tokens (every token present, float probabilities summing to ~1); zero_tail: the last token gets
// probability 0 and is left out (a truncated target)
static dist_t make_dist(const std::vector<double> & logits, bool zero_tail) {
    double mx = -1e30;
    for (double l : logits) {
        mx = std::max(mx, l);
    }
    std::vector<double> e(V);
    double s = 0.0;
    for (int k = 0; k < V; ++k) {
        e[k] = (zero_tail && k == V - 1) ? 0.0 : std::exp(logits[k] - mx);
        s += e[k];
    }
    dist_t d;
    for (int k = 0; k < V; ++k) {
        if (e[k] > 0.0) {
            d.push_back({ (llama_token) k, 0.0f, (float) (e[k]/s) });
        }
    }
    return d;
}

static double prob(const dist_t & d, llama_token id) {
    for (const auto & e : d) {
        if (e.id == id) {
            return e.p;
        }
    }
    return 0.0;
}

static common_spec_block_dist view(const dist_t & d) {
    return { d.data(), d.size() };
}

struct model {
    std::mt19937 rng;
    double sharp, mix;
    bool truncate;
    std::map<std::vector<llama_token>, std::pair<dist_t, dist_t>> m; // prefix -> (target, draft)

    const std::pair<dist_t, dist_t> & at(const std::vector<llama_token> & prefix) {
        auto it = m.find(prefix);
        if (it != m.end()) {
            return it->second;
        }
        std::normal_distribution<double> n(0.0, 1.0);
        std::vector<double> lp(V), lq(V);
        for (int k = 0; k < V; ++k) {
            lp[k] = n(rng)*sharp;
        }
        for (int k = 0; k < V; ++k) {
            lq[k] = (1.0 - mix)*lp[k] + mix*n(rng)*sharp;
        }
        return m[prefix] = { make_dist(lp, truncate), make_dist(lq, false) };
    }
};

// exact output statistics of one configuration: max |joint(first two) - target|, E[tau] block and per-token
static void exact_case(int gamma, double sharp, double mix, bool truncate, uint32_t seed, double & err, double & e_blk, double & e_tok) {
    model md{ std::mt19937(seed), sharp, mix, truncate, {} };
    std::vector<std::vector<double>> out(V, std::vector<double>(V, 0.0));
    e_blk = e_tok = 0.0;

    std::vector<llama_token> xs(gamma, 0);
    const int n_seq = (int) std::pow(V, gamma);
    for (int code = 0; code < n_seq; ++code) {
        int c = code;
        for (int i = 0; i < gamma; ++i) {
            xs[i] = (llama_token) (c % V);
            c /= V;
        }
        // drafter probability of the sequence, and the per-row distributions
        double pr = 1.0;
        std::vector<dist_t> P(gamma + 1), Q(gamma);
        for (int i = 0; i <= gamma; ++i) {
            const std::vector<llama_token> prefix(xs.begin(), xs.begin() + i);
            const auto & pq = md.at(prefix);
            P[i] = pq.first;
            if (i < gamma) {
                Q[i] = pq.second;
                pr *= prob(Q[i], xs[i]);
            }
        }
        if (pr == 0.0) {
            continue;
        }
        // the per-token rule
        double cum = 1.0;
        for (int i = 0; i < gamma; ++i) {
            const double q = prob(Q[i], xs[i]);
            cum *= q > 0.0 ? std::min(1.0, prob(P[i], xs[i])/q) : 0.0;
            e_tok += pr*cum;
        }
        // the block rule
        std::vector<common_spec_block_dist> pd(gamma), qd(gamma);
        for (int i = 0; i < gamma; ++i) {
            pd[i] = view(P[i]);
            qd[i] = view(Q[i]);
        }
        std::vector<double> w, h;
        common_spec_block_weights(pd.data(), qd.data(), xs.data(), gamma, w, h);
        for (int t = 0; t <= gamma; ++t) {
            double pt = t > 0 ? h[t] : 1.0;
            for (int j = t + 1; j <= gamma; ++j) {
                pt *= 1.0 - h[j];
            }
            if (pt == 0.0) {
                continue;
            }
            e_blk += pr*pt*t;
            dist_t y;
            if (t == gamma) {
                y = P[gamma];
            } else {
                std::vector<llama_token_data> r;
                const double s = common_spec_block_excess(w[t], view(P[t]), view(Q[t]), &r);
                y = s > 0.0 ? r : P[t];
            }
            double ys = 0.0;
            for (const auto & e : y) {
                ys += e.p;
            }
            for (const auto & e : y) {
                const double wy = pr*pt*e.p/ys;
                std::vector<llama_token> seq(xs.begin(), xs.begin() + t);
                seq.push_back(e.id);
                if (seq.size() >= 2) {
                    out[seq[0]][seq[1]] += wy;
                } else {
                    // one token out: the next step samples the second from the target
                    const auto & p2 = md.at({ seq[0] }).first;
                    for (const auto & f : p2) {
                        out[seq[0]][f.id] += wy*f.p;
                    }
                }
            }
        }
    }
    // the target's joint of the first two tokens
    err = 0.0;
    const auto p1 = md.at({}).first;
    for (int a = 0; a < V; ++a) {
        const double pa = prob(p1, a);
        const auto p2 = pa > 0.0 ? md.at({ (llama_token) a }).first : dist_t();
        for (int b = 0; b < V; ++b) {
            const double tgt = pa*prob(p2, b);
            err = std::max(err, std::fabs(out[a][b] - tgt));
        }
    }
}

static void test_exact() {
    double worst = 0.0;
    for (int gamma = 1; gamma <= 3; ++gamma) {
        double sum_b = 0.0, sum_t = 0.0;
        int n = 0;
        for (double mix : { 0.15, 0.35, 0.6, 0.9 }) {
            for (bool truncate : { false, true }) {
                for (uint32_t seed = 1; seed <= 6; ++seed) {
                    double err, eb, et;
                    exact_case(gamma, 1.6, mix, truncate, seed*7919 + gamma, err, eb, et);
                    worst = std::max(worst, err);
                    TCHECK(err < 2e-6, "gamma %d mix %.2f trunc %d seed %u: joint error %.2e", gamma, mix, (int) truncate, seed, err);
                    TCHECK(eb >= et - 1e-9, "gamma %d mix %.2f: block %.6f < per-token %.6f", gamma, mix, eb, et);
                    if (gamma == 1) {
                        TCHECK(std::fabs(eb - et) < 1e-9, "one draft: block %.9f != per-token %.9f", eb, et);
                    }
                    sum_b += eb;
                    sum_t += et;
                    n++;
                }
            }
        }
        printf("  gamma %d: kept drafts per step, per-token %.4f, block %.4f (+%.2f %% tokens per step incl. the bonus)\n",
                gamma, sum_t/n, sum_b/n, 100.0*((1.0 + sum_b/n)/(1.0 + sum_t/n) - 1.0));
    }
    printf("  worst joint error %.2e\n", worst);
}

static void test_draw() {
    // fixed cases
    const std::vector<double> h = { 0.0, 0.3, 0.8 };
    const float u00[] = { 0.5f, 0.9f };
    const float u10[] = { 0.2f, 0.9f };
    const float u02[] = { 0.5f, 0.7f };
    const float u12[] = { 0.2f, 0.7f };
    TCHECK(common_spec_block_tau(h, 2, u00) == 0, "tau 0");
    TCHECK(common_spec_block_tau(h, 2, u10) == 1, "tau 1");
    TCHECK(common_spec_block_tau(h, 2, u02) == 2, "tau 2 without the first");
    TCHECK(common_spec_block_tau(h, 2, u12) == 2, "tau 2");
    const std::vector<double> h0 = { 0.0, 0.0, 0.0 };
    const float uz[] = { 0.0f, 0.0f };
    TCHECK(common_spec_block_tau(h0, 2, uz) == 0, "h = 0 never keeps, even at u = 0");

    // Monte Carlo against the exact distribution
    const std::vector<double> hm = { 0.0, 0.45, 0.2, 0.6 };
    double exact[4];
    for (int t = 0; t <= 3; ++t) {
        double pt = t > 0 ? hm[t] : 1.0;
        for (int j = t + 1; j <= 3; ++j) {
            pt *= 1.0 - hm[j];
        }
        exact[t] = pt;
    }
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    int cnt[4] = { 0, 0, 0, 0 };
    const int N = 400000;
    for (int k = 0; k < N; ++k) {
        float u[3] = { uni(rng), uni(rng), uni(rng) };
        cnt[common_spec_block_tau(hm, 3, u)]++;
    }
    for (int t = 0; t <= 3; ++t) {
        TCHECK(std::fabs((double) cnt[t]/N - exact[t]) < 0.004, "tau %d: %.4f vs exact %.4f", t, (double) cnt[t]/N, exact[t]);
    }
}

static void test_edges() {
    // draft = target: every weight 1, every h 1, all kept
    const dist_t d = make_dist({ 0.5, 1.0, -0.3, 0.2, -1.0 }, false);
    std::vector<common_spec_block_dist> pd(3, view(d)), qd(3, view(d));
    const llama_token xs[3] = { 1, 0, 3 };
    std::vector<double> w, h;
    common_spec_block_weights(pd.data(), qd.data(), xs, 3, w, h);
    TCHECK(std::fabs(w[3] - 1.0) < 1e-6 && h[1] == 1.0 && h[2] == 1.0 && std::fabs(h[3] - 1.0) < 1e-6,
            "draft = target: w3 %.6f h %.3f %.3f %.6f", w[3], h[1], h[2], h[3]);

    // a draft token the target left out: its weight and everything after it is 0
    const dist_t pt = make_dist({ 0.5, 1.0, -0.3, 0.2, 2.0 }, true); // token 4 truncated
    const dist_t qt = make_dist({ 0.5, 1.0, -0.3, 0.2, 2.0 }, false);
    std::vector<common_spec_block_dist> p2(2, view(pt)), q2(2, view(qt));
    const llama_token ys[2] = { 4, 1 };
    common_spec_block_weights(p2.data(), q2.data(), ys, 2, w, h);
    TCHECK(w[1] == 0.0 && w[2] == 0.0 && h[1] == 0.0 && h[2] == 0.0, "truncated draft: w %.3f %.3f h %.3f %.3f", w[1], w[2], h[1], h[2]);
    // its correction: the residual of row 0 (w_0 = 1) keeps the target's support only
    std::vector<llama_token_data> r;
    const double s = common_spec_block_excess(1.0, view(pt), view(qt), &r);
    bool support_ok = s > 0.0;
    for (const auto & e : r) {
        support_ok = support_ok && e.id != 4 && e.p > 0.0f;
    }
    TCHECK(support_ok, "residual stays on the target's support (sum %.4f, %zu ids)", s, r.size());
}

// [TAG_V0262_BLOCKVERIFY] the exact output of the block rule on the drafts xs (drafted with probability pr), added to
// out[first][second]; q_rows: the proposals the verify reads (empty: the model's); no drafts: the target samples
static void block_out(model & md, const std::vector<llama_token> & xs, double pr, const std::vector<dist_t> & q_rows,
        std::vector<std::vector<double>> & out) {
    const int k = (int) xs.size();
    std::vector<dist_t> P(k + 1), Q(k);
    for (int i = 0; i <= k; ++i) {
        const std::vector<llama_token> prefix(xs.begin(), xs.begin() + i);
        P[i] = md.at(prefix).first;
        if (i < k) {
            Q[i] = q_rows.empty() ? md.at(prefix).second : q_rows[i];
        }
    }
    // one token out: the next step samples the second from the target
    auto add = [&](const std::vector<llama_token> & seq, double wy) {
        if (seq.size() >= 2) {
            out[seq[0]][seq[1]] += wy;
        } else {
            const dist_t p2 = md.at(seq).first;
            for (const auto & f : p2) {
                out[seq[0]][f.id] += wy*f.p;
            }
        }
    };
    if (k == 0) {
        for (const auto & e : P[0]) {
            add(std::vector<llama_token>(1, e.id), pr*e.p);
        }
        return;
    }
    std::vector<common_spec_block_dist> pd(k), qd(k);
    for (int i = 0; i < k; ++i) {
        pd[i] = view(P[i]);
        qd[i] = view(Q[i]);
    }
    std::vector<double> w, h;
    common_spec_block_weights(pd.data(), qd.data(), xs.data(), k, w, h);
    for (int t = 0; t <= k; ++t) {
        double pt = t > 0 ? h[t] : 1.0;
        for (int j = t + 1; j <= k; ++j) {
            pt *= 1.0 - h[j];
        }
        if (pt == 0.0) {
            continue;
        }
        dist_t y;
        if (t == k) {
            y = P[k];
        } else {
            std::vector<llama_token_data> r;
            const double s = common_spec_block_excess(w[t], view(P[t]), view(Q[t]), &r);
            y = s > 0.0 ? r : P[t];
        }
        double ys = 0.0;
        for (const auto & e : y) {
            ys += e.p;
        }
        for (const auto & e : y) {
            std::vector<llama_token> seq(xs.begin(), xs.begin() + t);
            seq.push_back(e.id);
            add(seq, pr*pt*e.p/ys);
        }
    }
}

// [TAG_V0262_BLOCKVERIFY] max |joint of the first two output tokens - the target's|
static double joint_err(model & md, const std::vector<std::vector<double>> & out) {
    double err = 0.0;
    const dist_t p1 = md.at(std::vector<llama_token>()).first;
    for (int a = 0; a < V; ++a) {
        const double pa = prob(p1, a);
        const dist_t p2 = pa > 0.0 ? md.at(std::vector<llama_token>(1, (llama_token) a)).first : dist_t();
        for (int b = 0; b < V; ++b) {
            err = std::max(err, std::fabs(out[a][b] - pa*prob(p2, b)));
        }
    }
    return err;
}

// [TAG_V0262_BLOCKVERIFY] every draft the drafter can make up to gamma_max tokens, verified by the block rule. Prefix
// stop: the drafter stops when its largest probability is below p_min (DFlash2's p_min since v0.26.1) or after token 2
// (a stop token); both depend on the prefix only. Drawn stop: it draws a token and stops before it when that token's
// own probability is below p_min (the p_min stop before v0.26.1): the drafts are no longer drawn from q.
static double stop_case(int gamma_max, double mix, bool truncate, uint32_t seed, double p_min, bool drawn_stop) {
    model md{ std::mt19937(seed), 1.6, mix, truncate, {} };
    std::vector<std::vector<double>> out(V, std::vector<double>(V, 0.0));
    const std::vector<dist_t> no_q;
    std::vector<std::pair<std::vector<llama_token>, double>> todo;
    todo.push_back({ std::vector<llama_token>(), 1.0 });
    while (!todo.empty()) {
        const std::pair<std::vector<llama_token>, double> cur = todo.back();
        todo.pop_back();
        const std::vector<llama_token> & xs = cur.first;
        const double pr = cur.second;
        const int i = (int) xs.size();
        if (i == gamma_max) {
            block_out(md, xs, pr, no_q, out);
            continue;
        }
        const dist_t q = md.at(xs).second;
        if (!drawn_stop) {
            double q_max = 0.0;
            for (const auto & e : q) {
                q_max = std::max(q_max, (double) e.p);
            }
            if (q_max < p_min || (i > 0 && xs.back() == 2)) {
                block_out(md, xs, pr, no_q, out);
                continue;
            }
        }
        double p_stop = 0.0;
        for (const auto & e : q) {
            if (drawn_stop && e.p < p_min) {
                p_stop += e.p; // drawn, then dropped: the draft ends before it
                continue;
            }
            std::vector<llama_token> nx = xs;
            nx.push_back(e.id);
            todo.push_back({ nx, pr*e.p });
        }
        if (p_stop > 0.0) {
            block_out(md, xs, pr*p_stop, no_q, out);
        }
    }
    return joint_err(md, out);
}

static void test_prefix_stop() {
    double worst = 0.0, worst_drawn = 0.0;
    for (int gamma = 2; gamma <= 3; ++gamma) {
        for (double mix : { 0.35, 0.9 }) {
            for (bool truncate : { false, true }) {
                for (uint32_t seed = 1; seed <= 4; ++seed) {
                    const uint32_t sd  = seed*104729 + gamma;
                    const double   err = stop_case(gamma, mix, truncate, sd, 0.45, false);
                    worst = std::max(worst, err);
                    TCHECK(err < 2e-6, "prefix stop: gamma %d mix %.2f trunc %d seed %u: joint error %.2e", gamma, mix, (int) truncate, seed, err);
                    worst_drawn = std::max(worst_drawn, stop_case(gamma, mix, truncate, sd, 0.45, true));
                }
            }
        }
    }
    // the check can see a skew: the drawn-token stop must fail it
    TCHECK(worst_drawn > 1e-2, "a stop on the drawn token's probability should skew the output (worst error %.2e)", worst_drawn);
    printf("  drafts stopped by the prefix: worst joint error %.2e (stopped by the drawn token: %.2e, biased)\n", worst, worst_drawn);
}

// [TAG_V0262_BLOCKVERIFY] proposals given as dists (common_sampler_sample_and_accept_n_block's dists overload): a token
// listed twice is one candidate with the summed probability, and a delta proposal (a lookup draft: one id with
// probability 1, here listed as 0.5 + 0.5) keeps the target's output, with drafts the target keeps, rejects or left out
static void test_dists() {
    const llama_token ids[]   = { 3, 1, 3, 4 };
    const float       probs[] = { 0.2f, 0.3f, 0.1f, 0.4f };
    std::vector<llama_token_data> q;
    common_spec_block_q_from_ids(ids, probs, 4, q);
    TCHECK(q.size() == 3 && q[0].id == 3 && q[1].id == 1 && q[2].id == 4 && std::fabs(q[0].p - 0.3f) < 1e-6f &&
           std::fabs(q[1].p - 0.3f) < 1e-6f && std::fabs(q[2].p - 0.4f) < 1e-6f, "duplicate ids: %zu candidates", q.size());

    // the model's proposal split into two entries per token: the same weights as the proposal itself
    double worst_wh = 0.0;
    for (uint32_t seed = 1; seed <= 6; ++seed) {
        model md{ std::mt19937(seed*131), 1.6, 0.5, seed % 2 == 0, {} };
        const int gamma = 3;
        std::vector<llama_token> xs;
        std::vector<dist_t> P(gamma), Q(gamma), Qs(gamma);
        for (int i = 0; i < gamma; ++i) {
            P[i] = md.at(xs).first;
            Q[i] = md.at(xs).second;
            std::vector<llama_token> sid;
            std::vector<float>       spr;
            for (int half = 0; half < 2; ++half) {
                for (const auto & e : Q[i]) {
                    sid.push_back(e.id);
                    spr.push_back(half == 0 ? 0.25f*e.p : e.p - 0.25f*e.p);
                }
            }
            common_spec_block_q_from_ids(sid.data(), spr.data(), sid.size(), Qs[i]);
            xs.push_back(Q[i][((size_t) seed + (size_t) i) % Q[i].size()].id);
        }
        std::vector<common_spec_block_dist> pd(gamma), qd(gamma), qsd(gamma);
        for (int i = 0; i < gamma; ++i) {
            pd[i]  = view(P[i]);
            qd[i]  = view(Q[i]);
            qsd[i] = view(Qs[i]);
        }
        std::vector<double> w, h, ws, hs;
        common_spec_block_weights(pd.data(), qd.data(),  xs.data(), gamma, w,  h);
        common_spec_block_weights(pd.data(), qsd.data(), xs.data(), gamma, ws, hs);
        for (int i = 0; i <= gamma; ++i) {
            worst_wh = std::max(worst_wh, std::max(std::fabs(w[i] - ws[i]), std::fabs(h[i] - hs[i])));
        }
    }
    TCHECK(worst_wh < 1e-6, "split proposal: weights differ by %.2e", worst_wh);

    // delta proposals: rows 0 and 2 draft the target's most likely token, row 1 its least likely one (the token the
    // target left out when truncated)
    double worst = 0.0;
    for (int gamma = 2; gamma <= 3; ++gamma) {
        for (bool truncate : { false, true }) {
            for (uint32_t seed = 1; seed <= 6; ++seed) {
                model md{ std::mt19937(seed*31 + gamma), 1.6, 0.5, truncate, {} };
                std::vector<llama_token> xs;
                std::vector<dist_t> Qd(gamma);
                for (int i = 0; i < gamma; ++i) {
                    const dist_t p = md.at(xs).first;
                    const auto by_p = [](const llama_token_data & a, const llama_token_data & b) { return a.p < b.p; };
                    llama_token x = std::max_element(p.begin(), p.end(), by_p)->id;
                    if (i % 2 == 1) {
                        x = truncate ? (llama_token) (V - 1) : std::min_element(p.begin(), p.end(), by_p)->id;
                    }
                    const llama_token dup[2]  = { x, x };
                    const float       half[2] = { 0.5f, 0.5f };
                    common_spec_block_q_from_ids(dup, half, 2, Qd[i]);
                    xs.push_back(x);
                }
                std::vector<std::vector<double>> out(V, std::vector<double>(V, 0.0));
                block_out(md, xs, 1.0, Qd, out);
                const double err = joint_err(md, out);
                worst = std::max(worst, err);
                TCHECK(Qd[0].size() == 1 && Qd[0][0].p == 1.0f, "delta proposal: %zu candidates", Qd[0].size());
                TCHECK(err < 2e-6, "delta proposal: gamma %d trunc %d seed %u: joint error %.2e", gamma, (int) truncate, seed, err);
            }
        }
    }
    printf("  dists with duplicate ids: weights within %.2e; delta proposals: worst joint error %.2e\n", worst_wh, worst);
}

int main() {
    printf("test-spec-block: [TAG_FN_L3_MTP_BLOCK] [TAG_V0262_BLOCKVERIFY]\n");
    test_exact();
    test_draw();
    test_edges();
    test_prefix_stop();
    test_dists();
    printf("%d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
