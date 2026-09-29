// [TAG_FN_MTP_COST] CPU unit test of the cost-aware MTP draft length (common/speculative-mtp-cost.h).
//
// Part 1 checks the pieces: probability bins, the calibration prior and its update, the union parser, the per-width
// cost estimate (monotone, shrunk to the union curve, outliers dropped) and the keep / more tests.
// Part 2 simulates MTP steps on three cost structures (hybrid with CPU experts, a GPU hot set, dense all-GPU) with a
// drafter whose confidence is not calibrated, runs the policy the way draft-mtp does, and compares its tokens per
// unit of time with the fixed p_min rule at p_min 0.3 / 0.5 / 0.7 / 0.9. The policy must reach at least 97 % of the
// best fixed threshold of each structure without being told which one it is on.
// No GPU, no model. Exit code 0 iff every check passes.

#include "../common/speculative-mtp-cost.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
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

static void test_pieces() {
    common_mtp_cost c;

    TCHECK(common_mtp_cost::bin_of(0.0f)   == 0,  "bin of 0");
    TCHECK(common_mtp_cost::bin_of(0.55f)  == 5,  "bin of 0.55");
    TCHECK(common_mtp_cost::bin_of(0.899f) == 8,  "bin of 0.899");
    TCHECK(common_mtp_cost::bin_of(0.93f)  == 9,  "bin of 0.93");
    TCHECK(common_mtp_cost::bin_of(0.97f)  == 10, "bin of 0.97");
    TCHECK(common_mtp_cost::bin_of(1.0f)   == 11, "bin of 1");

    // no data: the prior a = p (bin middle)
    TCHECK(std::fabs(c.acc(0, 0, 0.55f) - 0.55f) < 1e-6f, "prior acc %.4f", c.acc(0, 0, 0.55f));
    TCHECK(std::fabs(c.acc(1, 3, 0.99f) - 0.99f) < 1e-6f, "prior acc hi %.4f", c.acc(1, 3, 0.99f));

    // the drafter says 0.99 at position 1, the target accepts half of them: the estimate moves to ~0.5 there and
    // mostly there (position 0 of the same bin sees the pooled value)
    for (int k = 0; k < 400; ++k) {
        c.observe_accept(0, { 0.2f, 0.99f }, 1 + (k & 1)); // pos 0 always accepted, pos 1 every other time
    }
    TCHECK(std::fabs(c.acc(0, 1, 0.99f) - 0.5f) < 0.06f, "calibrated pos 1: %.3f", c.acc(0, 1, 0.99f));
    TCHECK(c.acc(0, 0, 0.2f) > 0.9f, "pos 0 at 0.2 always accepted: %.3f", c.acc(0, 0, 0.2f));
    TCHECK(std::fabs(c.acc(1, 1, 0.99f) - 0.99f) < 1e-6f, "sampled mode untouched: %.3f", c.acc(1, 1, 0.99f));
    // a token past the first rejection is not a trial
    common_mtp_cost c2;
    c2.observe_accept(0, { 0.99f, 0.99f, 0.99f }, 0);
    TCHECK(c2.ctr[0][0][11] > 0.0f && c2.ctr[0][1][11] == 0.0f, "only position 0 was tried");

    // union table
    TCHECK(c.set_union("1,1.6,2.1"), "union parse");
    TCHECK(c.uni.size() == 3 && std::fabs(c.uni[2] - 2.1f) < 1e-6f, "union values");
    TCHECK(std::fabs(c.g(4) - 1.6) < 1e-6, "union extrapolation g(4) %.4f", c.g(4));
    TCHECK(!c.set_union("2,3"), "union must start at 1");
    TCHECK(!c.set_union("1,0.5"), "union must not fall");

    // per-width cost: R = 1 and 2 measured on a 1 + 0.8*g(R) curve; R = 3, 4 follow the curve
    common_mtp_cost v;
    v.warm = 4;
    auto truth = [&](int R) { return 30000.0*(1.0 + 0.8*v.g(R)); };
    TCHECK(v.V(1) == 0.0 && !v.ready(), "empty model");
    for (int k = 0; k < 40; ++k) {
        v.add_step(1 + (k % 2), truth(1 + (k % 2)));
        v.add_td(1, 1500.0);
    }
    TCHECK(v.ready(), "ready after warm-up");
    TCHECK(std::fabs(v.beta() - 0.8) < 0.12, "learned beta %.3f", v.beta());
    for (int R = 1; R <= 4; ++R) {
        TCHECK(std::fabs(v.V(R)/truth(R) - 1.0) < 0.08, "V(%d) %.0f vs %.0f", R, v.V(R), truth(R));
    }
    for (int R = 1; R < 10; ++R) {
        TCHECK(v.V(R + 1) >= v.V(R), "V monotone at %d", R);
    }
    TCHECK(!v.add_step(1, 10.0*truth(1)), "outlier dropped");
    TCHECK(std::fabs(v.V(1)/truth(1) - 1.0) < 0.08, "V(1) unchanged by the outlier");
    TCHECK(v.V(common_mtp_cost::R_MAX + 1) > 1e20, "past R_MAX is never chosen");
    // [TAG_FN_MTP_COST] more drafting sequences than R_MAX rows (or R_MAX rows already): no row can be priced, so nothing
    // is kept (before the guard, R > R_MAX priced every row at 0 and kept every token)
    for (int R : { common_mtp_cost::R_MAX, common_mtp_cost::R_MAX + 1, common_mtp_cost::R_MAX + 8 }) {
        TCHECK(!v.keep(1.0, 1.0, R, 1, 1),       "keep at R = %d must refuse", R);
        TCHECK(!v.more(1.0, 1.0, 1.0, R, 0, 1),  "more at R = %d must refuse", R);
    }

    // keep: the second verify row costs 0.8*0.55 = 44 % of a one-row step here, the third ~20 % of a two-row step
    const double E1 = 1.0 + 0.95; // one kept token at 0.95
    TCHECK(v.keep(0.95, 1.0, 1, 1, 1),           "first token at 0.95 pays");
    TCHECK(!v.keep(0.30, 1.0, 1, 1, 1),          "first token at 0.30 does not");
    TCHECK(v.keep(0.95*0.6, E1, 2, 2, 1),        "second token at 0.6 pays for the third row");
    TCHECK(!v.keep(0.95*0.3, E1, 2, 2, 1),       "second token at 0.3 does not");
    TCHECK(!v.more(0.95*0.3, 0.99, E1, 2, 2, 1), "no further decode after a weak token");
    TCHECK(v.more(0.95, 0.99, E1, 2, 1, 1),      "a strong first token is worth another decode");

    // dense: rows nearly free, a second token at 0.5 pays
    common_mtp_cost d;
    d.warm = 4;
    for (int k = 0; k < 40; ++k) {
        const int R = 1 + (k % 3);
        d.add_step(R, 12000.0*(1.0 + 0.03*(R - 1)));
        d.add_td(1, 900.0);
    }
    TCHECK(d.keep(0.95*0.5, E1, 2, 2, 1), "dense: second token at 0.5 pays");
    TCHECK(d.more(0.95, 0.99, E1, 2, 1, 1),  "dense: drafting on after a strong token");
}

// one simulated cost structure: rest-of-step time per verify width (us) and the draft decode time (us)
struct sim_cost {
    const char * name;
    std::vector<double> V; // R = 1..
    double td;
};

struct sim_result {
    double tps;     // tokens per second
    double len;     // mean draft length
};

// n_steps MTP steps. p_min < 0: the policy (warm-up and probes on the p_min 0.5 rule, as in draft-mtp)
static sim_result simulate(const sim_cost & sc, float p_min, int n_max, int n_steps, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u01(0.0, 1.0);

    common_mtp_cost c;
    c.warm = 24;
    const bool policy_on = p_min < 0.0f;
    const float rule_p = policy_on ? 0.5f : p_min;

    double tokens = 0.0, time = 0.0, lens = 0.0;
    for (int step = 0; step < n_steps; ++step) {
        const bool use_policy = policy_on && c.policy_step();

        std::vector<float> ps;
        std::vector<double> a_true;
        double E = 1.0, P = 1.0;
        int R = 1, d = 0;

        bool go = !use_policy || c.more(1.0, c.acc_hi(0, 0), E, R, 0, 1);
        while (go && (int) ps.size() < n_max) {
            ++d;
            const int j = (int) ps.size();
            // the drafter: confident most of the time, the deeper positions less so; its confidence runs high
            const double hi = j == 0 ? 0.75 : 0.6;
            const float  p  = (float) (u01(rng) < hi ? 0.85 + 0.15*u01(rng) : 0.05 + 0.8*u01(rng));
            const double at = p*(j == 0 ? 0.95 : 0.85);

            bool keep;
            if (use_policy) {
                const double a = c.acc(0, j, p);
                keep = c.keep(P*a, E, R, d, 1);
                if (keep) {
                    P *= a;
                    E += P;
                    R += 1;
                }
            } else {
                keep = p >= rule_p;
            }
            if (!keep) {
                break;
            }
            ps.push_back(p);
            a_true.push_back(at);
            if (use_policy && (int) ps.size() < n_max) {
                go = c.more(P, c.acc_hi(0, (int) ps.size()), E, R, d, 1);
            }
        }

        int n_acc = 0;
        while (n_acc < (int) ps.size() && u01(rng) < a_true[n_acc]) {
            ++n_acc;
        }
        const int rows = (int) ps.size() + 1;
        const double v_us = sc.V[std::min<size_t>(rows, sc.V.size()) - 1]*(0.9 + 0.2*u01(rng));
        for (int k = 0; k < d; ++k) {
            c.add_td(1, sc.td*(0.9 + 0.2*u01(rng)));
        }
        c.add_step(rows, v_us);
        c.observe_accept(0, ps, n_acc);

        tokens += n_acc + 1;
        time   += v_us + d*sc.td;
        lens   += (double) ps.size();
    }
    return { tokens/(time*1e-6), lens/n_steps };
}

static void test_simulation() {
    const sim_cost regimes[] = {
        // Flash-Next with whole CPU expert layers: a verify row adds most of a token's expert bytes (T(4)/T(1) 2.42)
        { "hybrid-cpu",  { 30000, 44850, 57000, 68340, 79000 }, 1500 },
        // a GPU hot set serving ~half the expert bytes: rows cost less
        { "hot-set",     { 22000, 26500, 30200, 33600, 36800 }, 1200 },
        // dense all-GPU target: rows nearly free
        { "dense-gpu",   { 12000, 12400, 12800, 13200, 13600 },  900 },
    };
    const float fixed[] = { 0.3f, 0.5f, 0.7f, 0.9f };
    const int   n_steps = 30000;

    double len_hybrid = 0.0, len_dense = 0.0;
    for (const auto & sc : regimes) {
        double best = 0.0;
        float  best_p = 0.0f;
        std::string line;
        for (float p : fixed) {
            const sim_result r = simulate(sc, p, 3, n_steps, 1234);
            line += std::to_string(p).substr(0, 3) + ":" + std::to_string((int) r.tps) + " ";
            if (r.tps > best) {
                best   = r.tps;
                best_p = p;
            }
        }
        const sim_result pol = simulate(sc, -1.0f, 3, n_steps, 1234);
        printf("  %-10s fixed p_min {%s} best %.1f at %.1f, policy %.1f t/s (%.3f of best), mean draft %.2f\n",
                sc.name, line.c_str(), best, best_p, pol.tps, pol.tps/best, pol.len);
        TCHECK(pol.tps >= 0.97*best, "%s: policy %.1f t/s < 97%% of the best fixed rule %.1f", sc.name, pol.tps, best);
        if (std::string(sc.name) == "hybrid-cpu") {
            len_hybrid = pol.len;
        }
        if (std::string(sc.name) == "dense-gpu") {
            len_dense = pol.len;
        }
    }
    TCHECK(len_dense > len_hybrid + 0.3, "the policy drafts longer where rows are cheap: %.2f vs %.2f", len_dense, len_hybrid);
}

int main() {
    printf("test-mtp-cost: [TAG_FN_MTP_COST]\n");
    test_pieces();
    test_simulation();
    printf("%d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
