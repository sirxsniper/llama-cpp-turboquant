// [TAG_FN_L3_MTP_COST2] CPU unit test of the MTP draft length v2 (common/speculative-mtp-cost2.h).
//
// Part 1 checks the pieces: the median ring, the outlier drop, the slope of V on the cold experts (exact on linear
// data, the prior when C does not move), the per-row cold counts, dV, keep / more, the no-bridge mode.
// Part 2 simulates MTP steps the way draft-mtp runs them, on cost structures with a CPU-expert bridge: each verify row
// brings new cold expert-layers (content-dependent), a step costs a_R + k * C + noise, with stalls (x3-6), draft decode
// spikes (x10-40), a slow start after a prompt (x2 for 60 steps) and code / prose stretches. The policy must reach 97 %
// of the best fixed p_min rule (p_min 0.3 / 0.5 / 0.7 / 0.9 x n_max 1 / 2 / 3, chosen per structure with hindsight),
// never fall below version 1 by more than 1 %, and measure k within 20 %.
// No GPU, no model. Exit code 0 iff every check passes.

#include "../common/speculative-mtp-cost2.h"

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
    // median ring
    common_mtp_cost2::ring r;
    TCHECK(r.median() == 0.0f, "empty median");
    for (int k = 1; k <= 5; ++k) {
        r.add((float) k);
    }
    TCHECK(r.median() == 3.0f, "median of 1..5: %.1f", r.median());
    for (int k = 0; k < 100; ++k) {
        r.add(10.0f);
    }
    TCHECK(r.n == common_mtp_cost2::RING && r.median() == 10.0f, "ring keeps the last %d", common_mtp_cost2::RING);

    // exact linear data: V = a_R + k*C with a_R = 20000 + 700*(R-1), k = 45; C varies within each width
    common_mtp_cost2 c;
    c.warm = 8;
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (int s = 0; s < 600; ++s) {
        const int R = 1 + s % 3;
        uint32_t cold[common_mtp_cost2::R_MAX] = {};
        uint32_t acc = 0;
        for (int t = 0; t < R; ++t) {
            acc += (uint32_t) (t == 0 ? 140 + 60*u(rng) : 110 + 60*u(rng));
            cold[t] = acc;
        }
        const double V = 20000.0 + 700.0*(R - 1) + 45.0*cold[R - 1];
        const bool kept = c.add_step(R, cold, V);
        TCHECK(kept == (s >= 7), "linear sample %d: kept %d (the first 7 only seed the medians)", s, (int) kept);
        if (kept) {
            c.add_cycle(1.0 + (R - 1)*0.8, V + 1500.0*(R - 1), true);
        }
        if (R > 1) {
            c.add_td(1, 1500.0);
        }
    }
    c.refresh();
    TCHECK(std::fabs(c.k_ - 45.0) < 1.0, "slope %.2f vs 45", c.k_);
    TCHECK(std::fabs(c.da_[0] - 700.0) < 80.0, "row cost 1->2 %.0f vs 700", c.da_[0]);
    TCHECK(std::fabs(c.dc(1) - 170.0) < 8.0 && std::fabs(c.dc(2) - 140.0) < 8.0, "new cold per row %.1f %.1f", c.dc(1), c.dc(2));
    TCHECK(std::fabs(c.dc(5) - c.dc(3)) < 1e-9, "rows past the measured cost like the deepest measured");
    TCHECK(std::fabs(c.dV(1) - (c.da_[0] + c.k_*c.dc(2))) < 1e-6, "dV(1) = row cost + k * dC(2)");
    TCHECK(c.ready(), "ready after warm-up");
    TCHECK(c.dV(common_mtp_cost2::R_MAX) > 1e20 && !c.keep(1.0, common_mtp_cost2::R_MAX), "no row past R_MAX");
    TCHECK(!c.more(1.0, common_mtp_cost2::R_MAX, 0, 0, 1), "no decode past R_MAX");

    // outliers: a stall at its width is dropped and does not move the median
    const float med1 = c.vr[0].median();
    TCHECK(!c.add_step(1, nullptr, 10.0*med1), "a 10x step is dropped");
    TCHECK(c.vr[0].median() == med1, "median unchanged by the drop");
    // a width with few samples: tested against every width's median
    TCHECK(!c.add_step(6, nullptr, 100.0*c.vall.median()), "a 100x step at a new width is dropped");

    // a step without cold counts keeps its time for the outlier test but adds no regression sample
    const double w1 = c.w[0];
    TCHECK(c.add_step(1, nullptr, med1), "uncounted step kept");
    TCHECK(c.w[0] == w1, "no regression sample without cold counts");

    // keep: a row costs lam * dV tokens; lam is the realized rate
    const double thr = c.lam()*c.dV(1);
    TCHECK(c.keep(thr*1.05, 1) && !c.keep(thr*0.95, 1), "keep at the threshold %.3f", thr);
    // more with no confidence history: the best case
    TCHECK(c.more(1.0, 1, 0, 0, 1), "a first draft with a calibrated-0.99 best case pays");
    // more with a history of weak confidences at position 1: no decode for a weak sequence
    for (int k = 0; k < 64; ++k) {
        c.observe_conf(0, 1, 0.15f);
    }
    TCHECK(!c.more(0.3, 2, 0, 1, 1), "weak history, weak sequence: no decode");
    for (int k = 0; k < 400; ++k) {
        c.observe_conf(0, 2, 0.99f);
    }
    TCHECK(c.more(0.95, 2, 0, 2, 1) || c.lam()*c.dV(2) > 0.95*0.99, "strong history, strong sequence: decode when it can pay");

    // C that never moves: the prior slope
    common_mtp_cost2 f;
    for (int s = 0; s < 200; ++s) {
        uint32_t cold[2] = { 150, 300 };
        f.add_step(2, cold, 30000.0);
    }
    f.refresh();
    TCHECK(std::fabs(f.k_ - f.k0) < 1e-6, "constant C: k = k0 (%.2f)", f.k_);

    // no bridge: k = 0 and the widths' own means price a row
    common_mtp_cost2 n;
    n.no_cold = true;
    for (int s = 0; s < 400; ++s) {
        const int R = 1 + s % 3;
        n.add_step(R, nullptr, 20000.0 + 6000.0*(R - 1));
    }
    n.refresh();
    TCHECK(n.k_ == 0.0, "no bridge: k = 0");
    TCHECK(std::fabs(n.dV(1) - 6000.0) < 900.0 && std::fabs(n.dV(2) - 6000.0) < 900.0, "no bridge: dV %.0f %.0f vs 6000",
            n.dV(1), n.dV(2));
}

// one cost structure of the simulation (times in us)
struct sim_env {
    const char * name;
    double a1;     // rest-of-step time of a one-row step without cold experts
    double arow;   // each further row besides its cold experts
    double k;      // per cold expert-layer
    double dc1;    // new cold expert-layers of row 1 (the sampled token)
    double dc2;    // ... of each further row
    double td;     // draft decode
};

// the guard test: a policy whose prior swamps every measurement (rows look 100x too dear), so it never drafts
static bool    g_bad_prior = false;
static int64_t g_guards    = 0;

struct sim_out {
    double tps;
    double len;  // mean draft length
    double k_est;
    double len_before; // mean draft length over the 3000 steps before the pause (pause_at >= 0)
    double len_after;  // ... over the 3000 steps after it
};

// mode: 0 = fixed rule (p_min, n_max), 1 = version 1, 2 = version 2. pause_at >= 0: the bridge misses a deadline at that
// step (one step x20: the retry) and pauses 16 steps (x2.2, no cold counts), as in test/push1/g2 SHIP r1
static sim_out simulate(const sim_env & env, int mode, float p_min, int n_max, int n_steps, uint32_t seed, int pause_at = -1) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    std::normal_distribution<double> nrm(0.0, 1.0);

    common_mtp_cost  c1;
    common_mtp_cost2 c2;
    c1.warm = 24;
    if (g_bad_prior) {
        c2.k0   = 4000.0f;
        c2.kmax = 4000.0f;
        c2.wk   = 1e12f;
    }

    double tokens = 0.0, time = 0.0, lens = 0.0;
    double lb = 0.0, la = 0.0;
    double content = 1.0; // slowly varying cold-expert factor (hot set fit to the text)
    bool   code    = true;
    for (int step = 0; step < n_steps; ++step) {
        if (step % 300 == 0) {
            code = !code;
        }
        content = std::min(1.4, std::max(0.6, 0.98*content + 0.02*1.0 + 0.03*nrm(rng)));
        const double slow = step < 60 ? 2.0 : 1.0; // page-ins after the prompt

        bool use_policy = false;
        if (mode == 1) {
            use_policy = c1.policy_step();
        } else if (mode == 2) {
            use_policy = c2.policy_step();
            c2.refresh();
        }

        std::vector<float>  ps;
        std::vector<double> a_true;
        double P = 1.0, E = 1.0;
        int    R = 1, d = 0;
        double t_draft = 0.0;

        bool go = true;
        if (use_policy && mode == 1) {
            go = c1.more(1.0, c1.acc_hi(0, 0), E, R, 0, 1);
        } else if (use_policy && mode == 2) {
            go = c2.more(1.0, R, 0, 0, 1);
        }
        while (go && (int) ps.size() < n_max) {
            ++d;
            const double td = env.td*(1.0 + 0.05*nrm(rng))*(u01(rng) < 0.005 ? 10.0 + 30.0*u01(rng) : 1.0);
            t_draft += td;
            if (mode == 1) {
                c1.add_td(1, td);
            } else if (mode == 2) {
                c2.add_td(1, td);
            }
            const int j = (int) ps.size();
            // the drafter: code is confident more often; deeper positions less; its confidence runs high
            const double hi = (code ? 0.8 : 0.5)*(j == 0 ? 1.0 : 0.85);
            const float  p  = (float) (u01(rng) < hi ? 0.85 + 0.15*u01(rng) : 0.05 + 0.8*u01(rng));
            const double at = p*(j == 0 ? 0.95 : 0.88);
            if (mode == 2) {
                c2.observe_conf(0, j, p);
            }

            bool keep;
            if (use_policy && mode == 1) {
                const double a = c1.acc(0, j, p);
                keep = c1.keep(P*a, E, R, d, 1);
                if (keep) {
                    P *= a;
                    E += P;
                    R += 1;
                }
            } else if (use_policy && mode == 2) {
                const double a = c2.calib.acc(0, j, p);
                keep = c2.keep(P*a, R);
                if (keep) {
                    P *= a;
                    R += 1;
                }
            } else {
                keep = !(p < (mode == 0 ? p_min : 0.5f));
            }
            if (!keep) {
                break;
            }
            ps.push_back(p);
            a_true.push_back(at);
            if ((int) ps.size() >= n_max) {
                break;
            }
            if (use_policy && mode == 1) {
                go = c1.more(P, c1.acc_hi(0, (int) ps.size()), E, R, d, 1);
            } else if (use_policy && mode == 2) {
                go = c2.more(P, R, 0, (int) ps.size(), 1);
            }
        }

        int n_acc = 0;
        while (n_acc < (int) ps.size() && u01(rng) < a_true[n_acc]) {
            ++n_acc;
        }
        const int rows = (int) ps.size() + 1;
        uint32_t cold[common_mtp_cost2::R_MAX] = {};
        double   C = 0.0;
        for (int t = 0; t < rows; ++t) {
            const double mu = (t == 0 ? env.dc1 : env.dc2)*content;
            C += std::max(0.0, mu*(1.0 + 0.25*nrm(rng)));
            cold[t] = (uint32_t) C;
        }
        double V = (env.a1 + env.arow*(rows - 1) + env.k*C)*(1.0 + 0.04*nrm(rng))*slow;
        if (u01(rng) < 0.01) {
            V *= 3.0 + 3.0*u01(rng); // a stall
        }
        const bool paused = pause_at >= 0 && step >= pause_at && step < pause_at + 17;
        if (paused) {
            V *= step == pause_at ? 20.0 : 2.2; // the retried ubatch, then the sync fallback
        }
        if (mode == 1) {
            c1.add_step(rows, V);
            c1.observe_accept(0, ps, n_acc);
        } else if (mode == 2) {
            // the driver leaves the first c2.skip steps after a prompt out; a paused bridge counts no cold experts
            if (step >= c2.skip && c2.add_step(rows, paused ? nullptr : cold, V)) {
                c2.add_cycle(n_acc + 1.0, V + t_draft, use_policy);
            }
            c2.observe_accept(0, ps, n_acc);
        }
        if (pause_at >= 0 && step >= pause_at - 3000 && step < pause_at) {
            lb += (double) ps.size()/3000.0;
        }
        if (pause_at >= 0 && step >= pause_at + 17 && step < pause_at + 3017) {
            la += (double) ps.size()/3000.0;
        }

        tokens += n_acc + 1;
        time   += V + t_draft;
        lens   += (double) ps.size();
    }
    if (mode == 2) {
        g_guards = c2.n_guard;
    }
    return { tokens/(time*1e-6), lens/n_steps, mode == 2 ? c2.k_est() : 0.0, lb, la };
}

static void test_simulation() {
    const sim_env envs[] = {
        // file A at 32K-262K: ~3.3 cold experts per row and layer x 48 layers, ~45 us each on the critical path
        { "bridge-q4",   20000.0,  700.0,  45.0, 160.0, 135.0, 1500.0 },
        // the DMA share off: the CPU job is shorter per expert
        { "fast-cpu",    17000.0,  700.0,  25.0, 160.0, 135.0, 1500.0 },
        // a big hot set: few cold experts, rows nearly free
        { "gpu-bound",   15000.0,  500.0,  10.0,  60.0,  40.0, 1200.0 },
        // a starved hot set: rows are expensive
        { "slow-rows",   24000.0,  900.0,  80.0, 200.0, 180.0, 1800.0 },
    };
    const float pmins[] = { 0.3f, 0.5f, 0.7f, 0.9f };
    const int   nmaxs[] = { 1, 2, 3 };
    const int   n_steps = 20000;

    double len_cheap = 0.0, len_dear = 0.0;
    for (const auto & env : envs) {
        double best = 0.0;
        std::string bests;
        for (int nm : nmaxs) {
            for (float p : pmins) {
                const sim_out r = simulate(env, 0, p, nm, n_steps, 4321);
                if (r.tps > best) {
                    best  = r.tps;
                    bests = "p_min " + std::to_string(p).substr(0, 3) + " n_max " + std::to_string(nm);
                }
            }
        }
        const sim_out v1 = simulate(env, 1, 0.5f, 3, n_steps, 4321);
        const sim_out v2 = simulate(env, 2, 0.5f, 3, n_steps, 4321);
        printf("  %-10s best fixed %.1f t/s (%s); v1 %.1f (%.3f of best, mean draft %.2f); v2 %.1f (%.3f of best, "
               "mean draft %.2f, k %.1f vs %.1f)\n", env.name, best, bests.c_str(), v1.tps, v1.tps/best, v1.len,
               v2.tps, v2.tps/best, v2.len, v2.k_est, env.k);
        TCHECK(v2.tps >= 0.97*best, "%s: v2 %.1f t/s < 97%% of the best fixed rule %.1f", env.name, v2.tps, best);
        TCHECK(v2.tps >= 0.99*v1.tps, "%s: v2 %.1f t/s below v1 %.1f", env.name, v2.tps, v1.tps);
        TCHECK(std::fabs(v2.k_est - env.k) <= 0.2*env.k + 2.0, "%s: k %.1f vs %.1f", env.name, v2.k_est, env.k);
        if (std::string(env.name) == "gpu-bound") {
            len_cheap = v2.len;
        }
        if (std::string(env.name) == "slow-rows") {
            len_dear = v2.len;
        }
    }
    TCHECK(len_cheap > len_dear + 0.2, "v2 drafts longer where rows are cheap: %.2f vs %.2f", len_cheap, len_dear);

    // a bridge deadline miss: one x20 step and 16 paused steps (x2.2, no cold counts) must not shorten the drafts after it
    const sim_out p1 = simulate(envs[0], 1, 0.5f, 2, 12000, 99, 6000);
    const sim_out p2 = simulate(envs[0], 2, 0.5f, 2, 12000, 99, 6000);
    printf("  pause      v1 mean draft %.2f before, %.2f after; v2 %.2f before, %.2f after\n", p1.len_before, p1.len_after,
            p2.len_before, p2.len_after);
    TCHECK(p2.len_after >= 0.9*p2.len_before, "v2 drafts after a bridge pause %.2f vs %.2f before", p2.len_after, p2.len_before);

    // the guard: a policy that never drafts loses clearly to its own probes; the guard hands most steps to the rule
    const sim_out rule = simulate(envs[0], 0, 0.5f, 2, 12000, 7);
    g_bad_prior = true;
    const sim_out bad = simulate(envs[0], 2, 0.5f, 2, 12000, 7);
    g_bad_prior = false;
    printf("  guard      a never-drafting policy: %.1f t/s (mean draft %.2f, %lld guard events) vs the rule %.1f t/s\n",
            bad.tps, bad.len, (long long) g_guards, rule.tps);
    TCHECK(g_guards > 0, "the guard never set the never-drafting policy aside");
    TCHECK(bad.tps >= 0.85*rule.tps, "guarded bad policy %.1f t/s < 85%% of the rule %.1f", bad.tps, rule.tps);
}

int main() {
    printf("test-mtp-cost2: [TAG_FN_L3_MTP_COST2]\n");
    test_pieces();
    test_simulation();
    printf("%d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
