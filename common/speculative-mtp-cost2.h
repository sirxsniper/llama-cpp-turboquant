#pragma once

// [TAG_FN_L3_MTP_COST2] Draft length from measured costs, version 2, for the MTP driver (common/speculative.cpp,
// draft-mtp). Switch: SPEC_MTP_COST=2, qwen4exp only (other models keep version 1 there). SPEC_MTP_COST=1 is version 1
// (speculative-mtp-cost.h) unchanged, 0 / unset the fixed n_max / p_min rule.
//
// Version 1 priced a verify row with a curve through the mean step time of its two most used widths and an EMA of the
// draft decode. Single slow steps and its own choices moved both: on file A it cut the drafts to 1.26-1.30 tokens per
// step for whole answers (test/r2/ab4 PF0S_NODMA round 0: 22.9 ms/step, the same arm's next round kept 2.33 tokens at
// 36.0 ms/step, 18 % more tokens per second) and in test/r1/bs1 it took a third draft that did not pay.
//
// Version 2 measures what a verify row costs where it costs. The MoE bridge counts, per verify step, the distinct
// experts of every token prefix that are not in the VRAM hot set (llama_moe_bridge_last_ext); they run on the CPU (or
// the DMA share) and the device waits for them. With C(R) the cold expert-layers of a step with R verify rows:
//   V(R) = a_R + k * C(R)              rest-of-step time (verify decode, MTP catch-up, sampling, accept)
//   k                                  pooled within-width slope of V on C: C changes from step to step at the same
//                                      width (content, hot set), so k is measured whatever widths the policy picks
//   a_R                                per-width intercept; a_{R+1} - a_R is the row cost besides its cold experts
//   dC(t)                              new cold expert-layers of verify row t, from the prefix counts
//   dV(R) = (a_{R+1} - a_R) + k * dC(R + 1)
// The decisions use the realized rate lam (tokens per us over the recent steps), the Dinkelbach form of "more tokens per
// second": a step maximizes E - lam * T, which raises lam until no length choice can raise it further.
//   keep a draft token       P * a >= lam * dV(R)                             its expected tokens pay for its row
//   draft once more          E_p[max(0, P * a(p) - lam * dV(R))] >= lam * t_d  p: the drafter's confidence at that
//                                                                             position, from its own history
// a(p) is the acceptance calibrated per drafter confidence bin and draft position (version 1's calibration). Medians of
// recent steps drop stalls (page-ins, a prompt between steps); the draft decode time is a median too. A probe step (the
// p_min rule) every N policy steps keeps the longer widths measured. Exact: only the draft length changes, the target
// verifies every draft. Pure arithmetic, no llama calls: tests/test-mtp-cost2.cpp drives it with simulated steps.

#include "speculative-mtp-cost.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>

struct common_mtp_cost2 {
    static constexpr int R_MAX = 8;  // verify widths tracked (R = 1..R_MAX); wider steps are never chosen
    static constexpr int RING  = 31; // samples per median
    static constexpr int N_TD  = 4;  // draft batch widths with their own decode time (1, 2, 3, 4+)
    static constexpr int N_POS = common_mtp_cost::N_POS;
    static constexpr int N_BIN = common_mtp_cost::N_BIN;

    // configuration, SPEC_MTP_COST2_* (init_env)
    int   warm    = 32;      // _WARM: steps with cold counts before the policy acts; the p_min rule until then
    int   probe   = 32;      // _PROBE: one p_min-rule step after this many policy steps (0 = never)
    float rdecay  = 0.995f;  // _RDECAY: weight of a width's past regression samples per new sample of that width
    float age     = 0.999f;  // _AGE: weight of every width's samples per step (unused widths fade)
    float ldecay  = 0.99f;   // _LDECAY: weight of the past realized rate per step
    float dcdecay = 0.995f;  // _DCDECAY: weight of the past per-row cold counts per sample
    float k0      = 40.0f;   // _K0: prior us per cold expert-layer
    float wk      = 16.0f;   // _WK: the prior slope is worth this many samples' variance of C
    float kmax    = 400.0f;  // _KMAX
    float row0    = 1000.0f; // _ROW0: prior us of one more verify row besides its cold experts
    float wrow    = 8.0f;    // _WROW: the prior row cost is worth this many samples of the rarer width
    float dc0     = 150.0f;  // _DC0: prior new cold expert-layers of a verify row
    float outlier = 2.5f;    // _OUTLIER: a step above this multiple of its width's median is dropped
    int   hmin    = 16;      // _HMIN: drafter confidences seen at a position before its histogram decides
    float hdecay  = 0.995f;  // _HDECAY
    int   skip    = 16;      // _SKIP: steps after a prompt that are not samples (hot-set refill, page-ins)
    float gmargin = 0.85f;   // _GUARD: the policy's realized rate below this x the probes' turns the policy off ...
    float gdecay  = 0.999f;  // _GUARD_DECAY: weight of the past per step of both rates the guard compares
    int   gsteps  = 256;     // _GUARD_STEPS: ... for this many steps (the p_min rule meanwhile)
    int   gmin    = 24;      // _GUARD_MIN: probe steps measured before the guard compares
    int   log     = 0;       // _LOG (or SPEC_MTP_COST_LOG): 1 = a summary every 256 policy steps and at the end

    bool no_cold = false;    // no cold counts at all (no bridge): k = 0, the widths' own means price the rows

    common_mtp_cost calib;   // acceptance calibration per mode / position / confidence bin (version 1's)

    // median of the last RING samples
    struct ring {
        float v[RING] = {};
        int   n = 0;
        int   i = 0;

        void add(float x) {
            v[i] = x;
            i    = (i + 1) % RING;
            n    = std::min(n + 1, RING);
        }

        float median() const {
            if (n == 0) {
                return 0.0f;
            }
            float t[RING];
            std::copy(v, v + n, t);
            std::nth_element(t, t + n/2, t + n);
            return t[n/2];
        }
    };

    // per verify width R (index R - 1): the step times (median, outlier test) and the decayed sums of V on C
    ring   vr[R_MAX];
    ring   vall; // every width: the outlier test of a width with few samples
    double w  [R_MAX] = {};
    double sc [R_MAX] = {};
    double sv [R_MAX] = {};
    double scc[R_MAX] = {};
    double scv[R_MAX] = {};
    // new cold expert-layers of verify row t (index t - 1): decayed sum and count
    double dcs[R_MAX] = {};
    double dcn[R_MAX] = {};
    // draft decode time per draft batch width
    ring   tdr[N_TD];
    // realized tokens and us of whole steps (draft() to draft()), decayed: all steps (the rate), and policy / rule steps
    // apart over the same longer window (the guard)
    double lt = 0.0, lu = 0.0;
    double lt_pol = 0.0, lu_pol = 0.0, lt_rule = 0.0, lu_rule = 0.0;
    // drafter confidence per mode, draft position and bin, decayed counts
    float  ph[2][N_POS][N_BIN] = {};

    int64_t n_valid  = 0; // step samples kept
    int64_t n_cold   = 0; // ... with cold counts
    // per width: kept steps without cold counts (a paused bridge, a graph wider than the bridge takes) and with them
    int64_t n_wo[R_MAX] = {};
    int64_t n_wc[R_MAX] = {};
    int64_t n_drop   = 0; // step samples dropped as outliers
    int64_t n_cycle  = 0; // realized-rate samples
    int64_t n_policy = 0; // policy steps since the last probe
    int64_t n_rule_c = 0; // realized-rate samples of rule steps since the last guard
    int64_t n_pol_c  = 0; // ... of policy steps
    int64_t n_guard  = 0; // times the guard turned the policy off
    int     guard_left = 0;

    // refreshed by refresh() once per draft() call
    double k_   = 0.0;
    double lam_ = 0.0;
    double da_[R_MAX] = {}; // row cost besides cold experts, R -> R + 1 (index R - 1)
    double dv_[R_MAX] = {}; // marginal verify cost dV(R) (index R - 1)

    static float env_f(const char * name, float def) {
        const char * e = getenv(name);
        return (e && e[0]) ? (float) atof(e) : def;
    }

    void init_env() {
        calib.init_env(); // SPEC_MTP_COST_CDECAY / _CW0 / _CW1 for the calibration, as in version 1
        warm    = std::max(4, (int) env_f("SPEC_MTP_COST2_WARM", (float) warm));
        probe   = std::max(0, (int) env_f("SPEC_MTP_COST2_PROBE", (float) probe));
        rdecay  = std::min(0.9999f, std::max(0.9f, env_f("SPEC_MTP_COST2_RDECAY", rdecay)));
        age     = std::min(1.0f, std::max(0.99f, env_f("SPEC_MTP_COST2_AGE", age)));
        ldecay  = std::min(0.9999f, std::max(0.9f, env_f("SPEC_MTP_COST2_LDECAY", ldecay)));
        dcdecay = std::min(0.9999f, std::max(0.9f, env_f("SPEC_MTP_COST2_DCDECAY", dcdecay)));
        k0      = std::max(0.0f, env_f("SPEC_MTP_COST2_K0", k0));
        wk      = std::max(0.01f, env_f("SPEC_MTP_COST2_WK", wk));
        kmax    = std::max(k0, env_f("SPEC_MTP_COST2_KMAX", kmax));
        row0    = std::max(0.0f, env_f("SPEC_MTP_COST2_ROW0", row0));
        wrow    = std::max(0.01f, env_f("SPEC_MTP_COST2_WROW", wrow));
        dc0     = std::max(0.0f, env_f("SPEC_MTP_COST2_DC0", dc0));
        outlier = std::max(1.5f, env_f("SPEC_MTP_COST2_OUTLIER", outlier));
        hmin    = std::max(1, (int) env_f("SPEC_MTP_COST2_HMIN", (float) hmin));
        hdecay  = std::min(0.9999f, std::max(0.9f, env_f("SPEC_MTP_COST2_HDECAY", hdecay)));
        skip    = std::max(0, (int) env_f("SPEC_MTP_COST2_SKIP", (float) skip));
        gmargin = std::min(1.0f, std::max(0.0f, env_f("SPEC_MTP_COST2_GUARD", gmargin)));
        gsteps  = std::max(1, (int) env_f("SPEC_MTP_COST2_GUARD_STEPS", (float) gsteps));
        gmin    = std::max(4, (int) env_f("SPEC_MTP_COST2_GUARD_MIN", (float) gmin));
        gdecay  = std::min(0.99999f, std::max(0.9f, env_f("SPEC_MTP_COST2_GUARD_DECAY", gdecay)));
        log     = std::max(0, (int) env_f("SPEC_MTP_COST2_LOG", env_f("SPEC_MTP_COST_LOG", 0.0f)));
    }

    // one step: R verify rows, us from the end of a draft() call to the start of the next; cold[t] = C(t + 1), the
    // cold expert-layers of verify rows 0..t (cumulative), or nullptr when the step has no counts (a paused bridge).
    // false: dropped as an outlier (a stall, a page-in burst), or one of the process's first 8 steps, which only seed
    // the medians - the caller then leaves the realized rate alone too.
    bool add_step(int R, const uint32_t * cold, double us) {
        if (R < 1 || R > R_MAX || !(us > 0.0)) {
            return false;
        }
        const ring & rr = vr[R - 1];
        if ((rr.n >= 8 && us > outlier*rr.median()) || (rr.n < 8 && vall.n >= 8 && us > 2.0*outlier*vall.median())) {
            n_drop++;
            return false;
        }
        vr[R - 1].add((float) us);
        vall.add((float) us);
        if (vall.n < 8) {
            return false; // the first steps of the process only seed the medians (no outlier test was possible)
        }
        n_valid++;

        if (cold == nullptr && !no_cold) {
            n_wo[R - 1]++;
            return true; // the step's time is known but not its cold experts: no regression sample
        }
        n_wc[R - 1]++;
        for (int r = 0; r < R_MAX; ++r) {
            w[r] *= age; sc[r] *= age; sv[r] *= age; scc[r] *= age; scv[r] *= age;
        }
        const double C = cold ? (double) cold[R - 1] : 0.0;
        const int    i = R - 1;
        w  [i] = w  [i]*rdecay + 1.0;
        sc [i] = sc [i]*rdecay + C;
        sv [i] = sv [i]*rdecay + us;
        scc[i] = scc[i]*rdecay + C*C;
        scv[i] = scv[i]*rdecay + C*us;
        if (cold) {
            uint32_t prev = 0;
            for (int t = 0; t < R; ++t) {
                const double d = cold[t] >= prev ? (double) (cold[t] - prev) : 0.0;
                prev   = std::max(prev, cold[t]);
                dcs[t] = dcs[t]*dcdecay + d;
                dcn[t] = dcn[t]*dcdecay + 1.0;
            }
        }
        n_cold++;
        return true;
    }

    // one whole step (draft() to draft()) of a kept add_step: tokens it produced, its us, and who chose its length
    void add_cycle(double tokens, double us, bool policy) {
        if (!(us > 0.0) || !(tokens > 0.0)) {
            return;
        }
        lt = lt*ldecay + tokens;
        lu = lu*ldecay + us;
        lt_pol *= gdecay; lu_pol *= gdecay; lt_rule *= gdecay; lu_rule *= gdecay;
        if (policy) {
            lt_pol += tokens;
            lu_pol += us;
            n_pol_c++;
        } else {
            lt_rule += tokens;
            lu_rule += us;
            n_rule_c++;
        }
        n_cycle++;
        // the guard: probes run the p_min rule at regular steps, so their realized rate is the rule's on the same
        // text; a policy that does clearly worse is set aside for a while (and measured afresh after it)
        if (gmargin > 0.0f && guard_left == 0 && n_rule_c >= gmin && n_pol_c >= 4*gmin && lu_pol > 0.0 &&
                lu_rule > 0.0 && lt_pol/lu_pol < gmargin*(lt_rule/lu_rule)) {
            guard_left = gsteps;
            n_guard++;
            lt_pol = lu_pol = 0.0;
            n_pol_c  = 0;
            n_rule_c = 0;
        }
    }

    void add_td(int n_rows, double us) {
        if (!(us > 0.0)) {
            return;
        }
        tdr[std::min(N_TD, std::max(1, n_rows)) - 1].add((float) us);
    }

    // the drafter's confidence p at draft position pos (every drafted position, kept or not)
    void observe_conf(int mode, int pos, float p) {
        mode = mode ? 1 : 0;
        pos  = std::min(N_POS - 1, std::max(0, pos));
        const int b = common_mtp_cost::bin_of(p);
        for (int k = 0; k < N_BIN; ++k) {
            ph[mode][pos][k] *= hdecay;
        }
        ph[mode][pos][b] += 1.0f;
    }

    void observe_accept(int mode, const std::vector<float> & ps, int n_acc) {
        calib.observe_accept(mode, ps, n_acc);
    }

    double t_d(int n_rows) const {
        const int i = std::min(N_TD, std::max(1, n_rows)) - 1;
        if (tdr[i].n > 0) {
            return tdr[i].median();
        }
        for (int j = 0; j < N_TD; ++j) { // an unmeasured width borrows any measured one
            if (tdr[j].n > 0) {
                return tdr[j].median();
            }
        }
        return 0.0;
    }

    double lam() const {
        return lu > 0.0 ? lt/lu : 0.0;
    }

    // pooled within-width slope of V on C, shrunk towards k0
    double k_est() const {
        if (no_cold) {
            return 0.0;
        }
        double sxx = 0.0, sxy = 0.0, n = 0.0, cs = 0.0;
        for (int r = 0; r < R_MAX; ++r) {
            if (w[r] > 1e-9) {
                sxx += scc[r] - sc[r]*sc[r]/w[r];
                sxy += scv[r] - sc[r]*sv[r]/w[r];
                n   += w[r];
                cs  += sc[r];
            }
        }
        if (n < 2.0) {
            return k0;
        }
        const double cbar = cs/n;
        // the prior's weight: wk samples of the observed within-width variance, but at least of a 3 % spread of C
        const double vc  = std::max(std::max(0.0, sxx)/n, (0.03*cbar)*(0.03*cbar) + 1.0);
        const double den = std::max(0.0, sxx) + wk*vc;
        return std::min((double) kmax, std::max(0.0, (sxy + wk*vc*k0)/den));
    }

    // intercept of width R (with the slope k)
    double a_of(int R, double k) const {
        const int i = R - 1;
        return w[i] > 1e-9 ? (sv[i] - k*sc[i])/w[i] : 0.0;
    }

    // new cold expert-layers of verify row t (1-based); rows past the measured ones cost what the deepest measured did
    double dc(int t) const {
        t = std::max(1, std::min(R_MAX, t));
        for (int u = t; u >= 1; --u) {
            if (dcn[u - 1] >= 2.0) {
                return dcs[u - 1]/dcn[u - 1];
            }
        }
        return dc0;
    }

    // recompute k, lam, the row costs and dV once per draft() call
    void refresh() {
        k_   = k_est();
        lam_ = lam();
        bool   have[R_MAX] = {};
        double obs [R_MAX] = {};
        for (int R = 1; R < R_MAX; ++R) {
            const double wa = w[R - 1], wb = w[R];
            if (wa >= 4.0 && wb >= 4.0) {
                const double o  = a_of(R + 1, k_) - a_of(R, k_);
                const double nn = std::min(wa, wb);
                obs [R - 1] = std::max(0.0, (o*nn + (double) row0*wrow)/(nn + wrow));
                have[R - 1] = true;
            }
        }
        for (int R = 1; R < R_MAX; ++R) {
            double v = row0;
            bool found = false;
            for (int u = R; u >= 1 && !found; --u) { // the nearest measured row below, else above
                if (have[u - 1]) {
                    v = obs[u - 1];
                    found = true;
                }
            }
            for (int u = R + 1; u < R_MAX && !found; ++u) {
                if (have[u - 1]) {
                    v = obs[u - 1];
                    found = true;
                }
            }
            da_[R - 1] = v;
            dv_[R - 1] = v + k_*dc(R + 1);
        }
        da_[R_MAX - 1] = da_[R_MAX - 2];
        // a width that mostly runs without cold counts (with several streams a verify ubatch can be wider than the bridge
        // takes, and such a graph runs the plain CPU split) is priced by the measured medians at least
        for (int R = 1; R < R_MAX; ++R) {
            if (n_wo[R] > n_wc[R] && vr[R].n >= 8 && vr[R - 1].n >= 8) {
                dv_[R - 1] = std::max(dv_[R - 1], (double) vr[R].median() - (double) vr[R - 1].median());
            }
        }
        dv_[R_MAX - 1] = 1e30; // no row past R_MAX
    }

    double dV(int R) const {
        return R >= 1 && R <= R_MAX ? dv_[R - 1] : 1e30;
    }

    // enough measured for the policy: steps with cold counts (or plain steps without a bridge), whole steps for the
    // rate, and a draft decode time
    bool ready() const {
        return n_cold >= warm && n_cycle >= warm/2 && lu > 0.0 && t_d(1) > 0.0;
    }

    // a probe step (the p_min rule) every `probe` policy steps; none while the guard holds the policy off
    bool policy_step() {
        if (!ready()) {
            return false;
        }
        if (guard_left > 0) {
            guard_left--;
            return false;
        }
        if (probe > 0 && ++n_policy > probe) {
            n_policy = 0;
            return false;
        }
        return true;
    }

    // keep a drafted token: p_new = P_prev * a, the probability that it and every earlier token of its sequence are
    // accepted; R verify rows so far
    bool keep(double p_new, int R) const {
        if (R < 1 || R + 1 > R_MAX) {
            return false;
        }
        return p_new >= lam_*dV(R);
    }

    // decode one more draft token for a sequence whose kept tokens are all accepted with probability P, its next draft
    // position pos; R verify rows so far, n sequences in the draft batch (they share the decode)
    bool more(double P, int R, int mode, int pos, int n) const {
        if (R < 1 || R + 1 > R_MAX) {
            return false;
        }
        mode = mode ? 1 : 0;
        const int    pc   = std::min(N_POS - 1, std::max(0, pos));
        const double thr  = lam_*dV(R);
        const double cost = lam_*t_d(n)/std::max(1, n);
        double h = 0.0;
        for (int b = 0; b < N_BIN; ++b) {
            h += ph[mode][pc][b];
        }
        if (h < hmin) {
            return P*calib.acc_hi(mode, pos) - thr >= cost; // too few confidences seen here: the best case
        }
        double ev = 0.0;
        for (int b = 0; b < N_BIN; ++b) {
            if (ph[mode][pc][b] > 0.0f) {
                ev += ph[mode][pc][b]/h*std::max(0.0, P*calib.acc(mode, pos, common_mtp_cost::bin_mid(b)) - thr);
            }
        }
        return ev >= cost;
    }

    // median step time of width R, 0 when unmeasured (for the log)
    double v_med(int R) const {
        return R >= 1 && R <= R_MAX ? vr[R - 1].median() : 0.0;
    }
};
