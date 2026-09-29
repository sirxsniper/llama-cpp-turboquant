#pragma once

// [TAG_FN_MTP_COST] Cost-aware draft length for the MTP driver (common/speculative.cpp, draft-mtp).
//
// Switch: SPEC_MTP_COST=1 (default off; off = the fixed n_max / p_min rule, bit for bit).
//
// The fixed rule keeps a draft token when the drafter's top probability is at least p_min. On a hybrid MoE target
// (Flash-Next with CPU experts) one more verify row is not cheap: the rows of a verify batch touch the UNION of their
// routed experts, so a 3-row step reads ~2-3x the expert bytes of a 1-row step. On a dense all-GPU target the same row
// costs a few percent. The best threshold therefore depends on the build, the placement, the hot set and the depth.
//
// This policy measures the real cost and keeps a draft token only when it raises the expected tokens per unit of time:
//     E = 1 + sum_j P_j            expected tokens of the step, P_j = a_1 * ... * a_j (a = calibrated acceptance)
//     T = V(R) + d * t_d           step time: V(R) the rest of a step with R verify rows, d draft decodes of t_d each
//   keep the next token (position j, acceptance a) if   P_{j-1} * a * T(R, d)  >=  E * (V(R + 1) - V(R))
//   draft once more only if even a best-case token could pay for its decode and its row:
//                                                       P_j * a_hi * T(R, d)   >=  E * (V(R + 1) - V(R) + t_d)
// V(R) is measured per verify width (wall time from the end of one draft() call to the start of the next: verify
// decode, MTP catch-up, sampling, accept), shrunk towards a curve with the expert-union shape
//     V(R) ~ V_ref * (1 + beta * g(R)) / (1 + beta * g(R_ref)),   g(R) = U(R)/U(1) - 1
// U(R)/U(1) is the expert-union growth of R consecutive tokens (SPEC_MTP_COST_UNION, default from the 2026-09-10
// Flash-Next measurement T(4)/T(1) = 2.42; route_sim.py prints it per trace as U(k)). beta is learned from the two most
// measured widths, starting from SPEC_MTP_COST_ROW. a is the acceptance calibrated per drafter probability bin and draft
// position from the verify results (decayed counts, prior a = p). Exact: only the draft length changes, the target
// verifies every draft.
//
// Pure arithmetic, no llama calls: tests/test-mtp-cost.cpp drives it with synthetic costs and acceptances.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

struct common_mtp_cost {
    static constexpr int R_MAX = 16; // verify widths tracked (R = 1..R_MAX)
    static constexpr int N_POS = 4;  // draft positions with their own calibration, deeper ones share the last
    static constexpr int N_BIN = 12; // drafter probability bins
    static constexpr int N_TD  = 4;  // draft batch widths with their own decode time (1, 2, 3, 4+)

    // configuration, SPEC_MTP_COST_* (init_env)
    int   warm    = 24;     // _WARM: valid step samples before the policy acts; the p_min rule until then
    int   probe   = 48;     // _PROBE: one p_min-rule step after this many policy steps (keeps other widths measured; 0 = never)
    float decay   = 0.97f;  // _DECAY: weight of the past of one width per sample of that width
    float age     = 0.998f; // _AGE: weight of every width's count per step sample (unused widths fade to the curve)
    float w_fit   = 3.0f;   // _WFIT: the union curve is worth this many samples of one width
    float row0    = 0.15f;  // _ROW: prior cost of the second verify row relative to a one-row step
    float w_beta  = 2.0f;   // _WBETA: the prior row cost is worth this many samples of the rarer width
    float cdecay  = 0.995f; // _CDECAY: weight of the past per calibration trial
    float cw0     = 4.0f;   // _CW0: the prior a = p is worth this many trials of a bin
    float cw1     = 8.0f;   // _CW1: the pooled bin is worth this many trials of one position
    float outlier = 3.0f;   // _OUTLIER: step samples above this multiple of the estimate are dropped
    int   log     = 0;      // _LOG: 1 = a summary every 256 policy steps, 2 = also one line per step
    std::vector<float> uni = { 1.0f, 1.55f, 2.0f, 2.42f }; // _UNION: U(R)/U(1) for R = 1, 2, ...

    // step time per verify width, index R - 1: decayed sum (us) and count
    double vs[R_MAX] = {};
    double vn[R_MAX] = {};
    // draft decode time per draft batch width, EMA (us) and count
    double td [N_TD] = {};
    double tdn[N_TD] = {};
    // calibration per mode (0 = greedy, 1 = sampled), position and bin: decayed trials and accepted; pooled per bin
    float ctr[2][N_POS][N_BIN] = {};
    float cok[2][N_POS][N_BIN] = {};
    float ptr[2][N_BIN] = {};
    float pok[2][N_BIN] = {};

    int64_t n_valid  = 0; // step samples taken
    int64_t n_policy = 0; // policy steps since the last probe step

    static float env_f(const char * name, float def) {
        const char * e = getenv(name);
        return (e && e[0]) ? (float) atof(e) : def;
    }

    void init_env() {
        warm    = std::max(2, (int) env_f("SPEC_MTP_COST_WARM", (float) warm));
        probe   = std::max(0, (int) env_f("SPEC_MTP_COST_PROBE", (float) probe));
        decay   = std::min(0.999f, std::max(0.0f, env_f("SPEC_MTP_COST_DECAY", decay)));
        age     = std::min(1.0f, std::max(0.9f, env_f("SPEC_MTP_COST_AGE", age)));
        w_fit   = std::max(0.01f, env_f("SPEC_MTP_COST_WFIT", w_fit));
        row0    = std::min(4.0f, std::max(0.0f, env_f("SPEC_MTP_COST_ROW", row0)));
        w_beta  = std::max(0.01f, env_f("SPEC_MTP_COST_WBETA", w_beta));
        cdecay  = std::min(0.9999f, std::max(0.0f, env_f("SPEC_MTP_COST_CDECAY", cdecay)));
        cw0     = std::max(0.01f, env_f("SPEC_MTP_COST_CW0", cw0));
        cw1     = std::max(0.01f, env_f("SPEC_MTP_COST_CW1", cw1));
        outlier = std::max(1.5f, env_f("SPEC_MTP_COST_OUTLIER", outlier));
        log     = std::max(0, (int) env_f("SPEC_MTP_COST_LOG", 0.0f));
        if (const char * e = getenv("SPEC_MTP_COST_UNION"); e && e[0]) {
            set_union(e);
        }
    }

    // "1,1.6,2.1": U(R)/U(1) for R = 1, 2, ...; must start at 1 and not fall
    bool set_union(const std::string & s) {
        std::vector<float> v;
        size_t pos = 0;
        while (pos <= s.size()) {
            const size_t end = s.find(',', pos);
            const std::string t = s.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
            if (!t.empty()) {
                v.push_back((float) atof(t.c_str()));
            }
            if (end == std::string::npos) {
                break;
            }
            pos = end + 1;
        }
        if (v.empty() || std::fabs(v[0] - 1.0f) > 1e-3f) {
            return false;
        }
        for (size_t i = 1; i < v.size(); ++i) {
            if (!(v[i] >= v[i - 1])) {
                return false;
            }
        }
        uni = v;
        return true;
    }

    // drafter probability bin: 0.1 wide below 0.9, then [0.9, 0.95), [0.95, 0.98), [0.98, 1]
    static int bin_of(float p) {
        if (!(p > 0.0f)) {
            return 0;
        }
        if (p < 0.9f) {
            return std::min(8, (int) (p*10.0f));
        }
        return p < 0.95f ? 9 : (p < 0.98f ? 10 : 11);
    }

    static float bin_mid(int b) {
        static const float mid[N_BIN] = { 0.05f, 0.15f, 0.25f, 0.35f, 0.45f, 0.55f, 0.65f, 0.75f, 0.85f, 0.925f, 0.965f, 0.99f };
        return mid[std::min(N_BIN - 1, std::max(0, b))];
    }

    // calibrated acceptance of a draft token at position pos (0-based) with drafter probability p
    float acc(int mode, int pos, float p) const {
        mode = mode ? 1 : 0;
        pos  = std::min(N_POS - 1, std::max(0, pos));
        const int b = bin_of(p);
        const float pooled = (pok[mode][b] + bin_mid(b)*cw0) / (ptr[mode][b] + cw0);
        return (cok[mode][pos][b] + pooled*cw1) / (ctr[mode][pos][b] + cw1);
    }

    // best-case acceptance of the next position: the top bin, or a better one if the data says so
    float acc_hi(int mode, int pos) const {
        float hi = 0.0f;
        for (int b = N_BIN - 3; b < N_BIN; ++b) {
            hi = std::max(hi, acc(mode, pos, bin_mid(b)));
        }
        return hi;
    }

    // verify result of one sequence: ps = drafter probability of each proposed token, n_acc of them accepted
    void observe_accept(int mode, const std::vector<float> & ps, int n_acc) {
        mode = mode ? 1 : 0;
        for (int i = 0; i < (int) ps.size() && i <= n_acc; ++i) {
            const int   pos = std::min(N_POS - 1, i);
            const int   b   = bin_of(ps[i]);
            const float ok  = i < n_acc ? 1.0f : 0.0f;
            ctr[mode][pos][b] = ctr[mode][pos][b]*cdecay + 1.0f;
            cok[mode][pos][b] = cok[mode][pos][b]*cdecay + ok;
            ptr[mode][b]      = ptr[mode][b]*cdecay + 1.0f;
            pok[mode][b]      = pok[mode][b]*cdecay + ok;
        }
    }

    // union growth basis g(R) = U(R)/U(1) - 1, extended linearly past the table
    double g(int R) const {
        if (R <= 1) {
            return 0.0;
        }
        const int n = (int) uni.size();
        if (R <= n) {
            return uni[R - 1] - 1.0;
        }
        const double last = uni[n - 1] - 1.0;
        const double inc  = n >= 2 ? (double) (uni[n - 1] - uni[n - 2]) : 1.0;
        return last + (R - n)*std::max(0.0, inc);
    }

    double mean(int R) const {
        return vn[R - 1] > 0.0 ? vs[R - 1]/vn[R - 1] : 0.0;
    }

    // the two most measured widths, R_a < R_b (0 when there are fewer)
    void top2(int & ra, int & rb) const {
        ra = 0;
        rb = 0;
        for (int r = 1; r <= R_MAX; ++r) {
            if (vn[r - 1] <= 0.0) {
                continue;
            }
            if (ra == 0 || vn[r - 1] > vn[ra - 1]) {
                rb = ra;
                ra = r;
            } else if (rb == 0 || vn[r - 1] > vn[rb - 1]) {
                rb = r;
            }
        }
        if (ra > 0 && rb > 0 && rb < ra) {
            std::swap(ra, rb);
        }
    }

    // relative cost of the union basis: V(R)/V(1) = 1 + beta*g(R). From the two most measured widths, shrunk towards
    // the prior (row0 for the second row)
    double beta() const {
        const double g2 = g(2);
        const double b0 = g2 > 0.0 ? row0/g2 : row0;
        int ra = 0, rb = 0;
        top2(ra, rb);
        if (ra == 0 || rb == 0) {
            return b0;
        }
        const double ma = mean(ra), mb = mean(rb);
        const double ga = g(ra),    gb = g(rb);
        const double den = gb - (mb/ma)*ga;
        if (!(ma > 0.0) || !(mb > 0.0) || !(den > 1e-9)) {
            return b0;
        }
        const double b_obs = std::min(8.0, std::max(0.0, (mb/ma - 1.0)/den));
        const double n_eff = std::min(vn[ra - 1], vn[rb - 1]);
        return (b_obs*n_eff + b0*w_beta)/(n_eff + w_beta);
    }

    // estimated rest-of-step time (us) with R verify rows: the width's own mean shrunk towards the union curve anchored
    // at the most measured width; never falls with R. 0 when nothing is measured.
    double V(int R) const {
        R = std::max(1, R);
        if (R > R_MAX) {
            return 1e30;
        }
        int ref = 0;
        for (int r = 1; r <= R_MAX; ++r) {
            if (vn[r - 1] > 0.0 && (ref == 0 || vn[r - 1] > vn[ref - 1])) {
                ref = r;
            }
        }
        if (ref == 0) {
            return 0.0;
        }
        const double b  = beta();
        const double m0 = mean(ref);
        double v = 0.0;
        for (int r = 1; r <= R; ++r) {
            const double curve = m0*(1.0 + b*g(r))/(1.0 + b*g(ref));
            const double est   = (vs[r - 1] + w_fit*curve)/(vn[r - 1] + w_fit);
            v = std::max(v, est);
        }
        return v;
    }

    double t_d(int n_rows) const {
        const int i = std::min(N_TD, std::max(1, n_rows)) - 1;
        if (tdn[i] > 0.0) {
            return td[i];
        }
        for (int j = 0; j < N_TD; ++j) { // an unmeasured width borrows any measured one
            if (tdn[j] > 0.0) {
                return td[j];
            }
        }
        return 0.0;
    }

    void add_td(int n_rows, double us) {
        if (!(us > 0.0)) {
            return;
        }
        const int i = std::min(N_TD, std::max(1, n_rows)) - 1;
        td[i]   = tdn[i] > 0.0 ? 0.9*td[i] + 0.1*us : us;
        tdn[i] += 1.0;
    }

    // one step sample: R verify rows, us from the end of a draft() call to the start of the next. false if dropped.
    bool add_step(int R, double us) {
        if (R < 1 || R > R_MAX || !(us > 0.0)) {
            return false;
        }
        if (ready()) {
            const double v = V(R);
            if (v > 0.0 && us > outlier*v) {
                return false; // idle gap, a prompt chunk of another slot, a stall
            }
        }
        for (int r = 0; r < R_MAX; ++r) {
            vs[r] *= age;
            vn[r] *= age;
        }
        vs[R - 1] = vs[R - 1]*decay + us;
        vn[R - 1] = vn[R - 1]*decay + 1.0;
        n_valid++;
        return true;
    }

    // enough samples; with one width measured the other widths follow the prior curve until the policy tries them
    bool ready() const {
        int ra = 0, rb = 0;
        top2(ra, rb);
        return n_valid >= warm && ra > 0 && t_d(1) > 0.0;
    }

    // P_prev * a: probability that the new token and every earlier one of this sequence are accepted; E: the expected
    // tokens of every drafting sequence so far; R: verify rows so far; d: draft decodes so far; n: draft batch width
    bool keep(double p_new, double E, int R, int d, int n) const {
        const double t  = V(R) + d*t_d(n);
        const double dv = V(R + 1) - V(R);
        return p_new*t >= E*dv;
    }

    // one more draft decode: P = cumulative acceptance of the sequence's kept tokens, a_hi = acc_hi of the next position
    bool more(double P, double a_hi, double E, int R, int d, int n) const {
        const double t  = V(R) + d*t_d(n);
        const double dv = V(R + 1) - V(R);
        return P*a_hi*t >= E*(dv + t_d(n));
    }

    // a probe step (the p_min rule) every `probe` policy steps
    bool policy_step() {
        if (!ready()) {
            return false;
        }
        if (probe > 0 && ++n_policy > probe) {
            n_policy = 0;
            return false;
        }
        return true;
    }
};
