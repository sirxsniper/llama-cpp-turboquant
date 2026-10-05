#pragma once

// [TAG_FN_L3_MTP_BLOCK] Block verification of a sampled draft (Sun et al. 2024, "Block Verification Accelerates
// Speculative Decoding", arXiv 2403.10444, Algorithm 2).
//
// The per-token rule accepts draft token i with probability min(1, p/q) and stops at the first rejection. Block
// verification carries a weight through the block and keeps the longest prefix whose own test succeeds, so a token the
// per-token rule rejects can still be kept when the tokens after it make up for it. The output distribution is the
// target's, as with the per-token rule; the expected accepted length is never lower. Exact enumeration (5 tokens, random
// conditionals, first-token acceptance 0.66-0.89): +0.7-1.5 % tokens per verify step with 2 drafts, +2-4 % with 3.
//
//   p_r: the target's distribution at verify row r (after r drafts), q_r: the drafter's distribution of draft r + 1
//   w_0 = 1,  w_i = min(1, w_{i-1} * p_{i-1}(X_i) / q_{i-1}(X_i))
//   h_i = S_i / (S_i + 1 - w_i),  S_i = sum_x max(w_i * p_i(x) - q_i(x), 0)     (i < gamma);  h_gamma = w_gamma
//   tau = the last i in 1..gamma with u_i < h_i (0 when there is none), u_i independent uniforms
//   the token after the tau kept drafts: p_gamma when tau = gamma, else max(w_tau * p_tau - q_tau, 0) normalised
//
// Pure arithmetic over candidate arrays (ids with probabilities that sum to 1, any order). tests/test-spec-block.cpp
// checks it against an exact enumeration.

#include "llama.h"

#include <algorithm>
#include <cstddef>
#include <vector>

struct common_spec_block_dist {
    const llama_token_data * data = nullptr;
    size_t                   size = 0;
};

static inline double common_spec_block_prob(const common_spec_block_dist & d, llama_token id) {
    for (size_t k = 0; k < d.size; ++k) {
        if (d.data[k].id == id) {
            return d.data[k].p;
        }
    }
    return 0.0;
}

// w * p - q over p's candidates where it is positive (q's mass outside p's support only lowers the sum); out may be null
static inline double common_spec_block_excess(double w, const common_spec_block_dist & p, const common_spec_block_dist & q,
        std::vector<llama_token_data> * out = nullptr) {
    double s = 0.0;
    if (out) {
        out->clear();
    }
    for (size_t k = 0; k < p.size; ++k) {
        const double r = w*(double) p.data[k].p - common_spec_block_prob(q, p.data[k].id);
        if (r > 0.0) {
            s += r;
            if (out) {
                out->push_back({ p.data[k].id, 0.0f, (float) r });
            }
        }
    }
    return s;
}

// p: target distributions at rows 0..gamma-1, q: the drafter's distributions of drafts 1..gamma (q[i] for draft[i]).
// w receives w_0..w_gamma, h receives h_1..h_gamma (index 0 unused).
static inline void common_spec_block_weights(const common_spec_block_dist * p, const common_spec_block_dist * q,
        const llama_token * draft, int gamma, std::vector<double> & w, std::vector<double> & h) {
    w.assign((size_t) gamma + 1, 0.0);
    h.assign((size_t) gamma + 1, 0.0);
    w[0] = 1.0;
    for (int i = 1; i <= gamma; ++i) {
        const double px = common_spec_block_prob(p[i - 1], draft[i - 1]);
        const double qx = common_spec_block_prob(q[i - 1], draft[i - 1]);
        w[i] = qx > 0.0 ? std::min(1.0, w[i - 1]*px/qx) : 0.0;
        if (i < gamma) {
            const double s   = common_spec_block_excess(w[i], p[i], q[i]);
            const double den = s + 1.0 - w[i];
            h[i] = den > 0.0 ? s/den : 1.0; // w_i = 1 and no excess: p_i = q_i, keep (the next weight is 1 again)
        } else {
            h[i] = w[i];
        }
    }
}

// the number of kept drafts: the last i with u[i - 1] < h[i]
static inline int common_spec_block_tau(const std::vector<double> & h, int gamma, const float * u) {
    int tau = 0;
    for (int i = 1; i <= gamma; ++i) {
        if ((double) u[i - 1] < h[i]) {
            tau = i;
        }
    }
    return tau;
}
