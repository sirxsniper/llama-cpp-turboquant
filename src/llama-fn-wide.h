#pragma once

// [TAG_FN_WIDE_DECODE] wide decode on the MoE bridge: the pure rules. With --parallel 3 or 4 and MTP (n_max 2) a verify
// step is one ubatch of 9 or 12 rows (3 or 4 sequences of 3 rows). The bridge takes 8 rows by default and the hot chain
// 8, so today such a step runs the scheduler's CPU split and the hot-set observer counts it as a prompt. With the switch
// the bridge, the hot chain (in chunks of LLAMA_MOE_HOT_MAX_T rows, the LLAMA_FN_L4_PROMPT_HOT path) and the observer
// take decode ubatches up to the width the ring can roll back; prompt ubatches keep the width they have today.
// No model and no backend: the MoE bridge, the graph, the hot set, the context and tests/test-fn-wide.cpp use these.
//
// LLAMA_FN_WIDE_DECODE: 0 / unset off; 1 on; 2 on with the hot-set observer as before (a wide step still counts as a
// prompt), to tell the two effects apart in an A/B run.

#include <algorithm>
#include <utility>
#include <vector>

namespace fn_wide {

constexpr int BASE_T = 8;  // the bridge's default width (LLAMA_MOE_BRIDGE_MAX_T) and the widest hot chain (LLAMA_MOE_HOT_MAX_T)
constexpr int MAX_T  = 16; // the bridge's limit (llama_moe_bridge_create) and the CPU pool's (GGML_FN_MOE_MAX_T)

// 0 off, 1 on, 2 on with the observer as before; any other value is off
inline int mode_from_int(int v) {
    return v == 1 || v == 2 ? v : 0;
}

// the rows of the widest decode ubatch the ring can roll back with every sequence at most seq_lim rows: n_seq_max
// (= ring_cap / n_rs_seq) sequences of min(seq_lim, n_rs_seq) rows; 0 without a ring. seq_lim is the bridge's limit per
// sequence (its seq_t, or max_t when LLAMA_MOE_BRIDGE_MAX_T is below n_rs_seq)
inline int ring_rows(int ring_cap, int n_rs_seq, int seq_lim) {
    if (ring_cap <= 0 || n_rs_seq <= 0 || seq_lim <= 0) {
        return 0;
    }
    return (ring_cap/n_rs_seq)*std::min(seq_lim, n_rs_seq);
}

// the widest decode ubatch the switch puts on the bridge: ring_rows, up to MAX_T, when that is wider than BASE_T;
// 0 = nothing to widen (one or two connections with n_max <= 3, no ring, or the switch off)
inline int width(int mode, int rows) {
    if (mode <= 0 || rows <= BASE_T) {
        return 0;
    }
    return std::min(MAX_T, rows);
}

// the bridge's job and buffer width: its own width (LLAMA_MOE_BRIDGE_MAX_T, capped by the ring) or the wide one
inline int job_t(int max_t, int wide_t) {
    return std::max(max_t, wide_t);
}

// a wide decode ubatch: wider than BASE_T and no wider than wide_t, an equal split over 2 or more sequences of the same
// 1..seq_t rows each (seq_t: the rows of one sequence the ring rolls back), every row an output (a verify step asks
// for the logits of each row; a prompt chunk only for its last). A prompt ubatch of one sequence never is one, and a
// ubatch that holds a prompt chunk of another sequence is not one either.
inline bool decode_ub(int wide_t, int n_tokens, bool equal_seqs, int n_seqs, int n_seq_tokens, int seq_t, int n_outputs) {
    return wide_t > 0 && n_tokens > BASE_T && n_tokens <= wide_t && equal_seqs && n_seqs >= 2 && n_seq_tokens >= 1 &&
           n_seq_tokens <= seq_t && n_tokens == n_seqs*n_seq_tokens && n_outputs == n_tokens;
}

// the hot-set observer: the routed ids of a graph of n_tokens rows count as a decode step (else as a prompt). obs_t: the
// rows of the wide decode graph in flight (0: none)
inline bool obs_decode(int n_tokens, int hot_max_t, int obs_t) {
    return n_tokens <= hot_max_t || (obs_t > BASE_T && n_tokens == obs_t);
}

// the wide decode shapes a compute reserve of n_seqs sequences and n_tokens rows adds: every (sequences, rows each)
// with 2..n_seqs sequences of 1..seq_t rows that decode_ub accepts, no wider than n_tokens and n_out_max (the graph's
// outputs). Empty = no extra reserve.
inline std::vector<std::pair<int, int>> reserve_shapes(int wide_t, int n_seqs, int seq_t, int n_tokens, int n_out_max) {
    std::vector<std::pair<int, int>> out;
    for (int s = 2; s <= n_seqs; ++s) {
        for (int t = 1; t <= seq_t; ++t) {
            const int rows = s*t;
            if (rows <= n_tokens && rows <= n_out_max && decode_ub(wide_t, rows, true, s, t, seq_t, rows)) {
                out.emplace_back(s, t);
            }
        }
    }
    return out;
}

} // namespace fn_wide
