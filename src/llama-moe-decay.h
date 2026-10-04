#pragma once

// [TAG_FN_R4_ADAPT_DECAY] the long-memory admission policy of the adaptive hot set (LLAMA_MOE_HOT_DECAY), header only so
// tests/test-moe-decay.cpp runs the same code as src/llama-moecache.cpp.
//
// The windowed policy (SP-4, LLAMA_MOE_HOT_ADMIT=N/W) admits after N sightings in the last W steps. Strata replayed two
// decode routing traces of UD-Q4_K_XL on a 5090 + 9950X3D through both families (PR #407): short windows gave 24-96%
// MORE misses (they evict rarely-but-steadily used experts and swap them back and forth), long memory with decay 31-38%
// fewer. This is the decayed form (Strata 0.1.38 adapt(), --adapt-tuned; flashrt's decayed LFU and prompt warm-up):
//   - every expert a decode step routes adds 1 to its count (once per step, however many of the step's tokens use it:
//     the unit of the trace replay that chose the defaults);
//   - every `every` steps one pass: candidates are non-resident experts with count >= admit, best first; victims are the
//     free slots, then the resident experts with the lowest counts; candidate i replaces victim i while its count beats
//     both ratio x and hyst + the victim's; all layers' pairs are taken by gain (candidate - victim) under the caller's
//     upload budget; then every count is multiplied by decay;
//   - the prompt's routing (prefill ubatches) is folded in x seed at the next step, so a new request warms the set
//     toward its own experts before its first decode steps.
// Evict first, publish after the upload has landed: the caller's existing mechanics (llama-moecache.cpp).

#include <algorithm>
#include <cstdint>
#include <vector>

struct llama_moe_decay_params {
    float decay = 0.92f; // LLAMA_MOE_HOT_DECAY: counts x decay after every pass (0 = the policy is off)
    int   every = 2;     // LLAMA_MOE_HOT_DECAY_EVERY: decode steps per pass
    float admit = 2.0f;  // LLAMA_MOE_HOT_DECAY_ADMIT: a candidate's minimum count
    float ratio = 1.2f;  // LLAMA_MOE_HOT_DECAY_RATIO: ... and more than ratio x its victim's count
    float hyst  = 0.5f;  // LLAMA_MOE_HOT_DECAY_HYST:  ... and more than its victim's count + hyst
    float seed  = 0.03f; // LLAMA_MOE_HOT_SEED: prompt routing counts folded in x seed (0 = no warm-up)
};

struct llama_moe_decay_swap {
    int     layer;  // the caller's layer index
    int32_t expert; // admitted
    int32_t slot;   // the victim's slot
    int32_t victim; // the expert evicted from it, -1 for a free slot
    float   gain;   // count(expert) - count(victim), a free slot counting -1
};

// the pairs of one layer, appended to out. cnt: [n_expert] decayed counts; slot_expert[s]: the resident expert or -1;
// slot_busy[s]: an upload into s is in flight (neither a victim nor free); expert_busy[e]: e is resident or on its way.
inline void llama_moe_decay_pairs(int layer, const std::vector<float> & cnt, const std::vector<int32_t> & slot_expert,
        const std::vector<uint8_t> & slot_busy, const std::vector<uint8_t> & expert_busy, const llama_moe_decay_params & p,
        std::vector<llama_moe_decay_swap> & out) {
    std::vector<int32_t> cand;
    for (int32_t e = 0; e < (int32_t) cnt.size(); ++e) {
        if (!expert_busy[e] && cnt[e] >= p.admit) {
            cand.push_back(e);
        }
    }
    if (cand.empty()) {
        return;
    }
    std::stable_sort(cand.begin(), cand.end(), [&](int32_t a, int32_t b) { return cnt[a] > cnt[b]; });

    // victims: free slots first, then residents by ascending count (slot order on ties)
    std::vector<int32_t> vict;
    for (int32_t s = 0; s < (int32_t) slot_expert.size(); ++s) {
        if (!slot_busy[s] && slot_expert[s] < 0) {
            vict.push_back(s);
        }
    }
    const size_t n_free = vict.size();
    for (int32_t s = 0; s < (int32_t) slot_expert.size(); ++s) {
        if (!slot_busy[s] && slot_expert[s] >= 0) {
            vict.push_back(s);
        }
    }
    std::stable_sort(vict.begin() + n_free, vict.end(), [&](int32_t a, int32_t b) {
        return cnt[slot_expert[a]] < cnt[slot_expert[b]];
    });

    const size_t n = std::min(cand.size(), vict.size());
    for (size_t i = 0; i < n; ++i) {
        const int32_t e  = cand[i];
        const int32_t s  = vict[i];
        const int32_t ve = slot_expert[s];
        const float   vc = ve >= 0 ? cnt[ve] : -1.0f;
        if (ve >= 0 && !(cnt[e] > p.ratio*vc && cnt[e] > vc + p.hyst)) {
            break; // the best remaining candidate does not beat the weakest remaining resident
        }
        out.push_back({ layer, e, s, ve, cnt[e] - vc });
    }
}

// all layers' pairs, best gain first (stable: layer order, then candidate order on ties)
inline void llama_moe_decay_order(std::vector<llama_moe_decay_swap> & swaps) {
    std::stable_sort(swaps.begin(), swaps.end(), [](const llama_moe_decay_swap & a, const llama_moe_decay_swap & b) {
        return a.gain > b.gain;
    });
}

// one decode step's routing (ids [n], any value outside [0, n_expert) ignored) into the counts
inline void llama_moe_decay_count(std::vector<float> & cnt, const int32_t * ids, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        const int32_t e = ids[i];
        if (e >= 0 && e < (int32_t) cnt.size()) {
            cnt[e] += 1.0f;
        }
    }
}

// the prompt's counts x seed into the counts, then cleared
inline void llama_moe_decay_fold_seed(std::vector<float> & cnt, std::vector<float> & seed, float f) {
    for (size_t e = 0; e < cnt.size() && e < seed.size(); ++e) {
        cnt[e] += f*seed[e];
        seed[e] = 0.0f;
    }
}

inline void llama_moe_decay_apply(std::vector<float> & cnt, float decay) {
    for (float & c : cnt) {
        c *= decay;
    }
}
