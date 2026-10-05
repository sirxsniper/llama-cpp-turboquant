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
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <istream>
#include <sstream>
#include <string>
#include <utility>
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
    int     victim_layer = -1; // [TAG_FN_L3_POLICY_POOL] the victim's layer in a shared slot pool, -1: `layer`
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

// [TAG_FN_L3_POLICY_POOL] one slot pool shared by the layers of an expert shape class: the candidates and the residents
// of all its layers compete for its slots by the rules above (llama_moe_decay_pairs over the keys k*n_expert + e).
// cnt[k], expert_busy[k]: pool layer k. slot_layer[s] / slot_expert[s]: the pool layer and the expert resident in slot s
// (-1 / -1: free, or an upload in flight when slot_busy[s]). Out: layer = the candidate's pool layer, victim_layer = the
// victim's pool layer (-1 for a free slot).
inline void llama_moe_decay_pairs_pool(const std::vector<const std::vector<float> *> & cnt,
        const std::vector<const std::vector<uint8_t> *> & expert_busy, const std::vector<int32_t> & slot_layer,
        const std::vector<int32_t> & slot_expert, const std::vector<uint8_t> & slot_busy, const llama_moe_decay_params & p,
        std::vector<llama_moe_decay_swap> & out) {
    if (cnt.empty() || cnt.size() != expert_busy.size()) {
        return;
    }
    const int32_t n_exp = (int32_t) cnt[0]->size();
    std::vector<float>   fc((size_t) n_exp*cnt.size(), 0.0f);
    std::vector<uint8_t> fb(fc.size(), 1);
    for (size_t k = 0; k < cnt.size(); ++k) {
        for (int32_t e = 0; e < n_exp && e < (int32_t) cnt[k]->size() && e < (int32_t) expert_busy[k]->size(); ++e) {
            fc[k*n_exp + e] = (*cnt[k])[e];
            fb[k*n_exp + e] = (*expert_busy[k])[e];
        }
    }
    std::vector<int32_t> fs(slot_layer.size(), -1);
    for (size_t s = 0; s < fs.size(); ++s) {
        const int32_t k = slot_layer[s];
        const int32_t e = s < slot_expert.size() ? slot_expert[s] : -1;
        if (k >= 0 && k < (int32_t) cnt.size() && e >= 0 && e < n_exp) {
            fs[s] = k*n_exp + e;
        }
    }
    std::vector<llama_moe_decay_swap> tmp;
    llama_moe_decay_pairs(0, fc, fs, slot_busy, fb, p, tmp);
    for (const auto & w : tmp) {
        llama_moe_decay_swap o = { w.expert / n_exp, w.expert % n_exp, w.slot, w.victim >= 0 ? w.victim % n_exp : -1, w.gain };
        o.victim_layer = w.victim >= 0 ? w.victim / n_exp : -1;
        out.push_back(o);
    }
}

// all layers' pairs, best gain first (stable: layer order, then candidate order on ties)
inline void llama_moe_decay_order(std::vector<llama_moe_decay_swap> & swaps) {
    std::stable_sort(swaps.begin(), swaps.end(), [](const llama_moe_decay_swap & a, const llama_moe_decay_swap & b) {
        return a.gain > b.gain;
    });
}

// [TAG_FN_L3_POLICY_BURST] the swaps one pass queues, best gain first: any pair within pass_bytes; past it, up to
// total_bytes, only the pairs strong() accepts (LLAMA_MOE_HOT_BURST_MIB). total_bytes <= pass_bytes: the first pair over the
// pass budget ends the pass. out: (index into swaps, taken past the pass budget)
template <typename BytesOf, typename Strong>
inline void llama_moe_decay_select(const std::vector<llama_moe_decay_swap> & swaps, size_t pass_bytes, size_t total_bytes,
        BytesOf bytes_of, Strong strong, std::vector<std::pair<size_t, bool>> & out) {
    out.clear();
    size_t used = 0;
    for (size_t i = 0; i < swaps.size(); ++i) {
        const size_t b    = bytes_of(swaps[i]);
        const bool   over = used + b > pass_bytes;
        if (over) {
            if (used + b > total_bytes) {
                break;
            }
            if (!strong(swaps[i])) {
                continue;
            }
        }
        used += b;
        out.push_back({ i, over });
    }
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

// [TAG_FN_L3_POLICY_STATE] the saved state of the decayed set (LLAMA_MOE_HOT_STATE), one text file per model:
//   "moehot v1 n_expert=<n> layers=<n> steps=<decode steps> model=<name>"
//   per layer: "L <il> <bytes per expert> <n residents> <the residents, best first> ; <the count of every expert>"
struct llama_moe_hotstate_layer {
    int                  il    = -1;
    size_t               bytes = 0;    // up + gate + down per expert: a state of another quantization is not used
    std::vector<int32_t> res;          // resident experts, best count first
    std::vector<float>   cnt;          // decayed count of every expert
};

// the model name as one token of the header line
inline std::string llama_moe_hotstate_name(const std::string & s) {
    std::string r = s.empty() ? std::string("-") : s;
    for (char & c : r) {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            c = '_';
        }
    }
    return r;
}

inline std::string llama_moe_hotstate_format(int64_t n_expert, uint64_t steps, const std::string & model,
        const std::vector<llama_moe_hotstate_layer> & layers) {
    std::string out;
    char buf[64];
    out += "moehot v1 n_expert=" + std::to_string(n_expert) + " layers=" + std::to_string(layers.size()) + " steps=" +
           std::to_string(steps) + " model=" + llama_moe_hotstate_name(model) + "\n";
    for (const auto & l : layers) {
        out += "L " + std::to_string(l.il) + " " + std::to_string(l.bytes) + " " + std::to_string(l.res.size());
        for (const int32_t e : l.res) {
            out += " " + std::to_string(e);
        }
        out += " ;";
        for (int64_t e = 0; e < n_expert; ++e) {
            snprintf(buf, sizeof(buf), " %.6g", e < (int64_t) l.cnt.size() ? (double) l.cnt[e] : 0.0);
            out += buf;
        }
        out += "\n";
    }
    return out;
}

// false (and why) when the header is not a moehot v1 header of this expert count and model; a broken layer line (a
// number missing, no ';', an id out of range, a count that is not a finite number >= 0) is left out and counted in *n_bad
inline bool llama_moe_hotstate_parse(std::istream & in, int64_t n_expert, const std::string & model,
        std::vector<llama_moe_hotstate_layer> & out, uint64_t & steps, std::string & why, size_t * n_bad = nullptr) {
    out.clear();
    steps = 0;
    std::string line;
    if (!std::getline(in, line) || line.rfind("moehot v1", 0) != 0) {
        why = "not a moehot v1 file";
        return false;
    }
    const std::string want_exp   = "n_expert=" + std::to_string(n_expert);
    const std::string want_model = "model=" + llama_moe_hotstate_name(model);
    bool exp_ok = false;
    bool model_ok = false;
    {
        std::istringstream hs(line);
        std::string tok;
        while (hs >> tok) {
            exp_ok   = exp_ok   || tok == want_exp;
            model_ok = model_ok || tok == want_model;
            if (tok.rfind("steps=", 0) == 0) {
                steps = strtoull(tok.c_str() + 6, nullptr, 10);
            }
        }
    }
    if (!exp_ok || !model_ok) {
        why = "saved for another model or expert count: " + line;
        return false;
    }
    size_t bad = 0;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        std::istringstream ss(line);
        std::string tag;
        llama_moe_hotstate_layer l;
        size_t nr = 0;
        if (!(ss >> tag >> l.il >> l.bytes >> nr) || tag != "L" || l.il < 0 || nr > (size_t) n_expert) {
            bad++;
            continue;
        }
        bool ok = true;
        l.res.resize(nr);
        for (auto & e : l.res) {
            ok = ok && (ss >> e) && e >= 0 && e < n_expert;
        }
        std::string sep;
        ok = ok && (ss >> sep) && sep == ";";
        l.cnt.resize((size_t) n_expert);
        for (auto & c : l.cnt) {
            ok = ok && (ss >> c) && std::isfinite(c) && c >= 0.0f;
        }
        std::string extra;
        ok = ok && !(ss >> extra);
        if (!ok) {
            bad++;
            continue;
        }
        out.push_back(std::move(l));
    }
    if (n_bad) {
        *n_bad = bad;
    }
    return true;
}
