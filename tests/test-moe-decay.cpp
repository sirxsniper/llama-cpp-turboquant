// [TAG_FN_R4_ADAPT_DECAY] the decayed-count policy of the adaptive hot set (src/llama-moe-decay.h), the code that
// llama-moecache.cpp runs under LLAMA_MOE_HOT_DECAY. CPU only, no model, no ggml.
//
// Checks: the pairing rules (free slots first, admit threshold, ratio and hysteresis, busy slots and experts, gain
// order across layers), the prompt seed, and three replays of one layer: steady experts are never evicted by bursts (the
// failure of short windows in Strata's replay, PR #407), a shifted working set moves in within a bounded number of
// steps, and a long run with a skewed stream keeps the most used experts resident.

#include "../src/llama-moe-decay.h"

#include <cstdio>
#include <random>
#include <vector>

static int g_fail   = 0;
static int g_checks = 0;

#define TCHECK(cond, ...)                                        \
    do {                                                         \
        ++g_checks;                                              \
        if (!(cond)) {                                           \
            ++g_fail;                                            \
            fprintf(stderr, "FAIL %s:%d: ", __func__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                        \
            fprintf(stderr, "\n");                               \
        }                                                        \
    } while (0)

// one layer as llama-moecache.cpp keeps it; uploads land at once (the real set publishes them a step later)
struct sim_layer {
    std::vector<float>   cnt;
    std::vector<float>   seed;
    std::vector<int32_t> slot_expert;
    std::vector<int32_t> expert_slot;
    uint64_t evictions = 0;
    uint64_t admits    = 0;

    sim_layer(int n_expert, int n_slots) : cnt(n_expert, 0.0f), seed(n_expert, 0.0f), slot_expert(n_slots, -1), expert_slot(n_expert, -1) {}

    bool resident(int32_t e) const { return expert_slot[e] >= 0; }
};

static void sim_pass(std::vector<sim_layer *> & layers, const llama_moe_decay_params & p, int max_swaps,
                     std::vector<llama_moe_decay_swap> * out = nullptr) {
    std::vector<llama_moe_decay_swap> swaps;
    for (int li = 0; li < (int) layers.size(); ++li) {
        sim_layer & L = *layers[li];
        std::vector<uint8_t> slot_busy(L.slot_expert.size(), 0);
        std::vector<uint8_t> expert_busy(L.cnt.size(), 0);
        for (size_t e = 0; e < L.cnt.size(); ++e) {
            expert_busy[e] = L.resident((int32_t) e) ? 1 : 0;
        }
        llama_moe_decay_pairs(li, L.cnt, L.slot_expert, slot_busy, expert_busy, p, swaps);
    }
    llama_moe_decay_order(swaps);
    int n = 0;
    for (const auto & w : swaps) {
        if (n++ >= max_swaps) {
            break;
        }
        sim_layer & L = *layers[w.layer];
        if (w.victim >= 0) {
            L.expert_slot[w.victim] = -1;
            L.evictions++;
        }
        L.slot_expert[w.slot]   = w.expert;
        L.expert_slot[w.expert] = w.slot;
        L.admits++;
        if (out) {
            out->push_back(w);
        }
    }
    for (sim_layer * L : layers) {
        llama_moe_decay_apply(L->cnt, p.decay);
    }
}

static void test_pairs() {
    llama_moe_decay_params p;
    p.admit = 2.0f;
    p.ratio = 1.2f;
    p.hyst  = 0.5f;

    // 8 experts, 3 slots: slot 0 holds e0 (count 4), slot 1 is free, slot 2 holds e1 (count 1)
    std::vector<float>   cnt  = { 4.0f, 1.0f, 9.0f, 1.9f, 5.0f, 1.25f, 0.0f, 3.0f };
    std::vector<int32_t> se   = { 0, -1, 1 };
    std::vector<uint8_t> sb   = { 0, 0, 0 };
    std::vector<uint8_t> eb   = { 1, 1, 0, 0, 0, 0, 0, 0 };
    std::vector<llama_moe_decay_swap> out;
    llama_moe_decay_pairs(0, cnt, se, sb, eb, p, out);
    // candidates by count: e2 (9), e4 (5), e7 (3); e3 (1.9) is under admit. Victims: free slot 1, then e1 (slot 2),
    // then e0 (slot 0). e2 -> slot 1 (free), e4 (5) vs e1 (1): 5 > 1.2 and 5 > 1.5 -> slot 2; e7 (3) vs e0 (4): no
    TCHECK(out.size() == 2, "two pairs, got %zu", out.size());
    if (out.size() == 2) {
        TCHECK(out[0].expert == 2 && out[0].slot == 1 && out[0].victim == -1, "best candidate into the free slot");
        TCHECK(out[1].expert == 4 && out[1].slot == 2 && out[1].victim == 1, "next candidate over the weakest resident");
        TCHECK(out[0].gain == 10.0f && out[1].gain == 4.0f, "gains (free slot counts -1)");
    }

    // ratio: 2.3 vs 2.0 is +0.3, under the hysteresis; 2.6 vs 2.0 is under 1.2x (2.4)? no, 2.6 > 2.4 and > 2.5: admitted
    {
        std::vector<float>   c2 = { 2.0f, 2.3f };
        std::vector<int32_t> s2 = { 0 };
        std::vector<uint8_t> b2 = { 0 };
        std::vector<uint8_t> e2 = { 1, 0 };
        std::vector<llama_moe_decay_swap> o2;
        llama_moe_decay_pairs(0, c2, s2, b2, e2, p, o2);
        TCHECK(o2.empty(), "2.3 does not replace 2.0 (hysteresis)");
        c2[1] = 2.6f;
        llama_moe_decay_pairs(0, c2, s2, b2, e2, p, o2);
        TCHECK(o2.size() == 1, "2.6 replaces 2.0 (> 1.2x and > +0.5)");
        o2.clear();
        c2[0] = 10.0f;
        c2[1] = 11.5f; // +1.5 but only 1.15x
        llama_moe_decay_pairs(0, c2, s2, b2, e2, p, o2);
        TCHECK(o2.empty(), "11.5 does not replace 10 (ratio)");
    }

    // busy: an upload in flight takes neither its slot nor its expert
    {
        std::vector<float>   c3 = { 0.0f, 8.0f, 7.0f };
        std::vector<int32_t> s3 = { -1, -1 };
        std::vector<uint8_t> b3 = { 1, 0 };     // slot 0 is being written
        std::vector<uint8_t> e3 = { 0, 1, 0 };  // e1 is on its way
        std::vector<llama_moe_decay_swap> o3;
        llama_moe_decay_pairs(0, c3, s3, b3, e3, p, o3);
        TCHECK(o3.size() == 1 && o3[0].expert == 2 && o3[0].slot == 1, "busy slot and busy expert skipped");
    }

    // gain order across layers
    {
        std::vector<llama_moe_decay_swap> o;
        o.push_back({ 0, 1, 0, 3, 1.0f });
        o.push_back({ 1, 2, 0, -1, 5.0f });
        o.push_back({ 2, 3, 1, 4, 5.0f });
        llama_moe_decay_order(o);
        TCHECK(o[0].layer == 1 && o[1].layer == 2 && o[2].layer == 0, "best gain first, stable on ties");
    }
}

static void test_seed() {
    llama_moe_decay_params p;
    sim_layer L(16, 2);
    // a prompt routes expert 5 a hundred times and expert 6 twenty: x 0.03 they count 3 and 0.6
    for (int i = 0; i < 100; ++i) { const int32_t e = 5; llama_moe_decay_count(L.seed, &e, 1); }
    for (int i = 0; i < 20; ++i)  { const int32_t e = 6; llama_moe_decay_count(L.seed, &e, 1); }
    llama_moe_decay_fold_seed(L.cnt, L.seed, p.seed);
    TCHECK(L.seed[5] == 0.0f && L.seed[6] == 0.0f, "seed cleared after the fold");
    std::vector<sim_layer *> ls = { &L };
    sim_pass(ls, p, 64);
    TCHECK(L.resident(5), "the prompt's main expert is admitted before any decode step");
    TCHECK(!L.resident(6), "a seed under the admit threshold is not");
}

// steady experts, each routed every other step, against bursts of a new expert for 5 steps every 40
static void test_steady_vs_bursts() {
    llama_moe_decay_params p; // the defaults: 0.92 every 2 steps, admit 2, ratio 1.2, hyst 0.5
    sim_layer L(32, 4);
    for (int e = 0; e < 4; ++e) {
        L.slot_expert[e] = e;
        L.expert_slot[e] = e;
        L.cnt[e] = p.admit;
    }
    std::vector<sim_layer *> ls = { &L };
    uint64_t steady_evicted = 0;
    for (int step = 0; step < 800; ++step) {
        std::vector<int32_t> ids;
        if (step % 2 == 0) {
            ids = { 0, 1, 2, 3 };
        }
        if (step % 40 < 5) {
            ids.push_back((int32_t) (8 + (step/40) % 20)); // a burst expert
        }
        llama_moe_decay_count(L.cnt, ids.data(), (int64_t) ids.size());
        if ((step + 1) % p.every == 0) {
            std::vector<llama_moe_decay_swap> sw;
            sim_pass(ls, p, 64, &sw);
            for (const auto & w : sw) {
                steady_evicted += w.victim >= 0 && w.victim < 4;
            }
        }
    }
    TCHECK(steady_evicted == 0, "steady experts evicted %llu times by bursts", (unsigned long long) steady_evicted);
    for (int e = 0; e < 4; ++e) {
        TCHECK(L.resident(e), "steady expert %d resident at the end", e);
    }
}

// the working set moves from 0..3 to 10..13: the new set is resident within a bounded number of steps
static void test_shift() {
    llama_moe_decay_params p;
    sim_layer L(32, 4);
    std::vector<sim_layer *> ls = { &L };
    int moved_at = -1;
    for (int step = 0; step < 600; ++step) {
        const int base = step < 300 ? 0 : 10;
        std::vector<int32_t> ids = { base, base + 1, base + 2, base + 3 };
        llama_moe_decay_count(L.cnt, ids.data(), (int64_t) ids.size());
        if ((step + 1) % p.every == 0) {
            sim_pass(ls, p, 64);
        }
        if (step == 299) {
            for (int e = 0; e < 4; ++e) {
                TCHECK(L.resident(e), "first working set resident (expert %d)", e);
            }
        }
        if (step >= 300 && moved_at < 0 && L.resident(10) && L.resident(11) && L.resident(12) && L.resident(13)) {
            moved_at = step - 300;
        }
    }
    printf("  shift: the new working set was resident %d steps after the change\n", moved_at);
    TCHECK(moved_at >= 0 && moved_at <= 40, "the new working set moved in within 40 steps (took %d)", moved_at);
}

// a skewed stream (Zipf over 64 experts, 10 routed of 64 per step): after warm-up the resident set holds the most used
static void test_skewed() {
    llama_moe_decay_params p;
    sim_layer L(64, 16);
    std::vector<sim_layer *> ls = { &L };
    std::mt19937 rng(5);
    std::vector<double> w(64);
    for (int e = 0; e < 64; ++e) {
        w[e] = 1.0/(1.0 + e);
    }
    std::discrete_distribution<int> pick(w.begin(), w.end());
    uint64_t hit = 0, tot = 0;
    for (int step = 0; step < 4000; ++step) {
        std::vector<int32_t> ids;
        while (ids.size() < 10) {
            const int32_t e = (int32_t) pick(rng);
            bool dup = false;
            for (int32_t x : ids) {
                dup = dup || x == e;
            }
            if (!dup) {
                ids.push_back(e);
            }
        }
        if (step >= 1000) {
            for (int32_t e : ids) {
                hit += L.resident(e) ? 1 : 0;
                tot++;
            }
        }
        llama_moe_decay_count(L.cnt, ids.data(), (int64_t) ids.size());
        if ((step + 1) % p.every == 0) {
            sim_pass(ls, p, 64);
        }
    }
    int top = 0;
    for (int e = 0; e < 8; ++e) {
        top += L.resident(e) ? 1 : 0;
    }
    const double rate = tot ? (double) hit/tot : 0.0;
    printf("  skewed: hit rate %.3f with 16 of 64 slots, %d of the 8 most used resident, %llu admits\n", rate, top,
           (unsigned long long) L.admits);
    TCHECK(top == 8, "the 8 most used experts are resident (%d)", top);
    TCHECK(rate > 0.5, "hit rate %.3f > 0.5 on a Zipf stream with a quarter of the experts", rate);
}

int main() {
    printf("test-moe-decay: [TAG_FN_R4_ADAPT_DECAY]\n");
    test_pairs();
    test_seed();
    test_steady_vs_bursts();
    test_shift();
    test_skewed();
    printf("test-moe-decay: %d checks, %d errors%s\n", g_checks, g_fail, g_fail ? "" : " - PASSED");
    return g_fail == 0 ? 0 : 1;
}
