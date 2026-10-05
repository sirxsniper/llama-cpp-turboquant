// [TAG_FN_R4_ADAPT_DECAY] the decayed-count policy of the adaptive hot set (src/llama-moe-decay.h), the code that
// llama-moecache.cpp runs under LLAMA_MOE_HOT_DECAY. CPU only, no model, no ggml.
//
// Checks: the pairing rules (free slots first, admit threshold, ratio and hysteresis, busy slots and experts, gain
// order across layers), the prompt seed, and three replays of one layer: steady experts are never evicted by bursts (the
// failure of short windows in Strata's replay, PR #407), a shifted working set moves in within a bounded number of
// steps, and a long run with a skewed stream keeps the most used experts resident.

#include "../src/llama-moe-decay.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <sstream>
#include <string>
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

// [TAG_FN_L3_POLICY_POOL] the pairs of a slot pool shared by two layers: the same rules across the layers, victims of
// another layer, busy slots and experts
static void test_pool_pairs() {
    llama_moe_decay_params p;
    p.admit = 2.0f;
    p.ratio = 1.2f;
    p.hyst  = 0.5f;
    std::vector<float>   c0 = { 4.0f, 0.0f, 9.0f, 0.0f, 1.0f, 0.0f, 1.9f, 0.0f };
    std::vector<float>   c1 = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 5.0f, 0.0f, 3.0f };
    std::vector<uint8_t> b0 = { 1, 0, 0, 0, 1, 0, 0, 0 }; // e0 and e4 of layer 0 are resident
    std::vector<uint8_t> b1 = { 0, 0, 0, 0, 0, 0, 0, 0 };
    // 4 slots: (layer 0, e0), free, (layer 0, e4), an upload in flight
    std::vector<int32_t> sl = { 0, -1, 0, 1 };
    std::vector<int32_t> se = { 0, -1, 4, -1 };
    std::vector<uint8_t> sb = { 0, 0, 0, 1 };
    std::vector<llama_moe_decay_swap> out;
    llama_moe_decay_pairs_pool({ &c0, &c1 }, { &b0, &b1 }, sl, se, sb, p, out);
    // candidates: (0, e2) 9, (1, e5) 5, (1, e7) 3; (0, e6) 1.9 is under admit. Victims: slot 1 free, then (0, e4) 1 in
    // slot 2, then (0, e0) 4 in slot 0; slot 3 is busy. (0, e2) -> slot 1; (1, e5) 5 > 1.2 and > 1.5 -> slot 2 over
    // layer 0's e4; (1, e7) 3 vs (0, e0) 4: no
    TCHECK(out.size() == 2, "pool: two pairs, got %zu", out.size());
    if (out.size() == 2) {
        TCHECK(out[0].layer == 0 && out[0].expert == 2 && out[0].slot == 1 && out[0].victim == -1 && out[0].victim_layer == -1,
               "pool: the best candidate into the free slot");
        TCHECK(out[1].layer == 1 && out[1].expert == 5 && out[1].slot == 2 && out[1].victim == 4 && out[1].victim_layer == 0,
               "pool: layer 1 takes layer 0's weakest slot (layer %d e%d slot %d victim %d of layer %d)", out[1].layer,
               out[1].expert, out[1].slot, out[1].victim, out[1].victim_layer);
        TCHECK(out[0].gain == 10.0f && out[1].gain == 4.0f, "pool: gains");
    }
}

// [TAG_FN_L3_POLICY_POOL] two layers, a wide working set (12 experts a step) and a narrow one (2): 16 slots as one pool
// against a fixed 8 + 8. The pool gives the wide layer the slots the narrow one does not need.
static void test_pool_vs_fixed() {
    llama_moe_decay_params p;
    const int n_exp = 32;
    double hit_rate[2] = { 0.0, 0.0 };
    for (int pooled = 0; pooled < 2; ++pooled) {
        std::vector<float> cnt[2] = { std::vector<float>(n_exp, 0.0f), std::vector<float>(n_exp, 0.0f) };
        std::vector<int32_t> eslot[2] = { std::vector<int32_t>(n_exp, -1), std::vector<int32_t>(n_exp, -1) };
        const int n_slots = 16;
        std::vector<int32_t> sl(n_slots, -1); // pooled: owner layer of a slot; fixed: slots 0..7 layer 0, 8..15 layer 1
        std::vector<int32_t> se(n_slots, -1);
        uint64_t hit = 0, tot = 0;
        for (int step = 0; step < 400; ++step) {
            std::vector<int32_t> ids[2];
            for (int e = 0; e < 12; ++e) {
                ids[0].push_back(e);
            }
            ids[1] = { 0, 1 };
            for (int l = 0; l < 2; ++l) {
                if (step >= 100) {
                    for (const int32_t e : ids[l]) {
                        hit += eslot[l][e] >= 0 ? 1 : 0;
                        tot++;
                    }
                }
                llama_moe_decay_count(cnt[l], ids[l].data(), (int64_t) ids[l].size());
            }
            if ((step + 1) % p.every != 0) {
                continue;
            }
            std::vector<llama_moe_decay_swap> sw;
            if (pooled) {
                std::vector<uint8_t> b[2];
                for (int l = 0; l < 2; ++l) {
                    b[l].assign(n_exp, 0);
                    for (int e = 0; e < n_exp; ++e) {
                        b[l][e] = eslot[l][e] >= 0 ? 1 : 0;
                    }
                }
                std::vector<uint8_t> sb(n_slots, 0);
                llama_moe_decay_pairs_pool({ &cnt[0], &cnt[1] }, { &b[0], &b[1] }, sl, se, sb, p, sw);
            } else {
                for (int l = 0; l < 2; ++l) {
                    std::vector<int32_t> lse(se.begin() + 8*l, se.begin() + 8*l + 8);
                    std::vector<uint8_t> lsb(8, 0);
                    std::vector<uint8_t> b(n_exp, 0);
                    for (int e = 0; e < n_exp; ++e) {
                        b[e] = eslot[l][e] >= 0 ? 1 : 0;
                    }
                    std::vector<llama_moe_decay_swap> lw;
                    llama_moe_decay_pairs(l, cnt[l], lse, lsb, b, p, lw);
                    for (auto w : lw) {
                        w.slot += 8*l;
                        sw.push_back(w);
                    }
                }
            }
            llama_moe_decay_order(sw);
            for (const auto & w : sw) {
                const int vl = w.victim_layer >= 0 ? w.victim_layer : w.layer;
                if (w.victim >= 0) {
                    eslot[vl][w.victim] = -1;
                }
                eslot[w.layer][w.expert] = w.slot;
                se[w.slot] = w.expert;
                sl[w.slot] = w.layer;
            }
            for (int l = 0; l < 2; ++l) {
                llama_moe_decay_apply(cnt[l], p.decay);
            }
        }
        hit_rate[pooled] = tot ? (double) hit/tot : 0.0;
    }
    printf("  pool vs fixed: hit rate %.3f pooled, %.3f fixed (8 + 8 slots, working sets 12 and 2)\n", hit_rate[1], hit_rate[0]);
    TCHECK(hit_rate[1] > 0.99 && hit_rate[0] < 0.75, "the pool holds both working sets (%.3f), fixed shares do not (%.3f)",
           hit_rate[1], hit_rate[0]);
}

// [TAG_FN_L3_POLICY_STATE] the saved state's text format: round trip, the model / expert count check, broken lines
static void test_hotstate() {
    const int64_t n_exp = 8;
    std::vector<llama_moe_hotstate_layer> in(2);
    in[0].il = 3;
    in[0].bytes = 3072000;
    in[0].res = { 5, 1, 7 };
    in[0].cnt = { 0.0f, 3.25f, 0.5f, 0.0f, 0.0f, 9.125f, 1e-7f, 2.0f };
    in[1].il = 47;
    in[1].bytes = 3584000;
    in[1].res = {};
    in[1].cnt = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 123456.789f };
    const std::string model = "Qwen3.8 Flash Next";
    const std::string text = llama_moe_hotstate_format(n_exp, 4096, model, in);
    TCHECK(text.rfind("moehot v1 n_expert=8 layers=2 steps=4096 model=Qwen3.8_Flash_Next\n", 0) == 0, "header: %s",
           text.substr(0, 80).c_str());

    std::vector<llama_moe_hotstate_layer> out;
    uint64_t steps = 0;
    std::string why;
    size_t bad = 99;
    {
        std::istringstream is(text);
        TCHECK(llama_moe_hotstate_parse(is, n_exp, model, out, steps, why, &bad), "round trip parses (%s)", why.c_str());
    }
    TCHECK(steps == 4096 && bad == 0 && out.size() == 2, "steps %llu, bad %zu, layers %zu", (unsigned long long) steps, bad, out.size());
    if (out.size() == 2) {
        for (size_t i = 0; i < 2; ++i) {
            TCHECK(out[i].il == in[i].il && out[i].bytes == in[i].bytes && out[i].res == in[i].res, "layer %zu ids", i);
            for (int64_t e = 0; e < n_exp; ++e) {
                const float a = in[i].cnt[e];
                const float b = out[i].cnt[e];
                TCHECK(std::fabs(a - b) <= 1e-5f*std::max(1.0f, std::fabs(a)), "layer %zu count %lld: %g vs %g", i, (long long) e, a, b);
            }
        }
    }
    // another model, another expert count, not a state file: not used
    {
        std::istringstream is(text);
        TCHECK(!llama_moe_hotstate_parse(is, n_exp, "Qwen3.8 27B", out, steps, why), "another model is refused");
    }
    {
        std::istringstream is(text);
        TCHECK(!llama_moe_hotstate_parse(is, 16, model, out, steps, why), "another expert count is refused");
    }
    {
        std::istringstream is("moeprof v1 n_layer=48\n");
        TCHECK(!llama_moe_hotstate_parse(is, n_exp, model, out, steps, why), "a moeprof file is refused");
    }
    // broken layer lines are left out and counted; the good one is kept
    {
        std::string t = "moehot v1 n_expert=8 layers=6 steps=1 model=m\n";
        t += "L 0 100 2 1 2 ; 1 1 1 1 1 1 1 1\n";        // good
        t += "L 1 100 2 1 9 ; 1 1 1 1 1 1 1 1\n";        // id out of range
        t += "L 2 100 1 1 1 1 1 1 1 1 1 1\n";            // no ';'
        t += "L 3 100 1 1 ; 1 1 1 1 1 1 1\n";            // a count missing
        t += "L 4 100 1 1 ; 1 1 1 1 1 1 1 1 7\n";        // one number too many
        t += "L 5 100 1 1 ; 1 1 1 1 -2 1 1 1\n";         // a negative count
        std::istringstream is(t);
        bad = 0;
        TCHECK(llama_moe_hotstate_parse(is, n_exp, "m", out, steps, why, &bad), "header ok");
        TCHECK(out.size() == 1 && out[0].il == 0 && bad == 5, "one good line, 5 broken (got %zu good, %zu bad)", out.size(), bad);
    }
}

// [TAG_FN_L3_POLICY_BURST] the pass selection: the pass budget alone stops at the first pair over it; with a burst budget
// the pairs past it are taken only when strong, until the total
static void test_select() {
    std::vector<llama_moe_decay_swap> sw;
    for (int i = 0; i < 6; ++i) {
        sw.push_back({ 0, i, i, i % 2 ? i + 100 : -1, 10.0f - i });
    }
    auto bytes_of = [](const llama_moe_decay_swap &) { return (size_t) 3; };
    auto strong   = [](const llama_moe_decay_swap & w) { return w.victim < 0; }; // the free-slot pairs: 0, 2, 4
    std::vector<std::pair<size_t, bool>> sel;
    auto ids = [&]() {
        std::string s;
        for (const auto & [i, over] : sel) {
            s += std::to_string(i) + (over ? "*" : "") + " ";
        }
        return s;
    };
    llama_moe_decay_select(sw, 7, 7, bytes_of, strong, sel);
    TCHECK(ids() == "0 1 ", "pass budget only: %s", ids().c_str());
    llama_moe_decay_select(sw, 7, 0, bytes_of, strong, sel);
    TCHECK(ids() == "0 1 ", "a total under the pass budget changes nothing: %s", ids().c_str());
    llama_moe_decay_select(sw, 7, 13, bytes_of, strong, sel);
    TCHECK(ids() == "0 1 2* 4* ", "burst: the strong pairs past the pass budget up to the total: %s", ids().c_str());
    llama_moe_decay_select(sw, 7, 100, bytes_of, [](const llama_moe_decay_swap &) { return false; }, sel);
    TCHECK(ids() == "0 1 ", "burst without a strong pair: %s", ids().c_str());
    llama_moe_decay_select(sw, 0, 0, bytes_of, strong, sel);
    TCHECK(sel.empty(), "no budget: nothing (%s)", ids().c_str());
}

int main() {
    printf("test-moe-decay: [TAG_FN_R4_ADAPT_DECAY]\n");
    test_pairs();
    test_seed();
    test_steady_vs_bursts();
    test_shift();
    test_skewed();
    test_hotstate();
    test_pool_pairs();
    test_pool_vs_fixed();
    test_select();
    printf("test-moe-decay: %d checks, %d errors%s\n", g_checks, g_fail, g_fail ? "" : " - PASSED");
    return g_fail == 0 ? 0 : 1;
}
