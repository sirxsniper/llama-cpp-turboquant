// [TAG_FN_L22] round 22, P3: the server's pure policies (tools/server/server-fn-l22-p3.h, header only): where a merged
// image-turn batch ends (IMGPASS), media anchors (MEDIA_ANCHOR), the checkpoint ring's victim with anchors and the keep
// rule (CKPT_KEEP), the decode-phase rule (PHASE). CPU only, no model.

#include "../tools/server/server-convstore.h"
#include "../tools/server/server-fn-l22-p3.h"

#include <algorithm>
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

//
// [TAG_FN_L22] round 22, P3 (tools/server/server-fn-l22-p3.h)
//

// [TAG_FN_L22_IMGPASS] where a merged image-turn batch ends
static void test_l22_media_plan() {
    using S = l22_media_seg;
    // the whole rest fits
    TCHECK(l22_media_plan({ S{false, 5}, S{true, 1024}, S{false, 3000} }, 4096, 8192) == 4029, "whole rest");
    // the rest ends with a media chunk: per chunk as today
    TCHECK(l22_media_plan({ S{false, 5}, S{true, 1024} }, 4096, 8192) == 0, "ends with media");
    // no media at all
    TCHECK(l22_media_plan({ S{false, 500} }, 4096, 8192) == 0, "text only");
    // the last media boundary at or before W (the end of the second image)
    TCHECK(l22_media_plan({ S{true, 1000}, S{false, 50}, S{true, 1000}, S{false, 7000} }, 8192, 8192) == 2050, "end of image 2");
    // the boundary before a chunk that does not fit
    TCHECK(l22_media_plan({ S{false, 100}, S{true, 1000}, S{false, 20}, S{true, 900}, S{false, 30} }, 1500, 8192) == 1120,
           "before image 2");
    // the next batch of the same turn
    TCHECK(l22_media_plan({ S{true, 900}, S{false, 30} }, 1500, 8192) == 930, "rest of the turn");
    // text up to W with the first media beyond it: no merged batch (the text path stops at the media)
    TCHECK(l22_media_plan({ S{false, 1500}, S{true, 1000}, S{false, 10} }, 2048, 8192) == 0, "text first, media beyond W");
    TCHECK(l22_media_plan({ S{false, 3000}, S{true, 1000}, S{false, 10} }, 2048, 8192) == 0, "text beyond W");
    // a first chunk wider than W runs alone up to cap
    TCHECK(l22_media_plan({ S{true, 4000}, S{false, 10}, S{true, 500}, S{false, 5} }, 2048, 8192) == 4000, "wide first chunk");
    // a first chunk wider than cap: per chunk
    TCHECK(l22_media_plan({ S{true, 9000}, S{false, 10} }, 2048, 8192) == 0, "chunk over cap");
    // cap below W binds
    TCHECK(l22_media_plan({ S{true, 1000}, S{false, 10}, S{true, 1000}, S{false, 10} }, 8192, 1500) == 1010, "cap binds");
    TCHECK(l22_media_plan({}, 8192, 8192) == 0, "empty");
    TCHECK(l22_media_plan({ S{true, 10}, S{false, 1} }, 0, 8192) == 0, "no width");

    // random turns: the plan never splits a chunk, never passes W except for a lone first chunk, never passes cap, and
    // ends at a segment edge
    std::mt19937 rng(22);
    for (int it = 0; it < 20000; ++it) {
        std::vector<S> segs;
        const int n = 1 + (int) (rng() % 12);
        bool media = rng() % 2;
        for (int i = 0; i < n; ++i) {
            segs.push_back(S{ media, media ? (int64_t) (300 + rng() % 4000) : (int64_t) (1 + rng() % 3000) });
            media = !media;
        }
        const int64_t W   = 512 + (int64_t) (rng() % 16384);
        const int64_t cap = 1024 + (int64_t) (rng() % 16384);
        const int64_t r   = l22_media_plan(segs, W, cap);
        if (r == 0) {
            continue;
        }
        TCHECK(!segs.back().media, "random %d: merged although the rest ends with media", it);
        TCHECK(r <= cap, "random %d: %lld rows over cap %lld", it, (long long) r, (long long) cap);
        int64_t acc = 0;
        bool    edge = false;
        bool    has_media = false;
        for (const auto & s : segs) {
            if (acc == r) {
                edge = true;
                break;
            }
            if (acc + s.n_rows > r) {
                break; // r falls inside s
            }
            acc += s.n_rows;
            has_media |= s.media;
        }
        edge = edge || acc == r;
        TCHECK(edge, "random %d: %lld rows end inside a segment", it, (long long) r);
        TCHECK(has_media, "random %d: a merged batch without media", it);
        const bool lone_first = segs[0].media && r == segs[0].n_rows;
        TCHECK(r <= std::min(W, cap) || lone_first, "random %d: %lld rows over W %lld", it, (long long) r, (long long) W);
    }
}

// [TAG_FN_L22_MEDIA_ANCHOR]
static void test_l22_anchor() {
    // system 0..5000, user 5000 (picture at 5100), assistant 7000, user 9000 (text only), assistant 9500, user 12000
    // (picture at 12010), assistant 14000
    const std::vector<l22_span> spans = {
        { 0, 5000, false }, { 5000, 2000, true }, { 7000, 2000, false }, { 9000, 500, true },
        { 9500, 2500, false }, { 12000, 2000, true }, { 14000, 100, false },
    };
    const std::vector<int64_t> media = { 5100, 12010 };
    TCHECK(l22_ckpt_is_anchor(5000, spans, media, 8), "picture message");
    TCHECK(l22_ckpt_is_anchor(4996, spans, media, 8), "a few tokens before the picture message (TURN_MERGE)");
    TCHECK(!l22_ckpt_is_anchor(4990, spans, media, 8), "too far before");
    TCHECK(!l22_ckpt_is_anchor(9000, spans, media, 8), "text-only user message");
    TCHECK(l22_ckpt_is_anchor(12000, spans, media, 8), "second picture message");
    TCHECK(!l22_ckpt_is_anchor(7000, spans, media, 8), "assistant start");
    TCHECK(!l22_ckpt_is_anchor(5101, spans, media, 8), "inside the message");
    // len unknown: up to the next span
    const std::vector<l22_span> spans0 = { { 0, 0, false }, { 100, 0, true }, { 300, 0, false } };
    TCHECK(l22_ckpt_is_anchor(100, spans0, { 250 }, 8), "len 0: next span bounds the message");
    TCHECK(!l22_ckpt_is_anchor(100, spans0, { 300 }, 8), "len 0: media of the next message");
    TCHECK(!l22_ckpt_is_anchor(100, {}, { 150 }, 8), "no spans");
}

// [TAG_FN_L22_CKPT_KEEP] [TAG_FN_L22_MEDIA_ANCHOR]
static void test_l22_victim() {
    // without anchors and keep_at, the keeping rule is convstore_ckpt_victim's
    std::mt19937 rng(23);
    for (int it = 0; it < 20000; ++it) {
        const int n = 2 + (int) (rng() % 8);
        std::vector<int64_t> ns;
        int64_t p = (int64_t) (rng() % 100);
        for (int i = 0; i < n; ++i) {
            ns.push_back(p);
            p += 1 + (int64_t) (rng() % 5000);
        }
        const int64_t sys_end   = (rng() % 3 == 0) ? -1 : ns[rng() % n];
        const int64_t last_user = (rng() % 3 == 0) ? -1 : ns[n - 1] - (int64_t) (rng() % 3000);
        const int v0 = convstore_ckpt_victim(ns, sys_end, last_user);
        const int v1 = convstore_ckpt_victim_l22(ns, {}, true, sys_end, last_user, {});
        TCHECK(v0 == v1, "random %d: keeping rule %d vs %d", it, v0, v1);
        // FIFO without the rule and anchors: the oldest
        TCHECK(convstore_ckpt_victim_l22(ns, {}, false, sys_end, last_user, {}) == 0, "random %d: FIFO", it);
        // with random anchors: never the incoming; a non-anchor unprotected entry goes before any anchor
        std::vector<char> an(n, 0);
        for (int i = 0; i < n; ++i) {
            an[i] = rng() % 3 == 0;
        }
        for (const bool rule : { false, true }) {
            const int v = convstore_ckpt_victim_l22(ns, an, rule, sys_end, last_user, {});
            TCHECK(v >= 0 && v < n - 1, "random %d: victim %d of %d", it, v, n);
            const std::vector<char> prot = rule ? convstore_ckpt_protect(ns, sys_end, last_user, {}) : std::vector<char>(n, 0);
            bool plain = false;
            for (int i = 0; i < n - 1; ++i) {
                plain |= !prot[i] && !an[i];
            }
            if (plain) {
                TCHECK(!an[v] && !prot[v], "random %d: dropped an anchor / protected entry %d while a plain one exists", it, v);
            }
            // the oldest anchor goes only when nothing else can
            int a_old = -1;
            int a_new = -1;
            for (int i = 0; i < n; ++i) {
                if (an[i]) {
                    if (a_old < 0) {
                        a_old = i;
                    }
                    a_new = i;
                }
            }
            if (v == a_old || v == a_new) {
                for (int i = 0; i < n - 1; ++i) {
                    TCHECK(i == a_old || i == a_new, "random %d: kept anchor %d dropped while %d could go", it, v, i);
                }
            }
        }
    }

    // keep_at: the checkpoint at a user start 1-2 turns back survives
    {
        // ring: system 8000, user starts 20000 / 26000 (1-2 turns back), tail 31996; incoming 32000; last user 30000
        const std::vector<int64_t> ns = { 8000, 20000, 26000, 31996, 32000 };
        const int v_old = convstore_ckpt_victim(ns, 8000, 30000);
        const int v_new = convstore_ckpt_victim_l22(ns, {}, true, 8000, 30000, { 26000, 20000 });
        TCHECK(v_old == 1 || v_old == 2 || v_old == 3, "old victim %d", v_old);
        TCHECK(v_new == 3, "keep_at: victim %d, want 3 (31996: not a user start 1-2 turns back)", v_new);
    }
    TCHECK(l22_prev_user_starts({ 100, 200, 300, 400 }, 2) == std::vector<int64_t>({ 300, 200 }), "prev user starts");
    TCHECK(l22_prev_user_starts({ 100 }, 2).empty(), "one user message");

    // a Kyro-like session: a picture message every second turn, ring of 4. The oldest picture message's checkpoint (where
    // a picture-budget cut lands) stays in the ring or goes to disk as an anchor, the newest stays.
    {
        const size_t n_max = 4;
        std::vector<int64_t> ring;
        std::vector<char>    ring_an;
        int64_t n = 9000; // system block
        int64_t first_anchor = -1;
        std::mt19937 r2(24);
        for (int turn = 0; turn < 40; ++turn) {
            const int64_t user = n;
            const bool    pic  = turn % 2 == 1;
            const int64_t len  = (pic ? 4000 : 300) + (int64_t) (r2() % 2000);
            const int64_t tail = user + len - 4;
            if (pic && first_anchor < 0) {
                first_anchor = user;
            }
            for (const int64_t pos : { user, tail }) {
                const bool an = pos == user && pic;
                if (ring.size() >= n_max) {
                    std::vector<int64_t> ns = ring;
                    ns.push_back(pos);
                    std::vector<char> va = ring_an;
                    va.push_back(an);
                    const int v = convstore_ckpt_victim_l22(ns, va, true, 9000, user, {});
                    TCHECK(v >= 0 && v < (int) ring.size(), "kyro turn %d: victim %d", turn, v);
                    ring.erase(ring.begin() + v);
                    ring_an.erase(ring_an.begin() + v);
                }
                ring.push_back(pos);
                ring_an.push_back(an);
            }
            if (first_anchor >= 0) {
                TCHECK(std::find(ring.begin(), ring.end(), first_anchor) != ring.end(),
                       "kyro turn %d: the oldest picture message's checkpoint (%lld) left the ring", turn, (long long) first_anchor);
            }
            n = user + len;
        }
    }
    TCHECK(l22_phase_gen(3, 0), "generation rows only");
    TCHECK(!l22_phase_gen(3, 1), "a prompt row");
    TCHECK(!l22_phase_gen(0, 0), "empty");
}

static void test_l24_merged() {
    // the owner's 2-slot pool: 262,144 cells, 6 answer rows per step
    TCHECK(l24_merged_fits(262144 - 120000, 16384, 6), "room: idle slot with 120K cells");
    TCHECK(!l24_merged_fits(10000, 16384, 6), "no room: per-chunk path");
    TCHECK(!l24_merged_fits(16384, 16384, 6), "rows fit, the answers' next step does not");
    TCHECK(l24_merged_fits(16390, 16384, 6), "exactly rows + margin");
    TCHECK(!l24_merged_fits(-1, 100, 0), "no count from the memory");
    TCHECK(!l24_merged_fits(1000, 0, 0), "no rows");
    TCHECK(l24_merged_fits(100, 100, -5), "negative margin counts as 0");
    TCHECK(l24_merged_batch_rule(0, 0, false, false) == 0, "empty batch: merge (switch off)");
    TCHECK(l24_merged_batch_rule(0, 0, false, true) == 0, "empty batch: merge");
    TCHECK(l24_merged_batch_rule(3, 3, false, true) == 1, "answer rows only: next batch alone");
    TCHECK(l24_merged_batch_rule(3, 3, false, false) == -1, "answer rows only, switch off: per-chunk");
    TCHECK(l24_merged_batch_rule(500, 3, false, true) == -1, "prompt rows in the batch: per-chunk");
    TCHECK(l24_merged_batch_rule(500, 0, false, true) == -1, "another prompt claimed the batch");
    TCHECK(l24_merged_batch_rule(500, 0, true, true) == -1, "another merged batch");
}

int main() {
    test_l22_media_plan();
    test_l22_anchor();
    test_l22_victim();
    test_l24_merged();
    printf("test-fn-l22-p3: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
