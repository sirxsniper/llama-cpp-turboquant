// [TAG_FN_L22] round 22, P9: the server's two-connection policies (tools/server/server-fn-l22-two.h, header only): one
// prompt per batch (ONEPROMPT), the batch cap and window per stream pass (GENSHARE2), the priority pause (PAUSE), the
// zero-copy prefix share (SHAREKV), and the joint draft rule's p_min (DRAFT2, common/speculative.h). CPU only, no model.

#include "../tools/server/server-fn-l22-two.h"
#include "../common/speculative.h"

#include <cstdio>
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

// [TAG_FN_L22_ONEPROMPT] the visit order
static void test_order() {
    using C = l22_prompt_cand;
    TCHECK(l22_oneprompt_order({}).empty(), "empty");
    TCHECK((l22_oneprompt_order({ C{ 1, 900, 0 } }) == std::vector<int>{ 1 }), "one candidate");
    // shortest first
    TCHECK((l22_oneprompt_order({ C{ 0, 30000, 0 }, C{ 1, 900, 0 } }) == std::vector<int>{ 1, 0 }), "shortest first");
    TCHECK((l22_oneprompt_order({ C{ 0, 900, 0 }, C{ 1, 30000, 0 } }) == std::vector<int>{ 0, 1 }), "shortest first, in order");
    // ties by slot id (input order does not matter)
    TCHECK((l22_oneprompt_order({ C{ 3, 500, 0 }, C{ 1, 500, 0 }, C{ 2, 500, 0 } }) == std::vector<int>{ 1, 2, 3 }), "ties by id");
    // a slot that gave way twice goes first, one that gave way once does not
    TCHECK((l22_oneprompt_order({ C{ 0, 30000, 2 }, C{ 1, 900, 0 } }) == std::vector<int>{ 0, 1 }), "forced first");
    TCHECK((l22_oneprompt_order({ C{ 0, 30000, 1 }, C{ 1, 900, 0 } }) == std::vector<int>{ 1, 0 }), "one yield: not forced");
    // two forced slots: fewest left, then id
    TCHECK((l22_oneprompt_order({ C{ 2, 100, 0 }, C{ 0, 30000, 3 }, C{ 1, 20000, 2 } }) == std::vector<int>{ 1, 0, 2 }),
           "forced by left");
    TCHECK((l22_oneprompt_order({ C{ 1, 700, 2 }, C{ 0, 700, 2 }, C{ 2, 10, 0 } }) == std::vector<int>{ 0, 1, 2 }), "forced ties by id");
    // n_force = 1
    TCHECK((l22_oneprompt_order({ C{ 0, 30000, 1 }, C{ 1, 900, 0 } }, 1) == std::vector<int>{ 0, 1 }), "n_force 1");
}

// [TAG_FN_L22_ONEPROMPT] the yield counter
static void test_yield() {
    int32_t n = 0;
    int     last = -1;
    l22_yield_note(n, last, 7);
    TCHECK(n == 1 && last == 7, "first claimer: n = %d", n);
    l22_yield_note(n, last, 7);
    l22_yield_note(n, last, 7);
    TCHECK(n == 1, "the same claimer counts once: n = %d", n);
    l22_yield_note(n, last, -1);
    TCHECK(n == 1 && last == 7, "no claimer: unchanged");
    l22_yield_note(n, last, 9);
    TCHECK(n == 2 && last == 9, "second claimer: n = %d", n);
    // the slot is forced now: a third claimer does not get in front of it
    using C = l22_prompt_cand;
    TCHECK((l22_oneprompt_order({ C{ 0, 40000, n }, C{ 1, 12, 0 } }) == std::vector<int>{ 0, 1 }), "forced after two claimers");
    l22_yield_note(n, last, 11);
    TCHECK(n == 3, "a third claimer still counts: n = %d", n);
}

// [TAG_FN_L22_GENSHARE2] the batch cap while a slot answers
static void test_cap() {
    // off, or nobody answers
    TCHECK(l22_genshare2_prompt_cap(16384, 6, 8192, 0) == 16384, "mode 0");
    TCHECK(l22_genshare2_prompt_cap(16384, 0, 8192, 1) == 16384, "no generating rows");
    TCHECK(l22_genshare2_prompt_cap(16384, 0, 8192, 2) == 16384, "no generating rows, mode 2");
    // mode 1: the generating rows plus the context's ubatch, at most n_batch
    TCHECK(l22_genshare2_prompt_cap(16384, 6, 8192, 1) == 8198, "mode 1, 8K");
    TCHECK(l22_genshare2_prompt_cap(16384, 6, 16384, 1) == 16384, "mode 1, 16K capped");
    TCHECK(l22_genshare2_prompt_cap(16384, 3, 12288, 1) == 12291, "mode 1, 12K");
    // mode 2: at most 8192 prompt rows
    TCHECK(l22_genshare2_prompt_cap(16384, 6, 16384, 2) == 8198, "mode 2 caps 16K");
    TCHECK(l22_genshare2_prompt_cap(16384, 6, 4096, 2) == 4102, "mode 2 keeps a smaller width");
    TCHECK(l22_genshare2_prompt_cap(16384, 6, 0, 2) == 8198, "mode 2, no width");
    // the width clamps to [512, n_batch]; no width = n_batch
    TCHECK(l22_genshare2_prompt_cap(16384, 6, 0, 1) == 16384, "no width");
    TCHECK(l22_genshare2_prompt_cap(16384, 6, -5, 1) == 16384, "negative width");
    TCHECK(l22_genshare2_prompt_cap(16384, 6, 100, 1) == 518, "width below 512");
    TCHECK(l22_genshare2_prompt_cap(256, 6, 100, 1) == 256, "small n_batch");
    TCHECK(l22_genshare2_prompt_cap(16384, 6, 40000, 1) == 16384, "width above n_batch");
    // all: the same cap without generating rows (request P10 R4)
    TCHECK(l22_genshare2_prompt_cap(16384, 0, 12288, 1, true) == 12288, "all, no generating rows");
    TCHECK(l22_genshare2_prompt_cap(16384, 0, 12288, 0, true) == 16384, "all, mode 0");
}

// [TAG_FN_L22_GENSHARE2] the window cap
static void test_window() {
    const int64_t cap = 4000000;
    TCHECK(l22_window_cap_us(cap, 16384, 0) == cap, "mode 0");
    TCHECK(l22_window_cap_us(cap, 100, 1) == cap, "100 rows: x1");
    TCHECK(l22_window_cap_us(cap, 0, 1) == cap, "0 rows: x1");
    TCHECK(l22_window_cap_us(cap, 8192, 1) == cap, "8K: x1");
    TCHECK(l22_window_cap_us(cap, 16383, 1) == cap, "just under 16K: x1");
    TCHECK(l22_window_cap_us(cap, 16384, 1) == 2*cap, "16K: x2");
    TCHECK(l22_window_cap_us(cap, 8192 + 6, 2) == cap, "mode 2, 8K + 6: x1");
}

// [TAG_FN_L22_PAUSE] eligibility and resume
static void test_pause() {
    TCHECK(l22_pause_eligible(2048, 2048, 0), "2048 eligible");
    TCHECK(!l22_pause_eligible(2049, 2048, 0), "2049 not");
    TCHECK(l22_pause_eligible(1, 2048, 1), "second pause");
    TCHECK(!l22_pause_eligible(1, 2048, 2), "third pause refused");
    TCHECK(!l22_pause_eligible(-1, 2048, 0), "negative work");
    TCHECK(l22_pause_eligible(10, 2048, 2, 3), "max_pauses 3");

    const int64_t until = 8000000;
    TCHECK(!l22_pause_should_resume(1000, 0, false), "not paused");
    TCHECK(!l22_pause_should_resume(until - 1, until, true), "before the timeout, the short request alive");
    TCHECK(l22_pause_should_resume(until, until, true), "at the timeout");
    TCHECK(l22_pause_should_resume(until + 5, until, true), "after the timeout");
    TCHECK(l22_pause_should_resume(10, until, false), "the short request is gone");
}

// [TAG_FN_L22_SHAREKV] the follower predicate
static void test_sharekv() {
    l22_share_view v;
    v.task_now  = 41;
    v.task_rec  = 41;
    v.prompting = true;
    v.parked    = false;
    v.n_tokens  = 5900;
    v.n_shared  = 5900;
    v.lcp_task  = 5932;
    v.lcp_ret   = 120;
    v.n_task    = 6400;
    v.same_slot = false;
    v.ret_idle  = true;
    TCHECK(l22_sharekv_usable(v), "valid case");

    auto w = v;
    w.n_tokens = 5901;
    TCHECK(!l22_sharekv_usable(w), "the leader moved past the boundary");
    w = v;
    w.task_now = 42;
    TCHECK(!l22_sharekv_usable(w), "the leader's task changed");
    w = v;
    w.task_rec = -1;
    TCHECK(!l22_sharekv_usable(w), "no record");
    w = v;
    w.prompting = false;
    TCHECK(!l22_sharekv_usable(w), "the leader is not prompting");
    w = v;
    w.parked = true;
    TCHECK(!l22_sharekv_usable(w), "the leader is parked");
    w = v;
    w.same_slot = true;
    TCHECK(!l22_sharekv_usable(w), "same slot");
    w = v;
    w.ret_idle = false;
    TCHECK(!l22_sharekv_usable(w), "the follower's slot is busy");
    w = v;
    w.lcp_task = 5899;
    TCHECK(!l22_sharekv_usable(w), "the follower shares less than the boundary");
    w = v;
    w.lcp_ret = 5900;
    TCHECK(!l22_sharekv_usable(w), "the follower's slot holds the prefix already");
    w = v;
    w.lcp_ret = 5899;
    TCHECK(l22_sharekv_usable(w), "the follower's slot holds a bit less");
    w = v;
    w.n_task = 5900;
    TCHECK(!l22_sharekv_usable(w), "no token of its own after the boundary");
    w = v;
    w.n_shared = 0;
    w.n_tokens = 0;
    TCHECK(!l22_sharekv_usable(w), "empty boundary");
}

// [TAG_FN_L22_SHAREKV] park accounting
static void test_park_cells() {
    TCHECK(l22_park_cells(300, 6200, false) == 300, "not shared: freed");
    TCHECK(l22_park_cells(300, 6200, true) == 6200, "shared: all cells");
    TCHECK(l22_park_cells(7000, 6200, true) == 7000, "shared: max");
    TCHECK(l22_park_cells(0, 0, true) == 0, "nothing");
}

// [TAG_FN_L24_ATTACH] the donor pick
static void test_attach() {
    // slot 0: answering a 30,512-token prompt, checkpoints at 8192, 29,980 (the last user message) and 30,508 (tail)
    l24_donor_view a;
    a.id       = 0;
    a.usable   = true;
    a.idle     = false;
    a.whole_ok = true;
    a.n_tokens = 31200;
    a.lcp      = 30020;
    a.ckpts    = { 8192, 29980, 30508 };
    // the task: 30,100 tokens, its slot keeps nothing
    l24_attach_pick p = l24_attach_best({ a }, 30100, 0, 512, 256);
    TCHECK(p.donor == 0 && p.n == 29980 && p.ckpt == 1, "busy donor: newest checkpoint within the shared part (%d %d %d)",
           p.donor, p.n, p.ckpt);
    // the same slot idle and the task continues all of it: the whole content
    auto b = a;
    b.idle     = true;
    b.lcp      = 31200;
    p = l24_attach_best({ b }, 31260, 0, 512, 256);
    TCHECK(p.donor == 0 && p.n == 31200 && p.ckpt == -1, "idle donor, full continuation: whole content (%d %d)", p.n, p.ckpt);
    // ... but not while answering (its state moves on)
    auto c = b;
    c.idle = false;
    p = l24_attach_best({ c }, 31260, 0, 512, 256);
    TCHECK(p.n == 30508 && p.ckpt == 2, "busy donor, full continuation: the newest checkpoint (%d)", p.n);
    // nor when its memory does not end at its last token
    c = b;
    c.whole_ok = false;
    p = l24_attach_best({ c }, 31260, 0, 512, 256);
    TCHECK(p.n == 30508 && p.ckpt == 2, "memory and tokens disagree: checkpoint (%d)", p.n);
    // the whole content needs one token of the task's own after it
    p = l24_attach_best({ b }, 31200, 0, 512, 256);
    TCHECK(p.n == 30508 && p.ckpt == 2, "the task is exactly the donor's tokens: checkpoint (%d)", p.n);
    // a checkpoint at the task's last token is not usable (one token is evaluated)
    auto d = a;
    d.lcp   = 40000;
    d.ckpts = { 30100 };
    p = l24_attach_best({ d }, 30100, 0, 512, 256);
    TCHECK(p.donor < 0, "checkpoint at n_task: none");
    d.ckpts = { 30099 };
    p = l24_attach_best({ d }, 30100, 0, 512, 256);
    TCHECK(p.n == 30099, "checkpoint at n_task - 1 (%d)", p.n);
    // thresholds: at least n_min, and gain over what the slot keeps itself
    p = l24_attach_best({ a }, 30100, 29800, 512, 256);
    TCHECK(p.donor < 0, "own 29,800: 180 more is under the gain");
    p = l24_attach_best({ a }, 30100, 29700, 512, 256);
    TCHECK(p.n == 29980, "own 29,700: 280 more (%d)", p.n);
    auto e = a;
    e.lcp   = 600;
    e.ckpts = { 400 };
    p = l24_attach_best({ e }, 30100, 0, 512, 256);
    TCHECK(p.donor < 0, "under n_min");
    // unusable donors (parked, the task's slot) and empty ones
    auto f = a;
    f.usable = false;
    p = l24_attach_best({ f }, 30100, 0, 512, 256);
    TCHECK(p.donor < 0, "unusable donor");
    l24_donor_view g;
    p = l24_attach_best({ g }, 30100, 0, 512, 256);
    TCHECK(p.donor < 0, "empty view");
    p = l24_attach_best({ a }, 1, 0, 0, 0);
    TCHECK(p.donor < 0, "one-token task");
    // the longest of two donors; on a tie the checkpoint wins over a live snapshot
    auto h = b;    // idle, whole 31200
    h.id = 1;
    auto k = a;    // busy, checkpoint 31200
    k.id    = 2;
    k.lcp   = 31300;
    k.ckpts = { 31200 };
    p = l24_attach_best({ h, k }, 31300, 0, 512, 256);
    TCHECK(p.donor == 2 && p.n == 31200 && p.ckpt == 0, "tie: checkpoint (%d %d %d)", p.donor, p.n, p.ckpt);
    p = l24_attach_best({ k, h }, 31300, 0, 512, 256);
    TCHECK(p.donor == 2 && p.ckpt == 0, "tie, other order: checkpoint (%d %d)", p.donor, p.ckpt);
    auto m = a;
    m.id  = 3;
    m.lcp = 9000;
    p = l24_attach_best({ m, a }, 30100, 0, 512, 256);
    TCHECK(p.donor == 0 && p.n == 29980, "longest donor (%d %d)", p.donor, p.n);
}

// [TAG_FN_L22_DRAFT2] the joint draft rule's p_min
static void test_draft2() {
    TCHECK(common_speculative_pmin_draft2(0.2f, 0, 2, 0.6f) == 0.2f, "position 0 never raised");
    TCHECK(common_speculative_pmin_draft2(0.2f, 1, 1, 0.6f) == 0.2f, "one drafting sequence never raised");
    TCHECK(common_speculative_pmin_draft2(0.2f, 1, 2, 0.0f) == 0.2f, "draft2 0: identity");
    TCHECK(common_speculative_pmin_draft2(0.2f, 1, 2, -1.0f) == 0.2f, "negative draft2: identity");
    TCHECK(common_speculative_pmin_draft2(0.2f, 1, 2, 0.6f) == 0.6f, "position 2 raised");
    TCHECK(common_speculative_pmin_draft2(0.2f, 3, 4, 0.6f) == 0.6f, "later positions raised");
    TCHECK(common_speculative_pmin_draft2(0.85f, 2, 2, 0.6f) == 0.85f, "max: a higher position p_min stays");
}

// [TAG_FN_L24_OPFAIR] the fair order and the near-end rule; [TAG_FN_L24_GSCAP] the window cap
static void test_l24() {
    auto first = [](const std::vector<l24_prompt_cand> & c) {
        const auto o = l24_oneprompt_order(c);
        return o.empty() ? -1 : o[0];
    };
    // A (slot 1) in progress, 15.5K left of 30K, never gave way; B (slot 0) new with 11.4K: B may pre-empt once
    TCHECK(first({ { 0, 11389, 0, false, false }, { 1, 15536, 0, true, false } }) == 0, "one pre-emption allowed");
    // A gave way once in this request: it keeps the batch now
    TCHECK(first({ { 0, 11389, 0, false, false }, { 1, 15536, 1, true, false } }) == 1, "at most once per request");
    // A near its end keeps the batch
    TCHECK(first({ { 0, 2000, 0, false, false }, { 1, 6000, 0, true, true } }) == 1, "near the end: finishes first");
    // B, the pre-emptor, is now the claimer: A (more left, gave way) does not take the batch back
    TCHECK(first({ { 0, 9000, 0, true, false }, { 1, 15536, 1, false, false } }) == 0, "no ping-pong");
    // a challenger with more tokens left never pre-empts
    TCHECK(first({ { 0, 20000, 0, false, false }, { 1, 15536, 0, true, false } }) == 1, "shortest first only");
    // no claimer: a slot that gave way goes first (n_force 1), else the fewest left
    TCHECK(first({ { 0, 3000, 0, false, false }, { 1, 15536, 1, false, false } }) == 1, "gave way: first");
    TCHECK(first({ { 0, 3000, 0, false, false }, { 1, 15536, 0, false, false } }) == 0, "fewest left");
    // a claimer with nothing left is no claimer
    TCHECK(first({ { 0, 3000, 0, false, false }, { 1, 0, 0, true, false } }) == 1, "left 0 sorts first, as before");
    {
        const auto o = l24_oneprompt_order({ { 2, 500, 0, false, false }, { 0, 9000, 0, true, false }, { 1, 700, 0, false, false } });
        TCHECK(o.size() == 3 && o[0] == 2 && o[1] == 0 && o[2] == 1, "three slots: pre-emptor, claimer, rest (%zu)", o.size());
        std::vector<int> s = o;
        std::sort(s.begin(), s.end());
        TCHECK(s == std::vector<int>({ 0, 1, 2 }), "every slot once");
    }
    // near end
    TCHECK(l24_prompt_near_end(13616, 30000, false, 16384, 0.75), "one more 16K pass of text");
    TCHECK(!l24_prompt_near_end(13616, 30000, true, 16384, 0.75), "pictures left: by the fraction only");
    TCHECK(l24_prompt_near_end(6000, 30000, true, 16384, 0.75), "80 percent done");
    TCHECK(!l24_prompt_near_end(20000, 30000, false, 16384, 0.75), "a third done");
    TCHECK(l24_prompt_near_end(0, 30000, false, 0, 1.0), "nothing left");
    TCHECK(!l24_prompt_near_end(5, 0, false, 0, 0.75), "no turn rows: by tokens only");
    // window cap
    TCHECK(l24_window_cap_us(4000000, 16381) > 7990000 && l24_window_cap_us(4000000, 16381) < 8000000, "16K pass: ~2x the cap");
    TCHECK(l24_window_cap_us(4000000, 100) == 4000000, "at least the cap");
    TCHECK(l24_window_cap_us(4000000, 8192) == 4000000, "8K: the cap");
    TCHECK(l22_window_cap_us(4000000, 16381, 1) == 4000000, "the old cap counted whole 8Ks");
}

int main() {
    test_l24();
    test_order();
    test_yield();
    test_cap();
    test_window();
    test_pause();
    test_sharekv();
    test_park_cells();
    test_attach();
    test_draft2();

    fprintf(stderr, "test-fn-l22-two: %d checks, %d failures\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
