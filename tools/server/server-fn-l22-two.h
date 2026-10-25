#pragma once

// [TAG_FN_L22] round 22, P9: pure policy functions of the server's two-connection rules (no llama / server types), so
// that tests/test-fn-l22-two.cpp checks them without a model.
//
//   l22_oneprompt_order        the visit order of the prompt slots                      [TAG_FN_L22_ONEPROMPT]
//   l22_yield_note             a yield counts once per claiming task                    [TAG_FN_L22_ONEPROMPT]
//   l22_genshare2_prompt_cap   the batch limit while a slot answers                     [TAG_FN_L22_GENSHARE2]
//   l22_window_cap_us          the decode window cap of a stream pass                   [TAG_FN_L22_GENSHARE2]
//   l22_pause_eligible         a short request may pause an answer                      [TAG_FN_L22_PAUSE]
//   l22_pause_should_resume    when a paused answer goes on                             [TAG_FN_L22_PAUSE]
//   l22_sharekv_usable         a follower may take the leader's prefix by seq_cp        [TAG_FN_L22_SHAREKV]
//   l22_park_cells             the cells a parked slot needs again                      [TAG_FN_L22_SHAREKV]
//   l24_attach_best            the other slot whose cells a new prompt starts from      [TAG_FN_L24_ATTACH]

#include <algorithm>
#include <cstdint>
#include <vector>

// [TAG_FN_L22_ONEPROMPT] one slot with prompt tokens left: its id, the tokens left, the claimers it gave way to
struct l22_prompt_cand {
    int     id      = -1;
    int32_t left    = 0;
    int32_t n_yield = 0;
};

// a slot that gave way n_force times goes first; then the fewest tokens left; then the slot id
static inline std::vector<int> l22_oneprompt_order(std::vector<l22_prompt_cand> c, int32_t n_force = 2) {
    std::stable_sort(c.begin(), c.end(), [n_force](const l22_prompt_cand & a, const l22_prompt_cand & b) {
        const bool fa = a.n_yield >= n_force;
        const bool fb = b.n_yield >= n_force;
        if (fa != fb) {
            return fa;
        }
        if (a.left != b.left) {
            return a.left < b.left;
        }
        return a.id < b.id;
    });
    std::vector<int> out;
    out.reserve(c.size());
    for (const auto & x : c) {
        out.push_back(x.id);
    }
    return out;
}

// [TAG_FN_L24_OPFAIR] one slot with prompt tokens left, with what the fair order needs
struct l24_prompt_cand {
    int     id       = -1;
    int32_t left     = 0;
    int32_t n_yield  = 0;
    bool    claimer  = false; // its task claimed the last prompt batch (the prompt in progress)
    bool    near_end = false; // its rest is short or most of it is done: it finishes first
};

// whether a prompt is near its end: a text-only rest of at most near tokens, or at least frac of its turn done
static inline bool l24_prompt_near_end(int32_t left, int32_t turn_rows, bool media_left, int32_t near, double frac) {
    if (left <= 0) {
        return true;
    }
    if (!media_left && left <= near) {
        return true;
    }
    return turn_rows > 0 && (double) (turn_rows - left) >= frac * (double) turn_rows;
}

// The visit order with the prompt in progress kept (owner 10-24: a new prompt took the batch from one with 15.5K left,
// which then waited ~45 s). Shortest remaining first, with limits: the claimer goes first unless a challenger has fewer
// tokens left, the claimer never gave way in this request and is not near its end (the near-end rule is off by default:
// the short request matters more). The pre-emptor then keeps the batch (no ping-pong). Without a claimer, a slot that
// gave way goes first, then the fewest tokens left (l22_oneprompt_order with n_force 1). Every slot of c is in the result once.
static inline std::vector<int> l24_oneprompt_order(const std::vector<l24_prompt_cand> & c) {
    std::vector<l22_prompt_cand> base;
    for (const auto & x : c) {
        base.push_back({ x.id, x.left, x.n_yield });
    }
    std::vector<int> order = l22_oneprompt_order(base, 1);
    int k = -1;
    for (int i = 0; i < (int) c.size(); ++i) {
        if (c[i].claimer && c[i].left > 0) {
            k = i;
            break;
        }
    }
    if (k < 0) {
        return order;
    }
    const auto & cl = c[k];
    int best = -1; // the first challenger in the base order
    for (const int id : order) {
        if (id != cl.id) {
            best = id;
            break;
        }
    }
    int32_t best_left = 0;
    for (const auto & x : c) {
        if (x.id == best) {
            best_left = x.left;
        }
    }
    const bool preempt = best >= 0 && !cl.near_end && cl.n_yield == 0 && best_left < cl.left;
    std::vector<int> out;
    out.reserve(order.size());
    if (preempt) {
        out.push_back(best);
    }
    out.push_back(cl.id);
    for (const int id : order) {
        if (id != cl.id && !(preempt && id == best)) {
            out.push_back(id);
        }
    }
    return out;
}

// [TAG_FN_L24_GSCAP] l22_window_cap_us with the rows counted in fractions of 8K: a 16,381-row pass got the 8K cap (4 s),
// so its window was 2.0 s of a 6.6 s pass (share 0.30, the answer ~21 t/s) instead of half the pass
static inline int64_t l24_window_cap_us(int64_t cap_us, int64_t prompt_rows) {
    return (int64_t) ((double) cap_us * std::max(1.0, (double) prompt_rows / 8192.0));
}

// the slot gave way to claimer_task in this batch: counts once per claiming task (a long prompt gives way to a shorter
// one at most n_force times)
static inline void l22_yield_note(int32_t & n_yield, int & last_task, int claimer_task) {
    if (claimer_task < 0 || claimer_task == last_task) {
        return;
    }
    last_task = claimer_task;
    n_yield++;
}

// [TAG_FN_L22_GENSHARE2] the batch size the prompt rows may fill: the generating rows plus one context prompt ubatch
// (w; mode 2: at most 8192). mode 0 or no generating rows (unless all): n_batch, as before
static inline int32_t l22_genshare2_prompt_cap(int32_t n_batch, int32_t n_gen_rows, int32_t w, int mode, bool all = false) {
    if (mode <= 0 || n_batch <= 0 || (n_gen_rows <= 0 && !all)) {
        return n_batch;
    }
    if (w <= 0) {
        w = n_batch;
    }
    if (mode == 2) {
        w = std::min<int32_t>(w, 8192);
    }
    w = std::clamp<int32_t>(w, std::min<int32_t>(512, n_batch), n_batch);
    const int64_t cap = (int64_t) std::max<int32_t>(0, n_gen_rows) + w;
    return (int32_t) std::min<int64_t>(n_batch, cap);
}

// the decode window cap after a batch that held prompt_rows prompt rows: cap_us per 8K rows (at least one)
static inline int64_t l22_window_cap_us(int64_t cap_us, int64_t prompt_rows, int mode) {
    if (mode <= 0) {
        return cap_us;
    }
    return cap_us * std::max<int64_t>(1, prompt_rows / 8192);
}

// [TAG_FN_L22_PAUSE] a request with work prompt tokens to process may pause an answer that paused fewer than
// max_pauses times
static inline bool l22_pause_eligible(int32_t work, int32_t max_tok, int32_t n_pauses, int32_t max_pauses = 2) {
    return work >= 0 && work <= max_tok && n_pauses < max_pauses;
}

// a paused answer (until != 0) goes on at its timeout or when the request that paused it is gone
static inline bool l22_pause_should_resume(int64_t now, int64_t until, bool by_alive) {
    return until != 0 && (now >= until || !by_alive);
}

// [TAG_FN_L22_SHAREKV] what get_available_slot knows when a follower arrives (the leader L, the published record, the
// follower's slot ret)
struct l22_share_view {
    int     task_now  = -1;    // L's task id now
    int     task_rec  = -1;    // the task id the record was published for (-1: no record)
    bool    prompting = false; // L is in PROCESSING_PROMPT
    bool    parked    = false; // L is parked
    int32_t n_tokens  = 0;     // L's prompt tokens now
    int32_t n_shared  = 0;     // the published boundary
    int32_t lcp_task  = 0;     // common prefix of the follower's prompt and L's prompt
    int32_t lcp_ret   = 0;     // common prefix of the follower's prompt and what ret holds now
    int32_t n_task    = 0;     // the follower's prompt tokens
    bool    same_slot = false; // ret is L
    bool    ret_idle  = false; // ret processes nothing
};

// the leader still sits exactly on the boundary of the record, the follower shares the whole boundary and has tokens of
// its own after it, and its slot holds less of that prefix
static inline bool l22_sharekv_usable(const l22_share_view & v) {
    return v.task_rec >= 0 && v.task_now == v.task_rec && v.prompting && !v.parked && v.n_shared > 0 &&
           v.n_tokens == v.n_shared && v.lcp_task >= v.n_shared && v.n_task > v.n_shared && !v.same_slot && v.ret_idle &&
           v.lcp_ret < v.n_shared;
}

// a park of a slot that shares cells with another sequence frees only its own cells, but the resume writes all of them
static inline int32_t l22_park_cells(int32_t freed, int32_t n_tokens, bool shared) {
    return shared ? std::max(freed, n_tokens) : freed;
}

// [TAG_FN_L24_ATTACH] (upstream #30970 --slot-attach, automatic) another slot (the donor) as get_available_slot sees it
// for a new task: a donor that is idle, still in its prompt or answering lends the task's slot its cells (seq_cp, no copy)
// up to a point where the recurrent state is known exactly: one of its checkpoints, or (idle only) its whole content
struct l24_donor_view {
    int     id       = -1;
    bool    usable   = false;    // not parked, IDLE / PROCESSING_PROMPT / GENERATING, not the task's slot
    bool    idle     = false;    // IDLE: its current state is final, the whole content may be lent (a live snapshot)
    bool    whole_ok = false;    // its memory ends exactly at its last token (seq_pos_max + 1 == pos_next)
    int32_t n_tokens = 0;        // its tokens
    int32_t lcp      = 0;        // common prefix of its tokens and the task's tokens
    std::vector<int64_t> ckpts;  // n_tokens of its checkpoints that hold data and lie within its memory
};

// what to lend: the donor's id (-1: nothing), the tokens, and which state (-1: the donor's live state, else ckpts[ckpt])
struct l24_attach_pick {
    int     donor = -1;
    int32_t n     = 0;
    int     ckpt  = -1;
};

// The longest prefix any donor can lend: its whole content (idle, the task continues all of it) or its newest checkpoint
// within the shared part; at most n_task - 1 tokens (one prompt token is always evaluated [TAG_PROMPT_LOGITS]). A
// checkpoint wins a tie (one host->device copy instead of a snapshot and a copy). Taken only with at least n_min tokens
// and gain tokens more than own (what the task's slot keeps by itself).
static inline l24_attach_pick l24_attach_best(const std::vector<l24_donor_view> & donors, int32_t n_task, int32_t own,
                                              int32_t n_min, int32_t gain) {
    l24_attach_pick best;
    const int32_t n_max = n_task - 1;
    if (n_max <= 0) {
        return best;
    }
    for (const auto & d : donors) {
        if (!d.usable || d.id < 0) {
            continue;
        }
        const int32_t lim = std::min(d.lcp, n_max);
        for (size_t k = 0; k < d.ckpts.size(); ++k) {
            const int64_t c = d.ckpts[k];
            if (c > 0 && c <= lim && (c > best.n || (c == best.n && best.ckpt < 0 && best.donor >= 0))) {
                best = { d.id, (int32_t) c, (int) k };
            }
        }
        if (d.idle && d.whole_ok && d.n_tokens > 0 && d.lcp >= d.n_tokens && d.n_tokens <= n_max && d.n_tokens > best.n) {
            best = { d.id, d.n_tokens, -1 };
        }
    }
    if (best.donor < 0 || best.n < std::max<int32_t>(1, n_min) || best.n < (int64_t) own + std::max<int32_t>(0, gain)) {
        return l24_attach_pick();
    }
    return best;
}
