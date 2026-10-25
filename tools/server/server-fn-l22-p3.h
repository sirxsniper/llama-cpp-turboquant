#pragma once

// [TAG_FN_L22] round 22, P3: pure policy functions of the server's image turns and its checkpoint ring (no llama / server
// types), so that tests/test-fn-convstore.cpp checks them without a model.
//
//   l22_media_plan             where a merged image-turn batch ends                     [TAG_FN_L22_IMGPASS]
//   l22_ckpt_is_anchor         whether a checkpoint is a media anchor                   [TAG_FN_L22_MEDIA_ANCHOR]
//   convstore_ckpt_protect     the round-19 keeping rule's protected entries            [TAG_FN_L19_CONVSTORE_CKPT]
//   convstore_ckpt_victim_l22  which checkpoint a full ring drops (anchors, keep rule)   [TAG_FN_L22_CKPT_KEEP]
//   l22_prev_user_starts       the user-message starts 1-2 turns back                   [TAG_FN_L22_CKPT_KEEP]
//   l22_phase_gen              the decode-phase rule of a batch                         [TAG_FN_L22_PHASE]

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

// [TAG_FN_L22_IMGPASS] one segment of the rest of a prompt: a text run or one media chunk (never split)
struct l22_media_seg {
    bool    media  = false;
    int64_t n_rows = 0;
};

// segs: the rest of the prompt from the current position, in order. W: the preferred batch width (the context's next
// prompt ubatch), cap: the hard limit (the server batch, the explicit-mask budget).
// Returns the rows the merged batch takes, 0 for no merged batch:
//   - the whole rest when it fits in W;
//   - else up to the last media boundary (the start or the end of a chunk) at or before W;
//   - a media chunk that comes first may exceed W, up to cap (alone);
//   - 0 when the batch would hold no media chunk (the text path takes the text up to the media), when the first media
//     chunk exceeds cap, or when the rest ends with a media chunk (the per-chunk path keeps that case as today).
static inline int64_t l22_media_plan(const std::vector<l22_media_seg> & segs, int64_t W, int64_t cap) {
    if (segs.empty() || W <= 0 || cap <= 0 || segs.back().media) {
        return 0;
    }
    W = std::min(W, cap);

    int64_t total     = 0;
    bool    any_media = false;
    for (const auto & s : segs) {
        total     += s.n_rows;
        any_media |= s.media;
    }
    if (!any_media) {
        return 0;
    }
    if (total <= W) {
        return total;
    }

    int64_t rows      = 0;
    int64_t best      = 0;
    bool    has_media = false;
    for (const auto & s : segs) {
        if (s.media) {
            if (has_media && rows <= W) {
                best = rows; // the boundary before this chunk
            }
            if (rows + s.n_rows > W) {
                if (rows == 0 && s.n_rows <= cap) {
                    return s.n_rows; // the first item alone
                }
                break;
            }
            rows     += s.n_rows;
            has_media = true;
            best      = rows; // the boundary after this chunk
        } else {
            if (rows + s.n_rows > W) {
                break;
            }
            rows += s.n_rows;
        }
    }
    return has_media ? best : 0;
}

// [TAG_FN_L22_MEDIA_ANCHOR] one message span of a prompt (token indices; len to the next message delimiter)
struct l22_span {
    int64_t pos  = 0;
    int64_t len  = 0;
    bool    user = false;
};

// A checkpoint at n_tokens n is a media anchor when a user message starts at n, or at most merge_max tokens after n (the
// TURN_MERGE rule lets a checkpoint a few tokens before a user start stand for it), and a media chunk starts inside that
// message. spans: the prompt's message spans in order; media: the start indices of its media chunks, ascending.
static inline bool l22_ckpt_is_anchor(int64_t n, const std::vector<l22_span> & spans, const std::vector<int64_t> & media,
                                      int64_t merge_max) {
    for (size_t i = 0; i < spans.size(); ++i) {
        if (!spans[i].user) {
            continue;
        }
        const int64_t p = spans[i].pos;
        if (p < n || p > n + merge_max) {
            continue;
        }
        int64_t end = std::numeric_limits<int64_t>::max();
        if (spans[i].len > 0) {
            end = p + spans[i].len;
        } else if (i + 1 < spans.size()) {
            end = spans[i + 1].pos;
        }
        const auto it = std::lower_bound(media.begin(), media.end(), p);
        if (it != media.end() && *it < end) {
            return true;
        }
    }
    return false;
}

// [TAG_FN_L19_CONVSTORE_CKPT] the entries of ns the round-19 keeping rule protects (the same rule as
// convstore_ckpt_victim in server-convstore.h): the end of the system + tools block, the last user message, about halfway
// back, the newest (always the last of ns). keep_at ([TAG_FN_L22_CKPT_KEEP]): the newest checkpoint at or before each
// of these positions is protected too.
static inline std::vector<char> convstore_ckpt_protect(const std::vector<int64_t> & ns, int64_t sys_end, int64_t last_user,
                                                       const std::vector<int64_t> & keep_at) {
    const int n = (int) ns.size();
    std::vector<char> prot(n, 0);
    if (n == 0) {
        return prot;
    }
    prot[n - 1] = 1;

    auto newest_at_or_before = [&](int64_t pos) {
        int best = -1;
        for (int i = 0; i < n - 1; ++i) {
            if (ns[i] <= pos && (best < 0 || ns[i] >= ns[best])) {
                best = i;
            }
        }
        return best;
    };

    if (n >= 2) {
        const int i = sys_end >= 0 ? newest_at_or_before(sys_end) : 0;
        prot[i >= 0 ? i : 0] = 1;
    }
    if (last_user >= 0) {
        const int i = newest_at_or_before(last_user);
        if (i >= 0) {
            prot[i] = 1;
        }
    }
    {
        const double target = (double) ns[n - 1] / 2.0;
        int    best   = -1;
        double best_r = 0.0;
        for (int i = 0; i < n - 1; ++i) {
            if (ns[i] <= 0 || target <= 0.0) {
                continue;
            }
            const double p = (double) ns[i];
            const double r = p > target ? p / target : target / p;
            if (best < 0 || r < best_r) {
                best   = i;
                best_r = r;
            }
        }
        if (best >= 0) {
            prot[best] = 1;
        }
    }
    for (const int64_t pos : keep_at) {
        if (pos < 0) {
            continue;
        }
        const int i = newest_at_or_before(pos);
        if (i >= 0) {
            prot[i] = 1;
        }
    }
    return prot;
}

// [TAG_FN_L22_CKPT_KEEP] [TAG_FN_L22_MEDIA_ANCHOR] which checkpoint a full ring drops.
// ns: n_tokens of the ring's checkpoints in ring order (ascending) followed by the incoming one (the newest).
// anchor: per entry of ns, 1 = a media anchor (empty = none). keep_rule: the keeping rule's protections
// (convstore_ckpt_protect with keep_at); without it the ring drops its oldest first.
// Drop order: (1) entries that are neither protected nor anchors; (2) anchors other than the oldest and the newest
// anchor; (3) every entry but the oldest / newest anchor; (4) every entry but the incoming. Within a tier: with the
// keeping rule the one closest to its predecessor (the least prefill lost, the older of equals), else the oldest.
// Returns the index in ns to drop (never the last), or -1 when ns has fewer than 2 entries.
static inline int convstore_ckpt_victim_l22(const std::vector<int64_t> & ns, const std::vector<char> & anchor, bool keep_rule,
                                            int64_t sys_end, int64_t last_user, const std::vector<int64_t> & keep_at) {
    const int n = (int) ns.size();
    if (n < 2) {
        return -1;
    }
    const std::vector<char> prot = keep_rule ? convstore_ckpt_protect(ns, sys_end, last_user, keep_at) : std::vector<char>(n, 0);

    auto is_anchor = [&](int i) { return i < (int) anchor.size() && anchor[i] != 0; };

    int a_old = -1;
    int a_new = -1;
    for (int i = 0; i < n; ++i) {
        if (is_anchor(i)) {
            if (a_old < 0) {
                a_old = i;
            }
            a_new = i;
        }
    }
    auto kept_anchor = [&](int i) { return i == a_old || i == a_new; };

    auto pick = [&](auto && allowed) {
        int     best     = -1;
        int64_t best_gap = 0;
        for (int i = 0; i < n - 1; ++i) {
            if (!allowed(i)) {
                continue;
            }
            if (!keep_rule) {
                return i; // the oldest
            }
            const int64_t gap = ns[i] - (i > 0 ? ns[i - 1] : 0);
            if (best < 0 || gap < best_gap) {
                best     = i;
                best_gap = gap;
            }
        }
        return best;
    };

    int v = pick([&](int i) { return !prot[i] && !is_anchor(i); });
    if (v < 0) {
        v = pick([&](int i) { return !prot[i] && is_anchor(i) && !kept_anchor(i); });
    }
    if (v < 0) {
        v = pick([&](int i) { return !kept_anchor(i); });
    }
    if (v < 0) {
        v = pick([&](int) { return true; });
    }
    return v;
}

// [TAG_FN_L22_CKPT_KEEP] the starts of the k user messages before the last one (newest first); user_starts ascending
static inline std::vector<int64_t> l22_prev_user_starts(const std::vector<int64_t> & user_starts, int k) {
    std::vector<int64_t> out;
    for (int i = (int) user_starts.size() - 2; i >= 0 && (int) out.size() < k; --i) {
        out.push_back(user_starts[i]);
    }
    return out;
}

// [TAG_FN_L22_PHASE] a decode runs as GEN only when it has rows and none of them is a prompt row
static inline bool l22_phase_gen(int64_t n_rows, int64_t n_prompt_rows) {
    return n_rows > 0 && n_prompt_rows == 0;
}

// [TAG_FN_L24_TTFT] E0: a merged image-turn batch on a multi-slot unified pool decodes alone, so it fits when the free cells hold
// its rows and the answers' next step (margin); n_free < 0: the memory gives no count, no merged batch
static inline bool l24_merged_fits(int64_t n_free, int64_t rows, int64_t margin) {
    return n_free >= 0 && rows > 0 && n_free >= rows + std::max<int64_t>(0, margin);
}

// [TAG_FN_L24_TTFT] a merged batch and the rows already in the batch: 0 = merge now (empty batch), 1 = only answers' rows (sampled +
// draft, no embedding rows) and the exclusive rule is on: ask for the next batch alone, -1 = no merged batch in this one
static inline int l24_merged_batch_rule(int32_t n_batch_rows, int32_t n_gen_rows, bool has_embd, bool excl_on) {
    if (n_batch_rows == 0) {
        return 0;
    }
    return excl_on && !has_embd && n_gen_rows > 0 && n_batch_rows == n_gen_rows ? 1 : -1;
}
