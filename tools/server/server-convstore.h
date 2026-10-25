#pragma once

// [TAG_FN_L19_CONVSTORE] Pure policy functions of the agent conversation store (no llama / server types), so that
// tests/test-fn-convstore.cpp can check them without a model.
//
//   convstore_ckpt_victim  which context checkpoint a full ring drops ([TAG_FN_L19_CONVSTORE_CKPT])
//   convstore_gdsf         the value of a stored conversation for eviction ([TAG_FN_L19_CONVSTORE_EVICT])
//   convstore_thin         which spilled checkpoints a conversation keeps on disk ([TAG_FN_L19_CONVSTORE_SPILL])

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <vector>

// [TAG_FN_L19_CONVSTORE_CKPT] A full checkpoint ring used to drop its oldest entry (and, first, entries within
// --checkpoint-min-step of an older one). In an agent loop that drops the checkpoint at the end of the system + tools
// block after a few turns, so a subagent or a new session with the same preamble processes it again. The ring now
// keeps the useful resume points:
//   - the end of the system + tools block: the newest checkpoint at or before sys_end (the first user message), or the
//     oldest checkpoint when sys_end is unknown;
//   - the start of the prompt's last user message: the newest checkpoint at or before last_user;
//   - a point about halfway back: the checkpoint nearest (by ratio) to half of the newest checkpoint's position;
//   - the newest (the incoming one, always the last of ns).
// Of the others it drops the one closest to its predecessor (the least prefill lost), the older of equals. When every
// entry is protected (a ring of 1-3), the same rule runs over all entries but the newest.
//
// ns: n_tokens of the ring's checkpoints in ring order (ascending) followed by the incoming one (the newest).
// Returns the index in ns of the checkpoint to drop (never the last), or -1 when ns has fewer than 2 entries.
static inline int convstore_ckpt_victim(const std::vector<int64_t> & ns, int64_t sys_end, int64_t last_user) {
    const int n = (int) ns.size();
    if (n < 2) {
        return -1;
    }
    std::vector<char> prot(n, 0);
    prot[n - 1] = 1;

    // the newest checkpoint at or before a position, among the existing ones (not the incoming)
    auto newest_at_or_before = [&](int64_t pos) {
        int best = -1;
        for (int i = 0; i < n - 1; ++i) {
            if (ns[i] <= pos && (best < 0 || ns[i] >= ns[best])) {
                best = i;
            }
        }
        return best;
    };

    {
        const int i = sys_end >= 0 ? newest_at_or_before(sys_end) : 0;
        if (i >= 0) {
            prot[i] = 1;
        } else {
            prot[0] = 1; // nothing at or before the block end: the oldest is the shortest shared prefix
        }
    }
    if (last_user >= 0) {
        const int i = newest_at_or_before(last_user);
        if (i >= 0) {
            prot[i] = 1;
        }
    }
    {
        // nearest by ratio: a ring of 4 then keeps one point between about a quarter of the way and the end, moving up
        // as the conversation grows (a ladder with one rung)
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

    auto pick = [&](bool honour_prot) {
        int     best     = -1;
        int64_t best_gap = 0;
        for (int i = 0; i < n - 1; ++i) {
            if (honour_prot && prot[i]) {
                continue;
            }
            const int64_t gap = ns[i] - (i > 0 ? ns[i - 1] : 0);
            if (best < 0 || gap < best_gap) {
                best     = i;
                best_gap = gap;
            }
        }
        return best;
    };
    const int v = pick(true);
    return v >= 0 ? v : pick(false);
}

// [TAG_FN_L19_CONVSTORE_EVICT] Greedy-dual size-frequency value of a stored conversation (as Infernix): the prefill a
// restore saves per byte it holds, times (1 + the restores it served), on top of the inflation value L (the value of
// the last evicted entry), so an entry that is not used falls behind the newer ones. The prefill is counted in tokens
// plus a fixed cost per restore (fixed_tokens: what a prompt chunk costs whatever its size, in tokens). An entry holds
// at least min_bytes (the disk budget over the number of entries): while the count binds, a 2K-token side request is not
// worth more than a 30K-token session because it is small.
// Returns the value; the store evicts the lowest first and then sets L to the evicted value.
static inline double convstore_gdsf(double L, int64_t hits, int64_t n_tokens, uint64_t n_bytes, double fixed_tokens,
                                    uint64_t min_bytes = 0) {
    const uint64_t b   = n_bytes > min_bytes ? n_bytes : min_bytes;
    const double   gib = (double) (b > 0 ? b : 1) / (1024.0 * 1024.0 * 1024.0);
    return L + (1.0 + (double) (hits > 0 ? hits : 0)) * ((double) n_tokens + fixed_tokens) / gib;
}

// [TAG_FN_L24_KEEPVAL] Eviction by value: the re-read cost a restore saves (tokens + a fixed pass cost) times a capped
// restore count, times recency (1 / (1 + age / half_s)). The default of before evicted the least recently stored entry,
// so 3-4 one-shot side requests evicted the main conversation (owner 10-22: 48.6K re-read in 24.8 s); GDSF's inflation
// value lets enough of them do the same. Here the biggest conversation used within active_s is kept against the count
// limit (only the disk budget may take it, and last), and the others go by value.
struct convstore_val_entry {
    int64_t n_tokens = 0;
    int64_t hits     = 0;     // restores it served
    double  age_s    = 0.0;   // since it was stored or last restored
    bool    keep     = false; // never evicted (the conversation the next prompt restores)
};

static inline double convstore_val(int64_t n_tokens, int64_t hits, double age_s, double fixed_tokens, double half_s) {
    const double cost = (double) std::max<int64_t>(0, n_tokens) + std::max(0.0, fixed_tokens);
    const double freq = 1.0 + (double) std::min<int64_t>(std::max<int64_t>(0, hits), 3);
    const double rec  = half_s > 0.0 ? 1.0 / (1.0 + std::max(0.0, age_s) / half_s) : 1.0;
    return cost * freq * rec;
}

// the protected entry: the most tokens among the entries used within active_s (keep entries count too); -1 when none
static inline int convstore_val_protect(const std::vector<convstore_val_entry> & es, double active_s) {
    int best = -1;
    for (int i = 0; i < (int) es.size(); ++i) {
        if (es[i].age_s <= active_s && (best < 0 || es[i].n_tokens > es[best].n_tokens)) {
            best = i;
        }
    }
    return best;
}

// the entry to evict: the lowest value, never a keep entry. budget = false (the count limit): never the protected entry;
// budget = true (disk room): the protected entry only when nothing else is left. -1 when there is nothing to evict.
static inline int convstore_val_victim(const std::vector<convstore_val_entry> & es, double fixed_tokens, double half_s,
                                       double active_s, bool budget) {
    const int prot = convstore_val_protect(es, active_s);
    int    best   = -1;
    double best_v = 0.0;
    for (int i = 0; i < (int) es.size(); ++i) {
        if (es[i].keep || i == prot) {
            continue;
        }
        const double v = convstore_val(es[i].n_tokens, es[i].hits, es[i].age_s, fixed_tokens, half_s);
        if (best < 0 || v < best_v) {
            best   = i;
            best_v = v;
        }
    }
    if (best < 0 && budget && prot >= 0 && !es[prot].keep) {
        best = prot;
    }
    return best;
}

// [TAG_FN_L19_CONVSTORE_SPILL] Checkpoints a full ring drops go to a file of the conversation instead of being lost.
// A dropped checkpoint is written only when no kept one (in the ring or on disk) lies within min_gap tokens of it, and
// a conversation keeps at most n_max on disk: past that the one closest to its predecessor goes (the newest stays).
// ns: n_tokens of the conversation's spilled checkpoints, ascending. Returns the index to drop, or -1 when within n_max.
static inline int convstore_thin(const std::vector<int64_t> & ns, int n_max) {
    const int n = (int) ns.size();
    if (n <= n_max || n < 2) {
        return -1;
    }
    int     best     = -1;
    int64_t best_gap = 0;
    for (int i = 0; i < n - 1; ++i) {
        const int64_t gap = ns[i] - (i > 0 ? ns[i - 1] : 0);
        if (best < 0 || gap < best_gap) {
            best     = i;
            best_gap = gap;
        }
    }
    return best;
}

// whether a checkpoint at pos is within min_gap tokens of one of the kept positions
static inline bool convstore_near(const std::vector<int64_t> & kept, int64_t pos, int64_t min_gap) {
    for (const int64_t k : kept) {
        if ((k > pos ? k - pos : pos - k) < min_gap) {
            return true;
        }
    }
    return false;
}

// [TAG_FN_L20_CONVFIX_LAZY] A restore that leaves the stored checkpoints on disk splits the ring of before in two: the
// checkpoints in RAM (the ring) and the ones on disk. Without the keeping rule the ring kept its newest cap positions, so
// the two lists together keep them too. pos: the positions of both (any order, duplicates count once each). Returns the
// smallest position that stays: the cap-th newest, or the lowest int64 when all of them fit.
static inline int64_t convfix_keep_from(std::vector<int64_t> pos, size_t cap) {
    if (cap == 0 || pos.size() <= cap) {
        return std::numeric_limits<int64_t>::min();
    }
    std::sort(pos.begin(), pos.end(), std::greater<int64_t>());
    return pos[cap - 1];
}
