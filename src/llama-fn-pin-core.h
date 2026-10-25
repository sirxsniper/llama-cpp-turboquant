#pragma once

// [TAG_FN_L22_PIN] round 22 P6: the pure parts of the pinned expert ranges (no llama, no OS, no CUDA), shared by
// src/llama-fn-pin.cpp, the probe (tools/moe-bridge-probe test 7) and tests/test-fn-pin.cpp:
//   - the unit: per-expert coalesced ranges, aligned inward to pages (plan P6.3)
//   - the plan order: fully streamed layers first, then cold bytes x decode CPU use (P6.5)
//   - the budget rules: NON_LOCAL, LOCAL, RAM, cap (P6.4)
//   - the registry: address lookup for the stream (segments, tickets), registration units, a LIFO stack and a bounded
//     drain before an unregister (P6.6, plan 3.6)

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <vector>

namespace fn_pin {

// ---------------------------------------------------------------------------------------------------------------------
// unit

// one routed-expert tensor of a layer as it lies in host memory: expert e is [data + e x stride, + stride)
struct tensor_view {
    const uint8_t * data     = nullptr;
    size_t          stride   = 0;
    int             n_expert = 0;
};

struct layer_in {
    int                  il = -1;
    tensor_view          t[3];             // up, gate, down (data == nullptr: absent)
    std::vector<uint8_t> ok;               // ok[e] = 1: expert e may be pinned (RAM class, not VRAM-held, not SSD)
    bool                 fully_streamed = false; // every expert streams (lend / tail / borrow / SLOTLEND, no fixed residents)
    double               cpu_use = 0.0;    // decode CPU use of its cold experts (decayed counts), >= 0
    bool                 span_ok = false;  // spans mode: [min start, max end) of its tensors is one mapped view
};

struct range {
    const uint8_t * p     = nullptr; // page-aligned
    size_t          n     = 0;       // page multiple
    int             il    = -1;
    int             part  = -1;      // 0 up, 1 gate, 2 down, -1 a layer span
    int             e0    = 0;       // expert run [e0, e1)
    int             e1    = 0;
    int             group = -1;      // its (layer, run) group (index into groups)
};

// a (layer, expert run): the experts it covers are marked pinned only when all of its ranges are registered
struct group {
    int    il     = -1;
    int    e0     = 0;
    int    e1     = 0;
    int    first  = 0; // its ranges: [first, first + n_rng) in the plan's range list
    int    n_rng  = 0;
    size_t bytes  = 0;
    bool   whole  = false; // a fully streamed layer (whole-tensor ranges or a span)
    double score  = 0.0;
};

inline uintptr_t align_up(uintptr_t x, size_t a) {
    return (x + a - 1)/a*a;
}

inline uintptr_t align_down(uintptr_t x, size_t a) {
    return x/a*a;
}

// the inward page-aligned part of [s, e); false when nothing is left
inline bool inward(const uint8_t * s, const uint8_t * e, size_t page, const uint8_t ** ps, size_t * n) {
    const uintptr_t a = align_up((uintptr_t) s, page);
    const uintptr_t b = align_down((uintptr_t) e, page);
    if (b <= a) {
        return false;
    }
    *ps = (const uint8_t *) a;
    *n  = (size_t) (b - a);
    return true;
}

struct plan {
    std::vector<range> ranges; // in plan order: group by group, the parts of one group adjacent
    std::vector<group> groups; // in plan order
};

// spans: 0 per-expert ranges (default), 1 a whole span for a fully streamed layer with span_ok (else ranges).
// max_range > 0: a range longer than that is cut into pieces of at most max_range (page multiples), each its own
// registration (one driver call stays short), all in the same group
inline plan build_plan(const std::vector<layer_in> & layers, size_t page, bool spans, size_t max_range = 0) {
    plan out;
    if (page == 0) {
        page = 4096;
    }
    if (max_range > 0) {
        max_range = std::max(page, max_range/page*page);
    }
    // r cut into pieces, each counted in its group
    auto push_cut = [&](std::vector<range> & rs, group & g, const range & r) {
        for (size_t o = 0; o < r.n; ) {
            range x = r;
            x.p = r.p + o;
            x.n = max_range > 0 ? std::min(max_range, r.n - o) : r.n - o;
            rs.push_back(x);
            g.n_rng++;
            g.bytes += x.n;
            o += x.n;
        }
    };
    // per layer: the groups and their ranges in expert order
    struct lay {
        std::vector<group> gs;
        std::vector<range> rs;
        size_t             bytes = 0;
        bool               whole = false;
        double             cpu   = 0.0;
        int                il    = -1;
    };
    std::vector<lay> ls;
    for (const layer_in & L : layers) {
        int n_exp = 0;
        for (int k = 0; k < 3; ++k) {
            if (L.t[k].data != nullptr) {
                n_exp = n_exp == 0 ? L.t[k].n_expert : std::min(n_exp, L.t[k].n_expert);
            }
        }
        if (n_exp <= 0) {
            continue;
        }
        bool all_ok = (int) L.ok.size() >= n_exp;
        for (int e = 0; all_ok && e < n_exp; ++e) {
            all_ok = L.ok[e] != 0;
        }
        lay y;
        y.il    = L.il;
        y.whole = L.fully_streamed && all_ok;
        y.cpu   = std::max(0.0, L.cpu_use);
        if (spans && y.whole && L.span_ok) {
            const uint8_t * lo = nullptr;
            const uint8_t * hi = nullptr;
            for (int k = 0; k < 3; ++k) {
                if (L.t[k].data == nullptr) {
                    continue;
                }
                const uint8_t * s = L.t[k].data;
                const uint8_t * e = L.t[k].data + (size_t) n_exp*L.t[k].stride;
                lo = lo == nullptr ? s : std::min(lo, s);
                hi = hi == nullptr ? e : std::max(hi, e);
            }
            range r;
            if (lo && inward(lo, hi, page, &r.p, &r.n)) {
                r.il = L.il; r.part = -1; r.e0 = 0; r.e1 = n_exp;
                group g;
                g.il = L.il; g.e0 = 0; g.e1 = n_exp; g.whole = true;
                push_cut(y.rs, g, r);
                y.gs.push_back(g);
                y.bytes += g.bytes;
            }
        } else {
            for (int e = 0; e < n_exp; ) {
                if (e >= (int) L.ok.size() || !L.ok[e]) {
                    ++e;
                    continue;
                }
                int e1 = e + 1;
                while (e1 < n_exp && e1 < (int) L.ok.size() && L.ok[e1]) {
                    ++e1;
                }
                group g;
                g.il = L.il; g.e0 = e; g.e1 = e1; g.whole = y.whole;
                for (int k = 0; k < 3; ++k) {
                    const tensor_view & t = L.t[k];
                    if (t.data == nullptr) {
                        continue;
                    }
                    range r;
                    if (inward(t.data + (size_t) e*t.stride, t.data + (size_t) e1*t.stride, page, &r.p, &r.n)) {
                        r.il = L.il; r.part = k; r.e0 = e; r.e1 = e1;
                        push_cut(y.rs, g, r);
                    }
                }
                if (g.n_rng > 0) {
                    y.gs.push_back(g);
                    y.bytes += g.bytes;
                }
                e = e1;
            }
        }
        if (!y.gs.empty()) {
            ls.push_back(std::move(y));
        }
    }
    // the order: fully streamed layers first (layer order: the stream's order), then cold bytes x (1 + CPU use share)
    double cpu_sum = 0.0;
    int    n_part  = 0;
    for (const lay & y : ls) {
        if (!y.whole) {
            cpu_sum += y.cpu;
            n_part++;
        }
    }
    const double cpu_mean = n_part > 0 && cpu_sum > 0.0 ? cpu_sum/n_part : 0.0;
    std::vector<int> idx(ls.size());
    for (size_t i = 0; i < ls.size(); ++i) {
        idx[i] = (int) i;
    }
    auto score = [&](const lay & y) {
        return (double) y.bytes*(1.0 + (cpu_mean > 0.0 ? y.cpu/cpu_mean : 0.0));
    };
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
        const lay & x = ls[a];
        const lay & y = ls[b];
        if (x.whole != y.whole) {
            return x.whole;
        }
        if (x.whole) {
            return x.il < y.il;
        }
        const double sx = score(x);
        const double sy = score(y);
        return sx != sy ? sx > sy : x.il < y.il;
    });
    for (int i : idx) {
        lay & y = ls[i];
        const double sc = score(y);
        size_t r = 0;
        for (group g : y.gs) {
            g.first = (int) out.ranges.size();
            g.score = sc;
            for (int k = 0; k < g.n_rng; ++k) {
                range rr = y.rs[r++];
                rr.group = (int) out.groups.size();
                out.ranges.push_back(rr);
            }
            out.groups.push_back(g);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------------------------------------------------
// budget rules

enum stop_rule {
    RULE_OK       = 0,
    RULE_NONLOCAL = 1, // DXGI NON_LOCAL budget - usage - charge < margin
    RULE_LOCAL    = 2, // DXGI LOCAL budget - usage - charge < l_keep
    RULE_RAM      = 3, // available RAM - new bytes - warm keep < floor
    RULE_MAX      = 4, // PIN_MAX_MIB
};

inline const char * rule_name(int r) {
    switch (r) {
        case RULE_OK:       return "ok";
        case RULE_NONLOCAL: return "nonlocal";
        case RULE_LOCAL:    return "local";
        case RULE_RAM:      return "ram";
        case RULE_MAX:      return "max";
        default:            return "?";
    }
}

struct budget_in {
    bool     have_nl   = false;
    uint64_t nl_budget = 0;
    uint64_t nl_usage  = 0;
    bool     have_l    = false;
    uint64_t l_budget  = 0;
    uint64_t l_usage   = 0;
    uint64_t avail     = 0; // available RAM
    uint64_t pinned    = 0; // registered bytes so far
};

struct budget_cfg {
    uint64_t nl_margin    = (uint64_t) 8192 << 20;
    uint64_t l_margin     = (uint64_t) 1024 << 20; // input of l_keep (the caller computes it)
    uint64_t l_keep       = (uint64_t) 1024 << 20; // LOCAL headroom that must stay
    uint64_t floor        = (uint64_t) 6144 << 20;
    uint64_t warm_keep    = 0;   // RAM kept for the non-pinned warm need
    uint64_t max_pin      = 0;   // 0: no cap
    double   nl_ratio     = 1.0; // expected NON_LOCAL usage per registered byte
    double   l_ratio      = 0.0; // expected LOCAL usage per registered byte
};

// may add_bytes more be registered (add_new of them not resident yet)? add 0 = the check after a registration
inline int check(const budget_in & b, const budget_cfg & c, uint64_t add_bytes, uint64_t add_new) {
    if (c.max_pin > 0 && b.pinned + add_bytes > c.max_pin) {
        return RULE_MAX;
    }
    if (b.have_nl) {
        const double need = (double) b.nl_usage + (double) add_bytes*c.nl_ratio + (double) c.nl_margin;
        if ((double) b.nl_budget < need) {
            return RULE_NONLOCAL;
        }
    }
    if (b.have_l) {
        const double need = (double) b.l_usage + (double) add_bytes*c.l_ratio + (double) c.l_keep;
        if ((double) b.l_budget < need) {
            return RULE_LOCAL;
        }
    }
    if ((double) b.avail < (double) add_new + (double) c.floor + (double) c.warm_keep) {
        return RULE_RAM;
    }
    return RULE_OK;
}

// the bytes the rules allow at plan time (NON_LOCAL at its ratio, RAM with every planned byte new, the cap); the
// LOCAL rule only says yes / no here (its charge is measured per range)
inline uint64_t plan_bytes(const budget_in & b, const budget_cfg & c) {
    uint64_t lim = UINT64_MAX;
    if (c.max_pin > 0) {
        lim = std::min(lim, c.max_pin > b.pinned ? c.max_pin - b.pinned : 0);
    }
    if (b.have_nl) {
        const double room = (double) b.nl_budget - (double) b.nl_usage - (double) c.nl_margin;
        const double r    = c.nl_ratio > 1e-6 ? room/c.nl_ratio : (room >= 0.0 ? 1e30 : -1.0);
        lim = std::min(lim, r <= 0.0 ? (uint64_t) 0 : (r >= 1.8e19 ? UINT64_MAX : (uint64_t) r));
    }
    if (b.have_l) {
        const double room = (double) b.l_budget - (double) b.l_usage - (double) c.l_keep;
        if (room < 0.0) {
            return 0;
        }
        if (c.l_ratio > 1e-6) {
            const double r = room/c.l_ratio;
            lim = std::min(lim, r >= 1.8e19 ? UINT64_MAX : (uint64_t) r);
        }
    }
    const double ram = (double) b.avail - (double) c.floor - (double) c.warm_keep;
    lim = std::min(lim, ram <= 0.0 ? (uint64_t) 0 : (uint64_t) ram);
    return lim;
}

// R-REV-1: the OOM hook's stop rule. Below this LOCAL charge per registered byte, unpinning does not raise DXGI
// headroom in a useful way
constexpr double OOM_RATIO_MIN = 0.05;

// the most one OOM hook call unpins: `want` bytes when it stops by bytes (DXGI shows the room, cannot be read, or shows no
// charge), else the bytes whose measured charge covers the deficit (x 1.25 + 256 MiB), never less than want
inline uint64_t oom_cap_bytes(uint64_t want, int64_t head0, bool by_bytes, double l_ratio) {
    if (by_bytes || l_ratio < OOM_RATIO_MIN) {
        return want;
    }
    const double deficit = (double) want - (double) head0;
    if (deficit <= 0.0) {
        return want;
    }
    const double cap = 1.25*deficit/l_ratio + (double) ((uint64_t) 256 << 20);
    return cap >= 1.8e19 ? UINT64_MAX : std::max(want, (uint64_t) cap);
}

// [TAG_FN_L23_PINFIX] R-L23-1 (r22 T13): what an unpin can give an allocation. Unpinning frees host pages only: it raises
// DXGI LOCAL headroom by the charge per registered byte that the registration measured (l_ratio). Without that charge
// (< OOM_RATIO_MIN; this box: 0.000) it helps only an allocation whose part above the LOCAL headroom may land in shared
// system memory (NON_LOCAL budget - usage < that part) or one short of commit (commit room < want; only arena chunks give
// commit back: have_commit). Returns
// the bytes to unpin: oom_cap_bytes with a LOCAL charge or when DXGI LOCAL cannot be read (as before), else the larger
// NON_LOCAL / commit deficit + 256 MiB (at most want); 0 = an unpin cannot help, nothing goes (r22 T13: the 16K prompt
// buffer's pre-check unpinned 7.9 GiB at the first prompt of every PIN run, LOCAL headroom 6.85 -> 6.85 GiB).
struct oom_in {
    bool     have_l      = false;
    int64_t  l_head      = 0;
    double   l_ratio     = 0.0;
    bool     have_nl     = false;
    uint64_t nl_budget   = 0;
    uint64_t nl_usage    = 0;
    bool     have_commit = false;
    uint64_t commit_room = 0;
};

inline uint64_t oom_help_bytes(uint64_t want, const oom_in & o) {
    if (!o.have_l) {
        return want;
    }
    if (o.l_ratio >= OOM_RATIO_MIN) {
        return oom_cap_bytes(want, o.l_head, o.l_head >= (int64_t) want, o.l_ratio);
    }
    uint64_t def = 0;
    if (o.have_nl) {
        // only the part LOCAL cannot hold may spill into system memory
        const uint64_t spill = o.l_head > 0 ? (want > (uint64_t) o.l_head ? want - (uint64_t) o.l_head : 0) : want;
        const uint64_t room  = o.nl_budget > o.nl_usage ? o.nl_budget - o.nl_usage : 0;
        def = std::max<uint64_t>(def, spill > room ? spill - room : 0);
    }
    if (o.have_commit) {
        def = std::max<uint64_t>(def, want > o.commit_room ? want - o.commit_room : 0);
    }
    return def == 0 ? 0 : std::min<uint64_t>(want, def + ((uint64_t) 256 << 20));
}

// [TAG_FN_L23_PINFIX] R-L23-6: can the pins be the cause of a failed allocation? A host (pinned / sysmem) allocation, a LOCAL
// charge per pinned byte, NON_LOCAL unknown, NON_LOCAL room under the margin, or a spill past LOCAL that NON_LOCAL cannot hold.
// AFR_NONE: a plain VRAM failure, the pins stay (l23p B6: a 640.9 MiB VRAM failure dropped all 55 GiB for the whole run)
enum { AF_HOST = 0, AF_DEV = 1, AF_DEV_FATAL = 2 };
enum { AFR_NONE = 0, AFR_HOST, AFR_LOCAL_CHARGE, AFR_NL_UNKNOWN, AFR_NL_ROOM, AFR_NL_SPILL };

struct af_in {
    int      kind      = AF_DEV;
    double   l_ratio   = 0.0;
    bool     have_l    = false;
    int64_t  l_head    = 0;
    bool     have_nl   = false;
    uint64_t nl_room   = 0;
    uint64_t nl_margin = 0;
    uint64_t need      = 0;
};

inline int alloc_fail_reason(const af_in & a) {
    if (a.kind == AF_HOST) {
        return AFR_HOST;
    }
    if (a.l_ratio >= OOM_RATIO_MIN) {
        return AFR_LOCAL_CHARGE;
    }
    if (!a.have_nl) {
        return AFR_NL_UNKNOWN;
    }
    if (a.nl_room < a.nl_margin) {
        return AFR_NL_ROOM;
    }
    const uint64_t spill = !a.have_l || a.l_head <= 0 ? a.need : (a.need > (uint64_t) a.l_head ? a.need - (uint64_t) a.l_head : 0);
    return spill > a.nl_room ? AFR_NL_SPILL : AFR_NONE;
}

inline const char * af_reason_name(int r) {
    switch (r) {
        case AFR_HOST:         return "host";
        case AFR_LOCAL_CHARGE: return "local_charge";
        case AFR_NL_UNKNOWN:   return "nl_unknown";
        case AFR_NL_ROOM:      return "nl_room";
        case AFR_NL_SPILL:     return "nl_spill";
        default:               return "none";
    }
}

// [TAG_FN_L23_PINFIX] R-L23-2 (main session 10-21: pins may only ever add speed; under pressure behave exactly like NEW).
// The gate of the pin thread, one sample per 2 s: with pins, the available RAM (free + standby) below `drop` for `drop_n`
// samples in a row drops ALL pins (-1; no partial release, nothing leaves the working set: an unpin only unlocks). Without
// pins (or with released units left), room >= `repin` for `hold_n` samples in a row, `hold_n` samples after the last drop,
// lets resident units be pinned again (+1; the caller adds the bytes they would bring in to `repin` per unit).
struct gate_state {
    int     low       = 0;
    int     high      = 0;
    int64_t since_drop = INT64_MAX/2; // samples since the last drop
};

inline int gate_step(gate_state & g, uint64_t avail, bool pinned, bool repinnable, uint64_t drop, int drop_n, uint64_t repin, int hold_n) {
    if (g.since_drop < INT64_MAX/2) {
        g.since_drop++;
    }
    if (pinned && avail < drop) {
        g.high = 0;
        if (++g.low >= drop_n) {
            g.low        = 0;
            g.since_drop = 0;
            return -1;
        }
        return 0;
    }
    g.low = 0;
    if (repinnable && avail >= repin) {
        if (++g.high >= hold_n && g.since_drop >= hold_n) {
            g.high = 0;
            return 1;
        }
        return 0;
    }
    g.high = 0;
    return 0;
}

// the groups of the plan that fit in `bytes`, in plan order: a group that does not fit ends the list (whole experts,
// the order kept)
inline int plan_take(const plan & p, uint64_t bytes) {
    uint64_t sum = 0;
    int n = 0;
    for (const group & g : p.groups) {
        if (sum + g.bytes > bytes) {
            break;
        }
        sum += g.bytes;
        n++;
    }
    return n;
}

// ---------------------------------------------------------------------------------------------------------------------
// registry

struct seg {
    const void * src    = nullptr;
    size_t       len    = 0;
    bool         pinned = false;
    int32_t      ticket = -1;
};

enum rstate : int {
    ST_PLANNED  = 0,
    ST_ACTIVE   = 1,
    ST_DRAINING = 2,
    ST_GONE     = 3,
};

struct reg_range {
    const uint8_t *  p    = nullptr; // the mapped bytes [p, p + n)
    size_t           n    = 0;
    const uint8_t *  copy = nullptr; // arena: their copy (the DMA source), else nullptr (p itself)
    int              unit = -1;
    int              src  = -1;      // index in the plan's range list
    std::atomic<int> state{ST_PLANNED};
    std::atomic<int> users{0};
};

// one registration with the device: a mapped range, or an arena chunk that holds several ranges' copies
struct reg_unit {
    int              id = -1;
    void *           p = nullptr;
    size_t           n = 0;
    std::vector<int> ranges; // registry indices
    std::atomic<int> state{ST_PLANNED};
    uint64_t         order = 0; // registration order (LIFO)
    uint64_t         dev   = 0; // the device address of p (0: unknown), set before the unit is activated
};

class registry {
public:
    // ranges: [p, n, copy] per registry entry, sorted by address on the way in (overlaps are dropped: false)
    struct in_range {
        const uint8_t * p    = nullptr;
        size_t          n    = 0;
        const uint8_t * copy = nullptr;
        int             unit = -1;
        int             src  = -1;
    };

    // no unit may be active (no ticket out): the lookup is replaced under the exclusive lock
    bool build(std::vector<in_range> rs, int n_units) {
        std::unique_lock<std::shared_mutex> lk(mtx_);
        std::sort(rs.begin(), rs.end(), [](const in_range & a, const in_range & b) { return a.p < b.p; });
        bool clean = true;
        std::vector<in_range> keep;
        for (const in_range & r : rs) {
            if (r.n == 0 || r.unit < 0 || r.unit >= n_units) {
                clean = false;
                continue;
            }
            if (!keep.empty() && keep.back().p + keep.back().n > r.p) {
                clean = false; // overlaps the previous one
                continue;
            }
            keep.push_back(r);
        }
        n_rng_ = (int) keep.size();
        rng_.reset(new reg_range[keep.size() > 0 ? keep.size() : 1]);
        starts_.resize(keep.size());
        units_.reset(new reg_unit[n_units > 0 ? n_units : 1]);
        n_unit_ = n_units;
        for (int u = 0; u < n_units; ++u) {
            units_[u].id = u;
        }
        lifo_.clear();
        n_active_.store(0);
        active_bytes_.store(0);
        for (size_t i = 0; i < keep.size(); ++i) {
            rng_[i].p    = keep[i].p;
            rng_[i].n    = keep[i].n;
            rng_[i].copy = keep[i].copy;
            rng_[i].unit = keep[i].unit;
            rng_[i].src  = keep[i].src;
            starts_[i]   = keep[i].p;
            units_[keep[i].unit].ranges.push_back((int) i);
        }
        return clean;
    }

    void set_unit(int u, void * p, size_t n) {
        std::unique_lock<std::shared_mutex> lk(mtx_);
        units_[u].p = p;
        units_[u].n = n;
    }

    void set_unit_dev(int u, uint64_t dev) {
        std::unique_lock<std::shared_mutex> lk(mtx_);
        units_[u].dev = dev;
    }

    // the device address of a pinned segment's first byte (0: not pinned or unknown); valid while its ticket is held
    uint64_t seg_dev(const seg & s) const {
        if (!s.pinned || s.ticket < 0 || s.ticket >= n_rng_) {
            return 0;
        }
        const reg_unit & U = units_[rng_[s.ticket].unit];
        if (U.dev == 0 || U.p == nullptr) {
            return 0;
        }
        return U.dev + (uint64_t) ((const uint8_t *) s.src - (const uint8_t *) U.p);
    }

    // an arena unit's copies were placed as offsets (copy = offset): they become addresses in base (before activate)
    void fix_copies(int u, const uint8_t * base) {
        std::unique_lock<std::shared_mutex> lk(mtx_);
        for (int i : units_[u].ranges) {
            rng_[i].copy = base + (size_t) (uintptr_t) rng_[i].copy;
        }
    }

    int n_ranges() const { return n_rng_; }
    int n_units() const { return n_unit_; }
    const reg_range & range_at(int i) const { return rng_[i]; }
    const reg_unit & unit_at(int u) const { return units_[u]; }
    int  n_active_units() const { return n_active_.load(std::memory_order_acquire); }
    uint64_t active_bytes() const { return active_bytes_.load(std::memory_order_relaxed); }
    bool any_active() const { return n_active_.load(std::memory_order_acquire) > 0; }

    // the unit was registered: its ranges answer pinned from now on
    void activate(int u) {
        std::unique_lock<std::shared_mutex> lk(mtx_);
        reg_unit & U = units_[u];
        if (U.state.load() == ST_ACTIVE) {
            return;
        }
        for (int i : U.ranges) {
            rng_[i].state.store(ST_ACTIVE, std::memory_order_release);
        }
        U.state.store(ST_ACTIVE, std::memory_order_release);
        U.order = ++order_;
        lifo_.push_back(u);
        n_active_.fetch_add(1, std::memory_order_acq_rel);
        active_bytes_.fetch_add(U.n, std::memory_order_relaxed);
    }

    // the first segment of [p, p + n): a pinned segment never runs past its range; an unpinned one stops at the next
    // active range. A pinned segment holds a ticket (release it after the copy that reads it has completed)
    bool query(const void * p0, size_t n, seg * out) {
        out->src    = p0;
        out->len    = n;
        out->pinned = false;
        out->ticket = -1;
        if (n == 0 || n_active_.load(std::memory_order_acquire) == 0) {
            return true;
        }
        const uint8_t * p = (const uint8_t *) p0;
        std::shared_lock<std::shared_mutex> lk(mtx_);
        // the last range that starts at or before p
        size_t i = (size_t) (std::upper_bound(starts_.begin(), starts_.end(), p) - starts_.begin());
        if (i > 0) {
            reg_range & r = rng_[i - 1];
            if (p < r.p + r.n) {
                const size_t left = (size_t) (r.p + r.n - p);
                if (r.state.load(std::memory_order_acquire) == ST_ACTIVE) {
                    r.users.fetch_add(1, std::memory_order_acq_rel);
                    out->src    = r.copy ? (const void *) (r.copy + (p - r.p)) : p0;
                    out->len    = std::min(n, left);
                    out->pinned = true;
                    out->ticket = (int32_t) (i - 1);
                    return true;
                }
            }
        }
        // unpinned: up to the next active range
        const uint8_t * end = p + n;
        for (size_t j = i; j < (size_t) n_rng_ && rng_[j].p < end; ++j) {
            if (rng_[j].state.load(std::memory_order_acquire) == ST_ACTIVE) {
                end = rng_[j].p;
                break;
            }
        }
        out->len = (size_t) (end - p);
        return true;
    }

    void release(int32_t ticket) {
        if (ticket < 0 || ticket >= n_rng_) {
            return;
        }
        reg_range & r = rng_[ticket];
        if (r.users.fetch_sub(1, std::memory_order_acq_rel) == 1 && r.state.load(std::memory_order_acquire) == ST_DRAINING) {
            std::lock_guard<std::mutex> lk(cv_mtx_);
            cv_.notify_all();
        }
    }

    // the last registered unit still active (-1: none), taken off the stack
    int pop_lifo() {
        std::unique_lock<std::shared_mutex> lk(mtx_);
        while (!lifo_.empty()) {
            const int u = lifo_.back();
            lifo_.pop_back();
            if (units_[u].state.load() == ST_ACTIVE) {
                return u;
            }
        }
        return -1;
    }

    // a unit that could not be drained goes back on the stack, in its registration order
    void push_back_lifo(int u) {
        std::unique_lock<std::shared_mutex> lk(mtx_);
        if (units_[u].state.load() != ST_ACTIVE) {
            return;
        }
        auto it = std::lower_bound(lifo_.begin(), lifo_.end(), u, [&](int a, int b) { return units_[a].order < units_[b].order; });
        lifo_.insert(it, u);
    }

    // stop new users of the unit, wait (at most timeout_us) until none is left, then unreg(unit) and mark it gone.
    // false: users stayed (the unit is active again, still registered) or it was not active
    bool drain(int u, int64_t timeout_us, const std::function<void(const reg_unit &)> & unreg) {
        reg_unit & U = units_[u];
        {
            std::unique_lock<std::shared_mutex> lk(mtx_);
            if (U.state.load() != ST_ACTIVE) {
                return false;
            }
            U.state.store(ST_DRAINING);
            for (int i : U.ranges) {
                rng_[i].state.store(ST_DRAINING, std::memory_order_release);
            }
        }
        auto idle = [&]() {
            for (int i : U.ranges) {
                if (rng_[i].users.load(std::memory_order_acquire) != 0) {
                    return false;
                }
            }
            return true;
        };
        bool ok;
        {
            std::unique_lock<std::mutex> lk(cv_mtx_);
            ok = cv_.wait_for(lk, std::chrono::microseconds(std::max<int64_t>(0, timeout_us)), idle);
        }
        if (!ok) {
            std::unique_lock<std::shared_mutex> lk(mtx_);
            for (int i : U.ranges) {
                rng_[i].state.store(ST_ACTIVE, std::memory_order_release);
            }
            U.state.store(ST_ACTIVE);
            return false;
        }
        if (unreg) {
            unreg(U);
        }
        {
            std::unique_lock<std::shared_mutex> lk(mtx_);
            for (int i : U.ranges) {
                rng_[i].state.store(ST_GONE, std::memory_order_release);
            }
            U.state.store(ST_GONE);
        }
        n_active_.fetch_sub(1, std::memory_order_acq_rel);
        active_bytes_.fetch_sub(U.n, std::memory_order_relaxed);
        return true;
    }

    // a unit whose registration failed or was never tried
    void drop(int u) {
        std::unique_lock<std::shared_mutex> lk(mtx_);
        reg_unit & U = units_[u];
        if (U.state.load() != ST_PLANNED) {
            return;
        }
        for (int i : U.ranges) {
            rng_[i].state.store(ST_GONE, std::memory_order_release);
        }
        U.state.store(ST_GONE);
    }

    // unpin last-registered units until stop() says enough (checked before each one) or none is left; a unit whose
    // users do not leave within timeout_us is skipped (kept). Returns the units unregistered; *bytes their size
    int unpin_lifo(const std::function<bool()> & stop, int64_t timeout_us, const std::function<void(const reg_unit &)> & unreg,
            uint64_t * bytes, std::vector<int> * gone = nullptr) {
        int n = 0;
        uint64_t b = 0;
        std::vector<int> busy;
        while (!(stop && stop())) {
            const int u = pop_lifo();
            if (u < 0) {
                break;
            }
            const size_t sz = units_[u].n;
            if (drain(u, timeout_us, unreg)) {
                n++;
                b += sz;
                if (gone) {
                    gone->push_back(u);
                }
            } else {
                busy.push_back(u);
            }
        }
        for (auto it = busy.rbegin(); it != busy.rend(); ++it) {
            push_back_lifo(*it);
        }
        if (bytes) {
            *bytes = b;
        }
        return n;
    }

    // users still holding tickets (diagnostics)
    int users_total() const {
        int s = 0;
        for (int i = 0; i < n_rng_; ++i) {
            s += rng_[i].users.load(std::memory_order_relaxed);
        }
        return s;
    }

private:
    std::unique_ptr<reg_range[]> rng_;
    std::vector<const uint8_t *> starts_;
    int                          n_rng_ = 0;
    std::unique_ptr<reg_unit[]>  units_;
    int                          n_unit_ = 0;
    std::vector<int>             lifo_;
    uint64_t                     order_ = 0;
    std::atomic<int>             n_active_{0};
    std::atomic<uint64_t>        active_bytes_{0};
    mutable std::shared_mutex    mtx_;
    std::mutex                   cv_mtx_;
    std::condition_variable      cv_;
};

} // namespace fn_pin
