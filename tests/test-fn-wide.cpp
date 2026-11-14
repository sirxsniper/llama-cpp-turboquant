// [TAG_FN_WIDE_DECODE] CPU unit test of the wide decode rules (src/llama-fn-wide.h): the switch, the width, the ubatch
// rule, the observer rule and the extra reserve, each against the rule written out by enumeration, plus the bridge's
// decision for whole verify steps and prompt ubatches with the switch off and on. make_bridge below copies the width
// lines of llama_moe_bridge_create (src/llama-moe-bridge.cpp) rather than calling it: that function needs a model and
// a device. No GPU, no model. Exit code 0 iff every check passes.

#include "llama-fn-wide.h"

#include <algorithm>
#include <cstdio>
#include <utility>
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

namespace {

// llama_moe_bridge_create: max_t = min(16, max(1, LLAMA_MOE_BRIDGE_MAX_T or 8)), then capped by the ring (n_rs_seq x
// n_seq_max) when there is one; seq_t = n_rs_seq when below max_t, else 0 (= max_t)
struct bridge_cfg {
    int max_t  = 0;
    int seq_t  = 0;
    int wide_t = 0;
    int job_t  = 0;
};

bridge_cfg make_bridge(int env_max_t, int n_rs_seq, int n_seq_max, int mode) {
    bridge_cfg b;
    const int cap = n_rs_seq*n_seq_max;
    b.max_t  = std::min(16, std::max(1, env_max_t));
    if (cap > 0 && b.max_t > cap) {
        b.max_t = cap;
    }
    b.seq_t  = n_rs_seq > 0 && n_rs_seq < b.max_t ? n_rs_seq : 0;
    b.wide_t = fn_wide::width(mode, fn_wide::ring_rows(cap, n_rs_seq, b.seq_t > 0 ? b.seq_t : b.max_t));
    b.job_t  = fn_wide::job_t(b.max_t, b.wide_t);
    return b;
}

// llama_moe_bridge_fits / ring_ok as written in src/llama-moe-bridge.cpp
bool fits(const bridge_cfg & b, int n_tokens, int n_seq_tokens) {
    return n_tokens >= 1 && n_tokens <= b.max_t && n_seq_tokens <= (b.seq_t > 0 ? b.seq_t : b.max_t);
}

// the graph's test (src/llama-graph.cpp, the bridge post): fits(n_tokens, 1) or the wide rule, then the ring for a post.
// n_out: the ubatch's output rows (-1: every row)
bool on_bridge(const bridge_cfg & b, int n_tokens, bool equal, int n_seqs, int n_seq_tokens, int n_out = -1) {
    const bool wide = fn_wide::decode_ub(b.wide_t, n_tokens, equal, n_seqs, n_seq_tokens, b.seq_t > 0 ? b.seq_t : b.max_t,
            n_out < 0 ? n_tokens : n_out);
    const bool ring = n_seq_tokens <= (b.seq_t > 0 ? b.seq_t : b.max_t);
    return (fits(b, n_tokens, 1) || wide) && ring;
}

void test_mode() {
    for (int v = -3; v <= 20; ++v) {
        const int m = fn_wide::mode_from_int(v);
        TCHECK(m == (v == 1 ? 1 : v == 2 ? 2 : 0), "mode(%d) = %d", v, m);
    }
}

void test_ring_rows() {
    for (int np = 0; np <= 9; ++np) {
        for (int nrs = 0; nrs <= 6; ++nrs) {
            for (int lim = 0; lim <= 17; ++lim) {
                const int r = fn_wide::ring_rows(np*nrs, nrs, lim);
                const int want = np > 0 && nrs > 0 && lim > 0 ? np*std::min(lim, nrs) : 0;
                TCHECK(r == want, "ring_rows(%d, %d, %d) = %d, want %d", np*nrs, nrs, lim, r, want);
            }
        }
    }
    TCHECK(fn_wide::ring_rows(12, 3, 3) == 12, "np4 n_max 2");
    TCHECK(fn_wide::ring_rows(12, 3, 2) == 8, "np4, LLAMA_MOE_BRIDGE_MAX_T=2: 4 x 2");
    TCHECK(fn_wide::ring_rows(0, 3, 3) == 0, "no ring");
}

void test_width() {
    for (int mode = 0; mode <= 2; ++mode) {
        for (int rows = -1; rows <= 64; ++rows) {
            const int w = fn_wide::width(mode, rows);
            int want = 0;
            if (mode > 0 && rows > 8) {
                want = rows < 16 ? rows : 16;
            }
            TCHECK(w == want, "width(%d, %d) = %d, want %d", mode, rows, w, want);
            TCHECK(w == 0 || (w > fn_wide::BASE_T && w <= fn_wide::MAX_T), "width(%d, %d) = %d out of range", mode, rows, w);
        }
    }
    // the cases in the notes: n_rs_seq = n_max + 1 (the bridge's ring adds a row), --parallel 1..6, max_t 8
    struct row { int n_max, np, want; };
    const row rows[] = {
        { 2, 1, 0 }, { 2, 2, 0 }, { 2, 3, 9 }, { 2, 4, 12 }, { 3, 2, 0 }, { 3, 3, 12 }, { 3, 4, 16 }, { 1, 4, 0 }, { 1, 5, 10 },
        { 2, 6, 16 },
    };
    for (const row & r : rows) {
        const int nrs = r.n_max + 1;
        const int w = fn_wide::width(1, fn_wide::ring_rows(nrs*r.np, nrs, nrs < 8 ? nrs : 8));
        TCHECK(w == r.want, "n_max %d --parallel %d: width %d, want %d", r.n_max, r.np, w, r.want);
    }
}

void test_job_t() {
    for (int a = 1; a <= 16; ++a) {
        for (int w = 0; w <= 16; ++w) {
            const int j = fn_wide::job_t(a, w);
            TCHECK(j >= a && j >= w && (j == a || j == w), "job_t(%d, %d) = %d", a, w, j);
        }
    }
}

void test_decode_ub() {
    int n_true = 0;
    for (int wide_t = 0; wide_t <= 16; ++wide_t) {
        for (int seq_t = 1; seq_t <= 16; ++seq_t) {
            for (int n_seqs = 1; n_seqs <= 17; ++n_seqs) {
                for (int nst = 0; nst <= 17; ++nst) {
                    for (int eq = 0; eq <= 1; ++eq) {
                        for (int extra = 0; extra <= 1; ++extra) {
                            // extra = 1: a ubatch whose row count is not n_seqs x n_seq_tokens (a layer cut to its outputs)
                            const int n_tokens = n_seqs*nst + extra;
                            for (int miss = 0; miss <= 2; ++miss) {
                                // miss: rows without an output (a prompt chunk asks only for its last row)
                                const int n_out = n_tokens - miss;
                                const bool got = fn_wide::decode_ub(wide_t, n_tokens, eq != 0, n_seqs, nst, seq_t, n_out);
                                const bool want = wide_t > 0 && eq && n_seqs >= 2 && nst >= 1 && nst <= seq_t && extra == 0 &&
                                                  miss == 0 && n_tokens >= 9 && n_tokens <= wide_t;
                                TCHECK(got == want, "decode_ub(wide %d, rows %d, eq %d, seqs %d, nst %d, seq_t %d, out %d) = %d",
                                        wide_t, n_tokens, eq, n_seqs, nst, seq_t, n_out, (int) got);
                                n_true += got;
                            }
                        }
                    }
                }
            }
        }
    }
    TCHECK(n_true > 0, "no wide ubatch at all");
    // pinned: the verify steps of 3 and 4 drafting slots at n_max 2 (ring 3 rows per sequence)
    TCHECK(fn_wide::decode_ub(12, 12, true, 4, 3, 3, 12), "4 x 3 at width 12");
    TCHECK(fn_wide::decode_ub(9, 9, true, 3, 3, 3, 9), "3 x 3 at width 9");
    TCHECK(fn_wide::decode_ub(12, 9, true, 3, 3, 3, 9), "3 x 3 at width 12");
    TCHECK(!fn_wide::decode_ub(12, 12, true, 1, 12, 3, 12), "a 12-token prompt of one sequence");
    TCHECK(!fn_wide::decode_ub(12, 12, false, 4, 3, 3, 12), "not an equal split");
    TCHECK(!fn_wide::decode_ub(12, 8, true, 4, 2, 3, 8), "8 rows fit today");
    TCHECK(!fn_wide::decode_ub(0, 12, true, 4, 3, 3, 12), "switch off");
    TCHECK(!fn_wide::decode_ub(12, 16, true, 4, 4, 3, 16), "4 rows per sequence: more than the ring");
    // three verify steps and 3 prompt rows of a fourth slot (one output: its last row) in one 4 x 3 ubatch
    TCHECK(!fn_wide::decode_ub(12, 12, true, 4, 3, 3, 10), "a 4 x 3 ubatch with a prompt chunk");
    TCHECK(!fn_wide::decode_ub(12, 12, true, 4, 3, 3, 9), "a 4 x 3 ubatch with a prompt chunk, no output in it");
}

void test_obs() {
    for (int hot = 0; hot <= 8; ++hot) {
        for (int n = 1; n <= 40; ++n) {
            // no wide graph in flight: the old rule (n_tokens > hot_max_t = prompt), bit for bit
            TCHECK(fn_wide::obs_decode(n, hot, 0) == (n <= hot), "obs(%d, %d, 0)", n, hot);
            for (int obs = 1; obs <= 16; ++obs) {
                const bool want = n <= hot || (obs > 8 && n == obs);
                TCHECK(fn_wide::obs_decode(n, hot, obs) == want, "obs(%d, %d, %d)", n, hot, obs);
            }
        }
    }
}

void test_reserve() {
    for (int wide_t = 0; wide_t <= 16; ++wide_t) {
        for (int n_seqs = 0; n_seqs <= 17; ++n_seqs) {
            for (int seq_t = 0; seq_t <= 5; ++seq_t) {
                for (int n_tok : { 1, 8, 9, 12, 16, 512 }) {
                    for (int n_out_max : { 8, 9, 512 }) {
                        const auto sh = fn_wide::reserve_shapes(wide_t, n_seqs, seq_t, n_tok, n_out_max);
                        // the rule written out: every s x t the bridge takes as wide and the reserve can hold, once each
                        size_t want = 0;
                        for (int s = 2; s <= n_seqs; ++s) {
                            for (int t = 1; t <= seq_t; ++t) {
                                const int rows = s*t;
                                const bool in = wide_t > 0 && rows > 8 && rows <= wide_t && rows <= n_tok && rows <= n_out_max;
                                const bool got = std::find(sh.begin(), sh.end(), std::make_pair(s, t)) != sh.end();
                                TCHECK(got == in, "reserve_shapes(%d, %d, %d, %d, %d): %d x %d %s", wide_t, n_seqs, seq_t, n_tok,
                                        n_out_max, s, t, in ? "missing" : "extra");
                                want += in;
                            }
                        }
                        TCHECK(sh.size() == want, "reserve_shapes size %zu, want %zu", sh.size(), want);
                        for (const auto & p : sh) {
                            // each reserved graph is a wide decode ubatch itself (graph_reserve's ubatch: equal split, every
                            // row an output)
                            const int rows = p.first*p.second;
                            TCHECK(fn_wide::decode_ub(wide_t, rows, true, p.first, p.second, seq_t, rows), "shape %d x %d not wide",
                                    p.first, p.second);
                        }
                    }
                }
            }
        }
    }
    using shapes = std::vector<std::pair<int, int>>;
    TCHECK(fn_wide::reserve_shapes(12, 4, 3, 512, 512) == (shapes{ { 3, 3 }, { 4, 3 } }), "np4: 3 x 3 and 4 x 3");
    TCHECK(fn_wide::reserve_shapes(9, 3, 3, 512, 512) == (shapes{ { 3, 3 } }), "np3: 3 x 3");
    TCHECK(fn_wide::reserve_shapes(0, 2, 3, 512, 512).empty(), "np2");
    TCHECK(fn_wide::reserve_shapes(16, 6, 3, 512, 512) == (shapes{ { 3, 3 }, { 4, 3 }, { 5, 2 }, { 5, 3 }, { 6, 2 } }),
            "np6 at width 16");
    TCHECK(fn_wide::reserve_shapes(12, 4, 3, 8, 512).empty(), "a reserve of 8 rows holds no wide graph");
}

// the bridge's decision for whole ubatches, switch off against on, over LLAMA_MOE_BRIDGE_MAX_T 1..16 and every ring
void test_bridge() {
    int n_wide_steps = 0;
    for (int env = 1; env <= 16; ++env) {
        for (int n_max = 0; n_max <= 4; ++n_max) {
            const int nrs = n_max + 1; // the bridge's ring: one row more than the drafts' n_rs_seq
            for (int np = 1; np <= 8; ++np) {
                const bridge_cfg off = make_bridge(env, nrs, np, 0);
                const bridge_cfg on1 = make_bridge(env, nrs, np, 1);
                const bridge_cfg on2 = make_bridge(env, nrs, np, 2);
                TCHECK(off.wide_t == 0 && off.job_t == off.max_t, "off has a wide width");
                TCHECK(on1.max_t == off.max_t && on1.seq_t == off.seq_t && on2.wide_t == on1.wide_t, "max_t / seq_t moved");
                // decode ubatches: s sequences of t rows each
                for (int s = 1; s <= np; ++s) {
                    for (int t = 1; t <= nrs; ++t) {
                        const int rows = s*t;
                        const bool b_off = on_bridge(off, rows, true, s, t);
                        const bool b_on  = on_bridge(on1, rows, true, s, t);
                        // the switch only adds: a ubatch on the bridge today stays on it
                        TCHECK(!b_off || b_on, "env %d np %d: %d x %d left the bridge", env, np, s, t);
                        if (b_on != b_off) {
                            // and only what is wider than 8 rows, of 2+ sequences, within the ring and 16 rows
                            TCHECK(rows > 8 && rows <= 16 && s >= 2 && rows <= on1.wide_t && rows <= on1.job_t,
                                    "env %d np %d: %d x %d widened", env, np, s, t);
                            ++n_wide_steps;
                        }
                        // one and two connections never change (n_max <= 3: at most 8 rows)
                        if (np <= 2 && n_max <= 3) {
                            TCHECK(b_on == b_off, "np %d n_max %d: %d x %d changed", np, n_max, s, t);
                        }
                        // the verify step of every slot drafting, wider than 8 rows: on the bridge with the switch when it
                        // fits 16 rows and the ring holds its rows per sequence
                        if (s == np && t == nrs && rows > 8 && rows <= 16 && t <= (on1.seq_t > 0 ? on1.seq_t : on1.max_t)) {
                            TCHECK(b_on, "env %d np %d n_max %d: the %d-row verify step is off the bridge", env, np, n_max, rows);
                        }
                    }
                }
                // a ubatch with rows that are no output (a prompt chunk of another slot): the same decision with the switch
                for (int s = 2; s <= np; ++s) {
                    for (int t = 1; t <= nrs; ++t) {
                        TCHECK(on_bridge(off, s*t, true, s, t, s*t - 1) == on_bridge(on1, s*t, true, s, t, s*t - 1),
                                "env %d np %d: %d x %d with a prompt row changed", env, np, s, t);
                    }
                }
                // a wide width is always reachable: some equal split of the bridge's own limits has that many rows
                if (on1.wide_t > 0) {
                    const int lim = on1.seq_t > 0 ? on1.seq_t : on1.max_t;
                    TCHECK(on1.wide_t <= np*lim, "env %d np %d n_max %d: width %d, but %d x %d at most", env, np, n_max,
                            on1.wide_t, np, lim);
                }
                // prompt ubatches of one sequence and unequal splits: the same decision with the switch
                for (int n = 1; n <= 64; ++n) {
                    for (int nst = 1; nst <= std::min(n, 8); ++nst) {
                        TCHECK(on_bridge(off, n, true, 1, n) == on_bridge(on1, n, true, 1, n), "prompt %d changed", n);
                        TCHECK(on_bridge(off, n, false, 2, nst) == on_bridge(on1, n, false, 2, nst), "unequal %d changed", n);
                    }
                }
            }
        }
    }
    TCHECK(n_wide_steps > 0, "the switch widened nothing");
    // their setting: n_max 2 (ring 3), LLAMA_MOE_BRIDGE_MAX_T unset (8)
    const bridge_cfg np4 = make_bridge(8, 3, 4, 1);
    const bridge_cfg np2 = make_bridge(8, 3, 2, 1);
    TCHECK(np4.max_t == 8 && np4.wide_t == 12 && np4.job_t == 12 && np4.seq_t == 3, "np4: %d %d %d %d", np4.max_t, np4.wide_t,
            np4.job_t, np4.seq_t);
    TCHECK(np2.max_t == 6 && np2.wide_t == 0 && np2.job_t == 6, "np2: %d %d %d", np2.max_t, np2.wide_t, np2.job_t);
    TCHECK(!on_bridge(make_bridge(8, 3, 4, 0), 12, true, 4, 3) && on_bridge(np4, 12, true, 4, 3), "np4 12 rows");
    TCHECK(!on_bridge(np4, 12, true, 1, 12), "a 12-token prompt stays off the bridge at np4");
    // LLAMA_MOE_BRIDGE_MAX_T=12 by hand: the 12-row step fits already, the switch keeps max_t (and adds the hot chain and
    // the observer in the graph)
    const bridge_cfg h12 = make_bridge(12, 3, 4, 1);
    TCHECK(h12.max_t == 12 && h12.wide_t == 12 && h12.job_t == 12, "hand 12: %d %d %d", h12.max_t, h12.wide_t, h12.job_t);
    // LLAMA_MOE_BRIDGE_MAX_T=2 by hand at np4: 2 rows per sequence, 4 x 2 = 8 fit already, nothing to widen
    const bridge_cfg h2 = make_bridge(2, 3, 4, 1);
    TCHECK(h2.max_t == 2 && h2.wide_t == 0 && h2.job_t == 2, "hand 2: %d %d %d", h2.max_t, h2.wide_t, h2.job_t);
    // at np6 the same setting widens to 6 x 2 = 12
    const bridge_cfg h2n6 = make_bridge(2, 3, 6, 1);
    TCHECK(h2n6.wide_t == 12 && h2n6.job_t == 12, "hand 2 np6: %d %d", h2n6.wide_t, h2n6.job_t);
}

} // namespace

int main() {
    test_mode();
    test_ring_rows();
    test_width();
    test_job_t();
    test_decode_ub();
    test_obs();
    test_reserve();
    test_bridge();
    printf("test-fn-wide: %d checks, %d failures\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
