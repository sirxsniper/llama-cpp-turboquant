// [TAG_FN_L22_PIN] round 22 P6: the pure parts of the pinned expert ranges (src/llama-fn-pin-core.h), the code that
// src/llama-fn-pin.cpp and the probe (tools/moe-bridge-probe test 7) run. CPU only, no model, no ggml, no device.
//
// Checks: the unit (per-expert runs, inward page alignment, whole-tensor ranges of fully streamed layers, spans), the
// plan order, the budget rules and the plan budget, the registry (segments never cross a range end, unpinned segments
// stop at the next active range, arena copies, tickets, a drain waits for tickets with a bounded timeout, the LIFO
// unpin keeps busy units), and a concurrent query / drain run.

#include "../src/llama-fn-pin-core.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
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

static constexpr size_t PAGE = 4096;

// a page-aligned fake host memory block
struct block {
    std::vector<uint8_t> mem;
    uint8_t *            base = nullptr;
    explicit block(size_t n) : mem(n + PAGE) {
        base = (uint8_t *) fn_pin::align_up((uintptr_t) mem.data(), PAGE);
    }
};

static fn_pin::layer_in make_layer(int il, uint8_t * at, size_t stride, int n_exp, const std::vector<uint8_t> & ok, bool fully) {
    fn_pin::layer_in L;
    L.il = il;
    for (int k = 0; k < 3; ++k) {
        L.t[k].data     = at + (size_t) k*stride*n_exp + 777*(k + 1); // unaligned tensor starts
        L.t[k].stride   = stride;
        L.t[k].n_expert = n_exp;
    }
    L.ok             = ok;
    L.fully_streamed = fully;
    return L;
}

static void test_unit() {
    const size_t stride = 3*PAGE + 100; // not a page multiple
    const int    n_exp  = 8;
    block b(3*stride*n_exp + 16*PAGE);
    // a fully streamed layer, every expert pinnable: three whole-tensor ranges
    {
        auto L = make_layer(4, b.base, stride, n_exp, std::vector<uint8_t>(n_exp, 1), true);
        auto p = fn_pin::build_plan({ L }, PAGE, false);
        TCHECK(p.groups.size() == 1, "groups %zu", p.groups.size());
        TCHECK(p.ranges.size() == 3, "ranges %zu", p.ranges.size());
        for (int k = 0; k < (int) p.ranges.size(); ++k) {
            const auto & r = p.ranges[k];
            const uint8_t * s = L.t[r.part].data;
            const uint8_t * e = L.t[r.part].data + stride*n_exp;
            TCHECK((uintptr_t) r.p % PAGE == 0 && r.n % PAGE == 0, "alignment");
            TCHECK(r.p >= s && r.p + r.n <= e, "inside the tensor");
            TCHECK(r.p - s < (ptrdiff_t) PAGE && e - (r.p + r.n) < (ptrdiff_t) PAGE, "inward by less than a page");
            TCHECK(r.e0 == 0 && r.e1 == n_exp && r.group == 0 && r.il == 4, "run");
        }
        TCHECK(p.groups[0].whole, "whole layer");
    }
    // holes: runs [0,2) and [3,6); expert 2, 6, 7 never inside a range
    {
        std::vector<uint8_t> ok = { 1, 1, 0, 1, 1, 1, 0, 0 };
        auto L = make_layer(7, b.base, stride, n_exp, ok, false);
        auto p = fn_pin::build_plan({ L }, PAGE, false);
        TCHECK(p.groups.size() == 2, "groups %zu", p.groups.size());
        TCHECK(p.ranges.size() == 6, "ranges %zu", p.ranges.size());
        if (p.groups.size() == 2) {
            TCHECK(p.groups[0].e0 == 0 && p.groups[0].e1 == 2, "run 0");
            TCHECK(p.groups[1].e0 == 3 && p.groups[1].e1 == 6, "run 1");
            TCHECK(!p.groups[0].whole && p.groups[0].n_rng == 3 && p.groups[1].first == 3, "group layout");
        }
        for (const auto & r : p.ranges) {
            const uint8_t * t = L.t[r.part].data;
            for (int e = 0; e < n_exp; ++e) {
                if (ok[e]) {
                    continue;
                }
                const uint8_t * es = t + (size_t) e*stride;
                const uint8_t * ee = es + stride;
                TCHECK(r.p + r.n <= es || r.p >= ee, "range covers non-pinnable expert %d", e);
            }
            // the parts of one group are adjacent in the plan
            TCHECK(r.group == (r.e0 == 0 ? 0 : 1), "group id");
        }
    }
    // a run shorter than a page after alignment gives no range
    {
        const size_t small = 1000;
        block b2(3*small*4 + 4*PAGE);
        auto L = make_layer(1, b2.base, small, 4, { 1, 0, 0, 0 }, false);
        auto p = fn_pin::build_plan({ L }, PAGE, false);
        TCHECK(p.ranges.empty() && p.groups.empty(), "tiny run dropped (%zu)", p.ranges.size());
    }
    // max_range cuts a long range into page-multiple pieces of one group, adjacent and covering the same bytes
    {
        auto L = make_layer(4, b.base, stride, n_exp, std::vector<uint8_t>(n_exp, 1), true);
        auto p0 = fn_pin::build_plan({ L }, PAGE, false);
        auto p1 = fn_pin::build_plan({ L }, PAGE, false, 5*PAGE + 17);
        TCHECK(p1.groups.size() == 1 && p1.groups[0].bytes == p0.groups[0].bytes, "same bytes");
        TCHECK((int) p1.ranges.size() == p1.groups[0].n_rng && p1.ranges.size() > 3, "cut (%zu)", p1.ranges.size());
        size_t sum = 0;
        for (size_t i = 0; i < p1.ranges.size(); ++i) {
            const auto & r = p1.ranges[i];
            TCHECK(r.n <= 5*PAGE && r.n % PAGE == 0 && (uintptr_t) r.p % PAGE == 0 && r.group == 0, "piece");
            if (i > 0 && p1.ranges[i - 1].part == r.part) {
                TCHECK(p1.ranges[i - 1].p + p1.ranges[i - 1].n == r.p, "adjacent pieces");
            }
            sum += r.n;
        }
        TCHECK(sum == p0.groups[0].bytes, "pieces cover the range");
    }
    // spans: one span for a fully streamed layer with span_ok, ranges otherwise
    {
        auto L = make_layer(2, b.base, stride, n_exp, std::vector<uint8_t>(n_exp, 1), true);
        L.span_ok = true;
        auto p = fn_pin::build_plan({ L }, PAGE, true);
        TCHECK(p.ranges.size() == 1 && p.ranges[0].part == -1, "span");
        if (!p.ranges.empty()) {
            TCHECK(p.ranges[0].p >= L.t[0].data && p.ranges[0].p + p.ranges[0].n <= L.t[2].data + stride*n_exp, "span bounds");
        }
        L.span_ok = false;
        p = fn_pin::build_plan({ L }, PAGE, true);
        TCHECK(p.ranges.size() == 3, "no span without span_ok");
        auto L2 = make_layer(3, b.base, stride, n_exp, { 1, 1, 1, 1, 1, 1, 1, 0 }, true);
        L2.span_ok = true;
        p = fn_pin::build_plan({ L2 }, PAGE, true);
        TCHECK(p.ranges.size() == 3 && p.ranges[0].part == 0, "no span with an SSD / VRAM expert");
    }
}

static void test_order() {
    const size_t stride = 2*PAGE;
    const int    n_exp  = 4;
    block b(4*3*stride*n_exp + 64*PAGE);
    std::vector<fn_pin::layer_in> ls;
    // il 1 partial, small CPU use; il 2 partial, big CPU use; il 5 and il 3 fully streamed
    ls.push_back(make_layer(1, b.base + 0*3*stride*n_exp + 0*PAGE, stride, n_exp, { 1, 1, 0, 1 }, false));
    ls.back().cpu_use = 1.0;
    ls.push_back(make_layer(2, b.base + 1*3*stride*n_exp + 4*PAGE, stride, n_exp, { 1, 1, 0, 1 }, false));
    ls.back().cpu_use = 9.0;
    ls.push_back(make_layer(5, b.base + 2*3*stride*n_exp + 8*PAGE, stride, n_exp, { 1, 1, 1, 1 }, true));
    ls.push_back(make_layer(3, b.base + 3*3*stride*n_exp + 12*PAGE, stride, n_exp, { 1, 1, 1, 1 }, true));
    auto p = fn_pin::build_plan(ls, PAGE, false);
    std::vector<int> il_order;
    for (const auto & g : p.groups) {
        if (il_order.empty() || il_order.back() != g.il) {
            il_order.push_back(g.il);
        }
    }
    TCHECK(il_order.size() == 4, "layers %zu", il_order.size());
    if (il_order.size() == 4) {
        TCHECK(il_order[0] == 3 && il_order[1] == 5, "fully streamed first, in layer order (%d %d)", il_order[0], il_order[1]);
        TCHECK(il_order[2] == 2 && il_order[3] == 1, "then by bytes x CPU use (%d %d)", il_order[2], il_order[3]);
    }
    // groups list their ranges contiguously and the ranges point back
    for (int g = 0; g < (int) p.groups.size(); ++g) {
        for (int k = 0; k < p.groups[g].n_rng; ++k) {
            TCHECK(p.ranges[p.groups[g].first + k].group == g, "range -> group");
        }
    }
    // plan_take keeps whole groups in order
    uint64_t two = p.groups[0].bytes + p.groups[1].bytes;
    TCHECK(fn_pin::plan_take(p, two) == 2, "take 2");
    TCHECK(fn_pin::plan_take(p, two - 1) == 1, "take 1");
    TCHECK(fn_pin::plan_take(p, 0) == 0, "take 0");
}

static void test_rules() {
    const uint64_t G = 1ull << 30;
    fn_pin::budget_cfg c;
    c.nl_margin = 8*G;
    c.l_keep    = 2*G;
    c.floor     = 6*G;
    c.max_pin   = 0;
    fn_pin::budget_in b;
    b.have_nl = true;  b.nl_budget = 48*G; b.nl_usage = 2*G;
    b.have_l  = true;  b.l_budget  = 31*G; b.l_usage  = 28*G;
    b.avail   = 40*G;
    TCHECK(fn_pin::check(b, c, 1*G, 1*G) == fn_pin::RULE_OK, "ok");
    TCHECK(fn_pin::check(b, c, 39*G, 0) == fn_pin::RULE_NONLOCAL, "nonlocal");
    c.l_ratio = 1.0;
    TCHECK(fn_pin::check(b, c, 2*G, 0) == fn_pin::RULE_LOCAL, "local with a charge");
    c.l_ratio = 0.0;
    b.l_usage = 30*G;
    TCHECK(fn_pin::check(b, c, 0, 0) == fn_pin::RULE_LOCAL, "local headroom below keep");
    b.l_usage = 28*G;
    TCHECK(fn_pin::check(b, c, 1*G, 35*G) == fn_pin::RULE_RAM, "ram");
    c.warm_keep = 34*G;
    TCHECK(fn_pin::check(b, c, 1*G, 1*G) == fn_pin::RULE_RAM, "ram with warm keep");
    c.warm_keep = 0;
    c.max_pin = 4*G;
    b.pinned  = 3*G;
    TCHECK(fn_pin::check(b, c, 2*G, 0) == fn_pin::RULE_MAX, "max");
    // the plan budget: min(NON_LOCAL room / ratio, RAM - floor, cap - pinned)
    c.max_pin = 0;
    b.pinned  = 0;
    TCHECK(fn_pin::plan_bytes(b, c) == 34*G, "plan bytes ram-bound %.2f", fn_pin::plan_bytes(b, c)/(double) G);
    b.avail = 90*G;
    TCHECK(fn_pin::plan_bytes(b, c) == 38*G, "plan bytes nonlocal-bound %.2f", fn_pin::plan_bytes(b, c)/(double) G);
    c.nl_ratio = 0.5;
    TCHECK(fn_pin::plan_bytes(b, c) == 76*G, "plan bytes with ratio 0.5");
    c.max_pin = 10*G;
    TCHECK(fn_pin::plan_bytes(b, c) == 10*G, "plan bytes capped");
    b.l_usage = 30*G;
    TCHECK(fn_pin::plan_bytes(b, c) == 0, "no LOCAL room: nothing");
    b.have_nl = false;
    b.have_l  = false;
    c.max_pin = 0;
    TCHECK(fn_pin::plan_bytes(b, c) == 84*G, "no DXGI (the caller decides): RAM only");
}

// three ranges with gaps: [b+1p, +4p) [b+6p, +2p) [b+8p, +4p) (the last two touch)
static void make_reg(fn_pin::registry & r, uint8_t * base) {
    std::vector<fn_pin::registry::in_range> rs(3);
    rs[0].p = base + 1*PAGE; rs[0].n = 4*PAGE; rs[0].unit = 0; rs[0].src = 0;
    rs[2].p = base + 6*PAGE; rs[2].n = 2*PAGE; rs[2].unit = 1; rs[2].src = 1;
    rs[1].p = base + 8*PAGE; rs[1].n = 4*PAGE; rs[1].unit = 2; rs[1].src = 2;
    TCHECK(r.build(rs, 3), "build");
    for (int u = 0; u < 3; ++u) {
        r.set_unit(u, (void *) (base + PAGE*(u == 0 ? 1 : u == 1 ? 6 : 8)), u == 1 ? 2*PAGE : 4*PAGE);
    }
}

static void test_registry() {
    block b(32*PAGE);
    uint8_t * B = b.base;
    fn_pin::registry r;
    make_reg(r, B);
    fn_pin::seg s;
    // nothing active: one unpinned segment
    r.query(B, 20*PAGE, &s);
    TCHECK(!s.pinned && s.len == 20*PAGE && s.src == B && s.ticket == -1, "inactive");
    r.activate(0);
    r.activate(2);
    TCHECK(r.n_active_units() == 2 && r.active_bytes() == 8*PAGE, "active %d", r.n_active_units());
    // before range 0: up to its start
    r.query(B, 20*PAGE, &s);
    TCHECK(!s.pinned && s.len == 1*PAGE, "unpinned up to the first active range (%zu)", s.len / PAGE);
    // inside range 0: pinned up to its end
    r.query(B + 2*PAGE + 5, 20*PAGE, &s);
    TCHECK(s.pinned && s.src == B + 2*PAGE + 5 && s.len == 3*PAGE - 5 && s.ticket >= 0, "pinned to the range end");
    TCHECK(r.seg_dev(s) == 0, "no device address known");
    const int32_t t0 = s.ticket;
    // the gap and the inactive unit 1 are skipped up to range 2
    r.query(B + 5*PAGE, 20*PAGE, &s);
    TCHECK(!s.pinned && s.len == 3*PAGE, "unpinned across the inactive range (%zu)", s.len / PAGE);
    // a short query inside a range
    r.query(B + 9*PAGE, 100, &s);
    TCHECK(s.pinned && s.len == 100, "short pinned");
    const int32_t t1 = s.ticket;
    // past the last range
    r.query(B + 12*PAGE, 4*PAGE, &s);
    TCHECK(!s.pinned && s.len == 4*PAGE, "after the last range");
    TCHECK(r.users_total() == 2, "two tickets out (%d)", r.users_total());

    // drain unit 0 while its ticket is out: times out, unit stays active and usable
    int unregs = 0;
    auto unreg = [&](const fn_pin::reg_unit &) { unregs++; };
    TCHECK(!r.drain(0, 20000, unreg) && unregs == 0, "drain waits for the ticket");
    r.query(B + 1*PAGE, PAGE, &s);
    TCHECK(s.pinned, "still active after a failed drain");
    r.release(s.ticket);
    r.release(t0);
    TCHECK(r.drain(0, 20000, unreg) && unregs == 1, "drain after release");
    r.query(B + 1*PAGE, PAGE, &s);
    TCHECK(!s.pinned, "gone");
    TCHECK(r.n_active_units() == 1 && r.active_bytes() == 4*PAGE, "one active");
    // a release on another thread wakes a waiting drain
    std::thread rel([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        r.release(t1);
    });
    const auto w0 = std::chrono::steady_clock::now();
    TCHECK(r.drain(2, 2000000, unreg) && unregs == 2, "drain woken by the release");
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();
    TCHECK(ms < 1500.0, "woken, not timed out (%.0f ms)", ms);
    rel.join();
    TCHECK(!r.any_active(), "none active");
    // a late release of a gone range is harmless
    r.release(t1);
    r.release(-1);
    r.release(999);
}

static void test_arena_copy() {
    block b(32*PAGE);
    block c(8*PAGE);
    fn_pin::registry r;
    std::vector<fn_pin::registry::in_range> rs(2);
    rs[0].p = b.base + 2*PAGE; rs[0].n = 2*PAGE; rs[0].copy = (const uint8_t *) (uintptr_t) (4*PAGE); rs[0].unit = 0;
    rs[1].p = b.base + 10*PAGE; rs[1].n = 3*PAGE; rs[1].copy = (const uint8_t *) (uintptr_t) (0); rs[1].unit = 0;
    TCHECK(r.build(rs, 1), "build");
    r.set_unit(0, c.base, 8*PAGE);
    r.set_unit_dev(0, 0x7000000000ull);
    r.fix_copies(0, c.base);
    r.activate(0);
    fn_pin::seg s;
    r.query(b.base + 3*PAGE + 7, 10*PAGE, &s);
    TCHECK(s.pinned && s.src == c.base + 4*PAGE + PAGE + 7 && s.len == PAGE - 7, "arena address of range 0");
    TCHECK(r.seg_dev(s) == 0x7000000000ull + 5*PAGE + 7, "device address of the copy");
    r.release(s.ticket);
    r.query(b.base + 10*PAGE, 3*PAGE, &s);
    TCHECK(s.pinned && s.src == c.base && s.len == 3*PAGE, "arena address of range 1");
    r.release(s.ticket);
    int unregs = 0;
    TCHECK(r.drain(0, 1000, [&](const fn_pin::reg_unit & U) { unregs++; TCHECK(U.p == c.base && U.ranges.size() == 2, "unit"); }), "drain");
    TCHECK(unregs == 1, "one unregister for the chunk");
    // overlapping input ranges are refused (dropped)
    fn_pin::registry r2;
    std::vector<fn_pin::registry::in_range> ov(2);
    ov[0].p = b.base; ov[0].n = 4*PAGE; ov[0].unit = 0;
    ov[1].p = b.base + 3*PAGE; ov[1].n = 4*PAGE; ov[1].unit = 1;
    TCHECK(!r2.build(ov, 2) && r2.n_ranges() == 1, "overlap dropped");
}

static void test_lifo() {
    block b(64*PAGE);
    fn_pin::registry r;
    std::vector<fn_pin::registry::in_range> rs(4);
    for (int i = 0; i < 4; ++i) {
        rs[i].p = b.base + (size_t) (1 + 4*i)*PAGE;
        rs[i].n = 2*PAGE;
        rs[i].unit = i;
    }
    TCHECK(r.build(rs, 4), "build");
    for (int i = 0; i < 4; ++i) {
        r.set_unit(i, (void *) rs[i].p, rs[i].n);
    }
    r.activate(0);
    r.activate(2);
    r.activate(1);
    r.activate(3); // order 0, 2, 1, 3
    // a ticket on unit 1 keeps it
    fn_pin::seg s;
    r.query(b.base + 5*PAGE, PAGE, &s);
    TCHECK(s.pinned && s.ticket >= 0, "ticket on unit 1");
    std::vector<int> gone;
    int n_unreg = 0;
    uint64_t bytes = 0;
    // unpin until 3 units are gone or none left
    const int n = r.unpin_lifo([&]() { return n_unreg >= 3; }, 1000, [&](const fn_pin::reg_unit &) { n_unreg++; }, &bytes, &gone);
    TCHECK(n == 3 && gone.size() == 3, "three gone (%d)", n);
    if (gone.size() == 3) {
        TCHECK(gone[0] == 3 && gone[1] == 2 && gone[2] == 0, "LIFO order, busy unit 1 skipped (%d %d %d)", gone[0], gone[1], gone[2]);
    }
    TCHECK(r.n_active_units() == 1 && bytes == 6*PAGE, "unit 1 left");
    r.release(s.ticket);
    gone.clear();
    TCHECK(r.unpin_lifo(nullptr, 1000, nullptr, &bytes, &gone) == 1 && gone.size() == 1 && gone[0] == 1, "unit 1 back on the stack");
    TCHECK(r.pop_lifo() == -1, "empty");
}

// queries and releases from 4 threads while the main thread drains every unit: no ticket is lost, every drain ends
static void test_concurrent() {
    block b(256*PAGE);
    fn_pin::registry r;
    const int n_u = 32;
    std::vector<fn_pin::registry::in_range> rs(n_u);
    for (int i = 0; i < n_u; ++i) {
        rs[i].p = b.base + (size_t) (8*i)*PAGE;
        rs[i].n = 6*PAGE;
        rs[i].unit = i;
    }
    TCHECK(r.build(rs, n_u), "build");
    for (int i = 0; i < n_u; ++i) {
        r.set_unit(i, (void *) rs[i].p, rs[i].n);
        r.activate(i);
    }
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> n_pinned{0};
    std::vector<std::thread> th;
    for (int t = 0; t < 4; ++t) {
        th.emplace_back([&, t]() {
            uint32_t x = 12345u + 77u*t;
            while (!stop.load()) {
                x = x*1664525u + 1013904223u;
                const size_t off = (x >> 8) % (256*PAGE - 2*PAGE);
                fn_pin::seg s;
                r.query(b.base + off, 2*PAGE, &s);
                if (s.pinned) {
                    n_pinned.fetch_add(1);
                    std::this_thread::yield();
                    r.release(s.ticket);
                }
            }
        });
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    int drained = 0;
    for (int round = 0; round < 50 && r.any_active(); ++round) {
        drained += r.unpin_lifo(nullptr, 200000, nullptr, nullptr);
    }
    stop.store(true);
    for (auto & x : th) {
        x.join();
    }
    TCHECK(drained == n_u && !r.any_active(), "all drained (%d)", drained);
    TCHECK(r.users_total() == 0, "no ticket left (%d)", r.users_total());
    TCHECK(n_pinned.load() > 0, "queries saw pinned ranges");
}

// R-REV-1: the OOM hook's cap never lets one call drain every range when DXGI shows no charge per registered byte
static void test_oom_cap() {
    const uint64_t M = 1ull << 20;
    const uint64_t G = 1ull << 30;
    TCHECK(fn_pin::oom_cap_bytes(512*M, 400*M, false, 0.0) == 512*M, "no charge seen: want bytes only");
    TCHECK(fn_pin::oom_cap_bytes(512*M, 400*M, false, 0.01) == 512*M, "charge below the minimum: want bytes only");
    TCHECK(fn_pin::oom_cap_bytes(512*M, 400*M, true, 1.0) == 512*M, "by bytes: want");
    TCHECK(fn_pin::oom_cap_bytes(512*M, 600*M, false, 1.0) == 512*M, "no deficit: want");
    TCHECK(fn_pin::oom_cap_bytes(512*M, 400*M, false, 1.0) == 512*M, "small deficit at ratio 1: at least want");
    const uint64_t c1 = fn_pin::oom_cap_bytes(1*G, 0, false, 0.1);
    TCHECK(c1 == (uint64_t) (1.25*(double) G/0.1) + 256*M, "deficit / ratio x 1.25 + 256 MiB (%.2f GiB)", c1/(double) G);
    const uint64_t c2 = fn_pin::oom_cap_bytes(4*G, 1*G, false, 0.5);
    TCHECK(c2 == (uint64_t) (1.25*3.0*(double) G/0.5) + 256*M, "deficit 3 GiB at ratio 0.5 (%.2f GiB)", c2/(double) G);
    TCHECK(fn_pin::oom_cap_bytes(4*G, -1*(int64_t) G, false, 2.0) >= 4*G, "negative headroom");
}

// [TAG_FN_L23_PINFIX] R-L23-1: the OOM hook unpins only what can raise the room; R-L23-2: the valve's room
static void test_pinfix() {
    const uint64_t M = 1ull << 20;
    const uint64_t G = 1ull << 30;
    fn_pin::oom_in o;
    TCHECK(fn_pin::oom_help_bytes(8*G, o) == 8*G, "DXGI LOCAL unknown: want bytes (as before)");
    // r22 T13: 7573 MiB + 512 MiB, LOCAL headroom 6.85 GiB, charge 0, NON_LOCAL 65.39 / 73.75 GiB: nothing helps
    o.have_l = true; o.l_head = (int64_t) (6.85*(double) G); o.l_ratio = 0.0;
    o.have_nl = true; o.nl_budget = (uint64_t) (73.75*(double) G); o.nl_usage = (uint64_t) (65.39*(double) G);
    const uint64_t want = 7573*M + 512*M;
    TCHECK(fn_pin::oom_help_bytes(want, o) == 0, "T13 case: no charge, NON_LOCAL holds the spill: no unpin");
    o.have_nl = false;
    TCHECK(fn_pin::oom_help_bytes(want, o) == 0, "no charge, NON_LOCAL unknown, no commit: no unpin");
    // NON_LOCAL short of the spill: the deficit + 256 MiB, at most want
    o.have_nl = true; o.nl_usage = o.nl_budget - 512*M;
    const uint64_t spill = want - (uint64_t) o.l_head;
    TCHECK(fn_pin::oom_help_bytes(want, o) == spill - 512*M + 256*M, "NON_LOCAL deficit + 256 MiB (%.0f MiB)",
            fn_pin::oom_help_bytes(want, o)/(double) M);
    o.l_head = -1;
    o.nl_usage = o.nl_budget;
    TCHECK(fn_pin::oom_help_bytes(want, o) == want, "no LOCAL room, NON_LOCAL full: want (capped)");
    // the arena: commit short
    o.l_head = 20*(int64_t) G; o.nl_usage = 0; o.have_commit = true; o.commit_room = 1*G;
    TCHECK(fn_pin::oom_help_bytes(2*G, o) == 1*G + 256*M, "arena commit deficit + 256 MiB");
    o.commit_room = 4*G;
    TCHECK(fn_pin::oom_help_bytes(2*G, o) == 0, "arena with commit room: no unpin");
    // a measured LOCAL charge: as oom_cap_bytes
    o.have_commit = false; o.l_ratio = 0.5; o.l_head = 1*(int64_t) G;
    TCHECK(fn_pin::oom_help_bytes(4*G, o) == fn_pin::oom_cap_bytes(4*G, 1*G, false, 0.5), "LOCAL charge: the R-REV-1 cap");
    o.l_head = 5*(int64_t) G;
    TCHECK(fn_pin::oom_help_bytes(4*G, o) == 4*G, "LOCAL charge, headroom shows room: want bytes");
    // the gate: 3 samples under 4 GiB drop every pin; 30 samples (60 s) >= 10 GiB, 30 samples after the drop, re-pin
    fn_pin::gate_state gs;
    const uint64_t D = 4*G, P = 10*G;
    TCHECK(fn_pin::gate_step(gs, 3*G, true, false, D, 3, P, 30) == 0, "1 sample under: nothing");
    TCHECK(fn_pin::gate_step(gs, 3*G, true, false, D, 3, P, 30) == 0, "2 samples under: nothing");
    TCHECK(fn_pin::gate_step(gs, 5*G, true, false, D, 3, P, 30) == 0, "back over: the count restarts");
    TCHECK(fn_pin::gate_step(gs, 3*G, true, false, D, 3, P, 30) == 0, "1 again");
    TCHECK(fn_pin::gate_step(gs, 3*G, true, false, D, 3, P, 30) == 0, "2 again");
    TCHECK(fn_pin::gate_step(gs, 3*G, true, false, D, 3, P, 30) == -1, "3 samples under 4 GiB: drop all");
    TCHECK(fn_pin::gate_step(gs, 1*G, false, true, D, 3, P, 30) == 0, "no pins: no drop under pressure");
    int got = 0, k = 0;
    for (k = 0; k < 29; ++k) {
        got = fn_pin::gate_step(gs, 11*G, false, true, D, 3, P, 30);
        TCHECK(got == 0, "room for %d samples: not yet", k + 1);
    }
    got = fn_pin::gate_step(gs, 11*G, false, true, D, 3, P, 30);
    TCHECK(got == 1, "room for 30 samples, 30+ after the drop: re-pin");
    for (k = 0; k < 10; ++k) {
        got = fn_pin::gate_step(gs, 9*G, true, true, D, 3, P, 30);
        TCHECK(got == 0, "between 4 and 10 GiB with pins: nothing (%d)", k);
    }
    TCHECK(fn_pin::gate_step(gs, 11*G, true, false, D, 3, P, 30) == 0, "nothing to re-pin: nothing");
}

// [TAG_FN_L23_PINFIX] R-L23-6: the pins go on a failed allocation only when they can be the cause
static void test_alloc_fail_reason() {
    const uint64_t M = 1ull << 20;
    const uint64_t G = 1ull << 30;
    fn_pin::af_in a;
    // l23p B6 14:23:48: 640.9 MiB VRAM, no LOCAL charge, NON_LOCAL room ~16.2 GiB >= 16 GiB margin, LOCAL headroom 2.9 GiB
    a.kind = fn_pin::AF_DEV; a.need = 641*M; a.have_l = true; a.l_head = (int64_t) (2.9*(double) G); a.l_ratio = 0.0;
    a.have_nl = true; a.nl_room = (uint64_t) (16.2*(double) G); a.nl_margin = 16*G;
    TCHECK(fn_pin::alloc_fail_reason(a) == fn_pin::AFR_NONE, "B6 case: plain VRAM failure, pins stay");
    a.kind = fn_pin::AF_DEV_FATAL;
    TCHECK(fn_pin::alloc_fail_reason(a) == fn_pin::AFR_NONE, "fatal site: reason none (the caller unpins partly)");
    a.kind = fn_pin::AF_HOST;
    TCHECK(fn_pin::alloc_fail_reason(a) == fn_pin::AFR_HOST, "pinned host allocation: pins can be the cause");
    a.kind = fn_pin::AF_DEV; a.l_ratio = 0.2;
    TCHECK(fn_pin::alloc_fail_reason(a) == fn_pin::AFR_LOCAL_CHARGE, "LOCAL charge per pinned byte");
    a.l_ratio = 0.0; a.have_nl = false;
    TCHECK(fn_pin::alloc_fail_reason(a) == fn_pin::AFR_NL_UNKNOWN, "NON_LOCAL unknown: drop (safe)");
    // FAULT 12:48: NON_LOCAL room 8.36 GiB < 16 GiB margin
    a.have_nl = true; a.nl_room = (uint64_t) (8.36*(double) G);
    TCHECK(fn_pin::alloc_fail_reason(a) == fn_pin::AFR_NL_ROOM, "12:48 case: NON_LOCAL room under the margin");
    // a spill past LOCAL that NON_LOCAL cannot hold (margin 0)
    a.nl_margin = 0; a.nl_room = 1*G; a.need = 8*G; a.l_head = 2*(int64_t) G;
    TCHECK(fn_pin::alloc_fail_reason(a) == fn_pin::AFR_NL_SPILL, "spill 6 GiB > NON_LOCAL room 1 GiB");
    a.need = 2*G + 512*M;
    TCHECK(fn_pin::alloc_fail_reason(a) == fn_pin::AFR_NONE, "spill 0.5 GiB fits NON_LOCAL room 1 GiB");
}

int main() {
    test_alloc_fail_reason();
    test_unit();
    test_order();
    test_rules();
    test_oom_cap();
    test_pinfix();
    test_registry();
    test_arena_copy();
    test_lifo();
    test_concurrent();
    printf("test-fn-pin: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
