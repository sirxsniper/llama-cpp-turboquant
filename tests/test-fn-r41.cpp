// [TAG_FN_R41] CPU unit test of round 41's rules (src/llama-fn-r41.h, header only: the engine runs the same code):
//   - [TAG_FN_R41_SLOWJOB] the MoE bridge's error policy: the v0.33.0 rules with the switch off (the owner's three timeouts
//     of 14 Nov turn the bridge off), and with it on slow jobs that the host answers are not counted, a hang, a storm of
//     back-to-back timeouts and a steady stream (more than 3 slow jobs within 1024 bridged graphs) still are
//   - [TAG_FN_R41_FLOOR] the free-RAM floor's gate: hold, chunk, gap, hysteresis
//   - [TAG_FN_R41_PRIO] the priority rule, and on Windows the real raise of this process's and this thread's lowered I/O
//     and page priority
// No GPU, no model. Exit code 0 iff every check passes.

#include "../src/llama-fn-r41.h"

#include <cstdint>
#include <cstdio>

static int g_fail   = 0;
static int g_checks = 0;

#define TCHECK(cond, ...)                                        \
    do {                                                         \
        ++g_checks;                                              \
        if (!(cond)) {                                           \
            ++g_fail;                                            \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                        \
            fprintf(stderr, "\n");                               \
        }                                                        \
    } while (0)

using pol_t = fn_r41::bridge_policy;

// the profile's bridge: LLAMA_FN_HOST_REARM=1 (256 steps or 30 s)
static pol_t make_pol(bool slow) {
    pol_t p;
    p.rearm_steps = 256;
    p.rearm_ms    = 30000;
    p.slow_on     = slow;
    return p;
}

// steps until the bridge is active again (reset answers ok), at most max_steps; -1 if it never is
static int steps_to_rearm(pol_t & p, uint64_t & n_graphs, int64_t & now, bool host_done, int max_steps, unsigned * ev_all) {
    for (int s = 1; s <= max_steps; ++s) {
        now += 10; // a step of 10 ms
        const unsigned ev = p.on_step([host_done]() { return host_done; }, n_graphs, now);
        if (ev_all) {
            *ev_all |= ev;
        }
        if (p.active && !p.disabled) {
            return s;
        }
    }
    return -1;
}

static void test_v0330_rules() {
    // the switch off: exactly v0.33.0 (TAG_FN_R1_BRIDGE_RB, TAG_FN_L7_PAUSE1, TAG_FN_L3_HOST_REARM)
    pol_t p = make_pol(false);
    uint64_t g = 100;
    int64_t now = 1000;
    TCHECK(p.on_error(true, g, now) == pol_t::E_PAUSE1 && !p.active && p.pause_left == 1, "first error: the retry only");
    TCHECK(steps_to_rearm(p, g, now, true, 100, nullptr) == 2, "re-armed at the second step (one paused step)");
    g += 120;
    TCHECK(p.on_error(true, g, now) == pol_t::E_PAUSE16 && p.pause_left == 16, "second within 1024 graphs: 16 steps");
    TCHECK(steps_to_rearm(p, g, now, true, 100, nullptr) == 17, "re-armed after 16 paused steps");
    g += 120;
    TCHECK(p.on_error(true, g, now) == pol_t::E_OFF_REARM && p.disabled && p.off_steps == 256 && p.off_ms == 30000 && p.n_off == 1,
           "third: off for 256 steps or 30 s (turn-off 1)");
    unsigned ev = 0;
    TCHECK(steps_to_rearm(p, g, now, true, 1000, &ev) == 256 && (ev & pol_t::EV_COOLED) && (ev & pol_t::EV_REARMED),
           "the cool-down ends after its steps, then re-armed");
    TCHECK(p.n_errors == 0, "the cool-down clears the count");
    // the second turn-off doubles, the cool-down also ends by time
    g += 50;
    p.on_error(true, g, now);
    g += 50;
    p.on_error(true, g, now);
    g += 50;
    TCHECK(p.on_error(true, g, now) == pol_t::E_OFF_REARM && p.off_steps == 512 && p.off_ms == 60000 && p.n_off == 2,
           "turn-off 2: 512 steps or 60 s");
    now += 60001;
    TCHECK(steps_to_rearm(p, g, now, true, 10, nullptr) == 1, "the time ends a cool-down too");
    // errors far apart are not counted together
    pol_t q = make_pol(false);
    g = 10;
    q.on_error(true, g, now);
    steps_to_rearm(q, g, now, true, 10, nullptr);
    g += 2000;
    TCHECK(q.on_error(true, g, now) == pol_t::E_PAUSE1 && q.n_errors == 1, "an error 2000 graphs later starts over");
    // a host that owes the job keeps the bridge paused
    TCHECK(steps_to_rearm(q, g, now, false, 50, nullptr) == -1 && !q.active, "no re-arm while the host owes a job");
    // without the re-arm switch the third error is for good
    pol_t r;
    r.on_error(true, 10, now);
    r.on_error(false, 20, now);
    TCHECK(r.on_error(true, 30, now) == pol_t::E_OFF && r.disabled, "no REARM: off for the context");
    TCHECK(steps_to_rearm(r, g, now, true, 300, nullptr) == -1, "and it never comes back");
}

// the owner's 07:41 session of 14 Nov (llama-1794642109-port-18080.log 14.04-14.12): wait timeouts at layers 15, 22 and 8,
// about 4 s (~120 bridged graphs) apart, each late job answered within the second
static int owner_sequence(pol_t & p, int * n_off, int * n_paused16) {
    uint64_t g = 5000;
    int64_t now = 840000;
    *n_off = 0;
    *n_paused16 = 0;
    int paused = 0;
    for (int i = 0; i < 3; ++i) {
        const auto k = p.on_error(true, g, now);
        *n_off += k == pol_t::E_OFF_REARM || k == pol_t::E_OFF ? 1 : 0;
        *n_paused16 += k == pol_t::E_PAUSE16 ? 1 : 0;
        // the host answers the late job 400 ms later
        for (int s = 0; s < 40 && !(p.active && !p.disabled); ++s) {
            now += 10;
            p.on_step([&]() { return s >= 39 || now % 1000 > 400; }, g, now);
            paused++;
        }
        for (int s = 0; s < 400 && !(p.active && !p.disabled); ++s) { // a cool-down, if any
            now += 25;
            p.on_step([]() { return true; }, g, now);
            paused++;
        }
        g   += 120;
        now += 4000;
    }
    return paused;
}

static void test_slowjob() {
    {
        pol_t p = make_pol(false);
        int n_off = 0, n16 = 0;
        const int paused = owner_sequence(p, &n_off, &n16);
        TCHECK(n_off == 1 && n16 == 1, "v0.33.0: the owner's three timeouts pause 16 steps, then turn the bridge off");
        TCHECK(paused >= 256, "v0.33.0: %d steps without the bridge", paused);
    }
    {
        pol_t p = make_pol(true);
        int n_off = 0, n16 = 0;
        const int paused = owner_sequence(p, &n_off, &n16);
        TCHECK(n_off == 0 && n16 == 0 && p.n_errors == 0 && p.n_slow == 3, "SLOWJOB: three slow jobs, no error, never off");
        TCHECK(paused < 3*45, "SLOWJOB: %d steps without the bridge (each until its late job is answered)", paused);
    }
    // a slow job waits for the host, however many steps that takes, until the ceiling
    {
        pol_t p = make_pol(true);
        p.slow_ceil_ms = 20000;
        uint64_t g = 1000;
        int64_t now = 0;
        TCHECK(p.on_error(true, g, now) == pol_t::E_SLOW && p.slow_pending && p.pause_left == 1 && !p.active, "a timeout waits");
        TCHECK(p.on_step([]() { return true; }, g, now + 10) == 0, "the retried ubatch's step stays paused");
        TCHECK(p.on_step([]() { return false; }, g, now + 5000) == 0 && !p.active, "the host is still on it at 5 s");
        const unsigned ev = p.on_step([]() { return true; }, g, now + 6000);
        TCHECK((ev & pol_t::EV_SLOW_OK) && (ev & pol_t::EV_REARMED) && p.active && p.slow_ms == 6000 && p.n_errors == 0,
               "answered at 6 s: re-armed, not an error");
    }
    // a hang: not answered within the ceiling, then counted by the v0.33.0 rules
    {
        pol_t p = make_pol(true);
        p.slow_ceil_ms = 20000;
        uint64_t g = 1000;
        int64_t now = 0;
        p.on_error(true, g, now);
        p.on_step([]() { return false; }, g, now + 10);
        TCHECK(p.on_step([]() { return false; }, g, now + 19990) == 0, "not yet a hang");
        const unsigned ev = p.on_step([]() { return false; }, g, now + 20000);
        TCHECK((ev & pol_t::EV_HANG) && p.hang_kind == pol_t::E_PAUSE1 && p.n_errors == 1 && !p.slow_pending, "a hang counts");
        TCHECK(steps_to_rearm(p, g, now, false, 100, nullptr) == -1, "and the bridge stays paused while the job is owed");
        TCHECK(steps_to_rearm(p, g, now, true, 100, nullptr) >= 1 && p.active, "until the host answers");
    }
    // a host job that failed is never a slow job
    {
        pol_t p = make_pol(true);
        TCHECK(p.on_error(false, 100, 0) == pol_t::E_PAUSE1 && p.n_errors == 1 && !p.slow_pending, "a failed job counts");
    }
    // a storm: timeouts back to back (within the gap) count, so a host that cannot keep up still leaves the bridge
    {
        pol_t p = make_pol(true);
        p.slow_gap = 32;
        uint64_t g = 1000;
        int64_t now = 0;
        TCHECK(p.on_error(true, g, now) == pol_t::E_SLOW, "the first is a slow job");
        steps_to_rearm(p, g, now, true, 10, nullptr);
        g += 10;
        TCHECK(p.on_error(true, g, now) == pol_t::E_PAUSE1, "10 graphs later: counted");
        steps_to_rearm(p, g, now, true, 10, nullptr);
        g += 10;
        TCHECK(p.on_error(true, g, now) == pol_t::E_PAUSE16, "again: 16 steps");
        steps_to_rearm(p, g, now, true, 30, nullptr);
        g += 10;
        TCHECK(p.on_error(true, g, now) == pol_t::E_OFF_REARM && p.disabled, "and the third turns it off");
        // far apart again: slow jobs, not counted
        steps_to_rearm(p, g, now, true, 1000, nullptr);
        g += 500;
        TCHECK(p.on_error(true, g, now) == pol_t::E_SLOW, "500 graphs later: a slow job again");
    }
    // a steady stream: a host squeezed for minutes times out every `every` bridged graphs and answers each late job. Past
    // slow_max slow jobs within slow_win graphs the timeouts count, so the v0.33.0 turn-off is still reached
    auto stream = [](int every, int n, int * first_off, int * n_counted) {
        pol_t p = make_pol(true);
        uint64_t g = 10000;
        int64_t now = 0;
        *first_off = -1;
        *n_counted = 0;
        for (int i = 0; i < n; ++i) {
            const auto k = p.on_error(true, g, now);
            *n_counted += k != pol_t::E_SLOW ? 1 : 0;
            if (*first_off < 0 && (k == pol_t::E_OFF_REARM || k == pol_t::E_OFF)) {
                *first_off = i;
            }
            steps_to_rearm(p, g, now, true, 100000, nullptr); // answered; a cool-down, if any, runs out
            g += (uint64_t) every;
        }
        return p;
    };
    {
        int first_off = 0, n_counted = 0;
        const pol_t p = stream(40, 200, &first_off, &n_counted);
        TCHECK(first_off >= 0 && first_off <= 5, "a timeout every 40 graphs: the bridge turns off at timeout %d (3 forgiven, "
               "then 3 errors)", first_off);
        TCHECK(p.n_slow <= 3 + 3*200*40/1024 + 3, "a timeout every 40 graphs: %llu slow jobs forgiven of 200",
               (unsigned long long) p.n_slow);
    }
    {
        int first_off = 0, n_counted = 0;
        const pol_t p = stream(300, 60, &first_off, &n_counted);
        TCHECK(n_counted >= 12 && p.n_slow <= 3*(60*300/1024 + 1), "a timeout every 300 graphs (4 within 1024): %d of 60 "
               "count, %llu slow jobs (at most 3 per 1024 graphs)", n_counted, (unsigned long long) p.n_slow);
    }
    {
        int first_off = 0, n_counted = 0;
        const pol_t p = stream(400, 60, &first_off, &n_counted);
        TCHECK(first_off < 0 && n_counted == 0 && p.n_errors == 0 && p.n_slow == 60,
               "a timeout every 400 graphs (3 within 1024): every one a slow job (%d counted)", n_counted);
    }
    // the window moves on: after a quiet stretch the next slow jobs are forgiven again
    {
        pol_t p = make_pol(true);
        uint64_t g = 1000;
        int64_t now = 0;
        for (int i = 0; i < 3; ++i, g += 100) {
            TCHECK(p.on_error(true, g, now) == pol_t::E_SLOW && !p.slow_capped, "slow job %d of the window", i + 1);
            steps_to_rearm(p, g, now, true, 100, nullptr);
        }
        TCHECK(p.on_error(true, g, now) == pol_t::E_PAUSE1 && p.slow_capped && p.n_errors == 1,
               "the fourth within 1024 graphs counts (a first error)");
        steps_to_rearm(p, g, now, true, 100, nullptr);
        g += 1100;
        TCHECK(p.on_error(true, g, now) == pol_t::E_SLOW && !p.slow_capped, "1100 graphs later: a slow job again");
        steps_to_rearm(p, g, now, true, 100, nullptr);
        pol_t q = make_pol(true);
        q.slow_max = 1;
        g = 1000;
        TCHECK(q.on_error(true, g, now) == pol_t::E_SLOW, "slow_max 1: the first is a slow job");
        steps_to_rearm(q, g, now, true, 100, nullptr);
        g += 500;
        TCHECK(q.on_error(true, g, now) == pol_t::E_PAUSE1 && q.slow_capped, "slow_max 1: the next within 1024 graphs counts");
    }
    // the switch off: the window plays no part (v0.33.0)
    {
        pol_t p = make_pol(false);
        TCHECK(p.on_error(true, 100, 0) == pol_t::E_PAUSE1 && !p.slow_capped && p.slow_ring_n == 0, "switch off: v0.33.0");
    }
}

static void test_floor() {
    fn_r41::floor_gate off;
    TCHECK(off.sample(0, 0) == 0 && off.sample(0, 5000000) == 0, "floor 0: off");

    const size_t MiB = (size_t) 1 << 20;
    fn_r41::floor_gate f;
    f.floor_b = 1536*MiB;
    int64_t t = 10000000;
    TCHECK(f.sample(4096*MiB, t) == 0, "plenty of RAM");
    TCHECK(f.sample(1000*MiB, t += 100000) == 0, "one sample under the floor: wait");
    TCHECK(f.sample(1600*MiB, t += 100000) == 0 && f.below == 0, "back over the floor before the hold: nothing");
    TCHECK(f.sample(1000*MiB, t += 100000) == 0, "under again: wait");
    TCHECK(f.sample(1000*MiB, t += 100000) == 256*MiB, "two samples: 256 MiB (the chunk) of the 1048 MiB short");
    TCHECK(f.sample(1000*MiB, t += 100000) == 0, "within the gap: nothing");
    TCHECK(f.sample(1900*MiB, t += 1000000) == 148*MiB, "acting goes on to floor + margin: 148 MiB");
    TCHECK(f.sample(2048*MiB, t += 1000000) == 0 && !f.acting, "at floor + margin: stop");
    TCHECK(f.sample(1900*MiB, t += 1000000) == 0, "between floor and margin without an action: nothing");
    TCHECK(f.sample(1400*MiB, t += 100000) == 0 && f.sample(1400*MiB, t += 100000) == 256*MiB,
           "under the floor again: the hold, then the chunk");
    TCHECK(f.sample(1400*MiB, t += 1000000) == 256*MiB, "a second later the next chunk");

    // the cold experts the floor moves to the SSD set: a squeeze of 10 minutes without an idle plan (a long busy stretch
    // under the floor) asks 256 MiB every second; the queued drops free nothing, each give-back marks what it is asked plus part
    // of one expert (3 MiB)
    fn_r41::floor_gate g;
    g.floor_b = 1536*MiB;
    TCHECK(g.mark_cap_b == 1024*MiB, "the cap: 1024 MiB between two idle plans");
    size_t marked = 0;
    int    n_act  = 0;
    for (int64_t s = 0; s < 600*10; ++s) {
        const size_t want = g.sample(900*MiB, 50000000 + s*100000);
        if (want == 0) {
            continue;
        }
        n_act++;
        const size_t ask = g.give_back_ask(want, 0);
        const size_t got = ask > 0 ? ask + 3*MiB : 0;
        g.on_marked(got);
        marked += got;
    }
    TCHECK(n_act >= 590 && marked <= 1024*MiB + 4*3*MiB, "%d actions mark %.0f MiB of cold experts (at most the cap)", n_act,
           marked/(double) MiB);
    TCHECK(g.give_back_ask(256*MiB, 0) == 0, "past the cap: queued drops only");
    g.on_idle_plan();
    TCHECK(g.give_back_ask(256*MiB, 0) == 256*MiB, "after an idle plan: cold experts again");
    TCHECK(g.give_back_ask(256*MiB, 100*MiB) == 156*MiB && g.give_back_ask(256*MiB, 300*MiB) == 0,
           "what the queued drops freed is not asked of the cold experts");
    g.on_marked(1000*MiB);
    TCHECK(g.give_back_ask(256*MiB, 0) == 24*MiB, "the last 24 MiB of the cap");
    fn_r41::floor_gate z;
    z.mark_cap_b = 0;
    TCHECK(z.give_back_ask(256*MiB, 0) == 0, "cap 0: queued drops only");
}

static void test_prio() {
    fn_r41::prio a;
    a.io   = 0;
    a.page = 1;
    const fn_r41::prio ta = fn_r41::prio_target(a);
    TCHECK(ta.io == 2 && ta.page == 5, "very low I/O and page priority 1 are raised to normal");
    fn_r41::prio b;
    b.io   = 3;
    b.page = 5;
    const fn_r41::prio tb = fn_r41::prio_target(b);
    TCHECK(tb.io == 3 && tb.page == 5, "never lowered");
#ifdef _WIN32
    const fn_r41::prio p0 = fn_r41::prio_read(GetCurrentProcess());
    const fn_r41::prio t0 = fn_r41::prio_read(GetCurrentThread(), true);
    TCHECK(p0.ok && t0.ok, "the process's and the thread's values read (process %u / %u, thread %u / %u)", p0.io, p0.page,
           t0.io, t0.page);
    // a launcher's low values: I/O low (1), page priority 2, and this thread's very low (0) / 1
    fn_r41::prio lo;
    lo.io   = 1;
    lo.page = 2;
    TCHECK(fn_r41::prio_write(GetCurrentProcess(), p0, lo), "the process's values lowered");
    fn_r41::prio tlo;
    tlo.io   = 0;
    tlo.page = 1;
    TCHECK(fn_r41::prio_write(GetCurrentThread(), t0, tlo, true), "the thread's values lowered");
    const fn_r41::prio p1 = fn_r41::prio_read(GetCurrentProcess());
    const fn_r41::prio t1 = fn_r41::prio_read(GetCurrentThread(), true);
    TCHECK(p1.ok && p1.io == 1 && p1.page == 2 && t1.ok && t1.io == 0 && t1.page == 1, "lowered as asked (process %u / %u, "
           "thread %u / %u)", p1.io, p1.page, t1.io, t1.page);
    const auto r = fn_r41::prio_fix();
    const fn_r41::prio t2 = fn_r41::prio_read(GetCurrentThread(), true);
    TCHECK(r.raised && r.ok && r.before.io == 1 && r.before.page == 2, "the check saw the low values");
    TCHECK(r.after.io == 2 && r.after.page == 5, "process raised to normal (%u / %u)", r.after.io, r.after.page);
    TCHECK(t2.io == 2 && t2.page == 5, "thread raised to normal (%u / %u)", t2.io, t2.page);
    const auto r2 = fn_r41::prio_fix();
    TCHECK(!r2.raised && r2.ok, "normal values: nothing to do");
    fn_r41::prio_write(GetCurrentProcess(), fn_r41::prio_read(GetCurrentProcess()), p0); // as the test found them
#endif
}

int main() {
    test_v0330_rules();
    test_slowjob();
    test_floor();
    test_prio();
    printf("test-fn-r41: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
