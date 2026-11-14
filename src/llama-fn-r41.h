#pragma once

// [TAG_FN_R41] round 41 (stalls when RAM is short): the pure rules of three switches, header only so the unit test
// (tests/test-fn-r41.cpp) runs the same code as the engine. Nothing here changes a value the model computes.
//
//   LLAMA_FN_R41_SLOWJOB=1      [TAG_FN_R41_SLOWJOB] (the profile: 0 = v0.33.0) a MoE bridge wait timeout is
//                               first taken for a slow host job (hard page faults on expert pages Windows trimmed): the
//                               ubatch is computed again without the bridge as before, and the bridge re-arms as soon as
//                               the host has answered the late job, without counting an error. Only a job the host does
//                               not answer within LLAMA_FN_R41_SLOWJOB_MS (default 20000) counts, so does a timeout
//                               within LLAMA_FN_R41_SLOWJOB_GAP (default 32) bridged graphs of the last one, and so does
//                               one that comes after LLAMA_FN_R41_SLOWJOB_MAX (default 3, at most 8) slow jobs within the
//                               last 1024 bridged graphs (a host that keeps stalling, however far apart): then the
//                               v0.33.0 rules apply (16 paused steps at the second error within 1024 graphs, the
//                               turn-off at the third)
//   LLAMA_FN_R41_FLOOR_MIB=1536 [TAG_FN_R41_FLOOR] (the profile: 0 = v0.33.0) the residency service keeps this much
//                               RAM available itself: below it for 200 ms, also while answers and prompts run, it gives
//                               back VRAM-held host copies first, then the least-routed resident cold experts (they move
//                               to the SSD tier and stay readable), at most 256 MiB per second, until 512 MiB more than
//                               the floor is available. They come back as before: by use, the idle plan and the probe.
//                               Between two idle plans the floor moves at most LLAMA_FN_R41_FLOOR_MARK_MIB (default
//                               1024) of cold experts to the SSD tier; past that it gives back queued drops only
//   LLAMA_FN_R41_PRIO=1         [TAG_FN_R41_PRIO] (the profile: 0 = v0.33.0) at load the process's I/O priority
//                               (NtQueryInformationProcess class 33) and page priority (class 39) are read and logged;
//                               a value below normal (inherited from the launcher) is raised to normal, never above, and
//                               so are the loading thread's own (thread classes 22 and 24)

#include <algorithm>
#include <cstddef>
#include <cstdint>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#endif

namespace fn_r41 {

// ---------------------------------------------------------------------------------------------------------------------
// [TAG_FN_R41_SLOWJOB] the MoE bridge's error policy. The v0.33.0 fields and rules (TAG_FN_R1_BRIDGE_RB,
// TAG_FN_L7_PAUSE1, TAG_FN_L3_HOST_REARM) moved here unchanged; slow_on adds the slow-job verdict in front of them.
struct bridge_policy {
    // v0.33.0: owner thread only
    bool     active      = true;
    bool     disabled    = false;
    int      n_errors    = 0;
    int      pause_left  = 0;
    uint64_t err_graph   = 0; // n_graphs at the last counted error
    int      rearm_steps = 0; // [TAG_FN_L3_HOST_REARM] 0: off for good after 3 errors
    int64_t  rearm_ms    = 0;
    int      n_off       = 0;
    int64_t  off_steps   = 0;
    int64_t  off_until   = 0;
    uint64_t off_graph   = 0;
    int64_t  off_ms      = 0; // the length of the last cool-down (for the log)

    // [TAG_FN_R41_SLOWJOB]
    bool     slow_on      = false;
    int64_t  slow_ceil_ms = 20000; // a late job not answered by then is a hang: it counts
    uint64_t slow_gap     = 32;    // a timeout within this many bridged graphs of the last one counts
    bool     slow_pending = false; // a timeout waits for the host to answer its late job
    bool     slow_seen    = false;
    uint64_t slow_graph   = 0;     // n_graphs at the last timeout
    int64_t  slow_t0      = 0;     // ms at the pending one
    uint64_t n_slow       = 0;     // slow jobs so far (not counted as errors)
    int64_t  slow_ms      = 0;     // how long the last one took to be answered
    // at most slow_max slow jobs within slow_win bridged graphs: a timeout with slow_max slow jobs in the window before it
    // counts, so a host that keeps stalling still meets v0.33.0's turn-off (3 errors within 1024 graphs)
    static constexpr int SLOW_RING = 8;
    int      slow_max     = 3;
    uint64_t slow_win     = 1024;
    uint64_t slow_ring[SLOW_RING] = {}; // n_graphs of the last slow jobs (slow_ring_n of them, newest at slow_ring_i - 1)
    int      slow_ring_n  = 0;
    int      slow_ring_i  = 0;
    bool     slow_capped  = false;      // the last timeout counted because the window was full (for the log)

    // the slow jobs within slow_win graphs before n_graphs
    int slow_in_window(uint64_t n_graphs) const {
        int n = 0;
        for (int i = 0; i < slow_ring_n; ++i) {
            n += n_graphs - slow_ring[i] <= slow_win ? 1 : 0;
        }
        return n;
    }

    enum error_kind {
        E_PAUSE1,     // only the retried ubatch runs without the bridge
        E_PAUSE16,    // the bridge pauses for 16 steps
        E_OFF_REARM,  // 3 errors: off for off_steps or off_ms, then re-armed
        E_OFF,        // 3 errors: off for the context
        E_SLOW,       // [TAG_FN_R41_SLOWJOB] not counted: waits for the host to answer the late job
    };

    // v0.33.0's counting of an error of the bridged graph that just ended (n_graphs counts it)
    error_kind count_error(uint64_t n_graphs, int64_t now_ms) {
        // [TAG_FN_R1_BRIDGE_RB] errors far apart are transient stalls whose ubatch was computed again: only 3 within 1024
        // bridged graphs turn the bridge off (a host side that keeps failing)
        if (n_graphs - err_graph > 1024) {
            n_errors = 0;
        }
        err_graph = n_graphs;
        n_errors++;
        active = false;
        // [TAG_FN_L7_PAUSE1] a first error pauses only the retried ubatch, a second within 1024 graphs 16 steps
        pause_left = n_errors <= 1 ? 1 : 16;
        if (n_errors >= 3) {
            disabled = true;
            if (rearm_steps > 0) { // [TAG_FN_L3_HOST_REARM] back-off; a long clean run since the last turn-off starts over
                if (n_off > 0 && n_graphs - off_graph > 65536) {
                    n_off = 0;
                }
                const int k = std::min(n_off, 6);
                n_off++;
                off_graph = n_graphs;
                off_steps = (int64_t) rearm_steps << k;
                off_ms    = rearm_ms << k;
                off_until = now_ms + off_ms;
                return E_OFF_REARM;
            }
            return E_OFF;
        }
        return pause_left > 1 ? E_PAUSE16 : E_PAUSE1;
    }

    // an error of the bridged graph that just ended; timeout: a wait gave up (not a failed host job)
    error_kind on_error(bool timeout, uint64_t n_graphs, int64_t now_ms) {
        slow_capped = false;
        if (slow_on && timeout) {
            const bool back_to_back = slow_seen && n_graphs - slow_graph < slow_gap;
            const int  cap          = std::max(1, std::min(slow_max, (int) SLOW_RING));
            slow_seen   = true;
            slow_graph  = n_graphs;
            slow_capped = !slow_pending && !back_to_back && slow_in_window(n_graphs) >= cap;
            if (!slow_pending && !back_to_back && !slow_capped) {
                active       = false;
                pause_left   = 1; // the retried ubatch, as after a first error
                slow_pending = true;
                slow_t0      = now_ms;
                slow_ring[slow_ring_i] = n_graphs;
                slow_ring_i = (slow_ring_i + 1) % SLOW_RING;
                slow_ring_n = std::min(slow_ring_n + 1, (int) SLOW_RING);
                return E_SLOW;
            }
        }
        return count_error(n_graphs, now_ms);
    }

    // what step() did (bits)
    enum step_event : unsigned {
        EV_COOLED  = 1, // [TAG_FN_L3_HOST_REARM] the cool-down ended
        EV_REARMED = 2, // the device error was cleared: graphs may take the bridge again
        EV_SLOW_OK = 4, // [TAG_FN_R41_SLOWJOB] the host answered the late job in time (slow_ms): no error counted
        EV_HANG    = 8, // [TAG_FN_R41_SLOWJOB] it did not within slow_ceil_ms: counted (hang_kind)
    };
    error_kind hang_kind = E_PAUSE1;

    // before the graph parameters of a step: reset() clears the device error and returns false while the host still
    // owes a posted job (try again next step)
    template <typename R>
    unsigned on_step(R && reset, uint64_t n_graphs, int64_t now_ms) {
        if (active) {
            return 0;
        }
        if (slow_pending) {
            if (pause_left > 0) {
                pause_left--;
                return 0;
            }
            if (reset()) {
                slow_pending = false;
                active       = true;
                n_slow++;
                slow_ms = now_ms - slow_t0;
                return EV_SLOW_OK | EV_REARMED;
            }
            if (now_ms - slow_t0 < slow_ceil_ms) {
                return 0; // the host is still on it
            }
            slow_pending = false;
            hang_kind    = count_error(n_graphs, now_ms);
            return EV_HANG;
        }
        unsigned ev = 0;
        if (disabled) {
            // [TAG_FN_L3_HOST_REARM] the cool-down ends after its steps or its time, then the re-arm below runs as after a pause
            if (rearm_steps <= 0 || (--off_steps > 0 && now_ms < off_until)) {
                return 0;
            }
            disabled   = false;
            n_errors   = 0;
            pause_left = 0;
            ev |= EV_COOLED;
        }
        if (pause_left > 0) {
            pause_left--;
            return ev;
        }
        if (!reset()) {
            return ev; // the executor still owes the stalled job: try again next step
        }
        active = true;
        return ev | EV_REARMED;
    }
};

// ---------------------------------------------------------------------------------------------------------------------
// [TAG_FN_R41_FLOOR] the free-RAM floor of the residency service: one sample per tick (100 ms). Below floor_b for `hold`
// samples in a row it acts, at most once per gap_us and chunk_b at a time, until avail is back at floor_b + margin_b.
struct floor_gate {
    size_t  floor_b  = 0; // 0: off
    size_t  margin_b = (size_t) 512 << 20;
    size_t  chunk_b  = (size_t) 256 << 20;
    int     hold     = 2;
    int64_t gap_us   = 1000000;

    int     below  = 0;
    bool    acting = false;
    int64_t t_last = 0;
    bool    acted  = false;

    // the bytes to give back now (0: nothing)
    size_t sample(size_t avail, int64_t now_us) {
        if (floor_b == 0) {
            return 0;
        }
        if (avail >= floor_b + margin_b) {
            below  = 0;
            acting = false;
            return 0;
        }
        if (avail < floor_b) {
            below++;
        } else if (!acting) {
            below = 0; // between the floor and its margin: only an action that has begun goes on
            return 0;
        }
        if (!acting && below < hold) {
            return 0;
        }
        acting = true;
        if (acted && now_us - t_last < gap_us) {
            return 0;
        }
        acted  = true;
        t_last = now_us;
        return std::min(floor_b + margin_b - avail, chunk_b);
    }

    // the cold experts an action may move to the SSD set: what the queued drops (b_queued) left of `want`, within what the
    // floor has not yet marked since the last idle plan (mark_cap_b; 0: queued drops only). Only the idle plan, which sizes
    // the SSD set against the room, and the probe bring marks back, and in a long busy stretch neither runs: the cap keeps
    // a squeeze from moving more and more of the streamed set to direct I/O (an action may pass it by one expert)
    size_t  mark_cap_b = (size_t) 1024 << 20;
    size_t  marked_b   = 0;

    size_t give_back_ask(size_t want, size_t b_queued) const {
        const size_t rest = want > b_queued ? want - b_queued : 0;
        return std::min(rest, mark_cap_b > marked_b ? mark_cap_b - marked_b : 0);
    }
    void on_marked(size_t b) {
        marked_b += b;
    }
    void on_idle_plan() {
        marked_b = 0;
    }
};

// ---------------------------------------------------------------------------------------------------------------------
// [TAG_FN_R41_PRIO] the process's I/O and page priority. Windows: IO_PRIORITY_HINT (0 very low, 1 low, 2 normal, 3 high,
// 4 critical) and page priority 0..7 (MEMORY_PRIORITY_NORMAL = 5). A launcher in background mode, or one that lowered its
// own values, can hand them to its children; the engine's pages then leave RAM first and its file reads queue last.
constexpr uint32_t IO_NORMAL   = 2;
constexpr uint32_t PAGE_NORMAL = 5;

struct prio {
    bool     ok   = false; // read
    uint32_t io   = IO_NORMAL;
    uint32_t page = PAGE_NORMAL;
};

// what a raise sets: normal where a value is below it, the rest unchanged (never above normal, never lower)
inline prio prio_target(const prio & cur) {
    prio t = cur;
    t.io   = std::max(cur.io, IO_NORMAL);
    t.page = std::max(cur.page, PAGE_NORMAL);
    return t;
}

#ifdef _WIN32
// ntdll's information classes (through GetProcAddress: no SDK version macro, no import library)
using nt_info_t = LONG (WINAPI *)(HANDLE, ULONG, PVOID, ULONG, PULONG);
using nt_set_t  = LONG (WINAPI *)(HANDLE, ULONG, PVOID, ULONG);

constexpr ULONG PROC_IO_PRIORITY     = 33; // ProcessIoPriority (ULONG IO_PRIORITY_HINT)
constexpr ULONG PROC_PAGE_PRIORITY   = 39; // ProcessPagePriority (PAGE_PRIORITY_INFORMATION: one ULONG)
constexpr ULONG THREAD_IO_PRIORITY   = 22; // ThreadIoPriority
constexpr ULONG THREAD_PAGE_PRIORITY = 24; // ThreadPagePriority

inline FARPROC nt_proc(const char * name) {
    const HMODULE nt = GetModuleHandleW(L"ntdll.dll");
    return nt ? GetProcAddress(nt, name) : nullptr;
}

// the two values of a process (thread = false) or of a thread (thread = true)
inline prio prio_read(HANDLE h, bool thread = false) {
    prio p;
    const nt_info_t q = (nt_info_t) (void *) nt_proc(thread ? "NtQueryInformationThread" : "NtQueryInformationProcess");
    if (q == nullptr) {
        return p;
    }
    ULONG io = 0, page = 0, n = 0;
    const bool ok_io   = q(h, thread ? THREAD_IO_PRIORITY : PROC_IO_PRIORITY, &io, sizeof(io), &n) >= 0;
    const bool ok_page = q(h, thread ? THREAD_PAGE_PRIORITY : PROC_PAGE_PRIORITY, &page, sizeof(page), &n) >= 0;
    p.ok   = ok_io && ok_page;
    p.io   = ok_io ? (uint32_t) io : IO_NORMAL;
    p.page = ok_page ? (uint32_t) page : PAGE_NORMAL;
    return p;
}

// sets the values of want that differ from cur (no privilege is needed up to normal); false when a call failed
inline bool prio_write(HANDLE h, const prio & cur, const prio & want, bool thread = false) {
    const nt_set_t s = (nt_set_t) (void *) nt_proc(thread ? "NtSetInformationThread" : "NtSetInformationProcess");
    if (s == nullptr) {
        return false;
    }
    bool ok = true;
    if (want.io != cur.io) {
        ULONG v = want.io;
        ok = s(h, thread ? THREAD_IO_PRIORITY : PROC_IO_PRIORITY, &v, sizeof(v)) >= 0 && ok;
    }
    if (want.page != cur.page) {
        ULONG v = want.page;
        ok = s(h, thread ? THREAD_PAGE_PRIORITY : PROC_PAGE_PRIORITY, &v, sizeof(v)) >= 0 && ok;
    }
    return ok;
}

// the check at load: the process's values, raised to normal where lower, and the calling thread's (a thread takes its
// process's values when it is made, so one made before the raise keeps the old ones). *before / *after: the process's
struct prio_fix_result {
    prio before;
    prio after;
    prio thread_before;
    bool raised = false; // a value was below normal
    bool ok     = true;  // every raise held
};

inline prio_fix_result prio_fix() {
    prio_fix_result r;
    r.before = prio_read(GetCurrentProcess());
    r.after  = r.before;
    if (!r.before.ok) {
        r.ok = false;
        return r;
    }
    const prio want = prio_target(r.before);
    if (want.io != r.before.io || want.page != r.before.page) {
        r.raised = true;
        r.ok     = prio_write(GetCurrentProcess(), r.before, want) && r.ok;
        r.after  = prio_read(GetCurrentProcess());
    }
    r.thread_before = prio_read(GetCurrentThread(), true);
    if (r.thread_before.ok) {
        const prio tw = prio_target(r.thread_before);
        if (tw.io != r.thread_before.io || tw.page != r.thread_before.page) {
            r.raised = true;
            r.ok     = prio_write(GetCurrentThread(), r.thread_before, tw, true) && r.ok;
        }
    }
    return r;
}
#endif

} // namespace fn_r41
