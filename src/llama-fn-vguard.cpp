// [TAG_R40_VGUARD] the WDDM paging guard and its zero-commit glue: see llama-fn-vguard.h
#include "llama-fn-vguard.h"

#include "llama-impl.h"
#include "llama-moecache.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cwchar>

#if defined(_WIN32)
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#    include <pdh.h>
#    include <pdhmsg.h>
#endif

namespace {

constexpr double VG_MIB = 1048576.0;

[[maybe_unused]] double vg_env(const char * name, double def) {
    const char * e = getenv(name);
    return (e && *e) ? atof(e) : def;
}

struct vg_knobs {
    double htgt = 2944, ftgt = 3200;   // MiB: the switch / encode levels and the regrow floor (a switch that commits)
    double zhtgt = 2304, zftgt = 2560; // MiB: the same for a zero-commit switch (it commits nothing)
    double dem = 64, drop = 256, max = 2048, react_f = 3800; // MiB
    int    poll_ms = 20;
    double calm_s = 10, quiet_s = 2, regrow_s = 2;
    int    press = 1, react = 0, force = 0;
};

// one poll's numbers (MiB)
struct vg_sample {
    int64_t t      = 0;
    double  budget = 0, usage = 0, limit = 0, seg_res = 0;
    double  ded    = -1.0; // our PDH dedicated usage (-1: not read)
    double  H = 0, F = 0, raw = 0;
};

#if defined(_WIN32)
// D3DKMTQueryStatistics on a raw 808-byte D3DKMT_QUERYSTATISTICS (SDK 10.0.26100 layout, r40 understand/detect_notes.txt):
// Type at 0, AdapterLuid at 4, hProcess at 16, QueryResult at 24, the input union at 800. Any layout surprise shows as a
// failed or implausible answer at arm time, and the guard stays off
using vg_kmt_qs_fn = LONG (APIENTRY *)(void *);

enum : uint32_t {
    VG_QS_ADAPTER               = 0,
    VG_QS_SEGMENT               = 3,
    VG_QS_PROCESS_SEGMENT_GROUP = 9,
};

struct vg_kmt {
    vg_kmt_qs_fn qs   = nullptr;
    HANDLE       proc = nullptr;
    uint64_t     luid = 0;
    uint32_t     seg  = 0;
    alignas(8) unsigned char buf[808];

    LONG query(uint32_t type, HANDLE h, uint32_t in) {
        memset(buf, 0, sizeof(buf));
        memcpy(buf + 0, &type, 4);
        memcpy(buf + 4, &luid, 8);
        memcpy(buf + 16, &h, sizeof(HANDLE));
        memcpy(buf + 800, &in, 4);
        return qs(buf);
    }
    uint64_t u64(size_t off) const {
        uint64_t v;
        memcpy(&v, buf + off, 8);
        return v;
    }
    uint32_t u32(size_t off) const {
        uint32_t v;
        memcpy(&v, buf + off, 4);
        return v;
    }
    bool poll(vg_sample & s) {
        if (query(VG_QS_PROCESS_SEGMENT_GROUP, proc, 0 /* LOCAL */) != 0) {
            return false;
        }
        s.budget = u64(24)/VG_MIB;
        s.usage  = u64(40)/VG_MIB;
        if (query(VG_QS_SEGMENT, nullptr, seg) != 0) {
            return false;
        }
        s.limit   = u64(24)/VG_MIB;
        s.seg_res = u64(40)/VG_MIB;
        return true;
    }
};

// our process's "GPU Process Memory(pid_<pid>_luid_0x<high>_0x<low>_phys_0)\Dedicated Usage" (pdh.dll loaded at arm; one
// counter, ~1.9 us per collect + read on an idle GPU, r40 understand latency table)
struct vg_pdh {
    using open_fn    = PDH_STATUS (WINAPI *)(LPCWSTR, DWORD_PTR, PDH_HQUERY *);
    using add_fn     = PDH_STATUS (WINAPI *)(PDH_HQUERY, LPCWSTR, DWORD_PTR, PDH_HCOUNTER *);
    using collect_fn = PDH_STATUS (WINAPI *)(PDH_HQUERY);
    using fmt_fn     = PDH_STATUS (WINAPI *)(PDH_HCOUNTER, DWORD, LPDWORD, PPDH_FMT_COUNTERVALUE);
    using close_fn   = PDH_STATUS (WINAPI *)(PDH_HQUERY);
    collect_fn   collect = nullptr;
    fmt_fn       fmt     = nullptr;
    close_fn     close   = nullptr;
    PDH_HQUERY   q       = nullptr;
    PDH_HCOUNTER c       = nullptr;
    wchar_t      path[160] = {};

    // false: why says what failed
    bool open(uint64_t luid, const char ** why) {
        HMODULE dll = LoadLibraryA("pdh.dll");
        auto * o  = dll ? (open_fn) (void *) GetProcAddress(dll, "PdhOpenQueryW") : nullptr;
        auto * a  = dll ? (add_fn) (void *) GetProcAddress(dll, "PdhAddEnglishCounterW") : nullptr;
        collect   = dll ? (collect_fn) (void *) GetProcAddress(dll, "PdhCollectQueryData") : nullptr;
        fmt       = dll ? (fmt_fn) (void *) GetProcAddress(dll, "PdhGetFormattedCounterValue") : nullptr;
        close     = dll ? (close_fn) (void *) GetProcAddress(dll, "PdhCloseQuery") : nullptr;
        if (!o || !a || !collect || !fmt || !close) {
            *why = "pdh.dll or its functions not found";
            return false;
        }
        swprintf(path, sizeof(path)/sizeof(path[0]), L"\\GPU Process Memory(pid_%lu_luid_0x%08X_0x%08X_phys_0)\\Dedicated Usage",
                (unsigned long) GetCurrentProcessId(), (unsigned) (luid >> 32), (unsigned) (luid & 0xffffffffu));
        if (o(nullptr, 0, &q) != ERROR_SUCCESS) {
            q    = nullptr;
            *why = "PdhOpenQuery failed";
            return false;
        }
        if (a(q, path, 0, &c) != ERROR_SUCCESS) {
            close(q);
            q    = nullptr;
            *why = "PdhAddEnglishCounter failed (no such GPU Process Memory instance)";
            return false;
        }
        return true;
    }
    bool read(double & mib) {
        if (!q || collect(q) != ERROR_SUCCESS) {
            return false;
        }
        PDH_FMT_COUNTERVALUE v = {};
        if (fmt(c, PDH_FMT_LARGE, nullptr, &v) != ERROR_SUCCESS ||
            (v.CStatus != PDH_CSTATUS_VALID_DATA && v.CStatus != PDH_CSTATUS_NEW_DATA)) {
            return false;
        }
        mib = (double) v.largeValue/VG_MIB;
        return true;
    }
};
#endif

// the zero-commit glue: ggml-cuda's procs on the hot set's device (stats order: ggml_cuda_zc_stat in ggml-cuda.cu)
enum {
    ZC_N_PARK, ZC_B_PARK, ZC_N_REUSE, ZC_B_REUSE, ZC_N_NEW, ZC_B_NEW, ZC_N_DROP, ZC_B_DROP, ZC_N_FAIL, ZC_N_ARENA, ZC_B_ARENA,
    ZC_N_POOL, ZC_B_POOL, ZC_B_STASH, ZC_US, ZC_NST,
};

struct vg_zc {
    bool               on     = false;
    bool               tried  = false;
    ggml_backend_dev_t dev    = nullptr;
    bool   (*enable)(ggml_backend_dev_t, bool)            = nullptr;
    int    (*scope)(ggml_backend_dev_t, int)              = nullptr;
    int    (*window)(ggml_backend_dev_t, int)             = nullptr;
    size_t (*drop)(ggml_backend_dev_t)                    = nullptr;
    int    (*stats)(ggml_backend_dev_t, uint64_t *, int)  = nullptr;
    uint64_t last[ZC_NST] = {};
    uint64_t n_full = 0, n_small = 0;
    double   ms_full = 0.0, ms_small = 0.0;
};

vg_zc g_zc;

struct vg_state {
    int          mode   = -1;
    bool         armed  = false;
    bool         logged_off = false;
    const void * owner  = nullptr;
    llama_fn_vg_hooks h;
    vg_knobs     k;
#if defined(_WIN32)
    vg_kmt       K;
    vg_pdh       P;
#endif
    // the last poll and the rules' memory
    vg_sample cur;
    bool      have     = false;
    int       n_fail   = 0;
    bool      det      = false;  // DEMOTED detection armed (PDH read, |D| <= 32 MiB at arm)
    double    d_off    = 0.0;    // the calm offset of usage - PDH dedicated
    double    D        = 0.0;
    int       dem_n    = 0;
    int64_t   t_dem0   = 0;      // the first poll of the current D >= DEM run
    bool      demoted  = false;
    double    solo     = 0.0;    // the solo budget: at arm, raised when the budget rises, never lowered by an action
    double    deficit  = 0.0;
    bool      pressure = false;
    int64_t   t_calm0  = 0;      // deficit < DROP and D <= 16 since (0: not now)
    int64_t   t_prompt = 0;      // the last batch with prompt rows, FULL switch or encode
    int64_t   t_regrow = 0;
    int64_t   t_react  = 0;
    int64_t   t_obs    = 0;      // observe mode: the last 1 Hz line
    // the share
    size_t    group    = 0;      // one KV group's bytes
    // statistics
    uint64_t  n_poll = 0, n_act = 0, n_sw = 0, n_dec = 0, n_gate = 0, n_defer = 0, n_trans = 0, n_pdh_fail = 0;
    double    poll_us_max = 0.0;
};

vg_state g_vg;

vg_state & vg() {
    if (g_vg.mode < 0) {
        const char * e = getenv("LLAMA_R40_VGUARD");
        const int    m = (e && *e) ? atoi(e) : 1;
        g_vg.mode = (m == 0 || m == 2) ? m : 1;
    }
    return g_vg;
}

double vg_ms(int64_t t0) {
    return (ggml_time_us() - t0)/1000.0;
}

void vg_off(const char * why) {
    vg_state & g = vg();
    if (!g.logged_off) {
        g.logged_off = true;
        LLAMA_LOG_INFO("%s: [TAG_R40_VGUARD] off: %s\n", __func__, why);
    }
}

[[maybe_unused]] void vg_disarm(const char * why) {
    vg_state & g = vg();
    if (g.armed) {
        g.armed = false;
        LLAMA_LOG_WARN("%s: [TAG_R40_VGUARD] off from now on: %s (the share of %.0f MiB stays until the KV cache takes it)\n",
                __func__, why, llama_moe_hot_vg_out()/VG_MIB);
    }
}

// the zero-commit switch's H / F levels (a switch that commits: HTGT / FTGT)
double vg_lvl_h() {
    const vg_state & g = vg();
    return g_zc.on ? g.k.zhtgt : g.k.htgt;
}

double vg_lvl_f() {
    const vg_state & g = vg();
    return g_zc.on ? g.k.zftgt : g.k.ftgt;
}

[[maybe_unused]] void vg_note(const char * what) {
    vg_state & g = vg();
    g.n_trans++;
    if (g.n_trans <= 32 || g.n_trans % 64 == 0) {
        const vg_sample & s = g.cur;
        LLAMA_LOG_WARN("%s: [TAG_R40_VGUARD] %s: H %.0f F %.0f D %.0f MiB (budget %.0f solo %.0f deficit %.0f, usage %.0f, PDH "
                "dedicated %.0f), share %.0f MiB\n", __func__, what, s.H, s.F, g.D, s.budget, g.solo, g.deficit, s.usage, s.ded,
                llama_moe_hot_vg_out()/VG_MIB);
    }
}

// one poll (rate-limited to POLL_MS unless forced); false: no fresh sample
bool vg_poll(bool force) {
    vg_state & g = vg();
#if defined(_WIN32)
    const int64_t t = ggml_time_us();
    if (!force && g.have && t - g.cur.t < (int64_t) g.k.poll_ms*1000) {
        return false;
    }
    vg_sample s;
    if (!g.K.poll(s)) {
        if (++g.n_fail >= 16) {
            vg_disarm("16 queries in a row failed");
        }
        return false;
    }
    g.n_fail = 0;
    if (g.det) {
        double ded = 0.0;
        if (g.P.read(ded)) {
            s.ded = ded;
        } else {
            g.n_pdh_fail++;
        }
    }
    g.poll_us_max = std::max(g.poll_us_max, (double) (ggml_time_us() - t));
    g.n_poll++;
    s.t = t;
    s.H = s.budget - s.usage;
    s.F = s.limit - s.seg_res;
    // PRESSURE: the budget below the solo budget (another process's commitment lowers it in the same second, r40 replay)
    if (s.budget > g.solo) {
        g.solo = s.budget;
    }
    g.deficit = g.solo - s.budget;
    const bool was_press = g.pressure;
    g.pressure = g.deficit >= g.k.drop;
    // DEMOTED: our usage above our PDH dedicated usage (a failed PDH read keeps the last D)
    const bool was_dem = g.demoted;
    if (s.ded >= 0.0) {
        s.raw = s.usage - s.ded;
        g.D   = s.raw - g.d_off;
        if (g.D >= g.k.dem) {
            if (g.dem_n++ == 0) {
                g.t_dem0 = t;
            }
        } else {
            g.dem_n = 0;
        }
        g.demoted = g.dem_n >= 2 && t - g.t_dem0 >= 20000;
        if (g.D < 0.0) {
            g.d_off = s.raw; // a smaller calm offset at once
        } else if (g.D <= 16.0) {
            g.d_off += 0.02*(s.raw - g.d_off); // a calm offset that creeps up is not a demotion
        }
    }
    if (!g.pressure && g.D <= 16.0) {
        if (g.t_calm0 == 0) {
            g.t_calm0 = t;
        }
    } else {
        g.t_calm0 = 0;
    }
    g.cur  = s;
    g.have = true;
    if (g.demoted != was_dem) {
        vg_note(g.demoted ? "DEMOTED begins" : "DEMOTED ends");
    }
    if (g.pressure != was_press) {
        vg_note(g.pressure ? "PRESSURE begins" : "PRESSURE ends");
    }
    if (g.mode == 2 && t - g.t_obs >= 1000000) {
        // observe (T0): one line per second, D next to the harness's PDH d
        g.t_obs = t;
        LLAMA_LOG_WARN("%s: [TAG_R40_VGUARD] observe: H %.0f F %.0f D %.1f (usage %.1f - PDH dedicated %.1f - offset %.1f) MiB, budget "
                "%.0f solo %.0f deficit %.0f%s%s, share %.0f MiB, stash %.0f MiB\n", __func__, s.H, s.F, g.D, s.usage, s.ded, g.d_off,
                s.budget, g.solo, g.deficit, g.demoted ? " DEMOTED" : "", g.pressure ? " PRESSURE" : "",
                llama_moe_hot_vg_out()/VG_MIB, llama_fn_zc_stash_bytes()/VG_MIB);
    }
    return true;
#else
    GGML_UNUSED(force);
    GGML_UNUSED(g);
    return false;
#endif
}

size_t vg_max_b() {
    return (size_t) (vg().k.max*VG_MIB);
}

size_t vg_groups_for(double mib) {
    const vg_state & g = vg();
    if (mib <= 0.0 || g.group == 0) {
        return 0;
    }
    return (size_t) std::ceil(mib*VG_MIB/(double) g.group);
}

// the share to `target` bytes (whole groups, at most MAX); sync: the owner's graphs first (r35_hook_sync). A growing share
// frees the zero-commit stash first (handles nothing maps). One log line
size_t vg_hold(size_t target, bool sync, const char * why) {
    vg_state & g = vg();
    target = std::min(target, std::max(vg_max_b(), llama_moe_hot_vg_out()));
    const size_t before = llama_moe_hot_vg_out();
    if (target == before) {
        return before;
    }
    const vg_sample s0 = g.cur;
    const double    d0 = g.D;
    const double    f0 = g.deficit;
    const int64_t   t0 = ggml_time_us();
    size_t dropped = 0;
    if (target > before && g_zc.on && g_zc.drop) {
        dropped = g_zc.drop(g_zc.dev);
    }
    if (sync && g.h.sync) {
        g.h.sync(g.h.user);
    }
    const double  ms_sync = vg_ms(t0);
    const size_t  after   = llama_moe_hot_vg_hold(g.owner, target);
    const double  ms      = vg_ms(t0);
    (void) vg_poll(true);
    g.n_act++;
    LLAMA_LOG_WARN("%s: [TAG_R40_VGUARD] %s (%" PRIu64 "): H %.0f F %.0f D %.0f deficit %.0f MiB (budget %.0f) -> the share %.0f -> "
            "%.0f MiB (wanted %.0f, %zu groups%s) in %.1f ms (sync %.1f, stash freed %.0f MiB); now H %.0f F %.0f D %.0f deficit "
            "%.0f, %.0f MiB of the lend mapped\n", __func__, why, g.n_act, s0.H, s0.F, d0, f0, s0.budget, before/VG_MIB,
            after/VG_MIB, target/VG_MIB, g.group ? (size_t) (after/g.group) : (size_t) 0, g_zc.on ? ", zero-commit" : "", ms,
            ms_sync, dropped/VG_MIB, g.cur.H, g.cur.F, g.D, g.deficit, llama_moe_hot_vg_free()/VG_MIB);
    return after;
}

// PRESSURE's share: whole groups for the deficit (at most MAX), 0 without PRESSURE
size_t vg_press_b() {
    const vg_state & g = vg();
    if (!g.k.press || !g.pressure) {
        return 0;
    }
    return std::min(vg_groups_for(std::min(g.deficit, g.k.max))*g.group, std::max(vg_max_b(), llama_moe_hot_vg_out()));
}

// the switch / encode rule: PRESSURE, or H / F below the levels -> groups until both are at the levels (+ extra MiB) and the
// share covers the deficit; returns the MiB the share fell short of that (0: covered or nothing to do)
double vg_make_room(const char * why, double extra) {
    vg_state & g = vg();
    const double lh  = vg_lvl_h();
    const double lf  = vg_lvl_f();
    const bool   low = g.cur.H < lh || g.cur.F < lf;
    const size_t before = llama_moe_hot_vg_out();
    double need = low ? std::max(lh - g.cur.H, lf - g.cur.F) + extra : 0.0;
    const size_t press = vg_press_b();
    if (press > before) {
        need = std::max(need, (press - before)/VG_MIB);
    }
    if (g.k.force) {
        need = std::max(need, 2.0*g.group/VG_MIB);
    }
    if (need <= 0.0) {
        return 0.0;
    }
    const size_t target = std::min(before + vg_groups_for(need)*g.group, std::max(vg_max_b(), before));
    const size_t after  = vg_hold(target, true, why);
    const double got    = (after > before ? after - before : 0)/VG_MIB;
    return got < need ? need - got : 0.0;
}

} // namespace

int llama_fn_vg_mode() {
    return vg().mode;
}

bool llama_fn_vg_check() {
    static const bool on = [] {
        const char * e = getenv("LLAMA_R40_VGUARD_CHECK");
        return e && atoi(e) != 0;
    }();
    return on;
}

bool llama_fn_vg_arm(const void * owner, ggml_backend_device * dev, const llama_fn_vg_hooks & hooks, const char * why_not) {
    vg_state & g = vg();
    if (g.armed) {
        return g.owner == owner;
    }
    if (g.mode == 0) {
        vg_off("LLAMA_R40_VGUARD=0 (v0.30.0 behaviour: no WDDM query, no zero-commit)");
        return false;
    }
    if (why_not) {
        vg_off(why_not);
        return false;
    }
#if defined(_WIN32)
    if (!dev || !owner) {
        vg_off("no hot set device");
        return false;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto * luid_fn = reg ? (bool (*)(ggml_backend_dev_t, uint64_t *)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_fn_device_luid") : nullptr;
    uint64_t luid = 0;
    if (!luid_fn || !luid_fn(dev, &luid) || luid == 0) {
        vg_off("the hot set's device has no adapter LUID (not a CUDA device under WDDM)");
        return false;
    }
    HMODULE gdi = GetModuleHandleA("gdi32.dll");
    if (!gdi) {
        gdi = LoadLibraryA("gdi32.dll");
    }
    auto * qs = gdi ? (vg_kmt_qs_fn) (void *) GetProcAddress(gdi, "D3DKMTQueryStatistics") : nullptr;
    if (!qs) {
        vg_off("D3DKMTQueryStatistics not found");
        return false;
    }
    g.K.qs   = qs;
    g.K.luid = luid;
    g.K.proc = OpenProcess(PROCESS_QUERY_INFORMATION, FALSE, GetCurrentProcessId());
    if (!g.K.proc) {
        vg_off("OpenProcess(PROCESS_QUERY_INFORMATION) on itself failed");
        return false;
    }
    const auto fail = [&](const char * why, LONG st) {
        LLAMA_LOG_INFO("%s: [TAG_R40_VGUARD] off: %s (status 0x%lx)\n", "llama_fn_vg_arm", why, (unsigned long) st);
        g.logged_off = true;
        CloseHandle(g.K.proc);
        g.K.proc = nullptr;
        return false;
    };
    LONG st = g.K.query(VG_QS_ADAPTER, nullptr, 0);
    if (st != 0) {
        return fail("the adapter query failed", st);
    }
    const uint32_t n_seg = g.K.u32(24);
    const uint64_t flags = g.K.u64(736);
    if ((flags & 4) == 0) {
        return fail("the adapter does not demote allocations (SupportsDemotion 0): no WDDM paging to guard", 0);
    }
    bool found = false;
    for (uint32_t s = 0; s < n_seg && s < 64 && !found; ++s) {
        if (g.K.query(VG_QS_SEGMENT, nullptr, s) != 0) {
            continue;
        }
        const uint64_t limit = g.K.u64(24);
        const uint32_t aper  = g.K.u32(64);
        if (aper == 0 && limit >= (1ull << 30) && limit < (1ull << 42)) {
            g.K.seg = s;
            found   = true;
        }
    }
    if (!found) {
        return fail("no local (VRAM) segment", 0);
    }
    vg_sample s0;
    if (!g.K.poll(s0) || s0.budget < 1024.0 || s0.usage < 64.0 || s0.limit < 1024.0) {
        return fail("the per-process queries failed or look wrong", 0);
    }
    g.group = llama_moe_hot_vg_group();
    if (g.group == 0) {
        return fail("no KV-lend slot groups (the slot lend on virtual memory)", 0);
    }
    vg_knobs & k = g.k;
    k.htgt     = vg_env("LLAMA_R40_VGUARD_HTGT", k.htgt);
    k.ftgt     = vg_env("LLAMA_R40_VGUARD_FTGT", k.ftgt);
    k.zhtgt    = vg_env("LLAMA_R40_VGUARD_ZHTGT", k.zhtgt);
    k.zftgt    = vg_env("LLAMA_R40_VGUARD_ZFTGT", k.zftgt);
    k.dem      = vg_env("LLAMA_R40_VGUARD_DEM", k.dem);
    k.drop     = vg_env("LLAMA_R40_VGUARD_DROP", k.drop);
    k.max      = std::max(0.0, vg_env("LLAMA_R40_VGUARD_MAX", k.max));
    k.react_f  = vg_env("LLAMA_R40_VGUARD_REACT_F", k.react_f);
    k.poll_ms  = std::max(1, (int) vg_env("LLAMA_R40_VGUARD_POLL_MS", k.poll_ms));
    k.calm_s   = vg_env("LLAMA_R40_VGUARD_CALM_S", k.calm_s);
    k.quiet_s  = vg_env("LLAMA_R40_VGUARD_QUIET_S", k.quiet_s);
    k.regrow_s = vg_env("LLAMA_R40_VGUARD_REGROW_S", k.regrow_s);
    k.press    = (int) vg_env("LLAMA_R40_VGUARD_PRESS", k.press);
    k.react    = (int) vg_env("LLAMA_R40_VGUARD_REACT", k.react);
    k.force    = (int) vg_env("LLAMA_R40_VGUARD_FORCE", k.force);
    // the demotion signal: usage - our PDH dedicated usage, armed only when it is calm now (|D| <= 32 MiB)
    const char * pwhy = nullptr;
    double       ded  = 0.0;
    const int64_t tp  = ggml_time_us();
    if (g.P.open(luid, &pwhy) && g.P.read(ded)) {
        const double raw0 = s0.usage - ded;
        if (std::fabs(raw0) <= 32.0) {
            g.det   = true;
            g.d_off = raw0;
        } else {
            LLAMA_LOG_WARN("%s: [TAG_R40_VGUARD] demotion detection off: usage %.0f - PDH dedicated %.0f = %.0f MiB at arm (more than "
                    "32 MiB)\n", __func__, s0.usage, ded, raw0);
        }
    } else {
        LLAMA_LOG_WARN("%s: [TAG_R40_VGUARD] demotion detection off: %s\n", __func__, pwhy ? pwhy : "the first PDH read failed");
    }
    const double pdh_ms = vg_ms(tp);
    g.solo  = s0.budget;
    g.owner = owner;
    g.h     = hooks;
    g.armed = true;
    (void) vg_poll(true);
    char path[192] = {};
    (void) WideCharToMultiByte(CP_UTF8, 0, g.P.path, -1, path, (int) sizeof(path) - 1, nullptr, nullptr);
    LLAMA_LOG_WARN("%s: [TAG_R40_VGUARD] %s (LLAMA_R40_VGUARD=%d): adapter LUID 0x%016" PRIx64 ", local segment %u (%.0f MiB); now H %.0f "
            "F %.0f MiB (budget %.0f = solo, usage %.0f, PDH dedicated %.0f, D offset %.1f%s; %s, open %.1f ms); levels H %.0f F %.0f "
            "(zero-commit switch %.0f / %.0f), PRESSURE deficit >= %.0f, DEMOTED D >= %.0f (react %d to F %.0f), share <= %.0f MiB "
            "in groups of %.0f MiB, regrow after %.0f s calm / %.0f s quiet, one per %.0f s%s%s%s\n", __func__,
            g.mode == 2 ? "observing" : "armed", g.mode, luid, g.K.seg, g.cur.limit, g.cur.H, g.cur.F, g.cur.budget, g.cur.usage,
            g.cur.ded, g.d_off, g.det ? "" : ", detection off", path, pdh_ms, k.htgt, k.ftgt, k.zhtgt, k.zftgt, k.drop, k.dem,
            k.react, k.react_f, k.max, g.group/VG_MIB, k.calm_s, k.quiet_s, k.regrow_s, k.press ? "" : ", no PRESSURE share",
            llama_fn_vg_check() ? ", CHECK" : "", k.force ? ", FORCE (test)" : "");
    return true;
#else
    GGML_UNUSED(owner);
    GGML_UNUSED(dev);
    GGML_UNUSED(hooks);
    vg_off("not a Windows (WDDM) build");
    return false;
#endif
}

bool llama_fn_vg_switch_pre() {
    vg_state & g = vg();
    if (!g.armed) {
        return false;
    }
    g.n_sw++;
    g.t_prompt = ggml_time_us();
    if (!vg_poll(true)) {
        return false;
    }
    const bool low = g.cur.H < vg_lvl_h() || g.cur.F < vg_lvl_f();
    const bool act = low || vg_press_b() > llama_moe_hot_vg_out();
    if (g.mode == 2 || act || g.n_sw <= 8 || g.n_sw % 64 == 0) {
        LLAMA_LOG_WARN("%s: [TAG_R40_VGUARD] FULL switch %" PRIu64 ": H %.0f F %.0f D %.0f MiB (budget %.0f solo %.0f deficit %.0f, usage "
                "%.0f, PDH dedicated %.0f)%s%s%s; share %.0f MiB, %.0f MiB of the lend mapped; %" PRIu64 " polls (max %.1f us, %" PRIu64
                " PDH misses), %" PRIu64 " actions, gates %" PRIu64 " (%" PRIu64 " deferred)\n", __func__, g.n_sw, g.cur.H, g.cur.F,
                g.D, g.cur.budget, g.solo, g.deficit, g.cur.usage, g.cur.ded, act ? " ACT" : "", low ? " (low)" : "",
                g.demoted ? " DEMOTED" : g.pressure ? " PRESSURE" : "", llama_moe_hot_vg_out()/VG_MIB, llama_moe_hot_vg_free()/VG_MIB,
                g.n_poll, g.poll_us_max, g.n_pdh_fail, g.n_act, g.n_gate, g.n_defer);
    }
    if (g.mode != 1) {
        return false;
    }
    const double short_ = vg_make_room("FULL switch: room before the tail goes", 0.0);
    if (short_ > 0.0 && !g_zc.on) {
        // the v0.30.0 switch commits the FULL reserve: the base width commits less (a zero-commit switch commits nothing)
        LLAMA_LOG_WARN("%s: [TAG_R40_VGUARD] the groups fell %.0f MiB short: this prompt runs at the base width (no borrow)\n",
                __func__, short_);
        return true;
    }
    return false;
}

void llama_fn_vg_decode(bool small, bool gen_only) {
    vg_state & g = vg();
    if (!g.armed) {
        return;
    }
    g.n_dec++;
    if (!gen_only) {
        g.t_prompt = ggml_time_us();
    }
    const bool fresh = vg_poll(false);
    if (g.mode != 1) {
        return;
    }
    if (g.k.force) {
        if (g.n_dec % 8 == 0) {
            const size_t out = llama_moe_hot_vg_out();
            (void) vg_hold(out > 0 ? 0 : 2*g.group, true, "FORCE (test): decode entry");
        }
        return;
    }
    if (!fresh) {
        return;
    }
    const int64_t t   = g.cur.t;
    const size_t  out = llama_moe_hot_vg_out();
    const size_t  pb  = vg_press_b();
    if (pb > out) {
        (void) vg_hold(pb, true, "decode entry: PRESSURE");
        return;
    }
    if (g.demoted && g.k.react && g.cur.F < g.k.react_f && t - g.t_react >= 1000000) {
        g.t_react = t;
        (void) vg_hold(out + vg_groups_for(g.k.react_f - g.cur.F)*g.group, true, "decode entry: DEMOTED (REACT)");
        return;
    }
    const double grp = g.group/VG_MIB;
    if (small && gen_only && out > 0 && g.t_calm0 > 0 && t - g.t_calm0 >= (int64_t) (g.k.calm_s*1e6) &&
        t - g.t_prompt >= (int64_t) (g.k.quiet_s*1e6) && t - g.t_regrow >= (int64_t) (g.k.regrow_s*1e6) &&
        g.cur.H - grp >= vg_lvl_h() && g.cur.F - grp >= vg_lvl_f()) {
        g.t_regrow = t;
        (void) vg_hold(out > g.group ? out - g.group : 0, true, "regrow: one group back");
    }
}

void llama_fn_vg_encode() {
    vg_state & g = vg();
    if (!g.armed) {
        return;
    }
    g.t_prompt = ggml_time_us();
    if (!vg_poll(true)) {
        return;
    }
    const bool low = g.cur.H < vg_lvl_h() || g.cur.F < vg_lvl_f();
    const bool act = low || vg_press_b() > llama_moe_hot_vg_out();
    if (g.mode == 2 || act) {
        LLAMA_LOG_WARN("%s: [TAG_R40_VGUARD] encode: H %.0f F %.0f D %.0f deficit %.0f MiB%s, the projector needs %.0f MiB%s\n",
                __func__, g.cur.H, g.cur.F, g.D, g.deficit, act ? " ACT" : "", llama_moe_hot_vis_bytes()/VG_MIB,
                g_zc.on ? " (zero-commit take)" : "");
    }
    if (g.mode != 1) {
        return;
    }
    // a zero-commit take maps the projector from the lend's own handles: no room needed for it
    (void) vg_make_room("encode: room before the projector maps", g_zc.on ? 0.0 : llama_moe_hot_vis_bytes()/VG_MIB);
}

void llama_fn_vg_gate() {
    vg_state & g = vg();
    if (!g.armed || g.mode != 1) {
        return;
    }
    if (!vg_poll(false)) {
        return;
    }
    const int64_t t     = g.cur.t;
    const size_t  out   = llama_moe_hot_vg_out();
    const bool    force = g.k.force && out == 0;
    const size_t  pb    = vg_press_b();
    const bool    react = g.demoted && g.k.react && g.cur.F < g.k.react_f && t - g.t_react >= 1000000;
    if (!force && pb <= out && !react) {
        return;
    }
    g.n_gate++;
    if (g.h.idle && !g.h.idle(g.h.user)) {
        g.n_defer++; // a stream or the hot step still runs: the next gate tries again (never wait here)
        return;
    }
    if (force) {
        (void) vg_hold(g.group, false, "FORCE (test): prefill-stream gate");
        return;
    }
    if (pb > out) {
        (void) vg_hold(pb, false, "prefill-stream gate: PRESSURE");
        return;
    }
    g.t_react = t;
    (void) vg_hold(out + vg_groups_for(g.k.react_f - g.cur.F)*g.group, false, "prefill-stream gate: DEMOTED (REACT)");
}

// [TAG_R40_VGUARD] zero-commit glue ---------------------------------------------------------------------------------------

bool llama_fn_zc_arm(ggml_backend_device * dev) {
    if (g_zc.on || g_zc.tried) {
        return g_zc.on;
    }
    g_zc.tried = true;
    if (vg().mode == 0) {
        return false; // LLAMA_R40_VGUARD=0: v0.30.0 (the guard's "off" line says so)
    }
    const char * e = getenv("LLAMA_R40_ZC");
    if (e && *e && atoi(e) == 0) {
        LLAMA_LOG_INFO("%s: [TAG_R40_VGUARD] zero-commit off: LLAMA_R40_ZC=0 (the v0.30.0 switch: frees and creates its memory)\n",
                __func__);
        return false;
    }
#if defined(_WIN32)
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    auto * luid_fn = reg ? (bool (*)(ggml_backend_dev_t, uint64_t *)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_fn_device_luid") : nullptr;
    uint64_t luid = 0;
    if (!luid_fn || !luid_fn(dev, &luid) || luid == 0) {
        LLAMA_LOG_INFO("%s: [TAG_R40_VGUARD] zero-commit off: the hot set's device has no adapter LUID (not a CUDA device under WDDM)\n",
                __func__);
        return false;
    }
    g_zc.enable = (decltype(g_zc.enable)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_zc_enable");
    g_zc.scope  = (decltype(g_zc.scope))  ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_zc_scope");
    g_zc.window = (decltype(g_zc.window)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_zc_window");
    g_zc.drop   = (decltype(g_zc.drop))   ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_zc_drop");
    g_zc.stats  = (decltype(g_zc.stats))  ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_zc_stats");
    const bool park = ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_buffer_unmap_park") != nullptr;
    if (!g_zc.enable || !g_zc.scope || !g_zc.window || !g_zc.drop || !g_zc.stats || !park) {
        LLAMA_LOG_INFO("%s: [TAG_R40_VGUARD] zero-commit off: the backend has no zero-commit procs\n", __func__);
        return false;
    }
    if (!g_zc.enable(dev, true)) {
        LLAMA_LOG_INFO("%s: [TAG_R40_VGUARD] zero-commit off: no virtual memory management on the device\n", __func__);
        return false;
    }
    g_zc.dev = dev;
    g_zc.on  = true;
    (void) g_zc.stats(dev, g_zc.last, ZC_NST);
    LLAMA_LOG_WARN("%s: [TAG_R40_VGUARD] zero-commit on (LLAMA_R40_ZC=1): the tail's and the borrow's handles carry the FULL reserve "
            "and the compute pool's FULL growth and come back at SMALL; a KV growth and the projector take slot-group handles; no "
            "new WDDM allocation at a switch unless the stash falls short (logged)\n", __func__);
    return true;
#else
    GGML_UNUSED(dev);
    LLAMA_LOG_INFO("%s: [TAG_R40_VGUARD] zero-commit off: not a Windows (WDDM) build\n", __func__);
    return false;
#endif
}

bool llama_fn_zc_on() {
    return g_zc.on;
}

void llama_fn_zc_scope(int delta) {
    if (g_zc.on) {
        (void) g_zc.scope(g_zc.dev, delta);
    }
}

void llama_fn_zc_window(int delta) {
    if (g_zc.on) {
        (void) g_zc.window(g_zc.dev, delta);
    }
}

size_t llama_fn_zc_stash_bytes() {
    if (!g_zc.on) {
        return 0;
    }
    uint64_t now[ZC_NST] = {};
    return g_zc.stats(g_zc.dev, now, ZC_NST) >= ZC_NST ? (size_t) now[ZC_B_STASH] : 0;
}

void llama_fn_zc_note(bool full, double ms) {
    if (!g_zc.on) {
        return;
    }
    uint64_t now[ZC_NST] = {};
    if (g_zc.stats(g_zc.dev, now, ZC_NST) < ZC_NST) {
        return;
    }
    uint64_t d[ZC_NST];
    for (int i = 0; i < ZC_NST; ++i) {
        d[i] = now[i] - g_zc.last[i];
        g_zc.last[i] = now[i];
    }
    d[ZC_B_STASH] = now[ZC_B_STASH];
    const uint64_t n = full ? ++g_zc.n_full : ++g_zc.n_small;
    (full ? g_zc.ms_full : g_zc.ms_small) += ms;
    if (n > 8 && n % 64 != 0 && d[ZC_B_NEW] == 0 && d[ZC_N_FAIL] == 0) {
        return;
    }
    const double avg = (full ? g_zc.ms_full : g_zc.ms_small)/(double) n;
    if (full) {
        LLAMA_LOG_WARN("%s: [TAG_R40_VGUARD] zero-commit FULL %" PRIu64 " in %.1f ms (%.1f on average): %.0f MiB parked (%" PRIu64
                " handles), the reserve %.0f MiB in %" PRIu64 " arena buffer(s) from %.0f MiB of them (%" PRIu64 " handles, %.1f ms), "
                "new %.0f MiB (%" PRIu64 "), %.0f MiB stay for the pool, %" PRIu64 " failures\n", __func__, n, ms, avg,
                d[ZC_B_PARK]/VG_MIB, d[ZC_N_PARK], d[ZC_B_ARENA]/VG_MIB, d[ZC_N_ARENA], d[ZC_B_REUSE]/VG_MIB, d[ZC_N_REUSE],
                d[ZC_US]/1000.0, d[ZC_B_NEW]/VG_MIB, d[ZC_N_NEW], d[ZC_B_STASH]/VG_MIB, d[ZC_N_FAIL]);
    } else {
        LLAMA_LOG_WARN("%s: [TAG_R40_VGUARD] zero-commit SMALL %" PRIu64 " in %.1f ms (%.1f on average): the FULL period's pool took "
                "%.0f MiB (%" PRIu64 " growths) from the stash; %.0f MiB parked (%" PRIu64 " handles: the reserve, the pool, KV), %.0f "
                "MiB mapped from the stash (%" PRIu64 " handles: the tail, the borrow, KV), new %.0f MiB (%" PRIu64 "), %.0f MiB freed "
                "(%" PRIu64 " handles left over), %" PRIu64 " failures\n", __func__, n, ms, avg, d[ZC_B_POOL]/VG_MIB, d[ZC_N_POOL],
                d[ZC_B_PARK]/VG_MIB, d[ZC_N_PARK], d[ZC_B_REUSE]/VG_MIB, d[ZC_N_REUSE], d[ZC_B_NEW]/VG_MIB, d[ZC_N_NEW],
                d[ZC_B_DROP]/VG_MIB, d[ZC_N_DROP], d[ZC_N_FAIL]);
    }
}
