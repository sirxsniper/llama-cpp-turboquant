// [TAG_R40_VGUARD] the WDDM paging guard: see llama-fn-vguard.h
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

#if defined(_WIN32)
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#endif

namespace {

constexpr double VG_MIB = 1048576.0;

[[maybe_unused]] double vg_env(const char * name, double def) {
    const char * e = getenv(name);
    return (e && *e) ? atof(e) : def;
}

struct vg_knobs {
    double hact = 2304, fact = 2560, htgt = 2944, ftgt = 3200; // MiB
    double dem = 64, drop = 256, max = 2048;                    // MiB
    int    poll_ms = 20;
    double calm_s  = 10;
    int    react = 1, pregrow = 1, force = 0;
};

// one poll's numbers (MiB)
struct vg_sample {
    int64_t t      = 0;
    double  budget = 0, usage = 0, resident = 0, limit = 0, seg_res = 0;
    double  H = 0, F = 0, draw = 0;
};

#if defined(_WIN32)
// D3DKMTQueryStatistics on a raw 808-byte D3DKMT_QUERYSTATISTICS (SDK 10.0.26100 layout, r40 understand/detect_notes.txt):
// Type at 0, AdapterLuid at 4, hProcess at 16, QueryResult at 24, the input union at 800. Any layout surprise shows as a
// failed or implausible answer at arm time, and the guard stays off
using vg_kmt_qs_fn = LONG (APIENTRY *)(void *);

enum : uint32_t {
    VG_QS_ADAPTER               = 0,
    VG_QS_SEGMENT               = 3,
    VG_QS_PROCESS_SEGMENT       = 4,
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
        if (query(VG_QS_PROCESS_SEGMENT, proc, seg) != 0) {
            return false;
        }
        uint64_t r = u64(152); // AllocsResidentInNonPreferred.Bytes
        for (int i = 0; i < 5; ++i) {
            r += u64(72 + 16*(size_t) i); // AllocsResidentInP[i].Bytes
        }
        s.resident = r/VG_MIB;
        if (query(VG_QS_SEGMENT, nullptr, seg) != 0) {
            return false;
        }
        s.limit   = u64(24)/VG_MIB;
        s.seg_res = u64(40)/VG_MIB;
        return true;
    }
};
#endif

struct vg_state {
    int          mode   = -1;
    bool         armed  = false;
    bool         logged_off = false;
    const void * owner  = nullptr;
    llama_fn_vg_hooks h;
    vg_knobs     k;
#if defined(_WIN32)
    vg_kmt       K;
#endif
    // the last poll and the rules' memory
    vg_sample cur;
    bool      have      = false;
    int       n_fail    = 0;
    double    d_base    = 0.0;
    bool      d_base_ok = false;
    double    D         = 0.0;
    int       dem_n     = 0;
    int64_t   t_dem0    = 0;     // the first poll of the current D >= DEM run
    int64_t   t_d_low   = 0;     // D <= 16 since (0: not now)
    bool      demoted   = false;
    double    calm_budget = 0.0;
    bool      calm_ok   = false;
    bool      pressure  = false;
    int64_t   t_press   = 0;     // the last poll with PRESSURE
    // the share
    size_t    group     = 0;     // one KV group's bytes
    size_t    sw_part   = 0;     // of the share: taken at a switch / an encode (returns at SMALL)
    double    react_last = 0.0;  // the last reactive amount (MiB)
    int64_t   t_react   = 0;
    int64_t   t_regrow  = 0;
    size_t    full_hwm  = 0;
    // statistics
    uint64_t  n_poll = 0, n_act = 0, n_sw = 0, n_dec = 0, n_gate = 0, n_defer = 0, n_trans = 0;
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

[[maybe_unused]] void vg_note(const char * what) {
    vg_state & g = vg();
    g.n_trans++;
    if (g.n_trans <= 32 || g.n_trans % 64 == 0) {
        const vg_sample & s = g.cur;
        LLAMA_LOG_WARN("%s: [TAG_R40_VGUARD] %s: H %.0f F %.0f D %.0f MiB (budget %.0f usage %.0f resident %.0f, calm budget %.0f), "
                "share %.0f MiB\n", __func__, what, s.H, s.F, g.D, s.budget, s.usage, s.resident, g.calm_budget,
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
    g.poll_us_max = std::max(g.poll_us_max, (double) (ggml_time_us() - t));
    g.n_poll++;
    s.t    = t;
    s.H    = s.budget - s.usage;
    s.F    = s.limit - s.seg_res;
    s.draw = s.usage - s.resident;
    if (!g.d_base_ok) {
        g.d_base    = s.draw;
        g.d_base_ok = true;
    }
    if (s.draw < g.d_base) {
        g.d_base = s.draw;
    }
    g.D = s.draw - g.d_base;
    if (g.D >= g.k.dem) {
        if (g.dem_n++ == 0) {
            g.t_dem0 = t;
        }
    } else {
        g.dem_n = 0;
    }
    const bool was_dem = g.demoted;
    g.demoted = g.dem_n >= 2 && t - g.t_dem0 >= 20000;
    if (g.D <= 16.0) {
        if (g.t_d_low == 0) {
            g.t_d_low = t;
        }
        g.d_base += 0.02*(s.draw - g.d_base); // a calm offset that creeps up is not a demotion
    } else {
        g.t_d_low = 0;
    }
    g.cur  = s;
    g.have = true;
    const bool calm = s.H >= g.k.hact && s.F >= g.k.fact && !g.demoted;
    if (calm) {
        g.calm_budget = s.budget;
        g.calm_ok     = true;
    }
    const bool was_press = g.pressure;
    g.pressure = g.calm_ok && g.calm_budget - s.budget >= g.k.drop && (s.H < g.k.htgt || s.F < g.k.ftgt);
    if (g.pressure) {
        g.t_press = t;
    }
    if (g.demoted != was_dem) {
        vg_note(g.demoted ? "DEMOTED begins" : "DEMOTED ends");
    }
    if (g.pressure != was_press) {
        vg_note(g.pressure ? "PRESSURE begins" : "PRESSURE ends");
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

// the share to `target` bytes (whole groups, at most MAX); sync: the owner's graphs first (r35_hook_sync). One log line
size_t vg_hold(size_t target, bool sync, const char * why) {
    vg_state & g = vg();
    target = std::min(target, std::max(vg_max_b(), llama_moe_hot_vg_out()));
    const size_t before = llama_moe_hot_vg_out();
    if (target == before) {
        return before;
    }
    const vg_sample s0 = g.cur;
    const double    d0 = g.D;
    const int64_t   t0 = ggml_time_us();
    if (sync && g.h.sync) {
        g.h.sync(g.h.user);
    }
    const double  ms_sync = vg_ms(t0);
    const size_t  after   = llama_moe_hot_vg_hold(g.owner, target);
    const double  ms      = vg_ms(t0);
    (void) vg_poll(true);
    g.n_act++;
    LLAMA_LOG_WARN("%s: [TAG_R40_VGUARD] %s (%" PRIu64 "): H %.0f F %.0f D %.0f MiB (budget %.0f) -> the share %.0f -> %.0f MiB "
            "(wanted %.0f, %zu groups) in %.1f ms (sync %.1f); now H %.0f F %.0f D %.0f, %.0f MiB of the lend mapped\n", __func__,
            why, g.n_act, s0.H, s0.F, d0, s0.budget, before/VG_MIB, after/VG_MIB, target/VG_MIB,
            g.group ? (size_t) (after/g.group) : (size_t) 0, ms, ms_sync, g.cur.H, g.cur.F, g.D, llama_moe_hot_vg_free()/VG_MIB);
    return after;
}

// what DEMOTED / PRESSURE asks for now (MiB, 0: nothing)
double vg_react_mib(int64_t t) {
    vg_state & g = vg();
    const double grp = g.group/VG_MIB;
    double a = 0.0;
    if (g.demoted) {
        a = std::max(2.0*grp, g.D + grp);
        if (g.react_last > 0.0 && g.t_react > 0 && t - g.t_react <= 1500000) {
            a = std::max(a, 2.0*g.react_last); // doubled per further second of demotion
        }
    }
    if (g.pressure) {
        const double drop = g.calm_budget - g.cur.budget;
        const double tgt  = std::max(g.k.htgt - g.cur.H, g.k.ftgt - g.cur.F);
        a = std::max(a, std::min(drop, std::max(tgt, 0.0)));
    }
    return a;
}

void vg_react(bool sync, const char * why) {
    vg_state & g = vg();
    const int64_t t = ggml_time_us();
    const double  a = vg_react_mib(t);
    if (a <= 0.0) {
        return;
    }
    const size_t before = llama_moe_hot_vg_out();
    const size_t target = std::min(before + vg_groups_for(a)*g.group, std::max(vg_max_b(), before));
    g.t_react    = t;
    g.react_last = a;
    if (target <= before) {
        return; // at MAX: nothing more to give
    }
    (void) vg_hold(target, sync, why);
}

// the switch / encode rule: ACT -> groups up to TARGET (+ extra MiB); returns true when the groups fell short of the need
bool vg_make_room(const char * why, double extra) {
    vg_state & g = vg();
    const bool act   = g.cur.H < g.k.hact || g.cur.F < g.k.fact;
    const bool force = g.k.force != 0;
    if (!act && !force) {
        return false;
    }
    double need = act ? std::max(g.k.htgt - g.cur.H, g.k.ftgt - g.cur.F) + extra : 0.0;
    if (force) {
        need = std::max(need, 2.0*g.group/VG_MIB);
    }
    if (need <= 0.0) {
        return false;
    }
    const size_t before = llama_moe_hot_vg_out();
    const size_t target = std::min(before + vg_groups_for(need)*g.group, std::max(vg_max_b(), before));
    const size_t after  = vg_hold(target, true, why);
    g.sw_part += after > before ? after - before : 0;
    return (double) (after > before ? after - before : 0) < need*VG_MIB;
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
        vg_off("LLAMA_R40_VGUARD=0 (v0.30.0 behaviour: no WDDM query, no pool pre-grow)");
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
    if (!g.K.poll(s0) || s0.budget < 1024.0 || s0.resident < 64.0 || s0.limit < 1024.0) {
        return fail("the per-process queries failed or look wrong", 0);
    }
    g.group = llama_moe_hot_vg_group();
    if (g.group == 0) {
        return fail("no KV-lend slot groups (the slot lend on virtual memory)", 0);
    }
    vg_knobs & k = g.k;
    k.hact    = vg_env("LLAMA_R40_VGUARD_HACT", k.hact);
    k.fact    = vg_env("LLAMA_R40_VGUARD_FACT", k.fact);
    k.htgt    = std::max(k.hact, vg_env("LLAMA_R40_VGUARD_HTGT", k.htgt));
    k.ftgt    = std::max(k.fact, vg_env("LLAMA_R40_VGUARD_FTGT", k.ftgt));
    k.dem     = vg_env("LLAMA_R40_VGUARD_DEM", k.dem);
    k.drop    = vg_env("LLAMA_R40_VGUARD_DROP", k.drop);
    k.max     = std::max(0.0, vg_env("LLAMA_R40_VGUARD_MAX", k.max));
    k.poll_ms = std::max(1, (int) vg_env("LLAMA_R40_VGUARD_POLL_MS", k.poll_ms));
    k.calm_s  = vg_env("LLAMA_R40_VGUARD_CALM_S", k.calm_s);
    k.react   = (int) vg_env("LLAMA_R40_VGUARD_REACT", k.react);
    k.pregrow = (int) vg_env("LLAMA_R40_VGUARD_PREGROW", k.pregrow);
    k.force   = (int) vg_env("LLAMA_R40_VGUARD_FORCE", k.force);
    g.owner = owner;
    g.h     = hooks;
    g.armed = true;
    (void) vg_poll(true);
    LLAMA_LOG_WARN("%s: [TAG_R40_VGUARD] %s (LLAMA_R40_VGUARD=%d): adapter LUID 0x%016" PRIx64 ", local segment %u (%.0f MiB); "
            "now H %.0f F %.0f MiB (budget %.0f, usage %.0f, resident %.0f); ACT H < %.0f or F < %.0f -> TARGET H >= %.0f and "
            "F >= %.0f; DEMOTED D >= %.0f, PRESSURE drop >= %.0f, share <= %.0f MiB in groups of %.0f MiB, react %d, pre-grow %d%s%s\n",
            __func__, g.mode == 2 ? "observing" : "armed", g.mode, luid, g.K.seg, g.cur.limit, g.cur.H, g.cur.F, g.cur.budget,
            g.cur.usage, g.cur.resident, k.hact, k.fact, k.htgt, k.ftgt, k.dem, k.drop, k.max, g.group/VG_MIB, k.react,
            k.pregrow, llama_fn_vg_check() ? ", CHECK" : "", k.force ? ", FORCE (test)" : "");
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
    if (!vg_poll(true)) {
        return false;
    }
    const bool act = g.cur.H < g.k.hact || g.cur.F < g.k.fact;
    if (g.mode == 2 || act || g.n_sw <= 8 || g.n_sw % 64 == 0) {
        LLAMA_LOG_WARN("%s: [TAG_R40_VGUARD] FULL switch %" PRIu64 ": H %.0f F %.0f D %.0f MiB (budget %.0f, usage %.0f, resident %.0f)%s%s; "
                "share %.0f MiB, %.0f MiB of the lend mapped; %" PRIu64 " polls (max %.1f us), %" PRIu64 " actions\n", __func__,
                g.n_sw, g.cur.H, g.cur.F, g.D, g.cur.budget, g.cur.usage, g.cur.resident, act ? " ACT" : "",
                g.demoted ? " DEMOTED" : g.pressure ? " PRESSURE" : "", llama_moe_hot_vg_out()/VG_MIB,
                llama_moe_hot_vg_free()/VG_MIB, g.n_poll, g.poll_us_max, g.n_act);
    }
    if (g.mode != 1) {
        return false;
    }
    const bool short_ = vg_make_room("FULL switch: room before the tail goes", 0.0);
    if (short_) {
        LLAMA_LOG_WARN("%s: [TAG_R40_VGUARD] the groups did not cover the need: this prompt runs at the base width (no borrow)\n",
                __func__);
    }
    return short_;
}

size_t llama_fn_vg_pregrow_bytes() {
    const vg_state & g = vg();
    if (!g.armed || g.mode != 1 || g.k.pregrow == 0) {
        return 0;
    }
    return std::min(g.full_hwm, (size_t) 512 << 20);
}

void llama_fn_vg_full_hwm(size_t hwm) {
    vg_state & g = vg();
    if (g.armed) {
        g.full_hwm = hwm;
    }
}

void llama_fn_vg_small() {
    vg_state & g = vg();
    if (!g.armed || g.mode != 1) {
        return;
    }
    const size_t out = llama_moe_hot_vg_out();
    g.sw_part = std::min(g.sw_part, out); // the KV cache / projector may have taken some over
    if (g.sw_part == 0) {
        return;
    }
    (void) vg_poll(true);
    if (g.cur.H - g.sw_part/VG_MIB < 512.0) {
        LLAMA_LOG_WARN("%s: [TAG_R40_VGUARD] SMALL: the switch share of %.0f MiB stays out (H %.0f MiB: mapping it now would leave "
                "< 512 MiB); the decode-entry regrow brings it back\n", __func__, g.sw_part/VG_MIB, g.cur.H);
        g.sw_part = 0; // the regrow's from here on
        return;
    }
    const size_t keep = out - g.sw_part;
    g.sw_part = 0;
    (void) vg_hold(keep, true, "SMALL: the switch share comes back");
}

void llama_fn_vg_decode(bool small) {
    vg_state & g = vg();
    if (!g.armed) {
        return;
    }
    g.n_dec++;
    const bool fresh = vg_poll(false);
    if (g.mode != 1) {
        return;
    }
    const int64_t t = g.cur.t;
    if (g.k.force) {
        if (g.n_dec % 8 == 0) {
            const size_t out = llama_moe_hot_vg_out();
            g.sw_part = 0;
            (void) vg_hold(out > 0 ? 0 : 2*g.group, true, "FORCE (test): decode entry");
        }
        return;
    }
    if (!fresh) {
        return;
    }
    if (g.k.react && (g.demoted || g.pressure) && ggml_time_us() - g.t_react >= 1000000) {
        vg_react(true, g.demoted ? "decode entry: DEMOTED" : "decode entry: PRESSURE");
        return;
    }
    const size_t out = llama_moe_hot_vg_out();
    if (small && out > 0 && !g.demoted && !g.pressure && g.t_d_low > 0 && t - g.t_d_low >= 3000000 &&
        (g.t_press == 0 || t - g.t_press >= (int64_t) (g.k.calm_s*1e6)) && g.cur.H - g.group/VG_MIB >= 512.0 &&
        t - g.t_regrow >= 2000000) {
        g.t_regrow = t;
        const size_t target = out > g.group ? out - g.group : 0;
        g.sw_part = std::min(g.sw_part, target);
        (void) vg_hold(target, true, "regrow: one group back");
    }
}

void llama_fn_vg_encode() {
    vg_state & g = vg();
    if (!g.armed) {
        return;
    }
    if (!vg_poll(true)) {
        return;
    }
    const bool act = g.cur.H < g.k.hact || g.cur.F < g.k.fact;
    if (g.mode == 2 || act) {
        LLAMA_LOG_WARN("%s: [TAG_R40_VGUARD] encode: H %.0f F %.0f D %.0f MiB%s, the projector needs %.0f MiB\n", __func__,
                g.cur.H, g.cur.F, g.D, act ? " ACT" : "", llama_moe_hot_vis_bytes()/VG_MIB);
    }
    if (g.mode != 1) {
        return;
    }
    (void) vg_make_room("encode: room before the projector maps", llama_moe_hot_vis_bytes()/VG_MIB);
}

void llama_fn_vg_gate() {
    vg_state & g = vg();
    if (!g.armed || g.mode != 1 || (!g.k.react && !g.k.force)) {
        return;
    }
    if (!vg_poll(false)) {
        return;
    }
    const bool force = g.k.force && llama_moe_hot_vg_out() == 0;
    if (!force && !(g.demoted || g.pressure)) {
        return;
    }
    if (!force && ggml_time_us() - g.t_react < 1000000) {
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
    vg_react(false, g.demoted ? "prefill-stream gate: DEMOTED" : "prefill-stream gate: PRESSURE");
}
