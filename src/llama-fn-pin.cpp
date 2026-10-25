// [TAG_FN_L22_PIN] round 22 P6: pinned (registered) host ranges of the streamed experts, the DXGI budget helper and the
// out-of-memory unpin hook (src/llama-fn-l22.h, plan section P6, GPU-fault class rules 3.6).
//
// LLAMA_FN_L22_PIN (qwen4exp profile / environment; unset or 0 = nothing here runs: no thread, no state, no call):
//   1  at ready (llama_fn_pin_on_ready, called by the warm pass after the FIT and the residency table), the RAM-class
//      experts are planned as per-expert coalesced ranges of the mapped model (LLAMA_FN_L22_PIN_UNIT=ranges, default;
//      spans = one span per fully streamed layer; arena = the fallback copy), marked PIN so the warm pass skips them,
//      and one background thread registers them range by range (Portable | ReadOnly) on pages this process has not
//      touched: the driver brings them in. Budget after every range: DXGI NON_LOCAL (budget - usage - charge >=
//      LLAMA_FN_L22_PIN_MARGIN_MIB 16384, at least reserve_need + 2 GiB: [TAG_FN_L23_PINFIX] R-L23-5), DXGI LOCAL (headroom >= min(reserve_need + LLAMA_FN_L22_PIN_LOCAL_MARGIN_MIB
//      1024, headroom before the first pin - 256 MiB)), RAM (available - new bytes >= LLAMA_FN_L22_PIN_FLOOR_MIB 6144 +
//      LLAMA_FN_L22_PIN_WARM_KEEP_MIB 0), LLAMA_FN_L22_PIN_MAX_MIB (0 = no cap). Stop at the first refusal, CUDA error
//      or rule; the last range is unregistered on a rule break.
//   2  as 1, but each range is touched (read in) before it is registered ("touched" mode, when the probe says
//      untouched registration fails or is slow but touched pages lock).
// The arena (spec 9; LLAMA_FN_L22_PIN_UNIT=arena or a refusal at the very first range): committed private chunks of
// 1 GiB (<= min(LLAMA_FN_L22_PIN_ARENA_MIB 16384, commit headroom - 8 GiB)) registered at once (before touch), filled
// with the planned experts by direct I/O (llama_fn_dio_submit; the mapped bytes when it cannot), checked against the
// mapping; llama_fn_pin_query answers the copy's address. Arena experts are not marked PIN: the decode CPU path still
// reads their mapped copies, which the warm pass keeps.
// After registration the thread watches every 2 s: LOCAL headroom below the watcher threshold (reserve_need, at most
// the ready headroom - 512 MiB; LLAMA_FN_L22_PIN_TEST_LOCAL_MIB forces one) or a RAM room below the floor for two
// samples unpins the last ranges (as the OOM hook). [TAG_FN_L23_PINFIX] Pins may only ever add speed: the thread waits for
// the warm pass and pins only resident ranges (no PIN mark before; a pin never reads the disk); available RAM under 4 GiB for
// 6 s drops ALL pins (the process then runs as without them; an unpin only unlocks), room >= 10 GiB for 60 s pins the
// resident units again (fn_pin::gate_step). An unpin by the OOM hook or the watcher ends the first registration pass.
// llama_fn_pin_on_oom(need): the callers' allocation hook (plan 3.6): drain the last ranges (tickets back to 0, bounded),
// unregister them until LOCAL headroom >= need + the pre-check margin (512 MiB, at most the watcher threshold), or until
// the freed bytes reach the cap from the measured LOCAL charge per byte. [TAG_FN_L23_PINFIX] With no measured LOCAL
// charge it unpins only the NON_LOCAL spill / arena commit deficit, and nothing when an unpin cannot raise the room.

#include "llama-fn-l22.h"
#include "llama-fn-l22-p5.h" // [TAG_FN_L23_PINFIX] llama_fn_p5_warm_done
#include "llama-fn-pin-core.h"

#include "llama-fn-arena.h"
#include "llama-fn-auto.h"
#include "llama-fn-tier.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-moecache.h"

#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#    include <psapi.h>
#    include <dxgi1_4.h>
#else
#    include <sys/mman.h>
#    include <unistd.h>
#    include <fstream>
#endif

namespace {

constexpr uint64_t FN_MIB = 1ull << 20;
constexpr uint64_t FN_GIB = 1ull << 30;

int64_t fn_now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void fn_sleep_ms(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

double fn_gib(uint64_t b) {
    return (double) b/(double) FN_GIB;
}

// ---------------------------------------------------------------------------------------------------------------------
// OS helpers

size_t fn_page_size() {
#if defined(_WIN32)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwPageSize ? si.dwPageSize : 4096;
#else
    const long p = sysconf(_SC_PAGESIZE);
    return p > 0 ? (size_t) p : 4096;
#endif
}

uint64_t fn_avail_ram() {
#if defined(_WIN32)
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    return GlobalMemoryStatusEx(&ms) ? ms.ullAvailPhys : 0;
#else
    std::ifstream f("/proc/meminfo");
    std::string k;
    uint64_t v = 0;
    std::string unit;
    while (f >> k >> v >> unit) {
        if (k == "MemAvailable:") {
            return v*1024;
        }
    }
    return 0;
#endif
}

uint64_t fn_total_ram() {
#if defined(_WIN32)
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    return GlobalMemoryStatusEx(&ms) ? ms.ullTotalPhys : 0;
#else
    const long n = sysconf(_SC_PHYS_PAGES);
    const long p = sysconf(_SC_PAGESIZE);
    return n > 0 && p > 0 ? (uint64_t) n*(uint64_t) p : 0;
#endif
}

// commit limit - commit charge (bytes); 0 when unknown
uint64_t fn_commit_room() {
#if defined(_WIN32)
    PERFORMANCE_INFORMATION pi;
    pi.cb = sizeof(pi);
    if (!GetPerformanceInfo(&pi, sizeof(pi))) {
        return 0;
    }
    return pi.CommitLimit > pi.CommitTotal ? (uint64_t) (pi.CommitLimit - pi.CommitTotal)*(uint64_t) pi.PageSize : 0;
#else
    return fn_avail_ram();
#endif
}

// the share of [p, p + n) in this process's working set, one page sampled per 2 MiB (at most 4096 samples)
double fn_resident_share(const uint8_t * p, size_t n) {
    if (n == 0) {
        return 1.0;
    }
    const size_t page = fn_page_size();
    size_t step = (size_t) 2 << 20;
    while (n/step > 4096) {
        step *= 2;
    }
#if defined(_WIN32)
    std::vector<PSAPI_WORKING_SET_EX_INFORMATION> w;
    for (size_t o = 0; o < n; o += step) {
        PSAPI_WORKING_SET_EX_INFORMATION x = {};
        x.VirtualAddress = (PVOID) (p + o/page*page);
        w.push_back(x);
    }
    if (w.empty() || !QueryWorkingSetEx(GetCurrentProcess(), w.data(), (DWORD) (w.size()*sizeof(w[0])))) {
        return 0.0;
    }
    size_t valid = 0;
    for (const auto & x : w) {
        valid += x.VirtualAttributes.Valid ? 1 : 0;
    }
    return (double) valid/(double) w.size();
#elif !defined(__linux__)
    (void) p; (void) page;
    return 0.0;
#else
    size_t valid = 0;
    size_t total = 0;
    for (size_t o = 0; o < n; o += step) {
        unsigned char v = 0;
        void * a = (void *) ((uintptr_t) (p + o)/page*page);
        if (mincore(a, page, &v) == 0) {
            valid += (v & 1) ? 1 : 0;
        }
        total++;
    }
    return total ? (double) valid/(double) total : 0.0;
#endif
}

// [lo, hi) lies in one mapped view of a file
bool fn_one_view(const uint8_t * lo, const uint8_t * hi) {
#if defined(_WIN32)
    if (hi <= lo) {
        return false;
    }
    MEMORY_BASIC_INFORMATION a = {};
    MEMORY_BASIC_INFORMATION b = {};
    if (!VirtualQuery(lo, &a, sizeof(a)) || !VirtualQuery(hi - 1, &b, sizeof(b))) {
        return false;
    }
    return a.Type == MEM_MAPPED && b.Type == MEM_MAPPED && a.AllocationBase == b.AllocationBase;
#else
    (void) lo; (void) hi;
    return false;
#endif
}

// asks the OS to read [p, p + n) (asynchronous); a no-op elsewhere
void fn_prefetch(const uint8_t * p, size_t n) {
#if defined(_WIN32)
    using prefetch_t = BOOL (WINAPI *)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
    static const prefetch_t pf = (prefetch_t) (void *) GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "PrefetchVirtualMemory");
    if (pf != nullptr && n > 0) {
        WIN32_MEMORY_RANGE_ENTRY e = { (PVOID) p, (SIZE_T) n };
        pf(GetCurrentProcess(), 1, &e, 0);
    }
#else
    (void) p; (void) n;
#endif
}

void * fn_commit_alloc(size_t n) {
#if defined(_WIN32)
    return VirtualAlloc(nullptr, n, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void * p = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? nullptr : p;
#endif
}

void fn_commit_free(void * p, size_t n) {
    if (p == nullptr) {
        return;
    }
#if defined(_WIN32)
    (void) n;
    VirtualFree(p, 0, MEM_RELEASE);
#else
    munmap(p, n);
#endif
}

// ---------------------------------------------------------------------------------------------------------------------
// DXGI: the adapter by LUID (plan P6.2)

using fn_luid_fn_t  = bool (*)(ggml_backend_dev_t, uint64_t *);
using fn_reg_fn_t   = bool (*)(void *, size_t);
using fn_unreg_fn_t = void (*)(void *);
using fn_devptr_t   = void * (*)(const void *); // R-P14-5 (P7): cudaHostGetDevicePointer of a registered address
using fn_regmap_t   = bool (*)(void *, size_t, void **); // R-P14-5 (P7): register Portable | ReadOnly | Mapped + device pointer
using fn_oomfn_t    = bool (*)(size_t, int, void *);   // need, after_failure, ud (P7's ggml_cuda_fn_l22_oom_fn_t)
using fn_set_oom_t  = void (*)(fn_oomfn_t, void *);    // R-P7-2 (P7): ggml-cuda calls it once before failing an allocation

std::atomic<uint64_t> g_default_luid{0};

#if defined(_WIN32)
struct fn_dxgi_entry {
    uint64_t        luid   = 0;
    IDXGIAdapter3 * ad     = nullptr;
    size_t          ded    = 0;
    UINT            vendor = 0;
};

std::mutex                 g_dxgi_mtx;
std::vector<fn_dxgi_entry> g_dxgi;
bool                       g_dxgi_done = false;

void fn_dxgi_enum_locked() {
    if (g_dxgi_done) {
        return;
    }
    g_dxgi_done = true;
    HMODULE lib = LoadLibraryA("dxgi.dll");
    using create_t = HRESULT (WINAPI *)(REFIID, void **);
    create_t create = lib ? (create_t) (void *) GetProcAddress(lib, "CreateDXGIFactory1") : nullptr;
    IDXGIFactory1 * fac = nullptr;
    if (!create || FAILED(create(__uuidof(IDXGIFactory1), (void **) &fac)) || !fac) {
        return;
    }
    IDXGIAdapter1 * a1 = nullptr;
    for (UINT i = 0; fac->EnumAdapters1(i, &a1) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 d = {};
        IDXGIAdapter3 * a3 = nullptr;
        if (SUCCEEDED(a1->GetDesc1(&d)) && SUCCEEDED(a1->QueryInterface(__uuidof(IDXGIAdapter3), (void **) &a3))) {
            fn_dxgi_entry e;
            static_assert(sizeof(d.AdapterLuid) == sizeof(uint64_t), "LUID size");
            memcpy(&e.luid, &d.AdapterLuid, sizeof(uint64_t));
            e.ad     = a3;
            e.ded    = (size_t) d.DedicatedVideoMemory;
            e.vendor = d.VendorId;
            g_dxgi.push_back(e);
        }
        a1->Release();
    }
    fac->Release();
}
#endif

// the LUID of a device (nullptr: CUDA device 0): the CUDA proc (P7), else the NVIDIA adapter whose dedicated memory is
// closest to its total. 0: no such device (no CUDA device visible)
uint64_t fn_dev_luid(ggml_backend_dev_t dev) {
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CUDA");
    if (dev == nullptr && reg != nullptr && ggml_backend_reg_dev_count(reg) > 0) {
        dev = ggml_backend_reg_dev_get(reg, 0);
    }
    if (dev == nullptr) {
        return 0;
    }
    auto fn = reg ? (fn_luid_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_fn_device_luid") : nullptr;
    uint64_t l = 0;
    if (fn != nullptr && dev != nullptr && fn(dev, &l) && l != 0) {
        return l;
    }
#if defined(_WIN32)
    size_t fr = 0;
    size_t tot = 0;
    if (dev != nullptr) {
        ggml_backend_dev_memory(dev, &fr, &tot);
    }
    std::lock_guard<std::mutex> lk(g_dxgi_mtx);
    fn_dxgi_enum_locked();
    size_t best = SIZE_MAX;
    for (const auto & e : g_dxgi) {
        if (e.vendor != 0x10DE) {
            continue;
        }
        const size_t dist = e.ded > tot ? e.ded - tot : tot - e.ded;
        if (dist < best) {
            best = dist;
            l    = e.luid;
        }
    }
#endif
    return l;
}

// ---------------------------------------------------------------------------------------------------------------------
// the pin state (one per process: the model the warm pass called first)

struct fn_group_state {
    bool marked = false; // its experts carry the PIN mark
    int  n_on   = 0;     // its ranges registered now
};

// one layer at plan time: the residency classes and the hot set's view (tv: 1 read, 0 busy, -1 no hot set / layer)
struct fn_layer_view {
    std::vector<uint8_t> cls;
    std::vector<int32_t> slot;
    std::vector<float>   cnt;
    std::vector<uint8_t> vol;
    int                  tv = -1;
};

struct fn_pin_state {
    const llama_model * model = nullptr;
    int  mode = 1;     // 1 untouched, 2 touched
    int  unit = 0;     // 0 ranges, 1 spans, 2 arena
    bool arena = false;

    fn_pin::plan           plan;
    int                    n_take = 0;   // groups planned (in plan order)
    fn_pin::registry       reg;
    std::vector<int>       unit_group;   // mapped mode: unit -> group
    std::vector<std::vector<int>> unit_groups; // arena: groups served by a chunk
    std::vector<fn_group_state> gst;

    fn_pin::budget_cfg cfg;
    uint64_t reserve_need = 0;
    uint64_t l_ready      = 0;    // LOCAL headroom before the first registration
    bool     l_ready_ok   = false;
    uint64_t watch_local  = 0;    // the watcher's LOCAL threshold
    uint64_t arena_max    = 0;
    uint64_t nl_base = 0, l_base = 0;
    bool     base_ok = false;

    fn_reg_fn_t   reg_fn   = nullptr;
    fn_unreg_fn_t unreg_fn = nullptr;
    fn_devptr_t   devptr   = nullptr;
    fn_regmap_t   regmap   = nullptr; // set only with LLAMA_FN_L22_DMASHARE (P14 reads pinned pages by device address)
    fn_set_oom_t  set_oom  = nullptr;
    bool          oom_hooked = false;
    ggml_backend_dev_t dev = nullptr;
    uint64_t luid = 0;

    std::thread             th;
    std::mutex              op_mtx;   // one register / unregister at a time
    std::mutex              wake_mtx;
    std::condition_variable wake_cv;
    std::atomic<bool>       stop{false};
    std::atomic<bool>       frozen{false};   // no more registrations (an unpin happened, or registration ended)
    std::atomic<bool>       registering{false};

    // arena chunks (registry units)
    std::vector<std::pair<void *, size_t>> chunks;

    // counters
    std::atomic<uint64_t> n_refused{0};
    std::atomic<uint64_t> n_oom{0};
    std::atomic<uint64_t> n_oom_skip{0}; // [TAG_FN_L23_PINFIX] OOM hook calls an unpin could not help
    std::atomic<uint64_t> n_valve{0};
    std::atomic<uint64_t> n_repin{0};    // [TAG_FN_L23_PINFIX] re-pin rounds
    std::atomic<uint64_t> n_watch{0};
    std::atomic<uint64_t> unpinned{0};
    double   reg_secs = 0.0;
    int      stop_rule = fn_pin::RULE_OK;
};

std::mutex                  g_pin_mtx;
std::atomic<fn_pin_state *> g_pin{nullptr};

// llama_fn_pin_phase: 0 not started, 1 planning / registering, 2 registration ended, 3 no pins in this run
std::atomic<int> g_pin_phase{0};

// R-REV-2: the headroom a pre-check keeps on top of an allocation: 512 MiB, but never more than the watcher threshold
// (the pin budget may leave less than 512 MiB at ready; an allocation that runs today must not unpin for good)
constexpr uint64_t FN_PRECHECK_MARGIN = (uint64_t) 512 << 20;

uint64_t fn_precheck_margin(const fn_pin_state * st) {
    return st != nullptr ? std::min<uint64_t>(FN_PRECHECK_MARGIN, st->watch_local) : FN_PRECHECK_MARGIN;
}

// decode activity (llama_fn_pin_decode_mark) and the stream's queries while registering
std::atomic<int>     g_dec_running{0};
std::atomic<int64_t> g_dec_end_us{0};
std::atomic<int64_t> g_query_us{0};
std::atomic<bool>    g_query_stamp{false};

// llama_fn_pin_note_need: the largest value per kind (llama_fn_pin_need)
constexpr int         FN_NEED_N = 5;
std::atomic<uint64_t> g_need[FN_NEED_N];

void fn_log(const char * fmt, ...) {
    char text[1536];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(text, sizeof(text), fmt, ap);
    va_end(ap);
    llama_fn_l22_log("pin", "%s", text);
}

uint64_t fn_mib_switch(const llama_model & m, const char * name, int def) {
    return (uint64_t) std::max(0, llama_fn_l3_int(m, name, def))*FN_MIB;
}

// reserve_need (plan P6.4): the device bytes a later allocation may add beyond what the hot set gives back for it
uint64_t fn_reserve_need(const llama_model & m) {
    if (const char * v = llama_fn_env(m, "LLAMA_FN_L22_PIN_RESERVE_MIB")) {
        if (v[0]) {
            return (uint64_t) std::max(0, atoi(v))*FN_MIB;
        }
    }
    const uint64_t full   = g_need[LLAMA_FN_PIN_NEED_FULL_EXTRA].load();
    const uint64_t borrow = g_need[LLAMA_FN_PIN_NEED_BORROW].load();
    const uint64_t vision = g_need[LLAMA_FN_PIN_NEED_VISION_LEND].load();
    const uint64_t kv     = g_need[LLAMA_FN_PIN_NEED_KV_GROW].load();
    const uint64_t inst   = g_need[LLAMA_FN_PIN_NEED_GRAPH_INST].load();
    return full + borrow + vision + (kv ? kv : 512*FN_MIB) + 64*(inst ? inst : 2*FN_MIB) + 512*FN_MIB;
}

// the budget inputs now
fn_pin::budget_in fn_read_budget(fn_pin_state * st) {
    fn_pin::budget_in b;
#if defined(_WIN32)
    b.have_nl = llama_fn_dxgi_query(st->luid, 1, &b.nl_budget, &b.nl_usage);
    b.have_l  = llama_fn_dxgi_query(st->luid, 0, &b.l_budget, &b.l_usage);
#else
    // Linux (plan P6.4): RAM/2 - 8 GiB for the registered bytes; LOCAL from the device's free memory
    const uint64_t tot = fn_total_ram();
    b.have_nl   = true;
    b.nl_budget = tot/2 > 8*FN_GIB ? tot/2 - 8*FN_GIB : 0;
    b.nl_usage  = st->reg.active_bytes();
    if (st->dev != nullptr) {
        size_t fr = 0;
        size_t to = 0;
        ggml_backend_dev_memory(st->dev, &fr, &to);
        b.have_l   = to > 0;
        b.l_budget = to;
        b.l_usage  = to - fr;
    }
#endif
    b.avail  = fn_avail_ram();
    b.pinned = st->reg.active_bytes();
    return b;
}

int64_t fn_local_headroom(fn_pin_state * st, bool * ok) {
    uint64_t bud = 0;
    uint64_t use = 0;
    *ok = false;
#if defined(_WIN32)
    if (!llama_fn_dxgi_query(st->luid, 0, &bud, &use)) {
        return 0;
    }
#else
    if (st->dev == nullptr) {
        return 0;
    }
    size_t fr = 0;
    size_t to = 0;
    ggml_backend_dev_memory(st->dev, &fr, &to);
    bud = to;
    use = to - fr;
#endif
    *ok = true;
    return (int64_t) bud - (int64_t) use;
}

// the experts of group g: PIN on / off (mapped mode only; the model must still be alive)
void fn_mark_group(fn_pin_state * st, int g, bool on) {
    if (g < 0 || g >= (int) st->gst.size() || st->gst[g].marked == on) {
        return;
    }
    const fn_pin::group & G = st->plan.groups[g];
    for (int e = G.e0; e < G.e1; ++e) {
        llama_fn_res_mark_pin(st->model, G.il, e, on);
    }
    st->gst[g].marked = on;
}

// the unregister of one unit (called by the registry after its drain): the device registration goes, an arena chunk is
// freed, the experts of its groups lose the mark. alive = false: the model is going away (no mark changes)
void fn_unreg_unit(fn_pin_state * st, const fn_pin::reg_unit & U, bool alive) {
    if (st->unreg_fn != nullptr && U.p != nullptr) {
        st->unreg_fn(U.p);
    }
    const int u = U.id;
    if (st->arena) {
        fn_commit_free(U.p, U.n);
        if (alive && u >= 0 && u < (int) st->unit_groups.size()) {
            for (int g : st->unit_groups[u]) {
                st->gst[g].n_on = 0;
            }
        }
        return;
    }
    if (u >= 0 && u < (int) st->unit_group.size()) {
        const int g = st->unit_group[u];
        if (g >= 0) {
            st->gst[g].n_on = std::max(0, st->gst[g].n_on - 1);
            if (alive) {
                fn_mark_group(st, g, false);
            }
        }
    }
}

// the contexts of the model are all gone: the server is shutting down (never dereferences the model)
bool fn_model_gone(fn_pin_state * st) {
    return llama_fn_ctx_list(*st->model).empty();
}

// unpin the last registered units until done() (checked before each one); timeout per busy unit. Returns bytes freed
uint64_t fn_unpin(fn_pin_state * st, const std::function<bool(uint64_t freed)> & done, int64_t timeout_us, bool alive, int * n_units) {
    uint64_t freed = 0;
    int n = st->reg.unpin_lifo([&]() { return done(freed); }, timeout_us,
            [&](const fn_pin::reg_unit & U) {
                fn_unreg_unit(st, U, alive);
                freed += U.n;
            }, nullptr);
    st->unpinned.fetch_add(freed);
    if (n_units) {
        *n_units = n;
    }
    return freed;
}

// waits while a decode runs or ended < 50 ms ago, while a prompt scope hushes the background readers, while the warm
// pass waits for answer steps, and while the stream queried < 200 ms ago. false: stop
// [TAG_FN_L23_PINFIX] R-L23-7 (hard rule): pins register only while the server is idle, never while a prompt or an answer runs:
// no decode for LLAMA_FN_L23_PIN_IDLE_MS (1500; an answer's steps are ms apart). FAULT 15:20:50: 51 GiB registered during an answer
std::atomic<int64_t> g_pin_idle_us{1500000};
// [TAG_FN_L23_PINFIX] R-L23-8: the server marks requests (llama_fn_pin_server_busy): a request handler, a slot from launch to
// release, a keepctx copy; every mark also stamps the time, so a gap shorter than the idle time counts as busy
std::atomic<int>     g_srv_busy{0};
std::atomic<int64_t> g_srv_us{0};
// [TAG_FN_L23_PINFIX] R-L23-8: the first refused registration stops pinning for the whole process (no re-pin until a restart)
std::atomic<bool>    g_pin_refused_ever{false};

bool fn_wait_quiet(fn_pin_state * st) {
    size_t mark = 0;
    while (!st->stop.load(std::memory_order_relaxed)) {
        const int64_t now = fn_now_us();
        const bool busy = g_dec_running.load(std::memory_order_relaxed) > 0 ||
                          now - g_dec_end_us.load(std::memory_order_relaxed) < g_pin_idle_us.load(std::memory_order_relaxed) ||
                          g_srv_busy.load(std::memory_order_relaxed) > 0 ||
                          now - g_srv_us.load(std::memory_order_relaxed) < g_pin_idle_us.load(std::memory_order_relaxed) ||
                          now - g_query_us.load(std::memory_order_relaxed) < 200000 ||
                          llama_fn_hushed() || llama_fn_warm_state(&mark) == 2;
        if (!busy) {
            return true;
        }
        fn_sleep_ms(2);
    }
    return false;
}

// reads one byte per page of [p, p + n) after a prefetch; checks stop / quiet every 2 MiB
bool fn_touch(fn_pin_state * st, const uint8_t * p, size_t n) {
    const size_t page = fn_page_size();
    fn_prefetch(p, n);
    for (size_t o = 0; o < n; o += page) {
        if ((o & (((size_t) 2 << 20) - 1)) == 0 && !fn_wait_quiet(st)) {
            return false;
        }
        (void) *(volatile const uint8_t *) (p + o);
    }
    return true;
}

// [TAG_FN_L23_PINFIX] R-L23-1: the inputs of fn_pin::oom_help_bytes now (NON_LOCAL and commit read only when there is
// no LOCAL charge; commit counts only for the arena, whose chunks are private memory)
fn_pin::oom_in fn_oom_in(fn_pin_state * st, bool lok, int64_t head) {
    fn_pin::oom_in oi;
    oi.have_l  = lok;
    oi.l_head  = head;
    oi.l_ratio = st->cfg.l_ratio;
    if (lok && oi.l_ratio < fn_pin::OOM_RATIO_MIN) {
        const fn_pin::budget_in b = fn_read_budget(st);
        oi.have_nl     = b.have_nl;
        oi.nl_budget   = b.nl_budget;
        oi.nl_usage    = b.nl_usage;
        oi.have_commit = st->arena;
        oi.commit_room = st->arena ? fn_commit_room() : 0;
    }
    return oi;
}

// [TAG_FN_L23_PINFIX] R-L23-2: room came back: units not registered now are registered, the most streamed first (the
// plan's order), only those whose pages are all in the working set already (a pin never reads the disk), each only while
// the available RAM minus what it would bring in stays >= keep and the registration's rules allow it (NON_LOCAL / LOCAL
// budgets, the cap). Mapped mode only. Returns the units registered
bool fn_register(fn_pin_state * st, void * p, size_t n, uint64_t * dev);

// [TAG_FN_L23_PINFIX] R-L23-4: a range is pinned only when this share of its pages (one sample per 2 MiB) is in the working
// set: pins lock what the warm pass or the prompts brought in, never more (l23p: with untouched registrations ~10 GiB of the
// streamed set went to the SSD tier)
constexpr double FN_RESIDENT_MIN = 0.98;

int fn_repin(fn_pin_state * st, uint64_t keep, uint64_t * bytes) {
    int n = 0;
    uint64_t b = 0;
    uint64_t pending = 0;
    if (g_pin_refused_ever.load()) { // [TAG_FN_L23_PINFIX] R-L23-8: no re-pin after a refusal until a restart
        if (bytes) {
            *bytes = 0;
        }
        return 0;
    }
    if (!st->arena && st->reg_fn != nullptr) {
        for (int u = 0; u < st->reg.n_units() && !st->stop.load(); ++u) {
            const fn_pin::reg_unit & U = st->reg.unit_at(u);
            if (U.state.load() != fn_pin::ST_GONE || U.p == nullptr || U.n == 0) {
                continue;
            }
            if (!fn_wait_quiet(st) || fn_model_gone(st)) {
                break;
            }
            std::lock_guard<std::mutex> op(st->op_mtx);
            if (U.state.load() != fn_pin::ST_GONE) {
                continue;
            }
            const double   res = fn_resident_share((const uint8_t *) U.p, U.n);
            if (res < FN_RESIDENT_MIN) {
                continue; // not resident yet: no pin brings pages in
            }
            const uint64_t add = (uint64_t) ((1.0 - res)*(double) U.n);
            fn_pin::budget_in bi = fn_read_budget(st);
            if (bi.avail < pending + add + keep || fn_pin::check(bi, st->cfg, U.n, 0) != fn_pin::RULE_OK) {
                break;
            }
            uint64_t dev = 0;
            if (!fn_register(st, U.p, U.n, &dev)) {
                // [TAG_FN_L23_PINFIX] R-L23-7: a refusal stops pinning; later re-pins stay below it
                st->n_refused.fetch_add(1);
                g_pin_refused_ever.store(true);
                st->cfg.max_pin = st->reg.active_bytes();
                break;
            }
            st->reg.set_unit_dev(u, dev);
            st->reg.activate(u);
            pending += add;
            const int g = u < (int) st->unit_group.size() ? st->unit_group[u] : -1;
            if (g >= 0) {
                st->gst[g].n_on++;
                if (st->gst[g].n_on >= st->plan.groups[g].n_rng) {
                    fn_mark_group(st, g, true);
                }
            }
            n++;
            b += U.n;
        }
    }
    if (bytes) {
        *bytes = b;
    }
    return n;
}

// [TAG_FN_L23_PINFIX] R-L23-2: the resident bytes of the planned (streamed host expert) ranges that are not registered
// now: what the memory manager can trim from the model for other programs (the status line). One page per 4 MiB,
// one QueryWorkingSetEx call (~16K samples for 64 GiB). Mapped mode: unit index == plan range index; the arena pins its
// own copies, so there every mapped range counts
uint64_t fn_trimmable(fn_pin_state * st) {
    const size_t step = (size_t) 4 << 20;
    std::vector<const uint8_t *> at;
    std::vector<uint32_t>        len;
    for (int ri = 0; ri < (int) st->plan.ranges.size(); ++ri) {
        if (!st->arena && ri < st->reg.n_units() &&
            st->reg.unit_at(ri).state.load(std::memory_order_relaxed) == fn_pin::ST_ACTIVE) {
            continue;
        }
        const fn_pin::range & r = st->plan.ranges[ri];
        for (size_t o = 0; o < r.n; o += step) {
            at.push_back(r.p + o);
            len.push_back((uint32_t) std::min(step, r.n - o));
        }
    }
    if (at.empty()) {
        return 0;
    }
    uint64_t res = 0;
#if defined(_WIN32)
    const size_t page = fn_page_size();
    std::vector<PSAPI_WORKING_SET_EX_INFORMATION> w(at.size());
    for (size_t i = 0; i < at.size(); ++i) {
        w[i] = {};
        w[i].VirtualAddress = (PVOID) ((uintptr_t) at[i]/page*page);
    }
    if (!QueryWorkingSetEx(GetCurrentProcess(), w.data(), (DWORD) (w.size()*sizeof(w[0])))) {
        return 0; // unknown: nothing counts (the valve stays as strict as the available RAM alone)
    }
    for (size_t i = 0; i < w.size(); ++i) {
        res += w[i].VirtualAttributes.Valid ? len[i] : 0;
    }
#else
    for (size_t i = 0; i < at.size(); ++i) {
        res += (uint64_t) (fn_resident_share(at[i], 1)*(double) len[i]);
    }
#endif
    return res;
}

// [TAG_FN_L23_PINFIX] every pin goes (an unpin only unlocks); the process runs as without pins until the gate pins again.
// op_mtx held by the caller. Returns the bytes unpinned
uint64_t fn_drop_all(fn_pin_state * st, int * n_units) {
    int n = 0;
    uint64_t b = 0;
    for (int k = 0; k < 8 && st->reg.any_active(); ++k) {
        int nk = 0;
        b += fn_unpin(st, [](uint64_t) { return false; }, 200000, true, &nk);
        n += nk;
    }
    if (n_units) {
        *n_units = n;
    }
    return b;
}

// [TAG_FN_L23_PINFIX] R-L23-5 / R-L23-6: an allocation FAILED while pins were held (the caller retries). The pins go only when
// they can be the cause (fn_pin::alloc_fail_reason). A plain VRAM failure leaves them (existing handling), except at a site
// whose failed retry aborts (AF_DEV_FATAL: VMM pool growth, graph instantiate): attempt 0 unpins need + 1 GiB, the least
// streamed first, attempt 1 drops all. true: something was unpinned
bool fn_on_alloc_fail(size_t need, const char * site, int kind, int attempt) {
    fn_pin_state * st = g_pin.load(std::memory_order_acquire);
    if (st == nullptr || !st->reg.any_active()) {
        return false;
    }
    std::lock_guard<std::mutex> op(st->op_mtx);
    bool lok = false;
    const int64_t head = fn_local_headroom(st, &lok);
    const fn_pin::budget_in bi = fn_read_budget(st);
    fn_pin::af_in a;
    a.kind      = kind;
    a.l_ratio   = st->cfg.l_ratio;
    a.have_l    = lok;
    a.l_head    = head;
    a.have_nl   = bi.have_nl;
    a.nl_room   = bi.nl_budget > bi.nl_usage ? bi.nl_budget - bi.nl_usage : 0;
    a.nl_margin = st->cfg.nl_margin;
    a.need      = need;
    const int r = fn_pin::alloc_fail_reason(a);
    const uint64_t had = st->reg.active_bytes();
    if (r == fn_pin::AFR_NONE && kind == fn_pin::AF_DEV) {
        const uint64_t k = st->n_oom_skip.fetch_add(1) + 1;
        if (k <= 3 || k % 64 == 0) {
            LLAMA_LOG_WARN("%s: [TAG_FN_L23_PINFIX] an allocation of %.1f MiB failed (%s) with %.2f GiB pinned: the pins cannot be the cause "
                    "(LOCAL headroom %.2f GiB, NON_LOCAL room %.2f GiB >= %.2f): pins kept\n", __func__, need/1048576.0, site, fn_gib(had),
                    head/(double) FN_GIB, fn_gib(a.nl_room), fn_gib(a.nl_margin));
        }
        fn_log("event=alloc_skip need_mib=%.1f site=%s pinned_gib=%.2f headroom_gib=%.2f nl_room_gib=%.2f margin_gib=%.2f n=%d", need/1048576.0,
                site, fn_gib(had), head/(double) FN_GIB, fn_gib(a.nl_room), fn_gib(a.nl_margin), (int) k);
        return false;
    }
    st->n_oom.fetch_add(1);
    int n = 0;
    uint64_t b = 0;
    const bool partial = r == fn_pin::AFR_NONE && attempt == 0;
    if (partial) {
        const uint64_t cap = (uint64_t) need + FN_GIB;
        b = fn_unpin(st, [&](uint64_t f) { return f >= cap; }, 200000, true, &n);
    } else {
        b = fn_drop_all(st, &n);
    }
    LLAMA_LOG_WARN("%s: [TAG_FN_L23_PINFIX] an allocation of %.1f MiB failed (%s, attempt %d, cause %s) with %.2f GiB pinned: %s (%d range(s), "
            "%.2f GiB), retry\n", __func__, need/1048576.0, site, attempt, fn_pin::af_reason_name(r), fn_gib(had),
            partial ? "some pins dropped" : "all pins dropped", n, fn_gib(b));
    fn_log("event=alloc_fail need_mib=%.1f site=%s kind=%d attempt=%d reason=%s mode=%s had_gib=%.2f units=%d unpinned_gib=%.2f left_gib=%.2f "
            "headroom_gib=%.2f nl_room_gib=%.2f margin_gib=%.2f", need/1048576.0, site, kind, attempt, fn_pin::af_reason_name(r),
            partial ? "partial" : "all", fn_gib(had), n, fn_gib(b), fn_gib(st->reg.active_bytes()), head/(double) FN_GIB, fn_gib(a.nl_room),
            fn_gib(a.nl_margin));
    return b > 0;
}

// ggml-cuda's allocation hook (R-P7-2), before a graph instantiate, a VMM pool growth or a device buffer: after_failure
// 0 = pre-check, unpin only when LOCAL headroom < need + the pre-check margin (on_oom unpins for good); 1 = the
// allocation failed, unpin once
bool fn_cuda_oom(size_t need, int after_failure, void * ud) {
    (void) ud;
    if (after_failure == 0) {
        fn_pin_state * st = g_pin.load(std::memory_order_acquire);
        if (st == nullptr || !st->reg.any_active()) {
            return false;
        }
        bool ok = false;
        const int64_t head = fn_local_headroom(st, &ok);
        if (!ok || head >= (int64_t) need + (int64_t) fn_precheck_margin(st)) {
            return false;
        }
        return llama_fn_pin_on_oom(need);
    }
    // [TAG_FN_L23_PINFIX] R-L23-6: 1 = device, 2 = pinned host, 3 / 4 = device whose failed retry aborts (attempt 0 / 1)
    const int kind    = after_failure == 2 ? fn_pin::AF_HOST : after_failure == 1 ? fn_pin::AF_DEV : fn_pin::AF_DEV_FATAL;
    const int attempt = after_failure == 3 ? 0 : after_failure == 4 ? 1 : 0;
    return fn_on_alloc_fail(need, after_failure == 2 ? "ggml-cuda-host" : "ggml-cuda", kind, attempt);
}

// one registration; *dev = the device address when known (0 else)
bool fn_register(fn_pin_state * st, void * p, size_t n, uint64_t * dev) {
    *dev = 0;
    void * d = nullptr;
    const bool ok = st->regmap != nullptr ? st->regmap(p, n, &d) : st->reg_fn(p, n);
    if (!ok) {
        return false;
    }
    if (d == nullptr && st->devptr != nullptr) {
        d = st->devptr(p);
    }
    *dev = (uint64_t) (uintptr_t) d;
    // pins are active from here: ggml-cuda may ask for an unpin before it fails an allocation
    if (!st->oom_hooked && st->set_oom != nullptr) {
        st->set_oom(fn_cuda_oom, nullptr);
        st->oom_hooked = true;
    }
    return true;
}

void fn_unhook_oom(fn_pin_state * st) {
    if (st->oom_hooked && st->set_oom != nullptr) {
        st->set_oom(nullptr, nullptr);
        st->oom_hooked = false;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// the arena (spec 9)

constexpr size_t FN_ARENA_CHUNK = (size_t) 1 << 30;

// rebuilds the registry for the arena: the planned groups in plan order, packed into chunks of 1 GiB (whole groups,
// pieces of a range larger than a chunk split at page edges). Returns the bytes planned. No unit is active.
uint64_t fn_arena_plan(fn_pin_state * st) {
    const size_t page = fn_page_size();
    struct piece {
        const uint8_t * p;
        size_t          n;
        int             group;
        int             src;
    };
    std::vector<std::vector<piece>> ch;   // per chunk
    std::vector<size_t>             used;
    uint64_t total = 0;
    for (int g = 0; g < st->n_take; ++g) {
        const fn_pin::group & G = st->plan.groups[g];
        std::vector<piece> ps;
        size_t gbytes = 0;
        for (int k = 0; k < G.n_rng; ++k) {
            const fn_pin::range & r = st->plan.ranges[G.first + k];
            for (size_t o = 0; o < r.n; o += FN_ARENA_CHUNK) {
                const size_t n = std::min(FN_ARENA_CHUNK, r.n - o);
                ps.push_back({ r.p + o, n, g, G.first + k });
                gbytes += n;
            }
        }
        if (total + gbytes > st->arena_max) {
            break;
        }
        for (const piece & x : ps) {
            if (ch.empty() || used.back() + x.n > FN_ARENA_CHUNK) {
                ch.emplace_back();
                used.push_back(0);
            }
            ch.back().push_back(x);
            used.back() += (x.n + page - 1)/page*page;
        }
        total += gbytes;
    }
    std::vector<fn_pin::registry::in_range> rs;
    st->unit_groups.assign(ch.size(), {});
    st->chunks.assign(ch.size(), { nullptr, 0 });
    for (size_t u = 0; u < ch.size(); ++u) {
        size_t off = 0;
        for (const piece & x : ch[u]) {
            fn_pin::registry::in_range r;
            r.p    = x.p;
            r.n    = x.n;
            r.copy = (const uint8_t *) (uintptr_t) off; // an offset until the chunk exists (fn_arena_run fixes it)
            r.unit = (int) u;
            r.src  = x.src;
            rs.push_back(r);
            off += (x.n + page - 1)/page*page;
            if (std::find(st->unit_groups[u].begin(), st->unit_groups[u].end(), x.group) == st->unit_groups[u].end()) {
                st->unit_groups[u].push_back(x.group);
            }
        }
        st->chunks[u].second = off;
    }
    st->reg.build(rs, (int) ch.size());
    return total;
}

// allocates, registers and fills the chunks in order; stops at the first rule, refusal or failed fill
void fn_arena_run(fn_pin_state * st) {
    const size_t page = fn_page_size();
    const int n_units = st->reg.n_units();
    int n_ok = 0;
    uint64_t bytes_ok = 0;
    const int64_t t0 = fn_now_us();
    for (int u = 0; u < n_units && !st->stop.load() && !st->frozen.load(); ++u) {
        const size_t sz = st->chunks[u].second;
        if (sz == 0) {
            st->reg.drop(u);
            continue;
        }
        if (!fn_wait_quiet(st) || fn_model_gone(st)) {
            break;
        }
        void * p = nullptr;
        {
            std::lock_guard<std::mutex> op(st->op_mtx);
            if (st->frozen.load()) {
                break;
            }
            fn_pin::budget_cfg cfg = st->cfg;
            const fn_pin::budget_in b = fn_read_budget(st);
            const int rule = fn_pin::check(b, cfg, sz, sz);
            const uint64_t room = fn_commit_room();
            if (rule != fn_pin::RULE_OK || room < sz + 8*FN_GIB) {
                st->stop_rule = rule != fn_pin::RULE_OK ? rule : fn_pin::RULE_RAM;
                fn_log("event=arena_stop chunk=%d rule=%s commit_room_gib=%.2f", u, rule != fn_pin::RULE_OK ? fn_pin::rule_name(rule) : "commit",
                        fn_gib(room));
                break;
            }
            p = fn_commit_alloc(sz);
            if (p == nullptr) {
                fn_log("event=arena_stop chunk=%d reason=alloc", u);
                break;
            }
            // registered at once, before any byte is written (register before touch)
            uint64_t dev = 0;
            if (!fn_register(st, p, sz, &dev)) {
                st->n_refused.fetch_add(1);
                fn_commit_free(p, sz);
                fn_log("event=arena_stop chunk=%d reason=refused", u);
                break;
            }
            st->chunks[u].first = p;
            st->reg.set_unit(u, p, sz);
            st->reg.set_unit_dev(u, dev);
        }
        // the copies (outside the lock: an OOM hook meanwhile freezes registration and this chunk goes below):
        // direct I/O, else the mapped bytes
        const fn_pin::reg_unit & U = st->reg.unit_at(u);
        std::vector<llama_fn_dio_req> reqs;
        for (int i : U.ranges) {
            const fn_pin::reg_range & r = st->reg.range_at(i);
            reqs.push_back({ r.p, r.n, (uint8_t *) p + (size_t) (uintptr_t) r.copy });
        }
        bool filled = false;
        const int ticket = llama_fn_dio_submit(st->model, reqs.data(), (int) reqs.size());
        if (ticket >= 0) {
            filled = llama_fn_dio_wait(ticket);
        }
        if (!filled) {
            for (const auto & q : reqs) {
                memcpy(q.dst, q.src, q.len);
            }
        }
        // the same bytes as the mapping: the first and last 4 KiB of every piece
        bool same = true;
        for (const auto & q : reqs) {
            const size_t k = std::min<size_t>(q.len, 4096);
            same = same && memcmp(q.dst, q.src, k) == 0 && memcmp((const uint8_t *) q.dst + q.len - k, (const uint8_t *) q.src + q.len - k, k) == 0;
        }
        std::lock_guard<std::mutex> op(st->op_mtx);
        if (!same || st->frozen.load() || st->stop.load()) {
            st->unreg_fn(p);
            fn_commit_free(p, sz);
            st->chunks[u].first = nullptr;
            st->reg.set_unit(u, nullptr, 0);
            st->reg.drop(u);
            if (!same) {
                LLAMA_LOG_ERROR("%s: [TAG_FN_L22_PIN] arena chunk %d differs from the mapped model: the arena stops\n", __func__, u);
            }
            break;
        }
        (void) page;
        st->reg.fix_copies(u, (const uint8_t *) p);
        st->reg.activate(u);
        n_ok++;
        bytes_ok += sz;
        if (n_ok == 1 || (bytes_ok & (2*FN_GIB - 1)) < sz) {
            const fn_pin::budget_in b2 = fn_read_budget(st);
            fn_log("event=arena_step chunks=%d pinned_gib=%.2f dio=%d nl_use_gib=%.2f/%.2f l_use_gib=%.2f/%.2f avail_gib=%.2f",
                    n_ok, fn_gib(bytes_ok), ticket >= 0 && filled ? 1 : 0, fn_gib(b2.nl_usage), fn_gib(b2.nl_budget),
                    fn_gib(b2.l_usage), fn_gib(b2.l_budget), fn_gib(b2.avail));
        }
    }
    for (int u = 0; u < n_units; ++u) {
        if (st->reg.unit_at(u).state.load() == fn_pin::ST_PLANNED) {
            st->reg.drop(u);
        }
    }
    st->reg_secs = (fn_now_us() - t0)/1e6;
    LLAMA_LOG_INFO("%s: [TAG_FN_L22_PIN] arena: %d chunk(s), %.2f GiB of experts registered in %.1f s\n", __func__, n_ok,
            fn_gib(bytes_ok), st->reg_secs);
}

// ---------------------------------------------------------------------------------------------------------------------
// mapped ranges (spec 3, 5)

// registers the planned groups range by range; true when it ended with a refusal at the very first range (arena)
bool fn_ranges_run(fn_pin_state * st) {
    int n_ok = 0;
    uint64_t bytes_ok = 0;
    uint64_t next_log = 2*FN_GIB;
    const int64_t t0 = fn_now_us();
    bool first_refused = false;
    int      n_skip = 0; // [TAG_FN_L23_PINFIX] ranges not resident at registration
    uint64_t b_skip = 0;
    for (int g = 0; g < st->n_take && !st->stop.load() && !st->frozen.load(); ++g) {
        const fn_pin::group & G = st->plan.groups[g];
        bool ok_group = true;
        for (int k = 0; k < G.n_rng && ok_group; ++k) {
            const int ri = G.first + k; // mapped mode: unit index == plan range index
            const fn_pin::range & r = st->plan.ranges[ri];
            // the model's contexts all gone: the mapping may close soon, nothing more is read or registered
            if (!fn_wait_quiet(st) || fn_model_gone(st)) {
                ok_group = false;
                st->frozen.store(true);
                break;
            }
            if (st->mode == 2 && !fn_touch(st, r.p, r.n)) {
                ok_group = false;
                break;
            }
            std::lock_guard<std::mutex> op(st->op_mtx);
            if (st->frozen.load() || st->stop.load()) {
                ok_group = false;
                break;
            }
            fn_pin::budget_cfg cfg = st->cfg;
            fn_pin::budget_in b = fn_read_budget(st);
            const double res = st->mode == 2 ? 1.0 : fn_resident_share(r.p, r.n);
            if (res < FN_RESIDENT_MIN) {
                n_skip++; // [TAG_FN_L23_PINFIX] R-L23-4: not resident: no pin (the re-pin pass may take it later)
                b_skip += r.n;
                continue;
            }
            const uint64_t add_new = (uint64_t) ((1.0 - res)*(double) r.n);
            int rule = fn_pin::check(b, cfg, r.n, add_new);
            if (rule != fn_pin::RULE_OK) {
                st->stop_rule = rule;
                fn_log("event=stop rule=%s range=%d pinned_gib=%.2f nl_gib=%.2f/%.2f l_gib=%.2f/%.2f l_keep_gib=%.2f avail_gib=%.2f",
                        fn_pin::rule_name(rule), ri, fn_gib(b.pinned), fn_gib(b.nl_usage), fn_gib(b.nl_budget), fn_gib(b.l_usage),
                        fn_gib(b.l_budget), fn_gib(cfg.l_keep), fn_gib(b.avail));
                ok_group = false;
                st->frozen.store(true);
                break;
            }
            const int64_t tr = fn_now_us();
            uint64_t dev = 0;
            const bool done = fn_register(st, (void *) r.p, r.n, &dev);
            const int64_t dt = fn_now_us() - tr;
            if (!done) {
                st->n_refused.fetch_add(1);
                g_pin_refused_ever.store(true); // [TAG_FN_L23_PINFIX] R-L23-8
                st->reg.drop(ri);
                first_refused = n_ok == 0;
                LLAMA_LOG_WARN("%s: [TAG_FN_L22_PIN] range %d (%.1f MiB, layer %d part %d) refused by the driver after %.1f ms: "
                        "registration stops\n", __func__, ri, r.n/1048576.0, r.il, r.part, dt/1e3);
                fn_log("event=refused range=%d mib=%.1f first=%d", ri, r.n/1048576.0, first_refused ? 1 : 0);
                ok_group = false;
                st->frozen.store(true);
                break;
            }
            st->reg.set_unit_dev(ri, dev);
            st->reg.activate(ri);
            st->gst[g].n_on++;
            n_ok++;
            bytes_ok += r.n;
            // the measured charge per registered byte, then the rules again with the real numbers
            const fn_pin::budget_in b2 = fn_read_budget(st);
            if (st->base_ok && b2.pinned >= 256*FN_MIB) {
                if (b2.have_nl) {
                    st->cfg.nl_ratio = std::max(0.0, std::min(2.0, ((double) b2.nl_usage - (double) st->nl_base)/(double) b2.pinned));
                }
                if (b2.have_l) {
                    st->cfg.l_ratio = std::max(0.0, std::min(2.0, ((double) b2.l_usage - (double) st->l_base)/(double) b2.pinned));
                }
            }
            rule = fn_pin::check(b2, st->cfg, 0, 0);
            if (rule != fn_pin::RULE_OK) {
                st->stop_rule = rule;
                st->frozen.store(true);
                if (st->reg.drain(ri, 2000000, [&](const fn_pin::reg_unit & U) { fn_unreg_unit(st, U, true); })) {
                    n_ok--;
                    bytes_ok -= r.n;
                }
                fn_log("event=stop_after rule=%s range=%d nl_gib=%.2f/%.2f l_gib=%.2f/%.2f avail_gib=%.2f", fn_pin::rule_name(rule), ri,
                        fn_gib(b2.nl_usage), fn_gib(b2.nl_budget), fn_gib(b2.l_usage), fn_gib(b2.l_budget), fn_gib(b2.avail));
                ok_group = false;
                break;
            }
            if (bytes_ok >= next_log || n_ok == 1) {
                next_log = (bytes_ok/(2*FN_GIB) + 1)*2*FN_GIB;
                fn_log("event=step ranges=%d pinned_gib=%.2f secs=%.1f last_ms=%.1f nl_gib=%.2f/%.2f l_gib=%.2f/%.2f "
                        "nl_ratio=%.3f l_ratio=%.4f avail_gib=%.2f resident=%.2f", n_ok, fn_gib(bytes_ok), (fn_now_us() - t0)/1e6, dt/1e3,
                        fn_gib(b2.nl_usage), fn_gib(b2.nl_budget), fn_gib(b2.l_usage), fn_gib(b2.l_budget), st->cfg.nl_ratio,
                        st->cfg.l_ratio, fn_gib(b2.avail), res);
            }
        }
        // [TAG_FN_L23_PINFIX] the group's experts carry the PIN mark once all of its ranges are registered (no mark before:
        // the warm pass reads them as without pins)
        if (st->gst[g].n_on >= G.n_rng) {
            std::lock_guard<std::mutex> op(st->op_mtx);
            fn_mark_group(st, g, true);
        }
        (void) ok_group;
    }
    // what did not get registered loses its mark; planned units never tried are dropped
    {
        std::lock_guard<std::mutex> op(st->op_mtx);
        for (int g = 0; g < (int) st->gst.size(); ++g) {
            if (st->gst[g].marked && st->gst[g].n_on < st->plan.groups[g].n_rng) {
                fn_mark_group(st, g, false);
            }
        }
        for (int u = 0; u < st->reg.n_units(); ++u) {
            if (st->reg.unit_at(u).state.load() == fn_pin::ST_PLANNED) {
                st->reg.drop(u);
            }
        }
    }
    // [TAG_FN_L23_PINFIX] R-L23-7: a refusal means the driver is at its limit for locked host memory; its own pinned staging
    // (copies with pageable memory) needs room too: LLAMA_FN_L23_PIN_REFUSE_BACK_MIB (4096) of the least streamed pins go
    if (st->n_refused.load() > 0 && st->reg.any_active()) {
        std::lock_guard<std::mutex> op(st->op_mtx);
        const uint64_t back = fn_mib_switch(*st->model, "LLAMA_FN_L23_PIN_REFUSE_BACK_MIB", 4096);
        int n = 0;
        const uint64_t b = fn_unpin(st, [&](uint64_t f) { return f >= back; }, 200000, true, &n);
        bytes_ok = bytes_ok > b ? bytes_ok - b : 0;
        const uint64_t cap = st->reg.active_bytes();
        if (st->cfg.max_pin == 0 || st->cfg.max_pin > cap) {
            st->cfg.max_pin = cap; // a re-pin stays below the refusal
        }
        LLAMA_LOG_WARN("%s: [TAG_FN_L23_PINFIX] refused by the driver: %d range(s), %.2f GiB unpinned again for the driver's staging\n",
                __func__, n, fn_gib(b));
        fn_log("event=refuse_back units=%d unpinned_gib=%.2f pinned_gib=%.2f", n, fn_gib(b), fn_gib(st->reg.active_bytes()));
    }
    st->reg_secs = (fn_now_us() - t0)/1e6;
    LLAMA_LOG_INFO("%s: [TAG_FN_L22_PIN] %d range(s), %.2f GiB of experts registered in %.1f s (%s, stop: %s, refused %d; %d range(s), "
            "%.2f GiB not resident: not pinned)\n", __func__, n_ok, fn_gib(bytes_ok), st->reg_secs, st->mode == 2 ? "touched" : "resident only",
            fn_pin::rule_name(st->stop_rule), (int) st->n_refused.load(), n_skip, fn_gib(b_skip));
    fn_log("event=registered ranges=%d pinned_gib=%.2f skipped=%d skipped_gib=%.2f secs=%.1f", n_ok, fn_gib(bytes_ok), n_skip, fn_gib(b_skip),
            st->reg_secs);
    return first_refused;
}

// ---------------------------------------------------------------------------------------------------------------------
// the thread: registration, then the watcher and the valve (spec 5, 7, 8)

void fn_thread(fn_pin_state * st) {
#if defined(_WIN32)
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#endif
    llama_fn_l22_log("pin", "event=thread_start mode=%d unit=%d groups=%d", st->mode, st->unit, st->n_take);
    g_query_stamp.store(true);
    st->registering.store(true);
    // [TAG_FN_L23_PINFIX] R-L23-4: mapped pins wait for the warm pass (at most 300 s): it brings the streamed set into the
    // working set, the registration pins what is resident then
    if (st->unit != 2) {
        const int64_t t_w = fn_now_us();
        while (!st->stop.load() && !llama_fn_p5_warm_done(*st->model) && fn_now_us() - t_w < 300000000) {
            fn_sleep_ms(200);
        }
        fn_log("event=warm_wait ms=%.0f done=%d", (fn_now_us() - t_w)/1e3, llama_fn_p5_warm_done(*st->model) ? 1 : 0);
    }
    if (st->unit == 2) {
        fn_arena_run(st);
    } else if (fn_ranges_run(st) && !st->stop.load()) {
        // a refusal at the very first range: the arena (spec 9), the marks off (arena experts keep their mapped copies)
        {
            std::lock_guard<std::mutex> op(st->op_mtx);
            for (int g = 0; g < (int) st->gst.size(); ++g) {
                fn_mark_group(st, g, false);
            }
            st->arena = true;
            st->frozen.store(false);
            const uint64_t planned = fn_arena_plan(st);
            fn_log("event=arena_fallback planned_gib=%.2f chunks=%d", fn_gib(planned), st->reg.n_units());
        }
        fn_arena_run(st);
    }
    st->registering.store(false);
    g_query_stamp.store(false);
    st->frozen.store(true);
    g_pin_phase.store(2, std::memory_order_release);
    {
        const fn_pin::budget_in b = fn_read_budget(st);
        fn_log("event=ready pinned_gib=%.2f units=%d secs=%.1f stop=%s nl_gib=%.2f/%.2f l_gib=%.2f/%.2f avail_gib=%.2f reserve_need_gib=%.2f "
                "watch_local_gib=%.2f", fn_gib(st->reg.active_bytes()), st->reg.n_active_units(), st->reg_secs,
                fn_pin::rule_name(st->stop_rule), fn_gib(b.nl_usage), fn_gib(b.nl_budget), fn_gib(b.l_usage), fn_gib(b.l_budget),
                fn_gib(b.avail), fn_gib(st->reserve_need), fn_gib(st->watch_local));
    }
    // the watcher and the valve, every 2 s
    fn_pin::gate_state gst8; // [TAG_FN_L23_PINFIX] the drop / re-pin gate
    // [TAG_FN_L23_PINFIX] R-L23-6: re-pin at the drop line + LLAMA_FN_L23_PIN_REPIN_GAP_MIB (4096) for 60 s: a re-pin locks resident
    // pages only (no RAM). The old 10 GiB never came back at 7-10 GiB available (l23p B6). LLAMA_FN_L23_PIN_REPIN_MIB > 0 sets it
    uint64_t repin_b = fn_mib_switch(*st->model, "LLAMA_FN_L23_PIN_REPIN_MIB", 0);
    if (repin_b == 0) {
        repin_b = st->cfg.floor + fn_mib_switch(*st->model, "LLAMA_FN_L23_PIN_REPIN_GAP_MIB", 4096);
    }
    bool watch_noted = false;
    int64_t t_status = fn_now_us();
    while (!st->stop.load()) {
        {
            std::unique_lock<std::mutex> lk(st->wake_mtx);
            st->wake_cv.wait_for(lk, std::chrono::seconds(2), [&]() { return st->stop.load(); });
        }
        if (st->stop.load()) {
            break;
        }
        if (!st->reg.any_active() && (st->arena || st->reg.n_units() == 0)) {
            continue; // [TAG_FN_L23_PINFIX] mapped units can come back (re-pin): the valve keeps watching
        }
        if (fn_model_gone(st)) {
            // every context is gone (shutdown): nothing copies any more; unregister before the mapping closes
            std::lock_guard<std::mutex> op(st->op_mtx);
            int n = 0;
            const int64_t t_end = fn_now_us() + 3000000;
            const uint64_t b = fn_unpin(st, [&](uint64_t) { return fn_now_us() > t_end; }, 200000, false, &n);
            if (!st->reg.any_active()) {
                fn_unhook_oom(st);
            }
            fn_log("event=contexts_gone unpinned_gib=%.2f units=%d left=%d", fn_gib(b), n, st->reg.n_active_units());
            break;
        }
        bool lok = false;
        const int64_t head = fn_local_headroom(st, &lok);
        const uint64_t thr = st->watch_local;
        // [TAG_FN_L23_PINFIX] R-L23-1: as the OOM hook, nothing goes when an unpin cannot raise the room (no LOCAL charge
        // and no NON_LOCAL spill): one note per low-headroom episode
        uint64_t cap_w = 0;
        const uint64_t want_w = thr + 512*FN_MIB;
        if (lok && head < (int64_t) thr) {
            cap_w = fn_pin::oom_help_bytes(want_w, fn_oom_in(st, true, head));
            if (cap_w == 0 && !watch_noted) {
                watch_noted = true;
                fn_log("event=watcher_skip headroom_gib=%.2f thr_gib=%.2f l_ratio=%.4f pinned_gib=%.2f", head/(double) FN_GIB, fn_gib(thr),
                        st->cfg.l_ratio, fn_gib(st->reg.active_bytes()));
            }
        } else {
            watch_noted = false;
        }
        if (cap_w > 0) {
            std::lock_guard<std::mutex> op(st->op_mtx);
            st->n_watch.fetch_add(1);
            int n = 0;
            // R-REV-1: as the OOM hook, at most the bytes whose measured LOCAL charge covers the deficit per check (no visible
            // charge: the target's bytes), so one check never unpins every range for headroom it cannot raise
            const uint64_t cap = cap_w;
            const uint64_t b = fn_unpin(st, [&](uint64_t f) {
                if (f >= cap) {
                    return true;
                }
                bool ok2 = false;
                const int64_t h2 = fn_local_headroom(st, &ok2);
                return !ok2 || h2 >= (int64_t) want_w;
            }, 200000, true, &n);
            bool ok3 = false;
            const int64_t h3 = fn_local_headroom(st, &ok3);
            LLAMA_LOG_WARN("%s: [TAG_FN_L22_PIN] watcher: LOCAL headroom %.2f GiB < %.2f GiB: %d range(s), %.2f GiB unpinned, headroom now %.2f GiB\n",
                    __func__, head/(double) FN_GIB, fn_gib(thr), n, fn_gib(b), h3/(double) FN_GIB);
            fn_log("event=watcher headroom_gib=%.2f thr_gib=%.2f unpinned_gib=%.2f units=%d headroom_after_gib=%.2f pinned_gib=%.2f",
                    head/(double) FN_GIB, fn_gib(thr), fn_gib(b), n, h3/(double) FN_GIB, fn_gib(st->reg.active_bytes()));
        }
        // [TAG_FN_L23_PINFIX] R-L23-2: pins may only ever add speed. With pins, available RAM (free + standby) under the
        // drop line (LLAMA_FN_L23_PIN_DROP_MIB 4096) for 3 samples (6 s) drops ALL of them: from there on the process runs as
        // without pins (staged stream, no PIN marks); an unpin only unlocks, no page leaves the working set. Room >=
        // the drop line + 4 GiB (R-L23-6) for 60 s, 60 s after a drop, pins the resident units again (fn_pin::gate_step)
        const uint64_t avail = fn_avail_ram();
        const bool pinned = st->reg.any_active();
        const bool repinnable = !st->arena && st->reg.n_active_units() < st->reg.n_units() && !g_pin_refused_ever.load();
        // [TAG_FN_L23_PINFIX] R-L23-5: NON_LOCAL room under half the registration margin (other allocations grew) counts as
        // pressure too: the gate sees "no RAM" and drops every pin after 3 samples
        uint64_t nl_room = UINT64_MAX;
        if (pinned) {
            uint64_t nb = 0, nu = 0;
            if (llama_fn_dxgi_query(st->luid, 1, &nb, &nu)) {
                nl_room = nb > nu ? nb - nu : 0;
            }
        }
        const bool nl_low = nl_room < st->cfg.nl_margin/2;
        const int gs = fn_pin::gate_step(gst8, nl_low ? 0 : avail, pinned, repinnable, st->cfg.floor, 3, repin_b, 30);
        if (gs < 0) {
            std::lock_guard<std::mutex> op(st->op_mtx);
            st->n_valve.fetch_add(1);
            const uint64_t had = st->reg.active_bytes();
            int n = 0;
            const uint64_t b = fn_drop_all(st, &n);
            if (nl_low) {
                LLAMA_LOG_WARN("%s: [TAG_FN_L23_PINFIX] NON_LOCAL room %.2f GiB < %.2f GiB: dropping the pins\n", __func__, fn_gib(nl_room),
                        fn_gib(st->cfg.nl_margin/2));
            }
            LLAMA_LOG_WARN("%s: [TAG_FN_L23_PINFIX] RAM pressure: %.2f GiB available (< %.2f GiB for 6 s): all pins dropped (%d range(s), %.2f of "
                    "%.2f GiB; %.2f GiB left busy), running as without pins until room comes back\n", __func__, fn_gib(avail), fn_gib(st->cfg.floor),
                    n, fn_gib(b), fn_gib(had), fn_gib(st->reg.active_bytes()));
            fn_log("event=drop avail_gib=%.2f drop_gib=%.2f units=%d unpinned_gib=%.2f left_gib=%.2f", fn_gib(avail), fn_gib(st->cfg.floor), n,
                    fn_gib(b), fn_gib(st->reg.active_bytes()));
        } else if (gs > 0) {
            uint64_t b = 0;
            const int n = fn_repin(st, repin_b, &b);
            if (n > 0) {
                st->n_repin.fetch_add(1);
                LLAMA_LOG_WARN("%s: [TAG_FN_L23_PINFIX] room again (%.2f GiB available for 60 s): %d resident range(s) pinned (%.2f GiB), %.2f GiB "
                        "pinned, %.2f GiB available now\n", __func__, fn_gib(avail), n, fn_gib(b), fn_gib(st->reg.active_bytes()), fn_gib(fn_avail_ram()));
                fn_log("event=repin avail_gib=%.2f units=%d repinned_gib=%.2f pinned_gib=%.2f avail_after_gib=%.2f", fn_gib(avail), n, fn_gib(b),
                        fn_gib(st->reg.active_bytes()), fn_gib(fn_avail_ram()));
            }
        }
        if (fn_now_us() - t_status > 60000000) {
            t_status = fn_now_us();
            const fn_pin::budget_in b = fn_read_budget(st);
            fn_log("event=status pinned_gib=%.2f units=%d users=%d nl_gib=%.2f/%.2f l_gib=%.2f/%.2f avail_gib=%.2f trim_gib=%.2f oom=%d oom_skip=%d "
                    "valve=%d repin=%d watcher=%d unpinned_gib=%.2f", fn_gib(st->reg.active_bytes()), st->reg.n_active_units(), st->reg.users_total(),
                    fn_gib(b.nl_usage), fn_gib(b.nl_budget), fn_gib(b.l_usage), fn_gib(b.l_budget), fn_gib(b.avail), fn_gib(fn_trimmable(st)),
                    (int) st->n_oom.load(), (int) st->n_oom_skip.load(), (int) st->n_valve.load(), (int) st->n_repin.load(), (int) st->n_watch.load(),
                    fn_gib(st->unpinned.load()));
        }
    }
}

} // namespace

// ---------------------------------------------------------------------------------------------------------------------
// src/llama-fn-l22.h

bool llama_fn_dxgi_query(uint64_t luid, int non_local, uint64_t * budget, uint64_t * usage) {
    if (budget) {
        *budget = 0;
    }
    if (usage) {
        *usage = 0;
    }
#if defined(_WIN32)
    if (luid == 0) {
        luid = g_default_luid.load(std::memory_order_relaxed);
        if (luid == 0) {
            luid = fn_dev_luid(nullptr);
            g_default_luid.store(luid, std::memory_order_relaxed);
        }
        if (luid == 0) {
            return false; // no CUDA device
        }
    }
    std::lock_guard<std::mutex> lk(g_dxgi_mtx);
    fn_dxgi_enum_locked();
    for (const auto & e : g_dxgi) {
        if (e.luid != luid || e.ad == nullptr) {
            continue;
        }
        DXGI_QUERY_VIDEO_MEMORY_INFO info = {};
        if (FAILED(e.ad->QueryVideoMemoryInfo(0, non_local ? DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL : DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) {
            return false;
        }
        if (budget) {
            *budget = info.Budget;
        }
        if (usage) {
            *usage = info.CurrentUsage;
        }
        return true;
    }
    return false;
#else
    (void) luid; (void) non_local;
    return false;
#endif
}

void llama_fn_pin_on_ready(const llama_model * model) {
    if (model == nullptr || model->arch != LLM_ARCH_QWEN4EXP) {
        return;
    }
    const int pin = llama_fn_l3_int(*model, "LLAMA_FN_L22_PIN", 0);
    if (pin <= 0) {
        return;
    }
    std::lock_guard<std::mutex> lk(g_pin_mtx);
    if (g_pin.load() != nullptr) {
        return;
    }
    // llama_fn_pin_phase: 1 while this runs and the thread registers, 3 when it returns without a thread
    struct phase_guard {
        bool started = false;
        phase_guard() { g_pin_phase.store(1, std::memory_order_release); }
        ~phase_guard() {
            if (!started) {
                g_pin_phase.store(3, std::memory_order_release);
            }
        }
    } phase;
    const int64_t t0 = fn_now_us();
    ggml_backend_reg_t creg = ggml_backend_reg_by_name("CUDA");
    auto reg_fn   = creg ? (fn_reg_fn_t)   ggml_backend_reg_get_proc_address(creg, "ggml_backend_fn_register_host")   : nullptr;
    auto unreg_fn = creg ? (fn_unreg_fn_t) ggml_backend_reg_get_proc_address(creg, "ggml_backend_fn_unregister_host") : nullptr;
    if (reg_fn == nullptr || unreg_fn == nullptr) {
        LLAMA_LOG_WARN("%s: [TAG_FN_L22_PIN] no CUDA host registration in this build: no pins\n", __func__);
        return;
    }

    std::unique_ptr<fn_pin_state> st(new fn_pin_state());
    st->model    = model;
    st->mode     = pin >= 2 ? 2 : 1;
    st->reg_fn   = reg_fn;
    st->unreg_fn = unreg_fn;
    st->devptr   = (fn_devptr_t) ggml_backend_reg_get_proc_address(creg, "ggml_backend_fn_host_dev_ptr");
    st->set_oom  = (fn_set_oom_t) ggml_backend_reg_get_proc_address(creg, "ggml_backend_cuda_set_oom_hook");
    // [TAG_FN_L23_PINFIX] R-L23-8: ggml-cuda refuses any registration past 0.4 x RAM in total (backstop of the pin cap)
    using fn_set_cap_t = void (*)(size_t);
    if (auto set_cap = (fn_set_cap_t) ggml_backend_reg_get_proc_address(creg, "ggml_backend_cuda_fn_set_reg_cap")) {
        set_cap((size_t) (0.4*(double) fn_total_ram()));
    }
    // device addresses of pinned pages only for the decode DMA share (P14): the mapped flag is not needed by copies
    if (llama_fn_l22_switch_on(llama_fn_env(*model, "LLAMA_FN_L22_DMASHARE"))) {
        st->regmap = (fn_regmap_t) ggml_backend_reg_get_proc_address(creg, "ggml_backend_fn_register_host_mapped");
    }
    st->dev      = llama_moe_hot_device(*model);
    if (st->dev == nullptr && ggml_backend_reg_dev_count(creg) > 0) {
        st->dev = ggml_backend_reg_dev_get(creg, 0);
    }
    st->luid = fn_dev_luid(st->dev);
    g_default_luid.store(st->luid, std::memory_order_relaxed);
    {
        const char * u = llama_fn_env(*model, "LLAMA_FN_L22_PIN_UNIT");
        const std::string s = u ? u : "";
        st->unit = s == "spans" ? 1 : s == "arena" ? 2 : 0;
    }
    st->cfg.nl_margin = fn_mib_switch(*model, "LLAMA_FN_L22_PIN_MARGIN_MIB", 16384);
    st->cfg.l_margin  = fn_mib_switch(*model, "LLAMA_FN_L22_PIN_LOCAL_MARGIN_MIB", 1024);
    // [TAG_FN_L23_PINFIX] R-L23-2: the registration's RAM rule and the drop line are one value (pins add no RAM demand:
    // they lock resident pages only)
    st->cfg.floor     = fn_mib_switch(*model, "LLAMA_FN_L23_PIN_DROP_MIB", 4096);
    g_pin_idle_us.store((int64_t) std::max(50, llama_fn_l3_int(*model, "LLAMA_FN_L23_PIN_IDLE_MS", 1500))*1000);
    st->cfg.warm_keep = fn_mib_switch(*model, "LLAMA_FN_L22_PIN_WARM_KEEP_MIB", 0);
    st->cfg.max_pin   = fn_mib_switch(*model, "LLAMA_FN_L22_PIN_MAX_MIB", 0);
    st->arena_max     = fn_mib_switch(*model, "LLAMA_FN_L22_PIN_ARENA_MIB", 16384);
    {
        // [TAG_FN_L23_PINFIX] R-L23-8 HARD CAP (owner 10-25, upstream #31122: the WDDM page-lock budget is about half the RAM and
        // unstable past it; 10-21: three TDRs at 51-56 GiB): RAM/4 by default, LLAMA_FN_L22_PIN_MAX_MIB never above 0.4 x RAM
        const uint64_t ram  = fn_total_ram();
        const uint64_t hard = (uint64_t) (0.4*(double) ram);
        const uint64_t def  = ram/4;
        st->cfg.max_pin = st->cfg.max_pin > 0 ? std::min(st->cfg.max_pin, hard) : def;
        st->arena_max   = std::min<uint64_t>(st->arena_max, st->cfg.max_pin);
        LLAMA_LOG_INFO("%s: [TAG_FN_L23_PINFIX] pin cap %.2f GiB (RAM %.2f GiB; hard limit 0.4 x RAM = %.2f GiB)\n", __func__,
                fn_gib(st->cfg.max_pin), fn_gib(ram), fn_gib(hard));
    }
#if !defined(_WIN32)
    st->cfg.nl_margin = 0; // the Linux rule is the cap RAM/2 - 8 GiB itself
#endif

    // the layers: host-mapped routed experts, their classes and the hot set's view
    const size_t page = fn_page_size();
    const int n_layer = (int) model->layers.size();
    std::vector<fn_pin::layer_in> layers;
    std::vector<fn_layer_view>    views;
    int n_busy = 0;
    int n_vram = 0;
    size_t host_bytes = 0;
    const int64_t t_view_end = fn_now_us() + 3000000; // the hot set's view: retries while busy, 3 s in all
    for (int il = 0; il < n_layer; ++il) {
        const auto & L = model->layers[il];
        const ggml_tensor * ts[3] = { L.ffn_up_exps, L.ffn_gate_exps, L.ffn_down_exps };
        fn_pin::layer_in in;
        in.il = il;
        bool ok = true;
        int n_exp = 0;
        for (int k = 0; k < 3 && ok; ++k) {
            const uint8_t * p0 = nullptr;
            size_t n0 = 0;
            // the mapped model (CPU_Mapped) or plain host memory; never a buffer the driver already pins (CUDA_Host)
            const char * bt = ts[k] && ts[k]->buffer ? ggml_backend_buft_name(ggml_backend_buffer_get_type(ts[k]->buffer)) : "";
            if (ts[k] == nullptr || !llama_fn_expert_host_range(model, il, 0, k, &p0, &n0) ||
                (strcmp(bt, "CPU_Mapped") != 0 && strcmp(bt, "CPU") != 0)) {
                ok = false;
                break;
            }
            in.t[k].data     = p0;
            in.t[k].stride   = n0;
            in.t[k].n_expert = (int) ts[k]->ne[2];
            n_exp = k == 0 ? in.t[k].n_expert : std::min(n_exp, in.t[k].n_expert);
        }
        if (!ok || n_exp <= 0) {
            continue;
        }
        fn_layer_view v;
        v.slot.assign(n_exp, -1);
        v.cnt.assign(n_exp, 0.0f);
        v.vol.assign(n_exp, 0);
        for (int k = 0; k < 50 && (v.tv = llama_moe_hot_tier_view(ts[0], v.slot.data(), v.cnt.data(), v.vol.data(), n_exp)) == 0 &&
                fn_now_us() < t_view_end; ++k) {
            fn_sleep_ms(10);
        }
        n_busy += v.tv == 0 ? 1 : 0;
        v.cls.resize(n_exp);
        for (int e = 0; e < n_exp; ++e) {
            v.cls[e] = llama_fn_res_class(model, il, e);
            n_vram  += v.cls[e] == LLAMA_FN_RES_VRAM ? 1 : 0;
        }
        for (int k = 0; k < 3; ++k) {
            host_bytes += (size_t) n_exp*in.t[k].stride;
        }
        views.push_back(std::move(v));
        layers.push_back(std::move(in));
    }
    // the classes: P5's residency table when it is built (it says VRAM somewhere; it also knows the stream's lend layers
    // before their first lend), else the hot set's view (a resident in a slot that keeps its VRAM is not streamed)
    const bool trust_res = n_vram > 0;
    for (size_t li = 0; li < layers.size(); ++li) {
        fn_pin::layer_in & in = layers[li];
        const fn_layer_view & v = views[li];
        const int n_exp = (int) v.cls.size();
        in.ok.assign(n_exp, 0);
        bool streamed = true;
        double cpu = 0.0;
        for (int e = 0; e < n_exp; ++e) {
            bool can = v.cls[e] == LLAMA_FN_RES_RAM || v.cls[e] == LLAMA_FN_RES_PIN;
            if (trust_res) {
                streamed = streamed && v.cls[e] != LLAMA_FN_RES_VRAM;
            } else if (v.tv == 1 && v.slot[e] >= 0 && !v.vol[e]) {
                can      = false;
                streamed = false;
            }
            if (v.tv == 1 && v.slot[e] < 0) {
                cpu += v.cnt[e];
            }
            in.ok[e] = can ? 1 : 0;
        }
        in.fully_streamed = trust_res ? streamed : (v.tv == -1 || (v.tv == 1 && streamed));
        in.cpu_use        = cpu;
        if (st->unit == 1 && in.fully_streamed) {
            const uint8_t * lo = nullptr;
            const uint8_t * hi = nullptr;
            for (int k = 0; k < 3; ++k) {
                const uint8_t * s = in.t[k].data;
                const uint8_t * e = in.t[k].data + (size_t) n_exp*in.t[k].stride;
                lo = lo ? std::min(lo, s) : s;
                hi = hi ? std::max(hi, e) : e;
            }
            in.span_ok = fn_one_view(lo, hi);
        }
    }
    if (layers.empty()) {
        LLAMA_LOG_WARN("%s: [TAG_FN_L22_PIN] no layer with host-mapped routed experts: no pins\n", __func__);
        return;
    }
    const size_t max_range = (size_t) fn_mib_switch(*model, "LLAMA_FN_L22_PIN_RANGE_MIB", 256);
    st->plan = fn_pin::build_plan(layers, page, st->unit == 1, max_range);

    // the budget now (before any registration)
    st->reserve_need = fn_reserve_need(*model);
    fn_pin::budget_in b = fn_read_budget(st.get());
#if defined(_WIN32)
    if (!b.have_nl || !b.have_l) {
        LLAMA_LOG_WARN("%s: [TAG_FN_L22_PIN] DXGI budgets not readable (NON_LOCAL %d, LOCAL %d): no pins\n", __func__, b.have_nl, b.have_l);
        return;
    }
#endif
    st->nl_base = b.nl_usage;
    st->l_base  = b.l_usage;
    st->base_ok = true;
    // [TAG_FN_L23_PINFIX] R-L23-5 (FAULT 2026-10-21 12:48, nvlddmkm 153): 63.92 GiB of resident pins left NON_LOCAL 8.36 GiB
    // of room (65.39 / 73.75 GiB) and the first prompt's FULL reserve spilled past it (CUDA out of memory in a stream
    // synchronize, then a TDR). NON_LOCAL keeps LLAMA_FN_L22_PIN_MARGIN_MIB (now 16384) and at least what a later allocation
    // may add (reserve_need, 13.8 GiB here) + 2 GiB: the room a VRAM spill and the host allocations of a prompt need
    st->cfg.nl_margin = std::max<uint64_t>(st->cfg.nl_margin, st->reserve_need + 2*FN_GIB);
    // LOCAL: keep reserve_need + margin, but never less than today's headroom allows (the hot set takes the free VRAM,
    // so today's headroom is what every later allocation already runs with): at most the ready headroom - 256 MiB
    const int64_t l_head = b.have_l ? (int64_t) b.l_budget - (int64_t) b.l_usage : 0;
    st->l_ready    = l_head > 0 ? (uint64_t) l_head : 0;
    st->l_ready_ok = b.have_l;
    const uint64_t tol = 256*FN_MIB;
    st->cfg.l_keep = std::min<uint64_t>(st->reserve_need + st->cfg.l_margin, st->l_ready > tol ? st->l_ready - tol : 0);
    const uint64_t test_local = fn_mib_switch(*model, "LLAMA_FN_L22_PIN_TEST_LOCAL_MIB", 0);
    st->watch_local = test_local > 0 ? test_local
                                     : std::min<uint64_t>(st->reserve_need, st->l_ready > 2*tol ? st->l_ready - 2*tol : 0);
    // [TAG_FN_L23_PINFIX] R-L23-8: in mapped mode the cap stops the registration (RULE_MAX), not the plan. The fully streamed
    // layers come first in the plan but are read by direct I/O, so their mapped pages are not resident and get skipped; the
    // next groups fill the cap (l23p st12_1: the capped plan held only those 15 layers, 0 GiB pinned)
    fn_pin::budget_cfg c_plan = st->cfg;
    if (st->unit != 2) {
        c_plan.max_pin = 0;
    }
    const uint64_t allow = fn_pin::plan_bytes(b, c_plan);
    st->n_take = fn_pin::plan_take(st->plan, st->unit == 2 ? std::min(allow, st->arena_max) : allow);
    st->gst.assign(st->plan.groups.size(), {});
    uint64_t planned = 0;
    for (int g = 0; g < st->n_take; ++g) {
        planned += st->plan.groups[g].bytes;
    }

    // the registry: mapped mode = one unit per planned range; the arena packs its chunks
    if (st->unit == 2) {
        st->arena = true;
        planned = fn_arena_plan(st.get());
    } else {
        std::vector<fn_pin::registry::in_range> rs;
        int n_rng = 0;
        for (int g = 0; g < st->n_take; ++g) {
            n_rng += st->plan.groups[g].n_rng;
        }
        st->unit_group.assign(n_rng, -1);
        for (int i = 0; i < n_rng; ++i) {
            const fn_pin::range & r = st->plan.ranges[i];
            fn_pin::registry::in_range x;
            x.p    = r.p;
            x.n    = r.n;
            x.unit = i;
            x.src  = i;
            rs.push_back(x);
            st->unit_group[i] = r.group;
        }
        if (!st->reg.build(rs, n_rng)) {
            LLAMA_LOG_WARN("%s: [TAG_FN_L22_PIN] overlapping ranges in the plan were dropped\n", __func__);
        }
        for (int i = 0; i < n_rng; ++i) {
            st->reg.set_unit(i, (void *) st->plan.ranges[i].p, st->plan.ranges[i].n);
        }
        // [TAG_FN_L23_PINFIX] no marks before registration: the warm pass reads every streamed expert as without pins, the
        // thread pins what is resident after it (fn_ranges_run marks complete groups)
    }
    size_t n_whole = 0;
    for (int g = 0; g < st->n_take; ++g) {
        n_whole += st->plan.groups[g].whole ? 1 : 0;
    }
    LLAMA_LOG_INFO("%s: [TAG_FN_L22_PIN] plan: %zu layers, %.2f GiB of host experts, %zu ranges in %zu groups; taking %d groups "
            "(%zu whole-layer) = %.2f GiB (%s, %s); NON_LOCAL %.2f / %.2f GiB, LOCAL %.2f / %.2f GiB (keep %.2f, reserve_need %.2f, "
            "watcher %.2f), RAM available %.2f GiB (floor %.2f), tier view busy on %d layers, %.1f ms\n", __func__, layers.size(),
            fn_gib(host_bytes), st->plan.ranges.size(), st->plan.groups.size(), st->n_take, n_whole, fn_gib(planned),
            st->unit == 2 ? "arena" : st->unit == 1 ? "spans" : "ranges", st->mode == 2 ? "touched" : "before touch",
            fn_gib(b.nl_usage), fn_gib(b.nl_budget), fn_gib(b.l_usage), fn_gib(b.l_budget), fn_gib(st->cfg.l_keep),
            fn_gib(st->reserve_need), fn_gib(st->watch_local), fn_gib(b.avail), fn_gib(st->cfg.floor), n_busy, (fn_now_us() - t0)/1e3);
    fn_log("event=plan classes=%s layers=%zu host_gib=%.2f ranges=%zu groups=%zu take=%d whole=%zu planned_gib=%.2f unit=%d mode=%d "
            "nl_gib=%.2f/%.2f l_gib=%.2f/%.2f l_keep_gib=%.2f reserve_need_gib=%.2f avail_gib=%.2f", trust_res ? "table" : "hotset",
            layers.size(), fn_gib(host_bytes), st->plan.ranges.size(), st->plan.groups.size(), st->n_take, n_whole, fn_gib(planned),
            st->unit, st->mode,
            fn_gib(b.nl_usage), fn_gib(b.nl_budget), fn_gib(b.l_usage), fn_gib(b.l_budget), fn_gib(st->cfg.l_keep),
            fn_gib(st->reserve_need), fn_gib(b.avail));
    if (st->n_take == 0) {
        const uint64_t g0 = st->plan.groups.empty() ? 0 : st->plan.groups[0].bytes;
        const int rule = fn_pin::check(b, st->cfg, g0, g0);
        LLAMA_LOG_WARN("%s: [TAG_FN_L22_PIN] nothing to pin (the rule that stops the first group: %s)\n", __func__, fn_pin::rule_name(rule));
        fn_log("event=nothing rule=%s", fn_pin::rule_name(rule));
        return;
    }
    fn_pin_state * raw = st.release();
    g_pin.store(raw, std::memory_order_release);
    phase.started = true;
    raw->th = std::thread(fn_thread, raw);
}

bool llama_fn_pin_query(const void * p, size_t n, llama_fn_pin_seg * seg) {
    if (seg == nullptr) {
        return false;
    }
    seg->src    = p;
    seg->len    = n;
    seg->pinned = false;
    seg->ticket = -1;
    fn_pin_state * st = g_pin.load(std::memory_order_acquire);
    if (st != nullptr && n > 0) {
        if (g_query_stamp.load(std::memory_order_relaxed)) {
            g_query_us.store(fn_now_us(), std::memory_order_relaxed);
        }
        fn_pin::seg s;
        st->reg.query(p, n, &s);
        seg->src    = s.src;
        seg->len    = s.len;
        seg->pinned = s.pinned;
        seg->ticket = s.ticket;
        if (s.pinned) {
            return true;
        }
        n = s.len;
    }
    // [TAG_FN_L14_ARENA] the round-14 arena's registered chunks (LLAMA_FN_L14_ARENA_PIN_MIB): pinned without a ticket
    if (n > 0 && llama_fn_arena_pinned_any()) {
        bool in = false;
        const size_t a = llama_fn_arena_pinned_span(p, n, &in);
        seg->src    = p;
        seg->len    = a;
        seg->pinned = in;
        seg->ticket = -1;
    }
    return true;
}

bool llama_fn_pin_query_dev(const void * p, size_t n, llama_fn_pin_seg * seg, uint64_t * dev) {
    if (dev) {
        *dev = 0;
    }
    if (!llama_fn_pin_query(p, n, seg)) {
        return false;
    }
    fn_pin_state * st = g_pin.load(std::memory_order_acquire);
    if (dev && st != nullptr && seg->pinned && seg->ticket >= 0) {
        fn_pin::seg s;
        s.src    = seg->src;
        s.len    = seg->len;
        s.pinned = true;
        s.ticket = seg->ticket;
        *dev = st->reg.seg_dev(s);
    }
    return true;
}

void llama_fn_pin_release(int32_t ticket) {
    if (ticket < 0) {
        return;
    }
    fn_pin_state * st = g_pin.load(std::memory_order_acquire);
    if (st != nullptr) {
        st->reg.release(ticket);
    }
}

bool llama_fn_pin_active(void) {
    fn_pin_state * st = g_pin.load(std::memory_order_acquire);
    return (st != nullptr && st->reg.any_active()) || llama_fn_arena_pinned_any();
}

bool llama_fn_pin_on_oom(size_t need_bytes) {
    fn_pin_state * st = g_pin.load(std::memory_order_acquire);
    if (st == nullptr || !st->reg.any_active()) {
        return false;
    }
    std::lock_guard<std::mutex> op(st->op_mtx);
    const uint64_t want = (uint64_t) need_bytes + fn_precheck_margin(st);
    bool lok = false;
    const int64_t head0 = fn_local_headroom(st, &lok);
    // R-REV-1: stop on bytes as well as on headroom. The cap comes from the LOCAL charge per registered byte that the
    // registration measured (l_ratio): when DXGI shows the room already, cannot be read, or shows (almost) no charge,
    // unpinning does not raise the headroom there, so `want` bytes go and no more (never every range in one call)
    const double ratio = st->cfg.l_ratio;
    const bool   by_bytes = !lok || head0 >= (int64_t) want || ratio < fn_pin::OOM_RATIO_MIN;
    // [TAG_FN_L23_PINFIX] R-L23-1: with no LOCAL charge an unpin only helps a NON_LOCAL spill or an arena's commit;
    // otherwise nothing is unpinned (fn_pin::oom_help_bytes)
    const fn_pin::oom_in oi = fn_oom_in(st, lok, head0);
    const uint64_t nl_room = oi.nl_budget > oi.nl_usage ? oi.nl_budget - oi.nl_usage : 0;
    const uint64_t cap = fn_pin::oom_help_bytes(want, oi);
    if (cap == 0) {
        const uint64_t k = st->n_oom_skip.fetch_add(1) + 1;
        if (k <= 3 || k % 64 == 0) {
            LLAMA_LOG_INFO("%s: [TAG_FN_L23_PINFIX] OOM hook (need %.1f MiB): no unpin, it cannot raise the room (LOCAL headroom %.2f GiB, "
                    "LOCAL charge %.3f per byte, NON_LOCAL room %.2f GiB, pinned %.2f GiB; call %d)\n", __func__, need_bytes/1048576.0,
                    head0/(double) FN_GIB, ratio, fn_gib(nl_room), fn_gib(st->reg.active_bytes()), (int) k);
        }
        fn_log("event=oom_skip need_mib=%.1f headroom_gib=%.2f l_ratio=%.4f nl_room_gib=%.2f pinned_gib=%.2f n=%d", need_bytes/1048576.0,
                head0/(double) FN_GIB, ratio, fn_gib(nl_room), fn_gib(st->reg.active_bytes()), (int) k);
        return false;
    }
    st->n_oom.fetch_add(1);
    const int64_t t_end = fn_now_us() + 3000000;
    int n = 0;
    const uint64_t freed = fn_unpin(st, [&](uint64_t f) {
        if (fn_now_us() > t_end || f >= cap) {
            return true;
        }
        if (by_bytes) {
            return false;
        }
        bool ok2 = false;
        const int64_t h = fn_local_headroom(st, &ok2);
        return !ok2 || h >= (int64_t) want;
    }, 50000, true, &n);
    if (freed > 0) {
        st->frozen.store(true); // an unpin ends registration (no relock); nothing freed: registration goes on
    }
    bool ok3 = false;
    const int64_t head1 = fn_local_headroom(st, &ok3);
    LLAMA_LOG_WARN("%s: [TAG_FN_L22_PIN] OOM hook (need %.1f MiB): %d range(s), %.2f GiB unpinned (cap %.2f GiB, LOCAL charge %.3f per "
            "byte), LOCAL headroom %.2f -> %.2f GiB\n", __func__, need_bytes/1048576.0, n, fn_gib(freed), fn_gib(cap), ratio,
            head0/(double) FN_GIB, head1/(double) FN_GIB);
    fn_log("event=oom need_mib=%.1f unpinned_gib=%.2f units=%d by_bytes=%d cap_gib=%.2f l_ratio=%.4f headroom_gib=%.2f->%.2f pinned_gib=%.2f",
            need_bytes/1048576.0, fn_gib(freed), n, by_bytes ? 1 : 0, fn_gib(cap), ratio, head0/(double) FN_GIB, head1/(double) FN_GIB,
            fn_gib(st->reg.active_bytes()));
    return freed > 0;
}

bool llama_fn_pin_on_alloc_fail(size_t need_bytes) {
    return fn_on_alloc_fail(need_bytes, "libllama", fn_pin::AF_DEV_FATAL, 0); // a failed reserve fails the batch: partial unpin first
}

uint64_t llama_fn_pin_precheck_margin(void) {
    return fn_precheck_margin(g_pin.load(std::memory_order_acquire));
}

int llama_fn_pin_phase(void) {
    return g_pin_phase.load(std::memory_order_acquire);
}

void llama_fn_pin_decode_mark(bool begin) {
    if (begin) {
        g_dec_running.fetch_add(1, std::memory_order_relaxed);
    } else {
        g_dec_end_us.store(fn_now_us(), std::memory_order_relaxed);
        g_dec_running.fetch_sub(1, std::memory_order_relaxed);
    }
}

void llama_fn_pin_server_busy(int delta) {
    g_srv_us.store(fn_now_us(), std::memory_order_relaxed);
    if (delta != 0) {
        g_srv_busy.fetch_add(delta, std::memory_order_relaxed);
    }
    g_srv_us.store(fn_now_us(), std::memory_order_relaxed);
}

void llama_fn_pin_note_need(const llama_model * model, int kind, size_t bytes) {
    (void) model;
    if (kind < 0 || kind >= FN_NEED_N) {
        return;
    }
    uint64_t cur = g_need[kind].load(std::memory_order_relaxed);
    while ((uint64_t) bytes > cur && !g_need[kind].compare_exchange_weak(cur, (uint64_t) bytes, std::memory_order_relaxed)) {
    }
}

void llama_fn_pin_shutdown(const llama_model * model) {
    std::lock_guard<std::mutex> lk(g_pin_mtx);
    fn_pin_state * st = g_pin.load(std::memory_order_acquire);
    if (st == nullptr || (model != nullptr && st->model != model)) {
        return;
    }
    st->stop.store(true);
    {
        std::lock_guard<std::mutex> wk(st->wake_mtx);
        st->wake_cv.notify_all();
    }
    if (st->th.joinable()) {
        st->th.join();
    }
    std::lock_guard<std::mutex> op(st->op_mtx);
    int n = 0;
    // the contexts are gone: tickets still held after 3 s in all are leaks, those units go anyway
    const int64_t t_end = fn_now_us() + 3000000;
    const uint64_t b = fn_unpin(st, [&](uint64_t) { return fn_now_us() > t_end; }, 200000, false, &n);
    int left = 0;
    for (int u = 0; u < st->reg.n_units(); ++u) {
        const fn_pin::reg_unit & U = st->reg.unit_at(u);
        if (U.state.load() == fn_pin::ST_ACTIVE || U.state.load() == fn_pin::ST_DRAINING) {
            fn_unreg_unit(st, U, false);
            left++;
        }
    }
    fn_unhook_oom(st);
    LLAMA_LOG_INFO("%s: [TAG_FN_L22_PIN] %d range(s) (%.2f GiB) unregistered before the model's mapping closes (%d forced)\n", __func__,
            n + left, fn_gib(b), left);
    // the state stays allocated (a late release or query sees no active range)
    g_pin.store(nullptr, std::memory_order_release);
    g_pin_phase.store(3, std::memory_order_release);
}
