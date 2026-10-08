// [TAG_FN_L14_TIER] [TAG_FN_L15_TIER] see llama-fn-tier.h

#include "llama-fn-tier.h"

#include "llama-fn-auto.h"
#include "llama-impl.h"
#include "llama-model-loader.h"
#include "llama-model.h"
#include "llama-moecache.h"

#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#    include <psapi.h>
static HANDLE g_fn_warm_go = nullptr; // [TAG_FN_L15_WARM] set when the trunk context's first decode returns
#endif

// [TAG_FN_L15_WARM] the running warm pass's progress, for the prefill stream (llama_fn_warm_wait_layer): ranges [0, mark) are
// in the working set; layer il is complete once mark reaches layer_end[il]
static std::atomic<bool>   g_warm_running{false};
static std::atomic<size_t> g_warm_mark{0};
static std::vector<size_t> g_warm_layer_end;

namespace {

// the routed expert tensors of the trunk layers, by layer
std::vector<std::vector<ggml_tensor *>> fn_tier_layers(const llama_model & model) {
    std::vector<std::vector<ggml_tensor *>> res;
    const int n_layer = (int) std::min<size_t>(model.hparams.n_layer(), model.layers.size());
    for (int il = 0; il < n_layer; ++il) {
        const auto & L = model.layers[il];
        std::vector<ggml_tensor *> ts;
        for (ggml_tensor * t : { L.ffn_up_exps, L.ffn_gate_exps, L.ffn_down_exps }) {
            if (t != nullptr) {
                ts.push_back(t);
            }
        }
        res.push_back(ts);
    }
    return res;
}

bool fn_tier_on(const llama_model & model) {
    return model.arch == LLM_ARCH_QWEN4EXP && llama_fn_l3_flag(model, "LLAMA_FN_L14_TIER");
}

// [TAG_FN_L15_WARM] see llama_fn_tier_plan
bool fn_warm_on(const llama_model & model) {
#ifdef _WIN32
    return model.arch == LLM_ARCH_QWEN4EXP && !fn_tier_on(model) && llama_fn_l3_int(model, "LLAMA_FN_L15_WARM", 1) != 0;
#else
    GGML_UNUSED(model);
    return false;
#endif
}

} // namespace

struct llama_fn_tier {
    struct layer {
        int                                  il      = -1;
        const ggml_tensor *                  up      = nullptr; // the hot set's key
        char *                               base[3] = {};      // up, gate, down
        size_t                               nb[3]   = {};      // bytes of one expert in each
        int                                  n_exp   = 0;
        std::unique_ptr<std::atomic<uint8_t>[]> lk;             // locked (read by the hot set's eviction gate)
        std::vector<int32_t>                 slot;              // the hot set's view: slot or -1
        std::vector<float>                   cnt;               // decayed count
        std::vector<uint8_t>                 vol;               // resident in a slot that can go without a choice
    };
    struct item {
        int l;
        int e;
    };
    std::vector<layer>                            layers;
    std::unordered_map<const ggml_tensor *, int>  by_up;
    std::unordered_map<uintptr_t, int>            edge;     // first / last page of a locked slice: the locked slices that hold it
    std::mutex                                    mtx;      // edge, the counters and the working-set size (fill threads)
    std::atomic<bool>                             enforce{false}; // the eviction gate is on (after the first fill)
    size_t                                        page         = 4096;
    size_t                                        total        = 0;
    size_t                                        locked_bytes = 0;
    size_t                                        reserved     = 0; // bytes a fill thread is locking right now
    size_t                                        n_locked     = 0; // experts
    size_t                                        cap          = 0;
    size_t                                        free_min     = 0;
    size_t                                        low          = 0;
    int                                           delay_ms     = 0;
    int                                           n_fill       = 8;
    size_t                                        commit_min   = 0;  // commit kept free (LLAMA_FN_L15_TIER_COMMIT_MIB)
    std::atomic<uint64_t>                         n_gated{0};        // evictions the gate held back (for the log)
#ifdef _WIN32
    HANDLE                                        stop = nullptr;
    HANDLE                                        go   = nullptr; // [TAG_FN_L15_WARM]
    std::thread                                   worker;
#endif
};

namespace {

// [TAG_FN_L15_TIER] the hot set's eviction gate: before the first fill everything may go (as without a tier)
[[maybe_unused]] bool fn_tier_ready(const ggml_tensor * up, int32_t e, void * ud) {
    llama_fn_tier * t = (llama_fn_tier *) ud;
    if (t == nullptr || !t->enforce.load(std::memory_order_relaxed)) {
        return true;
    }
    const auto it = t->by_up.find(up);
    if (it == t->by_up.end() || e < 0 || e >= t->layers[it->second].n_exp) {
        return true;
    }
    const bool ok = t->layers[it->second].lk[e].load(std::memory_order_relaxed) != 0;
    if (!ok) {
        t->n_gated.fetch_add(1, std::memory_order_relaxed);
    }
    return ok;
}

} // namespace

#ifdef _WIN32

namespace {

using tier_t = llama_fn_tier;

// the page range of slice s (up, gate, down) of expert e
void fn_tier_slice(const tier_t * t, const tier_t::layer & L, int e, int s, char ** lo, char ** hi) {
    const uintptr_t a = (uintptr_t) L.base[s] + (uintptr_t) e*L.nb[s];
    *lo = (char *) (a/t->page*t->page);
    *hi = (char *) ((a + L.nb[s] + t->page - 1)/t->page*t->page);
}

size_t fn_tier_bytes(const tier_t * t, const tier_t::item & it) {
    size_t b = 0;
    for (int s = 0; s < 3; ++s) {
        char * lo;
        char * hi;
        fn_tier_slice(t, t->layers[it.l], it.e, s, &lo, &hi);
        b += (size_t) (hi - lo);
    }
    return b;
}

size_t fn_tier_sum(const tier_t * t, const std::vector<tier_t::item> & items) {
    size_t b = 0;
    for (const auto & it : items) {
        b += fn_tier_bytes(t, it);
    }
    return b;
}

// the bytes of items [i0, i1) not in this process's working set yet, from every 16th page (pages the server already
// touched cost no RAM to lock: only the rest comes out of the available memory)
size_t fn_tier_nonresident(const tier_t * t, const std::vector<tier_t::item> & items, size_t i0, size_t i1, size_t bytes) {
    std::vector<PSAPI_WORKING_SET_EX_INFORMATION> q;
    for (size_t i = i0; i < i1; ++i) {
        for (int s = 0; s < 3; ++s) {
            char * lo;
            char * hi;
            fn_tier_slice(t, t->layers[items[i].l], items[i].e, s, &lo, &hi);
            for (char * p = lo; p < hi; p += 16*t->page) {
                PSAPI_WORKING_SET_EX_INFORMATION x = {};
                x.VirtualAddress = p;
                q.push_back(x);
            }
        }
    }
    if (q.empty() || !K32QueryWorkingSetEx(GetCurrentProcess(), q.data(), (DWORD) (q.size()*sizeof(q[0])))) {
        return bytes;
    }
    size_t out = 0;
    for (const auto & x : q) {
        out += x.VirtualAttributes.Valid ? 0 : 1;
    }
    return (size_t) ((double) bytes*out/q.size());
}

size_t fn_tier_avail() {
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    GlobalMemoryStatusEx(&ms);
    return (size_t) ms.ullAvailPhys;
}

// [TAG_FN_L15_TIER] the commit the system has left: a larger working-set minimum is charged to commit, and at the limit
// VirtualLock failed with 1455 (the GPU's allocations already hold 22-33 GB of commit)
size_t fn_tier_avail_commit() {
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    GlobalMemoryStatusEx(&ms);
    return (size_t) ms.ullAvailPageFile;
}

// the working-set minimum (and, when needed, the maximum) moved by delta bytes
bool fn_tier_ws_add(int64_t delta) {
    SIZE_T mn = 0;
    SIZE_T mx = 0;
    DWORD  fl = 0;
    if (!GetProcessWorkingSetSizeEx(GetCurrentProcess(), &mn, &mx, &fl)) {
        return false;
    }
    const SIZE_T nmn = (SIZE_T) std::max<int64_t>((int64_t) mn + delta, (int64_t) 1 << 20);
    const SIZE_T nmx = std::max<SIZE_T>(mx, nmn + ((SIZE_T) 64 << 20));
    return SetProcessWorkingSetSizeEx(GetCurrentProcess(), nmn, nmx,
            QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE) != 0;
}

// the first and last page of a slice (one page when they are the same)
int fn_tier_ends(const tier_t * t, char * lo, char * hi, char ** ends) {
    ends[0] = lo;
    ends[1] = hi - t->page;
    return ends[1] != ends[0] ? 2 : 1;
}

// unlock one expert: the inner pages of its slices, and their first / last page when no other locked slice holds it.
// Only while no fill thread runs (the edge counts and the pages' lock state must agree). The gate flag goes first.
void fn_tier_unlock(tier_t * t, const tier_t::item & it) {
    tier_t::layer & L = t->layers[it.l];
    if (!L.lk[it.e].load()) {
        return;
    }
    L.lk[it.e].store(0);
    size_t b = 0;
    for (int s = 0; s < 3; ++s) {
        char * lo;
        char * hi;
        fn_tier_slice(t, L, it.e, s, &lo, &hi);
        b += (size_t) (hi - lo);
        if (hi - lo > (ptrdiff_t) (2*t->page)) {
            VirtualUnlock(lo + t->page, (size_t) (hi - lo) - 2*t->page);
        }
        char * ends[2];
        const int n = fn_tier_ends(t, lo, hi, ends);
        for (int k = 0; k < n; ++k) {
            auto e = t->edge.find((uintptr_t) ends[k]);
            if (e != t->edge.end() && --e->second == 0) {
                t->edge.erase(e);
                VirtualUnlock(ends[k], t->page);
            }
        }
    }
    t->locked_bytes -= b;
    t->n_locked--;
    fn_tier_ws_add(-(int64_t) b);
}

// lock items in order with n_fill threads, 32 experts at a time, while the cap, the available RAM and max_bytes allow.
// Each thread hints the read of its chunk, touches every page (the faults run in parallel) and locks it. Returns the bytes.
size_t fn_tier_fill(tier_t * t, const std::vector<tier_t::item> & items, size_t max_bytes, const char ** why) {
    constexpr size_t chunk = 32;
    using prefetch_t = BOOL (WINAPI *)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
    static const prefetch_t pf = (prefetch_t) (void *) GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "PrefetchVirtualMemory");

    std::atomic<size_t> next{0};
    std::atomic<bool>   full{false};
    std::atomic<const char *> stop_why{nullptr};
    size_t              added  = 0;
    size_t              failed = 0;
    auto run = [&]() {
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
        std::vector<WIN32_MEMORY_RANGE_ENTRY> rs;
        std::vector<size_t> sizes;
        for (;;) {
            if (full.load() || WaitForSingleObject(t->stop, 0) == WAIT_OBJECT_0) {
                return;
            }
            const size_t c0 = next.fetch_add(chunk);
            if (c0 >= items.size()) {
                return;
            }
            const size_t c1 = std::min(c0 + chunk, items.size());
            sizes.clear();
            size_t b = 0;
            for (size_t i = c0; i < c1; ++i) {
                sizes.push_back(fn_tier_bytes(t, items[i]));
                b += sizes.back();
            }
            const size_t nonres = fn_tier_nonresident(t, items, c0, c1, b);
            {
                std::lock_guard<std::mutex> lk(t->mtx);
                const char * w = nullptr;
                if (t->locked_bytes + t->reserved + b > t->cap) {
                    w = "cap";
                } else if (added + b > max_bytes) {
                    w = "step limit";
                } else if (fn_tier_avail() < t->free_min + nonres + t->reserved) {
                    w = "available RAM";
                } else if (fn_tier_avail_commit() < t->commit_min + b + t->reserved) {
                    w = "commit limit";
                } else if (!fn_tier_ws_add((int64_t) b)) {
                    w = "working-set quota";
                }
                if (w) {
                    stop_why.store(w);
                    full.store(true);
                    return;
                }
                t->reserved += b;
                added       += b;
            }
            rs.clear();
            for (size_t i = c0; i < c1; ++i) {
                for (int s = 0; s < 3; ++s) {
                    char * lo;
                    char * hi;
                    fn_tier_slice(t, t->layers[items[i].l], items[i].e, s, &lo, &hi);
                    rs.push_back({ lo, (SIZE_T) (hi - lo) });
                }
            }
            if (pf != nullptr) {
                pf(GetCurrentProcess(), (ULONG_PTR) rs.size(), rs.data(), 0);
            }
            for (const auto & r : rs) {
                for (size_t o = 0; o < r.NumberOfBytes; o += t->page) {
                    (void) *(volatile const char *) ((const char *) r.VirtualAddress + o);
                }
            }
            std::vector<uint8_t> ok(c1 - c0, 0);
            for (size_t i = c0; i < c1; ++i) {
                int s = 0;
                for (; s < 3; ++s) {
                    if (!VirtualLock(rs[(i - c0)*3 + s].VirtualAddress, rs[(i - c0)*3 + s].NumberOfBytes)) {
                        break;
                    }
                }
                if (s == 3) {
                    ok[i - c0] = 1;
                } else {
                    for (int k = 0; k < s; ++k) { // inner pages only: a neighbour may hold the ends
                        const auto & r = rs[(i - c0)*3 + k];
                        if (r.NumberOfBytes > 2*t->page) {
                            VirtualUnlock((char *) r.VirtualAddress + t->page, r.NumberOfBytes - 2*t->page);
                        }
                    }
                }
            }
            std::lock_guard<std::mutex> lk(t->mtx);
            size_t lost = 0;
            for (size_t i = c0; i < c1; ++i) {
                if (!ok[i - c0]) {
                    lost += sizes[i - c0];
                    failed++;
                    continue;
                }
                for (int s = 0; s < 3; ++s) {
                    const auto & r = rs[(i - c0)*3 + s];
                    char * ends[2];
                    const int n = fn_tier_ends(t, (char *) r.VirtualAddress, (char *) r.VirtualAddress + r.NumberOfBytes, ends);
                    for (int k = 0; k < n; ++k) {
                        t->edge[(uintptr_t) ends[k]]++;
                    }
                }
                t->layers[items[i].l].lk[items[i].e].store(1);
                t->locked_bytes += sizes[i - c0];
                t->n_locked++;
            }
            t->reserved -= b;
            if (lost > 0) {
                fn_tier_ws_add(-(int64_t) lost);
                added -= lost;
                stop_why.store("VirtualLock failed");
                full.store(true);
            }
        }
    };
    std::vector<std::thread> ths;
    for (int k = 1; k < t->n_fill; ++k) {
        ths.emplace_back(run);
    }
    run();
    for (auto & th : ths) {
        th.join();
    }
    if (failed > 0) {
        LLAMA_LOG_WARN("[TAG_FN_L15_TIER] %zu experts could not be locked (error %lu)\n", failed, (unsigned long) GetLastError());
    }
    if (why) {
        *why = stop_why.load();
    }
    return added;
}

// the hot set's view of every layer; a layer whose bookkeeping is busy keeps its last view
void fn_tier_look(tier_t * t) {
    for (auto & L : t->layers) {
        std::vector<int32_t> s((size_t) L.n_exp, -1);
        std::vector<float>   c((size_t) L.n_exp, 0.0f);
        std::vector<uint8_t> v((size_t) L.n_exp, 0);
        const int r = llama_moe_hot_tier_view(L.up, s.data(), c.data(), v.data(), L.n_exp);
        if (r != 0) {
            L.slot.swap(s);
            L.cnt.swap(c);
            L.vol.swap(v);
        }
    }
}

// [TAG_FN_L15_TIER] v6: one fill, then the lock set stays. The fill order: the experts the CPU computes (the most used
// first), the residents that can lose their slot without a choice, then the other residents with the lowest counts first
// (the hot set's likely victims) until the cap or the commit limit stops it. The hot set then evicts only locked residents:
// a swap moves a locked victim to the CPU and a locked CPU expert into VRAM, so the locked set does not change. Only an
// expert that left VRAM without a choice (the KV cache's lend) can be a CPU expert without a lock; a round every 2 s locks
// it, taking room from the residents with the highest counts (the least likely to be evicted)
struct tier_plan {
    std::vector<tier_t::item> cpu;      // computed by the CPU, most used first
    std::vector<tier_t::item> vol;      // residents that can lose their slot without a choice
    std::vector<tier_t::item> res_asc;  // the other residents, lowest count first
};

tier_plan fn_tier_plan_now(const tier_t * t) {
    tier_plan p;
    std::vector<std::pair<float, tier_t::item>> cpu;
    std::vector<std::pair<float, tier_t::item>> res;
    for (int l = 0; l < (int) t->layers.size(); ++l) {
        const auto & L = t->layers[l];
        for (int e = 0; e < L.n_exp; ++e) {
            if (L.slot[e] < 0) {
                cpu.push_back({ L.cnt[e], { l, e } });
            } else if (L.vol[e]) {
                p.vol.push_back({ l, e });
            } else {
                res.push_back({ L.cnt[e], { l, e } });
            }
        }
    }
    std::stable_sort(cpu.begin(), cpu.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
    std::stable_sort(res.begin(), res.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
    for (const auto & c : cpu) {
        p.cpu.push_back(c.second);
    }
    for (const auto & r : res) {
        p.res_asc.push_back(r.second);
    }
    return p;
}

std::vector<tier_t::item> fn_tier_missing(const tier_t * t, const std::vector<tier_t::item> & v) {
    std::vector<tier_t::item> out;
    for (const auto & it : v) {
        if (!t->layers[it.l].lk[it.e].load()) {
            out.push_back(it);
        }
    }
    return out;
}

// unlock about `bytes` of the listed experts (those locked), from the end of the list
size_t fn_tier_release(tier_t * t, const std::vector<tier_t::item> & v, size_t bytes) {
    const size_t b0 = t->locked_bytes;
    for (size_t i = v.size(); i-- > 0 && b0 - t->locked_bytes < bytes; ) {
        fn_tier_unlock(t, v[i]);
    }
    return b0 - t->locked_bytes;
}

size_t fn_tier_locked_sum(const tier_t * t, const std::vector<tier_t::item> & v) {
    return fn_tier_sum(t, v) - fn_tier_sum(t, fn_tier_missing(t, v));
}

// after the start delay (the context, its warm-up and the first kernels' loads come first): the fill, then the eviction
// gate goes on and a round runs every 2 s. Below `low` for three looks in a row about 1 GiB goes back: residents with the
// highest counts first, then volatile ones, then the least used CPU experts
void fn_tier_worker(tier_t * t) {
    if (WaitForSingleObject(t->stop, (DWORD) t->delay_ms) == WAIT_OBJECT_0) {
        return;
    }
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    const auto t0 = std::chrono::steady_clock::now();
    fn_tier_look(t);
    tier_plan p = fn_tier_plan_now(t);
    const char * why = nullptr;
    size_t b0 = 0;
    for (const auto * v : { &p.cpu, &p.vol, &p.res_asc }) {
        const char * w = nullptr;
        b0 += fn_tier_fill(t, *v, SIZE_MAX, &w);
        if (w) {
            why = w;
            break;
        }
    }
    LLAMA_LOG_INFO("[TAG_FN_L15_TIER] fill in %.1f s: %.2f GiB locked = %.2f of %.2f GiB of experts the CPU computes, %.2f of %.2f "
            "GiB of volatile residents, %.2f of %.2f GiB of the other residents (cap %.1f GiB%s%s); %.1f GiB RAM available; the "
            "eviction gate is on\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
            b0/1073741824.0, fn_tier_locked_sum(t, p.cpu)/1073741824.0, fn_tier_sum(t, p.cpu)/1073741824.0,
            fn_tier_locked_sum(t, p.vol)/1073741824.0, fn_tier_sum(t, p.vol)/1073741824.0,
            fn_tier_locked_sum(t, p.res_asc)/1073741824.0, fn_tier_sum(t, p.res_asc)/1073741824.0, t->cap/1073741824.0,
            why ? ", stopped by: " : "", why ? why : "", fn_tier_avail()/1073741824.0);
    t->enforce.store(true);

    int    n_low   = 0;
    size_t q_bytes = 0;
    size_t q_freed = 0;
    auto   q_since = std::chrono::steady_clock::now();
    for (;;) {
        if (WaitForSingleObject(t->stop, 2000) == WAIT_OBJECT_0) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        fn_tier_look(t);
        p = fn_tier_plan_now(t);
        if (fn_tier_avail() < t->low && t->n_locked > 0) {
            if (++n_low >= 3) {
                const size_t gib = (size_t) 1 << 30;
                size_t r = fn_tier_release(t, p.res_asc, gib);
                if (r < gib) {
                    r += fn_tier_release(t, p.vol, gib - r);
                }
                if (r < gib) {
                    r += fn_tier_release(t, p.cpu, gib - r);
                }
                LLAMA_LOG_WARN("[TAG_FN_L15_TIER] %.1f GiB RAM available (below %.1f GiB for 6 s): %.2f GiB unlocked, %.2f GiB stay "
                        "locked\n", fn_tier_avail()/1073741824.0, t->low/1073741824.0, r/1073741824.0, t->locked_bytes/1073741824.0);
                n_low = 0;
            }
            continue;
        }
        n_low = 0;
        // CPU experts without a lock (they left VRAM without a choice): room from the residents with the highest counts
        const auto mc = fn_tier_missing(t, p.cpu);
        if (!mc.empty()) {
            const size_t need = std::min<size_t>(fn_tier_sum(t, mc), (size_t) 1 << 30);
            if (t->locked_bytes + need > t->cap) {
                q_freed += fn_tier_release(t, p.res_asc, t->locked_bytes + need - t->cap);
            }
            q_bytes += fn_tier_fill(t, mc, (size_t) 1 << 30, nullptr);
        }
        // volatile residents: only into free room
        const auto mo = fn_tier_missing(t, p.vol);
        if (!mo.empty()) {
            q_bytes += fn_tier_fill(t, mo, (size_t) 512 << 20, nullptr);
        }
        if (now - q_since > std::chrono::seconds(60)) {
            LLAMA_LOG_INFO("[TAG_FN_L15_TIER] last minute: %.2f GiB locked, %.2f GiB unlocked for them, %llu eviction checks held back; "
                    "now %.2f GiB locked = %.2f of %.2f GiB of CPU experts, %.2f of %.2f GiB of volatile residents, %.2f of %.2f GiB of "
                    "the other residents\n", q_bytes/1073741824.0, q_freed/1073741824.0,
                    (unsigned long long) t->n_gated.exchange(0), t->locked_bytes/1073741824.0,
                    fn_tier_locked_sum(t, p.cpu)/1073741824.0, fn_tier_sum(t, p.cpu)/1073741824.0,
                    fn_tier_locked_sum(t, p.vol)/1073741824.0, fn_tier_sum(t, p.vol)/1073741824.0,
                    fn_tier_locked_sum(t, p.res_asc)/1073741824.0, fn_tier_sum(t, p.res_asc)/1073741824.0);
            q_bytes = 0;
            q_freed = 0;
            q_since = now;
        }
    }
}

} // namespace

#endif

void llama_fn_tier_deleter::operator()(llama_fn_tier * t) const {
    if (t == nullptr) {
        return;
    }
    llama_moe_hot_set_host_ready(nullptr, nullptr);
#ifdef _WIN32
    if (t->stop) {
        SetEvent(t->stop);
    }
    if (t->worker.joinable()) {
        t->worker.join();
    }
    for (int l = 0; l < (int) t->layers.size(); ++l) {
        for (int e = 0; e < t->layers[l].n_exp; ++e) {
            fn_tier_unlock(t, { l, e });
        }
    }
    if (t->stop) {
        CloseHandle(t->stop);
    }
    if (t->go) {
        if (g_fn_warm_go == t->go) {
            g_fn_warm_go = nullptr;
        }
        CloseHandle(t->go);
    }
#endif
    delete t;
}

void llama_fn_tier_plan(const llama_model & model, llama_model_loader & ml) {
    // [TAG_FN_L15_TRIM] qwen4exp: the GPU tensors' pages (5.7 GiB) leave RAM first, whether or not the tier runs
    if (model.arch == LLM_ARCH_QWEN4EXP && ml.use_mmap && llama_fn_l3_int(model, "LLAMA_FN_L15_TRIM", 1) != 0) {
        ml.trim_uploaded = true;
    }
    // [TAG_FN_L15_WARM] the routed experts (71.7 GiB) are read by the warm pass after the load instead: the load's prefetch of
    // the whole file pushed the oldest cached pages out of RAM, often this model's own pages from an earlier run, and the
    // first prompt faulted ~30 GB back in at ~3 GB/s (l15 trace1: first 8K ubatch 19.2 s instead of 3.1 s)
    const bool warm = fn_warm_on(model);
    if ((!fn_tier_on(model) && !warm) || !ml.use_mmap) {
        return;
    }
    size_t n = 0;
    for (const auto & ts : fn_tier_layers(model)) {
        for (ggml_tensor * t : ts) {
            const auto it = ml.weights_map.find(ggml_get_name(t));
            if (it == ml.weights_map.end()) {
                continue;
            }
            ml.noprefetch[it->second.idx].emplace_back(it->second.offs, it->second.offs + ggml_nbytes(t));
            n += ggml_nbytes(t);
        }
    }
    LLAMA_LOG_INFO("%s: [%s] %.2f GiB of routed experts kept out of the load's prefetch\n", __func__,
            warm ? "TAG_FN_L15_WARM" : "TAG_FN_L14_TIER", n/1073741824.0);
}

#ifndef _WIN32

llama_fn_tier_ptr llama_fn_tier_build(llama_model & /*model*/, llama_model_loader & /*ml*/) {
    return nullptr;
}

#else

// [TAG_FN_L15_WARM] layer by layer, one prefetch per 64 MiB piece of the routed experts, then every page touched: the
// pages enter the working set as they arrive, so a later piece's read cannot push them out. Up to the available RAM
// minus LLAMA_FN_L15_WARM_FREE_MIB (4096); LLAMA_FN_L15_WARM_THREADS (16) threads.
static void fn_warm(const llama_model & model, HANDLE stop) {
    using prefetch_t = BOOL (WINAPI *)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
    static const prefetch_t pf = (prefetch_t) (void *) GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "PrefetchVirtualMemory");
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    const size_t page  = si.dwPageSize ? si.dwPageSize : 4096;
    const size_t piece = (size_t) 64 << 20;
    struct range {
        const char * p;
        size_t       n;
    };
    std::vector<range> rs;
    std::vector<size_t> layer_end;
    size_t total = 0;
    for (const auto & ts : fn_tier_layers(model)) {
        layer_end.push_back(rs.size());
        for (const ggml_tensor * t : ts) {
            if (!t->buffer || !ggml_backend_buffer_is_host(t->buffer) || !t->data) {
                continue;
            }
            const size_t n = ggml_nbytes(t);
            for (size_t o = 0; o < n; o += piece) {
                rs.push_back({ (const char *) t->data + o, std::min(piece, n - o) });
            }
            total += n;
        }
        layer_end.back() = rs.size();
    }
    if (rs.empty()) {
        return;
    }
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    GlobalMemoryStatusEx(&ms);
    const size_t keep   = (size_t) std::max(0, llama_fn_l3_int(model, "LLAMA_FN_L15_WARM_FREE_MIB", 4096)) << 20;
    const size_t budget = (size_t) ms.ullAvailPhys > keep ? (size_t) ms.ullAvailPhys - keep : 0;
    size_t n_use = 0;
    size_t bytes = 0;
    for (; n_use < rs.size() && bytes + rs[n_use].n <= budget; ++n_use) {
        bytes += rs[n_use].n;
    }
    PROCESS_MEMORY_COUNTERS pm0 = {};
    GetProcessMemoryInfo(GetCurrentProcess(), &pm0, sizeof(pm0));
    const int n_thr = std::max(1, std::min(32, llama_fn_l3_int(model, "LLAMA_FN_L15_WARM_THREADS", 16)));
    std::atomic<size_t> next{0};
    std::unique_ptr<std::atomic<uint8_t>[]> done(new std::atomic<uint8_t>[n_use + 1]());
    g_warm_layer_end = layer_end;
    g_warm_mark.store(0);
    g_warm_running.store(true, std::memory_order_release);
    auto run = [&]() {
        for (size_t i = next.fetch_add(1); i < n_use; i = next.fetch_add(1)) {
            if (WaitForSingleObject(stop, 0) == WAIT_OBJECT_0) {
                return;
            }
            WIN32_MEMORY_RANGE_ENTRY e = { (PVOID) rs[i].p, (SIZE_T) rs[i].n };
            if (pf != nullptr) {
                pf(GetCurrentProcess(), 1, &e, 0);
            }
            for (size_t o = 0; o < rs[i].n; o += page) {
                (void) *(volatile const char *) (rs[i].p + o);
            }
            (void) *(volatile const char *) (rs[i].p + rs[i].n - 1);
            done[i].store(1);
            for (size_t m = g_warm_mark.load(); m < n_use && done[m].load(); ) {
                if (g_warm_mark.compare_exchange_weak(m, m + 1)) {
                    m++;
                }
            }
        }
    };
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> ths;
    for (int k = 1; k < n_thr; ++k) {
        ths.emplace_back(run);
    }
    run();
    for (auto & th : ths) {
        th.join();
    }
    g_warm_running.store(false, std::memory_order_release);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    PROCESS_MEMORY_COUNTERS pm1 = {};
    GetProcessMemoryInfo(GetCurrentProcess(), &pm1, sizeof(pm1));
    LLAMA_LOG_INFO("%s: [TAG_FN_L15_WARM] %.2f of %.2f GiB of routed experts read into the working set in %.1f s by %d threads "
            "(%.1f GiB of RAM was available, %.1f GiB kept free; working set %.1f -> %.1f GiB)\n", __func__, bytes/1073741824.0,
            total/1073741824.0, secs, n_thr, ms.ullAvailPhys/1073741824.0, keep/1073741824.0, pm0.WorkingSetSize/1073741824.0,
            pm1.WorkingSetSize/1073741824.0);
    if (n_use < rs.size()) {
        LLAMA_LOG_WARN("%s: [TAG_FN_L15_WARM] %.2f GiB of routed experts did not fit in the available RAM: the first prompt reads "
                "them from the disk\n", __func__, (total - bytes)/1073741824.0);
    }
}

llama_fn_tier_ptr llama_fn_tier_build(llama_model & model, llama_model_loader & ml) {
    if (!fn_tier_on(model)) {
        if (!fn_warm_on(model) || !ml.use_mmap) {
            return nullptr;
        }
        // [TAG_FN_L15_WARM] the working-set maximum first: every VirtualLock retry (EMBDLOCK's table and the PLE row cache,
        // llama-mmap.cpp raw_lock) sets it to its old value + the lock, about 0.74 GiB, and that trimmed a 72 GiB working set
        // to 0.8 GiB (l15 abw_new1/2, abx_new1/2). Only the minimum is charged to commit.
        MEMORYSTATUSEX ms;
        ms.dwLength = sizeof(ms);
        GlobalMemoryStatusEx(&ms);
        SIZE_T mn = 0;
        SIZE_T mx = 0;
        DWORD  fl = 0;
        if (GetProcessWorkingSetSizeEx(GetCurrentProcess(), &mn, &mx, &fl)) {
            const SIZE_T want = std::max<SIZE_T>(mx, (SIZE_T) (ms.ullTotalPhys/10*9));
            if (!SetProcessWorkingSetSizeEx(GetCurrentProcess(), mn, want, QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE)) {
                LLAMA_LOG_WARN("%s: [TAG_FN_L15_WARM] the working-set maximum stays %.2f GiB (error %lu)\n", __func__,
                        mx/1073741824.0, (unsigned long) GetLastError());
            }
        }
        // then the background warm pass, once the first decode (the server's warm-up run) has returned: started at the load
        // it shared the disk and the page faults with the start, and the server was ready after 27-29 s instead of 7-8 s
        // (l15 abx5 vs abx3); LLAMA_FN_L15_WARM_DELAY_MS (15000) after the load when no decode comes
        llama_fn_tier_ptr w(new llama_fn_tier());
        w->stop = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        w->go   = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (w->stop == nullptr || w->go == nullptr) {
            return nullptr;
        }
        g_fn_warm_go = w->go;
        const DWORD delay = (DWORD) std::max(0, llama_fn_l3_int(model, "LLAMA_FN_L15_WARM_DELAY_MS", 15000));
        const llama_model * m = &model;
        llama_fn_tier * t = w.get();
        w->worker = std::thread([m, t, delay]() {
            const HANDLE hs[2] = { t->stop, t->go };
            if (WaitForMultipleObjects(2, hs, FALSE, delay) == WAIT_OBJECT_0) {
                return;
            }
            fn_warm(*m, t->stop);
        });
        LLAMA_LOG_INFO("%s: [TAG_FN_L15_WARM] the routed experts are read into the working set in the background after the first "
                "decode (working-set maximum %.1f GiB)\n", __func__, (ms.ullTotalPhys/10*9)/1073741824.0);
        return w;
    }
    if (!ml.use_mmap) {
        LLAMA_LOG_INFO("%s: [TAG_FN_L14_TIER] the model is not mapped: nothing to lock\n", __func__);
        return nullptr;
    }

    SYSTEM_INFO si;
    GetSystemInfo(&si);

    llama_fn_tier_ptr tier(new llama_fn_tier());
    tier->page = si.dwPageSize ? si.dwPageSize : 4096;
    // the host layers' experts (a layer whose three tensors are not all in host buffers, or differ in expert count, is skipped)
    int il = 0;
    for (const auto & ts : fn_tier_layers(model)) {
        llama_fn_tier::layer L;
        bool ok = ts.size() == 3;
        for (int s = 0; ok && s < 3; ++s) {
            const ggml_tensor * t = ts[s];
            ok = t->buffer && ggml_backend_buffer_is_host(t->buffer) && t->data && t->ne[2] >= 1 &&
                 (s == 0 || t->ne[2] == L.n_exp) && t->nb[2]*t->ne[2] == ggml_nbytes(t);
            if (ok) {
                L.base[s] = (char *) t->data;
                L.nb[s]   = t->nb[2];
                L.n_exp   = (int) t->ne[2];
            }
        }
        if (ok) {
            L.il = il;
            L.up = ts[0];
            L.lk.reset(new std::atomic<uint8_t>[(size_t) L.n_exp]());
            L.slot.assign((size_t) L.n_exp, -1);
            L.cnt.assign((size_t) L.n_exp, 0.0f);
            L.vol.assign((size_t) L.n_exp, 0);
            tier->by_up[L.up] = (int) tier->layers.size();
            tier->layers.push_back(std::move(L));
            for (int e = 0; e < tier->layers.back().n_exp; ++e) {
                tier->total += fn_tier_bytes(tier.get(), { (int) tier->layers.size() - 1, e });
            }
        }
        il++;
    }
    if (tier->layers.empty()) {
        LLAMA_LOG_INFO("%s: [TAG_FN_L14_TIER] no host layer with mapped routed experts\n", __func__);
        return nullptr;
    }
    // [TAG_FN_L15_TIER] the cap: 72 % of the RAM and at least 22 GiB left (the GPU driver locks memory too: its pinned
    // buffers and every kernel module it loads; with ~61 GiB locked during the load, module loads failed)
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    GlobalMemoryStatusEx(&ms);
    const int64_t ram = (int64_t) ms.ullTotalPhys;
    const int64_t def = std::max<int64_t>(0, std::min<int64_t>(ram*72/100, ram - ((int64_t) 22 << 30)));
    const int cap_mib = llama_fn_l3_int(model, "LLAMA_FN_L14_TIER_MAX_MIB", (int) (def >> 20));
    tier->cap       = (size_t) std::max(0, cap_mib) << 20;
    tier->free_min  = (size_t) std::max(0, llama_fn_l3_int(model, "LLAMA_FN_L14_TIER_FREE_MIB", 6144)) << 20;
    tier->low       = (size_t) std::max(0, llama_fn_l3_int(model, "LLAMA_FN_L14_TIER_LOW_MIB", 4096)) << 20;
    tier->delay_ms  = std::max(0, llama_fn_l3_int(model, "LLAMA_FN_L14_TIER_DELAY_MS", 20000));
    tier->n_fill    = std::max(1, std::min(16, llama_fn_l3_int(model, "LLAMA_FN_L14_TIER_THREADS", 8)));
    tier->commit_min = (size_t) std::max(0, llama_fn_l3_int(model, "LLAMA_FN_L15_TIER_COMMIT_MIB", 8192)) << 20;
    tier->stop      = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (tier->stop == nullptr) {
        return nullptr;
    }
    llama_moe_hot_set_host_ready(fn_tier_ready, tier.get());
    tier->worker = std::thread(fn_tier_worker, tier.get());
    LLAMA_LOG_INFO("%s: [TAG_FN_L15_TIER] %zu host layers (%.2f GiB of routed experts): from %.0f s after the load %d threads "
            "lock the experts the CPU computes, the volatile residents and the residents with the lowest counts, up to %.1f GiB "
            "while %.1f GiB of RAM stays available; then the hot set evicts only experts whose RAM copy is locked\n",
            __func__, tier->layers.size(), tier->total/1073741824.0, tier->delay_ms/1000.0, tier->n_fill, tier->cap/1073741824.0,
            tier->free_min/1073741824.0);
    return tier;
}

#endif

// [TAG_FN_L15_WARM] see llama-fn-tier.h
void llama_fn_warm_go() {
    static std::atomic<bool> done{false};
    if (done.load(std::memory_order_relaxed) || done.exchange(true)) {
        return;
    }
#ifdef _WIN32
    if (g_fn_warm_go != nullptr) {
        SetEvent(g_fn_warm_go);
    }
#endif
}

// [TAG_FN_L15_WARM] see llama-fn-tier.h
void llama_fn_warm_wait_layer(int il) {
#ifdef _WIN32
    if (!g_warm_running.load(std::memory_order_acquire) || il < 0 || il >= (int) g_warm_layer_end.size()) {
        return;
    }
    const size_t need = g_warm_layer_end[il];
    while (g_warm_running.load(std::memory_order_acquire) && g_warm_mark.load() < need) {
        Sleep(1);
    }
#else
    GGML_UNUSED(il);
#endif
}

bool llama_fn_warm_running() {
#ifdef _WIN32
    return g_warm_running.load(std::memory_order_acquire);
#else
    return false;
#endif
}
