// [TAG_FN_L14_TIER] see llama-fn-tier.h

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
#endif

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

} // namespace

struct llama_fn_tier {
    struct layer {
        int                  il      = -1;
        const ggml_tensor *  up      = nullptr; // the hot set's key
        char *               base[3] = {};      // up, gate, down
        size_t               nb[3]   = {};      // bytes of one expert in each
        int                  n_exp   = 0;
        std::vector<uint8_t> locked;
        std::vector<uint8_t> hot;               // the hot set holds it (last look)
    };
    struct item {
        int l;
        int e;
    };
    std::vector<layer>                 layers;
    std::unordered_map<uintptr_t, int> edge;     // first / last page of a locked slice: the locked slices that hold it
    std::mutex                         mtx;      // edge, the counters and the working-set size (fill threads)
    size_t                             page         = 4096;
    size_t                             total        = 0;
    size_t                             locked_bytes = 0;
    size_t                             reserved     = 0; // bytes a fill thread is locking right now
    size_t                             n_locked     = 0; // experts
    size_t                             cap          = 0;
    size_t                             free_min     = 0;
    size_t                             low          = 0;
    int                                delay_ms     = 0;
    int                                n_fill       = 4;
#ifdef _WIN32
    HANDLE                             stop = nullptr;
    std::thread                        worker;
#endif
};

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
// Only while no fill thread runs (the edge counts and the pages' lock state must agree).
void fn_tier_unlock(tier_t * t, const tier_t::item & it) {
    tier_t::layer & L = t->layers[it.l];
    if (!L.locked[it.e]) {
        return;
    }
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
    L.locked[it.e] = 0;
    t->locked_bytes -= b;
    t->n_locked--;
    fn_tier_ws_add(-(int64_t) b);
}

// lock items in order with n_fill threads, 32 experts at a time, while the cap, the available RAM and max_bytes allow.
// Each thread hints the read of its chunk, touches every page (the faults run in parallel) and locks it. Returns the bytes.
size_t fn_tier_fill(tier_t * t, const std::vector<tier_t::item> & items, size_t max_bytes) {
    constexpr size_t chunk = 32;
    using prefetch_t = BOOL (WINAPI *)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
    static const prefetch_t pf = (prefetch_t) (void *) GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "PrefetchVirtualMemory");

    std::atomic<size_t> next{0};
    std::atomic<bool>   full{false};
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
                if (t->locked_bytes + t->reserved + b > t->cap || added + b > max_bytes ||
                    fn_tier_avail() < t->free_min + nonres + t->reserved || !fn_tier_ws_add((int64_t) b)) {
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
                t->layers[items[i].l].locked[items[i].e] = 1;
                t->locked_bytes += sizes[i - c0];
                t->n_locked++;
            }
            t->reserved -= b;
            if (lost > 0) {
                fn_tier_ws_add(-(int64_t) lost);
                added -= lost;
                full.store(true); // the working-set quota or the RAM is short: stop here
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
        LLAMA_LOG_WARN("[TAG_FN_L14_TIER] %zu experts could not be locked (working-set quota or RAM)\n", failed);
    }
    return added;
}

// which experts the hot set holds now; a layer whose bookkeeping is busy keeps its last look
void fn_tier_look(tier_t * t) {
    std::vector<uint8_t> buf;
    for (auto & L : t->layers) {
        buf.assign((size_t) L.n_exp, 0);
        const int r = llama_moe_hot_residents(L.up, buf.data(), L.n_exp);
        if (r != 0) {
            L.hot = buf;
        }
    }
}

// the experts to lock: those the CPU computes (cpu = true) or those the hot set holds, not locked yet, in layer order
std::vector<tier_t::item> fn_tier_wanted(const tier_t * t, bool cpu) {
    std::vector<tier_t::item> res;
    for (int l = 0; l < (int) t->layers.size(); ++l) {
        const auto & L = t->layers[l];
        for (int e = 0; e < L.n_exp; ++e) {
            if (!L.locked[e] && (L.hot[e] == 0) == cpu) {
                res.push_back({ l, e });
            }
        }
    }
    return res;
}

// unlock about `bytes`: experts the hot set holds first, then the CPU's, from the last layer down
size_t fn_tier_release(tier_t * t, size_t bytes, bool cpu_too) {
    const size_t b0 = t->locked_bytes;
    for (int pass = 0; pass < (cpu_too ? 2 : 1) && b0 - t->locked_bytes < bytes; ++pass) {
        for (int l = (int) t->layers.size() - 1; l >= 0 && b0 - t->locked_bytes < bytes; --l) {
            auto & L = t->layers[l];
            for (int e = L.n_exp - 1; e >= 0 && b0 - t->locked_bytes < bytes; --e) {
                if (L.locked[e] && (pass == 1 || L.hot[e])) {
                    fn_tier_unlock(t, { l, e });
                }
            }
        }
    }
    return b0 - t->locked_bytes;
}

size_t fn_tier_sum(const tier_t * t, const std::vector<tier_t::item> & items) {
    size_t b = 0;
    for (const auto & it : items) {
        b += fn_tier_bytes(t, it);
    }
    return b;
}

// after the start delay (the context, its warm-up and the first kernels' loads come first): lock the experts the CPU
// computes, then the hot set's, up to the cap. Then every 2 s follow the hot set: a CPU expert not locked yet is locked,
// if need be in place of one the hot set holds; below `low` about 1 GiB goes back (the hot set's first); after a minute
// with room the hot set's experts are added again
void fn_tier_worker(tier_t * t) {
    if (WaitForSingleObject(t->stop, (DWORD) t->delay_ms) == WAIT_OBJECT_0) {
        return;
    }
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    const auto t0 = std::chrono::steady_clock::now();
    fn_tier_look(t);
    const auto   cpu   = fn_tier_wanted(t, true);
    const size_t b_cpu = fn_tier_fill(t, cpu, SIZE_MAX);
    const size_t b_hot = fn_tier_fill(t, fn_tier_wanted(t, false), SIZE_MAX);
    LLAMA_LOG_INFO("[TAG_FN_L14_TIER] locked in %.1f s: %.2f of the %.2f GiB of experts the CPU computes, %.2f GiB of the hot "
            "set's; %.2f of %.2f GiB locked (cap %.1f GiB), %.1f GiB RAM available\n",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), b_cpu/1073741824.0,
            fn_tier_sum(t, cpu)/1073741824.0, b_hot/1073741824.0, t->locked_bytes/1073741824.0, t->total/1073741824.0,
            t->cap/1073741824.0, fn_tier_avail()/1073741824.0);

    auto   healthy_since = std::chrono::steady_clock::now();
    size_t q_bytes = 0; // locked by the watcher since its last line
    auto   q_since = healthy_since;
    for (;;) {
        if (WaitForSingleObject(t->stop, 2000) == WAIT_OBJECT_0) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        if (fn_tier_avail() < t->low && t->n_locked > 0) {
            const size_t r = fn_tier_release(t, (size_t) 1 << 30, true);
            LLAMA_LOG_WARN("[TAG_FN_L14_TIER] %.1f GiB RAM available (below %.1f GiB): %.2f GiB unlocked, %.2f GiB stay locked\n",
                    fn_tier_avail()/1073741824.0, t->low/1073741824.0, r/1073741824.0, t->locked_bytes/1073741824.0);
            healthy_since = now;
            if (WaitForSingleObject(t->stop, 4000) == WAIT_OBJECT_0) {
                return;
            }
            continue;
        }
        fn_tier_look(t);
        const auto want = fn_tier_wanted(t, true);
        if (!want.empty()) {
            const size_t need = std::min<size_t>(fn_tier_sum(t, want), (size_t) 2 << 30);
            if (t->locked_bytes + need > t->cap) {
                fn_tier_release(t, t->locked_bytes + need - t->cap, false);
            }
            q_bytes += fn_tier_fill(t, want, (size_t) 2 << 30);
        } else if (now - healthy_since > std::chrono::seconds(60)) {
            q_bytes += fn_tier_fill(t, fn_tier_wanted(t, false), (size_t) 1 << 30);
            healthy_since = now;
        }
        if (q_bytes > 0 && now - q_since > std::chrono::seconds(60)) {
            LLAMA_LOG_INFO("[TAG_FN_L14_TIER] the last minute: %.2f GiB more locked as the hot set moved; %.2f of %.2f GiB locked\n",
                    q_bytes/1073741824.0, t->locked_bytes/1073741824.0, t->total/1073741824.0);
            q_bytes = 0;
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
#endif
    delete t;
}

void llama_fn_tier_plan(const llama_model & model, llama_model_loader & ml) {
    if (!fn_tier_on(model) || !ml.use_mmap) {
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
    LLAMA_LOG_INFO("%s: [TAG_FN_L14_TIER] %.2f GiB of routed experts kept out of the load's prefetch\n", __func__, n/1073741824.0);
}

#ifndef _WIN32

llama_fn_tier_ptr llama_fn_tier_build(llama_model & /*model*/, llama_model_loader & /*ml*/) {
    return nullptr;
}

#else

llama_fn_tier_ptr llama_fn_tier_build(llama_model & model, llama_model_loader & ml) {
    if (!fn_tier_on(model)) {
        return nullptr;
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
            L.locked.assign((size_t) L.n_exp, 0);
            L.hot.assign((size_t) L.n_exp, 0);
            tier->layers.push_back(L);
            for (int e = 0; e < L.n_exp; ++e) {
                tier->total += fn_tier_bytes(tier.get(), { (int) tier->layers.size() - 1, e });
            }
        }
        il++;
    }
    if (tier->layers.empty()) {
        LLAMA_LOG_INFO("%s: [TAG_FN_L14_TIER] no host layer with mapped routed experts\n", __func__);
        return nullptr;
    }
    // the cap: the GPU driver locks memory too (its pinned buffers, every kernel module it loads on first use); with ~61
    // GiB locked at the load, module loads failed ("shared object initialization failed"), and page-locked budgets past
    // ~55 GB of 96 GB fail on Windows. Default: 60 % of the RAM (~56 GiB here)
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    GlobalMemoryStatusEx(&ms);
    const int cap_mib = llama_fn_l3_int(model, "LLAMA_FN_L14_TIER_MAX_MIB", (int) (ms.ullTotalPhys*6/10 >> 20));
    tier->cap      = (size_t) std::max(0, cap_mib) << 20;
    tier->free_min = (size_t) std::max(0, llama_fn_l3_int(model, "LLAMA_FN_L14_TIER_FREE_MIB", 6144)) << 20;
    tier->low      = (size_t) std::max(0, llama_fn_l3_int(model, "LLAMA_FN_L14_TIER_LOW_MIB", 4096)) << 20;
    tier->delay_ms = std::max(0, llama_fn_l3_int(model, "LLAMA_FN_L14_TIER_DELAY_MS", 20000));
    tier->n_fill   = std::max(1, std::min(16, llama_fn_l3_int(model, "LLAMA_FN_L14_TIER_THREADS", 4)));
    tier->stop     = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (tier->stop == nullptr) {
        return nullptr;
    }
    tier->worker = std::thread(fn_tier_worker, tier.get());
    LLAMA_LOG_INFO("%s: [TAG_FN_L14_TIER] %zu host layers (%.2f GiB of routed experts): from %.0f s after the load %d threads "
            "lock the experts the CPU computes, then the hot set's, up to %.1f GiB while %.1f GiB of RAM stays available\n",
            __func__, tier->layers.size(), tier->total/1073741824.0, tier->delay_ms/1000.0, tier->n_fill, tier->cap/1073741824.0,
            tier->free_min/1073741824.0);
    return tier;
}

#endif
