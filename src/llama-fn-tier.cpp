// [TAG_FN_L14_TIER] see llama-fn-tier.h

#include "llama-fn-tier.h"

#include "llama-fn-auto.h"
#include "llama-impl.h"
#include "llama-model-loader.h"
#include "llama-model.h"

#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
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
    struct range {
        char * lo;
        size_t n;
    };
    std::vector<std::vector<range>> layers;      // candidates, in lock order
    std::vector<int>                layer_il;
    std::vector<size_t>             layer_bytes;
    size_t                          total        = 0;
    size_t                          n_locked     = 0; // layers [0, n_locked) are locked
    size_t                          locked_bytes = 0;
    size_t                          reserve      = 0;
    size_t                          low          = 0;
    int                             delay_ms     = 0;
#ifdef _WIN32
    HANDLE                          stop = nullptr;
    std::thread                     worker;
#endif
};

#ifdef _WIN32

namespace {

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

// lock candidate layer t->n_locked if the RAM above the reserve allows it
bool fn_tier_lock_next(llama_fn_tier * t) {
    if (t->n_locked >= t->layers.size()) {
        return false;
    }
    const size_t i = t->n_locked;
    const size_t b = t->layer_bytes[i];
    if (fn_tier_avail() < b + t->reserve) {
        return false;
    }
    if (!fn_tier_ws_add((int64_t) b)) {
        return false;
    }
    using prefetch_t = BOOL (WINAPI *)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
    static const prefetch_t pf = (prefetch_t) (void *) GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "PrefetchVirtualMemory");
    if (pf != nullptr) {
        std::vector<WIN32_MEMORY_RANGE_ENTRY> e;
        for (auto & r : t->layers[i]) {
            WIN32_MEMORY_RANGE_ENTRY x;
            x.VirtualAddress = r.lo;
            x.NumberOfBytes  = r.n;
            e.push_back(x);
        }
        pf(GetCurrentProcess(), (ULONG_PTR) e.size(), e.data(), 0);
    }
    size_t k = 0;
    for (; k < t->layers[i].size(); ++k) {
        if (!VirtualLock(t->layers[i][k].lo, t->layers[i][k].n)) {
            break;
        }
    }
    if (k < t->layers[i].size()) {
        for (size_t j = 0; j < k; ++j) {
            VirtualUnlock(t->layers[i][j].lo, t->layers[i][j].n);
        }
        fn_tier_ws_add(-(int64_t) b);
        return false;
    }
    t->n_locked++;
    t->locked_bytes += b;
    return true;
}

void fn_tier_unlock_last(llama_fn_tier * t) {
    if (t->n_locked == 0) {
        return;
    }
    const size_t i = --t->n_locked;
    for (auto & r : t->layers[i]) {
        VirtualUnlock(r.lo, r.n);
    }
    t->locked_bytes -= t->layer_bytes[i];
    fn_tier_ws_add(-(int64_t) t->layer_bytes[i]);
}

// after the start delay (the context, its warm-up and the first kernels' loads come first): lock layer by layer at
// background priority while the RAM above the reserve allows; then watch: below `low` one layer goes back every 5 s at
// most, and after a minute with room above the reserve one more layer is locked
void fn_tier_worker(llama_fn_tier * t) {
    if (WaitForSingleObject(t->stop, (DWORD) t->delay_ms) == WAIT_OBJECT_0) {
        return;
    }
    SetThreadPriority(GetCurrentThread(), THREAD_MODE_BACKGROUND_BEGIN);
    const auto t0 = std::chrono::steady_clock::now();
    while (WaitForSingleObject(t->stop, 0) != WAIT_OBJECT_0 && fn_tier_lock_next(t)) {
        WaitForSingleObject(t->stop, 20);
    }
    LLAMA_LOG_INFO("[TAG_FN_L14_TIER] %zu of %zu host layers' routed experts (%.2f GiB of %.2f GiB) locked in place in %.1f s; "
            "%.1f GiB RAM available (reserve %.1f GiB)\n", t->n_locked, t->layers.size(), t->locked_bytes/1073741824.0,
            t->total/1073741824.0, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
            fn_tier_avail()/1073741824.0, t->reserve/1073741824.0);
    SetThreadPriority(GetCurrentThread(), THREAD_MODE_BACKGROUND_END);

    auto healthy_since = std::chrono::steady_clock::now();
    for (;;) {
        if (WaitForSingleObject(t->stop, 1000) == WAIT_OBJECT_0) {
            return;
        }
        const size_t a   = fn_tier_avail();
        const auto   now = std::chrono::steady_clock::now();
        if (a < t->low && t->n_locked > 0) {
            fn_tier_unlock_last(t);
            LLAMA_LOG_WARN("[TAG_FN_L14_TIER] %.1f GiB RAM available (below %.1f GiB): one layer of experts unlocked, %zu layers / "
                    "%.2f GiB stay locked\n", a/1073741824.0, t->low/1073741824.0, t->n_locked, t->locked_bytes/1073741824.0);
            healthy_since = now;
            if (WaitForSingleObject(t->stop, 4000) == WAIT_OBJECT_0) {
                return;
            }
            continue;
        }
        const bool room = t->n_locked < t->layers.size() &&
                          a > t->reserve + t->layer_bytes[t->n_locked] + ((size_t) 8 << 30);
        if (!room) {
            healthy_since = now;
            continue;
        }
        if (now - healthy_since > std::chrono::seconds(60)) {
            if (fn_tier_lock_next(t)) {
                LLAMA_LOG_INFO("[TAG_FN_L14_TIER] room again: one more layer of experts locked, %zu layers / %.2f GiB locked\n",
                        t->n_locked, t->locked_bytes/1073741824.0);
            }
            healthy_since = now;
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
    while (t->n_locked > 0) {
        fn_tier_unlock_last(t);
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
    const size_t page = si.dwPageSize ? si.dwPageSize : 4096;

    llama_fn_tier_ptr tier(new llama_fn_tier());
    // the host layers' expert ranges, page-aligned (a layer whose experts are not all in host buffers is skipped)
    int il = 0;
    for (const auto & ts : fn_tier_layers(model)) {
        std::vector<llama_fn_tier::range> rs;
        bool ok = ts.size() == 3;
        size_t b = 0;
        for (ggml_tensor * t : ts) {
            ok = ok && t->buffer && ggml_backend_buffer_is_host(t->buffer) && t->data;
            if (!ok) {
                break;
            }
            const uintptr_t a  = (uintptr_t) t->data;
            const uintptr_t lo = a/page*page;
            const uintptr_t hi = (a + ggml_nbytes(t) + page - 1)/page*page;
            rs.push_back({ (char *) lo, (size_t) (hi - lo) });
            b += (size_t) (hi - lo);
        }
        if (ok) {
            tier->layers.push_back(rs);
            tier->layer_il.push_back(il);
            tier->layer_bytes.push_back(b);
            tier->total += b;
        }
        il++;
    }
    if (tier->layers.empty()) {
        LLAMA_LOG_INFO("%s: [TAG_FN_L14_TIER] no host layer with mapped routed experts\n", __func__);
        return nullptr;
    }
    // the reserve: the GPU driver locks memory too (its pinned buffers, every kernel module it loads on first use); with
    // ~60 GiB locked at load and ~12 GiB left, module loads failed ("shared object initialization failed")
    tier->reserve  = (size_t) std::max(0, llama_fn_l3_int(model, "LLAMA_FN_L14_TIER_RESERVE_MIB", 24576)) << 20;
    tier->low      = (size_t) std::max(0, llama_fn_l3_int(model, "LLAMA_FN_L14_TIER_LOW_MIB", 4096)) << 20;
    tier->delay_ms = std::max(0, llama_fn_l3_int(model, "LLAMA_FN_L14_TIER_DELAY_MS", 20000));
    tier->stop     = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (tier->stop == nullptr) {
        return nullptr;
    }
    tier->worker = std::thread(fn_tier_worker, tier.get());
    LLAMA_LOG_INFO("%s: [TAG_FN_L14_TIER] %zu host layers (%.2f GiB of routed experts) are locked in place in the background "
            "from %.0f s after the load, above a %.1f GiB RAM reserve\n", __func__, tier->layers.size(), tier->total/1073741824.0,
            tier->delay_ms/1000.0, tier->reserve/1073741824.0);
    return tier;
}

#endif
