// [TAG_FN_L14_TIER] see llama-fn-tier.h

#include "llama-fn-tier.h"

#include "llama-fn-auto.h"
#include "llama-impl.h"
#include "llama-model-loader.h"
#include "llama-model.h"

#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
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
    std::vector<std::vector<range>> locked; // per locked layer, in lock order
    size_t                          bytes = 0;
    std::mutex                      mtx;
#ifdef _WIN32
    HANDLE                          low   = nullptr;
    HANDLE                          stop  = nullptr;
    std::thread                     watcher;
#endif
};

void llama_fn_tier_deleter::operator()(llama_fn_tier * t) const {
    if (t == nullptr) {
        return;
    }
#ifdef _WIN32
    if (t->stop) {
        SetEvent(t->stop);
    }
    if (t->watcher.joinable()) {
        t->watcher.join();
    }
    for (auto & L : t->locked) {
        for (auto & r : L) {
            VirtualUnlock(r.lo, r.n);
        }
    }
    if (t->low) {
        CloseHandle(t->low);
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
    const int64_t t0 = ggml_time_us();

    SYSTEM_INFO si;
    GetSystemInfo(&si);
    const size_t page = si.dwPageSize ? si.dwPageSize : 4096;

    // the host layers' expert ranges, page-aligned (a layer whose experts are not all in host buffers is skipped)
    std::vector<std::vector<llama_fn_tier::range>> layers;
    std::vector<int> layer_il;
    size_t total = 0;
    int il = 0;
    for (const auto & ts : fn_tier_layers(model)) {
        std::vector<llama_fn_tier::range> rs;
        bool ok = ts.size() == 3;
        for (ggml_tensor * t : ts) {
            ok = ok && t->buffer && ggml_backend_buffer_is_host(t->buffer) && t->data;
            if (!ok) {
                break;
            }
            const uintptr_t a  = (uintptr_t) t->data;
            const uintptr_t lo = a/page*page;
            const uintptr_t hi = (a + ggml_nbytes(t) + page - 1)/page*page;
            rs.push_back({ (char *) lo, (size_t) (hi - lo) });
        }
        if (ok) {
            size_t s = 0;
            for (auto & r : rs) {
                s += r.n;
            }
            total += s;
            layers.push_back(rs);
            layer_il.push_back(il);
        }
        il++;
    }
    if (layers.empty()) {
        LLAMA_LOG_INFO("%s: [TAG_FN_L14_TIER] no host layer with mapped routed experts\n", __func__);
        return nullptr;
    }

    // the budget: RAM available now (cached file pages count as available) less the reserve
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    GlobalMemoryStatusEx(&ms);
    const size_t reserve = (size_t) std::max(0, llama_fn_l3_int(model, "LLAMA_FN_L14_TIER_RESERVE_MIB", 12288)) << 20;
    const size_t budget  = ms.ullAvailPhys > reserve ? (size_t) ms.ullAvailPhys - reserve : 0;
    size_t want   = 0;
    size_t n_take = 0;
    for (; n_take < layers.size(); ++n_take) {
        size_t s = 0;
        for (auto & r : layers[n_take]) {
            s += r.n;
        }
        if (want + s > budget) {
            break;
        }
        want += s;
    }
    if (n_take == 0) {
        LLAMA_LOG_WARN("%s: [TAG_FN_L14_TIER] %.1f GiB available, reserve %.1f GiB: no layer fits, nothing locked\n", __func__,
                ms.ullAvailPhys/1073741824.0, reserve/1073741824.0);
        return nullptr;
    }

    SIZE_T ws_min = 0;
    SIZE_T ws_max = 0;
    DWORD  ws_fl  = 0;
    GetProcessWorkingSetSizeEx(GetCurrentProcess(), &ws_min, &ws_max, &ws_fl);
    const size_t slack = (size_t) 64 << 20;
    if (!SetProcessWorkingSetSizeEx(GetCurrentProcess(), ws_min + want + slack, std::max<size_t>(ws_max, ws_min + want + slack) + want,
            QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE)) {
        LLAMA_LOG_WARN("%s: [TAG_FN_L14_TIER] the working set cannot grow by %.1f GiB (error %lu): nothing locked\n", __func__,
                want/1073741824.0, (unsigned long) GetLastError());
        return nullptr;
    }

    // one asynchronous prefetch of every selected range (large reads), then the locks in layer order: a lock waits only
    // for the pages of its layer that the prefetch has not brought in yet
    using prefetch_t = BOOL (WINAPI *)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
    const prefetch_t pf = (prefetch_t) (void *) GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "PrefetchVirtualMemory");
    if (pf != nullptr) {
        std::vector<WIN32_MEMORY_RANGE_ENTRY> e;
        for (size_t i = 0; i < n_take; ++i) {
            for (auto & r : layers[i]) {
                WIN32_MEMORY_RANGE_ENTRY x;
                x.VirtualAddress = r.lo;
                x.NumberOfBytes  = r.n;
                e.push_back(x);
            }
        }
        pf(GetCurrentProcess(), (ULONG_PTR) e.size(), e.data(), 0);
    }

    llama_fn_tier_ptr tier(new llama_fn_tier());
    for (size_t i = 0; i < n_take; ++i) {
        bool ok = true;
        size_t k = 0;
        for (; k < layers[i].size(); ++k) {
            if (!VirtualLock(layers[i][k].lo, layers[i][k].n)) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            LLAMA_LOG_WARN("%s: [TAG_FN_L14_TIER] VirtualLock failed at layer %d (error %lu): it and the rest stay unlocked\n",
                    __func__, layer_il[i], (unsigned long) GetLastError());
            for (size_t j = 0; j < k; ++j) {
                VirtualUnlock(layers[i][j].lo, layers[i][j].n);
            }
            break;
        }
        for (auto & r : layers[i]) {
            tier->bytes += r.n;
        }
        tier->locked.push_back(layers[i]);
    }
    if (tier->locked.empty()) {
        SetProcessWorkingSetSizeEx(GetCurrentProcess(), ws_min, ws_max, ws_fl);
        return nullptr;
    }

    // low memory: unlock the last locked layer, one at a time, while Windows keeps signalling (its pages go to the
    // standby list: no I/O, and the decode then reads them through the file cache as before)
    tier->low  = CreateMemoryResourceNotification(LowMemoryResourceNotification);
    tier->stop = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (tier->low && tier->stop) {
        llama_fn_tier * tp = tier.get();
        tier->watcher = std::thread([tp]() {
            HANDLE hs[2] = { tp->stop, tp->low };
            for (;;) {
                const DWORD w = WaitForMultipleObjects(2, hs, FALSE, INFINITE);
                if (w != WAIT_OBJECT_0 + 1) {
                    return;
                }
                {
                    std::lock_guard<std::mutex> lk(tp->mtx);
                    if (tp->locked.empty()) {
                        return;
                    }
                    size_t freed = 0;
                    for (auto & r : tp->locked.back()) {
                        VirtualUnlock(r.lo, r.n);
                        freed += r.n;
                    }
                    tp->locked.pop_back();
                    tp->bytes -= std::min(tp->bytes, freed);
                    LLAMA_LOG_WARN("[TAG_FN_L14_TIER] low memory: one layer of experts unlocked (%.2f GiB), %zu layers / %.2f GiB stay locked\n",
                            freed/1073741824.0, tp->locked.size(), tp->bytes/1073741824.0);
                }
                if (WaitForSingleObject(tp->stop, 1000) == WAIT_OBJECT_0) {
                    return;
                }
            }
        });
    }

    GlobalMemoryStatusEx(&ms);
    LLAMA_LOG_INFO("%s: [TAG_FN_L14_TIER] %zu of %zu host layers' routed experts (%.2f GiB) locked in place in %.1f s; %.2f GiB stay "
            "unlocked; %.1f GiB RAM available now (reserve %.1f GiB)\n", __func__, tier->locked.size(), layers.size(),
            tier->bytes/1073741824.0, (ggml_time_us() - t0)/1e6, (total - tier->bytes)/1073741824.0, ms.ullAvailPhys/1073741824.0,
            reserve/1073741824.0);
    return tier;
}

#endif
