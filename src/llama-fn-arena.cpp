// [TAG_FN_L14_ARENA] see llama-fn-arena.h

#include "llama-fn-arena.h"

#include "llama-fn-auto.h"
#include "llama-impl.h"
#include "llama-model-loader.h"
#include "llama-model.h"

#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <string>
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

struct llama_fn_arena {
    struct chunk {
        void * p      = nullptr;
        size_t size   = 0;
        bool   locked = false;
        bool   pinned = false;
    };
    std::vector<chunk>                 chunks;
    std::vector<ggml_backend_buffer_t> bufs;
    void (*unreg)(void *) = nullptr;
};

// the registered chunks of every arena, for llama_fn_arena_pinned
static std::mutex                                     g_fn_arena_mtx;
static std::vector<std::pair<const char *, size_t>>   g_fn_arena_pinned;

bool llama_fn_arena_pinned(const void * p, size_t n) {
    const char * c = (const char *) p;
    std::lock_guard<std::mutex> lk(g_fn_arena_mtx);
    for (const auto & r : g_fn_arena_pinned) {
        if (c >= r.first && c + n <= r.first + r.second) {
            return true;
        }
    }
    return false;
}

void llama_fn_arena_deleter::operator()(llama_fn_arena * a) const {
    if (a == nullptr) {
        return;
    }
    for (ggml_backend_buffer_t b : a->bufs) {
        ggml_backend_buffer_free(b);
    }
    for (auto & c : a->chunks) {
        if (c.pinned) {
            {
                std::lock_guard<std::mutex> lk(g_fn_arena_mtx);
                g_fn_arena_pinned.erase(std::remove_if(g_fn_arena_pinned.begin(), g_fn_arena_pinned.end(),
                        [&](const std::pair<const char *, size_t> & r) { return r.first == (const char *) c.p; }), g_fn_arena_pinned.end());
            }
            if (a->unreg) {
                a->unreg(c.p);
            }
        }
    }
#ifdef _WIN32
    for (auto & c : a->chunks) {
        if (c.locked) {
            VirtualUnlock(c.p, c.size);
        }
        VirtualFree(c.p, 0, MEM_RELEASE);
    }
#endif
    delete a;
}

#ifndef _WIN32

llama_fn_arena_ptr llama_fn_arena_build(llama_model & /*model*/, llama_model_loader & /*ml*/) {
    return nullptr;
}

#else

namespace {

constexpr size_t FN_ARENA_SECTOR = 4096;            // unbuffered reads: file offset, length and address in sectors
constexpr size_t FN_ARENA_PIECE  = (size_t) 64 << 20; // one read call

size_t fn_arena_up(size_t x, size_t a) {
    return (x + a - 1)/a*a;
}

struct fn_arena_item {
    ggml_tensor * t;
    int           file; // loader file index
    size_t        off;  // sector-aligned file offset of the read
    size_t        head; // tensor data starts head bytes after it
    size_t        len;  // sectors read
    size_t        at;   // offset in the layer's chunk
};

struct fn_arena_layer {
    int                        il;
    std::vector<fn_arena_item> items;
    size_t                     size  = 0;
    void *                     chunk = nullptr;
};

std::wstring fn_arena_wide(const std::string & s) {
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring w(n > 0 ? n - 1 : 0, L'\0');
    if (n > 1) {
        MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), n);
    }
    return w;
}

} // namespace

llama_fn_arena_ptr llama_fn_arena_build(llama_model & model, llama_model_loader & ml) {
    if (model.arch != LLM_ARCH_QWEN4EXP || !llama_fn_l3_flag(model, "LLAMA_FN_L14_ARENA")) {
        return nullptr;
    }
    if (!ml.use_mmap) {
        LLAMA_LOG_INFO("%s: [TAG_FN_L14_ARENA] the model is not mapped: nothing to move\n", __func__);
        return nullptr;
    }
    const int64_t t0 = ggml_time_us();

    // the host layers' routed experts, with their place in the files
    std::vector<fn_arena_layer> layers;
    size_t total = 0;
    const int n_layer = (int) std::min<size_t>(model.hparams.n_layer(), model.layers.size());
    for (int il = 0; il < n_layer; ++il) {
        const auto & L = model.layers[il];
        ggml_tensor * ts[3] = { L.ffn_up_exps, L.ffn_gate_exps, L.ffn_down_exps };
        fn_arena_layer s;
        s.il = il;
        bool ok = true;
        for (ggml_tensor * t : ts) {
            if (t == nullptr || t->buffer == nullptr || !ggml_backend_buffer_is_host(t->buffer) || t->extra != nullptr) {
                ok = false;
                break;
            }
            const auto it = ml.weights_map.find(ggml_get_name(t));
            if (it == ml.weights_map.end() || it->second.idx >= ml.files_paths.size() || ml.files_paths[it->second.idx].empty()) {
                ok = false;
                break;
            }
            const size_t offs = it->second.offs;
            fn_arena_item x;
            x.t    = t;
            x.file = it->second.idx;
            x.off  = offs/FN_ARENA_SECTOR*FN_ARENA_SECTOR;
            x.head = offs - x.off;
            x.len  = fn_arena_up(x.head + ggml_nbytes(t), FN_ARENA_SECTOR);
            x.at   = s.size;
            s.size += x.len;
            s.items.push_back(x);
        }
        if (ok) {
            total += s.size;
            layers.push_back(std::move(s));
        }
    }
    if (layers.empty()) {
        LLAMA_LOG_INFO("%s: [TAG_FN_L14_ARENA] no host layer with mapped routed experts\n", __func__);
        return nullptr;
    }

    // the budget: RAM available now (the model's cached pages count as available) less the reserve
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    GlobalMemoryStatusEx(&ms);
    const size_t reserve = (size_t) std::max(0, llama_fn_l3_int(model, "LLAMA_FN_L14_ARENA_RESERVE_MIB", 16384)) << 20;
    const size_t budget  = ms.ullAvailPhys > reserve ? (size_t) ms.ullAvailPhys - reserve : 0;
    size_t want = 0;
    size_t n_take = 0;
    while (n_take < layers.size() && want + layers[n_take].size <= budget) {
        want += layers[n_take].size;
        n_take++;
    }
    if (n_take == 0) {
        LLAMA_LOG_WARN("%s: [TAG_FN_L14_ARENA] %.1f GiB available, reserve %.1f GiB: no layer fits, the experts stay mapped\n",
                __func__, ms.ullAvailPhys/1073741824.0, reserve/1073741824.0);
        return nullptr;
    }
    layers.resize(n_take);

    // the working set has to hold the locked pages
    SIZE_T ws_min = 0;
    SIZE_T ws_max = 0;
    DWORD  ws_fl  = 0;
    GetProcessWorkingSetSizeEx(GetCurrentProcess(), &ws_min, &ws_max, &ws_fl);
    const size_t slack = (size_t) 64 << 20;
    if (!SetProcessWorkingSetSizeEx(GetCurrentProcess(), ws_min + want + slack, std::max<size_t>(ws_max, ws_min + want + slack) + want,
            QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE)) {
        LLAMA_LOG_WARN("%s: [TAG_FN_L14_ARENA] the working set cannot grow by %.1f GiB (error %lu): the experts stay mapped\n",
                __func__, want/1073741824.0, (unsigned long) GetLastError());
        return nullptr;
    }

    llama_fn_arena_ptr arena(new llama_fn_arena());
    for (auto & s : layers) {
        s.chunk = VirtualAlloc(nullptr, s.size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        if (s.chunk == nullptr) {
            LLAMA_LOG_WARN("%s: [TAG_FN_L14_ARENA] VirtualAlloc of %.0f MiB failed at layer %d (error %lu)\n", __func__,
                    s.size/1048576.0, s.il, (unsigned long) GetLastError());
            break;
        }
        arena->chunks.push_back({ s.chunk, s.size, false });
    }
    layers.resize(arena->chunks.size());
    if (layers.empty()) {
        SetProcessWorkingSetSizeEx(GetCurrentProcess(), ws_min, ws_max, ws_fl);
        return nullptr;
    }

    // unbuffered reads, pieces of 64 MiB over a few threads (each with its own handles)
    struct piece {
        int    file;
        size_t off;
        size_t len;
        char * dst;
    };
    std::vector<piece> pieces;
    for (auto & s : layers) {
        for (auto & x : s.items) {
            for (size_t o = 0; o < x.len; o += FN_ARENA_PIECE) {
                pieces.push_back({ x.file, x.off + o, std::min(FN_ARENA_PIECE, x.len - o), (char *) s.chunk + x.at + o });
            }
        }
    }
    const int n_thr = std::max(1, std::min(llama_fn_l3_int(model, "LLAMA_FN_L14_ARENA_THREADS", 8), 32));
    std::atomic<size_t> next{0};
    std::atomic<size_t> n_bytes{0};
    std::atomic<bool>   failed{false};
    std::atomic<unsigned long> err{0};
    auto reader = [&]() {
        std::vector<HANDLE> h(ml.files_paths.size(), INVALID_HANDLE_VALUE);
        while (!failed.load()) {
            const size_t i = next.fetch_add(1);
            if (i >= pieces.size()) {
                break;
            }
            const piece & p = pieces[i];
            if (h[p.file] == INVALID_HANDLE_VALUE) {
                h[p.file] = CreateFileW(fn_arena_wide(ml.files_paths[p.file]).c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                        OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
                if (h[p.file] == INVALID_HANDLE_VALUE) {
                    err = GetLastError();
                    failed = true;
                    break;
                }
            }
            size_t done = 0;
            while (done < p.len) {
                OVERLAPPED ov = {};
                const uint64_t pos = p.off + done;
                ov.Offset     = (DWORD) (pos & 0xffffffffu);
                ov.OffsetHigh = (DWORD) (pos >> 32);
                DWORD got = 0;
                if (!ReadFile(h[p.file], p.dst + done, (DWORD) (p.len - done), &got, &ov)) {
                    err = GetLastError();
                    failed = true;
                    break;
                }
                if (got == 0) {
                    break; // end of the file: the last sector of the last tensor
                }
                done += got;
            }
            n_bytes.fetch_add(done);
        }
        for (HANDLE x : h) {
            if (x != INVALID_HANDLE_VALUE) {
                CloseHandle(x);
            }
        }
    };
    std::vector<std::thread> thr;
    for (int i = 0; i < n_thr; ++i) {
        thr.emplace_back(reader);
    }
    for (auto & t : thr) {
        t.join();
    }
    if (failed) {
        LLAMA_LOG_WARN("%s: [TAG_FN_L14_ARENA] an unbuffered read failed (error %lu): the experts stay mapped\n", __func__, err.load());
        arena.reset();
        SetProcessWorkingSetSizeEx(GetCurrentProcess(), ws_min, ws_max, ws_fl);
        return nullptr;
    }
    const int64_t t_read = ggml_time_us() - t0;

    // the arena must hold the same bytes as the mapping: the first and last 4 KiB of every tensor (all of it with
    // LLAMA_FN_L14_ARENA_VERIFY=1, which reads the whole mapping once)
    const bool verify_all = llama_fn_l3_flag(model, "LLAMA_FN_L14_ARENA_VERIFY");
    for (auto & s : layers) {
        for (auto & x : s.items) {
            const char * a = (const char *) s.chunk + x.at + x.head;
            const char * m = (const char *) x.t->data;
            const size_t n = ggml_nbytes(x.t);
            const size_t k = std::min<size_t>(n, 4096);
            const bool same = verify_all ? memcmp(a, m, n) == 0 : (memcmp(a, m, k) == 0 && memcmp(a + n - k, m + n - k, k) == 0);
            if (!same) {
                LLAMA_LOG_ERROR("%s: [TAG_FN_L14_ARENA] %s differs from the mapped file: the experts stay mapped\n", __func__,
                        ggml_get_name(x.t));
                arena.reset();
                SetProcessWorkingSetSizeEx(GetCurrentProcess(), ws_min, ws_max, ws_fl);
                return nullptr;
            }
        }
    }

    // lock layer by layer; a layer that cannot be locked (and every one after it) stays mapped
    size_t n_locked = 0;
    for (; n_locked < layers.size(); ++n_locked) {
        auto & c = arena->chunks[n_locked];
        if (!VirtualLock(c.p, c.size)) {
            LLAMA_LOG_WARN("%s: [TAG_FN_L14_ARENA] VirtualLock of layer %d (%.0f MiB) failed (error %lu): it and the rest stay mapped\n",
                    __func__, layers[n_locked].il, c.size/1048576.0, (unsigned long) GetLastError());
            break;
        }
        c.locked = true;
    }
    for (size_t i = n_locked; i < arena->chunks.size(); ++i) {
        VirtualFree(arena->chunks[i].p, 0, MEM_RELEASE);
    }
    arena->chunks.resize(n_locked);
    layers.resize(n_locked);
    if (layers.empty()) {
        arena.reset();
        SetProcessWorkingSetSizeEx(GetCurrentProcess(), ws_min, ws_max, ws_fl);
        return nullptr;
    }

    // the tensors point into the arena now, each layer's chunk a host weight buffer of its own
    size_t moved = 0;
    size_t locked_bytes = 0;
    for (size_t i = 0; i < layers.size(); ++i) {
        auto & s = layers[i];
        ggml_backend_buffer_t buf = ggml_backend_cpu_buffer_from_ptr(s.chunk, s.size);
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        arena->bufs.push_back(buf);
        locked_bytes += s.size;
        for (auto & x : s.items) {
            x.t->buffer = buf;
            x.t->data   = (char *) s.chunk + x.at + x.head;
            moved += ggml_nbytes(x.t);
        }
    }

    // [TAG_FN_L14_ARENA] chunks registered for direct device copies (the prefill stream reads them without its staging
    // ring), at most LLAMA_FN_L14_ARENA_PIN_MIB (default 0: none). Registering all 58 GiB made the context's device
    // allocations fail (CUDA out of memory at the KV cache, nvlddmkm 153: gpu_faults.txt FAULT 2026-10-07 22:54:35):
    // the driver counts registered host memory against the device, so the budget stays off until a safe size is measured
    size_t n_pinned = 0;
    size_t pinned_bytes = 0;
    const int64_t t_pin0 = ggml_time_us();
    const size_t pin_budget = (size_t) std::max(0, llama_fn_l3_int(model, "LLAMA_FN_L14_ARENA_PIN_MIB", 0)) << 20;
    if (pin_budget > 0) {
        ggml_backend_reg_t reg = ggml_backend_reg_by_name("CUDA");
        auto reg_fn = reg ? (bool (*)(void *, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_fn_register_host") : nullptr;
        arena->unreg = reg ? (void (*)(void *)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_fn_unregister_host") : nullptr;
        if (reg_fn != nullptr && arena->unreg != nullptr) {
            for (auto & c : arena->chunks) {
                if (pinned_bytes + c.size > pin_budget || !reg_fn(c.p, c.size)) {
                    break;
                }
                c.pinned = true;
                n_pinned++;
                pinned_bytes += c.size;
                std::lock_guard<std::mutex> lk(g_fn_arena_mtx);
                g_fn_arena_pinned.emplace_back((const char *) c.p, c.size);
            }
        }
    }
    const double secs_pin = (ggml_time_us() - t_pin0)/1e6;

    GlobalMemoryStatusEx(&ms);
    const double secs = (ggml_time_us() - t0)/1e6;
    LLAMA_LOG_INFO("%s: [TAG_FN_L14_ARENA] %zu of %d host layers' routed experts (%.2f GiB) read into locked RAM in %.1f s "
            "(reads %.1f GB/s); %zu layer(s) (%.2f GiB) stay mapped; %.1f GiB RAM available now (reserve %.1f GiB)\n", __func__,
            layers.size(), n_layer, moved/1073741824.0, secs, n_bytes.load()/1e9/std::max(1e-6, t_read/1e6),
            (size_t) n_layer - layers.size(), (total - locked_bytes)/1073741824.0, ms.ullAvailPhys/1073741824.0,
            reserve/1073741824.0);
    LLAMA_LOG_INFO("%s: [TAG_FN_L14_ARENA] %zu of %zu arena layers (%.2f GiB) registered for direct device copies in %.1f s\n",
            __func__, n_pinned, arena->chunks.size(), pinned_bytes/1073741824.0, secs_pin);
    return arena;
}

#endif
