// [TAG_FN_L22_PIN] moe-bridge-probe test 7 (--only pin): can the GPU driver pin the model file's own pages, how much,
// and is it safe (E:/turbot-gates/flashnext/test/r22/PLAN.md P6.1). Runs only when asked (the round-22 test's pin
// block, after every non-pin stage). It never provokes an out-of-memory: every device allocation is made only while the
// DXGI LOCAL headroom covers it plus 1 GiB, and the registration ladder stops at the NON_LOCAL budget - usage - margin.
//   1. CPU only: DXGI LOCAL and NON_LOCAL budget / usage of the adapter whose LUID is the CUDA device's, commit,
//      available RAM, standby list (when the system allows the query)
//   2. registration ladder 4 -> 8 -> 16 -> 32 GiB with the unit the server uses (per-expert coalesced ranges,
//      src/llama-fn-pin-core.h) over the expert tensors of the shard(s); per step: time, ranges, time per range,
//      commit, NON_LOCAL usage, LOCAL usage (the device charge per registered GiB), working-set Valid share; then the
//      real allocation test (FULL reserve, borrow through cuMemCreate + cuMemMap, vision lend; memset + synchronize;
//      sizes are arguments) only within the LOCAL headroom
//   3. lock test without memory pressure: two equal resident ranges R (registered) and U (not); VirtualUnlock trims
//      each from the working set; locked iff R's available-RAM gain < 10 % of |R| while U's > 50 % of |U|; then a DMA
//      round trip of R compared byte for byte
//   4. touched vs untouched: register a range this process touched and one it never touched; time, lock, LOCAL charge
//   5. unregister all: NON_LOCAL and LOCAL usage must come back; the ladder once more
// Verdict: FILE_OK (mode untouched / touched, budgets, LOCAL charge per GiB), FILE_UNLOCKED (use the arena), FAIL.
//
//   moe-bridge-probe --only pin [--pin-files a.gguf,b.gguf] [--pin-ladder 4,8,16,32] [--pin-hot-every 0]
//                    [--alloc-full-mib N] [--alloc-borrow-mib N] [--alloc-vision-mib N] [--lock-mib 1024]
//                    [--pin-margin-mib 8192] [--pin-floor-mib 6144] [--no-repeat] [--pin-mapped 1] [--pin-range-mib 256]

#include "moe-bridge-pin.h"

#include "../../src/llama-fn-pin-core.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
#    include <psapi.h>
#    include <dxgi1_4.h>
#endif

namespace {

constexpr uint64_t MIB = 1ull << 20;
constexpr uint64_t GIB = 1ull << 30;

pin_say_fn g_say = nullptr;

#define SAY(...) g_say(__VA_ARGS__)

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

double gib(uint64_t b) {
    return (double) b/(double) GIB;
}

double gib_s(int64_t b) {
    return (double) b/(double) GIB;
}

#ifdef _WIN32

// ---------------------------------------------------------------------------------------------------------------------
// measurements

struct dxgi_ad {
    IDXGIAdapter3 * ad = nullptr;
    bool            ok = false;
};

dxgi_ad g_ad;

bool dxgi_open(uint64_t luid) {
    HMODULE lib = LoadLibraryA("dxgi.dll");
    using create_t = HRESULT (WINAPI *)(REFIID, void **);
    create_t create = lib ? (create_t) (void *) GetProcAddress(lib, "CreateDXGIFactory1") : nullptr;
    IDXGIFactory1 * fac = nullptr;
    if (!create || FAILED(create(__uuidof(IDXGIFactory1), (void **) &fac)) || !fac) {
        return false;
    }
    IDXGIAdapter1 * a1 = nullptr;
    for (UINT i = 0; fac->EnumAdapters1(i, &a1) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 d = {};
        uint64_t l = 0;
        if (SUCCEEDED(a1->GetDesc1(&d))) {
            memcpy(&l, &d.AdapterLuid, sizeof(l));
        }
        IDXGIAdapter3 * a3 = nullptr;
        if (l == luid && SUCCEEDED(a1->QueryInterface(__uuidof(IDXGIAdapter3), (void **) &a3))) {
            g_ad.ad = a3;
            g_ad.ok = true;
        }
        a1->Release();
        if (g_ad.ok) {
            break;
        }
    }
    fac->Release();
    return g_ad.ok;
}

struct meas {
    uint64_t nl_b = 0, nl_u = 0, l_b = 0, l_u = 0;
    bool     dx = false;
    uint64_t avail = 0, commit = 0, commit_limit = 0, standby = 0;
    bool     standby_ok = false;
    int64_t  l_head() const { return (int64_t) l_b - (int64_t) l_u; }
    int64_t  nl_head() const { return (int64_t) nl_b - (int64_t) nl_u; }
};

struct mem_list {
    ULONG_PTR ZeroPageCount;
    ULONG_PTR FreePageCount;
    ULONG_PTR ModifiedPageCount;
    ULONG_PTR ModifiedNoWritePageCount;
    ULONG_PTR BadPageCount;
    ULONG_PTR PageCountByPriority[8];
    ULONG_PTR RepurposedPagesByPriority[8];
    ULONG_PTR ModifiedPageCountPageFile;
};

meas measure() {
    meas m;
    if (g_ad.ok) {
        DXGI_QUERY_VIDEO_MEMORY_INFO a = {};
        DXGI_QUERY_VIDEO_MEMORY_INFO b = {};
        if (SUCCEEDED(g_ad.ad->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &a)) &&
            SUCCEEDED(g_ad.ad->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &b))) {
            m.dx   = true;
            m.l_b  = a.Budget;
            m.l_u  = a.CurrentUsage;
            m.nl_b = b.Budget;
            m.nl_u = b.CurrentUsage;
        }
    }
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) {
        m.avail = ms.ullAvailPhys;
    }
    PERFORMANCE_INFORMATION pi;
    pi.cb = sizeof(pi);
    if (GetPerformanceInfo(&pi, sizeof(pi))) {
        m.commit       = (uint64_t) pi.CommitTotal*pi.PageSize;
        m.commit_limit = (uint64_t) pi.CommitLimit*pi.PageSize;
    }
    using q_t = LONG (WINAPI *)(ULONG, PVOID, ULONG, PULONG);
    static const q_t q = (q_t) (void *) GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtQuerySystemInformation");
    if (q) {
        mem_list ml = {};
        ULONG got = 0;
        if (q(80 /* SystemMemoryListInformation */, &ml, sizeof(ml), &got) >= 0) {
            uint64_t s = 0;
            for (int i = 0; i < 8; ++i) {
                s += ml.PageCountByPriority[i];
            }
            m.standby    = s*4096;
            m.standby_ok = true;
        }
    }
    return m;
}

// the working-set Valid share of ranges, one page per 2 MiB
double valid_share(const std::vector<std::pair<const uint8_t *, size_t>> & rs) {
    std::vector<PSAPI_WORKING_SET_EX_INFORMATION> w;
    for (const auto & r : rs) {
        for (size_t o = 0; o < r.second; o += (size_t) 2 << 20) {
            PSAPI_WORKING_SET_EX_INFORMATION x = {};
            x.VirtualAddress = (PVOID) ((uintptr_t) (r.first + o) & ~(uintptr_t) 4095);
            w.push_back(x);
        }
    }
    if (w.empty()) {
        return 0.0;
    }
    size_t valid = 0;
    for (size_t i = 0; i < w.size(); i += 65536) {
        const size_t n = std::min<size_t>(65536, w.size() - i);
        if (!QueryWorkingSetEx(GetCurrentProcess(), w.data() + i, (DWORD) (n*sizeof(w[0])))) {
            return -1.0;
        }
    }
    for (const auto & x : w) {
        valid += x.VirtualAttributes.Valid ? 1 : 0;
    }
    return (double) valid/(double) w.size();
}

double valid_share1(const uint8_t * p, size_t n) {
    return valid_share({ { p, n } });
}

void touch(const uint8_t * p, size_t n) {
    for (size_t o = 0; o < n; o += 4096) {
        (void) *(volatile const uint8_t *) (p + o);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// shards

struct shard {
    std::string     path;
    HANDLE          f = INVALID_HANDLE_VALUE;
    HANDLE          m = nullptr;
    const uint8_t * v = nullptr;
    size_t          size = 0;
};

bool shard_open(shard & s) {
    s.f = CreateFileA(s.path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (s.f == INVALID_HANDLE_VALUE) {
        return false;
    }
    LARGE_INTEGER sz;
    GetFileSizeEx(s.f, &sz);
    s.size = (size_t) sz.QuadPart;
    s.m = CreateFileMappingA(s.f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    s.v = s.m ? (const uint8_t *) MapViewOfFile(s.m, FILE_MAP_READ, 0, 0, 0) : nullptr;
    return s.v != nullptr;
}

void shard_close(shard & s) {
    if (s.v) {
        UnmapViewOfFile(s.v);
    }
    if (s.m) {
        CloseHandle(s.m);
    }
    if (s.f != INVALID_HANDLE_VALUE) {
        CloseHandle(s.f);
    }
    s = shard();
}

// the routed-expert tensors of a shard as the server's unit sees them
void shard_layers(const shard & s, int hot_every, std::vector<fn_pin::layer_in> & out) {
    ggml_context * gctx = nullptr;
    gguf_init_params ip = { true, &gctx };
    gguf_context * g = gguf_init_from_file(s.path.c_str(), ip);
    if (g == nullptr) {
        SAY("  %s: not a GGUF file\n", s.path.c_str());
        return;
    }
    const size_t data_off = gguf_get_data_offset(g);
    std::vector<fn_pin::layer_in> ls;
    const int64_t n_t = gguf_get_n_tensors(g);
    for (int64_t i = 0; i < n_t; ++i) {
        const char * name = gguf_get_tensor_name(g, i);
        int il = -1;
        char part[64] = {};
        if (sscanf(name, "blk.%d.%63[^.]", &il, part) != 2) {
            continue;
        }
        const int k = strcmp(part, "ffn_up_exps") == 0 ? 0 : strcmp(part, "ffn_gate_exps") == 0 ? 1 : strcmp(part, "ffn_down_exps") == 0 ? 2 : -1;
        if (k < 0) {
            continue;
        }
        const ggml_tensor * t = ggml_get_tensor(gctx, name);
        if (t == nullptr || t->ne[2] <= 0) {
            continue;
        }
        const size_t off = data_off + gguf_get_tensor_offset(g, i);
        if (off + ggml_nbytes(t) > s.size) {
            continue;
        }
        auto it = std::find_if(ls.begin(), ls.end(), [&](const fn_pin::layer_in & L) { return L.il == il; });
        if (it == ls.end()) {
            ls.emplace_back();
            ls.back().il = il;
            it = ls.end() - 1;
        }
        it->t[k].data     = s.v + off;
        it->t[k].stride   = t->nb[2];
        it->t[k].n_expert = (int) t->ne[2];
    }
    for (auto & L : ls) {
        int n_exp = 0;
        for (int k = 0; k < 3; ++k) {
            if (L.t[k].data) {
                n_exp = n_exp ? std::min(n_exp, L.t[k].n_expert) : L.t[k].n_expert;
            }
        }
        L.ok.assign(n_exp, 1);
        for (int e = 0; hot_every > 0 && e < n_exp; ++e) {
            L.ok[e] = e % hot_every != 0 ? 1 : 0;
        }
        L.fully_streamed = hot_every <= 0;
        out.push_back(L);
    }
    gguf_free(g);
    if (gctx) {
        ggml_free(gctx);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// registration

struct reg_entry {
    const uint8_t * p;
    size_t          n;
};

bool g_sticky = false; // a CUDA error other than out-of-memory: stop all device work
bool g_mapped = true;  // cudaHostRegisterMapped with the server's flags (LLAMA_FN_L22_DMASHARE registers mapped)

bool cuda_ok(cudaError_t e, const char * what) {
    if (e == cudaSuccess) {
        return true;
    }
    (void) cudaGetLastError();
    SAY("  CUDA %s: %s (%s)\n", what, cudaGetErrorName(e), cudaGetErrorString(e));
    if (e != cudaErrorMemoryAllocation && e != cudaErrorHostMemoryAlreadyRegistered && e != cudaErrorInvalidValue &&
        e != cudaErrorNotSupported) {
        g_sticky = true;
    }
    return false;
}

bool reg(const uint8_t * p, size_t n, double * ms) {
    const double t0 = now_s();
    const cudaError_t e = cudaHostRegister((void *) p, n, cudaHostRegisterPortable | cudaHostRegisterReadOnly | (g_mapped ? cudaHostRegisterMapped : 0));
    if (ms) {
        *ms = (now_s() - t0)*1e3;
    }
    return cuda_ok(e, "cudaHostRegister");
}

void unreg(const uint8_t * p) {
    const cudaError_t e = cudaHostUnregister((void *) p);
    if (e != cudaSuccess) {
        (void) cudaGetLastError();
        SAY("  cudaHostUnregister: %s\n", cudaGetErrorName(e));
    }
}

// one device allocation test (kind 0 cudaMalloc, 1 cuMemCreate + cuMemMap), memset, synchronize; kept until free_all
struct alloc_set {
    std::vector<void *> plain;
    struct vmm {
        CUdeviceptr                  va = 0;
        size_t                       sz = 0;
        CUmemGenericAllocationHandle h  = 0;
    };
    std::vector<vmm> maps;

    void free_all() {
        for (void * p : plain) {
            cudaFree(p);
        }
        plain.clear();
        for (auto & m : maps) {
            cuMemUnmap(m.va, m.sz);
            cuMemRelease(m.h);
            cuMemAddressFree(m.va, m.sz);
        }
        maps.clear();
        cudaDeviceSynchronize();
    }
};

// 1 done, 0 skipped (headroom), -1 failed
int alloc_test(alloc_set & as, const char * what, size_t mib, int kind) {
    if (mib == 0) {
        return 0;
    }
    const size_t sz = mib*MIB;
    const meas m = measure();
    if (!m.dx || m.l_head() < (int64_t) (sz + GIB)) {
        SAY("    %s %zu MiB: skipped (LOCAL headroom %.2f GiB < size + 1 GiB)\n", what, mib, gib_s(m.l_head()));
        return 0;
    }
    const double t0 = now_s();
    if (kind == 0) {
        void * p = nullptr;
        if (!cuda_ok(cudaMalloc(&p, sz), "cudaMalloc")) {
            return -1;
        }
        as.plain.push_back(p);
        if (!cuda_ok(cudaMemset(p, 0, sz), "cudaMemset") || !cuda_ok(cudaStreamSynchronize(0), "synchronize")) {
            return -1;
        }
    } else {
        int dev = 0;
        cudaGetDevice(&dev);
        CUmemAllocationProp prop = {};
        prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id   = dev;
        size_t gran = 0;
        if (cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS || gran == 0) {
            SAY("    %s: no VMM granularity\n", what);
            return 0;
        }
        alloc_set::vmm v;
        v.sz = (sz + gran - 1)/gran*gran;
        CUresult r = cuMemCreate(&v.h, v.sz, &prop, 0);
        if (r != CUDA_SUCCESS) {
            SAY("    %s: cuMemCreate %d\n", what, (int) r);
            return -1;
        }
        r = cuMemAddressReserve(&v.va, v.sz, 0, 0, 0);
        if (r != CUDA_SUCCESS) {
            cuMemRelease(v.h);
            SAY("    %s: cuMemAddressReserve %d\n", what, (int) r);
            return -1;
        }
        r = cuMemMap(v.va, v.sz, 0, v.h, 0);
        if (r != CUDA_SUCCESS) {
            cuMemAddressFree(v.va, v.sz);
            cuMemRelease(v.h);
            SAY("    %s: cuMemMap %d\n", what, (int) r);
            return -1;
        }
        as.maps.push_back(v);
        CUmemAccessDesc ad = {};
        ad.location = prop.location;
        ad.flags    = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        r = cuMemSetAccess(v.va, v.sz, &ad, 1);
        if (r != CUDA_SUCCESS) {
            SAY("    %s: cuMemSetAccess %d\n", what, (int) r);
            return -1;
        }
        r = cuMemsetD8(v.va, 0, v.sz);
        if (r != CUDA_SUCCESS || !cuda_ok(cudaDeviceSynchronize(), "synchronize")) {
            SAY("    %s: cuMemsetD8 %d\n", what, (int) r);
            if (r != CUDA_SUCCESS) {
                g_sticky = true;
            }
            return -1;
        }
    }
    const meas m2 = measure();
    SAY("    %s %zu MiB: ok in %.1f ms, LOCAL headroom %.2f -> %.2f GiB\n", what, mib, (now_s() - t0)*1e3, gib_s(m.l_head()), gib_s(m2.l_head()));
    return 1;
}

// a DMA round trip of [p, p + n): host (registered) -> device -> pinned host, compared with the source
bool dma_check(const uint8_t * p, size_t n, double * gbs) {
    const size_t chunk = std::min<size_t>(n, (size_t) 256 << 20);
    void * d = nullptr;
    void * h = nullptr;
    if (!cuda_ok(cudaMalloc(&d, chunk), "cudaMalloc (dma)")) {
        return false;
    }
    if (!cuda_ok(cudaMallocHost(&h, chunk), "cudaMallocHost (dma)")) {
        cudaFree(d);
        return false;
    }
    bool same = true;
    double t_h2d = 0.0;
    for (size_t o = 0; o < n && same && !g_sticky; o += chunk) {
        const size_t k = std::min(chunk, n - o);
        const double t0 = now_s();
        if (!cuda_ok(cudaMemcpy(d, p + o, k, cudaMemcpyHostToDevice), "H2D")) {
            same = false;
            break;
        }
        t_h2d += now_s() - t0;
        if (!cuda_ok(cudaMemcpy(h, d, k, cudaMemcpyDeviceToHost), "D2H")) {
            same = false;
            break;
        }
        same = memcmp(h, p + o, k) == 0;
    }
    // [TAG_FN_L23_PINFIX] the stream's way: asynchronous H2D straight from the registered pages, one copy stream, in
    // 32 MiB copies (LLAMA_PREFILL_STREAM_CHUNK_MIB) and in 256 MiB copies, 4 rounds over the range each; the same
    // bytes from the pinned bounce buffer (cudaMallocHost) for the PCIe ceiling
    if (same && !g_sticky) {
        cudaStream_t s = nullptr;
        if (cuda_ok(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream (dma async)")) {
            auto run = [&](const uint8_t * src, size_t len, size_t ck) {
                const double t0 = now_s();
                size_t moved = 0;
                for (int rep = 0; rep < 4 && !g_sticky; ++rep) {
                    for (size_t o = 0; o < len; o += ck) {
                        const size_t k = std::min(ck, len - o);
                        if (!cuda_ok(cudaMemcpyAsync(d, src + o, k, cudaMemcpyHostToDevice, s), "H2D async")) {
                            return 0.0;
                        }
                        moved += k;
                    }
                }
                if (!cuda_ok(cudaStreamSynchronize(s), "sync (dma async)")) {
                    return 0.0;
                }
                return (double) moved/(now_s() - t0)/1e9;
            };
            const double g32  = run(p, n, (size_t) 32 << 20);
            const double g256 = run(p, n, chunk);
            const double gb32 = run((const uint8_t *) h, chunk, (size_t) 32 << 20);
            SAY("  DMA async from the registered pages: %.1f GB/s in 32 MiB copies, %.1f GB/s in %zu MiB copies | pinned bounce buffer "
                "%.1f GB/s (32 MiB copies)\n", g32, g256, chunk >> 20, gb32);
            cudaStreamDestroy(s);
        }
    }
    cudaFreeHost(h);
    cudaFree(d);
    if (gbs) {
        *gbs = t_h2d > 0.0 ? (double) n/t_h2d/1e9 : 0.0;
    }
    return same;
}

// VirtualUnlock on pages that are not locked trims them from the working set (they go to standby)
void trim(const uint8_t * p, size_t n) {
    VirtualUnlock((LPVOID) p, n);
}

#endif // _WIN32

} // namespace

int moe_bridge_pin_test(const pin_probe_args & a, pin_say_fn say) {
    g_say = say;
    g_mapped = a.mapped;
    SAY("\n== 7. [TAG_FN_L22_PIN] pinned file pages: registration ladder, allocation, lock and touched / untouched tests\n");
#ifndef _WIN32
    (void) a;
    SAY("  (Windows only in this probe)\nverdict: FAIL (not Windows)\n");
    return 1;
#else
    cudaDeviceProp prop;
    if (!cuda_ok(cudaGetDeviceProperties(&prop, 0), "properties")) {
        return 1;
    }
    uint64_t luid = 0;
    memcpy(&luid, prop.luid, sizeof(luid));
    const bool dx = dxgi_open(luid);
    int ro = 0;
    cudaDeviceGetAttribute(&ro, cudaDevAttrHostRegisterReadOnlySupported, 0);
    SAY("  device %s, LUID %016llx (node mask %u), DXGI adapter %s, read-only registration %d, flags portable|readonly%s\n", prop.name,
            (unsigned long long) luid, prop.luidDeviceNodeMask, dx ? "found" : "NOT FOUND", ro, g_mapped ? "|mapped" : "");

    // 1. CPU only
    const meas m0 = measure();
    SAY("  1. NON_LOCAL budget %.2f GiB usage %.2f GiB | LOCAL budget %.2f GiB usage %.2f GiB | commit %.2f / %.2f GiB | RAM available %.2f GiB | standby %s%.2f GiB\n",
            gib(m0.nl_b), gib(m0.nl_u), gib(m0.l_b), gib(m0.l_u), gib(m0.commit), gib(m0.commit_limit), gib(m0.avail),
            m0.standby_ok ? "" : "(n/a) ", gib(m0.standby));
    if (!dx || !m0.dx) {
        SAY("verdict: FAIL (no DXGI budget for the CUDA device: the server would not pin either)\n");
        return 1;
    }

    // the shards and the unit
    std::vector<shard> sh;
    std::vector<fn_pin::layer_in> layers;
    for (const auto & f : a.files) {
        shard s;
        s.path = f;
        if (!shard_open(s)) {
            SAY("  cannot map %s\n", f.c_str());
            shard_close(s);
            continue;
        }
        sh.push_back(s);
    }
    for (const auto & s : sh) {
        shard_layers(s, a.hot_every, layers);
    }
    const fn_pin::plan plan = fn_pin::build_plan(layers, 4096, false, a.range_mib*MIB);
    uint64_t plan_bytes = 0;
    for (const auto & r : plan.ranges) {
        plan_bytes += r.n;
    }
    SAY("  unit: %zu shard(s), %zu expert layer parts, %zu ranges in %zu groups, %.2f GiB (hot every %d, cut %zu MiB)\n", sh.size(),
            layers.size(), plan.ranges.size(), plan.groups.size(), gib(plan_bytes), a.hot_every, a.range_mib);
    if (plan.ranges.empty()) {
        SAY("verdict: FAIL (no expert tensors found)\n");
        for (auto & s : sh) {
            shard_close(s);
        }
        return 1;
    }

    // the test regions for steps 3-4 come from the end of the last shard: N (untouched), T (touched), R, U
    const size_t lock_sz = std::max<size_t>(256*MIB, a.lock_mib*MIB) & ~(size_t) 0xFFFF;
    const shard & ls = sh.back();
    const uint8_t * reg_end = ls.v + (ls.size & ~(size_t) 0xFFFF) - 2*MIB;
    const uint8_t * N = reg_end - 1*lock_sz;
    const uint8_t * T = reg_end - 2*lock_sz;
    const uint8_t * R = reg_end - 3*lock_sz;
    const uint8_t * U = reg_end - 4*lock_sz;
    const bool have_regions = ls.size > 4*lock_sz + 64*MIB;
    auto overlaps_regions = [&](const uint8_t * p, size_t n) {
        return have_regions && p < reg_end && p + n > U;
    };

    bool fail = false;
    std::string fail_why;
    std::vector<reg_entry> regd;
    uint64_t pinned = 0;
    uint64_t pinned_max = 0;
    double charge_per_gib = 0.0;
    size_t next = 0;
    std::string ladder_stop = "end of the ladder";

    auto ladder = [&](int pass, uint64_t limit) {
        const meas mb = measure();
        alloc_set as;
        for (double step : a.ladder_gib) {
            const uint64_t target = std::min<uint64_t>((uint64_t) (step*(double) GIB), limit);
            const double t0 = now_s();
            const size_t n0 = regd.size();
            double ms_sum = 0.0;
            uint64_t added = 0;
            bool stop = false;
            while (pinned < target && next < plan.ranges.size()) {
                const fn_pin::range & r = plan.ranges[next];
                if (overlaps_regions(r.p, r.n)) {
                    next++;
                    continue;
                }
                const meas m = measure();
                if (m.nl_head() < (int64_t) (r.n + a.margin_mib*MIB)) {
                    ladder_stop = "NON_LOCAL budget - usage - margin";
                    stop = true;
                    break;
                }
                if (m.avail < r.n + a.floor_mib*MIB) {
                    ladder_stop = "RAM floor";
                    stop = true;
                    break;
                }
                if (m.l_head() < (int64_t) GIB) {
                    ladder_stop = "LOCAL headroom < 1 GiB";
                    stop = true;
                    break;
                }
                double ms = 0.0;
                if (!reg(r.p, r.n, &ms)) {
                    ladder_stop = "refused by the driver";
                    if (regd.empty()) {
                        fail = true;
                        fail_why = "the first registration was refused";
                    }
                    stop = true;
                    break;
                }
                ms_sum += ms;
                regd.push_back({ r.p, r.n });
                pinned += r.n;
                added += r.n;
                next++;
            }
            const double secs = now_s() - t0;
            const meas m = measure();
            std::vector<std::pair<const uint8_t *, size_t>> rs;
            for (const auto & e : regd) {
                rs.push_back({ e.p, e.n });
            }
            const size_t n_new = regd.size() - n0;
            const double dl = (double) ((int64_t) m.l_u - (int64_t) mb.l_u)/(double) MIB;
            charge_per_gib = pinned > 0 ? dl/gib(pinned) : 0.0;
            SAY("  2.%d ladder %.0f GiB: pinned %.2f GiB in %zu ranges (+%zu in %.2f s, %.3f ms/range, %.2f s/GiB) | commit %.2f GiB | "
                "NON_LOCAL usage %.2f (+%.2f) / %.2f GiB | LOCAL usage %.2f (%+.0f MiB = %.1f MiB per pinned GiB) / %.2f GiB | Valid %.3f | avail %.2f GiB\n",
                    pass, step, gib(pinned), regd.size(), n_new, secs, n_new ? ms_sum/n_new : 0.0,
                    added > 0 ? secs/gib(added) : 0.0, gib(m.commit), gib(m.nl_u),
                    gib_s((int64_t) m.nl_u - (int64_t) mb.nl_u), gib(m.nl_b), gib(m.l_u), dl, charge_per_gib, gib(m.l_b), valid_share(rs),
                    gib(m.avail));
            pinned_max = std::max(pinned_max, pinned);
            // the real allocations, only within the LOCAL headroom
            if (!g_sticky) {
                const int r1 = alloc_test(as, "FULL reserve (cudaMalloc)", a.full_mib, 0);
                const int r2 = r1 >= 0 && !g_sticky ? alloc_test(as, "borrow (cuMemCreate + cuMemMap)", a.borrow_mib, 1) : 0;
                const int r3 = r2 >= 0 && r1 >= 0 && !g_sticky ? alloc_test(as, "vision lend (cudaMalloc)", a.vision_mib, 0) : 0;
                as.free_all();
                if (r1 < 0 || r2 < 0 || r3 < 0) {
                    fail = true;
                    fail_why = "a device allocation failed inside the LOCAL headroom with pins";
                    stop = true;
                }
            }
            if (g_sticky) {
                fail = true;
                fail_why = "CUDA error";
                stop = true;
            }
            if (stop || next >= plan.ranges.size()) {
                break;
            }
        }
        SAY("  2.%d ladder ends: %s\n", pass, ladder_stop.c_str());
    };
    ladder(1, UINT64_MAX);

    // 3. lock test without memory pressure
    bool locked_R = false;
    bool unclear_R = true;
    bool dma_R = false;
    if (!fail && have_regions && !g_sticky) {
        const meas mq = measure();
        if (mq.avail < 4*lock_sz + a.floor_mib*MIB) {
            SAY("  3. skipped: %.2f GiB available, the lock test needs %.2f GiB above the floor\n", gib(mq.avail), gib(4*lock_sz));
        } else {
            touch(R, lock_sz);
            touch(U, lock_sz);
            double ms = 0.0;
            if (!reg(R, lock_sz, &ms)) {
                fail = true;
                fail_why = "registering the lock test range failed";
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                const meas a0 = measure();
                const double vr0 = valid_share1(R, lock_sz);
                const double vu0 = valid_share1(U, lock_sz);
                trim(U, lock_sz);
                std::this_thread::sleep_for(std::chrono::milliseconds(300));
                const meas a1 = measure();
                const double vu1 = valid_share1(U, lock_sz);
                trim(R, lock_sz);
                std::this_thread::sleep_for(std::chrono::milliseconds(300));
                const meas a2 = measure();
                const double vr2 = valid_share1(R, lock_sz);
                const double gain_u = (double) ((int64_t) a1.avail - (int64_t) a0.avail);
                const double gain_r = (double) ((int64_t) a2.avail - (int64_t) a1.avail);
                const bool u_moved = gain_u > 0.5*(double) lock_sz;
                locked_R  = u_moved && gain_r < 0.1*(double) lock_sz;
                unclear_R = !u_moved;
                double gbs = 0.0;
                dma_R = dma_check(R, lock_sz, &gbs);
                SAY("  3. lock test (%.2f GiB each, registered in %.1f ms): Valid R %.2f U %.2f | trim U: available %+.2f GiB (Valid %.2f), standby %+.2f GiB | "
                    "trim R: available %+.2f GiB (Valid %.2f), standby %+.2f GiB | -> R %s | DMA round trip %s (%.1f GB/s H2D)\n",
                        gib(lock_sz), ms, vr0, vu0, gain_u/(double) GIB, vu1,
                        a1.standby_ok ? gib_s((int64_t) a1.standby - (int64_t) a0.standby) : 0.0, gain_r/(double) GIB, vr2,
                        a2.standby_ok ? gib_s((int64_t) a2.standby - (int64_t) a1.standby) : 0.0,
                        unclear_R ? "UNCLEAR (U did not move: counts as unlocked)" : locked_R ? "LOCKED" : "NOT LOCKED",
                        dma_R ? "equal" : "DIFFERENT", gbs);
                if (!dma_R) {
                    fail = true;
                    fail_why = "the DMA round trip of a registered file range differs";
                }
                unreg(R);
            }
        }
    } else if (!have_regions) {
        SAY("  3./4. skipped: the last shard is too small for the test regions\n");
    }

    // 4. touched vs untouched
    bool locked_T = false;
    bool locked_N = false;
    double spg_T = 0.0;
    double spg_N = 0.0;
    bool dma_N = false;
    if (!fail && have_regions && !g_sticky) {
        const double vn = valid_share1(N, lock_sz);
        touch(T, lock_sz);
        const meas b0 = measure();
        double ms_t = 0.0;
        const bool okt = reg(T, lock_sz, &ms_t);
        const meas b1 = measure();
        trim(T, lock_sz);
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        const meas b2 = measure();
        const double gain_t = (double) ((int64_t) b2.avail - (int64_t) b1.avail);
        locked_T = okt && gain_t < 0.1*(double) lock_sz;
        spg_T = ms_t/1e3/gib(lock_sz);
        const meas c0 = measure();
        double ms_n = 0.0;
        const bool okn = reg(N, lock_sz, &ms_n);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        const meas c1 = measure();
        const double vn1 = valid_share1(N, lock_sz);
        trim(N, lock_sz);
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        const meas c2 = measure();
        const double drop_n = (double) ((int64_t) c0.avail - (int64_t) c1.avail);
        const double gain_n = (double) ((int64_t) c2.avail - (int64_t) c1.avail);
        locked_N = okn && gain_n < 0.1*(double) lock_sz && (vn1 > 0.5 || drop_n > 0.5*(double) lock_sz);
        spg_N = ms_n/1e3/gib(lock_sz);
        double gbs = 0.0;
        dma_N = okn && dma_check(N, lock_sz, &gbs);
        SAY("  4. touched T: %s in %.1f ms (%.2f s/GiB), LOCAL %+.0f MiB, trim gain %+.2f GiB -> %s | untouched N (Valid before %.2f): %s in %.1f ms "
            "(%.2f s/GiB), LOCAL %+.0f MiB, available %+.2f GiB at registration, Valid after %.2f, trim gain %+.2f GiB -> %s, DMA %s (%.1f GB/s)\n",
                okt ? "registered" : "REFUSED", ms_t, spg_T, (double) ((int64_t) b1.l_u - (int64_t) b0.l_u)/(double) MIB, gain_t/(double) GIB,
                locked_T ? "LOCKED" : "NOT LOCKED", vn, okn ? "registered" : "REFUSED", ms_n, spg_N,
                (double) ((int64_t) c1.l_u - (int64_t) c0.l_u)/(double) MIB, -drop_n/(double) GIB, vn1, gain_n/(double) GIB,
                locked_N ? "LOCKED" : "NOT LOCKED / UNCLEAR", dma_N ? "equal" : (okn ? "DIFFERENT" : "-"), gbs);
        if (okn && !dma_N) {
            fail = true;
            fail_why = "the DMA round trip of an untouched registered range differs";
        }
        if (okt) {
            unreg(T);
        }
        if (okn) {
            unreg(N);
        }
    }

    // 5. unregister all; the usage must come back; the ladder once more
    const meas mpre = measure();
    for (const auto & e : regd) {
        unreg(e.p);
    }
    regd.clear();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    const meas m5 = measure();
    const bool back = (int64_t) m5.nl_u - (int64_t) m0.nl_u < (int64_t) (256*MIB) && (int64_t) m5.l_u - (int64_t) m0.l_u < (int64_t) (256*MIB);
    SAY("  5. unregistered %.2f GiB: NON_LOCAL usage %.2f -> %.2f GiB (start %.2f), LOCAL usage %.2f -> %.2f GiB (start %.2f): %s\n", gib(pinned),
            gib(mpre.nl_u), gib(m5.nl_u), gib(m0.nl_u), gib(mpre.l_u), gib(m5.l_u), gib(m0.l_u), back ? "back" : "NOT BACK");
    if (!back && !fail) {
        fail = true;
        fail_why = "the driver did not return the budget after the unregister";
    }
    if (a.repeat && !fail && !g_sticky && pinned_max > 0) {
        pinned = 0;
        next = 0;
        ladder_stop = "end of the ladder";
        ladder(2, pinned_max);
        for (const auto & e : regd) {
            unreg(e.p);
        }
        regd.clear();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        const meas m6 = measure();
        SAY("  5. second ladder unregistered: NON_LOCAL usage %.2f GiB, LOCAL usage %.2f GiB (start %.2f / %.2f)\n", gib(m6.nl_u), gib(m6.l_u),
                gib(m0.nl_u), gib(m0.l_u));
    }
    for (auto & s : sh) {
        shard_close(s);
    }

    // the verdict
    const char * verdict = fail || g_sticky ? "FAIL" : (locked_R && !unclear_R) ? "FILE_OK" : "FILE_UNLOCKED";
    const bool untouched_ok = locked_N && dma_N && spg_N <= std::max(1.0, 4.0*spg_T);
    const char * mode = strcmp(verdict, "FILE_OK") != 0 ? "-" : untouched_ok ? "untouched" : locked_T ? "touched" : "-";
    SAY("\nverdict: %s mode=%s mapped=%d pinned_max_gib=%.2f nonlocal_budget_gib=%.2f nonlocal_usage_gib=%.2f local_budget_gib=%.2f local_usage_gib=%.2f "
        "local_charge_mib_per_gib=%.1f untouched_s_per_gib=%.2f touched_s_per_gib=%.2f%s%s\n", verdict, mode, g_mapped ? 1 : 0, gib(pinned_max), gib(m0.nl_b),
            gib(m0.nl_u), gib(m0.l_b), gib(m0.l_u), charge_per_gib, spg_N, spg_T, fail ? " why=" : "", fail ? fail_why.c_str() : "");
    if (strcmp(verdict, "FILE_OK") == 0) {
        SAY("  server: LLAMA_FN_L22_PIN=%d (LLAMA_FN_L22_PIN_UNIT=ranges)\n", strcmp(mode, "untouched") == 0 ? 1 : 2);
    } else if (strcmp(verdict, "FILE_UNLOCKED") == 0) {
        SAY("  server: LLAMA_FN_L22_PIN=1 LLAMA_FN_L22_PIN_UNIT=arena (file pages cannot be trusted to stay)\n");
    } else {
        SAY("  server: keep LLAMA_FN_L22_PIN=0\n");
    }
    return fail ? 1 : 0;
#endif
}

// [TAG_FN_L23_PINFIX] --only bounce (FAULT 2026-10-21 10:10): a host -> device copy that starts inside a registered host
// range and runs past its end. (1) the raw cudaMemcpyAsync: CUDA's answer (expected: invalid argument); (2) the same copy
// through ggml_backend_tensor_set (ggml-cuda sends it through its pinned bounce buffer): it must land, bytes equal;
// (3) a copy that starts before the range and runs into it (pageable start), raw. No model; sanitizer-friendly (16 MiB).
int moe_bridge_bounce_test(pin_say_fn say) {
    const size_t MB = (size_t) 1 << 20;
    const size_t N  = 16*MB;
#if defined(_WIN32)
    uint8_t * host = (uint8_t *) VirtualAlloc(nullptr, N, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
    uint8_t * host = (uint8_t *) aligned_alloc(4096, N);
#endif
    if (host == nullptr) {
        say("bounce: no host memory\n");
        return 1;
    }
    for (size_t i = 0; i < N; ++i) {
        host[i] = (uint8_t) (i*2654435761u >> 13);
    }
    int rc = 0;
    if (!cuda_ok(cudaHostRegister(host + 4*MB, 8*MB, cudaHostRegisterPortable), "cudaHostRegister (bounce)")) {
        return 1;
    }
    void * d = nullptr;
    cuda_ok(cudaMalloc(&d, 8*MB), "cudaMalloc (bounce)");
    const cudaError_t e1 = cudaMemcpy(d, host + 8*MB, 8*MB, cudaMemcpyHostToDevice);   // starts inside, runs past
    (void) cudaGetLastError();
    const cudaError_t e3 = cudaMemcpy(d, host + 2*MB, 4*MB, cudaMemcpyHostToDevice);   // starts before, runs into
    (void) cudaGetLastError();
    say("bounce: raw copy inside -> past the registered end: %s | outside -> into the range: %s\n", cudaGetErrorName(e1), cudaGetErrorName(e3));
    cudaFree(d);
    // through ggml-cuda
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (dev == nullptr) {
        say("bounce: no GPU device in ggml\n");
        rc = 1;
    } else {
        ggml_init_params ip = { 4*ggml_tensor_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, (int64_t) (8*MB));
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_dev_buffer_type(dev));
        ggml_backend_tensor_set(t, host + 8*MB, 0, 8*MB);
        std::vector<uint8_t> back(8*MB);
        ggml_backend_tensor_get(t, back.data(), 0, 8*MB);
        const bool same = memcmp(back.data(), host + 8*MB, 8*MB) == 0;
        say("bounce: ggml_backend_tensor_set of the same copy: %s\n", same ? "landed, bytes equal" : "BYTES DIFFER");
        rc = same ? 0 : 1;
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    cuda_ok(cudaHostUnregister(host + 4*MB), "cudaHostUnregister (bounce)");
#if defined(_WIN32)
    VirtualFree(host, 0, MEM_RELEASE);
#else
    free(host);
#endif
    say("bounce verdict: %s (raw inside->past %s)\n", rc == 0 ? "PASS" : "FAIL", cudaGetErrorName(e1));
    return rc;
}
