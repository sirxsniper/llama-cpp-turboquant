// [TAG_FN_BRIDGE_PROBE] Standalone CUDA probe for the MoE doorbell design (E:/turbot-gates/flashnext/PLAN.md SP-2b, D4).
// No model, no ggml. Measures on this machine (WDDM):
//   1. GPU -> CPU -> GPU round trip through mapped pinned flags (post kernel + CPU responder + bounded spin kernel),
//      outside and inside a captured CUDA graph, with 0 and 10 KiB payloads, 1 and 48 hops per graph
//   2. the same hop with cudaLaunchHostFunc (D2H copy node + host node + H2D copy node) inside a graph
//   3. stream memory operations: the device attribute and a cuStreamWaitValue32 round trip (outside a graph)
//   4. pinned H2D / D2H bandwidth from one cudaHostAlloc buffer (<= 4 GiB)
//   5. item 4 while 16 CPU threads (mask 0x55555555) stream a pageable region (<= 8 GiB): combined DRAM ceiling
// Limits: <= 4096 MiB pinned, <= 8192 MiB streamed, about 5 s per test, the spin kernel gives up after 50 ms and
// reports a timeout instead of hanging (far below the 2 s TDR). Stops at the first CUDA error.
//
//   moe-bridge-probe [--log E:/turbot-gates/flashnext/probe.log] [--pinned-mib 4096] [--stream-mib 8192]
//                    [--threads 16] [--mask 0x55555555] [--quick] [--only rt|hostfn|memops|bw|bwcpu]
//   moe-bridge-probe --selftest [--threads 8] [--experts 64] [--reps 4] [--quick]
//                    [TAG_MOE_BRIDGE] the ggml bridge ops against the CPU MUL_MAT_ID chain (moe-bridge-selftest.cpp)

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
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
#define NOMINMAX
#include <windows.h>
#include <immintrin.h>
#else
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#endif

static FILE * g_log = nullptr;

static void say(const char * fmt, ...) {
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    fputs(buf, stdout);
    fflush(stdout);
    if (g_log) {
        fputs(buf, g_log);
        fflush(g_log);
    }
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    say("CUDA error %s at %s:%d: %s\n", cudaGetErrorName(e_), __FILE__, __LINE__, cudaGetErrorString(e_)); \
    exit(3); } } while (0)

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void pin_this_thread(int cpu) {
#ifdef _WIN32
    if (cpu >= 0 && cpu < 64) {
        SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) 1ull << cpu);
    }
#else
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(cpu, &s);
    pthread_setaffinity_np(pthread_self(), sizeof(s), &s);
#endif
}

// ---------------------------------------------------------------------------------------------------------------------
// 1. mapped-flag doorbell

struct bridge_host {             // cudaHostAlloc(Mapped | Portable)
    volatile uint32_t post;      // GPU writes the sequence number after the payload
    volatile uint32_t done;      // CPU writes the sequence number after the result
    uint32_t pad[14];
};

__device__ __forceinline__ uint64_t gtimer_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// one block: bump the device sequence, write the payload to host memory, fence, then publish the sequence
__global__ void k_post(bridge_host * h, uint32_t * d_seq, const float * x, float * h_x, int n) {
    __shared__ uint32_t seq;
    if (threadIdx.x == 0) {
        seq = *d_seq + 1;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        h_x[i] = x[i] + (float) (seq & 7);
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        *d_seq = seq;
        h->post = seq;
    }
}

// one block: spin until the CPU has answered this sequence (bounded), then read the result back
__global__ void k_wait(bridge_host * h, const uint32_t * d_seq, const float * h_y, float * y, int n, uint64_t limit_ns,
        uint32_t * d_err) {
    __shared__ int ok;
    if (threadIdx.x == 0) {
        const uint32_t want = *d_seq;
        const uint64_t t0   = gtimer_ns();
        ok = 1;
        while (h->done != want) {
            __nanosleep(256);
            if (gtimer_ns() - t0 > limit_ns) {
                ok = 0;
                atomicAdd(d_err, 1u);
                break;
            }
        }
    }
    __syncthreads();
    __threadfence_system();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        y[i] = ok ? h_y[i] : 0.0f;
    }
}

__global__ void k_empty(float * y) {
    if (threadIdx.x == 0 && y) {
        y[0] += 1.0f;
    }
}

struct responder {
    bridge_host * h;
    const float * h_x;
    float *       h_y;
    int           n;
    int           cpu;
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> served{0};
    std::thread th;

    void start() {
        th = std::thread([this]() {
            pin_this_thread(cpu);
            uint32_t last = h->post;
            while (!stop.load(std::memory_order_relaxed)) {
                const uint32_t p = h->post;
                if (p == last) {
                    _mm_pause();
                    continue;
                }
                last = p;
                std::atomic_thread_fence(std::memory_order_acquire);
                for (int i = 0; i < n; ++i) {
                    h_y[i] = h_x[i] * 0.5f;
                }
                std::atomic_thread_fence(std::memory_order_release);
                _mm_sfence();
                h->done = p;
                served.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    void join() {
        stop = true;
        if (th.joinable()) {
            th.join();
        }
    }
};

static void test_roundtrip(bool quick, int cpu) {
    say("\n== 1. mapped-flag doorbell round trip (post kernel -> CPU responder -> bounded spin kernel)\n");
    bridge_host * h = nullptr;
    CK(cudaHostAlloc((void **) &h, sizeof(bridge_host), cudaHostAllocMapped | cudaHostAllocPortable));
    memset((void *) h, 0, sizeof(bridge_host));
    bridge_host * dh = nullptr;
    CK(cudaHostGetDevicePointer((void **) &dh, h, 0));

    const int n_max = 2560 * 4; // 10 KiB of f32, one token's hidden state x 4
    float * h_x = nullptr;
    float * h_y = nullptr;
    CK(cudaHostAlloc((void **) &h_x, n_max * sizeof(float), cudaHostAllocMapped | cudaHostAllocPortable));
    CK(cudaHostAlloc((void **) &h_y, n_max * sizeof(float), cudaHostAllocMapped | cudaHostAllocPortable));
    float * dh_x = nullptr;
    float * dh_y = nullptr;
    CK(cudaHostGetDevicePointer((void **) &dh_x, h_x, 0));
    CK(cudaHostGetDevicePointer((void **) &dh_y, h_y, 0));

    float * d_x = nullptr;
    float * d_y = nullptr;
    uint32_t * d_seq = nullptr;
    uint32_t * d_err = nullptr;
    CK(cudaMalloc(&d_x, n_max * sizeof(float)));
    CK(cudaMalloc(&d_y, n_max * sizeof(float)));
    CK(cudaMalloc(&d_seq, sizeof(uint32_t)));
    CK(cudaMalloc(&d_err, sizeof(uint32_t)));
    CK(cudaMemset(d_x, 0, n_max * sizeof(float)));
    CK(cudaMemset(d_seq, 0, sizeof(uint32_t)));
    CK(cudaMemset(d_err, 0, sizeof(uint32_t)));

    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

    const uint64_t limit_ns = 50ull * 1000 * 1000;

    for (int payload : {0, 1}) {
        const int n = payload ? n_max : 0;
        responder r;
        r.h = h; r.h_x = h_x; r.h_y = h_y; r.n = n; r.cpu = cpu;
        r.start();

        // baseline: the same two launches with no spin, to separate launch cost from the hop
        {
            const int iters = quick ? 2000 : 10000;
            const double t0 = now_s();
            for (int i = 0; i < iters; ++i) {
                k_empty<<<1, 32, 0, s>>>(d_y);
                k_empty<<<1, 32, 0, s>>>(d_y);
            }
            CK(cudaStreamSynchronize(s));
            say("  payload %5d B  baseline 2 empty launches (stream):   %8.2f us/iter\n", n * 4, (now_s() - t0) / iters * 1e6);
        }

        // outside a graph
        {
            int iters = quick ? 2000 : 10000;
            const double t0 = now_s();
            int done_iters = 0;
            for (int i = 0; i < iters; ++i) {
                k_post<<<1, 256, 0, s>>>(dh, d_seq, d_x, dh_x, n);
                k_wait<<<1, 256, 0, s>>>(dh, d_seq, dh_y, d_y, n, limit_ns, d_err);
                ++done_iters;
                if ((i & 255) == 255) {
                    CK(cudaStreamSynchronize(s));
                    if (now_s() - t0 > 5.0) {
                        break;
                    }
                }
            }
            CK(cudaStreamSynchronize(s));
            const double dt = now_s() - t0;
            uint32_t err = 0;
            CK(cudaMemcpy(&err, d_err, sizeof(err), cudaMemcpyDeviceToHost));
            say("  payload %5d B  hop outside a graph (stream):          %8.2f us/hop   timeouts %u\n", n * 4, dt / done_iters * 1e6, err);
        }

        // inside captured graphs: 1 hop and 48 hops (one per MoE layer) per graph
        for (int hops : {1, 48}) {
            cudaGraph_t g;
            cudaGraphExec_t ge;
            CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed));
            for (int k = 0; k < hops; ++k) {
                k_post<<<1, 256, 0, s>>>(dh, d_seq, d_x, dh_x, n);
                k_wait<<<1, 256, 0, s>>>(dh, d_seq, dh_y, d_y, n, limit_ns, d_err);
            }
            CK(cudaStreamEndCapture(s, &g));
            CK(cudaGraphInstantiate(&ge, g, 0));
            const int launches = (quick ? 2000 : 10000) / hops + 1;
            const double t0 = now_s();
            int done_l = 0;
            for (int i = 0; i < launches; ++i) {
                CK(cudaGraphLaunch(ge, s));
                ++done_l;
                if ((i & 63) == 63) {
                    CK(cudaStreamSynchronize(s));
                    if (now_s() - t0 > 5.0) {
                        break;
                    }
                }
            }
            CK(cudaStreamSynchronize(s));
            const double dt = now_s() - t0;
            uint32_t err = 0;
            CK(cudaMemcpy(&err, d_err, sizeof(err), cudaMemcpyDeviceToHost));
            say("  payload %5d B  hop inside a graph (%2d hops/graph):     %8.2f us/hop   timeouts %u\n", n * 4, hops,
                    dt / (done_l * (double) hops) * 1e6, err);
            CK(cudaGraphExecDestroy(ge));
            CK(cudaGraphDestroy(g));
        }
        r.join();
        say("  payload %5d B  responder served %llu hops\n", n * 4, (unsigned long long) r.served.load());
    }

    // stall check: no responder, one hop must time out after ~50 ms and not hang
    {
        const double t0 = now_s();
        k_post<<<1, 256, 0, s>>>(dh, d_seq, d_x, dh_x, 0);
        k_wait<<<1, 256, 0, s>>>(dh, d_seq, dh_y, d_y, 0, limit_ns, d_err);
        CK(cudaStreamSynchronize(s));
        uint32_t err = 0;
        CK(cudaMemcpy(&err, d_err, sizeof(err), cudaMemcpyDeviceToHost));
        say("  stall test (no responder): returned after %.1f ms, timeouts so far %u (expect +1)\n", (now_s() - t0) * 1e3, err);
    }

    CK(cudaStreamDestroy(s));
    CK(cudaFree(d_x)); CK(cudaFree(d_y)); CK(cudaFree(d_seq)); CK(cudaFree(d_err));
    CK(cudaFreeHost(h_x)); CK(cudaFreeHost(h_y)); CK(cudaFreeHost(h));
}

// ---------------------------------------------------------------------------------------------------------------------
// 2. host-function hop inside a graph

struct hostfn_ctx {
    float * h_in;
    float * h_out;
    int     n;
    uint64_t calls = 0;
};

static void CUDART_CB hostfn(void * p) {
    hostfn_ctx * c = (hostfn_ctx *) p;
    for (int i = 0; i < c->n; ++i) {
        c->h_out[i] = c->h_in[i] * 0.5f;
    }
    c->calls++;
}

static void test_hostfn(bool quick) {
    say("\n== 2. cudaLaunchHostFunc hop inside a graph (D2H copy node -> host node -> H2D copy node)\n");
    const int n = 2560 * 4;
    hostfn_ctx c;
    c.n = n;
    CK(cudaMallocHost((void **) &c.h_in, n * sizeof(float)));
    CK(cudaMallocHost((void **) &c.h_out, n * sizeof(float)));
    float * d_a = nullptr;
    float * d_b = nullptr;
    CK(cudaMalloc(&d_a, n * sizeof(float)));
    CK(cudaMalloc(&d_b, n * sizeof(float)));
    CK(cudaMemset(d_a, 0, n * sizeof(float)));
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    for (int hops : {1, 48}) {
        cudaGraph_t g;
        cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed));
        for (int k = 0; k < hops; ++k) {
            k_empty<<<1, 32, 0, s>>>(d_a);
            CK(cudaMemcpyAsync(c.h_in, d_a, n * sizeof(float), cudaMemcpyDeviceToHost, s));
            CK(cudaLaunchHostFunc(s, hostfn, &c));
            CK(cudaMemcpyAsync(d_b, c.h_out, n * sizeof(float), cudaMemcpyHostToDevice, s));
        }
        CK(cudaStreamEndCapture(s, &g));
        CK(cudaGraphInstantiate(&ge, g, 0));
        const int launches = (quick ? 1000 : 5000) / hops + 1;
        const double t0 = now_s();
        int done_l = 0;
        for (int i = 0; i < launches; ++i) {
            CK(cudaGraphLaunch(ge, s));
            ++done_l;
            if ((i & 31) == 31) {
                CK(cudaStreamSynchronize(s));
                if (now_s() - t0 > 5.0) {
                    break;
                }
            }
        }
        CK(cudaStreamSynchronize(s));
        say("  10 KiB each way, %2d hops/graph: %8.2f us/hop (host calls %llu)\n", hops,
                (now_s() - t0) / (done_l * (double) hops) * 1e6, (unsigned long long) c.calls);
        CK(cudaGraphExecDestroy(ge));
        CK(cudaGraphDestroy(g));
    }
    CK(cudaStreamDestroy(s));
    CK(cudaFree(d_a)); CK(cudaFree(d_b));
    CK(cudaFreeHost(c.h_in)); CK(cudaFreeHost(c.h_out));
}

// ---------------------------------------------------------------------------------------------------------------------
// 3. stream memory operations

static void test_memops(int cpu) {
    say("\n== 3. stream memory operations (cuStreamWaitValue32)\n");
    CUdevice dev;
    if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) {
        say("  cuDeviceGet failed\n");
        return;
    }
    int v = -1;
    // 92 = CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS(_V1); by number, the enum name changed across CUDA versions
    const CUresult rq = cuDeviceGetAttribute(&v, (CUdevice_attribute) 92, dev);
    say("  attribute 92 (can use stream mem ops): %d (query rc %d)\n", v, (int) rq);

    volatile uint32_t * flag = nullptr;
    CK(cudaHostAlloc((void **) &flag, 64, cudaHostAllocMapped | cudaHostAllocPortable));
    *flag = 0;
    CUdeviceptr dflag = 0;
    CK(cudaHostGetDevicePointer((void **) &dflag, (void *) flag, 0));
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    float * d_y = nullptr;
    CK(cudaMalloc(&d_y, 4));

    const int iters = 2000;
    std::atomic<int> go{0};
    std::thread t([&]() {
        pin_this_thread(cpu);
        for (int i = 1; i <= iters; ++i) {
            while (go.load() < i) {
                _mm_pause();
            }
            *flag = (uint32_t) i;
        }
    });
    int ok = 0;
    double t_total = 0.0;
    for (int i = 1; i <= iters; ++i) {
        const CUresult rc = cuStreamWaitValue32((CUstream) s, dflag, (cuuint32_t) i, CU_STREAM_WAIT_VALUE_GEQ);
        if (rc != CUDA_SUCCESS) {
            const char * name = nullptr;
            cuGetErrorName(rc, &name);
            say("  cuStreamWaitValue32 failed: %s\n", name ? name : "?");
            go = iters + 1;
            break;
        }
        k_empty<<<1, 32, 0, s>>>(d_y);
        const double t0 = now_s();
        go = i;
        CK(cudaStreamSynchronize(s));
        t_total += now_s() - t0;
        ++ok;
    }
    t.join();
    if (ok > 0) {
        say("  host flag -> stream wait -> kernel -> synchronize: %.2f us per round (%d rounds)\n", t_total / ok * 1e6, ok);
    }
    CK(cudaStreamDestroy(s));
    CK(cudaFree(d_y));
    CK(cudaFreeHost((void *) flag));
}

// ---------------------------------------------------------------------------------------------------------------------
// 4/5. bandwidth

struct streamer {
    const uint8_t * base = nullptr;
    size_t          bytes = 0;
    int             n_threads = 16;
    uint64_t        mask = 0x55555555ull;
    std::atomic<bool> stop{false};
    std::vector<std::thread> th;
    std::vector<double> gbytes;

    void start() {
        gbytes.assign(n_threads, 0.0);
        std::vector<int> cpus;
        for (int b = 0; b < 64 && (int) cpus.size() < n_threads; ++b) {
            if (mask & (1ull << b)) {
                cpus.push_back(b);
            }
        }
        for (int t = 0; t < n_threads; ++t) {
            th.emplace_back([this, t, cpus]() {
                pin_this_thread(t < (int) cpus.size() ? cpus[t] : -1);
                const size_t per = bytes / n_threads / 64 * 64;
                const uint8_t * p0 = base + per * t;
                __m256i acc = _mm256_setzero_si256();
                while (!stop.load(std::memory_order_relaxed)) {
                    for (size_t off = 0; off + 256 <= per; off += 256) {
                        const __m256i * q = (const __m256i *) (p0 + off);
                        acc = _mm256_xor_si256(acc, _mm256_load_si256(q + 0));
                        acc = _mm256_xor_si256(acc, _mm256_load_si256(q + 1));
                        acc = _mm256_xor_si256(acc, _mm256_load_si256(q + 2));
                        acc = _mm256_xor_si256(acc, _mm256_load_si256(q + 3));
                        acc = _mm256_xor_si256(acc, _mm256_load_si256(q + 4));
                        acc = _mm256_xor_si256(acc, _mm256_load_si256(q + 5));
                        acc = _mm256_xor_si256(acc, _mm256_load_si256(q + 6));
                        acc = _mm256_xor_si256(acc, _mm256_load_si256(q + 7));
                        if ((off & ((64u << 20) - 1)) == 0 && stop.load(std::memory_order_relaxed)) {
                            break;
                        }
                    }
                    gbytes[t] += per / 1e9;
                }
                volatile int sink = _mm256_extract_epi32(acc, 0);
                (void) sink;
            });
        }
    }
    double join() {
        stop = true;
        for (auto & t : th) {
            t.join();
        }
        double s = 0.0;
        for (double g : gbytes) {
            s += g;
        }
        return s;
    }
};

static void test_bw(size_t pinned_mib, size_t stream_mib, int n_threads, uint64_t mask, bool with_cpu, bool quick) {
    say(with_cpu ? "\n== 5. pinned H2D while %d CPU threads stream a %zu MiB region\n" : "\n== 4. pinned copy bandwidth\n",
            n_threads, stream_mib);
    const size_t bytes = pinned_mib << 20;
    const size_t chunk = std::min<size_t>(bytes, (size_t) 1024 << 20);
    uint8_t * h = nullptr;
    CK(cudaHostAlloc((void **) &h, bytes, cudaHostAllocPortable));
    memset(h, 1, bytes);
    uint8_t * d = nullptr;
    CK(cudaMalloc(&d, chunk));
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

    uint8_t * region = nullptr;
    streamer st;
    if (with_cpu) {
        const size_t rb = stream_mib << 20;
#ifdef _WIN32
        region = (uint8_t *) VirtualAlloc(nullptr, rb, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
        region = (uint8_t *) aligned_alloc(4096, rb);
#endif
        if (!region) {
            say("  cannot allocate the streamed region\n");
            CK(cudaFreeHost(h)); CK(cudaFree(d)); CK(cudaStreamDestroy(s));
            return;
        }
        memset(region, 2, rb);
        st.base = region; st.bytes = rb; st.n_threads = n_threads; st.mask = mask;
        // CPU alone first
        const double t0 = now_s();
        st.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(quick ? 800 : 2000));
        const double g = st.join();
        say("  CPU alone: %.1f GB/s\n", g / (now_s() - t0));
        st.stop = false;
        st.th.clear();
        st.start();
    }
    const double t_cpu0 = now_s();

    for (int dir = 0; dir < (with_cpu ? 1 : 2); ++dir) {
        const double t0 = now_s();
        double moved = 0.0;
        while (now_s() - t0 < (quick ? 1.5 : 4.0)) {
            for (size_t off = 0; off < bytes; off += chunk) {
                const size_t nb = std::min(chunk, bytes - off);
                if (dir == 0) {
                    CK(cudaMemcpyAsync(d, h + off, nb, cudaMemcpyHostToDevice, s));
                } else {
                    CK(cudaMemcpyAsync(h + off, d, nb, cudaMemcpyDeviceToHost, s));
                }
                moved += nb;
            }
            CK(cudaStreamSynchronize(s));
        }
        const double dt = now_s() - t0;
        say("  %s from a %zu MiB pinned buffer: %.1f GB/s\n", dir == 0 ? "H2D" : "D2H", pinned_mib, moved / dt / 1e9);
    }

    if (with_cpu) {
        const double g = st.join();
        say("  CPU during the copies: %.1f GB/s (sum with H2D is the combined DRAM read ceiling)\n", g / (now_s() - t_cpu0));
#ifdef _WIN32
        VirtualFree(region, 0, MEM_RELEASE);
#else
        free(region);
#endif
    }
    CK(cudaStreamDestroy(s));
    CK(cudaFree(d));
    CK(cudaFreeHost(h));
}

int moe_bridge_selftest(int argc, char ** argv); // [TAG_MOE_BRIDGE] moe-bridge-selftest.cpp

int main(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--selftest") == 0) {
            return moe_bridge_selftest(argc, argv);
        }
    }
    std::string log_path = "E:/turbot-gates/flashnext/probe.log";
    size_t pinned_mib = 4096;
    size_t stream_mib = 8192;
    int n_threads = 16;
    uint64_t mask = 0x55555555ull;
    int responder_cpu = 30;
    bool quick = false;
    std::string only;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--log") { log_path = next(); }
        else if (a == "--pinned-mib") { pinned_mib = std::min<size_t>(4096, strtoull(next(), nullptr, 10)); }
        else if (a == "--stream-mib") { stream_mib = std::min<size_t>(8192, strtoull(next(), nullptr, 10)); }
        else if (a == "--threads") { n_threads = std::max(1, std::min(32, atoi(next()))); }
        else if (a == "--mask") { mask = strtoull(next(), nullptr, 16); }
        else if (a == "--responder-cpu") { responder_cpu = atoi(next()); }
        else if (a == "--quick") { quick = true; }
        else if (a == "--only") { only = next(); }
        else { printf("unknown argument %s\n", a.c_str()); return 1; }
    }
    if (pinned_mib < 64) {
        pinned_mib = 64;
    }
    g_log = fopen(log_path.c_str(), "a");
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    int drv = 0, rt = 0;
    cudaDriverGetVersion(&drv);
    cudaRuntimeGetVersion(&rt);
    say("\nmoe-bridge-probe [TAG_FN_BRIDGE_PROBE]: %s, cc %d.%d, driver %d, runtime %d, tcc %d, pinned %zu MiB, streamed %zu MiB\n",
            prop.name, prop.major, prop.minor, drv, rt, prop.tccDriver, pinned_mib, stream_mib);
    CK(cudaSetDeviceFlags(cudaDeviceMapHost));
    CK(cudaFree(0));

    if (only.empty() || only == "rt")     { test_roundtrip(quick, responder_cpu); }
    if (only.empty() || only == "hostfn") { test_hostfn(quick); }
    if (only.empty() || only == "memops") { test_memops(responder_cpu); }
    if (only.empty() || only == "bw")     { test_bw(pinned_mib, stream_mib, n_threads, mask, false, quick); }
    if (only.empty() || only == "bwcpu")  { test_bw(pinned_mib, stream_mib, n_threads, mask, true, quick); }

    say("\nmoe-bridge-probe done. D4: in-graph hop <= 20 us -> spin bridge; 20-40 us -> build, expect the low end;"
        " > 40 us -> check the host-function hop, else stop at F4.\n");
    if (g_log) {
        fclose(g_log);
    }
    return 0;
}
