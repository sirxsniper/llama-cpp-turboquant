// [TAG_MOE_BRIDGE] GPU <-> host doorbell for host-resident MoE experts (ggml-moe-bridge.h, PLAN.md SP-5).
//
// Memory: one cudaHostAlloc(Portable | Mapped) block per bridge, the allreduce.cu pattern:
//   glob  error word and wait statistics (device writes, host reads)
//   ring  MB_RING entries {stamp, chan, seq} in post order: one writer (the device), one reader (the executor)
//   hdr   per channel {seq, n_tokens, n_used, flags} (device) and {done} (host), on separate cache lines
//   data  per channel x, ids, w (device writes) and out (host writes)
// Device memory holds the ring counter, the sticky error and the channel sequence numbers: a captured CUDA graph
// replays fixed kernel arguments, so every replay must read and bump its counters there.
//
// Order: post writes the payload, __threadfence_system, then the header and the ring stamp. The host reads the stamp,
// then the payload (acquire), computes, writes out, then done (release). Wait polls done (volatile) and reads out with
// __ldcv, so it never uses a stale L2 line of an earlier step. One writer per word and no system-scope atomics
// (consumer GPUs on PCIe have no host-native atomics).
//
// Bounded: a wait gives up after timeout_ms if the host has not taken the job, or after job_max_ms if it has (a slow
// job, e.g. page faults on cold experts), sets the sticky error and writes zeros. Later posts of the graph then skip and
// later waits return zeros at once, so a stalled host costs one timeout per graph and never a driver watchdog reset.

#include "moe-bridge.cuh"

#include "ggml-cuda.h"
#include "ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstring>
#include <mutex>
#include <vector>

#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
#define GGML_MOE_BRIDGE_DISABLED // needs %globaltimer, __nanosleep and __ldcv
#endif

namespace {

constexpr int MB_RING            = 256;
constexpr int MB_MAX_BRIDGES     = 8;
constexpr int MB_POST_THREADS    = 512;
constexpr int MB_WAIT_THREADS    = 256;
constexpr int MB_WAIT_MAX_BLOCKS = 8;

struct alignas(64) mb_ring_entry {
    uint32_t stamp;
    uint32_t chan;
    uint32_t seq;
    uint32_t pad[13];
};

struct alignas(64) mb_chan_hdr {
    uint32_t seq; // device: the posted job
    int32_t  n_tokens;
    int32_t  n_used;
    int32_t  flags;
    uint32_t pad0[12];
    uint32_t done;  // host: the answered job
    uint32_t taken; // host: the job being computed (a wait then allows job_max_ns instead of timeout_ns)
    uint32_t pad1[14];
};

struct alignas(64) mb_glob {
    uint32_t err;
    int32_t  err_chan;
    uint32_t err_seq;
    uint32_t posted; // device: the last ring stamp (spin mode)
    uint32_t pad0[12];
    unsigned long long waits;
    unsigned long long waits_ready;
    unsigned long long wait_ns;
    uint32_t pad1[10];
};

static_assert(sizeof(mb_ring_entry) == 64,  "mb_ring_entry size");
static_assert(sizeof(mb_chan_hdr)   == 128, "mb_chan_hdr size");
static_assert(sizeof(mb_glob)       == 128, "mb_glob size");

// the device view, passed by value to the kernels
struct mb_dev {
    mb_glob       * glob;
    mb_ring_entry * ring;
    mb_chan_hdr   * hdr;
    char          * data;
    size_t          chan_bytes;
    size_t          off_ids;
    size_t          off_w;
    size_t          off_out;
    uint32_t      * state; // device memory: [0] ring counter, [1] sticky error, [2 + chan] channel sequence
    int             n_chan;
    int             mode;
    int             stats;
    unsigned long long timeout_ns;
    unsigned long long job_max_ns;
};

size_t mb_pad(size_t n) {
    return (n + 63) & ~(size_t) 63;
}

} // namespace

struct ggml_moe_bridge {
    int32_t id = -1;
    ggml_moe_bridge_params params = {};

    uint8_t * host     = nullptr; // the mapped block, host view
    uint8_t * host_dev = nullptr; // the same block, device view

    mb_glob       * glob = nullptr; // host views
    mb_ring_entry * ring = nullptr;
    mb_chan_hdr   * hdr  = nullptr;
    char          * data = nullptr;

    uint32_t * state = nullptr; // device memory
    mb_dev     dev   = {};

    cudaStream_t aux = nullptr; // reset, and the host function nodes of hostfunc mode
    std::vector<cudaEvent_t> ev_post;
    std::vector<cudaEvent_t> ev_done;

    struct hf_arg {
        ggml_moe_bridge * b;
        int32_t  chan;
        uint32_t last;
    };
    std::vector<hf_arg> hf;

    ggml_moe_bridge_runner_t runner    = nullptr;
    void *                   runner_ud = nullptr;

    // host side: in spin mode one executor thread polls and completes
    std::atomic<uint32_t> served{0};
    std::atomic<uint64_t> taken{0};
    std::atomic<uint64_t> completed{0};
    std::atomic<uint32_t> n_lapped{0};
};

static std::mutex                       g_mb_mtx;
static std::atomic<ggml_moe_bridge *>   g_mb[MB_MAX_BRIDGES];

static ggml_moe_bridge * mb_get(int32_t id) {
    if (id < 0 || id >= MB_MAX_BRIDGES) {
        return nullptr;
    }
    return g_mb[id].load(std::memory_order_acquire);
}

#ifndef GGML_MOE_BRIDGE_DISABLED

static __device__ __forceinline__ unsigned long long mb_now_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

static __device__ __forceinline__ void mb_sleep() {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    __nanosleep(256);
#endif
}

// one block: take the channel's next sequence number, copy x / ids / w to the channel, fence, publish
static __global__ void k_mb_post(const mb_dev v, const int chan,
        const char * __restrict__ x, const int64_t x_nb1,
        const char * __restrict__ ids, const int64_t ids_nb1,
        const float * __restrict__ w,
        const int n_embd, const int n_used, const int n_tokens, const int flags, int32_t * __restrict__ ticket) {
    __shared__ uint32_t s_seq;
    const int tid = threadIdx.x;
    if (tid == 0) {
        uint32_t s = 0;
        if (*(volatile uint32_t *) &v.state[1] == 0) {
            s = v.state[2 + chan] + 1;
            if (s == 0) {
                s = 1;
            }
            v.state[2 + chan] = s;
        }
        s_seq = s;
    }
    __syncthreads();
    const uint32_t seq = s_seq;
    if (seq == 0) { // sticky error: nothing is posted, the wait returns zeros
        if (tid == 0) {
            ticket[0] = 0;
        }
        return;
    }

    char    * p    = v.data + (size_t) chan*v.chan_bytes;
    float   * hx   = (float   *) p;
    int32_t * hids = (int32_t *) (p + v.off_ids);
    float   * hw   = (float   *) (p + v.off_w);

    if ((n_embd & 3) == 0 && (x_nb1 & 15) == 0 && ((uintptr_t) x & 15) == 0) {
        const int n4 = n_embd/4;
        for (int i = tid; i < n4*n_tokens; i += blockDim.x) {
            const int t = i/n4;
            ((float4 *) hx)[i] = ((const float4 *) (x + t*x_nb1))[i - t*n4];
        }
    } else {
        for (int i = tid; i < n_embd*n_tokens; i += blockDim.x) {
            const int t = i/n_embd;
            hx[i] = ((const float *) (x + t*x_nb1))[i - t*n_embd];
        }
    }
    for (int i = tid; i < n_used*n_tokens; i += blockDim.x) {
        const int t = i/n_used;
        hids[i] = ((const int32_t *) (ids + t*ids_nb1))[i - t*n_used];
        hw[i]   = w[i];
    }

    __threadfence_system();
    __syncthreads();

    if (tid == 0) {
        mb_chan_hdr * h = v.hdr + chan;
        *(volatile int32_t *) &h->n_tokens = n_tokens;
        *(volatile int32_t *) &h->n_used   = n_used;
        *(volatile int32_t *) &h->flags    = flags;
        __threadfence_system();
        *(volatile uint32_t *) &h->seq = seq;
        if (v.mode == GGML_MOE_BRIDGE_WAIT_SPIN) {
            const uint32_t g = atomicAdd(&v.state[0], 1u) + 1u;
            mb_ring_entry * e = v.ring + (g % MB_RING);
            *(volatile uint32_t *) &e->chan = (uint32_t) chan;
            *(volatile uint32_t *) &e->seq  = seq;
            __threadfence_system();
            *(volatile uint32_t *) &e->stamp        = g;
            *(volatile uint32_t *) &v.glob->posted  = g;
        }
        __threadfence_system();
        ticket[0] = (int32_t) seq;
    }
}

// thread 0 of each block: bounded poll of the channel's done flag; then all threads copy the result (or zeros)
static __global__ void k_mb_wait(const mb_dev v, const int chan, const int32_t * __restrict__ ticket,
        float * __restrict__ dst, const int n) {
    __shared__ int s_ok;
    if (threadIdx.x == 0) {
        const uint32_t want = (uint32_t) ticket[0];
        int ok = 0;
        if (want != 0) {
            const volatile uint32_t * done  = &v.hdr[chan].done;
            const volatile uint32_t * taken = &v.hdr[chan].taken;
            const volatile uint32_t * err   = &v.state[1];
            const unsigned long long t0 = mb_now_ns();
            bool ready = true;
            for (;;) {
                if ((int32_t) (*done - want) >= 0) {
                    ok = 1;
                    break;
                }
                if (*err != 0) {
                    break;
                }
                const unsigned long long el = mb_now_ns() - t0;
                if (el > v.timeout_ns && ((int32_t) (*taken - want) < 0 || el > v.job_max_ns)) {
                    if (atomicCAS(&v.state[1], 0u, (unsigned int) GGML_MOE_BRIDGE_ERR_TIMEOUT) == 0u) {
                        *(volatile int32_t  *) &v.glob->err_chan = chan;
                        *(volatile uint32_t *) &v.glob->err_seq  = want;
                        __threadfence_system();
                        *(volatile uint32_t *) &v.glob->err = GGML_MOE_BRIDGE_ERR_TIMEOUT;
                        __threadfence_system();
                    }
                    break;
                }
                ready = false;
                mb_sleep();
            }
            if (v.stats && blockIdx.x == 0) {
                volatile mb_glob * g = v.glob;
                g->waits       = g->waits + 1;
                g->waits_ready = g->waits_ready + (ready && ok ? 1 : 0);
                g->wait_ns     = g->wait_ns + (mb_now_ns() - t0);
            }
        }
        s_ok = ok;
    }
    __syncthreads();
    __threadfence_system(); // acquire: the host wrote out before done

    const float * out    = (const float *) (v.data + (size_t) chan*v.chan_bytes + v.off_out);
    const int     stride = gridDim.x*blockDim.x;
    const int     i0     = blockIdx.x*blockDim.x + threadIdx.x;
    if (!s_ok) {
        for (int i = i0; i < n; i += stride) {
            dst[i] = 0.0f;
        }
    } else if ((n & 3) == 0) {
        for (int i = i0; i < n/4; i += stride) {
            ((float4 *) dst)[i] = __ldcv((const float4 *) out + i);
        }
    } else {
        for (int i = i0; i < n; i += stride) {
            dst[i] = __ldcv(out + i);
        }
    }
}

static void mb_make_job(const ggml_moe_bridge * b, int32_t chan, uint32_t seq, ggml_moe_bridge_job * job) {
    const mb_chan_hdr * h = b->hdr + chan;
    char * p = b->data + (size_t) chan*b->dev.chan_bytes;
    job->chan     = chan;
    job->seq      = seq;
    job->n_tokens = *(const volatile int32_t *) &h->n_tokens;
    job->n_used   = *(const volatile int32_t *) &h->n_used;
    job->flags    = *(const volatile int32_t *) &h->flags;
    job->n_embd   = b->params.n_embd;
    job->x        = (const float *)   p;
    job->ids      = (const int32_t *) (p + b->dev.off_ids);
    job->w        = (const float *)   (p + b->dev.off_w);
    job->out      = (float *)         (p + b->dev.off_out);
}

// the host has started this job: its wait now allows job_max_ns
static void mb_mark_taken(ggml_moe_bridge * b, const ggml_moe_bridge_job * job) {
    *(volatile uint32_t *) &b->hdr[job->chan].taken = job->seq;
}

static bool mb_job_valid(const ggml_moe_bridge * b, const ggml_moe_bridge_job * job) {
    return job->n_tokens >= 1 && job->n_tokens <= b->params.max_tokens && job->n_used >= 1 && job->n_used <= b->params.n_used;
}

// hostfunc mode: runs on the driver's callback thread when the stream reaches it, after this channel's post
static void CUDART_CB mb_hostfn(void * ud) {
    auto * a = (ggml_moe_bridge::hf_arg *) ud;
    ggml_moe_bridge * b = a->b;
    const uint32_t seq = *(volatile uint32_t *) &b->hdr[a->chan].seq;
    if (seq == a->last) {
        return; // the post was skipped (sticky error): nothing new to run
    }
    a->last = seq;
    std::atomic_thread_fence(std::memory_order_acquire);
    ggml_moe_bridge_job job;
    mb_make_job(b, a->chan, seq, &job);
    mb_mark_taken(b, &job);
    b->taken.fetch_add(1, std::memory_order_relaxed);
    const bool ok = mb_job_valid(b, &job) && b->runner && b->runner(&job, b->runner_ud);
    ggml_backend_cuda_moe_bridge_complete(b, &job, ok);
}

#endif // GGML_MOE_BRIDGE_DISABLED

// ---- public API -----------------------------------------------------------------------------------------------------

static void mb_destroy(ggml_moe_bridge * b) {
    if (b == nullptr) {
        return;
    }
    if (b->id >= 0) {
        std::lock_guard<std::mutex> lk(g_mb_mtx);
        g_mb[b->id].store(nullptr, std::memory_order_release);
    }
    ggml_cuda_set_device(b->params.device);
    if (b->aux) {
        (void) cudaStreamSynchronize(b->aux);
    }
    for (auto e : b->ev_post) {
        if (e) {
            (void) cudaEventDestroy(e);
        }
    }
    for (auto e : b->ev_done) {
        if (e) {
            (void) cudaEventDestroy(e);
        }
    }
    if (b->aux) {
        (void) cudaStreamDestroy(b->aux);
    }
    if (b->state) {
        (void) cudaFree(b->state);
    }
    if (b->host) {
        (void) cudaFreeHost(b->host);
    }
    (void) cudaGetLastError();
    delete b;
}

ggml_moe_bridge * ggml_backend_cuda_moe_bridge_new(const ggml_moe_bridge_params * p) {
#ifdef GGML_MOE_BRIDGE_DISABLED
    GGML_UNUSED(p);
    return nullptr;
#else
    if (p == nullptr || p->device < 0 || p->device >= ggml_backend_cuda_get_device_count() ||
        p->n_chan < 1 || p->n_chan > 4096 || p->n_embd < 1 || p->n_embd > (1 << 20) || p->n_used < 1 || p->n_used > 256 ||
        p->max_tokens < 1 || p->max_tokens > 64 ||
        (p->wait_mode != GGML_MOE_BRIDGE_WAIT_SPIN && p->wait_mode != GGML_MOE_BRIDGE_WAIT_HOSTFUNC)) {
        GGML_LOG_ERROR("%s: invalid parameters\n", __func__);
        return nullptr;
    }
    int can_map = 0;
    if (cudaDeviceGetAttribute(&can_map, cudaDevAttrCanMapHostMemory, p->device) != cudaSuccess || !can_map) {
        (void) cudaGetLastError();
        GGML_LOG_WARN("%s: device %d cannot map host memory\n", __func__, p->device);
        return nullptr;
    }
    ggml_cuda_set_device(p->device);

    auto * b = new ggml_moe_bridge();
    b->params = *p;
    b->params.timeout_ms = std::min(10000, std::max(1, p->timeout_ms));
    b->params.job_max_ms = p->job_max_ms > 0 ? std::min(10000, std::max(b->params.timeout_ms, p->job_max_ms))
                                             : std::max(b->params.timeout_ms, 1000);

    const int64_t n_embd = p->n_embd;
    const int     max_t  = p->max_tokens;
    const size_t  sx     = mb_pad((size_t) n_embd*max_t*sizeof(float));
    const size_t  si     = mb_pad((size_t) p->n_used*max_t*sizeof(int32_t));
    const size_t  chan_bytes = sx + si + si + sx;

    const size_t off_ring = mb_pad(sizeof(mb_glob));
    const size_t off_hdr  = off_ring + (size_t) MB_RING*sizeof(mb_ring_entry);
    const size_t off_data = off_hdr  + (size_t) p->n_chan*sizeof(mb_chan_hdr);
    const size_t total    = off_data + (size_t) p->n_chan*chan_bytes;

    auto fail = [&](const char * what, cudaError_t rc) -> ggml_moe_bridge * {
        GGML_LOG_WARN("%s: %s failed: %s\n", __func__, what, cudaGetErrorString(rc));
        (void) cudaGetLastError();
        mb_destroy(b);
        return nullptr;
    };

    cudaError_t rc = cudaHostAlloc((void **) &b->host, total, cudaHostAllocPortable | cudaHostAllocMapped);
    if (rc != cudaSuccess) {
        b->host = nullptr;
        return fail("cudaHostAlloc", rc);
    }
    memset(b->host, 0, total);
    rc = cudaHostGetDevicePointer((void **) &b->host_dev, b->host, 0);
    if (rc != cudaSuccess) {
        return fail("cudaHostGetDevicePointer", rc);
    }
    b->glob = (mb_glob *)       (b->host);
    b->ring = (mb_ring_entry *) (b->host + off_ring);
    b->hdr  = (mb_chan_hdr *)   (b->host + off_hdr);
    b->data = (char *)          (b->host + off_data);
    b->glob->err_chan = -1;

    rc = cudaMalloc((void **) &b->state, (2 + (size_t) p->n_chan)*sizeof(uint32_t));
    if (rc != cudaSuccess) {
        b->state = nullptr;
        return fail("cudaMalloc", rc);
    }
    rc = cudaStreamCreateWithFlags(&b->aux, cudaStreamNonBlocking);
    if (rc != cudaSuccess) {
        b->aux = nullptr;
        return fail("cudaStreamCreateWithFlags", rc);
    }
    rc = cudaMemsetAsync(b->state, 0, (2 + (size_t) p->n_chan)*sizeof(uint32_t), b->aux);
    if (rc == cudaSuccess) {
        rc = cudaStreamSynchronize(b->aux);
    }
    if (rc != cudaSuccess) {
        return fail("clearing the device counters", rc);
    }

    if (p->wait_mode == GGML_MOE_BRIDGE_WAIT_HOSTFUNC) {
        b->ev_post.assign(p->n_chan, nullptr);
        b->ev_done.assign(p->n_chan, nullptr);
        b->hf.resize(p->n_chan);
        for (int c = 0; c < p->n_chan; ++c) {
            rc = cudaEventCreateWithFlags(&b->ev_post[c], cudaEventDisableTiming);
            if (rc == cudaSuccess) {
                rc = cudaEventCreateWithFlags(&b->ev_done[c], cudaEventDisableTiming);
            }
            if (rc != cudaSuccess) {
                return fail("cudaEventCreateWithFlags", rc);
            }
            b->hf[c] = { b, c, 0 };
        }
    }

    const uint8_t * base = b->host_dev;
    b->dev.glob       = (mb_glob *)       (base);
    b->dev.ring       = (mb_ring_entry *) (base + off_ring);
    b->dev.hdr        = (mb_chan_hdr *)   (base + off_hdr);
    b->dev.data       = (char *)          (base + off_data);
    b->dev.chan_bytes = chan_bytes;
    b->dev.off_ids    = sx;
    b->dev.off_w      = sx + si;
    b->dev.off_out    = sx + si + si;
    b->dev.state      = b->state;
    b->dev.n_chan     = p->n_chan;
    b->dev.mode       = p->wait_mode;
    b->dev.stats      = p->stats ? 1 : 0;
    b->dev.timeout_ns = (unsigned long long) b->params.timeout_ms*1000000ull;
    b->dev.job_max_ns = (unsigned long long) b->params.job_max_ms*1000000ull;

    {
        std::lock_guard<std::mutex> lk(g_mb_mtx);
        for (int i = 0; i < MB_MAX_BRIDGES; ++i) {
            if (g_mb[i].load(std::memory_order_acquire) == nullptr) {
                b->id = i;
                g_mb[i].store(b, std::memory_order_release);
                break;
            }
        }
    }
    if (b->id < 0) {
        return fail("registering (too many bridges)", cudaSuccess);
    }

    GGML_LOG_INFO("%s: bridge %d on device %d: %d channels, n_embd %" PRId64 ", n_used %d, T <= %d, %s wait, timeout %d ms (%d ms once taken), %.2f MiB mapped\n",
            __func__, b->id, p->device, p->n_chan, n_embd, p->n_used, max_t,
            p->wait_mode == GGML_MOE_BRIDGE_WAIT_SPIN ? "spin" : "hostfunc", b->params.timeout_ms, b->params.job_max_ms, total/1024.0/1024.0);
    return b;
#endif
}

void ggml_backend_cuda_moe_bridge_free(ggml_moe_bridge * b) {
    mb_destroy(b);
}

int32_t ggml_backend_cuda_moe_bridge_id(const ggml_moe_bridge * b) {
    return b ? b->id : -1;
}

void ggml_backend_cuda_moe_bridge_set_runner(ggml_moe_bridge * b, ggml_moe_bridge_runner_t runner, void * user_data) {
    if (b) {
        b->runner    = runner;
        b->runner_ud = user_data;
    }
}

bool ggml_backend_cuda_moe_bridge_poll(ggml_moe_bridge * b, ggml_moe_bridge_job * job) {
#ifdef GGML_MOE_BRIDGE_DISABLED
    GGML_UNUSED(b);
    GGML_UNUSED(job);
    return false;
#else
    const uint32_t next = b->served.load(std::memory_order_relaxed) + 1;
    const mb_ring_entry * e = b->ring + (next % MB_RING);
    const uint32_t st = *(const volatile uint32_t *) &e->stamp;
    if (st != next) {
        if ((int32_t) (st - next) > 0) {
            // the device lapped the ring: the jobs before st are lost and their waits time out; resume at st
            if (b->n_lapped.fetch_add(1, std::memory_order_relaxed) < 8) {
                GGML_LOG_WARN("%s: bridge %d: the executor fell %u posts behind, resuming at %u\n", __func__, b->id, st - next, st);
            }
            b->served.store(st - 1, std::memory_order_release);
        }
        return false;
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    const uint32_t chan = *(const volatile uint32_t *) &e->chan;
    const uint32_t seq  = *(const volatile uint32_t *) &e->seq;
    b->served.store(next, std::memory_order_release);
    if (chan >= (uint32_t) b->params.n_chan) {
        return false;
    }
    mb_make_job(b, (int32_t) chan, seq, job);
    mb_mark_taken(b, job);
    b->taken.fetch_add(1, std::memory_order_relaxed);
    if (!mb_job_valid(b, job)) {
        ggml_backend_cuda_moe_bridge_complete(b, job, false);
        return false;
    }
    return true;
#endif
}

void ggml_backend_cuda_moe_bridge_complete(ggml_moe_bridge * b, const ggml_moe_bridge_job * job, bool ok) {
    if (!ok) {
        const int T = std::min(std::max(job->n_tokens, 0), b->params.max_tokens);
        memset(job->out, 0, (size_t) b->params.n_embd*T*sizeof(float));
        if (*(volatile uint32_t *) &b->glob->err == 0) {
            *(volatile int32_t  *) &b->glob->err_chan = job->chan;
            *(volatile uint32_t *) &b->glob->err_seq  = job->seq;
            std::atomic_thread_fence(std::memory_order_release);
            *(volatile uint32_t *) &b->glob->err = GGML_MOE_BRIDGE_ERR_RUNNER;
        }
    }
    std::atomic_thread_fence(std::memory_order_release);
    *(volatile uint32_t *) &b->hdr[job->chan].done = job->seq;
    b->completed.fetch_add(1, std::memory_order_release);
}

uint32_t ggml_backend_cuda_moe_bridge_error(const ggml_moe_bridge * b) {
    return b ? *(const volatile uint32_t *) &b->glob->err : (uint32_t) GGML_MOE_BRIDGE_ERR_NONE;
}

bool ggml_backend_cuda_moe_bridge_reset(ggml_moe_bridge * b) {
#ifdef GGML_MOE_BRIDGE_DISABLED
    GGML_UNUSED(b);
    return false;
#else
    if (b->params.wait_mode == GGML_MOE_BRIDGE_WAIT_SPIN) {
        const uint32_t posted = *(const volatile uint32_t *) &b->glob->posted;
        if (posted != b->served.load(std::memory_order_acquire) ||
            b->completed.load(std::memory_order_acquire) != b->taken.load(std::memory_order_acquire)) {
            return false; // the executor still owes a job
        }
    }
    ggml_cuda_set_device(b->params.device);
    cudaError_t rc = cudaMemsetAsync(b->state + 1, 0, sizeof(uint32_t), b->aux);
    if (rc == cudaSuccess) {
        rc = cudaStreamSynchronize(b->aux);
    }
    if (rc != cudaSuccess) {
        GGML_LOG_WARN("%s: bridge %d: clearing the device error failed: %s\n", __func__, b->id, cudaGetErrorString(rc));
        (void) cudaGetLastError();
        return false;
    }
    *(volatile int32_t  *) &b->glob->err_chan = -1;
    *(volatile uint32_t *) &b->glob->err_seq  = 0;
    std::atomic_thread_fence(std::memory_order_release);
    *(volatile uint32_t *) &b->glob->err = GGML_MOE_BRIDGE_ERR_NONE;
    return true;
#endif
}

void ggml_backend_cuda_moe_bridge_get_stats(const ggml_moe_bridge * b, ggml_moe_bridge_stats * s) {
    memset(s, 0, sizeof(*s));
    if (b == nullptr) {
        return;
    }
    const volatile mb_glob * g = b->glob;
    s->posts       = b->params.wait_mode == GGML_MOE_BRIDGE_WAIT_SPIN ? (uint64_t) b->served.load() : b->taken.load();
    s->completed   = b->completed.load();
    s->waits       = g->waits;
    s->waits_ready = g->waits_ready;
    s->wait_ns     = g->wait_ns;
    s->error       = g->err;
    s->error_chan  = g->err_chan;
}

// ---- ops ------------------------------------------------------------------------------------------------------------

bool ggml_cuda_moe_bridge_supports_op(int device, const ggml_tensor * op) {
#ifdef GGML_MOE_BRIDGE_DISABLED
    GGML_UNUSED(device);
    GGML_UNUSED(op);
    return false;
#else
    const ggml_moe_bridge * b = mb_get(ggml_get_op_params_i32(op, 0));
    const int32_t chan = ggml_get_op_params_i32(op, 1);
    if (b == nullptr || b->params.device != device || chan < 0 || chan >= b->params.n_chan) {
        return false;
    }
    const auto & p = b->params;
    if (op->op == GGML_OP_MOE_HOST_POST) {
        const ggml_tensor * x   = op->src[0];
        const ggml_tensor * ids = op->src[1];
        const ggml_tensor * w   = op->src[2];
        return x->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32 && w->type == GGML_TYPE_F32 &&
            x->ne[0] == p.n_embd && x->ne[1] >= 1 && x->ne[1] <= p.max_tokens && x->nb[0] == sizeof(float) &&
            ids->ne[0] >= 1 && ids->ne[0] <= p.n_used && ids->nb[0] == sizeof(int32_t) && ggml_is_contiguous(w);
    }
    if (op->op == GGML_OP_MOE_HOST_WAIT) {
        return op->type == GGML_TYPE_F32 && op->ne[0] == p.n_embd && op->ne[1] >= 1 && op->ne[1] <= p.max_tokens &&
            ggml_is_contiguous(op);
    }
    return false;
#endif
}

void ggml_cuda_op_moe_host_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#ifdef GGML_MOE_BRIDGE_DISABLED
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("MOE_HOST_POST is not supported on this backend");
#else
    ggml_moe_bridge * b = mb_get(ggml_get_op_params_i32(dst, 0));
    GGML_ASSERT(b != nullptr && b->params.device == ctx.device);
    const int32_t chan  = ggml_get_op_params_i32(dst, 1);
    const int32_t flags = ggml_get_op_params_i32(dst, 2);

    const ggml_tensor * x   = dst->src[0];
    const ggml_tensor * ids = dst->src[1];
    const ggml_tensor * w   = dst->src[2];

    cudaStream_t stream = ctx.stream();
    k_mb_post<<<1, MB_POST_THREADS, 0, stream>>>(b->dev, chan,
            (const char *) x->data, x->nb[1], (const char *) ids->data, ids->nb[1], (const float *) w->data,
            (int) x->ne[0], (int) ids->ne[0], (int) x->ne[1], flags, (int32_t *) dst->data);
    CUDA_CHECK(cudaGetLastError());

    if (b->params.wait_mode == GGML_MOE_BRIDGE_WAIT_HOSTFUNC) {
        // fork: the host job runs on the aux stream while this stream continues with the device work of the layer
        CUDA_CHECK(cudaEventRecord(b->ev_post[chan], stream));
        CUDA_CHECK(cudaStreamWaitEvent(b->aux, b->ev_post[chan], 0));
        CUDA_CHECK(cudaLaunchHostFunc(b->aux, mb_hostfn, &b->hf[chan]));
        CUDA_CHECK(cudaEventRecord(b->ev_done[chan], b->aux));
    }
#endif
}

void ggml_cuda_op_moe_host_wait(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#ifdef GGML_MOE_BRIDGE_DISABLED
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("MOE_HOST_WAIT is not supported on this backend");
#else
    ggml_moe_bridge * b = mb_get(ggml_get_op_params_i32(dst, 0));
    GGML_ASSERT(b != nullptr && b->params.device == ctx.device);
    const int32_t chan = ggml_get_op_params_i32(dst, 1);

    cudaStream_t stream = ctx.stream();
    if (b->params.wait_mode == GGML_MOE_BRIDGE_WAIT_HOSTFUNC) {
        CUDA_CHECK(cudaStreamWaitEvent(stream, b->ev_done[chan], 0)); // join the fork of the post
    }

    const int n      = (int) ggml_nelements(dst);
    const int n_vec  = (n & 3) == 0 ? n/4 : n;
    const int blocks = std::min(MB_WAIT_MAX_BLOCKS, std::max(1, (n_vec + MB_WAIT_THREADS - 1)/MB_WAIT_THREADS));
    k_mb_wait<<<blocks, MB_WAIT_THREADS, 0, stream>>>(b->dev, chan, (const int32_t *) dst->src[0]->data, (float *) dst->data, n);
    CUDA_CHECK(cudaGetLastError());
#endif
}
