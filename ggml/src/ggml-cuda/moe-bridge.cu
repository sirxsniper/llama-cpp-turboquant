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
//
// [TAG_FN_R4_BRIDGE_DMA] The PCIe share: a job posted with GGML_MOE_BRIDGE_JOB_DMA carries a plan area (mapped). The
// host writes the plan (ring offsets and bank slots of the experts the GPU takes, and the bank slot of every routed
// (slot, token)) and publishes it (hdr.plan, a release store) before it computes the rest. GGML_OP_MOE_HOST_FETCH waits
// (bounded, as a wait) for the plan, copies it into device memory, and SM copy blocks read the experts from the pinned
// ring through its device mapping (ld.global.cv: a refilled ring slot is never served from a stale L2 line) into the
// bank slots. The host makes no CUDA call inside a running graph (the Strata #31 driver deadlock: a host copy issued
// while the GPU spins on a mapped flag). The device writes its wait and fetch times per channel into the mapped header,
// for the host's DMA/CPU split. glob.release (host memory) makes every wait and fetch give up at once (exit paths).
//
// [TAG_FN_L3_CPU_DEVPRED] A hint (a MOE_HOST_POST node with op param 3 = 1, ggml_moe_host_hint) runs after a layer's
// post: it copies the device's predicted ids of the next layer (the top-k of that layer's router on this layer's input)
// into the channel's hint area under the post's ticket, seqlock style (seq 0, ids, fence, seq). The executor reads them
// after the job (ggml_backend_moe_bridge_read_hint) and prefetches the next layer's experts without a router on the host.
//
// [TAG_FN_L16_PLEGATE] A channel can also carry jobs that no device post makes: the graph's ticket is an input, and a
// host thread takes the job (host_take: owed, taken), writes out and completes it. The wait is the same k_mb_wait.

#include "moe-bridge.cuh"
#include "topk-moe.cuh" // [TAG_FN_L4_POST]
#include "fn-l22-dmafetch.cuh" // [TAG_FN_L22_DMASHARE]

#include "ggml-cuda.h"
#include "ggml-fn-l4-gpumoe.h" // [TAG_FN_L4_POST]
#include "ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cfloat>
#include <cinttypes>
#include <cstddef>
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
    unsigned long long wait_ns;  // [TAG_FN_R4_BRIDGE_DMA] device: the last wait for the host's result
    unsigned long long fetch_ns; // [TAG_FN_R4_BRIDGE_DMA] device: the last fetch, plan to last copied byte
    uint32_t n_fetch;            // [TAG_FN_R4_BRIDGE_DMA] device: experts the last fetch copied
    uint32_t times_seq;          // [TAG_FN_R4_BRIDGE_DMA] device: the job of these times
    unsigned long long t_post;   // [TAG_FN_L14_PROBE] device clock: the job's post
    unsigned long long t_wstart; // [TAG_FN_L14_PROBE] device clock: the wait for its result began / ended
    unsigned long long t_wend;
    uint32_t done;  // host: the answered job
    uint32_t taken; // host: the job being computed (a wait then allows job_max_ns instead of timeout_ns)
    uint32_t plan;  // [TAG_FN_R4_BRIDGE_DMA] host: the job whose plan is published
    uint32_t pad1a;
    unsigned long long t_pub; // [TAG_FN_L14_PROBE] device clock: the post's stamp was published (the host may see it)
    uint32_t pad1[10];
};

struct alignas(64) mb_glob {
    uint32_t err;
    int32_t  err_chan;
    uint32_t err_seq;
    uint32_t posted; // device: the last ring stamp (spin mode)
    uint32_t release; // [TAG_FN_R4_BRIDGE_DMA] host: nonzero = every wait and fetch gives up at once
    uint32_t pad0[11];
    unsigned long long waits;
    unsigned long long waits_ready;
    unsigned long long wait_ns;
    uint32_t pad1[10];
};

static_assert(sizeof(mb_ring_entry) == 64,  "mb_ring_entry size");
static_assert(sizeof(mb_chan_hdr)   == 128, "mb_chan_hdr size");
static_assert(sizeof(mb_glob)       == 128, "mb_glob size");

// [TAG_FN_L3_CPU_DEVPRED] the head of a channel's hint area; the ids [n_tokens][k] follow it
struct alignas(64) mb_hint_hdr {
    uint32_t seq;      // the job whose prediction the ids are; 0 while the device writes them
    int32_t  k;
    int32_t  n_tokens;
    uint32_t pad[13];
};
static_assert(sizeof(mb_hint_hdr) == 64, "mb_hint_hdr size");

// [TAG_FN_R4_BRIDGE_DMA] the plan of a channel's last fetch, in device memory (the copy blocks read it there)
struct mb_fetch_dev {
    int32_t            n_copy;
    uint32_t           done_blocks; // copy blocks finished (the last one writes the times)
    uint32_t           seq;
    uint32_t           pad;
    unsigned long long t0;          // device time the plan was taken
    unsigned long long off [GGML_MOE_BRIDGE_MAX_FETCH];
    int32_t            slot[GGML_MOE_BRIDGE_MAX_FETCH];
};

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
    size_t          off_plan;  // [TAG_FN_R4_BRIDGE_DMA] the channel's plan area (0 without fetches)
    size_t          off_hint;  // [TAG_FN_L3_CPU_DEVPRED] the channel's hint area (0 without hints)
    size_t          off_raw;   // [TAG_FN_L22_X_PRIOR] the channel's raw-id area: the unbiased top-k (0: none)
    int             hint_k;    // [TAG_FN_L3_CPU_DEVPRED] the largest k of a hint (0: no hints)
    mb_fetch_dev  * fetch;     // [TAG_FN_R4_BRIDGE_DMA] device memory, one per channel (nullptr without fetches)
    const char    * dma_ring;  // [TAG_FN_R4_BRIDGE_DMA] the registered DMA ring, device view (nullptr: none)
    unsigned long long dma_ring_size;
    int             max_fetch;
    uint32_t      * state; // device memory: [0] ring counter, [1] sticky error, [2 + chan] channel sequence
    uint32_t      * l4sync; // [TAG_FN_L4_POST] device memory after state: per channel 4 words (blocks done, seq, stamp, -)
    int             n_chan;
    int             mode;
    int             stats;
    int             probe;     // [TAG_FN_L14_PROBE] LLAMA_FN_L14_PROBE=1: the device stamps its posts and waits (else no stamps)
    unsigned long long timeout_ns;
    unsigned long long job_max_ns;
};

size_t mb_pad(size_t n) {
    return (n + 63) & ~(size_t) 63;
}

// [TAG_FN_L3_HOST_EXEC] the ring stamp after s: 0 is skipped (the ring watch marks a taken entry with 0)
__host__ __device__ inline uint32_t mb_next_stamp(uint32_t s) {
    return s + 1u != 0u ? s + 1u : 1u;
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

    mb_fetch_dev * fetch = nullptr; // [TAG_FN_R4_BRIDGE_DMA] device memory, n_chan entries

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

    // [TAG_FN_L22_X_PRIOR] R-REV-9: a multi-row prior route wrote the unbiased top-k into the channel's raw-id area for the
    // post whose ids it picked (that ids pointer); the post's launch then flags the job RAW (launch time, so a CUDA graph
    // replays the same flags)
    std::vector<const void *> fn_l22_raw_ids;

    // host side: in spin mode one executor thread polls and completes
    std::atomic<uint32_t> served{0};
    std::atomic<uint64_t> taken{0};
    std::atomic<uint64_t> completed{0};
    std::atomic<uint32_t> n_lapped{0};
    bool                  watch = false; // [TAG_FN_L3_HOST_EXEC] poll zeroes the stamps it takes
};

static std::mutex                       g_mb_mtx;
static std::atomic<ggml_moe_bridge *>   g_mb[MB_MAX_BRIDGES];
static std::atomic<uint32_t>            g_mb_gen{0}; // [TAG_FN_BRIDGE_GRAPHGEN]

uint32_t ggml_cuda_moe_bridge_generation() {
    return g_mb_gen.load(std::memory_order_acquire);
}

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
        if (v.probe) {
            *(volatile unsigned long long *) &h->t_post = mb_now_ns(); // [TAG_FN_L14_PROBE]
        }
        __threadfence_system();
        *(volatile uint32_t *) &h->seq = seq;
        if (v.mode == GGML_MOE_BRIDGE_WAIT_SPIN) {
            uint32_t g = atomicAdd(&v.state[0], 1u) + 1u;
            if (g == 0u) { // [TAG_FN_L3_HOST_EXEC] stamp 0 is never a post (mb_next_stamp)
                g = atomicAdd(&v.state[0], 1u) + 1u;
            }
            mb_ring_entry * e = v.ring + (g % MB_RING);
            *(volatile uint32_t *) &e->chan = (uint32_t) chan;
            *(volatile uint32_t *) &e->seq  = seq;
            if (v.probe) {
                *(volatile unsigned long long *) &h->t_pub = mb_now_ns(); // [TAG_FN_L14_PROBE]
            }
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
            const volatile uint32_t * rel = &v.glob->release; // [TAG_FN_R4_BRIDGE_DMA]
            for (;;) {
                if ((int32_t) (*done - want) >= 0) {
                    ok = 1;
                    break;
                }
                if (*err != 0 || *rel != 0) {
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
            const unsigned long long dt = mb_now_ns() - t0;
            if (v.stats && blockIdx.x == 0) {
                volatile mb_glob * g = v.glob;
                g->waits       = g->waits + 1;
                g->waits_ready = g->waits_ready + (ready && ok ? 1 : 0);
                g->wait_ns     = g->wait_ns + dt;
            }
            if (blockIdx.x == 0) { // [TAG_FN_R4_BRIDGE_DMA] the host's DMA/CPU split reads this
                volatile mb_chan_hdr * h = v.hdr + chan;
                h->wait_ns   = ready ? 0ull : dt;
                if (v.probe) { // [TAG_FN_L14_PROBE]
                    h->t_wstart = t0;
                    h->t_wend   = t0 + dt;
                }
                h->times_seq = want;
            }
        }
        // acquire: the host wrote out before done. The fence is in the thread that read done and comes before the
        // barrier that hands the result to the other threads, so their reads of out are ordered after the host's writes.
        __threadfence_system();
        s_ok = ok;
    }
    __syncthreads();

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

// [TAG_FN_L18_GPUFLOOR_PFWAIT] the ranges ggml-cuda.cu handed for the next marked wait of each device
struct fn_l18_pf_pending {
    const ggml_tensor * wait = nullptr;
    const char *        p[GGML_CUDA_FN_L18_PF_MAX];
    size_t              n[GGML_CUDA_FN_L18_PF_MAX];
    int                 nr = 0;
};
static fn_l18_pf_pending g_fn_l18_pf[GGML_CUDA_MAX_DEVICES];

void ggml_cuda_fn_l18_pf_set(int device, const ggml_tensor * wait, const char * const * p, const size_t * n, int nr) {
    fn_l18_pf_pending & pf = g_fn_l18_pf[device];
    pf.nr = 0;
    for (int k = 0; k < nr && k < GGML_CUDA_FN_L18_PF_MAX; ++k) {
        pf.p[pf.nr] = p[k];
        pf.n[pf.nr] = n[k];
        pf.nr++;
    }
    pf.wait = wait;
}

// [TAG_FN_L18_GPUFLOOR_PFWAIT] the byte ranges the loader blocks of a wait read into L2 (16-byte aligned, multiples of 16)
#define FN_L18_PF_CHUNK (MB_WAIT_THREADS*64) // bytes per block and pass: 4 x 16 bytes per thread
struct fn_l18_pf_args {
    const char * p[GGML_CUDA_FN_L18_PF_MAX];
    unsigned int nc[GGML_CUDA_FN_L18_PF_MAX];   // chunks of range r
    unsigned int last[GGML_CUDA_FN_L18_PF_MAX]; // bytes of its last chunk (a multiple of 16, > 0)
    int          nr;
    unsigned int n_chunks;
    int          n_copy; // the blocks of k_mb_wait; the blocks after them load
};

// [TAG_FN_L18_GPUFLOOR_PFWAIT] k_mb_wait with loader blocks: blocks [0, n_copy) are k_mb_wait's (block 0 also writes the
// ticket to the channel's word l4sync[3] when its poll ends); the blocks after them read the prefetch ranges with real loads
// (so the lines are in L2), chunk by chunk, until that word holds the ticket or all chunks are read
static __global__ void k_mb_wait_pf(const mb_dev v, const int chan, const int32_t * __restrict__ ticket,
        float * __restrict__ dst, const int n, const fn_l18_pf_args a) {
    if ((int) blockIdx.x >= a.n_copy) {
        const uint32_t want = (uint32_t) ticket[0];
        const volatile uint32_t * flag = v.l4sync + 4*chan + 3;
        const unsigned int nb = gridDim.x - a.n_copy;
        uint32_t acc = 0;
        for (unsigned int c = blockIdx.x - a.n_copy; c < a.n_chunks; c += nb) {
            if (*flag == want) {
                break;
            }
            int          r  = 0;
            unsigned int c0 = 0;
            while (r < a.nr - 1 && c >= c0 + a.nc[r]) {
                c0 += a.nc[r];
                ++r;
            }
            const unsigned int k  = c - c0;
            const char *       pc = a.p[r] + (size_t) k*FN_L18_PF_CHUNK;
            const unsigned int sz = k + 1 == a.nc[r] ? a.last[r] : FN_L18_PF_CHUNK;
            uint4 w[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const unsigned int o = (u*MB_WAIT_THREADS + threadIdx.x)*16;
                w[u] = o < sz ? __ldcg((const uint4 *) (pc + o)) : make_uint4(0, 0, 0, 0);
            }
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                acc ^= w[u].x ^ w[u].y ^ w[u].z ^ w[u].w;
            }
        }
        if (acc == 0x9E3779B9u && threadIdx.x == 1023u) {
            ((volatile uint32_t *) v.l4sync)[4*chan + 3] = acc; // never (256 threads): keeps the loads
        }
        return;
    }
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
            const volatile uint32_t * rel = &v.glob->release; // [TAG_FN_R4_BRIDGE_DMA]
            for (;;) {
                if ((int32_t) (*done - want) >= 0) {
                    ok = 1;
                    break;
                }
                if (*err != 0 || *rel != 0) {
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
            const unsigned long long dt = mb_now_ns() - t0;
            if (v.stats && blockIdx.x == 0) {
                volatile mb_glob * g = v.glob;
                g->waits       = g->waits + 1;
                g->waits_ready = g->waits_ready + (ready && ok ? 1 : 0);
                g->wait_ns     = g->wait_ns + dt;
            }
            if (blockIdx.x == 0) { // [TAG_FN_R4_BRIDGE_DMA] the host's DMA/CPU split reads this
                volatile mb_chan_hdr * h = v.hdr + chan;
                h->wait_ns   = ready ? 0ull : dt;
                if (v.probe) { // [TAG_FN_L14_PROBE]
                    h->t_wstart = t0;
                    h->t_wend   = t0 + dt;
                }
                h->times_seq = want;
            }
        }
        // acquire: the host wrote out before done. The fence is in the thread that read done and comes before the
        // barrier that hands the result to the other threads, so their reads of out are ordered after the host's writes.
        __threadfence_system();
        s_ok = ok;
        if (blockIdx.x == 0) {
            *(volatile uint32_t *) &v.l4sync[4*chan + 3] = (uint32_t) ticket[0]; // the loader blocks stop
        }
    }
    __syncthreads();

    const float * out    = (const float *) (v.data + (size_t) chan*v.chan_bytes + v.off_out);
    const int     stride = a.n_copy*blockDim.x; // the copy blocks only (the loader blocks returned)
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

// [TAG_FN_R4_BRIDGE_DMA] thread 0: wait (bounded, as k_mb_wait) for the plan of the ticket's job and check it; then the
// block copies it into the device scratch and the slot ids into dst. A plan that never comes, or does not fit the
// ring, the banks or the budget, sets the error and makes every slot id the zero slot (nothing is copied).
static __global__ void k_mb_fetch_plan(const mb_dev v, const int chan, const int32_t * __restrict__ ticket,
        int32_t * __restrict__ dst, const int n_ids, const int zero_slot, const unsigned long long expert_bytes) {
    __shared__ int s_ok;
    __shared__ int s_n;
    mb_fetch_dev * f = v.fetch + chan;
    const ggml_moe_bridge_plan * pl = (const ggml_moe_bridge_plan *) (v.data + (size_t) chan*v.chan_bytes + v.off_plan);
    if (threadIdx.x == 0) {
        const uint32_t want = (uint32_t) ticket[0];
        int ok = 0;
        int n  = 0;
        if (want != 0) {
            const volatile uint32_t * plan  = &v.hdr[chan].plan;
            const volatile uint32_t * done  = &v.hdr[chan].done;
            const volatile uint32_t * taken = &v.hdr[chan].taken;
            const volatile uint32_t * err   = &v.state[1];
            const volatile uint32_t * rel   = &v.glob->release;
            const unsigned long long t0 = mb_now_ns();
            for (;;) {
                if ((int32_t) (*plan - want) >= 0) {
                    ok = 1;
                    break;
                }
                if ((int32_t) (*done - want) >= 0) {
                    // the host answered the job: a plan it published before done is visible after this fence; none
                    // means a failed job (its error is the host's, raised by complete): nothing to fetch
                    __threadfence_system();
                    ok = (int32_t) (*plan - want) >= 0 ? 1 : 0;
                    break;
                }
                if (*err != 0 || *rel != 0) {
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
                mb_sleep();
            }
            __threadfence_system(); // acquire: the host wrote the plan before plan
            if (ok) {
                n = __ldcv(&pl->n_copy);
                bool good = n >= 0 && n <= v.max_fetch && v.dma_ring != nullptr;
                for (int i = 0; good && i < n; ++i) {
                    const unsigned long long off  = __ldcv((const unsigned long long *) &pl->off[i]);
                    const int32_t            slot = __ldcv(&pl->slot[i]);
                    good = expert_bytes <= v.dma_ring_size && off <= v.dma_ring_size - expert_bytes &&
                           slot >= 0 && slot < zero_slot;
                    f->off[i]  = off;
                    f->slot[i] = slot;
                }
                if (!good) {
                    ok = 0;
                    n  = 0;
                    if (atomicCAS(&v.state[1], 0u, (unsigned int) GGML_MOE_BRIDGE_ERR_RUNNER) == 0u) {
                        *(volatile int32_t  *) &v.glob->err_chan = chan;
                        *(volatile uint32_t *) &v.glob->err_seq  = want;
                        __threadfence_system();
                        *(volatile uint32_t *) &v.glob->err = GGML_MOE_BRIDGE_ERR_RUNNER;
                        __threadfence_system();
                    }
                }
            }
        }
        f->n_copy      = n;
        f->done_blocks = 0;
        f->seq         = want;
        f->t0          = mb_now_ns();
        s_ok = ok;
        s_n  = n;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n_ids; i += blockDim.x) {
        int32_t sl = zero_slot;
        if (s_ok) {
            sl = __ldcv(&pl->slot_ids[i]);
            sl = sl >= 0 && sl <= zero_slot ? sl : zero_slot;
        }
        dst[i] = sl;
    }
    GGML_UNUSED(s_n);
}

// [TAG_FN_R4_BRIDGE_DMA] blockIdx.z: the plan's expert, blockIdx.y: its part (up, gate, down); the blocks of x stride
// over the part's 16-byte words. The ring holds each expert as up | gate | down, packed. The last block to finish
// writes the fetch time and count into the mapped header.
static __global__ void k_mb_fetch_copy(const mb_dev v, const int chan,
        char * __restrict__ d0, const long long nb0, const long long sz0,
        char * __restrict__ d1, const long long nb1, const long long sz1,
        char * __restrict__ d2, const long long nb2, const long long sz2) {
    mb_fetch_dev * f = v.fetch + chan;
    const int e = blockIdx.z;
    const int n = f->n_copy;
    if (e < n) {
        const int part = blockIdx.y;
        const long long sz  = part == 0 ? sz0 : part == 1 ? sz1 : sz2;
        const long long po  = part == 0 ? 0   : part == 1 ? sz0 : sz0 + sz1;
        const char * src = v.dma_ring + f->off[e] + po;
        char       * dst = (part == 0 ? d0 + f->slot[e]*nb0 : part == 1 ? d1 + f->slot[e]*nb1 : d2 + f->slot[e]*nb2);
        const long long stride = (long long) gridDim.x*blockDim.x;
        const long long i0     = (long long) blockIdx.x*blockDim.x + threadIdx.x;
        if ((((uintptr_t) src | (uintptr_t) dst | (uintptr_t) sz) & 15) == 0) {
            const uint4 * s4 = (const uint4 *) src;
            uint4       * t4 = (uint4 *) dst;
            for (long long i = i0; i < sz/16; i += stride) {
                t4[i] = __ldcv(s4 + i);
            }
        } else {
            for (long long i = i0; i < sz; i += stride) {
                dst[i] = __ldcv(src + i);
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        const unsigned int total = gridDim.x*gridDim.y*gridDim.z;
        if (atomicAdd(&f->done_blocks, 1u) == total - 1u) {
            volatile mb_chan_hdr * h = v.hdr + chan;
            h->fetch_ns = n > 0 ? mb_now_ns() - f->t0 : 0ull;
            h->n_fetch  = (uint32_t) n;
            __threadfence_system();
        }
    }
}

// [TAG_FN_L3_CPU_DEVPRED] one block: the predicted ids [n_tokens][k] of the job of ticket into the channel's hint area
static __global__ void k_mb_hint(const mb_dev v, const int chan, const int32_t * __restrict__ ticket,
        const char * __restrict__ ids, const int64_t ids_nb1, const int k, const int n_tokens) {
    __shared__ uint32_t s_seq;
    mb_hint_hdr * h    = (mb_hint_hdr *) (v.data + (size_t) chan*v.chan_bytes + v.off_hint);
    int32_t     * hids = (int32_t *) (h + 1);
    if (threadIdx.x == 0) {
        s_seq = (uint32_t) ticket[0]; // 0: the post was skipped (sticky error)
        *(volatile uint32_t *) &h->seq = 0;
        __threadfence_system();
    }
    __syncthreads();
    if (s_seq == 0) {
        return;
    }
    for (int i = threadIdx.x; i < k*n_tokens; i += blockDim.x) {
        const int t = i/k;
        hids[i] = ((const int32_t *) (ids + t*ids_nb1))[i - t*k];
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        *(volatile int32_t *) &h->k        = k;
        *(volatile int32_t *) &h->n_tokens = n_tokens;
        __threadfence_system();
        *(volatile uint32_t *) &h->seq = s_seq;
        __threadfence_system();
    }
}

// [TAG_FN_L9_HINTTOPK] the hint from the next layer's router logits in one launch: one warp per token picks the top k
// (k argmax rounds over the row in registers, ties to the lower id, NaN as -inf), then the seqlock publish of k_mb_hint.
// Replaces ARGSORT (a 512-wide bitonic sort, ~13.6 us) + k_mb_hint. Only the prefetch reads the ids: no value changes.
template <int EPT>
static __global__ void k_fn_l9_hint_topk(const mb_dev v, const int chan, const int32_t * __restrict__ ticket,
        const char * __restrict__ logits, const int64_t nb1, const int n_exp, const int k, const int n_tokens) {
    __shared__ uint32_t s_seq;
    mb_hint_hdr * h    = (mb_hint_hdr *) (v.data + (size_t) chan*v.chan_bytes + v.off_hint);
    int32_t     * hids = (int32_t *) (h + 1);
    if (threadIdx.x == 0) {
        s_seq = (uint32_t) ticket[0]; // 0: the post was skipped (sticky error)
        *(volatile uint32_t *) &h->seq = 0;
        __threadfence_system();
    }
    __syncthreads();
    if (s_seq == 0) {
        return;
    }
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    for (int t = warp; t < n_tokens; t += blockDim.x / WARP_SIZE) {
        const float * row = (const float *) (logits + t*nb1);
        float wt[EPT];
#pragma unroll
        for (int i = 0; i < EPT; ++i) {
            const int e = lane + i*WARP_SIZE;
            const float x = e < n_exp ? row[e] : -INFINITY;
            wt[i] = x != x ? -INFINITY : x;
        }
        for (int r = 0; r < k; ++r) {
            float max_val    = wt[0];
            int   max_expert = lane;
#pragma unroll
            for (int i = 1; i < EPT; ++i) {
                const int e = lane + i*WARP_SIZE;
                if (wt[i] > max_val) {
                    max_val    = wt[i];
                    max_expert = e;
                }
            }
#pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
                const float val    = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
                const int   expert = __shfl_xor_sync(0xFFFFFFFF, max_expert, mask, WARP_SIZE);
                if (val > max_val || (val == max_val && expert < max_expert)) {
                    max_val    = val;
                    max_expert = expert;
                }
            }
            if ((max_expert & (WARP_SIZE - 1)) == lane) {
                wt[max_expert / WARP_SIZE] = -INFINITY;
            }
            if (lane == 0) {
                hids[t*k + r] = max_expert < n_exp ? max_expert : 0;
            }
        }
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        *(volatile int32_t *) &h->k        = k;
        *(volatile int32_t *) &h->n_tokens = n_tokens;
        __threadfence_system();
        *(volatile uint32_t *) &h->seq = s_seq;
        __threadfence_system();
    }
}

// ---- [TAG_FN_L4_POST] the post of a node marked GGML_FN_L4_POST (qwen4exp only) ------------------------------------
// The host reads the payload, the header and the ring entry only after it sees the stamp. Here every thread writes its
// part of all of them, then makes one system fence, then the block syncs and one thread writes the stamp. k_mb_post
// made four fences in a row on thread 0 (payload, header, ring entry, stamp). Same bytes, same order for the host.

constexpr int FN_L4_POST_THREADS = 512;

// the thread that takes the sequence number and writes the header and the stamp: the last one of the block
static __device__ __forceinline__ int fn_l4_ctl() {
    return blockDim.x - 1;
}

// thread ctl: the channel's next sequence number (0: sticky error, nothing is posted) and, in spin mode, the ring stamp
static __device__ __forceinline__ void fn_l4_take_seq(const mb_dev v, const int chan, uint32_t & seq, uint32_t & g) {
    seq = 0;
    g   = 0;
    if (*(volatile uint32_t *) &v.state[1] == 0) {
        uint32_t s = v.state[2 + chan] + 1;
        if (s == 0) {
            s = 1;
        }
        v.state[2 + chan] = s;
        seq = s;
        if (v.mode == GGML_MOE_BRIDGE_WAIT_SPIN) {
            g = atomicAdd(&v.state[0], 1u) + 1u;
            if (g == 0u) { // stamp 0 is never a post (mb_next_stamp)
                g = atomicAdd(&v.state[0], 1u) + 1u;
            }
        }
    }
}

// thread ctl, before the fence: the header and the ring entry, all but the stamp
static __device__ __forceinline__ void fn_l4_post_head(const mb_dev v, const int chan, const uint32_t seq, const uint32_t g,
        const int n_tokens, const int n_used, const int flags) {
    mb_chan_hdr * h = v.hdr + chan;
    *(volatile int32_t  *) &h->n_tokens = n_tokens;
    *(volatile int32_t  *) &h->n_used   = n_used;
    *(volatile int32_t  *) &h->flags    = flags;
    if (v.probe) {
        *(volatile unsigned long long *) &h->t_post = mb_now_ns(); // [TAG_FN_L14_PROBE]
    }
    *(volatile uint32_t *) &h->seq      = seq;
    if (v.mode == GGML_MOE_BRIDGE_WAIT_SPIN) {
        mb_ring_entry * e = v.ring + (g % MB_RING);
        *(volatile uint32_t *) &e->chan = (uint32_t) chan;
        *(volatile uint32_t *) &e->seq  = seq;
    }
}

// thread ctl, after every thread's fence and the barrier: the stamp, then the ticket (0 when nothing was posted)
static __device__ __forceinline__ void fn_l4_post_publish(const mb_dev v, const int chan, const uint32_t seq, const uint32_t g, int32_t * ticket) {
    if (seq != 0 && v.mode == GGML_MOE_BRIDGE_WAIT_SPIN) {
        if (v.probe) {
            *(volatile unsigned long long *) &v.hdr[chan].t_pub = mb_now_ns(); // [TAG_FN_L14_PROBE]
        }
        mb_ring_entry * e = v.ring + (g % MB_RING);
        *(volatile uint32_t *) &e->stamp       = g;
        *(volatile uint32_t *) &v.glob->posted = g;
        __threadfence_system();
    }
    ticket[0] = (int32_t) seq;
}

// threads [t, t + nt) copy x (n_tokens rows of n_embd floats, row stride x_nb1) packed into hx, as k_mb_post lays it out;
// up to 8 loads per thread are in flight before the stores
static __device__ __forceinline__ void fn_l4_copy_x(float * hx, const char * x, const int64_t x_nb1, const int n_embd,
        const int n_tokens, const int t, const int nt) {
    if ((n_embd & 3) == 0 && (x_nb1 & 15) == 0 && ((uintptr_t) x & 15) == 0) {
        constexpr int U = 8;
        const int n4 = n_embd/4;
        const int n  = n4*n_tokens;
        for (int base = t; base < n; base += U*nt) {
            float4 r[U];
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int i = base + u*nt;
                if (i < n) {
                    const int tk = i/n4;
                    r[u] = ((const float4 *) (x + tk*x_nb1))[i - tk*n4];
                }
            }
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int i = base + u*nt;
                if (i < n) {
                    ((float4 *) hx)[i] = r[u];
                }
            }
        }
    } else {
        for (int i = t; i < n_embd*n_tokens; i += nt) {
            const int tk = i/n_embd;
            hx[i] = ((const float *) (x + tk*x_nb1))[i - tk*n_embd];
        }
    }
}

// the post alone (no fused top-k in front of it): k_mb_post's arguments and result
static __global__ void __launch_bounds__(FN_L4_POST_THREADS) k_fn_l4_mb_post(const mb_dev v, const int chan,
        const char * x, const int64_t x_nb1, const char * ids, const int64_t ids_nb1, const float * w,
        const int n_embd, const int n_used, const int n_tokens, const int flags, int32_t * ticket) {
    __shared__ uint32_t s_seq;
    __shared__ uint32_t s_g;
    const int tid = threadIdx.x;
    const int ctl = fn_l4_ctl();
    if (tid == ctl) {
        uint32_t seq;
        uint32_t g;
        fn_l4_take_seq(v, chan, seq, g);
        s_seq = seq;
        s_g   = g;
    }
    __syncthreads();
    const uint32_t seq = s_seq;
    if (seq == 0) {
        if (tid == ctl) {
            ticket[0] = 0;
        }
        return;
    }

    char    * p    = v.data + (size_t) chan*v.chan_bytes;
    float   * hx   = (float   *) p;
    int32_t * hids = (int32_t *) (p + v.off_ids);
    float   * hw   = (float   *) (p + v.off_w);

    fn_l4_copy_x(hx, x, x_nb1, n_embd, n_tokens, tid, blockDim.x);
    for (int i = tid; i < n_used*n_tokens; i += blockDim.x) {
        const int t = i/n_used;
        hids[i] = ((const int32_t *) (ids + t*ids_nb1))[i - t*n_used];
        hw[i]   = w[i];
    }
    if (tid == ctl) {
        fn_l4_post_head(v, chan, seq, s_g, n_tokens, n_used, flags);
    }

    __threadfence_system();
    __syncthreads();

    if (tid == ctl) {
        fn_l4_post_publish(v, chan, seq, s_g, ticket);
    }
}

// [TAG_FN_L4_POST] topk_moe_cuda (topk-moe.cu, no bias) and the post in one launch. Block 0: warp r < n_rows computes row r
// with the code of topk_moe_cuda (the same ops in the same order, so the same ids and weights) and writes ids and weights
// to the device tensors and to the channel; its last thread takes the sequence number and writes the header. Blocks 1..n
// copy x to the channel meanwhile, one part each (PCIe writes from many SMs, not one). Every block fences its writes and
// counts itself done; the last block writes the stamp and the ticket. Same bytes, same order for the host.
struct fn_l4_topk_config {
    bool use_sigmoid;
    bool use_sqrt_softplus;
    bool with_norm;
};

constexpr int FN_L4_TOPK_THREADS = TOPK_MOE_ROWS_PER_BLOCK * WARP_SIZE; // block 0: one warp per row
constexpr int FN_L4_XBLK_MAX     = 16;                                  // x blocks, at most

template <int experts_per_thread, bool use_limit>
static __device__ __forceinline__ void fn_l4_softmax_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
    float max_val = -INFINITY;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            max_val = max(max_val, vals[i]);
        }
    }

    max_val = warp_reduce_max(max_val);

    float sum = 0.f;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            const float val = expf(vals[i] - max_val);
            vals[i]         = val;
            sum += val;
        } else {
            vals[i] = 0.f;
        }
    }

    sum = warp_reduce_sum(sum);

    const float inv_sum = 1.0f / sum;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            vals[i] *= inv_sum;
        }
    }
}

template <int experts_per_thread, bool use_limit>
static __device__ __forceinline__ void fn_l4_sigmoid_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        vals[i]           = active ? 1.f / (1.f + expf(-vals[i])) : -INFINITY;
    }
}

template <int experts_per_thread, bool use_limit>
static __device__ __forceinline__ void fn_l4_sqrt_softplus_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        vals[i]           = active ? sqrtf(vals[i] > 20.0f ? vals[i] : logf(1.0f + expf(vals[i]))) : -INFINITY;
    }
}

// x block `part` of n_part: its share of x (n_tokens rows of n_embd floats, row stride x_nb1), packed into hx as k_mb_post
static __device__ __forceinline__ void fn_l4_copy_x_part(float * hx, const char * x, const int64_t x_nb1, const int n_embd,
        const int n_tokens, const int part, const int n_part) {
    if ((n_embd & 3) == 0 && (x_nb1 & 15) == 0 && ((uintptr_t) x & 15) == 0) {
        const int n4  = n_embd/4;
        const int n   = n4*n_tokens;
        const int per = (n + n_part - 1)/n_part;
        const int end = min(n, (part + 1)*per);
        for (int i = part*per + threadIdx.x; i < end; i += blockDim.x) {
            const int tk = i/n4;
            ((float4 *) hx)[i] = ((const float4 *) (x + tk*x_nb1))[i - tk*n4];
        }
    } else {
        const int n   = n_embd*n_tokens;
        const int per = (n + n_part - 1)/n_part;
        const int end = min(n, (part + 1)*per);
        for (int i = part*per + threadIdx.x; i < end; i += blockDim.x) {
            const int tk = i/n_embd;
            hx[i] = ((const float *) (x + tk*x_nb1))[i - tk*n_embd];
        }
    }
}

template <int n_experts>
static __global__ void __launch_bounds__(FN_L4_TOPK_THREADS, 1) k_fn_l4_topk_post(const mb_dev v, const int chan,
        const float * logits, float * weights, int32_t * ids, const int n_rows, const int n_expert_used,
        const float clamp_val, const float scale_val, const fn_l4_topk_config config,
        const char * x, const int64_t x_nb1, const int n_embd, const int flags, int32_t * ticket) {
    uint32_t * sy = v.l4sync + 4*chan; // [0] blocks done, [1] seq, [2] stamp

    char    * p    = v.data + (size_t) chan*v.chan_bytes;
    float   * hx   = (float   *) p;
    int32_t * hids = (int32_t *) (p + v.off_ids);
    float   * hw   = (float   *) (p + v.off_w);

    ggml_cuda_pdl_sync();

    if (blockIdx.x > 0) {
        ggml_cuda_pdl_lc();
        // also with a sticky error: then nothing is posted and the host does not read it
        fn_l4_copy_x_part(hx, x, x_nb1, n_embd, n_rows, blockIdx.x - 1, gridDim.x - 1);
    } else {
        __shared__ uint32_t s_seq;
        __shared__ uint32_t s_g;
        const int  lane   = threadIdx.x % WARP_SIZE;
        const int  row    = threadIdx.x / WARP_SIZE;
        const bool is_row = row < n_rows;
        const int  ctl    = fn_l4_ctl();

        constexpr int experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

        float wt[experts_per_thread];

#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            wt[i] = -INFINITY;
        }

        if (is_row) {
            const float * lg = logits + n_experts * row;
#pragma unroll
            for (int i = 0; i < n_experts; i += WARP_SIZE) {
                const int expert  = i + lane;
                wt[i / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? lg[expert] : -INFINITY;
            }
        }
        if (threadIdx.x == ctl) {
            uint32_t seq;
            uint32_t g;
            fn_l4_take_seq(v, chan, seq, g);
            s_seq = seq;
            s_g   = g;
        }

        // weights and ids can alias logits: every row reads its logits before any thread writes
        __syncthreads();
        ggml_cuda_pdl_lc();

        const uint32_t seq = s_seq;

        if (is_row) {
            float   * wrow  = weights + n_expert_used * row;
            int32_t * irow  = ids + n_experts * row;
            int32_t * hirow = hids + n_expert_used * row;
            float   * hwrow = hw + n_expert_used * row;

            if (config.use_sigmoid) {
                fn_l4_sigmoid_warp_inplace<experts_per_thread, false>(wt, n_experts, lane);
            } else if (config.use_sqrt_softplus) {
                fn_l4_sqrt_softplus_warp_inplace<experts_per_thread, false>(wt, n_experts, lane);
            } else {
                fn_l4_softmax_warp_inplace<experts_per_thread, false>(wt, n_experts, lane);
            }

            // NaN -> -FLT_MAX, as topk_moe_cuda: the iterative argmax then picks unique ids
#pragma unroll
            for (int i = 0; i < experts_per_thread; i++) {
                if (__isnanf(wt[i])) {
                    wt[i] = -FLT_MAX;
                }
            }

            float wt_sum = 0.f;

            float output_weights[experts_per_thread];

#pragma unroll
            for (int i = 0; i < experts_per_thread; i++) {
                output_weights[i] = 0.f;
            }

            for (int k = 0; k < n_expert_used; k++) {
                float max_val    = wt[0];
                int   max_expert = lane;

#pragma unroll
                for (int i = 1; i < experts_per_thread; i++) {
                    const int expert = lane + i * WARP_SIZE;
                    if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && wt[i] > max_val) {
                        max_val    = wt[i];
                        max_expert = expert;
                    }
                }

#pragma unroll
                for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
                    const float val    = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
                    const int   expert = __shfl_xor_sync(0xFFFFFFFF, max_expert, mask, WARP_SIZE);
                    if (val > max_val || (val == max_val && expert < max_expert)) {
                        max_val    = val;
                        max_expert = expert;
                    }
                }

                if ((max_expert & (WARP_SIZE - 1)) == lane) {
                    wt[max_expert / WARP_SIZE] = -INFINITY;
                }

                if ((k & (WARP_SIZE - 1)) == lane) {
                    output_weights[k / WARP_SIZE] = max_val;
                }

                if ((max_expert & (WARP_SIZE - 1)) == lane) {
                    irow[k] = max_expert;
                    if (seq != 0) {
                        hirow[k] = max_expert;
                    }
                    if (config.with_norm) {
                        wt_sum += max_val;
                    }
                }
            }

            if (config.with_norm) {
                wt_sum              = warp_reduce_sum(wt_sum);
                wt_sum              = max(wt_sum, clamp_val);
                const float inv_sum = 1.0f / wt_sum;

#pragma unroll
                for (int i = 0; i < experts_per_thread; i++) {
                    output_weights[i] *= inv_sum;
                }
            }

#pragma unroll
            for (int i = 0; i < experts_per_thread; i++) {
                const int idx = i * WARP_SIZE + lane;
                if (idx < n_expert_used) {
                    const float wv = output_weights[i] * scale_val;
                    wrow[idx] = wv;
                    if (seq != 0) {
                        hwrow[idx] = wv;
                    }
                }
            }
        }

        if (threadIdx.x == ctl) {
            if (seq != 0) {
                fn_l4_post_head(v, chan, seq, s_g, n_rows, n_expert_used, flags);
            }
            *(volatile uint32_t *) &sy[1] = seq;
            *(volatile uint32_t *) &sy[2] = s_g;
        }
    }

    // each block: its writes, then it counts itself done; the last one publishes
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint32_t n_done = atomicAdd(&sy[0], 1u) + 1u;
        if (n_done == gridDim.x) {
            __threadfence(); // acquire: the writes of every block and the seq and stamp of block 0 come before its count
            const uint32_t seq = *(volatile uint32_t *) &sy[1];
            const uint32_t g   = *(volatile uint32_t *) &sy[2];
            *(volatile uint32_t *) &sy[0] = 0;
            fn_l4_post_publish(v, chan, seq, g, ticket);
        }
    }
}

// ---- [TAG_FN_L22_X_PRIOR] the route prior and the resident swap (ggml-fn-l22-prior.h; qwen4exp, opt-in) ------------------
// One warp routes one row of 512 experts with the code of k_fn_l4_topk_post (the same softmax, NaN rule, argmax tie rule:
// the larger value, then the lower expert; the same sum order), on sel = p * scale instead of p. A row whose scale is 1
// everywhere (no hot expert, or lambda 0) picks the stock ids and weights bit for bit. The resident swap: for ranks
// r0..k-1 in order, a cold pick is replaced by the most probable hot expert not yet picked when p_hot >= p_cold * expnm.
// On return lane k < n_used holds entry k: its expert, its final weight and (raw) entry k of the unbiased top-k.
template <int EPT>
static __device__ __forceinline__ void fn_l22_route_row(float (&wt)[EPT], const float * scale, const float * mask,
        const int n_used, const int pflags, const float expnm, const int r0, const bool with_norm, const float clamp_val,
        const float scale_val, const int lane, int & o_id, float & o_w, int & o_raw) {
    fn_l4_softmax_warp_inplace<EPT, false>(wt, EPT*WARP_SIZE, lane);
#pragma unroll
    for (int i = 0; i < EPT; i++) {
        if (__isnanf(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    float    sel[EPT];
    uint32_t hot = 0;
#pragma unroll
    for (int i = 0; i < EPT; i++) {
        const int   e = lane + i*WARP_SIZE;
        const float s = wt[i]*scale[e];
        sel[i] = s < -FLT_MAX ? -FLT_MAX : s;
        hot |= (mask[e] != 0.0f ? 1u : 0u) << i;
    }

    // the unbiased top-k, for the hot-set observer
    o_raw = -1;
    if (pflags & GGML_FN_L22_PRIOR_F_RAW) {
        float rv[EPT];
#pragma unroll
        for (int i = 0; i < EPT; i++) {
            rv[i] = wt[i];
        }
        for (int k = 0; k < n_used; k++) {
            float mv = rv[0];
            int   me = lane;
#pragma unroll
            for (int i = 1; i < EPT; i++) {
                if (rv[i] > mv) {
                    mv = rv[i];
                    me = lane + i*WARP_SIZE;
                }
            }
#pragma unroll
            for (int m = WARP_SIZE/2; m > 0; m /= 2) {
                const float vv = __shfl_xor_sync(0xFFFFFFFF, mv, m, WARP_SIZE);
                const int   ee = __shfl_xor_sync(0xFFFFFFFF, me, m, WARP_SIZE);
                if (vv > mv || (vv == mv && ee < me)) {
                    mv = vv;
                    me = ee;
                }
            }
            if ((me & (WARP_SIZE - 1)) == lane) {
#pragma unroll
                for (int i = 0; i < EPT; i++) {
                    if (i == me/WARP_SIZE) {
                        rv[i] = -INFINITY;
                    }
                }
            }
            if ((k & (WARP_SIZE - 1)) == lane) {
                o_raw = me;
            }
        }
    }

    // the top-k on sel, carrying the unscaled p of each pick
    int      my_id  = 0;
    float    my_p   = 0.0f;
    uint32_t picked = 0;
    for (int k = 0; k < n_used; k++) {
        float mv = sel[0];
        int   me = lane;
        float mp = wt[0];
#pragma unroll
        for (int i = 1; i < EPT; i++) {
            if (sel[i] > mv) {
                mv = sel[i];
                me = lane + i*WARP_SIZE;
                mp = wt[i];
            }
        }
#pragma unroll
        for (int m = WARP_SIZE/2; m > 0; m /= 2) {
            const float vv = __shfl_xor_sync(0xFFFFFFFF, mv, m, WARP_SIZE);
            const int   ee = __shfl_xor_sync(0xFFFFFFFF, me, m, WARP_SIZE);
            const float pp = __shfl_xor_sync(0xFFFFFFFF, mp, m, WARP_SIZE);
            if (vv > mv || (vv == mv && ee < me)) {
                mv = vv;
                me = ee;
                mp = pp;
            }
        }
        if ((me & (WARP_SIZE - 1)) == lane) {
#pragma unroll
            for (int i = 0; i < EPT; i++) {
                if (i == me/WARP_SIZE) {
                    sel[i]  = -INFINITY;
                    picked |= 1u << i;
                }
            }
        }
        if ((k & (WARP_SIZE - 1)) == lane) {
            my_id = me;
            my_p  = mp;
        }
    }

    // the resident swap (sequential, rank r0 first)
    if (pflags & GGML_FN_L22_PRIOR_F_RESIDENT) {
        for (int r = r0 > 0 ? r0 : 0; r < n_used; r++) {
            const int   id_r = __shfl_sync(0xFFFFFFFF, my_id, r, WARP_SIZE);
            const float p_r  = __shfl_sync(0xFFFFFFFF, my_p,  r, WARP_SIZE);
            if (mask[id_r] != 0.0f) {
                continue; // a hot pick stays (warp-uniform)
            }
            float bv = -INFINITY;
            int   be = 0x7fffffff;
            const uint32_t cand = hot & ~picked;
#pragma unroll
            for (int i = 0; i < EPT; i++) {
                if (((cand >> i) & 1u) && wt[i] > bv) {
                    bv = wt[i];
                    be = lane + i*WARP_SIZE;
                }
            }
#pragma unroll
            for (int m = WARP_SIZE/2; m > 0; m /= 2) {
                const float vv = __shfl_xor_sync(0xFFFFFFFF, bv, m, WARP_SIZE);
                const int   ee = __shfl_xor_sync(0xFFFFFFFF, be, m, WARP_SIZE);
                if (vv > bv || (vv == bv && ee < be)) {
                    bv = vv;
                    be = ee;
                }
            }
            if (be != 0x7fffffff && bv >= p_r*expnm) {
                if ((be & (WARP_SIZE - 1)) == lane) {
#pragma unroll
                    for (int i = 0; i < EPT; i++) {
                        if (i == be/WARP_SIZE) {
                            picked |= 1u << i;
                        }
                    }
                }
                if (lane == r) {
                    my_id = be;
                    my_p  = bv;
                }
            }
        }
    }

    // the weights: lane (id & 31) sums its picks in rank order, then the warp sum (k_fn_l4_topk_post's order)
    float inv = 1.0f;
    if (with_norm) {
        float part = 0.0f;
        for (int k = 0; k < n_used; k++) {
            const int   id = __shfl_sync(0xFFFFFFFF, my_id, k, WARP_SIZE);
            const float pk = __shfl_sync(0xFFFFFFFF, my_p,  k, WARP_SIZE);
            if ((id & (WARP_SIZE - 1)) == lane) {
                part += pk;
            }
        }
        part = warp_reduce_sum(part);
        part = max(part, clamp_val);
        inv  = 1.0f / part;
    }
    o_id = my_id;
    o_w  = (with_norm ? my_p*inv : my_p) * scale_val;
}

// [TAG_FN_L22_X_PRIOR] k_fn_l4_topk_post with the prior's routing in block 0: the same x blocks, sequence number, header,
// fences and publish; the header's flags carry GGML_MOE_BRIDGE_JOB_PRIOR (and _RAW when the unbiased ids are written)
template <int n_experts>
static __global__ void __launch_bounds__(FN_L4_TOPK_THREADS, 1) k_fn_l22_topk_post_prior(const mb_dev v, const int chan,
        const float * logits, float * weights, int32_t * ids, const int n_rows, const int n_expert_used,
        const float clamp_val, const float scale_val, const bool with_norm, const float * scale_row, const float * mask_row,
        const int pflags, const float expnm, const int r0,
        const char * x, const int64_t x_nb1, const int n_embd, const int flags, int32_t * ticket) {
    uint32_t * sy = v.l4sync + 4*chan; // [0] blocks done, [1] seq, [2] stamp

    char    * p    = v.data + (size_t) chan*v.chan_bytes;
    float   * hx   = (float   *) p;
    int32_t * hids = (int32_t *) (p + v.off_ids);
    float   * hw   = (float   *) (p + v.off_w);
    int32_t * hraw = (pflags & GGML_FN_L22_PRIOR_F_RAW) && v.off_raw ? (int32_t *) (p + v.off_raw) : nullptr;

    ggml_cuda_pdl_sync();

    if (blockIdx.x > 0) {
        ggml_cuda_pdl_lc();
        fn_l4_copy_x_part(hx, x, x_nb1, n_embd, n_rows, blockIdx.x - 1, gridDim.x - 1);
    } else {
        __shared__ uint32_t s_seq;
        __shared__ uint32_t s_g;
        const int  lane   = threadIdx.x % WARP_SIZE;
        const int  row    = threadIdx.x / WARP_SIZE;
        const bool is_row = row < n_rows;
        const int  ctl    = fn_l4_ctl();

        constexpr int EPT = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

        float wt[EPT];
#pragma unroll
        for (int i = 0; i < EPT; i++) {
            wt[i] = -INFINITY;
        }
        if (is_row) {
            const float * lg = logits + n_experts * row;
#pragma unroll
            for (int i = 0; i < n_experts; i += WARP_SIZE) {
                wt[i / WARP_SIZE] = lg[i + lane];
            }
        }
        if (threadIdx.x == ctl) {
            uint32_t seq;
            uint32_t g;
            fn_l4_take_seq(v, chan, seq, g);
            s_seq = seq;
            s_g   = g;
        }

        // weights and ids can alias logits: every row reads its logits before any thread writes
        __syncthreads();
        ggml_cuda_pdl_lc();

        const uint32_t seq = s_seq;

        if (is_row) {
            int   o_id  = 0;
            float o_w   = 0.0f;
            int   o_raw = -1;
            fn_l22_route_row<EPT>(wt, scale_row, mask_row, n_expert_used, pflags, expnm, r0, with_norm, clamp_val, scale_val,
                    lane, o_id, o_w, o_raw);
            if (lane < n_expert_used) {
                ids[n_experts*row + lane]         = o_id;
                weights[n_expert_used*row + lane] = o_w;
                if (seq != 0) {
                    hids[n_expert_used*row + lane] = o_id;
                    hw[n_expert_used*row + lane]   = o_w;
                    if (hraw) {
                        hraw[n_expert_used*row + lane] = o_raw;
                    }
                }
            }
        }

        if (threadIdx.x == ctl) {
            if (seq != 0) {
                fn_l4_post_head(v, chan, seq, s_g, n_rows, n_expert_used, flags);
            }
            *(volatile uint32_t *) &sy[1] = seq;
            *(volatile uint32_t *) &sy[2] = s_g;
        }
    }

    // each block: its writes, then it counts itself done; the last one publishes
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint32_t n_done = atomicAdd(&sy[0], 1u) + 1u;
        if (n_done == gridDim.x) {
            __threadfence();
            const uint32_t seq = *(volatile uint32_t *) &sy[1];
            const uint32_t g   = *(volatile uint32_t *) &sy[2];
            *(volatile uint32_t *) &sy[0] = 0;
            fn_l4_post_publish(v, chan, seq, g, ticket);
        }
    }
}

// [TAG_FN_L22_X_PRIOR] the prior's routing alone (no post behind it, e.g. the sync retry graph): one warp per row, <= 8 rows;
// the same routing function, so a retried ubatch picks the ids and weights of its bridged run
template <int n_experts>
static __global__ void __launch_bounds__(FN_L4_TOPK_THREADS, 1) k_fn_l22_topk_route(const float * logits, float * weights,
        int32_t * ids, const int n_rows, const int n_expert_used, const float clamp_val, const float scale_val,
        const bool with_norm, const float * scale_row, const float * mask_row, const int pflags, const float expnm,
        const int r0) {
    const int  lane   = threadIdx.x % WARP_SIZE;
    const int  row    = threadIdx.x / WARP_SIZE;
    const bool is_row = row < n_rows;

    constexpr int EPT = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

    ggml_cuda_pdl_sync();

    float wt[EPT];
#pragma unroll
    for (int i = 0; i < EPT; i++) {
        wt[i] = -INFINITY;
    }
    if (is_row) {
        const float * lg = logits + n_experts * row;
#pragma unroll
        for (int i = 0; i < n_experts; i += WARP_SIZE) {
            wt[i / WARP_SIZE] = lg[i + lane];
        }
    }

    // weights and ids can alias logits: every row reads its logits before any thread writes
    __syncthreads();
    ggml_cuda_pdl_lc();

    if (!is_row) {
        return;
    }
    int   o_id  = 0;
    float o_w   = 0.0f;
    int   o_raw = -1;
    fn_l22_route_row<EPT>(wt, scale_row, mask_row, n_expert_used, pflags & ~GGML_FN_L22_PRIOR_F_RAW, expnm, r0, with_norm,
            clamp_val, scale_val, lane, o_id, o_w, o_raw);
    if (lane < n_expert_used) {
        ids[n_experts*row + lane]         = o_id;
        weights[n_expert_used*row + lane] = o_w;
    }
}

// [TAG_FN_L22_X_PRIOR] R-REV-9: the same routing for more rows than one block takes (verify ubatches of two slots, MTP n_max
// >= 4): one warp per row over many blocks, into compact [k, T] scratch (a block may not write where another block's rows
// still read: k_fn_l22_route_scatter writes ids / weights after), the unbiased top-k straight to raw (null: none)
template <int n_experts>
static __global__ void __launch_bounds__(FN_L4_TOPK_THREADS, 1) k_fn_l22_topk_route_rows(const float * logits, float * w_tmp,
        int32_t * id_tmp, const int n_rows, const int n_expert_used, const float clamp_val, const float scale_val,
        const bool with_norm, const float * scale_row, const float * mask_row, const int pflags, const float expnm,
        const int r0, int32_t * raw) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int row  = blockIdx.x*(FN_L4_TOPK_THREADS/WARP_SIZE) + threadIdx.x/WARP_SIZE;
    if (row >= n_rows) {
        return;
    }
    constexpr int EPT = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;
    float wt[EPT];
    const float * lg = logits + (size_t) n_experts*row;
#pragma unroll
    for (int i = 0; i < n_experts; i += WARP_SIZE) {
        wt[i / WARP_SIZE] = lg[i + lane];
    }
    int   o_id  = 0;
    float o_w   = 0.0f;
    int   o_raw = -1;
    fn_l22_route_row<EPT>(wt, scale_row, mask_row, n_expert_used, raw ? pflags : (pflags & ~GGML_FN_L22_PRIOR_F_RAW), expnm, r0,
            with_norm, clamp_val, scale_val, lane, o_id, o_w, o_raw);
    if (lane < n_expert_used) {
        id_tmp[n_expert_used*row + lane] = o_id;
        w_tmp[n_expert_used*row + lane]  = o_w;
        if (raw) {
            raw[n_expert_used*row + lane] = o_raw;
        }
    }
}

static __global__ void k_fn_l22_route_scatter(const int32_t * id_tmp, const float * w_tmp, int32_t * ids, float * weights,
        const int n, const int k, const int ids_stride) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    ids[ids_stride*(i / k) + i % k] = id_tmp[i];
    weights[i] = w_tmp[i];
}

// hostfunc mode, after a post on stream: fork, the host job runs on the aux stream while this stream continues with the
// device work of the layer
static void mb_post_fork(ggml_moe_bridge * b, const int32_t chan, cudaStream_t stream);

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
    // [TAG_FN_R4_BRIDGE_DMA] the plan area of a job the graph fetches for
    job->plan     = b->params.max_fetch > 0 && (job->flags & GGML_MOE_BRIDGE_JOB_DMA) ?
                    (ggml_moe_bridge_plan *) (p + b->dev.off_plan) : nullptr;
    // [TAG_FN_L22_X_PRIOR] the unbiased top-k a prior job's post wrote
    job->raw_ids  = b->dev.off_raw > 0 && (job->flags & GGML_MOE_BRIDGE_JOB_RAW) ? (const int32_t *) (p + b->dev.off_raw) : nullptr;
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

static void mb_post_fork(ggml_moe_bridge * b, const int32_t chan, cudaStream_t stream) {
    if (b->params.wait_mode == GGML_MOE_BRIDGE_WAIT_HOSTFUNC) {
        CUDA_CHECK(cudaEventRecord(b->ev_post[chan], stream));
        CUDA_CHECK(cudaStreamWaitEvent(b->aux, b->ev_post[chan], 0));
        CUDA_CHECK(cudaLaunchHostFunc(b->aux, mb_hostfn, &b->hf[chan]));
        CUDA_CHECK(cudaEventRecord(b->ev_done[chan], b->aux));
    }
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
        g_mb_gen.fetch_add(1, std::memory_order_acq_rel); // [TAG_FN_BRIDGE_GRAPHGEN] the id and the memory can go to a new bridge
    }
    if (b->glob) {
        // [TAG_FN_R4_BRIDGE_DMA] no device wait or fetch of this bridge spins past this point
        *(volatile uint32_t *) &b->glob->release = 1;
        std::atomic_thread_fence(std::memory_order_seq_cst);
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
    if (b->fetch) {
        (void) cudaFree(b->fetch);
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
        p->max_tokens < 1 || p->max_tokens > 64 || p->max_fetch < 0 || p->max_fetch > GGML_MOE_BRIDGE_MAX_FETCH ||
        p->hint_k < 0 || p->hint_k > GGML_MOE_BRIDGE_MAX_HINT_K || p->raw_ids < 0 || p->raw_ids > 1 ||
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
    // one wait kernel spins at most job_max_ms: keep it well below the ~2 s Windows driver watchdog (TDR)
    constexpr int max_ms = 1500;
    b->params.timeout_ms = std::min(max_ms, std::max(1, p->timeout_ms));
    b->params.job_max_ms = p->job_max_ms > 0 ? std::min(max_ms, std::max(b->params.timeout_ms, p->job_max_ms))
                                             : std::max(b->params.timeout_ms, 1000);

    const int64_t n_embd = p->n_embd;
    const int     max_t  = p->max_tokens;
    const size_t  sx     = mb_pad((size_t) n_embd*max_t*sizeof(float));
    const size_t  si     = mb_pad((size_t) p->n_used*max_t*sizeof(int32_t));
    // [TAG_FN_R4_BRIDGE_DMA] the plan area: the fixed part and the slot ids of [n_used, max_t]
    const size_t  sp = p->max_fetch > 0 ?
        mb_pad(offsetof(ggml_moe_bridge_plan, slot_ids) + (size_t) p->n_used*max_t*sizeof(int32_t)) : 0;
    // [TAG_FN_L3_CPU_DEVPRED] the hint area: its head and the ids of [hint_k, max_t]
    const size_t  sh = p->hint_k > 0 ? mb_pad(sizeof(mb_hint_hdr) + (size_t) p->hint_k*max_t*sizeof(int32_t)) : 0;
    // [TAG_FN_L22_X_PRIOR] the raw-id area: the unbiased top-k of [n_used, max_t] (0 without the prior's raw ids)
    const size_t  sr = p->raw_ids ? si : 0;
    const size_t  chan_bytes = sx + si + si + sx + sp + sh + sr;

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

    rc = cudaMalloc((void **) &b->state, (2 + 5*(size_t) p->n_chan)*sizeof(uint32_t)); // [TAG_FN_L4_POST] + l4sync
    if (rc != cudaSuccess) {
        b->state = nullptr;
        return fail("cudaMalloc", rc);
    }
    if (p->max_fetch > 0) { // [TAG_FN_R4_BRIDGE_DMA]
        rc = cudaMalloc((void **) &b->fetch, (size_t) p->n_chan*sizeof(mb_fetch_dev));
        if (rc != cudaSuccess) {
            b->fetch = nullptr;
            return fail("cudaMalloc (fetch)", rc);
        }
    }
    rc = cudaStreamCreateWithFlags(&b->aux, cudaStreamNonBlocking);
    if (rc != cudaSuccess) {
        b->aux = nullptr;
        return fail("cudaStreamCreateWithFlags", rc);
    }
    rc = cudaMemsetAsync(b->state, 0, (2 + 5*(size_t) p->n_chan)*sizeof(uint32_t), b->aux);
    if (rc == cudaSuccess) {
        rc = cudaStreamSynchronize(b->aux);
    }
    if (rc != cudaSuccess) {
        return fail("clearing the device counters", rc);
    }

    b->fn_l22_raw_ids.assign(p->n_chan, nullptr); // [TAG_FN_L22_X_PRIOR] R-REV-9

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
    b->dev.off_plan   = sx + si + si + sx; // [TAG_FN_R4_BRIDGE_DMA]
    b->dev.off_hint   = sh > 0 ? sx + si + si + sx + sp : 0; // [TAG_FN_L3_CPU_DEVPRED]
    b->dev.off_raw    = sr > 0 ? sx + si + si + sx + sp + sh : 0; // [TAG_FN_L22_X_PRIOR]
    b->dev.hint_k     = p->hint_k;
    b->dev.fetch      = b->fetch;
    b->dev.dma_ring      = nullptr;
    b->dev.dma_ring_size = 0;
    b->dev.max_fetch  = p->max_fetch;
    b->dev.state      = b->state;
    b->dev.l4sync     = b->state + 2 + p->n_chan; // [TAG_FN_L4_POST]
    b->dev.n_chan     = p->n_chan;
    b->dev.mode       = p->wait_mode;
    b->dev.stats      = p->stats ? 1 : 0;
    {
        const char * e = getenv("LLAMA_FN_L14_PROBE"); // [TAG_FN_L14_PROBE]
        b->dev.probe = e && atoi(e) != 0 ? 1 : 0;
    }
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

    GGML_LOG_INFO("%s: bridge %d on device %d: %d channels, n_embd %" PRId64 ", n_used %d, T <= %d, %s wait, timeout %d ms (%d ms once taken), %.2f MiB mapped, fetch <= %d experts, hints <= %d ids per token\n",
            __func__, b->id, p->device, p->n_chan, n_embd, p->n_used, max_t,
            p->wait_mode == GGML_MOE_BRIDGE_WAIT_SPIN ? "spin" : "hostfunc", b->params.timeout_ms, b->params.job_max_ms, total/1024.0/1024.0,
            p->max_fetch, p->hint_k);
    if (sr > 0) {
        GGML_LOG_INFO("%s: [TAG_FN_L22_X_PRIOR] bridge %d: every channel carries the unbiased top-k of a prior post (%zu bytes)\n",
                __func__, b->id, sr);
    }
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
    const uint32_t next = mb_next_stamp(b->served.load(std::memory_order_relaxed));
    mb_ring_entry * e = b->ring + (next % MB_RING);
    const uint32_t st = *(const volatile uint32_t *) &e->stamp;
    if (st != next) {
        if (b->watch && st == 0) {
            return false; // [TAG_FN_L3_HOST_EXEC] a zeroed entry is not posted yet, never a lap (whatever next is)
        }
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
    if (b->watch) {
        // [TAG_FN_L3_HOST_EXEC] the device writes this entry again only one lap (MB_RING posts) later
        *(volatile uint32_t *) &e->stamp = 0;
    }
    if (chan >= (uint32_t) b->params.n_chan) {
        b->served.store(next, std::memory_order_release);
        return false;
    }
    // owed before served: reset() reads served first, so it never sees this job served but not yet owed (it would then
    // let a new post overwrite the channel this job still reads)
    b->taken.fetch_add(1, std::memory_order_seq_cst);
    b->served.store(next, std::memory_order_seq_cst);
    mb_make_job(b, (int32_t) chan, seq, job);
    mb_mark_taken(b, job);
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

// [TAG_FN_L16_PLEGATE] see ggml-moe-bridge.h
bool ggml_backend_cuda_moe_bridge_host_take(ggml_moe_bridge * b, int32_t chan, uint32_t seq, int32_t n_tokens, ggml_moe_bridge_job * job) {
#ifdef GGML_MOE_BRIDGE_DISABLED
    GGML_UNUSED(b);
    GGML_UNUSED(chan);
    GGML_UNUSED(seq);
    GGML_UNUSED(n_tokens);
    GGML_UNUSED(job);
    return false;
#else
    if (b == nullptr || job == nullptr || chan < 0 || chan >= b->params.n_chan || seq == 0 || n_tokens < 1 ||
        n_tokens > b->params.max_tokens) {
        return false;
    }
    char * p = b->data + (size_t) chan*b->dev.chan_bytes;
    memset(job, 0, sizeof(*job));
    job->chan     = chan;
    job->seq      = seq;
    job->n_tokens = n_tokens;
    job->n_embd   = b->params.n_embd;
    job->out      = (float *) (p + b->dev.off_out);
    // owed as a polled job is: complete() counts it done, and reset() waits until it is
    b->taken.fetch_add(1, std::memory_order_seq_cst);
    mb_mark_taken(b, job);
    return true;
#endif
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

// [TAG_FN_R4_BRIDGE_DMA]
bool ggml_backend_cuda_moe_bridge_set_ring(ggml_moe_bridge * b, void * host_ptr, size_t size) {
#ifdef GGML_MOE_BRIDGE_DISABLED
    GGML_UNUSED(b);
    GGML_UNUSED(host_ptr);
    GGML_UNUSED(size);
    return false;
#else
    if (b == nullptr || b->params.max_fetch <= 0 || b->fetch == nullptr || host_ptr == nullptr || size == 0) {
        return false;
    }
    ggml_cuda_set_device(b->params.device);
    void * dptr = nullptr;
    const cudaError_t rc = cudaHostGetDevicePointer(&dptr, host_ptr, 0);
    if (rc != cudaSuccess || dptr == nullptr) {
        GGML_LOG_WARN("%s: bridge %d: the ring is not readable by device %d: %s\n", __func__, b->id, b->params.device,
                cudaGetErrorString(rc));
        (void) cudaGetLastError();
        return false;
    }
    b->dev.dma_ring      = (const char *) dptr;
    b->dev.dma_ring_size = (unsigned long long) size;
    g_mb_gen.fetch_add(1, std::memory_order_acq_rel); // [TAG_FN_BRIDGE_GRAPHGEN] captured kernels hold the old device view
    GGML_LOG_INFO("%s: bridge %d: fetches read a %.0f MiB pinned ring\n", __func__, b->id, size/1048576.0);
    return true;
#endif
}

void ggml_backend_cuda_moe_bridge_publish_plan(ggml_moe_bridge * b, const ggml_moe_bridge_job * job) {
    std::atomic_thread_fence(std::memory_order_release);
    *(volatile uint32_t *) &b->hdr[job->chan].plan = job->seq;
}

void ggml_backend_cuda_moe_bridge_chan_times(const ggml_moe_bridge * b, int32_t chan, ggml_moe_bridge_chan_times * t) {
    memset(t, 0, sizeof(*t));
    if (b == nullptr || chan < 0 || chan >= b->params.n_chan) {
        return;
    }
    const volatile mb_chan_hdr * h = b->hdr + chan;
    t->seq      = h->times_seq;
    std::atomic_thread_fence(std::memory_order_acquire);
    t->wait_ns  = h->wait_ns;
    t->fetch_ns = h->fetch_ns;
    t->n_fetch  = h->n_fetch;
    t->t_post   = h->t_post;   // [TAG_FN_L14_PROBE]
    t->t_wstart = h->t_wstart;
    t->t_wend   = h->t_wend;
    t->t_pub    = h->t_pub;
}

// [TAG_FN_L14_PROBE] the device clock (%globaltimer) minus the host clock (steady_clock ns): five round trips through
// mapped memory, each the mean of a host -> device and a device -> host stamp (their latencies cancel), the median kept.
// false when the device cannot run it
#ifndef GGML_MOE_BRIDGE_DISABLED
static __global__ void k_mb_clock(volatile uint32_t * flag, volatile unsigned long long * g, const int rounds) {
    for (int r = 1; r <= rounds; ++r) {
        const unsigned long long t_lim = mb_now_ns() + 200000000ull;
        while (flag[0] != (uint32_t) r) {
            if (mb_now_ns() > t_lim) {
                return;
            }
        }
        g[2*r] = mb_now_ns();
        __threadfence_system();
        g[2*r + 1] = mb_now_ns();
        __threadfence_system();
        flag[1] = (uint32_t) r;
        __threadfence_system();
    }
}
#endif

bool ggml_backend_cuda_moe_bridge_clock_offset(const ggml_moe_bridge * b, int64_t * off_ns) {
#ifdef GGML_MOE_BRIDGE_DISABLED
    GGML_UNUSED(b);
    GGML_UNUSED(off_ns);
    return false;
#else
    if (b == nullptr || off_ns == nullptr) {
        return false;
    }
    constexpr int R = 5;
    void * hp = nullptr;
    if (cudaHostAlloc(&hp, 256, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    memset(hp, 0, 256);
    volatile uint32_t           * flag = (volatile uint32_t *) hp;
    volatile unsigned long long * g    = (volatile unsigned long long *) ((char *) hp + 64);
    void * dflag = nullptr;
    void * dg    = nullptr;
    cudaStream_t st = nullptr;
    bool ok = cudaHostGetDevicePointer(&dflag, hp, 0) == cudaSuccess &&
              cudaHostGetDevicePointer(&dg, (char *) hp + 64, 0) == cudaSuccess &&
              cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking) == cudaSuccess;
    std::vector<int64_t> offs;
    if (ok) {
        k_mb_clock<<<1, 1, 0, st>>>((volatile uint32_t *) dflag, (volatile unsigned long long *) dg, R);
        ok = cudaGetLastError() == cudaSuccess;
    }
    auto now = []() -> int64_t {
        return (int64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    for (int r = 1; ok && r <= R; ++r) {
        const int64_t h1 = now();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        flag[0] = (uint32_t) r;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        int64_t h2 = 0;
        for (;;) {
            if (flag[1] == (uint32_t) r) {
                h2 = now();
                break;
            }
            if (now() - h1 > 1000000000ll) {
                ok = false;
                break;
            }
        }
        if (ok) {
            std::atomic_thread_fence(std::memory_order_acquire);
            const int64_t g1 = (int64_t) g[2*r];
            const int64_t g2 = (int64_t) g[2*r + 1];
            offs.push_back(((g1 - h1) + (g2 - h2))/2);
        }
    }
    if (st) {
        flag[0] = 0xffffffffu; // a kernel still waiting for a round gives up at its own limit
        cudaStreamSynchronize(st);
        cudaStreamDestroy(st);
    }
    cudaFreeHost(hp);
    (void) cudaGetLastError();
    if (!ok || offs.empty()) {
        return false;
    }
    std::sort(offs.begin(), offs.end());
    *off_ns = offs[offs.size()/2];
    return true;
#endif
}

void ggml_backend_cuda_moe_bridge_release(ggml_moe_bridge * b) {
    if (b && b->glob) {
        *(volatile uint32_t *) &b->glob->release = 1;
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }
}

// [TAG_FN_L3_HOST_EXEC] no graph runs (before the first one): the entries the poll has taken get stamp 0, so from now on
// every entry ahead of served reads 0 until the device posts there
void ggml_backend_cuda_moe_bridge_set_watch(ggml_moe_bridge * b, bool on) {
    if (b == nullptr || b->ring == nullptr || b->params.wait_mode != GGML_MOE_BRIDGE_WAIT_SPIN) {
        return;
    }
    if (on && !b->watch) {
        const uint32_t served = b->served.load(std::memory_order_acquire);
        for (int i = 0; i < MB_RING; ++i) {
            volatile uint32_t * st = &b->ring[i].stamp;
            if ((int32_t) (*st - served) <= 0) {
                *st = 0;
            }
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }
    b->watch = on;
}

const volatile int32_t * ggml_backend_cuda_moe_bridge_next_post_word(ggml_moe_bridge * b) {
    if (b == nullptr || b->ring == nullptr || !b->watch) {
        return nullptr;
    }
    const uint32_t next = mb_next_stamp(b->served.load(std::memory_order_relaxed));
    return (const volatile int32_t *) &b->ring[next % MB_RING].stamp;
}

// [TAG_FN_L3_HOST_EXEC] test hook: the ring counter of a spin bridge restarts at g (no graph runs, no job is owed), so a
// unit test crosses the 2^32 wrap of the stamps in a few posts. Every entry reads as taken (watch: 0, else g).
bool ggml_backend_cuda_moe_bridge_test_seed(ggml_moe_bridge * b, uint32_t g) {
#ifdef GGML_MOE_BRIDGE_DISABLED
    GGML_UNUSED(b);
    GGML_UNUSED(g);
    return false;
#else
    if (b == nullptr || b->ring == nullptr || b->params.wait_mode != GGML_MOE_BRIDGE_WAIT_SPIN || g == 0) {
        return false;
    }
    const uint32_t posted = *(const volatile uint32_t *) &b->glob->posted;
    if (posted != b->served.load(std::memory_order_acquire) ||
        b->completed.load(std::memory_order_acquire) != b->taken.load(std::memory_order_acquire)) {
        return false; // a job is owed
    }
    ggml_cuda_set_device(b->params.device);
    cudaError_t rc = cudaMemcpyAsync(b->state, &g, sizeof(g), cudaMemcpyHostToDevice, b->aux);
    if (rc == cudaSuccess) {
        rc = cudaStreamSynchronize(b->aux);
    }
    if (rc != cudaSuccess) {
        GGML_LOG_WARN("%s: bridge %d: %s\n", __func__, b->id, cudaGetErrorString(rc));
        (void) cudaGetLastError();
        return false;
    }
    for (int i = 0; i < MB_RING; ++i) {
        *(volatile uint32_t *) &b->ring[i].stamp = b->watch ? 0u : g;
    }
    *(volatile uint32_t *) &b->glob->posted = g;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    b->served.store(g, std::memory_order_seq_cst);
    return true;
#endif
}

// [TAG_FN_L3_CPU_DEVPRED] seqlock read: the seq, the ids, the seq again; both must be the job's
bool ggml_backend_cuda_moe_bridge_read_hint(const ggml_moe_bridge * b, int32_t chan, uint32_t seq, int32_t * ids, int max_ids,
                                            int * k, int * n_tokens) {
#ifdef GGML_MOE_BRIDGE_DISABLED
    GGML_UNUSED(b);
    GGML_UNUSED(chan);
    GGML_UNUSED(seq);
    GGML_UNUSED(ids);
    GGML_UNUSED(max_ids);
    GGML_UNUSED(k);
    GGML_UNUSED(n_tokens);
    return false;
#else
    if (b == nullptr || b->dev.hint_k <= 0 || chan < 0 || chan >= b->params.n_chan || seq == 0 || ids == nullptr) {
        return false;
    }
    const char * area = (const char *) b->data + (size_t) chan*b->dev.chan_bytes + b->dev.off_hint;
    const volatile mb_hint_hdr * h = (const volatile mb_hint_hdr *) area;
    if (h->seq != seq) {
        return false;
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    const int hk = h->k;
    const int ht = h->n_tokens;
    if (hk < 1 || hk > b->dev.hint_k || ht < 1 || ht > b->params.max_tokens || hk*ht > max_ids) {
        return false;
    }
    const volatile int32_t * src = (const volatile int32_t *) (area + sizeof(mb_hint_hdr));
    for (int i = 0; i < hk*ht; ++i) {
        ids[i] = src[i];
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    if (h->seq != seq) {
        return false; // the device rewrote it meanwhile
    }
    *k        = hk;
    *n_tokens = ht;
    return true;
#endif
}

// [TAG_FN_L22_DMASHARE] R-P14-2 (a), see moe-bridge.cuh
bool ggml_cuda_moe_bridge_fetch_view(int32_t bridge_id, int device, int32_t chan, ggml_cuda_mb_fetch_view * out) {
#ifdef GGML_MOE_BRIDGE_DISABLED
    GGML_UNUSED(bridge_id);
    GGML_UNUSED(device);
    GGML_UNUSED(chan);
    GGML_UNUSED(out);
    return false;
#else
    const ggml_moe_bridge * b = mb_get(bridge_id);
    if (b == nullptr || out == nullptr || b->params.device != device || chan < 0 || chan >= b->params.n_chan ||
        b->dev.off_plan == 0 || b->dev.max_fetch <= 0) {
        return false;
    }
    const mb_dev & v = b->dev;
    out->plan       = (ggml_moe_bridge_plan *) (v.data + (size_t) chan*v.chan_bytes + v.off_plan);
    out->plan_word  = &v.hdr[chan].plan;
    out->done_word  = &v.hdr[chan].done;
    out->taken_word = &v.hdr[chan].taken;
    out->err_state  = v.state + 1;
    out->release    = &v.glob->release;
    out->g_err_chan = &v.glob->err_chan;
    out->g_err_seq  = &v.glob->err_seq;
    out->g_err      = &v.glob->err;
    out->timeout_ns = v.timeout_ns;
    out->job_max_ns = v.job_max_ns;
    out->max_fetch  = v.max_fetch;
    return true;
#endif
}

// ---- ops ------------------------------------------------------------------------------------------------------------

#ifndef GGML_MOE_BRIDGE_DISABLED
// [TAG_FN_L22_X_PRIOR] the post's ids come from a top-k view of an ARGSORT over the prior's marked MUL (graph metadata only)
static bool fn_l22_prior_ids(const ggml_tensor * ids) {
    const ggml_tensor * a = ids ? ids->view_src : nullptr;
    return a != nullptr && a->op == GGML_OP_ARGSORT && ggml_fn_l22_prior_get(a->src[0], nullptr, nullptr, nullptr, nullptr);
}
#endif

// [TAG_UPSTREAM_30804] hostfunc-mode bridges fork the post onto their aux stream and join it at the wait, a later node:
// a CUDA graph chunk must not end between the two (the capture would end with an unjoined stream), so ggml-cuda.cu
// captures in one piece while such a bridge exists
bool ggml_cuda_moe_bridge_hostfunc_any() {
#ifdef GGML_MOE_BRIDGE_DISABLED
    return false;
#else
    for (int id = 0; id < MB_MAX_BRIDGES; ++id) {
        const ggml_moe_bridge * b = mb_get(id);
        if (b != nullptr && b->params.wait_mode == GGML_MOE_BRIDGE_WAIT_HOSTFUNC) {
            return true;
        }
    }
    return false;
#endif
}

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
    if (op->op == GGML_OP_MOE_HOST_POST && ggml_get_op_params_i32(op, 3) == 2) {
        // [TAG_FN_L9_HINTTOPK] a hint from logits: src[1] f32 [n_expert <= 512, T]
        const ggml_tensor * ticket = op->src[0];
        const ggml_tensor * lg     = op->src[1];
        const int32_t       k      = ggml_get_op_params_i32(op, 4);
        return p.hint_k > 0 && ticket && lg && ticket->type == GGML_TYPE_I32 && ggml_nelements(ticket) == 1 &&
            lg->type == GGML_TYPE_F32 && lg->nb[0] == sizeof(float) && lg->ne[0] >= 1 && lg->ne[0] <= 16*WARP_SIZE &&
            k >= 1 && k <= p.hint_k && k <= lg->ne[0] && lg->ne[1] >= 1 && lg->ne[1] <= p.max_tokens && lg->ne[1] <= 8 &&
            lg->ne[2] == 1 && lg->ne[3] == 1;
    }
    if (op->op == GGML_OP_MOE_HOST_POST && ggml_get_op_params_i32(op, 3) == 1) {
        // [TAG_FN_L3_CPU_DEVPRED] a hint: src[0] the post's ticket, src[1] the predicted ids [k, T]
        const ggml_tensor * ticket = op->src[0];
        const ggml_tensor * ids    = op->src[1];
        return p.hint_k > 0 && ticket && ids && ticket->type == GGML_TYPE_I32 && ggml_nelements(ticket) == 1 &&
            ids->type == GGML_TYPE_I32 && ids->nb[0] == sizeof(int32_t) && ids->ne[0] >= 1 && ids->ne[0] <= p.hint_k &&
            ids->ne[1] >= 1 && ids->ne[1] <= p.max_tokens && ids->ne[2] == 1 && ids->ne[3] == 1;
    }
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
    if (op->op == GGML_OP_MOE_HOST_FETCH) { // [TAG_FN_R4_BRIDGE_DMA] [TAG_FN_L22_DMASHARE] R-P14-2 (b): or P14's fetch
        return p.max_fetch > 0 && (b->dev.dma_ring != nullptr || ggml_cuda_fn_l22_dmafetch_supported(device, op)) &&
            op->type == GGML_TYPE_I32 && ggml_is_contiguous(op) &&
            op->ne[0] >= 1 && op->ne[0] <= p.n_used && op->ne[1] >= 1 && op->ne[1] <= p.max_tokens &&
            op->src[1] && op->src[2] && op->src[3] && op->src[1]->ne[2] >= 2 && op->src[1]->ne[2] - 1 <= INT32_MAX;
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

    if (ggml_get_op_params_i32(dst, 3) == 2) { // [TAG_FN_L9_HINTTOPK] a hint from logits
        GGML_ASSERT(b->dev.hint_k > 0 && b->dev.off_hint > 0);
        const ggml_tensor * lg = dst->src[1];
        const int k = ggml_get_op_params_i32(dst, 4);
        GGML_ASSERT(k >= 1 && k <= b->dev.hint_k && lg->ne[1] <= b->params.max_tokens && lg->ne[1] <= 8 && lg->ne[0] <= 16*WARP_SIZE);
        k_fn_l9_hint_topk<16><<<1, WARP_SIZE*(int) lg->ne[1], 0, ctx.stream()>>>(b->dev, chan, (const int32_t *) dst->src[0]->data,
                (const char *) lg->data, lg->nb[1], (int) lg->ne[0], k, (int) lg->ne[1]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (ggml_get_op_params_i32(dst, 3) == 1) { // [TAG_FN_L3_CPU_DEVPRED] a hint
        GGML_ASSERT(b->dev.hint_k > 0 && b->dev.off_hint > 0);
        const ggml_tensor * ids = dst->src[1];
        GGML_ASSERT(ids->ne[0] <= b->dev.hint_k && ids->ne[1] <= b->params.max_tokens);
        k_mb_hint<<<1, 128, 0, ctx.stream()>>>(b->dev, chan, (const int32_t *) dst->src[0]->data, (const char *) ids->data,
                ids->nb[1], (int) ids->ne[0], (int) ids->ne[1]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const ggml_tensor * x   = dst->src[0];
    const ggml_tensor * ids = dst->src[1];
    const ggml_tensor * w   = dst->src[2];

    // [TAG_FN_L22_X_PRIOR] ids that an unfused (or post-less fused) prior top-k picked: tell the host; R-REV-9: RAW when the
    // multi-row route wrote this post's unbiased ids
    int32_t flags_p = flags | (fn_l22_prior_ids(ids) ? GGML_MOE_BRIDGE_JOB_PRIOR : 0);
    if ((flags_p & GGML_MOE_BRIDGE_JOB_PRIOR) && chan < (int32_t) b->fn_l22_raw_ids.size() && b->fn_l22_raw_ids[chan] != nullptr) {
        if (b->fn_l22_raw_ids[chan] == ids->data) {
            flags_p |= GGML_MOE_BRIDGE_JOB_RAW;
        }
        b->fn_l22_raw_ids[chan] = nullptr;
    }

    cudaStream_t stream = ctx.stream();
    if (ggml_fn_l3_get(dst) == GGML_FN_L4_POST) { // [TAG_FN_L4_POST]
        k_fn_l4_mb_post<<<1, FN_L4_POST_THREADS, 0, stream>>>(b->dev, chan,
                (const char *) x->data, x->nb[1], (const char *) ids->data, ids->nb[1], (const float *) w->data,
                (int) x->ne[0], (int) ids->ne[0], (int) x->ne[1], flags_p, (int32_t *) dst->data);
    } else {
        k_mb_post<<<1, MB_POST_THREADS, 0, stream>>>(b->dev, chan,
                (const char *) x->data, x->nb[1], (const char *) ids->data, ids->nb[1], (const float *) w->data,
                (int) x->ne[0], (int) ids->ne[0], (int) x->ne[1], flags_p, (int32_t *) dst->data);
    }
    CUDA_CHECK(cudaGetLastError());

    mb_post_fork(b, chan, stream);
#endif
}

// [TAG_FN_L4_POST] see moe-bridge.cuh
bool ggml_cuda_fn_l4_topk_post(ggml_backend_cuda_context & ctx, const ggml_tensor * logits, ggml_tensor * weights,
        ggml_tensor * ids, const ggml_tensor * clamp, const ggml_tensor * scale, const ggml_tensor * bias,
        const ggml_cuda_topk_moe_args & args, ggml_tensor * post) {
#ifdef GGML_MOE_BRIDGE_DISABLED
    GGML_UNUSED(ctx);
    GGML_UNUSED(logits);
    GGML_UNUSED(weights);
    GGML_UNUSED(ids);
    GGML_UNUSED(clamp);
    GGML_UNUSED(scale);
    GGML_UNUSED(bias);
    GGML_UNUSED(args);
    GGML_UNUSED(post);
    return false;
#else
    if (post == nullptr || post->op != GGML_OP_MOE_HOST_POST || ggml_fn_l3_get(post) != GGML_FN_L4_POST ||
        ggml_get_op_params_i32(post, 3) != 0 || bias != nullptr || args.delayed_softmax || args.prob_bias) {
        return false;
    }
    ggml_moe_bridge * b = mb_get(ggml_get_op_params_i32(post, 0));
    const int32_t chan  = ggml_get_op_params_i32(post, 1);
    const int32_t flags = ggml_get_op_params_i32(post, 2);
    if (b == nullptr || b->params.device != ctx.device || chan < 0 || chan >= b->params.n_chan) {
        return false;
    }
    const ggml_tensor * x    = post->src[0];
    const ggml_tensor * pids = post->src[1];
    const ggml_tensor * pw   = post->src[2];

    const int64_t n_experts = logits->ne[0];
    const int64_t n_rows    = logits->ne[1];
    const int64_t n_used    = weights->ne[1];
    // the top-k of 512 experts (Flash-Next), one block; the post reads exactly the top-k's two outputs
    if (n_experts != 512 || n_rows < 1 || n_rows > TOPK_MOE_ROWS_PER_BLOCK || n_rows > b->params.max_tokens ||
        n_used < 1 || n_used > b->params.n_used || n_used > n_experts || ggml_nrows(logits) != n_rows ||
        logits->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 ||
        !ggml_is_contiguous(logits) || !ggml_is_contiguous(weights) || ggml_nelements(weights) != n_used*n_rows ||
        ids->nb[1] != (size_t) n_experts*sizeof(int32_t) ||
        pids->data != ids->data || pids->nb[1] != ids->nb[1] || pids->ne[0] != n_used || pids->ne[1] != n_rows ||
        pw->data != weights->data || ggml_nelements(pw) != n_used*n_rows ||
        x->type != GGML_TYPE_F32 || x->ne[0] != b->params.n_embd || x->ne[1] != n_rows || x->nb[0] != sizeof(float)) {
        return false;
    }

    fn_l4_topk_config config;
    config.use_sigmoid       = args.sigmoid;
    config.use_sqrt_softplus = args.sqrt_softplus;
    config.with_norm         = clamp != nullptr;

    const float clamp_val = clamp ? ggml_get_op_params_f32(clamp, 0) : -INFINITY;
    const float scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    // x blocks: about one 16-byte word per thread
    const int64_t x_words = n_rows*x->ne[0]/4;
    const int     n_xblk  = (int) std::min<int64_t>(FN_L4_XBLK_MAX, std::max<int64_t>(1, (x_words + FN_L4_TOPK_THREADS - 1)/FN_L4_TOPK_THREADS));

    cudaStream_t stream = ctx.stream();
    const ggml_cuda_kernel_launch_params lp(dim3(1 + n_xblk, 1, 1), dim3(FN_L4_TOPK_THREADS, 1, 1), 0, stream);
    ggml_cuda_kernel_launch(k_fn_l4_topk_post<512>, lp, b->dev, (int) chan,
            (const float *) logits->data, (float *) weights->data, (int32_t *) ids->data, (int) n_rows, (int) n_used,
            clamp_val, scale_val, config, (const char *) x->data, (int64_t) x->nb[1], (int) x->ne[0], (int) flags,
            (int32_t *) post->data);

    mb_post_fork(b, chan, stream);

    static std::atomic<bool> noted{false};
    if (!noted.exchange(true)) {
        GGML_LOG_INFO("%s: [TAG_FN_L4_POST] the router's top-k and the bridge post run in one launch\n", __func__);
    }
    return true;
#endif
}

// ---- [TAG_FN_L22_X_PRIOR] R-P12-1: the route prior's pattern, matched and launched for ggml-cuda.cu ---------------------

#ifndef GGML_MOE_BRIDGE_DISABLED
namespace {

constexpr int64_t FN_L22_PRIOR_MAX_ROWS = 1024; // R-REV-9

struct fn_l22_prior_match {
    int                 n       = 0;       // nodes from i, the last one = the weights
    const ggml_tensor * logits  = nullptr; // the SOFT_MAX's input [512, T]
    const ggml_tensor * mul     = nullptr; // the marked MUL; src[1] = row 0 of layer il of the table
    ggml_tensor *       ids     = nullptr; // the top-k VIEW of the ARGSORT
    ggml_tensor *       weights = nullptr;
    const ggml_tensor * clamp   = nullptr; // the norm's CLAMP (nullptr: no norm)
    const ggml_tensor * scale   = nullptr; // the SCALE (nullptr: none)
    int32_t             pflags  = 0;
    float               expnm   = 1.0f;
    int32_t             r0      = 0;
};

// graph metadata only (graph_optimize calls it before the allocation)
int fn_l22_prior_match_at(const ggml_cgraph * cg, const int i, fn_l22_prior_match & m) {
    m = fn_l22_prior_match();
    const int n_nodes = cg->n_nodes;
    if (i < 0 || i + 6 > n_nodes) {
        return 0;
    }
    ggml_tensor ** nd = cg->nodes;
    const ggml_tensor * sm = nd[i];
    if (sm->op != GGML_OP_SOFT_MAX || sm->src[1] != nullptr || sm->src[2] != nullptr) {
        return 0;
    }
    float sm_scale = 1.0f;
    float sm_bias  = 0.0f;
    memcpy(&sm_scale, (const float *) sm->op_params + 0, sizeof(float));
    memcpy(&sm_bias,  (const float *) sm->op_params + 1, sizeof(float));
    const ggml_tensor * logits = sm->src[0];
    // R-REV-9: up to FN_L22_PRIOR_MAX_ROWS rows (more than one block's 8: the multi-row route below)
    if (sm_scale != 1.0f || sm_bias != 0.0f || logits == nullptr || logits->type != GGML_TYPE_F32 || logits->ne[0] != 512 ||
        !ggml_is_contiguous(logits) || logits->ne[1] < 1 || logits->ne[1] > FN_L22_PRIOR_MAX_ROWS ||
        ggml_nrows(logits) != logits->ne[1]) {
        return 0;
    }
    const ggml_tensor * rs = nd[i + 1];
    if (rs->op != GGML_OP_RESHAPE || rs->src[0] != sm) {
        return 0;
    }
    const ggml_tensor * mul = nd[i + 2];
    int32_t pflags = 0;
    float   expnm  = 1.0f;
    int32_t r0     = 0;
    int32_t il     = 0;
    if (!ggml_fn_l22_prior_get(mul, &pflags, &expnm, &r0, &il) || mul->src[0] != sm) {
        return 0;
    }
    const ggml_tensor * row = mul->src[1];
    const ggml_tensor * tab = row ? row->view_src : nullptr;
    if (row == nullptr || tab == nullptr || row->type != GGML_TYPE_F32 || row->ne[0] != 512 || ggml_nelements(row) != 512 ||
        !ggml_is_contiguous(row) || tab->type != GGML_TYPE_F32 || tab->ne[0] != 512 || tab->ne[1] < 2 ||
        !ggml_is_contiguous(tab) || row->view_offs % tab->nb[2] != 0 || row->view_offs + 2*tab->nb[1] > ggml_nbytes(tab)) {
        return 0;
    }
    const ggml_tensor * as = nd[i + 3];
    if (as->op != GGML_OP_ARGSORT || as->src[0] != mul || ggml_get_op_params_i32(as, 0) != GGML_SORT_ORDER_DESC) {
        return 0;
    }
    ggml_tensor * vw = nd[i + 4];
    if (vw->op != GGML_OP_VIEW || vw->src[0] != as) {
        return 0;
    }
    const ggml_tensor * gr = nd[i + 5];
    if (gr->op != GGML_OP_GET_ROWS || gr->src[0] != rs || gr->src[1] != vw) {
        return 0;
    }

    ggml_op ops[16] = { GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_MUL, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS };
    int n_ops = 6;
    int j     = i + 6;
    const ggml_tensor * clamp = nullptr;
    const ggml_tensor * scale = nullptr;
    // RESHAPE -> SUM_ROWS -> CLAMP -> DIV (of the RESHAPE by the CLAMP) -> RESHAPE: the norm
    if (j + 4 < n_nodes && nd[j]->op == GGML_OP_RESHAPE && nd[j]->src[0] == gr &&
        nd[j + 1]->op == GGML_OP_SUM_ROWS && nd[j + 1]->src[0] == nd[j] &&
        nd[j + 2]->op == GGML_OP_CLAMP && nd[j + 2]->src[0] == nd[j + 1] &&
        nd[j + 3]->op == GGML_OP_DIV && nd[j + 3]->src[0] == nd[j] && nd[j + 3]->src[1] == nd[j + 2] &&
        nd[j + 4]->op == GGML_OP_RESHAPE && nd[j + 4]->src[0] == nd[j + 3]) {
        clamp = nd[j + 2];
        const ggml_op norm_ops[5] = { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE };
        for (const ggml_op o : norm_ops) {
            ops[n_ops++] = o;
        }
        j += 5;
    }
    if (j < n_nodes && nd[j]->op == GGML_OP_SCALE && nd[j]->src[0] == nd[j - 1] && ggml_get_op_params_f32(nd[j], 1) == 0.0f) {
        scale = nd[j];
        ops[n_ops++] = GGML_OP_SCALE;
        j++;
    }
    ggml_tensor * weights = nd[j - 1];
    const int out_nodes[2] = { i + 4, j - 1 };
    if (!ggml_can_fuse_subgraph(cg, i, n_ops, ops, out_nodes, 2) || !ggml_cuda_should_use_topk_moe(sm, logits, weights, vw)) {
        return 0;
    }
    const int64_t k = vw->ne[0];
    if (k < 1 || k > WARP_SIZE || vw->ne[1] != logits->ne[1] || vw->nb[0] != sizeof(int32_t) ||
        vw->nb[1] != 512*sizeof(int32_t) || vw->type != GGML_TYPE_I32 || weights->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(weights) || ggml_nelements(weights) != k*logits->ne[1]) {
        return 0;
    }
    m.n       = n_ops;
    m.logits  = logits;
    m.mul     = mul;
    m.ids     = vw;
    m.weights = weights;
    m.clamp   = clamp;
    m.scale   = scale;
    m.pflags  = pflags;
    m.expnm   = expnm;
    m.r0      = r0;
    return n_ops;
}

bool fn_l22_is_view_or_noop(const ggml_tensor * t) { // ggml_cuda_is_view_or_noop (static in ggml-cuda.cu)
    return ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE || t->op == GGML_OP_VIEW ||
           t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}

} // namespace
#endif // GGML_MOE_BRIDGE_DISABLED

int ggml_cuda_fn_l22_prior_nodes(const ggml_cgraph * cgraph, int i) {
#ifdef GGML_MOE_BRIDGE_DISABLED
    GGML_UNUSED(cgraph);
    GGML_UNUSED(i);
    return 0;
#else
    fn_l22_prior_match m;
    return fn_l22_prior_match_at(cgraph, i, m);
#endif
}

int ggml_cuda_fn_l22_try_prior(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
#ifdef GGML_MOE_BRIDGE_DISABLED
    GGML_UNUSED(ctx);
    GGML_UNUSED(cgraph);
    GGML_UNUSED(i);
    return 0;
#else
    fn_l22_prior_match m;
    if (fn_l22_prior_match_at(cgraph, i, m) == 0) {
        int32_t pflags = 0;
        if (i + 2 < cgraph->n_nodes && ggml_fn_l22_prior_get(cgraph->nodes[i + 2], &pflags, nullptr, nullptr, nullptr) &&
            (pflags & GGML_FN_L22_PRIOR_F_RESIDENT)) {
            static std::atomic<bool> told{false};
            if (!told.exchange(true)) {
                GGML_LOG_WARN("%s: [TAG_FN_L22_X_PRIOR] a prior top-k does not fit the fused path (%lld rows): it runs unfused, "
                        "with the prior but without the resident swap\n", __func__, cgraph->nodes[i]->src[0] ? (long long) cgraph->nodes[i]->src[0]->ne[1] : 0LL);
            }
        }
        return 0;
    }
    const ggml_tensor * row = m.mul->src[1];
    if (row->data == nullptr || m.logits->data == nullptr || m.ids->data == nullptr || m.weights->data == nullptr) {
        return 0;
    }
    const float * scale_row = (const float *) row->data;
    const float * mask_row  = (const float *) ((const char *) row->data + row->view_src->nb[1]);
    const int     n_rows    = (int) m.logits->ne[1];
    const int     n_used    = (int) m.ids->ne[0];
    const bool    with_norm = m.clamp != nullptr;
    const float   clamp_val = m.clamp ? ggml_get_op_params_f32(m.clamp, 0) : -INFINITY;
    const float   scale_val = m.scale ? ggml_get_op_params_f32(m.scale, 0) : 1.0f;
    cudaStream_t  stream    = ctx.stream();

    // the bridge post right behind it (only views between): one launch for both, as k_fn_l4_topk_post
    int j = i + m.n;
    while (j < cgraph->n_nodes && fn_l22_is_view_or_noop(cgraph->nodes[j])) {
        ++j;
    }
    ggml_tensor * post = j < cgraph->n_nodes ? cgraph->nodes[j] : nullptr;

    // R-REV-9: more rows than one block: the multi-row route (prior and resident swap); the unbiased top-k goes to the raw
    // area of the post behind it, which then runs on its own and flags the job RAW
    if (n_rows > TOPK_MOE_ROWS_PER_BLOCK) {
        int32_t * raw = nullptr;
        if ((m.pflags & GGML_FN_L22_PRIOR_F_RAW) && post != nullptr && post->op == GGML_OP_MOE_HOST_POST &&
            (post->flags & GGML_TENSOR_FLAG_COMPUTE) && ggml_get_op_params_i32(post, 3) == 0) {
            ggml_moe_bridge * b    = mb_get(ggml_get_op_params_i32(post, 0));
            const int32_t     chan = ggml_get_op_params_i32(post, 1);
            const ggml_tensor * pids = post->src[1];
            if (b != nullptr && b->params.device == ctx.device && b->dev.off_raw > 0 && chan >= 0 && chan < b->params.n_chan &&
                chan < (int32_t) b->fn_l22_raw_ids.size() && n_rows <= b->params.max_tokens && n_used <= b->params.n_used &&
                pids != nullptr && pids->data == m.ids->data) {
                raw = (int32_t *) (b->dev.data + (size_t) chan*b->dev.chan_bytes + b->dev.off_raw);
                b->fn_l22_raw_ids[chan] = m.ids->data;
            }
        }
        const int n_ent = n_rows*n_used;
        ggml_cuda_pool_alloc<int32_t> id_tmp(ctx.pool(), n_ent);
        ggml_cuda_pool_alloc<float>   w_tmp(ctx.pool(), n_ent);
        const int rpb = FN_L4_TOPK_THREADS/WARP_SIZE;
        k_fn_l22_topk_route_rows<512><<<(n_rows + rpb - 1)/rpb, FN_L4_TOPK_THREADS, 0, stream>>>((const float *) m.logits->data,
                w_tmp.get(), id_tmp.get(), n_rows, n_used, clamp_val, scale_val, with_norm, scale_row, mask_row, (int) m.pflags,
                m.expnm, (int) m.r0, raw);
        CUDA_CHECK(cudaGetLastError());
        k_fn_l22_route_scatter<<<(n_ent + 255)/256, 256, 0, stream>>>(id_tmp.get(), w_tmp.get(), (int32_t *) m.ids->data,
                (float *) m.weights->data, n_ent, n_used, (int) (m.ids->nb[1]/sizeof(int32_t)));
        CUDA_CHECK(cudaGetLastError());
        static std::atomic<bool> noted{false};
        if (!noted.exchange(true)) {
            GGML_LOG_INFO("%s: [TAG_FN_L22_X_PRIOR] the prior top-k of %d rows (flags %d) runs over %d blocks%s\n", __func__, n_rows,
                    (int) m.pflags, (n_rows + rpb - 1)/rpb, raw ? ", the unbiased ids to the post's raw area" : "");
        }
        return m.n - 1;
    }
    if (post != nullptr && post->op == GGML_OP_MOE_HOST_POST && ggml_fn_l3_get(post) == GGML_FN_L4_POST &&
        (post->flags & GGML_TENSOR_FLAG_COMPUTE) && ggml_get_op_params_i32(post, 3) == 0) {
        ggml_moe_bridge * b = mb_get(ggml_get_op_params_i32(post, 0));
        const int32_t chan  = ggml_get_op_params_i32(post, 1);
        const int32_t flags = ggml_get_op_params_i32(post, 2);
        const ggml_tensor * x    = post->src[0];
        const ggml_tensor * pids = post->src[1];
        const ggml_tensor * pw   = post->src[2];
        if (b != nullptr && b->params.device == ctx.device && chan >= 0 && chan < b->params.n_chan &&
            n_rows <= b->params.max_tokens && n_used <= b->params.n_used &&
            pids->data == m.ids->data && pids->nb[1] == m.ids->nb[1] && pids->ne[0] == n_used && pids->ne[1] == n_rows &&
            pw->data == m.weights->data && ggml_nelements(pw) == (int64_t) n_used*n_rows &&
            x->type == GGML_TYPE_F32 && x->ne[0] == b->params.n_embd && x->ne[1] == n_rows && x->nb[0] == sizeof(float)) {
            const bool raw = (m.pflags & GGML_FN_L22_PRIOR_F_RAW) && b->dev.off_raw > 0;
            const int  hdr_flags = flags | GGML_MOE_BRIDGE_JOB_PRIOR | (raw ? GGML_MOE_BRIDGE_JOB_RAW : 0);
            const int  pflags    = raw ? m.pflags : (m.pflags & ~GGML_FN_L22_PRIOR_F_RAW);
            const int64_t x_words = (int64_t) n_rows*x->ne[0]/4;
            const int     n_xblk  = (int) std::min<int64_t>(FN_L4_XBLK_MAX,
                    std::max<int64_t>(1, (x_words + FN_L4_TOPK_THREADS - 1)/FN_L4_TOPK_THREADS));
            const ggml_cuda_kernel_launch_params lp(dim3(1 + n_xblk, 1, 1), dim3(FN_L4_TOPK_THREADS, 1, 1), 0, stream);
            ggml_cuda_kernel_launch(k_fn_l22_topk_post_prior<512>, lp, b->dev, (int) chan,
                    (const float *) m.logits->data, (float *) m.weights->data, (int32_t *) m.ids->data, n_rows, n_used,
                    clamp_val, scale_val, with_norm, scale_row, mask_row, pflags, m.expnm, (int) m.r0,
                    (const char *) x->data, (int64_t) x->nb[1], (int) x->ne[0], hdr_flags, (int32_t *) post->data);
            mb_post_fork(b, chan, stream);
            static std::atomic<bool> noted{false};
            if (!noted.exchange(true)) {
                GGML_LOG_INFO("%s: [TAG_FN_L22_X_PRIOR] the prior top-k (flags %d) and the bridge post run in one launch\n",
                        __func__, (int) pflags);
            }
            return j - i;
        }
    }

    const ggml_cuda_kernel_launch_params lp(dim3(1, 1, 1), dim3(FN_L4_TOPK_THREADS, 1, 1), 0, stream);
    ggml_cuda_kernel_launch(k_fn_l22_topk_route<512>, lp, (const float *) m.logits->data, (float *) m.weights->data,
            (int32_t *) m.ids->data, n_rows, n_used, clamp_val, scale_val, with_norm, scale_row, mask_row, (int) m.pflags,
            m.expnm, (int) m.r0);
    static std::atomic<bool> noted{false};
    if (!noted.exchange(true)) {
        GGML_LOG_INFO("%s: [TAG_FN_L22_X_PRIOR] the prior top-k (flags %d) runs in one launch (no fused post behind it)\n",
                __func__, (int) m.pflags);
    }
    return m.n - 1;
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
    // [TAG_FN_L18_GPUFLOOR_PFWAIT] the ranges ggml-cuda.cu stored for this node: 2 loader blocks per SM
    fn_l18_pf_pending & pf = g_fn_l18_pf[ctx.device];
    if (pf.wait == dst && pf.nr > 0) {
        fn_l18_pf_args a = {};
        a.n_copy = blocks;
        for (int k = 0; k < pf.nr; ++k) {
            GGML_ASSERT(((uintptr_t) pf.p[k] & 15) == 0);
            const size_t nb16 = pf.n[k] & ~(size_t) 15;
            if (nb16 == 0) {
                continue;
            }
            const size_t nc = (nb16 + FN_L18_PF_CHUNK - 1)/FN_L18_PF_CHUNK;
            GGML_ASSERT(nc < (1u << 24));
            a.p[a.nr]    = pf.p[k];
            a.nc[a.nr]   = (unsigned int) nc;
            a.last[a.nr] = (unsigned int) (nb16 - (nc - 1)*FN_L18_PF_CHUNK);
            a.n_chunks  += (unsigned int) nc;
            a.nr++;
        }
        pf.wait = nullptr;
        pf.nr   = 0;
        if (a.n_chunks > 0) {
            const int n_load = 2*ggml_cuda_info().devices[ctx.device].nsm;
            k_mb_wait_pf<<<blocks + n_load, MB_WAIT_THREADS, 0, stream>>>(b->dev, chan, (const int32_t *) dst->src[0]->data,
                    (float *) dst->data, n, a);
            CUDA_CHECK(cudaGetLastError());
            static std::atomic<bool> noted{false};
            if (!noted.exchange(true)) {
                GGML_LOG_INFO("%s: [TAG_FN_L18_GPUFLOOR_PFWAIT] wait with %d loader blocks, %d ranges, %u chunks of %d bytes\n",
                        __func__, n_load, a.nr, a.n_chunks, (int) FN_L18_PF_CHUNK);
            }
            return;
        }
    }
    pf.wait = nullptr;
    pf.nr   = 0;
    k_mb_wait<<<blocks, MB_WAIT_THREADS, 0, stream>>>(b->dev, chan, (const int32_t *) dst->src[0]->data, (float *) dst->data, n);
    CUDA_CHECK(cudaGetLastError());
#endif
}

// [TAG_FN_R4_BRIDGE_DMA] the plan, then the copies; both on the compute stream, so the bank chain after it sees the bytes
void ggml_cuda_op_moe_host_fetch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#ifdef GGML_MOE_BRIDGE_DISABLED
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("MOE_HOST_FETCH is not supported on this backend");
#else
    // [TAG_FN_L22_DMASHARE] P14: the fetch from pinned pages (stub: false, the ring fetch below)
    if (ggml_cuda_fn_l22_dmafetch(ctx, dst)) {
        return;
    }
    ggml_moe_bridge * b = mb_get(ggml_get_op_params_i32(dst, 0));
    GGML_ASSERT(b != nullptr && b->params.device == ctx.device && b->fetch != nullptr && b->dev.dma_ring != nullptr);
    const int32_t chan = ggml_get_op_params_i32(dst, 1);

    const ggml_tensor * up   = dst->src[1];
    const ggml_tensor * gate = dst->src[2];
    const ggml_tensor * down = dst->src[3];

    const int zero_slot = (int) up->ne[2] - 1;
    const unsigned long long expert_bytes = (unsigned long long) (up->nb[2] + gate->nb[2] + down->nb[2]);

    cudaStream_t stream = ctx.stream();
    k_mb_fetch_plan<<<1, 128, 0, stream>>>(b->dev, chan, (const int32_t *) dst->src[0]->data, (int32_t *) dst->data,
            (int) ggml_nelements(dst), zero_slot, expert_bytes);
    CUDA_CHECK(cudaGetLastError());

    // 48 blocks per part: about 20 KiB per block for a 1 MB q4_K part, enough loads in flight to fill a Gen5 link
    const dim3 grid(48, 3, (unsigned) b->params.max_fetch);
    k_mb_fetch_copy<<<grid, 256, 0, stream>>>(b->dev, chan,
            (char *) up->data,   (long long) up->nb[2],   (long long) up->nb[2],
            (char *) gate->data, (long long) gate->nb[2], (long long) gate->nb[2],
            (char *) down->data, (long long) down->nb[2], (long long) down->nb[2]);
    CUDA_CHECK(cudaGetLastError());
#endif
}
