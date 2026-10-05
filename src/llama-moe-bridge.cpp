#include "llama-moe-bridge.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-moecache.h"
#include "llama-moe-gen5.h" // [TAG_FN_R4_BRIDGE_DMA]
#include "llama-moe-gen5-impl.h" // [TAG_FN_R4_BRIDGE_DMA] gen5::dma_requested

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-moe-bridge.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#    include <immintrin.h>
#endif

#if defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <psapi.h>
#else
#    include <sys/resource.h>
#endif

namespace {

using pool_params_default_t = ggml_cpu_moe_pool_params (*)(int);
using pool_new_t            = ggml_cpu_moe_pool * (*)(const ggml_cpu_moe_pool_params *);
using pool_free_t           = void (*)(ggml_cpu_moe_pool *);
using pool_run_t            = ggml_status (*)(ggml_cpu_moe_pool *, const ggml_cpu_moe_job *);
using pool_park_t           = void (*)(ggml_cpu_moe_pool *);
using layer_supported_t     = bool (*)(const ggml_cpu_moe_layer *);
using pool_pf_t             = ggml_status (*)(ggml_cpu_moe_pool *, const ggml_cpu_moe_prefetch_job *); // [TAG_FN_R2_BRIDGE_PF]
using pool_pf_stop_t        = void (*)(ggml_cpu_moe_pool *);
using pool_pf_stats_t       = void (*)(ggml_cpu_moe_pool *, uint64_t *, uint64_t *, uint64_t *, uint64_t *);
using pool_set_solo_t       = void (*)(ggml_cpu_moe_pool *, bool);
using pool_get_stats_t      = void (*)(ggml_cpu_moe_pool *, ggml_cpu_moe_pool_stats *, bool); // [TAG_FN_L3_CPU_STATS]

inline void br_relax() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    _mm_pause();
#else
    std::this_thread::yield();
#endif
}

int env_int(const char * name, int def) {
    const char * v = getenv(name);
    return v && v[0] ? atoi(v) : def;
}

std::atomic<bool> g_bridge_owned{false};

// [TAG_FN_L3_CPU_STATS] page faults of this process so far (soft and hard)
uint64_t br_page_faults() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    memset(&pmc, 0, sizeof(pmc));
    pmc.cb = sizeof(pmc);
    return GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)) ? (uint64_t) pmc.PageFaultCount : 0;
#else
    struct rusage ru;
    memset(&ru, 0, sizeof(ru));
    return getrusage(RUSAGE_SELF, &ru) == 0 ? (uint64_t) ru.ru_minflt + (uint64_t) ru.ru_majflt : 0;
#endif
}

} // namespace

struct llama_moe_bridge {
    // backend (device) side, ggml-moe-bridge.h
    ggml_backend_moe_bridge_new_t        fn_new        = nullptr;
    ggml_backend_moe_bridge_free_t       fn_free       = nullptr;
    ggml_backend_moe_bridge_id_t         fn_id         = nullptr;
    ggml_backend_moe_bridge_set_runner_t fn_set_runner = nullptr;
    ggml_backend_moe_bridge_poll_t       fn_poll       = nullptr;
    ggml_backend_moe_bridge_complete_t   fn_complete   = nullptr;
    ggml_backend_moe_bridge_error_t      fn_error      = nullptr;
    ggml_backend_moe_bridge_reset_t      fn_reset      = nullptr;
    ggml_backend_moe_bridge_get_stats_t  fn_get_stats  = nullptr;
    // [TAG_FN_R4_BRIDGE_DMA]
    ggml_backend_moe_bridge_set_ring_t     fn_set_ring     = nullptr;
    ggml_backend_moe_bridge_publish_plan_t fn_publish_plan = nullptr;
    ggml_backend_moe_bridge_chan_times_t   fn_chan_times   = nullptr;
    ggml_backend_moe_bridge_release_t      fn_release      = nullptr;

    // CPU MoE pool, ggml-cpu.h
    pool_params_default_t pool_params_default = nullptr;
    pool_new_t            pool_new            = nullptr;
    pool_free_t           pool_free           = nullptr;
    pool_run_t            pool_run            = nullptr;
    pool_park_t           pool_park           = nullptr;
    pool_park_t           pool_wake           = nullptr;

    // [TAG_FN_R2_BRIDGE_PF] next-layer prefetch (LLAMA_MOE_BRIDGE_PF=1, spin mode): after a layer's job the pool
    // predicts the next layer's experts from its router and pulls the cold ones into the CPU caches, while the device
    // runs that layer's attention; the next job stops it
    pool_pf_t                             pool_pf       = nullptr;
    pool_pf_stop_t                        pool_pf_stop  = nullptr;
    pool_pf_stats_t                       pool_pf_stats = nullptr;
    bool                                  pf_on         = false;
    int                                   pf_k          = 12;
    int                                   pf_mode       = 0; // LLAMA_MOE_BRIDGE_PF_MODE: 0 real loads, 1 prefetches
    bool                                  pf_solo       = false; // LLAMA_MOE_BRIDGE_PF_SOLO: the executor only dispatches
    pool_set_solo_t                       pool_set_solo = nullptr;
    std::vector<int>                      pf_next;   // per channel: the channel of layer il + 1, or -1
    std::vector<std::vector<ggml_fp16_t>> pf_router; // per channel: its router rows in f16 ([n_expert][n_embd]) if predicted

    // lever round 3, qwen4exp only ([TAG_FN_L3_CPU_*])
    bool                                pool_stats   = false; // LLAMA_MOE_POOL_STATS
    int                                 pool_split   = 0;     // LLAMA_MOE_POOL_SPLIT
    int                                 pool_swpf    = 0;     // LLAMA_MOE_POOL_SWPF
    bool                                pf_rank      = false; // LLAMA_MOE_BRIDGE_PF_RANK
    int                                 exec_cpu     = -1;    // [TAG_FN_L3_CPU_PLACE] LLAMA_MOE_POOL_EXEC_CPU
    int                                 pf_streams   = 1;     // [TAG_FN_L3_CPU_PFSTREAMS] LLAMA_MOE_POOL_PF_STREAMS
    bool                                fill_gap     = false; // [TAG_FN_L3_CPU_FILL] LLAMA_MOE_DMA_FILL_GAP (with the DMA share)
    uint64_t                            fill_mask    = 0;     // [TAG_FN_L3_CPU_FILL] LLAMA_MOE_DMA_FILL_CPUS
    int                                 hint_k       = 0;     // [TAG_FN_L3_CPU_DEVPRED] LLAMA_MOE_BRIDGE_PF_DEV: ids per token
    int                                 hint_mode    = 0;     // [TAG_FN_L3_CPU_DEVPRED] 1: this layer's FFN input, 2: the next layer's FFN mixer
    int                                 hint_wait_us = 50;
    ggml_backend_moe_bridge_read_hint_t fn_read_hint = nullptr;
    pool_get_stats_t                    pool_get_stats = nullptr;
    std::atomic<uint64_t>               disp_ns{0};        // [TAG_FN_L3_CPU_STATS] executor: job taken -> pool run
    std::atomic<uint64_t>               n_disp{0};
    std::atomic<uint64_t>               n_hint_ok{0};
    std::atomic<uint64_t>               n_hint_miss{0};
    uint64_t                            st_faults = 0;     // owner thread: page faults and jobs at the last stats line
    uint64_t                            st_jobs   = 0;

    ggml_moe_bridge * gb  = nullptr;
    int32_t           bid = -1;

    int     mode       = GGML_MOE_BRIDGE_WAIT_SPIN;
    int     max_t      = 8;
    bool    sync_on    = true; // [TAG_FN_R2_BRIDGE_SYNC] LLAMA_MOE_BRIDGE_SYNC
    int     n_used     = 0;
    int64_t n_embd     = 0;
    int     spin_us    = 2000;
    int     timeout_ms = 500; // [TAG_FN_R1_BRIDGE_RETRY] was 50: a cold start missed it
    int     job_max_ms = 1000;
    bool    stats      = false;
    int     max_fetch  = 0;     // [TAG_FN_R4_BRIDGE_DMA] experts per fetch the device side was made for (0: no fetch)
    bool    dma_on     = false; // [TAG_FN_R4_BRIDGE_DMA] the ring is registered: graphs may fetch

    struct chan {
        int il = -1;
        const ggml_tensor * up = nullptr;
        std::string gate_name;
        ggml_cpu_moe_layer layer     = {};  // every expert on the host
        ggml_cpu_moe_layer layer_tbl = {};  // the hot experts of the moe-cache table skipped
        bool tbl_looked = false;
        bool tbl_ok     = false;

        // [TAG_FN_R4_BRIDGE_DMA] the DMA share of the layer: the DMA state's handle, the skip table of a job (the hot
        // experts and the fetched ones), and the last job's host time and expert count (for the split)
        void *               dma = nullptr;
        std::vector<int32_t> skip;
        ggml_cpu_moe_layer   layer_dma = {};
        uint64_t             job_ns = 0;
        uint32_t             job_n_cpu = 0;
        bool                 job_fetch = false; // the job had a plan (its device fetch times are its own)
        uint32_t             job_seq = 0;       // written last (release fence), read first (acquire fence)
        std::vector<uint8_t> seen;              // [TAG_FN_R4_REVIEW] [n_expert] scratch of the distinct CPU expert count
    };
    std::vector<chan> chans;
    std::unordered_map<const ggml_tensor *, int> by_up;

    ggml_cpu_moe_pool *      pool = nullptr;
    ggml_cpu_moe_pool_params pool_params = {};

    // executor thread (spin mode)
    std::thread             exec;
    std::mutex              mtx;
    std::condition_variable cv;
    std::atomic<bool>       stop{false};
    std::atomic<bool>       parked{false};
    std::atomic<int>        n_sleeping{0};
    alignas(64) std::atomic<uint32_t> wake_gen{0};
    bool                    ready = false; // guarded by mtx

    // the owner's bridged graph runs (begin .. end). A job that comes later is stale: its wait gave up, the owner has
    // moved on and may run the moe-cache step or a CPU split that calls the same routing observer. [TAG_MOE_BRIDGE]
    std::mutex graph_mtx;
    bool       in_graph = false; // guarded by graph_mtx

    // state, owning context's thread
    bool     active     = true;
    bool     disabled   = false;
    int      n_errors   = 0;
    int      pause_left = 0;
    uint64_t n_graphs   = 0;
    uint64_t err_graph  = 0; // [TAG_FN_R1_BRIDGE_RB] n_graphs at the last error

    // job statistics (runner thread)
    std::atomic<uint64_t> n_jobs{0};
    std::atomic<uint64_t> job_ns{0};
    std::atomic<uint64_t> job_ns_max{0};

    int stall_ms    = 0;
    int stall_every = 0;
};

// the job on the CPU MoE pool: runs on the executor thread (spin) or the driver's callback thread (hostfunc)
static bool br_run(const ggml_moe_bridge_job * j, void * ud) {
    auto * br = (llama_moe_bridge *) ud;
    const auto t_take = std::chrono::steady_clock::now();
    if (j->chan < 0 || j->chan >= (int) br->chans.size() || j->n_embd != br->n_embd) {
        return false;
    }
    auto & c = br->chans[j->chan];
    const ggml_cpu_moe_layer * layer = &c.layer;
    if (j->flags & GGML_MOE_BRIDGE_JOB_TABLE) {
        // the graph built a device chain over the moe-cache slots of this layer: skip the experts it serves
        if (!c.tbl_looked) {
            c.tbl_looked = true;
            const llama_moe_cache_layer * mc = llama_moe_cache_lookup(c.up);
            if (mc && mc->host_table && mc->host_table->data) {
                c.layer_tbl            = c.layer;
                c.layer_tbl.table      = (const int32_t *) mc->host_table->data;
                c.layer_tbl.table_miss = mc->n_slots;
                c.tbl_ok               = true;
            }
        }
        if (!c.tbl_ok) {
            return false;
        }
        layer = &c.layer_tbl;
    }

    // routing observation, as the CPU MUL_MAT_ID of ffn_gate_exps reports it (moe-cache LRU and hot statistics). Under
    // graph_mtx: end() takes it too, so no observer call of this job runs once the owner's graph is closed.
    // [TAG_FN_R4_REVIEW] The DMA plan runs under it as well: the plan writes the DMA state's per-layer scratch and its
    // admission lists (cur_ids, cur_seen) that the owner's step boundary (dma_step_impl) swaps and clears. A stale job
    // whose wait timed out could otherwise still be planning after end() returned, while the next step rolls the lists.
    {
        std::lock_guard<std::mutex> lk(br->graph_mtx);
        if (!br->in_graph) {
            return false; // stale: zeros and no observer call; the bridge error of its wait is already set
        }
        void * obs_ud = nullptr;
        ggml_moe_obs_cb_t obs = ggml_get_moe_obs_callback(&obs_ud);
        if (obs) {
            ggml_tensor t;
            memset(&t, 0, sizeof(t));
            t.type  = GGML_TYPE_I32;
            t.ne[0] = j->n_used;
            t.ne[1] = j->n_tokens;
            t.ne[2] = 1;
            t.ne[3] = 1;
            t.nb[0] = sizeof(int32_t);
            t.nb[1] = sizeof(int32_t)*j->n_used;
            t.nb[2] = t.nb[1]*j->n_tokens;
            t.nb[3] = t.nb[2];
            t.data  = (void *) j->ids;
            obs(c.gate_name.c_str(), &t, obs_ud);
        }

        br->n_jobs.fetch_add(1, std::memory_order_relaxed);

        // [TAG_FN_R4_BRIDGE_DMA] the graph fetches a share of this job: plan it (ring-ready cold experts), publish the plan
        // for the device (mapped memory, no CUDA call), then the CPU skips the hot experts and the fetched ones. The plan
        // is published even when it fetches nothing, so the device's fetch never waits for nothing.
        if ((j->flags & GGML_MOE_BRIDGE_JOB_DMA) && j->plan) {
            if (!c.dma || !br->dma_on || j->n_used*j->n_tokens > br->max_t*br->n_used) {
                return false; // the graph asked for a fetch this bridge cannot plan: fail the job (and with it the ubatch)
            }
            const int64_t n_exp = c.up->ne[2];
            if (c.skip.size() != (size_t) n_exp) {
                c.skip.assign(n_exp, 0);
            }
            const int32_t * hot_tbl  = layer->table;
            const int32_t   hot_miss = layer->table_miss;
            int32_t d_exp[GGML_MOE_BRIDGE_MAX_FETCH];
            const int n_d = llama_moe_dma_bridge_plan(c.dma, j->ids, j->n_used, j->n_tokens, hot_tbl, hot_miss,
                    std::min(br->max_fetch, GGML_MOE_BRIDGE_MAX_FETCH), j->plan->off, j->plan->slot, j->plan->slot_ids, d_exp);
            if (n_d < 0) {
                return false;
            }
            j->plan->n_copy = n_d;
            br->fn_publish_plan(br->gb, j);

            // the CPU computes an expert iff its skip entry is the miss value 0: hot experts and the fetched ones are 1
            if (hot_tbl) {
                for (int64_t e = 0; e < n_exp; ++e) {
                    c.skip[e] = hot_tbl[e] != hot_miss ? 1 : 0;
                }
            } else {
                std::fill(c.skip.begin(), c.skip.end(), 0);
            }
            for (int i = 0; i < n_d; ++i) {
                c.skip[d_exp[i]] = 1;
            }
            c.layer_dma            = c.layer;
            c.layer_dma.table      = c.skip.data();
            c.layer_dma.table_miss = 0;
            layer = &c.layer_dma;
        }
    }

    const auto t0 = std::chrono::steady_clock::now();
    if (br->pool_stats) { // [TAG_FN_L3_CPU_STATS] observer + plan, the work before the pool starts
        br->disp_ns.fetch_add((uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(t0 - t_take).count(), std::memory_order_relaxed);
        br->n_disp.fetch_add(1, std::memory_order_relaxed);
    }
    const ggml_cpu_moe_job job = { layer, j->n_tokens, j->n_used, j->x, j->ids, j->w, j->out };
    const bool hold = br->fill_gap && br->dma_on; // [TAG_FN_L3_CPU_FILL] the ring fills wait while the pool reads DRAM
    if (hold) {
        llama_moe_dma_bridge_hold(true);
    }
    const bool ok = br->pool_run(br->pool, &job) == GGML_STATUS_SUCCESS;
    if (hold) {
        llama_moe_dma_bridge_hold(false);
    }
    const uint64_t ns = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    if (br->stats) {
        br->job_ns.fetch_add(ns, std::memory_order_relaxed);
        uint64_t m = br->job_ns_max.load(std::memory_order_relaxed);
        while (ns > m && !br->job_ns_max.compare_exchange_weak(m, ns, std::memory_order_relaxed)) {
        }
    }
    if (br->dma_on) {
        // [TAG_FN_R4_BRIDGE_DMA] the job's CPU time and its distinct CPU experts, read by end() after the graph synced
        // [TAG_FN_R4_REVIEW] one pass with a per-channel mark (the pairwise scan was O(n^2) in n_used x n_tokens, up to
        // 512 x 512 for a bridged 64-token ubatch, and it runs before complete(): the device's wait waited for it too)
        uint32_t n_cpu = 0;
        const int     n     = j->n_used*j->n_tokens;
        const int64_t n_exp = c.up->ne[2];
        if (c.seen.size() != (size_t) n_exp) {
            c.seen.assign(n_exp, 0);
        }
        for (int a = 0; a < n; ++a) {
            const int32_t e = j->ids[a];
            if (e < 0 || e >= n_exp || c.seen[e] || (layer->table && layer->table[e] != layer->table_miss)) {
                continue;
            }
            c.seen[e] = 1;
            n_cpu++;
        }
        for (int a = 0; a < n; ++a) {
            const int32_t e = j->ids[a];
            if (e >= 0 && e < n_exp) {
                c.seen[e] = 0;
            }
        }
        c.job_ns    = ns;
        c.job_n_cpu = n_cpu;
        c.job_fetch = (j->flags & GGML_MOE_BRIDGE_JOB_DMA) && j->plan;
        std::atomic_thread_fence(std::memory_order_release);
        c.job_seq   = j->seq;
    }
    return ok;
}

// [TAG_FN_R2_BRIDGE_PF] after the job of channel j.chan: predict the next layer's experts from its router on this
// layer's input (still in the channel's mapped memory: the device posts to this channel again only in the next step)
// and let the pool's workers pull the cold ones into the caches until the next job comes (ggml_cpu_moe_run stops them)
static void br_prefetch_next(llama_moe_bridge * br, ggml_cpu_moe_pool * pool, const ggml_moe_bridge_job & j) {
    if (j.chan < 0 || j.chan >= (int) br->pf_next.size() || j.x == nullptr || j.n_tokens < 1 || j.n_tokens > br->max_t) {
        return;
    }
    const int nc = br->pf_next[j.chan];
    if (nc < 0 || (br->hint_k <= 0 && br->pf_router[nc].empty())) {
        return;
    }
    {
        std::lock_guard<std::mutex> lk(br->graph_mtx);
        if (!br->in_graph) {
            return; // the owner closed the graph (a stale job): a CPU split may run next
        }
    }
    auto & cn = br->chans[nc];
    // the table the next job takes: the hot set's when its layer has one (br_run sets it up at the layer's first job)
    const ggml_cpu_moe_layer * lay = cn.tbl_ok ? &cn.layer_tbl : &cn.layer;
    if (br->hint_k > 0) {
        // [TAG_FN_L3_CPU_DEVPRED] the device's top-k of the next router, written right after this job's post; a job
        // that took less time than the hint kernel waits for it a little
        if (!(j.flags & GGML_MOE_BRIDGE_JOB_HINT)) {
            return;
        }
        int32_t ids[GGML_MOE_BRIDGE_MAX_HINT_K*16];
        int k = 0;
        int t = 0;
        bool ok = br->fn_read_hint(br->gb, j.chan, j.seq, ids, (int) (sizeof(ids)/sizeof(ids[0])), &k, &t);
        if (!ok && br->hint_wait_us > 0) {
            const auto t0 = std::chrono::steady_clock::now();
            while (!ok && std::chrono::steady_clock::now() - t0 < std::chrono::microseconds(br->hint_wait_us)) {
                br_relax();
                ok = br->fn_read_hint(br->gb, j.chan, j.seq, ids, (int) (sizeof(ids)/sizeof(ids[0])), &k, &t);
            }
        }
        if (!ok || t != j.n_tokens) {
            br->n_hint_miss.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        br->n_hint_ok.fetch_add(1, std::memory_order_relaxed);
        ggml_cpu_moe_prefetch_job pj = {};
        pj.layer    = lay;
        pj.n_tokens = t;
        pj.k        = k;
        pj.mode     = br->pf_mode;
        pj.list     = ids;
        br->pool_pf(pool, &pj);
        return;
    }
    const ggml_cpu_moe_prefetch_job pj = { lay, br->pf_router[nc].data(), j.n_tokens, j.x, br->pf_k, br->pf_mode, nullptr };
    br->pool_pf(pool, &pj);
}

// spin mode: poll the posts in order, spin for spin_us after the last job, then sleep in 1 ms slices (so a post that
// comes without a wake is still served) until the owning context wakes it
static void br_exec_main(llama_moe_bridge * br) {
    ggml_cpu_moe_pool_params pp = br->pool_params;
    pp.pin_caller = true;
    ggml_cpu_moe_pool * pool = br->pool_new(&pp);
    if (pool && br->pf_on && br->pf_solo) {
        br->pool_set_solo(pool, true); // [TAG_FN_R2_BRIDGE_PF]
    }
    {
        std::lock_guard<std::mutex> lk(br->mtx);
        br->pool  = pool;
        br->ready = true;
    }
    br->cv.notify_all();
    if (pool == nullptr) {
        return;
    }

    const auto spin = std::chrono::microseconds(br->spin_us);
    auto     last    = std::chrono::steady_clock::now();
    uint32_t wg_seen = br->wake_gen.load(std::memory_order_relaxed);
    uint64_t n_stalled = 0; // job count of the last test stall (a stale job does not count, so n can repeat)
    for (uint32_t k = 1; ; ++k) {
        ggml_moe_bridge_job job;
        if (br->fn_poll(br->gb, &job)) {
            const bool ok = br_run(&job, br);
            br->fn_complete(br->gb, &job, ok);
            if (br->pf_on && ok) {
                br_prefetch_next(br, pool, job); // [TAG_FN_R2_BRIDGE_PF]
            }
            last = std::chrono::steady_clock::now();
            // test: stall before the next job is taken, so its wait runs into timeout_ms
            const uint64_t n = br->n_jobs.load(std::memory_order_relaxed);
            if (br->stall_ms > 0 && n != n_stalled && (n == 199 || (br->stall_every > 0 && n > 199 && (n - 199) % br->stall_every == 0))) {
                n_stalled = n;
                LLAMA_LOG_WARN("%s: LLAMA_MOE_BRIDGE_TEST_STALL: the executor sleeps %d ms after job %" PRIu64 "\n", __func__, br->stall_ms, n);
                std::this_thread::sleep_for(std::chrono::milliseconds(br->stall_ms));
            }
            continue;
        }
        if (br->stop.load(std::memory_order_acquire)) {
            break;
        }
        br_relax();
        if ((k & 63) != 0) {
            continue;
        }
        const uint32_t wg = br->wake_gen.load(std::memory_order_relaxed);
        if (wg != wg_seen) {
            wg_seen = wg;
            last    = std::chrono::steady_clock::now();
        }
        if (!br->parked.load(std::memory_order_relaxed) && std::chrono::steady_clock::now() - last < spin) {
            continue;
        }
        std::unique_lock<std::mutex> lk(br->mtx);
        const uint32_t wg_sleep = br->wake_gen.load(std::memory_order_seq_cst);
        br->n_sleeping.fetch_add(1, std::memory_order_seq_cst);
        br->cv.wait_for(lk, std::chrono::milliseconds(1), [&] {
            return br->stop.load(std::memory_order_seq_cst) || br->wake_gen.load(std::memory_order_seq_cst) != wg_sleep;
        });
        br->n_sleeping.fetch_sub(1, std::memory_order_relaxed);
    }

    {
        std::lock_guard<std::mutex> lk(br->mtx);
        br->pool = nullptr;
    }
    br->pool_free(pool);
}

static void br_destroy(llama_moe_bridge * br) {
    if (br == nullptr) {
        return;
    }
    if (br->gb && br->fn_release) {
        br->fn_release(br->gb); // [TAG_FN_R4_BRIDGE_DMA] no device wait or fetch of this bridge keeps spinning
    }
    if (br->exec.joinable()) { // spin mode: the executor frees its pool
        br->stop.store(true, std::memory_order_seq_cst);
        {
            std::lock_guard<std::mutex> lk(br->mtx);
        }
        br->cv.notify_all();
        br->exec.join();
    }
    if (br->gb) { // waits for host function nodes still queued (hostfunc mode) before their pool goes
        br->fn_free(br->gb);
        br->gb = nullptr;
    }
    if (br->pool) {
        br->pool_free(br->pool);
        br->pool = nullptr;
    }
    delete br;
    g_bridge_owned.store(false);
}

// [TAG_FN_R2_BRIDGE_NOSPEC] the layer test of llama_moe_bridge_create without the CPU pool: routed experts in host memory,
// the router on a device
bool llama_moe_bridge_wanted(const llama_model & model) {
    if (env_int("LLAMA_MOE_BRIDGE", 0) <= 0) {
        return false;
    }
    const auto & hparams = model.hparams;
    const bool nextn_split = hparams.n_layer_nextn > 0 && hparams.n_layer() > 0 && hparams.router_layer < 0;
    for (size_t il = 0; il < model.layers.size(); ++il) {
        const auto & l = model.layers[il];
        if (nextn_split && il >= hparams.n_layer()) {
            continue;
        }
        if (!l.ffn_up_exps || !l.ffn_gate_exps || !l.ffn_down_exps || !l.ffn_gate_inp || l.ffn_gate_up_exps) {
            continue;
        }
        const ggml_tensor * ws[3] = { l.ffn_up_exps, l.ffn_gate_exps, l.ffn_down_exps };
        bool host = true;
        for (const ggml_tensor * w : ws) {
            host = host && w->data && w->buffer && ggml_backend_buffer_is_host(w->buffer) && w->extra == nullptr;
        }
        if (host && l.ffn_gate_inp->buffer && !ggml_backend_buffer_is_host(l.ffn_gate_inp->buffer)) {
            return true;
        }
    }
    return false;
}

llama_moe_bridge * llama_moe_bridge_create(const llama_model & model, int n_threads, int max_t_cap) {
    if (env_int("LLAMA_MOE_BRIDGE", 0) <= 0) {
        return nullptr;
    }
    if (g_bridge_owned.exchange(true)) {
        LLAMA_LOG_INFO("%s: another context owns the MoE bridge, this one runs without it\n", __func__);
        return nullptr;
    }

    auto * br = new llama_moe_bridge();
    auto give_up = [&](const char * why) -> llama_moe_bridge * {
        LLAMA_LOG_WARN("%s: LLAMA_MOE_BRIDGE=1 but %s: running without the bridge\n", __func__, why);
        br_destroy(br);
        return nullptr;
    };

    // CPU MoE pool
    ggml_backend_reg_t cpu_reg = ggml_backend_reg_by_name("CPU");
    if (cpu_reg == nullptr) {
        return give_up("no CPU backend");
    }
    br->pool_params_default = (pool_params_default_t) ggml_backend_reg_get_proc_address(cpu_reg, "ggml_cpu_moe_pool_params_default");
    br->pool_new            = (pool_new_t)            ggml_backend_reg_get_proc_address(cpu_reg, "ggml_cpu_moe_pool_new");
    br->pool_free           = (pool_free_t)           ggml_backend_reg_get_proc_address(cpu_reg, "ggml_cpu_moe_pool_free");
    br->pool_run            = (pool_run_t)            ggml_backend_reg_get_proc_address(cpu_reg, "ggml_cpu_moe_run");
    br->pool_park           = (pool_park_t)           ggml_backend_reg_get_proc_address(cpu_reg, "ggml_cpu_moe_pool_park");
    br->pool_wake           = (pool_park_t)           ggml_backend_reg_get_proc_address(cpu_reg, "ggml_cpu_moe_pool_wake");
    auto layer_supported    = (layer_supported_t)     ggml_backend_reg_get_proc_address(cpu_reg, "ggml_cpu_moe_layer_supported");
    if (!br->pool_params_default || !br->pool_new || !br->pool_free || !br->pool_run || !br->pool_park || !br->pool_wake || !layer_supported) {
        return give_up("the CPU backend has no MoE pool");
    }

    // the host-resident expert layers whose router lives on one device
    const auto & hparams = model.hparams;
    ggml_backend_dev_t dev = nullptr;
    int n_skipped = 0;
    // nextn (MTP) layers run in an MTP context's graphs only, never in this one's (as the KV filter of create_memory)
    // [TAG_MOE_BRIDGE]
    const bool nextn_split = hparams.n_layer_nextn > 0 && hparams.n_layer() > 0 && hparams.router_layer < 0;
    for (size_t il = 0; il < model.layers.size(); ++il) {
        const auto & l = model.layers[il];
        if (nextn_split && il >= hparams.n_layer()) {
            continue;
        }
        if (!l.ffn_up_exps || !l.ffn_gate_exps || !l.ffn_down_exps || !l.ffn_gate_inp || l.ffn_gate_up_exps) {
            continue;
        }
        if (l.ffn_up_exps_b || l.ffn_gate_exps_b || l.ffn_down_exps_b || l.ffn_up_exps_s || l.ffn_gate_exps_s || l.ffn_down_exps_s) {
            continue;
        }
        const ggml_tensor * ws[3] = { l.ffn_up_exps, l.ffn_gate_exps, l.ffn_down_exps };
        bool host = true;
        for (const ggml_tensor * w : ws) {
            // extra != NULL: a CPU extra buffer (repack, AMX, ...) keeps its own layout, which the pool cannot read
            host = host && w->data && w->buffer && ggml_backend_buffer_is_host(w->buffer) && w->extra == nullptr;
        }
        if (!host || !l.ffn_gate_inp->buffer || ggml_backend_buffer_is_host(l.ffn_gate_inp->buffer)) {
            continue;
        }
        ggml_backend_dev_t d = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(l.ffn_gate_inp->buffer));
        if (d == nullptr || (dev != nullptr && d != dev)) {
            n_skipped++;
            continue;
        }
        const ggml_cpu_moe_layer cl = { l.ffn_up_exps, l.ffn_gate_exps, l.ffn_down_exps, nullptr, 0 };
        if (!layer_supported(&cl)) {
            LLAMA_LOG_WARN("%s: layer %zu: the CPU MoE pool does not take up %s / gate %s / down %s, the layer keeps its CPU split\n",
                    __func__, il, ggml_type_name(cl.up->type), ggml_type_name(cl.gate->type), ggml_type_name(cl.down->type));
            n_skipped++;
            continue;
        }
        dev = d;
        llama_moe_bridge::chan c;
        c.il        = (int) il;
        c.up        = l.ffn_up_exps;
        c.gate_name = l.ffn_gate_exps->name;
        c.layer     = cl;
        br->by_up[l.ffn_up_exps] = (int) br->chans.size();
        br->chans.push_back(c);
        br->n_used = std::max<int>(br->n_used, (int) hparams.n_expert_used((uint32_t) il));
        br->n_embd = l.ffn_up_exps->ne[0];
    }
    if (br->chans.empty()) {
        return give_up("no host-resident expert layer with a device router was found");
    }

    // the device backend must provide the bridge
    ggml_backend_reg_t dev_reg = ggml_backend_dev_backend_reg(dev);
    auto proc = [&](const char * name) { return ggml_backend_reg_get_proc_address(dev_reg, name); };
    br->fn_new        = (ggml_backend_moe_bridge_new_t)        proc("ggml_backend_moe_bridge_new");
    br->fn_free       = (ggml_backend_moe_bridge_free_t)       proc("ggml_backend_moe_bridge_free");
    br->fn_id         = (ggml_backend_moe_bridge_id_t)         proc("ggml_backend_moe_bridge_id");
    br->fn_set_runner = (ggml_backend_moe_bridge_set_runner_t) proc("ggml_backend_moe_bridge_set_runner");
    br->fn_poll       = (ggml_backend_moe_bridge_poll_t)       proc("ggml_backend_moe_bridge_poll");
    br->fn_complete   = (ggml_backend_moe_bridge_complete_t)   proc("ggml_backend_moe_bridge_complete");
    br->fn_error      = (ggml_backend_moe_bridge_error_t)      proc("ggml_backend_moe_bridge_error");
    br->fn_reset      = (ggml_backend_moe_bridge_reset_t)      proc("ggml_backend_moe_bridge_reset");
    br->fn_get_stats  = (ggml_backend_moe_bridge_get_stats_t)  proc("ggml_backend_moe_bridge_get_stats");
    // [TAG_FN_R4_BRIDGE_DMA] optional: without them the bridge never fetches
    br->fn_set_ring     = (ggml_backend_moe_bridge_set_ring_t)     proc("ggml_backend_moe_bridge_set_ring");
    br->fn_publish_plan = (ggml_backend_moe_bridge_publish_plan_t) proc("ggml_backend_moe_bridge_publish_plan");
    br->fn_chan_times   = (ggml_backend_moe_bridge_chan_times_t)   proc("ggml_backend_moe_bridge_chan_times");
    br->fn_release      = (ggml_backend_moe_bridge_release_t)      proc("ggml_backend_moe_bridge_release");
    if (!br->fn_new || !br->fn_free || !br->fn_id || !br->fn_set_runner || !br->fn_poll || !br->fn_complete ||
        !br->fn_error || !br->fn_reset || !br->fn_get_stats) {
        br->fn_free = nullptr;
        return give_up("the expert router's device backend has no MoE bridge");
    }
    int dev_index = -1;
    for (size_t i = 0; i < ggml_backend_reg_dev_count(dev_reg); ++i) {
        if (ggml_backend_reg_dev_get(dev_reg, i) == dev) {
            dev_index = (int) i;
        }
    }
    if (dev_index < 0) {
        return give_up("the router's device index is unknown");
    }

    // switches
    const char * wm = getenv("LLAMA_MOE_BRIDGE_WAIT");
    br->mode       = wm && strcmp(wm, "hostfunc") == 0 ? GGML_MOE_BRIDGE_WAIT_HOSTFUNC : GGML_MOE_BRIDGE_WAIT_SPIN;
    br->max_t      = std::min(16, std::max(1, env_int("LLAMA_MOE_BRIDGE_MAX_T", 8)));
    br->sync_on    = env_int("LLAMA_MOE_BRIDGE_SYNC", 1) > 0; // [TAG_FN_R2_BRIDGE_SYNC]
    if (max_t_cap > 0 && br->max_t > max_t_cap) {
        // [TAG_FN_R1_BRIDGE_RB] wider graphs (a prompt's last few tokens) run the plain CPU split
        br->max_t = max_t_cap;
    }
    br->spin_us    = std::max(0, env_int("LLAMA_MOE_BRIDGE_SPIN_US", 2000));
    br->timeout_ms = std::min(1200, std::max(1, env_int("LLAMA_MOE_BRIDGE_TIMEOUT_MS", 500))); // [TAG_FN_R1_BRIDGE_RETRY]
    br->job_max_ms = std::min(1200, std::max(br->timeout_ms, env_int("LLAMA_MOE_BRIDGE_JOB_MAX_MS", 1000)));
    br->stats      = env_int("LLAMA_MOE_BRIDGE_STATS", 0) > 0;
    br->stall_ms    = std::max(0, env_int("LLAMA_MOE_BRIDGE_TEST_STALL", 0));
    br->stall_every = std::max(0, env_int("LLAMA_MOE_BRIDGE_TEST_STALL_EVERY", 0));

    // lever round 3, CPU side [TAG_FN_L3_CPU_*]: qwen4exp only
    {
        const char * sp = getenv("LLAMA_MOE_POOL_SPLIT");
        int split = 0;
        if (sp && sp[0]) {
            split = strcmp(sp, "steal") == 0 ? 2 : strcmp(sp, "stable") == 0 ? 1 : strcmp(sp, "range") == 0 ? 0 : atoi(sp);
        }
        split = std::min(2, std::max(0, split));
        const bool stats = env_int("LLAMA_MOE_POOL_STATS", 0) > 0;
        const int  swpf  = std::min(64, std::max(0, env_int("LLAMA_MOE_POOL_SWPF", 0)));
        const bool rank  = env_int("LLAMA_MOE_BRIDGE_PF_RANK", 0) > 0;
        const int  dev   = std::min(2, std::max(0, env_int("LLAMA_MOE_BRIDGE_PF_DEV", 0)));
        const int  exec  = env_int("LLAMA_MOE_POOL_EXEC_CPU", -1);
        const int  pfs   = env_int("LLAMA_MOE_POOL_PF_STREAMS", 1);
        const bool fgap  = env_int("LLAMA_MOE_DMA_FILL_GAP", 0) > 0;
        const char * fm  = getenv("LLAMA_MOE_DMA_FILL_CPUS");
        if (model.arch == LLM_ARCH_QWEN4EXP) {
            br->pool_split = split;
            br->pool_swpf  = swpf;
            br->pool_stats = stats;
            br->pf_rank    = rank;
            br->hint_k     = dev > 0 ? 1 : 0; // the count is set with the prefetch below
            br->hint_mode  = dev;
            br->exec_cpu   = exec >= 0 && exec < GGML_MAX_N_THREADS ? exec : -1;
            br->pf_streams = std::min(8, std::max(1, pfs));
            br->fill_gap   = fgap;
            br->fill_mask  = fm && fm[0] ? strtoull(fm, nullptr, 16) : 0;
        } else if (split != 0 || swpf != 0 || stats || rank || dev > 0 || exec >= 0 || pfs > 1 || fgap || (fm && fm[0])) {
            LLAMA_LOG_WARN("%s: LLAMA_MOE_POOL_SPLIT / _SWPF / _STATS / _EXEC_CPU / _PF_STREAMS, LLAMA_MOE_BRIDGE_PF_RANK / _PF_DEV and "
                    "LLAMA_MOE_DMA_FILL_GAP / _CPUS act on qwen4exp only: ignored\n", __func__);
        }
        br->hint_wait_us = std::min(1000, std::max(0, env_int("LLAMA_MOE_BRIDGE_PF_DEV_WAIT_US", 50)));
    }

    const int n_pool = std::max(1, env_int("LLAMA_MOE_BRIDGE_THREADS", n_threads > 0 ? n_threads : 1));
    br->pool_params = br->pool_params_default(std::min(n_pool, GGML_MAX_N_THREADS));
    br->pool_params.prio    = std::min(3, std::max(0, env_int("LLAMA_MOE_BRIDGE_PRIO", GGML_SCHED_PRIO_HIGH)));
    br->pool_params.spin_us = br->spin_us;
    br->pool_params.split   = br->pool_split; // [TAG_FN_L3_CPU_SPLIT]
    br->pool_params.swpf    = br->pool_swpf;  // [TAG_FN_L3_CPU_SWPF]
    br->pool_params.stats   = br->pool_stats; // [TAG_FN_L3_CPU_STATS]
    br->pool_params.pf_rank = br->pf_rank;    // [TAG_FN_L3_CPU_PFRANK]
    br->pool_params.caller_cpu1 = br->exec_cpu + 1; // [TAG_FN_L3_CPU_PLACE] 0: the first CPU of the list, as before
    br->pool_params.pf_streams  = br->pf_streams;   // [TAG_FN_L3_CPU_PFSTREAMS]
    if (br->pool_stats) {
        br->pool_get_stats = (pool_get_stats_t) ggml_backend_reg_get_proc_address(cpu_reg, "ggml_cpu_moe_pool_get_stats");
        if (br->pool_get_stats == nullptr) {
            LLAMA_LOG_WARN("%s: LLAMA_MOE_POOL_STATS: the CPU backend has no pool statistics\n", __func__);
            br->pool_stats = false;
            br->pool_params.stats = false;
        }
    }
    // default: one thread per physical core except the first, where the ggml threadpool pins the main thread (it spins
    // in the device synchronize while the step runs; a pool thread there would share its core)
    br->pool_params.skip_first_core = true;
    if (const char * m = getenv("LLAMA_MOE_BRIDGE_CPUMASK"); m && m[0]) {
        br->pool_params.skip_first_core = false;
        const char * h = (m[0] == '0' && (m[1] == 'x' || m[1] == 'X')) ? m + 2 : m;
        const size_t len = strlen(h);
        for (size_t i = 0; i < len; ++i) {
            const char ch = h[len - 1 - i];
            const int v = ch >= '0' && ch <= '9' ? ch - '0' : ch >= 'a' && ch <= 'f' ? ch - 'a' + 10 : ch >= 'A' && ch <= 'F' ? ch - 'A' + 10 : 0;
            for (int b = 0; b < 4; ++b) {
                const size_t cpu = i*4 + b;
                if (cpu < GGML_MAX_N_THREADS) {
                    br->pool_params.cpumask[cpu] = (v >> b) & 1;
                }
            }
        }
    }

    ggml_moe_bridge_params bp = {};
    bp.device     = dev_index;
    bp.n_chan     = (int) br->chans.size();
    bp.n_embd     = br->n_embd;
    bp.n_used     = br->n_used;
    bp.max_tokens = br->max_t;
    bp.wait_mode  = br->mode;
    bp.timeout_ms = br->timeout_ms;
    bp.job_max_ms = br->job_max_ms;
    bp.stats      = br->stats;
    // [TAG_FN_R4_BRIDGE_DMA] LLAMA_MOE_BRIDGE_DMA=1 with a DMA share: the fetch side (plan areas, device scratch); the ring
    // is registered later (llama_moe_bridge_attach_dma). Spin mode only: a fetch spins on the plan the executor writes.
    if (env_int("LLAMA_MOE_BRIDGE_DMA", 0) > 0 && gen5::dma_requested() && br->fn_set_ring && br->fn_publish_plan &&
            br->fn_chan_times) {
        if (br->mode == GGML_MOE_BRIDGE_WAIT_SPIN) {
            br->max_fetch = std::min(GGML_MOE_BRIDGE_MAX_FETCH, std::max(1, env_int("LLAMA_MOE_DMA_SLOTS", 8)));
        } else {
            LLAMA_LOG_WARN("%s: LLAMA_MOE_BRIDGE_DMA needs LLAMA_MOE_BRIDGE_WAIT=spin: no DMA share in the bridged graphs\n", __func__);
        }
    }
    bp.max_fetch = br->max_fetch;
    // [TAG_FN_L3_CPU_DEVPRED] hints need the next-layer prefetch (spin mode) and a device side that reads them back
    if (br->hint_k > 0) {
        br->fn_read_hint = (ggml_backend_moe_bridge_read_hint_t) proc("ggml_backend_moe_bridge_read_hint");
        if (env_int("LLAMA_MOE_BRIDGE_PF", 0) <= 0 || br->mode != GGML_MOE_BRIDGE_WAIT_SPIN || br->fn_read_hint == nullptr) {
            LLAMA_LOG_WARN("%s: LLAMA_MOE_BRIDGE_PF_DEV needs LLAMA_MOE_BRIDGE_PF=1, spin waits and device hints: off\n", __func__);
            br->hint_k = 0;
        } else {
            br->hint_k = std::min(GGML_MOE_BRIDGE_MAX_HINT_K, std::max(1, std::min(64, env_int("LLAMA_MOE_BRIDGE_PF_K", 12))));
        }
    }
    bp.hint_k = br->hint_k;
    br->gb = br->fn_new(&bp);
    if (br->gb == nullptr) {
        return give_up("the device could not create the bridge");
    }
    br->bid = br->fn_id(br->gb);

    // [TAG_FN_R2_BRIDGE_PF] next-layer prefetch: the channels whose layer il + 1 is a channel too, and that layer's
    // router rows in f16 on the host (2.5 MiB per layer at n_embd 2560 x 512 experts)
    if (env_int("LLAMA_MOE_BRIDGE_PF", 0) > 0) {
        br->pool_pf       = (pool_pf_t)       ggml_backend_reg_get_proc_address(cpu_reg, "ggml_cpu_moe_prefetch");
        br->pool_pf_stop  = (pool_pf_stop_t)  ggml_backend_reg_get_proc_address(cpu_reg, "ggml_cpu_moe_prefetch_stop");
        br->pool_pf_stats = (pool_pf_stats_t) ggml_backend_reg_get_proc_address(cpu_reg, "ggml_cpu_moe_prefetch_stats");
        if (br->mode != GGML_MOE_BRIDGE_WAIT_SPIN) {
            LLAMA_LOG_WARN("%s: LLAMA_MOE_BRIDGE_PF needs LLAMA_MOE_BRIDGE_WAIT=spin: no next-layer prefetch\n", __func__);
        } else if (!br->pool_pf || !br->pool_pf_stop || !br->pool_pf_stats) {
            LLAMA_LOG_WARN("%s: LLAMA_MOE_BRIDGE_PF: the CPU backend has no prefetch: no next-layer prefetch\n", __func__);
        } else {
            br->pf_k    = std::min(64, std::max(1, env_int("LLAMA_MOE_BRIDGE_PF_K", 12)));
            br->pf_mode = std::min(1, std::max(0, env_int("LLAMA_MOE_BRIDGE_PF_MODE", 0)));
            br->pool_set_solo = (pool_set_solo_t) ggml_backend_reg_get_proc_address(cpu_reg, "ggml_cpu_moe_pool_set_solo");
            br->pf_solo = env_int("LLAMA_MOE_BRIDGE_PF_SOLO", 0) > 0 && br->pool_set_solo != nullptr;
            std::unordered_map<int, int> chan_of;
            for (size_t c = 0; c < br->chans.size(); ++c) {
                chan_of[br->chans[c].il] = (int) c;
            }
            br->pf_next.assign(br->chans.size(), -1);
            br->pf_router.resize(br->chans.size());
            size_t n_bytes = 0;
            int    n_pf    = 0;
            for (size_t c = 0; c < br->chans.size(); ++c) {
                const auto it = chan_of.find(br->chans[c].il + 1);
                if (it == chan_of.end()) {
                    continue;
                }
                const int nc = it->second;
                const ggml_tensor * r = model.layers[br->chans[nc].il].ffn_gate_inp;
                if (r == nullptr || r->ne[0] != br->n_embd || r->ne[1] != br->chans[nc].up->ne[2] || r->ne[2] != 1 ||
                    r->ne[3] != 1 || !ggml_is_contiguous(r)) {
                    continue;
                }
                if (br->pf_router[nc].empty() && br->hint_k <= 0) { // [TAG_FN_L3_CPU_DEVPRED] no host router with hints
                    const ggml_type_traits * tt = ggml_get_type_traits(r->type);
                    if (r->type != GGML_TYPE_F32 && (tt == nullptr || tt->to_float == nullptr)) {
                        continue;
                    }
                    std::vector<uint8_t> raw(ggml_nbytes(r));
                    ggml_backend_tensor_get(r, raw.data(), 0, raw.size());
                    std::vector<float> f32((size_t) ggml_nelements(r));
                    if (r->type == GGML_TYPE_F32) {
                        memcpy(f32.data(), raw.data(), f32.size()*sizeof(float));
                    } else {
                        tt->to_float(raw.data(), f32.data(), (int64_t) f32.size());
                    }
                    br->pf_router[nc].resize(f32.size());
                    ggml_fp32_to_fp16_row(f32.data(), br->pf_router[nc].data(), (int64_t) f32.size());
                    n_bytes += br->pf_router[nc].size()*sizeof(ggml_fp16_t);
                }
                br->pf_next[c] = nc;
                n_pf++;
            }
            br->pf_on = n_pf > 0;
            LLAMA_LOG_INFO("%s: [TAG_FN_R2_BRIDGE_PF] next-layer prefetch: %d of %zu layers predict their successor's experts "
                    "(top-%d per token, the cold ones pulled into the CPU caches between two jobs by %s%s), router copies "
                    "%.1f MiB\n", __func__, n_pf, br->chans.size(), br->pf_k, br->pf_mode == 0 ? "real loads" : "software prefetches",
                    br->pf_solo ? "; solo: the workers compute, the executor dispatches" : "", n_bytes/1048576.0);
            if (br->pool_split > 0 || br->pf_rank || br->hint_k > 0 || br->pf_streams > 1) {
                LLAMA_LOG_INFO("%s: [TAG_FN_L3_CPU] pool split %s, prefetch order %s, prediction %s, %d pull streams%s\n", __func__,
                        br->pool_split == 2 ? "steal" : br->pool_split == 1 ? "stable" : "range",
                        br->pf_rank || br->hint_k > 0 ? "by rank" : "by expert id",
                        br->hint_k > 0 ? (br->hint_mode == 2 ? "on the device (hints, the next layer's FFN mixer)" : "on the device (hints)") :
                        "host router", br->pf_streams,
                        br->pf_streams > 1 && br->pool_split == 0 ? " (they need split stable or steal)" : "");
            }
        }
    }

    if (!br->pf_on && br->hint_k > 0) {
        br->hint_k = 0; // [TAG_FN_L3_CPU_DEVPRED] the graphs then build no hint (the device areas stay unused)
    }

    if (br->mode == GGML_MOE_BRIDGE_WAIT_HOSTFUNC) {
        br->pool = br->pool_new(&br->pool_params); // run by the driver's callback thread, which keeps its affinity
        if (br->pool == nullptr) {
            return give_up("the CPU MoE pool could not start");
        }
        br->fn_set_runner(br->gb, br_run, br);
    } else {
        br->exec = std::thread(br_exec_main, br);
        std::unique_lock<std::mutex> lk(br->mtx);
        br->cv.wait(lk, [&] { return br->ready; });
        if (br->pool == nullptr) {
            lk.unlock();
            return give_up("the CPU MoE pool could not start");
        }
    }

    LLAMA_LOG_INFO("%s: MoE bridge %d on %s: %zu host expert layers (%d skipped), T <= %d, %s wait, timeout %d ms (%d ms once "
            "taken), spin %d us, up to %d pool threads, prio %d\n", __func__, br->bid, ggml_backend_dev_name(dev), br->chans.size(),
            n_skipped, br->max_t, br->mode == GGML_MOE_BRIDGE_WAIT_SPIN ? "spin" : "hostfunc", br->timeout_ms, br->job_max_ms,
            br->spin_us, br->pool_params.n_threads, br->pool_params.prio);
    if (br->pool_stats) {
        br->st_faults = br_page_faults();
        LLAMA_LOG_INFO("%s: [TAG_FN_L3_CPU_STATS] pool statistics every 256 bridged graphs\n", __func__);
    }
    if (br->exec_cpu >= 0) {
        LLAMA_LOG_INFO("%s: [TAG_FN_L3_CPU_PLACE] the executor runs on CPU %d, the pool's workers on the rest of its list\n",
                __func__, br->exec_cpu);
    }
    return br;
}

void llama_moe_bridge_free(llama_moe_bridge * br) {
    br_destroy(br);
}

// [TAG_FN_R4_BRIDGE_DMA]
bool llama_moe_bridge_attach_dma(llama_moe_bridge * br, const void * owner) {
    if (br == nullptr) {
        return false;
    }
    // [TAG_FN_R4_REVIEW] a bridge-mode DMA state (llama_moe_gen5_init made one for this bridge) that no graph will fetch
    // from - hostfunc waits, or a ring the device cannot read - would keep its pinned ring (LLAMA_MOE_DMA_RING_MIB,
    // 1 GiB by default) and its fill threads to the end of the context: give them back now. Another owner's state is
    // left alone (llama_moe_dma_free checks the owner), and no graph has been built yet.
    auto give_up = [owner]() {
        if (llama_moe_dma_bridge_mode()) {
            llama_moe_dma_free(owner);
        }
        return false;
    };
    if (br->max_fetch <= 0) {
        return give_up();
    }
    void * ptr       = nullptr;
    size_t size      = 0;
    int    max_fetch = 0;
    if (!llama_moe_dma_bridge_ring(owner, &ptr, &size, &max_fetch)) {
        LLAMA_LOG_WARN("%s: LLAMA_MOE_BRIDGE_DMA: the DMA state has no pinned ring in bridge mode: no DMA share\n", __func__);
        return give_up();
    }
    if (max_fetch > br->max_fetch) {
        LLAMA_LOG_WARN("%s: the DMA banks hold %d experts, the bridge fetches %d: no DMA share\n", __func__, max_fetch, br->max_fetch);
        return give_up();
    }
    int n = 0;
    for (auto & c : br->chans) {
        const llama_moe_dma_view * v = llama_moe_dma_bridge_lookup(c.up, 1);
        c.dma = v ? v->handle : nullptr;
        n += c.dma != nullptr;
    }
    // the ring goes to the device side only when a bridged layer has a bank: give_up() frees it otherwise, and the
    // device view must not keep a pointer into a freed ring
    if (n == 0 || !br->fn_set_ring(br->gb, ptr, size)) {
        for (auto & c : br->chans) {
            c.dma = nullptr;
        }
        if (n == 0) {
            LLAMA_LOG_WARN("%s: no bridged layer has a DMA bank: no DMA share\n", __func__);
        }
        return give_up();
    }
    br->dma_on = true;
    LLAMA_LOG_INFO("%s: MoE bridge %d: DMA share inside the bridged graphs on %d of %zu layers (fetch <= %d experts)\n",
            __func__, br->bid, n, br->chans.size(), max_fetch);
    if (br->fill_gap || br->fill_mask != 0) {
        llama_moe_dma_bridge_fill_policy(br->fill_gap, br->fill_mask); // [TAG_FN_L3_CPU_FILL]
    }
    return br->dma_on;
}

bool llama_moe_bridge_dma(const llama_moe_bridge * br) {
    return br != nullptr && br->dma_on;
}

bool llama_moe_bridge_active(const llama_moe_bridge * br) {
    return br != nullptr && br->active && !br->disabled;
}

// [TAG_FN_R2_BRIDGE_SYNC]
bool llama_moe_bridge_sync(const llama_moe_bridge * br) {
    return br != nullptr && br->sync_on && !llama_moe_bridge_active(br);
}

// [TAG_FN_L3_CPU_DEVPRED]
int llama_moe_bridge_hint_k(const llama_moe_bridge * br) {
    return br != nullptr && br->pf_on ? br->hint_k : 0;
}

int llama_moe_bridge_hint_mode(const llama_moe_bridge * br) {
    return llama_moe_bridge_hint_k(br) > 0 ? br->hint_mode : 0;
}

int llama_moe_bridge_max_t(const llama_moe_bridge * br) {
    return br ? br->max_t : 0;
}

int llama_moe_bridge_n_used(const llama_moe_bridge * br) {
    return br ? br->n_used : 0;
}

bool llama_moe_bridge_layer(const llama_moe_bridge * br, const ggml_tensor * up_exps, int32_t * id, int32_t * chan) {
    if (br == nullptr || up_exps == nullptr) {
        return false;
    }
    const auto it = br->by_up.find(up_exps);
    if (it == br->by_up.end()) {
        return false;
    }
    *id   = br->bid;
    *chan = it->second;
    return true;
}

void llama_moe_bridge_step(llama_moe_bridge * br) {
    if (br == nullptr || br->active || br->disabled) {
        return;
    }
    if (br->pause_left > 0) {
        br->pause_left--;
        return;
    }
    if (!br->fn_reset(br->gb)) {
        return; // the executor still owes the stalled job: try again next step
    }
    br->active = true;
    LLAMA_LOG_INFO("%s: MoE bridge %d re-armed\n", __func__, br->bid);
}

void llama_moe_bridge_begin(llama_moe_bridge * br, bool used) {
    if (br == nullptr) {
        return;
    }
    if (used) {
        {
            std::lock_guard<std::mutex> lk(br->graph_mtx);
            br->in_graph = true;
        }
        br->parked.store(false, std::memory_order_relaxed);
        br->wake_gen.fetch_add(1, std::memory_order_seq_cst);
        if (br->n_sleeping.load(std::memory_order_seq_cst) > 0) {
            {
                std::lock_guard<std::mutex> lk(br->mtx);
            }
            br->cv.notify_all();
        }
        br->pool_wake(br->pool);
    } else {
        // other CPU work (prefill splits) is about to run on the pool's cores
        br->parked.store(true, std::memory_order_relaxed);
        if (br->pf_on && br->pool) {
            br->pool_pf_stop(br->pool); // [TAG_FN_R2_BRIDGE_PF] the workers leave a prefetch at their next piece
        }
        br->pool_park(br->pool);
    }
}

bool llama_moe_bridge_end(llama_moe_bridge * br) {
    if (br == nullptr) {
        return true;
    }
    {
        std::lock_guard<std::mutex> lk(br->graph_mtx); // waits for an observer call in progress
        br->in_graph = false;
    }
    br->n_graphs++;
    const uint32_t err = br->fn_error(br->gb);

    // [TAG_FN_R4_BRIDGE_DMA] the graph synced: the device's wait and fetch times and the host's job times of its layers
    // move the DMA/CPU split (only layers whose last job and device times belong to the same post)
    if (br->dma_on && err == GGML_MOE_BRIDGE_ERR_NONE) {
        llama_moe_dma_bridge_step st;
        for (int ch = 0; ch < (int) br->chans.size(); ++ch) {
            auto & c = br->chans[ch];
            const uint32_t seq = c.job_seq;
            std::atomic_thread_fence(std::memory_order_acquire);
            if (!c.dma || seq == 0) {
                continue;
            }
            ggml_moe_bridge_chan_times t;
            br->fn_chan_times(br->gb, ch, &t);
            if (t.seq != seq) {
                continue; // not posted in this graph
            }
            st.n_layers++;
            st.gpu_wait_us += t.wait_ns/1e3;
            st.cpu_us      += c.job_ns/1e3;
            st.n_cpu       += c.job_n_cpu;
            if (c.job_fetch) {
                st.fetch_us += t.fetch_ns/1e3;
                st.n_fetch  += t.n_fetch;
            }
            c.job_seq = 0;
        }
        llama_moe_dma_bridge_feedback(st);
    }

    if (br->stats && (br->n_graphs % 256 == 0 || err != GGML_MOE_BRIDGE_ERR_NONE)) {
        ggml_moe_bridge_stats s;
        br->fn_get_stats(br->gb, &s);
        const uint64_t nj = br->n_jobs.load();
        LLAMA_LOG_INFO("%s: MoE bridge %d: %" PRIu64 " graphs, %" PRIu64 " jobs, job %.1f us avg / %.1f us max, "
                "device wait %.1f us avg, %.1f%% ready at the first look\n", __func__, br->bid, br->n_graphs, nj,
                nj ? br->job_ns.load()/1e3/nj : 0.0, br->job_ns_max.load()/1e3,
                s.waits ? s.wait_ns/1e3/s.waits : 0.0, s.waits ? 100.0*s.waits_ready/s.waits : 0.0);
        if (br->pf_on && br->pool) { // [TAG_FN_R2_BRIDGE_PF]
            uint64_t pj = 0, ps = 0, pb = 0, pe = 0;
            br->pool_pf_stats(br->pool, &pj, &ps, &pb, &pe);
            LLAMA_LOG_INFO("%s: MoE bridge %d: [TAG_FN_R2_BRIDGE_PF] %" PRIu64 " prefetches, %.1f%% stopped by the next job, "
                    "%.2f predicted cold experts and %.2f MiB per prefetch\n", __func__, br->bid, pj, pj ? 100.0*ps/pj : 0.0,
                    pj ? (double) pe/pj : 0.0, pj ? pb/1048576.0/pj : 0.0);
        }
    }
    if (br->pool_stats && br->pool && br->n_graphs % 256 == 0) {
        // [TAG_FN_L3_CPU_STATS] the window since the last line (the pool resets with the read)
        ggml_cpu_moe_pool_stats ps;
        br->pool_get_stats(br->pool, &ps, true);
        const uint64_t nj     = br->n_jobs.load();
        const uint64_t faults = br_page_faults();
        const uint64_t dj     = nj > br->st_jobs ? nj - br->st_jobs : 0;
        const double   f_job  = dj ? (double) (faults - br->st_faults)/dj : 0.0;
        br->st_faults = faults;
        br->st_jobs   = nj;
        const uint64_t nd = br->n_disp.exchange(0);
        const double   disp_us = nd ? br->disp_ns.exchange(0)/1e3/nd : 0.0;
        LLAMA_LOG_INFO("%s: MoE bridge %d: [TAG_FN_L3_CPU_STATS] %" PRIu64 " jobs: %.1f us, %.2f MiB = %.1f GB/s, %.2f experts; "
                "dispatch %.1f us, start lag %.1f us, gate/up %.1f + wait %.1f us (spread %.1f), down %.1f + wait %.1f us "
                "(spread %.1f), sum %.1f us; taken %.2f pieces, prefetched-and-computed %.2f MiB; %.1f page faults per job\n",
                __func__, br->bid, ps.jobs, ps.job_us, ps.mib, ps.job_us > 0.0 ? ps.mib*1.048576/ps.job_us*1e3 : 0.0, ps.experts,
                disp_us, ps.lag_us, ps.p3_us, ps.w3_us, ps.spread3_us, ps.p4_us, ps.w4_us, ps.spread4_us, ps.p5_us, ps.taken,
                ps.pf_hit_mib, f_job);
        std::string doms;
        for (int d = 0; d < ps.n_dom; ++d) {
            char tmp[96];
            snprintf(tmp, sizeof(tmp), "%sL3 %d: %d threads x %.1f GB/s", d ? ", " : "", d, ps.dom_threads[d], ps.dom_gbs[d]);
            doms += tmp;
        }
        LLAMA_LOG_INFO("%s: MoE bridge %d: [TAG_FN_L3_CPU_STATS] per domain (busy time) %s; prefetch: %" PRIu64 " posts, window "
                "%.1f us (host router %.1f us), %.2f MiB and %.2f experts each, %.1f%% stopped (the job waited %.1f us, max %.1f); "
                "hints %" PRIu64 " read, %" PRIu64 " missing\n",
                __func__, br->bid, doms.c_str(), ps.pf_jobs, ps.pf_us, ps.pf_router_us, ps.pf_mib, ps.pf_experts,
                100.0*ps.pf_stopped, ps.pf_stop_us, ps.pf_stop_max_us, br->n_hint_ok.exchange(0), br->n_hint_miss.exchange(0));
        // the workers by their rate over the window: the slowest gate every barrier
        std::vector<int> ord;
        for (int k = 0; k < ps.n_thr; ++k) {
            if (ps.thr_busy_us[k] > 0.0) {
                ord.push_back(k);
            }
        }
        std::sort(ord.begin(), ord.end(), [&ps](int a, int b) { return ps.thr_gbs[a] < ps.thr_gbs[b]; });
        std::string wl;
        for (size_t i = 0; i < ord.size(); ++i) {
            if (i >= 4 && i + 2 < ord.size()) {
                continue; // the 4 slowest and the 2 fastest
            }
            char tmp[80];
            snprintf(tmp, sizeof(tmp), "%scpu %d %.1f GB/s %.0f us", wl.empty() ? "" : ", ", ps.thr_cpu[ord[i]], ps.thr_gbs[ord[i]],
                    ps.thr_busy_us[ord[i]]);
            wl += tmp;
        }
        uint64_t held_us = 0;
        uint64_t held_n  = 0;
        if (br->fill_gap && br->dma_on) {
            llama_moe_dma_bridge_fill_held(&held_us, &held_n);
        }
        LLAMA_LOG_INFO("%s: MoE bridge %d: [TAG_FN_L3_CPU_STATS] workers slowest first: %s; slowest job %.1f us, %" PRIu64 " jobs "
                "over 1 ms; fills held %" PRIu64 " times, %.1f ms in all; prediction over %" PRIu64 " jobs: precision %.2f (first 4: "
                "%.2f), recall %.2f\n", __func__, br->bid, wl.c_str(), ps.job_max_us, ps.jobs_slow, held_n, held_us/1e3,
                ps.pred_jobs, ps.pf_precision, ps.pf_prec_top4, ps.pf_recall);
    }
    if (err == GGML_MOE_BRIDGE_ERR_NONE) {
        return true;
    }

    ggml_moe_bridge_stats s;
    br->fn_get_stats(br->gb, &s);
    const int il = s.error_chan >= 0 && s.error_chan < (int) br->chans.size() ? br->chans[s.error_chan].il : -1;
    // [TAG_FN_R1_BRIDGE_RB] errors far apart are transient stalls whose ubatch was computed again: only 3 within 1024
    // bridged graphs turn the bridge off for the rest of the context (a host side that keeps failing)
    if (br->n_graphs - br->err_graph > 1024) {
        br->n_errors = 0;
    }
    br->err_graph = br->n_graphs;
    br->n_errors++;
    br->active     = false;
    br->pause_left = 16;
    if (br->n_errors >= 3) {
        br->disabled = true;
    }
    LLAMA_LOG_WARN("%s: MoE bridge %d: %s at layer %d: this ubatch fails; %s\n", __func__, br->bid,
            err == GGML_MOE_BRIDGE_ERR_TIMEOUT ? "wait timeout" : "host job failed", il,
            br->disabled ? "3 errors, the bridge stays off for this context" : "the bridge pauses for 16 steps");
    return false;
}
