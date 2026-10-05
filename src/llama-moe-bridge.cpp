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

namespace {

using pool_params_default_t = ggml_cpu_moe_pool_params (*)(int);
using pool_new_t            = ggml_cpu_moe_pool * (*)(const ggml_cpu_moe_pool_params *);
using pool_free_t           = void (*)(ggml_cpu_moe_pool *);
using pool_run_t            = ggml_status (*)(ggml_cpu_moe_pool *, const ggml_cpu_moe_job *);
using pool_park_t           = void (*)(ggml_cpu_moe_pool *);
using layer_supported_t     = bool (*)(const ggml_cpu_moe_layer *);

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

    ggml_moe_bridge * gb  = nullptr;
    int32_t           bid = -1;

    int     mode       = GGML_MOE_BRIDGE_WAIT_SPIN;
    int     max_t      = 8;
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
    const ggml_cpu_moe_job job = { layer, j->n_tokens, j->n_used, j->x, j->ids, j->w, j->out };
    const bool ok = br->pool_run(br->pool, &job) == GGML_STATUS_SUCCESS;
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

// spin mode: poll the posts in order, spin for spin_us after the last job, then sleep in 1 ms slices (so a post that
// comes without a wake is still served) until the owning context wakes it
static void br_exec_main(llama_moe_bridge * br) {
    ggml_cpu_moe_pool_params pp = br->pool_params;
    pp.pin_caller = true;
    ggml_cpu_moe_pool * pool = br->pool_new(&pp);
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

llama_moe_bridge * llama_moe_bridge_create(const llama_model & model, int n_threads) {
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
    br->spin_us    = std::max(0, env_int("LLAMA_MOE_BRIDGE_SPIN_US", 2000));
    br->timeout_ms = std::min(1200, std::max(1, env_int("LLAMA_MOE_BRIDGE_TIMEOUT_MS", 500))); // [TAG_FN_R1_BRIDGE_RETRY]
    br->job_max_ms = std::min(1200, std::max(br->timeout_ms, env_int("LLAMA_MOE_BRIDGE_JOB_MAX_MS", 1000)));
    br->stats      = env_int("LLAMA_MOE_BRIDGE_STATS", 0) > 0;
    br->stall_ms    = std::max(0, env_int("LLAMA_MOE_BRIDGE_TEST_STALL", 0));
    br->stall_every = std::max(0, env_int("LLAMA_MOE_BRIDGE_TEST_STALL_EVERY", 0));

    const int n_pool = std::max(1, env_int("LLAMA_MOE_BRIDGE_THREADS", n_threads > 0 ? n_threads : 1));
    br->pool_params = br->pool_params_default(std::min(n_pool, GGML_MAX_N_THREADS));
    br->pool_params.prio    = std::min(3, std::max(0, env_int("LLAMA_MOE_BRIDGE_PRIO", GGML_SCHED_PRIO_HIGH)));
    br->pool_params.spin_us = br->spin_us;
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
    br->gb = br->fn_new(&bp);
    if (br->gb == nullptr) {
        return give_up("the device could not create the bridge");
    }
    br->bid = br->fn_id(br->gb);

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
    return br->dma_on;
}

bool llama_moe_bridge_dma(const llama_moe_bridge * br) {
    return br != nullptr && br->dma_on;
}

bool llama_moe_bridge_active(const llama_moe_bridge * br) {
    return br != nullptr && br->active && !br->disabled;
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
    }
    if (err == GGML_MOE_BRIDGE_ERR_NONE) {
        return true;
    }

    ggml_moe_bridge_stats s;
    br->fn_get_stats(br->gb, &s);
    const int il = s.error_chan >= 0 && s.error_chan < (int) br->chans.size() ? br->chans[s.error_chan].il : -1;
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
