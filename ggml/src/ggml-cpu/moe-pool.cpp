// [TAG_FN_CPU_MOE_FUSE] persistent CPU worker pool for the host experts of one MoE layer (ggml_cpu_moe_run).
// The calling thread is worker 0; the other workers spin for spin_us after a job, then sleep until the next one.
// The work itself is ggml_fn_moe_compute (ggml-cpu.c), the same kernel as the fused graph op.

#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <psapi.h> // [TAG_FN_L7_FAULTS] K32GetProcessMemoryInfo
#else
#    include <pthread.h>
#    include <sched.h>
#    include <unistd.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#    include <immintrin.h>
#endif

#define GGML_MOE_POOL_MAX_RANK 64 // [TAG_FN_L6_PF]

namespace {

inline void moe_pool_relax() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    _mm_pause();
#elif defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
    __asm__ volatile("yield" ::: "memory");
#else
    std::this_thread::yield();
#endif
}

// [TAG_FN_L7_FAULTS] the process's page faults so far (soft + hard), for the pool statistics only
uint64_t moe_proc_faults() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    memset(&pmc, 0, sizeof(pmc));
    pmc.cb = sizeof(pmc);
    return K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)) ? (uint64_t) pmc.PageFaultCount : 0;
#else
    return 0;
#endif
}

// one logical CPU per physical core (the lowest-numbered SMT sibling), in CPU order; empty if unknown
std::vector<int> physical_core_cpus() {
    std::vector<int> cpus;
#if defined(_WIN32)
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) {
        return cpus;
    }
    std::vector<char> buf(len);
    auto * info = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) buf.data();
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, info, &len)) {
        return cpus;
    }
    for (DWORD off = 0; off < len;) {
        auto * p = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) (buf.data() + off);
        if (p->Relationship == RelationProcessorCore && p->Processor.GroupCount >= 1 && p->Processor.GroupMask[0].Group == 0) {
            const KAFFINITY m = p->Processor.GroupMask[0].Mask;
            for (int b = 0; b < 64; b++) {
                if ((m >> b) & 1) {
                    cpus.push_back(b);
                    break;
                }
            }
        }
        off += p->Size;
    }
#elif defined(__linux__)
    const long n = sysconf(_SC_NPROCESSORS_CONF);
    for (long c = 0; c < n && c < GGML_MAX_N_THREADS; c++) {
        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%ld/topology/thread_siblings_list", c);
        FILE * f = fopen(path, "r");
        if (!f) {
            continue;
        }
        long first = -1;
        if (fscanf(f, "%ld", &first) != 1) {
            first = -1;
        }
        fclose(f);
        if (first == c) {
            cpus.push_back((int) c);
        }
    }
#endif
    std::sort(cpus.begin(), cpus.end());
    return cpus;
}

// [TAG_FN_L3_CPU_STATS] the L3 cache domain of a logical CPU (0, 1, ... in order of appearance), or 0 if unknown
std::vector<int> l3_domains(int n_cpu) {
    std::vector<int> dom(n_cpu, 0);
#if defined(_WIN32)
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationCache, nullptr, &len);
    if (len == 0) {
        return dom;
    }
    std::vector<char> buf(len);
    auto * info = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) buf.data();
    if (!GetLogicalProcessorInformationEx(RelationCache, info, &len)) {
        return dom;
    }
    int next = 0;
    for (DWORD off = 0; off < len;) {
        auto * p = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) (buf.data() + off);
        if (p->Relationship == RelationCache && p->Cache.Level == 3 && p->Cache.GroupMask.Group == 0) {
            const KAFFINITY m = p->Cache.GroupMask.Mask;
            for (int b = 0; b < 64 && b < n_cpu; b++) {
                if ((m >> b) & 1) {
                    dom[b] = next;
                }
            }
            next++;
        }
        off += p->Size;
    }
#elif defined(__linux__)
    std::vector<long> ids;
    for (int c = 0; c < n_cpu; c++) {
        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cache/index3/id", c);
        FILE * f = fopen(path, "r");
        long id = -1;
        if (f) {
            if (fscanf(f, "%ld", &id) != 1) {
                id = -1;
            }
            fclose(f);
        }
        if (id < 0) {
            continue;
        }
        const auto it = std::find(ids.begin(), ids.end(), id);
        dom[c] = (int) (it - ids.begin());
        if (it == ids.end()) {
            ids.push_back(id);
        }
    }
#endif
    return dom;
}

// the CPU the calling thread is pinned to, or -1 if its affinity allows more than one
int current_thread_single_cpu() {
#if defined(_WIN32)
    DWORD_PTR proc_mask = 0;
    DWORD_PTR sys_mask  = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &proc_mask, &sys_mask) || proc_mask == 0) {
        return -1;
    }
    // SetThreadAffinityMask returns the previous mask: set the process mask, then put the old one back
    const DWORD_PTR prev = SetThreadAffinityMask(GetCurrentThread(), proc_mask);
    if (prev == 0) {
        return -1;
    }
    SetThreadAffinityMask(GetCurrentThread(), prev);
    int cpu = -1;
    for (int b = 0; b < 64; b++) {
        if ((prev >> b) & 1) {
            if (cpu >= 0) {
                return -1;
            }
            cpu = b;
        }
    }
    return cpu;
#elif defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (pthread_getaffinity_np(pthread_self(), sizeof(set), &set) != 0 || CPU_COUNT(&set) != 1) {
        return -1;
    }
    for (int c = 0; c < CPU_SETSIZE; c++) {
        if (CPU_ISSET(c, &set)) {
            return c;
        }
    }
    return -1;
#else
    return -1;
#endif
}

void pin_current_thread(int cpu) {
    if (cpu < 0) {
        return;
    }
#if defined(_WIN32)
    if (cpu < 64) {
        SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) 1 << cpu);
    }
#elif defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
    GGML_UNUSED(cpu);
#endif
}

void set_current_thread_prio(int prio) {
#if defined(_WIN32)
    if (prio != GGML_SCHED_PRIO_LOW) {
#    if _WIN32_WINNT >= 0x0602
        // keep the scheduler from parking the worker on a throttled core
        THREAD_POWER_THROTTLING_STATE t;
        ZeroMemory(&t, sizeof(t));
        t.Version     = THREAD_POWER_THROTTLING_CURRENT_VERSION;
        t.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
        t.StateMask   = 0;
        SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &t, sizeof(t));
#    endif
    }
    int p = THREAD_PRIORITY_NORMAL;
    switch (prio) {
        case GGML_SCHED_PRIO_LOW:      p = THREAD_PRIORITY_BELOW_NORMAL;  break;
        case GGML_SCHED_PRIO_MEDIUM:   p = THREAD_PRIORITY_ABOVE_NORMAL;  break;
        case GGML_SCHED_PRIO_HIGH:     p = THREAD_PRIORITY_HIGHEST;       break;
        case GGML_SCHED_PRIO_REALTIME: p = THREAD_PRIORITY_TIME_CRITICAL; break;
        default: return;
    }
    SetThreadPriority(GetCurrentThread(), p);
#else
    GGML_UNUSED(prio); // real-time policies need privileges; keep the inherited policy
#endif
}

// [TAG_FN_L3_CPU_STATS] one compute thread's job record (ggml_fn_moe_args.ts), and one worker's prefetch record
struct alignas(64) moe_ts_rec {
    uint64_t v[GGML_FN_MOE_TS_N];
};
struct alignas(64) moe_pf_rec {
    uint64_t t0;     // entry
    uint64_t t1;     // router phase done (= t0 with a given list)
    uint64_t t2;     // left the prefetch
    uint64_t bytes;
    uint64_t pad[4];
};

// [TAG_FN_L3_CPU_STATS] sums in ticks over the window (the caller adds a job after it, the owner reads them)
struct moe_stats_acc {
    uint64_t jobs = 0;
    double   job = 0, bytes = 0, experts = 0, lag = 0, p3 = 0, w3 = 0, p4 = 0, w4 = 0, p5 = 0, spread3 = 0, spread4 = 0;
    double   taken = 0, pfhit = 0;
    double   dom_bytes[4] = {}, dom_busy[4] = {};
    int      dom_threads[4] = {};
    uint64_t pf_jobs = 0, pf_stopped = 0;
    double   pf_t = 0, pf_router = 0, pf_bytes = 0, pf_experts = 0;
    uint64_t job_max = 0, jobs_slow = 0, pf_stop_max = 0;
    double   flt = 0, flt_slow = 0, slow_t = 0; // [TAG_FN_L7_FAULTS] page faults inside jobs, inside slow jobs; slow jobs' time
    uint64_t slow_nofault = 0, fast_fault = 0;  // slow jobs without a page fault, jobs under 1 ms with one
    // [TAG_FN_L7_SLOW] per slow job: the compute thread that ended last (by worker index) and where its excess over the
    // median thread went: start lag, gate/up, down, sum
    std::vector<uint32_t> slow_last;            // by worker index
    double   slow_x_lag = 0, slow_x_p3 = 0, slow_x_p4 = 0, slow_x_p5 = 0, slow_x_w = 0;
    double   pf_stop = 0;
    uint64_t pred_jobs = 0;
    double   pred_n = 0, pred_job = 0, pred_hit = 0, pred_hit4 = 0, pred_top4 = 0;
    std::vector<double> thr_busy, thr_bytes; // by worker index
};

} // namespace

struct ggml_cpu_moe_pool {
    int n_threads = 1;
    int spin_us   = 0;
    int prio      = 0;
    std::vector<int>         cpus; // per worker; worker 0 is the caller
    std::vector<std::thread> workers;

    alignas(64) std::atomic<uint32_t> seq{0};
    alignas(64) std::atomic<int>      n_done{0};
    alignas(64) std::atomic<int>      bar_count{0};
    alignas(64) std::atomic<int>      bar_gen{0};
    std::atomic<int>  n_sleeping{0};
    std::atomic<bool> stop{false};
    std::atomic<bool> parked{false};                 // [TAG_MOE_BRIDGE] idle workers sleep at once
    alignas(64) std::atomic<uint32_t> wake_gen{0};   // [TAG_MOE_BRIDGE] bumped by ggml_cpu_moe_pool_wake

    std::mutex              mtx;
    std::condition_variable cv;

    ggml_fn_moe_args     args{};
    std::vector<uint8_t> work;

    // [TAG_FN_R2_BRIDGE_PF] the asynchronous next-layer prefetch (ggml_cpu_moe_prefetch)
    std::atomic<int>          kind{0};             // of the posted job: 0 compute (args), 1 prefetch (pf), 2 compute
                                                   // by the workers alone (solo)
    bool                      solo = false;        // ggml_cpu_moe_pool_set_solo
    volatile int32_t          pf_stop     = 0;     // nonzero: the prefetching workers stop at their next piece
    bool                      pf_inflight = false; // the caller's side: a prefetch is posted and not waited for yet
    alignas(64) std::atomic<int> pf_bar_count{0};  // the workers' own barrier (the caller does not take part)
    alignas(64) std::atomic<int> pf_bar_gen{0};
    ggml_cpu_moe_prefetch_job pf{};
    std::vector<ggml_fp16_t>  pf_xh;               // [n_tokens][n_embd] the input in f16 (the router rows' dot type)
    std::vector<float>        pf_logits;           // [n_tokens][n_expert]
    std::vector<int32_t>      pf_list;             // the predicted experts the table does not serve: ascending, or by
                                                   // rank ([TAG_FN_L3_CPU_PFRANK] / a given list)
    std::vector<int32_t>      pf_idx;              // selection scratch
    std::vector<uint8_t>      pf_mark;
    int                       pf_n = 0;            // entries of pf_list (worker 1 writes it before the second barrier)
    std::atomic<uint64_t>     pf_jobs{0};
    std::atomic<uint64_t>     pf_stopped{0};
    std::atomic<uint64_t>     pf_bytes{0};
    std::atomic<uint64_t>     pf_experts{0};

    // [TAG_FN_L3_HOST_EXEC] the caller's part of a prefetch: pf_gen numbers the posted prefetches (caller side),
    // worker 1 stores it in pf_list_gen once pf_list / pf_n hold that prefetch's prediction
    uint32_t                     pf_gen = 0;
    alignas(64) std::atomic<uint32_t> pf_list_gen{0};
    std::atomic<uint64_t>        pfc_calls{0};
    std::atomic<uint64_t>        pfc_stopped{0};
    std::atomic<uint64_t>        pfc_bytes{0};

    // [TAG_FN_L3_CPU_SPLIT]
    int                   split   = GGML_CPU_MOE_SPLIT_RANGE;
    bool                  pf_rank = false;         // [TAG_FN_L3_CPU_PFRANK]
    int                   swpf    = 0;             // [TAG_FN_L3_CPU_SWPF]
    int                   pf_streams = 1;          // [TAG_FN_L3_CPU_PFSTREAMS]
    bool                  pf_given = false;        // the posted prefetch carries its list: no router phase
    const ggml_tensor *   pf_up   = nullptr;       // the layer (its up tensor) the last prefetch predicted
    std::vector<int32_t>  pf_topk;                 // [n_tokens][k] by rank (pf_rank)
    std::vector<int32_t>  claim;                   // split STEAL: the claim words, kept between jobs
    uint32_t              epoch   = 0;

    // [TAG_FN_L4_MEM_POOLBAR] the dataflow counters of a job (cleared before it is posted), 64-byte aligned
    bool                  dflow   = false;
    std::vector<int32_t>  dfl;
    bool                  pf_fix  = false;         // [TAG_FN_L4_MEM_PFDEV]
    // [TAG_FN_L6_PF] pf_score: the given list of the last prefetch ([T][k] by rank) and the hit counts per rank (decayed)
    bool                  pf_score = false;
    int                   pf_pull  = 0;            // [TAG_FN_L6_PF] GGML_FN_PF_* flags of the pull
    int                   pf_cap   = 0;            // [TAG_FN_L6_PF] experts of a given list pulled at most (0: all)
    int                   pf_fresh = 0;            // [TAG_FN_L6_PF] see ggml_cpu_moe_pool_params.pf_fresh
    uint32_t              pf_clock = 0;            // jobs run
    std::unordered_map<const ggml_tensor *, std::vector<uint32_t>> pf_seen; // per layer (up tensor): the job clock of each expert's last CPU job
    std::vector<int32_t>  pf_gt;
    int                   pf_gT = 0;
    int                   pf_gk = 0;
    double                rk_n[GGML_MOE_POOL_MAX_RANK] = {};
    double                rk_hit[GGML_MOE_POOL_MAX_RANK] = {};
    std::vector<float>    pf_q;                    // per expert scratch

    // [TAG_FN_L3_CPU_STATS]
    bool                    stats = false;
    double                  tick_per_us = 1.0;
    std::vector<moe_ts_rec> ts;                    // [n_threads], by compute thread index
    std::vector<moe_pf_rec> pf_ts;                 // [n_threads], by worker index
    std::vector<uint8_t>    pf_done;               // [n_threads][n_expert], by compute thread index
    int64_t                 pf_done_n_exp = 0;
    bool                    pf_done_valid = false; // pf_done belongs to pf_up's prefetch
    std::vector<int>        dom;                   // per worker: its L3 domain (0..3)
    uint64_t                pf_t_post = 0;
    std::mutex              st_mtx;
    moe_stats_acc           st;
    // the last job's and prefetch's records are summed later, off the job's critical path (before the next job or
    // prefetch overwrites them)
    bool                    st_job_pending = false;
    uint64_t                st_t_pub  = 0;
    uint64_t                st_t_done = 0;
    uint64_t                st_f_pub  = 0; // [TAG_FN_L7_FAULTS]
    uint64_t                st_f_done = 0;
    int                     st_nth    = 0;
    int                     st_base   = 0;
    bool                    st_pf_pending = false;
    bool                    st_pf_stopped = false;
    uint64_t                st_pf_stop_t  = 0;     // ticks the caller waited for the stopped prefetch
};

static void moe_pool_barrier(void * ctx) {
    auto * p = (ggml_cpu_moe_pool *) ctx;
    if (p->n_threads == 1) {
        return;
    }
    const int gen = p->bar_gen.load(std::memory_order_acquire);
    if (p->bar_count.fetch_add(1, std::memory_order_acq_rel) == p->n_threads - 1) {
        p->bar_count.store(0, std::memory_order_relaxed);
        p->bar_gen.fetch_add(1, std::memory_order_acq_rel);
        return;
    }
    while (p->bar_gen.load(std::memory_order_acquire) == gen) {
        moe_pool_relax();
    }
}

// [TAG_FN_R2_BRIDGE_PF] the barrier of the prefetching workers (1 .. n_threads-1)
static void moe_pool_pf_barrier(ggml_cpu_moe_pool * p, int nw) {
    if (nw <= 1) {
        return;
    }
    const int gen = p->pf_bar_gen.load(std::memory_order_acquire);
    if (p->pf_bar_count.fetch_add(1, std::memory_order_acq_rel) == nw - 1) {
        p->pf_bar_count.store(0, std::memory_order_relaxed);
        p->pf_bar_gen.fetch_add(1, std::memory_order_acq_rel);
        return;
    }
    while (p->pf_bar_gen.load(std::memory_order_acquire) == gen) {
        moe_pool_relax();
    }
}

// [TAG_FN_R2_BRIDGE_PF] the barrier of a solo compute job (the workers only)
static void moe_pool_solo_barrier(void * ctx) {
    auto * p = (ggml_cpu_moe_pool *) ctx;
    moe_pool_pf_barrier(p, p->n_threads - 1);
}

// [TAG_FN_L3_CPU_PFRANK] the predicted experts by rank: every token's first expert, then every token's second, ...;
// duplicates and the table's experts dropped. topk: [n_tokens][k]; mark: scratch of >= n_exp entries
static int moe_pool_rank_list(const ggml_cpu_moe_layer * L, const int32_t * topk, int n_tokens, int k, int64_t n_exp,
                              std::vector<uint8_t> & mark, int32_t * out) {
    std::fill(mark.begin(), mark.end(), (uint8_t) 0);
    int n = 0;
    for (int r = 0; r < k; ++r) {
        for (int t = 0; t < n_tokens; ++t) {
            const int32_t e = topk[(size_t) t*k + r];
            if (e < 0 || e >= n_exp || mark[e] || (L->table != nullptr && L->table[e] != L->table_miss)) {
                continue;
            }
            mark[e] = 1;
            out[n++] = e;
        }
    }
    return n;
}

// [TAG_FN_L6_PF] a given list by the estimated chance that the job computes each expert: 1 - prod over its (token, rank)
// entries of (1 - P(rank)), P learned from the jobs (moe_pool_score_learn); ties keep the rank order
static int moe_pool_score_list(ggml_cpu_moe_pool * p, const ggml_cpu_moe_layer * L, const int32_t * topk, int n_tokens, int k,
                               int64_t n_exp, int32_t * out) {
    const int n = moe_pool_rank_list(L, topk, n_tokens, k, n_exp, p->pf_mark, out);
    if (n < 2) {
        return n;
    }
    p->pf_q.resize((size_t) n_exp);
    for (int i = 0; i < n; ++i) {
        p->pf_q[out[i]] = 1.0f;
    }
    for (int t = 0; t < n_tokens; ++t) {
        for (int r = 0; r < k && r < GGML_MOE_POOL_MAX_RANK; ++r) {
            const int32_t e = topk[(size_t) t*k + r];
            if (e >= 0 && e < n_exp && p->pf_mark[e]) {
                const double pr = (p->rk_hit[r] + 1.0)/(p->rk_n[r] + 2.0);
                p->pf_q[e] *= (float) (1.0 - pr);
            }
        }
    }
    const float * q = p->pf_q.data();
    std::stable_sort(out, out + n, [q](int32_t a, int32_t b) { return q[a] < q[b]; });
    return n;
}

// [TAG_FN_L6_PF] the job of the layer the last given list predicted: count, per rank, its cold entries and those the job
// computes (any token). mark: scratch of >= n_exp entries
static void moe_pool_score_learn(ggml_cpu_moe_pool * p, const ggml_cpu_moe_layer * L, const int32_t * ids, int n_ids, int64_t n_exp) {
    auto & mk = p->pf_mark;
    std::fill(mk.begin(), mk.end(), (uint8_t) 0);
    for (int i = 0; i < n_ids; ++i) {
        if (ids[i] >= 0 && ids[i] < n_exp) {
            mk[ids[i]] = 1;
        }
    }
    const int k = std::min(p->pf_gk, (int) GGML_MOE_POOL_MAX_RANK);
    for (int t = 0; t < p->pf_gT; ++t) {
        for (int r = 0; r < k; ++r) {
            const int32_t e = p->pf_gt[(size_t) t*p->pf_gk + r];
            if (e < 0 || e >= n_exp || (L->table != nullptr && L->table[e] != L->table_miss)) {
                continue;
            }
            p->rk_n[r]   += 1.0;
            p->rk_hit[r] += mk[e];
        }
    }
    if (p->rk_n[0] > 50000.0) { // a slow decay: the rates follow the context
        for (int r = 0; r < GGML_MOE_POOL_MAX_RANK; ++r) {
            p->rk_n[r]   *= 0.5;
            p->rk_hit[r] *= 0.5;
        }
    }
}

// [TAG_FN_R2_BRIDGE_PF] the pull phase of worker ith: its pieces of a job of the predicted experts, as thread ith of
// n_threads computes them, or (solo) thread ith - 1 of the n_threads - 1 workers. [TAG_FN_L3_CPU_SPLIT] With a split >= 1
// every piece goes to the owner the job gives it, in the order of the list.
static void moe_pool_prefetch_pull(ggml_cpu_moe_pool * p, int ith, moe_pf_rec * rec) {
    const ggml_cpu_moe_prefetch_job & J = p->pf;
    const ggml_cpu_moe_layer * L = J.layer;
    const int64_t n_exp = L->up->ne[2];
    const int c_ith = p->solo ? ith - 1 : ith;
    const int c_nth = p->solo ? p->n_threads - 1 : p->n_threads;
    size_t b = 0;
    if (p->split >= GGML_CPU_MOE_SPLIT_STABLE && c_nth >= 2) { // as ggml_cpu_moe_run: one compute thread splits by range
        uint8_t * done = rec && p->pf_done_n_exp == n_exp ? p->pf_done.data() + (size_t) c_ith*n_exp : nullptr;
        b = ggml_fn_moe_prefetch_stable(L->up, L->gate, L->down, p->pf_list.data(), p->pf_n, c_ith, c_nth, &p->pf_stop,
                J.mode, done, p->pf_streams, p->pf_pull);
    } else {
        b = ggml_fn_moe_prefetch(L->up, L->gate, L->down, p->pf_list.data(), p->pf_n, c_ith, c_nth, &p->pf_stop, J.mode);
    }
    p->pf_bytes.fetch_add((uint64_t) b, std::memory_order_relaxed);
    if (rec) {
        rec->bytes = b;
        rec->t2    = ggml_fn_moe_tick();
    }
}

// [TAG_FN_R2_BRIDGE_PF] worker ith (>= 1) of a prefetch job: the router logits of its share of the experts, then (worker
// 1) the top-k of every token minus the table's experts, then its pieces of a job of those experts (as thread ith of
// n_threads computes them). Every worker passes both barriers even after a stop, so the job always ends cleanly.
static void moe_pool_prefetch_part(ggml_cpu_moe_pool * p, int ith) {
    const ggml_cpu_moe_prefetch_job & J = p->pf;
    const ggml_cpu_moe_layer * L = J.layer;
    const int64_t n_embd = L->up->ne[0];
    const int     n_exp  = (int) L->up->ne[2];
    const int     T      = J.n_tokens;
    const int     nw     = p->n_threads - 1;
    const int     w      = ith - 1;
    moe_pf_rec *  rec    = p->stats ? &p->pf_ts[ith] : nullptr; // [TAG_FN_L3_CPU_STATS]
    if (rec) {
        rec->t0 = rec->t1 = ggml_fn_moe_tick();
    }
    if (p->pf_given) {
        moe_pool_prefetch_pull(p, ith, rec); // [TAG_FN_L3_CPU_DEVPRED] the caller ranked the given list: no router, no barrier
        return;
    }

    if (!p->pf_stop) {
        const struct ggml_type_traits_cpu * tf = ggml_get_type_traits_cpu(GGML_TYPE_F16);
        const int e0 = n_exp*w/nw;
        const int e1 = n_exp*(w + 1)/nw;
        for (int e = e0; e < e1 && !p->pf_stop; ++e) { // a job that comes now waits for one row at most
            const ggml_fp16_t * row = J.router + (size_t) e*n_embd;
            for (int t = 0; t < T; ++t) {
                float v = 0.0f;
                tf->vec_dot((int) n_embd, &v, 0, row, 0, p->pf_xh.data() + (size_t) t*n_embd, 0, 1);
                p->pf_logits[(size_t) t*n_exp + e] = v;
            }
        }
    }
    moe_pool_pf_barrier(p, nw);

    if (w == 0) {
        int n = 0;
        if (!p->pf_stop && !p->pf_rank) {
            std::fill(p->pf_mark.begin(), p->pf_mark.end(), (uint8_t) 0);
            const int k = std::min<int>(J.k, n_exp);
            for (int t = 0; t < T; ++t) {
                const float * lg = p->pf_logits.data() + (size_t) t*n_exp;
                for (int e = 0; e < n_exp; ++e) {
                    p->pf_idx[e] = e;
                }
                std::nth_element(p->pf_idx.begin(), p->pf_idx.begin() + (k - 1), p->pf_idx.begin() + n_exp,
                        [lg](int32_t a, int32_t b) { return lg[a] > lg[b]; });
                for (int i = 0; i < k; ++i) {
                    p->pf_mark[p->pf_idx[i]] = 1;
                }
            }
            for (int e = 0; e < n_exp; ++e) {
                if (p->pf_mark[e] && !(L->table != nullptr && L->table[e] != L->table_miss)) {
                    p->pf_list[n++] = e;
                }
            }
        }
        if (!p->pf_stop && p->pf_rank) {
            // [TAG_FN_L3_CPU_PFRANK] each token's top k by logit (ties by id, a NaN last), then merged by rank
            const int k = std::min<int>(J.k, n_exp);
            for (int t = 0; t < T; ++t) {
                const float * lg = p->pf_logits.data() + (size_t) t*n_exp;
                for (int e = 0; e < n_exp; ++e) {
                    p->pf_idx[e] = e;
                }
                auto key = [lg](int32_t e) { const float v = lg[e]; return v == v ? v : -INFINITY; };
                std::partial_sort(p->pf_idx.begin(), p->pf_idx.begin() + k, p->pf_idx.begin() + n_exp,
                        [&key](int32_t a, int32_t b) { const float ka = key(a), kb = key(b); return ka > kb || (ka == kb && a < b); });
                std::copy(p->pf_idx.begin(), p->pf_idx.begin() + k, p->pf_topk.begin() + (size_t) t*k);
            }
            n = moe_pool_rank_list(L, p->pf_topk.data(), T, k, n_exp, p->pf_mark, p->pf_list.data());
        }
        p->pf_n = n;
        p->pf_experts.fetch_add((uint64_t) n, std::memory_order_relaxed);
        p->pf_list_gen.store(p->pf_gen, std::memory_order_release); // [TAG_FN_L3_HOST_EXEC] the caller may read the list
    }
    moe_pool_pf_barrier(p, nw);
    if (rec) {
        rec->t1 = ggml_fn_moe_tick();
    }

    moe_pool_prefetch_pull(p, ith, rec);
}

// [TAG_FN_R2_BRIDGE_PF] the caller's side: stop a posted prefetch and wait until every worker has left it
static void moe_pool_prefetch_finish(ggml_cpu_moe_pool * p) {
    if (!p->pf_inflight) {
        return;
    }
    bool stopped = false;
    uint64_t t_stop = 0;
    if (p->n_done.load(std::memory_order_acquire) != p->n_threads - 1) {
        const uint64_t t0 = p->stats ? ggml_fn_moe_tick() : 0;
        p->pf_stop = 1;
        stopped = true;
        p->pf_stopped.fetch_add(1, std::memory_order_relaxed);
        while (p->n_done.load(std::memory_order_acquire) != p->n_threads - 1) {
            moe_pool_relax();
        }
        t_stop = p->stats ? ggml_fn_moe_tick() - t0 : 0;
    }
    p->pf_inflight = false;
    if (p->stats) { // [TAG_FN_L3_CPU_STATS] summed by moe_pool_stats_flush
        p->st_pf_pending = true;
        p->st_pf_stopped = stopped;
        p->st_pf_stop_t  = t_stop;
    }
}

// [TAG_FN_L3_CPU_STATS] the caller, while no worker runs: add the last job's and the last prefetch's records to the window
static void moe_pool_stats_add_job(ggml_cpu_moe_pool * p, int nth, int base, uint64_t t_pub, uint64_t t_done, uint64_t faults);

static void moe_pool_stats_flush(ggml_cpu_moe_pool * p) {
    if (p->st_job_pending) {
        p->st_job_pending = false;
        moe_pool_stats_add_job(p, p->st_nth, p->st_base, p->st_t_pub, p->st_t_done,
                p->st_f_done > p->st_f_pub ? p->st_f_done - p->st_f_pub : 0);
    }
    if (p->st_pf_pending) {
        p->st_pf_pending = false;
        uint64_t t_end = 0, t_router = 0, bytes = 0;
        for (int k = 1; k < p->n_threads; ++k) {
            const moe_pf_rec & r = p->pf_ts[k];
            t_end     = std::max(t_end, r.t2);
            t_router += r.t1 - r.t0;
            bytes    += r.bytes;
        }
        std::lock_guard<std::mutex> lk(p->st_mtx);
        p->st.pf_jobs++;
        p->st.pf_stopped += p->st_pf_stopped ? 1 : 0;
        p->st.pf_t       += (double) (t_end > p->pf_t_post ? t_end - p->pf_t_post : 0);
        p->st.pf_router  += (double) t_router/std::max(1, p->n_threads - 1);
        p->st.pf_bytes   += (double) bytes;
        p->st.pf_experts += (double) p->pf_n;
        p->st.pf_stop    += (double) p->st_pf_stop_t;
        p->st.pf_stop_max = std::max(p->st.pf_stop_max, p->st_pf_stop_t);
    }
}

static void moe_pool_worker(ggml_cpu_moe_pool * p, int ith) {
    pin_current_thread(p->cpus[ith]);
    set_current_thread_prio(p->prio);

    uint32_t last = 0;
    for (;;) {
        uint32_t s = p->seq.load(std::memory_order_acquire);
        if (s == last && !p->stop.load(std::memory_order_acquire)) {
            // spin for spin_us, then sleep until the caller publishes a job
            auto t0 = std::chrono::steady_clock::now();
            uint32_t wg_seen = p->wake_gen.load(std::memory_order_relaxed);
            for (uint32_t k = 1; ; k++) {
                s = p->seq.load(std::memory_order_acquire);
                if (s != last || p->stop.load(std::memory_order_acquire)) {
                    break;
                }
                moe_pool_relax();
                if ((k & 255) == 0) {
                    // [TAG_MOE_BRIDGE] a wake restarts the spin window; a park ends it
                    const uint32_t wg = p->wake_gen.load(std::memory_order_relaxed);
                    if (wg != wg_seen) {
                        wg_seen = wg;
                        t0 = std::chrono::steady_clock::now();
                    }
                    if (!p->parked.load(std::memory_order_relaxed) &&
                        std::chrono::steady_clock::now() - t0 < std::chrono::microseconds(p->spin_us)) {
                        continue;
                    }
                    std::unique_lock<std::mutex> lk(p->mtx);
                    const uint32_t wg_sleep = p->wake_gen.load(std::memory_order_seq_cst);
                    p->n_sleeping.fetch_add(1, std::memory_order_seq_cst);
                    p->cv.wait(lk, [&] {
                        return p->seq.load(std::memory_order_seq_cst) != last || p->stop.load(std::memory_order_seq_cst) ||
                               p->wake_gen.load(std::memory_order_seq_cst) != wg_sleep;
                    });
                    p->n_sleeping.fetch_sub(1, std::memory_order_relaxed);
                    wg_seen = p->wake_gen.load(std::memory_order_relaxed);
                    t0 = std::chrono::steady_clock::now();
                }
            }
        }
        if (p->stop.load(std::memory_order_acquire)) {
            break;
        }
        last = s;
        const int kind = p->kind.load(std::memory_order_acquire);
        if (kind == 1) { // [TAG_FN_R2_BRIDGE_PF]
            moe_pool_prefetch_part(p, ith);
        } else if (kind == 2) {
            ggml_fn_moe_compute(&p->args, ith - 1, p->n_threads - 1, moe_pool_solo_barrier, p);
        } else {
            ggml_fn_moe_compute(&p->args, ith, p->n_threads, moe_pool_barrier, p);
        }
        p->n_done.fetch_add(1, std::memory_order_acq_rel);
    }
}

struct ggml_cpu_moe_pool_params ggml_cpu_moe_pool_params_default(int n_threads) {
    struct ggml_cpu_moe_pool_params p;
    memset(&p, 0, sizeof(p));
    p.n_threads = n_threads;
    p.prio      = GGML_SCHED_PRIO_NORMAL;
    p.spin_us   = 200;
    p.split     = GGML_CPU_MOE_SPLIT_RANGE;
    return p;
}

// Workers spin between jobs, so they must not share CPUs with other spinning threads (a ggml threadpool with --poll
// in a GGML_OPENMP=OFF build: pause it with ggml_threadpool_pause while the pool runs, or give the two disjoint CPUs);
// measured: a shared CPU turns every barrier into a scheduler time slice (~64 ms per job instead of ~0.5 ms).
struct ggml_cpu_moe_pool * ggml_cpu_moe_pool_new(const struct ggml_cpu_moe_pool_params * pp) {
    ggml_cpu_init();
    if (pp == nullptr || pp->n_threads < 1 || pp->n_threads > GGML_MAX_N_THREADS) {
        return nullptr;
    }
    auto * p = new ggml_cpu_moe_pool();
    p->n_threads = pp->n_threads;
    p->spin_us   = std::max(0, pp->spin_us);
    p->prio      = pp->prio;
    p->split     = std::min(std::max(pp->split, (int) GGML_CPU_MOE_SPLIT_RANGE), (int) GGML_CPU_MOE_SPLIT_STEAL); // [TAG_FN_L3_CPU_SPLIT]
    p->pf_rank   = pp->pf_rank;
    p->stats     = pp->stats;
    p->swpf      = std::min(std::max(pp->swpf, 0), 64); // [TAG_FN_L3_CPU_SWPF]
    p->pf_streams = std::min(std::max(pp->pf_streams, 1), 8); // [TAG_FN_L3_CPU_PFSTREAMS]
    p->dflow     = pp->dflow;                                // [TAG_FN_L4_MEM_POOLBAR]
    p->pf_fix    = pp->pf_fix;                               // [TAG_FN_L4_MEM_PFDEV]
    p->pf_score  = pp->pf_score;                             // [TAG_FN_L6_PF]
    p->pf_pull   = pp->pf_pull & (GGML_FN_PF_FINE_STOP | GGML_FN_PF_VEC);
    p->pf_cap    = std::max(0, pp->pf_cap);
    p->pf_fresh  = std::max(0, pp->pf_fresh);

    std::vector<int> list;
    for (int i = 0; i < GGML_MAX_N_THREADS; i++) {
        if (pp->cpumask[i]) {
            list.push_back(i);
        }
    }
    if (list.empty()) {
        list = physical_core_cpus();
    }
    // [TAG_MOE_BRIDGE]
    if (pp->skip_first_core && list.size() > 1) {
        list.erase(list.begin());
        p->n_threads = std::min<int>(p->n_threads, (int) list.size());
    }
    // [TAG_FN_L3_CPU_PLACE] worker 0 on its own CPU (e.g. the SMT sibling of the main thread's core): the workers take
    // the rest of the list from its start, so a core the executor held computes now
    const bool caller_set = pp->pin_caller && pp->caller_cpu1 > 0 && pp->caller_cpu1 <= GGML_MAX_N_THREADS;
    if (caller_set) {
        const int cc = pp->caller_cpu1 - 1;
        list.erase(std::remove(list.begin(), list.end(), cc), list.end());
        list.insert(list.begin(), cc);
        p->n_threads = std::min<int>(pp->n_threads, (int) list.size());
    }
    // worker 0 is the caller: if it is pinned to one CPU of the list now (the ggml threadpool of a GGML_OPENMP=OFF build
    // pins the main thread to the last CPU of its mask), that CPU moves to the front so no worker shares it
    const int caller_cpu = caller_set ? -1 : current_thread_single_cpu();
    for (size_t i = 1; i < list.size(); i++) {
        if (list[i] == caller_cpu) {
            std::swap(list[0], list[i]);
            break;
        }
    }
    p->cpus.assign(p->n_threads, -1);
    for (int k = 1; k < p->n_threads; k++) {
        p->cpus[k] = k < (int) list.size() ? list[k] : -1;
    }
    // [TAG_MOE_BRIDGE] a dedicated executor thread runs the jobs: it takes the first CPU of the list itself
    if (pp->pin_caller && !list.empty()) {
        p->cpus[0] = list[0];
        pin_current_thread(list[0]);
        set_current_thread_prio(p->prio);
    }
    if (p->stats) { // [TAG_FN_L3_CPU_STATS]
        p->ts.resize(p->n_threads);
        p->pf_ts.resize(p->n_threads);
        memset((void *) p->ts.data(), 0, sizeof(moe_ts_rec)*p->ts.size());
        memset((void *) p->pf_ts.data(), 0, sizeof(moe_pf_rec)*p->pf_ts.size());
        const std::vector<int> d = l3_domains(GGML_MAX_N_THREADS);
        p->dom.assign(p->n_threads, 0);
        for (int k = 0; k < p->n_threads; k++) {
            p->dom[k] = p->cpus[k] >= 0 && p->cpus[k] < (int) d.size() ? std::min(3, d[p->cpus[k]]) : 0;
        }
        const auto     t0 = std::chrono::steady_clock::now();
        const uint64_t k0 = ggml_fn_moe_tick();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        const uint64_t k1 = ggml_fn_moe_tick();
        const double   us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        p->tick_per_us = us > 0.0 && k1 > k0 ? (double) (k1 - k0)/us : 1.0;
    }
    for (int k = 1; k < p->n_threads; k++) {
        p->workers.emplace_back(moe_pool_worker, p, k);
    }
    return p;
}

void ggml_cpu_moe_pool_free(struct ggml_cpu_moe_pool * p) {
    if (p == nullptr) {
        return;
    }
    moe_pool_prefetch_finish(p); // [TAG_FN_R2_BRIDGE_PF]
    p->stop.store(true, std::memory_order_seq_cst);
    {
        std::lock_guard<std::mutex> lk(p->mtx);
    }
    p->cv.notify_all();
    for (auto & t : p->workers) {
        t.join();
    }
    delete p;
}

void ggml_cpu_moe_pool_park(struct ggml_cpu_moe_pool * p) {
    if (p) {
        p->parked.store(true, std::memory_order_relaxed);
    }
}

void ggml_cpu_moe_pool_wake(struct ggml_cpu_moe_pool * p) {
    if (p == nullptr) {
        return;
    }
    p->parked.store(false, std::memory_order_relaxed);
    p->wake_gen.fetch_add(1, std::memory_order_seq_cst);
    if (p->n_sleeping.load(std::memory_order_seq_cst) > 0) {
        {
            std::lock_guard<std::mutex> lk(p->mtx);
        }
        p->cv.notify_all();
    }
}

bool ggml_cpu_moe_layer_supported(const struct ggml_cpu_moe_layer * l) {
    ggml_cpu_init();
    return l != nullptr && ggml_fn_moe_supported(l->up, l->gate, l->down);
}

// [TAG_FN_L3_CPU_STATS] the caller, after a job: add the compute threads' records to the window
static void moe_pool_stats_add_job(ggml_cpu_moe_pool * p, int nth, int base, uint64_t t_pub, uint64_t t_done, uint64_t faults) {
    double lag = 0, p3 = 0, w3 = 0, p4 = 0, w4 = 0, p5 = 0, bytes = 0, taken = 0, pfhit = 0;
    uint64_t e3_min = UINT64_MAX, e3_max = 0, e4_min = UINT64_MAX, e4_max = 0;
    double dom_bytes[4] = {}, dom_busy[4] = {};
    int    dom_threads[4] = {};
    for (int c = 0; c < nth; ++c) {
        const uint64_t * r = p->ts[c].v;
        lag   += (double) (r[GGML_FN_MOE_TS_START] > t_pub ? r[GGML_FN_MOE_TS_START] - t_pub : 0);
        p3    += (double) (r[GGML_FN_MOE_TS_P3]  - r[GGML_FN_MOE_TS_START]);
        w3    += (double) (r[GGML_FN_MOE_TS_B1]  - r[GGML_FN_MOE_TS_P3]);
        p4    += (double) (r[GGML_FN_MOE_TS_P4]  - r[GGML_FN_MOE_TS_B1]);
        w4    += (double) (r[GGML_FN_MOE_TS_B2]  - r[GGML_FN_MOE_TS_P4]);
        p5    += (double) (r[GGML_FN_MOE_TS_END] - r[GGML_FN_MOE_TS_B2]);
        bytes += (double) r[GGML_FN_MOE_TS_BYTES];
        taken += (double) r[GGML_FN_MOE_TS_TAKEN];
        pfhit += (double) r[GGML_FN_MOE_TS_PFHIT];
        e3_min = std::min(e3_min, r[GGML_FN_MOE_TS_P3]);
        e3_max = std::max(e3_max, r[GGML_FN_MOE_TS_P3]);
        e4_min = std::min(e4_min, r[GGML_FN_MOE_TS_P4]);
        e4_max = std::max(e4_max, r[GGML_FN_MOE_TS_P4]);
        const int d = p->dom[base + c];
        dom_bytes[d]   += (double) r[GGML_FN_MOE_TS_BYTES];
        dom_busy[d]    += (double) (r[GGML_FN_MOE_TS_P3] - r[GGML_FN_MOE_TS_START]) + (double) (r[GGML_FN_MOE_TS_P4] - r[GGML_FN_MOE_TS_B1]);
        dom_threads[d] += 1;
    }
    std::lock_guard<std::mutex> lk(p->st_mtx);
    moe_stats_acc & s = p->st;
    if (s.thr_busy.size() != (size_t) p->n_threads) {
        s.thr_busy.assign(p->n_threads, 0.0);
        s.thr_bytes.assign(p->n_threads, 0.0);
    }
    for (int c = 0; c < nth && base + c < p->n_threads; ++c) {
        const uint64_t * r = p->ts[c].v;
        s.thr_busy[base + c]  += (double) (r[GGML_FN_MOE_TS_P3] - r[GGML_FN_MOE_TS_START]) + (double) (r[GGML_FN_MOE_TS_P4] - r[GGML_FN_MOE_TS_B1]);
        s.thr_bytes[base + c] += (double) r[GGML_FN_MOE_TS_BYTES];
    }
    const uint64_t t_job = t_done > t_pub ? t_done - t_pub : 0;
    s.job_max    = std::max(s.job_max, t_job);
    s.jobs_slow += (double) t_job > 1000.0*p->tick_per_us ? 1 : 0;
    s.flt       += (double) faults; // [TAG_FN_L7_FAULTS]
    if ((double) t_job > 1000.0*p->tick_per_us) {
        // [TAG_FN_L7_SLOW] the thread that ended last and its phases against the median thread's
        if (s.slow_last.size() != (size_t) p->n_threads) {
            s.slow_last.assign(p->n_threads, 0);
        }
        int last = 0;
        std::vector<double> lagv, p3v, p4v, p5v;
        for (int c = 0; c < nth; ++c) {
            const uint64_t * r = p->ts[c].v;
            if (r[GGML_FN_MOE_TS_END] > p->ts[last].v[GGML_FN_MOE_TS_END]) {
                last = c;
            }
            lagv.push_back((double) (r[GGML_FN_MOE_TS_START] > t_pub ? r[GGML_FN_MOE_TS_START] - t_pub : 0));
            p3v.push_back((double) (r[GGML_FN_MOE_TS_P3]  - r[GGML_FN_MOE_TS_START]));
            p4v.push_back((double) (r[GGML_FN_MOE_TS_P4]  - r[GGML_FN_MOE_TS_B1]));
            p5v.push_back((double) (r[GGML_FN_MOE_TS_END] - r[GGML_FN_MOE_TS_B2]));
        }
        auto med = [](std::vector<double> v) { std::nth_element(v.begin(), v.begin() + v.size()/2, v.end()); return v[v.size()/2]; };
        if (base + last < p->n_threads) {
            s.slow_last[base + last]++;
        }
        const uint64_t * rl = p->ts[last].v;
        s.slow_x_lag += lagv[last] - med(lagv);
        s.slow_x_p3  += p3v[last]  - med(p3v);
        s.slow_x_p4  += p4v[last]  - med(p4v);
        s.slow_x_p5  += p5v[last]  - med(p5v);
        s.slow_x_w   += (double) (rl[GGML_FN_MOE_TS_B1] - rl[GGML_FN_MOE_TS_P3]) + (double) (rl[GGML_FN_MOE_TS_B2] - rl[GGML_FN_MOE_TS_P4]);
        s.flt_slow     += (double) faults;
        s.slow_t       += (double) t_job;
        s.slow_nofault += faults == 0 ? 1 : 0;
    } else {
        s.fast_fault   += faults > 0 ? 1 : 0;
    }
    s.jobs++;
    s.job     += (double) (t_done - t_pub);
    s.bytes   += bytes;
    s.experts += (double) p->ts[0].v[GGML_FN_MOE_TS_NACT];
    s.lag     += lag/nth;
    s.p3      += p3/nth;
    s.w3      += w3/nth;
    s.p4      += p4/nth;
    s.w4      += w4/nth;
    s.p5      += p5/nth;
    s.spread3 += (double) (e3_max - e3_min);
    s.spread4 += (double) (e4_max - e4_min);
    s.taken   += taken;
    s.pfhit   += pfhit;
    for (int d = 0; d < 4; ++d) {
        s.dom_bytes[d]  += dom_bytes[d];
        s.dom_busy[d]   += dom_busy[d];
        s.dom_threads[d] = std::max(s.dom_threads[d], dom_threads[d]);
    }
}

enum ggml_status ggml_cpu_moe_run(struct ggml_cpu_moe_pool * p, const struct ggml_cpu_moe_job * job) {
    if (p == nullptr || job == nullptr || job->layer == nullptr || job->x == nullptr || job->ids == nullptr || job->out == nullptr) {
        return GGML_STATUS_FAILED;
    }
    const ggml_cpu_moe_layer * l = job->layer;
    if (job->n_tokens < 1 || job->n_tokens > GGML_FN_MOE_MAX_T || job->n_used < 1 || job->n_used > l->up->ne[2] ||
        !ggml_fn_moe_supported(l->up, l->gate, l->down)) {
        return GGML_STATUS_FAILED;
    }
    const int64_t n_embd = l->up->ne[0];
    const int     T      = job->n_tokens;
    const int     n_used = job->n_used;

    moe_pool_prefetch_finish(p); // [TAG_FN_R2_BRIDGE_PF] the workers leave a prefetch before the job (and its buffers)

    const size_t need = ggml_fn_moe_work_size(l->up, l->gate, l->down, n_used, T, p->n_threads);
    if (p->work.size() < need) {
        p->work.resize(need); // the workers are idle between jobs
    }

    ggml_fn_moe_args & a = p->args;
    memset(&a, 0, sizeof(a));
    a.up         = l->up;
    a.gate       = l->gate;
    a.down       = l->down;
    a.table      = l->table;
    a.table_miss = l->table_miss;
    a.n_tokens   = T;
    a.n_used     = n_used;
    a.x          = (const char *) job->x;
    a.x_nb       = (size_t) n_embd*sizeof(float);
    a.ids        = (const char *) job->ids;
    a.ids_nb0    = sizeof(int32_t);
    a.ids_nb1    = (size_t) n_used*sizeof(int32_t);
    if (job->w) {
        a.w       = job->w;
        a.out_sum = job->out;
    } else {
        a.out     = (char *) job->out;
        a.out_nb1 = (size_t) n_embd*sizeof(float);
        a.out_nb2 = (size_t) n_embd*n_used*sizeof(float);
    }
    a.wdata = p->work.data();

    const bool solo = p->solo && p->n_threads >= 2; // [TAG_FN_R2_BRIDGE_PF]
    const int  nth  = solo ? p->n_threads - 1 : p->n_threads;

    // [TAG_FN_L3_CPU_SPLIT] the split, the claim words of a stealing job, and the last prefetch's order when it
    // predicted this layer
    a.split = nth >= 2 ? p->split : GGML_CPU_MOE_SPLIT_RANGE;
    a.swpf  = p->swpf; // [TAG_FN_L3_CPU_SWPF]
    if (a.split >= GGML_CPU_MOE_SPLIT_STEAL) {
        const size_t words = (size_t) GGML_FN_MOE_CLAIM_WORDS(n_used*T, ggml_fn_moe_np_max(l->up, l->down), nth);
        if (p->claim.size() < words) {
            p->claim.resize(words, 0);
        }
        if (++p->epoch > (uint32_t) INT32_MAX) {
            std::fill(p->claim.begin(), p->claim.end(), 0); // no word may still hold the epoch that comes back
            p->epoch = 1;
        }
        a.claim = p->claim.data();
        a.epoch = (int32_t) p->epoch;
    }
    const bool pf_mine = p->pf_up != nullptr && p->pf_up == l->up;
    if (p->stats && pf_mine && p->pf_n > 0 && p->pf_mark.size() >= (size_t) l->up->ne[2]) {
        // [TAG_FN_L3_CPU_STATS] the prediction against this job: the predicted experts it computes (the first 4 apart)
        auto & mk = p->pf_mark;
        std::fill(mk.begin(), mk.end(), (uint8_t) 0);
        for (int i = 0; i < p->pf_n; ++i) {
            mk[p->pf_list[i]] = i < 4 ? 3 : 1;
        }
        int n_job = 0, n_hit = 0, n_hit4 = 0;
        for (int i = 0; i < n_used*T; ++i) {
            const int32_t e = job->ids[i];
            if (e < 0 || e >= l->up->ne[2] || (l->table != nullptr && l->table[e] != l->table_miss) || (mk[e] & 4)) {
                continue;
            }
            mk[e] |= 4;
            n_job++;
            n_hit  += mk[e] & 1;
            n_hit4 += (mk[e] & 2) ? 1 : 0;
        }
        std::lock_guard<std::mutex> lk(p->st_mtx);
        p->st.pred_jobs++;
        p->st.pred_n    += p->pf_n;
        p->st.pred_job  += n_job;
        p->st.pred_hit  += n_hit;
        p->st.pred_hit4 += n_hit4;
        p->st.pred_top4 += std::min(4, p->pf_n);
    }
    if (a.split >= GGML_CPU_MOE_SPLIT_STABLE && pf_mine && p->pf_n > 0) {
        a.prio   = p->pf_list.data();
        a.n_prio = p->pf_n;
    }
    if (p->stats) { // [TAG_FN_L3_CPU_STATS]
        a.ts = &p->ts[0].v[0];
        if (a.split >= GGML_CPU_MOE_SPLIT_STABLE && pf_mine && p->pf_done_valid && p->pf_done_n_exp == l->up->ne[2]) {
            a.pf_done = p->pf_done.data();
        }
    }
    if (p->dflow) { // [TAG_FN_L4_MEM_POOLBAR] zero before the post below publishes them
        const size_t words = (size_t) ggml_fn_moe_dflow_words(l->up, n_used, T);
        if (p->dfl.size() < words + GGML_FN_MOE_DFLOW_STRIDE) {
            p->dfl.resize(words + GGML_FN_MOE_DFLOW_STRIDE);
        }
        int32_t * d = p->dfl.data();
        d += ((64 - ((uintptr_t) d & 63)) & 63)/sizeof(int32_t);
        memset(d, 0, words*sizeof(int32_t));
        a.dflow = d;
    }
    // [TAG_FN_L6_PF] learn from this job after it (off its path)
    const bool score_learn = p->pf_score && pf_mine && p->pf_gT > 0 && p->pf_mark.size() >= (size_t) l->up->ne[2];
    p->pf_up = nullptr; // a later job of this layer (the next step) is not predicted by this prefetch
    p->pf_done_valid = false;
    if (p->stats) {
        moe_pool_stats_flush(p); // usually done already, after the last job (ggml_cpu_moe_prefetch)
    }

    const uint64_t f_pub = p->stats ? moe_proc_faults() : 0; // [TAG_FN_L7_FAULTS]
    const uint64_t t_pub = p->stats ? ggml_fn_moe_tick() : 0;
    p->kind.store(solo ? 2 : 0, std::memory_order_relaxed);
    p->n_done.store(0, std::memory_order_relaxed);
    p->parked.store(false, std::memory_order_relaxed); // [TAG_MOE_BRIDGE]
    p->seq.fetch_add(1, std::memory_order_seq_cst);
    if (p->n_sleeping.load(std::memory_order_seq_cst) > 0) {
        {
            std::lock_guard<std::mutex> lk(p->mtx);
        }
        p->cv.notify_all();
    }

    if (!solo) {
        ggml_fn_moe_compute(&a, 0, p->n_threads, moe_pool_barrier, p);
    }

    while (p->n_done.load(std::memory_order_acquire) != p->n_threads - 1) {
        moe_pool_relax();
    }
    if (score_learn) {
        moe_pool_score_learn(p, l, job->ids, n_used*T, l->up->ne[2]);
    }
    if (p->pf_fresh > 0) { // [TAG_FN_L6_PF] the experts this job read on the CPU (their pages are resident now)
        auto & v = p->pf_seen[l->up];
        v.resize((size_t) l->up->ne[2], 0);
        const uint32_t clk = ++p->pf_clock;
        for (int i = 0; i < n_used*T; ++i) {
            const int32_t e = job->ids[i];
            if (e >= 0 && e < l->up->ne[2] && !(l->table != nullptr && l->table[e] != l->table_miss)) {
                v[e] = clk;
            }
        }
    }
    p->pf_gT = 0;
    if (p->stats) {
        p->st_job_pending = true;
        p->st_t_pub       = t_pub;
        p->st_t_done      = ggml_fn_moe_tick();
        p->st_f_pub       = f_pub;
        p->st_f_done      = moe_proc_faults();
        p->st_nth         = nth;
        p->st_base        = solo ? 1 : 0;
    }
    return GGML_STATUS_SUCCESS;
}

// [TAG_FN_R2_BRIDGE_PF]
enum ggml_status ggml_cpu_moe_prefetch(struct ggml_cpu_moe_pool * p, const struct ggml_cpu_moe_prefetch_job * job) {
    if (p == nullptr || job == nullptr || job->layer == nullptr || (job->list == nullptr && (job->router == nullptr || job->x == nullptr)) ||
        p->n_threads < 2 || job->n_tokens < 1 || job->n_tokens > GGML_FN_MOE_MAX_T || job->k < 1) {
        return GGML_STATUS_FAILED;
    }
    const ggml_cpu_moe_layer * l = job->layer;
    if (!ggml_fn_moe_supported(l->up, l->gate, l->down)) {
        return GGML_STATUS_FAILED;
    }
    moe_pool_prefetch_finish(p);
    if (p->stats) {
        moe_pool_stats_flush(p); // [TAG_FN_L3_CPU_STATS] the last job's and prefetch's records, before the workers reuse them
    }

    const int64_t n_embd = l->up->ne[0];
    const int64_t n_exp  = l->up->ne[2];
    const int     T      = job->n_tokens;
    p->pf = *job;
    p->pf.x = nullptr; // copied below, in the router's dot type
    p->pf.list = nullptr; // [TAG_FN_L3_CPU_DEVPRED] ranked below
    p->pf_given = job->list != nullptr;
    if (!p->pf_given) {
        p->pf_xh.resize((size_t) n_embd*T);
        ggml_cpu_fp32_to_fp16(job->x, p->pf_xh.data(), n_embd*T);
        p->pf_logits.resize((size_t) n_exp*T);
    }
    p->pf_list.resize((size_t) n_exp);
    p->pf_idx.resize((size_t) n_exp);
    p->pf_mark.resize((size_t) n_exp);
    p->pf_n    = 0;
    p->pf_stop = 0;
    p->pf_gen++; // [TAG_FN_L3_HOST_EXEC] published to the workers by the seq increment below
    p->pf_jobs.fetch_add(1, std::memory_order_relaxed);
    if (p->pf_given) {
        // [TAG_FN_L3_CPU_DEVPRED] the workers only pull: the given list is ranked here, on the caller
        if (p->pf_score) { // [TAG_FN_L6_PF]
            p->pf_gt.assign(job->list, job->list + (size_t) T*job->k);
            p->pf_gT = T;
            p->pf_gk = job->k;
            p->pf_n  = moe_pool_score_list(p, l, job->list, T, job->k, n_exp, p->pf_list.data());
        } else {
            p->pf_n = moe_pool_rank_list(l, job->list, T, job->k, n_exp, p->pf_mark, p->pf_list.data());
        }
        if (p->pf_fresh > 0) { // [TAG_FN_L6_PF] only experts a recent job read (no page-in of a mispredicted expert)
            const auto it = p->pf_seen.find(l->up);
            int m = 0;
            for (int i = 0; i < p->pf_n; ++i) {
                const int32_t e = p->pf_list[i];
                const uint32_t s = it != p->pf_seen.end() && (size_t) e < it->second.size() ? it->second[e] : 0;
                if (s != 0 && p->pf_clock - s <= (uint32_t) p->pf_fresh) {
                    p->pf_list[m++] = e;
                }
            }
            p->pf_n = m;
        }
        if (p->pf_cap > 0 && p->pf_n > p->pf_cap) {
            p->pf_n = p->pf_cap; // [TAG_FN_L6_PF]
        }
        p->pf_experts.fetch_add((uint64_t) p->pf_n, std::memory_order_relaxed);
        if (p->pf_fix) {
            p->pf_list_gen.store(p->pf_gen, std::memory_order_release); // [TAG_FN_L4_MEM_PFDEV] no router phase publishes it
        }
    } else if (p->pf_rank) {
        p->pf_topk.resize((size_t) T*std::min<int64_t>(job->k, n_exp)); // [TAG_FN_L3_CPU_PFRANK]
    }
    p->pf_up = l->up; // [TAG_FN_L3_CPU_SPLIT] the next job of this layer computes the listed experts first
    if (p->stats) { // [TAG_FN_L3_CPU_STATS]
        if (p->split >= GGML_CPU_MOE_SPLIT_STABLE) {
            p->pf_done.assign((size_t) p->n_threads*n_exp, 0);
            p->pf_done_n_exp = n_exp;
            p->pf_done_valid = true;
        }
        p->pf_t_post = ggml_fn_moe_tick();
    }

    p->kind.store(1, std::memory_order_relaxed);
    p->n_done.store(0, std::memory_order_relaxed);
    p->parked.store(false, std::memory_order_relaxed);
    p->seq.fetch_add(1, std::memory_order_seq_cst);
    if (p->n_sleeping.load(std::memory_order_seq_cst) > 0) {
        {
            std::lock_guard<std::mutex> lk(p->mtx);
        }
        p->cv.notify_all();
    }
    p->pf_inflight = true;
    return GGML_STATUS_SUCCESS;
}

void ggml_cpu_moe_prefetch_stop(struct ggml_cpu_moe_pool * p) {
    if (p) {
        p->pf_stop = 1;
    }
}

void ggml_cpu_moe_pool_set_solo(struct ggml_cpu_moe_pool * p, bool solo) {
    if (p) {
        moe_pool_prefetch_finish(p);
        p->solo = solo && p->n_threads >= 2;
    }
}

// [TAG_FN_L3_HOST_EXEC] the caller computes as thread 0 of n_threads (not solo), so it pulls its own pieces: the
// workers pull theirs in moe_pool_prefetch_part with the same split. The list is the one worker 1 published for the
// prefetch posted last; nothing writes it until the caller posts the next prefetch or job (it returns before both).
size_t ggml_cpu_moe_prefetch_caller(struct ggml_cpu_moe_pool * p, const volatile int32_t * stop) {
    if (p == nullptr || stop == nullptr || !p->pf_inflight || p->solo || p->n_threads < 2) {
        return 0;
    }
    p->pfc_calls.fetch_add(1, std::memory_order_relaxed);
    const uint32_t gen = p->pf_gen;
    while (p->pf_list_gen.load(std::memory_order_acquire) != gen) {
        if (*stop) {
            p->pfc_stopped.fetch_add(1, std::memory_order_relaxed);
            return 0;
        }
        moe_pool_relax();
    }
    const ggml_cpu_moe_layer * L = p->pf.layer;
    const int n = p->pf_n;
    size_t b = 0;
    if (n > 0 && L != nullptr && !*stop) {
        if (p->pf_fix && p->split >= GGML_CPU_MOE_SPLIT_STABLE) {
            // [TAG_FN_L4_MEM_PFDEV] the pieces compute thread 0 owns in a split >= 1 job, as the workers' pull takes theirs
            b = ggml_fn_moe_prefetch_stable(L->up, L->gate, L->down, p->pf_list.data(), n, 0, p->n_threads, stop, p->pf.mode,
                    nullptr, p->pf_streams, p->pf_pull);
        } else {
            b = ggml_fn_moe_prefetch(L->up, L->gate, L->down, p->pf_list.data(), n, 0, p->n_threads, stop, p->pf.mode);
        }
    }
    if (*stop) {
        p->pfc_stopped.fetch_add(1, std::memory_order_relaxed);
    }
    p->pfc_bytes.fetch_add((uint64_t) b, std::memory_order_relaxed);
    return b;
}

void ggml_cpu_moe_prefetch_caller_stats(struct ggml_cpu_moe_pool * p, uint64_t * calls, uint64_t * stopped, uint64_t * bytes) {
    if (calls) {
        *calls = p ? p->pfc_calls.load() : 0;
    }
    if (stopped) {
        *stopped = p ? p->pfc_stopped.load() : 0;
    }
    if (bytes) {
        *bytes = p ? p->pfc_bytes.load() : 0;
    }
}

void ggml_cpu_moe_prefetch_stats(struct ggml_cpu_moe_pool * p, uint64_t * jobs, uint64_t * stopped, uint64_t * bytes, uint64_t * experts) {
    if (jobs) {
        *jobs = p ? p->pf_jobs.load() : 0;
    }
    if (stopped) {
        *stopped = p ? p->pf_stopped.load() : 0;
    }
    if (bytes) {
        *bytes = p ? p->pf_bytes.load() : 0;
    }
    if (experts) {
        *experts = p ? p->pf_experts.load() : 0;
    }
}

// [TAG_FN_L3_CPU_STATS]
void ggml_cpu_moe_pool_get_stats(struct ggml_cpu_moe_pool * p, struct ggml_cpu_moe_pool_stats * out, bool reset) {
    if (out) {
        memset(out, 0, sizeof(*out));
    }
    if (p == nullptr || !p->stats) {
        return;
    }
    std::lock_guard<std::mutex> lk(p->st_mtx);
    const moe_stats_acc & s = p->st;
    if (out) {
        const double nj = s.jobs > 0 ? (double) s.jobs : 1.0;
        const double us = p->tick_per_us;
        out->jobs       = s.jobs;
        out->job_us     = s.job/nj/us;
        out->mib        = s.bytes/nj/1048576.0;
        out->experts    = s.experts/nj;
        out->lag_us     = s.lag/nj/us;
        out->p3_us      = s.p3/nj/us;
        out->w3_us      = s.w3/nj/us;
        out->p4_us      = s.p4/nj/us;
        out->w4_us      = s.w4/nj/us;
        out->p5_us      = s.p5/nj/us;
        out->spread3_us = s.spread3/nj/us;
        out->spread4_us = s.spread4/nj/us;
        out->taken      = s.taken/nj;
        out->pf_hit_mib = s.pfhit/nj/1048576.0;
        for (int d = 0; d < 4; ++d) {
            if (s.dom_threads[d] > 0) {
                out->n_dom = d + 1;
            }
            out->dom_threads[d] = s.dom_threads[d];
            out->dom_gbs[d]     = s.dom_busy[d] > 0.0 ? s.dom_bytes[d]/(s.dom_busy[d]/us)/1e3 : 0.0;
        }
        const double np = s.pf_jobs > 0 ? (double) s.pf_jobs : 1.0;
        out->pf_jobs      = s.pf_jobs;
        out->pf_us        = s.pf_t/np/us;
        out->pf_router_us = s.pf_router/np/us;
        out->pf_mib       = s.pf_bytes/np/1048576.0;
        out->pf_experts   = s.pf_experts/np;
        out->pf_stopped   = (double) s.pf_stopped/np;
        out->job_max_us     = (double) s.job_max/us;
        out->jobs_slow      = s.jobs_slow;
        out->flt_job        = s.flt/nj; // [TAG_FN_L7_FAULTS]
        out->flt_slow       = s.jobs_slow ? s.flt_slow/(double) s.jobs_slow : 0.0;
        out->slow_us        = s.jobs_slow ? s.slow_t/(double) s.jobs_slow/us : 0.0;
        out->slow_nofault   = s.slow_nofault;
        out->fast_fault     = s.fast_fault;
        if (s.jobs_slow > 0) { // [TAG_FN_L7_SLOW]
            const double ns = (double) s.jobs_slow;
            out->slow_x_lag_us = s.slow_x_lag/ns/us;
            out->slow_x_p3_us  = s.slow_x_p3/ns/us;
            out->slow_x_p4_us  = s.slow_x_p4/ns/us;
            out->slow_x_p5_us  = s.slow_x_p5/ns/us;
            out->slow_w_us     = s.slow_x_w/ns/us;
            for (int k = 0; k < 3; ++k) {
                out->slow_cpu[k] = -1;
            }
            std::vector<int> ord;
            for (int k = 0; k < (int) s.slow_last.size(); ++k) {
                if (s.slow_last[k] > 0) {
                    ord.push_back(k);
                }
            }
            std::sort(ord.begin(), ord.end(), [&s](int a, int b) { return s.slow_last[a] > s.slow_last[b]; });
            for (int k = 0; k < 3 && k < (int) ord.size(); ++k) {
                out->slow_cpu[k]   = ord[k] < (int) p->cpus.size() ? p->cpus[ord[k]] : -1;
                out->slow_cpu_n[k] = s.slow_last[ord[k]];
            }
        }
        out->pf_stop_us     = s.pf_stop/(s.pf_stopped > 0 ? (double) s.pf_stopped : 1.0)/us;
        out->pf_stop_max_us = (double) s.pf_stop_max/us;
        out->pred_jobs      = s.pred_jobs;
        out->pf_precision   = s.pred_n    > 0.0 ? s.pred_hit /s.pred_n    : 0.0;
        out->pf_prec_top4   = s.pred_top4 > 0.0 ? s.pred_hit4/s.pred_top4 : 0.0;
        out->pf_recall      = s.pred_job  > 0.0 ? s.pred_hit /s.pred_job  : 0.0;
        const int rk[4] = { 0, 1, 3, 7 }; // [TAG_FN_L6_PF] (read without the job's lock: statistics only)
        for (int i = 0; i < 4; ++i) {
            out->pf_rank_p[i] = p->rk_n[rk[i]] > 0.0 ? p->rk_hit[rk[i]]/p->rk_n[rk[i]] : 0.0;
        }
        out->n_thr = std::min<int>(p->n_threads, GGML_CPU_MOE_STATS_MAX_THR);
        for (int k = 0; k < out->n_thr; ++k) {
            const double busy = k < (int) s.thr_busy.size() ? s.thr_busy[k] : 0.0;
            out->thr_cpu[k]     = k < (int) p->cpus.size() ? p->cpus[k] : -1;
            out->thr_busy_us[k] = busy/nj/us;
            out->thr_gbs[k]     = busy > 0.0 ? s.thr_bytes[k]/(busy/us)/1e3 : 0.0;
        }
    }
    if (reset) {
        p->st = moe_stats_acc();
    }
}
