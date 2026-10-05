// [TAG_FN_CPU_MOE_FUSE] persistent CPU worker pool for the host experts of one MoE layer (ggml_cpu_moe_run).
// The calling thread is worker 0; the other workers spin for spin_us after a job, then sleep until the next one.
// The work itself is ggml_fn_moe_compute (ggml-cpu.c), the same kernel as the fused graph op.

#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <pthread.h>
#    include <sched.h>
#    include <unistd.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#    include <immintrin.h>
#endif

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
    std::vector<int32_t>      pf_list;             // the predicted experts the table does not serve, ascending
    std::vector<int32_t>      pf_idx;              // selection scratch
    std::vector<uint8_t>      pf_mark;
    int                       pf_n = 0;            // entries of pf_list (worker 1 writes it before the second barrier)
    std::atomic<uint64_t>     pf_jobs{0};
    std::atomic<uint64_t>     pf_stopped{0};
    std::atomic<uint64_t>     pf_bytes{0};
    std::atomic<uint64_t>     pf_experts{0};
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
        if (!p->pf_stop) {
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
        p->pf_n = n;
        p->pf_experts.fetch_add((uint64_t) n, std::memory_order_relaxed);
    }
    moe_pool_pf_barrier(p, nw);

    // the pieces this worker computes in a job of the predicted experts: thread ith of n_threads, or (solo) thread
    // ith - 1 of the n_threads - 1 workers
    const int c_ith = p->solo ? ith - 1 : ith;
    const int c_nth = p->solo ? p->n_threads - 1 : p->n_threads;
    const size_t b = ggml_fn_moe_prefetch(L->up, L->gate, L->down, p->pf_list.data(), p->pf_n, c_ith, c_nth, &p->pf_stop,
            J.mode);
    p->pf_bytes.fetch_add((uint64_t) b, std::memory_order_relaxed);
}

// [TAG_FN_R2_BRIDGE_PF] the caller's side: stop a posted prefetch and wait until every worker has left it
static void moe_pool_prefetch_finish(ggml_cpu_moe_pool * p) {
    if (!p->pf_inflight) {
        return;
    }
    if (p->n_done.load(std::memory_order_acquire) != p->n_threads - 1) {
        p->pf_stop = 1;
        p->pf_stopped.fetch_add(1, std::memory_order_relaxed);
        while (p->n_done.load(std::memory_order_acquire) != p->n_threads - 1) {
            moe_pool_relax();
        }
    }
    p->pf_inflight = false;
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
    // worker 0 is the caller: if it is pinned to one CPU of the list now (the ggml threadpool of a GGML_OPENMP=OFF build
    // pins the main thread to the last CPU of its mask), that CPU moves to the front so no worker shares it
    const int caller_cpu = current_thread_single_cpu();
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
    return GGML_STATUS_SUCCESS;
}

// [TAG_FN_R2_BRIDGE_PF]
enum ggml_status ggml_cpu_moe_prefetch(struct ggml_cpu_moe_pool * p, const struct ggml_cpu_moe_prefetch_job * job) {
    if (p == nullptr || job == nullptr || job->layer == nullptr || job->router == nullptr || job->x == nullptr ||
        p->n_threads < 2 || job->n_tokens < 1 || job->n_tokens > GGML_FN_MOE_MAX_T || job->k < 1) {
        return GGML_STATUS_FAILED;
    }
    const ggml_cpu_moe_layer * l = job->layer;
    if (!ggml_fn_moe_supported(l->up, l->gate, l->down)) {
        return GGML_STATUS_FAILED;
    }
    moe_pool_prefetch_finish(p);

    const int64_t n_embd = l->up->ne[0];
    const int64_t n_exp  = l->up->ne[2];
    const int     T      = job->n_tokens;
    p->pf = *job;
    p->pf.x = nullptr; // copied below, in the router's dot type
    p->pf_xh.resize((size_t) n_embd*T);
    ggml_cpu_fp32_to_fp16(job->x, p->pf_xh.data(), n_embd*T);
    p->pf_logits.resize((size_t) n_exp*T);
    p->pf_list.resize((size_t) n_exp);
    p->pf_idx.resize((size_t) n_exp);
    p->pf_mark.resize((size_t) n_exp);
    p->pf_n    = 0;
    p->pf_stop = 0;
    p->pf_jobs.fetch_add(1, std::memory_order_relaxed);

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
