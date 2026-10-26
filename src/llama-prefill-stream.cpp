// [TAG_FN_PREFILL_STREAM] LLAMA_PREFILL_STREAM=1: large ubatches stream the host-resident experts layer by layer into
// VRAM banks through a pinned ring, and the MoE of each layer runs on the GPU from its bank. See llama-moe-gen5.h.
//
// Order of one ubatch (positions p = 0..n-1 over the host expert layers, bank(p) = p % n_bufs):
//   gate(p)    CPU node before the MoE of p: the host has synchronized the device here. At p == 0 it queues the copies
//              of every layer (and of layer 0 again for the next ubatch). It waits until the streamer has issued the
//              copies of p, then makes the compute stream wait for them (event).
//   MoE(p)     GPU, weights = the bank aliases of p.
//   release(p) CPU node after the down projection of p: the host has synchronized the device, so bank(p) may be
//              overwritten; the streamer then starts the copies of p + n_bufs into it.
// The streamer copies mmap -> pinned chunk (worker threads) -> bank (copy stream, one event per chunk) and stages
// chunks of the next layer while it waits for its bank.

#include "llama-moe-gen5.h"
#include "llama-moe-gen5-impl.h"
#include "llama-moecache.h" // [TAG_FN_R1_PFS_LEND]
#include "llama-fn-stall.h" // [TAG_FN_L18_FIRSTANS_STALL]
#include "llama-fn-arena.h"  // [TAG_FN_L14_ARENA]
#include "llama-fn-tier.h" // [TAG_FN_L15_WARM]
#include "llama-fn-l22.h"    // [TAG_FN_L22_P4] pins, direct I/O, residency, PASSLOG
#include "llama-fn-l22-p4.h" // [TAG_FN_L22_P4]
#include "llama-fn-l22-p10.h" // [TAG_FN_L22_P10]
#include "llama-fn-auto.h"    // [TAG_FN_L22_STAGERS] llama_fn_l22_stagers
#include "llama-fn-l22-p16.h" // [TAG_FN_L22_CPUSHARE]

#include "llama-impl.h"
#include "llama-mmap.h" // [TAG_FN_L15_PFSAHEAD]

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#    include <psapi.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono> // [TAG_FN_L22_CPUSHARE]
#include <cinttypes>
#include <condition_variable>
#include <cstdlib> // [TAG_FN_L22_STAGERS]
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string> // [TAG_FN_R25_PIN]
#include <thread>
#include <vector>

namespace {

struct pfs_layer {
    int il   = -1;
    int pos  = 0;
    int bank = 0;

    ggml_tensor * src[3] = { nullptr, nullptr, nullptr };
    ggml_tensor * dev[3] = { nullptr, nullptr, nullptr }; // aliases in the bank
    size_t        nbytes[3] = { 0, 0, 0 };
    size_t        pad[3]    = { 0, 0, 0 };                 // alloc size - nbytes, zeroed after each copy
    bool          src_pinned = false;
    // [TAG_FN_L22_CPUSHARE] the zero-expert aliases [ne0, ne1, n_expert + 1] at the same bank address (mode 1) and the
    // bytes behind the layer's experts that must read as zero (the zero expert and the alloc padding of that shape)
    ggml_tensor * zdev[3] = { nullptr, nullptr, nullptr };
    size_t        zpad[3] = { 0, 0, 0 };
    bool          exp_ok  = false; // every tensor is n_expert whole nb[2] slices (expert ranges can be skipped)

    llama_pfs_view view;
};

struct pfs_bank {
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor *         raw = nullptr; // I8 over the whole bank: byte-offset writes

    int  holds    = -1;    // position whose copies were last dispatched here
    bool complete = false; // all copies of holds are issued and ev_done follows them
    bool consumed = true;  // holds has been used, or is not needed: a new job may overwrite the bank

    // [TAG_FN_PREFILL_STREAM] the gate of used_pos passed in generation used_gen: a late "keep" of a queued job for that
    // layer must not take the bank back after the layer's release (the next job of the bank would wait forever)
    uint64_t used_gen = 0;
    int      used_pos = -1;

    ggml_backend_event_t ev_done = nullptr; // copy stream, after the copies of holds
    ggml_backend_event_t ev_free = nullptr; // compute stream, at the last release

    // [TAG_FN_L22_PIN] pin tickets of the copies dispatched into this bank, released once ev_tk (copy stream, after
    // those copies) has completed: at the bank's next job, at a return of the banks, at the end
    std::vector<int32_t> tk;
    ggml_backend_event_t ev_tk = nullptr;
    bool                 tk_rec = false;

    // [TAG_FN_L22_CPUSHARE] holds lacks some late experts (a share job skipped them): never kept for another ubatch;
    // zc[k]: the layer bytes of tensor k behind which the zero-expert region is known to be zero (0: unknown)
    bool   partial = false;
    size_t zc[3]   = { 0, 0, 0 };
};

struct pfs_job {
    int      pos  = -1;
    uint64_t gen  = 0;
    bool     wrap = false; // [TAG_FN_L15_D2DWRAP] the next ubatch's layer 0, copied while the owner may run a hot-set step
};

struct pfs_state {
    const void *       owner    = nullptr;
    ggml_backend_dev_t dev      = nullptr;
    ggml_backend_t     compute  = nullptr;
    ggml_backend_t     copy     = nullptr; // own stream
    bool               own_copy = false;
    // [TAG_FN_L3_HOST_POKE] flush the copy stream after a dispatch (a no-op unless LLAMA_FN_HOST_POKE turned it on)
    void (*poke)(ggml_backend_t) = nullptr;
    // [TAG_FN_L15_PFSAHEAD] LLAMA_PREFILL_STREAM_AHEAD=<n> (default 2, 0 = off): when the streamer starts a layer it asks the
    // OS to read the mapped pages of the next n layers, so pages that are not in RAM arrive in large reads ahead of the
    // copy threads instead of one 4 KiB fault at a time (the first prompt after a start paged in 24-36 GB that way)
    int  ahead    = 2;
    int  ahead_to = -1; // the last position asked for
    // [TAG_FN_L14_PFSD2D] LLAMA_PREFILL_STREAM_D2D=1: the experts the hot set holds go slot -> bank on the device
    bool (*d2d)(ggml_backend_t, int, void * const *, const void * const *, const size_t *) = nullptr;
    // [TAG_FN_L22_PIN] R-REV-3: a non-blocking test of a copy-stream event (the device backend's proc; null: none). The
    // banks' tickets (pfs_bank::tk) are touched under tk_mtx only; tk_banks counts the banks that hold some
    bool (*ev_done)(ggml_backend_event_t) = nullptr;
    std::mutex       tk_mtx;
    std::atomic<int> tk_banks{0};

    // [TAG_FN_R1_PFS_LEND] LLAMA_PREFILL_STREAM_LEND=1: the banks are a range of the hot set's buffer, borrowed while
    // ubatches stream (lent) and returned before a decode; the aliases are placed there at the first lend
    bool                  lend        = false;
    bool                  lent        = false;
    bool                  lend_ready  = false;
    bool                  lend_warned = false;
    ggml_backend_buffer_t lend_buf    = nullptr;
    uint8_t *             lend_base   = nullptr;

    int64_t min_tokens = 32;
    int     n_bufs     = 2;
    bool    wrap       = true;
    bool    stats      = false;

    size_t off[3] = { 0, 0, 0 }; // tensor offsets in every bank
    size_t bank_size = 0;

    std::vector<pfs_layer>                  layers; // by position
    std::map<const ggml_tensor *, int>      by_up;
    std::vector<pfs_bank>                   banks;
    std::vector<ggml_context *>             ctxs;

    gen5::host_mem                    ring;
    size_t                            chunk    = 32u << 20;
    int                               n_chunks = 0;
    std::vector<ggml_backend_event_t> chunk_ev;
    std::vector<int>                  chunk_state; // 0 free, 1 staged (not dispatched), 2 dispatched; streamer only
    gen5::host_mem                    zeros;
    // [TAG_FN_L22_STAGERS] on the heap: a new thread count is a new pool (a restarted pool's workers would see the old
    // job generation and run the last copy list again)
    std::unique_ptr<gen5::copy_pool>  pool;

    std::thread             thr;
    std::mutex              mtx;
    std::condition_variable cv;      // streamer: job queued, bank consumed, stop
    std::condition_variable cv_done; // gates: a job completed
    std::deque<pfs_job>     queue;   // front = the job in progress
    uint64_t                gen      = 0; // a gate at position 0 cancels older jobs
    uint64_t                cur_gen  = 0; // generation of the job in progress
    int                     cur_pos  = -1;
    bool                    stop     = false;

    llama_pfs_counters ctr;
    int64_t            t_ub0   = 0;
    uint64_t           ub_bytes = 0;
    uint64_t           ub_d2d   = 0; // [TAG_FN_L14_PFSD2D]
    double             ub_wait  = 0.0;

    // [TAG_FN_L22_P4] round 22 (all off unless llama_prefill_stream_l22_config turned them on)
    llama_pfs_l22_cfg   l22;
    const llama_model * model      = nullptr;
    int64_t             rows_after = 0;     // [TAG_FN_L22_KEEPLEND] prompt rows still to come after the next ubatch
    bool                tail_now   = false; // [TAG_FN_L22_ENDOVERLAP] the next ubatch is a prompt's narrow last part
    bool                async_ret  = false; // [TAG_FN_L22_ENDOVERLAP] the next return queues the refill (upload worker)
    bool                hidden     = false; // [TAG_FN_L22_KEEPLEND] the lent layers are hidden from the hot chains
    llama_pfs_l22_sums  sums;               // under mtx
    uint64_t            ub_pinned  = 0;     // this ubatch (under mtx)
    uint64_t            ub_ssd     = 0;
    // [TAG_FN_L22_READAHEAD] the read-ahead thread: woken at gate 0 of each ubatch (ra_gen), it samples every layer once
    // and prefetches the missing ones in stream order
    std::thread             ra_thr;
    std::mutex              ra_mtx;
    std::condition_variable ra_cv;
    uint64_t                ra_gen  = 0;
    bool                    ra_stop = false;
    std::unique_ptr<std::atomic<uint16_t>[]> l_miss; // by position: sampled share out of RAM, permille (the read-ahead)
    std::unique_ptr<std::atomic<uint8_t>[]>  l_map;  // [TAG_FN_L22_RESSET_YIELD] by position: mapped at start-up (gate's RAM plan)
    std::atomic<uint64_t>   l_miss_gen{0};           // the generation l_miss belongs to (0: none yet)
    // [TAG_FN_L22_SSDTIER] direct I/O has read for the stream (1), refused before it ever read (-1), not tried (0): the
    // read-ahead leaves the direct-I/O experts out only once it has read, so a refusing reader costs no read-ahead
    std::atomic<int>        dio_state{0};

    // [TAG_FN_L22_P10] round 22, package P10 (all off unless the context turned them on)
    int            n_threads_max = 8;     // LLAMA_PREFILL_STREAM_THREADS: the pool at init
    // [TAG_FN_R25_STAGE] the copy pool's mode (every pool made for this state, also a stagers restart)
    bool           cp_tmp  = false;       // LLAMA_PREFILL_STREAM_CACHED: plain stores into an L3-sized ring
    bool           cp_pf   = false;       // LLAMA_PREFILL_STREAM_CACHED_PF: NTA prefetch of the source
    int            cp_spin = 0;           // LLAMA_PREFILL_STREAM_SPIN_US
    bool           cp_async = false;      // LLAMA_PREFILL_STREAM_ASYNC: a chunk is dispatched while the next one is copied
    int            cp_pin   = 0;          // [TAG_FN_R25_PIN] LLAMA_PREFILL_STREAM_PIN: 0 free, 1 one core each, 2 one CCD each
    std::vector<gen5::cpu_mask> cp_affs;  // [TAG_FN_R25_PIN] the copy workers' affinities (empty: free)
    size_t         cp_grain = 0;          // [TAG_FN_R25_PIN] LLAMA_PREFILL_STREAM_GRAIN_KIB (0: fixed parts)
    int            stagers_mode  = 0;     // [TAG_FN_L22_STAGERS] 0 off, -1 auto, n >= 2 fixed
    double         stg_share     = -1.0;  // EWMA (alpha 0.5) of staged / host bytes per pass (-1: none since a pin change)
    bool           stg_pin       = false; // pins were live at the last lend
    uint64_t       stg_restarts  = 0;
    int64_t        cur_rows      = 0;     // [TAG_FN_L22_XOVER] rows of the owner's running ubatch (its thread)
    llama_pfs_pass last_pass;             // the last finished pass (under mtx)
    uint64_t       pass_seq      = 0;     // under mtx
    uint64_t       pass_read     = 0;     // under mtx
    double         fix_ms        = 0.0;   // host ms of the lends and returns (the owner's thread)

    // [TAG_FN_L22_CPUSHARE] round 22, package P16 (all off unless a share mode was configured before the init)
    int                      cps_mode    = 0;     // the configured share mode (0 off)
    bool                     zexp        = false; // the banks carry a zero expert behind every layer (mode 1)
    int                      cps_wait_ms = 100;   // the most a share job waits for its layer's plan
    int64_t                  n_exp_max   = 0;     // mask stride
    uint64_t                 share_gen   = 0;     // the generation whose jobs have a late phase (0: none)
    std::vector<uint8_t>     late;                // [n_pos][n_exp_max] under mtx
    std::vector<uint8_t>     late_any;            // [n_pos] under mtx
    std::vector<uint8_t>     skip;                // [n_pos][n_exp_max] under mtx
    std::vector<uint64_t>    pub_gen;             // [n_pos] the generation of the published skip
    std::vector<int64_t>     pub_us;              // [n_pos] when it was published
    llama_pfs_share_counters sctr;                // under mtx
};

pfs_state * g_pfs = nullptr;

// [TAG_FN_L15_PFSAHEAD] a sample of the layer's mapped pages (every 256th, 1 MiB apart) is not all in this process's
// working set: a read ahead would bring pages from the disk
bool pfs_pages_missing(const pfs_layer & L) {
#ifdef _WIN32
    std::vector<PSAPI_WORKING_SET_EX_INFORMATION> q;
    for (int k = 0; k < 3; ++k) {
        if (!L.src[k] || !L.src[k]->data) {
            continue;
        }
        const char * p0 = (const char *) L.src[k]->data;
        for (size_t o = 0; o < L.nbytes[k]; o += (size_t) 1 << 20) {
            PSAPI_WORKING_SET_EX_INFORMATION x = {};
            x.VirtualAddress = (void *) (p0 + o);
            q.push_back(x);
        }
    }
    if (q.empty() || !K32QueryWorkingSetEx(GetCurrentProcess(), q.data(), (DWORD) (q.size()*sizeof(q[0])))) {
        return false;
    }
    size_t out = 0;
    for (const auto & x : q) {
        out += x.VirtualAttributes.Valid ? 0 : 1;
    }
    return out*50 > q.size(); // more than 2 % out
#else
    GGML_UNUSED(L);
    return false;
#endif
}

// [TAG_FN_L22_READAHEAD] the same sample as a share (0..1) of the layer's sampled pages that are out of the working set;
// -1 when the query is not possible. [TAG_FN_L22_SSDFIX] with the residency table the share returned counts only the
// experts the stream reads through the mapping (RAM class): VRAM-held experts come from their slots (their host copies go
// to standby on purpose), the SSD set by direct I/O. Counting them made layers look mostly missing, so the whole layer
// went through direct I/O on every pass (~2.9 GiB per pass under the 8 GiB eater, its RAM pages read from the disk)
double pfs_missing_share(const pfs_state * s, const pfs_layer & L, double * all_share = nullptr) {
    if (all_share) {
        *all_share = -1.0;
    }
#ifdef _WIN32
    // [TAG_FN_L22_SSDFIX] cls: the residency classes (none: no table, every expert counts as mapped); the SSD set is
    // never sampled (direct I/O: its pages stay out on purpose). ram[i]: sample i belongs to a mapped (RAM-class) expert.
    // *all_share: the share over every sampled expert (the VRAM-held standby copies too: the read-ahead's trigger)
    std::vector<uint8_t> cls;
    bool by_cls = false;
    if (s->model && L.src[0] && L.src[0]->ne[2] > 0) {
        cls.resize((size_t) L.src[0]->ne[2]);
        by_cls = llama_fn_res_classes(s->model, L.il, cls.data(), (int) cls.size());
    }
    std::vector<PSAPI_WORKING_SET_EX_INFORMATION> q;
    std::vector<uint8_t> ram;
    for (int k = 0; k < 3; ++k) {
        if (!L.src[k] || !L.src[k]->data) {
            continue;
        }
        const char * p0 = (const char *) L.src[k]->data;
        const size_t nb = L.src[k]->nb[2];
        if (!by_cls || nb == 0 || nb*cls.size() > L.nbytes[k]) {
            for (size_t o = 0; o < L.nbytes[k]; o += (size_t) 1 << 20) {
                PSAPI_WORKING_SET_EX_INFORMATION x = {};
                x.VirtualAddress = (void *) (p0 + o);
                q.push_back(x);
                ram.push_back(1);
            }
            continue;
        }
        for (size_t e = 0; e < cls.size(); ++e) {
            if (cls[e] == LLAMA_FN_RES_SSD) {
                continue;
            }
            for (size_t o = 0; o < nb; o += (size_t) 1 << 20) {
                PSAPI_WORKING_SET_EX_INFORMATION x = {};
                x.VirtualAddress = (void *) (p0 + e*nb + o);
                q.push_back(x);
                ram.push_back(cls[e] == LLAMA_FN_RES_RAM ? 1 : 0);
            }
        }
    }
    if (q.empty()) {
        if (all_share) {
            *all_share = by_cls ? 0.0 : -1.0;
        }
        return by_cls ? 0.0 : -1.0; // no expert of this layer is read through the mapping: nothing to wait for
    }
    if (!K32QueryWorkingSetEx(GetCurrentProcess(), q.data(), (DWORD) (q.size()*sizeof(q[0])))) {
        return -1.0;
    }
    size_t out = 0, out_ram = 0, n_ram = 0;
    for (size_t i = 0; i < q.size(); ++i) {
        const size_t m = q[i].VirtualAttributes.Valid ? 0 : 1;
        out += m;
        if (ram[i]) {
            out_ram += m;
            n_ram++;
        }
    }
    if (all_share) {
        *all_share = (double) out / (double) q.size();
    }
    return n_ram > 0 ? (double) out_ram / (double) n_ram : 0.0;
#else
    GGML_UNUSED(s);
    GGML_UNUSED(L);
    return -1.0;
#endif
}

// [TAG_FN_L22_SSDTIER] the experts of a layer that go through direct I/O in this job (empty: none); true: the whole layer.
// [TAG_FN_L22_SSDFIX] `miss` is the share of the mapped (RAM-class) pages that are out (pfs_missing_share): a layer goes
// whole only when the pages the stream would map are mostly not in RAM (bringing them in would take RAM the plan did not
// give them; the SSD probe reads them in when the prompts' low-water has room). Its RAM pages no longer go to the disk
// because VRAM-held standby copies or the SSD set made the layer look missing
bool pfs_dio_experts(const pfs_state * s, const pfs_layer & L, double miss, std::vector<uint8_t> & out) {
    out.clear();
    if (s->l22.ssdtier <= 0 || !s->model || !L.src[0]) {
        return false;
    }
    const int64_t n_exp = L.src[0]->ne[2];
    const bool whole = s->l22.ssdtier >= 2 && miss > 0.5;
    bool any = whole;
    out.assign((size_t) n_exp, whole ? 1 : 0);
    if (!whole && llama_fn_res_classes(s->model, L.il, out.data(), (int) n_exp)) {
        for (auto & c : out) {
            c = (uint8_t) (c == LLAMA_FN_RES_SSD ? 1 : 0);
            any = any || c != 0;
        }
    } else if (!whole) {
        std::fill(out.begin(), out.end(), 0);
    }
    if (!any) {
        out.clear();
    }
    return whole && any;
}

// [TAG_FN_L22_READAHEAD] the read-ahead thread: at each new ubatch (gate 0) one sample of every layer, then the layers
// with pages out of RAM are prefetched in stream order, at most ra_call bytes per call; the experts that go through direct
// I/O are left out (their pages never come through the mapping in a prompt)
void pfs_ra_run(pfs_state * s) {
    llama_fn_thread_name("prefill-readahead");
    uint64_t done = 0;
    std::vector<int>    miss;
    std::vector<uint8_t> dio;
    for (;;) {
        uint64_t g = 0;
        {
            std::unique_lock<std::mutex> lk(s->ra_mtx);
            s->ra_cv.wait(lk, [&]() { return s->ra_stop || s->ra_gen != done; });
            if (s->ra_stop) {
                return;
            }
            g = s->ra_gen;
            done = g;
        }
        const int np = (int) s->layers.size();
        miss.clear();
        double sum = 0.0;
        int    n_q = 0;
        for (int p = 0; p < np; ++p) {
            double all = -1.0;
            const double sh = pfs_missing_share(s, s->layers[p], &all);
            s->l_miss[p].store((uint16_t) (sh < 0 ? 0 : std::min(1000.0, sh*1000.0 + 0.5)), std::memory_order_relaxed);
            if (sh >= 0) {
                sum += sh;
                n_q++;
            }
            // [TAG_FN_L22_SSDFIX] l_miss (the job's whole-layer rule) counts the mapped pages only; the read-ahead fires as
            // before on the share over every expert but the SSD set (the VRAM-held standby copies too: a prompt-end pass
            // that evicts experts sends a prompt's narrow last part to the CPU path on their host copies, which then fault
            // one by one: 2-8 s at the end of a 10K-32K prompt under the 8 GiB eater when they were left out), and on any
            // mapped page out (Windows trims a scattered share of the set under pressure)
            if (all > 0.02 || sh > 0.0) {
                miss.push_back(p);
            }
        }
        // [TAG_FN_L22_RESSET_YIELD] start-up (a yielding RESSET warm pass has not ended): the layers the whole-layer rule would
        // read by direct I/O are mapped, in stream order, while their missing pages fit in the available RAM above the warm
        // floor (the plan gives them RAM: the warm pass would read them). Direct I/O does not leave them in RAM, so the next
        // prompt read them from the disk again (2 x 16K after a start: second pass ~16 s by direct I/O vs ~5.8 s from RAM)
        if (s->l_map) {
            size_t room = s->model ? llama_fn_warm_startup_room(*s->model) : 0;
            for (int p = 0; p < np; ++p) {
                const pfs_layer & L = s->layers[p];
                const double sh = s->l_miss[p].load(std::memory_order_relaxed) / 1000.0;
                uint8_t mp = 0;
                if (room > 0 && sh > 0.5) {
                    const size_t need = (size_t) (sh * (double) (L.nbytes[0] + L.nbytes[1] + L.nbytes[2]));
                    if (need <= room) {
                        room -= need;
                        mp = 1;
                    }
                }
                s->l_map[p].store(mp, std::memory_order_relaxed);
            }
        }
        s->l_miss_gen.store(g, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(s->mtx);
            s->sums.miss_share  = n_q > 0 ? sum / n_q : 0.0;
            s->sums.miss_layers = (uint32_t) miss.size();
        }
        if (llama_fn_warm_leads()) {
            continue; // [TAG_FN_L15_WARM] the warm pass brings the pages in itself ([TAG_FN_L22_RESSET_YIELD] not a yielding one)
        }
        for (int p : miss) {
            {
                std::lock_guard<std::mutex> lk(s->ra_mtx);
                if (s->ra_stop || s->ra_gen != g) {
                    break; // a newer ubatch: sample again
                }
            }
            const pfs_layer & L = s->layers[p];
            double sh = s->l_miss[p].load(std::memory_order_relaxed) / 1000.0;
            if (s->l_map && s->l_map[p].load(std::memory_order_relaxed) != 0) {
                sh = std::min(sh, 0.5); // [TAG_FN_L22_RESSET_YIELD] mapped at start-up
            }
            if (s->dio_state.load(std::memory_order_relaxed) == 1) {
                if (pfs_dio_experts(s, L, sh, dio)) {
                    continue; // read by direct I/O as a whole
                }
            } else {
                dio.clear(); // direct I/O has not read yet: the mapped pages are what the stream reads
            }
            std::vector<std::pair<const void *, size_t>> rr;
            size_t bytes = 0;
            for (int k = 0; k < 3; ++k) {
                if (!L.src[k] || !L.src[k]->data) {
                    continue;
                }
                const uint8_t * base = (const uint8_t *) L.src[k]->data;
                if (dio.empty()) {
                    rr.push_back({ base, L.nbytes[k] });
                    bytes += L.nbytes[k];
                    continue;
                }
                const size_t  nb    = L.src[k]->nb[2];
                const int64_t n_exp = (int64_t) dio.size();
                for (int64_t e = 0; e < n_exp; ) {
                    if (dio[(size_t) e]) {
                        ++e;
                        continue;
                    }
                    int64_t e1 = e + 1;
                    while (e1 < n_exp && !dio[(size_t) e1]) {
                        ++e1;
                    }
                    rr.push_back({ base + (size_t) e*nb, (size_t) (e1 - e)*nb });
                    bytes += (size_t) (e1 - e)*nb;
                    e = e1;
                }
            }
            // one call per ra_call bytes at most
            std::vector<std::pair<const void *, size_t>> part;
            size_t in_part = 0;
            for (const auto & r : rr) {
                for (size_t o = 0; o < r.second; ) {
                    const size_t n = std::min(r.second - o, s->l22.ra_call - in_part);
                    part.push_back({ (const uint8_t *) r.first + o, n });
                    in_part += n;
                    o += n;
                    if (in_part >= s->l22.ra_call) {
                        llama_fn_prefetch_ranges(part);
                        part.clear();
                        in_part = 0;
                    }
                }
            }
            if (!part.empty()) {
                llama_fn_prefetch_ranges(part);
            }
            std::lock_guard<std::mutex> lk(s->mtx);
            s->sums.ra_bytes += bytes;
        }
    }
}

// [TAG_FN_L22_PIN] the bank's tickets go back (tk_mtx held; the copies behind them have landed)
void pfs_bank_drop_tk_locked(pfs_state * s, pfs_bank & B) {
    for (int32_t t : B.tk) {
        llama_fn_pin_release(t);
    }
    B.tk.clear();
    B.tk_rec = false;
    s->tk_banks.fetch_sub(1, std::memory_order_relaxed);
}

// [TAG_FN_L22_PIN] release the bank's pin tickets once the copies behind them have landed (waits for them)
void pfs_bank_release_tk(pfs_state * s, pfs_bank & B) {
    std::lock_guard<std::mutex> lk(s->tk_mtx);
    if (B.tk.empty()) {
        return;
    }
    if (B.tk_rec && B.ev_tk) {
        ggml_backend_event_synchronize(B.ev_tk);
    } else if (s->copy) {
        ggml_backend_synchronize(s->copy); // no event: everything queued so far
    }
    pfs_bank_drop_tk_locked(s, B);
}

// [TAG_FN_L22_PIN] R-REV-3: the tickets of every bank whose copies have landed go back now (no wait): the OOM hook, the
// watcher and the valve can unpin those ranges while the banks stay lent (KEEPLEND, ENDOVERLAP) or wait for a layer
void pfs_release_landed(pfs_state * s) {
    if (s->ev_done == nullptr || s->tk_banks.load(std::memory_order_relaxed) == 0) {
        return;
    }
    std::lock_guard<std::mutex> lk(s->tk_mtx);
    for (auto & B : s->banks) {
        if (!B.tk.empty() && B.tk_rec && B.ev_tk && s->ev_done(B.ev_tk)) {
            pfs_bank_drop_tk_locked(s, B);
        }
    }
}

std::mutex  g_pfs_init;

size_t align_up(size_t x, size_t a) {
    return (x + a - 1) / a * a;
}

// the streamer thread
void pfs_run(pfs_state * s) {
    llama_fn_thread_name("prefill-stream"); // [TAG_FN_L18_FIRSTANS_STALL]
    struct item { int slot; const uint8_t * src; size_t dst; size_t n; };
    std::vector<item> staged;
    int next_slot = 0;
    // [TAG_FN_L14_PFSD2D]
    std::vector<int32_t>                     hot_slot;
    std::vector<std::pair<size_t, size_t>>   ranges;
    std::vector<gen5::copy_seg>              segs;
    std::vector<item>                        pieces;
    std::vector<item>                        pend; // [TAG_FN_R25_STAGE] the pieces of the chunk copied last (async)
    std::vector<void *>                      d2d_dst;
    std::vector<const void *>                d2d_src;
    std::vector<size_t>                      d2d_len;
    // [TAG_FN_L22_PIN] [TAG_FN_L22_SSDTIER]
    std::vector<std::pair<size_t, size_t>>   rr_stage;
    std::vector<uint8_t>                     dio;
    std::vector<llama_fn_dio_req>            dio_req;
    // [TAG_FN_L22_CPUSHARE]
    std::vector<uint8_t>                     late_l;
    std::vector<uint8_t>                     skip_l;

    for (;;) {
        pfs_job job;
        bool share = false; // [TAG_FN_L22_CPUSHARE] this job copies its late experts after the layer's plan
        {
            std::unique_lock<std::mutex> lk(s->mtx);
            auto has_job = [&]() { return s->stop || !s->queue.empty(); };
            // [TAG_FN_L22_PIN] R-REV-3: while banks hold tickets, look at their copies every 2 ms (only this thread adds
            // tickets, so none appear while it waits)
            while (!has_job() && s->ev_done != nullptr && s->tk_banks.load(std::memory_order_relaxed) > 0) {
                s->cv.wait_for(lk, std::chrono::milliseconds(2), has_job);
                pfs_release_landed(s);
            }
            s->cv.wait(lk, has_job);
            if (s->stop) {
                return;
            }
            job = s->queue.front();
            pfs_bank & b0 = s->banks[s->layers[job.pos].bank];
            if (b0.holds == job.pos && b0.complete && !b0.partial) {
                // the bank has this layer already (weights never change): keep it until the layer's release, unless
                // the layer's gate already took it in this generation (then its release frees it, maybe already did)
                if (!(b0.used_gen == s->gen && b0.used_pos == job.pos)) {
                    b0.consumed = false;
                }
                s->queue.pop_front();
                s->cv_done.notify_all();
                continue;
            }
            s->cur_gen = job.gen;
            s->cur_pos = job.pos;
            // [TAG_FN_L22_CPUSHARE] a job of a share ubatch (never layer 0: a new ubatch keeps bank 0's copy)
            if (s->cps_mode > 0 && s->share_gen != 0 && job.gen == s->share_gen && !job.wrap && job.pos > 0 &&
                s->late_any[job.pos] != 0 && s->layers[job.pos].exp_ok) {
                const int64_t ne = s->layers[job.pos].src[0]->ne[2];
                const uint8_t * m = s->late.data() + (size_t) job.pos*s->n_exp_max;
                late_l.assign(m, m + ne);
                skip_l.assign((size_t) ne, 0);
                share = true;
            }
        }
        const pfs_layer & L = s->layers[job.pos];
        pfs_bank &        B = s->banks[L.bank];
        const int64_t t_job0 = share ? ggml_time_us() : 0; // [TAG_FN_L22_CPUSHARE]
        llama_fn_warm_wait_layer(L.il); // [TAG_FN_L15_WARM] the warm pass brings this layer in first
        // [TAG_FN_L15_PFSAHEAD] the next layers' pages: read ahead of the copies (positions wrap to 0 for the next ubatch)
        // [TAG_FN_L22_READAHEAD] not while the round-22 read-ahead thread runs (it covers every layer of the pass)
        if (s->ahead > 0 && !L.src_pinned && !s->l22.readahead && !llama_fn_warm_leads()) {
            const int np = (int) s->layers.size();
            llama_memory_ranges mr;
            for (int d = 1; d <= s->ahead && d < np; ++d) {
                const int q = (job.pos + d) % np;
                if (q == s->ahead_to) {
                    continue;
                }
                const pfs_layer & N = s->layers[q];
                // asking for resident pages cost ~50 ms per layer (131K prompts 2057 -> 1263 t/s): only when a sample of
                // the layer's pages (every 256th) finds some out of RAM
                if (!pfs_pages_missing(N)) {
                    s->ahead_to = q;
                    continue;
                }
                for (int k = 0; k < 3; ++k) {
                    if (N.src[k] && N.src[k]->data) {
                        mr.push_back({ N.src[k]->data, N.nbytes[k] });
                    }
                }
                s->ahead_to = q;
            }
            llama_prefetch(mr);
        }
        bool dispatching = false;
        bool cancelled   = false;
        size_t bytes = 0;
        size_t bytes_d2d = 0;
        size_t bytes_pin = 0; // [TAG_FN_L22_PIN] dispatched straight from host pages (slot -1)
        size_t bytes_ssd = 0; // [TAG_FN_L22_SSDTIER] read into the ring by direct I/O
        std::vector<int32_t> job_tk; // [TAG_FN_L22_PIN] tickets of this job's pinned segments
        // [TAG_FN_L22_SSDTIER] the experts of this layer that go through direct I/O (empty: none)
        if (s->l22.ssdtier > 0 && s->dio_state.load(std::memory_order_relaxed) != -1) { // -1: the reader refuses
            double miss = 0.0;
            if (s->l22.ssdtier >= 2) {
                miss = s->l_miss && s->l_miss_gen.load(std::memory_order_acquire) != 0 ?
                       s->l_miss[job.pos].load(std::memory_order_relaxed) / 1000.0 : pfs_missing_share(s, L);
                // [TAG_FN_L22_RESSET_YIELD] start-up: mapped when the read-ahead's gate found RAM for it (or, without the
                // read-ahead, when its missing pages fit in the available RAM above the warm floor now)
                if (miss > 0.5) {
                    if (s->l_map && s->l_miss_gen.load(std::memory_order_acquire) != 0) {
                        if (s->l_map[job.pos].load(std::memory_order_relaxed) != 0) {
                            miss = 0.5;
                        }
                    } else if (s->model && llama_fn_warm_startup_room(*s->model) >=
                               (size_t) (miss * (double) (L.nbytes[0] + L.nbytes[1] + L.nbytes[2]))) {
                        miss = 0.5;
                    }
                }
            }
            pfs_dio_experts(s, L, miss, dio);
        } else {
            dio.clear();
        }
        // [TAG_FN_L22_PIN] pinned segments through llama_fn_pin_query instead of the load-time flag (P6's registry)
        const bool pin_live = s->l22.pin && llama_fn_pin_active();

        // true while this job is still wanted
        auto alive = [&]() -> bool {
            std::lock_guard<std::mutex> lk(s->mtx);
            return !s->stop && s->cur_gen == s->gen;
        };
        // wait until the bank may be written, then own it; false when the job was cancelled
        auto begin = [&]() -> bool {
            {
                std::unique_lock<std::mutex> lk(s->mtx);
                s->cv.wait(lk, [&]() { return s->stop || s->cur_gen != s->gen || B.consumed; });
                if (s->stop || s->cur_gen != s->gen) {
                    return false;
                }
                B.holds    = job.pos;
                B.complete = false;
                B.consumed = false;
                B.partial  = false; // [TAG_FN_L22_CPUSHARE]
                for (int k = 0; k < 3; ++k) {
                    if (B.zc[k] != L.nbytes[k]) {
                        B.zc[k] = 0; // this layer's copies may write over the zero region of the last one
                    }
                }
            }
            pfs_bank_release_tk(s, B); // [TAG_FN_L22_PIN] the bank's last copies have landed (its layer was used)
            gen5::ev_wait(s->copy, B.ev_free, s->compute);
            dispatching = true;
            return true;
        };
        auto dispatch = [&](const item & it) {
            ggml_backend_tensor_set_async(s->copy, B.raw, it.src, it.dst, it.n);
            if (it.slot >= 0) {
                gen5::ev_record(s->chunk_ev[it.slot], s->copy);
                s->chunk_state[it.slot] = 2;
            } else {
                bytes_pin += it.n;
            }
            if (s->poke) {
                s->poke(s->copy); // [TAG_FN_L3_HOST_POKE] the copy starts now, while this thread stages the next chunk
            }
            bytes += it.n;
        };
        auto flush = [&]() -> bool {
            if (!dispatching && !begin()) {
                return false;
            }
            for (const item & it : staged) {
                dispatch(it);
            }
            staged.clear();
            return true;
        };

        // the byte ranges rr of tensor k straight from the host (pinned pages: no staging ring), in chunks
        auto stream_direct = [&](int k, const std::vector<std::pair<size_t, size_t>> & rr) {
            const uint8_t * src = (const uint8_t *) L.src[k]->data;
            for (const auto & r : rr) {
                for (size_t o = r.first; o < r.second && !cancelled; o += s->chunk) {
                    if (!dispatching && !begin()) {
                        cancelled = true;
                        break;
                    }
                    dispatch({ -1, src + o, s->off[k] + o, std::min(s->chunk, r.second - o) });
                }
            }
        };
        // the pieces pcs of one ring slot are copied: dispatch them (the bank is owned) or stage them until it is free
        auto settle = [&](std::vector<item> & pcs) {
            if (pcs.empty()) {
                return;
            }
            s->chunk_state[pcs.front().slot] = 1;
            for (const item & it : pcs) {
                if (dispatching) {
                    dispatch(it);
                } else {
                    staged.push_back(it);
                }
            }
            pcs.clear();
            if (!dispatching) {
                bool free_now;
                {
                    std::lock_guard<std::mutex> lk(s->mtx);
                    free_now = B.consumed;
                }
                if (free_now && !flush()) {
                    cancelled = true;
                }
            }
        };
        // the byte ranges rr of tensor k from the host: through the ring, a slot filled by one copy of its pieces
        // [TAG_FN_L22_SSDTIER] the experts of dio are read into the slot by direct I/O instead (a refusal: the mapped read)
        auto stream_ring = [&](int k, const std::vector<std::pair<size_t, size_t>> & rr) {
            const uint8_t * src = (const uint8_t *) L.src[k]->data;
            const size_t    nb  = L.src[k]->nb[2];
            size_t ri = 0;
            size_t ro = rr.empty() ? 0 : rr[0].first;
            pend.clear();
            while (ri < rr.size() && !cancelled) {
                const int slot = next_slot;
                next_slot = (next_slot + 1) % s->n_chunks;
                if (s->chunk_state[slot] == 1 && !flush()) { // the ring is full of staged chunks of this job
                    cancelled = true;
                    break;
                }
                if (s->chunk_state[slot] == 2) {
                    gen5::ev_sync(s->chunk_ev[slot], s->copy);
                    s->chunk_state[slot] = 0;
                }
                uint8_t * dst = s->ring.ptr + (size_t) slot*s->chunk;
                size_t pos = 0;
                segs.clear();
                pieces.clear();
                dio_req.clear();
                while (pos < s->chunk && ri < rr.size()) {
                    const size_t n = std::min(s->chunk - pos, rr[ri].second - ro);
                    if (dio.empty() || nb == 0) {
                        segs.push_back({ dst + pos, src + ro, n });
                    } else {
                        // [TAG_FN_L22_SSDTIER] split at expert edges into runs of the same path
                        for (size_t o = ro; o < ro + n; ) {
                            const size_t e  = o / nb;
                            const bool   dd = e < dio.size() && dio[e] != 0;
                            size_t o1 = std::min(ro + n, (e + 1)*nb);
                            while (o1 < ro + n) {
                                const size_t e2 = o1 / nb;
                                if ((e2 < dio.size() && dio[e2] != 0) != dd) {
                                    break;
                                }
                                o1 = std::min(ro + n, (e2 + 1)*nb);
                            }
                            if (dd) {
                                dio_req.push_back({ src + o, o1 - o, dst + pos + (o - ro) });
                            } else {
                                segs.push_back({ dst + pos + (o - ro), src + o, o1 - o });
                            }
                            o = o1;
                        }
                    }
                    pieces.push_back({ slot, dst + pos, s->off[k] + ro, n });
                    pos += n;
                    ro  += n;
                    if (ro == rr[ri].second && ++ri < rr.size()) {
                        ro = rr[ri].first;
                    }
                }
                // [TAG_FN_R25_STAGE] async (a chunk without direct I/O): the workers copy this chunk while the last chunk's
                // pieces are dispatched; every submit is waited for in its own iteration, so no copy runs past it
                if (s->cp_async && dio_req.empty()) {
                    if (!segs.empty()) {
                        s->pool->submit(segs.data(), (int) segs.size());
                    }
                    settle(pend);
                    s->pool->wait();
                    pend.swap(pieces);
                    if (!alive()) {
                        cancelled = true;
                    }
                    continue;
                }
                settle(pend); // a direct-I/O chunk after async ones: the pending pieces first
                int dio_tk = -1;
                if (!dio_req.empty()) {
                    dio_tk = llama_fn_dio_submit(s->model, dio_req.data(), (int) dio_req.size());
                    if (dio_tk < 0) {
                        int unknown = 0;
                        s->dio_state.compare_exchange_strong(unknown, -1); // refused before it ever read
                        for (const auto & r : dio_req) {
                            segs.push_back({ (uint8_t *) r.dst, (const uint8_t *) r.src, r.len });
                        }
                    }
                }
                if (!segs.empty()) {
                    s->pool->copy_list(segs.data(), (int) segs.size());
                }
                if (dio_tk >= 0) {
                    if (llama_fn_dio_wait(dio_tk)) {
                        s->dio_state.store(1, std::memory_order_relaxed);
                        for (const auto & r : dio_req) {
                            bytes_ssd += r.len;
                        }
                    } else {
                        // a failed read: the mapped bytes (same values)
                        segs.clear();
                        for (const auto & r : dio_req) {
                            segs.push_back({ (uint8_t *) r.dst, (const uint8_t *) r.src, r.len });
                        }
                        s->pool->copy_list(segs.data(), (int) segs.size());
                    }
                }
                settle(pieces);
                if (!alive()) {
                    cancelled = true;
                }
            }
            // [TAG_FN_R25_STAGE] the last async chunk: dispatched / staged, or (cancelled) staged so the job's end frees it
            if (!pend.empty()) {
                if (!cancelled) {
                    settle(pend);
                } else {
                    s->chunk_state[pend.front().slot] = 1;
                    staged.insert(staged.end(), pend.begin(), pend.end());
                    pend.clear();
                }
            }
        };
        // the byte ranges of tensor k in `ranges` from the host
        auto stream_ranges = [&](int k) {
            if (L.src_pinned) {
                stream_direct(k, ranges); // pinned at load (a host buffer or the round-14 arena)
                return;
            }
            if (pin_live) {
                // [TAG_FN_L22_PIN] split at segment edges: pinned segments straight from their pages (a copy never runs past
                // a registered range), the rest through the ring. R-REV-3 / R-REV-4: one query per chunk right before its
                // copy, with the bank owned (no ticket is held while begin() waits), and the copy reads seg.src (the
                // registered pages: the mapped model, or the arena copy the ticket protects)
                const uint8_t * src = (const uint8_t *) L.src[k]->data;
                rr_stage.clear();
                auto stage = [&](size_t a, size_t b) {
                    if (!rr_stage.empty() && rr_stage.back().second == a) {
                        rr_stage.back().second = b;
                    } else {
                        rr_stage.push_back({ a, b });
                    }
                };
                for (const auto & r : ranges) {
                    for (size_t o = r.first; o < r.second && !cancelled; ) {
                        const size_t want = std::min(s->chunk, r.second - o);
                        llama_fn_pin_seg seg = {};
                        if (!llama_fn_pin_query(src + o, want, &seg) || seg.len == 0 || seg.len > want) {
                            stage(o, r.second);
                            break;
                        }
                        if (!seg.pinned) {
                            stage(o, o + seg.len);
                            o += seg.len;
                            continue;
                        }
                        if (!dispatching) {
                            llama_fn_pin_release(seg.ticket); // begin() may wait for the bank: query again once it is ours
                            if (!begin()) {
                                cancelled = true;
                            }
                            continue;
                        }
                        dispatch({ -1, (const uint8_t *) seg.src, s->off[k] + o, seg.len });
                        if (seg.ticket >= 0) {
                            job_tk.push_back(seg.ticket);
                        }
                        o += seg.len;
                    }
                }
                if (!cancelled && !rr_stage.empty()) {
                    if (s->n_chunks > 0) {
                        stream_ring(k, rr_stage);
                    } else {
                        stream_direct(k, rr_stage); // no ring (every layer was pinned at load): pageable copies
                    }
                }
                return;
            }
            stream_ring(k, ranges);
        };
        // [TAG_FN_L14_PFSD2D] the runs of experts with (held = true) or without a hot slot, as byte ranges of tensor k
        const int64_t n_exp = L.src[0]->ne[2];
        auto expert_ranges = [&](int k, bool held) {
            ranges.clear();
            const size_t nb = L.src[k]->nb[2];
            for (int64_t e = 0; e < n_exp; ) {
                if ((hot_slot[e] >= 0) != held) {
                    ++e;
                    continue;
                }
                int64_t e1 = e + 1;
                while (e1 < n_exp && (hot_slot[e1] >= 0) == held) {
                    ++e1;
                }
                ranges.push_back({ (size_t) e*nb, (size_t) e1*nb });
                e = e1;
            }
        };

        // the experts the hot set holds come from its VRAM slots, the rest from the host
        // [TAG_FN_L15_D2DWRAP] not the wrap job: a hot-set step between two calls (a decode row in the batch) can upload another
        // expert into a slot of its snapshot before its device copy runs, and the next ubatch keeps bank 0 without a new snapshot
        const ggml_tensor * hs3[3] = { nullptr, nullptr, nullptr };
        bool d2d = s->d2d != nullptr && n_exp > 0 && !job.wrap;
        if (d2d) {
            hot_slot.resize((size_t) n_exp);
            d2d = llama_moe_hot_slots(L.src[0], hot_slot.data(), n_exp, hs3);
            for (int k = 0; d2d && k < 3; ++k) {
                d2d = hs3[k] != nullptr && hs3[k]->nb[2] == L.src[k]->nb[2] && L.src[k]->ne[2] == n_exp &&
                      L.src[k]->nb[2]*(size_t) n_exp == L.nbytes[k];
            }
        }
        // [TAG_FN_L22_CPUSHARE] the runs of experts for which pred(e) holds, as byte ranges of tensor k
        auto ranges_where = [&](int k, const auto & pred) {
            ranges.clear();
            const size_t nb = L.src[k]->nb[2];
            for (int64_t e = 0; e < n_exp; ) {
                if (!pred(e)) {
                    ++e;
                    continue;
                }
                int64_t e1 = e + 1;
                while (e1 < n_exp && pred(e1)) {
                    ++e1;
                }
                ranges.push_back({ (size_t) e*nb, (size_t) e1*nb });
                e = e1;
            }
        };
        if (!share) {
            for (int k = 0; k < 3 && !cancelled; ++k) {
                if (d2d) {
                    expert_ranges(k, false);
                } else {
                    ranges.assign(1, { (size_t) 0, L.nbytes[k] });
                }
                stream_ranges(k);
            }
        } else {
            // [TAG_FN_L22_CPUSHARE] the early experts first: host experts that are neither in a hot slot nor late
            for (int k = 0; k < 3 && !cancelled; ++k) {
                ranges_where(k, [&](int64_t e) { return !(d2d && hot_slot[e] >= 0) && late_l[e] == 0; });
                stream_ranges(k);
            }
        }
        if (!cancelled && !flush()) {
            cancelled = true;
        }
        if (!cancelled && d2d) { // the bank is owned now: slot -> bank on the copy stream
            d2d_dst.clear();
            d2d_src.clear();
            d2d_len.clear();
            for (int k = 0; k < 3; ++k) {
                const size_t    nb    = L.src[k]->nb[2];
                uint8_t *       bank  = (uint8_t *) B.raw->data + s->off[k];
                const uint8_t * slots = (const uint8_t *) hs3[k]->data;
                for (int64_t e = 0; e < n_exp; ++e) {
                    if (hot_slot[e] >= 0) {
                        d2d_dst.push_back(bank + (size_t) e*nb);
                        d2d_src.push_back(slots + (size_t) hot_slot[e]*nb);
                        d2d_len.push_back(nb);
                        bytes_d2d += nb;
                    }
                }
            }
            if (!s->d2d(s->copy, (int) d2d_dst.size(), d2d_dst.data(), d2d_src.data(), d2d_len.data())) {
                LLAMA_LOG_WARN("prefill-stream: [TAG_FN_L14_PFSD2D] the device copies failed: these experts and every later "
                        "one come from the host\n");
                s->d2d    = nullptr;
                bytes_d2d = 0;
                for (int k = 0; k < 3 && !cancelled; ++k) {
                    expert_ranges(k, true);
                    stream_ranges(k);
                }
                if (!cancelled && !flush()) {
                    cancelled = true;
                }
            }
        }
        // [TAG_FN_L22_CPUSHARE] the late experts: wait (bounded) for the layer's plan, then copy the ones it does not skip
        bool     skipped_any = false;
        uint64_t b_skip      = 0;
        uint64_t b_late      = 0;
        if (share && !cancelled) {
            const int64_t tw0 = ggml_time_us();
            bool got = false;
            bool timed_out = false;
            {
                std::unique_lock<std::mutex> lk(s->mtx);
                auto ready = [&]() { return s->stop || s->cur_gen != s->gen || s->pub_gen[job.pos] == job.gen; };
                if (!ready()) {
                    timed_out = !s->cv.wait_for(lk, std::chrono::milliseconds(s->cps_wait_ms), ready);
                }
                if (s->stop || s->cur_gen != s->gen) {
                    cancelled = true;
                } else if (s->pub_gen[job.pos] == job.gen) {
                    got = true;
                    const uint8_t * m = s->skip.data() + (size_t) job.pos*s->n_exp_max;
                    skip_l.assign(m, m + n_exp);
                    const double plan_us = (double) std::max<int64_t>(0, s->pub_us[job.pos] - t_job0);
                    s->sctr.plan_us_ewma = s->sctr.plan_us_ewma < 0 ? plan_us : 0.8*s->sctr.plan_us_ewma + 0.2*plan_us;
                }
                s->sctr.plan_wait_ms += (ggml_time_us() - tw0)/1e3;
                if (timed_out && !cancelled) {
                    s->sctr.timeouts++;
                }
            }
            if (!got) {
                std::fill(skip_l.begin(), skip_l.end(), 0); // no plan in time: every late expert is copied
            }
            for (int64_t e = 0; e < n_exp && !cancelled; ++e) {
                if (late_l[e] != 0 && !(d2d && hot_slot[e] >= 0)) {
                    const uint64_t eb = L.src[0]->nb[2] + L.src[1]->nb[2] + L.src[2]->nb[2];
                    (skip_l[e] != 0 ? b_skip : b_late) += eb;
                    skipped_any = skipped_any || skip_l[e] != 0;
                }
            }
            for (int k = 0; k < 3 && !cancelled; ++k) {
                ranges_where(k, [&](int64_t e) { return !(d2d && hot_slot[e] >= 0) && late_l[e] != 0 && skip_l[e] == 0; });
                stream_ranges(k);
            }
            if (!cancelled && !flush()) {
                cancelled = true;
            }
        }
        if (!cancelled) {
            for (int k = 0; k < 3; ++k) {
                if (s->zexp) {
                    // [TAG_FN_L22_CPUSHARE] the zero expert and the padding of the zero-expert shape, unless this bank's
                    // last zeroing was for a layer of the same size and no larger copy came since
                    if (B.zc[k] != L.nbytes[k]) {
                        for (size_t o = 0; o < L.zpad[k]; o += s->zeros.size) {
                            const size_t n = std::min(s->zeros.size, L.zpad[k] - o);
                            ggml_backend_tensor_set_async(s->copy, B.raw, s->zeros.ptr, s->off[k] + L.nbytes[k] + o, n);
                        }
                        B.zc[k] = L.nbytes[k];
                    }
                    continue;
                }
                for (size_t o = 0; o < L.pad[k]; o += s->zeros.size) {
                    const size_t n = std::min(s->zeros.size, L.pad[k] - o);
                    ggml_backend_tensor_set_async(s->copy, B.raw, s->zeros.ptr, s->off[k] + L.nbytes[k] + o, n);
                }
            }
            if (B.ev_done) {
                gen5::ev_record(B.ev_done, s->copy);
            } else {
                ggml_backend_synchronize(s->copy);
            }
        } else {
            // staged chunks go back to free; dispatched ones are synchronized before their slot is reused
            for (const item & it : staged) {
                s->chunk_state[it.slot] = 0;
            }
            staged.clear();
        }
        // [TAG_FN_L22_PIN] this job's tickets stay with the bank until its copies have landed
        if (!job_tk.empty()) {
            if (dispatching) {
                std::lock_guard<std::mutex> tl(s->tk_mtx);
                if (!B.ev_tk) {
                    B.ev_tk = gen5::ev_new(s->dev);
                }
                if (B.ev_tk) {
                    gen5::ev_record(B.ev_tk, s->copy);
                    B.tk_rec = true;
                } else {
                    B.tk_rec = false; // released after a synchronize of the copy stream
                }
                if (B.tk.empty()) {
                    s->tk_banks.fetch_add(1, std::memory_order_relaxed);
                }
                B.tk.insert(B.tk.end(), job_tk.begin(), job_tk.end());
            } else {
                for (int32_t t : job_tk) {
                    llama_fn_pin_release(t); // nothing was dispatched
                }
            }
            job_tk.clear();
        }
        pfs_release_landed(s); // [TAG_FN_L22_PIN] R-REV-3: the other banks' landed copies give their tickets back

        {
            std::lock_guard<std::mutex> lk(s->mtx);
            if (!cancelled) {
                B.complete = true;
                if (share) { // [TAG_FN_L22_CPUSHARE]
                    B.partial = skipped_any;
                    s->sctr.jobs++;
                    s->sctr.skip_bytes += b_skip;
                    s->sctr.late_bytes += b_late;
                }
                s->ctr.jobs++;
                s->ctr.bytes += bytes;
                s->ub_bytes  += bytes;
                s->ctr.d2d_bytes += bytes_d2d; // [TAG_FN_L14_PFSD2D]
                s->ub_d2d        += bytes_d2d;
                s->ub_pinned     += bytes_pin; // [TAG_FN_L22_P4]
                s->ub_ssd        += bytes_ssd;
                s->sums.bytes_pinned += bytes_pin;
                s->sums.bytes_ssd    += bytes_ssd;
                s->sums.bytes_staged += bytes > bytes_pin + bytes_ssd ? bytes - bytes_pin - bytes_ssd : 0; // mapped reads
                s->sums.bytes_d2d    += bytes_d2d;
            } else if (B.holds == job.pos && dispatching) {
                B.holds    = -1; // partly written
                B.consumed = true;
            }
            if (!s->queue.empty()) {
                s->queue.pop_front(); // the running job is always the front, also after a gate reset
            }
            s->cur_pos = -1;
        }
        s->cv_done.notify_all();
    }
}

// a new ubatch (gate 0, or the first share layer's plan): the device is idle, so no bank has a user. Older jobs are
// cancelled, except a copy of layer 0 (the wrap), which is kept running or kept in bank 0. s->mtx held
void pfs_ubatch_begin(pfs_state * s, int64_t t0) {
    pfs_bank & B = s->banks[s->layers[0].bank];
    bool had = false;
    // [TAG_FN_L22_CPUSHARE] a bank a share job left without some late experts never serves another ubatch
    for (auto & b : s->banks) {
        if (b.partial) {
            b.holds    = -1;
            b.complete = false;
            b.partial  = false;
            s->sctr.invalidated++;
        }
    }
    s->gen++;
    const bool run0  = s->cur_pos == 0;                               // a copy of layer 0 is running
    const bool data0 = B.holds == 0 && (B.complete || run0);          // bank 0 has (or is getting) layer 0
    if (run0) {
        s->cur_gen = s->gen;
    }
    std::deque<pfs_job> q;
    if (s->cur_pos >= 0 && !s->queue.empty()) {
        q.push_back(s->queue.front()); // the running job stays at the front; the streamer pops it
        if (run0) {
            q.back().gen = s->gen;
        }
    }
    for (size_t b = 0; b < s->banks.size(); ++b) {
        s->banks[b].consumed = !(b == 0 && data0); // gate 0 uses bank 0 now: nothing may overwrite it
    }
    if (!run0 && !data0) {
        q.push_back({ 0, s->gen });
    }
    for (int p = 1; p < (int) s->layers.size(); ++p) {
        q.push_back({ p, s->gen });
    }
    if (s->wrap) {
        q.push_back({ 0, s->gen, true });
    }
    s->queue.swap(q);
    s->ctr.ubatches++;
    s->t_ub0    = t0;
    s->ub_bytes = 0;
    s->ub_d2d   = 0;
    s->ub_wait  = 0.0;
    s->ub_pinned = 0; // [TAG_FN_L22_P4]
    s->ub_ssd    = 0;
    s->sums.ubatches++;
    had = B.holds == 0 && B.complete;
    if (had) {
        s->ctr.reused++;
    }
    // [TAG_FN_L22_READAHEAD] a new pass: the read-ahead samples every layer once and prefetches what is out
    if (s->l22.readahead && s->ra_thr.joinable()) {
        std::lock_guard<std::mutex> rlk(s->ra_mtx);
        s->ra_gen = s->gen;
        s->ra_cv.notify_one();
    }
}

// wait until the layer's bank holds it, then claim the bank. s->mtx held (lk)
void pfs_layer_wait(pfs_state * s, pfs_layer * L, std::unique_lock<std::mutex> & lk) {
    pfs_bank & B = s->banks[L->bank];
    while (!(B.holds == L->pos && B.complete) && !s->stop) {
        bool queued = s->cur_pos == L->pos;
        for (const auto & j : s->queue) {
            queued = queued || (j.pos == L->pos && j.gen == s->gen);
        }
        if (!queued) {
            s->queue.push_back({ L->pos, s->gen }); // guard: never wait for a copy nobody will do
            s->cv.notify_all();
        }
        s->cv_done.wait(lk);
    }
    // the layer is in use from here to its release: claim the bank, also when its data is left from an earlier
    // ubatch and the streamer has not reached this layer's job yet
    if (B.holds == L->pos && B.complete) {
        B.consumed = false;
        B.used_gen = s->gen;
        B.used_pos = L->pos;
    }
}

pfs_state * pfs_gate_state(pfs_layer * L) {
    pfs_state * s = g_pfs;
    if (!s || L->pos >= (int) s->layers.size() || &s->layers[L->pos] != L) {
        GGML_ABORT("prefill-stream: a graph outlived its stream state");
    }
    if (s->lend && !s->lent) {
        // [TAG_FN_R1_PFS_LEND] the banks are the hot set's slots right now: a decode could be reading them
        GGML_ABORT("prefill-stream: a streamed graph ran while its banks were not borrowed");
    }
    return s;
}

void pfs_gate_end(pfs_state * s, pfs_layer * L, int64_t t0) {
    pfs_bank & B = s->banks[L->bank];
    gen5::ev_wait(s->compute, B.ev_done, s->copy);
    const double w = (ggml_time_us() - t0) / 1e6;
    std::lock_guard<std::mutex> lk(s->mtx);
    s->ctr.wait_s += w;
    s->ub_wait    += w;
    s->sums.wait_ms += w*1e3; // [TAG_FN_L22_P4]
}

void pfs_gate_op(ggml_tensor * dst, int ith, int nth, void * ud) {
    GGML_UNUSED(dst);
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }
    pfs_layer * L = (pfs_layer *) ud;
    pfs_state * s = pfs_gate_state(L);
    const int64_t t0 = ggml_time_us();
    {
        std::unique_lock<std::mutex> lk(s->mtx);
        if (L->pos == 0) {
            pfs_ubatch_begin(s, t0);
        }
        s->cv.notify_all();
        pfs_layer_wait(s, L, lk);
    }
    pfs_gate_end(s, L, t0);
}

// [TAG_FN_L22_CPUSHARE] the gate of a share layer: the new-ubatch work ran in the first share layer's plan
void pfs_gate_share_op(ggml_tensor * dst, int ith, int nth, void * ud) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }
    if (dst->data != nullptr && dst->type == GGML_TYPE_F32) {
        *(float *) dst->data = 0.0f; // an ordering node: its value is never read, but keep it defined
    }
    pfs_layer * L = (pfs_layer *) ud;
    pfs_state * s = pfs_gate_state(L);
    const int64_t t0 = ggml_time_us();
    {
        std::unique_lock<std::mutex> lk(s->mtx);
        s->cv.notify_all();
        pfs_layer_wait(s, L, lk);
    }
    pfs_gate_end(s, L, t0);
}

void pfs_release_op(ggml_tensor * dst, int ith, int nth, void * ud) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }
    // [TAG_NAN_SCAN] the one-element output carries no value (an ordering node): written, so the NaN scan never reads
    // whatever bytes its buffer held (seen as a [NAN] line with the l3-cpu switches at -c 262144)
    if (dst->data != nullptr && dst->type == GGML_TYPE_F32) {
        *(float *) dst->data = 0.0f;
    }
    pfs_state * s = g_pfs;
    pfs_layer * L = (pfs_layer *) ud;
    if (!s || L->pos >= (int) s->layers.size() || &s->layers[L->pos] != L) {
        GGML_ABORT("prefill-stream: a graph outlived its stream state");
    }
    pfs_bank & B = s->banks[L->bank];
    gen5::ev_record(B.ev_free, s->compute);
    {
        std::lock_guard<std::mutex> lk(s->mtx);
        if (B.holds == L->pos) {
            B.consumed = true;
        }
        // [TAG_FN_L22_XOVER] [TAG_FN_L22_STAGERS] the pass ends at its last layer's release
        if (L->pos == (int) s->layers.size() - 1) {
            llama_pfs_pass & p = s->last_pass;
            p.seq      = ++s->pass_seq;
            p.rows     = (uint32_t) std::max<int64_t>(0, s->cur_rows);
            p.wall_ms  = (ggml_time_us() - s->t_ub0)/1e3;
            p.wait_ms  = s->ub_wait*1e3;
            p.b_host   = s->ub_bytes;
            p.b_pinned = s->ub_pinned;
            p.b_ssd    = s->ub_ssd;
            p.b_d2d    = s->ub_d2d;
            p.stagers  = s->pool ? s->pool->size() : 0;
            if (s->stagers_mode != 0 && s->ub_bytes > 0) {
                const uint64_t direct = std::min<uint64_t>(s->ub_bytes, s->ub_pinned + s->ub_ssd);
                const double   sh     = (double) (s->ub_bytes - direct) / (double) s->ub_bytes;
                s->stg_share = s->stg_share < 0.0 ? sh : 0.5*s->stg_share + 0.5*sh;
            }
            // [TAG_FN_L22_CPUSHARE] the copy rate of share passes (bytes per us of the pass), for the CPU budget
            if (s->share_gen != 0 && s->share_gen == s->gen && s->ub_bytes > 0 && p.wall_ms > 0.0) {
                const double r = (double) s->ub_bytes / (p.wall_ms*1e3);
                s->sctr.dma_bpu_ewma = s->sctr.dma_bpu_ewma < 0 ? r : 0.7*s->sctr.dma_bpu_ewma + 0.3*r;
            }
        }
        if (s->stats && L->pos == (int) s->layers.size() - 1) {
            const double dt = (ggml_time_us() - s->t_ub0) / 1e6;
            LLAMA_LOG_INFO("prefill-stream: ubatch %" PRIu64 ": %zu layers in %.1f ms, %.2f GiB copied (%.1f GiB/s), gates waited %.1f ms, "
                    "%.2f GiB from the hot set's slots [TAG_FN_L14_PFSD2D]\n",
                    s->ctr.ubatches, s->layers.size(), dt*1e3, s->ub_bytes/1073741824.0,
                    dt > 0 ? s->ub_bytes/1073741824.0/dt : 0.0, s->ub_wait*1e3, s->ub_d2d/1073741824.0);
        }
        // [TAG_FN_L22_PASSLOG] level 2: one line per streamed ubatch
        if (s->l22.passlog >= 2 && L->pos == (int) s->layers.size() - 1) {
            const double dt = (ggml_time_us() - s->t_ub0) / 1e6;
            llama_fn_l22_log("pass", "ubatch=%" PRIu64 " layers=%zu ms=%.1f host_gib=%.2f pinned_gib=%.2f ssd_gib=%.2f d2d_gib=%.2f "
                    "gate_wait_ms=%.1f miss_share=%.3f miss_layers=%u rows=%lld stagers=%d", s->ctr.ubatches, s->layers.size(), dt*1e3,
                    s->ub_bytes/1073741824.0, s->ub_pinned/1073741824.0, s->ub_ssd/1073741824.0, s->ub_d2d/1073741824.0,
                    s->ub_wait*1e3, s->sums.miss_share, s->sums.miss_layers, (long long) s->cur_rows, s->pool ? s->pool->size() : 0);
        }
    }
    s->cv.notify_all();
}

void pfs_destroy(pfs_state * s) {
    {
        std::lock_guard<std::mutex> lk(s->mtx);
        s->stop = true;
    }
    s->cv.notify_all();
    s->cv_done.notify_all();
    if (s->thr.joinable()) {
        s->thr.join();
    }
    if (s->ra_thr.joinable()) { // [TAG_FN_L22_READAHEAD]
        {
            std::lock_guard<std::mutex> lk(s->ra_mtx);
            s->ra_stop = true;
        }
        s->ra_cv.notify_all();
        s->ra_thr.join();
    }
    if (s->pool) {
        s->pool->stop();
    }
    if (s->copy) {
        ggml_backend_synchronize(s->copy);
    }
    for (auto & b : s->banks) {
        pfs_bank_release_tk(s, b); // [TAG_FN_L22_PIN] every copy has landed
    }
    if (s->hidden) {
        llama_moe_hot_lend_hide(s->owner, false); // [TAG_FN_L22_KEEPLEND]
        s->hidden = false;
    }
    if (s->lent) {
        llama_moe_hot_unlend(s->owner); // [TAG_FN_R1_PFS_LEND] the hot set gets its range back (no copy runs any more)
        s->lent = false;
    }
    for (auto & b : s->banks) {
        if (b.ev_done) { ggml_backend_event_free(b.ev_done); }
        if (b.ev_free) { ggml_backend_event_free(b.ev_free); }
        if (b.ev_tk)   { ggml_backend_event_free(b.ev_tk); }
        if (b.buf)     { ggml_backend_buffer_free(b.buf); }
    }
    for (auto * e : s->chunk_ev) {
        if (e) { ggml_backend_event_free(e); }
    }
    for (auto * c : s->ctxs) {
        ggml_free(c);
    }
    if (s->own_copy && s->copy) {
        ggml_backend_free(s->copy);
    }
    s->ring.release();
    s->zeros.release();
    delete s;
}

} // namespace

// [TAG_FN_L22_STAGERS] M7: the copy threads of the next pass, n_max x staged / host (4..n_max). Without live pins every byte
// is staged (n_max, today's count). The share is the larger of the measured one (EWMA of the last passes, dropped when the
// pins come or go) and P5's class bytes (a partial unpin shows there at once). A new count is a new pool, made only here:
// s->mtx held, the streamer idle (no job running or queued), so nothing is inside copy_list
static void pfs_stagers_apply(pfs_state * s) {
    if (s->stagers_mode == 0 || !s->pool || s->cur_pos >= 0 || !s->queue.empty()) {
        return;
    }
    const bool pin_live = s->l22.pin && llama_fn_pin_active();
    if (pin_live != s->stg_pin) {
        s->stg_pin   = pin_live;
        s->stg_share = -1.0;
    }
    int target = s->n_threads_max;
    if (s->stagers_mode > 0) {
        target = s->stagers_mode;
    } else if (pin_live) {
        double share = s->stg_share;
        if (s->model) {
            const uint64_t ram  = llama_fn_res_stream_bytes(s->model, LLAMA_FN_RES_RAM);
            const uint64_t pin  = llama_fn_res_stream_bytes(s->model, LLAMA_FN_RES_PIN);
            const uint64_t ssd  = llama_fn_res_stream_bytes(s->model, LLAMA_FN_RES_SSD);
            const uint64_t host = ram + pin + ssd;
            if (host > 0) {
                const double pred = (double) (ram + (s->l22.ssdtier > 0 ? 0 : ssd)) / (double) host;
                share = std::max(share, pred);
            }
        }
        if (share >= 0.0) {
            target = llama_fn_l22_stagers((uint64_t) (std::min(1.0, share)*1e6), 1000000, s->n_threads_max, 4);
        }
    }
    target = std::max(1, std::min(64, target));
    const int diff = std::abs(target - s->pool->size());
    if (diff == 0 || (s->stagers_mode < 0 && diff < 2)) {
        return;
    }
    auto np = std::make_unique<gen5::copy_pool>();
    np->set_mode(s->cp_tmp, s->cp_pf, s->cp_spin); // [TAG_FN_R25_STAGE]
    np->set_affinities(s->cp_affs);                // [TAG_FN_R25_PIN]
    np->set_grain(s->cp_grain);
    np->start(target);
    s->pool.swap(np); // the old pool's workers stop and join here (~1-3 ms)
    s->stg_restarts++;
    if (s->stg_restarts <= 3 || s->stg_restarts % 64 == 0) {
        LLAMA_LOG_INFO("prefill-stream: [TAG_FN_L22_STAGERS] %d copy threads (restart %" PRIu64 ", staged share %.2f, pins %s)\n",
                target, s->stg_restarts, s->stg_share, pin_live ? "live" : "off");
    }
}

// [TAG_FN_R1_PFS_LEND] borrow the banks from the hot set (the owner's graphs are synchronized); the first time, the bank
// and layer aliases are placed in the lent range, which is the same range every time
static bool pfs_take(pfs_state * s) {
    const size_t need = (size_t) s->n_bufs*s->bank_size;
    ggml_backend_buffer_t buf  = nullptr;
    uint8_t *             base = nullptr;
    if (!llama_moe_hot_lend(s->owner, need, &buf, &base)) {
        if (!s->lend_warned) {
            s->lend_warned = true;
            LLAMA_LOG_WARN("prefill-stream: [TAG_FN_R1_PFS_LEND] the hot set cannot lend %.0f MiB (none yet, or too small): this "
                    "prompt uses the scheduler's op offload\n", need/1048576.0);
        }
        return false;
    }
    if (!s->lend_ready) {
        bool ok = true;
        for (size_t b = 0; b < s->banks.size() && ok; ++b) {
            ok = ggml_backend_tensor_alloc(buf, s->banks[b].raw, base + b*s->bank_size) == GGML_STATUS_SUCCESS;
        }
        for (auto & L : s->layers) {
            for (int k = 0; k < 3 && ok; ++k) {
                ok = ggml_backend_tensor_alloc(buf, L.dev[k], base + (size_t) L.bank*s->bank_size + s->off[k]) == GGML_STATUS_SUCCESS;
                if (ok && L.zdev[k]) { // [TAG_FN_L22_CPUSHARE] the zero-expert alias at the same address
                    ok = ggml_backend_tensor_alloc(buf, L.zdev[k], base + (size_t) L.bank*s->bank_size + s->off[k]) == GGML_STATUS_SUCCESS;
                }
            }
        }
        if (!ok) {
            // the aliases cannot be placed: never stream (a partly placed set must not be used)
            LLAMA_LOG_WARN("prefill-stream: [TAG_FN_R1_PFS_LEND] placing the banks in the lent range failed - stream off\n");
            llama_moe_hot_unlend(s->owner);
            s->lend_warned = true;
            s->min_tokens  = INT64_MAX;
            return false;
        }
        s->lend_buf   = buf;
        s->lend_base  = base;
        s->lend_ready = true;
    } else if (buf != s->lend_buf || base != s->lend_base) {
        llama_moe_hot_unlend(s->owner); // cannot happen: the hot buffer never moves
        return false;
    }
    {
        std::lock_guard<std::mutex> lk(s->mtx);
        for (auto & B : s->banks) {
            B.holds    = -1;
            B.complete = false;
            B.consumed = true;
            B.used_pos = -1;
            B.partial  = false; // [TAG_FN_L22_CPUSHARE] the range held hot slots until now
            B.zc[0] = B.zc[1] = B.zc[2] = 0;
        }
        s->ctr.lends++;
        pfs_stagers_apply(s); // [TAG_FN_L22_STAGERS] before any job of this lend
        // [TAG_FN_L22_REFILL] (c) bank 0 starts filling now, before the first gate (which keeps a running or finished copy
        // of layer 0); the slot snapshot of its device copies is taken by the streamer, and no hot-set step runs before
        // that gate
        if (s->l22.prestage && !s->layers.empty()) {
            s->queue.push_back({ 0, s->gen });
            s->cv.notify_all();
        }
    }
    s->lent = true;
    return true;
}

// [TAG_FN_R1_PFS_LEND] give the banks back: the queued and the running copies stop and land, then the hot set refills them
static void pfs_return(pfs_state * s) {
    {
        std::unique_lock<std::mutex> lk(s->mtx);
        s->gen++; // the running job sees a new generation and stops; queued ones never start
        std::deque<pfs_job> q;
        if (s->cur_pos >= 0 && !s->queue.empty()) {
            q.push_back(s->queue.front()); // the streamer pops its running job itself
        }
        s->queue.swap(q);
        s->cv.notify_all();
        s->cv_done.wait(lk, [&]() { return s->cur_pos < 0 || s->stop; });
        for (auto & B : s->banks) {
            B.holds    = -1;
            B.complete = false;
            B.consumed = true;
            B.used_pos = -1;
            B.partial  = false; // [TAG_FN_L22_CPUSHARE] the hot set refills the range
            B.zc[0] = B.zc[1] = B.zc[2] = 0;
        }
    }
    if (s->copy) {
        ggml_backend_synchronize(s->copy); // what it dispatched has landed
    }
    for (auto & B : s->banks) {
        pfs_bank_release_tk(s, B); // [TAG_FN_L22_PIN]
    }
    s->lent = false;
    if (s->hidden) {
        llama_moe_hot_lend_hide(s->owner, false); // [TAG_FN_L22_KEEPLEND] the lent layers' hot chains come back with them
        s->hidden = false;
    }
    if (s->async_ret) {
        s->async_ret = false;
        llama_moe_hot_unlend_async(s->owner); // [TAG_FN_L22_ENDOVERLAP] refill through the upload worker
    } else {
        llama_moe_hot_unlend(s->owner);
    }
}

bool llama_prefill_stream_init_layers(const std::vector<llama_moe_gen5_layer_desc> & layers, const llama_moe_gen5_device & d, const void * owner) {
    std::lock_guard<std::mutex> init_lock(g_pfs_init);
    if (g_pfs || layers.empty() || !d.compute || !d.buft) {
        return false;
    }
    auto * s = new pfs_state();
    s->owner      = owner;
    s->dev        = d.dev;
    s->compute    = d.compute;
    s->min_tokens = gen5::env_int("LLAMA_PREFILL_STREAM_MIN", 32, 9, 1 << 20);
    s->n_bufs     = gen5::env_int("LLAMA_PREFILL_STREAM_BUFS", 2, 1, 3);
    s->wrap       = gen5::env_int("LLAMA_PREFILL_STREAM_WRAP", 1, 0, 1) != 0;
    s->stats      = gen5::env_flag("LLAMA_PREFILL_STREAM_STATS");
    s->chunk      = (size_t) gen5::env_int("LLAMA_PREFILL_STREAM_CHUNK_MIB", 32, 1, 256) << 20;
    s->lend       = gen5::env_int("LLAMA_PREFILL_STREAM_LEND", 0, 0, 1) != 0; // [TAG_FN_R1_PFS_LEND]
    s->ahead      = gen5::env_int("LLAMA_PREFILL_STREAM_AHEAD", 2, 0, 8); // [TAG_FN_L15_PFSAHEAD]
    size_t       ring_bytes = (size_t) gen5::env_int("LLAMA_PREFILL_STREAM_RING_MIB", 512, 2, 4096) << 20;
    const int    n_threads  = gen5::env_int("LLAMA_PREFILL_STREAM_THREADS", 8, 1, 64);
    // [TAG_FN_R25_STAGE] LLAMA_PREFILL_STREAM_CACHED=1: the copy threads write the ring with plain stores and the ring is
    // small enough to stay in the L3 (default 4 x 8 MiB), so the DMA reads the staged lines from the cache. With
    // non-temporal stores into the 512 MiB ring every staged byte crossed DRAM three times (source read, ring write, DMA
    // read): the stream pass ran at ~22 GB/s against a measured DRAM copy ceiling of ~31 GB/s (r25 probe). An explicit
    // LLAMA_PREFILL_STREAM_RING_MIB / _CHUNK_MIB still wins over the cached sizes
    s->cp_tmp = gen5::env_int("LLAMA_PREFILL_STREAM_CACHED", 0, 0, 1) != 0;
    if (s->cp_tmp) {
        const char * er = getenv("LLAMA_PREFILL_STREAM_RING_MIB");
        const char * ec = getenv("LLAMA_PREFILL_STREAM_CHUNK_MIB");
        if (!er || !er[0]) {
            ring_bytes = (size_t) gen5::env_int("LLAMA_PREFILL_STREAM_CACHED_RING_MIB", 32, 2, 512) << 20;
        }
        if (!ec || !ec[0]) {
            s->chunk = (size_t) gen5::env_int("LLAMA_PREFILL_STREAM_CACHED_CHUNK_MIB", 8, 1, 64) << 20;
        }
        s->cp_pf = gen5::env_int("LLAMA_PREFILL_STREAM_CACHED_PF", 1, 0, 1) != 0;
    }
    s->cp_spin = gen5::env_int("LLAMA_PREFILL_STREAM_SPIN_US", s->cp_tmp ? 200 : 0, 0, 10000);
    // [TAG_FN_R25_STAGE] LLAMA_PREFILL_STREAM_ASYNC=1: the streamer dispatches a chunk's DMA while the copy threads fill the
    // next slot (the streamer no longer copies a part itself). Default on with the cached ring: the r25 probe (copy + DMA,
    // workers one CCD each) 30.1 GB/s synchronous, 31.2 GB/s async; the r25 GPU test ran cached + async + PIN 2
    s->cp_async = gen5::env_int("LLAMA_PREFILL_STREAM_ASYNC", s->cp_tmp ? 1 : 0, 0, 1) != 0;
    // [TAG_FN_R25_PIN] LLAMA_PREFILL_STREAM_PIN: 1 each copy worker on its own physical core, the cores in turn from each L3
    // domain; 2 the workers in turn on the whole CCDs (the scheduler still moves a worker inside its CCD). Left to the
    // scheduler the 16 copy threads ran on the cache CCD (r25 probe: 22.2 GB/s copy + DMA; one per core across both CCDs
    // 29.6 GB/s; 32 free threads 22.6 GB/s)
    s->cp_pin  = gen5::env_int("LLAMA_PREFILL_STREAM_PIN", 0, 0, 2);
    s->cp_affs = gen5::copy_affinities(s->cp_pin);
    s->cp_grain = (size_t) gen5::env_int("LLAMA_PREFILL_STREAM_GRAIN_KIB", 0, 0, 65536) << 10;
    if (s->min_tokens > 32 && !s->lend) { // [TAG_FN_R1_PFS_LEND] a lending stream reserves with the op offload
        LLAMA_LOG_WARN("prefill-stream: LLAMA_PREFILL_STREAM_MIN=%" PRId64 " > 32: ubatches of 32..%" PRId64 " tokens keep the "
                "op-offload copies, which the compute buffer was not reserved for\n", s->min_tokens, s->min_tokens - 1);
    }

    const bool cpu_dev = !d.dev || ggml_backend_dev_type(d.dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
    ggml_backend_buffer_type_t host_buft = !cpu_dev ? ggml_backend_dev_host_buffer_type(d.dev) : nullptr;

    // [TAG_FN_L22_CPUSHARE] P16's share mode, configured by the owner context before this init
    if (const llama_fn_cps_cfg * cps = llama_fn_cpushare_cfg()) {
        s->cps_mode    = cps->mode;
        s->zexp        = cps->mode == 1;
        s->cps_wait_ms = std::max(1, cps->wait_ms);
    }
    ggml_context * zctx = nullptr; // the zero-expert shapes, only to ask the buffer type for their alloc size
    if (s->zexp) {
        ggml_init_params zp = { ggml_tensor_overhead()*(3*layers.size() + 4), nullptr, true };
        zctx = ggml_init(zp);
        s->zexp = zctx != nullptr;
    }

    // one layout for every bank: up at 0, gate and down after the largest up / gate of any layer
    const size_t align = std::max<size_t>(256, ggml_backend_buft_get_alignment(d.buft));
    size_t max_alloc[3] = { 0, 0, 0 };
    s->layers.resize(layers.size());
    for (size_t p = 0; p < layers.size(); ++p) {
        pfs_layer & L = s->layers[p];
        L.il     = layers[p].il;
        L.pos    = (int) p;
        L.src[0] = layers[p].up;
        L.src[1] = layers[p].gate;
        L.src[2] = layers[p].down;
        L.src_pinned = host_buft != nullptr;
        for (int k = 0; k < 3; ++k) {
            L.nbytes[k] = ggml_nbytes(L.src[k]);
            const size_t a = ggml_backend_buft_get_alloc_size(d.buft, L.src[k]);
            L.pad[k] = a > L.nbytes[k] ? a - L.nbytes[k] : 0;
            max_alloc[k] = std::max(max_alloc[k], a);
            L.src_pinned = L.src_pinned && L.src[k]->buffer && ggml_backend_buffer_get_type(L.src[k]->buffer) == host_buft;
        }
        // [TAG_FN_L14_ARENA] the expert arena's registered layers: direct copies, no staging ring
        bool arena = true;
        for (int k = 0; k < 3; ++k) {
            arena = arena && llama_fn_arena_pinned(L.src[k]->data, ggml_nbytes(L.src[k]));
        }
        L.src_pinned = L.src_pinned || arena;
        s->by_up[L.src[0]] = (int) p;
        // [TAG_FN_L22_CPUSHARE] expert-granular ranges, and with the zero expert the bytes behind the layer that must be 0
        const int64_t ne_e = L.src[0]->ne[2];
        L.exp_ok = ne_e > 0;
        for (int k = 0; k < 3; ++k) {
            L.exp_ok = L.exp_ok && L.src[k]->ne[2] == ne_e && L.src[k]->ne[3] == 1 && L.src[k]->nb[2]*(size_t) ne_e == L.nbytes[k];
        }
        s->n_exp_max = std::max(s->n_exp_max, ne_e);
        if (zctx && L.exp_ok) {
            for (int k = 0; k < 3; ++k) {
                const ggml_tensor * t = L.src[k];
                ggml_tensor * z = ggml_new_tensor_4d(zctx, t->type, t->ne[0], t->ne[1], t->ne[2] + 1, t->ne[3]);
                const size_t za = ggml_backend_buft_get_alloc_size(d.buft, z);
                L.zpad[k]    = za > L.nbytes[k] ? za - L.nbytes[k] : 0;
                max_alloc[k] = std::max(max_alloc[k], za);
            }
        }
    }
    if (zctx) {
        ggml_free(zctx);
    }
    s->off[0]    = 0;
    s->off[1]    = align_up(max_alloc[0], align);
    s->off[2]    = s->off[1] + align_up(max_alloc[1], align);
    s->bank_size = s->off[2] + align_up(max_alloc[2], align);

    auto fail = [&](const char * why) {
        LLAMA_LOG_WARN("prefill-stream: %s - off, prefill keeps the scheduler's op-offload copies\n", why);
        pfs_destroy(s);
        return false;
    };

    // banks
    for (int b = 0; b < s->n_bufs; ++b) {
        ggml_init_params ip = { ggml_tensor_overhead()*2, nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        if (!ctx) {
            return fail("no context");
        }
        s->ctxs.push_back(ctx);
        pfs_bank bank;
        bank.raw = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, (int64_t) s->bank_size);
        ggml_format_name(bank.raw, "pfs_bank.%d", b);
        if (s->lend) {
            // [TAG_FN_R1_PFS_LEND] placed in the hot set's range at the first lend (pfs_take)
            bank.ev_done = gen5::ev_new(d.dev);
            bank.ev_free = gen5::ev_new(d.dev);
            s->banks.push_back(bank);
            continue;
        }
        bank.buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, d.buft);
        if (!bank.buf) {
            if (b == 0) {
                return fail("bank allocation failed");
            }
            LLAMA_LOG_WARN("prefill-stream: bank %d of %.0f MiB failed, running with %d bank(s)\n", b, s->bank_size/1048576.0, b);
            s->n_bufs = b;
            break;
        }
        ggml_backend_buffer_set_usage(bank.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_buffer_clear(bank.buf, 0);
        bank.ev_done = gen5::ev_new(d.dev);
        bank.ev_free = gen5::ev_new(d.dev);
        s->banks.push_back(bank);
    }

    // per layer aliases with the layer's own types and shapes
    {
        ggml_init_params ip = { ggml_tensor_overhead()*((s->zexp ? 6 : 3)*layers.size() + 8), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        if (!ctx) {
            return fail("no context");
        }
        s->ctxs.push_back(ctx);
        static const char * nm[3] = { "up", "gate", "down" };
        for (auto & L : s->layers) {
            L.bank = L.pos % s->n_bufs;
            pfs_bank & B = s->banks[L.bank];
            for (int k = 0; k < 3; ++k) {
                const ggml_tensor * t = L.src[k];
                L.dev[k] = ggml_new_tensor_4d(ctx, t->type, t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
                ggml_format_name(L.dev[k], "pfs.%d.%s", L.il, nm[k]);
                if (s->zexp && L.exp_ok) { // [TAG_FN_L22_CPUSHARE] one more expert, the zero one, behind the layer
                    L.zdev[k] = ggml_new_tensor_4d(ctx, t->type, t->ne[0], t->ne[1], t->ne[2] + 1, t->ne[3]);
                    ggml_format_name(L.zdev[k], "pfsz.%d.%s", L.il, nm[k]);
                }
                if (s->lend) {
                    continue; // [TAG_FN_R1_PFS_LEND] placed at the first lend
                }
                void * addr = (char *) ggml_backend_buffer_get_base(B.buf) + s->off[k];
                if (ggml_backend_tensor_alloc(B.buf, L.dev[k], addr) != GGML_STATUS_SUCCESS) {
                    return fail("alias allocation failed");
                }
                if (L.zdev[k] && ggml_backend_tensor_alloc(B.buf, L.zdev[k], addr) != GGML_STATUS_SUCCESS) {
                    return fail("zero-expert alias allocation failed");
                }
            }
            L.view.up     = L.dev[0];
            L.view.gate   = L.dev[1];
            L.view.down   = L.dev[2];
            L.view.handle = &L;
        }
    }

    // staging ring (only for sources that are not pinned already)
    bool need_ring = false;
    for (const auto & L : s->layers) {
        need_ring = need_ring || !L.src_pinned;
    }
    if (need_ring) {
        s->chunk = std::min(s->chunk, ring_bytes/2);
        s->n_chunks = (int) std::max<size_t>(2, ring_bytes / s->chunk);
        if (!s->ring.alloc(d.dev, (size_t) s->n_chunks*s->chunk, "prefill stream ring")) {
            return fail("no staging ring");
        }
        s->chunk_ev.resize(s->n_chunks, nullptr);
        s->chunk_state.assign(s->n_chunks, 0);
        for (auto & e : s->chunk_ev) {
            e = gen5::ev_new(d.dev);
        }
    }
    if (!s->zeros.alloc(d.dev, s->zexp ? 1u << 20 : 64u << 10, "prefill stream zeros")) { // [TAG_FN_L22_CPUSHARE] 1 MiB
        return fail("no zero page");
    }
    memset(s->zeros.ptr, 0, s->zeros.size);

    if (!cpu_dev) {
        s->copy = ggml_backend_dev_init(d.dev, nullptr);
        s->own_copy = s->copy != nullptr;
        // [TAG_FN_L3_HOST_POKE] the device backend's flush, if it has one (it does nothing unless the switch is on)
        if (ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(d.dev)) {
            s->poke = (void (*)(ggml_backend_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_stream_poke");
            s->ev_done = (bool (*)(ggml_backend_event_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_fn_event_done");
            if (gen5::env_flag("LLAMA_PREFILL_STREAM_D2D")) { // [TAG_FN_L14_PFSD2D]
                s->d2d = (bool (*)(ggml_backend_t, int, void * const *, const void * const *, const size_t *))
                        ggml_backend_reg_get_proc_address(reg, "ggml_backend_copy_d2d_batch_async");
            }
        }
    } else {
        s->copy = d.compute; // CPU: copies are synchronous memcpy
    }
    if (!s->copy) {
        return fail("no copy backend");
    }

    // [TAG_FN_L22_CPUSHARE] late / skip masks per position
    if (s->cps_mode > 0) {
        const size_t np = s->layers.size();
        s->late.assign(np*(size_t) s->n_exp_max, 0);
        s->late_any.assign(np, 0);
        s->skip.assign(np*(size_t) s->n_exp_max, 0);
        s->pub_gen.assign(np, 0);
        s->pub_us.assign(np, 0);
    }

    s->pool = std::make_unique<gen5::copy_pool>();
    s->pool->set_mode(s->cp_tmp, s->cp_pf, s->cp_spin); // [TAG_FN_R25_STAGE]
    s->pool->set_affinities(s->cp_affs);                // [TAG_FN_R25_PIN]
    s->pool->set_grain(s->cp_grain);
    s->pool->start(n_threads);
    s->n_threads_max = s->pool->size(); // [TAG_FN_L22_STAGERS]
    s->thr = std::thread(pfs_run, s);
    g_pfs = s;

    LLAMA_LOG_INFO("prefill-stream: %zu host expert layers, ubatches >= %" PRId64 " tokens, %d VRAM bank(s) x %.0f MiB%s, "
            "%s ring %zu x %zu MiB, %d copy threads, wrap %s%s\n", s->layers.size(), s->min_tokens, s->n_bufs,
            s->bank_size/1048576.0, s->lend ? " borrowed from the hot set while a prompt streams [TAG_FN_R1_PFS_LEND]" : "",
            s->ring.pinned ? "pinned" : (need_ring ? "plain" : "no"), (size_t) s->n_chunks,
            s->chunk >> 20, s->pool->size(), s->wrap ? "on" : "off",
            s->d2d ? ", the hot set's experts slot -> bank on the device [TAG_FN_L14_PFSD2D]" : "");
    LLAMA_LOG_INFO("prefill-stream: [TAG_FN_R25_STAGE] ring copies %s%s, copy threads spin %d us before they sleep%s\n",
            s->cp_tmp ? "with plain stores (the ring stays in the L3, the DMA reads it there)" : "non-temporal (through DRAM)",
            s->cp_pf ? ", source prefetch NTA" : "", s->cp_spin,
            s->cp_async ? ", a chunk's DMA is queued while the next chunk is copied" : "");
    if (s->cp_pin > 0) { // [TAG_FN_R25_PIN]
        std::string al;
        for (size_t i = 0; i < s->cp_affs.size() && i < 16; ++i) {
            char b[40];
            snprintf(b, sizeof(b), "%s%d:0x%llx", al.empty() ? "" : ",", s->cp_affs[i].group, (unsigned long long) s->cp_affs[i].mask);
            al += b;
        }
        LLAMA_LOG_INFO("prefill-stream: [TAG_FN_R25_PIN] copy workers pinned (mode %d: %s): %s\n", s->cp_pin,
                s->cp_pin == 1 ? "one physical core each, the L3 domains in turn" : "one CCD each, in turn",
                al.empty() ? "no topology - not pinned" : al.c_str());
    }
    if (s->cps_mode > 0) {
        LLAMA_LOG_INFO("prefill-stream: [TAG_FN_L22_CPUSHARE] share mode %d: late experts after each layer's plan (wait <= %d ms)%s\n",
                s->cps_mode, s->cps_wait_ms, s->zexp ? ", a zero expert behind every layer in the banks" : "");
    }
    return true;
}

void llama_prefill_stream_free(const void * owner) {
    std::lock_guard<std::mutex> init_lock(g_pfs_init);
    if (!g_pfs || g_pfs->owner != owner) {
        return;
    }
    pfs_state * s = g_pfs;
    g_pfs = nullptr;
    pfs_destroy(s);
}

const llama_pfs_view * llama_prefill_stream_lookup(ggml_backend_sched_t sched, const ggml_tensor * up_exps, int64_t n_tokens) {
    pfs_state * s = g_pfs;
    if (!s || n_tokens < s->min_tokens || !gen5::sched_has(sched, s->compute)) {
        return nullptr;
    }
    if (s->lend && !s->lent) {
        return nullptr; // [TAG_FN_R1_PFS_LEND] not borrowed (also at the reserve): the scheduler's op offload
    }
    auto it = s->by_up.find(up_exps);
    return it != s->by_up.end() ? &s->layers[it->second].view : nullptr;
}

ggml_tensor * llama_prefill_stream_build_gate(ggml_context * ctx, ggml_cgraph * gf, const llama_pfs_view * v, ggml_tensor * cur) {
    pfs_layer * L = (pfs_layer *) v->handle;
    ggml_tensor * args[1] = { ggml_view_1d(ctx, cur, 1, 0) };
    ggml_tensor * g = ggml_custom_4d(ctx, GGML_TYPE_F32, 1, 1, 1, 1, args, 1, pfs_gate_op, 1, L);
    ggml_format_name(g, "pfs_gate-%d", L->il);
    ggml_build_forward_expand(gf, g);
    return g;
}

ggml_tensor * llama_prefill_stream_build_release(ggml_context * ctx, ggml_cgraph * gf, const llama_pfs_view * v, ggml_tensor * down) {
    pfs_layer * L = (pfs_layer *) v->handle;
    ggml_tensor * args[1] = { ggml_view_1d(ctx, down, 1, 0) };
    ggml_tensor * r = ggml_custom_4d(ctx, GGML_TYPE_F32, 1, 1, 1, 1, args, 1, pfs_release_op, 1, L);
    ggml_format_name(r, "pfs_release-%d", L->il);
    ggml_build_forward_expand(gf, r);
    return r;
}

llama_pfs_counters llama_prefill_stream_get_counters() {
    pfs_state * s = g_pfs;
    if (!s) {
        return {};
    }
    std::lock_guard<std::mutex> lk(s->mtx);
    return s->ctr;
}

size_t llama_prefill_stream_lend_bytes(const void * owner) {
    const pfs_state * s = g_pfs;
    if (!s || !s->lend || s->owner != owner || s->min_tokens == INT64_MAX) {
        return 0;
    }
    return (size_t) s->n_bufs*s->bank_size;
}

bool llama_prefill_stream_before_ubatch(const void * owner, ggml_backend_sched_t sched, int64_t n_tokens) {
    pfs_state * s = g_pfs;
    if (!s || !s->lend || s->owner != owner) {
        return false;
    }
    if (n_tokens > 0) {
        s->cur_rows = n_tokens; // [TAG_FN_L22_XOVER] the pass record's rows
    }
    const bool want = n_tokens >= s->min_tokens;
    if (want == s->lent) {
        return false;
    }
    // [TAG_FN_L22_KEEPLEND] a narrow ubatch of the owner's graphs (not a reserve, not a switch to SMALL: n_tokens 0) keeps
    // the banks while prompt rows follow it (KEEPLEND), or when it is a prompt's narrow last part (ENDOVERLAP). The lent
    // layers' tables say "not hot" for every expert (llama_moe_hot_lend), but their zero slot may hold bank bytes now:
    // the hot set hides those layers from the hot chains until the banks go back (the bridge / CPU computes them), and
    // the hide bumps the hot generation, so no graph built under the other state is reused
    if (!want && sched && n_tokens > 0 && ((s->l22.keeplend && s->rows_after > 0) || (s->l22.endoverlap && s->tail_now))) {
        if (!s->hidden) {
            ggml_backend_sched_synchronize(sched); // the owner's last graph is done before the tables / lookups change
            llama_moe_hot_lend_hide(s->owner, true);
            s->hidden = true;
        }
        pfs_release_landed(s); // [TAG_FN_L22_PIN] R-REV-3: the kept banks' landed copies give their tickets back
        std::lock_guard<std::mutex> lk(s->mtx);
        s->sums.keeps++;
        return false;
    }
    if (sched) {
        ggml_backend_sched_synchronize(sched); // the owner's last graph is done: nothing reads the slots or the banks
    }
    const int64_t t_fix0 = ggml_time_us(); // [TAG_FN_L22_XOVER] the stream's fixed cost
    if (want) {
        const bool ok = pfs_take(s);
        s->fix_ms += (ggml_time_us() - t_fix0)/1e3;
        return ok;
    }
    pfs_return(s);
    s->fix_ms += (ggml_time_us() - t_fix0)/1e3;
    return true;
}

// [TAG_FN_L22_P4] ----------------------------------------------------------------------------------------------------

void llama_prefill_stream_l22_config(const void * owner, const llama_model * model, const llama_pfs_l22_cfg & cfg) {
    std::lock_guard<std::mutex> init_lock(g_pfs_init);
    pfs_state * s = g_pfs;
    if (!s || s->owner != owner) {
        return;
    }
    s->model = model;
    s->l22   = cfg;
    if (s->l22.ra_call == 0) {
        s->l22.ra_call = (size_t) 2048 << 20;
    }
    if (!s->lend) {
        s->l22.keeplend   = false; // only a lending stream returns its banks between ubatches
        s->l22.endoverlap = false;
        s->l22.prestage   = false;
    }
    if (s->l22.readahead && !s->ra_thr.joinable() && !s->layers.empty()) {
        s->l_miss.reset(new std::atomic<uint16_t>[s->layers.size()]);
        s->l_map.reset(new std::atomic<uint8_t>[s->layers.size()]); // [TAG_FN_L22_RESSET_YIELD]
        for (size_t p = 0; p < s->layers.size(); ++p) {
            s->l_miss[p].store(0);
            s->l_map[p].store(0);
        }
        s->ra_thr = std::thread(pfs_ra_run, s);
    }
    if (s->l22.keeplend || s->l22.endoverlap || s->l22.prestage || s->l22.readahead || s->l22.ssdtier > 0 || s->l22.pin) {
        LLAMA_LOG_INFO("prefill-stream: [TAG_FN_L22_P4] round 22:%s%s%s%s%s%s\n",
                s->l22.keeplend ? " KEEPLEND (narrow ubatches with prompt rows after them keep the banks)" : "",
                s->l22.endoverlap ? " ENDOVERLAP (the prompt's last part keeps them, the answer's first step returns them)" : "",
                s->l22.prestage ? " REFILL (layer 0 copied at the lend)" : "",
                s->l22.readahead ? " READAHEAD (every layer sampled at gate 0)" : "",
                s->l22.ssdtier > 0 ? (s->l22.ssdtier >= 2 ? " SSDTIER=2" : " SSDTIER=1") : "",
                s->l22.pin ? " PIN (pinned segments straight to the banks)" : "");
    }
}

void llama_prefill_stream_l22_set_after(const void * owner, int64_t rows_after, bool tail) {
    pfs_state * s = g_pfs;
    if (!s || s->owner != owner) {
        return;
    }
    s->rows_after = rows_after; // the owner's thread only (with before_ubatch)
    s->tail_now   = tail;
}

void llama_prefill_stream_l22_async_return(const void * owner, bool on) {
    pfs_state * s = g_pfs;
    if (!s || s->owner != owner) {
        return;
    }
    s->async_ret = on && s->lent;
}

bool llama_prefill_stream_l22_lent(const void * owner) {
    const pfs_state * s = g_pfs;
    return s && s->owner == owner && s->lent;
}

void llama_prefill_stream_l22_hold(const void * owner, ggml_backend_sched_t sched) {
    pfs_state * s = g_pfs;
    if (!s || s->owner != owner || !s->lent || s->hidden) {
        return;
    }
    if (sched) {
        ggml_backend_sched_synchronize(sched); // the owner's last graph is done before the lookups change
    }
    llama_moe_hot_lend_hide(s->owner, true);
    s->hidden = true;
}

llama_pfs_l22_sums llama_prefill_stream_l22_take(const void * owner) {
    pfs_state * s = g_pfs;
    if (!s || s->owner != owner) {
        return {};
    }
    std::lock_guard<std::mutex> lk(s->mtx);
    llama_pfs_l22_sums r = s->sums;
    const double miss = s->sums.miss_share;
    const uint32_t ml = s->sums.miss_layers;
    s->sums = {};
    s->sums.miss_share  = miss; // the last sample stays
    s->sums.miss_layers = ml;
    return r;
}

// [TAG_FN_L22_P10] ---------------------------------------------------------------------------------------------------

void llama_prefill_stream_set_stagers(const void * owner, int mode) {
    std::lock_guard<std::mutex> init_lock(g_pfs_init);
    pfs_state * s = g_pfs;
    if (!s || s->owner != owner) {
        return;
    }
    std::lock_guard<std::mutex> lk(s->mtx);
    s->stagers_mode = mode;
    s->stg_share    = -1.0;
    if (mode != 0) {
        LLAMA_LOG_INFO("prefill-stream: [TAG_FN_L22_STAGERS] copy threads %s (pool of %d at the start)\n",
                mode > 0 ? "fixed" : "by the staged share of the host bytes (16 x staged / host, 4..16) while pins are live",
                s->n_threads_max);
    }
}

bool llama_prefill_stream_set_min(const void * owner, int64_t n) {
    pfs_state * s = g_pfs;
    if (!s || s->owner != owner) {
        return false;
    }
    if (s->min_tokens != INT64_MAX) { // a stream turned off stays off
        s->min_tokens = std::max<int64_t>(9, n);
    }
    return true;
}

int64_t llama_prefill_stream_get_min(const void * owner) {
    const pfs_state * s = g_pfs;
    return s && s->owner == owner ? s->min_tokens : 0;
}

bool llama_prefill_stream_pass_info(const void * owner, llama_pfs_pass * out) {
    pfs_state * s = g_pfs;
    if (!s || s->owner != owner || !out) {
        return false;
    }
    std::lock_guard<std::mutex> lk(s->mtx);
    if (s->pass_seq == s->pass_read) {
        return false;
    }
    s->pass_read = s->pass_seq;
    *out = s->last_pass;
    return true;
}

double llama_prefill_stream_fix_take(const void * owner) {
    pfs_state * s = g_pfs;
    if (!s || s->owner != owner) {
        return 0.0;
    }
    const double ms = s->fix_ms;
    s->fix_ms = 0.0;
    return ms;
}

// [TAG_FN_L22_CPUSHARE] -------------------------------------------------------------------------------------------------

int llama_prefill_stream_pos(const llama_pfs_view * v) {
    return v && v->handle ? ((const pfs_layer *) v->handle)->pos : -1;
}

int llama_prefill_stream_n_pos() {
    const pfs_state * s = g_pfs;
    return s ? (int) s->layers.size() : 0;
}

bool llama_prefill_stream_view_z(const llama_pfs_view * v, ggml_tensor * out[3]) {
    const pfs_state * s = g_pfs;
    if (!s || !s->zexp || !v || !v->handle) {
        return false;
    }
    const pfs_layer * L = (const pfs_layer *) v->handle;
    for (int k = 0; k < 3; ++k) {
        if (!L->zdev[k]) {
            return false;
        }
        out[k] = L->zdev[k];
    }
    return true;
}

ggml_backend_t llama_prefill_stream_device() {
    const pfs_state * s = g_pfs;
    return s ? s->compute : nullptr;
}

void llama_prefill_stream_share_begin(const uint8_t * late, int n_expert) {
    pfs_state * s = g_pfs;
    if (!s || s->layers.empty()) {
        return;
    }
    if (s->lend && !s->lent) {
        GGML_ABORT("prefill-stream: a streamed graph ran while its banks were not borrowed");
    }
    const int64_t t0 = ggml_time_us();
    {
        std::lock_guard<std::mutex> lk(s->mtx);
        const bool masks = late != nullptr && s->cps_mode > 0 && n_expert > 0 && n_expert <= s->n_exp_max;
        if (s->cps_mode > 0) {
            std::fill(s->late.begin(), s->late.end(), 0);
            std::fill(s->late_any.begin(), s->late_any.end(), 0);
            if (masks) {
                for (size_t p = 1; p < s->layers.size(); ++p) { // position 0 is never late
                    uint8_t * dst = s->late.data() + p*(size_t) s->n_exp_max;
                    const uint8_t * src = late + p*(size_t) n_expert;
                    uint8_t any = 0;
                    for (int e = 0; e < n_expert; ++e) {
                        dst[e] = src[e] != 0;
                        any |= dst[e];
                    }
                    s->late_any[p] = any;
                }
            }
        }
        pfs_ubatch_begin(s, t0);
        s->share_gen = masks ? s->gen : 0;
    }
    s->cv.notify_all();
}

bool llama_prefill_stream_share_active() {
    pfs_state * s = g_pfs;
    if (!s) {
        return false;
    }
    std::lock_guard<std::mutex> lk(s->mtx);
    return s->share_gen != 0 && s->share_gen == s->gen;
}

void llama_prefill_stream_share_publish(const llama_pfs_view * v, const uint8_t * skip, int n_expert) {
    pfs_state * s = g_pfs;
    if (!s || s->cps_mode <= 0 || !v || !v->handle) {
        return;
    }
    const pfs_layer * L = (const pfs_layer *) v->handle;
    if (L->pos < 0 || L->pos >= (int) s->layers.size() || &s->layers[L->pos] != L) {
        return;
    }
    {
        std::lock_guard<std::mutex> lk(s->mtx);
        uint8_t * dst = s->skip.data() + (size_t) L->pos*s->n_exp_max;
        std::fill(dst, dst + s->n_exp_max, 0);
        if (skip != nullptr && n_expert > 0 && n_expert <= s->n_exp_max) {
            for (int e = 0; e < n_expert; ++e) {
                dst[e] = skip[e] != 0;
            }
        }
        s->pub_gen[L->pos] = s->gen;
        s->pub_us[L->pos]  = ggml_time_us();
    }
    s->cv.notify_all();
}

bool llama_prefill_stream_layer_full(const llama_pfs_view * v) {
    pfs_state * s = g_pfs;
    if (!s || !v || !v->handle) {
        return false;
    }
    const pfs_layer * L = (const pfs_layer *) v->handle;
    if (L->pos < 0 || L->pos >= (int) s->layers.size() || &s->layers[L->pos] != L) {
        return false;
    }
    std::lock_guard<std::mutex> lk(s->mtx);
    const pfs_bank & B = s->banks[L->bank];
    return B.holds == L->pos && B.complete && !B.partial;
}

ggml_tensor * llama_prefill_stream_build_gate_share(ggml_context * ctx, ggml_cgraph * gf, const llama_pfs_view * v, ggml_tensor * dep) {
    pfs_layer * L = (pfs_layer *) v->handle;
    ggml_tensor * args[1] = { dep };
    ggml_tensor * g = ggml_custom_4d(ctx, GGML_TYPE_F32, 1, 1, 1, 1, args, 1, pfs_gate_share_op, 1, L);
    ggml_format_name(g, "pfs_gate_share-%d", L->il);
    ggml_build_forward_expand(gf, g);
    return g;
}

llama_pfs_share_counters llama_prefill_stream_share_counters() {
    pfs_state * s = g_pfs;
    if (!s) {
        return {};
    }
    std::lock_guard<std::mutex> lk(s->mtx);
    return s->sctr;
}
