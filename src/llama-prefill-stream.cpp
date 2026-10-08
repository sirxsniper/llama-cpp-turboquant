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
#include "llama-fn-arena.h"  // [TAG_FN_L14_ARENA]
#include "llama-fn-tier.h" // [TAG_FN_L15_WARM]

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
#include <cinttypes>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
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
    gen5::copy_pool                   pool;

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
std::mutex  g_pfs_init;

size_t align_up(size_t x, size_t a) {
    return (x + a - 1) / a * a;
}

// the streamer thread
void pfs_run(pfs_state * s) {
    struct item { int slot; const uint8_t * src; size_t dst; size_t n; };
    std::vector<item> staged;
    int next_slot = 0;
    // [TAG_FN_L14_PFSD2D]
    std::vector<int32_t>                     hot_slot;
    std::vector<std::pair<size_t, size_t>>   ranges;
    std::vector<gen5::copy_seg>              segs;
    std::vector<item>                        pieces;
    std::vector<void *>                      d2d_dst;
    std::vector<const void *>                d2d_src;
    std::vector<size_t>                      d2d_len;

    for (;;) {
        pfs_job job;
        {
            std::unique_lock<std::mutex> lk(s->mtx);
            s->cv.wait(lk, [&]() { return s->stop || !s->queue.empty(); });
            if (s->stop) {
                return;
            }
            job = s->queue.front();
            pfs_bank & b0 = s->banks[s->layers[job.pos].bank];
            if (b0.holds == job.pos && b0.complete) {
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
        }
        const pfs_layer & L = s->layers[job.pos];
        pfs_bank &        B = s->banks[L.bank];
        llama_fn_warm_wait_layer(L.il); // [TAG_FN_L15_WARM] the warm pass brings this layer in first
        // [TAG_FN_L15_PFSAHEAD] the next layers' pages: read ahead of the copies (positions wrap to 0 for the next ubatch)
        if (s->ahead > 0 && !L.src_pinned && !llama_fn_warm_running()) {
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
            }
            gen5::ev_wait(s->copy, B.ev_free, s->compute);
            dispatching = true;
            return true;
        };
        auto dispatch = [&](const item & it) {
            ggml_backend_tensor_set_async(s->copy, B.raw, it.src, it.dst, it.n);
            if (it.slot >= 0) {
                gen5::ev_record(s->chunk_ev[it.slot], s->copy);
                s->chunk_state[it.slot] = 2;
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

        // the byte ranges of tensor k in `ranges` from the host: through the ring, a slot filled by one copy of its pieces
        auto stream_ranges = [&](int k) {
            const uint8_t * src = (const uint8_t *) L.src[k]->data;
            if (L.src_pinned) {
                for (const auto & r : ranges) {
                    for (size_t o = r.first; o < r.second && !cancelled; o += s->chunk) {
                        if (!dispatching && !begin()) {
                            cancelled = true;
                            break;
                        }
                        dispatch({ -1, src + o, s->off[k] + o, std::min(s->chunk, r.second - o) });
                    }
                }
                return;
            }
            size_t ri = 0;
            size_t ro = ranges.empty() ? 0 : ranges[0].first;
            while (ri < ranges.size() && !cancelled) {
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
                while (pos < s->chunk && ri < ranges.size()) {
                    const size_t n = std::min(s->chunk - pos, ranges[ri].second - ro);
                    segs.push_back({ dst + pos, src + ro, n });
                    pieces.push_back({ slot, dst + pos, s->off[k] + ro, n });
                    pos += n;
                    ro  += n;
                    if (ro == ranges[ri].second && ++ri < ranges.size()) {
                        ro = ranges[ri].first;
                    }
                }
                s->pool.copy_list(segs.data(), (int) segs.size());
                s->chunk_state[slot] = 1;
                for (const item & it : pieces) {
                    if (dispatching) {
                        dispatch(it);
                    } else {
                        staged.push_back(it);
                    }
                }
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
                if (!alive()) {
                    cancelled = true;
                }
            }
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
        for (int k = 0; k < 3 && !cancelled; ++k) {
            if (d2d) {
                expert_ranges(k, false);
            } else {
                ranges.assign(1, { (size_t) 0, L.nbytes[k] });
            }
            stream_ranges(k);
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
        if (!cancelled) {
            for (int k = 0; k < 3; ++k) {
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

        {
            std::lock_guard<std::mutex> lk(s->mtx);
            if (!cancelled) {
                B.complete = true;
                s->ctr.jobs++;
                s->ctr.bytes += bytes;
                s->ub_bytes  += bytes;
                s->ctr.d2d_bytes += bytes_d2d; // [TAG_FN_L14_PFSD2D]
                s->ub_d2d        += bytes_d2d;
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

void pfs_gate_op(ggml_tensor * dst, int ith, int nth, void * ud) {
    GGML_UNUSED(dst);
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }
    pfs_state * s = g_pfs;
    pfs_layer * L = (pfs_layer *) ud;
    if (!s || L->pos >= (int) s->layers.size() || &s->layers[L->pos] != L) {
        GGML_ABORT("prefill-stream: a graph outlived its stream state");
    }
    if (s->lend && !s->lent) {
        // [TAG_FN_R1_PFS_LEND] the banks are the hot set's slots right now: a decode could be reading them
        GGML_ABORT("prefill-stream: a streamed graph ran while its banks were not borrowed");
    }
    const int64_t t0 = ggml_time_us();
    pfs_bank & B = s->banks[L->bank];
    bool had = false;
    {
        std::unique_lock<std::mutex> lk(s->mtx);
        if (L->pos == 0) {
            // a new ubatch: the device is idle, so no bank has a user. Older jobs are cancelled, except a copy of
            // layer 0 (the wrap), which is kept running or kept in bank 0.
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
            had = B.holds == 0 && B.complete;
            if (had) {
                s->ctr.reused++;
            }
        }
        s->cv.notify_all();
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
    gen5::ev_wait(s->compute, B.ev_done, s->copy);
    const double w = (ggml_time_us() - t0) / 1e6;
    std::lock_guard<std::mutex> lk(s->mtx);
    s->ctr.wait_s += w;
    s->ub_wait    += w;
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
        if (s->stats && L->pos == (int) s->layers.size() - 1) {
            const double dt = (ggml_time_us() - s->t_ub0) / 1e6;
            LLAMA_LOG_INFO("prefill-stream: ubatch %" PRIu64 ": %zu layers in %.1f ms, %.2f GiB copied (%.1f GiB/s), gates waited %.1f ms, "
                    "%.2f GiB from the hot set's slots [TAG_FN_L14_PFSD2D]\n",
                    s->ctr.ubatches, s->layers.size(), dt*1e3, s->ub_bytes/1073741824.0,
                    dt > 0 ? s->ub_bytes/1073741824.0/dt : 0.0, s->ub_wait*1e3, s->ub_d2d/1073741824.0);
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
    s->pool.stop();
    if (s->copy) {
        ggml_backend_synchronize(s->copy);
    }
    if (s->lent) {
        llama_moe_hot_unlend(s->owner); // [TAG_FN_R1_PFS_LEND] the hot set gets its range back (no copy runs any more)
        s->lent = false;
    }
    for (auto & b : s->banks) {
        if (b.ev_done) { ggml_backend_event_free(b.ev_done); }
        if (b.ev_free) { ggml_backend_event_free(b.ev_free); }
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
        }
        s->ctr.lends++;
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
        }
    }
    if (s->copy) {
        ggml_backend_synchronize(s->copy); // what it dispatched has landed
    }
    s->lent = false;
    llama_moe_hot_unlend(s->owner);
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
    const size_t ring_bytes = (size_t) gen5::env_int("LLAMA_PREFILL_STREAM_RING_MIB", 512, 2, 4096) << 20;
    const int    n_threads  = gen5::env_int("LLAMA_PREFILL_STREAM_THREADS", 8, 1, 64);
    if (s->min_tokens > 32 && !s->lend) { // [TAG_FN_R1_PFS_LEND] a lending stream reserves with the op offload
        LLAMA_LOG_WARN("prefill-stream: LLAMA_PREFILL_STREAM_MIN=%" PRId64 " > 32: ubatches of 32..%" PRId64 " tokens keep the "
                "op-offload copies, which the compute buffer was not reserved for\n", s->min_tokens, s->min_tokens - 1);
    }

    const bool cpu_dev = !d.dev || ggml_backend_dev_type(d.dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
    ggml_backend_buffer_type_t host_buft = !cpu_dev ? ggml_backend_dev_host_buffer_type(d.dev) : nullptr;

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
        ggml_init_params ip = { ggml_tensor_overhead()*(3*layers.size() + 8), nullptr, true };
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
                if (s->lend) {
                    continue; // [TAG_FN_R1_PFS_LEND] placed at the first lend
                }
                void * addr = (char *) ggml_backend_buffer_get_base(B.buf) + s->off[k];
                if (ggml_backend_tensor_alloc(B.buf, L.dev[k], addr) != GGML_STATUS_SUCCESS) {
                    return fail("alias allocation failed");
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
    if (!s->zeros.alloc(d.dev, 64u << 10, "prefill stream zeros")) {
        return fail("no zero page");
    }
    memset(s->zeros.ptr, 0, s->zeros.size);

    if (!cpu_dev) {
        s->copy = ggml_backend_dev_init(d.dev, nullptr);
        s->own_copy = s->copy != nullptr;
        // [TAG_FN_L3_HOST_POKE] the device backend's flush, if it has one (it does nothing unless the switch is on)
        if (ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(d.dev)) {
            s->poke = (void (*)(ggml_backend_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_stream_poke");
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

    s->pool.start(n_threads);
    s->thr = std::thread(pfs_run, s);
    g_pfs = s;

    LLAMA_LOG_INFO("prefill-stream: %zu host expert layers, ubatches >= %" PRId64 " tokens, %d VRAM bank(s) x %.0f MiB%s, "
            "%s ring %zu x %zu MiB, %d copy threads, wrap %s%s\n", s->layers.size(), s->min_tokens, s->n_bufs,
            s->bank_size/1048576.0, s->lend ? " borrowed from the hot set while a prompt streams [TAG_FN_R1_PFS_LEND]" : "",
            s->ring.pinned ? "pinned" : (need_ring ? "plain" : "no"), (size_t) s->n_chunks,
            s->chunk >> 20, s->pool.size(), s->wrap ? "on" : "off",
            s->d2d ? ", the hot set's experts slot -> bank on the device [TAG_FN_L14_PFSD2D]" : "");
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
    const bool want = n_tokens >= s->min_tokens;
    if (want == s->lent) {
        return false;
    }
    if (sched) {
        ggml_backend_sched_synchronize(sched); // the owner's last graph is done: nothing reads the slots or the banks
    }
    if (want) {
        return pfs_take(s);
    }
    pfs_return(s);
    return true;
}
