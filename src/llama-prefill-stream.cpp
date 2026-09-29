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

#include "llama-impl.h"

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

    ggml_backend_event_t ev_done = nullptr; // copy stream, after the copies of holds
    ggml_backend_event_t ev_free = nullptr; // compute stream, at the last release
};

struct pfs_job {
    int      pos = -1;
    uint64_t gen = 0;
};

struct pfs_state {
    const void *       owner    = nullptr;
    ggml_backend_dev_t dev      = nullptr;
    ggml_backend_t     compute  = nullptr;
    ggml_backend_t     copy     = nullptr; // own stream
    bool               own_copy = false;

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
    double             ub_wait  = 0.0;
};

pfs_state * g_pfs = nullptr;
std::mutex  g_pfs_init;

size_t align_up(size_t x, size_t a) {
    return (x + a - 1) / a * a;
}

// the streamer thread
void pfs_run(pfs_state * s) {
    struct item { int slot; const uint8_t * src; size_t dst; size_t n; };
    std::vector<item> staged;
    int next_slot = 0;

    for (;;) {
        pfs_job job;
        {
            std::unique_lock<std::mutex> lk(s->mtx);
            s->cv.wait(lk, [&]() { return s->stop || !s->queue.empty(); });
            if (s->stop) {
                return;
            }
            job = s->queue.front();
            const pfs_bank & b0 = s->banks[s->layers[job.pos].bank];
            if (b0.holds == job.pos && b0.complete) {
                s->queue.pop_front(); // the bank has this layer already (weights never change)
                s->cv_done.notify_all();
                continue;
            }
            s->cur_gen = job.gen;
            s->cur_pos = job.pos;
        }
        const pfs_layer & L = s->layers[job.pos];
        pfs_bank &        B = s->banks[L.bank];
        bool dispatching = false;
        bool cancelled   = false;
        size_t bytes = 0;

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

        for (int k = 0; k < 3 && !cancelled; ++k) {
            const uint8_t * src = (const uint8_t *) L.src[k]->data;
            for (size_t o = 0; o < L.nbytes[k] && !cancelled; o += s->chunk) {
                const size_t n = std::min(s->chunk, L.nbytes[k] - o);
                if (L.src_pinned) {
                    if (!dispatching && !begin()) {
                        cancelled = true;
                        break;
                    }
                    dispatch({ -1, src + o, s->off[k] + o, n });
                    continue;
                }
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
                s->pool.copy(dst, src + o, n);
                s->chunk_state[slot] = 1;
                const item it = { slot, dst, s->off[k] + o, n };
                if (dispatching) {
                    dispatch(it);
                } else {
                    staged.push_back(it);
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
        }
        if (!cancelled && !flush()) {
            cancelled = true;
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
                q.push_back({ 0, s->gen });
            }
            s->queue.swap(q);
            s->ctr.ubatches++;
            s->t_ub0    = t0;
            s->ub_bytes = 0;
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
    }
    gen5::ev_wait(s->compute, B.ev_done, s->copy);
    const double w = (ggml_time_us() - t0) / 1e6;
    std::lock_guard<std::mutex> lk(s->mtx);
    s->ctr.wait_s += w;
    s->ub_wait    += w;
}

void pfs_release_op(ggml_tensor * dst, int ith, int nth, void * ud) {
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
    pfs_bank & B = s->banks[L->bank];
    gen5::ev_record(B.ev_free, s->compute);
    {
        std::lock_guard<std::mutex> lk(s->mtx);
        if (B.holds == L->pos) {
            B.consumed = true;
        }
        if (s->stats && L->pos == (int) s->layers.size() - 1) {
            const double dt = (ggml_time_us() - s->t_ub0) / 1e6;
            LLAMA_LOG_INFO("prefill-stream: ubatch %" PRIu64 ": %zu layers in %.1f ms, %.2f GiB copied (%.1f GiB/s), gates waited %.1f ms\n",
                    s->ctr.ubatches, s->layers.size(), dt*1e3, s->ub_bytes/1073741824.0,
                    dt > 0 ? s->ub_bytes/1073741824.0/dt : 0.0, s->ub_wait*1e3);
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
    const size_t ring_bytes = (size_t) gen5::env_int("LLAMA_PREFILL_STREAM_RING_MIB", 512, 2, 4096) << 20;
    const int    n_threads  = gen5::env_int("LLAMA_PREFILL_STREAM_THREADS", 8, 1, 64);
    if (s->min_tokens > 32) {
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
    } else {
        s->copy = d.compute; // CPU: copies are synchronous memcpy
    }
    if (!s->copy) {
        return fail("no copy backend");
    }

    s->pool.start(n_threads);
    s->thr = std::thread(pfs_run, s);
    g_pfs = s;

    LLAMA_LOG_INFO("prefill-stream: %zu host expert layers, ubatches >= %" PRId64 " tokens, %d VRAM bank(s) x %.0f MiB, "
            "%s ring %zu x %zu MiB, %d copy threads, wrap %s\n", s->layers.size(), s->min_tokens, s->n_bufs,
            s->bank_size/1048576.0, s->ring.pinned ? "pinned" : (need_ring ? "plain" : "no"), (size_t) s->n_chunks,
            s->chunk >> 20, s->pool.size(), s->wrap ? "on" : "off");
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
