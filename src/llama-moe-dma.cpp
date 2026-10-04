// [TAG_MOE_DMA_SHARE] [TAG_MOE_PREFETCH] a share of the cold (not hot) experts of each decode step goes over PCIe from
// a bounded pinned ring to a small VRAM bank and is computed on the GPU, while the CPU computes the rest. See
// llama-moe-gen5.h for the switches.
//
// Per host MoE layer L (position p among the DMA layers, bank = (type group, p % 2)):
//   plan(L)   CPU node, first of the CPU split (the host has synchronized the device): picks D = routed cold experts
//             that were prefetched into the bank or sit ready in the ring; writes the CPU skip table (hot + D) and the
//             bank table; the issuer thread copies ring -> bank and the table, then records the bank event.
//   CPU       MUL_MAT_ID up/gate/down skip every expert whose table entry is not 0.
//   fence(L)  CPU node after the CPU down projection: waits for the issuer, makes the compute stream wait for the
//             bank event (share = auto: waits on the host and moves the share), then releases the prefetch of L+1
//             (issuer) and the urgent ring fills (filler), so both run while the GPU works and the CPU is idle.
//   GPU       the bank chain (MUL_MAT_ID over the bank, zero slot for every id that is not in D) is added to the CPU
//             and hot results before the weights are applied.
// The ring changes only at step boundaries (admission) and at fences (urgent fills, into slots no copy of this step
// reads). All copies of a step are finished at the step boundary (copy stream synchronized).
//
// [TAG_FN_R4_BRIDGE_DMA] Bridge mode (LLAMA_MOE_BRIDGE=1 with LLAMA_MOE_BRIDGE_DMA=1 and a share): the same ring, banks,
// admission and warm start, but no plan / fence nodes, no issuer thread and no copy stream. The bridge's host executor
// plans each bridged job (llama_moe_dma_bridge_plan: D = routed cold experts that sit ready in the ring, most tokens
// first, up to share x the job's cold experts and the bank), the device fetch (GGML_OP_MOE_HOST_FETCH) copies D from
// the ring through its device mapping into the bank with SM loads, and the CPU pool skips D. The ring is read only by
// fetches of the running graph, which the owner synchronizes before the step boundary, so admission stays at the step
// boundary as above. share = auto moves the share by measured rates (llama_moe_dma_bridge_feedback): the CPU's time per
// expert against the fetch's, and whether the device waited for the host's part (the CPU is the long pole: more to the
// GPU) or not (the GPU is: less).

#include "llama-moe-gen5.h"
#include "llama-moe-gen5-impl.h"

#include "llama-impl.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace {

enum : int32_t { RING_FREE = 0, RING_FILLING = 1, RING_READY = 2 };

struct ring_slot {
    std::atomic<int32_t> state{RING_FREE};
    int32_t  pos      = -1;
    int32_t  expert   = -1;
    uint64_t last_use = 0; // step of the last copy that read it (a slot is refilled only after that step)
};

struct dma_bank {
    int     n_slots = 0;
    int64_t n_expert = 0;
    size_t  bytes[3] = { 0, 0, 0 }; // per expert: up, gate, down

    ggml_tensor * up    = nullptr; // [ne0, ne1, n_slots + 1]
    ggml_tensor * gate  = nullptr;
    ggml_tensor * down  = nullptr;
    ggml_tensor * table = nullptr; // I32 [1, n_expert]

    std::vector<int32_t> mirror;      // the device table as last written
    std::vector<int32_t> slot_expert; // expert whose bytes were last copied into each slot
    std::vector<uint8_t> staged;      // slot was prefetched for staged_for
    int                  staged_for = -1;
    bool                 dirty      = false; // copies into the bank not yet waited for by a fence

    gen5::host_mem       tbl_host;    // pinned source of the table copy
    ggml_backend_event_t ev = nullptr;
};

struct dma_layer {
    int il   = -1;
    int pos  = 0;
    int bank = 0;

    ggml_tensor * src[3] = { nullptr, nullptr, nullptr };
    size_t        bytes[3] = { 0, 0, 0 };
    size_t        exp_bytes = 0;
    int64_t       n_expert  = 0;

    llama_moe_dma_view view;

    const int32_t * hot_tbl  = nullptr; // set at graph build
    int32_t         hot_miss = 0;

    std::vector<int32_t> ring_slot; // expert -> ring slot, -1

    // cold sightings in the last W steps (ring admission)
    std::vector<uint16_t>             win_cnt;
    std::vector<std::vector<int32_t>> win_ring;
    std::vector<int32_t>              cur_ids;
    std::vector<uint8_t>              cur_seen;

    std::vector<double> prof; // routing profile counts (warm start), may be empty

    std::vector<uint8_t> pred_mark;     // predicted by the layer before, for step pred_step - 1
    uint64_t             pred_step = 0;

    uint64_t plan_seq  = 0;
    uint64_t fence_seq = 0;
    uint64_t wait_seq  = 0;     // issuer task the fence waits for, 0 = none
    bool     wait_copy = false; // that task copied experts (share = auto measures it)

    // plan scratch, [n_expert]; tok and s_dslot are reset after each use
    std::vector<uint8_t> s_tok;
    std::vector<int32_t> s_dslot;
    std::vector<int32_t> s_uniq;
    std::vector<int32_t> s_cold;

    bool is_hot(int32_t e) const {
        return hot_tbl && hot_tbl[e] != hot_miss;
    }
};

struct copy_item {
    int32_t ring;
    int32_t slot;
};

struct issue_task {
    int                    bank  = 0;
    bool                   plan  = false; // record the bank event and report done
    bool                   table = false;
    std::vector<copy_item> copies;
    uint64_t               seq   = 0;
};

struct fill_job {
    int32_t slot;
    int32_t pos;
    int32_t expert;
};

struct dma_state {
    const void *       owner    = nullptr;
    ggml_backend_dev_t dev      = nullptr;
    ggml_backend_t     compute  = nullptr;
    ggml_backend_t     copy     = nullptr;
    bool               own_copy = false;

    int    K           = 8;
    double share       = 0.0;
    bool   share_auto  = false;
    bool   bridge      = false; // [TAG_FN_R4_BRIDGE_DMA] bridge mode: the bridge's executor plans, the device fetches

    // [TAG_FN_R4_BRIDGE_DMA] the share the executor reads (the owner's thread moves it between graphs)
    std::atomic<double> br_share{0.0};
    double br_cpu_us   = 0.0;   // EMA: CPU pool time per expert of a job (the whole pool)
    double br_fetch_us = 0.0;   // EMA: fetch time per fetched expert
    double br_wait_hi  = 20.0;  // the device waited more than this per layer (us): the CPU is the long pole
    double br_wait_lo  = 3.0;   // ... less than this: the GPU is
    double br_step     = 0.02;  // share change per graph
    double br_max      = 1.0;   // share cap
    uint64_t br_graphs = 0;
    double br_wait_sum = 0.0;   // per-layer device wait, summed over the graphs (stats)
    bool   prefetch    = false;
    int    pf_slots    = 4;
    int    pf_fill     = 2;
    int    pf_k        = 12;
    bool   stats       = false;
    bool   sync        = false;
    bool   inline_issue = false;
    int    ad_min      = 2;
    int    ad_win      = 32;
    int    ad_hyst     = 1;
    size_t fill_step   = 32u << 20;

    std::vector<dma_layer>             layers;
    std::map<const ggml_tensor *, int> by_up;
    std::map<int, int>                 by_il;
    std::vector<dma_bank>              banks;
    std::vector<ggml_context *>        ctxs;
    std::vector<ggml_backend_buffer_t> bufs;

    gen5::host_mem               ring;
    size_t                       slot_size = 0;
    int                          n_ring    = 0;
    std::unique_ptr<ring_slot[]> rs;
    std::mutex                   ring_mtx;

    std::thread             issuer;
    std::mutex              imtx;
    std::condition_variable icv;
    std::condition_variable icv_done;
    std::deque<issue_task>  itasks;
    bool                    ibusy    = false;
    uint64_t                done_seq = 0;
    uint64_t                next_seq = 0;

    std::vector<std::thread> fillers;
    std::mutex               fmtx;
    std::condition_variable  fcv;
    std::condition_variable  fcv_idle;
    std::deque<fill_job>     fq_urgent;
    std::deque<fill_job>     fq;
    int                      f_running = 0;

    bool stop = false;

    // decided by plan(L), applied and released by fence(L)
    bool                                     pf_ready = false;
    int                                      pf_pos   = -1;
    std::vector<std::pair<int32_t, int32_t>> pf_list;      // (ring slot, expert) for bank slots 0..
    std::vector<fill_job>                    pf_fills_req; // (pos, expert) wanted; slots chosen at the fence

    uint64_t step      = 0;
    bool     warm_done = false;

    llama_moe_dma_counters ctr;
    double   fence_wait_s = 0.0;
};

dma_state * g_dma = nullptr;
std::mutex  g_dma_init;

void dma_copy_expert(dma_state * s, const dma_bank & B, const uint8_t * src, int32_t slot) {
    ggml_backend_tensor_set_async(s->copy, B.up,   src,                           (size_t) slot*B.up->nb[2],   B.bytes[0]);
    ggml_backend_tensor_set_async(s->copy, B.gate, src + B.bytes[0],              (size_t) slot*B.gate->nb[2], B.bytes[1]);
    ggml_backend_tensor_set_async(s->copy, B.down, src + B.bytes[0] + B.bytes[1], (size_t) slot*B.down->nb[2], B.bytes[2]);
}

void dma_issue(dma_state * s, const issue_task & t) {
    dma_bank & B = s->banks[t.bank];
    for (const copy_item & c : t.copies) {
        dma_copy_expert(s, B, s->ring.ptr + (size_t) c.ring*s->slot_size, c.slot);
    }
    if (t.table) {
        ggml_backend_tensor_set_async(s->copy, B.table, B.tbl_host.ptr, 0, (size_t) B.n_expert*sizeof(int32_t));
    }
    if (t.plan) {
        if (B.ev) {
            gen5::ev_record(B.ev, s->copy);
        } else {
            ggml_backend_synchronize(s->copy);
        }
    }
}

void dma_issuer_run(dma_state * s) {
    for (;;) {
        issue_task t;
        {
            std::unique_lock<std::mutex> lk(s->imtx);
            s->icv.wait(lk, [&]() { return s->stop || !s->itasks.empty(); });
            if (s->stop) {
                return;
            }
            t = std::move(s->itasks.front());
            s->itasks.pop_front();
            s->ibusy = true;
        }
        dma_issue(s, t);
        {
            std::lock_guard<std::mutex> lk(s->imtx);
            s->ibusy = false;
            if (t.plan) {
                s->done_seq = t.seq;
            }
        }
        s->icv_done.notify_all();
    }
}

void dma_submit(dma_state * s, issue_task && t) {
    if (s->inline_issue) {
        dma_issue(s, t);
        std::lock_guard<std::mutex> lk(s->imtx);
        if (t.plan) {
            s->done_seq = t.seq;
        }
        return;
    }
    {
        std::lock_guard<std::mutex> lk(s->imtx);
        s->itasks.push_back(std::move(t));
    }
    s->icv.notify_one();
}

void dma_wait_seq(dma_state * s, uint64_t seq) {
    std::unique_lock<std::mutex> lk(s->imtx);
    s->icv_done.wait(lk, [&]() { return s->stop || s->done_seq >= seq; });
}

void dma_wait_issuer_idle(dma_state * s) {
    if (s->bridge) {
        return; // [TAG_FN_R4_BRIDGE_DMA] no issuer
    }
    std::unique_lock<std::mutex> lk(s->imtx);
    s->icv_done.wait(lk, [&]() { return s->stop || (s->itasks.empty() && !s->ibusy); });
}

void dma_filler_run(dma_state * s) {
    for (;;) {
        fill_job j;
        {
            std::unique_lock<std::mutex> lk(s->fmtx);
            s->fcv.wait(lk, [&]() { return s->stop || !s->fq_urgent.empty() || !s->fq.empty(); });
            if (s->stop) {
                return;
            }
            if (!s->fq_urgent.empty()) {
                j = s->fq_urgent.front();
                s->fq_urgent.pop_front();
            } else {
                j = s->fq.front();
                s->fq.pop_front();
            }
            s->f_running++;
        }
        const dma_layer & L = s->layers[j.pos];
        uint8_t * dst = s->ring.ptr + (size_t) j.slot*s->slot_size;
        size_t o = 0;
        for (int k = 0; k < 3; ++k) {
            memcpy(dst + o, (const uint8_t *) L.src[k]->data + (size_t) j.expert*L.bytes[k], L.bytes[k]);
            o += L.bytes[k];
        }
        s->rs[j.slot].state.store(RING_READY, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(s->fmtx);
            s->f_running--;
            s->ctr.fills++;
        }
        s->fcv_idle.notify_all();
    }
}

// under ring_mtx: a slot that no copy of this step reads, preferring free slots, then the lowest windowed count;
// -1 when none. The victim leaves the index.
int32_t ring_victim(dma_state * s, int min_keep_cnt) {
    int32_t best = -1;
    int     best_cnt = 1 << 30;
    for (int32_t i = 0; i < s->n_ring; ++i) {
        ring_slot & r = s->rs[i];
        const int32_t st = r.state.load(std::memory_order_acquire);
        if (st == RING_FILLING || (st == RING_READY && r.last_use >= s->step)) {
            continue;
        }
        if (st == RING_FREE) {
            return i;
        }
        const int c = s->layers[r.pos].win_cnt.empty() ? 0 : s->layers[r.pos].win_cnt[r.expert];
        if (c < best_cnt) {
            best_cnt = c;
            best     = i;
        }
    }
    if (best >= 0 && best_cnt > min_keep_cnt) {
        return -1;
    }
    return best;
}

// under ring_mtx: bind slot to (pos, expert) as FILLING and queue the copy
void ring_assign(dma_state * s, int32_t slot, int32_t pos, int32_t expert, bool urgent) {
    ring_slot & r = s->rs[slot];
    if (r.pos >= 0 && r.expert >= 0 && s->layers[r.pos].ring_slot[r.expert] == slot) {
        s->layers[r.pos].ring_slot[r.expert] = -1;
    }
    r.pos    = pos;
    r.expert = expert;
    r.state.store(RING_FILLING, std::memory_order_release);
    s->layers[pos].ring_slot[expert] = slot;
    std::lock_guard<std::mutex> lk(s->fmtx);
    (urgent ? s->fq_urgent : s->fq).push_back({ slot, pos, expert });
    if (urgent) {
        s->ctr.fill_urgent++;
    }
}

void dma_plan_op(ggml_tensor * dst, int ith, int nth, void * ud) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }
    dma_state * s = g_dma;
    dma_layer * L = (dma_layer *) ud;
    if (!s || L->pos >= (int) s->layers.size() || &s->layers[L->pos] != L) {
        GGML_ABORT("moe-dma: a graph outlived its DMA state");
    }
    int32_t * cpu_tbl = (int32_t *) dst->data;
    const ggml_tensor * ids  = dst->src[0];
    const ggml_tensor * pred = dst->src[1];
    const int64_t n_exp  = L->n_expert;
    const int64_t n_used = ids->ne[0];
    const int64_t T      = ids->ne[1];
    dma_bank & B = s->banks[L->bank];
    GGML_ASSERT(dst->ne[1] == n_exp);

    // unique routed experts and tokens per expert
    auto & tok  = L->s_tok;
    auto & uniq = L->s_uniq;
    auto & cold = L->s_cold;
    uniq.clear();
    cold.clear();
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t i = 0; i < n_used; ++i) {
            const int32_t e = *(const int32_t *) ((const char *) ids->data + t*ids->nb[1] + i*ids->nb[0]);
            if (e < 0 || e >= n_exp) {
                continue;
            }
            if (tok[e]++ == 0) {
                uniq.push_back(e);
            }
        }
    }
    uint64_t n_pred = 0;
    for (int32_t e : uniq) {
        if (L->is_hot(e)) {
            continue;
        }
        cold.push_back(e);
        if (!L->cur_seen[e]) {
            L->cur_seen[e] = 1;
            L->cur_ids.push_back(e);
        }
        n_pred += L->pred_step == s->step + 1 && L->pred_mark[e];
    }

    // D: experts prefetched for this layer first, then ring-ready cold experts (most tokens first) up to share x cold
    auto & d_slot = L->s_dslot;
    std::vector<uint8_t> used(B.n_slots, 0);
    issue_task task;
    task.bank = L->bank;
    task.plan = true;
    int n_staged = 0;
    if (B.staged_for == L->pos) {
        for (int j = 0; j < B.n_slots; ++j) {
            const int32_t e = B.slot_expert[j];
            if (B.staged[j] && e >= 0 && e < n_exp && tok[e] && !L->is_hot(e) && d_slot[e] < 0) {
                d_slot[e] = j;
                used[j]   = 1;
                n_staged++;
            }
        }
    }
    B.staged_for = -1;
    std::fill(B.staged.begin(), B.staged.end(), 0);

    int n_od = 0;
    uint64_t n_ready = 0;
    {
        std::vector<std::pair<int32_t, int32_t>> cand; // (expert, ring slot)
        std::lock_guard<std::mutex> lk(s->ring_mtx);
        for (int32_t e : cold) {
            const int32_t r = L->ring_slot[e];
            if (r < 0) {
                continue;
            }
            int32_t st = s->rs[r].state.load(std::memory_order_acquire);
            while (st == RING_FILLING && s->sync) { // deterministic mode: a fill that was started counts
                std::this_thread::yield();
                st = s->rs[r].state.load(std::memory_order_acquire);
            }
            if (st != RING_READY) {
                continue;
            }
            n_ready++;
            if (d_slot[e] < 0) {
                cand.push_back({ e, r });
            }
        }
        std::stable_sort(cand.begin(), cand.end(), [&](const std::pair<int32_t, int32_t> & a, const std::pair<int32_t, int32_t> & b) {
            return tok[a.first] != tok[b.first] ? tok[a.first] > tok[b.first] : a.first < b.first;
        });
        const int cap = std::max(0, (int) std::lround(s->share*(double) cold.size()) - n_staged);
        int j = 0;
        for (const auto & c : cand) {
            if (n_od >= cap) {
                break;
            }
            while (j < B.n_slots && used[j]) {
                j++;
            }
            if (j >= B.n_slots) {
                break;
            }
            used[j]          = 1;
            d_slot[c.first]  = j;
            B.slot_expert[j] = c.first;
            s->rs[c.second].last_use = s->step;
            task.copies.push_back({ c.second, j });
            n_od++;
        }
    }

    // CPU skip table (hot or in D), bank table (D -> slot, else the zero slot)
    bool changed = false;
    for (int64_t e = 0; e < n_exp; ++e) {
        const int32_t v = d_slot[e] >= 0 ? d_slot[e] : B.n_slots;
        cpu_tbl[e] = (d_slot[e] >= 0 || L->is_hot((int32_t) e)) ? 1 : 0;
        if (B.mirror[e] != v) {
            B.mirror[e] = v;
            changed = true;
        }
    }
    for (int32_t e : uniq) {
        tok[e]    = 0;
        d_slot[e] = -1;
    }
    if (changed) {
        memcpy(B.tbl_host.ptr, B.mirror.data(), (size_t) n_exp*sizeof(int32_t));
    }
    task.table = changed;
    L->plan_seq++;
    L->wait_seq  = 0;
    L->wait_copy = false;
    if (changed || !task.copies.empty() || B.dirty) {
        B.dirty      = false;
        task.seq     = ++s->next_seq;
        L->wait_seq  = task.seq;
        L->wait_copy = !task.copies.empty();
    }

    // prefetch for the next layer: decided here, applied and started by the fence
    s->pf_ready = false;
    s->pf_fills_req.clear();
    s->pf_list.clear();
    const auto nit = s->by_il.find(L->il + 1);
    if (s->prefetch && pred && nit != s->by_il.end()) {
        dma_layer & N  = s->layers[nit->second];
        dma_bank &  NB = s->banks[N.bank];
        const int64_t pk = std::min<int64_t>(s->pf_k, pred->ne[0]);
        std::fill(N.pred_mark.begin(), N.pred_mark.end(), 0);
        N.pred_step = s->step + 1;
        const int max_pf = std::min(s->pf_slots, NB.n_slots);
        std::lock_guard<std::mutex> lk(s->ring_mtx);
        for (int64_t r = 0; r < pk; ++r) {
            for (int64_t t = 0; t < pred->ne[1]; ++t) {
                const int32_t e = *(const int32_t *) ((const char *) pred->data + t*pred->nb[1] + r*pred->nb[0]);
                if (e < 0 || e >= N.n_expert || N.pred_mark[e] || N.is_hot(e)) {
                    continue;
                }
                N.pred_mark[e] = 1;
                const int32_t rsl = N.ring_slot[e];
                if (rsl >= 0 && s->rs[rsl].state.load(std::memory_order_acquire) == RING_READY) {
                    if ((int) s->pf_list.size() < max_pf) {
                        s->rs[rsl].last_use = s->step;
                        s->pf_list.push_back({ rsl, e });
                    }
                } else if (rsl < 0 && (int) s->pf_fills_req.size() < s->pf_fill) {
                    s->pf_fills_req.push_back({ -1, N.pos, e });
                }
            }
        }
        s->pf_pos   = N.pos;
        s->pf_ready = !s->pf_list.empty();
    }

    s->ctr.layer_steps++;
    s->ctr.cold          += cold.size();
    s->ctr.dma_staged    += n_staged;
    s->ctr.dma_on_demand += n_od;
    s->ctr.ring_ready    += n_ready;
    s->ctr.pred_cold     += n_pred;

    if (L->wait_seq) {
        dma_submit(s, std::move(task));
    }
}

void dma_fence_op(ggml_tensor * dst, int ith, int nth, void * ud) {
    GGML_UNUSED(dst);
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }
    dma_state * s = g_dma;
    dma_layer * L = (dma_layer *) ud;
    if (!s || L->pos >= (int) s->layers.size() || &s->layers[L->pos] != L) {
        GGML_ABORT("moe-dma: a graph outlived its DMA state");
    }
    if (L->plan_seq == L->fence_seq) {
        return; // no plan ran for this layer since the last fence
    }
    L->fence_seq = L->plan_seq;
    dma_bank & B = s->banks[L->bank];

    if (L->wait_seq) {
        dma_wait_seq(s, L->wait_seq);
        if (s->share_auto && L->wait_copy) {
            // the copies of this layer ran beside the CPU part: if they are still running, the share was too high
            const int64_t t0 = ggml_time_us();
            gen5::ev_sync(B.ev, s->copy);
            const double dt = (ggml_time_us() - t0) / 1e6;
            s->fence_wait_s += dt;
            s->share = dt > 30e-6 ? std::max(0.05, s->share*0.95) : std::min(1.0, s->share + 0.01);
        }
        gen5::ev_wait(s->compute, B.ev, s->copy);
    }

    // the CPU part is done: start the prefetch copies and the urgent fills now, while the GPU runs
    if (s->pf_ready) {
        dma_layer & N  = s->layers[s->pf_pos];
        dma_bank &  NB = s->banks[N.bank];
        issue_task pt;
        pt.bank = N.bank;
        std::fill(NB.staged.begin(), NB.staged.end(), 0);
        for (size_t j = 0; j < s->pf_list.size(); ++j) {
            NB.slot_expert[j] = s->pf_list[j].second;
            NB.staged[j]      = 1;
            pt.copies.push_back({ s->pf_list[j].first, (int32_t) j });
        }
        NB.staged_for = N.pos;
        NB.dirty      = true;
        s->ctr.prefetched += s->pf_list.size();
        dma_submit(s, std::move(pt));
        s->pf_ready = false;
        s->pf_list.clear();
    }
    if (!s->pf_fills_req.empty()) {
        std::lock_guard<std::mutex> lk(s->ring_mtx);
        for (const fill_job & f : s->pf_fills_req) {
            if (s->layers[f.pos].ring_slot[f.expert] >= 0) {
                continue;
            }
            const int32_t v = ring_victim(s, 1 << 29);
            if (v < 0) {
                break;
            }
            ring_assign(s, v, f.pos, f.expert, true);
        }
        s->pf_fills_req.clear();
        s->fcv.notify_all();
    }
}

bool dma_read_profile(const char * path, std::map<int, std::vector<double>> & out) {
    std::ifstream f(path);
    if (!f) {
        return false;
    }
    std::string line;
    std::getline(f, line);
    if (line.rfind("moeprof v1", 0) != 0) {
        return false;
    }
    const char * want_env = getenv("LLAMA_MOE_HOT_SECTION");
    const std::string want = want_env ? want_env : "decode_union";
    std::map<std::string, std::map<int, std::vector<double>>> sec;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream ss(line);
        std::string name;
        int il = -1;
        if (!(ss >> name >> il) || il < 0) {
            continue;
        }
        std::vector<double> v;
        double x;
        while (ss >> x) {
            v.push_back(x);
        }
        sec[name][il] = std::move(v);
    }
    for (const std::string & sname : { want, std::string("decode_union"), std::string("decode_tokens"), std::string("prefill_tokens") }) {
        auto it = sec.find(sname);
        if (it != sec.end() && !it->second.empty()) {
            out = it->second;
            return true;
        }
    }
    return false;
}

// step boundary of the owner: every copy of the step is done; roll the windows, admit into the ring, warm start
// [TAG_FN_R4_BRIDGE_DMA] in bridge mode the fetches ran in the owner's bridged graphs, which it synchronized
void dma_step_impl(dma_state * s) {
    dma_wait_issuer_idle(s);
    if (s->copy) {
        ggml_backend_synchronize(s->copy);
    }
    s->step++;
    s->ctr.steps++;

    bool any = false;
    for (const auto & L : s->layers) {
        any = any || !L.cur_ids.empty();
    }
    const int pos_w = (int) (s->step % (uint64_t) s->ad_win);
    struct cand { int pos; int32_t e; uint16_t c; };
    std::vector<cand> cands;
    if (any) {
        for (auto & L : s->layers) {
            auto & slot = L.win_ring[pos_w];
            for (int32_t e : slot) {
                if (L.win_cnt[e] > 0) {
                    L.win_cnt[e]--;
                }
            }
            slot.swap(L.cur_ids);
            L.cur_ids.clear();
            for (int32_t e : slot) {
                L.win_cnt[e]++;
                L.cur_seen[e] = 0;
                if (L.win_cnt[e] >= s->ad_min && L.ring_slot[e] < 0 && !L.is_hot(e)) {
                    cands.push_back({ L.pos, e, L.win_cnt[e] });
                }
            }
        }
    }

    std::lock_guard<std::mutex> lk(s->ring_mtx);
    // ring entries of experts that became hot are useless now
    for (int32_t i = 0; i < s->n_ring; ++i) {
        ring_slot & r = s->rs[i];
        if (r.state.load(std::memory_order_acquire) == RING_READY && s->layers[r.pos].is_hot(r.expert)) {
            s->layers[r.pos].ring_slot[r.expert] = -1;
            r.pos    = -1;
            r.expert = -1;
            r.state.store(RING_FREE, std::memory_order_release);
        }
    }
    // warm start: the best experts of the profile that are not hot, once a plan node has seen the hot set
    if (!s->warm_done && s->ctr.layer_steps > 0) {
        s->warm_done = true;
        std::vector<cand> w;
        for (auto & L : s->layers) {
            for (int32_t e = 0; e < (int32_t) L.prof.size() && e < L.n_expert; ++e) {
                if (L.prof[e] > 0 && !L.is_hot(e)) {
                    w.push_back({ L.pos, e, (uint16_t) std::min(65535.0, L.prof[e]) });
                }
            }
        }
        std::vector<double> val(w.size());
        std::vector<size_t> ord(w.size());
        for (size_t i = 0; i < w.size(); ++i) {
            val[i] = s->layers[w[i].pos].prof[w[i].e];
            ord[i] = i;
        }
        std::stable_sort(ord.begin(), ord.end(), [&](size_t a, size_t b) { return val[a] > val[b]; });
        int n = 0;
        for (size_t i : ord) {
            const int32_t v = ring_victim(s, 1 << 29);
            if (v < 0 || s->rs[v].state.load() != RING_FREE) {
                break;
            }
            ring_assign(s, v, w[i].pos, w[i].e, false);
            n++;
        }
        if (n > 0) {
            LLAMA_LOG_INFO("moe-dma: warm start: %d profile experts queued for the ring\n", n);
        }
    }
    // admission
    std::stable_sort(cands.begin(), cands.end(), [](const cand & a, const cand & b) { return a.c > b.c; });
    size_t bytes = 0;
    for (const auto & c : cands) {
        const dma_layer & L = s->layers[c.pos];
        if (bytes + L.exp_bytes > s->fill_step) {
            break;
        }
        const int32_t v = ring_victim(s, (int) c.c - 1 - s->ad_hyst);
        if (v < 0) {
            continue;
        }
        ring_assign(s, v, c.pos, c.e, false);
        bytes += L.exp_bytes;
    }
    s->fcv.notify_all();

    if (s->stats && s->ctr.steps % 256 == 0 && s->bridge) { // [TAG_FN_R4_BRIDGE_DMA]
        const auto & c = s->ctr;
        const double ls = c.layer_steps ? (double) c.layer_steps : 1.0;
        LLAMA_LOG_INFO("moe-dma (bridge): %" PRIu64 " steps: per layer-step %.2f cold, %.2f ring-ready, %.2f fetched; fills %" PRIu64
                "; share %.3f%s; CPU %.1f us/expert, fetch %.1f us/expert, device wait %.1f us/layer\n",
                c.steps, c.cold/ls, c.ring_ready/ls, c.dma_on_demand/ls, c.fills, s->br_share.load(),
                s->share_auto ? " (auto)" : "", s->br_cpu_us, s->br_fetch_us, s->br_graphs ? s->br_wait_sum/s->br_graphs : 0.0);
    } else if (s->stats && s->ctr.steps % 256 == 0) {
        const auto & c = s->ctr;
        const double ls = c.layer_steps ? (double) c.layer_steps : 1.0;
        LLAMA_LOG_INFO("moe-dma: %" PRIu64 " steps: per layer-step %.2f cold, %.2f ring-ready, %.2f to the GPU (%.2f prefetched), "
                "%.2f prefetched, prediction recall %.2f; fills %" PRIu64 " (%" PRIu64 " urgent); share %.2f%s, fence wait %.1f ms\n",
                c.steps, c.cold/ls, c.ring_ready/ls, (c.dma_on_demand + c.dma_staged)/ls, c.dma_staged/ls,
                c.prefetched/ls, c.cold ? (double) c.pred_cold/c.cold : 0.0, c.fills, c.fill_urgent, s->share,
                s->share_auto ? " (auto)" : "", s->fence_wait_s*1e3);
    }
}

void dma_destroy(dma_state * s) {
    {
        std::lock_guard<std::mutex> l1(s->imtx);
        std::lock_guard<std::mutex> l2(s->fmtx);
        s->stop = true;
    }
    s->icv.notify_all();
    s->icv_done.notify_all();
    s->fcv.notify_all();
    s->fcv_idle.notify_all();
    if (s->issuer.joinable()) {
        s->issuer.join();
    }
    for (auto & t : s->fillers) {
        t.join();
    }
    if (s->copy) {
        ggml_backend_synchronize(s->copy);
    }
    for (auto & b : s->banks) {
        if (b.ev) {
            ggml_backend_event_free(b.ev);
        }
        b.tbl_host.release();
    }
    for (auto * b : s->bufs) {
        ggml_backend_buffer_free(b);
    }
    for (auto * c : s->ctxs) {
        ggml_free(c);
    }
    if (s->own_copy && s->copy) {
        ggml_backend_free(s->copy);
    }
    s->ring.release();
    delete s;
}

} // namespace

namespace gen5 {
bool dma_requested() {
    const char * sh = getenv("LLAMA_MOE_DMA_SHARE");
    const bool share_on = sh && sh[0] && (strcmp(sh, "auto") == 0 || atof(sh) > 0.0);
    return share_on || env_flag("LLAMA_MOE_PREFETCH");
}
} // namespace gen5

bool llama_moe_dma_init_layers(const std::vector<llama_moe_gen5_layer_desc> & layers, const llama_moe_gen5_device & d, const void * owner,
        bool bridge) {
    std::lock_guard<std::mutex> init_lock(g_dma_init);
    if (g_dma || layers.empty() || !d.compute || !d.buft || !gen5::dma_requested()) {
        return false;
    }
    auto * s = new dma_state();
    s->owner   = owner;
    s->dev     = d.dev;
    s->compute = d.compute;
    s->bridge  = bridge; // [TAG_FN_R4_BRIDGE_DMA]

    const char * sh = getenv("LLAMA_MOE_DMA_SHARE");
    if (sh && strcmp(sh, "auto") == 0) {
        s->share_auto = true;
        s->share      = 0.5;
    } else if (sh) {
        s->share = std::max(0.0, std::min(1.0, atof(sh)));
    }
    s->prefetch     = gen5::env_flag("LLAMA_MOE_PREFETCH");
    s->K            = gen5::env_int("LLAMA_MOE_DMA_SLOTS", 8, 1, 64);
    if (s->bridge) {
        // [TAG_FN_R4_BRIDGE_DMA] the prefetch needs the fences of the CPU split; a fetch copies at most
        // GGML_MOE_BRIDGE_MAX_FETCH (32) experts; share = auto starts low (Strata: 0-0.3 for Q4 on a 5090)
        if (s->prefetch) {
            LLAMA_LOG_WARN("moe-dma: LLAMA_MOE_PREFETCH is off with the bridge (it needs the CPU split's fences)\n");
        }
        s->prefetch = false;
        s->K = std::min(s->K, 32);
        if (s->share_auto) {
            s->share = gen5::env_int("LLAMA_MOE_DMA_SHARE_START_PCT", 25, 0, 100) / 100.0;
        }
        s->br_wait_hi = gen5::env_int("LLAMA_MOE_DMA_WAIT_HI_US", 20, 0, 100000);
        s->br_wait_lo = gen5::env_int("LLAMA_MOE_DMA_WAIT_LO_US", 3, 0, 100000);
        s->br_step    = gen5::env_int("LLAMA_MOE_DMA_STEP_PCT10", 20, 1, 1000) / 1000.0;
        s->br_max     = gen5::env_int("LLAMA_MOE_DMA_SHARE_MAX_PCT", 100, 0, 100) / 100.0;
        s->br_share.store(s->share);
    }
    s->pf_slots     = gen5::env_int("LLAMA_MOE_PREFETCH_SLOTS", 4, 1, 64);
    s->pf_fill      = gen5::env_int("LLAMA_MOE_PREFETCH_FILL", 2, 0, 64);
    s->pf_k         = gen5::env_int("LLAMA_MOE_PREFETCH_K", 12, 1, 64);
    s->stats        = gen5::env_flag("LLAMA_MOE_DMA_STATS");
    s->sync         = gen5::env_flag("LLAMA_MOE_DMA_SYNC");
    s->inline_issue = gen5::env_flag("LLAMA_MOE_DMA_INLINE");
    s->ad_hyst      = gen5::env_int("LLAMA_MOE_DMA_HYST", 1, 0, 1000);
    s->fill_step    = (size_t) gen5::env_int("LLAMA_MOE_DMA_FILL_MIB", 32, 0, 4096) << 20;
    if (const char * e = getenv("LLAMA_MOE_DMA_ADMIT")) {
        int n = 0, w = 0;
        if (sscanf(e, "%d/%d", &n, &w) == 2 && n >= 1 && w >= n && w <= 256) {
            s->ad_min = n;
            s->ad_win = w;
        }
    }
    const size_t ring_bytes = (size_t) gen5::env_int("LLAMA_MOE_DMA_RING_MIB", 1024, 1, 4096) << 20;
    const int    n_fill     = gen5::env_int("LLAMA_MOE_DMA_FILL_THREADS", 2, 1, 16);

    auto fail = [&](const char * why) {
        LLAMA_LOG_WARN("moe-dma: %s - off\n", why);
        dma_destroy(s);
        return false;
    };

    // banks per (types, shapes) and position parity
    struct gkey {
        ggml_type t[3];
        int64_t   ne[3][2];
        int64_t   n_exp;
        int       parity;
        // field by field: memcmp would also compare the padding bytes, which a copy need not keep
        bool operator<(const gkey & o) const {
            return std::tie(t[0], t[1], t[2], ne[0][0], ne[0][1], ne[1][0], ne[1][1], ne[2][0], ne[2][1], n_exp, parity) <
                   std::tie(o.t[0], o.t[1], o.t[2], o.ne[0][0], o.ne[0][1], o.ne[1][0], o.ne[1][1], o.ne[2][0], o.ne[2][1], o.n_exp, o.parity);
        }
    };
    std::map<gkey, int> bank_of;
    s->layers.resize(layers.size());
    for (size_t p = 0; p < layers.size(); ++p) {
        dma_layer & L = s->layers[p];
        L.il       = layers[p].il;
        L.pos      = (int) p;
        L.src[0]   = layers[p].up;
        L.src[1]   = layers[p].gate;
        L.src[2]   = layers[p].down;
        L.n_expert = L.src[0]->ne[2];
        gkey k;
        memset(&k, 0, sizeof(k));
        for (int i = 0; i < 3; ++i) {
            L.bytes[i]   = L.src[i]->nb[2];
            L.exp_bytes += L.bytes[i];
            k.t[i]       = L.src[i]->type;
            k.ne[i][0]   = L.src[i]->ne[0];
            k.ne[i][1]   = L.src[i]->ne[1];
        }
        k.n_exp  = L.n_expert;
        k.parity = (int) (p % 2);
        auto it = bank_of.find(k);
        if (it == bank_of.end()) {
            it = bank_of.emplace(k, (int) s->banks.size()).first;
            s->banks.emplace_back();
        }
        L.bank = it->second;
        s->by_up[L.src[0]] = (int) p;
        s->by_il[L.il]     = (int) p;
        s->slot_size = std::max(s->slot_size, L.exp_bytes);
    }
    s->slot_size = (s->slot_size + 4095) & ~(size_t) 4095;

    {
        ggml_init_params ip = { ggml_tensor_overhead()*(s->banks.size()*4 + 8), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        if (!ctx) {
            return fail("no context");
        }
        s->ctxs.push_back(ctx);
        std::vector<bool> made(s->banks.size(), false);
        for (const auto & L : s->layers) {
            dma_bank & B = s->banks[L.bank];
            if (made[L.bank]) {
                continue;
            }
            made[L.bank] = true;
            B.n_slots  = s->K;
            B.n_expert = L.n_expert;
            const ggml_tensor * u = L.src[0];
            const ggml_tensor * g = L.src[1];
            const ggml_tensor * dn = L.src[2];
            B.up    = ggml_new_tensor_3d(ctx, u->type,  u->ne[0],  u->ne[1],  s->K + 1);
            B.gate  = ggml_new_tensor_3d(ctx, g->type,  g->ne[0],  g->ne[1],  s->K + 1);
            B.down  = ggml_new_tensor_3d(ctx, dn->type, dn->ne[0], dn->ne[1], s->K + 1);
            B.table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, L.n_expert);
            ggml_format_name(B.up,    "moe_dma_up.%d",   L.bank);
            ggml_format_name(B.gate,  "moe_dma_gate.%d", L.bank);
            ggml_format_name(B.down,  "moe_dma_down.%d", L.bank);
            ggml_format_name(B.table, "moe_dma_tbl.%d",  L.bank);
            for (int i = 0; i < 3; ++i) {
                B.bytes[i] = L.bytes[i];
            }
        }
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, d.buft);
        if (!buf) {
            return fail("bank allocation failed");
        }
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_buffer_clear(buf, 0); // every slot starts zero, the last one of each bank stays zero
        s->bufs.push_back(buf);
    }
    size_t vram = 0;
    for (auto & B : s->banks) {
        B.mirror.assign(B.n_expert, B.n_slots);
        B.slot_expert.assign(B.n_slots, -1);
        B.staged.assign(B.n_slots, 0);
        ggml_backend_tensor_set(B.table, B.mirror.data(), 0, B.mirror.size()*sizeof(int32_t));
        if (!B.tbl_host.alloc(d.dev, B.mirror.size()*sizeof(int32_t), "dma table")) {
            return fail("no pinned table staging");
        }
        B.ev = gen5::ev_new(d.dev);
        vram += ggml_nbytes(B.up) + ggml_nbytes(B.gate) + ggml_nbytes(B.down) + ggml_nbytes(B.table);
    }

    // ring
    s->n_ring = (int) (ring_bytes / s->slot_size);
    if (s->n_ring < 1 || !s->ring.alloc(d.dev, (size_t) s->n_ring*s->slot_size, "dma ring")) {
        return fail("no pinned ring");
    }
    s->rs.reset(new ring_slot[s->n_ring]);

    const bool cpu_dev = !d.dev || ggml_backend_dev_type(d.dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
    if (s->bridge) {
        s->copy = nullptr; // [TAG_FN_R4_BRIDGE_DMA] the device fetch copies: no copy stream, no host CUDA call
    } else if (!cpu_dev) {
        s->copy = ggml_backend_dev_init(d.dev, nullptr);
        s->own_copy = s->copy != nullptr;
    } else {
        s->copy = d.compute;
    }
    if (!s->copy && !s->bridge) {
        return fail("no copy backend");
    }

    // profile for the warm start
    std::map<int, std::vector<double>> prof;
    const char * pp = getenv("LLAMA_MOE_DMA_PROFILE");
    if (!pp || !pp[0]) {
        pp = getenv("LLAMA_MOE_HOT_PROFILE");
    }
    const bool have_prof = pp && pp[0] && dma_read_profile(pp, prof);

    for (auto & L : s->layers) {
        L.ring_slot.assign(L.n_expert, -1);
        L.win_cnt.assign(L.n_expert, 0);
        L.win_ring.assign(s->ad_win, {});
        L.cur_seen.assign(L.n_expert, 0);
        L.pred_mark.assign(L.n_expert, 0);
        L.s_tok.assign(L.n_expert, 0);
        L.s_dslot.assign(L.n_expert, -1);
        if (have_prof) {
            auto it = prof.find(L.il);
            if (it != prof.end() && (int64_t) it->second.size() == L.n_expert) {
                L.prof = it->second;
            }
        }
        dma_bank & B = s->banks[L.bank];
        L.view.up        = B.up;
        L.view.gate      = B.gate;
        L.view.down      = B.down;
        L.view.dev_table = B.table;
        L.view.handle    = &L;
    }
    if (!have_prof) {
        s->warm_done = true;
    }

    if (!s->inline_issue && !s->bridge) {
        s->issuer = std::thread(dma_issuer_run, s);
    }
    for (int i = 0; i < n_fill; ++i) {
        s->fillers.emplace_back(dma_filler_run, s);
    }
    s->ctr.share = s->share;
    g_dma = s;

    if (s->bridge) {
        LLAMA_LOG_INFO("moe-dma: [TAG_FN_R4_BRIDGE_DMA] bridge mode: the bridge's executor plans the share, the device fetches it\n");
    }
    LLAMA_LOG_INFO("moe-dma: %zu host expert layers, %zu banks x %d slots (%.1f MiB VRAM), %s ring %d x %.2f MiB, share %s%.2f, "
            "prefetch %s (%d slots, %d fills, top %d), admit %d/%d, %.0f MiB fills per step%s\n",
            s->layers.size(), s->banks.size(), s->K, vram/1048576.0, s->ring.pinned ? "pinned" : "plain", s->n_ring,
            s->slot_size/1048576.0, s->share_auto ? "auto from " : "", s->share, s->prefetch ? "on" : "off", s->pf_slots,
            s->pf_fill, s->pf_k, s->ad_min, s->ad_win, s->fill_step/1048576.0, have_prof ? ", warm start from the profile" : "");
    return true;
}

void llama_moe_dma_free(const void * owner) {
    std::lock_guard<std::mutex> init_lock(g_dma_init);
    if (!g_dma || g_dma->owner != owner) {
        return;
    }
    dma_state * s = g_dma;
    g_dma = nullptr;
    dma_destroy(s);
}

void llama_moe_dma_step(const void * owner) {
    dma_state * s = g_dma;
    if (!s || s->owner != owner) {
        return;
    }
    dma_step_impl(s);
    s->ctr.share = s->share;
}

const llama_moe_dma_view * llama_moe_dma_lookup(ggml_backend_sched_t sched, const ggml_tensor * up_exps, int64_t n_tokens) {
    dma_state * s = g_dma;
    if (!s || s->bridge || n_tokens < 1 || n_tokens > LLAMA_MOE_DMA_MAX_T || !gen5::sched_has(sched, s->compute)) {
        return nullptr; // [TAG_FN_R4_BRIDGE_DMA] bridge mode has no plan / fence path
    }
    auto it = s->by_up.find(up_exps);
    return it != s->by_up.end() ? &s->layers[it->second].view : nullptr;
}

int llama_moe_dma_pred_k() {
    dma_state * s = g_dma;
    return s && s->prefetch ? 16 : 0;
}

ggml_tensor * llama_moe_dma_build_plan(ggml_context * ctx, const llama_moe_dma_view * v, ggml_tensor * ids,
        ggml_tensor * pred_next, const ggml_tensor * hot_table, int32_t hot_miss) {
    dma_layer * L = (dma_layer *) v->handle;
    L->hot_tbl  = hot_table ? (const int32_t *) hot_table->data : nullptr;
    L->hot_miss = hot_miss;
    ggml_tensor * args[2] = { ids, pred_next };
    ggml_tensor * t = ggml_custom_4d(ctx, GGML_TYPE_I32, 1, L->n_expert, 1, 1, args, pred_next ? 2 : 1, dma_plan_op, 1, L);
    ggml_format_name(t, "moe_dma_plan-%d", L->il);
    return t;
}

ggml_tensor * llama_moe_dma_build_fence(ggml_context * ctx, ggml_cgraph * gf, const llama_moe_dma_view * v, ggml_tensor * cpu_down) {
    dma_layer * L = (dma_layer *) v->handle;
    ggml_tensor * args[1] = { ggml_view_1d(ctx, cpu_down, 1, 0) };
    ggml_tensor * f = ggml_custom_4d(ctx, GGML_TYPE_F32, 1, 1, 1, 1, args, 1, dma_fence_op, 1, L);
    ggml_format_name(f, "moe_dma_fence-%d", L->il);
    ggml_build_forward_expand(gf, f);
    return f;
}

llama_moe_dma_counters llama_moe_dma_get_counters() {
    dma_state * s = g_dma;
    if (!s) {
        return {};
    }
    llama_moe_dma_counters c = s->ctr;
    c.share = s->share;
    std::lock_guard<std::mutex> lk(s->fmtx);
    c.fills       = s->ctr.fills;
    c.fill_urgent = s->ctr.fill_urgent;
    return c;
}

// ---- [TAG_FN_R4_BRIDGE_DMA] bridge mode ------------------------------------------------------------------------------

bool llama_moe_dma_bridge_mode() {
    dma_state * s = g_dma;
    return s && s->bridge;
}

const llama_moe_dma_view * llama_moe_dma_bridge_lookup(const ggml_tensor * up_exps, int64_t n_tokens) {
    dma_state * s = g_dma;
    if (!s || !s->bridge || n_tokens < 1 || n_tokens > LLAMA_MOE_DMA_MAX_T) {
        return nullptr;
    }
    auto it = s->by_up.find(up_exps);
    return it != s->by_up.end() ? &s->layers[it->second].view : nullptr;
}

bool llama_moe_dma_bridge_ring(const void * owner, void ** ptr, size_t * size, int * max_fetch) {
    dma_state * s = g_dma;
    if (!s || !s->bridge || s->owner != owner || !s->ring.ptr) {
        return false; // pinned or plain (CPU device): the device side checks that it can read it
    }
    *ptr       = s->ring.ptr;
    *size      = (size_t) s->n_ring*s->slot_size;
    *max_fetch = s->K;
    return true;
}

int llama_moe_dma_bridge_plan(void * handle, const int32_t * ids, int n_used, int n_tokens, const int32_t * hot_tbl,
        int32_t hot_miss, int max_copy, uint64_t * off, int32_t * slot, int32_t * slot_ids, int32_t * d_experts) {
    dma_state * s = g_dma;
    dma_layer * L = (dma_layer *) handle;
    if (!s || !s->bridge || !L || L->pos < 0 || L->pos >= (int) s->layers.size() || &s->layers[L->pos] != L) {
        return -1;
    }
    const dma_bank & B = s->banks[L->bank];
    const int64_t n_exp = L->n_expert;
    auto is_hot = [&](int32_t e) { return hot_tbl && hot_tbl[e] != hot_miss; };

    // unique routed experts and tokens per expert, in routing order; the cold ones feed the ring admission
    auto & tok    = L->s_tok;
    auto & uniq   = L->s_uniq;
    auto & cold   = L->s_cold;
    auto & d_slot = L->s_dslot;
    uniq.clear();
    cold.clear();
    for (int t = 0; t < n_tokens; ++t) {
        for (int i = 0; i < n_used; ++i) {
            const int32_t e = ids[(size_t) t*n_used + i];
            if (e < 0 || e >= n_exp) {
                continue;
            }
            if (tok[e]++ == 0) {
                uniq.push_back(e);
            }
        }
    }
    for (int32_t e : uniq) {
        if (is_hot(e)) {
            continue;
        }
        cold.push_back(e);
        if (!L->cur_seen[e]) {
            L->cur_seen[e] = 1;
            L->cur_ids.push_back(e);
        }
    }

    // D: ring-ready cold experts, most tokens first, up to share x cold, the bank and the fetch budget
    const double share = s->br_share.load(std::memory_order_relaxed);
    const int cap = std::min({ max_copy, B.n_slots, (int) std::lround(share*(double) cold.size()) });
    int n = 0;
    uint64_t n_ready = 0;
    if (cap > 0) {
        std::vector<std::pair<int32_t, int32_t>> cand; // (expert, ring slot)
        std::lock_guard<std::mutex> lk(s->ring_mtx);
        for (int32_t e : cold) {
            const int32_t r = L->ring_slot[e];
            if (r >= 0 && s->rs[r].state.load(std::memory_order_acquire) == RING_READY) {
                cand.push_back({ e, r });
            }
        }
        n_ready = cand.size();
        std::stable_sort(cand.begin(), cand.end(), [&](const std::pair<int32_t, int32_t> & a, const std::pair<int32_t, int32_t> & b) {
            return tok[a.first] != tok[b.first] ? tok[a.first] > tok[b.first] : a.first < b.first;
        });
        for (const auto & c : cand) {
            if (n >= cap) {
                break;
            }
            d_slot[c.first]       = n;
            off[n]                = (uint64_t) c.second*s->slot_size;
            slot[n]               = n;
            d_experts[n]          = c.first;
            s->rs[c.second].last_use = s->step;
            n++;
        }
    }

    // the bank slot of every routed (slot, token): its D slot, else the zero slot
    for (int t = 0; t < n_tokens; ++t) {
        for (int i = 0; i < n_used; ++i) {
            const int32_t e = ids[(size_t) t*n_used + i];
            slot_ids[(size_t) t*n_used + i] = e >= 0 && e < n_exp && d_slot[e] >= 0 ? d_slot[e] : B.n_slots;
        }
    }
    for (int32_t e : uniq) {
        tok[e]    = 0;
        d_slot[e] = -1;
    }

    s->ctr.layer_steps++;
    s->ctr.cold          += cold.size();
    s->ctr.dma_on_demand += n;
    s->ctr.ring_ready    += n_ready;
    return n;
}

void llama_moe_dma_bridge_feedback(const llama_moe_dma_bridge_step & st) {
    dma_state * s = g_dma;
    if (!s || !s->bridge || st.n_layers <= 0) {
        return;
    }
    const double a = 0.1; // EMA weight of a graph
    if (st.n_cpu > 0) {
        const double c = st.cpu_us/(double) st.n_cpu;
        s->br_cpu_us = s->br_cpu_us > 0.0 ? (1.0 - a)*s->br_cpu_us + a*c : c;
    }
    if (st.n_fetch > 0) {
        const double f = st.fetch_us/(double) st.n_fetch;
        s->br_fetch_us = s->br_fetch_us > 0.0 ? (1.0 - a)*s->br_fetch_us + a*f : f;
    }
    const double wait = st.gpu_wait_us/(double) st.n_layers;
    s->br_graphs++;
    s->br_wait_sum += wait;
    if (!s->share_auto) {
        return;
    }
    // the rate balance: the share at which the fetched experts take as long as the CPU's (both measured per expert);
    // the share never goes past 1.5x of it
    double cap = s->br_max;
    if (s->br_cpu_us > 0.0 && s->br_fetch_us > 0.0) {
        cap = std::min(cap, 1.5*s->br_cpu_us/(s->br_cpu_us + s->br_fetch_us));
    }
    double share = s->br_share.load(std::memory_order_relaxed);
    if (wait > s->br_wait_hi) {
        share += s->br_step;        // the device idled waiting for the CPU part: more to the GPU
    } else if (wait < s->br_wait_lo) {
        share -= s->br_step;        // the CPU part was ready: the GPU is the long pole
    }
    share = std::max(0.0, std::min(cap, share));
    s->br_share.store(share, std::memory_order_relaxed);
    s->share = share;
}

void llama_moe_dma_wait_idle() {
    dma_state * s = g_dma;
    if (!s) {
        return;
    }
    dma_wait_issuer_idle(s);
    std::unique_lock<std::mutex> lk(s->fmtx);
    s->fcv_idle.wait(lk, [&]() { return s->stop || (s->fq.empty() && s->fq_urgent.empty() && s->f_running == 0); });
}
