#include "llama-moecache.h"

#include "llama-fn-auto.h" // [TAG_FN_AUTO]
#include "llama-moe-decay.h" // [TAG_FN_R4_ADAPT_DECAY]
#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

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
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

struct layer_state {
    llama_moe_cache_layer pub;

    // LRU bookkeeping (host side; the tables mirror expert_slot)
    std::vector<int32_t>  slot_expert;   // slot -> expert id, -1 when empty
    std::vector<int32_t>  expert_slot;   // expert id -> slot, -1 when uncached
    std::vector<uint64_t> slot_last_use; // slot -> lamport clock of last hit
    std::vector<int32_t>  pending;       // uncached ids observed since last step (dedup, obs order)

    std::vector<bool>     slot_in_flight; // slot has an upload pending
    // [TAG_MOE_CACHE_INFLIGHT] expert has an upload pending. expert_slot[] only becomes valid
    // when the worker reports done, so without this an expert in flight looks uncached on every
    // following step and gets scheduled into a second slot.
    std::vector<bool>     expert_in_flight;

    uint64_t n_hit  = 0;
    uint64_t n_miss = 0;

    // [TAG_FN_MOE_HOT_ADAPT] sightings per expert in the last W decode steps; the observer fills cur_* during a step
    std::vector<uint16_t>             win_cnt;   // [n_expert]
    std::vector<std::vector<int32_t>> win_ring;  // W steps of unique ids
    std::vector<int32_t>              cur_ids;   // unique ids of the running step
    std::vector<uint8_t>              cur_seen;  // [n_expert]
    std::vector<double>               prof;      // static profile counts (eviction tie-break)
    int64_t                           tbl_off = 0; // row offset in the group's table tensor

    // [TAG_FN_R4_ADAPT_DECAY] the decayed policy (LLAMA_MOE_HOT_DECAY): routing counts, and the prompt's counts not yet
    // folded in (the observer fills both; the owner's step reads them, never while a graph of the owner runs)
    std::vector<float>                dcnt;      // [n_expert]
    std::vector<float>                dseed;     // [n_expert]
    bool                              dseed_any = false;
    float                             dseed_tok = 0.0f; // [TAG_FN_L3_POLICY_SEED] prompt tokens in dseed (LLAMA_MOE_HOT_SEED_NORM)

    // [TAG_FN_R1_PFS_LEND] the slots of this layer lie in the range the prefill stream borrowed: both tables say "not
    // hot" for every expert, nothing uploads into it, and llama_moe_hot_unlend refills it from slot_expert
    bool                              lent = false;

    // [TAG_FN_L3_POLICY_POOL] the layer's slots are those of a pool shared by its expert shape class: slot_expert and
    // slot_in_flight are pool sized and hold only this layer's entries (resident / upload into it in flight)
    int                               pool   = -1;
    int                               pool_k = -1; // the layer's index inside its pool
};

// [TAG_FN_L3_POLICY_POOL] one dynamic slot pool per expert shape class (LLAMA_MOE_HOT_POOL=1): any layer of the class can
// take any slot of it. The class tensors (n_slots + 1 slots, the last one zero) are every member layer's up_c/gate_c/down_c
// and n_slots is every member's miss value, so the graph, the bridge and the CPU skip tables work unchanged.
struct hot_pool {
    std::vector<size_t>  layers;     // mc->layers indices, pool_k order
    std::vector<int32_t> slot_layer; // the mc->layers index that holds or receives a slot, -1 = free
    int32_t              n_slots = 0;
};

struct upload_job {
    size_t  layer_idx;
    int32_t expert;
    int32_t slot;
    bool    done = false;
    bool    failed = false; // [TAG_FN_L3_POLICY_UPLOAD] not uploaded (a slice out of range): the slot comes back empty
};

struct moe_cache {
    int32_t n_slots     = 0;
    int32_t max_inserts = 2;

    uint64_t clock   = 0;
    uint64_t n_steps = 0;

    std::mutex mtx; // guards pending lists + clock (observe runs during graph exec)

    std::vector<layer_state> layers;
    std::map<const ggml_tensor *, size_t> by_up_src;

    std::vector<ggml_context *>         ctxs;
    std::vector<ggml_backend_buffer_t>  bufs;

    // [TAG_FN_MOE_HOT] static hot set: tables written once, no observer / worker / step
    bool    hot       = false;
    int     hot_max_t = 0;
    uint64_t hot_steps = 0;

    // [TAG_FN_MOE_HOT_ADAPT] adaptive hot set: tables change only in step() of the owning context, after its compute
    // has been synchronized; victims leave both tables before their slot is overwritten, and a new expert enters them
    // only after its upload (own backend and stream, pinned staging) has completed
    bool          adapt       = false;
    const void *  owner       = nullptr;
    int           ad_win      = 16;
    int           ad_min      = 3;
    int           ad_hyst     = 1;
    size_t        ad_bytes    = 64u << 20;
    int           ad_verify   = 0;
    int           ad_pos      = 0;
    uint64_t      ad_steps    = 0;
    uint64_t      ad_admitted = 0;
    uint64_t      ad_bad      = 0;
    bool          ad_stats    = false;

    // [TAG_FN_R4_ADAPT_DECAY] the decayed policy instead of the window (llama-moe-decay.h), and the learned set saved
    // as a routing profile (LLAMA_MOE_HOT_SAVE=<file>, every LLAMA_MOE_HOT_SAVE_EVERY decode steps and at the owner's end)
    bool                   dc_on      = false;
    llama_moe_decay_params dc;
    size_t                 dc_bytes   = 0;     // upload budget per pass (LLAMA_MOE_HOT_DECAY_MIB, else ADAPT_MIB x every)
    uint64_t               dc_swaps   = 0;
    uint64_t               dc_passes  = 0;
    std::string            save_path;
    uint64_t               save_every = 0;
    ggml_backend_t        up_backend = nullptr;
    ggml_backend_buffer_t staging    = nullptr;
    uint8_t *             stage_ptr  = nullptr;
    size_t                stage_size = 0;
    std::vector<uint8_t>  stage_buf;
    ggml_tensor *         tbl_all    = nullptr; // one group's device tables, [1, n_expert, n_layers]
    std::vector<int32_t>  tbl_mirror;
    bool                  tbl_dirty  = false;

    // async upload worker: slices are copied to the device off the decode
    // thread; the new table mapping is only published at a later step() once
    // the upload has completed, so a running graph never reads a torn slot
    std::thread              worker;
    std::mutex               wmtx;
    std::condition_variable  wcv;
    std::deque<upload_job>   todo;
    std::vector<upload_job>  done;
    bool                     stop = false;

    // [TAG_FN_R1_PFS_LEND] the worker holds a batch (its uploads may still be writing slots); wcv_idle: it put one down
    bool                     worker_busy = false;
    std::condition_variable  wcv_idle;
    bool                     lent_any    = false;
    size_t                   lent_bytes  = 0;
    uint64_t                 n_lend      = 0;
    double                   unlend_ms   = 0.0;
    size_t                   unlend_b    = 0;

    // [TAG_FN_L3_POLICY_POOL]
    std::vector<hot_pool> pools;

    // [TAG_FN_L3_POLICY_SEED] prompt routing from the graph node instead of the CPU MUL_MAT_ID observer
    bool                  sd_node   = false;
    float                 sd_norm   = 0.0f; // LLAMA_MOE_HOT_SEED_NORM: the prompt folds in as this many decode steps of its mix
    std::atomic<uint64_t> sd_tokens { 0 }; // prompt tokens counted by the node (owner's graphs only; stats)

    // [TAG_FN_L3_POLICY_STATE] the learned set saved and loaded per model
    std::string st_path;
    std::string st_model;     // the model's name (general.name): a state of another model is not loaded
    uint64_t    st_every  = 0;
    uint64_t    st_saves  = 0;
    bool        st_load   = true;

    // [TAG_FN_L3_POLICY_UPLOAD] two small pinned halves (one fills while the other uploads) and a per-step byte bucket
    bool                  up_pipe     = false;
    size_t                up_half     = 8u << 20;
    ggml_backend_buffer_t up_stage    = nullptr;
    uint8_t *             up_stage_p  = nullptr;
    ggml_backend_event_t  up_ev[2]    = { nullptr, nullptr };
    bool                  up_ev_rec[2] = { false, false };
    size_t                up_rate     = 0;     // bytes per decode step, 0 = no limit
    size_t                up_cap      = 0;     // bucket size: two steps' worth, at least the largest slice
    size_t                up_bucket   = 0;     // guarded by wmtx
    bool                  up_drain    = false; // guarded by wmtx: a lend waits for the worker, so the bucket is ignored
    std::atomic<uint64_t> up_bytes    { 0 };   // written by the worker, read by the stats line
    std::atomic<uint64_t> up_waits    { 0 };   // times the bucket made the worker wait
    uint64_t              up_batches  = 0;     // worker only
    size_t                dc_burst       = 0;     // [TAG_FN_L3_POLICY_BURST] LLAMA_MOE_HOT_BURST_MIB, 0 = off
    float                 dc_burst_ratio = 2.0f;  // LLAMA_MOE_HOT_BURST_RATIO
    uint64_t              dc_burst_n     = 0;     // swaps taken on the burst budget
    uint64_t              dc_burst_logs  = 0;
    uint64_t              sd_folded   = 0;     // [TAG_FN_L3_POLICY_SEED] prompt tokens folded so far (owner's step)
    uint64_t              sd_folds    = 0;
    std::atomic<uint64_t> st_step_us  { 0 };   // [TAG_FN_L3_POLICY] LLAMA_MOE_HOT_STATS: host time of the owner's steps
    std::atomic<uint64_t> st_step_n   { 0 };
};

moe_cache * g_cache = nullptr;

void hot_adapt_step(moe_cache * mc); // [TAG_FN_MOE_HOT_ADAPT]
void hot_state_save(moe_cache * mc); // [TAG_FN_L3_POLICY_STATE]
void hot_state_load(moe_cache * mc);
std::mutex g_init_mtx;
bool g_init_done = false;

int parse_layer_from_name(const char * name) {
    // "blk.<il>.ffn_gate_exps.weight"
    if (strncmp(name, "blk.", 4) != 0) {
        return -1;
    }
    return atoi(name + 4);
}

void moe_obs_cb(const char * name, const struct ggml_tensor * ids, void * ud) {
    moe_cache * mc = (moe_cache *) ud;

    const int64_t n_ids    = ids->ne[0];
    const int64_t n_tokens = ids->ne[1];
    if (n_tokens > 4) {
        return; // batch/prefill: the cache graph is not built there, don't pollute the LRU
    }

    const int il = parse_layer_from_name(name);
    if (il < 0) {
        return;
    }

    layer_state * ls = nullptr;
    for (auto & l : mc->layers) {
        if (l.pub.il == il) { ls = &l; break; }
    }
    if (!ls) {
        return;
    }

    std::lock_guard<std::mutex> lock(mc->mtx);
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t i = 0; i < n_ids; ++i) {
            const int32_t id = *(const int32_t *) ((const char *) ids->data + t*ids->nb[1] + i*ids->nb[0]);
            if (id < 0 || id >= (int32_t) ls->expert_slot.size()) {
                continue;
            }
            const int32_t slot = ls->expert_slot[id];
            if (slot >= 0) {
                ls->n_hit++;
                ls->slot_last_use[slot] = ++mc->clock;
            } else {
                ls->n_miss++;
                if (ls->expert_in_flight[id]) {
                    continue; // [TAG_MOE_CACHE_INFLIGHT] upload already scheduled for this id
                }
                bool dup = false;
                for (int32_t p : ls->pending) {
                    if (p == id) { dup = true; break; }
                }
                if (!dup) {
                    ls->pending.push_back(id);
                }
            }
        }
    }
}

void upload_slice(ggml_tensor * dst_c, const ggml_tensor * src, int32_t expert, int32_t slot) {
    const size_t sz = src->nb[2];
    if ((size_t) slot*dst_c->nb[2] + sz > ggml_nbytes(dst_c) || (size_t) expert*sz + sz > ggml_nbytes(src)) {
        LLAMA_LOG_ERROR("moe-cache: bad upload %s <- %s expert=%d slot=%d sz=%zu dst_nb2=%zu dst_bytes=%zu src_bytes=%zu\n",
                dst_c->name, src->name, expert, slot, sz, dst_c->nb[2], ggml_nbytes(dst_c), ggml_nbytes(src));
        return;
    }
    ggml_backend_tensor_set(dst_c, (const char *) src->data + (size_t) expert*sz, (size_t) slot*dst_c->nb[2], sz);
}

void set_table_entry(llama_moe_cache_layer & pub, int32_t expert, int32_t slot_or_dummy) {
    const int32_t v = slot_or_dummy;
    ggml_backend_tensor_set(pub.dev_table,  &v, (size_t) expert*sizeof(int32_t), sizeof(int32_t));
    ggml_backend_tensor_set(pub.host_table, &v, (size_t) expert*sizeof(int32_t), sizeof(int32_t));
}

} // namespace

void llama_moe_cache_init(const llama_model & model, int32_t n_slots, int32_t max_inserts) {
    std::lock_guard<std::mutex> init_lock(g_init_mtx);
    if (g_init_done) {
        return;
    }
    // [TAG_FN_MOE_HOT] a hot profile replaces the LRU cache; llama_moe_hot_init builds it after the scheduler reserve
    // [TAG_FN_AUTO] from the environment or the model's automatic profile
    if (llama_moe_hot_wanted(model)) {
        if (n_slots > 0) {
            LLAMA_LOG_WARN("%s: a MoE hot set is configured, --moe-expert-cache %d is ignored\n", __func__, n_slots);
        }
        return;
    }
    [&]() {
        if (n_slots <= 0) {
            // [TAG_FN_AUTO] no attempt: a -fit memory probe context comes here first and must not block the hot set of
            // the loaded model (its profile values are not visible on the probe model)
            return;
        }

        auto * mc = new moe_cache();
        mc->n_slots = n_slots;
        if (max_inserts > 0) {
            mc->max_inserts = max_inserts;
        }

        // collect the host-resident expert layers, grouped by the device buffer
        // type of that layer's router (the cache lives next to the router)
        struct cand { int il; const llama_layer * l; };
        std::map<ggml_backend_buffer_type_t, std::vector<cand>> groups;

        for (size_t il = 0; il < model.layers.size(); ++il) {
            const auto & l = model.layers[il];
            if (!l.ffn_up_exps || !l.ffn_gate_exps || !l.ffn_down_exps || !l.ffn_gate_inp) {
                continue;
            }
            if (!l.ffn_up_exps->data || !l.ffn_gate_exps->data || !l.ffn_down_exps->data) {
                continue; // dry-run / memory-estimation model: weights not loaded, don't bind to it
            }
            if (!l.ffn_up_exps->buffer || !ggml_backend_buffer_is_host(l.ffn_up_exps->buffer)) {
                continue; // experts already on a device: nothing to cache
            }
            if (!l.ffn_gate_inp->buffer || ggml_backend_buffer_is_host(l.ffn_gate_inp->buffer)) {
                continue; // no device home for the cache
            }
            groups[ggml_backend_buffer_get_type(l.ffn_gate_inp->buffer)].push_back({(int) il, &l});
        }

        if (groups.empty()) {
            LLAMA_LOG_INFO("%s: LLAMA_MOE_CACHE_SLOTS=%d but no host-resident expert layers found - disabled\n", __func__, n_slots);
            delete mc;
            return;
        }

        // host buffer for the CPU-side tables
        std::vector<cand> all;
        for (auto & g : groups) {
            all.insert(all.end(), g.second.begin(), g.second.end());
        }

        auto alloc_group = [&](ggml_backend_buffer_type_t buft, const std::vector<cand> & cands, bool tables_only) -> bool {
            ggml_init_params ip = {
                /*.mem_size  =*/ ggml_tensor_overhead()*(cands.size()*4 + 8),
                /*.mem_buffer=*/ nullptr,
                /*.no_alloc  =*/ true,
            };
            ggml_context * ctx = ggml_init(ip);
            if (!ctx) {
                return false;
            }
            mc->ctxs.push_back(ctx);

            for (const auto & c : cands) {
                layer_state * ls = nullptr;
                for (auto & l : mc->layers) {
                    if (l.pub.il == c.il) { ls = &l; break; }
                }
                if (!ls) {
                    mc->layers.push_back({});
                    ls = &mc->layers.back();
                    ls->pub.il       = c.il;
                    ls->pub.n_slots  = n_slots;
                    ls->pub.up_src   = c.l->ffn_up_exps;
                    ls->pub.gate_src = c.l->ffn_gate_exps;
                    ls->pub.down_src = c.l->ffn_down_exps;
                }

                if (tables_only) {
                    ls->pub.host_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, ls->pub.up_src->ne[2]);
                    ggml_format_name(ls->pub.host_table, "moe_cache_htbl.%d", c.il);
                } else {
                    const ggml_tensor * u = c.l->ffn_up_exps;
                    const ggml_tensor * g = c.l->ffn_gate_exps;
                    const ggml_tensor * d = c.l->ffn_down_exps;
                    ls->pub.up_c   = ggml_new_tensor_3d(ctx, u->type, u->ne[0], u->ne[1], n_slots + 1);
                    ls->pub.gate_c = ggml_new_tensor_3d(ctx, g->type, g->ne[0], g->ne[1], n_slots + 1);
                    ls->pub.down_c = ggml_new_tensor_3d(ctx, d->type, d->ne[0], d->ne[1], n_slots + 1);
                    ls->pub.dev_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, u->ne[2]);
                    ggml_format_name(ls->pub.up_c,      "moe_cache_up.%d",   c.il);
                    ggml_format_name(ls->pub.gate_c,    "moe_cache_gate.%d", c.il);
                    ggml_format_name(ls->pub.down_c,    "moe_cache_down.%d", c.il);
                    ggml_format_name(ls->pub.dev_table, "moe_cache_tbl.%d",  c.il);
                }
            }

            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
            if (!buf) {
                LLAMA_LOG_WARN("%s: failed to allocate MoE cache buffer on %s - cache disabled\n",
                        __func__, ggml_backend_buft_name(buft));
                return false;
            }
            ggml_backend_buffer_clear(buf, 0);
            mc->bufs.push_back(buf);
            return true;
        };

        bool ok = alloc_group(ggml_backend_cpu_buffer_type(), all, /*tables_only=*/true);
        for (auto & g : groups) {
            if (!ok) {
                break;
            }
            ok = alloc_group(g.first, g.second, /*tables_only=*/false);
        }

        if (!ok) {
            for (auto * b : mc->bufs) { ggml_backend_buffer_free(b); }
            for (auto * c : mc->ctxs) { ggml_free(c); }
            delete mc;
            g_init_done = true; // a real model was seen and allocation failed: stay disabled
            return;
        }

        // init LRU state + tables (everything uncached -> dummy slot n_slots)
        size_t vram = 0;
        for (auto & ls : mc->layers) {
            const int64_t n_expert = ls.pub.up_src->ne[2];
            ls.slot_expert.assign(n_slots, -1);
            ls.expert_slot.assign(n_expert, -1);
            ls.slot_last_use.assign(n_slots, 0);
            ls.slot_in_flight.assign(n_slots, false);
            ls.expert_in_flight.assign(n_expert, false); // [TAG_MOE_CACHE_INFLIGHT]

            std::vector<int32_t> dummy(n_expert, n_slots);
            ggml_backend_tensor_set(ls.pub.dev_table,  dummy.data(), 0, n_expert*sizeof(int32_t));
            ggml_backend_tensor_set(ls.pub.host_table, dummy.data(), 0, n_expert*sizeof(int32_t));

            mc->by_up_src[ls.pub.up_src] = &ls - mc->layers.data();
            vram += ggml_nbytes(ls.pub.up_c) + ggml_nbytes(ls.pub.gate_c) + ggml_nbytes(ls.pub.down_c);
            LLAMA_LOG_DEBUG("moe-cache: init layer %d '%s' %zu bytes/expert\n",
                    ls.pub.il, ls.pub.up_src->name, ls.pub.up_src->nb[2]);
        }

        mc->worker = std::thread([mc]() {
            for (;;) {
                upload_job j;
                {
                    std::unique_lock<std::mutex> lk(mc->wmtx);
                    mc->wcv.wait(lk, [mc]() { return mc->stop || !mc->todo.empty(); });
                    if (mc->stop) {
                        return;
                    }
                    j = mc->todo.front();
                    mc->todo.pop_front();
                }
                auto & ls = mc->layers[j.layer_idx];
                upload_slice(ls.pub.up_c,   ls.pub.up_src,   j.expert, j.slot);
                upload_slice(ls.pub.gate_c, ls.pub.gate_src, j.expert, j.slot);
                upload_slice(ls.pub.down_c, ls.pub.down_src, j.expert, j.slot);
                {
                    std::lock_guard<std::mutex> lk(mc->wmtx);
                    j.done = true;
                    mc->done.push_back(j);
                }
            }
        });

        ggml_set_moe_obs_callback(moe_obs_cb, mc);
        g_cache = mc;
        g_init_done = true;

        LLAMA_LOG_INFO("%s: MoE expert cache enabled: %zu layers x %d slots, %d inserts/step, %.1f MiB device memory\n",
                __func__, mc->layers.size(), n_slots, mc->max_inserts, vram/1024.0/1024.0);
    }();
}

const llama_moe_cache_layer * llama_moe_cache_lookup(const ggml_tensor * up_exps) {
    if (!g_cache) {
        return nullptr;
    }
    auto it = g_cache->by_up_src.find(up_exps);
    if (it == g_cache->by_up_src.end()) {
        return nullptr;
    }
    return &g_cache->layers[it->second].pub;
}

void llama_moe_cache_step(const void * ctx) {
    moe_cache * mc = g_cache;
    if (!mc) {
        return;
    }
    if (mc->hot) { // [TAG_FN_MOE_HOT] a static hot set never changes; an adaptive one only in its owner's step
        if (mc->adapt && ctx == mc->owner) {
            hot_adapt_step(mc);
        }
        return;
    }

    // 1) publish completed uploads (sync point: no graph is executing)
    {
        std::lock_guard<std::mutex> wlk(mc->wmtx);
        std::lock_guard<std::mutex> lk(mc->mtx);
        for (const auto & j : mc->done) {
            auto & ls = mc->layers[j.layer_idx];
            ls.slot_expert[j.slot]     = j.expert;
            ls.expert_slot[j.expert]   = j.slot;
            ls.slot_last_use[j.slot]   = ++mc->clock;
            ls.slot_in_flight[j.slot]  = false;
            ls.expert_in_flight[j.expert] = false; // [TAG_MOE_CACHE_INFLIGHT]
            set_table_entry(ls.pub, j.expert, j.slot);
        }
        mc->done.clear();
    }

    std::lock_guard<std::mutex> lock(mc->mtx);
    mc->n_steps++;

    // 2) schedule new uploads: evict at a sync point (clear the victim's table
    //    entry now), then hand the slice copies to the worker
    for (size_t li = 0; li < mc->layers.size(); ++li) {
        auto & ls = mc->layers[li];
        if (ls.pending.empty()) {
            continue;
        }

        int budget = mc->max_inserts;
        for (auto it = ls.pending.rbegin(); it != ls.pending.rend() && budget > 0; ++it, --budget) {
            const int32_t id = *it;
            // [TAG_MOE_CACHE_INFLIGHT] resident already, or already on its way to a slot
            if (ls.expert_slot[id] >= 0 || ls.expert_in_flight[id]) {
                ++budget; // an id we did not upload must not cost an insert
                continue;
            }

            // victim: an empty non-in-flight slot if any, else the LRU non-in-flight slot
            int32_t slot = -1;
            uint64_t best = UINT64_MAX;
            for (int32_t s = 0; s < mc->n_slots; ++s) {
                if (ls.slot_in_flight[s]) {
                    continue;
                }
                if (ls.slot_expert[s] < 0) { slot = s; break; }
                if (ls.slot_last_use[s] < best) { best = ls.slot_last_use[s]; slot = s; }
            }
            if (slot < 0) {
                break; // every slot is in flight; try again next step
            }

            const int32_t victim = ls.slot_expert[slot];
            if (victim >= 0) {
                ls.expert_slot[victim] = -1;
                ls.slot_expert[slot]   = -1;
                set_table_entry(ls.pub, victim, mc->n_slots);
            }
            ls.slot_in_flight[slot]   = true;
            ls.expert_in_flight[id]   = true; // [TAG_MOE_CACHE_INFLIGHT]

            std::lock_guard<std::mutex> wlk(mc->wmtx);
            mc->todo.push_back({li, id, slot});
        }
        ls.pending.clear();
    }
    mc->wcv.notify_one();

    if (mc->n_steps % 512 == 0) {
        uint64_t h = 0, m = 0;
        for (auto & ls : mc->layers) { h += ls.n_hit; m += ls.n_miss; }
        LLAMA_LOG_DEBUG("moe-cache: steps=%" PRIu64 " hits=%" PRIu64 " misses=%" PRIu64 " hit-rate=%.1f%%\n",
                mc->n_steps, h, m, h + m ? 100.0*h/(h + m) : 0.0);
    }
}

// [TAG_FN_MOE_HOT] ------------------------------------------------------------------------------------------------

namespace {

struct hot_stats {
    std::vector<uint64_t> hit;   // by layer index
    std::vector<uint64_t> tot;
    int first_il = -1;
};

hot_stats g_hot_stats;

// LLAMA_MOE_HOT_STATS: the CPU MUL_MAT_ID of every host-resident ffn_gate_exps reports its routed ids here (the
// observer hook of the LRU cache, ggml-cpu.c); decode ubatches only, counted against the fixed host table
void hot_obs_cb(const char * name, const struct ggml_tensor * ids, void * ud) {
    moe_cache * mc = (moe_cache *) ud;
    const int64_t n_used   = ids->ne[0];
    const int64_t n_tokens = ids->ne[1];
    if (n_tokens > mc->hot_max_t) {
        // [TAG_FN_R4_ADAPT_DECAY] prefill: the prompt's routing x LLAMA_MOE_HOT_SEED warms the decayed counts at the next
        // step; otherwise not counted, and the adaptive set stays frozen
        // [TAG_FN_L3_POLICY_SEED] with the seed node the graph reports every prompt ubatch, so the observer does not
        if (mc->adapt && mc->dc_on && mc->dc.seed > 0.0f && !mc->sd_node) {
            const int il = parse_layer_from_name(name);
            for (auto & l : mc->layers) {
                if (l.pub.il != il || l.dseed.empty()) {
                    continue;
                }
                for (int64_t t = 0; t < n_tokens; ++t) {
                    for (int64_t i = 0; i < n_used; ++i) {
                        const int32_t e = *(const int32_t *) ((const char *) ids->data + t*ids->nb[1] + i*ids->nb[0]);
                        llama_moe_decay_count(l.dseed, &e, 1);
                    }
                }
                l.dseed_any = true;
                l.dseed_tok += (float) n_tokens;
                break;
            }
        }
        return;
    }
    const int il = parse_layer_from_name(name);
    layer_state * ls = nullptr;
    for (auto & l : mc->layers) {
        if (l.pub.il == il) { ls = &l; break; }
    }
    const bool stats = mc->ad_stats && il >= 0 && il < (int) g_hot_stats.hit.size();
    if (ls) {
        const int32_t * tbl = (const int32_t *) ls->pub.host_table->data;
        const int64_t n_exp = ls->pub.host_table->ne[1];
        for (int64_t t = 0; t < n_tokens; ++t) {
            for (int64_t i = 0; i < n_used; ++i) {
                const int32_t e = *(const int32_t *) ((const char *) ids->data + t*ids->nb[1] + i*ids->nb[0]);
                if (e < 0 || e >= n_exp) {
                    continue;
                }
                if (stats) {
                    g_hot_stats.tot[il]++;
                    g_hot_stats.hit[il] += tbl[e] != ls->pub.n_slots;
                }
                // [TAG_FN_MOE_HOT_ADAPT] one sighting per expert and step
                if (mc->adapt && !ls->cur_seen.empty() && !ls->cur_seen[e]) {
                    ls->cur_seen[e] = 1;
                    ls->cur_ids.push_back(e);
                }
            }
        }
    }
    if (!stats || il != g_hot_stats.first_il) {
        return;
    }
    if (++mc->hot_steps % 256 != 0) {
        return;
    }
    uint64_t h = 0, t = 0;
    double lo = 1.0, hi = 0.0;
    for (size_t i = 0; i < g_hot_stats.hit.size(); ++i) {
        if (g_hot_stats.tot[i] == 0) {
            continue;
        }
        h += g_hot_stats.hit[i];
        t += g_hot_stats.tot[i];
        const double r = (double) g_hot_stats.hit[i] / g_hot_stats.tot[i];
        lo = std::min(lo, r);
        hi = std::max(hi, r);
    }
    LLAMA_LOG_INFO("moe-hot: %" PRIu64 " decode steps, hit rate %.3f (host layers only; layer min %.3f max %.3f)%s\n",
            mc->hot_steps, t ? (double) h / t : 0.0, lo, hi, mc->adapt ? " adaptive" : "");
    if (mc->adapt) {
        const uint64_t sn = mc->st_step_n.load(std::memory_order_relaxed);
        LLAMA_LOG_INFO("moe-hot: adaptive: %" PRIu64 " experts admitted, %" PRIu64 " verify mismatches%s, step host time %.0f us%s\n",
                mc->ad_admitted, mc->ad_bad,
                mc->dc_on ? format(", %" PRIu64 " decay passes, %" PRIu64 " swaps queued", mc->dc_passes, mc->dc_swaps).c_str() : "",
                sn ? (double) mc->st_step_us.load(std::memory_order_relaxed)/sn : 0.0,
                mc->up_pipe || mc->up_rate > 0 || mc->sd_node || mc->dc_burst > 0 ? format(" [TAG_FN_L3_POLICY] uploaded %.0f MiB, %" PRIu64
                        " bucket waits, %" PRIu64 " prompt tokens seeded, %" PRIu64 " burst swaps", mc->up_bytes.load()/1048576.0,
                        mc->up_waits.load(), mc->sd_tokens.load(), mc->dc_burst_n).c_str() : "");
    }
}

// moeprof v1 (src/llama-moetrace.cpp, tools/moe-trace/make_profile.py): "<section> <il> c0 c1 ..."
bool hot_read_profile(const char * path, const std::string & want, std::map<int, std::vector<double>> & out, std::string & used) {
    std::ifstream f(path);
    if (!f) {
        LLAMA_LOG_WARN("moe-hot: cannot read %s\n", path);
        return false;
    }
    std::string line;
    std::getline(f, line);
    if (line.rfind("moeprof v1", 0) != 0) {
        LLAMA_LOG_WARN("moe-hot: %s is not a moeprof v1 file\n", path);
        return false;
    }
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
    for (const std::string & s : { want, std::string("decode_union"), std::string("decode_tokens"), std::string("prefill_tokens") }) {
        auto it = sec.find(s);
        if (it != sec.end() && !it->second.empty()) {
            out  = it->second;
            used = s;
            return true;
        }
    }
    LLAMA_LOG_WARN("moe-hot: %s has no usable section\n", path);
    return false;
}

double hot_type_cost(ggml_type t, const std::map<std::string, double> & cost) {
    auto it = cost.find(ggml_type_name(t));
    return it != cost.end() ? it->second : 1.0;
}

// [TAG_FN_MOE_HOT_ADAPT] -------------------------------------------------------------------------------------------

// one table entry: the host table directly (CPU memory, read only by the next graph), the device table through the
// mirror that step() uploads once
void hot_adapt_set_entry(moe_cache * mc, layer_state & ls, int32_t expert, int32_t v) {
    ((int32_t *) ls.pub.host_table->data)[expert] = v;
    mc->tbl_mirror[ls.tbl_off + expert] = v;
    mc->tbl_dirty = true;
}

// [TAG_FN_L3_POLICY_UPLOAD] the per-step byte bucket: takes `bytes`, waiting until it holds them when `wait` (else false
// without taking anything); a lend that waits for the worker (and the process stop) lets it through, so the bucket never
// holds a batch the prompt path waits for
bool hot_rate_take(moe_cache * mc, size_t bytes, bool wait = true) {
    if (mc->up_rate == 0) {
        return true;
    }
    std::unique_lock<std::mutex> lk(mc->wmtx);
    if (mc->up_bucket < bytes && !mc->up_drain && !mc->stop) {
        if (!wait) {
            return false;
        }
        mc->up_waits.fetch_add(1, std::memory_order_relaxed);
        mc->wcv.wait(lk, [&]() { return mc->up_bucket >= bytes || mc->up_drain || mc->stop; });
    }
    mc->up_bucket = mc->up_bucket > bytes ? mc->up_bucket - bytes : 0;
    return true;
}

// a slice's source and destination are inside their tensors (upload_slice's check)
bool hot_slice_ok(const ggml_tensor * dst_c, const ggml_tensor * src, int32_t expert, int32_t slot) {
    const size_t sz = src->nb[2];
    if (slot < 0 || expert < 0 || (size_t) slot*dst_c->nb[2] + sz > ggml_nbytes(dst_c) || (size_t) expert*sz + sz > ggml_nbytes(src)) {
        LLAMA_LOG_ERROR("moe-hot: bad upload %s <- %s expert=%d slot=%d - skipped\n", dst_c->name, src->name, expert, slot);
        return false;
    }
    return true;
}

// [TAG_FN_L3_POLICY_UPLOAD] LLAMA_MOE_HOT_UP_PIPE: the batch through two small pinned halves - the copy threads fill one
// while the upload stream reads the other (an event per half orders the reuse) - so the staging stays cache resident and
// the memcpy overlaps the DMA. Returns after every copy of the batch has landed.
void hot_upload_pipe(moe_cache * mc, std::vector<upload_job> & batch) {
    struct staged { ggml_tensor * dst; size_t off; size_t size; size_t stage_off; };
    std::vector<staged> st;
    int    h    = 0;
    size_t used = 0;
    auto half = [&](int i) { return mc->up_stage_p + (size_t) i*mc->up_half; };
    auto wait_half = [&](int i) {
        if (mc->up_ev_rec[i]) {
            ggml_backend_event_synchronize(mc->up_ev[i]); // the copies of its previous content have read it
            mc->up_ev_rec[i] = false;
        }
    };
    auto flush = [&]() {
        if (st.empty()) {
            return;
        }
        for (const auto & x : st) {
            ggml_backend_tensor_set_async(mc->up_backend, x.dst, half(h) + x.stage_off, x.off, x.size);
        }
        ggml_backend_event_record(mc->up_ev[h], mc->up_backend);
        mc->up_ev_rec[h] = true;
        st.clear();
        used = 0;
        h ^= 1;
        wait_half(h);
    };
    wait_half(h);
    const int64_t t0 = ggml_time_us();
    size_t batch_bytes = 0;
    for (auto & j : batch) {
        layer_state & ls = mc->layers[j.layer_idx];
        ggml_tensor *       dsts[3] = { ls.pub.up_c,   ls.pub.gate_c,   ls.pub.down_c   };
        const ggml_tensor * srcs[3] = { ls.pub.up_src, ls.pub.gate_src, ls.pub.down_src };
        for (int k = 0; k < 3 && !j.failed; ++k) {
            j.failed = !hot_slice_ok(dsts[k], srcs[k], j.expert, j.slot);
        }
        if (j.failed) {
            continue;
        }
        for (int k = 0; k < 3; ++k) {
            const size_t sz = srcs[k]->nb[2];
            batch_bytes += sz;
            if (!hot_rate_take(mc, sz, false)) {
                flush(); // what is staged uploads while the bucket refills
                hot_rate_take(mc, sz, true);
            }
            if (sz > mc->up_half) {
                flush();
                upload_slice(dsts[k], srcs[k], j.expert, j.slot);
                mc->up_bytes.fetch_add(sz, std::memory_order_relaxed);
                continue;
            }
            if (used + sz > mc->up_half) {
                flush();
            }
            memcpy(half(h) + used, (const char *) srcs[k]->data + (size_t) j.expert*sz, sz);
            st.push_back({ dsts[k], (size_t) j.slot*dsts[k]->nb[2], sz, used });
            used += sz;
            mc->up_bytes.fetch_add(sz, std::memory_order_relaxed);
        }
    }
    flush();
    ggml_backend_synchronize(mc->up_backend); // every copy of the batch has landed
    if (mc->up_batches++ < 2) {
        const double ms = (ggml_time_us() - t0)/1000.0;
        LLAMA_LOG_INFO("moe-hot: [TAG_FN_L3_POLICY_UPLOAD] batch %llu: %zu experts, %.1f MiB through the pinned halves in %.1f ms "
                "(%.1f GiB/s, incl. bucket waits)\n", (unsigned long long) mc->up_batches, batch.size(), batch_bytes/1048576.0, ms,
                ms > 0 ? batch_bytes/1073741824.0/(ms/1000.0) : 0.0);
    }
}

// the batch through the one large pinned staging buffer: fill it, upload, wait, repeat
void hot_upload_staged(moe_cache * mc, const std::vector<upload_job> & batch) {
    struct staged { ggml_tensor * dst; size_t off; size_t size; size_t stage_off; };
    std::vector<staged> st;
    size_t used = 0;
    auto flush = [&]() {
        for (const auto & x : st) {
            ggml_backend_tensor_set_async(mc->up_backend, x.dst, mc->stage_ptr + x.stage_off, x.off, x.size);
        }
        ggml_backend_synchronize(mc->up_backend);
        st.clear();
        used = 0;
    };
    for (const auto & j : batch) {
        layer_state & ls = mc->layers[j.layer_idx];
        ggml_tensor *       dsts[3] = { ls.pub.up_c,   ls.pub.gate_c,   ls.pub.down_c   };
        const ggml_tensor * srcs[3] = { ls.pub.up_src, ls.pub.gate_src, ls.pub.down_src };
        for (int k = 0; k < 3; ++k) {
            const size_t sz = srcs[k]->nb[2];
            hot_rate_take(mc, sz); // [TAG_FN_L3_POLICY_UPLOAD] a no-op without LLAMA_MOE_HOT_UP_MIB_STEP
            mc->up_bytes.fetch_add(sz, std::memory_order_relaxed);
            if (sz > mc->stage_size) {
                upload_slice(dsts[k], srcs[k], j.expert, j.slot);
                continue;
            }
            if (used + sz > mc->stage_size) {
                flush();
            }
            memcpy(mc->stage_ptr + used, (const char *) srcs[k]->data + (size_t) j.expert*sz, sz);
            st.push_back({ dsts[k], (size_t) j.slot*dsts[k]->nb[2], sz, used });
            used += sz;
        }
    }
    flush();
}

// uploads the queued slices: mmap -> pinned staging -> the upload backend's own stream, then marks them done
void hot_adapt_worker(moe_cache * mc) {
    std::vector<upload_job> batch;
    for (;;) {
        {
            std::unique_lock<std::mutex> lk(mc->wmtx);
            mc->wcv.wait(lk, [mc]() { return mc->stop || !mc->todo.empty(); });
            if (mc->stop) {
                return;
            }
            size_t n = mc->todo.size();
            if (mc->up_rate > 0 && !mc->up_drain) {
                // [TAG_FN_L3_POLICY_UPLOAD] only what the bucket holds now (at least one expert): each part is published
                // as soon as it has landed, instead of the whole pass at the end
                size_t b = 0;
                n = 0;
                for (const auto & j : mc->todo) {
                    const layer_state & ls = mc->layers[j.layer_idx];
                    const size_t jb = ls.pub.up_src->nb[2] + ls.pub.gate_src->nb[2] + ls.pub.down_src->nb[2];
                    if (n > 0 && b + jb > mc->up_bucket) {
                        break;
                    }
                    b += jb;
                    n++;
                }
            }
            batch.assign(mc->todo.begin(), mc->todo.begin() + (std::ptrdiff_t) n);
            mc->todo.erase(mc->todo.begin(), mc->todo.begin() + (std::ptrdiff_t) n);
            mc->worker_busy = true; // [TAG_FN_R1_PFS_LEND]
        }
        if (mc->up_pipe) {
            hot_upload_pipe(mc, batch); // [TAG_FN_L3_POLICY_UPLOAD]
        } else {
            hot_upload_staged(mc, batch);
        }
        std::lock_guard<std::mutex> lk(mc->wmtx);
        for (auto j : batch) {
            j.done = true;
            mc->done.push_back(j);
        }
        mc->worker_busy = false; // [TAG_FN_R1_PFS_LEND]
        mc->wcv_idle.notify_all();
    }
}

// LLAMA_MOE_HOT_VERIFY: one resident slot read back and compared with its source bytes
void hot_adapt_verify(moe_cache * mc) {
    static uint64_t pick = 0;
    for (size_t tries = 0; tries < mc->layers.size(); ++tries) {
        layer_state * lp = &mc->layers[(pick++) % mc->layers.size()];
        if (lp->pub.n_slots == 0 || lp->lent) { // [TAG_FN_R1_PFS_LEND] a lent layer's slots hold the stream's bytes
            continue;
        }
        const int32_t s = (int32_t) ((pick*2654435761u) % (uint64_t) lp->pub.n_slots);
        if (lp->pool >= 0) { // [TAG_FN_L3_POLICY_POOL] the layer that holds this pool slot
            const int32_t o = mc->pools[lp->pool].slot_layer[s];
            if (o < 0) {
                continue;
            }
            lp = &mc->layers[o];
        }
        layer_state & ls = *lp;
        const int32_t e = ls.slot_expert[s];
        if (e < 0 || ls.slot_in_flight[s]) {
            continue;
        }
        ggml_tensor *       dsts[3] = { ls.pub.up_c,   ls.pub.gate_c,   ls.pub.down_c   };
        const ggml_tensor * srcs[3] = { ls.pub.up_src, ls.pub.gate_src, ls.pub.down_src };
        std::vector<uint8_t> buf;
        bool same = true;
        for (int k = 0; k < 3 && same; ++k) {
            const size_t sz = srcs[k]->nb[2];
            buf.resize(sz);
            ggml_backend_tensor_get(dsts[k], buf.data(), (size_t) s*dsts[k]->nb[2], sz);
            same = memcmp(buf.data(), (const char *) srcs[k]->data + (size_t) e*sz, sz) == 0;
        }
        if (!same) {
            mc->ad_bad++;
            LLAMA_LOG_ERROR("moe-hot: VERIFY MISMATCH blk.%d slot %d expert %d - evicted, the CPU computes it again\n",
                    ls.pub.il, s, e);
            ls.expert_slot[e] = -1;
            ls.slot_expert[s] = -1;
            hot_adapt_set_entry(mc, ls, e, ls.pub.n_slots);
            if (ls.pool >= 0) {
                mc->pools[ls.pool].slot_layer[s] = -1; // [TAG_FN_L3_POLICY_POOL]
            }
        }
        return;
    }
}

// [TAG_FN_R4_ADAPT_DECAY] the learned set as a moeprof v1 routing profile: resident experts first (1e9 + count), then
// the routing counts (decayed counts, or the window's), the static profile breaking ties. LLAMA_MOE_HOT_PROFILE=<it>
// starts the next run from it.
// [TAG_FN_R4_REVIEW] Printed with 17 significant digits: "%.6g" wrote every resident expert as 1e+09, so the order among
// them (and the 1e-6 tie-break) was lost, and a next run with fewer slots took its residents by layer order - the greedy
// pick of hot_init runs over all layers - instead of by count.
void hot_adapt_save(moe_cache * mc) {
    if (mc->save_path.empty() || mc->layers.empty()) {
        return;
    }
    const std::string tmp = mc->save_path + ".tmp";
    FILE * f = fopen(tmp.c_str(), "w");
    if (!f) {
        LLAMA_LOG_WARN("moe-hot: cannot write %s\n", tmp.c_str());
        return;
    }
    int max_il = 0;
    for (const auto & ls : mc->layers) {
        max_il = std::max(max_il, ls.pub.il);
    }
    const int64_t n_exp = mc->layers.front().pub.host_table->ne[1];
    fprintf(f, "moeprof v1 n_layer=%d n_expert=%lld tokens=%llu source=moe-hot-save policy=%s\n", max_il + 1,
            (long long) n_exp, (unsigned long long) mc->ad_steps, mc->dc_on ? "decay" : "window");
    for (const auto & ls : mc->layers) {
        fprintf(f, "decode_union %d", ls.pub.il);
        for (int64_t e = 0; e < n_exp; ++e) {
            double v = mc->dc_on && !ls.dcnt.empty() ? ls.dcnt[e] : (!ls.win_cnt.empty() ? ls.win_cnt[e] : 0.0);
            if (e < (int64_t) ls.expert_slot.size() && ls.expert_slot[e] >= 0) {
                v += 1e9;
            }
            if (e < (int64_t) ls.prof.size()) {
                v += 1e-6*ls.prof[e];
            }
            fprintf(f, " %.17g", v);
        }
        fprintf(f, "\n");
    }
    const bool ok = fclose(f) == 0;
    std::remove(mc->save_path.c_str());
    if (!ok || std::rename(tmp.c_str(), mc->save_path.c_str()) != 0) {
        LLAMA_LOG_WARN("moe-hot: cannot replace %s\n", mc->save_path.c_str());
    }
}

// [TAG_FN_R4_ADAPT_DECAY] the decayed policy's part of the step: fold the prompt seeds, then every `every` decode steps
// one pass (llama_moe_decay_pairs over all layers, best gain first, under the upload budget) and the decay
void hot_adapt_decay(moe_cache * mc, bool decode_step) {
    bool folded = false;
    for (auto & ls : mc->layers) {
        if (!ls.dseed_any) {
            continue;
        }
        if (mc->sd_norm > 0.0f) {
            // [TAG_FN_L3_POLICY_SEED] LLAMA_MOE_HOT_SEED_NORM: the whole prompt (all its ubatches) at its first decode step,
            // as sd_norm decode steps of the share of its tokens that route to each expert, whatever its length
            if (!decode_step) {
                continue;
            }
            llama_moe_decay_fold_seed(ls.dcnt, ls.dseed, ls.dseed_tok > 0.0f ? mc->dc.seed*mc->sd_norm/ls.dseed_tok : 0.0f);
        } else {
            llama_moe_decay_fold_seed(ls.dcnt, ls.dseed, mc->dc.seed);
        }
        ls.dseed_any = false;
        ls.dseed_tok = 0.0f;
        folded = true;
    }
    // [TAG_FN_L3_POLICY_SEED] the node's prompt tokens, logged for the first folds
    const uint64_t sd_now = mc->sd_tokens.load(std::memory_order_relaxed);
    if (folded && mc->sd_node && sd_now > mc->sd_folded) {
        if (mc->sd_folds++ < 3 || mc->sd_folds % 64 == 0) {
            LLAMA_LOG_INFO("moe-hot: [TAG_FN_L3_POLICY_SEED] the routing of %llu prompt tokens (graph node) folded into the "
                    "counts x %.3f (fold %llu, %llu tokens so far)\n", (unsigned long long) (sd_now - mc->sd_folded),
                    mc->dc.seed, (unsigned long long) mc->sd_folds, (unsigned long long) sd_now);
        }
        mc->sd_folded = sd_now;
    }
    if (!decode_step || mc->ad_steps % (uint64_t) std::max(1, mc->dc.every) != 0) {
        return;
    }
    mc->dc_passes++;
    std::vector<llama_moe_decay_swap> swaps;
    std::vector<uint8_t> slot_busy;
    std::vector<uint8_t> expert_busy;
    for (int li = 0; li < (int) mc->layers.size(); ++li) {
        const layer_state & ls = mc->layers[li];
        if (ls.pub.n_slots == 0 || ls.dcnt.empty() || ls.lent) { // [TAG_FN_R1_PFS_LEND] no upload into a lent layer
            continue;
        }
        if (ls.pool >= 0) {
            continue; // [TAG_FN_L3_POLICY_POOL] below, per pool
        }
        slot_busy.assign(ls.pub.n_slots, 0);
        for (int32_t s = 0; s < ls.pub.n_slots; ++s) {
            slot_busy[s] = ls.slot_in_flight[s] ? 1 : 0;
        }
        expert_busy.assign(ls.dcnt.size(), 0);
        for (size_t e = 0; e < ls.dcnt.size(); ++e) {
            expert_busy[e] = ls.expert_slot[e] >= 0 || ls.expert_in_flight[e] ? 1 : 0;
        }
        llama_moe_decay_pairs(li, ls.dcnt, ls.slot_expert, slot_busy, expert_busy, mc->dc, swaps);
    }
    // [TAG_FN_L3_POLICY_POOL] the candidates and residents of all layers of a pool compete for its slots
    for (const auto & P : mc->pools) {
        std::vector<const std::vector<float> *>   cnts;
        std::vector<std::vector<uint8_t>>         busy_own(P.layers.size());
        std::vector<const std::vector<uint8_t> *> busy;
        bool skip = false;
        for (size_t k = 0; k < P.layers.size() && !skip; ++k) {
            const layer_state & ls = mc->layers[P.layers[k]];
            skip = ls.lent || ls.dcnt.empty();
            busy_own[k].assign(ls.dcnt.size(), 0);
            for (size_t e = 0; e < ls.dcnt.size(); ++e) {
                busy_own[k][e] = ls.expert_slot[e] >= 0 || ls.expert_in_flight[e] ? 1 : 0;
            }
            cnts.push_back(&ls.dcnt);
            busy.push_back(&busy_own[k]);
        }
        if (skip) {
            continue;
        }
        std::vector<int32_t> sl(P.n_slots, -1);
        std::vector<int32_t> se(P.n_slots, -1);
        std::vector<uint8_t> sb(P.n_slots, 0);
        for (int32_t s = 0; s < P.n_slots; ++s) {
            const int32_t o = P.slot_layer[s];
            if (o < 0) {
                continue;
            }
            const layer_state & ow = mc->layers[o];
            sl[s] = ow.pool_k;
            se[s] = ow.slot_expert[s];
            sb[s] = ow.slot_in_flight[s] ? 1 : 0;
        }
        std::vector<llama_moe_decay_swap> ps;
        llama_moe_decay_pairs_pool(cnts, busy, sl, se, sb, mc->dc, ps);
        for (auto w : ps) {
            w.layer        = (int) P.layers[w.layer];
            w.victim_layer = w.victim_layer >= 0 ? (int) P.layers[w.victim_layer] : -1;
            swaps.push_back(w);
        }
    }
    llama_moe_decay_order(swaps);

    // [TAG_FN_L3_POLICY_UPLOAD] with a rate limit a pass queues at most what the bucket gives the worker until the next
    // pass, minus what is still queued: evicted slots must not wait empty behind a growing queue
    // [TAG_FN_L3_POLICY_BURST] LLAMA_MOE_HOT_BURST_MIB: past the pass budget, strong pairs (a free slot, or a candidate over
    // LLAMA_MOE_HOT_BURST_RATIO x its victim) take up to that many bytes more, so a shifted working set refills fast
    // while the steady state keeps the small budget
    size_t pass_bytes  = mc->dc_bytes;
    size_t total_bytes = mc->dc_bytes + mc->dc_burst;
    if (mc->up_rate > 0) {
        size_t queued_b = 0;
        {
            std::lock_guard<std::mutex> lk(mc->wmtx);
            for (const auto & j : mc->todo) {
                const layer_state & ls = mc->layers[j.layer_idx];
                queued_b += ls.pub.up_src->nb[2] + ls.pub.gate_src->nb[2] + ls.pub.down_src->nb[2];
            }
        }
        const size_t cap = mc->up_rate*(size_t) std::max(1, mc->dc.every);
        pass_bytes  = std::min(pass_bytes,  cap > queued_b ? cap - queued_b : 0);
        total_bytes = std::min(total_bytes, cap > queued_b ? cap - queued_b : 0);
    }

    auto bytes_of = [mc](const llama_moe_decay_swap & w) {
        const layer_state & l = mc->layers[w.layer];
        return l.pub.up_src->nb[2] + l.pub.gate_src->nb[2] + l.pub.down_src->nb[2];
    };
    // [TAG_FN_L3_POLICY_BURST] strong: a free slot, or a candidate over LLAMA_MOE_HOT_BURST_RATIO x its victim
    auto strong = [mc](const llama_moe_decay_swap & w) {
        const layer_state & l  = mc->layers[w.layer];
        const layer_state & vl = w.victim_layer >= 0 ? mc->layers[w.victim_layer] : l;
        return w.victim < 0 || l.dcnt[w.expert] > mc->dc_burst_ratio*vl.dcnt[w.victim];
    };
    std::vector<std::pair<size_t, bool>> sel;
    llama_moe_decay_select(swaps, pass_bytes, total_bytes, bytes_of, strong, sel);

    size_t bytes = 0;
    size_t n_burst = 0;
    bool queued = false;
    for (const auto & [si, over] : sel) {
        const llama_moe_decay_swap & w = swaps[si];
        layer_state & ls = mc->layers[w.layer];
        const size_t b = bytes_of(w);
        n_burst += over ? 1 : 0;
        // evict first: the victim leaves both tables now, so no later graph reads the slot being overwritten
        // [TAG_FN_L3_POLICY_POOL] in a pool the victim may belong to another layer of the class
        layer_state & vs = w.victim_layer >= 0 ? mc->layers[w.victim_layer] : ls;
        if (w.victim >= 0) {
            vs.expert_slot[w.victim] = -1;
            vs.slot_expert[w.slot]   = -1;
            hot_adapt_set_entry(mc, vs, w.victim, vs.pub.n_slots);
        }
        ls.slot_expert[w.slot]       = -1;
        ls.slot_in_flight[w.slot]    = true;
        ls.expert_in_flight[w.expert] = true;
        if (ls.pool >= 0) {
            mc->pools[ls.pool].slot_layer[w.slot] = w.layer;
        }
        {
            std::lock_guard<std::mutex> lk(mc->wmtx);
            upload_job j;
            j.layer_idx = (size_t) w.layer;
            j.expert    = w.expert;
            j.slot      = w.slot;
            mc->todo.push_back(j);
        }
        bytes += b;
        queued = true;
        mc->dc_swaps++;
    }
    if (queued) {
        mc->wcv.notify_one();
    }
    if (n_burst > 0) {
        mc->dc_burst_n += n_burst;
        if (mc->dc_burst_logs++ < 3) {
            LLAMA_LOG_INFO("moe-hot: [TAG_FN_L3_POLICY_BURST] pass %" PRIu64 ": %zu of its swaps on the burst budget, %.0f MiB "
                    "queued\n", mc->dc_passes, n_burst, bytes/1048576.0);
        }
    }
    for (auto & ls : mc->layers) {
        llama_moe_decay_apply(ls.dcnt, mc->dc.decay);
    }
}

// finished uploads enter the bookkeeping and both tables. [TAG_FN_R1_PFS_LEND] a lent layer keeps the expert in
// slot_expert (llama_moe_hot_unlend uploads it again), but its tables stay "not hot" until then
void hot_adapt_publish(moe_cache * mc) {
    std::lock_guard<std::mutex> lk(mc->wmtx);
    for (const auto & j : mc->done) {
        layer_state & ls = mc->layers[j.layer_idx];
        if (j.failed) { // [TAG_FN_L3_POLICY_UPLOAD] nothing was copied: the slot is free again, the expert stays cold
            ls.slot_in_flight[j.slot]     = false;
            ls.expert_in_flight[j.expert] = false;
            if (ls.pool >= 0) {
                mc->pools[ls.pool].slot_layer[j.slot] = -1;
            }
            continue;
        }
        ls.slot_expert[j.slot]        = j.expert;
        ls.expert_slot[j.expert]      = j.slot;
        ls.slot_in_flight[j.slot]     = false;
        ls.expert_in_flight[j.expert] = false;
        hot_adapt_set_entry(mc, ls, j.expert, ls.lent ? ls.pub.n_slots : j.slot);
        mc->ad_admitted++;
    }
    mc->done.clear();
}

// the owning context's step boundary, after its compute has been synchronized: publish finished uploads, roll the
// windows, then evict and queue new uploads (byte-capped). Nothing here runs while a graph of this context runs.
void hot_adapt_step_impl(moe_cache * mc) {
    hot_adapt_publish(mc);

    bool any = false;
    for (const auto & ls : mc->layers) {
        any = any || !ls.cur_ids.empty();
    }
    // [TAG_FN_L3_POLICY_UPLOAD] every decode step gives the upload worker its bytes (at most two steps' worth kept)
    if (any && mc->up_rate > 0) {
        std::lock_guard<std::mutex> lk(mc->wmtx);
        mc->up_bucket = std::min(mc->up_cap, mc->up_bucket + mc->up_rate);
        mc->wcv.notify_one();
    }
    // [TAG_FN_R4_ADAPT_DECAY] the decayed policy replaces the window: every expert a decode step routed counts once (the
    // unit of the r1 trace replay, E:/turbot-gates/flashnext/test/r4/route_decay), then the pass
    if (mc->dc_on) {
        if (any) {
            mc->ad_steps++;
            for (auto & ls : mc->layers) {
                if (!ls.dcnt.empty()) {
                    llama_moe_decay_count(ls.dcnt, ls.cur_ids.data(), (int64_t) ls.cur_ids.size());
                }
                for (const int32_t e : ls.cur_ids) {
                    ls.cur_seen[e] = 0;
                }
                ls.cur_ids.clear();
            }
        }
        hot_adapt_decay(mc, any);
        if (any && mc->ad_verify > 0 && mc->ad_steps % (uint64_t) mc->ad_verify == 0) {
            hot_adapt_verify(mc);
        }
        if (any && mc->save_every > 0 && mc->ad_steps % mc->save_every == 0) {
            hot_adapt_save(mc);
        }
        if (any && mc->st_every > 0 && mc->ad_steps % mc->st_every == 0) {
            hot_state_save(mc); // [TAG_FN_L3_POLICY_STATE]
        }
        any = false; // the window part below is skipped
    }
    if (any && mc->save_every > 0 && (mc->ad_steps + 1) % mc->save_every == 0) {
        hot_adapt_save(mc); // [TAG_FN_R4_ADAPT_DECAY] the window policy's learned set
    }
    if (any) {
        mc->ad_steps++;
        const int pos = mc->ad_pos;
        mc->ad_pos = (mc->ad_pos + 1) % mc->ad_win;
        for (auto & ls : mc->layers) {
            auto & slot = ls.win_ring[pos];
            for (const int32_t e : slot) {
                if (ls.win_cnt[e] > 0) {
                    ls.win_cnt[e]--;
                }
            }
            slot.swap(ls.cur_ids);
            ls.cur_ids.clear();
            for (const int32_t e : slot) {
                ls.win_cnt[e]++;
                ls.cur_seen[e] = 0;
            }
        }

        struct cand { int li; int32_t e; uint16_t c; };
        std::vector<cand> cands;
        for (int li = 0; li < (int) mc->layers.size(); ++li) {
            const layer_state & ls = mc->layers[li];
            if (ls.pub.n_slots == 0 || ls.lent || ls.pool >= 0) { // [TAG_FN_R1_PFS_LEND] no upload into a lent layer
                continue;
            }
            for (const int32_t e : ls.win_ring[pos]) {
                if (ls.expert_slot[e] < 0 && !ls.expert_in_flight[e] && ls.win_cnt[e] >= mc->ad_min) {
                    cands.push_back({ li, e, ls.win_cnt[e] });
                }
            }
        }
        std::stable_sort(cands.begin(), cands.end(), [](const cand & a, const cand & b) { return a.c > b.c; });

        size_t bytes = 0;
        bool queued = false;
        for (const auto & c : cands) {
            layer_state & ls = mc->layers[c.li];
            const size_t b = ls.pub.up_src->nb[2] + ls.pub.gate_src->nb[2] + ls.pub.down_src->nb[2];
            if (bytes + b > mc->ad_bytes) {
                break;
            }
            // victim: the resident slot with the fewest recent sightings, then the lowest static count
            int32_t  vs = -1;
            uint16_t vc = 0;
            double   vp = 0.0;
            for (int32_t s = 0; s < ls.pub.n_slots; ++s) {
                if (ls.slot_in_flight[s]) {
                    continue;
                }
                const int32_t  ve = ls.slot_expert[s];
                const uint16_t cc = ve >= 0 ? ls.win_cnt[ve] : 0;
                const double   pp = ve >= 0 ? ls.prof[ve] : -1.0;
                if (vs < 0 || cc < vc || (cc == vc && pp < vp)) {
                    vs = s;
                    vc = cc;
                    vp = pp;
                }
            }
            if (vs < 0 || (int) c.c <= (int) vc + mc->ad_hyst) {
                continue;
            }
            // evict first: the victim leaves both tables now, so no later graph reads the slot being overwritten
            const int32_t ve = ls.slot_expert[vs];
            if (ve >= 0) {
                ls.expert_slot[ve] = -1;
                hot_adapt_set_entry(mc, ls, ve, ls.pub.n_slots);
            }
            ls.slot_expert[vs]      = -1;
            ls.slot_in_flight[vs]   = true;
            ls.expert_in_flight[c.e] = true;
            {
                std::lock_guard<std::mutex> lk(mc->wmtx);
                upload_job j;
                j.layer_idx = (size_t) c.li;
                j.expert    = c.e;
                j.slot      = vs;
                mc->todo.push_back(j);
            }
            bytes += b;
            queued = true;
        }
        if (queued) {
            mc->wcv.notify_one();
        }

        if (mc->ad_verify > 0 && mc->ad_steps % (uint64_t) mc->ad_verify == 0) {
            hot_adapt_verify(mc);
        }
    }

    // every table change of this step reaches the device in one copy, before the next graph is launched
    if (mc->tbl_dirty) {
        ggml_backend_tensor_set(mc->tbl_all, mc->tbl_mirror.data(), 0, mc->tbl_mirror.size()*sizeof(int32_t));
        mc->tbl_dirty = false;
    }
}

// [TAG_FN_L3_POLICY] LLAMA_MOE_HOT_STATS: the step's host time (it runs between two graphs, so it adds to the step)
void hot_adapt_step(moe_cache * mc) {
    if (!mc->ad_stats) {
        hot_adapt_step_impl(mc);
        return;
    }
    const int64_t t0 = ggml_time_us();
    hot_adapt_step_impl(mc);
    mc->st_step_us.fetch_add((uint64_t) (ggml_time_us() - t0), std::memory_order_relaxed);
    mc->st_step_n.fetch_add(1, std::memory_order_relaxed);
}

void hot_adapt_init(moe_cache * mc, const llama_model & model, const void * owner, ggml_backend_dev_t dev) {
    if (!owner || !dev || !mc->tbl_all) {
        LLAMA_LOG_WARN("moe-hot: LLAMA_MOE_HOT_ADAPT needs one device and equal expert counts - static hot set only\n");
        return;
    }
    if (const char * e = llama_fn_env(model, "LLAMA_MOE_HOT_ADMIT")) { // N/W
        int n = 0, w = 0;
        if (sscanf(e, "%d/%d", &n, &w) == 2 && n >= 1 && w >= n && w <= 256) {
            mc->ad_min = n;
            mc->ad_win = w;
        }
    }
    if (const char * e = llama_fn_env(model, "LLAMA_MOE_HOT_ADAPT_MIB")) {
        mc->ad_bytes = (size_t) std::max(1, atoi(e)) << 20;
    }
    if (const char * e = llama_fn_env(model, "LLAMA_MOE_HOT_HYST")) {
        mc->ad_hyst = std::max(0, atoi(e));
    }
    if (const char * e = llama_fn_env(model, "LLAMA_MOE_HOT_VERIFY")) {
        mc->ad_verify = std::max(0, atoi(e));
    }

    // [TAG_FN_L3_POLICY_*] the switches of lever round 3 act for qwen4exp only; every other model keeps the code above
    const bool l3 = model.arch == LLM_ARCH_QWEN4EXP;
    if (l3) { // [TAG_FN_L3_POLICY_UPLOAD]
        if (const char * e = llama_fn_env(model, "LLAMA_MOE_HOT_UP_PIPE")) {
            mc->up_pipe = atoi(e) != 0;
        }
        if (const char * e = llama_fn_env(model, "LLAMA_MOE_HOT_UP_STAGE_MIB")) {
            mc->up_half = (size_t) std::max(2, std::min(256, atoi(e))) << 20;
        }
        if (const char * e = llama_fn_env(model, "LLAMA_MOE_HOT_UP_MIB_STEP")) {
            mc->up_rate = (size_t) std::max(0, atoi(e)) << 20;
        }
    }

    mc->up_backend = ggml_backend_dev_init(dev, nullptr);
    if (!mc->up_backend) {
        LLAMA_LOG_WARN("moe-hot: no upload backend - static hot set only\n");
        return;
    }
    if (mc->up_pipe) { // [TAG_FN_L3_POLICY_UPLOAD] two small pinned halves instead of the one large staging buffer
        if (ggml_backend_buffer_type_t hb = ggml_backend_dev_host_buffer_type(dev)) {
            mc->up_stage = ggml_backend_buft_alloc_buffer(hb, 2*mc->up_half);
        }
        mc->up_ev[0] = mc->up_stage ? ggml_backend_event_new(dev) : nullptr;
        mc->up_ev[1] = mc->up_stage ? ggml_backend_event_new(dev) : nullptr;
        if (!mc->up_stage || !mc->up_ev[0] || !mc->up_ev[1]) {
            LLAMA_LOG_WARN("moe-hot: [TAG_FN_L3_POLICY_UPLOAD] no pinned halves or events - the one staging buffer\n");
            for (auto & ev : mc->up_ev) {
                if (ev) {
                    ggml_backend_event_free(ev);
                    ev = nullptr;
                }
            }
            if (mc->up_stage) {
                ggml_backend_buffer_free(mc->up_stage);
                mc->up_stage = nullptr;
            }
            mc->up_pipe = false;
        } else {
            mc->up_stage_p = (uint8_t *) ggml_backend_buffer_get_base(mc->up_stage);
        }
    }
    if (!mc->up_pipe) {
        mc->stage_size = std::max<size_t>(mc->ad_bytes, 16u << 20);
        if (ggml_backend_buffer_type_t hb = ggml_backend_dev_host_buffer_type(dev)) {
            mc->staging = ggml_backend_buft_alloc_buffer(hb, mc->stage_size);
        }
        if (mc->staging) {
            mc->stage_ptr = (uint8_t *) ggml_backend_buffer_get_base(mc->staging);
        } else {
            mc->stage_buf.resize(mc->stage_size); // pageable: correct, but the copies stage through the driver
            mc->stage_ptr = mc->stage_buf.data();
        }
    }
    if (mc->up_rate > 0) { // [TAG_FN_L3_POLICY_UPLOAD] two steps' worth, and never less than the largest slice
        size_t max_slice = 0;
        for (const auto & ls : mc->layers) {
            max_slice = std::max({ max_slice, ls.pub.up_src->nb[2], ls.pub.gate_src->nb[2], ls.pub.down_src->nb[2] });
        }
        mc->up_cap    = std::max(2*mc->up_rate, max_slice);
        mc->up_bucket = mc->up_cap;
    }

    const int64_t n_exp = mc->tbl_all->ne[1];
    mc->tbl_mirror.assign((size_t) ggml_nelements(mc->tbl_all), 0);
    ggml_backend_tensor_get(mc->tbl_all, mc->tbl_mirror.data(), 0, mc->tbl_mirror.size()*sizeof(int32_t));
    for (auto & ls : mc->layers) {
        ls.win_cnt.assign(n_exp, 0);
        ls.win_ring.assign(mc->ad_win, {});
        ls.cur_seen.assign(n_exp, 0);
        ls.cur_ids.clear();
    }
    // [TAG_FN_R4_ADAPT_DECAY] LLAMA_MOE_HOT_DECAY=<0..1> (unset / 0 = the window policy above)
    if (const char * e = llama_fn_env(model, "LLAMA_MOE_HOT_DECAY")) {
        const float d = (float) atof(e);
        if (d > 0.0f && d < 1.0f) {
            mc->dc_on    = true;
            mc->dc.decay = d;
            if (const char * v = llama_fn_env(model, "LLAMA_MOE_HOT_DECAY_EVERY")) { mc->dc.every = std::max(1, atoi(v)); }
            if (const char * v = llama_fn_env(model, "LLAMA_MOE_HOT_DECAY_ADMIT")) { mc->dc.admit = std::max(0.0f, (float) atof(v)); }
            if (const char * v = llama_fn_env(model, "LLAMA_MOE_HOT_DECAY_RATIO")) { mc->dc.ratio = std::max(1.0f, (float) atof(v)); }
            if (const char * v = llama_fn_env(model, "LLAMA_MOE_HOT_DECAY_HYST"))  { mc->dc.hyst  = std::max(0.0f, (float) atof(v)); }
            if (const char * v = llama_fn_env(model, "LLAMA_MOE_HOT_SEED"))        { mc->dc.seed  = std::max(0.0f, (float) atof(v)); }
            // the same upload bytes per decode step as the window policy: a pass every `every` steps moves every x the
            // step budget (the r1 trace replay: 128 MiB per 2-step pass beat the window at 64 MiB per step on code,
            // prose and chat, E:/turbot-gates/flashnext/test/r4/route_decay)
            mc->dc_bytes = mc->ad_bytes*(size_t) mc->dc.every;
            if (const char * v = llama_fn_env(model, "LLAMA_MOE_HOT_DECAY_MIB")) { mc->dc_bytes = (size_t) std::max(1, atoi(v)) << 20; }
            for (auto & ls : mc->layers) {
                ls.dcnt.assign(n_exp, 0.0f);
                ls.dseed.assign(n_exp, 0.0f);
                // the starting set counts as known: its static profile rank keeps it until the routing says otherwise
                for (int64_t x = 0; x < n_exp && x < (int64_t) ls.expert_slot.size(); ++x) {
                    if (ls.expert_slot[x] >= 0) {
                        ls.dcnt[x] = mc->dc.admit;
                    }
                }
            }
        } else if (d != 0.0f) {
            LLAMA_LOG_WARN("moe-hot: LLAMA_MOE_HOT_DECAY=%s is not in (0, 1) - the window policy\n", e);
        }
    }
    if (const char * e = llama_fn_env(model, "LLAMA_MOE_HOT_SAVE"); e && e[0]) {
        mc->save_path  = e;
        mc->save_every = 1024;
        if (const char * v = llama_fn_env(model, "LLAMA_MOE_HOT_SAVE_EVERY")) {
            mc->save_every = (uint64_t) std::max(0, atoi(v));
        }
    }
    // [TAG_FN_L3_POLICY_BURST] strong pairs past the pass budget
    if (l3 && mc->dc_on) {
        if (const char * e = llama_fn_env(model, "LLAMA_MOE_HOT_BURST_MIB")) {
            mc->dc_burst = (size_t) std::max(0, atoi(e)) << 20;
        }
        if (const char * e = llama_fn_env(model, "LLAMA_MOE_HOT_BURST_RATIO")) {
            mc->dc_burst_ratio = std::max(1.0f, (float) atof(e));
        }
    }
    // [TAG_FN_L3_POLICY_SEED] the prompt's routing from the graph node (llama_moe_hot_build_seed)
    if (l3 && mc->dc_on && mc->dc.seed > 0.0f) {
        if (const char * e = llama_fn_env(model, "LLAMA_MOE_HOT_SEED_NODE")) {
            mc->sd_node = atoi(e) != 0;
        }
        if (const char * e = llama_fn_env(model, "LLAMA_MOE_HOT_SEED_NORM")) {
            mc->sd_norm = std::max(0.0f, (float) atof(e));
        }
    }
    // [TAG_FN_L3_POLICY_STATE] the learned set per model: saved on a slot save, at the end and every N steps, loaded at
    // the start (llama_moe_hot_init)
    if (l3 && mc->dc_on) {
        const char * e = llama_fn_env(model, "LLAMA_MOE_HOT_STATE");
        if (e && e[0] && strcmp(e, "0") != 0 && strcmp(e, "off") != 0) {
            const std::string mpath = model.fn_auto ? model.fn_auto->model_path : std::string();
            mc->st_path = strcmp(e, "auto") == 0 ? (mpath.empty() ? std::string() : mpath + ".hotstate") : std::string(e);
            if (mc->st_path.empty()) {
                LLAMA_LOG_WARN("moe-hot: [TAG_FN_L3_POLICY_STATE] LLAMA_MOE_HOT_STATE=auto needs the model path (the qwen4exp "
                        "profile keeps it) - no saved state; give a file name\n");
            }
            mc->st_model = llama_moe_hotstate_name(model.name);
            if (const char * v = llama_fn_env(model, "LLAMA_MOE_HOT_STATE_EVERY")) {
                mc->st_every = (uint64_t) std::max(0, atoi(v));
            }
            if (const char * v = llama_fn_env(model, "LLAMA_MOE_HOT_STATE_LOAD")) {
                mc->st_load = atoi(v) != 0;
            }
        }
    }

    mc->owner = owner;
    mc->adapt = true;
    mc->worker = std::thread(hot_adapt_worker, mc);
    if (mc->sd_node || mc->sd_norm > 0.0f || mc->up_pipe || mc->up_rate > 0 || !mc->st_path.empty() || mc->dc_burst > 0) {
        const std::string up = mc->up_pipe ? format("2 x %zu MiB pinned halves", mc->up_half >> 20) : std::string("one staging buffer");
        const std::string rt = mc->up_rate > 0 ? format("<= %zu MiB per decode step (bucket %zu MiB)", mc->up_rate >> 20, mc->up_cap >> 20) :
                                                 std::string("no step limit");
        const std::string st = mc->st_path.empty() ? std::string("off") :
                format("%s (load %s, every %llu steps)", mc->st_path.c_str(), mc->st_load ? "on" : "off", (unsigned long long) mc->st_every);
        const std::string bu = mc->dc_burst > 0 ? format("+%zu MiB for pairs over %.1fx their victim", mc->dc_burst >> 20, mc->dc_burst_ratio) :
                                                  std::string("off");
        LLAMA_LOG_INFO("moe-hot: [TAG_FN_L3_POLICY] prompt seed from the graph node: %s%s; uploads: %s, %s; burst: %s; saved state: %s\n",
                mc->sd_node ? "on" : "off", mc->sd_norm > 0.0f ? format(" (a prompt counts as %.0f decode steps x %.3f)", mc->sd_norm, mc->dc.seed).c_str() : "",
                up.c_str(), rt.c_str(), bu.c_str(), st.c_str());
    }
    if (mc->dc_on) {
        LLAMA_LOG_INFO("moe-hot: adaptive [TAG_FN_R4_ADAPT_DECAY]: decayed counts x %.2f every %d steps, admit at %.2f and "
                "> %.2fx / +%.2f over the victim, prompt seed x %.3f, <= %zu MiB uploads per pass, %s staging, verify every "
                "%d steps%s%s\n", mc->dc.decay, mc->dc.every, mc->dc.admit, mc->dc.ratio, mc->dc.hyst, mc->dc.seed,
                mc->dc_bytes >> 20, mc->up_pipe ? "pipe" : (mc->staging ? "pinned" : "pageable"), mc->ad_verify,
                mc->save_path.empty() ? "" : ", saved to ", mc->save_path.c_str());
    } else {
        LLAMA_LOG_INFO("moe-hot: adaptive: admit after %d sightings in %d steps, hysteresis %d, <= %zu MiB uploads per step, "
                "%s staging, verify every %d steps%s%s\n", mc->ad_min, mc->ad_win, mc->ad_hyst, mc->ad_bytes >> 20,
                mc->staging ? "pinned" : "pageable", mc->ad_verify, mc->save_path.empty() ? "" : ", saved to ",
                mc->save_path.c_str());
    }
}

} // namespace

int llama_moe_hot_max_t() {
    const moe_cache * mc = g_cache;
    return mc && mc->hot ? mc->hot_max_t : 0;
}

namespace {

// host-resident expert layers with a device-resident router (the same rule as the LRU cache). A context of a model
// whose weights are not loaded (memory estimation) finds none.
std::vector<int> hot_host_layers(const llama_model & model) {
    std::vector<int> out;
    for (size_t il = 0; il < model.layers.size(); ++il) {
        const auto & l = model.layers[il];
        if (l.ffn_up_exps && l.ffn_gate_exps && l.ffn_down_exps && l.ffn_gate_inp && !l.ffn_gate_up_exps &&
                l.ffn_up_exps->data && l.ffn_up_exps->buffer && ggml_backend_buffer_is_host(l.ffn_up_exps->buffer) &&
                l.ffn_gate_inp->buffer && !ggml_backend_buffer_is_host(l.ffn_gate_inp->buffer)) {
            out.push_back((int) il);
        }
    }
    return out;
}

// [TAG_FN_AUTO] LLAMA_MOE_HOT_PROFILE unset, empty, off, 0 or none: no hot set
bool hot_profile_off(const char * p) {
    return !p || !p[0] || strcmp(p, "off") == 0 || strcmp(p, "0") == 0 || strcmp(p, "none") == 0;
}

} // namespace

bool llama_moe_hot_wanted(const llama_model & model) {
    return !hot_profile_off(llama_fn_env(model, "LLAMA_MOE_HOT_PROFILE"));
}

bool llama_moe_hot_fit_wanted(const llama_model & model) {
    if (!llama_moe_hot_wanted(model)) {
        return false;
    }
    const char * mib = llama_fn_env(model, "LLAMA_MOE_HOT_MIB");
    const char * fit = llama_fn_env(model, "LLAMA_MOE_HOT_FIT");
    return mib && strcmp(mib, "auto") == 0 && fit && atoi(fit) != 0;
}

ggml_backend_dev_t llama_moe_hot_device(const llama_model & model) {
    const std::vector<int> hl = hot_host_layers(model);
    if (hl.empty()) {
        return nullptr;
    }
    return ggml_backend_buft_get_device(ggml_backend_buffer_get_type(model.layers[hl[0]].ffn_gate_inp->buffer));
}

size_t llama_moe_hot_device_bytes() {
    const moe_cache * mc = g_cache;
    if (!mc || !mc->hot) {
        return 0;
    }
    size_t b = 0;
    std::vector<const ggml_tensor *> seen; // [TAG_FN_L3_POLICY_POOL] the layers of a pool share their tensors
    for (const auto & ls : mc->layers) {
        for (const ggml_tensor * t : { ls.pub.up_c, ls.pub.gate_c, ls.pub.down_c }) {
            if (std::find(seen.begin(), seen.end(), t) == seen.end()) {
                seen.push_back(t);
                b += ggml_nbytes(t);
            }
        }
    }
    return b;
}

bool llama_moe_hot_init(const llama_model & model, const void * owner, size_t budget_bytes) {
    // [TAG_FN_AUTO] every variable through llama_fn_env(): the environment, else the model's automatic profile
    const char * path = llama_fn_env(model, "LLAMA_MOE_HOT_PROFILE");
    if (hot_profile_off(path)) {
        return false;
    }
    std::lock_guard<std::mutex> init_lock(g_init_mtx);
    if (g_init_done) {
        return false;
    }

    const std::vector<int> host_layers = hot_host_layers(model);
    if (host_layers.empty()) {
        LLAMA_LOG_WARN("moe-hot: no host-resident expert layer (all experts on a device?) - hot set off\n");
        return false;
    }
    g_init_done = true; // one attempt per process: on any failure below the model runs without a hot set

    int max_t = 8;
    if (const char * e = llama_fn_env(model, "LLAMA_MOE_HOT_MAX_T")) {
        max_t = std::max(1, std::min(8, atoi(e)));
    }
    std::map<std::string, double> cost = { { "q5_1", 1.3 } };
    if (const char * e = llama_fn_env(model, "LLAMA_MOE_HOT_COST")) {
        std::stringstream ss(e);
        std::string kv;
        while (std::getline(ss, kv, ',')) {
            const size_t eq = kv.find('=');
            if (eq != std::string::npos) {
                cost[kv.substr(0, eq)] = atof(kv.c_str() + eq + 1);
            }
        }
    }
    const bool adapt_req = [&]() {
        const char * ad = llama_fn_env(model, "LLAMA_MOE_HOT_ADAPT");
        return ad && atoi(ad) != 0;
    }();

    // [TAG_FN_AUTO] "even": no routing profile; equal slot counts, empty at start, filled by the adaptive set
    bool even = strcmp(path, "even") == 0;
    const char * sec_env = llama_fn_env(model, "LLAMA_MOE_HOT_SECTION");
    std::map<int, std::vector<double>> prof;
    std::string sec_used;
    if (!even && !hot_read_profile(path, sec_env ? sec_env : "decode_union", prof, sec_used)) {
        if (!adapt_req) {
            return false;
        }
        LLAMA_LOG_WARN("moe-hot: the profile %s cannot be used - even slots, filled by the adaptive set\n", path);
        even = true;
        prof.clear();
    }
    if (even && !adapt_req) {
        LLAMA_LOG_WARN("moe-hot: LLAMA_MOE_HOT_PROFILE=even needs LLAMA_MOE_HOT_ADAPT=1 (only the adaptive set fills the slots) - hot set off\n");
        return false;
    }
    if (even) {
        sec_used = "none (even slots)";
        for (int il : host_layers) {
            prof[il] = std::vector<double>((size_t) model.layers[il].ffn_up_exps->ne[2], 0.0);
        }
    }
    // [TAG_FN_L3_POLICY_POOL] LLAMA_MOE_HOT_POOL=1 (qwen4exp, even slots, the decayed adaptive set): the layers of one
    // expert shape class share one slot pool instead of a fixed share each
    const bool pool_req = [&]() {
        const char * pe = llama_fn_env(model, "LLAMA_MOE_HOT_POOL");
        const char * de = llama_fn_env(model, "LLAMA_MOE_HOT_DECAY");
        const float  d  = de ? (float) atof(de) : 0.0f;
        return model.arch == LLM_ARCH_QWEN4EXP && even && adapt_req && pe && atoi(pe) != 0 && d > 0.0f && d < 1.0f;
    }();

    struct cand_layer {
        int il;
        const llama_layer * l;
        ggml_backend_buffer_type_t buft;
        size_t bytes;     // per expert, up + gate + down
        double cost;      // CPU cost per byte relative to 1.0
    };
    std::vector<cand_layer> layers;
    for (int il : host_layers) {
        const auto & l = model.layers[il];
        const auto it = prof.find(il);
        if (it == prof.end() || (int64_t) it->second.size() != l.ffn_up_exps->ne[2]) {
            continue;
        }
        cand_layer c;
        c.il    = il;
        c.l     = &l;
        c.buft  = ggml_backend_buffer_get_type(l.ffn_gate_inp->buffer);
        c.bytes = l.ffn_up_exps->nb[2] + l.ffn_gate_exps->nb[2] + l.ffn_down_exps->nb[2];
        c.cost  = (l.ffn_up_exps->nb[2]  *hot_type_cost(l.ffn_up_exps->type,   cost) +
                   l.ffn_gate_exps->nb[2]*hot_type_cost(l.ffn_gate_exps->type, cost) +
                   l.ffn_down_exps->nb[2]*hot_type_cost(l.ffn_down_exps->type, cost)) / (double) c.bytes;
        layers.push_back(c);
    }
    if (layers.empty()) {
        LLAMA_LOG_WARN("moe-hot: the profile %s has no layer that matches the host-resident experts - hot set off\n", path);
        return false;
    }

    // budget
    size_t budget = budget_bytes; // [TAG_FN_VRAM_FIT] the caller's fit, logged there
    if (budget == 0) {
        const char * mib = llama_fn_env(model, "LLAMA_MOE_HOT_MIB");
        if (mib && strcmp(mib, "auto") == 0) {
            size_t headroom = 1536;
            if (const char * e = llama_fn_env(model, "LLAMA_MOE_HOT_HEADROOM_MIB")) {
                headroom = (size_t) std::max(0, atoi(e));
            }
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(layers[0].buft);
            size_t mem_free = 0, mem_total = 0;
            if (dev) {
                ggml_backend_dev_memory(dev, &mem_free, &mem_total);
            }
            budget = mem_free > (headroom << 20) ? mem_free - (headroom << 20) : 0;
            // LLAMA_MOE_HOT_CAP_MIB: keep device use (all processes) at or below this, e.g. 28500 on a 32 GB card whose
            // decode degrades above ~28.5 GB well before it spills
            if (const char * e = llama_fn_env(model, "LLAMA_MOE_HOT_CAP_MIB"); e && atoi(e) > 0) {
                const size_t cap  = (size_t) atoi(e) << 20;
                const size_t used = mem_total > mem_free ? mem_total - mem_free : 0;
                const size_t room = cap > used + (headroom << 20) ? cap - used - (headroom << 20) : 0;
                budget = std::min(budget, room);
            }
            LLAMA_LOG_INFO("moe-hot: auto budget %.0f MiB: %.0f MiB free of %.0f, %zu MiB headroom\n", budget/1048576.0,
                    mem_free/1048576.0, mem_total/1048576.0, headroom);
        } else if (mib) {
            budget = (size_t) (std::max(0.0, atof(mib))*1048576.0);
        }
    }
    if (budget == 0) {
        LLAMA_LOG_WARN("moe-hot: the budget is 0 (LLAMA_MOE_HOT_MIB unset or 0, or no VRAM left) - hot set off\n");
        return false;
    }

    // greedy by count x cost per byte (all experts of a layer have the same size)
    struct cand_expert { double value; int li; int e; };
    std::vector<cand_expert> cands;
    for (int li = 0; li < (int) layers.size() && !even; ++li) {
        const auto & v = prof[layers[li].il];
        for (int e = 0; e < (int) v.size(); ++e) {
            if (v[e] > 0) {
                cands.push_back({ v[e]*layers[li].cost, li, e });
            }
        }
    }
    std::stable_sort(cands.begin(), cands.end(), [](const cand_expert & a, const cand_expert & b) { return a.value > b.value; });

    auto * mc = new moe_cache();
    mc->hot       = true;
    mc->hot_max_t = max_t;

    auto free_all = [&]() {
        for (auto * b : mc->bufs) { ggml_backend_buffer_free(b); }
        for (auto * c : mc->ctxs) { ggml_free(c); }
        mc->bufs.clear();
        mc->ctxs.clear();
        mc->layers.clear();
        mc->by_up_src.clear();
        mc->tbl_all = nullptr;
        mc->pools.clear(); // [TAG_FN_L3_POLICY_POOL]
    };

    // [TAG_FN_L3_POLICY_POOL] candidate layers that keep their own slot tensors even with pools: an integration point for a
    // lever that needs whole layers at the top of the device buffer (e.g. a tail of layers that gives its VRAM back)
    std::vector<bool> pool_excl(layers.size(), false);

    for (int attempt = 0; attempt < 32 && budget > 0; ++attempt) {
        std::vector<std::vector<int32_t>> hot_ids(layers.size());
        std::vector<int32_t>              n_slots_of(layers.size(), 0);
        size_t used = 0;
        if (even) {
            // [TAG_FN_AUTO] the same slot count in every host layer
            std::vector<size_t> bytes;
            int32_t n_exp_min = INT32_MAX;
            for (const auto & c : layers) {
                bytes.push_back(c.bytes);
                n_exp_min = std::min<int32_t>(n_exp_min, (int32_t) c.l->ffn_up_exps->ne[2]);
            }
            const int32_t n = llama_fn_even_slots(bytes, budget, n_exp_min);
            for (size_t li = 0; li < layers.size(); ++li) {
                n_slots_of[li] = n;
                used += (size_t) n*layers[li].bytes;
            }
        } else {
            for (const auto & c : cands) {
                const size_t b = layers[c.li].bytes;
                if (used + b > budget) {
                    continue;
                }
                used += b;
                hot_ids[c.li].push_back(c.e);
            }
            for (size_t li = 0; li < layers.size(); ++li) {
                std::sort(hot_ids[li].begin(), hot_ids[li].end());
                n_slots_of[li] = (int32_t) hot_ids[li].size();
            }
        }

        free_all();

        // device tensors grouped by the router's buffer type, host tables in one CPU buffer
        std::map<ggml_backend_buffer_type_t, std::vector<int>> groups;
        size_t n_hot_layers = 0;
        for (int li = 0; li < (int) layers.size(); ++li) {
            if (n_slots_of[li] > 0) {
                groups[layers[li].buft].push_back(li);
                n_hot_layers++;
            }
        }
        if (n_hot_layers == 0) {
            break;
        }
        mc->layers.reserve(n_hot_layers); // layer_state addresses must stay stable

        auto new_ctx = [&](size_t n) -> ggml_context * {
            ggml_init_params ip = { ggml_tensor_overhead()*(n + 8), nullptr, true };
            ggml_context * ctx = ggml_init(ip);
            if (ctx) {
                mc->ctxs.push_back(ctx);
            }
            return ctx;
        };
        bool ok = true;
        ggml_context * ctx_h = new_ctx(n_hot_layers);
        ok = ctx_h != nullptr;
        std::vector<std::pair<int, size_t>> state_of; // (candidate layer index, mc->layers index)
        for (auto & g : groups) {
            if (!ok) {
                break;
            }
            ggml_context * ctx_d = new_ctx(g.second.size()*5 + 1);
            if (!ctx_d) {
                ok = false;
                break;
            }
            // [TAG_FN_MOE_HOT_ADAPT] all device tables of a group in one tensor, so the adaptive set rewrites them with
            // one copy per step; each layer's dev_table is a view of its row
            const int64_t n_exp_g = layers[g.second[0]].l->ffn_up_exps->ne[2];
            bool same_n_exp = true;
            for (int li : g.second) {
                same_n_exp = same_n_exp && layers[li].l->ffn_up_exps->ne[2] == n_exp_g;
            }
            ggml_tensor * tbl_all = same_n_exp ? ggml_new_tensor_3d(ctx_d, GGML_TYPE_I32, 1, n_exp_g, (int64_t) g.second.size()) : nullptr;
            if (tbl_all) {
                ggml_format_name(tbl_all, "moe_hot_tbl_all");
            }
            // [TAG_FN_L3_POLICY_POOL] the layers of one shape class (types, shapes, slot count) share one tensor set of their
            // slot rows: k layers x (n slots + their zero slot) = one pool of k*(n + 1) - 1 slots and one zero slot
            struct pool_class {
                std::vector<int> lis;
                ggml_tensor *    t[3]    = { nullptr, nullptr, nullptr };
                int32_t          n_slots = 0;
                int              pool    = -1;
            };
            std::vector<pool_class> classes;
            std::map<int, size_t>   class_of;
            if (pool_req) {
                std::map<std::vector<int64_t>, size_t> by_key;
                for (int li : g.second) {
                    if (pool_excl[li]) {
                        continue; // keeps its own slots, created after the pools: the top of the buffer
                    }
                    const llama_layer * l = layers[li].l;
                    std::vector<int64_t> key;
                    for (const ggml_tensor * t : { l->ffn_up_exps, l->ffn_gate_exps, l->ffn_down_exps }) {
                        key.insert(key.end(), { (int64_t) t->type, t->ne[0], t->ne[1], t->ne[2] });
                    }
                    key.push_back(n_slots_of[li]);
                    auto it = by_key.find(key);
                    if (it == by_key.end()) {
                        it = by_key.emplace(key, classes.size()).first;
                        classes.emplace_back();
                    }
                    classes[it->second].lis.push_back(li);
                }
                for (size_t c = 0; c < classes.size(); ++c) {
                    pool_class & C = classes[c];
                    if (C.lis.size() < 2) {
                        continue; // a class of one keeps its own slots
                    }
                    C.n_slots = (int32_t) C.lis.size()*(n_slots_of[C.lis[0]] + 1) - 1;
                    const llama_layer * l = layers[C.lis[0]].l;
                    const ggml_tensor * src[3] = { l->ffn_up_exps, l->ffn_gate_exps, l->ffn_down_exps };
                    static const char * nm[3] = { "up", "gate", "down" };
                    for (int w = 0; w < 3; ++w) {
                        C.t[w] = ggml_new_tensor_3d(ctx_d, src[w]->type, src[w]->ne[0], src[w]->ne[1], C.n_slots + 1);
                        ggml_format_name(C.t[w], "moe_hot_pool%zu_%s", mc->pools.size(), nm[w]);
                    }
                    C.pool = (int) mc->pools.size();
                    mc->pools.emplace_back();
                    mc->pools.back().n_slots = C.n_slots;
                    for (int li : C.lis) {
                        class_of[li] = c;
                    }
                }
            }
            int64_t k = 0;
            for (int li : g.second) {
                const llama_layer * l = layers[li].l;
                const int32_t n_hot = n_slots_of[li];
                mc->layers.push_back({});
                layer_state & ls = mc->layers.back();
                state_of.push_back({ li, mc->layers.size() - 1 });
                ls.pub.il       = layers[li].il;
                ls.pub.n_slots  = n_hot;
                ls.pub.up_src   = l->ffn_up_exps;
                ls.pub.gate_src = l->ffn_gate_exps;
                ls.pub.down_src = l->ffn_down_exps;
                const ggml_tensor * u  = l->ffn_up_exps;
                const ggml_tensor * gt = l->ffn_gate_exps;
                const ggml_tensor * d  = l->ffn_down_exps;
                const auto cit = class_of.find(li);
                if (cit != class_of.end()) {
                    const pool_class & C = classes[cit->second];
                    hot_pool & P = mc->pools[C.pool];
                    ls.pub.n_slots = C.n_slots;
                    ls.pub.up_c    = C.t[0];
                    ls.pub.gate_c  = C.t[1];
                    ls.pub.down_c  = C.t[2];
                    ls.pool        = C.pool;
                    ls.pool_k      = (int) P.layers.size();
                    P.layers.push_back(mc->layers.size() - 1);
                } else {
                    ls.pub.up_c   = ggml_new_tensor_3d(ctx_d, u->type,  u->ne[0],  u->ne[1],  n_hot + 1);
                    ls.pub.gate_c = ggml_new_tensor_3d(ctx_d, gt->type, gt->ne[0], gt->ne[1], n_hot + 1);
                    ls.pub.down_c = ggml_new_tensor_3d(ctx_d, d->type,  d->ne[0],  d->ne[1],  n_hot + 1);
                    ggml_format_name(ls.pub.up_c,   "moe_hot_up.%d",   ls.pub.il);
                    ggml_format_name(ls.pub.gate_c, "moe_hot_gate.%d", ls.pub.il);
                    ggml_format_name(ls.pub.down_c, "moe_hot_down.%d", ls.pub.il);
                }
                if (tbl_all) {
                    ls.pub.dev_table = ggml_view_2d(ctx_d, tbl_all, 1, n_exp_g, tbl_all->nb[1], k*tbl_all->nb[2]);
                    ls.tbl_off       = k*n_exp_g;
                } else {
                    ls.pub.dev_table = ggml_new_tensor_2d(ctx_d, GGML_TYPE_I32, 1, u->ne[2]);
                }
                ls.pub.host_table = ggml_new_tensor_2d(ctx_h, GGML_TYPE_I32, 1, u->ne[2]);
                ggml_format_name(ls.pub.dev_table,  "moe_hot_tbl.%d",  ls.pub.il);
                ggml_format_name(ls.pub.host_table, "moe_hot_htbl.%d", ls.pub.il);
                ++k;
            }
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_d, g.first);
            if (!buf) {
                ok = false;
                break;
            }
            ggml_backend_buffer_clear(buf, 0); // every slot, so the last one of each tensor is the zero slot
            mc->bufs.push_back(buf);
            if (groups.size() == 1) {
                mc->tbl_all = tbl_all;
            }
        }
        if (ok) {
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_h, ggml_backend_cpu_buffer_type());
            ok = buf != nullptr;
            if (ok) {
                mc->bufs.push_back(buf);
            }
        }
        if (!ok) {
            LLAMA_LOG_WARN("moe-hot: allocating %.0f MiB failed, retrying with 90%% of the budget\n", used/1048576.0);
            budget = (size_t) (budget*0.9);
            continue;
        }

        // upload the hot slices from the host (mmap) copy and write both tables
        size_t vram = 0;
        const int64_t t0 = ggml_time_us();
        for (const auto & [li, si] : state_of) {
            layer_state & ls = mc->layers[si];
            const auto & ids = hot_ids[li];
            const int64_t n_exp = ls.pub.up_src->ne[2];
            std::vector<int32_t> tbl(n_exp, ls.pub.n_slots);
            for (int32_t s = 0; s < (int32_t) ids.size(); ++s) {
                upload_slice(ls.pub.up_c,   ls.pub.up_src,   ids[s], s);
                upload_slice(ls.pub.gate_c, ls.pub.gate_src, ids[s], s);
                upload_slice(ls.pub.down_c, ls.pub.down_src, ids[s], s);
                tbl[ids[s]] = s;
            }
            ggml_backend_tensor_set(ls.pub.dev_table,  tbl.data(), 0, n_exp*sizeof(int32_t));
            ggml_backend_tensor_set(ls.pub.host_table, tbl.data(), 0, n_exp*sizeof(int32_t));
            mc->by_up_src[ls.pub.up_src] = si;

            // [TAG_FN_MOE_HOT_ADAPT] bookkeeping, used only by the adaptive set; [TAG_FN_AUTO] slots past ids stay empty
            ls.slot_expert.assign(ls.pub.n_slots, -1);
            ls.expert_slot.assign(n_exp, -1);
            for (int32_t s = 0; s < (int32_t) ids.size(); ++s) {
                ls.slot_expert[s]      = ids[s];
                ls.expert_slot[ids[s]] = s;
            }
            ls.slot_in_flight.assign(ls.pub.n_slots, false);
            ls.expert_in_flight.assign(n_exp, false);
            ls.prof = prof[ls.pub.il];

            const size_t lb = ggml_nbytes(ls.pub.up_c) + ggml_nbytes(ls.pub.gate_c) + ggml_nbytes(ls.pub.down_c);
            if (ls.pool < 0 || ls.pool_k == 0) {
                vram += lb; // [TAG_FN_L3_POLICY_POOL] a pool's tensors once
            }
            if (even) {
                LLAMA_LOG_DEBUG("moe-hot: blk.%-2d n_hot %3d  empty  %7.1f MiB\n", ls.pub.il, ls.pub.n_slots, lb/1048576.0);
                continue;
            }
            const auto & v = prof[ls.pub.il];
            double hit = 0.0, all = 0.0;
            for (size_t e = 0; e < v.size(); ++e) {
                all += v[e];
                hit += tbl[e] != ls.pub.n_slots ? v[e] : 0.0;
            }
            LLAMA_LOG_INFO("moe-hot: blk.%-2d n_hot %3d  profile hit %.3f  %7.1f MiB\n", ls.pub.il, ls.pub.n_slots,
                    all > 0 ? hit/all : 0.0, lb/1048576.0);
        }
        for (auto & P : mc->pools) {
            P.slot_layer.assign(P.n_slots, -1); // [TAG_FN_L3_POLICY_POOL] every slot free
        }
        if (even && mc->pools.empty()) {
            LLAMA_LOG_INFO("moe-hot: %zu layers x %d empty slots, the adaptive set fills them from the routing\n",
                    mc->layers.size(), mc->layers.front().pub.n_slots);
        } else if (even) {
            std::string pl;
            size_t n_pooled = 0;
            for (const auto & P : mc->pools) {
                pl += format(" %zu layers x %d slots (blk.%d ...),", P.layers.size(), P.n_slots, mc->layers[P.layers[0]].pub.il);
                n_pooled += P.layers.size();
            }
            LLAMA_LOG_INFO("moe-hot: [TAG_FN_L3_POLICY_POOL] %zu slot pools:%s %zu layers with their own %d slots; the adaptive "
                    "set fills them from the routing, any layer of a pool can take any of its slots\n", mc->pools.size(),
                    pl.c_str(), mc->layers.size() - n_pooled, n_slots_of.empty() ? 0 : n_slots_of[0]);
        }
        LLAMA_LOG_INFO("moe-hot: %zu layers, %.1f MiB device memory (budget %.0f MiB), uploaded in %.1f s, section %s of %s, "
                "graphs of <= %d tokens\n", mc->layers.size(), vram/1048576.0, budget/1048576.0, (ggml_time_us() - t0)/1e6,
                sec_used.c_str(), path, max_t);

        if (const char * st = llama_fn_env(model, "LLAMA_MOE_HOT_STATS"); st && atoi(st) != 0) {
            mc->ad_stats = true;
            g_hot_stats.hit.assign(model.layers.size(), 0);
            g_hot_stats.tot.assign(model.layers.size(), 0);
            g_hot_stats.first_il = mc->layers.front().pub.il;
            for (const auto & ls : mc->layers) {
                g_hot_stats.first_il = std::min(g_hot_stats.first_il, ls.pub.il);
            }
        }

        if (adapt_req) {
            hot_adapt_init(mc, model, owner, groups.size() == 1 ? ggml_backend_buft_get_device(groups.begin()->first) : nullptr);
            if (mc->adapt) {
                hot_state_load(mc); // [TAG_FN_L3_POLICY_STATE] LLAMA_MOE_HOT_STATE: the saved residents into the empty slots
            }
        }
        if (even && !mc->adapt) {
            // [TAG_FN_AUTO] nothing would ever fill the empty slots
            LLAMA_LOG_WARN("moe-hot: the adaptive set could not start, so the even slots would stay empty - hot set off\n");
            break;
        }

        if (mc->ad_stats || mc->adapt) {
            ggml_set_moe_obs_callback(hot_obs_cb, mc);
        }
        g_cache = mc;
        return true;
    }

    free_all();
    delete mc;
    LLAMA_LOG_WARN("moe-hot: no hot set could be allocated - hot set off\n");
    return false;
}

const void * llama_moe_hot_adapt_owner() {
    const moe_cache * mc = g_cache;
    return mc && mc->adapt ? mc->owner : nullptr;
}

// [TAG_FN_R4_ADAPT_DECAY]
void llama_moe_hot_save_now(const void * owner) {
    moe_cache * mc = g_cache;
    if (mc && mc->adapt && mc->owner == owner) {
        hot_adapt_save(mc);
    }
}

// [TAG_FN_L3_POLICY_SEED] ----------------------------------------------------------------------------------------------

namespace {

// the CPU node after a prompt ubatch's experts: its ids (copied to the host by the scheduler) into the layer's prompt
// counts, which the owner's next step folds into the decayed counts x LLAMA_MOE_HOT_SEED. Runs inside the owner's graph,
// never beside its step (the step runs after the owner has synchronized its compute)
void hot_seed_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }
    moe_cache * mc = g_cache;
    const ggml_tensor * ids = dst->src[0];
    const int il = (int) (intptr_t) userdata;
    if (!mc || !mc->adapt || !mc->dc_on || !mc->sd_node || !ids || ids->type != GGML_TYPE_I32) {
        return;
    }
    for (auto & ls : mc->layers) {
        if (ls.pub.il != il) {
            continue;
        }
        if (ls.dseed.empty()) {
            return;
        }
        const int32_t n_exp = (int32_t) ls.dseed.size();
        for (int64_t t = 0; t < ids->ne[1]; ++t) {
            for (int64_t i = 0; i < ids->ne[0]; ++i) {
                const int32_t e = *(const int32_t *) ((const char *) ids->data + t*ids->nb[1] + i*ids->nb[0]);
                if (e >= 0 && e < n_exp) {
                    ls.dseed[e] += 1.0f;
                }
            }
        }
        ls.dseed_any = true;
        ls.dseed_tok += (float) ids->ne[1];
        if (&ls == &mc->layers.front()) {
            mc->sd_tokens.fetch_add((uint64_t) ids->ne[1], std::memory_order_relaxed);
        }
        return;
    }
}

} // namespace

bool llama_moe_hot_seed_node_wanted(const ggml_tensor * up_exps, int64_t n_tokens) {
    const moe_cache * mc = g_cache;
    if (!mc || !mc->hot || !mc->adapt || !mc->dc_on || !mc->sd_node || mc->dc.seed <= 0.0f || n_tokens <= mc->hot_max_t) {
        return false;
    }
    return mc->by_up_src.find(up_exps) != mc->by_up_src.end();
}

ggml_tensor * llama_moe_hot_build_seed(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * ids, int il) {
    ggml_tensor * args[1] = { ids };
    ggml_tensor * t = ggml_custom_4d(ctx, GGML_TYPE_F32, 1, 1, 1, 1, args, 1, hot_seed_op, 1, (void *) (intptr_t) il);
    ggml_format_name(t, "moe_hot_seed-%d", il);
    ggml_build_forward_expand(gf, t);
    return t;
}

// [TAG_FN_R1_PFS_LEND] ---------------------------------------------------------------------------------------------

namespace {

// both tables of one layer: every expert "not hot" (lent), or the resident ones from expert_slot. The host table
// directly, the device table in one copy (the caller has synchronized the owner's graphs), the adaptive set's mirror too
void hot_write_tables(moe_cache * mc, layer_state & ls, bool lent) {
    const int64_t n_exp = ls.pub.host_table->ne[1];
    std::vector<int32_t> tbl((size_t) n_exp, ls.pub.n_slots);
    if (!lent) {
        for (int64_t e = 0; e < n_exp && e < (int64_t) ls.expert_slot.size(); ++e) {
            if (ls.expert_slot[e] >= 0) {
                tbl[e] = ls.expert_slot[e];
            }
        }
    }
    memcpy(ls.pub.host_table->data, tbl.data(), tbl.size()*sizeof(int32_t));
    ggml_backend_tensor_set(ls.pub.dev_table, tbl.data(), 0, tbl.size()*sizeof(int32_t));
    if (!mc->tbl_mirror.empty() && ls.tbl_off + n_exp <= (int64_t) mc->tbl_mirror.size()) {
        memcpy(mc->tbl_mirror.data() + ls.tbl_off, tbl.data(), tbl.size()*sizeof(int32_t));
    }
}

// the lent range as one I8 tensor (byte-offset clears of the tails that the slot tensors' alloc sizes pad after them)
struct hot_lend_state {
    ggml_context *        ctx  = nullptr;
    ggml_tensor *         raw  = nullptr;
    ggml_backend_buffer_t buf  = nullptr;
    uint8_t *             lo   = nullptr;
    size_t                size = 0;

    // the refill: two pinned staging halves, filled by REFILL_THREADS copy threads while the other half uploads on the
    // hot set's own upload stream (mmap -> pinned -> device, the prefill stream's pattern); events order the reuse
    ggml_backend_buffer_t stage    = nullptr;
    uint8_t *             stage_p  = nullptr;
    size_t                half     = 0;
    ggml_backend_event_t  ev[2]    = { nullptr, nullptr };
    bool                  ev_rec[2] = { false, false };
};
hot_lend_state g_lend;

constexpr int    REFILL_THREADS = 8;
constexpr size_t REFILL_HALF    = 128u << 20;

struct refill_job {
    ggml_tensor * dst;
    size_t        off;  // byte offset in dst
    const void *  src;
    size_t        size;
};

// the resident experts of the lent layers whose bytes the stream may have overwritten: through the pinned halves when
// the hot set has an upload stream, else one synchronous copy per slice. Returns the bytes uploaded.
size_t hot_refill(moe_cache * mc, const std::vector<refill_job> & jobs) {
    size_t bytes = 0;
    ggml_backend_dev_t dev = mc->up_backend ? ggml_backend_get_device(mc->up_backend) : nullptr;
    if (dev && !g_lend.stage) {
        if (ggml_backend_buffer_type_t hb = ggml_backend_dev_host_buffer_type(dev)) {
            g_lend.stage = ggml_backend_buft_alloc_buffer(hb, 2*REFILL_HALF);
        }
        if (g_lend.stage) {
            g_lend.stage_p = (uint8_t *) ggml_backend_buffer_get_base(g_lend.stage);
            g_lend.half    = REFILL_HALF;
            g_lend.ev[0]   = ggml_backend_event_new(dev);
            g_lend.ev[1]   = ggml_backend_event_new(dev);
            if (!g_lend.ev[0] || !g_lend.ev[1]) {
                if (g_lend.ev[0]) { ggml_backend_event_free(g_lend.ev[0]); }
                if (g_lend.ev[1]) { ggml_backend_event_free(g_lend.ev[1]); }
                g_lend.ev[0] = g_lend.ev[1] = nullptr;
                ggml_backend_buffer_free(g_lend.stage);
                g_lend.stage   = nullptr;
                g_lend.stage_p = nullptr;
            }
        }
    }
    if (!g_lend.stage) {
        for (const auto & j : jobs) {
            ggml_backend_tensor_set(j.dst, j.src, j.off, j.size);
            bytes += j.size;
        }
        return bytes;
    }
    size_t i = 0;
    int    h = 0;
    std::vector<size_t> stage_off;
    while (i < jobs.size()) {
        // the jobs that fit into this half (a slice larger than a half goes alone, synchronously)
        size_t n = 0, used = 0;
        stage_off.clear();
        while (i + n < jobs.size() && used + jobs[i + n].size <= g_lend.half) {
            stage_off.push_back(used);
            used += jobs[i + n].size;
            n++;
        }
        if (n == 0) {
            ggml_backend_tensor_set(jobs[i].dst, jobs[i].src, jobs[i].off, jobs[i].size);
            bytes += jobs[i].size;
            i++;
            continue;
        }
        if (g_lend.ev_rec[h]) {
            ggml_backend_event_synchronize(g_lend.ev[h]); // this half's previous uploads have read it
        }
        uint8_t * base = g_lend.stage_p + (size_t) h*g_lend.half;
        std::vector<std::thread> th;
        for (int t = 0; t < REFILL_THREADS; ++t) {
            th.emplace_back([&, t]() {
                for (size_t k = (size_t) t; k < n; k += REFILL_THREADS) {
                    memcpy(base + stage_off[k], jobs[i + k].src, jobs[i + k].size);
                }
            });
        }
        for (auto & x : th) {
            x.join();
        }
        for (size_t k = 0; k < n; ++k) {
            ggml_backend_tensor_set_async(mc->up_backend, jobs[i + k].dst, base + stage_off[k], jobs[i + k].off, jobs[i + k].size);
            bytes += jobs[i + k].size;
        }
        ggml_backend_event_record(g_lend.ev[h], mc->up_backend);
        g_lend.ev_rec[h] = true;
        i += n;
        h ^= 1;
    }
    ggml_backend_synchronize(mc->up_backend);
    return bytes;
}

} // namespace

bool llama_moe_hot_lend(const void * owner, size_t bytes, ggml_backend_buffer_t * out_buf, uint8_t ** out_base) {
    moe_cache * mc = g_cache;
    if (!mc || !mc->hot || bytes == 0 || mc->layers.empty() || !mc->tbl_all || !out_buf || !out_base) {
        return false;
    }
    if (mc->adapt && owner != mc->owner) {
        return false;
    }
    if (mc->lent_any) {
        *out_buf  = g_lend.buf;
        *out_base = g_lend.lo;
        return bytes <= g_lend.size;
    }
    ggml_backend_buffer_t dbuf = mc->layers.front().pub.up_c->buffer;
    if (!dbuf || mc->tbl_all->buffer != dbuf) {
        return false;
    }
    for (const auto & ls : mc->layers) {
        if (ls.pub.up_c->buffer != dbuf || ls.pub.gate_c->buffer != dbuf || ls.pub.down_c->buffer != dbuf) {
            return false; // the slots of one device buffer only
        }
    }
    uint8_t *    base  = (uint8_t *) ggml_backend_buffer_get_base(dbuf);
    const size_t size  = ggml_backend_buffer_get_size(dbuf);
    const size_t align = std::max<size_t>(256, ggml_backend_buffer_get_alignment(dbuf));
    if (bytes + align > size) {
        return false;
    }
    const size_t start = (size - bytes) / align * align;
    uint8_t * lo = base + start;
    if ((uint8_t *) mc->tbl_all->data + ggml_nbytes(mc->tbl_all) > lo) {
        return false; // the tables must stay below the lent range
    }
    if (!g_lend.ctx) {
        ggml_init_params ip = { ggml_tensor_overhead()*2, nullptr, true };
        g_lend.ctx = ggml_init(ip);
        if (!g_lend.ctx) {
            return false;
        }
        g_lend.raw = ggml_new_tensor_1d(g_lend.ctx, GGML_TYPE_I8, (int64_t) (size - start));
        ggml_set_name(g_lend.raw, "moe_hot_lent");
        if (ggml_backend_tensor_alloc(dbuf, g_lend.raw, lo) != GGML_STATUS_SUCCESS) {
            ggml_free(g_lend.ctx);
            g_lend = {};
            return false;
        }
        g_lend.buf  = dbuf;
        g_lend.lo   = lo;
        g_lend.size = size - start;
    } else if (g_lend.buf != dbuf || g_lend.lo != lo) {
        return false; // the hot buffer never moves; another range would invalidate the stream's bank aliases
    }

    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(dbuf);
    auto overlaps = [&](const ggml_tensor * t) {
        return (uint8_t *) t->data + ggml_backend_buft_get_alloc_size(buft, t) > lo;
    };
    size_t n_lent = 0;
    for (auto & ls : mc->layers) {
        if (overlaps(ls.pub.up_c) || overlaps(ls.pub.gate_c) || overlaps(ls.pub.down_c)) {
            ls.lent = true;
            n_lent++;
        }
    }
    if (mc->adapt) {
        // no upload goes into a lent layer any more: queued ones are dropped (their slots were evicted already, so they
        // come back empty), and the worker puts its running batch down before anything overwrites a slot
        std::unique_lock<std::mutex> lk(mc->wmtx);
        for (auto it = mc->todo.begin(); it != mc->todo.end(); ) {
            layer_state & ls = mc->layers[it->layer_idx];
            if (ls.lent) {
                ls.slot_in_flight[it->slot]     = false;
                ls.expert_in_flight[it->expert] = false;
                if (ls.pool >= 0) {
                    mc->pools[ls.pool].slot_layer[it->slot] = -1; // [TAG_FN_L3_POLICY_POOL]
                }
                it = mc->todo.erase(it);
            } else {
                ++it;
            }
        }
        // [TAG_FN_L3_POLICY_UPLOAD] no decode step refills the bucket during a prompt: the worker ignores it now, and with
        // a rate limit the lend also waits for the rest of the queue (the bucket may have left jobs in it), so no upload
        // runs beside the stream
        mc->up_drain = true;
        mc->wcv.notify_all();
        mc->wcv_idle.wait(lk, [mc]() { return !mc->worker_busy && (mc->up_rate == 0 || mc->todo.empty()); });
        mc->up_drain = false;
    }
    if (mc->adapt) {
        hot_adapt_publish(mc); // the batch the worker just finished: slot_expert knows it, the tables stay "not hot"
    }
    for (auto & ls : mc->layers) {
        if (ls.lent) {
            hot_write_tables(mc, ls, true);
        }
    }
    mc->lent_any   = true;
    mc->lent_bytes = bytes;
    mc->n_lend++;
    if (mc->n_lend == 1) {
        LLAMA_LOG_INFO("moe-hot: [TAG_FN_R1_PFS_LEND] the prefill stream borrows the top %.0f MiB of the hot set's %.0f MiB "
                "while a prompt streams (%zu of %zu layers refilled after it)\n", bytes/1048576.0, size/1048576.0, n_lent,
                mc->layers.size());
    }
    *out_buf  = dbuf;
    *out_base = lo;
    return true;
}

void llama_moe_hot_unlend(const void * owner) {
    moe_cache * mc = g_cache;
    if (!mc || !mc->lent_any) {
        return;
    }
    if (mc->adapt && owner != mc->owner) {
        return;
    }
    const int64_t t0 = ggml_time_us();
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(g_lend.buf);
    // the slices of resident experts that reach into the lent range (the stream wrote only there)
    std::vector<refill_job> jobs;
    for (auto & ls : mc->layers) {
        if (!ls.lent) {
            continue;
        }
        ggml_tensor *       dsts[3] = { ls.pub.up_c,   ls.pub.gate_c,   ls.pub.down_c   };
        const ggml_tensor * srcs[3] = { ls.pub.up_src, ls.pub.gate_src, ls.pub.down_src };
        for (int32_t s = 0; s < ls.pub.n_slots; ++s) {
            const int32_t e = ls.slot_expert[s];
            if (e < 0) {
                continue; // empty: no table entry names it
            }
            for (int k = 0; k < 3; ++k) {
                const size_t sz  = srcs[k]->nb[2];
                const size_t off = (size_t) s*dsts[k]->nb[2];
                if ((uint8_t *) dsts[k]->data + off + sz <= g_lend.lo) {
                    continue; // below the lent range: never overwritten
                }
                if (off + sz > ggml_nbytes(dsts[k]) || (size_t) (e + 1)*sz > ggml_nbytes(srcs[k])) {
                    continue; // upload_slice's bound check
                }
                jobs.push_back({ dsts[k], off, (const char *) srcs[k]->data + (size_t) e*sz, sz });
            }
        }
    }
    const size_t bytes = hot_refill(mc, jobs);
    std::vector<const ggml_tensor *> zeroed; // [TAG_FN_L3_POLICY_POOL] the layers of a pool share their tensors
    for (auto & ls : mc->layers) {
        if (!ls.lent) {
            continue;
        }
        // the zero slot every miss reads, and the alloc-size tail after it (kernels may read padded rows there)
        ggml_tensor * ts[3] = { ls.pub.up_c, ls.pub.gate_c, ls.pub.down_c };
        for (ggml_tensor * t : ts) {
            if (std::find(zeroed.begin(), zeroed.end(), t) != zeroed.end()) {
                continue;
            }
            zeroed.push_back(t);
            ggml_backend_tensor_memset(t, 0, (size_t) ls.pub.n_slots*t->nb[2], t->nb[2]);
            uint8_t * tail_lo = std::max((uint8_t *) t->data + ggml_nbytes(t), g_lend.lo);
            uint8_t * tail_hi = (uint8_t *) t->data + ggml_backend_buft_get_alloc_size(buft, t);
            if (tail_hi > tail_lo) {
                ggml_backend_tensor_memset(g_lend.raw, 0, (size_t) (tail_lo - g_lend.lo), (size_t) (tail_hi - tail_lo));
            }
        }
        ls.lent = false;
        hot_write_tables(mc, ls, false);
    }
    mc->lent_any = false;
    const double ms = (ggml_time_us() - t0)/1000.0;
    mc->unlend_ms += ms;
    mc->unlend_b  += bytes;
    if (mc->n_lend <= 2 || mc->n_lend % 16 == 0) {
        LLAMA_LOG_INFO("moe-hot: [TAG_FN_R1_PFS_LEND] lend %" PRIu64 ": %.2f GiB of resident experts refilled in %.1f ms "
                "(%.1f GiB/s)\n", mc->n_lend, bytes/1073741824.0, ms, ms > 0 ? bytes/1073741824.0/(ms/1000.0) : 0.0);
    }
}

// [TAG_FN_L3_POLICY_STATE] ---------------------------------------------------------------------------------------------

namespace {

size_t hot_layer_bytes(const layer_state & ls) {
    return ls.pub.up_src->nb[2] + ls.pub.gate_src->nb[2] + ls.pub.down_src->nb[2];
}

// the file format is llama_moe_hotstate_format / _parse (llama-moe-decay.h, tests/test-moe-decay.cpp)
void hot_state_save(moe_cache * mc) {
    if (mc->st_path.empty() || mc->layers.empty() || !mc->dc_on) {
        return;
    }
    const int64_t n_exp = mc->layers.front().pub.host_table->ne[1];
    std::vector<llama_moe_hotstate_layer> out;
    size_t n_res = 0;
    for (const auto & ls : mc->layers) {
        llama_moe_hotstate_layer l;
        l.il    = ls.pub.il;
        l.bytes = hot_layer_bytes(ls);
        l.cnt   = ls.dcnt;
        for (int32_t s = 0; s < ls.pub.n_slots; ++s) {
            if (ls.slot_expert[s] >= 0) {
                l.res.push_back(ls.slot_expert[s]);
            }
        }
        std::stable_sort(l.res.begin(), l.res.end(), [&](int32_t a, int32_t b) { return ls.dcnt[a] > ls.dcnt[b]; });
        n_res += l.res.size();
        out.push_back(std::move(l));
    }
    const std::string text = llama_moe_hotstate_format(n_exp, mc->ad_steps, mc->st_model, out);
    const std::string tmp  = mc->st_path + ".tmp";
    FILE * f = fopen(tmp.c_str(), "wb");
    if (!f) {
        LLAMA_LOG_WARN("moe-hot: [TAG_FN_L3_POLICY_STATE] cannot write %s\n", tmp.c_str());
        return;
    }
    const bool ok = fwrite(text.data(), 1, text.size(), f) == text.size() && fclose(f) == 0;
    if (!ok) {
        std::remove(tmp.c_str());
        LLAMA_LOG_WARN("moe-hot: [TAG_FN_L3_POLICY_STATE] cannot write %s\n", tmp.c_str());
        return;
    }
    std::remove(mc->st_path.c_str());
    if (std::rename(tmp.c_str(), mc->st_path.c_str()) != 0) {
        LLAMA_LOG_WARN("moe-hot: [TAG_FN_L3_POLICY_STATE] cannot replace %s\n", mc->st_path.c_str());
        return;
    }
    if (mc->st_saves++ < 3) {
        LLAMA_LOG_INFO("moe-hot: [TAG_FN_L3_POLICY_STATE] saved %zu residents of %zu layers and the counts of %llu decode "
                "steps to %s\n", n_res, mc->layers.size(), (unsigned long long) mc->ad_steps, mc->st_path.c_str());
    }
}

// at the start of the adaptive set (no graph of the owner runs): the saved residents of every layer go into its empty
// slots, best first, as many as fit, and the counts come back with them. A state of another model, expert count or expert
// size is not used. Every slice has landed before a table names its slot.
void hot_state_load(moe_cache * mc) {
    if (mc->st_path.empty() || !mc->st_load || !mc->dc_on || mc->layers.empty() || !mc->tbl_all) {
        return;
    }
    std::ifstream f(mc->st_path);
    if (!f) {
        LLAMA_LOG_INFO("moe-hot: [TAG_FN_L3_POLICY_STATE] no saved state %s yet - the set starts empty\n", mc->st_path.c_str());
        return;
    }
    const int64_t t0    = ggml_time_us();
    const int64_t n_exp = mc->layers.front().pub.host_table->ne[1];
    std::vector<llama_moe_hotstate_layer> saved;
    uint64_t steps = 0;
    std::string why;
    size_t n_bad = 0;
    if (!llama_moe_hotstate_parse(f, n_exp, mc->st_model, saved, steps, why, &n_bad)) {
        LLAMA_LOG_WARN("moe-hot: [TAG_FN_L3_POLICY_STATE] %s not used: %s\n", mc->st_path.c_str(), why.c_str());
        return;
    }
    std::vector<refill_job> jobs;
    size_t n_layers = 0;
    size_t n_res    = 0;
    size_t n_skip   = n_bad;
    // [TAG_FN_L3_POLICY_POOL] (count, layer index, expert) of the saved residents of every pool
    struct pool_cand { float c; size_t li; int32_t e; };
    std::vector<std::vector<pool_cand>> pool_cands(mc->pools.size());
    auto add_jobs = [&](layer_state & ls, int32_t e, int32_t s) -> bool {
        ggml_tensor *       dsts[3] = { ls.pub.up_c,   ls.pub.gate_c,   ls.pub.down_c   };
        const ggml_tensor * srcs[3] = { ls.pub.up_src, ls.pub.gate_src, ls.pub.down_src };
        for (int k = 0; k < 3; ++k) {
            if (!hot_slice_ok(dsts[k], srcs[k], e, s)) {
                return false;
            }
        }
        for (int k = 0; k < 3; ++k) {
            const size_t sz = srcs[k]->nb[2];
            jobs.push_back({ dsts[k], (size_t) s*dsts[k]->nb[2], (const char *) srcs[k]->data + (size_t) e*sz, sz });
        }
        ls.slot_expert[s] = e;
        ls.expert_slot[e] = s;
        return true;
    };
    for (const auto & l : saved) {
        layer_state * ls = nullptr;
        for (auto & x : mc->layers) {
            if (x.pub.il == l.il) {
                ls = &x;
                break;
            }
        }
        if (!ls || ls->dcnt.size() != l.cnt.size() || hot_layer_bytes(*ls) != l.bytes || ls->lent) {
            n_skip++;
            continue;
        }
        ls->dcnt = l.cnt;
        if (ls->pool >= 0) {
            for (const int32_t e : l.res) {
                pool_cands[ls->pool].push_back({ l.cnt[e], (size_t) (ls - mc->layers.data()), e });
            }
            n_layers++;
            continue;
        }
        int32_t s = 0;
        for (const int32_t e : l.res) {
            while (s < ls->pub.n_slots && (ls->slot_expert[s] >= 0 || ls->slot_in_flight[s])) {
                s++;
            }
            if (s >= ls->pub.n_slots) {
                break;
            }
            if (ls->expert_slot[e] >= 0 || ls->expert_in_flight[e] || !add_jobs(*ls, e, s)) {
                continue;
            }
            n_res++;
            s++;
        }
        n_layers++;
    }
    for (size_t pi = 0; pi < mc->pools.size(); ++pi) {
        hot_pool & P = mc->pools[pi];
        auto & cand = pool_cands[pi];
        std::stable_sort(cand.begin(), cand.end(), [](const pool_cand & a, const pool_cand & b) { return a.c > b.c; });
        int32_t s = 0;
        for (const auto & c : cand) {
            while (s < P.n_slots && P.slot_layer[s] >= 0) {
                s++;
            }
            if (s >= P.n_slots) {
                break;
            }
            layer_state & ls = mc->layers[c.li];
            if (ls.expert_slot[c.e] >= 0 || ls.expert_in_flight[c.e] || !add_jobs(ls, c.e, s)) {
                continue;
            }
            P.slot_layer[s] = (int32_t) c.li;
            n_res++;
            s++;
        }
    }
    const size_t bytes = hot_refill(mc, jobs);
    for (auto & ls : mc->layers) {
        for (int32_t s = 0; s < ls.pub.n_slots; ++s) {
            if (ls.slot_expert[s] >= 0) {
                hot_adapt_set_entry(mc, ls, ls.slot_expert[s], s);
            }
        }
    }
    if (mc->tbl_dirty) {
        ggml_backend_tensor_set(mc->tbl_all, mc->tbl_mirror.data(), 0, mc->tbl_mirror.size()*sizeof(int32_t));
        mc->tbl_dirty = false;
    }
    LLAMA_LOG_INFO("moe-hot: [TAG_FN_L3_POLICY_STATE] loaded %s: %zu residents in %zu layers (%.2f GiB in %.0f ms), the "
            "counts of %llu decode steps%s\n", mc->st_path.c_str(), n_res, n_layers, bytes/1073741824.0,
            (ggml_time_us() - t0)/1000.0, (unsigned long long) steps,
            n_skip ? format(", %zu layer lines not used", n_skip).c_str() : "");
}

} // namespace

void llama_moe_hot_state_save(const void * owner) {
    moe_cache * mc = g_cache;
    if (mc && mc->adapt && mc->owner == owner) {
        hot_state_save(mc);
    }
}
