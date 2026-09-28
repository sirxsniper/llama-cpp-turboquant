#include "llama-moecache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <condition_variable>
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
};

struct upload_job {
    size_t  layer_idx;
    int32_t expert;
    int32_t slot;
    bool    done = false;
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

    // async upload worker: slices are copied to the device off the decode
    // thread; the new table mapping is only published at a later step() once
    // the upload has completed, so a running graph never reads a torn slot
    std::thread              worker;
    std::mutex               wmtx;
    std::condition_variable  wcv;
    std::deque<upload_job>   todo;
    std::vector<upload_job>  done;
    bool                     stop = false;
};

moe_cache * g_cache = nullptr;
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
    if (const char * hp = getenv("LLAMA_MOE_HOT_PROFILE"); hp && hp[0]) {
        if (n_slots > 0) {
            LLAMA_LOG_WARN("%s: LLAMA_MOE_HOT_PROFILE is set, --moe-expert-cache %d is ignored\n", __func__, n_slots);
        }
        return;
    }
    [&]() {
        if (n_slots <= 0) {
            g_init_done = true;
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

void llama_moe_cache_step() {
    moe_cache * mc = g_cache;
    if (!mc || mc->hot) { // [TAG_FN_MOE_HOT] the hot set never changes
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
        return;
    }
    const int il = parse_layer_from_name(name);
    if (il < 0 || il >= (int) g_hot_stats.hit.size()) {
        return;
    }
    const layer_state * ls = nullptr;
    for (const auto & l : mc->layers) {
        if (l.pub.il == il) { ls = &l; break; }
    }
    if (ls) {
        const int32_t * tbl = (const int32_t *) ls->pub.host_table->data;
        const int64_t n_exp = ls->pub.host_table->ne[1];
        for (int64_t t = 0; t < n_tokens; ++t) {
            for (int64_t i = 0; i < n_used; ++i) {
                const int32_t e = *(const int32_t *) ((const char *) ids->data + t*ids->nb[1] + i*ids->nb[0]);
                if (e >= 0 && e < n_exp) {
                    g_hot_stats.tot[il]++;
                    g_hot_stats.hit[il] += tbl[e] != ls->pub.n_slots;
                }
            }
        }
    }
    if (il != g_hot_stats.first_il) {
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
    LLAMA_LOG_INFO("moe-hot: %" PRIu64 " decode steps, hit rate %.3f (host layers only; layer min %.3f max %.3f)\n",
            mc->hot_steps, t ? (double) h / t : 0.0, lo, hi);
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

} // namespace

int llama_moe_hot_max_t() {
    const moe_cache * mc = g_cache;
    return mc && mc->hot ? mc->hot_max_t : 0;
}

bool llama_moe_hot_init(const llama_model & model) {
    const char * path = getenv("LLAMA_MOE_HOT_PROFILE");
    if (!path || !path[0]) {
        return false;
    }
    std::lock_guard<std::mutex> init_lock(g_init_mtx);
    if (g_init_done) {
        return false;
    }

    // host-resident expert layers with a device-resident router (the same rule as the LRU cache). A context of a
    // model whose weights are not loaded (memory estimation) finds none and leaves the one attempt to a later context.
    std::vector<int> host_layers;
    for (size_t il = 0; il < model.layers.size(); ++il) {
        const auto & l = model.layers[il];
        if (l.ffn_up_exps && l.ffn_gate_exps && l.ffn_down_exps && l.ffn_gate_inp && !l.ffn_gate_up_exps &&
                l.ffn_up_exps->data && l.ffn_up_exps->buffer && ggml_backend_buffer_is_host(l.ffn_up_exps->buffer) &&
                l.ffn_gate_inp->buffer && !ggml_backend_buffer_is_host(l.ffn_gate_inp->buffer)) {
            host_layers.push_back((int) il);
        }
    }
    if (host_layers.empty()) {
        LLAMA_LOG_WARN("moe-hot: no host-resident expert layer (all experts on a device?) - hot set off\n");
        return false;
    }
    g_init_done = true; // one attempt per process: on any failure below the model runs without a hot set

    int max_t = 8;
    if (const char * e = getenv("LLAMA_MOE_HOT_MAX_T")) {
        max_t = std::max(1, std::min(8, atoi(e)));
    }
    std::map<std::string, double> cost = { { "q5_1", 1.3 } };
    if (const char * e = getenv("LLAMA_MOE_HOT_COST")) {
        std::stringstream ss(e);
        std::string kv;
        while (std::getline(ss, kv, ',')) {
            const size_t eq = kv.find('=');
            if (eq != std::string::npos) {
                cost[kv.substr(0, eq)] = atof(kv.c_str() + eq + 1);
            }
        }
    }
    const char * sec_env = getenv("LLAMA_MOE_HOT_SECTION");
    std::map<int, std::vector<double>> prof;
    std::string sec_used;
    if (!hot_read_profile(path, sec_env ? sec_env : "decode_union", prof, sec_used)) {
        return false;
    }

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
    size_t budget = 0;
    const char * mib = getenv("LLAMA_MOE_HOT_MIB");
    if (mib && strcmp(mib, "auto") == 0) {
        size_t headroom = 1536;
        if (const char * e = getenv("LLAMA_MOE_HOT_HEADROOM_MIB")) {
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
        if (const char * e = getenv("LLAMA_MOE_HOT_CAP_MIB"); e && atoi(e) > 0) {
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
    if (budget == 0) {
        LLAMA_LOG_WARN("moe-hot: LLAMA_MOE_HOT_MIB is unset or 0 - hot set off\n");
        return false;
    }

    // greedy by count x cost per byte (all experts of a layer have the same size)
    struct cand_expert { double value; int li; int e; };
    std::vector<cand_expert> cands;
    for (int li = 0; li < (int) layers.size(); ++li) {
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
    };

    for (int attempt = 0; attempt < 32 && budget > 0; ++attempt) {
        std::vector<std::vector<int32_t>> hot_ids(layers.size());
        size_t used = 0;
        for (const auto & c : cands) {
            const size_t b = layers[c.li].bytes;
            if (used + b > budget) {
                continue;
            }
            used += b;
            hot_ids[c.li].push_back(c.e);
        }
        for (auto & v : hot_ids) {
            std::sort(v.begin(), v.end());
        }

        free_all();

        // device tensors grouped by the router's buffer type, host tables in one CPU buffer
        std::map<ggml_backend_buffer_type_t, std::vector<int>> groups;
        size_t n_hot_layers = 0;
        for (int li = 0; li < (int) layers.size(); ++li) {
            if (!hot_ids[li].empty()) {
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
            ggml_context * ctx_d = new_ctx(g.second.size()*4);
            if (!ctx_d) {
                ok = false;
                break;
            }
            for (int li : g.second) {
                const llama_layer * l = layers[li].l;
                const int32_t n_hot = (int32_t) hot_ids[li].size();
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
                ls.pub.up_c       = ggml_new_tensor_3d(ctx_d, u->type,  u->ne[0],  u->ne[1],  n_hot + 1);
                ls.pub.gate_c     = ggml_new_tensor_3d(ctx_d, gt->type, gt->ne[0], gt->ne[1], n_hot + 1);
                ls.pub.down_c     = ggml_new_tensor_3d(ctx_d, d->type,  d->ne[0],  d->ne[1],  n_hot + 1);
                ls.pub.dev_table  = ggml_new_tensor_2d(ctx_d, GGML_TYPE_I32, 1, u->ne[2]);
                ls.pub.host_table = ggml_new_tensor_2d(ctx_h, GGML_TYPE_I32, 1, u->ne[2]);
                ggml_format_name(ls.pub.up_c,       "moe_hot_up.%d",   ls.pub.il);
                ggml_format_name(ls.pub.gate_c,     "moe_hot_gate.%d", ls.pub.il);
                ggml_format_name(ls.pub.down_c,     "moe_hot_down.%d", ls.pub.il);
                ggml_format_name(ls.pub.dev_table,  "moe_hot_tbl.%d",  ls.pub.il);
                ggml_format_name(ls.pub.host_table, "moe_hot_htbl.%d", ls.pub.il);
            }
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_d, g.first);
            if (!buf) {
                ok = false;
                break;
            }
            ggml_backend_buffer_clear(buf, 0); // every slot, so the last one of each tensor is the zero slot
            mc->bufs.push_back(buf);
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

        // upload the hot slices from the host (mmap) copy and write both tables, once
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

            const size_t lb = ggml_nbytes(ls.pub.up_c) + ggml_nbytes(ls.pub.gate_c) + ggml_nbytes(ls.pub.down_c);
            vram += lb;
            const auto & v = prof[ls.pub.il];
            double hit = 0.0, all = 0.0;
            for (size_t e = 0; e < v.size(); ++e) {
                all += v[e];
                hit += tbl[e] != ls.pub.n_slots ? v[e] : 0.0;
            }
            LLAMA_LOG_INFO("moe-hot: blk.%-2d n_hot %3d  profile hit %.3f  %7.1f MiB\n", ls.pub.il, ls.pub.n_slots,
                    all > 0 ? hit/all : 0.0, lb/1048576.0);
        }
        g_cache = mc;
        LLAMA_LOG_INFO("moe-hot: %zu layers, %.1f MiB device memory (budget %.0f MiB), uploaded in %.1f s, section %s of %s, "
                "graphs of <= %d tokens\n", mc->layers.size(), vram/1048576.0, budget/1048576.0, (ggml_time_us() - t0)/1e6,
                sec_used.c_str(), path, max_t);

        if (const char * st = getenv("LLAMA_MOE_HOT_STATS"); st && atoi(st) != 0) {
            g_hot_stats.hit.assign(model.layers.size(), 0);
            g_hot_stats.tot.assign(model.layers.size(), 0);
            g_hot_stats.first_il = mc->layers.front().pub.il;
            for (const auto & ls : mc->layers) {
                g_hot_stats.first_il = std::min(g_hot_stats.first_il, ls.pub.il);
            }
            ggml_set_moe_obs_callback(hot_obs_cb, mc);
        }
        return true;
    }

    free_all();
    delete mc;
    LLAMA_LOG_WARN("moe-hot: no hot set could be allocated - hot set off\n");
    return false;
}
