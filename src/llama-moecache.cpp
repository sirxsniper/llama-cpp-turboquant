#include "llama-moecache.h"

#include "llama-fn-auto.h" // [TAG_FN_AUTO]
#include "llama-fn-stall.h" // [TAG_FN_L18_FIRSTANS_STALL]
#include "llama-fn-tier.h"  // [TAG_FN_L18_FIRSTANS_STATE] llama_fn_prefetch_ranges
#include "llama-moe-decay.h" // [TAG_FN_R4_ADAPT_DECAY]
#include "llama-moe-bridge.h" // [TAG_FN_L3_HOST_QUIET]
#include "llama-fn-l22.h"     // [TAG_FN_L22_P4] pins, PASSLOG
#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h> // [TAG_FN_L17_START_NT]
#endif

// [TAG_FN_L21_VRAM_DXGI] defined at the end of this file (the Windows headers stay out of the code above)
void fn_l21_dev_memory(const llama_model * model, ggml_backend_dev_t dev, size_t * mem_free, size_t * mem_total);
// [TAG_FN_L22_REFILL] M2.4: the calling thread's memory priority to low (Windows; elsewhere a no-op), also at the end
static void fn_l22_mem_prio_low();
// [TAG_FN_L22_ENDOVERLAP] r22 T6 fix: the answer's first step sends a lend's or a tail's refill to the upload worker only up
// to this many bytes; a larger one is refilled at once by the copy threads. The worker is one thread that shares the CPU
// with the decode's MoE pool: after a 12-screenshot turn its 4 GiB tail took 1.1-1.5 s to land (vs ~0.26 s for the copy
// threads before the decode), the CPU computed those experts meanwhile and the 256-token answers after image turns ran at
// 54-90 t/s vs 72-130 t/s on v0.24.0 (r22 end test T6). The first token is out before this step either way.
constexpr size_t L22_ASYNC_REFILL_MAX = (size_t) 256 << 20;

// [TAG_FN_L22_REFILL] (b) r22 T5 fix: the refill set's read-ahead, bounded (defined at the end: Windows only, elsewhere the
// old unbounded request). Only ranges whose first page is outside the working set, at most the free RAM above a 4 GiB
// floor, in requests of 64 ranges, and none after `stop` (set when the unlend / restore starts reading them itself)
struct fn_l22_refill_pf_stats {
    size_t ranges   = 0; // asked for by the caller
    size_t resident = 0; // ... whose first page was in the working set (left out)
    size_t asked_b  = 0; // bytes requested
    size_t cut_b    = 0; // bytes left out by the RAM budget
    size_t avail_b  = 0; // available RAM at the start
    bool   stopped  = false;
    double ms       = 0.0;
};
static void fn_l22_refill_prefetch_run(const std::vector<std::pair<const void *, size_t>> & rr, const std::atomic<bool> & stop,
                                       fn_l22_refill_pf_stats & st);

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
    bool                              lend_range = false; // [TAG_FN_L15_TIER] lent at least once (the range never moves)

    // [TAG_FN_L3_VRAM_CBUF] the layer lies in the tail of the device buffer (cb_tail), and the tail is out right now
    // (cb_lent): as lent, and in addition no graph may read its tensors (not even the zero slot), so the lookup hides it
    bool                              cb_tail = false;
    bool                              cb_lent = false;

    // [TAG_FN_L8_KVLEND] the layer's slots lie in a region of their own [kv_lo, kv_hi) near the bottom of the device buffer
    // (kv_tail), mapped apart, that gives its VRAM to the KV cache while the context needs it (kv_lent: as cb_lent)
    bool                              kv_tail = false;
    bool                              kv_lent = false;
    size_t                            kv_lo   = 0;
    size_t                            kv_hi   = 0;
    ggml_tensor *                     kv_raw  = nullptr; // I8 over the region: one clear after it is mapped again

    // [TAG_FN_L10_SLOTLEND] the slots from sl_base on lie (with their tensors' other slots) over granule segments of the
    // device buffer that go to the KV cache while its context needs them; slot_out[s]: the slot, or SL_SLACK bytes after
    // it, touches a segment that is out now - it holds no expert and takes no upload (-1 / empty: the layer lends nothing)
    int32_t                           sl_base = -1;
    // [TAG_FN_L21_BIGUB] slots [sl_bbase, sl_base) lie over segments of their own that go out only while a 16K prompt
    // ubatch's compute reserve needs their VRAM (-1: none); mapped, they are ordinary slots (and stream sources)
    int32_t                           sl_bbase = -1;
    std::vector<uint8_t>              slot_out;
    size_t                            sl_toff[3] = { 0, 0, 0 }; // up_c / gate_c / down_c: byte offset in the buffer
    size_t                            sl_sb[3]   = { 0, 0, 0 }; // and slot stride

    bool out() const { return lent || cb_lent || kv_lent; }

    // [TAG_FN_L3_POLICY_POOL] the layer's slots are those of a pool shared by its expert shape class: slot_expert and
    // slot_in_flight are pool sized and hold only this layer's entries (resident / upload into it in flight)
    int                               pool   = -1;
    int                               pool_k = -1; // the layer's index inside its pool

    // [TAG_FN_L22_LENDCLASS] slot_expert names an expert whose bytes never reached the slot (an upload dropped while the
    // layer was out keeps its expert for the next refill): a restore that skips intact slots must refill these
    std::vector<uint8_t>              slot_dirty;
};

// [TAG_FN_L3_POLICY_POOL] one dynamic slot pool per expert shape class (LLAMA_MOE_HOT_POOL=1): any layer of the class can
// take any slot of it. The class tensors (n_slots + 1 slots, the last one zero) are every member layer's up_c/gate_c/down_c
// and n_slots is every member's miss value, so the graph, the bridge and the CPU skip tables work unchanged.
struct hot_pool {
    std::vector<size_t>  layers;     // mc->layers indices, pool_k order
    std::vector<int32_t> slot_layer; // the mc->layers index that holds or receives a slot, -1 = free
    int32_t              n_slots = 0;
};

// one slice of a synchronous refill (hot_refill)
struct refill_job {
    ggml_tensor * dst;
    size_t        off;  // byte offset in dst
    const void *  src;
    size_t        size;
};

struct upload_job {
    size_t  layer_idx;
    int32_t expert;
    int32_t slot;
    bool    done = false;
    bool    failed = false; // [TAG_FN_L3_POLICY_UPLOAD] not uploaded (a slice out of range): the slot comes back empty
    bool    cb   = false; // [TAG_FN_L3_VRAM_CBUF] a refill job of the tail's restore
    bool    st   = false; // [TAG_FN_L18_FIRSTANS_STATE] a saved resident of the background state load
    bool    ul   = false; // [TAG_FN_L22_ENDOVERLAP] a refill job of an asynchronous unlend (the stream's range)
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
    int     hot_short_t = 0; // [TAG_FN_L4_MEM_PROMPT] LLAMA_FN_L4_PROMPT_HOT: graphs of up to this many tokens build the hot chain too
    int     hot_prompt_t = 0; // [TAG_FN_L19_TTFT_PHOT] LLAMA_FN_L19_TTFT_PHOT: wider CPU-path prompt graphs (up to this) too
    bool    phot_stats   = false; // [TAG_FN_L19_TTFT_PHOT] LLAMA_FN_L19_TTFT_PHOT_STATS=1: hit rate of each CPU-path prompt graph
    uint64_t phot_n = 0, phot_hit = 0, phot_lent = 0, phot_layers = 0;
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
    bool          ad_stats_l  = false; // [TAG_FN_L10_STATS2] LLAMA_MOE_HOT_STATS=2: per-layer rates of each 256-step window

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

    // [TAG_FN_L4_EXIT] paused (under wmtx): the owner is idle (llama_moe_hot_idle), the queue waits for its next step.
    // up_yield = stop, or paused without a drain: the worker takes no new batch and a bucket wait gives up. up_quit = stop:
    // the worker leaves its batch after the chunk in flight. A batch holds at most UP_INFLIGHT_MAX bytes.
    bool                     paused = false;
    std::atomic<bool>        up_yield { false };
    std::atomic<bool>        up_quit  { false };

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
    // [TAG_FN_L15_BURST] the burst budget only for this many passes after the set starts or a big prompt (a shifted working
    // set); in the steady state strong pairs took it on every pass: ~100 MiB of uploads per step, about 4.5 GB/s of RAM reads
    // beside the CPU experts' (LLAMA_MOE_HOT_BURST_PASSES, default 32; -1 = every pass, as before)
    int                   dc_burst_passes = -1; // every pass, as before (l15 meas15b: a steadier set measured no gain)
    int                   dc_burst_left   = 0;
    float                 dc_burst_ratio = 2.0f;  // LLAMA_MOE_HOT_BURST_RATIO
    uint64_t              dc_burst_n     = 0;     // swaps taken on the burst budget
    uint64_t              dc_burst_logs  = 0;
    uint64_t              sd_folded   = 0;     // [TAG_FN_L3_POLICY_SEED] prompt tokens folded so far (owner's step)
    uint64_t              sd_folds    = 0;
    std::atomic<uint64_t> st_step_us  { 0 };   // [TAG_FN_L3_POLICY] LLAMA_MOE_HOT_STATS: host time of the owner's steps
    std::atomic<uint64_t> st_step_n   { 0 };

    // [TAG_FN_L3_POLICY_STATE] the owner's step, the lend / unlend and the saves change or read the bookkeeping: one at a
    // time, also when a step runs on a helper thread while the server saves a slot. Taken before wmtx, never inside it.
    std::mutex            step_mtx;

    // [TAG_FN_L3_VRAM_CBUF] the device buffer on CUDA virtual memory; its tail [cb_lo, cb_hi) holds whole layers and can
    // give its VRAM back (cb_out) while the owner runs a big prompt with the full compute buffer
    ggml_backend_buffer_t cb_buf   = nullptr;
    size_t                cb_lo    = 0;
    size_t                cb_hi    = 0;
    bool                  cb_out   = false;
    // [TAG_FN_L20_BIGPROMPT_TAIL] while out, [cb_rel, cb_hi) has no VRAM (cb_lo after a whole release; a part from the top
    // for a prompt that needs a smaller compute reserve), at a boundary of the tail's mapped segments (cb_seg from cb_lo)
    size_t                cb_rel   = 0;
    size_t                cb_gran  = 0;
    size_t                cb_seg   = 0;
    uint64_t              cb_n_part = 0;
    ggml_context *        cb_ctx   = nullptr;
    ggml_tensor *         cb_raw   = nullptr; // I8 over the tail: one clear after it is mapped again
    bool                  cb_async = false;   // LLAMA_FN_CBUF_ASYNC=1: the refill through the upload worker ([TAG_FN_L4_EXIT] off by default)
    size_t                cb_async_left = 0;  // tail jobs of the last restore not published yet
    int64_t               cb_async_t0   = 0;
    bool (*cb_map)(ggml_backend_buffer_t, size_t, size_t)   = nullptr;
    bool (*cb_unmap)(ggml_backend_buffer_t, size_t, size_t) = nullptr;
    // [TAG_FN_L8_KVLEND] the KV cache's share: the kv_tail layers' regions, what is given to the KV cache now, statistics
    size_t                kv_bytes = 0;   // sum of the kv_tail regions
    size_t                kv_out   = 0;   // of them given back now
    uint64_t              kv_n_rel = 0, kv_n_res = 0;
    double                kv_rel_ms = 0.0, kv_res_ms = 0.0;
    // [TAG_FN_L10_SLOTLEND] the slot-granular KV lend (LLAMA_FN_L10_SLOTLEND): the lend layers' slot segments in groups
    // (group j = the j-th share of every lend tensor), given to the KV cache top group first, taken back bottom first
    bool                  sl_req  = false; // asked for at this init
    bool                  sl_fail = false; // the placement could not lend: the init fails, the KV cache maps whole
    bool                  sl_retry = false; // the segments came out short of the KV cache's want: one slot more per lend layer
    int32_t               sl_extra = 0;
    bool                  sl_on   = false;
    std::vector<std::vector<std::pair<size_t, size_t>>> sl_grp; // per group: [lo, hi) byte ranges of the buffer
    std::vector<size_t>   sl_grp_bytes;
    std::vector<size_t>   sl_given;        // per group: bytes given to the KV cache now (0: mapped)
    // [TAG_FN_L21_BIGUB] the groups [0, sl_n_kv) go to the KV cache; group sl_n_kv (when there is one) is the borrow group:
    // its sb_bytes go to the compute reserve of a 16K prompt ubatch (sb_out: out now), its residents come back after it
    size_t                sl_n_kv  = 0;
    size_t                sb_bytes = 0;
    size_t                sb_out   = 0;
    std::vector<std::array<int32_t, 3>> sb_res; // (layer, slot, expert) resident when the borrow went out
    uint64_t              sb_n = 0;
    double                sb_rel_ms = 0.0, sb_res_ms = 0.0;
    ggml_tensor *         sl_raw = nullptr; // I8 over [0, cb_lo): the clear of a segment mapped again
    // [TAG_FN_L17_VISION_LEND] the projector's share of the KV lend: its bytes, what it holds out now, statistics
    size_t                vis_bytes = 0;
    size_t                vis_out   = 0;
    uint64_t              vis_n     = 0;
    double                vis_take_ms = 0.0, vis_give_ms = 0.0;
    // [TAG_FN_L4_MEM_UPLOAD] LLAMA_FN_L4_UPLOAD=1 (qwen4exp): the tail's refill first (no swap is queued behind it for up to
    // l4_hold_ms after a restore), and the pinned halves (LLAMA_MOE_HOT_UP_PIPE) unless the environment says otherwise
    bool                  l4_upload  = false;
    int64_t               l4_hold_ms = 2000;
    uint64_t              l4_held    = 0;     // passes held for a refill
    uint64_t              cb_n_rel  = 0;
    uint64_t              cb_n_res  = 0;
    uint64_t              cb_n_fail = 0;
    double                cb_rel_ms = 0.0;
    double                cb_res_ms = 0.0;
    size_t                cb_res_b  = 0;
    // [TAG_FN_L17_START_PRE] LLAMA_FN_L17_START_PRE=<MiB> (qwen4exp): at the end of a prompt the old topic's counts fade
    // (x l17_td), the prompt's routing comes in as l17_sn decode steps of its mix, and one pass of the decay rules swaps
    // up to that many bytes; they land before the answer's first decode step (prompt time)
    size_t                l17_pre     = 0;
    float                 l17_td      = 0.5f;
    float                 l17_sn      = 32.0f;
    int                   l17_threads = 16;
    uint64_t              l17_n       = 0;
    uint64_t              l17_swaps   = 0;
    size_t                l17_bytes   = 0;
    double                l17_ms      = 0.0;
    // [TAG_FN_L17_START_NT] LLAMA_FN_L17_START_NT=1: the staging copies of uploads and refills with non-temporal stores
    bool                  l17_nt      = false;
    // [TAG_FN_L17_START_BURST] LLAMA_FN_L17_START_BURST=<passes>: the burst budget for that many passes after the set's start
    // only (a CBUF restore or the stream's unlend does not open the window again)
    bool                  l17_burst_start = false;
    // [TAG_FN_L17_START_STATE] LLAMA_FN_L17_START_STATE=<s>: the learned set (LLAMA_MOE_HOT_STATE) saved when the owner goes
    // idle, at most every <s> seconds and after >= 64 new decode steps, so a hard stop keeps a recent one
    int64_t               l17_st_every_us = 0;
    int64_t               l17_st_last_us  = 0;
    uint64_t              l17_st_steps    = 0;
    // [TAG_FN_L18_FIRSTANS_STATE] LLAMA_FN_L18_FIRSTANS_STATE=1: the saved residents go into the upload worker's queue at the
    // set's start (their pages read ahead first) instead of one synchronous refill; each lands and is published at an owner
    // step like any admission. The counts of what is still in flight (owner's step / wmtx)
    bool                  fa18_state       = false;
    std::thread           fa18_pf_th; // the read-ahead of the saved residents' pages, then their uploads are queued
    std::vector<upload_job> fa18_st_jobs; // in flight in the bookkeeping, not queued yet (until their pages are in)
    std::atomic<bool>     fa18_pf_abort{false}; // the stop: the read-ahead ends at its next range
    size_t                fa18_st_left     = 0;
    size_t                fa18_st_n        = 0;
    size_t                fa18_st_bytes    = 0;
    int64_t               fa18_st_t0       = 0;
    // [TAG_FN_L18_FIRSTANS_OVERLAP] LLAMA_FN_L18_FIRSTANS_OVERLAP=1: the prompt-end pass decides and evicts as before, then its
    // copies run on a thread of their own while the caller goes on (the first token's sampling, the draft context's pass
    // over the prompt, the context checkpoint); the owner's next call into the hot set - the next decode at the latest -
    // joins it and publishes its experts, so every graph sees the tables it would see after the synchronous pass. The
    // upload worker stays paused until then, and nothing else uses the refill's staging meanwhile (every user joins first).
    int                               fa18_ovl      = 0; // 1: the owner's next decode waits for the copies; 2: it does not
    std::thread                       fa18_th;
    std::atomic<bool>                 fa18_busy{false};
    std::atomic<bool>                 fa18_done{false};
    std::vector<refill_job>           fa18_jobs;
    std::vector<llama_moe_decay_swap> fa18_taken;
    size_t                            fa18_bytes    = 0;
    int64_t                           fa18_t0       = 0;
    int64_t                           fa18_t_idle   = 0;
    int64_t                           fa18_t_launch = 0;
    int64_t                           fa18_t_done   = 0;
    double                            fa18_wait_ms  = 0.0;
    double                            fa18_hid_ms   = 0.0;
    uint64_t                          fa18_n        = 0;
    // [TAG_FN_L18_CHURN_PACE] LLAMA_FN_L18_CHURN_PACE=<MiB/s>: the worker's copies of the steady state at no more than this
    // rate (wall time, a bucket of l18_pace_cap bytes), so the uploads run beside the CPU experts as a stream instead of
    // bursts; off while the start's burst window is open and while a lend or a stop drains the queue
    double                l18_pace      = 0.0;  // bytes per microsecond, 0 = off
    size_t                l18_pace_cap  = 0;
    double                l18_pace_cr   = 0.0;  // worker only: the bucket
    int64_t               l18_pace_t    = 0;    // worker only: its last refill
    std::atomic<bool>     l18_pace_live { false }; // written by the owner's pass: no burst window now
    std::atomic<bool>     l18_drain     { false }; // hot_drain_out waits for the worker's batch
    std::atomic<uint64_t> l18_pace_waits { 0 };
    std::atomic<uint64_t> l18_pace_us    { 0 };
    // [TAG_FN_L18_CHURN_GATE] LLAMA_FN_L18_CHURN_GATE=<KiB>: the worker copies in pieces of this size and starts a piece, and
    // the DMA of what it staged, only while no CPU expert job runs (llama_moe_bridge_gate_wait), so the uploads' DRAM reads
    // and PCIe traffic fall between the jobs instead of on them (0 = off)
    size_t                l18_gate      = 0;
    int                   l18_settle    = 0; // LLAMA_FN_L18_CHURN_GATE_SETTLE=<us>: a DMA starts that long after the last job's end
    // [TAG_FN_L18_CHURN_AB] LLAMA_FN_L18_CHURN_AB=pipe|mib|gate (design measurements, needs LLAMA_MOE_HOT_STATS): one lever
    // flips at every 256-step statistics window (mode 0 = off, 1 = on), so one session measures both under the same content
    int                   l18_ab        = 0; // 0 off, 1 pipe (staged <-> pinned halves), 2 mib (pass budget), 3 gate
    std::atomic<int>      l18_ab_mode   { 0 };
    // [TAG_FN_L18_CHURN_MIB] LLAMA_FN_L18_CHURN_MIB=<MiB>: the pass budget after the burst window (0 = LLAMA_MOE_HOT_DECAY_MIB)
    size_t                l18_mib       = 0;

    // [TAG_FN_L22_P4] round 22, prompt stream state machine
    bool                  l22_hide      = false; // [TAG_FN_L22_KEEPLEND] lent layers hidden from the hot chains (lookup)
    bool                  l22_refill    = false; // [TAG_FN_L22_REFILL] pinned refill sources, memory priority low, prefetch
    size_t                l22_ul_left   = 0;     // [TAG_FN_L22_ENDOVERLAP] async unlend jobs not published yet
    int64_t               l22_ul_t0     = 0;
    uint64_t              l22_ul_n      = 0;
    size_t                l22_ul_bytes  = 0;
    double                l22_ul_ms     = 0.0;   // host time of the unlends (sync or async) since the last take
    size_t                l22_ul_b      = 0;     // bytes refilled (sync) or queued (async) since the last take
    double                l22_res_ms    = 0.0;   // the same for the tail's restores
    size_t                l22_res_b     = 0;
    std::atomic<uint64_t> l22_pin_b{0};          // [TAG_FN_L22_REFILL] refill / upload bytes straight from pinned pages
    size_t                l22_rel_b     = 0;     // [TAG_FN_L22_LENDCLASS] tail bytes unmapped since the last take
    std::thread           l22_pf_th;             // [TAG_FN_L22_REFILL] (b) the refill set's prefetch
    std::atomic<bool>     l22_pf_busy{false};
    std::atomic<bool>     l22_pf_stop{false};       // [TAG_FN_L22_REFILL] r22 T5: the unlend / restore reads the set itself now
    uint64_t              l22_sync_ul  = 0;          // [TAG_FN_L22_ENDOVERLAP] r22 T6: big lend refills done synchronously
    uint64_t              l22_sync_res = 0;          // ... big tail refills
    // [TAG_FN_L22_LENDCLASS] the tail is out with fewer bytes unmapped than layers hidden (a narrow FULL class): the
    // restore refills only the slots that reach into the unmapped part, and the dirty ones
    bool                  l22_part      = false;

    // [TAG_FN_R25_REFILL] LLAMA_FN_R25_REFILL: synchronous refills through the cache-resident copy pool (g_r25)
    bool                  r25_on        = false;
    int                   r25_threads   = 8;          // LLAMA_FN_R25_REFILL_THREADS
    size_t                r25_half      = 2u << 20;   // LLAMA_FN_R25_REFILL_HALF_KIB
    int                   r25_verify    = 0;          // LLAMA_FN_R25_VERIFY=<k>: every k-th piece read back and compared
    size_t                r25_sb_max    = 1u << 30;   // LLAMA_FN_R25_SBSYNC_MIB: a borrow return up to this size lands at once
    uint64_t              r25_bad       = 0;          // verify mismatches
    uint64_t              r25_sb_sync   = 0;
    const void *          r25_owner     = nullptr;    // the context that stops the pool (llama_moe_hot_stop)
};

moe_cache * g_cache = nullptr;

void hot_adapt_step(moe_cache * mc); // [TAG_FN_MOE_HOT_ADAPT]
void hot_state_save(moe_cache * mc); // [TAG_FN_L3_POLICY_STATE]
void hot_state_load(moe_cache * mc);
std::string hot_state_cache_path(const std::string & model_path); // [TAG_FN_L18_FIRSTANS_STATE]
void hot_turn_join(moe_cache * mc, bool wait = true); // [TAG_FN_L18_FIRSTANS_OVERLAP]
void hot_state_queue(moe_cache * mc); // [TAG_FN_L18_FIRSTANS_STATE]
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
    const layer_state & ls = g_cache->layers[it->second];
    if (ls.cb_lent || ls.kv_lent) {
        return nullptr; // [TAG_FN_L3_VRAM_CBUF] [TAG_FN_L8_KVLEND] its slots have no VRAM now: the host computes all of its experts
    }
    if (ls.lent && g_cache->l22_hide) {
        return nullptr; // [TAG_FN_L22_KEEPLEND] its slots hold bank bytes (also its zero slot): the host computes all of them
    }
    return &ls.pub;
}

const llama_moe_cache_layer * llama_moe_cache_lookup_table(const ggml_tensor * up_exps) {
    if (!g_cache) {
        return nullptr;
    }
    auto it = g_cache->by_up_src.find(up_exps);
    return it == g_cache->by_up_src.end() ? nullptr : &g_cache->layers[it->second].pub;
}

// [TAG_FN_L14_TIER] another thread: g_cache under the init lock, the bookkeeping under step_mtx, never waiting for either
int llama_moe_hot_residents(const ggml_tensor * up_exps, uint8_t * out, int64_t n_expert) {
    std::unique_lock<std::mutex> ilk(g_init_mtx, std::try_to_lock);
    if (!ilk.owns_lock()) {
        return 0;
    }
    moe_cache * mc = g_cache;
    ilk.unlock();
    if (!mc || !mc->hot) {
        return -1;
    }
    const auto it = mc->by_up_src.find(up_exps);
    if (it == mc->by_up_src.end()) {
        return -1;
    }
    std::unique_lock<std::mutex> slk(mc->step_mtx, std::try_to_lock);
    if (!slk.owns_lock()) {
        return 0;
    }
    const layer_state & ls = mc->layers[it->second];
    for (int64_t e = 0; e < n_expert; ++e) {
        out[e] = !ls.kv_lent && e < (int64_t) ls.expert_slot.size() && ls.expert_slot[e] >= 0 ? 1 : 0;
    }
    return 1;
}

// [TAG_FN_L15_TIER] the eviction gate (set by the RAM tier of the model, read in the owner's step)
std::atomic<llama_moe_host_ready_fn> g_host_ready{nullptr};
std::atomic<void *>                  g_host_ready_ud{nullptr};

void llama_moe_hot_set_host_ready(llama_moe_host_ready_fn fn, void * ud) {
    g_host_ready_ud.store(ud);
    g_host_ready.store(fn);
}

int llama_moe_hot_tier_view(const ggml_tensor * up_exps, int32_t * slot, float * cnt, uint8_t * vol, int64_t n_expert) {
    std::unique_lock<std::mutex> ilk(g_init_mtx, std::try_to_lock);
    if (!ilk.owns_lock()) {
        return 0;
    }
    moe_cache * mc = g_cache;
    ilk.unlock();
    if (!mc || !mc->hot) {
        return -1;
    }
    const auto it = mc->by_up_src.find(up_exps);
    if (it == mc->by_up_src.end()) {
        return -1;
    }
    std::unique_lock<std::mutex> slk(mc->step_mtx, std::try_to_lock);
    if (!slk.owns_lock()) {
        return 0;
    }
    const layer_state & ls = mc->layers[it->second];
    // the whole layer can go: the tail, the stream's range, a KV region; else only the slots over the KV cache's segments
    const bool vol_layer = ls.cb_tail || ls.lend_range || ls.kv_tail;
    for (int64_t e = 0; e < n_expert; ++e) {
        const int32_t s = !ls.kv_lent && e < (int64_t) ls.expert_slot.size() ? ls.expert_slot[e] : -1;
        slot[e] = s;
        cnt[e]  = e < (int64_t) ls.dcnt.size() ? ls.dcnt[e] : 0.0f;
        vol[e]  = s >= 0 && (vol_layer || (ls.sl_base >= 0 && s >= ls.sl_base)) ? 1 : 0;
    }
    return 1;
}

// [TAG_FN_L14_PFSD2D] see llama-moecache.h
bool llama_moe_hot_slots(const ggml_tensor * up_exps, int32_t * slot, int64_t n_expert, const ggml_tensor ** slots3) {
    moe_cache * mc = g_cache;
    if (!mc || !mc->hot) {
        return false;
    }
    const auto it = mc->by_up_src.find(up_exps);
    if (it == mc->by_up_src.end()) {
        return false;
    }
    std::unique_lock<std::mutex> slk(mc->step_mtx, std::try_to_lock);
    if (!slk.owns_lock()) {
        return false;
    }
    const layer_state & ls = mc->layers[it->second];
    // a layer whose whole region the KV cache can take back (kv_tail) is never a source: a KV growth between two streamed
    // ubatches unmaps it, while a copy queued for the next ubatch (the stream's wrap job) may still be waiting to run
    if (ls.out() || ls.kv_tail || ls.pub.n_slots == 0 || !ls.pub.up_c || !ls.pub.gate_c || !ls.pub.down_c) {
        return false;
    }
    const int64_t n_cap = ls.pub.up_c->ne[2];
    bool any = false;
    for (int64_t e = 0; e < n_expert; ++e) {
        const int32_t s = e < (int64_t) ls.expert_slot.size() ? ls.expert_slot[e] : -1;
        // [TAG_FN_L10_SLOTLEND] the slots from sl_base on can go to the KV cache the same way: never a source either
        const bool ok = s >= 0 && s < n_cap && (ls.slot_out.empty() || !ls.slot_out[s]) && !(ls.sl_base >= 0 && s >= ls.sl_base);
        slot[e] = ok ? s : -1;
        any = any || ok;
    }
    slots3[0] = ls.pub.up_c;
    slots3[1] = ls.pub.gate_c;
    slots3[2] = ls.pub.down_c;
    return any;
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
        // [TAG_FN_L19_TTFT_PHOT] LLAMA_FN_L19_TTFT_PHOT_STATS: the share of a prompt graph's routed experts the hot set holds
        // (the CPU skips those with PHOT); one line per graph, at its last host layer
        if (mc->phot_stats) {
            const int il_s = parse_layer_from_name(name);
            for (auto & l : mc->layers) {
                if (l.pub.il != il_s) {
                    continue;
                }
                const int32_t * tbl = (const int32_t *) l.pub.host_table->data;
                const int64_t n_exp = l.pub.host_table->ne[1];
                const bool lent = l.cb_lent || l.kv_lent;
                for (int64_t t = 0; t < n_tokens; ++t) {
                    for (int64_t i = 0; i < n_used; ++i) {
                        const int32_t e = *(const int32_t *) ((const char *) ids->data + t*ids->nb[1] + i*ids->nb[0]);
                        if (e < 0 || e >= n_exp) {
                            continue;
                        }
                        mc->phot_n++;
                        mc->phot_hit += !lent && tbl[e] != l.pub.n_slots;
                    }
                }
                mc->phot_lent += lent;
                mc->phot_layers++;
                if (&l == &mc->layers.back()) {
                    LLAMA_LOG_WARN("moe-hot: [TAG_FN_L19_TTFT_PHOT] prompt graph of %lld tokens: %.3f of %llu routed experts are "
                            "hot (%llu of %llu host layers lent out)\n", (long long) n_tokens,
                            mc->phot_n ? (double) mc->phot_hit/mc->phot_n : 0.0, (unsigned long long) mc->phot_n,
                            (unsigned long long) mc->phot_lent, (unsigned long long) mc->phot_layers);
                    mc->phot_n = mc->phot_hit = mc->phot_lent = mc->phot_layers = 0;
                }
                break;
            }
        }
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
    if (mc->l18_ab > 0) { // [TAG_FN_L18_CHURN_AB] the window that ends now ran in this mode; the next one in the other
        const int m = mc->l18_ab_mode.load(std::memory_order_relaxed);
        LLAMA_LOG_INFO("moe-hot: [TAG_FN_L18_CHURN_AB] window %" PRIu64 " ran in mode %d\n", mc->hot_steps/256, m);
        mc->l18_ab_mode.store(m ^ 1, std::memory_order_relaxed);
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
    if (mc->ad_stats_l) {
        // [TAG_FN_L10_STATS2] the last window per layer (hits / routed), then the window restarts
        static std::vector<uint64_t> ph, pt;
        ph.resize(g_hot_stats.hit.size(), 0);
        pt.resize(g_hot_stats.tot.size(), 0);
        std::string line;
        uint64_t wh = 0, wt = 0;
        for (size_t i = 0; i < g_hot_stats.hit.size(); ++i) {
            const uint64_t dh = g_hot_stats.hit[i] - ph[i];
            const uint64_t dt = g_hot_stats.tot[i] - pt[i];
            ph[i] = g_hot_stats.hit[i];
            pt[i] = g_hot_stats.tot[i];
            if (dt == 0) {
                continue;
            }
            wh += dh;
            wt += dt;
            line += format(" %zu:%.2f", i, (double) dh/dt);
        }
        LLAMA_LOG_INFO("moe-hot: [TAG_FN_L10_STATS2] window hit rate %.3f, per layer%s\n", wt ? (double) wh/wt : 0.0, line.c_str());
    }
    if (mc->adapt) {
        const uint64_t sn = mc->st_step_n.load(std::memory_order_relaxed);
        LLAMA_LOG_INFO("moe-hot: adaptive: %" PRIu64 " experts admitted, %" PRIu64 " verify mismatches%s, step host time %.0f us%s%s\n",
                mc->ad_admitted, mc->ad_bad,
                mc->dc_on ? format(", %" PRIu64 " decay passes, %" PRIu64 " swaps queued", mc->dc_passes, mc->dc_swaps).c_str() : "",
                sn ? (double) mc->st_step_us.load(std::memory_order_relaxed)/sn : 0.0,
                mc->up_pipe || mc->up_rate > 0 || mc->sd_node || mc->dc_burst > 0 ? format(" [TAG_FN_L3_POLICY] uploaded %.0f MiB, %" PRIu64
                        " bucket waits, %" PRIu64 " prompt tokens seeded, %" PRIu64 " burst swaps", mc->up_bytes.load()/1048576.0,
                        mc->up_waits.load(), mc->sd_tokens.load(), mc->dc_burst_n).c_str() : "",
                mc->l18_pace > 0.0 ? format(" [TAG_FN_L18_CHURN_PACE] %" PRIu64 " pace waits, %.0f ms", mc->l18_pace_waits.load(),
                        mc->l18_pace_us.load()/1000.0).c_str() : "");
        if (mc->l18_gate > 0) {
            uint64_t gw = 0, gus = 0;
            llama_moe_bridge_gate_stats(&gw, &gus);
            LLAMA_LOG_INFO("moe-hot: [TAG_FN_L18_CHURN_GATE] %" PRIu64 " gate waits, %.1f ms in all (%.1f us each)\n", gw, gus/1000.0,
                    gw ? (double) gus/gw : 0.0);
        }
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
    if (mc->up_bucket < bytes && !mc->up_drain) {
        if (!wait || mc->up_yield.load()) {
            return false; // [TAG_FN_L4_EXIT] stop or idle: the worker puts its batch down instead of waiting for a step
        }
        mc->up_waits.fetch_add(1, std::memory_order_relaxed);
        mc->wcv.wait(lk, [&]() { return mc->up_bucket >= bytes || mc->up_drain || mc->up_yield.load(); });
        if (mc->up_bucket < bytes && !mc->up_drain) {
            return false;
        }
    }
    mc->up_bucket = mc->up_bucket > bytes ? mc->up_bucket - bytes : 0;
    return true;
}

// [TAG_FN_L4_EXIT] under wmtx: the worker yields (puts its batch down after the chunk in flight) on stop, and while the
// owner is idle unless a lend / release drains the queue
void hot_set_yield(moe_cache * mc) {
    // a drain needs the worker only with a rate limit (it then waits for the queue to empty)
    mc->up_yield.store(mc->stop || (mc->paused && !(mc->up_drain && mc->up_rate > 0)));
    mc->up_quit.store(mc->stop);
}

// [TAG_FN_L4_EXIT] at most this many bytes of the worker's copies are in flight at any time
constexpr size_t UP_INFLIGHT_MAX = 64u << 20;

// a slice's source and destination are inside their tensors (upload_slice's check)
bool hot_slice_ok(const ggml_tensor * dst_c, const ggml_tensor * src, int32_t expert, int32_t slot) {
    const size_t sz = src->nb[2];
    if (slot < 0 || expert < 0 || (size_t) slot*dst_c->nb[2] + sz > ggml_nbytes(dst_c) || (size_t) expert*sz + sz > ggml_nbytes(src)) {
        LLAMA_LOG_ERROR("moe-hot: bad upload %s <- %s expert=%d slot=%d - skipped\n", dst_c->name, src->name, expert, slot);
        return false;
    }
    return true;
}

// [TAG_FN_L18_CHURN_PACE] the worker waits until the bucket holds `bytes` (refilled at l18_pace bytes per us, at most
// l18_pace_cap). No wait while the start's burst window is open, while a lend or a stop drains the queue, or while the
// owner is idle (the CPU experts do not run then).
void hot_pace(moe_cache * mc, size_t bytes) {
    if (mc->l18_pace <= 0.0) {
        return;
    }
    int64_t now = ggml_time_us();
    if (mc->l18_pace_t == 0) {
        mc->l18_pace_t  = now;
        mc->l18_pace_cr = (double) mc->l18_pace_cap;
    }
    mc->l18_pace_cr = std::min((double) mc->l18_pace_cap, mc->l18_pace_cr + (double) (now - mc->l18_pace_t)*mc->l18_pace);
    mc->l18_pace_t  = now;
    const bool live = mc->l18_pace_live.load(std::memory_order_relaxed) && !mc->up_yield.load(std::memory_order_relaxed);
    if (!live || mc->l18_drain.load(std::memory_order_relaxed) || mc->l18_pace_cr >= (double) bytes) {
        mc->l18_pace_cr = std::max(0.0, mc->l18_pace_cr - (double) bytes);
        return;
    }
    const int64_t wait_us = (int64_t) (((double) bytes - mc->l18_pace_cr)/mc->l18_pace) + 1;
    std::this_thread::sleep_for(std::chrono::microseconds(wait_us));
    now = ggml_time_us();
    mc->l18_pace_waits.fetch_add(1, std::memory_order_relaxed);
    mc->l18_pace_us.fetch_add((uint64_t) std::max<int64_t>(0, now - mc->l18_pace_t), std::memory_order_relaxed);
    mc->l18_pace_cr = std::min((double) mc->l18_pace_cap, mc->l18_pace_cr + (double) (now - mc->l18_pace_t)*mc->l18_pace);
    mc->l18_pace_cr = std::max(0.0, mc->l18_pace_cr - (double) bytes);
    mc->l18_pace_t  = now;
}

// [TAG_FN_L18_CHURN_GATE] max wait for the gate per piece: a job that runs longer (page faults) does not hold the worker
constexpr int L18_GATE_MAX_US = 2000;

// [TAG_FN_L18_CHURN_GATE] the copy into staging in gated pieces (a plain copy when the gate is off)
void hot_copy_gated(const moe_cache * mc, void * dst, const void * src, size_t n, bool nt);

// [TAG_FN_L17_START_NT] a copy into staging that only the DMA reads: non-temporal stores, so no destination line is read
// first and the staging does not push the CPU experts' lines out of the caches (the CRT memcpy streams only from ~1.5 MiB,
// above an expert slice). Same bytes as memcpy. For the large staging buffers only: the small pinned halves of
// LLAMA_MOE_HOT_UP_PIPE are meant to stay in the caches.
void hot_copy(const moe_cache * mc, void * dst, const void * src, size_t n) {
#if defined(_M_X64) || defined(__x86_64__)
    if (mc->l17_nt) {
        uint8_t *       d = (uint8_t *) dst;
        const uint8_t * s = (const uint8_t *) src;
        size_t i = std::min(n, (size_t) ((16 - ((uintptr_t) d & 15)) & 15));
        if (i > 0) {
            memcpy(d, s, i);
        }
        for (; i + 64 <= n; i += 64) {
            const __m128i a = _mm_loadu_si128((const __m128i *) (s + i));
            const __m128i b = _mm_loadu_si128((const __m128i *) (s + i + 16));
            const __m128i c = _mm_loadu_si128((const __m128i *) (s + i + 32));
            const __m128i e = _mm_loadu_si128((const __m128i *) (s + i + 48));
            _mm_stream_si128((__m128i *) (d + i),      a);
            _mm_stream_si128((__m128i *) (d + i + 16), b);
            _mm_stream_si128((__m128i *) (d + i + 32), c);
            _mm_stream_si128((__m128i *) (d + i + 48), e);
        }
        if (i < n) {
            memcpy(d + i, s + i, n - i);
        }
        _mm_sfence(); // the streamed lines are in memory before the DMA is queued
        return;
    }
#endif
    memcpy(dst, src, n);
}

// [TAG_FN_L18_CHURN_GATE] the gate holds the worker in the steady state only: the start's burst window fills at full speed
bool hot_gate_on(const moe_cache * mc) {
    return mc->l18_gate > 0 && mc->l18_pace_live.load(std::memory_order_relaxed) &&
           (mc->l18_ab != 3 || mc->l18_ab_mode.load(std::memory_order_relaxed) == 1); // [TAG_FN_L18_CHURN_AB]
}

void hot_copy_gated(const moe_cache * mc, void * dst, const void * src, size_t n, bool nt) {
    if (!hot_gate_on(mc)) {
        if (nt) {
            hot_copy(mc, dst, src, n);
        } else {
            memcpy(dst, src, n);
        }
        return;
    }
    for (size_t o = 0; o < n; o += mc->l18_gate) {
        const size_t k = std::min(mc->l18_gate, n - o);
        llama_moe_bridge_gate_wait(L18_GATE_MAX_US);
        if (nt) {
            hot_copy(mc, (uint8_t *) dst + o, (const uint8_t *) src + o, k);
        } else {
            memcpy((uint8_t *) dst + o, (const uint8_t *) src + o, k);
        }
    }
}

// [TAG_FN_L3_POLICY_UPLOAD] LLAMA_MOE_HOT_UP_PIPE: the batch through two small pinned halves - the copy threads fill one
// while the upload stream reads the other (an event per half orders the reuse) - so the staging stays cache resident and
// the memcpy overlaps the DMA. Returns after every copy of the batch has landed.
// [TAG_FN_L4_EXIT] returns the jobs done; on stop (or a bucket wait at idle) the rest, from that index, is not uploaded
size_t hot_upload_pipe(moe_cache * mc, std::vector<upload_job> & batch) {
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
        llama_moe_bridge_quiet_wait(100); // [TAG_FN_L3_HOST_QUIET] the copies read host memory too
        if (hot_gate_on(mc)) {
            llama_moe_bridge_gate_wait(L18_GATE_MAX_US, mc->l18_settle); // [TAG_FN_L18_CHURN_GATE] the DMA starts between two jobs
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
    size_t n_done = 0;
    // [TAG_FN_L22_REFILL] a slice on pinned pages goes straight from there (no staging); tickets after the synchronize
    const bool pin_live = mc->l22_refill && llama_fn_pin_active();
    std::vector<int32_t> pin_tk;
    for (; n_done < batch.size(); ++n_done) {
        if (mc->up_quit.load(std::memory_order_relaxed)) {
            break; // [TAG_FN_L4_EXIT] stop: what is staged lands below, the rest is dropped
        }
        upload_job & j = batch[n_done];
        llama_moe_bridge_quiet_wait(100); // [TAG_FN_L3_HOST_QUIET] no expert copy while a bridged graph runs
        layer_state & ls = mc->layers[j.layer_idx];
        ggml_tensor *       dsts[3] = { ls.pub.up_c,   ls.pub.gate_c,   ls.pub.down_c   };
        const ggml_tensor * srcs[3] = { ls.pub.up_src, ls.pub.gate_src, ls.pub.down_src };
        for (int k = 0; k < 3 && !j.failed; ++k) {
            j.failed = !hot_slice_ok(dsts[k], srcs[k], j.expert, j.slot);
        }
        if (j.failed) {
            continue;
        }
        bool cut = false;
        for (int k = 0; k < 3; ++k) {
            const size_t sz = srcs[k]->nb[2];
            batch_bytes += sz;
            if (!hot_rate_take(mc, sz, false)) {
                flush(); // what is staged uploads while the bucket refills
                if (!hot_rate_take(mc, sz, true)) {
                    cut = true; // [TAG_FN_L4_EXIT] stop / idle: the job is dropped whole (its slot is in flight, no table names it)
                    break;
                }
            }
            if (pin_live) {
                const uint8_t *  p   = (const uint8_t *) srcs[k]->data + (size_t) j.expert*sz;
                llama_fn_pin_seg seg = {};
                const bool       ok  = llama_fn_pin_query(p, sz, &seg);
                if (ok && seg.ticket >= 0) {
                    pin_tk.push_back(seg.ticket);
                }
                if (ok && seg.pinned && seg.len == sz) {
                    hot_pace(mc, sz); // [TAG_FN_L18_CHURN_PACE]
                    // R-REV-4: seg.src = the registered pages the ticket protects (the arena copy in arena mode)
                    ggml_backend_tensor_set_async(mc->up_backend, dsts[k], seg.src, (size_t) j.slot*dsts[k]->nb[2], sz);
                    mc->up_bytes.fetch_add(sz, std::memory_order_relaxed);
                    mc->l22_pin_b.fetch_add(sz, std::memory_order_relaxed);
                    continue;
                }
            }
            if (sz > mc->up_half) {
                flush();
                hot_pace(mc, sz); // [TAG_FN_L18_CHURN_PACE]
                upload_slice(dsts[k], srcs[k], j.expert, j.slot);
                mc->up_bytes.fetch_add(sz, std::memory_order_relaxed);
                continue;
            }
            if (used + sz > mc->up_half) {
                flush();
            }
            hot_pace(mc, sz); // [TAG_FN_L18_CHURN_PACE]
            hot_copy_gated(mc, half(h) + used, (const char *) srcs[k]->data + (size_t) j.expert*sz, sz, false); // [TAG_FN_L18_CHURN_GATE]
            st.push_back({ dsts[k], (size_t) j.slot*dsts[k]->nb[2], sz, used });
            used += sz;
            mc->up_bytes.fetch_add(sz, std::memory_order_relaxed);
        }
        if (cut) {
            break;
        }
    }
    flush();
    ggml_backend_synchronize(mc->up_backend); // every copy of the batch has landed
    for (int32_t t : pin_tk) {
        llama_fn_pin_release(t); // [TAG_FN_L22_REFILL]
    }
    if (mc->up_batches++ < 2) {
        const double ms = (ggml_time_us() - t0)/1000.0;
        LLAMA_LOG_INFO("moe-hot: [TAG_FN_L3_POLICY_UPLOAD] batch %llu: %zu experts, %.1f MiB through the pinned halves in %.1f ms "
                "(%.1f GiB/s, incl. bucket waits)\n", (unsigned long long) mc->up_batches, batch.size(), batch_bytes/1048576.0, ms,
                ms > 0 ? batch_bytes/1073741824.0/(ms/1000.0) : 0.0);
    }
    return n_done;
}

// the batch through the one large pinned staging buffer: fill it, upload, wait, repeat
// [TAG_FN_L4_EXIT] returns the jobs done, as hot_upload_pipe
size_t hot_upload_staged(moe_cache * mc, const std::vector<upload_job> & batch) {
    struct staged { ggml_tensor * dst; size_t off; size_t size; size_t stage_off; };
    std::vector<staged> st;
    size_t used = 0;
    auto flush = [&]() {
        llama_moe_bridge_quiet_wait(100); // [TAG_FN_L3_HOST_QUIET] the copies read host memory too
        if (hot_gate_on(mc)) {
            llama_moe_bridge_gate_wait(L18_GATE_MAX_US, mc->l18_settle); // [TAG_FN_L18_CHURN_GATE]
        }
        for (const auto & x : st) {
            ggml_backend_tensor_set_async(mc->up_backend, x.dst, mc->stage_ptr + x.stage_off, x.off, x.size);
        }
        ggml_backend_synchronize(mc->up_backend);
        st.clear();
        used = 0;
    };
    size_t n_done = 0;
    for (; n_done < batch.size(); ++n_done) {
        if (mc->up_quit.load(std::memory_order_relaxed)) {
            break; // [TAG_FN_L4_EXIT] stop: what is staged lands below, the rest is dropped
        }
        const upload_job & j = batch[n_done];
        llama_moe_bridge_quiet_wait(100); // [TAG_FN_L3_HOST_QUIET] no expert copy while a bridged graph runs
        layer_state & ls = mc->layers[j.layer_idx];
        ggml_tensor *       dsts[3] = { ls.pub.up_c,   ls.pub.gate_c,   ls.pub.down_c   };
        const ggml_tensor * srcs[3] = { ls.pub.up_src, ls.pub.gate_src, ls.pub.down_src };
        bool cut = false;
        for (int k = 0; k < 3; ++k) {
            const size_t sz = srcs[k]->nb[2];
            if (!hot_rate_take(mc, sz)) { // [TAG_FN_L3_POLICY_UPLOAD] a no-op without LLAMA_MOE_HOT_UP_MIB_STEP
                cut = true; // [TAG_FN_L4_EXIT] stop / idle: the job is dropped whole
                break;
            }
            mc->up_bytes.fetch_add(sz, std::memory_order_relaxed);
            hot_pace(mc, sz); // [TAG_FN_L18_CHURN_PACE]
            if (sz > mc->stage_size) {
                upload_slice(dsts[k], srcs[k], j.expert, j.slot);
                continue;
            }
            if (used + sz > mc->stage_size) {
                flush();
            }
            hot_copy_gated(mc, mc->stage_ptr + used, (const char *) srcs[k]->data + (size_t) j.expert*sz, sz, true); // [TAG_FN_L17_START_NT] [TAG_FN_L18_CHURN_GATE]
            st.push_back({ dsts[k], (size_t) j.slot*dsts[k]->nb[2], sz, used });
            used += sz;
        }
        if (cut) {
            break;
        }
    }
    flush();
    return n_done;
}

// uploads the queued slices: mmap -> pinned staging -> the upload backend's own stream, then marks them done
void hot_adapt_worker(moe_cache * mc) {
    llama_fn_thread_name("hot-upload"); // [TAG_FN_L18_FIRSTANS_STALL]
    if (mc->l22_refill) {
        fn_l22_mem_prio_low(); // [TAG_FN_L22_REFILL] M2.4: the pages it faults in sit low on the standby list
    }
    std::vector<upload_job> batch;
    for (;;) {
        {
            std::unique_lock<std::mutex> lk(mc->wmtx);
            // [TAG_FN_L4_EXIT] an idle owner's queue waits for its next step (or a drain)
            mc->wcv.wait(lk, [mc]() { return mc->stop || (!mc->todo.empty() && !mc->up_yield.load()); });
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
            // [TAG_FN_L4_EXIT] at most UP_INFLIGHT_MAX bytes per batch (at least one job), so the worker is back at the
            // queue within one bounded chunk: an idle owner or a stop finds no long copy run in flight, and a long queue
            // (the tail's refill with LLAMA_FN_CBUF_ASYNC=1) is published as it lands
            {
                const size_t cap = UP_INFLIGHT_MAX;
                size_t take  = 0;
                size_t bytes = 0;
                while (take < n) {
                    const layer_state & ls = mc->layers[mc->todo[take].layer_idx];
                    const size_t jb = ls.pub.up_src->nb[2] + ls.pub.gate_src->nb[2] + ls.pub.down_src->nb[2];
                    if (take > 0 && bytes + jb > cap) {
                        break;
                    }
                    bytes += jb;
                    take++;
                }
                n = take;
            }
            batch.assign(mc->todo.begin(), mc->todo.begin() + (std::ptrdiff_t) n);
            mc->todo.erase(mc->todo.begin(), mc->todo.begin() + (std::ptrdiff_t) n);
            mc->worker_busy = true; // [TAG_FN_R1_PFS_LEND]
        }
        llama_fn_stall_mark("hot upload batch begins (jobs)", (int64_t) batch.size()); // [TAG_FN_L18_FIRSTANS_STALL]
        // [TAG_FN_L18_CHURN_AB] pipe: the staged path in mode 0 (both buffers are allocated)
        const bool pipe = mc->up_pipe && (mc->l18_ab != 1 || mc->l18_ab_mode.load(std::memory_order_relaxed) == 1);
        const size_t n_done = pipe ? hot_upload_pipe(mc, batch) : hot_upload_staged(mc, batch); // [TAG_FN_L3_POLICY_UPLOAD]
        llama_fn_stall_mark("hot upload batch ends (done)", (int64_t) n_done);
        std::lock_guard<std::mutex> lk(mc->wmtx);
        for (size_t i = 0; i < batch.size(); ++i) {
            upload_job j = batch[i];
            j.failed = j.failed || i >= n_done; // [TAG_FN_L4_EXIT] cut (stop, or a bucket wait at idle): the slot comes back empty
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
        if (lp->pub.n_slots == 0 || lp->out()) { // [TAG_FN_R1_PFS_LEND] a lent layer's slots hold the stream's bytes
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
        if (mc->l17_pre > 0) {
            // [TAG_FN_L17_START_PRE] llama_moe_hot_turn_start folds the prompt at its end; a prompt it did not see folds here
            // at the first decode step, with the same weight
            if (!decode_step) {
                continue;
            }
            llama_moe_decay_fold_seed(ls.dcnt, ls.dseed, ls.dseed_tok > 0.0f ? mc->l17_sn/ls.dseed_tok : 0.0f);
        } else if (mc->sd_norm > 0.0f) {
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

    // [TAG_FN_L4_MEM_UPLOAD] the tail's refill lands first: while its jobs are in flight (at most l4_hold_ms after the
    // restore; a failed job never counts down) the pass queues no swap, the counts decay as usual
    if (mc->l4_upload && mc->cb_async_left > 0 && (ggml_time_us() - mc->cb_async_t0)/1000 < mc->l4_hold_ms) {
        if (mc->l4_held++ < 3) {
            LLAMA_LOG_INFO("moe-hot: [TAG_FN_L4_MEM_UPLOAD] pass %" PRIu64 " held: %zu refill uploads of the tail in flight\n",
                    mc->dc_passes, mc->cb_async_left);
        }
        for (auto & ls : mc->layers) {
            llama_moe_decay_apply(ls.dcnt, mc->dc.decay);
        }
        return;
    }

    // [TAG_FN_L3_POLICY_UPLOAD] with a rate limit a pass queues at most what the bucket gives the worker until the next
    // pass, minus what is still queued: evicted slots must not wait empty behind a growing queue
    // [TAG_FN_L3_POLICY_BURST] LLAMA_MOE_HOT_BURST_MIB: past the pass budget, strong pairs (a free slot, or a candidate over
    // LLAMA_MOE_HOT_BURST_RATIO x its victim) take up to that many bytes more, so a shifted working set refills fast
    // while the steady state keeps the small budget
    size_t pass_bytes  = mc->dc_bytes;
    // [TAG_FN_L15_BURST] the burst only in its window after a start or a big prompt
    const bool burst_now = mc->dc_burst_passes < 0 || mc->dc_burst_left > 0;
    if (mc->dc_burst_left > 0) {
        mc->dc_burst_left--;
    }
    mc->l18_pace_live.store(!(burst_now && mc->dc_burst > 0), std::memory_order_relaxed); // [TAG_FN_L18_CHURN_PACE]
    if (mc->l18_gate > 0 && (mc->dc_passes == 32 || mc->dc_passes % 1024 == 0)) { // [TAG_FN_L18_CHURN_GATE]
        uint64_t gw = 0, gus = 0;
        llama_moe_bridge_gate_stats(&gw, &gus);
        LLAMA_LOG_INFO("moe-hot: [TAG_FN_L18_CHURN_GATE] pass %" PRIu64 ": %" PRIu64 " gate waits, %.1f ms in all, %.0f MiB uploaded\n",
                mc->dc_passes, gw, gus/1000.0, mc->up_bytes.load()/1048576.0);
    }
    // [TAG_FN_L18_CHURN_MIB] the steady state's pass budget once the start's burst window is closed (the window keeps the
    // small one, so the burst's strong pairs are not queued behind a big pass)
    if (mc->l18_mib > 0 && !(burst_now && mc->dc_burst > 0) &&
            (mc->l18_ab != 2 || mc->l18_ab_mode.load(std::memory_order_relaxed) == 1)) { // [TAG_FN_L18_CHURN_AB]
        pass_bytes = mc->l18_mib;
    }
    size_t total_bytes = pass_bytes + (burst_now ? mc->dc_burst : 0);
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

    std::vector<llama_moe_decay_swap> swaps;
    std::vector<uint8_t> slot_busy;
    std::vector<uint8_t> expert_busy;
    for (int li = 0; li < (int) mc->layers.size(); ++li) {
        const layer_state & ls = mc->layers[li];
        if (ls.pub.n_slots == 0 || ls.dcnt.empty() || ls.out()) { // [TAG_FN_R1_PFS_LEND] no upload into a lent layer
            continue;
        }
        if (ls.pool >= 0) {
            continue; // [TAG_FN_L3_POLICY_POOL] below, per pool
        }
        slot_busy.assign(ls.pub.n_slots, 0);
        const llama_moe_host_ready_fn ready = g_host_ready.load(); // [TAG_FN_L15_TIER]
        void * ready_ud = g_host_ready_ud.load();
        for (int32_t s = 0; s < ls.pub.n_slots; ++s) {
            slot_busy[s] = ls.slot_in_flight[s] || (!ls.slot_out.empty() && ls.slot_out[s]) ? 1 : 0; // [TAG_FN_L10_SLOTLEND]
            // [TAG_FN_L15_TIER] a resident whose host copy is not locked in RAM stays until the tier has locked it
            if (!slot_busy[s] && ready && ls.slot_expert[s] >= 0 && !ready(ls.pub.up_src, ls.slot_expert[s], ready_ud)) {
                slot_busy[s] = 1;
            }
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
            skip = ls.out() || ls.dcnt.empty();
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
            const llama_moe_host_ready_fn ready = g_host_ready.load(); // [TAG_FN_L15_TIER]
            if (!sb[s] && ready && se[s] >= 0 && !ready(ow.pub.up_src, se[s], g_host_ready_ud.load())) {
                sb[s] = 1;
            }
        }
        // a pool's pairs come in falling gain and the strong ones first, so the pass takes a prefix of them: no more than
        // its byte budget holds (exact, and the sort runs over that many instead of every candidate of the pool)
        const layer_state & l0 = mc->layers[P.layers[0]];
        const size_t pb = l0.pub.up_src->nb[2] + l0.pub.gate_src->nb[2] + l0.pub.down_src->nb[2];
        const size_t max_pairs = std::max(pass_bytes, total_bytes)/std::max<size_t>(1, pb) + 1;
        std::vector<llama_moe_decay_swap> ps;
        llama_moe_decay_pairs_pool(cnts, busy, sl, se, sb, mc->dc, ps, max_pairs);
        for (auto w : ps) {
            w.layer        = (int) P.layers[w.layer];
            w.victim_layer = w.victim_layer >= 0 ? (int) P.layers[w.victim_layer] : -1;
            swaps.push_back(w);
        }
    }
    llama_moe_decay_order(swaps);

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
        if (j.st && mc->fa18_st_left > 0 && --mc->fa18_st_left == 0) { // [TAG_FN_L18_FIRSTANS_STATE]
            LLAMA_LOG_INFO("moe-hot: [TAG_FN_L18_FIRSTANS_STATE] the saved state's %zu residents (%.2f GiB) are in, the last "
                    "published %.0f ms after its start\n", mc->fa18_st_n, mc->fa18_st_bytes/1073741824.0,
                    (ggml_time_us() - mc->fa18_st_t0)/1000.0);
            llama_fn_stall_mark("saved state: all residents published");
        }
        // [TAG_FN_L22_ENDOVERLAP] the asynchronous unlend's refill: counted down as its jobs come back
        if (j.ul && mc->l22_ul_left > 0 && --mc->l22_ul_left == 0) {
            llama_fn_l22_log("switch", "what=unlend_async_landed ms=%.1f bytes_mib=%.0f", (ggml_time_us() - mc->l22_ul_t0)/1000.0,
                    mc->l22_ul_bytes/1048576.0);
        }
        // [TAG_FN_L10_SLOTLEND] an upload into a slot that went out meanwhile: as failed (the slot stays empty)
        if (j.failed || (!ls.slot_out.empty() && ls.slot_out[j.slot])) { // [TAG_FN_L3_POLICY_UPLOAD] nothing was copied: the slot is free again, the expert stays cold
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
        if ((size_t) j.slot < ls.slot_dirty.size()) {
            ls.slot_dirty[j.slot] = 0; // [TAG_FN_L22_LENDCLASS] its bytes landed
        }
        hot_adapt_set_entry(mc, ls, j.expert, ls.out() ? ls.pub.n_slots : j.slot);
        mc->ad_admitted++;
        // [TAG_FN_L3_VRAM_CBUF] the asynchronous refill of the tail landed
        if (j.cb && mc->cb_async_left > 0 && --mc->cb_async_left == 0 && (mc->cb_n_res <= 2 || mc->cb_n_res % 16 == 0)) {
            LLAMA_LOG_INFO("moe-hot: [TAG_FN_L3_VRAM_CBUF] restore %" PRIu64 ": the tail's resident experts are back %.1f ms "
                    "after the restore\n", mc->cb_n_res, (ggml_time_us() - mc->cb_async_t0)/1000.0);
        }
    }
    mc->done.clear();
}

// the owning context's step boundary, after its compute has been synchronized: publish finished uploads, roll the
// windows, then evict and queue new uploads (byte-capped). Nothing here runs while a graph of this context runs.
void hot_adapt_step_impl(moe_cache * mc) {
    // [TAG_FN_L4_EXIT] the owner computes again: an idle pause ends (paused changes under step_mtx and wmtx)
    if (mc->paused) {
        std::lock_guard<std::mutex> lk(mc->wmtx);
        mc->paused = false;
        hot_set_yield(mc);
        mc->wcv.notify_all();
    }
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
            if (ls.pub.n_slots == 0 || ls.out() || ls.pool >= 0) { // [TAG_FN_R1_PFS_LEND] no upload into a lent layer
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
    std::lock_guard<std::mutex> slk(mc->step_mtx); // [TAG_FN_L3_POLICY_STATE]
    hot_turn_join(mc, mc->fa18_ovl < 2); // [TAG_FN_L18_FIRSTANS_OVERLAP] a pass in flight lands first (2: once its copies did)
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
        // [TAG_FN_L4_MEM_UPLOAD] the refill first, and the pinned halves by default (staging that stays in the CPU caches,
        // the memcpy overlapping the DMA)
        if (llama_fn_l3_flag(model, "LLAMA_FN_L4_UPLOAD")) {
            mc->l4_upload = true;
            mc->up_pipe   = true;
            if (const char * e = llama_fn_env(model, "LLAMA_FN_L4_UPLOAD_HOLD_MS")) {
                mc->l4_hold_ms = std::max(0, atoi(e));
            }
        }
        if (const char * e = llama_fn_env(model, "LLAMA_MOE_HOT_UP_PIPE")) {
            mc->up_pipe = atoi(e) != 0;
        }
        if (const char * e = llama_fn_env(model, "LLAMA_MOE_HOT_UP_STAGE_MIB")) {
            mc->up_half = (size_t) std::max(2, std::min(256, atoi(e))) << 20;
            mc->up_half = std::min(mc->up_half, UP_INFLIGHT_MAX/2); // [TAG_FN_L4_EXIT] both halves in flight stay bounded
        }
        if (const char * e = llama_fn_env(model, "LLAMA_MOE_HOT_UP_MIB_STEP")) {
            mc->up_rate = (size_t) std::max(0, atoi(e)) << 20;
        }
        // [TAG_FN_L18_CHURN_AB] before the staging buffers are made: the pipe A/B needs both
        if (const char * e = llama_fn_env(model, "LLAMA_FN_L18_CHURN_AB"); e && e[0]) {
            mc->l18_ab = strcmp(e, "pipe") == 0 ? 1 : strcmp(e, "mib") == 0 ? 2 : strcmp(e, "gate") == 0 ? 3 : 0;
            if (mc->l18_ab == 1) {
                mc->up_pipe = true;
            }
            LLAMA_LOG_INFO("moe-hot: [TAG_FN_L18_CHURN_AB] %s flips at every 256-step statistics window (mode 0 off, 1 on; needs "
                    "LLAMA_MOE_HOT_STATS)\n", mc->l18_ab == 1 ? "staged / pinned halves" : mc->l18_ab == 2 ? "the pass budget" :
                    mc->l18_ab == 3 ? "the gate" : "nothing");
        }
        // [TAG_FN_L22_REFILL] pinned refill sources, the upload / refill threads at low memory priority, the refill set's
        // prefetch at a prompt's end (before the worker starts, so it sees the flag)
        mc->l22_refill = llama_fn_l22_switch_on(llama_fn_env(model, "LLAMA_FN_L22_REFILL"));
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
    if (!mc->up_pipe || mc->l18_ab == 1) { // [TAG_FN_L18_CHURN_AB] both buffers for the pipe A/B
        mc->stage_size = std::min(std::max<size_t>(mc->ad_bytes, 16u << 20), UP_INFLIGHT_MAX); // [TAG_FN_L4_EXIT] one flush in flight
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
        if (const char * e = llama_fn_env(model, "LLAMA_MOE_HOT_BURST_PASSES")) { // [TAG_FN_L15_BURST]
            mc->dc_burst_passes = atoi(e);
        }
        mc->dc_burst_left = mc->dc_burst_passes;
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
            if (strcmp(e, "cache") == 0) { // [TAG_FN_L18_FIRSTANS_STATE] the user's cache folder, one file per model file
                mc->st_path = mpath.empty() ? std::string() : hot_state_cache_path(mpath);
            }
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
            mc->fa18_state = llama_fn_l3_flag(model, "LLAMA_FN_L18_FIRSTANS_STATE"); // [TAG_FN_L18_FIRSTANS_STATE]
        }
    }
    // [TAG_FN_L17_START_*] the answer-start levers of round 17 (qwen4exp only; every one off by default)
    if (l3 && mc->dc_on) {
        if (const char * e = llama_fn_env(model, "LLAMA_FN_L17_START_PRE"); e && atoi(e) > 0) {
            if (mc->dc.seed > 0.0f) { // the observer (or the seed node) counts the prompt's routing only with a seed
                mc->l17_pre = (size_t) std::min(65536, atoi(e)) << 20;
                if (const char * v = llama_fn_env(model, "LLAMA_FN_L17_START_TD")) { mc->l17_td = std::max(0.0f, std::min(1.0f, (float) atof(v))); }
                if (const char * v = llama_fn_env(model, "LLAMA_FN_L17_START_SN")) { mc->l17_sn = std::max(0.0f, (float) atof(v)); }
                if (const char * v = llama_fn_env(model, "LLAMA_FN_L17_START_THREADS")) { mc->l17_threads = std::max(1, std::min(64, atoi(v))); }
            } else {
                LLAMA_LOG_WARN("moe-hot: [TAG_FN_L17_START_PRE] needs LLAMA_MOE_HOT_SEED > 0 (the prompt's routing) - off\n");
            }
        }
        mc->l17_nt          = llama_fn_l3_flag(model, "LLAMA_FN_L17_START_NT");
        mc->fa18_ovl        = mc->l17_pre > 0 ? std::max(0, std::min(2, llama_fn_l3_int(model, "LLAMA_FN_L18_FIRSTANS_OVERLAP", 0))) : 0; // [TAG_FN_L18_FIRSTANS_OVERLAP]
        if (const int bp = llama_fn_l3_int(model, "LLAMA_FN_L17_START_BURST", 0); bp > 0) {
            mc->l17_burst_start = true; // its value is the window (LLAMA_MOE_HOT_BURST_PASSES), opened once now
            mc->dc_burst_passes = bp;
            mc->dc_burst_left   = bp;
        }
        if (!mc->st_path.empty()) {
            mc->l17_st_every_us = (int64_t) std::max(0, llama_fn_l3_int(model, "LLAMA_FN_L17_START_STATE", 0))*1000000;
        }
        if (mc->l17_pre > 0 || mc->l17_nt || mc->l17_burst_start || mc->l17_st_every_us > 0) {
            LLAMA_LOG_INFO("moe-hot: [TAG_FN_L17_START] prompt-end pass: %s; non-temporal staging copies: %s; burst window at the "
                    "start only: %s; state saved when idle: %s\n",
                    mc->l17_pre > 0 ? format("<= %zu MiB, counts x %.2f, the prompt as %.0f decode steps, %d copy threads",
                            mc->l17_pre >> 20, mc->l17_td, mc->l17_sn, mc->l17_threads).c_str() : "off",
                    mc->l17_nt ? "on" : "off", mc->l17_burst_start ? format("on (%d passes)", mc->dc_burst_passes).c_str() : "off",
                    mc->l17_st_every_us > 0 ? format("every >= %lld s", (long long) (mc->l17_st_every_us/1000000)).c_str() : "off");
        }
    }
    // [TAG_FN_L18_CHURN_*] the steady state's upload levers of round 18 (qwen4exp only; every one off by default)
    if (l3 && mc->dc_on) {
        // the steady state: no burst window open (the pass keeps this up to date)
        mc->l18_pace_live.store(!(mc->dc_burst > 0 && (mc->dc_burst_passes < 0 || mc->dc_burst_left > 0)));
        // [TAG_FN_L18_CHURN_PACE] the steady state's uploads as a paced stream
        if (const int mibs = llama_fn_l3_int(model, "LLAMA_FN_L18_CHURN_PACE", 0); mibs > 0) {
            mc->l18_pace     = (double) mibs*1048576.0/1e6;
            mc->l18_pace_cap = std::max<size_t>((size_t) 8 << 20, mc->up_pipe ? 2*mc->up_half : 0);
            LLAMA_LOG_INFO("moe-hot: [TAG_FN_L18_CHURN_PACE] uploads of the steady state at <= %d MiB/s (bucket %zu MiB); full "
                    "speed in the burst window and when a prompt drains the queue\n", mibs, mc->l18_pace_cap >> 20);
        }
        // [TAG_FN_L18_CHURN_MIB]
        if (const int mib = llama_fn_l3_int(model, "LLAMA_FN_L18_CHURN_MIB", 0); mib > 0) {
            mc->l18_mib = (size_t) std::min(4096, mib) << 20;
            LLAMA_LOG_INFO("moe-hot: [TAG_FN_L18_CHURN_MIB] <= %zu MiB uploads per pass once the burst window is closed (%zu MiB "
                    "in it)\n", mc->l18_mib >> 20, mc->dc_bytes >> 20);
        }
        // [TAG_FN_L18_CHURN_GATE]
        if (const int kib = llama_fn_l3_int(model, "LLAMA_FN_L18_CHURN_GATE", 0); kib > 0) {
            mc->l18_gate   = (size_t) std::max(16, std::min(65536, kib)) << 10;
            mc->l18_settle = std::max(0, std::min(1000, llama_fn_l3_int(model, "LLAMA_FN_L18_CHURN_GATE_SETTLE", 0)));
            llama_moe_bridge_gate_enable(true);
            LLAMA_LOG_INFO("moe-hot: [TAG_FN_L18_CHURN_GATE] uploads copied in %zu KiB pieces between the CPU expert jobs (a "
                    "piece and a DMA start only while no job runs, a DMA %d us after the last job's end at the earliest, at most "
                    "%d us of waiting each; the burst window at full speed)\n", mc->l18_gate >> 10, mc->l18_settle, L18_GATE_MAX_US);
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

// [TAG_FN_L4_MEM_PROMPT]
int llama_moe_hot_graph_max_t() {
    const moe_cache * mc = g_cache;
    return mc && mc->hot ? std::max(mc->hot_max_t, mc->hot_short_t) : 0;
}

// [TAG_FN_L19_TTFT_PHOT]
int llama_moe_hot_prompt_max_t() {
    const moe_cache * mc = g_cache;
    return mc && mc->hot ? mc->hot_prompt_t : 0;
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

// [TAG_FN_L3_VRAM_CBUF] ---------------------------------------------------------------------------------------------

namespace {

// the device backend's virtual-memory buffer functions (ggml-cuda), all or none
struct hot_vmm_fns {
    size_t                (*gran)(ggml_backend_buffer_type_t)               = nullptr;
    ggml_backend_buffer_t (*alloc)(ggml_backend_buffer_type_t, size_t)      = nullptr;
    bool                  (*map)(ggml_backend_buffer_t, size_t, size_t)     = nullptr;
    bool                  (*unmap)(ggml_backend_buffer_t, size_t, size_t)   = nullptr;
};

hot_vmm_fns hot_vmm_lookup(ggml_backend_buffer_type_t buft) {
    hot_vmm_fns f;
    ggml_backend_dev_t dev = buft ? ggml_backend_buft_get_device(buft) : nullptr;
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (!reg) {
        return f;
    }
    f.gran  = (decltype(f.gran))  ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_granularity");
    f.alloc = (decltype(f.alloc)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_buffer_alloc");
    f.map   = (decltype(f.map))   ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_buffer_map");
    f.unmap = (decltype(f.unmap)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_buffer_unmap");
    if (!f.gran || !f.alloc || !f.map || !f.unmap) {
        f = hot_vmm_fns();
    }
    return f;
}

// [TAG_FN_L3_VRAM_ACCOUNT] device use (all processes) on buft's device while LLAMA_FN_VRAM_ACCOUNT=1, else 0
size_t hot_acc_used(ggml_backend_buffer_type_t buft) {
    static const bool on = [] {
        const char * e = getenv("LLAMA_FN_VRAM_ACCOUNT");
        return e && atoi(e) != 0;
    }();
    ggml_backend_dev_t dev = on && buft ? ggml_backend_buft_get_device(buft) : nullptr;
    if (!dev) {
        return 0;
    }
    size_t mem_free = 0, mem_total = 0;
    ggml_backend_dev_memory(dev, &mem_free, &mem_total);
    return mem_total > mem_free ? mem_total - mem_free : 0;
}

size_t hot_round_up(size_t x, size_t g) {
    return g > 0 ? (x + g - 1) / g * g : x;
}

// [TAG_FN_L8_KVLEND] the KV cache's request for the next hot set init (llama_moe_hot_set_kv_lend): the bytes it may take
// back while its context grows, the bytes to keep between the KV layers and the tail (the prefill stream's range), and the
// trunk's layer count (the MTP block's layers never lend: the draft context's graphs read them)
size_t g_kv_want    = 0;
size_t g_kv_keep    = 0;
int    g_kv_n_trunk = 0;
// [TAG_FN_L21_BIGUB] the bytes the next hot set init puts in the borrow group of the slot lend (0: none)
size_t g_kv_borrow  = 0;
bool   g_kv_borrow_1seq = false; // [TAG_FN_L22_BIGUB2] the borrow is the 1-sequence big reserve's extra

// [TAG_FN_L10_SLOTLEND] a usable slot keeps this many bytes after it mapped (a kernel that reads a little past a slot's end
// never reaches a segment the KV cache holds)
constexpr size_t SL_SLACK = (size_t) 64 << 10;

// the group's tensors (layers li0.. of mc->layers) on reserved virtual memory in creation order (the tables, then each
// layer's up / gate / down), the first tail layer moved up to a granule boundary plus a guard granule (and one more after
// the end). The tail is whole layers from the top,
// at least tail_bytes, and leaves one layer or more below it. The base and the tail are mapped apart, so the tail can give
// its VRAM back alone. Returns the buffer, cleared; nullptr leaves nothing allocated (the tensors may point into a freed
// buffer then: the caller frees their contexts).
ggml_backend_buffer_t hot_vmm_place(moe_cache * mc, ggml_context * ctx_d, ggml_backend_buffer_type_t buft, size_t li0,
        size_t tail_bytes, const hot_vmm_fns & vmm, size_t gran) {
    const size_t align = std::max<size_t>(1, ggml_backend_buft_get_alignment(buft));
    auto tsize = [&](const ggml_tensor * t) {
        return hot_round_up(ggml_backend_buft_get_alloc_size(buft, t), align);
    };
    const size_t n = mc->layers.size() - li0;
    std::vector<size_t> layer_bytes;
    for (size_t i = 0; i < n; ++i) {
        const layer_state & ls = mc->layers[li0 + i];
        layer_bytes.push_back(tsize(ls.pub.up_c) + tsize(ls.pub.gate_c) + tsize(ls.pub.down_c));
    }
    const int k = llama_fn_cbuf_first_tail_layer(layer_bytes, tail_bytes);
    if (k < 1) {
        LLAMA_LOG_WARN("moe-hot: [TAG_FN_L3_VRAM_CBUF] a tail of %.0f MiB needs every one of the %zu layers of the hot set\n",
                tail_bytes/1048576.0, n);
        return nullptr;
    }
    const ggml_tensor * first_tail = mc->layers[li0 + k].pub.up_c;

    // [TAG_FN_L8_KVLEND] the KV layers [kv0, kv1): the first trunk layers, whole layers of at least g_kv_want bytes, each
    // starting at a granule boundary after a guard granule (the guard belongs to the region below it), with at least
    // g_kv_keep bytes of layers between them and the tail
    int kv0 = -1;
    int kv1 = -1;
    if (g_kv_want > 0 && !mc->sl_req) {
        int i = 0;
        while (i < k && mc->layers[li0 + i].pub.il >= g_kv_n_trunk) {
            i++;
        }
        size_t sum = 0;
        int    j   = i;
        while (j < k && sum < g_kv_want) {
            sum += layer_bytes[j];
            j++;
        }
        size_t above = 0;
        for (int t = j; t < k; ++t) {
            above += layer_bytes[t];
        }
        if (sum >= g_kv_want && j > i && above >= g_kv_keep + 2*gran) {
            kv0 = i;
            kv1 = j;
        } else {
            LLAMA_LOG_WARN("moe-hot: [TAG_FN_L8_KVLEND] %.0f MiB of KV layers (above them %.0f MiB, %.0f MiB wanted) do not fit "
                    "below the stream's range and the tail - the fit runs again without the KV lend\n", sum/1048576.0,
                    above/1048576.0, g_kv_want/1048576.0);
            return nullptr;
        }
    }
    std::vector<size_t> kv_start((size_t) std::max(0, kv1 - kv0) + 1, 0); // region starts, and the base start last

    std::vector<std::pair<ggml_tensor *, size_t>> place;
    size_t off = 0;
    size_t lo  = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx_d); t != nullptr; t = ggml_get_next_tensor(ctx_d, t)) {
        if (t->view_src != nullptr) {
            continue;
        }
        if (kv0 >= 0) {
            for (int i = kv0; i <= kv1; ++i) {
                if (i < (int) n && t == mc->layers[li0 + i].pub.up_c) {
                    off = hot_round_up(off, gran) + gran;
                    kv_start[(size_t) (i - kv0)] = off;
                }
            }
        }
        if (t == first_tail) {
            // one granule of guard below the tail (mapped with the base): a kernel that reads a little past the last
            // base layer's slots never reaches the tail's addresses, which have no memory while the tail is out
            off = hot_round_up(off, gran) + gran;
            lo  = off;
        }
        place.push_back({ t, off });
        off += tsize(t);
    }
    // and one past the last tail layer (mapped with the tail), as a cudaMalloc buffer's 2 MiB pages leave room after it
    const size_t total = hot_round_up(off, gran) + gran;
    if (lo == 0 || lo >= total) {
        return nullptr;
    }

    // [TAG_FN_L10_SLOTLEND] each lend layer's tensors: the granules from SL_SLACK past the last base slot up to the zero
    // slot, cut into the groups; all of them below the stream's range (g_kv_keep bytes under the tail) and a guard granule
    std::vector<std::vector<std::pair<size_t, size_t>>> sl_grp;
    std::vector<std::pair<size_t, size_t>>              sb_grp; // [TAG_FN_L21_BIGUB]
    if (mc->sl_req) {
        int n_grp = 16;
        if (const char * e = getenv("LLAMA_FN_L10_SLOTLEND_GROUPS"); e && atoi(e) > 0) {
            n_grp = std::min(256, atoi(e));
        }
        sl_grp.resize((size_t) n_grp);
        std::map<const ggml_tensor *, size_t> off_of;
        for (const auto & [t, o] : place) {
            off_of[t] = o;
        }
        const size_t lim = lo > g_kv_keep + 2*gran ? lo - g_kv_keep - 2*gran : 0;
        for (size_t i = 0; i < n; ++i) {
            layer_state & ls = mc->layers[li0 + i];
            if (ls.sl_base < 0) {
                continue;
            }
            const ggml_tensor * ts[3] = { ls.pub.up_c, ls.pub.gate_c, ls.pub.down_c };
            for (int k = 0; k < 3; ++k) {
                const size_t o  = off_of.at(ts[k]);
                const size_t sb = ts[k]->nb[2];
                ls.sl_toff[k] = o;
                ls.sl_sb[k]   = sb;
                const size_t a = hot_round_up(o + (size_t) ls.sl_base*sb + SL_SLACK, gran);
                const size_t b = (o + (size_t) ls.pub.n_slots*sb)/gran*gran;
                if (o + (size_t) (ls.pub.n_slots + 1)*sb > lim) {
                    LLAMA_LOG_WARN("moe-hot: [TAG_FN_L10_SLOTLEND] %s reaches into the stream's range or the tail - no slot lend\n",
                            ts[k]->name);
                    mc->sl_fail = true;
                    return nullptr;
                }
                if (ls.sl_bbase >= 0 && ls.sl_bbase < ls.sl_base) {
                    // [TAG_FN_L21_BIGUB] the borrow slots: from SL_SLACK past the last base slot to the granule below the
                    // first KV-lend slot's (a slot that touches either segment is out with it)
                    const size_t ba = hot_round_up(o + (size_t) ls.sl_bbase*sb + SL_SLACK, gran);
                    const size_t bb = (o + (size_t) ls.sl_base*sb)/gran*gran;
                    if (bb > ba && bb <= a) {
                        sb_grp.push_back({ ba, bb });
                    }
                }
                if (b <= a) {
                    continue;
                }
                const size_t ng = (b - a)/gran;
                for (size_t j = 0; j < (size_t) n_grp; ++j) {
                    const size_t s0 = a + gran*(j*ng/(size_t) n_grp);
                    const size_t s1 = a + gran*((j + 1)*ng/(size_t) n_grp);
                    if (s1 > s0) {
                        sl_grp[j].push_back({ s0, s1 });
                    }
                }
            }
        }
        size_t sum = 0;
        for (const auto & g : sl_grp) {
            for (const auto & r : g) {
                sum += r.second - r.first;
            }
        }
        if (sum < g_kv_want) {
            mc->sl_retry = true; // the caller sizes one slot more per lend layer
            return nullptr;
        }
        if (g_kv_borrow > 0) {
            // [TAG_FN_L21_BIGUB] the borrow group after the KV groups (short of its want: one slot more, as for the KV)
            size_t bsum = 0;
            for (const auto & r : sb_grp) {
                bsum += r.second - r.first;
            }
            if (bsum < g_kv_borrow && mc->sl_extra < 2) {
                mc->sl_retry = true;
                return nullptr;
            }
            if (bsum < g_kv_borrow) {
                // the borrow never costs the KV lend: none this time (the 16K prompt ubatches stay off)
                LLAMA_LOG_WARN("moe-hot: [TAG_FN_L21_BIGUB] the borrow segments come to %.0f of %.0f MiB%s: no borrow\n",
                        bsum/1048576.0, g_kv_borrow/1048576.0, g_kv_borrow_1seq ? " (1-seq reserve) [TAG_FN_L22_BIGUB2]" : "");
                g_kv_borrow = 0;
            } else {
                sl_grp.push_back(sb_grp);
            }
        }
    }
    ggml_backend_buffer_t buf = vmm.alloc(buft, total);
    if (!buf) {
        return nullptr;
    }
    bool mapped_ok = true;
    if (kv0 >= 0) {
        // [TAG_FN_L8_KVLEND] the tables (and the MTP block) below the first KV region, each KV region, the base, the tail
        const size_t base_lo = kv_start.back();
        mapped_ok = base_lo > 0 && base_lo < lo && vmm.map(buf, 0, kv_start[0]);
        for (int i = kv0; mapped_ok && i < kv1; ++i) {
            const size_t a = kv_start[(size_t) (i - kv0)];
            const size_t b = kv_start[(size_t) (i - kv0) + 1];
            mapped_ok = a < b && vmm.map(buf, a, b - a);
        }
        mapped_ok = mapped_ok && vmm.map(buf, base_lo, lo - base_lo);
    } else if (!sl_grp.empty()) {
        // [TAG_FN_L10_SLOTLEND] every segment mapped by itself (an unmap takes whole earlier maps), the rest between them
        std::vector<std::pair<size_t, size_t>> segs;
        for (const auto & g : sl_grp) {
            segs.insert(segs.end(), g.begin(), g.end());
        }
        std::sort(segs.begin(), segs.end());
        size_t c = 0;
        mapped_ok = true;
        for (const auto & s : segs) {
            if (s.first > c) {
                mapped_ok = mapped_ok && vmm.map(buf, c, s.first - c);
            }
            mapped_ok = mapped_ok && vmm.map(buf, s.first, s.second - s.first);
            c = s.second;
        }
        if (c < lo) {
            mapped_ok = mapped_ok && vmm.map(buf, c, lo - c);
        }
    } else {
        mapped_ok = vmm.map(buf, 0, lo);
    }
    if (!mapped_ok || !vmm.map(buf, lo, total - lo)) {
        ggml_backend_buffer_free(buf); // unmaps what was mapped
        return nullptr;
    }
    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(buf);
    for (const auto & [t, o] : place) {
        if (ggml_backend_tensor_alloc(buf, t, base + o) != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buf);
            return nullptr;
        }
    }
    for (ggml_tensor * t = ggml_get_first_tensor(ctx_d); t != nullptr; t = ggml_get_next_tensor(ctx_d, t)) {
        if (t->view_src != nullptr && t->buffer == nullptr && ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buf);
            return nullptr;
        }
    }
    // every slot tensor of a layer below the tail ends below it and every tail layer's starts in it (a slot tensor shared
    // by layers on both sides, e.g. a shape-class pool, would lose its VRAM under a layer that still reads it)
    for (size_t i = 0; i < n; ++i) {
        const layer_state & ls = mc->layers[li0 + i];
        for (const ggml_tensor * t : { ls.pub.up_c, ls.pub.gate_c, ls.pub.down_c }) {
            const uint8_t * p = (const uint8_t *) t->data;
            const bool ok = i < (size_t) k ? p + tsize(t) <= base + lo : p >= base + lo && p + tsize(t) <= base + total;
            if (!ok) {
                LLAMA_LOG_WARN("moe-hot: [TAG_FN_L3_VRAM_CBUF] %s crosses the tail boundary - no tail\n", t->name);
                ggml_backend_buffer_free(buf);
                return nullptr;
            }
        }
    }
    // [TAG_FN_L8_KVLEND] every slot tensor of a KV layer lies in its own region
    for (int i = kv0; kv0 >= 0 && i < kv1; ++i) {
        const layer_state & ls = mc->layers[li0 + i];
        const size_t a = kv_start[(size_t) (i - kv0)];
        const size_t b = kv_start[(size_t) (i - kv0) + 1];
        for (const ggml_tensor * t : { ls.pub.up_c, ls.pub.gate_c, ls.pub.down_c }) {
            const uint8_t * p = (const uint8_t *) t->data;
            if (!(p >= base + a && p + tsize(t) <= base + b)) {
                LLAMA_LOG_WARN("moe-hot: [TAG_FN_L8_KVLEND] %s crosses its KV region - no hot set\n", t->name);
                ggml_backend_buffer_free(buf);
                return nullptr;
            }
        }
    }
    ggml_init_params ip = { ggml_tensor_overhead()*(3 + (size_t) std::max(0, kv1 - kv0)), nullptr, true };
    ggml_context * cctx = ggml_init(ip);
    if (!cctx) {
        ggml_backend_buffer_free(buf);
        return nullptr;
    }
    mc->kv_bytes = 0;
    for (int i = kv0; kv0 >= 0 && i < kv1; ++i) {
        layer_state & ls = mc->layers[li0 + i];
        ls.kv_tail = true;
        ls.kv_lo   = kv_start[(size_t) (i - kv0)];
        ls.kv_hi   = kv_start[(size_t) (i - kv0) + 1];
        ls.kv_raw  = ggml_new_tensor_1d(cctx, GGML_TYPE_I8, (int64_t) (ls.kv_hi - ls.kv_lo));
        ggml_format_name(ls.kv_raw, "moe_hot_kv.%d", ls.pub.il);
        if (ggml_backend_tensor_alloc(buf, ls.kv_raw, base + ls.kv_lo) != GGML_STATUS_SUCCESS) {
            ggml_free(cctx);
            ggml_backend_buffer_free(buf);
            return nullptr;
        }
        mc->kv_bytes += ls.kv_hi - ls.kv_lo;
    }
    if (kv0 >= 0) {
        LLAMA_LOG_INFO("moe-hot: [TAG_FN_L8_KVLEND] %d layers (blk.%d..%d, %.0f MiB) can give their VRAM to the KV cache while "
                "its context grows (%.0f MiB wanted)\n", kv1 - kv0, mc->layers[li0 + kv0].pub.il, mc->layers[li0 + kv1 - 1].pub.il,
                mc->kv_bytes/1048576.0, g_kv_want/1048576.0);
    }
    if (!sl_grp.empty()) {
        // [TAG_FN_L10_SLOTLEND]
        mc->sl_raw = ggml_new_tensor_1d(cctx, GGML_TYPE_I8, (int64_t) lo);
        ggml_set_name(mc->sl_raw, "moe_hot_slotlend");
        if (ggml_backend_tensor_alloc(buf, mc->sl_raw, base) != GGML_STATUS_SUCCESS) {
            mc->sl_raw = nullptr;
            ggml_free(cctx);
            ggml_backend_buffer_free(buf);
            mc->sl_fail = true;
            return nullptr;
        }
        mc->sl_grp = sl_grp;
        mc->sl_grp_bytes.assign(sl_grp.size(), 0);
        mc->sl_given.assign(sl_grp.size(), 0);
        mc->sl_n_kv  = g_kv_borrow > 0 ? sl_grp.size() - 1 : sl_grp.size(); // [TAG_FN_L21_BIGUB]
        mc->kv_bytes = 0;
        mc->sb_bytes = 0;
        mc->sb_out   = 0;
        for (size_t j = 0; j < sl_grp.size(); ++j) {
            for (const auto & r : sl_grp[j]) {
                mc->sl_grp_bytes[j] += r.second - r.first;
            }
            (j < mc->sl_n_kv ? mc->kv_bytes : mc->sb_bytes) += mc->sl_grp_bytes[j];
        }
        int n_ll = 0, il_a = -1, il_b = -1, n_lend = 0;
        for (size_t i = 0; i < n; ++i) {
            layer_state & ls = mc->layers[li0 + i];
            if (ls.sl_base >= 0) {
                ls.slot_out.assign((size_t) ls.pub.n_slots, 0);
                il_a = il_a < 0 ? ls.pub.il : il_a;
                il_b = ls.pub.il;
                n_lend = ls.pub.n_slots - ls.sl_base;
                n_ll++;
            }
        }
        mc->sl_on = true;
        LLAMA_LOG_INFO("moe-hot: [TAG_FN_L10_SLOTLEND] %d layers (blk.%d..%d) hold %d slots more each over %.0f MiB that go to the "
                "KV cache in %zu groups while its context grows (%.0f MiB wanted); the other layers keep theirs\n", n_ll, il_a,
                il_b, n_lend, mc->kv_bytes/1048576.0, mc->sl_n_kv, g_kv_want/1048576.0);
        if (mc->sb_bytes > 0) {
            int n_b = 0;
            for (size_t i = 0; i < n; ++i) {
                const layer_state & ls = mc->layers[li0 + i];
                if (ls.sl_bbase >= 0 && ls.sl_base > ls.sl_bbase) {
                    n_b = ls.sl_base - ls.sl_bbase;
                    break;
                }
            }
            LLAMA_LOG_INFO("moe-hot: [TAG_FN_L21_BIGUB] the same layers' %d slots below those (%.0f MiB, %.0f MiB wanted%s) give their "
                    "VRAM to a 16K prompt ubatch's compute reserve while it runs; the slots per layer are as without it\n", n_b,
                    mc->sb_bytes/1048576.0, g_kv_borrow/1048576.0, g_kv_borrow_1seq ? ", 1-seq reserve [TAG_FN_L22_BIGUB2]" : "");
        }
    }
    ggml_tensor * raw = ggml_new_tensor_1d(cctx, GGML_TYPE_I8, (int64_t) (total - lo));
    ggml_set_name(raw, "moe_hot_tail");
    if (ggml_backend_tensor_alloc(buf, raw, base + lo) != GGML_STATUS_SUCCESS) {
        ggml_free(cctx);
        ggml_backend_buffer_free(buf);
        return nullptr;
    }
    ggml_backend_buffer_clear(buf, 0); // every slot, so the last one of each tensor is the zero slot
    for (size_t i = (size_t) k; i < n; ++i) {
        mc->layers[li0 + i].cb_tail = true;
    }
    mc->ctxs.push_back(cctx);
    mc->cb_ctx   = cctx;
    mc->cb_raw   = raw;
    mc->cb_buf   = buf;
    mc->cb_lo    = lo;
    mc->cb_hi    = total;
    mc->cb_out   = false;
    mc->cb_rel   = total;                                               // [TAG_FN_L20_BIGPROMPT_TAIL]
    mc->cb_gran  = gran;
    mc->cb_seg   = std::max(gran, ((size_t) 1 << 30) / gran * gran);    // the segments vmm.map makes (ggml-cuda: <= 1 GiB)
    mc->cb_map   = vmm.map;
    mc->cb_unmap = vmm.unmap;
    if (const char * e = getenv("LLAMA_FN_CBUF_ASYNC")) {
        mc->cb_async = atoi(e) != 0;
    }
    return buf;
}

} // namespace

size_t llama_moe_hot_vmm_granularity(const llama_model & model) {
    const std::vector<int> hl = hot_host_layers(model);
    if (hl.empty()) {
        return 0;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(model.layers[hl[0]].ffn_gate_inp->buffer);
    for (int il : hl) {
        if (ggml_backend_buffer_get_type(model.layers[il].ffn_gate_inp->buffer) != buft) {
            return 0; // the hot set would span devices
        }
    }
    const hot_vmm_fns f = hot_vmm_lookup(buft);
    return f.gran ? f.gran(buft) : 0;
}

bool llama_moe_hot_init(const llama_model & model, const void * owner, size_t budget_bytes, size_t tail_bytes) {
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

    // [TAG_FN_L3_VRAM_CBUF] a failed attempt with a tail leaves one more attempt to the caller (without a tail)
    struct retry_on_fail {
        bool armed;
        ~retry_on_fail() {
            if (armed) {
                g_init_done = false;
            }
        }
    } retry { tail_bytes > 0 };

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
    // [TAG_FN_L10_SLOTLEND] the KV lend slot by slot (LLAMA_FN_L10_SLOTLEND, qwen4exp): even slots filled by the adaptive
    // set only; anything else maps the KV cache whole (the init fails here, the caller fits again without the lend)
    const bool sl_req = g_kv_want > 0 && llama_fn_l3_flag(model, "LLAMA_FN_L10_SLOTLEND");
    if (sl_req && !even) {
        LLAMA_LOG_WARN("moe-hot: [TAG_FN_L10_SLOTLEND] the slot lend needs even slots (LLAMA_MOE_HOT_PROFILE=even) - the KV cache "
                "maps whole\n");
        return false;
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

    // [TAG_FN_L18_CHURN_LAYERS] LLAMA_FN_L18_CHURN_LAYERS=auto | w0,w1,... (by il; qwen4exp, even slots): the even layout's
    // slots per layer in proportion to a weight instead of one count for all, the same bytes. auto: the table below for the
    // 48 x 512 trunk of Qwen3.8-Flash-Next. Layers past the list keep weight 1. Placement only: the values are not touched.
    std::vector<float> l18_w;
    if (even && model.arch == LLM_ARCH_QWEN4EXP) {
        const char * e = llama_fn_env(model, "LLAMA_FN_L18_CHURN_LAYERS");
        if (e && e[0] && strcmp(e, "0") != 0 && strcmp(e, "off") != 0) {
            // the base slots that the marginal routing frequency gives each layer at 119 per layer, / 119 (l18 replay of the
            // l17 owner-like sessions t1 / t2 and the l3 long answers, E:/turbot-gates/flashnext/test/l18/churn/sim)
            static const float k_fn_layer_w[48] = {
                1.387f, 1.176f, 1.235f, 0.975f, 0.950f, 0.983f, 1.168f, 1.109f, 0.992f, 0.966f, 1.067f, 0.992f,
                1.311f, 1.092f, 1.050f, 0.723f, 0.849f, 1.076f, 1.017f, 0.966f, 0.975f, 0.983f, 1.210f, 0.815f,
                1.042f, 1.050f, 0.983f, 1.017f, 1.134f, 1.042f, 0.958f, 0.723f, 0.798f, 0.950f, 1.067f, 0.891f,
                0.882f, 1.067f, 1.303f, 0.714f, 0.723f, 0.941f, 1.092f, 0.790f, 1.025f, 0.874f, 0.958f, 0.908f,
            };
            std::vector<float> by_il;
            if (strcmp(e, "auto") == 0 || strcmp(e, "1") == 0) {
                bool fits = model.hparams.n_layer() == 48 && model.hparams.n_expert == 512;
                for (const auto & c : layers) {
                    fits = fits && c.l->ffn_up_exps->ne[2] == 512;
                }
                if (fits) {
                    by_il.assign(std::begin(k_fn_layer_w), std::end(k_fn_layer_w));
                }
            } else {
                by_il = llama_fn_parse_weights(e);
            }
            if (by_il.empty()) {
                LLAMA_LOG_WARN("moe-hot: [TAG_FN_L18_CHURN_LAYERS] LLAMA_FN_L18_CHURN_LAYERS=%s: no weights for this model - even slots\n", e);
            } else {
                for (const auto & c : layers) {
                    l18_w.push_back(c.il >= 0 && c.il < (int) by_il.size() ? by_il[c.il] : 1.0f);
                }
            }
        }
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
                fn_l21_dev_memory(&model, dev, &mem_free, &mem_total); // [TAG_FN_L21_VRAM_DXGI]
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

    // [TAG_FN_L3_VRAM_CBUF] a tail needs one device buffer type with virtual-memory buffers and one table tensor
    hot_vmm_fns vmm;
    size_t      vmm_gran = 0;
    if (tail_bytes > 0) {
        bool one = true;
        for (const auto & c : layers) {
            one = one && c.buft == layers[0].buft && c.l->ffn_up_exps->ne[2] == layers[0].l->ffn_up_exps->ne[2];
        }
        vmm      = hot_vmm_lookup(layers[0].buft);
        vmm_gran = one && vmm.gran ? vmm.gran(layers[0].buft) : 0;
        if (vmm_gran == 0) {
            LLAMA_LOG_WARN("moe-hot: [TAG_FN_L3_VRAM_CBUF] no virtual-memory buffers for one device here - hot set off\n");
            return false;
        }
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

    if (sl_req && pool_req) {
        LLAMA_LOG_WARN("moe-hot: [TAG_FN_L10_SLOTLEND] the slot lend does not work with slot pools - the KV cache maps whole\n");
        return false;
    }

    auto * mc = new moe_cache();
    mc->hot       = true;
    mc->hot_max_t = max_t;
    mc->sl_req    = sl_req; // [TAG_FN_L10_SLOTLEND]
    // [TAG_FN_R25_REFILL] the cache-resident copy pool for the synchronous refills (static and adaptive sets)
    mc->r25_on    = llama_fn_l3_flag(model, "LLAMA_FN_R25_REFILL");
    mc->r25_owner = owner;
    if (mc->r25_on) {
        mc->r25_threads = std::max(1, std::min(32, llama_fn_l3_int(model, "LLAMA_FN_R25_REFILL_THREADS", 8)));
        mc->r25_half    = (size_t) std::max(256, std::min(65536, llama_fn_l3_int(model, "LLAMA_FN_R25_REFILL_HALF_KIB", 2048))) << 10;
        mc->r25_verify  = std::max(0, llama_fn_l3_int(model, "LLAMA_FN_R25_VERIFY", 0));
        mc->r25_sb_max  = (size_t) std::max(0, llama_fn_l3_int(model, "LLAMA_FN_R25_SBSYNC_MIB", 1024)) << 20;
    }
    // [TAG_FN_L4_MEM_PROMPT] the short path's ubatches (<= 31 tokens, LLAMA_FN_L4_PROMPT) build the hot chain as well, so
    // the CPU computes only the cold experts of a short prompt; the decode accounting (prompt seed above max_t) is unchanged
    if (llama_fn_l3_flag(model, "LLAMA_FN_L4_PROMPT_HOT")) {
        mc->hot_short_t = 31;
    }
    // [TAG_FN_L19_TTFT_PHOT] LLAMA_FN_L19_TTFT_PHOT=1 (or =<n>: graphs of up to n tokens): a prompt graph on the CPU path
    // (wider than the graphs above, no prefill stream, below GGML_OP_OFFLOAD_MIN_BATCH) builds the hot chain as well, one
    // row per routed (token, expert) so the batched kernels (MMQ) serve it; the CPU skips the hot experts. Off = 0
    if (const int v = llama_fn_l3_int(model, "LLAMA_FN_L19_TTFT_PHOT", 0); v > 0) {
        mc->hot_prompt_t = v == 1 ? INT32_MAX : v;
        mc->phot_stats   = llama_fn_l3_flag(model, "LLAMA_FN_L19_TTFT_PHOT_STATS");
        LLAMA_LOG_WARN("moe-hot: [TAG_FN_L19_TTFT_PHOT] CPU-path prompt graphs of up to %d tokens compute the hot experts on "
                "the GPU (one row per routed expert), the CPU only the others\n", mc->hot_prompt_t);
    }

    auto free_all = [&]() {
        for (auto * b : mc->bufs) { ggml_backend_buffer_free(b); }
        for (auto * c : mc->ctxs) { ggml_free(c); }
        mc->bufs.clear();
        mc->ctxs.clear();
        mc->layers.clear();
        mc->by_up_src.clear();
        mc->tbl_all = nullptr;
        mc->pools.clear(); // [TAG_FN_L3_POLICY_POOL]
        mc->cb_buf  = nullptr; // [TAG_FN_L3_VRAM_CBUF] freed with mc->bufs / mc->ctxs
        mc->cb_ctx  = nullptr;
        mc->cb_raw  = nullptr;
    };

    // [TAG_FN_L3_POLICY_POOL] candidate layers that keep their own slot tensors even with pools: an integration point for a
    // lever that needs whole layers at the top of the device buffer (e.g. a tail of layers that gives its VRAM back)
    std::vector<bool> pool_excl(layers.size(), false);

    const size_t acc_used0 = layers.empty() ? 0 : hot_acc_used(layers[0].buft); // [TAG_FN_L3_VRAM_ACCOUNT]
    for (int attempt = 0; attempt < 32 && budget > 0; ++attempt) {
        std::vector<std::vector<int32_t>> hot_ids(layers.size());
        std::vector<int32_t>              n_slots_of(layers.size(), 0);
        std::vector<int32_t>              sl_base_of(layers.size(), -1); // [TAG_FN_L10_SLOTLEND]
        std::vector<int32_t>              sl_bbase_of(layers.size(), -1); // [TAG_FN_L21_BIGUB]
        size_t used = 0;
        // [TAG_FN_L3_VRAM_CBUF] the virtual-memory layout pads the tail start and the end to granules and adds a guard
        // granule below the tail and after it
        size_t pad_gran = 5;
        if (tail_bytes > 0 && g_kv_want > 0 && !sl_req) {
            // [TAG_FN_L8_KVLEND] each KV region adds up to two granules (alignment + guard); about one layer per 1/48 of the budget
            const size_t per_layer = std::max<size_t>(1, budget / std::max<size_t>(1, layers.size()));
            pad_gran += 2*(g_kv_want / per_layer + 2);
        }
        const size_t budget_slots = tail_bytes > 0 ? (budget > pad_gran*vmm_gran ? budget - pad_gran*vmm_gran : 0) : budget;
        if (even) {
            // [TAG_FN_AUTO] the same slot count in every host layer
            std::vector<size_t> bytes;
            int32_t n_exp_min = INT32_MAX;
            for (const auto & c : layers) {
                bytes.push_back(c.bytes);
                n_exp_min = std::min<int32_t>(n_exp_min, (int32_t) c.l->ffn_up_exps->ne[2]);
            }
            // [TAG_FN_L10_SLOTLEND] the lend: the trunk layers from the bottom that lie below the tail and the stream's range
            // (whole layers from the top, by bytes, plus 4 granules) take E slots more each, E sized for the KV cache's
            // unmapped bytes plus up to two granules and the slack per tensor; the even count comes from the rest
            // [TAG_FN_L18_CHURN_LAYERS] the count of each layer for a budget: even, or by the weights
            auto slots_for = [&](size_t bud) {
                return l18_w.empty() ? std::vector<int32_t>(layers.size(), llama_fn_even_slots(bytes, bud, n_exp_min)) :
                                       llama_fn_weighted_slots(bytes, bud, n_exp_min, l18_w);
            };
            std::vector<int> sl_layers;
            int32_t          sl_e = 0;
            int32_t          sl_bx = 0; // [TAG_FN_L21_BIGUB]
            if (sl_req) {
                std::vector<int> trunk;
                for (int li = 0; li < (int) layers.size(); ++li) {
                    if (layers[li].il < g_kv_n_trunk) {
                        trunk.push_back(li);
                    }
                }
                std::sort(trunk.begin(), trunk.end(), [&](int a, int b) { return layers[a].il < layers[b].il; });
                const std::vector<int32_t> n0 = slots_for(budget_slots > g_kv_want ? budget_slots - g_kv_want : 0);
                // one more layer: the final even count may come out a little lower than n0
                const size_t need_above = tail_bytes + g_kv_keep + 4*vmm_gran +
                    (trunk.empty() ? 0 : (size_t) (n0[trunk.back()] + 1)*layers[trunk.back()].bytes);
                size_t above = 0;
                int    top   = (int) trunk.size();
                while (top > 0 && above < need_above) {
                    --top;
                    above += (size_t) (n0[trunk[top]] + 1)*layers[trunk[top]].bytes;
                }
                size_t per = 0;
                for (int i = 0; i < top; ++i) {
                    sl_layers.push_back(trunk[i]);
                    per += layers[trunk[i]].bytes;
                }
                if (per > 0) {
                    // about one granule per tensor stays mapped at its segment ends (the start's slack, the zero slot's
                    // granule); a placement that comes out short asks for one slot more (sl_extra)
                    const size_t waste = sl_layers.size()*3*(vmm_gran + SL_SLACK);
                    sl_e = (int32_t) ((g_kv_want + waste + per - 1)/per) + mc->sl_extra;
                    if (g_kv_borrow > 0) {
                        // [TAG_FN_L21_BIGUB] the borrow: that many of each lend layer's base slots more lie over segments
                        // of their own (the slots per layer stay as they are: the learned set and the routing see no change)
                        sl_bx = (int32_t) ((g_kv_borrow + waste + sl_layers.size()*3*vmm_gran + per - 1)/per) + mc->sl_extra;
                    }
                }
            }
            // the base slots from the budget without the KV cache's unmapped bytes: what a fit with the KV cache mapped
            // whole would give (the lend's few granules of waste come out of the headroom)
            const size_t sl_base_cut = sl_e > 0 ? g_kv_want : 0;
            const std::vector<int32_t> nv = slots_for(budget_slots > sl_base_cut ? budget_slots - sl_base_cut : 0);
            for (size_t li = 0; li < layers.size(); ++li) {
                n_slots_of[li] = nv[li];
                used += (size_t) nv[li]*layers[li].bytes;
            }
            if (sl_e > 0) {
                for (int li : sl_layers) {
                    const int32_t e = std::min<int32_t>(sl_e, (int32_t) layers[li].l->ffn_up_exps->ne[2] - nv[li]);
                    if (e <= 0 || nv[li] <= 0) {
                        continue;
                    }
                    sl_base_of[li]  = nv[li];
                    sl_bbase_of[li] = sl_bx > 0 && nv[li] > sl_bx ? nv[li] - sl_bx : -1; // [TAG_FN_L21_BIGUB]
                    n_slots_of[li] += e;
                    used += (size_t) e*layers[li].bytes;
                }
            }
            if (!l18_w.empty()) {
                std::string s;
                for (size_t li = 0; li < layers.size(); ++li) {
                    s += format(" %d:%d", layers[li].il, nv[li]);
                }
                LLAMA_LOG_INFO("moe-hot: [TAG_FN_L18_CHURN_LAYERS] slots by layer weight (even %d):%s\n",
                        llama_fn_even_slots(bytes, budget_slots > sl_base_cut ? budget_slots - sl_base_cut : 0, n_exp_min), s.c_str());
            }
        } else {
            for (const auto & c : cands) {
                const size_t b = layers[c.li].bytes;
                if (used + b > budget_slots) {
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
        if (tail_bytes > 0) {
            // [TAG_FN_L3_VRAM_CBUF] the layers past the trunk (an MTP block) first: the tail and the stream's range below it
            // then hold trunk layers only, which no other context's graphs read (the draft context runs while the tail is out)
            const int n_trunk = (int) model.hparams.n_layer();
            for (auto & g : groups) {
                std::stable_partition(g.second.begin(), g.second.end(), [&](int li) { return layers[li].il >= n_trunk; });
            }
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
            const size_t li0 = mc->layers.size(); // [TAG_FN_L3_VRAM_CBUF] the first layer of this group
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
                ls.sl_base      = sl_base_of[li]; // [TAG_FN_L10_SLOTLEND]
                ls.sl_bbase     = sl_bbase_of[li]; // [TAG_FN_L21_BIGUB]
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
            ggml_backend_buffer_t buf = nullptr;
            if (tail_bytes > 0) {
                // [TAG_FN_L3_VRAM_CBUF] one group (checked above), on virtual memory with the tail mapped apart
                buf = tbl_all && groups.size() == 1 ? hot_vmm_place(mc, ctx_d, g.first, li0, tail_bytes, vmm, vmm_gran) : nullptr;
            } else {
                buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_d, g.first);
                if (buf) {
                    ggml_backend_buffer_clear(buf, 0); // every slot, so the last one of each tensor is the zero slot
                }
            }
            if (!buf) {
                ok = false;
                break;
            }
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
        if (!ok && mc->sl_retry) {
            // [TAG_FN_L10_SLOTLEND] the segments fell short: one slot more per lend layer, the same budget
            mc->sl_retry = false;
            if (++mc->sl_extra <= 4) {
                LLAMA_LOG_INFO("moe-hot: [TAG_FN_L10_SLOTLEND] the lend layers' segments fall short of the KV cache's want: "
                        "%d slot(s) more per lend layer\n", mc->sl_extra);
                continue;
            }
            mc->sl_fail = true;
        }
        if (!ok && mc->sl_fail) {
            break; // [TAG_FN_L10_SLOTLEND] logged: the caller maps the KV cache whole and fits again
        }
        if (!ok) {
            LLAMA_LOG_WARN("moe-hot: allocating %.0f MiB failed, retrying with 90%% of the budget\n", used/1048576.0);
            budget = (size_t) (budget*0.9);
            continue;
        }

        const size_t acc_used1 = acc_used0 ? hot_acc_used(layers[0].buft) : 0; // [TAG_FN_L3_VRAM_ACCOUNT]

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
        if (mc->cb_buf) {
            int n_tail = 0;
            int il_tail = -1;
            for (const auto & ls : mc->layers) {
                if (ls.cb_tail) {
                    il_tail = il_tail < 0 ? ls.pub.il : il_tail;
                    n_tail++;
                }
            }
            LLAMA_LOG_INFO("moe-hot: [TAG_FN_L3_VRAM_CBUF] the device buffer is on virtual memory: %.1f MiB, its tail %.1f MiB "
                    "(%d layers from blk.%d, asked %.1f MiB) at +%.1f MiB gives its VRAM back while a big prompt runs\n",
                    mc->cb_hi/1048576.0, (mc->cb_hi - mc->cb_lo)/1048576.0, n_tail, il_tail, tail_bytes/1048576.0,
                    mc->cb_lo/1048576.0);
        }

        if (const char * st = llama_fn_env(model, "LLAMA_MOE_HOT_STATS"); st && atoi(st) != 0) {
            mc->ad_stats = true;
            mc->ad_stats_l = atoi(st) >= 2; // [TAG_FN_L10_STATS2]
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
        if (acc_used0) {
            // [TAG_FN_L3_VRAM_ACCOUNT] what the hot set adds beyond its tensors: the buffers' own cost, then the adaptive
            // set's start (its upload backend and stream, the pinned staging)
            const size_t acc_used2 = hot_acc_used(layers[0].buft);
            LLAMA_LOG_INFO("moe-hot: [TAG_FN_L3_VRAM_ACCOUNT] device use %.1f MiB before the hot set, %+.1f MiB with its buffers "
                    "(%.1f MiB of tensors%s), %+.1f MiB more after the uploads and the adaptive set's start\n",
                    acc_used0/1048576.0, ((double) acc_used1 - (double) acc_used0)/1048576.0, vram/1048576.0,
                    mc->cb_buf ? ", virtual memory" : "", ((double) acc_used2 - (double) acc_used1)/1048576.0);
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
        retry.armed = false;
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

// [TAG_FN_L16_MTP_ASYNC]
bool llama_moe_cache_lru_on() {
    const moe_cache * mc = g_cache;
    return mc && !mc->hot;
}

// [TAG_FN_L3_HOST_STEP] the layer list is fixed once the set exists (only per-layer state changes in a step)
bool llama_moe_hot_has_layer_from(int il_min) {
    const moe_cache * mc = g_cache;
    if (!mc) {
        return false;
    }
    for (const auto & ls : mc->layers) {
        if (ls.pub.il >= il_min) {
            return true;
        }
    }
    return false;
}

// [TAG_FN_R4_ADAPT_DECAY]
void llama_moe_hot_save_now(const void * owner) {
    moe_cache * mc = g_cache;
    if (mc && mc->adapt && mc->owner == owner) {
        std::lock_guard<std::mutex> slk(mc->step_mtx); // [TAG_FN_L3_POLICY_STATE]
        hot_turn_join(mc); // [TAG_FN_L18_FIRSTANS_OVERLAP] a pass in flight lands first
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
// [TAG_FN_L3_VRAM_CBUF] "not hot" while either range holds the layer (ls.out()): the stream's lend or the tail
void hot_write_tables(moe_cache * mc, layer_state & ls) {
    const int64_t n_exp = ls.pub.host_table->ne[1];
    std::vector<int32_t> tbl((size_t) n_exp, ls.pub.n_slots);
    if (!ls.out()) {
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

// no upload goes into a layer that is out (lent or in the given-back tail) any more: queued ones are dropped (their slots
// were evicted already, so they come back empty), the worker puts its running batch down before anything overwrites a slot,
// and the batch it finished is published (slot_expert knows it, the tables of out layers stay "not hot")
void hot_drain_out(moe_cache * mc) {
    if (!mc->adapt) {
        return;
    }
    {
        std::unique_lock<std::mutex> lk(mc->wmtx);
        for (auto it = mc->todo.begin(); it != mc->todo.end(); ) {
            layer_state & ls = mc->layers[it->layer_idx];
            const bool sl_o = !ls.slot_out.empty() && ls.slot_out[it->slot]; // [TAG_FN_L10_SLOTLEND] a slot that went out
            if (ls.out() || sl_o) {
                ls.slot_in_flight[it->slot]     = false;
                ls.expert_in_flight[it->expert] = false;
                if (it->st && mc->fa18_st_left > 0) { // [TAG_FN_L18_FIRSTANS_STATE] dropped here: a tail slot keeps it (below)
                    mc->fa18_st_left--;
                }
                // [TAG_FN_L3_VRAM_CBUF] a tail slot keeps the expert it was getting: the next restore uploads it with the
                // others, so prompts in quick succession do not empty the tail (a lent layer's slot comes back empty)
                // [TAG_FN_L22_ENDOVERLAP] so does a slot of the stream's range whose asynchronous refill had not run yet
                if (it->ul && mc->l22_ul_left > 0) {
                    mc->l22_ul_left--;
                }
                if ((ls.cb_lent || ls.kv_lent || (it->ul && ls.lent)) && !sl_o) {
                    ls.slot_expert[it->slot]   = it->expert;
                    ls.expert_slot[it->expert] = it->slot;
                    if (ls.slot_dirty.size() < ls.slot_expert.size()) {
                        ls.slot_dirty.resize(ls.slot_expert.size(), 0);
                    }
                    ls.slot_dirty[it->slot] = 1; // [TAG_FN_L22_LENDCLASS] named, not uploaded
                } else if (ls.pool >= 0) {
                    mc->pools[ls.pool].slot_layer[it->slot] = -1; // [TAG_FN_L3_POLICY_POOL]
                }
                it = mc->todo.erase(it);
            } else {
                ++it;
            }
        }
        // [TAG_FN_L3_POLICY_UPLOAD] no decode step refills the bucket during a prompt: the worker ignores it now, and with
        // a rate limit this also waits for the rest of the queue (the bucket may have left jobs in it), so no upload runs
        // beside the stream
        mc->up_drain = true;
        mc->l18_drain.store(true); // [TAG_FN_L18_CHURN_PACE] the worker's batch lands at full speed
        hot_set_yield(mc); // [TAG_FN_L4_EXIT]
        mc->wcv.notify_all();
        mc->wcv_idle.wait(lk, [mc]() { return !mc->worker_busy && (mc->stop || mc->up_rate == 0 || mc->todo.empty()); });
        mc->up_drain = false;
        mc->l18_drain.store(false);
        hot_set_yield(mc);
    }
    hot_adapt_publish(mc);
}

// [TAG_FN_L10_SLOTLEND] which lend slots of the layer touch a segment that is out (the slot's bytes in each of its tensors
// plus SL_SLACK after them)
void hot_sl_slots(moe_cache * mc, layer_state & ls) {
    const int32_t s0 = ls.sl_bbase >= 0 ? std::min(ls.sl_bbase, ls.sl_base) : ls.sl_base; // [TAG_FN_L21_BIGUB]
    for (int32_t s = std::max(0, s0); s < ls.pub.n_slots && s < (int32_t) ls.slot_out.size(); ++s) {
        bool o = false;
        for (int k = 0; k < 3 && !o; ++k) {
            const size_t lo = ls.sl_toff[k] + (size_t) s*ls.sl_sb[k];
            const size_t hi = lo + ls.sl_sb[k] + SL_SLACK;
            for (size_t g = 0; g < mc->sl_grp.size() && !o; ++g) {
                if (mc->sl_given[g] == 0) {
                    continue;
                }
                for (const auto & r : mc->sl_grp[g]) {
                    if (r.first < hi && lo < r.second) {
                        o = true;
                        break;
                    }
                }
            }
        }
        ls.slot_out[(size_t) s] = o ? 1 : 0;
    }
}

// [TAG_FN_L10_SLOTLEND] the KV cache needs `need` bytes out in all: groups from the top go out until it has them - their
// slots leave the tables (residents evicted, queued uploads dropped, the worker's batch landed first), then the segments
// are unmapped. The owner's graphs are synchronized (the caller is its memory apply, between graphs)
size_t hot_sl_release(moe_cache * mc, const void * owner, size_t need) {
    if (mc->kv_out >= need || !mc->cb_buf || (mc->adapt && owner != mc->owner)) {
        return mc->kv_out;
    }
    std::lock_guard<std::mutex> slk(mc->step_mtx);
    hot_turn_join(mc); // [TAG_FN_L18_FIRSTANS_OVERLAP] a pass in flight lands first
    const int64_t t0 = ggml_time_us();
    std::vector<size_t> rel;
    size_t out = mc->kv_out;
    for (size_t g = mc->sl_n_kv; g > 0 && out < need; --g) { // [TAG_FN_L21_BIGUB] the KV groups only
        if (mc->sl_given[g - 1] == 0 && mc->sl_grp_bytes[g - 1] > 0) {
            mc->sl_given[g - 1] = mc->sl_grp_bytes[g - 1]; // out from here on (the slots below see it)
            rel.push_back(g - 1);
            out += mc->sl_grp_bytes[g - 1];
        }
    }
    if (rel.empty()) {
        return mc->kv_out;
    }
    size_t n_out = 0;
    for (auto & ls : mc->layers) {
        if (ls.sl_base >= 0) {
            hot_sl_slots(mc, ls);
        }
    }
    hot_drain_out(mc); // queued uploads into out slots dropped, the worker's batch landed (in memory still mapped)
    size_t n_ev = 0;
    for (auto & ls : mc->layers) {
        if (ls.sl_base < 0) {
            continue;
        }
        bool changed = false;
        for (int32_t s = ls.sl_base; s < ls.pub.n_slots; ++s) {
            if (!ls.slot_out[(size_t) s]) {
                continue;
            }
            n_out++;
            const int32_t e = ls.slot_expert[s];
            if (e >= 0) {
                ls.expert_slot[e] = -1;
                ls.slot_expert[s] = -1;
                changed = true;
                n_ev++;
            }
        }
        if (changed) {
            hot_write_tables(mc, ls);
        }
    }
    if (mc->up_backend) {
        ggml_backend_synchronize(mc->up_backend);
    }
    size_t given = 0;
    for (size_t g : rel) {
        bool ok = true;
        for (const auto & r : mc->sl_grp[g]) {
            ok = mc->cb_unmap(mc->cb_buf, r.first, r.second - r.first) && ok;
        }
        if (!ok) {
            // its slots stay out; what could not be unmapped holds VRAM, so the group counts nothing given
            LLAMA_LOG_WARN("moe-hot: [TAG_FN_L10_SLOTLEND] group %zu could not give all of its VRAM back\n", g);
            mc->sl_given[g] = 1; // out, a token amount (the restore maps it again)
        }
        given      += mc->sl_given[g];
        mc->kv_out += mc->sl_given[g];
    }
    mc->kv_n_rel++;
    const double ms = (ggml_time_us() - t0)/1000.0;
    mc->kv_rel_ms += ms;
    LLAMA_LOG_INFO("moe-hot: [TAG_FN_L10_SLOTLEND] %zu groups gave %.0f MiB to the KV cache in %.1f ms: %zu slots out, %zu "
            "residents evicted (%.0f of %.0f MiB out)\n", rel.size(), given/1048576.0, ms, n_out, n_ev, mc->kv_out/1048576.0,
            mc->kv_bytes/1048576.0);
    return mc->kv_out;
}

// [TAG_FN_L10_SLOTLEND] the KV cache keeps `keep` bytes out: the lowest groups out come back while it has more than that -
// mapped again, cleared, their slots usable and empty (the adaptive set fills them)
size_t hot_sl_restore(moe_cache * mc, const void * owner, size_t keep) {
    if (mc->kv_out == 0 || !mc->cb_buf || (mc->adapt && owner != mc->owner)) {
        return mc->kv_out;
    }
    std::lock_guard<std::mutex> slk(mc->step_mtx);
    hot_turn_join(mc); // [TAG_FN_L18_FIRSTANS_OVERLAP] a pass in flight lands first
    const int64_t t0 = ggml_time_us();
    size_t n_res = 0;
    size_t bytes = 0;
    for (size_t g = 0; g < mc->sl_n_kv; ++g) { // [TAG_FN_L21_BIGUB] the KV groups only
        if (mc->sl_given[g] == 0) {
            continue;
        }
        const size_t b = mc->sl_given[g];
        if (mc->kv_out < keep + b) {
            break;
        }
        bool   ok   = true;
        size_t done = 0;
        for (const auto & r : mc->sl_grp[g]) {
            (void) mc->cb_unmap(mc->cb_buf, r.first, r.second - r.first); // a failed release may have left it mapped
            if (!mc->cb_map(mc->cb_buf, r.first, r.second - r.first)) {
                ok = false;
                break;
            }
            done++;
        }
        if (!ok) {
            for (size_t i = 0; i < done; ++i) {
                const auto & r = mc->sl_grp[g][i];
                (void) mc->cb_unmap(mc->cb_buf, r.first, r.second - r.first);
            }
            LLAMA_LOG_WARN("moe-hot: [TAG_FN_L10_SLOTLEND] mapping group %zu again failed: its slots stay out until the next try\n", g);
            break;
        }
        for (const auto & r : mc->sl_grp[g]) {
            ggml_backend_tensor_memset(mc->sl_raw, 0, r.first, r.second - r.first);
        }
        mc->sl_given[g] = 0;
        mc->kv_out -= b;
        bytes += b;
        n_res++;
    }
    if (n_res > 0) {
        for (auto & ls : mc->layers) {
            if (ls.sl_base >= 0) {
                hot_sl_slots(mc, ls);
            }
        }
        mc->kv_n_res++;
        const double ms = (ggml_time_us() - t0)/1000.0;
        mc->kv_res_ms += ms;
        LLAMA_LOG_INFO("moe-hot: [TAG_FN_L10_SLOTLEND] %zu groups (%.0f MiB) came back from the KV cache in %.1f ms (%.0f MiB still "
                "out); their slots fill from the routing\n", n_res, bytes/1048576.0, ms, mc->kv_out/1048576.0);
    }
    return mc->kv_out;
}

size_t hot_refill(moe_cache * mc, const std::vector<refill_job> & jobs, int n_threads); // [TAG_FN_L21_BIGUB] below

// [TAG_FN_L21_BIGUB] the borrow group goes out for a 16K prompt ubatch's compute reserve: its slots leave the tables (their
// residents are remembered), queued uploads into them drop, the segments are unmapped. The owner's graphs are synchronized
// (the caller is its switch to FULL). Returns the bytes out (0: none, nothing changed)
size_t hot_sb_release(moe_cache * mc, const void * owner) {
    if (!mc->sl_on || mc->sb_bytes == 0 || mc->sl_n_kv >= mc->sl_grp.size() || !mc->cb_buf || (mc->adapt && owner != mc->owner)) {
        return 0;
    }
    if (mc->sb_out > 0) {
        return mc->sb_out;
    }
    std::lock_guard<std::mutex> slk(mc->step_mtx);
    hot_turn_join(mc); // [TAG_FN_L18_FIRSTANS_OVERLAP] a pass in flight lands first
    const int64_t t0 = ggml_time_us();
    const size_t  g  = mc->sl_n_kv;
    mc->sl_given[g] = mc->sl_grp_bytes[g];
    for (auto & ls : mc->layers) {
        if (ls.sl_base >= 0) {
            hot_sl_slots(mc, ls);
        }
    }
    hot_drain_out(mc); // queued uploads into out slots dropped, the worker's batch landed (in memory still mapped)
    mc->sb_res.clear();
    for (size_t li = 0; li < mc->layers.size(); ++li) {
        layer_state & ls = mc->layers[li];
        if (ls.sl_bbase < 0 || ls.slot_out.empty()) {
            continue;
        }
        bool changed = false;
        for (int32_t sl = ls.sl_bbase; sl < ls.sl_base; ++sl) {
            if (!ls.slot_out[(size_t) sl]) {
                continue;
            }
            const int32_t e = ls.slot_expert[sl];
            if (e >= 0) {
                mc->sb_res.push_back({ (int32_t) li, sl, e });
                ls.expert_slot[e]  = -1;
                ls.slot_expert[sl] = -1;
                changed = true;
            }
        }
        if (changed) {
            hot_write_tables(mc, ls);
        }
    }
    if (mc->up_backend) {
        ggml_backend_synchronize(mc->up_backend);
    }
    bool ok = true;
    for (const auto & r : mc->sl_grp[g]) {
        ok = mc->cb_unmap(mc->cb_buf, r.first, r.second - r.first) && ok;
    }
    if (!ok) {
        LLAMA_LOG_WARN("moe-hot: [TAG_FN_L21_BIGUB] the borrow group could not give all of its VRAM back\n");
    }
    mc->sb_out = ok ? mc->sl_grp_bytes[g] : 0;
    if (!ok) {
        mc->sl_given[g] = 1; // out (its slots hold nothing), a token amount: the restore maps it again
    }
    mc->sb_n++;
    const double ms = (ggml_time_us() - t0)/1000.0;
    mc->sb_rel_ms += ms;
    if (mc->sb_n <= 3 || mc->sb_n % 32 == 0) {
        LLAMA_LOG_INFO("moe-hot: [TAG_FN_L21_BIGUB] borrow %" PRIu64 ": %.0f MiB out in %.1f ms (%zu residents remembered)\n",
                mc->sb_n, mc->sb_out/1048576.0, ms, mc->sb_res.size());
    }
    return mc->sb_out;
}

// [TAG_FN_L21_BIGUB] the borrow group comes back (the compute reserve is gone): mapped, cleared, its remembered residents
// uploaded again (through the upload worker with the adaptive set, else here) unless they are resident elsewhere now
void hot_sb_restore(moe_cache * mc, const void * owner) {
    if (!mc->sl_on || mc->sl_n_kv >= mc->sl_grp.size() || mc->sl_given[mc->sl_n_kv] == 0 || !mc->cb_buf ||
        (mc->adapt && owner != mc->owner)) {
        return;
    }
    std::lock_guard<std::mutex> slk(mc->step_mtx);
    hot_turn_join(mc);
    const int64_t t0 = ggml_time_us();
    const size_t  g  = mc->sl_n_kv;
    size_t done = 0;
    bool   ok   = true;
    for (const auto & r : mc->sl_grp[g]) {
        (void) mc->cb_unmap(mc->cb_buf, r.first, r.second - r.first); // a failed release may have left it mapped
        if (!mc->cb_map(mc->cb_buf, r.first, r.second - r.first)) {
            ok = false;
            break;
        }
        done++;
    }
    if (!ok) {
        for (size_t i = 0; i < done; ++i) {
            const auto & r = mc->sl_grp[g][i];
            (void) mc->cb_unmap(mc->cb_buf, r.first, r.second - r.first);
        }
        LLAMA_LOG_WARN("moe-hot: [TAG_FN_L21_BIGUB] mapping the borrow group again failed: its slots stay out until the next "
                "restore\n");
        return;
    }
    for (const auto & r : mc->sl_grp[g]) {
        ggml_backend_tensor_memset(mc->sl_raw, 0, r.first, r.second - r.first);
    }
    mc->sl_given[g] = 0;
    mc->sb_out      = 0;
    for (auto & ls : mc->layers) {
        if (ls.sl_base >= 0) {
            hot_sl_slots(mc, ls);
        }
    }
    size_t n_up  = 0;
    size_t bytes = 0;
    // [TAG_FN_R25_REFILL] a small borrow return lands here at once (the copy pool), so the answer's first steps find the whole
    // set instead of the CPU computing these experts while the upload worker brings them back
    bool sb_sync = !mc->adapt;
    if (mc->adapt && mc->r25_on && mc->r25_sb_max > 0 && mc->pools.empty()) {
        size_t need = 0;
        for (const auto & r : mc->sb_res) {
            const layer_state & ls = mc->layers[(size_t) r[0]];
            const int32_t sl = r[1];
            const int32_t e  = r[2];
            if (ls.slot_out[(size_t) sl] || ls.slot_expert[sl] >= 0 || ls.slot_in_flight[sl] || ls.expert_slot[e] >= 0 ||
                ls.expert_in_flight[e]) {
                continue;
            }
            need += ls.pub.up_src->nb[2] + ls.pub.gate_src->nb[2] + ls.pub.down_src->nb[2];
        }
        sb_sync = need <= mc->r25_sb_max;
        mc->r25_sb_sync += sb_sync ? 1 : 0;
    }
    if (!sb_sync) {
        std::lock_guard<std::mutex> lk(mc->wmtx);
        for (const auto & r : mc->sb_res) {
            layer_state & ls = mc->layers[(size_t) r[0]];
            const int32_t sl = r[1];
            const int32_t e  = r[2];
            if (ls.slot_out[(size_t) sl] || ls.slot_expert[sl] >= 0 || ls.slot_in_flight[sl] || ls.expert_slot[e] >= 0 ||
                ls.expert_in_flight[e]) {
                continue;
            }
            ls.slot_in_flight[sl]  = true;
            ls.expert_in_flight[e] = true;
            upload_job j;
            j.layer_idx = (size_t) r[0];
            j.expert    = e;
            j.slot      = sl;
            mc->todo.push_back(j);
            bytes += ls.pub.up_src->nb[2] + ls.pub.gate_src->nb[2] + ls.pub.down_src->nb[2];
            n_up++;
        }
    } else {
        std::vector<refill_job> jobs;
        std::vector<char> touched(mc->layers.size(), 0);
        for (const auto & r : mc->sb_res) {
            layer_state & ls = mc->layers[(size_t) r[0]];
            const int32_t sl = r[1];
            const int32_t e  = r[2];
            if (ls.slot_out[(size_t) sl] || ls.slot_expert[sl] >= 0 || ls.expert_slot[e] >= 0) {
                continue;
            }
            if (mc->adapt && (ls.slot_in_flight[sl] || ls.expert_in_flight[e])) {
                continue; // [TAG_FN_R25_REFILL]
            }
            ggml_tensor *       dsts[3] = { ls.pub.up_c,   ls.pub.gate_c,   ls.pub.down_c   };
            const ggml_tensor * srcs[3] = { ls.pub.up_src, ls.pub.gate_src, ls.pub.down_src };
            bool fit = true;
            for (int k = 0; k < 3; ++k) {
                const size_t sz = srcs[k]->nb[2];
                fit = fit && (size_t) sl*dsts[k]->nb[2] + sz <= ggml_nbytes(dsts[k]) && (size_t) (e + 1)*sz <= ggml_nbytes(srcs[k]);
            }
            if (!fit) {
                continue;
            }
            for (int k = 0; k < 3; ++k) {
                const size_t sz = srcs[k]->nb[2];
                jobs.push_back({ dsts[k], (size_t) sl*dsts[k]->nb[2], (const char *) srcs[k]->data + (size_t) e*sz, sz });
            }
            ls.slot_expert[sl] = e;
            ls.expert_slot[e]  = sl;
            if ((size_t) sl < ls.slot_dirty.size()) {
                ls.slot_dirty[sl] = 0; // [TAG_FN_R25_REFILL] refilled below
            }
            touched[(size_t) r[0]] = 1;
            n_up++;
        }
        bytes = hot_refill(mc, jobs, 8);
        for (size_t li = 0; li < mc->layers.size(); ++li) {
            if (touched[li]) {
                hot_write_tables(mc, mc->layers[li]);
            }
        }
    }
    mc->sb_res.clear();
    if (!sb_sync && n_up > 0) {
        mc->wcv.notify_one();
    }
    const double ms = (ggml_time_us() - t0)/1000.0;
    mc->sb_res_ms += ms;
    if (mc->sb_n <= 3 || mc->sb_n % 32 == 0) {
        LLAMA_LOG_INFO("moe-hot: [TAG_FN_L21_BIGUB] borrow %" PRIu64 " back in %.1f ms: %zu residents (%.2f GiB) %s\n", mc->sb_n, ms,
                n_up, bytes/1073741824.0, !sb_sync ? "queued for the upload worker" : mc->adapt ? "refilled at once [TAG_FN_R25_REFILL]" : "refilled");
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

// [TAG_FN_R25_REFILL] the synchronous refills through persistent copy threads, each with its own CUDA stream and two small
// pinned halves that stay in the CPU caches: the DMA reads the halves from the caches, so DRAM sees only the source reads.
// The 128 MiB halves with non-temporal stores cost DRAM three passes (source read, staging write, DMA read): ~19 GiB/s for
// the 7.9 GiB of a 16K prompt's switch (0.44 s of the 0.6 s between the first and the second token). Same bytes, same
// slots, every copy landed before the call returns.
struct fn_r25_pool {
    struct piece { ggml_tensor * dst; size_t off; const uint8_t * src; size_t size; };

    bool                              tried   = false;
    bool                              ok      = false;
    int                               n_th    = 0;
    size_t                            half    = 0;
    bool                              prio    = false; // [TAG_FN_L22_REFILL] M2.4 memory priority low
    ggml_backend_buffer_t             stage   = nullptr;
    uint8_t *                         stage_p = nullptr;
    std::vector<ggml_backend_t>       be;
    std::vector<ggml_backend_event_t> ev;
    std::vector<std::thread>          th;

    std::mutex              call_mtx; // one refill at a time
    std::mutex              mtx;
    std::condition_variable cv_go;
    std::condition_variable cv_done;
    uint64_t                gen  = 0;
    int                     left = 0;
    bool                    stop = false;

    std::vector<piece>  pieces; // the running refill: pieces of at most one half
    std::atomic<size_t> next{0};

    uint64_t n_runs = 0;
    uint64_t n_logs = 0;

    void worker(int t) {
        llama_fn_thread_name("hot-refill-r25", t);
        if (prio) {
            fn_l22_mem_prio_low();
        }
        uint64_t seen = 0;
        struct staged { ggml_tensor * dst; size_t off; size_t so; size_t size; };
        std::vector<staged> st;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv_go.wait(lk, [&]() { return stop || gen != seen; });
                if (stop) {
                    return;
                }
                seen = gen;
            }
            ggml_backend_t       b       = be[(size_t) t];
            ggml_backend_event_t e[2]    = { ev[2*(size_t) t], ev[2*(size_t) t + 1] };
            bool                 rec[2]  = { false, false };
            int                  h       = 0;
            size_t               used    = 0;
            auto hp = [&](int i) { return stage_p + (2*(size_t) t + (size_t) i)*half; };
            auto flush = [&]() {
                if (st.empty()) {
                    return;
                }
                for (const auto & x : st) {
                    ggml_backend_tensor_set_async(b, x.dst, hp(h) + x.so, x.off, x.size);
                }
                ggml_backend_event_record(e[h], b);
                rec[h] = true;
                st.clear();
                used = 0;
                h ^= 1;
                if (rec[h]) {
                    ggml_backend_event_synchronize(e[h]); // the DMA has read this half's previous bytes
                    rec[h] = false;
                }
            };
            for (;;) {
                const size_t i = next.fetch_add(1, std::memory_order_relaxed);
                if (i >= pieces.size()) {
                    break;
                }
                const piece & p = pieces[i];
                size_t o = 0;
                while (o < p.size) {
                    if (used == half) {
                        flush();
                    }
                    const size_t k = std::min(p.size - o, half - used);
                    memcpy(hp(h) + used, p.src + o, k); // plain stores: the half stays in the caches for the DMA
                    st.push_back({ p.dst, p.off + o, used, k });
                    used += k;
                    o    += k;
                }
            }
            flush();
            ggml_backend_synchronize(b);
            std::lock_guard<std::mutex> lk(mtx);
            if (--left == 0) {
                cv_done.notify_all();
            }
        }
    }

    bool init(ggml_backend_dev_t dev, int n, size_t half_b, bool prio_low) {
        tried = true;
        ggml_backend_buffer_type_t hb = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
        if (!hb || n < 1 || half_b < (64u << 10)) {
            return false;
        }
        n_th = n;
        half = half_b;
        prio = prio_low;
        stage = ggml_backend_buft_alloc_buffer(hb, 2*(size_t) n*half);
        if (!stage) {
            return false;
        }
        stage_p = (uint8_t *) ggml_backend_buffer_get_base(stage);
        for (int t = 0; t < n; ++t) {
            ggml_backend_t b = ggml_backend_dev_init(dev, nullptr);
            ggml_backend_event_t e0 = b ? ggml_backend_event_new(dev) : nullptr;
            ggml_backend_event_t e1 = e0 ? ggml_backend_event_new(dev) : nullptr;
            if (b) { be.push_back(b); }
            if (e0) { ev.push_back(e0); }
            if (e1) { ev.push_back(e1); }
            if (!b || !e0 || !e1) {
                release();
                return false;
            }
        }
        for (int t = 0; t < n; ++t) {
            th.emplace_back([this, t]() { worker(t); });
        }
        ok = true;
        return true;
    }

    void release() { // only before the threads start, or after they stopped
        for (auto * e : ev) { ggml_backend_event_free(e); }
        for (auto * b : be) { ggml_backend_free(b); }
        ev.clear();
        be.clear();
        if (stage) {
            ggml_backend_buffer_free(stage);
            stage   = nullptr;
            stage_p = nullptr;
        }
        ok = false;
    }

    void shutdown() {
        std::lock_guard<std::mutex> call(call_mtx);
        if (th.empty()) {
            return;
        }
        {
            std::lock_guard<std::mutex> lk(mtx);
            stop = true;
        }
        cv_go.notify_all();
        for (auto & x : th) {
            x.join();
        }
        th.clear();
        release();
    }

    ~fn_r25_pool() {
        for (auto & x : th) {
            if (x.joinable()) {
                x.detach(); // process exit: never wait inside the loader lock
            }
        }
    }

    // false: not usable (the caller copies its own way)
    bool run(moe_cache * mc, const std::vector<refill_job> & jobs, size_t & bytes);
};
fn_r25_pool g_r25;

bool fn_r25_pool::run(moe_cache * mc, const std::vector<refill_job> & jobs, size_t & bytes) {
    std::lock_guard<std::mutex> call(call_mtx);
    if (!tried) {
        ggml_backend_dev_t dev = mc->up_backend ? ggml_backend_get_device(mc->up_backend) : nullptr;
        if (!dev && jobs[0].dst->buffer) {
            dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(jobs[0].dst->buffer));
        }
        const bool good = init(dev, mc->r25_threads, mc->r25_half, mc->l22_refill);
        LLAMA_LOG_INFO("moe-hot: [TAG_FN_R25_REFILL] copy pool %s: %d threads, each with its own stream and 2 x %zu KiB pinned "
                "halves (%.0f MiB in all)\n", good ? "ready" : "NOT available (the old refill path)", mc->r25_threads,
                mc->r25_half >> 10, good ? 2.0*mc->r25_threads*mc->r25_half/1048576.0 : 0.0);
    }
    if (!ok || stop) {
        return false;
    }
    const int64_t t0 = ggml_time_us();
    pieces.clear();
    bytes = 0;
    for (const auto & j : jobs) {
        for (size_t o = 0; o < j.size; o += half) {
            pieces.push_back({ j.dst, j.off + o, (const uint8_t *) j.src + o, std::min(half, j.size - o) });
        }
        bytes += j.size;
    }
    if (pieces.empty()) {
        return true;
    }
    next.store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(mtx);
        left = n_th;
        gen++;
    }
    cv_go.notify_all();
    {
        std::unique_lock<std::mutex> lk(mtx);
        cv_done.wait(lk, [&]() { return left == 0; });
    }
    const double ms = (ggml_time_us() - t0)/1000.0;
    n_runs++;
    // [TAG_FN_R25_REFILL] LLAMA_FN_R25_VERIFY=<k>: every k-th piece read back (test runs only)
    uint64_t bad = 0, checked = 0;
    if (mc->r25_verify > 0) {
        std::vector<uint8_t> tmp;
        for (size_t i = 0; i < pieces.size(); i += (size_t) mc->r25_verify) {
            const piece & p = pieces[i];
            tmp.resize(p.size);
            ggml_backend_tensor_get(p.dst, tmp.data(), p.off, p.size);
            bad += memcmp(tmp.data(), p.src, p.size) != 0;
            checked++;
        }
        mc->r25_bad += bad;
        if (bad > 0 || n_logs < 4) {
            LLAMA_LOG_WARN("moe-hot: [TAG_FN_R25_REFILL] verify: %llu of %llu pieces differ from their source (%llu so far)\n",
                    (unsigned long long) bad, (unsigned long long) checked, (unsigned long long) mc->r25_bad);
        }
    }
    if (n_logs < 6 || n_runs % 64 == 0) {
        n_logs++;
        LLAMA_LOG_INFO("moe-hot: [TAG_FN_R25_REFILL] refill %llu: %.2f GiB (%zu pieces) in %.1f ms (%.1f GiB/s, %d threads)\n",
                (unsigned long long) n_runs, bytes/1073741824.0, pieces.size(), ms, ms > 0 ? bytes/1073741824.0/(ms/1000.0) : 0.0,
                n_th);
    }
    return true;
}

// the resident experts of the lent layers whose bytes the stream may have overwritten: through the pinned halves when
// the hot set has an upload stream, else one synchronous copy per slice. Returns the bytes uploaded.
// [TAG_FN_L3_VRAM_CBUF] n_threads: the copy threads (the stream's unlend keeps REFILL_THREADS)
size_t hot_refill_staged(moe_cache * mc, const std::vector<refill_job> & jobs, int n_threads);

size_t hot_refill(moe_cache * mc, const std::vector<refill_job> & jobs, int n_threads = REFILL_THREADS) {
    // [TAG_FN_L22_REFILL] (a) the slices on pinned host pages go to the device straight from there (one async copy per
    // pinned segment on the upload stream), the rest through the staging halves as before; the tickets are released after
    // the stream has synchronized (every copy landed)
    if (!mc->l22_refill || !mc->up_backend || !llama_fn_pin_active() || jobs.empty()) {
        return hot_refill_staged(mc, jobs, n_threads);
    }
    std::vector<refill_job> rest;
    std::vector<int32_t>    tickets;
    size_t pinned = 0;
    for (const auto & j : jobs) {
        size_t o = 0;
        while (o < j.size) {
            const uint8_t *  p   = (const uint8_t *) j.src + o;
            llama_fn_pin_seg seg = {};
            if (!llama_fn_pin_query(p, j.size - o, &seg) || seg.len == 0 || seg.len > j.size - o) {
                rest.push_back({ j.dst, j.off + o, p, j.size - o });
                break;
            }
            if (seg.pinned) {
                ggml_backend_tensor_set_async(mc->up_backend, j.dst, seg.src, j.off + o, seg.len); // R-REV-4: the registered pages
                pinned += seg.len;
            } else {
                rest.push_back({ j.dst, j.off + o, p, seg.len });
            }
            if (seg.ticket >= 0) {
                tickets.push_back(seg.ticket);
            }
            o += seg.len;
        }
    }
    const size_t staged = hot_refill_staged(mc, rest, n_threads); // ends with a synchronize of the upload stream
    ggml_backend_synchronize(mc->up_backend);
    for (int32_t t : tickets) {
        llama_fn_pin_release(t);
    }
    mc->l22_pin_b.fetch_add(pinned, std::memory_order_relaxed);
    return pinned + staged;
}

size_t hot_refill_staged(moe_cache * mc, const std::vector<refill_job> & jobs, int n_threads) {
    llama_fn_stall_scope st_scope("hot refill begins (jobs, threads)", "hot refill ends", (int64_t) jobs.size(), n_threads); // [TAG_FN_L18_FIRSTANS_STALL]
    if (mc->r25_on && !jobs.empty()) {
        size_t b = 0;
        if (g_r25.run(mc, jobs, b)) { // [TAG_FN_R25_REFILL]
            return b;
        }
    }
    n_threads = std::max(1, std::min(64, n_threads));
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
        for (int t = 0; t < n_threads; ++t) {
            th.emplace_back([&, t]() {
                llama_fn_thread_name("hot-refill", t); // [TAG_FN_L18_FIRSTANS_STALL]
                if (mc->l22_refill) {
                    fn_l22_mem_prio_low(); // [TAG_FN_L22_REFILL] M2.4
                }
                for (size_t k = (size_t) t; k < n; k += (size_t) n_threads) {
                    hot_copy(mc, base + stage_off[k], jobs[i + k].src, jobs[i + k].size); // [TAG_FN_L17_START_NT]
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
    std::lock_guard<std::mutex> slk(mc->step_mtx); // [TAG_FN_L3_POLICY_STATE]
    hot_turn_join(mc); // [TAG_FN_L18_FIRSTANS_OVERLAP] a pass in flight lands first
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
    // [TAG_FN_L3_VRAM_CBUF] with a tail the stream's range ends where the tail starts, so the two never share a layer
    const size_t end   = mc->cb_buf == dbuf ? mc->cb_lo : size;
    const size_t align = std::max<size_t>(256, ggml_backend_buffer_get_alignment(dbuf));
    if (bytes + align > end) {
        return false;
    }
    const size_t start = (end - bytes) / align * align;
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
        g_lend.raw = ggml_new_tensor_1d(g_lend.ctx, GGML_TYPE_I8, (int64_t) (end - start));
        ggml_set_name(g_lend.raw, "moe_hot_lent");
        if (ggml_backend_tensor_alloc(dbuf, g_lend.raw, lo) != GGML_STATUS_SUCCESS) {
            ggml_free(g_lend.ctx);
            g_lend = {};
            return false;
        }
        g_lend.buf  = dbuf;
        g_lend.lo   = lo;
        g_lend.size = end - start;
    } else if (g_lend.buf != dbuf || g_lend.lo != lo) {
        return false; // the hot buffer never moves; another range would invalidate the stream's bank aliases
    }

    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(dbuf);
    auto overlaps = [&](const ggml_tensor * t) {
        return (uint8_t *) t->data + ggml_backend_buft_get_alloc_size(buft, t) > lo && (uint8_t *) t->data < base + end;
    };
    size_t n_lent = 0;
    for (auto & ls : mc->layers) {
        if (overlaps(ls.pub.up_c) || overlaps(ls.pub.gate_c) || overlaps(ls.pub.down_c)) {
            ls.lent = true;
            ls.lend_range = true; // [TAG_FN_L15_TIER]
            n_lent++;
        }
    }
    hot_drain_out(mc);
    for (auto & ls : mc->layers) {
        if (ls.lent) {
            hot_write_tables(mc, ls);
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
    if (mc) {
        mc->l22_pf_stop.store(true); // [TAG_FN_L22_REFILL] r22 T5: the refill set's read-ahead ends here
    }
    if (!mc || !mc->lent_any) {
        return;
    }
    if (mc->adapt && owner != mc->owner) {
        return;
    }
    std::lock_guard<std::mutex> slk(mc->step_mtx); // [TAG_FN_L3_POLICY_STATE]
    hot_turn_join(mc); // [TAG_FN_L18_FIRSTANS_OVERLAP] a pass in flight lands first
    if (!mc->l17_burst_start) { // [TAG_FN_L17_START_BURST] the window opens at the start only
        mc->dc_burst_left = std::max(mc->dc_burst_left, mc->dc_burst_passes); // [TAG_FN_L15_BURST] a big prompt ran
    }
    if (!mc->lent_any) {
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
    // [TAG_FN_L19_UNLEND_CLEAR] the whole range reads 0 before the residents come back, as after the tail's and the KV lend's
    // remap: the stream left its bank bytes in the empty slots, and the batched kernels of the CPU-path prompt graphs
    // ([TAG_FN_L19_TTFT_PHOT], MMQ) read past the last row of each slot into the next one (the last K iteration loads
    // ITER_K/qk blocks, e.g. 24 of the 20 blocks of a 640-wide Q5_1 down row, against zero-padded activations). A NaN or Inf
    // scale in those bytes times 0 gave NaN in the expert's last output feature; the hidden state, then the recurrent state
    // went NaN and every later answer of the slot drafted nothing (the l19 MTP draft collapse after a streamed prompt).
    ggml_backend_tensor_memset(g_lend.raw, 0, 0, ggml_nbytes(g_lend.raw));
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
        hot_write_tables(mc, ls);
    }
    mc->lent_any = false;
    const double ms = (ggml_time_us() - t0)/1000.0;
    mc->unlend_ms += ms;
    mc->unlend_b  += bytes;
    mc->l22_ul_ms += ms;    // [TAG_FN_L22_PASSLOG]
    mc->l22_ul_b  += bytes;
    if (mc->n_lend <= 2 || mc->n_lend % 16 == 0) {
        LLAMA_LOG_INFO("moe-hot: [TAG_FN_R1_PFS_LEND] lend %" PRIu64 ": %.2f GiB of resident experts refilled in %.1f ms "
                "(%.1f GiB/s)\n", mc->n_lend, bytes/1073741824.0, ms, ms > 0 ? bytes/1073741824.0/(ms/1000.0) : 0.0);
    }
}

// [TAG_FN_L22_ENDOVERLAP] llama_moe_hot_unlend with the refill on the upload worker: the range is cleared as there, the
// residents whose slices reach into it leave the tables and go in flight (evict, upload, publish at an owner step once
// landed: the device tables never name a slot before its copy completed), the others stay; the bridge / CPU computes the
// experts in flight meanwhile. Without the adaptive set (no worker) the synchronous unlend runs.
void llama_moe_hot_unlend_async(const void * owner) {
    moe_cache * mc = g_cache;
    if (mc) {
        mc->l22_pf_stop.store(true); // [TAG_FN_L22_REFILL] r22 T5: the refill set's read-ahead ends here
    }
    if (!mc || !mc->lent_any) {
        return;
    }
    if (!mc->adapt || !mc->worker.joinable()) {
        llama_moe_hot_unlend(owner);
        return;
    }
    if (owner != mc->owner) {
        return;
    }
    std::unique_lock<std::mutex> slk(mc->step_mtx);
    hot_turn_join(mc); // [TAG_FN_L18_FIRSTANS_OVERLAP] a pass in flight lands first
    if (!mc->lent_any) {
        return;
    }
    // [TAG_FN_L22_ENDOVERLAP] r22 T6 fix: a refill above L22_ASYNC_REFILL_MAX goes the synchronous way (the copy threads)
    {
        size_t need = 0;
        for (const auto & ls : mc->layers) {
            if (!ls.lent) {
                continue;
            }
            const ggml_tensor * dsts[3] = { ls.pub.up_c,   ls.pub.gate_c,   ls.pub.down_c   };
            const ggml_tensor * srcs[3] = { ls.pub.up_src, ls.pub.gate_src, ls.pub.down_src };
            for (int32_t s = 0; s < ls.pub.n_slots; ++s) {
                if (ls.slot_expert[s] < 0 || ls.slot_in_flight[s]) {
                    continue;
                }
                bool reach = false;
                for (int k = 0; k < 3; ++k) {
                    reach = reach || (const uint8_t *) dsts[k]->data + (size_t) s*dsts[k]->nb[2] + srcs[k]->nb[2] > g_lend.lo;
                }
                if (reach) {
                    need += srcs[0]->nb[2] + srcs[1]->nb[2] + srcs[2]->nb[2];
                }
            }
        }
        if (need > L22_ASYNC_REFILL_MAX) {
            mc->l22_sync_ul++;
            if (mc->l22_sync_ul <= 2 || mc->l22_sync_ul % 16 == 0) {
                LLAMA_LOG_INFO("moe-hot: [TAG_FN_L22_ENDOVERLAP] lend refill of %.2f GiB: by the copy threads now (more than %zu MiB "
                        "would land through the upload worker for seconds while the CPU computes those experts; %" PRIu64 " so far)\n",
                        need/1073741824.0, L22_ASYNC_REFILL_MAX >> 20, mc->l22_sync_ul);
            }
            slk.unlock();
            llama_moe_hot_unlend(owner);
            return;
        }
    }
    if (!mc->l17_burst_start) { // [TAG_FN_L17_START_BURST] the window opens at the start only
        mc->dc_burst_left = std::max(mc->dc_burst_left, mc->dc_burst_passes); // [TAG_FN_L15_BURST] a big prompt ran
    }
    const int64_t t0 = ggml_time_us();
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(g_lend.buf);
    // [TAG_FN_L19_UNLEND_CLEAR] the whole range reads 0 first (the stream left its bank bytes there)
    ggml_backend_tensor_memset(g_lend.raw, 0, 0, ggml_nbytes(g_lend.raw));
    size_t bytes  = 0;
    size_t n_jobs = 0;
    std::vector<const ggml_tensor *> zeroed; // [TAG_FN_L3_POLICY_POOL] the layers of a pool share their tensors
    {
        std::lock_guard<std::mutex> lk(mc->wmtx);
        for (size_t li = 0; li < mc->layers.size(); ++li) {
            layer_state & ls = mc->layers[li];
            if (!ls.lent) {
                continue;
            }
            ggml_tensor *       dsts[3] = { ls.pub.up_c,   ls.pub.gate_c,   ls.pub.down_c   };
            const ggml_tensor * srcs[3] = { ls.pub.up_src, ls.pub.gate_src, ls.pub.down_src };
            for (int32_t s = 0; s < ls.pub.n_slots; ++s) {
                const int32_t e = ls.slot_expert[s];
                if (e < 0 || ls.slot_in_flight[s]) {
                    continue;
                }
                bool reach = false; // a slice that reaches into the lent range was overwritten
                for (int k = 0; k < 3; ++k) {
                    const size_t sz  = srcs[k]->nb[2];
                    const size_t off = (size_t) s*dsts[k]->nb[2];
                    reach = reach || (uint8_t *) dsts[k]->data + off + sz > g_lend.lo;
                }
                if (!reach) {
                    continue; // below the lent range: never overwritten, stays resident
                }
                ls.slot_expert[s]      = -1;
                ls.expert_slot[e]      = -1;
                ls.slot_in_flight[s]   = true;
                ls.expert_in_flight[e] = true;
                upload_job j;
                j.layer_idx = li;
                j.expert    = e;
                j.slot      = s;
                j.ul        = true;
                mc->todo.push_back(j);
                bytes += srcs[0]->nb[2] + srcs[1]->nb[2] + srcs[2]->nb[2];
                n_jobs++;
            }
        }
        mc->l22_ul_left += n_jobs;
    }
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
        hot_write_tables(mc, ls); // the residents that stayed; the slots in flight read "not hot"
    }
    mc->lent_any = false;
    if (n_jobs > 0) {
        mc->l22_ul_t0    = t0;
        mc->l22_ul_bytes = bytes;
        mc->wcv.notify_one();
    }
    const double ms = (ggml_time_us() - t0)/1000.0;
    mc->l22_ul_n++;
    mc->unlend_ms += ms;
    mc->l22_ul_ms += ms;
    mc->l22_ul_b  += bytes;
    if (mc->l22_ul_n <= 2 || mc->l22_ul_n % 16 == 0) {
        LLAMA_LOG_INFO("moe-hot: [TAG_FN_L22_ENDOVERLAP] lend %" PRIu64 ": %.2f GiB of resident experts queued for the upload worker "
                "in %.1f ms (%zu experts; the CPU computes them until each has landed)\n", mc->n_lend, bytes/1073741824.0, ms,
                n_jobs);
    }
}

// [TAG_FN_L22_REFILL] (b) see llama-moecache.h
size_t llama_moe_hot_prefetch_refill(const void * owner) {
    moe_cache * mc = g_cache;
    if (!mc || !mc->l22_refill || (mc->adapt && owner != mc->owner)) {
        return 0;
    }
    if (mc->l22_pf_busy.load()) {
        return 0; // the last one still runs
    }
    std::vector<std::pair<const void *, size_t>> rr;
    size_t bytes = 0;
    {
        std::unique_lock<std::mutex> slk(mc->step_mtx, std::try_to_lock);
        if (!slk.owns_lock()) {
            return 0;
        }
        const uint8_t * cb_base = mc->cb_buf ? (const uint8_t *) ggml_backend_buffer_get_base(mc->cb_buf) : nullptr;
        for (const auto & ls : mc->layers) {
            if (!ls.lent && !ls.cb_lent) {
                continue;
            }
            ggml_tensor *       dsts[3] = { ls.pub.up_c,   ls.pub.gate_c,   ls.pub.down_c   };
            const ggml_tensor * srcs[3] = { ls.pub.up_src, ls.pub.gate_src, ls.pub.down_src };
            for (int32_t s = 0; s < ls.pub.n_slots && s < (int32_t) ls.slot_expert.size(); ++s) {
                const int32_t e = ls.slot_expert[s];
                if (e < 0) {
                    continue;
                }
                for (int k = 0; k < 3; ++k) {
                    const size_t sz  = srcs[k]->nb[2];
                    const size_t off = (size_t) s*dsts[k]->nb[2];
                    const uint8_t * end = (const uint8_t *) dsts[k]->data + off + sz;
                    bool reach = false;
                    if (ls.lent && g_lend.lo) {
                        reach = end > g_lend.lo; // the stream's range: what the unlend refills
                    }
                    if (!reach && ls.cb_lent && cb_base) {
                        reach = end > cb_base + mc->cb_rel; // the tail's part that is unmapped: what the restore refills
                    }
                    if (reach && (size_t) (e + 1)*sz <= ggml_nbytes(srcs[k])) {
                        rr.push_back({ (const uint8_t *) srcs[k]->data + (size_t) e*sz, sz });
                        bytes += sz;
                    }
                }
            }
        }
    }
    if (rr.empty()) {
        return 0;
    }
    if (mc->l22_pf_th.joinable()) {
        mc->l22_pf_th.join(); // finished (not busy)
    }
    mc->l22_pf_busy.store(true);
    mc->l22_pf_stop.store(false);
    mc->l22_pf_th = std::thread([mc, rr = std::move(rr)]() {
        llama_fn_thread_name("hot-refill-prefetch");
        fn_l22_mem_prio_low();
        // r22 T5 fix: bounded (resident ranges, the RAM budget, the stop at the unlend): the unbounded request of the whole
        // set (~7.5 GiB at Flash-Next's 16-layer lend) read pages into a full RAM and slowed the first answer by +38 % ms per
        // step (r22 end test T2, first answer after a streamed prompt while the RAM held no room)
        fn_l22_refill_pf_stats st;
        fn_l22_refill_prefetch_run(rr, mc->l22_pf_stop, st);
        llama_fn_l22_log("switch", "what=refill_prefetch ranges=%zu resident=%zu asked_mib=%.0f cut_mib=%.0f avail_mib=%.0f "
                "ms=%.1f stopped=%d", st.ranges, st.resident, st.asked_b/1048576.0, st.cut_b/1048576.0, st.avail_b/1048576.0,
                st.ms, st.stopped ? 1 : 0);
        mc->l22_pf_busy.store(false);
    });
    return bytes;
}

// [TAG_FN_L22_PASSLOG] see llama-moecache.h
llama_moe_hot_l22_stats llama_moe_hot_l22_take() {
    llama_moe_hot_l22_stats r;
    moe_cache * mc = g_cache;
    if (!mc) {
        return r;
    }
    r.unlend_ms  = mc->l22_ul_ms;
    r.unlend_b   = mc->l22_ul_b;
    r.restore_ms = mc->l22_res_ms;
    r.restore_b  = mc->l22_res_b;
    r.pinned_b   = mc->l22_pin_b.exchange(0);
    r.released_b = mc->l22_rel_b;
    r.ul_left    = mc->l22_ul_left;
    mc->l22_ul_ms  = 0.0;
    mc->l22_ul_b   = 0;
    mc->l22_res_ms = 0.0;
    mc->l22_res_b  = 0;
    mc->l22_rel_b  = 0;
    return r;
}

// [TAG_FN_L3_POLICY_STATE] ---------------------------------------------------------------------------------------------

namespace {

size_t hot_layer_bytes(const layer_state & ls) {
    return ls.pub.up_src->nb[2] + ls.pub.gate_src->nb[2] + ls.pub.down_src->nb[2];
}

// [TAG_FN_L18_FIRSTANS_STATE] LLAMA_MOE_HOT_STATE=cache: <LLAMA_CACHE, else %LOCALAPPDATA%\llama.cpp (Windows) or
// $XDG_CACHE_HOME/llama.cpp or ~/.cache/llama.cpp>/hotstate/<model file name>.hotstate - one file per model file, nothing
// written next to the model; the folder is made when missing ("" when there is no such folder)
std::string hot_state_cache_path(const std::string & model_path) {
    namespace fs = std::filesystem;
    fs::path dir;
    if (const char * c = getenv("LLAMA_CACHE"); c && c[0]) {
        dir = fs::u8path(c);
    } else {
#ifdef _WIN32
        const char * la = getenv("LOCALAPPDATA");
        if (la && la[0]) {
            dir = fs::u8path(la) / "llama.cpp";
        }
#else
        if (const char * x = getenv("XDG_CACHE_HOME"); x && x[0]) {
            dir = fs::u8path(x) / "llama.cpp";
        } else if (const char * h = getenv("HOME"); h && h[0]) {
            dir = fs::u8path(h) / ".cache" / "llama.cpp";
        }
#endif
    }
    if (dir.empty()) {
        return std::string();
    }
    dir /= "hotstate";
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
        LLAMA_LOG_WARN("moe-hot: [TAG_FN_L18_FIRSTANS_STATE] cannot make %s (%s) - no saved state\n", dir.u8string().c_str(),
                ec.message().c_str());
        return std::string();
    }
    return (dir / (fs::u8path(model_path).filename().u8string() + ".hotstate")).u8string();
}

// [TAG_FN_L18_FIRSTANS_STATE] the saved residents' uploads into the worker's queue (their pages are in by now); a layer or slot
// that is out now drops its job as hot_drain_out drops a queued one (a tail slot keeps the expert for its restore)
void hot_state_queue(moe_cache * mc) {
    std::lock_guard<std::mutex> slk(mc->step_mtx);
    std::lock_guard<std::mutex> lk(mc->wmtx);
    size_t n_q = 0;
    for (const auto & j : mc->fa18_st_jobs) {
        layer_state & ls = mc->layers[j.layer_idx];
        const bool sl_o = !ls.slot_out.empty() && ls.slot_out[j.slot];
        if (mc->stop || ls.out() || sl_o) {
            ls.slot_in_flight[j.slot]     = false;
            ls.expert_in_flight[j.expert] = false;
            if (!mc->stop && (ls.cb_lent || ls.kv_lent) && !sl_o) {
                ls.slot_expert[j.slot]   = j.expert;
                ls.expert_slot[j.expert] = j.slot;
            } else if (ls.pool >= 0) {
                mc->pools[ls.pool].slot_layer[j.slot] = -1;
            }
            if (mc->fa18_st_left > 0) {
                mc->fa18_st_left--;
            }
            continue;
        }
        mc->todo.push_back(j);
        n_q++;
    }
    mc->fa18_st_jobs.clear();
    mc->wcv.notify_all();
    llama_fn_stall_mark("saved state: uploads queued (jobs)", (int64_t) n_q); // [TAG_FN_L18_FIRSTANS_STALL]
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
    // [TAG_FN_L18_FIRSTANS_STATE] in the background: upload jobs (marked in flight) instead of refill jobs, best count first
    const bool async = mc->fa18_state && mc->worker.joinable();
    std::vector<std::pair<float, upload_job>> ups;
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
        if (async) { // [TAG_FN_L18_FIRSTANS_STATE] the slot and the expert are in flight until the worker's copy is published
            upload_job u;
            u.layer_idx = (size_t) (&ls - mc->layers.data());
            u.expert    = e;
            u.slot      = s;
            u.st        = true;
            ups.push_back({ e < (int32_t) ls.dcnt.size() ? ls.dcnt[e] : 0.0f, u });
            ls.slot_in_flight[s]   = true;
            ls.expert_in_flight[e] = true;
            return true;
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
        if (!ls || ls->dcnt.size() != l.cnt.size() || hot_layer_bytes(*ls) != l.bytes || ls->out()) {
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
    if (async) {
        // [TAG_FN_L18_FIRSTANS_STATE] best count first over all layers (the first request may come before the last lands);
        // the sources' pages are read ahead in the background, then the worker copies them in 64 MiB batches
        std::stable_sort(ups.begin(), ups.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
        std::vector<std::pair<const void *, size_t>> ranges;
        size_t bytes = 0;
        for (const auto & u : ups) {
            const layer_state & ls = mc->layers[u.second.layer_idx];
            const ggml_tensor * srcs[3] = { ls.pub.up_src, ls.pub.gate_src, ls.pub.down_src };
            for (int k = 0; k < 3; ++k) {
                const size_t sz = srcs[k]->nb[2];
                ranges.push_back({ (const char *) srcs[k]->data + (size_t) u.second.expert*sz, sz });
                bytes += sz;
            }
        }
        // the pages first (read-ahead, then every page touched), then the uploads: the worker's copies never fault a page in
        // from the disk (a 64 MiB batch that did took 9.4 s beside the warm pass, and a FULL switch's drain waited for it)
        {
            std::lock_guard<std::mutex> lk(mc->wmtx);
            for (const auto & u : ups) {
                mc->fa18_st_jobs.push_back(u.second);
            }
            mc->fa18_st_left  += ups.size();
            mc->fa18_st_n      = ups.size();
            mc->fa18_st_bytes  = bytes;
            mc->fa18_st_t0     = t0;
        }
        if (!mc->fa18_pf_th.joinable()) {
            mc->fa18_pf_th = std::thread([mc, r = std::move(ranges)]() {
                llama_fn_thread_name("state-readahead"); // [TAG_FN_L18_FIRSTANS_STALL]
                const int64_t t1 = ggml_time_us();
                llama_fn_prefetch_ranges(r);
                llama_fn_touch_ranges(r, 8, &mc->fa18_pf_abort);
                const int64_t t2 = ggml_time_us();
                hot_state_queue(mc);
                LLAMA_LOG_INFO("moe-hot: [TAG_FN_L18_FIRSTANS_STATE] the saved residents' pages were read in %.0f ms, their uploads "
                        "are queued\n", (t2 - t1)/1000.0);
            });
        }
        llama_fn_stall_mark("saved state queued (residents, MiB)", (int64_t) ups.size(), (int64_t) (bytes >> 20)); // [TAG_FN_L18_FIRSTANS_STALL]
        LLAMA_LOG_INFO("moe-hot: [TAG_FN_L18_FIRSTANS_STATE] %s: %zu residents in %zu layers (%.2f GiB) queued for the upload "
                "worker in %.0f ms (pages read ahead), the counts of %llu decode steps%s\n", mc->st_path.c_str(), n_res,
                n_layers, bytes/1073741824.0, (ggml_time_us() - t0)/1000.0, (unsigned long long) steps,
                n_skip ? format(", %zu layer lines not used", n_skip).c_str() : "");
        return;
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
    if (mc && mc->adapt && mc->owner == owner && !mc->st_path.empty()) {
        std::lock_guard<std::mutex> slk(mc->step_mtx);
        hot_turn_join(mc); // [TAG_FN_L18_FIRSTANS_OVERLAP] a pass in flight lands first
        hot_state_save(mc);
    }
}

// [TAG_FN_L17_START_PRE] -------------------------------------------------------------------------------------------------

namespace {
// [TAG_FN_L18_FIRSTANS_OVERLAP] the background copies of the last pass: wait for them, name the new experts in the tables,
// let the worker go on. The caller holds step_mtx; a no-op when no pass is in flight.
void hot_turn_join(moe_cache * mc, bool wait) {
    if (!mc->fa18_busy.load(std::memory_order_acquire)) {
        return;
    }
    if (!wait && !mc->fa18_done.load(std::memory_order_acquire)) {
        return; // [OVERLAP=2] a step boundary after the copies landed names them
    }
    const int64_t tj = ggml_time_us();
    if (mc->fa18_th.joinable()) {
        mc->fa18_th.join();
    }
    const int64_t tw = ggml_time_us();
    for (const auto & w : mc->fa18_taken) {
        layer_state & ls = mc->layers[w.layer];
        ls.slot_expert[w.slot]        = w.expert;
        ls.expert_slot[w.expert]      = w.slot;
        ls.slot_in_flight[w.slot]     = false;
        ls.expert_in_flight[w.expert] = false;
        hot_adapt_set_entry(mc, ls, w.expert, w.slot);
    }
    if (mc->tbl_dirty) {
        ggml_backend_tensor_set(mc->tbl_all, mc->tbl_mirror.data(), 0, mc->tbl_mirror.size()*sizeof(int32_t));
        mc->tbl_dirty = false;
    }
    {
        std::lock_guard<std::mutex> lk(mc->wmtx);
        mc->paused = false;
        hot_set_yield(mc);
        mc->wcv.notify_all();
    }
    const double wait_ms = (tw - tj)/1000.0;
    const double copy_ms = (mc->fa18_t_done - mc->fa18_t_launch)/1000.0;
    const double hid_ms  = std::max(0.0, copy_ms - wait_ms);
    const double ms      = (mc->fa18_t_launch - mc->fa18_t0)/1000.0 + wait_ms; // the time the callers waited for
    mc->fa18_wait_ms += wait_ms;
    mc->fa18_hid_ms  += hid_ms;
    mc->fa18_n++;
    mc->l17_n++;
    mc->l17_swaps += mc->fa18_taken.size();
    mc->l17_bytes += mc->fa18_bytes;
    mc->l17_ms    += ms;
    llama_fn_stall_mark("prompt-end pass joined (wait us, hidden us)", (int64_t) (wait_ms*1000), (int64_t) (hid_ms*1000)); // [TAG_FN_L18_FIRSTANS_STALL]
    if (mc->l17_n <= 3 || mc->l17_n % 32 == 0) {
        LLAMA_LOG_INFO("moe-hot: [TAG_FN_L17_START_PRE] [TAG_FN_L18_FIRSTANS_OVERLAP] prompt end %" PRIu64 ": %zu experts (%.0f MiB) "
                "swapped in from the prompt's routing; decision %.1f ms (worker idle after %.1f ms), copies %.1f ms (%.1f GiB/s) "
                "behind the caller, %.1f ms of them waited for at the join; %.1f ms waited and %.1f ms hidden per pass so far\n",
                mc->l17_n, mc->fa18_taken.size(), mc->fa18_bytes/1048576.0, (mc->fa18_t_launch - mc->fa18_t0)/1000.0,
                (mc->fa18_t_idle - mc->fa18_t0)/1000.0, copy_ms, copy_ms > 0 ? mc->fa18_bytes/1073741824.0/(copy_ms/1000.0) : 0.0,
                wait_ms, mc->fa18_wait_ms/mc->fa18_n, mc->fa18_hid_ms/mc->fa18_n);
    }
    mc->fa18_jobs.clear();
    mc->fa18_taken.clear();
    mc->fa18_bytes = 0;
    mc->fa18_done.store(false, std::memory_order_relaxed);
    mc->fa18_busy.store(false, std::memory_order_release);
}
} // namespace

void llama_moe_hot_turn_finish(const void * owner) {
    moe_cache * mc = g_cache;
    if (!mc || !mc->adapt || owner != mc->owner || !mc->fa18_busy.load(std::memory_order_acquire)) {
        return;
    }
    std::lock_guard<std::mutex> slk(mc->step_mtx);
    hot_turn_join(mc, mc->fa18_ovl < 2);
}

bool llama_moe_hot_turn_start_on() {
    const moe_cache * mc = g_cache;
    return mc && mc->adapt && mc->dc_on && mc->l17_pre > 0;
}

bool llama_moe_hot_turn_pending() {
    moe_cache * mc = g_cache;
    if (!mc || !mc->adapt || !mc->dc_on || mc->l17_pre == 0) {
        return false;
    }
    std::lock_guard<std::mutex> slk(mc->step_mtx); // the pass's copies touch none of this: no join
    for (const auto & ls : mc->layers) {
        if (ls.dseed_any) {
            return true;
        }
    }
    return false;
}

bool llama_moe_hot_turn_start(const void * owner) {
    moe_cache * mc = g_cache;
    if (!mc || !mc->adapt || !mc->dc_on || mc->l17_pre == 0 || owner != mc->owner) {
        return false;
    }
    std::lock_guard<std::mutex> slk(mc->step_mtx);
    hot_turn_join(mc); // [TAG_FN_L18_FIRSTANS_OVERLAP]
    bool seeded = false;
    for (const auto & ls : mc->layers) {
        seeded = seeded || ls.dseed_any;
    }
    if (!seeded) {
        return false; // a streamed prompt (no observer) or a decode batch
    }
    const int64_t t0 = ggml_time_us();
    // the worker takes no new batch and puts its running one down: its slots in flight stay busy below, and the refill
    // uses its upload stream alone
    {
        std::unique_lock<std::mutex> lk(mc->wmtx);
        mc->paused = true;
        hot_set_yield(mc);
        mc->wcv.notify_all();
        mc->wcv_idle.wait(lk, [mc]() { return !mc->worker_busy; });
    }
    const int64_t t_idle = ggml_time_us();
    hot_adapt_publish(mc);

    // the counts: the last answer's fade, the prompt comes in as l17_sn decode steps of its mix
    for (auto & ls : mc->layers) {
        if (ls.dcnt.empty()) {
            continue;
        }
        if (mc->l17_td < 1.0f) {
            llama_moe_decay_apply(ls.dcnt, mc->l17_td);
        }
        if (ls.dseed_any) {
            llama_moe_decay_fold_seed(ls.dcnt, ls.dseed, ls.dseed_tok > 0.0f ? mc->l17_sn/ls.dseed_tok : 0.0f);
        }
        ls.dseed_any = false;
        ls.dseed_tok = 0.0f;
    }

    // one pass of the decay rules over the layers outside a pool (as hot_adapt_decay), best gain first, under the budget
    std::vector<llama_moe_decay_swap> swaps;
    std::vector<uint8_t> slot_busy;
    std::vector<uint8_t> expert_busy;
    const llama_moe_host_ready_fn ready = g_host_ready.load(); // [TAG_FN_L15_TIER]
    void * ready_ud = g_host_ready_ud.load();
    for (int li = 0; li < (int) mc->layers.size(); ++li) {
        const layer_state & ls = mc->layers[li];
        if (ls.pub.n_slots == 0 || ls.dcnt.empty() || ls.out() || ls.pool >= 0) {
            continue;
        }
        slot_busy.assign(ls.pub.n_slots, 0);
        for (int32_t s = 0; s < ls.pub.n_slots; ++s) {
            slot_busy[s] = ls.slot_in_flight[s] || (!ls.slot_out.empty() && ls.slot_out[s]) ? 1 : 0;
            if (!slot_busy[s] && ready && ls.slot_expert[s] >= 0 && !ready(ls.pub.up_src, ls.slot_expert[s], ready_ud)) {
                slot_busy[s] = 1;
            }
        }
        expert_busy.assign(ls.dcnt.size(), 0);
        for (size_t e = 0; e < ls.dcnt.size(); ++e) {
            expert_busy[e] = ls.expert_slot[e] >= 0 || ls.expert_in_flight[e] ? 1 : 0;
        }
        const size_t lb = hot_layer_bytes(ls);
        llama_moe_decay_pairs(li, ls.dcnt, ls.slot_expert, slot_busy, expert_busy, mc->dc, swaps, mc->l17_pre/std::max<size_t>(1, lb) + 1);
    }
    llama_moe_decay_order(swaps);

    // evict, then the new expert owns the slot in the bookkeeping; the tables name it only after its bytes landed
    std::vector<refill_job>           jobs;
    std::vector<llama_moe_decay_swap> taken;
    size_t used = 0;
    for (const auto & w : swaps) {
        layer_state & ls = mc->layers[w.layer];
        const size_t b = hot_layer_bytes(ls);
        if (used + b > mc->l17_pre) {
            break;
        }
        ggml_tensor *       dsts[3] = { ls.pub.up_c,   ls.pub.gate_c,   ls.pub.down_c   };
        const ggml_tensor * srcs[3] = { ls.pub.up_src, ls.pub.gate_src, ls.pub.down_src };
        bool ok = true;
        for (int k = 0; k < 3 && ok; ++k) {
            ok = hot_slice_ok(dsts[k], srcs[k], w.expert, w.slot);
        }
        if (!ok) {
            continue;
        }
        if (w.victim >= 0) {
            ls.expert_slot[w.victim] = -1;
            hot_adapt_set_entry(mc, ls, w.victim, ls.pub.n_slots);
        }
        if (mc->fa18_ovl > 0) {
            // [TAG_FN_L18_FIRSTANS_OVERLAP] in flight, as the worker's admissions, until hot_turn_join names it
            ls.slot_expert[w.slot]        = -1;
            ls.slot_in_flight[w.slot]     = true;
            ls.expert_in_flight[w.expert] = true;
        } else {
            ls.slot_expert[w.slot]   = w.expert;
            ls.expert_slot[w.expert] = w.slot;
        }
        for (int k = 0; k < 3; ++k) {
            const size_t sz = srcs[k]->nb[2];
            jobs.push_back({ dsts[k], (size_t) w.slot*dsts[k]->nb[2], (const char *) srcs[k]->data + (size_t) w.expert*sz, sz });
        }
        taken.push_back(w);
        used += b;
    }
    // the victims leave the device table before any slot is overwritten (no graph runs now; the next one reads the new table)
    if (mc->tbl_dirty) {
        ggml_backend_tensor_set(mc->tbl_all, mc->tbl_mirror.data(), 0, mc->tbl_mirror.size()*sizeof(int32_t));
        mc->tbl_dirty = false;
    }
    if (mc->fa18_ovl && !jobs.empty()) {
        // [TAG_FN_L18_FIRSTANS_OVERLAP] the same copies on a thread of their own; hot_turn_join names the experts
        size_t b = 0;
        for (const auto & j : jobs) {
            b += j.size;
        }
        mc->fa18_jobs     = std::move(jobs);
        mc->fa18_taken    = std::move(taken);
        mc->fa18_bytes    = b;
        mc->fa18_t0       = t0;
        mc->fa18_t_idle   = t_idle;
        mc->fa18_t_launch = ggml_time_us();
        mc->fa18_t_done   = mc->fa18_t_launch;
        mc->fa18_done.store(false, std::memory_order_relaxed);
        mc->fa18_busy.store(true, std::memory_order_release);
        llama_fn_stall_mark("prompt-end pass copies launched (jobs, MiB)", (int64_t) mc->fa18_jobs.size(), (int64_t) (b >> 20)); // [TAG_FN_L18_FIRSTANS_STALL]
        mc->fa18_th = std::thread([mc]() {
            llama_fn_thread_name("hot-pass"); // [TAG_FN_L18_FIRSTANS_STALL]
            (void) hot_refill(mc, mc->fa18_jobs, mc->l17_threads);
            mc->fa18_t_done = ggml_time_us();
            mc->fa18_done.store(true, std::memory_order_release);
        });
        return true;
    }
    const size_t bytes = jobs.empty() ? 0 : hot_refill(mc, jobs, mc->l17_threads);
    for (const auto & w : taken) {
        hot_adapt_set_entry(mc, mc->layers[w.layer], w.expert, w.slot);
    }
    if (mc->tbl_dirty) {
        ggml_backend_tensor_set(mc->tbl_all, mc->tbl_mirror.data(), 0, mc->tbl_mirror.size()*sizeof(int32_t));
        mc->tbl_dirty = false;
    }
    {
        std::lock_guard<std::mutex> lk(mc->wmtx);
        mc->paused = false;
        hot_set_yield(mc);
        mc->wcv.notify_all();
    }
    const double ms = (ggml_time_us() - t0)/1000.0;
    mc->l17_n++;
    mc->l17_swaps += taken.size();
    mc->l17_bytes += bytes;
    mc->l17_ms    += ms;
    if (mc->l17_n <= 3 || mc->l17_n % 32 == 0) {
        LLAMA_LOG_INFO("moe-hot: [TAG_FN_L17_START_PRE] prompt end %" PRIu64 ": %zu experts (%.0f MiB) swapped in from the prompt's "
                "routing in %.1f ms (worker idle after %.1f ms; %.1f GiB/s); %.1f ms and %.0f MiB per prompt so far\n", mc->l17_n,
                taken.size(), bytes/1048576.0, ms, (t_idle - t0)/1000.0, ms > 0 ? bytes/1073741824.0/(ms/1000.0) : 0.0,
                mc->l17_ms/mc->l17_n, mc->l17_bytes/1048576.0/mc->l17_n);
    }
    return true;
}

// [TAG_FN_L18_KEEPCTX_HOT] -----------------------------------------------------------------------------------------

bool llama_moe_hot_counts_get(const void * owner, std::vector<float> & out) {
    moe_cache * mc = g_cache;
    out.clear();
    if (!mc || !mc->adapt || !mc->dc_on || owner != mc->owner) {
        return false;
    }
    std::lock_guard<std::mutex> slk(mc->step_mtx);
    for (const auto & ls : mc->layers) {
        out.insert(out.end(), ls.dcnt.begin(), ls.dcnt.end());
    }
    return !out.empty();
}

bool llama_moe_hot_counts_set(const void * owner, const std::vector<float> & in) {
    moe_cache * mc = g_cache;
    if (!mc || !mc->adapt || !mc->dc_on || owner != mc->owner) {
        return false;
    }
    std::lock_guard<std::mutex> slk(mc->step_mtx);
    size_t n = 0;
    for (const auto & ls : mc->layers) {
        n += ls.dcnt.size();
    }
    if (n != in.size() || n == 0) {
        return false;
    }
    size_t o = 0;
    for (auto & ls : mc->layers) {
        std::copy(in.begin() + o, in.begin() + o + ls.dcnt.size(), ls.dcnt.begin());
        o += ls.dcnt.size();
    }
    return true;
}

// [TAG_FN_L3_VRAM_CBUF] ---------------------------------------------------------------------------------------------

size_t llama_moe_hot_cbuf_tail_bytes() {
    const moe_cache * mc = g_cache;
    return mc && mc->cb_buf ? mc->cb_hi - mc->cb_lo : 0;
}

bool llama_moe_hot_cbuf_out() {
    const moe_cache * mc = g_cache;
    return mc && mc->cb_buf && mc->cb_out;
}

size_t llama_moe_hot_cbuf_release(const void * owner, size_t need) {
    moe_cache * mc = g_cache;
    if (!mc || !mc->cb_buf) {
        return 0;
    }
    if (mc->adapt && owner != mc->owner) {
        return 0;
    }
    // [TAG_FN_L20_BIGPROMPT_TAIL] need > 0: only the top of the tail, at least need bytes (whole mapped segments); a later
    // call for more (need 0: the whole tail) gives back the rest. Already out at or below that offset: nothing to do
    const size_t a    = llama_fn_cbuf_part_lo(mc->cb_lo, mc->cb_hi, mc->cb_seg, need);
    const size_t prev = mc->cb_out ? mc->cb_rel : mc->cb_hi;
    if (prev <= a) {
        return 0;
    }
    mc->l22_part = false; // [TAG_FN_L22_LENDCLASS] every hidden layer is refilled at the restore, as before
    std::lock_guard<std::mutex> slk(mc->step_mtx); // [TAG_FN_L3_POLICY_STATE] the tail's bookkeeping, as the lend
    hot_turn_join(mc); // [TAG_FN_L18_FIRSTANS_OVERLAP] a pass in flight lands first
    const int64_t t0 = ggml_time_us();
    // a layer is lent when any of its bytes lie at or above a minus one granule (a kernel that reads a little past a slot's
    // end never reaches the range that goes out); the whole tail: every tail layer, as before
    const uint8_t * base = (const uint8_t *) ggml_backend_buffer_get_base(mc->cb_buf);
    const size_t    lim  = a > mc->cb_lo + mc->cb_gran ? a - mc->cb_gran : mc->cb_lo;
    std::vector<layer_state *> newly;
    for (auto & ls : mc->layers) {
        if (!ls.cb_tail || ls.cb_lent) {
            continue;
        }
        size_t end = 0;
        for (const ggml_tensor * t : { ls.pub.up_c, ls.pub.gate_c, ls.pub.down_c }) {
            end = std::max(end, (size_t) ((const uint8_t *) t->data - base) + ggml_nbytes(t));
        }
        if (a == mc->cb_lo || end > lim) {
            ls.cb_lent = true;
            newly.push_back(&ls);
        }
    }
    hot_drain_out(mc);
    mc->cb_async_left = 0; // the refill jobs not uploaded yet were dropped; their slots keep the expert for the next restore
    for (layer_state * ls : newly) {
        hot_write_tables(mc, *ls);
    }
    const size_t bytes = prev - a;
    const bool   more  = mc->cb_out;
    if (!mc->cb_unmap(mc->cb_buf, a, bytes)) {
        // nothing was given back: the lent layers stay out until the restore (which then maps nothing new)
        LLAMA_LOG_WARN("moe-hot: [TAG_FN_L3_VRAM_CBUF] the tail could not give its VRAM back - its layers sit out this prompt\n");
        mc->cb_out = true;
        mc->cb_rel = std::min(prev, a);
        return 0;
    }
    if (a > mc->cb_lo || more) {
        mc->cb_n_part++;
        if (mc->cb_n_part <= 3 || mc->cb_n_part % 64 == 0) {
            LLAMA_LOG_INFO("moe-hot: [TAG_FN_L20_BIGPROMPT_TAIL] tail release %" PRIu64 ": %.0f MiB %sfrom +%.0f MiB (%zu layers lent "
                    "now, %.0f of %.0f MiB of the tail out, %.0f MiB asked)\n", mc->cb_n_part, bytes/1048576.0, more ? "more " : "",
                    (a - mc->cb_lo)/1048576.0, newly.size(), (mc->cb_hi - a)/1048576.0, (mc->cb_hi - mc->cb_lo)/1048576.0,
                    need/1048576.0);
        }
    }
    mc->cb_out = true;
    mc->cb_rel = a;
    mc->cb_n_rel++;
    mc->cb_rel_ms += (ggml_time_us() - t0)/1000.0;
    mc->l22_rel_b += bytes;
    return bytes;
}

// [TAG_FN_L22_LENDCLASS] see llama-moecache.h: the layers a release of need_hide would lend leave the tables (the graphs
// are those of that release, so the outputs are too), but only the top need_unmap bytes go back to the driver; the layers
// below keep their bytes, and the restore refills only what reaches into the unmapped part (and dirty slots)
size_t llama_moe_hot_cbuf_release2(const void * owner, size_t need_hide, size_t need_unmap) {
    moe_cache * mc = g_cache;
    if (!mc || !mc->cb_buf || (mc->adapt && owner != mc->owner)) {
        return 0;
    }
    if (need_unmap == 0 || (need_hide > 0 && need_unmap >= need_hide)) {
        return llama_moe_hot_cbuf_release(owner, need_hide); // nothing narrower: the plain release
    }
    const size_t a_h  = llama_fn_cbuf_part_lo(mc->cb_lo, mc->cb_hi, mc->cb_seg, need_hide);
    const size_t a_u  = std::max(a_h, llama_fn_cbuf_part_lo(mc->cb_lo, mc->cb_hi, mc->cb_seg, need_unmap));
    const size_t prev = mc->cb_out ? mc->cb_rel : mc->cb_hi;
    std::lock_guard<std::mutex> slk(mc->step_mtx);
    hot_turn_join(mc);
    const int64_t t0 = ggml_time_us();
    const uint8_t * base = (const uint8_t *) ggml_backend_buffer_get_base(mc->cb_buf);
    const size_t    lim  = a_h > mc->cb_lo + mc->cb_gran ? a_h - mc->cb_gran : mc->cb_lo;
    std::vector<layer_state *> newly;
    for (auto & ls : mc->layers) {
        if (!ls.cb_tail || ls.cb_lent) {
            continue;
        }
        size_t end = 0;
        for (const ggml_tensor * t : { ls.pub.up_c, ls.pub.gate_c, ls.pub.down_c }) {
            end = std::max(end, (size_t) ((const uint8_t *) t->data - base) + ggml_nbytes(t));
        }
        if (a_h == mc->cb_lo || end > lim) {
            ls.cb_lent = true;
            newly.push_back(&ls);
        }
    }
    if (newly.empty() && prev <= a_u) {
        return 0;
    }
    const bool first = !mc->cb_out;
    hot_drain_out(mc);
    mc->cb_async_left = 0;
    for (layer_state * ls : newly) {
        hot_write_tables(mc, *ls);
    }
    // the drain published the worker's last batch into the host tables: the device tables follow now, so a narrow graph
    // of this FULL period (the tail, a window) never sees a host table that skips what the device table misses
    if (mc->tbl_dirty && mc->tbl_all) {
        ggml_backend_tensor_set(mc->tbl_all, mc->tbl_mirror.data(), 0, mc->tbl_mirror.size()*sizeof(int32_t));
        mc->tbl_dirty = false;
    }
    size_t bytes = 0;
    if (prev > a_u) {
        if (!mc->cb_unmap(mc->cb_buf, a_u, prev - a_u)) {
            LLAMA_LOG_WARN("moe-hot: [TAG_FN_L22_LENDCLASS] the tail could not give its VRAM back - its layers sit out this prompt\n");
            mc->cb_out = true;
            mc->cb_rel = std::min(prev, a_u);
            mc->l22_part = false; // refill everything hidden at the restore
            return 0;
        }
        bytes = prev - a_u;
        mc->cb_rel = a_u;
    }
    mc->cb_out   = true;
    mc->l22_part = first ? true : mc->l22_part; // a later plain release clears it
    mc->cb_n_rel++;
    mc->cb_rel_ms += (ggml_time_us() - t0)/1000.0;
    mc->l22_rel_b += bytes;
    return bytes;
}

// [TAG_FN_L8_KVLEND] -------------------------------------------------------------------------------------------------

namespace {
std::atomic<uint64_t> g_hot_gen{0};
}

// [TAG_FN_L22_KEEPLEND] see llama-moecache.h
void llama_moe_hot_lend_hide(const void * owner, bool hide) {
    moe_cache * mc = g_cache;
    if (!mc || (mc->adapt && owner != mc->owner) || mc->l22_hide == hide) {
        return;
    }
    std::lock_guard<std::mutex> slk(mc->step_mtx);
    mc->l22_hide = hide;
    // narrow graphs come next: the host and device tables agree (a drain at the lend may have published into the host
    // tables only; the owner's graphs are synchronized)
    if (mc->tbl_dirty && mc->tbl_all) {
        ggml_backend_tensor_set(mc->tbl_all, mc->tbl_mirror.data(), 0, mc->tbl_mirror.size()*sizeof(int32_t));
        mc->tbl_dirty = false;
    }
    g_hot_gen.fetch_add(1, std::memory_order_acq_rel); // no graph built under the other state is reused
}

void llama_moe_hot_set_kv_lend(size_t want, size_t keep, int n_trunk) {
    g_kv_want    = want;
    g_kv_keep    = keep;
    g_kv_n_trunk = n_trunk;
}

void llama_moe_hot_set_cb_borrow(size_t bytes) { // [TAG_FN_L21_BIGUB]
    g_kv_borrow = bytes;
}

// [TAG_FN_L22_BIGUB2] see llama-moecache.h
void llama_moe_hot_l22_borrow_1seq(bool on) {
    g_kv_borrow_1seq = on;
}

// [TAG_FN_L22_UBPLAN] P10 R2, see llama-moecache.h
bool llama_moe_hot_borrow_range(const ggml_tensor * up_exps, int32_t * lo, int32_t * hi) {
    std::unique_lock<std::mutex> ilk(g_init_mtx, std::try_to_lock);
    if (!ilk.owns_lock()) {
        return false;
    }
    moe_cache * mc = g_cache;
    ilk.unlock();
    if (!mc || !mc->hot || !mc->sl_on || mc->sb_bytes == 0) {
        return false;
    }
    const auto it = mc->by_up_src.find(up_exps);
    if (it == mc->by_up_src.end()) {
        return false;
    }
    const layer_state & ls = mc->layers[it->second];
    if (ls.sl_bbase < 0 || ls.sl_base <= ls.sl_bbase) {
        return false;
    }
    if (lo) {
        *lo = ls.sl_bbase;
    }
    if (hi) {
        *hi = ls.sl_base;
    }
    return true;
}

size_t llama_moe_hot_cb_borrow_bytes() {
    const moe_cache * mc = g_cache;
    return mc && mc->sl_on ? mc->sb_bytes : 0;
}

size_t llama_moe_hot_cb_borrow(const void * owner) {
    moe_cache * mc = g_cache;
    return mc ? hot_sb_release(mc, owner) : 0;
}

void llama_moe_hot_cb_unborrow(const void * owner) {
    moe_cache * mc = g_cache;
    if (mc) {
        hot_sb_restore(mc, owner);
    }
}

size_t llama_moe_hot_kv_bytes() {
    const moe_cache * mc = g_cache;
    return mc ? mc->kv_bytes : 0;
}

size_t llama_moe_hot_kv_out() {
    const moe_cache * mc = g_cache;
    return mc ? mc->kv_out : 0;
}

uint64_t llama_moe_hot_generation() {
    return g_hot_gen.load(std::memory_order_acquire);
}

void llama_moe_hot_generation_bump() {
    g_hot_gen.fetch_add(1, std::memory_order_acq_rel);
}

size_t llama_moe_hot_kv_release(const void * owner, size_t need) {
    moe_cache * mc = g_cache;
    if (mc && mc->sl_on) {
        return hot_sl_release(mc, owner, need); // [TAG_FN_L10_SLOTLEND]
    }
    if (!mc || mc->kv_bytes == 0 || !mc->cb_buf || mc->kv_out >= need) {
        return mc ? mc->kv_out : 0;
    }
    if (mc->adapt && owner != mc->owner) {
        return mc->kv_out;
    }
    std::lock_guard<std::mutex> slk(mc->step_mtx);
    hot_turn_join(mc); // [TAG_FN_L18_FIRSTANS_OVERLAP] a pass in flight lands first
    const int64_t t0 = ggml_time_us();
    std::vector<layer_state *> rel;
    size_t out = mc->kv_out;
    for (auto it = mc->layers.rbegin(); it != mc->layers.rend() && out < need; ++it) { // the top KV layer first
        if (it->kv_tail && !it->kv_lent) {
            it->kv_lent = true;
            rel.push_back(&*it);
            out += it->kv_hi - it->kv_lo;
        }
    }
    if (rel.empty()) {
        return mc->kv_out;
    }
    hot_drain_out(mc); // no upload into them any more, the worker's batch landed
    for (auto * ls : rel) {
        hot_write_tables(mc, *ls);
    }
    if (mc->up_backend) {
        ggml_backend_synchronize(mc->up_backend);
    }
    size_t n_ok  = 0;
    size_t given = 0;
    for (auto * ls : rel) {
        if (!mc->cb_unmap(mc->cb_buf, ls->kv_lo, ls->kv_hi - ls->kv_lo)) {
            LLAMA_LOG_WARN("moe-hot: [TAG_FN_L8_KVLEND] blk.%d could not give its VRAM back - it sits out\n", ls->pub.il);
            continue; // out (the host computes it), nothing given back
        }
        mc->kv_out += ls->kv_hi - ls->kv_lo;
        given      += ls->kv_hi - ls->kv_lo;
        n_ok++;
    }
    g_hot_gen.fetch_add(1, std::memory_order_acq_rel);
    mc->kv_n_rel++;
    const double ms = (ggml_time_us() - t0)/1000.0;
    mc->kv_rel_ms += ms;
    LLAMA_LOG_INFO("moe-hot: [TAG_FN_L8_KVLEND] %zu layers gave %.0f MiB to the KV cache in %.1f ms (%.0f of %.0f MiB out)\n",
            n_ok, given/1048576.0, ms, mc->kv_out/1048576.0,
            mc->kv_bytes/1048576.0);
    return mc->kv_out;
}

size_t llama_moe_hot_kv_restore(const void * owner, size_t keep) {
    moe_cache * mc = g_cache;
    if (mc && mc->sl_on) {
        return hot_sl_restore(mc, owner, keep); // [TAG_FN_L10_SLOTLEND]
    }
    if (!mc || mc->kv_bytes == 0 || !mc->cb_buf || mc->kv_out == 0) {
        return mc ? mc->kv_out : 0;
    }
    if (mc->adapt && owner != mc->owner) {
        return mc->kv_out;
    }
    std::lock_guard<std::mutex> slk(mc->step_mtx);
    hot_turn_join(mc); // [TAG_FN_L18_FIRSTANS_OVERLAP] a pass in flight lands first
    const int64_t t0 = ggml_time_us();
    size_t n_res = 0;
    size_t bytes = 0;
    for (auto & ls : mc->layers) { // the lowest KV layer first (the release went top-down)
        if (!ls.kv_tail || !ls.kv_lent) {
            continue;
        }
        const size_t r = ls.kv_hi - ls.kv_lo;
        if (mc->kv_out < keep + r) {
            break;
        }
        (void) mc->cb_unmap(mc->cb_buf, ls.kv_lo, r); // a failed release may have left it mapped
        if (!mc->cb_map(mc->cb_buf, ls.kv_lo, r)) {
            LLAMA_LOG_WARN("moe-hot: [TAG_FN_L8_KVLEND] mapping blk.%d again failed: it stays on the host until the next try\n",
                    ls.pub.il);
            break;
        }
        ggml_backend_tensor_memset(ls.kv_raw, 0, 0, ggml_nbytes(ls.kv_raw)); // zero slot, padding and empty slots read 0
        std::vector<refill_job> jobs;
        ggml_tensor *       dsts[3] = { ls.pub.up_c,   ls.pub.gate_c,   ls.pub.down_c   };
        const ggml_tensor * srcs[3] = { ls.pub.up_src, ls.pub.gate_src, ls.pub.down_src };
        for (int32_t s = 0; s < ls.pub.n_slots; ++s) {
            const int32_t e = ls.slot_expert[s];
            if (e < 0) {
                continue;
            }
            for (int k = 0; k < 3; ++k) {
                const size_t sz  = srcs[k]->nb[2];
                const size_t off = (size_t) s*dsts[k]->nb[2];
                if (off + sz > ggml_nbytes(dsts[k]) || (size_t) (e + 1)*sz > ggml_nbytes(srcs[k])) {
                    continue;
                }
                jobs.push_back({ dsts[k], off, (const char *) srcs[k]->data + (size_t) e*sz, sz });
            }
        }
        bytes += hot_refill(mc, jobs);
        ls.kv_lent = false;
        hot_write_tables(mc, ls);
        mc->kv_out -= r;
        n_res++;
    }
    if (n_res > 0) {
        g_hot_gen.fetch_add(1, std::memory_order_acq_rel);
        mc->kv_n_res++;
        const double ms = (ggml_time_us() - t0)/1000.0;
        mc->kv_res_ms += ms;
        LLAMA_LOG_INFO("moe-hot: [TAG_FN_L8_KVLEND] %zu layers came back from the KV cache (%.2f GiB refilled) in %.1f ms (%.0f MiB "
                "still out)\n", n_res, bytes/1073741824.0, ms, mc->kv_out/1048576.0);
    }
    return mc->kv_out;
}

// [TAG_FN_L17_VISION_LEND] -------------------------------------------------------------------------------------------
// The KV cache takes groups (layers) from the top and gives back the lowest first, so the projector's share is the
// range just below the KV cache's: its give maps exactly what its take unmapped (an encode runs between two graphs,
// no KV growth in between).

void llama_moe_hot_vis_set(size_t bytes) {
    if (moe_cache * mc = g_cache) {
        mc->vis_bytes = bytes;
    }
}

size_t llama_moe_hot_vis_bytes() {
    const moe_cache * mc = g_cache;
    return mc ? mc->vis_bytes : 0;
}

size_t llama_moe_hot_vis_take(size_t bytes) {
    moe_cache * mc = g_cache;
    if (!mc || mc->vis_bytes == 0 || mc->vis_out > 0 || bytes == 0) {
        return 0;
    }
    const int64_t t0     = ggml_time_us();
    const size_t  before = mc->kv_out;
    const size_t  after  = llama_moe_hot_kv_release(mc->owner, before + bytes);
    mc->vis_out = after > before ? after - before : 0;
    mc->vis_n++;
    const double ms = (ggml_time_us() - t0)/1000.0;
    mc->vis_take_ms += ms;
    if (mc->vis_out < bytes) {
        LLAMA_LOG_WARN("moe-hot: [TAG_FN_L17_VISION_LEND] the projector asked for %.0f MiB and got %.0f MiB (KV cache %.0f of %.0f "
                "MiB out): the rest comes from free VRAM\n", bytes/1048576.0, mc->vis_out/1048576.0, before/1048576.0,
                mc->kv_bytes/1048576.0);
    }
    LLAMA_LOG_INFO("moe-hot: [TAG_FN_L17_VISION_LEND] encode %" PRIu64 ": the hot set gave %.0f MiB to the projector in %.1f ms\n",
            mc->vis_n, mc->vis_out/1048576.0, ms);
    return mc->vis_out;
}

void llama_moe_hot_vis_give() {
    moe_cache * mc = g_cache;
    if (!mc || mc->vis_out == 0) {
        return;
    }
    const int64_t t0    = ggml_time_us();
    const size_t  keep  = mc->kv_out > mc->vis_out ? mc->kv_out - mc->vis_out : 0;
    const size_t  after = llama_moe_hot_kv_restore(mc->owner, keep);
    const double  ms    = (ggml_time_us() - t0)/1000.0;
    const size_t  left  = after > keep ? after - keep : 0;
    mc->vis_give_ms += ms;
    if (left > 0) {
        // still out: the KV cache's next release counts it, its next restore maps it
        LLAMA_LOG_WARN("moe-hot: [TAG_FN_L17_VISION_LEND] %.0f MiB of the projector's share could not be mapped again\n",
                left/1048576.0);
    }
    LLAMA_LOG_INFO("moe-hot: [TAG_FN_L17_VISION_LEND] encode %" PRIu64 ": %.0f MiB back in the hot set in %.1f ms (take %.1f, give "
            "%.1f ms on average)\n", mc->vis_n, (mc->vis_out > left ? mc->vis_out - left : 0)/1048576.0, ms,
            mc->vis_take_ms/(double) mc->vis_n, mc->vis_give_ms/(double) mc->vis_n);
    mc->vis_out = 0;
}

// [TAG_FN_L4_EXIT] ---------------------------------------------------------------------------------------------------

void llama_moe_hot_idle(const void * owner) {
    moe_cache * mc = g_cache;
    if (!mc || !mc->adapt || owner != mc->owner || !mc->worker.joinable()) {
        return;
    }
    std::lock_guard<std::mutex> slk(mc->step_mtx);
    hot_turn_join(mc); // [TAG_FN_L18_FIRSTANS_OVERLAP] a pass in flight lands first
    std::unique_lock<std::mutex> lk(mc->wmtx);
    mc->paused = true;
    hot_set_yield(mc);
    mc->wcv.notify_all(); // a bucket wait gives up
    // the batch in flight holds at most UP_INFLIGHT_MAX bytes; the bound only guards against a stuck copy
    mc->wcv_idle.wait_for(lk, std::chrono::seconds(10), [mc]() { return !mc->worker_busy; });
    lk.unlock();
    // [TAG_FN_L17_START_STATE] the learned set to its file now and then, so a stop without the owner's end keeps it
    if (mc->l17_st_every_us > 0 && mc->ad_steps >= mc->l17_st_steps + 64) {
        const int64_t now = ggml_time_us();
        if (mc->l17_st_last_us == 0 || now - mc->l17_st_last_us >= mc->l17_st_every_us) {
            hot_state_save(mc);
            mc->l17_st_last_us = now;
            mc->l17_st_steps   = mc->ad_steps;
            if (mc->st_saves <= 3) {
                LLAMA_LOG_INFO("moe-hot: [TAG_FN_L17_START_STATE] state saved at idle in %.1f ms (%llu decode steps)\n",
                        (ggml_time_us() - now)/1000.0, (unsigned long long) mc->ad_steps);
            }
        }
    }
}

void llama_moe_hot_stop(const void * owner) {
    moe_cache * mc = g_cache;
    if (mc && owner == mc->r25_owner) {
        g_r25.shutdown(); // [TAG_FN_R25_REFILL] the copy threads end; later refills take the old path
    }
    if (!mc || !mc->adapt || owner != mc->owner || !mc->worker.joinable()) {
        return;
    }
    {
        std::lock_guard<std::mutex> slk(mc->step_mtx);
        hot_turn_join(mc); // [TAG_FN_L18_FIRSTANS_OVERLAP]
    }
    if (mc->fa18_pf_th.joinable()) {
        mc->fa18_pf_abort.store(true);
        mc->fa18_pf_th.join(); // [TAG_FN_L18_FIRSTANS_STATE] its jobs are queued (or dropped) before the worker stops
    }
    if (mc->l22_pf_th.joinable()) {
        mc->l22_pf_stop.store(true);
        mc->l22_pf_th.join(); // [TAG_FN_L22_REFILL] the refill set's prefetch (requests of 64 ranges, stops)
    }
    {
        std::lock_guard<std::mutex> lk(mc->wmtx);
        mc->stop = true;
        hot_set_yield(mc);
    }
    mc->wcv.notify_all();
    mc->worker.join();
    if (mc->up_backend) {
        ggml_backend_synchronize(mc->up_backend);
    }
    if (mc->l18_gate > 0) {
        uint64_t gw = 0, gus = 0;
        llama_moe_bridge_gate_stats(&gw, &gus);
        LLAMA_LOG_INFO("moe-hot: [TAG_FN_L18_CHURN_GATE] %" PRIu64 " gate waits, %.1f ms in all, %.0f MiB uploaded\n", gw, gus/1000.0,
                mc->up_bytes.load()/1048576.0);
        mc->l18_gate = 0;
        llama_moe_bridge_gate_enable(false); // [TAG_FN_L18_CHURN_GATE] the executors stop marking their jobs
    }
    LLAMA_LOG_INFO("moe-hot: [TAG_FN_L4_EXIT] the upload worker stopped (%zu jobs still queued, dropped)\n", mc->todo.size());
}

bool llama_moe_hot_cbuf_restore(const void * owner) {
    moe_cache * mc = g_cache;
    return llama_moe_hot_cbuf_restore_ex(owner, mc && mc->adapt && mc->cb_async);
}

bool llama_moe_hot_cbuf_restore_ex(const void * owner, bool async_req) {
    moe_cache * mc = g_cache;
    if (mc) {
        mc->l22_pf_stop.store(true); // [TAG_FN_L22_REFILL] r22 T5: the refill set's read-ahead ends here
    }
    if (!mc || !mc->cb_buf || !mc->cb_out) {
        return true;
    }
    if (mc->adapt && owner != mc->owner) {
        return false;
    }
    std::lock_guard<std::mutex> slk(mc->step_mtx); // [TAG_FN_L3_POLICY_STATE] the tail's bookkeeping, as the lend
    hot_turn_join(mc); // [TAG_FN_L18_FIRSTANS_OVERLAP] a pass in flight lands first
    if (!mc->l17_burst_start) { // [TAG_FN_L17_START_BURST] the window opens at the start only
        mc->dc_burst_left = std::max(mc->dc_burst_left, mc->dc_burst_passes); // [TAG_FN_L15_BURST] a big prompt ran
    }
    const int64_t t0 = ggml_time_us();
    // [TAG_FN_L20_BIGPROMPT_TAIL] the part that is out: [cb_rel, cb_hi) (the whole tail unless a prompt took only its top)
    const size_t  r0 = mc->cb_rel < mc->cb_hi && mc->cb_rel >= mc->cb_lo ? mc->cb_rel : mc->cb_lo;
    const size_t  n  = mc->cb_hi - r0;
    // a failed release may have left a part mapped, and map() refuses a mapped range: give that back first (nothing
    // reads the tail while it is out; unmap skips what is not mapped)
    (void) mc->cb_unmap(mc->cb_buf, r0, n);
    if (!mc->cb_map(mc->cb_buf, r0, n)) {
        mc->cb_n_fail++;
        if (mc->cb_n_fail == 1 || mc->cb_n_fail % 64 == 0) {
            LLAMA_LOG_WARN("moe-hot: [TAG_FN_L3_VRAM_CBUF] mapping the tail again failed (%" PRIu64 " times): its layers stay "
                    "on the host until the next try\n", mc->cb_n_fail);
        }
        return false;
    }
    // fresh memory: zero slots, padding and empty slots read 0, then the resident experts come back
    ggml_backend_tensor_memset(mc->cb_raw, 0, r0 - mc->cb_lo, n); // [TAG_FN_L20_BIGPROMPT_TAIL] the part mapped again
    size_t bytes = 0;
    bool async = mc->adapt && async_req && mc->worker.joinable(); // [TAG_FN_L22_ENDOVERLAP] async_req: also per call
    // [TAG_FN_L22_LENDCLASS] after a narrow class's release: a hidden layer's slot below the unmapped part kept its bytes
    // (nothing writes an out layer's slots; an upload dropped meanwhile marked its slot dirty), so it needs no refill
    const bool      part    = mc->l22_part;
    const uint8_t * cb_base = (const uint8_t *) ggml_backend_buffer_get_base(mc->cb_buf);
    auto intact = [&](const layer_state & ls, int32_t s) -> bool {
        if (!part || ((size_t) s < ls.slot_dirty.size() && ls.slot_dirty[s])) {
            return false;
        }
        const ggml_tensor * ts[3] = { ls.pub.up_c, ls.pub.gate_c, ls.pub.down_c };
        const ggml_tensor * ss[3] = { ls.pub.up_src, ls.pub.gate_src, ls.pub.down_src };
        for (int k = 0; k < 3; ++k) {
            if ((const uint8_t *) ts[k]->data + (size_t) s*ts[k]->nb[2] + ss[k]->nb[2] > cb_base + r0) {
                return false;
            }
        }
        return true;
    };
    // [TAG_FN_L22_ENDOVERLAP] r22 T6 fix: an asked-for asynchronous refill above L22_ASYNC_REFILL_MAX runs synchronously here
    // (the copy threads, ~0.26 s for 4 GiB) instead of landing through the upload worker for 1-1.5 s during the answer
    if (async && async_req && !mc->cb_async) {
        size_t need = 0;
        for (const auto & ls : mc->layers) {
            if (!ls.cb_lent) {
                continue;
            }
            const size_t b = ls.pub.up_src->nb[2] + ls.pub.gate_src->nb[2] + ls.pub.down_src->nb[2];
            for (int32_t s = 0; s < ls.pub.n_slots; ++s) {
                if (ls.slot_expert[s] >= 0 && !ls.slot_in_flight[s] && !intact(ls, s)) {
                    need += b;
                }
            }
        }
        if (need > L22_ASYNC_REFILL_MAX) {
            async = false;
            mc->l22_sync_res++;
            if (mc->l22_sync_res <= 2 || mc->l22_sync_res % 16 == 0) {
                LLAMA_LOG_INFO("moe-hot: [TAG_FN_L22_ENDOVERLAP] tail refill of %.2f GiB: by the copy threads now (more than %zu MiB "
                        "would land through the upload worker for seconds while the CPU computes those experts; %" PRIu64 " so far)\n",
                        need/1073741824.0, L22_ASYNC_REFILL_MAX >> 20, mc->l22_sync_res);
            }
        }
    }
    if (async) {
        // through the upload worker, published at the owner's step boundaries as they land, the CPU computing them until
        // then: every resident slot leaves the tables and goes in flight, as for an admission (evict, upload, publish)
        std::lock_guard<std::mutex> lk(mc->wmtx);
        for (size_t li = 0; li < mc->layers.size(); ++li) {
            layer_state & ls = mc->layers[li];
            if (!ls.cb_lent) { // [TAG_FN_L20_BIGPROMPT_TAIL] the lent tail layers (every tail layer after a whole release)
                continue;
            }
            const size_t b = ls.pub.up_src->nb[2] + ls.pub.gate_src->nb[2] + ls.pub.down_src->nb[2];
            for (int32_t s = 0; s < ls.pub.n_slots; ++s) {
                const int32_t e = ls.slot_expert[s];
                if (e < 0 || ls.slot_in_flight[s] || intact(ls, s)) {
                    continue;
                }
                ls.slot_expert[s]     = -1;
                ls.expert_slot[e]     = -1;
                ls.slot_in_flight[s]  = true;
                ls.expert_in_flight[e] = true;
                upload_job j;
                j.layer_idx = li;
                j.expert    = e;
                j.slot      = s;
                j.cb        = true;
                mc->todo.push_back(j);
                bytes += b;
                mc->cb_async_left++; // published by this thread only, so counted before any can land
            }
        }
        mc->cb_async_t0 = t0;
    } else {
        std::vector<refill_job> jobs;
        for (auto & ls : mc->layers) {
            if (!ls.cb_lent) { // [TAG_FN_L20_BIGPROMPT_TAIL]
                continue;
            }
            ggml_tensor *       dsts[3] = { ls.pub.up_c,   ls.pub.gate_c,   ls.pub.down_c   };
            const ggml_tensor * srcs[3] = { ls.pub.up_src, ls.pub.gate_src, ls.pub.down_src };
            for (int32_t s = 0; s < ls.pub.n_slots; ++s) {
                const int32_t e = ls.slot_expert[s];
                if (e < 0 || intact(ls, s)) {
                    continue;
                }
                for (int k = 0; k < 3; ++k) {
                    const size_t sz  = srcs[k]->nb[2];
                    const size_t off = (size_t) s*dsts[k]->nb[2];
                    if (off + sz > ggml_nbytes(dsts[k]) || (size_t) (e + 1)*sz > ggml_nbytes(srcs[k])) {
                        continue; // upload_slice's bound check
                    }
                    jobs.push_back({ dsts[k], off, (const char *) srcs[k]->data + (size_t) e*sz, sz });
                }
                if ((size_t) s < ls.slot_dirty.size()) {
                    ls.slot_dirty[s] = 0; // [TAG_FN_L22_LENDCLASS] refilled below
                }
            }
        }
        int n_threads = REFILL_THREADS;
        if (const char * e = getenv("LLAMA_FN_CBUF_REFILL_THREADS"); e && atoi(e) > 0) {
            n_threads = atoi(e);
        }
        bytes = hot_refill(mc, jobs, n_threads);
    }
    for (auto & ls : mc->layers) {
        if (ls.cb_lent) { // [TAG_FN_L20_BIGPROMPT_TAIL] (every tail layer after a whole release)
            ls.cb_lent = false;
            hot_write_tables(mc, ls);
        }
    }
    mc->cb_out   = false;
    mc->cb_rel   = mc->cb_hi;
    mc->l22_part = false; // [TAG_FN_L22_LENDCLASS]
    if (async) {
        mc->wcv.notify_one();
    }
    mc->cb_n_res++;
    const double ms = (ggml_time_us() - t0)/1000.0;
    mc->cb_res_ms += ms;
    mc->cb_res_b  += bytes;
    mc->l22_res_ms += ms; // [TAG_FN_L22_PASSLOG]
    mc->l22_res_b  += bytes;
    if (mc->cb_n_res <= 2 || mc->cb_n_res % 16 == 0) {
        LLAMA_LOG_INFO("moe-hot: [TAG_FN_L3_VRAM_CBUF] restore %" PRIu64 ": %.0f MiB tail mapped in %.1f ms, %.2f GiB of resident "
                "experts %s (releases %.1f ms, restores %.1f ms on average)\n", mc->cb_n_res, n/1048576.0, ms,
                bytes/1073741824.0, async ? "queued for the upload worker (LLAMA_FN_CBUF_ASYNC=1)" : "refilled",
                mc->cb_n_rel ? mc->cb_rel_ms/mc->cb_n_rel : 0.0, mc->cb_res_ms/mc->cb_n_res);
    }
    return true;
}


// [TAG_FN_L21_VRAM_DXGI] LLAMA_FN_L21_VRAM_DXGI (Windows; qwen4exp profile: 2, else 0): under WDDM cudaMemGetInfo can miss
// what other processes hold (Strata #1911: 30,994 MiB read free with 6 GiB held elsewhere; #2065: ~400 MiB really free
// pages the card, 91 -> 3.3 t/s). The process's DXGI video memory budget minus its current usage is what the OS lets it
// add before it starts to demote. 1 = the hot set sizes from min(CUDA free, DXGI free); 2 = both are logged, CUDA's is used.
#if defined(_WIN32)
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#    include <dxgi1_4.h>

static IDXGIAdapter3 * fn_l21_dxgi_adapter(size_t mem_total) {
    static IDXGIAdapter3 * ad   = nullptr;
    static bool            done = false;
    if (done) {
        return ad;
    }
    done = true;
    HMODULE lib = LoadLibraryA("dxgi.dll");
    using create_t = HRESULT (WINAPI *)(REFIID, void **);
    create_t create = lib ? (create_t) GetProcAddress(lib, "CreateDXGIFactory1") : nullptr;
    IDXGIFactory1 * fac = nullptr;
    if (!create || FAILED(create(__uuidof(IDXGIFactory1), (void **) &fac)) || !fac) {
        LLAMA_LOG_WARN("moe-hot: [TAG_FN_L21_VRAM_DXGI] no DXGI factory, the CUDA free memory is used\n");
        return nullptr;
    }
    // the NVIDIA adapter whose dedicated memory is closest to the CUDA device's total
    size_t best_d = SIZE_MAX;
    IDXGIAdapter1 * a1 = nullptr;
    for (UINT i = 0; fac->EnumAdapters1(i, &a1) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 d = {};
        IDXGIAdapter3 * a3 = nullptr;
        if (SUCCEEDED(a1->GetDesc1(&d)) && d.VendorId == 0x10DE && SUCCEEDED(a1->QueryInterface(__uuidof(IDXGIAdapter3), (void **) &a3))) {
            const size_t ded  = (size_t) d.DedicatedVideoMemory;
            const size_t dist = ded > mem_total ? ded - mem_total : mem_total - ded;
            if (dist < best_d) {
                if (ad) {
                    ad->Release();
                }
                ad     = a3;
                best_d = dist;
            } else {
                a3->Release();
            }
        }
        a1->Release();
    }
    fac->Release();
    return ad;
}

void fn_l21_dev_memory(const llama_model * model, ggml_backend_dev_t dev, size_t * mem_free, size_t * mem_total) {
    ggml_backend_dev_memory(dev, mem_free, mem_total);
    // a LLAMA_FN_ name is a profile lookup (never put into the environment): the model's profile, else the environment
    const char * e    = model ? llama_fn_env(*model, "LLAMA_FN_L21_VRAM_DXGI") : getenv("LLAMA_FN_L21_VRAM_DXGI");
    const int    mode = e && e[0] ? atoi(e) : 0;
    if (mode <= 0) {
        return;
    }
    IDXGIAdapter3 * ad = fn_l21_dxgi_adapter(*mem_total);
    DXGI_QUERY_VIDEO_MEMORY_INFO info = {};
    if (!ad || FAILED(ad->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) {
        return;
    }
    const size_t dfree = info.Budget > info.CurrentUsage ? (size_t) (info.Budget - info.CurrentUsage) : 0;
    LLAMA_LOG_INFO("moe-hot: [TAG_FN_L21_VRAM_DXGI] free VRAM: CUDA %zu MiB, DXGI %zu MiB (budget %zu, this process %zu MiB), "
            "total %zu MiB: %s\n", *mem_free >> 20, dfree >> 20, (size_t) (info.Budget >> 20), (size_t) (info.CurrentUsage >> 20),
            *mem_total >> 20, mode == 1 && dfree < *mem_free ? "the DXGI value is used" : "the CUDA value is used");
    if (mode == 1 && dfree < *mem_free) {
        *mem_free = dfree;
    }
}
#else
void fn_l21_dev_memory(const llama_model * model, ggml_backend_dev_t dev, size_t * mem_free, size_t * mem_total) {
    (void) model;
    ggml_backend_dev_memory(dev, mem_free, mem_total);
}
#endif

// [TAG_FN_L22_REFILL] M2.4: the calling thread at MEMORY_PRIORITY_LOW, so the pages it faults in (host copies of experts
// that sit in VRAM) are the first the memory manager takes back (SetThreadInformation, Windows 8+)
#if defined(_WIN32)
static void fn_l22_mem_prio_low() {
    using set_info_t = BOOL (WINAPI *)(HANDLE, int, LPVOID, DWORD);
    static const set_info_t set_info = (set_info_t) (void *) GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "SetThreadInformation");
    if (!set_info) {
        return;
    }
    struct { ULONG MemoryPriority; } mpi = { 2 };       // MEMORY_PRIORITY_INFORMATION { MEMORY_PRIORITY_LOW }
    (void) set_info(GetCurrentThread(), 0 /* ThreadMemoryPriority */, &mpi, (DWORD) sizeof(mpi));
}

// [TAG_FN_L22_REFILL] (b) r22 T5 fix: see the declaration at the top
struct fn_l22_ws_ex { // PSAPI_WORKING_SET_EX_INFORMATION: bit 0 of attr = Valid (in the working set)
    void *    va;
    uintptr_t attr;
};
static void fn_l22_refill_prefetch_run(const std::vector<std::pair<const void *, size_t>> & rr, const std::atomic<bool> & stop,
                                       fn_l22_refill_pf_stats & st) {
    const int64_t t0 = ggml_time_us();
    st.ranges = rr.size();
    using qws_t = BOOL (WINAPI *)(HANDLE, PVOID, DWORD);
    using pvm_t = BOOL (WINAPI *)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
    static const qws_t qws = (qws_t) (void *) GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "K32QueryWorkingSetEx");
    static const pvm_t pvm = (pvm_t) (void *) GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "PrefetchVirtualMemory");
    if (pvm == nullptr || rr.empty()) {
        return;
    }
    // the ranges whose first page is out of the working set (the rest needs no read)
    std::vector<std::pair<const void *, size_t>> out;
    out.reserve(rr.size());
    if (qws != nullptr) {
        std::vector<fn_l22_ws_ex> w;
        for (size_t i0 = 0; i0 < rr.size() && !stop.load(std::memory_order_relaxed); i0 += 2048) {
            const size_t n = std::min<size_t>(2048, rr.size() - i0);
            w.resize(n);
            for (size_t i = 0; i < n; ++i) {
                w[i].va   = (void *) ((uintptr_t) rr[i0 + i].first & ~(uintptr_t) 4095);
                w[i].attr = 0;
            }
            const bool ok = qws(GetCurrentProcess(), w.data(), (DWORD) (n*sizeof(fn_l22_ws_ex))) != 0;
            for (size_t i = 0; i < n; ++i) {
                if (ok && (w[i].attr & 1)) {
                    st.resident++;
                } else {
                    out.push_back(rr[i0 + i]);
                }
            }
        }
    } else {
        out = rr;
    }
    // the RAM budget: reading more than the available RAM above a 4 GiB floor only pushes other pages out (the decode's
    // CPU experts); the unlend reads what is left itself, as without the switch
    MEMORYSTATUSEX ms;
    memset(&ms, 0, sizeof(ms));
    ms.dwLength = sizeof(ms);
    size_t budget = SIZE_MAX;
    if (GlobalMemoryStatusEx(&ms)) {
        st.avail_b = (size_t) ms.ullAvailPhys;
        const size_t floor_b = (size_t) 4 << 30;
        budget = st.avail_b > floor_b ? st.avail_b - floor_b : 0;
    }
    std::vector<WIN32_MEMORY_RANGE_ENTRY> e;
    e.reserve(64);
    size_t i = 0;
    for (; i < out.size(); ++i) {
        if (st.asked_b + out[i].second > budget) {
            break;
        }
        if (stop.load(std::memory_order_relaxed)) {
            st.stopped = true;
            break;
        }
        e.push_back({ (PVOID) out[i].first, (SIZE_T) out[i].second });
        st.asked_b += out[i].second;
        if (e.size() == 64) {
            pvm(GetCurrentProcess(), (ULONG_PTR) e.size(), e.data(), 0);
            e.clear();
        }
    }
    if (!e.empty() && !stop.load(std::memory_order_relaxed)) {
        pvm(GetCurrentProcess(), (ULONG_PTR) e.size(), e.data(), 0);
    }
    for (; i < out.size(); ++i) {
        st.cut_b += out[i].second;
    }
    st.stopped = st.stopped || stop.load(std::memory_order_relaxed);
    st.ms      = (ggml_time_us() - t0)/1000.0;
}
#else
static void fn_l22_mem_prio_low() {
}
static void fn_l22_refill_prefetch_run(const std::vector<std::pair<const void *, size_t>> & rr, const std::atomic<bool> & stop,
                                       fn_l22_refill_pf_stats & st) {
    (void) stop;
    st.ranges = rr.size();
    for (const auto & r : rr) {
        st.asked_b += r.second;
    }
    llama_fn_prefetch_ranges(rr);
}
#endif
