#pragma once

// GPU-resident LRU cache for MoE expert weights that -ot pinned to host memory.
//
// Motivation (measured on Qwen3.8-Flash-Next, 512 experts / 10 routed): expert
// routing has strong temporal locality (LRU-64 hit rate ~67% over a mixed
// workload) even though the long-run distribution is near-uniform. Decode on a
// host-offloaded MoE layer is bound by host RAM bandwidth, so serving the hot
// experts from VRAM removes most of the per-token DIMM traffic.
//
// Mechanism (no custom kernels):
//  - per cached layer, companion tensors up_c/gate_c/down_c of shape
//    [ne0, ne1, n_slots+1] live in the device buffer of that layer's router;
//    slot n_slots is permanently zero (the "dummy" slot).
//  - an I32 table[512] maps expert id -> slot, or n_slots when uncached.
//    One copy on device (read by get_rows to remap ids for the cache-side
//    mul_mat_id chain) and one on host (read by the CPU mul_mat_id via
//    src[3] to SKIP cached ids, zeroing their dst rows).
//  - the two down-projection outputs are summed; uncached ids contribute 0
//    through the cache chain (zero slot) and cached ids contribute 0 through
//    the CPU chain (skip), so the result is exact.
//  - llama_moe_cache_step(), called at the end of llama_context::decode(),
//    performs throttled LRU updates: at most LLAMA_MOE_CACHE_INSERTS expert
//    uploads per layer per step via ggml_backend_tensor_set.
//
// Enabled via llama_context_params.n_moe_cache_slots (CLI: --moe-expert-cache).

#include <cstddef>
#include <cstdint>
#include <vector>

struct llama_model;
struct ggml_tensor;

struct llama_moe_cache_layer {
    int il = -1;

    int32_t n_slots = 0;

    // host-resident source weights (the authoritative experts)
    ggml_tensor * up_src   = nullptr;
    ggml_tensor * gate_src = nullptr;
    ggml_tensor * down_src = nullptr;

    // device-resident cache slots, ne[2] == n_slots + 1 (last slot all zeros)
    ggml_tensor * up_c   = nullptr;
    ggml_tensor * gate_c = nullptr;
    ggml_tensor * down_c = nullptr;

    // expert id -> slot (or n_slots when uncached); I32 [1, n_expert]
    ggml_tensor * dev_table  = nullptr;
    ggml_tensor * host_table = nullptr;
};

// build the cache for every host-resident expert layer of the model.
// Safe to call more than once; only the first call does work.
void llama_moe_cache_init(const llama_model & model, int32_t n_slots, int32_t max_inserts);

// nullptr when the cache is disabled or this tensor has no cached layer
const llama_moe_cache_layer * llama_moe_cache_lookup(const ggml_tensor * up_exps);

// [TAG_FN_L3_VRAM_CBUF] as llama_moe_cache_lookup, also while the layer's slots are out (the compute-buffer lend): for a
// reader of the host table only (the bridge), which is always valid ("not hot" for every expert while the layer is out)
const llama_moe_cache_layer * llama_moe_cache_lookup_table(const ggml_tensor * up_exps);

// [TAG_FN_L14_TIER] for the RAM tier: out[e] = 1 when the hot set holds expert e of the layer of up_exps, 0 when the host
// computes it. A layer lent to a prompt or in the given-back tail keeps its residents (they come back); a layer that gave
// its VRAM to the KV cache has none. 1: done; 0: busy right now (try later); -1: no hot set or no such layer
int llama_moe_hot_residents(const ggml_tensor * up_exps, uint8_t * out, int64_t n_expert);

// [TAG_FN_L14_PFSD2D] for the prefill stream: slot[e] = the slot that holds expert e of the layer of up_exps, or -1; slots3:
// the up / gate / down slot tensors. Only slots whose upload has landed, and never a slot the KV cache can take back (a
// kv_tail layer, or a slot from sl_base on): a KV growth between two streamed ubatches unmaps those. No hot-set pass runs
// during a prompt (passes run on decode steps), so a slot read here keeps its expert while the stream copies it. false: no
// such layer, the layer is out (lent, tail, KV) or can go to the KV cache, the bookkeeping is busy, or no expert is held
bool llama_moe_hot_slots(const ggml_tensor * up_exps, int32_t * slot, int64_t n_expert, const ggml_tensor ** slots3);

// [TAG_FN_L15_TIER] the RAM tier's view of one layer: slot[e] = the hot slot of expert e or -1, cnt[e] = its decayed count
// (0 without the decayed policy), vol[e] = 1 when it is resident in a slot that can lose its VRAM without a choice (the
// tail given back for a big prompt, the prefill stream's lend, the KV cache's lend). 1: done; 0: busy (try later); -1: no
// hot set or no such layer
int llama_moe_hot_tier_view(const ggml_tensor * up_exps, int32_t * slot, float * cnt, uint8_t * vol, int64_t n_expert);

// [TAG_FN_L15_TIER] the RAM tier's gate on evictions: the adaptive set evicts a resident only when ready(up_exps, e) says
// its host copy is in RAM (locked), so the CPU never reads an evicted expert from the disk. nullptr: no gate
using llama_moe_host_ready_fn = bool (*)(const ggml_tensor * up_exps, int32_t e, void * ud);
void llama_moe_hot_set_host_ready(llama_moe_host_ready_fn fn, void * ud);

// apply throttled LRU updates; call between graph executions only. ctx: the calling llama_context
// ([TAG_FN_MOE_HOT_ADAPT] only the owner of an adaptive hot set updates it)
void llama_moe_cache_step(const void * ctx);

// [TAG_FN_L16_MTP_ASYNC] the LRU cache runs (no hot set): the step of every context changes slots that any graph reads
bool llama_moe_cache_lru_on();

// [TAG_FN_MOE_HOT] static per-expert hot set, chosen once from a routing profile (tools/moe-trace, LLAMA_MOE_PROFILE):
//   LLAMA_MOE_HOT_PROFILE=<file.moeprof>  enables it (the LRU cache above is then not built)
//   LLAMA_MOE_HOT_MIB=<n>|auto            VRAM budget; auto = free VRAM after the reserve minus LLAMA_MOE_HOT_HEADROOM_MIB (1536)
//   LLAMA_MOE_HOT_CAP_MIB=<n>             with auto: keep device use (all processes) at or below n MiB
//   LLAMA_MOE_HOT_MAX_T=<n>               graphs of up to n tokens (MTP verify, 2 streams) use it, 1..8, default 8
//   LLAMA_MOE_HOT_SECTION=<name>          profile section, default decode_union
//   LLAMA_MOE_HOT_COST=<type=f,...>       relative CPU cost per byte by type, default q5_1=1.3
//   LLAMA_MOE_HOT_STATS=1                 hit rate per layer every 256 decode steps
// The same companion-tensor + two-table mechanism as the LRU cache, with a variable slot count per layer; the tables
// are written once at init, and there is no worker and no step, so nothing races with a running graph (the CPU
// observer is set only for LLAMA_MOE_HOT_STATS, and only counts).
// Returns true when this call enabled it (the caller then reserves the scheduler again). Once per process.
// [TAG_FN_AUTO] every variable is read through llama_fn_env(), so the qwen4exp profile (src/llama-fn-auto.h) supplies
// the ones the environment does not set. LLAMA_MOE_HOT_PROFILE=even: no routing profile; every host layer gets the same
// number of slots, empty at start, and the adaptive set (required) fills them. =off / 0 / none: no hot set.
// [TAG_FN_VRAM_FIT] budget_bytes > 0: the VRAM fit's budget (LLAMA_MOE_HOT_MIB is then not read).
// [TAG_FN_L3_VRAM_CBUF] tail_bytes > 0: the device buffer lives on CUDA virtual memory, and its top layers (whole layers, at
// least tail_bytes, starting at a granule boundary) form a tail that gives its VRAM back while a big prompt runs
// (llama_moe_hot_cbuf_release / _restore). Needs one device with llama_moe_hot_vmm_granularity() > 0 and at least one
// layer below the tail; otherwise nothing is allocated and false comes back.
bool llama_moe_hot_init(const llama_model & model, const void * owner, size_t budget_bytes = 0, size_t tail_bytes = 0);

// [TAG_FN_L3_VRAM_CBUF] the virtual-memory granularity of the hot set's device (0: no such buffers there)
size_t llama_moe_hot_vmm_granularity(const llama_model & model);

// [TAG_FN_L3_VRAM_CBUF] the tail's bytes (0: no tail), and whether it is out (its VRAM given back) right now
size_t llama_moe_hot_cbuf_tail_bytes();
bool   llama_moe_hot_cbuf_out();

// [TAG_FN_L3_VRAM_CBUF] the owner's graphs are synchronized: the tail layers leave both tables and the hot chain
// (llama_moe_cache_lookup returns nullptr for them while they are out), uploads into them stop, then the tail's VRAM goes
// back to the driver. Returns the bytes given back (0: no tail, already out, or another owner).
// [TAG_FN_L20_BIGPROMPT_TAIL] need > 0: only the tail's top part (whole mapped segments, at least need bytes; the layers in it
// or within a granule below it are lent); called again with more (need 0 = the whole tail) it gives back the rest. Returns
// the bytes given back by this call.
size_t llama_moe_hot_cbuf_release(const void * owner, size_t need = 0);

// [TAG_FN_L3_VRAM_CBUF] map the tail again, clear it, upload the tail layers' resident experts and restore their tables.
// false: the mapping failed, the tail layers stay out (correct, fewer hot slots) and the next call tries again.
bool   llama_moe_hot_cbuf_restore(const void * owner);

// [TAG_FN_L8_KVLEND] the KV cache lends the hot set the VRAM of its cells that no context needs yet. Before the hot set's
// init: want = the bytes the KV cache may take back as its context grows, keep = the bytes to leave between the KV layers
// and the tail (the prefill stream's lend range), n_trunk = the trunk's layer count (the MTP block's layers never lend).
// The hot set then puts whole trunk layers of at least `want` bytes into regions of their own near the bottom of its
// buffer. _kv_release (the owner's graphs synchronized): those layers leave both tables and the hot chain from the top
// until `need` bytes are out, then their VRAM goes back to the driver; returns the bytes out. _kv_restore: maps them
// again (lowest first) while at least `keep` bytes stay out, clears and refills them; returns the bytes out.
// _generation changes with every release / restore: a graph built before must not be reused.
void     llama_moe_hot_set_kv_lend(size_t want, size_t keep, int n_trunk);
// [TAG_R40_LEND_PARTIAL] before the init (with _set_kv_lend): a slot lend whose segments still fall short at the last retry
// keeps what they hold (llama_moe_hot_kv_bytes() < want) instead of failing; false (default): it fails, as v0.30.0
void     llama_moe_hot_set_kv_partial(bool on);
// [TAG_R40_LEND_SPREAD] before the init (with _set_kv_lend): every layer's count from the whole budget, the lend out of the
// lend layers' own slots (more lend layers, each as big as the others); false (default): base cut + lend on top, as v0.30.0
void     llama_moe_hot_set_kv_spread(bool on);
// [TAG_FN_L21_BIGUB] with the slot lend: the next init also puts `bytes` of the lend layers' base slots over segments of
// their own (the slots per layer unchanged); _cb_borrow_bytes: what it placed (0: none); _cb_borrow (the owner's graphs
// synchronized, its switch to FULL): those segments go out for a wide compute reserve, returns the bytes out (0: none);
// _cb_unborrow: they come back, cleared, their residents queued for upload again. The KV cache's groups are separate.
void     llama_moe_hot_set_cb_borrow(size_t bytes);
size_t   llama_moe_hot_cb_borrow_bytes();
size_t   llama_moe_hot_cb_borrow(const void * owner);
void     llama_moe_hot_cb_unborrow(const void * owner);
size_t   llama_moe_hot_kv_bytes();

// [TAG_FN_L21_VRAM_DXGI] ggml_backend_dev_memory, plus (LLAMA_FN_L21_VRAM_DXGI) the DXGI budget logged or used (llama-moecache.cpp)
void     fn_l21_dev_memory(const llama_model * model, struct ggml_backend_device * dev, size_t * mem_free, size_t * mem_total);
size_t   llama_moe_hot_kv_out();
size_t   llama_moe_hot_kv_release(const void * owner, size_t need);
size_t   llama_moe_hot_kv_restore(const void * owner, size_t keep);
uint64_t llama_moe_hot_generation();
// R-REV-12: no graph built before this call is reused (a graph-shaping state outside the hot set changed, e.g. P14 off)
void     llama_moe_hot_generation_bump();

// [TAG_FN_L17_VISION_LEND] the parked projector's VRAM rides on the KV cache's lend: the fit adds its bytes to `want`, and
// _vis_set(bytes) after an init that holds them all. _vis_take (the owner's graphs synchronized): `bytes` more go out
// the same way as for the KV cache (top first); returns the bytes given back. _vis_give: maps them again (empty slots).
void   llama_moe_hot_vis_set(size_t bytes);
// [TAG_R34_F_VISFLAT] before the init (with _set_kv_lend, whose want includes them): the projector's bytes of the lend. With the
// slot lend they come out of the lend layers' own slots (more of them can go), so the counts per layer are those of the hot
// set without the projector's share; 0 (default, or LLAMA_R34_VISFLAT=0 at the caller): on top of a base cut for them (v0.26.6)
void   llama_moe_hot_set_vis_part(size_t bytes);
size_t llama_moe_hot_vis_bytes();
size_t llama_moe_hot_vis_take(size_t bytes);
void   llama_moe_hot_vis_give();

// [TAG_R35_NP2PARK] the idle slot's VRAM (parked recurrent cells, the wide SMALL reserve of a multi-slot context) rides on
// the same lend: the fit adds it to `want`. _np2_hold (the owner's graphs synchronized): the share holds at least `target`
// bytes out (whole groups from the bottom, kept apart from the KV cache's and the projector's), groups it no longer needs
// come back (cleared, their slots fill from the routing); returns what it holds. The KV cache's hooks see the lend
// without this share (_kv_release_kvside / _kv_restore_kvside add it to their absolute need).
size_t llama_moe_hot_np2_hold(const void * owner, size_t target);
size_t llama_moe_hot_np2_out();
// [TAG_R38_E3] true when _np2_hold(target) would move nothing: the share holds at least `target` and none of its groups can
// go back without dropping below it (a whole group is coarser than the target: 2 x 261 MiB held for a 288 MiB cell)
bool   llama_moe_hot_np2_settled(size_t target);
size_t llama_moe_hot_kv_release_kvside(const void * owner, size_t need);
size_t llama_moe_hot_kv_restore_kvside(const void * owner, size_t keep);

// [TAG_FN_AUTO] a hot set is configured for this model (LLAMA_MOE_HOT_PROFILE names a file or "even")
bool llama_moe_hot_wanted(const llama_model & model);

// [TAG_FN_VRAM_FIT] the budget comes from the VRAM fit (LLAMA_MOE_HOT_MIB=auto and LLAMA_MOE_HOT_FIT=1): the context
// defers llama_moe_hot_init until every context of the model exists, then sizes it from what is left
bool llama_moe_hot_fit_wanted(const llama_model & model);

// [TAG_FN_VRAM_FIT] the device of the hot set (the router device of the first host-expert layer), or nullptr
struct ggml_backend_device;
struct ggml_backend_device * llama_moe_hot_device(const llama_model & model);

// [TAG_FN_VRAM_FIT] device bytes the hot set holds, 0 without one
size_t llama_moe_hot_device_bytes();

// [TAG_FN_L3_POLICY_POOL] LLAMA_MOE_HOT_POOL=1 (qwen4exp, even slots, the decayed adaptive set): the layers of one expert
// shape class (same types and shapes) share one slot pool, k layers x (n + 1) - 1 slots and one zero slot in the bytes of
// their k fixed shares, and the decayed policy pairs the candidates and residents of all of them: a layer with a wide
// working set takes the slots a narrow one does not use. Each layer keeps its own tables; the pool's tensors are its
// up_c / gate_c / down_c and the pool's size its miss value, so the graph, the bridge and the CPU skip tables are unchanged.
// [TAG_FN_L3_POLICY_UPLOAD] LLAMA_MOE_HOT_UP_PIPE=1 (qwen4exp, adaptive set): the upload worker stages through two small
// pinned halves (LLAMA_MOE_HOT_UP_STAGE_MIB each, default 8) - one is filled while the other uploads, so the staging stays
// in the CPU caches - and LLAMA_MOE_HOT_UP_MIB_STEP=<n> (0 = no limit) caps its bytes per decode step (a token bucket that
// the owner's step refills, at most two steps' worth), so a large refill spreads over steps instead of taking DRAM and
// PCIe from one; a pass then queues at most what the bucket gives until the next pass. No host CUDA call waits on the
// compute stream; the worker waits only for its own copies.
// [TAG_FN_L3_POLICY_BURST] LLAMA_MOE_HOT_BURST_MIB=<n> (qwen4exp, decayed adaptive set; 0 = off): past the pass budget
// (LLAMA_MOE_HOT_DECAY_MIB) a pass may queue up to n MiB more for strong pairs only - a free slot, or a candidate whose count
// is over LLAMA_MOE_HOT_BURST_RATIO (default 2) x its victim's - so a shifted working set (a new request) refills fast while
// the steady state keeps the small budget. LLAMA_MOE_HOT_STATS adds the step's host time to the adaptive stats line.
// [TAG_FN_MOE_HOT_ADAPT] LLAMA_MOE_HOT_ADAPT=1: windowed-frequency admission into the hot slots, evict first, publish
// after the upload has landed (LLAMA_MOE_HOT_ADMIT=N/W default 3/16, LLAMA_MOE_HOT_HYST=1, LLAMA_MOE_HOT_ADAPT_MIB=64 per
// step, LLAMA_MOE_HOT_VERIFY=<steps> compares one resident slot with its source). The owner context must synchronize
// its compute before llama_moe_cache_step; this returns that context, or nullptr when no adaptive set exists.
const void * llama_moe_hot_adapt_owner();

// [TAG_FN_L3_HOST_STEP] true when the hot set holds a layer with index >= il_min (with il_min = n_layer(): an MTP
// block's host experts, which the draft context's graphs read, so its owner's step may not run beside them)
bool llama_moe_hot_has_layer_from(int il_min);

// [TAG_FN_R4_ADAPT_DECAY] LLAMA_MOE_HOT_DECAY=<0..1> (e.g. 0.92): the decayed-count policy instead of the window
// (llama-moe-decay.h: every LLAMA_MOE_HOT_DECAY_EVERY=2 steps one pass, admit at LLAMA_MOE_HOT_DECAY_ADMIT=2 and over
// LLAMA_MOE_HOT_DECAY_RATIO=1.2 x / LLAMA_MOE_HOT_DECAY_HYST=0.5 + the victim, LLAMA_MOE_HOT_DECAY_MIB per pass, by default
// LLAMA_MOE_HOT_ADAPT_MIB x every: the window's bytes per step), with the
// prompt's routing x LLAMA_MOE_HOT_SEED=0.03 folded in (0 = off). LLAMA_MOE_HOT_SAVE=<file>: the learned set as a moeprof
// v1 profile every LLAMA_MOE_HOT_SAVE_EVERY=1024 decode steps and at llama_moe_hot_save_now (the owner's destructor).
void llama_moe_hot_save_now(const void * owner);

// max graph width for the hot chain, 0 when hot mode is off
int llama_moe_hot_max_t();

// [TAG_FN_L4_MEM_PROMPT] the widest graph that builds the hot chain: llama_moe_hot_max_t, or 31 with LLAMA_FN_L4_PROMPT_HOT=1
// (qwen4exp: the short prompt path's ubatches); 0 when hot mode is off
int llama_moe_hot_graph_max_t();

// [TAG_FN_L19_TTFT_PHOT] LLAMA_FN_L19_TTFT_PHOT: the widest CPU-path prompt graph (wider than llama_moe_hot_graph_max_t, no
// prefill stream, no op offload) that builds the hot chain in its one-row-per-routed-expert form; 0 when off
int llama_moe_hot_prompt_max_t();

// [TAG_FN_R1_PFS_LEND] the prefill stream borrows the top `bytes` of the hot set's device buffer as its VRAM banks while
// a prompt streams (LLAMA_PREFILL_STREAM_LEND=1), so the banks cost the decode no hot slots. The caller must have
// synchronized every graph of the owner context: the hot layers whose slots lie in the lent range leave both tables
// (the CPU and the zero slot serve them), uploads into them stop, and their slots and zero slots may be overwritten.
// Returns false (and lends nothing) without a single-device hot set with one table tensor, or when `bytes` does not fit
// above the tables. *buf / *base: the device buffer and the first lent byte (the same every time).
struct ggml_backend_buffer;
bool llama_moe_hot_lend(const void * owner, size_t bytes, struct ggml_backend_buffer ** buf, uint8_t ** base);

// [TAG_FN_R1_PFS_LEND] the lent range is back (the device is idle, nothing writes it any more): the resident experts of
// the lent layers are uploaded again, their zero slots cleared, and both tables restored. No-op when nothing is lent.
void llama_moe_hot_unlend(const void * owner);

// [TAG_FN_L22_ENDOVERLAP] as llama_moe_hot_unlend, but the residents whose slices reach into the range go through the upload
// worker (they leave the tables now and come back at an owner step once their copy landed; the CPU computes them until
// then). Without the adaptive set it is llama_moe_hot_unlend.
void llama_moe_hot_unlend_async(const void * owner);

// [TAG_FN_L22_KEEPLEND] while the stream keeps its banks over a narrow ubatch: the lent layers are hidden from the hot
// chains (llama_moe_cache_lookup returns nullptr for them: their zero slot may hold bank bytes). A change bumps the hot
// generation (llama_moe_hot_generation), so graphs built under the other state are not reused. The owner's graphs are
// synchronized.
void llama_moe_hot_lend_hide(const void * owner, bool hide);

// [TAG_FN_L22_REFILL] (b) at a prompt's last batch: a background PrefetchVirtualMemory of the host pages the unlend and the
// tail restore will read (the resident experts of the lent and tail-out layers). Returns the bytes asked for (0: nothing,
// off, or the last prefetch still runs). Needs LLAMA_FN_L22_REFILL.
size_t llama_moe_hot_prefetch_refill(const void * owner);

// [TAG_FN_L22_LENDCLASS] the tail release of a narrow FULL class: the layers a release of need_hide would lend leave the
// tables and the hot chains (so the graphs and outputs are those of that release), but only the top need_unmap bytes go
// back; the restore then refills only the slots that reach into what was unmapped (and dirty ones). need_unmap 0 or >=
// need_hide: llama_moe_hot_cbuf_release(owner, need_hide). Returns the bytes given back.
size_t llama_moe_hot_cbuf_release2(const void * owner, size_t need_hide, size_t need_unmap);

// [TAG_FN_L22_ENDOVERLAP] llama_moe_hot_cbuf_restore with the refill on the upload worker when async (adaptive set only)
bool   llama_moe_hot_cbuf_restore_ex(const void * owner, bool async);

// [TAG_FN_L22_PASSLOG] the switch parts since the last call (taken and reset)
struct llama_moe_hot_l22_stats {
    double   unlend_ms  = 0.0; // the stream's returns (host time)
    size_t   unlend_b   = 0;   // refilled (sync) or queued (async)
    double   restore_ms = 0.0; // the tail's restores
    size_t   restore_b  = 0;
    uint64_t pinned_b   = 0;   // refill / upload bytes straight from pinned pages
    size_t   released_b = 0;   // tail bytes unmapped
    size_t   ul_left    = 0;   // async unlend jobs not landed yet
};
llama_moe_hot_l22_stats llama_moe_hot_l22_take();

// [TAG_FN_L22_BIGUB2] the borrow wanted next comes from the big FULL reserve for one prompt sequence (the logs say so)
void llama_moe_hot_l22_borrow_1seq(bool on);

// [TAG_FN_L22_UBPLAN] P10 R2 (for P5's residency classes): the borrow slots [*lo, *hi) of the layer of up_exps (the 16K FULL
// reserve's borrow, out only while such a period runs); false: none (no borrow, another layer, no hot set)
bool llama_moe_hot_borrow_range(const ggml_tensor * up_exps, int32_t * lo, int32_t * hi);

// [TAG_FN_L3_POLICY_SEED] LLAMA_MOE_HOT_SEED_NODE=1 (qwen4exp, decayed adaptive set, LLAMA_MOE_HOT_SEED > 0): a prompt
// ubatch (more than LLAMA_MOE_HOT_MAX_T tokens) reports its routing through a CPU graph node after the layer's experts,
// wherever they ran. Without it the prompt seed sees only prompts the CPU computes: the CPU MUL_MAT_ID observer never runs
// for ubatches that the prefill stream or the scheduler's op offload put on the GPU (32 tokens or more). The observer then
// counts decode steps only, so nothing is counted twice. LLAMA_MOE_HOT_SEED_NORM=<steps> (default 0 = the raw prompt counts x
// LLAMA_MOE_HOT_SEED): the whole prompt folds in once, at its first decode step, as <steps> decode steps x LLAMA_MOE_HOT_SEED
// of the share of its tokens that route to each expert, so a 100K prompt does not outweigh the decode for hundreds of steps.
struct ggml_context;
struct ggml_cgraph;
bool          llama_moe_hot_seed_node_wanted(const ggml_tensor * up_exps, int64_t n_tokens);
// ids: [n_expert_used, n_tokens] I32, contiguous. Returns the node (pin it to the CPU backend).
ggml_tensor * llama_moe_hot_build_seed(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * ids, int il);

// [TAG_FN_L3_POLICY_STATE] LLAMA_MOE_HOT_STATE=<file>|auto (qwen4exp, decayed adaptive set; auto = <model file>.hotstate):
// the learned set (decayed counts + residents, best first) is saved on a slot save, at the owner's end and every
// LLAMA_MOE_HOT_STATE_EVERY decode steps (0 = off), and loaded when the hot set starts: the even slots start filled with
// the saved residents instead of empty, so a new process does not start cold. LLAMA_MOE_HOT_STATE_LOAD=0 saves only.
void llama_moe_hot_state_save(const void * owner);

// [TAG_FN_L4_EXIT] the owner is idle: the upload worker lands its batch in flight (at most 64 MiB) and keeps the rest of
// its queue until the owner's next step, so nothing of the hot set runs on the device. Returns after that batch.
void llama_moe_hot_idle(const void * owner);

// [TAG_FN_L17_START_PRE] LLAMA_FN_L17_START_PRE=<MiB> (qwen4exp, decayed adaptive set): the owner calls turn_start after a
// prompt batch (no graph of it runs). If the batch's routing was counted (prompts below the stream's size), the decayed
// counts fade (x LLAMA_FN_L17_START_TD), the prompt's routing comes in as LLAMA_FN_L17_START_SN decode steps of its mix,
// and one pass of the decay rules swaps up to <MiB> of experts, landed before it returns. Returns true if it ran a pass.
bool llama_moe_hot_turn_start_on();
// [TAG_FN_L18_FIRSTANS_ONCE] a prompt's routing is waiting for the pass (batches that ran without one)
bool llama_moe_hot_turn_pending();
// [TAG_FN_L18_FIRSTANS_OVERLAP] the owner, before its next graph: a pass whose copies run in the background lands and its
// experts enter the tables (a no-op when none is in flight)
void llama_moe_hot_turn_finish(const void * owner);
bool llama_moe_hot_turn_start(const void * owner);

// [TAG_FN_L18_KEEPCTX_HOT] the decayed counts of every layer (each layer's n_expert values, none for a layer without
// counts): get copies them out, set puts them back, so the passes move the set toward a conversation that returns to the
// slot. false when the owner has no decayed adaptive set, or set got another layout.
bool llama_moe_hot_counts_get(const void * owner, std::vector<float> & out);
bool llama_moe_hot_counts_set(const void * owner, const std::vector<float> & in);

// [TAG_FN_L4_EXIT] the owner goes away: the upload worker stops after its chunk in flight and is joined, the upload stream
// is synchronized; the hot set stays static from then on
void llama_moe_hot_stop(const void * owner);
