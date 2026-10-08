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
// the up / gate / down slot tensors. While a prompt streams nothing is evicted (no decode step runs), so a slot read here
// keeps its expert until the stream gives the lend back. false: no such layer, the layer is out (lent, tail, KV), the
// bookkeeping is busy, or no expert is held
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
size_t llama_moe_hot_cbuf_release(const void * owner);

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
size_t   llama_moe_hot_kv_bytes();
size_t   llama_moe_hot_kv_out();
size_t   llama_moe_hot_kv_release(const void * owner, size_t need);
size_t   llama_moe_hot_kv_restore(const void * owner, size_t keep);
uint64_t llama_moe_hot_generation();

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

// [TAG_FN_L4_EXIT] the owner goes away: the upload worker stops after its chunk in flight and is joined, the upload stream
// is synchronized; the hot set stays static from then on
void llama_moe_hot_stop(const void * owner);
