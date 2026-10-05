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
bool llama_moe_hot_init(const llama_model & model, const void * owner, size_t budget_bytes = 0);

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
