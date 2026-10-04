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
bool llama_moe_hot_init(const llama_model & model, const void * owner);

// [TAG_FN_MOE_HOT_ADAPT] LLAMA_MOE_HOT_ADAPT=1: windowed-frequency admission into the hot slots, evict first, publish
// after the upload has landed (LLAMA_MOE_HOT_ADMIT=N/W default 3/16, LLAMA_MOE_HOT_HYST=1, LLAMA_MOE_HOT_ADAPT_MIB=64 per
// step, LLAMA_MOE_HOT_VERIFY=<steps> compares one resident slot with its source). The owner context must synchronize
// its compute before llama_moe_cache_step; this returns that context, or nullptr when no adaptive set exists.
const void * llama_moe_hot_adapt_owner();

// max graph width for the hot chain, 0 when hot mode is off
int llama_moe_hot_max_t();
