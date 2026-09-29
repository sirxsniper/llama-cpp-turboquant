#pragma once

// PCIe Gen5 paths for host-resident MoE experts (Qwen3.8-Flash-Next with --n-cpu-moe / the hot set). All are off unless
// their switch is set; with every switch unset no graph changes and nothing is allocated.
//
// [TAG_MOE_DMA_SHARE] LLAMA_MOE_DMA_SHARE=<0..1>|auto   decode graphs of <= 8 tokens
//   A bounded pinned ring (LLAMA_MOE_DMA_RING_MIB, default 1024, all gen5 pinned memory <= 4096 MiB) holds copies of
//   often-missed experts. At each host MoE layer a CPU "plan" node (first node of the CPU split) picks D: routed experts
//   that are not hot and sit in the ring, at most share x the layer's cold experts and LLAMA_MOE_DMA_SLOTS (8). An
//   issuer thread copies D ring -> VRAM bank on its own stream while the CPU computes the rest; the CPU skips D through
//   the src[3] table; a GPU chain over the bank (zero slot for all other ids) runs after the CPU split and waits for the
//   copies with an event; the outputs are added. Every expert is computed by exactly one path: the CPU (the CPU math),
//   the hot chain or the bank chain (the GPU math), the same as the hot set.
//   share = auto: a fence node at the end of the CPU split waits for the copies; if they end after the CPU part the
//   share goes down, else up (both paths read DRAM, so the balance point is measured, not assumed).
//   Ring admission: seen >= N times in W steps as cold (LLAMA_MOE_DMA_ADMIT=2/32), LLAMA_MOE_DMA_FILL_MIB (32) per step,
//   warm start from LLAMA_MOE_DMA_PROFILE (default LLAMA_MOE_HOT_PROFILE): the best non-hot experts of the profile.
//   LLAMA_MOE_DMA_STATS=1 prints every 256 steps; LLAMA_MOE_DMA_SYNC=1 waits for pending fills (deterministic).
//
// [TAG_MOE_PREFETCH] LLAMA_MOE_PREFETCH=1   (with or without a share; share 0 = only prefetched experts go to the GPU)
//   The next layer's router on this layer's FFN input (the LLAMA_MOE_TRACE_PRED lookahead, qwen4exp) predicts layer
//   L+1's experts at layer L. At the fence of layer L (the CPU is done, the GPU runs attention) the issuer copies the
//   predicted ring-resident experts into layer L+1's bank (LLAMA_MOE_PREFETCH_SLOTS, 4), and the filler copies up to
//   LLAMA_MOE_PREFETCH_FILL (2) predicted experts that are not in the ring into it. At L+1 the prefetched experts that
//   were routed need no copy on the critical path.
//
// [TAG_FN_PREFILL_STREAM] LLAMA_PREFILL_STREAM=1   graphs of >= LLAMA_PREFILL_STREAM_MIN tokens (32)
//   The host experts of each layer stream into a VRAM bank (LLAMA_PREFILL_STREAM_BUFS, 2) through a pinned staging
//   ring (LLAMA_PREFILL_STREAM_RING_MIB, 512; LLAMA_PREFILL_STREAM_THREADS, 8, copy mmap -> pinned) on a copy stream,
//   and the MoE of that layer runs on the GPU from the bank. A gate node before each MoE waits for its bank and a
//   release node after it frees the bank, so the copy of the next layers overlaps the compute of this one. The GPU
//   math is the one the scheduler's op offload uses today for these ubatches (same kernels, same weights). A source in
//   a pinned buffer (--no-mmap) is copied without staging. LLAMA_PREFILL_STREAM_WRAP=1 (default) loads the first layer
//   for the next ubatch after the last one; LLAMA_PREFILL_STREAM_STATS=1 prints per ubatch.

#include "llama.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <vector>

struct llama_model;

// one host-resident MoE layer (tests build these directly)
struct llama_moe_gen5_layer_desc {
    int           il   = -1;
    ggml_tensor * up   = nullptr; // [n_embd, n_ff, n_expert] host
    ggml_tensor * gate = nullptr;
    ggml_tensor * down = nullptr; // [n_ff, n_embd, n_expert] host
};

// the device side the gen5 paths use; dev == nullptr or a CPU device runs everything on the host (CPU tests)
struct llama_moe_gen5_device {
    ggml_backend_dev_t         dev     = nullptr;
    ggml_backend_buffer_type_t buft    = nullptr; // bank memory
    ggml_backend_t             compute = nullptr; // the backend that runs the owner's graphs on dev
};

// graph-building views, valid while the owner lives
struct llama_moe_dma_view {
    ggml_tensor * up        = nullptr; // bank [ne0, ne1, n_slots + 1], slot n_slots is zero
    ggml_tensor * gate      = nullptr;
    ggml_tensor * down      = nullptr;
    ggml_tensor * dev_table = nullptr; // I32 [1, n_expert]: expert -> bank slot, n_slots when not in the bank
    void        * handle    = nullptr;
};

struct llama_pfs_view {
    ggml_tensor * up     = nullptr; // bank aliases with the layer's own types and shapes
    ggml_tensor * gate   = nullptr;
    ggml_tensor * down   = nullptr;
    void        * handle = nullptr;
};

// context lifetime (llama_context calls these; owner = the llama_context). init before the first scheduler reserve.
void llama_moe_gen5_init(const llama_model & model, const void * owner, const std::vector<ggml_backend_t> & backends);
void llama_moe_gen5_step(const void * owner); // end of the owner's decode()
void llama_moe_gen5_free(const void * owner); // owner destructor

// --- [TAG_MOE_DMA_SHARE] [TAG_MOE_PREFETCH] ---

constexpr int LLAMA_MOE_DMA_MAX_T = 8; // one zero slot per bank: MMVQ widths only

// explicit init (tests): returns true when enabled
LLAMA_API bool llama_moe_dma_init_layers(const std::vector<llama_moe_gen5_layer_desc> & layers, const llama_moe_gen5_device & d, const void * owner);
LLAMA_API void llama_moe_dma_free(const void * owner);
LLAMA_API void llama_moe_dma_step(const void * owner);

// sched: the graph's scheduler (nullptr = no owner check, tests); nullptr when this layer has no DMA path for this graph
LLAMA_API const llama_moe_dma_view * llama_moe_dma_lookup(ggml_backend_sched_t sched, const ggml_tensor * up_exps, int64_t n_tokens);

// top-k of the next-layer router prediction the prefetch needs (16), 0 when LLAMA_MOE_PREFETCH is off
LLAMA_API int llama_moe_dma_pred_k();

// CPU plan node: returns the I32 [1, n_expert] table for src[3] of the CPU MUL_MAT_IDs (0 = the CPU computes the
// expert, so op_params[0] stays 0). hot_table/hot_miss: the hot set's host table and its "not hot" value, or nullptr.
// pred_next: [k, T] I32 prediction for the next layer, or nullptr.
LLAMA_API ggml_tensor * llama_moe_dma_build_plan(ggml_context * ctx, const llama_moe_dma_view * v, ggml_tensor * ids,
        ggml_tensor * pred_next, const ggml_tensor * hot_table, int32_t hot_miss);

// CPU fence node after the CPU down projection (same split); expanded into gf here, returned for backend pinning
LLAMA_API ggml_tensor * llama_moe_dma_build_fence(ggml_context * ctx, ggml_cgraph * gf, const llama_moe_dma_view * v, ggml_tensor * cpu_down);

// counters for tests and stats
struct llama_moe_dma_counters {
    uint64_t steps        = 0;
    uint64_t layer_steps  = 0;
    uint64_t cold         = 0; // unique cold (non-hot) experts seen by plan nodes
    uint64_t dma_on_demand= 0; // experts copied ring -> bank at their own layer
    uint64_t dma_staged   = 0; // cold experts served from a prefetched bank slot
    uint64_t prefetched   = 0; // experts copied into a bank ahead of their layer
    uint64_t pred_cold    = 0; // cold experts that were in the prediction
    uint64_t ring_ready   = 0; // cold experts found ready in the ring
    uint64_t fills        = 0; // ring fills done
    uint64_t fill_urgent  = 0; // of which requested by a prediction
    double   share        = 0.0;
};
LLAMA_API llama_moe_dma_counters llama_moe_dma_get_counters();
// tests: wait until the filler has no queued or running job
LLAMA_API void llama_moe_dma_wait_idle();

// --- [TAG_FN_PREFILL_STREAM] ---

LLAMA_API bool llama_prefill_stream_init_layers(const std::vector<llama_moe_gen5_layer_desc> & layers, const llama_moe_gen5_device & d, const void * owner);
LLAMA_API void llama_prefill_stream_free(const void * owner);

LLAMA_API const llama_pfs_view * llama_prefill_stream_lookup(ggml_backend_sched_t sched, const ggml_tensor * up_exps, int64_t n_tokens);

// CPU gate node before the layer's MoE (waits for the bank, starts the next copies), expanded into gf here
LLAMA_API ggml_tensor * llama_prefill_stream_build_gate(ggml_context * ctx, ggml_cgraph * gf, const llama_pfs_view * v, ggml_tensor * cur);
// CPU release node after the layer's down projection (frees the bank), expanded into gf here
LLAMA_API ggml_tensor * llama_prefill_stream_build_release(ggml_context * ctx, ggml_cgraph * gf, const llama_pfs_view * v, ggml_tensor * down);

struct llama_pfs_counters {
    uint64_t ubatches = 0;
    uint64_t jobs     = 0; // layer copies done
    uint64_t bytes    = 0;
    uint64_t reused   = 0; // gates that found their layer already in the bank
    double   wait_s   = 0; // host time the gates waited for copies
};
LLAMA_API llama_pfs_counters llama_prefill_stream_get_counters();
