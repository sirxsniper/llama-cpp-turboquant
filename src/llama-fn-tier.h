#pragma once

// [TAG_FN_L14_TIER] [TAG_FN_L15_TIER] qwen4exp (profile switch LLAMA_FN_L14_TIER): VRAM holds the hot experts, RAM the rest.
// The routed experts of the host layers stay in the mapped model file, at the same addresses; a background thread locks
// their pages in the working set, expert by expert, in this order: the residents the hot set will evict next (the
// LLAMA_FN_L15_TIER_VICTIMS lowest counts per layer, default 16), the experts the CPU computes (most used first), and the
// residents that can lose their slot without a choice (the tail, the prefill stream's lend, the KV cache's lend). Room
// under the cap comes from the other residents, whose RAM copy is not needed while they stay in VRAM. After the first
// fill the hot set evicts only experts whose RAM copy is locked, so the CPU never reads an expert from the disk.
// From LLAMA_FN_L14_TIER_DELAY_MS after the load (default 20000), a round every 500 ms, LLAMA_FN_L14_TIER_THREADS (8)
// threads read the pages; up to LLAMA_FN_L14_TIER_MAX_MIB (default 72 % of the RAM, at least 22 GiB left) while
// LLAMA_FN_L14_TIER_FREE_MIB (6144) of RAM stays available. Below LLAMA_FN_L14_TIER_LOW_MIB (4096) for 1.5 s about 1 GiB
// is unlocked. Nothing is copied or charged to commit, and the load does not prefetch those ranges.
// A no-op on other architectures and other platforms.

#include <cstddef>
#include <memory>

struct llama_model;
struct llama_model_loader;

struct llama_fn_tier;
struct llama_fn_tier_deleter {
    void operator()(llama_fn_tier * t) const;
};
using llama_fn_tier_ptr = std::unique_ptr<llama_fn_tier, llama_fn_tier_deleter>;

// before the mappings exist: the expert ranges the load must not prefetch (ml.noprefetch)
void llama_fn_tier_plan(const llama_model & model, llama_model_loader & ml);

// after the tensor data is loaded (mapped); nullptr when off, not possible, or nothing fits
llama_fn_tier_ptr llama_fn_tier_build(llama_model & model, llama_model_loader & ml);

// [TAG_FN_L15_WARM] qwen4exp on Windows (LLAMA_FN_L15_WARM, default on): the load does not prefetch the routed experts; a
// background thread started by llama_fn_tier_build reads them into the working set, whose maximum it first raises. It
// starts when the trunk context's first decode returns (this call; a no-op after the first one)
void llama_fn_warm_go();

// [TAG_FN_L15_WARM] while the warm pass runs, a prompt's stream waits for each layer it has not reached: both faulting the same
// pages in at once took the first 32K prompt from 21 s to 55 s (l15 abx6_v5_i5)
void llama_fn_warm_wait_layer(int il);
bool llama_fn_warm_running();

// [TAG_FN_L15_WARMYIELD] a decode of small ubatches (answer steps, bridged) begins / ends: the warm pass waits meanwhile
void llama_fn_warm_small_decode(bool begin);
