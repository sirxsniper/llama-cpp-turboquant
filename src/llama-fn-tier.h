#pragma once

// [TAG_FN_L14_TIER] qwen4exp (profile switch LLAMA_FN_L14_TIER): the routed experts of the host layers stay in the mapped
// model file, at the same addresses, and a background thread locks their pages in the working set expert by expert: first
// the experts the CPU computes (not in the VRAM hot set), then the hot set's, from LLAMA_FN_L14_TIER_DELAY_MS after the load
// (default 20000: the context's warm-up and first kernel loads come first), up to LLAMA_FN_L14_TIER_MAX_MIB (default 60 % of
// the RAM: the GPU driver locks memory too) while LLAMA_FN_L14_TIER_FREE_MIB (6144) of RAM stays available.
// LLAMA_FN_L14_TIER_THREADS (4) threads read the pages in parallel. Every 2 s it follows the hot set: a CPU expert that is
// not locked is locked, in place of a hot one when the cap is in the way. Nothing is copied or charged to commit, and the
// load no longer prefetches those ranges. Below LLAMA_FN_L14_TIER_LOW_MIB (4096) of available RAM about 1 GiB is unlocked
// (the hot set's first); after a minute with room the hot set's experts are added again.
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
