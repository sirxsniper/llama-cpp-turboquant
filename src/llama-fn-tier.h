#pragma once

// [TAG_FN_L14_TIER] qwen4exp (profile switch LLAMA_FN_L14_TIER): the routed experts of the host layers stay in the mapped
// model file, at the same addresses, but their pages are locked in the working set, whole layers while the RAM above a
// reserve allows (LLAMA_FN_L14_TIER_RESERVE_MIB, default 12288). Nothing is copied and nothing is charged to commit, and
// the load no longer prefetches those ranges (the whole set does not fit in RAM: the prefetch only churned the cache).
// When Windows signals low memory, the last locked layers are unlocked again. A no-op on other architectures/platforms.

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
