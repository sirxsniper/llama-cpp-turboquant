#pragma once

// [TAG_FN_L14_ARENA] qwen4exp (profile switch LLAMA_FN_L14_ARENA): the routed experts of the host layers are read once,
// with unbuffered file reads, into the engine's own locked RAM, whole layers while the RAM above a reserve allows
// (LLAMA_FN_L14_ARENA_RESERVE_MIB, default 16384). Their tensors then point into it, so no later read of them goes
// through the Windows file cache: no page-in when other files or programs push the model's pages out. The layers that
// do not fit stay mapped. A no-op on other architectures and other platforms.

#include <cstddef>
#include <memory>

struct llama_model;
struct llama_model_loader;

struct llama_fn_arena;
struct llama_fn_arena_deleter {
    void operator()(llama_fn_arena * a) const;
};
using llama_fn_arena_ptr = std::unique_ptr<llama_fn_arena, llama_fn_arena_deleter>;

// after the tensor data is loaded (mapped); nullptr when off, not possible, or nothing fits
llama_fn_arena_ptr llama_fn_arena_build(llama_model & model, llama_model_loader & ml);

// [TAG_FN_L14_ARENA] [p, p + n) lies in an arena chunk that is registered with the GPU driver (LLAMA_FN_L14_ARENA_PIN,
// default on): a device copy may read it directly, without a staging copy
bool llama_fn_arena_pinned(const void * p, size_t n);
