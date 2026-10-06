#pragma once

// [TAG_FN_L4_GPUMOE] Flash-Next (qwen4exp) lever round 4, GPU chain B (router -> top-k -> bridge post).
//
// The qwen4exp graph builder writes these marks into op_params slot GGML_FN_L3_SLOT, and only when its LLAMA_FN_L4_*
// switch is on. No other model writes them, so no other graph reaches these paths. Every other backend ignores the
// slot. Every marked path gives the same bits as the unmarked one (the host job reads the same bytes).

#include "ggml-fn-l3.h"

#ifdef __cplusplus
extern "C" {
#endif

enum ggml_fn_l4_gpumoe_mark {
    // MOE_HOST_POST (a job, not a hint): one system fence per post instead of four. When the fused top-k nodes of the
    // router come right before it, the top-k and the post run in one launch (x is copied while the top-k runs).
    GGML_FN_L4_POST = 0x4C344D01,
};

#ifdef __cplusplus
}
#endif
