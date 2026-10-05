#pragma once

// [TAG_FN_L3_GPU] Flash-Next (qwen4exp) device-time levers of lever round 3.
//
// The qwen4exp graph builder marks the nodes that a CUDA fused or reworked path may take: it writes one of the marks
// below into op_params slot GGML_FN_L3_SLOT, and only when its LLAMA_FN_GPU_* switch is on. No other model writes the
// slot, so no other graph reaches these paths. Every other backend ignores the slot and computes the nodes as they are.
// Every marked path gives the same bits as the unmarked one.

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// int32 op_params slot 15 (bytes 60..63): free in SCALE, UNARY, MUL_MAT, TOP_K, LIGHTNING_INDEXER, FLASH_ATTN_EXT
// (turbot uses bytes 24..47) and DSV4_HC_POST
#define GGML_FN_L3_SLOT 15

enum ggml_fn_l3_mark {
    GGML_FN_L3_NONE    = 0,
    GGML_FN_L3_HCW     = 0x4C334701, // SCALE -> SIGMOID -> SCALE [-> DSV4_HC_POST]: the hc combine weights 2*sigmoid(x/hc)
    GGML_FN_L3_HCLO    = 0x4C334702, // SCALE -> SILU: the low-rank activation of an hc mixer
    GGML_FN_L3_COMPACT = 0x4C334703, // FLASH_ATTN_EXT: the sparse-index compaction runs on many blocks
    GGML_FN_L3_TOPK    = 0x4C334704, // TOP_K (unordered): chunked two-stage select for a large k
    GGML_FN_L3_IDXQ8   = 0x4C334705, // LIGHTNING_INDEXER: reads its keys through the GET_ROWS that feeds it
    GGML_FN_L3_MMV     = 0x4C334706, // MUL_MAT with few rows and a long row: the mat-vec loads run ahead (same sums)
};

static inline void ggml_fn_l3_set(struct ggml_tensor * t, int32_t mark) {
    t->op_params[GGML_FN_L3_SLOT] = mark;
}

static inline int32_t ggml_fn_l3_get(const struct ggml_tensor * t) {
    return t ? t->op_params[GGML_FN_L3_SLOT] : 0;
}

#ifdef __cplusplus
}
#endif
