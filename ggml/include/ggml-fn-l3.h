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

// int32 op_params slot 15 (bytes 60..63): free in SCALE, UNARY, MUL, MUL_MAT, TOP_K, LIGHTNING_INDEXER, FLASH_ATTN_EXT
// (turbot uses bytes 24..47), DSV4_HC_PRE, DSV4_HC_POST, GET_ROWS and RESHAPE (only ROPE uses slot 15)
#define GGML_FN_L3_SLOT 15

enum ggml_fn_l3_mark {
    GGML_FN_L3_NONE    = 0,
    GGML_FN_L3_HCW     = 0x4C334701, // SCALE -> SIGMOID -> SCALE [-> DSV4_HC_POST]: the hc combine weights 2*sigmoid(x/hc)
    GGML_FN_L3_HCLO    = 0x4C334702, // SCALE -> SILU: the low-rank activation of an hc mixer
    GGML_FN_L3_COMPACT = 0x4C334703, // FLASH_ATTN_EXT: the sparse-index compaction runs on many blocks
    GGML_FN_L3_TOPK    = 0x4C334704, // TOP_K (unordered): chunked two-stage select for a large k
    GGML_FN_L3_IDXQ8   = 0x4C334705, // LIGHTNING_INDEXER: reads its keys through the GET_ROWS that feeds it; on that
                                     // GET_ROWS: not computed when only marked indexers read it
    GGML_FN_L3_MMV     = 0x4C334706, // MUL_MAT with few rows and a long row: the mat-vec loads run ahead (same sums)
    GGML_FN_L3_Q8OUT   = 0x4C334707, // MUL of an RMS_NORM, SILU of an HCLO chain, DSV4_HC_PRE: also writes the q8_1 copy
                                     // of its output for the MUL_MAT that reads it (the MMVQ reuse cache, same bytes)
    GGML_FN_L3_Q8IN    = 0x4C334708, // the src1 of MUL_MAT_IDs: they read a q8_1 copy of it from the reuse cache if it has one
    GGML_FN_L3_ZSKIP   = 0x4C334709, // the ids of a MUL_MAT_ID: matrix ne02 - 1 of src0 is all zeros (the zero slot of the
                                     // hot set / a DMA bank), so its outputs are 0 and are written without reading it
    GGML_FN_L3_GDNAB   = 0x4C33470A, // the GDN beta and alpha MUL_MATs: beta -> sigmoid, then alpha -> (+ dt, softplus,
                                     // * a), next to each other in the graph, in one launch

    // [TAG_FN_L4_HC] lever round 4 (LLAMA_FN_L4_HC): a whole hc mixer of a qwen4exp layer in three launches (norm with
    // its q8_1 copy, down + inject, up + hc_pre), and the combine before it folded into the norm launch
    GGML_FN_L4_HC      = 0x4C344801, // MUL (rms_norm * gamma) that starts an hc mixer: RMS_NORM -> MUL -> MUL_MAT down
                                     // [-> MUL_MAT inject] -> SCALE -> SILU -> MUL_MAT up -> DSV4_HC_PRE
    GGML_FN_L4_HCPOST  = 0x4C344802, // SCALE of the combine weights: SCALE -> SIGMOID -> SCALE -> DSV4_HC_POST, then a
                                     // GGML_FN_L4_HC mixer that reads the post
    GGML_FN_L4_HCQ8    = 0x4C344803, // DSV4_HC_PRE of such a mixer: also writes the q8_1 copy of its output for the MUL_MAT
                                     // that reads it (the MMVQ reuse cache, same bytes)
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
