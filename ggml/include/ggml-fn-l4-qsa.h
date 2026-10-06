#pragma once

// [TAG_FN_L4_QSA] Flash-Next (qwen4exp) lever round 4: the 12 QSA (full attention) layers.
//
// The qwen4exp graph builder writes the bits below into op_params slot GGML_FN_L4_QSA_SLOT of a QSA layer's
// LIGHTNING_INDEXER node, each one only under its LLAMA_FN_L4_QSA_<NAME> switch. No other graph writes the slot, and
// every backend but CUDA ignores it, so no other model reaches these paths. The CUDA backend sets the *DEP bits itself
// (graph_optimize) once it added the allocation dependencies a path needs; a path runs only with its DEP bit.

#include "ggml.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// int32 op_params slot 14 (bytes 56..59): free in LIGHTNING_INDEXER (ggml-fn-l3.h uses slot 15)
#define GGML_FN_L4_QSA_SLOT  14
#define GGML_FN_L4_QSA_MAGIC 0x4C340000u

enum ggml_fn_l4_qsa_bit {
    GGML_FN_L4_QSA_STREAMS = 0x0001, // the layer's indexer chain and its q / k / v chain run on two CUDA streams
    GGML_FN_L4_QSA_SEL     = 0x0002, // the top-k merge also writes the picked scores, the selected cells and the tail
    GGML_FN_L4_QSA_KVW     = 0x0004, // the layer's turbot K and V rows in one launch
    GGML_FN_L4_QSA_POOL    = 0x0008, // the layer's k-pool update (raw key rows, new pooled keys) in one launch (on its FILL)
    GGML_FN_L4_QSA_IDXDEP  = 0x0100, // backend: the IDXQ8 gather's indices stay allocated until the indexer ran
    GGML_FN_L4_QSA_SELDEP  = 0x0200, // backend: the SEL nodes' inputs and outputs stay allocated until the last one ran
};

static inline int32_t ggml_fn_l4_qsa_get(const struct ggml_tensor * t) {
    const uint32_t v = t ? (uint32_t) t->op_params[GGML_FN_L4_QSA_SLOT] : 0u;
    return (v & 0xFFFF0000u) == GGML_FN_L4_QSA_MAGIC ? (int32_t) (v & 0xFFFFu) : 0;
}

static inline void ggml_fn_l4_qsa_add(struct ggml_tensor * t, int32_t bits) {
    t->op_params[GGML_FN_L4_QSA_SLOT] = (int32_t) (GGML_FN_L4_QSA_MAGIC | (uint32_t) ggml_fn_l4_qsa_get(t) | ((uint32_t) bits & 0xFFFFu));
}

// [TAG_FN_L4_QSA_FASPLIT] on a QSA layer's FLASH_ATTN_EXT (slot 14 is free there too): live cells per stream-k block of
// the sparse turbot attention, in units of 16 (bits 0..11); 0 = the default layout
static inline int32_t ggml_fn_l4_qsa_fa_cells(const struct ggml_tensor * t) {
    return t->op == GGML_OP_FLASH_ATTN_EXT ? (ggml_fn_l4_qsa_get(t) & 0xFFF) * 16 : 0;
}

static inline void ggml_fn_l4_qsa_set_fa_cells(struct ggml_tensor * t, int32_t cells) {
    const int32_t u = cells <= 0 ? 0 : (cells + 15) / 16 > 0xFFF ? 0xFFF : (cells + 15) / 16;
    t->op_params[GGML_FN_L4_QSA_SLOT] = (int32_t) (GGML_FN_L4_QSA_MAGIC | (uint32_t) u);
}

#ifdef __cplusplus
}
#endif
