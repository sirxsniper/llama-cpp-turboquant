#pragma once

// [TAG_FN_L3_GPU] Flash-Next (qwen4exp) device-time levers of lever round 3: the CUDA paths that the marks of
// ggml-fn-l3.h select. Nothing here runs for a graph without the marks. Every path gives the same bits as the
// unmarked one; GGML_CUDA_FN_L3=0 turns all of them off in one binary (the marks are then ignored).

#include "common.cuh"
#include "ggml-fn-l3.h"

// false when GGML_CUDA_FN_L3=0
bool ggml_cuda_fn_l3_enabled();

// one log line per path, the first time it runs in this process
void ggml_cuda_fn_l3_note(int path, const char * what);

enum ggml_cuda_fn_l3_path {
    GGML_CUDA_FN_L3_PATH_HCW3 = 0,
    GGML_CUDA_FN_L3_PATH_HCW4,
    GGML_CUDA_FN_L3_PATH_HCLO,
    GGML_CUDA_FN_L3_PATH_COMPACT,
    GGML_CUDA_FN_L3_PATH_TOPK,
    GGML_CUDA_FN_L3_PATH_IDXQ8,
    GGML_CUDA_FN_L3_PATH_IDXQ8_SKIP,
    GGML_CUDA_FN_L3_PATH_MMVF,
    GGML_CUDA_FN_L3_PATH_MMVQ,
    GGML_CUDA_FN_L3_PATH_MMVQ_SMK,
    GGML_CUDA_FN_L3_PATH_Q8F_NORM,
    GGML_CUDA_FN_L3_PATH_Q8F_HCLO,
    GGML_CUDA_FN_L3_PATH_Q8F_HCPRE,
    GGML_CUDA_FN_L3_PATH_Q8F_MMID,
    GGML_CUDA_FN_L3_PATH_COUNT,
};

// [TAG_FN_L3_GPU_Q8F] where a producer kernel writes the q8_1 copy of its output: the MMVQ reuse cache's buffer, laid
// out as quantize_row_q8_1_cuda lays out the src1 that reads it (nrows rows of bpr blocks, the last bpr - ne10/32 of them
// padding)
struct ggml_cuda_fn_l3_q8_dst {
    block_q8_1 * y     = nullptr;
    int64_t      ne10  = 0;
    int64_t      bpr   = 0;
    int64_t      nrows = 0;
};

// mmvq.cu: finds the MUL_MAT that reads `producer` next and points the reuse cache at its src1 (false: no copy). The
// caller must then write the whole copy: padding_ok = it also writes the zero blocks of rows padded to MATRIX_ROW_PADDING.
bool ggml_cuda_mmvq_q8_produce(ggml_backend_cuda_context & ctx, const ggml_tensor * producer, ggml_cuda_fn_l3_q8_dst & out,
        bool padding_ok);

// [TAG_FN_L3_GPU_Q8F] the producers: each computes its output as the unmarked op does and also its q8_1 copy, bit for bit
// what quantize_q8_1 makes of that output. false: not taken (nothing launched, the caller runs the op as usual).
//   rms_norm_mul: the fused RMS_NORM -> MUL (MUL marked Q8OUT)
//   hc_pre:       DSV4_HC_PRE (gated, marked Q8OUT)
bool ggml_cuda_fn_l3_rms_norm_mul_q8(ggml_backend_cuda_context & ctx, ggml_tensor * rms_norm, ggml_tensor * mul);
bool ggml_cuda_fn_l3_hc_pre_q8(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// [TAG_FN_L3_GPU_HCFUSE] the hc chains in one launch (ggml-cuda.cu matches the marked nodes):
//   hcw3: s1 = scale(x), sigmoid, s2 = scale -> writes s2
//   hcw4: the same, then dsv4_hc_post(x', residual, s2) without comb -> writes post (s1 .. s2 are not written)
//   hclo: s = scale(x), silu -> writes silu
void ggml_cuda_fn_l3_hcw3(ggml_backend_cuda_context & ctx, const ggml_tensor * s1, ggml_tensor * s2);
void ggml_cuda_fn_l3_hcw4(ggml_backend_cuda_context & ctx, const ggml_tensor * s1, const ggml_tensor * s2, ggml_tensor * post);
void ggml_cuda_fn_l3_hclo(ggml_backend_cuda_context & ctx, const ggml_tensor * s, ggml_tensor * silu);

// [TAG_FN_L3_GPU_MMV] a marked mat-vec with few rows and a long row, loads issued ahead of the sums: f32 x f32
// (mmvf.cu) and q8_0 x q8_1 (mmvq.cu). Same per-thread sums in the same order and the same reductions as
// mul_mat_vec_f / mul_mat_vec_q at the launch shape those pick, so the same bits. false: not taken (nothing launched).
bool ggml_cuda_fn_l3_mul_mat_vec_f(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
bool ggml_cuda_fn_l3_mul_mat_vec_q(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// [TAG_FN_L3_GPU_COMPACT] the sparse-index lists of flash_attn_mask_to_sparse_indices (fattn.cu), counted and written
// by many blocks. Same arguments, same lists in the same order.
void ggml_cuda_fn_l3_compact_mask(ggml_cuda_pool & pool, const ggml_tensor * mask, int32_t * indices, int32_t * counts,
        int32_t n_queries, int32_t ncols1, int32_t n_kv_max, cudaStream_t stream);

// [TAG_FN_L3_GPU_IDXQ8] the GET_ROWS node a marked lightning indexer reads its keys through (q8_0 rows by I32
// indices), or nullptr when it reads its key tensor as usual
const ggml_tensor * ggml_cuda_fn_l3_idxq8_gather(const ggml_tensor * indexer);

// true when node i is a GET_ROWS whose only readers are marked lightning indexers that read through it: the node is
// not computed
bool ggml_cuda_fn_l3_idxq8_skip(const ggml_cgraph * cgraph, int i);
