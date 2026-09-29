#pragma once

// [TAG_FN_QSA_FUSED] GGML_OP_QSA_SCORE / GGML_OP_QSA_TOPK: the qwen4exp QSA indexer without the [n_kv, n_tokens] score
// tensor (ggml.h ggml_qsa_score / ggml_qsa_topk).

#include "common.cuh"

void ggml_cuda_op_qsa_score(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_qsa_topk(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_qsa_supported(const ggml_tensor * op);
