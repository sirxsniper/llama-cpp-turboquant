#pragma once

// [TAG_FN_R4_QSA_POS] GGML_OP_QSA_MASK, see ggml_qsa_mask in ggml.h

#include "common.cuh"

bool ggml_cuda_qsa_mask_supported(const ggml_tensor * op);

void ggml_cuda_op_qsa_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
