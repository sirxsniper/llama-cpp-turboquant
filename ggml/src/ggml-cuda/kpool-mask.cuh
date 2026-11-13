#pragma once

// [TAG_QSA_POOL_RANGE] GGML_OP_KPOOL_MASK, see ggml_kpool_mask in ggml.h

#include "common.cuh"

bool ggml_cuda_kpool_mask_supported(const ggml_tensor * op);

void ggml_cuda_op_kpool_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
