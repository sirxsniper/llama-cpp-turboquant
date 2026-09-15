#pragma once

#include "common.cuh"

// [TAG_TURBOT] CUDA writer of the turbot tiered KV cache, GGML_OP_TURBOT_SET_ROWS (docs/turbot/SPEC.md section 8).
// Host declarations only: this header must not pull turbot-tables.cuh into ggml-cuda.cu (SPEC 7.1).
void ggml_cuda_op_turbot_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_turbot_set_rows_supported(const ggml_tensor * op);
