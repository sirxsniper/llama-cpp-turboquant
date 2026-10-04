#pragma once

// [TAG_MOE_BRIDGE] GGML_OP_MOE_HOST_POST / GGML_OP_MOE_HOST_WAIT, see ggml-moe-bridge.h

#include "common.cuh"

bool ggml_cuda_moe_bridge_supports_op(int device, const ggml_tensor * op);

void ggml_cuda_op_moe_host_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_moe_host_wait(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_moe_host_fetch(ggml_backend_cuda_context & ctx, ggml_tensor * dst); // [TAG_FN_R4_BRIDGE_DMA]
