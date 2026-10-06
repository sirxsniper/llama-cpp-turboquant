#pragma once

// [TAG_MOE_BRIDGE] GGML_OP_MOE_HOST_POST / GGML_OP_MOE_HOST_WAIT, see ggml-moe-bridge.h

#include "common.cuh"

struct ggml_cuda_topk_moe_args; // topk-moe.cuh

bool ggml_cuda_moe_bridge_supports_op(int device, const ggml_tensor * op);

void ggml_cuda_op_moe_host_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_moe_host_wait(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_moe_host_fetch(ggml_backend_cuda_context & ctx, ggml_tensor * dst); // [TAG_FN_R4_BRIDGE_DMA]

// [TAG_FN_L4_POST] the fused topk-moe nodes (ggml_cuda_op_topk_moe's arguments) and the MOE_HOST_POST `post` that reads
// their ids and weights, in one launch: same ids and weights, same job for the host. Only for a post marked
// GGML_FN_L4_POST (ggml-fn-l4-gpumoe.h), 512 experts, no bias, no delayed softmax. false: nothing launched.
bool ggml_cuda_fn_l4_topk_post(ggml_backend_cuda_context & ctx, const ggml_tensor * logits, ggml_tensor * weights,
        ggml_tensor * ids, const ggml_tensor * clamp, const ggml_tensor * scale, const ggml_tensor * bias,
        const ggml_cuda_topk_moe_args & args, ggml_tensor * post);
