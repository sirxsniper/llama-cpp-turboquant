#pragma once

// [TAG_MOE_BRIDGE] GGML_OP_MOE_HOST_POST / GGML_OP_MOE_HOST_WAIT, see ggml-moe-bridge.h

#include "common.cuh"
#include "ggml-fn-l22-prior.h" // [TAG_FN_L22_X_PRIOR] the prior's mark (ggml-cuda.cu tests it before the topk-moe fusion)

struct ggml_cuda_topk_moe_args; // topk-moe.cuh

bool ggml_cuda_moe_bridge_supports_op(int device, const ggml_tensor * op);
bool ggml_cuda_moe_bridge_hostfunc_any(); // [TAG_UPSTREAM_30804] a hostfunc-mode bridge exists (no graph chunk cuts)
// [TAG_FN_BRIDGE_GRAPHGEN] goes up when a bridge is freed or its device view changes: CUDA graphs captured before hold
// the old bridge's kernel arguments and host nodes, and a new bridge can get the same id and memory
uint32_t ggml_cuda_moe_bridge_generation();

// [TAG_FN_L22_DMASHARE] R-P14-2 (a): the device views a fetch of channel chan reads (P14's fetch from pinned pages waits
// and fails exactly as the ring fetch does). false: no such bridge / channel, another device, or no plan areas
struct ggml_cuda_mb_fetch_view {
    struct ggml_moe_bridge_plan * plan;  // the channel's plan area (device view of mapped memory)
    const uint32_t * plan_word;          // host writes: the job whose plan is published
    const uint32_t * done_word;          // host writes: the answered job
    const uint32_t * taken_word;         // host writes: the job being computed
    uint32_t       * err_state;          // device memory: the sticky bridge error
    const uint32_t * release;            // host writes: nonzero = every wait and fetch gives up at once
    int32_t        * g_err_chan;         // mapped: the first failing channel / job / error
    uint32_t       * g_err_seq;
    uint32_t       * g_err;
    unsigned long long timeout_ns;
    unsigned long long job_max_ns;
    int              max_fetch;
};
bool ggml_cuda_moe_bridge_fetch_view(int32_t bridge_id, int device, int32_t chan, ggml_cuda_mb_fetch_view * out);

void ggml_cuda_op_moe_host_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_moe_host_wait(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// [TAG_FN_L18_GPUFLOOR_PFWAIT] GGML_CUDA_FN_L18_PFWAIT=1 (ggml-cuda.cu): the L2 prefetch of a marked bridge wait
// (TAG_FN_L6_L2PF) is done by the wait launch itself, with real loads from extra blocks while it waits. ggml-cuda.cu hands
// the ranges for the wait node right before it computes it (16-byte aligned starts); the wait launch takes them.
#define GGML_CUDA_FN_L18_PF_MAX 64
bool ggml_cuda_fn_l18_pfwait_enabled();
void ggml_cuda_fn_l18_pf_set(int device, const ggml_tensor * wait, const char * const * p, const size_t * n, int nr);
void ggml_cuda_op_moe_host_fetch(ggml_backend_cuda_context & ctx, ggml_tensor * dst); // [TAG_FN_R4_BRIDGE_DMA]

// [TAG_FN_L4_POST] the fused topk-moe nodes (ggml_cuda_op_topk_moe's arguments) and the MOE_HOST_POST `post` that reads
// their ids and weights, in one launch: same ids and weights, same job for the host. Only for a post marked
// GGML_FN_L4_POST (ggml-fn-l4-gpumoe.h), 512 experts, no bias, no delayed softmax. false: nothing launched.
bool ggml_cuda_fn_l4_topk_post(ggml_backend_cuda_context & ctx, const ggml_tensor * logits, ggml_tensor * weights,
        ggml_tensor * ids, const ggml_tensor * clamp, const ggml_tensor * scale, const ggml_tensor * bias,
        const ggml_cuda_topk_moe_args & args, ggml_tensor * post);

// [TAG_FN_L22_X_PRIOR] R-P12-1 (ggml-cuda.cu): the route prior's marked pattern at node i: SOFT_MAX of 512 logits (<= 8 rows),
// RESHAPE, the MUL marked GGML_FN_L22_PRIOR, ARGSORT, VIEW, GET_ROWS [, the norm] [, SCALE]. prior_nodes: its node count,
// 0 when it does not match (graph metadata only: graph_optimize keeps its nodes allocated as it does for topk-moe).
// try_prior: the prior / resident top-k in one launch, with the bridge post marked GGML_FN_L4_POST behind it in the same
// launch when there is one; returns the nodes to skip after i (0: not taken, nothing launched)
int ggml_cuda_fn_l22_prior_nodes(const ggml_cgraph * cgraph, int i);
int ggml_cuda_fn_l22_try_prior(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i);
