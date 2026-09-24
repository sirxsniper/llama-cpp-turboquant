#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);

// Returns true if chunked prefill can be used; false for recurrent kernel
bool ggml_cuda_gdn_op_is_chunked(const ggml_tensor * dst);

// [TAG_4C_GDN_REPLAY] GGML_OP_GATED_DELTA_NET_REPLAY: committed state and new ring go to dst, or straight into the
// recurrent cache when fused (see ggml_cuda_try_gdn_replay_fusion)
struct ggml_cuda_gdn_replay_out {
    float * state; // committed states, per-seq stride D
    float * ring;  // new rings, per-seq stride ring->ne[0]
};

bool ggml_cuda_gdn_replay_supported(const ggml_tensor * dst);
void ggml_cuda_op_gated_delta_net_replay(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gdn_replay_out * out);
