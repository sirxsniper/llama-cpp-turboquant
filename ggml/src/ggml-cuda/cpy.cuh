#include "common.cuh"

#define CUDA_CPY_BLOCK_SIZE 64

void ggml_cuda_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1);

// [TAG_CPY_CHAIN_FUSION] one launch for n same-shaped F32 copies whose src/dst pointers advance by
// dsrc/ddst bytes per copy (deltas may be negative). src0/src1 describe the first copy.
void ggml_cuda_cpy_chain(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                         int n, int64_t dsrc, int64_t ddst);

void ggml_cuda_dup(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
