#pragma once

#include "common.cuh"

// [TAG_MMSB] [TAG_SMALLB] Small-batch quantized matmul: 2..16 src1 columns of Q4_K / Q5_K / Q6_K / Q8_0 weights on the
// int8 tensor cores, one pass over the weights (mmsb.cu). It is routed ahead of MMVQ/MMQ in ggml_cuda_mul_mat; MMQ
// itself is untouched, so prefill is not affected.
//
// Environment (read once; A/B switches only, the measured defaults go into code afterwards):
//   GGML_CUDA_SMALLB=0              kill switch: today's dispatch exactly
//   GGML_CUDA_SMALLB=1              also enable on any Ampere+ NVIDIA GPU (unset: Blackwell cc 1200 only)
//   GGML_CUDA_SMALLB_MIN=n          also take widths >= n that MMVQ would take (phase 2: n = 2)
//   GGML_CUDA_SMALLB_MAX=n          upper width, 1..16 (default 16)
//   GGML_CUDA_SMALLB_RG=1|2|4       force the row-group count
//   GGML_CUDA_SMALLB_MAX_ROWS=n     decline weights with more than n rows (isolates the LM head)
//   GGML_CUDA_SMALLB_XPF=1          test only: prefetch the weights before the PDL wait for every src0 buffer
//   TURBO_PATH_PROBE=1              one stderr line per (type, NT, RG)

// True when the small-batch kernel should compute dst = src0 x src1 (cc = the device's compute capability).
bool ggml_cuda_should_use_mmsb(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);

// Quantizes src1 to q8_1 (MMQ block layout) and runs the small-batch kernel. Only valid after
// ggml_cuda_should_use_mmsb returned true for the same tensors.
void ggml_cuda_mul_mat_sb(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
