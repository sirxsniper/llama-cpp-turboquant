#pragma once

#include "common.cuh"

// [TAG_MMQSN] MMQ with a cp.async weight ring for 5..16 src1 columns of Q4_K / Q5_K / Q6_K weights (mmqsn-impl.cuh,
// mmqsn.cu). Bit-identical to MMQ; it only overlaps the weight loads with the compute. Routed after mmsb and before
// MMVQ/MMQ in ggml_cuda_mul_mat; MMQ itself is untouched (prefill is not affected).
//
// Environment (read once). [TAG_MMQSN_DEFAULT] on by default on cc 1200; GGML_CUDA_MMQSN=0 restores MMQ exactly:
//   GGML_CUDA_MMQSN=1                 (default) on for cc 1200 (RTX 50xx) only; =2 on for any Ampere+ NVIDIA GPU (tests);
//                                     =0 off
//   GGML_CUDA_MMQSN_MODE=ring|pf|stream  K-loop: cp.async ring (default), L2 prefetch only, or the ring without compute
//                                     (stream measures the ceiling and gives WRONG output)
//   GGML_CUDA_MMQSN_PF=n              L2 prefetch distance in steps (default 0 for ring, 2 for pf)
//   GGML_CUDA_MMQSN_PF_RUN=1|2|4      steps per prefetch run (contiguous bytes per row), default 1
//   GGML_CUDA_MMQSN_L2HINT=0|1        cp.async with the .L2::256B hint (default 1)
//   GGML_CUDA_MMQSN_MIN=n             0 (default) = only the widths MMQ takes today; n = also MMVQ's widths >= n
//   GGML_CUDA_MMQSN_MAX=n             widest src1 batch, 2..16 (default 16)
//   GGML_CUDA_MMQSN_MIN_ROWS=n        smallest weight row count (default 2048)
//   GGML_CUDA_MMQSN_MAX_ROWS=n        largest weight row count (default 0 = no limit)
//   GGML_CUDA_MMQSN_TYPES=mask        1 = Q4_K, 2 = Q5_K, 4 = Q6_K (default 7)
//   GGML_CUDA_MMQSN_PERSIST=0|1       persistent grid where MMQ uses its tiling grid (the LM head), default 1
//   GGML_CUDA_MMQSN_XPF=1             test only: weight copies before the PDL wait for any src0 buffer
//   GGML_CUDA_MMQSN_CHECK=1           also run stock MMQ into a scratch buffer and compare bitwise (no CUDA graphs:
//                                     set GGML_CUDA_DISABLE_GRAPHS=1)
//   TURBO_PATH_PROBE=1                one env line, then one line per (type, J, mode, grid, persist, fixup)

// True when the ring MMQ should compute dst = src0 x src1 (cc = the device's compute capability).
bool ggml_cuda_should_use_mmqsn(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);

// Quantizes src1 to q8_1 exactly as MMQ does and runs the ring MMQ. Only valid after ggml_cuda_should_use_mmqsn
// returned true for the same tensors.
void ggml_cuda_mul_mat_qsn(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
