#pragma once

// [TAG_FN_L4_HC] Flash-Next (qwen4exp) lever round 4: a whole hc mixer in three launches. The graph marks the chain
// (ggml-fn-l3.h: GGML_FN_L4_HC / GGML_FN_L4_HCPOST / GGML_FN_L4_HCQ8, only with LLAMA_FN_L4_HC), ggml-cuda.cu matches it:
//   [SCALE -> SIGMOID -> SCALE -> DSV4_HC_POST ->] RMS_NORM -> MUL -> MUL_MAT down [-> MUL_MAT inject] -> SCALE -> SILU
//   -> MUL_MAT up -> DSV4_HC_PRE
//   K1 norm: the combine (weights 2*sigmoid(inject/hc) computed in place), the grouped rms_norm * gamma, the q8_1 copy
//   K2 down: the down mat-vec (mul_mat_vec_q's sums, one row per block) and the inject mat-vec's per-warp sums
//            (mul_mat_vec_f's) in one grid
//   K3 up:   the inject's last reduction, scale -> silu -> q8_1 of the low rank (in shared memory), the up mat-vec
//            (mul_mat_vec_q's sums), dsv4_hc_pre, and the q8_1 copy of its output (GGML_FN_L4_HCQ8)
// Each value is computed by the same operations in the same order as the unfused kernels, so the same bits.
// GGML_CUDA_FN_L4=0 turns it off in one binary (the marks are then ignored).

#include "common.cuh"
#include "ggml-fn-l3.h"

// false when GGML_CUDA_FN_L4=0
bool ggml_cuda_fn_l4_enabled();

// mmvq.cu: true when mul_mat_vec_q takes its generic launch table on this device (4 warps for 1..4 columns)
bool ggml_cuda_fn_l4_mmvq_generic(int cc);

// the tensors of one matched mixer; s1 == nullptr: no combine before it (the norm reads rms->src[0])
struct ggml_cuda_fn_l4_hc_chain {
    const ggml_tensor * s1     = nullptr; // SCALE 1/hc of the raw inject
    const ggml_tensor * s2     = nullptr; // SCALE 2 after the sigmoid
    ggml_tensor *       post   = nullptr; // DSV4_HC_POST (written)
    const ggml_tensor * rms    = nullptr; // RMS_NORM (not written)
    ggml_tensor *       mul    = nullptr; // MUL: the normed streams xn (written)
    const ggml_tensor * down   = nullptr; // MUL_MAT down (not written: only the SCALE reads it)
    ggml_tensor *       inject = nullptr; // MUL_MAT inject (written), nullptr: none
    bool                inject_fused = false; // false: the inject runs as its own op (plain_mm)
    int                 inject_bs    = 0;     // the block size mul_mat_vec_f takes for it
    const ggml_tensor * lo_s   = nullptr; // SCALE of the low rank (not written)
    const ggml_tensor * up     = nullptr; // MUL_MAT up (not written)
    ggml_tensor *       pre    = nullptr; // DSV4_HC_PRE (written)
    bool                q8     = false;   // write the q8_1 copy of pre's output (reuse cache)
};

// launches the three kernels; plain_mm computes a MUL_MAT node as the backend would (the inject when not fused)
void ggml_cuda_fn_l4_hc_run(ggml_backend_cuda_context & ctx, const ggml_cuda_fn_l4_hc_chain & c, ggml_cgraph * cgraph,
        int i_first, int i_last, void (*plain_mm)(ggml_backend_cuda_context & ctx, ggml_tensor * dst));
