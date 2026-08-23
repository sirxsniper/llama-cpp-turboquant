static constexpr __host__ __device__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config_blackwell(ggml_type type, int J, bool fallback, bool has_ids) {
    CASE(GGML_TYPE_MXFP4, 256, 1, 128,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true);
    CASE(GGML_TYPE_MXFP4, 256, 1, 128,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true);
    CASE(GGML_TYPE_MXFP4, 256, 1, 128,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true);
    CASE(GGML_TYPE_MXFP4, 256, 1, 128,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true);
    CASE(GGML_TYPE_MXFP4, 256, 1, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true);
    CASE(GGML_TYPE_MXFP4, 256, 1, 128,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_MXFP4, 256, 1, 128,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_MXFP4, 256, 1, 128,  24, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_MXFP4, 256, 1, 128,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_MXFP4, 256, 1, 128,  40, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_MXFP4, 256, 1, 128,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_MXFP4, 256, 1, 128,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_MXFP4, 256, 1, 128,  80, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_MXFP4, 256, 1, 128,  96, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_MXFP4, 256, 1, 128, 112, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_MXFP4, 256, 1, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);

    CASE(GGML_TYPE_NVFP4, 256, 1, 128,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  24, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  40, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  80, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  96, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128, 112, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);
    CASE(GGML_TYPE_NVFP4, 256, 1, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false);


    // ---- Blackwell tuning for the integer-MMA K-quants --------------------
    // Upstream only tunes MXFP4/NVFP4 here; every K-quant falls through to the
    // Ampere table below. On sm_120 that leaves Q6_K at ~23% and Q4_K at ~31%
    // of int8 tensor-core peak (measured: 190 / 260 TFLOPS vs NVFP4's 550).
    // Tuning history on RTX 5090 (sm_120), Qwen3.8-27B shapes:
    //   occupancy 1->2 : Q6_K -27%, Q4_K -40%. These kernels are register-limited,
    //                    not latency-limited.
    //   tile I 128->256: Q6_K +49%, Q4_K +50% on MUL_MAT, BUT it corrupts
    //                    MUL_MAT_ID (MoE) - ERR ~1.0 on 39 cases. No upstream
    //                    config uses I=256 anywhere; the ids path does not
    //                    support it. Reverted.
    //   nthreads 256->512 at the supported I=128: under test here.
    // Dense matmul only. I=256 corrupts MUL_MAT_ID (MoE): the ids path
    // has an internal invariant that assumes I<=128, and no upstream
    // config uses I=256 anywhere. MoE falls through to the Ampere table.
    if (!has_ids) {
    CASE(GGML_TYPE_Q6_K, 256, 1, 256,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q6_K, 256, 1, 256,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q6_K, 256, 1, 256,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q6_K, 256, 1, 256,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q6_K, 256, 1, 256, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q6_K, 256, 1, 256,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q6_K, 256, 1, 256,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q6_K, 256, 1, 256,  24, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q6_K, 256, 1, 256,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q6_K, 256, 1, 256,  40, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q6_K, 256, 1, 256,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q6_K, 256, 1, 256,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q6_K, 256, 1, 256,  80, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q6_K, 256, 1, 256,  96, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q6_K, 256, 1, 256, 112, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q6_K, 256, 1, 256, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, true, false);
    }

    // Dense matmul only. I=256 corrupts MUL_MAT_ID (MoE): the ids path
    // has an internal invariant that assumes I<=128, and no upstream
    // config uses I=256 anywhere. MoE falls through to the Ampere table.
    if (!has_ids) {
    CASE(GGML_TYPE_Q4_K, 256, 1, 256,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q4_K, 256, 1, 256,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q4_K, 256, 1, 256,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q4_K, 256, 1, 256,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q4_K, 256, 1, 256, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q4_K, 256, 1, 256,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_K, 256, 1, 256,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_K, 256, 1, 256,  24, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_K, 256, 1, 256,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_K, 256, 1, 256,  40, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_K, 256, 1, 256,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_K, 256, 1, 256,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_K, 256, 1, 256,  80, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_K, 256, 1, 256,  96, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_K, 256, 1, 256, 112, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_K, 256, 1, 256, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    }


    // Q5_K and IQ4_XS: these dominate the Unsloth Dynamic "Q4_K_XL" mixes
    // (measured composition: 68.87% Q5_K, 21.21% IQ4_XS, only 4.74% real Q4_K),
    // so tuning Q4_K alone moved that model just +4.8% end-to-end.
    // Dense matmul only. I=256 corrupts MUL_MAT_ID (MoE): the ids path
    // has an internal invariant that assumes I<=128, and no upstream
    // config uses I=256 anywhere. MoE falls through to the Ampere table.
    if (!has_ids) {
    CASE(GGML_TYPE_Q5_K, 256, 1, 256,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q5_K, 256, 1, 256,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q5_K, 256, 1, 256,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q5_K, 256, 1, 256,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q5_K, 256, 1, 256, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q5_K, 256, 1, 256,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q5_K, 256, 1, 256,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q5_K, 256, 1, 256,  24, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q5_K, 256, 1, 256,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q5_K, 256, 1, 256,  40, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q5_K, 256, 1, 256,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q5_K, 256, 1, 256,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q5_K, 256, 1, 256,  80, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q5_K, 256, 1, 256,  96, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q5_K, 256, 1, 256, 112, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q5_K, 256, 1, 256, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    }

    // Dense matmul only. I=256 corrupts MUL_MAT_ID (MoE): the ids path
    // has an internal invariant that assumes I<=128, and no upstream
    // config uses I=256 anywhere. MoE falls through to the Ampere table.
    if (!has_ids) {
    CASE(GGML_TYPE_IQ4_XS, 256, 1, 256,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_IQ4_XS, 256, 1, 256,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_IQ4_XS, 256, 1, 256,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_IQ4_XS, 256, 1, 256,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_IQ4_XS, 256, 1, 256, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_IQ4_XS, 256, 1, 256,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_IQ4_XS, 256, 1, 256,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_IQ4_XS, 256, 1, 256,  24, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_IQ4_XS, 256, 1, 256,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_IQ4_XS, 256, 1, 256,  40, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_IQ4_XS, 256, 1, 256,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_IQ4_XS, 256, 1, 256,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_IQ4_XS, 256, 1, 256,  80, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_IQ4_XS, 256, 1, 256,  96, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_IQ4_XS, 256, 1, 256, 112, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_IQ4_XS, 256, 1, 256, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    }

    return ggml_cuda_mmq_get_config_ampere(type, J, fallback, has_ids);
}
