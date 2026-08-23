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
    //   tile I 128->256: REJECTED. It reported +49%/+50% on MUL_MAT microbenchmarks
    //                    and was first thought to break only MUL_MAT_ID (MoE), so
    //                    it was gated to the dense path. That was wrong: I=256
    //                    also corrupts DENSE matmul at production (m, k) with
    //                    n >= ~11, producing NaN / ERR ~1.0 (127 of 128 cases at
    //                    Qwen3.8-27B shapes). End to end the model emitted pure
    //                    garbage tokens, and the corrupted hidden state poisoned
    //                    the KV cache for every later request in the process.
    //                    The MMQ kernel has internal invariants assuming I <= 128
    //                    and no upstream config uses I=256 for any type or arch.
    //                    Do not reintroduce it without reworking the SRAM layout,
    //                    stream-k fixup and write-back paths.
    //   nthreads 256->512 at I=128: REJECTED. CUDA "illegal memory access", crashes
    //                    test-backend-ops outright. Upstream does ship 148 configs at
    //                    nthreads=512/occ=1/I=128, but every one is in mmq-config-cdna.cuh:
    //                    CDNA is AMD with a 64-wide warp, so 512 threads is 8 warps there,
    //                    the same warp count NVIDIA gets from 256. The tile code is written
    //                    in warps and does not go past 8. No NVIDIA arch uses 512 anywhere.
    //                    (The dp4a vec_dot also declares y_df[J/nwarps] while stepping j by
    //                    nwarps, so J must be a non-zero multiple of nwarps - J=8,24,40 fail
    //                    to compile outright at nwarps=16.)
    //   K_vram is MMQ_ITER_K in every table for every arch and type, and stream_k is
    //   pinned to occupancy (occ 1 <-> stream_k true). With I, nthreads and occupancy
    //   all spent, this table has no headroom left for K-quants on sm_120: the Ampere
    //   fallthrough is effectively already the tuned answer. The has_ids split is kept
    //   so dense and MoE can still be tuned apart if a future kernel allows it.

    // Dense matmul only. I=256 corrupts MUL_MAT_ID (MoE): the ids path
    // has an internal invariant that assumes I<=128, and no upstream
    // config uses I=256 anywhere. MoE falls through to the Ampere table.


    // Q5_K and IQ4_XS: these dominate the Unsloth Dynamic "Q4_K_XL" mixes
    // (measured composition: 68.87% Q5_K, 21.21% IQ4_XS, only 4.74% real Q4_K),
    // so tuning Q4_K alone moved that model just +4.8% end-to-end.
    // Dense matmul only. I=256 corrupts MUL_MAT_ID (MoE): the ids path
    // has an internal invariant that assumes I<=128, and no upstream
    // config uses I=256 anywhere. MoE falls through to the Ampere table.

    // Dense matmul only. I=256 corrupts MUL_MAT_ID (MoE): the ids path
    // has an internal invariant that assumes I<=128, and no upstream
    // config uses I=256 anywhere. MoE falls through to the Ampere table.



    return ggml_cuda_mmq_get_config_ampere(type, J, fallback, has_ids);
}
