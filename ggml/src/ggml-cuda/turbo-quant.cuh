/*
 * TurboQuant CUDA kernels for KV cache compression
 * Based on: arXiv 2504.19874 (ICLR 2026)
 *
 * Implements GGML_TYPE_TURBO3_0 (3-bit PolarQuant, block size 32)
 * Constants, WHT rotation, quantize/dequantize device functions.
 */

#pragma once

#include "common.cuh"
#include "ggml-turbo-innerq.h"
#include <cstdlib>
#include <cmath>
#include <type_traits>

// ---- Quantization ratios for dequantize_block template ----
#define QR_TURBO3 1  // Each dequantize call produces 2 consecutive elements (like q8_0)
#define QR_TURBO2 1  // Each dequantize call produces 2 consecutive elements (like q8_0)
#define QR_TURBO4 1  // Each dequantize call produces 2 consecutive elements (like q8_0)
#define QR_TURBO4P 1 // Each dequantize call produces 2 consecutive elements (like q8_0)

// ---- 2-bit centroids (Lloyd-Max for N(0, 1/128)) ----

static __constant__ float TURBO_CENTROIDS_2BIT[4] = {
    -0.133462f, -0.039994f, 0.039994f, 0.133462f
};

static __constant__ float TURBO_MID_2BIT[3] = {
    -0.086728f, 0.0f, 0.086728f
};

// ---- 3-bit centroids (Lloyd-Max for N(0, 1/128)) ----

static __constant__ float TURBO_CENTROIDS_3BIT[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

// ---- Midpoints for nearest centroid lookup ----

static __constant__ float TURBO_MID_3BIT[7] = {
    -0.154259f, -0.091775f, -0.043589f, 0.0f,
     0.043589f,  0.091775f,  0.154259f
};

// ---- WHT sign arrays (seed=42) ----

static __constant__ float TURBO_WHT_SIGNS1[128] = {
    -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f,
    -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f
};

static __constant__ float TURBO_WHT_SIGNS2[128] = {
    1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f,
    -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f
};

// ---- 64-element WHT sign arrays (first 64 of the 128-element arrays) ----

static __constant__ float TURBO_WHT_SIGNS1_64[64] = {
    -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f
};

static __constant__ float TURBO_WHT_SIGNS2_64[64] = {
    1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f
};

// ---- Fast Walsh-Hadamard Transform (in-place, normalized) ----
// O(n log n) = 896 ops for n=128

static __device__ __forceinline__ void turbo_fwht_128(float * x) {
    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j];
                float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    const float inv_sqrt_128 = 0.08838834764831845f;
    for (int i = 0; i < 128; i++) {
        x[i] *= inv_sqrt_128;
    }
}

// ---- Fast Walsh-Hadamard Transform for 64-element groups ----
// O(n log n) = 384 ops for n=64

static __device__ __forceinline__ void turbo_fwht_64(float * x) {
    for (int h = 1; h < 64; h *= 2) {
        for (int i = 0; i < 64; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j];
                float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    const float inv_sqrt_64 = 0.125f;
    for (int i = 0; i < 64; i++) {
        x[i] *= inv_sqrt_64;
    }
}

// ---- Forward rotation: signs1 → FWHT → signs2 ----

static __device__ __forceinline__ void turbo_rotate_forward(float * x) {
    for (int i = 0; i < 128; i++) x[i] *= TURBO_WHT_SIGNS1[i];
    turbo_fwht_128(x);
    for (int i = 0; i < 128; i++) x[i] *= TURBO_WHT_SIGNS2[i];
}

// ---- Forward rotation for 64-element groups ----

static __device__ __forceinline__ void turbo_rotate_forward_64(float * x) {
    for (int i = 0; i < 64; i++) x[i] *= TURBO_WHT_SIGNS1_64[i];
    turbo_fwht_64(x);
    for (int i = 0; i < 64; i++) x[i] *= TURBO_WHT_SIGNS2_64[i];
}

// ---- [TAG_TURBO4_HIST] post-WHT histogram, DEBUG ONLY ----
// Set TURBO_WHT_HIST=<path> to accumulate a histogram of the post-WHT values that the
// 4-bit quantiser actually sees, then fit a codebook to THAT rather than to an assumed
// normal - which is exactly what the [TAG_TURBO4_CODEBOOK] note asks for. Off unless the
// env var is set: one predicated atomicAdd per element, and the branch is uniform.
#define TURBO_HIST_BINS 4096
#define TURBO_HIST_LO   (-0.6f)
#define TURBO_HIST_HI   ( 0.6f)
static __device__ unsigned long long d_wht_hist[TURBO_HIST_BINS];
static __device__ int d_wht_hist_on;

static __device__ __forceinline__ void turbo_wht_hist_add(float v) {
#ifndef TURBO_WHT_HIST_BUILD
    // Compiled out by default: the runtime flag alone still costs a __device__
    // global load per element on every KV write, which measured ~9% of pp and tg
    // at d131072. Build with -DTURBO_WHT_HIST_BUILD to re-enable the histogram.
    (void) v; return;
#else
    if (!d_wht_hist_on) return;
    const float t = (v - TURBO_HIST_LO) / (TURBO_HIST_HI - TURBO_HIST_LO);
    int b = (int) (t * TURBO_HIST_BINS);
    b = b < 0 ? 0 : (b >= TURBO_HIST_BINS ? TURBO_HIST_BINS - 1 : b);
    atomicAdd(&d_wht_hist[b], 1ULL);
#endif
}

static bool turbo_wht_hist_enabled = false;
static bool turbo_wht_hist_init_done = false;
static void turbo_wht_hist_init(void) {
    if (turbo_wht_hist_init_done) return;
    turbo_wht_hist_init_done = true;
    const char * e = getenv("TURBO_WHT_HIST");
    if (!e || !*e) return;
    turbo_wht_hist_enabled = true;
    unsigned long long z[TURBO_HIST_BINS] = {0};
    int one = 1;
    cudaMemcpyToSymbol(d_wht_hist, z, sizeof(z));
    cudaMemcpyToSymbol(d_wht_hist_on, &one, sizeof(int));
    GGML_LOG_INFO("%s: post-WHT histogram enabled -> %s\n", __func__, e);
}

// Dump and reset. Called opportunistically from the set-rows launcher.
static void turbo_wht_hist_dump(void) {
    if (!turbo_wht_hist_enabled) return;
    static int calls = 0;
    if (++calls % 512 != 0) return;                 // amortise the copy
    const char * e = getenv("TURBO_WHT_HIST");
    unsigned long long h[TURBO_HIST_BINS];
    cudaMemcpyFromSymbol(h, d_wht_hist, sizeof(h));
    unsigned long long tot = 0;
    for (int i = 0; i < TURBO_HIST_BINS; i++) tot += h[i];
    if (tot < 2000000ULL) return;                   // wait for a decent sample
    FILE * f = fopen(e, "wb");
    if (f) { fwrite(h, sizeof(unsigned long long), TURBO_HIST_BINS, f); fclose(f);
             GGML_LOG_INFO("turbo_wht_hist: wrote %llu samples to %s\n", tot, e); }
}

// ---- InnerQ per-channel equalization ----
// Equalizes K channel variances before WHT rotation to reduce quantization error.
// Enabled via TURBO_INNERQ=N env var (N = calibration token count).
// Math: <Q/s, s*K> = <Q, K> preserves dot products.
// INNERQ_MAX_CHANNELS is defined in turbo-innerq.cuh

static __device__ float d_innerq_scale[INNERQ_MAX_CHANNELS];
static __device__ float d_innerq_scale_inv[INNERQ_MAX_CHANNELS];
static __device__ float d_innerq_sq_accum[INNERQ_MAX_CHANNELS];
static __device__ int   d_innerq_count;
static __device__ int   d_innerq_active;       // 0 = scales are identity, 1 = scales applied
static __device__ int   d_innerq_calibrating;  // 1 = accumulating K² stats

static int  innerq_enabled       = 0;  // host: 0=off, 1=calibrating, 2=active
static int  innerq_target_tokens = 0;
static float innerq_strength     = 0.5f;
static bool  innerq_initialized  = false;

// Host: read TURBO_INNERQ env, start calibration if enabled
static void turbo_innerq_init(void) {
    if (innerq_initialized) return;
    innerq_initialized = true;

    const char * env = getenv("TURBO_INNERQ");
    if (!env || atoi(env) <= 0) {
        innerq_enabled = 0;
        return;
    }
    innerq_target_tokens = atoi(env);
    innerq_enabled = 1;  // calibrating

    const char * env_str = getenv("TURBO_INNERQ_STRENGTH");
    if (env_str) innerq_strength = atof(env_str);
    if (innerq_strength <= 0.0f || innerq_strength > 1.0f) innerq_strength = 0.5f;

    // Zero accumulators and set calibrating flag on device
    float zeros[INNERQ_MAX_CHANNELS] = {0};
    int zero = 0, one = 1;
    cudaMemcpyToSymbol(d_innerq_sq_accum, zeros, sizeof(zeros));
    cudaMemcpyToSymbol(d_innerq_count, &zero, sizeof(int));
    cudaMemcpyToSymbol(d_innerq_active, &zero, sizeof(int));
    cudaMemcpyToSymbol(d_innerq_calibrating, &one, sizeof(int));

    GGML_LOG_INFO("%s: InnerQ calibration started (target=%d tokens, strength=%.2f)\n",
                   __func__, innerq_target_tokens, innerq_strength);
}

// Host: finalize calibration — compute scales, upload, activate
static void turbo_innerq_finalize(int group_size) {
    // Read accumulators from device
    float sq_accum[INNERQ_MAX_CHANNELS];
    int count = 0;
    cudaMemcpyFromSymbol(sq_accum, d_innerq_sq_accum, group_size * sizeof(float));
    cudaMemcpyFromSymbol(&count, d_innerq_count, sizeof(int));

    if (count <= 0) {
        GGML_LOG_WARN("%s: InnerQ calibration got 0 tokens, disabling\n", __func__);
        innerq_enabled = 0;
        int zero = 0;
        cudaMemcpyToSymbol(d_innerq_calibrating, &zero, sizeof(int));
        return;
    }

    // Compute per-channel RMS
    float rms[INNERQ_MAX_CHANNELS];
    float mean_rms = 0.0f;
    float max_ratio = 0.0f, min_ratio = 1e30f;
    for (int i = 0; i < group_size; i++) {
        rms[i] = sqrtf(sq_accum[i] / (float)count);
        mean_rms += rms[i];
    }
    mean_rms /= (float)group_size;

    // Compute scale[i] = (mean_rms / channel_rms[i])^strength, clamp to [0.5, 2.0]
    float scale[INNERQ_MAX_CHANNELS];
    float scale_inv[INNERQ_MAX_CHANNELS];
    for (int i = 0; i < group_size; i++) {
        float ratio = (rms[i] > 1e-10f) ? (mean_rms / rms[i]) : 1.0f;
        float s = powf(ratio, innerq_strength);
        if (s < 0.5f) s = 0.5f;
        if (s > 2.0f) s = 2.0f;
        scale[i] = s;
        scale_inv[i] = 1.0f / s;
        if (ratio > max_ratio) max_ratio = ratio;
        if (ratio < min_ratio) min_ratio = ratio;
    }

    // Auto-skip if max channel ratio < 1.2 (already balanced)
    if (max_ratio < 1.2f && min_ratio > (1.0f / 1.2f)) {
        GGML_LOG_INFO("%s: InnerQ auto-disabled (channels already balanced, max_ratio=%.3f)\n",
                       __func__, max_ratio);
        innerq_enabled = 0;
        int zero = 0;
        cudaMemcpyToSymbol(d_innerq_calibrating, &zero, sizeof(int));
        return;
    }

    // Stop calibrating, upload scales, activate
    int zero = 0, one = 1;
    cudaMemcpyToSymbol(d_innerq_calibrating, &zero, sizeof(int));
    cudaMemcpyToSymbol(d_innerq_scale, scale, group_size * sizeof(float));
    cudaMemcpyToSymbol(d_innerq_scale_inv, scale_inv, group_size * sizeof(float));
    cudaDeviceSynchronize();  // ensure scales are visible before activating
    cudaMemcpyToSymbol(d_innerq_active, &one, sizeof(int));

    innerq_enabled = 2;  // active

    // Publish scale_inv to shared host state for cross-TU tensor update
    turbo_innerq_publish(scale_inv, group_size);

    GGML_LOG_INFO("%s: InnerQ finalized (%d tokens, max_ratio=%.3f, min_ratio=%.3f)\n",
                   __func__, count, max_ratio, min_ratio);
}

// Host: called before each set_rows kernel launch
static void turbo_innerq_check_finalize(int group_size, int64_t ne00) {
    if (!innerq_initialized) {
        turbo_innerq_init();
    }
    if (innerq_enabled == 0) return;

    // InnerQ only works when each WHT group = one head (group_size == head_dim).
    // For standard models: ne00 = n_heads * head_dim, group_size = head_dim → ne00 % group_size == 0, fine.
    // For non-standard models (head_dim > group_size, e.g. GLM 576 → 64-group):
    //   ne00 = head_dim (single head), group_size = 64, ne00/group_size = 9 groups per head → WRONG.
    // Detect: if ne00 / group_size doesn't divide evenly into standard head counts (1,2,4,8,16,32,64,128),
    // it's likely multi-group-per-head. Simpler check: group_size < 128 means head_dim > 128.
    const bool multi_group_per_head = (group_size < 128);  // 64-group → head_dim > 128, multi-group
    if (multi_group_per_head) {
        if (innerq_enabled == 1) {
            GGML_LOG_WARN("%s: InnerQ disabled (ne00=%lld != group_size=%d, multi-group heads)\n",
                           __func__, (long long)ne00, group_size);
            innerq_enabled = 0;
            int zero = 0;
            cudaMemcpyToSymbol(d_innerq_calibrating, &zero, sizeof(int));
        }
        return;
    }

    // Check if calibration is complete
    if (innerq_enabled == 1) {
        int count = 0;
        cudaMemcpyFromSymbol(&count, d_innerq_count, sizeof(int));
        if (count >= innerq_target_tokens) {
            turbo_innerq_finalize(group_size);
        }
    }
}

// Host: check if InnerQ is currently active (finalized)
static bool turbo_innerq_is_active(void) {
    return innerq_enabled == 2;
}

// ---- 4-bit centroids ----
//
// [TAG_TURBO4_CODEBOOK] Lloyd-Max for N(0, 1/128), the post-WHT component distribution.
//
// This REPLACES an equiprobable-bin codebook whose entries were the conditional means of
// 16 equal-probability bins. That table clipped 4.85% of all values at its outermost
// level of 1.968 sigma, against 0.59% here at 2.733 sigma, and measured 2.09x worse MSE
// on real data.
//
// The previous comment argued the post-WHT values are 'evidently lighter-tailed than
// normal' and recorded that Lloyd-Max measured WORSE end to end (acceptance 41.7 -> 33.0
// at depth 0). It asked for the codebook to be fitted to a HISTOGRAM of real post-WHT
// values instead of an assumed normal. That was done - see [TAG_TURBO4_HIST], which
// accumulates exactly that histogram when TURBO_WHT_HIST is set.
//
// MEASURED on 2.107e9 real post-WHT samples from Qwen3.8-27B K and V:
//   std       0.088388  = 1/sqrt(128) to six figures, exactly as theory predicts
//   kurtosis  2.954     vs 3.0 for a normal - lighter tailed, but only by 1.5%
//   MSE   equiprobable 1.522e-04 | Lloyd-Max 7.292e-05 | fitted-to-histogram 7.309e-05
// The empirical fit and Lloyd-Max agree to within 0.2%, so the distribution IS normal
// for codebook purposes and the earlier premise does not hold.
//
// Why the earlier Lloyd-Max attempt regressed is not proven, but note that FIVE tables
// are derived from these centroids: TURBO_C4_I8_LIST, the four packed LUT words, the
// static_asserts guarding them, TURBO_INT8_4BIT_SCALE_REVERSE and TURBO_MID_4BIT.
// Changing only the float array leaves the __dp4a K-dot path scoring against the OLD
// centroids while dequant uses the new ones - which would wreck draft acceptance and
// drag TG down with it, exactly the signature recorded above. Change all five together.

static __constant__ float TURBO_CENTROIDS_4BIT[16] = {
    -0.241530f, -0.182875f, -0.143021f, -0.111033f,
    -0.083297f, -0.058053f, -0.034304f, -0.011349f,
     0.011349f,  0.034304f,  0.058053f,  0.083297f,
     0.111033f,  0.143021f,  0.182875f,  0.241530f
};

// PERF (int8 / __dp4a path): pre-quantized 4-bit centroids as int8 in [-127, 127].
// Each value = round(centroid[i] / max_abs_centroid * 127). Allows turbo4 K dot to
// use the Blackwell __dp4a hardware instruction (same path q8 uses) instead of
// per-element float multiplies. Final scale factor TURBO_INT8_4BIT_SCALE recovers
// real value: float_K = int8_K * (norm * TURBO_INT8_4BIT_SCALE).
//   max_abs_centroid = 0.241530
//   int8 centroids   = round(centroid / 0.241530 * 127)
// Single source of truth for the int8 centroids. The __constant__ array below and the
// pre-packed LUT words are both derived from this, so they cannot drift apart.
#define TURBO_C4_I8_LIST -127, -96, -75, -58, -44, -31, -18,  -6, \
                            6,  18,  31,  44,  58,  75,  96, 127

static __constant__ int8_t TURBO_CENTROIDS_4BIT_INT8[16] = { TURBO_C4_I8_LIST };

// Host-side mirror, used only to compute the packed LUT words at compile time.
static constexpr int8_t TURBO_CENTROIDS_4BIT_I8_CE[16] = { TURBO_C4_I8_LIST };

// Four centroids packed per word, in the byte order turbo4_int8_lut::gather4 expects.
// Building these with constexpr rather than reading the __constant__ array at run time
// turns LUT setup from 16 constant-memory loads plus ~24 ALU ops per call into four
// immediates the compiler materialises with MOV32I - no memory traffic at all. Setup is
// paid once per dequant call, i.e. once per few KV elements, so it scales with depth.
static constexpr uint32_t turbo4_i8_pack(int b0, int b1, int b2, int b3) {
    return  (uint32_t) (uint8_t) TURBO_CENTROIDS_4BIT_I8_CE[b0]        |
           ((uint32_t) (uint8_t) TURBO_CENTROIDS_4BIT_I8_CE[b1] <<  8) |
           ((uint32_t) (uint8_t) TURBO_CENTROIDS_4BIT_I8_CE[b2] << 16) |
           ((uint32_t) (uint8_t) TURBO_CENTROIDS_4BIT_I8_CE[b3] << 24);
}
static constexpr uint32_t TURBO_C4_LUT_W0 = turbo4_i8_pack( 0,  1,  2,  3);
static constexpr uint32_t TURBO_C4_LUT_W1 = turbo4_i8_pack( 4,  5,  6,  7);
static constexpr uint32_t TURBO_C4_LUT_W2 = turbo4_i8_pack( 8,  9, 10, 11);
static constexpr uint32_t TURBO_C4_LUT_W3 = turbo4_i8_pack(12, 13, 14, 15);

// Guards against a silent change to the centroid list reordering the packed bytes.
static_assert(TURBO_C4_LUT_W0 == 0xC6B5A081u, "turbo4 LUT w0 changed");
static_assert(TURBO_C4_LUT_W1 == 0xFAEEE1D4u, "turbo4 LUT w1 changed");
static_assert(TURBO_C4_LUT_W2 == 0x2C1F1206u, "turbo4 LUT w2 changed");
static_assert(TURBO_C4_LUT_W3 == 0x7F604B3Au, "turbo4 LUT w3 changed");
// SCALE_REVERSE = 0.241530 / 127.0 — used to recover float dot product from int sum
#define TURBO_INT8_4BIT_SCALE_REVERSE (0.241530f / 127.0f)
// ---- [TAG_TURBO5P] int8 mirror of the 32-entry codebook, same derivation as 4-bit ----
//   max_abs_centroid = 0.271948
//   int8 centroids   = round(centroid / 0.271948 * 127)
#define TURBO_C5_I8_LIST -127, -104,  -88,  -76,  -66,  -58,  -50,  -43, \
                          -37,  -31,  -25,  -20,  -15,  -10,   -6,   -2, \
                            2,    6,   10,   15,   20,   25,   31,   37, \
                           43,   50,   58,   66,   76,   88,  104,  127
static __constant__ int8_t TURBO_CENTROIDS_5BIT_INT8[32] = { TURBO_C5_I8_LIST };
static constexpr int8_t TURBO_CENTROIDS_5BIT_I8_CE[32] = { TURBO_C5_I8_LIST };
static constexpr uint32_t turbo5_i8_pack(int b0, int b1, int b2, int b3) {
    return  (uint32_t) (uint8_t) TURBO_CENTROIDS_5BIT_I8_CE[b0]        |
           ((uint32_t) (uint8_t) TURBO_CENTROIDS_5BIT_I8_CE[b1] <<  8) |
           ((uint32_t) (uint8_t) TURBO_CENTROIDS_5BIT_I8_CE[b2] << 16) |
           ((uint32_t) (uint8_t) TURBO_CENTROIDS_5BIT_I8_CE[b3] << 24);
}
static constexpr uint32_t TURBO_C5_LUT_W0 = turbo5_i8_pack( 0,  1,  2,  3);
static constexpr uint32_t TURBO_C5_LUT_W1 = turbo5_i8_pack( 4,  5,  6,  7);
static constexpr uint32_t TURBO_C5_LUT_W2 = turbo5_i8_pack( 8,  9, 10, 11);
static constexpr uint32_t TURBO_C5_LUT_W3 = turbo5_i8_pack(12, 13, 14, 15);
static constexpr uint32_t TURBO_C5_LUT_W4 = turbo5_i8_pack(16, 17, 18, 19);
static constexpr uint32_t TURBO_C5_LUT_W5 = turbo5_i8_pack(20, 21, 22, 23);
static constexpr uint32_t TURBO_C5_LUT_W6 = turbo5_i8_pack(24, 25, 26, 27);
static constexpr uint32_t TURBO_C5_LUT_W7 = turbo5_i8_pack(28, 29, 30, 31);
static_assert(TURBO_C5_LUT_W0 == 0xB4A89881u, "turbo5 LUT w0 changed");
static_assert(TURBO_C5_LUT_W1 == 0xD5CEC6BEu, "turbo5 LUT w1 changed");
static_assert(TURBO_C5_LUT_W2 == 0xECE7E1DBu, "turbo5 LUT w2 changed");
static_assert(TURBO_C5_LUT_W3 == 0xFEFAF6F1u, "turbo5 LUT w3 changed");
static_assert(TURBO_C5_LUT_W4 == 0x0F0A0602u, "turbo5 LUT w4 changed");
static_assert(TURBO_C5_LUT_W5 == 0x251F1914u, "turbo5 LUT w5 changed");
static_assert(TURBO_C5_LUT_W6 == 0x423A322Bu, "turbo5 LUT w6 changed");
static_assert(TURBO_C5_LUT_W7 == 0x7F68584Cu, "turbo5 LUT w7 changed");
#define TURBO_INT8_5BIT_SCALE_REVERSE (0.271948f / 127.0f)

// ---- Midpoints for nearest 4-bit centroid lookup ----

static __constant__ float TURBO_MID_4BIT[15] = {
    -0.212203f, -0.162948f, -0.127027f, -0.097165f,
    -0.070675f, -0.046178f, -0.022826f,  0.000000f,
     0.022826f,  0.046178f,  0.070675f,  0.097165f,
     0.127027f,  0.162948f,  0.212203f
};
// ---- [TAG_TURBO5P] 5-bit codebook: 32 Lloyd-Max centroids fitted to the measured post-WHT
// histogram (see [TAG_TURBO4_CODEBOOK]). 3.60x less MSE than the 16-level table. Absolute
// scale is divided out by the norm correction; the SHAPE is what matters.
static __constant__ float TURBO_CENTROIDS_5BIT[32] = {
    -0.271948f, -0.222223f, -0.189260f, -0.163683f, -0.142366f, -0.123814f, -0.107237f, -0.092232f,
    -0.078519f, -0.065814f, -0.053979f, -0.042868f, -0.032338f, -0.022390f, -0.013026f, -0.004245f,
     0.004245f,  0.013026f,  0.022390f,  0.032338f,  0.042868f,  0.053979f,  0.065814f,  0.078519f,
     0.092232f,  0.107237f,  0.123814f,  0.142366f,  0.163683f,  0.189260f,  0.222223f,  0.271948f
};
static __constant__ float TURBO_MID_5BIT[31] = {
    -0.247085f, -0.205741f, -0.176471f, -0.153025f, -0.133090f, -0.115525f, -0.099734f, -0.085376f,
    -0.072167f, -0.059897f, -0.048423f, -0.037603f, -0.027364f, -0.017708f, -0.008635f,  0.000000f,
     0.008635f,  0.017708f,  0.027364f,  0.037603f,  0.048423f,  0.059897f,  0.072167f,  0.085376f,
     0.099734f,  0.115525f,  0.133090f,  0.153025f,  0.176471f,  0.205741f,  0.247085f
};
// Same branchless midpoint count as the 4-bit version: 31 independent compare+adds.
static __device__ __forceinline__ uint8_t turbo_nearest_centroid_5bit(float val) {
    int idx = 0;
#pragma unroll
    for (int i = 0; i < 31; ++i) {
        idx += (val >= TURBO_MID_5BIT[i]);
    }
    return (uint8_t) idx;
}

// ---- Nearest 4-bit centroid index ----

static __device__ __forceinline__ uint8_t turbo_nearest_centroid_4bit(float val) {
    // Branchless. TURBO_MID_4BIT is sorted ascending, so the index the if/else chain
    // returned is exactly "how many midpoints val is >= to" - identical semantics.
    //
    // Why it matters: this runs once per element for every K and V written, i.e. on
    // every token of prefill. The if/else form is a 15-deep DEPENDENT branch chain,
    // and because lanes in a warp exit at different depths the warp executes all 15
    // branches anyway - the early-out never pays, only the serial dependency costs.
    // This is 15 INDEPENDENT compare+add ops the scheduler can pipeline, with no
    // divergence. Every TURBO_MID_4BIT index is a compile-time constant, so each read
    // is a uniform constant-memory broadcast rather than a divergent lookup.
    int idx = 0;
#pragma unroll
    for (int i = 0; i < 15; ++i) {
        idx += (val >= TURBO_MID_4BIT[i]);
    }
    return (uint8_t) idx;
}

// ---- Per-block quantize for turbo4 (128 elements, expects already-rotated input) ----

static __device__ void quantize_f32_turbo4_0_block(const float * __restrict__ src,
                                                    block_turbo4_0 * __restrict__ dst) {
    for (int j = 0; j < QK_TURBO4 / 2; j++) dst->qs[j] = 0;

    for (int j = 0; j < QK_TURBO4; j++) {
        uint8_t idx = turbo_nearest_centroid_4bit(src[j]);
        dst->qs[j / 2] |= (idx & 0xF) << ((j % 2) * 4);
    }
}

// ---- Inline dequant helper: extract one float from turbo4 block ----

static __device__ __forceinline__ float turbo4_dequant_element(
        const block_turbo4_0 * __restrict__ x, int j, float norm) {
    uint8_t idx = (x->qs[j / 2] >> ((j % 2) * 4)) & 0xF;
    return TURBO_CENTROIDS_4BIT[idx] * norm;
}

// ---- turbo4p: same quantization as turbo4_0, split-plane layout ----
//
// The nibble order is IDENTICAL to turbo4_0 - element j of the 1024-element block lives in
// nibble (j%2) of qs[j/2], low nibble for even j - so the only thing a reader has to change
// is where the norm comes from. Eight independent 128-element WHT groups share one block,
// group g owning elements [128*g, 128*g+128) and its own norm at norm[g]. Splitting a
// dequant into "which norm" and "which centroid" keeps the norm lookup hoistable: it is
// uniform for a whole group, while the centroid index is per element.

static __device__ __forceinline__ float turbo4p_group_norm(
        const block_turbo4p_0 * __restrict__ x, int j) {
    return __half2float(x->norm[j / QK_TURBO4P_GROUP]);
}

static __device__ __forceinline__ float turbo4p_dequant_element(
        const block_turbo4p_0 * __restrict__ x, int j, float norm) {
    uint8_t idx = (x->qs[j / 2] >> ((j % 2) * 4)) & 0xF;
    return TURBO_CENTROIDS_4BIT[idx] * norm;
}

// ---- turbo4p bulk dequant, one lane's worth ----
//
// Shared by convert.cu (both the contiguous and the strided kernel) and getrows.cu so the
// three of them cannot drift apart on nibble order or on the norm-to-group mapping.
//
// turbo4p_lane_shape<dst_t>::elems is 16/sizeof(dst_t), so every lane stores EXACTLY 16
// bytes and a
// warp stores 512 contiguous bytes in ONE instruction - the widest perfectly coalesced
// store a warp can issue. It deliberately does not go higher even though the layout would
// permit a 16-byte qs LOAD (32 elements per lane, block offset 16*k from a 16-byte aligned
// block base): a lane can move at most 16 bytes per store, so 32 elements per lane means
// four stores at a 64-byte lane stride, and each of those touches 32 half-filled sectors
// instead of a contiguous run. The destination is 4x the traffic of the qs source here, so
// the store shape decides the shape of everything else.
//
// Alignment of the qs load, argued from the block geometry alone: sizeof(block_turbo4p_0)
// is 528 = 16*33 and qs sits at offset 0, so with CUDA's 128-byte buffer alignment and view
// offsets that are whole numbers of blocks, every block base is 16-byte aligned. jb is a
// multiple of elems, so the byte offset jb/2 is a multiple of elems/2 - 4 bytes for a
// 16-bit dst_t, 2 bytes for a 32-bit one. That is the
// width used, and no wider.
//
// raw_centroid is lane L's copy of TURBO_CENTROIDS_4BIT[L] for L < 16. Unlike turbo4_0 the
// centroid is NOT pre-scaled by the norm, because a warp here can span two WHT groups with
// two different norms. Scaling after the broadcast costs one multiply per element and is
// what lets the lane cover 8 elements instead of 4.

// Plain constexpr members rather than a constexpr function, so the launchers can use the
// same numbers on the host without dragging in relaxed-constexpr device call rules.
template <typename dst_t>
struct turbo4p_lane_shape {
    static constexpr int elems    = 16 / (int) sizeof(dst_t);  // 8 for f16/bf16, 4 for f32
    static constexpr int qs_bytes = elems / 2;                 // 4 for f16/bf16, 2 for f32
    static constexpr int per_warp = WARP_SIZE * elems;         // 256 for f16/bf16, 128 for f32
};

template <typename dst_t>
static __device__ __forceinline__ void turbo4p_dequant_lane(
        const block_turbo4p_0 * __restrict__ x, const int jb,
        const float raw_centroid, dst_t * __restrict__ yout) {

    constexpr int per_lane = turbo4p_lane_shape<dst_t>::elems;
    constexpr int qs_bytes = turbo4p_lane_shape<dst_t>::qs_bytes;
    using qs_word_t = typename std::conditional<qs_bytes == 4, uint32_t, uint16_t>::type;

    // Uniform across the whole group, so the compiler hoists it out of the unrolled loop.
    const float norm = __half2float(x->norm[jb / QK_TURBO4P_GROUP]);

    // jb is even, so element jb+e is nibble e counting from bit 0 of this word: byte jb/2
    // holds elements jb+0 (low nibble) and jb+1 (high nibble), byte jb/2+1 holds jb+2 and
    // jb+3, and so on. Same nibble order as turbo4_0.
    qs_word_t qsw;
    ggml_cuda_memcpy_1<qs_bytes>(&qsw, x->qs + jb/2);

    alignas(16) dst_t v[per_lane];
#pragma unroll
    for (int e = 0; e < per_lane; ++e) {
        const unsigned idx = (unsigned) (qsw >> (4*e)) & 0xFu;
        v[e] = (dst_t) (norm * __shfl_sync(0xFFFFFFFFu, raw_centroid, idx, WARP_SIZE));
    }

    ggml_cuda_memcpy_1<sizeof(dst_t)*per_lane>(yout, v);
}


// ---- [TAG_TURBO5P] turbo5p dequant helpers: turbo4p's, plus the high-bit plane ----
// raw_centroid for the lane helper is TURBO_CENTROIDS_5BIT[lane] for ALL 32 lanes - a 5-bit
// index addresses the full warp, so the __shfl_sync gather needs no lane<16 guard.
static __device__ __forceinline__ float turbo5p_group_norm(
        const block_turbo5p_0 * __restrict__ x, int j) {
    return __half2float(x->norm[j / QK_TURBO5P_GROUP]);
}

static __device__ __forceinline__ float turbo5p_dequant_element(
        const block_turbo5p_0 * __restrict__ x, int j, float norm) {
    const unsigned lo  = (x->qs[j / 2] >> ((j % 2) * 4)) & 0xFu;
    const unsigned hi  = (x->qh[j / 8] >> (j % 8)) & 1u;
    const uint8_t  idx = (uint8_t) (lo | (hi << 4));
    return TURBO_CENTROIDS_5BIT[idx] * norm;
}

// ---- turbo5p bulk dequant, one lane's worth ----
//
// Shared by convert.cu (both the contiguous and the strided kernel) and getrows.cu so the
// three of them cannot drift apart on nibble order or on the norm-to-group mapping.
//
// turbo5p_lane_shape<dst_t>::elems is 16/sizeof(dst_t), so every lane stores EXACTLY 16
// bytes and a
// warp stores 512 contiguous bytes in ONE instruction - the widest perfectly coalesced
// store a warp can issue. It deliberately does not go higher even though the layout would
// permit a 16-byte qs LOAD (32 elements per lane, block offset 16*k from a 16-byte aligned
// block base): a lane can move at most 16 bytes per store, so 32 elements per lane means
// four stores at a 64-byte lane stride, and each of those touches 32 half-filled sectors
// instead of a contiguous run. The destination is 4x the traffic of the qs source here, so
// the store shape decides the shape of everything else.
//
// Alignment of the qs load, argued from the block geometry alone: sizeof(block_turbo5p_0)
// is 528 = 16*33 and qs sits at offset 0, so with CUDA's 128-byte buffer alignment and view
// offsets that are whole numbers of blocks, every block base is 16-byte aligned. jb is a
// multiple of elems, so the byte offset jb/2 is a multiple of elems/2 - 4 bytes for a
// 16-bit dst_t, 2 bytes for a 32-bit one. That is the
// width used, and no wider.
//
// raw_centroid is lane L's copy of TURBO_CENTROIDS_5BIT[L] for L < 16. Unlike turbo4_0 the
// centroid is NOT pre-scaled by the norm, because a warp here can span two WHT groups with
// two different norms. Scaling after the broadcast costs one multiply per element and is
// what lets the lane cover 8 elements instead of 4.

// Plain constexpr members rather than a constexpr function, so the launchers can use the
// same numbers on the host without dragging in relaxed-constexpr device call rules.
template <typename dst_t>
struct turbo5p_lane_shape {
    static constexpr int elems    = 16 / (int) sizeof(dst_t);  // 8 for f16/bf16, 4 for f32
    static constexpr int qs_bytes = elems / 2;                 // 4 for f16/bf16, 2 for f32
    static constexpr int per_warp = WARP_SIZE * elems;         // 256 for f16/bf16, 128 for f32
};

template <typename dst_t>
static __device__ __forceinline__ void turbo5p_dequant_lane(
        const block_turbo5p_0 * __restrict__ x, const int jb,
        const float raw_centroid, dst_t * __restrict__ yout) {

    constexpr int per_lane = turbo5p_lane_shape<dst_t>::elems;
    constexpr int qs_bytes = turbo5p_lane_shape<dst_t>::qs_bytes;
    using qs_word_t = typename std::conditional<qs_bytes == 4, uint32_t, uint16_t>::type;

    // Uniform across the whole group, so the compiler hoists it out of the unrolled loop.
    const float norm = __half2float(x->norm[jb / QK_TURBO5P_GROUP]);

    // jb is even, so element jb+e is nibble e counting from bit 0 of this word: byte jb/2
    // holds elements jb+0 (low nibble) and jb+1 (high nibble), byte jb/2+1 holds jb+2 and
    // jb+3, and so on. Same nibble order as turbo4_0.
    qs_word_t qsw;
    ggml_cuda_memcpy_1<qs_bytes>(&qsw, x->qs + jb/2);
    // [TAG_TURBO5P] per_lane high bits start at bit jb of the qh plane; jb is a multiple of
    // per_lane (4 or 8) so they never straddle a byte: bits (jb%8)..(jb%8)+per_lane-1 of qh[jb/8].
    const unsigned qhw = ((unsigned) x->qh[jb / 8]) >> (jb % 8);

    alignas(16) dst_t v[per_lane];
#pragma unroll
    for (int e = 0; e < per_lane; ++e) {
        const unsigned idx = ((unsigned) (qsw >> (4*e)) & 0xFu) | (((qhw >> e) & 1u) << 4);
        v[e] = (dst_t) (norm * __shfl_sync(0xFFFFFFFFu, raw_centroid, idx, WARP_SIZE));
    }

    ggml_cuda_memcpy_1<sizeof(dst_t)*per_lane>(yout, v);
}

// ---- Nearest 3-bit centroid index ----

static __device__ __forceinline__ uint8_t turbo_nearest_centroid_3bit(float val) {
    if      (val < TURBO_MID_3BIT[0]) return 0;
    else if (val < TURBO_MID_3BIT[1]) return 1;
    else if (val < TURBO_MID_3BIT[2]) return 2;
    else if (val < TURBO_MID_3BIT[3]) return 3;
    else if (val < TURBO_MID_3BIT[4]) return 4;
    else if (val < TURBO_MID_3BIT[5]) return 5;
    else if (val < TURBO_MID_3BIT[6]) return 6;
    else                              return 7;
}

// ---- Per-block quantize (32 elements, expects already-rotated input) ----
// Used by set_rows after group-level WHT rotation

static __device__ void quantize_f32_turbo3_0_block(const float * __restrict__ src,
                                                    block_turbo3_0 * __restrict__ dst) {
    for (int j = 0; j < QK_TURBO3 / 4; j++) dst->qs[j] = 0;
    for (int j = 0; j < QK_TURBO3 / 8; j++) dst->signs[j] = 0;

    for (int j = 0; j < QK_TURBO3; j++) {
        uint8_t idx = turbo_nearest_centroid_3bit(src[j]);
        dst->qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
        if (idx & 0x4) {
            dst->signs[j / 8] |= (1 << (j % 8));
        }
    }
}

// ---- Inline dequant helper: extract one float from turbo3 block ----

static __device__ __forceinline__ float turbo3_dequant_element(
        const block_turbo3_0 * __restrict__ x, int j, float norm) {
    uint8_t low2 = (x->qs[j / 4] >> ((j % 4) * 2)) & 0x3;
    uint8_t hi1  = (x->signs[j / 8] >> (j % 8)) & 0x1;
    uint8_t idx  = low2 | (hi1 << 2);
    return TURBO_CENTROIDS_3BIT[idx] * norm;
}

// ---- Nearest 2-bit centroid index ----

static __device__ __forceinline__ uint8_t turbo_nearest_centroid_2bit(float val) {
    if      (val < TURBO_MID_2BIT[0]) return 0;
    else if (val < TURBO_MID_2BIT[1]) return 1;
    else if (val < TURBO_MID_2BIT[2]) return 2;
    else                              return 3;
}

// ---- Per-block quantize for turbo2 (32 elements, expects already-rotated input) ----

static __device__ void quantize_f32_turbo2_0_block(const float * __restrict__ src,
                                                    block_turbo2_0 * __restrict__ dst) {
    for (int j = 0; j < QK_TURBO2 / 4; j++) dst->qs[j] = 0;

    for (int j = 0; j < QK_TURBO2; j++) {
        uint8_t idx = turbo_nearest_centroid_2bit(src[j]);
        dst->qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
    }
}

// ---- Inline dequant helper: extract one float from turbo2 block ----

static __device__ __forceinline__ float turbo2_dequant_element(
        const block_turbo2_0 * __restrict__ x, int j, float norm) {
    uint8_t idx = (x->qs[j / 4] >> ((j % 4) * 2)) & 0x3;
    return TURBO_CENTROIDS_2BIT[idx] * norm;
}
