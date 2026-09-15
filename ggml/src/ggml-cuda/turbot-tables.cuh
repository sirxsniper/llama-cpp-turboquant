#pragma once

// [TAG_TURBOT] Device copies of the turbot codebook tables, shared by the CUDA reader (fattn-turbot.cuh, owner B) and
// writer (turbot-set-rows.cu, owner C). Owned by the architect; do not edit, regenerate ggml/include/ggml-turbot-tables.h.
//
// Include this ONLY from fattn-turbot.cuh (compiled into the turbot template-instance TUs) and turbot-set-rows.cu. No
// existing kernel TU and not fattn.cu may include it: unreferenced __constant__ tables must not be able to move the
// constant memory of existing kernels (SPEC 7.1).
//
// These expand the SAME generated lists as the host arrays in ggml-turbot.h, so CPU reference, writer and reader use
// bit-identical float32 values. A data-dependent index into __constant__ memory replays per distinct address (see
// [TAG_TURBO4P_CENT]); readers copy the runs they need into shared memory once per call (SPEC 7.4 gives the exact
// place) rather than gather from these arrays per element. Indices that are uniform across lanes (a head's width, a
// run offset) broadcast.
//
// No int8 mirror of the young LUT on purpose: measured, a direct int8 young LUT costs x1.5-1.6 young error at y = 7
// (SPEC 7.4). Old reads of b <= 5 may use int8 register LUTs (ggml_turbot_old_level_i8; for b = 4, 5 the existing
// turbo4_int8_lut / turbo5_int8_lut).

#include "common.cuh"
#include "ggml-turbot.h"

static __constant__ float   TURBOT_D_OLD_LEVELS[GGML_TURBOT_OLD_TOTAL]   = { GGML_TURBOT_OLD_LEVELS_LIST };
static __constant__ float   TURBOT_D_OLD_THR   [GGML_TURBOT_OLD_TOTAL]   = { GGML_TURBOT_OLD_THR_LIST };
static __constant__ float   TURBOT_D_YOUNG_LUT [GGML_TURBOT_YOUNG_TOTAL] = { GGML_TURBOT_YOUNG_LUT_LIST };
static __constant__ float   TURBOT_D_YOUNG_THR [GGML_TURBOT_YOUNG_TOTAL] = { GGML_TURBOT_YOUNG_THR_LIST };
static __constant__ uint8_t TURBOT_D_FILL_CODE [GGML_TURBOT_FILL_TOTAL]  = { GGML_TURBOT_FILL_LIST };

// Same closed forms as ggml_turbot_old_off / ggml_turbot_young_off / ggml_turbot_fill_off in ggml-turbot.h.
static __device__ __forceinline__ int turbot_d_old_off(int b) {
    return (1 << b) - 4;
}

static __device__ __forceinline__ int turbot_d_young_off(int b, int y) {
    return (b - 2)*512 + 8 + (1 << y) - (1 << (b + 2));
}

static __device__ __forceinline__ int turbot_d_fill_off(int b, int y) {
    const int base = b <= 2 ? 0 : b == 3 ? 24 : b == 4 ? 64 : b == 5 ? 128 : 224;
    return base + (y - b - 1)*(1 << b);
}
