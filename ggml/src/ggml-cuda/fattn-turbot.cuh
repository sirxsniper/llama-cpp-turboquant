#pragma once

// [TAG_TURBOT] CUDA flash attention read path of the turbot tiered KV cache, docs/turbot/SPEC.md section 7.
//
// Everything turbot-specific on the device lives here and is compiled ONLY into the 20 hand-written
// template-instances/fattn-mma-turbot-instance-ncols1_*-ncols2_*.cu TUs. No existing kernel, loader, template or
// instance TU changes or includes this file, so every existing KV type keeps identical codegen and constant memory.
//
// The kernel is a specialised copy of flash_attn_ext_f16 (fattn-mma-f16.cuh) for D = 256:
//   - nstages = 0, V_is_K_view = false, Q always in registers (every D = 256 config has Q_in_reg);
//   - the K/V tile loaders read the turbot base row, and for young granules the refinement from the young pool;
//   - softmax, mask, sinks, softcap, combine and fixup code is the f16 kernel's, copied verbatim.
// launch_fattn_turbot is launch_fattn without the F16 conversion, with the same KV_max scans and fixup launches, except
// that fixup layouts are work-balanced by default ([TAG_TURBOT_FA_BALANCE], SPEC section 13).

#include "common.cuh"
#include "fattn-mma-f16.cuh"
#include "turbot-tables.cuh"
#include "ggml-turbot.h"

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <tuple>

// [TAG_TURBOT_I8] OLD read representation for b <= 5 (SPEC 7.4): 1 = int8 register LUTs (the trade turbo4p and turbo5p
// ship, x1.000 / x1.001 / x1.003 / x1.014 old error for b = 2..5), 0 = float LUT runs in shared memory (the measured
// quality). b = 6 and every YOUNG read always use the float runs. Gate B0 builds and measures both.
#ifndef GGML_CUDA_TURBOT_OLD_I8
#define GGML_CUDA_TURBOT_OLD_I8 1
#endif // GGML_CUDA_TURBOT_OLD_I8

// ------------------------------------------------------------------------------------------------------------------
// [TAG_TURBOT_I8] int8 register LUTs of the old codebooks.
//
// Entry k of each table is ggml_turbot_old_level_i8(b, k) = round(C_b[k] / max|C_b| * 127), k in the turbot index
// order (ascending levels, k = number of thresholds <= u), and the matching scale is ggml_turbot_old_i8_scale(b).
// Derived here at compile time from the same generated list the host arrays use. For b = 4 and b = 5 the result IS
// the existing turbo4_int8_lut / turbo5_int8_lut (fattn-common.cuh) word for word and the scales are
// TURBO_INT8_4BIT/5BIT_SCALE_REVERSE bit for bit, static_asserted below, so those two structs are reused unchanged.
// b = 2 and b = 3 need at most 8 entries, i.e. one __byte_perm over the 8-byte pool {w1:w0} and no blend.

static constexpr float TURBOT_OLD_LEVELS_CE[GGML_TURBOT_OLD_TOTAL] = { GGML_TURBOT_OLD_LEVELS_LIST };

static constexpr int8_t turbot_old_i8_ce(const int b, const int j) {
    const int   off = (1 << b) - 4;   // ggml_turbot_old_off
    const float q   = TURBOT_OLD_LEVELS_CE[off + j] / -TURBOT_OLD_LEVELS_CE[off] * 127.0f;
    return (int8_t) (q >= 0.0f ? (int) (q + 0.5f) : -(int) (-q + 0.5f));   // roundf, half away from zero
}

// entries 4w..4w+3 of codebook b in the byte order turbo4_int8_lut::gather4 expects, 0 past the end of the codebook
static constexpr uint32_t turbot_old_i8_pack(const int b, const int w) {
    uint32_t v = 0;
    for (int k = 0; k < 4; ++k) {
        const int j = 4*w + k;
        if (j < (1 << b)) {
            v |= (uint32_t) (uint8_t) turbot_old_i8_ce(b, j) << (8*k);
        }
    }
    return v;
}

static constexpr uint32_t TURBOT_I8_B2_W0    = turbot_old_i8_pack(2, 0);
static constexpr uint32_t TURBOT_I8_B3_W0    = turbot_old_i8_pack(3, 0);
static constexpr uint32_t TURBOT_I8_B3_W1    = turbot_old_i8_pack(3, 1);
static constexpr float    TURBOT_I8_B2_SCALE = -TURBOT_OLD_LEVELS_CE[0] / 127.0f;   // ggml_turbot_old_i8_scale(2)
static constexpr float    TURBOT_I8_B3_SCALE = -TURBOT_OLD_LEVELS_CE[4] / 127.0f;   // ggml_turbot_old_i8_scale(3)

// Checked against the generated tables on the host (float32 arithmetic): b2 {-127, -38, 38, 127},
// b3 {-127, -79, -45, -14, 14, 45, 79, 127}. A regenerated codebook that changes them fails here, not silently.
static_assert(TURBOT_I8_B2_W0 == 0x7F26DA81u, "turbot b2 int8 LUT changed");
static_assert(TURBOT_I8_B3_W0 == 0xF2D3B181u, "turbot b3 int8 LUT w0 changed");
static_assert(TURBOT_I8_B3_W1 == 0x7F4F2D0Eu, "turbot b3 int8 LUT w1 changed");
static_assert(turbot_old_i8_pack(4, 0) == TURBO_C4_LUT_W0 && turbot_old_i8_pack(4, 1) == TURBO_C4_LUT_W1 &&
              turbot_old_i8_pack(4, 2) == TURBO_C4_LUT_W2 && turbot_old_i8_pack(4, 3) == TURBO_C4_LUT_W3,
              "turbot C4 int8 LUT is no longer turbo4_int8_lut");
static_assert(turbot_old_i8_pack(5, 0) == TURBO_C5_LUT_W0 && turbot_old_i8_pack(5, 1) == TURBO_C5_LUT_W1 &&
              turbot_old_i8_pack(5, 2) == TURBO_C5_LUT_W2 && turbot_old_i8_pack(5, 3) == TURBO_C5_LUT_W3 &&
              turbot_old_i8_pack(5, 4) == TURBO_C5_LUT_W4 && turbot_old_i8_pack(5, 5) == TURBO_C5_LUT_W5 &&
              turbot_old_i8_pack(5, 6) == TURBO_C5_LUT_W6 && turbot_old_i8_pack(5, 7) == TURBO_C5_LUT_W7,
              "turbot C5 int8 LUT is no longer turbo5_int8_lut");
static_assert(-TURBOT_OLD_LEVELS_CE[12] / 127.0f == TURBO_INT8_4BIT_SCALE_REVERSE, "turbot C4 int8 scale changed");
static_assert(-TURBOT_OLD_LEVELS_CE[28] / 127.0f == TURBO_INT8_5BIT_SCALE_REVERSE, "turbot C5 int8 scale changed");

// ------------------------------------------------------------------------------------------------------------------
// Shared-memory LUT area (SPEC 7.4, 7.5).
//
// The float LUT runs one call needs (the current KV head's K and V codebooks) are copied from turbot-tables.cuh into
// shared memory once per output tile, so the per-element data-dependent gathers read shared memory, never constant
// memory. The area starts right after the mask tile: lut_off bytes from tile_Q (== tile_K, Q is in registers).

constexpr int TURBOT_LUT_K_OLD   =   0;   // C_b of the K head,       <= 64 floats
constexpr int TURBOT_LUT_K_YOUNG =  64;   // LUT_{b,y} of the K head, <= 256 floats
constexpr int TURBOT_LUT_V_OLD   = 320;
constexpr int TURBOT_LUT_V_YOUNG = 384;
constexpr size_t TURBOT_NBYTES_SHARED_LUT = GGML_PAD(2*(64 + 256 + 64)*sizeof(float), 16);   // 3072 B, SPEC 7.4
static_assert((TURBOT_LUT_V_YOUNG + 256)*sizeof(float) <= TURBOT_NBYTES_SHARED_LUT, "turbot LUT area too small");

// Same formula on the host (sizing) and the device (placement): the KV tile, then the mask tile, padded to 16.
static constexpr __host__ __device__ int ggml_cuda_fattn_turbot_lut_off(const int nbatch_fa, const int nbatch_K2, const int nbatch_V2, const int ncols1) {
    return GGML_PAD(nbatch_fa * (nbatch_K2 > nbatch_V2 ? nbatch_K2 + 4 : nbatch_V2 + 4) * (int) sizeof(half2) +
                    ncols1    * (nbatch_fa/2 + 4)                                          * (int) sizeof(half2), 16);
}

// ------------------------------------------------------------------------------------------------------------------
// Layout of one KV head, per side. Passed to the kernel as three int64 register arguments, never __device__ globals
// (a host write to a global races with kernels still queued for earlier layers).
//
//   desc0: side K bits 0..31, side V bits 32..63; per side: b[h] at 4h, y[h] at 16 + 4h (4 bits each)
//   desc1: side K offsets in units of 32 B: base_off[h] at 5h, young_off[h] at 20 + 5h, base_gain_off at 40,
//          young_gain_off at 45 (5 bits each; every value is <= 24)
//   desc2: side V offsets, same layout, plus pool_v_off / 16 at bit 50 (6 bits)

struct turbot_planes_state {
    int has4;   // the plane is present (0/1), ggml_turbot_planes_of
    int has2;
    int has1;
    int p4;     // byte offset of the plane from the row base (base row, or pool row with the side part included)
    int p2;
    int p1;
    int sh2;    // code bit of the P2 field
    int sh1;    // code bit of the P1 bit
};

struct turbot_side_state {
    int b;                       // old width
    int r;                       // refinement width y - b
    turbot_planes_state base;    // planes of the old code, width b, in the base row
    turbot_planes_state pool;    // planes of the refinement, width r, in the pool row
    int base_gain;               // gain of (head, group 0) in the base row: base_gain_off + 4h
    int pool_gain;               // gain of (head, group 0) in the pool row: side part + young_gain_off + 4h
    int old_run;                 // C_b     in TURBOT_D_OLD_LEVELS
    int young_run;               // LUT_b,y in TURBOT_D_YOUNG_LUT
};

struct turbot_head_state {
    turbot_side_state k;
    turbot_side_state v;
    int pool_v_off;              // V part of a pool row
    int generic;                 // [TAG_TURBOT_CT_WIDTH] force the runtime-width loaders (LLAMA_TURBOT_FA_GENERIC=1)
};

static __device__ __forceinline__ turbot_planes_state turbot_planes_of(const int w, const int run) {
    turbot_planes_state p;
    p.has4 = w >= 4 ? 1 : 0;
    p.has2 = (w & 2) ? 1 : 0;
    p.has1 = (w & 1) ? 1 : 0;
    p.p4   = run;
    p.p2   = run + (p.has4 ? 128 : 0);
    p.p1   = run + (p.has4 ? 128 : 0) + (p.has2 ? 64 : 0);
    p.sh2  = p.has4 ? 4 : 0;
    p.sh1  = p.sh2 + (p.has2 ? 2 : 0);
    return p;
}

// [TAG_TURBOT_CT_WIDTH] Compile-time twin of turbot_planes_of (SPEC 7.4, read-speed plan item 4).
//
// A tile's width is uniform across the CUDA block (one head, one side), so the plane presence, the plane offsets and
// the code shifts can be constants instead of per-chunk register reads and compare chains. These give the shape of a
// code of width w; the plane BASE offset stays runtime (it is the per-head run, which only the descriptor knows).
static constexpr __host__ __device__ int turbot_ct_has4(const int w) { return w >= 4 ? 1 : 0; }
static constexpr __host__ __device__ int turbot_ct_has2(const int w) { return (w & 2) ? 1 : 0; }
static constexpr __host__ __device__ int turbot_ct_has1(const int w) { return (w & 1) ? 1 : 0; }
static constexpr __host__ __device__ int turbot_ct_d2  (const int w) { return turbot_ct_has4(w) ? 128 : 0; }
static constexpr __host__ __device__ int turbot_ct_d1  (const int w) { return turbot_ct_d2(w) + (turbot_ct_has2(w) ? 64 : 0); }
static constexpr __host__ __device__ int turbot_ct_sh2 (const int w) { return turbot_ct_has4(w) ? 4 : 0; }
static constexpr __host__ __device__ int turbot_ct_sh1 (const int w) { return turbot_ct_sh2(w) + (turbot_ct_has2(w) ? 2 : 0); }

// The plane state a loader should use: for W >= 0 the shape is folded to constants and only the run offset (rt.p4,
// which IS the run in turbot_planes_of) stays runtime, so the result is bit-identical to the runtime state; for
// W < 0 the runtime state is returned unchanged (the generic fallback).
template <int W>
static __device__ __forceinline__ turbot_planes_state turbot_planes_eff(const turbot_planes_state & rt) {
    if constexpr (W >= 0) {
        turbot_planes_state p;
        p.has4 = turbot_ct_has4(W);
        p.has2 = turbot_ct_has2(W);
        p.has1 = turbot_ct_has1(W);
        p.p4   = rt.p4;
        p.p2   = rt.p4 + turbot_ct_d2(W);
        p.p1   = rt.p4 + turbot_ct_d1(W);
        p.sh2  = turbot_ct_sh2(W);
        p.sh1  = turbot_ct_sh1(W);
        return p;
    } else {
        return rt;
    }
}

static __device__ __forceinline__ turbot_side_state turbot_side_state_of(const int h, const uint64_t widths, const uint64_t offs, const int part_off) {
    const int b              = (int) ((widths >> (4*h))      & 0xFu);
    const int y              = (int) ((widths >> (16 + 4*h)) & 0xFu);
    const int base_off       = 32*(int) ((offs >> (5*h))      & 0x1Fu);
    const int young_off      = 32*(int) ((offs >> (20 + 5*h)) & 0x1Fu);
    const int base_gain_off  = 32*(int) ((offs >> 40)         & 0x1Fu);
    const int young_gain_off = 32*(int) ((offs >> 45)         & 0x1Fu);

    turbot_side_state st;
    st.b         = b;
    st.r         = y - b;
    st.base      = turbot_planes_of(b,     base_off);
    st.pool      = turbot_planes_of(y - b, part_off + young_off);
    st.base_gain = base_gain_off + 4*h;
    st.pool_gain = part_off + young_gain_off + 4*h;
    st.old_run   = turbot_d_old_off(b);
    st.young_run = turbot_d_young_off(b, y);
    return st;
}

// [TAG_TURBOT_HEAD_STATE] The ONLY place a turbot kernel derives per-head addressing. flash_attn_ext_turbot calls it at
// exactly two sites, the tile loop and the final is_fixup block (the two copies of that code in flash_attn_ext_f16
// drifted, [TAG_TURBO4P_HEAD] vs TURBO5P512). Both call sites carry the tag.
static __device__ __forceinline__ turbot_head_state turbot_head_state_of(int z_KV, int64_t desc0, int64_t desc1, int64_t desc2) {
    const uint64_t d0 = (uint64_t) desc0;
    const uint64_t d1 = (uint64_t) desc1;
    const uint64_t d2 = (uint64_t) desc2;

    turbot_head_state hs;
    hs.pool_v_off = 16*(int) ((d2 >> 50) & 0x3Fu);
    hs.generic    =     (int) ((d1 >> 50) & 0x1u);   // [TAG_TURBOT_CT_WIDTH] spare desc1 bit, see pack_desc
    hs.k = turbot_side_state_of(z_KV,  d0        & 0xFFFFFFFFu, d1, 0);
    hs.v = turbot_side_state_of(z_KV, (d0 >> 32) & 0xFFFFFFFFu, d2, hs.pool_v_off);
    return hs;
}

// Code of chunk element e (0..31) from the plane words of one 32-element chunk; absent planes are loaded as 0.
// P4: 16 bytes -> 4 words, element e at bits 4*(e%8) of word e/8. P2: 8 bytes -> 2 words, bits 2*(e%16) of word e/16.
// P1: 4 bytes -> 1 word, bit e. Same bit assembly as ggml_turbot_get_code.
static __device__ __forceinline__ unsigned turbot_code(
        const uint32_t p4w, const uint32_t p2w, const uint32_t p1w, const int e, const turbot_planes_state & ps) {
    return ((p4w >> (4*(e & 7))) & 0xFu) | (((p2w >> (2*(e & 15))) & 0x3u) << ps.sh2) | (((p1w >> e) & 0x1u) << ps.sh1);
}

// ------------------------------------------------------------------------------------------------------------------
// Tile loaders. Same chunking and lane layout as flash_attn_ext_turbo5p_load_tile: 32 elements = 16 half2 per chunk.
//
// Alignment: every turbot row is 16 | 32S + 16 bytes from a 128-byte aligned buffer, every run and plane starts at a
// multiple of 32 from the row base, and a chunk's element offset e0 is a multiple of 32. So the P4 load at e0/2 is
// 16-byte aligned (one 16-byte load), P2 at e0/4 is 8-byte aligned, P1 at e0/8 4-byte aligned. A chunk never
// crosses a WHT group, so its single gain is gain[2h + e0/128]. Rows of a tile are consecutive cells of one granule.

// OLD: code j of width b from the base row; v = C_b[j] * gain_b.
// [TAG_TURBOT_CT_WIDTH] B >= 0 compiles the loader for exactly that width (plane presence, offsets, shifts and the
// gather path all constant); B < 0 keeps the runtime-width body. Both compute the same values.
template<int B, int stride_tile, int nwarps, int nbatch_fa, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_turbot_load_tile_old(
        const char * const __restrict__ row0,       // base row of the tile's first cell
        half2 * const __restrict__ tile_KV,
        const int D2,                                // half2 per row to produce, a multiple of 16
        const int64_t stride_row,                    // base row bytes
        const int elem0,                             // first element inside the head, a multiple of 32
        const turbot_side_state & st,
        const float * const __restrict__ lut_old,    // shared copy of C_b
        const int i_sup) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr int h2_per_chunk   = 16;
    constexpr int elem_per_chunk = 2*h2_per_chunk;
    const int chunks_per_row = D2 / h2_per_chunk;

    constexpr bool old_i8 = GGML_CUDA_TURBOT_OLD_I8 != 0;

    // [TAG_TURBOT_CT_WIDTH] width and plane shape, constant for B >= 0.
    const turbot_planes_state base = turbot_planes_eff<B>(st.base);
    const int                 bb   = B >= 0 ? B : st.b;

    auto load = [&] __device__ (const int n) {
        const int stride_k = warp_size >> n;
        const int k0_start = stride_k == warp_size ? 0 : chunks_per_row - chunks_per_row % (2*stride_k);
        const int k0_stop  =                             chunks_per_row - chunks_per_row % (1*stride_k);
        const int stride_i = warp_size / stride_k;

        if (k0_start == k0_stop) {
            return;
        }

#pragma unroll
        for (int i0 = 0; i0 < nbatch_fa; i0 += nwarps*stride_i) {
            const int i = i0 + threadIdx.y*stride_i + (stride_k == warp_size ? 0 : threadIdx.x / stride_k);

            if (i0 + nwarps*stride_i > nbatch_fa && i >= nbatch_fa) {
                break;
            }

#pragma unroll
            for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                __align__(16) half2 out[h2_per_chunk];

                if (oob_check && i >= i_sup) {
#pragma unroll
                    for (int e = 0; e < h2_per_chunk; ++e) {
                        out[e] = make_half2(0.0f, 0.0f);
                    }
                } else {
                    const int e0 = elem0 + k*elem_per_chunk;
                    const char * __restrict__ row = row0 + (size_t) i * (size_t) stride_row;

                    // Width-dependent plane presence is uniform across the CUDA block (one head, one side).
                    __align__(16) uint32_t p4[4] = {0u, 0u, 0u, 0u};
                    __align__(16) uint32_t p2[2] = {0u, 0u};
                    uint32_t p1 = 0u;
                    if (base.has4) {
                        ggml_cuda_memcpy_1<16>(p4, row + base.p4 + (e0 >> 1));
                    }
                    if (base.has2) {
                        ggml_cuda_memcpy_1<8>(p2, row + base.p2 + (e0 >> 2));
                    }
                    if (base.has1) {
                        ggml_cuda_memcpy_1<4>(&p1, row + base.p1 + (e0 >> 3));
                    }
                    const float gain = __half2float(((const ggml_half *) (row + st.base_gain))[e0 >> 7]);

                    if (old_i8 && bb == 4) {
                        // P4 only: exactly the turbo4p input, [TAG_TURBO4P_CENT].
                        turbo4_int8_lut lut;
                        lut.init();
                        const float nscale = gain * TURBO_INT8_4BIT_SCALE_REVERSE;
#pragma unroll
                        for (int w = 0; w < 4; ++w) {
                            const int g0 = (int) lut.gather4( p4[w]        & 0xFFFFu);   // elements 8w+0..3
                            const int g1 = (int) lut.gather4((p4[w] >> 16) & 0xFFFFu);   // elements 8w+4..7

                            out[4*w + 0] = make_half2((float) (int8_t) (g0      ) * nscale, (float) (int8_t) (g0 >>  8) * nscale);
                            out[4*w + 1] = make_half2((float) (int8_t) (g0 >> 16) * nscale, (float) (int8_t) (g0 >> 24) * nscale);
                            out[4*w + 2] = make_half2((float) (int8_t) (g1      ) * nscale, (float) (int8_t) (g1 >>  8) * nscale);
                            out[4*w + 3] = make_half2((float) (int8_t) (g1 >> 16) * nscale, (float) (int8_t) (g1 >> 24) * nscale);
                        }
                    } else if (old_i8 && bb == 5) {
                        // P4 then P1: exactly the (qs, qh) pair of flash_attn_ext_turbo5p_load_tile.
                        turbo5_int8_lut lut;
                        lut.init();
                        const float nscale = gain * TURBO_INT8_5BIT_SCALE_REVERSE;
#pragma unroll
                        for (int w = 0; w < 4; ++w) {
                            const int g0 = (int) lut.gather4( p4[w]        & 0xFFFFu, (p1 >> (8*w    )) & 0xFu);   // elements 8w+0..3
                            const int g1 = (int) lut.gather4((p4[w] >> 16) & 0xFFFFu, (p1 >> (8*w + 4)) & 0xFu);   // elements 8w+4..7

                            out[4*w + 0] = make_half2((float) (int8_t) (g0      ) * nscale, (float) (int8_t) (g0 >>  8) * nscale);
                            out[4*w + 1] = make_half2((float) (int8_t) (g0 >> 16) * nscale, (float) (int8_t) (g0 >> 24) * nscale);
                            out[4*w + 2] = make_half2((float) (int8_t) (g1      ) * nscale, (float) (int8_t) (g1 >>  8) * nscale);
                            out[4*w + 3] = make_half2((float) (int8_t) (g1 >> 16) * nscale, (float) (int8_t) (g1 >> 24) * nscale);
                        }
                    } else if (old_i8 && bb <= 3) {
                        // P2 (then P1 for b = 3): indices < 8, one __byte_perm per 4 elements, no blend.
                        // (casts: immediates, never a reference to the constexpr objects)
                        const uint32_t pw0    = bb == 2 ? (uint32_t) TURBOT_I8_B2_W0 : (uint32_t) TURBOT_I8_B3_W0;
                        const uint32_t pw1    = bb == 2 ? 0u                         : (uint32_t) TURBOT_I8_B3_W1;
                        const float    nscale = gain * (bb == 2 ? (float) TURBOT_I8_B2_SCALE : (float) TURBOT_I8_B3_SCALE);
#pragma unroll
                        for (int m = 0; m < 8; ++m) {
                            // byte m of the P2 plane holds the 2-bit fields of elements 4m..4m+3; spread to nibbles
                            uint32_t nib = (p2[m >> 2] >> (8*(m & 3))) & 0xFFu;
                            nib = (nib | (nib << 4)) & 0x0F0Fu;
                            nib = (nib | (nib << 2)) & 0x3333u;
                            // their P1 bits (all 0 for b = 2) to bit 2 of each nibble
                            uint32_t hib = (p1 >> (4*m)) & 0xFu;
                            hib = (hib | (hib << 6)) & 0x0303u;
                            hib = (hib | (hib << 3)) & 0x1111u;
                            const int g = (int) __byte_perm(pw0, pw1, nib | (hib << 2));

                            out[2*m + 0] = make_half2((float) (int8_t) (g      ) * nscale, (float) (int8_t) (g >>  8) * nscale);
                            out[2*m + 1] = make_half2((float) (int8_t) (g >> 16) * nscale, (float) (int8_t) (g >> 24) * nscale);
                        }
                    } else {
                        // float run in shared memory: b = 6 always, every b when GGML_CUDA_TURBOT_OLD_I8 is 0
#pragma unroll
                        for (int p = 0; p < h2_per_chunk; ++p) {
                            const unsigned c0 = turbot_code(p4[(2*p    ) >> 3], p2[(2*p    ) >> 4], p1, 2*p,     base);
                            const unsigned c1 = turbot_code(p4[(2*p + 1) >> 3], p2[(2*p + 1) >> 4], p1, 2*p + 1, base);
                            out[p] = make_half2(lut_old[c0] * gain, lut_old[c1] * gain);
                        }
                    }
                }

                // 64 bytes out, in the 16-byte units the helper can emit.
#pragma unroll
                for (int c = 0; c < h2_per_chunk/4; ++c) {
                    ggml_cuda_memcpy_1<16>(tile_KV + i*stride_tile + k*h2_per_chunk + 4*c, out + 4*c);
                }
            }
        }
    };
    ggml_cuda_unroll<6>{}(load);
}

// YOUNG: code j of width b from the base row, refinement s of width r from the pool row;
// v = LUT_{b,y}[(j << r) | s] * gain_y. Always a float run: an int8 young LUT or an int8 base level is forbidden (SPEC 4.3).
// [TAG_TURBOT_CT_WIDTH] B/R >= 0 compile the loader for exactly that (b, r); B < 0 keeps the runtime-width body.
template<int B, int R, int stride_tile, int nwarps, int nbatch_fa, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_turbot_load_tile_young(
        const char * const __restrict__ row0,       // base row of the tile's first cell
        const char * const __restrict__ prow0,      // pool row of the tile's first cell
        half2 * const __restrict__ tile_KV,
        const int D2,                                // half2 per row to produce, a multiple of 16
        const int64_t stride_row,                    // base row bytes
        const int64_t stride_prow,                   // pool row bytes
        const int elem0,                             // first element inside the head, a multiple of 32
        const turbot_side_state & st,
        const float * const __restrict__ lut_young,  // shared copy of LUT_{b,y}
        const int i_sup) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr int h2_per_chunk   = 16;
    constexpr int elem_per_chunk = 2*h2_per_chunk;
    const int chunks_per_row = D2 / h2_per_chunk;

    // [TAG_TURBOT_CT_WIDTH] width and plane shapes, constant for B/R >= 0.
    const turbot_planes_state base = turbot_planes_eff<B>(st.base);
    const turbot_planes_state pool = turbot_planes_eff<R>(st.pool);
    const int                 rr   = R >= 0 ? R : st.r;

    auto load = [&] __device__ (const int n) {
        const int stride_k = warp_size >> n;
        const int k0_start = stride_k == warp_size ? 0 : chunks_per_row - chunks_per_row % (2*stride_k);
        const int k0_stop  =                             chunks_per_row - chunks_per_row % (1*stride_k);
        const int stride_i = warp_size / stride_k;

        if (k0_start == k0_stop) {
            return;
        }

#pragma unroll
        for (int i0 = 0; i0 < nbatch_fa; i0 += nwarps*stride_i) {
            const int i = i0 + threadIdx.y*stride_i + (stride_k == warp_size ? 0 : threadIdx.x / stride_k);

            if (i0 + nwarps*stride_i > nbatch_fa && i >= nbatch_fa) {
                break;
            }

#pragma unroll
            for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                __align__(16) half2 out[h2_per_chunk];

                if (oob_check && i >= i_sup) {
#pragma unroll
                    for (int e = 0; e < h2_per_chunk; ++e) {
                        out[e] = make_half2(0.0f, 0.0f);
                    }
                } else {
                    const int e0 = elem0 + k*elem_per_chunk;
                    const char * __restrict__ row  = row0  + (size_t) i * (size_t) stride_row;
                    const char * __restrict__ prow = prow0 + (size_t) i * (size_t) stride_prow;

                    __align__(16) uint32_t b4[4] = {0u, 0u, 0u, 0u};
                    __align__(16) uint32_t b2[2] = {0u, 0u};
                    uint32_t b1 = 0u;
                    if (base.has4) {
                        ggml_cuda_memcpy_1<16>(b4, row + base.p4 + (e0 >> 1));
                    }
                    if (base.has2) {
                        ggml_cuda_memcpy_1<8>(b2, row + base.p2 + (e0 >> 2));
                    }
                    if (base.has1) {
                        ggml_cuda_memcpy_1<4>(&b1, row + base.p1 + (e0 >> 3));
                    }

                    __align__(16) uint32_t r4[4] = {0u, 0u, 0u, 0u};
                    __align__(16) uint32_t r2[2] = {0u, 0u};
                    uint32_t r1 = 0u;
                    if (pool.has4) {
                        ggml_cuda_memcpy_1<16>(r4, prow + pool.p4 + (e0 >> 1));
                    }
                    if (pool.has2) {
                        ggml_cuda_memcpy_1<8>(r2, prow + pool.p2 + (e0 >> 2));
                    }
                    if (pool.has1) {
                        ggml_cuda_memcpy_1<4>(&r1, prow + pool.p1 + (e0 >> 3));
                    }

                    const float gain = __half2float(((const ggml_half *) (prow + st.pool_gain))[e0 >> 7]);

#pragma unroll
                    for (int p = 0; p < h2_per_chunk; ++p) {
                        const int      e_0  = 2*p;
                        const int      e_1  = 2*p + 1;
                        const unsigned idx0 = (turbot_code(b4[e_0 >> 3], b2[e_0 >> 4], b1, e_0, base) << rr) |
                                               turbot_code(r4[e_0 >> 3], r2[e_0 >> 4], r1, e_0, pool);
                        const unsigned idx1 = (turbot_code(b4[e_1 >> 3], b2[e_1 >> 4], b1, e_1, base) << rr) |
                                               turbot_code(r4[e_1 >> 3], r2[e_1 >> 4], r1, e_1, pool);
                        out[p] = make_half2(lut_young[idx0] * gain, lut_young[idx1] * gain);
                    }
                }

                // 64 bytes out, in the 16-byte units the helper can emit.
#pragma unroll
                for (int c = 0; c < h2_per_chunk/4; ++c) {
                    ggml_cuda_memcpy_1<16>(tile_KV + i*stride_tile + k*h2_per_chunk + 4*c, out + 4*c);
                }
            }
        }
    };
    ggml_cuda_unroll<6>{}(load);
}

// ------------------------------------------------------------------------------------------------------------------
// [TAG_TURBOT_CT_WIDTH] Width dispatch (read-speed plan item 4).
//
// One lane-uniform switch per tile, outside the per-chunk unroll: the width of a tile is uniform across the CUDA block
// (one head, one side), so the whole loader can be compiled for that width instead of re-testing plane presence and
// the gather path on every chunk. Old widths are 2..6 (SPEC 4.3), so the switch is total and the generic body is
// reachable only through hs.generic. Young is specialised for y = 7, the deployed plan's refinement width; every other
// y (the MIXED test carries y = 8 and y = 5) keeps the runtime body. Both bodies compute identical values, so
// LLAMA_TURBOT_FA_GENERIC=1 must be bit-identical to the default.
//
// [TAG_TURBOT_CT_SPILL] The specialisation is compiled only into kernel configs with ncols >= 32 (ct_width, see
// TURBOT_CT_WIDTH_MIN_NCOLS). At ncols 16 the inlined width switch pushed <2,8>, <4,4> and <8,2> to 336-704 B of
// STACK (cuobjdump -res-usage, E:/kv-turbot/fix3/cuobj/post_resusage.txt), and <2,8> is the Q = 2 DFlash2 batch
// kernel; ncols <= 16 configs therefore keep the runtime-width loaders (round-1 codegen) and ignore hs.generic.
// [TAG_TURBOT_Q2_ROUTE] Q = 2 now runs <4,8> (ncols 32, compiled widths) by default; <2,8> only with TURBOT_Q2_ROUTE=0.
static constexpr int TURBOT_CT_WIDTH_MIN_NCOLS = 32;

template<bool ct_width, int stride_tile, int nwarps, int nbatch_fa, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_turbot_load_tile_old_dispatch(
        const int generic,
        const char * const __restrict__ row0,
        half2 * const __restrict__ tile_KV,
        const int D2,
        const int64_t stride_row,
        const int elem0,
        const turbot_side_state & st,
        const float * const __restrict__ lut_old,
        const int i_sup) {
#define TURBOT_OLD_CASE(BB)                                                                       \
    case BB: flash_attn_ext_turbot_load_tile_old<BB, stride_tile, nwarps, nbatch_fa, oob_check>   \
        (row0, tile_KV, D2, stride_row, elem0, st, lut_old, i_sup); return;

    if constexpr (ct_width) {
        if (!generic) {
            switch (st.b) {
                TURBOT_OLD_CASE(2)
                TURBOT_OLD_CASE(3)
                TURBOT_OLD_CASE(4)
                TURBOT_OLD_CASE(5)
                TURBOT_OLD_CASE(6)
                default: break;
            }
        }
    } else {
        (void) generic;   // [TAG_TURBOT_CT_SPILL] ncols <= 16: runtime loader only
    }
#undef TURBOT_OLD_CASE
    flash_attn_ext_turbot_load_tile_old<-1, stride_tile, nwarps, nbatch_fa, oob_check>
        (row0, tile_KV, D2, stride_row, elem0, st, lut_old, i_sup);
}

template<bool ct_width, int stride_tile, int nwarps, int nbatch_fa, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_turbot_load_tile_young_dispatch(
        const int generic,
        const char * const __restrict__ row0,
        const char * const __restrict__ prow0,
        half2 * const __restrict__ tile_KV,
        const int D2,
        const int64_t stride_row,
        const int64_t stride_prow,
        const int elem0,
        const turbot_side_state & st,
        const float * const __restrict__ lut_young,
        const int i_sup) {
#define TURBOT_YOUNG_CASE(BB)                                                                               \
    case BB: flash_attn_ext_turbot_load_tile_young<BB, 7 - BB, stride_tile, nwarps, nbatch_fa, oob_check>   \
        (row0, prow0, tile_KV, D2, stride_row, stride_prow, elem0, st, lut_young, i_sup); return;

    if constexpr (ct_width) {
        if (!generic && st.b + st.r == 7) {
            switch (st.b) {
                TURBOT_YOUNG_CASE(2)
                TURBOT_YOUNG_CASE(3)
                TURBOT_YOUNG_CASE(4)
                TURBOT_YOUNG_CASE(5)
                TURBOT_YOUNG_CASE(6)
                default: break;
            }
        }
    } else {
        (void) generic;   // [TAG_TURBOT_CT_SPILL] ncols <= 16: runtime loader only
    }
#undef TURBOT_YOUNG_CASE
    flash_attn_ext_turbot_load_tile_young<-1, -1, stride_tile, nwarps, nbatch_fa, oob_check>
        (row0, prow0, tile_KV, D2, stride_row, stride_prow, elem0, st, lut_young, i_sup);
}

// ------------------------------------------------------------------------------------------------------------------
// Copy of flash_attn_ext_f16_iter for turbot: nstages = 0, V_is_K_view = false, Q in registers.

template<int DKQ, int DV, int ncols1, int ncols2, int nwarps,
    bool use_logit_softcap, bool needs_fixup, bool is_fixup, bool last_iter, bool oob_check,
    typename T_A_KQ, typename T_B_KQ, typename T_C_KQ, typename T_A_VKQ, typename T_B_VKQ, typename T_C_VKQ>
static __device__ __forceinline__ void flash_attn_ext_turbot_iter(
        const float2 * const __restrict__ Q_f2,
        const char   * const __restrict__ K_row,     // [TAG_TURBOT] base row of cell 0 (row base, not a head base)
        const char   * const __restrict__ V_row,
        const char   * const __restrict__ pool,      // young pool of the layer
        const int32_t * const __restrict__ gtab,     // granule table
        const half   * const __restrict__ mask_h,
        const int32_t * const __restrict__ kv_pos,   // [TAG_FA_POS_MASK]
        const int32_t * const __restrict__ q_pos,
        float2       * const __restrict__ dstk,
        float2       * const __restrict__ dstk_fixup,
        const float scale,
        const float slope,
        const float logit_softcap,
        const uint3 ne01,
        const int ne02,
        const int64_t nb11,                          // base row bytes, K
        const int64_t nb21,                          // base row bytes, V
        const int64_t nb_pool,                       // pool row bytes
        const int stride_mask,
        half2        * const __restrict__ tile_Q,
        half2        * const __restrict__ tile_K,
        half2        * const __restrict__ tile_V,
        half         * const __restrict__ tile_mask,
        const float  * const __restrict__ lut,       // shared LUT area, TURBOT_LUT_*
        T_B_KQ       * const __restrict__ Q_B,
        T_C_VKQ      * const __restrict__ VKQ_C,
        float        * const __restrict__ KQ_max,
        float        * const __restrict__ KQ_rowsum,
        const int jt,
        const int kb0,
        const int k_VKQ_sup,
        const turbot_head_state & hs) {
#if defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    constexpr int  warp_size       = ggml_cuda_get_physical_warp_size();
    constexpr int  ncols           = ncols1 * ncols2;
    constexpr int  cols_per_warp   = T_B_KQ::I;
    constexpr int  cols_per_thread = get_cols_per_thread();
    constexpr int  np              = cols_per_warp > ncols ? nwarps : nwarps * cols_per_warp/ncols; // Number of parallel CUDA warps per Q column.
    constexpr int  nbatch_fa       = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, ncols);
    constexpr int  nbatch_K2       = ggml_cuda_fattn_mma_get_nbatch_K2(DKQ, DV, ncols);
    constexpr int  nbatch_V2       = ggml_cuda_fattn_mma_get_nbatch_V2(DKQ, DV, ncols);
    constexpr bool Q_in_reg        = ggml_cuda_fattn_mma_get_Q_in_reg (DKQ, DV, ncols);

    static_assert(Q_in_reg, "turbot: every D=256 config keeps Q in registers, so tile_K == tile_Q");
    static_assert(GGML_TURBOT_GRANULE % nbatch_fa == 0, "turbot: a KV tile must lie inside one granule");
    static_assert(nbatch_K2 % 16 == 0 && nbatch_V2 % 16 == 0, "turbot tile loaders need 32-element chunks");

    constexpr int stride_tile_K = nbatch_K2 + 4;
    constexpr int stride_tile_V = nbatch_V2 + 4;

    const int k_VKQ_0 = kb0 * nbatch_fa;
#if defined(TURING_MMA_AVAILABLE)
    T_C_KQ KQ_C[nbatch_fa/(np*(cols_per_warp == 8 ? T_C_KQ::I : T_C_KQ::J))];
#elif defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    T_C_KQ KQ_C[nbatch_fa/(np*T_C_KQ::J)];
#else // Volta
    T_C_KQ KQ_C[nbatch_fa/(np*T_C_KQ::J)];
#endif // defined(TURING_MMA_AVAILABLE)

    // [TAG_TURBOT] One granule table load per KV tile. k_VKQ_0 is the tile's first cell and a tile lies inside one
    // granule, so every lane of the block reads the same entry: the OLD/YOUNG branch below is lane-uniform. The lookup
    // stays in range, (ne11 - 1) >> 6 < n_granules (asserted on the host). Pool addresses are size_t.
    const int32_t      slot    = gtab[k_VKQ_0 >> GGML_TURBOT_LOG2_GRANULE];
    const bool         young   = slot >= 0;
    const char * const K_base0 = K_row + (size_t) k_VKQ_0 * (size_t) nb11;
    const char * const V_base0 = V_row + (size_t) k_VKQ_0 * (size_t) nb21;
    const char * const prow0   = young ?
        pool + ((size_t) slot * (size_t) GGML_TURBOT_GRANULE + (size_t) (k_VKQ_0 & (GGML_TURBOT_GRANULE - 1))) * (size_t) nb_pool : pool;

    {
        constexpr bool use_cp_async = false;
        if (kv_pos) {   // [TAG_FA_POS_MASK]
            flash_attn_ext_f16_gen_mask<ncols1, nwarps, nbatch_fa, oob_check>
                (kv_pos + k_VKQ_0, q_pos, tile_mask, k_VKQ_sup, jt*ncols1, ne01);
        } else if (ncols2 > 1 || mask_h) {
            // [TAG_SYNC_LOAD_MASK] upstream load_mask takes the tile start and a sparse index list; turbot never
            // uses sparse FA, so use_sparse = false and no indices (same addresses as the old mask_h + k_VKQ_0).
            flash_attn_ext_f16_load_mask<ncols1, nwarps, nbatch_fa, use_cp_async, oob_check, false>
                (mask_h, tile_mask, stride_mask, k_VKQ_0, k_VKQ_sup, jt*ncols1, ne01, nullptr);
        }
    }

#pragma unroll
    for (int k0_start = (DKQ/2-1) - (DKQ/2-1) % nbatch_K2; k0_start >= 0; k0_start -= nbatch_K2) {
        const int k0_stop = k0_start + nbatch_K2 < DKQ/2 ? k0_start + nbatch_K2 : DKQ/2;
        const int k0_diff = k0_stop - k0_start;

        // [TAG_TURBOT] K side: element offset inside the head is k0_start*2 (the rows are row bases, no head bias).
        if (young) {
            flash_attn_ext_turbot_load_tile_young_dispatch<(ncols >= TURBOT_CT_WIDTH_MIN_NCOLS), stride_tile_K, nwarps, nbatch_fa, oob_check>
                (hs.generic, K_base0, prow0, tile_K, k0_diff, nb11, nb_pool, k0_start*2, hs.k, lut + TURBOT_LUT_K_YOUNG, k_VKQ_sup);
        } else {
            flash_attn_ext_turbot_load_tile_old_dispatch<(ncols >= TURBOT_CT_WIDTH_MIN_NCOLS), stride_tile_K, nwarps, nbatch_fa, oob_check>
                (hs.generic, K_base0, tile_K, k0_diff, nb11, k0_start*2, hs.k, lut + TURBOT_LUT_K_OLD, k_VKQ_sup);
        }
        __syncthreads();

        // Calculate tile of KQ:
#pragma unroll
        for (int i_KQ_00 = 0; i_KQ_00 < nbatch_fa; i_KQ_00 += np*T_A_KQ::I) {
            const int i_KQ_0 = i_KQ_00 + (threadIdx.y % np)*T_A_KQ::I;
#pragma unroll
            for (int k_KQ_0 = k0_start; k_KQ_0 < k0_stop; k_KQ_0 += T_A_KQ::J) {
                T_A_KQ K_A;
                load_ldmatrix(K_A, tile_K + i_KQ_0*stride_tile_K + (k_KQ_0 - k0_start), stride_tile_K);
                if constexpr (cols_per_warp == 8) {
                    mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], K_A, Q_B[k_KQ_0/T_A_KQ::J]);
                } else {
                    // Wide version of KQ_C is column-major
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    // AMD matrix C is column-major.
                    mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], K_A, Q_B[k_KQ_0/T_A_KQ::J]);
#else
                    // swap A and B for CUDA.
                    mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], Q_B[k_KQ_0/T_A_KQ::J], K_A);
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                }
            }
        }

        __syncthreads(); // Only needed if tile_K == tile_V.
    }

    if (use_logit_softcap) {
        constexpr int stride = cols_per_warp == 8 ? np*T_C_KQ::I : np*T_C_KQ::J;
        static_assert(nbatch_fa % stride == 0, "bad loop size");
#pragma unroll
        for (int i = 0; i < nbatch_fa/stride; ++i) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                KQ_C[i].x[l] = logit_softcap*tanhf(KQ_C[i].x[l]);
            }
        }
    }

    float KQ_max_new[cols_per_thread];
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max_new[col] = KQ_max[col];
    }
    float KQ_rowsum_add[cols_per_thread] = {0.0f};

    if constexpr (cols_per_warp == 8) {
        if (ncols2 > 1 || mask_h || kv_pos) {
#pragma unroll
            for (int i00 = 0; i00 < nbatch_fa; i00 += np*T_C_KQ::I) {
                const int i0 = i00 + (threadIdx.y % np)*T_C_KQ::I;
#pragma unroll
                for (int l = 0; l < T_C_KQ::ne; ++l) {
                    const int i = i0 + T_C_KQ::get_i(l);
                    const int j = ((threadIdx.y / np)*T_C_KQ::J + T_C_KQ::get_j(l)) / ncols2;

                    KQ_C[i00/(np*T_C_KQ::I)].x[l] += slope * __half2float(tile_mask[j*(nbatch_fa + 8) + i]);
                }
            }
        }

        // Calculate softmax for each KQ column using the current max. value.
        // The divisor is stored in KQ_rowsum and will be applied at the end.
        static_assert(nbatch_fa % (np*T_C_KQ::I) == 0, "bad loop size");
#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::I) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                if (!oob_check || k0 + (threadIdx.y % np)*T_C_KQ::I + T_C_KQ::get_i(l) < k_VKQ_sup) {
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    constexpr int KQ_idx = 0;
#else
                    // Turing + Volta:
                    const int KQ_idx = l % 2;
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    KQ_max_new[KQ_idx] = fmaxf(KQ_max_new[KQ_idx], KQ_C[k0/(np*T_C_KQ::I)].x[l] + FATTN_KQ_MAX_OFFSET);
                }
            }
        }

        // Values per KQ column are spread across 8 threads:
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#pragma unroll
            for (int offset = 16; offset >= 4; offset >>= 1) {
                KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[col], offset, warp_size));
            }
        }

        static_assert(nbatch_fa % (np*T_C_KQ::I) == 0, "bad loop size");
#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::I) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                if (!oob_check || k0 + (threadIdx.y % np)*T_C_KQ::I + T_C_KQ::get_i(l) < k_VKQ_sup) {
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    constexpr int KQ_idx = 0;
#else
                    // Turing + Volta:
                    const int KQ_idx = l % 2;
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    KQ_C[k0/(np*T_C_KQ::I)].x[l] = expf(KQ_C[k0/(np*T_C_KQ::I)].x[l] - KQ_max_new[KQ_idx]);
                    KQ_rowsum_add[KQ_idx] += KQ_C[k0/(np*T_C_KQ::I)].x[l];
                } else {
                    KQ_C[k0/(np*T_C_KQ::I)].x[l] = 0.0f;
                }
            }
        }
    } else { // not Turing mma or T_B_KQ::I > 8
        if (ncols2 > 1 || mask_h || kv_pos) {
#pragma unroll
            for (int i00 = 0; i00 < nbatch_fa; i00 += np*T_C_KQ::J) {
                const int i0 = i00 + (threadIdx.y % np)*T_C_KQ::J;

                // The mask is stored as 16 bit half values, loading them as 32 bit half2 values is preferred in terms of speed.
                // However, this is not possible for RDNA3 where 2 consecutive l indices are not consecutive in the mask memory layout.
#ifdef RDNA3
#pragma unroll
                for (int l = 0; l < T_C_KQ::ne; ++l) {
                    const int i = i0 + T_C_KQ::get_j(l);
                    const int j = ((threadIdx.y / np)*cols_per_warp + T_C_KQ::get_i(l)) / ncols2;

                    KQ_C[i00/(np*T_C_KQ::J)].x[l] += __half2float(tile_mask[j*(nbatch_fa + 8) + i]);
                }
#else
#pragma unroll
                for (int l0 = 0; l0 < T_C_KQ::ne; l0 += 2) {
                    const int i = (i0 + T_C_KQ::get_j(l0)) / 2;
                    const int j = ((threadIdx.y / np)*cols_per_warp + T_C_KQ::get_i(l0)) / ncols2;

                    const float2 tmp = __half22float2(((const half2 *)tile_mask)[j*(nbatch_fa/2 + 4) + i]);
                    KQ_C[i00/(np*T_C_KQ::J)].x[l0 + 0] += slope*tmp.x;
                    KQ_C[i00/(np*T_C_KQ::J)].x[l0 + 1] += slope*tmp.y;
                }
#endif // RDNA3
            }
        }

        // Calculate softmax for each KQ column using the current max. value.
        // The divisor is stored in KQ_rowsum and will be applied at the end.
        static_assert(nbatch_fa % (np*T_C_KQ::J) == 0, "bad loop size");
#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::J) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                if (!oob_check || k0 + (threadIdx.y % np)*T_C_KQ::J + T_C_KQ::get_j(l) < k_VKQ_sup) {
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    constexpr int KQ_idx = 0;
#else
                    // Turing + Volta:
                    const int KQ_idx = (l/2) % 2;
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    KQ_max_new[KQ_idx] = fmaxf(KQ_max_new[KQ_idx], KQ_C[(k0/(np*T_C_KQ::J))].x[l] + FATTN_KQ_MAX_OFFSET);
                }
            }
        }

#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#if defined(TURING_MMA_AVAILABLE)
            // Values per KQ column are spread across 4 threads:
            constexpr int offset_first = 2;
            constexpr int offset_last  = 1;
#elif defined(AMD_MFMA_AVAILABLE)
            // MFMA: 4 threads per Q column (threadIdx.x % 16 == col, spaced by 16).
            constexpr int offset_first = 32;
            constexpr int offset_last  = 16;
#elif defined(AMD_WMMA_AVAILABLE)
            // Values per KQ column are spread across 2 threads:
            constexpr int offset_first = 16;
            constexpr int offset_last  = 16;
#else // Volta
            // Values per KQ column are spread across 2 threads:
            constexpr int offset_first = 2;
            constexpr int offset_last  = 2;
#endif // defined(TURING_MMA_AVAILABLE)
#pragma unroll
            for (int offset = offset_first; offset >= offset_last; offset >>= 1) {
                KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[col], offset, warp_size));
            }
        }

        static_assert(nbatch_fa % (np*T_C_KQ::J) == 0, "bad loop size");
#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::J) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                if (!oob_check || k0 + (threadIdx.y % np)*T_C_KQ::J + T_C_KQ::get_j(l) < k_VKQ_sup) {
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    constexpr int KQ_idx = 0;
#else
                    // Turing + Volta:
                    const int KQ_idx = (l/2) % 2;
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    KQ_C[(k0/(np*T_C_KQ::J))].x[l] = expf(KQ_C[(k0/(np*T_C_KQ::J))].x[l] - KQ_max_new[KQ_idx]);
                    KQ_rowsum_add[KQ_idx] += KQ_C[(k0/(np*T_C_KQ::J))].x[l];
                } else {
                    KQ_C[(k0/(np*T_C_KQ::J))].x[l] = 0.0f;
                }
            }
        }
    }

    {
        float KQ_max_scale[cols_per_thread];
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            const float KQ_max_diff = KQ_max[col] - KQ_max_new[col];
            KQ_max_scale[col] = expf(KQ_max_diff);
            KQ_max[col] = KQ_max_new[col];

            *((uint32_t *) &KQ_max_scale[col]) *= KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD;

            // Scale previous KQ_rowsum to account for a potential increase in KQ_max:
            KQ_rowsum[col] = KQ_max_scale[col]*KQ_rowsum[col] + KQ_rowsum_add[col];
        }

#if defined(TURING_MMA_AVAILABLE)
        if constexpr (cols_per_warp == 8) {
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[cols_per_thread - 1]);
#pragma unroll
            for (int i = 0; i < DV/T_C_VKQ::I; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale_h2;
                }
            }
        } else {
#pragma unroll
            for (int col = 0; col < cols_per_thread; ++col) {
                const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[col], KQ_max_scale[col]);
#pragma unroll
                for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
                    for (int l0 = 0; l0 < T_C_VKQ::ne; l0 += 2) {
                        VKQ_C[i].x[l0 + col] *= KQ_max_scale_h2;
                    }
                }
            }
        }
#elif defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
        if constexpr (std::is_same_v<decltype(T_C_VKQ::x), half2[T_C_VKQ::ne]>) {
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[0]);
#pragma unroll
            for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale_h2;
                }
            }
        } else {
            static_assert(std::is_same_v<decltype(T_C_VKQ::x), float[T_C_VKQ::ne]>, "bad VKQ type");
#pragma unroll
            for (int i = 0; i < DV/T_C_VKQ::J; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale[0];
                }
            }
        }
#else // Volta
        const half2 KQ_max_scale_h2 = make_half2(
            KQ_max_scale[(threadIdx.x / 2) % 2], KQ_max_scale[(threadIdx.x / 2) % 2]);
#pragma unroll
        for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
            for (int l = 0; l < T_C_VKQ::ne; ++l) {
                VKQ_C[i].x[l] *= KQ_max_scale_h2;
            }
        }
#endif // defined(TURING_MMA_AVAILABLE)
    }

    // Convert KQ C tiles into B tiles for VKQ calculation:
    T_B_VKQ B[nbatch_fa/(np*2*T_B_VKQ::J)];
    static_assert(nbatch_fa % (np*2*T_B_VKQ::J) == 0, "bad loop size");
    if constexpr (cols_per_warp == 8) {
#pragma unroll
        for (int k = 0; k < nbatch_fa/(np*2*T_B_VKQ::J); ++k) {
            B[k] = get_transposed(get_half2(KQ_C[k]));
        }
    } else {
        for (int k = 0; k < nbatch_fa/(np*2*T_B_VKQ::J); ++k) {
            B[k] = get_half2(KQ_C[k]);
        }
    }

    // Calculate VKQ tile, need to use logical rather than physical elements for i0 due to transposition of V:
#pragma unroll
    for (int i0_start = 0; i0_start < DV; i0_start += 2*nbatch_V2) {
        static_assert(DV % (2*nbatch_V2) == 0, "bad loop size");
        const int i0_stop = i0_start + 2*nbatch_V2;
        const int i0_diff = i0_stop - i0_start;

        // [TAG_TURBOT] V side: V base row offsets, V refinement at pool_v_off (folded into hs.v.pool).
        if (young) {
            flash_attn_ext_turbot_load_tile_young_dispatch<(ncols >= TURBOT_CT_WIDTH_MIN_NCOLS), stride_tile_V, nwarps, nbatch_fa, oob_check>
                (hs.generic, V_base0, prow0, tile_V, i0_diff/2, nb21, nb_pool, i0_start, hs.v, lut + TURBOT_LUT_V_YOUNG, k_VKQ_sup);
        } else {
            flash_attn_ext_turbot_load_tile_old_dispatch<(ncols >= TURBOT_CT_WIDTH_MIN_NCOLS), stride_tile_V, nwarps, nbatch_fa, oob_check>
                (hs.generic, V_base0, tile_V, i0_diff/2, nb21, i0_start, hs.v, lut + TURBOT_LUT_V_OLD, k_VKQ_sup);
        }
        __syncthreads();

        const half2 * tile_V_i = tile_V;

#if defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
#pragma unroll
        for (int i_VKQ_0 = i0_start; i_VKQ_0 < i0_stop; i_VKQ_0 += T_A_VKQ::I) {
            static_assert((nbatch_fa/2) % (np*T_A_VKQ::J) == 0, "bad loop size");
#pragma unroll
            for (int k00 = 0; k00 < nbatch_fa/2; k00 += np*T_A_VKQ::J) {
                const int k0 = k00 + (threadIdx.y % np)*T_A_VKQ::J;

                T_A_VKQ A; // Transposed in SRAM but not in registers, gets transposed on load.
                load_ldmatrix_trans(A, tile_V_i + 2*k0*stride_tile_V + (i_VKQ_0 - i0_start)/2, stride_tile_V);
                if constexpr (T_B_KQ::I == 8) {
                    mma(VKQ_C[i_VKQ_0/T_A_VKQ::I], A, B[k00/(np*T_A_VKQ::J)]);
                } else {
                    // Wide version of VKQ_C is column-major.
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    // AMD matrix C is column-major.
                    mma(VKQ_C[i_VKQ_0/T_A_VKQ::I], A, B[k00/(np*T_A_VKQ::J)]);
#else
                    // swap A and B for CUDA.
                    mma(VKQ_C[i_VKQ_0/T_A_VKQ::I], B[k00/(np*T_A_VKQ::J)], A);
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                }
            }
        }
#else // Volta
        constexpr int i0_stride = 2*T_C_VKQ::J;
#pragma unroll
        for (int i_VKQ_0 = i0_start; i_VKQ_0 < i0_stop; i_VKQ_0 += i0_stride) {
            static_assert(nbatch_fa % (np*T_A_VKQ::I) == 0, "bad loop size");
            static_assert(2*T_B_VKQ::J == T_A_VKQ::I, "bad tile sizes");
#pragma unroll
            for (int k00 = 0; k00 < nbatch_fa; k00 += np*T_A_VKQ::I) {
                const int k0 = k00 + (threadIdx.y % np)*T_A_VKQ::I;

                T_A_VKQ A; // Transposed in both SRAM and registers, load normally.
                load_ldmatrix(A, tile_V_i + k0*stride_tile_V + (i_VKQ_0 - i0_start)/2, stride_tile_V);
                mma(VKQ_C[i_VKQ_0/i0_stride], B[k00/(np*T_A_VKQ::I)], A);
            }
        }
#endif // defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)

        __syncthreads(); // Only needed if tile_K == tile_V.
    }
#else
    GGML_UNUSED_VARS(Q_f2, K_row, V_row, pool, gtab, mask_h, kv_pos, q_pos, dstk, dstk_fixup,
        scale, slope, logit_softcap, ne01, ne02, nb11, nb21, nb_pool, stride_mask,
        tile_Q, tile_K, tile_V, tile_mask, lut, Q_B, VKQ_C, KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup, hs);
    NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
}

// ------------------------------------------------------------------------------------------------------------------
// Copy of flash_attn_ext_f16_process_tile for turbot: nstages = 0, V_is_K_view = false, Q in registers, plus the
// shared-memory LUT copy.

template<int DKQ, int DV, int ncols1, int ncols2, int nwarps, bool use_logit_softcap, bool needs_fixup, bool is_fixup>
static __device__ __forceinline__ void flash_attn_ext_turbot_process_tile(
        const float2 * const __restrict__ Q_f2,
        const char   * const __restrict__ K_row,     // [TAG_TURBOT] base row of cell 0
        const char   * const __restrict__ V_row,
        const char   * const __restrict__ pool,
        const int32_t * const __restrict__ gtab,
        const half   * const __restrict__ mask_h,
        const int32_t * const __restrict__ kv_pos,   // [TAG_FA_POS_MASK]
        const int32_t * const __restrict__ q_pos,
        const float  * const __restrict__ sinks_f,
        float2       * const __restrict__ dstk,
        float2       * const __restrict__ dstk_fixup,
        const float scale,
        const float slope,
        const float logit_softcap,
        const uint3 ne01,
        const int ne02,
        const int gqa_ratio,
        const int ne11,
        const int stride_Q1,
        const int stride_Q2,
        const int64_t nb11,
        const int64_t nb21,
        const int64_t nb_pool,
        const int stride_mask,
        const int jt,
        const int zt_gqa,
        const int kb0_start,
        const int kb0_stop,
        const int kb0_stride,                        // [TAG_TURBOT_FA_STRIPE] 1, or the stripe of a striped block
        const turbot_head_state & hs) {
#if defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int ncols = ncols1 * ncols2;
    using     T_A_KQ    = typename mma_tile_sizes<DV, ncols>::T_A_KQ;
    using     T_B_KQ    = typename mma_tile_sizes<DV, ncols>::T_B_KQ;
    using     T_C_KQ    = typename mma_tile_sizes<DV, ncols>::T_C_KQ;
    using     T_A_VKQ   = typename mma_tile_sizes<DV, ncols>::T_A_VKQ;
    using     T_B_VKQ   = typename mma_tile_sizes<DV, ncols>::T_B_VKQ;
    using     T_C_VKQ   = typename mma_tile_sizes<DV, ncols>::T_C_VKQ;

    constexpr int  cols_per_warp   = T_B_KQ::I;
    constexpr int  cols_per_thread = get_cols_per_thread();
    constexpr int  np              = cols_per_warp > ncols ? nwarps : nwarps * cols_per_warp/ncols; // Number of parallel CUDA warps per Q column.
    constexpr int  nbatch_fa       = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols);
    constexpr int  nbatch_K2       = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols);
    constexpr int  nbatch_V2       = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols);
    constexpr int  nbatch_combine  = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols);
    constexpr bool Q_in_reg        = ggml_cuda_fattn_mma_get_Q_in_reg      (DKQ, DV, ncols);

    if (cols_per_warp > ncols) {
        NO_DEVICE_CODE;
        return;
    }

    static_assert(nwarps * (cols_per_warp/ncols2) % ncols1 == 0, "bad nwarps");
    static_assert(Q_in_reg, "turbot: every D=256 config keeps Q in registers, so tile_K == tile_Q");

    constexpr int stride_tile_Q = DKQ/2     + 4;
    constexpr int stride_tile_K = nbatch_K2 + 4;
    constexpr int stride_tile_V = nbatch_V2 + 4;
    constexpr int stride_tile_KV_max = stride_tile_K > stride_tile_V ? stride_tile_K : stride_tile_V;

    extern __shared__ half2 tile_Q[];
    half2 * tile_K    = tile_Q;
    half2 * tile_V    = tile_K;
    half  * tile_mask = (half *) (tile_V + nbatch_fa * stride_tile_KV_max);

    // [TAG_TURBOT] LUT area, directly after the mask tile (SPEC 7.4). The host sizes shared memory with the same
    // ggml_cuda_fattn_turbot_lut_off + TURBOT_NBYTES_SHARED_LUT.
    constexpr int lut_off = ggml_cuda_fattn_turbot_lut_off(nbatch_fa, nbatch_K2, nbatch_V2, ncols1);
    static_assert(lut_off >= nbatch_fa*stride_tile_KV_max*(int) sizeof(half2) + ncols1*(nbatch_fa + 8)*(int) sizeof(half),
                  "turbot LUT area overlaps the KV or mask tile");
    float * const lut = ((float *) tile_Q) + lut_off/(int) sizeof(float);

    T_B_KQ    Q_B[DKQ/(2*T_B_KQ::J)];
#if defined(TURING_MMA_AVAILABLE)
    T_C_VKQ VKQ_C[cols_per_warp == 8 ? DV/T_C_VKQ::I : DV/(2*T_C_VKQ::J)];
#elif defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
    T_C_VKQ VKQ_C[DV % 32 != 0       ? DV/T_C_VKQ::J : DV/(2*T_C_VKQ::J)];
#elif defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    T_C_VKQ VKQ_C[                                     DV/(2*T_C_VKQ::J)];
#else // Volta
    T_C_VKQ VKQ_C[                                     DV/(2*T_C_VKQ::J)];
#endif // defined(TURING_MMA_AVAILABLE)

    float KQ_rowsum[cols_per_thread] = {0.0f};
    float KQ_max[cols_per_thread];
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max[col] = -FLT_MAX/2.0f;
    }

    // Load Q data into tile_Q, either temporarily or permanently.
    // Q in registers is faster, but register pressure is the biggest bottleneck.
    // The loading is done with decreasing granularity for D for better memory bandwidth.
    const half2 scale_h2 = make_half2(scale, scale);
#pragma unroll
    for (int stride_k : {warp_size, warp_size/2, warp_size/4, warp_size/8}) {
        const int k0_start  = stride_k == warp_size ? 0 : DKQ/2 - (DKQ/2) % (2*stride_k);
        const int k0_stop   =                             DKQ/2 - (DKQ/2) % (1*stride_k);
        const int stride_jc = warp_size / stride_k;

        if (k0_start == k0_stop) {
            continue;
        }

#pragma unroll
        for (int jc0 = 0; jc0 < ncols; jc0 += nwarps*stride_jc) {
            const int jc = jc0 + threadIdx.y*stride_jc + (stride_k == warp_size ? 0 : threadIdx.x / stride_k);

            if (jc0 + nwarps*stride_jc > ncols && jc >= ncols) {
                break;
            }

            const int j = jc / ncols2;
            const int c = jc % ncols2;

            if ((ncols1 == 1 || jt*ncols1 + j < int(ne01.z)) && (ncols2 == 1 || zt_gqa*ncols2 + c < gqa_ratio)) {
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                    const float2 tmp = Q_f2[(jt*ncols1 + j)*stride_Q1 + c*stride_Q2 + k];
                    tile_Q[jc*stride_tile_Q + k] = scale_h2 * make_half2(tmp.x, tmp.y);
                }
            } else {
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                    tile_Q[jc*stride_tile_Q + k] = make_half2(0.0f, 0.0f);
                }
            }
        }
    }

    __syncthreads();

    {
        const int j0 = (threadIdx.y / np) * cols_per_warp;

#pragma unroll
        for (int k0 = 0; k0 < DKQ/2; k0 += T_B_KQ::J) {
            load_ldmatrix(Q_B[k0/T_B_KQ::J], tile_Q + j0*stride_tile_Q + k0, stride_tile_Q);
        }
    }

    __syncthreads();

    // [TAG_TURBOT] LUT copy point (SPEC 7.4): after the barrier that follows the Q_B register load, before the first
    // iter, and nowhere earlier. With Q in registers the KV, mask and LUT areas all overlap tile_Q (for ncols 128 the
    // Q tile spans 67,584 B while the LUT area starts near 27,000 B), and tile_Q only reaches Q_B at the load above.
    //
    // The runs are the current KV head's K and V codebooks. Their offsets are uniform across the block, so the
    // constant-memory loads are the lane-consecutive, coalescing kind; warp 0 stores each entry exactly once and the
    // barrier below publishes the area to every warp before any loader gathers from it. Every read stays inside the
    // tables (C_6 is the last old run, LUT_{6,8} the last young run).
    //
    // [TAG_TURBOT_LUT_TRIM] Only the entries a loader can index are copied: 1 << y young entries, and the old run only
    // when flash_attn_ext_turbot_load_tile_old takes its float branch for that side (b = 6, or every b when
    // GGML_CUDA_TURBOT_OLD_I8 is 0). Counts are rounded up to whole warps, so the guards are warp-uniform (one j0 per
    // warp) and never lane-divergent; the rounded tail stays inside the 64 / 256 entries the copy always covered.
    if (threadIdx.y == 0) {
        constexpr bool lut_old_i8 = GGML_CUDA_TURBOT_OLD_I8 != 0;
        const int n_old_k   = (!lut_old_i8 || hs.k.b > 5) ? GGML_PAD(1 << hs.k.b, warp_size) : 0;
        const int n_old_v   = (!lut_old_i8 || hs.v.b > 5) ? GGML_PAD(1 << hs.v.b, warp_size) : 0;
        const int n_young_k = GGML_PAD(1 << (hs.k.b + hs.k.r), warp_size);
        const int n_young_v = GGML_PAD(1 << (hs.v.b + hs.v.r), warp_size);
#pragma unroll
        for (int j0 = 0; j0 < 64; j0 += warp_size) {
            const int j = j0 + threadIdx.x;
            if (j0 < n_old_k) {
                lut[TURBOT_LUT_K_OLD + j] = TURBOT_D_OLD_LEVELS[hs.k.old_run + j];
            }
            if (j0 < n_old_v) {
                lut[TURBOT_LUT_V_OLD + j] = TURBOT_D_OLD_LEVELS[hs.v.old_run + j];
            }
        }
#pragma unroll
        for (int j0 = 0; j0 < 256; j0 += warp_size) {
            const int j = j0 + threadIdx.x;
            if (j0 < n_young_k) {
                lut[TURBOT_LUT_K_YOUNG + j] = TURBOT_D_YOUNG_LUT[hs.k.young_run + j];
            }
            if (j0 < n_young_v) {
                lut[TURBOT_LUT_V_YOUNG + j] = TURBOT_D_YOUNG_LUT[hs.v.young_run + j];
            }
        }
    }

    __syncthreads();

    int kb0 = kb0_start;

    // kb0_start is always < kb0_stop so the last iter can be executed unconditionally.
    if constexpr (ncols2 == 1) {
        constexpr bool oob_check = true;
        for (; kb0 + kb0_stride < kb0_stop; kb0 += kb0_stride) {   // [TAG_TURBOT_FA_STRIPE] stride 1: kb0 < kb0_stop-1
            constexpr bool last_iter = false;
            constexpr int  k_VKQ_sup = nbatch_fa;
            flash_attn_ext_turbot_iter
                <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, needs_fixup, is_fixup, last_iter, oob_check,
                 T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ>
                (Q_f2, K_row, V_row, pool, gtab, mask_h, kv_pos, q_pos, dstk, dstk_fixup, scale, slope, logit_softcap,
                 ne01, ne02, nb11, nb21, nb_pool, stride_mask, tile_Q, tile_K, tile_V, tile_mask, lut, Q_B, VKQ_C,
                 KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup, hs);
        }
        constexpr bool last_iter = true;
        const     int  k_VKQ_sup = ne11 - kb0*nbatch_fa;
        flash_attn_ext_turbot_iter
            <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, needs_fixup, is_fixup, last_iter, oob_check,
              T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ>
            (Q_f2, K_row, V_row, pool, gtab, mask_h, kv_pos, q_pos, dstk, dstk_fixup, scale, slope, logit_softcap,
             ne01, ne02, nb11, nb21, nb_pool, stride_mask, tile_Q, tile_K, tile_V, tile_mask, lut, Q_B, VKQ_C,
             KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup, hs);
    } else {
        constexpr bool oob_check = false;
        for (; kb0 + kb0_stride < kb0_stop; kb0 += kb0_stride) {   // [TAG_TURBOT_FA_STRIPE] stride 1: kb0 < kb0_stop-1
            constexpr bool last_iter = false;
            constexpr int  k_VKQ_sup = nbatch_fa;
            flash_attn_ext_turbot_iter
                <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, needs_fixup, is_fixup, last_iter, oob_check,
                 T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ>
                (Q_f2, K_row, V_row, pool, gtab, mask_h, kv_pos, q_pos, dstk, dstk_fixup, scale, slope, logit_softcap,
                 ne01, ne02, nb11, nb21, nb_pool, stride_mask, tile_Q, tile_K, tile_V, tile_mask, lut, Q_B, VKQ_C,
                 KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup, hs);
        }
        constexpr bool last_iter = true;
        constexpr int  k_VKQ_sup = nbatch_fa;
        flash_attn_ext_turbot_iter
            <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, needs_fixup, is_fixup, last_iter, oob_check,
             T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ>
            (Q_f2, K_row, V_row, pool, gtab, mask_h, kv_pos, q_pos, dstk, dstk_fixup, scale, slope, logit_softcap,
             ne01, ne02, nb11, nb21, nb_pool, stride_mask, tile_Q, tile_K, tile_V, tile_mask, lut, Q_B, VKQ_C,
             KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup, hs);
    }

    // Finally, sum up partial KQ rowsums.
    {
#if defined(TURING_MMA_AVAILABLE)
        // The partial sums are spread across 8/4 threads.
        constexpr int offset_first = cols_per_warp == 8 ? 16 : 2;
        constexpr int offset_last  = cols_per_warp == 8 ?  4 : 1;
#elif defined(AMD_MFMA_AVAILABLE)
        // The partial sums are spread across 4 threads (wavefront64, 16 cols).
        constexpr int offset_first = 32;
        constexpr int offset_last  = 16;
#elif defined(AMD_WMMA_AVAILABLE)
        // The partial sums are spread across 2 threads.
        constexpr int offset_first = 16;
        constexpr int offset_last  = 16;
#else // Volta
        // The partial sums are spread across 2 threads.
        constexpr int offset_first = 2;
        constexpr int offset_last  = 2;
#endif // defined(TURING_MMA_AVAILABLE)
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#pragma unroll
            for (int offset = offset_first; offset >= offset_last; offset >>= 1) {
                KQ_rowsum[col] += __shfl_xor_sync(0xFFFFFFFF, KQ_rowsum[col], offset, warp_size);
            }
        }
    }

    // If attention sinks are used, potentially re-scale if KQ_max is small.
    // Also add the sink as a value to KQ_rowsum, this is done after synchronization of KQ_rowsum
    //     so it's being done unconditionally for every thread.
    if (!is_fixup && (np == 1 || threadIdx.y % np == 0) && sinks_f) {
        float KQ_max_scale[cols_per_thread];
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            const int jc = (threadIdx.y/np)*cols_per_warp + (cols_per_warp == 8 ? T_C_KQ::get_j(col) : T_C_KQ::get_i(2*col));
            const float sink = sinks_f[jc % ncols2];

            const float KQ_max_new = fmaxf(KQ_max[col], sink);
            const float KQ_max_diff = KQ_max[col] - KQ_max_new;
            KQ_max_scale[col] = expf(KQ_max_diff);
            KQ_max[col] = KQ_max_new;

            *((uint32_t *) &KQ_max_scale[col]) *= KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD;

            const float KQ_max_add = expf(sink - KQ_max_new);
            KQ_rowsum[col] = KQ_max_scale[col]*KQ_rowsum[col] + KQ_max_add;
        }

#if defined(TURING_MMA_AVAILABLE)
        if constexpr (cols_per_warp == 8) {
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[cols_per_thread - 1]);
#pragma unroll
            for (int i = 0; i < DV/T_C_VKQ::I; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale_h2;
                }
            }
        } else {
#pragma unroll
            for (int col = 0; col < cols_per_thread; ++col) {
                const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[col], KQ_max_scale[col]);
#pragma unroll
                for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
                    for (int l0 = 0; l0 < T_C_VKQ::ne; l0 += 2) {
                        VKQ_C[i].x[l0 + col] *= KQ_max_scale_h2;
                    }
                }
            }
        }
#elif defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
        if constexpr (std::is_same_v<decltype(T_C_VKQ::x), half2[T_C_VKQ::ne]>) {
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[0]);
#pragma unroll
            for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale_h2;
                }
            }
        } else {
            static_assert(std::is_same_v<decltype(T_C_VKQ::x), float[T_C_VKQ::ne]>, "bad VKQ type");
#pragma unroll
            for (int i = 0; i < DV/T_C_VKQ::J; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale[0];
                }
            }
        }
#else // Volta
        const int col = (threadIdx.x / 2) % 2;
        const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[col], KQ_max_scale[col]);
#pragma unroll
        for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
            for (int l = 0; l < T_C_VKQ::ne; ++l) {
                VKQ_C[i].x[l] *= KQ_max_scale_h2;
            }
        }
#endif // defined(TURING_MMA_AVAILABLE)
    }

    // Combine VKQ accumulator values if np > 1.
    // It's also faster to do small writes to shared memory, then large write to VRAM than to do small writes to VRAM.
    // So also write VKQ accumulators to shared memory in column-major format if np == 1.

    constexpr int tile_stride = nbatch_combine + 4;
    static_assert((DV/2) % nbatch_combine == 0, "bad nbatch_combine");

    if constexpr (cols_per_warp == 8) {
        const int jc_cwmo = (threadIdx.x % (2*T_C_VKQ::J)) / T_C_VKQ::J; // jc combine write meta offset
        const int jc_cwm = threadIdx.y*(2*T_C_VKQ::J) + 2*T_C_VKQ::get_j(-1) + jc_cwmo; // jc combine write meta
        const float2 KQ_cmr = make_float2(KQ_max[jc_cwmo], KQ_rowsum[jc_cwmo]); // KQ combine max rowsum

        if (((!needs_fixup && !is_fixup) || np > 1) && threadIdx.x < 2*T_C_VKQ::J) {
            // Use the 16 bytes of padding in each row to store the meta data: KQ max, KQ rowsum, KQ max scale.
            ((float2 *) tile_Q)[jc_cwm*(tile_stride/2) + nbatch_combine/2] = KQ_cmr;
        }

        __syncthreads();

        if (np == 1) {
            // No combination is needed, the meta data can be directly written from registers to VRAM.
            if (needs_fixup && threadIdx.x < T_B_KQ::I) {
                float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
            if (is_fixup && threadIdx.x < T_B_KQ::I) {
                float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
        }
    } else {
        // jc_cwm = jc combine write meta
        // KQ_cmr = KQ combine max rowsum
        // Use the 16 bytes of padding in each Q column to store the meta data: KQ max, KQ rowsum, KQ max scale.
#if defined(TURING_MMA_AVAILABLE)
        const int jc_cwm = threadIdx.y*cols_per_warp + T_C_VKQ::get_i(threadIdx.x % 4);
        const float2 KQ_cmr = make_float2(KQ_max[threadIdx.x % cols_per_thread], KQ_rowsum[threadIdx.x % cols_per_thread]);
        const bool thread_should_write = threadIdx.x % 4 < cols_per_thread;
#elif defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
        const int jc_cwm = threadIdx.y*cols_per_warp + T_C_VKQ::get_i(0);
        const float2 KQ_cmr = make_float2(KQ_max[0], KQ_rowsum[0]);
        const bool thread_should_write = threadIdx.x / 16 < cols_per_thread;
#else // Volta
        const int jc_cwm = threadIdx.y*cols_per_warp + T_C_KQ::get_i(threadIdx.x & 2);
        const float2 KQ_cmr = make_float2(KQ_max[(threadIdx.x & 2) / 2], KQ_rowsum[(threadIdx.x & 2) / 2]);
        const bool thread_should_write = T_C_KQ::J == 8 || T_C_KQ::get_j(threadIdx.x & 2) < 8;
#endif // defined(TURING_MMA_AVAILABLE)

        if (((!needs_fixup && !is_fixup) || np > 1) && thread_should_write) {
            ((float2 *) tile_Q)[jc_cwm*(tile_stride/2) + nbatch_combine/2] = KQ_cmr;
        }

        __syncthreads();

        if (np == 1) {
            // No combination is needed, the meta data can be directly written from registers to VRAM.
            if (needs_fixup && thread_should_write) {
                float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
            if (is_fixup && thread_should_write) {
                float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
        }
    }

    // [TAG_SYNC_TURBOT_BARRIER] upstream b74f590ea (#27870) ported: every warp reaches the SAME __syncthreads below. The
    // old form had the combining warps (threadIdx.y % np == 0) and the rest sync on two different barriers in two
    // branches, which is divergent-barrier undefined behaviour; the arithmetic and the writes are unchanged.
    if (np > 1) {
        constexpr int nmeta = np*cols_per_warp >= warp_size ? np*cols_per_warp/warp_size : 1;

        float KQ_cmn;
        float KQ_cms[nmeta];
        float KQ_crs;

        const int jc_meta = threadIdx.y*cols_per_warp + (np*cols_per_warp < warp_size ? threadIdx.x % (np*cols_per_warp) : threadIdx.x);
        float2 * const meta_ptr = ((float2 *) tile_Q) + jc_meta*(tile_stride/2) + nbatch_combine/2;

        if (threadIdx.y % np == 0) {
            // Combine the meta data for parallel warps via shared memory.
            float2 meta[nmeta];
#pragma unroll
            for (int imeta = 0; imeta < nmeta; ++imeta) {
                meta[imeta] = meta_ptr[imeta * warp_size * tile_stride/2];
            }

            KQ_cmn = meta[0].x; // KQ combine max new, max between all parallel warps.
#pragma unroll
            for (int imeta = 1; imeta < nmeta; ++imeta) {
                KQ_cmn = fmaxf(KQ_cmn, meta[imeta].x);
            }
#pragma unroll
            for (int offset = np*cols_per_warp/2; offset >= cols_per_warp; offset >>= 1) {
                if (offset < warp_size) {
                    KQ_cmn = fmaxf(KQ_cmn, __shfl_xor_sync(0xFFFFFFFF, KQ_cmn, offset, warp_size));
                }
            }

#pragma unroll
            for (int imeta = 0; imeta < nmeta; ++imeta) {
                KQ_cms[imeta] = expf(meta[imeta].x - KQ_cmn);
            }

            KQ_crs = KQ_cms[0]*meta[0].y; // KQ combine rowsum, scaled sum of all parallel warps.
#pragma unroll
            for (int imeta = 1; imeta < nmeta; ++imeta) {
                KQ_crs += KQ_cms[imeta]*meta[imeta].y;
            }
#pragma unroll
            for (int offset = np*cols_per_warp/2; offset >= cols_per_warp; offset >>= 1) {
                if (offset < warp_size) {
                    KQ_crs += __shfl_xor_sync(0xFFFFFFFF, KQ_crs, offset, warp_size);
                }
            }
        }

        __syncthreads();

        if (threadIdx.y % np == 0) {
            // Write back combined meta data:
#pragma unroll
            for (int imeta = 0; imeta < nmeta; ++imeta) {
                if (np*cols_per_warp >= warp_size || threadIdx.x < np*cols_per_warp) {
                    // Combined KQ max scale + rowsum.
                    meta_ptr[imeta * warp_size * tile_stride/2] = make_float2(KQ_cms[imeta], KQ_crs);
                }
            }

            // Combined KQ max + rowsum.
            static_assert(cols_per_warp <= warp_size);
            if (needs_fixup && (cols_per_warp == warp_size || threadIdx.x < cols_per_warp)) {
                float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
                dstk_fixup_meta[(threadIdx.y/np)*cols_per_warp + threadIdx.x] = make_float2(KQ_cmn, KQ_crs);
            }
            if (is_fixup && (cols_per_warp == warp_size || threadIdx.x < cols_per_warp)) {
                float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
                dstk_fixup_meta[(threadIdx.y/np)*cols_per_warp + threadIdx.x] = make_float2(KQ_cmn, KQ_crs);
            }
        }
    }

#pragma unroll
    for (int k00 = 0; k00 < DV/2; k00 += nbatch_combine) {
        if constexpr (cols_per_warp == 8) {
            static_assert(std::is_same_v<decltype(T_C_VKQ::x), half2[T_C_VKQ::ne]>, "bad VKQ type");
            const int jc_cwd = threadIdx.y*T_B_KQ::I + T_B_KQ::get_i(-1); // jc combine write data
#pragma unroll
            for (int k1 = 0; k1 < nbatch_combine; k1 += T_B_KQ::J) {
                const T_B_KQ B = get_transposed(VKQ_C[(k00 + k1)/T_B_KQ::J]); // Conversion of C to B matrix puts it in column-major format.

#pragma unroll
                for (int l = 0; l < T_B_KQ::ne; ++l) {
                    const int k = k1 + T_B_KQ::get_j(l);

                    tile_Q[jc_cwd*tile_stride + k] = B.x[l];
                }
            }
        } else {
            const int j0 = threadIdx.y*cols_per_warp;
            if constexpr (std::is_same_v<decltype(T_C_VKQ::x), half2[T_C_VKQ::ne]>) {
                if constexpr (T_C_VKQ::dl == DATA_LAYOUT_I_MAJOR) {
#pragma unroll
                    for (int k1 = 0; k1 < nbatch_combine; k1 += T_C_VKQ::J) {
#pragma unroll
                        for (int l = 0; l < T_C_VKQ::ne; ++l) {
                            const int j = j0 + T_C_VKQ::get_i(l);
                            const int k = k1 + T_C_VKQ::get_j(l);

                            tile_Q[j*tile_stride + k] = VKQ_C[(k00 + k1)/T_C_VKQ::J].x[l];
                        }
                    }
                } else {
                    static_assert(T_C_VKQ::dl == DATA_LAYOUT_I_MAJOR_SCRAMBLED, "bad T_C_VKQ data layout");
                    using T_C_VKQ_us = tile<T_C_VKQ::I, T_C_VKQ::J, half2, DATA_LAYOUT_I_MAJOR>; // us == unscrambled
#pragma unroll
                    for (int k1 = 0; k1 < nbatch_combine; k1 += T_C_VKQ::J) {
                        const T_C_VKQ_us VKQ_C_us = unscramble(VKQ_C[(k00 + k1)/T_C_VKQ::J]);
#pragma unroll
                        for (int l = 0; l < T_C_VKQ_us::ne; ++l) {
                            const int j = j0 + T_C_VKQ_us::get_i(l);
                            const int k = k1 + T_C_VKQ_us::get_j(l);

                            tile_Q[j*tile_stride + k] = VKQ_C_us.x[l];
                        }
                    }
                }
            } else {
                static_assert(std::is_same_v<decltype(T_C_VKQ::x), float[T_C_VKQ::ne]>, "bad VKQ type");
                half * tile_Q_h = (half *) tile_Q;
#pragma unroll
                for (int k1 = 0; k1 < nbatch_combine; k1 += T_C_VKQ::J/2) {
#pragma unroll
                    for (int l = 0; l < T_C_VKQ::ne; ++l) {
                        const int j = j0 + T_C_VKQ::get_i(l);
                        const int k = 2*k1 + T_C_VKQ::get_j(l);

                        tile_Q_h[j*(2*tile_stride) + k] = VKQ_C[(k00 + k1)/(T_C_VKQ::J/2)].x[l];
                    }
                }
            }
        }

        __syncthreads();

        if (np == 1 || threadIdx.y % np == 0) {
            // The first 2*2*gridDim.x*ncols floats in dstk_fixup are for storing max. values and row sums.
            // The values after that are for the partial results of the individual blocks.
            float2 * dstk_fixup_data = dstk_fixup + gridDim.x*(2*ncols) + blockIdx.x*(ncols*(DV/2));

#pragma unroll
            for (int stride_k : {warp_size, warp_size/2, warp_size/4, warp_size/8}) {
                const int k0_start  = stride_k == warp_size ? 0 : nbatch_combine - nbatch_combine % (2*stride_k);
                const int k0_stop   =                             nbatch_combine - nbatch_combine % (1*stride_k);
                const int stride_jc = warp_size / stride_k;

                if (k0_start == k0_stop) {
                    continue;
                }

#pragma unroll
                for (int jc0_dst = 0; jc0_dst < ncols; jc0_dst += (nwarps/np)*stride_jc) {
                    const int jc_dst = jc0_dst + (threadIdx.y/np)*stride_jc + (stride_k == warp_size ? 0 : threadIdx.x / stride_k);

                    if (jc0_dst + (nwarps/np)*stride_jc > ncols && jc_dst >= ncols) {
                        break;
                    }

                    const int jc_tile_K = (jc_dst/cols_per_warp)*(np*cols_per_warp) + jc_dst % cols_per_warp;

                    const int j_dst = jc_dst / ncols2;
                    const int c_dst = jc_dst % ncols2;

                    if (!is_fixup && ((ncols1 > 1 && jt*ncols1 + j_dst >= int(ne01.z)) || (ncols2 > 1 && zt_gqa*ncols2 + c_dst >= gqa_ratio))) {
                        continue;
                    }

                    const float * meta_j = (const float *) tile_Q + jc_tile_K*tile_stride + nbatch_combine;
#pragma unroll
                    for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                        const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                        float2 dstk_val = make_float2(0.0f, 0.0f);
#pragma unroll
                        for (int ip = 0; ip < np; ++ip) {
                            const float KQ_crs = np == 1 ? 1.0f : meta_j[ip*cols_per_warp * tile_stride + 0];
                            const float2 dstk_val_add = __half22float2(tile_Q[(jc_tile_K + ip*cols_per_warp) * tile_stride + k]);
                            dstk_val.x += dstk_val_add.x*KQ_crs;
                            dstk_val.y += dstk_val_add.y*KQ_crs;
                        }

                        if (!needs_fixup && !is_fixup) {
                            const float KQ_rowsum_j = meta_j[1];
                            dstk_val.x /= KQ_rowsum_j;
                            dstk_val.y /= KQ_rowsum_j;
                        }

                        if (is_fixup) {
                            dstk_fixup_data[jc_dst*(DV/2) + k00 + k] = dstk_val;
                        } else {
                            dstk[((jt*ncols1 + j_dst)*ne02 + c_dst)*(DV/2) + k00 + k] = dstk_val;
                        }
                    }
                }
            }
        }
        if (np > 1) {
            __syncthreads();
        }
    }
#else
    GGML_UNUSED_VARS(Q_f2, K_row, V_row, pool, gtab, mask_h, kv_pos, q_pos, sinks_f, dstk, dstk_fixup,
        scale, slope, logit_softcap, ne01, ne02, gqa_ratio, ne11,
        stride_Q1, stride_Q2, nb11, nb21, nb_pool, stride_mask,
        jt, zt_gqa, kb0_start, kb0_stop, kb0_stride, hs);
    NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
}

// ------------------------------------------------------------------------------------------------------------------
// Kernel entry: every parameter of fattn_kernel_t in the same order, then the turbot inputs (SPEC 7.2).

template<int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap>
__launch_bounds__(ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols1*ncols2), ggml_cuda_fattn_mma_get_occupancy(DKQ, DV, ncols1*ncols2))
static __global__ void flash_attn_ext_turbot(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        const int32_t * __restrict__ kv_pos, // [TAG_FA_POS_MASK]
        const int32_t * __restrict__ q_pos,
        const int  * KV_max_ptr,
        float      * dst_ptr,
        float2     * dst_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33,
        const char    * pool_ptr,        // young pool data (layer), never nullptr (SPEC 9.4 keeps >= 64 rows)
        const int32_t * gtab_ptr,        // granule table, gtab[cell >> 6] = slot or -1
        const int64_t   nb_pool,         // pool->nb[1]
        const int64_t   turbot_desc0,    // packed layout, see turbot_head_state_of
        const int64_t   turbot_desc1,
        const int64_t   turbot_desc2,
        const int     * balance_bounds_ptr,     // [TAG_TURBOT_FA_BALANCE] seams [gridDim.x + 1], or nullptr (uniform)
        const int       balance_stripe) {       // [TAG_TURBOT_FA_STRIPE] blocks per output tile (>= 2), or 0
    ggml_cuda_pdl_sync(); // TODO optimize placement
#if defined(FLASH_ATTN_AVAILABLE) && (defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE))
    const char    * GGML_CUDA_RESTRICT Q        = Q_ptr;
    const char    * GGML_CUDA_RESTRICT K        = K_ptr;
    const char    * GGML_CUDA_RESTRICT V        = V_ptr;
    const char    * GGML_CUDA_RESTRICT mask     = mask_ptr;
    const char    * GGML_CUDA_RESTRICT sinks    = sinks_ptr;
    const int     * GGML_CUDA_RESTRICT KV_max   = KV_max_ptr;
    float         * GGML_CUDA_RESTRICT dst      = dst_ptr;
    float2        * GGML_CUDA_RESTRICT dst_meta = dst_meta_ptr;
    const char    * GGML_CUDA_RESTRICT pool     = pool_ptr;
    const int32_t * GGML_CUDA_RESTRICT gtab     = gtab_ptr;
    const int     * GGML_CUDA_RESTRICT balance_bounds = balance_bounds_ptr;

    static_assert(DKQ == 256 && DV == 256, "turbot kernels exist for D=256 only");

#ifdef VOLTA_MMA_AVAILABLE
    if (ncols1*ncols2 < 32) {
        NO_DEVICE_CODE;
        return;
    }
#endif // VOLTA_MMA_AVAILABLE

#if __CUDA_ARCH__ == GGML_CUDA_CC_TURING
    if (ncols1*ncols2 > 32) {
        NO_DEVICE_CODE;
        return;
    }
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_TURING

#if defined(AMD_WMMA_AVAILABLE)
    if (ncols1*ncols2 < 16 || ncols2 == 1 || DKQ > 128) {
        NO_DEVICE_CODE;
        return;
    }
#endif // defined(AMD_WMMA_AVAILABLE)

#if defined(AMD_MFMA_AVAILABLE)
    if (ncols1*ncols2 < 16 || DKQ > 256) {
        NO_DEVICE_CODE;
        return;
    }
#endif // defined(AMD_MFMA_AVAILABLE)

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int ncols     = ncols1 * ncols2;
    constexpr int nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, ncols);
    constexpr int nthreads  = ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols);
    constexpr int nwarps    = nthreads / warp_size;

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.

    const int stride_Q1   = nb01 / sizeof(float2);
    const int stride_Q2   = nb02 / sizeof(float2);
    const int stride_mask = nb31 / sizeof(half);

    const int iter_k     = (ne11      + (nbatch_fa - 1)) / nbatch_fa;
    const int iter_j     = (ne01.z    + (ncols1    - 1)) / ncols1;
    const int iter_z_gqa = (gqa_ratio + (ncols2    - 1)) / ncols2;

    // kbc == k block continuous, current index in continuous ijk space.
    // [TAG_TURBOT_FA_BALANCE] With a seam table the block's [kbc, kbc_stop) is the work-balanced slice the host
    // launched flash_attn_turbot_balance_bounds for; without one it is the uniform slice of launch_fattn. Everything
    // below (block_owns_tile_start, needs_fixup / is_fixup) derives from kbc and kbc_stop, so it follows either layout.
    //
    // [TAG_TURBOT_FA_STRIPE] Striped blocks, the default for the uniform-fixup layout (every output tile has `stripe`
    // >= 2 blocks). Block b = tile*stripe + j processes KV tiles j, j + stripe, j + 2*stripe, ... of output tile `tile`
    // instead of a contiguous run, so every block sees the same mix of old and young tiles (the young band is the
    // newest cells of every head) and no block waits on an all-young tail, without any pass over gtab. The softmax
    // accumulators do not care about the order of the KV tiles, so the partial results combine exactly as before:
    // block j = stripe - 1 takes the while loop with needs_fixup (kb0_start = j != 0) and writes dst, every other block
    // takes the final is_fixup call, and flash_attn_stream_k_fixup_uniform (unchanged) combines blocks b_last-1 .. b_first
    // into b_last. The KV_min / KV_max skips below keep kb0_start on the stripe.
    const int  bidx            = int(blockIdx.x);
    const int  stripe          = balance_stripe;
    const int  kb0_stride      = stripe > 0 ? stripe : 1;
    const bool stripe_is_fixup = stripe > 0 && bidx % stripe != stripe - 1;
    int       kbc      = stripe > 0     ? (bidx / stripe)*iter_k + bidx % stripe :
                         balance_bounds ? balance_bounds[blockIdx.x] :
                         int(int64_t(blockIdx.x + 0)*(iter_k*iter_j*iter_z_gqa*ne12*ne03) / gridDim.x);
    const int kbc_stop = stripe > 0     ? (bidx / stripe + 1)*iter_k :
                         balance_bounds ? balance_bounds[blockIdx.x + 1] :
                         int(int64_t(blockIdx.x + 1)*(iter_k*iter_j*iter_z_gqa*ne12*ne03) / gridDim.x);

    // If the seams of 2 CUDA blocks fall within an output tile their results need to be combined.
    // For this we need to track both the block that starts the tile (needs_fixup) and the block that finishes the tile (is_fixup).
    // In the most general case >2 seams can fall into the same tile.

    // kb0 == k start index when in the output tile.
    int kb0_start = kbc % iter_k;
    int kb0_stop  = min(iter_k, kb0_start + kbc_stop - kbc);

    while (kbc < kbc_stop && kb0_stop == iter_k && !stripe_is_fixup) {
        // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
        const int sequence =  kbc /(iter_k*iter_j*iter_z_gqa*ne12);
        const int z_KV     = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence)/(iter_k*iter_j*iter_z_gqa);
        const int zt_gqa   = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV)/(iter_k*iter_j);
        const int jt       = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV - iter_k*iter_j * zt_gqa) / iter_k;

        const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

        // [TAG_TURBOT_HEAD_STATE] site 1 of 2. Heads have different widths, so K and V are row bases (never nb12/nb22)
        // and every per-head offset comes from the head state.
        const turbot_head_state hs = turbot_head_state_of(z_KV, turbot_desc0, turbot_desc1, turbot_desc2);
        const char * const K_row = K + nb13*sequence;
        const char * const V_row = V + nb23*sequence;

        const float2 * Q_f2   = (const float2 *) (Q + nb03*sequence + nb02*zt_Q);
        const half   * mask_h = (ncols2 == 1 && !mask) || kv_pos ? nullptr :
            (const half *) (mask + nb33*(sequence % ne33));
        float2       * dstk   = ((float2 *) dst) + (sequence*ne01.z*ne02 + zt_Q) * (DV/2);

        const float * sinks_f = sinks ? (const float *) sinks + zt_Q : nullptr;

        const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, zt_Q, n_head_log2, m0, m1) : 1.0f;

        // [TAG_TURBOT_KVMIN_FIXUP] Whether this CUDA block owns the start of the output tile is block geometry and must be
        // decided BEFORE the [TAG_FA_KVMIN] skip below raises kb0_start. Otherwise a block that owns a whole tile whose
        // leading KV tiles are fully masked takes the needs_fixup variant: the tile is written without the rowsum
        // normalisation, no fixup kernel ever finishes it (the general fixup skips blocks starting on a tile boundary),
        // and its meta overwrites the slot of the block's real leading partial tile. The skipped KV tiles contribute
        // nothing for any query in the tile, so starting the accumulators at kb0_min and normalising here is exact.
        const bool block_owns_tile_start = kb0_start == 0;

        if (KV_max) {
            // [TAG_FA_KVMIN] the array holds {max, min} pairs
            const int kvb = 2*(sequence*iter_j + jt);
            kb0_stop = min(kb0_stop, KV_max[kvb + 0] / nbatch_fa);

            // Skip the fully-masked leading tiles. The tile loop runs its LAST iteration
            // unconditionally on the invariant "kb0_start is always < kb0_stop", so never
            // raise kb0_start past kb0_stop - 1: a block lying entirely below KV_min still
            // processes one zero-contribution tile instead of thousands.
            const int kb0_min = KV_max[kvb + 1] / nbatch_fa;
            if (stripe > 0) {
                // [TAG_TURBOT_FA_STRIPE] first tile of the stripe at or above KV_min; if none is left below KV_max,
                // one tile of the stripe that lies outside [KV_min, KV_max) (fully masked, zero contribution)
                if (kb0_min > kb0_start) {
                    kb0_start += ((kb0_min - kb0_start + stripe - 1) / stripe) * stripe;
                }
                if (kb0_start >= kb0_stop) {
                    kb0_start = kbc % iter_k;
                    kb0_stop  = kb0_start + 1;
                }
            } else if (kb0_min > kb0_start && kb0_start < kb0_stop) {
                kb0_start = min(kb0_min, kb0_stop - 1);
            }
        }
        constexpr bool is_fixup = false; // All but (potentially) the last iterations write their data to dst rather than the fixup buffer.
        if (block_owns_tile_start) {
            constexpr bool needs_fixup = false; // CUDA block is working on an entire tile.
            flash_attn_ext_turbot_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, needs_fixup, is_fixup>
                (Q_f2, K_row, V_row, pool, gtab, mask_h, kv_pos, q_pos, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
                 ne01, ne02, gqa_ratio, ne11, stride_Q1, stride_Q2, (int64_t) nb11, (int64_t) nb21, nb_pool, stride_mask,
                 jt, zt_gqa, kb0_start, kb0_stop, kb0_stride, hs);
        } else {
            constexpr bool needs_fixup = true; // CUDA block is missing the beginning of a tile.
            flash_attn_ext_turbot_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, needs_fixup, is_fixup>
                (Q_f2, K_row, V_row, pool, gtab, mask_h, kv_pos, q_pos, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
                 ne01, ne02, gqa_ratio, ne11, stride_Q1, stride_Q2, (int64_t) nb11, (int64_t) nb21, nb_pool, stride_mask,
                 jt, zt_gqa, kb0_start, kb0_stop, kb0_stride, hs);
        }

        kbc += iter_k;
        kbc -= kbc % iter_k;

        kb0_start = 0;
        kb0_stop  = min(iter_k, kbc_stop - kbc);
    }

    if (kbc >= kbc_stop) {
        return;
    }

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index.
    const int sequence =  kbc /(iter_k*iter_j*iter_z_gqa*ne12);
    const int z_KV     = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence)/(iter_k*iter_j*iter_z_gqa);
    const int zt_gqa   = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV)/(iter_k*iter_j);
    const int jt       = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV - iter_k*iter_j * zt_gqa) / iter_k;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    // [TAG_TURBOT_HEAD_STATE] site 2 of 2, the final is_fixup block: identical to site 1 by construction.
    const turbot_head_state hs = turbot_head_state_of(z_KV, turbot_desc0, turbot_desc1, turbot_desc2);
    const char * const K_row = K + nb13*sequence;
    const char * const V_row = V + nb23*sequence;

    const float2 * Q_f2   = (const float2 *) (Q + nb03*sequence + nb02*zt_Q);
    const half   * mask_h = (ncols2 == 1 && !mask) || kv_pos ? nullptr :
        (const half *) (mask + nb33*(sequence % ne33));
    float2       * dstk   = ((float2 *) dst) + (sequence*ne01.z*ne02 + zt_Q) * (DV/2);

    const float * sinks_f = sinks ? (const float *) sinks + zt_Q : nullptr;

    const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, zt_Q, n_head_log2, m0, m1) : 1.0f;

    if (KV_max) {
        // [TAG_FA_KVMIN] the array holds {max, min} pairs
        const int kvb = 2*(sequence*iter_j + jt);
        kb0_stop = min(kb0_stop, KV_max[kvb + 0] / nbatch_fa);

        // Skip the fully-masked leading tiles. The tile loop runs its LAST iteration
        // unconditionally on the invariant "kb0_start is always < kb0_stop", so never
        // raise kb0_start past kb0_stop - 1: a block lying entirely below KV_min still
        // processes one zero-contribution tile instead of thousands.
        const int kb0_min = KV_max[kvb + 1] / nbatch_fa;
        if (stripe > 0) {
            // [TAG_TURBOT_FA_STRIPE] as in the loop above
            if (kb0_min > kb0_start) {
                kb0_start += ((kb0_min - kb0_start + stripe - 1) / stripe) * stripe;
            }
            if (kb0_start >= kb0_stop) {
                kb0_start = kbc % iter_k;
                kb0_stop  = kb0_start + 1;
            }
        } else if (kb0_min > kb0_start && kb0_start < kb0_stop) {
            kb0_start = min(kb0_min, kb0_stop - 1);
        }
    }

    constexpr bool is_fixup = true; // Last index writes its data to fixup buffer to avoid data races with other blocks.
    constexpr bool needs_fixup = false;
    flash_attn_ext_turbot_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, needs_fixup, is_fixup>
        (Q_f2, K_row, V_row, pool, gtab, mask_h, kv_pos, q_pos, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
         ne01, ne02, gqa_ratio, ne11, stride_Q1, stride_Q2, (int64_t) nb11, (int64_t) nb21, nb_pool, stride_mask,
         jt, zt_gqa, kb0_start, kb0_stop, kb0_stride, hs);
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, kv_pos, q_pos, KV_max_ptr, dst_ptr, dst_meta_ptr, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33,
        pool_ptr, gtab_ptr, nb_pool, turbot_desc0, turbot_desc1, turbot_desc2, balance_bounds_ptr, balance_stripe);
    NO_DEVICE_CODE;
#endif // defined(FLASH_ATTN_AVAILABLE) && (defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE))
}

typedef void (* fattn_turbot_kernel_t)(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int32_t * __restrict__ kv_pos,
        const int32_t * __restrict__ q_pos,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33,
        const char    * __restrict__ pool,
        const int32_t * __restrict__ gtab,
        const int64_t nb_pool,
        const int64_t turbot_desc0,
        const int64_t turbot_desc1,
        const int64_t turbot_desc2,
        const int   * __restrict__ balance_bounds,
        const int     balance_stripe);

// ------------------------------------------------------------------------------------------------------------------
// [TAG_TURBOT_Q1_ROUTE] KV bounds scan over an explicit mask whose rows wrap like flash_attn_ext_f16_load_mask.
//
// flash_attn_mask_to_KV_max<ncols1> (fattn-common.cuh) reads mask rows jt*ncols1 + j for every j < ncols1, but a mask has
// exactly Q->ne[1] rows. When an instance is wider than the batch (Q = 1 on <4,8>, or the last tile of a Q that is not a
// multiple of ncols1) it reads past the mask. This copy wraps the row index with fastmodulo(jt*ncols1 + j, ne01), the
// same rows the kernel's mask load reads, so the AND over rows only sees real rows (a repeated row changes nothing).
// Launched with ncols1 = 1 for Q = 1 (one row per tile, no wrap needed), with the instance's ncols1 when
// Q % ncols1 != 0, and never otherwise: the common kernel keeps every other shape.
template <int ncols1>
__launch_bounds__(FATTN_KQ_STRIDE/2, 1)
static __global__ void flash_attn_turbot_mask_to_KV_max(
        const half2 * mask_ptr, int * KV_max_ptr, const int ne30, const int64_t s31, const int64_t s33, const uint3 ne01) {
    const half2 * GGML_CUDA_RESTRICT mask   = mask_ptr;
    int         * GGML_CUDA_RESTRICT KV_max = KV_max_ptr;

    const int ne31     = gridDim.x;
    const int tid      = threadIdx.x;
    const int sequence = blockIdx.y;
    const int jt       = blockIdx.x;

    mask += sequence*s33;

    __shared__ int buf_iw[WARP_SIZE];
    if (tid < WARP_SIZE) {
        buf_iw[tid] = 1;
    }
    ggml_cuda_pdl_sync();
    __syncthreads();

    int KV_max_sj = (ne30 - 1) * FATTN_KQ_STRIDE;
    for (; KV_max_sj >= 0; KV_max_sj -= FATTN_KQ_STRIDE) {
        int all_inf = 1;

#pragma unroll
        for (int j = 0; j < ncols1; ++j) {
            const int64_t row = ncols1 == 1 ? int64_t(jt) : int64_t(fastmodulo(jt*ncols1 + j, ne01));
            const float2 tmp = __half22float2(mask[row*s31 + KV_max_sj/2 + tid]);
            all_inf = all_inf && int(isinf(tmp.x)) && int(isinf(tmp.y));
        }

        all_inf = warp_reduce_all(all_inf);
        if (tid % WARP_SIZE == 0) {
            buf_iw[tid / WARP_SIZE] = all_inf;
        }
        __syncthreads();
        all_inf = buf_iw[tid % WARP_SIZE];
        __syncthreads();
        all_inf = warp_reduce_all(all_inf);

        if (!all_inf) {
            break;
        }
    }

    // Walk back the last decrement, as in flash_attn_mask_to_KV_max.
    KV_max_sj += FATTN_KQ_STRIDE;

    // [TAG_FA_KVMIN] the first tile that is not entirely masked
    int KV_min_sj = 0;
    for (; KV_min_sj < KV_max_sj; KV_min_sj += FATTN_KQ_STRIDE) {
        int all_inf = 1;

#pragma unroll
        for (int j = 0; j < ncols1; ++j) {
            const int64_t row = ncols1 == 1 ? int64_t(jt) : int64_t(fastmodulo(jt*ncols1 + j, ne01));
            const float2 tmp = __half22float2(mask[row*s31 + KV_min_sj/2 + tid]);
            all_inf = all_inf && int(isinf(tmp.x)) && int(isinf(tmp.y));
        }

        all_inf = warp_reduce_all(all_inf);
        if (tid % WARP_SIZE == 0) {
            buf_iw[tid / WARP_SIZE] = all_inf;
        }
        __syncthreads();
        all_inf = buf_iw[tid % WARP_SIZE];
        __syncthreads();
        all_inf = warp_reduce_all(all_inf);

        if (!all_inf) {
            break;
        }
    }

    if (threadIdx.x != 0) {
        return;
    }

    KV_max[2*(sequence*ne31 + jt) + 0] = KV_max_sj;
    KV_max[2*(sequence*ne31 + jt) + 1] = KV_min_sj;
}

// ------------------------------------------------------------------------------------------------------------------
// [TAG_TURBOT_FA_BALANCE] Work-balanced stream_k blocks (read-speed plan item 2, SPEC section 13 deviation).
//
// launch_fattn slices the continuous unit space kbc = (((sequence*ne12 + z_KV)*iter_z_gqa + zt_gqa)*iter_j + jt)*iter_k
// + kb into gridDim.x equal slices, and every slice launches at once, so the op waits for its slowest block. With a
// turbot cache the cost of a unit depends on its head (width) and on whether its granule is young, and the young band
// is the newest cells of every head: the tail blocks of each head are all young and gate the op. Here the slices hold
// equal estimated WORK instead:
//
//   weight(unit) = w_old[z_KV] + (w_young[z_KV] - w_old[z_KV]) * young(granule of kb),   1 <= w <= 0xFFFF
//
// 1. flash_attn_turbot_young_prefix: prefix[kb] = young KV tiles among tiles [0, kb), kb = 0..iter_k (one pass over
//    gtab, 32 tiles per step through a lane prefix sum).
// 2. flash_attn_turbot_balance_bounds: seams[i] = smallest kbc with W(kbc) >= floor(i * W_total / nblocks), W(kbc)
//    the weight of the units before kbc. Every output tile of a head has the same weights, so the head and the
//    output tile follow by division and only kb is a binary search over the prefix (fixed trip count, lanes in
//    parallel: lane t computes seam i0 + t). seams[0] = 0, seams[nblocks] = every unit.
// 3. The kernel entry reads [seams[b], seams[b+1]); block_owns_tile_start, needs_fixup and is_fixup follow from them.
// 4. The fixup combines exactly like flash_attn_stream_k_fixup_general (same order, same arithmetic), with the seams
//    read from the table: flash_attn_turbot_stream_k_fixup_general (one launch block per stream_k block, default) or
//    flash_attn_turbot_stream_k_fixup_tile (one launch block per output tile, LLAMA_TURBOT_FA_BALANCE_FIXUP=tile).
//
// The weights never affect values, only where the seams fall; every unit is still processed exactly once and a seam
// changes only the last float bits of the combine. Units are strictly positive, so W is strictly increasing and the
// seams are non-decreasing; a block may be empty (seams[b] == seams[b+1]) and both fixups skip empty blocks.
// Launched only for general-fixup layouts (prefill): uniform-fixup layouts (every decode shape) take striped blocks
// instead ([TAG_TURBOT_FA_STRIPE], kernel entry), because at decode the two passes cost more than the young penalty
// they remove (LLAMA_TURBOT_FA_BALANCE_DIAG, 131K nb 4: prefix +32 us, seams +16 us, E:/kv-turbot/fix5). Layouts whose
// blocks cover whole output tiles keep the uniform slice.

template <int log2_tpg>   // KV tiles per granule = 1 << log2_tpg (nbatch_fa 64: 0, 32: 1)
__launch_bounds__(WARP_SIZE, 1)
static __global__ void flash_attn_turbot_young_prefix(
        const int32_t * gtab_ptr, int * prefix_ptr, const int iter_k) {
    const int32_t * GGML_CUDA_RESTRICT gtab   = gtab_ptr;
    int           * GGML_CUDA_RESTRICT prefix = prefix_ptr;

    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();

    if (tid == 0) {
        prefix[0] = 0;
    }

    // Lanes load tiles kb0 + tid; the step count is uniform. A tile past iter_k loads nothing and counts 0.
    // (iter_k - 1) >> log2_tpg < gtab->ne[0], asserted on the host.
    int acc = 0;
    for (int kb0 = 0; kb0 < iter_k; kb0 += WARP_SIZE) {
        const int kb       = kb0 + tid;
        const int in_range = int(kb < iter_k);
        const int young    = in_range ? int(gtab[kb >> log2_tpg] >= 0) : 0;
        const int run      = warp_prefix_inclusive_sum<int>(young);
        if (in_range) {
            prefix[kb + 1] = acc + run;
        }
        acc += warp_reduce_sum(young);
    }
}

// cumulative weight of KV tiles [0, kb) of one output tile of a head
static __device__ __forceinline__ int64_t turbot_balance_cum(
        const int * const __restrict__ prefix, const int kb, const int64_t w_old, const int64_t w_delta) {
    return w_old*kb + w_delta*prefix[kb];
}

template <int n_head>
__launch_bounds__(WARP_SIZE, 1)
static __global__ void flash_attn_turbot_balance_bounds(
        const int * prefix_ptr,      // [iter_k + 1], flash_attn_turbot_young_prefix
        int       * seams_ptr,       // [nblocks + 1]
        const int nblocks,
        const int iter_k,
        const int nit,               // ceil(log2(iter_k)): the kb search interval is <= 1 after nit halvings
        const int n_o,               // output tiles per head: iter_j * iter_z_gqa
        const int n_seq,
        const int64_t w_old_pack,    // 16 bits per head
        const int64_t w_young_pack) {
    const int * GGML_CUDA_RESTRICT prefix = prefix_ptr;
    int       * GGML_CUDA_RESTRICT seams  = seams_ptr;

    static_assert(n_head == 4, "turbot balance packs 4 head weights");

    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();

    // Per-head totals, uniform across lanes.
    const int64_t y_all = prefix[iter_k];
    int64_t w_old   [n_head];
    int64_t w_delta [n_head];
    int64_t w_tile  [n_head];
    int64_t head_end[n_head];
    int64_t acc = 0;
    for (int h = 0; h < n_head; ++h) {
        w_old[h]    = int64_t(((uint64_t) w_old_pack   >> (16*h)) & 0xFFFFu);
        w_delta[h]  = int64_t(((uint64_t) w_young_pack >> (16*h)) & 0xFFFFu) - w_old[h];
        w_tile[h]   = w_old[h]*iter_k + w_delta[h]*y_all;   // >= iter_k: every unit weighs >= 1
        acc        += w_tile[h]*n_o;
        head_end[h] = acc;
    }
    const int64_t w_seq   = acc;
    const int64_t w_total = w_seq*n_seq;
    const int64_t n_units = int64_t(iter_k)*n_o*n_head*n_seq;

    for (int i0 = 0; i0 <= nblocks; i0 += WARP_SIZE) {
        const int     i   = i0 + tid;
        const int64_t t   = int64_t(i)*w_total / nblocks;
        // Clamp so every lane stays in range (i past nblocks, or t == 0); those lanes' results are replaced below.
        const int64_t tc  = t <= 0 ? 1 : (t < w_total ? t : w_total);
        const int64_t seq = (tc - 1) / w_seq;
        const int64_t ts  = tc - seq*w_seq;                                                  // (0, w_seq]
        const int     h   = int(ts > head_end[0]) + int(ts > head_end[1]) + int(ts > head_end[2]);
        const int64_t hb  = h == 0 ? 0         : (h == 1 ? head_end[0] : (h == 2 ? head_end[1] : head_end[2]));
        const int64_t wt  = h == 0 ? w_tile[0] : (h == 1 ? w_tile[1]   : (h == 2 ? w_tile[2]   : w_tile[3]));
        const int64_t wo  = h == 0 ? w_old[0]  : (h == 1 ? w_old[1]    : (h == 2 ? w_old[2]    : w_old[3]));
        const int64_t wd  = h == 0 ? w_delta[0]: (h == 1 ? w_delta[1]  : (h == 2 ? w_delta[2]  : w_delta[3]));
        const int64_t th  = ts - hb;                                                         // (0, wt*n_o]
        const int64_t o   = (th - 1) / wt;                                                   // output tile in the head
        const int64_t rem = th - o*wt;                                                       // (0, wt]

        // smallest kb in (0, iter_k] with cum(kb) >= rem; invariant cum(lo) < rem <= cum(hi)
        int lo = 0;
        int hi = iter_k;
        for (int it = 0; it < nit; ++it) {
            const int  mid = (lo + hi) >> 1;
            const bool ge  = turbot_balance_cum(prefix, mid, wo, wd) >= rem;
            hi = ge ? mid : hi;
            lo = ge ? lo  : mid;
        }

        const int64_t kbc  = ((seq*n_head + h)*n_o + o)*iter_k + hi;
        const int64_t seam = t <= 0 ? 0 : (i >= nblocks ? n_units : kbc);
        if (i <= nblocks) {
            seams[i] = int(seam);
        }
    }
}

// Copy of flash_attn_stream_k_fixup_general (fattn-common.cuh) with the block seams read from the balance table instead
// of int64_t(bidx)*total_work / gridDim.x. The combine is unchanged.
template <int D, int ncols1, int ncols2> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_turbot_stream_k_fixup_general(
        float * dst_ptr,
        const float2 * dst_fixup_ptr,
        const int * seams_ptr,
        const int ne01, const int ne02,
        const int gqa_ratio,
        const uint3 fd_iter_k_j_z_ne12,
        const uint3 fd_iter_k_j_z,
        const uint3 fd_iter_k_j,
        const uint3 fd_iter_k) {
    float        * GGML_CUDA_RESTRICT dst       = dst_ptr;
    const float2 * GGML_CUDA_RESTRICT dst_fixup = dst_fixup_ptr;
    const int    * GGML_CUDA_RESTRICT seams     = seams_ptr;
    constexpr int ncols = ncols1*ncols2;

    const int bidx0 = blockIdx.x;
    const int j     = blockIdx.y;
    const int c     = blockIdx.z;
    const int jc    = j*ncols2 + c;
    const int tid   = threadIdx.x;

    const float * dst_fixup_data = ((const float *) dst_fixup) + gridDim.x*(2*2*ncols);

    // The seams are data: synchronize before the first load (the common kernel syncs later, its seams are arithmetic).
    ggml_cuda_pdl_sync();

    const int kbc0      = seams[bidx0];
    const int kbc0_stop = seams[bidx0 + 1];

    const bool did_not_have_any_data   = kbc0 == kbc0_stop;
    const bool wrote_beginning_of_tile = fastmodulo(kbc0, fd_iter_k) == 0;
    const bool did_not_write_last      = fastdiv(kbc0, fd_iter_k) == fastdiv(kbc0_stop, fd_iter_k) && fastmodulo(kbc0_stop, fd_iter_k) != 0;
    if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        return;
    }

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    const uint2 dm0 = fast_div_modulo(kbc0, fd_iter_k_j_z_ne12);
    const uint2 dm1 = fast_div_modulo(dm0.y, fd_iter_k_j_z);
    const uint2 dm2 = fast_div_modulo(dm1.y, fd_iter_k_j);
    const uint2 dm3 = fast_div_modulo(dm2.y, fd_iter_k);

    const int sequence = dm0.x;
    const int z_KV     = dm1.x;
    const int zt_gqa   = dm2.x;
    const int jt       = dm3.x;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    if (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_ratio) {
        return;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    // Load the partial result that needs a fixup:
    float dst_val = 0.0f;
    float max_val = 0.0f;
    float rowsum  = 0.0f;
    {
        dst_val = *dst;

        const float2 tmp = dst_fixup[bidx0*ncols + jc];
        max_val = tmp.x;
        rowsum  = tmp.y;
    }

    // Iterate over previous blocks and compute the combined results.
    // All CUDA blocks that get here must have a previous block that needs a fixup.
    const int tile_kbc0 = fastdiv(kbc0, fd_iter_k);
    int bidx = bidx0 - 1;
    int kbc_stop = kbc0;
    while(true) {
        const int kbc = seams[bidx];
        if (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            continue;
        }

        const float dst_add = dst_fixup_data[bidx*ncols*D + jc*D + tid];

        const float2 tmp = dst_fixup[(gridDim.x + bidx)*ncols + jc];

        // Scale the current and new value accumulators depending on the max. values.
        const float max_val_new = fmaxf(max_val, tmp.x);

        const float diff_val = max_val - max_val_new;
        const float diff_add = tmp.x   - max_val_new;

        const float scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_val) : 0.0f;
        const float scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val*rowsum  + scale_add*tmp.y;

        max_val = max_val_new;

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        if (fastmodulo(kbc, fd_iter_k) == 0 || fastdiv(kbc, fd_iter_k) < tile_kbc0) {
            break;
        }
        bidx--;
        kbc_stop = kbc;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

// Same combine, one launch block per output tile (the flash_attn_stream_k_fixup_uniform launch shape) instead of one
// per stream_k block: the block holding the tile's last KV tile is found by binary search over the seams. If that
// block starts at or before the tile start it wrote the tile normalised and nothing is done; otherwise it wrote the
// tile's end with needs_fixup, and every earlier non-empty block down to the one holding the tile start wrote an
// is_fixup part of the tile.
template <int D, int ncols1, int ncols2> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_turbot_stream_k_fixup_tile(
        float * dst_ptr,
        const float2 * dst_fixup_ptr,
        const int * seams_ptr,
        const int ne01, const int ne02,
        const int gqa_ratio,
        const int nblocks,
        const int iter_k,
        const uint3 fd_iter_j_z_ne12,
        const uint3 fd_iter_j_z,
        const uint3 fd_iter_j) {
    constexpr int ncols = ncols1*ncols2;
    ggml_cuda_pdl_lc();
    float        * GGML_CUDA_RESTRICT dst       = dst_ptr;
    const float2 * GGML_CUDA_RESTRICT dst_fixup = dst_fixup_ptr;
    const int    * GGML_CUDA_RESTRICT seams     = seams_ptr;

    const int tile_idx = blockIdx.x; // One block per output tile.
    const int j        = blockIdx.y;
    const int c        = blockIdx.z;
    const int jc       = j*ncols2 + c;
    const int tid      = threadIdx.x;

    const float * dst_fixup_data = ((const float *) dst_fixup) + nblocks*(2*2*ncols);

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    const uint2 dm0 = fast_div_modulo(tile_idx, fd_iter_j_z_ne12);
    const uint2 dm1 = fast_div_modulo(dm0.y,    fd_iter_j_z);
    const uint2 dm2 = fast_div_modulo(dm1.y,    fd_iter_j);

    const int sequence = dm0.x;
    const int z_KV     = dm1.x;
    const int zt_gqa   = dm2.x;
    const int jt       = dm2.y;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    if (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_ratio) {
        return;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    ggml_cuda_pdl_sync();

    const int kbc_first = tile_idx*iter_k;
    const int kbc_last  = kbc_first + iter_k - 1;

    // b_last: seams[b_last] <= kbc_last < seams[b_last + 1]. seams[0] == 0 and seams[nblocks] == every unit bound it.
    int lo = 0;
    int hi = nblocks;
    while (hi - lo > 1) {
        const int mid = (lo + hi) >> 1;
        if (seams[mid] <= kbc_last) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    const int b_last     = lo;
    const int kbc_b_last = seams[b_last];
    if (kbc_b_last <= kbc_first) {
        return; // b_last owns the tile start: it wrote the whole tile with the rowsum normalisation.
    }

    // Load the partial result that needs a fixup (b_last started mid tile and finished it).
    float dst_val = *dst;
    float max_val;
    float rowsum;
    {
        const float2 tmp = dst_fixup[b_last*ncols + jc];
        max_val = tmp.x;
        rowsum  = tmp.y;
    }

    // Combine with every earlier non-empty block of this tile, newest first, as flash_attn_stream_k_fixup_general.
    int bidx     = b_last - 1;
    int kbc_stop = kbc_b_last;
    while (true) {
        const int kbc = seams[bidx];
        if (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            continue;
        }

        const float dst_add = dst_fixup_data[bidx*ncols*D + jc*D + tid];

        const float2 tmp = dst_fixup[(nblocks + bidx)*ncols + jc];

        const float max_val_new = fmaxf(max_val, tmp.x);

        const float diff_val = max_val - max_val_new;
        const float diff_add = tmp.x   - max_val_new;

        const float scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_val) : 0.0f;
        const float scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val*rowsum  + scale_add*tmp.y;

        max_val = max_val_new;

        // This block holds the tile start: done.
        if (kbc <= kbc_first) {
            break;
        }
        bidx--;
        kbc_stop = kbc;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

// ------------------------------------------------------------------------------------------------------------------
// Host side.

// [TAG_TURBOT_FA_DEBUG] LLAMA_TURBOT_FA_DEBUG=1 prints the stream_k block layout of launch_fattn_turbot once per distinct
// shape: the op waits for its slowest block, so this is what the read speed analysis needs to see.
static __host__ bool ggml_cuda_fattn_turbot_debug_on() {
    static const bool on = [] {
        const char * e = getenv("LLAMA_TURBOT_FA_DEBUG");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

// Params, layout and every tensor property the kernel relies on. supports_op already refused anything else, so a
// failure here is a graph-construction bug and aborts.
static __host__ ggml_turbot_layer ggml_cuda_fattn_turbot_layer_of(const ggml_tensor * dst) {
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * pool = dst->src[7];
    const ggml_tensor * gtab = dst->src[8];

    ggml_turbot_op_params p;
    ggml_turbot_layer     l;
    GGML_ASSERT(ggml_turbot_op_params_get(dst, &p) && p.side == GGML_TURBOT_SIDE_BOTH);
    GGML_ASSERT(ggml_turbot_layer_from_op_params(&p, &l));

    GGML_ASSERT(ggml_turbot_is_type(K->type) && ggml_turbot_is_type(V->type));
    GGML_ASSERT(ggml_turbot_s_of_type(K->type) == l.k.s && ggml_turbot_s_of_type(V->type) == l.v.s);
    GGML_ASSERT(K->ne[0] == GGML_TURBOT_HEAD_DIM && V->ne[0] == GGML_TURBOT_HEAD_DIM);
    GGML_ASSERT(K->ne[2] == GGML_TURBOT_N_HEAD   && V->ne[2] == GGML_TURBOT_N_HEAD);
    GGML_ASSERT(K->ne[3] == 1 && V->ne[3] == 1);
    GGML_ASSERT(K->nb[1] == (size_t) l.k.base_row_bytes && V->nb[1] == (size_t) l.v.base_row_bytes);

    GGML_ASSERT(pool != nullptr && pool->type == GGML_TYPE_I8 && pool->nb[0] == 1);
    GGML_ASSERT(pool->ne[0] == (int64_t) l.pool_row_bytes && pool->ne[1] >= GGML_TURBOT_POOL_MIN_ROWS);

    GGML_ASSERT(gtab != nullptr && gtab->type == GGML_TYPE_I32 && ggml_is_contiguous(gtab));
    GGML_ASSERT(gtab->ne[0]*GGML_TURBOT_GRANULE >= K->ne[1]);
    return l;
}

static __host__ void ggml_cuda_fattn_turbot_pack_side(const ggml_turbot_side & sd, uint64_t & widths, uint64_t & offs) {
    widths = 0;
    offs   = 0;
    for (int h = 0; h < GGML_TURBOT_N_HEAD; ++h) {
        GGML_ASSERT(sd.base_off[h] % 32 == 0 && sd.base_off[h]/32 <= 0x1F);
        GGML_ASSERT(sd.young_off[h] % 32 == 0 && sd.young_off[h]/32 <= 0x1F);
        widths |= (uint64_t) (sd.b[h] & 0xF) << (4*h);
        widths |= (uint64_t) (sd.y[h] & 0xF) << (16 + 4*h);
        offs   |= (uint64_t) (sd.base_off[h]/32)  << (5*h);
        offs   |= (uint64_t) (sd.young_off[h]/32) << (20 + 5*h);
    }
    GGML_ASSERT(sd.base_gain_off % 32 == 0 && sd.base_gain_off/32 <= 0x1F);
    GGML_ASSERT(sd.young_gain_off % 32 == 0 && sd.young_gain_off/32 <= 0x1F);
    offs |= (uint64_t) (sd.base_gain_off/32)  << 40;
    offs |= (uint64_t) (sd.young_gain_off/32) << 45;
}

// [TAG_TURBOT_CT_WIDTH] LLAMA_TURBOT_FA_GENERIC=1 forces the runtime-width loaders (A/B and a bit-identity check
// against the compile-time widths). Read once; it rides in a spare desc1 bit, never a __device__ global, because a
// host write to a global races with kernels still queued for earlier layers.
static __host__ bool ggml_cuda_fattn_turbot_generic_on() {
    static const bool on = [] {
        const char * e = getenv("LLAMA_TURBOT_FA_GENERIC");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

// Inverse of turbot_head_state_of.
static __host__ void ggml_cuda_fattn_turbot_pack_desc(const ggml_turbot_layer & l, int64_t & desc0, int64_t & desc1, int64_t & desc2) {
    uint64_t wk = 0, ok = 0, wv = 0, ov = 0;
    ggml_cuda_fattn_turbot_pack_side(l.k, wk, ok);
    ggml_cuda_fattn_turbot_pack_side(l.v, wv, ov);
    GGML_ASSERT(l.pool_v_off % 16 == 0 && l.pool_v_off/16 <= 0x3F);
    desc0 = (int64_t) (wk | (wv << 32));
    desc1 = (int64_t) (ok | ((uint64_t) (ggml_cuda_fattn_turbot_generic_on() ? 1 : 0) << 50));
    desc2 = (int64_t) (ov | ((uint64_t) (l.pool_v_off/16) << 50));
}

// [TAG_TURBOT_FA_BALANCE] Read once.
//   LLAMA_TURBOT_FA_BALANCE=0          uniform stream_k slice and the common fixup kernels (A/B against the default);
//   LLAMA_TURBOT_FA_BALANCE_SEED=<n>   per-shape random weights in [1, 4096] instead of the measured ones, so the seams
//                                      land anywhere, young may weigh less than old and blocks may be empty (tests);
//   LLAMA_TURBOT_FA_BALANCE_FIXUP=tile the per-output-tile fixup instead of the per-block copy of the general one.
//   LLAMA_TURBOT_FA_BALANCE_DIAG=1|2   cost diagnostics: launch the prefix kernel (1) or prefix and seams (2) but keep
//                                      the uniform slice and the common fixups, so perf shows what the passes cost.
//   LLAMA_TURBOT_FA_BALANCE_STRIPE=0   no striped blocks ([TAG_TURBOT_FA_STRIPE]): uniform-fixup layouts take the seam
//                                      table too (the plan-item-2 design, for A/B and for the seeded seam tests).
struct ggml_cuda_fattn_turbot_balance_cfg {
    int      mode;         // 0 off, 1 measured weights, 2 seeded random weights
    uint64_t seed;
    bool     fixup_tile;
    int      diag;         // 0 normal, 1 prefix pass only, 2 prefix + seam passes; seams unused when diag != 0
    bool     stripe;       // striped blocks for uniform-fixup layouts
};

static __host__ const ggml_cuda_fattn_turbot_balance_cfg & ggml_cuda_fattn_turbot_balance_cfg_get() {
    static const ggml_cuda_fattn_turbot_balance_cfg cfg = [] {
        ggml_cuda_fattn_turbot_balance_cfg r = { 1, 0, false, 0, true };
        const char * st = getenv("LLAMA_TURBOT_FA_BALANCE_STRIPE");
        r.stripe = !(st != nullptr && strcmp(st, "0") == 0);
        const char * dg = getenv("LLAMA_TURBOT_FA_BALANCE_DIAG");
        r.diag = dg != nullptr ? atoi(dg) : 0;
        const char * e = getenv("LLAMA_TURBOT_FA_BALANCE");
        if (e != nullptr && strcmp(e, "0") == 0) {
            r.mode = 0;
        }
        const char * s = getenv("LLAMA_TURBOT_FA_BALANCE_SEED");
        if (r.mode != 0 && s != nullptr && s[0] != '\0') {
            r.mode = 2;
            r.seed = strtoull(s, nullptr, 0);
        }
        const char * f = getenv("LLAMA_TURBOT_FA_BALANCE_FIXUP");
        r.fixup_tile = f != nullptr && strcmp(f, "tile") == 0;
        return r;
    }();
    return cfg;
}

static __host__ uint64_t ggml_cuda_fattn_turbot_splitmix64(uint64_t & s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// Work weight of one KV tile of each head, 16 bits per head: w_old for an old granule, w_young for a young one. Per
// side 100, plus a b = 6 surcharge (the float LUT old path), plus young_extra when the granule is young; a head is
// K + V. Derived from the gate B0 logs (us/op ratios between cells of the same instance, so the attention compute
// cancels):
//   ncols 32, compile-time widths (E:/kv-turbot/fix4/b0_table.txt): L23 131K nb 4 band16k / old - 1 = 0.22 at the
//     slowest (young) block -> +22 per side; UNIFORM6 old / UNIFORM5 old 1.007 -> b6 +1.
//   ncols >= 64, compile-time widths: 32K all-young / old 1.099 and 131K band64k / old 1.090 (half the cells young,
//     half-tile blocks) -> +10; L23 old / UNIFORM5 old 1.020 at 131K nb 512 -> b6 +4.
//   runtime-width loaders (ncols <= 16, or LLAMA_TURBOT_FA_GENERIC=1; the round-1 codegen, E:/kv-turbot/fix2):
//     L23 nb 4 band16k / old 1.41 and <2,8> nb 2 1.53 -> +45; UNIFORM6 old / UNIFORM5 old 1.24 -> b6 +24; at
//     ncols >= 64 32K all-young / old 1.27 -> +25, L23 old / UNIFORM5 old 1.04 -> b6 +8.
// Wrong weights cost speed only, never values.
static __host__ void ggml_cuda_fattn_turbot_balance_weights(
        const ggml_turbot_layer & l, const int ncols, const uint64_t shape_key, int64_t & w_old, int64_t & w_young) {
    const ggml_cuda_fattn_turbot_balance_cfg & cfg = ggml_cuda_fattn_turbot_balance_cfg_get();
    uint64_t wo = 0;
    uint64_t wy = 0;
    if (cfg.mode == 2) {
        uint64_t s = cfg.seed ^ (shape_key * 0xD6E8FEB86659FD93ull);
        for (int h = 0; h < GGML_TURBOT_N_HEAD; ++h) {
            const uint64_t r  = ggml_cuda_fattn_turbot_splitmix64(s);
            const uint64_t ho = 1 + (((r >>  0) & 0xFFFu) >> ((r >> 12) % 12));   // log-spread in [1, 4096]
            const uint64_t hy = 1 + (((r >> 16) & 0xFFFu) >> ((r >> 28) % 12));
            wo |= ho << (16*h);
            wy |= hy << (16*h);
        }
    } else {
        const bool ct          = ncols >= TURBOT_CT_WIDTH_MIN_NCOLS && !ggml_cuda_fattn_turbot_generic_on();
        const int  b6_extra    = ct ? (ncols >= 64 ?  4 :  1) : (ncols >= 64 ?  8 : 24);
        const int  young_extra = ct ? (ncols >= 64 ? 10 : 22) : (ncols >= 64 ? 25 : 45);
        for (int h = 0; h < GGML_TURBOT_N_HEAD; ++h) {
            const int ho = 200 + (l.k.b[h] == 6 ? b6_extra : 0) + (l.v.b[h] == 6 ? b6_extra : 0);
            const int hy = ho + 2*young_extra;
            wo |= (uint64_t) ho << (16*h);
            wy |= (uint64_t) hy << (16*h);
        }
    }
    w_old   = (int64_t) wo;
    w_young = (int64_t) wy;
}

// Copy of launch_fattn (fattn-common.cuh) without the F16 conversion: same asserts, same KV_max scans with the same
// gates, same stream_k block layout and fixup launches, plus the turbot kernel arguments.
template <int DV, int ncols1, int ncols2>
static void launch_fattn_turbot(
    ggml_backend_cuda_context & ctx, ggml_tensor * dst, fattn_turbot_kernel_t fattn_kernel, const int nwarps, const size_t nbytes_shared,
    const int nbatch_fa, const int warp_size = WARP_SIZE
) {
    constexpr int ncols = ncols1 * ncols2;
    constexpr bool stream_k = true;   // what ggml_cuda_flash_attn_ext_mma_f16_case passes

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    const ggml_tensor * kv_pos_t = dst->src[5];   // [TAG_FA_POS_MASK]
    const ggml_tensor * q_pos_t  = dst->src[6];
    const ggml_tensor * pool_t   = dst->src[7];   // [TAG_TURBOT]
    const ggml_tensor * gtab_t   = dst->src[8];

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    GGML_ASSERT(Q->nb[0] == ggml_element_size(Q));
    GGML_ASSERT(K->nb[0] == ggml_element_size(K));
    GGML_ASSERT(V->nb[0] == ggml_element_size(V));

    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);
    GGML_ASSERT(Q->ne[3] == 1);

    const ggml_turbot_layer layer = ggml_cuda_fattn_turbot_layer_of(dst);
    int64_t turbot_desc0 = 0;
    int64_t turbot_desc1 = 0;
    int64_t turbot_desc2 = 0;
    ggml_cuda_fattn_turbot_pack_desc(layer, turbot_desc0, turbot_desc1, turbot_desc2);

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t main_stream = ctx.stream();
    const int id  = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[id].cc;
    const int nsm = ggml_cuda_info().devices[id].nsm;

    ggml_cuda_pool_alloc<int>    KV_max(pool);
    ggml_cuda_pool_alloc<float>  dst_tmp(pool);
    ggml_cuda_pool_alloc<float2> dst_tmp_meta(pool);

    // Base rows are read in place: no F16 scratch exists for a turbot cache (SPEC decision 10).
    const char * K_data = (const char *) K->data;
    const size_t nb11 = K->nb[1];
    const size_t nb12 = K->nb[2];
    const size_t nb13 = K->nb[3];

    const char * V_data = (const char *) V->data;
    const size_t nb21 = V->nb[1];
    const size_t nb22 = V->nb[2];
    const size_t nb23 = V->nb[3];

    const int ntiles_x     = ((Q->ne[1] + ncols1 - 1) / ncols1);
    const int gqa_ratio    = Q->ne[2] / K->ne[2];
    const int ntiles_z_gqa = ((gqa_ratio + ncols2 - 1) / ncols2);
    const int ntiles_dst   = ntiles_x * ntiles_z_gqa * K->ne[2] * Q->ne[3];

    // [TAG_FA_KVMAX_UNIFIED] same gates as launch_fattn, see the notes there.
    static const int kvmax_min_kv = [] {
        const char * e = getenv("FA_KVMAX_MIN_KV");
        const int    v = e ? atoi(e) : -1;
        return v >= 0 ? v : 4096;
    }();
    const bool kvmax_worth_it = kvmax_min_kv > 0 && K->ne[1] >= kvmax_min_kv;

    const char * kv_scan = "none";   // [TAG_TURBOT_FA_DEBUG]
    if (mask && K->ne[1] % FATTN_KQ_STRIDE == 0 && (Q->ne[1] >= 1024 || Q->ne[3] > 1 || kvmax_worth_it)) {
        const int64_t s31 = mask->nb[1] / sizeof(half2);
        const int64_t s33 = mask->nb[3] / sizeof(half2);

        const dim3 blocks_num_KV_max(ntiles_x, Q->ne[3], 1);
        const dim3 block_dim_KV_max(FATTN_KQ_STRIDE/2, 1, 1);

        const int ne_KV_max = blocks_num_KV_max.x*blocks_num_KV_max.y;
        const int iter_k = K->ne[1] / FATTN_KQ_STRIDE;

        KV_max.alloc(2*ne_KV_max);   // [TAG_FA_KVMIN] {max, min} per entry
        ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_KV_max, block_dim_KV_max, 0, main_stream);
        // [TAG_TURBOT_Q1_ROUTE] the mask has Q->ne[1] rows; never scan rows past them
        if (Q->ne[1] == 1 && ncols1 > 1) {
            kv_scan = "turbot<1>";
            ggml_cuda_kernel_launch(flash_attn_turbot_mask_to_KV_max<1>, launch_params,
                (const half2 *) mask->data, KV_max.ptr, iter_k, s31, s33, init_fastdiv_values(Q->ne[1]));
        } else if (Q->ne[1] % ncols1 != 0) {
            kv_scan = "turbot<ncols1> wrap";
            ggml_cuda_kernel_launch(flash_attn_turbot_mask_to_KV_max<ncols1>, launch_params,
                (const half2 *) mask->data, KV_max.ptr, iter_k, s31, s33, init_fastdiv_values(Q->ne[1]));
        } else {
            kv_scan = "common<ncols1>";
            ggml_cuda_kernel_launch(flash_attn_mask_to_KV_max<ncols1>, launch_params,
                (const half2 *) mask->data, KV_max.ptr, iter_k, s31, s33);
        }
        CUDA_CHECK(cudaGetLastError());
    }
    const uint3 ne01_pos = init_fastdiv_values(Q->ne[1]);   // [TAG_FA_POS_MASK]
    // [TAG_FA_KVMAX_POS] same gate as launch_fattn.
    if (kv_pos_t && K->ne[1] % FATTN_KQ_STRIDE == 0 && (Q->ne[1] >= 1024 || kvmax_worth_it)) {   // [TAG_FA_POS_MASK]
        const dim3 blocks_num_KV_max(ntiles_x, 1, 1);
        const dim3 block_dim_KV_max(FATTN_KQ_STRIDE/2, 1, 1);
        const int iter_k = K->ne[1] / FATTN_KQ_STRIDE;
        KV_max.alloc(2*ntiles_x);    // [TAG_FA_KVMIN] {max, min} per entry
        ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_KV_max, block_dim_KV_max, 0, main_stream);
        ggml_cuda_kernel_launch(flash_attn_pos_to_KV_max<ncols1>, launch_params,
            (const int32_t *) kv_pos_t->data, (const int32_t *) q_pos_t->data, KV_max.ptr, iter_k, ne01_pos);
        CUDA_CHECK(cudaGetLastError());
        kv_scan = "pos<ncols1>";
    }

    const dim3 block_dim(warp_size, nwarps, 1);
    int max_blocks_per_sm = 1; // Max. number of active blocks limited by occupancy.
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, fattn_kernel, block_dim.x * block_dim.y * block_dim.z, nbytes_shared));
    GGML_ASSERT(max_blocks_per_sm > 0);
    int parallel_blocks = max_blocks_per_sm;

    const int ntiles_KV = (K->ne[1] + nbatch_fa - 1) / nbatch_fa; // Max. number of parallel blocks limited by KV cache length.

    // [TAG_TURBOT_FA_BALANCE]
    const ggml_cuda_fattn_turbot_balance_cfg & balance_cfg = ggml_cuda_fattn_turbot_balance_cfg_get();
    bool    balanced        = false;
    int     balance_stripe  = 0;   // [TAG_TURBOT_FA_STRIPE]
    int64_t balance_w_old   = 0;
    int64_t balance_w_young = 0;

    dim3 blocks_num;
    if (stream_k) {
        // For short contexts it can be faster to have the SMs work on whole tiles because this lets us skip the fixup.
        const int max_blocks = max_blocks_per_sm*nsm;
        const int tiles_nwaves = (ntiles_dst + max_blocks - 1) / max_blocks;
        const int tiles_efficiency_percent = 100 * ntiles_dst / (max_blocks*tiles_nwaves);

        const bool use_stream_k = cc >= GGML_CUDA_CC_ADA_LOVELACE || amd_wmma_available(cc) || tiles_efficiency_percent < 75;

        blocks_num.x = ntiles_dst;
        blocks_num.y = 1;
        blocks_num.z = 1;

        if(use_stream_k) {
            const int nblocks_stream_k_raw = std::min(max_blocks, ntiles_KV*ntiles_dst);
            // Round down to a multiple of ntiles_dst so that each output tile gets the same number of blocks (avoids fixup).
            // Only do this if the occupancy loss from rounding is acceptable.
            const int nblocks_stream_k_rounded = (nblocks_stream_k_raw / ntiles_dst) * ntiles_dst;
            const int max_efficiency_loss_percent = 5;
            const int efficiency_loss_percent = nblocks_stream_k_rounded > 0
                ? 100 * (nblocks_stream_k_raw - nblocks_stream_k_rounded) / nblocks_stream_k_raw
                : 100;
            const int nblocks_stream_k = efficiency_loss_percent <= max_efficiency_loss_percent
                ? nblocks_stream_k_rounded
                : nblocks_stream_k_raw;

            blocks_num.x = nblocks_stream_k;
        }

        if (ntiles_dst % blocks_num.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            dst_tmp_meta.alloc((size_t(blocks_num.x) * ncols * (2 + DV/2)));
        }

        // [TAG_TURBOT_FA_BALANCE] Only layouts that already carry the fixup buffer: blocks covering whole output tiles
        // keep the uniform slice (balancing them would split tiles and need a buffer they do not have).
        // [TAG_TURBOT_FA_STRIPE] Uniform-fixup layouts (every decode shape) take striped blocks: no pass, no seam table,
        // the common uniform fixup. The seam table remains for general-fixup layouts (prefill), where its passes are
        // small against the op.
        const bool uniform_fixup_layout = (int) blocks_num.x % ntiles_dst == 0 && (int) blocks_num.x > ntiles_dst;
        if (balance_cfg.mode != 0 && balance_cfg.stripe && uniform_fixup_layout) {
            balance_stripe = (int) blocks_num.x / ntiles_dst;
        }
        balanced = balance_cfg.mode != 0 && balance_stripe == 0 && ntiles_dst % (int) blocks_num.x != 0;
        if (balanced) {
            uint64_t shape_key = 0xCBF29CE484222325ull;
            for (const uint64_t v : { (uint64_t) Q->ne[1], (uint64_t) K->ne[1], (uint64_t) ncols1, (uint64_t) ncols2,
                                      (uint64_t) turbot_desc0, (uint64_t) (mask != nullptr), (uint64_t) (kv_pos_t != nullptr),
                                      (uint64_t) blocks_num.x }) {
                shape_key = (shape_key ^ v) * 0x100000001B3ull;
            }
            ggml_cuda_fattn_turbot_balance_weights(layer, ncols, shape_key, balance_w_old, balance_w_young);
        }

        // [TAG_TURBOT_FA_DEBUG]
        if (ggml_cuda_fattn_turbot_debug_on()) {
            static std::mutex mtx;
            static std::set<std::tuple<int64_t, int64_t, int, int, int, int>> seen;
            const auto key = std::make_tuple((int64_t) Q->ne[1], (int64_t) K->ne[1], (int) blocks_num.x, max_blocks_per_sm,
                                             (int) (mask != nullptr), (int) (kv_pos_t != nullptr));
            std::lock_guard<std::mutex> lock(mtx);
            if (seen.insert(key).second) {
                const int     nblocks    = (int) blocks_num.x;
                const int64_t total_work = (int64_t) ntiles_KV * ntiles_dst;
                const char *  fixup      = balance_stripe > 0 ? "striped-uniform" :
                                           balanced && balance_cfg.diag == 0 ? (balance_cfg.fixup_tile ? "balanced-tile" : "balanced-general") :
                                           nblocks % ntiles_dst == 0 && nblocks > ntiles_dst ? "uniform" :
                                           ntiles_dst % nblocks != 0                          ? "general" : "none";
                fprintf(stderr,
                    "turbot-fa: ncols1=%d ncols2=%d Q=%lld kv=%lld nbatch_fa=%d nsm=%d max_blocks_per_sm=%d max_blocks=%d "
                    "ntiles_x=%d ntiles_z_gqa=%d ntiles_dst=%d ntiles_KV=%d blocks_num.x=%d kv_tiles_per_block=%.2f "
                    "blocks_per_dst_tile=%.3f fixup=%s nbytes_shared=%zu kv_scan=%s balance_mode=%d w_old=0x%016llx w_young=0x%016llx\n",
                    ncols1, ncols2, (long long) Q->ne[1], (long long) K->ne[1], nbatch_fa, nsm, max_blocks_per_sm,
                    max_blocks_per_sm*nsm, ntiles_x, ntiles_z_gqa, ntiles_dst, ntiles_KV, nblocks,
                    (double) total_work / (double) nblocks, (double) nblocks / (double) ntiles_dst, fixup, nbytes_shared, kv_scan,
                    balance_cfg.mode, (unsigned long long) balance_w_old, (unsigned long long) balance_w_young);
                fflush(stderr);
            }
        }
    } else {
        // parallel_blocks must not be larger than what the tensor size allows:
        parallel_blocks = std::min(parallel_blocks, ntiles_KV);

        // If ntiles_total % blocks_per_wave != 0 then some efficiency is lost due to tail effects.
        // Test whether parallel_blocks can be set to a higher value for better efficiency.
        const int blocks_per_wave = nsm * max_blocks_per_sm;
        int nwaves_best = 0;
        int efficiency_percent_best = 0;
        for (int parallel_blocks_test = parallel_blocks; parallel_blocks_test <= ntiles_KV; ++parallel_blocks_test) {
            const int nblocks_total = ntiles_dst * parallel_blocks_test;
            const int nwaves = (nblocks_total + blocks_per_wave - 1) / blocks_per_wave;
            const int efficiency_percent = 100 * nblocks_total / (nwaves*blocks_per_wave);

            // Stop trying configurations with more waves if we already have good efficiency to avoid excessive overhead.
            if (efficiency_percent_best >= 95 && nwaves > nwaves_best) {
                break;
            }

            if (efficiency_percent > efficiency_percent_best) {
                nwaves_best = nwaves;
                efficiency_percent_best = efficiency_percent;
                parallel_blocks = parallel_blocks_test;
            }
        }

        blocks_num.x = ntiles_x;
        blocks_num.y = parallel_blocks;
        blocks_num.z = ntiles_z_gqa*K->ne[2]*Q->ne[3];

        if (parallel_blocks > 1) {
            dst_tmp.alloc(parallel_blocks*ggml_nelements(KQV));
            dst_tmp_meta.alloc(parallel_blocks*ggml_nrows(KQV));
        }
    }

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (const float *) KQV->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) KQV->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const uint32_t n_head      = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // TODO other tensor dimensions after removal of WMMA kernel:
    const uint3 ne01 = init_fastdiv_values(Q->ne[1]);

    GGML_ASSERT(block_dim.x % warp_size == 0);

    // [TAG_TURBOT_FA_BALANCE] Young prefix, then the seams. Both queue on the main stream ahead of the FA kernel and the
    // fixup, which read the seams after their own ggml_cuda_pdl_sync.
    ggml_cuda_pool_alloc<int> balance_buf(pool);
    if (balanced) {
        const int nblocks = (int) blocks_num.x;
        const int iter_k  = ntiles_KV;
        const int n_o     = ntiles_x * ntiles_z_gqa;
        const int n_seq   = (int) Q->ne[3];

        int log2_tpg = 0;
        while ((nbatch_fa << log2_tpg) < GGML_TURBOT_GRANULE) {
            ++log2_tpg;
        }
        GGML_ASSERT((nbatch_fa << log2_tpg) == GGML_TURBOT_GRANULE && log2_tpg <= 2);
        GGML_ASSERT(K->ne[2] == GGML_TURBOT_N_HEAD);
        GGML_ASSERT(((iter_k - 1) >> log2_tpg) < gtab_t->ne[0]);
        GGML_ASSERT((int64_t) iter_k * n_o * GGML_TURBOT_N_HEAD * n_seq < INT_MAX);

        int nit = 0;
        while ((1 << nit) < iter_k) {
            ++nit;
        }

        balance_buf.alloc((size_t) nblocks + 1 + (size_t) iter_k + 1);
        int * const seams  = balance_buf.ptr;
        int * const prefix = balance_buf.ptr + nblocks + 1;

        const dim3 blocks_num_bal(1, 1, 1);
        const dim3 block_dim_bal(WARP_SIZE, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params_bal = ggml_cuda_kernel_launch_params(blocks_num_bal, block_dim_bal, 0, main_stream);
        switch (log2_tpg) {
            case 0:
                ggml_cuda_kernel_launch(flash_attn_turbot_young_prefix<0>, launch_params_bal,
                    (const int32_t *) gtab_t->data, prefix, iter_k);
                break;
            case 1:
                ggml_cuda_kernel_launch(flash_attn_turbot_young_prefix<1>, launch_params_bal,
                    (const int32_t *) gtab_t->data, prefix, iter_k);
                break;
            default:
                ggml_cuda_kernel_launch(flash_attn_turbot_young_prefix<2>, launch_params_bal,
                    (const int32_t *) gtab_t->data, prefix, iter_k);
                break;
        }
        CUDA_CHECK(cudaGetLastError());
        if (balance_cfg.diag != 1) {
            ggml_cuda_kernel_launch(flash_attn_turbot_balance_bounds<GGML_TURBOT_N_HEAD>, launch_params_bal,
                (const int *) prefix, seams, nblocks, iter_k, nit, n_o, n_seq, balance_w_old, balance_w_young);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    // LLAMA_TURBOT_FA_BALANCE_DIAG: the passes ran, the seams stay unused.
    const bool balance_use = balanced && balance_cfg.diag == 0;

    ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num, block_dim, nbytes_shared, main_stream);
    ggml_cuda_kernel_launch(fattn_kernel, launch_params,
        (const char *) Q->data,
        K_data,
        V_data,
        mask ? ((const char *) mask->data) : nullptr,
        sinks ? ((const char *) sinks->data) : nullptr,
        kv_pos_t ? (const int32_t *) kv_pos_t->data : nullptr,
        q_pos_t  ? (const int32_t *) q_pos_t->data  : nullptr,
        KV_max.ptr,
        !stream_k && parallel_blocks > 1 ? dst_tmp.ptr : (float *) KQV->data, dst_tmp_meta.ptr,
        scale, max_bias, m0, m1, n_head_log2, logit_softcap,
        Q->ne[0], ne01,     Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
        K->ne[0], K->ne[1], K->ne[2], K->ne[3], nb11, nb12, nb13,
        nb21, nb22, nb23,
        mask ? mask->ne[1] : 0, mask ? mask->ne[2] : 0, mask ? mask->ne[3] : 0,
        mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0,
        (const char *) pool_t->data,
        (const int32_t *) gtab_t->data,
        (int64_t) pool_t->nb[1],
        turbot_desc0,
        turbot_desc1,
        turbot_desc2,
        balance_use ? (const int *) balance_buf.ptr : nullptr,
        balance_stripe
    );
    CUDA_CHECK(cudaGetLastError());

    if (stream_k) {
        if (balance_use) {
            // [TAG_TURBOT_FA_BALANCE] The seams moved: the uniform fixup's "bpt blocks per tile" no longer holds, and the
            // general fixup's arithmetic seams are not the kernel's. Both turbot fixups read the same table.
            const dim3 block_dim_combine(DV, 1, 1);
            if (balance_cfg.fixup_tile) {
                const uint3 fd0 = init_fastdiv_values(ntiles_x * ntiles_z_gqa * K->ne[2]);
                const uint3 fd1 = init_fastdiv_values(ntiles_x * ntiles_z_gqa);
                const uint3 fd2 = init_fastdiv_values(ntiles_x);

                const dim3 blocks_num_combine = {(unsigned)ntiles_dst, ncols1, ncols2};

                const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, 0, main_stream);
                ggml_cuda_kernel_launch(flash_attn_turbot_stream_k_fixup_tile<DV, ncols1, ncols2>, launch_params,
                    (float *) KQV->data, dst_tmp_meta.ptr, (const int *) balance_buf.ptr,
                     Q->ne[1], Q->ne[2], gqa_ratio, (int) blocks_num.x, ntiles_KV,
                     fd0, fd1, fd2);
            } else {
                const uint3 fd_k_j_z_ne12 = init_fastdiv_values(ntiles_KV * ntiles_x * ntiles_z_gqa * K->ne[2]);
                const uint3 fd_k_j_z      = init_fastdiv_values(ntiles_KV * ntiles_x * ntiles_z_gqa);
                const uint3 fd_k_j        = init_fastdiv_values(ntiles_KV * ntiles_x);
                const uint3 fd_k          = init_fastdiv_values(ntiles_KV);

                const dim3 blocks_num_combine = {blocks_num.x, ncols1, ncols2};

                const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, 0, main_stream);
                ggml_cuda_kernel_launch(flash_attn_turbot_stream_k_fixup_general<DV, ncols1, ncols2>, launch_params,
                    (float *) KQV->data, dst_tmp_meta.ptr, (const int *) balance_buf.ptr,
                     Q->ne[1], Q->ne[2], gqa_ratio,
                     fd_k_j_z_ne12, fd_k_j_z, fd_k_j, fd_k);
            }
        } else if ((int)blocks_num.x % ntiles_dst == 0 && (int)blocks_num.x > ntiles_dst) {
            // Optimized fixup: nblocks_stream_k is a multiple of ntiles_dst, launch one block per tile.
            const int nblocks_sk  = (int)blocks_num.x;
            const int bpt         = nblocks_sk / ntiles_dst;

            const uint3 fd0 = init_fastdiv_values(ntiles_x * ntiles_z_gqa * K->ne[2]);
            const uint3 fd1 = init_fastdiv_values(ntiles_x * ntiles_z_gqa);
            const uint3 fd2 = init_fastdiv_values(ntiles_x);

            const dim3 block_dim_combine(DV, 1, 1);
            const dim3 blocks_num_combine = {(unsigned)ntiles_dst, ncols1, ncols2};

            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, 0, main_stream);
            ggml_cuda_kernel_launch(flash_attn_stream_k_fixup_uniform<DV, ncols1, ncols2>, launch_params,
                (float *) KQV->data, dst_tmp_meta.ptr,
                 Q->ne[1], Q->ne[2], K->ne[2], nblocks_sk,
                 gqa_ratio, bpt, fd0, fd1, fd2);
        } else if (ntiles_dst % blocks_num.x != 0) {
            // General fixup for the cases where nblocks_stream_k < ntiles_dst.
            const int total_work = ntiles_KV * ntiles_dst;

            const uint3 fd_k_j_z_ne12 = init_fastdiv_values(ntiles_KV * ntiles_x * ntiles_z_gqa * K->ne[2]);
            const uint3 fd_k_j_z      = init_fastdiv_values(ntiles_KV * ntiles_x * ntiles_z_gqa);
            const uint3 fd_k_j        = init_fastdiv_values(ntiles_KV * ntiles_x);
            const uint3 fd_k          = init_fastdiv_values(ntiles_KV);

            const dim3 block_dim_combine(DV, 1, 1);
            const dim3 blocks_num_combine = {blocks_num.x, ncols1, ncols2};

            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, 0, main_stream);
            ggml_cuda_kernel_launch(flash_attn_stream_k_fixup_general<DV, ncols1, ncols2>, launch_params,
                (float *) KQV->data, dst_tmp_meta.ptr,
                 Q->ne[1], Q->ne[2], gqa_ratio, total_work,
                 fd_k_j_z_ne12, fd_k_j_z, fd_k_j, fd_k);
        }
    } else if (parallel_blocks > 1) {
        const dim3 block_dim_combine(DV, 1, 1);
        const dim3 blocks_num_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
        const size_t nbytes_shared_combine = parallel_blocks*sizeof(float2);

        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, nbytes_shared_combine, main_stream);
        ggml_cuda_kernel_launch(flash_attn_combine_results<DV>, launch_params,
            dst_tmp.ptr, dst_tmp_meta.ptr, (float *) KQV->data, parallel_blocks);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <int DKQ, int DV, int ncols1, int ncols2>
void ggml_cuda_flash_attn_ext_turbot_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    static_assert(DKQ == 256 && DV == 256, "turbot kernels exist for D=256 only");

    const ggml_tensor * KQV = dst;
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    constexpr int ncols = ncols1 * ncols2;

    // Validates the params and every tensor property the kernel relies on (the launcher packs the same layout).
    const ggml_turbot_layer layer = ggml_cuda_fattn_turbot_layer_of(dst);
    GGML_UNUSED(layer);

    const int  nthreads       = ggml_cuda_fattn_mma_get_nthreads      (DKQ, DV, ncols, cc);
    const int  nbatch_fa      = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols, cc);
    const int  nbatch_K2      = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols, cc);
    const int  nbatch_V2      = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols, cc);
    const int  nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols, cc);
    const bool Q_in_reg       = ggml_cuda_fattn_mma_get_Q_in_reg      (DKQ, DV, ncols, cc);
    const int  nstages_cfg    = ggml_cuda_fattn_mma_get_nstages       (DKQ, DV, ncols1, ncols2, cc);

    GGML_ASSERT(Q_in_reg);
    GGML_ASSERT(nbatch_fa > 0 && GGML_TURBOT_GRANULE % nbatch_fa == 0);
    GGML_ASSERT(nbatch_K2 % 16 == 0 && nbatch_V2 % 16 == 0);

    const int cols_per_warp = std::min(ncols, get_cols_per_warp(cc));
    const int warp_size_host = ggml_cuda_info().devices[ctx.device].warp_size;
    const int nwarps         = nthreads / warp_size_host;

    // SPEC 7.5. The turbot kernel is single-stage: the KV tile, then the mask tile, then the LUT area.
    const size_t nbytes_shared_KV_1stage = nbatch_fa            * std::max(nbatch_K2 + 4,  nbatch_V2 + 4) * sizeof(half2);
    const size_t nbytes_shared_KV_2stage = nbatch_fa            *         (nbatch_K2 + 4 + nbatch_V2 + 4) * sizeof(half2);
    const size_t nbytes_shared_Q         = ncols                * (DKQ/2 + 4)                             * sizeof(half2);
    const size_t nbytes_shared_mask      = ncols1               * (nbatch_fa/2 + 4)                       * sizeof(half2);
    const size_t nbytes_shared_combine   = nwarps*cols_per_warp * (nbatch_combine + 4)                    * sizeof(half2);

    const size_t lut_off = (size_t) ggml_cuda_fattn_turbot_lut_off(nbatch_fa, nbatch_K2, nbatch_V2, ncols1);
    GGML_ASSERT(lut_off == GGML_PAD(nbytes_shared_KV_1stage + nbytes_shared_mask, 16));

    const size_t nbytes_shared_total = std::max(nbytes_shared_combine, std::max(nbytes_shared_Q, lut_off + TURBOT_NBYTES_SHARED_LUT));

    // Never more than the f16 case asks for with the same config, plus the LUT area.
    {
        const size_t nbytes_shared_KV_f16    = nstages_cfg <= 1 ? nbytes_shared_KV_1stage : nbytes_shared_KV_2stage;
        const size_t nbytes_shared_total_f16 = std::max(nbytes_shared_combine, Q_in_reg ?
            std::max(nbytes_shared_Q,  nbytes_shared_KV_f16 + nbytes_shared_mask) :
                     nbytes_shared_Q + nbytes_shared_KV_f16 + nbytes_shared_mask);
        GGML_ASSERT(nbytes_shared_total <= nbytes_shared_total_f16 + TURBOT_NBYTES_SHARED_LUT);
    }

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

#if defined(GGML_USE_HIP)
    using fattn_kernel_ptr_t = const void*;
#else
    using fattn_kernel_ptr_t = fattn_turbot_kernel_t;
#endif // defined(GGML_USE_HIP)
    fattn_turbot_kernel_t fattn_kernel;
    const int softcap_idx = logit_softcap == 0.0f ? 0 : 1;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        fattn_kernel = flash_attn_ext_turbot<DKQ, DV, ncols1, ncols2, use_logit_softcap>;
    } else {
        constexpr bool use_logit_softcap = true;
        fattn_kernel = flash_attn_ext_turbot<DKQ, DV, ncols1, ncols2, use_logit_softcap>;
    }

#if !defined(GGML_USE_MUSA)
    // One slot per softcap variant: each is a different kernel and needs its own shared-memory limit raised.
    static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES][2] = {{false}};
    if (!shared_memory_limit_raised[id][softcap_idx]) {
        CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
        shared_memory_limit_raised[id][softcap_idx] = true;
    }
#else
    GGML_UNUSED(softcap_idx);
#endif // !defined(GGML_USE_MUSA)

    launch_fattn_turbot<DV, ncols1, ncols2>(ctx, dst, fattn_kernel, nwarps, nbytes_shared_total, nbatch_fa, warp_size_host);
}

// Declarations shared with fattn.cu (explicit instantiation declarations of the 20 instances and the macro the
// instance files use), after the definition like the DECL_FATTN_MMA_F16_CASE lines in fattn-mma-f16.cuh.
#include "fattn-turbot-decl.cuh"
