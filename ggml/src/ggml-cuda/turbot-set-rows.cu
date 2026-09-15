// [TAG_TURBOT] CUDA writer of the turbot tiered KV cache: GGML_OP_TURBOT_SET_ROWS (docs/turbot/SPEC.md section 8).
//
// Two kernels, launched on the same stream in this order:
//   k_turbot_fill      one CUDA block per (fill entry, granule cell, WHT group). Gives a live cell that only holds an
//                      old code a centre-fill refinement (SPEC 4.4; oracle ggml_turbot_fill_side).
//   k_turbot_set_rows  one CUDA block per (row, WHT group). Base code always, refinement when the row has a young
//                      pool row (SPEC 4.2; oracle ggml_turbot_encode_side).
//
// Both use 128 lanes = 4 warps x 32 lanes, element j of the group on lane j. The load, norm and rotation sequence is
// k_set_rows_turbo5p's verbatim (set-rows.cu, [TAG_TURBO5P]) without InnerQ, which turbot refuses (SPEC 9.3). The
// turbo5p writer itself is not touched, and this is the only writer TU that includes turbot-tables.cuh.
//
// The per-head layout arrives as block-uniform scalar kernel arguments, never as __device__ globals (a host write to
// a global races with kernels still queued for earlier layers): b[h] and y[h] as nibbles, base_off[h] and
// young_off[h] as 16-bit fields, the two gain offsets, and the side's byte offset inside a pool row.

#include "turbot-set-rows.cuh"
#include "turbo-quant.cuh"      // TURBO_WHT_SIGNS1/2, read-only use
#include "turbot-tables.cuh"    // TURBOT_D_* codebook tables, ggml-turbot.h

#include <climits>

// [TAG_TURBOT] Index search of the row kernel (SPEC 8.2 step 3).
//   0 (default): one count over the __constant__ threshold run. The run offset is block-uniform, so every lane reads
//      the same addresses and the loads broadcast.
//   1: copy the threshold run into shared memory next to the codebook runs, then a branchless binary search
//      (b or y data-dependent shared loads per element).
// Both return "number of thresholds t with u >= t", i.e. ggml_turbot_young_index / ggml_turbot_old_index, for ties,
// duplicate thresholds and NaN (every compare false -> 0) alike. Switch to 1 only if the writer perf case (SPEC 11.2)
// measures it faster.
#define GGML_CUDA_TURBOT_WRITER_BSEARCH 0

static_assert(GGML_TURBOT_GROUP == 4*WARP_SIZE, "turbot writer: one WHT group must be 4 warps");

// Interleave two 4-bit lane masks into one P2 byte: bit 2k is bit k of lo (field bit 0 of element k), bit 2k+1 is
// bit k of hi (field bit 1). Bit-parallel ALU, no loop.
static __device__ __forceinline__ uint8_t turbot_p2_byte(unsigned lo, unsigned hi) {
    lo &= 0xFu;
    hi &= 0xFu;
    lo = (lo | (lo << 2)) & 0x33u;
    lo = (lo | (lo << 1)) & 0x55u;
    hi = (hi | (hi << 2)) & 0x33u;
    hi = (hi | (hi << 1)) & 0x55u;
    return (uint8_t) (lo | (hi << 1));
}

// Pack this lane's code of width w (1..6) as element 128*grp + j of `run` (SPEC 3.4, ggml_turbot_set_code).
//   P4 (w >= 4, offset 0):   code bits 0..3; the even lane stores its nibble low and its partner's (lane ^ 1) high
//   P2 (w & 2):              the next 2 code bits; lanes with (lane & 3) == 0 store the byte of lanes lane..lane+3
//   P1 (w & 1):              the next code bit; lanes with (lane & 7) == 0 store the byte of lanes lane..lane+7
// Every lane runs all four collectives whatever w is, so no lane ever waits on a partner that skipped one. Plane
// presence and `store` are block-uniform and gate only the stores. Every byte of this group's part of each present
// plane is written, so the run needs no pre-clear. Ballot bit l is lane l of the warp ([TAG_TURBO5P] qh rule).
#define TURBOT_PACK_CODE(run, w, grp, code, store)                                                       \
    {                                                                                                    \
        const unsigned tp_code = (unsigned) (code);                                                      \
        const int      tp_w    = (w);                                                                    \
        const bool     tp_p4   = tp_w >= 4;                                                              \
        const bool     tp_p2   = (tp_w & 2) != 0;                                                        \
        const bool     tp_p1   = (tp_w & 1) != 0;                                                        \
        const int      tp_o2   = tp_p4 ? 128 : 0;                                                        \
        const int      tp_o1   = tp_o2 + (tp_p2 ? 64 : 0);                                               \
        const int      tp_sh2  = tp_p4 ? 4 : 0;                                                          \
        const int      tp_sh1  = tp_sh2 + (tp_p2 ? 2 : 0);                                               \
        const uint8_t  tp_nib  = (uint8_t) (tp_code & 0xFu);                                             \
        const uint8_t  tp_pnib = __shfl_sync(0xffffffff, tp_nib, lane ^ 1);                              \
        const unsigned tp_m20  = __ballot_sync(0xffffffff, (tp_code >> tp_sh2) & 1u);                    \
        const unsigned tp_m21  = __ballot_sync(0xffffffff, (tp_code >> (tp_sh2 + 1)) & 1u);              \
        const unsigned tp_m1   = __ballot_sync(0xffffffff, (tp_code >> tp_sh1) & 1u);                    \
        if ((store) && tp_p4 && (lane & 1) == 0) {                                                       \
            (run)[64*(grp) + j/2] = (uint8_t) (tp_nib | (tp_pnib << 4));                                 \
        }                                                                                                \
        if ((store) && tp_p2 && (lane & 3) == 0) {                                                       \
            (run)[tp_o2 + 32*(grp) + j/4] = turbot_p2_byte(tp_m20 >> lane, tp_m21 >> lane);              \
        }                                                                                                \
        if ((store) && tp_p1 && (lane & 7) == 0) {                                                       \
            (run)[tp_o1 + 16*(grp) + j/8] = (uint8_t) ((tp_m1 >> lane) & 0xFFu);                         \
        }                                                                                                \
    }

// ---- [TAG_TURBOT] row kernel: base code always, a_lloyd refinement for rows with a young pool row ----
template <typename idx_t>
__launch_bounds__(128)
static __global__ void k_turbot_set_rows(
        const float   * __restrict__ src0,       // rows, F32 [1024, n_rows], contiguous inside a row
        const idx_t   * __restrict__ src1,       // destination cell of each row
        const int32_t * __restrict__ young,      // young pool row of each row, or -1
        char          * __restrict__ dst,        // base cache data
        char          * __restrict__ pool,       // young pool data of this layer
        const int64_t   s01,                     // src0->nb[1] / sizeof(float)
        const int64_t   s10,                     // src1->nb[0] / sizeof(idx_t)
        const int64_t   s_young,                 // young->nb[0] / sizeof(int32_t)
        const int64_t   nb_row,                  // base row bytes, 32*S + 16
        const int64_t   nb_pool,                 // pool->nb[1]
        const int64_t   part_off,                // this side's part of a pool row: 0 (K) or pool_v_off (V)
        const uint32_t  bw,                      // b[h] in bits 4h..4h+3
        const uint32_t  yw,                      // y[h] in bits 4h..4h+3
        const uint64_t  base_offs,               // base_off[h] in bits 16h..16h+15
        const uint64_t  young_offs,              // young_off[h] in bits 16h..16h+15
        const int       base_gain_off,           // 32*S
        const int       young_gain_off) {        // 32*R

    // threadIdx.x = element within the group (0..127)
    const int j    = threadIdx.x;
    const int lane = j % WARP_SIZE;

    // blockIdx.x = 8*row + (2*h + g). Shifts only: no 64-bit or even 32-bit division on the GPU ([TAG_TURBO5P]).
    const uint32_t blk = (uint32_t) blockIdx.x;
    const int64_t  i   = (int64_t) (blk >> 3);
    const int      ig  = (int) (blk & 7u);
    const int      h   = ig >> 1;
    const int      g   = ig & 1;

    // Head layout, uniform over the whole CUDA block.
    const int b         = (int) ((bw >> (4*h)) & 0xFu);
    const int y         = (int) ((yw >> (4*h)) & 0xFu);
    const int r         = y - b;
    const int base_off  = (int) ((base_offs  >> (16*h)) & 0xFFFFu);
    const int young_off = (int) ((young_offs >> (16*h)) & 0xFFFFu);
    const int ooff      = turbot_d_old_off(b);
    const int yoff      = turbot_d_young_off(b, y);

    const int64_t cell     = (int64_t) *(src1 + i*s10);
    const int64_t yrow     = (int64_t) *(young + i*s_young);
    const bool    is_young = yrow >= 0;   // block-uniform tier branch

    // size_t pool arithmetic (SPEC 3.3). An old-only row never dereferences its pool pointers.
    const float   * src_row   = src0 + i*s01;
    const uint8_t * base_row  = (const uint8_t *) dst  + (size_t) cell*(size_t) nb_row;
    const uint8_t * young_row = (const uint8_t *) pool + (size_t) (is_young ? yrow : 0)*(size_t) nb_pool + (size_t) part_off;
    uint8_t * __restrict__ base_run  = (uint8_t *) base_row  + base_off;
    uint8_t * __restrict__ young_run = (uint8_t *) young_row + young_off;
    half    * __restrict__ gains_b   = (half *) (base_row  + base_gain_off);    // gain (h, g) at + 2*(2h + g)
    half    * __restrict__ gains_y   = (half *) (young_row + young_gain_off);

    // ---- Step 1: Load element j (coalesced). Head h group g of the row starts at 256*h + 128*g = 128*ig. ----
    __shared__ float x[GGML_TURBOT_GROUP];
    x[j] = src_row[GGML_TURBOT_GROUP*ig + j];
    __syncthreads();

    // ---- Step 2: Parallel L2 norm ----
    constexpr int n_warps = GGML_TURBOT_GROUP / WARP_SIZE;  // = 4
    __shared__ float warp_accum[n_warps];
    float v = x[j];
    float v2 = v * v;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        v2 += __shfl_xor_sync(0xffffffff, v2, offset);
    if (j % WARP_SIZE == 0)
        warp_accum[j / WARP_SIZE] = v2;
    __syncthreads();

    __shared__ float s_norm_sq;
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += warp_accum[w];
        s_norm_sq = total;
    }
    __syncthreads();
    const float grp_norm  = sqrtf(s_norm_sq);
    const float inv_norm  = (grp_norm > GGML_TURBOT_NORM_EPS) ? 1.0f / grp_norm : 0.0f;

    // ---- Step 3: Normalize ----
    x[j] *= inv_norm;
    __syncthreads();

    // ---- Step 4: Forward WHT (signs1 -> butterfly -> signs2, normalized), turbo5p's sequence ----
    x[j] *= TURBO_WHT_SIGNS1[j];
    __syncthreads();

    // For h < 32 the partner j^h is always in the same warp, so those stages run in registers. The lane holding the
    // LOW element wants a+b; the lane holding the high element wants a-b, which for that lane is (partner - self).
    {
        float vw = x[j];
#pragma unroll
        for (int hs = 1; hs < 32; hs <<= 1) {
            const float partner = __shfl_xor_sync(0xffffffffu, vw, hs);
            vw = (j & hs) ? (partner - vw) : (vw + partner);
        }
        x[j] = vw;
    }
    // MANDATORY barrier between the warp-scope and block-scope halves of the butterfly: stage 32 reads x[j+32],
    // written by another warp above (turbo4 non-determinism, fixed in 9fab66162).
    __syncthreads();

#define WHT_STAGE_SHARED_TBT(hs) \
    if (j % (2*(hs)) < (hs)) { float a = x[j], c = x[j+(hs)]; x[j] = a+c; x[j+(hs)] = a-c; } \
    __syncthreads();

    WHT_STAGE_SHARED_TBT(32)
    WHT_STAGE_SHARED_TBT(64)
#undef WHT_STAGE_SHARED_TBT

    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    x[j] = x[j] * inv_sqrt_128 * TURBO_WHT_SIGNS2[j];

    // ---- Step 5: codebook runs into shared memory (SPEC 8.2 step 5) ----
    // Lane k copies entry k (old) and entries 2k, 2k+1 (young). The run offsets are block-uniform, so these are
    // broadcast loads, and the per-element reads in step 7 gather from shared memory instead of replaying
    // data-dependent __constant__ addresses. Every copied entry lies inside its run: old [ooff, ooff + 2^b),
    // young [yoff, yoff + 2^y).
    __shared__ float lev_old[1 << GGML_TURBOT_B_MAX];
    __shared__ float lev_young[1 << GGML_TURBOT_Y_MAX];
    if (j < (1 << b)) {
        lev_old[j] = TURBOT_D_OLD_LEVELS[ooff + j];
    }
    if (is_young && j < (1 << (y - 1))) {
        lev_young[2*j]     = TURBOT_D_YOUNG_LUT[yoff + 2*j];
        lev_young[2*j + 1] = TURBOT_D_YOUNG_LUT[yoff + 2*j + 1];
    }
#if GGML_CUDA_TURBOT_WRITER_BSEARCH
    // Threshold run of the search below, same copy rule. The last slot of each run is the generator's 0 padding and
    // is never probed.
    __shared__ float thr[1 << GGML_TURBOT_Y_MAX];
    if (is_young) {
        if (j < (1 << (y - 1))) {
            thr[2*j]     = TURBOT_D_YOUNG_THR[yoff + 2*j];
            thr[2*j + 1] = TURBOT_D_YOUNG_THR[yoff + 2*j + 1];
        }
    } else {
        if (j < (1 << (b - 1))) {
            thr[2*j]     = TURBOT_D_OLD_THR[ooff + 2*j];
            thr[2*j + 1] = TURBOT_D_OLD_THR[ooff + 2*j + 1];
        }
    }
#endif
    __syncthreads();

    // ---- Step 6: Index (SPEC 4.2 index rule) ----
    // Young row: one count over all 2^y - 1 young thresholds; the old thresholds are embedded bit for bit at
    // ((j+1) << r) - 1, so idx >> r is the old index and idx & (2^r - 1) the refinement (ggml_turbot_young_index).
    const float u = x[j];
    int idx = 0;
#if GGML_CUDA_TURBOT_WRITER_BSEARCH
    // Branchless binary search over the sorted run of 2^n - 1 thresholds: probe idx + 2^k - 1 <= 2^n - 2.
    const int n_bits = is_young ? y : b;
    for (int k = n_bits - 1; k >= 0; --k) {
        idx += ((int) (u >= thr[idx + (1 << k) - 1])) << k;
    }
#else
    if (is_young) {
        const int n_thr = (1 << y) - 1;
        for (int k = 0; k < n_thr; ++k) {
            idx += (u >= TURBOT_D_YOUNG_THR[yoff + k]);
        }
    } else {
        const int n_thr = (1 << b) - 1;
        for (int k = 0; k < n_thr; ++k) {
            idx += (u >= TURBOT_D_OLD_THR[ooff + k]);
        }
    }
#endif
    const int code_b = is_young ? (idx >> r) : idx;
    const int code_r = is_young ? (idx & ((1 << r) - 1)) : 0;

    // ---- Step 7: Pack the base code and the refinement ----
    TURBOT_PACK_CODE(base_run,  b, g, code_b, true)
    TURBOT_PACK_CODE(young_run, r, g, code_r, is_young)

    // ---- Step 8: Reconstruction norms of both tiers (parallel) ----
    const float cb = lev_old[code_b];
    const float cy = is_young ? lev_young[(code_b << r) | code_r] : 0.0f;
    float rb2 = cb * cb;
    float ry2 = cy * cy;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        rb2 += __shfl_xor_sync(0xffffffff, rb2, offset);
        ry2 += __shfl_xor_sync(0xffffffff, ry2, offset);
    }
    __shared__ float warp_accum_y[n_warps];
    if (j % WARP_SIZE == 0) {
        warp_accum[j / WARP_SIZE]   = rb2;
        warp_accum_y[j / WARP_SIZE] = ry2;
    }
    __syncthreads();

    __shared__ float s_recon_b_sq;
    __shared__ float s_recon_y_sq;
    if (j == 0) {
        float total_b = 0.0f;
        float total_y = 0.0f;
        for (int w = 0; w < n_warps; w++) {
            total_b += warp_accum[w];
            total_y += warp_accum_y[w];
        }
        s_recon_b_sq = total_b;
        s_recon_y_sq = total_y;
    }
    __syncthreads();

    // ---- Step 9: Gain-corrected norms, norm / |recon| per tier (ggml_turbot_gain) ----
    // Each of the eight groups of a layer-side row is its own CUDA block and writes its own 2-byte slot.
    if (j == 0) {
        const float recon_b = sqrtf(s_recon_b_sq);
        gains_b[ig] = __float2half((recon_b > GGML_TURBOT_NORM_EPS) ? grp_norm / recon_b : grp_norm);
        if (is_young) {
            const float recon_y = sqrtf(s_recon_y_sq);
            gains_y[ig] = __float2half((recon_y > GGML_TURBOT_NORM_EPS) ? grp_norm / recon_y : grp_norm);
        }
    }
}

// ---- [TAG_TURBOT] centre fill: refinement for a live cell that only holds an old code ----
__launch_bounds__(128)
static __global__ void k_turbot_fill(
        const int32_t * __restrict__ fill_ents,  // I32 [4, n_fill]: granule, slot, (int32) mask_lo, (int32) mask_hi
        char          * __restrict__ dst,        // base cache data (read)
        char          * __restrict__ pool,       // young pool data of this layer (written)
        const int64_t   s_fill0,                 // fill->nb[0] / sizeof(int32_t)
        const int64_t   s_fill1,                 // fill->nb[1] / sizeof(int32_t)
        const int64_t   nb_row,                  // base row bytes, 32*S + 16
        const int64_t   nb_pool,                 // pool->nb[1]
        const int64_t   part_off,                // this side's part of a pool row: 0 (K) or pool_v_off (V)
        const uint32_t  bw,
        const uint32_t  yw,
        const uint64_t  base_offs,
        const uint64_t  young_offs,
        const int       base_gain_off,
        const int       young_gain_off) {

    const int j    = threadIdx.x;
    const int lane = j % WARP_SIZE;

    // blockIdx.x = 512*entry + 8*c + (2*h + g), c = cell inside the granule
    const uint32_t blk = (uint32_t) blockIdx.x;
    const int64_t  f   = (int64_t) (blk >> 9);
    const int      c   = (int) ((blk >> 3) & 63u);
    const int      ig  = (int) (blk & 7u);
    const int      h   = ig >> 1;
    const int      g   = ig & 1;

    const int32_t * ent     = fill_ents + f*s_fill1;
    const int64_t   granule = (int64_t) ent[0];
    const int64_t   slot    = (int64_t) ent[s_fill0];
    const uint32_t  mask    = (uint32_t) (c < 32 ? ent[2*s_fill0] : ent[3*s_fill0]);

    // Block-uniform: a cell whose bit is clear keeps its pool row untouched (it was written by this ubatch's rows,
    // is empty, or already has a valid refinement).
    if (((mask >> (c & 31)) & 1u) == 0u) {
        return;
    }

    const int b         = (int) ((bw >> (4*h)) & 0xFu);
    const int y         = (int) ((yw >> (4*h)) & 0xFu);
    const int r         = y - b;
    const int base_off  = (int) ((base_offs  >> (16*h)) & 0xFFFFu);
    const int young_off = (int) ((young_offs >> (16*h)) & 0xFFFFu);
    const int ooff      = turbot_d_old_off(b);
    const int yoff      = turbot_d_young_off(b, y);
    const int foff      = turbot_d_fill_off(b, y);

    const int64_t   cell      = granule*GGML_TURBOT_GRANULE + c;
    const uint8_t * base_row  = (const uint8_t *) dst  + (size_t) cell*(size_t) nb_row;
    const uint8_t * young_row = (const uint8_t *) pool + (size_t) (slot*GGML_TURBOT_GRANULE + c)*(size_t) nb_pool + (size_t) part_off;
    const uint8_t * __restrict__ base_run  = base_row + base_off;
    uint8_t       * __restrict__ young_run = (uint8_t *) young_row + young_off;
    const half    * __restrict__ gains_b   = (const half *) (base_row + base_gain_off);
    half          * __restrict__ gains_y   = (half *) (young_row + young_gain_off);

    // ---- Step 1: Unpack the old code (ggml_turbot_get_code) and copy the codebook runs ----
    // Plane presence is block-uniform; an absent plane is never loaded.
    const bool p4  = b >= 4;
    const bool p2  = (b & 2) != 0;
    const bool p1  = (b & 1) != 0;
    const int  o2  = p4 ? 128 : 0;
    const int  o1  = o2 + (p2 ? 64 : 0);
    const int  sh2 = p4 ? 4 : 0;
    const int  sh1 = sh2 + (p2 ? 2 : 0);
    unsigned code_bu = 0;
    if (p4) {
        code_bu |= ((unsigned) base_run[64*g + j/2] >> ((j & 1)*4)) & 0xFu;
    }
    if (p2) {
        code_bu |= (((unsigned) base_run[o2 + 32*g + j/4] >> (2*(j & 3))) & 3u) << sh2;
    }
    if (p1) {
        code_bu |= (((unsigned) base_run[o1 + 16*g + j/8] >> (j & 7)) & 1u) << sh1;
    }
    const int code_b = (int) code_bu;

    // Same broadcast copy as the row kernel; the fill-code run [foff, foff + 2^b) is copied alongside C_b.
    __shared__ float   lev_old[1 << GGML_TURBOT_B_MAX];
    __shared__ float   lev_young[1 << GGML_TURBOT_Y_MAX];
    __shared__ uint8_t fill_code[1 << GGML_TURBOT_B_MAX];
    if (j < (1 << b)) {
        lev_old[j]   = TURBOT_D_OLD_LEVELS[ooff + j];
        fill_code[j] = TURBOT_D_FILL_CODE[foff + j];
    }
    if (j < (1 << (y - 1))) {
        lev_young[2*j]     = TURBOT_D_YOUNG_LUT[yoff + 2*j];
        lev_young[2*j + 1] = TURBOT_D_YOUNG_LUT[yoff + 2*j + 1];
    }
    __syncthreads();

    // ---- Step 2: Centre refinement, pack, reconstruction norms ----
    const int   code_r = (int) fill_code[code_b];
    const float cb     = lev_old[code_b];
    const float cy     = lev_young[(code_b << r) | code_r];

    TURBOT_PACK_CODE(young_run, r, g, code_r, true)

    constexpr int n_warps = GGML_TURBOT_GROUP / WARP_SIZE;  // = 4
    __shared__ float warp_accum_b[n_warps];
    __shared__ float warp_accum_y[n_warps];
    float rb2 = cb * cb;
    float ry2 = cy * cy;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        rb2 += __shfl_xor_sync(0xffffffff, rb2, offset);
        ry2 += __shfl_xor_sync(0xffffffff, ry2, offset);
    }
    if (j % WARP_SIZE == 0) {
        warp_accum_b[j / WARP_SIZE] = rb2;
        warp_accum_y[j / WARP_SIZE] = ry2;
    }
    __syncthreads();

    __shared__ float s_recon_b_sq;
    __shared__ float s_recon_y_sq;
    if (j == 0) {
        float total_b = 0.0f;
        float total_y = 0.0f;
        for (int w = 0; w < n_warps; w++) {
            total_b += warp_accum_b[w];
            total_y += warp_accum_y[w];
        }
        s_recon_b_sq = total_b;
        s_recon_y_sq = total_y;
    }
    __syncthreads();

    // ---- Step 3: young gain = old gain * |C_b| / |LUT| (ggml_turbot_fill_side) ----
    if (j == 0) {
        const float gb  = __half2float(gains_b[ig]);
        const float nry = sqrtf(s_recon_y_sq);
        const float gy  = (nry > GGML_TURBOT_NORM_EPS) ? gb * (sqrtf(s_recon_b_sq) / nry) : gb;
        gains_y[ig] = __float2half(gy);
    }
}

#undef TURBOT_PACK_CODE

// ---- host side ----

// Block-uniform layout arguments of one layer-side (SPEC 8.2 step 1).
struct turbot_writer_layout {
    uint32_t bw         = 0;   // b[h] in bits 4h..4h+3
    uint32_t yw         = 0;   // y[h] in bits 4h..4h+3 (y <= 8 fits a nibble)
    uint64_t base_offs  = 0;   // base_off[h] in bits 16h..16h+15
    uint64_t young_offs = 0;   // young_off[h] in bits 16h..16h+15
};

static turbot_writer_layout turbot_writer_layout_of(const ggml_turbot_side & sd) {
    turbot_writer_layout lo;
    for (int h = 0; h < GGML_TURBOT_N_HEAD; ++h) {
        lo.bw         |= (uint32_t) sd.b[h] << (4*h);
        lo.yw         |= (uint32_t) sd.y[h] << (4*h);
        lo.base_offs  |= (uint64_t) sd.base_off[h]  << (16*h);
        lo.young_offs |= (uint64_t) sd.young_off[h] << (16*h);
    }
    return lo;
}

// Parses and validates the op without asserting. On success fills the layer, the written side and its pool part.
static bool turbot_set_rows_layout(const ggml_tensor * op, ggml_turbot_layer & layer, const ggml_turbot_side *& sd, int64_t & part_off) {
    if (op->op != GGML_OP_TURBOT_SET_ROWS) {
        return false;
    }
    const ggml_tensor * rows  = op->src[0];
    const ggml_tensor * cells = op->src[1];
    const ggml_tensor * base  = op->src[2];
    const ggml_tensor * pool  = op->src[3];
    const ggml_tensor * young = op->src[4];
    const ggml_tensor * fill  = op->src[5];

    // SPEC 5.2 asserts
    if (base == nullptr || rows == nullptr || cells == nullptr || pool == nullptr || young == nullptr) {
        return false;
    }
    if (!ggml_turbot_is_type(base->type)) {
        return false;
    }
    if (rows->type != GGML_TYPE_F32 || rows->ne[0] != GGML_TURBOT_ROW_ELEMS || rows->ne[2] != 1 || rows->ne[3] != 1 ||
            !ggml_is_contiguous_rows(rows)) {
        return false;
    }
    if (cells->ne[0] != rows->ne[1] || (cells->type != GGML_TYPE_I64 && cells->type != GGML_TYPE_I32)) {
        return false;
    }
    if (pool->type != GGML_TYPE_I8) {
        return false;
    }
    if (young->type != GGML_TYPE_I32 || young->ne[0] != rows->ne[1]) {
        return false;
    }
    if (fill != nullptr && (fill->type != GGML_TYPE_I32 || fill->ne[0] != 4)) {
        return false;
    }

    ggml_turbot_op_params params;
    if (!ggml_turbot_op_params_get(op, &params) || params.side > GGML_TURBOT_SIDE_V ||
            !ggml_turbot_layer_from_op_params(&params, &layer)) {
        return false;
    }
    sd       = params.side == GGML_TURBOT_SIDE_K ? &layer.k : &layer.v;
    part_off = params.side == GGML_TURBOT_SIDE_K ? 0 : (int64_t) layer.pool_v_off;
    if (base->type != ggml_turbot_type_of_s(sd->s)) {
        return false;
    }

    // Geometry the kernels address directly: whole 1024-value cell rows in the base cache, one pool row per pool
    // cell with the bytes of a row consecutive, and launch counts that fit the int grid size.
    if (base->ne[0] != GGML_TURBOT_ROW_ELEMS || base->nb[1] != (size_t) sd->base_row_bytes) {
        return false;
    }
    if (pool->ne[0] != (int64_t) layer.pool_row_bytes || pool->nb[0] != 1) {
        return false;
    }
    if (rows->ne[1] > INT_MAX / GGML_TURBOT_N_GAINS) {
        return false;
    }
    if (fill != nullptr && fill->ne[1] > INT_MAX / (GGML_TURBOT_GRANULE*GGML_TURBOT_N_GAINS)) {
        return false;
    }
    return true;
}

template <typename idx_t>
static void turbot_set_rows_cuda(
        ggml_backend_cuda_context  & ctx,
        ggml_tensor                * dst,
        const ggml_turbot_side     & sd,
        const turbot_writer_layout & lo,
        const int64_t                part_off) {

    const ggml_tensor * src0  = dst->src[0];
    const ggml_tensor * src1  = dst->src[1];
    const ggml_tensor * pool  = dst->src[3];
    const ggml_tensor * young = dst->src[4];

    const int64_t n_rows = src0->ne[1];

    cudaStream_t stream = ctx.stream();

    // One CUDA block per (row, WHT group): 8 per row, 128 lanes each.
    k_turbot_set_rows<idx_t><<<(int) (n_rows*GGML_TURBOT_N_GAINS), GGML_TURBOT_GROUP, 0, stream>>>(
        (const float *) src0->data, (const idx_t *) src1->data, (const int32_t *) young->data,
        (char *) dst->data, (char *) pool->data,
        (int64_t) (src0->nb[1]/sizeof(float)), (int64_t) (src1->nb[0]/sizeof(idx_t)), (int64_t) (young->nb[0]/sizeof(int32_t)),
        (int64_t) dst->nb[1], (int64_t) pool->nb[1], part_off,
        lo.bw, lo.yw, lo.base_offs, lo.young_offs, (int) sd.base_gain_off, (int) sd.young_gain_off);
}

void ggml_cuda_op_turbot_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_turbot_layer        layer;
    const ggml_turbot_side * sd       = nullptr;
    int64_t                  part_off = 0;
    GGML_ASSERT(turbot_set_rows_layout(dst, layer, sd, part_off) && "turbot_set_rows: invalid op (see SPEC 5.2)");

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * base = dst->src[2];
    const ggml_tensor * pool = dst->src[3];
    const ggml_tensor * fill = dst->src[5];

    // dst is ggml_view_tensor(base): same data, same row stride.
    GGML_ASSERT(dst->data == base->data && dst->nb[1] == base->nb[1]);
    GGML_ASSERT(src0->ne[0] == GGML_TURBOT_ROW_ELEMS);
    GGML_ASSERT(src1->ne[0] == src0->ne[1]);

    const int64_t n_rows = src0->ne[1];
    if (n_rows == 0) {
        return;
    }

    const turbot_writer_layout lo     = turbot_writer_layout_of(*sd);
    cudaStream_t               stream = ctx.stream();

    // Fill entries first (SPEC 5.2): a granule that just gained a slot gets refinements for its live old cells
    // before this layer's FA reads it through the young path. Rows written by this ubatch are never in a mask.
    if (fill != nullptr && fill->ne[1] > 0) {
        const int64_t n_blocks = fill->ne[1]*GGML_TURBOT_GRANULE*GGML_TURBOT_N_GAINS;
        k_turbot_fill<<<(int) n_blocks, GGML_TURBOT_GROUP, 0, stream>>>(
            (const int32_t *) fill->data, (char *) dst->data, (char *) pool->data,
            (int64_t) (fill->nb[0]/sizeof(int32_t)), (int64_t) (fill->nb[1]/sizeof(int32_t)),
            (int64_t) dst->nb[1], (int64_t) pool->nb[1], part_off,
            lo.bw, lo.yw, lo.base_offs, lo.young_offs, (int) sd->base_gain_off, (int) sd->young_gain_off);
    }

    if (src1->type == GGML_TYPE_I64) {
        turbot_set_rows_cuda<int64_t>(ctx, dst, *sd, lo, part_off);
    } else {
        turbot_set_rows_cuda<int32_t>(ctx, dst, *sd, lo, part_off);
    }
}

// SPEC 8.5: the 5.2 asserts, returning false instead of asserting, plus the geometry the kernels rely on. The CUDA
// buffer check of the base cache and the pool lives at the supports_op call site in ggml-cuda.cu, which owns
// ggml_backend_buft_is_cuda.
bool ggml_cuda_turbot_set_rows_supported(const ggml_tensor * op) {
    ggml_turbot_layer        layer;
    const ggml_turbot_side * sd       = nullptr;
    int64_t                  part_off = 0;
    return turbot_set_rows_layout(op, layer, sd, part_off);
}
