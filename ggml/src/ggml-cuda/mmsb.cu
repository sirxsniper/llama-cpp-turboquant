// [TAG_MMSB] [TAG_SMALLB] Tensor-core matmul for 2..16-column batches of Q4_K / Q5_K / Q6_K / Q8_0 weights.
//
// Why: with DFlash2 (n_max 3) every stream verifies 4 tokens per step, so 2 streams run the target model at 8 rows and
// 4 streams at 16. Today those widths go to MMQ (MMVQ stops at 5 for Q4_K/Q5_K and 7 for Q6_K on Blackwell), whose
// 128-row tiles and stream-k fixups reach ~0.83 TB/s at 8 columns against a ~1.2 TB/s mat-vec floor. This kernel
// streams the weights once, straight from global memory into mma fragments, with one unit of register prefetch.
//
// Maths (same as today's paths, see the per-type comments): src1 is quantized to q8_1 in MMQ's block layout
// (block_q8_1_mmq, 128 values + 16 bytes of scales) by a PDL copy of quantize_mmq_q8_1 (quantize.cu). Every 32-value
// chunk is an exact int8 x int8 dot product on the tensor cores (m16n8k32, or m16n8k16 per 16-value Q6_K sub-block);
// the float scaling per chunk is the MMQ/MMVQ one, never coarser. Results are not bit-identical to MMQ (different
// summation order over K), but they are deterministic: the split-K partials are summed in a fixed order.
//
// Geometry: one block = 8 warps. The block covers RG groups of 16 weight rows; the KS = 8/RG warps of a row group
// split K. A unit is 128 values of K (one block_q8_1_mmq per column; for K-quants half a 256-value super-block).
// Every warp computes a 16 x (8*NT) tile over its K range with NT n8 tiles (NT = 1 for <= 8 columns, 2 above), then
// the KS partials are reduced through shared memory in a fixed order and written to dst.
//
// Fragment layout (PTX mma.m16n8k32 .s8, lane = 4g + t): A x[0] = (row g, k-int t), x[1] = (row g+8, t),
// x[2] = (row g, t+4), x[3] = (row g+8, t+4); B x[0] = (k-int t, column g), x[1] = (t+4, g); C x[0] = (g, 2t),
// x[1] = (g, 2t+1), x[2] = (g+8, 2t), x[3] = (g+8, 2t+1). A and B both map logical k-int t -> physical int 2t and
// t+4 -> 2t+1, so each lane reads 8 contiguous bytes (8t..8t+7 of the 32-value chunk) and the dot product is the same.
// The k16 variant (Q6_K) uses the natural layout: A x[0] = (row g, t), x[1] = (row g+8, t); B x[0] = (t, column g).

#include "mmsb.cuh"
#include "common.cuh"
#include "mma.cuh"
#include "quantize.cuh"   // block_q8_1_mmq, quantize_mmq_q8_1_pdl_cuda
#include "mmvq.cuh"       // ggml_cuda_should_use_mmvq (phase gate)
#include "vecdotq.cuh"    // get_int_b2

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>

#define MMSB_NWARPS      8     // warps per block (RG row groups x KS K splits)
#define MMSB_MAX_COLS   16     // widest src1 batch (2 n8 tiles)
#define MMSB_UNIT      128     // K values per unit = one block_q8_1_mmq per column
#define MMSB_YB        144     // bytes per block_q8_1_mmq: 16 bytes of scales + 128 int8
#define MMSB_RED_STRIDE 17     // shared-memory reduction stride per column (16 rows + 1 pad)

// Default row limit. MMQ is faster once its 128-row tiles fill the GPU; this kernel only wins on weights with few rows,
// where MMQ gets at most 8 tiles. 5090, K = 5120, 9-16 columns, MMQ -> this kernel: 48 rows 33 -> 7 us, 1024 rows
// 13.4 -> 11.3 us; 2048 rows 13.0 -> 14.8 us, 4096 rows 15.3 -> 24.4 us, 17408 rows 35 -> 82 us.
// GGML_CUDA_SMALLB_MAX_ROWS=0 removes the limit.
#define MMSB_MAX_ROWS_DEFAULT 1024

static_assert(sizeof(block_q8_1_mmq) == MMSB_YB, "mmsb: unexpected block_q8_1_mmq size");
static_assert(QK8_1_MMQ == MMSB_UNIT, "mmsb: unexpected block_q8_1_mmq length");

// ------------------------------------------------------------------------------------------------------------------
// Environment
// ------------------------------------------------------------------------------------------------------------------

struct mmsb_env_t {
    int     mode;      // -1 = GGML_CUDA_SMALLB unset (Blackwell only), 0 = off, 1 = any Ampere+ NVIDIA GPU
    int     min_n;     // 0 = keep every width MMVQ takes today; n = also take widths >= n (phase 2: 2)
    int     max_n;     // upper width, 1..16
    int     rg;        // 0 = automatic, else 1 / 2 / 4
    int64_t max_rows;  // decline weights with more rows (default MMSB_MAX_ROWS_DEFAULT, 0 = no limit)
    bool    xpf;       // test only: prefetch the weights before the PDL wait for any src0 buffer
    bool    probe;     // TURBO_PATH_PROBE=1
};

static int mmsb_env_int(const char * name, const int def) {
    const char * e = getenv(name);
    return (e && e[0]) ? atoi(e) : def;
}

static const mmsb_env_t & mmsb_env() {
    static const mmsb_env_t env = [] {
        mmsb_env_t v;
        const char * s = getenv("GGML_CUDA_SMALLB");
        v.mode  = (s && s[0]) ? (atoi(s) != 0 ? 1 : 0) : -1;
        v.min_n = std::max(0, mmsb_env_int("GGML_CUDA_SMALLB_MIN", 0));
        v.max_n = std::min(MMSB_MAX_COLS, std::max(1, mmsb_env_int("GGML_CUDA_SMALLB_MAX", MMSB_MAX_COLS)));
        const int rg = mmsb_env_int("GGML_CUDA_SMALLB_RG", 0);
        v.rg    = (rg == 1 || rg == 2 || rg == 4) ? rg : 0;
        const char * r = getenv("GGML_CUDA_SMALLB_MAX_ROWS");
        v.max_rows = (r && r[0]) ? std::max<int64_t>(0, (int64_t) atoll(r)) : MMSB_MAX_ROWS_DEFAULT;
        v.xpf   = mmsb_env_int("GGML_CUDA_SMALLB_XPF", 0) != 0;
        const char * p = getenv("TURBO_PATH_PROBE");
        v.probe = p && p[0] == '1';
        if (v.probe) {
            fprintf(stderr, "turbo-probe: mmsb env mode=%d min=%d max=%d rg=%d max_rows=%lld xpf=%d\n",
                    v.mode, v.min_n, v.max_n, v.rg, (long long) v.max_rows, (int) v.xpf);
            fflush(stderr);
        }
        return v;
    }();
    return env;
}

// ------------------------------------------------------------------------------------------------------------------
// Device helpers
// ------------------------------------------------------------------------------------------------------------------

// Weight loads for Q4_K / Q5_K (16-byte aligned rows): read-only path, no L1 allocation (every weight byte is used once
// per kernel; the activations should keep L1). Build with -DGGML_CUDA_MMSB_PLAIN_NC if ptxas rejects the qualifier.
#if defined(GGML_CUDA_MMSB_PLAIN_NC)
#define MMSB_LD_X "ld.global.nc"
#else
#define MMSB_LD_X "ld.global.nc.L1::no_allocate"
#endif // defined(GGML_CUDA_MMSB_PLAIN_NC)

static __device__ __forceinline__ int4 mmsb_ld_x4(const char * p) {
    int4 v;
#if defined(TURING_MMA_AVAILABLE) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    asm volatile(MMSB_LD_X ".v4.s32 {%0, %1, %2, %3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
#else
    v = *(const int4 *) p;
#endif // defined(TURING_MMA_AVAILABLE) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    return v;
}

static __device__ __forceinline__ int2 mmsb_ld_x2(const char * p) {
    int2 v;
#if defined(TURING_MMA_AVAILABLE) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    asm volatile(MMSB_LD_X ".v2.s32 {%0, %1}, [%2];"
        : "=r"(v.x), "=r"(v.y) : "l"(p));
#else
    v = *(const int2 *) p;
#endif // defined(TURING_MMA_AVAILABLE) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    return v;
}

static __device__ __forceinline__ float2 mmsb_h2f2(const int v) {
    half2 h;
    memcpy(&h, &v, sizeof(h));
    return __half22float2(h);
}

// Component c (0..3) of an int4; c is a compile-time constant after unrolling, so this is a plain register read.
static __device__ __forceinline__ int mmsb_i4(const int4 & v, const int c) {
    return c == 0 ? v.x : (c == 1 ? v.y : (c == 2 ? v.z : v.w));
}

// Signed byte i (0..3) of v.
static __device__ __forceinline__ int mmsb_s8(const int v, const int i) {
    return (int) (int8_t) (v >> (8*i));
}

// Activation fragment columns of n8 tile nt: the B column of this lane (8nt+g) and the two C columns (8nt+2t, 8nt+2t+1),
// clamped to the last column. Clamped columns are computed and never stored.
struct mmsb_cols {
    int b, e, o;
    __device__ __forceinline__ mmsb_cols(const int nt, const int ne11, const int g, const int t)
        : b(min(8*nt + g, ne11 - 1)), e(min(8*nt + 2*t, ne11 - 1)), o(min(8*nt + 2*t + 1, ne11 - 1)) {}
};

// ------------------------------------------------------------------------------------------------------------------
// Per-type traits: raw = the registers of one unit for rows g and g+8 of this lane (loads only, no arithmetic, so the
// prefetch never waits); load() issues the loads of unit u; compute() consumes a loaded unit.
// ------------------------------------------------------------------------------------------------------------------

template <ggml_type type> struct mmsb_traits;

// Q4_K (144-byte super-blocks: dm, 12 scale bytes, 128 qs) and Q5_K (176 bytes: dm, scales, 32 qh, 128 qs).
// The MMVQ bit order (vecdotq.cuh vec_dot_q5_K_q8_1_impl_vmmq) with MMQ's DS4 min term (mmq-vec-dot.cuh, q8_1 x q8_1
// mma: d8 * (sc * dot) and s8 * m, s8 = the float sum of the 32 activations from the q8_1 block):
//   unit u = half h = u&1 of super-block b = u>>1 = sub-blocks j = 4h..4h+3, chunk c = j - 4h;
//   q of sub-block j at position l = nibble (j&1) of qs[32*(j>>1) + l] | bit j of qh[l] << 4.
template <bool has_qh>
struct mmsb_q45_K {
    static constexpr int bs = has_qh ? 176 : 144;   // bytes per 256-value super-block
    static constexpr int qs = has_qh ?  48 :  16;   // offset of qs in the super-block

    struct raw {
        int4 hdr[2];   // dm + 12 scale bytes (the 4 lanes of a quad read the same 16 bytes)
        int2 qh[2];    // Q5_K: qh bytes 8t..8t+7
        int2 q0[2];    // qs bytes of chunks 0/1 (low/high nibble): 64h + 8t..
        int2 q1[2];    // qs bytes of chunks 2/3:                   64h + 32 + 8t..
    };

    static __device__ __forceinline__ void load(raw & r, const char * xA, const char * xB, const int u, const int t) {
        const int o = (u >> 1)*bs;
        const int h = u & 1;
        const char * p[2] = {xA + o, xB + o};
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            r.hdr[i] = mmsb_ld_x4(p[i]);
            if constexpr (has_qh) {
                r.qh[i] = mmsb_ld_x2(p[i] + 16 + 8*t);
            }
            r.q0[i] = mmsb_ld_x2(p[i] + qs + 64*h +      8*t);
            r.q1[i] = mmsb_ld_x2(p[i] + qs + 64*h + 32 + 8*t);
        }
    }

    template <int NT>
    static __device__ __forceinline__ void compute(const raw & r, const char * yu, const int u, const int ne11,
            const int g, const int t, float (&out)[NT][4]) {
        const int h = u & 1;

        // 6-bit scales and mins of sub-blocks 4h..4h+3, byte c = chunk c. Equal to unpack_scales_q45_K(sc, h) and
        // (sc, h + 2) in mmq-load-tiles.cuh; kept in registers (no dynamic indexing).
        int    sc[2];
        int    mn[2];
        float2 dm[2];
        int2   qhs[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int s0 = r.hdr[i].y;
            const int s1 = r.hdr[i].z;
            const int s2 = r.hdr[i].w;
            sc[i] = h ? ((s2 & 0x0F0F0F0F)        | ((s0 >> 2) & 0x30303030)) : (s0 & 0x3F3F3F3F);
            mn[i] = h ? (((s2 >> 4) & 0x0F0F0F0F) | ((s1 >> 2) & 0x30303030)) : (s1 & 0x3F3F3F3F);
            dm[i] = mmsb_h2f2(r.hdr[i].x);
            if constexpr (has_qh) {
                qhs[i] = make_int2(r.qh[i].x >> (4*h), r.qh[i].y >> (4*h));
            }
        }
        if constexpr (!has_qh) {
            GGML_UNUSED(qhs);
        }

#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const mmsb_cols col(nt, ne11, g, t);
            const int4 ys_e = *(const int4 *) (yu + col.e*MMSB_YB);   // ds4 of column 8nt+2t
            const int4 ys_o = *(const int4 *) (yu + col.o*MMSB_YB);   // ds4 of column 8nt+2t+1
            const char * yb = yu + col.b*MMSB_YB + 16 + 8*t;

            float accd[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float accm[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const int e = 4*(c & 1);
                ggml_cuda_mma::tile<16, 8, int> A;
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const int2 q = c < 2 ? r.q0[i] : r.q1[i];
                    int ax = (q.x >> e) & 0x0F0F0F0F;
                    int ay = (q.y >> e) & 0x0F0F0F0F;
                    if constexpr (has_qh) {
                        ax |= ((qhs[i].x >> c) << 4) & 0x10101010;
                        ay |= ((qhs[i].y >> c) << 4) & 0x10101010;
                    }
                    A.x[i]     = ax;   // row g / g+8, physical int 2t
                    A.x[i + 2] = ay;   // row g / g+8, physical int 2t+1
                }
                const int2 bv = *(const int2 *) (yb + 32*c);
                ggml_cuda_mma::tile<8, 8, int> B;
                B.x[0] = bv.x;
                B.x[1] = bv.y;
                ggml_cuda_mma::tile<16, 8, int> C;
                ggml_cuda_mma::mma(C, A, B);

                const float2 ds_e = mmsb_h2f2(mmsb_i4(ys_e, c));
                const float2 ds_o = mmsb_h2f2(mmsb_i4(ys_o, c));
                const int   scA = (sc[0] >> (8*c)) & 0xFF;
                const int   scB = (sc[1] >> (8*c)) & 0xFF;
                const float mA  = (float) ((mn[0] >> (8*c)) & 0xFF);
                const float mB  = (float) ((mn[1] >> (8*c)) & 0xFF);

                // |C*sc| <= 32*31*127*63 < 2^24: the int -> float conversion is exact.
                accd[0] += ds_e.x*(float) (C.x[0]*scA);
                accd[1] += ds_o.x*(float) (C.x[1]*scA);
                accd[2] += ds_e.x*(float) (C.x[2]*scB);
                accd[3] += ds_o.x*(float) (C.x[3]*scB);
                accm[0] += ds_e.y*mA;
                accm[1] += ds_o.y*mA;
                accm[2] += ds_e.y*mB;
                accm[3] += ds_o.y*mB;
            }
            out[nt][0] += dm[0].x*accd[0] - dm[0].y*accm[0];
            out[nt][1] += dm[0].x*accd[1] - dm[0].y*accm[1];
            out[nt][2] += dm[1].x*accd[2] - dm[1].y*accm[2];
            out[nt][3] += dm[1].x*accd[3] - dm[1].y*accm[3];
        }
    }
};

template <> struct mmsb_traits<GGML_TYPE_Q4_K> : mmsb_q45_K<false> {};
template <> struct mmsb_traits<GGML_TYPE_Q5_K> : mmsb_q45_K<true>  {};

// Q6_K (210-byte super-blocks, only 2-byte aligned: ql[128], qh[64], int8 scales[16], half d). MMQ's order
// (mmq-vec-dot.cuh q6_K mma: tmp += (C0*sc0 + C1*sc1)*d8, sum += tmp*d) and unpack (mmq-load-tiles.cuh q6_K):
//   unit u = half h of super-block b; chunk c: ql[64h + 32*(c&1) + l] nibble (c>>1), qh[32h + l] bits 2c..2c+1,
//   scale sc[8h + 2c + l/16]; the two 16-value sub-blocks s of a chunk each get one m16n8k16 mma.
// The weights go through get_int_b2 (two 16-bit loads) and stay L1-cached so the two halves merge in L1.
template <>
struct mmsb_traits<GGML_TYPE_Q6_K> {
    struct raw {
        int  ql[2][2][2];   // [row][qq = c&1][s]: ql bytes 64h + 32qq + 16s + 4t..
        int  qh[2][2];      // [row][s]:           qh bytes 32h + 16s + 4t..
        int  sc[2][2];      // [row]: the 8 int8 scales of this half
        half d[2];
    };

    static __device__ __forceinline__ void load(raw & r, const char * xA, const char * xB, const int u, const int t) {
        const int o = (u >> 1)*210;
        const int h = u & 1;
        const char * p[2] = {xA + o, xB + o};
#pragma unroll
        for (int i = 0; i < 2; ++i) {
#pragma unroll
            for (int qq = 0; qq < 2; ++qq) {
#pragma unroll
                for (int s = 0; s < 2; ++s) {
                    r.ql[i][qq][s] = get_int_b2(p[i] + 64*h + 32*qq + 16*s, t);
                }
            }
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                r.qh[i][s] = get_int_b2(p[i] + 128 + 32*h + 16*s, t);
            }
            r.sc[i][0] = get_int_b2(p[i] + 192 + 8*h, 0);
            r.sc[i][1] = get_int_b2(p[i] + 192 + 8*h, 1);
            r.d[i]     = *(const half *) (p[i] + 208);
        }
    }

    template <int NT>
    static __device__ __forceinline__ void compute(const raw & r, const char * yu, const int u, const int ne11,
            const int g, const int t, float (&out)[NT][4]) {
        GGML_UNUSED(u);
        const float dA = __half2float(r.d[0]);
        const float dB = __half2float(r.d[1]);

#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const mmsb_cols col(nt, ne11, g, t);
            const int4 ys_e = *(const int4 *) (yu + col.e*MMSB_YB);   // d4 of column 8nt+2t
            const int4 ys_o = *(const int4 *) (yu + col.o*MMSB_YB);   // d4 of column 8nt+2t+1
            const char * yb = yu + col.b*MMSB_YB + 16 + 4*t;

            float tmp[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const int qq = c & 1;
                const int sh = 4*(c >> 1);
                ggml_cuda_mma::tile<16, 8, int> C[2];
#pragma unroll
                for (int s = 0; s < 2; ++s) {
                    ggml_cuda_mma::tile<16, 4, int> A;
#pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        A.x[i] = __vsubss4(((r.ql[i][qq][s] >> sh) & 0x0F0F0F0F) |
                                           (((r.qh[i][s] >> (2*c)) << 4) & 0x30303030), 0x20202020);
                    }
                    ggml_cuda_mma::tile<8, 4, int> B;
                    B.x[0] = *(const int *) (yb + 32*c + 16*s);
                    ggml_cuda_mma::mma(C[s], A, B);
                }
                const float d8_e = __int_as_float(mmsb_i4(ys_e, c));
                const float d8_o = __int_as_float(mmsb_i4(ys_o, c));
                const int scA0 = mmsb_s8(r.sc[0][c >> 1], 2*(c & 1) + 0);
                const int scA1 = mmsb_s8(r.sc[0][c >> 1], 2*(c & 1) + 1);
                const int scB0 = mmsb_s8(r.sc[1][c >> 1], 2*(c & 1) + 0);
                const int scB1 = mmsb_s8(r.sc[1][c >> 1], 2*(c & 1) + 1);

                // |C0*sc0 + C1*sc1| <= 2*16*32*127*128 < 2^24: exact, as in MMQ.
                tmp[0] += (float) (C[0].x[0]*scA0 + C[1].x[0]*scA1)*d8_e;
                tmp[1] += (float) (C[0].x[1]*scA0 + C[1].x[1]*scA1)*d8_o;
                tmp[2] += (float) (C[0].x[2]*scB0 + C[1].x[2]*scB1)*d8_e;
                tmp[3] += (float) (C[0].x[3]*scB0 + C[1].x[3]*scB1)*d8_o;
            }
            out[nt][0] += tmp[0]*dA;
            out[nt][1] += tmp[1]*dA;
            out[nt][2] += tmp[2]*dB;
            out[nt][3] += tmp[3]*dB;
        }
    }
};

// Q8_0 (34-byte blocks: half d, 32 int8): chunk c of unit u is block 4u + c. MMQ's scaling (mmq-vec-dot.cuh q8_0 mma:
// C*dA*dB). 2-byte aligned rows: get_int_b2, L1-cached.
template <>
struct mmsb_traits<GGML_TYPE_Q8_0> {
    struct raw {
        int  q[2][4][2];   // [row][chunk][bytes 8t..8t+3 / 8t+4..8t+7]
        half d[2][4];      // [row][chunk]
    };

    static __device__ __forceinline__ void load(raw & r, const char * xA, const char * xB, const int u, const int t) {
        const char * p[2] = {xA, xB};
#pragma unroll
        for (int i = 0; i < 2; ++i) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const char * blk = p[i] + (4*u + c)*34;
                r.d[i][c]    = *(const half *) blk;
                r.q[i][c][0] = get_int_b2(blk + 2 + 8*t, 0);
                r.q[i][c][1] = get_int_b2(blk + 6 + 8*t, 0);
            }
        }
    }

    template <int NT>
    static __device__ __forceinline__ void compute(const raw & r, const char * yu, const int u, const int ne11,
            const int g, const int t, float (&out)[NT][4]) {
        GGML_UNUSED(u);
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const mmsb_cols col(nt, ne11, g, t);
            const int4 ys_e = *(const int4 *) (yu + col.e*MMSB_YB);   // d4 of column 8nt+2t
            const int4 ys_o = *(const int4 *) (yu + col.o*MMSB_YB);   // d4 of column 8nt+2t+1
            const char * yb = yu + col.b*MMSB_YB + 16 + 8*t;

#pragma unroll
            for (int c = 0; c < 4; ++c) {
                ggml_cuda_mma::tile<16, 8, int> A;
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    A.x[i]     = r.q[i][c][0];
                    A.x[i + 2] = r.q[i][c][1];
                }
                const int2 bv = *(const int2 *) (yb + 32*c);
                ggml_cuda_mma::tile<8, 8, int> B;
                B.x[0] = bv.x;
                B.x[1] = bv.y;
                ggml_cuda_mma::tile<16, 8, int> C;
                ggml_cuda_mma::mma(C, A, B);

                const float d8_e = __int_as_float(mmsb_i4(ys_e, c));
                const float d8_o = __int_as_float(mmsb_i4(ys_o, c));
                const float dA   = __half2float(r.d[0][c]);
                const float dB   = __half2float(r.d[1][c]);
                out[nt][0] += (float) C.x[0]*dA*d8_e;
                out[nt][1] += (float) C.x[1]*dA*d8_o;
                out[nt][2] += (float) C.x[2]*dB*d8_e;
                out[nt][3] += (float) C.x[3]*dB*d8_o;
            }
        }
    }
};

// ------------------------------------------------------------------------------------------------------------------
// Kernel
// ------------------------------------------------------------------------------------------------------------------

// x: weights (ne01 rows of nb01 bytes); y: q8_1 activations, unit u of column c at y + (u*ne11 + c)*144;
// dst: column c at dst + c*s1_dst. x_pf != 0: the weights do not depend on the previous kernel (a weights buffer), so
// the first unit is loaded before the PDL wait.
template <ggml_type type, int NT>
__launch_bounds__(MMSB_NWARPS*WARP_SIZE, 2)
static __global__ void mul_mat_sb(
        const char * x, const char * y, float * dst, const int U, const int ne01, const int ne11,
        const int64_t nb01, const int s1_dst, const int RG, const int x_pf) {
#if defined(TURING_MMA_AVAILABLE) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    typedef mmsb_traits<type>  tr;
    typedef typename tr::raw   raw;
    constexpr int ncols = 8*NT;

    const int lane = threadIdx.x;
    const int g    = lane >> 2;
    const int t    = lane & 3;
    const int w    = threadIdx.y;
    const int KS   = MMSB_NWARPS/RG;
    const int rg   = w % RG;
    const int ks   = w / RG;

    const int row0 = (blockIdx.x*RG + rg)*16;
    const char * xA = x + (int64_t) min(row0 + g,     ne01 - 1)*nb01;
    const char * xB = x + (int64_t) min(row0 + g + 8, ne01 - 1)*nb01;

    const int u_beg = (ks*U)/KS;
    const int u_end = ((ks + 1)*U)/KS;
    const int64_t y_unit = (int64_t) ne11*MMSB_YB;   // bytes of one unit over all columns

    float out[NT][4];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            out[nt][l] = 0.0f;
        }
    }

    raw r0;
    raw r1;
    if (x_pf && u_beg < u_end) {
        tr::load(r0, xA, xB, u_beg, t);   // weights do not depend on the previous kernel
    }
    ggml_cuda_pdl_sync();                 // every thread, no return before it
    if (!x_pf && u_beg < u_end) {
        tr::load(r0, xA, xB, u_beg, t);
    }

    // Manual 2x ping-pong with one unit of prefetch; the loop body is one basic block (the second load is clamped to
    // the last unit instead of being guarded), so the loads of the next unit overlap the mma work of this one.
    int u = u_beg;
#pragma unroll 1
    for (; u + 1 < u_end; u += 2) {
        tr::load(r1, xA, xB, u + 1, t);
        tr::template compute<NT>(r0, y + u*y_unit, u, ne11, g, t, out);
        tr::load(r0, xA, xB, min(u + 2, u_end - 1), t);
        tr::template compute<NT>(r1, y + (u + 1)*y_unit, u + 1, ne11, g, t, out);
    }
    if (u < u_end) {
        tr::template compute<NT>(r0, y + u*y_unit, u, ne11, g, t, out);
    }

    ggml_cuda_pdl_lc();

    // Split-K reduction: warp w = ks*RG + rg stores its 16 x ncols tile, then the block sums the KS partials of each
    // row group in a fixed order (deterministic) and writes the valid rows/columns.
    extern __shared__ float mmsb_red[];
    float * rw = mmsb_red + w*(ncols*MMSB_RED_STRIDE);
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            rw[(8*nt + 2*t + (l & 1))*MMSB_RED_STRIDE + g + 8*(l >> 1)] = out[nt][l];
        }
    }
    __syncthreads();

    for (int i = w*WARP_SIZE + lane; i < RG*16*ncols; i += MMSB_NWARPS*WARP_SIZE) {
        const int r = i % 16;
        const int c = (i/16) % ncols;
        const int q = i/(16*ncols);
        float s = 0.0f;
        for (int k = 0; k < KS; ++k) {
            s += mmsb_red[((k*RG + q)*ncols + c)*MMSB_RED_STRIDE + r];
        }
        const int row = (blockIdx.x*RG + q)*16 + r;
        if (row < ne01 && c < ne11) {
            dst[(int64_t) c*s1_dst + row] = s;
        }
    }
#else
    GGML_UNUSED_VARS(x, y, dst, U, ne01, ne11, nb01, s1_dst, RG, x_pf);
    NO_DEVICE_CODE;
#endif // defined(TURING_MMA_AVAILABLE) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

// ------------------------------------------------------------------------------------------------------------------
// Host
// ------------------------------------------------------------------------------------------------------------------

bool ggml_cuda_should_use_mmsb(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, const int cc) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA) || defined(GGML_CUDA_FORCE_CUBLAS)
    GGML_UNUSED_VARS(src0, src1, dst, cc);
    return false;
#else
    const mmsb_env_t & env = mmsb_env();
    if (env.mode == 0) {
        return false;                                   // kill switch
    }
    if (!ampere_mma_available(cc)) {
        return false;
    }
    if (env.mode < 0 && cc != GGML_CUDA_CC_BLACKWELL) {
        return false;                                   // default: only where it is measured
    }

    const ggml_type type = src0->type;
    size_t align0;                                      // data and row-stride alignment the weight loads need
    switch (type) {
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
            align0 = 16;
            break;
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_0:
            align0 = 2;
            break;
        default:
            return false;
    }
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne11 = src1->ne[1];
    if (ne00 < MMSB_UNIT || ne00 % MMSB_UNIT != 0 || src1->ne[0] != ne00 || ne01 < 1 || ne01 >= (INT_MAX/2)) {
        return false;
    }
    if (src0->nb[0] != ggml_type_size(type)) {
        return false;
    }
    if ((uintptr_t) src0->data % align0 != 0 || src0->nb[1] % align0 != 0) {
        return false;
    }
    if (src1->nb[0] != sizeof(float) || (uintptr_t) src1->data % 16 != 0 || src1->nb[1] % 16 != 0) {
        return false;
    }
    if (dst->nb[0] != sizeof(float) || dst->nb[1] % sizeof(float) != 0 || dst->nb[1]/sizeof(float) >= (size_t) INT_MAX) {
        return false;
    }
    if (ne11 < 2 || ne11 > env.max_n) {
        return false;                                   // ne11 == 1 stays on MMVQ (fused gate/up/GLU)
    }
    if (env.max_rows > 0 && ne01 > env.max_rows) {
        return false;
    }
    // Phase gate: by default MMVQ keeps every width it takes today (Q4_K/Q5_K <= 5, Q6_K <= 7, Q8_0 and weights with
    // fewer than 128 rows <= 8 on Blackwell). GGML_CUDA_SMALLB_MIN=n also takes those widths from n up.
    if (ne11 < (env.min_n ? env.min_n : INT_MAX) && ggml_cuda_should_use_mmvq(type, cc, ne11, ne01)) {
        return false;
    }
    return true;
#endif // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA) || defined(GGML_CUDA_FORCE_CUBLAS)
}

template <ggml_type type, int NT>
static void mmsb_launch(const char * x, const char * y, float * dst, const int U, const int ne01, const int ne11,
        const int64_t nb01, const int s1_dst, const int RG, const int x_pf, const int nblocks, cudaStream_t stream) {
    const size_t nbytes_shared = (size_t) MMSB_NWARPS*(8*NT)*MMSB_RED_STRIDE*sizeof(float);
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    // Prefer L1 over shared memory (the activations are re-read from L1 by every row group of a block), but keep the
    // 2 blocks per SM of __launch_bounds__: a 0% (MaxL1) carveout rounds up to the smallest config that fits 1 block
    // (16 KB for NT = 2), which halves the occupancy. Ask for exactly 2 blocks (+ reserved), rounded up by the driver.
    {
        static bool carveout_set[GGML_CUDA_MAX_DEVICES] = { false };
        const int id = ggml_cuda_get_device();
        if (!carveout_set[id]) {
            int smem_sm  = 0;
            int reserved = 0;
            CUDA_CHECK(cudaDeviceGetAttribute(&smem_sm,  cudaDevAttrMaxSharedMemoryPerMultiprocessor, id));
            CUDA_CHECK(cudaDeviceGetAttribute(&reserved, cudaDevAttrReservedSharedMemoryPerBlock,     id));
            const int64_t per_block = (((int64_t) nbytes_shared + reserved + 127)/128)*128;
            const int     pct       = smem_sm > 0 ? (int) std::min<int64_t>(100, (100*2*per_block + smem_sm - 1)/smem_sm)
                                                  : (int) cudaSharedmemCarveoutDefault;
            CUDA_CHECK(cudaFuncSetAttribute(mul_mat_sb<type, NT>, cudaFuncAttributePreferredSharedMemoryCarveout, pct));
            carveout_set[id] = true;
        }
    }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    const dim3 block_nums(nblocks, 1, 1);
    const dim3 block_dims(WARP_SIZE, MMSB_NWARPS, 1);
    ggml_cuda_kernel_launch(mul_mat_sb<type, NT>, ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream),
        x, y, dst, U, ne01, ne11, nb01, s1_dst, RG, x_pf);
}

template <ggml_type type>
static void mmsb_launch_nt(const int NT, const char * x, const char * y, float * dst, const int U, const int ne01,
        const int ne11, const int64_t nb01, const int s1_dst, const int RG, const int x_pf, const int nblocks,
        cudaStream_t stream) {
    if (NT == 1) {
        mmsb_launch<type, 1>(x, y, dst, U, ne01, ne11, nb01, s1_dst, RG, x_pf, nblocks, stream);
    } else {
        mmsb_launch<type, 2>(x, y, dst, U, ne01, ne11, nb01, s1_dst, RG, x_pf, nblocks, stream);
    }
}

void ggml_cuda_mul_mat_sb(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS;

    GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ne00 % MMSB_UNIT == 0);
    GGML_ASSERT(ne11 >= 1 && ne11 <= MMSB_MAX_COLS);
    GGML_ASSERT(nb10 == sizeof(float) && nb0 == sizeof(float));

    const mmsb_env_t & env = mmsb_env();
    cudaStream_t stream = ctx.stream();
    const int id  = ggml_cuda_get_device();
    const int nsm = ggml_cuda_info().devices[id].nsm;

    // 1. q8_1 activations in MMQ's block layout: unit u of column c at (u*ne11 + c)*144. The size is exact (ne00 is a
    //    multiple of 128, no padding is read). DS4 for Q4_K/Q5_K, D4 for Q6_K/Q8_0 (mmq_get_q8_1_ds_layout).
    const int U = (int) (ne00/MMSB_UNIT);
    ggml_cuda_pool_alloc<char> q8(ctx.pool(), (size_t) ne11*U*sizeof(block_q8_1_mmq));
    quantize_mmq_q8_1_pdl_cuda((const float *) src1->data, nullptr, q8.get(), src0->type,
        ne10, nb11/sizeof(float), nb12/sizeof(float), nb13/sizeof(float), /*ne0 =*/ ne10, ne11, 1, 1, stream);

    // 2. Geometry: NT n8 tiles per warp; RG = the largest of {4, 2} that still gives >= 15/16 of the SMs a block,
    //    else 1 (KS = 8/RG warps split K).
    const int NT = ne11 <= 8 ? 1 : 2;
    const int64_t ntiles16 = (ne01 + 15)/16;
    int RG = env.rg;
    if (RG == 0) {
        const int64_t thr = (int64_t) nsm*15/16;
        RG = (ntiles16 + 3)/4 >= thr ? 4 : ((ntiles16 + 1)/2 >= thr ? 2 : 1);
    }
    const int nblocks = (int) ((ntiles16 + RG - 1)/RG);
    const int x_pf = (env.xpf || (src0->buffer && ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS)) ? 1 : 0;

    if (env.probe) {
        static std::mutex    probe_mutex;
        static std::set<int> probe_seen;
        const int key = ((int) src0->type << 8) | (NT << 4) | RG;
        std::lock_guard<std::mutex> lock(probe_mutex);
        if (probe_seen.insert(key).second) {
            fprintf(stderr, "turbo-probe: mmsb type=%s NT=%d RG=%d (first: ne01=%lld ne00=%lld ne11=%lld blocks=%d x_pf=%d)\n",
                    ggml_type_name(src0->type), NT, RG, (long long) ne01, (long long) ne00, (long long) ne11, nblocks, x_pf);
            fflush(stderr);
        }
    }

    const char * x      = (const char *) src0->data;
    const char * y      = (const char *) q8.get();
    float       * dst_d = (float *) dst->data;
    const int     s1_dst = (int) (nb1/sizeof(float));

    switch (src0->type) {
        case GGML_TYPE_Q4_K:
            mmsb_launch_nt<GGML_TYPE_Q4_K>(NT, x, y, dst_d, U, (int) ne01, (int) ne11, (int64_t) nb01, s1_dst, RG, x_pf, nblocks, stream);
            break;
        case GGML_TYPE_Q5_K:
            mmsb_launch_nt<GGML_TYPE_Q5_K>(NT, x, y, dst_d, U, (int) ne01, (int) ne11, (int64_t) nb01, s1_dst, RG, x_pf, nblocks, stream);
            break;
        case GGML_TYPE_Q6_K:
            mmsb_launch_nt<GGML_TYPE_Q6_K>(NT, x, y, dst_d, U, (int) ne01, (int) ne11, (int64_t) nb01, s1_dst, RG, x_pf, nblocks, stream);
            break;
        case GGML_TYPE_Q8_0:
            mmsb_launch_nt<GGML_TYPE_Q8_0>(NT, x, y, dst_d, U, (int) ne01, (int) ne11, (int64_t) nb01, s1_dst, RG, x_pf, nblocks, stream);
            break;
        default:
            GGML_ABORT("mmsb: unsupported type %s", ggml_type_name(src0->type));
    }
}
