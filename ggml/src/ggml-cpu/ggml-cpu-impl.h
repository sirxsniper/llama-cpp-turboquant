#pragma once

// GGML CPU internal header

#include "ggml.h"
#include "ggml-impl.h"

#include <stdlib.h> // load `stdlib.h` before other headers to work around MinGW bug: https://sourceforge.net/p/mingw-w64/bugs/192/
//#include <stddef.h>
#include <stdbool.h>
#include <string.h> // memcpy
#include <math.h>   // fabsf

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_compute_params {
    // ith = thread index, nth = number of threads
    int ith, nth;

    // work buffer for all threads
    size_t wsize;
    void * wdata;

    struct ggml_threadpool * threadpool;

    // use reference implementation
    bool use_ref;
};

// [TAG_FN_CPU_SWITCHES] values of the switches of enum ggml_cpu_fn_switch (ggml-cpu.h), indexed by it: set once in
// ggml_cpu_init() from the environment and afterwards only by ggml_cpu_fn_set_switch() while no graph is computing
extern int ggml_cpu_fn_sw[];

// [TAG_FN_R4_VNNI] the CPU runs AVX512-VNNI (CPUID, set once in ggml_cpu_init())
extern bool ggml_cpu_fn_vnni_cpu;

// [TAG_FN_CPU_MOE_FUSE] the fused CPU MoE split (ggml-cpu.c), shared by the graph op and the worker pool (moe-pool.cpp)
#define GGML_FN_MOE_MAX_T 16

struct ggml_fn_moe_args {
    const struct ggml_tensor * up;      // [n_embd, n_ff, n_expert]
    const struct ggml_tensor * gate;    // [n_embd, n_ff, n_expert]
    const struct ggml_tensor * down;    // [n_ff, n_embd, n_expert]
    const int32_t * table;              // or NULL: expert e is computed only if table[e] == table_miss
    int32_t         table_miss;
    int             n_tokens;           // 1..GGML_FN_MOE_MAX_T
    int             n_used;
    const char *    x;                  // token t: (const float *) (x + t*x_nb)
    size_t          x_nb;
    const char *    ids;                // ids[slot, t]: *(const int32_t *) (ids + slot*ids_nb0 + t*ids_nb1)
    size_t          ids_nb0;
    size_t          ids_nb1;
    char *          out;                // row (slot, t): (float *) (out + slot*out_nb1 + t*out_nb2), n_embd floats
    size_t          out_nb1;
    size_t          out_nb2;
    const float *   w;                  // or NULL: [n_used, n_tokens] weights of the weighted sum
    float *         out_sum;            // w != NULL: [n_embd, n_tokens]
    void *          wdata;              // ggml_fn_moe_work_size() bytes

    // [TAG_FN_L3_CPU_SPLIT] (the worker pool only; zero = the default split) how the pieces go to the threads:
    // 0: contiguous ranges of the expert-major piece list; 1: piece p of expert e to thread (h(e) + p) mod nth, the same
    // owner the stable prefetch uses; 2: as 1, then a thread with no own piece left takes the last free pieces of the
    // others (claims, one word per piece). Any owner computes the same values.
    int             split;
    int32_t *       claim;              // split 2: GGML_FN_MOE_CLAIM_WORDS() words, kept between jobs
    int32_t         epoch;              // split 2: != 0 and != the previous job's
    const int32_t * prio;               // split >= 1, or NULL: experts computed first, in this order (the prefetch's list)
    int             n_prio;
    uint64_t *      ts;                 // [TAG_FN_L3_CPU_STATS] or NULL: GGML_FN_MOE_TS_N words per thread (ts + ith*GGML_FN_MOE_TS_N)
    const uint8_t * pf_done;            // [TAG_FN_L3_CPU_STATS] stats with split >= 1, or NULL: [nth][n_expert], 1 = the
                                        // thread's stable prefetch covered all its pieces of the expert
    int             swpf;               // [TAG_FN_L3_CPU_SWPF] 0, or software-prefetch this many cache lines at every 4 KiB
                                        // page of a thread's next piece before it computes the current one
};

// [TAG_FN_L3_CPU_STATS] the per-thread record of a job (ggml_fn_moe_args.ts): ticks of ggml_fn_moe_tick()
enum {
    GGML_FN_MOE_TS_START = 0,  // entry
    GGML_FN_MOE_TS_P3    = 1,  // gate / up pieces done (own and taken)
    GGML_FN_MOE_TS_B1    = 2,  // after the barrier before down
    GGML_FN_MOE_TS_P4    = 3,  // down pieces done
    GGML_FN_MOE_TS_B2    = 4,  // after the barrier before the weighted sum (= P4 without one)
    GGML_FN_MOE_TS_END   = 5,  // exit
    GGML_FN_MOE_TS_BYTES = 6,  // weight bytes of the pieces computed
    GGML_FN_MOE_TS_TAKEN = 7,  // pieces taken from other threads (split 2)
    GGML_FN_MOE_TS_PFHIT = 8,  // weight bytes of pieces whose expert this thread's prefetch covered (split >= 1)
    GGML_FN_MOE_TS_NACT  = 9,  // experts the job computes
    GGML_FN_MOE_TS_N     = 16, // words per thread (two cache lines)
};

// [TAG_FN_L3_CPU_SPLIT] claim words a split-2 job needs: two phases x nth threads x one padded run per thread
#define GGML_FN_MOE_CLAIM_RUN(n_act, np_max, nth) ((((int64_t) (n_act)*(((np_max) + (nth) - 1)/(nth)) + 15)/16)*16)
#define GGML_FN_MOE_CLAIM_WORDS(n_act, np_max, nth) (2*(int64_t) (nth)*GGML_FN_MOE_CLAIM_RUN(n_act, np_max, nth))

// [TAG_FN_L3_CPU_SPLIT] the first owner of expert e's pieces (piece p goes to (owner0 + p) mod nth)
static inline int ggml_fn_moe_owner0(int32_t e, int nth) {
    return (int) (((uint32_t) e*2654435761u >> 16) % (uint32_t) nth);
}

// [TAG_FN_L3_CPU_SPLIT] the largest number of pieces of one expert in either phase (the claim run size)
int64_t ggml_fn_moe_np_max(const struct ggml_tensor * up, const struct ggml_tensor * down);

// [TAG_FN_L3_CPU_STATS] a cheap monotonic tick (the TSC on x86, else microseconds)
uint64_t ggml_fn_moe_tick(void);

// false if the fused kernel does not take these weights (types, shapes)
bool   ggml_fn_moe_supported(const struct ggml_tensor * up, const struct ggml_tensor * gate, const struct ggml_tensor * down);
size_t ggml_fn_moe_work_size(const struct ggml_tensor * up, const struct ggml_tensor * gate, const struct ggml_tensor * down,
                             int n_used, int n_tokens, int nth);
// every one of the nth threads calls it with its ith; barrier(barrier_ctx) must synchronize all of them
void   ggml_fn_moe_compute(const struct ggml_fn_moe_args * a, int ith, int nth, void (*barrier)(void *), void * barrier_ctx);
// [TAG_FN_R2_BRIDGE_SYNC] GGML_OP_MOE_HOST_SUM on the CPU: types, shapes and the fused kernel (ggml-cpu.c)
bool   ggml_cpu_moe_host_sum_supported(const struct ggml_tensor * op);
// [TAG_FN_R2_BRIDGE_PF] prefetch the pieces thread ith of nth reads in a job of the experts list[0..n) (ascending);
// stops before the next piece once *stop != 0; returns the bytes covered. mode 0: real loads, 1: software prefetches
size_t ggml_fn_moe_prefetch(const struct ggml_tensor * up, const struct ggml_tensor * gate, const struct ggml_tensor * down,
                            const int32_t * list, int n, int ith, int nth, const volatile int32_t * stop, int mode);
// [TAG_FN_L3_CPU_SPLIT] the same for a split >= 1 job: thread ith's own pieces of the experts list[0..n) in list order
// (per expert its gate + up pieces, then its down pieces). done (or NULL): [n_expert], set to 1 for every expert whose
// pieces of this thread were all pulled before the stop.
size_t ggml_fn_moe_prefetch_stable(const struct ggml_tensor * up, const struct ggml_tensor * gate, const struct ggml_tensor * down,
                                   const int32_t * list, int n, int ith, int nth, const volatile int32_t * stop, int mode,
                                   uint8_t * done);


#if defined(_MSC_VER)

#define m512bh(p) p
#define m512i(p) p

#else

#define m512bh(p) (__m512bh)(p)
#define m512i(p) (__m512i)(p)

#endif

// __FMA__ and __F16C__ are not defined in MSVC, however they are implied with AVX2/AVX512
#if defined(_MSC_VER) && (defined(__AVX2__) || defined(__AVX512F__))
#ifndef __FMA__
#define __FMA__
#endif
#ifndef __F16C__
#define __F16C__
#endif
#endif

// __SSE3__ and __SSSE3__ are not defined in MSVC, but SSE3/SSSE3 are present when AVX/AVX2/AVX512 are available
#if defined(_MSC_VER) && (defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__))
#ifndef __SSE3__
#define __SSE3__
#endif
#ifndef __SSSE3__
#define __SSSE3__
#endif
#endif

#if defined(__s390x__) && defined(__VEC__)
#ifndef __VXE__
#define __VXE__
#endif  // __VXE__
#ifndef __VXE2__
#define __VXE2__
#endif  // __VXE2__
#endif  // __s390x__ && __VEC__

#if defined(__ARM_FEATURE_SVE) && defined(__linux__)
#include <sys/prctl.h>
#endif

#if defined(__ARM_NEON)

// ref: https://github.com/ggml-org/llama.cpp/pull/5404
#if defined(_MSC_VER) && !defined(__clang__)
#define ggml_vld1q_u32(w,x,y,z) { ((w) + ((uint64_t)(x) << 32)), ((y) + ((uint64_t)(z) << 32)) }
#else
#define ggml_vld1q_u32(w,x,y,z) { (w), (x), (y), (z) }
#endif // _MSC_VER

#if !defined(__aarch64__)

// 32-bit ARM compatibility

// vaddlvq_s16
// vpaddq_s16
// vpaddq_s32
// vaddvq_s32
// vaddvq_f32
// vmaxvq_f32
// vcvtnq_s32_f32
// vzip1_u8
// vzip2_u8

inline static int32_t vaddlvq_s16(int16x8_t v) {
    int32x4_t v0 = vreinterpretq_s32_s64(vpaddlq_s32(vpaddlq_s16(v)));
    return vgetq_lane_s32(v0, 0) + vgetq_lane_s32(v0, 2);
}

inline static int16x8_t vpaddq_s16(int16x8_t a, int16x8_t b) {
    int16x4_t a0 = vpadd_s16(vget_low_s16(a), vget_high_s16(a));
    int16x4_t b0 = vpadd_s16(vget_low_s16(b), vget_high_s16(b));
    return vcombine_s16(a0, b0);
}

inline static int32x4_t vpaddq_s32(int32x4_t a, int32x4_t b) {
    int32x2_t a0 = vpadd_s32(vget_low_s32(a), vget_high_s32(a));
    int32x2_t b0 = vpadd_s32(vget_low_s32(b), vget_high_s32(b));
    return vcombine_s32(a0, b0);
}

inline static int32_t vaddvq_s32(int32x4_t v) {
    return vgetq_lane_s32(v, 0) + vgetq_lane_s32(v, 1) + vgetq_lane_s32(v, 2) + vgetq_lane_s32(v, 3);
}

inline static float vaddvq_f32(float32x4_t v) {
    return vgetq_lane_f32(v, 0) + vgetq_lane_f32(v, 1) + vgetq_lane_f32(v, 2) + vgetq_lane_f32(v, 3);
}

inline static float vmaxvq_f32(float32x4_t v) {
    return
        MAX(MAX(vgetq_lane_f32(v, 0), vgetq_lane_f32(v, 1)),
            MAX(vgetq_lane_f32(v, 2), vgetq_lane_f32(v, 3)));
}

inline static int32x4_t vcvtnq_s32_f32(float32x4_t v) {
    int32x4_t res;

    res[0] = roundf(vgetq_lane_f32(v, 0));
    res[1] = roundf(vgetq_lane_f32(v, 1));
    res[2] = roundf(vgetq_lane_f32(v, 2));
    res[3] = roundf(vgetq_lane_f32(v, 3));

    return res;
}

inline static uint8x8_t vzip1_u8(uint8x8_t a, uint8x8_t b) {
    uint8x8_t res;

    res[0] = a[0]; res[1] = b[0];
    res[2] = a[1]; res[3] = b[1];
    res[4] = a[2]; res[5] = b[2];
    res[6] = a[3]; res[7] = b[3];

    return res;
}

inline static uint8x8_t vzip2_u8(uint8x8_t a, uint8x8_t b) {
    uint8x8_t res;

    res[0] = a[4]; res[1] = b[4];
    res[2] = a[5]; res[3] = b[5];
    res[4] = a[6]; res[5] = b[6];
    res[6] = a[7]; res[7] = b[7];

    return res;
}

// vld1q_s16_x2
// vld1q_u8_x2
// vld1q_u8_x4
// vld1q_s8_x2
// vld1q_s8_x4
// TODO: double-check these work correctly

typedef struct ggml_int16x8x2_t {
    int16x8_t val[2];
} ggml_int16x8x2_t;

inline static ggml_int16x8x2_t ggml_vld1q_s16_x2(const int16_t * ptr) {
    ggml_int16x8x2_t res;

    res.val[0] = vld1q_s16(ptr + 0);
    res.val[1] = vld1q_s16(ptr + 8);

    return res;
}

typedef struct ggml_uint8x16x2_t {
    uint8x16_t val[2];
} ggml_uint8x16x2_t;

inline static ggml_uint8x16x2_t ggml_vld1q_u8_x2(const uint8_t * ptr) {
    ggml_uint8x16x2_t res;

    res.val[0] = vld1q_u8(ptr + 0);
    res.val[1] = vld1q_u8(ptr + 16);

    return res;
}

typedef struct ggml_uint8x16x4_t {
    uint8x16_t val[4];
} ggml_uint8x16x4_t;

inline static ggml_uint8x16x4_t ggml_vld1q_u8_x4(const uint8_t * ptr) {
    ggml_uint8x16x4_t res;

    res.val[0] = vld1q_u8(ptr + 0);
    res.val[1] = vld1q_u8(ptr + 16);
    res.val[2] = vld1q_u8(ptr + 32);
    res.val[3] = vld1q_u8(ptr + 48);

    return res;
}

typedef struct ggml_int8x16x2_t {
    int8x16_t val[2];
} ggml_int8x16x2_t;

inline static ggml_int8x16x2_t ggml_vld1q_s8_x2(const int8_t * ptr) {
    ggml_int8x16x2_t res;

    res.val[0] = vld1q_s8(ptr + 0);
    res.val[1] = vld1q_s8(ptr + 16);

    return res;
}

typedef struct ggml_int8x16x4_t {
    int8x16_t val[4];
} ggml_int8x16x4_t;

inline static ggml_int8x16x4_t ggml_vld1q_s8_x4(const int8_t * ptr) {
    ggml_int8x16x4_t res;

    res.val[0] = vld1q_s8(ptr + 0);
    res.val[1] = vld1q_s8(ptr + 16);
    res.val[2] = vld1q_s8(ptr + 32);
    res.val[3] = vld1q_s8(ptr + 48);

    return res;
}

// NOTE: not tested
inline static int8x16_t ggml_vqtbl1q_s8(int8x16_t a, uint8x16_t b) {
    int8x16_t res;

    res[ 0] = a[b[ 0]];
    res[ 1] = a[b[ 1]];
    res[ 2] = a[b[ 2]];
    res[ 3] = a[b[ 3]];
    res[ 4] = a[b[ 4]];
    res[ 5] = a[b[ 5]];
    res[ 6] = a[b[ 6]];
    res[ 7] = a[b[ 7]];
    res[ 8] = a[b[ 8]];
    res[ 9] = a[b[ 9]];
    res[10] = a[b[10]];
    res[11] = a[b[11]];
    res[12] = a[b[12]];
    res[13] = a[b[13]];
    res[14] = a[b[14]];
    res[15] = a[b[15]];

    return res;
}

// NOTE: not tested
inline static uint8x16_t ggml_vqtbl1q_u8(uint8x16_t a, uint8x16_t b) {
    uint8x16_t res;

    res[ 0] = a[b[ 0]];
    res[ 1] = a[b[ 1]];
    res[ 2] = a[b[ 2]];
    res[ 3] = a[b[ 3]];
    res[ 4] = a[b[ 4]];
    res[ 5] = a[b[ 5]];
    res[ 6] = a[b[ 6]];
    res[ 7] = a[b[ 7]];
    res[ 8] = a[b[ 8]];
    res[ 9] = a[b[ 9]];
    res[10] = a[b[10]];
    res[11] = a[b[11]];
    res[12] = a[b[12]];
    res[13] = a[b[13]];
    res[14] = a[b[14]];
    res[15] = a[b[15]];

    return res;
}

#else

#define ggml_int16x8x2_t  int16x8x2_t
#define ggml_uint8x16x2_t uint8x16x2_t
#define ggml_uint8x16x4_t uint8x16x4_t
#define ggml_int8x16x2_t  int8x16x2_t
#define ggml_int8x16x4_t  int8x16x4_t

#define ggml_vld1q_s16_x2 vld1q_s16_x2
#define ggml_vld1q_u8_x2  vld1q_u8_x2
#define ggml_vld1q_u8_x4  vld1q_u8_x4
#define ggml_vld1q_s8_x2  vld1q_s8_x2
#define ggml_vld1q_s8_x4  vld1q_s8_x4
#define ggml_vqtbl1q_s8   vqtbl1q_s8
#define ggml_vqtbl1q_u8   vqtbl1q_u8

#endif // !defined(__aarch64__)

#if !defined(__ARM_FEATURE_DOTPROD)

// NOTE: this fallback produces the same total sum as native vdotq_s32 but with different per-lane grouping — do not use when individual lane values matter.
inline static int32x4_t ggml_vdotq_s32(int32x4_t acc, int8x16_t a, int8x16_t b) {
    const int16x8_t p0 = vmull_s8(vget_low_s8 (a), vget_low_s8 (b));
    const int16x8_t p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));

    return vaddq_s32(acc, vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)));
}

#else

#define ggml_vdotq_s32(a, b, c) vdotq_s32(a, b, c)

#endif // !defined(__ARM_FEATURE_DOTPROD)

static inline int32x4_t ggml_nvfp4_dot8(const int8x8_t q4_lo, const int8x8_t q8_lo,
                                         const int8x8_t q4_hi, const int8x8_t q8_hi) {
    const int16x8_t p_lo = vmull_s8(q4_lo, q8_lo);
    const int16x8_t p_hi = vmull_s8(q4_hi, q8_hi);
    const int32x4_t sum_lo = vpaddlq_s16(p_lo);
    const int32x4_t sum_hi = vpaddlq_s16(p_hi);
    return vaddq_s32(sum_lo, sum_hi);
}

#endif // defined(__ARM_NEON)

#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#endif

#ifdef __POWER9_VECTOR__
#include <altivec.h>
#endif

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <intrin.h>
#elif defined(__SSE__) || defined(__SSE3__) || defined(__SSSE3__) || defined(__AVX__) || defined(__F16C__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX512BF16__)
#include <immintrin.h>
#endif

#ifdef __riscv_v_intrinsic
#include <riscv_vector.h>
#endif

#if defined(__loongarch64)
#if defined(__loongarch_asx)
#include <lasxintrin.h>
#endif
#if defined(__loongarch_sx)
#include <lsxintrin.h>
#endif
#endif

#if defined(__VXE__) || defined(__VXE2__)
#include <vecintrin.h>

#define vec_neg(a)    (-(a))                // Vector Negate
#define vec_add(a, b) ((a) + (b))           // Vector Add
#define vec_sub(a, b) ((a) - (b))           // Vector Subtract
#define vec_mul(a, b) ((a) * (b))           // Vector Multiply
#define vec_div(a, b) ((a) / (b))           // Vector Divide
#define vec_sl(a, b)  ((a) << (b))          // Vector Shift Left
#define vec_sra(a, b) ((a) >> (b))          // Vector Shift Right
#define vec_sr(a, b)  ((a) >> (b))          // Vector Shift Right Algebraic
#define vec_slo(a, b) vec_slb(a, (b) << 64) // Vector Shift Left by Octet
#define vec_sro(a, b) vec_srb(a, (b) << 64) // Vector Shift Right by Octet

#ifndef vec_and
#define vec_and(a, b) ((a) & (b)) // Vector AND
#endif

#ifndef vec_or
#define vec_or(a, b)  ((a) | (b)) // Vector OR
#endif

#ifndef vec_xor
#define vec_xor(a, b) ((a) ^ (b)) // Vector XOR
#endif

typedef signed   char char8x16_t  __attribute__((vector_size(16)));
typedef unsigned char uchar8x16_t __attribute__((vector_size(16)));

typedef int8_t  int8x16_t __attribute__((vector_size(16)));
typedef int16_t int16x8_t __attribute__((vector_size(16)));
typedef int32_t int32x4_t __attribute__((vector_size(16)));

typedef uint8_t  uint8x16_t __attribute__((vector_size(16)));
typedef uint16_t uint16x8_t __attribute__((vector_size(16)));
typedef uint32_t uint32x4_t __attribute__((vector_size(16)));

typedef float  float32x4_t  __attribute__((vector_size(16)));
typedef double double64x2_t __attribute__((vector_size(16)));

typedef signed   long long long64x2_t  __attribute__((vector_size(16)));
typedef unsigned long long ulong64x2_t __attribute__((vector_size(16)));

typedef struct ggml_uint8x16x2_t {
    uint8x16_t val[2];
} ggml_uint8x16x2_t;

inline static ggml_uint8x16x2_t ggml_vec_xl_u8x2(const uint8_t * ptr) {
    ggml_uint8x16x2_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);

    return res;
}

typedef struct ggml_uint8x16x4_t {
    uint8x16_t val[4];
} ggml_uint8x16x4_t;

inline static ggml_uint8x16x4_t ggml_vec_xl_u8x4(const uint8_t * ptr) {
    ggml_uint8x16x4_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);
    res.val[2] = vec_xl(32, ptr);
    res.val[3] = vec_xl(48, ptr);

    return res;
}

typedef struct ggml_int8x16x4_t {
    int8x16_t val[4];
} ggml_int8x16x4_t;

inline static ggml_int8x16x4_t ggml_vec_xl_s8x4(const int8_t * ptr) {
    ggml_int8x16x4_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);
    res.val[2] = vec_xl(32, ptr);
    res.val[3] = vec_xl(48, ptr);

    return res;
}

typedef struct ggml_int16x8x2_t {
    int16x8_t val[2];
} ggml_int16x8x2_t;

inline static ggml_int16x8x2_t ggml_vec_xl_s16x2(const int16_t * ptr) {
    ggml_int16x8x2_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);

    return res;
}

/*
    ! WARNING: Very slow. Use vec_perm if possible. Refer to iq4_xs
    !          or iq4_nl for example implementation.
*/
inline static int8x16_t ggml_vec_tbl(int8x16_t a, uint8x16_t b) {
    int8x16_t res;

    res[ 0] = a[b[ 0]];
    res[ 1] = a[b[ 1]];
    res[ 2] = a[b[ 2]];
    res[ 3] = a[b[ 3]];
    res[ 4] = a[b[ 4]];
    res[ 5] = a[b[ 5]];
    res[ 6] = a[b[ 6]];
    res[ 7] = a[b[ 7]];
    res[ 8] = a[b[ 8]];
    res[ 9] = a[b[ 9]];
    res[10] = a[b[10]];
    res[11] = a[b[11]];
    res[12] = a[b[12]];
    res[13] = a[b[13]];
    res[14] = a[b[14]];
    res[15] = a[b[15]];

    return res;
}

inline static int16x8_t vec_padd_s16(int16x8_t a, int16x8_t b) {
    const uchar8x16_t v_maske = {  0,  1,  4,  5,  8,  9, 12, 13,
                                  16, 17, 20, 21, 24, 25, 28, 29 };

    const int16x8_t v_abo = vec_pack((int32x4_t)a, (int32x4_t)b);
    const int16x8_t v_abe = vec_perm(a, b, v_maske);
    return v_abo + v_abe;
}

/**
 * @see https://github.com/ggml-org/llama.cpp/pull/14037
 */
inline static float vec_hsum_f32x4(float32x4_t v) {
    float32x4_t v_temp = v + vec_reve(v);
    return v_temp[0] + v_temp[1];
}

inline static int32_t vec_hsum_i32x4(int32x4_t v) {
    int32x4_t v_temp = v + vec_reve(v);
    return v_temp[0] + v_temp[1];
}

inline static int32x4_t ggml_vec_dot(int32x4_t acc, int8x16_t a, int8x16_t b) {
    const int16x8_t p = vec_mule(a, b) + vec_mulo(a, b);
    return acc + (vec_unpackh(p) + vec_unpackl(p));
}

#endif

#if defined(__loongarch_sx)
/* float type data load instructions */
static __m128 __lsx_vreplfr2vr_s(const float val) {
    v4f32 res = {val, val, val, val};
    return (__m128)res;
}
#endif

#if defined(__loongarch_asx)
static __m256 __lasx_xvreplfr2vr_s(const float val) {
    v8f32 res = {val, val, val, val, val, val, val, val};
    return (__m256)res;
}
#endif

// TODO: move to ggml-threading
void ggml_barrier(struct ggml_threadpool * tp);

void ggml_threadpool_chunk_set(struct ggml_threadpool * tp, int value);
int  ggml_threadpool_chunk_add(struct ggml_threadpool * tp, int value);

#ifdef __cplusplus
}
#endif
