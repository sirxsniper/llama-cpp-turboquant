#pragma once

// [TAG_TURBOT] Shared definitions of the turbot tiered KV cache format. docs/turbot/SPEC.md is the contract; this
// header is its executable part. Owned by the architect: implementers include it and do NOT re-derive any of this.
//
// Usable from C (ggml.c, ggml-cpu), C++ (llama, tests) and the host side of CUDA translation units. Nothing here
// is device code: the CUDA kernels take the same tables through ggml-cuda/turbot-tables.cuh.
//
// Summary (SPEC sections 3-4):
//   - one attention layer-side (K or V) row = 1024 values = 4 KV heads x 256 = 8 WHT-128 groups
//   - base row, every cell:      head runs of 32*b[h] bytes (planes P4, P2, P1), then 8 ggml_fp16_t gains = 32*S + 16
//   - young refinement per cell: head runs of 32*(y[h]-b[h]) bytes, then 8 gains              = 32*R + 16
//   - old read:   C_b[code_b] * gain_b
//   - young read: LUT_{b,y}[(code_b << (y-b)) | code_r] * gain_y      (one lookup, "a_lloyd" nested code)
//   - values are in the rotated (signed WHT-128) domain, exactly like turbo5p
//
// [TAG_TURBOT_ANY_GEOM] other shapes (SPEC section 14): a layer-side row holds NR = 1, 2 or 4 runs of 256 values, one
// old/young width per run. Run r holds values [256r, 256r+256): with head dim 256 run z is head z, with head dim 128
// head z sits in run z>>1 at element 128*(z&1). The geometry travels as `flags` (op params byte 31, 0 = the 4 x 256
// shape above); a row is still 32*S + 16 bytes with 8 gain slots (2*NR used) inside a [1024, kv_size] container tensor.
// flags 0 and NR 4 run exactly the code paths and numbers of the 4 x 256 format.

#include "ggml.h"
#include "ggml-turbot-tables.h"

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__) || defined(__clang__)
#    define GGML_TURBOT_UNUSED __attribute__((unused))
#else
#    define GGML_TURBOT_UNUSED
#endif

//
// geometry and limits
//

#define GGML_TURBOT_HEAD_DIM         256    // head dim of the 4 x 256 geometry (flags 0); others: ggml_turbot_geom_head_dim
#define GGML_TURBOT_N_HEAD           4      // KV heads per layer-side row at flags 0 (= runs); others: ggml_turbot_geom_n_head
#define GGML_TURBOT_GROUP            128    // WHT group
#define GGML_TURBOT_GROUPS_PER_HEAD  2      // WHT groups per run (per head at head dim 256)
#define GGML_TURBOT_ROW_ELEMS        1024   // container width: blck_size of every turbot type and ne[0] of every base
                                            // tensor; the values of a row are ggml_turbot_geom_row_elems (NR*256)
#define GGML_TURBOT_N_GAINS          8      // gain slots per row, gain[2*r + g]; runs r >= NR leave theirs 0
#define GGML_TURBOT_GRANULE          64     // cells per granule (tier unit)
#define GGML_TURBOT_LOG2_GRANULE     6
#define GGML_TURBOT_B_MIN            2      // old width per head
#define GGML_TURBOT_B_MAX            6
#define GGML_TURBOT_Y_MAX            8      // young width per head, b < y <= 8
#define GGML_TURBOT_Y_DEFAULT        7
#define GGML_TURBOT_R_MAX            6      // y - b
#define GGML_TURBOT_S_MIN            8      // S = sum of the 4 old widths of a layer-side (NR = 4); first of the S8..S24 ids
#define GGML_TURBOT_S_MAX            24
#define GGML_TURBOT_N_TYPES          (GGML_TURBOT_S_MAX - GGML_TURBOT_S_MIN + 1)   // the S8..S24 block

// [TAG_TURBOT_ANY_GEOM] geometry (SPEC section 14). flags = log2(4/NR) | (head dim 128 ? GGML_TURBOT_GEOM_D128 : 0):
//   (256, 4) = 0   (256, 2) = 1   (256, 1) = 2   (128, 8) = 4   (128, 4) = 5   (128, 2) = 6        as (head dim, KV heads)
// flags 0 is the Qwen3.8-27B geometry. 3 and 7 (log2 field 3) and every value with a bit above GGML_TURBOT_GEOM_MASK are
// invalid.
#define GGML_TURBOT_RUN_ELEMS        256    // values per run; a run has one old width and one young width
#define GGML_TURBOT_MAX_RUNS         4      // NR is 1, 2 or 4
#define GGML_TURBOT_S_MIN_ANY        2      // smallest S over every geometry (NR = 1, b = 2); S2..S7 are GGML_TYPE_TURBOT_S2..S7
#define GGML_TURBOT_GEOM_LOG2_MASK   0x03u  // flags bits 0-1 = log2(4/NR)
#define GGML_TURBOT_GEOM_D128        0x04u  // head dim 128 (else 256)
#define GGML_TURBOT_GEOM_MASK        0x07u
#define GGML_TURBOT_POOL_DEFAULT     65536  // young pool cells (plan "POOL")
#define GGML_TURBOT_CAP_DEFAULT      16384  // per-sequence young cap in cells (plan "CAP")
#define GGML_TURBOT_POOL_MIN_ROWS    64     // pool tensor rows = max(POOL, this): a POOL 0 plan keeps one never-addressed granule
#define GGML_TURBOT_QUOTA_SLACK_GRANULES 2  // per active sequence, taken off the pool before the young quotas (SPEC 9.6)
#define GGML_TURBOT_NORM_EPS         1e-10f
#define GGML_TURBOT_INV_SQRT_128     0.08838834764831845f

#define GGML_TURBOT_MAGIC            0x54425254u   // "TRBT" little endian
// [TAG_SYNC_TURBOT_OP_OFFSET] was 16. Upstream ggml_flash_attn_ext_set_n_kv_max writes int32 slot 4 (bytes 16..19),
// which held the turbot magic, so the params moved to byte 24. Runtime only (GGUF and state files carry no op params).
#define GGML_TURBOT_OP_PARAMS_OFFSET 24            // bytes 0..19 of FLASH_ATTN_EXT: scale, max_bias, softcap, prec, n_kv_max
#define GGML_TURBOT_OP_PARAMS_VERSION 1
#define GGML_TURBOT_SIDE_K           0
#define GGML_TURBOT_SIDE_V           1
#define GGML_TURBOT_SIDE_BOTH        2             // flash attention

#define GGML_TURBOT_BLOB_MAGIC       0x32544254u   // "TBT2"
#define GGML_TURBOT_BLOB_VERSION     2

#define GGML_TURBOT_FNV_OFFSET       0xcbf29ce484222325ull
#define GGML_TURBOT_FNV_PRIME        0x100000001b3ull

//
// type family: GGML_TYPE_TURBOT_S8 .. GGML_TYPE_TURBOT_S24 (ids 49..65) and, [TAG_TURBOT_ANY_TYPES], GGML_TYPE_TURBOT_S2
// .. GGML_TYPE_TURBOT_S7 (ids 66..71, appended so no id moved) for layers with NR < 4 (ggml.h, owner A). blck 1024,
// type_size 32*S + 16. GGML_TYPE_TURBOT_S8 doubles as the "turbot requested" sentinel in llama_context_params.type_k/type_v.
//

static inline bool ggml_turbot_is_type(enum ggml_type t) {
    return ((int) t >= (int) GGML_TYPE_TURBOT_S8 && (int) t <= (int) GGML_TYPE_TURBOT_S24) ||
           ((int) t >= (int) GGML_TYPE_TURBOT_S2 && (int) t <= (int) GGML_TYPE_TURBOT_S7);
}

// s in [GGML_TURBOT_S_MIN_ANY, GGML_TURBOT_S_MAX]; 8..24 map exactly as before
static inline enum ggml_type ggml_turbot_type_of_s(int s) {
    assert(s >= GGML_TURBOT_S_MIN_ANY && s <= GGML_TURBOT_S_MAX);
    if (s < GGML_TURBOT_S_MIN) {
        return (enum ggml_type) ((int) GGML_TYPE_TURBOT_S2 + (s - GGML_TURBOT_S_MIN_ANY));
    }
    return (enum ggml_type) ((int) GGML_TYPE_TURBOT_S8 + (s - GGML_TURBOT_S_MIN));
}

static inline int ggml_turbot_s_of_type(enum ggml_type t) {
    assert(ggml_turbot_is_type(t));
    if ((int) t >= (int) GGML_TYPE_TURBOT_S2) {
        return GGML_TURBOT_S_MIN_ANY + ((int) t - (int) GGML_TYPE_TURBOT_S2);
    }
    return GGML_TURBOT_S_MIN + ((int) t - (int) GGML_TYPE_TURBOT_S8);
}

static inline size_t ggml_turbot_base_row_bytes(int s) {
    return (size_t) (32*s + 16);
}

//
// [TAG_TURBOT_ANY_GEOM] geometry flags (SPEC section 14). The accessors below take a valid flags value.
//

// flags of a (head dim, KV heads) shape, -1 when turbot has no layout for it
static inline int ggml_turbot_geom_flags(int head_dim, int n_head_kv) {
    unsigned d128;
    if (head_dim == 256) {
        d128 = 0;
    } else if (head_dim == 128) {
        d128 = GGML_TURBOT_GEOM_D128;
    } else {
        return -1;
    }
    if (n_head_kv <= 0 || n_head_kv > 8) {
        return -1;
    }
    switch (head_dim*n_head_kv) {
        case 4*GGML_TURBOT_RUN_ELEMS: return (int) (d128 | 0u);   // NR 4
        case 2*GGML_TURBOT_RUN_ELEMS: return (int) (d128 | 1u);   // NR 2
        case 1*GGML_TURBOT_RUN_ELEMS: return (int) (d128 | 2u);   // NR 1
        default:                      return -1;
    }
}

static inline bool ggml_turbot_geometry_supported(int head_dim, int n_head_kv) {
    return ggml_turbot_geom_flags(head_dim, n_head_kv) >= 0;
}

static inline bool ggml_turbot_geom_valid(unsigned flags) {
    return (flags & ~GGML_TURBOT_GEOM_MASK) == 0 && (flags & GGML_TURBOT_GEOM_LOG2_MASK) != GGML_TURBOT_GEOM_LOG2_MASK;
}

// runs per row: 4, 2 or 1
static inline int ggml_turbot_geom_nr(unsigned flags) {
    return GGML_TURBOT_MAX_RUNS >> (flags & GGML_TURBOT_GEOM_LOG2_MASK);
}

// 128 or 256
static inline int ggml_turbot_geom_head_dim(unsigned flags) {
    return (flags & GGML_TURBOT_GEOM_D128) ? 128 : 256;
}

// KV heads per row: nr*256/head_dim
static inline int ggml_turbot_geom_n_head(unsigned flags) {
    return ggml_turbot_geom_nr(flags)*GGML_TURBOT_RUN_ELEMS/ggml_turbot_geom_head_dim(flags);
}

// values per row: nr*256 (the F32 rows a writer takes, the decoded row of the CPU reference)
static inline int ggml_turbot_geom_row_elems(unsigned flags) {
    return ggml_turbot_geom_nr(flags)*GGML_TURBOT_RUN_ELEMS;
}

// WHT-128 groups per row (the writer's NG): 2*nr
static inline int ggml_turbot_geom_n_groups(unsigned flags) {
    return GGML_TURBOT_GROUPS_PER_HEAD*ggml_turbot_geom_nr(flags);
}

//
// tables (generated, ggml-turbot-tables.h)
//

static const float   ggml_turbot_old_levels[GGML_TURBOT_OLD_TOTAL]   GGML_TURBOT_UNUSED = { GGML_TURBOT_OLD_LEVELS_LIST };
static const float   ggml_turbot_old_thr   [GGML_TURBOT_OLD_TOTAL]   GGML_TURBOT_UNUSED = { GGML_TURBOT_OLD_THR_LIST };
static const float   ggml_turbot_young_lut [GGML_TURBOT_YOUNG_TOTAL] GGML_TURBOT_UNUSED = { GGML_TURBOT_YOUNG_LUT_LIST };
static const float   ggml_turbot_young_thr [GGML_TURBOT_YOUNG_TOTAL] GGML_TURBOT_UNUSED = { GGML_TURBOT_YOUNG_THR_LIST };
static const uint8_t ggml_turbot_fill_code [GGML_TURBOT_FILL_TOTAL]  GGML_TURBOT_UNUSED = { GGML_TURBOT_FILL_LIST };
static const int16_t ggml_turbot_young_off_tab[7*9]                  GGML_TURBOT_UNUSED = { GGML_TURBOT_YOUNG_OFF_LIST };
static const int16_t ggml_turbot_fill_off_tab [7*9]                  GGML_TURBOT_UNUSED = { GGML_TURBOT_FILL_OFF_LIST };

// same signs as ggml-turbo-quant.c turbo_cpu_s1/s2, ggml-cuda/turbo-quant.cuh TURBO_WHT_SIGNS1/2 and llama-kvfq.cpp
static const float ggml_turbot_wht_s1[128] GGML_TURBOT_UNUSED = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1 };
static const float ggml_turbot_wht_s2[128] GGML_TURBOT_UNUSED = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1 };

// run of old codebook b: levels [off, off + 2^b), thresholds [off, off + 2^b - 1)
static inline int ggml_turbot_old_off(int b) {
    return (1 << b) - 4;
}

// run of young codebook (b, y): LUT [off, off + 2^y), thresholds [off, off + 2^y - 1). Closed form of the generated
// GGML_TURBOT_YOUNG_OFF_LIST (tests/test-turbot.cpp compares the two).
static inline int ggml_turbot_young_off(int b, int y) {
    assert(b >= GGML_TURBOT_B_MIN && b <= GGML_TURBOT_B_MAX && y > b && y <= GGML_TURBOT_Y_MAX);
    return (b - 2)*512 + 8 + (1 << y) - (1 << (b + 2));
}

// run of center-fill codes (b, y): [off, off + 2^b)
static inline int ggml_turbot_fill_off(int b, int y) {
    static const int base[7] = { 0, 0, 0, 24, 64, 128, 224 };
    assert(b >= GGML_TURBOT_B_MIN && b <= GGML_TURBOT_B_MAX && y > b && y <= GGML_TURBOT_Y_MAX);
    return base[b] + (y - b - 1)*(1 << b);
}

//
// planes: a run of width w (1..6) is P4 (128 B: element e in nibble e%2 of byte e/2), then P2 (64 B: bits 2*(e%4)
// of byte e/4), then P1 (32 B: bit e%8 of byte e/8), each present only if the width needs it. Code bits: P4 holds
// bits 0..3, the next plane present holds the next bits (P2: 2 bits, P1: 1 bit).
//   w = 1: P1        w = 2: P2        w = 3: P2 P1        w = 4: P4        w = 5: P4 P1        w = 6: P4 P2
//

struct ggml_turbot_planes {
    int16_t p4;   // byte offset inside the run, -1 if absent
    int16_t p2;
    int16_t p1;
};

static inline struct ggml_turbot_planes ggml_turbot_planes_of(int w) {
    struct ggml_turbot_planes p;
    assert(w >= 1 && w <= 6);
    p.p4 = (int16_t) (w >= 4 ? 0 : -1);
    p.p2 = (int16_t) ((w & 2) ? (w >= 4 ? 128 : 0) : -1);
    p.p1 = (int16_t) ((w & 1) ? (w >= 4 ? 128 : 0) + ((w & 2) ? 64 : 0) : -1);
    return p;
}

static inline unsigned ggml_turbot_get_code(const uint8_t * run, int w, int e) {
    const struct ggml_turbot_planes p = ggml_turbot_planes_of(w);
    unsigned v  = 0;
    int      sh = 0;
    if (p.p4 >= 0) { v |= ((unsigned) run[p.p4 + e/2] >> ((e % 2)*4)) & 0xFu;       sh  = 4; }
    if (p.p2 >= 0) { v |= (((unsigned) run[p.p2 + e/4] >> (2*(e % 4))) & 3u) << sh; sh += 2; }
    if (p.p1 >= 0) { v |= (((unsigned) run[p.p1 + e/8] >> (e % 8)) & 1u) << sh; }
    return v;
}

static inline void ggml_turbot_set_code(uint8_t * run, int w, int e, unsigned v) {
    const struct ggml_turbot_planes p = ggml_turbot_planes_of(w);
    int sh = 0;
    if (p.p4 >= 0) {
        const int s = (e % 2)*4;
        run[p.p4 + e/2] = (uint8_t) ((run[p.p4 + e/2] & ~(0xFu << s)) | ((v & 0xFu) << s));
        sh = 4;
    }
    if (p.p2 >= 0) {
        const int s = 2*(e % 4);
        run[p.p2 + e/4] = (uint8_t) ((run[p.p2 + e/4] & ~(3u << s)) | (((v >> sh) & 3u) << s));
        sh += 2;
    }
    if (p.p1 >= 0) {
        const int s = e % 8;
        run[p.p1 + e/8] = (uint8_t) ((run[p.p1 + e/8] & ~(1u << s)) | (((v >> sh) & 1u) << s));
    }
}

//
// layout of one layer-side and of one layer
//

// Run r (r < nr) holds values [256r, 256r+256): at flags 0 run r is KV head r (SPEC 14.2 for the other geometries).
struct ggml_turbot_side {
    uint8_t  b[4];              // old width per run, 2..6; 0 for runs r >= nr
    uint8_t  y[4];              // young width per run, b+1..8; 0 for runs r >= nr
    uint8_t  s;                 // sum(b)
    uint8_t  rsum;              // sum(y - b)
    uint16_t base_off[4];       // run r inside the base row (0 for r >= nr)
    uint16_t young_off[4];      // run r refinement inside this side's part of a pool row (0 for r >= nr)
    uint16_t base_row_bytes;    // 32*s + 16 (== ggml_type_size of ggml_turbot_type_of_s(s))
    uint16_t young_bytes;       // 32*rsum + 16
    uint16_t base_gain_off;     // 32*s: 8 ggml_fp16_t, gain of run r group g at + 2*(2*r + g) (slots 2*nr.. stay 0)
    uint16_t young_gain_off;    // 32*rsum
    uint8_t  nr;                // [TAG_TURBOT_ANY_GEOM] runs in the row: 4 (flags 0), 2 or 1
};

struct ggml_turbot_layer {
    struct ggml_turbot_side k;
    struct ggml_turbot_side v;
    uint32_t pool_row_bytes;    // k.young_bytes + v.young_bytes: one pool row = [K part][V part]
    uint32_t pool_v_off;        // k.young_bytes
    uint8_t  flags;             // [TAG_TURBOT_ANY_GEOM] geometry, GGML_TURBOT_GEOM_*; 0 = 4 KV heads x 256
};

// [TAG_TURBOT_ANY_GEOM] nr = 1, 2 or 4 runs; reads b[0..nr) and y[0..nr) only. Returns false (and leaves *sd
// unspecified) if nr or a width is illegal.
static inline bool ggml_turbot_side_init_nr(struct ggml_turbot_side * sd, const uint8_t * b, const uint8_t * y, int nr) {
    int ob = 0, oy = 0;
    memset(sd, 0, sizeof(*sd));
    if (nr != 1 && nr != 2 && nr != GGML_TURBOT_MAX_RUNS) {
        return false;
    }
    for (int h = 0; h < nr; ++h) {
        if (b[h] < GGML_TURBOT_B_MIN || b[h] > GGML_TURBOT_B_MAX || y[h] <= b[h] || y[h] > GGML_TURBOT_Y_MAX) {
            return false;
        }
        sd->b[h]         = b[h];
        sd->y[h]         = y[h];
        sd->base_off[h]  = (uint16_t) ob;
        sd->young_off[h] = (uint16_t) oy;
        ob += 32*b[h];
        oy += 32*(y[h] - b[h]);
    }
    sd->s              = (uint8_t) (ob/32);
    sd->rsum           = (uint8_t) (oy/32);
    sd->base_gain_off  = (uint16_t) ob;
    sd->young_gain_off = (uint16_t) oy;
    sd->base_row_bytes = (uint16_t) (ob + 16);
    sd->young_bytes    = (uint16_t) (oy + 16);
    sd->nr             = (uint8_t) nr;
    return true;
}

// the 4 x 256 layer-side: nr = 4
static inline bool ggml_turbot_side_init(struct ggml_turbot_side * sd, const uint8_t * b, const uint8_t * y) {
    return ggml_turbot_side_init_nr(sd, b, y, GGML_TURBOT_N_HEAD);
}

// [TAG_TURBOT_ANY_GEOM] both sides with nr = ggml_turbot_geom_nr(flags). Zeroes the whole struct first, so two layers
// built from the same widths and flags compare equal with memcmp.
static inline bool ggml_turbot_layer_init_geom(struct ggml_turbot_layer * l,
        const uint8_t * bk, const uint8_t * bv, const uint8_t * yk, const uint8_t * yv, unsigned flags) {
    memset(l, 0, sizeof(*l));
    if (!ggml_turbot_geom_valid(flags)) {
        return false;
    }
    const int nr = ggml_turbot_geom_nr(flags);
    if (!ggml_turbot_side_init_nr(&l->k, bk, yk, nr) || !ggml_turbot_side_init_nr(&l->v, bv, yv, nr)) {
        return false;
    }
    l->pool_row_bytes = (uint32_t) l->k.young_bytes + l->v.young_bytes;
    l->pool_v_off     = l->k.young_bytes;
    l->flags          = (uint8_t) flags;
    return true;
}

// the 4 x 256 layer: flags = 0
static inline bool ggml_turbot_layer_init(struct ggml_turbot_layer * l,
        const uint8_t * bk, const uint8_t * bv, const uint8_t * yk, const uint8_t * yv) {
    return ggml_turbot_layer_init_geom(l, bk, bv, yk, yv, 0);
}

//
// op params: 24 bytes at op_params byte GGML_TURBOT_OP_PARAMS_OFFSET (24) of GGML_OP_TURBOT_SET_ROWS and of a
// GGML_OP_FLASH_ATTN_EXT that ggml_flash_attn_ext_set_turbot() marked. Bytes 48..63 stay zero; bytes 20..23 are unused.
// Byte 31 (flags) carries the geometry [TAG_TURBOT_ANY_GEOM]; the version stays 1 because flags 0 reads as before.
//

struct ggml_turbot_op_params {
    uint32_t magic;          // GGML_TURBOT_MAGIC
    uint8_t  version;        // GGML_TURBOT_OP_PARAMS_VERSION
    uint8_t  side;           // GGML_TURBOT_SIDE_K / _V (writer), GGML_TURBOT_SIDE_BOTH (flash attention)
    uint8_t  log2_granule;   // GGML_TURBOT_LOG2_GRANULE
    uint8_t  flags;          // [TAG_TURBOT_ANY_GEOM] geometry (GGML_TURBOT_GEOM_*), 0 = 4 KV heads x 256
    uint8_t  bk[4];          // per run; entries r >= nr are 0
    uint8_t  bv[4];
    uint8_t  yk[4];
    uint8_t  yv[4];
};

typedef char ggml_turbot_op_params_size_check[(sizeof(struct ggml_turbot_op_params) == 24 &&
        GGML_TURBOT_OP_PARAMS_OFFSET + 24 <= GGML_MAX_OP_PARAMS) ? 1 : -1];

static inline void ggml_turbot_op_params_make(struct ggml_turbot_op_params * p, const struct ggml_turbot_layer * l, int side) {
    memset(p, 0, sizeof(*p));
    p->magic        = GGML_TURBOT_MAGIC;
    p->version      = GGML_TURBOT_OP_PARAMS_VERSION;
    p->side         = (uint8_t) side;
    p->log2_granule = GGML_TURBOT_LOG2_GRANULE;
    p->flags        = l->flags;
    memcpy(p->bk, l->k.b, 4);
    memcpy(p->bv, l->v.b, 4);
    memcpy(p->yk, l->k.y, 4);
    memcpy(p->yv, l->v.y, 4);
}

static inline void ggml_turbot_op_params_set(struct ggml_tensor * t, const struct ggml_turbot_op_params * p) {
    memcpy((char *) t->op_params + GGML_TURBOT_OP_PARAMS_OFFSET, p, sizeof(*p));
}

// false if the tensor carries no valid turbot params (flags with a reserved bit or an invalid geometry included)
static inline bool ggml_turbot_op_params_get(const struct ggml_tensor * t, struct ggml_turbot_op_params * p) {
    memcpy(p, (const char *) t->op_params + GGML_TURBOT_OP_PARAMS_OFFSET, sizeof(*p));
    return p->magic == GGML_TURBOT_MAGIC && p->version == GGML_TURBOT_OP_PARAMS_VERSION &&
           p->log2_granule == GGML_TURBOT_LOG2_GRANULE && p->side <= GGML_TURBOT_SIDE_BOTH &&
           ggml_turbot_geom_valid(p->flags);
}

// [TAG_TURBOT_ANY_GEOM] also false when a run beyond the geometry's nr carries a width (never at flags 0)
static inline bool ggml_turbot_layer_from_op_params(const struct ggml_turbot_op_params * p, struct ggml_turbot_layer * l) {
    if (!ggml_turbot_layer_init_geom(l, p->bk, p->bv, p->yk, p->yv, p->flags)) {
        return false;
    }
    for (int r = l->k.nr; r < GGML_TURBOT_MAX_RUNS; ++r) {
        if (p->bk[r] != 0 || p->bv[r] != 0 || p->yk[r] != 0 || p->yv[r] != 0) {
            return false;
        }
    }
    return true;
}

//
// granules and pool rows
//

static inline uint32_t ggml_turbot_granule_of(uint32_t cell) {
    return cell >> GGML_TURBOT_LOG2_GRANULE;
}

// pool row of `cell` when its granule owns young slot `slot`
static inline int32_t ggml_turbot_pool_row(int32_t slot, uint32_t cell) {
    return slot*GGML_TURBOT_GRANULE + (int32_t) (cell & (GGML_TURBOT_GRANULE - 1));
}

//
// coder (SPEC section 4). Reference math: CPU op, host tests and the oracle for both CUDA kernels.
//

// x *= S1; butterfly; x *= INV_SQRT_128 * S2   (turbo_cpu_fwht / llama-kvfq.cpp, same operation order)
static inline void ggml_turbot_fwht128(float * x) {
    for (int i = 0; i < 128; ++i) x[i] *= ggml_turbot_wht_s1[i];
    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += 2*h) {
            for (int j = i; j < i + h; ++j) {
                const float a = x[j], c = x[j + h];
                x[j]     = a + c;
                x[j + h] = a - c;
            }
        }
    }
    for (int i = 0; i < 128; ++i) x[i] *= GGML_TURBOT_INV_SQRT_128 * ggml_turbot_wht_s2[i];
}

// inverse of ggml_turbot_fwht128: x *= S2; butterfly; x *= INV_SQRT_128 * S1
static inline void ggml_turbot_iwht128(float * x) {
    for (int i = 0; i < 128; ++i) x[i] *= ggml_turbot_wht_s2[i];
    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += 2*h) {
            for (int j = i; j < i + h; ++j) {
                const float a = x[j], c = x[j + h];
                x[j]     = a + c;
                x[j + h] = a - c;
            }
        }
    }
    for (int i = 0; i < 128; ++i) x[i] *= GGML_TURBOT_INV_SQRT_128 * ggml_turbot_wht_s1[i];
}

// number of thresholds t with x >= t (thresholds ascending): the index rule of every turbot codebook
static inline int ggml_turbot_count_ge(const float * thr, int n, float x) {
    int c = 0;
    for (int i = 0; i < n; ++i) {
        c += (x >= thr[i]);
    }
    return c;
}

// old index j of rotated coordinate u
static inline int ggml_turbot_old_index(int b, float u) {
    return ggml_turbot_count_ge(ggml_turbot_old_thr + ggml_turbot_old_off(b), (1 << b) - 1, u);
}

// refinement s of rotated coordinate u inside old cell j. Equal to young_index & (2^r - 1) where
// young_index = count_ge(all 2^y - 1 young thresholds, u), because the young thresholds are sorted and embed the old
// thresholds bit for bit (asserted by the generator), so young_index >> r == j always.
static inline int ggml_turbot_refine_index(int b, int y, int j, float u) {
    const int r = y - b;
    return ggml_turbot_count_ge(ggml_turbot_young_thr + ggml_turbot_young_off(b, y) + (j << r), (1 << r) - 1, u);
}

// young index of rotated coordinate u: one count over all 2^y - 1 young thresholds (the CUDA writer's lane-uniform
// search, SPEC 8.2). young >> (y - b) == ggml_turbot_old_index(b, u) and young & (2^(y-b) - 1) ==
// ggml_turbot_refine_index(b, y, j, u), by the bit-exact embedding of the old thresholds.
static inline int ggml_turbot_young_index(int b, int y, float u) {
    return ggml_turbot_count_ge(ggml_turbot_young_thr + ggml_turbot_young_off(b, y), (1 << y) - 1, u);
}

static inline float ggml_turbot_old_level(int b, int j) {
    return ggml_turbot_old_levels[ggml_turbot_old_off(b) + j];
}

// int8 register-LUT form of the old level that a CUDA reader MAY use for b <= 5 (SPEC 7.4): level ~= i8 * scale with
// i8 = round(C_b[j] / max|C_b| * 127) and scale = max|C_b| / 127. For b = 4, 5 these are the turbo4_int8_lut /
// turbo5_int8_lut entries and TURBO_INT8_4BIT/5BIT_SCALE_REVERSE. Test oracle; the CPU reference itself reads float.
static inline float ggml_turbot_old_i8_scale(int b) {
    assert(b >= GGML_TURBOT_B_MIN && b <= 5);
    return -ggml_turbot_old_levels[ggml_turbot_old_off(b)] / 127.0f;   // ascending, antisymmetric: max|C_b| = -C_b[0]
}

static inline int8_t ggml_turbot_old_level_i8(int b, int j) {
    return (int8_t) roundf(ggml_turbot_old_level(b, j) / -ggml_turbot_old_levels[ggml_turbot_old_off(b)] * 127.0f);
}

static inline float ggml_turbot_young_level(int b, int y, int j, int s) {
    return ggml_turbot_young_lut[ggml_turbot_young_off(b, y) + ((j << (y - b)) | s)];
}

static inline float ggml_turbot_gain(float norm, float recon_sq) {
    const float recon = sqrtf(recon_sq);
    return recon > GGML_TURBOT_NORM_EPS ? norm / recon : norm;
}

// Quantize one 128-value group. code_r / gain_y may be NULL (old code only). Codes are one per element.
static inline void ggml_turbot_quantize_group(const float * x, int b, int y,
        uint8_t * code_b, uint8_t * code_r, ggml_fp16_t * gain_b, ggml_fp16_t * gain_y) {
    float u[128];
    float norm_sq = 0.0f;
    for (int i = 0; i < 128; ++i) norm_sq += x[i]*x[i];
    const float norm = sqrtf(norm_sq);
    const float inv  = norm > GGML_TURBOT_NORM_EPS ? 1.0f/norm : 0.0f;
    for (int i = 0; i < 128; ++i) u[i] = x[i]*inv;
    ggml_turbot_fwht128(u);

    float rb = 0.0f, ry = 0.0f;
    for (int i = 0; i < 128; ++i) {
        const int   j = ggml_turbot_old_index(b, u[i]);
        const float c = ggml_turbot_old_level(b, j);
        code_b[i] = (uint8_t) j;
        rb += c*c;
        if (code_r) {
            const int   s  = ggml_turbot_refine_index(b, y, j, u[i]);
            const float cy = ggml_turbot_young_level(b, y, j, s);
            code_r[i] = (uint8_t) s;
            ry += cy*cy;
        }
    }
    *gain_b = ggml_fp32_to_fp16(ggml_turbot_gain(norm, rb));
    if (code_r) {
        *gain_y = ggml_fp32_to_fp16(ggml_turbot_gain(norm, ry));
    }
}

static inline float ggml_turbot_read_gain(const uint8_t * p, int idx) {
    ggml_fp16_t g;
    memcpy(&g, p + 2*idx, sizeof(g));
    return ggml_fp16_to_fp32(g);
}

static inline void ggml_turbot_write_gain(uint8_t * p, int idx, ggml_fp16_t g) {
    memcpy(p + 2*idx, &g, sizeof(g));
}

// Encode one layer-side row (256*nr values, run h group g = x[256*h + 128*g ..]; 1024 values and run = KV head at
// flags 0). Overwrites the whole base row and, if young != NULL, the whole side part of the pool row.
static inline void ggml_turbot_encode_side(const float * x, const struct ggml_turbot_side * sd, uint8_t * base, uint8_t * young) {
    memset(base, 0, sd->base_row_bytes);
    if (young) {
        memset(young, 0, sd->young_bytes);
    }
    for (int h = 0; h < sd->nr; ++h) {
        const int b = sd->b[h], y = sd->y[h];
        for (int g = 0; g < GGML_TURBOT_GROUPS_PER_HEAD; ++g) {
            uint8_t     cb[128], cr[128];
            ggml_fp16_t gb, gy;
            ggml_turbot_quantize_group(x + 256*h + 128*g, b, y, cb, young ? cr : NULL, &gb, young ? &gy : NULL);
            for (int e = 0; e < 128; ++e) {
                ggml_turbot_set_code(base + sd->base_off[h], b, 128*g + e, cb[e]);
                if (young) {
                    ggml_turbot_set_code(young + sd->young_off[h], y - b, 128*g + e, cr[e]);
                }
            }
            ggml_turbot_write_gain(base + sd->base_gain_off, 2*h + g, gb);
            if (young) {
                ggml_turbot_write_gain(young + sd->young_gain_off, 2*h + g, gy);
            }
        }
    }
}

// Decode one layer-side row to 256*nr rotated-domain values (1024 at flags 0). young == NULL -> old read.
static inline void ggml_turbot_decode_side(const uint8_t * base, const uint8_t * young, const struct ggml_turbot_side * sd, float * out) {
    for (int h = 0; h < sd->nr; ++h) {
        const int b = sd->b[h], y = sd->y[h], r = y - b;
        for (int g = 0; g < GGML_TURBOT_GROUPS_PER_HEAD; ++g) {
            const float gain = young ? ggml_turbot_read_gain(young + sd->young_gain_off, 2*h + g)
                                     : ggml_turbot_read_gain(base  + sd->base_gain_off,  2*h + g);
            for (int e = 0; e < 128; ++e) {
                const int j = (int) ggml_turbot_get_code(base + sd->base_off[h], b, 128*g + e);
                float c;
                if (young) {
                    const int s = (int) ggml_turbot_get_code(young + sd->young_off[h], r, 128*g + e);
                    c = ggml_turbot_young_level(b, y, j, s);
                } else {
                    c = ggml_turbot_old_level(b, j);
                }
                out[256*h + 128*g + e] = c*gain;
            }
        }
    }
}

// Center fill: give a cell that only has an old code a refinement whose young read approximates its old read.
//   s_i    = ggml_turbot_fill_code[fill_off(b, y) + j_i]
//   gain_y = gain_b * (sqrt(sum C_b[j_i]^2) / sqrt(sum LUT[(j_i << r) | s_i]^2)), gain_b if the young norm <= eps
// Overwrites the whole side part of the pool row. Runs 0..nr-1.
static inline void ggml_turbot_fill_side(const uint8_t * base, const struct ggml_turbot_side * sd, uint8_t * young) {
    memset(young, 0, sd->young_bytes);
    for (int h = 0; h < sd->nr; ++h) {
        const int b = sd->b[h], y = sd->y[h], r = y - b;
        const int fo = ggml_turbot_fill_off(b, y);
        for (int g = 0; g < GGML_TURBOT_GROUPS_PER_HEAD; ++g) {
            float rb = 0.0f, ry = 0.0f;
            for (int e = 0; e < 128; ++e) {
                const int   j  = (int) ggml_turbot_get_code(base + sd->base_off[h], b, 128*g + e);
                const int   s  = ggml_turbot_fill_code[fo + j];
                const float cb = ggml_turbot_old_level(b, j);
                const float cy = ggml_turbot_young_level(b, y, j, s);
                rb += cb*cb;
                ry += cy*cy;
                ggml_turbot_set_code(young + sd->young_off[h], r, 128*g + e, (unsigned) s);
            }
            const float gb     = ggml_turbot_read_gain(base + sd->base_gain_off, 2*h + g);
            const float nry    = sqrtf(ry);
            const float gy     = nry > GGML_TURBOT_NORM_EPS ? gb*(sqrtf(rb)/nry) : gb;
            ggml_turbot_write_gain(young + sd->young_gain_off, 2*h + g, ggml_fp32_to_fp16(gy));
        }
    }
}

//
// plan identity (state blobs refuse a different plan)
//

static inline uint64_t ggml_turbot_fnv1a64(uint64_t h, const void * data, size_t n) {
    const uint8_t * p = (const uint8_t *) data;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= GGML_TURBOT_FNV_PRIME;
    }
    return h;
}

// Fold one attention layer into a plan hash. Call in ascending il order starting from GGML_TURBOT_FNV_OFFSET, then
// fold pool_cells and cap_cells with ggml_turbot_plan_hash_finish.
// [TAG_TURBOT_ANY_GEOM] a layer with flags != 0 also folds { 'G', 'E', 'O', '1', flags }, so two geometries with the
// same widths never share a hash. At flags 0 nothing is added: the Qwen3.8-27B hash 0x56c3503c949a7749 is unchanged.
static inline uint64_t ggml_turbot_plan_hash_layer(uint64_t h, int32_t il, const struct ggml_turbot_layer * l) {
    const uint8_t tag[4] = { 'L', 'A', 'Y', '1' };
    h = ggml_turbot_fnv1a64(h, tag, 4);
    h = ggml_turbot_fnv1a64(h, &il, sizeof(il));
    h = ggml_turbot_fnv1a64(h, l->k.b, 4);
    h = ggml_turbot_fnv1a64(h, l->v.b, 4);
    h = ggml_turbot_fnv1a64(h, l->k.y, 4);
    h = ggml_turbot_fnv1a64(h, l->v.y, 4);
    if (l->flags != 0) {
        const uint8_t geo[5] = { 'G', 'E', 'O', '1', l->flags };
        h = ggml_turbot_fnv1a64(h, geo, 5);
    }
    return h;
}

static inline uint64_t ggml_turbot_plan_hash_finish(uint64_t h, uint32_t pool_cells, uint32_t cap_cells) {
    h = ggml_turbot_fnv1a64(h, &pool_cells, sizeof(pool_cells));
    h = ggml_turbot_fnv1a64(h, &cap_cells,  sizeof(cap_cells));
    return h;
}

#ifdef __cplusplus
}
#endif
