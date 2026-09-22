// [TAG_TURBOT_ANY_GEOM] [TAG_TURBOT_ANY_TYPES] CPU unit tests of turbot on shapes other than 4 KV heads x 256,
// docs/turbot/SPEC.md section 14 (contract C1 in ggml/include/ggml-turbot.h).
//
//   (a) geometry flags of the six supported (head dim, KV heads) shapes, the accessors, the run mapping, -1 elsewhere
//   (b) side / layer init for nr 1, 2, 4: sums, offsets, row and young bytes, zero runs beyond nr, illegal input
//   (c) coder: encode / decode / fill per nr at old widths 2..6 and young 7 / 8 against ggml_turbot_quantize_group;
//       an NR-run row equals the first NR runs of a 4-run row; nr 4 (flags 0) is byte-identical to a copy of the
//       pre-change loops (golden buffers from a fixed seed)
//   (d) plan hash: flags 0 equals a copy of the old fold (default plan 0x56c3503c949a7749), flags != 0 differs
//   (e) type family S2..S24: ids, sizes, block, names
//   (f) op params: make / get / layer round trip carries flags; reserved bits, invalid geometries and widths beyond
//       nr are rejected; flags-0 bytes equal a copy of the old packing
//   (g) CPU flash attention reference over turbot K/V at D 128 NR 4 (GQA 4 and 7), D 256 NR 2 (GQA 8) and the other
//       geometries, against dense attention over the decoded K/V; supports_op follows the op-params geometry
//   (h) CPU writer op (GGML_OP_TURBOT_SET_ROWS) at NR 1, 2 and 4, fill entries included, against the header coder
//
// (g) and (h) call the CPU backend directly (ggml-cpu.h), which links only without GGML_BACKEND_DL: tests/CMakeLists.txt
// defines TURBOT_GEOM_CPU_OPS in that case, and the two parts print SKIPPED otherwise.
//
// No GPU, no model. Exit code 0 iff every check passes.
//
//   test-turbot-geom

#include "ggml.h"
#include "ggml-turbot.h"

#ifdef TURBOT_GEOM_CPU_OPS
#    include "ggml-backend.h"
#    include "ggml-cpu.h"
#endif

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int g_fail   = 0;
static int g_checks = 0;

#define TCHECK(cond, ...)                                                   \
    do {                                                                    \
        ++g_checks;                                                         \
        if (!(cond)) {                                                      \
            ++g_fail;                                                       \
            if (g_fail <= 300) {                                            \
                fprintf(stderr, "FAIL %s:%d: ", __func__, __LINE__);        \
                fprintf(stderr, __VA_ARGS__);                               \
                fprintf(stderr, "\n");                                      \
            }                                                               \
        }                                                                   \
    } while (0)

static bool same_bits(float a, float b) {
    return std::memcmp(&a, &b, sizeof(float)) == 0;
}

// the six supported shapes, SPEC 14.1
struct geom_shape {
    int      hd;
    int      nh;
    unsigned flags;
    int      nr;
};

static const geom_shape SHAPES[6] = {
    { 256, 4, 0, 4 }, { 256, 2, 1, 2 }, { 256, 1, 2, 1 },
    { 128, 8, 4, 4 }, { 128, 4, 5, 2 }, { 128, 2, 6, 1 },
};

// widths used by several parts: every run r has y[r] > b[r]; a geometry with nr runs uses the first nr entries
static const uint8_t MIX_BK[4] = { 2, 5, 3, 6 };
static const uint8_t MIX_BV[4] = { 4, 2, 6, 3 };
static const uint8_t MIX_YK[4] = { 7, 8, 7, 7 };
static const uint8_t MIX_YV[4] = { 8, 7, 7, 5 };

// assorted rows: gaussian, uniform, outlier channels, tiny and huge scales, zero groups, one-hot groups
static void make_row(std::mt19937 & gen, int kind, float * x, int n) {
    std::normal_distribution<float>       nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
    const float scales[6] = { 1.0f, 1e-3f, 30.0f, 1e3f, 0.05f, 4.0f };
    const float sc = scales[kind % 6];
    for (int i = 0; i < n; ++i) {
        x[i] = sc * ((kind % 3 == 0) ? nd(gen) : ud(gen));
    }
    if (kind % 4 == 1) {
        for (int i = 0; i < n; i += 97) {
            x[i] *= 40.0f;
        }
    }
    if (kind % 5 == 2 && n >= 256) {
        std::memset(x + 128, 0, 128 * sizeof(float));   // run 0 group 1 all zero
    }
    if (kind % 7 == 3) {
        std::memset(x + n - 128, 0, 128 * sizeof(float));
        x[n - 100] = sc;                                // one-hot last group
    }
}

// ---------------------------------------------------------------------------------------------------------------
// copies of the pre-change code (ggml-turbot.h at 0fc83cc8d): the coder loops over GGML_TURBOT_N_HEAD = 4 heads, the
// hash has no geometry fold, the op params have no flags. Kept verbatim: flags 0 / nr 4 must reproduce them exactly.
// ---------------------------------------------------------------------------------------------------------------

static void old_encode_side(const float * x, const ggml_turbot_side * sd, uint8_t * base, uint8_t * young) {
    std::memset(base, 0, sd->base_row_bytes);
    if (young) {
        std::memset(young, 0, sd->young_bytes);
    }
    for (int h = 0; h < 4; ++h) {
        const int b = sd->b[h], y = sd->y[h];
        for (int g = 0; g < 2; ++g) {
            uint8_t     cb[128], cr[128];
            ggml_fp16_t gb, gy;
            ggml_turbot_quantize_group(x + 256*h + 128*g, b, y, cb, young ? cr : nullptr, &gb, young ? &gy : nullptr);
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

static void old_decode_side(const uint8_t * base, const uint8_t * young, const ggml_turbot_side * sd, float * out) {
    for (int h = 0; h < 4; ++h) {
        const int b = sd->b[h], y = sd->y[h], r = y - b;
        for (int g = 0; g < 2; ++g) {
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

static void old_fill_side(const uint8_t * base, const ggml_turbot_side * sd, uint8_t * young) {
    std::memset(young, 0, sd->young_bytes);
    for (int h = 0; h < 4; ++h) {
        const int b = sd->b[h], y = sd->y[h], r = y - b;
        const int fo = ggml_turbot_fill_off(b, y);
        for (int g = 0; g < 2; ++g) {
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

static uint64_t old_plan_hash_layer(uint64_t h, int32_t il, const ggml_turbot_layer * l) {
    const uint8_t tag[4] = { 'L', 'A', 'Y', '1' };
    h = ggml_turbot_fnv1a64(h, tag, 4);
    h = ggml_turbot_fnv1a64(h, &il, sizeof(il));
    h = ggml_turbot_fnv1a64(h, l->k.b, 4);
    h = ggml_turbot_fnv1a64(h, l->v.b, 4);
    h = ggml_turbot_fnv1a64(h, l->k.y, 4);
    h = ggml_turbot_fnv1a64(h, l->v.y, 4);
    return h;
}

static void old_op_params_make(ggml_turbot_op_params * p, const ggml_turbot_layer * l, int side) {
    std::memset(p, 0, sizeof(*p));
    p->magic        = GGML_TURBOT_MAGIC;
    p->version      = GGML_TURBOT_OP_PARAMS_VERSION;
    p->side         = (uint8_t) side;
    p->log2_granule = GGML_TURBOT_LOG2_GRANULE;
    std::memcpy(p->bk, l->k.b, 4);
    std::memcpy(p->bv, l->v.b, 4);
    std::memcpy(p->yk, l->k.y, 4);
    std::memcpy(p->yv, l->v.y, 4);
}

// docs/turbot/plans/turbot-default.plan (SPEC appendix A), as in tests/test-turbot.cpp
static const int     DEFAULT_IL[16] = { 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63 };
static const uint8_t DEFAULT_BK[16][4] = {
    {2,2,2,4}, {2,5,3,2}, {4,5,4,2}, {4,4,3,4}, {4,4,3,4}, {5,4,4,6}, {5,5,6,5}, {6,5,5,5},
    {4,5,5,5}, {5,5,4,6}, {4,4,6,5}, {4,5,5,5}, {5,5,5,6}, {5,5,5,5}, {4,5,5,5}, {4,4,4,4} };
static const uint8_t DEFAULT_BV[16][4] = {
    {2,2,2,4}, {2,5,2,2}, {3,4,5,2}, {5,3,3,4}, {3,4,2,5}, {5,4,4,5}, {5,5,6,5}, {5,4,5,4},
    {4,4,4,5}, {4,4,4,5}, {4,4,5,4}, {4,4,5,5}, {5,4,4,5}, {5,5,5,5}, {5,6,5,5}, {5,4,5,5} };
static const uint64_t DEFAULT_PLAN_HASH = 0x56c3503c949a7749ull;

// ---------------------------------------------------------------------------------------------------------------
// (a) geometry flags
// ---------------------------------------------------------------------------------------------------------------

static void test_geom_flags() {
    printf("[a] geometry flags\n");

    TCHECK(GGML_TURBOT_RUN_ELEMS == 256 && GGML_TURBOT_MAX_RUNS == 4 && GGML_TURBOT_S_MIN_ANY == 2, "contract constants");
    TCHECK(GGML_TURBOT_GEOM_LOG2_MASK == 0x03u && GGML_TURBOT_GEOM_D128 == 0x04u && GGML_TURBOT_GEOM_MASK == 0x07u, "flag masks");
    TCHECK(GGML_TURBOT_HEAD_DIM == 256 && GGML_TURBOT_N_HEAD == 4 && GGML_TURBOT_ROW_ELEMS == 1024 &&
           GGML_TURBOT_S_MIN == 8 && GGML_TURBOT_S_MAX == 24 && GGML_TURBOT_N_GAINS == 8, "a flags-0 constant moved");

    for (const auto & sh : SHAPES) {
        const int f = ggml_turbot_geom_flags(sh.hd, sh.nh);
        TCHECK(f == (int) sh.flags, "geom_flags(%d, %d) = %d, expected %u", sh.hd, sh.nh, f, sh.flags);
        TCHECK(ggml_turbot_geometry_supported(sh.hd, sh.nh), "(%d, %d) not supported", sh.hd, sh.nh);
        TCHECK(ggml_turbot_geom_valid(sh.flags), "flags %u not valid", sh.flags);
        TCHECK(ggml_turbot_geom_nr(sh.flags) == sh.nr, "flags %u: nr %d, expected %d", sh.flags, ggml_turbot_geom_nr(sh.flags), sh.nr);
        TCHECK(ggml_turbot_geom_head_dim(sh.flags) == sh.hd, "flags %u: head dim %d", sh.flags, ggml_turbot_geom_head_dim(sh.flags));
        TCHECK(ggml_turbot_geom_n_head(sh.flags) == sh.nh, "flags %u: n_head %d", sh.flags, ggml_turbot_geom_n_head(sh.flags));
        TCHECK(ggml_turbot_geom_row_elems(sh.flags) == sh.nr * 256 && ggml_turbot_geom_row_elems(sh.flags) == sh.hd * sh.nh,
               "flags %u: row elems %d", sh.flags, ggml_turbot_geom_row_elems(sh.flags));
        TCHECK(ggml_turbot_geom_n_groups(sh.flags) == 2 * sh.nr, "flags %u: n_groups %d", sh.flags, ggml_turbot_geom_n_groups(sh.flags));
        TCHECK((sh.flags & GGML_TURBOT_GEOM_LOG2_MASK) == (sh.nr == 4 ? 0u : sh.nr == 2 ? 1u : 2u) &&
               ((sh.flags & GGML_TURBOT_GEOM_D128) != 0) == (sh.hd == 128), "flags %u: bit layout", sh.flags);

        // run mapping (SPEC 14.2): head z covers values [D*z, D*z + D). D 256: run z. D 128: run z>>1 at 128*(z&1).
        for (int z = 0; z < sh.nh; ++z) {
            const int run  = sh.hd == 256 ? z : z >> 1;
            const int base = sh.hd == 256 ? 0 : 128 * (z & 1);
            TCHECK(run < sh.nr && 256 * run + base == sh.hd * z, "flags %u head %d: run %d base %d", sh.flags, z, run, base);
        }
    }
    TCHECK(ggml_turbot_geom_flags(256, 4) == 0, "the Qwen3.8-27B shape (256, 4) is not flags 0");

    const int bad[][2] = { { 64, 8 }, { 128, 16 }, { 256, 8 }, { 512, 1 }, { 128, 1 }, { 256, 3 }, { 128, 6 }, { 96, 8 },
                           { 256, 0 }, { 128, -2 }, { 0, 4 }, { 80, 8 }, { 192, 4 }, { 64, 16 }, { 512, 2 } };
    for (const auto & s : bad) {
        TCHECK(ggml_turbot_geom_flags(s[0], s[1]) == -1 && !ggml_turbot_geometry_supported(s[0], s[1]),
               "(%d, %d) accepted as flags %d", s[0], s[1], ggml_turbot_geom_flags(s[0], s[1]));
    }

    int n_valid = 0;
    for (unsigned f = 0; f < 256; ++f) {
        const bool expect = f == 0 || f == 1 || f == 2 || f == 4 || f == 5 || f == 6;
        TCHECK(ggml_turbot_geom_valid(f) == expect, "geom_valid(%u) = %d", f, (int) ggml_turbot_geom_valid(f));
        n_valid += ggml_turbot_geom_valid(f) ? 1 : 0;
    }
    TCHECK(n_valid == 6, "%d valid flag values, expected 6", n_valid);
    TCHECK(!ggml_turbot_geom_valid(0x100u) && !ggml_turbot_geom_valid(0x104u) && !ggml_turbot_geom_valid(0xFFFFFFFFu), "wide flag values accepted");
}

// ---------------------------------------------------------------------------------------------------------------
// (b) side and layer init
// ---------------------------------------------------------------------------------------------------------------

static void test_init() {
    printf("[b] side and layer init for nr 1, 2, 4\n");

    struct side_case {
        int     nr;
        uint8_t b[4];
        uint8_t y[4];
    };
    const side_case cases[] = {
        { 1, { 3, 0, 0, 0 }, { 7, 0, 0, 0 } },
        { 1, { 6, 0, 0, 0 }, { 8, 0, 0, 0 } },
        { 1, { 2, 0, 0, 0 }, { 3, 0, 0, 0 } },
        { 2, { 2, 6, 0, 0 }, { 8, 7, 0, 0 } },
        { 2, { 5, 4, 0, 0 }, { 7, 7, 0, 0 } },
        { 2, { 6, 6, 0, 0 }, { 8, 8, 0, 0 } },
        { 4, { 2, 6, 3, 5 }, { 8, 7, 7, 7 } },
        { 4, { 4, 4, 4, 4 }, { 7, 7, 7, 7 } },
        { 4, { 2, 2, 2, 2 }, { 3, 3, 3, 3 } },
    };
    for (const auto & c : cases) {
        ggml_turbot_side sd;
        std::memset(&sd, 0xCD, sizeof(sd));
        TCHECK(ggml_turbot_side_init_nr(&sd, c.b, c.y, c.nr), "side_init_nr(nr %d) of legal widths failed", c.nr);
        int ob = 0, oy = 0, s = 0, rsum = 0;
        for (int r = 0; r < c.nr; ++r) {
            TCHECK(sd.b[r] == c.b[r] && sd.y[r] == c.y[r], "nr %d run %d widths", c.nr, r);
            TCHECK(sd.base_off[r] == ob && sd.young_off[r] == oy, "nr %d run %d offsets %d %d, expected %d %d", c.nr, r, sd.base_off[r], sd.young_off[r], ob, oy);
            ob   += 32 * c.b[r];
            oy   += 32 * (c.y[r] - c.b[r]);
            s    += c.b[r];
            rsum += c.y[r] - c.b[r];
        }
        for (int r = c.nr; r < 4; ++r) {
            TCHECK(sd.b[r] == 0 && sd.y[r] == 0 && sd.base_off[r] == 0 && sd.young_off[r] == 0, "nr %d run %d beyond nr is not zero", c.nr, r);
        }
        TCHECK(sd.nr == c.nr && sd.s == s && sd.rsum == rsum, "nr %d: nr %d s %d rsum %d, expected s %d rsum %d", c.nr, sd.nr, sd.s, sd.rsum, s, rsum);
        TCHECK(sd.base_row_bytes == 32 * s + 16 && sd.young_bytes == 32 * rsum + 16, "nr %d: row bytes %d young bytes %d", c.nr, sd.base_row_bytes, sd.young_bytes);
        TCHECK(sd.base_gain_off == 32 * s && sd.young_gain_off == 32 * rsum, "nr %d: gain offsets %d %d", c.nr, sd.base_gain_off, sd.young_gain_off);
        TCHECK(ggml_turbot_base_row_bytes(sd.s) == sd.base_row_bytes && sd.base_row_bytes % 16 == 0 && sd.young_bytes % 16 == 0, "nr %d: row sizes", c.nr);
        TCHECK(sd.s >= 2 * c.nr && sd.s <= 6 * c.nr && sd.s >= GGML_TURBOT_S_MIN_ANY && sd.s <= GGML_TURBOT_S_MAX, "nr %d: S %d outside its range", c.nr, sd.s);
        if (c.nr == 4) {
            ggml_turbot_side s4;
            std::memset(&s4, 0x3C, sizeof(s4));
            TCHECK(ggml_turbot_side_init(&s4, c.b, c.y) && std::memcmp(&s4, &sd, sizeof(sd)) == 0, "side_init differs from side_init_nr(4)");
        }
    }

    // worked numbers of SPEC 14.3
    {
        const uint8_t b[4] = { 3, 0, 0, 0 }, y[4] = { 7, 0, 0, 0 };
        ggml_turbot_side sd;
        TCHECK(ggml_turbot_side_init_nr(&sd, b, y, 1), "nr 1 example");
        TCHECK(sd.s == 3 && sd.rsum == 4 && sd.base_row_bytes == 112 && sd.young_bytes == 144 && sd.base_gain_off == 96 && sd.young_gain_off == 128,
               "nr 1 example layout %d %d %d %d %d %d", sd.s, sd.rsum, sd.base_row_bytes, sd.young_bytes, sd.base_gain_off, sd.young_gain_off);
    }
    {
        const uint8_t b[4] = { 2, 6, 0, 0 }, y[4] = { 8, 7, 0, 0 };
        ggml_turbot_side sd;
        TCHECK(ggml_turbot_side_init_nr(&sd, b, y, 2), "nr 2 example");
        TCHECK(sd.base_off[1] == 64 && sd.young_off[1] == 192 && sd.base_row_bytes == 272 && sd.young_bytes == 240 &&
               sd.base_gain_off == 256 && sd.young_gain_off == 224, "nr 2 example layout");
    }

    // entries beyond nr are not read
    {
        const uint8_t b[4] = { 4, 99, 1, 0 }, y[4] = { 7, 0, 200, 9 };
        ggml_turbot_side sd;
        TCHECK(ggml_turbot_side_init_nr(&sd, b, y, 1) && sd.b[1] == 0 && sd.y[2] == 0 && sd.s == 4, "nr 1 read a width beyond run 0");
    }

    // illegal nr
    {
        const uint8_t b[4] = { 4, 4, 4, 4 }, y[4] = { 7, 7, 7, 7 };
        ggml_turbot_side sd;
        for (int nr : { 0, 3, 5, 8, -1 }) {
            TCHECK(!ggml_turbot_side_init_nr(&sd, b, y, nr), "nr %d accepted", nr);
        }
    }

    // illegal widths in the last run of each nr
    for (int nr : { 1, 2, 4 }) {
        const uint8_t bad[][2] = { { 1, 3 }, { 7, 8 }, { 4, 4 }, { 4, 9 }, { 5, 3 }, { 0, 0 } };
        for (const auto & w : bad) {
            uint8_t b[4] = { 4, 4, 4, 4 }, y[4] = { 7, 7, 7, 7 };
            b[nr - 1] = w[0];
            y[nr - 1] = w[1];
            ggml_turbot_side sd;
            TCHECK(!ggml_turbot_side_init_nr(&sd, b, y, nr), "nr %d: b %d y %d in run %d accepted", nr, w[0], w[1], nr - 1);
        }
    }

    // layer init per geometry
    for (const auto & sh : SHAPES) {
        ggml_turbot_layer l, l2;
        std::memset(&l, 0x5A, sizeof(l));
        std::memset(&l2, 0xA5, sizeof(l2));
        TCHECK(ggml_turbot_layer_init_geom(&l,  MIX_BK, MIX_BV, MIX_YK, MIX_YV, sh.flags), "layer_init_geom flags %u failed", sh.flags);
        TCHECK(ggml_turbot_layer_init_geom(&l2, MIX_BK, MIX_BV, MIX_YK, MIX_YV, sh.flags), "layer_init_geom flags %u failed", sh.flags);
        TCHECK(std::memcmp(&l, &l2, sizeof(l)) == 0, "flags %u: two inits differ (padding not zeroed?)", sh.flags);
        TCHECK(l.flags == sh.flags && l.k.nr == sh.nr && l.v.nr == sh.nr, "flags %u: stored flags %u nr %d %d", sh.flags, l.flags, l.k.nr, l.v.nr);
        TCHECK(l.pool_row_bytes == (uint32_t) l.k.young_bytes + l.v.young_bytes && l.pool_v_off == l.k.young_bytes && l.pool_row_bytes % 16 == 0,
               "flags %u: pool row layout", sh.flags);
        ggml_turbot_side sk, sv;
        TCHECK(ggml_turbot_side_init_nr(&sk, MIX_BK, MIX_YK, sh.nr) && std::memcmp(&sk, &l.k, sizeof(sk)) == 0, "flags %u: K side", sh.flags);
        TCHECK(ggml_turbot_side_init_nr(&sv, MIX_BV, MIX_YV, sh.nr) && std::memcmp(&sv, &l.v, sizeof(sv)) == 0, "flags %u: V side", sh.flags);
    }
    {
        ggml_turbot_layer a, b;
        std::memset(&a, 0x11, sizeof(a));
        std::memset(&b, 0x22, sizeof(b));
        TCHECK(ggml_turbot_layer_init(&a, MIX_BK, MIX_BV, MIX_YK, MIX_YV) && ggml_turbot_layer_init_geom(&b, MIX_BK, MIX_BV, MIX_YK, MIX_YV, 0) &&
               std::memcmp(&a, &b, sizeof(a)) == 0 && a.flags == 0 && a.k.nr == 4, "layer_init differs from layer_init_geom(flags 0)");
        for (unsigned f : { 3u, 7u, 8u, 0x10u, 0x80u, 0xFFu, 0x104u }) {
            TCHECK(!ggml_turbot_layer_init_geom(&a, MIX_BK, MIX_BV, MIX_YK, MIX_YV, f), "invalid flags %u accepted", f);
        }
        const uint8_t bad_v[4] = { 1, 4, 4, 4 };
        for (const auto & sh : SHAPES) {
            TCHECK(!ggml_turbot_layer_init_geom(&a, MIX_BK, bad_v, MIX_YK, MIX_YV, sh.flags), "flags %u: illegal V width accepted", sh.flags);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// (c) coder
// ---------------------------------------------------------------------------------------------------------------

static void test_coder() {
    printf("[c] coder: encode / decode / fill per nr against quantize_group, prefix of the 4-run row, flags-0 golden\n");
    std::mt19937 gen(20260922);

    // width sets: uniform b 2..6 with y 7 and 8, plus mixed sets
    std::vector<std::pair<std::vector<uint8_t>, std::vector<uint8_t>>> sets;
    for (int b = GGML_TURBOT_B_MIN; b <= GGML_TURBOT_B_MAX; ++b) {
        for (int y : { 7, 8 }) {
            sets.push_back({ std::vector<uint8_t>(4, (uint8_t) b), std::vector<uint8_t>(4, (uint8_t) y) });
        }
    }
    sets.push_back({ { 2, 6, 3, 5 }, { 8, 7, 7, 7 } });
    sets.push_back({ { 6, 2, 5, 4 }, { 7, 8, 8, 7 } });
    sets.push_back({ { 4, 5, 2, 6 }, { 5, 7, 3, 8 } });

    size_t n_groups = 0, bad_codes = 0, bad_gains = 0, bad_dec = 0, bad_fill = 0, bad_prefix = 0, bad_guard = 0;
    const int GUARD = 32;

    for (int nr : { 1, 2, 4 }) {
        const int n = nr * 256;
        for (const auto & st : sets) {
            ggml_turbot_side sd;
            TCHECK(ggml_turbot_side_init_nr(&sd, st.first.data(), st.second.data(), nr), "side_init_nr(nr %d) failed", nr);

            // the same widths in the first nr runs of a 4-run side, width 4 / 7 in the others
            uint8_t b4[4] = { 4, 4, 4, 4 }, y4[4] = { 7, 7, 7, 7 };
            for (int r = 0; r < nr; ++r) {
                b4[r] = st.first[r];
                y4[r] = st.second[r];
            }
            ggml_turbot_side sd4;
            TCHECK(ggml_turbot_side_init(&sd4, b4, y4), "4-run side failed");

            std::vector<float>   x(1024), dec_o(1024 + 16), dec_y(1024 + 16), dec4(1024);
            std::vector<uint8_t> base(sd.base_row_bytes + GUARD), young(sd.young_bytes + GUARD), fill(sd.young_bytes + GUARD);
            std::vector<uint8_t> base4(sd4.base_row_bytes), young4(sd4.young_bytes), fill4(sd4.young_bytes);

            for (int t = 0; t < 12; ++t) {
                make_row(gen, t, x.data(), 1024);   // the first n values are the nr-run row, the rest pad the 4-run row
                std::fill(base.begin(), base.end(), 0xA5);
                std::fill(young.begin(), young.end(), 0xA5);
                std::fill(fill.begin(), fill.end(), 0x5A);
                ggml_turbot_encode_side(x.data(), &sd, base.data(), young.data());
                for (int i = 0; i < GUARD; ++i) {
                    bad_guard += base[sd.base_row_bytes + i] != 0xA5 || young[sd.young_bytes + i] != 0xA5;
                }

                const float sentinel = -12345.0f;
                std::fill(dec_o.begin(), dec_o.end(), sentinel);
                std::fill(dec_y.begin(), dec_y.end(), sentinel);
                ggml_turbot_decode_side(base.data(), nullptr, &sd, dec_o.data());
                ggml_turbot_decode_side(base.data(), young.data(), &sd, dec_y.data());
                for (int i = n; i < (int) dec_o.size(); ++i) {
                    bad_guard += dec_o[i] != sentinel || dec_y[i] != sentinel;
                }

                ggml_turbot_fill_side(base.data(), &sd, fill.data());
                for (int i = 0; i < GUARD; ++i) {
                    bad_guard += fill[sd.young_bytes + i] != 0x5A;
                }

                for (int r = 0; r < nr; ++r) {
                    const int b = sd.b[r], y = sd.y[r], rw = y - b;
                    const int fo = ggml_turbot_fill_off(b, y);
                    for (int g = 0; g < 2; ++g) {
                        uint8_t     cb[128], cr[128];
                        ggml_fp16_t gb, gy;
                        ggml_turbot_quantize_group(x.data() + 256 * r + 128 * g, b, y, cb, cr, &gb, &gy);
                        const float fgb = ggml_fp16_to_fp32(gb), fgy = ggml_fp16_to_fp32(gy);
                        float rb = 0.0f, ry = 0.0f;
                        for (int e = 0; e < 128; ++e) {
                            bad_codes += ggml_turbot_get_code(base.data() + sd.base_off[r], b, 128 * g + e) != cb[e];
                            bad_codes += ggml_turbot_get_code(young.data() + sd.young_off[r], rw, 128 * g + e) != cr[e];
                            bad_dec   += !same_bits(dec_o[256 * r + 128 * g + e], ggml_turbot_old_level(b, cb[e]) * fgb);
                            bad_dec   += !same_bits(dec_y[256 * r + 128 * g + e], ggml_turbot_young_level(b, y, cb[e], cr[e]) * fgy);
                            // center fill of the same old codes
                            const int   s  = ggml_turbot_fill_code[fo + cb[e]];
                            const float c0 = ggml_turbot_old_level(b, cb[e]);
                            const float c1 = ggml_turbot_young_level(b, y, cb[e], s);
                            rb += c0 * c0;
                            ry += c1 * c1;
                            bad_fill += (int) ggml_turbot_get_code(fill.data() + sd.young_off[r], rw, 128 * g + e) != s;
                        }
                        uint16_t raw_b, raw_y, want_b, want_y;
                        std::memcpy(&raw_b, base.data()  + sd.base_gain_off  + 2 * (2 * r + g), 2);
                        std::memcpy(&raw_y, young.data() + sd.young_gain_off + 2 * (2 * r + g), 2);
                        std::memcpy(&want_b, &gb, 2);
                        std::memcpy(&want_y, &gy, 2);
                        bad_gains += raw_b != want_b || raw_y != want_y;
                        const float nry = sqrtf(ry);
                        const float gfy = nry > GGML_TURBOT_NORM_EPS ? fgb * (sqrtf(rb) / nry) : fgb;
                        const ggml_fp16_t hfy = ggml_fp32_to_fp16(gfy);
                        // the gain sums here and in the header may be contracted differently by the compiler (FMA),
                        // so allow one f16 step (gains are >= 0, so adjacent bit patterns are adjacent values); the
                        // flags-0 golden below is exact
                        uint16_t raw_f, want_f;
                        std::memcpy(&raw_f, fill.data() + sd.young_gain_off + 2 * (2 * r + g), 2);
                        std::memcpy(&want_f, &hfy, 2);
                        bad_fill += std::abs((int) raw_f - (int) want_f) > 1;
                        ++n_groups;
                    }
                }
                // unused gain slots stay zero
                for (int i = 2 * nr; i < GGML_TURBOT_N_GAINS; ++i) {
                    uint16_t gb0, gy0, gf0;
                    std::memcpy(&gb0, base.data()  + sd.base_gain_off  + 2 * i, 2);
                    std::memcpy(&gy0, young.data() + sd.young_gain_off + 2 * i, 2);
                    std::memcpy(&gf0, fill.data()  + sd.young_gain_off + 2 * i, 2);
                    bad_gains += gb0 != 0 || gy0 != 0 || gf0 != 0;
                }

                // an nr-run row is the first nr runs of the 4-run row: same run bytes, gains, decoded values and fill
                ggml_turbot_encode_side(x.data(), &sd4, base4.data(), young4.data());
                ggml_turbot_fill_side(base4.data(), &sd4, fill4.data());
                for (int r = 0; r < nr; ++r) {
                    bad_prefix += sd4.base_off[r] != sd.base_off[r] || sd4.young_off[r] != sd.young_off[r];
                    bad_prefix += std::memcmp(base.data() + sd.base_off[r], base4.data() + sd4.base_off[r], 32 * sd.b[r]) != 0;
                    bad_prefix += std::memcmp(young.data() + sd.young_off[r], young4.data() + sd4.young_off[r], 32 * (sd.y[r] - sd.b[r])) != 0;
                    bad_prefix += std::memcmp(fill.data() + sd.young_off[r], fill4.data() + sd4.young_off[r], 32 * (sd.y[r] - sd.b[r])) != 0;
                    for (int g = 0; g < 2; ++g) {
                        bad_prefix += std::memcmp(base.data()  + sd.base_gain_off  + 2 * (2 * r + g), base4.data()  + sd4.base_gain_off  + 2 * (2 * r + g), 2) != 0;
                        bad_prefix += std::memcmp(young.data() + sd.young_gain_off + 2 * (2 * r + g), young4.data() + sd4.young_gain_off + 2 * (2 * r + g), 2) != 0;
                        bad_prefix += std::memcmp(fill.data()  + sd.young_gain_off + 2 * (2 * r + g), fill4.data()  + sd4.young_gain_off + 2 * (2 * r + g), 2) != 0;
                    }
                }
                ggml_turbot_decode_side(base4.data(), young4.data(), &sd4, dec4.data());
                for (int i = 0; i < n; ++i) {
                    bad_prefix += !same_bits(dec4[i], dec_y[i]);
                }
            }
        }
    }
    TCHECK(bad_codes == 0, "%zu codes differ from quantize_group", bad_codes);
    TCHECK(bad_gains == 0, "%zu gains differ from quantize_group or unused slots are not zero", bad_gains);
    TCHECK(bad_dec == 0, "%zu decoded values differ from level * gain", bad_dec);
    TCHECK(bad_fill == 0, "%zu fill codes or gains differ from the fill rule", bad_fill);
    TCHECK(bad_prefix == 0, "%zu differences between an nr-run row and the first nr runs of a 4-run row", bad_prefix);
    TCHECK(bad_guard == 0, "%zu bytes or values written beyond the row", bad_guard);
    printf("    %zu groups over nr 1, 2, 4, %zu width sets, y 7 and 8\n", n_groups, sets.size());

    // golden: nr 4 (flags 0) is the pre-change coder, byte for byte and bit for bit
    {
        std::mt19937 g2(1234);
        size_t n_rows = 0, bad = 0;
        std::vector<float> x(1024), a(1024), b(1024);
        std::vector<std::pair<std::vector<uint8_t>, std::vector<uint8_t>>> gsets = sets;
        for (int i = 0; i < 16; ++i) {
            gsets.push_back({ std::vector<uint8_t>(DEFAULT_BK[i], DEFAULT_BK[i] + 4), std::vector<uint8_t>(4, 7) });
            gsets.push_back({ std::vector<uint8_t>(DEFAULT_BV[i], DEFAULT_BV[i] + 4), std::vector<uint8_t>(4, 7) });
        }
        for (const auto & st : gsets) {
            ggml_turbot_side sd;
            TCHECK(ggml_turbot_side_init(&sd, st.first.data(), st.second.data()), "golden side init");
            std::vector<uint8_t> bn(sd.base_row_bytes), yn(sd.young_bytes), bo(sd.base_row_bytes), yo(sd.young_bytes);
            std::vector<uint8_t> fn(sd.young_bytes), fo(sd.young_bytes), bno(sd.base_row_bytes), boo(sd.base_row_bytes);
            for (int t = 0; t < 8; ++t) {
                make_row(g2, t + 3, x.data(), 1024);
                std::fill(bn.begin(), bn.end(), 0x77);
                std::fill(yn.begin(), yn.end(), 0x77);
                old_encode_side(x.data(), &sd, bo.data(), yo.data());
                ggml_turbot_encode_side(x.data(), &sd, bn.data(), yn.data());
                bad += bn != bo || yn != yo;
                old_encode_side(x.data(), &sd, boo.data(), nullptr);
                ggml_turbot_encode_side(x.data(), &sd, bno.data(), nullptr);
                bad += bno != boo;
                old_decode_side(bo.data(), nullptr, &sd, a.data());
                ggml_turbot_decode_side(bn.data(), nullptr, &sd, b.data());
                bad += std::memcmp(a.data(), b.data(), 1024 * sizeof(float)) != 0;
                old_decode_side(bo.data(), yo.data(), &sd, a.data());
                ggml_turbot_decode_side(bn.data(), yn.data(), &sd, b.data());
                bad += std::memcmp(a.data(), b.data(), 1024 * sizeof(float)) != 0;
                std::fill(fn.begin(), fn.end(), 0x13);
                old_fill_side(bo.data(), &sd, fo.data());
                ggml_turbot_fill_side(bn.data(), &sd, fn.data());
                bad += fn != fo;
                ++n_rows;
            }
        }
        TCHECK(bad == 0, "flags 0: %zu differences from the pre-change coder", bad);
        printf("    flags 0: %zu rows byte-identical to the pre-change encode / decode / fill\n", n_rows);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// (d) plan hash
// ---------------------------------------------------------------------------------------------------------------

static void test_hash() {
    printf("[d] plan hash\n");
    const uint8_t y7[4] = { 7, 7, 7, 7 };

    uint64_t h_new = GGML_TURBOT_FNV_OFFSET, h_old = GGML_TURBOT_FNV_OFFSET;
    for (int i = 0; i < 16; ++i) {
        ggml_turbot_layer l;
        TCHECK(ggml_turbot_layer_init(&l, DEFAULT_BK[i], DEFAULT_BV[i], y7, y7), "default plan layer %d", DEFAULT_IL[i]);
        h_new = ggml_turbot_plan_hash_layer(h_new, DEFAULT_IL[i], &l);
        h_old = old_plan_hash_layer(h_old, DEFAULT_IL[i], &l);
        TCHECK(h_new == h_old, "layer %d: flags-0 fold differs from the old fold", DEFAULT_IL[i]);
    }
    h_new = ggml_turbot_plan_hash_finish(h_new, GGML_TURBOT_POOL_DEFAULT, GGML_TURBOT_CAP_DEFAULT);
    TCHECK(h_new == DEFAULT_PLAN_HASH, "default plan hash 0x%016" PRIx64 ", expected 0x%016" PRIx64, h_new, DEFAULT_PLAN_HASH);
    printf("    default plan (flags 0): 0x%016" PRIx64 "\n", h_new);

    for (const auto & sh : SHAPES) {
        ggml_turbot_layer l;
        TCHECK(ggml_turbot_layer_init_geom(&l, MIX_BK, MIX_BV, MIX_YK, MIX_YV, sh.flags), "flags %u layer", sh.flags);
        const uint64_t hn = ggml_turbot_plan_hash_layer(GGML_TURBOT_FNV_OFFSET, 5, &l);
        const uint64_t ho = old_plan_hash_layer(GGML_TURBOT_FNV_OFFSET, 5, &l);
        if (sh.flags == 0) {
            TCHECK(hn == ho, "flags 0 hash differs from the old fold");
        } else {
            const uint8_t geo[5] = { 'G', 'E', 'O', '1', (uint8_t) sh.flags };
            TCHECK(hn != ho, "flags %u hash equals the old fold", sh.flags);
            TCHECK(hn == ggml_turbot_fnv1a64(ho, geo, 5), "flags %u hash is not the old fold + GEO1 flags", sh.flags);
        }
    }
    // same widths, same nr, other head dim: different hashes
    {
        ggml_turbot_layer a, b;
        TCHECK(ggml_turbot_layer_init_geom(&a, MIX_BK, MIX_BV, MIX_YK, MIX_YV, 1) &&
               ggml_turbot_layer_init_geom(&b, MIX_BK, MIX_BV, MIX_YK, MIX_YV, 5), "nr 2 layers");
        TCHECK(ggml_turbot_plan_hash_layer(GGML_TURBOT_FNV_OFFSET, 3, &a) != ggml_turbot_plan_hash_layer(GGML_TURBOT_FNV_OFFSET, 3, &b),
               "2 x 256 and 4 x 128 with the same widths share a hash");
    }
}

// ---------------------------------------------------------------------------------------------------------------
// (e) type family
// ---------------------------------------------------------------------------------------------------------------

static void test_types() {
    printf("[e] type family S2..S24\n");
    TCHECK((int) GGML_TYPE_TURBOT_S8 == 49 && (int) GGML_TYPE_TURBOT_S24 == 65, "S8..S24 ids moved");
    TCHECK((int) GGML_TYPE_TURBOT_S2 == 66 && (int) GGML_TYPE_TURBOT_S3 == 67 && (int) GGML_TYPE_TURBOT_S4 == 68 &&
           (int) GGML_TYPE_TURBOT_S5 == 69 && (int) GGML_TYPE_TURBOT_S6 == 70 && (int) GGML_TYPE_TURBOT_S7 == 71, "S2..S7 ids");
    TCHECK((int) GGML_TYPE_COUNT == 72, "GGML_TYPE_COUNT %d, expected 72", (int) GGML_TYPE_COUNT);
    TCHECK(ggml_turbot_type_of_s(2) == GGML_TYPE_TURBOT_S2 && ggml_turbot_type_of_s(7) == GGML_TYPE_TURBOT_S7 &&
           ggml_turbot_type_of_s(8) == GGML_TYPE_TURBOT_S8 && ggml_turbot_type_of_s(24) == GGML_TYPE_TURBOT_S24, "type_of_s end points");

    for (int s = GGML_TURBOT_S_MIN_ANY; s <= GGML_TURBOT_S_MAX; ++s) {
        const ggml_type t = ggml_turbot_type_of_s(s);
        TCHECK(ggml_turbot_is_type(t) && ggml_turbot_s_of_type(t) == s, "S %d: type %d round trip", s, (int) t);
        TCHECK((int) t == (s < 8 ? 66 + (s - 2) : 49 + (s - 8)), "S %d: id %d", s, (int) t);
        TCHECK(ggml_type_size(t) == ggml_turbot_base_row_bytes(s) && ggml_type_size(t) == (size_t) (32 * s + 16), "S %d: type size %zu", s, ggml_type_size(t));
        TCHECK(ggml_blck_size(t) == 1024 && ggml_row_size(t, 1024) == (size_t) (32 * s + 16), "S %d: block", s);
        TCHECK(ggml_is_quantized(t), "S %d: not quantized", s);
        char want[32];
        snprintf(want, sizeof(want), "turbot_s%d", s);
        TCHECK(std::strcmp(ggml_type_name(t), want) == 0, "S %d: name %s", s, ggml_type_name(t));
        TCHECK(ggml_get_type_traits(t)->to_float == nullptr && ggml_get_type_traits(t)->from_float_ref == nullptr, "S %d: has a to_float", s);
    }
    const int sizes[6] = { 80, 112, 144, 176, 208, 240 };
    for (int s = 2; s <= 7; ++s) {
        TCHECK(ggml_type_size(ggml_turbot_type_of_s(s)) == (size_t) sizes[s - 2], "S %d size %zu", s, ggml_type_size(ggml_turbot_type_of_s(s)));
    }
    for (int t = 0; t < (int) GGML_TYPE_COUNT + 2; ++t) {
        const bool expect = (t >= 49 && t <= 65) || (t >= 66 && t <= 71);
        TCHECK(ggml_turbot_is_type((ggml_type) t) == expect, "is_type(%d)", t);
    }
    TCHECK(!ggml_turbot_is_type(GGML_TYPE_TURBO5P512_0) && !ggml_turbot_is_type(GGML_TYPE_COUNT), "is_type at 48 or 72");
}

// ---------------------------------------------------------------------------------------------------------------
// (f) op params
// ---------------------------------------------------------------------------------------------------------------

static void test_op_params() {
    printf("[f] op params carry the geometry\n");
    for (const auto & sh : SHAPES) {
        ggml_turbot_layer l;
        TCHECK(ggml_turbot_layer_init_geom(&l, MIX_BK, MIX_BV, MIX_YK, MIX_YV, sh.flags), "flags %u layer", sh.flags);
        for (int side : { GGML_TURBOT_SIDE_K, GGML_TURBOT_SIDE_V, GGML_TURBOT_SIDE_BOTH }) {
            ggml_tensor t;
            std::memset(&t, 0, sizeof(t));
            std::memset(t.op_params, 0x77, GGML_TURBOT_OP_PARAMS_OFFSET);
            ggml_turbot_op_params p, q;
            ggml_turbot_op_params_make(&p, &l, side);
            TCHECK(p.flags == sh.flags, "flags %u: make wrote flags %u", sh.flags, p.flags);
            for (int r = sh.nr; r < 4; ++r) {
                TCHECK(p.bk[r] == 0 && p.bv[r] == 0 && p.yk[r] == 0 && p.yv[r] == 0, "flags %u: run %d beyond nr packed", sh.flags, r);
            }
            ggml_turbot_op_params_set(&t, &p);
            TCHECK(((const uint8_t *) t.op_params)[GGML_TURBOT_OP_PARAMS_OFFSET + 7] == sh.flags, "flags %u: byte 31", sh.flags);
            TCHECK(ggml_turbot_op_params_get(&t, &q) && std::memcmp(&p, &q, sizeof(p)) == 0, "flags %u: get failed or differs", sh.flags);
            ggml_turbot_layer l2;
            std::memset(&l2, 0xEE, sizeof(l2));
            TCHECK(ggml_turbot_layer_from_op_params(&q, &l2) && std::memcmp(&l, &l2, sizeof(l)) == 0, "flags %u: layer from op params differs", sh.flags);
            for (int i = 0; i < GGML_TURBOT_OP_PARAMS_OFFSET; ++i) {
                TCHECK(((const uint8_t *) t.op_params)[i] == 0x77, "op params byte %d overwritten", i);
            }
            if (sh.flags == 0) {
                ggml_turbot_op_params o;
                old_op_params_make(&o, &l, side);
                TCHECK(std::memcmp(&o, &p, sizeof(p)) == 0, "flags 0 op params differ from the old packing");
            }

            // reserved bits and invalid geometries
            for (unsigned bad : { sh.flags | 0x08u, sh.flags | 0x10u, sh.flags | 0x80u, 3u, 7u }) {
                ggml_tensor tb = t;
                ((uint8_t *) tb.op_params)[GGML_TURBOT_OP_PARAMS_OFFSET + 7] = (uint8_t) bad;
                TCHECK(!ggml_turbot_op_params_get(&tb, &q), "flags %u accepted by get", bad);
            }
            // a width beyond nr: get passes, the layer is refused
            if (sh.nr < 4) {
                ggml_tensor tb = t;
                ((uint8_t *) tb.op_params)[GGML_TURBOT_OP_PARAMS_OFFSET + 8 + sh.nr] = 4;   // bk[nr]
                TCHECK(ggml_turbot_op_params_get(&tb, &q) && !ggml_turbot_layer_from_op_params(&q, &l2), "flags %u: bk[%d] beyond nr accepted", sh.flags, sh.nr);
                tb = t;
                ((uint8_t *) tb.op_params)[GGML_TURBOT_OP_PARAMS_OFFSET + 20 + 3] = 7;       // yv[3]
                TCHECK(ggml_turbot_op_params_get(&tb, &q) && !ggml_turbot_layer_from_op_params(&q, &l2), "flags %u: yv[3] beyond nr accepted", sh.flags);
            }
        }
    }
}

#ifdef TURBOT_GEOM_CPU_OPS

// ---------------------------------------------------------------------------------------------------------------
// (g) CPU flash attention reference
// ---------------------------------------------------------------------------------------------------------------

struct fa_case {
    const char * name;
    int   hd;
    int   nh;
    int   gqa;
    int   n_kv;
    int   n_q;
    bool  mask;
    float softcap;
};

static void test_fa() {
    printf("[g] CPU flash attention over turbot K/V vs dense attention over the decoded rows\n");
    const fa_case cases[] = {
        { "D128 NR4 GQA4",          128, 8,  4, 192, 3, false, 0.0f  },
        { "D128 NR4 GQA7 mask",     128, 8,  7, 100, 4, true,  0.0f  },
        { "D256 NR2 GQA8",          256, 2,  8, 160, 2, false, 0.0f  },
        { "D256 NR2 GQA8 softcap",  256, 2,  8, 128, 3, true,  30.0f },
        { "D256 NR1 GQA4",          256, 1,  4, 130, 2, true,  0.0f  },
        { "D128 NR2 GQA4",          128, 4,  4, 192, 2, false, 0.0f  },
        { "D128 NR1 GQA16",         128, 2, 16, 150, 2, true,  0.0f  },
        { "D256 NR4 GQA6 (Qwen)",   256, 4,  6, 192, 2, true,  0.0f  },
    };

    ggml_backend_t cpu = ggml_backend_cpu_init();
    TCHECK(cpu != nullptr, "ggml_backend_cpu_init failed");

    std::mt19937 gen(77);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    for (const auto & c : cases) {
        const int flags = ggml_turbot_geom_flags(c.hd, c.nh);
        TCHECK(flags >= 0, "%s: unsupported shape", c.name);
        if (flags < 0) {
            continue;
        }
        const int nr        = ggml_turbot_geom_nr((unsigned) flags);
        const int row_elems = nr * 256;
        const int hq        = c.nh * c.gqa;
        const int n_gran    = (c.n_kv + 63) / 64;

        ggml_turbot_layer l;
        TCHECK(ggml_turbot_layer_init_geom(&l, MIX_BK, MIX_BV, MIX_YK, MIX_YV, (unsigned) flags), "%s: layer", c.name);

        // granule tier pattern: granule 0 old, then young slots 1, 0, 1, ... (the last partial granule included)
        std::vector<int32_t> gtab(n_gran);
        for (int gi = 0; gi < n_gran; ++gi) {
            gtab[gi] = gi == 0 ? -1 : (gi % 2);
        }
        const int n_pool_rows = 2 * 64;

        ggml_init_params ip = { (size_t) 64 * 1024 * 1024, nullptr, false };
        ggml_context * ctx = ggml_init(ip);
        const ggml_type tk = ggml_turbot_type_of_s(l.k.s);
        const ggml_type tv = ggml_turbot_type_of_s(l.v.s);
        ggml_tensor * kc = ggml_new_tensor_2d(ctx, tk, 1024, c.n_kv);   // container rows of 32*S + 16 bytes
        ggml_tensor * vc = ggml_new_tensor_2d(ctx, tv, 1024, c.n_kv);
        ggml_tensor * pl = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, l.pool_row_bytes, n_pool_rows);
        ggml_tensor * gt = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_gran);
        ggml_tensor * q  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.hd, c.n_q, hq, 1);
        ggml_tensor * m  = c.mask ? ggml_new_tensor_4d(ctx, GGML_TYPE_F16, c.n_kv, c.n_q, 1, 1) : nullptr;

        TCHECK(kc->nb[1] == l.k.base_row_bytes && vc->nb[1] == l.v.base_row_bytes, "%s: container row bytes", c.name);

        // cells: encode K and V; young granules also get their pool rows
        std::memset(pl->data, 0x3C, ggml_nbytes(pl));
        std::vector<float> xk(row_elems), xv(row_elems);
        std::vector<float> kdec((size_t) c.n_kv * row_elems), vdec((size_t) c.n_kv * row_elems);
        for (int i = 0; i < c.n_kv; ++i) {
            for (auto & v : xk) v = nd(gen);
            for (auto & v : xv) v = nd(gen) * 0.5f;
            const int32_t slot = gtab[i >> GGML_TURBOT_LOG2_GRANULE];
            uint8_t * krow = (uint8_t *) kc->data + (size_t) i * kc->nb[1];
            uint8_t * vrow = (uint8_t *) vc->data + (size_t) i * vc->nb[1];
            uint8_t * yrow = slot >= 0 ? (uint8_t *) pl->data + (size_t) ggml_turbot_pool_row(slot, (uint32_t) i) * pl->nb[1] : nullptr;
            ggml_turbot_encode_side(xk.data(), &l.k, krow, yrow);
            ggml_turbot_encode_side(xv.data(), &l.v, vrow, yrow ? yrow + l.pool_v_off : nullptr);
            ggml_turbot_decode_side(krow, yrow, &l.k, kdec.data() + (size_t) i * row_elems);
            ggml_turbot_decode_side(vrow, yrow ? yrow + l.pool_v_off : nullptr, &l.v, vdec.data() + (size_t) i * row_elems);
        }
        std::memcpy(gt->data, gtab.data(), gtab.size() * sizeof(int32_t));
        std::vector<float> qd((size_t) ggml_nelements(q));
        for (auto & v : qd) v = nd(gen);
        std::memcpy(q->data, qd.data(), qd.size() * sizeof(float));
        std::vector<float> md;
        if (m) {
            md.resize((size_t) c.n_kv * c.n_q);
            ggml_fp16_t * mp = (ggml_fp16_t *) m->data;
            for (int iq = 0; iq < c.n_q; ++iq) {
                for (int i = 0; i < c.n_kv; ++i) {
                    // hide a stripe of cells per query, keep at least cell 0 visible
                    const bool hidden = i > 0 && ((i + 5 * iq) % 7 == 0);
                    const float v = hidden ? -INFINITY : 0.0f;
                    md[(size_t) iq * c.n_kv + i] = v;
                    mp[(size_t) iq * c.n_kv + i] = ggml_fp32_to_fp16(v);
                }
            }
        }

        // views [D, n_kv, H, 1] with head stride D values of the row, built like the cache's views. The byte strides
        // are written out because ggml_row_size asserts a whole block.
        auto view = [&](ggml_tensor * cache) {
            const size_t row  = cache->nb[1];
            const size_t head = row * (size_t) c.hd / GGML_TURBOT_ROW_ELEMS;
            ggml_tensor * t = ggml_view_4d(ctx, cache, c.hd, c.nh, c.n_kv, 1, head, row, row * (size_t) c.n_kv, 0);
            return ggml_permute(ctx, t, 0, 2, 1, 3);
        };
        const float scale = 1.0f / std::sqrt((float) c.hd);
        ggml_tensor * out = ggml_flash_attn_ext(ctx, q, view(kc), view(vc), m, scale, 0.0f, c.softcap);
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        ggml_turbot_op_params p;
        ggml_turbot_op_params_make(&p, &l, GGML_TURBOT_SIDE_BOTH);
        ggml_flash_attn_ext_set_turbot(out, pl, gt, &p);

        TCHECK(cpu && ggml_backend_supports_op(cpu, out), "%s: CPU backend does not support the op", c.name);
        // supports_op follows the op-params geometry: another valid geometry or an invalid one is refused
        {
            uint8_t * fb = (uint8_t *) out->op_params + GGML_TURBOT_OP_PARAMS_OFFSET + 7;
            const uint8_t keep = *fb;
            const unsigned other = (unsigned) flags ^ GGML_TURBOT_GEOM_D128;   // same nr, other head dim
            *fb = (uint8_t) other;
            TCHECK(!cpu || !ggml_backend_supports_op(cpu, out), "%s: supported with flags %u", c.name, other);
            *fb = 3;
            TCHECK(!cpu || !ggml_backend_supports_op(cpu, out), "%s: supported with flags 3", c.name);
            *fb = keep;
        }

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        const ggml_status st = ggml_graph_compute_with_ctx(ctx, gf, 2);
        TCHECK(st == GGML_STATUS_SUCCESS, "%s: compute failed", c.name);

        // dense reference in double over the decoded rows; head z at value 256*run + base (the run mapping)
        double max_err = 0.0, max_ref = 0.0;
        const float * od = (const float *) out->data;   // [D, Hq, n_q]
        for (int iq = 0; iq < c.n_q; ++iq) {
            for (int qh = 0; qh < hq; ++qh) {
                const int z    = qh / c.gqa;
                const int run  = c.hd == 256 ? z : z >> 1;
                const int base = c.hd == 256 ? 0 : 128 * (z & 1);
                const int off  = 256 * run + base;
                const float * qv = qd.data() + ((size_t) qh * c.n_q + iq) * c.hd;
                std::vector<double> s(c.n_kv);
                double smax = -INFINITY;
                for (int i = 0; i < c.n_kv; ++i) {
                    const float * kr = kdec.data() + (size_t) i * row_elems + off;
                    double d = 0.0;
                    for (int e = 0; e < c.hd; ++e) {
                        d += (double) qv[e] * kr[e];
                    }
                    d *= scale;
                    if (c.softcap != 0.0f) {
                        d = c.softcap * std::tanh(d / c.softcap);
                    }
                    if (m) {
                        d += md[(size_t) iq * c.n_kv + i];
                    }
                    s[i] = d;
                    smax = std::max(smax, d);
                }
                double sum = 0.0;
                for (int i = 0; i < c.n_kv; ++i) {
                    s[i] = std::isinf(s[i]) ? 0.0 : std::exp(s[i] - smax);
                    sum += s[i];
                }
                for (int e = 0; e < c.hd; ++e) {
                    double acc = 0.0;
                    for (int i = 0; i < c.n_kv; ++i) {
                        acc += s[i] * vdec[(size_t) i * row_elems + off + e];
                    }
                    const double ref = acc / sum;
                    const double got = od[((size_t) iq * hq + qh) * c.hd + e];
                    max_ref = std::max(max_ref, std::fabs(ref));
                    max_err = std::max(max_err, std::isfinite(got) ? std::fabs(got - ref) : INFINITY);
                }
            }
        }
        const double tol = 1e-4 * std::max(1.0, max_ref);
        TCHECK(st == GGML_STATUS_SUCCESS && max_err <= tol, "%s: max |FA - dense| = %.3g (tolerance %.3g)", c.name, max_err, tol);
        printf("    %-24s flags %d, S %d/%d, max |FA - dense| %.3g of max |out| %.3g\n", c.name, flags, l.k.s, l.v.s, max_err, max_ref);
        ggml_free(ctx);
    }
    if (cpu) {
        ggml_backend_free(cpu);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// (h) CPU writer op
// ---------------------------------------------------------------------------------------------------------------

static void test_writer() {
    printf("[h] CPU TURBOT_SET_ROWS at NR 1, 2, 4 against the header coder (fill entries included)\n");
    std::mt19937 gen(4242);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    TCHECK(cpu != nullptr, "ggml_backend_cpu_init failed");

    for (const auto & sh : SHAPES) {
        for (int side : { GGML_TURBOT_SIDE_K, GGML_TURBOT_SIDE_V }) {
            ggml_turbot_layer l;
            TCHECK(ggml_turbot_layer_init_geom(&l, MIX_BK, MIX_BV, MIX_YK, MIX_YV, sh.flags), "flags %u layer", sh.flags);
            const ggml_turbot_side & sd = side == GGML_TURBOT_SIDE_K ? l.k : l.v;
            const size_t part = side == GGML_TURBOT_SIDE_K ? 0 : l.pool_v_off;
            const int row_elems = ggml_turbot_geom_row_elems(sh.flags);
            const int n_cells = 192, n_pool_rows = 128;

            // initial state: every cell holds an old-only row, the pool holds garbage
            std::vector<uint8_t> base0((size_t) n_cells * sd.base_row_bytes), pool0((size_t) n_pool_rows * l.pool_row_bytes, 0x33);
            std::vector<float> x(row_elems);
            for (int i = 0; i < n_cells; ++i) {
                for (auto & v : x) v = nd(gen);
                ggml_turbot_encode_side(x.data(), &sd, base0.data() + (size_t) i * sd.base_row_bytes, nullptr);
            }

            // fill (granule 1 -> slot 0: cells 64..67, 76..79, 96) and 6 rows, one of them onto a filled cell
            const int32_t fill_e[4] = { 1, 0, (int32_t) 0x0000F00Fu, 0x1 };
            const int32_t cells[6]  = { 5, 70, 76, 130, 131, 7 };
            const int32_t young[6]  = { -1, 6, 12, -1, 64 + 3, -1 };
            std::vector<float> rows((size_t) 6 * row_elems);
            for (auto & v : rows) v = nd(gen);

            // expected bytes through the header coder, in the op's order: fill first, then the rows
            std::vector<uint8_t> base_ref = base0, pool_ref = pool0;
            const uint64_t mask = (uint64_t) (uint32_t) fill_e[2] | ((uint64_t) (uint32_t) fill_e[3] << 32);
            for (int cc = 0; cc < 64; ++cc) {
                if ((mask >> cc) & 1) {
                    const int cell = fill_e[0] * 64 + cc;
                    ggml_turbot_fill_side(base_ref.data() + (size_t) cell * sd.base_row_bytes, &sd,
                                          pool_ref.data() + (size_t) (fill_e[1] * 64 + cc) * l.pool_row_bytes + part);
                }
            }
            for (int i = 0; i < 6; ++i) {
                ggml_turbot_encode_side(rows.data() + (size_t) i * row_elems, &sd, base_ref.data() + (size_t) cells[i] * sd.base_row_bytes,
                                        young[i] >= 0 ? pool_ref.data() + (size_t) young[i] * l.pool_row_bytes + part : nullptr);
            }

            ggml_init_params ip = { (size_t) 16 * 1024 * 1024, nullptr, false };
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * a  = ggml_new_tensor_2d(ctx, ggml_turbot_type_of_s(sd.s), 1024, n_cells);
            ggml_tensor * b  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, row_elems, 6);
            ggml_tensor * c  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 6);
            ggml_tensor * pl = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, l.pool_row_bytes, n_pool_rows);
            ggml_tensor * yr = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 6);
            ggml_tensor * fl = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 4, 1);
            TCHECK(ggml_nbytes(a) == base0.size() && ggml_nbytes(pl) == pool0.size(), "flags %u: tensor sizes", sh.flags);
            std::memcpy(a->data, base0.data(), base0.size());
            std::memcpy(b->data, rows.data(), rows.size() * sizeof(float));
            std::memcpy(c->data, cells, sizeof(cells));
            std::memcpy(pl->data, pool0.data(), pool0.size());
            std::memcpy(yr->data, young, sizeof(young));
            std::memcpy(fl->data, fill_e, sizeof(fill_e));

            ggml_turbot_op_params p;
            ggml_turbot_op_params_make(&p, &l, side);
            ggml_tensor * out = ggml_turbot_set_rows(ctx, a, b, c, pl, yr, fl, &p);
            TCHECK(cpu && ggml_backend_supports_op(cpu, out), "flags %u side %d: CPU backend does not support the writer", sh.flags, side);

            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, out);
            const ggml_status st = ggml_graph_compute_with_ctx(ctx, gf, 2);
            TCHECK(st == GGML_STATUS_SUCCESS, "flags %u side %d: compute failed", sh.flags, side);
            TCHECK(std::memcmp(a->data, base_ref.data(), base_ref.size()) == 0, "flags %u side %d: base bytes differ from the header coder", sh.flags, side);
            TCHECK(std::memcmp(pl->data, pool_ref.data(), pool_ref.size()) == 0, "flags %u side %d: pool bytes differ from the header coder", sh.flags, side);
            ggml_free(ctx);
        }
    }
    if (cpu) {
        ggml_backend_free(cpu);
    }
}

#endif // TURBOT_GEOM_CPU_OPS

int main(int argc, char ** argv) {
    if (argc > 1) {
        fprintf(stderr, "usage: %s\n", argv[0]);
        return 2;
    }

    test_geom_flags();
    test_init();
    test_coder();
    test_hash();
    test_types();
    test_op_params();
#ifdef TURBOT_GEOM_CPU_OPS
    test_fa();
    test_writer();
#else
    printf("[g] CPU flash attention reference: SKIPPED (the CPU backend is not linked with GGML_BACKEND_DL)\n");
    printf("[h] CPU writer op: SKIPPED (the CPU backend is not linked with GGML_BACKEND_DL)\n");
#endif

    printf("\n%d checks, %d failed\n", g_checks, g_fail);
    printf("%s\n", g_fail == 0 ? "OK" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
