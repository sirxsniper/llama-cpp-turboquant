// [TAG_TURBOT] CPU vs CUDA byte-level checks of the turbot KV cache kernels, docs/turbot/SPEC.md section 11.3.
//
//   1. writer (GGML_OP_TURBOT_SET_ROWS, fill entries included): base and pool bytes of the CUDA kernel against the CPU
//      op, and the CPU op against ggml-turbot.h encode_side / fill_side. Codes must agree on >= 99.99% of the elements,
//      gains within one f16 step (SPEC 4.5), the CPU op must equal the header bit for bit, and no byte outside the
//      written rows and pool parts may change on either backend.
//   2. old-tier read: a FLASH_ATTN_EXT whose query i sees only cell i returns that cell's V row, so the CUDA OLD loader
//      is checked element by element against ggml_turbot_old_level_i8 * ggml_turbot_old_i8_scale * gain (b <= 5,
//      GGML_CUDA_TURBOT_OLD_I8 1 builds) or C_b * gain (b = 6, or --old-read float for OLD_I8 0 builds), within one half
//      step.
//   [TAG_TURBOT_ANY_TEST] 1 and 2 again at the other KV geometries (ggml_turbot_geom_*): writer rows of 1, 2 and 4 runs
//   (256, 512, 1024 values, S < 8 included) at D = 256 and D = 128, and the old read of D = 128 heads (head z = run
//   z >> 1 at element 128*(z & 1)) and of 2 x 256 / 1 x 256 rows. The Qwen3.8-27B cases above run first and unchanged.
//
// Needs a CUDA device; exits 0 with a SKIP line when there is none. Not registered with ctest (it uses the GPU, and GPU
// jobs run one at a time, see docs/turbot/TESTING.md). Run it under compute-sanitizer with --quick first.
//
//   test-turbot-backend [--quick] [--old-read i8|float]

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-turbot.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <vector>

static int  g_fail  = 0;
static bool g_quick = false;

#define TBCHECK(cond, ...)                                                  \
    do {                                                                    \
        if (!(cond)) {                                                      \
            ++g_fail;                                                       \
            if (g_fail <= 200) {                                            \
                fprintf(stderr, "FAIL %s:%d: ", __func__, __LINE__);        \
                fprintf(stderr, __VA_ARGS__);                               \
                fprintf(stderr, "\n");                                      \
            }                                                               \
        }                                                                   \
    } while (0)

// spacing of f16 values around |x| (normal range 2^(e-10), subnormal 2^-24)
static double f16_step(double x) {
    const double ax = std::fabs(x);
    if (ax < 65504.0 * 2.0 && ax >= 0.0) {
        int e = ax > 0.0 ? (int) std::floor(std::log2(ax)) : -14;
        e = std::max(e, -14);
        return std::ldexp(1.0, e - 10);
    }
    return std::numeric_limits<double>::infinity();
}

// Equal values are close, the same inf on both sides included: a group whose norm / |recon| exceeds 65504 (the 1e4
// rows of make_row) stores f16 inf on both backends, and inf - inf is NaN, which the step test would reject.
static bool gains_close(float a, float b, double steps = 1.0) {
    if (a == b) {
        return true;
    }
    return std::fabs((double) a - (double) b) <= (steps + 1e-4) * f16_step(std::max(std::fabs(a), std::fabs(b)));
}

static bool make_layer(ggml_turbot_layer & l, const uint8_t bk[4], const uint8_t bv[4], const uint8_t yk[4], const uint8_t yv[4]) {
    const bool ok = ggml_turbot_layer_init(&l, bk, bv, yk, yv);
    TBCHECK(ok, "illegal test layer");
    return ok;
}

// [TAG_TURBOT_ANY_TEST] a layer at geometry flags (runs r >= NR are b = y = 0)
static bool make_layer_geom(ggml_turbot_layer & l, const uint8_t bk[4], const uint8_t bv[4], const uint8_t yk[4], const uint8_t yv[4], int flags) {
    const bool ok = ggml_turbot_layer_init_geom(&l, bk, bv, yk, yv, (uint8_t) flags);
    TBCHECK(ok, "illegal test layer (flags %d)", flags);
    return ok;
}

// [TAG_TURBOT_ANY_TEST] values per written row of a layer: NR * 256 (1024 for the 4 x 256 and 8 x 128 rows)
static int64_t layer_row_elems(const ggml_turbot_layer & l) {
    return ggml_turbot_geom_row_elems(l.flags);
}

// deterministic assorted rows: per-row scale over 1e-6 .. 1e4, outlier channels, zero groups, negative zeros
static void make_row(std::mt19937 & gen, int64_t i, float * x) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    static const float scales[8] = { 1.0f, 1e-6f, 0.05f, 3.0f, 1e4f, 0.25f, 40.0f, 1.0f };
    const float sc = scales[i % 8];
    for (int k = 0; k < 1024; ++k) {
        x[k] = sc * nd(gen);
    }
    if (i % 5 == 1) {
        for (int k = 3; k < 1024; k += 61) {
            x[k] *= 50.0f;
        }
    }
    if (i % 11 == 4) {
        std::fill(x + 512, x + 640, 0.0f);
    }
    if (i % 13 == 6) {
        std::fill(x, x + 1024, -0.0f);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// writer
// ---------------------------------------------------------------------------------------------------------------

struct writer_case {
    std::string              name;
    ggml_turbot_layer        l;
    int                      side;
    bool                     idx_i32;
    int64_t                  kv_size;
    int64_t                  n_pool_rows;
    std::vector<int64_t>     cells;    // destination cell of each row
    std::vector<int32_t>     young;    // pool row of each row or -1
    std::vector<int32_t>     fill;     // 4 * n_fill: granule, slot, mask_lo, mask_hi
    std::vector<float>       rows;     // row_n * n_rows
    int64_t                  row_n = 1024;   // [TAG_TURBOT_ANY_TEST] values per row, NR * 256
    std::vector<uint8_t>     base0;    // kv_size * row bytes, the cache before the op
    std::vector<uint8_t>     pool0;    // n_pool_rows * pool_row_bytes
};

// run one TURBOT_SET_ROWS on a backend, return the cache and pool bytes after it
static bool run_writer(ggml_backend_t be, const writer_case & wc, std::vector<uint8_t> & base, std::vector<uint8_t> & pool) {
    const ggml_turbot_side & sd = wc.side == GGML_TURBOT_SIDE_K ? wc.l.k : wc.l.v;
    const int64_t n_rows = (int64_t) wc.cells.size();
    const int64_t n_fill = (int64_t) wc.fill.size() / 4;

    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a   = ggml_new_tensor_2d(ctx, ggml_turbot_type_of_s(sd.s), 1024, wc.kv_size);
    ggml_tensor * b   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, wc.row_n, n_rows);
    ggml_tensor * c   = ggml_new_tensor_1d(ctx, wc.idx_i32 ? GGML_TYPE_I32 : GGML_TYPE_I64, n_rows);
    ggml_tensor * pl  = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, wc.l.pool_row_bytes, wc.n_pool_rows);
    ggml_tensor * yr  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_rows);
    ggml_tensor * fl  = n_fill > 0 ? ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 4, n_fill) : nullptr;
    ggml_turbot_op_params p;
    ggml_turbot_op_params_make(&p, &wc.l, wc.side);
    ggml_tensor * out = ggml_turbot_set_rows(ctx, a, b, c, pl, yr, fl, &p);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    if (!ggml_backend_supports_op(be, out)) {
        fprintf(stderr, "%s: %s does not support TURBOT_SET_ROWS\n", wc.name.c_str(), ggml_backend_name(be));
        ggml_free(ctx);
        return false;
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) {
        fprintf(stderr, "%s: allocation failed on %s\n", wc.name.c_str(), ggml_backend_name(be));
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(a, wc.base0.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, wc.rows.data(), 0, ggml_nbytes(b));
    if (wc.idx_i32) {
        std::vector<int32_t> ci(wc.cells.begin(), wc.cells.end());
        ggml_backend_tensor_set(c, ci.data(), 0, ggml_nbytes(c));
    } else {
        ggml_backend_tensor_set(c, wc.cells.data(), 0, ggml_nbytes(c));
    }
    ggml_backend_tensor_set(pl, wc.pool0.data(), 0, ggml_nbytes(pl));
    ggml_backend_tensor_set(yr, wc.young.data(), 0, ggml_nbytes(yr));
    if (fl) {
        ggml_backend_tensor_set(fl, wc.fill.data(), 0, ggml_nbytes(fl));
    }
    const ggml_status st = ggml_backend_graph_compute(be, gf);
    bool ok = st == GGML_STATUS_SUCCESS;
    if (ok) {
        ggml_backend_synchronize(be);
        base.resize(ggml_nbytes(a));
        pool.resize(ggml_nbytes(pl));
        ggml_backend_tensor_get(a, base.data(), 0, base.size());
        ggml_backend_tensor_get(pl, pool.data(), 0, pool.size());
    } else {
        fprintf(stderr, "%s: compute failed on %s: %s\n", wc.name.c_str(), ggml_backend_name(be), ggml_status_to_string(st));
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

// host reference of the op (SPEC 6.1): fills first, then rows
static void reference_writer(const writer_case & wc, std::vector<uint8_t> & base, std::vector<uint8_t> & pool) {
    const ggml_turbot_side & sd = wc.side == GGML_TURBOT_SIDE_K ? wc.l.k : wc.l.v;
    const size_t part = wc.side == GGML_TURBOT_SIDE_K ? 0 : wc.l.pool_v_off;
    base = wc.base0;
    pool = wc.pool0;
    for (size_t f = 0; f + 3 < wc.fill.size(); f += 4) {
        const int32_t G = wc.fill[f], slot = wc.fill[f + 1];
        const uint64_t mask = (uint64_t) (uint32_t) wc.fill[f + 2] | ((uint64_t) (uint32_t) wc.fill[f + 3] << 32);
        for (int c = 0; c < 64; ++c) {
            if (mask >> c & 1) {
                ggml_turbot_fill_side(base.data() + (size_t) (G * 64 + c) * sd.base_row_bytes, &sd,
                                      pool.data() + (size_t) (slot * 64 + c) * wc.l.pool_row_bytes + part);
            }
        }
    }
    for (size_t i = 0; i < wc.cells.size(); ++i) {
        ggml_turbot_encode_side(wc.rows.data() + wc.row_n * i, &sd, base.data() + (size_t) wc.cells[i] * sd.base_row_bytes,
                                wc.young[i] >= 0 ? pool.data() + (size_t) wc.young[i] * wc.l.pool_row_bytes + part : nullptr);
    }
}

struct cmp_stats {
    uint64_t elems      = 0;
    uint64_t code_diff  = 0;
    uint64_t gains      = 0;
    uint64_t gain_far   = 0;   // beyond one f16 step
    uint64_t gain_bad   = 0;   // beyond two f16 steps: never allowed
    double   gain_steps = 0;   // worst gain distance in f16 steps (finite gains only)
    double   gain_a     = 0;   // the gains at that worst distance
    double   gain_b     = 0;
    uint64_t gain_skip  = 0;   // groups whose codes differ: a flipped threshold tie legitimately moves |recon| and the gain
};

// codes and gains of one side row (base) or one side part of a pool row (young)
static void compare_run(const ggml_turbot_side & sd, bool young, const uint8_t * pa, const uint8_t * pb, cmp_stats & s) {
    bool group_codes_differ[8] = {};
    const int nr = sd.nr;   // [TAG_TURBOT_ANY_TEST] 4 for every Qwen case: the same runs and gains as before
    for (int h = 0; h < nr; ++h) {
        const int w   = young ? sd.y[h] - sd.b[h] : sd.b[h];
        const int off = young ? sd.young_off[h] : sd.base_off[h];
        for (int e = 0; e < 256; ++e) {
            s.elems++;
            const bool d = ggml_turbot_get_code(pa + off, w, e) != ggml_turbot_get_code(pb + off, w, e);
            s.code_diff += d;
            group_codes_differ[2*h + (e < 128 ? 0 : 1)] |= d;
        }
    }
    const int goff = young ? sd.young_gain_off : sd.base_gain_off;
    for (int i = 0; i < 2*nr; ++i) {
        s.gains++;
        const float ga = ggml_turbot_read_gain(pa + goff, i);
        const float gb = ggml_turbot_read_gain(pb + goff, i);
        if (group_codes_differ[i]) {
            s.gain_skip++;
            continue;
        }
        s.gain_far += !gains_close(ga, gb, 1.0);
        s.gain_bad += !gains_close(ga, gb, 2.0);
        if (ga != gb && std::isfinite(ga) && std::isfinite(gb)) {
            const double st = std::fabs((double) ga - (double) gb) / f16_step(std::max(std::fabs(ga), std::fabs(gb)));
            if (st > s.gain_steps) { s.gain_steps = st; s.gain_a = ga; s.gain_b = gb; }
        }
    }
}

static void check_writer_case(ggml_backend_t cpu, ggml_backend_t gpu, const writer_case & wc) {
    const ggml_turbot_side & sd = wc.side == GGML_TURBOT_SIDE_K ? wc.l.k : wc.l.v;
    const size_t part = wc.side == GGML_TURBOT_SIDE_K ? 0 : wc.l.pool_v_off;
    const size_t prb  = wc.l.pool_row_bytes;

    std::vector<uint8_t> ref_base, ref_pool, cpu_base, cpu_pool, gpu_base, gpu_pool;
    reference_writer(wc, ref_base, ref_pool);
    if (!run_writer(cpu, wc, cpu_base, cpu_pool) || !run_writer(gpu, wc, gpu_base, gpu_pool)) {
        TBCHECK(false, "%s: writer did not run", wc.name.c_str());
        return;
    }
    TBCHECK(cpu_base == ref_base && cpu_pool == ref_pool, "%s: CPU op differs from ggml-turbot.h encode_side/fill_side", wc.name.c_str());

    // which base rows and pool bytes the op may touch
    std::set<int64_t> written(wc.cells.begin(), wc.cells.end());
    std::vector<uint8_t> pool_touch(wc.pool0.size(), 0);
    std::vector<std::pair<size_t, int>> young_parts;   // (pool part byte offset, 0)
    for (size_t i = 0; i < wc.young.size(); ++i) {
        if (wc.young[i] >= 0) {
            const size_t o = (size_t) wc.young[i] * prb + part;
            std::fill(pool_touch.begin() + o, pool_touch.begin() + o + sd.young_bytes, 1);
            young_parts.push_back({ o, 0 });
        }
    }
    for (size_t f = 0; f + 3 < wc.fill.size(); f += 4) {
        const uint64_t mask = (uint64_t) (uint32_t) wc.fill[f + 2] | ((uint64_t) (uint32_t) wc.fill[f + 3] << 32);
        for (int c = 0; c < 64; ++c) {
            if (mask >> c & 1) {
                const size_t o = (size_t) (wc.fill[f + 1] * 64 + c) * prb + part;
                std::fill(pool_touch.begin() + o, pool_touch.begin() + o + sd.young_bytes, 1);
                young_parts.push_back({ o, 1 });
            }
        }
    }

    uint64_t stray = 0;
    cmp_stats sb, sy, sf;
    for (int64_t cell = 0; cell < wc.kv_size; ++cell) {
        const size_t o = (size_t) cell * sd.base_row_bytes;
        if (written.count(cell)) {
            compare_run(sd, false, cpu_base.data() + o, gpu_base.data() + o, sb);
        } else {
            stray += std::memcmp(gpu_base.data() + o, wc.base0.data() + o, sd.base_row_bytes) != 0;
        }
    }
    for (size_t i = 0; i < wc.pool0.size(); ++i) {
        stray += !pool_touch[i] && gpu_pool[i] != wc.pool0[i];
    }
    for (auto & yp : young_parts) {
        compare_run(sd, true, cpu_pool.data() + yp.first, gpu_pool.data() + yp.first, yp.second ? sf : sy);
    }
    auto agree = [](const cmp_stats & s) {
        const bool codes = s.elems == 0 || (double) s.code_diff <= 1e-4 * (double) s.elems;
        const bool gains = s.gain_far == 0;   // over groups with identical codes (gain_skip counts the rest)
        return codes && gains;
    };
    TBCHECK(stray == 0, "%s: CUDA changed %" PRIu64 " bytes/rows outside the op's rows and pool parts", wc.name.c_str(), stray);
    TBCHECK(agree(sb), "%s: base codes differ on %" PRIu64 "/%" PRIu64 ", gains beyond one f16 step %" PRIu64 "/%" PRIu64 ", beyond two %" PRIu64 ", worst %.2f steps (%.9g vs %.9g), groups skipped for differing codes %" PRIu64,
            wc.name.c_str(), sb.code_diff, sb.elems, sb.gain_far, sb.gains, sb.gain_bad, sb.gain_steps, sb.gain_a, sb.gain_b, sb.gain_skip);
    TBCHECK(agree(sy), "%s: refinement codes differ on %" PRIu64 "/%" PRIu64 ", gains beyond one f16 step %" PRIu64 "/%" PRIu64,
            wc.name.c_str(), sy.code_diff, sy.elems, sy.gain_far, sy.gains);
    TBCHECK(agree(sf), "%s: fill codes differ on %" PRIu64 "/%" PRIu64 ", gains beyond one f16 step %" PRIu64 "/%" PRIu64,
            wc.name.c_str(), sf.code_diff, sf.elems, sf.gain_far, sf.gains);
    printf("  %-58s base %" PRIu64 "/%" PRIu64 "  young %" PRIu64 "/%" PRIu64 "  fill %" PRIu64 "/%" PRIu64 " codes differ\n",
           wc.name.c_str(), sb.code_diff, sb.elems, sy.code_diff, sy.elems, sf.code_diff, sf.elems);
}

static writer_case build_writer_case(const char * lname, const ggml_turbot_layer & l, int side, int64_t n_rows, int young_mode, bool fill, bool idx_i32, uint32_t seed) {
    static const char * ymode[3] = { "none", "all", "mixed" };
    writer_case wc;
    wc.name    = std::string(lname) + (side == GGML_TURBOT_SIDE_K ? " K" : " V") + " rows=" + std::to_string(n_rows) +
                 " young=" + ymode[young_mode] + (fill ? " fill" : "") + (idx_i32 ? " i32" : " i64");
    wc.l       = l;
    wc.row_n   = layer_row_elems(l);   // [TAG_TURBOT_ANY_TEST]
    wc.side    = side;
    wc.idx_i32 = idx_i32;
    wc.kv_size = GGML_PAD(2 * n_rows + 128, 64);
    const ggml_turbot_side & sd = side == GGML_TURBOT_SIDE_K ? l.k : l.v;
    const int64_t n_gran = wc.kv_size / 64;
    std::mt19937 gen(seed);

    std::vector<int64_t> perm(wc.kv_size);
    for (int64_t i = 0; i < wc.kv_size; ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), gen);
    wc.cells.assign(perm.begin(), perm.begin() + n_rows);

    std::vector<uint64_t> written(n_gran, 0);
    for (int64_t c : wc.cells) {
        written[c / 64] |= 1ull << (c & 63);
    }
    std::vector<int32_t> gslot(n_gran, -1);
    std::vector<int64_t> young_gran, fill_gran;
    for (int64_t g = 0; g < n_gran; ++g) {
        const bool has = written[g] != 0;
        bool y = false;
        if (young_mode == 1) y = has;
        if (young_mode == 2) y = has && g % 3 != 0;
        if (y) young_gran.push_back(g);
    }
    if (fill) {
        // one granule without rows and one with rows gain a slot while holding old cells
        int64_t g_empty = -1, g_rows = -1;
        for (int64_t g = n_gran - 1; g >= 0 && (g_empty < 0 || g_rows < 0); --g) {
            const bool is_young = std::find(young_gran.begin(), young_gran.end(), g) != young_gran.end();
            if (is_young) continue;
            if (written[g] == 0 && g_empty < 0) g_empty = g;
            if (written[g] != 0 && g_rows < 0 && written[g] != ~0ull) g_rows = g;
        }
        for (int64_t g : { g_empty, g_rows }) {
            if (g >= 0) fill_gran.push_back(g);
        }
    }
    std::vector<int64_t> all_young = young_gran;
    all_young.insert(all_young.end(), fill_gran.begin(), fill_gran.end());
    for (size_t k = 0; k < all_young.size(); ++k) {
        gslot[all_young[k]] = (int32_t) (all_young.size() - 1 - k);   // reversed: slot order != granule order
    }
    wc.n_pool_rows = std::max<int64_t>(64, 64 * (int64_t) all_young.size());

    wc.young.resize(n_rows);
    for (int64_t i = 0; i < n_rows; ++i) {
        const int32_t s = gslot[wc.cells[i] / 64];
        wc.young[i] = s >= 0 ? ggml_turbot_pool_row(s, (uint32_t) wc.cells[i]) : -1;
    }
    for (int64_t g : fill_gran) {
        const uint64_t mask = ~written[g];
        wc.fill.push_back((int32_t) g);
        wc.fill.push_back(gslot[g]);
        wc.fill.push_back((int32_t) (uint32_t) (mask & 0xffffffffu));
        wc.fill.push_back((int32_t) (uint32_t) (mask >> 32));
    }

    // cache before the op: every cell holds a valid old code, pool rows are noise
    wc.base0.assign((size_t) wc.kv_size * sd.base_row_bytes, 0);
    std::vector<float> x(1024);
    for (int64_t cell = 0; cell < wc.kv_size; ++cell) {
        make_row(gen, cell + 3, x.data());
        ggml_turbot_encode_side(x.data(), &sd, wc.base0.data() + (size_t) cell * sd.base_row_bytes, nullptr);
    }
    wc.pool0.resize((size_t) wc.n_pool_rows * l.pool_row_bytes);
    for (auto & v : wc.pool0) {
        v = (uint8_t) gen();
    }
    wc.rows.resize(wc.row_n * n_rows);
    for (int64_t i = 0; i < n_rows; ++i) {
        if (wc.row_n == 1024) {
            make_row(gen, i, wc.rows.data() + 1024 * i);
        } else {   // [TAG_TURBOT_ANY_TEST] the first row_n values of a full test row
            make_row(gen, i, x.data());
            std::copy(x.begin(), x.begin() + wc.row_n, wc.rows.begin() + wc.row_n * i);
        }
    }
    return wc;
}

static void test_writer(ggml_backend_t cpu, ggml_backend_t gpu) {
    printf("[1] TURBOT_SET_ROWS: CUDA vs CPU bytes\n");
    const uint8_t mixed_bk[4] = { 2, 6, 3, 5 }, mixed_yk[4] = { 8, 7, 7, 7 };
    const uint8_t mixed_bv[4] = { 4, 2, 6, 3 }, mixed_yv[4] = { 7, 7, 8, 5 };
    const uint8_t l23_bk[4]   = { 5, 4, 4, 6 }, l23_bv[4]   = { 5, 4, 4, 5 };
    const uint8_t b2[4] = { 2, 2, 2, 2 }, b6[4] = { 6, 6, 6, 6 }, y7[4] = { 7, 7, 7, 7 }, y8[4] = { 8, 8, 8, 8 };
    struct named_layer { const char * name; ggml_turbot_layer l; };
    std::vector<named_layer> layers(4);
    layers[0].name = "mixed";  make_layer(layers[0].l, mixed_bk, mixed_bv, mixed_yk, mixed_yv);
    layers[1].name = "l23";    make_layer(layers[1].l, l23_bk, l23_bv, y7, y7);
    layers[2].name = "b2y8";   make_layer(layers[2].l, b2, b2, y8, y8);
    layers[3].name = "b6y7";   make_layer(layers[3].l, b6, b6, y7, y7);

    const std::vector<int64_t> row_counts = g_quick ? std::vector<int64_t>{ 1, 7, 64 } : std::vector<int64_t>{ 1, 7, 64, 1280, 2048 };
    uint32_t seed = 1;
    for (const auto & nl : layers) {
        for (int side : { GGML_TURBOT_SIDE_K, GGML_TURBOT_SIDE_V }) {
            for (int64_t n : row_counts) {
                for (int ym = 0; ym < 3; ++ym) {
                    const bool fill = ym != 1;
                    const bool i32  = (seed % 2) == 0;
                    writer_case wc = build_writer_case(nl.name, nl.l, side, n, ym, fill, i32, seed++);
                    check_writer_case(cpu, gpu, wc);
                }
            }
        }
    }

    // [TAG_TURBOT_ANY_TEST] the other geometries, after the Qwen cases (whose seeds and names above are unchanged):
    // NR 4 at D = 128 (8 x 128, the same NG 8 kernel with other op params), NR 2 (512-value rows: 2 x 256, 4 x 128) and
    // NR 1 (256-value rows: 1 x 256, 2 x 128), with S < 8 widths, young 8 and every refinement width r = 1..6.
    printf("[1b] TURBOT_SET_ROWS at other KV geometries: CUDA vs CPU bytes\n");
    struct geom_layer { const char * name; int flags; uint8_t bk[4], bv[4], yk[4], yv[4]; };
    const geom_layer glayers[] = {
        { "d128x8 mixed", GGML_TURBOT_GEOM_D128 | 0, { 2, 6, 3, 5 }, { 4, 2, 6, 3 }, { 8, 7, 7, 7 }, { 7, 7, 8, 5 } },
        { "d128x8 l23",   GGML_TURBOT_GEOM_D128 | 0, { 5, 4, 4, 6 }, { 5, 4, 4, 5 }, { 7, 7, 7, 7 }, { 7, 7, 7, 7 } },
        { "d256x2 a",     1,                         { 2, 2, 0, 0 }, { 3, 4, 0, 0 }, { 7, 7, 0, 0 }, { 7, 7, 0, 0 } },
        { "d256x2 b",     1,                         { 3, 4, 0, 0 }, { 2, 2, 0, 0 }, { 8, 5, 0, 0 }, { 8, 7, 0, 0 } },
        { "d128x4 b6",    GGML_TURBOT_GEOM_D128 | 1, { 6, 5, 0, 0 }, { 6, 6, 0, 0 }, { 7, 8, 0, 0 }, { 8, 7, 0, 0 } },
        { "d256x1 b2",    2,                         { 2, 0, 0, 0 }, { 3, 0, 0, 0 }, { 8, 0, 0, 0 }, { 7, 0, 0, 0 } },
        { "d256x1 b4",    2,                         { 4, 0, 0, 0 }, { 5, 0, 0, 0 }, { 7, 0, 0, 0 }, { 7, 0, 0, 0 } },
        { "d128x2 b6",    GGML_TURBOT_GEOM_D128 | 2, { 6, 0, 0, 0 }, { 2, 0, 0, 0 }, { 8, 0, 0, 0 }, { 7, 0, 0, 0 } },
        { "d128x2 b3",    GGML_TURBOT_GEOM_D128 | 2, { 3, 0, 0, 0 }, { 5, 0, 0, 0 }, { 7, 0, 0, 0 }, { 8, 0, 0, 0 } },
    };
    const std::vector<int64_t> geom_rows = g_quick ? std::vector<int64_t>{ 1, 64 } : std::vector<int64_t>{ 1, 7, 64, 1280 };
    for (const auto & gl : glayers) {
        ggml_turbot_layer l;
        if (!make_layer_geom(l, gl.bk, gl.bv, gl.yk, gl.yv, gl.flags)) {
            continue;
        }
        for (int side : { GGML_TURBOT_SIDE_K, GGML_TURBOT_SIDE_V }) {
            for (int64_t n : geom_rows) {
                for (int ym = 0; ym < 3; ++ym) {
                    const bool fill = ym != 1;
                    const bool i32  = (seed % 2) == 0;
                    writer_case wc = build_writer_case(gl.name, l, side, n, ym, fill, i32, seed++);
                    check_writer_case(cpu, gpu, wc);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// old-tier read through the CUDA FA loaders
// ---------------------------------------------------------------------------------------------------------------

static void test_old_read(ggml_backend_t gpu, bool expect_i8) {
    printf("[2] old-tier FA read: V value per element vs %s levels\n", expect_i8 ? "int8 register (b <= 5)" : "float");
    const uint8_t sets[][4] = { { 2, 2, 2, 2 }, { 3, 3, 3, 3 }, { 4, 4, 4, 4 }, { 5, 5, 5, 5 }, { 2, 3, 4, 5 }, { 6, 6, 6, 6 }, { 6, 2, 5, 3 } };
    const uint8_t y8[4] = { 8, 8, 8, 8 };
    const int64_t n_cells = g_quick ? 64 : 256;
    const int64_t nq      = n_cells;
    std::mt19937 gen(2026);

    for (const auto & bset : sets) {
        ggml_turbot_layer l;
        const uint8_t bk[4] = { 4, 4, 4, 4 };
        if (!make_layer(l, bk, bset, y8, y8)) {
            continue;
        }
        std::vector<uint8_t> kb((size_t) n_cells * l.k.base_row_bytes), vb((size_t) n_cells * l.v.base_row_bytes);
        std::vector<float> x(1024);
        for (int64_t c = 0; c < n_cells; ++c) {
            // Skip the 1e-6 scale (half steps would dominate) and the 1e4 scale: its group norms (~1.1e5) overflow the
            // f16 gain to inf, and one inf K or V cell made every CUDA FA output of this test NaN.
            const int64_t ri = (c % 8 == 1 || c % 8 == 4) ? 0 : c;
            make_row(gen, ri, x.data());
            ggml_turbot_encode_side(x.data(), &l.k, kb.data() + (size_t) c * l.k.base_row_bytes, nullptr);
            make_row(gen, ri, x.data());
            ggml_turbot_encode_side(x.data(), &l.v, vb.data() + (size_t) c * l.v.base_row_bytes, nullptr);
        }

        ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * q   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, nq, 24, 1);
        ggml_tensor * kc  = ggml_new_tensor_2d(ctx, ggml_turbot_type_of_s(l.k.s), 1024, n_cells);
        ggml_tensor * vc  = ggml_new_tensor_2d(ctx, ggml_turbot_type_of_s(l.v.s), 1024, n_cells);
        ggml_tensor * pl  = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, l.pool_row_bytes, GGML_TURBOT_POOL_MIN_ROWS);
        ggml_tensor * gt  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_cells / 64);
        ggml_tensor * m   = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_cells, nq, 1, 1);
        auto view = [&](ggml_tensor * cache) {
            ggml_tensor * t = ggml_view_4d(ctx, cache, 256, 4, n_cells, 1, ggml_row_size(cache->type, 256),
                                           ggml_row_size(cache->type, 1024), ggml_row_size(cache->type, 1024 * n_cells), 0);
            return ggml_permute(ctx, t, 0, 2, 1, 3);
        };
        ggml_tensor * out = ggml_flash_attn_ext(ctx, q, view(kc), view(vc), m, 1.0f / 16.0f, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        ggml_turbot_op_params p;
        ggml_turbot_op_params_make(&p, &l, GGML_TURBOT_SIDE_BOTH);
        ggml_flash_attn_ext_set_turbot(out, pl, gt, &p);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);

        char name[64];
        snprintf(name, sizeof(name), "V b %d %d %d %d", bset[0], bset[1], bset[2], bset[3]);
        if (!ggml_backend_supports_op(gpu, out)) {
            TBCHECK(false, "%s: CUDA does not support the turbot FA", name);
            ggml_free(ctx);
            continue;
        }
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
        std::vector<float> qd(ggml_nelements(q));
        std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
        for (auto & v : qd) v = ud(gen);
        ggml_backend_tensor_set(q, qd.data(), 0, ggml_nbytes(q));
        ggml_backend_tensor_set(kc, kb.data(), 0, kb.size());
        ggml_backend_tensor_set(vc, vb.data(), 0, vb.size());
        std::vector<uint8_t> pz(ggml_nbytes(pl), 0);
        ggml_backend_tensor_set(pl, pz.data(), 0, pz.size());
        std::vector<int32_t> gd(n_cells / 64, -1);
        ggml_backend_tensor_set(gt, gd.data(), 0, ggml_nbytes(gt));
        std::vector<ggml_fp16_t> md((size_t) (n_cells * nq), ggml_fp32_to_fp16(-INFINITY));
        for (int64_t i = 0; i < nq; ++i) {
            md[(size_t) (i * n_cells + i)] = ggml_fp32_to_fp16(0.0f);   // query i sees only cell i
        }
        ggml_backend_tensor_set(m, md.data(), 0, ggml_nbytes(m));

        const ggml_status st = ggml_backend_graph_compute(gpu, gf);
        TBCHECK(st == GGML_STATUS_SUCCESS, "%s: FA compute failed", name);
        std::vector<float> od(ggml_nelements(out));
        if (st == GGML_STATUS_SUCCESS) {
            ggml_backend_tensor_get(out, od.data(), 0, ggml_nbytes(out));
        }

        uint64_t n = 0, bad = 0, match_i8 = 0, match_float = 0;
        double worst = 0.0;
        for (int64_t i = 0; st == GGML_STATUS_SUCCESS && i < nq; ++i) {
            const uint8_t * row = vb.data() + (size_t) i * l.v.base_row_bytes;
            for (int qh = 0; qh < 24; ++qh) {
                const int h = qh / 6;
                const int b = l.v.b[h];
                for (int e = 0; e < 256; ++e) {
                    const int   j    = (int) ggml_turbot_get_code(row + l.v.base_off[h], b, e);
                    const float gain = ggml_turbot_read_gain(row + l.v.base_gain_off, 2 * h + e / 128);
                    const double vf  = (double) ggml_turbot_old_level(b, j) * gain;
                    const double vi  = b <= 5 ? (double) ggml_turbot_old_level_i8(b, j) * ggml_turbot_old_i8_scale(b) * gain : vf;
                    const double want = expect_i8 ? vi : vf;
                    const double got = od[(size_t) ((i * 24 + qh) * 256 + e)];
                    const double tol = 1.5 * f16_step(want) + 1e-7 * std::fabs(want);
                    ++n;
                    bad         += !(std::fabs(got - want) <= tol);   // a NaN read is bad, not silently skipped
                    match_i8    += std::fabs(got - vi) <= 1.5 * f16_step(vi) + 1e-7 * std::fabs(vi);
                    match_float += std::fabs(got - vf) <= 1.5 * f16_step(vf) + 1e-7 * std::fabs(vf);
                    worst = std::max(worst, std::fabs(got - want) / std::max(tol, 1e-30));
                }
            }
        }
        TBCHECK(bad == 0, "%s: %" PRIu64 "/%" PRIu64 " V elements beyond one half step of the expected old read (worst %.2f tolerances)", name, bad, n, worst);
        printf("  %-18s %" PRIu64 " elements: match int8 read %" PRIu64 ", match float read %" PRIu64 "\n", name, n, match_i8, match_float);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
}

// [TAG_TURBOT_ANY_TEST] The same old-tier read at the other KV geometries: D = 128 heads (head z = run z >> 1 at element
// base 128*(z & 1), so both halves of every run and their two gains are checked), 2 x 256 and 1 x 256 rows, and GQA 7
// (padded head columns). Expected values are computed from the run and element, independently of the kernel's eb path.
static void test_old_read_geom(ggml_backend_t gpu, bool expect_i8) {
    printf("[2b] old-tier FA read at other KV geometries: V value per element vs %s levels\n", expect_i8 ? "int8 register (b <= 5)" : "float");
    struct geom_read { const char * name; int64_t d, hkv, gqa; int flags; std::vector<std::array<uint8_t, 4>> vsets; };
    const std::vector<geom_read> geoms = {
        { "d128x8", 128, 8, 4, GGML_TURBOT_GEOM_D128 | 0, { { 2, 3, 4, 5 }, { 6, 6, 6, 6 }, { 6, 2, 5, 3 } } },
        { "d128x4", 128, 4, 7, GGML_TURBOT_GEOM_D128 | 1, { { 5, 2, 0, 0 }, { 6, 4, 0, 0 } } },
        { "d128x2", 128, 2, 8, GGML_TURBOT_GEOM_D128 | 2, { { 3, 0, 0, 0 }, { 4, 0, 0, 0 }, { 6, 0, 0, 0 } } },
        { "d256x2", 256, 2, 8, 1,                         { { 2, 3, 0, 0 }, { 4, 5, 0, 0 }, { 6, 2, 0, 0 } } },
        { "d256x1", 256, 1, 8, 2,                         { { 2, 0, 0, 0 }, { 5, 0, 0, 0 }, { 6, 0, 0, 0 } } },
    };
    const int64_t n_cells = g_quick ? 64 : 256;
    const int64_t nq      = n_cells;
    std::mt19937 gen(2027);

    for (const geom_read & gr : geoms) {
        const int     nr = ggml_turbot_geom_nr(gr.flags);
        const int64_t hq = gr.hkv*gr.gqa;
        const int64_t D  = gr.d;
        for (const auto & vset : gr.vsets) {
            uint8_t bk[4] = { 0, 0, 0, 0 }, yk[4] = { 0, 0, 0, 0 }, yv[4] = { 0, 0, 0, 0 };
            for (int r = 0; r < nr; ++r) {
                bk[r] = 4;
                yk[r] = 8;
                yv[r] = 8;
            }
            ggml_turbot_layer l;
            if (!make_layer_geom(l, bk, vset.data(), yk, yv, gr.flags)) {
                continue;
            }
            std::vector<uint8_t> kb((size_t) n_cells * l.k.base_row_bytes), vb((size_t) n_cells * l.v.base_row_bytes);
            std::vector<float> x(1024);
            for (int64_t c = 0; c < n_cells; ++c) {
                const int64_t ri = (c % 8 == 1 || c % 8 == 4) ? 0 : c;   // as test_old_read: no 1e-6 / 1e4 rows
                make_row(gen, ri, x.data());
                ggml_turbot_encode_side(x.data(), &l.k, kb.data() + (size_t) c * l.k.base_row_bytes, nullptr);
                make_row(gen, ri, x.data());
                ggml_turbot_encode_side(x.data(), &l.v, vb.data() + (size_t) c * l.v.base_row_bytes, nullptr);
            }

            ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * q   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, nq, hq, 1);
            ggml_tensor * kc  = ggml_new_tensor_2d(ctx, ggml_turbot_type_of_s(l.k.s), 1024, n_cells);
            ggml_tensor * vc  = ggml_new_tensor_2d(ctx, ggml_turbot_type_of_s(l.v.s), 1024, n_cells);
            ggml_tensor * pl  = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, l.pool_row_bytes, GGML_TURBOT_POOL_MIN_ROWS);
            ggml_tensor * gt  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_cells / 64);
            ggml_tensor * m   = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_cells, nq, 1, 1);
            auto view = [&](ggml_tensor * cache) {
                ggml_tensor * t = ggml_view_4d(ctx, cache, D, gr.hkv, n_cells, 1, ggml_row_size(cache->type, D),
                                               ggml_row_size(cache->type, 1024), ggml_row_size(cache->type, 1024 * n_cells), 0);
                return ggml_permute(ctx, t, 0, 2, 1, 3);
            };
            ggml_tensor * out = ggml_flash_attn_ext(ctx, q, view(kc), view(vc), m, 1.0f / sqrtf((float) D), 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
            ggml_turbot_op_params p;
            ggml_turbot_op_params_make(&p, &l, GGML_TURBOT_SIDE_BOTH);
            ggml_flash_attn_ext_set_turbot(out, pl, gt, &p);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, out);

            char name[64];
            snprintf(name, sizeof(name), "%s V b %d %d %d %d", gr.name, vset[0], vset[1], vset[2], vset[3]);
            if (!ggml_backend_supports_op(gpu, out)) {
                TBCHECK(false, "%s: CUDA does not support the turbot FA", name);
                ggml_free(ctx);
                continue;
            }
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
            std::vector<float> qd(ggml_nelements(q));
            std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
            for (auto & v : qd) v = ud(gen);
            ggml_backend_tensor_set(q, qd.data(), 0, ggml_nbytes(q));
            ggml_backend_tensor_set(kc, kb.data(), 0, kb.size());
            ggml_backend_tensor_set(vc, vb.data(), 0, vb.size());
            std::vector<uint8_t> pz(ggml_nbytes(pl), 0);
            ggml_backend_tensor_set(pl, pz.data(), 0, pz.size());
            std::vector<int32_t> gd(n_cells / 64, -1);
            ggml_backend_tensor_set(gt, gd.data(), 0, ggml_nbytes(gt));
            std::vector<ggml_fp16_t> md((size_t) (n_cells * nq), ggml_fp32_to_fp16(-INFINITY));
            for (int64_t i = 0; i < nq; ++i) {
                md[(size_t) (i * n_cells + i)] = ggml_fp32_to_fp16(0.0f);   // query i sees only cell i
            }
            ggml_backend_tensor_set(m, md.data(), 0, ggml_nbytes(m));

            const ggml_status st = ggml_backend_graph_compute(gpu, gf);
            TBCHECK(st == GGML_STATUS_SUCCESS, "%s: FA compute failed", name);
            std::vector<float> od(ggml_nelements(out));
            if (st == GGML_STATUS_SUCCESS) {
                ggml_backend_tensor_get(out, od.data(), 0, ggml_nbytes(out));
            }

            uint64_t n = 0, bad = 0;
            double worst = 0.0;
            for (int64_t i = 0; st == GGML_STATUS_SUCCESS && i < nq; ++i) {
                const uint8_t * row = vb.data() + (size_t) i * l.v.base_row_bytes;
                for (int64_t qh = 0; qh < hq; ++qh) {
                    const int64_t h  = qh / gr.gqa;                   // KV head
                    const int     r  = (int) (D == 256 ? h : h >> 1);  // its run
                    const int     eb = (int) (D == 256 ? 0 : 128*(h & 1));
                    const int     b  = l.v.b[r];
                    for (int64_t e = 0; e < D; ++e) {
                        const int    ee   = eb + (int) e;
                        const int    j    = (int) ggml_turbot_get_code(row + l.v.base_off[r], b, ee);
                        const float  gain = ggml_turbot_read_gain(row + l.v.base_gain_off, 2 * r + ee / 128);
                        const double vf   = (double) ggml_turbot_old_level(b, j) * gain;
                        const double vi   = b <= 5 ? (double) ggml_turbot_old_level_i8(b, j) * ggml_turbot_old_i8_scale(b) * gain : vf;
                        const double want = expect_i8 ? vi : vf;
                        const double got  = od[(size_t) ((i * hq + qh) * D + e)];
                        const double tol  = 1.5 * f16_step(want) + 1e-7 * std::fabs(want);
                        ++n;
                        bad  += !(std::fabs(got - want) <= tol);
                        worst = std::max(worst, std::fabs(got - want) / std::max(tol, 1e-30));
                    }
                }
            }
            TBCHECK(bad == 0, "%s: %" PRIu64 "/%" PRIu64 " V elements beyond one half step of the expected old read (worst %.2f tolerances)", name, bad, n, worst);
            printf("  %-28s %" PRIu64 " elements, %" PRIu64 " off\n", name, n, bad);
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
        }
    }
}

int main(int argc, char ** argv) {
    bool expect_i8 = true;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--quick") == 0) {
            g_quick = true;
        } else if (std::strcmp(argv[i], "--old-read") == 0 && i + 1 < argc) {
            const char * v = argv[++i];
            if (std::strcmp(v, "i8") == 0) {
                expect_i8 = true;
            } else if (std::strcmp(v, "float") == 0) {
                expect_i8 = false;
            } else {
                fprintf(stderr, "--old-read takes i8 or float\n");
                return 2;
            }
        } else {
            fprintf(stderr, "usage: %s [--quick] [--old-read i8|float]\n", argv[0]);
            return 2;
        }
    }

    ggml_backend_dev_t gdev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!gdev) {
        printf("SKIP: no GPU backend device\n");
        return 0;
    }
    ggml_backend_t gpu = ggml_backend_dev_init(gdev, nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!gpu || !cpu) {
        fprintf(stderr, "backend init failed\n");
        return 1;
    }
    printf("GPU backend: %s (%s)\n", ggml_backend_name(gpu), ggml_backend_dev_description(gdev));

    test_writer(cpu, gpu);
    test_old_read(gpu, expect_i8);
    test_old_read_geom(gpu, expect_i8);   // [TAG_TURBOT_ANY_TEST]

    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    printf("\n%s (%d failures)\n", g_fail == 0 ? "OK" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
