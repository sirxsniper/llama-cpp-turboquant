// [TAG_TURBOT] CPU unit tests of the turbot tiered KV cache format, docs/turbot/SPEC.md section 11.1.
//
// Items 1-6 (tables, nesting, planes, coder, fill, op params and layout) need only ggml and ggml-turbot.h. Item 7
// (plan parser and llama_kv_tier scenarios) drives src/llama-kv-tier.h, whose class and parser are exported with
// LLAMA_API, against a real llama_kv_cells (header-only); tests/CMakeLists.txt defines TURBOT_TEST_TIER for it.
//
// No GPU, no model. Exit code 0 iff every check passes.
//
//   test-turbot [--quick]      --quick: 10^5 nesting values and 4,000 Monte Carlo groups instead of 10^6 and 16,000

#include "ggml.h"
#include "ggml-turbot.h"

#ifdef TURBOT_TEST_TIER
#    include "llama.h"
#    include "../src/llama-batch.h"
#    include "../src/llama-ext.h"   // [TAG_TURBOT_EMBED_PLAN] llama_turbot_set_plan_path
#    include "../src/llama-kv-cells.h"
#    include "../src/llama-kv-tier.h"
#endif

#include <algorithm>
#include <cfloat>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

static int  g_fail   = 0;
static int  g_checks = 0;
static bool g_quick  = false;

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
    return memcmp(&a, &b, sizeof(float)) == 0;
}

// float32 unit in the last place at |x|
static double ulp32(float x) {
    const float ax = std::fabs(x);
    return (double) (std::nextafter(ax, FLT_MAX) - ax);
}

// ---------------------------------------------------------------------------------------------------------------
// references copied from their sources (keep in step with them)
// ---------------------------------------------------------------------------------------------------------------

// src/llama-kvfq.cpp (research build, the fake-quant coder behind every quality number): signs, 4/5-bit literals
static const float KVFQ_S1[128] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1 };
static const float KVFQ_S2[128] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1 };
static const float KVFQ_C5N[16] = { -0.271948f, -0.222223f, -0.189260f, -0.163683f, -0.142366f, -0.123814f, -0.107237f, -0.092232f,
                                    -0.078519f, -0.065814f, -0.053979f, -0.042868f, -0.032338f, -0.022390f, -0.013026f, -0.004245f };
static const float KVFQ_C4N[8]  = { -0.241530f, -0.182875f, -0.143021f, -0.111033f, -0.083297f, -0.058053f, -0.034304f, -0.011349f };
static const float KVFQ_INV_SQRT_128 = 0.08838834764831845f;

// ggml/src/ggml-cuda/turbo-quant.cuh TURBO_C4_I8_LIST / TURBO_C5_I8_LIST and the two scale literals
static const int8_t TURBO_C4_I8[16] = { -127, -96, -75, -58, -44, -31, -18,  -6,
                                           6,  18,  31,  44,  58,  75,  96, 127 };
static const int8_t TURBO_C5_I8[32] = { -127, -104,  -88,  -76,  -66,  -58,  -50,  -43,
                                         -37,  -31,  -25,  -20,  -15,  -10,   -6,   -2,
                                           2,    6,   10,   15,   20,   25,   31,   37,
                                          43,   50,   58,   66,   76,   88,  104,  127 };
static const float TURBO_C4_SCALE = 0.241530f / 127.0f;
static const float TURBO_C5_SCALE = 0.271948f / 127.0f;

// llama-kvfq.cpp lloyd_max_gauss, step for step
static std::vector<double> kvfq_lloyd_max_gauss(int bits) {
    const int L = 1 << bits;
    auto Phi = [](double x) { return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0))); };
    auto phi = [](double x) { return std::exp(-0.5 * x * x) / std::sqrt(2.0 * 3.14159265358979323846); };
    std::vector<double> c(L), cn(L);
    for (int i = 0; i < L; ++i) {
        const double target = (i + 0.5) / L;
        double lo = -12.0, hi = 12.0;
        for (int it = 0; it < 100; ++it) {
            const double m = 0.5 * (lo + hi);
            (Phi(m) < target ? lo : hi) = m;
        }
        c[i] = 0.5 * (lo + hi);
    }
    for (int it = 0; it < 20000; ++it) {
        double delta = 0.0;
        for (int i = 0; i < L; ++i) {
            const bool   has_lo = i > 0, has_hi = i < L - 1;
            const double tlo = has_lo ? 0.5 * (c[i - 1] + c[i]) : 0.0;
            const double thi = has_hi ? 0.5 * (c[i] + c[i + 1]) : 0.0;
            const double pl  = has_lo ? phi(tlo) : 0.0;
            const double ph  = has_hi ? phi(thi) : 0.0;
            const double Pl  = has_lo ? Phi(tlo) : 0.0;
            const double Ph  = has_hi ? Phi(thi) : 1.0;
            cn[i] = (pl - ph) / std::max(Ph - Pl, 1e-300);
            delta = std::max(delta, std::fabs(cn[i] - c[i]));
        }
        c.swap(cn);
        if (delta < 1e-12) {
            break;
        }
    }
    return c;
}

struct kvfq_levels {
    std::vector<float> c;
    std::vector<float> mids;
};

// llama-kvfq.cpp get_levels
static const kvfq_levels & kvfq_get_levels(int bits) {
    static std::map<int, kvfq_levels> cache;
    auto it = cache.find(bits);
    if (it != cache.end()) {
        return it->second;
    }
    kvfq_levels lv;
    if (bits == 5) {
        for (int i = 0; i < 16; ++i) lv.c.push_back(KVFQ_C5N[i]);
        for (int i = 15; i >= 0; --i) lv.c.push_back(-KVFQ_C5N[i]);
    } else if (bits == 4) {
        for (int i = 0; i < 8; ++i) lv.c.push_back(KVFQ_C4N[i]);
        for (int i = 7; i >= 0; --i) lv.c.push_back(-KVFQ_C4N[i]);
    } else {
        for (double v : kvfq_lloyd_max_gauss(bits)) lv.c.push_back((float) (v / std::sqrt(128.0)));
    }
    for (size_t i = 0; i + 1 < lv.c.size(); ++i) {
        lv.mids.push_back(0.5f * (lv.c[i] + lv.c[i + 1]));
    }
    return cache.emplace(bits, std::move(lv)).first->second;
}

static void kvfq_butterfly(float * x) {
    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += 2 * h) {
            for (int j = i; j < i + h; ++j) {
                const float a = x[j], b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
}

// llama-kvfq.cpp fq_group (quantize-dequantize one group in place), plus the index of every element
static void kvfq_fq_group(float * x, int bits, int * idx_out) {
    const kvfq_levels & lv = kvfq_get_levels(bits);
    float norm_sq = 0.0f;
    for (int i = 0; i < 128; ++i) norm_sq += x[i] * x[i];
    const float norm = std::sqrt(norm_sq);
    float u[128];
    if (norm > 1e-10f) {
        const float inv = 1.0f / norm;
        for (int i = 0; i < 128; ++i) u[i] = x[i] * inv;
    } else {
        std::memset(u, 0, sizeof(u));
    }
    for (int i = 0; i < 128; ++i) u[i] *= KVFQ_S1[i];
    kvfq_butterfly(u);
    for (int i = 0; i < 128; ++i) u[i] *= KVFQ_INV_SQRT_128 * KVFQ_S2[i];

    float recon_sq = 0.0f;
    for (int i = 0; i < 128; ++i) {
        const size_t idx = std::upper_bound(lv.mids.begin(), lv.mids.end(), u[i]) - lv.mids.begin();
        idx_out[i] = (int) idx;
        u[i] = lv.c[idx];
        recon_sq += u[i] * u[i];
    }
    const float recon = std::sqrt(recon_sq);
    const float g     = ggml_fp16_to_fp32(ggml_fp32_to_fp16(recon > 1e-10f ? norm / recon : norm));
    for (int i = 0; i < 128; ++i) u[i] *= g * KVFQ_S2[i];
    kvfq_butterfly(u);
    for (int i = 0; i < 128; ++i) x[i] = u[i] * KVFQ_INV_SQRT_128 * KVFQ_S1[i];
}

// kv_nested_study.py / gen_turbot_tables.py a_lloyd in float64: each base Lloyd cell refined by a 2^r level
// Lloyd-Max of the standard Gaussian truncated to the cell (z units), outer cells initialised over 2 * (t1 - t0)
static double tt_Phi(double x) {
    if (std::isinf(x)) {
        return x < 0 ? 0.0 : 1.0;
    }
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

static double tt_phi(double x) {
    if (std::isinf(x)) {
        return 0.0;
    }
    return std::exp(-0.5 * std::min(x * x, 1e6)) / std::sqrt(2.0 * 3.141592653589793);
}

static void tt_a_lloyd(int b, int y, std::vector<float> & lut, std::vector<float> & thr, int & iters, double & dlt) {
    const int    r   = y - b;
    const int    L   = 1 << b;
    const int    sub = 1 << r;
    const double sq  = std::sqrt(128.0);
    const double inf = std::numeric_limits<double>::infinity();
    const float * mids = ggml_turbot_old_thr + ggml_turbot_old_off(b);

    std::vector<double> t(L - 1);
    for (int i = 0; i < L - 1; ++i) {
        t[i] = (double) mids[i] * sq;
    }
    const double wadj = t[1] - t[0];
    std::vector<double> e_lo(L), e_hi(L), lev(L * sub), nw(L * sub);
    for (int j = 0; j < L; ++j) {
        e_lo[j] = j == 0     ? -inf : t[j - 1];
        e_hi[j] = j == L - 1 ?  inf : t[j];
        const double lo = std::isfinite(e_lo[j]) ? e_lo[j] : e_hi[j] - 2.0 * wadj;
        const double hi = std::isfinite(e_hi[j]) ? e_hi[j] : e_lo[j] + 2.0 * wadj;
        for (int s = 0; s < sub; ++s) {
            lev[j * sub + s] = lo + ((hi - lo) * (s + 0.5)) / sub;
        }
    }
    std::vector<double> E(sub + 1);
    iters = 0;
    dlt   = inf;
    for (int it = 1; it <= 20000; ++it) {
        iters = it;
        double d = 0.0;
        for (int j = 0; j < L; ++j) {
            E[0]   = e_lo[j];
            E[sub] = e_hi[j];
            for (int s = 0; s + 1 < sub; ++s) {
                E[s + 1] = 0.5 * (lev[j * sub + s] + lev[j * sub + s + 1]);
            }
            for (int s = 0; s < sub; ++s) {
                const double lo = E[s], hi = E[s + 1];
                const double m0 = tt_Phi(hi) - tt_Phi(lo);
                const double m1 = -(tt_phi(hi) - tt_phi(lo));
                const double mid = 0.5 * (std::min(std::max(lo, -12.0), 12.0) + std::min(std::max(hi, -12.0), 12.0));
                nw[j * sub + s] = m0 > 1e-300 ? m1 / std::max(m0, 1e-300) : mid;
                d = std::max(d, std::fabs(nw[j * sub + s] - lev[j * sub + s]));
            }
        }
        lev.swap(nw);
        dlt = d;
        if (d < 1e-11) {
            break;
        }
    }
    lut.assign(L * sub, 0.0f);
    thr.assign(L * sub - 1, 0.0f);
    int k = 0;
    for (int j = 0; j < L; ++j) {
        for (int s = 0; s < sub; ++s) {
            lut[j * sub + s] = (float) (lev[j * sub + s] / sq);
        }
        for (int s = 0; s + 1 < sub; ++s) {
            thr[k++] = (float) ((0.5 * (lev[j * sub + s] + lev[j * sub + s + 1])) / sq);
        }
        if (j < L - 1) {
            thr[k++] = mids[j];   // bit-exact nesting: the float32 old threshold verbatim
        }
    }
}

// analytic Gaussian distortion (z units) of a scalar quantizer: levels C (r units), thresholds T (r units)
static double tt_analytic_D(const float * C, const float * T, int L) {
    const double sq  = std::sqrt(128.0);
    const double inf = std::numeric_limits<double>::infinity();
    double D = 0.0;
    for (int i = 0; i < L; ++i) {
        const double lo = i == 0     ? -inf : (double) T[i - 1] * sq;
        const double hi = i == L - 1 ?  inf : (double) T[i] * sq;
        const double c  = (double) C[i] * sq;
        const double P  = tt_Phi(hi) - tt_Phi(lo);
        const double m1 = tt_phi(lo) - tt_phi(hi);
        const double m2 = P - (std::isinf(hi) ? 0.0 : hi * tt_phi(hi)) + (std::isinf(lo) ? 0.0 : lo * tt_phi(lo));
        D += m2 - 2.0 * c * m1 + c * c * P;
    }
    return D;
}

static int tt_upper_count(const float * thr, int n, float x) {
    return (int) (std::upper_bound(thr, thr + n, x) - thr);
}

static float tt_f16(float x) {
    return ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));
}

// ---------------------------------------------------------------------------------------------------------------
// 1. tables
// ---------------------------------------------------------------------------------------------------------------

static void test_tables() {
    printf("[1] tables\n");

    for (int i = 0; i < 128; ++i) {
        TCHECK(ggml_turbot_wht_s1[i] == KVFQ_S1[i] && ggml_turbot_wht_s2[i] == KVFQ_S2[i], "WHT sign %d differs from llama-kvfq.cpp", i);
    }

    int old_run = 0;
    for (int b = GGML_TURBOT_B_MIN; b <= GGML_TURBOT_B_MAX; ++b) {
        const int L   = 1 << b;
        const int off = ggml_turbot_old_off(b);
        TCHECK(off == old_run, "old run of b%d starts at %d, expected %d", b, off, old_run);
        old_run += L;

        const float * C = ggml_turbot_old_levels + off;
        const float * T = ggml_turbot_old_thr + off;
        const kvfq_levels & lv = kvfq_get_levels(b);
        for (int j = 0; j < L; ++j) {
            TCHECK(same_bits(C[j], lv.c[j]), "b%d level %d: header %.9g, kvfq get_levels %.9g", b, j, C[j], lv.c[j]);
            TCHECK(same_bits(C[j], -C[L - 1 - j]), "b%d level %d not antisymmetric", b, j);
            if (j + 1 < L) {
                TCHECK(C[j] < C[j + 1], "b%d levels not ascending at %d", b, j);
                TCHECK(same_bits(T[j], 0.5f * (C[j] + C[j + 1])), "b%d threshold %d is not the float32 midpoint", b, j);
                TCHECK(same_bits(T[j], lv.mids[j]), "b%d threshold %d differs from kvfq mids", b, j);
            }
        }
        TCHECK(T[L / 2 - 1] == 0.0f, "b%d middle threshold is not 0", b);
        TCHECK(T[L - 1] == 0.0f, "b%d threshold padding is not 0", b);
    }
    TCHECK(old_run == GGML_TURBOT_OLD_TOTAL, "old tables hold %d entries, expected %d", GGML_TURBOT_OLD_TOTAL, old_run);

    int young_run = 0, fill_run = 0;
    int n_exact = 0, n_total = 0;
    double worst_rel = 0.0;
    for (int b = GGML_TURBOT_B_MIN; b <= GGML_TURBOT_B_MAX; ++b) {
        const int L = 1 << b;
        const float * C = ggml_turbot_old_levels + ggml_turbot_old_off(b);
        const float * T = ggml_turbot_old_thr    + ggml_turbot_old_off(b);
        for (int y = b + 1; y <= GGML_TURBOT_Y_MAX; ++y) {
            const int r = y - b, n = 1 << y;
            const int off  = ggml_turbot_young_off(b, y);
            const int foff = ggml_turbot_fill_off(b, y);
            TCHECK(off == ggml_turbot_young_off_tab[b * 9 + y], "young_off(b%d, y%d) %d != generated %d", b, y, off, ggml_turbot_young_off_tab[b * 9 + y]);
            TCHECK(foff == ggml_turbot_fill_off_tab[b * 9 + y], "fill_off(b%d, y%d) %d != generated %d", b, y, foff, ggml_turbot_fill_off_tab[b * 9 + y]);
            TCHECK(off == young_run, "young run (b%d, y%d) starts at %d, expected %d", b, y, off, young_run);
            TCHECK(foff == fill_run, "fill run (b%d, y%d) starts at %d, expected %d", b, y, foff, fill_run);
            young_run += n;
            fill_run  += L;

            const float * LUT = ggml_turbot_young_lut + off;
            const float * TY  = ggml_turbot_young_thr + off;
            for (int i = 0; i + 2 < n; ++i) {
                TCHECK(TY[i] <= TY[i + 1], "b%d y%d young thresholds not sorted at %d", b, y, i);
            }
            TCHECK(TY[n - 1] == 0.0f, "b%d y%d young threshold padding is not 0", b, y);
            for (int j = 0; j + 1 < L; ++j) {
                TCHECK(same_bits(TY[((j + 1) << r) - 1], T[j]), "b%d y%d old threshold %d not embedded bit for bit", b, y, j);
            }
            for (int k = 0; k < n; ++k) {
                if (k + 1 < n) {
                    TCHECK(LUT[k] < LUT[k + 1], "b%d y%d young LUT not ascending at %d", b, y, k);
                }
                TCHECK(ggml_turbot_count_ge(TY, n - 1, LUT[k]) == k, "b%d y%d young level %d lies outside its cell", b, y, k);
            }
            for (int j = 0; j < L; ++j) {
                int    best = 0;
                double bd   = std::numeric_limits<double>::infinity();
                for (int s = 0; s < (1 << r); ++s) {
                    const double d = std::fabs((double) LUT[(j << r) | s] - (double) C[j]);
                    if (d < bd) {
                        bd   = d;
                        best = s;
                    }
                }
                TCHECK(ggml_turbot_fill_code[foff + j] == best, "fill code (b%d, y%d, j%d) = %d, nearest is %d", b, y, j, ggml_turbot_fill_code[foff + j], best);
            }

            // the generator's a_lloyd, recomputed here in float64
            std::vector<float> lut2, thr2;
            int    iters = 0;
            double dlt   = 0.0;
            tt_a_lloyd(b, y, lut2, thr2, iters, dlt);
            TCHECK(dlt < 1e-9, "b%d y%d a_lloyd did not converge (%d iterations, change %.3g)", b, y, iters, dlt);
            for (int k = 0; k < n; ++k) {
                const double tol = 4.0 * ulp32(LUT[k]) + 1e-12;
                const double d   = std::fabs((double) lut2[k] - (double) LUT[k]);
                n_exact += same_bits(lut2[k], LUT[k]);
                n_total += 1;
                worst_rel = std::max(worst_rel, d / ulp32(LUT[k]));
                TCHECK(d <= tol, "b%d y%d young LUT %d: header %.9g, recomputed %.9g", b, y, k, LUT[k], lut2[k]);
                if (k + 1 < n) {
                    const double tolt = 4.0 * ulp32(TY[k]) + 1e-12;
                    TCHECK(std::fabs((double) thr2[k] - (double) TY[k]) <= tolt, "b%d y%d young threshold %d: header %.9g, recomputed %.9g", b, y, k, TY[k], thr2[k]);
                }
            }
        }
    }
    TCHECK(young_run == GGML_TURBOT_YOUNG_TOTAL, "young tables hold %d, runs sum to %d", GGML_TURBOT_YOUNG_TOTAL, young_run);
    TCHECK(fill_run  == GGML_TURBOT_FILL_TOTAL,  "fill table holds %d, runs sum to %d",  GGML_TURBOT_FILL_TOTAL,  fill_run);
    printf("    a_lloyd recomputed: %d/%d young LUT entries bit-identical, worst difference %.1f float32 ULP\n", n_exact, n_total, worst_rel);

    for (int b = 0; b <= GGML_TURBOT_B_MAX; ++b) {
        for (int y = 0; y <= GGML_TURBOT_Y_MAX; ++y) {
            const bool legal = b >= GGML_TURBOT_B_MIN && y > b && y <= GGML_TURBOT_Y_MAX;
            if (!legal) {
                TCHECK(ggml_turbot_young_off_tab[b * 9 + y] == -1 && ggml_turbot_fill_off_tab[b * 9 + y] == -1,
                       "offset tables not -1 for illegal (b%d, y%d)", b, y);
            }
        }
    }

    // int8 register LUTs that a CUDA OLD reader may reuse for b = 4, 5 (SPEC 7.4, [TAG_TURBOT_I8])
    for (int k = 0; k < 16; ++k) {
        TCHECK(ggml_turbot_old_level_i8(4, k) == TURBO_C4_I8[k], "old_level_i8(4, %d) = %d, turbo4_int8_lut %d", k, ggml_turbot_old_level_i8(4, k), TURBO_C4_I8[k]);
    }
    for (int k = 0; k < 32; ++k) {
        TCHECK(ggml_turbot_old_level_i8(5, k) == TURBO_C5_I8[k], "old_level_i8(5, %d) = %d, turbo5_int8_lut %d", k, ggml_turbot_old_level_i8(5, k), TURBO_C5_I8[k]);
    }
    TCHECK(same_bits(ggml_turbot_old_i8_scale(4), TURBO_C4_SCALE), "old_i8_scale(4) %.9g != TURBO_INT8_4BIT_SCALE_REVERSE %.9g", ggml_turbot_old_i8_scale(4), TURBO_C4_SCALE);
    TCHECK(same_bits(ggml_turbot_old_i8_scale(5), TURBO_C5_SCALE), "old_i8_scale(5) %.9g != TURBO_INT8_5BIT_SCALE_REVERSE %.9g", ggml_turbot_old_i8_scale(5), TURBO_C5_SCALE);
    for (int b = 2; b <= 5; ++b) {
        for (int k = 0; k < (1 << b); ++k) {
            const int8_t q = ggml_turbot_old_level_i8(b, k);
            TCHECK(q == -ggml_turbot_old_level_i8(b, (1 << b) - 1 - k), "old_level_i8(%d, %d) not antisymmetric", b, k);
            TCHECK(k == 0 || q >= ggml_turbot_old_level_i8(b, k - 1), "old_level_i8(%d) not monotone at %d", b, k);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// 2. nesting
// ---------------------------------------------------------------------------------------------------------------

static void test_nesting() {
    printf("[2] nesting\n");
    std::vector<float> xs;
    const size_t n_rand = g_quick ? 100000 : 1000000;
    xs.reserve(n_rand + 12000);
    std::mt19937 gen(20260915);
    std::normal_distribution<double> nd(0.0, 1.0 / std::sqrt(128.0));
    for (size_t i = 0; i < n_rand; ++i) {
        xs.push_back((float) nd(gen));
    }
    auto add_edge = [&](float t) {
        xs.push_back(t);
        xs.push_back(std::nextafter(t, -FLT_MAX));
        xs.push_back(std::nextafter(t,  FLT_MAX));
    };
    for (int i = 0; i < GGML_TURBOT_OLD_TOTAL; ++i) {
        add_edge(ggml_turbot_old_thr[i]);
        add_edge(ggml_turbot_old_levels[i]);
    }
    for (int i = 0; i < GGML_TURBOT_YOUNG_TOTAL; ++i) {
        add_edge(ggml_turbot_young_thr[i]);
        add_edge(ggml_turbot_young_lut[i]);
    }
    for (float v : { 0.0f, -0.0f, 1.0f, -1.0f, FLT_MAX, -FLT_MAX, FLT_MIN, -FLT_MIN, 1e-45f, -1e-45f,
                     std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity() }) {
        xs.push_back(v);
    }
    xs.push_back(std::numeric_limits<float>::quiet_NaN());

    for (int b = GGML_TURBOT_B_MIN; b <= GGML_TURBOT_B_MAX; ++b) {
        for (int y = b + 1; y <= GGML_TURBOT_Y_MAX; ++y) {
            const int r = y - b, mask = (1 << r) - 1;
            size_t bad = 0;
            float  first_bad = 0.0f;
            for (float x : xs) {
                const int j  = ggml_turbot_old_index(b, x);
                const int iy = ggml_turbot_young_index(b, y, x);
                const int s  = ggml_turbot_refine_index(b, y, j, x);
                if ((iy >> r) != j || s != (iy & mask)) {
                    if (bad++ == 0) {
                        first_bad = x;
                    }
                }
            }
            TCHECK(bad == 0, "b%d y%d: %zu of %zu values break young >> r == old or refine == young & mask (first %.9g)", b, y, bad, xs.size(), first_bad);
        }
    }
    const float nan = std::numeric_limits<float>::quiet_NaN();
    TCHECK(ggml_turbot_old_index(5, nan) == 0 && ggml_turbot_young_index(5, 7, nan) == 0, "NaN must index 0");
    TCHECK(ggml_turbot_old_index(3, 0.0f) == 4 && ggml_turbot_old_index(3, -0.0f) == 4, "zero must index 2^(b-1)");
}

// ---------------------------------------------------------------------------------------------------------------
// 3. planes
// ---------------------------------------------------------------------------------------------------------------

static void test_planes() {
    printf("[3] planes\n");
    std::mt19937 gen(7);
    const int expect_p4[7] = { 0, -1, -1, -1, 0, 0, 0 };
    const int expect_p2[7] = { 0, -1, 0, 0, -1, -1, 128 };
    const int expect_p1[7] = { 0, 0, -1, 64, -1, 128, -1 };
    for (int w = 1; w <= 6; ++w) {
        const ggml_turbot_planes p = ggml_turbot_planes_of(w);
        TCHECK(p.p4 == expect_p4[w] && p.p2 == expect_p2[w] && p.p1 == expect_p1[w], "planes_of(%d) = %d %d %d", w, p.p4, p.p2, p.p1);

        const int run = 32 * w, guard = 16;
        std::vector<uint8_t> buf(run + 2 * guard, 0xA5);
        uint8_t * r = buf.data() + guard;
        std::vector<unsigned> codes(256);
        for (int e = 0; e < 256; ++e) {
            codes[e] = gen() & ((1u << w) - 1);
            ggml_turbot_set_code(r, w, e, codes[e]);
        }
        bool ok = true;
        for (int e = 0; e < 256; ++e) {
            ok = ok && ggml_turbot_get_code(r, w, e) == codes[e];
        }
        TCHECK(ok, "w%d: set/get roundtrip of 256 elements failed", w);
        for (int g = 0; g < guard; ++g) {
            TCHECK(buf[g] == 0xA5 && buf[guard + run + g] == 0xA5, "w%d: write outside the %d byte run", w, run);
        }
        for (int t = 0; t < 2000; ++t) {
            const int      e = gen() % 256;
            const unsigned v = gen() & ((1u << w) - 1);
            ggml_turbot_set_code(r, w, e, v);
            codes[e] = v;
            if (t % 97 == 0) {
                bool all = true;
                for (int e2 = 0; e2 < 256; ++e2) {
                    all = all && ggml_turbot_get_code(r, w, e2) == codes[e2];
                }
                TCHECK(all, "w%d: overwriting element %d disturbed a neighbour", w, e);
            }
        }
        for (int g = 0; g < guard; ++g) {
            TCHECK(buf[g] == 0xA5 && buf[guard + run + g] == 0xA5, "w%d: overwrite outside the run", w);
        }
        // every byte of the run is covered: setting all codes to all-ones must give all-ones bytes
        std::vector<uint8_t> ones(run, 0);
        for (int e = 0; e < 256; ++e) {
            ggml_turbot_set_code(ones.data(), w, e, (1u << w) - 1);
        }
        TCHECK(std::all_of(ones.begin(), ones.end(), [](uint8_t v) { return v == 0xFF; }), "w%d: a run byte is not covered by any element", w);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// 4. coder
// ---------------------------------------------------------------------------------------------------------------

static void make_side(ggml_turbot_side & sd, const uint8_t b[4], const uint8_t y[4]) {
    const bool ok = ggml_turbot_side_init(&sd, b, y);
    TCHECK(ok, "side_init(%d %d %d %d / %d %d %d %d)", b[0], b[1], b[2], b[3], y[0], y[1], y[2], y[3]);
}

// assorted 1024-value rows: gaussian, uniform, outlier channels, tiny and huge scales, one-hot, zero groups
static void test_row(std::mt19937 & gen, int kind, float * x) {
    std::normal_distribution<float>       nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
    const float scales[6] = { 1.0f, 1e-3f, 30.0f, 1e3f, 0.05f, 4.0f };
    const float sc = scales[kind % 6];
    for (int i = 0; i < 1024; ++i) {
        x[i] = sc * ((kind % 3 == 0) ? nd(gen) : ud(gen));
    }
    if (kind % 4 == 1) {
        for (int i = 0; i < 1024; i += 97) {
            x[i] *= 40.0f;       // outlier channels
        }
    }
    if (kind % 5 == 2) {
        std::memset(x + 256, 0, 128 * sizeof(float));   // head 1 group 0 all zero
    }
    if (kind % 7 == 3) {
        std::memset(x + 768, 0, 256 * sizeof(float));
        x[800] = sc;                                    // one-hot group
    }
}

static void test_coder() {
    printf("[4] coder\n");
    std::mt19937 gen(42);

    // primary gate: the old tier is bit-identical to llama-kvfq.cpp fq_group
    {
        std::vector<float> x(1024), dec(1024), ref(1024);
        std::vector<uint8_t> base(32 * 24 + 16);
        size_t n_groups = 0, bad_vals = 0, bad_codes = 0;
        for (int b = GGML_TURBOT_B_MIN; b <= GGML_TURBOT_B_MAX; ++b) {
            const uint8_t bw[4] = { (uint8_t) b, (uint8_t) b, (uint8_t) b, (uint8_t) b };
            const uint8_t yw[4] = { 8, 8, 8, 8 };
            ggml_turbot_side sd;
            make_side(sd, bw, yw);
            for (int t = 0; t < 400; ++t) {
                test_row(gen, t, x.data());
                ggml_turbot_encode_side(x.data(), &sd, base.data(), nullptr);
                ggml_turbot_decode_side(base.data(), nullptr, &sd, dec.data());
                ref = x;
                for (int h = 0; h < 4; ++h) {
                    for (int g = 0; g < 2; ++g) {
                        const int o = 256 * h + 128 * g;
                        int idx[128];
                        kvfq_fq_group(ref.data() + o, b, idx);
                        ggml_turbot_iwht128(dec.data() + o);
                        for (int e = 0; e < 128; ++e) {
                            bad_vals  += !same_bits(dec[o + e], ref[o + e]);
                            bad_codes += (int) ggml_turbot_get_code(base.data() + sd.base_off[h], b, 128 * g + e) != idx[e];
                        }
                        ++n_groups;
                    }
                }
            }
        }
        TCHECK(bad_vals == 0 && bad_codes == 0, "old tier vs kvfq fq_group over %zu groups: %zu values and %zu codes differ", n_groups, bad_vals, bad_codes);
        printf("    old tier == kvfq fq_group bit for bit over %zu groups (b 2..6)\n", n_groups);
    }

    // Gaussian sphere Monte Carlo: old nmse vs analytic D_b, young nmse vs an independent y-bit Lloyd code
    {
        const int NG = g_quick ? 4000 : 16000;
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> xs(NG * 128), xr(NG * 128), us(NG * 128), norms(NG);
        double sig = 0.0;
        for (int g = 0; g < NG; ++g) {
            float * x = xs.data() + 128 * g;
            float norm_sq = 0.0f;
            for (int i = 0; i < 128; ++i) {
                x[i] = nd(gen);
                norm_sq += x[i] * x[i];
            }
            const float norm = std::sqrt(norm_sq);
            norms[g] = norm;
            for (int i = 0; i < 128; ++i) {
                us[128 * g + i] = x[i] / norm;
            }
            ggml_turbot_fwht128(us.data() + 128 * g);
            for (int i = 0; i < 128; ++i) {
                xr[128 * g + i] = us[128 * g + i] * norm;
                sig += (double) xr[128 * g + i] * xr[128 * g + i];
            }
        }

        // independent y-bit Lloyd references, same f16 gain rule
        std::map<int, double> indep;
        for (int y = 3; y <= GGML_TURBOT_Y_MAX; ++y) {
            std::vector<float> c, m;
            for (double v : kvfq_lloyd_max_gauss(y)) c.push_back((float) (v / std::sqrt(128.0)));
            for (size_t i = 0; i + 1 < c.size(); ++i) m.push_back(0.5f * (c[i] + c[i + 1]));
            double err = 0.0;
            for (int g = 0; g < NG; ++g) {
                float rec_sq = 0.0f;
                int   idx[128];
                for (int i = 0; i < 128; ++i) {
                    idx[i] = tt_upper_count(m.data(), (int) m.size(), us[128 * g + i]);
                    rec_sq += c[idx[i]] * c[idx[i]];
                }
                const float gain = tt_f16(ggml_turbot_gain(norms[g], rec_sq));
                for (int i = 0; i < 128; ++i) {
                    const double d = (double) (c[idx[i]] * gain) - xr[128 * g + i];
                    err += d * d;
                }
            }
            indep[y] = err / sig;
        }

        printf("    b  old nmse/D_b [0.93, 1.03] | young nmse / independent y-bit Lloyd, y = b+1..8 (gate <= 1.18 at y 7, 8)\n");
        for (int b = GGML_TURBOT_B_MIN; b <= GGML_TURBOT_B_MAX; ++b) {
            const int L = 1 << b;
            const float * C = ggml_turbot_old_levels + ggml_turbot_old_off(b);
            const float * T = ggml_turbot_old_thr    + ggml_turbot_old_off(b);
            const double D  = tt_analytic_D(C, T, L);
            std::vector<int> jv(NG * 128);
            double err_old = 0.0;
            for (int g = 0; g < NG; ++g) {
                float rec_sq = 0.0f;
                for (int i = 0; i < 128; ++i) {
                    const int j = tt_upper_count(T, L - 1, us[128 * g + i]);
                    jv[128 * g + i] = j;
                    rec_sq += C[j] * C[j];
                }
                const float gain = tt_f16(ggml_turbot_gain(norms[g], rec_sq));
                for (int i = 0; i < 128; ++i) {
                    const double d = (double) (C[jv[128 * g + i]] * gain) - xr[128 * g + i];
                    err_old += d * d;
                }
            }
            const double ratio_old = err_old / sig / D;
            TCHECK(ratio_old >= 0.93 && ratio_old <= 1.03, "b%d old nmse / D_b = %.4f outside [0.93, 1.03]", b, ratio_old);
            printf("    %d  %.4f   |", b, ratio_old);

            for (int y = b + 1; y <= GGML_TURBOT_Y_MAX; ++y) {
                const int r = y - b;
                const float * LUT = ggml_turbot_young_lut + ggml_turbot_young_off(b, y);
                const float * TY  = ggml_turbot_young_thr + ggml_turbot_young_off(b, y);
                double err_y = 0.0;
                for (int g = 0; g < NG; ++g) {
                    float rec_sq = 0.0f;
                    int   iy[128];
                    for (int i = 0; i < 128; ++i) {
                        iy[i] = tt_upper_count(TY, (1 << y) - 1, us[128 * g + i]);
                        rec_sq += LUT[iy[i]] * LUT[iy[i]];
                    }
                    const float gain = tt_f16(ggml_turbot_gain(norms[g], rec_sq));
                    for (int i = 0; i < 128; ++i) {
                        const double d = (double) (LUT[iy[i]] * gain) - xr[128 * g + i];
                        err_y += d * d;
                    }
                }
                const double ratio = err_y / sig / indep[y];
                if (y >= 7) {
                    TCHECK(ratio <= 1.18, "b%d y%d young nmse / independent %d-bit Lloyd = %.4f > 1.18", b, y, y, ratio);
                }
                printf(" y%d x%.3f", y, ratio);
                (void) r;
            }
            printf("\n");
        }

        // the upper_bound path above equals the header coder (linear counts, stored codes and gains)
        size_t bad = 0;
        const int n_cross = std::min(NG, 1000);
        for (int b = GGML_TURBOT_B_MIN; b <= GGML_TURBOT_B_MAX; ++b) {
            for (int y = b + 1; y <= GGML_TURBOT_Y_MAX; ++y) {
                const int r = y - b;
                const float * TY = ggml_turbot_young_thr + ggml_turbot_young_off(b, y);
                const float * T  = ggml_turbot_old_thr   + ggml_turbot_old_off(b);
                for (int g = 0; g < n_cross; ++g) {
                    uint8_t cb[128], cr[128];
                    ggml_fp16_t gb, gy;
                    ggml_turbot_quantize_group(xs.data() + 128 * g, b, y, cb, cr, &gb, &gy);
                    for (int i = 0; i < 128; ++i) {
                        const float u = us[128 * g + i];
                        // us was normalised with x / norm, the coder with x * (1 / norm): allow the one-ULP index
                        // flip next to a threshold that SPEC 4.5 describes, count only real disagreements
                        const int j  = tt_upper_count(T, (1 << b) - 1, u);
                        const int iy = tt_upper_count(TY, (1 << y) - 1, u);
                        if (cb[i] != j && std::abs((int) cb[i] - j) > 1) {
                            ++bad;
                        }
                        if (((cb[i] << r) | cr[i]) != iy && std::abs(((cb[i] << r) | cr[i]) - iy) > 1) {
                            ++bad;
                        }
                    }
                }
            }
        }
        TCHECK(bad == 0, "quantize_group disagrees with the reference indices on %zu elements", bad);
    }

    // zero row -> exact zeros in both tiers, zero gains
    {
        const uint8_t bw[4] = { 2, 3, 5, 6 };
        const uint8_t yw[4] = { 8, 7, 7, 7 };
        ggml_turbot_side sd;
        make_side(sd, bw, yw);
        std::vector<float> x(1024, 0.0f), out(1024, 1.0f);
        std::vector<uint8_t> base(sd.base_row_bytes, 0xFF), young(sd.young_bytes, 0xFF);
        ggml_turbot_encode_side(x.data(), &sd, base.data(), young.data());
        ggml_turbot_decode_side(base.data(), nullptr, &sd, out.data());
        TCHECK(std::all_of(out.begin(), out.end(), [](float v) { return v == 0.0f; }), "zero row: old read not exactly zero");
        std::fill(out.begin(), out.end(), 1.0f);
        ggml_turbot_decode_side(base.data(), young.data(), &sd, out.data());
        TCHECK(std::all_of(out.begin(), out.end(), [](float v) { return v == 0.0f; }), "zero row: young read not exactly zero");
        for (int h = 0; h < 4; ++h) {
            for (int e = 0; e < 256; ++e) {
                TCHECK((int) ggml_turbot_get_code(base.data() + sd.base_off[h], sd.b[h], e) == (1 << (sd.b[h] - 1)), "zero row: head %d code is not 2^(b-1)", h);
            }
        }
        for (int i = 0; i < 8; ++i) {
            TCHECK(ggml_turbot_read_gain(base.data() + sd.base_gain_off, i) == 0.0f && ggml_turbot_read_gain(young.data() + sd.young_gain_off, i) == 0.0f, "zero row: gain %d not 0", i);
        }
    }

    // encode overwrites the whole row: garbage in the destination never leaks into the result
    {
        const uint8_t bw[4] = { 4, 2, 6, 3 };
        const uint8_t yw[4] = { 7, 7, 8, 5 };
        ggml_turbot_side sd;
        make_side(sd, bw, yw);
        std::vector<float> x(1024);
        test_row(gen, 5, x.data());
        std::vector<uint8_t> b0(sd.base_row_bytes, 0x00), b1(sd.base_row_bytes, 0xFF), y0(sd.young_bytes, 0x00), y1(sd.young_bytes, 0xFF);
        ggml_turbot_encode_side(x.data(), &sd, b0.data(), y0.data());
        ggml_turbot_encode_side(x.data(), &sd, b1.data(), y1.data());
        TCHECK(b0 == b1 && y0 == y1, "encode_side result depends on the previous destination bytes");
        // the young tier reads closer to the input than the old tier
        std::vector<float> xr = x, dold(1024), dyoung(1024);
        for (int o = 0; o < 1024; o += 128) {
            ggml_turbot_fwht128(xr.data() + o);
        }
        ggml_turbot_decode_side(b0.data(), nullptr, &sd, dold.data());
        ggml_turbot_decode_side(b0.data(), y0.data(), &sd, dyoung.data());
        double eo = 0.0, ey = 0.0;
        for (int i = 0; i < 1024; ++i) {
            eo += (dold[i] - xr[i]) * (double) (dold[i] - xr[i]);
            ey += (dyoung[i] - xr[i]) * (double) (dyoung[i] - xr[i]);
        }
        TCHECK(ey < eo, "young read error %.6g not below old read error %.6g", ey, eo);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// 5. center fill
// ---------------------------------------------------------------------------------------------------------------

static void test_fill() {
    printf("[5] center fill: MSE(fill, old)/D_b <= 0.30 and MSE(fill, true)/D_b <= 1.30\n");
    const int NR = g_quick ? 500 : 2000;   // rows x 8 groups
    std::mt19937 gen(99);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> xs(NR * 1024), xr(NR * 1024);
    double sig = 0.0;
    for (size_t i = 0; i < xs.size(); ++i) {
        xs[i] = nd(gen);
    }
    xr = xs;
    for (int t = 0; t < NR; ++t) {
        for (int o = 0; o < 1024; o += 128) {
            ggml_turbot_fwht128(xr.data() + 1024 * t + o);
        }
    }
    for (float v : xr) {
        sig += (double) v * v;
    }
    double worst_fo = 0.0, worst_ft = 0.0;
    for (int b = GGML_TURBOT_B_MIN; b <= GGML_TURBOT_B_MAX; ++b) {
        const float * C = ggml_turbot_old_levels + ggml_turbot_old_off(b);
        const float * T = ggml_turbot_old_thr    + ggml_turbot_old_off(b);
        const double D  = tt_analytic_D(C, T, 1 << b);
        printf("    b%d:", b);
        for (int y = b + 1; y <= GGML_TURBOT_Y_MAX; ++y) {
            const uint8_t bw[4] = { (uint8_t) b, (uint8_t) b, (uint8_t) b, (uint8_t) b };
            const uint8_t yw[4] = { (uint8_t) y, (uint8_t) y, (uint8_t) y, (uint8_t) y };
            ggml_turbot_side sd;
            make_side(sd, bw, yw);
            std::vector<uint8_t> base(sd.base_row_bytes), young(sd.young_bytes);
            std::vector<float> dold(1024), dfill(1024);
            double e_fo = 0.0, e_ft = 0.0;
            for (int t = 0; t < NR; ++t) {
                ggml_turbot_encode_side(xs.data() + 1024 * t, &sd, base.data(), nullptr);
                std::fill(young.begin(), young.end(), 0x5A);
                ggml_turbot_fill_side(base.data(), &sd, young.data());
                ggml_turbot_decode_side(base.data(), nullptr, &sd, dold.data());
                ggml_turbot_decode_side(base.data(), young.data(), &sd, dfill.data());
                for (int i = 0; i < 1024; ++i) {
                    const double a = (double) dfill[i] - dold[i];
                    const double c = (double) dfill[i] - xr[1024 * t + i];
                    e_fo += a * a;
                    e_ft += c * c;
                }
            }
            const double rfo = e_fo / sig / D, rft = e_ft / sig / D;
            worst_fo = std::max(worst_fo, rfo);
            worst_ft = std::max(worst_ft, rft);
            TCHECK(rfo <= 0.30, "b%d y%d MSE(fill, old)/D_b = %.4f > 0.30", b, y, rfo);
            TCHECK(rft <= 1.30, "b%d y%d MSE(fill, true)/D_b = %.4f > 1.30", b, y, rft);
            printf(" y%d %.4f/%.4f", y, rfo, rft);
        }
        printf("\n");
    }
    printf("    worst: fill vs old %.4f, fill vs true %.4f (record in docs/turbot/TESTING.md)\n", worst_fo, worst_ft);
}

// ---------------------------------------------------------------------------------------------------------------
// 6. op params and layout
// ---------------------------------------------------------------------------------------------------------------

// docs/turbot/plans/turbot-default.plan (SPEC appendix A)
static const int     DEFAULT_IL[16] = { 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63 };
static const uint8_t DEFAULT_BK[16][4] = {
    {2,2,2,4}, {2,5,3,2}, {4,5,4,2}, {4,4,3,4}, {4,4,3,4}, {5,4,4,6}, {5,5,6,5}, {6,5,5,5},
    {4,5,5,5}, {5,5,4,6}, {4,4,6,5}, {4,5,5,5}, {5,5,5,6}, {5,5,5,5}, {4,5,5,5}, {4,4,4,4} };
static const uint8_t DEFAULT_BV[16][4] = {
    {2,2,2,4}, {2,5,2,2}, {3,4,5,2}, {5,3,3,4}, {3,4,2,5}, {5,4,4,5}, {5,5,6,5}, {5,4,5,4},
    {4,4,4,5}, {4,4,4,5}, {4,4,5,4}, {4,4,5,5}, {5,4,4,5}, {5,5,5,5}, {5,6,5,5}, {5,4,5,5} };
// SPEC appendix A byte table: base K, base V, young K part, young V part
static const int DEFAULT_BYTES[4][16] = {
    { 336, 400, 496, 496, 496, 624, 688, 688, 624, 656, 624, 624, 688, 656, 624, 528 },
    { 336, 368, 464, 496, 464, 592, 688, 592, 560, 560, 560, 592, 592, 656, 688, 624 },
    { 592, 528, 432, 432, 432, 304, 240, 240, 304, 272, 304, 304, 240, 272, 304, 400 },
    { 592, 560, 464, 432, 464, 336, 240, 336, 368, 368, 368, 336, 336, 272, 240, 304 } };
// tools/turbot/plan_vram.py docs/turbot/plans/turbot-default.plan
static const uint64_t DEFAULT_PLAN_HASH = 0x56c3503c949a7749ull;

static const char * DEFAULT_PLAN_TEXT =
    "# turbot default plan\n"
    "POOL 65536\n"
    "CAP 16384\n"
    "L 3 K 2 2 2 4 V 2 2 2 4\n"
    "L 7 K 2 5 3 2 V 2 5 2 2\n"
    "L 11 K 4 5 4 2 V 3 4 5 2\n"
    "L 15 K 4 4 3 4 V 5 3 3 4\n"
    "L 19 K 4 4 3 4 V 3 4 2 5\n"
    "L 23 K 5 4 4 6 V 5 4 4 5\n"
    "L 27 K 5 5 6 5 V 5 5 6 5\n"
    "L 31 K 6 5 5 5 V 5 4 5 4\n"
    "L 35 K 4 5 5 5 V 4 4 4 5\n"
    "L 39 K 5 5 4 6 V 4 4 4 5\n"
    "L 43 K 4 4 6 5 V 4 4 5 4\n"
    "L 47 K 4 5 5 5 V 4 4 5 5\n"
    "L 51 K 5 5 5 6 V 5 4 4 5\n"
    "L 55 K 5 5 5 5 V 5 5 5 5\n"
    "L 59 K 4 5 5 5 V 5 6 5 5\n"
    "L 63 K 4 4 4 4 V 5 4 5 5\n";

static void test_params_layout() {
    printf("[6] op params and layout\n");
    const uint8_t y7[4] = { 7, 7, 7, 7 };

    // pack -> set -> get -> layer roundtrip
    {
        const uint8_t bk[4] = { 2, 6, 3, 5 }, yk[4] = { 8, 7, 7, 7 };
        const uint8_t bv[4] = { 4, 2, 6, 3 }, yv[4] = { 7, 7, 8, 5 };
        ggml_turbot_layer l;
        TCHECK(ggml_turbot_layer_init(&l, bk, bv, yk, yv), "layer_init of a legal layer failed");
        for (int side : { GGML_TURBOT_SIDE_K, GGML_TURBOT_SIDE_V, GGML_TURBOT_SIDE_BOTH }) {
            ggml_tensor t;
            std::memset(&t, 0, sizeof(t));
            std::memset(t.op_params, 0x77, GGML_TURBOT_OP_PARAMS_OFFSET);   // FA bytes 0..23 must survive
            ggml_turbot_op_params p;
            ggml_turbot_op_params_make(&p, &l, side);
            ggml_turbot_op_params_set(&t, &p);
            ggml_turbot_op_params q;
            TCHECK(ggml_turbot_op_params_get(&t, &q), "op params get failed");
            TCHECK(std::memcmp(&p, &q, sizeof(p)) == 0 && q.side == side && q.flags == 0, "op params roundtrip differs");
            for (int i = 0; i < GGML_TURBOT_OP_PARAMS_OFFSET; ++i) {
                TCHECK(((uint8_t *) t.op_params)[i] == 0x77, "op params byte %d overwritten", i);
            }
            for (int i = GGML_TURBOT_OP_PARAMS_OFFSET + 24; i < GGML_MAX_OP_PARAMS; ++i) {
                TCHECK(((uint8_t *) t.op_params)[i] == 0, "op params byte %d beyond the struct written", i);
            }
            ggml_turbot_layer l2;
            TCHECK(ggml_turbot_layer_from_op_params(&q, &l2), "layer_from_op_params failed");
            TCHECK(std::memcmp(&l, &l2, sizeof(l)) == 0, "layer from op params differs");

            ggml_tensor bad = t;
            ((uint8_t *) bad.op_params)[GGML_TURBOT_OP_PARAMS_OFFSET] ^= 1;
            TCHECK(!ggml_turbot_op_params_get(&bad, &q), "bad magic accepted");
            bad = t;
            ((uint8_t *) bad.op_params)[GGML_TURBOT_OP_PARAMS_OFFSET + 4] = 2;
            TCHECK(!ggml_turbot_op_params_get(&bad, &q), "bad version accepted");
            bad = t;
            ((uint8_t *) bad.op_params)[GGML_TURBOT_OP_PARAMS_OFFSET + 5] = 3;
            TCHECK(!ggml_turbot_op_params_get(&bad, &q), "bad side accepted");
            bad = t;
            ((uint8_t *) bad.op_params)[GGML_TURBOT_OP_PARAMS_OFFSET + 6] = 7;
            TCHECK(!ggml_turbot_op_params_get(&bad, &q), "bad log2_granule accepted");
        }
        TCHECK(l.k.base_off[0] == 0 && l.k.base_off[1] == 64 && l.k.base_off[2] == 256 && l.k.base_off[3] == 352, "K base offsets");
        TCHECK(l.k.young_off[0] == 0 && l.k.young_off[1] == 192 && l.k.young_off[2] == 224 && l.k.young_off[3] == 352, "K young offsets");
        TCHECK(l.k.base_gain_off == 512 && l.k.base_row_bytes == 528 && l.k.young_gain_off == 416 && l.k.young_bytes == 432, "K gain offsets / sizes");
        TCHECK(l.pool_v_off == l.k.young_bytes && l.pool_row_bytes == (uint32_t) l.k.young_bytes + l.v.young_bytes && l.pool_row_bytes % 16 == 0, "pool row layout");
        TCHECK(ggml_turbot_base_row_bytes(l.k.s) == l.k.base_row_bytes, "base_row_bytes(s)");
    }

    // default plan layout (SPEC 3.6, appendix A)
    {
        uint64_t base_cell = 0, pool_cell = 0, h = GGML_TURBOT_FNV_OFFSET;
        int s_sum = 0;
        for (int i = 0; i < 16; ++i) {
            ggml_turbot_layer l;
            TCHECK(ggml_turbot_layer_init(&l, DEFAULT_BK[i], DEFAULT_BV[i], y7, y7), "default plan layer %d rejected", DEFAULT_IL[i]);
            TCHECK(l.k.base_row_bytes == DEFAULT_BYTES[0][i] && l.v.base_row_bytes == DEFAULT_BYTES[1][i] &&
                   l.k.young_bytes == DEFAULT_BYTES[2][i] && l.v.young_bytes == DEFAULT_BYTES[3][i],
                   "il %d bytes %d %d %d %d, appendix %d %d %d %d", DEFAULT_IL[i], l.k.base_row_bytes, l.v.base_row_bytes,
                   l.k.young_bytes, l.v.young_bytes, DEFAULT_BYTES[0][i], DEFAULT_BYTES[1][i], DEFAULT_BYTES[2][i], DEFAULT_BYTES[3][i]);
            TCHECK(l.k.s >= GGML_TURBOT_S_MIN && l.v.s <= GGML_TURBOT_S_MAX, "il %d S out of the type family", DEFAULT_IL[i]);
            base_cell += (uint64_t) l.k.base_row_bytes + l.v.base_row_bytes;
            pool_cell += l.pool_row_bytes;
            s_sum += l.k.s + l.v.s;
            h = ggml_turbot_plan_hash_layer(h, DEFAULT_IL[i], &l);
        }
        h = ggml_turbot_plan_hash_finish(h, GGML_TURBOT_POOL_DEFAULT, GGML_TURBOT_CAP_DEFAULT);
        TCHECK(s_sum == 549, "default plan old width sum %d, expected 549", s_sum);
        TCHECK(base_cell == 18080 && pool_cell == 11616, "default plan bytes per cell %" PRIu64 " / %" PRIu64 ", expected 18080 / 11616", base_cell, pool_cell);
        const uint64_t total = 262144ull * base_cell + (uint64_t) GGML_TURBOT_POOL_DEFAULT * pool_cell;
        TCHECK(262144ull * base_cell == 4739563520ull && total == 5500829696ull, "default plan VRAM %" PRIu64 " B", total);
        TCHECK(total <= 262144ull * 16 * 2 * 656, "default plan above turbo5p");
        TCHECK(h == DEFAULT_PLAN_HASH, "default plan hash 0x%016" PRIx64 ", plan_vram.py says 0x%016" PRIx64, h, DEFAULT_PLAN_HASH);
        printf("    default plan: base %.2f MiB + pool %.2f MiB = %.2f MiB, hash 0x%016" PRIx64 "\n",
               262144.0 * base_cell / 1048576.0, 65536.0 * pool_cell / 1048576.0, total / 1048576.0, h);
    }

    // illegal widths
    {
        ggml_turbot_side sd;
        const uint8_t ok_b[4] = { 2, 3, 4, 6 }, ok_y[4] = { 3, 8, 5, 7 };
        TCHECK(ggml_turbot_side_init(&sd, ok_b, ok_y), "legal widths rejected");
        const uint8_t bad[][8] = {
            { 1, 3, 4, 6,   3, 8, 5, 7 },    // b < 2
            { 7, 3, 4, 6,   8, 8, 5, 7 },    // b > 6
            { 2, 3, 4, 6,   2, 8, 5, 7 },    // y == b
            { 2, 3, 4, 6,   3, 9, 5, 7 },    // y > 8
            { 2, 3, 4, 6,   3, 8, 3, 7 },    // y < b
            { 0, 0, 0, 0,   0, 0, 0, 0 },
        };
        for (const auto & w : bad) {
            TCHECK(!ggml_turbot_side_init(&sd, w, w + 4), "illegal widths %d %d %d %d / %d %d %d %d accepted", w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]);
        }
        ggml_turbot_layer l;
        TCHECK(!ggml_turbot_layer_init(&l, ok_b, bad[0], ok_y, bad[0] + 4), "layer with an illegal V side accepted");
    }

    // granule helpers
    TCHECK(ggml_turbot_granule_of(0) == 0 && ggml_turbot_granule_of(63) == 0 && ggml_turbot_granule_of(64) == 1 && ggml_turbot_granule_of(262143) == 4095, "granule_of");
    TCHECK(ggml_turbot_pool_row(0, 5) == 5 && ggml_turbot_pool_row(3, 64 * 7 + 9) == 3 * 64 + 9 && ggml_turbot_pool_row(1023, 262143) == 65535, "pool_row");
    TCHECK(ggml_turbot_type_of_s(8) == GGML_TYPE_TURBOT_S8 && ggml_turbot_type_of_s(24) == GGML_TYPE_TURBOT_S24 && ggml_turbot_s_of_type(GGML_TYPE_TURBOT_S24) == 24, "type family mapping");
    for (int s = GGML_TURBOT_S_MIN; s <= GGML_TURBOT_S_MAX; ++s) {
        const ggml_type t = ggml_turbot_type_of_s(s);
        TCHECK(ggml_turbot_is_type(t) && ggml_type_size(t) == ggml_turbot_base_row_bytes(s) && ggml_blck_size(t) == 1024, "type traits of S %d", s);
        TCHECK(ggml_row_size(t, 1024) == ggml_turbot_base_row_bytes(s) && ggml_row_size(t, 256) == (size_t) (8 * s + 4), "row sizes of S %d", s);
        TCHECK(ggml_is_quantized(t), "turbot_s%d not quantized", s);
    }
    TCHECK(!ggml_turbot_is_type(GGML_TYPE_TURBO5P_0) && !ggml_turbot_is_type(GGML_TYPE_F16) && !ggml_turbot_is_type(GGML_TYPE_COUNT), "is_type matches a foreign type");
}

// ---------------------------------------------------------------------------------------------------------------
// 7. plan parser and llama_kv_tier
// ---------------------------------------------------------------------------------------------------------------

#ifdef TURBOT_TEST_TIER

static std::vector<int32_t> qwen38_attn_layers() {
    return std::vector<int32_t>(DEFAULT_IL, DEFAULT_IL + 16);
}

static void test_plan_parser() {
    printf("[7a] plan parser\n");
    const auto layers = qwen38_attn_layers();
    const uint32_t kv = 262144;

    {
        llama_turbot_plan plan;
        std::string err;
        const bool ok = llama_turbot_plan_parse_text(DEFAULT_PLAN_TEXT, layers, kv, plan, err);
        TCHECK(ok, "default plan refused: %s", err.c_str());
        TCHECK(plan.layers.size() == 16 && plan.pool_cells == 65536 && plan.cap_cells == 16384, "default plan fields");
        TCHECK(plan.hash == DEFAULT_PLAN_HASH, "parsed plan hash 0x%016" PRIx64 ", expected 0x%016" PRIx64, plan.hash, DEFAULT_PLAN_HASH);
        for (int i = 0; ok && i < 16; ++i) {
            auto it = plan.layers.find(DEFAULT_IL[i]);
            TCHECK(it != plan.layers.end(), "layer %d missing", DEFAULT_IL[i]);
            if (it != plan.layers.end()) {
                TCHECK(std::memcmp(it->second.k.b, DEFAULT_BK[i], 4) == 0 && std::memcmp(it->second.v.b, DEFAULT_BV[i], 4) == 0, "layer %d widths", DEFAULT_IL[i]);
                TCHECK(it->second.k.y[0] == 7 && it->second.v.y[3] == 7, "layer %d default young width", DEFAULT_IL[i]);
            }
        }
    }

    auto accept = [&](const std::string & text, const char * what, llama_turbot_plan * out = nullptr) {
        llama_turbot_plan plan;
        std::string err;
        const bool ok = llama_turbot_plan_parse_text(text, layers, kv, plan, err);
        TCHECK(ok, "%s refused: %s", what, err.c_str());
        if (out) {
            *out = plan;
        }
    };
    auto refuse = [&](const std::string & text, const char * what) {
        llama_turbot_plan plan;
        std::string err;
        const bool ok = llama_turbot_plan_parse_text(text, layers, kv, plan, err);
        TCHECK(!ok, "%s accepted", what);
        TCHECK(ok || !err.empty(), "%s refused without a message", what);
    };
    const std::string base = DEFAULT_PLAN_TEXT;
    std::string no_pool = base;
    no_pool.erase(no_pool.find("POOL 65536\n"), std::strlen("POOL 65536\n"));
    no_pool.erase(no_pool.find("CAP 16384\n"), std::strlen("CAP 16384\n"));

    {
        llama_turbot_plan p;
        accept(no_pool + "\n   \n# comment only\nW 256\nW2 16384\nM 7\n", "kvfq keys, blank and comment lines", &p);
        TCHECK(p.pool_cells == GGML_TURBOT_POOL_DEFAULT && p.cap_cells == GGML_TURBOT_CAP_DEFAULT, "POOL/CAP defaults");
        accept(no_pool + "POOL 0\n", "POOL 0", &p);
        TCHECK(p.pool_cells == 0, "POOL 0 not kept");
        accept(no_pool + "POOL 262144\nCAP 0\n", "POOL == kv_size, CAP 0");
        // layer 23 old widths are K 5 4 4 6 / V 5 4 4 5, so every young width here is in [b+1, 8]
        accept(base + "Y 23 K 6 8 5 7 V 6 5 8 6\n", "Y line", &p);
        auto it = p.layers.find(23);
        TCHECK(it != p.layers.end() && it->second.k.y[1] == 8 && it->second.v.y[0] == 6 && it->second.k.y[0] == 6, "Y line not applied");
        std::string commented = base;
        commented.replace(commented.find("L 3 K 2 2 2 4 V 2 2 2 4\n"), std::strlen("L 3 K 2 2 2 4 V 2 2 2 4\n"), "L 3 K 2 2 2 4 V 2 2 2 4   # trailing comment\n");
        accept(commented, "trailing comment");
    }
    refuse(base + "FOO 1\n", "unknown tag");
    refuse(no_pool + "POOL\n", "POOL without a value");
    refuse(no_pool + "POOL 100\n", "POOL not a multiple of 64");
    {
        // a pool larger than the cache (explicit, or the implicit default 65536) is clamped to the cache in whole
        // granules, never refused and never passed on to the tier assert
        llama_turbot_plan plan;
        accept(no_pool + "POOL 262208\n", "POOL above kv_size is clamped", &plan);
        TCHECK(plan.pool_cells == 262144, "POOL 262208 at kv_size 262144 gave %u, want 262144", plan.pool_cells);
        std::string err;
        bool ok = llama_turbot_plan_parse_text(no_pool, layers, 4096, plan, err);
        TCHECK(ok && plan.pool_cells == 4096, "implicit POOL 65536 at kv_size 4096: ok %d pool %u (%s)", (int) ok, plan.pool_cells, err.c_str());
        ok = llama_turbot_plan_parse_text(base, layers, 32768, plan, err);
        TCHECK(ok && plan.pool_cells == 32768 && plan.cap_cells == 16384, "default plan at kv_size 32768: ok %d pool %u cap %u (%s)",
                (int) ok, plan.pool_cells, plan.cap_cells, err.c_str());
        ok = llama_turbot_plan_parse_text(no_pool + "POOL 0\n", layers, 4096, plan, err);
        TCHECK(ok && plan.pool_cells == 0, "POOL 0 at kv_size 4096: ok %d pool %u (%s)", (int) ok, plan.pool_cells, err.c_str());
    }
    refuse(no_pool + "CAP -1\n", "negative CAP");
    refuse(base + "POOL 64\n", "duplicate POOL");
    refuse(base + "CAP 64\n", "duplicate CAP");
    refuse(base + "L 3 K 2 2 2 4 V 2 2 2 4\n", "duplicate L");
    refuse(base + "Y 3 K 7 7 7 7 V 7 7 7 7\nY 3 K 7 7 7 7 V 7 7 7 7\n", "duplicate Y");
    {
        std::string t = base;
        t.replace(t.find("L 3 K 2 2 2 4"), std::strlen("L 3 K 2 2 2 4"), "L 3 K 1 2 2 4");
        refuse(t, "old width 1");
        t = base;
        t.replace(t.find("L 3 K 2 2 2 4"), std::strlen("L 3 K 2 2 2 4"), "L 3 K 7 2 2 4");
        refuse(t, "old width 7");
        t = base;
        t.replace(t.find("L 3 K 2 2 2 4"), std::strlen("L 3 K 2 2 2 4"), "L 3 K 2 2 2");
        refuse(t, "malformed L line");
        t = base;
        t.erase(t.find("L 63 K 4 4 4 4 V 5 4 5 5\n"), std::strlen("L 63 K 4 4 4 4 V 5 4 5 5\n"));
        refuse(t, "missing L for an attention layer");
    }
    refuse(base + "L 64 K 4 4 4 4 V 4 4 4 4\n", "L for a layer the cache does not hold");
    refuse(base + "Y 64 K 7 7 7 7 V 7 7 7 7\n", "Y for a layer the cache does not hold");
    refuse(base + "Y 23 K 5 7 7 7 V 7 7 7 7\n", "young width == old width");
    refuse(base + "Y 23 K 7 7 7 9 V 7 7 7 7\n", "young width 9");
    refuse(base + "L 3 K 2 2 x 4 V 2 2 2 4\n", "non-numeric width");
    {
        llama_turbot_plan p1, p2;
        accept(base, "base", &p1);
        accept(no_pool + "POOL 65536\nCAP 8192\n", "CAP 8192", &p2);
        TCHECK(p1.hash != p2.hash, "plan hash ignores CAP");
    }
}

// [TAG_TURBOT_EMBED_PLAN] the built-in plan (src/llama-turbot-default-plan.h), llama_turbot_plan_matches and the plan source
static void test_default_plan() {
    printf("[7a'] built-in default plan and plan source\n");
    const auto     layers = qwen38_attn_layers();
    const uint32_t kv     = 262144;

    const std::string text = llama_turbot_default_plan_text();
    TCHECK(!text.empty() && text.find('\r') == std::string::npos, "built-in plan text is empty or not LF");

    std::string err;
    {
        llama_turbot_plan plan;
        const bool ok = llama_turbot_plan_parse_text(text, layers, kv, plan, err);
        TCHECK(ok, "built-in plan refused: %s", err.c_str());
        TCHECK(plan.hash == DEFAULT_PLAN_HASH, "built-in plan hash 0x%016" PRIx64 ", expected 0x%016" PRIx64
                " (docs/turbot/plans/turbot-default.plan changed: regenerate the header and update this test)", plan.hash, DEFAULT_PLAN_HASH);
        TCHECK(llama_turbot_default_plan_hash() == DEFAULT_PLAN_HASH, "LLAMA_TURBOT_DEFAULT_PLAN_HASH 0x%016" PRIx64,
                llama_turbot_default_plan_hash());
    }

    std::string why = "stale";
    TCHECK(llama_turbot_plan_matches(text, layers, kv, why) && why.empty(), "Qwen3.8 layers do not match: %s", why.c_str());
    TCHECK(llama_turbot_plan_matches(text, layers, 4096, why), "Qwen3.8 layers at kv_size 4096 (POOL clamp) do not match: %s", why.c_str());

    // Spark-X2.5-4B shape: full-attention layers 3, 7, ..., 35. The first L line it does not hold is layer 39 (line 14).
    std::vector<int32_t> spark;
    for (int32_t il = 3; il <= 35; il += 4) {
        spark.push_back(il);
    }
    TCHECK(!llama_turbot_plan_matches(text, spark, kv, why), "9 attention layers match");
    TCHECK(why.find("layer 39 ") != std::string::npos && why.find(LLAMA_TURBOT_PLAN_BUILTIN_NAME) != std::string::npos,
            "mismatch reason for 9 layers: %s", why.c_str());

    std::vector<int32_t> shifted;
    for (int32_t il = 0; il < 64; il += 4) {
        shifted.push_back(il);
    }
    TCHECK(!llama_turbot_plan_matches(text, shifted, kv, why) && !why.empty(), "layers 0, 4, ..., 60 match");

    std::vector<int32_t> extra = layers;
    extra.push_back(64);
    TCHECK(!llama_turbot_plan_matches(text, extra, kv, why) && why.find("missing L line for attention layer 64") != std::string::npos,
            "17 attention layers: %s", why.c_str());

    TCHECK(!llama_turbot_plan_matches(text, std::vector<int32_t>(), kv, why) && !why.empty(), "no attention layers match");
    TCHECK(!llama_turbot_plan_matches("L 3 K 1 2 2 4 V 2 2 2 4\n", std::vector<int32_t>(1, 3), kv, why, "custom.plan") &&
            why.find("custom.plan") != std::string::npos, "source name missing from the reason: %s", why.c_str());

    // plan source: llama_turbot_set_plan_path wins over env LLAMA_TURBOT_PLAN; "default" is the built-in plan
    llama_turbot_set_plan_path(LLAMA_TURBOT_PLAN_KEYWORD_DEFAULT);
    llama_turbot_plan_source src = llama_turbot_plan_get_source();
    TCHECK(src.origin == LLAMA_TURBOT_PLAN_BUILTIN_FORCED && src.name == LLAMA_TURBOT_PLAN_BUILTIN_NAME && src.path.empty(),
            "'default' keyword: origin %d name '%s'", (int) src.origin, src.name.c_str());
    std::string got;
    TCHECK(llama_turbot_plan_read(src, got, err) && got == text, "reading the built-in plan: %s", err.c_str());

    llama_turbot_set_plan_path("./no-such-dir/default");
    src = llama_turbot_plan_get_source();
    TCHECK(src.origin == LLAMA_TURBOT_PLAN_FILE && src.path == "./no-such-dir/default" && src.name == src.path, "a path ending in default is a file");
    TCHECK(!llama_turbot_plan_read(src, got, err) && err.find("cannot open plan file") != std::string::npos, "missing plan file: %s", err.c_str());

    llama_turbot_set_plan_path(nullptr);
}

// quota of SPEC 9.6
static uint32_t tier_quota(uint32_t pool, uint32_t cap, const std::vector<uint32_t> & n, size_t s) {
    uint64_t sum = 0;
    uint32_t n_active = 0;
    for (uint32_t v : n) {
        sum += v;
        n_active += v > 0;
    }
    const int64_t n_eff = std::max<int64_t>(0, (int64_t) pool - 64 * GGML_TURBOT_QUOTA_SLACK_GRANULES * (int64_t) n_active);
    if (sum == 0) {
        return 0;
    }
    return (uint32_t) std::min<uint64_t>(cap, (uint64_t) n_eff * n[s] / sum);
}

// Drives a real llama_kv_cells and a llama_kv_tier the way llama_kv_cache and llama_context do (SPEC 9.6, 9.7).
struct tier_sim {
    const uint32_t kv_size;
    llama_kv_cells cells;
    llama_kv_tier  tier;
    std::set<uint32_t> empty;
    std::vector<std::map<llama_pos, uint32_t>> seq_cells;   // seq -> pos -> cell

    tier_sim(uint32_t kv, uint32_t pool, uint32_t cap) : kv_size(kv), tier(kv, pool, cap), seq_cells(LLAMA_MAX_SEQ) {
        cells.resize(kv);
        for (uint32_t i = 0; i < kv; ++i) {
            empty.insert(i);
        }
    }

    llama_pos next_pos(llama_seq_id s) const {
        return seq_cells[s].empty() ? 0 : seq_cells[s].rbegin()->first + 1;
    }

    std::vector<uint32_t> lowest_empty(uint32_t n) const {
        std::vector<uint32_t> out;
        for (auto it = empty.begin(); it != empty.end() && out.size() < n; ++it) {
            out.push_back(*it);
        }
        GGML_ASSERT(out.size() == n);
        return out;
    }

    // one ubatch: rows[i] of sequence seqs[i] at cell idx[i]; ok == false runs the failure path of llama_context::decode
    void ubatch(const std::vector<llama_seq_id> & seqs, const std::vector<uint32_t> & idx, bool ok = true) {
        const uint32_t n = (uint32_t) seqs.size();
        GGML_ASSERT(idx.size() == n);
        std::vector<llama_pos>      pos(n);
        std::vector<int32_t>        n_seq_id(n, 1);
        std::vector<llama_seq_id>   sid(seqs);
        std::vector<llama_seq_id *> sidp(n);
        std::vector<llama_seq_id>   unq;
        std::map<llama_seq_id, llama_pos> pos_min;
        std::map<llama_seq_id, llama_pos> cursor;
        for (uint32_t i = 0; i < n; ++i) {
            const llama_seq_id s = seqs[i];
            if (!cursor.count(s)) {
                cursor[s] = next_pos(s);
                pos_min[s] = cursor[s];
                unq.push_back(s);
            }
            pos[i]  = cursor[s]++;
            sidp[i] = &sid[i];
            cells.pos_set(idx[i], pos[i]);
            cells.seq_add(idx[i], s);
            empty.erase(idx[i]);
            seq_cells[s][pos[i]] = idx[i];
        }
        llama_ubatch ub{};
        ub.n_tokens     = n;
        ub.n_seq_tokens = n;
        ub.n_seqs       = 1;
        ub.n_seqs_unq   = (uint32_t) unq.size();
        ub.n_pos        = 1;
        ub.pos          = pos.data();
        ub.n_seq_id     = n_seq_id.data();
        ub.seq_id       = sidp.data();
        ub.seq_id_unq   = unq.data();
        tier.begin_ubatch(ub, idx, cells);
        last_young = tier.young_rows();
        last_fill  = tier.fill_entries();
        if (ok) {
            tier.commit_ubatch(cells);
        } else {
            tier.abort_ubatch();
            for (auto & pm : pos_min) {
                seq_rm(pm.first, pm.second, -1);
            }
        }
    }

    std::vector<int32_t> last_young;
    std::vector<int32_t> last_fill;

    // contiguous rows for one sequence at the lowest empty cells
    void decode(llama_seq_id s, uint32_t n, bool ok = true) {
        ubatch(std::vector<llama_seq_id>(n, s), lowest_empty(n), ok);
    }

    void prefill(llama_seq_id s, uint32_t n, uint32_t ub = 512) {
        for (uint32_t done = 0; done < n; done += ub) {
            decode(s, std::min(ub, n - done));
        }
    }

    // llama_kv_cache::seq_rm with the turbot hooks (SPEC 9.7)
    void seq_rm(llama_seq_id s, llama_pos p0, llama_pos p1) {
        const bool tail = p1 < 0 || p1 == std::numeric_limits<llama_pos>::max();
        if (p0 < 0) p0 = 0;
        if (p1 < 0) p1 = std::numeric_limits<llama_pos>::max();
        if (s >= 0) {
            uint64_t min_st  = UINT64_MAX;
            bool     removed = false;
            auto & m = seq_cells[s];
            for (auto it = m.lower_bound(p0); it != m.end() && it->first < p1;) {
                const uint32_t c = it->second;
                min_st  = std::min(min_st, tier.stamp(c));
                removed = true;
                if (cells.seq_rm(c, s)) {
                    tier.on_cell_emptied(c);
                    empty.insert(c);
                }
                it = m.erase(it);
            }
            if (tail && removed) {
                tier.on_seq_tail_removed(s, min_st);
            }
        } else {
            std::map<llama_seq_id, uint64_t> min_st;
            for (uint32_t c = 0; c < kv_size; ++c) {
                if (cells.is_empty(c) || !cells.pos_in(c, p0, p1)) {
                    continue;
                }
                const llama_pos p = cells.pos_get(c);
                for (llama_seq_id q = 0; q < LLAMA_MAX_SEQ; ++q) {
                    if (cells.seq_has(c, q)) {
                        auto & ms = min_st[q];
                        ms = ms == 0 ? tier.stamp(c) : std::min(ms, tier.stamp(c));
                        seq_cells[q].erase(p);
                    }
                }
                cells.rm(c);
                tier.on_cell_emptied(c);
                empty.insert(c);
            }
            if (tail) {
                for (auto & ms : min_st) {
                    tier.on_seq_tail_removed(ms.first, ms.second);
                }
            }
        }
    }

    void seq_cp(llama_seq_id src, llama_seq_id dst) {
        const bool dst_was_empty = cells.seq_n_cells(dst) == 0;
        for (auto & pc : seq_cells[src]) {
            if (!cells.seq_has(pc.second, dst)) {
                cells.seq_add(pc.second, dst);
                seq_cells[dst][pc.first] = pc.second;
            }
        }
        tier.on_seq_cp(src, dst, dst_was_empty);
    }

    // state_read_meta applies the cells, then state_read_turbot calls restore_cells (SPEC 9.10)
    std::vector<int32_t> restore(llama_seq_id s, const std::vector<uint32_t> & idx, const std::vector<uint64_t> & stamps,
                                 const std::vector<uint8_t> & young, uint64_t counter) {
        llama_pos p = next_pos(s);
        for (uint32_t c : idx) {
            cells.pos_set(c, p);
            cells.seq_add(c, s);
            empty.erase(c);
            seq_cells[s][p++] = c;
        }
        return tier.restore_cells(idx, stamps, young, { { s, counter } }, cells);
    }

    uint32_t n_live(llama_seq_id s) const {
        return (uint32_t) seq_cells[s].size();
    }

    uint32_t n_young(llama_seq_id s) const {
        uint32_t n = 0;
        for (auto & pc : seq_cells[s]) {
            n += tier.cell_young(pc.second);
        }
        return n;
    }

    // every one of the newest Y live cells of s (by stamp) is young, and at most Y + slack cells of s are young
    void check_band(llama_seq_id s, uint32_t Y, uint32_t slack, const char * what) {
        std::vector<std::pair<uint64_t, uint32_t>> v;
        for (auto & pc : seq_cells[s]) {
            v.push_back({ tier.stamp(pc.second), pc.second });
        }
        std::sort(v.begin(), v.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
        uint32_t missing = 0;
        for (size_t i = 0; i < std::min<size_t>(Y, v.size()); ++i) {
            missing += !tier.cell_young(v[i].second);
        }
        const uint32_t ny = n_young(s);
        TCHECK(missing == 0, "%s: seq %d: %u of its newest %u cells are not young", what, s, missing, Y);
        TCHECK(ny <= Y + slack, "%s: seq %d has %u young cells, band %u + slack %u", what, s, ny, Y, slack);
    }

    uint32_t live_granules_young() const {
        uint32_t n = 0;
        for (int32_t g : tier.granule_slots()) {
            n += g >= 0;
        }
        return n;
    }
};

static void test_tier() {
    printf("[7b] llama_kv_tier scenarios\n");

    // single sequence: the young band is the newest Y_s cells
    {
        tier_sim sim(262144, 65536, 16384);
        TCHECK(sim.tier.n_granules() == 4096 && sim.tier.n_slots() == 1024, "geometry");
        sim.prefill(0, 40000);
        for (int t = 0; t < 1000; ++t) {
            sim.decode(0, 1);
        }
        const uint32_t Y = tier_quota(65536, 16384, { sim.n_live(0) }, 0);
        TCHECK(Y == 16384, "single sequence quota %u", Y);
        sim.check_band(0, Y, 128, "single sequence");
        TCHECK(sim.tier.row_counter(0) == sim.n_live(0), "row counter %" PRIu64 " != live %u", sim.tier.row_counter(0), sim.n_live(0));
        TCHECK(sim.tier.n_evictions() == 0, "single sequence evicted %" PRIu64 " granules", sim.tier.n_evictions());
    }

    // four interleaved sequences at quota: no evictions after warm-up, every band intact
    {
        tier_sim sim(262144, 65536, 16384);
        for (llama_seq_id s = 0; s < 4; ++s) {
            sim.prefill(s, 17000);
        }
        uint64_t ev_warm = 0;
        for (int round = 0; round < 2000; ++round) {
            if (round == 100) {
                ev_warm = sim.tier.n_evictions();
            }
            sim.ubatch({ 0, 1, 2, 3 }, sim.lowest_empty(4));
        }
        std::vector<uint32_t> n = { sim.n_live(0), sim.n_live(1), sim.n_live(2), sim.n_live(3) };
        const uint32_t Y = tier_quota(65536, 16384, n, 0);
        TCHECK(Y == 16256, "four-sequence quota %u, expected 16256", Y);
        TCHECK(sim.tier.n_evictions() == ev_warm, "four sequences: %" PRIu64 " evictions after warm-up", sim.tier.n_evictions() - ev_warm);
        for (llama_seq_id s = 0; s < 4; ++s) {
            sim.check_band(s, Y, 128 + 64, "four sequences");
        }
        TCHECK(sim.live_granules_young() <= 1024, "more young granules than slots");
    }

    // eviction order with the pool forced full: smallest margin first, never the previous ubatch's granule
    {
        tier_sim sim(4096, 512, 100000);
        std::vector<uint32_t> idx7;
        for (uint32_t g = 0; g < 7; ++g) idx7.push_back(g * 64);
        sim.ubatch(std::vector<llama_seq_id>(7, 0), idx7);
        sim.ubatch({ 0 }, { 7 * 64 });
        for (uint32_t g = 0; g < 8; ++g) {
            TCHECK(sim.tier.granule_slots()[g] >= 0, "eviction order: granule %u lost its slot before the pool was full", g);
        }
        sim.ubatch({ 0 }, { 8 * 64 });
        const auto & gs = sim.tier.granule_slots();
        TCHECK(gs[0] < 0, "eviction order: the smallest-margin granule 0 kept its slot");
        TCHECK(gs[7] >= 0 && gs[8] >= 0, "eviction order: previous-ubatch granule 7 or the new granule 8 has no slot");
        TCHECK(sim.tier.n_evictions() == 1, "eviction order: %" PRIu64 " evictions, expected 1", sim.tier.n_evictions());
        TCHECK(sim.last_young.size() == 1 && sim.last_young[0] >= 0, "eviction order: the new row got no pool row");
    }

    // an empty owned slot is taken before any live granule
    {
        tier_sim sim(4096, 512, 100000);
        std::vector<uint32_t> idx8;
        for (uint32_t g = 0; g < 8; ++g) idx8.push_back(g * 64);
        sim.ubatch(std::vector<llama_seq_id>(8, 0), idx8);
        sim.seq_rm(0, 4, 5);                                  // empties granule 4, not a tail removal
        TCHECK(sim.tier.row_counter(0) == 8, "non-tail seq_rm moved the counter to %" PRIu64, sim.tier.row_counter(0));
        TCHECK(sim.tier.granule_slots()[4] >= 0, "empty-first: slot of granule 4 returned before commit");
        sim.ubatch({ 0 }, { 8 * 64 });
        const auto & gs = sim.tier.granule_slots();
        TCHECK(gs[4] < 0 && gs[8] >= 0, "empty-first: granule 4 slot %d, granule 8 slot %d", gs[4], gs[8]);
        TCHECK(sim.tier.n_evictions() == 0, "empty-first: a live granule was evicted");
        // one sequence: the quota is N_eff = 384 cells, so every live granule is still wanted after the commit
        for (uint32_t g : { 0u, 1u, 2u, 3u, 5u, 6u, 7u }) {
            TCHECK(gs[g] >= 0, "empty-first: live granule %u lost its slot", g);
        }
    }

    // release then new prefill, and release then restore: empty owned slots are reused, nothing live is evicted
    for (int variant = 0; variant < 2; ++variant) {
        tier_sim sim(8192, 1024, 100000);
        sim.prefill(0, 400, 64);
        sim.prefill(1, 400, 64);
        const uint32_t young0 = sim.n_young(0);
        sim.seq_rm(1, -1, -1);
        TCHECK(sim.tier.row_counter(1) == 0, "release: counter of seq 1 is %" PRIu64, sim.tier.row_counter(1));
        const uint64_t ev = sim.tier.n_evictions();
        std::vector<uint32_t> idx;
        for (uint32_t c = 4096; c < 4096 + 400; ++c) idx.push_back(c);
        // one step touching 7 granules against 3 free slots: 4 must come from the released, still owned granules
        if (variant == 0) {
            sim.ubatch(std::vector<llama_seq_id>(400, 2), idx);
        } else {
            std::vector<uint64_t> st(400);
            for (uint32_t i = 0; i < 400; ++i) st[i] = i + 1;
            const auto rows = sim.restore(2, idx, st, std::vector<uint8_t>(400, 1), 400);
            TCHECK(std::all_of(rows.begin(), rows.end(), [](int32_t r) { return r >= 0; }), "restore after release: a young cell lost its refinement");
        }
        TCHECK(sim.tier.n_evictions() == ev, "%s after release evicted %" PRIu64 " live granules", variant ? "restore" : "prefill", sim.tier.n_evictions() - ev);
        TCHECK(sim.n_young(0) == young0, "%s after release: seq 0 young cells %u -> %u", variant ? "restore" : "prefill", young0, sim.n_young(0));
        TCHECK(sim.n_young(2) == 400, "%s after release: seq 2 has %u young cells", variant ? "restore" : "prefill", sim.n_young(2));
    }

    // draft loop: write 4 rows, drop the last 2, 10,000 times
    {
        tier_sim sim(65536, 16384, 4096);
        sim.prefill(0, 8000);
        for (int t = 0; t < 10000; ++t) {
            sim.decode(0, 4);
            sim.seq_rm(0, sim.next_pos(0) - 2, -1);
        }
        TCHECK(sim.tier.row_counter(0) == sim.n_live(0), "draft loop: row counter %" PRIu64 ", live rows %u", sim.tier.row_counter(0), sim.n_live(0));
        const uint32_t Y = tier_quota(16384, 4096, { sim.n_live(0) }, 0);
        sim.decode(0, 1);                                     // a commit after the last rollback
        sim.check_band(0, Y, 128, "draft loop");
    }

    // seq_cp into an empty dst adopts the source counter
    {
        tier_sim sim(262144, 65536, 16384);
        sim.prefill(1, 50000, 2048);
        sim.seq_rm(1, 0, 49990);                              // not a tail: the counter stays ahead
        sim.seq_rm(1, -1, -1);                                // copy_state_to: seq_rm(other, -1, -1)
        TCHECK(sim.tier.row_counter(1) > 40000, "seq_cp setup: follower counter %" PRIu64, sim.tier.row_counter(1));
        sim.prefill(0, 30000, 2048);
        sim.seq_cp(0, 1);
        TCHECK(sim.tier.row_counter(1) == sim.tier.row_counter(0), "seq_cp into an empty dst: counter %" PRIu64 ", source %" PRIu64, sim.tier.row_counter(1), sim.tier.row_counter(0));
        sim.seq_rm(0, -1, -1);                                // the leader releases
        sim.decode(1, 1);
        sim.check_band(1, tier_quota(65536, 16384, { 0, sim.n_live(1) }, 1), 128, "seq_cp follower");
    }

    // sequence restore into a slot id whose counter is larger than the saved one
    {
        tier_sim sim(262144, 65536, 16384);
        sim.prefill(3, 90000, 2048);
        sim.seq_rm(3, 0, 89990);
        sim.seq_rm(3, -1, -1);                                // state_read_meta: seq_rm(dest, -1, -1)
        TCHECK(sim.tier.row_counter(3) > 20000, "restore setup: counter %" PRIu64, sim.tier.row_counter(3));
        std::vector<uint32_t> idx = sim.lowest_empty(20000);
        std::vector<uint64_t> st(20000);
        for (uint32_t i = 0; i < 20000; ++i) st[i] = i + 1;
        std::vector<uint8_t> young(20000, 0);
        for (uint32_t i = 20000 - 16384; i < 20000; ++i) young[i] = 1;
        sim.restore(3, idx, st, young, 20000);
        TCHECK(sim.tier.row_counter(3) == 20000, "restore assigns the counter: %" PRIu64, sim.tier.row_counter(3));
        sim.decode(3, 1);
        sim.check_band(3, 16384, 128, "restore into a larger counter");
    }

    // stale bits: seq_rm a cell of a young granule, restore a cell there, check before any begin
    // (16 slots: three active sequences still leave granule 0 wanted at the final commit)
    {
        tier_sim sim(4096, 1024, 1000);
        sim.decode(0, 64);
        TCHECK(sim.tier.cell_young(10) && sim.tier.cell_young(11), "stale bits setup: granule 0 not young");
        sim.seq_rm(0, 10, 12);
        TCHECK(!sim.tier.cell_young(10) && !sim.tier.cell_young(11), "an emptied cell still reports young");
        auto r0 = sim.restore(1, { 10 }, { 5 }, { 0 }, 5);
        TCHECK(r0.size() == 1 && r0[0] == -1, "restored old cell got pool row %d", r0.empty() ? -2 : r0[0]);
        TCHECK(!sim.tier.cell_young(10), "a cell restored without refinement reports young (stale ref_valid bit)");
        auto r1 = sim.restore(2, { 11 }, { 7 }, { 1 }, 7);
        TCHECK(r1.size() == 1 && r1[0] == ggml_turbot_pool_row(sim.tier.granule_slots()[0], 11), "restored young cell in a young granule got pool row %d", r1.empty() ? -2 : r1[0]);
        TCHECK(sim.tier.cell_young(11), "restored young cell is not young");
        for (uint32_t c = 64; c < 128; ++c) {
            TCHECK(!sim.tier.cell_young(c), "empty cell %u reports young", c);
        }
        sim.decode(0, 1);
        TCHECK(sim.tier.cell_young(10), "cell 10 not refined by the next ubatch's fill");
    }

    // trim then rewrite into an old granule -> fill entry; failure -> abort keeps state and re-queues the fill
    {
        tier_sim sim(4096, 256, 64);
        sim.prefill(0, 300, 32);
        TCHECK(sim.tier.granule_slots()[1] < 0, "trim setup: granule 1 is still young");
        sim.seq_rm(0, 100, -1);
        TCHECK(sim.tier.row_counter(0) == 100, "trim rollback: counter %" PRIu64 ", expected 100", sim.tier.row_counter(0));

        auto check_fill = [&](const char * what) {
            const auto & f = sim.last_fill;
            TCHECK(f.size() == 4, "%s: %zu fill ints, expected one entry", what, f.size());
            if (f.size() == 4) {
                const int32_t slot = sim.tier.granule_slots()[1];
                TCHECK(f[0] == 1 && f[1] == slot, "%s: fill entry granule %d slot %d, expected 1 / %d", what, f[0], f[1], slot);
                TCHECK((uint32_t) f[2] == 0xFFFFFFFFu && (uint32_t) f[3] == 0xFu, "%s: fill mask %08x %08x, expected ffffffff 0000000f", what, (uint32_t) f[3], (uint32_t) f[2]);
            }
            TCHECK(sim.last_young.size() == 1 && sim.last_young[0] == ggml_turbot_pool_row(sim.tier.granule_slots()[1], 100), "%s: young row of the rewrite", what);
        };

        sim.decode(0, 1, /*ok =*/ false);
        check_fill("rewrite into an old granule (failed compute)");
        TCHECK(sim.tier.row_counter(0) == 100, "failure seq_rm did not roll the counter back: %" PRIu64, sim.tier.row_counter(0));
        TCHECK(sim.n_live(0) == 100, "failure seq_rm left %u live rows", sim.n_live(0));
        sim.decode(0, 1);
        check_fill("retry after the failure");
        for (uint32_t c = 64; c <= 100; ++c) {
            TCHECK(sim.tier.cell_young(c), "cell %u of the filled granule is not young", c);
        }
    }

    // restore failure after restore_cells -> abort_restore frees the slots it allocated
    {
        tier_sim sim(4096, 256, 1000);
        sim.decode(0, 64);
        const int32_t slot0 = sim.tier.granule_slots()[0];
        std::vector<uint32_t> idx;
        for (uint32_t c = 1024; c < 1088; ++c) idx.push_back(c);
        std::vector<uint64_t> st(64);
        for (uint32_t i = 0; i < 64; ++i) st[i] = i + 1;
        auto rows = sim.restore(1, idx, st, std::vector<uint8_t>(64, 1), 64);
        TCHECK(sim.tier.granule_slots()[16] >= 0 && rows[0] >= 0, "restore did not allocate a slot for granule 16");
        sim.tier.abort_restore();
        TCHECK(sim.tier.granule_slots()[16] < 0, "abort_restore kept the slot of granule 16");
        TCHECK(!sim.tier.cell_young(1024) && !sim.tier.cell_young(1087), "abort_restore left restored cells young");
        TCHECK(sim.tier.granule_slots()[0] == slot0 && sim.tier.cell_young(0), "abort_restore touched granule 0");
    }

    // POOL 0
    {
        tier_sim sim(4096, 0, 16384);
        TCHECK(sim.tier.n_slots() == 0, "POOL 0 has %u slots", sim.tier.n_slots());
        sim.decode(0, 100);
        TCHECK(std::all_of(sim.last_young.begin(), sim.last_young.end(), [](int32_t r) { return r == -1; }), "POOL 0: a row got a pool row");
        TCHECK(std::all_of(sim.tier.granule_slots().begin(), sim.tier.granule_slots().end(), [](int32_t s) { return s == -1; }), "POOL 0: a granule got a slot");
        TCHECK(sim.last_fill.empty(), "POOL 0: fill entries");
        sim.seq_rm(0, 50, -1);
        sim.decode(0, 10);
        TCHECK(sim.last_fill.empty() && !sim.tier.cell_young(0), "POOL 0: fill or young after a trim");
    }

    // the last sequence id with kv_unified
    {
        tier_sim sim(4096, 256, 1000);
        const llama_seq_id s = LLAMA_MAX_SEQ - 1;
        sim.decode(s, 20);
        sim.decode(s, 20);
        TCHECK(sim.tier.row_counter(s) == 40, "seq %d counter %" PRIu64, s, sim.tier.row_counter(s));
        TCHECK(sim.tier.cell_young(0) && sim.tier.stamp(39) == 40, "seq %d rows not young or stamped", s);
    }

    // determinism: identical scripts give identical states
    {
        auto script = [](tier_sim & sim) {
            sim.prefill(0, 3000, 128);
            sim.prefill(1, 2000, 100);
            for (int t = 0; t < 300; ++t) {
                sim.ubatch({ 0, 1 }, sim.lowest_empty(2));
                if (t % 7 == 0) {
                    sim.seq_rm(t % 2, sim.next_pos(t % 2) - 1, -1);
                }
            }
            sim.seq_rm(1, -1, -1);
            sim.prefill(2, 1500, 256);
            sim.seq_cp(0, 3);
            sim.decode(3, 5);
        };
        tier_sim a(8192, 1024, 1024), b(8192, 1024, 1024);
        script(a);
        script(b);
        TCHECK(a.tier.granule_slots() == b.tier.granule_slots(), "determinism: granule tables differ");
        TCHECK(a.last_young == b.last_young && a.last_fill == b.last_fill, "determinism: last ubatch inputs differ");
        TCHECK(a.tier.n_evictions() == b.tier.n_evictions(), "determinism: eviction counts differ");
        bool same = true;
        for (uint32_t c = 0; c < 8192; ++c) {
            same = same && a.tier.stamp(c) == b.tier.stamp(c) && a.tier.cell_young(c) == b.tier.cell_young(c);
        }
        for (llama_seq_id s = 0; s < 4; ++s) {
            same = same && a.tier.row_counter(s) == b.tier.row_counter(s);
        }
        TCHECK(same, "determinism: stamps, young cells or counters differ");
    }
}

#endif // TURBOT_TEST_TIER

int main(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--quick") == 0) {
            g_quick = true;
        } else {
            fprintf(stderr, "usage: %s [--quick]\n", argv[0]);
            return 2;
        }
    }

    test_tables();
    test_nesting();
    test_planes();
    test_coder();
    test_fill();
    test_params_layout();
#ifdef TURBOT_TEST_TIER
    test_plan_parser();
    test_default_plan();
    test_tier();
#else
    printf("[7] plan parser and llama_kv_tier: SKIPPED (internal llama symbols do not link in this build, see docs/turbot/TESTING.md)\n");
#endif

    printf("\n%d checks, %d failed\n", g_checks, g_fail);
    printf("%s\n", g_fail == 0 ? "OK" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
