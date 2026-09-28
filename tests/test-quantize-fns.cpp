// Unit tests for quantization specific functions - quantize, dequantize and dot product

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-turbot.h"

#undef NDEBUG
#include <assert.h>
#include <algorithm>
#include <cmath>
#include <math.h>
#include <stdio.h>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

constexpr float MAX_QUANTIZATION_REFERENCE_ERROR = 0.0001f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR = 0.002f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_BINARY = 0.025f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_TERNARY = 0.01f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_2BITS = 0.0075f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_3BITS = 0.0040f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS = 0.0050f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_FP4 = 0.0030f;
constexpr float MAX_DOT_PRODUCT_ERROR = 0.02f;
constexpr float MAX_DOT_PRODUCT_ERROR_LOWBIT = 0.04f;
constexpr float MAX_DOT_PRODUCT_ERROR_FP4 = 0.03f;
constexpr float MAX_DOT_PRODUCT_ERROR_BINARY = 0.40f;
constexpr float MAX_DOT_PRODUCT_ERROR_TERNARY = 0.15f;

static const char* RESULT_STR[] = {"ok", "FAILED"};

// [TAG_TURBO_QFNS_ROT] the turbo KV types store and dequantize the 128-point signed WHT of the data (the graph
// rotates Q the same way), so their round trip is checked against WHT(x) and their F32 vec_dot operand is rotated too
static bool type_is_wht_rotated(ggml_type type) {
    switch (type) {
        case GGML_TYPE_TURBO2_0:
        case GGML_TYPE_TURBO3_0:
        case GGML_TYPE_TURBO4_0:
        case GGML_TYPE_TURBO4P_0:
        case GGML_TYPE_TURBO5P_0:
        case GGML_TYPE_TURBO5P512_0:
            return true;
        default:
            return false;
    }
}

static std::vector<float> wht_rotated(const float * x, size_t n) {
    assert(n % 128 == 0);
    std::vector<float> r(x, x + n);
    for (size_t i = 0; i < n; i += 128) {
        ggml_turbot_fwht128(r.data() + i);
    }
    return r;
}

// Lloyd-Max codebooks on the rotated (near-Gaussian) values: error about 0.0076 (2 bits), 0.0039 (3), 0.0021 (4),
// 0.0011 (5) on this data; a wrong basis gives about 0.031
static float max_turbo_quantization_error(ggml_type type) {
    switch (type) {
        case GGML_TYPE_TURBO2_0:     return 0.0095f;
        case GGML_TYPE_TURBO3_0:     return 0.0052f;
        case GGML_TYPE_TURBO4_0:
        case GGML_TYPE_TURBO4P_0:    return 0.0028f;
        default:                     return 0.0015f; // turbo5p, turbo5p512
    }
}

// the norm correction keeps |x| but shrinks the part along x by about sqrt(1 - D) (D = codebook MSE), so on this
// correlated data the dot error is mostly that bias: about 0.066 (2 bits), 0.019 (3), 0.0065 (4), 0.0013 (5)
static float max_turbo_dot_product_error(ggml_type type) {
    switch (type) {
        case GGML_TYPE_TURBO2_0: return 0.09f;
        case GGML_TYPE_TURBO3_0: return MAX_DOT_PRODUCT_ERROR_LOWBIT;
        default:                 return MAX_DOT_PRODUCT_ERROR;
    }
}


// Generate synthetic data
static void generate_data(float offset, size_t n, float * dst, float amplitude = 2.0f) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = 0.1 + amplitude*cosf(i + offset);
    }
}

// Calculate RMSE between two float arrays
static float array_rmse(const float * a1, const float * a2, size_t n) {
    double sum = 0;
    for (size_t i = 0; i < n; i++) {
        double diff = a1[i] - a2[i];
        sum += diff * diff;
    }
    return sqrtf(sum) / n;
}

// Total quantization error on test data
static float total_quantization_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data, bool rotated = false) {
    std::vector<uint8_t> tmp_q(2*test_size);
    std::vector<float> tmp_out(test_size);

    qfns_cpu->from_float(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out.data(), test_size);
    if (rotated) {
        return array_rmse(wht_rotated(test_data, test_size).data(), tmp_out.data(), test_size);
    }
    return array_rmse(test_data, tmp_out.data(), test_size);
}

// Total quantization error on test data
static float reference_quantization_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data) {
    std::vector<uint8_t> tmp_q(2*test_size);
    std::vector<float> tmp_out(test_size);
    std::vector<float> tmp_out_ref(test_size);

    // FIXME: why is done twice?
    qfns_cpu->from_float(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out.data(), test_size);

    qfns->from_float_ref(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out_ref.data(), test_size);

    return array_rmse(tmp_out.data(), tmp_out_ref.data(), test_size);
}

static float dot_product(const float * a1, const float * a2, size_t test_size) {
    double sum = 0;
    for (size_t i = 0; i < test_size; i++) {
        sum += a1[i] * a2[i];
    }
    return sum;
}

// Total dot product error
static float dot_product_error(const ggml_type_traits_cpu * qfns_cpu, ggml_type src0_type, size_t test_size,
                               const float * test_data1, const float * test_data2,
                               const float * test_data3, const float * test_data4,
                               const int nrc) {
    const auto * vdot = ggml_get_type_traits_cpu(qfns_cpu->vec_dot_type);
    const size_t pad  = 64;
    const size_t bx   = ggml_row_size(src0_type, test_size) + pad;
    const size_t by   = ggml_row_size(qfns_cpu->vec_dot_type, test_size) + pad;

    std::vector<uint8_t> tmp_q1(bx * nrc);
    std::vector<uint8_t> tmp_q2(by * nrc);

    qfns_cpu->from_float(test_data1, tmp_q1.data(), test_size);
    if (type_is_wht_rotated(src0_type)) {
        // [TAG_TURBO_QFNS_ROT] the WHT is orthonormal: <WHT(x), WHT(y)> = <x, y>, so the reference below is unchanged
        GGML_ASSERT(nrc == 1 && qfns_cpu->vec_dot_type == GGML_TYPE_F32);
        vdot->from_float(wht_rotated(test_data2, test_size).data(), tmp_q2.data(), test_size);
    } else {
        vdot->from_float(test_data2, tmp_q2.data(), test_size);
    }

    if (nrc == 1) {
        float result = INFINITY;
        qfns_cpu->vec_dot(test_size, &result, 0, tmp_q1.data(), 0, tmp_q2.data(), 0, 1);

        const float dot_ref = dot_product(test_data1, test_data2, test_size);
        return fabsf(result - dot_ref) / test_size;
    }

    // nrc == 2: kernel computes a 2x2 dot product matrix
    // Output layout: s[0]=dot(vx0,vy0), s[1]=dot(vx1,vy0), s[bs]=dot(vx0,vy1), s[bs+1]=dot(vx1,vy1)
    // row and output strides are padded, same as in the mul_mat path
    qfns_cpu->from_float(test_data3, tmp_q1.data() + bx, test_size);
    vdot->from_float(test_data4, tmp_q2.data() + by, test_size);

    const size_t bs = 16;
    std::vector<float> result(bs + 2, INFINITY);
    qfns_cpu->vec_dot(test_size, result.data(), bs, tmp_q1.data(), bx, tmp_q2.data(), by, 2);

    const float ref00 = dot_product(test_data1, test_data2, test_size);
    const float ref10 = dot_product(test_data3, test_data2, test_size);
    const float ref01 = dot_product(test_data1, test_data4, test_size);
    const float ref11 = dot_product(test_data3, test_data4, test_size);

    const auto err = [test_size](float val, float ref) {
        const float e = fabsf(val - ref) / test_size;
        return std::isfinite(e) ? e : INFINITY;
    };

    return std::max({err(result[0], ref00), err(result[1], ref10), err(result[bs], ref01), err(result[bs + 1], ref11)});
}

static int test_vec_dot_f32(bool verbose) {
    const auto * f32 = ggml_get_type_traits_cpu(GGML_TYPE_F32);
    int num_failed = 0;
    for (int n : {1, 2, 3, 5, 7, 8, 15, 16, 17, 31, 33, 63, 67, 127, 129, 193, 255, 1023}) {
        std::vector<float> a(n);
        std::vector<float> b(n);
        generate_data(0.0, n, a.data());
        generate_data(1.0, n, b.data());

        float result = 0.0f;
        f32->vec_dot(n, &result, 0, a.data(), 0, b.data(), 0, 1);
        const float ref = dot_product(a.data(), b.data(), n);
        const float error = fabsf(result - ref) / n;

        const bool failed = !(error < MAX_QUANTIZATION_REFERENCE_ERROR);
        num_failed += failed;
        if (failed || verbose) {
            printf(" f32 vec_dot n=%4d:                 %s (ref=%f got=%f err=%f)\n",
                   n, RESULT_STR[failed], ref, result, error);
        }
    }
    return num_failed;
}

static int test_vec_dot_q(bool verbose) {
    int num_failed = 0;

    const size_t test_size = 32 * 128;

    std::vector<float> test_data(test_size);
    std::vector<float> test_data2(test_size);
    std::vector<float> test_data3(test_size);
    std::vector<float> test_data4(test_size);

    generate_data(0.0, test_data.size(), test_data.data());
    generate_data(1.0, test_data2.size(), test_data2.data());
    generate_data(3.0, test_data3.size(), test_data3.data(), 1.0f);
    generate_data(4.0, test_data4.size(), test_data4.data(), 1.5f);

    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        ggml_type type = (ggml_type) i;
        const auto * qfns = ggml_get_type_traits(type);
        const auto * qfns_cpu = ggml_get_type_traits_cpu(type);

        // deprecated - skip
        if (qfns->blck_size == 0) {
            continue;
        }

        const ggml_type ei = (ggml_type)i;

        printf("Testing %s\n", ggml_type_name((ggml_type) i));
        ggml_quantize_init(ei);

        if (qfns_cpu->from_float && qfns->to_float) {
            const bool rotated = type_is_wht_rotated(type);
            const float total_error = total_quantization_error(qfns, qfns_cpu, test_size, test_data.data(), rotated);
            const float max_quantization_error =
                rotated                   ? max_turbo_quantization_error(type) :
                type == GGML_TYPE_Q1_0    ? MAX_QUANTIZATION_TOTAL_ERROR_BINARY :
                type == GGML_TYPE_TQ1_0   ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_TQ2_0   ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_Q2_0    ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_Q2_K    ? MAX_QUANTIZATION_TOTAL_ERROR_2BITS :
                type == GGML_TYPE_IQ2_S   ? MAX_QUANTIZATION_TOTAL_ERROR_2BITS :
                type == GGML_TYPE_Q3_K    ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :
                type == GGML_TYPE_IQ3_S   ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :
                type == GGML_TYPE_IQ3_XXS ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS :
                type == GGML_TYPE_NVFP4   ? MAX_QUANTIZATION_TOTAL_ERROR_FP4 : MAX_QUANTIZATION_TOTAL_ERROR;
            bool failed = !(total_error < max_quantization_error);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s absolute quantization error:    %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], total_error);
            }

            const float reference_error = reference_quantization_error(qfns, qfns_cpu, test_size, test_data.data());
            failed = !(reference_error < MAX_QUANTIZATION_REFERENCE_ERROR);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s reference implementation error: %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], reference_error);
            }

            const float vec_dot_error = dot_product_error(qfns_cpu, type, test_size, test_data.data(), test_data2.data(), nullptr, nullptr, 1);
            const float max_allowed_error = type == GGML_TYPE_Q2_K || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_XXS ||
                type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ2_S
                ? MAX_DOT_PRODUCT_ERROR_LOWBIT
                : type_is_wht_rotated(type)
                ? max_turbo_dot_product_error(type)
                : type == GGML_TYPE_Q1_0
                ? MAX_DOT_PRODUCT_ERROR_BINARY
                : type == GGML_TYPE_TQ1_0 || type == GGML_TYPE_TQ2_0 || type == GGML_TYPE_Q2_0
                ? MAX_DOT_PRODUCT_ERROR_TERNARY
                : type == GGML_TYPE_NVFP4
                ? MAX_DOT_PRODUCT_ERROR_FP4
                : MAX_DOT_PRODUCT_ERROR;
            failed = !(vec_dot_error < max_allowed_error);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s dot product error:              %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], vec_dot_error);
            }

            // Test nrc=2 path for types that support it
            if (qfns_cpu->nrows == 2) {
                const float vec_dot_error_nrc2 = dot_product_error(qfns_cpu, type, test_size, test_data.data(), test_data2.data(), test_data3.data(), test_data4.data(), 2);
                failed = !(vec_dot_error_nrc2 < max_allowed_error);
                num_failed += failed;
                if (failed || verbose) {
                    printf("%5s dot product error (nrc=2):    %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], vec_dot_error_nrc2);
                }
            }
        }
    }

    return num_failed;
}

// [TAG_FN_CPU_Q5_1_AVX512] [TAG_FN_CPU_MMID_MR] the kernels of the CPU expert path claim to be bitwise equal to the
// AVX2 dot products. That holds exactly where the compiler does not contract a*b + c differently in the two functions
// (MSVC never contracts); elsewhere allow a rounding step.
static bool fn_same_float(float a, float b) {
#if defined(_MSC_VER) && !defined(__clang__)
    return memcmp(&a, &b, sizeof(float)) == 0;
#else
    return a == b || fabsf(a - b) <= 1e-6f * std::max(1.0f, fabsf(a));
#endif
}

// raw blocks with random bytes and random scales in the first nhalf fp16 fields of each block (all bit patterns of
// the packed quants, including the q5_1 high bits)
static std::vector<uint8_t> fn_random_blocks(ggml_type type, int n, int nhalf, std::mt19937 & rng) {
    const size_t bsize = ggml_type_size(type);
    const size_t nblk  = (size_t) n / ggml_blck_size(type);
    std::vector<uint8_t> data(bsize * nblk);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> scale(-0.05f, 0.05f);
    for (size_t b = 0; b < nblk; b++) {
        uint8_t * blk = data.data() + b * bsize;
        for (size_t j = 0; j < bsize; j++) {
            blk[j] = (uint8_t) byte(rng);
        }
        for (int h = 0; h < nhalf; h++) {
            const ggml_fp16_t v = ggml_fp32_to_fp16(scale(rng));
            memcpy(blk + h * sizeof(ggml_fp16_t), &v, sizeof(v));
        }
    }
    return data;
}

// the activation side: random floats quantized with the CPU from_float of the type (so the q8_1 sums are consistent)
static std::vector<uint8_t> fn_random_act(ggml_type vec_dot_type, int n, std::mt19937 & rng) {
    std::vector<float> f(n);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    for (float & v : f) {
        v = uni(rng);
    }
    std::vector<uint8_t> q(ggml_row_size(vec_dot_type, n));
    ggml_get_type_traits_cpu(vec_dot_type)->from_float(f.data(), q.data(), n);
    return q;
}

// [TAG_FN_CPU_Q5_1_AVX512] GGML_CPU_Q5_1_AVX512 must not change a single bit: the q5_1 dot product with the switch on
// against the same call with it off, on random blocks (every high bit pattern) and odd block counts (the 256-bit tail)
static int test_fn_q5_1_avx512(bool verbose) {
    const auto * q51 = ggml_get_type_traits_cpu(GGML_TYPE_Q5_1);
    const int saved = ggml_cpu_fn_get_switch(GGML_CPU_FN_Q5_1_AVX512);

    std::mt19937 rng(1234);
    int num_failed = 0;
    int n_cases    = 0;
    for (int n : {32, 64, 96, 640, 672, 2560}) {
        for (int rep = 0; rep < 16; ++rep) {
            const std::vector<uint8_t> xq = fn_random_blocks(GGML_TYPE_Q5_1, n, 2, rng); // d, m
            const std::vector<uint8_t> yq = fn_random_act(GGML_TYPE_Q8_1, n, rng);
            float r[2];
            for (int v = 0; v < 2; ++v) {
                ggml_cpu_fn_set_switch(GGML_CPU_FN_Q5_1_AVX512, v);
                q51->vec_dot(n, &r[v], 0, xq.data(), 0, yq.data(), 0, 1);
            }
            n_cases++;
            const bool failed = !fn_same_float(r[0], r[1]);
            num_failed += failed;
            if (failed || (verbose && rep == 0)) {
                printf(" q5_1 AVX-512 vs AVX2 n=%4d:        %s (%.9g vs %.9g)\n", n, RESULT_STR[failed], r[1], r[0]);
            }
        }
    }
    ggml_cpu_fn_set_switch(GGML_CPU_FN_Q5_1_AVX512, saved);

    if (num_failed || verbose) {
        printf(" q5_1 AVX-512 dot product: %d cases, %d not bitwise equal to AVX2%s\n", n_cases, num_failed,
               ggml_cpu_has_avx512() ? "" : " (no AVX-512 in this build: both runs took the AVX2 body)");
    }
    return num_failed;
}

// [TAG_FN_CPU_MMID_MR] the multi-row x multi-token kernels against vec_dot, value by value: nr rows (3 and 16 cover
// the row loop), nc = 1..4 tokens, both bodies (GGML_CPU_MMID_MR=1: 512-bit where built, =2: 256-bit), row lengths
// with odd block counts (the tails), random raw weight blocks with random fp16 scales
static int test_fn_mmid_mr(bool verbose) {
    struct mr_type { ggml_type type; int nhalf; std::vector<int> ns; };
    const mr_type types[] = {
        { GGML_TYPE_Q4_K,   2, { 256, 768, 2560 } },       // d, dmin
        { GGML_TYPE_Q5_K,   2, { 256, 768, 2560 } },       // d, dmin
        { GGML_TYPE_Q5_1,   2, { 32, 96, 640, 672, 2560 } }, // d, m
        { GGML_TYPE_Q8_0,   1, { 32, 96, 640, 672, 2560 } }, // d
        { GGML_TYPE_IQ4_NL, 1, { 32, 96, 640, 672, 2560 } }, // d
    };

    const int saved_mr   = ggml_cpu_fn_get_switch(GGML_CPU_FN_MMID_MR);
    const int saved_q5_1 = ggml_cpu_fn_get_switch(GGML_CPU_FN_Q5_1_AVX512);
    ggml_cpu_fn_set_switch(GGML_CPU_FN_Q5_1_AVX512, 0); // vec_dot = the AVX2 bodies

    std::mt19937 rng(4321);
    int num_failed = 0;
    int n_values   = 0;
    for (const mr_type & t : types) {
        const auto * tr = ggml_get_type_traits_cpu(t.type);
        {
            // does this build have a kernel for the type
            const int n0 = (int) ggml_blck_size(t.type);
            std::vector<uint8_t> x0(ggml_row_size(t.type, n0), 0);
            std::vector<uint8_t> y0(ggml_row_size(tr->vec_dot_type, n0), 0);
            const void * c0[1] = { y0.data() };
            float o0 = 0.0f;
            if (!ggml_cpu_fn_vec_dot_mr(t.type, n0, &o0, 1, x0.data(), 0, 1, c0, 1)) {
                printf(" %6s mmid_mr: no kernel in this build, skipped\n", ggml_type_name(t.type));
                continue;
            }
        }
        for (int n : t.ns) {
            const int    max_nr = 16;
            const size_t bx     = ggml_row_size(t.type, n);
            const size_t by     = ggml_row_size(tr->vec_dot_type, n);
            const std::vector<uint8_t> xq = fn_random_blocks(t.type, n * max_nr, t.nhalf, rng); // max_nr rows
            std::vector<std::vector<uint8_t>> yq;
            for (int c = 0; c < 4; ++c) {
                yq.push_back(fn_random_act(tr->vec_dot_type, n, rng));
            }
            GGML_ASSERT(xq.size() == bx * max_nr && yq[0].size() == by);

            // reference: vec_dot per (row, column)
            float ref[4][max_nr];
            for (int c = 0; c < 4; ++c) {
                for (int r = 0; r < max_nr; ++r) {
                    tr->vec_dot(n, &ref[c][r], 0, xq.data() + r * bx, 0, yq[c].data(), 0, 1);
                }
            }

            for (int body : {1, 2}) {
                ggml_cpu_fn_set_switch(GGML_CPU_FN_MMID_MR, body);
                for (int nr : {1, 3, max_nr}) {
                    for (int nc = 1; nc <= 4; ++nc) {
                        const void * cols[4] = { yq[0].data(), yq[1].data(), yq[2].data(), yq[3].data() };
                        const size_t bs = 24; // padded like the MUL_MAT_ID scratch
                        std::vector<float> out(bs * 4, NAN);
                        if (!ggml_cpu_fn_vec_dot_mr(t.type, n, out.data(), bs, xq.data(), bx, nr, cols, nc)) {
                            num_failed++;
                            printf(" %6s mmid_mr n=%4d nr=%2d nc=%d: FAILED, the kernel refused the call\n", ggml_type_name(t.type), n, nr, nc);
                            continue;
                        }
                        for (int c = 0; c < nc; ++c) {
                            for (int r = 0; r < nr; ++r) {
                                n_values++;
                                const bool failed = !fn_same_float(out[c * bs + r], ref[c][r]);
                                num_failed += failed;
                                if (failed) {
                                    printf(" %6s mmid_mr n=%4d nr=%2d nc=%d body=%d (r=%d, c=%d): FAILED (%.9g vs vec_dot %.9g)\n",
                                           ggml_type_name(t.type), n, nr, nc, body, r, c, out[c * bs + r], ref[c][r]);
                                }
                            }
                        }
                        // nothing written outside the nr x nc block
                        for (int c = 0; c < 4; ++c) {
                            for (int r = 0; r < (int) bs; ++r) {
                                if ((c >= nc || r >= nr) && !std::isnan(out[c * bs + r])) {
                                    num_failed++;
                                    printf(" %6s mmid_mr n=%4d nr=%2d nc=%d body=%d: FAILED, wrote s[%d]\n",
                                           ggml_type_name(t.type), n, nr, nc, body, c * (int) bs + r);
                                }
                            }
                        }
                    }
                }
            }
        }
        if (verbose) {
            printf(" %6s mmid_mr: checked against vec_dot\n", ggml_type_name(t.type));
        }
    }

    ggml_cpu_fn_set_switch(GGML_CPU_FN_MMID_MR, saved_mr);
    ggml_cpu_fn_set_switch(GGML_CPU_FN_Q5_1_AVX512, saved_q5_1);

    if (num_failed || verbose) {
        printf(" mmid_mr kernels: %d values, %d not bitwise equal to vec_dot\n", n_values, num_failed);
    }
    return num_failed;
}

int main(int argc, char * argv[]) {
    bool verbose = false;

    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        if (arg == "-v") {
            verbose = true;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }

    ggml_cpu_init();

    int num_failed = 0;

    num_failed += test_vec_dot_f32(verbose);
    num_failed += test_vec_dot_q(verbose);
    num_failed += test_fn_q5_1_avx512(verbose); // [TAG_FN_CPU_Q5_1_AVX512]
    num_failed += test_fn_mmid_mr(verbose);     // [TAG_FN_CPU_MMID_MR]

    if (num_failed || verbose) {
        printf("%d tests failed\n", num_failed);
    }

    return num_failed > 0;
}
