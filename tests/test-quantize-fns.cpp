// Unit tests for quantization specific functions - quantize, dequantize and dot product

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-turbot.h"

#undef NDEBUG
#include <assert.h>
#include <algorithm>
#include <cmath>
#include <limits>
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

// Positive finite E4M3 values in encoding order, from OCP FP8 rev. 1.1, section 5.1.
static const float F8_E4M3_VALUES[] = {
    0.0f, 0.001953125f, 0.00390625f, 0.005859375f, 0.0078125f, 0.009765625f, 0.01171875f, 0.013671875f,
    0.015625f, 0.017578125f, 0.01953125f, 0.021484375f, 0.0234375f, 0.025390625f, 0.02734375f, 0.029296875f,
    0.03125f, 0.03515625f, 0.0390625f, 0.04296875f, 0.046875f, 0.05078125f, 0.0546875f, 0.05859375f,
    0.0625f, 0.0703125f, 0.078125f, 0.0859375f, 0.09375f, 0.1015625f, 0.109375f, 0.1171875f,
    0.125f, 0.140625f, 0.15625f, 0.171875f, 0.1875f, 0.203125f, 0.21875f, 0.234375f,
    0.25f, 0.28125f, 0.3125f, 0.34375f, 0.375f, 0.40625f, 0.4375f, 0.46875f,
    0.5f, 0.5625f, 0.625f, 0.6875f, 0.75f, 0.8125f, 0.875f, 0.9375f,
    1.0f, 1.125f, 1.25f, 1.375f, 1.5f, 1.625f, 1.75f, 1.875f,
    2.0f, 2.25f, 2.5f, 2.75f, 3.0f, 3.25f, 3.5f, 3.75f,
    4.0f, 4.5f, 5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f,
    8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
    16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f,
    32.0f, 36.0f, 40.0f, 44.0f, 48.0f, 52.0f, 56.0f, 60.0f,
    64.0f, 72.0f, 80.0f, 88.0f, 96.0f, 104.0f, 112.0f, 120.0f,
    128.0f, 144.0f, 160.0f, 176.0f, 192.0f, 208.0f, 224.0f, 240.0f,
    256.0f, 288.0f, 320.0f, 352.0f, 384.0f, 416.0f, 448.0f,
};

static uint8_t fp32_to_f8_e4m3_ref(float value) {
    if (isnan(value)) {
        return 0x7F;
    }
    const uint8_t sign = signbit(value) ? 0x80 : 0;
    const double magnitude = fabs((double) value);
    if (magnitude >= F8_E4M3_VALUES[126]) {
        return sign | 126;
    }
    int nearest = 0;
    double distance = magnitude;
    for (int i = 1; i < 127; ++i) {
        const double candidate = fabs(magnitude - F8_E4M3_VALUES[i]);
        if (candidate < distance || (candidate == distance && (i & 1) == 0)) {
            nearest = i;
            distance = candidate;
        }
    }
    return sign | (uint8_t) nearest;
}

static int test_f8_e4m3(bool verbose) {
    static_assert(sizeof(ggml_fp8_e4m3_t) == 1);

    const auto * traits = ggml_get_type_traits(GGML_TYPE_F8_E4M3);
    const auto * traits_cpu = ggml_get_type_traits_cpu(GGML_TYPE_F8_E4M3);
    std::vector<uint8_t> encoded(256);
    std::vector<uint8_t> roundtrip(256);
    std::vector<float> decoded(256);
    int num_failed = 0;

    const bool traits_failed = traits->blck_size != 1 || traits->type_size != sizeof(ggml_fp8_e4m3_t) || traits->is_quantized;
    num_failed += traits_failed;
    if (verbose || traits_failed) {
        printf("f8_e4m3 scalar type traits:          %s\n", RESULT_STR[traits_failed]);
    }

    for (int i = 0; i < 256; ++i) {
        encoded[i] = (uint8_t) i;
    }
    traits->to_float(encoded.data(), decoded.data(), decoded.size());
    traits_cpu->from_float(decoded.data(), roundtrip.data(), decoded.size());

    for (int i = 0; i < 256; ++i) {
        const bool is_nan = (i & 0x7F) == 0x7F;
        const float expected = is_nan ? NAN : copysignf(F8_E4M3_VALUES[i & 0x7F], i & 0x80 ? -1.0f : 1.0f);
        const bool decode_failed = is_nan ? !isnan(decoded[i])
            : decoded[i] != expected || (expected == 0.0f && signbit(decoded[i]) != signbit(expected));
        const bool encode_failed = !is_nan && roundtrip[i] != encoded[i];
        if (decode_failed || encode_failed) {
            num_failed++;
            if (verbose) {
                printf("f8_e4m3 code 0x%02x failed: decoded=%f expected=%f roundtrip=0x%02x\n",
                        i, decoded[i], expected, roundtrip[i]);
            }
        }
    }

    std::vector<float> inputs;
    for (float value : F8_E4M3_VALUES) {
        inputs.push_back(value);
        inputs.push_back(-value);
    }
    for (int i = 0; i < 126; ++i) {
        const float midpoint = (F8_E4M3_VALUES[i] + F8_E4M3_VALUES[i + 1]) / 2.0f;
        for (float value : {nextafterf(midpoint, -INFINITY), midpoint, nextafterf(midpoint, INFINITY)}) {
            inputs.push_back(value);
            inputs.push_back(-value);
        }
    }
    for (float value : {std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::max(),
                       nextafterf(448.0f, INFINITY), nextafterf(464.0f, -INFINITY), 464.0f,
                       nextafterf(464.0f, INFINITY), INFINITY, NAN}) {
        inputs.push_back(value);
        inputs.push_back(-value);
    }

    std::vector<uint8_t> quantized(inputs.size());
    std::vector<uint8_t> quantized_ref(inputs.size());
    traits_cpu->from_float(inputs.data(), quantized.data(), inputs.size());
    traits->from_float_ref(inputs.data(), quantized_ref.data(), inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        const uint8_t expected = fp32_to_f8_e4m3_ref(inputs[i]);
        if (quantized[i] != expected || quantized_ref[i] != expected) {
            num_failed++;
            if (verbose) {
                printf("f8_e4m3 input %a failed: encoded=0x%02x reference=0x%02x expected=0x%02x\n",
                       inputs[i], quantized[i], quantized_ref[i], expected);
            }
        }
    }

    for (int n : {0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 65, 129, 257}) {
        std::vector<uint8_t> x(n + 1);
        std::vector<float> y(n + 1);
        double expected = 0.0;
        for (int i = 0; i < n; ++i) {
            const uint8_t code = (uint8_t) (i % 254 + (i % 254 >= 127));
            x[i + 1] = code;
            y[i + 1] = (i % 7 - 3) * 0.125f;
            expected += copysignf(F8_E4M3_VALUES[code & 0x7F], code & 0x80 ? -1.0f : 1.0f) * y[i + 1];
        }
        float actual;
        traits_cpu->vec_dot(n, &actual, 0, x.data() + 1, 0, y.data() + 1, 0, 1);
        const bool failed = fabs(actual - expected) > 1e-5 * std::max(1.0, fabs(expected));
        num_failed += failed;
        if (verbose || failed) {
            printf("f8_e4m3 dot n=%d:                  %s (ref=%f got=%f)\n", n, RESULT_STR[failed], expected, actual);
        }
    }

    for (uint8_t code : {0x7F, 0xFF}) {
        std::vector<uint8_t> x(65, 0);
        std::vector<float> y(65, 1.0f);
        x[0] = x.back() = code;
        float actual;
        traits_cpu->vec_dot(x.size(), &actual, 0, x.data(), 0, y.data(), 0, 1);
        num_failed += !isnan(actual);
    }

    if (verbose || num_failed) {
        printf("f8_e4m3 exhaustive conversion:      %s (%d failures)\n", RESULT_STR[num_failed != 0], num_failed);
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

// In every group, all values with importance are equal, and the max (0) has none.
// The scale search in make_qkx3_quants then fits min == max and passes inf/nan to nearest_int (#29804).
static int test_quantize_imatrix_degenerate(bool verbose) {
    const int64_t n = 256;
    std::vector<float> x(n);
    std::vector<float> imatrix(n);
    std::vector<float> out(n);
    int num_failed = 0;

    printf("Testing degenerate imatrix:\n");
    for (ggml_type type : {GGML_TYPE_Q2_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1}) {
        const int64_t group = type == GGML_TYPE_Q2_K ? 16 : 32;
        for (int64_t i = 0; i < n; ++i) {
            const int64_t g = i / group;
            const int64_t p = i % group;
            const bool important = p % 3 == 1;
            x[i] = important ? -0.02f*(g + 1) : (p % 2 ? -1.0f : 0.0f);
            imatrix[i] = important ? 1.0f : 0.0f;
        }
        printf("  - %s\n", ggml_type_name(type));

        std::vector<uint8_t> q(ggml_row_size(type, n));
        ggml_quantize_init(type);
        ggml_quantize_chunk(type, x.data(), q.data(), 0, 1, n, imatrix.data());
        ggml_get_type_traits(type)->to_float(q.data(), out.data(), n);

        const bool failed = !std::all_of(out.begin(), out.end(), [](float v) { return std::isfinite(v); });
        num_failed += failed;
        if (failed || verbose) {
            printf("%5s imatrix degenerate groups:      %s\n", ggml_type_name(type), RESULT_STR[failed]);
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
        { GGML_TYPE_NVFP4,  0, { 64, 192, 640, 2560 } },     // [TAG_R38_E2] raw UE4M3 sub-block scales
    };

    const int saved_mr   = ggml_cpu_fn_get_switch(GGML_CPU_FN_MMID_MR);
    const int saved_q5_1 = ggml_cpu_fn_get_switch(GGML_CPU_FN_Q5_1_AVX512);
    const int saved_vnni = ggml_cpu_fn_get_switch(GGML_CPU_FN_VNNI); // [TAG_FN_R4_VNNI]
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

            // [TAG_FN_R4_VNNI] body 3: GGML_CPU_MMID_MR=1 with GGML_CPU_VNNI=1 (the VNNI bodies of q4_K, q5_1 and iq4_nl on a
            // VNNI CPU; elsewhere the maddubs bodies run again)
            for (int body : {1, 2, 3}) {
                ggml_cpu_fn_set_switch(GGML_CPU_FN_MMID_MR, body == 3 ? 1 : body);
                ggml_cpu_fn_set_switch(GGML_CPU_FN_VNNI,   body == 3 ? 1 : 0);
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
    ggml_cpu_fn_set_switch(GGML_CPU_FN_VNNI, saved_vnni);

    if (num_failed || verbose) {
        printf(" mmid_mr kernels: %d values, %d not bitwise equal to vec_dot (VNNI bodies %s)\n", n_values, num_failed,
               ggml_cpu_fn_vnni_available() ? "checked" : "not available: body 3 ran the maddubs bodies");
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
    num_failed += test_f8_e4m3(verbose);
    num_failed += test_vec_dot_q(verbose);
    num_failed += test_quantize_imatrix_degenerate(verbose);
    num_failed += test_fn_q5_1_avx512(verbose); // [TAG_FN_CPU_Q5_1_AVX512]
    num_failed += test_fn_mmid_mr(verbose);     // [TAG_FN_CPU_MMID_MR]

    if (num_failed || verbose) {
        printf("%d tests failed\n", num_failed);
    }

    return num_failed > 0;
}
