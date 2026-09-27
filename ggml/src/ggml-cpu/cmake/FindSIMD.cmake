include(CheckCSourceRuns)

set(AVX_CODE "
    #include <immintrin.h>
    int main()
    {
        __m256 a;
        a = _mm256_set1_ps(0);
        return 0;
    }
")

set(AVX512_CODE "
    #include <immintrin.h>
    int main()
    {
        __m512i a = _mm512_set_epi8(0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0);
        __m512i b = a;
        __mmask64 equality_mask = _mm512_cmp_epi8_mask(a, b, _MM_CMPINT_EQ);
        return 0;
    }
")

set(AVX2_CODE "
    #include <immintrin.h>
    int main()
    {
        __m256i a = {0};
        a = _mm256_abs_epi16(a);
        __m256i x;
        _mm256_extract_epi64(x, 0); // we rely on this in our AVX2 code
        return 0;
    }
")

set(FMA_CODE "
    #include <immintrin.h>
    int main()
    {
        __m256 acc = _mm256_setzero_ps();
        const __m256 d = _mm256_setzero_ps();
        const __m256 p = _mm256_setzero_ps();
        acc = _mm256_fmadd_ps( d, p, acc );
        return 0;
    }
")

macro(check_sse type flags)
    set(__FLAG_I 1)
    set(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
    foreach (__FLAG ${flags})
        if (NOT ${type}_FOUND)
            set(CMAKE_REQUIRED_FLAGS ${__FLAG})
            check_c_source_runs("${${type}_CODE}" HAS_${type}_${__FLAG_I})
            if (HAS_${type}_${__FLAG_I})
                set(${type}_FOUND TRUE CACHE BOOL "${type} support")
                set(${type}_FLAGS "${__FLAG}" CACHE STRING "${type} flags")
            endif()
            math(EXPR __FLAG_I "${__FLAG_I}+1")
        endif()
    endforeach()
    set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})

    if (NOT ${type}_FOUND)
        set(${type}_FOUND FALSE CACHE BOOL "${type} support")
        set(${type}_FLAGS "" CACHE STRING "${type} flags")
    endif()

    mark_as_advanced(${type}_FOUND ${type}_FLAGS)
endmacro()

# flags are for MSVC only!
check_sse("AVX" " ;/arch:AVX")
if (NOT ${AVX_FOUND})
    set(GGML_AVX OFF)
else()
    set(GGML_AVX ON)
endif()

check_sse("AVX2" " ;/arch:AVX2")
check_sse("FMA" " ;/arch:AVX2")
if ((NOT ${AVX2_FOUND}) OR (NOT ${FMA_FOUND}))
    set(GGML_AVX2 OFF)
else()
    set(GGML_AVX2 ON)
endif()

check_sse("AVX512" " ;/arch:AVX512")
if (NOT ${AVX512_FOUND})
    set(GGML_AVX512 OFF)
else()
    set(GGML_AVX512 ON)
endif()

# [TAG_Q2_0_CPU] MSVC defines no macros for the AVX512 extensions or BMI2, so /arch:AVX512 alone leaves their code paths
# off. Opt-in: run a probe for each on the build machine and turn on the ones it has. VNNI, VBMI and BMI2 are integer
# instructions (same results expected); BF16 has its own switch as its dot product rounds differently.
set(AVX512_VNNI_CODE "
    #include <immintrin.h>
    int main(int argc, char ** argv)
    {
        (void) argv;
        __m512i a = _mm512_set1_epi8((char) argc);
        __m512i c = _mm512_dpbusd_epi32(_mm512_setzero_si512(), a, a);
        return _mm_cvtsi128_si32(_mm512_castsi512_si128(c)) == 4*argc*argc ? 0 : 1;
    }
")

set(AVX512_VBMI_CODE "
    #include <immintrin.h>
    int main(int argc, char ** argv)
    {
        (void) argv;
        __m512i a = _mm512_set1_epi8((char) argc);
        __m512i c = _mm512_multishift_epi64_epi8(_mm512_setzero_si512(), a);
        return (_mm_cvtsi128_si32(_mm512_castsi512_si128(c)) & 0xFF) == argc ? 0 : 1;
    }
")

set(AVX512_BF16_CODE "
    #include <immintrin.h>
    int main(int argc, char ** argv)
    {
        (void) argv;
        __m512 a = _mm512_set1_ps((float) argc);
        __m512 c = _mm512_dpbf16_ps(_mm512_setzero_ps(), _mm512_cvtne2ps_pbh(a, a), _mm512_cvtne2ps_pbh(a, a));
        return _mm_cvtss_f32(_mm512_castps512_ps128(c)) == 2.0f*argc*argc ? 0 : 1;
    }
")

set(BMI2_CODE "
    #include <immintrin.h>
    int main(int argc, char ** argv)
    {
        (void) argv;
        return _pdep_u32((unsigned) argc, 0xFFu) == (unsigned) argc ? 0 : 1;
    }
")

if (GGML_NATIVE_MSVC_EXT)
    check_sse("BMI2" " ;/arch:AVX2")
    if (${BMI2_FOUND})
        set(GGML_BMI2 ON)
    endif()
    if (${AVX512_FOUND})
        check_sse("AVX512_VNNI" " ;/arch:AVX512")
        if (${AVX512_VNNI_FOUND})
            set(GGML_AVX512_VNNI ON)
        endif()
        check_sse("AVX512_VBMI" " ;/arch:AVX512")
        if (${AVX512_VBMI_FOUND})
            set(GGML_AVX512_VBMI ON)
        endif()
    endif()
    message(STATUS "GGML_NATIVE_MSVC_EXT: BMI2 ${BMI2_FOUND}, AVX512-VNNI ${AVX512_VNNI_FOUND}, AVX512-VBMI ${AVX512_VBMI_FOUND}")
endif()

if (GGML_NATIVE_MSVC_BF16 AND ${AVX512_FOUND})
    check_sse("AVX512_BF16" " ;/arch:AVX512")
    if (${AVX512_BF16_FOUND})
        set(GGML_AVX512_BF16 ON)
    endif()
    message(STATUS "GGML_NATIVE_MSVC_BF16: AVX512-BF16 ${AVX512_BF16_FOUND}")
endif()
