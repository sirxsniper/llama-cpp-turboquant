// [TAG_UPSTREAM_30969] ggml_time_ticks_scale: the Windows timer's tick -> ms / us conversion at long uptimes. CPU only, no model.
#include "../ggml/src/ggml-impl.h"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <initializer_list>

static int g_fail   = 0;
static int g_checks = 0;

#define TCHECK(cond, ...)                      \
    do {                                       \
        g_checks++;                            \
        if (!(cond)) {                         \
            g_fail++;                          \
            fprintf(stderr, "FAIL: ");         \
            fprintf(stderr, __VA_ARGS__);      \
            fprintf(stderr, "\n");             \
        }                                      \
    } while (0)

int main() {
    const int64_t f10 = 10000000;       // 10 MHz, the usual QueryPerformanceCounter rate
    const int64_t day = 86400;
    // 11 days: ticks * 1e6 = 9.5e18 > INT64_MAX (the old formula went negative here)
    const int64_t t11 = 11 * day * f10 + 1234567;
    TCHECK(ggml_time_ticks_scale(t11, f10, 1000000) == 11 * day * 1000000 + 123456, "11 days in us");
    TCHECK(ggml_time_ticks_scale(t11, f10, 1000) == 11 * day * 1000 + 123, "11 days in ms");
    // 1 year at 10 MHz and at a 3 GHz counter (TSC based)
    const int64_t f3g = 3000000000LL;
    TCHECK(ggml_time_ticks_scale(365 * day * f10, f10, 1000000) == 365 * day * 1000000, "1 year in us at 10 MHz");
    TCHECK(ggml_time_ticks_scale(365 * day * f3g + f3g / 2, f3g, 1000000) == 365 * day * 1000000 + 500000, "1 year in us at 3 GHz");
    // short times: the same values as the old formula
    for (int64_t t : { (int64_t) 0, (int64_t) 1, (int64_t) 9, (int64_t) 12345, f10 - 1, f10, 3 * f10 + 7, 600 * f10 + 99999 }) {
        TCHECK(ggml_time_ticks_scale(t, f10, 1000000) == (t * 1000000) / f10, "us at %lld ticks", (long long) t);
        TCHECK(ggml_time_ticks_scale(t, f10, 1000) == (t * 1000) / f10, "ms at %lld ticks", (long long) t);
    }
    // the live timer still runs forward
    ggml_time_init();
    const int64_t a = ggml_time_us();
    const int64_t b = ggml_time_us();
    TCHECK(a >= 0 && b >= a, "ggml_time_us monotonic (%lld, %lld)", (long long) a, (long long) b);
    printf("test-fn-l24-time: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
