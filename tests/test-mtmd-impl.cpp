#include "testing.h"

#include "mtmd-image.h"
#include "mtmd-internal.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// this test file contains:
// 1. test cases for mtmd helpers
// 2. test cases for internal mtmd components
// internal headers can be included here

struct test_registry {
    using fn_t = void (*)(testing &);

    struct entry {
        std::string name;
        fn_t fn;
    };

    static std::vector<entry> & all() {
        static std::vector<entry> entries;
        return entries;
    }

    test_registry(const char * name, fn_t fn) {
        all().push_back({ name, fn });
    }
};

#define MAKE_TEST(name)                                               \
    static void name(testing & t);                                    \
    static const test_registry test_registry_ ## name(#name, &name);  \
    static void name(testing & t)


//
// mtmd_image
//

MAKE_TEST(test_image_preprocessor_lfm2) {
    clip_hparams hparams;
    hparams.patch_size = 16;
    hparams.n_merge = 2;
    hparams.set_limit_image_tokens(64, 256);

    // { image size, expected tiling }
    const std::vector<std::pair<clip_image_size, bool>> cases = {
        { {  704, 704 }, false },
        // 720 / (patch_size * n_merge) is exactly 22.5, so this only matches HF
        // if round_by_factor rounds half to even (22) instead of away from zero (23)
        { {  720, 720 }, false },
        { {  736, 736 }, true  },
        { { 1024, 977 }, true  },
        { { 1056, 384 }, false },
    };

    for (const auto & [size, expected] : cases) {
        const bool actual = mtmd_image_preprocessor_lfm2::should_tile(hparams, size);

        t.assert_equal(
            "tiling for " + std::to_string(size.width) + "x" + std::to_string(size.height),
            std::string(expected ? "tiled" : "single"),
            std::string(actual   ? "tiled" : "single"));
    }
}

// [TAG_MTMD_EXTREME_ASPECT] dyn_size (Qwen2/2.5/3-VL, ...) target sizes. Normal images keep upstream's smart_resize
// size exactly; an extreme aspect ratio (1x20000) must give a size resize_pillow() accepts (sides <= 65536) inside the
// token budget, instead of the "resize target 8x144832 is out of range" refusal.
MAKE_TEST(test_image_preprocessor_dyn_size_extreme_aspect) {
    clip_hparams hparams;
    hparams.patch_size = 16;
    hparams.n_merge = 2;
    hparams.set_limit_image_tokens(1024, 4096); // the Qwen3.8 server setting (--image-min/max-tokens 1024/4096)
    const int align = 32;

    // upstream's calc_size_preserved_ratio (no longest_edge), without the side limit
    auto reference = [&](const clip_image_size & in) {
        auto round_f = [&](float x) { return static_cast<int>(std::round(x / static_cast<float>(align))) * align; };
        auto ceil_f  = [&](float x) { return static_cast<int>(std::ceil(x / static_cast<float>(align))) * align; };
        auto floor_f = [&](float x) { return static_cast<int>(std::floor(x / static_cast<float>(align))) * align; };
        int w_bar = std::max(align, round_f(in.width));
        int h_bar = std::max(align, round_f(in.height));
        if (h_bar * w_bar > hparams.image_max_pixels) {
            const auto beta = std::sqrt(static_cast<float>(in.height) * in.width / hparams.image_max_pixels);
            h_bar = std::max(align, floor_f(in.height / beta));
            w_bar = std::max(align, floor_f(in.width / beta));
        } else if (h_bar * w_bar < hparams.image_min_pixels) {
            const auto beta = std::sqrt(static_cast<float>(hparams.image_min_pixels) / (static_cast<float>(in.height) * in.width));
            h_bar = ceil_f(in.height * beta);
            w_bar = ceil_f(in.width * beta);
        }
        return std::to_string(w_bar) + "x" + std::to_string(h_bar);
    };
    auto str = [](const clip_image_size & s) { return std::to_string(s.width) + "x" + std::to_string(s.height); };

    const std::vector<clip_image_size> normal = {
        { 2560, 1600 }, { 640, 400 }, { 1920, 1080 }, { 4000, 3000 }, { 1080, 20000 }, { 20000, 1080 },
        { 32, 60000 }, { 1, 3000 }, { 3, 4 }, { 1, 1 },
    };
    for (const auto & in : normal) {
        t.assert_equal("dyn_size " + str(in) + " keeps the smart_resize size", reference(in),
                       str(mtmd_image_preprocessor_dyn_size::calc_target_size(hparams, in)));
    }

    const std::vector<clip_image_size> extreme = {
        { 1, 20000 }, { 20000, 1 }, { 32, 70000 }, { 70000, 32 }, { 2, 100000 }, { 1, 10000000 },
    };
    for (const auto & in : extreme) {
        const clip_image_size out = mtmd_image_preprocessor_dyn_size::calc_target_size(hparams, in);
        const long long px = (long long) out.width * out.height;
        const bool ok = out.width >= align && out.height >= align && out.width <= 65536 && out.height <= 65536 &&
                        out.width % align == 0 && out.height % align == 0 && px <= hparams.image_max_pixels;
        t.assert_true("dyn_size " + str(in) + " -> " + str(out) + " is encodable (sides 32..65536, <= max_pixels)", ok);
    }
    t.assert_equal("dyn_size 1x20000", std::string("32x65536"),
                   str(mtmd_image_preprocessor_dyn_size::calc_target_size(hparams, { 1, 20000 })));
}

//
// mtmd temporal merge
//

MAKE_TEST(test_temporal_merge_grouping) {
    std::vector<mtmd::bitmap_ptr> pool; // keeps the bitmaps alive until the end of the test

    // spec chars:
    //   v = video frame, w = video frame of another size, a = audio, i = plain image, t = text
    auto make_parts = [&pool](const std::string & spec) {
        std::vector<mtmd_internal_part> parts;
        for (char c : spec) {
            if (c == 't') {
                parts.push_back({ "hello", nullptr });
                continue;
            }
            mtmd_bitmap * bm = nullptr;
            switch (c) {
                case 'v': bm = mtmd_bitmap_init(100, 100, nullptr);   break;
                case 'w': bm = mtmd_bitmap_init(200, 200, nullptr);   break;
                case 'a': bm = mtmd_bitmap_init_from_audio(100, nullptr); break;
                case 'i': bm = mtmd_bitmap_init(100, 100, nullptr);   break;
                default: throw std::runtime_error(std::string("unknown spec char: ") + c);
            }
            mtmd_bitmap_set_mergeable(bm, c != 'i');
            pool.emplace_back(bm);
            parts.push_back({ "", bm });
        }
        return parts;
    };

    // { parts, n_merge, expected size of each group }
    const std::vector<std::tuple<std::string, int, std::string>> cases = {
        { "vv",   2, "2"    },
        { "vvv",  2, "21"   },
        { "vvvv", 2, "22"   },
        { "vvi",  2, "21"   },
        { "tvvt", 2, "2"    },
        { "vtv",  2, "11"   }, // text in between breaks the merge
        { "vw",   2, "11"   }, // different sizes cannot be merged
        { "aa",   2, "11"   }, // audio is never merged
        { "ii",   2, "11"   }, // two unrelated images must stay separated
        { "iv",   2, "11"   },
        { "vi",   2, "11"   },
        { "vv",   1, "11"   }, // model without temporal merge
    };

    for (const auto & [spec, n_merge, expected] : cases) {
        auto parts  = make_parts(spec);
        auto groups = mtmd_group_mergeable_bitmaps(parts, n_merge);

        std::string actual;
        for (const auto & group : groups) {
            actual += std::to_string(group.size());
        }

        const std::string name = "\"" + spec + "\" with n_merge=" + std::to_string(n_merge);
        t.assert_equal("groups for " + name, expected, actual);

        size_t n_bitmap_parts = 0;
        for (const auto & p : parts) {
            n_bitmap_parts += p.bitmap != nullptr ? 1 : 0;
        }
        t.assert_equal("remaining bitmap parts for " + name, groups.size(), n_bitmap_parts);
    }
}

//
// main
//

int main(int argc, char ** argv) {
    testing t(std::cout);
    t.verbose = true;

    // usage: test-mtmd-impl [filter_regex]
    for (int i = 1; i < argc; i++) {
        t.set_filter(argv[i]);
    }

    for (const auto & e : test_registry::all()) {
        t.test(e.name, e.fn);
    }

    return t.summary();
}
