#include "testing.h"

#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "mtmd-audio.h"
#include "mtmd-subcontext.h"
#include "mtmd-helper.h"
#include "mtmd-image.h"
#include "mtmd-internal.h"
#include "mtmd-fork.h" // [TAG_FN_L22_VLBATCH] [TAG_R39_MTMD]
#include "mtmd-impl.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <random>
#include <cstdint>
#include <cstdlib>
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

MAKE_TEST(test_image_preprocessor_longest_edge) {
    mtmd_hparams hparams;
    hparams.image_longest_edge = 1540;
    hparams.patch_size = 14;
    hparams.n_merge = 2;

    const std::vector<std::pair<mtmd_image_size, mtmd_image_size>> cases = {
        { {  463,  463 }, {  476,  476 } },
        { {  463,  300 }, {  476,  308 } },
        { {  448,  280 }, {  448,  280 } },
        { { 2000, 1000 }, { 1540,  784 } },
    };

    for (const auto & [input, expected] : cases) {
        const mtmd_image_size actual = mtmd_image_preprocessor_longest_edge::get_target_size(hparams, input);
        const std::string name = std::to_string(input.width) + "x" + std::to_string(input.height);
        t.assert_equal(name + " width", expected.width, actual.width);
        t.assert_equal(name + " height", expected.height, actual.height);
    }
}

MAKE_TEST(test_image_preprocessor_dyn_size) {
    // { patch_size, image size, expected size }
    // a side that is an even number plus .5 merged patches rounds down, like Python round() in smart_resize
    const std::vector<std::tuple<int, mtmd_image_size, mtmd_image_size>> cases = {
        { 16, {  768, 400 }, {  768, 384 } },
        { 16, {  800, 336 }, {  800, 320 } },
        { 16, {  400, 400 }, {  384, 384 } },
        { 16, {  640, 240 }, {  640, 256 } },
        { 14, {  966, 560 }, {  952, 560 } },
    };

    for (const auto & [patch_size, input, expected] : cases) {
        mtmd_hparams hparams;
        hparams.patch_size = patch_size;
        hparams.n_merge = 2;
        hparams.image_round_half_even = true;
        hparams.set_limit_image_tokens(8, 4096);

        const mtmd_image_size actual = mtmd_image_preprocessor_dyn_size::get_target_size(hparams, input);
        const std::string name = std::to_string(input.width) + "x" + std::to_string(input.height);
        t.assert_equal(name + " width", expected.width, actual.width);
        t.assert_equal(name + " height", expected.height, actual.height);
    }
}

MAKE_TEST(test_image_preprocessor_qwen3vl_video) {
    mtmd_hparams hparams;
    hparams.patch_size = 16;
    hparams.n_merge = 2;
    const mtmd_image_preprocessor_qwen3vl preproc(hparams);

    // { n_frames, frame size, expected size from Qwen3VLVideoProcessor smart_resize }
    const std::vector<std::tuple<int, mtmd_image_size, mtmd_image_size>> cases = {
        {  20, {  720,  358 }, {  704,  352 } }, // 720 / 32 = 22.5 rounds half to even
        { 768, { 1920, 1080 }, {  224,  128 } }, // over the longest_edge budget
        {   4, {   16,   40 }, {   32,   64 } }, // edge below the factor is upscaled first
        {   2, {   32,   32 }, {   64,   64 } }, // under the shortest_edge budget
        {   5, { 2760, 1552 }, { 2752, 1536 } }, // 5 / 2 = 2.5 rounds half to even
        {   4, {  250, 6000 }, {  256, 6016 } },
    };

    for (const auto & [n_frames, input, expected] : cases) {
        mtmd_image_size actual;
        const std::string name = std::to_string(n_frames) + " frames of " + std::to_string(input.width) + "x" + std::to_string(input.height);
        if (!t.assert_true(name + " has a frame size", preproc.get_video_frame_size(n_frames, input.width, input.height, actual))) {
            continue;
        }
        t.assert_equal(name + " width", expected.width, actual.width);
        t.assert_equal(name + " height", expected.height, actual.height);
    }
}

static std::string image_size_to_str(const mtmd_image_size & size) {
    return std::to_string(size.width) + "x" + std::to_string(size.height);
}

static std::string resize_algo_to_str(resize_algo algo) {
    std::string str;
    switch (algo & ~RESIZE_ALGO_FLAG_TORCH) {
        case RESIZE_ALGO_BILINEAR: str = "bilinear"; break;
        case RESIZE_ALGO_BICUBIC:  str = "bicubic";  break;
        case RESIZE_ALGO_LANCZOS:  str = "lanczos";  break;
        default:                   str = "unknown";  break;
    }
    if (algo & RESIZE_ALGO_FLAG_TORCH) {
        str += "_torch";
    }
    return str;
}

// pseudo-random pixels: s = s * 1103515245 + 12345 from s = 1, one byte (s >> 16) & 255 per channel
static mtmd_image_u8 make_test_image(const mtmd_image_size & size) {
    mtmd_image_u8 img;
    img.set_size(size, false);
    uint32_t s = 1;
    auto next = [&s]() {
        s = s * 1103515245u + 12345u;
        return static_cast<uint8_t>(s >> 16);
    };
    for (int y = 0; y < size.height; y++) {
        for (int x = 0; x < size.width; x++) {
            img.set_pixel(x, y, { next(), next(), next() });
        }
    }
    return img;
}

// FNV-1a of the pixels
static uint32_t hash_pixels(const mtmd_image_u8 & img, uint32_t hash = 2166136261u) {
    for (uint8_t v : img.get_ro_buf()) {
        hash = (hash ^ v) * 16777619u;
    }
    return hash;
}

MAKE_TEST(test_image_preprocessor_lfm2) {
    mtmd_hparams hparams;
    hparams.patch_size = 16;
    hparams.n_merge = 2;
    hparams.set_limit_image_tokens(64, 256);
    hparams.preproc_min_tiles            = 2;
    hparams.preproc_max_tiles            = 10;
    hparams.preproc_tile_size            = 512;
    hparams.preproc_max_pixels_tolerance = 2.0f;

    // { image size, expected tiling }
    const std::vector<std::pair<mtmd_image_size, bool>> cases = {
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

    // { image size, expected size from HF smart_resize }
    const std::vector<std::pair<mtmd_image_size, mtmd_image_size>> resize_cases = {
        { {  400, 600 }, { 384, 608 } }, // 400 / 32 = 12.5 rounds half to even
        { {  480, 336 }, { 480, 320 } },
        { {   64, 169 }, { 160, 416 } }, // 169 * beta / 32 is exactly 13 in double
        { { 1024, 768 }, { 576, 416 } },
    };

    for (const auto & [size, expected] : resize_cases) {
        t.assert_equal("smart_resize for " + image_size_to_str(size), image_size_to_str(expected),
            image_size_to_str(mtmd_image_preprocessor_lfm2::smart_resize(hparams, size)));
    }

    // { image size, expected grid from HF }
    const std::vector<std::pair<mtmd_image_size, mtmd_image_size>> grid_cases = {
        { {  720, 1600 }, { 2, 4 } }, // 0.45 is an exact tie between 2/4 and 2/5 in double, not in float
        { {  420, 1440 }, { 1, 3 } },
        { {  255, 2160 }, { 1, 8 } },
        { { 1024,  768 }, { 3, 2 } },
    };

    for (const auto & [size, expected] : grid_cases) {
        t.assert_equal("grid for " + image_size_to_str(size), image_size_to_str(expected),
            image_size_to_str(mtmd_image_preprocessor_lfm2::get_grid_layout(hparams, size.height, size.width)));
    }

    // do_image_splitting=false
    hparams.preproc_min_tiles = 1;
    hparams.preproc_max_tiles = 1;
    t.assert_equal("tiling for 1024x977 with max_tiles=1", std::string("single"),
            std::string(mtmd_image_preprocessor_lfm2::should_tile(hparams, { 1024, 977 }) ? "tiled" : "single"));
}

MAKE_TEST(test_image_preprocessor_lfm2_torch_resize) {
    mtmd_hparams hparams;
    hparams.patch_size = 16;
    hparams.n_merge = 2;
    hparams.set_limit_image_tokens(64, 256);
    hparams.preproc_min_tiles            = 2;
    hparams.preproc_max_tiles            = 10;
    hparams.preproc_tile_size            = 512;
    hparams.preproc_max_pixels_tolerance = 2.0f;
    // Lfm2VlImageProcessor stretches the image to the tile grid
    hparams.image_pad_rf = PAD_NONE;

    // { resize algo, image size, expected hash of the tiles and the thumbnail from Lfm2VlImageProcessor (torchvision 0.29, native uint8 resize: arm64, or x86 with AVX2/AVX512) }
    const std::vector<std::tuple<resize_algo, mtmd_image_size, uint32_t>> cases = {
        { RESIZE_ALGO_BILINEAR, {  620,  410 }, 0xf5e9c941 }, // 608x416
        { RESIZE_ALGO_BILINEAR, { 2000, 1500 }, 0x3331f56c }, // 3x2 tiles, 576x416 thumbnail
        { RESIZE_ALGO_BICUBIC,  {  620,  410 }, 0x2b34eac9 },
        { RESIZE_ALGO_BICUBIC,  { 2000, 1500 }, 0xb035d942 },
    };

    for (const auto & [base_algo, size, expected] : cases) {
        const resize_algo algo = static_cast<resize_algo>(base_algo | RESIZE_ALGO_FLAG_TORCH);
        hparams.image_resize_algo_rf = algo;
        hparams.image_resize_algo_ov = algo;
        const mtmd_image_preprocessor_lfm2 preproc(hparams);
        const auto sliced = preproc.slice_image(make_test_image(size), preproc.get_slice_instructions(size));

        uint32_t hash = 2166136261u;
        for (const auto & slice : sliced.slices) {
            hash = hash_pixels(slice, hash);
        }
        hash = hash_pixels(sliced.overview, hash);
        t.assert_equal("pixels of " + std::to_string(size.width) + "x" + std::to_string(size.height) + " with resize algo " + resize_algo_to_str(algo), expected, hash);
    }
}

// [TAG_MTMD_EXTREME_ASPECT] dyn_size (Qwen2/2.5/3-VL, ...) target sizes. Normal images keep upstream's smart_resize
// size exactly; an extreme aspect ratio (1x20000) must give a size resize_pillow() accepts (sides <= 65536) inside the
// token budget, instead of the "resize target 8x144832 is out of range" refusal.
MAKE_TEST(test_image_preprocessor_dyn_size_extreme_aspect) {
    mtmd_hparams hparams;
    hparams.patch_size = 16;
    hparams.n_merge = 2;
    hparams.set_limit_image_tokens(1024, 4096); // the Qwen3.8 server setting (--image-min/max-tokens 1024/4096)
    const int align = 32;

    // upstream's calc_size_preserved_ratio (no longest_edge), without the side limit
    auto reference = [&](const mtmd_image_size & in) {
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
    auto str = [](const mtmd_image_size & s) { return std::to_string(s.width) + "x" + std::to_string(s.height); };

    const std::vector<mtmd_image_size> normal = {
        { 2560, 1600 }, { 640, 400 }, { 1920, 1080 }, { 4000, 3000 }, { 1080, 20000 }, { 20000, 1080 },
        { 32, 60000 }, { 1, 3000 }, { 3, 4 }, { 1, 1 },
    };
    for (const auto & in : normal) {
        t.assert_equal("dyn_size " + str(in) + " keeps the smart_resize size", reference(in),
                       str(mtmd_image_preprocessor_dyn_size::calc_target_size(hparams, in)));
    }

    const std::vector<mtmd_image_size> extreme = {
        { 1, 20000 }, { 20000, 1 }, { 32, 70000 }, { 70000, 32 }, { 2, 100000 }, { 1, 10000000 },
    };
    for (const auto & in : extreme) {
        const mtmd_image_size out = mtmd_image_preprocessor_dyn_size::calc_target_size(hparams, in);
        const long long px = (long long) out.width * out.height;
        const bool ok = out.width >= align && out.height >= align && out.width <= 65536 && out.height <= 65536 &&
                        out.width % align == 0 && out.height % align == 0 && px <= hparams.image_max_pixels;
        t.assert_true("dyn_size " + str(in) + " -> " + str(out) + " is encodable (sides 32..65536, <= max_pixels)", ok);
    }
    t.assert_equal("dyn_size 1x20000", std::string("32x65536"),
                   str(mtmd_image_preprocessor_dyn_size::calc_target_size(hparams, { 1, 20000 })));
}

// [TAG_FN_L22_VLRESIZE] upstream #30736: with image_round_half_even a side of an even number plus .5 merged patches rounds
// down like Python round() in smart_resize; off keeps std::round (the old sizes)
MAKE_TEST(test_image_preprocessor_dyn_size_l22_vlresize) {
    // { patch_size, image size, expected with round-half-even, expected without }
    const std::vector<std::tuple<int, mtmd_image_size, mtmd_image_size, mtmd_image_size>> cases = {
        { 16, {  768, 400 }, {  768, 384 }, {  768, 416 } },
        { 16, {  800, 336 }, {  800, 320 }, {  800, 352 } },
        { 16, {  400, 400 }, {  384, 384 }, {  416, 416 } },
        { 16, {  640, 240 }, {  640, 256 }, {  640, 256 } },
        { 14, {  966, 560 }, {  952, 560 }, {  980, 560 } },
    };

    for (const auto & [patch_size, input, expected_even, expected_old] : cases) {
        for (const bool even : { true, false }) {
            mtmd_hparams hparams;
            hparams.patch_size = patch_size;
            hparams.n_merge = 2;
            hparams.image_round_half_even = even;
            hparams.set_limit_image_tokens(8, 4096);

            const mtmd_image_size actual   = mtmd_image_preprocessor_dyn_size::calc_target_size(hparams, input);
            const mtmd_image_size expected = even ? expected_even : expected_old;
            const std::string name = std::to_string(input.width) + "x" + std::to_string(input.height) + (even ? " even" : " old");
            t.assert_equal(name + " width", expected.width, actual.width);
            t.assert_equal(name + " height", expected.height, actual.height);
        }
    }
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
// mtmd_helper video
//

MAKE_TEST(test_video_config) {
    mtmd_hparams hparams;
    hparams.patch_size = 16;
    hparams.n_merge = 2;
    const mtmd_image_preprocessor_qwen3vl preproc(hparams);

    // { template, n_frames and fps of the source, expected fps, n_frames and frame size }
    // the 1920x1080 sizes are from Qwen3VLVideoProcessor smart_resize
    const std::vector<std::tuple<mtmd_video_template, int32_t, float, float, int32_t, mtmd_image_size>> cases = {
        { MTMD_VIDEO_TEMPLATE_QWEN3VL,   36000, 30.0f, 2.0f, 768, { 224, 128 } },
        { MTMD_VIDEO_TEMPLATE_LFM3,      36000, 30.0f, 2.0f, 768, { 224, 128 } },
        { MTMD_VIDEO_TEMPLATE_LFM3_OMNI, 36000, 30.0f, 2.0f, 384, { 320, 192 } },
        { MTMD_VIDEO_TEMPLATE_QWEN3VL,      -1, 30.0f, 2.0f,   0, {   0,   0 } }, // frame count unknown
        { MTMD_VIDEO_TEMPLATE_DEFAULT,   36000, 30.0f, 0.0f,   0, {   0,   0 } },
    };

    for (const auto & [tmpl, n_frames, fps, expected_fps, expected_n_frames, expected_size] : cases) {
        const mtmd_video_config cfg = mtmd_video_config_init(tmpl, &preproc, 1920, 1080, n_frames, fps);
        const std::string name = "template " + std::to_string(tmpl) + " with " + std::to_string(n_frames) + " frames";
        t.assert_equal(name + " template", (int) tmpl, (int) cfg.tmpl);
        t.assert_equal(name + " fps", expected_fps, cfg.fps);
        t.assert_equal(name + " n_frames", expected_n_frames, cfg.n_frames);
        t.assert_equal(name + " frame size", image_size_to_str(expected_size), image_size_to_str({ cfg.frame_width, cfg.frame_height }));
    }
}

// n_frames source frames, each one starts with its index
struct test_video_source : mtmd_helper_video_seq {
    int32_t n_frames;
    int32_t n_read = 0;

    test_video_source(int32_t n_frames) : n_frames(n_frames) {}

    bool read_raw_frame() override {
        if (n_read >= n_frames) {
            return false;
        }
        frame_buf[0] = static_cast<uint8_t>(n_read & 255);
        frame_buf[1] = static_cast<uint8_t>(n_read >> 8);
        n_read++;
        return true;
    }
};

// text as is, a frame as [source index], an audio bitmap as {n_samples}
static std::string video_sequence_to_str(testing & t, mtmd_helper_video_seq & video) {
    std::string out;
    mtmd_bitmap * bitmap;
    char * text;
    int32_t ret;
    while ((ret = video.read_next(&bitmap, &text)) == 0) {
        if (text) {
            out += text;
            free(text);
        } else if (mtmd_bitmap_is_audio(bitmap)) {
            out += "{" + std::to_string(mtmd_bitmap_get_n_bytes(bitmap) / sizeof(float)) + "}";
            mtmd_bitmap_free(bitmap);
        } else {
            const unsigned char * data = mtmd_bitmap_get_data(bitmap);
            out += "[" + std::to_string(data[0] | data[1] << 8) + "]";
            mtmd_bitmap_free(bitmap);
        }
    }
    t.assert_equal("end of stream", -1, ret);
    return out;
}

MAKE_TEST(test_video_sequence_qwen3vl) {
    mtmd_hparams hparams;
    hparams.patch_size = 16;
    hparams.n_merge = 2;
    const mtmd_image_preprocessor_qwen3vl preproc(hparams);

    // { source video, expected frame size and sequence from Qwen3VLProcessor (transformers 4.57.1) }
    const std::vector<std::tuple<mtmd_helper_video_src, mtmd_image_size, std::string>> cases = {
        { {  720, 358, 24.0f, 240 }, { 704, 352 },
            "<0.3 seconds>[0][13]<1.3 seconds>[25][38]<2.4 seconds>[50][63]<3.4 seconds>[75][88]<4.5 seconds>[101][113]"
            "<5.5 seconds>[126][138]<6.6 seconds>[151][164]<7.6 seconds>[176][189]<8.6 seconds>[201][214]<9.7 seconds>[226][239]" },
        // odd frame count: the last group repeats its last frame
        { {  640, 480, 10.0f,  35 }, { 640, 480 },
            "<0.3 seconds>[0][6]<1.4 seconds>[11][17]<2.5 seconds>[23][28]<3.4 seconds>[34]" },
        // fewer frames than the minimum of 4
        { {  320, 240, 30.0f,   3 }, { 320, 256 },
            "<0.0 seconds>[0][1]<0.1 seconds>[2]" },
        { { 1280, 720, 30000.0f / 1001.0f, 300 }, { 1280, 704 },
            "<0.3 seconds>[0][16]<1.3 seconds>[31][47]<2.4 seconds>[63][79]<3.4 seconds>[94][110]<4.5 seconds>[126][142]"
            "<5.5 seconds>[157][173]<6.6 seconds>[189][205]<7.6 seconds>[220][236]<8.7 seconds>[252][268]<9.7 seconds>[283][299]" },
    };

    for (const auto & [src, expected_size, expected] : cases) {
        const std::string name = std::to_string(src.n_frames) + " frames of " + std::to_string(src.width) + "x" + std::to_string(src.height);
        const mtmd_video_config cfg = mtmd_video_config_init(MTMD_VIDEO_TEMPLATE_QWEN3VL, &preproc, src.width, src.height, src.n_frames, src.fps);
        test_video_source video(src.n_frames);
        video.configure(cfg, src, 0.0f);
        video.src_alive = true;
        t.assert_equal(name + " frame size", image_size_to_str(expected_size), image_size_to_str({ (int) video.info.width, (int) video.info.height }));
        t.assert_equal(name + " sequence", expected, video_sequence_to_str(t, video));
    }
}

// the soundtrack of each group comes after its frames
MAKE_TEST(test_video_sequence_lfm3_omni) {
    mtmd_hparams hparams;
    hparams.patch_size = 16;
    hparams.n_merge = 2;
    const mtmd_image_preprocessor_lfm3vl preproc(hparams);

    const mtmd_helper_video_src src = { 64, 64, 2.0f, 9 };
    const mtmd_video_config cfg = mtmd_video_config_init(MTMD_VIDEO_TEMPLATE_LFM3_OMNI, &preproc, src.width, src.height, src.n_frames, src.fps);
    test_video_source video(src.n_frames);
    video.configure(cfg, src, 0.0f);
    video.src_alive  = true;
    video.audio_rate = 100;
    video.soundtrack.assign(450, 0.0f);

    t.assert_equal("sequence",
        std::string("<0.2 seconds>[0][1]{100}<1.2 seconds>[2][3]{100}<2.2 seconds>[4][5]{100}<3.2 seconds>[6][7]{100}<4.0 seconds>[8]{50}"),
        video_sequence_to_str(t, video));
}

//
// mtmd_audio
//

MAKE_TEST(test_audio_preprocessor_conformer) {
    mtmd_hparams hparams;
    hparams.n_mel_bins        = 128;
    hparams.audio_sample_rate = 16000;
    hparams.audio_n_fft       = 512;
    hparams.audio_window_len  = 400;
    hparams.audio_hop_len     = 160;

    // 0.4 s of tones, 0.2 s of silence, 0.4 s of a quiet tone
    const double pi = 3.14159265358979323846;
    const int sr = hparams.audio_sample_rate;
    std::vector<float> samples(sr, 0.0f);
    for (int i = 0; i < sr; i++) {
        const double ts = (double) i / sr;
        if (i < 0.4 * sr) {
            samples[i] = (float) (0.3 * std::sin(2 * pi * 300 * ts) + 0.2 * std::sin(2 * pi * 1200 * ts) + 0.1 * std::sin(2 * pi * 3500 * ts));
        } else if (i >= 0.6 * sr) {
            samples[i] = (float) (1e-3 * std::sin(2 * pi * 800 * ts));
        }
    }

    // reference values from NeMo AudioToMelSpectrogramPreprocessor (liquid-audio 1.3.0)
    // { mel bin, frame, value }
    const std::vector<std::tuple<int, int, float>> cases = {
        {   8,  0,  3.6104f }, // tones
        {  24, 50, -0.6371f }, // silence
        {  24, 60, -0.0476f }, // silence -> quiet tone
        { 100, 20, -0.0913f }, // tones, high bin
    };

    ggml_backend_ptr backend(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    ggml_backend_t backends[] = { backend.get() };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 1, 256, false, true));

    // the result must not depend on the number of threads
    for (int n_threads : { 1, 4, 7 }) {
        mtmd_audio_preprocessor_conformer preproc(hparams, sched.get());
        preproc.mel_config.n_threads = n_threads;
        preproc.initialize();
        mtmd_audio_preproc_out out = preproc.preprocess(samples.data(), samples.size());
        if (!t.assert_true("preprocess", !out.entries.empty() && preproc.compute(out))) {
            return;
        }
        const mtmd_image_f32 & mel = out.entries[0];
        t.assert_equal("n_len", 101, mel.nx());

        for (const auto & [bin, frame, expected] : cases) {
            const float actual = mel.get_ro_buf()[bin * mel.nx() + frame];
            t.assert_true("mel[" + std::to_string(bin) + "][" + std::to_string(frame) + "] = " + std::to_string(actual) + ", expected " + std::to_string(expected), std::fabs(actual - expected) < 1e-3f);
        }
    }
}

MAKE_TEST(test_audio_preprocessor_threads) {
    mtmd_hparams hparams;
    hparams.n_mel_bins        = 128;
    hparams.audio_sample_rate = 16000;
    hparams.audio_n_fft       = 512;
    hparams.audio_window_len  = 400;
    hparams.audio_hop_len     = 160;

    // chirp: every frame has a different content
    std::vector<float> samples(hparams.audio_sample_rate * 5);
    for (size_t i = 0; i < samples.size(); i++) {
        samples[i] = (float) (0.3 * std::sin(0.05 * i + 2e-6 * i * i));
    }

    ggml_backend_ptr backend(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    ggml_backend_t backends[] = { backend.get() };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 1, 256, false, true));

    mtmd_audio_preprocessor_conformer preproc(hparams, sched.get());
    preproc.initialize();
    // tokenize only gets the shapes, encode must get the same ones
    mtmd_audio_preproc_out ref = preproc.preprocess(samples.data(), samples.size());
    if (!t.assert_true("preprocess", ref.entries.size() == 1 && ref.needs_compute())) {
        return;
    }
    const mtmd_image_size shape = ref.entries[0].get_size();
    if (!t.assert_true("compute", preproc.compute(ref))) {
        return;
    }
    t.assert_true("shape", ref.entries.size() == 1 && ref.entries[0].get_size() == shape);
    t.assert_true("computed", !ref.needs_compute());

    // frames are independent, so the whole output must be bit-exact with the single-threaded one
    for (int n_threads : { 4, 12 }) {
        preproc.mel_config.n_threads = n_threads;
        mtmd_audio_preproc_out out = preproc.preprocess(samples.data(), samples.size());
        const bool ok = preproc.compute(out);
        t.assert_true("n_threads = " + std::to_string(n_threads), ok && out.entries[0].get_ro_buf() == ref.entries[0].get_ro_buf());
    }
}

MAKE_TEST(test_audio_log_mel_precision) {
    const int n_mel    = 80;
    const int n_fft    = 400;
    const int hop      = 160;
    const int n_frames = 200;

    mtmd_audio_cache cache;
    cache.fill_sin_cos_table(n_fft);
    cache.fill_hann_window(n_fft, true);
    cache.fill_mel_filterbank_matrix(n_mel, n_fft, 16000);

    ggml_backend_ptr backend(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    ggml_backend_t backends[] = { backend.get() };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 1, 256, false, true));

    mtmd_audio_mel_config config;
    config.sched      = sched.get();
    config.n_threads  = 2;
    config.n_mel      = n_mel;
    config.n_fft_bins = 1 + n_fft / 2;
    config.hop_length = hop;
    config.cache      = &cache;

    // a loud tone next to a 74 dB weaker one: rounding the DFT inputs to F16 or TF32 loses the weak one
    const double pi = 3.14159265358979323846;
    std::vector<float> samples((size_t) (n_frames - 1) * hop + n_fft);
    for (size_t i = 0; i < samples.size(); i++) {
        samples[i] = (float) (0.5 * std::sin(2 * pi * 200 * i / 16000) + 1e-4 * std::sin(2 * pi * 6000 * i / 16000));
    }

    std::vector<float> out((size_t) n_frames * n_mel);
    if (!t.assert_true("log mel", mtmd_audio_log_mel(config, samples.data(), cache.hann_window.data(), n_frames, out.data()))) {
        return;
    }

    const int n_bins = n_fft / 2 + 1;
    double max_diff = 0.0;
    std::vector<double> power(n_bins);
    for (int i = 0; i < n_frames; i++) {
        for (int k = 0; k < n_bins; k++) {
            double re = 0.0, im = 0.0;
            for (int j = 0; j < n_fft; j++) {
                const double x = (double) cache.hann_window[j] * samples[(size_t) i * hop + j];
                re += x * std::cos(2 * pi * j * k / n_fft);
                im -= x * std::sin(2 * pi * j * k / n_fft);
            }
            power[k] = re * re + im * im;
        }
        for (int m = 0; m < n_mel; m++) {
            double sum = 0.0;
            for (int k = 0; k < n_bins; k++) {
                sum += power[k] * cache.filters.data[(size_t) m * n_bins + k];
            }
            const double expected = std::log10(std::max(sum, (double) config.mel_floor));
            max_diff = std::max(max_diff, std::fabs(expected - out[(size_t) m * n_frames + i]));
        }
    }
    t.assert_true("max_diff = " + std::to_string(max_diff), max_diff < 0.05);
}

MAKE_TEST(test_audio_log_mel_blocks) {
    const int n_mel    = 80;
    const int n_fft    = 400;
    const int hop      = 160;
    const int n_frames = 8192 + 100; // long audio is split into blocks of 8192 frames

    mtmd_audio_cache cache;
    cache.fill_sin_cos_table(n_fft);
    cache.fill_hann_window(n_fft, true);
    cache.fill_mel_filterbank_matrix(n_mel, n_fft, 16000);

    ggml_backend_ptr backend(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    ggml_backend_t backends[] = { backend.get() };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 1, 256, false, true));

    mtmd_audio_mel_config config;
    config.sched       = sched.get();
    config.n_threads   = 2;
    config.n_mel       = n_mel;
    config.n_fft_bins  = 1 + n_fft / 2;
    config.hop_length  = hop;
    config.cache       = &cache;

    std::vector<float> samples((size_t) (n_frames - 1) * hop + n_fft);
    for (size_t i = 0; i < samples.size(); i++) {
        samples[i] = (float) (0.3 * std::sin(0.05 * i + 1e-6 * i * i));
    }

    // frames are independent: the frames around the seam must not change when computed on their own
    // not bit-exact, matmul kernels may round differently depending on the shape
    const int off   = 8192 - 50;
    const int n_sub = n_frames - off;
    std::vector<float> all((size_t) n_frames * n_mel);
    std::vector<float> sub((size_t) n_sub * n_mel);
    const float * window = cache.hann_window.data();
    if (!t.assert_true("all", mtmd_audio_log_mel(config, samples.data(), window, n_frames, all.data())) ||
        !t.assert_true("sub", mtmd_audio_log_mel(config, samples.data() + (size_t) off * hop, window, n_sub, sub.data()))) {
        return;
    }

    float max_diff = 0.0f;
    for (int j = 0; j < n_mel; j++) {
        for (int i = 0; i < n_sub; i++) {
            max_diff = std::max(max_diff, std::fabs(all[(size_t) j * n_frames + off + i] - sub[(size_t) j * n_sub + i]));
        }
    }
    t.assert_true("max_diff = " + std::to_string(max_diff), max_diff < 1e-5f);
}

//
// [TAG_FN_L22_VLBATCH] batched Qwen3-VL encode (round 22 P2b)
//

MAKE_TEST(test_fn_l22_vlbatch_admit) {
    auto still = [](int nx, int ny, int px_w, int px_h) {
        mtmd_fn_l22_vlb_info r;
        r.image     = true;
        r.n_entries = 1;
        r.nx        = nx;
        r.ny        = ny;
        r.pos       = 1;
        r.px_w      = px_w;
        r.px_h      = px_h;
        r.n_tokens  = nx * ny;
        return r;
    };
    const auto a = still(32, 32, 1024, 1024); // 1024 rows
    t.assert_equal("same-size stills", 0, mtmd_fn_l22_vlbatch_admit(a, a, 1024, 2116));
    t.assert_equal("exactly the cap", 0, mtmd_fn_l22_vlbatch_admit(a, a, 1024, 2048));
    t.assert_equal("over the cap", 2, mtmd_fn_l22_vlbatch_admit(a, a, 2048, 2116));
    auto pair = a;
    pair.n_entries = 2;
    t.assert_equal("video pair next", 3, mtmd_fn_l22_vlbatch_admit(a, pair, 1024, 1 << 20));
    t.assert_equal("video pair first", 3, mtmd_fn_l22_vlbatch_admit(pair, a, 1024, 1 << 20));
    t.assert_equal("other grid", 3, mtmd_fn_l22_vlbatch_admit(a, still(32, 26, 1024, 832), 1024, 1 << 20));
    t.assert_equal("same grid, other pixel size", 3, mtmd_fn_l22_vlbatch_admit(a, still(32, 32, 1024, 1056), 1024, 1 << 20));
    auto ph = a;
    ph.placeholder = true;
    t.assert_equal("placeholder", 3, mtmd_fn_l22_vlbatch_admit(a, ph, 1024, 1 << 20));
    t.assert_equal("placeholder first", 3, mtmd_fn_l22_vlbatch_admit(ph, a, 1024, 1 << 20));
    mtmd_fn_l22_vlb_info audio;
    audio.n_entries = 1;
    audio.n_tokens  = 100;
    t.assert_equal("audio", 3, mtmd_fn_l22_vlbatch_admit(a, audio, 1024, 1 << 20));
    auto other_pos = a;
    other_pos.pos = 0;
    t.assert_equal("other position type", 3, mtmd_fn_l22_vlbatch_admit(a, other_pos, 1024, 1 << 20));
}

MAKE_TEST(test_fn_l22_vlbatch_split_plan) {
    auto plan_str = [](int n, int kmax) {
        const auto plan = mtmd_fork_l22_split_plan(n, [kmax](int k) { return k <= kmax; });
        std::string s;
        for (const int k : plan) {
            s += (s.empty() ? "" : ",") + std::to_string(k);
        }
        return s;
    };
    t.assert_equal("4 of 4", std::string("4"), plan_str(4, 4));
    t.assert_equal("4, fits 3", std::string("2,2"), plan_str(4, 3));
    t.assert_equal("5, fits 2", std::string("2,2,1"), plan_str(5, 2));
    t.assert_equal("3, fits 1", std::string("1,1,1"), plan_str(3, 1));
    t.assert_equal("3, fits nothing", std::string("1,1,1"), plan_str(3, 0));
    t.assert_equal("1", std::string("1"), plan_str(1, 4));
    t.assert_equal("12, fits 5", std::string("4,4,4"), plan_str(12, 5));
    t.assert_equal("0", std::string(""), plan_str(0, 4));
}

// a tiny qwen3vl_merger mmproj (2 blocks, n_embd 32, 2 heads, patch 16) with seeded random weights
static bool fn_l22_write_qwen3vl_gguf(const std::string & path, bool deepstack) {
    const int n_embd = 32, n_ff = 64, n_layer = 2, patch = 16, image_size = 64, proj = 24, n_pos_grid = 16;
    ggml_init_params ip = { (size_t) 16 << 20, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    gguf_context * g = gguf_init_empty();
    std::mt19937 rng(1234 + (deepstack ? 1 : 0));

    gguf_set_val_str (g, "general.architecture", "clip");
    gguf_set_val_bool(g, "clip.has_vision_encoder", true);
    gguf_set_val_str (g, "clip.projector_type", "qwen3vl_merger");
    gguf_set_val_bool(g, "clip.use_gelu", true);
    gguf_set_val_u32 (g, "clip.vision.image_size", image_size);
    gguf_set_val_u32 (g, "clip.vision.patch_size", patch);
    gguf_set_val_u32 (g, "clip.vision.embedding_length", n_embd);
    gguf_set_val_u32 (g, "clip.vision.feed_forward_length", n_ff);
    gguf_set_val_u32 (g, "clip.vision.block_count", n_layer);
    gguf_set_val_u32 (g, "clip.vision.attention.head_count", 2);
    gguf_set_val_f32 (g, "clip.vision.attention.layer_norm_epsilon", 1e-6f);
    gguf_set_val_u32 (g, "clip.vision.projection_dim", proj);
    gguf_set_val_u32 (g, "clip.vision.spatial_merge_size", 2);
    const float mean[3] = { 0.5f, 0.5f, 0.5f };
    const float stdv[3] = { 0.5f, 0.5f, 0.5f };
    gguf_set_arr_data(g, "clip.vision.image_mean", GGUF_TYPE_FLOAT32, mean, 3);
    gguf_set_arr_data(g, "clip.vision.image_std",  GGUF_TYPE_FLOAT32, stdv, 3);

    auto add = [&](const std::string & name, ggml_type type, std::vector<int64_t> ne, float scale, float offset = 0.0f) {
        ggml_tensor * tt = ggml_new_tensor(ctx, type, (int) ne.size(), ne.data());
        ggml_set_name(tt, name.c_str());
        std::uniform_real_distribution<float> u(-scale, scale);
        const int64_t n = ggml_nelements(tt);
        for (int64_t i = 0; i < n; ++i) {
            const float v = offset + u(rng);
            if (type == GGML_TYPE_F32) {
                ((float *) tt->data)[i] = v;
            } else {
                ((ggml_fp16_t *) tt->data)[i] = ggml_fp32_to_fp16(v);
            }
        }
        gguf_add_tensor(g, tt);
    };

    add("v.patch_embd.weight",   GGML_TYPE_F16, { patch, patch, 3, n_embd }, 0.05f);
    add("v.patch_embd.weight.1", GGML_TYPE_F16, { patch, patch, 3, n_embd }, 0.05f);
    add("v.patch_embd.bias",     GGML_TYPE_F32, { n_embd }, 0.1f);
    add("v.position_embd.weight", GGML_TYPE_F32, { n_embd, n_pos_grid }, 0.5f);
    for (int il = 0; il < n_layer; ++il) {
        const std::string p = "v.blk." + std::to_string(il) + ".";
        add(p + "attn_qkv.weight", GGML_TYPE_F16, { n_embd, 3 * n_embd }, 0.2f);
        add(p + "attn_qkv.bias",   GGML_TYPE_F32, { 3 * n_embd }, 0.1f);
        add(p + "attn_out.weight", GGML_TYPE_F16, { n_embd, n_embd }, 0.2f);
        add(p + "attn_out.bias",   GGML_TYPE_F32, { n_embd }, 0.1f);
        add(p + "ffn_up.weight",   GGML_TYPE_F16, { n_embd, n_ff }, 0.2f);
        add(p + "ffn_up.bias",     GGML_TYPE_F32, { n_ff }, 0.1f);
        add(p + "ffn_down.weight", GGML_TYPE_F16, { n_ff, n_embd }, 0.2f);
        add(p + "ffn_down.bias",   GGML_TYPE_F32, { n_embd }, 0.1f);
        add(p + "ln1.weight", GGML_TYPE_F32, { n_embd }, 0.1f, 1.0f);
        add(p + "ln1.bias",   GGML_TYPE_F32, { n_embd }, 0.1f);
        add(p + "ln2.weight", GGML_TYPE_F32, { n_embd }, 0.1f, 1.0f);
        add(p + "ln2.bias",   GGML_TYPE_F32, { n_embd }, 0.1f);
    }
    if (deepstack) {
        add("v.deepstack.0.norm.weight", GGML_TYPE_F32, { 4 * n_embd }, 0.1f, 1.0f);
        add("v.deepstack.0.norm.bias",   GGML_TYPE_F32, { 4 * n_embd }, 0.1f);
        add("v.deepstack.0.fc1.weight",  GGML_TYPE_F16, { 4 * n_embd, 4 * n_embd }, 0.1f);
        add("v.deepstack.0.fc1.bias",    GGML_TYPE_F32, { 4 * n_embd }, 0.1f);
        add("v.deepstack.0.fc2.weight",  GGML_TYPE_F16, { 4 * n_embd, proj }, 0.1f);
        add("v.deepstack.0.fc2.bias",    GGML_TYPE_F32, { proj }, 0.1f);
    }
    add("v.post_ln.weight", GGML_TYPE_F32, { n_embd }, 0.1f, 1.0f);
    add("v.post_ln.bias",   GGML_TYPE_F32, { n_embd }, 0.1f);
    add("mm.0.weight", GGML_TYPE_F16, { 4 * n_embd, 4 * n_embd }, 0.1f);
    add("mm.0.bias",   GGML_TYPE_F32, { 4 * n_embd }, 0.1f);
    add("mm.2.weight", GGML_TYPE_F16, { 4 * n_embd, proj }, 0.1f);
    add("mm.2.bias",   GGML_TYPE_F32, { proj }, 0.1f);

    const bool ok = gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    return ok;
}

static mtmd_image_f32 fn_l22_random_image(std::mt19937 & rng, int nx, int ny) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> buf((size_t) nx * ny * 3);
    for (auto & v : buf) {
        v = u(rng);
    }
    mtmd_image_f32 img;
    img.set_size({ nx, ny }, false, false);
    img.cpy_buf(buf);
    return img;
}

static double fn_l22_max_abs_diff(const float * a, const float * b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) {
        m = std::max(m, (double) std::fabs(a[i] - b[i]));
    }
    return m;
}

MAKE_TEST(test_fn_l22_vlbatch_encode) {
    const double tol = 1e-5;
    for (const bool deepstack : { false, true }) {
        for (const auto fa : { MTMD_FLASH_ATTN_TYPE_AUTO, MTMD_FLASH_ATTN_TYPE_DISABLED }) {
            const std::string tag = std::string(deepstack ? "deepstack" : "plain") + (fa == MTMD_FLASH_ATTN_TYPE_DISABLED ? ", no FA" : ", FA auto");
            const std::string path = (std::filesystem::temp_directory_path() /
                ("test-fn-l22-vlbatch-" + std::to_string(deepstack ? 1 : 0) + std::to_string((int) fa) + ".gguf")).string();
            if (!t.assert_true("write the synthetic mmproj (" + tag + ")", fn_l22_write_qwen3vl_gguf(path, deepstack))) {
                continue;
            }
            // [TAG_R39_MTMD] the model, the backends and the subcontext of upstream's layout (was clip_init)
            mtmd_model_params mp;
            mtmd_models models;
            try {
                models = mtmd_model_load(path.c_str(), mp, ggml_backend_dev_buffer_type(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)));
            } catch (const std::exception &) {
            }
            std::unique_ptr<mtmd_backends>   backends;
            std::unique_ptr<mtmd_subcontext> sctx;
            if (models.v) {
                backends = std::make_unique<mtmd_backends>(nullptr, 2, models.v->max_nodes, nullptr, nullptr);
                mtmd_subctx_params sp { fa, /* warmup */ true };
                sctx = std::make_unique<mtmd_subcontext>(*models.v, *backends, sp);
            }
            mtmd_subcontext * ctx = sctx.get();
            if (!t.assert_true("model load (" + tag + ")", ctx != nullptr)) {
                std::filesystem::remove(path);
                continue;
            }

            t.assert_true("off until enabled (" + tag + ")", !mtmd_fork_l22_batch_on(ctx) && !ctx->support_batch);
            t.assert_true("enable (" + tag + ")", mtmd_fork_l22_enable_batch(ctx));
            t.assert_true("support_batch unchanged (" + tag + ")", !ctx->support_batch && mtmd_fork_l22_batch_on(ctx));
            t.assert_equal("warm-up reserve rows (" + tag + ")", 46 * 46, mtmd_fork_l22_reserve_tokens(ctx));

            std::mt19937 rng(42);
            std::vector<mtmd_image_f32> imgs;
            for (int i = 0; i < 4; ++i) {
                imgs.push_back(fn_l22_random_image(rng, 64, 96));
            }
            const int    n_tok = ctx->model.n_output_tokens(&imgs[0]);
            const size_t per   = (size_t) n_tok * (size_t) ctx->model.n_mmproj_embd();
            t.assert_equal("rows per 64x96 picture (" + tag + ")", 6, n_tok);
            t.assert_equal("projector width (" + tag + ")", deepstack ? 48 : 24, ctx->model.n_mmproj_embd());

            // single pictures: today's path
            std::vector<std::vector<float>> single(4);
            for (int i = 0; i < 4; ++i) {
                mtmd_image_f32_batch b;
                b.entries.push_back(imgs[i]);
                single[i].resize(per);
                t.assert_true("single encode " + std::to_string(i) + " (" + tag + ")", ctx->encode_batch(2, &b, single[i]));
            }
            t.assert_true("pictures differ (" + tag + ")", fn_l22_max_abs_diff(single[0].data(), single[1].data(), per) > 1e-3);

            auto encode_batch = [&](int first, int n, int fn_n, std::vector<float> & out) {
                mtmd_image_f32_batch b;
                for (int i = 0; i < n; ++i) {
                    b.entries.push_back(imgs[first + i]);
                }
                b.fn_n_images = fn_n;
                return ctx->encode_batch(2, &b, out);
            };
            auto check_vs_single = [&](const std::string & what, const std::vector<float> & out, int first, int n) {
                for (int i = 0; i < n; ++i) {
                    const double d = fn_l22_max_abs_diff(out.data() + per * i, single[first + i].data(), per);
                    t.log(what + " picture " + std::to_string(i) + ": max|d| " + std::to_string(d));
                    t.assert_true(what + " picture " + std::to_string(i) + " equals the single encode (" + tag + ")", d <= tol);
                }
            };

            // one graph of 3
            {
                std::vector<float> out(per * 3);
                const uint64_t s0 = mtmd_fork_l22_n_split(ctx);
                t.assert_true("batch of 3 (" + tag + ")", encode_batch(0, 3, 3, out));
                t.assert_equal("batch of 3 not split (" + tag + ")", s0, mtmd_fork_l22_n_split(ctx));
                check_vs_single("batch of 3", out, 0, 3);
            }
            // B = 2: stills vs the legacy frame pair
            {
                std::vector<float> out(per * 2);
                t.assert_true("batch of 2 stills (" + tag + ")", encode_batch(1, 2, 2, out));
                check_vs_single("batch of 2", out, 1, 2);
                std::vector<float> pair(per);
                t.assert_true("2 entries without the flag = one frame pair (" + tag + ")", encode_batch(1, 2, 0, pair));
                t.assert_true("the frame pair is not picture 1 (" + tag + ")", fn_l22_max_abs_diff(pair.data(), single[1].data(), per) > 1e-4);
            }
            // forced limits: split into groups, same output
            {
                const size_t n1 = mtmd_fork_l22_need_bytes(ctx, 64, 96, 1);
                const size_t n2 = mtmd_fork_l22_need_bytes(ctx, 64, 96, 2);
                const size_t n3 = mtmd_fork_l22_need_bytes(ctx, 64, 96, 3);
                const size_t n4 = mtmd_fork_l22_need_bytes(ctx, 64, 96, 4);
                t.log("compute bytes for 1/2/3/4 pictures: " + std::to_string(n1) + " / " + std::to_string(n2) + " / " +
                      std::to_string(n3) + " / " + std::to_string(n4));
                t.assert_true("compute grows with the batch (" + tag + ")", n1 < n2 && n2 < n3 && n3 < n4);

                mtmd_fork_l22_set_test_limit(ctx, (n1 + n2) / 2);
                std::vector<float> out(per * 4);
                uint64_t s0 = mtmd_fork_l22_n_split(ctx);
                t.assert_true("batch of 4, limit for 1 (" + tag + ")", encode_batch(0, 4, 4, out));
                t.assert_equal("split once (" + tag + ")", s0 + 1, mtmd_fork_l22_n_split(ctx));
                check_vs_single("groups 1+1+1+1", out, 0, 4);

                mtmd_fork_l22_set_test_limit(ctx, (n2 + n3) / 2);
                std::fill(out.begin(), out.end(), 0.0f);
                s0 = mtmd_fork_l22_n_split(ctx);
                t.assert_true("batch of 4, limit for 2 (" + tag + ")", encode_batch(0, 4, 4, out));
                t.assert_equal("split again (" + tag + ")", s0 + 1, mtmd_fork_l22_n_split(ctx));
                check_vs_single("groups 2+2", out, 0, 4);

                mtmd_fork_l22_set_test_limit(ctx, n4);
                std::fill(out.begin(), out.end(), 0.0f);
                s0 = mtmd_fork_l22_n_split(ctx);
                t.assert_true("batch of 4, limit for 4 (" + tag + ")", encode_batch(0, 4, 4, out));
                t.assert_equal("no split at the exact need (" + tag + ")", s0, mtmd_fork_l22_n_split(ctx));
                check_vs_single("one graph of 4", out, 0, 4);
                mtmd_fork_l22_set_test_limit(ctx, 0);
            }
            // a batch of two sizes is rejected, no abort
            {
                mtmd_image_f32_batch b;
                b.entries.push_back(imgs[0]);
                b.entries.push_back(fn_l22_random_image(rng, 96, 64));
                b.fn_n_images = 2;
                std::vector<float> out(per * 2);
                t.assert_true("two sizes rejected (" + tag + ")", !ctx->encode_batch(2, &b, out));
            }

            sctx.reset(); // [TAG_R39_MTMD] the subcontext before the backends it uses
            backends.reset();
            std::filesystem::remove(path);
        }
    }
}

//
// mtmd_helper_resampler
//

static std::vector<float> make_tone(double freq, int rate, size_t n) {
    const double pi = 3.14159265358979323846;
    std::vector<float> x(n);
    for (size_t i = 0; i < n; i++) {
        x[i] = (float) (0.5 * std::sin(2 * pi * freq * i / rate));
    }
    return x;
}

// amplitude of the freq component, skipping the edges
static double tone_amplitude(const std::vector<float> & x, double freq, int rate) {
    const double pi = 3.14159265358979323846;
    const size_t skip = rate / 10;
    double re = 0.0, im = 0.0;
    for (size_t i = skip; i < x.size() - skip; i++) {
        re += x[i] * std::cos(2 * pi * freq * i / rate);
        im += x[i] * std::sin(2 * pi * freq * i / rate);
    }
    return 2.0 * std::sqrt(re * re + im * im) / (x.size() - 2 * skip);
}

static std::vector<float> resample_chunked(mtmd_helper_resampler * rs, const std::vector<float> & in, size_t chunk) {
    std::vector<float> res;
    for (size_t off = 0; off < in.size(); off += chunk) {
        const float * out = nullptr;
        const int64_t n = mtmd_helper_resampler_process(rs, in.data() + off, std::min(chunk, in.size() - off), &out);
        if (n < 0) {
            return {};
        }
        res.insert(res.end(), out, out + n);
    }
    return res;
}

MAKE_TEST(test_helper_resampler) {
    const int rate_out = 16000;
    for (int rate_in : { 8000, 24000, 44100, 48000 }) {
        const std::string name = std::to_string(rate_in) + " -> " + std::to_string(rate_out);
        const std::vector<float> in = make_tone(440, rate_in, (size_t) rate_in * 2);

        mtmd_helper::resampler_ptr rs_all(mtmd_helper_resampler_init(rate_in, rate_out));
        mtmd_helper::resampler_ptr rs_chk(mtmd_helper_resampler_init(rate_in, rate_out));
        if (!t.assert_true(name + " init", rs_all && rs_chk)) {
            continue;
        }
        const std::vector<float> all = resample_chunked(rs_all.get(), in, in.size());
        const std::vector<float> chk = resample_chunked(rs_chk.get(), in, 1234); // odd size, not a multiple of the ratio

        t.assert_equal(name + " n_samples", (size_t) rate_out * 2, all.size());
        t.assert_true(name + " chunked == one-shot", all == chk);

        const double amp = tone_amplitude(all, 440, rate_out);
        t.assert_true(name + " amplitude = " + std::to_string(amp), std::fabs(amp - 0.5) < 0.02);
    }

    // above the output Nyquist: 10 kHz at 48 kHz aliases to 6 kHz at 16 kHz, the low-pass filter must cut it
    {
        mtmd_helper::resampler_ptr rs(mtmd_helper_resampler_init(48000, rate_out));
        const std::vector<float> out = resample_chunked(rs.get(), make_tone(10000, 48000, 48000 * 2), 4800);
        const double db = 20 * std::log10(tone_amplitude(out, 6000, rate_out) / 0.5);
        t.assert_true("aliasing = " + std::to_string(db) + " dB", db < -15); // about -20 dB with lpfOrder = 8, -10 dB with the default
    }

    // same rate: no copy
    {
        mtmd_helper::resampler_ptr rs(mtmd_helper_resampler_init(rate_out, rate_out));
        const std::vector<float> in = make_tone(440, rate_out, 1000);
        const float * out = nullptr;
        const int64_t n = rs ? mtmd_helper_resampler_process(rs.get(), in.data(), in.size(), &out) : -1;
        t.assert_true("passthrough", n == (int64_t) in.size() && out == in.data());
    }

    t.assert_true("invalid rate", mtmd_helper_resampler_init(0, rate_out) == nullptr);
}

//
// main
//

int main(int argc, char ** argv) {
    testing t(std::cout);
    t.verbose = true;

    // the audio preprocessors run on the CPU backend
    ggml_backend_load_all();

    // the video helper logs every frame at debug level
    mtmd_helper_log_set([](ggml_log_level level, const char * text, void *) {
        if (level != GGML_LOG_LEVEL_DEBUG) {
            fputs(text, stderr);
        }
    }, nullptr);

    // usage: test-mtmd-impl [filter_regex]
    for (int i = 1; i < argc; i++) {
        t.set_filter(argv[i]);
    }

    for (const auto & e : test_registry::all()) {
        t.test(e.name, e.fn);
    }

    return t.summary();
}
