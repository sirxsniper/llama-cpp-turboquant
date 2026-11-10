#pragma once

#include "mtmd.h"

#include <string>
#include <vector>

// !!! Internal header, to be used by mtmd and its unit tests only !!!

#define MTMD_INTERNAL_HEADER

// bitmap is null for text parts
struct mtmd_internal_part {
    std::string text;
    const mtmd_bitmap * bitmap;
    // only used for text parts
    bool parse_special = false;
};

// [QWEN_VIDEO] merged parts are erased from `parts`, so one group always maps to one part
std::vector<std::vector<const mtmd_bitmap *>> mtmd_group_mergeable_bitmaps(std::vector<mtmd_internal_part> & parts, int n_merge);

// [TAG_FN_L22_VLBATCH] round 22 (P2b): may `next` join a batch that starts with `first` and holds n_tokens_now rows?
// Only still pictures (one entry each, no placeholder) of the same token grid and pixel size, up to cap rows.
// Returns 0 (add), 2 (too large) or 3 (cannot batch), the codes of mtmd_batch_add_chunk
struct mtmd_fn_l22_vlb_info {
    bool image       = false;
    bool placeholder = false;
    int  n_entries   = 0;
    int  nx          = 0; // token grid
    int  ny          = 0;
    int  pos         = 0; // position type
    int  px_w        = 0; // picture size of the first entry
    int  px_h        = 0;
    int  n_tokens    = 0;
};
int32_t mtmd_fn_l22_vlbatch_admit(const mtmd_fn_l22_vlb_info & first, const mtmd_fn_l22_vlb_info & next, int32_t n_tokens_now, int32_t cap);

struct mtmd_image_preprocessor;

// mtmd_get_video_config() for a given template, without a context
mtmd_video_config mtmd_video_config_init(mtmd_video_template tmpl, const mtmd_image_preprocessor * preproc,
                                         int32_t src_width, int32_t src_height,
                                         int32_t src_n_frames, float src_fps);
