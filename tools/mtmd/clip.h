#pragma once

#include "ggml.h"
#include "mtmd.h"

#include <stddef.h>
#include <stdint.h>

#include <functional>
#include <map>
#include <vector>

// !!! Internal header, to be used by mtmd only !!!

#define MTMD_INTERNAL_HEADER

struct clip_ctx;
struct clip_enc_state;
struct clip_gen_state;

struct clip_image_size {
    int width;
    int height;
    bool operator==(const clip_image_size & other) const {
        return width == other.width && height == other.height;
    }
    bool operator!=(const clip_image_size & other) const {
        return !(*this == other);
    }
    int area() const {
        // avoid overflow when computing area
        GGML_ASSERT(width  >= 0 && width  <= 46000);
        GGML_ASSERT(height >= 0 && height <= 46000);
        return width * height;
    }
};

struct clip_image_f32;
struct clip_image_f32_batch;

enum clip_modality {
    CLIP_MODALITY_VISION,
    CLIP_MODALITY_AUDIO,
    CLIP_MODALITY_GEN_AUDIO,
};

enum clip_flash_attn_type {
    CLIP_FLASH_ATTN_TYPE_AUTO     = -1,
    CLIP_FLASH_ATTN_TYPE_DISABLED = 0,
    CLIP_FLASH_ATTN_TYPE_ENABLED  = 1,
};

struct clip_context_params {
    bool use_gpu;
    ggml_backend_dev_t device;
    enum clip_flash_attn_type flash_attn_type;
    int image_min_tokens;
    int image_max_tokens;
    bool warmup;
    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;
    bool no_alloc;
    mtmd_progress_callback progress_callback;
    void * progress_callback_user_data;
    bool l22_vlresize; // [TAG_FN_L22_VLRESIZE] Qwen-VL family: resize as the HF processor does (#30736: no padding, round half to even)
};

struct clip_init_result {
    struct clip_ctx * ctx_v; // vision context
    struct clip_ctx * ctx_a; // audio context
    struct clip_ctx * ctx_gen_a; // audio generation context
};

struct clip_init_result clip_init(const char * fname, struct clip_context_params ctx_params);

// max number of output tokens per image, -1 if not dynamic size
int clip_get_image_max_tokens(const struct clip_ctx * ctx);

void clip_free(struct clip_ctx * ctx);

// TODO: should be enum, not string
const char * clip_patch_merge_type(const struct clip_ctx * ctx);

int clip_n_output_tokens(const clip_ctx * ctx, const clip_image_f32 * img);

// for M-RoPE, this will be the number of token positions in X and Y directions
// for other models, X will be the total number of tokens and Y will be 1
int clip_n_output_tokens_x(const clip_ctx * ctx, const clip_image_f32 * img);
int clip_n_output_tokens_y(const clip_ctx * ctx, const clip_image_f32 * img);

// this should be equal to the embedding dimension of the text model
int clip_n_mmproj_embd(const struct clip_ctx * ctx);

// TODO: remove clip_image_encode() and always use batched version
bool clip_image_encode      (struct clip_ctx * ctx, int n_threads, const clip_image_f32 * img, std::vector<float> & out_vec);
bool clip_image_batch_encode(struct clip_ctx * ctx, int n_threads, const struct clip_image_f32_batch * imgs, std::vector<float> & out_batch_embd);

enum clip_gen_process_type {
    CLIP_GEN_PROCESS_GEN_UNKNOWN,
    CLIP_GEN_PROCESS_GEN_CODE, // h_state to codes
    CLIP_GEN_PROCESS_GEN_WAV,  // codes to raw PCM audio
};
struct clip_encode_params {
    int n_threads = 1;
    const clip_image_f32_batch * imgs = nullptr;
    std::vector<float> * out_embd = nullptr;

    // for audio gen, imgs has exactly one entry: hidden state from backbone (GEN_CODE) or unused (GEN_WAV)
    clip_gen_process_type gen_process = CLIP_GEN_PROCESS_GEN_UNKNOWN;

    // GEN_CODE: out_embd receives the embd to feed back to the backbone
    int32_t code0 = 0; // semantic code sampled by the backbone
    int32_t top_k = 50;
    float   top_p = 1.0f;
    std::vector<int32_t> * out_codes = nullptr; // this frame's 16 sampled codes
    std::vector<float> * out_feats = nullptr; // continuous counterpart of out_codes
    float   temp = 0.0f;                      // sampling temperature, noise scale for flow-matching decoders
    bool * out_is_eos = nullptr;

    // GEN_WAV
    const std::vector<int32_t> * codes = nullptr;     // this frame's 16 RVQ codes
    const std::vector<float> *   feats = nullptr;     // continuous counterpart of codes
    std::vector<float> * out_audio = nullptr;         // decoded PCM samples, F32

    // audio gen: state of one sequence, clip_encode() updates it in place
    struct clip_gen_state * gen_state = nullptr;

    // streaming audio input, updated in place by clip_encode()
    struct clip_enc_state * enc_state = nullptr;
};
bool clip_encode(struct clip_ctx * ctx, struct clip_encode_params * params);

// [TAG_FN_L17_VISION_LEND] park: the weights wait in host memory, the device weight and compute buffers are freed between
// encodes. clip_encode calls take(ud, bytes) first (the VRAM comes back from its other user), uploads the weights, encodes
// as before, frees both, then calls give(ud). Returns the device bytes freed (0: not parked: no device buffer, no_alloc)
typedef size_t (*clip_lend_take_fn)(void * ud, size_t bytes);
typedef void   (*clip_lend_give_fn)(void * ud);
size_t clip_lend_park(struct clip_ctx * ctx, clip_lend_take_fn take, clip_lend_give_fn give, void * ud);
// [TAG_FN_L22_OOMHOOK] round 22 (P2): a failed lend buffer allocation of bytes; true = memory was released, retry once
typedef bool   (*clip_lend_oom_fn)(void * ud, size_t bytes);
// called once when the weight buffer or the encoder's compute buffer of a lent context cannot be allocated (NULL: none)
void   clip_lend_set_oom(struct clip_ctx * ctx, clip_lend_oom_fn fn, void * ud);
// [TAG_FN_L22_VLEND_SIZE] before clip_lend_park: size each take from the picture's own compute buffer (measured at park)
void   clip_lend_set_size(struct clip_ctx * ctx, bool on);
// the most a take asks for (the largest picture), 0 when not parked
size_t clip_lend_max_bytes(const struct clip_ctx * ctx);
// [TAG_FN_L22_VLEND_HOLD] on: count a hold, off: release one. While held, an encode keeps the weights and the VRAM after it
// (the next encode needs no take / upload); the last release parks and gives the VRAM back. suspend: park and give back
// now, keep the count (call it before any decode of the text model). reset: release every hold
void   clip_lend_hold(struct clip_ctx * ctx, bool on);
void   clip_lend_suspend(struct clip_ctx * ctx);
void   clip_lend_hold_reset(struct clip_ctx * ctx);

// the device the encoder runs on (nullptr: the CPU)
ggml_backend_dev_t clip_get_device(const struct clip_ctx * ctx);
// persistent state of a causal audio encoder, for streaming input (one per stream)
// it holds the sliding window of past keys/values, so each clip_encode() call only takes the new frames
// n_frames_max is the maximum number of encoder frames per call
struct clip_enc_state * clip_enc_state_init(struct clip_ctx * ctx, int32_t n_frames_max);
void                    clip_enc_state_free(struct clip_enc_state * state);

// per-sequence state of audio generation: RNG and the slots kept between GEN_WAV calls
// seed UINT32_MAX means random; reset() clears the slots and keeps the RNG stream
struct clip_gen_state * clip_gen_state_init(struct clip_ctx * ctx, uint32_t seed);
void                    clip_gen_state_free(struct clip_gen_state * state);
void                    clip_gen_state_reset(struct clip_gen_state * state);

bool clip_is_llava(const struct clip_ctx * ctx);
// note for contributor: this clip_is_(model) pattern is deprecated
//                       do NOT add new functions like this

bool clip_has_vision_encoder(const struct clip_ctx * ctx);
bool clip_has_audio_encoder(const struct clip_ctx * ctx);

bool clip_support_batch(const struct clip_ctx * ctx);

// [TAG_FN_L22_VLBATCH] round 22 (P2b): Qwen3-VL encodes a batch of independent still images of one size
// (clip_image_f32_batch::fn_n_images > 1) as one graph, split into balanced groups when it needs more device compute
// than this encode may use. enable: true for a Qwen3-VL vision context (changes nothing for others)
bool    clip_fn_l22_enable_batch(struct clip_ctx * ctx);
bool    clip_fn_l22_batch_on(const struct clip_ctx * ctx);
int32_t clip_fn_l22_reserve_tokens(const struct clip_ctx * ctx); // output tokens of the warm-up picture (0: no warm-up)
uint64_t clip_fn_l22_n_split(const struct clip_ctx * ctx);        // batches that ran as several groups
// group sizes for n pictures: the largest k with fits(k) (1 always fits), as balanced groups
std::vector<int> clip_fn_l22_split_plan(int n, const std::function<bool(int)> & fits);
// device compute bytes of a k-image graph (cached per size); tests: a forced limit on any backend (0: off)
size_t  clip_fn_l22_need_bytes(struct clip_ctx * ctx, int nx, int ny, int k);
void    clip_fn_l22_set_test_limit(struct clip_ctx * ctx, size_t bytes);

int clip_model_n_temporal_merge(const struct clip_ctx * ctx); // TODO @ngxson : remove, refactor this

std::map<ggml_backend_dev_t, size_t> clip_get_mem_usage(const struct clip_ctx * ctx);

struct clip_cap {
    bool has_vision;
    bool has_audio;
    bool has_gen_audio;
};
struct clip_cap clip_get_cap(const char * fname);
