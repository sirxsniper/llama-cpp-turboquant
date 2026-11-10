#pragma once

// [TAG_R39_MTMD] the fork's vision code on upstream's mtmd layout (#31865: clip.cpp split into mtmd-model / mtmd-subcontext /
// mtmd-graph, clip_ctx -> mtmd_subcontext over a const mtmd_model and the shared mtmd_backends, #31945). Up to v0.29.0 this
// lived in clip.h / clip.cpp: the vision VRAM lend ([TAG_FN_L17_VISION_LEND], [TAG_FN_L22_VLEND_HOLD], [TAG_FN_L22_VLEND_SIZE],
// [TAG_FN_L22_OOMHOOK], [TAG_R34_F_VISFILE]), the device fallback to a CPU copy ([TAG_MTMD_DEVICE_FALLBACK]) and the batch of
// same-size Qwen3-VL stills ([TAG_FN_L22_VLBATCH]). The state is one member of mtmd_subcontext (fk); the code is
// mtmd-fork.cpp; upstream's files carry only the hooks.
// !!! Internal header, to be used by mtmd only !!!

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "mtmd-l22.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

struct mtmd_subcontext;
struct mtmd_subctx_encode_params;
struct mtmd_image_f32_batch;
struct mtmd_fork_twin;

// [TAG_FN_L17_VISION_LEND] park: the weights wait in host memory, the device weight and compute buffers are freed between
// encodes. An encode calls take(ud, bytes) first (the VRAM comes back from its other user), uploads the weights, encodes
// as before, frees both, then calls give(ud). park returns the device bytes freed (0: not parked: no device buffer, no_alloc)
typedef size_t (*mtmd_fork_lend_take_fn)(void * ud, size_t bytes);
typedef void   (*mtmd_fork_lend_give_fn)(void * ud);
// [TAG_FN_L22_OOMHOOK] round 22 (P2): a failed lend buffer allocation of bytes; true = memory was released, retry once
typedef bool   (*mtmd_fork_lend_oom_fn)(void * ud, size_t bytes);

// [TAG_MTMD_DEVICE_FALLBACK] where encode_impl stopped. A failure after the graph is built (allocation, input upload,
// compute, output download) is a failure of the device; a rejected input or a graph builder that throws is not, and
// never moves the subcontext to the CPU.
struct mtmd_fork_encode_status {
    bool graph_built   = false; // the graph builder returned; an exception after this comes from the backend
    bool device_failed = false; // graph allocation or graph compute returned an error
    bool alloc_failed  = false; // [TAG_FN_L22_OOMHOOK] it was the graph allocation
};

// [TAG_FN_L17_VISION_LEND] parked: the weights wait in host (pinned if possible) memory, the device weight buffer and
// the scheduler's compute buffers are freed. The encode takes the VRAM back, uploads, encodes and parks again.
struct mtmd_fork_lend {
    bool on     = false;
    bool parked = false;
    ggml_backend_buffer_type_t buft = nullptr; // the weights' device buffer type
    ggml_backend_buffer_ptr host_buf;
    std::vector<uint8_t> host_vec;              // when no pinned buffer could be had
    uint8_t * host = nullptr;
    std::vector<std::pair<ggml_tensor *, size_t>> offs; // tensor, its offset in host
    size_t bytes = 0; // device bytes freed at park: weights + compute
    mtmd_fork_lend_take_fn take = nullptr;
    mtmd_fork_lend_give_fn give = nullptr;
    void * ud = nullptr;
    uint64_t n = 0;

    // [TAG_FN_L22_VLEND_HOLD] hold > 0: an encode keeps the device side up for the next one (one take per turn)
    std::recursive_mutex mu;   // an encode (maybe on the media worker) against hold / suspend calls
    int      hold      = 0;
    bool     out       = false; // take() called, give() not yet
    size_t   taken     = 0;     // the bytes the current take was sized for
    uint64_t n_in_take = 0;     // encodes since that take
    bool     fresh     = false; // a new scheduler since the take: its buffer size is this picture's
    // [TAG_FN_L22_OOMHOOK]
    mtmd_fork_lend_oom_fn oom    = nullptr;
    void *                oom_ud = nullptr;
    // [TAG_FN_L22_VLEND_SIZE]
    bool               size_on = false;
    size_t             w       = 0; // weight buffer bytes
    mtmd_l22_lend_size sz;
    // [TAG_R34_F_VISFILE] the weights whose bytes equal the projector file's wait in the file (the OS file cache: reclaimable
    // RAM, not a locked or committed copy) and are read again at each take; the rest (none for a plain projector) keeps
    // the host copy. LLAMA_R34_VISFILE=1 turns it on; unset / 0: every weight in the host copy, pinned if possible (v0.26.6)
    std::vector<uint64_t> foff;            // per offs entry: its file offset, UINT64_MAX = in the host copy
    size_t                file_bytes = 0;  // read from the file at each take
    uintmax_t             file_size  = 0;  // the file as parked: a change fails the take (no silent other weights)
    int64_t               file_time  = 0;
    ggml_backend_buffer_ptr stage_buf;     // 2 staging halves for the file reads (pinned when possible)
    std::vector<uint8_t>    stage_vec;
    uint8_t *               stage      = nullptr;
    size_t                  stage_half = 0;
    bool                    file_err   = false; // the last unpark failed on the file, not on the allocation
};

// the fork's state of one subcontext (mtmd_subcontext::fk)
struct mtmd_fork_subctx {
    // [TAG_FN_L22_VLBATCH] Qwen3-VL: batches of independent stills of one size (mtmd_image_f32_batch::fn_n_images > 1)
    bool    fn_vlbatch_on      = false;
    int32_t fn_reserve_tokens  = 0; // output tokens of the warm-up picture
    size_t  fn_reserve_compute = 0; // its device compute bytes
    int64_t fn_hw_patches      = 0; // most patches encoded in one graph so far
    size_t  fn_limit_test      = 0; // tests: device compute limit forced on any backend (0: off)
    uint64_t fn_n_split        = 0;
    std::map<std::tuple<int, int, int, int>, size_t> fn_need; // (nx, ny, k, flash attn) -> device compute bytes of a k-image graph
    ggml_backend_sched_ptr fn_sched_measure;             // measures only, never allocates

    // [TAG_MTMD_DEVICE_FALLBACK] what a CPU copy of this subcontext is built from: the mmproj path and the parameters the
    // context was made with (mtmd_fork_set_source). cpu_twin is that copy, loaded on first use from the file (the weights
    // of this subcontext may sit in a GPU buffer). cpu_only: the device failed an encode, every encode now runs on the copy.
    std::string fname;
    bool        no_alloc        = false;
    int32_t     image_min_tokens = -1;
    int32_t     image_max_tokens = -1;
    bool        l22_vlresize    = false;
    int         n_threads       = 1;
    int         flash_attn_req  = -1; // the mtmd_flash_attn_type the context asked for (AUTO = -1)
    ggml_backend_sched_eval_callback cb_eval    = nullptr;
    void *                           cb_eval_ud = nullptr;
    mtmd_fork_twin * cpu_twin = nullptr;
    bool cpu_only = false;

    // the status of the last encode_impl
    mtmd_fork_encode_status st;

    mtmd_fork_lend lend;

    mtmd_fork_subctx() = default;
    ~mtmd_fork_subctx(); // mtmd-fork.cpp: the CPU copy and the measuring scheduler, before the backends they refer to
    mtmd_fork_subctx(const mtmd_fork_subctx &) = delete;
    mtmd_fork_subctx & operator=(const mtmd_fork_subctx &) = delete;
};

// the source of a subcontext's CPU copy (call once after the subcontext is made)
void mtmd_fork_set_source(mtmd_subcontext * ctx, const char * fname, bool no_alloc, int32_t image_min_tokens,
                          int32_t image_max_tokens, bool l22_vlresize, int n_threads, int flash_attn_req,
                          ggml_backend_sched_eval_callback cb_eval, void * cb_eval_ud);

// [TAG_FN_L22_VLBATCH] the warm-up reserve: called by the subcontext after its warm-up
void mtmd_fork_on_warmup(mtmd_subcontext * ctx, const mtmd_image_f32_batch & batch);

// [TAG_FN_L17_VISION_LEND] see mtmd_fork_lend_take_fn
size_t mtmd_fork_lend_park(mtmd_subcontext * ctx, mtmd_fork_lend_take_fn take, mtmd_fork_lend_give_fn give, void * ud);
// called once when the weight buffer or the encoder's compute buffer of a lent subcontext cannot be allocated (NULL: none)
void   mtmd_fork_lend_set_oom(mtmd_subcontext * ctx, mtmd_fork_lend_oom_fn fn, void * ud);
// [TAG_FN_L22_VLEND_SIZE] before mtmd_fork_lend_park: size each take from the picture's own compute buffer (measured at park)
void   mtmd_fork_lend_set_size(mtmd_subcontext * ctx, bool on);
// the most a take asks for (the largest picture), 0 when not parked
size_t mtmd_fork_lend_max_bytes(const mtmd_subcontext * ctx);
// [TAG_FN_L22_VLEND_HOLD] on: count a hold, off: release one. While held, an encode keeps the weights and the VRAM after it
// (the next encode needs no take / upload); the last release parks and gives the VRAM back. suspend: park and give back
// now, keep the count (call it before any decode of the text model). reset: release every hold
void   mtmd_fork_lend_hold(mtmd_subcontext * ctx, bool on);
void   mtmd_fork_lend_suspend(mtmd_subcontext * ctx);
void   mtmd_fork_lend_hold_reset(mtmd_subcontext * ctx);

// the device the encoder runs on (nullptr: the CPU)
ggml_backend_dev_t mtmd_fork_get_device(const mtmd_subcontext * ctx);

// [TAG_FN_L22_VLBATCH] round 22 (P2b): Qwen3-VL encodes a batch of independent still images of one size
// (mtmd_image_f32_batch::fn_n_images > 1) as one graph, split into balanced groups when it needs more device compute
// than this encode may use. enable: true for a Qwen3-VL vision subcontext (changes nothing for others)
bool     mtmd_fork_l22_enable_batch(mtmd_subcontext * ctx);
bool     mtmd_fork_l22_batch_on(const mtmd_subcontext * ctx);
int32_t  mtmd_fork_l22_reserve_tokens(const mtmd_subcontext * ctx); // output tokens of the warm-up picture (0: no warm-up)
uint64_t mtmd_fork_l22_n_split(const mtmd_subcontext * ctx);        // batches that ran as several groups
// group sizes for n pictures: the largest k with fits(k) (1 always fits), as balanced groups
std::vector<int> mtmd_fork_l22_split_plan(int n, const std::function<bool(int)> & fits);
// device compute bytes of a k-image graph (cached per size); tests: a forced limit on any backend (0: off)
size_t   mtmd_fork_l22_need_bytes(mtmd_subcontext * ctx, int nx, int ny, int k);
void     mtmd_fork_l22_set_test_limit(mtmd_subcontext * ctx, size_t bytes);
