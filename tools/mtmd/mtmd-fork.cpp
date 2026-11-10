// [TAG_R39_MTMD] the fork's vision code on upstream's mtmd layout (see mtmd-fork.h). The bodies are v0.29.0's clip.cpp
// (8df993820) with clip_ctx -> mtmd_subcontext: the weights live in the (const) mtmd_model, the backends and the scheduler
// in the shared mtmd_backends, the fork's per-subcontext state in mtmd_subcontext::fk.

#include "mtmd-fork.h"
#include "mtmd-subcontext.h"
#include "mtmd-graph.h"
#include "mtmd-impl.h"
#include "mtmd-model.h"
#include "mtmd-l22.h"
#include "../src/llama-fn-l22.h" // [TAG_FN_L22_PASSLOG] llama_fn_l22_log, llama_fn_pin_active

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "gguf.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem> // [TAG_R34_F_VISFILE]
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <tuple>
#include <vector>

// [TAG_MTMD_DEVICE_FALLBACK] the CPU copy of a subcontext: its own model (CPU weights), backends and subcontext
struct mtmd_fork_twin {
    mtmd_models                      models;
    std::unique_ptr<mtmd_backends>   backends;
    std::unique_ptr<mtmd_subcontext> sctx;

    ~mtmd_fork_twin() {
        sctx.reset(); // before the backends and the model it refers to
        backends.reset();
    }
};

mtmd_fork_subctx::~mtmd_fork_subctx() {
    fn_sched_measure.reset(); // [TAG_FN_L22_VLBATCH] before the backends it refers to
    delete cpu_twin;
    cpu_twin = nullptr;
}

void mtmd_backends::remake_sched() {
    sched.reset(ggml_backend_sched_new(ptrs.data(), bufts.data(), ptrs.size(), sched_max_nodes, false, true));
    if (sched_cb_eval != nullptr) {
        ggml_backend_sched_set_eval_callback(sched.get(), sched_cb_eval, sched_cb_eval_user_data);
    }
}

// the lend parks and loads the weights of the subcontext's model: the model is const for the encoders, the lend owns its
// buffer between encodes (the mtmd_model objects are created non-const by mtmd_model_load)
static mtmd_model & fork_model_rw(mtmd_subcontext * ctx) {
    return const_cast<mtmd_model &>(ctx->model);
}

void mtmd_fork_set_source(mtmd_subcontext * ctx, const char * fname, bool no_alloc, int32_t image_min_tokens,
                          int32_t image_max_tokens, bool l22_vlresize, int n_threads, int flash_attn_req,
                          ggml_backend_sched_eval_callback cb_eval, void * cb_eval_ud) {
    if (ctx == nullptr) {
        return;
    }
    auto & f = ctx->fk;
    f.fname            = fname ? fname : "";
    f.no_alloc         = no_alloc;
    f.image_min_tokens = image_min_tokens;
    f.image_max_tokens = image_max_tokens;
    f.l22_vlresize     = l22_vlresize;
    f.n_threads        = n_threads;
    f.flash_attn_req   = flash_attn_req;
    f.cb_eval          = cb_eval;
    f.cb_eval_ud       = cb_eval_ud;
}

// [TAG_FN_L22_VLBATCH] the reserve picture: a batch of at most this many output tokens needs about this compute
void mtmd_fork_on_warmup(mtmd_subcontext * ctx, const mtmd_image_f32_batch & batch) {
    if (ctx->model.modality == MTMD_MODALITY_VISION && !batch.entries.empty()) {
        ctx->fk.fn_reserve_tokens = ctx->model.n_output_tokens(&batch.entries[0]);
        if (ctx->backends.backend != nullptr && ctx->backends.backend != ctx->backends.backend_cpu && ctx->backends.sched) {
            ctx->fk.fn_reserve_compute = ggml_backend_sched_get_buffer_size(ctx->backends.sched.get(), ctx->backends.backend);
        }
    }
}

// [TAG_MTMD_DEVICE_FALLBACK] An encoder that survives its device.
//
// Measured: with -mmdev igpu (Vulkan, 2-CU AMD iGPU) a ~4000-token Qwen3.8 image encode lost the Vulkan device after
// ~11 s (Windows TDR, LiveKernelEvent 141). The request failed and the projector stayed unusable for the rest of the
// process. Now, when an encode fails on a non-CPU device (graph allocation or compute returns an error, or the backend
// throws, as ggml-vulkan does with vk::DeviceLostError), the subcontext moves to the CPU for good: a CPU copy (cpu_twin)
// reads the weights again from the mmproj file, the device's weight and compute buffers are freed, and the same encode
// runs once more on the CPU, so the request still succeeds. One warning says so. The fallback lives here, below
// mtmd_encode, so the server's async encoder thread gets it too.
//
// Proactive guard: on an integrated GPU, an image of more than MTMD_IGPU_MAX_TOKENS output tokens (default 1536,
// 0 = off) is encoded on the CPU copy directly, and smaller images stay on the iGPU. Measured: 1066 tokens took 44 s
// on the iGPU and 4.7 s on the CPU. The CPU copy then stays loaded (one more copy of the weights in RAM).
//
// Not covered, unchanged: CUDA (and ROCm, MUSA), which abort on a device error instead of returning it; gen-audio
// subcontexts; no_alloc contexts. MTMD_DEVICE_FALLBACK=0 turns both the fallback and the guard off.
// [TAG_R39_MTMD] the device backend and its scheduler are shared by the subcontexts of one context now: a failed device
// moves the failing subcontext to its CPU copy; another subcontext follows on its own first failure.
static bool clip_device_fallback_enabled() {
    static const bool enabled = [] {
        const char * e = std::getenv("MTMD_DEVICE_FALLBACK");
        return !(e != nullptr && std::atoi(e) == 0);
    }();
    return enabled;
}

static int clip_igpu_max_tokens() {
    static const int n = [] {
        const char * e = std::getenv("MTMD_IGPU_MAX_TOKENS");
        if (e == nullptr || *e == '\0') {
            return 1536;
        }
        return std::max(0, std::atoi(e));
    }();
    return n;
}

static bool clip_device_fallback_eligible(const mtmd_subcontext * ctx) {
    if (!clip_device_fallback_enabled() || ctx->fk.no_alloc || ctx->fk.fname.empty()) {
        return false;
    }
    if (ctx->model.modality != MTMD_MODALITY_VISION && ctx->model.modality != MTMD_MODALITY_AUDIO) {
        return false;
    }
    if (ctx->fk.cpu_only) {
        return true;
    }
    if (ctx->backends.backend == nullptr || ctx->backends.backend == ctx->backends.backend_cpu) {
        return false;
    }
    ggml_backend_dev_t dev      = ggml_backend_get_device(ctx->backends.backend);
    ggml_backend_reg_t reg      = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    const char *       reg_name = reg ? ggml_backend_reg_name(reg) : nullptr;
    if (reg_name == nullptr) {
        return false;
    }
    return strcmp(reg_name, "CUDA") != 0 && strcmp(reg_name, "ROCm") != 0 && strcmp(reg_name, "MUSA") != 0;
}

// the CPU copy of ctx, loaded from the mmproj file on first use; nullptr when it cannot be loaded
static mtmd_subcontext * clip_get_cpu_twin(mtmd_subcontext * ctx) {
    auto & f = ctx->fk;
    if (f.cpu_twin != nullptr) {
        return f.cpu_twin->sctx.get();
    }

    LOG_INF("%s: loading a CPU copy of the %s encoder from %s\n", __func__,
            ctx->model.modality == MTMD_MODALITY_AUDIO ? "audio" : "vision", f.fname.c_str());
    mtmd_fork_twin * twin = new mtmd_fork_twin();
    try {
        mtmd_model_params mp;
        mp.image_min_tokens = f.image_min_tokens;
        mp.image_max_tokens = f.image_max_tokens;
        mp.no_alloc         = false;
        mp.l22_vlresize     = f.l22_vlresize;
        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        twin->models = mtmd_model_load(f.fname.c_str(), mp, ggml_backend_dev_buffer_type(cpu_dev));
        const mtmd_model * m = ctx->model.modality == MTMD_MODALITY_AUDIO ? twin->models.a.get() : twin->models.v.get();
        if (m == nullptr) {
            throw std::runtime_error("the mmproj file has no encoder of this modality");
        }
        twin->backends = std::make_unique<mtmd_backends>(nullptr, f.n_threads, m->max_nodes, f.cb_eval, f.cb_eval_ud);
        mtmd_subctx_params sp {
            /* flash_attn_type */ (mtmd_flash_attn_type) f.flash_attn_req,
            /* warmup          */ false, // the first encode reserves the compute buffers for its own batch
        };
        twin->sctx = std::make_unique<mtmd_subcontext>(*m, *twin->backends, sp);
        twin->sctx->fk.fname         = f.fname;
        twin->sctx->fk.fn_vlbatch_on = f.fn_vlbatch_on; // [TAG_FN_L22_VLBATCH] a fallback encode of a batch of stills
    } catch (const std::exception & e) {
        LOG_ERR("%s: failed to load the CPU copy of the encoder from %s: %s\n", __func__, f.fname.c_str(), e.what());
        delete twin;
        return nullptr;
    }
    f.cpu_twin = twin;
    return twin->sctx.get();
}

static bool clip_encode_on_cpu_twin(const mtmd_subcontext * ctx, mtmd_subcontext * twin, mtmd_subctx_encode_params * params) {
    twin->debug_output_embeddings = ctx->debug_output_embeddings;
    return twin->encode_impl(params);
}

// [TAG_FN_L17_VISION_LEND] ------------------------------------------------------------------------------------------------

// [TAG_R34_F_VISFILE] -----------------------------------------------------------------------------------------------------
// off by default (measured 11-01: each take reads the 862 MiB again, weights up 393 ms instead of 26 ms per image turn; the
// first answer after a start showed no gain over the old build in the same session (90.6 vs 91.4 t/s), steady decode
// unchanged): LLAMA_R34_VISFILE=1 turns it on
static bool clip_visfile_wanted() {
    const char * e = std::getenv("LLAMA_R34_VISFILE");
    return e != nullptr && strcmp(e, "1") == 0;
}

static bool clip_file_stamp(const std::string & fname, uintmax_t & size, int64_t & time) {
    std::error_code ec;
#ifdef _WIN32
    const std::filesystem::path fp(std::filesystem::u8path(fname));
#else
    const std::filesystem::path fp(fname);
#endif
    size = std::filesystem::file_size(fp, ec);
    if (ec) {
        return false;
    }
    const auto t = std::filesystem::last_write_time(fp, ec);
    if (ec) {
        return false;
    }
    time = (int64_t) t.time_since_epoch().count();
    return true;
}

// at park, while the weights are on the device: every weight whose file bytes (by name, same size) equal its device bytes
// gets its file offset; returns the bytes found
static size_t clip_lend_map_file(mtmd_subcontext * ctx) {
    auto & L = ctx->fk.lend;
    const std::string & fname = ctx->fk.fname;
    L.foff.assign(L.offs.size(), UINT64_MAX);
    L.file_bytes = 0;
    if (fname.empty() || !clip_file_stamp(fname, L.file_size, L.file_time)) {
        return 0;
    }
    gguf_init_params gp = { /*.no_alloc = */ true, /*.ctx = */ nullptr };
    gguf_context * g = gguf_init_from_file(fname.c_str(), gp);
    if (g == nullptr) {
        return 0;
    }
    auto fin = open_ifstream_binary(fname);
    if (!fin) {
        gguf_free(g);
        return 0;
    }
    const size_t doff = gguf_get_data_offset(g);
    const size_t CH   = (size_t) 16 << 20;
    std::vector<uint8_t> a, b;
    for (size_t i = 0; i < L.offs.size(); ++i) {
        ggml_tensor * t  = L.offs[i].first;
        const int64_t id = gguf_find_tensor(g, ggml_get_name(t));
        const size_t  nb = ggml_nbytes(t);
        if (id < 0 || gguf_get_tensor_size(g, id) != nb) {
            continue;
        }
        const uint64_t fo = (uint64_t) doff + (uint64_t) gguf_get_tensor_offset(g, id);
        bool same = true;
        for (size_t p = 0; p < nb && same; p += CH) {
            const size_t k = std::min(CH, nb - p);
            a.resize(k);
            b.resize(k);
            ggml_backend_tensor_get(t, a.data(), p, k);
            fin.seekg((std::streamoff) (fo + p), std::ios::beg);
            fin.read(reinterpret_cast<char *>(b.data()), (std::streamsize) k);
            same = (bool) fin && memcmp(a.data(), b.data(), k) == 0;
        }
        if (!same) {
            fin.clear();
            continue;
        }
        L.foff[i]     = fo;
        L.file_bytes += nb;
    }
    gguf_free(g);
    return L.file_bytes;
}

// at each take: the file-backed weights from the file, through the two staging halves (one is read while the other copies)
static bool clip_lend_read_file(mtmd_subcontext * ctx) {
    auto & L = ctx->fk.lend;
    const std::string & fname = ctx->fk.fname;
    ggml_backend_t backend = ctx->backends.backend;
    if (L.file_bytes == 0) {
        return true;
    }
    uintmax_t sz = 0;
    int64_t   tm = 0;
    if (!clip_file_stamp(fname, sz, tm) || sz != L.file_size || tm != L.file_time) {
        LOG_ERR("%s: [TAG_R34_F_VISFILE] the projector file %s changed or is gone since the start: this encode fails (restart the "
                "server)\n", __func__, fname.c_str());
        return false;
    }
    auto fin = open_ifstream_binary(fname);
    if (!fin) {
        LOG_ERR("%s: [TAG_R34_F_VISFILE] cannot open %s: this encode fails\n", __func__, fname.c_str());
        return false;
    }
    int half = 0;
    for (size_t i = 0; i < L.offs.size(); ++i) {
        if (L.foff[i] == UINT64_MAX) {
            continue;
        }
        ggml_tensor * t  = L.offs[i].first;
        const size_t  nb = ggml_nbytes(t);
        for (size_t p = 0; p < nb; p += L.stage_half) {
            const size_t k = std::min(L.stage_half, nb - p);
            uint8_t * dst = L.stage + (size_t) half*L.stage_half;
            fin.seekg((std::streamoff) (L.foff[i] + p), std::ios::beg);
            fin.read(reinterpret_cast<char *>(dst), (std::streamsize) k);
            if (!fin) {
                ggml_backend_synchronize(backend);
                LOG_ERR("%s: [TAG_R34_F_VISFILE] reading %s failed: this encode fails\n", __func__, fname.c_str());
                return false;
            }
            ggml_backend_tensor_set_async(backend, t, dst, p, k);
            half ^= 1;
            if (half == 0) {
                ggml_backend_synchronize(backend); // both halves queued: the next read reuses the first
            }
        }
    }
    ggml_backend_synchronize(backend);
    return true;
}

static void clip_lend_clear_tensors(mtmd_subcontext * ctx) {
    ggml_context * cd = fork_model_rw(ctx).ctx_data.get();
    for (ggml_tensor * t = ggml_get_first_tensor(cd); t != nullptr; t = ggml_get_next_tensor(cd, t)) {
        t->data   = nullptr;
        t->buffer = nullptr;
    }
}

// frees the device weight buffer (the tensors keep their shapes, no data) and the compute buffers (a new scheduler)
static void clip_lend_free_device(mtmd_subcontext * ctx) {
    ggml_backend_synchronize(ctx->backends.backend);
    fork_model_rw(ctx).buf.reset();
    clip_lend_clear_tensors(ctx);
    ctx->backends.remake_sched();
    ctx->fk.lend.parked = true;
}

// a new device weight buffer (the same allocation as at load), the weights uploaded from host
// [TAG_R34_F_VISFILE_ERR] returns 0: weights up; 1: the weight buffer could not be allocated; 2: the projector file could not
// be read (memory was not what failed: no OOM hook)
static int clip_lend_unpark_once(mtmd_subcontext * ctx) {
    mtmd_model & model = fork_model_rw(ctx);
    ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors_from_buft(model.ctx_data.get(), ctx->fk.lend.buft);
    if (b == nullptr) {
        // tensors placed before the failure point into the freed buffer: no data again
        clip_lend_clear_tensors(ctx);
        return 1;
    }
    model.buf.reset(b);
    ggml_backend_buffer_set_usage(b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    const auto & L = ctx->fk.lend;
    for (size_t i = 0; i < L.offs.size(); ++i) {
        if (i < L.foff.size() && L.foff[i] != UINT64_MAX) {
            continue; // [TAG_R34_F_VISFILE] from the file below
        }
        ggml_backend_tensor_set_async(ctx->backends.backend, L.offs[i].first, L.host + L.offs[i].second, 0, ggml_nbytes(L.offs[i].first));
    }
    ggml_backend_synchronize(ctx->backends.backend);
    if (!clip_lend_read_file(ctx)) { // [TAG_R34_F_VISFILE] parked again: no weights on the device
        model.buf.reset();
        clip_lend_clear_tensors(ctx);
        return 2; // [TAG_R34_F_VISFILE_ERR]
    }
    ctx->fk.lend.parked = false;
    return 0;
}

// [TAG_FN_L22_OOMHOOK] a failed weight buffer allocation asks the OOM hook once (e.g. pins give memory back), then retries
static bool clip_lend_unpark(mtmd_subcontext * ctx, size_t need) {
    auto & L = ctx->fk.lend;
    L.file_err = false;
    const int r = clip_lend_unpark_once(ctx);
    if (r == 0) {
        return true;
    }
    if (r == 2) {
        L.file_err = true; // [TAG_R34_F_VISFILE_ERR] a file error: no OOM hook, no retry (the read logged why)
        return false;
    }
    if (L.oom == nullptr) {
        return false;
    }
    const bool freed = L.oom(L.oom_ud, need);
    LOG_WRN("%s: [TAG_FN_L22_OOMHOOK] the weight buffer could not be allocated, the OOM hook %s (%.1f MiB)\n", __func__,
            freed ? "released memory: retrying once" : "released nothing", need/1048576.0);
    if (!freed) {
        return false;
    }
    const int r2 = clip_lend_unpark_once(ctx);
    L.file_err = r2 == 2; // [TAG_R34_F_VISFILE_ERR]
    return r2 == 0;
}

static int clip_backend_index(const mtmd_subcontext * ctx) {
    int bi = -1;
    for (size_t i = 0; i < ctx->backends.ptrs.size(); ++i) {
        if (ctx->backends.ptrs[i] == ctx->backends.backend) {
            bi = (int) i;
        }
    }
    return bi;
}

// [TAG_FN_L22_VLEND_SIZE] at park, while the weights are still on the device: the compute buffer of the largest picture
// (no allocation). c_w is the warm-up picture's, which today's lend already covers
static void clip_lend_probe_sizes(mtmd_subcontext * ctx, size_t c_w) {
    auto & L = ctx->fk.lend;
    const auto & hp = ctx->model.hparams;
    L.sz = mtmd_l22_lend_size();
    if (!L.size_on || ctx->model.modality != MTMD_MODALITY_VISION || hp.patch_size <= 0 || hp.warmup_image_size <= 0) {
        return;
    }
    const int ps = hp.patch_size;
    L.sz.on  = true;
    L.sz.c_w = c_w;
    L.sz.n_w = (int64_t) (hp.warmup_image_size / ps) * (int64_t) (hp.warmup_image_size / ps);
    if (hp.image_max_pixels <= 0) {
        LOG_INF("%s: [TAG_FN_L22_VLEND_SIZE] no largest picture size: takes are sized from earlier encodes only\n", __func__);
        return;
    }
    const int align = ps * std::max(1, hp.n_merge);
    const int side  = ((int) std::sqrt((double) hp.image_max_pixels)) / align * align;
    if (side <= hp.warmup_image_size) {
        return; // the warm-up picture is the largest one
    }
    const int bi = clip_backend_index(ctx);
    if (bi < 0) {
        return;
    }
    const int64_t t0 = ggml_time_us();
    size_t c_max = 0;
    try {
        mtmd_image_f32_batch b;
        mtmd_image_f32 img;
        img.set_size({ side, side }, false, false);
        b.entries.push_back(std::move(img));
        ggml_cgraph * gf = mtmd_get_graph_builder(ctx, b)->build();
        std::vector<size_t> sizes(ctx->backends.ptrs.size(), 0);
        ggml_backend_sched_reserve_size(ctx->backends.sched.get(), gf, sizes.data());
        c_max = sizes[(size_t) bi];
    } catch (const std::exception & e) {
        LOG_WRN("%s: [TAG_FN_L22_VLEND_SIZE] the largest picture's graph failed (%s): takes are sized from earlier encodes only\n",
                __func__, e.what());
        return;
    }
    L.sz.n_max = (int64_t) (side / ps) * (int64_t) (side / ps);
    L.sz.c_max = c_max;
    llama_fn_l22_log("media", "[TAG_FN_L22_VLEND_SIZE] projector compute %.0f MiB at %lld patches (warm-up), %.0f MiB at %lld patches "
            "(%dx%d, the largest picture), measured in %.1f ms", c_w/1048576.0, (long long) L.sz.n_w, c_max/1048576.0,
            (long long) L.sz.n_max, side, side, (ggml_time_us() - t0)/1000.0);
}

size_t mtmd_fork_lend_park(mtmd_subcontext * ctx, mtmd_fork_lend_take_fn take, mtmd_fork_lend_give_fn give, void * ud) {
    if (ctx == nullptr || ctx->fk.lend.on || ctx->fk.no_alloc || ctx->fk.cpu_only || !ctx->model.buf || !ctx->backends.sched ||
        ctx->backends.backend == nullptr || ctx->backends.backend == ctx->backends.backend_cpu) {
        return 0;
    }
    mtmd_model & model = fork_model_rw(ctx);
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(model.buf.get());
    if (ggml_backend_buft_is_host(buft)) {
        return 0;
    }
    auto & L = ctx->fk.lend;
    size_t n = 0;
    L.offs.clear();
    size_t n_all = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(model.ctx_data.get()); t != nullptr; t = ggml_get_next_tensor(model.ctx_data.get(), t)) {
        if (t->view_src != nullptr || t->data == nullptr) {
            continue;
        }
        L.offs.push_back({ t, 0 });
        n_all += ggml_nbytes(t);
    }
    if (n_all == 0) {
        return 0;
    }
    const int64_t t0 = ggml_time_us();
    // [TAG_R34_F_VISFILE] the weights the file holds byte for byte stay there; only the others get a host copy
    L.foff.assign(L.offs.size(), UINT64_MAX);
    L.file_bytes = 0;
    if (clip_visfile_wanted()) {
        clip_lend_map_file(ctx);
    }
    for (size_t i = 0; i < L.offs.size(); ++i) {
        if (L.foff[i] != UINT64_MAX) {
            continue;
        }
        const size_t off = GGML_PAD(n, 64);
        L.offs[i].second = off;
        n = off + ggml_nbytes(L.offs[i].first);
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(ctx->backends.backend);
    ggml_backend_buffer_type_t hbt = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
    if (n > 0 && hbt != nullptr) {
        L.host_buf.reset(ggml_backend_buft_alloc_buffer(hbt, n));
    }
    if (n == 0) {
        L.host = nullptr;
    } else if (L.host_buf) {
        L.host = (uint8_t *) ggml_backend_buffer_get_base(L.host_buf.get());
    } else {
        try {
            L.host_vec.resize(n);
        } catch (const std::exception & e) {
            LOG_WRN("%s: [TAG_FN_L17_VISION_LEND] no host memory for the weights (%s): not parked\n", __func__, e.what());
            L.offs.clear();
            L.foff.clear();
            return 0;
        }
        L.host = L.host_vec.data();
    }
    if (L.file_bytes > 0) {
        // [TAG_R34_F_VISFILE] two staging halves of 16 MiB for the reads (pinned when the device has a host buffer type)
        L.stage_half = (size_t) 16 << 20;
        if (hbt != nullptr) {
            L.stage_buf.reset(ggml_backend_buft_alloc_buffer(hbt, 2*L.stage_half));
        }
        if (L.stage_buf) {
            L.stage = (uint8_t *) ggml_backend_buffer_get_base(L.stage_buf.get());
        } else {
            try {
                L.stage_vec.resize(2*L.stage_half);
                L.stage = L.stage_vec.data();
            } catch (const std::exception &) {
                L.file_bytes = 0; // no staging: not parked (the host copy path would need the bytes again)
                L.offs.clear();
                L.foff.clear();
                L.host_buf.reset();
                L.host_vec.clear();
                LOG_WRN("%s: [TAG_R34_F_VISFILE] no staging memory: not parked\n", __func__);
                return 0;
            }
        }
    }
    for (size_t i = 0; i < L.offs.size(); ++i) {
        if (L.foff[i] == UINT64_MAX) {
            ggml_backend_tensor_get(L.offs[i].first, L.host + L.offs[i].second, 0, ggml_nbytes(L.offs[i].first));
        }
    }
    const size_t w = ggml_backend_buffer_get_size(model.buf.get());
    const size_t c = ggml_backend_sched_get_buffer_size(ctx->backends.sched.get(), ctx->backends.backend);
    L.w = w;
    clip_lend_probe_sizes(ctx, c); // [TAG_FN_L22_VLEND_SIZE]
    L.buft  = buft;
    L.take  = take;
    L.give  = give;
    L.ud    = ud;
    L.bytes = w + c;
    L.on    = true;
    clip_lend_free_device(ctx);
    if (L.file_bytes > 0) {
        size_t n_file = 0;
        for (const auto fo : L.foff) {
            n_file += fo != UINT64_MAX ? 1 : 0;
        }
        LOG_INF("%s: [TAG_R34_F_VISFILE] %.1f MiB of weights (%zu of %zu tensors) wait in the projector file, read again at each "
                "take; %.1f MiB in a host copy\n", __func__, L.file_bytes/1048576.0, n_file, L.offs.size(), n/1048576.0);
    }
    LOG_INF("%s: [TAG_FN_L17_VISION_LEND] weights (%.1f MiB, %zu tensors) wait in %s, %.1f MiB of %s freed until "
            "an encode (%.1f ms)\n", __func__, n_all/1048576.0, L.offs.size(),
            L.file_bytes == n_all ? "the projector file" : L.host_buf ? "pinned host memory" : "pageable host memory",
            L.bytes/1048576.0, ggml_backend_name(ctx->backends.backend), (ggml_time_us() - t0)/1000.0);
    return L.bytes;
}

ggml_backend_dev_t mtmd_fork_get_device(const mtmd_subcontext * ctx) {
    if (ctx == nullptr || ctx->backends.backend == nullptr || ctx->backends.backend == ctx->backends.backend_cpu) {
        return nullptr;
    }
    return ggml_backend_get_device(ctx->backends.backend);
}

// [TAG_FN_L22_VLEND_HOLD] park (if the device side is up) and give the VRAM back, once per take. Lend mutex held
static void clip_lend_release(mtmd_subcontext * ctx, const char * why) {
    auto & L = ctx->fk.lend;
    if (!L.out) {
        return;
    }
    const int64_t t0 = ggml_time_us();
    if (!L.parked && !ctx->fk.cpu_only) {
        clip_lend_free_device(ctx);
    }
    const int64_t t1 = ggml_time_us();
    if (L.give) {
        L.give(L.ud);
    }
    if (why != nullptr) {
        llama_fn_l22_log("media", "[TAG_FN_L22_VLEND_HOLD] %s: %" PRIu64 " encodes on one lend of %.0f MiB, freed in %.1f ms, returned in "
                "%.1f ms", why, L.n_in_take, L.taken/1048576.0, (t1 - t0)/1000.0, (ggml_time_us() - t1)/1000.0);
    }
    L.out       = false;
    L.taken     = 0;
    L.n_in_take = 0;
    L.fresh     = false;
}

// [TAG_FN_L22_VLEND_SIZE] the bytes to take for an encode: today's bytes, or the weights plus the compute buffer this
// picture needs when that is more (never less: a short take allocates from nearly full VRAM)
static size_t clip_lend_need(const mtmd_subcontext * ctx, const mtmd_image_f32_batch * imgs, int & ne, int & nx, int & ny) {
    const auto & L = ctx->fk.lend;
    ne = nx = ny = 0;
    if (!L.sz.on || imgs == nullptr || imgs->entries.empty()) {
        return L.bytes;
    }
    const int ps = std::max(1, ctx->model.hparams.patch_size);
    int64_t n = 0;
    for (const auto & e : imgs->entries) {
        n += (int64_t) (e.nx() / ps) * (int64_t) (e.ny() / ps);
    }
    ne = (int) imgs->entries.size();
    nx = imgs->entries[0].nx();
    ny = imgs->entries[0].ny();
    return std::max(L.bytes, L.w + L.sz.estimate(n, L.sz.find(ne, nx, ny)));
}

// around one encode of a parked subcontext: VRAM back, weights up; at scope exit both freed again and the VRAM returned.
// [TAG_FN_L22_VLEND_HOLD] inside a hold the device side stays up for the next encode
struct clip_lend_scope {
    mtmd_subcontext * ctx;
    std::unique_lock<std::recursive_mutex> lk;
    bool active = false; // took the VRAM for this encode
    bool kept   = false; // found the device side kept by a hold
    bool ok     = true;
    int ne = 0, nx = 0, ny = 0;
    size_t need = 0;
    int64_t t0 = 0, t_take = 0, t_up = 0;
    size_t got = 0;

    clip_lend_scope(mtmd_subcontext * c, const mtmd_image_f32_batch * imgs) : ctx(c) {
        auto & L = ctx->fk.lend;
        if (!L.on || ctx->fk.cpu_only) {
            return; // no lock: a CPU encode can take minutes
        }
        lk = std::unique_lock<std::recursive_mutex>(L.mu);
        if (ctx->fk.cpu_only) {
            return;
        }
        need = clip_lend_need(ctx, imgs, ne, nx, ny);
        if (!L.parked) {
            if (L.out && need <= L.taken) {
                kept = true;
                L.n_in_take++;
                return;
            }
            clip_lend_release(ctx, "a bigger picture needs a bigger lend"); // then the bigger take below
        }
        if (!L.parked) {
            return;
        }
        active = true;
        t0  = ggml_time_us();
        got = L.take ? L.take(L.ud, need) : 0;
        t_take = ggml_time_us();
        L.out       = true;
        L.taken     = need;
        L.n_in_take = 1;
        ok = clip_lend_unpark(ctx, need);
        L.fresh = ok;
        t_up = ggml_time_us();
        if (!ok && L.file_err) {
            LOG_ERR("%s: [TAG_R34_F_VISFILE_ERR] the projector's weights could not be read from its file: this encode fails\n",
                    __func__);
        } else if (!ok) {
            LOG_ERR("%s: [TAG_FN_L17_VISION_LEND] the weight buffer (%.1f MiB) could not be allocated: this encode fails\n", __func__,
                    need/1048576.0);
        } else if (need > L.bytes) {
            llama_fn_l22_log("media", "[TAG_FN_L22_VLEND_SIZE] picture %dx%d (%d entries): lend %.0f MiB instead of %.0f MiB",
                    nx, ny, ne, need/1048576.0, L.bytes/1048576.0);
        }
    }

    ~clip_lend_scope() {
        if (!active && !kept) {
            return;
        }
        auto & L = ctx->fk.lend;
        // [TAG_FN_L22_VLEND_SIZE] a compute buffer allocated fresh for this picture: its size is exact for the next one
        if (L.sz.on && L.fresh && ok && !ctx->fk.cpu_only && !L.parked && ne > 0 && ctx->backends.sched) {
            L.sz.record(ne, nx, ny, ggml_backend_sched_get_buffer_size(ctx->backends.sched.get(), ctx->backends.backend));
        }
        L.fresh = false;
        L.n++;
        const int64_t t_enc = ggml_time_us();
        if (L.hold > 0 && ok && !ctx->fk.cpu_only && !L.parked) {
            if (active) {
                LOG_INF("%s: [TAG_FN_L22_VLEND_HOLD] encode %" PRIu64 ": VRAM back %.0f MiB in %.1f ms, weights up in %.1f ms, encode "
                        "%.1f ms, kept for the next encode (hold)\n", "clip_encode", L.n, got/1048576.0, (t_take - t0)/1000.0,
                        (t_up - t_take)/1000.0, (t_enc - t_up)/1000.0);
            }
            return;
        }
        if (!active) {
            clip_lend_release(ctx, "released after the encode");
            return;
        }
        if (!L.parked && !ctx->fk.cpu_only) {
            clip_lend_free_device(ctx);
        }
        const int64_t t_free = ggml_time_us();
        if (L.give) {
            L.give(L.ud);
        }
        L.out       = false;
        L.taken     = 0;
        L.n_in_take = 0;
        const int64_t t_end = ggml_time_us();
        size_t w = 0;
        for (const auto & to : L.offs) {
            w += ggml_nbytes(to.first);
        }
        LOG_INF("%s: [TAG_FN_L17_VISION_LEND] encode %" PRIu64 ": VRAM back %.0f MiB in %.1f ms, weights up in %.1f ms (%.1f GiB/s), "
                "encode %.1f ms, freed in %.1f ms, returned in %.1f ms\n", "clip_encode", L.n, got/1048576.0,
                (t_take - t0)/1000.0, (t_up - t_take)/1000.0, t_up > t_take ? w/1073741824.0/((t_up - t_take)/1e6) : 0.0,
                (t_enc - t_up)/1000.0, (t_free - t_enc)/1000.0, (t_end - t_free)/1000.0);
    }

    clip_lend_scope(const clip_lend_scope &) = delete;
    clip_lend_scope & operator=(const clip_lend_scope &) = delete;
};

void mtmd_fork_lend_set_oom(mtmd_subcontext * ctx, mtmd_fork_lend_oom_fn fn, void * ud) {
    if (ctx == nullptr) {
        return;
    }
    std::lock_guard<std::recursive_mutex> lk(ctx->fk.lend.mu);
    ctx->fk.lend.oom    = fn;
    ctx->fk.lend.oom_ud = ud;
}

void mtmd_fork_lend_set_size(mtmd_subcontext * ctx, bool on) {
    if (ctx != nullptr) {
        ctx->fk.lend.size_on = on;
    }
}

size_t mtmd_fork_lend_max_bytes(const mtmd_subcontext * ctx) {
    if (ctx == nullptr || !ctx->fk.lend.on) {
        return 0;
    }
    const auto & L = ctx->fk.lend;
    size_t m = L.bytes;
    if (L.sz.on && L.sz.n_max > 0) {
        m = std::max(m, L.w + L.sz.estimate(L.sz.n_max, nullptr));
    }
    return m;
}

void mtmd_fork_lend_hold(mtmd_subcontext * ctx, bool on) {
    if (ctx == nullptr || !ctx->fk.lend.on) {
        return;
    }
    auto & L = ctx->fk.lend;
    std::lock_guard<std::recursive_mutex> lk(L.mu);
    if (on) {
        L.hold++;
    } else if (L.hold > 0 && --L.hold == 0) {
        clip_lend_release(ctx, "hold released");
    }
}

void mtmd_fork_lend_suspend(mtmd_subcontext * ctx) {
    if (ctx == nullptr || !ctx->fk.lend.on) {
        return;
    }
    auto & L = ctx->fk.lend;
    std::lock_guard<std::recursive_mutex> lk(L.mu);
    if (L.hold > 0 && L.out) {
        clip_lend_release(ctx, "parked for a decode inside the hold");
    }
}

void mtmd_fork_lend_hold_reset(mtmd_subcontext * ctx) {
    if (ctx == nullptr || !ctx->fk.lend.on) {
        return;
    }
    auto & L = ctx->fk.lend;
    std::lock_guard<std::recursive_mutex> lk(L.mu);
    L.hold = 0;
    clip_lend_release(ctx, "holds reset");
}

// [TAG_FN_L22_OOMHOOK] a lent subcontext whose compute buffer could not be allocated asks the OOM hook once and retries,
// before the device fallback moves every later encode to the CPU
static bool clip_encode_impl_lent(mtmd_subcontext * ctx, mtmd_subctx_encode_params * params) {
    if (ctx->encode_impl(params)) {
        return true;
    }
    auto & L = ctx->fk.lend;
    if (!ctx->fk.st.alloc_failed || !L.on || L.oom == nullptr || ctx->fk.cpu_only) {
        return false;
    }
    const size_t need = L.taken > L.w ? L.taken - L.w : (L.bytes > L.w ? L.bytes - L.w : L.bytes);
    const bool freed = L.oom(L.oom_ud, need);
    LOG_WRN("%s: [TAG_FN_L22_OOMHOOK] the encoder's compute buffer could not be allocated, the OOM hook %s (%.1f MiB)\n", __func__,
            freed ? "released memory: retrying once" : "released nothing", need/1048576.0);
    if (!freed) {
        return false;
    }
    return ctx->encode_impl(params);
}

// one graph with the device fallback (the body of encode inside the lend scope)
static bool clip_encode_one(mtmd_subcontext * ctx, mtmd_subctx_encode_params * params) {
    if (!clip_device_fallback_eligible(ctx)) {
        return clip_encode_impl_lent(ctx, params);
    }

    // [TAG_MTMD_DEVICE_FALLBACK] the device failed an earlier encode
    if (ctx->fk.cpu_only) {
        return clip_encode_on_cpu_twin(ctx, ctx->fk.cpu_twin->sctx.get(), params);
    }

    // [TAG_MTMD_DEVICE_FALLBACK] proactive guard: a large image on an iGPU is encoded on the CPU
    const int igpu_max_tokens = clip_igpu_max_tokens();
    const auto & entries = params->imgs->entries;
    if (igpu_max_tokens > 0 && ctx->model.modality == MTMD_MODALITY_VISION && !entries.empty() &&
        ggml_backend_dev_type(ggml_backend_get_device(ctx->backends.backend)) == GGML_BACKEND_DEVICE_TYPE_IGPU) {
        int n_tokens = ctx->model.n_output_tokens(&entries[0]);
        if (ctx->support_batch || params->imgs->fn_n_images > 1) { // [TAG_FN_L22_VLBATCH] stills count each
            n_tokens = 0;
            for (const auto & entry : entries) {
                n_tokens += ctx->model.n_output_tokens(&entry);
            }
        }
        if (n_tokens > igpu_max_tokens) {
            mtmd_subcontext * twin = clip_get_cpu_twin(ctx);
            if (twin != nullptr) {
                LOG_INF("%s: image of %d tokens is above MTMD_IGPU_MAX_TOKENS=%d, encoding it on the CPU instead of %s\n",
                        "clip_encode", n_tokens, igpu_max_tokens, ggml_backend_name(ctx->backends.backend));
                return clip_encode_on_cpu_twin(ctx, twin, params);
            }
        }
    }

    std::string reason;
    bool threw = false; // [TAG_MTMD_DEVICE_FALLBACK] only a backend that threw may be lost; see backend_abandoned below
    try {
        if (clip_encode_impl_lent(ctx, params)) {
            return true;
        }
        if (!ctx->fk.st.device_failed) {
            return false; // the input was rejected, the device is fine
        }
        reason = "graph allocation or compute returned an error";
    } catch (const std::exception & e) {
        if (!ctx->fk.st.graph_built) {
            throw; // the graph builder rejected the input, not a device failure
        }
        reason = e.what();
        threw  = true;
    } catch (...) {
        if (!ctx->fk.st.graph_built) {
            throw;
        }
        reason = "unknown exception";
        threw  = true;
    }

    const std::string dev_name = ggml_backend_name(ctx->backends.backend);
    LOG_WRN("%s: WARNING: the %s encoder failed on %s (%s); it runs on the CPU from now on, retrying this encode on the CPU\n",
            "clip_encode", ctx->model.modality == MTMD_MODALITY_AUDIO ? "audio" : "vision", dev_name.c_str(), reason.c_str());

    mtmd_subcontext * twin = clip_get_cpu_twin(ctx);
    if (twin == nullptr) {
        LOG_ERR("%s: no CPU copy of the encoder, this encode fails\n", "clip_encode");
        return false;
    }

    // free what the failed device holds. A backend that threw (Vulkan throws vk::DeviceLostError) may be lost, and
    // its teardown would wait on it: it is abandoned, not freed (see mtmd_backends::backend_abandoned). A backend that
    // returned an error (Vulkan: only a failed buffer allocation, the device is fine) is freed as usual by ~mtmd_backends.
    // [TAG_R39_MTMD] the scheduler is shared: a new one without compute buffers instead of none
    ctx->backends.remake_sched();
    fork_model_rw(ctx).buf.reset();
    ctx->is_allocated = false;
    ctx->backends.backend_abandoned = ctx->backends.backend_abandoned || threw;
    ctx->fk.cpu_only = true;

    return clip_encode_on_cpu_twin(ctx, twin, params);
}

// [TAG_FN_L22_VLBATCH] ----------------------------------------------------------------------------------------------------
// A batch of B same-size stills is one graph when its compute buffer fits the device memory this encode may use, else it
// runs as balanced groups. Limit: the compute part of the current lend; without a lend what the scheduler holds (at least
// the warm-up reserve); no limit on the CPU. While pins are active, no graph may have more patches than one already run
// (the CUDA pool then reaches no new high-water).

std::vector<int> mtmd_fork_l22_split_plan(int n, const std::function<bool(int)> & fits) {
    std::vector<int> plan;
    if (n <= 0) {
        return plan;
    }
    int kmax = 1;
    for (int k = n; k > 1; --k) {
        if (fits(k)) {
            kmax = k;
            break;
        }
    }
    const int groups = (n + kmax - 1) / kmax;
    for (int g = 0; g < groups; ++g) {
        plan.push_back(n / groups + (g < n % groups ? 1 : 0));
    }
    return plan;
}

static size_t clip_fn_l22_limit(const mtmd_subcontext * ctx) {
    if (ctx->fk.fn_limit_test > 0) {
        return ctx->fk.fn_limit_test;
    }
    if (ctx->fk.cpu_only || ctx->backends.backend == nullptr || ctx->backends.backend == ctx->backends.backend_cpu) {
        return SIZE_MAX;
    }
    const auto & L = ctx->fk.lend;
    if (L.on) {
        const size_t t = L.out ? L.taken : L.bytes;
        return t > L.w ? t - L.w : 0;
    }
    const size_t cur = ctx->backends.sched ? ggml_backend_sched_get_buffer_size(ctx->backends.sched.get(), ctx->backends.backend) : 0;
    return std::max(cur, ctx->fk.fn_reserve_compute);
}

// device compute bytes of a k-image graph of nx x ny pictures, measured with a scheduler that never allocates (the
// weights must be on the device: parked weights would count as compute)
size_t mtmd_fork_l22_need_bytes(mtmd_subcontext * ctx, int nx, int ny, int k) {
    auto & f = ctx->fk;
    const auto key = std::make_tuple(nx, ny, k, (int) ctx->flash_attn_type);
    const auto it  = f.fn_need.find(key);
    if (it != f.fn_need.end()) {
        return it->second;
    }
    const int bi = clip_backend_index(ctx);
    GGML_ASSERT(bi >= 0);
    if (!f.fn_sched_measure) {
        f.fn_sched_measure.reset(
            ggml_backend_sched_new(ctx->backends.ptrs.data(), ctx->backends.bufts.data(), ctx->backends.ptrs.size(), 8192, false, true));
    }
    const int64_t t0 = ggml_time_us();
    mtmd_image_f32_batch b;
    for (int i = 0; i < k; ++i) {
        mtmd_image_f32 img;
        img.set_size({ nx, ny }, true, false); // placeholder: the graph reads only the size
        b.entries.push_back(std::move(img));
    }
    b.fn_n_images = k > 1 ? k : 0;
    ggml_cgraph * gf = mtmd_get_graph_builder(ctx, b)->build();
    std::vector<size_t> sizes(ctx->backends.ptrs.size(), 0);
    ggml_backend_sched_reserve_size(f.fn_sched_measure.get(), gf, sizes.data());
    const size_t need = sizes[(size_t) bi];
    f.fn_need[key] = need;
    LOG_INF("%s: [TAG_FN_L22_VLBATCH] %d x %dx%d: %.1f MiB compute on %s (measured in %.1f ms)\n", __func__, k, nx, ny,
            need/1048576.0, ggml_backend_name(ctx->backends.backend), (ggml_time_us() - t0)/1000.0);
    return need;
}

static bool clip_fn_l22_encode_stills(mtmd_subcontext * ctx, mtmd_subctx_encode_params * params) {
    const mtmd_image_f32_batch & imgs = *params->imgs;
    const int B = (int) imgs.entries.size();
    if (!ctx->fk.fn_vlbatch_on || imgs.fn_n_images != B || params->out_embd == nullptr || ctx->model.hparams.patch_size <= 0) {
        return clip_encode_one(ctx, params); // encode_impl rejects a wrong batch
    }
    const mtmd_image_f32 & e0 = imgs.entries[0];
    const int     ps    = ctx->model.hparams.patch_size;
    const int64_t p1    = (int64_t) (e0.nx() / ps) * (int64_t) (e0.ny() / ps);
    const size_t  limit = clip_fn_l22_limit(ctx);
    const bool    pins  = limit != SIZE_MAX && llama_fn_pin_active();
    const int64_t hw    = std::max(ctx->fk.fn_hw_patches, p1);
    auto fits = [&](int k) {
        if (pins && (int64_t) k * p1 > hw) {
            return false;
        }
        return limit == SIZE_MAX || mtmd_fork_l22_need_bytes(ctx, e0.nx(), e0.ny(), k) <= limit;
    };
    std::vector<int> plan;
    try {
        plan = mtmd_fork_l22_split_plan(B, fits);
    } catch (const std::exception & e) {
        LOG_WRN("%s: [TAG_FN_L22_VLBATCH] the batch graph could not be measured (%s): one picture per graph\n", __func__, e.what());
        plan.assign((size_t) B, 1);
    }
    if (plan.size() == 1) {
        return clip_encode_one(ctx, params);
    }

    ctx->fk.fn_n_split++;
    std::string groups;
    for (const int k : plan) {
        groups += (groups.empty() ? "" : "+") + std::to_string(k);
    }
    const auto it_b = ctx->fk.fn_need.find(std::make_tuple(e0.nx(), e0.ny(), B, (int) ctx->flash_attn_type));
    LOG_WRN("%s: [TAG_FN_L22_VLBATCH] %d pictures %dx%d: %.0f MiB compute, limit %.0f MiB%s: groups %s\n", __func__, B,
            e0.nx(), e0.ny(), it_b != ctx->fk.fn_need.end() ? it_b->second/1048576.0 : 0.0, limit/1048576.0,
            pins ? " (pins: no new high-water)" : "", groups.c_str());
    // the lend's size record would describe one group, not this batch
    ctx->fk.lend.fresh = false;

    const size_t per = (size_t) ctx->model.n_output_tokens(&e0) * (size_t) ctx->model.n_mmproj_embd();
    auto & out = *params->out_embd;
    if (!out.empty() && out.size() != per * (size_t) B) {
        LOG_ERR("%s: [TAG_FN_L22_VLBATCH] output buffer has %zu elements but expected %zu\n", __func__, out.size(), per * (size_t) B);
        return false;
    }
    int done = 0;
    for (const int k : plan) {
        mtmd_image_f32_batch sub;
        sub.entries.assign(imgs.entries.begin() + done, imgs.entries.begin() + done + k);
        sub.fn_n_images = k > 1 ? k : 0;
        std::vector<float> sub_out(per * (size_t) k);
        mtmd_subctx_encode_params sp = *params;
        sp.imgs     = &sub;
        sp.out_embd = &sub_out;
        if (!clip_encode_one(ctx, &sp)) {
            return false;
        }
        if (!out.empty()) {
            std::memcpy(out.data() + per * (size_t) done, sub_out.data(), per * (size_t) k * sizeof(float));
        }
        done += k;
    }
    return true;
}

// [TAG_R39_MTMD] the subcontext's public encode (v0.29.0's clip_encode): the lend scope, the stills batch, the device
// fallback around upstream's encode (encode_impl)
bool mtmd_subcontext::encode(mtmd_subctx_encode_params * params) {
    clip_lend_scope lend(this, params ? params->imgs : nullptr); // [TAG_FN_L17_VISION_LEND] no-op unless parked
    if (!lend.ok) {
        return false;
    }
    if (params != nullptr && params->imgs != nullptr && params->imgs->fn_n_images > 1) {
        return clip_fn_l22_encode_stills(this, params); // [TAG_FN_L22_VLBATCH]
    }
    return clip_encode_one(this, params);
}

bool mtmd_fork_l22_enable_batch(mtmd_subcontext * ctx) {
    if (ctx == nullptr || ctx->model.proj_type != PROJECTOR_TYPE_QWEN3VL || ctx->model.modality != MTMD_MODALITY_VISION) {
        return false;
    }
    ctx->fk.fn_vlbatch_on = true;
    if (ctx->fk.cpu_twin != nullptr) {
        ctx->fk.cpu_twin->sctx->fk.fn_vlbatch_on = true;
    }
    return true;
}

bool mtmd_fork_l22_batch_on(const mtmd_subcontext * ctx) {
    return ctx != nullptr && ctx->fk.fn_vlbatch_on;
}

int32_t mtmd_fork_l22_reserve_tokens(const mtmd_subcontext * ctx) {
    return ctx != nullptr ? ctx->fk.fn_reserve_tokens : 0;
}

void mtmd_fork_l22_set_test_limit(mtmd_subcontext * ctx, size_t bytes) {
    if (ctx != nullptr) {
        ctx->fk.fn_limit_test = bytes;
    }
}

uint64_t mtmd_fork_l22_n_split(const mtmd_subcontext * ctx) {
    return ctx != nullptr ? ctx->fk.fn_n_split : 0;
}
