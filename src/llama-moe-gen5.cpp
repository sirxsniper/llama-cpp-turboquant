// [TAG_MOE_DMA_SHARE] [TAG_MOE_PREFETCH] [TAG_FN_PREFILL_STREAM] context hooks shared by the gen5 paths
// (llama-moe-dma.cpp, llama-prefill-stream.cpp). See llama-moe-gen5.h.

#include "llama-moe-gen5.h"
#include "llama-moe-gen5-impl.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-fn-auto.h" // [TAG_FN_SHIP1] llama_fn_env

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>

#if (defined(_M_X64) || defined(__x86_64__)) && (defined(__AVX__) || (defined(_MSC_VER) && !defined(__clang__)))
#define LLAMA_GEN5_NT_AVX 1
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <immintrin.h>
#endif
#endif
#if defined(_M_X64) || defined(__x86_64__)
#    include <immintrin.h> // [TAG_FN_L15_NTCOPY]
#endif

namespace gen5 {

static std::atomic<size_t> g_pinned{0};

size_t pin_cap() {
    size_t mib = 4096;
    if (const char * e = getenv("LLAMA_GEN5_PIN_MAX_MIB"); e && atoi(e) > 0) {
        mib = std::min<size_t>(4096, (size_t) atoi(e));
    }
    return mib << 20;
}

host_mem::~host_mem() {
    release();
}

void host_mem::release() {
    if (buf) {
        ggml_backend_buffer_free(buf);
        g_pinned -= size;
        buf = nullptr;
    }
    plain.clear();
    plain.shrink_to_fit();
    ptr  = nullptr;
    size = 0;
}

bool host_mem::alloc(ggml_backend_dev_t dev, size_t n, const char * what) {
    release();
    ggml_backend_buffer_type_t hb = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
    if (hb && dev && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        if (g_pinned + n > pin_cap()) {
            LLAMA_LOG_WARN("gen5: %s: %.0f MiB pinned would pass the %.0f MiB cap (%.0f MiB in use)\n", what,
                    n/1048576.0, pin_cap()/1048576.0, g_pinned/1048576.0);
            return false;
        }
        buf = ggml_backend_buft_alloc_buffer(hb, n);
        if (!buf) {
            LLAMA_LOG_WARN("gen5: %s: pinned allocation of %.0f MiB failed\n", what, n/1048576.0);
            return false;
        }
        g_pinned += n;
        ptr    = (uint8_t *) ggml_backend_buffer_get_base(buf);
        pinned = true;
    } else {
        // CPU device (tests): plain memory, the copies are synchronous memcpy
        try {
            plain.assign(n, 0);
        } catch (...) {
            return false;
        }
        ptr    = plain.data();
        pinned = false;
    }
    size = n;
    return true;
}

bool env_flag(const char * name) {
    const char * e = getenv(name);
    return e && atoi(e) != 0;
}

int env_int(const char * name, int def, int lo, int hi) {
    const char * e = getenv(name);
    if (!e || !e[0]) {
        return def;
    }
    return std::max(lo, std::min(hi, atoi(e)));
}

bool sched_has(ggml_backend_sched_t sched, ggml_backend_t b) {
    if (!sched) {
        return true;
    }
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
        if (ggml_backend_sched_get_backend(sched, i) == b) {
            return true;
        }
    }
    return false;
}

void ev_record(ggml_backend_event_t ev, ggml_backend_t b) {
    if (ev && b) {
        ggml_backend_event_record(ev, b);
    }
}

void ev_wait(ggml_backend_t b, ggml_backend_event_t ev, ggml_backend_t src) {
    if (!b) {
        return;
    }
    if (ev) {
        ggml_backend_event_wait(b, ev);
    } else if (src) {
        ggml_backend_synchronize(src); // no events: order on the host
    }
}

void ev_sync(ggml_backend_event_t ev, ggml_backend_t src) {
    if (ev) {
        ggml_backend_event_synchronize(ev);
    } else if (src) {
        ggml_backend_synchronize(src);
    }
}

ggml_backend_event_t ev_new(ggml_backend_dev_t dev) {
    if (!dev || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        return nullptr;
    }
    return ggml_backend_event_new(dev);
}

// part idx of n parts, 4 KiB aligned
static void part_range(size_t n, int idx, int parts, size_t & off, size_t & len) {
    const size_t unit = 4096;
    const size_t units = (n + unit - 1) / unit;
    const size_t u0 = units * idx / parts;
    const size_t u1 = units * (idx + 1) / parts;
    off = std::min(n, u0 * unit);
    len = std::min(n, u1 * unit) - off;
}

void copy_pool::start(int n_threads) {
    stop();
    quit = false;
    for (int i = 1; i < n_threads; ++i) {
        workers.emplace_back([this, i]() { run(i); });
    }
}

void copy_pool::stop() {
    {
        std::lock_guard<std::mutex> lk(mtx);
        quit = true;
    }
    cv_go.notify_all();
    for (auto & t : workers) {
        t.join();
    }
    workers.clear();
}

// bytes [off, off + len) of the segments taken as one range
// [TAG_FN_L15_NTCOPY] a copy into the pinned ring with non-temporal stores whatever its size: the CRT memcpy uses them
// only above ~1.5 MiB, and a smaller piece (a run of experts, a thread's share of one) read each destination line first,
// twice the RAM traffic of the stream's copies. The ring is written once and read by the DMA, never by the CPU.
static void copy_nt(uint8_t * dst, const uint8_t * src, size_t n) {
#if defined(LLAMA_GEN5_NT_AVX)
#if !defined(__AVX__)
    // [TAG_FN_L15_NTAVX] MSVC builds the AVX intrinsics without /arch:AVX: use them only when the CPU and the OS have AVX
    static const bool has_avx = [] {
        int r[4];
        __cpuid(r, 1);
        return (r[2] & (1 << 27)) != 0 && (r[2] & (1 << 28)) != 0 && (_xgetbv(0) & 6) == 6;
    }();
    if (!has_avx) {
        memcpy(dst, src, n);
        return;
    }
#endif
    size_t i = 0;
    // head up to a 32-byte aligned destination
    const size_t head = std::min(n, (size_t) ((32 - ((uintptr_t) dst & 31)) & 31));
    if (head) {
        memcpy(dst, src, head);
        i = head;
    }
    for (; i + 128 <= n; i += 128) {
        const __m256i a = _mm256_loadu_si256((const __m256i *) (src + i));
        const __m256i b = _mm256_loadu_si256((const __m256i *) (src + i + 32));
        const __m256i c = _mm256_loadu_si256((const __m256i *) (src + i + 64));
        const __m256i d = _mm256_loadu_si256((const __m256i *) (src + i + 96));
        _mm256_stream_si256((__m256i *) (dst + i),      a);
        _mm256_stream_si256((__m256i *) (dst + i + 32), b);
        _mm256_stream_si256((__m256i *) (dst + i + 64), c);
        _mm256_stream_si256((__m256i *) (dst + i + 96), d);
    }
    if (i < n) {
        memcpy(dst + i, src + i, n - i);
    }
#else
    memcpy(dst, src, n);
#endif
}

static void copy_segs_part(const copy_seg * sg, int n_sg, size_t off, size_t len) {
    for (int i = 0; i < n_sg && len > 0; ++i) {
        if (off >= sg[i].n) {
            off -= sg[i].n;
            continue;
        }
        const size_t n = std::min(len, sg[i].n - off);
        copy_nt(sg[i].dst + off, sg[i].src + off, n);
        len -= n;
        off  = 0;
    }
#if defined(_M_X64) || defined(__x86_64__)
    _mm_sfence(); // the streamed lines are visible before the copy is reported done
#endif
}

void copy_pool::run(int idx) {
    uint64_t seen = 0;
    for (;;) {
        const copy_seg * sg;
        int n_sg;
        size_t n;
        {
            std::unique_lock<std::mutex> lk(mtx);
            cv_go.wait(lk, [&]() { return quit || gen != seen; });
            if (quit) {
                return;
            }
            seen = gen;
            sg   = segs_p;
            n_sg = n_segs;
            n    = n_bytes;
        }
        size_t off, len;
        part_range(n, idx, size(), off, len);
        if (len > 0) {
            copy_segs_part(sg, n_sg, off, len);
        }
        {
            std::lock_guard<std::mutex> lk(mtx);
            if (--pending == 0) {
                cv_done.notify_one();
            }
        }
    }
}

void copy_pool::copy(void * dst, const void * src, size_t n) {
    const copy_seg sg = { (uint8_t *) dst, (const uint8_t *) src, n };
    copy_list(&sg, 1);
}

void copy_pool::copy_list(const copy_seg * segs, int n_sg) {
    size_t n = 0;
    for (int i = 0; i < n_sg; ++i) {
        n += segs[i].n;
    }
    const int parts = size();
    if (parts == 1 || n < (1u << 20)) {
        copy_segs_part(segs, n_sg, 0, n);
        return;
    }
    {
        std::lock_guard<std::mutex> lk(mtx);
        segs_p  = segs;
        n_segs  = n_sg;
        n_bytes = n;
        pending = parts - 1;
        gen++;
    }
    cv_go.notify_all();
    size_t off, len;
    part_range(n, 0, parts, off, len);
    if (len > 0) {
        copy_segs_part(segs, n_sg, off, len);
    }
    std::unique_lock<std::mutex> lk(mtx);
    cv_done.wait(lk, [&]() { return pending == 0; });
}

} // namespace gen5

// host-resident expert layers of a loaded model, and the device their routers sit on
static bool gen5_model_layers(const llama_model & model, std::vector<llama_moe_gen5_layer_desc> & out, ggml_backend_dev_t & dev) {
    out.clear();
    dev = nullptr;
    // nextn (MTP) layers run in an MTP context's graphs only, never in the owner's: a prefill stream job for one would
    // copy it every ubatch and hold its bank until the next ubatch (the KV filter of create_memory)
    // [TAG_FN_PREFILL_STREAM] [TAG_MOE_DMA_SHARE]
    const auto & hp = model.hparams;
    const bool nextn_split = hp.n_layer_nextn > 0 && hp.n_layer() > 0 && hp.router_layer < 0;
    for (size_t il = 0; il < model.layers.size(); ++il) {
        const auto & l = model.layers[il];
        if (nextn_split && il >= hp.n_layer()) {
            continue;
        }
        if (!l.ffn_up_exps || !l.ffn_gate_exps || !l.ffn_down_exps || !l.ffn_gate_inp || l.ffn_gate_up_exps) {
            continue;
        }
        if (!l.ffn_up_exps->data || !l.ffn_gate_exps->data || !l.ffn_down_exps->data) {
            continue; // memory estimation model: weights not loaded
        }
        const ggml_tensor * w[3] = { l.ffn_up_exps, l.ffn_gate_exps, l.ffn_down_exps };
        bool host = true;
        for (const ggml_tensor * t : w) {
            host = host && t->buffer && ggml_backend_buffer_is_host(t->buffer) && t->ne[2] == l.ffn_up_exps->ne[2];
        }
        if (!host || !l.ffn_gate_inp->buffer || ggml_backend_buffer_is_host(l.ffn_gate_inp->buffer)) {
            continue;
        }
        ggml_backend_dev_t d = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(l.ffn_gate_inp->buffer));
        if (!d) {
            continue;
        }
        if (!dev) {
            dev = d;
        } else if (d != dev) {
            continue; // one device only: layers routed on another device keep the CPU path
        }
        llama_moe_gen5_layer_desc desc;
        desc.il   = (int) il;
        desc.up   = l.ffn_up_exps;
        desc.gate = l.ffn_gate_exps;
        desc.down = l.ffn_down_exps;
        out.push_back(desc);
    }
    return !out.empty();
}

void llama_moe_gen5_init(const llama_model & model, const void * owner, const std::vector<ggml_backend_t> & backends,
        bool host_bridge) {
    bool       want_dma = gen5::dma_requested();
    // [TAG_FN_L3_HOST_DMAOFF] LLAMA_FN_HOST_DMAOFF=1 (qwen4exp): no DMA share in any context of the model. Without it,
    // LLAMA_MOE_BRIDGE_DMA=0 leaves the share to the next context that asks: the MTP draft context then made a DMA state
    // of its own (banks, a 1 GiB pinned ring, fill threads, a warm start) that none of its graphs can use
    if (want_dma && llama_fn_l3_flag(model, "LLAMA_FN_HOST_DMAOFF")) {
        want_dma = false;
        static std::atomic<bool> logged{false};
        if (!logged.exchange(true)) {
            LLAMA_LOG_INFO("gen5: [TAG_FN_L3_HOST_DMAOFF] no DMA share: no banks, no pinned ring, no fill threads in any context "
                    "of this model\n");
        }
    }
    const bool want_pfs = gen5::env_flag("LLAMA_PREFILL_STREAM");
    // [TAG_FN_MERGE] the bridge computes the cold experts on its host executor inside one device graph: there is no CPU
    // split for the plan and fence nodes. The two are exclusive; the bridge wins (its graphs never build a DMA plan).
    // [TAG_FN_R4_BRIDGE_DMA] LLAMA_MOE_BRIDGE_DMA=1: the share runs inside the bridged graphs (bridge mode)
    const bool dma_bridge = want_dma && host_bridge && gen5::env_flag("LLAMA_MOE_BRIDGE_DMA");
    if (want_dma && host_bridge && !dma_bridge) {
        LLAMA_LOG_WARN("gen5: LLAMA_MOE_DMA_SHARE / LLAMA_MOE_PREFETCH are off in this context: LLAMA_MOE_BRIDGE owns the "
                       "host experts of its decode graphs (LLAMA_MOE_BRIDGE_DMA=1 runs the share inside them)\n");
        want_dma = false;
    }
    if (!want_dma && !want_pfs) {
        return;
    }
    std::vector<llama_moe_gen5_layer_desc> layers;
    ggml_backend_dev_t dev = nullptr;
    if (!gen5_model_layers(model, layers, dev)) {
        return; // no host expert layer (or a memory estimation model): a later context may own it
    }
    llama_moe_gen5_device d;
    d.dev = dev;
    for (ggml_backend_t b : backends) {
        if (ggml_backend_get_device(b) == dev) {
            d.compute = b;
            break;
        }
    }
    const ggml_tensor * router = model.layers[layers[0].il].ffn_gate_inp;
    d.buft = ggml_backend_buffer_get_type(router->buffer);
    if (!d.compute || !d.buft) {
        LLAMA_LOG_WARN("gen5: no backend for device %s in this context - off\n", ggml_backend_dev_name(dev));
        return;
    }
    if (want_pfs) {
        llama_prefill_stream_init_layers(layers, d, owner);
    }
    if (want_dma) {
        // [TAG_FN_SHIP1] the warm start reads the same routing profile as the hot set: the environment first, then the
        // qwen4exp profile's value (its <model>.moeprof sidecar), not the environment alone
        const char * prof = llama_fn_env(model, "LLAMA_MOE_DMA_PROFILE");
        if (!prof || !prof[0]) {
            prof = llama_fn_env(model, "LLAMA_MOE_HOT_PROFILE");
        }
        d.profile = prof ? prof : "";
        llama_moe_dma_init_layers(layers, d, owner, dma_bridge);
    }
}

void llama_moe_gen5_step(const void * owner) {
    llama_moe_dma_step(owner);
}

bool llama_moe_gen5_before_ubatch(const void * owner, ggml_backend_sched_t sched, int64_t n_tokens) {
    return llama_prefill_stream_before_ubatch(owner, sched, n_tokens); // [TAG_FN_R1_PFS_LEND]
}

void llama_moe_gen5_free(const void * owner) {
    llama_moe_dma_free(owner);
    llama_prefill_stream_free(owner);
}
