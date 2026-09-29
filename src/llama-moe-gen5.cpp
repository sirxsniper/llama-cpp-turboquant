// [TAG_MOE_DMA_SHARE] [TAG_MOE_PREFETCH] [TAG_FN_PREFILL_STREAM] context hooks shared by the gen5 paths
// (llama-moe-dma.cpp, llama-prefill-stream.cpp). See llama-moe-gen5.h.

#include "llama-moe-gen5.h"
#include "llama-moe-gen5-impl.h"

#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>

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

void copy_pool::run(int idx) {
    uint64_t seen = 0;
    for (;;) {
        uint8_t * d;
        const uint8_t * s;
        size_t n;
        {
            std::unique_lock<std::mutex> lk(mtx);
            cv_go.wait(lk, [&]() { return quit || gen != seen; });
            if (quit) {
                return;
            }
            seen = gen;
            d = dst_p;
            s = src_p;
            n = n_bytes;
        }
        size_t off, len;
        part_range(n, idx, size(), off, len);
        if (len > 0) {
            memcpy(d + off, s + off, len);
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
    const int parts = size();
    if (parts == 1 || n < (1u << 20)) {
        memcpy(dst, src, n);
        return;
    }
    {
        std::lock_guard<std::mutex> lk(mtx);
        dst_p   = (uint8_t *) dst;
        src_p   = (const uint8_t *) src;
        n_bytes = n;
        pending = parts - 1;
        gen++;
    }
    cv_go.notify_all();
    size_t off, len;
    part_range(n, 0, parts, off, len);
    if (len > 0) {
        memcpy((uint8_t *) dst + off, (const uint8_t *) src + off, len);
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
    const bool want_pfs = gen5::env_flag("LLAMA_PREFILL_STREAM");
    // [TAG_FN_MERGE] the bridge computes the cold experts on its host executor inside one device graph: there is no CPU
    // split for the plan and fence nodes. The two are exclusive; the bridge wins (its graphs never build a DMA plan).
    if (want_dma && host_bridge) {
        LLAMA_LOG_WARN("gen5: LLAMA_MOE_DMA_SHARE / LLAMA_MOE_PREFETCH are off in this context: LLAMA_MOE_BRIDGE owns the "
                       "host experts of its decode graphs (unset LLAMA_MOE_BRIDGE to use them)\n");
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
        llama_moe_dma_init_layers(layers, d, owner);
    }
}

void llama_moe_gen5_step(const void * owner) {
    llama_moe_dma_step(owner);
}

void llama_moe_gen5_free(const void * owner) {
    llama_moe_dma_free(owner);
    llama_prefill_stream_free(owner);
}
