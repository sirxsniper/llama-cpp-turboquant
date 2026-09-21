#pragma once

// [TAG_KV_RESOLVE] KV cache type resolver.
//
// llama_init_from_model runs llama_kv_resolve() before any KV cache is built. It keeps the requested -ctk/-ctv types
// when the model and the context settings support them and otherwise steps down, one LLAMA_LOG_WARN line per downgrade
// with the reason, instead of failing context creation:
//
//   turbot  -> turbo5p (turbo5p512 for 512-element KV rows) -> turbo4 -> q8_0 -> f16
//   turbo4p -> turbo4 -> q8_0 -> f16,  turbo2/turbo3/q4_0/... -> q8_0 -> f16
//
// turbo2/turbo3/turbo4 asked for with -ctk/-ctv keep every shape the zero-padding path serves (head 64/80/96 padded to
// 128, MLA K 576 padded to 640, 512-element heads through the F16-converted FA). As a step down the chain only the
// unpadded 128/256 heads are taken.
//
// Env LLAMA_KV_RESOLVE=0 turns the resolver off: an unsupported type then fails context creation as before.
//
// This header holds two things:
//   - the turbot cache preconditions, shared with the turbot cache constructor (llama-kv-cache.cpp), which still throws
//     "turbot: <reason>" on direct misuse. Each helper returns "" when its conditions hold, else the refusal text
//     without the "turbot: " prefix. The resolver calls the same helpers, so the two cannot drift apart.
//   - the pure decision function llama_kv_resolve() (llama-context.cpp): no model, no log line, no global state. The
//     caller describes the model and the settings in llama_kv_resolve_input; tests/test-kv-resolve.cpp drives it with
//     synthetic hparams.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-turbot.h"

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

// printf into a std::string (the header is also compiled into tests, which do not link llama-impl's format())
#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 1, 2)))
#endif
inline std::string llama_kv_resolve_fmt(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    char buf[512];
    const int n = vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n < 0) {
        return std::string();
    }
    return std::string(buf, (size_t) n < sizeof(buf) ? (size_t) n : sizeof(buf) - 1);
}

//
// turbot preconditions (docs/turbot/SPEC.md 9.3), shared by the cache constructor and the resolver
//

// the cache-level facts the turbot constructor refuses on
struct llama_turbot_cache_desc {
    bool     k_turbot     = false;
    bool     v_turbot     = false;
    uint32_t n_stream     = 1;       // 1 with --kv-unified or a single sequence
    bool     v_trans      = false;   // flash attention off
    bool     shared_cells = false;   // the cache views another cache's cells
    bool     mla          = false;
    bool     swa          = false;   // swa_type != LLAMA_SWA_TYPE_NONE || n_swa > 0
    uint32_t kv_size      = 0;
};

// the cache-level refusals, in the constructor's order
inline std::string llama_turbot_cache_refusal(const llama_turbot_cache_desc & d) {
    if (!d.k_turbot || !d.v_turbot) {
        return "needs -ctk turbot and -ctv turbot together";
    }
    if (d.n_stream > 1) {
        return "needs a single KV stream: use --kv-unified or -np 1";
    }
    if (d.v_trans) {
        return "needs flash attention";
    }
    if (d.shared_cells || d.mla) {
        return "shared cells and MLA caches are unsupported";
    }
    if (d.swa) {
        return "SWA caches are unsupported";
    }
    if (d.kv_size % GGML_TURBOT_GRANULE != 0) {
        return llama_kv_resolve_fmt("kv_size must be a multiple of %d, got %u", GGML_TURBOT_GRANULE, d.kv_size);
    }
    return "";
}

// the env variables the turbot cache rejects, read on every call as the constructor does
inline std::string llama_turbot_env_refusal() {
    const auto env_positive = [](const char * name) {
        const char * e = getenv(name);
        return e != nullptr && atoi(e) > 0;
    };
    if (env_positive("TURBO_KV_CPU_LAYERS")) {
        return "TURBO_KV_CPU_LAYERS is incompatible: attention KV must be on CUDA";
    }
    if (env_positive("TURBO_LAYER_ADAPTIVE")) {
        return "TURBO_LAYER_ADAPTIVE is incompatible: the plan sets the widths of every layer";
    }
    if (env_positive("TURBO_INNERQ")) {
        return "TURBO_INNERQ is incompatible";
    }
    return "";
}

// head geometry of one attention layer of the cache
inline std::string llama_turbot_layer_refusal(uint32_t il, uint32_t head_k, uint32_t head_v, uint32_t n_head_kv) {
    if (head_k != GGML_TURBOT_HEAD_DIM || head_v != GGML_TURBOT_HEAD_DIM || n_head_kv != GGML_TURBOT_N_HEAD) {
        return llama_kv_resolve_fmt("unsupported head geometry on layer %u: head_k %u, head_v %u, n_head_kv %u (needs %d, %d, %d)",
                il, head_k, head_v, n_head_kv, GGML_TURBOT_HEAD_DIM, GGML_TURBOT_HEAD_DIM, GGML_TURBOT_N_HEAD);
    }
    return "";
}

// where the KV of one attention layer lives: dev = model.dev_layer(il) with KV offload on, nullptr with it off
inline std::string llama_turbot_layer_device_refusal(uint32_t il, ggml_backend_dev_t dev) {
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (reg == nullptr || strcmp(ggml_backend_reg_name(reg), "CUDA") != 0) {
        return llama_kv_resolve_fmt("attention KV must be on a CUDA device (layer %u: %s)", il,
                dev ? ggml_backend_dev_name(dev) : "KV offload is off");
    }
    return "";
}

//
// resolver
//

// one layer of the model that carries KV
struct llama_kv_resolve_layer {
    int32_t     il        = 0;
    uint32_t    head_k    = 0;      // n_embd_head_k(il)
    uint32_t    head_v    = 0;      // n_embd_head_v(il)
    uint32_t    n_head_kv = 0;      // 0: the layer holds no KV row (skipped by the row checks)
    bool        attn      = true;   // held by the attention cache that would become turbot (turbot and plan checks)
    std::string dev_refusal;        // llama_turbot_layer_device_refusal() of the layer, "" when on CUDA (attn layers)
};

struct llama_kv_resolve_input {
    ggml_type type_k = GGML_TYPE_F16;   // requested; GGML_TYPE_TURBOT_S8 (any turbot type) means turbot
    ggml_type type_v = GGML_TYPE_F16;

    // the layers of every KV cache the context would build: each layer with a KV row, plus every layer the attention
    // cache holds (attn = true). The row checks run over all of them, the turbot and plan checks over the attn ones.
    std::vector<llama_kv_resolve_layer> layers;

    bool no_kv        = false;   // recurrent-only model: no KV cache, the types are ignored
    bool mla          = false;   // MLA: V is a view of K
    bool same_type    = false;   // K and V must share one type (MLA, DeepSeek V4)
    bool turbo_graph  = true;    // the attention graph of this arch applies the turbo query rotation
    bool swa          = false;   // swa_type != LLAMA_SWA_TYPE_NONE || n_swa > 0
    bool shared_cells = false;   // the cache would view another context's cells

    enum llama_flash_attn_type flash_attn = LLAMA_FLASH_ATTN_TYPE_AUTO;   // AUTO counts as on: quantized V forces it on

    uint32_t n_stream = 1;       // KV streams: 1 with --kv-unified or n_seq_max 1
    uint32_t kv_size  = 0;       // cells of the attention cache (n_ctx_seq)

    std::string env_refusal;     // llama_turbot_env_refusal() of the process
    std::string turbot_refused;  // non-empty: turbot is refused with this reason (the constructor threw it anyway)

    // true when the plan the turbot cache would load names exactly attn_layers, else false and why.
    // Unset: no plan is available.
    std::function<bool(const std::vector<int32_t> & attn_layers, uint32_t kv_size, std::string & why)> plan_check;
};

// one downgrade: from the requested type to the type the resolver kept
struct llama_kv_resolve_step {
    char        side = 'B';               // 'K', 'V', or 'B' for K and V together
    ggml_type   from = GGML_TYPE_F16;
    ggml_type   to   = GGML_TYPE_F16;
    std::string reason;                   // "<type>: <why>" for every type passed over, joined by "; "
};

struct llama_kv_resolve_result {
    ggml_type type_k = GGML_TYPE_F16;
    ggml_type type_v = GGML_TYPE_F16;
    bool      no_kv  = false;             // the model keeps no KV cache: the requested types were set aside
    std::vector<llama_kv_resolve_step> steps;
};

// the best K/V cache types for in; pure (llama-context.cpp)
LLAMA_API llama_kv_resolve_result llama_kv_resolve(const llama_kv_resolve_input & in);

// "turbot" for the turbot types, ggml_type_name() otherwise
LLAMA_API const char * llama_kv_resolve_type_name(ggml_type t);
