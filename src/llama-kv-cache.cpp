#include "llama-kv-cache.h"

#include <cstdlib>

#include <set>
#include "llama-triattention.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"
#include "llama-context.h"
#include "llama-kv-tier.h"
#include "llama-kv-cache-resolve.h"   // [TAG_KV_RESOLVE] turbot refusals shared with the KV type resolver

#include "ggml-turbot.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>

// InnerQ: cross-TU shared state. Defined in ggml-base (ggml-turbo-innerq.cpp)
// so it's available to both ggml-cuda.dll (CUDA path writes it) and llama.dll
// regardless of whether GGML_BACKEND_DL is on.
#include "ggml-turbo-innerq.h"

static bool ggml_is_power_of_2(int n) {
    return (n & (n - 1)) == 0;
}

// orthonormal Walsh-Hadamard rotation matrix
// note: res^2 == I
static void ggml_gen_hadamard(ggml_tensor * tensor) {
    assert(tensor->type == GGML_TYPE_F32);

    const int n = tensor->ne[0];

    assert(ggml_is_power_of_2(n));
    assert(tensor->ne[1] == n);
    assert(tensor->ne[2] == 1);
    assert(tensor->ne[3] == 1);

    std::vector<float> data_f32;

    float * data = (float *) tensor->data;

    if (tensor->type != GGML_TYPE_F32) {
        data_f32.resize(n*n);
        data = data_f32.data();
    }

    data[0*n + 0] = 1.0 / sqrtf(n);

    for (int s = 1; s < n; s *= 2) {
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s; j++) {
                const float val = data[i*n + j];

                data[(i + s)*n + (j    )] =  val;
                data[(i    )*n + (j + s)] =  val;
                data[(i + s)*n + (j + s)] = -val;
            }
        }
    }

    if (tensor->type != GGML_TYPE_F32) {
        ggml_quantize_chunk(tensor->type, data, tensor->data, 0, 1, n*n, nullptr);
    }
}

//
// llama_kv_cache
//

// [TAG_TURBO4P] All four turbo KV formats share the same 128-element WHT group, so
// anywhere the cache asks "is this a turbo type" the answer must include turbo4p_0.
// They differ only in how a block is packed, which is the FA path's problem, not this
// layer's. Kept as one predicate because this question is asked in a dozen places and
// they were drifting.
static inline bool llama_type_is_turbo(ggml_type t) {
    return t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 ||
           t == GGML_TYPE_TURBO4_0 || t == GGML_TYPE_TURBO4P_0 || t == GGML_TYPE_TURBO5P_0 ||
           t == GGML_TYPE_TURBO5P512_0;
}

llama_kv_cache::llama_kv_cache(
        const llama_model & model,
        const llama_hparams & hparams,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   unified,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
           llama_memory_t   mem_other,
    const layer_filter_cb & filter,
    const  layer_reuse_cb & reuse,
    const  layer_share_cb & share,
             const char *   name_tag) :
    model(model), hparams(hparams), v_trans(v_trans),
    n_seq_max(n_seq_max), n_stream(unified ? 1 : n_seq_max), n_pad(n_pad), n_swa(n_swa), swa_type(swa_type),
    other(static_cast<llama_kv_cache *>(mem_other)),
    v_cells_impl(other ? other->v_cells_impl : std::make_shared<llama_kv_cells_vec>()),
    v_cells(*v_cells_impl) {

    // shared cells view the source cache's K/V tensors, so the cell count
    // follows the source allocation: a fitted target can be smaller than the
    // draft default and oversized views would overflow the source tensors
    if (other) {
        const uint32_t size_other = other->get_size();
        if (kv_size != size_other) {
            LLAMA_LOG_WARN("%s: kv_size = %u overridden to %u to match the shared source cache\n", __func__, kv_size, size_other);
            kv_size = size_other;
        }
    }

    GGML_ASSERT(kv_size % n_pad == 0);

    const uint32_t n_layer = hparams.n_layer_all;

    // [TAG_TURBOT] tiered KV cache requested (GGML_TYPE_TURBOT_S8 is the sentinel; the plan picks each layer's type)
    const bool turbot_req = ggml_turbot_is_type(type_k) || ggml_turbot_is_type(type_v);

    // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    // create a context for each buffer type
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                // upstream: 2 tensors per layer (K, V), plus one view per tensor per stream
                // +3 for turbo rotation matrices (turbo_rotation + turbo_rotation_inv + turbo_innerq_scale_inv)
                // [TAG_TURBOT] +1 young pool per layer for a turbot cache
                // [TAG_TURBOT_ANY_STREAMS] + one pool view per stream per layer when there is more than one stream
                /*.mem_size   =*/ size_t((2u*(1 + n_stream)*n_layer + 3 +
                        (turbot_req ? n_layer*(n_stream > 1 ? 1 + n_stream : 1u) : 0u))*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map.emplace(buft, ctx);

            return ctx;
        }

        return it->second.get();
    };

    GGML_ASSERT(n_stream == 1 || n_stream == n_seq_max);

    v_heads.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_heads[s] = 0;
    }

    v_cells.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].resize(kv_size);
    }

    // by default, all sequence ids are mapped to the 0th stream
    seq_to_stream.resize(LLAMA_MAX_SEQ, 0);

    if (n_stream > 1) {
        seq_to_stream.resize(n_stream, 0);
        for (uint32_t s = 0; s < n_stream; ++s) {
            seq_to_stream[s] = s;
        }
    }

    // [TAG_V_CACHE_VARIABLE]
    if (v_trans && hparams.is_n_embd_v_gqa_variable()) {
        LLAMA_LOG_WARN("%s: the V embeddings have different sizes across layers and FA is not enabled - padding V cache to %d\n",
                __func__, hparams.n_embd_v_gqa_max());
    }

    const bool is_mla = hparams.is_mla();

    // [TAG_KV_CPU_LAYERS] Adaptive KV placement: park the KV of N attention layers in host RAM.
    //
    // Each attention layer's KV at ctx 262144 / turbo5p is 328 MiB (262144 x 1024 x 5.125/8 x 2),
    // so every layer moved off the GPU frees exactly that much VRAM. Nothing about the cached data
    // changes - same type, same precision, same context - so output is identical; only where the
    // bytes live changes.
    //
    // The cost is an extra PCIe read per offloaded layer per token, proportional to the context
    // ACTUALLY used: ~0.37 ms at 16K, ~1.5 ms at 64K, ~6 ms at 262K on PCIe 5 x16. Cheap for short
    // conversations, expensive for deep ones - measure at your real depth before committing.
    //
    // Layers are taken from the MIDDLE of the attention stack, leaving the first and last in VRAM.
    const uint32_t kv_cpu_layers = [] {
        const char * e = getenv("TURBO_KV_CPU_LAYERS");
        const int    n = e ? atoi(e) : 0;
        return (uint32_t) (n > 0 ? n : 0);
    }();

    // Build the offload set up front: we need to know how many layers this cache actually holds
    // before we can pick the middle ones, and the filter decides that.
    std::vector<uint32_t> kv_layers;
    if (kv_cpu_layers > 0) {
        for (uint32_t il = 0; il < n_layer; il++) {
            if (!filter || filter(il)) {
                kv_layers.push_back(il);
            }
        }
    }
    std::set<uint32_t> kv_on_cpu;
    if (kv_cpu_layers > 0 && !kv_layers.empty()) {
        const uint32_t n_cache  = (uint32_t) kv_layers.size();
        const uint32_t n_to_cpu = std::min(kv_cpu_layers, n_cache > 2 ? n_cache - 2 : 0u);
        if (n_to_cpu < kv_cpu_layers) {
            LLAMA_LOG_WARN("%s: TURBO_KV_CPU_LAYERS=%u capped to %u "
                           "(this cache holds %u layers; first and last stay on the GPU)\n",
                           __func__, kv_cpu_layers, n_to_cpu, n_cache);
        }
        // centre the selection so the first and last attention layers stay resident
        const uint32_t first = (n_cache - n_to_cpu) / 2;
        for (uint32_t i = 0; i < n_to_cpu; ++i) {
            kv_on_cpu.insert(kv_layers[first + i]);
        }
        if (n_to_cpu > 0) {
            LLAMA_LOG_INFO("%s: KV for %u of %u attention layers placed in HOST RAM "
                           "(TURBO_KV_CPU_LAYERS)\n", __func__, n_to_cpu, n_cache);
        }
    }

    // [TAG_TURBOT] tiered KV cache: refusals, plan and tier state (docs/turbot/SPEC.md 9.3-9.5). Every later turbot
    // branch in this file keys on turbot_plan (nullptr) / turbot_tier (empty, is_turbot()) for every other cache type.
    if (turbot_req) {
        const auto refuse = [](const std::string & msg) {
            throw std::runtime_error("turbot: " + msg);
        };

        // [TAG_KV_RESOLVE] the refusals live in llama-kv-cache-resolve.h, shared with the KV type resolver, which runs
        // them before any cache is built and falls back to turbo5p instead. Here they still throw on direct misuse.
        // TURBO_KV_CPU_LAYERS is refused through llama_turbot_env_refusal (kv_cpu_layers > 0 is the same test).
        llama_turbot_cache_desc desc;
        desc.k_turbot     = ggml_turbot_is_type(type_k);
        desc.v_turbot     = ggml_turbot_is_type(type_v);
        desc.n_stream     = n_stream;
        desc.v_trans      = v_trans;
        desc.shared_cells = other != nullptr;
        desc.mla          = is_mla;
        desc.swa          = swa_type != LLAMA_SWA_TYPE_NONE || n_swa > 0;
        desc.kv_size      = kv_size;

        std::string why = llama_turbot_cache_refusal(desc);
        if (why.empty()) {
            why = llama_turbot_env_refusal();
        }
        if (!why.empty()) {
            refuse(why);
        }

        // [TAG_TURBOT_ANY_GEOM] the shape of this cache: head dim and KV heads of every attention layer it holds, the
        // cells and the streams. llama_turbot_layer_refusal (llama-kv-cache-resolve.h) has checked K = V and a geometry
        // turbot has a layout for (4 KV heads x 256 only with LLAMA_TURBOT_ANY=0).
        llama_turbot_cache_shape shape;
        shape.kv_size   = kv_size;
        shape.n_stream  = n_stream;
        shape.n_seq_max = n_seq_max;
        shape.model     = &model;

        for (uint32_t il = 0; il < n_layer; il++) {
            if (!hparams.has_kv(il) || (filter && !filter(il))) {
                continue;
            }
            why = llama_turbot_layer_refusal(il, hparams.n_embd_head_k(il), hparams.n_embd_head_v(il), hparams.n_head_kv(il));
            if (why.empty()) {
                why = llama_turbot_layer_device_refusal(il, offload ? model.dev_layer(il) : nullptr);
            }
            if (!why.empty()) {
                refuse(why);
            }
            shape.layers.push_back({ (int32_t) il, (uint16_t) hparams.n_embd_head_k(il), (uint8_t) hparams.n_head_kv(il) });
        }

        // [TAG_TURBOT_EMBED_PLAN] plan source precedence (llama_turbot_plan_get_source): llama_turbot_set_plan_path
        // (--kv-tier-plan), then env LLAMA_TURBOT_PLAN, then the built-in default plan; "default" in either place selects
        // the built-in plan. A plan that does not name exactly attn_layers is refused here: code that wants to fall back
        // instead checks llama_turbot_plan_matches before the cache is built.
        // [TAG_TURBOT_ANY_PLAN] llama_context chooses the plan (llama_turbot_plan_choose: file, sidecar, built-in,
        // automatic) and hands it over in a llama_turbot_plan_scope; the loader parses that text against this shape.
        turbot_plan = std::make_unique<llama_turbot_plan>();

        std::string err;
        if (!llama_turbot_plan_load(shape, *turbot_plan, err)) {
            throw std::runtime_error(err);
        }

        // [TAG_TURBOT_ANY_STREAMS] one tier per stream, each with POOL / n_stream young pool rows in whole granules and the
        // plan's CAP. One stream: the one tier of before, with the whole pool.
        turbot_pool_s = turbot_plan->pool_cells / n_stream / GGML_TURBOT_GRANULE * GGML_TURBOT_GRANULE;
        for (uint32_t s = 0; s < n_stream; ++s) {
            turbot_tier.push_back(std::make_unique<llama_kv_tier>(kv_size, turbot_pool_s, turbot_plan->cap_cells));
        }
    }

    for (uint32_t il = 0; il < n_layer; il++) {
        if (!hparams.has_kv(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: does not have KV cache\n", __func__, il);
            continue;
        }

        if (filter && !filter(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: filtered\n", __func__, il);
            continue;
        }

        if (share && other) {
            const int32_t il_share = share(il);

            if (il_share >= 0) {
                const auto & layer_share = other->layers[other->map_layer_ids[il_share]];

                LLAMA_LOG_WARN("%s: layer %3d: sharing with layer %d. k = %p, v = %p\n", __func__, il, il_share,
                        layer_share.k->data, layer_share.v->data);

                map_layer_ids[il] = layers.size();

                layers.push_back(layer_share);
                layers.back().il = il;

                continue;
            }
        }

        if (n_embd_head_k_all == 0) {
            n_embd_head_k_all = (int32_t) hparams.n_embd_head_k(il);
        } else if (n_embd_head_k_all > 0 && n_embd_head_k_all != (int32_t) hparams.n_embd_head_k(il)) {
            n_embd_head_k_all = -1;
        }

        if (!is_mla) {
            if (n_embd_head_v_all == 0) {
                n_embd_head_v_all = (int32_t) hparams.n_embd_head_v(il);
            } else if (n_embd_head_v_all > 0 && n_embd_head_v_all != (int32_t) hparams.n_embd_head_v(il)) {
                n_embd_head_v_all = -1;
            }
        }

        // [TAG_V_CACHE_VARIABLE]
        const uint32_t n_embd_k_gqa =            hparams.n_embd_k_gqa(il);
        const uint32_t n_embd_v_gqa = !v_trans ? hparams.n_embd_v_gqa(il) : hparams.n_embd_v_gqa_max();

        const char * dev_name = "CPU";

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        // [TAG_KV_CPU_LAYERS] a layer in the offload set keeps the CPU buffer type, which leaves
        // its K and V in host RAM. Everything else about the layer is unchanged.
        const bool kv_to_cpu = kv_on_cpu.count(il) > 0;

        if (offload && !kv_to_cpu) {
            auto * dev = model.dev_layer(il);
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);
        } else if (kv_to_cpu) {
            dev_name = "CPU (TURBO_KV_CPU_LAYERS)";
        }

        LLAMA_LOG_DEBUG("%s: layer %3d: dev = %s\n", __func__, il, dev_name);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for kv cache");
        }

        // TurboQuant zero-padding: for models with non-128-aligned head_dim (e.g. DeepSeek
        // head_dim_k=192), pad each head to the next multiple of 128. The padded zeros don't
        // affect dot products since WHT preserves inner products:
        //   <WHT(Q_padded), WHT(K_padded)> = <Q_padded, K_padded> = <Q, K> + <0, 0> = <Q, K>
        const uint32_t n_embd_head_k = hparams.n_embd_head_k(il);


        const bool has_k = true;
        const bool has_v = !is_mla;

        // Layer-adaptive: use higher precision for quality-sensitive layers
        // Config: TURBO_LAYER_ADAPTIVE env var controls the strategy
        //   0 = uniform (default)
        //   1 = q8_0 K+V for first+last 4 layers
        //   2 = q8_0 K+V for last 8 layers
        //   5 = Boundary V: first2+last2 V=turbo4, rest V=turbo2 (K unchanged)
        //   6 = V-only: last 8 V=turbo4, rest V=turbo2 (K unchanged)
        //   7 = Boundary V (recommended): first2+last2 V=q8_0, rest V=turbo2 (K unchanged)
        ggml_type layer_type_k = type_k;
        ggml_type layer_type_v = type_v;
        {
            static const int adaptive_mode = [&]() {
                const char * env = getenv("TURBO_LAYER_ADAPTIVE");
                if (env) {
                    int mode = atoi(env);
                    if (mode > 0) {
                        LLAMA_LOG_INFO("llama_kv_cache: layer-adaptive mode %d enabled (env)\n", mode);
                    }
                    return mode;
                }
                // Auto-enable Boundary V (mode 7) when V is turbo2
                if (type_v == GGML_TYPE_TURBO2_0 && hparams.n_layer() >= 8) {
                    LLAMA_LOG_INFO("llama_kv_cache: Boundary V auto-enabled for turbo2-V (opt-out: TURBO_LAYER_ADAPTIVE=0)\n");
                    return 7;
                }
                return 0;
            }();
            const bool is_turbo = llama_type_is_turbo(type_k);
            const bool v_is_turbo = llama_type_is_turbo(type_v);
            const uint32_t n_layer = hparams.n_layer();
            if (adaptive_mode == 1 && is_turbo && n_layer >= 8) {
                if (il < 4 || il >= n_layer - 4) {
                    layer_type_k = GGML_TYPE_Q8_0;
                    layer_type_v = GGML_TYPE_Q8_0;
                }
            } else if (adaptive_mode == 2 && is_turbo && n_layer >= 8) {
                if (il >= n_layer - 8) {
                    layer_type_k = GGML_TYPE_Q8_0;
                    layer_type_v = GGML_TYPE_Q8_0;
                }
            } else if (adaptive_mode == 5 && v_is_turbo && n_layer >= 8) {
                // Boundary V (turbo4 boundaries): first2+last2 V=turbo4, rest V=turbo2
                const bool is_boundary = (il < 2 || il >= n_layer - 2);
                layer_type_v = is_boundary ? GGML_TYPE_TURBO4_0 : GGML_TYPE_TURBO2_0;
                if (il == 0) {
                    LLAMA_LOG_INFO("llama_kv_cache: Boundary V mode 5: first2+last2 V=turbo4, rest V=turbo2\n");
                }
            } else if (adaptive_mode == 6 && v_is_turbo && n_layer >= 8) {
                // V-only: last 8 V=turbo4, rest V=turbo2
                layer_type_v = (il >= n_layer - 8) ? GGML_TYPE_TURBO4_0 : GGML_TYPE_TURBO2_0;
                if (il == 0) {
                    LLAMA_LOG_INFO("llama_kv_cache: V-only LA mode 6: last8 V=turbo4, rest V=turbo2\n");
                }
            } else if (adaptive_mode == 7 && v_is_turbo && n_layer >= 8) {
                // Boundary V (recommended): first2+last2 V=q8_0, rest V=turbo2
                const bool is_boundary = (il < 2 || il >= n_layer - 2);
                layer_type_v = is_boundary ? GGML_TYPE_Q8_0 : GGML_TYPE_TURBO2_0;
                if (il == 0) {
                    LLAMA_LOG_INFO("llama_kv_cache: Boundary V mode 7: first2+last2 V=q8_0, rest V=turbo2\n");
                }
            }
        }
        // [TAG_TURBOT] the plan picks the base type of each layer-side: S = sum of its 4 old widths (turbot_s8..s24)
        if (turbot_plan) {
            const ggml_turbot_layer & tl = turbot_plan->layers.at((int32_t) il);
            layer_type_k = ggml_turbot_type_of_s(tl.k.s);
            layer_type_v = ggml_turbot_type_of_s(tl.v.s);
        }

        // For turbo types, pad K head_dim to next multiple of 128 for full WHT groups
        uint32_t n_embd_k_gqa_eff = n_embd_k_gqa;
        const bool k_is_turbo = llama_type_is_turbo(layer_type_k);
        if (k_is_turbo && n_embd_head_k % 128 != 0) {
            const uint32_t padded_head_k = ((n_embd_head_k + 127) / 128) * 128;
            const uint32_t n_head_kv = n_embd_k_gqa / n_embd_head_k;
            n_embd_k_gqa_eff = n_head_kv * padded_head_k;
            if (il == 0) {
                LLAMA_LOG_INFO("%s: turbo zero-padding K head_dim %u -> %u (cache %u -> %u)\n",
                               __func__, n_embd_head_k, padded_head_k, n_embd_k_gqa, n_embd_k_gqa_eff);
            }
        }

        // For turbo types, pad V head_dim to next multiple of 128 if needed
        const uint32_t n_embd_head_v = hparams.n_embd_head_v(il);
        uint32_t n_embd_v_gqa_eff = n_embd_v_gqa;
        const bool v_is_turbo = llama_type_is_turbo(layer_type_v);
        if (v_is_turbo && !is_mla && n_embd_head_v % 128 != 0) {
            const uint32_t padded_head_v = ((n_embd_head_v + 127) / 128) * 128;
            const uint32_t n_head_kv = n_embd_v_gqa / n_embd_head_v;
            n_embd_v_gqa_eff = n_head_kv * padded_head_v;
            if (il == 0) {
                LLAMA_LOG_INFO("%s: turbo zero-padding V head_dim %u -> %u (cache %u -> %u)\n",
                               __func__, n_embd_head_v, padded_head_v, n_embd_v_gqa, n_embd_v_gqa_eff);
            }
        }

        // [TAG_TURBOT_ANY_GEOM] a turbot layer is a [1024, kv_size, n_stream] container whatever its runs: one 1024-element
        // block of 32*S + 16 bytes holds the nr*256 values of a row (1024 = n_embd_k_gqa at 4 KV heads x 256)
        if (turbot_plan) {
            n_embd_k_gqa_eff = GGML_TURBOT_ROW_ELEMS;
            n_embd_v_gqa_eff = GGML_TURBOT_ROW_ELEMS;
        }

        // [TAG_TURBO4P] turbo4p_0 packs 8 WHT groups into one 1024-element block so that a
        // block base is 16-byte aligned. That only works if a whole row is an exact number
        // of blocks. Every other turbo type has a 128-element block and does not care.
        // Qwen3.8-27B is 4 kv heads x 256 = 1024 exactly, i.e. one block per position.
        // Fail loudly rather than silently truncating or over-reading a partial block.
        // QK_TURBO4P lives in ggml-common.h, which this layer does not include. Keep the
        // value local and named rather than pulling that header in for one constant.
        // [TAG_TURBO5P512] turbo5p512 carries a 512-element block for exactly the rows this
        // guard used to reject; llama_context has already substituted it where appropriate.
        const uint32_t turbo4p_blk = 1024;
        const uint32_t t5p512_blk  = 512;
        if (layer_type_k == GGML_TYPE_TURBO5P512_0 && n_embd_k_gqa_eff % t5p512_blk != 0) {
            throw std::runtime_error(format(
                "turbo5p512 K needs n_embd_k_gqa to be a multiple of %d, got %u on layer %d.",
                (int) t5p512_blk, n_embd_k_gqa_eff, il));
        }
        if (layer_type_v == GGML_TYPE_TURBO5P512_0 && n_embd_v_gqa_eff % t5p512_blk != 0) {
            throw std::runtime_error(format(
                "turbo5p512 V needs n_embd_v_gqa to be a multiple of %d, got %u on layer %d.",
                (int) t5p512_blk, n_embd_v_gqa_eff, il));
        }
        if ((layer_type_k == GGML_TYPE_TURBO4P_0 || layer_type_k == GGML_TYPE_TURBO5P_0) && n_embd_k_gqa_eff % turbo4p_blk != 0) {
            throw std::runtime_error(format(
                "split-plane K (turbo4p/turbo5p) needs n_embd_k_gqa to be a multiple of %d, got %u on layer %d. Use turbo4 instead.",
                (int) turbo4p_blk, n_embd_k_gqa_eff, il));
        }
        if ((layer_type_v == GGML_TYPE_TURBO4P_0 || layer_type_v == GGML_TYPE_TURBO5P_0) && n_embd_v_gqa_eff % turbo4p_blk != 0) {
            throw std::runtime_error(format(
                "split-plane V (turbo4p/turbo5p) needs n_embd_v_gqa to be a multiple of %d, got %u on layer %d. Use turbo4 instead.",
                (int) turbo4p_blk, n_embd_v_gqa_eff, il));
        }

        ggml_tensor * k = has_k ? ggml_new_tensor_3d(ctx, layer_type_k, n_embd_k_gqa_eff, kv_size, n_stream) : nullptr;
        ggml_tensor * v = has_v ? ggml_new_tensor_3d(ctx, layer_type_v, n_embd_v_gqa_eff, kv_size, n_stream) : nullptr;

        has_k && ggml_format_name(k, "cache_%sk_l%d", name_tag, il);
        has_v && ggml_format_name(v, "cache_%sv_l%d", name_tag, il);

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;

        for (uint32_t s = 0; s < n_stream; ++s) {
            k_stream.push_back(has_k ? ggml_view_2d(ctx, k, n_embd_k_gqa_eff, kv_size, k->nb[1], s*k->nb[2]) : nullptr);
            v_stream.push_back(has_v ? ggml_view_2d(ctx, v, n_embd_v_gqa_eff, kv_size, v->nb[1], s*v->nb[2]) : nullptr);
        }

        map_layer_ids[il] = layers.size();

        layers.push_back({ il, k, v, k_stream, v_stream, });

        // [TAG_TURBOT] young pool: one row per pool cell, [K part][V part]. POOL 0 keeps one never-addressed granule so
        // that every op has a pool tensor (SPEC 9.4)
        if (turbot_plan) {
            const ggml_turbot_layer & tl = turbot_plan->layers.at((int32_t) il);
            ggml_tensor * pool = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, (int64_t) tl.pool_row_bytes,
                    std::max<int64_t>(turbot_plan->pool_cells, GGML_TURBOT_POOL_MIN_ROWS));
            ggml_format_name(pool, "cache_%sturbot_young_l%d", name_tag, il);
            layers.back().pool = pool;

            // [TAG_TURBOT_ANY_STREAMS] the slice of each stream, for the pool copy of a cross-stream seq_cp (update())
            if (n_stream > 1 && turbot_pool_s > 0) {
                for (uint32_t s = 0; s < n_stream; ++s) {
                    ggml_tensor * ps = ggml_view_2d(ctx, pool, pool->ne[0], turbot_pool_s, pool->nb[1], (size_t) s*turbot_pool_s*pool->nb[1]);
                    ggml_format_name(ps, "cache_%sturbot_young_l%d_s%u", name_tag, il, s);
                    layers.back().pool_stream.push_back(ps);
                }
            }
        }

        // TurboQuant: create rotation matrix tensors (once, shared across layers)
        if (turbo_rotation == nullptr &&
            llama_type_is_turbo(type_k)) {
            turbo_rotation = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 128);
            ggml_format_name(turbo_rotation, "turbo_rotation");  // R^T
            turbo_rotation_inv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 128);
            ggml_format_name(turbo_rotation_inv, "turbo_rotation_inv");  // R

            // InnerQ: per-channel scale_inv tensor (128 floats, initialized to all 1.0)
            turbo_innerq_scale_inv = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, INNERQ_MAX_CHANNELS);
            ggml_format_name(turbo_innerq_scale_inv, "turbo_innerq_scale_inv");
        }
    }

    if (reuse) {
        LLAMA_LOG_DEBUG("%s: reusing layers:\n", __func__);

        for (uint32_t il = 0; il < n_layer; il++) {
            const int32_t il_reuse = reuse(il);

            if (il_reuse < 0) {
                LLAMA_LOG_DEBUG("%s: - layer %3d: no reuse\n", __func__, il);
                continue;
            }

            if (filter && !filter(il)) {
                LLAMA_LOG_DEBUG("%s: - layer %3d: filtered\n", __func__, il);
                continue;
            }

            GGML_ASSERT(map_layer_ids.find(il_reuse) != map_layer_ids.end());

            map_layer_ids[il] = map_layer_ids[il_reuse];

            LLAMA_LOG_DEBUG("%s: - layer %3d: reuse layer %d, is_swa = %d\n", __func__, il, il_reuse, hparams.is_swa(il));
        }
    }

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf;
        if (hparams.no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                t->buffer = buf; // set dummy buffer for KV cache so that the backend scheduler won't try to allocate it
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft); // real buffer
        }
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for kv cache");
        }

        LLAMA_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

        ggml_backend_buffer_clear(buf, 0);

        // Fill turbo rotation matrices AFTER buffer clear (clear zeroes everything)
        if (turbo_rotation != nullptr && turbo_rotation->buffer != nullptr && !model.hparams.no_alloc) {
            #include "turbo-rotation-data.h"
            // ggml is column-major; C arrays are row-major. Storing a row-major matrix
            // into ggml implicitly transposes it. ggml_mul_mat(A, x) computes A^T @ x.
            // To get R @ q: store R^T → ggml sees (R^T)^T_col = R → mul_mat gives R @ q. Wait no —
            // store R so ggml col-major reads it as R^T, then mul_mat gives (R^T)^T = R. ✓
            // Store R for Q forward rotation, R^T for V inverse rotation
            // ggml_mul_mat(A,x) computes A@x for row-major stored A (verified by test)
            ggml_backend_tensor_set(turbo_rotation, TURBO_ROTATION_R, 0, 128 * 128 * sizeof(float));
            ggml_backend_tensor_set(turbo_rotation_inv, TURBO_ROTATION_RT, 0, 128 * 128 * sizeof(float));

            // Initialize InnerQ scale_inv to all 1.0 (identity scaling)
            if (turbo_innerq_scale_inv != nullptr && turbo_innerq_scale_inv->buffer != nullptr) {
                float ones[INNERQ_MAX_CHANNELS];
                for (int i = 0; i < INNERQ_MAX_CHANNELS; i++) ones[i] = 1.0f;
                ggml_backend_tensor_set(turbo_innerq_scale_inv, ones, 0, INNERQ_MAX_CHANNELS * sizeof(float));
            }

            LLAMA_LOG_INFO("%s: TurboQuant rotation matrices initialized (128x128)\n", __func__);
        }
        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    if (turbot_plan) {
        // [TAG_TURBOT] K and V are the base caches only; the size includes the young pool (SPEC 3.6, 9.4)
        const size_t memory_size_k = size_k_bytes();
        const size_t memory_size_v = size_v_bytes();

        size_t memory_size_pool = 0;
        int    sum_s            = 0;
        int    sum_runs         = 0;   // [TAG_TURBOT_ANY_GEOM] 2*nr per layer: 8 per layer at 4 KV heads x 256, as before
        for (const auto & layer : layers) {
            memory_size_pool += ggml_nbytes(layer.pool);
        }
        for (const auto & [il, tl] : turbot_plan->layers) {
            sum_s    += tl.k.s  + tl.v.s;
            sum_runs += tl.k.nr + tl.v.nr;
        }

        const size_t memory_size_total = memory_size_k + memory_size_v + memory_size_pool;

        // [TAG_TURBOT_ANY_STREAMS] with several streams the pool is split: POOL_s rows per stream
        const std::string pool_streams = n_stream > 1 ? format(", %u streams x %u cells", n_stream, turbot_pool_s) : std::string();

        LLAMA_LOG_INFO("%s: turbot plan %s: %d layers, old bits %.3f (sum %d), young pool %u cells (%u granules)%s, cap %u, hash 0x%016llx\n",
                __func__, turbot_plan->path.c_str(), (int) turbot_plan->layers.size(),
                (double) sum_s / (double) sum_runs, sum_s,
                turbot_plan->pool_cells, turbot_plan->pool_cells / GGML_TURBOT_GRANULE, pool_streams.c_str(), turbot_plan->cap_cells,
                (unsigned long long) turbot_plan->hash);

        LLAMA_LOG_INFO("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u/%u seqs), K (turbot): %7.2f MiB, V (turbot): %7.2f MiB, young pool: %7.2f MiB\n", __func__,
                (float)memory_size_total / (1024.0f * 1024.0f), kv_size, (int) layers.size(), n_seq_max, n_stream,
                (float)memory_size_k    / (1024.0f * 1024.0f),
                (float)memory_size_v    / (1024.0f * 1024.0f),
                (float)memory_size_pool / (1024.0f * 1024.0f));

        // turbot is meant to fit in the budget of its fallback type for the same cache; a plan that does not is allowed,
        // but said. [TAG_TURBOT_ANY_GEOM] the fallback is turbo5p for rows of 1024 values (the only reference before),
        // turbo5p512 when every row is a multiple of 512, else turbo4 (llama_turbot_budget_type, one type per cache)
        bool rows_1024 = true;
        bool rows_512  = true;
        for (const auto & layer : layers) {
            const uint32_t row = (uint32_t) ggml_turbot_geom_row_elems(turbot_plan->layers.at((int32_t) layer.il).flags);
            rows_1024 = rows_1024 && row % 1024 == 0;
            rows_512  = rows_512  && row % 512  == 0;
        }
        const ggml_type ref_type = llama_turbot_budget_type(rows_1024 ? 1024 : rows_512 ? 512 : 256);

        size_t memory_size_ref = 0;
        for (const auto & layer : layers) {
            const uint32_t row = (uint32_t) ggml_turbot_geom_row_elems(turbot_plan->layers.at((int32_t) layer.il).flags);
            memory_size_ref += (size_t) kv_size*n_stream*2*ggml_row_size(ref_type, row);
        }
        if (memory_size_total > memory_size_ref) {
            LLAMA_LOG_WARN("%s: turbot: %.2f MiB exceeds the %s size of this cache (%.2f MiB)\n", __func__,
                    memory_size_total / (1024.0*1024.0), ggml_type_name(ref_type), memory_size_ref / (1024.0*1024.0));
        }
    } else {
        const size_t memory_size_k = size_k_bytes();
        const size_t memory_size_v = size_v_bytes();

        LLAMA_LOG_INFO("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u/%u seqs), K (%s): %7.2f MiB, V (%s): %7.2f MiB\n", __func__,
                (float)(memory_size_k + memory_size_v) / (1024.0f * 1024.0f), kv_size, (int) layers.size(), n_seq_max, n_stream,
                ggml_type_name(type_k), (float)memory_size_k / (1024.0f * 1024.0f),
                ggml_type_name(type_v), (float)memory_size_v / (1024.0f * 1024.0f));
    }

    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        n_embd_head_k_all = other->n_embd_head_k_all;
        n_embd_head_v_all = other->n_embd_head_v_all;

        attn_rot_k = other->attn_rot_k;
        attn_rot_v = other->attn_rot_v;
    } else {
        const char * LLAMA_ATTN_ROT_DISABLE = getenv("LLAMA_ATTN_ROT_DISABLE");
        const bool attn_rot_disable = LLAMA_ATTN_ROT_DISABLE ? atoi(LLAMA_ATTN_ROT_DISABLE) : false;
        if (attn_rot_disable) {
            LLAMA_LOG_WARN("%s: attention rotation force disabled (LLAMA_ATTN_ROT_DISABLE)\n", __func__);
        }

        attn_rot_k =
            !attn_rot_disable &&
            n_embd_head_k_all > 0 &&
            ggml_is_quantized(type_k) &&
            hparams.n_embd_head_k() % 64 == 0;

        // always create Hadamard rotation tensors for DeepSeek lightning indexers
        if ((model.arch == LLM_ARCH_DEEPSEEK32 || model.arch == LLM_ARCH_DEEPSEEK4 ||
                model.arch == LLM_ARCH_GLM_DSA || model.arch == LLM_ARCH_DOTS3NOTE) &&
                hparams.n_embd_head_k_full == hparams.indexer_head_size) {
            attn_rot_k = true;
        }

        attn_rot_v =
            !attn_rot_disable &&
            n_embd_head_v_all > 0 &&
            ggml_is_quantized(type_v) &&
            hparams.n_embd_head_v() % 64 == 0;

        // [TAG_TURBOT] every turbot quality number and the per-head widths of the plan were measured on an F16 cache,
        // where this rotation is off. WHT-128 is already turbot's incoherence rotation (SPEC 10.3).
        if (turbot_plan) {
            const char * LLAMA_TURBOT_ATTN_ROT = getenv("LLAMA_TURBOT_ATTN_ROT");
            const bool keep_attn_rot = LLAMA_TURBOT_ATTN_ROT && atoi(LLAMA_TURBOT_ATTN_ROT) == 1;
            if (!keep_attn_rot) {
                attn_rot_k = false;
                attn_rot_v = false;
                LLAMA_LOG_INFO("%s: turbot: upstream Hadamard off (quality was measured without it; LLAMA_TURBOT_ATTN_ROT=1 enables it)\n", __func__);
            }
        }
    }

    LLAMA_LOG_INFO("%s: attn_rot_k = %d, n_embd_head_k_all = %d\n", __func__, attn_rot_k, n_embd_head_k_all);
    LLAMA_LOG_INFO("%s: attn_rot_v = %d, n_embd_head_k_all = %d\n", __func__, attn_rot_v, n_embd_head_v_all);

    // pre-compute the haramard matrices and keep them in host memory
    // TODO: in the future, we can make copies in the backend buffers to avoid host -> device transfers
    if (attn_rot_k || attn_rot_v) {
        for (int64_t n = 64; n <= std::max(n_embd_head_k_all, n_embd_head_v_all); n *= 2) {
            attn_rot_hadamard[n] = std::vector<float>(n*n);

            ggml_init_params params = {
                /* .mem_size   = */ 1*ggml_tensor_overhead(),
                /* .mem_buffer = */ nullptr,
                /* .no_alloc   = */ true,
            };

            ggml_context_ptr ctx { ggml_init(params) };

            ggml_tensor * tmp = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n, n);
            tmp->data = attn_rot_hadamard[n].data();

            ggml_gen_hadamard(tmp);
        }
    }

    const char * LLAMA_KV_CACHE_DEBUG = getenv("LLAMA_KV_CACHE_DEBUG");
    debug = LLAMA_KV_CACHE_DEBUG ? atoi(LLAMA_KV_CACHE_DEBUG) : 0;
}

llama_kv_cache::~llama_kv_cache() {
    triattention_free(triattention_st);
    triattention_st = nullptr;
}

void llama_kv_cache::clear(bool data) {
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].reset();
        v_heads[s] = 0;
    }

    // [TAG_TURBOT] every slot free; stamps, counters, cuts, ref_valid and pending zeroed ([TAG_TURBOT_ANY_STREAMS] every stream)
    for (auto & tier : turbot_tier) {
        tier->clear();
    }

    // Reset TriAttention position tracking
    triattention_on_reset(triattention_st);

    if (data) {
        for (auto & [_, buf] : ctxs_bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }

        // Re-initialize turbo rotation matrices after buffer clear (clear zeroes everything)
        if (turbo_rotation != nullptr && turbo_rotation->buffer != nullptr) {
            #include "turbo-rotation-data.h"
            ggml_backend_tensor_set(turbo_rotation, TURBO_ROTATION_R, 0, 128 * 128 * sizeof(float));
            ggml_backend_tensor_set(turbo_rotation_inv, TURBO_ROTATION_RT, 0, 128 * 128 * sizeof(float));

            // Re-initialize InnerQ scale_inv to all 1.0
            if (turbo_innerq_scale_inv != nullptr && turbo_innerq_scale_inv->buffer != nullptr) {
                float ones[INNERQ_MAX_CHANNELS];
                for (int i = 0; i < INNERQ_MAX_CHANNELS; i++) ones[i] = 1.0f;
                ggml_backend_tensor_set(turbo_innerq_scale_inv, ones, 0, INNERQ_MAX_CHANNELS * sizeof(float));
            }
        }
    }
}

bool llama_kv_cache::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return true;
    }

    // TODO: fix incosistent handling of `seq_id < 0` and `seq_id == -1` in the codebase [TAG_LLAMA_SEQ_ID_NEG]
    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // [TAG_TURBOT] the same removal with the tier hooks
    if (is_turbot()) {
        return seq_rm_turbot(seq_id, p0, p1);
    }

    if (seq_id >= 0) {
        auto & cells = v_cells[seq_to_stream[seq_id]];
        auto & head  = v_heads[seq_to_stream[seq_id]];

        uint32_t new_head = cells.size();

        // [TAG_SEQ_RM_BOUNDS] The unbounded walk streamed the whole pos array and ran
        // cells.size() iterations - 262144 at our context - on every call. The server
        // issues three of these per speculative decode step (one on the target, two
        // mirrored onto the draft context), so it was ~786k branchy iterations per step
        // regardless of depth. Two cheap facts make almost all of it unnecessary:
        //   - nothing outside [used_min, used_max_p1) can match, and
        //   - if p0 is past this sequence's highest position there is nothing to remove,
        //     which is the common case right after a fully accepted draft.
        if (cells.seq_pos_max(seq_id) < p0) {
            return true;
        }

        const uint32_t i_beg = cells.used_min();
        const uint32_t i_end = cells.used_max_p1();

        for (uint32_t i = i_beg; i < i_end; ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id) && cells.seq_rm(i, seq_id)) {
                triattention_on_cell_removed(triattention_st, i);
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }

        // If we freed up a slot, set head to it so searching can start there.
        if (new_head != cells.size() && new_head < head) {
            head = new_head;
        }
    } else {
        // match any sequence
        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];
            auto & head  = v_heads[s];

            uint32_t new_head = cells.size();

            for (uint32_t i = 0; i < cells.size(); ++i) {
                if (!cells.pos_in(i, p0, p1)) {
                    continue;
                }

                cells.rm(i);
                triattention_on_cell_removed(triattention_st, i);

                if (new_head == cells.size()) {
                    new_head = i;
                }
            }

            // If we freed up a slot, set head to it so searching can start there.
            if (new_head != cells.size() && new_head < head) {
                head = new_head;
            }
        }
    }

    return true;
}

void llama_kv_cache::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id_src >= 0 && (size_t) seq_id_src < seq_to_stream.size());
    GGML_ASSERT(seq_id_dst >= 0 && (size_t) seq_id_dst < seq_to_stream.size());

    const auto s0 = seq_to_stream[seq_id_src];
    const auto s1 = seq_to_stream[seq_id_dst];

    if (s0 == s1) {
        // since both sequences are in the same stream, no data copy is necessary
        // we just have to update the cells meta data

        auto & cells = v_cells[s0];

        if (seq_id_src == seq_id_dst) {
            return;
        }

        if (p0 < 0) {
            p0 = 0;
        }

        if (p1 < 0) {
            p1 = std::numeric_limits<llama_pos>::max();
        }

        // [TAG_TURBOT] a copy into an empty sequence adopts the source's write-row counter (SPEC 9.7)
        const bool turbot_dst_was_empty = is_turbot() && cells.seq_n_cells(seq_id_dst) == 0;

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id_src)) {
                cells.seq_add(i, seq_id_dst);
            }
        }

        if (is_turbot()) {
            turbot_tier[s0]->on_seq_cp(seq_id_src, seq_id_dst, turbot_dst_was_empty);
        }

        return;
    }

    // cross-stream sequence copies require to copy the actual buffer data

    bool is_full = true;

    if (p0 > 0 && p0 + 1 < (int) get_size()) {
        is_full = false;
    }

    if (p1 > 0 && p1 + 1 < (int) get_size()) {
        is_full = false;
    }

    GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers");

    // enqueue the copy operation - the buffer copy will be performed during the next update
    sc_info.ssrc.push_back(s0);
    sc_info.sdst.push_back(s1);

    v_cells[s1].reset();
    for (uint32_t i = 0; i < v_cells[s0].size(); ++i) {
        if (v_cells[s0].seq_has(i, seq_id_src)) {
            llama_pos pos   = v_cells[s0].pos_get(i);
            llama_pos shift = v_cells[s0].get_shift(i);

            llama_kv_cell_ext ext = v_cells[s0].ext_get(i);

            if (shift != 0) {
                pos -= shift;
                assert(pos >= 0);
            }

            v_cells[s1].pos_set(i, pos);
            v_cells[s1].seq_add(i, seq_id_dst);

            if (shift != 0) {
                v_cells[s1].pos_add(i, shift);
            }

            v_cells[s1].ext_set(i, ext);
        }
    }

    v_heads[s1] = v_heads[s0];

    // [TAG_TURBOT_ANY_STREAMS] the destination tier becomes the source's (granule table, stamps, refinement bits), with
    // seq_id_src's counter and cut on seq_id_dst; update() copies the source stream's pool rows under it
    if (is_turbot()) {
        turbot_tier[s1]->copy_stream_from(*turbot_tier[s0], seq_id_src, seq_id_dst);
        turbot_tier[s1]->drop_empty_cells(v_cells[s1]);
    }

    //for (uint32_t s = 0; s < n_stream; ++s) {
    //    LLAMA_LOG_WARN("%s: seq %d: min = %d, max = %d\n", __func__, s, v_cells[s].seq_pos_min(s), v_cells[s].seq_pos_max(s));
    //}
}

void llama_kv_cache::seq_keep(llama_seq_id seq_id) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    uint32_t new_head = cells.size();

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.seq_keep(i, seq_id)) {
            if (is_turbot()) {   // [TAG_TURBOT] [TAG_TURBOT_ANY_STREAMS] the tier of this stream
                turbot_tier[seq_to_stream[seq_id]]->on_cell_emptied(i);
            }
            if (new_head == cells.size()) {
                new_head = i;
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != cells.size() && new_head < head) {
        head = new_head;
    }
}

void llama_kv_cache::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_add() is only supported for n_pos_per_embd() == 1");

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    if (shift == 0) {
        return;
    }

    uint32_t new_head = cells.size();

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over all cells.
    if (p0 == p1) {
        return;
    }

    // [TAG_TURBOT] a turbot K cache cannot be shifted (SPEC 9.3); the no-op calls above stay harmless
    if (is_turbot()) {
        GGML_ABORT("turbot: seq_add/seq_div are not supported (no K shift)");
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            if (cells.pos_add(i, shift)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    // Otherwise we just start the next search from the beginning.
    head = new_head != cells.size() ? new_head : 0;

    // Update TriAttention position tracking for the shifted range
    triattention_on_position_shift(triattention_st, shift, p0, p1);
}

void llama_kv_cache::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_div() is only supported for n_pos_per_embd() == 1");

    auto & cells = v_cells[seq_to_stream[seq_id]];

    if (d == 1) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the cache.
    if (p0 == p1) {
        return;
    }

    // [TAG_TURBOT] a turbot K cache cannot be shifted (SPEC 9.3); the no-op calls above stay harmless
    if (is_turbot()) {
        GGML_ABORT("turbot: seq_add/seq_div are not supported (no K shift)");
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            cells.pos_div(i, d);
        }
    }
}

llama_pos llama_kv_cache::seq_pos_min(llama_seq_id seq_id) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return other->seq_pos_min(seq_id);
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];

    return cells.seq_pos_min(seq_id);
}

llama_pos llama_kv_cache::seq_pos_max(llama_seq_id seq_id) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return other->seq_pos_max(seq_id);
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];

    return cells.seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [ctx, buf] : ctxs_bufs) {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf.get());

        if (hparams.no_alloc) {
            GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) == nullptr);
            ret[buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        } else {
            // GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) != nullptr); // multi_buffer does not have a defined base
            ret[buft] += ggml_backend_buffer_get_size(buf.get());
        }
    }

    return ret;
}

llama_memory_context_ptr llama_kv_cache::init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) {
    GGML_UNUSED(embd_all);

    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = n_stream == 1 ? balloc.split_simple(n_ubatch) : balloc.split_equal(n_ubatch, true, 0);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos = prepare(ubatches);
        if (sinfos.empty()) {
            break;
        }

        return std::make_unique<llama_kv_cache_context>(
                this, std::move(sinfos), std::move(ubatches));
    } while (false);

    return std::make_unique<llama_kv_cache_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_kv_cache::init_full() {
    return std::make_unique<llama_kv_cache_context>(this);
}

llama_memory_context_ptr llama_kv_cache::init_update(llama_context * lctx, bool optimize) {
    GGML_UNUSED(optimize);

    bool do_shift = get_has_shift();

    return std::make_unique<llama_kv_cache_context>(this, lctx, do_shift, std::move(sc_info));
}

llama_kv_cache::slot_info_vec_t llama_kv_cache::prepare(const std::vector<llama_ubatch> & ubatches) {
    llama_kv_cache::slot_info_vec_t res;

    struct state_t {
        slot_info sinfo; // slot info for the ubatch

        std::vector<uint32_t> v_heads_old; // old positions of the heads, before placing the ubatch

        std::vector<llama_kv_cells> v_cells; // copy of the old cells, before placing the ubatch
    };

    // remember the old state of the cells so we can restore it in the end
    std::vector<state_t> states;

    bool success = true;

    for (const auto & ubatch : ubatches) {
        // only find a suitable slot for the ubatch. don't modify the cells yet
        const auto sinfo_new = find_slot(ubatch, false);
        if (sinfo_new.empty()) {
            success = false;
            break;
        }

        // remember the position that we found
        res.push_back(sinfo_new);

        // store the old state of the cells in the recovery stack
        {
            state_t state = { sinfo_new, v_heads, {} };

            for (uint32_t s = 0; s < sinfo_new.n_stream(); ++s) {
                auto & cells = v_cells[sinfo_new.strm[s]];

                state.v_cells.push_back(cells.cp(sinfo_new.idxs[s]));
            }

            states.push_back(std::move(state));
        }

        // now emplace the ubatch
        apply_ubatch(sinfo_new, ubatch);
    }

    GGML_ASSERT(!states.empty() || !success);

    // iterate backwards and restore the cells to their original state
    for (auto it = states.rbegin(); it != states.rend(); ++it) {
        const auto & sinfo = it->sinfo;

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            auto & cells = v_cells[sinfo.strm[s]];
            auto & head  = v_heads[sinfo.strm[s]];

            cells.set(sinfo.idxs[s], it->v_cells[s]);
            head = it->v_heads_old[s];
        }
    }

    if (!success) {
        return {};
    }

    return res;
}

bool llama_kv_cache::update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return true;
    }

    bool updated = false;

    auto * sched = lctx->get_sched();

    if (!sc_info.empty()) {
        assert(n_stream > 1 && "stream copy should never happen with a single stream");

        llama_synchronize(lctx);

        const size_t n_copy = sc_info.ssrc.size();

        for (size_t i = 0; i < n_copy; ++i) {
            const auto ssrc = sc_info.ssrc[i];
            const auto sdst = sc_info.sdst[i];

            assert(ssrc < n_stream);
            assert(sdst < n_stream);

            LLAMA_LOG_DEBUG("%s: copying KV buffer: stream %d to stream %d\n", __func__, ssrc, sdst);

            assert(ssrc != sdst);

            for (uint32_t il = 0; il < layers.size(); ++il) {
                const auto & layer = layers[il];

                ggml_backend_tensor_copy(layer.k_stream[ssrc], layer.k_stream[sdst]);

                if (layer.v_stream[ssrc]) {
                    ggml_backend_tensor_copy(layer.v_stream[ssrc], layer.v_stream[sdst]);
                }

                // [TAG_TURBOT_ANY_STREAMS] the young pool rows of the source stream, under the tier seq_cp copied
                if (!layer.pool_stream.empty()) {
                    ggml_backend_tensor_copy(layer.pool_stream[ssrc], layer.pool_stream[sdst]);
                }
            }
        }
    }

    if (do_shift) {
        if (!get_can_shift()) {
            GGML_ABORT("The current KV cache / model configuration does not support K-shift");
        }

        LLAMA_LOG_DEBUG("%s: applying K-shift\n", __func__);

        // apply K-shift if needed
        if (hparams.rope_type != LLAMA_ROPE_TYPE_NONE) {
            ggml_backend_sched_reset(sched);

            auto * res = lctx->get_gf_res_reserve();

            res->reset();

            auto * gf = build_graph_shift(res, lctx);
            if (!ggml_backend_sched_alloc_graph(sched, gf)) {
                LLAMA_LOG_ERROR("%s: failed to allocate compute graph for K-shift\n", __func__);
                return updated;
            }

            res->set_inputs(nullptr);

            if (lctx->graph_compute(gf, false) != GGML_STATUS_SUCCESS) {
                LLAMA_LOG_ERROR("%s: failed to compute K-shift\n", __func__);
                return updated;
            }

            updated = true;
        }

        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];

            cells.reset_shift();
        }
    }

    return updated;
}

llama_kv_cache::slot_info llama_kv_cache::find_slot(const llama_ubatch & ubatch, bool cont) const {

    if (debug > 0) {
        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
            const auto seq_id = ubatch.seq_id_unq[s];
            const auto stream_id = seq_to_stream[seq_id];
            const auto & cells = v_cells[stream_id];
            const uint32_t head_cur = v_heads[stream_id];

            LLAMA_LOG_DEBUG("%s: stream[%d], n = %5d, used = %5d, head = %5d, size = %5d, n_swa = %5d\n",
                    __func__, stream_id, cells.used_max_p1(), cells.get_used(), head_cur, get_size(), n_swa);

            if ((debug == 2 && n_swa > 0) || debug > 2) {
                std::string ss;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    if (cells.is_empty(i)) {
                        ss += '.';
                    } else {
                        assert(cells.seq_count(i) >= 1);

                        if (cells.seq_count(i) == 1) {
                            ss += std::to_string(cells.seq_get(i));
                        } else {
                            ss += 'M';
                        }
                    }
                    if (i%256 == 255) {
                        ss += " *";
                        ss += '\n';
                    }
                }
                LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
            }

            if ((debug == 2 && n_swa > 0) || debug > 2) {
                std::string ss;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    std::string cur;
                    if (cells.is_empty(i)) {
                        cur = '.';
                    } else {
                        cur = std::to_string(cells.pos_get(i));
                    }
                    const int n = cur.size();
                    for (int j = 0; j < 5 - n; ++j) {
                        cur += ' ';
                    }
                    ss += cur;
                    if (i%256 == 255) {
                        ss += " *";
                    }
                    if (i%64 == 63) {
                        ss += '\n';
                    }
                }
                LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (cells.seq_pos_min(s) < 0) {
                    continue;
                }

                LLAMA_LOG_DEBUG("%s: stream[%d] min[%d] = %5d, max[%d] = %5d\n", __func__, stream_id, s, cells.seq_pos_min(s), s, cells.seq_pos_max(s));
            }
        }
    }

    uint32_t n_tokens = ubatch.n_tokens;
    uint32_t n_seqs   = 1;

    if (n_stream > 1) {
        GGML_ASSERT(n_tokens % ubatch.n_seqs_unq == 0);

        n_seqs   = ubatch.n_seqs_unq;
        n_tokens = n_tokens / n_seqs;
    }

    slot_info res = {
        /*.s0   =*/ LLAMA_MAX_SEQ,
        /*.s1   =*/ 0,
        /*.strm =*/ { },
        /*.idxs =*/ { },
    };

    res.resize(n_seqs);

    for (uint32_t s = 0; s < n_seqs; ++s) {
        const auto seq_id = ubatch.seq_id_unq[s];

        if (n_stream > 1) {
            GGML_ASSERT(ubatch.n_seq_id[s*n_tokens]    == 1);
            GGML_ASSERT(ubatch.seq_id  [s*n_tokens][0] == seq_id);
        }

        res.s0 = std::min<uint32_t>(res.s0, seq_to_stream[seq_id]);
        res.s1 = std::max<uint32_t>(res.s1, seq_to_stream[seq_id]);

        res.strm[s] = seq_to_stream[seq_id];
        res.idxs[s].reserve(n_tokens);

        const auto & cells = v_cells[seq_to_stream[seq_id]];

        uint32_t head_cur = v_heads[seq_to_stream[seq_id]];

        // if we have enough unused cells before the current head ->
        //   better to start searching from the beginning of the cache, hoping to fill it
        if (head_cur > cells.get_used() + 2*n_tokens) {
            head_cur = 0;
        }

        if (n_tokens > cells.size()) {
            LLAMA_LOG_ERROR("%s: n_tokens = %d > size = %u\n", __func__, n_tokens, cells.size());
            return { };
        }

        uint32_t n_tested = 0;

        // for continuous slots, we test that all tokens in the ubatch fit, starting from the current head
        // for non-continuous slots, we test the tokens one by one
        const uint32_t n_test = cont ? n_tokens : 1;

        while (true) {
            if (head_cur + n_test > cells.size()) {
                n_tested += cells.size() - head_cur;
                head_cur = 0;
                continue;
            }

            for (uint32_t i = 0; i < n_test; i++) {
                const auto idx = head_cur;

                head_cur++;
                n_tested++;

                //const llama_pos    pos    = ubatch.pos[i];
                //const llama_seq_id seq_id = ubatch.seq_id[i][0];

                // can we use this cell? either:
                //  - the cell is empty
                //  - the cell is occupied only by one sequence:
                //    - (disabled) mask causally, if the sequence is the same as the one we are inserting
                //    - mask SWA, using current max pos for that sequence in the cache
                //                always insert in the cell with minimum pos
                bool can_use = cells.is_empty(idx);

                if (!can_use && cells.seq_count(idx) == 1) {
                    const llama_pos pos_cell = cells.pos_get(idx);

                    // (disabled) causal mask
                    // note: it's better to purge any "future" tokens beforehand
                    //if (cells.seq_has(idx, seq_id)) {
                    //    can_use = pos_cell >= pos;
                    //}

                    if (!can_use) {
                        const llama_seq_id seq_id_cell = cells.seq_get(idx);

                        // SWA mask
                        if (llama_hparams::is_masked_swa(n_swa, swa_type, pos_cell, cells.seq_pos_max(seq_id_cell) + 1)) {
                            can_use = true;
                        }
                    }
                }

                if (can_use) {
                    res.idxs[s].push_back(idx);
                } else {
                    if (cont) {
                        break;
                    }
                }
            }

            if (res.idxs[s].size() == n_tokens) {
                break;
            }

            if (cont) {
                res.idxs[s].clear();
            }

            if (n_tested >= cells.size()) {
                //LLAMA_LOG_ERROR("%s: failed to find a slot for %d tokens\n", __func__, n_tokens);
                return { };
            }
        }

        // we didn't find a suitable slot - return empty result
        if (res.idxs[s].size() < n_tokens) {
            return { };
        }
    }

    assert(res.s1 >= res.s0);

    return res;
}

void llama_kv_cache::apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    // keep track of the max sequence position that we would overwrite with this ubatch
    // for non-SWA cache, this would be always empty
    llama_seq_id seq_pos_max_rm[LLAMA_MAX_SEQ];
    for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        seq_pos_max_rm[s] = -1;
    }

    assert(ubatch.n_tokens == sinfo.n_stream()*sinfo.size());

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        for (uint32_t ii = 0; ii < sinfo.size(); ++ii) {
            const uint32_t i = s*sinfo.size() + ii;

            auto & cells = v_cells[sinfo.strm[s]];

            const auto idx = sinfo.idxs[s][ii];

            if (!cells.is_empty(idx)) {
                assert(cells.seq_count(idx) == 1);

                const llama_seq_id seq_id = cells.seq_get(idx);
                const llama_pos    pos    = cells.pos_get(idx);

                seq_pos_max_rm[seq_id] = std::max(seq_pos_max_rm[seq_id], pos);

                triattention_on_cell_removed(triattention_st, idx);
                cells.rm(idx);
            }

            cells.pos_set(idx, ubatch.pos[i]);
            triattention_on_token_added(triattention_st, idx, ubatch.pos[i]);

            if (ubatch.is_pos_2d() || ubatch.token || hparams.ple_n_heads > 0) {
                llama_kv_cell_ext ext;

                if (ubatch.is_pos_2d()) {
                    ext.x = ubatch.pos[i + ubatch.n_tokens*2];
                    ext.y = ubatch.pos[i + ubatch.n_tokens];
                }

                if (ubatch.token) {
                    ext.tok = ubatch.token[i];
                } else if (hparams.ple_n_heads > 0) {
                    // embd batch (multimodal input) has no token ids, need to pad it with the correct ID for PLE layers
                    // TODO @ngxson : check if we can do the same as gemma 3n / gemma 4
                    ext.tok = hparams.ple_image_token_id != 0
                        ? (llama_token) hparams.ple_image_token_id
                        : (llama_token) hparams.ple_eos_token_id;
                }

                cells.ext_set(idx, ext);
            }

            for (int32_t s = 0; s < ubatch.n_seq_id[i]; s++) {
                cells.seq_add(idx, ubatch.seq_id[i][s]);
            }
        }
    }

    // note: we want to preserve the invariant that all positions between [pos_min, pos_max] for each sequence
    //       will be present in the cache. so we have to purge any position which is less than those we would overwrite
    //       ref: https://github.com/ggml-org/llama.cpp/pull/13746#issuecomment-2916057092
    for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        if (seq_pos_max_rm[s] == -1) {
            continue;
        }

        GGML_ASSERT(s < seq_to_stream.size());

        auto & cells = v_cells[seq_to_stream[s]];

        if (cells.seq_pos_min(s) <= seq_pos_max_rm[s]) {
            LLAMA_LOG_DEBUG("%s: purging positions [%d, %d] of sequence %d from KV cache\n",
                    __func__, cells.seq_pos_min(s), seq_pos_max_rm[s], s);

            seq_rm(s, cells.seq_pos_min(s), seq_pos_max_rm[s] + 1);
        }
    }

    // move the head at the end of the slot
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        auto & head = v_heads[sinfo.strm[s]];

        head = sinfo.idxs[s].back() + 1;
    }

    // TriAttention: set prefix length and check if pruning should trigger
    if (triattention_st != nullptr) {
        // Set prefix_length once on the first prompt batch (contains position 0, >1 token).
        // This enables prefix protection during pruning so prompt tokens are never evicted.
        if (triattention_st->prefix_length == 0 && ubatch.n_tokens > 1) {
            bool has_pos_zero = false;
            llama_pos max_batch_pos = 0;
            for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
                if (ubatch.pos[i] == 0) has_pos_zero = true;
                if (ubatch.pos[i] > max_batch_pos) max_batch_pos = ubatch.pos[i];
            }
            if (has_pos_zero) {
                triattention_st->prefix_length = max_batch_pos + 1;
            }
        }

        // Count used cells in stream 0 (primary stream). llama_kv_cells already maintains
        // this incrementally, so the old open-coded loop was an O(kv_size) walk - 262144
        // iterations at our context, run once per apply_ubatch and therefore twice per
        // ubatch, since prepare() calls apply_ubatch speculatively before the real one.
        const uint32_t n_used = v_cells[0].get_used();
        if (triattention_should_prune(triattention_st, n_used)) {
            triattention_try_prune();
        }
    }
}

bool llama_kv_cache::get_can_shift() const {
    // A turbo K cache is stored WHT-rotated. RoPE is a per-channel-pair rotation and is
    // only meaningful in the original channel basis, but the 128-wide WHT mixes every
    // channel in the group, so build_rope_shift would rotate the wrong quantity and then
    // requantize through the CPU path. The Hadamard applied around it there is the
    // upstream QuaRot matrix, which does not undo the turbo WHT. Refuse the shift rather
    // than silently corrupting the cache; callers fall back to reprocessing the prompt.
    for (const auto & layer : layers) {
        if (layer.k && llama_type_is_turbo(layer.k->type)) {
            return false;
        }
        // [TAG_TURBOT] same reason: a WHT-128 rotated base code, and no requantization path at all
        if (layer.k && ggml_turbot_is_type(layer.k->type)) {
            return false;
        }
    }

    // Step35 uses per-layer RoPE dims; K-shift assumes a single global n_rot.
    if (model.arch == LLM_ARCH_STEP35) {
        return false;
    }
    if (hparams.n_pos_per_embd() > 1) {
        return false;
    }
    return true;
}

uint32_t llama_kv_cache::get_size() const {
    const auto & cells = v_cells[seq_to_stream[0]];

    return cells.size();
}

uint32_t llama_kv_cache::get_n_stream() const {
    return n_stream;
}

bool llama_kv_cache::get_has_shift() const {
    bool result = false;

    for (uint32_t s = 0; s < n_stream; ++s) {
        result |= v_cells[s].get_has_shift();
    }

    return result;
}

ggml_type llama_kv_cache::type_k() const {
    return layers[0].k->type;
}

ggml_type llama_kv_cache::type_v() const {
    return layers[0].v->type;
}

std::vector<uint32_t> llama_kv_cache::get_layer_ids() const {
    std::vector<uint32_t> res;
    res.reserve(layers.size());

    for (const auto & layer : layers) {
        res.push_back(layer.il);
    }

    return res;
}

ggml_tensor * llama_kv_cache::get_k_storage(int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);

    return layers[ikv].k;
}

const llama_kv_cells & llama_kv_cache::get_cells(llama_seq_id seq_id) const {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    return v_cells[seq_to_stream[seq_id]];
}

uint32_t llama_kv_cache::get_n_kv(const slot_info & sinfo) const {
    uint32_t result = 0;

    // pad the n_kv value so that the graph remains constant across batches and can be reused
    // note: this also helps some backends with performance (f.ex https://github.com/ggml-org/llama.cpp/pull/16812#issuecomment-3455112220)
    const uint32_t n_pad_cur = std::max(n_pad, 256u);

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const auto & cells = v_cells[sinfo.strm[s]];

        result = std::max(std::min(cells.size(), std::max(n_pad_cur, GGML_PAD(cells.used_max_p1(), n_pad_cur))), result);
    }

    return result;
}

ggml_tensor * llama_kv_cache::get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * k = layers[ikv].k;

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_k_gqa = k->ne[0];

    // [TAG_TURBOT_ANY_GEOM] a turbot layer is a [1024, kv_size, n_stream] container of one 32*S + 16 byte block per row:
    // the view is [head dim, KV heads, n_kv, ns] with nb2 = the row bytes (nb1 is fractional and unused: the turbot FA
    // reads the row through the plan's run layout). At 4 KV heads x 256 these are the strides of before.
    if (ggml_turbot_is_type(k->type)) {
        const uint32_t ns      = sinfo.s1 - sinfo.s0 + 1;
        const size_t   row     = ggml_type_size(k->type);
        const int64_t  head    = hparams.n_embd_head_k(il);

        return ggml_view_4d(ctx, k,
                head, hparams.n_head_kv(il), n_kv, ns,
                ggml_row_size(k->type, head),
                row,
                row*kv_size,
                row*kv_size*sinfo.s0);
    }

    // For turbo-padded caches, n_embd_k_gqa may be larger than hparams value
    const bool k_is_turbo = llama_type_is_turbo(k->type);
    if (k_is_turbo) {
        assert(n_embd_k_gqa >= hparams.n_embd_k_gqa(il));
    } else {
        assert(n_embd_k_gqa == hparams.n_embd_k_gqa(il));
    }

    // Use padded head_dim for turbo types so the full padded data is returned
    const uint32_t head_k = hparams.n_embd_head_k(il);
    const uint32_t head_k_eff = (k_is_turbo && head_k % 128 != 0)
        ? ((head_k + 127) / 128) * 128 : head_k;

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    return ggml_view_4d(ctx, k,
            head_k_eff, hparams.n_head_kv(il), n_kv, ns,
            ggml_row_size(k->type, head_k_eff),
            ggml_row_size(k->type, n_embd_k_gqa),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size)*sinfo.s0);
}

ggml_tensor * llama_kv_cache::get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_v_gqa = v->ne[0];

    // [TAG_TURBOT_ANY_GEOM] the container view of get_k (turbot needs flash attention: V is never transposed)
    if (ggml_turbot_is_type(v->type)) {
        GGML_ASSERT(!v_trans);

        const uint32_t ns      = sinfo.s1 - sinfo.s0 + 1;
        const size_t   row     = ggml_type_size(v->type);
        const int64_t  head    = hparams.n_embd_head_v(il);

        return ggml_view_4d(ctx, v,
                head, hparams.n_head_kv(il), n_kv, ns,
                ggml_row_size(v->type, head),
                row,
                row*kv_size,
                row*kv_size*sinfo.s0);
    }

    // [TAG_V_CACHE_VARIABLE] — for turbo-padded V, cache may be larger
    assert(n_embd_v_gqa >= hparams.n_embd_v_gqa(il));

    // Use padded head_dim for turbo types
    const bool v_is_turbo = llama_type_is_turbo(v->type);
    const uint32_t head_v = hparams.n_embd_head_v(il);
    const uint32_t head_v_eff = (v_is_turbo && head_v % 128 != 0)
        ? ((head_v + 127) / 128) * 128 : head_v;

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    if (!v_trans) {
        // note: v->nb[1] <= v->nb[2]
        return ggml_view_4d(ctx, v,
                head_v_eff, hparams.n_head_kv(il), n_kv, ns,
                ggml_row_size(v->type, head_v_eff),                      // v->nb[1]
                ggml_row_size(v->type, n_embd_v_gqa),                    // v->nb[2]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size),            // v->nb[3]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size)*sinfo.s0);
    }

    // note: v->nb[1] > v->nb[2]
    return ggml_view_4d(ctx, v,
            n_kv, hparams.n_head_kv(il), head_v_eff, ns,
            ggml_row_size(v->type, kv_size*head_v_eff),              // v->nb[1]
            ggml_row_size(v->type, kv_size),                         // v->nb[2]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa),            // v->nb[3]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa)*sinfo.s0);
}

ggml_tensor * llama_kv_cache::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    const int32_t ikv = map_layer_ids.at(il);

    ggml_tensor * k = layers[ikv].k;

    int64_t n_embd_head = k_cur->ne[0];
    const int64_t n_head      = k_cur->ne[1];
    const int64_t n_tokens    = k_cur->ne[2];

    // Turbo zero-padding: pad each head to next multiple of 128 before merging dims.
    // k_cur shape here is (n_embd_head, n_head, n_tokens).
    // ggml_pad pads ne[0] with zeros — exactly what we need per-head.
    const bool k_is_turbo = llama_type_is_turbo(k->type);
    const bool k_needs_pad = k_is_turbo && (n_embd_head % 128 != 0);
    if (k_needs_pad) {
        const int64_t pad_amount = ((n_embd_head + 127) / 128) * 128 - n_embd_head;
        k_cur = ggml_pad(ctx, k_cur, pad_amount, 0, 0, 0);
        n_embd_head = k_cur->ne[0];  // now 128-aligned
    }

    int64_t n_embd_gqa = n_embd_head * n_head;

    // we can merge dims 0 and 1
    // TODO: add ggml helper function for this?
    GGML_ASSERT(ggml_row_size(k_cur->type, n_embd_head) == k_cur->nb[1]);

    k_cur = ggml_view_2d(ctx, k_cur, n_embd_gqa, n_tokens, k_cur->nb[2], 0);

    const int64_t n_stream = k->ne[2];

    if (n_stream > 1) {
        const int64_t kv_size = get_size();

        assert(n_embd_gqa == k->ne[0]);
        assert(kv_size    == k->ne[1]);

        // merge the buffer across all streams because the idxs are global
        k = ggml_reshape_2d(ctx, k, n_embd_gqa, kv_size*n_stream);
    }

    // store the current K values into the cache
    ggml_tensor * result = ggml_set_rows(ctx, k, k_cur, k_idxs);

    // For turbo: store WHT group size in op_params so the CUDA kernel knows.
    // With zero-padding, all groups are always full 128-element WHT groups.
    if (k_is_turbo) {
        int32_t wht_group = 128;  // always 128 with padding
        memcpy(result->op_params, &wht_group, sizeof(int32_t));
    }

    return result;
}

ggml_tensor * llama_kv_cache::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;

    int64_t n_embd_head = v_cur->ne[0];
    const int64_t n_head      = v_cur->ne[1];
    const int64_t n_tokens    = v_cur->ne[2];

    // Turbo zero-padding: pad V head_dim to next multiple of 128
    const bool v_is_turbo = llama_type_is_turbo(v->type);
    const bool v_needs_pad = v_is_turbo && (n_embd_head % 128 != 0);
    if (v_needs_pad) {
        const int64_t pad_amount = ((n_embd_head + 127) / 128) * 128 - n_embd_head;
        v_cur = ggml_pad(ctx, v_cur, pad_amount, 0, 0, 0);
        n_embd_head = v_cur->ne[0];  // now 128-aligned
    }

    int64_t n_embd_gqa = n_embd_head * n_head;

    // we can merge dims 0 and 1
    GGML_ASSERT(ggml_row_size(v_cur->type, n_embd_head) == v_cur->nb[1]);

    const int64_t n_stream = v->ne[2];

    // take this branch when FA is enabled (the V cache is not transposed)
    if (!v_trans) {
        v_cur = ggml_view_2d(ctx, v_cur, n_embd_gqa, n_tokens, v_cur->nb[2], 0);

        if (n_stream > 1) {
            const int64_t kv_size = get_size();

            assert(n_embd_gqa == v->ne[0]);
            assert(kv_size    == v->ne[1]);

            // merge the buffer across all streams because the idxs are global
            v = ggml_reshape_2d(ctx, v, n_embd_gqa, kv_size*n_stream);
        }

        ggml_tensor * result = ggml_set_rows(ctx, v, v_cur, v_idxs);
        // With zero-padding, all groups are always full 128-element WHT groups
        if (v_is_turbo) {
            int32_t wht_group = 128;  // always 128 with padding
            memcpy(result->op_params, &wht_group, sizeof(int32_t));
        }
        return result;
    }

    if (ggml_row_size(v_cur->type, n_embd_gqa) == v_cur->nb[2]) {
        // we can merge dims 0, 1 and 2
        v_cur = ggml_reshape_2d(ctx, v_cur, n_embd_gqa, n_tokens);
    } else {
        // otherwise -> make a copy to get contiguous data
        v_cur = ggml_cont_2d   (ctx, v_cur, n_embd_gqa, n_tokens);
    }

    // [TAG_V_CACHE_VARIABLE]
    if (n_embd_gqa < v->ne[0]) {
        v_cur = ggml_pad(ctx, v_cur, v->ne[0] - n_embd_gqa, 0, 0, 0);
    }

    // in this branch the v_idxs are constructed in such a way that each row is a single head element
    ggml_tensor * v_view = ggml_reshape_2d(ctx, v, 1, ggml_nelements(v));

    v_cur = ggml_reshape_2d(ctx, v_cur, 1, ggml_nelements(v_cur));

    return ggml_set_rows(ctx, v_view, v_cur, v_idxs);
}

ggml_tensor * llama_kv_cache::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);

    ggml_set_input(k_idxs);
    ggml_set_name(k_idxs, "attn_inp_k_idxs");

    return k_idxs;
}

ggml_tensor * llama_kv_cache::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * v_idxs;

    if (!v_trans) {
        v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    } else {
        v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens*hparams.n_embd_v_gqa_max());
    }

    ggml_set_input(v_idxs);
    ggml_set_name(v_idxs, "attn_inp_v_idxs");

    return v_idxs;
}

ggml_tensor * llama_kv_cache::build_input_k_rot(ggml_context * ctx) const {
    ggml_tensor * res = nullptr;

    if (attn_rot_k) {
        int nrot = 64;

        // TODO: investigate if using the smallest rotation matrix is beneficial also for K (similar as for V)
        // ref: https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4141323088
        do {
            nrot *= 2;
        } while (n_embd_head_k_all % nrot == 0);
        nrot /= 2;

        res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nrot, nrot);
        ggml_set_input(res);
        ggml_set_name(res, "attn_inp_k_rot");
    }

    return res;
}

ggml_tensor * llama_kv_cache::build_input_v_rot(ggml_context * ctx) const {
    ggml_tensor * res = nullptr;

    if (attn_rot_v) {
        int nrot = 64;
        // using smaller rotation matrices for V seems beneficial
        // ref: https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4146397570
        //do {
        //    nrot *= 2;
        //} while (hparams.n_embd_head_v() % nrot == 0);
        //nrot /= 2;

        res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nrot, nrot);
        ggml_set_input(res);
        ggml_set_name(res, "attn_inp_v_rot");
    }

    return res;
}

void llama_kv_cache::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const int64_t offs = sinfo.strm[s]*get_size();

        for (uint32_t i = 0; i < sinfo.size(); ++i) {
            data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
        }
    }
}

void llama_kv_cache::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    if (!v_trans) {
        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*get_size();

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
            }
        }
    } else {
        // note: the V cache is transposed when not using flash attention
        const int64_t kv_size = get_size();

        const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa_max();

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*kv_size*n_embd_v_gqa;

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    data[s*sinfo.size()*n_embd_v_gqa + i*n_embd_v_gqa + j] = offs + j*kv_size + sinfo.idxs[s][i];
                }
            }
        }
    }
}

void llama_kv_cache::set_input_k_shift(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    int32_t * data = (int32_t *) dst->data;

    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & cells = v_cells[s];

        for (uint32_t i = 0; i < cells.size(); ++i) {
            data[s*cells.size() + i] = cells.is_empty(i) ? 0 : cells.get_shift(i);
        }
    }
}

struct args_set_input_kq_mask {
    const llama_hparams & hparams;
    const llama_ubatch  * ubatch;

    const std::vector<llama_kv_cells> & v_cells;
    const std::vector<uint32_t>       & seq_to_stream;

    uint32_t       n_swa;
    llama_swa_type swa_type;

    int64_t n_kv;
    int64_t n_stream;
    int64_t n_tps;
};

template<typename T, bool causal, bool swa, bool is_2d, bool alibi>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
  //const auto & hparams = args.hparams;
    const auto & ubatch  = args.ubatch;

    const auto & v_cells       = args.v_cells;
    const auto & seq_to_stream = args.seq_to_stream;

    const uint32_t       n_swa    = args.n_swa;
    const llama_swa_type swa_type = args.swa_type;

    const int64_t n_kv     = args.n_kv;
    const int64_t n_stream = args.n_stream;
    const int64_t n_tps    = args.n_tps;

    const T mask_keep = llama_cast<T>(0.0f);
    const T mask_drop = llama_cast<T>(-INFINITY);

    // the min position in the batch for each sequence
    llama_pos seq_pos_min[LLAMA_MAX_SEQ];
    std::fill(seq_pos_min, seq_pos_min + LLAMA_MAX_SEQ, INT32_MAX);

    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        const llama_seq_id seq_id = ubatch->seq_id[i][0];

        seq_pos_min[seq_id] = std::min(seq_pos_min[seq_id], ubatch->pos[i]);
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        // bookkeeping of the KQ mask cells that could change for other tokens of the same sequence
        std::unordered_map<llama_seq_id, uint32_t>              seq_srct;
        std::unordered_map<llama_seq_id, std::vector<uint32_t>> seq_idxs;

        for (uint32_t ii = 0; ii < n_tps; ++ii) {
            const uint32_t i = s*n_tps + ii;

            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            const auto & cells = v_cells.at(seq_to_stream[seq_id]);

                  llama_pos p0 = -1;
            const llama_pos p1 = ubatch->pos[i];

            // for M-RoPE
            const llama_pos p1_x = is_2d ? ubatch->pos[i + ubatch->n_tokens*2] : 0;
            const llama_pos p1_y = is_2d ? ubatch->pos[i + ubatch->n_tokens]   : 0;

            const uint64_t idst = n_kv*i;

            // for tokens of the same sequence, the mask is mostly the same, so we can reuse it
            // the only cells that could change are the ones that are with similar positions as the
            //   ones in the batch (i.e. due to causal masking, SWA, etc.)
            // keep track of those cells and shortcut the loop to save time
            // note: this optimization is not compatible with Alibi position encoding
            // ref:  https://github.com/ggml-org/llama.cpp/pull/18842
            bool prev = false;

            auto & idxs = seq_idxs[seq_id];

            if (!alibi) {
                if (seq_srct.find(seq_id) != seq_srct.end()) {
                    const uint32_t srct = seq_srct[seq_id];

                    const uint64_t idst_prev = n_kv*srct;

                    std::copy(data + idst_prev, data + idst_prev + n_kv, data + idst);

                    prev = true;
                } else {
                    idxs.clear();
                    idxs.reserve(ubatch->n_tokens + n_swa + 32);

                    seq_srct[seq_id] = i;
                }
            }

            for (uint32_t jj = 0; jj < n_kv; ++jj) {
                uint32_t j = jj;

                // we have an exiting mask for this sequence -> update just seq_idxs
                if (!alibi) {
                    if (prev) {
                        if (jj >= idxs.size()) {
                            break;
                        }

                        j = idxs[jj];
                    }
                }

                if (cells.is_empty(j)) {
                    goto skip;
                }

                // mask the token if not the same sequence
                if (!cells.seq_has(j, seq_id)) {
                    goto skip;
                }

                p0 = cells.pos_get(j);

                if (!alibi) {
                    if (!prev) {
                        // record all cells for which: p0 >= seq_pos_min[seq_id] - n_swa - 32
                        if (p0 + (int32_t) (n_swa + 32) >= seq_pos_min[seq_id]) {
                            idxs.push_back(j);
                        }
                    }
                }

                if (causal) {
                    // mask future tokens
                    if (p0 > p1) {
                        goto skip;
                    }

                    // M-RoPE causal mask
                    if (is_2d) {
                        if (p0 == p1) {
                            const auto & p0_ext = cells.ext_get(j);

                            if (p0_ext.is_2d_gt(p1_x, p1_y)) {
                                goto skip;
                            }
                        }
                    }
                }

                // apply SWA if any
                if (swa) {
                    // see llama_non_causal_type
                    const bool in_span = !causal && args.hparams.non_causal_type == LLAMA_NON_CAUSAL_TYPE_SWA_FULL && p0 >= seq_pos_min[seq_id];
                    if (!in_span && llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
                        goto skip;
                    }
                }

                if (alibi) {
                    data[idst + j] = llama_cast<T>(static_cast<float>(-std::abs(p0 - p1)));
                } else {
                    data[idst + j] = mask_keep;
                }

                continue;
skip:
                data[idst + j] = mask_drop;
            }
        }
    }
}

template<typename T, bool causal, bool swa, bool is_2d>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool alibi = args.hparams.use_alibi;
    if (alibi) {
        set_input_kq_mask_impl<T, causal, swa, is_2d, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, swa, is_2d, false>(args, data);
    }
}

template<typename T, bool causal, bool swa>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool is_2d = args.ubatch->is_pos_2d();
    if (is_2d) {
        set_input_kq_mask_impl<T, causal, swa, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, swa, false>(args, data);
    }
}

template<typename T, bool causal>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool swa = args.swa_type != LLAMA_SWA_TYPE_NONE;
    if (swa) {
        set_input_kq_mask_impl<T, causal, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, false>(args, data);
    }
}

template<typename T>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data, bool causal_attn) {
    if (causal_attn) {
        set_input_kq_mask_impl<T, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, false>(args, data);
    }
}

// [TAG_FA_POS_MASK] one position per cell of the current n_kv window: the cell's position when it belongs to
// the batch's sequence, -1 when it is empty or belongs to another sequence. With causal attention the kernel
// derives the mask from this and the query positions instead of reading an [n_kv, n_tokens] F16 mask.
void llama_kv_cache::set_input_kv_pos(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    int32_t * data = (int32_t *) dst->data;
    const int64_t n_kv = dst->ne[0];
    const llama_seq_id seq_id = ubatch->seq_id[0][0];
    const auto & cells = v_cells.at(seq_to_stream[seq_id]);
    const int64_t n_cells = cells.size();
    // [TAG_MASK_PROBE] same probe as the explicit mask: this walk runs once per single-sequence ubatch
    static const bool pos_probe = [] {
        const char * e = getenv("TURBO_MASK_PROBE");
        return e != nullptr && atoi(e) != 0;
    }();
    const int64_t t_start = pos_probe ? ggml_time_us() : 0;
    for (int64_t j = 0; j < n_kv; ++j) {
        data[j] = (j < n_cells && !cells.is_empty(j) && cells.seq_has(j, seq_id)) ? cells.pos_get(j) : -1;
    }
    if (pos_probe) {
        static int64_t acc_us = 0, acc_cells = 0, calls = 0, max_kv = 0;
        acc_us    += ggml_time_us() - t_start;
        acc_cells += n_kv;
        max_kv     = std::max<int64_t>(max_kv, n_kv);
        if (++calls % 512 == 0) {
            fprintf(stderr, "turbo-probe: kv-pos fill %lld calls  avg %.3f ms/call  %.2f ns/cell  max n_kv %lld\n",
                    (long long) calls, acc_us / 1000.0 / calls, acc_cells ? 1000.0 * acc_us / acc_cells : 0.0, (long long) max_kv);
        }
    }
}

void llama_kv_cache::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    const uint32_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const int64_t n_kv     = dst->ne[0];
    const int64_t n_stream = dst->ne[3]; // num streams in the current ubatch

    GGML_ASSERT(n_tokens%n_stream == 0);

    // n_tps == n_tokens_per_stream
    const int64_t n_tps = n_tokens/n_stream;

    // see llama_non_causal_type
    // only the SWA cache (or the SWA layers of a single cache) become non-causal
    if (!causal_attn && hparams.non_causal_type == LLAMA_NON_CAUSAL_TYPE_SWA_ONLY) {
        causal_attn = swa_type == LLAMA_SWA_TYPE_NONE;
    }

    // [TAG_MASK_PROBE] TURBO_MASK_PROBE=1 times the host mask fill, which runs on the inference thread
    // before every graph compute. With --kv-unified and several agents the explicit mask walks all n_kv
    // cells once per sequence and uploads [n_kv x n_tokens] F16 - an estimate of 2-4 ms/step at 4x64K
    // that no probe had ever measured (it sits inside tgt_decode, not in the host phases).
    static const bool mask_probe = [] {
        const char * e = getenv("TURBO_MASK_PROBE");
        return e != nullptr && atoi(e) != 0;
    }();
    const int64_t t_start = mask_probe ? ggml_time_us() : 0;

    const args_set_input_kq_mask args = {
        /*.hparams          =*/ hparams,
        /*.ubatch           =*/ ubatch,
        /*.v_cells          =*/ v_cells,
        /*.seq_to_stream    =*/ seq_to_stream,
        /*.n_swa            =*/ n_swa,
        /*.swa_type         =*/ swa_type,
        /*.n_kv             =*/ n_kv,
        /*.n_stream         =*/ n_stream,
        /*.n_tps            =*/ n_tps,
    };

    if (dst->type == GGML_TYPE_F16) {
        set_input_kq_mask_impl<ggml_fp16_t>(args, (ggml_fp16_t *) dst->data, causal_attn);
    } else {
        set_input_kq_mask_impl<float>(args, (float *) dst->data, causal_attn);
    }

    if (mask_probe) {
        // one thread fills masks (the inference thread), so plain statics are enough
        static int64_t acc_us = 0, acc_cells = 0, calls = 0, max_kv = 0, max_seqs = 0;
        acc_us    += ggml_time_us() - t_start;
        acc_cells += n_kv * (int64_t) ubatch->n_seqs_unq;
        max_kv     = std::max<int64_t>(max_kv, n_kv);
        max_seqs   = std::max<int64_t>(max_seqs, ubatch->n_seqs_unq);
        if (++calls % 512 == 0) {
            fprintf(stderr, "turbo-probe: kq-mask fill %lld calls  avg %.3f ms/call  %.2f ns/cell-walk  max n_kv %lld  max seqs %lld\n",
                    (long long) calls, acc_us / 1000.0 / calls, acc_cells ? 1000.0 * acc_us / acc_cells : 0.0,
                    (long long) max_kv, (long long) max_seqs);
        }
    }
}

void llama_kv_cache::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(n_stream == 1 && "TODO: support multiple streams");
    const auto & cells = v_cells[0];

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

    int32_t * data = (int32_t *) dst->data;

    const int32_t n_kv = dst->ne[0];

    for (int h = 0; h < 1; ++h) {
        for (int i = 0; i < n_tokens; ++i) {
            for (int j = 0; j < n_kv; ++j) {
                // the position when the cells is empty is irrelevant - it will be masked out later in the attention
                const llama_pos p0 = cells.is_empty(j) ? -1 : cells.pos_get(j);

                data[h*(n_kv*n_tokens) + i*n_kv + j] = llama_relative_position_bucket(p0, ubatch->pos[i], hparams.n_rel_attn_bkts, false);
            }
        }
    }
}

void llama_kv_cache::set_input_k_rot(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const auto n_rot = dst->ne[0];
    GGML_ASSERT(attn_rot_hadamard.count(dst->ne[0]));

    memcpy(dst->data, attn_rot_hadamard.at(n_rot).data(), ggml_nbytes(dst));
}

void llama_kv_cache::set_input_v_rot(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const auto n_rot = dst->ne[0];
    GGML_ASSERT(attn_rot_hadamard.count(dst->ne[0]));

    memcpy(dst->data, attn_rot_hadamard.at(n_rot).data(), ggml_nbytes(dst));
}

bool llama_kv_cache::has_cell_ext() const {
    // M-RoPE needs the 2D position, the PLE n-gram hash needs the token id
    return hparams.n_pos_per_embd() > 1 || hparams.ple_n_heads > 0;
}

void llama_kv_cache::get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    res.clear();
    res.resize(n_tokens*n, LLAMA_TOKEN_NULL);

    if (n == 0) {
        return;
    }

    // note: apply_ubatch() has already stored the current ubatch, so the cells cover the tokens
    //       of this very ubatch as well, which is what we want
    // the nearest cell at or before a position also resolves M-RoPE gaps, where multiple tokens
    // share the same temporal pos

    // an embd (multimodal) ubatch can repeat one position for a whole image, so positions
    // do not encode the token order; resolve its predecessors by ubatch order instead
    std::vector<uint32_t> ord; // index among the ubatch tokens of the same seq
    std::unordered_map<llama_seq_id, std::vector<uint32_t>> seq_idx;

    if (!ubatch.token) {
        ord.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; ++i) {
            auto & v = seq_idx[ubatch.seq_id[i][0]];
            ord[i] = v.size();
            v.push_back(i);
        }
    }

    for (uint32_t i = 0; i < n_tokens; ++i) {
        // TODO: a token that belongs to more than one sequence has an ambiguous history.
        //       the n-gram architectures have to reject such batches
        const llama_seq_id seq_id = ubatch.seq_id[i][0];

        for (uint32_t j = 0; j < n; ++j) {
            const llama_pos d = (llama_pos) (n - j);

            llama_pos p;
            if (!ubatch.token) {
                const auto & v = seq_idx[seq_id];
                const int64_t k = (int64_t) ord[i] - d;
                // k >= 0: an earlier token of this very ubatch; k < 0: before the chunk
                p = k >= 0 ? ubatch.pos[v[k]] : ubatch.pos[v[0]] + (llama_pos) k;
            } else {
                p = ubatch.pos[i] - d;
            }

            if (p < 0) {
                continue;
            }

            res[i*n + j] = v_cells[seq_to_stream[seq_id]].seq_pos_tok_le(seq_id, p);
        }
    }
}

size_t llama_kv_cache::total_size() const {
    size_t size = 0;

    for (const auto & [_, buf] : ctxs_bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

size_t llama_kv_cache::size_k_bytes() const {
    size_t size_k_bytes = 0;

    for (const auto & layer : layers) {
        size_k_bytes += ggml_nbytes(layer.k);
    }

    return size_k_bytes;
}

size_t llama_kv_cache::size_v_bytes() const {
    size_t size_v_bytes = 0;

    for (const auto & layer : layers) {
        size_v_bytes += layer.v ? ggml_nbytes(layer.v) : 0;
    }

    return size_v_bytes;
}

ggml_tensor * llama_kv_cache::build_rope_shift(
        const llama_cparams & cparams,
               ggml_context * ctx,
                ggml_tensor * cur,
                ggml_tensor * shift,
                ggml_tensor * rot,
                ggml_tensor * factors,
                      float   freq_base,
                      float   freq_scale,
                   uint32_t   il) const {
    const auto & n_ctx_orig = cparams.n_ctx_orig_yarn;

    const auto & yarn_ext_factor  = cparams.yarn_ext_factor;
    const auto & yarn_beta_fast   = cparams.yarn_beta_fast;
    const auto & yarn_beta_slow   = cparams.yarn_beta_slow;
    const auto & yarn_attn_factor = cparams.yarn_attn_factor;

    const auto & n_rot     = hparams.n_rot(il);
    const auto & rope_type = hparams.rope_type == LLAMA_ROPE_TYPE_MROPE || hparams.rope_type == LLAMA_ROPE_TYPE_IMROPE
                                // @ngxson : this is a workaround
                                // for M-RoPE, we want to rotate the whole vector when doing KV shift
                                // a normal RoPE should work, we just need to use the correct ordering
                                // ref: https://github.com/ggml-org/llama.cpp/pull/13870
                                ? LLAMA_ROPE_TYPE_NEOX
                                : hparams.rope_type;
    ggml_tensor * tmp;

    if (ggml_is_quantized(cur->type)) {
        // dequantize to f32 -> RoPE -> quantize back
        tmp = ggml_cast(ctx, cur, GGML_TYPE_F32);

        // rotate back
        tmp = llama_mul_mat_hadamard(ctx, tmp, rot);

        tmp = ggml_rope_ext(ctx, tmp,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);

        // rotate fwd
        tmp = llama_mul_mat_hadamard(ctx, tmp, rot);

        tmp = ggml_cpy(ctx, tmp, cur);
    } else {
        // we rotate only the first n_rot dimensions
        tmp = ggml_rope_ext_inplace(ctx, cur,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);
    }

    return tmp;
}

class llm_graph_input_k_shift : public llm_graph_input_i {
public:
    llm_graph_input_k_shift(const llama_kv_cache * kv_self) : kv_self(kv_self) {}
    virtual ~llm_graph_input_k_shift() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * k_shift; // I32 [kv_size*n_stream]

    // note: assumes k_rot^2 == I
    ggml_tensor * k_rot = nullptr;

    const llama_kv_cache * kv_self;
};

void llm_graph_input_k_shift::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (k_shift) {
        kv_self->set_input_k_shift(k_shift);
    }

    if (k_rot && k_rot->buffer) {
        kv_self->set_input_k_rot(k_rot);
    }
}

ggml_cgraph * llama_kv_cache::build_graph_shift(llm_graph_result * res, llama_context * lctx) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    GGML_ASSERT(!other);

    auto * ctx = res->get_ctx();
    auto * gf  = res->get_gf();

    auto inp = std::make_unique<llm_graph_input_k_shift>(this);

    inp->k_shift = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) get_size()*n_stream);
    ggml_set_input(inp->k_shift);

    inp->k_rot = build_input_k_rot(ctx);

    const auto & cparams = lctx->get_cparams();

    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        if (!hparams.has_rope(il)) {
            continue;
        }

        const int64_t n_head_kv    = hparams.n_head_kv(il);
        const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        const auto n_rot         = hparams.n_rot(il);
        const auto n_embd_head_k = hparams.n_embd_head_k(il);
        const auto n_embd_nope   = hparams.n_lora_kv > 0 ? n_embd_head_k - n_rot : 0;

        const float freq_base_l  = model.get_rope_freq_base (cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

        ggml_tensor * k =
            ggml_view_3d(ctx, layer.k,
                n_rot, n_head_kv, get_size()*n_stream,
                ggml_row_size(layer.k->type, n_embd_head_k),
                ggml_row_size(layer.k->type, n_embd_k_gqa),
                ggml_row_size(layer.k->type, n_embd_nope));

        ggml_tensor * cur = build_rope_shift(cparams, ctx, k, inp->k_shift, inp->k_rot, rope_factors, freq_base_l, freq_scale_l, il);

        ggml_build_forward_expand(gf, cur);
    }

    res->add_input(std::move(inp));

    return gf;
}

void llama_kv_cache::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_UNUSED(flags);

    io.write(&n_stream, sizeof(n_stream));

    for (uint32_t s = 0; s < n_stream; ++s) {
        cell_ranges_t cr { s, {} };

        uint32_t cell_count = 0;

        const auto & cells = v_cells[s];

        // Count the number of cells with the specified seq_id
        // Find all the ranges of cells with this seq id (or all, when -1)
        uint32_t cell_range_begin = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            bool add_cell = true;

            add_cell = add_cell && !cells.is_empty(i);
            add_cell = add_cell && (seq_id == -1 || cells.seq_has(i, seq_id));

            // check the cell is not SWA-masked
            if (add_cell && seq_id != -1) {
                const bool is_masked = llama_hparams::is_masked_swa(n_swa, swa_type, cells.pos_get(i), cells.seq_pos_max(seq_id));

                add_cell = !is_masked;
            }

            if (add_cell) {
                ++cell_count;
                if (cell_range_begin == cells.size()) {
                    cell_range_begin = i;
                }
            } else {
                if (cell_range_begin != cells.size()) {
                    cr.data.emplace_back(cell_range_begin, i);
                    cell_range_begin = cells.size();
                }
            }
        }

        if (cell_range_begin != cells.size()) {
            cr.data.emplace_back(cell_range_begin, cells.size());
        }

        // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
        uint32_t cell_count_check = 0;
        for (const auto & range : cr.data) {
            cell_count_check += range.second - range.first;
        }
        GGML_ASSERT(cell_count == cell_count_check);

        io.write(&cell_count, sizeof(cell_count));

        // skip empty streams
        if (cell_count == 0) {
            continue;
        }

        state_write_meta(io, cr, seq_id);
        state_write_data(io, cr);

        // [TAG_TURBOT] stamps, counters, young flags and the young pool bytes of the written cells (SPEC 9.10)
        if (is_turbot()) {
            state_write_turbot(io, cr, cell_count, seq_id);
        }
    }
}

void llama_kv_cache::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    state_read_sinfo(io, seq_id, flags, nullptr, nullptr);
}

void llama_kv_cache::state_read_sinfo(
        llama_io_read_i & io,
           llama_seq_id   seq_id,
  llama_state_seq_flags   flags,
      slot_info_vec_t *   sinfos_out,
const slot_info_vec_t *   sinfos_in) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_UNUSED(flags);

    // TODO: fix incosistent handling of `seq_id < 0` and `seq_id == -1` in the codebase [TAG_LLAMA_SEQ_ID_NEG]
    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    if (sinfos_out) {
        sinfos_out->assign(n_stream, slot_info{});
    }

    if (sinfos_in && sinfos_in->size() != n_stream) {
        throw std::runtime_error("failed to restore kv cache: mirrored slot layout has the wrong stream count");
    }

    uint32_t n_stream_cur;
    io.read(&n_stream_cur, sizeof(n_stream_cur));
    if (n_stream_cur != n_stream) {
        throw std::runtime_error("n_stream mismatch");
    }

    // a whole-context restore replaces every stream, so the cache is emptied once here
    // clear() resets all streams at once, so doing it per stream below would keep only the last one
    if (seq_id == -1) {
        clear(true);
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        uint32_t cell_count;
        io.read(&cell_count, sizeof(cell_count));

        if (cell_count == 0) {
            // a mirrored cache must be empty here as well, or the two no longer agree cell for cell
            if (sinfos_in && !(*sinfos_in)[s].empty()) {
                throw std::runtime_error("failed to restore kv cache: mirrored cache holds cells this one does not");
            }
            continue;
        }

        const uint32_t strm = seq_id == -1 ? s : seq_to_stream[seq_id];

        slot_info sinfo;

        bool res = true;
        res = res && state_read_meta(io, strm, cell_count, sinfo, seq_id, sinfos_in ? &(*sinfos_in)[s] : nullptr);

        try {
            res = res && state_read_data(io, strm, cell_count, sinfo);

            // [TAG_TURBOT] the v2 section follows the data of this stream (SPEC 9.10)
            if (is_turbot()) {
                res = res && state_read_turbot(io, strm, cell_count, sinfo, seq_id);
            }
        } catch (...) {
            res = false;
        }

        if (!res) {
            if (seq_id == -1) {
                clear(true);
            } else {
                seq_rm(seq_id, -1, -1);
            }
            throw std::runtime_error("failed to restore kv cache");
        }

        if (sinfos_out) {
            (*sinfos_out)[s] = sinfo;
        }
    }
}

void llama_kv_cache::state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id) const {
    const auto & cells = v_cells[cr.strm];

    for (const auto & range : cr.data) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            std::vector<llama_seq_id> seq_ids;

            for (llama_seq_id cur = 0; cur < (int) n_seq_max; ++cur) {
                if (cur == seq_id || seq_id == -1) {
                    if (cells.seq_has(i, cur)) {
                        seq_ids.push_back(cur);
                    }
                }
            }

            const llama_pos pos     = cells.pos_get(i);
            const uint32_t n_seq_id = seq_ids.size();

            io.write(&pos,      sizeof(pos));
            io.write(&n_seq_id, sizeof(n_seq_id));

            if (has_cell_ext()) {
                const llama_kv_cell_ext ext = cells.ext_get(i);
                io.write(&ext, sizeof(ext));
            }

            for (const auto & seq_id : seq_ids) {
                io.write(&seq_id, sizeof(seq_id));
            }
        }
    }
}

void llama_kv_cache::state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const {
    const auto & cells = v_cells[cr.strm];

    const uint32_t v_trans = this->v_trans ? 1 : 0;
    const uint32_t n_layer = layers.size();

    io.write(&v_trans, sizeof(v_trans));
    io.write(&n_layer, sizeof(n_layer));

    // Iterate and write all the keys first, each row is a cell
    // Get whole range at a time
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        auto * k = layer.k_stream[cr.strm];

        // Use actual tensor width (may be padded for turbo types: e.g. 576→640)
        const uint32_t n_embd_k_gqa = (uint32_t) k->ne[0];

        // Write key type
        const int32_t k_type_i = (int32_t) k->type;
        io.write(&k_type_i, sizeof(k_type_i));

        // Write row size of key
        const uint64_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);
        io.write(&k_size_row, sizeof(k_size_row));

        // Read each range of cells of k_size length and write out
        for (const auto & range : cr.data) {
            const size_t range_size = range.second - range.first;
            const size_t buf_size = range_size * k_size_row;
            io.write_tensor(k, range.first * k_size_row, buf_size);
        }
    }

    if (!v_trans) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            auto * v = layer.v_stream[cr.strm];
            if (!v) {
                continue;
            }

            // Use actual tensor width (may be padded for turbo types)
            const uint32_t n_embd_v_gqa = (uint32_t) v->ne[0];

            // Write value type
            const int32_t v_type_i = (int32_t) v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write row size of value
            const uint64_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
            io.write(&v_size_row, sizeof(v_size_row));

            // Read each range of cells of v_size length and write out
            for (const auto & range : cr.data) {
                const size_t range_size = range.second - range.first;
                const size_t buf_size = range_size * v_size_row;
                io.write_tensor(v, range.first * v_size_row, buf_size);
            }
        }
    } else {
        // When v is transposed, we also need the element size and get the element ranges from each row
        const uint32_t kv_size = cells.size();

        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[cr.strm];
            if (!v) {
                continue;
            }

            // Write value type
            const int32_t v_type_i = (int32_t) v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write element size
            const uint32_t v_size_el = ggml_type_size(v->type);
            io.write(&v_size_el, sizeof(v_size_el));

            // Write GQA embedding size
            io.write(&n_embd_v_gqa, sizeof(n_embd_v_gqa));

            // For each row, we get the element values of each cell
            for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                // Read each range of cells of v_size_el length and write out
                for (const auto & range : cr.data) {
                    const size_t range_size = range.second - range.first;
                    const size_t src_offset = (range.first + j * kv_size) * v_size_el;
                    const size_t buf_size = range_size * v_size_el;
                    io.write_tensor(v, src_offset, buf_size);
                }
            }
        }
    }
}

bool llama_kv_cache::state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, slot_info & sinfo, llama_seq_id dest_seq_id, const slot_info * sinfo_in) {
    auto & cells = v_cells[strm];
    auto & head  = v_heads[strm];

    if (dest_seq_id != -1) {
        // single sequence
        seq_rm(dest_seq_id, -1, -1);

        llama_batch_allocr balloc(hparams.n_pos_per_embd());

        llama_ubatch ubatch = balloc.ubatch_reserve(cell_count, 1);

        ubatch.seq_id_unq[0] = dest_seq_id;

        // the ext as it was saved, to put back after apply_ubatch()
        std::vector<llama_kv_cell_ext> exts;
        if (has_cell_ext()) {
            exts.resize(cell_count);
        }

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id != 1) {
                LLAMA_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
                return false;
            }

            if (has_cell_ext()) {
                llama_kv_cell_ext ext;
                io.read(&ext, sizeof(ext));

                if (hparams.n_pos_per_embd() > 1) {
                    ubatch.pos[i + ubatch.n_tokens]   = ext.y;
                    ubatch.pos[i + ubatch.n_tokens*2] = ext.x;
                }

                // apply_ubatch() below restores ext.tok from the ubatch tokens
                ubatch.token[i] = ext.tok;

                exts[i] = ext;
            }

            // read the sequence id, but directly discard it - we will use dest_seq_id instead
            {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));
            }

            ubatch.pos[i]      = pos;
            ubatch.n_seq_id[i] = n_seq_id;
            ubatch.seq_id[i]   = &dest_seq_id;
        }

        if (sinfo_in) {
            // this cache mirrors another one, so it takes that cache's layout instead of searching for its own cells
            if (sinfo_in->empty() || sinfo_in->n_stream() != 1 || sinfo_in->idxs[0].size() != cell_count) {
                LLAMA_LOG_ERROR("%s: mirrored slot layout holds %d cells, this cache restores %d\n", __func__,
                        sinfo_in->empty() ? 0 : (int) sinfo_in->idxs[0].size(), cell_count);
                return false;
            }

            sinfo = *sinfo_in;

            // the layout is cell indices, so it means the same in both caches only while their streams line up
            sinfo.s0 = strm;
            sinfo.s1 = strm;
            sinfo.strm[0] = strm;

            // seq_rm above freed exactly the cells this sequence held
            // anything else in the way is a cache that had already drifted, which this restore must not hide
            for (uint32_t i = 0; i < cell_count; ++i) {
                const uint32_t idx = sinfo.idxs[0][i];

                if (idx >= cells.size() || !cells.is_empty(idx)) {
                    LLAMA_LOG_ERROR("%s: cell %u of the mirrored slot layout is not free\n", __func__, idx);
                    return false;
                }
            }
        } else {
            sinfo = find_slot(ubatch, false);
            if (sinfo.empty()) {
                LLAMA_LOG_ERROR("%s: failed to find %d available cells in kv cache\n", __func__,  cell_count);
                return false;
            }
        }

        // note: apply_ubatch() rebuilds llama_kv_cell_ext from the ubatch
        //       only ext.tok and the M-RoPE 2D position round-trip through it
        //       see: https://github.com/ggml-org/llama.cpp/pull/16825#issuecomment-3460868350
        apply_ubatch(sinfo, ubatch);

        // apply_ubatch() takes the 2D position from the ubatch, and that ubatch is built with this
        // cache's own n_pos_per_embd. a cache that does not use M-RoPE itself but mirrors one that
        // does (the qwen4exp QSA indexer) would drop x and y. put the saved ext back instead, which
        // is what the whole-context path below already does.
        for (uint32_t i = 0; i < (uint32_t) exts.size(); ++i) {
            cells.ext_set(sinfo.idxs[0][i], exts[i]);
        }

        LLAMA_LOG_DEBUG("%s: cell_count = %d, dest_seq_id = %d\n", __func__, cell_count, dest_seq_id);

        // DEBUG CHECK: verify that all cells were allocated and have correct seq_id and pos values
        GGML_ASSERT(sinfo.n_stream() == 1);
        GGML_ASSERT(sinfo.idxs[0].size() == cell_count);
        for (uint32_t i = 0; i < cell_count; ++i) {
            const uint32_t idx = sinfo.idxs[0][i];
            GGML_ASSERT(cells.pos_get(idx) == ubatch.pos[i]);
            GGML_ASSERT(cells.seq_has(idx, dest_seq_id));
        }
    } else {
        // whole KV cache restore

        if (cell_count > cells.size()) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        // the cells go in from 0, so a mirrored cache lands on the same ones as long as it restores the same count. the layout itself carries no more information here
        if (sinfo_in && (sinfo_in->empty() || sinfo_in->n_stream() != 1 || sinfo_in->idxs[0].size() != cell_count)) {
            LLAMA_LOG_ERROR("%s: mirrored slot layout holds %d cells, this cache restores %d\n", __func__,
                    sinfo_in->empty() ? 0 : (int) sinfo_in->idxs[0].size(), cell_count);
            return false;
        }

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t  n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            cells.pos_set(i, pos);

            if (has_cell_ext()) {
                llama_kv_cell_ext ext;
                io.read(&ext, sizeof(ext));
                cells.ext_set(i, ext);
            }

            for (uint32_t j = 0; j < n_seq_id; ++j) {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));

                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, seq_id, n_seq_max);
                    return false;
                }

                cells.seq_add(i, seq_id);
            }
        }

        // Create contiguous slot_info for whole cache restore
        sinfo.s0 = strm;
        sinfo.s1 = strm;
        sinfo.resize(1);
        sinfo.strm[0] = strm;
        sinfo.idxs[0].resize(cell_count);
        for (uint32_t i = 0; i < cell_count; ++i) {
            sinfo.idxs[0][i] = i;
        }

        head = 0;
    }

    return true;
}

bool llama_kv_cache::state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo) {
    auto & cells = v_cells[strm];

    // batch the scatter reads per contiguous run of destination indices
    // from inclusive, to exclusive - same convention as cell_ranges_t
    // contiguous cells yield a single run covering the whole block
    struct cell_run { uint32_t from; uint32_t to; };
    std::vector<cell_run> runs;
    if (cell_count > 0) {
        const auto & idxs = sinfo.idxs[0];
        uint32_t i0 = 0;
        while (i0 < cell_count) {
            uint32_t i1 = i0 + 1;
            while (i1 < cell_count && idxs[i1] == idxs[i1 - 1] + 1) {
                ++i1;
            }
            runs.push_back({idxs[i0], idxs[i1 - 1] + 1});
            i0 = i1;
        }
    }

    uint32_t v_trans;
    uint32_t n_layer;

    io.read(&v_trans, sizeof(v_trans));
    io.read(&n_layer, sizeof(n_layer));

    if (n_layer != layers.size()) {
        LLAMA_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, n_layer, (uint32_t) layers.size());
        return false;
    }

    if (cell_count > cells.size()) {
        LLAMA_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, cell_count, cells.size());
        return false;
    }

    if (this->v_trans != (bool) v_trans) {
        LLAMA_LOG_ERROR("%s: incompatible V transposition\n", __func__);
        return false;
    }

    // For each layer, read the keys for each cell, one row is one cell, read as one contiguous block
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        auto * k = layer.k_stream[strm];

        // Use actual tensor width (may be padded for turbo types)
        const uint32_t n_embd_k_gqa = (uint32_t) k->ne[0];

        // Read type of key
        int32_t k_type_i_ref;
        io.read(&k_type_i_ref, sizeof(k_type_i_ref));
        const int32_t k_type_i = (int32_t) k->type;
        if (k_type_i != k_type_i_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key type (%d != %d, layer %d)\n", __func__, k_type_i, k_type_i_ref, il);
            // [TAG_TURBOT] a turbo5p blob into a turbot cache, or the reverse, stops here
            if (ggml_turbot_is_type(k->type) != ggml_turbot_is_type((ggml_type) k_type_i_ref)) {
                static bool logged = false;
                if (!logged) {
                    logged = true;
                    LLAMA_LOG_ERROR("%s: turbot: state blob type mismatch (turbo5p vs turbot)\n", __func__);
                }
            }
            return false;
        }

        // Read row size of key
        uint64_t k_size_row_ref;
        io.read(&k_size_row_ref, sizeof(k_size_row_ref));
        const size_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);
        if (k_size_row != k_size_row_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key row size (%zu != %zu, layer %d)\n", __func__, k_size_row, (size_t) k_size_row_ref, il);
            return false;
        }

        for (const auto & r : runs) {
            io.read_tensor(k, (size_t) r.from * k_size_row, (size_t) (r.to - r.from) * k_size_row);
        }
    }

    if (!this->v_trans) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            auto * v = layer.v_stream[strm];
            if (!v) {
                continue;
            }

            // Use actual tensor width (may be padded for turbo types)
            const uint32_t n_embd_v_gqa = (uint32_t) v->ne[0];

            // Read type of value
            int32_t v_type_i_ref;
            io.read(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t) v->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read row size of value
            uint64_t v_size_row_ref;
            io.read(&v_size_row_ref, sizeof(v_size_row_ref));
            const size_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
            if (v_size_row != v_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value row size (%zu != %zu, layer %d)\n", __func__, v_size_row, (size_t) v_size_row_ref, il);
                return false;
            }

            for (const auto & r : runs) {
                io.read_tensor(v, (size_t) r.from * v_size_row, (size_t) (r.to - r.from) * v_size_row);
            }
        }
    } else {
        // For each layer, read the values for each cell (transposed)
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[strm];
            if (!v) {
                continue;
            }

            // Read type of value
            int32_t v_type_i_ref;
            io.read(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t) v->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read element size of value
            uint32_t v_size_el_ref;
            io.read(&v_size_el_ref, sizeof(v_size_el_ref));
            const size_t v_size_el = ggml_type_size(v->type);
            if (v_size_el != v_size_el_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value element size (%zu != %zu, layer %d)\n", __func__, v_size_el, (size_t) v_size_el_ref, il);
                return false;
            }

            // Read GQA embedding size
            uint32_t n_embd_v_gqa_ref;
            io.read(&n_embd_v_gqa_ref, sizeof(n_embd_v_gqa_ref));
            if (n_embd_v_gqa != n_embd_v_gqa_ref) {
                LLAMA_LOG_ERROR("%s: mismatched GQA embedding size (%u != %u, layer %d)\n", __func__, n_embd_v_gqa, n_embd_v_gqa_ref, il);
                return false;
            }

            for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                for (const auto & r : runs) {
                    const size_t dst_offset = ((size_t) r.from + j * cells.size()) * v_size_el;
                    io.read_tensor(v, dst_offset, (size_t) (r.to - r.from) * v_size_el);
                }
            }
        }
    }

    return true;
}

//
// [TAG_TURBOT] llama_kv_cache: turbot tiered KV cache (docs/turbot/SPEC.md sections 9-10)
//

bool llama_kv_cache::seq_rm_turbot(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // [TAG_TURBOT_ANY_STREAMS] the stream of seq_id with its tier, or every stream for seq_id < 0 (one stream: stream 0)
    if (seq_id >= 0) {
        seq_rm_turbot_stream(seq_to_stream[seq_id], seq_id, p0, p1);
    } else {
        for (uint32_t s = 0; s < n_stream; ++s) {
            seq_rm_turbot_stream(s, seq_id, p0, p1);
        }
    }

    return true;
}

void llama_kv_cache::seq_rm_turbot_stream(uint32_t strm, llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    GGML_ASSERT(strm < n_stream && strm < turbot_tier.size());

    // a removal to the end of a sequence rolls its write-row counter back, so rejected draft rows leave no gap in stamp
    // space (SPEC 9.7). p1 is already normalised: max() here means the caller passed p1 < 0 or max() itself.
    const bool tail = p1 == std::numeric_limits<llama_pos>::max();

    auto & tier  = *turbot_tier[strm];
    auto & cells = v_cells[strm];
    auto & head  = v_heads[strm];

    uint32_t new_head = cells.size();

    // [TAG_SEQ_RM_BOUNDS] nothing outside [used_min, used_max_p1) can match
    const uint32_t i_beg = cells.used_min();
    const uint32_t i_end = cells.used_max_p1();

    if (seq_id >= 0) {
        if (cells.seq_pos_max(seq_id) < p0) {
            return;
        }

        uint64_t min_st  = std::numeric_limits<uint64_t>::max();
        bool     removed = false;

        for (uint32_t i = i_beg; i < i_end; ++i) {
            if (!cells.pos_in(i, p0, p1) || !cells.seq_has(i, seq_id)) {
                continue;
            }

            min_st  = std::min(min_st, tier.stamp(i));
            removed = true;

            if (cells.seq_rm(i, seq_id)) {
                tier.on_cell_emptied(i);
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }

        if (tail && removed) {
            tier.on_seq_tail_removed(seq_id, min_st);
        }
    } else {
        // match any sequence
        uint64_t                   min_st[LLAMA_MAX_SEQ];
        std::bitset<LLAMA_MAX_SEQ> seen;

        for (uint32_t i = i_beg; i < i_end; ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            const uint64_t st = tier.stamp(i);
            llama_turbot_for_each_seq(cells.seq_bits(i), [&](llama_seq_id s) {
                min_st[s] = seen.test(s) ? std::min(min_st[s], st) : st;
                seen.set(s);
            });

            cells.rm(i);
            tier.on_cell_emptied(i);

            if (new_head == cells.size()) {
                new_head = i;
            }
        }

        if (tail) {
            llama_turbot_for_each_seq(seen, [&](llama_seq_id s) {
                tier.on_seq_tail_removed(s, min_st[s]);
            });
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != cells.size() && new_head < head) {
        head = new_head;
    }
}

uint32_t llama_kv_cache::get_turbot_n_granules() const {
    GGML_ASSERT(is_turbot());

    return turbot_tier[0]->n_granules();
}

int64_t llama_kv_cache::get_turbot_n_fill(const slot_info & sinfo) const {
    GGML_ASSERT(is_turbot());

    return llama_turbot_streams_n_fill(turbot_tier, sinfo.strm);
}

const ggml_turbot_layer & llama_kv_cache::turbot_layer_of(int32_t il) const {
    GGML_ASSERT(turbot_plan);

    // [TAG_TURBOT_ANY_GEOM] a reused layer reads and writes the KV of the layer it maps to, with that layer's widths
    return turbot_plan->layers.at((int32_t) layers[map_layer_ids.at(il)].il);
}

ggml_tensor * llama_kv_cache::get_turbot_pool(int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);

    GGML_ASSERT(layers[ikv].pool != nullptr);

    return layers[ikv].pool;
}

void llama_kv_cache::get_turbot_op_params(int32_t il, int side, ggml_turbot_op_params & params) const {
    GGML_ASSERT(turbot_plan);

    ggml_turbot_op_params_make(&params, &turbot_layer_of(il), side);
}

ggml_tensor * llama_kv_cache::turbot_cpy(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * idxs, ggml_tensor * young, ggml_tensor * fill, int32_t il, int side) const {
    GGML_ASSERT(turbot_plan);

    const int32_t ikv = map_layer_ids.at(il);

    const auto & layer = layers[ikv];

    const ggml_turbot_layer & tl = turbot_layer_of(il);

    ggml_tensor * dst = side == GGML_TURBOT_SIDE_K ? layer.k : layer.v;

    const int64_t n_embd_head = cur->ne[0];
    const int64_t n_head      = cur->ne[1];
    const int64_t n_tokens    = cur->ne[2];

    // [TAG_TURBOT_ANY_GEOM] a row holds nr*256 values (1024 at 4 KV heads x 256) inside the 1024-wide container
    const int64_t row_elems = ggml_turbot_geom_row_elems(tl.flags);

    GGML_ASSERT(n_embd_head*n_head == row_elems);
    GGML_ASSERT(dst->ne[0] == GGML_TURBOT_ROW_ELEMS);

    // the heads of a row are laid out contiguously: merge dims 0 and 1 into one row, as cpy_k does
    GGML_ASSERT(ggml_row_size(cur->type, n_embd_head) == cur->nb[1]);

    cur = ggml_view_2d(ctx, cur, row_elems, n_tokens, cur->nb[2], 0);

    // [TAG_TURBOT_ANY_STREAMS] the idxs are global (stream*kv_size + cell), the young rows and fill entries too: flatten
    // the streams as cpy_k does. One stream: dst as before.
    if (dst->ne[2] > 1) {
        dst = ggml_reshape_2d(ctx, dst, dst->ne[0], dst->ne[1]*dst->ne[2]);
    }

    ggml_turbot_op_params params;
    ggml_turbot_op_params_make(&params, &tl, side);

    return ggml_turbot_set_rows(ctx, dst, cur, idxs, layer.pool, young, fill, &params);
}

ggml_tensor * llama_kv_cache::turbot_cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, ggml_tensor * young, ggml_tensor * fill, int32_t il) const {
    return turbot_cpy(ctx, k_cur, k_idxs, young, fill, il, GGML_TURBOT_SIDE_K);
}

ggml_tensor * llama_kv_cache::turbot_cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, ggml_tensor * young, ggml_tensor * fill, int32_t il) const {
    GGML_ASSERT(!v_trans);

    return turbot_cpy(ctx, v_cur, v_idxs, young, fill, il, GGML_TURBOT_SIDE_V);
}

void llama_kv_cache::turbot_begin_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch) {
    GGML_ASSERT(is_turbot());

    // [TAG_TURBOT_ANY_STREAMS] the tier of every stream of the ubatch on its rows; one stream: the tier on the whole
    // ubatch against the cells of its stream, as before
    llama_turbot_streams_begin(turbot_tier, ubatch, sinfo.strm, sinfo.idxs, v_cells);

    turbot_cur_strm = sinfo.strm;
}

void llama_kv_cache::turbot_commit_ubatch() {
    GGML_ASSERT(is_turbot());

    if (n_stream == 1) {
        turbot_tier[0]->commit_ubatch(v_cells[0]);
        return;
    }

    // [TAG_TURBOT_ANY_STREAMS] only the streams of the last begin, each against its own cells
    llama_turbot_streams_commit(turbot_tier, turbot_cur_strm, v_cells);
}

void llama_kv_cache::turbot_abort_ubatch() {
    GGML_ASSERT(is_turbot());

    if (n_stream == 1) {
        turbot_tier[0]->abort_ubatch();
        return;
    }

    // [TAG_TURBOT_ANY_STREAMS]
    llama_turbot_streams_abort(turbot_tier, turbot_cur_strm);
}

void llama_kv_cache::set_input_turbot(ggml_tensor * gtab, ggml_tensor * young, ggml_tensor * fill, const slot_info & sinfo) const {
    GGML_ASSERT(is_turbot());

    // [TAG_TURBOT_ANY_STREAMS] several streams: the composed, global inputs of llama-kv-tier.h
    if (n_stream > 1) {
        const int64_t n_view = (int64_t) (sinfo.s1 - sinfo.s0 + 1);
        const int64_t n_fill = llama_turbot_streams_n_fill(turbot_tier, sinfo.strm);

        if (gtab && gtab->buffer) {
            GGML_ASSERT(ggml_backend_buffer_is_host(gtab->buffer));
            GGML_ASSERT(gtab->type == GGML_TYPE_I32 && gtab->ne[0] == (int64_t) turbot_tier[0]->n_granules()*n_view);

            llama_turbot_streams_gtab(turbot_tier, sinfo.s0, sinfo.s1, (int32_t *) gtab->data);
        }

        if (young && young->buffer) {
            int64_t n_rows = 0;
            for (const llama_seq_id s : sinfo.strm) {
                n_rows += (int64_t) turbot_tier[s]->young_rows().size();
            }

            GGML_ASSERT(ggml_backend_buffer_is_host(young->buffer));
            GGML_ASSERT(young->type == GGML_TYPE_I32 && young->ne[0] == n_rows);

            llama_turbot_streams_young(turbot_tier, sinfo.strm, (int32_t *) young->data);

            // a graph built without the fill tensor must never run a ubatch that has fill entries (SPEC 10.1)
            GGML_ASSERT((fill != nullptr || n_fill == 0) && "turbot: graph reused across a fill");
        }

        if (fill && fill->buffer) {
            GGML_ASSERT(ggml_backend_buffer_is_host(fill->buffer));
            GGML_ASSERT(fill->type == GGML_TYPE_I32 && fill->ne[0] == 4 && fill->ne[1] == n_fill);

            llama_turbot_streams_fill(turbot_tier, sinfo.strm, (int32_t *) fill->data);
        }

        return;
    }

    const auto & tier = *turbot_tier[0];

    // an input the graph does not read stays unallocated
    if (gtab && gtab->buffer) {
        GGML_ASSERT(ggml_backend_buffer_is_host(gtab->buffer));
        GGML_ASSERT(gtab->type == GGML_TYPE_I32 && gtab->ne[0] == (int64_t) tier.n_granules());

        memcpy(gtab->data, tier.granule_slots().data(), ggml_nbytes(gtab));
    }

    if (young && young->buffer) {
        const auto & rows = tier.young_rows();

        GGML_ASSERT(ggml_backend_buffer_is_host(young->buffer));
        GGML_ASSERT(young->type == GGML_TYPE_I32 && young->ne[0] == (int64_t) rows.size());

        memcpy(young->data, rows.data(), ggml_nbytes(young));

        // a graph built without the fill tensor must never run a ubatch that has fill entries (SPEC 10.1)
        GGML_ASSERT((fill != nullptr || tier.fill_entries().empty()) && "turbot: graph reused across a fill");
    }

    if (fill && fill->buffer) {
        const auto & entries = tier.fill_entries();

        GGML_ASSERT(ggml_backend_buffer_is_host(fill->buffer));
        GGML_ASSERT(fill->type == GGML_TYPE_I32 && fill->ne[0] == 4 && fill->ne[1]*4 == (int64_t) entries.size());

        memcpy(fill->data, entries.data(), ggml_nbytes(fill));
    }
}

void llama_kv_cache::state_write_turbot(llama_io_write_i & io, const cell_ranges_t & cr, uint32_t cell_count, llama_seq_id seq_id) const {
    // [TAG_TURBOT_ANY_STREAMS] the tier of this stream; its pool rows start at row cr.strm*POOL_s (0 with one stream)
    const auto & tier    = *turbot_tier[cr.strm];
    const auto & cells   = v_cells[cr.strm];
    const size_t row_off = (size_t) cr.strm*turbot_pool_s;

    const uint32_t magic   = GGML_TURBOT_BLOB_MAGIC;
    const uint32_t version = GGML_TURBOT_BLOB_VERSION;
    const uint64_t hash    = turbot_plan->hash;

    io.write(&magic,      sizeof(magic));
    io.write(&version,    sizeof(version));
    io.write(&hash,       sizeof(hash));
    io.write(&cell_count, sizeof(cell_count));

    // a sequence blob carries the counter of the saved sequence, a full blob the counter of every sequence with cells
    std::vector<std::pair<int32_t, uint64_t>> counters;
    if (seq_id >= 0) {
        counters.emplace_back(seq_id, tier.row_counter(seq_id));
    } else {
        for (llama_seq_id s = 0; s < (llama_seq_id) LLAMA_MAX_SEQ; ++s) {
            if (cells.seq_n_cells(s) > 0) {
                counters.emplace_back(s, tier.row_counter(s));
            }
        }
    }

    const uint32_t n_counters = (uint32_t) counters.size();
    io.write(&n_counters, sizeof(n_counters));
    for (const auto & c : counters) {
        io.write(&c.first,  sizeof(c.first));
        io.write(&c.second, sizeof(c.second));
    }

    // stamps and young flags in blob cell order (the ranges of the meta section)
    std::vector<uint64_t> stamps;
    std::vector<uint8_t>  young;
    std::vector<uint32_t> young_cells;

    stamps.reserve(cell_count);
    young .reserve(cell_count);

    for (const auto & range : cr.data) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            const bool y = tier.cell_young(i);

            stamps.push_back(tier.stamp(i));
            young .push_back(y ? 1 : 0);

            if (y) {
                young_cells.push_back(i);
            }
        }
    }

    GGML_ASSERT(stamps.size() == cell_count);

    io.write(stamps.data(), stamps.size()*sizeof(uint64_t));
    io.write(young.data(),  young.size()*sizeof(uint8_t));

    const uint32_t n_young = (uint32_t) young_cells.size();
    io.write(&n_young, sizeof(n_young));

    // the layer table precedes every pool byte
    const uint32_t n_layer = (uint32_t) layers.size();
    io.write(&n_layer, sizeof(n_layer));

    for (const auto & layer : layers) {
        const uint32_t il             = layer.il;
        const uint32_t pool_row_bytes = (uint32_t) layer.pool->ne[0];

        io.write(&il,             sizeof(il));
        io.write(&pool_row_bytes, sizeof(pool_row_bytes));
    }

    // young pool rows, same layer order, one write per run of young cells with consecutive pool rows
    const auto & gslot = tier.granule_slots();

    for (const auto & layer : layers) {
        const size_t row_bytes = layer.pool->nb[1];

        size_t k = 0;
        while (k < young_cells.size()) {
            const int32_t row0 = ggml_turbot_pool_row(gslot[young_cells[k] >> GGML_TURBOT_LOG2_GRANULE], young_cells[k]);

            size_t n = 1;
            while (k + n < young_cells.size() &&
                   ggml_turbot_pool_row(gslot[young_cells[k + n] >> GGML_TURBOT_LOG2_GRANULE], young_cells[k + n]) == row0 + (int32_t) n) {
                ++n;
            }

            io.write_tensor(layer.pool, (row_off + (size_t) row0)*row_bytes, n*row_bytes);

            k += n;
        }
    }
}

bool llama_kv_cache::state_read_turbot(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo, llama_seq_id seq_id) {
    // [TAG_TURBOT_ANY_STREAMS] the tier of this stream; its pool rows start at row strm*POOL_s (0 with one stream)
    auto &       tier    = *turbot_tier[strm];
    auto &       cells   = v_cells[strm];
    const size_t row_off = (size_t) strm*turbot_pool_s;

    // 1. read and validate everything up to the raw pool bytes. On any mismatch the tier is untouched and the caller's
    //    failure path clears or seq_rm's the restored cells.
    uint32_t magic   = 0;
    uint32_t version = 0;
    io.read(&magic,   sizeof(magic));
    io.read(&version, sizeof(version));
    if (magic != GGML_TURBOT_BLOB_MAGIC || version != GGML_TURBOT_BLOB_VERSION) {
        LLAMA_LOG_ERROR("%s: turbot: not a turbot v%d state section (magic 0x%08x, version %u)\n", __func__,
                GGML_TURBOT_BLOB_VERSION, magic, version);
        return false;
    }

    uint64_t hash = 0;
    io.read(&hash, sizeof(hash));
    if (hash != turbot_plan->hash) {
        LLAMA_LOG_ERROR("%s: turbot: state blob from another plan (hash 0x%016llx, this cache 0x%016llx)\n", __func__,
                (unsigned long long) hash, (unsigned long long) turbot_plan->hash);
        return false;
    }

    uint32_t cell_count_ref = 0;
    io.read(&cell_count_ref, sizeof(cell_count_ref));
    if (cell_count_ref != cell_count) {
        LLAMA_LOG_ERROR("%s: turbot: state section holds %u cells, the stream %u\n", __func__, cell_count_ref, cell_count);
        return false;
    }

    uint32_t n_counters = 0;
    io.read(&n_counters, sizeof(n_counters));
    if (seq_id >= 0 ? n_counters != 1 : n_counters > (uint32_t) LLAMA_MAX_SEQ) {
        LLAMA_LOG_ERROR("%s: turbot: invalid counter count %u for a %s blob\n", __func__, n_counters, seq_id >= 0 ? "sequence" : "full");
        return false;
    }

    // a sequence blob restores into dest seq_id, whatever id it was saved from
    std::vector<std::pair<llama_seq_id, uint64_t>> counters;
    for (uint32_t k = 0; k < n_counters; ++k) {
        int32_t  s = -1;
        uint64_t v = 0;
        io.read(&s, sizeof(s));
        io.read(&v, sizeof(v));
        if (s < 0 || s >= (int32_t) LLAMA_MAX_SEQ) {
            LLAMA_LOG_ERROR("%s: turbot: invalid counter sequence id %d\n", __func__, s);
            return false;
        }
        counters.emplace_back(seq_id >= 0 ? seq_id : s, v);
    }

    std::vector<uint64_t> stamps(cell_count);
    std::vector<uint8_t>  young (cell_count);
    io.read(stamps.data(), (size_t) cell_count*sizeof(uint64_t));
    io.read(young.data(),  (size_t) cell_count*sizeof(uint8_t));

    uint32_t n_young_flags = 0;
    for (const uint8_t y : young) {
        if (y > 1) {
            LLAMA_LOG_ERROR("%s: turbot: invalid young flag %u\n", __func__, y);
            return false;
        }
        n_young_flags += y;
    }

    uint32_t n_young = 0;
    io.read(&n_young, sizeof(n_young));
    if (n_young != n_young_flags) {
        LLAMA_LOG_ERROR("%s: turbot: %u young cells announced, %u flagged\n", __func__, n_young, n_young_flags);
        return false;
    }

    uint32_t n_layer = 0;
    io.read(&n_layer, sizeof(n_layer));
    if (n_layer != layers.size()) {
        LLAMA_LOG_ERROR("%s: turbot: mismatched layer count (%u instead of %u)\n", __func__, n_layer, (uint32_t) layers.size());
        return false;
    }

    for (const auto & layer : layers) {
        uint32_t il_ref             = 0;
        uint32_t pool_row_bytes_ref = 0;
        io.read(&il_ref,             sizeof(il_ref));
        io.read(&pool_row_bytes_ref, sizeof(pool_row_bytes_ref));
        if (il_ref != layer.il || pool_row_bytes_ref != (uint32_t) layer.pool->ne[0]) {
            LLAMA_LOG_ERROR("%s: turbot: mismatched layer table entry (layer %u, %u B per pool row; this cache: layer %u, %u B)\n", __func__,
                    il_ref, pool_row_bytes_ref, layer.il, (uint32_t) layer.pool->ne[0]);
            return false;
        }
    }

    GGML_ASSERT(sinfo.n_stream() == 1 && sinfo.idxs[0].size() == cell_count);

    // 2. stamps, counters and young flags into the tier; may allocate or evict slots
    const std::vector<int32_t> rows = tier.restore_cells(sinfo.idxs[0], stamps, young, counters, cells);

    // 3. stream the pool bytes: runs of young cells with consecutive pool rows go straight into the pool, the bytes of
    //    cells whose refinement was dropped go through a scratch buffer of at most one granule of rows. The bytes are not
    //    read ahead of step 2: a 4-slot full blob carries up to 761 MiB of them.
    std::vector<size_t> young_idx;
    young_idx.reserve(n_young);
    for (uint32_t i = 0; i < cell_count; ++i) {
        if (young[i]) {
            young_idx.push_back(i);
        }
    }

    try {
        std::vector<uint8_t> scratch;

        for (const auto & layer : layers) {
            const size_t row_bytes = layer.pool->nb[1];

            size_t k = 0;
            while (k < young_idx.size()) {
                const int32_t row0 = rows[young_idx[k]];

                size_t n = 1;
                if (row0 >= 0) {
                    while (k + n < young_idx.size() && rows[young_idx[k + n]] == row0 + (int32_t) n) {
                        ++n;
                    }
                    io.read_tensor(layer.pool, (row_off + (size_t) row0)*row_bytes, n*row_bytes);
                } else {
                    while (k + n < young_idx.size() && rows[young_idx[k + n]] < 0 && n < (size_t) GGML_TURBOT_GRANULE) {
                        ++n;
                    }
                    scratch.resize(n*row_bytes);
                    io.read(scratch.data(), n*row_bytes);
                }

                k += n;
            }
        }
    } catch (...) {
        // short blob or io error: free the slots this restore allocated (SPEC 9.10); evictions it made stay
        tier.abort_restore();
        LLAMA_LOG_ERROR("%s: turbot: state blob truncated inside the young pool bytes\n", __func__);
        return false;
    }

    return true;
}

//
// llama_kv_cache: TriAttention integration
//

void llama_kv_cache::init_triattention(const char * stats_path, const triattention_config * cfg) {
    if (!stats_path || stats_path[0] == '\0') {
        return;
    }
    // [TAG_TURBOT] TriAttention evicts cells and rewrites K through the CPU path; neither knows the tier (SPEC 9.3)
    if (is_turbot()) {
        LLAMA_LOG_ERROR("%s: turbot: TriAttention is unsupported on a turbot KV cache\n", __func__);
        return;
    }
    if (triattention_st) {
        triattention_free(triattention_st);
        triattention_st = nullptr;
    }

    const uint32_t kv_size = v_cells.empty() ? 0 : (uint32_t)v_cells[0].size();
    const double rope_theta = (double)hparams.rope_freq_base_train;
    const uint32_t head_dim = hparams.n_embd_head_k(0);
    const uint32_t n_kv_heads = hparams.n_head_kv(0);

    triattention_st = triattention_init(stats_path, cfg, kv_size, rope_theta, head_dim, n_kv_heads);
    if (!triattention_st) {
        LLAMA_LOG_ERROR("%s: failed to initialize TriAttention from %s\n", __func__, stats_path);
    }
}

int32_t llama_kv_cache::triattention_try_prune() {
    if (!triattention_st) {
        return 0;
    }

    // Build K tensor array and layer map for triattention_prune_impl()
    const uint32_t n_kv_layers = (uint32_t)layers.size();
    std::vector<ggml_tensor *> k_tensors(n_kv_layers);
    std::vector<int32_t> layer_map(n_kv_layers);

    for (uint32_t i = 0; i < n_kv_layers; i++) {
        k_tensors[i] = layers[i].k;
        layer_map[i] = (int32_t)layers[i].il;
    }

    const uint32_t kv_size = (uint32_t)v_cells[0].size();

    int32_t n_evicted = triattention_prune_impl(
        triattention_st,
        k_tensors.data(),
        n_kv_layers,
        layer_map.data(),
        kv_size);

    if (n_evicted > 0) {
        // Sync cell metadata: remove cells that triattention marked as evicted
        // (cell_positions[i] == -1 after prune means the cell was evicted)
        auto & cells = v_cells[0];
        auto & head  = v_heads[0];
        uint32_t new_head = cells.size();

        for (uint32_t i = 0; i < kv_size; i++) {
            if (triattention_st->cell_positions[i] < 0 && !cells.is_empty(i)) {
                cells.rm(i);
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }

        if (new_head != cells.size() && new_head < head) {
            head = new_head;
        }

        // Note: position gaps from evicted cells are intentionally left as-is.
        // The recent-token protection in triattention_prune_impl() ensures that
        // the most recent divide_length tokens are never evicted, so seq_pos_max
        // always equals the server's expected position. The batch validation
        // (Y = X + 1) passes because seq_pos_max is unchanged. Position gaps in
        // the middle are harmless: RoPE handles non-consecutive positions, and
        // nothing in the attention computation requires contiguous positions.
    }

    return n_evicted;
}

bool llama_kv_cache::has_triattention() const {
    return triattention_st != nullptr;
}

//
// llama_kv_cache_context
//

llama_kv_cache_context::llama_kv_cache_context(llama_memory_status status) : status(status) {}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv) {
    n_kv = kv->get_size();

    const uint32_t n_stream = kv->get_n_stream();

    // create a dummy slot info - the actual data is irrelevant. we just need to build the graph
    sinfos.resize(1);
    sinfos[0].s0 = 0;
    sinfos[0].s1 = n_stream - 1;
    sinfos[0].idxs.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        sinfos[0].strm.push_back(s);
        sinfos[0].idxs[s].resize(1, 0);
    }
}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv,
        llama_context * lctx,
        bool do_shift,
        stream_copy_info sc_info) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), lctx(lctx), do_shift(do_shift), sc_info(std::move(sc_info)) {
    if (!do_shift && this->sc_info.empty()) {
        status = LLAMA_MEMORY_STATUS_NO_UPDATE;
    }
}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv,
        llama_kv_cache::slot_info_vec_t sinfos,
        std::vector<llama_ubatch> ubatches) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), sinfos(std::move(sinfos)), ubatches(std::move(ubatches)) {
}

llama_kv_cache_context::~llama_kv_cache_context() = default;

bool llama_kv_cache_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (++i_cur >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    // no ubatches -> this is a KV cache update
    if (ubatches.empty()) {
        kv->update(lctx, do_shift, sc_info);

        return true;
    }

    kv->apply_ubatch(sinfos[i_cur], ubatches[i_cur]);
    n_kv = kv->get_n_kv(sinfos[i_cur]);

    // [TAG_TURBOT] every row of this ubatch lands in a young granule before compute (SPEC 9.6). prepare() and
    // state_read_meta() call apply_ubatch() directly and never reach this.
    if (kv->is_turbot()) {
        kv->turbot_begin_ubatch(sinfos[i_cur], ubatches[i_cur]);
    }

    // InnerQ: check if CUDA calibration finalized and tensor needs update
    if (kv->get_turbo_innerq_scale_inv() != nullptr && turbo_innerq_needs_tensor_update()) {
        ggml_tensor * t = kv->get_turbo_innerq_scale_inv();
        if (t->buffer != nullptr) {
            ggml_backend_tensor_set(t, g_innerq_scale_inv_host, 0, INNERQ_MAX_CHANNELS * sizeof(float));
            turbo_innerq_mark_tensor_updated();
            LLAMA_LOG_INFO("%s: InnerQ scale_inv tensor updated\n", __func__);
        }
    }

    return true;
}

llama_memory_status llama_kv_cache_context::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_cur];
}

uint32_t llama_kv_cache_context::get_n_kv() const {
    return n_kv;
}

ggml_type llama_kv_cache_context::type_k() const {
    return kv->type_k();
}

ggml_type llama_kv_cache_context::type_v() const {
    return kv->type_v();
}

ggml_tensor * llama_kv_cache_context::get_k(ggml_context * ctx, int32_t il) const {
    return kv->get_k(ctx, il, n_kv, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::get_v(ggml_context * ctx, int32_t il) const {
    return kv->get_v(ctx, il, n_kv, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::get_turbo_rotation() const {
    return kv->get_turbo_rotation();
}

ggml_tensor * llama_kv_cache_context::get_turbo_rotation_inv() const {
    return kv->get_turbo_rotation_inv();
}

ggml_tensor * llama_kv_cache_context::get_turbo_rot_forward() const {
    return kv->get_turbo_rotation();
}

ggml_tensor * llama_kv_cache_context::get_turbo_rot_inverse() const {
    return kv->get_turbo_rotation_inv();
}

ggml_tensor * llama_kv_cache_context::get_turbo_innerq_scale_inv() const {
    return kv->get_turbo_innerq_scale_inv();
}

ggml_tensor * llama_kv_cache_context::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const {
    return kv->cpy_k(ctx, k_cur, k_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const {
    return kv->cpy_v(ctx, v_cur, v_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_k_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_context::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_v_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_context::build_input_k_rot(ggml_context * ctx) const {
    return kv->build_input_k_rot(ctx);
}

ggml_tensor * llama_kv_cache_context::build_input_v_rot(ggml_context * ctx) const {
    return kv->build_input_v_rot(ctx);
}

void llama_kv_cache_context::set_input_k_shift(ggml_tensor * dst) const {
    kv->set_input_k_shift(dst);
}

void llama_kv_cache_context::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_k_idxs(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_v_idxs(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    kv->set_input_kq_mask(dst, ubatch, causal_attn);
}

void llama_kv_cache_context::set_input_kv_pos(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_kv_pos(dst, ubatch);
}

void llama_kv_cache_context::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_pos_bucket(dst, ubatch);
}

void llama_kv_cache_context::set_input_k_rot(ggml_tensor * dst) const {
    kv->set_input_k_rot(dst);
}

void llama_kv_cache_context::set_input_v_rot(ggml_tensor * dst) const {
    kv->set_input_v_rot(dst);
}

void llama_kv_cache_context::get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const {
    kv->get_prev_tokens(ubatch, n, res);
}

// [TAG_TURBOT]

bool llama_kv_cache_context::is_turbot() const {
    return kv->is_turbot();
}

int64_t llama_kv_cache_context::get_turbot_n_stream() const {
    // [TAG_TURBOT_ANY_STREAMS] the streams of the K/V view of the current ubatch (get_k: s1 - s0 + 1); the full-cache
    // context's dummy slot info spans every stream
    if (sinfos.empty()) {
        return kv->get_n_stream();
    }

    const auto & sinfo = sinfos[i_cur];

    return (int64_t) (sinfo.s1 - sinfo.s0 + 1);
}

int64_t llama_kv_cache_context::get_turbot_n_granules() const {
    // [TAG_TURBOT_ANY_STREAMS] the granule table is stream-major over the view streams (one stream: kv_size / 64)
    return (int64_t) kv->get_turbot_n_granules()*get_turbot_n_stream();
}

int64_t llama_kv_cache_context::get_turbot_n_fill() const {
    // the tier's fill list belongs to the ubatch its last begin_ubatch saw, which is the current one only in a batch
    // processing context after apply(). A graph reserve context has no ubatches and builds no fill tensor.
    if (ubatches.empty() || !kv->is_turbot()) {
        return 0;
    }

    // [TAG_TURBOT_ANY_STREAMS] the tiers of the streams of the current ubatch
    return kv->get_turbot_n_fill(sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::get_turbot_pool(int32_t il) const {
    return kv->get_turbot_pool(il);
}

void llama_kv_cache_context::get_turbot_op_params(int32_t il, int side, ggml_turbot_op_params & params) const {
    kv->get_turbot_op_params(il, side, params);
}

ggml_tensor * llama_kv_cache_context::turbot_cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, ggml_tensor * young, ggml_tensor * fill, int32_t il) const {
    return kv->turbot_cpy_k(ctx, k_cur, k_idxs, young, fill, il);
}

ggml_tensor * llama_kv_cache_context::turbot_cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, ggml_tensor * young, ggml_tensor * fill, int32_t il) const {
    return kv->turbot_cpy_v(ctx, v_cur, v_idxs, young, fill, il);
}

void llama_kv_cache_context::set_input_turbot(ggml_tensor * gtab, ggml_tensor * young, ggml_tensor * fill) const {
    kv->set_input_turbot(gtab, young, fill, sinfos[i_cur]);
}
