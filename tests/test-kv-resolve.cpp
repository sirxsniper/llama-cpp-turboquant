// [TAG_KV_RESOLVE] CPU unit test of the KV cache type resolver (src/llama-kv-cache-resolve.h).
//
// Drives the pure decision function llama_kv_resolve() with synthetic hparams of real model shapes, and the turbot
// precondition helpers the cache constructor shares with it. The plan check uses the real plan chooser and parser
// (src/llama-kv-tier.h: llama_turbot_plan_choose with model = nullptr, llama_turbot_plan_parse_shape) on the built-in
// plan, the automatic plan and synthetic plans. No GPU, no model.
// [TAG_TURBOT_ANY_RESOLVE] the turbot env switches (LLAMA_TURBOT_ANY, _AUTO_PLAN, _AUTO_BUDGET, _ISWA, _SWA_TYPE,
// _MULTI_STREAM, _PLAN) are read on every call, so each test sets them and reset_env() clears them again.
// Exit code 0 iff every check passes.

#include "ggml.h"
#include "ggml-turbot.h"
#include "llama.h"

#include "../src/llama-kv-cache-resolve.h"
#include "../src/llama-kv-tier.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

static int g_fail   = 0;
static int g_checks = 0;

#define TCHECK(cond, ...)                                                   \
    do {                                                                    \
        ++g_checks;                                                         \
        if (!(cond)) {                                                      \
            ++g_fail;                                                       \
            fprintf(stderr, "FAIL %s:%d: ", __func__, __LINE__);            \
            fprintf(stderr, __VA_ARGS__);                                   \
            fprintf(stderr, "\n");                                          \
        }                                                                   \
    } while (0)

static const ggml_type T_TURBOT = GGML_TYPE_TURBOT_S8;
static const ggml_type T_5P     = GGML_TYPE_TURBO5P_0;
static const ggml_type T_5P512  = GGML_TYPE_TURBO5P512_0;
static const ggml_type T_4P     = GGML_TYPE_TURBO4P_0;
static const ggml_type T_4      = GGML_TYPE_TURBO4_0;
static const ggml_type T_Q8     = GGML_TYPE_Q8_0;
static const ggml_type T_F16    = GGML_TYPE_F16;
static const ggml_type T_SAME   = GGML_TYPE_COUNT;   // type_k_swa/type_v_swa: no iSWA split

static const char * tn(ggml_type t) {
    return t == GGML_TYPE_COUNT ? "same" : llama_kv_resolve_type_name(t);
}

using plan_check_fn = std::function<bool(const llama_turbot_cache_shape &, std::string &)>;

//
// environment
//

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

// every switch the resolver, the plan chooser and the cache constructor read, back to unset (= the defaults)
static void reset_env() {
    static const char * names[] = {
        "LLAMA_TURBOT_ANY", "LLAMA_TURBOT_AUTO_PLAN", "LLAMA_TURBOT_AUTO_BUDGET", "LLAMA_TURBOT_AUTO_PLAN_DUMP",
        "LLAMA_TURBOT_SIDECAR", "LLAMA_TURBOT_ISWA", "LLAMA_TURBOT_SWA_TYPE", "LLAMA_TURBOT_MULTI_STREAM",
        "LLAMA_TURBOT_PLAN", "LLAMA_KV_RESOLVE", "TURBO_KV_CPU_LAYERS", "TURBO_LAYER_ADAPTIVE", "TURBO_INNERQ",
    };
    for (const char * n : names) {
        set_env(n, nullptr);
    }
}

//
// plan checks
//

// the plan chooser llama_context uses (model = nullptr: no sidecar). Records the kind of the last plan it chose.
static int                    g_plan_chosen = 0;
static llama_turbot_plan_kind g_plan_kind   = LLAMA_TURBOT_PLAN_KIND_FILE;

static const char * kind_name(llama_turbot_plan_kind k) {
    switch (k) {
        case LLAMA_TURBOT_PLAN_KIND_FILE:    return "file";
        case LLAMA_TURBOT_PLAN_KIND_BUILTIN: return "built-in";
        case LLAMA_TURBOT_PLAN_KIND_SIDECAR: return "sidecar";
        case LLAMA_TURBOT_PLAN_KIND_AUTO:    return "auto";
    }
    return "?";
}

static plan_check_fn plan_check_choose() {
    return [](const llama_turbot_cache_shape & shape, std::string & why) {
        llama_turbot_plan_choice choice;
        const bool ok = llama_turbot_plan_choose(shape, choice, why);
        if (ok) {
            ++g_plan_chosen;
            g_plan_kind = choice.kind;
        }
        return ok;
    };
}

// the plan check llama_context uses under LLAMA_TURBOT_ANY=0: today's plan source and match check
static plan_check_fn plan_check_legacy() {
    return [](const llama_turbot_cache_shape & shape, std::string & why) {
        std::vector<int32_t> attn_layers;
        for (const auto & g : shape.layers) {
            attn_layers.push_back(g.il);
        }
        const llama_turbot_plan_source src = llama_turbot_plan_get_source();
        std::string text;
        if (!llama_turbot_plan_read(src, text, why)) {
            return false;
        }
        return llama_turbot_plan_matches(text, attn_layers, shape.kv_size, why, src.name);
    };
}

// a plan with an L line for each layer in layers (5-bit old widths of 4 heads, the default young width)
static std::string plan_text_for(const std::vector<int32_t> & layers) {
    std::string s = "# synthetic test plan\nPOOL 65536\nCAP 16384\n";
    for (const int32_t il : layers) {
        s += "L " + std::to_string(il) + " K 5 5 5 5 V 5 5 5 5\n";
    }
    return s;
}

static plan_check_fn plan_check_text(const std::string & text) {
    return [text](const llama_turbot_cache_shape & shape, std::string & why) {
        llama_turbot_plan plan;
        return llama_turbot_plan_parse_shape(text, "synthetic", shape, plan, why, true);
    };
}

//
// model shapes
//

// a model of n_layer layers, all with n_head_kv heads of size head (K and V); attn(il) marks the attention cache layers.
// The main context (ctx_default), one stream, 262144 cells, the real plan chooser.
static llama_kv_resolve_input make_model(ggml_type tk, ggml_type tv, uint32_t n_layer, uint32_t n_head_kv, uint32_t head,
        const std::function<bool(uint32_t)> & attn = nullptr) {
    llama_kv_resolve_input in;
    in.type_k      = tk;
    in.type_v      = tv;
    in.flash_attn  = LLAMA_FLASH_ATTN_TYPE_AUTO;
    in.n_stream    = 1;
    in.n_seq_max   = 1;
    in.kv_size     = 262144;
    in.ctx_default = true;
    in.plan_check  = plan_check_choose();
    for (uint32_t il = 0; il < n_layer; ++il) {
        llama_kv_resolve_layer L;
        L.il        = (int32_t) il;
        L.head_k    = head;
        L.head_v    = head;
        L.n_head_kv = n_head_kv;
        L.attn      = attn ? attn(il) : true;
        in.layers.push_back(L);
    }
    return in;
}

// an iSWA model: every layer is held by the attention cache, full(il) marks the full-attention ones, the rest are SWA
static llama_kv_resolve_input make_iswa(ggml_type tk, ggml_type tv, uint32_t n_layer, uint32_t n_head_kv, uint32_t head,
        const std::function<bool(uint32_t)> & full) {
    llama_kv_resolve_input in = make_model(tk, tv, n_layer, n_head_kv, head);
    in.swa  = true;
    in.iswa = true;
    for (auto & L : in.layers) {
        L.swa = !full((uint32_t) L.il);
    }
    return in;
}

static bool every_4th(uint32_t il) {
    return (il + 1) % 4 == 0;
}

static std::vector<int32_t> attn_layers_of(const llama_kv_resolve_input & in) {
    std::vector<int32_t> r;
    for (const auto & L : in.layers) {
        if (L.attn) {
            r.push_back(L.il);
        }
    }
    return r;
}

// Qwen3.8-27B: 64 trunk layers + 1 nextn (MTP) layer, full attention every 4th layer (il = 3, 7, ..., 63), GDN
// elsewhere. Every layer carries 4 KV heads x 256 in the hparams; the attention cache holds only the 16 attention layers.
static llama_kv_resolve_input qwen38(ggml_type tk, ggml_type tv) {
    return make_model(tk, tv, 65, 4, 256, [](uint32_t il) { return il < 64 && every_4th(il); });
}

// Ornith-1.5-9B: qwen35 hybrid, 32 trunk layers + MTP layer 32, attention 3, 7, ..., 31 (8), 4 x 256
static llama_kv_resolve_input ornith_9b(ggml_type tk, ggml_type tv, bool mtp = false) {
    llama_kv_resolve_input in = make_model(tk, tv, 33, 4, 256,
            [mtp](uint32_t il) { return mtp ? il == 32 : il < 32 && every_4th(il); });
    in.ctx_default = !mtp;
    return in;
}

// Ornith-1.5-35B: qwen35moe hybrid, 40 trunk layers + MTP layer 40, attention 3, 7, ..., 39 (10), 2 x 256
static llama_kv_resolve_input ornith_35b(ggml_type tk, ggml_type tv) {
    return make_model(tk, tv, 41, 2, 256, [](uint32_t il) { return il < 40 && every_4th(il); });
}

// Spark-X2.5-4B: spark2_5 iSWA (window 512), 36 layers, full attention 3, 7, ..., 35 (9), 27 SWA, 4 x 256
static llama_kv_resolve_input spark_4b(ggml_type tk, ggml_type tv) {
    return make_iswa(tk, tv, 36, 4, 256, every_4th);
}

// Spark-X2.5-1.7B: iSWA, 28 layers, 2 x 256 (512-element rows); full attention every 4th layer (synthetic pattern)
static llama_kv_resolve_input spark_1_7b(ggml_type tk, ggml_type tv) {
    return make_iswa(tk, tv, 28, 2, 256, every_4th);
}

// MiniCPM5-2B: llama, 42 layers, 2 x 128 (256-element rows)
static llama_kv_resolve_input minicpm5(ggml_type tk, ggml_type tv) {
    return make_model(tk, tv, 42, 2, 128);
}

// Muse-Glimmer-30B: iSWA (window 2048), 52 layers, 13 full-attention (every 4th), 2 x 128
static llama_kv_resolve_input muse_glimmer(ggml_type tk, ggml_type tv) {
    return make_iswa(tk, tv, 52, 2, 128, every_4th);
}

// Nemotron-3.5-Lightning-30B-A3B: nemotron_h_moe hybrid, 52 layers, attention 5, 12, 19, 26, 33, 42 (6) with 2 x 128;
// the Mamba and MoE layers hold no KV row
static llama_kv_resolve_input nemotron_h(ggml_type tk, ggml_type tv) {
    const auto attn = [](uint32_t il) { return il == 5 || il == 12 || il == 19 || il == 26 || il == 33 || il == 42; };
    llama_kv_resolve_input in = make_model(tk, tv, 52, 2, 128, attn);
    for (auto & L : in.layers) {
        if (!L.attn) {
            L.n_head_kv = 0;
        }
    }
    return in;
}

// Granite 4.2 8B: 40 layers, 8 x 128 (1024-element rows)
static llama_kv_resolve_input granite(ggml_type tk, ggml_type tv) {
    return make_model(tk, tv, 40, 8, 128);
}

//
// result checks
//

static bool has_step(const std::vector<llama_kv_resolve_step> & steps, char side, ggml_type from, ggml_type to, const char * reason_part) {
    for (const auto & st : steps) {
        if (st.side == side && st.from == from && st.to == to &&
            (reason_part == nullptr || st.reason.find(reason_part) != std::string::npos)) {
            return true;
        }
    }
    return false;
}

static bool has_step(const llama_kv_resolve_result & r, char side, ggml_type from, ggml_type to, const char * reason_part) {
    return has_step(r.steps, side, from, to, reason_part);
}

static bool is_pair(const llama_kv_resolve_result & r, ggml_type k, ggml_type v, ggml_type k_swa = T_SAME, ggml_type v_swa = T_SAME) {
    return r.type_k == k && r.type_v == v && r.type_k_swa == k_swa && r.type_v_swa == v_swa;
}

static void dump(const char * what, const llama_kv_resolve_result & r) {
    fprintf(stderr, "  %-52s -> K %-10s V %-10s", what, tn(r.type_k), tn(r.type_v));
    if (r.type_k_swa != T_SAME || r.type_v_swa != T_SAME) {
        fprintf(stderr, " SWA K %-10s V %-10s", tn(r.type_k_swa), tn(r.type_v_swa));
    }
    if (r.no_kv) {
        fprintf(stderr, " (no KV)");
    }
    fprintf(stderr, "\n");
    for (const auto & st : r.steps) {
        fprintf(stderr, "      step %c: %s -> %s (%s)\n", st.side, tn(st.from), tn(st.to), st.reason.c_str());
    }
    for (const auto & st : r.steps_swa) {
        fprintf(stderr, "      SWA step %c: %s -> %s (%s)\n", st.side, tn(st.from), tn(st.to), st.reason.c_str());
    }
    if (!r.swa_note.empty()) {
        fprintf(stderr, "      SWA note: %s\n", r.swa_note.c_str());
    }
}

// resolve, dump, and remember which plan the chooser picked (g_plan_chosen == 0: none)
static llama_kv_resolve_result resolve(const char * what, const llama_kv_resolve_input & in) {
    g_plan_chosen = 0;
    const auto r = llama_kv_resolve(in);
    dump(what, r);
    if (ggml_turbot_is_type(r.type_k) && g_plan_chosen > 0) {
        fprintf(stderr, "      plan: %s\n", kind_name(g_plan_kind));
    }
    return r;
}

//
// turbot preconditions shared with the cache constructor
//

static void test_turbot_helpers() {
    reset_env();

    llama_turbot_cache_desc d;
    d.k_turbot = true;
    d.v_turbot = true;
    d.kv_size  = 262144;
    TCHECK(llama_turbot_cache_refusal(d).empty(), "a valid cache refused: %s", llama_turbot_cache_refusal(d).c_str());

    llama_turbot_cache_desc e = d;
    e.v_turbot = false;
    TCHECK(llama_turbot_cache_refusal(e) == "needs -ctk turbot and -ctv turbot together", "pairing");
    e = d; e.v_trans = true;
    TCHECK(llama_turbot_cache_refusal(e) == "needs flash attention", "flash attention");
    e = d; e.mla = true;
    TCHECK(llama_turbot_cache_refusal(e) == "shared cells and MLA caches are unsupported", "mla");
    e = d; e.shared_cells = true;
    TCHECK(llama_turbot_cache_refusal(e) == "shared cells and MLA caches are unsupported", "shared cells");
    e = d; e.swa = true;   // an SWA cache (the SWA child, or a non-iSWA SWA cache) is still refused
    TCHECK(llama_turbot_cache_refusal(e) == "SWA caches are unsupported", "swa");
    e = d; e.kv_size = 1000;
    TCHECK(llama_turbot_cache_refusal(e) == "kv_size must be a multiple of 64, got 1000", "kv_size: %s", llama_turbot_cache_refusal(e).c_str());

    // [TAG_TURBOT_ANY_STREAMS] several streams: accepted (per-stream tiers), refused with MULTI_STREAM=0 or ANY=0
    e = d; e.n_stream = 4;
    TCHECK(llama_turbot_cache_refusal(e).empty(), "streams refused: %s", llama_turbot_cache_refusal(e).c_str());
    set_env("LLAMA_TURBOT_MULTI_STREAM", "0");
    TCHECK(llama_turbot_cache_refusal(e) == "needs a single KV stream: use --kv-unified or -np 1", "MULTI_STREAM=0");
    reset_env();
    set_env("LLAMA_TURBOT_ANY", "0");
    TCHECK(llama_turbot_cache_refusal(e) == "needs a single KV stream: use --kv-unified or -np 1", "ANY=0 streams");
    reset_env();

    // [TAG_TURBOT_ANY_RESOLVE] every geometry the kernels take, K = V
    TCHECK(llama_turbot_layer_refusal(3, 256, 256, 4).empty(), "4 x 256 refused");
    TCHECK(llama_turbot_layer_refusal(3, 256, 256, 2).empty(), "2 x 256 refused");
    TCHECK(llama_turbot_layer_refusal(3, 256, 256, 1).empty(), "1 x 256 refused");
    TCHECK(llama_turbot_layer_refusal(3, 128, 128, 8).empty(), "8 x 128 refused");
    TCHECK(llama_turbot_layer_refusal(3, 128, 128, 4).empty(), "4 x 128 refused");
    TCHECK(llama_turbot_layer_refusal(3, 128, 128, 2).empty(), "2 x 128 refused: %s", llama_turbot_layer_refusal(3, 128, 128, 2).c_str());
    {
        const std::string why = llama_turbot_layer_refusal(5, 64, 64, 8);
        TCHECK(why.find("layer 5") != std::string::npos && why.find("8 KV heads x K 64") != std::string::npos, "8 x 64: %s", why.c_str());
        TCHECK(!llama_turbot_layer_refusal(3, 256, 128, 4).empty(), "K 256 / V 128 accepted");
        TCHECK(!llama_turbot_layer_refusal(3, 128, 128, 16).empty(), "16 x 128 accepted");
        TCHECK(!llama_turbot_layer_refusal(3, 512, 512, 2).empty(), "2 x 512 accepted");
    }
    // LLAMA_TURBOT_ANY=0: 4 x 256 only, with the old text
    set_env("LLAMA_TURBOT_ANY", "0");
    TCHECK(llama_turbot_layer_refusal(3, 256, 256, 4).empty(), "ANY=0: 4 x 256 refused");
    TCHECK(llama_turbot_layer_refusal(3, 128, 128, 2) ==
           "unsupported head geometry on layer 3: head_k 128, head_v 128, n_head_kv 2 (needs 256, 256, 4)",
           "ANY=0 geometry: %s", llama_turbot_layer_refusal(3, 128, 128, 2).c_str());
    reset_env();

    TCHECK(llama_turbot_layer_device_refusal(7, nullptr) == "attention KV must be on a CUDA device (layer 7: KV offload is off)",
           "device: %s", llama_turbot_layer_device_refusal(7, nullptr).c_str());

    // the env variables the constructor rejects
    TCHECK(llama_turbot_env_refusal().empty(), "clean env refused: %s", llama_turbot_env_refusal().c_str());
    set_env("TURBO_INNERQ", "0");
    TCHECK(llama_turbot_env_refusal().empty(), "TURBO_INNERQ=0 refused");
    set_env("TURBO_INNERQ", "1");
    TCHECK(llama_turbot_env_refusal() == "TURBO_INNERQ is incompatible", "TURBO_INNERQ=1: %s", llama_turbot_env_refusal().c_str());
    set_env("TURBO_INNERQ", nullptr);
    set_env("TURBO_KV_CPU_LAYERS", "2");
    TCHECK(llama_turbot_env_refusal().rfind("TURBO_KV_CPU_LAYERS", 0) == 0, "TURBO_KV_CPU_LAYERS=2");
    reset_env();
}

// [TAG_TURBOT_ANY_RESOLVE] the plan shape the resolver hands to the plan check
static void test_shape() {
    reset_env();

    const llama_turbot_cache_shape q = llama_kv_resolve_turbot_shape_of(qwen38(T_TURBOT, T_TURBOT));
    TCHECK(q.layers.size() == 16 && q.layers.front().il == 3 && q.layers.back().il == 63, "Qwen shape: %zu layers", q.layers.size());
    TCHECK(q.layers.front().head_dim == 256 && q.layers.front().n_head_kv == 4, "Qwen shape geometry");
    TCHECK(q.kv_size == 262144 && q.n_stream == 1 && q.n_seq_max == 1 && q.auto_ok && q.model == nullptr, "Qwen shape fields");

    // an iSWA split: the full-attention layers only
    const llama_turbot_cache_shape s = llama_kv_resolve_turbot_shape_of(spark_4b(T_TURBOT, T_TURBOT));
    TCHECK(s.layers.size() == 9 && s.layers.front().il == 3 && s.layers.back().il == 35, "Spark shape: %zu layers", s.layers.size());
    set_env("LLAMA_TURBOT_ISWA", "0");
    TCHECK(llama_kv_resolve_turbot_shape_of(spark_4b(T_TURBOT, T_TURBOT)).layers.size() == 36, "Spark shape, ISWA=0");
    reset_env();

    // an MTP context: no automatic plan
    TCHECK(!llama_kv_resolve_turbot_shape_of(ornith_9b(T_TURBOT, T_TURBOT, true)).auto_ok, "MTP shape: auto_ok");

    const llama_turbot_cache_shape m = llama_kv_resolve_turbot_shape_of(ornith_35b(T_TURBOT, T_TURBOT));
    TCHECK(m.layers.size() == 10 && m.layers.front().n_head_kv == 2 && m.layers.front().head_dim == 256, "Ornith-35B shape");
}

//
// models
//

static void test_qwen38() {
    reset_env();

    // the built-in plan names exactly its 16 attention layers and their geometry: kept, silently, bit-identical to today
    {
        const auto r = resolve("Qwen3.8-27B turbot", qwen38(T_TURBOT, T_TURBOT));
        TCHECK(is_pair(r, T_TURBOT, T_TURBOT), "turbot not kept: %s/%s", tn(r.type_k), tn(r.type_v));
        TCHECK(r.steps.empty(), "turbot kept with %zu steps", r.steps.size());
        TCHECK(g_plan_chosen > 0 && g_plan_kind == LLAMA_TURBOT_PLAN_KIND_BUILTIN, "Qwen plan: %s", kind_name(g_plan_kind));
        TCHECK(r.n_turbot_layers == 16, "Qwen turbot layers %u", r.n_turbot_layers);
    }
    // a synthetic plan that names exactly its layers
    {
        llama_kv_resolve_input in = qwen38(T_TURBOT, T_TURBOT);
        in.plan_check = plan_check_text(plan_text_for(attn_layers_of(in)));
        const auto r = resolve("Qwen3.8-27B turbot, synthetic plan", in);
        TCHECK(is_pair(r, T_TURBOT, T_TURBOT) && r.steps.empty(), "synthetic plan: %s/%s", tn(r.type_k), tn(r.type_v));
    }
    // np 4 without --kv-unified: 4 streams. [TAG_TURBOT_ANY_STREAMS] turbot, per-stream tiers; MULTI_STREAM=0: turbo5p
    {
        llama_kv_resolve_input in = qwen38(T_TURBOT, T_TURBOT);
        in.n_stream  = 4;
        in.n_seq_max = 4;
        in.kv_size   = 65536;
        const auto r = resolve("Qwen3.8-27B turbot, 4 streams", in);
        TCHECK(is_pair(r, T_TURBOT, T_TURBOT) && r.steps.empty(), "4 streams: %s/%s", tn(r.type_k), tn(r.type_v));

        set_env("LLAMA_TURBOT_MULTI_STREAM", "0");
        const auto r2 = resolve("Qwen3.8-27B turbot, 4 streams, MULTI_STREAM=0", in);
        TCHECK(is_pair(r2, T_5P, T_5P), "4 streams MULTI_STREAM=0: %s/%s", tn(r2.type_k), tn(r2.type_v));
        TCHECK(has_step(r2, 'B', T_TURBOT, T_5P, "single KV stream"), "4 streams MULTI_STREAM=0: step");
        reset_env();
    }
    // a plan for other layers
    {
        llama_kv_resolve_input in = qwen38(T_TURBOT, T_TURBOT);
        in.plan_check = plan_check_text(plan_text_for({ 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59 }));
        const auto r = resolve("Qwen3.8-27B turbot, plan misses layer 63", in);
        TCHECK(is_pair(r, T_5P, T_5P), "plan mismatch: %s/%s", tn(r.type_k), tn(r.type_v));
        TCHECK(has_step(r, 'B', T_TURBOT, T_5P, "63"), "plan mismatch: step");
    }
    // 'auto' asked for on Qwen: the automatic plan, not the built-in one
    {
        set_env("LLAMA_TURBOT_PLAN", "auto");
        const auto r = resolve("Qwen3.8-27B turbot, LLAMA_TURBOT_PLAN=auto", qwen38(T_TURBOT, T_TURBOT));
        TCHECK(is_pair(r, T_TURBOT, T_TURBOT) && r.steps.empty(), "auto on Qwen: %s/%s", tn(r.type_k), tn(r.type_v));
        TCHECK(g_plan_chosen > 0 && g_plan_kind == LLAMA_TURBOT_PLAN_KIND_AUTO, "auto on Qwen: plan %s", kind_name(g_plan_kind));
        reset_env();
    }
    // no plan at all
    {
        llama_kv_resolve_input in = qwen38(T_TURBOT, T_TURBOT);
        in.plan_check = nullptr;
        const auto r = llama_kv_resolve(in);
        TCHECK(is_pair(r, T_5P, T_5P), "no plan: %s/%s", tn(r.type_k), tn(r.type_v));
    }
    // an env variable the turbot cache rejects
    {
        llama_kv_resolve_input in = qwen38(T_TURBOT, T_TURBOT);
        in.env_refusal = "TURBO_INNERQ is incompatible";
        const auto r = llama_kv_resolve(in);
        TCHECK(is_pair(r, T_5P, T_5P) && has_step(r, 'B', T_TURBOT, T_5P, "TURBO_INNERQ"), "env refusal");
    }
    // one attention layer's KV on the CPU
    {
        llama_kv_resolve_input in = qwen38(T_TURBOT, T_TURBOT);
        for (auto & L : in.layers) {
            if (L.il == 31) {
                L.dev_refusal = "attention KV must be on a CUDA device (layer 31: CPU)";
            }
        }
        const auto r = resolve("Qwen3.8-27B turbot, layer 31 on CPU", in);
        TCHECK(is_pair(r, T_5P, T_5P) && has_step(r, 'B', T_TURBOT, T_5P, "layer 31"), "device refusal");
    }
    // the turbot constructor refused anyway (the llama_context fallback)
    {
        llama_kv_resolve_input in = qwen38(T_TURBOT, T_TURBOT);
        in.turbot_refused = "turbot: plan x line 3: layer 1 is not an attention layer of this cache";
        const auto r = llama_kv_resolve(in);
        TCHECK(is_pair(r, T_5P, T_5P), "constructor refusal: %s/%s", tn(r.type_k), tn(r.type_v));
        TCHECK(has_step(r, 'B', T_TURBOT, T_5P, "turbot: plan x line 3"), "constructor refusal: the prefix is said once");
        TCHECK(r.steps.size() == 1 && r.steps[0].reason.find("turbot: turbot") == std::string::npos, "double prefix");
    }
    // -ctk turbot with another -ctv: the turbot side takes turbo5p
    {
        const auto r = resolve("Qwen3.8-27B -ctk turbot -ctv turbo5p", qwen38(T_TURBOT, T_5P));
        TCHECK(is_pair(r, T_5P, T_5P), "one-sided turbot: %s/%s", tn(r.type_k), tn(r.type_v));
        TCHECK(has_step(r, 'K', T_TURBOT, T_5P, "together"), "one-sided turbot: step");
    }
    // turbo5p and q8_0 fit as they are
    for (ggml_type t : { T_5P, T_4P, T_4, T_Q8, T_F16 }) {
        const auto r = llama_kv_resolve(qwen38(t, t));
        TCHECK(is_pair(r, t, t) && r.steps.empty(), "Qwen3.8 %s changed to %s/%s", tn(t), tn(r.type_k), tn(r.type_v));
    }
    // a mixed pair with an FA kernel stays
    {
        const auto r = llama_kv_resolve(qwen38(T_4, T_Q8));
        TCHECK(is_pair(r, T_4, T_Q8) && r.steps.empty(), "turbo4/q8_0: %s/%s", tn(r.type_k), tn(r.type_v));
    }
    // a split-plane K against an f16 V: the FA dispatch refuses the pair, K leaves the turbo family
    {
        const auto r = resolve("Qwen3.8-27B -ctk turbo5p -ctv f16", qwen38(T_5P, T_F16));
        TCHECK(is_pair(r, T_Q8, T_F16), "turbo5p/f16: %s/%s", tn(r.type_k), tn(r.type_v));
        TCHECK(has_step(r, 'K', T_5P, T_Q8, "no flash-attention kernel"), "turbo5p/f16: step");
    }
    // flash attention off: turbot and turbo need it; K may stay q8_0, a quantized V may not
    {
        llama_kv_resolve_input in = qwen38(T_TURBOT, T_TURBOT);
        in.flash_attn = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        const auto r = resolve("Qwen3.8-27B turbot, -fa off", in);
        TCHECK(is_pair(r, T_Q8, T_F16), "fa off: %s/%s", tn(r.type_k), tn(r.type_v));
        TCHECK(has_step(r, 'K', T_TURBOT, T_Q8, "needs flash attention"), "fa off: K step");
        TCHECK(has_step(r, 'V', T_TURBOT, T_F16, "quantized V cache needs flash attention"), "fa off: V step");
    }
    {
        llama_kv_resolve_input in = qwen38(T_Q8, T_Q8);
        in.flash_attn = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        const auto r = llama_kv_resolve(in);
        TCHECK(is_pair(r, T_Q8, T_F16), "fa off q8_0: %s/%s", tn(r.type_k), tn(r.type_v));
    }
    {
        llama_kv_resolve_input in = qwen38(T_F16, T_F16);
        in.flash_attn = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        const auto r = llama_kv_resolve(in);
        TCHECK(is_pair(r, T_F16, T_F16) && r.steps.empty(), "fa off f16 changed");
    }
}

// [TAG_TURBOT_ANY_RESOLVE] Ornith-1.5-9B: the Qwen geometry on 8 layers, no built-in plan: the automatic plan
static void test_ornith_9b() {
    reset_env();

    const auto r = resolve("Ornith-1.5-9B turbot (8 layers, 4 x 256)", ornith_9b(T_TURBOT, T_TURBOT));
    TCHECK(is_pair(r, T_TURBOT, T_TURBOT) && r.steps.empty(), "Ornith-9B: %s/%s", tn(r.type_k), tn(r.type_v));
    TCHECK(g_plan_chosen > 0 && g_plan_kind == LLAMA_TURBOT_PLAN_KIND_AUTO, "Ornith-9B plan: %s", kind_name(g_plan_kind));

    set_env("LLAMA_TURBOT_AUTO_PLAN", "0");
    const auto r2 = resolve("Ornith-1.5-9B turbot, AUTO_PLAN=0", ornith_9b(T_TURBOT, T_TURBOT));
    TCHECK(is_pair(r2, T_5P, T_5P) && r2.steps.size() == 1, "Ornith-9B AUTO_PLAN=0: %s/%s", tn(r2.type_k), tn(r2.type_v));
    reset_env();

    // the MTP context holds only layer 32 and never gets an automatic plan
    const auto r3 = resolve("Ornith-1.5-9B turbot, MTP context", ornith_9b(T_TURBOT, T_TURBOT, true));
    TCHECK(is_pair(r3, T_5P, T_5P), "Ornith-9B MTP: %s/%s", tn(r3.type_k), tn(r3.type_v));
    TCHECK(g_plan_chosen == 0, "Ornith-9B MTP: a plan was chosen (%s)", kind_name(g_plan_kind));
}

// [TAG_TURBOT_ANY_RESOLVE] Ornith-1.5-35B: 2 x 256 (512-element rows). 256 x 2 left the validated list (G5 failed on
// this model, 2026-09-22): turbo5p512 by default, the automatic plan in the turbo5p512 budget with AUTO_PLAN=all
static void test_ornith_35b() {
    reset_env();

    const auto r0 = resolve("Ornith-1.5-35B turbot (10 layers, 2 x 256)", ornith_35b(T_TURBOT, T_TURBOT));
    TCHECK(is_pair(r0, T_5P512, T_5P512) && r0.steps.size() == 1, "Ornith-35B: %s/%s", tn(r0.type_k), tn(r0.type_v));
    TCHECK(has_step(r0, 'B', T_TURBOT, T_5P, "not validated for an automatic plan"), "Ornith-35B: the step names the validated list");

    set_env("LLAMA_TURBOT_AUTO_PLAN", "all");
    const auto r = resolve("Ornith-1.5-35B turbot, AUTO_PLAN=all", ornith_35b(T_TURBOT, T_TURBOT));
    TCHECK(is_pair(r, T_TURBOT, T_TURBOT) && r.steps.empty(), "Ornith-35B AUTO_PLAN=all: %s/%s", tn(r.type_k), tn(r.type_v));
    TCHECK(g_plan_chosen > 0 && g_plan_kind == LLAMA_TURBOT_PLAN_KIND_AUTO, "Ornith-35B plan: %s", kind_name(g_plan_kind));
    reset_env();

    set_env("LLAMA_TURBOT_ANY", "0");
    llama_kv_resolve_input in = ornith_35b(T_TURBOT, T_TURBOT);
    in.plan_check = plan_check_legacy();
    const auto r2 = resolve("Ornith-1.5-35B turbot, ANY=0", in);
    TCHECK(is_pair(r2, T_5P512, T_5P512), "Ornith-35B ANY=0: %s/%s", tn(r2.type_k), tn(r2.type_v));
    TCHECK(r2.steps.size() == 1 && has_step(r2, 'B', T_TURBOT, T_5P, "unsupported head geometry"), "Ornith-35B ANY=0: one step");
    reset_env();
}

// Spark-X2.5-4B: iSWA, 4 x 256. [TAG_TURBOT_ANY_ISWA] turbot on the 9 full-attention layers, turbo5p on the 27 SWA ones
static void test_spark_4b() {
    reset_env();

    {
        const auto r = resolve("Spark-X2.5-4B turbot (iSWA, 4 x 256)", spark_4b(T_TURBOT, T_TURBOT));
        TCHECK(is_pair(r, T_TURBOT, T_TURBOT, T_5P, T_5P), "Spark 4B: %s/%s SWA %s/%s",
               tn(r.type_k), tn(r.type_v), tn(r.type_k_swa), tn(r.type_v_swa));
        TCHECK(r.steps.empty() && r.steps_swa.empty() && r.swa_note.empty(), "Spark 4B: steps");
        TCHECK(r.n_turbot_layers == 9, "Spark 4B: %u turbot layers", r.n_turbot_layers);
    }
    // ISWA=0: the SWA refusal, as before
    {
        set_env("LLAMA_TURBOT_ISWA", "0");
        const auto r = resolve("Spark-X2.5-4B turbot, ISWA=0", spark_4b(T_TURBOT, T_TURBOT));
        TCHECK(is_pair(r, T_5P, T_5P), "Spark 4B ISWA=0: %s/%s SWA %s/%s", tn(r.type_k), tn(r.type_v), tn(r.type_k_swa), tn(r.type_v_swa));
        TCHECK(r.steps.size() == 1 && has_step(r, 'B', T_TURBOT, T_5P, "SWA caches are unsupported"), "Spark 4B ISWA=0: step");
        reset_env();
    }
    // SWA_TYPE: the A/B arm for the SWA layers
    {
        set_env("LLAMA_TURBOT_SWA_TYPE", "q8_0");
        const auto r = resolve("Spark-X2.5-4B turbot, SWA_TYPE=q8_0", spark_4b(T_TURBOT, T_TURBOT));
        TCHECK(is_pair(r, T_TURBOT, T_TURBOT, T_Q8, T_Q8), "Spark 4B SWA q8_0: SWA %s/%s", tn(r.type_k_swa), tn(r.type_v_swa));
        set_env("LLAMA_TURBOT_SWA_TYPE", "turbot");
        const auto r2 = resolve("Spark-X2.5-4B turbot, SWA_TYPE=turbot", spark_4b(T_TURBOT, T_TURBOT));
        TCHECK(is_pair(r2, T_TURBOT, T_TURBOT, T_5P, T_5P) && !r2.swa_note.empty(), "Spark 4B SWA turbot: SWA %s/%s",
               tn(r2.type_k_swa), tn(r2.type_v_swa));
        reset_env();
    }
    // a draft or MTP context: no automatic plan, so turbot falls back on both children and there is no split
    {
        llama_kv_resolve_input in = spark_4b(T_TURBOT, T_TURBOT);
        in.ctx_default = false;
        const auto r = resolve("Spark-X2.5-4B turbot, not the main context", in);
        TCHECK(is_pair(r, T_5P, T_5P), "Spark 4B draft: %s/%s", tn(r.type_k), tn(r.type_v));
    }
}

// Spark-X2.5-1.7B: 2 x 256 (512-element rows), iSWA
static void test_spark_1_7b() {
    reset_env();

    // 256 x 2 is not on the validated list: turbo5p512 by default, no iSWA split (the SWA layers take the same type)
    const auto r0 = resolve("Spark-X2.5-1.7B turbot (iSWA, 2 x 256)", spark_1_7b(T_TURBOT, T_TURBOT));
    TCHECK(is_pair(r0, T_5P512, T_5P512) && has_step(r0, 'B', T_TURBOT, T_5P, "not validated for an automatic plan"),
           "Spark 1.7B: %s/%s SWA %s/%s", tn(r0.type_k), tn(r0.type_v), tn(r0.type_k_swa), tn(r0.type_v_swa));

    // AUTO_PLAN=all: turbot on the full-attention layers; the SWA layers take turbo5p, which runs as turbo5p512 on these rows
    set_env("LLAMA_TURBOT_AUTO_PLAN", "all");
    const auto r = resolve("Spark-X2.5-1.7B turbot, AUTO_PLAN=all", spark_1_7b(T_TURBOT, T_TURBOT));
    TCHECK(is_pair(r, T_TURBOT, T_TURBOT, T_5P512, T_5P512), "Spark 1.7B: %s/%s SWA %s/%s",
           tn(r.type_k), tn(r.type_v), tn(r.type_k_swa), tn(r.type_v_swa));
    TCHECK(r.steps_swa.empty(), "Spark 1.7B: the 512 swap is not a downgrade");
    reset_env();

    // turbo5p asked directly: the same swap, no warning
    const auto r2 = llama_kv_resolve(spark_1_7b(T_5P, T_5P));
    TCHECK(is_pair(r2, T_5P512, T_5P512) && r2.steps.empty(), "Spark 1.7B turbo5p: %s/%s", tn(r2.type_k), tn(r2.type_v));

    // turbo4p needs 1024-element rows: turbo4
    const auto r3 = llama_kv_resolve(spark_1_7b(T_4P, T_4P));
    TCHECK(is_pair(r3, T_4, T_4), "Spark 1.7B turbo4p: %s/%s", tn(r3.type_k), tn(r3.type_v));

    // a downgrade that would leave turbo4 against turbo5p512 (no FA kernel): both take turbo4
    const auto r4 = resolve("Spark-X2.5-1.7B -ctk turbo4p -ctv turbo5p", spark_1_7b(T_4P, T_5P));
    TCHECK(is_pair(r4, T_4, T_4), "Spark 1.7B turbo4p/turbo5p: %s/%s", tn(r4.type_k), tn(r4.type_v));

    // SWA_TYPE=turbo4p on 512-element rows: the SWA layers step down to turbo4 with a warning
    set_env("LLAMA_TURBOT_AUTO_PLAN", "all");
    set_env("LLAMA_TURBOT_SWA_TYPE", "turbo4p");
    const auto r5 = resolve("Spark-X2.5-1.7B turbot, SWA_TYPE=turbo4p", spark_1_7b(T_TURBOT, T_TURBOT));
    TCHECK(is_pair(r5, T_TURBOT, T_TURBOT, T_4, T_4) && has_step(r5.steps_swa, 'B', T_4P, T_4, "1024"),
           "Spark 1.7B SWA turbo4p: SWA %s/%s", tn(r5.type_k_swa), tn(r5.type_v_swa));
    reset_env();
}

// MiniCPM5-2B: 2 x 128 (256-element rows). An NR=1 shape: the automatic plan only with AUTO_BUDGET=turbo5p
static void test_minicpm5() {
    reset_env();

    const auto r = resolve("MiniCPM5 turbot (2 x 128)", minicpm5(T_TURBOT, T_TURBOT));
    TCHECK(is_pair(r, T_4, T_4), "MiniCPM5: %s/%s", tn(r.type_k), tn(r.type_v));
    TCHECK(r.steps.size() == 1 && has_step(r, 'B', T_TURBOT, T_4, "turbo5p: layer 0: KV row 256"), "MiniCPM5: one step");

    set_env("LLAMA_TURBOT_AUTO_BUDGET", "turbo5p");
    const auto r2 = resolve("MiniCPM5 turbot, AUTO_BUDGET=turbo5p", minicpm5(T_TURBOT, T_TURBOT));
    TCHECK(is_pair(r2, T_TURBOT, T_TURBOT) && r2.steps.empty(), "MiniCPM5 AUTO_BUDGET=turbo5p: %s/%s", tn(r2.type_k), tn(r2.type_v));
    TCHECK(g_plan_chosen > 0 && g_plan_kind == LLAMA_TURBOT_PLAN_KIND_AUTO, "MiniCPM5 AUTO_BUDGET plan: %s", kind_name(g_plan_kind));
    reset_env();

    const auto r3 = llama_kv_resolve(minicpm5(T_5P, T_5P));
    TCHECK(is_pair(r3, T_4, T_4), "MiniCPM5 turbo5p: %s/%s", tn(r3.type_k), tn(r3.type_v));

    // mixed turbo4p/turbo4: turbo4p cannot hold 256-element rows, the pair ends matched
    const auto r4 = llama_kv_resolve(minicpm5(T_4P, T_4));
    TCHECK(is_pair(r4, T_4, T_4), "MiniCPM5 turbo4p/turbo4: %s/%s", tn(r4.type_k), tn(r4.type_v));
}

// Muse Glimmer 30B: iSWA, 2 x 128. No automatic plan for NR=1: turbo4 on both children
static void test_muse_glimmer() {
    reset_env();

    const auto r = resolve("Muse Glimmer turbot (iSWA, 2 x 128)", muse_glimmer(T_TURBOT, T_TURBOT));
    TCHECK(is_pair(r, T_4, T_4), "Muse Glimmer: %s/%s SWA %s/%s", tn(r.type_k), tn(r.type_v), tn(r.type_k_swa), tn(r.type_v_swa));

    // the opt-in: turbot on the 13 full-attention layers; the SWA layers ask for turbo5p and step down to turbo4 (256-value rows)
    set_env("LLAMA_TURBOT_AUTO_BUDGET", "turbo5p");
    const auto r2 = resolve("Muse Glimmer turbot, AUTO_BUDGET=turbo5p", muse_glimmer(T_TURBOT, T_TURBOT));
    TCHECK(is_pair(r2, T_TURBOT, T_TURBOT, T_4, T_4), "Muse Glimmer AUTO_BUDGET: %s/%s SWA %s/%s",
           tn(r2.type_k), tn(r2.type_v), tn(r2.type_k_swa), tn(r2.type_v_swa));
    TCHECK(r2.n_turbot_layers == 13 && has_step(r2.steps_swa, 'B', T_5P, T_4, "KV row 256"), "Muse Glimmer AUTO_BUDGET: SWA step");
    reset_env();
}

// Nemotron-3.5-Lightning-30B-A3B: hybrid, 6 attention layers of 2 x 128
static void test_nemotron_h() {
    reset_env();

    const auto r = resolve("Nemotron-H turbot (6 layers, 2 x 128)", nemotron_h(T_TURBOT, T_TURBOT));
    TCHECK(is_pair(r, T_4, T_4), "Nemotron-H: %s/%s", tn(r.type_k), tn(r.type_v));
}

// Granite 4.2 8B: 8 x 128 (1024-element rows). The automatic plan only with AUTO_PLAN=all
static void test_granite() {
    reset_env();

    const auto r = resolve("Granite turbot (8 x 128)", granite(T_TURBOT, T_TURBOT));
    TCHECK(is_pair(r, T_5P, T_5P), "Granite: %s/%s", tn(r.type_k), tn(r.type_v));

    set_env("LLAMA_TURBOT_AUTO_PLAN", "all");
    const auto r2 = resolve("Granite turbot, AUTO_PLAN=all", granite(T_TURBOT, T_TURBOT));
    TCHECK(is_pair(r2, T_TURBOT, T_TURBOT) && r2.steps.empty(), "Granite AUTO_PLAN=all: %s/%s", tn(r2.type_k), tn(r2.type_v));
    reset_env();
}

// [TAG_TURBOT_ANY_ISWA] an all-SWA model: an iSWA cache whose base child holds no layer
static void test_all_swa() {
    reset_env();

    const auto r = resolve("all-SWA model turbot (4 x 256)", make_iswa(T_TURBOT, T_TURBOT, 24, 4, 256, [](uint32_t) { return false; }));
    TCHECK(is_pair(r, T_5P, T_5P), "all-SWA: %s/%s SWA %s/%s", tn(r.type_k), tn(r.type_v), tn(r.type_k_swa), tn(r.type_v_swa));
    TCHECK(has_step(r, 'B', T_TURBOT, T_5P, "no full-attention layers"), "all-SWA: step");
}

// [TAG_TURBOT_ANY_RESOLVE] a plan file is used alone: when it does not fit, no automatic plan replaces it
static void test_plan_file() {
    reset_env();

    const char * path = "test-kv-resolve-mismatch.plan";
    FILE * f = fopen(path, "wb");
    if (f == nullptr) {
        fprintf(stderr, "  (cannot write %s here: the plan-file check uses a missing file only)\n", path);
    } else {
        const std::string text = llama_turbot_default_plan_text();   // names Qwen3.8-27B's 16 layers
        const size_t n_written = fwrite(text.data(), 1, text.size(), f);
        fclose(f);
        TCHECK(n_written == text.size(), "short write of %s", path);

        set_env("LLAMA_TURBOT_PLAN", path);
        const auto r = resolve("Ornith-1.5-9B turbot, mismatching plan file", ornith_9b(T_TURBOT, T_TURBOT));
        TCHECK(is_pair(r, T_5P, T_5P), "mismatching file: %s/%s", tn(r.type_k), tn(r.type_v));
        TCHECK(g_plan_chosen == 0, "mismatching file: a plan was chosen (%s)", kind_name(g_plan_kind));
        reset_env();
        remove(path);
    }

    set_env("LLAMA_TURBOT_PLAN", "no-such-dir/no-such.plan");
    const auto r2 = resolve("Ornith-1.5-9B turbot, missing plan file", ornith_9b(T_TURBOT, T_TURBOT));
    TCHECK(is_pair(r2, T_5P, T_5P), "missing file: %s/%s", tn(r2.type_k), tn(r2.type_v));
    TCHECK(g_plan_chosen == 0, "missing file: a plan was chosen (%s)", kind_name(g_plan_kind));
    reset_env();

    // 'default' forces the built-in plan, which does not fit Ornith: no automatic plan either
    set_env("LLAMA_TURBOT_PLAN", "default");
    const auto r3 = resolve("Ornith-1.5-9B turbot, LLAMA_TURBOT_PLAN=default", ornith_9b(T_TURBOT, T_TURBOT));
    TCHECK(is_pair(r3, T_5P, T_5P) && g_plan_chosen == 0, "default on Ornith: %s/%s", tn(r3.type_k), tn(r3.type_v));
    reset_env();
}

// head 64 (8 KV heads x 64): no turbo FA kernel takes D = 64 without zero-padding
static void test_head64() {
    reset_env();

    const auto r = resolve("head-64 model turbo5p (8 x 64)", make_model(T_5P, T_5P, 24, 8, 64));
    TCHECK(is_pair(r, T_Q8, T_Q8), "head 64: %s/%s", tn(r.type_k), tn(r.type_v));
    TCHECK(r.steps.size() == 1 && has_step(r, 'B', T_5P, T_Q8, "head size 64"), "head 64: step");

    const auto r2 = llama_kv_resolve(make_model(T_TURBOT, T_TURBOT, 24, 8, 64));
    TCHECK(is_pair(r2, T_Q8, T_Q8), "head 64 turbot: %s/%s", tn(r2.type_k), tn(r2.type_v));

    // [TAG_KV_RESOLVE] turbo4 asked for: the zero-padding path (64 -> 128) is kept as before the resolver
    const auto r3 = llama_kv_resolve(make_model(T_4, T_4, 24, 8, 64));
    TCHECK(is_pair(r3, T_4, T_4) && r3.steps.empty(), "head 64 turbo4: %s/%s", tn(r3.type_k), tn(r3.type_v));

    // turbo5p K cannot hold 64-element heads; the asked turbo4 V is taken for both
    const auto r4 = llama_kv_resolve(make_model(T_5P, T_4, 24, 8, 64));
    TCHECK(is_pair(r4, T_4, T_4), "head 64 turbo5p/turbo4: %s/%s", tn(r4.type_k), tn(r4.type_v));
}

// [TAG_KV_RESOLVE] shapes the turbo2/3/4 zero-padding path serves when asked for, and never takes as a fallback
static void test_turbo_padding() {
    reset_env();

    // Gemma-4-like iSWA: 256-element SWA heads, 512-element global heads (F16-converted FA path)
    const auto gemma4 = [](ggml_type tk, ggml_type tv) {
        llama_kv_resolve_input in = make_iswa(tk, tv, 30, 8, 256, [](uint32_t il) { return il % 6 == 5; });
        for (auto & L : in.layers) {
            if (!L.swa) {
                L.head_k    = 512;
                L.head_v    = 512;
                L.n_head_kv = 2;
            }
        }
        return in;
    };
    const auto r = llama_kv_resolve(gemma4(T_4, T_4));
    TCHECK(is_pair(r, T_4, T_4) && r.steps.empty(), "gemma4 turbo4: %s/%s", tn(r.type_k), tn(r.type_v));
    const auto r2 = resolve("Gemma-4-like turbot (256 / 512 heads)", gemma4(T_TURBOT, T_TURBOT));
    TCHECK(is_pair(r2, T_Q8, T_Q8), "gemma4 turbot: %s/%s", tn(r2.type_k), tn(r2.type_v));

    // K 192 / V 128 (no MLA): the padded K 256 has no FA kernel against V 128
    llama_kv_resolve_input in = make_model(T_4, T_4, 27, 16, 192);
    for (auto & L : in.layers) {
        L.head_v = 128;
    }
    const auto r3 = resolve("K 192 / V 128 turbo4", in);
    TCHECK(is_pair(r3, T_Q8, T_Q8) && has_step(r3, 'B', T_4, T_Q8, "no turbo flash-attention kernel"),
           "k192/v128 turbo4: %s/%s", tn(r3.type_k), tn(r3.type_v));

    // head 80: turbo4 asked is padded to 128 and kept; turbot never lands on a padded type (q8_0 does not fit 80 either)
    const auto r4 = llama_kv_resolve(make_model(T_4, T_4, 32, 8, 80));
    TCHECK(is_pair(r4, T_4, T_4) && r4.steps.empty(), "head 80 turbo4: %s/%s", tn(r4.type_k), tn(r4.type_v));
    const auto r5 = llama_kv_resolve(make_model(T_TURBOT, T_TURBOT, 32, 8, 80));
    TCHECK(is_pair(r5, T_F16, T_F16), "head 80 turbot: %s/%s", tn(r5.type_k), tn(r5.type_v));
}

// head 80: not a whole q8_0 block
static void test_head80() {
    reset_env();

    const auto r = resolve("head-80 model q8_0 (8 x 80)", make_model(T_Q8, T_Q8, 32, 8, 80));
    TCHECK(is_pair(r, T_F16, T_F16), "head 80: %s/%s", tn(r.type_k), tn(r.type_v));
}

// MLA (DeepSeek-V2 / Ling-3.0-tiny style): one latent KV head, K 576, V 512 read from the K cache
static llama_kv_resolve_input mla_model(ggml_type tk, ggml_type tv, llama_flash_attn_type fa) {
    llama_kv_resolve_input in = make_model(tk, tv, 27, 1, 576);
    for (auto & L : in.layers) {
        L.head_v = 512;
    }
    in.mla        = true;
    in.same_type  = true;
    in.flash_attn = fa;
    return in;
}

static void test_mla() {
    reset_env();

    const auto r = resolve("MLA turbot", mla_model(T_TURBOT, T_TURBOT, LLAMA_FLASH_ATTN_TYPE_AUTO));
    TCHECK(is_pair(r, T_Q8, T_Q8), "MLA: %s/%s", tn(r.type_k), tn(r.type_v));

    const auto r2 = llama_kv_resolve(mla_model(T_5P, T_5P, LLAMA_FLASH_ATTN_TYPE_ENABLED));
    TCHECK(is_pair(r2, T_Q8, T_Q8) && has_step(r2, 'B', T_5P, T_Q8, "MLA"), "MLA turbo5p: %s/%s", tn(r2.type_k), tn(r2.type_v));

    const auto r3 = resolve("MLA turbo5p, -fa off", mla_model(T_5P, T_5P, LLAMA_FLASH_ATTN_TYPE_DISABLED));
    TCHECK(is_pair(r3, T_F16, T_F16), "MLA fa off: %s/%s", tn(r3.type_k), tn(r3.type_v));

    // K and V differ: V follows K
    const auto r4 = llama_kv_resolve(mla_model(T_Q8, T_F16, LLAMA_FLASH_ATTN_TYPE_AUTO));
    TCHECK(is_pair(r4, T_Q8, T_Q8) && has_step(r4, 'V', T_F16, T_Q8, nullptr), "MLA q8_0/f16: %s/%s", tn(r4.type_k), tn(r4.type_v));

    // [TAG_KV_RESOLVE] turbo3 asked for on MLA (GLM-4.7-Flash): K 576 zero-padded to 640, V 512, fattn.cu case 640
    const auto r5 = resolve("MLA turbo3 (asked)", mla_model(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0, LLAMA_FLASH_ATTN_TYPE_AUTO));
    TCHECK(is_pair(r5, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0) && r5.steps.empty(), "MLA turbo3: %s/%s", tn(r5.type_k), tn(r5.type_v));
    const auto r6 = llama_kv_resolve(mla_model(T_4, T_4, LLAMA_FLASH_ATTN_TYPE_DISABLED));
    TCHECK(is_pair(r6, T_F16, T_F16), "MLA turbo4 fa off: %s/%s", tn(r6.type_k), tn(r6.type_v));
}

// DeepSeek V4 style: no turbo query rotation in its attention input, one type for K and V
static void test_no_turbo_graph() {
    reset_env();

    llama_kv_resolve_input in = make_model(T_4, T_4, 16, 1, 512);
    in.turbo_graph = false;
    in.same_type   = true;
    in.swa         = true;
    const auto r = llama_kv_resolve(in);
    TCHECK(is_pair(r, T_Q8, T_Q8), "no turbo graph: %s/%s", tn(r.type_k), tn(r.type_v));
}

// recurrent-only (Mamba): no KV cache, nothing to decide or warn about
static void test_recurrent() {
    reset_env();

    llama_kv_resolve_input in = make_model(T_TURBOT, T_TURBOT, 48, 0, 0);
    in.no_kv = true;
    const auto r = resolve("recurrent-only turbot", in);
    TCHECK(r.no_kv && r.steps.empty(), "recurrent: no_kv / steps");
    TCHECK(is_pair(r, T_F16, T_F16), "recurrent: %s/%s", tn(r.type_k), tn(r.type_v));

    llama_kv_resolve_input in2 = make_model(T_F16, T_F16, 48, 0, 0);
    in2.no_kv = true;
    const auto r2 = llama_kv_resolve(in2);
    TCHECK(is_pair(r2, T_F16, T_F16) && r2.steps.empty(), "recurrent f16");
}

// the built-in default plan names Qwen3.8-27B's layers only
static void test_builtin_plan() {
    reset_env();

    std::string why;
    const std::vector<int32_t> qwen = { 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63 };
    TCHECK(llama_turbot_plan_matches(llama_turbot_default_plan_text(), qwen, 262144, why), "built-in plan vs Qwen3.8: %s", why.c_str());
    TCHECK(llama_turbot_default_plan_hash() == 0x56c3503c949a7749ull, "built-in plan hash 0x%016llx",
           (unsigned long long) llama_turbot_default_plan_hash());

    std::vector<int32_t> spark;
    for (int32_t il = 3; il < 36; il += 4) {
        spark.push_back(il);
    }
    TCHECK(!llama_turbot_plan_matches(llama_turbot_default_plan_text(), spark, 262144, why), "built-in plan fits Spark");
    TCHECK(why.find("layer 39 is not an attention layer") != std::string::npos, "built-in plan vs Spark: %s", why.c_str());
}

// [TAG_TURBOT_ANY_RESOLVE] LLAMA_TURBOT_ANY=0: every result of the README table ("What -ctk turbot -ctv turbot resolves
// to"), with today's plan check (llama_context uses it under ANY=0) and no iSWA split
static void test_any_off() {
    reset_env();
    set_env("LLAMA_TURBOT_ANY", "0");

    const auto legacy = [](llama_kv_resolve_input in) {
        in.plan_check = plan_check_legacy();
        return in;
    };

    {
        const auto r = resolve("ANY=0 Qwen3.8-27B", legacy(qwen38(T_TURBOT, T_TURBOT)));
        TCHECK(is_pair(r, T_TURBOT, T_TURBOT) && r.steps.empty(), "ANY=0 Qwen: %s/%s", tn(r.type_k), tn(r.type_v));
    }
    {
        llama_kv_resolve_input in = legacy(qwen38(T_TURBOT, T_TURBOT));
        in.n_stream  = 4;
        in.n_seq_max = 4;
        in.kv_size   = 65536;
        const auto r = resolve("ANY=0 Qwen3.8-27B, 4 streams", in);
        TCHECK(is_pair(r, T_5P, T_5P) && has_step(r, 'B', T_TURBOT, T_5P, "single KV stream"), "ANY=0 Qwen np 4: %s/%s", tn(r.type_k), tn(r.type_v));
    }
    {
        const auto r = resolve("ANY=0 Spark-X2.5-4B", legacy(spark_4b(T_TURBOT, T_TURBOT)));
        TCHECK(is_pair(r, T_5P, T_5P) && has_step(r, 'B', T_TURBOT, T_5P, "SWA caches are unsupported"),
               "ANY=0 Spark 4B: %s/%s SWA %s/%s", tn(r.type_k), tn(r.type_v), tn(r.type_k_swa), tn(r.type_v_swa));
    }
    {
        const auto r = resolve("ANY=0 Spark-X2.5-1.7B", legacy(spark_1_7b(T_TURBOT, T_TURBOT)));
        TCHECK(is_pair(r, T_5P512, T_5P512) && r.steps.size() == 1, "ANY=0 Spark 1.7B: %s/%s", tn(r.type_k), tn(r.type_v));
    }
    {
        const auto r = resolve("ANY=0 Granite", legacy(granite(T_TURBOT, T_TURBOT)));
        TCHECK(is_pair(r, T_5P, T_5P) && has_step(r, 'B', T_TURBOT, T_5P, "unsupported head geometry"), "ANY=0 Granite: %s/%s", tn(r.type_k), tn(r.type_v));
    }
    {
        const auto r = resolve("ANY=0 MiniCPM5", legacy(minicpm5(T_TURBOT, T_TURBOT)));
        TCHECK(is_pair(r, T_4, T_4) && r.steps.size() == 1 && has_step(r, 'B', T_TURBOT, T_4, "unsupported head geometry"),
               "ANY=0 MiniCPM5: %s/%s", tn(r.type_k), tn(r.type_v));
    }
    {
        const auto r = resolve("ANY=0 Muse Glimmer", legacy(muse_glimmer(T_TURBOT, T_TURBOT)));
        TCHECK(is_pair(r, T_4, T_4), "ANY=0 Muse: %s/%s", tn(r.type_k), tn(r.type_v));
    }
    {
        const auto r = llama_kv_resolve(legacy(make_model(T_TURBOT, T_TURBOT, 24, 8, 64)));
        TCHECK(is_pair(r, T_Q8, T_Q8), "ANY=0 head 64: %s/%s", tn(r.type_k), tn(r.type_v));
    }
    {
        const auto r = llama_kv_resolve(legacy(make_model(T_TURBOT, T_TURBOT, 32, 8, 80)));
        TCHECK(is_pair(r, T_F16, T_F16), "ANY=0 head 80: %s/%s", tn(r.type_k), tn(r.type_v));
    }
    {
        const auto r = llama_kv_resolve(legacy(mla_model(T_TURBOT, T_TURBOT, LLAMA_FLASH_ATTN_TYPE_AUTO)));
        TCHECK(is_pair(r, T_Q8, T_Q8), "ANY=0 MLA: %s/%s", tn(r.type_k), tn(r.type_v));
    }
    {
        llama_kv_resolve_input in = legacy(make_model(T_TURBOT, T_TURBOT, 48, 0, 0));
        in.no_kv = true;
        const auto r = llama_kv_resolve(in);
        TCHECK(r.no_kv && r.steps.empty() && is_pair(r, T_F16, T_F16), "ANY=0 recurrent");
    }
    // the new shapes fall back as before
    {
        const auto r = llama_kv_resolve(legacy(ornith_9b(T_TURBOT, T_TURBOT)));
        TCHECK(is_pair(r, T_5P, T_5P), "ANY=0 Ornith-9B: %s/%s", tn(r.type_k), tn(r.type_v));
        const auto r2 = llama_kv_resolve(legacy(nemotron_h(T_TURBOT, T_TURBOT)));
        TCHECK(is_pair(r2, T_4, T_4), "ANY=0 Nemotron-H: %s/%s", tn(r2.type_k), tn(r2.type_v));
    }
    // with ANY=0 every other switch reads off: ISWA, MULTI_STREAM, AUTO_BUDGET set to "on" values change nothing
    {
        set_env("LLAMA_TURBOT_ISWA", "1");
        set_env("LLAMA_TURBOT_MULTI_STREAM", "1");
        set_env("LLAMA_TURBOT_AUTO_BUDGET", "turbo5p");
        set_env("LLAMA_TURBOT_AUTO_PLAN", "all");
        const auto r = llama_kv_resolve(legacy(spark_4b(T_TURBOT, T_TURBOT)));
        TCHECK(is_pair(r, T_5P, T_5P), "ANY=0 + switches, Spark: %s/%s", tn(r.type_k), tn(r.type_v));
        const auto r2 = llama_kv_resolve(legacy(minicpm5(T_TURBOT, T_TURBOT)));
        TCHECK(is_pair(r2, T_4, T_4), "ANY=0 + switches, MiniCPM5: %s/%s", tn(r2.type_k), tn(r2.type_v));
        const auto r3 = llama_kv_resolve(legacy(granite(T_TURBOT, T_TURBOT)));
        TCHECK(is_pair(r3, T_5P, T_5P), "ANY=0 + switches, Granite: %s/%s", tn(r3.type_k), tn(r3.type_v));
    }

    reset_env();
}

// LLAMA_KV_RESOLVE=0 turns the resolver off in llama_context only: llama_kv_resolve and the constructor's own checks
// (the shared helpers) do not read it, so the cache constructor still refuses exactly what it refused before
static void test_kv_resolve_off() {
    reset_env();
    set_env("LLAMA_KV_RESOLVE", "0");

    const auto r = llama_kv_resolve(qwen38(T_TURBOT, T_TURBOT));
    TCHECK(is_pair(r, T_TURBOT, T_TURBOT) && r.steps.empty(), "KV_RESOLVE=0 Qwen: %s/%s", tn(r.type_k), tn(r.type_v));

    llama_turbot_cache_desc d;
    d.k_turbot = true;
    d.v_turbot = true;
    d.kv_size  = 262144;
    d.swa      = true;
    TCHECK(llama_turbot_cache_refusal(d) == "SWA caches are unsupported", "KV_RESOLVE=0: SWA refusal");
    TCHECK(llama_turbot_layer_refusal(3, 256, 256, 4).empty(), "KV_RESOLVE=0: Qwen geometry refused");

    reset_env();
}

int main() {
    fprintf(stderr, "test-kv-resolve:\n");

    test_turbot_helpers();
    test_shape();
    test_builtin_plan();
    test_qwen38();
    test_ornith_9b();
    test_ornith_35b();
    test_spark_4b();
    test_spark_1_7b();
    test_minicpm5();
    test_muse_glimmer();
    test_nemotron_h();
    test_granite();
    test_all_swa();
    test_plan_file();
    test_head64();
    test_turbo_padding();
    test_head80();
    test_mla();
    test_no_turbo_graph();
    test_recurrent();
    test_any_off();
    test_kv_resolve_off();

    reset_env();

    fprintf(stderr, "%d checks, %d failed\n", g_checks, g_fail);

    return g_fail == 0 ? 0 : 1;
}
