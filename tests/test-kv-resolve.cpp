// [TAG_KV_RESOLVE] CPU unit test of the KV cache type resolver (src/llama-kv-cache-resolve.h).
//
// Drives the pure decision function llama_kv_resolve() with synthetic hparams of real model shapes, and the turbot
// precondition helpers the cache constructor shares with it. The plan check uses the real plan parser
// (src/llama-kv-tier.h) on a synthetic plan and on the built-in default plan. No GPU, no model.
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

static const char * tn(ggml_type t) {
    return llama_kv_resolve_type_name(t);
}

// a plan with an L line for each layer in layers (5-bit old widths, the default young width)
static std::string plan_text_for(const std::vector<int32_t> & layers) {
    std::string s = "# synthetic test plan\nPOOL 65536\nCAP 16384\n";
    for (const int32_t il : layers) {
        s += "L " + std::to_string(il) + " K 5 5 5 5 V 5 5 5 5\n";
    }
    return s;
}

static std::function<bool(const std::vector<int32_t> &, uint32_t, std::string &)> plan_check_text(const std::string & text) {
    return [text](const std::vector<int32_t> & attn_layers, uint32_t kv_size, std::string & why) {
        llama_turbot_plan plan;
        return llama_turbot_plan_parse_text(text, attn_layers, kv_size, plan, why);
    };
}

// a model of n_layer layers, all with n_head_kv heads of size head (K and V); attn(il) marks the attention cache layers
static llama_kv_resolve_input make_model(ggml_type tk, ggml_type tv, uint32_t n_layer, uint32_t n_head_kv, uint32_t head,
        const std::function<bool(uint32_t)> & attn = nullptr) {
    llama_kv_resolve_input in;
    in.type_k     = tk;
    in.type_v     = tv;
    in.flash_attn = LLAMA_FLASH_ATTN_TYPE_AUTO;
    in.n_stream   = 1;
    in.kv_size    = 262144;
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
    llama_kv_resolve_input in = make_model(tk, tv, 65, 4, 256, [](uint32_t il) { return il < 64 && (il + 1) % 4 == 0; });
    in.plan_check = plan_check_text(plan_text_for(attn_layers_of(in)));
    return in;
}

static bool has_step(const llama_kv_resolve_result & r, char side, ggml_type from, ggml_type to, const char * reason_part) {
    for (const auto & st : r.steps) {
        if (st.side == side && st.from == from && st.to == to &&
            (reason_part == nullptr || st.reason.find(reason_part) != std::string::npos)) {
            return true;
        }
    }
    return false;
}

static void dump(const char * what, const llama_kv_resolve_result & r) {
    fprintf(stderr, "  %-44s -> K %-10s V %-10s", what, tn(r.type_k), tn(r.type_v));
    if (r.no_kv) {
        fprintf(stderr, " (no KV)");
    }
    fprintf(stderr, "\n");
    for (const auto & st : r.steps) {
        fprintf(stderr, "      step %c: %s -> %s (%s)\n", st.side, tn(st.from), tn(st.to), st.reason.c_str());
    }
}

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

//
// turbot preconditions shared with the cache constructor
//

static void test_turbot_helpers() {
    llama_turbot_cache_desc d;
    d.k_turbot = true;
    d.v_turbot = true;
    d.kv_size  = 262144;
    TCHECK(llama_turbot_cache_refusal(d).empty(), "a valid cache refused: %s", llama_turbot_cache_refusal(d).c_str());

    llama_turbot_cache_desc e = d;
    e.v_turbot = false;
    TCHECK(llama_turbot_cache_refusal(e) == "needs -ctk turbot and -ctv turbot together", "pairing");
    e = d; e.n_stream = 4;
    TCHECK(llama_turbot_cache_refusal(e) == "needs a single KV stream: use --kv-unified or -np 1", "streams");
    e = d; e.v_trans = true;
    TCHECK(llama_turbot_cache_refusal(e) == "needs flash attention", "flash attention");
    e = d; e.mla = true;
    TCHECK(llama_turbot_cache_refusal(e) == "shared cells and MLA caches are unsupported", "mla");
    e = d; e.shared_cells = true;
    TCHECK(llama_turbot_cache_refusal(e) == "shared cells and MLA caches are unsupported", "shared cells");
    e = d; e.swa = true;
    TCHECK(llama_turbot_cache_refusal(e) == "SWA caches are unsupported", "swa");
    e = d; e.kv_size = 1000;
    TCHECK(llama_turbot_cache_refusal(e) == "kv_size must be a multiple of 64, got 1000", "kv_size: %s", llama_turbot_cache_refusal(e).c_str());

    TCHECK(llama_turbot_layer_refusal(3, 256, 256, 4).empty(), "4 x 256 refused");
    TCHECK(llama_turbot_layer_refusal(3, 128, 128, 2) ==
           "unsupported head geometry on layer 3: head_k 128, head_v 128, n_head_kv 2 (needs 256, 256, 4)",
           "geometry: %s", llama_turbot_layer_refusal(3, 128, 128, 2).c_str());

    TCHECK(llama_turbot_layer_device_refusal(7, nullptr) == "attention KV must be on a CUDA device (layer 7: KV offload is off)",
           "device: %s", llama_turbot_layer_device_refusal(7, nullptr).c_str());

    // the env variables the constructor rejects
    set_env("TURBO_KV_CPU_LAYERS", nullptr);
    set_env("TURBO_LAYER_ADAPTIVE", nullptr);
    set_env("TURBO_INNERQ", nullptr);
    TCHECK(llama_turbot_env_refusal().empty(), "clean env refused: %s", llama_turbot_env_refusal().c_str());
    set_env("TURBO_INNERQ", "0");
    TCHECK(llama_turbot_env_refusal().empty(), "TURBO_INNERQ=0 refused");
    set_env("TURBO_INNERQ", "1");
    TCHECK(llama_turbot_env_refusal() == "TURBO_INNERQ is incompatible", "TURBO_INNERQ=1: %s", llama_turbot_env_refusal().c_str());
    set_env("TURBO_INNERQ", nullptr);
    set_env("TURBO_KV_CPU_LAYERS", "2");
    TCHECK(llama_turbot_env_refusal().rfind("TURBO_KV_CPU_LAYERS", 0) == 0, "TURBO_KV_CPU_LAYERS=2");
    set_env("TURBO_KV_CPU_LAYERS", nullptr);
}

//
// models
//

static void test_qwen38() {
    // turbot with a plan that names exactly its 16 attention layers: kept, silently
    {
        const auto r = llama_kv_resolve(qwen38(T_TURBOT, T_TURBOT));
        dump("Qwen3.8-27B turbot, plan fits", r);
        TCHECK(r.type_k == T_TURBOT && r.type_v == T_TURBOT, "turbot not kept: %s/%s", tn(r.type_k), tn(r.type_v));
        TCHECK(r.steps.empty(), "turbot kept with %zu steps", r.steps.size());
    }
    // the built-in default plan is the one calibrated on this model: it fits the same 16 layers
    {
        llama_kv_resolve_input in = qwen38(T_TURBOT, T_TURBOT);
        in.plan_check = [](const std::vector<int32_t> & attn_layers, uint32_t kv_size, std::string & why) {
            return llama_turbot_plan_matches(llama_turbot_default_plan_text(), attn_layers, kv_size, why);
        };
        const auto r = llama_kv_resolve(in);
        dump("Qwen3.8-27B turbot, built-in plan", r);
        TCHECK(r.type_k == T_TURBOT && r.type_v == T_TURBOT && r.steps.empty(), "built-in plan: %s/%s", tn(r.type_k), tn(r.type_v));
    }
    // np 4 without --kv-unified: 4 streams
    {
        llama_kv_resolve_input in = qwen38(T_TURBOT, T_TURBOT);
        in.n_stream = 4;
        const auto r = llama_kv_resolve(in);
        dump("Qwen3.8-27B turbot, 4 streams", r);
        TCHECK(r.type_k == T_5P && r.type_v == T_5P, "4 streams: %s/%s", tn(r.type_k), tn(r.type_v));
        TCHECK(has_step(r, 'B', T_TURBOT, T_5P, "single KV stream"), "4 streams: step");
    }
    // a plan for other layers
    {
        llama_kv_resolve_input in = qwen38(T_TURBOT, T_TURBOT);
        in.plan_check = plan_check_text(plan_text_for({ 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59 }));
        const auto r = llama_kv_resolve(in);
        dump("Qwen3.8-27B turbot, plan misses layer 63", r);
        TCHECK(r.type_k == T_5P && r.type_v == T_5P, "plan mismatch: %s/%s", tn(r.type_k), tn(r.type_v));
        TCHECK(has_step(r, 'B', T_TURBOT, T_5P, "missing L line for attention layer 63"), "plan mismatch: step");
    }
    // no plan at all
    {
        llama_kv_resolve_input in = qwen38(T_TURBOT, T_TURBOT);
        in.plan_check = nullptr;
        const auto r = llama_kv_resolve(in);
        TCHECK(r.type_k == T_5P && r.type_v == T_5P, "no plan: %s/%s", tn(r.type_k), tn(r.type_v));
    }
    // an env variable the turbot cache rejects
    {
        llama_kv_resolve_input in = qwen38(T_TURBOT, T_TURBOT);
        in.env_refusal = "TURBO_INNERQ is incompatible";
        const auto r = llama_kv_resolve(in);
        TCHECK(r.type_k == T_5P && r.type_v == T_5P && has_step(r, 'B', T_TURBOT, T_5P, "TURBO_INNERQ"), "env refusal");
    }
    // one attention layer's KV on the CPU
    {
        llama_kv_resolve_input in = qwen38(T_TURBOT, T_TURBOT);
        for (auto & L : in.layers) {
            if (L.il == 31) {
                L.dev_refusal = "attention KV must be on a CUDA device (layer 31: CPU)";
            }
        }
        const auto r = llama_kv_resolve(in);
        dump("Qwen3.8-27B turbot, layer 31 on CPU", r);
        TCHECK(r.type_k == T_5P && r.type_v == T_5P && has_step(r, 'B', T_TURBOT, T_5P, "layer 31"), "device refusal");
    }
    // the turbot constructor refused anyway (the llama_context fallback)
    {
        llama_kv_resolve_input in = qwen38(T_TURBOT, T_TURBOT);
        in.turbot_refused = "turbot: plan x line 3: layer 1 is not an attention layer of this cache";
        const auto r = llama_kv_resolve(in);
        TCHECK(r.type_k == T_5P && r.type_v == T_5P, "constructor refusal: %s/%s", tn(r.type_k), tn(r.type_v));
        TCHECK(has_step(r, 'B', T_TURBOT, T_5P, "turbot: plan x line 3"), "constructor refusal: the prefix is said once");
        TCHECK(r.steps.size() == 1 && r.steps[0].reason.find("turbot: turbot") == std::string::npos, "double prefix");
    }
    // -ctk turbot with another -ctv: the turbot side takes turbo5p
    {
        const auto r = llama_kv_resolve(qwen38(T_TURBOT, T_5P));
        dump("Qwen3.8-27B -ctk turbot -ctv turbo5p", r);
        TCHECK(r.type_k == T_5P && r.type_v == T_5P, "one-sided turbot: %s/%s", tn(r.type_k), tn(r.type_v));
        TCHECK(has_step(r, 'K', T_TURBOT, T_5P, "together"), "one-sided turbot: step");
    }
    // turbo5p and q8_0 fit as they are
    for (ggml_type t : { T_5P, T_4P, T_4, T_Q8, T_F16 }) {
        const auto r = llama_kv_resolve(qwen38(t, t));
        TCHECK(r.type_k == t && r.type_v == t && r.steps.empty(), "Qwen3.8 %s changed to %s/%s", tn(t), tn(r.type_k), tn(r.type_v));
    }
    // a mixed pair with an FA kernel stays
    {
        const auto r = llama_kv_resolve(qwen38(T_4, T_Q8));
        TCHECK(r.type_k == T_4 && r.type_v == T_Q8 && r.steps.empty(), "turbo4/q8_0: %s/%s", tn(r.type_k), tn(r.type_v));
    }
    // a split-plane K against an f16 V: the FA dispatch refuses the pair, K leaves the turbo family
    {
        const auto r = llama_kv_resolve(qwen38(T_5P, T_F16));
        dump("Qwen3.8-27B -ctk turbo5p -ctv f16", r);
        TCHECK(r.type_k == T_Q8 && r.type_v == T_F16, "turbo5p/f16: %s/%s", tn(r.type_k), tn(r.type_v));
        TCHECK(has_step(r, 'K', T_5P, T_Q8, "no flash-attention kernel"), "turbo5p/f16: step");
    }
    // flash attention off: turbot and turbo need it; K may stay q8_0, a quantized V may not
    {
        llama_kv_resolve_input in = qwen38(T_TURBOT, T_TURBOT);
        in.flash_attn = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        const auto r = llama_kv_resolve(in);
        dump("Qwen3.8-27B turbot, -fa off", r);
        TCHECK(r.type_k == T_Q8 && r.type_v == T_F16, "fa off: %s/%s", tn(r.type_k), tn(r.type_v));
        TCHECK(has_step(r, 'K', T_TURBOT, T_Q8, "needs flash attention"), "fa off: K step");
        TCHECK(has_step(r, 'V', T_TURBOT, T_F16, "quantized V cache needs flash attention"), "fa off: V step");
    }
    {
        llama_kv_resolve_input in = qwen38(T_Q8, T_Q8);
        in.flash_attn = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        const auto r = llama_kv_resolve(in);
        TCHECK(r.type_k == T_Q8 && r.type_v == T_F16, "fa off q8_0: %s/%s", tn(r.type_k), tn(r.type_v));
    }
    {
        llama_kv_resolve_input in = qwen38(T_F16, T_F16);
        in.flash_attn = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        const auto r = llama_kv_resolve(in);
        TCHECK(r.type_k == T_F16 && r.type_v == T_F16 && r.steps.empty(), "fa off f16 changed");
    }
}

// Spark-X2.5-4B: 36 layers (27 sliding-window, 9 full), 4 KV heads x 256, iSWA
static void test_spark_4b() {
    llama_kv_resolve_input in = make_model(T_TURBOT, T_TURBOT, 36, 4, 256);
    in.swa        = true;
    in.plan_check = plan_check_text(plan_text_for(attn_layers_of(in)));
    const auto r  = llama_kv_resolve(in);
    dump("Spark-X2.5-4B turbot (iSWA, 4 x 256)", r);
    TCHECK(r.type_k == T_5P && r.type_v == T_5P, "Spark 4B: %s/%s", tn(r.type_k), tn(r.type_v));
    TCHECK(r.steps.size() == 1 && has_step(r, 'B', T_TURBOT, T_5P, "SWA caches are unsupported"), "Spark 4B: step");
}

// Spark-X2.5-1.7B: 28 layers, 2 KV heads x 256 (512-element rows), iSWA
static void test_spark_1_7b() {
    llama_kv_resolve_input in = make_model(T_TURBOT, T_TURBOT, 28, 2, 256);
    in.swa = true;
    const auto r = llama_kv_resolve(in);
    dump("Spark-X2.5-1.7B turbot (iSWA, 2 x 256)", r);
    TCHECK(r.type_k == T_5P512 && r.type_v == T_5P512, "Spark 1.7B: %s/%s", tn(r.type_k), tn(r.type_v));
    TCHECK(r.steps.size() == 1 && has_step(r, 'B', T_TURBOT, T_5P, "SWA"), "Spark 1.7B: one step (the 512 swap is not a downgrade)");

    // turbo5p asked directly: the same swap, no warning
    const auto r2 = llama_kv_resolve(make_model(T_5P, T_5P, 28, 2, 256));
    TCHECK(r2.type_k == T_5P512 && r2.type_v == T_5P512 && r2.steps.empty(), "Spark 1.7B turbo5p: %s/%s", tn(r2.type_k), tn(r2.type_v));

    // turbo4p needs 1024-element rows: turbo4
    const auto r3 = llama_kv_resolve(make_model(T_4P, T_4P, 28, 2, 256));
    TCHECK(r3.type_k == T_4 && r3.type_v == T_4, "Spark 1.7B turbo4p: %s/%s", tn(r3.type_k), tn(r3.type_v));

    // a downgrade that would leave turbo4 against turbo5p512 (no FA kernel): both take turbo4
    const auto r4 = llama_kv_resolve(make_model(T_4P, T_5P, 28, 2, 256));
    dump("Spark-X2.5-1.7B -ctk turbo4p -ctv turbo5p", r4);
    TCHECK(r4.type_k == T_4 && r4.type_v == T_4, "Spark 1.7B turbo4p/turbo5p: %s/%s", tn(r4.type_k), tn(r4.type_v));
}

// MiniCPM5-2B: 2 KV heads x 128 (256-element rows), no SWA
static void test_minicpm5() {
    llama_kv_resolve_input in = make_model(T_TURBOT, T_TURBOT, 40, 2, 128);
    in.plan_check = plan_check_text(plan_text_for(attn_layers_of(in)));
    const auto r  = llama_kv_resolve(in);
    dump("MiniCPM5 turbot (2 x 128)", r);
    TCHECK(r.type_k == T_4 && r.type_v == T_4, "MiniCPM5: %s/%s", tn(r.type_k), tn(r.type_v));
    TCHECK(r.steps.size() == 1 && has_step(r, 'B', T_TURBOT, T_4, "unsupported head geometry"), "MiniCPM5: one step");
    TCHECK(has_step(r, 'B', T_TURBOT, T_4, "turbo5p: layer 0: KV row 256"), "MiniCPM5: the turbo5p reason is listed");

    const auto r2 = llama_kv_resolve(make_model(T_5P, T_5P, 40, 2, 128));
    TCHECK(r2.type_k == T_4 && r2.type_v == T_4, "MiniCPM5 turbo5p: %s/%s", tn(r2.type_k), tn(r2.type_v));

    // mixed turbo4p/turbo4: turbo4p cannot hold 256-element rows, the pair ends matched
    const auto r3 = llama_kv_resolve(make_model(T_4P, T_4, 40, 2, 128));
    TCHECK(r3.type_k == T_4 && r3.type_v == T_4, "MiniCPM5 turbo4p/turbo4: %s/%s", tn(r3.type_k), tn(r3.type_v));
}

// Muse Glimmer 30B: 2 KV heads x 128, SWA
static void test_muse_glimmer() {
    llama_kv_resolve_input in = make_model(T_TURBOT, T_TURBOT, 48, 2, 128);
    in.swa = true;
    const auto r = llama_kv_resolve(in);
    dump("Muse Glimmer turbot (SWA, 2 x 128)", r);
    TCHECK(r.type_k == T_4 && r.type_v == T_4, "Muse Glimmer: %s/%s", tn(r.type_k), tn(r.type_v));
}

// Granite 4.2 8B: 8 KV heads x 128 (1024-element rows)
static void test_granite() {
    const auto r = llama_kv_resolve(make_model(T_TURBOT, T_TURBOT, 40, 8, 128));
    TCHECK(r.type_k == T_5P && r.type_v == T_5P, "Granite: %s/%s", tn(r.type_k), tn(r.type_v));
}

// head 64 (8 KV heads x 64): no turbo FA kernel takes D = 64 without zero-padding
static void test_head64() {
    const auto r = llama_kv_resolve(make_model(T_5P, T_5P, 24, 8, 64));
    dump("head-64 model turbo5p (8 x 64)", r);
    TCHECK(r.type_k == T_Q8 && r.type_v == T_Q8, "head 64: %s/%s", tn(r.type_k), tn(r.type_v));
    TCHECK(r.steps.size() == 1 && has_step(r, 'B', T_5P, T_Q8, "head size 64"), "head 64: step");

    const auto r2 = llama_kv_resolve(make_model(T_TURBOT, T_TURBOT, 24, 8, 64));
    TCHECK(r2.type_k == T_Q8 && r2.type_v == T_Q8, "head 64 turbot: %s/%s", tn(r2.type_k), tn(r2.type_v));

    // [TAG_KV_RESOLVE] turbo4 asked for: the zero-padding path (64 -> 128) is kept as before the resolver
    const auto r3 = llama_kv_resolve(make_model(T_4, T_4, 24, 8, 64));
    TCHECK(r3.type_k == T_4 && r3.type_v == T_4 && r3.steps.empty(), "head 64 turbo4: %s/%s", tn(r3.type_k), tn(r3.type_v));

    // turbo5p K cannot hold 64-element heads; the asked turbo4 V is taken for both
    const auto r4 = llama_kv_resolve(make_model(T_5P, T_4, 24, 8, 64));
    TCHECK(r4.type_k == T_4 && r4.type_v == T_4, "head 64 turbo5p/turbo4: %s/%s", tn(r4.type_k), tn(r4.type_v));
}

// [TAG_KV_RESOLVE] shapes the turbo2/3/4 zero-padding path serves when asked for, and never takes as a fallback
static void test_turbo_padding() {
    // Gemma-4-like iSWA: 256-element SWA heads, 512-element global heads (F16-converted FA path)
    const auto gemma4 = [](ggml_type tk, ggml_type tv) {
        llama_kv_resolve_input in = make_model(tk, tv, 30, 8, 256);
        in.swa = true;
        for (auto & L : in.layers) {
            if (L.il % 6 == 5) {
                L.head_k    = 512;
                L.head_v    = 512;
                L.n_head_kv = 2;
            }
        }
        return in;
    };
    const auto r = llama_kv_resolve(gemma4(T_4, T_4));
    TCHECK(r.type_k == T_4 && r.type_v == T_4 && r.steps.empty(), "gemma4 turbo4: %s/%s", tn(r.type_k), tn(r.type_v));
    const auto r2 = llama_kv_resolve(gemma4(T_TURBOT, T_TURBOT));
    dump("Gemma-4-like turbot (256 / 512 heads)", r2);
    TCHECK(r2.type_k == T_Q8 && r2.type_v == T_Q8, "gemma4 turbot: %s/%s", tn(r2.type_k), tn(r2.type_v));

    // K 192 / V 128 (no MLA): the padded K 256 has no FA kernel against V 128
    llama_kv_resolve_input in = make_model(T_4, T_4, 27, 16, 192);
    for (auto & L : in.layers) {
        L.head_v = 128;
    }
    const auto r3 = llama_kv_resolve(in);
    dump("K 192 / V 128 turbo4", r3);
    TCHECK(r3.type_k == T_Q8 && r3.type_v == T_Q8 && has_step(r3, 'B', T_4, T_Q8, "no turbo flash-attention kernel"),
           "k192/v128 turbo4: %s/%s", tn(r3.type_k), tn(r3.type_v));

    // head 80: turbo4 asked is padded to 128 and kept; turbot never lands on a padded type (q8_0 does not fit 80 either)
    const auto r4 = llama_kv_resolve(make_model(T_4, T_4, 32, 8, 80));
    TCHECK(r4.type_k == T_4 && r4.type_v == T_4 && r4.steps.empty(), "head 80 turbo4: %s/%s", tn(r4.type_k), tn(r4.type_v));
    const auto r5 = llama_kv_resolve(make_model(T_TURBOT, T_TURBOT, 32, 8, 80));
    TCHECK(r5.type_k == T_F16 && r5.type_v == T_F16, "head 80 turbot: %s/%s", tn(r5.type_k), tn(r5.type_v));
}

// head 80: not a whole q8_0 block
static void test_head80() {
    const auto r = llama_kv_resolve(make_model(T_Q8, T_Q8, 32, 8, 80));
    dump("head-80 model q8_0 (8 x 80)", r);
    TCHECK(r.type_k == T_F16 && r.type_v == T_F16, "head 80: %s/%s", tn(r.type_k), tn(r.type_v));
}

// MLA (DeepSeek-V2 / Ling-3.0-tiny style): one latent KV head, K 576, V 512 read from the K cache
static void test_mla() {
    const auto mla = [](ggml_type tk, ggml_type tv, llama_flash_attn_type fa) {
        llama_kv_resolve_input in = make_model(tk, tv, 27, 1, 576);
        for (auto & L : in.layers) {
            L.head_v = 512;
        }
        in.mla        = true;
        in.same_type  = true;
        in.flash_attn = fa;
        return in;
    };

    const auto r = llama_kv_resolve(mla(T_TURBOT, T_TURBOT, LLAMA_FLASH_ATTN_TYPE_AUTO));
    dump("MLA turbot", r);
    TCHECK(r.type_k == T_Q8 && r.type_v == T_Q8, "MLA: %s/%s", tn(r.type_k), tn(r.type_v));

    const auto r2 = llama_kv_resolve(mla(T_5P, T_5P, LLAMA_FLASH_ATTN_TYPE_ENABLED));
    TCHECK(r2.type_k == T_Q8 && r2.type_v == T_Q8 && has_step(r2, 'B', T_5P, T_Q8, "MLA"), "MLA turbo5p: %s/%s", tn(r2.type_k), tn(r2.type_v));

    const auto r3 = llama_kv_resolve(mla(T_5P, T_5P, LLAMA_FLASH_ATTN_TYPE_DISABLED));
    dump("MLA turbo5p, -fa off", r3);
    TCHECK(r3.type_k == T_F16 && r3.type_v == T_F16, "MLA fa off: %s/%s", tn(r3.type_k), tn(r3.type_v));

    // K and V differ: V follows K
    const auto r4 = llama_kv_resolve(mla(T_Q8, T_F16, LLAMA_FLASH_ATTN_TYPE_AUTO));
    TCHECK(r4.type_k == T_Q8 && r4.type_v == T_Q8 && has_step(r4, 'V', T_F16, T_Q8, nullptr), "MLA q8_0/f16: %s/%s", tn(r4.type_k), tn(r4.type_v));

    // [TAG_KV_RESOLVE] turbo3 asked for on MLA (GLM-4.7-Flash): K 576 zero-padded to 640, V 512, fattn.cu case 640
    const auto r5 = llama_kv_resolve(mla(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0, LLAMA_FLASH_ATTN_TYPE_AUTO));
    dump("MLA turbo3 (asked)", r5);
    TCHECK(r5.type_k == GGML_TYPE_TURBO3_0 && r5.type_v == GGML_TYPE_TURBO3_0 && r5.steps.empty(),
           "MLA turbo3: %s/%s", tn(r5.type_k), tn(r5.type_v));
    const auto r6 = llama_kv_resolve(mla(T_4, T_4, LLAMA_FLASH_ATTN_TYPE_DISABLED));
    TCHECK(r6.type_k == T_F16 && r6.type_v == T_F16, "MLA turbo4 fa off: %s/%s", tn(r6.type_k), tn(r6.type_v));
}

// DeepSeek V4 style: no turbo query rotation in its attention input, one type for K and V
static void test_no_turbo_graph() {
    llama_kv_resolve_input in = make_model(T_4, T_4, 16, 1, 512);
    in.turbo_graph = false;
    in.same_type   = true;
    in.swa         = true;
    const auto r = llama_kv_resolve(in);
    TCHECK(r.type_k == T_Q8 && r.type_v == T_Q8, "no turbo graph: %s/%s", tn(r.type_k), tn(r.type_v));
}

// recurrent-only (Mamba): no KV cache, nothing to decide or warn about
static void test_recurrent() {
    llama_kv_resolve_input in = make_model(T_TURBOT, T_TURBOT, 48, 0, 0);
    in.no_kv = true;
    const auto r = llama_kv_resolve(in);
    dump("recurrent-only turbot", r);
    TCHECK(r.no_kv && r.steps.empty(), "recurrent: no_kv / steps");
    TCHECK(r.type_k == T_F16 && r.type_v == T_F16, "recurrent: %s/%s", tn(r.type_k), tn(r.type_v));

    llama_kv_resolve_input in2 = make_model(T_F16, T_F16, 48, 0, 0);
    in2.no_kv = true;
    const auto r2 = llama_kv_resolve(in2);
    TCHECK(r2.type_k == T_F16 && r2.type_v == T_F16 && r2.steps.empty(), "recurrent f16");
}

// the built-in default plan names Qwen3.8-27B's layers only
static void test_builtin_plan() {
    std::string why;
    const std::vector<int32_t> qwen = { 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63 };
    TCHECK(llama_turbot_plan_matches(llama_turbot_default_plan_text(), qwen, 262144, why), "built-in plan vs Qwen3.8: %s", why.c_str());

    std::vector<int32_t> spark;
    for (int32_t il = 3; il < 36; il += 4) {
        spark.push_back(il);
    }
    TCHECK(!llama_turbot_plan_matches(llama_turbot_default_plan_text(), spark, 262144, why), "built-in plan fits Spark");
    TCHECK(why.find("layer 39 is not an attention layer") != std::string::npos, "built-in plan vs Spark: %s", why.c_str());
}

int main() {
    fprintf(stderr, "test-kv-resolve:\n");

    test_turbot_helpers();
    test_builtin_plan();
    test_qwen38();
    test_spark_4b();
    test_spark_1_7b();
    test_minicpm5();
    test_muse_glimmer();
    test_granite();
    test_head64();
    test_turbo_padding();
    test_head80();
    test_mla();
    test_no_turbo_graph();
    test_recurrent();

    fprintf(stderr, "%d checks, %d failed\n", g_checks, g_fail);

    return g_fail == 0 ? 0 : 1;
}
