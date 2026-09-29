#pragma once

// [TAG_FN_AUTO] automatic Flash-Next (qwen4exp) defaults behind ONE switch, LLAMA_FLASHNEXT_PROFILE:
//   off       nothing automatic (the switch-by-switch behaviour of flashnext/all)
//   safe      default: only the measured winners (speed/r1 hot48ad, +35% code / +45% prose at 32K): every trunk layer's
//             routed experts on the host, the adaptive per-expert VRAM hot set sized by the VRAM fit, and the RAM fit
//   fast      safe + the built but not yet GPU-measured levers (host bridge, CPU MoE switches, split trims, QSA chunk,
//             PLE direct I/O, MTP cost); LLAMA_FLASHNEXT_FAST=<NAME=VALUE,...> replaces this list
//   fast-dma  fast with the gen5 DMA share instead of the host bridge (the two are exclusive)
// Active only for arch qwen4exp whose routed experts the tensor overrides (--n-cpu-moe, --cpu-moe, -ot, -fit) put on
// the host, with the weights loaded (not the -fit memory probes). Every other model is unchanged. A variable that is
// set in the environment always wins over the profile.
//
// The safe items are read through llama_fn_env() by the code that owns them (placement, hot set, VRAM fit, RAM fit),
// so they never leave the model. The fast levers are read by getenv() deep in other code, so while the model is loaded
// the profile puts them into the process environment (GGML_CPU_* through ggml_cpu_fn_set_switch) and takes them out
// again when the model is freed.

#include "llama.h"
#include "llama-ext.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

struct llama_model;
struct llama_model_loader;
struct llama_context;

enum llama_fn_profile {
    LLAMA_FN_PROFILE_OFF      = 0,
    LLAMA_FN_PROFILE_SAFE     = 1,
    LLAMA_FN_PROFILE_FAST     = 2,
    LLAMA_FN_PROFILE_FAST_DMA = 3,
};

struct llama_fn_opt {
    std::string name;
    std::string value;
    bool        inject = false; // read by getenv() outside the fn-auto lookup: set in the environment while the model lives
};

struct llama_fn_auto_state {
    bool             active      = false;
    llama_fn_profile profile     = LLAMA_FN_PROFILE_OFF;
    bool             promote     = false; // LLAMA_FN_PLACEMENT=auto: the routed experts of every trunk layer on the host
    int              n_layer     = 0;     // trunk layers
    int              n_host_user = 0;     // trunk layers with routed experts the overrides put on the host
    int              n_host      = 0;     // after load: trunk layers with host experts and a device router
    bool             mtp_loaded  = false; // the MTP block is loaded, so an MTP draft context will follow
    std::string      model_path;

    std::vector<llama_fn_opt> opts;     // the profile's values (explicit environment values are not copied here)
    std::vector<std::string>  injected; // names this state put into the environment
    std::vector<std::string>  cpu_set;  // GGML_CPU_* switches this state set
};

//
// pure pieces (tests/test-fn-auto.cpp)
//

// "off"/"0", "safe"/"1", "fast", "fast-dma"; nullptr or "" = safe. *ok = false for an unknown name (then safe)
LLAMA_API llama_fn_profile llama_fn_profile_parse(const char * s, bool * ok);
LLAMA_API const char *     llama_fn_profile_name(llama_fn_profile p);

// "NAME=VALUE,NAME=VALUE" (spaces around items allowed); false if an item has no '=' or an empty name
LLAMA_API bool llama_fn_parse_opt_list(const char * s, std::vector<std::pair<std::string, std::string>> & out);

// the options of a profile; fast_list (LLAMA_FLASHNEXT_FAST) replaces the built-in fast levers of "fast"
LLAMA_API std::vector<llama_fn_opt> llama_fn_profile_opts(llama_fn_profile p, const char * fast_list);

// trunk layers 0..n_layer-1 with at least one routed expert tensor that the first matching override puts on a host
// buffer type; overrides = (pattern, buft is host), in order
LLAMA_API int llama_fn_count_host_expert_layers(const std::vector<std::pair<std::string, bool>> & overrides, int n_layer);

// the environment value, else the state's profile value (when active), else nullptr
LLAMA_API const char * llama_fn_state_env(const llama_fn_auto_state * st, const char * name);

// put the inject options into the environment (skipping names set there by the user) and set the GGML_CPU_* switches;
// undo takes back only what inject set. Reference counted per name across models.
LLAMA_API void llama_fn_state_inject(llama_fn_auto_state & st);
LLAMA_API void llama_fn_state_undo  (llama_fn_auto_state & st);

// slots per layer when the budget is spread evenly (no routing profile): budget / sum of the layers' bytes per expert,
// at most n_expert
LLAMA_API int32_t llama_fn_even_slots(const std::vector<size_t> & bytes_per_expert, size_t budget, int32_t n_expert);

//
// model hooks (src/llama.cpp) and lookups
//

// before the tensors are created: decide the profile, the placement and the options; inject the fast levers
void llama_fn_auto_on_load(llama_model & model, llama_model_loader & ml, const std::string & fname, const llama_model_params & params);
// after the tensors are created: confirm host experts with a device router, else turn everything back off
void llama_fn_auto_after_load(llama_model & model);
// the model is being freed
void llama_fn_auto_on_free(llama_model & model);

// getenv(name), else this model's automatic value, else nullptr
const char * llama_fn_env(const llama_model & model, const char * name);
bool         llama_fn_active(const llama_model & model);

// the contexts of a model (the VRAM fit breakdown, and whether the MTP draft context exists yet)
void llama_fn_ctx_add   (const llama_model & model, llama_context * ctx, int ctx_type);
void llama_fn_ctx_remove(const llama_model & model, llama_context * ctx);
std::vector<std::pair<llama_context *, int>> llama_fn_ctx_list(const llama_model & model);
