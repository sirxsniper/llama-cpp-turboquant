// [TAG_FN_AUTO] automatic qwen4exp (Flash-Next) defaults behind LLAMA_FLASHNEXT_PROFILE. See llama-fn-auto.h.

#include "llama-fn-auto.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-model-loader.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <regex>

namespace {

enum : int {
    P_SAFE  = 1 << LLAMA_FN_PROFILE_SAFE,
    P_FAST  = 1 << LLAMA_FN_PROFILE_FAST,
    P_TRIAL = 1 << LLAMA_FN_PROFILE_TRIAL,
    P_TDMA  = 1 << LLAMA_FN_PROFILE_TRIAL_DMA,
    P_ALL   = P_SAFE | P_FAST | P_TRIAL | P_TDMA,
    P_TRY   = P_TRIAL | P_TDMA,
};

struct fn_item {
    const char * name;
    const char * value;
    int          profiles;
};

// safe: speed/r1 hot48ad against F0 (ncmoe 39), 3 interleaved rounds at 32K with turbot KV and MTP: +35.4% greedy code,
// +45.1% temp-1 prose, KLD on the pure-placement band (r2/TEST_PLAN.md 0.4). That run started the hot set from a routing
// profile (fn_r1_all.moeprof); "even" (no <model>.moeprof sidecar) is not measured yet.
// fast: safe + the measured winners. A lever moves from P_TRY to P_FAST only with its A/B result (speed/history.md).
// trial / trial-dma: built and unit tested, NOT measured on the GPU; for A/B runs only.
const fn_item k_items[] = {
    { "LLAMA_FN_PLACEMENT",     "auto",  P_ALL  }, // every trunk layer's routed experts on the host
    { "LLAMA_MOE_HOT_PROFILE",  "",      P_ALL  }, // set at load: <model>.moeprof if present, else "even"
    { "LLAMA_MOE_HOT_MIB",      "auto",  P_ALL  },
    { "LLAMA_MOE_HOT_FIT",      "1",     P_ALL  }, // [TAG_FN_VRAM_FIT]
    { "LLAMA_MOE_HOT_ADAPT",    "1",     P_ALL  },
    { "LLAMA_MOE_HOT_ADMIT",    "2/32",  P_ALL  },
    { "LLAMA_RAM_FIT",          "1",     P_ALL  }, // [TAG_FN_RAM_FIT]

    // P_FAST levers (measured winners) go here

    { "LLAMA_MOE_BRIDGE",       "1",     P_TRIAL },
    { "LLAMA_MOE_DMA_SHARE",    "auto",  P_TDMA  }, // exclusive with the bridge
    { "GGML_CPU_APPLY_ONCE",    "1",     P_TRY   },
    { "GGML_CPU_Q5_1_AVX512",   "1",     P_TRY   },
    { "GGML_CPU_MMID_MR",       "1",     P_TRY   },
    { "GGML_CPU_MOE_FUSE",      "1",     P_TRY   },
    { "GGML_SCHED_SPLIT_ASYNC", "1",     P_TRY   },
    { "LLAMA_PLE_HOST_GATHER",  "1",     P_TRY   },
    { "LLAMA_PLE_DIRECT_IO",    "1",     P_TRY   }, // flashnext/ple-dio; no effect before that branch is merged
    { "LLAMA_GRAPH_PER_WIDTH",  "1",     P_TRY   },
    { "TURBO_QSA_CHUNK",        "64",    P_TRY   },
    { "SPEC_MTP_COST",          "1",     P_TRY   },
    { "LLAMA_MTP_ATTN_WINDOW",  "32768", P_TRY   },
};

// names the fn-auto aware code reads through llama_fn_env(): never put into the environment
bool fn_is_lookup_name(const std::string & n) {
    return n.rfind("LLAMA_MOE_HOT_", 0) == 0 || n.rfind("LLAMA_FN_", 0) == 0 || n == "LLAMA_RAM_FIT";
}

std::string fn_trim(const std::string & s) {
    size_t b = 0;
    size_t e = s.size();
    while (b < e && isspace((unsigned char) s[b])) {
        ++b;
    }
    while (e > b && isspace((unsigned char) s[e - 1])) {
        --e;
    }
    return s.substr(b, e - b);
}

std::string fn_lower(const char * s) {
    std::string r = s ? s : "";
    for (char & c : r) {
        c = (char) tolower((unsigned char) c);
    }
    return fn_trim(r);
}

void fn_setenv(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : ""); // "" removes the variable
#else
    if (value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

// GGML_CPU_* switches: ggml_cpu_init() read them from the environment before any model was loaded
struct fn_cpu_sw {
    const char *            name;
    enum ggml_cpu_fn_switch sw;
};
const fn_cpu_sw k_cpu_sw[] = {
    { "GGML_CPU_APPLY_ONCE",  GGML_CPU_FN_APPLY_ONCE  },
    { "GGML_CPU_Q5_1_AVX512", GGML_CPU_FN_Q5_1_AVX512 },
    { "GGML_CPU_MMID_MR",     GGML_CPU_FN_MMID_MR     },
    { "GGML_CPU_MOE_FUSE",    GGML_CPU_FN_MOE_FUSE    },
};

const fn_cpu_sw * fn_cpu_switch(const std::string & name) {
    for (const auto & s : k_cpu_sw) {
        if (name == s.name) {
            return &s;
        }
    }
    return nullptr;
}

using fn_set_switch_t = void (*)(enum ggml_cpu_fn_switch, int);

fn_set_switch_t fn_cpu_setter() {
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CPU");
    return reg ? (fn_set_switch_t) ggml_backend_reg_get_proc_address(reg, "ggml_cpu_fn_set_switch") : nullptr;
}

std::mutex                 g_env_mtx;
std::map<std::string, int> g_env_ref; // names in the environment put there by a state, with their holder count
std::map<std::string, int> g_cpu_ref; // GGML_CPU_* switches set by a state, with their holder count

std::mutex g_ctx_mtx;
std::map<const llama_model *, std::vector<std::pair<llama_context *, int>>> g_ctxs;

bool fn_file_exists(const std::string & path) {
    std::ifstream f(path);
    return f.good();
}

// [TAG_FN_R4_REVIEW] the user's value of an item under another name: LLAMA_MTP_WINDOW is LLAMA_MTP_ATTN_WINDOW under
// Strata's name (llama-model.cpp reads it when LLAMA_MTP_ATTN_WINDOW is unset), so an injected default must not hide it
const char * fn_alias_value(const std::string & name, const char ** alias) {
    if (name == "LLAMA_MTP_ATTN_WINDOW") {
        const char * v = getenv("LLAMA_MTP_WINDOW");
        if (v && v[0]) {
            *alias = "LLAMA_MTP_WINDOW";
            return v;
        }
    }
    return nullptr;
}

} // namespace

llama_fn_profile llama_fn_profile_parse(const char * s, bool * ok) {
    if (ok) {
        *ok = true;
    }
    const std::string v = fn_lower(s);
    if (v.empty() || v == "safe" || v == "1" || v == "on" || v == "default") {
        return LLAMA_FN_PROFILE_SAFE;
    }
    if (v == "off" || v == "0" || v == "none") {
        return LLAMA_FN_PROFILE_OFF;
    }
    if (v == "fast" || v == "2") {
        return LLAMA_FN_PROFILE_FAST;
    }
    if (v == "trial") {
        return LLAMA_FN_PROFILE_TRIAL;
    }
    if (v == "trial-dma" || v == "trial_dma") {
        return LLAMA_FN_PROFILE_TRIAL_DMA;
    }
    if (ok) {
        *ok = false;
    }
    return LLAMA_FN_PROFILE_SAFE;
}

const char * llama_fn_profile_name(llama_fn_profile p) {
    switch (p) {
        case LLAMA_FN_PROFILE_OFF:       return "off";
        case LLAMA_FN_PROFILE_SAFE:      return "safe";
        case LLAMA_FN_PROFILE_FAST:      return "fast";
        case LLAMA_FN_PROFILE_TRIAL:     return "trial";
        case LLAMA_FN_PROFILE_TRIAL_DMA: return "trial-dma";
    }
    return "?";
}

int llama_fn_fast_lever_count() {
    int n = 0;
    for (const auto & it : k_items) {
        n += (it.profiles & P_FAST) && !(it.profiles & P_SAFE) ? 1 : 0;
    }
    return n;
}

bool llama_fn_parse_opt_list(const char * s, std::vector<std::pair<std::string, std::string>> & out) {
    out.clear();
    if (!s) {
        return true;
    }
    const std::string str = s;
    bool ok = true;
    size_t pos = 0;
    while (pos <= str.size()) {
        size_t end = str.find(',', pos);
        if (end == std::string::npos) {
            end = str.size();
        }
        const std::string item = fn_trim(str.substr(pos, end - pos));
        if (!item.empty()) {
            const size_t eq = item.find('=');
            const std::string name = eq == std::string::npos ? "" : fn_trim(item.substr(0, eq));
            if (name.empty()) {
                ok = false;
            } else {
                out.emplace_back(name, fn_trim(item.substr(eq + 1)));
            }
        }
        pos = end + 1;
    }
    return ok;
}

std::vector<llama_fn_opt> llama_fn_profile_opts(llama_fn_profile p, const char * fast_list) {
    std::vector<llama_fn_opt> out;
    if (p == LLAMA_FN_PROFILE_OFF) {
        return out;
    }
    std::vector<std::pair<std::string, std::string>> custom;
    const bool use_custom = p != LLAMA_FN_PROFILE_SAFE && fast_list && fast_list[0] && llama_fn_parse_opt_list(fast_list, custom);
    const int bit = 1 << p;
    for (const auto & it : k_items) {
        if (!(it.profiles & bit)) {
            continue;
        }
        if (use_custom && !(it.profiles & P_SAFE)) {
            continue; // LLAMA_FLASHNEXT_FAST replaces the built-in levers
        }
        llama_fn_opt o;
        o.name   = it.name;
        o.value  = it.value;
        o.inject = !fn_is_lookup_name(o.name);
        out.push_back(o);
    }
    for (const auto & [name, value] : custom) {
        // a lambda may capture a structured binding only from C++20 on (clang before 16 rejects it)
        const std::string & key = name;
        auto it = std::find_if(out.begin(), out.end(), [&key](const llama_fn_opt & o) { return o.name == key; });
        if (it != out.end()) {
            it->value = value;
        } else {
            llama_fn_opt o;
            o.name   = name;
            o.value  = value;
            o.inject = !fn_is_lookup_name(name);
            out.push_back(o);
        }
    }
    return out;
}

int llama_fn_count_host_expert_layers(const std::vector<std::pair<std::string, bool>> & overrides, int n_layer) {
    std::vector<std::regex> re;
    std::vector<bool>       re_ok;
    for (const auto & o : overrides) {
        try {
            re.emplace_back(o.first);
            re_ok.push_back(true);
        } catch (const std::regex_error &) {
            re.emplace_back();
            re_ok.push_back(false); // the loader throws on it later; here it matches nothing
        }
    }
    static const char * kinds[] = { "ffn_up_exps", "ffn_gate_exps", "ffn_down_exps", "ffn_gate_up_exps" };
    int n = 0;
    for (int il = 0; il < n_layer; ++il) {
        bool host = false;
        for (const char * k : kinds) {
            const std::string name = "blk." + std::to_string(il) + "." + k + ".weight";
            for (size_t i = 0; i < re.size(); ++i) {
                if (re_ok[i] && std::regex_search(name, re[i])) {
                    host = host || overrides[i].second; // the first matching override decides, as in the loader
                    break;
                }
            }
        }
        n += host ? 1 : 0;
    }
    return n;
}

const char * llama_fn_state_env(const llama_fn_auto_state * st, const char * name) {
    if (const char * v = getenv(name)) {
        return v;
    }
    if (!st || !st->active) {
        return nullptr;
    }
    for (const auto & o : st->opts) {
        if (o.name == name) {
            return o.value.c_str();
        }
    }
    return nullptr;
}

void llama_fn_state_inject(llama_fn_auto_state & st) {
    std::lock_guard<std::mutex> lk(g_env_mtx);
    fn_set_switch_t setter = nullptr;
    bool setter_looked = false;
    for (const auto & o : st.opts) {
        if (!o.inject) {
            continue;
        }
        if (const fn_cpu_sw * cs = fn_cpu_switch(o.name)) {
            auto it = g_cpu_ref.find(o.name);
            if (it != g_cpu_ref.end()) {
                it->second++;
                st.cpu_set.push_back(o.name);
                continue;
            }
            if (getenv(o.name.c_str())) {
                continue; // ggml_cpu_init() took the user's value
            }
            if (!setter_looked) {
                setter_looked = true;
                setter = fn_cpu_setter();
            }
            if (!setter) {
                LLAMA_LOG_WARN("fn-auto: the CPU backend has no ggml_cpu_fn_set_switch, %s=%s is not applied\n", o.name.c_str(), o.value.c_str());
                continue;
            }
            setter(cs->sw, atoi(o.value.c_str()));
            g_cpu_ref[o.name] = 1;
            st.cpu_set.push_back(o.name);
            continue;
        }
        auto it = g_env_ref.find(o.name);
        if (it != g_env_ref.end()) {
            it->second++;
            st.injected.push_back(o.name);
            continue;
        }
        if (getenv(o.name.c_str())) {
            continue; // the user's value wins
        }
        const char * alias = nullptr;
        if (fn_alias_value(o.name, &alias)) {
            continue; // [TAG_FN_R4_REVIEW] the user's value under the alias wins as well
        }
        fn_setenv(o.name.c_str(), o.value.c_str());
        g_env_ref[o.name] = 1;
        st.injected.push_back(o.name);
    }
}

void llama_fn_state_undo(llama_fn_auto_state & st) {
    std::lock_guard<std::mutex> lk(g_env_mtx);
    for (const auto & n : st.injected) {
        auto it = g_env_ref.find(n);
        if (it != g_env_ref.end() && --it->second <= 0) {
            fn_setenv(n.c_str(), nullptr);
            g_env_ref.erase(it);
        }
    }
    st.injected.clear();
    fn_set_switch_t setter = st.cpu_set.empty() ? nullptr : fn_cpu_setter();
    for (const auto & n : st.cpu_set) {
        auto it = g_cpu_ref.find(n);
        if (it != g_cpu_ref.end() && --it->second <= 0) {
            const fn_cpu_sw * cs = fn_cpu_switch(n);
            if (setter && cs) {
                setter(cs->sw, 0); // the environment did not set it, so ggml_cpu_init() left it at 0
            }
            g_cpu_ref.erase(it);
        }
    }
    st.cpu_set.clear();
}

int32_t llama_fn_even_slots(const std::vector<size_t> & bytes_per_expert, size_t budget, int32_t n_expert) {
    size_t sum = 0;
    for (size_t b : bytes_per_expert) {
        sum += b;
    }
    if (sum == 0 || n_expert <= 0) {
        return 0;
    }
    const size_t rows = budget / sum; // slot rows incl. the zero slot of every layer
    return rows > 1 ? (int32_t) std::min<size_t>((size_t) n_expert, rows - 1) : 0;
}

// [TAG_FN_VRAM_FIT] -----------------------------------------------------------------------------------------------

size_t llama_fn_vram_ceiling_default(size_t total) {
    const size_t keep = std::max<size_t>((size_t) 1536 << 20, total / 8);
    const size_t step = (size_t) 256 << 20;
    return total > keep ? (total - keep) / step * step : 0;
}

size_t llama_fn_vram_fit_budget(size_t total, size_t free, size_t ceiling, size_t margin) {
    const size_t used = total > free ? total - free : 0;
    const size_t a = free > margin ? free - margin : 0;
    const size_t b = ceiling > used + margin ? ceiling - used - margin : 0;
    return std::min(a, b);
}

// [TAG_FN_RAM_FIT] -------------------------------------------------------------------------------------------------

llama_ram_fit_out llama_ram_fit_plan(const llama_ram_fit_in & in) {
    constexpr int64_t MiB = 1 << 20;

    llama_ram_fit_out out;
    out.cache_ram_mib = in.cache_ram_mib;
    out.ckpt_mib      = in.ckpt_mib;

    const int64_t n_slots = std::max<int32_t>(1, in.n_slots);
    const int64_t limit   = (int64_t) (std::max(0.0, std::min(1.0, in.frac)) * (double) in.ram_total);
    const int64_t fixed   = (int64_t) (in.model_host + in.host_buffers);
    const int64_t p_req   = in.cache_ram_mib < 0 ? (int64_t) in.ram_total : in.cache_ram_mib*MiB;
    const int64_t c_req   = std::max<int64_t>(0, in.ckpt_mib)*MiB*n_slots;

    out.limit = (uint64_t) limit;
    out.need  = (uint64_t) (fixed + p_req + c_req);
    if (fixed + p_req + c_req <= limit) {
        return out;
    }
    out.over = true;

    const int64_t room_mib = (limit - fixed) / MiB; // negative when the model alone passes the limit

    // checkpoints first: they save re-prefills of the running conversation; never raised, never below the minimum
    if (!in.ckpt_set && in.ckpt_mib > 0) {
        const int64_t per_slot = room_mib > 0 ? room_mib / n_slots : 0;
        out.ckpt_mib = std::min<int64_t>(in.ckpt_mib, std::max<int64_t>(in.ckpt_min_mib, per_slot));
    }
    // then the prompt cache (idle conversations): what is left, and nothing below 256 MiB
    if (!in.cache_ram_set && in.cache_ram_mib != 0) {
        const int64_t left = room_mib - std::max<int64_t>(0, out.ckpt_mib)*n_slots;
        int64_t p = left < 256 ? 0 : left;
        if (in.cache_ram_mib > 0) {
            p = std::min<int64_t>(p, in.cache_ram_mib);
        }
        out.cache_ram_mib = p;
    }
    out.changed = out.cache_ram_mib != in.cache_ram_mib || out.ckpt_mib != in.ckpt_mib;
    return out;
}

// model hooks ---------------------------------------------------------------------------------------------------------

void llama_fn_auto_on_load(llama_model & model, llama_model_loader & ml, const std::string & fname, const llama_model_params & params) {
    if (model.arch != LLM_ARCH_QWEN4EXP || params.no_alloc || params.vocab_only) {
        return;
    }

    const int n_layer = (int) model.hparams.n_layer();
    std::vector<std::pair<std::string, bool>> ov;
    if (params.tensor_buft_overrides) {
        for (const auto * o = params.tensor_buft_overrides; o->pattern != nullptr; ++o) {
            ov.emplace_back(o->pattern, o->buft != nullptr && ggml_backend_buft_is_host(o->buft));
        }
    }
    const int n_host_user = llama_fn_count_host_expert_layers(ov, n_layer);
    if (n_host_user == 0) {
        return; // every routed expert on a device: nothing to do
    }

    const char * pe = getenv("LLAMA_FLASHNEXT_PROFILE");
    bool ok = true;
    const llama_fn_profile prof = llama_fn_profile_parse(pe, &ok);
    if (!ok) {
        LLAMA_LOG_WARN("fn-auto: LLAMA_FLASHNEXT_PROFILE=%s is not off, safe, fast, trial or trial-dma - using safe\n", pe);
    }
    if (prof == LLAMA_FN_PROFILE_OFF) {
        LLAMA_LOG_INFO("fn-auto: qwen4exp with host experts, LLAMA_FLASHNEXT_PROFILE=off: no automatic defaults\n");
        return;
    }
    const char * fast_list = getenv("LLAMA_FLASHNEXT_FAST");
    if (fast_list && fast_list[0]) {
        std::vector<std::pair<std::string, std::string>> tmp;
        if (prof == LLAMA_FN_PROFILE_SAFE) {
            LLAMA_LOG_WARN("fn-auto: LLAMA_FLASHNEXT_FAST is ignored by the safe profile (it sets the levers of fast and trial)\n");
        } else if (!llama_fn_parse_opt_list(fast_list, tmp)) {
            LLAMA_LOG_WARN("fn-auto: LLAMA_FLASHNEXT_FAST=%s is not NAME=VALUE,... - the built-in lever list is used\n", fast_list);
        }
    }

    auto st = std::make_shared<llama_fn_auto_state>();
    st->active      = true;
    st->profile     = prof;
    st->n_layer     = n_layer;
    st->n_host_user = n_host_user;
    st->mtp_loaded  = ml.load_mtp && model.hparams.n_layer_all > model.hparams.n_layer();
    st->model_path  = fname;
    st->opts        = llama_fn_profile_opts(prof, fast_list);
    for (auto & o : st->opts) {
        if (o.name == "LLAMA_MOE_HOT_PROFILE" && o.value.empty()) {
            const std::string sidecar = fname.empty() ? "" : fname + ".moeprof";
            o.value = !sidecar.empty() && fn_file_exists(sidecar) ? sidecar : "even";
        }
    }
    model.fn_auto = st;

    const char * pl = llama_fn_env(model, "LLAMA_FN_PLACEMENT");
    st->promote = pl && strcmp(pl, "auto") == 0;
    if (st->promote) {
        ml.fn_host_experts_n_layer = n_layer;
    }

    llama_fn_state_inject(*st);

    LLAMA_LOG_INFO("fn-auto: qwen4exp with the routed experts of %d of %d trunk layers on the host: LLAMA_FLASHNEXT_PROFILE=%s%s "
            "(off|safe|fast|trial|trial-dma)\n", n_host_user, n_layer, llama_fn_profile_name(prof), pe ? "" : " (default)");
    if (prof == LLAMA_FN_PROFILE_TRIAL || prof == LLAMA_FN_PROFILE_TRIAL_DMA) {
        LLAMA_LOG_WARN("fn-auto: the %s profile turns on levers that are NOT measured yet - for A/B runs only\n", llama_fn_profile_name(prof));
    }
    if (prof == LLAMA_FN_PROFILE_FAST && llama_fn_fast_lever_count() == 0 && !(fast_list && fast_list[0])) {
        LLAMA_LOG_INFO("fn-auto:   fast: no lever has won a measured A/B yet, so fast is the same as safe\n");
    }
    if (st->promote) {
        LLAMA_LOG_INFO("fn-auto:   placement: the routed experts of all %d trunk layers stay on the host and the hot set holds "
                "the hot ones in VRAM (LLAMA_FN_PLACEMENT=user keeps the overrides as given)\n", n_layer);
    }
    if (const char * hp = llama_fn_env(model, "LLAMA_MOE_HOT_PROFILE"); hp && strcmp(hp, "even") == 0 && !getenv("LLAMA_MOE_HOT_PROFILE")) {
        LLAMA_LOG_INFO("fn-auto:   hot set: no %s.moeprof routing profile, so every layer gets the same number of empty slots that "
                "the adaptive set fills (the measured r1 win started from a routing profile)\n", fname.c_str());
    }
    std::string own, env, injected;
    for (const auto & o : st->opts) {
        const bool ours = std::find(st->injected.begin(), st->injected.end(), o.name) != st->injected.end() ||
                          std::find(st->cpu_set.begin(),  st->cpu_set.end(),  o.name) != st->cpu_set.end();
        const char * e = getenv(o.name.c_str());
        const char * alias = nullptr;
        if (!e && !ours) {
            e = fn_alias_value(o.name, &alias); // [TAG_FN_R4_REVIEW]
        }
        std::string & dst = o.inject ? (ours ? injected : env) : (e ? env : own);
        if (o.inject && !ours && !e && fn_cpu_switch(o.name)) {
            continue; // no CPU setter: warned above
        }
        dst += " " + std::string(alias ? alias : o.name.c_str()) + "=" + (e && !ours ? std::string(e) : o.value);
    }
    if (!own.empty()) {
        LLAMA_LOG_INFO("fn-auto:   profile:%s\n", own.c_str());
    }
    if (!injected.empty()) {
        LLAMA_LOG_INFO("fn-auto:   set while the model is loaded:%s\n", injected.c_str());
    }
    if (!env.empty()) {
        LLAMA_LOG_INFO("fn-auto:   kept from the environment:%s\n", env.c_str());
    }
}

void llama_fn_auto_after_load(llama_model & model) {
    llama_fn_auto_state * st = model.fn_auto.get();
    if (!st || !st->active) {
        return;
    }
    int n = 0;
    for (int il = 0; il < st->n_layer && il < (int) model.layers.size(); ++il) {
        const auto & l = model.layers[il];
        if (l.ffn_up_exps && l.ffn_up_exps->buffer && ggml_backend_buffer_is_host(l.ffn_up_exps->buffer) &&
                l.ffn_gate_inp && l.ffn_gate_inp->buffer && !ggml_backend_buffer_is_host(l.ffn_gate_inp->buffer)) {
            n++;
        }
    }
    st->n_host = n;
    if (n == 0) {
        LLAMA_LOG_WARN("fn-auto: no trunk layer has host experts with its router on a device - the automatic defaults are off\n");
        llama_fn_state_undo(*st);
        st->active = false;
        return;
    }
    LLAMA_LOG_INFO("fn-auto: %d of %d trunk layers run their routed experts from the host with the router on a device%s\n",
            n, st->n_layer, st->mtp_loaded ? "; the MTP block is loaded" : "");
}

void llama_fn_auto_on_free(llama_model & model) {
    if (model.fn_auto) {
        llama_fn_state_undo(*model.fn_auto);
        model.fn_auto->active = false;
    }
    std::lock_guard<std::mutex> lk(g_ctx_mtx);
    g_ctxs.erase(&model);
}

const char * llama_fn_env(const llama_model & model, const char * name) {
    return llama_fn_state_env(model.fn_auto.get(), name);
}

bool llama_fn_active(const llama_model & model) {
    return model.fn_auto && model.fn_auto->active;
}

void llama_fn_ctx_add(const llama_model & model, llama_context * ctx, int ctx_type) {
    std::lock_guard<std::mutex> lk(g_ctx_mtx);
    g_ctxs[&model].emplace_back(ctx, ctx_type);
}

void llama_fn_ctx_remove(const llama_model & model, llama_context * ctx) {
    std::lock_guard<std::mutex> lk(g_ctx_mtx);
    auto it = g_ctxs.find(&model);
    if (it == g_ctxs.end()) {
        return;
    }
    auto & v = it->second;
    v.erase(std::remove_if(v.begin(), v.end(), [ctx](const std::pair<llama_context *, int> & p) { return p.first == ctx; }), v.end());
    if (v.empty()) {
        g_ctxs.erase(it);
    }
}

std::vector<std::pair<llama_context *, int>> llama_fn_ctx_list(const llama_model & model) {
    std::lock_guard<std::mutex> lk(g_ctx_mtx);
    auto it = g_ctxs.find(&model);
    return it == g_ctxs.end() ? std::vector<std::pair<llama_context *, int>>() : it->second;
}

// public (llama-ext.h) ------------------------------------------------------------------------------------------------

const char * llama_model_fn_env(const struct llama_model * model, const char * name) {
    return model ? llama_fn_env(*model, name) : getenv(name);
}

const char * llama_model_fn_profile(const struct llama_model * model) {
    return model && llama_fn_active(*model) ? llama_fn_profile_name(model->fn_auto->profile) : nullptr;
}
