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
    P_TRY   = P_TRIAL | P_TDMA,          // levers that are not measured yet (A/B runs only)
    P_BR    = P_SAFE | P_FAST | P_TRIAL, // [TAG_FN_SHIP1] the host bridge and what runs inside it: not trial-dma
};

struct fn_item {
    const char * name;
    const char * value;
    int          profiles;
};

// safe (the default) = the measured winners, all with file A (UD-Q4_K_XL + Q8_0 MTP head), turbot KV and MTP:
//   - placement auto + the adaptive hot set + the fits: speed/r1 hot48ad against F0 (ncmoe 39), 3 interleaved rounds at
//     32K: +35.4% greedy code, +45.1% temp-1 prose, KLD on the pure-placement band (r2/TEST_PLAN.md 0.4);
//   - [TAG_FN_SHIP1] lever round 1 at -c 262144 (E:/turbot-gates/flashnext/test/history.md): ALL-ON (the levers below)
//     against the hot set alone, same binary, 2 interleaved rounds: real use (temp 1, thinking, 32K / 131K / 246K filled)
//     +38.6 %, bench code +24.8 %, decode KLD on the placement band (0.0117 vs 0.0121); the bs3 binary search kept every
//     group (decay, host trims, CPU kernels, PLE direct I/O, -ub 8192); with the fix-r1 code (k-pool tail, wide
//     compaction, bridge rollback ring) real use 35.3 -> 58.6 t/s (+65.9 %), bench code 51.1 -> 64.8 t/s.
// fast: safe + levers that won only a narrower A/B (P_FAST). None: every measured lever is in safe.
// trial: safe + built levers that are NOT measured yet (P_TRY), for A/B runs only. None today.
// trial-dma: trial without the host bridge, i.e. the gen5 DMA share in its own CPU split (the pre-round-4 exclusive mode).
const fn_item k_items[] = {
    { "LLAMA_FN_PLACEMENT",           "auto",  P_ALL }, // every trunk layer's routed experts on the host
    { "LLAMA_MOE_HOT_PROFILE",        "",      P_ALL }, // set at load: <model>.moeprof if present, else "even"
    { "LLAMA_MOE_HOT_MIB",            "auto",  P_ALL },
    { "LLAMA_MOE_HOT_FIT",            "1",     P_ALL }, // [TAG_FN_VRAM_FIT]
    { "LLAMA_MOE_HOT_ADAPT",          "1",     P_ALL },
    { "LLAMA_MOE_HOT_ADMIT",          "2/32",  P_ALL },
    { "LLAMA_RAM_FIT",                "1",     P_ALL }, // [TAG_FN_RAM_FIT]

    // [TAG_FN_SHIP1] lever round 1 (see above). Lookups (llama_fn_env), never put into the environment:
    { "LLAMA_MOE_HOT_HEADROOM_MIB",   "1280",  P_ALL }, // the 768 default went over 28,500 MiB at depth (28,622)
    { "LLAMA_MOE_HOT_DECAY",          "0.92",  P_ALL }, // decayed hot set (bs3: without it -2.5 % code, -3.4 % prose)
    { "LLAMA_MOE_HOT_SEED",           "0.03",  P_ALL },
    { "LLAMA_PLE_DIO_FILE",           "",      P_ALL }, // set at load: <model>.ple (the unmapped PLE table copy)
    // read with getenv() elsewhere, so they are in the environment while the model is loaded:
    { "LLAMA_MOE_BRIDGE",             "1",     P_BR  }, // the host bridge (+ its rollback ring, [TAG_FN_R1_BRIDGE_RB])
    { "LLAMA_MOE_BRIDGE_DMA",         "1",     P_BR  }, // the DMA share inside the bridged graphs ([TAG_FN_R4_BRIDGE_DMA])
    // [TAG_FN_R2_BRIDGE_PF] lever round 2 (r2/ab4, 2 interleaved rounds vs the round-1 default, 262K, MTP): the next-layer
    // prefetch by real loads with the executor out of the compute - bench code +6.5 %, prose +5.8 %, real use +1.8 %;
    // exact (greedy identity, values never touched)
    { "LLAMA_MOE_BRIDGE_PF",          "1",     P_BR  },
    { "LLAMA_MOE_BRIDGE_PF_SOLO",     "1",     P_BR  },
    { "LLAMA_MOE_DMA_SHARE",          "auto",  P_ALL },
    { "GGML_CPU_APPLY_ONCE",          "1",     P_ALL },
    { "GGML_CPU_Q5_1_AVX512",         "1",     P_ALL },
    { "GGML_CPU_MMID_MR",             "1",     P_ALL },
    { "GGML_CPU_MOE_FUSE",            "1",     P_ALL },
    { "GGML_CPU_VNNI",                "1",     P_ALL }, // acts only on a CPU with AVX512-VNNI
    { "GGML_SCHED_SPLIT_ASYNC",       "1",     P_ALL },
    { "GGML_CUDA_GRAPH_POKE",         "1",     P_ALL },
    { "LLAMA_NO_ECOQOS",              "1",     P_ALL }, // applied at load: llama_backend_init() read it before
    { "LLAMA_GRAPH_PER_WIDTH",        "1",     P_ALL },
    { "LLAMA_PLE_HOST_GATHER",        "1",     P_ALL },
    { "LLAMA_PLE_DIRECT_IO",          "1",     P_ALL },
    { "LLAMA_QSA_POS_MASK",           "1",     P_ALL },
    { "LLAMA_QSA_POS_CHUNK",          "512",   P_ALL },
    { "TURBO_QSA_CHUNK",              "512",   P_ALL },
    { "TURBO_QSA_SPARSE",             "1",     P_ALL },
    { "TURBO_QSA_TOPK_UNORDERED",     "1",     P_ALL },
    { "SPEC_MTP_COST",                "1",     P_ALL },
    { "LLAMA_MTP_ATTN_WINDOW",        "32768", P_ALL },
    { "LLAMA_MTP_HEAD_ROWS",          "98304", P_ALL },
    { "SPEC_DFT_UBATCH",              "128",   P_ALL }, // the MTP draft context's ubatch (its logits buffer at -ub 8192)
    { "LLAMA_PREFILL_STREAM",         "1",     P_ALL },
    { "LLAMA_PREFILL_STREAM_LEND",    "1",     P_ALL },
    { "LLAMA_PREFILL_STREAM_THREADS", "16",    P_ALL },
};

// names the fn-auto aware code reads through llama_fn_env(): never put into the environment
// [TAG_FN_SHIP1] LLAMA_PLE_DIO_FILE as well: qwen4exp's loader tells the profile's default copy from the user's path
bool fn_is_lookup_name(const std::string & n) {
    return n.rfind("LLAMA_MOE_HOT_", 0) == 0 || n.rfind("LLAMA_FN_", 0) == 0 || n == "LLAMA_RAM_FIT" ||
           n == "LLAMA_PLE_DIO_FILE";
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
    { "GGML_CPU_VNNI",        GGML_CPU_FN_VNNI        }, // [TAG_FN_SHIP1]
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

// [TAG_FN_L3_VRAM_CBUF] ---------------------------------------------------------------------------------------------

uint32_t llama_fn_cbuf_small_t(uint32_t requested, int op_offload_min, uint32_t n_seqs) {
    const uint32_t below = (uint32_t) std::max(9, op_offload_min) - 1;
    const uint32_t t     = std::max<uint32_t>(8, std::min<uint32_t>(requested, below));
    n_seqs = std::max<uint32_t>(1, n_seqs);
    return std::max<uint32_t>(n_seqs, t / n_seqs * n_seqs);
}

size_t llama_fn_cbuf_tail(size_t full, size_t small, size_t pool_extra, size_t gran, size_t min_gain) {
    if (full < small + min_gain || full <= small) {
        return 0;
    }
    const size_t t = full - small + pool_extra;
    return gran > 0 ? (t + gran - 1) / gran * gran : t;
}

int llama_fn_cbuf_first_tail_layer(const std::vector<size_t> & layer_bytes, size_t tail_bytes) {
    if (tail_bytes == 0) {
        return -1;
    }
    size_t acc = 0;
    for (size_t k = layer_bytes.size(); k > 0; --k) {
        acc += layer_bytes[k - 1];
        if (acc >= tail_bytes) {
            return k - 1 >= 1 ? (int) (k - 1) : -1;
        }
    }
    return -1;
}

bool llama_fn_cbuf_fits(size_t budget, size_t tail, size_t stream_lend) {
    const size_t need = tail + stream_lend + budget/32 + ((size_t) 64 << 20);
    return tail > 0 && budget >= need;
}

size_t llama_fn_cbuf_budget(size_t budget, size_t pool_extra, size_t free, size_t budget_max) {
    const size_t keep = (size_t) 256 << 20;
    const size_t room = free > keep ? free - keep : 0;
    size_t b = budget + pool_extra;
    if (b > room) {
        b = std::max(budget, room);
    }
    return std::min(b, budget_max);
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
        // [TAG_FN_SHIP1] the unmapped copy of the PLE table that LLAMA_PLE_DIRECT_IO reads, next to the model: made on the
        // first load when its volume has room to spare, else the table is read through the mapping (qwen4exp.cpp)
        if (o.name == "LLAMA_PLE_DIO_FILE" && o.value.empty() && !fname.empty()) {
            o.value = fname + ".ple";
        }
    }
    model.fn_auto = st;

    const char * pl = llama_fn_env(model, "LLAMA_FN_PLACEMENT");
    st->promote = pl && strcmp(pl, "auto") == 0;
    if (st->promote) {
        ml.fn_host_experts_n_layer = n_layer;
    }

    llama_fn_state_inject(*st);
    // [TAG_FN_SHIP1] llama_backend_init() read LLAMA_NO_ECOQOS before any model existed: apply the profile's value now
    if (std::find(st->injected.begin(), st->injected.end(), "LLAMA_NO_ECOQOS") != st->injected.end()) {
        llama_ecoqos_opt_out();
    }

    LLAMA_LOG_INFO("fn-auto: qwen4exp with the routed experts of %d of %d trunk layers on the host: LLAMA_FLASHNEXT_PROFILE=%s%s "
            "(off|safe|fast|trial|trial-dma)\n", n_host_user, n_layer, llama_fn_profile_name(prof), pe ? "" : " (default)");
    if (prof == LLAMA_FN_PROFILE_TRIAL || prof == LLAMA_FN_PROFILE_TRIAL_DMA) {
        LLAMA_LOG_WARN("fn-auto: the %s profile is for A/B runs only (trial: safe + the levers not measured yet; trial-dma: "
                "trial without the host bridge)\n", llama_fn_profile_name(prof));
    }
    if (prof == LLAMA_FN_PROFILE_FAST && llama_fn_fast_lever_count() == 0 && !(fast_list && fast_list[0])) {
        LLAMA_LOG_INFO("fn-auto:   fast: every measured lever is in safe, so fast is the same as safe\n");
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
