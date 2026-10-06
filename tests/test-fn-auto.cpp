// [TAG_FN_AUTO] [TAG_FN_VRAM_FIT] [TAG_FN_RAM_FIT] CPU unit test of the automatic qwen4exp defaults (src/llama-fn-auto.h):
// profile names and option lists, which overrides put routed experts on the host, the lookup order (environment first),
// putting the fast levers into the environment and taking them out again (reference counted, user values kept), the VRAM
// fit ceiling / budget / even slot counts at the 262K Flash-Next numbers, and the RAM fit plan.
// No GPU, no model. Exit code 0 iff every check passes.

#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "../src/llama-fn-auto.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

static int g_fail   = 0;
static int g_checks = 0;

#define TCHECK(cond, ...)                                        \
    do {                                                         \
        ++g_checks;                                              \
        if (!(cond)) {                                           \
            ++g_fail;                                            \
            fprintf(stderr, "FAIL %s:%d: ", __func__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                        \
            fprintf(stderr, "\n");                               \
        }                                                        \
    } while (0)

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

static constexpr size_t MiB = 1024 * 1024;
static constexpr uint64_t GiB = 1024ull * 1024 * 1024;

static const llama_fn_opt * find_opt(const std::vector<llama_fn_opt> & v, const char * name) {
    for (const auto & o : v) {
        if (o.name == name) {
            return &o;
        }
    }
    return nullptr;
}

static void test_profile_names() {
    bool ok = false;
    TCHECK(llama_fn_profile_parse(nullptr, &ok) == LLAMA_FN_PROFILE_SAFE && ok, "unset = safe");
    TCHECK(llama_fn_profile_parse("", &ok) == LLAMA_FN_PROFILE_SAFE && ok, "empty = safe");
    TCHECK(llama_fn_profile_parse("off", &ok) == LLAMA_FN_PROFILE_OFF && ok, "off");
    TCHECK(llama_fn_profile_parse("0", &ok) == LLAMA_FN_PROFILE_OFF && ok, "0");
    TCHECK(llama_fn_profile_parse(" Safe ", &ok) == LLAMA_FN_PROFILE_SAFE && ok, "safe, case and spaces");
    TCHECK(llama_fn_profile_parse("FAST", &ok) == LLAMA_FN_PROFILE_FAST && ok, "fast");
    TCHECK(llama_fn_profile_parse("trial", &ok) == LLAMA_FN_PROFILE_TRIAL && ok, "trial");
    TCHECK(llama_fn_profile_parse("trial-dma", &ok) == LLAMA_FN_PROFILE_TRIAL_DMA && ok, "trial-dma");
    TCHECK(llama_fn_profile_parse("trial_dma", &ok) == LLAMA_FN_PROFILE_TRIAL_DMA && ok, "trial_dma");
    TCHECK(llama_fn_profile_parse("fast-dma", &ok) == LLAMA_FN_PROFILE_SAFE && !ok, "fast-dma is gone (trial-dma), not ok");
    TCHECK(llama_fn_profile_parse("turbo", &ok) == LLAMA_FN_PROFILE_SAFE && !ok, "unknown = safe, not ok");
    TCHECK(strcmp(llama_fn_profile_name(LLAMA_FN_PROFILE_TRIAL_DMA), "trial-dma") == 0, "name");
    TCHECK(strcmp(llama_fn_profile_name(LLAMA_FN_PROFILE_TRIAL), "trial") == 0, "name trial");

    std::vector<std::pair<std::string, std::string>> l;
    TCHECK(llama_fn_parse_opt_list(" A=1, B = x/y ,,C=", l) && l.size() == 3, "list parse");
    TCHECK(l.size() == 3 && l[0].first == "A" && l[0].second == "1" && l[1].first == "B" && l[1].second == "x/y" &&
           l[2].first == "C" && l[2].second.empty(), "list items");
    TCHECK(!llama_fn_parse_opt_list("A=1,noeq,=2", l) && l.size() == 1, "bad items reported, good ones kept");
    TCHECK(llama_fn_parse_opt_list(nullptr, l) && l.empty(), "null list");
}

static void test_profile_opts() {
    TCHECK(llama_fn_profile_opts(LLAMA_FN_PROFILE_OFF, nullptr).empty(), "off has no options");

    // [TAG_FN_SHIP1] safe = every measured winner: the hot-set items (lookups) and the lever-round-1 set at 262K
    const auto safe = llama_fn_profile_opts(LLAMA_FN_PROFILE_SAFE, nullptr);
    const char * lookup_names[] = { "LLAMA_FN_PLACEMENT", "LLAMA_MOE_HOT_PROFILE", "LLAMA_MOE_HOT_MIB", "LLAMA_MOE_HOT_FIT",
                                    "LLAMA_MOE_HOT_ADAPT", "LLAMA_MOE_HOT_ADMIT", "LLAMA_RAM_FIT",
                                    "LLAMA_MOE_HOT_HEADROOM_MIB", "LLAMA_MOE_HOT_DECAY", "LLAMA_MOE_HOT_SEED",
                                    "LLAMA_PLE_DIO_FILE",
                                    "LLAMA_FN_CBUF", "LLAMA_MOE_HOT_DECAY_RATIO", "LLAMA_MOE_HOT_DECAY_HYST", "LLAMA_MOE_HOT_DECAY_MIB", "LLAMA_MOE_HOT_BURST_MIB", "LLAMA_MOE_HOT_BURST_RATIO", "LLAMA_FN_GPU", "LLAMA_FN_GPU_IDXQ8", "LLAMA_FN_GPU_DEFER", "LLAMA_FN_GPU_CONVWB", "LLAMA_FN_GPU_HCFUSE", "LLAMA_FN_GPU_Q8F", "LLAMA_FN_HOST_EXEC", "LLAMA_FN_HOST_STEP", "LLAMA_FN_HOST_QOS", "LLAMA_FN_HOST_POKE", "LLAMA_FN_HOST_REARM", "LLAMA_FN_HOST_DMAOFF", "LLAMA_FN_L4_POST", "LLAMA_FN_L4_GDNSTALL", "LLAMA_FN_L4_HOIST", "LLAMA_FN_L4_QSA_SEL", "LLAMA_FN_L4_QSA_KVW", "LLAMA_FN_L4_QSA_POOL", "LLAMA_FN_L4_QSA_LIST", "LLAMA_FN_L4_QSA_QKV", "LLAMA_FN_L4_POOLBAR", "LLAMA_FN_L4_PFDEV", "LLAMA_FN_L4_HOST", "LLAMA_FN_L4_HC", "LLAMA_FN_L4_HCQ8", "LLAMA_FN_L4_HOST_MTPFUSE", "LLAMA_FN_L6_L2PF", "LLAMA_FN_L6_HCCOMB" }; // [TAG_FN_L4_ADOPT] [TAG_FN_L5_ADOPT] [TAG_FN_L6_ADOPT]
    const char * lever_names[] = { "LLAMA_MOE_BRIDGE", "LLAMA_MOE_BRIDGE_DMA", "LLAMA_MOE_BRIDGE_PF", "LLAMA_MOE_BRIDGE_PF_SOLO",
                                   "LLAMA_MOE_DMA_SHARE", "GGML_CPU_APPLY_ONCE",
                                   "GGML_CPU_Q5_1_AVX512", "GGML_CPU_MMID_MR", "GGML_CPU_MOE_FUSE", "GGML_CPU_VNNI",
                                   "GGML_SCHED_SPLIT_ASYNC", "GGML_CUDA_GRAPH_POKE", "LLAMA_NO_ECOQOS",
                                   "LLAMA_GRAPH_PER_WIDTH", "LLAMA_PLE_HOST_GATHER", "LLAMA_PLE_DIRECT_IO",
                                   "LLAMA_QSA_POS_MASK", "LLAMA_QSA_POS_CHUNK", "TURBO_QSA_CHUNK", "TURBO_QSA_SPARSE",
                                   "TURBO_QSA_TOPK_UNORDERED", "SPEC_MTP_COST", "LLAMA_MTP_ATTN_WINDOW",
                                   "LLAMA_MTP_HEAD_ROWS", "SPEC_DFT_UBATCH", "LLAMA_PREFILL_STREAM",
                                   "LLAMA_PREFILL_STREAM_LEND", "LLAMA_PREFILL_STREAM_THREADS",
                                   "SPEC_MTP_BLOCK_VERIFY", "LLAMA_MOE_POOL_SPLIT", "LLAMA_MOE_POOL_PF_STREAMS", "LLAMA_MOE_POOL_SWPF", "LLAMA_MOE_BRIDGE_PF_RANK", "LLAMA_MOE_POOL_EXEC_CPU" };
    const size_t n_lookup = sizeof(lookup_names)/sizeof(lookup_names[0]);
    const size_t n_lever  = sizeof(lever_names)/sizeof(lever_names[0]);
    TCHECK(safe.size() == n_lookup + n_lever, "safe = the measured winners (%zu, want %zu)", safe.size(), n_lookup + n_lever);
    for (const char * n : lookup_names) {
        const llama_fn_opt * o = find_opt(safe, n);
        TCHECK(o && !o->inject, "safe %s present and read through the lookup, never the environment", n);
    }
    for (const char * n : lever_names) {
        const llama_fn_opt * o = find_opt(safe, n);
        TCHECK(o && o->inject, "safe lever %s present and put into the environment", n);
    }
    // the values the lever-round-1 A/B measured (E:/turbot-gates/flashnext/test/BEST.json)
    struct { const char * name; const char * value; } want[] = {
        { "LLAMA_MOE_HOT_ADMIT", "2/32" }, { "LLAMA_MOE_HOT_MIB", "auto" }, { "LLAMA_MOE_HOT_ADAPT", "1" },
        { "LLAMA_MOE_HOT_HEADROOM_MIB", "1280" }, { "LLAMA_MOE_HOT_DECAY", "0.95" }, { "LLAMA_MOE_HOT_SEED", "0.03" },
        { "LLAMA_MOE_BRIDGE", "1" }, { "LLAMA_MOE_BRIDGE_DMA", "1" }, { "LLAMA_MOE_DMA_SHARE", "auto" },
        { "LLAMA_MOE_BRIDGE_PF", "1" }, { "LLAMA_MOE_BRIDGE_PF_SOLO", "1" }, // [TAG_FN_R2_BRIDGE_PF] r2/ab4
        { "LLAMA_QSA_POS_MASK", "1" }, { "LLAMA_QSA_POS_CHUNK", "512" }, { "TURBO_QSA_CHUNK", "512" },
        { "TURBO_QSA_SPARSE", "1" }, { "LLAMA_MTP_ATTN_WINDOW", "32768" }, { "LLAMA_MTP_HEAD_ROWS", "98304" },
        { "SPEC_DFT_UBATCH", "128" }, { "SPEC_MTP_COST", "0" }, { "LLAMA_FN_CBUF", "1" }, { "LLAMA_FN_GPU_IDXQ8", "1" }, // [TAG_FN_L3_ADOPT] [TAG_FN_L4_ADOPT] on since its fix
        { "LLAMA_PREFILL_STREAM_LEND", "1" }, { "LLAMA_PREFILL_STREAM_THREADS", "16" },
        { "LLAMA_FN_L4_GDNSTALL", "1" }, { "LLAMA_FN_L4_PFDEV", "2" }, { "LLAMA_FN_L4_HOST", "1" }, // [TAG_FN_L4_ADOPT]
    };
    for (const auto & w : want) {
        const llama_fn_opt * o = find_opt(safe, w.name);
        TCHECK(o && o->value == w.value, "safe %s = %s", w.name, w.value);
    }
    const llama_fn_opt * hp  = find_opt(safe, "LLAMA_MOE_HOT_PROFILE");
    const llama_fn_opt * ple = find_opt(safe, "LLAMA_PLE_DIO_FILE");
    TCHECK(hp && hp->value.empty() && ple && ple->value.empty(), "the sidecar paths are set at load");
    TCHECK(find_opt(safe, "LLAMA_MOE_HOT_STATS") == nullptr && find_opt(safe, "LLAMA_MOE_BRIDGE_STATS") == nullptr &&
           find_opt(safe, "LLAMA_FLASHNEXT_PROFILE") == nullptr, "no stats switch and no profile switch in a profile");

    // fast = safe: every measured lever is in safe
    const auto fast = llama_fn_profile_opts(LLAMA_FN_PROFILE_FAST, nullptr);
    TCHECK(llama_fn_fast_lever_count() == 0, "no lever only in fast (%d)", llama_fn_fast_lever_count());
    TCHECK(fast.size() == safe.size(), "fast = safe");

    // trial = safe + the levers that are not measured yet: none today
    const auto trial = llama_fn_profile_opts(LLAMA_FN_PROFILE_TRIAL, nullptr);
    TCHECK(trial.size() == safe.size(), "trial = safe today (%zu vs %zu)", trial.size(), safe.size());
    for (const auto & o : safe) {
        TCHECK(find_opt(fast, o.name.c_str()) && find_opt(trial, o.name.c_str()), "fast and trial keep %s", o.name.c_str());
    }

    // trial-dma = trial without the host bridge: the gen5 DMA share in its own CPU split
    const auto tdma = llama_fn_profile_opts(LLAMA_FN_PROFILE_TRIAL_DMA, nullptr);
    TCHECK(find_opt(tdma, "LLAMA_MOE_BRIDGE") == nullptr && find_opt(tdma, "LLAMA_MOE_BRIDGE_DMA") == nullptr &&
           find_opt(tdma, "LLAMA_MOE_BRIDGE_PF") == nullptr && find_opt(tdma, "LLAMA_MOE_BRIDGE_PF_SOLO") == nullptr,
           "trial-dma has no bridge (and none of its switches)");
    TCHECK(find_opt(tdma, "LLAMA_MOE_DMA_SHARE") && find_opt(tdma, "LLAMA_MOE_DMA_SHARE")->value == "auto", "trial-dma DMA share");
    TCHECK(find_opt(tdma, "GGML_CPU_MOE_FUSE") != nullptr && tdma.size() == safe.size() - 14, "trial-dma keeps the rest"); // [TAG_FN_L4_ADOPT] + POOLBAR, PFDEV (bridge only)

    // safe ignores LLAMA_FLASHNEXT_FAST
    const auto safe_c = llama_fn_profile_opts(LLAMA_FN_PROFILE_SAFE, "GGML_CPU_MOE_FUSE=0,LLAMA_FN_TEST_X=1");
    const llama_fn_opt * fuse = find_opt(safe_c, "GGML_CPU_MOE_FUSE");
    TCHECK(fuse && fuse->value == "1" && find_opt(safe_c, "LLAMA_FN_TEST_X") == nullptr && safe_c.size() == safe.size(),
           "safe ignores a lever list");

    // LLAMA_FLASHNEXT_FAST in fast / trial / trial-dma changes a value, adds a name and knows the lookup names; the safe
    // levers stay (only levers outside safe are replaced, none today)
    const auto cust = llama_fn_profile_opts(LLAMA_FN_PROFILE_TRIAL,
            "GGML_CPU_MOE_FUSE=0, LLAMA_MOE_HOT_ADMIT=3/16, LLAMA_MOE_HOT_ADAPT_MIB=128, LLAMA_FN_TEST_X=1");
    const llama_fn_opt * cf = find_opt(cust, "GGML_CPU_MOE_FUSE");
    const llama_fn_opt * ca = find_opt(cust, "LLAMA_MOE_HOT_ADMIT");
    const llama_fn_opt * cm = find_opt(cust, "LLAMA_MOE_HOT_ADAPT_MIB");
    const llama_fn_opt * cx = find_opt(cust, "LLAMA_FN_TEST_X");
    TCHECK(find_opt(cust, "LLAMA_MOE_BRIDGE") != nullptr, "a custom list keeps the safe bridge");
    TCHECK(cf && cf->value == "0" && cf->inject, "custom list changes a lever");
    TCHECK(ca && ca->value == "3/16", "custom list changes a safe value");
    TCHECK(cm && !cm->inject, "hot names stay lookups");
    TCHECK(cx && !cx->inject, "LLAMA_FN_ names stay lookups");
    TCHECK(cust.size() == safe.size() + 2, "two names added (%zu)", cust.size());
    const auto bad = llama_fn_profile_opts(LLAMA_FN_PROFILE_TRIAL, "noequals");
    TCHECK(bad.size() == trial.size() && find_opt(bad, "LLAMA_MOE_BRIDGE") != nullptr, "a bad custom list leaves the built-in list");
    const auto cust_fast = llama_fn_profile_opts(LLAMA_FN_PROFILE_FAST, "TURBO_QSA_CHUNK=0");
    const llama_fn_opt * cq = find_opt(cust_fast, "TURBO_QSA_CHUNK");
    TCHECK(cq && cq->value == "0" && cq->inject, "fast takes a lever list too");
}

static void test_host_expert_layers() {
    // --n-cpu-moe 39 (common.h llm_add_n_cpu_ffn_overrides with LLM_FFN_EXPS_REGEX)
    std::vector<std::pair<std::string, bool>> nc39;
    for (int i = 0; i < 39; ++i) {
        nc39.emplace_back("blk\\." + std::to_string(i) + "\\.ffn_(up|down|gate|gate_up)_(ch|)exps", true);
    }
    TCHECK(llama_fn_count_host_expert_layers(nc39, 48) == 39, "--n-cpu-moe 39");
    TCHECK(llama_fn_count_host_expert_layers({ { "\\.ffn_(up|down|gate|gate_up)_(ch|)exps", true } }, 48) == 48, "--cpu-moe");
    TCHECK(llama_fn_count_host_expert_layers({ { "exps", true } }, 48) == 48, "-ot exps=CPU");
    TCHECK(llama_fn_count_host_expert_layers({}, 48) == 0, "no overrides");
    TCHECK(llama_fn_count_host_expert_layers({ { "\\.ffn_(up|down|gate)\\.", true } }, 48) == 0, "dense FFN override only");
    TCHECK(llama_fn_count_host_expert_layers({ { "per_layer_token_embd", true } }, 48) == 0, "PLE override only");
    // the first matching override decides: layers 40-47 pinned to a device first
    TCHECK(llama_fn_count_host_expert_layers({ { "blk\\.4[0-7]\\.ffn_.*_exps", false }, { "exps", true } }, 48) == 40,
           "device override first");
    TCHECK(llama_fn_count_host_expert_layers({ { "blk\\.1\\.ffn_down_exps", true } }, 48) == 1, "one tensor of one layer");
    TCHECK(llama_fn_count_host_expert_layers({ { "blk\\.1\\.", true } }, 48) == 1, "blk.1 does not match blk.10-19");
    TCHECK(llama_fn_count_host_expert_layers({ { "([", true }, { "exps", true } }, 48) == 48, "a bad pattern matches nothing");
    TCHECK(llama_fn_count_host_expert_layers({ { "blk\\.48\\.ffn_up_exps", true } }, 48) == 0, "the MTP block is not a trunk layer");
}

static void test_lookup() {
    set_env("LLAMA_FN_TEST_A", nullptr);
    set_env("LLAMA_FN_TEST_B", nullptr);
    llama_fn_auto_state st;
    st.active = true;
    st.opts.push_back({ "LLAMA_FN_TEST_A", "profile", false });
    TCHECK(llama_fn_state_env(&st, "LLAMA_FN_TEST_A") && strcmp(llama_fn_state_env(&st, "LLAMA_FN_TEST_A"), "profile") == 0,
           "profile value when the environment is unset");
    set_env("LLAMA_FN_TEST_A", "user");
    TCHECK(llama_fn_state_env(&st, "LLAMA_FN_TEST_A") && strcmp(llama_fn_state_env(&st, "LLAMA_FN_TEST_A"), "user") == 0,
           "the environment wins");
    set_env("LLAMA_FN_TEST_A", nullptr);
    TCHECK(llama_fn_state_env(&st, "LLAMA_FN_TEST_B") == nullptr, "not in the profile");
    st.active = false;
    TCHECK(llama_fn_state_env(&st, "LLAMA_FN_TEST_A") == nullptr, "inactive state: environment only");
    TCHECK(llama_fn_state_env(nullptr, "LLAMA_FN_TEST_A") == nullptr, "no state: environment only");
}

static void test_inject() {
    set_env("LLAMA_FN_TEST_INJ", nullptr);
    set_env("LLAMA_FN_TEST_USER", "mine");

    llama_fn_auto_state a;
    a.active = true;
    a.opts.push_back({ "LLAMA_FN_TEST_INJ",  "5",     true  });
    a.opts.push_back({ "LLAMA_FN_TEST_USER", "other", true  });
    a.opts.push_back({ "LLAMA_FN_TEST_LOOK", "x",     false });
    llama_fn_state_inject(a);
    TCHECK(getenv("LLAMA_FN_TEST_INJ") && strcmp(getenv("LLAMA_FN_TEST_INJ"), "5") == 0, "lever put into the environment");
    TCHECK(strcmp(getenv("LLAMA_FN_TEST_USER"), "mine") == 0, "user value kept");
    TCHECK(getenv("LLAMA_FN_TEST_LOOK") == nullptr, "lookup names never reach the environment");
    TCHECK(a.injected.size() == 1 && a.injected[0] == "LLAMA_FN_TEST_INJ", "only the injected name is recorded");

    llama_fn_auto_state b = a;
    b.injected.clear();
    llama_fn_state_inject(b); // a second model with the same profile holds the name too
    TCHECK(b.injected.size() == 1, "second holder recorded");
    llama_fn_state_undo(a);
    TCHECK(getenv("LLAMA_FN_TEST_INJ") && strcmp(getenv("LLAMA_FN_TEST_INJ"), "5") == 0, "still set while the second model lives");
    llama_fn_state_undo(b);
    TCHECK(getenv("LLAMA_FN_TEST_INJ") == nullptr, "removed with the last holder");
    TCHECK(strcmp(getenv("LLAMA_FN_TEST_USER"), "mine") == 0, "user value survives the undo");
    set_env("LLAMA_FN_TEST_USER", nullptr);

    // [TAG_FN_R4_REVIEW] LLAMA_MTP_WINDOW (Strata's name) is the user's LLAMA_MTP_ATTN_WINDOW: the trial default stays out
    if (!getenv("LLAMA_MTP_ATTN_WINDOW") && !getenv("LLAMA_MTP_WINDOW")) {
        llama_fn_auto_state w;
        w.active = true;
        w.opts.push_back({ "LLAMA_MTP_ATTN_WINDOW", "32768", true });
        set_env("LLAMA_MTP_WINDOW", "16384");
        llama_fn_state_inject(w);
        TCHECK(getenv("LLAMA_MTP_ATTN_WINDOW") == nullptr && w.injected.empty(), "the alias LLAMA_MTP_WINDOW keeps the default out");
        llama_fn_state_undo(w);
        set_env("LLAMA_MTP_WINDOW", nullptr);
        llama_fn_state_inject(w);
        TCHECK(getenv("LLAMA_MTP_ATTN_WINDOW") && strcmp(getenv("LLAMA_MTP_ATTN_WINDOW"), "32768") == 0, "without the alias the default goes in");
        llama_fn_state_undo(w);
        TCHECK(getenv("LLAMA_MTP_ATTN_WINDOW") == nullptr, "and comes out with the state");
    }

    // GGML_CPU_* switches go through ggml_cpu_fn_set_switch (ggml_cpu_init read the environment long before)
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CPU");
    using get_t = int (*)(enum ggml_cpu_fn_switch);
    get_t get = reg ? (get_t) ggml_backend_reg_get_proc_address(reg, "ggml_cpu_fn_get_switch") : nullptr;
    if (!get || getenv("GGML_CPU_MMID_MR")) {
        fprintf(stderr, "note: CPU switch part skipped (no CPU backend getter, or GGML_CPU_MMID_MR is set)\n");
        return;
    }
    const int before = get(GGML_CPU_FN_MMID_MR);
    llama_fn_auto_state c;
    c.active = true;
    c.opts.push_back({ "GGML_CPU_MMID_MR", "1", true });
    llama_fn_state_inject(c);
    TCHECK(get(GGML_CPU_FN_MMID_MR) == 1, "CPU switch set");
    TCHECK(getenv("GGML_CPU_MMID_MR") == nullptr, "CPU switch not through the environment");
    TCHECK(c.cpu_set.size() == 1, "CPU switch recorded");
    llama_fn_state_undo(c);
    TCHECK(get(GGML_CPU_FN_MMID_MR) == 0 && before == 0, "CPU switch back to off");

    // [TAG_FN_SHIP1] GGML_CPU_VNNI goes the same way (the switch; its kernels act only on a CPU with AVX512-VNNI)
    if (!getenv("GGML_CPU_VNNI")) {
        llama_fn_auto_state v;
        v.active = true;
        v.opts.push_back({ "GGML_CPU_VNNI", "1", true });
        llama_fn_state_inject(v);
        TCHECK(get(GGML_CPU_FN_VNNI) == 1 && getenv("GGML_CPU_VNNI") == nullptr, "VNNI switch set through the CPU backend");
        llama_fn_state_undo(v);
        TCHECK(get(GGML_CPU_FN_VNNI) == 0, "VNNI switch back to off");
    }
}

// bytes per expert (up + gate + down) of the 48 trunk layers: q4_K/q4_K/q5_1, and q8_0 down on layers 2, 4, 30, 46, 47
static std::vector<size_t> flash_next_expert_bytes() {
    std::vector<size_t> bytes;
    for (int il = 0; il < 48; ++il) {
        const bool q8 = il == 2 || il == 4 || il == 30 || il == 46 || il == 47;
        bytes.push_back(q8 ? 3584000 : 3072000);
    }
    return bytes;
}

static void test_vram_fit() {
    // RTX 5090: cudaMemGetInfo total 32607 MiB -> total - total/8 = 28531.1 MiB, rounded down to 256 MiB: 28416 MiB
    // (under the 28500 MiB the card is run at)
    const size_t t5090 = (size_t) 32607 * MiB;
    const size_t c5090 = llama_fn_vram_ceiling_default(t5090);
    TCHECK(c5090 == (t5090 - t5090/8) / (256 * MiB) * (256 * MiB), "5090 ceiling is total - total/8, on 256 MiB steps");
    TCHECK(c5090 / MiB == 28416, "5090 ceiling %zu MiB", c5090 / MiB);
    TCHECK(llama_fn_vram_ceiling_default((size_t) 8192 * MiB) == (size_t) (8192 - 1536) * MiB, "8 GB card keeps 1536 MiB");
    TCHECK(llama_fn_vram_ceiling_default((size_t) 1024 * MiB) == 0, "tiny card: no room");

    // 262K example: model 7.4 GB + turbot pool/indexer 3.0 + compute 2.7 + MTP draft 0.7 + other 1.8 = 15.6 GB used
    const size_t used = (size_t) 15607 * MiB;
    const size_t b = llama_fn_vram_fit_budget(t5090, t5090 - used, c5090, 768 * MiB);
    TCHECK(b == c5090 - used - 768 * MiB, "budget = ceiling - used - margin");
    TCHECK(b / MiB == 12041, "262K budget %zu MiB", b / MiB);
    TCHECK(llama_fn_vram_fit_budget(t5090, (size_t) 1000 * MiB, t5090, 768 * MiB) == 232 * MiB, "free - margin when the ceiling is the card");
    TCHECK(llama_fn_vram_fit_budget(t5090, (size_t) 3000 * MiB, c5090, 768 * MiB) == 0, "used above the ceiling: 0");
    TCHECK(llama_fn_vram_fit_budget(t5090, (size_t) 500 * MiB, t5090, 768 * MiB) == 0, "free below the margin: 0");
    TCHECK(llama_fn_vram_fit_budget(t5090, t5090 - used, c5090, 0) == c5090 - used, "no margin");
    TCHECK(llama_fn_even_slots(flash_next_expert_bytes(), b, 512) == 83, "262K budget as even slots: 83 per layer + the zero slot");
}

// [TAG_FN_AUTO] even slots at Flash-Next UD-Q4_K_XL
static void test_even_slots() {
    const size_t b = ((size_t) 32607 * MiB - ((size_t) 32607 * MiB) / 8) - (size_t) 15607 * MiB - 768 * MiB;
    // even slots at Flash-Next UD-Q4_K_XL: 43 layers q4_K/q4_K/q5_1 (3,072,000 B), 5 layers with q8_0 down (3,584,000 B)
    const std::vector<size_t> bytes = flash_next_expert_bytes();
    const int32_t n = llama_fn_even_slots(bytes, b, 512);
    TCHECK(n == 83, "262K even slots per layer %d", n);
    size_t total = 0;
    for (size_t x : bytes) {
        total += x * (size_t) (n + 1); // every layer also holds its zero slot
    }
    TCHECK(total <= b, "even slots and the zero slots fit the budget");
    size_t sum = 0;
    for (size_t x : bytes) {
        sum += x;
    }
    TCHECK(llama_fn_even_slots(bytes, sum, 512) == 0, "a budget of one row per layer is the zero slot alone: 0");
    TCHECK(llama_fn_even_slots(bytes, 2*sum, 512) == 1, "two rows per layer: one slot");
    TCHECK(llama_fn_even_slots(bytes, (size_t) 1 << 50, 512) == 512, "at most every expert");
    TCHECK(llama_fn_even_slots(bytes, 100 * MiB, 512) == 0, "a budget below one expert per layer gives 0");
    TCHECK(llama_fn_even_slots({}, b, 512) == 0, "no layers");
}

// [TAG_FN_L3_VRAM_CBUF] the compute-buffer lend at the r2/ab3 numbers (file A, -c 262144, -ub 8192): FULL reserve
// 3725.44 MiB, a SMALL one of about 250 MiB, the 10312 MiB hot-set budget of today's fit, the stream's 2 x 1950 MiB banks
static void test_cbuf() {
    TCHECK(llama_fn_cbuf_small_t(31, 32, 1) == 31, "default: 31 below the op-offload minimum 32");
    TCHECK(llama_fn_cbuf_small_t(100, 32, 1) == 31, "never at or above the op-offload minimum");
    TCHECK(llama_fn_cbuf_small_t(4, 32, 1) == 4, "below the hot chain's 8 tokens on request (MTP n_max 2: 3)");
    TCHECK(llama_fn_cbuf_small_t(3, 32, 1) == 3, "3: the MTP verify of n_max 2");
    TCHECK(llama_fn_cbuf_small_t(3, 32, 2) == 2, "3 with 2 sequences: 2 (a multiple of n_seqs)");
    TCHECK(llama_fn_cbuf_small_t(1, 32, 1) == 1, "1: decode without drafts");
    TCHECK(llama_fn_cbuf_small_t(31, 16, 1) == 15, "GGML_OP_OFFLOAD_MIN_BATCH=16 -> 15");
    TCHECK(llama_fn_cbuf_small_t(31, 0, 1) == 8, "a nonsense minimum still leaves 8");
    TCHECK(llama_fn_cbuf_small_t(31, 32, 4) == 28, "4 sequences: 28 (the reserve rounds 31 up to 32, an op-offload graph)");
    TCHECK(llama_fn_cbuf_small_t(31, 32, 0) == 31, "n_seqs 0 counts as 1");
    TCHECK(llama_fn_cbuf_small_t(31, 32, 40) == 40, "more sequences than 31: one token each (FULL - SMALL then decides)");

    const size_t gran  = 2 * MiB;
    const size_t full  = (size_t) (3725.44 * MiB);
    const size_t small = 250 * MiB;
    const size_t tail  = llama_fn_cbuf_tail(full, small, 0, gran, 256 * MiB);
    TCHECK(tail == 3476 * MiB, "tail = FULL - SMALL on granules: %zu MiB", tail / MiB);
    TCHECK(tail % gran == 0, "tail on a granule");
    TCHECK(llama_fn_cbuf_tail(full, small, 300 * MiB, gran, 256 * MiB) == 3776 * MiB, "plus the pool's growth");
    TCHECK(llama_fn_cbuf_tail(500 * MiB, 400 * MiB, 0, gran, 256 * MiB) == 0, "under the minimum gain: no tail");
    TCHECK(llama_fn_cbuf_tail(400 * MiB, 500 * MiB, 0, gran, 0) == 0, "SMALL above FULL: no tail");

    // the hot set grows by the difference: 71 -> 95 even slots per layer (+33 %), 13 top layers form the tail
    const std::vector<size_t> bytes = flash_next_expert_bytes();
    const size_t b_old = 10312 * MiB;
    const size_t b_new = b_old + (full - small);
    const int32_t n_old = llama_fn_even_slots(bytes, b_old, 512);
    const int32_t n_new = llama_fn_even_slots(bytes, b_new - 5*gran, 512); // the layout's pads and guards
    TCHECK(n_old == 71 && n_new == 95, "even slots %d -> %d", n_old, n_new);
    std::vector<size_t> layer_bytes;
    for (size_t x : bytes) {
        layer_bytes.push_back(x * (size_t) (n_new + 1));
    }
    const int k = llama_fn_cbuf_first_tail_layer(layer_bytes, tail);
    TCHECK(k == 35, "the tail starts at layer %d", k);
    size_t in_tail = 0;
    for (size_t i = (size_t) std::max(k, 0); i < layer_bytes.size(); ++i) {
        in_tail += layer_bytes[i];
    }
    TCHECK(in_tail >= tail && in_tail - layer_bytes[(size_t) std::max(k, 0)] < tail, "whole layers, the fewest that hold it");
    TCHECK(llama_fn_cbuf_first_tail_layer(layer_bytes, 0) == -1, "no tail");
    TCHECK(llama_fn_cbuf_first_tail_layer(layer_bytes, layer_bytes.back()) == 47, "one layer's bytes: the last layer");
    TCHECK(llama_fn_cbuf_first_tail_layer(layer_bytes, layer_bytes.back() + 1) == 46, "a byte more: two layers");
    size_t all = 0;
    for (size_t x : layer_bytes) {
        all += x;
    }
    TCHECK(llama_fn_cbuf_first_tail_layer(layer_bytes, all) == -1, "every layer: refused (one stays below the tail)");
    TCHECK(llama_fn_cbuf_first_tail_layer(layer_bytes, all - layer_bytes[0]) == 1, "all but the first layer");
    TCHECK(llama_fn_cbuf_first_tail_layer({}, tail) == -1, "no layers");

    TCHECK(llama_fn_cbuf_fits(b_new, tail, 3900 * MiB), "13.8 GB hot set: the tail beside the stream's banks fits");
    TCHECK(!llama_fn_cbuf_fits(7 * 1024 * MiB, tail, 3900 * MiB), "7 GB: it does not");
    TCHECK(!llama_fn_cbuf_fits(b_new, 0, 3900 * MiB), "no tail: nothing to fit");

    // LLAMA_FN_CBUF_POOL_MIB also grows the budget: up to free - 256 MiB and the LLAMA_FN_HOT_BUDGET_MAX_MIB cap
    const size_t free_v = 15000 * MiB;
    TCHECK(llama_fn_cbuf_budget(b_new, 0, free_v, SIZE_MAX) == b_new, "no extra: the fit's budget");
    TCHECK(llama_fn_cbuf_budget(b_new, 512 * MiB, free_v, SIZE_MAX) == b_new + 512 * MiB, "+512 MiB");
    TCHECK(llama_fn_cbuf_budget(b_new, 4096 * MiB, free_v, SIZE_MAX) == free_v - 256 * MiB, "never above free - 256 MiB");
    TCHECK(llama_fn_cbuf_budget(free_v, 512 * MiB, free_v, SIZE_MAX) == free_v, "a budget already above it stays");
    TCHECK(llama_fn_cbuf_budget(b_new, 512 * MiB, free_v, 8000 * MiB) == 8000 * MiB, "the cap wins");
    TCHECK(llama_fn_cbuf_budget(b_new, 512 * MiB, 0, SIZE_MAX) == b_new, "free unknown (0): the fit's budget");
}

static void test_ram_fit() {
    // Flash-Next on a 96 GB box: host experts (72 GB) + PLE (28.8 GB) mapped, bigger than RAM on their own
    llama_ram_fit_in fn;
    fn.ram_total     = 96 * GiB;
    fn.model_host    = 101 * GiB;
    fn.host_buffers  = 3 * GiB;
    fn.cache_ram_mib = 8192;
    fn.ckpt_mib      = 2048;
    fn.n_slots       = 1;
    llama_ram_fit_out o = llama_ram_fit_plan(fn);
    TCHECK(o.over && o.changed, "Flash-Next passes 85%% of RAM");
    TCHECK(o.cache_ram_mib == 0, "prompt cache off (%lld)", (long long) o.cache_ram_mib);
    TCHECK(o.ckpt_mib == 512, "checkpoints down to the minimum (%lld)", (long long) o.ckpt_mib);
    TCHECK(o.limit == (uint64_t) (0.85 * (double) (96 * GiB)), "limit is 85%% of RAM");

    // the test servers' --cache-ram 16384 given by the user: kept; checkpoints still capped
    llama_ram_fit_in fu = fn;
    fu.cache_ram_mib = 16384;
    fu.cache_ram_set = true;
    o = llama_ram_fit_plan(fu);
    TCHECK(o.over && o.cache_ram_mib == 16384 && o.ckpt_mib == 512, "explicit --cache-ram kept");
    llama_ram_fit_in fc = fn;
    fc.ckpt_set = true;
    o = llama_ram_fit_plan(fc);
    TCHECK(o.over && o.ckpt_mib == 2048 && o.cache_ram_mib == 0, "explicit checkpoints kept");

    // Qwen3.8-27B fully on the GPU: nothing on the host, nothing changes
    llama_ram_fit_in q5;
    q5.ram_total     = 96 * GiB;
    q5.model_host    = 1 * GiB;
    q5.host_buffers  = 2 * GiB;
    q5.cache_ram_mib = 16384;
    q5.ckpt_mib      = 2048;
    q5.n_slots       = 4;
    o = llama_ram_fit_plan(q5);
    TCHECK(!o.over && !o.changed && o.cache_ram_mib == 16384 && o.ckpt_mib == 2048, "27B unchanged");

    // partial room: 64 GB RAM, 40 GB model, 4 GB buffers, 2 slots -> checkpoints kept, prompt cache shrunk to the rest
    llama_ram_fit_in pr;
    pr.ram_total     = 64 * GiB;
    pr.model_host    = 40 * GiB;
    pr.host_buffers  = 4 * GiB;
    pr.cache_ram_mib = 16384;
    pr.ckpt_mib      = 2048;
    pr.n_slots       = 2;
    o = llama_ram_fit_plan(pr);
    const int64_t room = ((int64_t) (0.85 * (double) (64 * GiB)) - (int64_t) (44 * GiB)) / (int64_t) MiB;
    TCHECK(o.over && o.ckpt_mib == 2048, "checkpoints fit (%lld)", (long long) o.ckpt_mib);
    TCHECK(o.cache_ram_mib == room - 2 * 2048, "prompt cache = room - checkpoints (%lld vs %lld)", (long long) o.cache_ram_mib,
           (long long) (room - 2 * 2048));

    // unlimited prompt cache (-1, not given explicitly) is bounded; a disabled one and disabled checkpoints stay off
    llama_ram_fit_in un = pr;
    un.cache_ram_mib = -1;
    o = llama_ram_fit_plan(un);
    TCHECK(o.over && o.cache_ram_mib == room - 2 * 2048, "unlimited prompt cache bounded");
    llama_ram_fit_in off = fn;
    off.cache_ram_mib = 0;
    off.ckpt_mib      = 0;
    o = llama_ram_fit_plan(off);
    TCHECK(o.over && o.cache_ram_mib == 0 && o.ckpt_mib == 0 && !o.changed, "off stays off");
    // never raised: a budget below the minimum is kept
    llama_ram_fit_in lo = fn;
    lo.ckpt_mib = 256;
    o = llama_ram_fit_plan(lo);
    TCHECK(o.ckpt_mib == 256, "a small budget is not raised to the minimum");
    // a prompt cache left below 256 MiB is turned off
    llama_ram_fit_in sm;
    sm.ram_total     = 100 * GiB;
    sm.model_host    = 84 * GiB + 900 * MiB;
    sm.host_buffers  = 0;
    sm.cache_ram_mib = 8192;
    sm.ckpt_mib      = 0;
    sm.n_slots       = 1;
    o = llama_ram_fit_plan(sm);
    TCHECK(o.over && o.cache_ram_mib == 0, "less than 256 MiB left: prompt cache off (%lld)", (long long) o.cache_ram_mib);
}

int main() {
    // no ggml_backend_load_all(): the backends are linked in (GGML_BACKEND_DL=OFF); run with CUDA_VISIBLE_DEVICES=-1
    test_profile_names();
    test_profile_opts();
    test_host_expert_layers();
    test_lookup();
    test_inject();
    test_even_slots();
    test_vram_fit();
    test_cbuf();
    test_ram_fit();

    printf("test-fn-auto: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
