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
    TCHECK(llama_fn_profile_parse("fast-dma", &ok) == LLAMA_FN_PROFILE_FAST_DMA && ok, "fast-dma");
    TCHECK(llama_fn_profile_parse("fast_dma", &ok) == LLAMA_FN_PROFILE_FAST_DMA && ok, "fast_dma");
    TCHECK(llama_fn_profile_parse("turbo", &ok) == LLAMA_FN_PROFILE_SAFE && !ok, "unknown = safe, not ok");
    TCHECK(strcmp(llama_fn_profile_name(LLAMA_FN_PROFILE_FAST_DMA), "fast-dma") == 0, "name");

    std::vector<std::pair<std::string, std::string>> l;
    TCHECK(llama_fn_parse_opt_list(" A=1, B = x/y ,,C=", l) && l.size() == 3, "list parse");
    TCHECK(l.size() == 3 && l[0].first == "A" && l[0].second == "1" && l[1].first == "B" && l[1].second == "x/y" &&
           l[2].first == "C" && l[2].second.empty(), "list items");
    TCHECK(!llama_fn_parse_opt_list("A=1,noeq,=2", l) && l.size() == 1, "bad items reported, good ones kept");
    TCHECK(llama_fn_parse_opt_list(nullptr, l) && l.empty(), "null list");
}

static void test_profile_opts() {
    TCHECK(llama_fn_profile_opts(LLAMA_FN_PROFILE_OFF, nullptr).empty(), "off has no options");

    const auto safe = llama_fn_profile_opts(LLAMA_FN_PROFILE_SAFE, nullptr);
    const char * safe_names[] = { "LLAMA_FN_PLACEMENT", "LLAMA_MOE_HOT_PROFILE", "LLAMA_MOE_HOT_MIB", "LLAMA_MOE_HOT_FIT",
                                  "LLAMA_MOE_HOT_ADAPT", "LLAMA_MOE_HOT_ADMIT", "LLAMA_RAM_FIT" };
    TCHECK(safe.size() == sizeof(safe_names)/sizeof(safe_names[0]), "safe = the measured winners only (%zu)", safe.size());
    for (const char * n : safe_names) {
        const llama_fn_opt * o = find_opt(safe, n);
        TCHECK(o && !o->inject, "safe %s present and read through the lookup, never the environment", n);
    }
    TCHECK(find_opt(safe, "LLAMA_MOE_HOT_ADMIT")->value == "2/32", "measured admission 2/32");
    TCHECK(find_opt(safe, "LLAMA_MOE_HOT_MIB")->value == "auto", "budget auto");
    TCHECK(find_opt(safe, "LLAMA_MOE_HOT_ADAPT")->value == "1", "adaptive");
    TCHECK(find_opt(safe, "LLAMA_MOE_BRIDGE") == nullptr, "no bridge in safe");

    const auto fast = llama_fn_profile_opts(LLAMA_FN_PROFILE_FAST, nullptr);
    TCHECK(fast.size() > safe.size(), "fast adds levers");
    for (const char * n : safe_names) {
        TCHECK(find_opt(fast, n) != nullptr, "fast keeps safe %s", n);
    }
    const char * fast_names[] = { "LLAMA_MOE_BRIDGE", "GGML_CPU_APPLY_ONCE", "GGML_CPU_Q5_1_AVX512", "GGML_CPU_MMID_MR",
                                  "GGML_CPU_MOE_FUSE", "GGML_SCHED_SPLIT_ASYNC", "LLAMA_PLE_HOST_GATHER", "LLAMA_PLE_DIRECT_IO",
                                  "LLAMA_GRAPH_PER_WIDTH", "TURBO_QSA_CHUNK", "SPEC_MTP_COST", "LLAMA_MTP_ATTN_WINDOW" };
    for (const char * n : fast_names) {
        const llama_fn_opt * o = find_opt(fast, n);
        TCHECK(o && o->inject, "fast %s present and put into the environment", n);
    }
    TCHECK(find_opt(fast, "LLAMA_MOE_DMA_SHARE") == nullptr, "fast uses the bridge, not the DMA share");

    const auto fdma = llama_fn_profile_opts(LLAMA_FN_PROFILE_FAST_DMA, nullptr);
    TCHECK(find_opt(fdma, "LLAMA_MOE_BRIDGE") == nullptr, "fast-dma has no bridge (exclusive)");
    TCHECK(find_opt(fdma, "LLAMA_MOE_DMA_SHARE") && find_opt(fdma, "LLAMA_MOE_DMA_SHARE")->value == "auto", "fast-dma DMA share");
    TCHECK(find_opt(fdma, "GGML_CPU_MOE_FUSE") != nullptr, "fast-dma keeps the CPU levers");

    // LLAMA_FLASHNEXT_FAST replaces the fast levers of "fast", may change a safe value, and knows lookup names
    const auto cust = llama_fn_profile_opts(LLAMA_FN_PROFILE_FAST, "GGML_CPU_MOE_FUSE=1, LLAMA_MOE_HOT_ADMIT=3/16, LLAMA_MOE_HOT_ADAPT_MIB=128");
    TCHECK(find_opt(cust, "LLAMA_MOE_BRIDGE") == nullptr, "custom list drops the built-in bridge");
    TCHECK(find_opt(cust, "GGML_CPU_MOE_FUSE") && find_opt(cust, "GGML_CPU_MOE_FUSE")->inject, "custom lever injected");
    TCHECK(find_opt(cust, "LLAMA_MOE_HOT_ADMIT")->value == "3/16", "custom list changes a safe value");
    TCHECK(find_opt(cust, "LLAMA_MOE_HOT_ADAPT_MIB") && !find_opt(cust, "LLAMA_MOE_HOT_ADAPT_MIB")->inject, "hot names stay lookups");
    TCHECK(find_opt(cust, "LLAMA_FN_PLACEMENT") != nullptr, "custom list keeps the safe items");
    const auto bad = llama_fn_profile_opts(LLAMA_FN_PROFILE_FAST, "noequals");
    TCHECK(find_opt(bad, "LLAMA_MOE_BRIDGE") != nullptr, "a bad custom list leaves the built-in fast list");
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
    // RTX 5090: cudaMemGetInfo total 32607 MiB -> ceiling 28531.1 MiB ("<= ~28.5 GB")
    const size_t t5090 = (size_t) 32607 * MiB;
    const size_t c5090 = llama_fn_vram_ceiling_default(t5090);
    TCHECK(c5090 == t5090 - t5090/8, "5090 ceiling is total - total/8");
    TCHECK(c5090 / MiB == 28531, "5090 ceiling %zu MiB", c5090 / MiB);
    TCHECK(llama_fn_vram_ceiling_default((size_t) 8192 * MiB) == (size_t) (8192 - 1536) * MiB, "8 GB card keeps 1536 MiB");
    TCHECK(llama_fn_vram_ceiling_default((size_t) 1024 * MiB) == 0, "tiny card: no room");

    // 262K example: model 7.4 GB + turbot pool/indexer 3.0 + compute 2.7 + MTP draft 0.7 + other 1.8 = 15.6 GB used
    const size_t used = (size_t) 15607 * MiB;
    const size_t b = llama_fn_vram_fit_budget(t5090, t5090 - used, c5090, 768 * MiB);
    TCHECK(b == c5090 - used - 768 * MiB, "budget = ceiling - used - margin");
    TCHECK(b / MiB == 12156, "262K budget %zu MiB", b / MiB);
    TCHECK(llama_fn_vram_fit_budget(t5090, (size_t) 1000 * MiB, t5090, 768 * MiB) == 232 * MiB, "free - margin when the ceiling is the card");
    TCHECK(llama_fn_vram_fit_budget(t5090, (size_t) 3000 * MiB, c5090, 768 * MiB) == 0, "used above the ceiling: 0");
    TCHECK(llama_fn_vram_fit_budget(t5090, (size_t) 500 * MiB, t5090, 768 * MiB) == 0, "free below the margin: 0");
    TCHECK(llama_fn_vram_fit_budget(t5090, t5090 - used, c5090, 0) == c5090 - used, "no margin");
    TCHECK(llama_fn_even_slots(flash_next_expert_bytes(), b, 512) == 84, "262K budget as even slots: 84 per layer");
}

// [TAG_FN_AUTO] even slots at Flash-Next UD-Q4_K_XL
static void test_even_slots() {
    const size_t b = ((size_t) 32607 * MiB - ((size_t) 32607 * MiB) / 8) - (size_t) 15607 * MiB - 768 * MiB;
    // even slots at Flash-Next UD-Q4_K_XL: 43 layers q4_K/q4_K/q5_1 (3,072,000 B), 5 layers with q8_0 down (3,584,000 B)
    const std::vector<size_t> bytes = flash_next_expert_bytes();
    const int32_t n = llama_fn_even_slots(bytes, b, 512);
    TCHECK(n == 84, "262K even slots per layer %d", n);
    size_t total = 0;
    for (size_t x : bytes) {
        total += x * (size_t) n;
    }
    TCHECK(total <= b, "even slots fit the budget");
    TCHECK(llama_fn_even_slots(bytes, (size_t) 1 << 50, 512) == 512, "at most every expert");
    TCHECK(llama_fn_even_slots(bytes, 100 * MiB, 512) == 0, "a budget below one expert per layer gives 0");
    TCHECK(llama_fn_even_slots({}, b, 512) == 0, "no layers");
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
    test_ram_fit();

    printf("test-fn-auto: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
