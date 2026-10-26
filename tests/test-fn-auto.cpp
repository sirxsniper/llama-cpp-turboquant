// [TAG_FN_AUTO] [TAG_FN_VRAM_FIT] [TAG_FN_RAM_FIT] CPU unit test of the automatic qwen4exp defaults (src/llama-fn-auto.h):
// profile names and option lists, which overrides put routed experts on the host, the lookup order (environment first),
// putting the fast levers into the environment and taking them out again (reference counted, user values kept), the VRAM
// fit ceiling / budget / even slot counts at the 262K Flash-Next numbers, and the RAM fit plan.
// No GPU, no model. Exit code 0 iff every check passes.

#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "../src/llama-fn-auto.h"
#include "../src/llama-fn-l22.h" // [TAG_FN_L22_P10] llama_fn_l22_offload_min

#include <algorithm>
#include <cmath>
#include <cstdint>
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

extern const size_t k_n_l22_rows_decl;
static void test_profile_opts() {
    TCHECK(llama_fn_profile_opts(LLAMA_FN_PROFILE_OFF, nullptr).empty(), "off has no options");

    // [TAG_FN_SHIP1] safe = every measured winner: the hot-set items (lookups) and the lever-round-1 set at 262K
    const auto safe = llama_fn_profile_opts(LLAMA_FN_PROFILE_SAFE, nullptr);
    const char * lookup_names[] = { "LLAMA_FN_PLACEMENT", "LLAMA_MOE_HOT_PROFILE", "LLAMA_MOE_HOT_MIB", "LLAMA_MOE_HOT_FIT",
                                    "LLAMA_MOE_HOT_ADAPT", "LLAMA_MOE_HOT_ADMIT", "LLAMA_RAM_FIT",
                                    "LLAMA_MOE_HOT_HEADROOM_MIB", "LLAMA_MOE_HOT_DECAY", "LLAMA_MOE_HOT_SEED",
                                    "LLAMA_PLE_DIO_FILE",
                                    "LLAMA_FN_CBUF", "LLAMA_MOE_HOT_DECAY_RATIO", "LLAMA_MOE_HOT_DECAY_HYST", "LLAMA_MOE_HOT_DECAY_MIB", "LLAMA_MOE_HOT_BURST_MIB", "LLAMA_MOE_HOT_BURST_RATIO", "LLAMA_FN_GPU", "LLAMA_FN_GPU_IDXQ8", "LLAMA_FN_GPU_DEFER", "LLAMA_FN_GPU_CONVWB", "LLAMA_FN_GPU_HCFUSE", "LLAMA_FN_GPU_Q8F", "LLAMA_FN_HOST_EXEC", "LLAMA_FN_HOST_STEP", "LLAMA_FN_HOST_QOS", "LLAMA_FN_HOST_POKE", "LLAMA_FN_HOST_REARM", "LLAMA_FN_HOST_DMAOFF", "LLAMA_FN_L4_POST", "LLAMA_FN_L4_GDNSTALL", "LLAMA_FN_L4_HOIST", "LLAMA_FN_L4_QSA_SEL", "LLAMA_FN_L4_QSA_KVW", "LLAMA_FN_L4_QSA_POOL", "LLAMA_FN_L4_QSA_LIST", "LLAMA_FN_L4_QSA_QKV", "LLAMA_FN_L4_POOLBAR", "LLAMA_FN_L4_PFDEV", "LLAMA_FN_L4_HOST", "LLAMA_FN_L4_HC", "LLAMA_FN_L4_HCQ8", "LLAMA_FN_L4_HOST_MTPFUSE", "LLAMA_FN_L6_L2PF", "LLAMA_FN_L6_HCCOMB", "LLAMA_FN_VRAM_KEEP_MIB", "LLAMA_FN_L9_HINTTOPK", "LLAMA_FN_L10_SLOTLEND", "LLAMA_FN_L11_EMBDLOCK", "LLAMA_FN_L14_ARENA", "LLAMA_FN_L14_TIER", "LLAMA_FN_L4_HOST_BATCHCPY", "LLAMA_FN_L16_SEQRM", "LLAMA_FN_L16_MTP_ASYNC", "LLAMA_FN_L16_PLE_GATE", "LLAMA_FN_L17_START_PRE", "LLAMA_FN_L17_START_NT", "LLAMA_FN_L17_START_BURST", "LLAMA_FN_L17_VISION", "LLAMA_FN_L18_KEEPCTX", "LLAMA_FN_L18_FIRSTANS_PIN", "LLAMA_FN_L18_FIRSTANS_FIT", "LLAMA_FN_L18_FIRSTANS_WARMLATE", "LLAMA_FN_L18_FIRSTANS_STATE", "LLAMA_MOE_HOT_STATE", "LLAMA_FN_L17_START_STATE", "LLAMA_FN_L18_FIRSTANS_ONCE", "LLAMA_FN_L18_FIRSTANS_WARM", "LLAMA_MOE_HOT_UP_PIPE", "LLAMA_FN_L18_CHURN_LAYERS", "LLAMA_FN_L18_GF_GDNOUT", "LLAMA_FN_L18_GF_QKVZ", "LLAMA_FN_L19_TTFT_HUSH", "LLAMA_FN_L19_TTFT_HUSH_MS", "LLAMA_FN_L19_TTFT_EARLY", "LLAMA_FN_L19_TTFT_PHOT", "LLAMA_FN_L4_PROMPT", "LLAMA_FN_L19_TTFT_TAILFULL", "LLAMA_FN_L19_CONVSTORE", "LLAMA_FN_L19_CONVSTORE_N", "LLAMA_FN_L19_CONVSTORE_EVICT", "LLAMA_FN_L19_CONVSTORE_CKPT", "LLAMA_FN_L19_CONVSTORE_SYS", "LLAMA_FN_L21_MTP_VISION", "LLAMA_FN_L21_MTP_ROWS", "LLAMA_FN_L21_MTP_STATE", "LLAMA_FN_L21_MEDIA_CKPT", "LLAMA_FN_L21_VRAM_DXGI", "LLAMA_FN_L20_BIGPROMPT_UB", "LLAMA_FN_L21_BIGUB" }; // [TAG_FN_L21_DEC] // [TAG_FN_L19_ADOPT] // [TAG_FN_L18_ADOPT] [TAG_FN_L17_ADOPT] [TAG_FN_L4_ADOPT] [TAG_FN_L5_ADOPT] [TAG_FN_L6_ADOPT] [TAG_FN_L7_ADOPT] [TAG_FN_L9_ADOPT] [TAG_FN_L10_ADOPT] [TAG_FN_L11_ADOPT]
    const char * lever_names[] = { "LLAMA_MOE_BRIDGE", "LLAMA_MOE_BRIDGE_DMA", "LLAMA_MOE_BRIDGE_PF", "LLAMA_MOE_BRIDGE_PF_SOLO",
                                   "LLAMA_MOE_DMA_SHARE", "GGML_CPU_APPLY_ONCE",
                                   "GGML_CPU_Q5_1_AVX512", "GGML_CPU_MMID_MR", "GGML_CPU_MOE_FUSE", "GGML_CPU_VNNI",
                                   "GGML_SCHED_SPLIT_ASYNC", "GGML_CUDA_GRAPH_POKE", "LLAMA_NO_ECOQOS",
                                   "LLAMA_GRAPH_PER_WIDTH", "LLAMA_PLE_HOST_GATHER", "LLAMA_PLE_DIRECT_IO",
                                   "LLAMA_QSA_POS_MASK", "LLAMA_QSA_POS_CHUNK", "TURBO_QSA_CHUNK", "TURBO_QSA_SPARSE",
                                   "TURBO_QSA_TOPK_UNORDERED", "SPEC_MTP_COST", "LLAMA_MTP_ATTN_WINDOW",
                                   "LLAMA_MTP_HEAD_ROWS", "SPEC_DFT_UBATCH", "LLAMA_PREFILL_STREAM",
                                   "LLAMA_PREFILL_STREAM_LEND", "LLAMA_PREFILL_STREAM_THREADS", "LLAMA_PREFILL_STREAM_CACHED", "LLAMA_PREFILL_STREAM_PIN", "LLAMA_PREFILL_STREAM_MIN", "GGML_OP_OFFLOAD_MIN_BATCH", "SPEC_MTP_EQUAL_SEQS", "LLAMA_PREFILL_STREAM_D2D", "LLAMA_SRV_GEN_SHARE",
                                   "SPEC_MTP_BLOCK_VERIFY", "LLAMA_MOE_POOL_SPLIT", "LLAMA_MOE_POOL_PF_STREAMS", "LLAMA_MOE_POOL_SWPF", "LLAMA_MOE_BRIDGE_PF_RANK", "LLAMA_MOE_POOL_EXEC_CPU", "LLAMA_KPOOL_PAD", "LLAMA_KV_PAD_INDEXER", "GGML_CUDA_FN_L18_HCUP" }; // [TAG_FN_L18_ADOPT] // [TAG_FN_L8_GRAPHPAD]
    const size_t n_lookup = sizeof(lookup_names)/sizeof(lookup_names[0]);
    const size_t n_lever  = sizeof(lever_names)/sizeof(lever_names[0]);
    // [TAG_FN_L22] plus the round 22 rows (test_l22_rows)
    TCHECK(safe.size() == n_lookup + n_lever + k_n_l22_rows_decl, "safe = the measured winners (%zu, want %zu)", safe.size(),
           n_lookup + n_lever + k_n_l22_rows_decl);
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
        { "LLAMA_PREFILL_STREAM_CACHED", "1" }, { "LLAMA_PREFILL_STREAM_PIN", "2" }, // [TAG_FN_R25_STAGE] [TAG_FN_R25_PIN]
        { "LLAMA_FN_L4_GDNSTALL", "1" }, { "LLAMA_FN_L4_PFDEV", "2" }, { "LLAMA_FN_L4_HOST", "1" }, // [TAG_FN_L4_ADOPT]
        { "LLAMA_FN_L4_HOST_BATCHCPY", "2" }, { "LLAMA_FN_L16_SEQRM", "1" }, { "LLAMA_FN_L16_MTP_ASYNC", "1" }, // [TAG_FN_L16]
        { "LLAMA_FN_L16_PLE_GATE", "1" }, // [TAG_FN_L16_PLEGATE]
        { "LLAMA_FN_L17_START_PRE", "2048" }, { "LLAMA_FN_L17_START_NT", "1" }, { "LLAMA_FN_L17_START_BURST", "64" }, { "LLAMA_FN_L17_VISION", "1" }, // [TAG_FN_L17_ADOPT]
        { "LLAMA_FN_L18_KEEPCTX", "1" }, { "LLAMA_FN_L18_FIRSTANS_PIN", "1" }, { "LLAMA_FN_L18_FIRSTANS_FIT", "1" }, { "LLAMA_FN_L18_FIRSTANS_WARMLATE", "1" }, { "LLAMA_FN_L18_FIRSTANS_STATE", "1" }, { "LLAMA_MOE_HOT_STATE", "cache" }, { "LLAMA_FN_L17_START_STATE", "60" }, { "LLAMA_FN_L18_FIRSTANS_ONCE", "1" }, { "LLAMA_FN_L18_FIRSTANS_WARM", "1" }, { "LLAMA_MOE_HOT_UP_PIPE", "1" }, { "LLAMA_FN_L18_CHURN_LAYERS", "auto" }, { "GGML_CUDA_FN_L18_HCUP", "1" }, { "LLAMA_FN_L18_GF_GDNOUT", "1" }, { "LLAMA_FN_L18_GF_QKVZ", "1" }, // [TAG_FN_L18_ADOPT]
        { "LLAMA_FN_L19_TTFT_HUSH", "3" }, { "LLAMA_FN_L19_TTFT_HUSH_MS", "10000" }, { "LLAMA_FN_L19_TTFT_EARLY", "256" }, { "LLAMA_FN_L19_TTFT_PHOT", "1" }, { "LLAMA_FN_L4_PROMPT", "160" }, { "LLAMA_FN_L19_TTFT_TAILFULL", "512" }, { "LLAMA_FN_L19_CONVSTORE", "1" }, { "LLAMA_FN_L19_CONVSTORE_N", "8" }, { "LLAMA_FN_L19_CONVSTORE_EVICT", "0" }, { "LLAMA_FN_L19_CONVSTORE_CKPT", "0" }, { "LLAMA_FN_L19_CONVSTORE_SYS", "0" }, { "LLAMA_PREFILL_STREAM_MIN", "768" }, { "GGML_OP_OFFLOAD_MIN_BATCH", "768" }, // [TAG_FN_L19_ADOPT]
        { "LLAMA_FN_L20_BIGPROMPT_UB", "16384" }, { "LLAMA_FN_L21_BIGUB", "1" }, // [TAG_FN_L20_ADOPT] [TAG_FN_L21_BIGUB]
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
    TCHECK(find_opt(tdma, "GGML_CPU_MOE_FUSE") != nullptr && tdma.size() == safe.size() - 15 - 3, "trial-dma keeps the rest"); // [TAG_FN_L22] - the 3 round 22 pool rows // [TAG_FN_L4_ADOPT] + POOLBAR, PFDEV, [TAG_FN_L9_ADOPT] HINTTOPK (bridge only)

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

// [TAG_FN_L22] round 22 rows (plan section 6): every row present in safe with its build default, read through the
// lookup (LLAMA_FN_*) or put into the environment (the getenv names), no name twice; the bridge rows (P_BR) are not in
// trial-dma. The answer-changing levers and the pin family stay 0
struct l22_row { const char * name; const char * value; bool br; };
static const l22_row k_l22_rows[] = {
    { "LLAMA_FN_L22_PASSLOG", "1", false },
    { "LLAMA_FN_L22_IMGPASS", "1", false },
    { "LLAMA_FN_L22_IMGPASS_MASK", "auto", false },
    { "LLAMA_FN_L22_IMGPASS_MASK_MIB", "1024", false },
    { "LLAMA_FN_L22_SAMPSTATIC", "1", false },
    { "LLAMA_FN_L22_KVGROW", "1", false },
    { "LLAMA_FN_L22_VLEND_HOLD", "1", false },
    { "LLAMA_FN_L22_VLEND_SIZE", "1", false },
    { "LLAMA_FN_L22_IMGCACHE", "24", false },
    { "LLAMA_FN_L22_IMGDEC_THREADS", "4", false },
    { "LLAMA_FN_L22_IMG_MAXROWS", "0", false },
    { "LLAMA_FN_L22_TOKCACHE", "1", false },
    { "LLAMA_FN_L22_TOKCACHE_VERIFY", "1", false },
    { "LLAMA_FN_L22_JSONFLOAT", "1", false },
    { "LLAMA_FN_L22_PLEQ8", "1", false }, // [TAG_FN_L22_PLEQ8]
    { "LLAMA_FN_L22_VLRESIZE", "0", false },
    { "LLAMA_FN_L22_VLBATCH", "0", false },
    { "LLAMA_FN_L22_MTP_KEEP", "1", false },
    { "LLAMA_FN_L22_MEDIA_ANCHOR", "1", false },
    { "LLAMA_FN_L22_CKPT_KEEP", "1", false },
    { "LLAMA_FN_L22_CKPT_SPLIT", "1", false },
    { "LLAMA_FN_L22_LENDCLASS", "1", false },
    { "LLAMA_FN_L22_REFILL", "1", false },
    { "LLAMA_FN_L22_READAHEAD", "1", false },
    { "LLAMA_FN_L22_KEEPLEND", "1", false },
    { "LLAMA_FN_L22_ENDOVERLAP", "1", false },
    { "LLAMA_FN_L22_GRAPHKEY", "1", false },
    { "LLAMA_FN_L22_SCHEDBUF", "1", false },
    { "GGML_SCHED_FN_L22_SPLITS", "4096", false },
    { "LLAMA_FN_L22_RESSET", "1", false },
    { "LLAMA_FN_L22_REWARM", "1", false },
    { "LLAMA_FN_L22_SSDTIER", "2", false },
    { "LLAMA_FN_L22_SSDTIER_IO", "8", false },
    { "LLAMA_FN_L22_PIN", "0", false },
    { "LLAMA_FN_L22_PIN_UNIT", "ranges", false },
    { "LLAMA_FN_L22_PIN_MARGIN_MIB", "16384", false },
    { "LLAMA_FN_L22_PIN_LOCAL_MARGIN_MIB", "1024", false },
    { "LLAMA_FN_L22_PIN_FLOOR_MIB", "6144", false },
    { "LLAMA_FN_L22_PIN_ARENA_MIB", "16384", false },
    { "GGML_CUDA_FN_L22_GRAPHKEEP", "1", false },
    { "GGML_CUDA_FN_L22_GRAPH_CAP", "512,64", false },
    { "GGML_CUDA_FN_L22_GRAPH_MAXROWS", "128", false },
    { "GGML_CUDA_FN_L22_GRAPH2", "1", false },
    { "GGML_CUDA_FN_L22_GRAPHDIAG", "0", false },
    { "GGML_CUDA_FN_L22_SKCAP", "1", false },
    { "GGML_CUDA_FN_L22_MMQ_Y8", "1", false },
    { "GGML_CUDA_FN_L22_GLUQ8", "1", false },
    { "GGML_CUDA_MMQ_GLU_FUSION", "1", false },
    { "GGML_CUDA_FN_L22_MMVQ_R1", "0", false },
    { "GGML_CUDA_GDN_MMA_SNAP", "1", false },
    { "GGML_CUDA_GDN_MMA_CHUNK", "0", false },
    { "LLAMA_MOE_POOL_STAGGER", "0", true },
    { "LLAMA_MOE_POOL_PRETOUCH", "0", true },
    { "LLAMA_MOE_POOL_JOBPF", "1", true },
    { "LLAMA_FN_L22_X_STATS", "0", false },
    { "LLAMA_FN_L22_X_TAILSKIP", "0", false },
    { "LLAMA_FN_L22_X_TAILSKIP_W", "0", false },
    { "LLAMA_FN_L22_X_TAILSKIP_MULTI", "0", false },
    { "LLAMA_FN_L22_X_TAILSKIP_PF", "0", false },
    { "LLAMA_FN_L22_ONEPROMPT", "1", false },
    { "LLAMA_FN_L22_GENSHARE2", "1", false },
    { "LLAMA_FN_L22_PAUSE", "1", false },
    { "LLAMA_FN_L22_PAUSE_TOKENS", "2048", false },
    { "LLAMA_FN_L22_PAUSE_S", "8", false },
    { "LLAMA_FN_L22_SHAREKV", "1", false },
    { "LLAMA_FN_L22_DRAFT2", "0", false },
    { "LLAMA_FN_L24_ATTACH", "1", false }, // [TAG_FN_L24_ATTACH]
    { "LLAMA_FN_L24_ATTACH_CKPT", "0", false }, // [TAG_FN_L24_ATTACH]
    { "LLAMA_FN_L22_STAGERS", "auto", false },
    { "LLAMA_FN_L22_XOVER", "1", false },
    { "LLAMA_FN_L22_UBPLAN", "1", false },
    { "LLAMA_FN_L22_BIGUB2", "1", false },
    { "LLAMA_FN_L23_BG2POOLS", "1", false }, // [TAG_FN_L23_TWOSLOT]
    { "LLAMA_FN_L22_SMALLKEEP", "1", false },
    { "LLAMA_FN_L22_X_DEFER", "0", false },
    { "LLAMA_FN_L22_X_DEFER_W", "0.1", false },
    { "LLAMA_FN_L22_X_DEFER_R", "7", false },
    { "LLAMA_KV_ROWS", "0", false },
    { "LLAMA_FN_L22_QSA_SEQPOOLS", "1", false },
    { "LLAMA_FN_L22_QSA_SHAREPFX", "1", false },
    { "LLAMA_FN_L22_TBO", "0", false },
    { "LLAMA_FN_L22_DMASHARE", "0", false },
    { "LLAMA_FN_L22_DMASHARE_SLOTS", "4", false },
    { "LLAMA_FN_L22_X_PRIOR", "0", false },
    { "LLAMA_FN_L22_X_PRIOR_STATS", "raw", false },
    { "LLAMA_FN_L22_X_RESIDENT", "0", false },
    { "LLAMA_FN_L22_CPUSHARE", "0", false },
    { "LLAMA_FN_L22_CPUSHARE_MAX", "3072", false },
    { "LLAMA_FN_L22_CPUSHARE_ROWS", "8", false },
    { "LLAMA_FN_L24_KEEPVAL", "1", false }, // [TAG_FN_L24]
    { "LLAMA_FN_L24_GENCKPT", "1", false },
    { "LLAMA_FN_L24_OPFAIR", "1", false },
    { "LLAMA_FN_L24_GSCAP", "1", false },
    { "LLAMA_FN_R25_REFILL", "1", false }, // [TAG_FN_R25_REFILL]
    { "LLAMA_FN_R25_REFILL_THREADS", "8", false },
    { "LLAMA_FN_R25_REFILL_HALF_KIB", "2048", false },
    { "LLAMA_FN_R25_SBSYNC_MIB", "1024", false },
};
static const size_t k_n_l22_rows = sizeof(k_l22_rows)/sizeof(k_l22_rows[0]);
const size_t k_n_l22_rows_decl = k_n_l22_rows;

static void test_l22_rows() {
    const auto safe = llama_fn_profile_opts(LLAMA_FN_PROFILE_SAFE, nullptr);
    const auto tdma = llama_fn_profile_opts(LLAMA_FN_PROFILE_TRIAL_DMA, nullptr);
    for (size_t i = 0; i < k_n_l22_rows; ++i) {
        const l22_row & r = k_l22_rows[i];
        const llama_fn_opt * o = find_opt(safe, r.name);
        const bool lookup = strncmp(r.name, "LLAMA_FN_", 9) == 0;
        TCHECK(o && o->value == r.value, "l22 row %s = %s", r.name, r.value);
        TCHECK(o && o->inject == !lookup, "l22 row %s: %s", r.name, lookup ? "lookup" : "environment");
        TCHECK((find_opt(tdma, r.name) == nullptr) == r.br, "l22 row %s in trial-dma iff not a bridge row", r.name);
        int n = 0;
        for (const auto & s2 : safe) {
            n += s2.name == r.name ? 1 : 0;
        }
        TCHECK(n == 1, "l22 row %s once (%d)", r.name, n);
    }
    // every LLAMA_FN_L22_ / *_FN_L22_ name of the profile is in the list above
    for (const auto & o : safe) {
        if (o.name.find("_L22_") != std::string::npos) {
            bool found = false;
            for (size_t i = 0; i < k_n_l22_rows && !found; ++i) {
                found = o.name == k_l22_rows[i].name;
            }
            TCHECK(found, "profile name %s listed", o.name.c_str());
        }
    }
    const char * off_names[] = { "LLAMA_FN_L22_X_TAILSKIP", "LLAMA_FN_L22_X_DEFER", "LLAMA_FN_L22_X_PRIOR", "LLAMA_FN_L22_X_RESIDENT",
                                 "LLAMA_FN_L22_PIN", "LLAMA_FN_L22_DMASHARE", "LLAMA_FN_L22_CPUSHARE", "LLAMA_FN_L22_IMG_MAXROWS" };
    for (const char * n : off_names) {
        const llama_fn_opt * o = find_opt(safe, n);
        TCHECK(o && o->value == "0", "%s stays 0 (owner)", n);
    }
    TCHECK(find_opt(safe, "LLAMA_FN_L22_LOGITDUMP") == nullptr, "no LOGITDUMP row");
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

// [TAG_FN_L18_CHURN_LAYERS] slots per layer by weight, at Flash-Next UD-Q4_K_XL sizes and the owner's budget
static void test_weighted_slots() {
    const std::vector<size_t> bytes = flash_next_expert_bytes();
    size_t row = 0;
    size_t big = 0;
    for (size_t x : bytes) {
        row += x;
        big  = std::max(big, x);
    }
    const size_t b = 120*row + row/3; // 119 even slots per layer and a part row
    const int32_t n_even = llama_fn_even_slots(bytes, b, 512);
    TCHECK(n_even == 119, "even count at the test budget %d", n_even);
    auto used_of = [&](const std::vector<int32_t> & n) {
        size_t u = 0;
        for (size_t i = 0; i < n.size(); ++i) {
            u += ((size_t) n[i] + 1)*bytes[i];
        }
        return u;
    };
    // equal weights: the even count in every layer, then the rest of the budget slot by slot
    const std::vector<int32_t> eq = llama_fn_weighted_slots(bytes, b, 512, std::vector<float>(48, 1.0f));
    bool ge = eq.size() == 48;
    for (int32_t x : eq) {
        ge = ge && (x == n_even || x == n_even + 1);
    }
    TCHECK(ge, "equal weights: the even count, some layers one more");
    TCHECK(used_of(eq) <= b && b - used_of(eq) < big, "equal weights fill the budget (%zu of %zu)", used_of(eq), b);
    // the l18 table shape: layer 0 heavy, layer 15 light
    std::vector<float> w(48, 1.0f);
    w[0] = 1.387f; w[15] = 0.723f; w[38] = 1.303f; w[39] = 0.714f;
    const std::vector<int32_t> ws = llama_fn_weighted_slots(bytes, b, 512, w);
    TCHECK(used_of(ws) <= b && b - used_of(ws) < big, "weighted slots fill the budget (%zu of %zu)", used_of(ws), b);
    TCHECK(ws[0] > ws[1] && ws[1] > ws[15], "more slots for a heavier layer: %d %d %d", ws[0], ws[1], ws[15]);
    TCHECK(std::abs(ws[0] - (int32_t) std::lround(1.387*ws[1])) <= 2, "in proportion: %d vs 1.387 x %d", ws[0], ws[1]);
    TCHECK(ws[38] > ws[39], "a q5_1 layer by weight: %d %d", ws[38], ws[39]);
    // the cap: a weight that would ask for more than every expert stops at n_expert, the rest goes to the others
    std::vector<float> wc(48, 1.0f);
    wc[3] = 40.0f;
    const std::vector<int32_t> wcs = llama_fn_weighted_slots(bytes, b, 512, wc);
    TCHECK(wcs[3] == 512, "capped at every expert: %d", wcs[3]);
    TCHECK(used_of(wcs) <= b && b - used_of(wcs) < big, "the cap's rest stays in the budget");
    // bad weights: the even count
    std::vector<float> bad = w;
    bad[7] = 0.0f;
    const std::vector<int32_t> be = llama_fn_weighted_slots(bytes, b, 512, bad);
    bool all_even = be.size() == 48;
    for (int32_t x : be) {
        all_even = all_even && x == n_even;
    }
    TCHECK(all_even, "a zero weight: even slots");
    const std::vector<int32_t> bn = llama_fn_weighted_slots(bytes, b, 512, std::vector<float>(47, 1.0f));
    TCHECK(bn.size() == 48 && bn[0] == n_even && bn[47] == n_even, "a weight count unlike the layers: even slots");
    TCHECK(llama_fn_weighted_slots(bytes, row/2, 512, w) == std::vector<int32_t>(48, 0), "no room for the zero slots: 0");
    TCHECK(llama_fn_weighted_slots({}, b, 512, {}).empty(), "no layers");
    // the parser
    TCHECK((llama_fn_parse_weights("1.5,0.75, 2") == std::vector<float>{ 1.5f, 0.75f, 2.0f }), "three weights");
    TCHECK(llama_fn_parse_weights("1,0,2").empty(), "a zero weight is refused");
    TCHECK(llama_fn_parse_weights("1,x").empty(), "not a number");
    TCHECK(llama_fn_parse_weights("1,,2").empty(), "an empty item");
    TCHECK(llama_fn_parse_weights("").empty() && llama_fn_parse_weights(nullptr).empty(), "nothing");
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

    // [TAG_FN_L19_TTFT_EARLY] the short path's limit
    TCHECK(llama_fn_cbuf_short_max(0, 0, true, 42, 31) == 0, "EARLY off: no short path (as before)");
    TCHECK(llama_fn_cbuf_short_max(64, 0, true, 42, 31) == 64, "EARLY off: LLAMA_FN_L4_PROMPT stays");
    TCHECK(llama_fn_cbuf_short_max(0, 256, true, 42, 31) == 256, "warm pass runs: a 42-token first message takes the short path");
    TCHECK(llama_fn_cbuf_short_max(0, 256, false, 42, 31) == 0, "warm pass done: the normal path");
    TCHECK(llama_fn_cbuf_short_max(0, 256, true, 300, 31) == 0, "wider than EARLY: the normal path");
    TCHECK(llama_fn_cbuf_short_max(512, 256, true, 300, 31) == 512, "a larger LLAMA_FN_L4_PROMPT wins");
    TCHECK(llama_fn_cbuf_short_max(0, 256, true, 20, 31) == 0, "a batch the SMALL reserve serves anyway: unchanged");

    // [TAG_FN_L20_BIGPROMPT_TAIL] a release from the top at the tail's 1 GiB segment boundaries
    {
        const size_t G = (size_t) 1 << 30, M = (size_t) 1 << 20;
        const size_t lo = 20000*M, hi = lo + 8600*M;
        TCHECK(llama_fn_cbuf_part_lo(lo, hi, G, 0) == lo, "need 0: the whole tail");
        TCHECK(llama_fn_cbuf_part_lo(lo, hi, G, 4300*M) == lo + 4*G, "4.2 GiB of 8.4: the top 4.4 GiB from segment 4");
        TCHECK(hi - llama_fn_cbuf_part_lo(lo, hi, G, 4300*M) >= 4300*M, "at least what was asked");
        TCHECK(llama_fn_cbuf_part_lo(lo, hi, G, 8600*M) == lo, "all of it: the whole tail");
        TCHECK(llama_fn_cbuf_part_lo(lo, hi, G, 9000*M) == lo, "more than the tail: the whole tail");
        TCHECK(llama_fn_cbuf_part_lo(lo, hi, G, 1) == lo + 8*G, "1 byte: the last (partial) segment only");
        TCHECK(llama_fn_cbuf_part_lo(lo, hi, 0, 100*M) == lo, "no segment size: the whole tail");
        TCHECK(llama_fn_cbuf_part_lo(lo, lo, G, 100*M) == lo, "empty tail");
        TCHECK((llama_fn_cbuf_part_lo(lo, hi, G, 3*G) - lo) % G == 0, "always a segment boundary");
    }
    // [TAG_FN_L20_BIGPROMPT_UB] the FULL width and the narrow parts of a prompt that goes on
    TCHECK(llama_fn_l20_full_width(0, 8192, 9000, -1) == 8192, "off: n_ubatch");
    TCHECK(llama_fn_l20_full_width(8192, 16384, 16384, 16388) == 16384, "a full big batch: big");
    TCHECK(llama_fn_l20_full_width(8192, 16384, 7647, 16388) == 16384, "the first part of a 24K prompt: big (the rest follows)");
    TCHECK(llama_fn_l20_full_width(8192, 16384, 2000, 0) == 8192, "a 2K turn: the command's width");
    TCHECK(llama_fn_l20_full_width(8192, 16384, 8192, 0) == 8192, "exactly 8192: the command's width");
    TCHECK(llama_fn_l20_full_width(8192, 16384, 6000, 2500) == 16384, "6000 + 2500 after it: big");
    TCHECK(llama_fn_l20_full_width(8192, 16384, 6000, 2192) == 8192, "6000 + 2192 = 8192: the command's width");
    TCHECK(llama_fn_l20_full_width(8192, 16384, 9000, -1) == 16384, "no hint, a 9000-token batch: big");
    // [TAG_FN_L21_BIGUB] the big width only above 16K prompts
    TCHECK(llama_fn_l21_full_width(0, 8192, 9000, -1) == 8192, "l21 off: n_ubatch");
    TCHECK(llama_fn_l21_full_width(8192, 16384, 16384, 16388) == 16384, "l21: the first 16K of a 32K prompt: big");
    TCHECK(llama_fn_l21_full_width(8192, 16384, 16384, 1) == 16384, "l21: 16385 tokens: big");
    TCHECK(llama_fn_l21_full_width(8192, 16384, 16384, 0) == 8192, "l21: exactly 16384: the command's width");
    TCHECK(llama_fn_l21_full_width(8192, 16384, 12000, 0) == 8192, "l21: a 12K paste: the command's width");
    TCHECK(llama_fn_l21_full_width(8192, 16384, 7647, 8000) == 8192, "l21: a 15.6K prompt's first part: the command's width");
    TCHECK(llama_fn_l21_full_width(8192, 16384, 9000, -1) == 8192, "l21: no hint, a 9000-token batch: the command's width");
    TCHECK(llama_fn_l21_full_width(8192, 16384, 16384, -1) == 16384, "l21: no hint, a full 16K batch: big");
    TCHECK(llama_fn_l20_full_width(8192, 16384, 5000, -1) == 8192, "no hint, 5000: the command's width");
    TCHECK(llama_fn_l20_full_width(16384, 16384, 20000, -1) == 16384, "base = n_ubatch: n_ubatch");
    TCHECK(llama_fn_l20_keep_full(8192, false, false, true, 16388), "a 23-token part between two big ones keeps FULL");
    TCHECK(!llama_fn_l20_keep_full(8192, false, false, true, 0), "the prompt's last part: as before");
    TCHECK(!llama_fn_l20_keep_full(8192, false, false, true, -1), "no hint: as before");
    TCHECK(!llama_fn_l20_keep_full(0, false, false, true, 16388), "off: as before");
    TCHECK(!llama_fn_l20_keep_full(8192, false, false, false, 16388), "SMALL state: nothing to keep");
    TCHECK(!llama_fn_l20_keep_full(8192, false, true, true, 16388), "the short path stays SMALL");
    TCHECK(!llama_fn_l20_keep_full(8192, true, false, true, 16388), "a wide batch is FULL anyway");

    // [TAG_FN_L19_TTFT_TAILFULL] the last 4 tokens of a split 400-token turn keep FULL; nothing else does
    TCHECK(llama_fn_cbuf_tail_full(512, true, false, 4, 1, 400, false, 31), "split short turn: the 4-token tail stays FULL");
    TCHECK(!llama_fn_cbuf_tail_full(0, true, false, 4, 1, 400, false, 31), "off");
    TCHECK(!llama_fn_cbuf_tail_full(512, true, false, 4, 1, 900, false, 31), "a split of >= 512 tokens switches as upstream");
    TCHECK(!llama_fn_cbuf_tail_full(512, true, false, 3, 1, 4, true, 31), "an answer step after the tail returns to SMALL");
    TCHECK(!llama_fn_cbuf_tail_full(512, true, false, 3, 3, 400, true, 31), "an answer step after an unsplit prompt returns to SMALL");
    TCHECK(!llama_fn_cbuf_tail_full(512, false, false, 4, 1, 400, false, 31), "SMALL state: nothing to keep");
    TCHECK(!llama_fn_cbuf_tail_full(512, true, true, 4, 1, 400, false, 31), "the short path is SMALL by definition");
    TCHECK(!llama_fn_cbuf_tail_full(512, true, false, 4, 0, 400, false, 31), "a narrow part without outputs (mid-prompt) is not a tail");
    TCHECK(!llama_fn_cbuf_tail_full(512, true, false, 40, 1, 400, false, 31), "a wide batch is FULL anyway");
    TCHECK(!llama_fn_cbuf_tail_full(512, true, false, 4, 1, 20, false, 31), "after a narrow part: switch");

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

// [TAG_FN_V0254_NARROWFULL] the narrow-part rule judges the rest batch alone (the owner's rows: small_t 30 on 2 slots,
// LLAMA_FN_L4_PROMPT 160, LLAMA_FN_L19_TTFT_EARLY 256, -b 4096)
static void test_v0254_narrowfull() {
    const uint32_t st = 30, sm = 160, early = 256, nb = 4096;
    // the verifier's case: 20 rows + 150 rows. The 150-row rest takes the short path (150 <= 160) and stays SMALL, so the
    // 20-row part stays SMALL too (v0.25.3); the old guard (170 > max(30, 160)) started FULL for no gain
    TCHECK(!llama_fn_v0254_narrow_full(20, 0, 150, nb, st, sm, early, false), "20 + 150 rows: both SMALL (the rest is short)");
    TCHECK(!llama_fn_v0254_narrow_full(20, 1, 150, nb, st, sm, early, false), "20 + 150 rows with an output row: SMALL");
    TCHECK(!llama_fn_v0254_narrow_full(20, 0, 160, nb, st, sm, early, false), "20 + 160 rows: the rest is short (<= 160)");
    TCHECK(llama_fn_v0254_narrow_full(20, 0, 161, nb, st, sm, early, false), "20 + 161 rows: the rest goes FULL -> FULL now");
    // the case the rule is for: the server's split at a context checkpoint, a 14-token system part before 9K rows
    TCHECK(llama_fn_v0254_narrow_full(14, 0, 9200, nb, st, sm, early, false), "14 + 9200 rows: FULL now (as the rest)");
    TCHECK(llama_fn_v0254_narrow_full(14, 0, 1500, nb, st, sm, early, false), "14 + 1500 rows: FULL now (as the rest)");
    // a narrow rest, a rest on the warm pass's early short path, no short path at all
    TCHECK(!llama_fn_v0254_narrow_full(20, 0, 25, nb, st, sm, early, false), "20 + 25 rows: a narrow rest, SMALL");
    TCHECK(!llama_fn_v0254_narrow_full(20, 0, 200, nb, st, sm, early, true), "20 + 200 rows during the warm pass: short (<= 256)");
    TCHECK(llama_fn_v0254_narrow_full(20, 0, 200, nb, st, sm, early, false), "20 + 200 rows after the warm pass: FULL now");
    TCHECK(llama_fn_v0254_narrow_full(20, 0, 300, nb, st, sm, early, true), "20 + 300 rows during the warm pass: FULL now");
    TCHECK(llama_fn_v0254_narrow_full(20, 0, 150, nb, st, 0, 0, false), "no short path (LLAMA_FN_L4_PROMPT 0): 150 rows go FULL");
    // the rest's next batch is at most n_batch rows: one of <= short_max rows stays SMALL (so does the whole prompt)
    TCHECK(!llama_fn_v0254_narrow_full(20, 0, 9000, 128, st, sm, early, false), "-b 128: the rest's batches are short");
    TCHECK(llama_fn_v0254_narrow_full(20, 0, 9000, 0, st, sm, early, false), "n_batch 0: the whole rest (FULL)");
    // not this rule's batches
    TCHECK(!llama_fn_v0254_narrow_full(20, 20, 9000, nb, st, sm, early, false), "answer-only batch: never (R-L23-3)");
    TCHECK(!llama_fn_v0254_narrow_full(20, 0, 0, nb, st, sm, early, false), "the prompt's last part: no rest");
    TCHECK(!llama_fn_v0254_narrow_full(20, 0, -1, nb, st, sm, early, false), "rest not known: no");
    TCHECK(!llama_fn_v0254_narrow_full(31, 0, 9000, nb, st, sm, early, false), "a wide batch: its own rule");
    TCHECK(!llama_fn_v0254_narrow_full(0, 0, 9000, nb, st, sm, early, false), "an empty batch: no");
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

// [TAG_FN_L22_P10] round 22 package P10: stagers, the prompt cost model, the crossover, the ubatch planner, the 1-sequence
// 16K reserve rule, the SMALLKEEP headroom rule
static void test_l22_p10() {
    // stagers switch and formula
    TCHECK(llama_fn_l22_stagers_mode(nullptr) == 0 && llama_fn_l22_stagers_mode("") == 0 && llama_fn_l22_stagers_mode("0") == 0 &&
           llama_fn_l22_stagers_mode("off") == 0, "stagers unset / 0 / off -> 0");
    TCHECK(llama_fn_l22_stagers_mode("auto") == -1 && llama_fn_l22_stagers_mode("AUTO") == -1 && llama_fn_l22_stagers_mode("1") == -1,
           "stagers auto / 1 -> formula");
    TCHECK(llama_fn_l22_stagers_mode("6") == 6 && llama_fn_l22_stagers_mode("x") == 0 && llama_fn_l22_stagers_mode("200") == 64,
           "stagers fixed n, junk off, at most 64");
    TCHECK(llama_fn_l22_stagers(0, 0, 16, 4) == 16, "no host bytes -> n_max");
    TCHECK(llama_fn_l22_stagers(1000, 1000, 16, 4) == 16, "all staged -> 16");
    TCHECK(llama_fn_l22_stagers(0, 1000, 16, 4) == 4, "nothing staged -> 4");
    TCHECK(llama_fn_l22_stagers(500, 1000, 16, 4) == 8, "half staged -> 8");
    TCHECK(llama_fn_l22_stagers(1000, 1000, 8, 4) == 8 && llama_fn_l22_stagers(10, 1000, 8, 4) == 4 &&
           llama_fn_l22_stagers(500, 1000, 8, 4) == 4, "n_max 8: 4..8");
    TCHECK(llama_fn_l22_stagers(530, 1000, 16, 4) == 8 && llama_fn_l22_stagers(550, 1000, 16, 4) == 9, "rounding");
    TCHECK(llama_fn_l22_stagers(2000, 1000, 16, 4) == 16 && llama_fn_l22_stagers(500, 1000, 16, 32) == 16, "staged > host, n_min > n_max");

    // depth buckets
    TCHECK(llama_fn_l22_depth_bucket(0) == 0 && llama_fn_l22_depth_bucket(16383) == 0 && llama_fn_l22_depth_bucket(16384) == 1 &&
           llama_fn_l22_depth_bucket(65535) == 1 && llama_fn_l22_depth_bucket(65536) == 2 && llama_fn_l22_depth_bucket(163839) == 2 &&
           llama_fn_l22_depth_bucket(163840) == 3 && llama_fn_l22_depth_bucket(262144) == 3, "bucket bounds 16K / 64K / 160K");

    // cost model: priors, EWMA, clamp
    llama_fn_l22_cost c;
    llama_fn_l22_cost_init(c, false);
    TCHECK(c.pass_ms[0] == 3400.0 && c.gpu_ms_row[3] == 0.42 && c.cpu_ms_row[2] == 4.55 && c.fix_ms == 240.0 && c.big_ms == 300.0 &&
           !c.pinned, "unpinned priors");
    llama_fn_l22_cost cp;
    llama_fn_l22_cost_init(cp, true);
    TCHECK(cp.pass_ms[1] == 1700.0 && cp.pinned, "pinned priors");
    llama_fn_l22_cost_pass(c, 0, 1000, 4000.0, 3000.0); // copy-bound: first sample replaces the prior
    TCHECK(std::fabs(c.pass_ms[0] - 4000.0) < 1e-9 && c.n_pass[0] == 1 && c.n_obs[0] == 1 && c.n_gpu[0] == 0, "first sample, < 2048 rows no gpu");
    llama_fn_l22_cost_pass(c, 0, 1000, 2000.0, 1500.0);
    TCHECK(std::fabs(c.pass_ms[0] - 3500.0) < 1e-9, "EWMA alpha 0.25 (%.1f)", c.pass_ms[0]);
    llama_fn_l22_cost_pass(c, 0, 1000, 3500.0, 3000.0);
    TCHECK(c.n_pass[0] == 3, "three samples");
    llama_fn_l22_cost_pass(c, 0, 1000, 14000.0, 13000.0); // a stall: clamped to 2x
    TCHECK(std::fabs(c.pass_ms[0] - (3500.0 + 0.25*3500.0)) < 1e-9, "outlier clamped to 2x after 3 samples (%.1f)", c.pass_ms[0]);
    llama_fn_l22_cost c2;
    llama_fn_l22_cost_init(c2, false);
    llama_fn_l22_cost_pass(c2, 1, 8192, 3000.0, 10.0); // compute-bound: only lowers the copy estimate
    TCHECK(c2.pass_ms[1] == 3000.0 && c2.n_pass[1] == 0 && c2.n_gpu[1] == 1 && std::fabs(c2.gpu_ms_row[1] - 2990.0/8192) < 1e-9,
           "compute-bound pass: copy bound, gpu row sample");
    llama_fn_l22_cost_pass(c2, 1, 8192, 3200.0, 10.0);
    TCHECK(c2.pass_ms[1] == 3000.0, "a longer compute-bound pass leaves the copy estimate");
    llama_fn_l22_cost_cpu(c2, 1, 100, 1000.0);
    TCHECK(c2.n_cpu[1] == 0, "CPU samples below 128 rows are dropped");
    llama_fn_l22_cost_cpu(c2, 1, 500, 2000.0);
    TCHECK(c2.n_cpu[1] == 1 && c2.cpu_ms_row[1] == 4.0, "CPU sample");
    llama_fn_l22_cost_fix(c2, 200.0, -1.0);
    TCHECK(c2.fix_ms == 200.0 && c2.n_fix == 1 && c2.n_big == 0 && c2.big_ms == 300.0, "fix sample, no 16K period");
    llama_fn_l22_cost_pass(c2, 7, 1000, 1.0, 1.0);
    llama_fn_l22_cost_pass(c2, 1, 0, 1.0, 1.0);
    TCHECK(c2.n_obs[1] == 2, "bad bucket / zero rows ignored");
    llama_fn_l22_cost_repin(c2, true);
    TCHECK(c2.pass_ms[1] == 1700.0 && c2.n_obs[1] == 0 && c2.n_cpu[1] == 1 && c2.pinned, "repin: copy cost from the prior");
    TCHECK(llama_fn_l22_est_pass(c, 2, 1000) == 3400.0 && std::fabs(llama_fn_l22_est_pass(c, 2, 16384) - 16384*0.42) < 1e-6 &&
           std::fabs(llama_fn_l22_est_cpu(c, 2, 100) - 455.0) < 1e-9, "estimates");

    // crossover
    llama_fn_l22_cost x;
    llama_fn_l22_cost_init(x, false);
    TCHECK(llama_fn_l22_xover(x, 0, 768, 384, 2048) == 768, "priors, no samples: 768 stays");
    auto feed = [](llama_fn_l22_cost & k, int b, double pass, double cpu_row, int n) {
        for (int i = 0; i < n; ++i) {
            llama_fn_l22_cost_pass(k, b, 1000, pass, pass*0.8);
            llama_fn_l22_cost_cpu(k, b, 400, cpu_row*400);
        }
    };
    llama_fn_l22_cost x2 = x;
    feed(x2, 0, 3400.0, 4.55, 2);
    TCHECK(llama_fn_l22_xover(x2, 0, 768, 384, 2048) == 768, "two samples: no change");
    feed(x2, 0, 3400.0, 4.55, 1);
    const int32_t t_prior = llama_fn_l22_xover(x2, 0, 700, 384, 2048);
    TCHECK(t_prior >= 795 && t_prior <= 805, "measured = priors: ~800 (%d)", t_prior);
    TCHECK(llama_fn_l22_xover(x2, 0, 768, 384, 2048) == 768, "within 10 %% of 768: kept");
    llama_fn_l22_cost xp;
    llama_fn_l22_cost_init(xp, true);
    feed(xp, 0, 1400.0, 4.55, 3);
    TCHECK(llama_fn_l22_xover(xp, 0, 768, 384, 2048) == 384, "fast pinned passes: the 384 floor");
    llama_fn_l22_cost xu;
    llama_fn_l22_cost_init(xu, false);
    feed(xu, 0, 5500.0, 4.55, 3);
    const int32_t t_real = llama_fn_l22_xover(xu, 0, 768, 384, 2048);
    TCHECK(t_real >= 1100 && t_real <= 1300, "unpinned real-use passes 5.5 s: 1,100-1,300 (%d)", t_real);
    TCHECK(llama_fn_l22_xover(xu, 0, 768, 384, 768) == 768, "hi 768: never above");
    TCHECK(llama_fn_l22_xover(xu, 1, 768, 384, 2048) == 768, "buckets are independent");
    llama_fn_l22_cost xs;
    llama_fn_l22_cost_init(xs, false);
    feed(xs, 0, 20000.0, 4.55, 3);
    TCHECK(llama_fn_l22_xover(xs, 0, 768, 384, 2048) == 2048, "a stream slower than any CPU path in range: hi");

    // ubatch planner (priors unless noted)
    llama_fn_l22_cost u;
    llama_fn_l22_cost_init(u, false);
    double e8 = -1.0, e16 = -1.0;
    TCHECK(llama_fn_l22_ubplan(u, 0, 8192, 16384, 6000, 2000, 768, true, &e8, &e16) == 8192 && e8 == 0.0 && e16 == 0.0,
           "R <= 8192: base, not priced");
    TCHECK(llama_fn_l22_ubplan(u, 0, 8192, 16384, 10000, 0, 768, true, &e8, &e16) == 16384, "R 10,000 unpinned: 16K (%.0f vs %.0f)", e8, e16);
    TCHECK(std::fabs(e8 - (8192*0.42 + 3400.0 + 240.0)) < 1e-6 && std::fabs(e16 - (10000*0.42 + 240.0 + 300.0)) < 1e-6,
           "R 10,000 estimates");
    llama_fn_l22_cost up;
    llama_fn_l22_cost_init(up, true);
    TCHECK(llama_fn_l22_ubplan(up, 0, 8192, 16384, 16000, 0, 768, true, &e8, &e16) == 8192, "R 16,000 pinned: 8K (%.0f vs %.0f)", e8, e16);
    TCHECK(llama_fn_l22_ubplan(up, 0, 8192, 16384, 16384, 16384, 768, true, &e8, &e16) == 16384,
           "R 32K pinned with the prior 16K extra: within 3 %%, today's 16K kept (%.0f vs %.0f)", e8, e16);
    llama_fn_l22_cost_fix(up, -1.0, 1000.0);
    TCHECK(llama_fn_l22_ubplan(up, 0, 8192, 16384, 16384, 16384, 768, true, &e8, &e16) == 8192,
           "R 32K pinned, measured 16K extra 1 s: 8K (%.0f vs %.0f)", e8, e16);
    llama_fn_l22_cost uu;
    llama_fn_l22_cost_init(uu, false);
    feed(uu, 0, 5500.0, 4.55, 3);
    TCHECK(llama_fn_l22_ubplan(uu, 0, 8192, 16384, 16384, 16384, 768, true, &e8, &e16) == 16384, "R 32K unpinned 5.5 s: 16K");
    TCHECK(llama_fn_l22_ubplan(uu, 0, 8192, 16384, 10000, 0, 768, false, &e8, &e16) == 8192 &&
           llama_fn_l22_ubplan(uu, 0, 8192, 16384, 16384, 16384, 768, false, &e8, &e16) == 8192, "no borrow: base always");
    llama_fn_l22_ubplan(u, 0, 8192, 16384, 8492, 0, 768, true, &e8, &e16);
    TCHECK(std::fabs(e8 - (8192*0.42 + 300*4.55 + 240.0)) < 1e-6, "a rest below the crossover is priced as CPU path (%.1f)", e8);
    llama_fn_l22_ubplan(u, 0, 8192, 16384, 8492, 0, 200, true, &e8, &e16);
    TCHECK(std::fabs(e8 - (8192*0.42 + 3400.0 + 240.0)) < 1e-6, "a rest above it streams (%.1f)", e8);
    // identity with today's rule where the planner does not price
    bool same = true;
    const uint32_t toks[]  = { 1, 31, 500, 8191, 8192, 8193, 12000, 16383, 16384 };
    const int64_t  afters[] = { -1, 0, 100, 8191, 8192, 30000 };
    for (uint32_t n : toks) {
        for (int64_t a : afters) {
            const uint32_t l21 = llama_fn_l21_full_width(8192, 16384, n, a);
            const uint32_t pl  = llama_fn_l22_ubplan(u, 0, 8192, 16384, n, a, 768, true, nullptr, nullptr);
            const int64_t  r   = (int64_t) n + a;
            if ((a < 0 || r <= 8192) && pl != l21) {
                same = false;
            }
        }
    }
    TCHECK(same, "unknown hint or a prompt within the base width: the l21 width");
    TCHECK(llama_fn_l22_ubplan(u, 0, 0, 16384, 20000, 0, 768, true, nullptr, nullptr) == 16384 &&
           llama_fn_l22_ubplan(u, 0, 16384, 16384, 20000, 0, 768, true, nullptr, nullptr) == 16384, "no base width: n_ubatch");

    // BIGUB2: the 1-sequence 16K reserve, prompt sequences of a batch
    TCHECK(llama_fn_l22_bigub2_ok(2, 4000u*MiB, 3000u*MiB) && llama_fn_l22_bigub2_ok(2, 3000u*MiB, 3000u*MiB), "2 slots, 1-seq 16K >= 2-seq 8K");
    TCHECK(!llama_fn_l22_bigub2_ok(1, 4000u*MiB, 3000u*MiB) && !llama_fn_l22_bigub2_ok(2, 2999u*MiB, 3000u*MiB) &&
           !llama_fn_l22_bigub2_ok(4, 0, 0), "1 slot / smaller / unknown: off");
    const uint32_t rows_a[] = { 9000, 1, 3, 0 };
    const uint32_t rows_b[] = { 9000, 4000, 2, 0 };
    TCHECK(llama_fn_l22_prompt_seqs(rows_a, 4, 31) == 1 && llama_fn_l22_prompt_seqs(rows_b, 4, 31) == 2 &&
           llama_fn_l22_prompt_seqs(rows_b, 4, 4000) == 1 && llama_fn_l22_prompt_seqs(nullptr, 4, 31) == 0, "prompt sequences");

    // SMALLKEEP headroom
    TCHECK(llama_fn_l22_smallkeep_fits(3000*MiB, 2000*MiB, 256*MiB, 512*MiB) && llama_fn_l22_smallkeep_fits(2768*MiB, 2000*MiB, 256*MiB, 512*MiB),
           "fits: free >= full + pool + margin");
    TCHECK(!llama_fn_l22_smallkeep_fits(2767*MiB, 2000*MiB, 256*MiB, 512*MiB), "one MiB short");
    TCHECK(!llama_fn_l22_smallkeep_fits(SIZE_MAX, 0, 0, 0) && !llama_fn_l22_smallkeep_fits(SIZE_MAX, SIZE_MAX - 10, 100, 0) &&
           llama_fn_l22_smallkeep_fits(SIZE_MAX, SIZE_MAX - 100, 50, 50), "unknown size, overflow");

    // the PHOT rule's crossover
    TCHECK(llama_fn_l22_offload_min() == 0, "offload minimum unset");
    llama_fn_l22_offload_min_set(1100);
    TCHECK(llama_fn_l22_offload_min() == 1100, "offload minimum set");
    llama_fn_l22_offload_min_set(-5);
    TCHECK(llama_fn_l22_offload_min() == 0, "offload minimum cleared");
}

int main() {
    // no ggml_backend_load_all(): the backends are linked in (GGML_BACKEND_DL=OFF); run with CUDA_VISIBLE_DEVICES=-1
    test_profile_names();
    test_profile_opts();
    test_l22_rows(); // [TAG_FN_L22]
    test_host_expert_layers();
    test_lookup();
    test_inject();
    test_even_slots();
    test_weighted_slots();
    test_vram_fit();
    test_cbuf();
    test_v0254_narrowfull(); // [TAG_FN_V0254_NARROWFULL]
    test_ram_fit();
    test_l22_p10(); // [TAG_FN_L22_P10]

    printf("test-fn-auto: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
