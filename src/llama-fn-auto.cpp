// [TAG_FN_AUTO] automatic qwen4exp (Flash-Next) defaults behind LLAMA_FLASHNEXT_PROFILE. See llama-fn-auto.h.

#include "llama-fn-auto.h"
#include "llama-fn-l22.h" // [TAG_FN_L22_PASSLOG]

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-model-loader.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <regex>
#include <atomic> // [TAG_FN_L22_XOVER]

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
    { "LLAMA_MOE_HOT_DECAY",          "0.95",  P_ALL }, // [TAG_FN_L3_ADOPT] 0.92 until lever round 3 (L3P_nopool) // decayed hot set (bs3: without it -2.5 % code, -3.4 % prose)
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
    { "SPEC_MTP_COST",                "0",     P_ALL }, // [TAG_FN_L3_ADOPT] the fixed p_min rule (1 until round 3): with no DMA share v1 and v2 (SPEC_MTP_COST=2) shorten the drafts in some server loads (l3/int conf: tau 1.06, -7 %)
    { "LLAMA_MTP_ATTN_WINDOW",        "32768", P_ALL },
    { "LLAMA_MTP_HEAD_ROWS",          "98304", P_ALL },
    { "SPEC_DFT_UBATCH",              "128",   P_ALL }, // the MTP draft context's ubatch (its logits buffer at -ub 8192)
    { "LLAMA_PREFILL_STREAM",         "1",     P_ALL },
    { "LLAMA_PREFILL_STREAM_LEND",    "1",     P_ALL },
    { "LLAMA_PREFILL_STREAM_THREADS", "16",    P_ALL },
    // [TAG_FN_R25_STAGE] round 25: the stream's ring copies with plain stores into a 4 x 8 MiB ring that stays in the L3 (the
    // DMA reads it there): a staged byte crosses DRAM once instead of three times (E:/turbot-gates/flashnext/test/r25/stream)
    { "LLAMA_PREFILL_STREAM_CACHED",  "1",     P_ALL },
    // [TAG_FN_R25_PIN] the copy workers in turn on the two CCDs (left free they stayed on the cache CCD)
    { "LLAMA_PREFILL_STREAM_PIN",     "2",     P_ALL },
    // [TAG_FN_L3_ADOPT] lever round 3 (flashnext/int-l3): ALL-ON against the shipped default at -c 262144, turbot, MTP,
    // 2 interleaved rounds: real use +19.5 % (70.5 vs 59.0 t/s, the SHIP round without a stall; +30.7 % over both rounds); see
    // E:/turbot-gates/flashnext/test/l3/int/REPORT.md
    { "LLAMA_FN_CBUF",               "1",     P_ALL }, // compute-buffer lend to the hot set [TAG_FN_L3_VRAM_CBUF]
    { "LLAMA_MOE_HOT_DECAY_RATIO",   "1.5",   P_ALL }, // L3P_nopool policy [TAG_FN_L3_POLICY_BURST]
    { "LLAMA_MOE_HOT_DECAY_HYST",    "1.0",   P_ALL },
    { "LLAMA_MOE_HOT_DECAY_MIB",     "64",    P_ALL },
    { "LLAMA_MOE_HOT_BURST_MIB",     "512",   P_ALL },
    { "LLAMA_MOE_HOT_BURST_RATIO",   "2",     P_ALL },
    { "LLAMA_FN_GPU",                "1",     P_ALL }, // device levers [TAG_FN_L3_GPU]
    { "LLAMA_FN_GPU_IDXQ8",          "1",     P_ALL }, // [TAG_FN_L4_ADOPT] the round-3 OOB read is fixed (TAG_FN_L4_QSA_IDXDEP, FIXED 2026-10-06 09:24:03): on
    { "LLAMA_FN_GPU_DEFER",          "0",     P_ALL }, // DEFER / CONVWB / HCFUSE / Q8F: not bitwise in the model (greedy identity, first diff at token 47): left off
    { "LLAMA_FN_GPU_CONVWB",         "0",     P_ALL },
    { "LLAMA_FN_GPU_HCFUSE",         "0",     P_ALL },
    { "LLAMA_FN_GPU_Q8F",            "0",     P_ALL },
    { "LLAMA_FN_HOST_EXEC",          "1",     P_BR  }, // host levers [TAG_FN_L3_HOST]
    { "LLAMA_FN_HOST_STEP",          "1",     P_ALL },
    { "LLAMA_FN_HOST_QOS",           "1",     P_ALL },
    { "LLAMA_FN_HOST_POKE",          "1",     P_ALL },
    { "LLAMA_FN_HOST_REARM",         "1",     P_BR  }, // a bridge off after 3 errors comes back with back-off [TAG_FN_L3_HOST_REARM]
    { "LLAMA_FN_HOST_DMAOFF",        "1",     P_BR  }, // no DMA share [TAG_FN_L3_HOST_DMAOFF]
    { "SPEC_MTP_BLOCK_VERIFY",       "1",     P_ALL }, // block verification of sampled drafts [TAG_FN_L3_MTP_BLOCK]
    { "LLAMA_MOE_POOL_SPLIT",        "steal", P_BR  }, // CPU pool levers [TAG_FN_L3_CPU] (LLAMA_MOE_BRIDGE_PF_DEV stays off: FAULT 21:21:24)
    { "LLAMA_MOE_POOL_PF_STREAMS",   "4",     P_BR  },
    { "LLAMA_MOE_POOL_SWPF",         "2",     P_BR  },
    { "LLAMA_MOE_BRIDGE_PF_RANK",    "1",     P_BR  },
    { "LLAMA_MOE_POOL_EXEC_CPU",     "1",     P_BR  },
    // [TAG_FN_L4_ADOPT] lever round 4 (flashnext/int-l4): real use +14.1 % (74.5 vs 65.3 t/s mean of 32K / 131K / 246K),
    // bench code +11.8 %, prose +8.9 % vs main f23c6f638 (2 interleaved rounds, paired). Only switches whose output is bitwise identical (greedy static
    // identity, no MTP and MTP, vs main f23c6f638: E:/turbot-gates/flashnext/test/l4/int/ident_*.txt); A/B at -c 262144 in
    // E:/turbot-gates/flashnext/test/l4/int/ab.txt. Left off: QSA STREAMS (not bitwise), QSA FASPLIT, HOST_MTPFUSE, PROMPT /
    // PROMPT_HOT / UPLOAD (not measured), the gpuhc family (not merged yet).
    { "LLAMA_FN_L4_POST",            "1",     P_ALL }, // top-k + bridge post in one launch, one fence [TAG_FN_L4_POST]
    { "LLAMA_FN_L4_GDNSTALL",        "1",     P_ALL }, // = 2|8: strided conv write-back + state copy after the post [TAG_FN_L4_GDNSTALL]
    { "LLAMA_FN_L4_HOIST",           "1",     P_ALL }, // next GDN layer's gathers after the post [TAG_FN_L4_HOIST]
    { "LLAMA_FN_L4_QSA_SEL",         "1",     P_ALL }, // QSA layer fusions [TAG_FN_L4_QSA]
    { "LLAMA_FN_L4_QSA_KVW",         "1",     P_ALL },
    { "LLAMA_FN_L4_QSA_POOL",        "1",     P_ALL },
    { "LLAMA_FN_L4_QSA_LIST",        "1",     P_ALL },
    { "LLAMA_FN_L4_QSA_QKV",         "1",     P_ALL },
    { "LLAMA_FN_L4_POOLBAR",         "1",     P_BR  }, // CPU pool job without barriers, same values [TAG_FN_L4_MEM]
    { "LLAMA_FN_L4_PFDEV",           "2",     P_BR  }, // device-side prefetch hints (prefetch only) [TAG_FN_L4_MEM]
    { "LLAMA_FN_L4_HOST",            "1",     P_ALL }, // host levers TOPK / SNAP / EARLYOUT / BATCHCPY / PLEPRE [TAG_FN_L4_HOST]
    // [TAG_FN_L5_ADOPT] round 5 (flashnext/int-l5): the hc mixer in three launches (bitwise: identity vs 1f570a818, no MTP and MTP;
    // memcheck / racecheck clean) and the fused MTP draft update; A/B l4/int5/ab5.txt (2 rounds): with the q3 MTP head file
    // bench code +6.6 %, prose +7.9 %, real +2.8 % vs 1f570a818
    { "LLAMA_FN_L4_HC",              "1",     P_ALL }, // [TAG_FN_L4_HC]
    { "LLAMA_FN_L4_HCQ8",            "1",     P_ALL },
    { "LLAMA_FN_L4_HOST_MTPFUSE",    "1",     P_ALL }, // [TAG_FN_L4_HOST_MTPFUSE] drafts only
    // [TAG_FN_L6_ADOPT] round 6 (flashnext/int-l6): the next layer's pre-post weights prefetched into L2 during each bridge
    // wait, and the hc combine kept fusable (graph_optimize holds its inputs); bitwise (identity vs 48de8cd41, no MTP and MTP);
    // A/B 32K real use +2.8 %, ms/step -5.4 % (l6/gk/ab6.log). PRESYNC stays off (net zero).
    { "LLAMA_FN_L6_L2PF",            "1",     P_ALL }, // [TAG_FN_L6_L2PF]
    { "LLAMA_FN_L6_HCCOMB",          "1",     P_ALL }, // [TAG_FN_L6_HCCOMB]
    // [TAG_FN_L7_ADOPT] round 7: the VRAM ceiling keeps 2100 MiB free instead of total/8 (5090: 30,464 instead of 28,416
    // MiB): +2 GB of hot experts, A/B l7/ab7 (2 rounds, 262K) real use +7.2 % (32K +7.7, 131K +9.2, 245K +4.7 %), device use
    // 29.7 GB in decode
    { "LLAMA_FN_VRAM_KEEP_MIB",      "2100",  P_ALL }, // [TAG_FN_L7_VRAMKEEP]
    // [TAG_FN_L8_GRAPHPAD] round 8: the indexer pool count and n_kv in 2048-token steps, so the decode graphs (and their
    // CUDA graphs) keep their shapes 8x longer
    { "LLAMA_KPOOL_PAD",             "512",   P_ALL },
    { "LLAMA_KV_PAD_INDEXER",        "2048",  P_ALL },
    // [TAG_FN_L9_ADOPT] round 9: the prefetch hint picks the next layer's top-k in its own kernel (exact, identity PASS);
    // A/B l9/ab9 (2 rounds, 262K) real use mean +3.3 % (131K +6.4, 245K +3.2 %), prose +5.2 %
    { "LLAMA_FN_L9_HINTTOPK",        "1",     P_BR  }, // [TAG_FN_L9_HINTTOPK]
    // [TAG_FN_L10_ADOPT] round 10: the KV caches' unused cells as extra hot slots in the lower layers, given back slot by slot
    // as the context grows; A/B l10s/ab10v (3 rounds, 262K) real use 88.4 -> 91.6 t/s (32K 92.1 -> 101.2, 131K 85.9 -> 90.2)
    { "LLAMA_FN_L10_SLOTLEND",       "1",     P_ALL }, // [TAG_FN_L10_SLOTLEND]
    // [TAG_FN_L11_ADOPT] round 11: the host token-embedding table and the PLE row cache locked in the working set (0.9 / 1.4
    // page faults per host input before); embd gather 69 -> 5 us (target) and 99 -> 9 us (draft) per graph
    { "LLAMA_FN_L11_EMBDLOCK",       "1",     P_ALL }, // [TAG_FN_L11_EMBDLOCK]
    // [TAG_FN_L12_ADOPT] round 12: a prompt chunk under 384 tokens runs its host experts on the CPU instead of streaming every
    // expert of all 48 layers through the GPU (~3.4 s whatever the size): on a 24K context 32 / 128 / 256 new tokens took
    // 3.39 / 3.40 / 3.38 s, now 0.33 / 0.84 / 1.42 s (l11/pp_mid; the CPU path costs more than the stream from ~600 tokens)
    { "LLAMA_PREFILL_STREAM_MIN",    "768",   P_ALL }, // [TAG_FN_L12_PROMPTMIN] [TAG_FN_L19_ADOPT] 384 until round 19 (the CPU path with PHOT beats the stream to ~768 tokens)
    { "GGML_OP_OFFLOAD_MIN_BATCH",   "768",   P_ALL }, // [TAG_FN_L12_PROMPTMIN] no op-offload copy below it either ([TAG_FN_L19_ADOPT] 384 until round 19)
    // [TAG_FN_L13_SPEC_EQUAL] round 13: streams that draft together draft the same length, so a step is one ubatch
    { "SPEC_MTP_EQUAL_SEQS",         "1",     P_ALL },
    // [TAG_FN_L14_ARENA] round 14: the host layers' routed experts read once into locked RAM, whole layers above a reserve
    { "LLAMA_FN_L14_ARENA",          "0",     P_ALL }, // the copy into private locked RAM: the fallback when the tier cannot run
    // [TAG_FN_L14_TIER] the routed experts locked in place in the mapped file (no copy, no commit, no load-time prefetch)
    { "LLAMA_FN_L14_TIER",           "0",     P_ALL }, // off: with the GPU holding ~33 GB of commit only ~40-55 GiB can be locked (VirtualLock 1455), and gating the hot set on locks froze it (l15 meas15b: 87 / 73 / 73 t/s)
    // [TAG_FN_L14_PFSD2D] the prefill stream copies the experts the hot set holds from their VRAM slots, not the host
    { "LLAMA_PREFILL_STREAM_D2D",         "1",     P_ALL },
    // [TAG_FN_L15_GENSHARE] with 2 connections a streamed prompt chunk (~3.2 s) held the other connection's answer to one
    // step per chunk (0.5-0.9 t/s); it now decodes alone for half the chunk's time between chunks: 28-30 t/s, the prompt
    // ~50 % longer (l15 c_gs_*)
    { "LLAMA_SRV_GEN_SHARE",         "0.5",   P_ALL },
    // [TAG_FN_L16] round 16: exact host trims between graphs, each behind its own switch (same values, same order)
    { "LLAMA_FN_L4_HOST_BATCHCPY",   "2",     P_ALL }, // [TAG_FN_L16_UPLOAD] packed input uploads (mode 2)
    { "LLAMA_FN_L16_SEQRM",          "1",     P_ALL }, // [TAG_FN_L16_SEQRM] seq_rm walks the sequence's own cells
    { "LLAMA_FN_L16_MTP_ASYNC",      "1",     P_ALL }, // [TAG_FN_L16_MTP_ASYNC] no tail synchronize in the MTP draft decode
    { "LLAMA_FN_L16_PLE_GATE",       "1",     P_ALL }, // [TAG_FN_L16_PLEGATE] bridged decode graphs wait for the PLE rows on the device (needs the bridge)
    // [TAG_FN_L17_ADOPT] round 17 (l17/int, the owner's command with vision, per-answer t/s, OLD 2 rounds vs 3): the hot set
    // is right at the start of an answer - short chats 75.2 -> 104.4, follow-ups 94.0 -> 106.2, ~20K answers 94.9 -> 106.6,
    // 131K answers 90.8 -> 100.4; short prompts +0.1 s (the pass). Not adopted: LLAMA_MOE_HOT_STATE (+LLAMA_FN_L17_START_STATE):
    // first answer 74 -> 96 t/s but its load runs in the first prompt (first token 6.2 -> 8.8 s on average); round 18 adopts it
    // with the load in the background ([TAG_FN_L18_FIRSTANS_STATE], below)
    { "LLAMA_FN_L17_START_PRE",      "2048",  P_ALL }, // [TAG_FN_L17_START_PRE]
    { "LLAMA_FN_L17_START_NT",       "1",     P_ALL }, // [TAG_FN_L17_START_NT]
    { "LLAMA_FN_L17_START_BURST",    "64",    P_ALL }, // [TAG_FN_L17_START_BURST]
    { "LLAMA_FN_L17_VISION",         "1",     P_ALL }, // [TAG_FN_L17_VISION_LEND]
    // [TAG_FN_L18_ADOPT] round 18 (l18/int, the owner's command with vision, per-answer t/s + time to first token, OLD = main
    // 364fd4f6a, 2 interleaved rounds + leave-one-out): the next message after a chat-title side request 27.2 s -> 0.68 s to
    // first token (t/s 72 -> 99), short chats 94.9 -> 106.3, first answer after a start 85 -> 95-98 t/s (first token 3.7 ->
    // 2.3 s, ready 22.9 -> 8.7 s), 32K / 131K answers 94.8 / 100.4 -> 106.4 / 102.5. LLAMA_FN_L18_FIRSTANS_OVERLAP stays off:
    // it moved ~60 ms into an answer's first 8 tokens (short chats 106.3 -> 99.3 t/s) for 0.1 s less to first token.
    // LLAMA_MOE_HOT_STATE=cache: the learned set is <LLAMA_CACHE or %LOCALAPPDATA%/llama.cpp>/hotstate/<model file>.hotstate
    { "LLAMA_FN_L18_KEEPCTX",            "1",     P_ALL }, // [TAG_FN_L18_KEEPCTX] a side request no longer throws the conversation away (store on disk)
    { "LLAMA_FN_L18_FIRSTANS_PIN",       "1",     P_ALL }, // [TAG_FN_L18_FIRSTANS_PIN] pool threads on both logical CPUs of their core (bridge wait timeouts)
    { "LLAMA_FN_L18_FIRSTANS_FIT",       "1",     P_ALL }, // [TAG_FN_L18_FIRSTANS_FIT] the hot set fit at the server's ready call
    { "LLAMA_FN_L18_FIRSTANS_WARMLATE",  "1",     P_ALL }, // [TAG_FN_L18_FIRSTANS_WARMLATE] the warm pass starts at ready (with FIT)
    { "LLAMA_FN_L18_FIRSTANS_STATE",     "1",     P_ALL }, // [TAG_FN_L18_FIRSTANS_STATE] the saved set loads in the background
    { "LLAMA_MOE_HOT_STATE",             "cache", P_ALL }, // <LLAMA_CACHE or %LOCALAPPDATA%/llama.cpp>/hotstate/<model file>.hotstate
    { "LLAMA_FN_L17_START_STATE",        "60",    P_ALL }, // [TAG_FN_L17_START_STATE] the set is saved at idle (60 s) and at a clean exit
    { "LLAMA_FN_L18_FIRSTANS_ONCE",      "1",     P_ALL }, // [TAG_FN_L18_FIRSTANS_ONCE] one prompt-end pass per prompt
    { "LLAMA_FN_L18_FIRSTANS_WARM",      "1",     P_ALL }, // [TAG_FN_L18_FIRSTANS_WARM] the warm pass keeps reading during short prompts
    { "LLAMA_MOE_HOT_UP_PIPE",           "1",     P_ALL }, // [TAG_FN_L3_POLICY_UPLOAD] steady-state uploads through two 8 MiB pinned halves
    { "LLAMA_FN_L18_CHURN_LAYERS",       "auto",  P_ALL }, // [TAG_FN_L18_CHURN_LAYERS] the even layout's slots per layer by routing weight
    { "GGML_CUDA_FN_L18_HCUP",           "1",     P_ALL }, // [TAG_FN_L18_GPUFLOOR_HCUP] hc up loads its weights before the PDL wait
    { "LLAMA_FN_L18_GF_GDNOUT",          "1",     P_ALL }, // [TAG_FN_L18_GPUFLOOR_GDNOUT] GDN output norm + gate + q8_1 copy in one launch
    { "LLAMA_FN_L18_GF_QKVZ",            "1",     P_ALL }, // [TAG_FN_L18_GPUFLOOR_QKVZ] GDN qkv and z projections in one launch
    // [TAG_FN_L19_ADOPT] round 19 (l19/int, the owner's command with vision, per-answer t/s + time to first token, OLD = main
    // 09d57d3b8, 1 round + arms; the smartness eval held the GPU until 21:42): the ttft family and convstore's assistant-turn
    // checkpoint. Follow-ups 0.98 -> 0.36 s to first token (median), short chats 0.43 -> 0.31 s, first answer 2.21 -> 1.55 s,
    // a first message 5 s after ready 18.8 -> 5.5 s, 128-token turns 2.1 -> 1.5 s, 450 / 750-token turns 4.2-4.5 -> 2.5-2.7 /
    // 3.4-3.6 s (CPU path to 768 tokens, LLAMA_PREFILL_STREAM_MIN / GGML_OP_OFFLOAD_MIN_BATCH above), a client that drops the
    // last answer's reasoning re-reads 85 instead of 517 tokens (5.1 -> 0.8 s); answer t/s unchanged. Not adopted: convstore's
    // 16-conversation store / keeping rule / disk checkpoints / system-block stores (agent workload cache share 75 -> 82 % but
    // TTFT sum 117.1 -> 118.6 s: a >= 384-token chunk costs one stream pass whatever its size), LLAMA_FN_L19_DRAFT_ADAPT (+ CUT:
    // 105 -> 75 t/s, the cut dropped ~99 % of drafts after an early 4-row verify sample priced width 2 at 42.75 ms) and
    // LLAMA_FN_L19_DRAFT_F16KV (neutral: pooled decode 111.7 -> 111.0 t/s)
    { "LLAMA_FN_L19_TTFT_HUSH",         "3",      P_ALL }, // [TAG_FN_L19_TTFT_HUSH] the warm pass / state read-ahead pause during a FULL reserve and prompt batches
    { "LLAMA_FN_L19_TTFT_HUSH_MS",      "10000",  P_ALL }, // [TAG_FN_L19_TTFT_HUSH_BUDGET] at most 10 s of those pauses in all: the warm pass ends under continuous prompts
    { "LLAMA_FN_L19_TTFT_EARLY",        "256",    P_ALL }, // [TAG_FN_L19_TTFT_EARLY] a <= 256-token prompt during the warm pass stays SMALL (no FULL reserve)
    { "LLAMA_FN_L19_TTFT_PHOT",         "1",      P_ALL }, // [TAG_FN_L19_TTFT_PHOT] CPU-path prompt graphs compute their hot experts on the GPU
    { "LLAMA_FN_L4_PROMPT",             "160",    P_ALL }, // [TAG_FN_L4_MEM_PROMPT] a <= 160-token prompt stays SMALL (ubatches of <= 31 tokens, no FULL switch)
    { "LLAMA_FN_L19_TTFT_TAILFULL",     "512",    P_ALL }, // [TAG_FN_L19_TTFT_TAILFULL] the narrow tail of a split short prompt keeps FULL
    { "LLAMA_FN_L19_CONVSTORE",         "1",      P_ALL }, // [TAG_FN_L19_CONVSTORE] with the four lines below: the assistant-turn checkpoint only (_OPENER auto)
    { "LLAMA_FN_L19_CONVSTORE_N",       "8",      P_ALL }, //   [TAG_FN_L24_KEEPVAL] 8 conversations (4 before; the files are on disk only, the 32 GiB budget binds)
    { "LLAMA_FN_L19_CONVSTORE_EVICT",   "0",      P_ALL }, //   least recently stored first, as before
    { "LLAMA_FN_L19_CONVSTORE_CKPT",    "0",      P_ALL }, //   the checkpoint ring as before (no keeping rule, no disk checkpoints)
    { "LLAMA_FN_L19_CONVSTORE_SYS",     "0",      P_ALL }, //   no store when a prompt keeps only the system block
    { "LLAMA_FN_L24_KEEPVAL",           "1",      P_ALL }, // [TAG_FN_L24_KEEPVAL] evict by re-read cost x recency, the biggest active conversation kept (owner 10-22: 4 side stores evicted a 48.6K chat)
    // [TAG_FN_L21_DEC] round 21 decode / TTFT items (exact). MTP_VISION stays off: on the owner's command with a
    // 2560x1600 image (greedy, one pair, l21/dec) the answer after the image drafted 0.784 -> 0.739 accepted and an
    // edit after it 0.870 -> 0.792, and the 4000 image rows cost the draft context ~0.9 s of the prompt
    { "LLAMA_FN_L21_MTP_VISION",        "0",      P_ALL }, // [TAG_FN_L21_MTP_VISION] the MTP draft decodes image rows (upstream #30257 + #30397)
    { "LLAMA_FN_L21_MTP_ROWS",          "1",      P_ALL }, // [TAG_FN_L21_MTP_ROWS] MTP catch-up rows stop after their K/V (no MoE / hc / head)
    { "LLAMA_FN_L21_MTP_STATE",         "1",      P_ALL }, // [TAG_FN_L21_MTP_STATE] context checkpoints carry the MTP h / fused rows (#30141)
    { "LLAMA_FN_L21_MEDIA_CKPT",        "1",      P_ALL }, // [TAG_FN_L21_MEDIA_CKPT] a checkpoint right after the last media chunk (#30201)
    // [TAG_FN_L21_VRAM_DXGI] the smoke read CUDA 23,803 = DXGI 23,803 MiB free (budget 31,419, this process 7,616) with
    // nothing else on the card, so the min changes nothing then and only shrinks the fit when another process holds VRAM
    { "LLAMA_FN_L21_VRAM_DXGI",         "1",      P_ALL }, // [TAG_FN_L21_VRAM_DXGI] the hot set's fit from min(CUDA free, DXGI budget - usage)
    // [TAG_FN_L20_ADOPT] round 20 (l20/int, the owner's command with vision, MTPq3, 1 slot, 262K, OLD = int-l19 20a90baca, 1 round):
    // the 16K prompt ubatches of the bigprompt family. Fresh prompts after a warm-in: 32K 2052 -> 2329 t/s (TTFT 15.9 -> 14.0 s),
    // 131K 2371 -> 2674 t/s (55.3 -> 49.0 s), 245K 2402 -> 2614 t/s (102.3 -> 94.0 s); 16K-ubatch KLD in the band. Not adopted:
    // LLAMA_FN_L20_BIGPROMPT_DFTUB / _PLEPRE (no gain), the draftfix and convfix switches (fixed, but no measured win)
    { "LLAMA_FN_L20_BIGPROMPT_UB",      "16384",  P_ALL }, // [TAG_FN_L20_BIGPROMPT_UB] prompts wider than the command's -ub run in 16384-token ubatches (FULL16)
    // [TAG_FN_L21_BIGUB] round 21: the 16K ubatches only for prompts above 16K, the tail sized for 8K FULL (start-up layout and KV slot lend as without them), the extra from the borrow slots - the test stage decides by KLD + needles + the first answer after a start
    { "LLAMA_FN_L21_BIGUB",             "1",      P_ALL }, // [TAG_FN_L21_BIGUB]
    // [TAG_FN_L22] round 22 (E:/turbot-gates/flashnext/test/r22/PLAN.md section 6): the build defaults of the test's NEW
    // arm, one row per switch (package, tag). Code default of every new switch = off; the answer-changing X levers, PIN,
    // DMASHARE and CPUSHARE stay 0 here (owner calls). No row: LLAMA_FN_L22_LOGITDUMP (unset) and _PIN_MAX_MIB (no value).
    { "LLAMA_FN_L22_PASSLOG",               "1",       P_ALL }, // P0 [TAG_FN_L22_PASSLOG] per-pass / switch / residency / pin / media / turn lines (0/1/2)
    { "LLAMA_FN_L22_IMGPASS",               "1",       P_ALL }, // P1 P3 [TAG_FN_L22_IMGPASS] one prompt pass per image turn
    { "LLAMA_FN_L22_IMGPASS_MASK",          "auto",    P_ALL }, // P1 [TAG_FN_L22_IMGPASS_MASK] auto = keys | keys | explicit
    { "LLAMA_FN_L22_IMGPASS_MASK_MIB",      "1024",    P_ALL }, // P3 the merged rows' cap in explicit-mask mode
    { "LLAMA_FN_L22_SAMPSTATIC",            "1",       P_ALL }, // P1 P4 [TAG_FN_L22_SAMPSTATIC] E7 (#30223)
    { "LLAMA_FN_L22_KVGROW",                "1",       P_ALL }, // P1 [TAG_FN_L22_KVGROW] F6 + OOM hook
    { "LLAMA_FN_L22_VLEND_HOLD",            "1",       P_ALL }, // P2 P3 [TAG_FN_L22_VLEND_HOLD] C1 one lend per turn
    { "LLAMA_FN_L22_VLEND_SIZE",            "1",       P_ALL }, // P2 [TAG_FN_L22_VLEND_SIZE] C8
    { "LLAMA_FN_L22_IMGCACHE",              "24",      P_ALL }, // P2 [TAG_FN_L22_IMGCACHE] C4 entries (0 off)
    { "LLAMA_FN_L22_IMGDEC_THREADS",        "4",       P_ALL }, // P2 [TAG_FN_L22_IMGDEC] C5
    { "LLAMA_FN_L22_IMG_MAXROWS",           "0",       P_ALL }, // P2 [TAG_FN_L22_IMG_MAXROWS] C10 changes answers: owner
    { "LLAMA_FN_L22_TOKCACHE",              "1",       P_ALL }, // P2 [TAG_FN_L22_TOKCACHE] E11
    { "LLAMA_FN_L22_TOKCACHE_VERIFY",       "1",       P_ALL }, // P2
    { "LLAMA_FN_L22_JSONFLOAT",             "1",       P_ALL }, // P2 [TAG_FN_L22_JSONFLOAT] E8 towards the reference
    { "LLAMA_FN_L22_PLEQ8",                 "1",       P_ALL }, // [TAG_FN_L22_PLEQ8] <model>.ple.q8_0 n-gram table when present
    { "LLAMA_FN_L22_VLRESIZE",              "0",       P_ALL }, // P2 F7 until an OCR A/B
    { "LLAMA_FN_L22_VLBATCH",               "0",       P_ALL }, // P2b until its encoder KLD
    { "LLAMA_FN_L22_MTP_KEEP",              "1",       P_ALL }, // P3 [TAG_FN_L22_MTP_KEEP] C0
    { "LLAMA_FN_L22_MEDIA_ANCHOR",          "1",       P_ALL }, // P3 [TAG_FN_L22_MEDIA_ANCHOR] C3
    { "LLAMA_FN_L22_CKPT_KEEP",             "1",       P_ALL }, // P3 [TAG_FN_L22_CKPT_KEEP] A5
    { "LLAMA_FN_L22_CKPT_SPLIT",            "1",       P_ALL }, // P3 [TAG_FN_L22_CKPT_SPLIT] X1 (kld, grid)
    { "LLAMA_FN_L22_LENDCLASS",             "1",       P_ALL }, // P4 [TAG_FN_L22_LENDCLASS] width classes
    { "LLAMA_FN_L22_REFILL",                "1",       P_ALL }, // P4 [TAG_FN_L22_REFILL]
    { "LLAMA_FN_L22_READAHEAD",             "1",       P_ALL }, // P4 [TAG_FN_L22_READAHEAD]
    { "LLAMA_FN_L22_KEEPLEND",              "1",       P_ALL }, // P3 P4 [TAG_FN_L22_KEEPLEND] placement
    { "LLAMA_FN_L22_ENDOVERLAP",            "1",       P_ALL }, // P4 [TAG_FN_L22_ENDOVERLAP] placement
    { "LLAMA_FN_L22_GRAPHKEY",              "1",       P_ALL }, // P4 P7 [TAG_FN_L22_GRAPHKEY] G4
    { "LLAMA_FN_L22_SCHEDBUF",              "1",       P_ALL }, // P4 P7 [TAG_FN_L22_SCHEDBUF] M1
    { "GGML_SCHED_FN_L22_SPLITS",           "4096",    P_ALL }, // P7 (getenv in ggml)
    { "LLAMA_FN_L22_RESSET",                "1",       P_ALL }, // P5 [TAG_FN_L22_RESSET] M2 resident set
    { "LLAMA_FN_L22_REWARM",                "1",       P_ALL }, // P5 [TAG_FN_L22_REWARM] N2 idle re-warm
    { "LLAMA_FN_L22_SSDTIER",               "2",       P_ALL }, // P4 P5 [TAG_FN_L22_SSDTIER] (0/1/2)
    { "LLAMA_FN_L22_SSDTIER_IO",            "8",       P_ALL }, // P5 I/O threads
    { "LLAMA_FN_L22_PIN",                   "0",       P_ALL }, // P4 P6 [TAG_FN_L22_PIN] GPU-fault class: the pin block sets 1; owner
    { "LLAMA_FN_L22_PIN_UNIT",              "ranges",  P_ALL }, // P6
    { "LLAMA_FN_L22_PIN_MARGIN_MIB",        "16384",   P_ALL }, // P6 NON_LOCAL margin ([TAG_FN_L23_PINFIX] R-L23-5: 8192 left 8.4 GiB, TDR)
    { "LLAMA_FN_L22_PIN_LOCAL_MARGIN_MIB",  "1024",    P_ALL }, // P6 LOCAL margin
    { "LLAMA_FN_L22_PIN_FLOOR_MIB",         "6144",    P_ALL }, // P6
    { "LLAMA_FN_L22_PIN_ARENA_MIB",         "16384",   P_ALL }, // P6 anonymous fallback
    { "GGML_CUDA_FN_L22_GRAPHKEEP",         "1",       P_ALL }, // P7 [TAG_FN_L22_GRAPHKEEP] G1
    { "GGML_CUDA_FN_L22_GRAPH_CAP",         "512,64",  P_ALL }, // P7
    { "GGML_CUDA_FN_L22_GRAPH_MAXROWS",     "128",     P_ALL }, // P7
    { "GGML_CUDA_FN_L22_GRAPH2",            "1",       P_ALL }, // P7 [TAG_FN_L22_GRAPH2] G2
    { "GGML_CUDA_FN_L22_GRAPHDIAG",         "0",       P_ALL }, // P7 log
    { "GGML_CUDA_FN_L22_SKCAP",             "1",       P_ALL }, // P7 [TAG_FN_L22_SKCAP]
    { "GGML_CUDA_FN_L22_MMQ_Y8",            "1",       P_ALL }, // P7 [TAG_FN_L22_MMQ_Y8] E4 (#30493)
    { "GGML_CUDA_FN_L22_GLUQ8",             "1",       P_ALL }, // P7 [TAG_FN_L22_GLUQ8] E5 (#30489)
    { "GGML_CUDA_MMQ_GLU_FUSION",           "1",       P_ALL }, // P7 #29948 mode 1
    { "GGML_CUDA_FN_L22_MMVQ_R1",           "0",       P_ALL }, // P7 by bench
    { "GGML_CUDA_GDN_MMA_SNAP",             "1",       P_ALL }, // P7 #29353 (kld)
    { "GGML_CUDA_GDN_MMA_CHUNK",            "0",       P_ALL }, // P7 #29353 chunked prefill
    { "LLAMA_MOE_POOL_STAGGER",             "0",       P_BR  }, // P8 [TAG_FN_L22_POOL_STAGGER] by stall A/B
    { "LLAMA_MOE_POOL_PRETOUCH",            "0",       P_BR  }, // P8 [TAG_FN_L22_PRETOUCH] by stall A/B
    { "LLAMA_MOE_POOL_JOBPF",               "1",       P_BR  }, // P8 [TAG_FN_L22_JOBPF]
    { "LLAMA_FN_L22_X_STATS",               "0",       P_ALL }, // P8 [TAG_FN_L22_X_STATS] (1 in T5)
    { "LLAMA_FN_L22_X_TAILSKIP",            "0",       P_ALL }, // P0 P8 [TAG_FN_L22_X_TAILSKIP] changes answers: owner
    { "LLAMA_FN_L22_X_TAILSKIP_W",          "0",       P_ALL }, // P8
    { "LLAMA_FN_L22_X_TAILSKIP_MULTI",      "0",       P_ALL }, // P0
    { "LLAMA_FN_L22_X_TAILSKIP_PF",         "0",       P_ALL }, // P8
    { "LLAMA_FN_L22_ONEPROMPT",             "1",       P_ALL }, // P9 [TAG_FN_L22_ONEPROMPT] D1
    { "LLAMA_FN_L22_GENSHARE2",             "1",       P_ALL }, // P9 [TAG_FN_L22_GENSHARE2] D2
    { "LLAMA_FN_L22_PAUSE",                 "1",       P_ALL }, // P9 [TAG_FN_L22_PAUSE] D10
    { "LLAMA_FN_L22_PAUSE_TOKENS",          "2048",    P_ALL }, // P9
    { "LLAMA_FN_L22_PAUSE_S",               "8",       P_ALL }, // P9
    { "LLAMA_FN_L22_SHAREKV",               "1",       P_ALL }, // P9 [TAG_FN_L22_SHAREKV] D9
    { "LLAMA_FN_L22_DRAFT2",                "0",       P_ALL }, // P9 [TAG_FN_L22_DRAFT2] D7 by A/B
    { "LLAMA_FN_L24_ATTACH",                "1",       P_ALL }, // [TAG_FN_L24_ATTACH] #30970 slot attach, idle donor whole (bitwise exact)
    { "LLAMA_FN_L24_ATTACH_CKPT",           "0",       P_ALL }, // [TAG_FN_L24_ATTACH] at a busy donor's checkpoint: off until the pool placement fix
    { "LLAMA_FN_L22_STAGERS",               "auto",    P_ALL }, // P10 [TAG_FN_L22_STAGERS] M7
    { "LLAMA_FN_L22_XOVER",                 "1",       P_ALL }, // P10 [TAG_FN_L22_XOVER] placement
    { "LLAMA_FN_L22_UBPLAN",                "1",       P_ALL }, // P10 [TAG_FN_L22_UBPLAN] M9 (kld)
    { "LLAMA_FN_L22_BIGUB2",                "1",       P_ALL }, // P10 [TAG_FN_L22_BIGUB2] D3 (kld)
    { "LLAMA_FN_L23_BG2POOLS",              "1",       P_ALL }, // [TAG_FN_L23_TWOSLOT] O7: BIGUB2's k-pools for one sequence (kld)
    { "LLAMA_FN_L22_SMALLKEEP",             "1",       P_ALL }, // P10 [TAG_FN_L22_SMALLKEEP] G3
    { "LLAMA_FN_L22_X_DEFER",               "0",       P_ALL }, // P11 [TAG_FN_L22_X_DEFER] changes answers: owner
    { "LLAMA_FN_L22_X_DEFER_W",             "0.1",     P_ALL }, // P11
    { "LLAMA_FN_L22_X_DEFER_R",             "7",       P_ALL }, // P11
    { "LLAMA_KV_ROWS",                      "0",       P_ALL }, // P13 [TAG_UPSTREAM_29510] not for qwen4exp (the 27B launcher sets it)
    { "LLAMA_FN_L22_QSA_SEQPOOLS",          "1",       P_ALL }, // P13 [TAG_FN_L22_QSA_SEQPOOLS] D12
    { "LLAMA_FN_L22_QSA_SHAREPFX",          "1",       P_ALL }, // P13 [TAG_FN_L22_QSA_SHAREPFX] R-P13-4 (exact)
    { "LLAMA_FN_L22_TBO",                   "0",       P_ALL }, // D5 [TAG_FN_L22_TBO] two-chain decode (kld; 2 active sequences only), R-D5-6: opt-in (owner 10-18)
    { "LLAMA_FN_L24_GENCKPT",               "1",       P_ALL }, // [TAG_FN_L24_GENCKPT] checkpoints inside long answers (exact: state copies)
    { "LLAMA_FN_L24_OPFAIR",                "1",       P_ALL }, // [TAG_FN_L24_OPFAIR] ONEPROMPT shortest remaining first, a prompt gives way at most once per request (2+ slots)
    { "LLAMA_FN_L24_GSCAP",                 "1",       P_ALL }, // [TAG_FN_L24_GSCAP] GENSHARE2 window cap per row fraction (2+ slots)
    { "LLAMA_FN_R25_REFILL",                "1",       P_ALL }, // [TAG_FN_R25_REFILL] sync refills through the cache-resident copy pool (exact: same bytes)
    { "LLAMA_FN_R25_REFILL_THREADS",        "8",       P_ALL }, // R25 copy threads (own stream each)
    { "LLAMA_FN_R25_REFILL_HALF_KIB",       "2048",    P_ALL }, // R25 pinned half per thread (two each)
    { "LLAMA_FN_R25_SBSYNC_MIB",            "1024",    P_ALL }, // R25 a borrow return up to this size lands at once (0: upload worker)
    { "LLAMA_FN_L22_DMASHARE",              "0",       P_ALL }, // P14 [TAG_FN_L22_DMASHARE] D11, the pin block sets 1; owner
    { "LLAMA_FN_L22_DMASHARE_SLOTS",        "4",       P_ALL }, // P14
    { "LLAMA_FN_L22_X_PRIOR",               "0",       P_ALL }, // P12 [TAG_FN_L22_X_PRIOR] changes answers: owner
    { "LLAMA_FN_L22_X_PRIOR_STATS",         "raw",     P_ALL }, // P12
    { "LLAMA_FN_L22_X_RESIDENT",            "0",       P_ALL }, // P12 changes answers: owner
    { "LLAMA_FN_L22_CPUSHARE",              "0",       P_ALL }, // P16 [TAG_FN_L22_CPUSHARE] M11, the pin block sets 1; owner
    { "LLAMA_FN_L22_CPUSHARE_MAX",          "3072",    P_ALL }, // P16
    { "LLAMA_FN_L22_CPUSHARE_ROWS",         "8",       P_ALL }, // P16
};

} // namespace

// [TAG_FN_L22_XOVER] P10 R1 (src/llama-fn-l22.h): the crossover XOVER set (0: unset, the PHOT rule reads
// GGML_OP_OFFLOAD_MIN_BATCH as before)
static std::atomic<int32_t> g_l22_offload_min{0};

int32_t llama_fn_l22_offload_min(void) {
    return g_l22_offload_min.load(std::memory_order_relaxed);
}

void llama_fn_l22_offload_min_set(int32_t v) {
    g_l22_offload_min.store(v > 0 ? v : 0, std::memory_order_relaxed);
}

namespace {

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

// [TAG_FN_L18_CHURN_LAYERS] -----------------------------------------------------------------------------------------

std::vector<int32_t> llama_fn_weighted_slots(const std::vector<size_t> & bytes_per_expert, size_t budget, int32_t n_expert,
        const std::vector<float> & weights) {
    const size_t n = bytes_per_expert.size();
    bool ok = weights.size() == n && n > 0 && n_expert > 0;
    for (size_t i = 0; ok && i < n; ++i) {
        ok = std::isfinite(weights[i]) && weights[i] > 0.0f && bytes_per_expert[i] > 0;
    }
    if (!ok) {
        return std::vector<int32_t>(n, llama_fn_even_slots(bytes_per_expert, budget, n_expert));
    }
    // n_l(lam) = min(n_expert, floor(lam * w_l)): the largest lam whose slots plus every layer's zero slot fit the budget
    auto counts = [&](double lam, std::vector<int32_t> & out) {
        size_t used = 0;
        for (size_t i = 0; i < n; ++i) {
            const double x = std::floor(lam * (double) weights[i]);
            out[i] = (int32_t) std::min<double>((double) n_expert, std::max(0.0, x));
            used += ((size_t) out[i] + 1)*bytes_per_expert[i];
        }
        return used;
    };
    std::vector<int32_t> res(n, 0);
    std::vector<int32_t> tmp(n, 0);
    float w_min = weights[0];
    for (float w : weights) {
        w_min = std::min(w_min, w);
    }
    double lo = 0.0;
    double hi = (double) n_expert / (double) w_min + 1.0;
    if (counts(0.0, res) > budget) {
        return std::vector<int32_t>(n, 0);
    }
    for (int it = 0; it < 64; ++it) {
        const double mid = 0.5*(lo + hi);
        if (counts(mid, tmp) <= budget) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    size_t used = counts(lo, res);
    // the rest of the budget, one slot at a time, to the layers furthest below their share (ties: layer order)
    for (;;) {
        size_t best = n;
        double best_def = 0.0;
        for (size_t i = 0; i < n; ++i) {
            if (res[i] >= n_expert || used + bytes_per_expert[i] > budget) {
                continue;
            }
            const double def = lo * (double) weights[i] - (double) res[i];
            if (best == n || def > best_def) {
                best = i;
                best_def = def;
            }
        }
        if (best == n) {
            break;
        }
        res[best]++;
        used += bytes_per_expert[best];
    }
    return res;
}

std::vector<float> llama_fn_parse_weights(const char * s) {
    std::vector<float> out;
    if (s == nullptr) {
        return out;
    }
    const char * p = s;
    while (*p) {
        char * end = nullptr;
        const float v = strtof(p, &end);
        if (end == p || !std::isfinite(v) || v <= 0.0f) {
            return {};
        }
        out.push_back(v);
        p = end;
        while (*p == ' ' || *p == '\t') {
            ++p;
        }
        if (*p == ',') {
            ++p;
        } else if (*p != '\0') {
            return {};
        }
    }
    return out;
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
    const uint32_t t     = std::max<uint32_t>(1, std::min<uint32_t>(requested, below));
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

uint32_t llama_fn_cbuf_short_max(uint32_t short_max, uint32_t early, bool warm_running, uint32_t n_tokens, uint32_t small_t) {
    if (early > 0 && warm_running && n_tokens > small_t && n_tokens <= early) {
        return std::max(short_max, early);
    }
    return short_max;
}

bool llama_fn_cbuf_tail_full(uint32_t tailfull, bool state_full, bool short_path, uint32_t n_tokens, uint32_t n_outputs,
        uint32_t prev_n, bool prev_out, uint32_t small_t) {
    return tailfull > 0 && state_full && !short_path && n_tokens <= small_t && n_outputs > 0 && !prev_out &&
           prev_n > small_t && prev_n < tailfull;
}

size_t llama_fn_cbuf_part_lo(size_t lo, size_t hi, size_t seg, size_t need) {
    if (need == 0 || seg == 0 || hi <= lo || need >= hi - lo) {
        return lo;
    }
    return lo + (hi - lo - need) / seg * seg;
}

uint32_t llama_fn_l20_full_width(uint32_t base_ub, uint32_t n_ubatch, uint32_t n_tokens, int64_t after) {
    if (base_ub == 0 || base_ub >= n_ubatch) {
        return n_ubatch;
    }
    const bool big = n_tokens > base_ub || (after > 0 && (int64_t) n_tokens + after > (int64_t) base_ub);
    return big ? n_ubatch : base_ub;
}

uint32_t llama_fn_l21_full_width(uint32_t base_ub, uint32_t n_ubatch, uint32_t n_tokens, int64_t after) {
    if (base_ub == 0 || base_ub >= n_ubatch) {
        return n_ubatch;
    }
    const bool big = after >= 0 ? (int64_t) n_tokens + after > (int64_t) n_ubatch : n_tokens >= n_ubatch;
    return big ? n_ubatch : base_ub;
}

bool llama_fn_l20_keep_full(uint32_t base_ub, bool wide, bool short_path, bool state_full, int64_t after) {
    return base_ub > 0 && !wide && !short_path && state_full && after > 0;
}

bool llama_fn_v0254_narrow_full(uint32_t n_tokens, uint32_t n_outputs, int64_t after, uint32_t n_batch,
        uint32_t small_t, uint32_t short_max, uint32_t early, bool warm_running) {
    if (n_tokens == 0 || n_tokens > small_t || n_outputs >= n_tokens || after <= 0) {
        return false; // a wide batch (its own rule), an answer-only batch (R-L23-3) or the prompt's last part
    }
    const int64_t rest = n_batch > 0 ? std::min<int64_t>(after, (int64_t) n_batch) : after;
    if (rest <= (int64_t) small_t) {
        return false; // a narrow rest: SMALL
    }
    const uint32_t r    = (uint32_t) std::min<int64_t>(rest, (int64_t) UINT32_MAX);
    const uint32_t smax = llama_fn_cbuf_short_max(short_max, early, warm_running, r, small_t);
    return !(smax > 0 && r <= smax); // a rest on the short path stays SMALL, so does this part
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

// [TAG_FN_L22_P4] ---------------------------------------------------------------------------------------------------

uint32_t llama_fn_l22_lend_class(uint32_t base_ub, uint32_t n_tokens, int64_t after) {
    if (base_ub == 0 || after < 0) {
        return 0;
    }
    const int64_t total = (int64_t) n_tokens + after;
    static const uint32_t k_cls[] = { 1024, 2048, 4096 };
    for (uint32_t c : k_cls) {
        if (c >= base_ub) {
            break;
        }
        if (total <= (int64_t) c) {
            return c;
        }
    }
    return 0;
}

size_t llama_fn_l22_class_need(size_t full_b, size_t small_b, size_t pool_b, size_t cap) {
    if (full_b <= small_b) {
        return 0;
    }
    const size_t slack = std::max<size_t>((size_t) 128 << 20, full_b/16);
    const size_t need  = full_b - small_b + pool_b + slack;
    return cap > 0 ? std::min(need, cap) : need;
}

uint32_t llama_fn_l22_sched_nodes(uint32_t formula, uint32_t nodes_measured) {
    if (nodes_measured == 0) {
        return formula;
    }
    const uint64_t n = (uint64_t) nodes_measured*3/2 + 8192;
    return (uint32_t) std::min<uint64_t>(formula, n);
}

uint32_t llama_fn_l22_sampling_nodes(uint32_t old_count, uint32_t sum_nodes, uint32_t n_out_per_seq) {
    const uint64_t s = (uint64_t) sum_nodes*std::max<uint32_t>(1, n_out_per_seq);
    return (uint32_t) std::max<uint64_t>(old_count, std::min<uint64_t>(s, UINT32_MAX));
}

void llama_fn_l22_io_merge(const std::vector<llama_fn_l22_io_range> & r, std::vector<std::pair<size_t, size_t>> & groups) {
    groups.clear();
    constexpr size_t small = (size_t) 64 << 10;
    size_t i = 0;
    while (i < r.size()) {
        size_t j = i + 1;
        if (r[i].size < small) {
            size_t hi      = r[i].offset + r[i].size;
            size_t payload = r[i].size;
            while (j < r.size()) {
                const auto & w = r[j];
                if (w.tensor != r[i].tensor || w.size >= small || w.offset < hi) {
                    break;
                }
                const size_t nhi = w.offset + w.size;
                if (nhi - r[i].offset > 8*(payload + w.size) + ((size_t) 1 << 20)) {
                    break;
                }
                hi       = nhi;
                payload += w.size;
                ++j;
            }
        }
        groups.push_back({ i, j });
        i = j;
    }
}

uint32_t llama_fn_l22_next_width(uint32_t base_ub, uint32_t n_ubatch, bool l21_bigub, bool l21_big_off, bool lendclass,
        int64_t after) {
    if (base_ub == 0) {
        return n_ubatch;
    }
    uint32_t w = !l21_bigub ? llama_fn_l20_full_width(base_ub, n_ubatch, 0, after) :
                 l21_big_off ? base_ub : llama_fn_l21_full_width(base_ub, n_ubatch, 0, after);
    if (lendclass && w == base_ub) {
        const uint32_t c = llama_fn_l22_lend_class(base_ub, 0, after);
        if (c > 0) {
            w = c;
        }
    }
    return w;
}

// [TAG_FN_L22_P10] ---------------------------------------------------------------------------------------------------

int llama_fn_l22_stagers_mode(const char * v) {
    if (!v) {
        return 0;
    }
    std::string s;
    for (const char * p = v; *p; ++p) {
        if (!isspace((unsigned char) *p)) {
            s += (char) tolower((unsigned char) *p);
        }
    }
    if (s.empty() || s == "off" || s == "false" || s == "no") {
        return 0;
    }
    if (s == "auto" || s == "on" || s == "true" || s == "yes") {
        return -1;
    }
    char * end = nullptr;
    const long n = strtol(s.c_str(), &end, 10);
    if (end == s.c_str() || *end != 0 || n <= 0) {
        return 0;
    }
    return n == 1 ? -1 : (int) std::min<long>(n, 64);
}

int llama_fn_l22_stagers(uint64_t staged, uint64_t host, int n_max, int n_min) {
    n_max = std::max(1, n_max);
    n_min = std::max(1, std::min(n_min, n_max));
    if (host == 0) {
        return n_max;
    }
    staged = std::min(staged, host);
    const int n = (int) std::lround((double) n_max * (double) staged / (double) host);
    return std::max(n_min, std::min(n_max, n));
}

int llama_fn_l22_depth_bucket(int64_t n_kv) {
    return n_kv < 16384 ? 0 : n_kv < 65536 ? 1 : n_kv < 163840 ? 2 : 3;
}

void llama_fn_l22_cost_init(llama_fn_l22_cost & c, bool pinned) {
    c = llama_fn_l22_cost();
    c.pinned = pinned;
    for (int b = 0; b < 4; ++b) {
        c.pass_ms[b]    = pinned ? 1700.0 : 3400.0; // EST: a round-20 8K pass 3.4-4 s, pinned about half
        c.gpu_ms_row[b] = 0.42;                     // EST: a 16K pass ~7.0 s (round 20)
        c.cpu_ms_row[b] = 4.55;                     // EST: the CPU path at 220 rows/s (A doc)
    }
    c.fix_ms = 240.0; // EST: the 10-08 refill
    c.big_ms = 300.0; // EST: borrow + unborrow
}

void llama_fn_l22_cost_repin(llama_fn_l22_cost & c, bool pinned) {
    if (c.pinned == pinned) {
        return;
    }
    c.pinned = pinned;
    for (int b = 0; b < 4; ++b) {
        c.pass_ms[b] = pinned ? 1700.0 : 3400.0;
        c.n_pass[b]  = 0;
        c.n_obs[b]   = 0;
    }
}

static void fn_l22_ewma(double & est, uint32_t & n, double x) {
    if (!std::isfinite(x) || x < 0.0) {
        return;
    }
    if (n == 0) {
        est = x; // the first measurement replaces the prior
    } else {
        if (n >= 3) {
            x = std::min(2.0*est, std::max(0.5*est, x)); // stalls of 2.5-14 s must not move it far
        }
        est += 0.25*(x - est);
    }
    n++;
}

static int fn_l22_bucket(int b) {
    return std::max(0, std::min(3, b));
}

void llama_fn_l22_cost_pass(llama_fn_l22_cost & c, int b, uint32_t rows, double wall_ms, double wait_ms) {
    if (b < 0 || b > 3 || rows == 0 || !(wall_ms > 0.0) || !std::isfinite(wall_ms)) {
        return;
    }
    wait_ms = std::max(0.0, std::min(wall_ms, std::isfinite(wait_ms) ? wait_ms : 0.0));
    c.n_obs[b]++;
    if (wait_ms >= 0.1*wall_ms) {
        fn_l22_ewma(c.pass_ms[b], c.n_pass[b], wall_ms);
    } else if (c.pass_ms[b] > wall_ms) {
        c.pass_ms[b] = wall_ms; // compute-bound: the copies took at most the wall
    }
    if (rows >= 2048) {
        fn_l22_ewma(c.gpu_ms_row[b], c.n_gpu[b], (wall_ms - wait_ms) / rows);
    }
}

void llama_fn_l22_cost_cpu(llama_fn_l22_cost & c, int b, uint32_t rows, double wall_ms) {
    if (b < 0 || b > 3 || rows < 128 || !(wall_ms > 0.0)) {
        return;
    }
    fn_l22_ewma(c.cpu_ms_row[b], c.n_cpu[b], wall_ms / rows);
}

void llama_fn_l22_cost_fix(llama_fn_l22_cost & c, double fix_ms, double big_ms) {
    if (fix_ms >= 0.0) {
        fn_l22_ewma(c.fix_ms, c.n_fix, fix_ms);
    }
    if (big_ms >= 0.0) {
        fn_l22_ewma(c.big_ms, c.n_big, big_ms);
    }
}

double llama_fn_l22_est_pass(const llama_fn_l22_cost & c, int b, uint32_t rows) {
    b = fn_l22_bucket(b);
    return std::max(c.pass_ms[b], (double) rows * c.gpu_ms_row[b]);
}

double llama_fn_l22_est_cpu(const llama_fn_l22_cost & c, int b, uint32_t rows) {
    return (double) rows * c.cpu_ms_row[fn_l22_bucket(b)];
}

int32_t llama_fn_l22_xover(const llama_fn_l22_cost & c, int b, int32_t cur, int32_t lo, int32_t hi) {
    if (b < 0 || b > 3 || lo <= 0 || hi < lo || c.n_obs[b] < 3 || c.n_cpu[b] < 3) {
        return cur;
    }
    // f grows with the rows while the CPU path costs more per row than the stream
    auto f = [&](int32_t r) {
        return llama_fn_l22_est_cpu(c, b, (uint32_t) r) - llama_fn_l22_est_pass(c, b, (uint32_t) r) - c.fix_ms;
    };
    int32_t t;
    if (f(lo) >= 0.0) {
        t = lo;
    } else if (f(hi) < 0.0) {
        t = hi;
    } else {
        int32_t a = lo; // f(a) < 0 <= f(z)
        int32_t z = hi;
        while (z - a > 1) {
            const int32_t m = a + (z - a)/2;
            (f(m) >= 0.0 ? z : a) = m;
        }
        t = z;
    }
    if (cur > 0 && (int64_t) std::abs(t - cur)*10 <= (int64_t) cur) {
        return cur; // hysteresis
    }
    return t;
}

uint32_t llama_fn_l22_ubplan(const llama_fn_l22_cost & c, int b, uint32_t base_ub, uint32_t n_ubatch, uint32_t n_tokens,
        int64_t after, int32_t xover, bool big_ok, double * est_base, double * est_big) {
    if (est_base) {
        *est_base = 0.0;
    }
    if (est_big) {
        *est_big = 0.0;
    }
    if (base_ub == 0 || base_ub >= n_ubatch) {
        return n_ubatch;
    }
    if (after < 0) {
        return llama_fn_l21_full_width(base_ub, n_ubatch, n_tokens, after);
    }
    const int64_t r_all = (int64_t) n_tokens + after;
    if (r_all <= (int64_t) base_ub || !big_ok) {
        return base_ub;
    }
    auto est = [&](uint32_t w) {
        double ms       = 0.0;
        bool   streamed = false;
        for (int64_t left = r_all; left > 0; ) {
            const uint32_t rows = (uint32_t) std::min<int64_t>(left, w);
            if (xover > 0 && rows < (uint32_t) xover) {
                ms += llama_fn_l22_est_cpu(c, b, rows);
            } else {
                ms += llama_fn_l22_est_pass(c, b, rows);
                streamed = true;
            }
            left -= rows;
        }
        return ms + (streamed ? c.fix_ms : 0.0) + (w > base_ub ? c.big_ms : 0.0);
    };
    const double e_base = est(base_ub);
    const double e_big  = est(n_ubatch);
    if (est_base) {
        *est_base = e_base;
    }
    if (est_big) {
        *est_big = e_big;
    }
    // today's rule (L21): the big ubatch only for a prompt wider than it; the other width only when clearly cheaper
    if (r_all <= (int64_t) n_ubatch) {
        return e_big < 0.97*e_base ? n_ubatch : base_ub;
    }
    return e_base < 0.97*e_big ? base_ub : n_ubatch;
}

bool llama_fn_l22_bigub2_ok(uint32_t n_seq_max, size_t full_big_1seq, size_t full_base_nseq) {
    return n_seq_max >= 2 && full_big_1seq > 0 && full_big_1seq >= full_base_nseq;
}

uint32_t llama_fn_l22_prompt_seqs(const uint32_t * rows_per_seq, int n_seq, uint32_t small_t) {
    uint32_t n = 0;
    for (int s = 0; rows_per_seq && s < n_seq; ++s) {
        n += rows_per_seq[s] > small_t ? 1 : 0;
    }
    return n;
}

bool llama_fn_l22_smallkeep_fits(size_t free_now, size_t expect_full, size_t pool, size_t margin) {
    if (expect_full == 0) {
        return false;
    }
    size_t need = expect_full;
    for (size_t add : { pool, margin }) {
        if (need > SIZE_MAX - add) {
            return false;
        }
        need += add;
    }
    return free_now >= need;
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
    // [TAG_FN_L3_HOST_QOS] [TAG_FN_L3_HOST_POKE] on every way out: after the profile state below exists, so a value the
    // profile sets counts as well as one from the environment
    struct l3_host_hook {
        llama_model & m;
        ~l3_host_hook() {
            try {
                llama_fn_l3_host_on_load(m);
            } catch (...) {
                // never out of a destructor (it may run while an exception unwinds the load)
            }
        }
    } l3_hook { model };

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
    llama_fn_l22_passlog_model(&model, true); // [TAG_FN_L22_PASSLOG] the profile's level
}

void llama_fn_auto_on_free(llama_model & model) {
    llama_fn_l3_host_on_free(model); // [TAG_FN_L3_HOST_QOS] [TAG_FN_L3_HOST_POKE] no-op for a model on_load skipped
    if (model.fn_auto) {
        if (model.fn_auto->active) {
            llama_fn_l22_passlog_model(&model, false); // [TAG_FN_L22_PASSLOG]
        }
        llama_fn_state_undo(*model.fn_auto);
        model.fn_auto->active = false;
    }
    std::lock_guard<std::mutex> lk(g_ctx_mtx);
    g_ctxs.erase(&model);
}

const char * llama_fn_env(const llama_model & model, const char * name) {
    return llama_fn_state_env(model.fn_auto.get(), name);
}

// [TAG_FN_L3_HOST] see llama-fn-auto.h
int llama_fn_l3_int(const llama_model & model, const char * name, int def) {
    if (model.arch != LLM_ARCH_QWEN4EXP) {
        return def;
    }
    const char * v = llama_fn_env(model, name);
    return v && v[0] ? atoi(v) : def;
}

bool llama_fn_l3_flag(const llama_model & model, const char * name) {
    return llama_fn_l3_int(model, name, 0) != 0;
}

// [TAG_FN_L21] see llama-fn-auto.h
int llama_fn_l21_int(const llama_model & model, const char * name, int def) {
    const char * v = model.arch == LLM_ARCH_QWEN4EXP ? llama_fn_env(model, name) : getenv(name);
    return v && v[0] ? atoi(v) : def;
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
