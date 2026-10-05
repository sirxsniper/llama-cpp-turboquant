# Flash-Next (qwen4exp): automatic defaults

One build serves every model. The Flash-Next speed work switches itself on only for a qwen4exp model whose routed experts
are placed on the host (`--n-cpu-moe N`, `--cpu-moe`, `-ot ...exps=CPU`, or the `-fit` result). Every other model, and
qwen4exp with all experts on the GPU, runs exactly as before. Code: `src/llama-fn-auto.{h,cpp}` `[TAG_FN_AUTO]`.

## Command line

No environment variable is needed: the default `safe` profile turns on every measured lever. The measured setup (RTX 5090,
Ryzen 9950X3D, 96 GB RAM, file A = Qwen3.8-Flash-Next UD-Q4_K_XL with the Q8_0 MTP head):

    llama-server -m Qwen3.8-Flash-Next-UD-Q4_K_XL-MTP-00001-of-00005.gguf -c 262144 -ngl 99 -fit off -fa on \
        -ctk turbot -ctv turbot --n-cpu-moe 48 -b 8192 -ub 8192 -t 16 \
        --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.5 --spec-draft-sampling probabilistic

- `-b 8192 -ub 8192` is the prefill setting: the prefill stream moves every host expert through VRAM once per ubatch.
- The model is larger than RAM: keep mmap (no `--no-mmap`, no `--load-mode none`).
- The measured runs also pinned one thread per core (`--cpu-mask 55555555 --cpu-strict 1 --prio 2`) and used
  `--cache-ram 0`.
- Lever round 1 (`[TAG_FN_SHIP1]`, -c 262144, 1 stream, E:/turbot-gates/flashnext/test/history.md):
  - benchmark: 64.8 t/s greedy code, 58.7 t/s temp-1 prose, ~4K prompt;
  - real use (temp 1, thinking on, 1536-token answers, context filled to 32K / 131K / 246K): 61.4 / 59.3 / 55.1 t/s;
  - prefill: 1650 t/s for a 32K prompt, 2104 t/s for a 131K prompt;
  - VRAM peak 28,320 MiB.
- Lever round 2 (`[TAG_FN_SHIP2]`, the same setup, 2048-token answers on prompts without literal end-of-turn text):
  - benchmark: 64.2 t/s greedy code, 58.9 t/s temp-1 prose (+6.5 % / +5.8 % over the round-1 default, same session);
  - real use: 65.4 / 58.6 / 60.7 t/s at 32K / 131K / 246K (mean 61.6 t/s, +1.8 %);
  - prefill unchanged (fill 1571 t/s to 32K, 2127 t/s to 131K); VRAM peak 27,999 MiB.

## The switch

`LLAMA_FLASHNEXT_PROFILE=off|safe|fast|trial|trial-dma` (unset = `safe`). `off` is the kill switch.

| Profile | What it turns on |
|---|---|
| `off` | nothing automatic; every switch works as a separate environment variable, as before |
| `safe` (default) | the measured winners: the safe items and levers below |
| `fast` | `safe` plus levers that won only a narrower A/B (`P_FAST` in `k_items`). None: every measured lever is in `safe`, so `fast` is `safe` |
| `trial` | `safe` plus every built lever that is not measured yet, for A/B runs only. None today, so `trial` is `safe` |
| `trial-dma` | `trial` without the host bridge (`LLAMA_MOE_BRIDGE`, `LLAMA_MOE_BRIDGE_DMA`): the gen5 DMA share in its own CPU split |

- A variable that is set in the environment always wins over the profile (also `0` to turn one lever off).
- `LLAMA_FLASHNEXT_FAST=NAME=VALUE,...` adds or changes options of `fast`, `trial` and `trial-dma` (one lever per A/B
  arm); `safe` ignores it. It would replace levers that are not in `safe`; there are none today.
- The lookup items (`LLAMA_MOE_HOT_*`, `LLAMA_FN_*`, `LLAMA_RAM_FIT`, `LLAMA_PLE_DIO_FILE`) are read only by the code that
  owns them. The other levers are read elsewhere with `getenv`, so while the model is loaded the profile puts them into
  the process environment (the `GGML_CPU_*` ones through `ggml_cpu_fn_set_switch`) and removes them when the model is
  freed. Switches that latch on first use keep their value for the process. `LLAMA_NO_ECOQOS` is applied at load
  (`llama_backend_init` read it before any model existed).
- The load log shows the decision: lines starting with `fn-auto:`.

## Safe items (lookups)

| Variable | Profile value | Meaning |
|---|---|---|
| `LLAMA_FN_PLACEMENT` | `auto` | the routed experts of every trunk layer stay on the host; `user` keeps the overrides as given |
| `LLAMA_MOE_HOT_PROFILE` | `<model>.moeprof` if that file exists, else `even` | `even`: no routing profile, the same number of slots in every layer (the budget also holds each layer's zero slot), empty at start, filled by the adaptive set; `off` = no hot set. The DMA share's warm start reads the same profile |
| `LLAMA_MOE_HOT_MIB` | `auto` | budget from free VRAM |
| `LLAMA_MOE_HOT_FIT` | `1` | the VRAM fit below |
| `LLAMA_MOE_HOT_ADAPT` | `1` | adaptive admission |
| `LLAMA_MOE_HOT_ADMIT` | `2/32` | admit after 2 sightings in 32 decode steps |
| `LLAMA_MOE_HOT_HEADROOM_MIB` | `1280` | VRAM fit margin (the 768 default went over 28,500 MiB at depth) |
| `LLAMA_MOE_HOT_DECAY` / `LLAMA_MOE_HOT_SEED` | `0.92` / `0.03` | the decayed hot set |
| `LLAMA_PLE_DIO_FILE` | `<model>.ple` | the unmapped copy of the PLE table that `LLAMA_PLE_DIRECT_IO` reads. It is made once, on the first load (28.8 GB for file A), when its volume keeps 16 GiB free beyond it; otherwise the table is read through the mapping as without the lever. Set the variable to put the copy elsewhere (a second drive) |
| `LLAMA_RAM_FIT` | `1` | the RAM fit below |

## Safe levers (put into the environment while the model is loaded)

| Group | Variables |
|---|---|
| host bridge and DMA share | `LLAMA_MOE_BRIDGE=1` (with its rollback ring), `LLAMA_MOE_BRIDGE_DMA=1`, `LLAMA_MOE_DMA_SHARE=auto`, `LLAMA_MOE_BRIDGE_PF=1` and `LLAMA_MOE_BRIDGE_PF_SOLO=1` (the next-layer prefetch, see below) |
| CPU expert kernels | `GGML_CPU_APPLY_ONCE=1`, `GGML_CPU_Q5_1_AVX512=1`, `GGML_CPU_MMID_MR=1`, `GGML_CPU_MOE_FUSE=1`, `GGML_CPU_VNNI=1` |
| host trims | `GGML_SCHED_SPLIT_ASYNC=1`, `LLAMA_GRAPH_PER_WIDTH=1`, `GGML_CUDA_GRAPH_POKE=1`, `LLAMA_NO_ECOQOS=1` |
| PLE | `LLAMA_PLE_HOST_GATHER=1`, `LLAMA_PLE_DIRECT_IO=1` |
| QSA attention | `LLAMA_QSA_POS_MASK=1`, `LLAMA_QSA_POS_CHUNK=512`, `TURBO_QSA_CHUNK=512`, `TURBO_QSA_SPARSE=1`, `TURBO_QSA_TOPK_UNORDERED=1` |
| MTP | `SPEC_MTP_COST=1`, `LLAMA_MTP_ATTN_WINDOW=32768`, `LLAMA_MTP_HEAD_ROWS=98304`, `SPEC_DFT_UBATCH=128` |
| prefill stream | `LLAMA_PREFILL_STREAM=1`, `LLAMA_PREFILL_STREAM_LEND=1`, `LLAMA_PREFILL_STREAM_THREADS=16` |

Evidence (E:/turbot-gates/flashnext/test/history.md, lever round 1): all of them together against the hot set alone, same
binary, 2 interleaved rounds at -c 262144: real use +38.6 %, benchmark code +24.8 %, decode KLD in the placement band
(0.0117 vs 0.0121). The binary search over the groups removed none. With the fix-r1 code (k-pool tail, wide sparse-index
compaction, bridge rollback ring): real use 35.3 -> 58.6 t/s (+65.9 %) against the round's start.

Lever round 2 (`[TAG_FN_R2_BRIDGE_PF]`, r2/ab4, 2 interleaved rounds against the round-1 default on the same binary):
after each layer's host job the CPU MoE pool predicts the next layer's experts from its router and loads the cold ones
into the CPU caches while the device runs that layer's attention (real loads; with `_SOLO` the executor only
dispatches, so every worker computes the pieces it loaded). Values are never touched: greedy identity on file A,
test-backend-ops bit for bit. Benchmark code +6.5 %, prose +5.8 %, real use +1.8 %, VRAM unchanged.
`LLAMA_MOE_BRIDGE_PF=0` turns it off.

## Bridge stalls `[TAG_FN_R2_BRIDGE_NOSPEC]` `[TAG_FN_R2_BRIDGE_SYNC]`

Code defaults of every context with the host bridge (not profile items):
- A bridged ubatch whose host side misses the device's deadline (page-ins after a long prefill or after another model
  ran) is rolled back and computed again. With MTP the speculative ring gets one more token (`[TAG_FN_R1_BRIDGE_RB]`);
  without speculative decoding the context gets a recurrent ring of `LLAMA_MOE_BRIDGE_RB_TOKENS` tokens (default 1,
  at most 8) and the bridge takes only graphs that ring can roll back whole. `LLAMA_MOE_BRIDGE_RB=0` turns both off.
- While the bridge is paused or off after a miss, its layers run the host job as a CPU graph op
  (`GGML_OP_MOE_HOST_SUM`) and sum like the bridged graph, so a stall never changes the output (forced-stall tests with
  and without MTP: identical tokens). `LLAMA_MOE_BRIDGE_SYNC=0` restores the plain CPU split.

## VRAM fit `[TAG_FN_VRAM_FIT]`

The trunk context sizes the hot set at its first decode after every context of the model exists (the MTP draft
context is created after it; without MTP at the first decode, at the latest at the second). By then the model, the KV
caches (turbot pool, indexer cache), the compute buffers (the QSA temporaries grow with `-ub`) and the draft context are
allocated. Budget = min(free - margin, ceiling - used - margin):

- ceiling: `LLAMA_MOE_HOT_CAP_MIB`, default total - max(1536 MiB, total / 8), rounded down to 256 MiB: 28,416 MiB on a
  32 GB card (under the 28,500 MiB the 5090 is run at);
- margin: `LLAMA_MOE_HOT_HEADROOM_MIB`, 1280 MiB in the qwen4exp profile, 768 MiB otherwise (CUDA graphs, pool growth,
  lazily loaded kernels).

The log prints the breakdown (`moe-hot: VRAM fit ...`). Works at any `-c`, including 262144. `LLAMA_MOE_HOT_FIT=1`
enables the same for any model that uses `LLAMA_MOE_HOT_MIB=auto`.

## Compute-buffer lend `[TAG_FN_L3_VRAM_CBUF]` (lever round 3, off by default)

`LLAMA_FN_CBUF=1` (qwen4exp only, the hot set's owner context, needs CUDA virtual memory management): the trunk's
compute buffer is sized for a `-ub 8192` prompt (3.7 GB at 262K) and idle in every decode step. With the switch the
scheduler holds a SMALL reserve between prompts (graphs of at most 31 tokens: decode, MTP verify, short prompts) and the
hot set keeps the difference as extra slots:

- At the VRAM fit the trunk reserves SMALL first, so the fit sees FULL - SMALL as free VRAM (about +3.4 GB, 71 -> about
  95 even slots per layer at 262K). The hot set's device buffer is then allocated on CUDA virtual memory; its top layers,
  whole layers from a 2 MiB boundary on and at least FULL - SMALL bytes plus a slack of max(128 MiB, FULL / 16) (the FULL reserve can grow once the hot set exists; a larger tail costs the decode nothing), form a tail that is mapped apart from the rest. FULL is reserved once more at the fit, so it includes the draft context's outputs.
- A batch wider than 31 tokens: the tail layers leave both tables and the hot chain, uploads into them stop, the tail's
  VRAM goes back to the driver, then the FULL reserve is allocated (the prefill stream borrows its banks from the part
  below the tail, as before).
- The next narrow batch: the stream's banks come back, the FULL compute buffer is freed, the trunk backend's pools give
  back what the prompt grew (`[TAG_FN_L3_VRAM_TRIM]`), the tail is mapped again at the same addresses and cleared, its
  resident experts come back (through the upload worker by default, see `LLAMA_FN_CBUF_ASYNC`), and the SMALL reserve is
  allocated. The device use never passes the larger of the two states'; the log prints it at each switch
  (`cbuf_set: [TAG_FN_L3_VRAM_CBUF] -> FULL / SMALL`).
- Layers past the trunk (an MTP block, when its experts are on the host) are placed below the tail, so the tail never
  holds a layer that the draft context reads while a prompt runs.
- Failures stay local: a FULL reserve that fails returns to SMALL and fails that batch only (so does a reserve left
  pending by a failed switch: decode returns -2 instead of throwing); a tail that cannot be
  mapped again leaves its layers on the host and is tried again every 64 narrow batches; a hot set with a tail that
  cannot be allocated falls back to the FULL reserve and a hot set without one. The bridge reads the hot tables through
  `llama_moe_cache_lookup_table`, which also answers while the tail is out.
- Exactness: with `-b` equal to `-ub` every batch is one ubatch, so a prompt never runs a narrow ubatch with the tail
  out, and a decode always sees the whole hot set: the outputs equal the lend-off run with the same hot set (static
  profile, MTP off). With `-b` larger than `-ub`, a prompt's trailing ubatch of at most 8 tokens computes the tail
  layers' experts on the host (placement noise, the early9 KLD band).

| Variable | Default | Meaning |
|---|---|---|
| `LLAMA_FN_CBUF` | 0 | the compute-buffer lend |
| `LLAMA_FN_CBUF_SMALL_T` | 31 | widest graph of the SMALL reserve (clamped to 1 .. `GGML_OP_OFFLOAD_MIN_BATCH` - 1, a multiple of the sequences); below 8 the 4..8-token graphs run with FULL and without the tail's hot chain (3 = the MTP verify of n_max 2: the smallest SMALL reserve at 262K, whose attention temporaries grow with the width) |
| `LLAMA_FN_CBUF_POOL_MIB` | 0 | N MiB more for the hot set and N MiB more in its tail: between prompts the experts use the room that a prompt's pool growth takes while the tail is out (the pools are trimmed at each return to SMALL). The FULL-state device use does not change; the SMALL state's grows by N - keep N at or below the measured pool growth of a FULL period (the `-> SMALL` line prints it) or the room under the 28,500 MiB peak |
| `LLAMA_FN_CBUF_TRIM` | 1 | trim the trunk backend's pools at each return to SMALL |
| `LLAMA_FN_CBUF_EAGER` | 1 | return to SMALL at the end of a FULL batch that did not fill `-b` (the last batch of a prompt with one stream), so the switch belongs to the prompt and the tail's refill starts before the first decode step; 0 = at the first narrow batch. Batches of exactly `-b` tokens stay FULL (KLD runs, multi-batch prompts) |
| `LLAMA_FN_CBUF_ASYNC` | 1 | with the adaptive hot set the tail's resident experts come back through its upload worker and are published as they land (the host computes them until then), so the first decode step after a prompt does not wait for the refill; 0 = a synchronous refill before that step (also the path of a static hot set) |
| `LLAMA_FN_CBUF_THRASH` | 8 | that many FULL switches within 64 batches (steps wider than the SMALL reserve, e.g. several streams drafting 8 tokens each) pin the FULL state: the tail stays out until 256 narrow batches in a row; 0 = never |
| `LLAMA_FN_CBUF_DRAFT_TRIM` | 1 | `[TAG_FN_L3_VRAM_TRIM]` with `LLAMA_FN_CBUF=1`, the MTP draft context gives its pools' prompt growth back at its first batch of <= 64 tokens after a wider one (its own backends, after its own synchronize). It gains hot-set room only with `LLAMA_FN_CBUF_EAGER=0` (the trunk then returns to SMALL after the draft's prompt pass) and `LLAMA_FN_CBUF_POOL_MIB` set to the measured growth (the `draft prompt` / `draft decode` account lines and the trim line) |
| `LLAMA_FN_CBUF_REFILL_THREADS` | 8 | copy threads of the synchronous refill (mmap -> pinned halves -> device) |
| `LLAMA_FN_HOT_BUDGET_MAX_MIB` | unset | upper bound on the VRAM fit's hot-set budget (any lend setting; used to give two configurations the same hot set) |
| `LLAMA_FN_VRAM_ACCOUNT` | 0 | `[TAG_FN_L3_VRAM_ACCOUNT]` device use at each stage (context, KV, bridge, gen5, reserve, fit, hot set, switches, prompt/decode changes), the pools, and the model's device tensors by kind (qwen4exp only, logging only) |

ggml-cuda (any model, used only by the code above): `ggml_backend_cuda_vmm_buffer_alloc / _map / _unmap / _mapped`,
`ggml_backend_cuda_vmm_granularity`, `ggml_backend_cuda_pool_stats / _trim` (also through get_proc_address as
`ggml_backend_vmm_*` / `ggml_backend_pool_*`); `tests/test-vmm-lend.cpp` checks them on the GPU.

## RAM fit `[TAG_FN_RAM_FIT]`

llama-server, after loading: when the host-resident model bytes (mmap), the host buffers, the prompt cache
(`--cache-ram`) and the context checkpoints need more than 85 % of physical RAM, the checkpoint budget per slot is
lowered (not below 512 MiB) and then the prompt cache (0 below 256 MiB), so the mapped expert pages stay in the page
cache. `--cache-ram`, `--ctx-checkpoints` and `LLAMA_CTX_CHECKPOINT_BUDGET_MIB` given by the user are kept.
`LLAMA_RAM_FIT=1` enables it for any model, `=0` disables it; `LLAMA_RAM_FIT_PCT` (85) and `LLAMA_RAM_FIT_CKPT_MIB`
(512) tune it.

## MTP draft switches of lever round 3 (`[TAG_FN_L3_MTP_*]`, off by default, not in any profile)

For A/B runs on qwen4exp only; every other model ignores them (a warning says so).

| Variable | Values | What it does |
|---|---|---|
| `SPEC_MTP_COST` | `2` | the draft length v2 (`[TAG_FN_L3_MTP_COST2]`, common/speculative-mtp-cost2.h): a verify row is priced by the cold experts the MoE bridge counts per step (k us per cold expert-layer, measured from the step-to-step variation at one width) plus a per-width row cost; a token is kept when its expected tokens pay for its row at the realized rate, and a further draft decode only when the expected next token pays for the decode. `1` is version 1 unchanged (the profile's value), `0` the fixed n_max / p_min rule. `SPEC_MTP_COST2_LOG=1` prints a summary every 256 policy steps and at the end |
| `SPEC_MTP_PMIN_POS` | `p1,p2,...` | the p_min rule per draft position, the last value for the rest (`[TAG_FN_L3_MTP_CHAIN]`), e.g. `0.5,0.5,0.85` with `--spec-draft-n-max 3`: a third draft only when the drafter is confident. Applies wherever the rule decides (no cost policy, and the warm-up and probe steps of the policies) |
| `LLAMA_MTP_HEAD_PROMPT` | `<n>` | with `LLAMA_MTP_HEAD_IDS` (or `_ROWS`): the drafts also score n more rows (`[TAG_FN_L3_MTP_HEADPROMPT]`, full head rows): each request's prompt tokens that the draft vocabulary leaves out, most frequent first, then the next ids outside the vocabulary in id order (all distinct). 1024 costs ~10 MiB of the draft context's compute buffer; verify is unchanged |
| `SPEC_MTP_BLOCK_VERIFY` | `1` | sampled drafts (`--spec-draft-sampling probabilistic`, temp > 0) verified as a block (`[TAG_FN_L3_MTP_BLOCK]`, common/speculative-block.h, Sun et al. 2024 Algorithm 2): the target's distribution as with the per-token min(1, p/q) rule, at least as many tokens kept (exact enumeration: +0.7-1.5 % tokens per step with 2 drafts, +2-4 % with 3). Requests with a grammar, a reasoning budget, token probabilities or backend-sampled rows keep the per-token rule. Costs one more target sampling pass for the rows after an early rejection |
| `SPEC_MTP_COST2_TRACE` | `<file>` | one CSV line per kept step of version 2 (rows, policy, tokens, rest-of-step us, step us, k, rate, dV, the cold prefix counts); `tools/qwen4exp`-style check: `E:/turbot-gates/flashnext/test/l3/mtp/tools/l3_trace_check.py` |

- The third draft needs `--spec-draft-n-max 3`; the recurrent ring and the bridge follow it (n_rs_seq = n_max, +1 with
  the bridge rollback ring), so 4-row verify graphs stay bridged.
- MTP head experts at the trunk's own mix (recipe C, `[TAG_FN_R4_MTP_MIX]`): the file set
  `Qwen3.8-Flash-Next-UD-Q4_K_XL-MTPmix-0000N-of-00005.gguf` (shards 2-4 hardlinked to the trunk, shard 5 = the head with
  its 512 experts at q4_K / q4_K / q5_1, 1.03 GiB less VRAM, more of it for the hot set). Only drafts change. Its PLE copy
  `<file>.ple` is a hardlink of file A's, so the first load copies nothing.
- `LLAMA_MTP_ATTN_WINDOW` (32768 in the profile) and `LLAMA_MTP_HEAD_ROWS` (98304) stay as they are; `=0` turns either off
  for an A/B arm.
- MTP head experts at q2_0 (`[TAG_FN_L3_MTP_Q2]`, the Strata / flashrt draft layer): `recipe_mtp_mix.ps1 -Mix
  q2_0,q2_0,q2_0 -Name MTPq2` writes `Qwen3.8-Flash-Next-UD-Q4_K_XL-MTPq2-0000N-of-00005.gguf` (1.83 GiB less VRAM than file
  A's q8_0 experts). ggml's own q2_0 quantizer (absmax scale, never code 3) has a relative RMS weight error of 0.73 on
  these experts and an expert output error of 1.26, so `mtp_requant.py` writes q2_0 by a scale search (0.36 / 0.59; recipe
  C: 0.075 / 0.11). `-Mix q3_K,q3_K,q4_0 -Name MTPq3` is the middle point (1.38 GiB, 0.16 / 0.24). The down projection's
  640-wide rows rule out k-quants for it. Only drafts change: the A/B is the acceptance and the speed.
- A ranked draft vocabulary for `LLAMA_MTP_HEAD_IDS` (`[TAG_FN_L3_MTP_HEADIDS]`, `tools/qwen4exp/mtp_head_rank.py`): ids
  by frequency in real answers, then in tokenized prompt text, then by id. On 160K held-out answer tokens 40,960 ids cover
  99.68 % and 32,768 ids 99.59 %, against 99.80 % for `LLAMA_MTP_HEAD_ROWS=98304`: 58 % fewer head rows per draft step for
  ~0.1 point of coverage, but the compact head is a ~109 MiB copy in VRAM, and the answers were English text about code
  (other languages fall back to the id order).
