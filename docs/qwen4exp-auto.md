# Flash-Next (qwen4exp): automatic defaults

One build serves every model. The Flash-Next speed work switches itself on only for a qwen4exp model whose routed experts
are placed on the host (`--n-cpu-moe N`, `--cpu-moe`, `-ot ...exps=CPU`, or the `-fit` result). Every other model, and
qwen4exp with all experts on the GPU, runs exactly as before. Code: `src/llama-fn-auto.{h,cpp}` `[TAG_FN_AUTO]`.

## The switch

`LLAMA_FLASHNEXT_PROFILE=off|safe|fast|fast-dma` (unset = `safe`).

| Profile | What it turns on |
|---|---|
| `off` | nothing automatic; every switch works as a separate environment variable, as before |
| `safe` (default) | the measured winners only: all trunk experts on the host plus the adaptive VRAM hot set (speed/r1 `hot48ad`: +35 % greedy code, +45 % temp-1 prose at 32K), the VRAM fit and the RAM fit |
| `fast` | `safe` plus the built but not yet GPU-measured levers: `LLAMA_MOE_BRIDGE=1`, `GGML_CPU_APPLY_ONCE=1`, `GGML_CPU_Q5_1_AVX512=1`, `GGML_CPU_MMID_MR=1`, `GGML_CPU_MOE_FUSE=1`, `GGML_SCHED_SPLIT_ASYNC=1`, `LLAMA_PLE_HOST_GATHER=1`, `LLAMA_PLE_DIRECT_IO=1`, `LLAMA_GRAPH_PER_WIDTH=1`, `TURBO_QSA_CHUNK=64`, `SPEC_MTP_COST=1`, `LLAMA_MTP_ATTN_WINDOW=32768` |
| `fast-dma` | `fast` with `LLAMA_MOE_DMA_SHARE=auto` instead of the bridge (the two are exclusive) |

- A variable that is set in the environment always wins over the profile (also `0` to turn one lever off).
- `LLAMA_FLASHNEXT_FAST=NAME=VALUE,...` replaces the fast levers of `fast` (for tests; the safe items stay).
- The safe items are read only by the code that owns them. The fast levers are read elsewhere with `getenv`, so while the
  model is loaded the profile puts them into the process environment (the `GGML_CPU_*` ones through
  `ggml_cpu_fn_set_switch`) and removes them when the model is freed. Switches that latch on first use keep their value.
- The load log shows the decision: lines starting with `fn-auto:`.

## Safe items

| Variable | Profile value | Meaning |
|---|---|---|
| `LLAMA_FN_PLACEMENT` | `auto` | the routed experts of every trunk layer stay on the host; `user` keeps the overrides as given |
| `LLAMA_MOE_HOT_PROFILE` | `<model>.moeprof` if that file exists, else `even` | `even`: no routing profile, the same number of slots in every layer, empty at start, filled by the adaptive set; `off` = no hot set |
| `LLAMA_MOE_HOT_MIB` | `auto` | budget from free VRAM |
| `LLAMA_MOE_HOT_FIT` | `1` | the VRAM fit below |
| `LLAMA_MOE_HOT_ADAPT` | `1` | adaptive admission |
| `LLAMA_MOE_HOT_ADMIT` | `2/32` | admit after 2 sightings in 32 decode steps |
| `LLAMA_RAM_FIT` | `1` | the RAM fit below |

## VRAM fit `[TAG_FN_VRAM_FIT]`

The trunk context sizes the hot set at its first decode after every context of the model exists (the MTP draft
context is created after it; without MTP at the first decode, at the latest at the second). By then the model, the KV
caches (turbot pool, indexer cache), the compute buffers (the QSA temporaries grow with `-ub`) and the draft context are
allocated. Budget = min(free - margin, ceiling - used - margin):

- ceiling: `LLAMA_MOE_HOT_CAP_MIB`, default total - max(1536 MiB, total / 8) = 28,531 MiB on a 32 GB card;
- margin: `LLAMA_MOE_HOT_HEADROOM_MIB`, default 768 MiB (CUDA graphs, pool growth, lazily loaded kernels).

The log prints the breakdown (`moe-hot: VRAM fit ...`). Works at any `-c`, including 262144. `LLAMA_MOE_HOT_FIT=1`
enables the same for any model that uses `LLAMA_MOE_HOT_MIB=auto`.

## RAM fit `[TAG_FN_RAM_FIT]`

llama-server, after loading: when the host-resident model bytes (mmap), the host buffers, the prompt cache
(`--cache-ram`) and the context checkpoints need more than 85 % of physical RAM, the checkpoint budget per slot is
lowered (not below 512 MiB) and then the prompt cache (0 below 256 MiB), so the mapped expert pages stay in the page
cache. `--cache-ram`, `--ctx-checkpoints` and `LLAMA_CTX_CHECKPOINT_BUDGET_MIB` given by the user are kept.
`LLAMA_RAM_FIT=1` enables it for any model, `=0` disables it; `LLAMA_RAM_FIT_PCT` (85) and `LLAMA_RAM_FIT_CKPT_MIB`
(512) tune it.
