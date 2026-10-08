<div align="center">

# TurboQuant

### Full 262K-context inference on a single RTX 5090 — dense and MoE, at speed

**by [@sirxsniper](https://github.com/sirxsniper)**

[![Release](https://img.shields.io/github/v/release/sirxsniper/llama-cpp-turboquant?style=for-the-badge&color=1f6feb&label=release)](https://github.com/sirxsniper/llama-cpp-turboquant/releases/latest)
[![CUDA](https://img.shields.io/badge/CUDA-13.1-76b900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-downloads)
[![GPU](https://img.shields.io/badge/RTX%2050--series-sm__120-8957e5?style=for-the-badge)](#build-from-source)
[![Context](https://img.shields.io/badge/context-262K-e3b341?style=for-the-badge)](#performance)
[![License](https://img.shields.io/badge/license-MIT-6e7681?style=for-the-badge)](LICENSE)

<br>

<table>
<tr>
<td align="center" width="50%">

**Qwen3.8-27B** · UD-Q5_K_XL

## ~150 t/s
code, 1 stream · 262K context · 4 slots

**250–290 t/s** total at 4 streams

</td>
<td align="center" width="50%">

**Qwen3.8-Flash-Next** · UD-Q4_K_XL (100+ GB MoE)

## 101 t/s
real use at 32K · **89 t/s at 131K and 245K** · 103 / 93 / 91 with the MTPq3 files · a short new turn starts in 0.3–1.4 s instead of 3.4 s · **2 conversations at 32K: 62–66 t/s each**

prompt reading **2,300 t/s** at 131K with the MTPq3 files (v0.20.0, +12 %) · **the first 32K prompt after a start 2,020 t/s** (v0.20.2, was 944)

</td>
</tr>
</table>

<sub>RTX 5090 32 GB · Ryzen 9 9950X3D · 96 GB DDR5-6400 · PCIe Gen5 · Windows 11 · turbot KV cache · measured 2026-10-06/08</sub>

</div>

---

## Contents

- [What is inside](#what-is-inside)
- [Performance](#performance)
- [Quick start](#quick-start)
- [Recommended configurations](#recommended-configurations) — [Qwen3.8-27B](#qwen38-27b) · [Qwen3.8-Flash-Next](#qwen38-flash-next)
- [How it works](#how-it-works)
- [Build from source](#build-from-source)
- [Releases](#releases)
- [Credits](#credits)

---

## What is inside

<table>
<tr>
<td width="50%" valign="top">

### turbot — a tiered KV cache

The newest tokens stay exact, the middle of the context is kept at 6–8 bits, the oldest at a per-head budget. A full 262K context fits next to the weights on one 32 GB card at **q8-level quality** (KLD vs f16 on Qwen3.8-27B: 0.0011 code / 0.0019 prose).

The plan is built into the binary and chosen per model automatically: turbot → turbo5p → turbo5p512 → turbo4 → q8_0, with one log line saying why. The server never refuses to start because of a KV type.

</td>
<td width="50%" valign="top">

### The Flash-Next engine

Qwen3.8-Flash-Next is larger than VRAM *and* RAM. Every decode step is split between GPU and CPU:

- an **adaptive per-expert GPU hot set** that learns what you use;
- a **MoE bridge**: the CPU computes the cold experts while the GPU computes the hot ones, in one CUDA graph per step;
- **prompt streaming over PCIe Gen5**, so long prompts read at 1.5–2.3K t/s;
- MTP drafting with the model's own head, and dozens of fused decode kernels.

It all switches on by itself for this architecture. Every other model is untouched.

</td>
</tr>
<tr>
<td width="50%" valign="top">

### Fast speculative decoding

**DFlash2** block drafting for the 27B (n_max 7, 4 slots), and **MTP** for models with their own head — with the drafter skipped over prompt stretches it can never draft from, and verification batched with the target.

</td>
<td width="50%" valign="top">

### Built to run all day

Five GPU memory bugs found with compute-sanitizer and fixed; no GPU work left in flight after a response; Ctrl+C / Ctrl+Break / closing the window drain the GPU before exit; slot files keep their context checkpoints (a restored 20K-token slot continues after 4 tokens instead of re-reading).

</td>
</tr>
</table>

---

## Performance

### Qwen3.8-27B — UD-Q5_K_XL, DFlash2-Q8_0 drafter

262K context, 4 slots, turbot KV, production flags below.

| Workload | Decode |
|:--|--:|
| Code, 1 stream | **147–152 t/s** |
| Prose, 1 stream | **106–110 t/s** |
| Short answers, 1 stream | **197 t/s** |
| 4 concurrent streams, total | **250–290 t/s** |

| KV cache quality (KLD vs f16, lower is better) | code | prose |
|:--|--:|--:|
| **turbot** | **0.00114** | **0.00186** |
| turbo5p | 0.00160 | 0.00258 |

### Qwen3.8-Flash-Next — UD-Q4_K_XL + Q8_0 MTP head

262K context, turbot KV, 1 stream. *Real use* = temperature 1, top-p 0.95, top-k 20, thinking on, 2,048-token answers on a context already filled to the given depth.

| | first 262K build (Oct 4) | v0.16.0 | **v0.17.0** | **v0.17.0 + MTPq3 files** |
|:--|--:|--:|--:|--:|
| Real use, 32K filled | 30.3 t/s | 102.3 t/s | **101.5 t/s** | **103.1 t/s** |
| Real use, 131K filled | 26.1 t/s | 85.7 t/s | **89.1 t/s** | **92.7 t/s** |
| Real use, 245K filled | 24.5 t/s | 85.1 t/s | **89.7 t/s** | **91.3 t/s** |
| Benchmark, greedy code | 39 t/s | 93 t/s | **93 t/s** | 94 t/s |
| Prompt reading, fresh 32K / 131K | ~150 t/s | 1,508 / 2,059 t/s | about the same | 1,862 / 2,068 t/s |

- Quality: KLD vs the original file **0.0109** (same top token 97.3 %); needle recall passes at 131K and 245K.

**New turn on a conversation** (v0.18.0): the time from sending a message to the first answer token, 24K conversation already in the cache.

| New tokens | 32 | 64 | 128 | 256 | 512 | 1,024 | 2,048 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| v0.17.0 | 3.39 s | 3.41 s | 3.40 s | 3.38 s | 3.41 s | 3.78 s | 3.88 s |
| **v0.18.0** | **0.33 s** | **0.54 s** | **0.84 s** | **1.39 s** | 3.50 s | 3.85 s | 3.90 s |

- Chunks under 384 tokens now compute their CPU-side experts on the CPU instead of streaming every expert through the GPU once (~3.4 s whatever the size); 384 tokens and more still stream (faster there). Decoding is not touched: output identical to v0.17.0, the real-use numbers above stand.
- v0.16.0 and v0.17.0 measured back to back in one session (2 interleaved rounds); the MTPq3 column against v0.17.0 with the standard files in another session (3 rounds: 100.2 / 88.4 / 88.3 -> 103.1 / 92.7 / 91.3). Real use at the first depth after a load varies by about 3 %.
- The model (105 GB) is larger than RAM. When Windows has dropped expert data from its file cache (seen after server restarts while other programs were busy; the cause is still being traced), real use is lower until the data is read back. v0.19.0 on the evening of 2026-10-07 (3 sessions of 2 rounds, other programs using 2–12 CPU cores): 86–99 / 79–86 / 82–89 t/s with the MTPq3 files, 84–89 / 78–82 / 82–86 t/s with the standard files, and a fresh 32K prompt read at 720–1,070 t/s instead of ~1,860.
- VRAM about 29.9 GB while generating (the expert hot set takes the free VRAM up to 2.1 GB below the card total, plus the part of the 262K KV cache a shorter context does not use yet; `LLAMA_FN_VRAM_KEEP_MIB=4096` keeps more free); the rest of the model stays memory-mapped.
- Vision works: add `--mmproj mmproj-Qwen3.8-Flash-Next-F16.gguf`.

**With vision** (`--mmproj mmproj-Qwen3.8-Flash-Next-F16.gguf`, v0.20.3, MTPq3, same session, 2 rounds): answers 96.1 / 87.7 / 85.5 t/s at 32K / 131K / 245K against 102.1 / 92.5 / 88.8 text only (about -5 %: the encoder takes 1.1 GiB of VRAM from the expert hot set); prompts unchanged (32K -2 %, 131K the same); needles at 131K and 245K pass with vision loaded.

**Two conversations at once** (`--parallel 2 --kv-unified`, both sharing the 262K pool; real-use sampling, a 32K conversation on each, 2,048-token answers; each conversation's own speed):

| Files | one conversation alone | **two at once, each** |
|:--|--:|--:|
| MTPq3 (v0.20.0: 3 runs, v0.20.1: 2 runs) | 94–102 t/s | **62–66 t/s** |
| standard (v0.20.0, 3 runs) | 87–96 t/s | 59–65 t/s |

- v0.18.0 gave about 38 t/s each, v0.19.0 56–61 t/s each (standard files, 1,024-token answers).
- Since v0.19.0 both conversations' verify rows run as one bridged graph: the MoE bridge takes up to n_rs_seq tokens of each sequence, and sequences that draft together draft the same length, the longer one: both keep drafting while either passes the p_min rule (`SPEC_MTP_EQUAL_SEQS`, on in the Flash-Next profile). This gives most of the gain (about 40 -> 56–61 t/s each); acceptance per conversation drops a little (0.76–0.79 -> 0.68–0.72).
- One conversation gives the same output as before (identity checks with and without MTP, every release since v0.18.0). Without MTP, the last layer of a long prompt chunk runs its CPU part synchronously, so a stall cannot fail the request.

**A new prompt while the other conversation answers** (v0.20.1, MTPq3 files; conversation 1 answers, conversation 2 sends a new prompt):

| | v0.20.0 | **v0.20.1** |
|:--|--:|--:|
| Conversation 1's answer while a 17K prompt is read (32K / 131K conversation) | 0.8–0.9 / 0.4 t/s | **26 / 19 t/s** |
| The same, a 32K prompt that reads the model from the disk | 0.5–0.7 / 0.7 t/s | **13 / 8 t/s** |
| Conversation 2's wait for its first token, 17K prompt (32K conversation) | 14–15 s | 21 s |
| The same, 32K prompt from the disk (32K / 131K conversation) | 25–28 / 25 s | 33 / 40 s |

- Before v0.20.1, an answering conversation got one step per 8K prompt chunk of the other (each chunk streams every expert, ~3.4 s), so it stopped for the whole prompt. Now, after each chunk, it decodes alone for half that chunk's time, at most 2 s (`LLAMA_SRV_GEN_SHARE=0.5` in the Flash-Next profile; `0` restores the old order). Once both answer, each runs at 47–60 t/s as before.
- Fixed in v0.20.1: with two conversations, the prompt stream could give the first layer of its next chunk a wrong expert. The hot set could swap that expert into a slot after the stream had picked the slot and before its GPU copy ran (v0.20.0 only, found by code review). The stream now reads that layer from RAM. The two-conversation identity check (GPU copy on vs off) passes, and a compute-sanitizer run with two conversations reports 0 errors.

**The first prompt after a start** (v0.20.2, MTPq3 files, same-session pairs, 2 rounds each):

| Prompt | v0.20.1 | **v0.20.2** |
|:--|--:|--:|
| 32K tokens, first prompt after a start | 944 t/s (744 / 1,144), 36-38 GB read from the disk | **2,020 t/s** (2,016 / 2,025), ~0.1 GB |
| The same after 38 GB of other files were read (start, 45 s idle, prompt) | 955-958 t/s | **1,554-1,570 t/s** |
| 131K tokens | 2,298 t/s | 2,316 t/s |

- The load used to prefetch the whole model file into the file cache. That left the routed experts (71.7 GiB) as the oldest cached pages, in the same order the first prompt reads the layers, so every page the prompt read pushed out a layer it had not reached yet: the first 8K chunk re-read 36-38 GB at 3-4 GB/s (17-37 s instead of ~3.5 s).
- Now the experts stay out of that prefetch, and a background pass reads them into the server's working set (about 20-30 s on the 9950X3D). It first raises the process's working-set maximum: the token-embedding lock had set it to ~0.74 GiB, which made Windows trim a filled working set back to 0.8 GiB. `LLAMA_FN_L15_WARM=0` restores the old load. Flash-Next on Windows only.
- v0.20.3: the pass starts once the server's warm-up run is done, so the server is ready after **6.6 s instead of 31-34 s** (after 38 GB of other files were read). A long prompt sent while the pass still runs waits behind it layer by layer; the standard benchmark's first 32K prompt finished 59 s after the start (v0.20.2: 62 s).
- Answers: same-session A/B, 5 rounds: v0.20.2 is equal or slightly faster (paired mean +0.7 % over 2 rounds, +2.9 % over 3 more) and pages in at most 0.6 GB per answer instead of 0.3-2.4 GB. Output token-identical to v0.20.1.

**Prompt reading** (v0.20.0, MTPq3 files, same-session pairs, 2 rounds):

| Prompt | v0.19.0 | **v0.20.0** |
|:--|--:|--:|
| 131K tokens (second prompt after a start) | 2,061 t/s | **2,317 t/s** |

- 131K: the prompt stream copies the experts the VRAM hot set already holds (about 8 GiB of each 72 GiB pass; never a slot the KV cache can take back) from their slots on the GPU instead of from RAM; a warm 8K pass takes about 3.0-3.2 s instead of 3.5 s.
- Decoding output is unchanged: token-identical to v0.19.0.

> Numbers are real-use medians over interleaved A/B rounds, not best-case benchmark runs. A greedy short-prompt benchmark always reads higher than a long, sampled answer — both are listed where it matters.

---

## Quick start

1. Download the latest Windows build from [**Releases**](https://github.com/sirxsniper/llama-cpp-turboquant/releases/latest): `llama-turboquant-*-win-x64-cuda13.1-sm120-vulkan.zip`.
2. Unzip `cudart-cuda13.1-win-x64.zip` from the same release into that folder (or install the CUDA 13.1 toolkit).
3. Start `llama-server.exe` with one of the configurations below and open `http://localhost:8080`.

Requirements: an RTX 50-series GPU (the release binary is built for `sm_120a`), an NVIDIA driver with CUDA 13.1 support, and a CPU with AVX-512 (Zen 4 / Zen 5) — or [build from source](#build-from-source) for anything else.

---

## Recommended configurations

### Qwen3.8-27B

UD-Q5_K_XL with the DFlash2-Q8_0 drafter and vision, 262K context, 4 slots.

```bat
set GGML_DISABLE_VULKAN=1
set SPEC_DFT_UBATCH=128

llama-server.exe --model Qwen3.8-27B-UD-Q5_K_XL.gguf --host 0.0.0.0 --port 8080 ^
  --n-gpu-layers 99 --device CUDA0 --ctx-size 262144 ^
  --cache-type-k turbot --cache-type-v turbot --flash-attn on --kv-unified ^
  --batch-size 2048 --ubatch-size 1024 --parallel 4 --threads 16 ^
  --load-mode none --cache-ram 45056 --ctx-checkpoints 32 --cache-idle-slots -bs ^
  --mmproj mmproj-Qwen3.8-27B-F16.gguf --image-min-tokens 1024 --image-max-tokens 4096 ^
  --spec-draft-model Qwen3.8-27B-DFlash2-Q8_0.gguf --spec-draft-device CUDA0 ^
  --spec-type draft-dflash --spec-draft-n-max 7 --spec-rs-seq 7 ^
  --jinja --reasoning-format deepseek --reasoning-budget -1 --no-context-shift --n-predict -1 ^
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --repeat-penalty 1 --metrics
```

<details>
<summary><b>What the important flags do</b></summary>

<br>

| Flag | Why |
|:--|:--|
| `--cache-type-k/v turbot` | the tiered KV cache: a full 262K context at q8-level quality |
| `--kv-unified --parallel 4` | four slots share one KV pool; every slot can use the whole context |
| `--spec-type draft-dflash --spec-draft-n-max 7 --spec-rs-seq 7` | DFlash2 block drafting; keep `--spec-rs-seq` equal to `n-max` (never `0`) |
| `-bs` | sampling on the GPU (it steps aside by itself while a tool-call grammar is active) |
| `--load-mode none` | weights read fully into memory (replaces the deprecated `--no-mmap`) |
| `--cache-ram 45056 --ctx-checkpoints 32 --cache-idle-slots` | the prompt cache in RAM, so returning to a conversation does not re-read it |
| `SPEC_DFT_UBATCH=128` | a smaller drafter batch: about 1.2 GB VRAM back, same output |
| `GGML_DISABLE_VULKAN=1` | skip the Vulkan backend when you do not need it |

The same command works for **UD-Q4_K_XL** with the `DFlash2-Q4_K_M` drafter.

</details>

### Qwen3.8-Flash-Next

UD-Q4_K_XL with the Q8_0 MTP head (the 5-shard `...-MTP-0000x-of-00005.gguf` set), 262K context, 1 stream. Needs ~28 GB of free VRAM and most of 96 GB RAM.

The MTP head ships as a separate file (`mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`); merge it into the Unsloth split once — no copy of the 100 GB trunk, the tensor shards are hard-linked:

```bat
python tools/qwen4exp/merge_mtp.py --src-pattern Qwen3.8-Flash-Next-UD-Q4_K_XL-%05d-of-00004.gguf --n-src 4 ^
  --head mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf ^
  --out-pattern Qwen3.8-Flash-Next-UD-Q4_K_XL-MTP-%05d-of-00005.gguf --mtp-ratio 0
```

```bat
set GGML_DISABLE_VULKAN=1

llama-server.exe --model Qwen3.8-Flash-Next-UD-Q4_K_XL-MTP-00001-of-00005.gguf --host 0.0.0.0 --port 8080 ^
  --device CUDA0 -ngl 99 -fit off --ctx-size 262144 ^
  --cache-type-k turbot --cache-type-v turbot --flash-attn on ^
  --threads 16 --cpu-mask 55555555 --cpu-strict 1 --prio 2 --parallel 1 --cache-ram 0 ^
  --n-cpu-moe 48 --batch-size 8192 --ubatch-size 8192 ^
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.5 --spec-draft-sampling probabilistic ^
  --jinja --reasoning-format deepseek --reasoning-budget -1 --no-context-shift --n-predict -1 ^
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --repeat-penalty 1 --metrics
```

<details>
<summary><b>What the important flags do</b></summary>

<br>

| Flag | Why |
|:--|:--|
| `--n-cpu-moe 48` | all experts start in RAM; the engine moves the ones you use into a GPU hot set by itself |
| `--batch-size 8192 --ubatch-size 8192` | large prompt batches, so prompt reading streams experts over PCIe at 1.5–2.3K t/s |
| `--cpu-mask 55555555 --cpu-strict 1 --threads 16` | one CPU worker per physical core for the CPU expert pool |
| `--cache-ram 0` | the model already uses most of the RAM |
| `--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.5` | drafting with the model's own MTP head (draft length 3 measured slower) |
| `-fit off` | the engine sizes VRAM itself (hot set, borrowed prompt buffer) |

- **Keep the model memory-mapped**: never `--load-mode none` / `--no-mmap` — it is larger than RAM.
- The first one or two answers after a load are slower while the GPU expert set fills.
- **Recommended:** the `...-MTPq3-*` shard set (the same model with a 1.3 GB instead of 2.8 GB draft head): the freed VRAM holds ~10 more experts per layer, real use +6 % (103 / 93 / 91 t/s at 32K / 131K / 245K). The main model checks every token either way, so the output quality is the same. Make it once from the shipped head: `python tools/qwen4exp/mtp_requant.py --head mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf --mix q3_K,q3_K,q4_0 --out mtp-q3.gguf`, then the `merge_mtp.py` line above with `--head mtp-q3.gguf --out-pattern Qwen3.8-Flash-Next-UD-Q4_K_XL-MTPq3-%05d-of-00005.gguf`.
- `LLAMA_FLASHNEXT_PROFILE=off` turns the whole engine off (for comparison).

</details>

---

## How it works

<details>
<summary><b>turbot, the tiered KV cache</b></summary>

<br>

turbot keeps three tiers per attention layer: the newest tokens exact, a middle tier at 6–8 bits, and an old tier at a per-head bit budget. The quantized tiers sit behind a Walsh-Hadamard rotation with Lloyd-Max centroids (the TurboQuant method), and the attention kernels read every tier natively — no f16 copy of the cache is ever made. A built-in plan picks the tier widths per model; models it does not fit fall back to turbo5p, turbo5p512, turbo4 or q8_0 automatically.

</details>

<details>
<summary><b>The Flash-Next engine</b></summary>

<br>

Flash-Next has 48 layers of 512 experts (10 active per token) — about 72 GB of expert weights at Q4, more than the VRAM and more than the RAM next to it. A decode step is split per layer:

1. The GPU runs attention (linear GDN layers and sparse QSA layers with a lightning indexer) and the router, and posts the token's cold experts to the CPU through mapped memory.
2. While the CPU expert pool computes those cold experts at the RAM limit (~69 GB/s), the GPU computes the hot experts from its VRAM hot set and the shared expert, and prefetches the next layer's weights into L2.
3. A device-side hint tells the CPU which experts the next layer will probably need, so the pool pulls them into cache ahead of time.

The hot set adapts to the conversation and borrows the idle prompt buffer while generating; prompts stream all experts over PCIe Gen5 instead. The step ends in one CUDA graph per decode width. Everything is exact: the CPU and GPU paths compute the same values, and a stalled CPU job rolls the step back instead of corrupting it.

</details>


---

## Build from source

**Requirements** — CUDA Toolkit 13.1 · CMake 3.27+ · Ninja · MSVC 2022 Build Tools with the C++ workload (Windows) or GCC 11+ (Linux).

<details open>
<summary><b>Windows (CUDA)</b></summary>

<br>

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

cmake -B build -G Ninja -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DCMAKE_CUDA_ARCHITECTURES=120a -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 16
```

`120a` targets RTX 50-series only. For a portable binary use `"75;80;86;89;120;121"` (a much longer build). The full recipe is in [docs/BUILD-WINDOWS.md](docs/BUILD-WINDOWS.md).

> Stop any running `llama-server` before rebuilding: it holds the DLLs open and the link fails with `LNK1104`.

</details>

<details>
<summary><b>Linux (CUDA)</b></summary>

<br>

```bash
cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;120;121" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

</details>

---

## Releases

Every verified speed round is pushed to `main` and published under [Releases](https://github.com/sirxsniper/llama-cpp-turboquant/releases) with its measured numbers. A round is released only after its checks pass: quality (KLD, needle recall at 131K and 245K), output identity where the change claims it, compute-sanitizer on every new GPU path, and the Qwen3.8-27B gate (24/24 identical answers plus the full backend test suite).

---

## Credits

**TurboQuant is designed, built and maintained by [@sirxsniper](https://github.com/sirxsniper).** That covers:

- **turbot**, the tiered KV cache, with turbo4p / turbo5p / turbo5p512, the built-in per-model plans and the automatic KV-type resolution;
- **the Flash-Next engine**: the adaptive expert hot set, the MoE bridge and CPU expert pool, prefill streaming over PCIe, the compute-buffer lend, MTP drafting and every fused decode kernel of the `qwen4exp` path;
- **the long-context performance work**: native quantized attention at depth, the MMA shared-tile loader, the K/V byte-permute centroid gathers, coalesced dequant stores, GQA packing by exact divisor, the `nbatch_fa` retune;
- **speculative decoding**: DFlash2 integration and tuning, the speculative prefill tail, cost-aware MTP drafting and block verify;
- **reliability**: the GPU memory and race fixes found with compute-sanitizer, the clean-shutdown work, slot checkpoints, the vision device selection;
- the Windows CUDA build, the InnerQ cross-DLL ABI, and every upstream merge from `b8650` on.

**Prior work this fork builds on** (attribution follows the commit history; see [CREDITS.md](CREDITS.md)):

- the CPU TurboQuant implementation — **TheTom** (March 2026);
- the original CUDA port of the turbo formats — **Gabe Ortiz** (March 2026);
- TriAttention KV-cache pruning — **atomicmilkshake** (April 2026);
- method papers: TurboQuant ([arXiv 2504.19874](https://arxiv.org/abs/2504.19874)), TriAttention ([arXiv 2604.04921](https://arxiv.org/abs/2604.04921)).

---

<div align="center">

**Built on [llama.cpp](https://github.com/ggml-org/llama.cpp)** by Georgi Gerganov and the ggml-org contributors — the backbone this fork extends.

<sub>MIT licensed, as is upstream llama.cpp.</sub>

</div>
