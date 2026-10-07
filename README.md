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

## 99 t/s
real use at 32K · **89 t/s at 131K** filled

prompt reading **1,500–2,060 t/s**

</td>
</tr>
</table>

<sub>RTX 5090 32 GB · Ryzen 9 9950X3D · 96 GB DDR5-6400 · PCIe Gen5 · Windows 11 · turbot KV cache · measured 2026-10-06</sub>

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
- **prompt streaming over PCIe Gen5**, so long prompts read at 1.5–2K t/s;
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

| | first 262K build (Oct 4) | v0.15.0 | **v0.16.0** |
|:--|--:|--:|--:|
| Real use, 32K filled | 30.3 t/s | 91.0 t/s | **99.1 t/s** |
| Real use, 131K filled | 26.1 t/s | 82.8 t/s | **89.5 t/s** |
| Real use, 245K filled | 24.5 t/s | 84.8 t/s | **92.0 t/s** |
| Benchmark, greedy code | 39 t/s | 89 t/s | **93 t/s** |
| Prompt reading, fresh 32K / 131K | ~150 t/s | 1,508 / 2,059 t/s | about the same |

- Quality: KLD vs the original file **0.0109** (same top token 97.3 %); needle recall passes at 131K and 245K.
- v0.15.0 and v0.16.0 measured back to back in one session (2 interleaved rounds; the first depth after a load varies by about 3 %).
- VRAM about 29.9 GB while generating (the expert hot set takes the free VRAM up to 2.1 GB below the card total, plus the part of the 262K KV cache a shorter context does not use yet; `LLAMA_FN_VRAM_KEEP_MIB=4096` keeps more free); the rest of the model stays memory-mapped.
- Vision works: add `--mmproj mmproj-Qwen3.8-Flash-Next-F16.gguf`.

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
| `--batch-size 8192 --ubatch-size 8192` | large prompt batches, so prompt reading streams experts over PCIe at 1.5–2K t/s |
| `--cpu-mask 55555555 --cpu-strict 1 --threads 16` | one CPU worker per physical core for the CPU expert pool |
| `--cache-ram 0` | the model already uses most of the RAM |
| `--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.5` | drafting with the model's own MTP head (draft length 3 measured slower) |
| `-fit off` | the engine sizes VRAM itself (hot set, borrowed prompt buffer) |

- **Keep the model memory-mapped**: never `--load-mode none` / `--no-mmap` — it is larger than RAM.
- The first one or two answers after a load are slower while the GPU expert set fills.
- **Optional:** the `...-MTPq3-*` shard set (the same model with a 1.3 GB instead of 2.8 GB draft head) is about 2–3 % faster in real use. The main model checks every token either way.
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
