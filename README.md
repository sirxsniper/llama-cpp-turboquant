<div align="center">

# TurboQuant

**A full 262,144-token context on a single 32 GB GPU — at speed.**

[![License](https://img.shields.io/badge/License-MIT-1f6feb?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-13.1-76b900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-downloads)
[![Arch](https://img.shields.io/badge/SM-75%20→%20121-8957e5?style=for-the-badge)](#build)
[![KV](https://img.shields.io/badge/KV%20cache-4.25%20bpw-e3b341?style=for-the-badge)](#the-turbo-formats)
[![Base](https://img.shields.io/badge/llama.cpp-b11093-6e7681?style=for-the-badge)](https://github.com/ggml-org/llama.cpp)

<br>

### 262,144 tokens · 4.52 GiB of KV cache · 117 t/s at 131K

</div>

---

A performance fork of llama.cpp built around a single target: run a 27B model at its **full 256K context on one RTX 5090**, without giving up throughput or output quality.

That target pulls in two directions at once. The KV cache has to be small enough to fit in 32 GB next to the weights — and the attention path has to be fast enough that a small cache costs nothing to read. Almost all the work here is in closing the gap between those two, and almost all of it came from measuring rather than guessing.

<div align="center">

| | Prefill | **Decode** | KV cache |
|:--|--:|--:|--:|
| **131,072 tokens** | 1161 t/s | **117.7 t/s** | 2.26 GiB |
| **245,760 tokens** | 649 t/s | **91.4 t/s** | 4.24 GiB |
| **262,144 tokens** | — | — | **4.52 GiB** |

<sub><code>Qwen3.8-27B-UD-Q5_K_XL.gguf</code> on one RTX 5090, running
<a href="#the-exact-production-config">the production config</a> in full: <code>turbo4</code> K and V,
flash attention, and the DFlash2 drafter. Prefill is 3247 t/s at empty context. Decode is the median
of three 1500-token samples on a fixed prompt and moves with how predictable the text is, so read it
as a band rather than a constant. Strip the drafter out and the same build decodes 50.5 and 39.6 t/s
at those depths.</sub>

</div>

> Decode at 245,760 tokens began this work at **23.14 t/s**. The same cache as `f16` would need **17.00 GiB**, which does not fit beside the weights on a 32 GB card at all — measured, not estimated: llama.cpp refuses the allocation.

---

## Contents

| | |
|:--|:--|
| [Highlights](#highlights) | what this fork adds, and what each part bought |
| [Test bench](#test-bench) | the exact hardware and software behind every number |
| [Why Q5_K_XL](#why-the-bench-moved-to-q5_k_xl) | what the quant labels hide, and what the switch cost |
| [Benchmarks](#benchmarks) | full results, with methodology |
| [Build](#build) | Windows and Linux, from source |
| [Configuration](#configuration) | launch recipes, flag reference, tuning knobs |
| [Any model](#using-the-fork-with-any-model) | four connections (status, production command, switches), KV type resolved per model, built-in turbot plan, turbot on other models, context checkpoints in slot files, vision device, new switches and architectures |
| [How it works](#how-it-works) | the turbo formats and the attention path |
| [turbot](#turbot-tiered-kv-cache) | a tiered KV cache: better quality than turbo5p in the same VRAM |
| [Engineering log](#engineering-log) | every change, with before and after |
| [Measured and ruled out](#measured-and-ruled-out) | the negative results, with numbers |
| [Current focus](#current-focus) | where the remaining headroom is |
| [Credits](#credits) | upstream and prior work |

---

## Highlights

<table>
<tr>
<td width="50%" valign="top">

### A 4.25-bit KV cache

`turbo4` stores the cache at **4.25 bits per weight** — 16 Lloyd-Max centroids, nibble packed, behind a Walsh-Hadamard rotation.

A 262,144-token cache costs **4.52 GiB**, against 9.03 GiB as `q8_0` and 17.00 GiB as `f16`.

That is the difference between holding a full context on this card and not holding it.

</td>
<td width="50%" valign="top">

### Native quantized attention

Decode reads the quantized cache **directly at every depth**, rather than first materialising the whole thing as F16 once per layer, per token.

<div align="center">

**+45% decode at full context**

</div>

</td>
</tr>
<tr>
<td width="50%" valign="top">

### Byte-permute centroid gather

The V dequant broadcast its 16 centroids with 16 warp-serializing shuffles per call. Two hardware byte-permutes replace all of them.

<div align="center">

**+7.9% decode at 131K**

</div>

</td>
<td width="50%" valign="top">

### Attention tuned to the real shape

Grouped-query packing chosen by **exact divisor** instead of rounded up to the next power of two, plus a re-tuned softmax batch size.

<div align="center">

**+15–17% prefill at depth**

</div>

</td>
</tr>
<tr>
<td colspan="2" valign="top">

### Speculation that doesn't tax prefill

DFlash2 block-diffusion drafting, with the drafter skipped over the stretch of a long prompt its sliding window will evict before it can ever draft from it. **+10% prefill** with a drafter attached, and draft acceptance is unchanged — the skipped work was genuinely dead.

</td>
</tr>
</table>

---

## Test bench

Every number in this README comes from this machine. Nothing is inherited, estimated from another card, or carried over from a different model.

<table>
<tr><td valign="top" width="50%">

**Hardware**

| | |
|:--|:--|
| GPU | NVIDIA GeForce RTX 5090, 32,607 MiB |
| Driver | 610.88 |
| CPU | AMD Ryzen 9 9950X3D, 16C / 32T |
| RAM | 96 GB (2 x 48 GB @ 4800 MT/s) |

</td><td valign="top" width="50%">

**Software**

| | |
|:--|:--|
| OS | Windows 11 Pro, build 26200 |
| CUDA | 13.1 (V13.1.80) |
| Compiler | MSVC 14.44.35207 |
| CUDA arch | `120a` (Blackwell) |

</td></tr>
</table>

**Model under test**

| | |
|:--|:--|
| File | `Qwen3.8-27B-UD-Q5_K_XL.gguf`, 18.83 GiB (20,218,178,624 bytes), 27.32 B params |
| Architecture | 65 blocks — **17 full-attention**, 48 Gated DeltaNet |
| Attention shape | 24 query heads / 4 KV heads (GQA 6:1), head dim 256 |
| Drafter | `Qwen3.8-27B-DFlash2-Q8_0.gguf`, 1.92 GiB (2,056,414,496 bytes) |
| Compared against | `Qwen3.8-27B-UD-Q4_K_XL.gguf`, 16.69 GiB (17,923,394,624 bytes) |

> **Neither quant label means what it says.** `general.file_type` is one nominal stamp and Unsloth Dynamic quants do not honour it. Measured over the actual tensors, `UD-Q4_K_XL` is **5.25 bpw** and `UD-Q5_K_XL` is **5.92 bpw**. Both are 5-bit-class files, and the real gap between them is 13%, not a whole bit.

### Why the bench moved to Q5_K_XL

The label is the least informative thing about either file. Read as tensors:

| | effective bpw | file | tensor mix |
|:--|--:|--:|:--|
| `UD-Q4_K_XL` | 5.25 | 16.68 GiB | Q5_K 68.9%, **IQ4_XS 21.2%**, Q6_K 5.2%, Q4_K 4.7% |
| `UD-Q5_K_XL` | 5.92 | 18.82 GiB | Q5_K 61.8%, **Q6_K 37.6%**, Q8_0 0.5% |

What matters is the floor, not the average. A fifth of `Q4_K_XL` sits in IQ4_XS at 4.25 bpw. `Q5_K_XL` has nothing below Q5_K and puts 37.6% of the model in Q6_K, so its least precise weight is still more precise than a fifth of the Q4 file.

**What that costs.** One `llama-bench` run, same build, same flags, all three files back to back:

| | file size | decode | effective bandwidth | % of 1792 GB/s |
|:--|--:|--:|--:|--:|
| `UD-Q4_K_XL` | 17.92 GB | 62.54 t/s | 1121 GB/s | 62.6% |
| `UD-Q5_K_XL` | 20.22 GB | 56.88 t/s | 1150 GB/s | 64.2% |
| `Q6_K` (plain) | 22.88 GB | 48.27 t/s | 1105 GB/s | 61.6% |

Q5 costs **9.1% of decode** and 2.14 GiB of VRAM. The third column is the one that matters: all
three land within 4% of the same effective bandwidth, so decode here is memory-bound and the cost
of the switch is simply the extra bytes. **For decode there is no per-quant kernel penalty.**

**Prefill is a different story, and it does have one.** Decode is memory-bound; prefill is
compute-bound, so the tensor-core throughput of each format matters there and the quant mix stops
being neutral. `pp512` at empty context, five repetitions:

| | prefill | vs Q4 |
|:--|--:|--:|
| `UD-Q4_K_XL` | **3286.29** ± 21.49 t/s | — |
| `UD-Q5_K_XL` | 2976.86 ± 17.61 t/s | **-9.4%** |
| `Q6_K` (plain) | 2720.77 ± 25.48 t/s | -17.2% |

Measured three times in both orders; the gap held at 8.5 to 11.1% every time, so call it **~10%**.
Absolute figures drift with GPU clock state, the ratio does not.

The cause is `Q6_K`, which is the slowest MMQ format on this card. At this model's FFN shape
(m=17408, k=5120) it reaches **103.7 TFLOPS** against `Q4_K`'s 140.3 and `NVFP4`'s 293.3, and it is
beaten by `Q2_K`, `Q3_K`, `Q4_0`, `Q8_0` and every IQ format. `UD-Q5_K_XL` is 37.6% `Q6_K`, so it
computes at roughly 119 TFLOPS weighted against `UD-Q4_K_XL`'s 142. The plain `Q6_K` row above is
the control: 100% of the slow format, and duly the slowest of the three.

That is a real kernel gap rather than a tuning one. `Q6_K` uses an identical MMQ config to `NVFP4`
(256 threads, occupancy 1, I=128, stream-k on), sits on the same tensor-core path and needs
comparable shared memory, so the whole deficit is in its unpack, which has to combine 4-bit `ql`,
2-bit `qh` and 16 scales per 256-block. `Q4_K` hitting 140.3 at the same shape is the proof that
more is available.

The decode half of that corrected an earlier claim in this README. A first pass compared decode figures
taken from different runs, read the gap as a Q6_K kernel weakness, and said so. Measured properly,
in one run, `Q6_K` reads at the same GB/s as everything else. Isolated `MUL_MAT` benchmarks appear
to disagree and show `Q6_K` at half speed for batch widths 4 to 7, but a 73 MB weight tensor fits
inside this card's 96 MB L2, so that test measures cache-resident throughput and not the streaming
case a real decode hits. Lowering the MMVQ ceiling to move those widths onto MMQ was tried twice
and measured **43.31 ms/step against 42.44**, a 2% regression both times.

**An honest note on quality.** No local perplexity run has been able to separate these two files. The sweep that was run put `Q8_0` last, which is not a credible ordering and means it was measuring something other than quality. The case for Q5 rests on the precision floor above and on the fact that it still fits, not on a quality number this bench can defend.

> **Match the drafter to the target.** Moving the target to Q5 and leaving the drafter at `Q4_K_M` gives up most of the benefit. Measured at 131K, the `Q8_0` drafter reaches **63.8%** acceptance against **54.6%** for `Q4_K_M` at the same step cost, worth about 6% of end-to-end decode for 871 MiB. On a five-layer drafter `Q8_0` also has a cheaper dequant path than `Q4_K_M`, so the larger file is not the slower one.

---

## Benchmarks

### Speeds with the full flag set

Everything below is `turbo4` K and V, flash attention, all 65 layers on the GPU. Two ways of
measuring it, because they answer different questions.

**The speed you get.** Full production config, drafter included, median of three 1500-token
samples on a fixed prompt:

| Depth | Prefill | Decode | `ms/step` | tokens/step | KV size |
|------:|--------:|-------:|----------:|------------:|--------:|
| 131,072 | 1161 t/s | **117.7 t/s** | 42.44 | 4.97 | **2.26 GiB** |
| 245,760 | 649 t/s | **91.4 t/s** | 50.55 | 4.62 | **4.23 GiB** |

**The reproducible floor.** Same flags without the drafter, so anyone can check it:

```bash
llama-bench -m Qwen3.8-27B-UD-Q5_K_XL.gguf   -ctk turbo4 -ctv turbo4 -fa 1 -ngl 99 -t 16   -p 512 -n 64 -d 0,131072,245760 -r 3
```

| Depth | Prefill (pp512) | Decode (tg64) |
|------:|----------------:|--------------:|
| 0 | 3247.29 ± 8.19 | 50.57 ± 1.28 |
| 131,072 | 1161.05 ± 30.39 | 50.50 ± 0.83 |
| 245,760 | 649.30 ± 8.65 | 39.60 ± 0.28 |

The drafter is worth **2.3x** on top of that floor at both depths.

The one comparison worth printing: at 262,144 tokens `turbo4` holds the cache in **4.52 GiB**
where `f16` needs **17.00 GiB**, which does not fit beside an 18.8 GiB model on a 32 GB card.
Where `f16` does fit, `turbo4` is still ahead of it, by 5.3% decode at empty context and 32.3%
at 131,072, on a quarter of the memory.

### A note on quoting decode numbers

Speculative decode makes throughput depend on the content being written, which makes single
headline figures unreproducible. Measured on one build, one config and one cached 131K prompt,
varying nothing but the run:

| run | `ms/step` | draft acceptance | decode |
|--:|--:|--:|--:|
| A | 57.27 | 63.8% | 96.94 t/s |
| B | 43.48 | 67.5% | 131.50 t/s |
| C | 51.80 | 44.4% | 79.06 t/s |

Decode swings from 79 to 132 t/s on identical settings. A verification step costs the same
whatever survives of the draft, but how many tokens it retires depends on how predictable the
text is, and code is far more predictable than prose. Anything in this README that compares two
builds therefore reports `ms/step` from one session, and anything meant to be reproduced by
someone else is `llama-bench`.

### Draft length: pin it, do not let it float

`--spec-draft-n-min 7` is worth about **6%** and costs nothing. It is the one setting here that
is easy to miss.

A verification step reads the whole model whether the draft holds 2 tokens or 7. Measured
directly, a matmul at width 8 costs only 5 to 12% more than the same matmul at width 1 once the
weights are streaming from DRAM rather than sitting in L2. So the step is priced almost entirely
by weight traffic, and a short draft does not make it cheaper, it just retires fewer tokens.

Left to itself the drafter sometimes emits fewer than 7. Forcing the full block fills the step
that is being paid for either way. Four samples of 2000 tokens per arm at 131K:

| | `ms/step` | tokens/step | decode |
|:--|--:|--:|--:|
| default | 43.02 | 4.72 | 109.63 t/s |
| **`--spec-draft-n-min 7`** | **42.46** | **4.95** | **116.43 t/s** |

An earlier run of the same pair measured +11.2% rather than +6.2%, so treat the size as a band.
The direction held in both, and in the second the worst pinned sample still beat the median
unpinned one.

### Two settings not to change

> `n_max 7` (Q=8) measures 81.46 t/s median at 131K against
> 71.51 for `n_max 3`: the shorter draft accepts far more of what it writes (74.8% against
> 37.2%) but retires fewer tokens per step. Raising it past 7 is worse still — Q would exceed 8
> and silently fall off the packing rule below. And leave `--spec-draft-p-min` at 0: truncating
> low-confidence drafts sounds like it should save the verification, but measured 47.50 ms/step
> at 0.4 and 49.47 at 0.6 against **43.21** at 0.

### KV cache footprint

| KV type | bits/weight | @131,072 | @262,144 | Fits on 32 GB? |
|:--|--:|--:|--:|:--|
| **`turbo4`** | **4.25** | **2.26 GiB** | **4.52 GiB** | **yes, with room** |
| `f16` | 16.0 | 8.50 GiB | 17.00 GiB | **no**, 17 GiB of cache plus an 18.8 GiB model |

<sub>Sizes are exact for the geometry above: 17 attention layers × 4 KV heads × 256 dims × 2 (K and V). <code>turbo4</code> packs 128 elements into a 68-byte block.</sub>

### Where prefill time goes

At `d245760`, 675 ms per 512-token batch:

| Component | Time | Share |
|:--|--:|--:|
| Attention (17 layers, 5.3e13 flops) | ~521 ms | **77%** |
| FFN | ~143 ms | 21% |
| Gated DeltaNet (48 layers) | ~10 ms | 1.5% |

Attention runs at roughly **101 TFLOPS** there — about half the card's realistic ceiling for FP16 tensor ops with FP32 accumulation, which is normal-to-good for flash attention.

### CUDA graphs are load-bearing

Easy to assume this is already handled. It is not free, and it does not fall off at depth:

| Depth | Graphs on | Graphs off | |
|------:|----------:|-----------:|:--|
| 0 | 66.14 | 46.34 | **+42.7%** |
| 131,072 | 45.77 | 33.48 | **+36.7%** |

Chunked Gated DeltaNet disables graph capture, but only during prefill — decode uses the sequential path and keeps them.

<details>
<summary><b>Methodology</b> — how these were measured, and the traps</summary>

<br>

- **`llama-bench`, never the server.** With a speculative drafter attached, server decode rate tracks draft acceptance, which swings with prompt text and completely swamps kernel-level effects.
- **`-r 3` with stddev reported.** Single runs drift with GPU temperature; several conclusions in this project were nearly drawn from thermal noise.
- **One process at a time.** A second 27B model on a 32 GB card silently spills to system RAM through WDDM and still appears to work. Any number from an oversubscribed run is invalid — check for `failed to fit params to free device memory` and a non-zero `CUDA_Host model buffer size`.
- **Correctness before throughput.** Throughput measures perfectly well on a build emitting garbage. Every change here passes `test-backend-ops -o FLASH_ATTN_EXT` and needle-in-a-haystack retrieval at 245K before its numbers are quoted.
- **Never size the KV cache from `nvidia-smi` deltas.** Allocator pooling and compute buffers swamp it.

</details>

---

## Build

**Requirements** — CUDA Toolkit 13.1 · CMake 3.27+ · Ninja 1.11+ · MSVC 2022 BuildTools with the C++ workload and Windows 11 SDK (Windows) or GCC 11+ (Linux).

<details open>
<summary><b>Windows (CUDA)</b></summary>

<br>

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

cmake -B build -G Ninja ^
  -DGGML_CUDA=ON ^
  -DGGML_CUDA_FA=ON ^
  -DCMAKE_CUDA_ARCHITECTURES=120a ^
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j 16 --target llama-server llama-bench test-backend-ops
```

`120a` targets Blackwell specifically. For a portable binary use `"75;80;86;89;120;121"` instead, at the cost of a much longer build.

The full recipe with toolchain pins and known gotchas is in [docs/BUILD-WINDOWS.md](docs/BUILD-WINDOWS.md).

> **Stop any running `llama-server` before rebuilding.** It holds `ggml-base.dll` open and the link fails with `LNK1104`, which some build wrappers report as success.

</details>

<details>
<summary><b>Linux (CUDA)</b></summary>

<br>

```bash
cmake -B build \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;120;121" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build -j$(nproc) --target llama-server llama-bench test-backend-ops
```

</details>

<details>
<summary><b>Verify the build</b></summary>

<br>

```bash
./build/bin/test-backend-ops -o FLASH_ATTN_EXT -b CUDA0
```

Expect `2/2 backends passed`. This covers the turbo K/V flash-attention paths, including the D=256 cases this fork added.

</details>

---

## Configuration

<details open>
<summary><b>Full 256K context on a 32 GB card</b> — the configuration this fork exists for</summary>

<br>

`turbo4` KV is what makes the context fit; the DFlash2 drafter is what makes it fast.

```bash
llama-server -m Qwen3.8-27B-UD-Q5_K_XL.gguf \
  -c 262144 -ngl 99 -fa 1 \
  -ctk turbo4 -ctv turbo4 \
  --kv-unified \
  -md Qwen3.8-27B-DFlash2-Q8_0.gguf \
  --spec-type draft-dflash --spec-draft-n-max 7 --spec-draft-n-min 7 \
  --host 0.0.0.0 --port 8080
```

</details>

<details>
<summary><b>Without a drafter</b> — lower decode, less VRAM, no prefill cost</summary>

<br>

```bash
llama-server -m Qwen3.8-27B-UD-Q5_K_XL.gguf \
  -c 262144 -ngl 99 -fa 1 \
  -ctk turbo4 -ctv turbo4 \
  --kv-unified \
  --port 8080
```

</details>

### The exact production config

Copy-paste runnable on a 32 GB Blackwell card. This is the configuration the benchmarks above were produced with.

```bash
llama-server   -m  Qwen3.8-27B-UD-Q5_K_XL.gguf   -md Qwen3.8-27B-DFlash2-Q8_0.gguf   --spec-type draft-dflash --spec-draft-n-max 7 --spec-draft-n-min 7   -c 262144 -ngl 99 -fa 1   -ctk turbo4 -ctv turbo4   --kv-unified   -b 2048 -ub 512   -t 16 --threads-batch 16   --parallel 1   --load-mode none   --jinja --chat-template-file qwen-fixed-chat-template.jinja   --host 0.0.0.0 --port 8080 --metrics
```

<details>
<summary><b>What each block is doing</b></summary>

<br>

| Block | Why |
|:--|:--|
| `-md` + `--spec-type draft-dflash` | The 1.92 GiB DFlash2 drafter at `Q8_0`. Yields 4.8 to 5.4 tokens per step. Drop both lines to save its VRAM at roughly 40% of decode. Use the `Q8_0` build of the drafter and not `Q4_K_M`, which measures 9 points of acceptance worse against a Q5 target at identical step cost. |
| `--spec-draft-n-max 7` | DFlash2 trains at `block_size` 8, so 7 is its ceiling. Higher is wasted work. |
| `-c 262144 -ctk/-ctv turbo4` | The full context, at 4.52 GiB of cache. `f16` here needs 17.00 GiB and is refused. |
| `--kv-unified` | One shared cache instead of per-sequence. Needed to fit at max context. |
| `-b 2048 -ub 512` | `-ub` is the cheap VRAM lever — 1024 → 512 frees roughly 360 MiB at this context for very little throughput. Raise it if you have headroom. |
| `-t 16 --threads-batch 16` | Physical cores. Everything heavy is on the GPU; these only feed it. |
| `--load-mode none` | Load resident, no mmap. Measured worth +17.7% on a CPU-offloaded MoE and it removes most of the run-to-run variance. Replaces the removed `--no-mmap`. |
| `--parallel 1` | One slot. Concurrent slots divide the context, and this configuration exists to give one request all of it. |

</details>

<details>
<summary><b>Chat template</b> — required for correct tool calling on Qwen3.8</summary>

<br>

The stock template embedded in the GGUF mis-serialises parallel tool calls. Point `--chat-template-file` at a corrected one; the two fixes that matter are:

- **Parallel `<tool_call>` blocks must be joined with a single newline, not a blank line.** The model emits them with one `
`. Joining with `

` changes the prefix on every multi-tool turn, which invalidates the KV cache prefix and forces a full re-prefill of the conversation each time. On a long context that is the single most expensive formatting bug available.
- **`error:` / `fatal:` matched anywhere in tool output** would trip failure handling on any text merely containing those words. They must be anchored to line start.

Set the default reasoning effort in the template to taste — this bench runs `xhigh`.

If you have no corrected template, run without `--chat-template-file`; everything in this README still holds, since the benchmarks are `llama-bench` and do not involve templating at all.

</details>

### MoE models: `--load-mode none` is not optional

For a mixture-of-experts model with expert layers offloaded to CPU (`-ncmoe`), the loader prints this and it should be obeyed:

```
tensor overrides to CPU are used with mmap enabled - consider using --load-mode none
```

Measured on Qwen3.8-Flash-Next (87 GiB, 512 experts, 31 layers on CPU):

| load mode | decode |
|:--|--:|
| `auto` (mmap) | 26.72 ± 3.52 |
| **`none`** | **31.46 ± 0.36** |

**+17.7%, and the error bar collapses by an order of magnitude.** That second number is the tell: the variance *was* mmap page-fault jitter. With CPU-resident experts the decode path re-reads roughly 809 MB of expert weights per token, and serving that through a memory mapping costs both throughput and predictability.

```bash
llama-server -m Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf   -c 131072 -ngl 99 -fa 1   -ctk turbo4 -ctv turbo4 --kv-unified   -ncmoe 31 --load-mode none   -t 16 --parallel 1 --port 8080
```

> `-ncmoe` is a fit decision, not a speed dial, and the right value depends on the context you actually use. At short prompts 30 is faster than 31 (32.37 against 31.43), but at `d131072` the KV cache is real and 30 spills — decode collapses to 6.46 t/s. Use 31 for long context.

### Flags that carry real weight

| Flag | Why it matters |
|:--|:--|
| `-ctk turbo4 -ctv turbo4` | 4.25 bpw KV. Without it a 262K context needs 17 GiB and will not fit. |
| `-fa 1` | Flash attention. Everything in this fork assumes it. |
| `--kv-unified` | One shared cache rather than per-sequence. Needed to fit at max context. |
| `--spec-type draft-dflash` | Block-diffusion drafting, ~3.05 tokens accepted per step. |
| `--spec-draft-n-max 7` | DFlash2 trains at `block_size` 8, so 7 is its natural ceiling. |
| `-ngl 99` | All layers on the GPU. |

> **Qwen3.8 and MTP.** Every Qwen3.8 GGUF ships an MTP head, and it is **silently ignored** unless you pass `--spec-type draft-mtp`. It cannot be combined with a separate drafter such as DFlash2 — `common_memory` owns a single draft context — and this fork rejects that pairing with an explicit reason. A draft model that is itself an MTP head (an `mtp-*.gguf` sidecar, as Nemotron 3.5 ships) is accepted. DFlash2 yields more per step (3.05 vs 2.55), so prefer it when the VRAM is there.

### Tuning knobs

Environment variables, for A/B testing rather than daily use.

| Variable | Default | Effect |
|:--|:--|:--|
| `SPEC_PREFILL_TAIL` | `2048` | Prompt tail the drafter actually prefills. `0` disables the skip. |
| `TURBO_MMA_NATIVE` | `1` | Native turbo reads in the MMA tile loader. `0` restores F16 conversion. |
| `TURBO_MMA_NATIVE_MAXQ` | `32` | Q width below which native reads are used. |
| `FA_NCOLS2_MAXQ` | `8` | Q width up to which FA rounds GQA packing up instead of using the exact divisor. `0` disables the rule. |
| `FA_NCOLS2` | off | `1\|2\|4\|8` forces the MMA GQA packing factor outright. |
| `FA_VEC_GQA` | off | `1..6` forces the VEC GQA packing factor. |
| `GGML_CUDA_MMVQ_MAX_K` | off | `1..8` overrides the k-quant MMVQ batch ceiling. Raising it measured slower here. |
| `SPEC_PHASE_PROBE` | off | `1` prints a wall-clock phase breakdown of a speculative step every 128 steps. |
| `TURBO_FA_MMA` | off | `1` restores the old depth-based MMA routing. |
| `TURBO_IDX_INHERIT` | off | `1` lets the sparse-attention indexer cache inherit the turbo type. |

---

## Using the fork with any model

Qwen3.8-27B is the model this fork is tuned for: hybrid Gated DeltaNet, 16 attention layers, 4 KV heads × 256. Other software also uses the fork as a general llama.cpp, so any model that fits the card has to load and run. A fork feature that cannot serve a model falls back to something that works and says so in one log line. It never stops the server from starting, and it never runs a path that would compute wrong output.

> **Status, 2026-09-22.** Built and measured on the `upstream-sync` branch (upstream `fb34fc262`, `b11093`), RTX 5090, old build = pre-sync `f52b7db3b` run back to back on the same day.
> - Test suites: `validate.ps1` 16732/16732 (the 64 old hsk=40 failures are fixed), FLASH_ATTN_EXT 4309/4309, turbot 195/195, compute-sanitizer memcheck and synccheck 0 errors, CPU FLASH_ATTN_EXT 5698/5698.
> - Qwen3.8-27B UD-Q5_K_XL, production server (262K, 4 slots, turbot, DFlash2 n_max 3): prose 106-107 t/s (old 103.5-105), code 147-151 t/s (old 142-143), 24.2 ms/step (old 25.0). KLD against the same reference: turbot code 0.001139 (old 0.001137), prose 0.001856 (old 0.001848); turbo5p code 0.001595, prose 0.002583. llama-bench tg64 new ≥ old at 0/131K/245K for turbo5p and turbot.
> - **Fixed: wrong tokens under concurrency** (`[TAG_XSEQ_PLANES]`, `LLAMA_XSEQ_FIX=0` restores the old path). With `--spec-rs-seq` and several slots, a split batch could move one sequence's recurrent cell after it had written its rollback snapshots; a later draft rejection then restored another sequence's state. Measured: 14 anomalous greedy tokens in 640 concurrent requests before, 0 in 320 after. The bug was in the pre-sync builds too.
> - Every new model listed below loads and was checked for facts, a tool call, vision where it has a projector, and speed.
>
> **Newer, built and gated on 2026-09-22** (`docs/turbot/TESTING.md` 9.9). Two changes sit on top of the measured build:
> - turbot on other KV shapes (`[TAG_TURBOT_ANY_*]`, commits `1bbcb0dc2`, `84f49d76e`, `4f5aa27d3`, `642d0dce7`): see "turbot on other models" below.
> - Context checkpoints in slot files (`[TAG_SLOT_FILE_CKPT]`, commit `6e505dbcc`): see "Slot files keep their context checkpoints" below.
>
> Qwen3.8-27B is bit-identical to the measured build (gate G2): the same SASS for every gated turbot kernel, the same test cases, the same greedy transcripts and DFlash2 acceptance, the same plan and size log lines, and KLD code 0.001139 and prose 0.001856 exactly. `LLAMA_TURBOT_ANY=0` with `GGML_TURBOT_ANY=0`, and `LLAMA_SLOT_FILE_CKPT=0`, restore the behaviour before these changes.

> **Status, 2026-09-25: four connections, deployed** (`docs/turbot/TESTING.md` section 10). Commits `0ae47b552`..`b5a5c7d54` make 4 concurrent streams faster and the production server about 2 GB smaller, with the same quality. Measured against the build deployed on 2026-09-22 (`f76a7a4b5`), interleaved runs with identical flags:
> - **Decode** (ms per speculative step, the fair metric when greedy texts fork at near-ties): 1 stream +0.3 % (inside the 1 % run-to-run spread), 2 streams -2 %, 4 streams -6.6 %. 4 streams at d=0: code 261.8 -> 290.6 t/s total, prose 235.7 -> 259.3.
> - **Prefill**, cold 16K code prompts: 1 stream 2515 t/s (old) -> 2559 (new, old flags) -> 2630 (new, `-b 2048 -ub 1024`); 4 streams 2075 -> 2093 -> 2262 t/s total. Cold 131K, back to back: prose 1670 -> 1708 t/s, code 1691 -> 1702.
> - **VRAM**: about 2.0 GB less (28.8 -> 26.7 GB after load). The recurrent state buffer drops from 2394 to 693 MiB, the target compute buffer from 397 to 124 MiB, the drafter's from 138 to 37 MiB. That leaves room for the vision encoder on the GPU: a 4000-token image encodes in 1.4 s instead of 36 s on the CPU.
> - **Quality**: KLD 16 x 32K identical to the old build bit for bit (turbot code 0.001139, prose 0.001856); needles at 131K and 245K and the 3-needle test pass; 1-stream greedy texts identical to the old build (with `GDN_CHUNKED_PF=0`; the chunked prefill changes the computation order, KL 0.0011 against a floor of 0.0010 for any order change).
> - **Correctness**: validate.ps1 `GATE PASSED` on the deployed binaries (test-backend-ops 18714/18714), compute-sanitizer 0 errors on the new kernels, crosstalk 0 cross-slot answers in 152 requests (vision on the GPU, a shared 4K prefix, UD-Q4_K_XL and Uncensored included), eos_repro 320 graded requests with 0 anomalous tokens (14 differences, all near-ties), slot park/resume and prompt-cache restore token-identical. Every Jarvis profile (Ornith, Nemotron, Muse, Spark, MiniCPM5, the NEO-CODER fine-tunes) passed its smoke test on the new build.
>
> Production command since 2026-09-25 (`launch-qwen38-dflash2.bat` and the Jarvis profiles of UD-Q5_K_XL, UD-Q4_K_XL and Uncensored-Q5_K_M; changes against the previous one marked):
> ```
> set SPEC_DFT_UBATCH=128
> set GGML_DISABLE_VULKAN=1              (a Vulkan build would list the 5090 again as Vulkan0)
> llama-server -m Qwen3.8-27B-UD-Q5_K_XL.gguf -c 262144 -ctk turbot -ctv turbot -bs -fa on --kv-unified -ngl 99
>   -b 2048 -ub 1024                     (was -b 1024 -ub 512)
>   --parallel 4 --threads 16 --load-mode none --cache-ram 45056 --ctx-checkpoints 32 --cache-idle-slots
>   --spec-draft-model Qwen3.8-27B-DFlash2-Q8_0.gguf --spec-type draft-dflash --spec-draft-n-max 3 --spec-rs-seq 3
>   --mmproj mmproj-Qwen3.8-27B-F16.gguf -mmdev gpu --image-min-tokens 1024 --image-max-tokens 4096   (was the CPU)
>   --device CUDA0 --spec-draft-device CUDA0
>   --jinja --chat-template-file qwen-fixed-chat-template.jinja --reasoning-format deepseek --reasoning-budget -1
> ```
> Fallbacks: `-b 1024 -ub 512` when decode matters more than prefill (0.5 % faster per step, 5 % slower prefill); `-mmdev cpu` when the desktop needs more than about 2.2 GB of VRAM (the image then takes about 36 s, but the other slots keep generating while it encodes).

### KV cache type, resolved per model

Ask for a cache type with `-ctk`/`-ctv` as before. `llama_init_from_model` now checks the request against the model before the cache is built (`[TAG_KV_RESOLVE]`, `src/llama-context.cpp`). When the model cannot take a type, it steps down this chain:

```
turbot -> turbo5p (turbo5p512 for 512-element rows) -> turbo4 -> q8_0 -> f16
```

Each downgrade logs one warning that gives the reason for every type it passed over, for example:

```
llama_init_from_model: KV cache type for K and V: turbot -> turbo5p (turbot: SWA caches are unsupported)
```

(Since turbot takes iSWA models, this particular warning appears only with `LLAMA_TURBOT_ISWA=0` or on an all-SWA model.)

The `llama_kv_cache: size = ...` line shows the types actually used. A request the model supports is left as it is, with no new log line. Qwen3.8-27B with turbot or turbo5p is unchanged.

| Type | Kept when |
|:--|:--|
| `turbot` | All of these hold (`[TAG_TURBOT_ANY_*]`, see "turbot on other models" below): <br>• `-ctk turbot -ctv turbot` are given together. <br>• Flash attention is not off. <br>• There is no MLA and there are no shared cells. <br>• Every turbot layer has K and V heads of the same size, in one of the shapes 4 × 256, 2 × 256, 1 × 256, 8 × 128, 4 × 128 or 2 × 128 (KV heads × head size), with its KV on CUDA (Turing or newer). <br>• SWA only as one half of an iSWA model: turbot goes on the full-attention layers and the SWA layers resolve their own type (turbo5p where the row allows it). All-SWA models fall back. <br>• Several KV streams (`-np N` without `--kv-unified`) are allowed, with one tier per stream. <br>• The arch has the turbo query rotation, and the cache size is a multiple of 64 cells. <br>• `TURBO_KV_CPU_LAYERS`, `TURBO_LAYER_ADAPTIVE` and `TURBO_INNERQ` are unset. <br>• There is a plan for the model: a plan file, a verified sidecar, the built-in plan (its `L` lines are exactly the model's attention layers), or the automatic plan for a validated shape that fits the fallback type's bytes. <br>`LLAMA_TURBOT_ANY=0` restores the old rule: one stream, no SWA, 4 × 256 only, and a plan whose `L` lines match. |
| `turbo5p`, `turbo5p512` | Flash attention is not off, and K and V heads are the same size, 128 or 256. Every KV row (KV heads × head size) must be a multiple of 512. When a row is not a multiple of 1024, turbo5p runs as turbo5p512. That swap logs an INFO line and does not count as a downgrade. |
| `turbo4p` | The same as turbo5p, but rows must be a multiple of 1024. |
| `turbo4`, `turbo3`, `turbo2` | Flash attention is not off. Asked for by name, any head the zero-padding path has a CUDA kernel for is kept: K = V of 128, 256 or 512 after padding to 128, or MLA with K 576 and V 512. As a fallback step, only unpadded heads of 128 or 256 are taken. |
| `q8_0` and the other block types | The head size is a multiple of the block (32 for q8_0). A quantized V also needs flash attention. |
| `f16` | Always. |

Other rules:
- **Archs that get no turbo type.** Their attention input has no turbo query rotation: DeepSeek 3.2 and V4, GLM DSA, Hy V4, dots3-note, the MiniMax-M3 sparse layers and the Qwen4exp QSA layers. MLA models get no split-plane type (turbo4p, turbo5p, turbo5p512).
- **Mixed K/V pairs** are resolved side by side. If the resulting pair has no flash-attention kernel, both sides take the lower turbo type, or a lone turbo side becomes q8_0.
- **Flash attention off (`-fa off`):** K can stay q8_0; a quantized V becomes f16.
- **Recurrent-only models** have no KV cache. The requested types are ignored, with one INFO line.
- **Safety net:** if the cache constructor still refuses turbot, the context rebuilds its memory once without turbot.
- **`LLAMA_KV_RESOLVE=0`** turns the resolver off. The old checks then apply: a type the model cannot take fails context creation with the reason.

What `-ctk turbot -ctv turbot` resolves to on a few shapes. These follow from the rules above and SPEC section 14 (`tests/test-kv-resolve.cpp` checks them on synthetic hyperparameters). The rows with a model on disk were confirmed on the GPU on 2026-09-22 (TESTING.md 9.9):

| Model | KV shape | Result |
|:--|:--|:--|
| Qwen3.8-27B | 16 attention layers, 4 × 256; the built-in plan fits | turbot, built-in plan, unchanged |
| Qwen3.8-27B with `-np 4` and no `--kv-unified` | 4 KV streams | turbot, one tier per stream (`LLAMA_TURBOT_MULTI_STREAM=0`: turbo5p) |
| Ornith-1.5-9B | 8 attention layers, 4 × 256 | turbot, automatic plan |
| Spark-X2.5-4B | iSWA, 4 × 256 | turbot on the 9 full-attention layers (automatic plan), turbo5p on the SWA layers |
| Spark-X2.5-1.7B | iSWA, 2 × 256 (512-element rows) | turbo5p512 on every layer. With `LLAMA_TURBOT_AUTO_PLAN=all`: turbot on the full-attention layers (automatic plan), turbo5p512 on the SWA layers |
| Ornith-1.5-35B | 10 attention layers, 2 × 256 | turbo5p512 (2 × 256 failed the quality gate on this model). With `LLAMA_TURBOT_AUTO_PLAN=all`: turbot, automatic plan (2 runs of 256 values per row) |
| Granite 4.2 8B | 8 × 128 | turbo5p; turbot with `LLAMA_TURBOT_AUTO_PLAN=all` (the shape has kernels but no model on disk to validate them) |
| MiniCPM5-2B | 2 × 128 (256-element rows) | turbo4; turbot with `LLAMA_TURBOT_AUTO_BUDGET=turbo5p` |
| Muse Glimmer 30B | iSWA, 2 × 128 | turbo4; with `LLAMA_TURBOT_AUTO_BUDGET=turbo5p`, turbot on the full-attention layers and turbo4 on the SWA layers |
| Nemotron 3.5 Lightning 30B-A3B | 6 attention layers, 2 × 128 | turbo4; turbot with `LLAMA_TURBOT_AUTO_BUDGET=turbo5p` |
| any model with head size 64 | 8 × 64 | q8_0 |
| any model with head size 80 | 8 × 80 | f16 |
| MLA (DeepSeek-V2 style) | K 576, V 512 | q8_0 |
| recurrent-only (Mamba) | no KV cache | f16, with no warning |

### turbot plan, built in

The default plan, `docs/turbot/plans/turbot-default.plan`, is calibrated on Qwen3.8-27B and compiled into libllama (`[TAG_TURBOT_EMBED_PLAN]`). `-ctk turbot -ctv turbot` therefore needs no plan file.

| You give | Plan used |
|:--|:--|
| nothing | Qwen3.8-27B-shaped models: the built-in plan, with one INFO line. Other models: a verified sidecar `<model>.turbot.plan` if there is one, otherwise the automatic plan (below). |
| `--kv-tier-plan default` or `LLAMA_TURBOT_PLAN=default` | the built-in plan |
| `--kv-tier-plan auto` or `LLAMA_TURBOT_PLAN=auto` | the automatic plan |
| `--kv-tier-plan <file>` or `LLAMA_TURBOT_PLAN=<file>` | that file only (write `./default` or `./auto` for files with those names). A file that does not fit falls back down the type chain; it never becomes an automatic plan. |

The flag wins over the variable. If the plan does not fit the model (its `L` lines are not the model's attention layers) or cannot be read, the resolver moves to turbo5p (or to the next type the model takes) and logs why. Fine-tunes that keep Qwen3.8-27B's attention layers (4 × 256 at layers 3, 7, ..., 63) match the built-in plan, so they run turbot with widths calibrated on the base model. Their quality is not yet measured.

### turbot on other models

> **Status, 2026-09-22: committed (`1bbcb0dc2`, `84f49d76e`, `4f5aa27d3`, `642d0dce7`), built and gated** (`docs/turbot/TESTING.md` 9.9): correctness (G1-G3) and Qwen3.8-27B bit identity (G2) pass; quality and VRAM were measured on Ornith-1.5-9B, Spark-X2.5-4B and Ornith-1.5-35B; speed (G4, G6) is not measured yet. SPEC section 14 has the details.

turbot used to take only Qwen3.8-27B's shape, 4 KV heads × 256. It now also takes 2 × 256, 1 × 256, 8 × 128, 4 × 128 and 2 × 128 (`[TAG_TURBOT_ANY_*]`). A row holds 1, 2 or 4 runs of 256 values, each with its own old and young width, so the format, the young tier and the plan work the same way at every shape. Qwen3.8-27B keeps the built-in plan and runs exactly the same kernels and numbers as before (G2, bit for bit).

What each shape gets with `-ctk turbot -ctv turbot`:

| KV shape (heads × head size) | Runs per row | Default | Opt-in | Examples |
|:--|:-:|:--|:--|:--|
| 4 × 256 | 4 | turbot: the built-in plan when the attention layers are Qwen3.8-27B's, else the automatic plan | — | Qwen3.8-27B, Ornith-1.5-9B, Spark-X2.5-4B |
| 2 × 256 | 2 | turbo5p512 (failed the G5 quality gate on Ornith-1.5-35B) | `LLAMA_TURBOT_AUTO_PLAN=all` | Ornith-1.5-35B, Spark-X2.5-1.7B |
| 8 × 128 | 4 | turbo5p | `LLAMA_TURBOT_AUTO_PLAN=all` | Granite 4.2 8B |
| 4 × 128 | 2 | turbo5p512 | `LLAMA_TURBOT_AUTO_PLAN=all` | — |
| 2 × 128 | 1 | turbo4 | `LLAMA_TURBOT_AUTO_BUDGET=turbo5p` (uses more VRAM than turbo4) | MiniCPM5-2B, Muse Glimmer 30B, Nemotron 3.5 30B |
| 1 × 256 | 1 | turbo4 | `LLAMA_TURBOT_AUTO_BUDGET=turbo5p` | — |

The head-size-128 shapes run on 16 new CUDA flash-attention kernels. Every shape can also run as several KV streams (`-np N` without `--kv-unified`, one tier per stream) and on the full-attention layers of an iSWA model.

**Not covered.** These still step down the chain, and the warning line says why:
- head sizes other than 128 and 256, and rows above 1024 values (8 × 256, 16 × 128);
- K and V heads of different sizes, MLA, shared cells, and archs without the turbo query rotation;
- all-SWA models, and the SWA layers of an iSWA model (they take turbo5p, turbo5p512 or turbo4);
- KV anywhere but a CUDA GPU of the Turing generation or newer: Vulkan, Metal, ROCm, or KV layers moved to the CPU with `TURBO_KV_CPU_LAYERS`;
- the DFlash2 and MTP draft contexts;
- context shift (`seq_add` / `seq_div`), which turbot has never supported.

The automatic plan is uncalibrated: it uses the same 4/5-bit split on every layer. A calibrated plan for another model can go next to the model as a verified sidecar. `tools/turbot/turbot_guard.py --plan <file> --sidecar` measures the plan against the fallback type and stamps it only when it passes.

**Default policy.** `-ctk turbot -ctv turbot` keeps turbot for a model only when the model meets the rules in the table above and has a plan. Everything else steps down the usual chain (turbot → turbo5p / turbo5p512 → turbo4 → q8_0 → f16), with one warning per step. The plan comes from, in this order: `--kv-tier-plan` / `LLAMA_TURBOT_PLAN`; a sidecar `<model>.turbot.plan` with a `# verified:` stamp and a matching `# model:` fingerprint; the built-in plan when it names exactly the model's attention layers and shape; the automatic plan. The DFlash2 and MTP draft contexts never use turbot.

**Automatic plan.** It gives every run an old width of 4 or 5 and every young run width 7. It uses the most width-5 runs that still fit in the bytes of the type the model would otherwise get (turbo5p, turbo5p512 or turbo4), so it never uses more VRAM than the fallback. It only runs for the main context, with the resolver on, and only for shapes on the VALIDATED list. After the gates of 2026-09-22 the list is 4 × 256 only: 2 × 256 failed G5 on Ornith-1.5-35B (see the table). `LLAMA_TURBOT_AUTO_PLAN_DUMP=<file>` writes the generated plan.

Measured on 2026-09-22 (TESTING.md 9.9). KLD: llama-perplexity, prose corpus, 32K × 8 chunks, `-ub 512`, against f16 base logits of the same model; the turbot arm ran with no plan (the resolver's own choice). KV size: the `llama_kv_cache: size` line at 262144 cells, one sequence.

| Model | `-ctk turbot` by default | KLD turbot / fallback (same-top, p99.9) | KV at 262K, turbot / fallback |
|:--|:--|:--|:--|
| Ornith-1.5-9B | turbot, automatic plan | 0.005942 / turbo5p 0.011281 (98.10 / 97.39 %, 0.86 / 1.94): passes | 2572.59 / 2624.00 MiB |
| Spark-X2.5-4B | turbot on the 9 full-attention layers, turbo5p on the 27 SWA layers | 0.010718 / turbo5p 0.014022 (95.62 / 94.86 %, 0.50 / 0.61): passes | 2894.17 + 34.59 / 2952.00 + 34.59 MiB |
| Ornith-1.5-35B | **turbo5p512** (was turbot, automatic plan) | 0.088414 / turbo5p512 0.070221 (91.12 / 91.41 %, 11.59 / 10.43): **fails** with the 32K automatic plan (4.0-bit old rows); the 262K plan (4.75 bits) is not measured yet | 1650.39 / 1680.00 MiB |
| MiniCPM5-2B, Muse Glimmer 30B, Nemotron 3.5 30B | turbo4 (2 × 128), one warning. turbot only with `LLAMA_TURBOT_AUTO_BUDGET=turbo5p`: 4-bit old rows alone cost 144 B per 256 values, turbo4 136 B. | not measured | turbo4: 2856 / 884 + 25.9 / 408 MiB. Opt-in sizing (not measured): +316 / +98 / +45 MiB |
| Granite 4.2 8B (8 × 128) | turbo5p. turbot only with `LLAMA_TURBOT_AUTO_PLAN=all` | no model on disk | — |
| Qwen3.8-27B, `-np 4` without `--kv-unified` | turbot on 4 streams | not measured | — |

With no cache type given, every model starts with f16 as before. All six models above answered three facts and a tool call correctly on a 262K server with `-ctk turbot`. On Ornith-1.5-35B llama-perplexity does not repeat itself (a second f16 run scores KLD 0.029 against the first; turbo5p512 repeated at 0.0697), so its KLD figures carry that noise; the one turbot run (0.0884) is above both turbo5p512 runs. Speed (G4, G6) is not measured yet.

**Switches.** Each one restores the behaviour before this change for its part:

| Variable | Default | Effect when set |
|:--|:--|:--|
| `LLAMA_TURBOT_ANY` | on | `0` restores exactly the old turbot rules: 4 × 256 only, no automatic plan, no sidecar, no iSWA split, one stream. Every other switch below then reads as off. |
| `GGML_TURBOT_ANY` | on | `0` makes the CUDA backend accept only the 4 × 256 kernels and writer, today's routing. |
| `LLAMA_TURBOT_AUTO_PLAN` | `1` | `0`: no automatic plan; a model without a plan falls back to turbo5p. `all`: every supported shape, not just the validated ones. |
| `LLAMA_TURBOT_AUTO_BUDGET` | unset | `turbo5p` budgets the automatic plan at turbo5p's rate even where the fallback is turbo4 (the 2 × 128 models). This uses more VRAM than the fallback. |
| `LLAMA_TURBOT_AUTO_PLAN_DUMP` | unset | `<file>` writes the generated plan text. |
| `LLAMA_TURBOT_SIDECAR` | on | `0` ignores `<model>.turbot.plan`. |
| `LLAMA_TURBOT_ISWA` | on | `0`: SWA models refuse turbot and take turbo5p for all layers, as before. |
| `LLAMA_TURBOT_SWA_TYPE` | unset | `<type>` sets the cache type of the SWA layers of an iSWA model, for A/B runs. Unset, they resolve turbo5p through their own chain. |
| `LLAMA_TURBOT_MULTI_STREAM` | on | `0`: several KV streams refuse turbot and take turbo5p, as before. |

`LLAMA_KV_RESOLVE=0` and `LLAMA_TURBOT=0` keep their meaning. With the resolver off, the cache uses the old plan precedence plus the explicit `auto` keyword only.

The build option `-DGGML_CUDA_FA_TURBOT_D128=OFF` leaves out the 16 head-size-128 turbot kernels (shorter CUDA build). Head-size-128 models then step down to their fallback type: the CUDA backend tells the resolver which shapes it can run.

### Slot files keep their context checkpoints

> **Status, 2026-09-22: committed (`6e505dbcc`), built and partly measured** (`docs/turbot/TESTING.md` 9.9). On Qwen3.8-27B (production command) the first request after a restore processed 4 prompt tokens instead of 20015. Without a draft model the output tokens are identical to the run before the save. With DFlash2 they are not: the restored draft state drafts differently (acceptance 67/84 against 65/86), and the greedy text leaves the reference at token 87 of 96. The prompt cache and the live slot are not affected. Use `LLAMA_SLOT_FILE_CKPT=0` where a restore must reproduce the earlier output exactly. A file written by the deployed build still restores (full prompt, identical tokens, no warning).

`/slots/{id}?action=save` and `?action=restore` (with `--slot-save-path`) now also carry the slot's context checkpoints (`[TAG_SLOT_FILE_CKPT]`, `tools/server/server-context.cpp`).

This matters for hybrid models such as Qwen3.8-27B:
- The server always evaluates at least one prompt token, so reusing a saved prompt needs the model state from at least one token earlier.
- The Gated DeltaNet state cannot be rolled back. Only a context checkpoint gives that earlier state.
- The live slot and the RAM prompt cache (`--cache-ram`) already kept checkpoints. A slot file did not, so the first request after a restore processed the whole prompt again.

What changes:
- **Save.** The file gets a versioned section after the state. It holds the checkpoints, each with its own checksum, and the draft model's sequence state when one is loaded (DFlash2, MTP).
- **Restore.** The slot gets the checkpoints back, up to the same limits a live slot has (`--ctx-checkpoints`, and `LLAMA_CTX_CHECKPOINT_BUDGET_MIB`, whose default of 2048 MiB allows 13 checkpoints on Qwen3.8-27B). The next request that re-sends the prompt only processes the tokens after the newest checkpoint, the same as after a RAM prompt cache hit. That is expected to be about 4 tokens. The draft sequence is restored too. When the saved draft state does not fit the loaded draft model, the draft sequence starts empty.
- **Safety.** The section is bound to its file (state size, prompt tokens) and to the loaded models. A section that does not match, or that fails a checksum, is ignored with one warning; the restore still succeeds and the prompt is processed in full, as before.
- **Compatibility.** Older servers read the new files and ignore the section. Old files restore exactly as before. A slot with no checkpoints and no draft model writes a file byte-identical to the old format.
- **Size.** Each checkpoint adds 149.6 MiB to the file on Qwen3.8-27B, about 1.9 GiB for 13. The save hashes it on the main loop, which is estimated (not measured) to pause the server for 1-3 s per save. `LLAMA_SLOT_FILE_CKPT_KEEP` limits this.

| Variable | Default | Effect |
|:--|:--|:--|
| `LLAMA_SLOT_FILE_CKPT` | `1` | `0`: a save appends nothing and a restore ignores any section, as before. |
| `LLAMA_SLOT_FILE_CKPT_KEEP` | all | A save writes only the newest N checkpoints. `0` keeps the draft state but writes no checkpoints. |

No kernel, KV cache or turbot code changes: the restored checkpoints are the same bytes the live slot would hold, loaded by the existing checkpoint restore. `tools/server/README.md` has the file format. Measured: the `save_restore` arm of `tools/turbot/blob_roundtrip.py` (4 prompt tokens after the restore; identical tokens without a draft model, not with DFlash2, see the status note), the old-file check, `LLAMA_SLOT_FILE_CKPT=0` (old file size, full prompt, identical tokens) and a corrupt last byte (section ignored with one warning, full prompt, identical tokens, server healthy). Not run yet: the budget check.

### Vision: pick the device

`-mmdev` (`--mmproj-device`, env `MTMD_BACKEND_DEVICE`) now also accepts a device type (`[TAG_MMDEV_TYPE]`):

| `-mmdev` | The vision encoder runs on |
|:--|:--|
| `cpu` (same as `none`) | the CPU, with `--mmproj-threads` threads |
| `gpu` | the first discrete GPU, which is CUDA0 because CUDA registers first. With no GPU device, the CPU (see below). |
| `igpu` | the first integrated GPU, for example an AMD iGPU through Vulkan. This needs a build with `-DGGML_VULKAN=ON` (Vulkan SDK). It is never the discrete GPU (see below). With no iGPU, the CPU. |
| a device name | that device, as before. A name that does not exist, such as `Vulkan7`, is still an error. |

When `gpu` or `igpu` finds no device of that type, the server still starts (`[TAG_MMDEV_FALLBACK]`). This happens on a build without Vulkan, with `GGML_DISABLE_VULKAN` set, or on a machine without an iGPU. The vision encoder then runs on the CPU, exactly as with `-mmdev cpu`, and one warning says why:

```
--mmproj-device igpu: no integrated GPU device found (Vulkan not built, GGML_DISABLE_VULKAN set, or no iGPU), the multimodal projector runs on the CPU instead
```

`igpu` cannot pick the RTX 5090 through Vulkan. The Vulkan backend reports a device as an integrated GPU only when the driver gives its type as `INTEGRATED_GPU`. The NVIDIA driver gives a discrete card the type `DISCRETE_GPU`, so `Vulkan0` counts as a discrete GPU, like `CUDA0`. As a second guard, `igpu` skips an integrated-GPU device that has exactly the name of a discrete-GPU device, and logs that it did.

`--mmproj-threads N` (server only, env `LLAMA_ARG_MMPROJ_THREADS`) sets the number of CPU threads for the vision encoder. `0`, the default, uses `-t`.

**Device pinning rule.** Since the upstream sync, the vision encoder and the draft model follow `--device` unless they are set on their own. A build with both CUDA and Vulkan sees the RTX 5090 twice, as `CUDA0` and `Vulkan0`, and the iGPU as `Vulkan1`. In such a build, always pin all three:

```
--device CUDA0 --spec-draft-device CUDA0 -mmdev cpu|gpu|igpu
```

Set the Vulkan environment to match:

| Vision on | Environment |
|:--|:--|
| `cpu` or `gpu` | `GGML_DISABLE_VULKAN=1`. The process then has only CUDA0 and the CPU. |
| `igpu` | `VK_LOADER_DRIVERS_SELECT=*amd-vulkan64*`, with `GGML_DISABLE_VULKAN` unset. The Vulkan loader loads only the AMD driver, so the 5090 never becomes a Vulkan device. `igpu` would not pick the 5090 without this, but the setting keeps the NVIDIA Vulkan driver out of the process. If `GGML_DISABLE_VULKAN` is left set, `igpu` falls back to the CPU with the warning above. |

Do not use `GGML_VK_VISIBLE_DEVICES` for this. It takes a raw device index, and that index depends on the order in which the drivers enumerate.

Other changes in the same area:
- The Vulkan backend refuses every fork turbo type and the fork's extra flash-attention inputs (`[TAG_VK_NO_TURBO]`), so a turbo cache never runs on Vulkan.
- A failed image encode fails only that request and frees its slot; the server keeps running (`[TAG_MTMD_ENCODE_CATCH]`).
- An encode that fails on a GPU other than CUDA no longer leaves the vision encoder broken (`[TAG_MTMD_DEVICE_FALLBACK]`, `tools/mtmd/clip.cpp`). The case seen: with `-mmdev igpu`, a ~4000-token image lost the Vulkan device after about 11 s (a Windows TDR). The encoder now logs one warning and loads a CPU copy of its weights from the mmproj file. It frees the device's buffers and runs the same encode again on the CPU, so the request still succeeds. It stays on the CPU until the server restarts.
- On an integrated GPU, an image of more than `MTMD_IGPU_MAX_TOKENS` output tokens (default 1536, `0` = off) is encoded on that CPU copy directly, with one INFO line. Smaller images stay on the iGPU. Measured: 1066 tokens took 44 s on the iGPU and 4.7 s on the CPU. Once loaded, the CPU copy stays in RAM beside the iGPU weights. `MTMD_DEVICE_FALLBACK=0` turns off both the fallback and this limit. CUDA devices are not affected.
- The CPU flash attention now takes its tiled path for head sizes 72 (the Qwen3.8 vision encoder) and 40 on AVX-512 (`[TAG_CPU_FA_DV_PAD]`). On the CPU, the encoder's K and V stay F32 into flash attention (`[TAG_CLIP_CPU_KV_F32]`).
- Images and audio are encoded on a thread of their own when the vision encoder runs on the CPU and the text model on a GPU, or on a GPU the text model does not use, such as the iGPU (`[TAG_MTMD_ASYNC_ENCODE]`). The request waits at its image until the encode is done, and the other slots keep generating in the meantime. The embeddings then go into the context on the inference thread through the same code as before, so the request's positions, checkpoints and DFlash2 drafter repair are unchanged. On a CUDA device the encode stays on the inference thread, and every slot waits for it as before.
- While a request waits for its encode, `/slots` shows `waiting_media: true` and a `media_encode` object (state, chunks, tokens, time queued and encoding). `/metrics` adds `requests_waiting_media` and `media_encode_jobs`. A request that is cancelled during its encode frees its slot at once. The encode itself cannot be interrupted, so it runs to the end and its result is dropped.

Measured on Qwen3.8-27B with the F16 mmproj (encode time for a ~1000-token / ~4000-token image):

| `-mmdev` | ~1000 tokens | ~4000 tokens |
|:--|--:|--:|
| `gpu` (CUDA0) | 0.13 s | 1.26 s |
| `cpu`, this branch | 4.7 s | 36 s |
| `cpu`, pre-sync build | 26.6 s | 352 s |
| `igpu` (2-CU Radeon) | 44 s | moved to the CPU by the limit above (38 s total); with the limit off, the device is lost and the encode is retried on the CPU |

With a 4-slot server and the encoder on the CPU, three text streams kept generating at about 64 t/s while a ~4000-token image was encoded (the longest gap was 1.5 s). The pre-sync build stalled them for 342 s.

### Loading: `--load-mode none` replaces `--no-mmap`

Upstream removed `--mmap`, `--no-mmap`, `--mlock` and `--direct-io` / `--no-direct-io` (#28334), and llama-bench lost `-mmp`. Use `-lm` / `--load-mode` instead: `auto`, `none`, `mmap`, `mlock`, `mmap+mlock` or `dio`. llama-bench takes the same `-lm`.
- `--no-mmap` becomes `--load-mode none`.
- This fork keeps the removed flags as deprecated aliases (`[TAG_SYNC_MMAP_COMPAT]`): `--no-mmap`, `--mmap`, `--mlock` and `--direct-io` still work and log one deprecation warning, so existing launchers keep starting.
- Builds on the pre-sync base `b10655` already accept `--load-mode`, so one command line works for both.

### New switches on the sync branch

Every switch defaults to the new behaviour. The defaults are what the measurements in the status note above ran with; the fused GDN norm is bit-identical to the unfused one (same KLD to six digits). Two sets of switches are listed elsewhere and are not measured yet: turbot on other shapes (`LLAMA_TURBOT_ANY` and the rest) under "turbot on other models", and `LLAMA_SLOT_FILE_CKPT` / `LLAMA_SLOT_FILE_CKPT_KEEP` under "Slot files keep their context checkpoints".

| Variable | Default | Effect when set |
|:--|:--|:--|
| `LLAMA_KV_RESOLVE` | on | `0` turns the KV type resolver off, so the old refusals apply. |
| `TURBOT_Q2_ROUTE` | on | `0` sends two-token turbot batches (a one-token draft verify, or two slots decoding) back to the `<2,8>` instance. Q = 1 stays on `<4,8>`. |
| `TURBO_RMSNORM_SCALE_FUSION` | on | `0` makes CUDA run the GDN q/k norm (RMS_NORM, then SCALE) as two kernels instead of one fused kernel. `GGML_CUDA_DISABLE_FUSION` turns it off too. |
| `GGML_CPU_FA_DV_PAD` | on | `0` restores upstream's rule that the CPU tiled flash attention needs a V head size that is a multiple of the SIMD width. Head sizes 72 and 40 on AVX-512 then take the per-row path again. |
| `MTMD_CPU_KV_F32` | on | `0` makes the vision encoder cast K and V to F16 before CPU flash attention, as upstream does. |
| `MTMD_ASYNC_ENCODE` | auto | `0` encodes images and audio on the inference thread everywhere, as upstream does, so every slot waits for the encode. `1` also uses the encoder thread when the text model runs on the same device (a CPU-only run, `-ngl 0`, or a shared Vulkan device). A CUDA projector always encodes on the inference thread. |
| `LLAMA_CTX_CHECKPOINT_MIN_STEP_ALWAYS` | off | `1` restores the pre-sync checkpoint eviction order: spacing eviction on every checkpoint, before the byte budget, and no replacement of a checkpoint at the same position. |
| `LLAMA_XSEQ_FIX` | on | `0` restores the old recurrent-state path, where a sequence moved inside a split batch could later restore another sequence's rollback snapshot. Only for A/B testing. |
| `MTMD_IGPU_MAX_TOKENS` | 1536 | Images above this many output tokens are encoded on the CPU when the projector is on an iGPU. `0` = no limit. |
| `MTMD_DEVICE_FALLBACK` | on | `0` turns off the CPU fallback after a failed non-CUDA encode, and the iGPU limit. |
| `SPEC_DFT_DUMP` | unset | `<file>` writes the DFlash2 selector lattice (top-k ids and scores for each drafted position) after every drafter decode, so two builds can be compared offline. |
| `TURBO_MMA_NATIVE` | `1` | Existing switch. `0` now also sends turbo5p512 back to the F16 conversion path. |

`TURBO_MMA_NATIVE` matters here because turbo5p512 now has its own MMA kernel at head size 256 (`[TAG_TURBO5P512_MMA]`). Before, a turbo5p512 cache whose rows were a multiple of 1024 reached the f16 kernel with raw turbo5p512 bytes and produced garbage. Caches with 512-element rows took the slower F16 conversion. The new kernel passes the `split_plane` FLASH_ATTN_EXT cases and compute-sanitizer memcheck (0 errors).

### Four connections: switches

Each switch restores the build of 2026-09-22 for its part. Leave them unset in production.

| Variable | Default | Effect when set |
|:--|:--|:--|
| `SPEC_DFT_SYNC` | auto | The DFlash2 drafter is synchronized after the inject only while a sequence is in prefill. `2` restores the old auto rule, which synced on every step with 3 or more generating slots. `0` is the old `0` rule, `1` syncs after every chunk. `SPEC_DFT_SYNC_PREFILL=0` also skips the prefill sync. |
| `DFLASH_RESERVE_FULL` | off | `1` reserves the drafter's token graph at `n_ubatch` (128) rows again: 137.53 MiB instead of about 37. |
| `TURBOT_QUOTA` | water-filling | `prop` restores the proportional young-pool quota (SPEC 9.6). Water-filling gives short slots next to a long one their whole young tier: 1 x 200K + 3 x 2K gives each 2K slot 2048 young cells instead of 646. |
| `LLAMA_KQ_MASK_POS_MS` | on | `0` builds the F16 explicit attention mask on the host for ubatches of several sequences, as before (256 MiB of compute buffer and 276 MiB of pinned host memory at 4 slots). `GGML_CUDA_FA_POS_MS=0` keeps the positional mask on the host but builds the explicit mask on the GPU for every CUDA kernel. |
| `GDN_REPLAY` | on | `0` keeps `1 + n_rs_seq` recurrent state snapshots per slot (2394 MiB at 4 slots) instead of one committed state and a ring of the last `n_rs_seq` tokens (693 MiB). Read when the context is created. Older slot files load under replay; a file written with replay does not load under `GDN_REPLAY=0`, and the server then processes the prompt again. |
| `GDN_CHUNKED_PF` | on | `0` runs prompt prefill under replay on the sequential per-token GDN kernel. `GDN_CHUNKED_PF_MIN` (default 64, minimum 16) is the number of tokens per sequence before the last `n_rs_seq` from which a ubatch takes the chunked kernel. |
| `GGML_CUDA_SMALLB` | on (Blackwell) | `0` sends 5-16 column batches back to MMQ. `1` also enables the kernel on other Ampere+ NVIDIA GPUs. `GGML_CUDA_SMALLB_MAX_ROWS` (default 1024, `0` = no limit) is the largest weight row count it takes; above that MMQ is faster. |

### Architectures new with the sync

| Arch | Model | KV on this fork |
|:--|:--|:--|
| `spark2_5` | Spark-X2.5 4B and 1.7B (#27868) | iSWA. turbot goes on the full-attention layers and the SWA layers take turbo5p (4B, 4 × 256) or turbo5p512 (1.7B, 2 × 256); not yet measured. With `LLAMA_TURBOT_ISWA=0`, turbot falls back to turbo5p or turbo5p512 on every layer, as measured below. |
| `maple` | Maple 20B-A1B ternary MoE (#27000) | Upstream supports it on the CPU only. |
| `hy_v4` | Tencent Hy 4 preview (#28127) | No turbo type, because its attention has no turbo query rotation. It falls back to q8_0, or to f16 where q8_0 does not fit. |
| `hrm_text` | HrmTextForCausalLM, DFM Mimir 1B (#27625) | Resolved by the rules above. |
| `deepseek4v` (mtmd projector) | DeepSeek-V4-Flash-Vision-Exp (#28133) | The DeepSeek V4 text model takes no turbo type. |
| `nemotron_h` (NemotronHPuzzle) | Nemotron-3-Puzzle-75B-A9B (#25444) | Needs the cherry-picked #28717 below for the CUDA SSM scan. |

Upstream PRs cherry-picked on top of the sync, before they were merged upstream:
- **#29242:** the Muse Glimmer parser fix for a reply that starts with a tool call (`common/parsers/muse-glimmer.cpp`, with `test-chat` cases).
- **#28717:** the CUDA SSM scan for state size 96 (Nemotron 3 Puzzle), with `test-backend-ops` SSM_SCAN cases.

Models checked on this branch (single stream, 2K-token prompt; KV type as picked by the resolver from `-ctk turbot -ctv turbot`):

| Model | KV picked | Decode t/s | With speculative decoding |
|:--|:--|--:|:--|
| Qwen3.8-27B UD-Q5_K_XL | turbot | 106-107 prose / 147-151 code (server) | DFlash2 (above); in-model MTP `--spec-type draft-mtp`: 100 / 135 |
| Spark-X2.5-4B Q8_0 | turbo5p (SWA) | 119-121 | none shipped |
| MiniCPM5-2B Q8_0 | turbo4 (2 × 128) | 170-204 | — |
| Ornith-1.5-9B Q8_0 + mmproj | turbo5p (plan layers do not match) | 109-113 | MTP: 180 |
| Ornith-1.5-35B-A3B Q4_K_M + mmproj | turbo5p (2 KV heads) | 165-174 | MTP: 255 |
| Muse Glimmer 30B Q4_K_M + mmproj | turbo4 (SWA) | 55 | its DFlash drafter: 107-114 |
| Nemotron 3.5 Lightning 30B-A3B Q4_0 | turbo4 (2 × 128) | 275 | MTP sidecar: 422-442 |

All of them answered the fact checks correctly and returned a valid tool call; the vision models read the test image.

The "KV picked" column is what the measured build picked, before turbot took other shapes. With "turbot on other models" above, `-ctk turbot` now picks turbot on Ornith-1.5-9B and on the full-attention layers of Spark-X2.5-4B (measured 2026-09-22, TESTING.md 9.9); Ornith-1.5-35B stays on turbo5p512 because 2 × 256 failed the quality gate. Nemotron's MTP ships as a separate `mtp-*.gguf`: use `--spec-type draft-mtp --spec-draft-model mtp-<model>.gguf`. The fork had refused that pairing; it now only refuses draft-mtp together with a non-MTP drafter.

---

## How it works

### The turbo formats

| Format | Bits/weight | Construction |
|:--|--:|:--|
| **`turbo4_0`** | **4.25** | 16 Lloyd-Max centroids, nibble packed, WHT pre-rotation |
| `turbo3_0` | ~3.0 | sub-byte, Hadamard pre-rotation |
| `turbo2_0` | ~2.0 | WHT-space centroids |
| `turbot` | 4.3 old, 7 young | tiered: per-head 2-6 bit base codes for every cell, 7-bit refinement for each sequence's newest 16K tokens |

Kernels cover Turing (SM75), Ampere (SM80/86), Ada (SM89) and Blackwell (SM120/121). `turbo4` is the one that matters here.

<details>
<summary><b>Why decode has to read the cache natively</b></summary>

<br>

llama.cpp has two flash-attention implementations. The **VEC** kernel reads a quantized KV cache directly. The **MMA** kernel uses tensor cores and cannot — anything routed to it must first materialise the entire cache as F16, rebuilt once per layer, per token.

A turbo-specific rule was sending decode to MMA past 4096 tokens. `q8_0` and `f16` never had that rule, which is why `q8_0` measured *faster* than `turbo4` at depth despite reading twice the bytes. Removing it is worth **+45% at full context**, but it only became viable once the native path was made competitive — see the engineering log.

</details>

<details>
<summary><b>Why the centroid gather dominated decode</b></summary>

<br>

`turbo4` stores a 4-bit index per element plus a per-block norm; reading a value means unpacking a nibble, looking up one of 16 centroids, and scaling. The lookup is paid **per element, per layer, per token**, so its cost scales with context depth.

The obvious 16-entry lookup is a chain of dependent selects. Doing that once per nibble builds a ~12-deep dependent ALU chain per four KV elements. The V path instead spread the centroids across an 8-lane sub-group and broadcast them with `__shfl_sync` — two shuffles per element, 16 warp-serializing shuffles per 8-element call.

Both are now a **byte-permute gather**: `PRMT` indexes an 8-byte pool, so 16 centroids need two permutes plus a per-byte blend on index bit 3. The blend operands are independent of the permutes, making the whole thing ~2 dependent levels instead of ~12, with no warp synchronization at all.

</details>

<details>
<summary><b>Why grouped-query packing was throwing away a quarter of the work</b></summary>

<br>

`ncols2` controls how many query heads share a single K/V tile read. The selection ladder rounded **up** to the next power of two — so this model's `gqa_ratio` of 24/4 = 6 was assigned 8 columns, and the kernel computed 8 heads' worth of attention for 6 real heads. A quarter of the math, discarded, at every depth.

Choosing by exact divisor instead (6 % 2 == 0, so 2) fixed it.

</details>

### turbot tiered KV cache

`turbot` keeps every cell as a compact per-head base code (2-6 bits in the WHT domain, widths chosen per layer, head and side by a plan file) and adds a 7-bit nested refinement for the newest 16,384 tokens of each sequence. Aging is metadata only: a granule of 64 cells drops its refinement slot when it leaves the young band, with no re-encode. The CUDA read runs natively in its own MMA instances; the writer is a single `TURBOT_SET_ROWS` op per layer side.

```
-ctk turbot -ctv turbot
```

The default plan, `docs/turbot/plans/turbot-default.plan` (calibrated on Qwen3.8-27B), is built into libllama. Plan selection:

- **No `--kv-tier-plan` and no `LLAMA_TURBOT_PLAN`:** the built-in plan.
- **`--kv-tier-plan default`** (or `LLAMA_TURBOT_PLAN=default`): the built-in plan, chosen explicitly.
- **`--kv-tier-plan <file>`:** that plan file. Use `./default` for a file named `default`.
- **`--kv-tier-plan auto`** (or `LLAMA_TURBOT_PLAN=auto`): the automatic plan (`[TAG_TURBOT_ANY_*]`, see [turbot on other models](#turbot-on-other-models)). Without a flag, a model that the built-in plan does not fit uses a verified sidecar plan or the automatic plan.

A plan is only used when its `L` lines are exactly the model's attention layers. When the plan does not fit (or the file cannot be read), turbot is not used for that model: the context falls back to turbo5p (or the next type the model supports) and the log says why. With `LLAMA_KV_RESOLVE=0`, context creation fails with the reason instead.

`LLAMA_TURBOT_PLAN` can carry the plan instead of the flag (llama-bench and llama-perplexity use it). `LLAMA_TURBOT=0` falls back to turbo5p. It needs flash attention and CUDA; the DFlash2 drafter cache stays turbo5p. Several KV streams and iSWA models are allowed since turbot took other shapes (not yet measured); `LLAMA_TURBOT_ANY=0` brings back the old requirement of one unified KV pool and no SWA. When a model or setting does not meet a turbot precondition, the KV resolver picks the next type and logs why ([KV cache type, resolved per model](#kv-cache-type-resolved-per-model)).

The built-in plan is compiled from `src/llama-turbot-default-plan.h`, a checked-in header generated by `python docs/turbot/gen_turbot_default_plan.py`. If you edit the plan file, re-run that script. CMake configure stops when the header and the plan differ, and `--check` reports the same thing.

<sub>Qwen3.8-27B-UD-Q5_K_XL, RTX 5090, 262,144 cells, same build for both caches.</sub>

| | turbo5p | **turbot** | f16 floor |
|:--|--:|--:|--:|
| KV cache at 262,144 cells | 5,248.00 MiB | **5,246.00 MiB** | |
| Code KLD / same-top (16 x 32K) | 0.001631 / 99.120% | **0.001137 / 99.267%** | 0.001041 / 99.306% |
| Prose KLD / same-top (16 x 32K) | 0.002550 / 97.862% | **0.001848 / 98.174%** | 0.001699 / 98.230% |
| tg64, depth 0 / 131K / 245K | 57.77 / 47.17 / 40.37 t/s | **61.39 / 50.07 / 43.78 t/s** | |
| pp512, depth 0 / 131K / 245K | 3463 / 1605 / 1091 t/s | 3437 / 1502 / 1003 t/s | |
| DFlash2 server ms/step, 131K / 200K | 26.20 / 28.20 | 27.04 / 28.72 | |
| DFlash2 acceptance, matched prompts, 131K | 0.783 | **0.788** | |

Short-context and multi-slot acceptance also match turbo5p within 0.02 (single prose 0.451 vs 0.440, code 0.764 vs 0.771, four agents 0.629 vs 0.638).

<details>
<summary><b>What made the read fast enough</b></summary>

<br>

The first working build decoded at 37.1 t/s at 131K, 21% below turbo5p. Four changes closed it:

- **Slowest-block layout.** stream_k blocks run in parallel and the op waits for the slowest one. With one KV head per block, a 6-bit head or a block full of young cells gated the whole op. Decode layouts now use striped blocks, so every block holds the same mix.
- **Compile-time widths.** The loaders are templated on the base width and refinement width, so plane layout, shifts and gather paths are constants. Mean -17% per op. Configs with 16 columns or fewer keep the runtime loaders to avoid a stack spill.
- **Q=1 routing.** Single-query decode runs the `<4,8>` instance instead of `<1,8>` (-46% per op), with a row-wrapping KV-bounds scan so the mask is never read past its last row.
- **KV_min classification.** A block that owns a whole output tile was treated as a fixup block once KV_min moved its start, leaving tiles unnormalised at ubatch 1280 with the positional mask.

</details>

<details>
<summary><b>Limits</b></summary>

<br>

- Prefill is 4-8% slower than turbo5p at 131K and 245K.
- A cached long prompt (over 16K tokens) can decode slightly differently from the same prompt sent cold: tail cells that aged out come back with fill codes when the generated tokens are trimmed.
- Two-token verify batches ran the `<2,8>` instance in the build measured above. The `upstream-sync` branch runs them on `<4,8>` (`TURBOT_Q2_ROUTE=0` restores `<2,8>`): pp2 at 131K 75.97 → 78.70 t/s, at 245K 67.34 → 70.55.

Contract and tests: [docs/turbot/SPEC.md](docs/turbot/SPEC.md), [docs/turbot/TESTING.md](docs/turbot/TESTING.md). Plan tools: `tools/turbot/`.

</details>

---

## Engineering log

<details open>
<summary><b>September 2026 — four connections</b> &nbsp;·&nbsp; <code>-6.6% step time at 4 streams, 2.0 GB less VRAM</code></summary>

<br>

With four agents on the server, a speculative step took 40 ms at d=0 against 26 ms for one stream, and the production server held 28.8 GB. Probes that attribute every step (`SPEC_PHASE_PROBE=1`) showed where both went. Seven changes, each with its own kill switch:

- **Drafter sync.** The DFlash2 drafter was synchronized after every inject whenever a batch held more than one live sequence, so every 4-slot step waited for it. It now syncs only while a sequence is in prefill (`SPEC_DFT_SYNC`).
- **Drafter reserve.** The drafter's token graph was reserved at `n_ubatch` = 128 rows, but a draft batch is `n_max + 1` rows per slot: 16 at 4 slots. 137.53 -> 36.79 MiB (`DFLASH_RESERVE_FULL`).
- **Positional mask for several sequences.** Since the cross-slot leak fix, multi-slot ubatches built a 256 MiB F16 attention mask on the host every step. The positional mask now carries a second row, the cell's set of sequences, and the flash-attention kernels test `(kv_seq & q_seq) != 0` beside the position (`LLAMA_KQ_MASK_POS_MS`). Target compute buffer 396.97 -> 124.04 MiB, pinned host 276 -> 22 MiB.
- **GDN rollback by replay.** Rolling back a rejected draft kept `1 + n_rs_seq` full recurrent states per slot. Now one committed state and a ring of the last 3 tokens' k, v, g and beta are kept, and a new op replays the ring before the new tokens. RS buffer 2394 -> 693 MiB. Replay is bit-identical to the snapshot kernel on decode (`GDN_REPLAY`).
- **Chunked prefill under replay.** With replay, 512-token prefill ubatches ran the per-token kernel. The prefix of each ubatch now takes the chunked kernel and only the last 3 tokens replay (`GDN_CHUNKED_PF`).
- **Small-batch tensor-core matmul.** At 2 and 4 streams the target runs 8 and 16 rows, which went to MMQ's 128-row tiles. A kernel that streams the weights once into int8 MMA fragments wins where MMQ gets few tiles: 48 rows 33 -> 7 us, 1024 rows 13.4 -> 11.3 us. At 2048 rows and more MMQ is faster, so it takes only weights of at most 1024 rows (`GGML_CUDA_SMALLB`, `GGML_CUDA_SMALLB_MAX_ROWS`).
- **Water-filling quota.** Short slots next to a long one lost their young tier to it (646 young cells for a 2K slot next to a 200K one). Each short slot now keeps all it can use and the long ones share the rest (`TURBOT_QUOTA`).

| ms per step (d=0) | old | new | |
|:--|--:|--:|:--|
| 1 stream, code / prose | 26.33 / 26.23 | 26.43 / 26.15 | within the 1% spread |
| 2 streams | 30.61 / 30.85 | 29.72 / 30.03 | -2.9% / -2.7% |
| 4 streams | 39.65 / 40.41 | 36.54 / 37.61 | **-7.8% / -6.9%** |

KLD is unchanged bit for bit, the greedy texts of one stream are identical to the old build, and 4-stream total throughput at d=0 went 261.8 -> 290.6 t/s on code. The 2 GB freed pays for the vision encoder on the GPU (a 4000-token image: 36 s -> 1.4 s) and for `-ub 1024`, which prefills 5% faster. Details and every gate: [docs/turbot/TESTING.md](docs/turbot/TESTING.md) section 10.

</details>

<details open>
<summary><b>September 2026 — turbot tiered KV cache</b> &nbsp;·&nbsp; <code>code KLD -30%, prose -28% vs turbo5p in the same VRAM</code></summary>

<br>

A two-tier KV cache sized to turbo5p's footprint: per-head base codes for every cell and a 7-bit refinement for each sequence's newest 16K tokens. Quality lands between turbo5p and q8_0 on code and at the f16 floor on prose. llama-bench decode is faster than turbo5p at every depth (50.07 vs 47.17 t/s at 131K, 43.78 vs 40.37 at 245K); the DFlash2 server step is within 3% of turbo5p. 195/195 turbot FA cases and 28/28 writer cases pass against the CPU reference, compute-sanitizer reports 0 errors, and every existing kernel keeps identical codegen.

</details>

<details open>
<summary><b>August 2026 — FA GQA packing chosen by Q width</b> &nbsp;·&nbsp; <code>-29% step time at 131K</code></summary>

<br>

Plain decode and speculative decode take different flash-attention kernels, and only one
of them was reading the cache once.

A speculative step submits `n_draft + 1` tokens for verification, so `Q->ne[1]` is 8 rather
than 1. That is wide enough to leave the VEC kernel — which packs six query heads per block
on this model and reads the cache **once** — and land on MMA, whose `ncols2` ladder picks the
largest power of two that *divides* `gqa_ratio`. At `gqa_ratio` 6 that is 2, so
`ntiles_z_gqa = 3` and every attention layer read the whole K/V region **three times per
step**, 17 layers deep.

**Why it hid.** The divisor rule is right, and there is a measured table in the source
saying so: at prefill it beats rounding up by 16.9% and 25.0%. But that was measured with
`Q = 512`, where the kernel is compute-bound and packing eight slots for six real heads
throws away a quarter of every tile. A verification batch is bound by the K/V read instead
and has almost no compute to waste, so the same rule inverts.

| `ncols2` | passes | `ms/step` | tok/s |
|---------:|-------:|----------:|------:|
| 1 | 6 | 120.52 | 30.00 |
| 2 | 3 | 60.51 | 64.93 |
| 4 | 2 | 53.39 | 89.02 |
| **8** | **1** | **43.58** | **107.00** |

Monotonic in the pass count, and eight wins despite the wasted slots. End to end that is
60.98 ms/step to 43.21 at 131K, a 29% cut; in throughput terms +41% at matched draft
acceptance, and more than that whenever the drafter happens to be doing well.

**The tail.** Gating this at `Q <= 32` — a number borrowed from the neighbouring
`TURBO_MMA_NATIVE` gate rather than measured — made widths 12, 16 and 32 *slower*, by up to
29%, because `ncols1` is capped at `64/ncols2`: past Q=8 the kernel re-tiles over Q as well
and pays the wasted slots **and** extra passes. Sweeping the width found a sharp boundary:

| `nb` | 4 | 8 | 12 | 16 | 32 |
|:--|--:|--:|--:|--:|--:|
| @131,072 | **-75.7%** | **-34.5%** | +3.8% | +3.4% | +26.9% |
| @245,760 | **-76.9%** | **-37.3%** | +3.8% | +4.0% | +29.4% |

Gated at 8 instead, every width outside 4..8 is neutral to within 0.6%. Widths below 4
never reach the code at all — `Q <= 2` is taken by VEC. The useful window is exactly a
speculative verification batch, which is what it was for.

This also removed an anomaly the width sweep exposed on the way in: `nb=4` used to cost
*more* than `nb=8` (874.56 µs against 523.78 at 131K). It is now 213.33.

`test-backend-ops FLASH_ATTN_EXT` passes 2/2 and a three-needle retrieval returns all
three codes exactly at both 131K and 245K context. `FA_NCOLS2_MAXQ` overrides the gate.

</details>

<details open>
<summary><b>August 2026 — GQA head packing in FA-vec</b> &nbsp;·&nbsp; <code>+21.9% decode</code></summary>

<br>

The flash-attention vec kernel bound one CUDA block to one query head:

```c
const int head = blockIdx.z - sequence*ne02;
K += nb13*sequence + nb12*(head / gqa_ratio);
```

So every query head in a group read **and dequantized** the same cache region — six times on this model, twelve on Qwen3.8-Flash-Next.

**Why it hid.** L2 absorbs most of the re-reads, so DRAM traffic is not `gqa_ratio` times higher and throughput was not `gqa_ratio` times worse. Only the dequant arithmetic is genuinely repeated, because each block dequantizes independently regardless of where the bytes came from.

**What exposed it.** On Flash-Next, `turbo4` measured *slower* than `q8_0` at identical settings — 27.20 against 29.39 — despite reading half the bytes. A format that reads less losing to one that reads more only makes sense if its per-element cost is being multiplied.

The kernel now takes an `ncols2` parameter: query heads packed per block, mirroring what the MMA path always did. Column `jc` addresses (token `jc/ncols2`, head `jc%ncols2`), with the Q pointer, alibi slope, attention sink, mask row, bounds test and both output writes resolved per column. `launch_fattn` already derived its grid from `ncols2` and was simply being passed a hardcoded `1`.

**The dependency that made it work.** Two-way packing helped; three- and four-way *collapsed*, to 26.28 and 13.97 against 49.18 — roughly halving per step, the signature of a register spill rather than of extra work. A packed column costs about 36 registers, dominated by `VKQ[(D/2)/nthreads_V]`. Raising `nthreads_V` for `ncols2 >= 3` halves that to ~18 and three-way then measured **52.19**, twice its spilled figure. The packing was never the problem; its register footprint was.

| Depth | `ncols2=1` | `ncols2=2` | `ncols2=3` | |
|------:|-----------:|-----------:|-----------:|:--|
| 131,072 | 46.53 | 49.75 | **52.13** | +12.2% |
| 245,760 | 35.96 | 41.57 | **43.55** | **+21.9%** |

The gain grows with depth because the redundant work scales with cache size. On Flash-Next the ordering flipped: `turbo4` went 25.13 → 27.91 and now beats `q8_0`'s 27.36 while using half the VRAM.

Selection is by **exact divisor**, never rounded up — at `gqa_ratio` 6, three-way needs two blocks and wastes nothing while four-way needs the same two blocks and wastes a quarter of its columns. Capped at 3: six-way **fails** `test-backend-ops` (1/2 backends), and without a cap the divisor rule would have selected exactly 6 for the very common `gqa_ratio` 6. `FA_VEC_GQA=<1..6>` overrides for measurement.

</details>

<details open>
<summary><b>August 2026 — turbo4 KV at depth</b> &nbsp;·&nbsp; <code>+45.4% decode</code></summary>

<br>

Decode with a `turbo4` cache was being routed away from the only kernel that can read it. Removing that routing required first making the native path competitive:

- the centroid lookup in the FA-vec `turbo4` dot product was a chain of three dependent selects called four times per four KV elements — replaced with two hardware byte-permutes plus a per-byte blend, verified bit-exact over all 65,536 possible inputs
- the `turbo4` dequant kernels wrote four consecutive elements as two 4-byte stores, discarding half of every 32-byte memory sector — now one wide store per lane
- a dequantizing tile loader lets the MMA kernel read `turbo4` directly for narrow Q, so speculative verification no longer pays the F16 materialisation either

| Depth | Before | After | |
|------:|-------:|------:|:--|
| 0 | 65.58 | 66.41 | |
| 65,536 | 45.10 | **53.82** | +19.3% |
| 131,072 | 33.61 | **44.39** | +32.1% |
| 245,760 | 23.14 | **33.64** | **+45.4%** |

For reference `q8_0`, which never had the bug, measures 35.33 at 245,760 — `turbo4` now sits just under it at half the KV VRAM. Speculative decode gains a further 11–15%. Prefill unchanged.

</details>

<details open>
<summary><b>August 2026 — the V centroid gather</b> &nbsp;·&nbsp; <code>+7.9% decode</code></summary>

<br>

Decode was reading the cache at 308 GB/s effective while `f16` hit 1621 GB/s moving 3.8× the bytes. Reading fewer bytes and taking longer pointed at the lookup, not the cache.

`dequantize_V_turbo4_0` broadcast its 16 scaled centroids with `__shfl_sync` — 16 warp-serializing shuffles per 8-element call, paid per V element per layer per token. The K dot had already been converted to a byte-permute LUT, and `qs_word` holds the eight nibbles in exactly that layout, so it applied to V unchanged.

| Depth | Before | After | |
|------:|-------:|------:|:--|
| 0 | 65.29 | 65.29 | unchanged — no KV to read |
| 131,072 | 43.51 | **46.93** | +7.9% |
| 245,760 | 34.05 | **36.50** | +7.2% |

The KV read now costs 6.00 ms against `q8_0`'s 6.01 ms at half the VRAM. Precision: the V gather reads the same int8 centroid table the K dot already used, each entry within 0.5% of its float value — far inside the error already introduced by binning to one of 16 centroids.

</details>

<details open>
<summary><b>August 2026 — prefill at depth</b> &nbsp;·&nbsp; <code>+17% prefill</code></summary>

<br>

Three independent findings, all from asking why attention ran so far under the card's ceiling.

**Grouped-query packing rounded the wrong way** — 25% of the attention math discarded at every depth. Fixed by selecting `ncols2` by exact divisor.

**The softmax batch size was tuned for a different shape** — raising `nbatch_fa` from 32 to 64 for the 256/256 case cut rescaling overhead. 128 exceeds the shared-memory budget and aborts, so 64 is the ceiling.

**The drafter prefilled prompt it could never help with** — drafting only ever begins at the tail, so `SPEC_PREFILL_TAIL` (default 2048) skips its forward passes over everything earlier.

| Depth | Before | After | |
|------:|-------:|------:|:--|
| 131,072 | 1023.15 | **1196.84** | +17.0% |
| 245,760 | 645.31 | **758.91** | +17.6% |

With a drafter attached `SPEC_PREFILL_TAIL` recovers a further ~10% on top, and acceptance is unchanged (29.6% → 29.4%) — the skipped work was dead.

</details>

<details>
<summary><b>May 2026 — earlier cycle</b></summary>

<br>

Commits `ccdce708f` to `fd0a94a4f` brought this fork from upstream `b8650` to `b9033` and reworked the turbo K/V flash-attention path. Full per-commit changelog with rationale in [docs/CHANGES-2026-05.md](docs/CHANGES-2026-05.md).

Reference decode from that cycle, on a **different model** (`Qwen3.6-27B-UD-Q6_K_XL`) than the one benchmarked above:

| Context fill | Decode |
|:--|--:|
| empty | 52.4 |
| 32k | 49.2 |
| 128k | 39.6 |

</details>

---

## Measured and ruled out

Negative results, kept because each one closes a question that looks promising from the outside and cost real time to settle.

| Idea | Result | Verdict |
|:--|:--|:--|
| Wider attention blocks (`nthreads` 128 → 256) | pp512 1196.84 → 1015.88 @131K | **−15%**, rejected |
| More resident blocks (`occupancy` 2 → 3) | pp512 1196.84 → 1102.26 @131K | **−8%**, rejected |
| `f16` KV for prefill instead of `turbo4` | 1105.15 → 1142.00 @131K, same run | +3%, not worth 3.8× the VRAM |
| NVFP4 KV on Blackwell FP4 tensor cores | faster, but loses needle recall | rejected on quality |
| FA-vec with 8 columns | no GQA packing → 6× cache re-read | rejected |
| Gated DeltaNet decode kernel | 0.44% of decode time | not where the time goes |
| Stacking MTP with a draft model | `common_memory` owns one `ctx_dft` | structurally blocked |
| Wider `ncols1` | not independent — derived as `64/ncols2` | no effect |

---

## Current focus

Long-context throughput on a single RTX 5090: holding a full 262,144-token context with `turbo4` KV while keeping decode and prefill as close to the hardware ceiling as possible.

- **Decode is at the practical ceiling.** After the V gather fix the KV read costs 6.00 ms at `d131072`, against `q8_0`'s 6.01 ms at twice the VRAM and `f16`'s 5.63 ms at 3.8×. The remaining 6% would have to come from the K dot, which already uses `__dp4a` with a byte-permute LUT.
- **Prefill attention has headroom, but not from tuning.** The MMA config is swept out — `ncols2` and `nbatch_fa` won, `nthreads`, `occupancy` and `ncols1` all lose. Attention is 77% of prefill at depth at ~101 TFLOPS. Further gain needs kernel work.
- **Wide-Q native turbo reads.** Prefill still uses F16 conversion, because it amortises across many Q tiles. Measured: an `f16` cache prefills only ~3% faster, so the conversion is close to free and this is not a promising lever.
- **turbot prefill.** The tiered cache still prefills 4-8% slower than turbo5p at depth. Two-token verify batches now run the `<4,8>` instance (+3.6% / +4.8% at 131K / 245K).
- **Four connections.** Deployed 2026-09-25: 4-stream steps 6.6% faster and 2.0 GB less VRAM (see [the status note](#using-the-fork-with-any-model)). Left: the small-batch matmul loses to MMQ above 1024 weight rows (shared-memory activations, fewer registers, deeper prefetch), and a many-prompt acceptance A/B at 2 and 4 streams (SMALLB on/off, `-ub 512/1024`).
- **turbot on other models.** turbot now takes six KV shapes, with an automatic plan for models that do not match the built-in one (see [turbot on other models](#turbot-on-other-models)). Gated on 2026-09-22: 4 × 256 passed and is the only shape with an automatic plan by default; 2 × 256 failed on Ornith-1.5-35B and needs `LLAMA_TURBOT_AUTO_PLAN=all`; speed is not measured yet. The automatic plan is uncalibrated, so a calibrated per-model plan still needs a calibration run. It can then ship as a verified sidecar.

---

## Credits

This fork stands on work by several people. Attribution follows the commit history.

**Upstream** — [llama.cpp](https://github.com/ggml-org/llama.cpp), Georgi Gerganov and contributors. The base this is forked from, currently `b11093` (`fb34fc262`, merged 2026-09-21; the benchmarks above were measured on `b10655`).

**The TurboQuant formats** — original CUDA port by **Gabe Ortiz** (March 2026): `turbo2_0`/`turbo3_0`/`turbo4_0`, the Walsh-Hadamard rotation, InnerQ per-channel equalization, and the type-id allocation. Method paper: [arXiv 2504.19874](https://arxiv.org/abs/2504.19874) (ICLR 2026).

**TriAttention** — KV-cache pruning by **atomicmilkshake** (April 2026), on the `feature/triattention` branch. Method paper: [arXiv 2604.04921](https://arxiv.org/abs/2604.04921).

**The long-context performance work** — [@sirxsniper](https://github.com/sirxsniper). Everything documented above: native `turbo4` reads at depth, the MMA shared-tile loader, the K and V byte-permute centroid gathers, coalesced dequant stores, GQA packing by exact divisor, the `nbatch_fa` retune, the speculative prefill tail, memory-fit estimation for unmeasurable drafters, GDN decode-shape test coverage, the Windows CUDA build recipe, and upstream merge and conflict resolution across `b8650` to `b11093`.

<sub><a href="CREDITS.md">CREDITS.md</a> holds the per-file inventory, with authorship derived from the commit history.</sub>

<div align="center">
<sub>MIT licensed, as is upstream llama.cpp.</sub>
</div>
