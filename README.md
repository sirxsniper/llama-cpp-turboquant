<div align="center">

# TurboQuant

**A full 262,144-token context on a single 32 GB GPU — at speed.**

[![License](https://img.shields.io/badge/License-MIT-1f6feb?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-13.1-76b900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-downloads)
[![Arch](https://img.shields.io/badge/SM-75%20→%20121-8957e5?style=for-the-badge)](#build)
[![KV](https://img.shields.io/badge/KV%20cache-4.25%20bpw-e3b341?style=for-the-badge)](#the-turbo-formats)
[![Base](https://img.shields.io/badge/llama.cpp-b10655-6e7681?style=for-the-badge)](https://github.com/ggml-org/llama.cpp)

<br>

### 262,144 tokens · 4.52 GiB of KV cache · 43.6 tok/s at 245K depth

</div>

---

A performance fork of llama.cpp built around a single target: run a 27B model at its **full 256K context on one RTX 5090**, without giving up throughput or output quality.

That target pulls in two directions at once. The KV cache has to be small enough to fit in 32 GB next to the weights — and the attention path has to be fast enough that a small cache costs nothing to read. Almost all the work here is in closing the gap between those two, and almost all of it came from measuring rather than guessing.

<div align="center">

| | Prefill | Decode | KV cache |
|:--|--:|--:|--:|
| **Empty context** | 3722.70 t/s | 66.57 t/s | — |
| **131,072 tokens** | 1242.29 t/s | 52.13 t/s | 2.26 GiB |
| **245,760 tokens** | 760.75 t/s | 43.55 t/s | 4.24 GiB |
| **262,144 tokens** | — | — | **4.52 GiB** |

<sub>All six throughput figures from a single <code>llama-bench -r 3</code> run on the shipped build, stock defaults.</sub>

</div>

> Decode at 245,760 tokens began this work at **23.14 t/s**. The same cache as `f16` would need **17.00 GiB**, which does not fit beside the weights on a 32 GB card at all — measured, not estimated: llama.cpp refuses the allocation.

---

## Contents

| | |
|:--|:--|
| [Highlights](#highlights) | what this fork adds, and what each part bought |
| [Test bench](#test-bench) | the exact hardware and software behind every number |
| [Benchmarks](#benchmarks) | full results, with methodology |
| [Build](#build) | Windows and Linux, from source |
| [Configuration](#configuration) | launch recipes, flag reference, tuning knobs |
| [How it works](#how-it-works) | the turbo formats and the attention path |
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
| CPU | AMD Ryzen 9 9950X3D, 16C/32T |
| RAM | 93 GB |

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
| File | `Qwen3.8-27B-UD-Q4_K_XL.gguf`, 16.69 GiB, 27.32 B params |
| Architecture | 65 blocks — **17 full-attention**, 48 Gated DeltaNet |
| Attention shape | 24 query heads / 4 KV heads (GQA 6:1), head dim 256 |
| Drafter | `Qwen3.8-27B-DFlash2-Q4_K_M.gguf`, 1.06 GiB |

> **The quant label is misleading, and it is worth knowing.** This file reports as `Q4_K - Small` because that is the nominal `general.file_type` stamped into the GGUF. The actual tensor mix is **68.9% Q5_K**, 21.2% IQ4_XS, 5.2% Q6_K, 4.7% Q4_K — roughly **5.2 bpw**, which is why it is 16.69 GiB rather than the ~14.6 GiB a true Q4_K_S would be. These benchmarks are on a 5-bit-class model.

---

## Benchmarks

### Throughput vs context depth

`llama-bench -r 3`, stock defaults, single run.

| Depth | Prefill (pp512) | Decode (tg64) |
|------:|----------------:|--------------:|
| 0 | **3722.70** ± 20.83 | **66.57** ± 0.17 |
| 131,072 | **1242.29** ± 3.63 | **47.03** ± 0.41 |
| 245,760 | **760.75** ± 1.26 | **36.70** ± 0.19 |

### KV cache footprint

| KV type | bits/weight | @131,072 | @262,144 | Fits on 32 GB? |
|:--|--:|--:|--:|:--|
| **`turbo4`** | **4.25** | **2.26 GiB** | **4.52 GiB** | **yes, with room** |
| `q8_0` | 8.5 | 4.52 GiB | 9.03 GiB | marginal |
| `f16` | 16.0 | 8.50 GiB | 17.00 GiB | **no** — allocation refused |

<sub>Sizes are exact for the geometry above: 17 attention layers × 4 KV heads × 256 dims × 2 (K and V). <code>turbo4</code> packs 128 elements into a 68-byte block.</sub>

### Where decode time actually goes

Subtracting the empty-context decode time from the decode time at depth isolates the cost of the KV traffic alone. Dividing by the bytes each type actually reads converts them all to one comparable number.

| KV type | bytes read @131K | added time | effective bandwidth |
|:--|--:|--:|--:|
| `turbo4` *(before fix)* | 2.42 GB | 7.86 ms | 308 GB/s |
| `q8_0` | 4.85 GB | 6.01 ms | 807 GB/s |
| `f16` | 9.13 GB | 5.63 ms | 1621 GB/s |
| **`turbo4` *(shipped)*** | **2.42 GB** | **6.00 ms** | **403 GB/s** |
| `turbo4` *(dequant ablated)* | 2.42 GB | 4.09 ms | 592 GB/s |

`f16` moved **3.8× the bytes in less time**. Reading fewer bytes while taking longer is the signature of a compute-bound gather, not a bandwidth-bound one — which is what localised the problem to the centroid lookup rather than the cache.

> **Read that effective-bandwidth column carefully.** It divides by bytes moved, so a format that reads very little is punished by it even when it is fast. In wall-clock — which is what a token costs — `turbo4` reads its cache in 6.00 ms against `f16`'s 5.63 ms while using **3.8× less memory**, and against `q8_0`'s 6.01 ms at **half** the VRAM. Chasing `f16`'s GB/s number is chasing an artefact of the metric; the real target is wall-clock, and there the gap is 6%.

### What is left, and why

An ablation that deletes every dequant gather but keeps all loads intact measured **51.76** at `d131072` against 47.03 at the time. So roughly a third of the KV read cost is arithmetic, and even with that arithmetic entirely free the kernel would land near 4.09 ms — not the ~3 ms that matching `q8_0`'s GB/s would require. Cheaper arithmetic alone cannot get there.

The reason the arithmetic costs so much is that it is done six times. FA-vec launches with `ncols2 = 1`, so one block serves exactly one query head:

```
const int head = blockIdx.z - sequence*ne02;    // ne02 = 24 query heads
K += nb13*sequence + nb12*(head / gqa_ratio);   // gqa_ratio = 6
```

With 24 query heads over 4 KV heads, **six blocks read and dequantize the same cache**. DRAM is spared — L2 absorbs most of the re-reads, which is why measured throughput is not 6× worse — but every block dequantizes independently, so the gather is paid six times over. That is the amplifier that turns turbo4's slightly-more-expensive lookup into a 2× gap against `q8_0` per byte.

Packing GQA heads into one block, the way the MMA kernel already does, is the remaining structural fix. It is bounded by registers: `VKQ[ncols][16]` half2 makes six-way packing infeasible at head dim 256, so two- or three-way is the realistic range, worth an estimated 5–12%.

Routing decode through the MMA kernel instead — which does pack GQA — was measured and rejected: **22.18 vs 46.91** at `d131072`, because at query width 1 it computes a 64-column tile for a single real column.

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
llama-server -m Qwen3.8-27B-UD-Q4_K_XL.gguf \
  -c 262144 -ngl 99 -fa 1 \
  -ctk turbo4 -ctv turbo4 \
  --kv-unified \
  -md Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --spec-type draft-dflash --draft-max 7 \
  --host 0.0.0.0 --port 8080
```

</details>

<details>
<summary><b>Without a drafter</b> — lower decode, less VRAM, no prefill cost</summary>

<br>

```bash
llama-server -m Qwen3.8-27B-UD-Q4_K_XL.gguf \
  -c 262144 -ngl 99 -fa 1 \
  -ctk turbo4 -ctv turbo4 \
  --kv-unified \
  --port 8080
```

</details>

### The exact production config

Copy-paste runnable on a 32 GB Blackwell card. This is the configuration the benchmarks above were produced with.

```bash
llama-server   -m  Qwen3.8-27B-UD-Q4_K_XL.gguf   -md Qwen3.8-27B-DFlash2-Q4_K_M.gguf   --spec-type draft-dflash --draft-max 7   -c 262144 -ngl 99 -fa 1   -ctk turbo4 -ctv turbo4   --kv-unified   -b 2048 -ub 512   -t 16 --threads-batch 16   --parallel 1   --no-mmap   --jinja --chat-template-file qwen-fixed-chat-template.jinja   --host 0.0.0.0 --port 8080 --metrics
```

<details>
<summary><b>What each block is doing</b></summary>

<br>

| Block | Why |
|:--|:--|
| `-md` + `--spec-type draft-dflash` | The 1.06 GiB DFlash2 drafter. Yields ~3.05 tokens per step. Drop both lines to save its VRAM at roughly 40% of decode. |
| `--draft-max 7` | DFlash2 trains at `block_size` 8, so 7 is its ceiling. Higher is wasted work. |
| `-c 262144 -ctk/-ctv turbo4` | The full context, at 4.52 GiB of cache. `f16` here needs 17.00 GiB and is refused. |
| `--kv-unified` | One shared cache instead of per-sequence. Needed to fit at max context. |
| `-b 2048 -ub 512` | `-ub` is the cheap VRAM lever — 1024 → 512 frees roughly 360 MiB at this context for very little throughput. Raise it if you have headroom. |
| `-t 16 --threads-batch 16` | Physical cores. Everything heavy is on the GPU; these only feed it. |
| `--no-mmap` | Load resident. On Windows, mmap lets the model page under memory pressure and decode falls off a cliff. |
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

### Flags that carry real weight

| Flag | Why it matters |
|:--|:--|
| `-ctk turbo4 -ctv turbo4` | 4.25 bpw KV. Without it a 262K context needs 17 GiB and will not fit. |
| `-fa 1` | Flash attention. Everything in this fork assumes it. |
| `--kv-unified` | One shared cache rather than per-sequence. Needed to fit at max context. |
| `--spec-type draft-dflash` | Block-diffusion drafting, ~3.05 tokens accepted per step. |
| `--draft-max 7` | DFlash2 trains at `block_size` 8, so 7 is its natural ceiling. |
| `-ngl 99` | All layers on the GPU. |

> **Qwen3.8 and MTP.** Every Qwen3.8 GGUF ships an MTP head, and it is **silently ignored** unless you pass `--spec-type draft-mtp`. It cannot be combined with a draft model — `common_memory` owns a single draft context — and this fork now rejects that pairing with an explicit reason rather than an opaque failure. DFlash2 yields more per step (3.05 vs 2.55), so prefer it when the VRAM is there.

### Tuning knobs

Environment variables, for A/B testing rather than daily use.

| Variable | Default | Effect |
|:--|:--|:--|
| `SPEC_PREFILL_TAIL` | `2048` | Prompt tail the drafter actually prefills. `0` disables the skip. |
| `TURBO_MMA_NATIVE` | `1` | Native turbo reads in the MMA tile loader. `0` restores F16 conversion. |
| `TURBO_MMA_NATIVE_MAXQ` | `32` | Q width below which native reads are used. |
| `TURBO_FA_MMA` | off | `1` restores the old depth-based MMA routing. |
| `TURBO_IDX_INHERIT` | off | `1` lets the sparse-attention indexer cache inherit the turbo type. |

---

## How it works

### The turbo formats

| Format | Bits/weight | Construction |
|:--|--:|:--|
| **`turbo4_0`** | **4.25** | 16 Lloyd-Max centroids, nibble packed, WHT pre-rotation |
| `turbo3_0` | ~3.0 | sub-byte, Hadamard pre-rotation |
| `turbo2_0` | ~2.0 | WHT-space centroids |

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

---

## Engineering log

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

---

## Credits

This fork stands on work by several people. Attribution follows the commit history.

**Upstream** — [llama.cpp](https://github.com/ggml-org/llama.cpp), Georgi Gerganov and contributors. The base this is forked from, currently `b10655`.

**The TurboQuant formats** — original CUDA port by **Gabe Ortiz** (March 2026): `turbo2_0`/`turbo3_0`/`turbo4_0`, the Walsh-Hadamard rotation, InnerQ per-channel equalization, and the type-id allocation. Method paper: [arXiv 2504.19874](https://arxiv.org/abs/2504.19874) (ICLR 2026).

**TriAttention** — KV-cache pruning by **atomicmilkshake** (April 2026), on the `feature/triattention` branch. Method paper: [arXiv 2604.04921](https://arxiv.org/abs/2604.04921).

**The long-context performance work** — [@sirxsniper](https://github.com/sirxsniper). Everything documented above: native `turbo4` reads at depth, the MMA shared-tile loader, the K and V byte-permute centroid gathers, coalesced dequant stores, GQA packing by exact divisor, the `nbatch_fa` retune, the speculative prefill tail, memory-fit estimation for unmeasurable drafters, GDN decode-shape test coverage, the Windows CUDA build recipe, and upstream merge and conflict resolution across `b8650` to `b10655`.

<sub><a href="CREDITS.md">CREDITS.md</a> holds the per-file inventory, with authorship derived from the commit history.</sub>

<div align="center">
<sub>MIT licensed, as is upstream llama.cpp.</sub>
</div>
