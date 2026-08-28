<div align="center">

# TurboQuant

### A full 262,144-token context on a single 32 GB GPU &mdash; at speed.

[![License: MIT](https://img.shields.io/badge/License-MIT-1f6feb?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-13.1-76b900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-downloads)
[![Blackwell](https://img.shields.io/badge/SM-75%20%E2%86%92%20121-8957e5?style=for-the-badge)](#)
[![KV cache](https://img.shields.io/badge/KV%20cache-4.25%20bpw-e3b341?style=for-the-badge)](#the-turboquant-kv-cache)

</div>

---

A performance fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) built around a single target: run a 27B model at its **full 256K context on one RTX 5090**, without giving up throughput or output quality.

That target pulls in two directions at once. The KV cache has to be small enough to fit in 32 GB alongside the weights &mdash; and the attention path has to be fast enough that the small cache costs nothing to read. Most of the work here is in closing the gap between those two.

<div align="center">

### Where it lands

`Qwen3.8-27B-UD-Q4_K_XL` &middot; RTX 5090 &middot; `turbo4` KV &middot; flash attention &middot; `llama-bench -r 3`

*All six figures from a single run on the shipped build, stock defaults.*

| Context depth | Prefill | Decode | KV cache size |
|:--------------|--------:|-------:|--------------:|
| **0** | 3722.70 t/s | 66.57 t/s | &mdash; |
| **131,072** | 1242.29 t/s | 47.03 t/s | 2.2 GiB |
| **245,760** | 760.75 t/s | 36.70 t/s | 4.1 GiB |
| **262,144** | &mdash; | &mdash; | **4.3 GiB** |

</div>

Decode at 245,760 tokens started this work at **23.14 t/s**. The same cache as `f16` would cost **16.0 GiB**, which does not fit next to the weights on a 32 GB card at all.

Quality is held to the same bar as the speed: needle-in-a-haystack passes at 32K / 128K / 245K, and the old and new attention routings produce byte-identical output on an identical 240K prompt.

---

## What's in it

<table>
<tr><td width="50%" valign="top">

### The TurboQuant KV cache

Custom low-bit formats &mdash; `turbo4` at **4.25 bpw**, plus `turbo3` and `turbo2` &mdash; with hand-written CUDA kernels for Turing through Blackwell.

A 262,144-token cache costs **4.3 GiB** against 8.0 GiB as `q8_0` and 16.0 GiB as `f16`. That is the difference between holding a full context on a 32 GB card and not holding it.

</td><td width="50%" valign="top">

### Native quantized attention

Decode reads the quantized cache **directly at every depth**, instead of first materialising the whole thing as F16 once per layer per token.

**+45% decode at full context.**

</td></tr>
<tr><td width="50%" valign="top">

### Attention tuned to the real shape

Grouped-query packing chosen by exact divisor rather than rounded up to the next power of two, plus a re-tuned softmax batch size.

**+15&ndash;17% prefill at depth.**

</td><td width="50%" valign="top">

### Speculation that doesn't tax prefill

DFlash2 block-diffusion drafting, with the drafter skipped over the part of a long prompt where it can never pay for itself.

**+10% prefill** with a drafter attached.

</td></tr>
</table>

---

## Running it

The flags that matter, in the order they matter.

### Full 256K context on a 32 GB card

The configuration this fork exists for. `turbo4` KV is what makes the context fit; the DFlash2 drafter is what makes it fast.

```bash
llama-server -m Qwen3.8-27B-UD-Q4_K_XL.gguf   -c 262144 -ngl 99 -fa 1   -ctk turbo4 -ctv turbo4   --kv-unified   -md Qwen3.8-27B-DFlash2-Q4_K_M.gguf   --spec-type draft-dflash --draft-max 7   --host 0.0.0.0 --port 8080
```

### Without a drafter

Drops ~40% of decode but frees the drafter's VRAM and its prefill cost.

```bash
llama-server -m Qwen3.8-27B-UD-Q4_K_XL.gguf   -c 262144 -ngl 99 -fa 1   -ctk turbo4 -ctv turbo4 --kv-unified   --port 8080
```

### Flags that carry real weight

| Flag | Why it matters |
|------|----------------|
| `-ctk turbo4 -ctv turbo4` | 4.25 bpw KV. Without it a 262K context needs 16 GiB and will not fit. |
| `-fa 1` | Flash attention. Everything in this fork assumes it. |
| `--kv-unified` | One shared cache instead of per-sequence. Required to fit at max context. |
| `--spec-type draft-dflash` | Block-diffusion drafting. Yields ~3.05 tokens per step. |
| `--draft-max 7` | DFlash2 uses `block_size` 8, so 7 is its natural ceiling. |
| `-ngl 99` | All layers on the GPU. |

> **Qwen3.8 and MTP** &mdash; every Qwen3.8 GGUF ships an MTP head, but it is silently ignored unless you pass `--spec-type draft-mtp`. It cannot be combined with a draft model: `common_memory` owns a single draft context. DFlash2 yields more per step (3.05 vs 2.55), so prefer it when you have the VRAM.

### Tuning knobs

Environment variables, for A/B testing rather than daily use.

| Variable | Default | Effect |
|----------|---------|--------|
| `SPEC_PREFILL_TAIL` | `2048` | Prompt tail the drafter actually prefills. `0` disables the skip. |
| `TURBO_MMA_NATIVE` | `1` | Native turbo reads in the MMA tile loader. `0` restores F16 conversion. |
| `TURBO_MMA_NATIVE_MAXQ` | `32` | Q width below which native reads are used. |
| `TURBO_FA_MMA` | off | `1` restores the old depth-based MMA routing. |
| `TURBO_IDX_INHERIT` | off | `1` lets the sparse-attention indexer cache inherit the turbo type. |

---

## What changed, and what it bought

### August 2026 — turbo4 KV at depth

Upstream base `b10655`. Decode with a `turbo4` KV cache was being routed away from the kernel that can read it: a turbo-specific rule sent decode to MMA past 4096 tokens, and MMA requires an F16 copy of the whole cache, rebuilt once per layer per token. `q8_0` and `f16` never had that rule, which is why `q8_0` measured *faster* than `turbo4` at depth despite reading twice the bytes.

Removing the rule required first making the native path competitive:

- the centroid lookup in the FA-vec `turbo4` dot product was a chain of three dependent selects called four times per four KV elements; replaced with two hardware byte-permutes (`PRMT`) plus a per-byte blend, verified bit-exact over all 65,536 possible inputs
- the `turbo4` dequant kernels wrote four consecutive elements as two 4-byte stores, discarding half of every 32-byte memory sector; now one wide store per lane
- a dequantizing tile loader lets the MMA kernel read `turbo4` directly for narrow Q, so speculative verification no longer pays the F16 materialisation either

`llama-bench` tg64, `Qwen3.8-27B-UD-Q4_K_XL`, RTX 5090, turbo4 KV, r=3:

| Context depth | Before | After | |
|---------------|--------|-------|---|
| 0 | 65.58 | 66.41 | |
| 65,536 | 45.10 | **53.82** | +19.3% |
| 131,072 | 33.61 | **44.39** | +32.1% |
| 245,760 | 23.14 | **33.64** | **+45.4%** |

For reference `q8_0`, which never had the bug, measures 35.33 at 245,760 — `turbo4` now sits just under it at half the KV VRAM. Speculative decode gains a further 11–15%. Prefill is unchanged.

Verified with `test-backend-ops -o FLASH_ATTN_EXT` on both routings, and needle-in-a-haystack at 32K / 128K / 245K; old and new routing produce identical output on an identical 240K prompt.

### August 2026 — the V centroid gather

Decode was reading the KV cache at a fraction of the rate the hardware allows, and the reason was not the cache.

Isolating the KV read by subtracting the `d0` decode time from the `d131072` decode time gives the cost of the cache traffic alone:

| KV type | bytes read | added time | effective bandwidth |
|---------|-----------:|-----------:|--------------------:|
| `turbo4` | 2.42 GB | 7.86 ms | 308 GB/s |
| `q8_0` | 4.85 GB | 6.01 ms | 807 GB/s |
| `f16` | 9.13 GB | 5.63 ms | 1621 GB/s |

`f16` moved **3.8x the bytes in less time**. Reading fewer bytes and taking longer is the signature of a compute-bound gather, not a bandwidth-bound one, which pointed at the lookup rather than the cache.

The V dequant spread the 16 scaled centroids across an 8-lane sub-group and broadcast them with `__shfl_sync` — two shuffles per element, **16 warp-serializing shuffles per 8-element call**, paid per V element per layer per token. The K dot had already been converted to `turbo4_int8_lut::gather4` (two hardware byte-permutes and a per-byte blend, four centroids per gather); `qs_word` holds the eight nibbles in exactly that layout, so it applied to V unchanged. Two gathers now replace sixteen shuffles, with no warp synchronization.

`llama-bench` tg64, r=3:

| Context depth | Before | After | |
|---------------|--------|-------|---|
| 0 | 65.29 | 65.29 | unchanged, no KV to read |
| 131,072 | 43.51 | **46.93** | +7.9% |
| 245,760 | 34.05 | **36.50** | +7.2% |

That closes almost the entire gap. The KV read now costs 6.00 ms against `q8_0`'s 6.01 ms at half the VRAM, and sits 6% off `f16` while using 3.8x less memory.

Precision: the V gather now reads the same int8 centroid table the K dot already used, each entry within 0.5% of its float value — far inside the error already introduced by binning to one of 16 centroids. Verified with `test-backend-ops -o FLASH_ATTN_EXT` and needle retrieval at 245K with the needle at 90% depth.

### August 2026 — prefill at depth

Two independent bugs, both found by asking why attention was running so far under the card's ceiling.

**Grouped-query packing rounded the wrong way.** `ncols2` controls how many query heads share one K/V tile read. The selection ladder rounded *up* to the next power of two, so a model with `gqa_ratio = 24/4 = 6` was assigned 8 columns and computed 8 heads' worth of work for 6 real heads — 25% of the attention math thrown away, at every depth. Choosing by exact divisor instead (6 % 2 == 0, so 2) fixed it.

**The softmax batch size was tuned for a different shape.** `nbatch_fa` sets how many KV rows are processed per softmax rescaling. Raising it 32 to 64 for the 256/256 head-dimension case cut rescaling overhead; 128 exceeds the shared-memory budget and aborts, so 64 is the ceiling.

**The drafter was prefilling prompt it could never help with.** With speculative decoding attached, the draft model prefilled the entire prompt, but drafting only ever begins at the tail. `SPEC_PREFILL_TAIL` (default 2048) skips the drafter's forward passes over everything earlier.

`llama-bench` pp512, same model and card, r=3:

| Context depth | Before | After | |
|---------------|--------|-------|---|
| 131,072 | 1023.15 | **1196.84** | +17.0% |
| 245,760 | 645.31 | **758.91** | +17.6% |

With a drafter attached, `SPEC_PREFILL_TAIL` recovers a further ~10% on top of that, and draft acceptance is unchanged (29.6% to 29.4%) — the skipped work was genuinely dead.

Verified with `test-backend-ops -o FLASH_ATTN_EXT` and needle retrieval at 245K.

### May 2026 — earlier work, for the record

The May 2026 update cycle (commits `ccdce708f` to `fd0a94a4f`) brought this fork from upstream `b8650` to `b9033` and reworked the turbo K/V flash-attention path. See **[docs/CHANGES-2026-05.md](docs/CHANGES-2026-05.md)** for the full per-commit changelog with rationale and perf numbers.

Reference decode speeds from that cycle, on a different model (`Qwen3.6-27B-UD-Q6_K_XL`) than the one benchmarked above:

| Context fill | Generation t/s |
|--------------|----------------|
| empty | 52.4 |
| 8k | 51.4 |
| 32k | 49.2 |
| 64k | 45.5 |
| 128k | 39.6 |

131k needle-in-a-haystack: 3 of 3 hits at 25, 50, 75 percent depths.

---

## Measured and ruled out

Negative results, kept because they cost real time to establish and each one closes a question that looks promising from the outside. All on `Qwen3.8-27B-UD-Q4_K_XL`, RTX 5090, `turbo4` KV, `llama-bench -r 3`.

| Idea | Result | Verdict |
|------|--------|---------|
| Wider attention blocks (`nthreads` 128 → 256) | pp512 1196.84 → 1015.88 @131K | −15%, rejected |
| More resident blocks (`occupancy` 2 → 3) | pp512 1196.84 → 1102.26 @131K | −8%, rejected |
| `f16` KV instead of `turbo4` for prefill | 1105.15 → 1142.00 @131K, same run | +3%, not worth 3.7× the VRAM |
| NVFP4 KV on Blackwell FP4 tensor cores | faster, but loses needle recall | rejected on quality |
| VEC path with 8 columns | no GQA packing → 6× cache re-read | rejected |
| Gated DeltaNet decode kernel | 0.44% of decode time | not where the time goes |
| Wider attention blocks, higher occupancy | see above | MMA config knobs exhausted |
| Stacking MTP with a draft model | `common_memory` owns one `ctx_dft` | structurally blocked |

One positive result worth recording because it is easy to assume it is already handled: **CUDA graphs are load-bearing at every depth**, not just for short contexts.

| Context depth | Graphs on | Graphs off | |
|--------------:|----------:|-----------:|---|
| 0 | 66.14 | 46.34 | +42.7% |
| 131,072 | 45.77 | 33.48 | +36.7% |

Chunked Gated DeltaNet disables graph capture, but only during prefill — decode uses the sequential path and keeps them.

## Current focus

Long-context throughput on a single RTX 5090: holding a full **262,144-token** context with `turbo4` KV while keeping decode and prefill as close to the hardware ceiling as possible. Work is measured with `llama-bench -d <depth>` (repeated, with stddev) rather than through the server, because with a speculative drafter attached the decode rate tracks draft acceptance, which varies with prompt text and swamps the effects being measured.

Where prefill time goes at `d245760` (675 ms per 512-token batch), on a model with 17 full-attention layers and 48 Gated DeltaNet layers:

| Component | Time | Share |
|-----------|-----:|------:|
| Attention (17 layers, 5.3e13 flops) | ~521 ms | 77% |
| FFN | ~143 ms | 21% |
| Gated DeltaNet (48 layers) | ~10 ms | 1.5% |

Attention runs at ~101 TFLOPS there. That is roughly half of the card's realistic ceiling for FP16 tensor ops with FP32 accumulation, which is normal-to-good for flash attention but leaves room.

Open items:

- **Attention efficiency at depth.** The MMA config knobs are now exhausted: `ncols2` by exact GQA divisor and `nbatch_fa` 32 to 64 both won, while `nthreads` 128 to 256 (−15%), `occupancy` 2 to 3 (−8%), and wider `ncols1` all lose. Further gain needs kernel work, not tuning.
- **Decode is at the practical ceiling.** After the V centroid gather fix the KV read costs 6.00 ms at `d131072`, against `q8_0`'s 6.01 ms at twice the VRAM and `f16`'s 5.63 ms at 3.8x. The remaining 6% would have to come from the K dot, which already uses `__dp4a` with a byte-permute LUT.
- **Wide-Q native turbo reads.** The MMA tile loader reads `turbo4` natively for narrow Q only; prefill still uses the F16 conversion, because that conversion is amortised across many Q tiles and wins there. Measured directly: an `f16` cache prefills only ~3% faster than `turbo4` at 131K, so the conversion is close to free and this is not a promising lever.
- **Stacking MTP with a draft model.** Blocked structurally rather than by policy: `common_memory` owns a single `ctx_dft` and mirrors sequence operations to it, so two drafters cannot coexist. DFlash2 alone already yields more tokens per step (3.05) than MTP alone (2.55), so the payoff would be small.

## The turbo formats

| Format | Bits/weight | Construction |
|--------|------------:|--------------|
| `turbo4_0` | **4.25** | 16 Lloyd-Max centroids, nibble packed, WHT pre-rotation |
| `turbo3_0` | ~3.0 | sub-byte, Hadamard pre-rotation |
| `turbo2_0` | ~2.0 | WHT-space centroids |

CUDA kernels cover Turing (SM75), Ampere (SM80/86), Ada (SM89) and Blackwell (SM120/121). `turbo4` is the one that matters here: it is what makes a 262,144-token cache fit in 4.3 GiB.

Implementation details — type IDs, file map, the InnerQ cross-DLL handshake, FA-vec dispatch, and how to add a new turbo type — are in [docs/TURBOQUANT-INTERNALS.md](docs/TURBOQUANT-INTERNALS.md).

---

## Building from source

### Requirements

- Windows 10/11 or Linux
- CUDA Toolkit 13.1 (13.2 untested, see notes in build guide)
- Visual Studio 2022 BuildTools with C++ workload + Win 11 SDK (Windows) or GCC 11+ (Linux)
- CMake 3.27+, Ninja 1.11+

### Windows (CUDA)

See **[docs/BUILD-WINDOWS.md](docs/BUILD-WINDOWS.md)** for the verified-working recipe with toolchain pins (VS 2022 BuildTools, CUDA 13.1, CMake, Ninja), the full CMake invocation, and known gotchas (CUDA component selection, MSVC version, ninja install, power-plan tuning).

### Linux (CUDA)

```bash
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;120;121" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --target llama-server -j$(nproc)
```

---

## Branches

| Branch | Description |
|--------|-------------|
| `turbo4-depth-fix-2026-08-28` | **Latest** — turbo4 KV at depth, native MMA tile loading |
| `feature/triattention` | TurboQuant + TriAttention |
| `feature/turboquant-kv-cache` | TurboQuant base (pre-TriAttention) |
| `master` | Upstream llama.cpp base |

---

## Credits

This fork stands on three separate bodies of work. Attribution below follows the commit history.

**Upstream** &mdash; [llama.cpp](https://github.com/ggml-org/llama.cpp), Georgi Gerganov and contributors. The base this is forked from, currently `b10655`.

**The TurboQuant formats** &mdash; original CUDA port by Gabe Ortiz (March 2026): `turbo2_0`/`turbo3_0`/`turbo4_0`, the Walsh-Hadamard rotation, InnerQ per-channel equalization, and the type-id allocation. Method paper: [arXiv 2504.19874](https://arxiv.org/abs/2504.19874) (ICLR 2026).

**TriAttention** &mdash; KV-cache pruning by atomicmilkshake (April 2026). Method paper: [arXiv 2604.04921](https://arxiv.org/abs/2604.04921).

**The long-context performance work** &mdash; [@sirxsniper](https://github.com/sirxsniper). Everything this README documents above:

- native `turbo4` reads at depth (+45% decode at 245K) and the turbo4 MMA shared-tile loader for narrow Q
- the V centroid gather via byte-permute, replacing 16 warp shuffles per call (+7.9% decode at 131K)
- the PRMT centroid gather and coalesced dequant stores on the K path
- GQA packing by exact divisor (+17% prefill at depth) and the `nbatch_fa` retune
- skipping drafter work on prompt its sliding window will evict (+10% prefill)
- memory-fit estimation for unmeasurable drafters, GDN decode-shape test coverage
- upstream merge and conflict resolution across `b8650` to `b10655`, and the verified Windows CUDA build recipe

[CREDITS.md](CREDITS.md) carries a per-file inventory, but note that it predates this audit and over-attributes the original TurboQuant and TriAttention work.
