# TurboQuant

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/github-sirxsniper%2Fllama--cpp--turboquant-blue?logo=github)](https://github.com/sirxsniper/llama-cpp-turboquant)

**A full 262,144-token context on a single 32 GB GPU, at speed.**

A performance fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) built around one target: run a 27B model at its full 256K context on one RTX 5090 without giving up throughput or output quality. That needs two things that pull against each other — a KV cache small enough to fit, and an attention path fast enough that the small cache doesn't cost you anything to read.

### What's in it

**TurboQuant KV cache.** Custom low-bit formats (`turbo4` at 4.25 bpw, plus `turbo3` and `turbo2`) with hand-written CUDA kernels for Turing through Blackwell. A 262,144-token cache on Qwen3.8-27B costs **4.3 GiB** as `turbo4`, against 8.0 GiB as `q8_0` and 16.0 GiB as `f16` — the difference between holding a full context on a 32 GB card and not holding it.

**Native quantized attention.** Decode reads the quantized cache directly at every depth instead of materialising it as F16 first. Worth **+45% decode at full context**.

**Attention tuned for the model's real shape.** Grouped-query packing chosen by exact divisor rather than rounded up, and a re-tuned softmax batch size. Worth **+15–17% prefill at depth**.

**Speculative decoding that doesn't tax prefill.** DFlash2 block-diffusion drafting, with the drafter skipped on the part of a long prompt where it cannot pay for itself. Worth **+10% prefill** with a drafter attached.

**TriAttention.** GPU-accelerated KV-cache pruning ([arXiv 2604.04921](https://arxiv.org/abs/2604.04921)) that scores token importance from RoPE-inverted key vectors and evicts low-value tokens, for inference inside a fixed memory budget.

### Where it lands

`Qwen3.8-27B-UD-Q4_K_XL`, RTX 5090, `turbo4` KV, flash attention on, `llama-bench -r 3`:

| Context depth | Prefill (t/s) | Decode (t/s) |
|--------------:|--------------:|-------------:|
| 0 | 3582.93 | 63.97 |
| 131,072 | 1196.84 | 42.92 |
| 245,760 | 758.91 | 33.60 |

Decode at 245,760 tokens started this work at 23.14 t/s. Quality is held to the same bar as the speed: needle-in-a-haystack passes at 32K / 128K / 245K, and the old and new attention routings produce identical output on an identical 240K prompt.

## TriAttention

TriAttention keeps your KV cache within a fixed token budget by periodically scoring all cached tokens and evicting the least important ones. Scoring uses the geometric structure of RoPE-encoded key vectors — no additional model weights or fine-tuning required.

### Performance (Qwen3-8B Q4\_K\_M, RTX 3080, `-c 512`)

| Mode | Prune overhead | Generation speed |
|------|---------------|-----------------|
| No budget limit | — | 17.5 tok/s |
| CPU scoring | ~5,900 ms/event | 17.5 tok/s |
| **GPU scoring** | **~4–9 ms/event** | **75.0 tok/s** |

GPU scoring is ~1,000× faster than CPU. The 4.3× generation speedup comes from keeping the KV cache within VRAM budget (no eviction stalls, consistent flash-attention batch sizes).

### Quick start

```bash
llama-server.exe -m YourModel.gguf -c 32768 -ngl 99 --port 8080 \
  --triattention-stats model.triattention \
  --triattention-budget 4096 \
  --triattention-window 256 \
  --triattention-log
```

A `.triattention` calibration file is required. Generate one from a representative text corpus:

```bash
llama-cli.exe -m YourModel.gguf -ngl 99 \
  --triattention-calibrate corpus.txt \
  --triattention-calibrate-out model.triattention
```

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--triattention-stats <file>` | *(none)* | Calibration file — **required to enable TriAttention** |
| `--triattention-budget <n>` | `512` | Maximum KV tokens to retain after each prune |
| `--triattention-window <n>` | `64` | Most-recent N tokens always protected from eviction |
| `--triattention-trigger <mode>` | `slack` | When to prune: `slack` (budget+window), `interval`, `fill` |
| `--triattention-log` | off | Print a line for each prune event |
| `--triattention-no-protect-prefill` | off | Allow evicting prompt (prefill) tokens |

### How it works

1. When occupied KV cells exceed `budget + window` (SLACK mode), a prune is triggered
2. The most recent `window` positions and all prefix/prompt tokens are protected
3. For each sampled `(layer, head)` pair, key vectors are read from the KV cache, RoPE rotation is inverted, and a geometric offset score is computed on the GPU
4. The top-`budget` tokens by importance score are kept; the rest are evicted
5. Position gaps left by evicted tokens are harmless — RoPE handles non-contiguous positions natively

---

## TurboQuant

TurboQuant provides three custom quantization formats that outperform standard GGUF quants at equivalent bit widths:

| Format | Bits/weight | Notes |
|--------|------------|-------|
| `turbo4_0` | 4.25 | 16 Lloyd-Max centroids, nibble packed, WHT pre-rotation |
| `turbo3_0` | ~3.0 | Sub-byte with Hadamard pre-rotation |
| `turbo2_0` | ~2.0 | Aggressive compression with WHT-space centroids |

All formats have CUDA kernels optimised for Turing+ (SM75), Ampere (SM80/86), Ada (SM89), and Blackwell (SM120/121).

For implementation details (type IDs, file map, the InnerQ cross-DLL handshake, FA-vec dispatch, and how to add a new turbo type) see [docs/TURBOQUANT-INTERNALS.md](docs/TURBOQUANT-INTERNALS.md).

### KV cache at depth

The primary use of the turbo formats here is the **KV cache**, where they cut VRAM enough to hold a full 256K context on a single 32 GB card. At 262144 tokens on Qwen3.8-27B the cache costs 4.3 GiB as `turbo4` against 8.0 GiB as `q8_0` and 16.0 GiB as `f16`.

Decode reads the quantized cache **natively at every context depth**. This matters more than it sounds: the flash-attention MMA kernel cannot read a quantized cache, so anything routed to it must first materialise the entire cache as F16 — once per layer, per token. Keeping decode on the path that reads `turbo4` directly is worth up to **+45% at full context** (see [Recent changes](#recent-changes)).

Escape hatches for A/B testing:

| Variable | Effect |
|----------|--------|
| `TURBO_FA_MMA=1` | restore the old depth-based MMA routing |
| `TURBO_MMA_NATIVE=0` | disable native turbo reads in the MMA tile loader |
| `TURBO_IDX_INHERIT=1` | let the sparse-attention indexer cache inherit the turbo type |

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
- **Wide-Q native turbo reads.** The MMA tile loader reads `turbo4` natively for narrow Q only; prefill still uses the F16 conversion, because that conversion is amortised across many Q tiles and wins there. Measured directly: an `f16` cache prefills only ~3% faster than `turbo4` at 131K, so the conversion is close to free and this is not a promising lever.
- **Stacking MTP with a draft model.** Blocked structurally rather than by policy: `common_memory` owns a single `ctx_dft` and mirrors sequence operations to it, so two drafters cannot coexist. DFlash2 alone already yields more tokens per step (3.05) than MTP alone (2.55), so the payoff would be small.

## Recent changes

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

### May 2026

The May 2026 update cycle (commits `ccdce708f` to `fd0a94a4f`) brought this fork from upstream `b8650` to `b9033` and reworked the turbo K/V flash-attention path. See **[docs/CHANGES-2026-05.md](docs/CHANGES-2026-05.md)** for the full per-commit changelog with rationale and perf numbers.

Reference TG speeds on RTX 5090, `Qwen3.6-27B-UD-Q6_K_XL`, turbo4 KV, FA on:

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
| Stacking MTP with a draft model | `common_memory` owns one `ctx_dft` | structurally blocked |

One positive result worth recording because it is easy to assume it is already handled: **CUDA graphs are load-bearing at every depth**, not just for short contexts.

| Context depth | Graphs on | Graphs off | |
|--------------:|----------:|-----------:|---|
| 0 | 66.14 | 46.34 | +42.7% |
| 131,072 | 45.77 | 33.48 | +36.7% |

Chunked Gated DeltaNet disables graph capture, but only during prefill — decode uses the sequential path and keeps them.

## Credits

For a per-file inventory of every function, kernel, and modification authored on this fork (TurboQuant CUDA/CPU/Metal integration, KV-cache wiring, TriAttention pruning, the May 2026 perf rework, and build infrastructure), see **[CREDITS.md](CREDITS.md)**.

Short version:

- [llama.cpp](https://github.com/ggml-org/llama.cpp), Georgi Gerganov and contributors. Upstream base.
- TurboQuant method paper, [arXiv 2504.19874](https://arxiv.org/abs/2504.19874) (ICLR 2026). The algorithm.
- TriAttention method paper, [arXiv 2604.04921](https://arxiv.org/abs/2604.04921). The algorithm.
- [@sirxsniper](https://github.com/sirxsniper): full GPU integration of TurboQuant (CUDA + CPU + Metal), TriAttention KV-cache pruning, all turbo CUDA kernels (`k_set_rows_turbo*`, `k_turbo_wht_*`, `vec_dot_fattn_vec_KQ_turbo*`, `dequantize_V_turbo*`), the public InnerQ cross-DLL ABI, TURBO type id allocation, KV cache wiring, the May 2026 perf rework (turbo K/V FA dispatch, Walsh-Hadamard barrier split, KQ_max scale skip, alignment fixes), the August 2026 long-context rework (native turbo4 reads at depth, PRMT centroid gather, coalesced dequant stores, GQA packing by exact divisor, `nbatch_fa` retune, speculative prefill tail), MSVC build compatibility, chat-parser robustness fix, the verified-working CMake build recipe, and conflict-resolution work across upstream merges b8650 to b10655.
