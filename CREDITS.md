# Credits

Attribution for work in this fork that exists nowhere upstream.

Everything below is derived from the commit history — file creation dates and
`git log` per path — rather than asserted. Where a file is shared with upstream,
attribution covers only the turbo-specific additions to it, not the file itself.

> An earlier version of this document stated that everything listed here was
> authored by @sirxsniper unless explicitly noted. That was wrong. The CPU
> TurboQuant implementation, the original CUDA port, and TriAttention were each
> written by someone else, and are credited to them below.

---

## Authorship map

| Component | Author | First landed |
|:--|:--|:--|
| CPU TurboQuant implementation | **TheTom** | March 2026 |
| CUDA port of the turbo formats | **Gabe Ortiz** | March 2026 |
| TriAttention KV-cache pruning | **atomicmilkshake** | April 2026 |
| InnerQ cross-DLL ABI | **@sirxsniper** | May 2026 |
| Long-context performance work | **@sirxsniper** | May & August 2026 |
| Everything else | upstream llama.cpp | — |

**Method papers** are the algorithms, not the code:
TurboQuant — [arXiv 2504.19874](https://arxiv.org/abs/2504.19874) (ICLR 2026).
TriAttention — [arXiv 2604.04921](https://arxiv.org/abs/2604.04921).

---

## Prior work this fork builds on

### CPU TurboQuant — TheTom, March 2026

`ggml/src/ggml-turbo-quant.c` — the reference CPU implementation of the turbo
formats: quantize/dequantize for `turbo2_0`, `turbo3_0`, `turbo4_0`, the
Walsh-Hadamard transform, and centroid selection. Later touched by Gabe Ortiz,
Sean, Tom turney and @sirxsniper.

### CUDA port — Gabe Ortiz, March 2026

Created:

* `ggml/src/ggml-cuda/turbo-quant.cuh` — device-side format definitions, the
  4-bit and 3-bit centroid tables, `turbo_nearest_centroid_*`, block layout.
* `ggml/src/ggml-cuda/turbo-wht.cu` — the CUDA Walsh-Hadamard kernels.

Turbo support threaded into the shared CUDA files (`set-rows.cu`,
`fattn-common.cuh`, `fattn-vec.cuh`), the GGML type-id allocation for
`GGML_TYPE_TURBO2_0` / `TURBO3_0` / `TURBO4_0`, InnerQ per-channel
equalization, the turbo2 64-group fallback, and 64-element WHT groups.

### TriAttention — atomicmilkshake, April 2026

`src/llama-triattention.cpp`, `src/llama-triattention.h`,
`ggml/src/ggml-cuda/triattention-score.cu`, `triattention-score.cuh`, plus the
`--triattention-*` flags in `common/arg.cpp` and the pruning hooks in
`src/llama-kv-cache.cpp`. Lives on the `feature/triattention` branch.

---

## @sirxsniper

### InnerQ cross-DLL ABI — May 2026

Both files created here:

* `ggml/include/ggml-turbo-innerq.h`
* `ggml/src/ggml-turbo-innerq.cpp`

A stable C ABI so the turbo InnerQ state can cross the `ggml-base` /
`ggml-cuda` DLL boundary on Windows, where the CUDA backend is a separate
module and cannot reach into the base library's statics.

Accompanying MSVC build compatibility work: `M_PI` definition, `dllexport`
annotation for the cross-DLL turbo symbols.

### May 2026 cycle

Brought the fork from upstream `b8650` to `b9033` and reworked the turbo K/V
flash-attention path — turbo K/V FA dispatch, the Walsh-Hadamard barrier split,
the `KQ_max` scale skip, and alignment fixes. Per-commit detail with rationale
and perf numbers in [docs/CHANGES-2026-05.md](docs/CHANGES-2026-05.md).

Also: the chat parser falling back to raw content on malformed tool-call XML.

### August 2026 — long-context performance

The work the README documents. Base moved `b9033` → `b10655`.

**Decode at depth**

* Removed the turbo-specific routing that sent decode past 4096 tokens to the
  MMA kernel, which cannot read a quantized cache and so rebuilt the entire
  cache as F16 once per layer, per token. **+45.4% decode at 245K.**
* Replaced the `turbo4` K centroid lookup — three dependent selects, called
  four times per four KV elements — with a `PRMT` byte-permute LUT
  (`turbo4_int8_lut::gather4`), verified bit-exact over all 65,536 inputs.
* Replaced the `turbo4` V centroid broadcast — two `__shfl_sync` per element,
  16 warp-serializing shuffles per 8-element call — with the same byte-permute
  gather. **+7.9% decode at 131K, +7.2% at 245K.**
* Coalesced the `turbo4` dequant stores: one wide store per lane instead of two
  4-byte stores discarding half of every 32-byte sector.
* A dequantizing tile loader so the MMA kernel reads `turbo4` natively for
  narrow Q. **+11–15% speculative decode.**
* Fixed the pathological `turbo4` → F16 conversion (divergent constant-memory
  lookup) and the `ne == 4` V dequant serialisation.

**Prefill at depth**

* GQA packing selected by exact divisor rather than rounded up to the next
  power of two — the kernel had been computing 8 heads' work for 6 real heads.
  **+17% prefill at depth.**
* `nbatch_fa` raised 32 → 64 for the 256/256 `ncols=64` config.
* `SPEC_PREFILL_TAIL`: skip drafter forward passes over prompt its sliding
  window will evict before it can draft from it. **+10% prefill**, acceptance
  unchanged.

**Correctness, robustness, tooling**

* `common/fit.cpp` — estimate an unmeasurable extra model from its weight file
  instead of counting it as free.
* Reject `draft-mtp` combined with a draft model with a stated reason rather
  than an opaque failure.
* `test-backend-ops` — turbo2/3/4 FA coverage at the deployed geometry
  (D=256, GQA 6:1) with f16/q8_0 baselines, prefill-scale `MUL_MAT`, and
  decode-shape cases for `GATED_DELTA_NET`. The D=256 turbo paths were
  previously untested.
* Grammar sampler: stop rescanning the full vocab on every sampling rejection.
* NVFP4 quantize mapping (the enum existed, the mapping did not).

**Qwen3.8-Flash-Next (`qwen4exp`)**

* Made `turbo4` KV correct on the architecture, and fixed the context-decode
  collapse (4.2 → 22.4 tok/s at 80K).
* Indexer cache falls back to `q8_0` rather than F16 — half the VRAM, same speed.
* Ported two graph optimisations from upstream's merged #27742.

**Negative results, recorded deliberately**

Blackwell MMQ tile configs for K-quants; the `I=256` tiles rejected for
corrupting dense matmul; `nthreads=512` recorded as AMD-only; the sm_120 MMVQ
table recorded as measured-neutral; GDN lane-contiguous rows with `float4`
loads/stores recorded as correct but neutral.

### Upstream merges and build

Conflict resolution across `b8650` → `b10655`, and the verified-working Windows
CUDA build recipe in [docs/BUILD-WINDOWS.md](docs/BUILD-WINDOWS.md).

---

## Upstream

Everything not listed above is [llama.cpp](https://github.com/ggml-org/llama.cpp)
by Georgi Gerganov and contributors. The fork tracks master and inherits
everything mainline does.

The shared CUDA files this fork modifies — `fattn-common.cuh`, `fattn-vec.cuh`,
`fattn-mma-f16.cuh`, `set-rows.cu`, `convert.cu` — are upstream files with
turbo-specific additions layered on. Attribution above covers those additions
only.
