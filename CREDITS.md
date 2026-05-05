# Credits

This is a per-file inventory of work in this fork that exists nowhere
upstream. Use it when attribution matters or when wiring blame back to
the right person.

The shorthand is:

* **Method paper** = TurboQuant: KV cache compression via PolarQuant + QJL
  ([arXiv 2504.19874](https://arxiv.org/abs/2504.19874), ICLR 2026). The
  algorithm. Not the code.
* **@atomicmilkshake** = author of the GPU integration, KV-cache wiring,
  multi-backend port (CUDA + CPU + Metal), TriAttention pruning, and the
  full May 2026 update cycle. Everything listed below was authored by
  @atomicmilkshake unless explicitly noted.
* **Upstream llama.cpp** = anything not listed here. The fork is rebased on
  current llama.cpp master and inherits everything mainline does, plus the
  PRs noted in `docs/CHANGES-2026-05.md`.

---

## TurboQuant types

GGML enum additions in `ggml/include/ggml.h`:

* `GGML_TYPE_TURBO3_0 = 42` (3-bit KV: 2-bit PolarQuant + 1-bit QJL)
* `GGML_TYPE_TURBO4_0 = 43` (4-bit KV: 3-bit PolarQuant + 1-bit QJL)
* `GGML_TYPE_TURBO2_0 = 44` (2-bit KV: PolarQuant only)

Block layouts in `ggml/src/ggml-common.h`:

* `block_turbo2_0`, `block_turbo3_0`, `block_turbo4_0`

---

## CPU implementation

### `ggml/src/ggml-turbo-quant.c`

Reference quantize / dequantize / vec_dot for all three turbo types.
628 lines, every function below is by @atomicmilkshake:

* `quantize_row_turbo2_0_ref`, `quantize_row_turbo3_0_ref`,
  `quantize_row_turbo4_0_ref`
* `dequantize_row_turbo2_0`, `dequantize_row_turbo3_0`,
  `dequantize_row_turbo4_0`
* `quantize_turbo2_0`, `quantize_turbo3_0`, `quantize_turbo4_0` (block-level)
* `ggml_vec_dot_turbo2_0_f32`, `ggml_vec_dot_turbo3_0_f32`,
  `ggml_vec_dot_turbo4_0_f32`
* `ggml_compute_forward_turbo_wht_f32` (CPU Walsh-Hadamard reference)
* `turbo_prng_normal` (Box-Muller normal sampler for QJL projection seeding)
* Global `int turbo3_cpu_wht_group_size` (cross-DLL state for the CPU
  set_rows handler)

May 2026 fixes:

* Added `_USE_MATH_DEFINES` so MSVC exposes `M_PI` (commit `ccdce708f`)
* Tagged `turbo3_cpu_wht_group_size` as `GGML_API` so the symbol is
  exported across `ggml-base.dll` -> `ggml-cpu.dll` (commit `ccdce708f`)

### `ggml/src/ggml-cpu/ops.cpp`

* CPU `set_rows` turbo dispatch (107 lines added on top of upstream)
* `extern "C" { GGML_API int turbo3_cpu_wht_group_size; }` declaration
  fix that ties the import side to the C-linkage definition (May 2026,
  commit `ccdce708f`)

---

## CUDA implementation

### `ggml/src/ggml-cuda/turbo-quant.cuh`

CUDA centroid tables and helpers, 435 lines:

* `TURBO_CENTROIDS_2BIT[4]`, `TURBO_CENTROIDS_3BIT[8]`,
  `TURBO_CENTROIDS_4BIT[16]` (constant-memory)
* `TURBO_MID_2BIT`, `TURBO_MID_3BIT`, `TURBO_MID_4BIT` (midpoint tables
  for nearest-centroid lookup)
* `turbo_nearest_centroid_2bit`, `_3bit`, `_4bit` (device-side lookup)
* `turbo2_dequant_element`, `turbo3_dequant_element`,
  `turbo4_dequant_element` (single-element dequant)
* `turbo_rotate_forward`, `turbo_rotate_forward_64` (in-place QJL
  rotation)
* `turbo_innerq_is_active` (fast InnerQ readiness check)

May 2026 addition:

* `TURBO_CENTROIDS_4BIT_INT8[16]` plus `TURBO_INT8_4BIT_SCALE_REVERSE`
  macro. Pre-quantised int8 centroid table set up for a future
  `__dp4a` path on Blackwell. Float path remains active because
  constant-memory serialisation on divergent indices outweighs the
  hardware-instruction win at depth, but the table is in place for the
  switch when the trade-off shifts (commit `3f160d33f`).

### `ggml/src/ggml-cuda/turbo-wht.cu`

CUDA Walsh-Hadamard transform, 184 lines:

* `k_turbo_wht_f32` (main kernel, group_size=64 or 128)
* `k_turbo_wht_copy_tail` (tail-block path)
* `turbo_fwht_64`, `turbo_fwht_128` (device-side helpers)

May 2026 perf optimization (commit `3f160d33f`):

* Split `__syncthreads()` into `WHT_STAGE_WARP` (h in {1,2,4,8,16}: pairs
  swap within a single warp, `__syncwarp()` is sufficient) and
  `WHT_STAGE_BLOCK` (h in {32, 64}: cross-warp, full barrier required).
  WHT runs 128 times per generated token (64 layers x Q+V) so each
  saved barrier matters. Roughly 10x cheaper per warp-internal stage.

### `ggml/src/ggml-cuda/set-rows.cu`

CUDA quantize-on-write turbo kernels, 929 lines added on top of
upstream:

* `k_set_rows_turbo2`, `k_set_rows_turbo2_tail`
* `k_set_rows_turbo3`, `k_set_rows_turbo3_tail`
* `k_set_rows_turbo4`
* `quantize_f32_turbo2_0_block`, `quantize_f32_turbo3_0_block`,
  `quantize_f32_turbo4_0_block`
* `dequantize_turbo2_0`, `dequantize_turbo3_0`, `dequantize_turbo4_0`

May 2026 cleanup (commit `3f160d33f`):

* Renamed `block_turbo4_0::rnorm` to `pad`. The field exists only to
  keep `qs[]` 4-byte aligned for the CUDA load instructions. New name
  is honest about its purpose.

### `ggml/src/ggml-cuda/fattn-common.cuh`

Turbo K vec_dot and V dequant template specialisations, 679 lines added
on top of upstream:

* `vec_dot_fattn_vec_KQ_turbo2_0<D, nthreads>`
* `vec_dot_fattn_vec_KQ_turbo3_0<D, nthreads>`
* `vec_dot_fattn_vec_KQ_turbo4_0<D, nthreads>`
* `dequantize_V_turbo2_0`, `dequantize_V_turbo3_0`, `dequantize_V_turbo4_0`

May 2026 addition (commit `1619a8ce3` / `3f160d33f`):

* New `vec_dot_fattn_vec_KQ_turbo4_0_int`, the int8/`__dp4a` candidate path.
  The previous `_turbo4_0` was renamed `_int`; a new float-path
  `_turbo4_0` was added as the active dispatch. The int path is kept
  ready for re-evaluation on architectures where `__dp4a` divergent
  lookups are cheaper.

### `ggml/src/ggml-cuda/fattn-vec.cuh`

Main flash-attention kernel for vector-shape attention. The turbo
dispatch logic and the May 2026 perf rework live here:

* `flash_attn_ext_vec` template (the kernel itself)
* `type_K_is_turbo` / `type_V_is_turbo` constexpr predicates

May 2026 perf rework (commit `3f160d33f`, see source for the inline
`PERF:` comments):

1. `__launch_bounds__` minBlocksPerSM bumped 2 -> 4. Higher SM
   occupancy hides DRAM/L2 stalls at depth.
2. `nthreads_KQ` for turbo K halved to 16. Per-thread byte base
   becomes 4-byte aligned, so the `qs` load compiles to a single
   `LDG.E.32` instead of two `LDG.E.16`.
3. Correctness fix: `cpy_ne_KQ = D/(2*nthreads_KQ)` cap. Without it
   the load loop wrote out-of-bounds into `Q_reg` (sized 2) at
   indices 0..3 in the turbo path. Latent bug introduced when turbo
   inherited the q8 thread layout.
4. `nthreads_V` for turbo V set to 16 explicitly. Earlier "8 threads
   / V" experiment was broken (each lane held a half-pair, shfl
   pattern wrong). 16 lanes hold one centroid each, single shfl per
   element with correct semantics, V dequant throughput roughly
   doubles.
5. `V_rows_per_thread = 2*cpy_ne` unconditionally for unquantized V.
   The previous `? 4 : 2*cpy_ne` ternary was a leftover from when
   turbo had `nthreads_V=32`.
6. Sparse-V skip is now `constexpr`-disabled. The branch is fully
   dead-code-eliminated by the compiler. Quality preserved at long
   context.
7. Branch around the rescale when `KQ_max_new == KQ_max`. After
   `KQ_max` stabilises, multiplying every prior `VKQ` by 1.0f for
   thousands of remaining positions is wasted work that scales with
   depth. The new code skips the rescale (and the half2 broadcast)
   entirely on the common no-change path.

### `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-*.cu`

Cross-product instantiation files for every K/V type pair. By
@atomicmilkshake:

* `fattn-vec-instance-q8_0-turbo{2,3,4}_0.cu` (q8 K with turbo V)
* `fattn-vec-instance-turbo{2,3,4}_0-q8_0.cu` (turbo K with q8 V)
* `fattn-vec-instance-turbo{2,3,4}_0-turbo{2,3,4}_0.cu` (full
  turbo-turbo cross-product, 9 files)

12 files total, ~84 lines, mechanical but required for the FA dispatch
to find the right kernel for each `(type_K, type_V)` combination.

---

## InnerQ cross-DLL ABI (NEW in May 2026)

This is the headline build-system fix of the cycle. Before May, the
InnerQ host-side state lived in `ggml-cuda.dll`, which fails under
`GGML_BACKEND_DL=ON` because `ggml-cuda` is loaded as a runtime plugin
and its symbols aren't resolvable from `llama.dll`.

### `ggml/include/ggml-turbo-innerq.h` (NEW file)

Public C ABI header. Lives in the `ggml` include directory so it ships
with installs:

* `extern GGML_API bool g_innerq_finalized`
* `extern GGML_API float g_innerq_scale_inv_host[INNERQ_MAX_CHANNELS]`
* `GGML_API void turbo_innerq_publish(const float * scale_inv, int group_size)`
* `GGML_API bool turbo_innerq_needs_tensor_update(void)`
* `GGML_API void turbo_innerq_mark_tensor_updated(void)`

### `ggml/src/ggml-turbo-innerq.cpp` (NEW file)

Implementation, 40 lines. Linked into `ggml-base.dll`. Always
present, regardless of `GGML_BACKEND_DL` setting.

### Consumers

* `ggml-cuda.dll` (writer) calls `turbo_innerq_publish` from
  `set-rows.cu` after on-GPU InnerQ calibration finishes.
* `llama.dll` (reader) calls `turbo_innerq_needs_tensor_update` once
  per generated token in `llama-kv-cache.cpp`, uploads the updated
  tensor, then calls `turbo_innerq_mark_tensor_updated`.

The interim shim before this refactor lived in
`ggml/src/ggml-cuda/turbo-innerq.cuh` (commit `ccdce708f`, 2026-05-02)
and was removed by `3f160d33f` (2026-05-03) once the public-header
move was complete.

---

## Llama-side wiring

### `src/llama-kv-cache.cpp` and `src/llama-kv-cache.h`

The KV cache wiring for turbo K/V tensors. ~427 lines added vs upstream
and ~50 lines of header changes.

By @atomicmilkshake (in this fork's history, includes the May 2026
merge-resolution work):

* Detection of `--cache-type-{k,v}` turbo arguments
* Allocation of the device-side `scale_inv` tensor (small
  `[INNERQ_MAX_CHANNELS]` float buffer in VRAM) for InnerQ
* Once-per-token check of `turbo_innerq_needs_tensor_update()` and the
  conditional upload
* Integration with TriAttention (KV pruning) and turbo simultaneously

### `common/arg.cpp`

Parsing of the `turbo2`, `turbo3`, `turbo4` string tokens for
`--cache-type-k` and `--cache-type-v`.

---

## TriAttention

(Inherited from `feature/triattention`, by @atomicmilkshake. Documented
fully in the existing `docs/TRIATTENTION.md` and `docs/TRIATTENTION-API.md`.)

* GPU-accelerated KV cache pruning, ~1000x faster than CPU scoring
* Implements arXiv 2604.04921
* CLI flags: `--triattention-stats`, `--triattention-budget`,
  `--triattention-window`, `--triattention-log`,
  `--triattention-calibrate`, `--triattention-calibrate-out`

---

## Robustness fixes (May 2026)

### `common/chat.cpp` (commit `b541f338a`)

`common_chat_peg_parse` previously threw an exception when the model
emitted malformed tool-call XML (most commonly: duplicate
`</parameter>` tags from Qwen / Hermes-style replies), which the
server surfaced as HTTP 500.

The patch returns the raw text as plain assistant content with a
stderr warning instead. Keeps the chat client functional while still
flagging genuine parser bugs in the log.

### Build-system fixes (commit `ccdce708f`)

* `_USE_MATH_DEFINES` gate so MSVC exposes `M_PI` (turbo CPU WHT path).
* `GGML_API` export on `turbo3_cpu_wht_group_size` for cross-DLL
  visibility.
* `extern "C" { GGML_API int turbo3_cpu_wht_group_size; }` declaration
  fix to avoid C++ name mangling on the import side.
* Interim `dllimport`/`dllexport` shim on the InnerQ symbols. Later
  superseded by the proper public-header move.

---

## Upstream merges in the May 2026 cycle

(All conflict resolution by @atomicmilkshake. Upstream content credited
to upstream llama.cpp authors, see PR threads on
github.com/ggml-org/llama.cpp.)

* `1619a8ce3`: 469 commits, b8672 -> b9008. Type-id renumbering for
  upstream's `Q1_0=41` collision. Introduction of the public InnerQ ABI.
  Forwarding turbo work through PR #21038 (rotate activations for better
  quantization), PR #21513 (heterogeneous iSWA attention rotation),
  PR #21352 (fast Walsh-Hadamard for KV rotation).
* `7194e167f`: 3 commits, b9008 -> b9010. Clean.
* `af4fc6008`: b9010 -> b9016. Clean.
* `fd0a94a4f`: 6 commits, b9016 -> b9033. Includes PR #22004 (per-model
  `load_hparams`/`load_tensors` refactor, 129 files). Despite its size
  no conflict with turbo code. PR #22654 autoparser fixes auto-merged
  with the chat-parser fallback above.

---

## Build infrastructure (May 2026)

By @atomicmilkshake. Lives at the repo root:

* `build-tq-env.bat`: VS 2022 BuildTools env source plus CUDA 13.1,
  CMake, Ninja `PATH` setup.
* `build-tq-go.bat`: Configure / build / clean wrapper. Single source
  of truth for the CMake invocation and the build flag set
  (`CMAKE_CUDA_ARCHITECTURES=120a`, `GGML_CUDA_FA_ALL_QUANTS=ON`,
  `GGML_CUDA_GRAPHS=ON`, full release config).

---

## Documentation

* `docs/CHANGES-2026-05.md`: Per-commit changelog of the May 2026
  cycle.
* `docs/BUILD-WINDOWS.md`: Verified Windows build recipe with
  toolchain pins and the gotchas hit during this cycle.
* `docs/TURBOQUANT-INTERNALS.md`: Contributor-oriented map of the
  TurboQuant code paths, type-id table, InnerQ handshake diagram,
  FA-vec dispatch table, recipe for adding a new turbo type.

(Documentation drafted from @atomicmilkshake's commits and inline code
comments per the AGENTS.md "Documentation drafts: For components the
contributor already understands thoroughly" provision.)

---

## Performance reference

Verified 2026-05-05 on RTX 5090, `Qwen3.6-27B-UD-Q6_K_XL.gguf`,
`turbo4` KV, FA on, `--no-mmap`, `ngl 99`, single GPU:

| Context fill | Generation t/s |
|---|---|
| empty (5-rep) | 52.42 +/- 0.70 |
| 8k | 51.45 |
| 32k | 49.20 |
| 64k | 45.47 |
| 128k | 39.58 |

131k needle-in-a-haystack: 3 of 3 hits at 25, 50, 75 percent depths.

The May 2026 perf rework (commit `3f160d33f`) closed the gap between
turbo4 and q8_0 KV at long context. Pre-rework, the gap was roughly
42 t/s in q8's favor on this hardware. Post-rework turbo4 sits within
single-digit-percent of q8_0 throughput while using one-quarter of
the KV memory.
