# TurboQuant internals

This is a contributor-oriented map of the TurboQuant code in this fork. If
you only want to use turbo KV (`--cache-type-k turbo4 --cache-type-v turbo4`),
the README is enough. Read this if you intend to modify the kernels, port
to a new architecture, or debug a perf regression.

> Method paper: *TurboQuant: KV Cache Compression via PolarQuant + QJL*,
> [arXiv 2504.19874](https://arxiv.org/abs/2504.19874) (ICLR 2026).

---

## 1. GGML type-id allocation

Three types are added to `enum ggml_type` (in `ggml/include/ggml.h`):

| ID | Name | Description |
|---|---|---|
| 42 | `GGML_TYPE_TURBO3_0` | 3-bit KV: 2-bit PolarQuant + 1-bit QJL |
| 43 | `GGML_TYPE_TURBO4_0` | 4-bit KV: 3-bit PolarQuant + 1-bit QJL |
| 44 | `GGML_TYPE_TURBO2_0` | 2-bit KV: PolarQuant only (no QJL) |

These IDs are post-2026-05 values. They were 41/42/43 in the previous
public snapshot and got renumbered when upstream's b9008 added
`GGML_TYPE_Q1_0 = 41`. See *CHANGES-2026-05.md* section 2 for the
migration note.

The corresponding `ggml_ftype` values are not allocated. Turbo is intended
for live KV cache, not on-disk model weights, so it's fine that no GGUF
loader path needs them.

The `block_turbo{2,3,4}_0` layouts (in `ggml-common.h`) all share the
half-precision `norm` field followed by quantised data. `block_turbo4_0`
has a 16-bit `pad` (was historically `rnorm`) that exists solely to keep
`qs[]` 4-byte aligned for the CUDA load instructions discussed below.

---

## 2. File map

### Per-block code paths

```
TurboQuant
|
|-- ggml-base   |---- ggml/include/ggml-turbo-innerq.h    public InnerQ ABI
|               '---- ggml/src/ggml-turbo-innerq.cpp      InnerQ host state
|
|-- ggml-cpu    |---- ggml/src/ggml-turbo-quant.c         CPU quant + dequant ref
|               '---- ggml/src/ggml-cpu/ops.cpp           CPU set_rows turbo dispatch
|
'-- ggml-cuda   |---- ggml/src/ggml-cuda/turbo-quant.cuh  CUDA centroid tables, helpers
                |---- ggml/src/ggml-cuda/turbo-wht.cu     Walsh-Hadamard transform
                |---- ggml/src/ggml-cuda/set-rows.cu      CUDA quantize-on-write
                |---- ggml/src/ggml-cuda/fattn-vec.cuh    main FA kernel (turbo-aware)
                '---- ggml/src/ggml-cuda/fattn-common.cuh turbo vec_dot + dequant impls
```

### Llama-side wiring

```
src/llama-kv-cache.{h,cpp}     turbo allocation, scale_inv tensor, per-token sync
common/arg.cpp                 --cache-type-{k,v} turbo{2,3,4} parser
```

---

## 3. The InnerQ cross-DLL handshake

InnerQ ("inner-quantization") calibrates a per-channel reciprocal-scale
vector `scale_inv[N_CHANNELS]` once, very early in the prefill of a
sequence. The calibration runs on the GPU (in `set-rows.cu`). The result
is needed on the host (so `llama-kv-cache.cpp` can know when to upload an
updated tensor).

The shared state lives in `ggml-base` (always linked) rather than
`ggml-cuda` (a runtime plugin under `GGML_BACKEND_DL=ON`):

```
ggml-cuda.dll                       ggml-base.dll                       llama.dll
-------------                       -------------                       ---------
set-rows.cu  -----[ publish ]--->   ggml-turbo-innerq.cpp  --[ check ]-->  llama-kv-cache.cpp
                                    g_innerq_finalized                     turbo_innerq_needs_tensor_update()
                                    g_innerq_scale_inv_host[]              turbo_innerq_mark_tensor_updated()
```

The public ABI is `ggml/include/ggml-turbo-innerq.h`. Don't add `extern`
declarations of these symbols in your own files. Include the header.

> Why this matters: prior to the May-2026 work the state lived in
> `ggml-cuda.dll` and the build broke under `GGML_BACKEND_DL=ON`. See
> *CHANGES-2026-05.md* section 2 for the move and section 1 for the
> interim `dllimport`/`dllexport` shim that preceded it.

---

## 4. CUDA flash-attention dispatch

The vector-shape FA kernel
(`ggml/src/ggml-cuda/fattn-vec.cuh::flash_attn_ext_vec`) is templated on
`(D, ncols, type_K, type_V, use_logit_softcap)`. Turbo specialisations
are chosen via:

```cpp
constexpr bool type_K_is_turbo = (type_K == GGML_TYPE_TURBO3_0
                                  || type_K == GGML_TYPE_TURBO2_0
                                  || type_K == GGML_TYPE_TURBO4_0);
constexpr bool type_V_is_turbo = /* same for V */;
```

### Thread-count settings (turbo path, post-2026-05)

| Parameter | Turbo K | Turbo V | Notes |
|---|---|---|---|
| `nthreads_KQ` | `nthreads_KQ_q / 2 = 16` | (n/a) | 4-byte aligned `qs` loads |
| `nthreads_V`  | (n/a) | `16` (explicit) | one centroid per lane |
| `cpy_ne_KQ`   | `D/(2*nthreads_KQ)` | (n/a) | bound to fit `Q_reg` |
| `V_rows_per_thread` | (n/a) | `2*cpy_ne` | covers full `D=128` in 2 iters |
| `__launch_bounds__ minBlocksPerSM` | `4` | `4` | up from `2` |

Each of those values is justified inline with a `// PERF (...)` comment
in `fattn-vec.cuh`. If you change one, update the comment.

### vec_dot and dequant

The per-K KQ dot product is:

```cpp
constexpr vec_dot_KQ_t vec_dot_KQ = get_vec_dot_KQ<type_K, D, nthreads_KQ>();
```

Specialisations live in `fattn-common.cuh`:

* `vec_dot_fattn_vec_KQ_turbo3_0<D, nthreads_KQ>`
* `vec_dot_fattn_vec_KQ_turbo4_0<D, nthreads_KQ>`
* `vec_dot_fattn_vec_KQ_turbo2_0<D, nthreads_KQ>`

These take the float-Q path (Q_reg is `half2`), not the q8_1 integer Q
path. The unused `Q_q8` argument is intentionally there to keep the
function signature uniform across types.

The V dequant uses a similar template family
(`get_vec_dot_VKQ<type_V, D, nthreads_V>`). See `fattn-common.cuh` for
the turbo specialisations.

### The `__dp4a` int8 path (currently unused)

`turbo-quant.cuh` ships a pre-quantised int8 centroid table:

```c
static __constant__ int8_t TURBO_CENTROIDS_4BIT_INT8[16] = { -127, -86, ..., 127 };
#define TURBO_INT8_4BIT_SCALE_REVERSE (0.173926f / 127.0f)
```

This was added in `3f160d33f` as a candidate path that uses Blackwell's
`__dp4a` 8-bit dot-product hardware instead of float multiplies. In
current builds the float path wins because the `__constant__`-memory
lookup serialises on divergent indices, eating the `__dp4a` advantage at
depth. The table is kept ready in case a future architecture changes
that trade-off. Switching paths is a one-line edit in `fattn-vec.cuh`.

---

## 5. The Walsh-Hadamard transform

`ggml/src/ggml-cuda/turbo-wht.cu` runs a `group_size` of 64 or 128 WHT on
each block before quantisation (and the inverse on dequant). The kernel
is 8 iterative stages where stage `h` swaps pairs that are `h` indices
apart:

```
stage h=1:  pairs 0-1, 2-3, 4-5, ...     (all within same warp)
stage h=2:  pairs 0-2, 1-3, 4-6, ...     (within warp)
stage h=4:  pairs 0-4, 1-5, ...          (within warp)
stage h=8:                               (within warp)
stage h=16:                              (within warp; warp size = 32)
stage h=32:                              (CROSS-WARP, full barrier)
stage h=64: only when group_size == 128  (CROSS-WARP)
```

Stages 1 to 16 only swap within a single warp, so `__syncwarp()` is
sufficient and roughly 10x cheaper than `__syncthreads()`. The split is
hard-coded:

```c
WHT_STAGE_WARP(1) WHT_STAGE_WARP(2) WHT_STAGE_WARP(4)
WHT_STAGE_WARP(8) WHT_STAGE_WARP(16)
WHT_STAGE_BLOCK(32)
if (group_size == 128) { WHT_STAGE_BLOCK(64) }
```

The WHT runs twice per layer (Q + V) on every generated token, which on
a 64-layer model is 128 invocations per token. Every saved barrier
matters.

---

## 6. KV-cache lifecycle

The relevant code is in `src/llama-kv-cache.cpp`. Per token:

1. `llama_kv_cache::set_input()` is called.
2. If turbo is in use, the cache checks
   `turbo_innerq_needs_tensor_update()`.
3. If true, it copies `g_innerq_scale_inv_host` to the device-side
   scale_inv tensor (a small `[INNERQ_MAX_CHANNELS]` float buffer in
   VRAM) and calls `turbo_innerq_mark_tensor_updated()`.
4. The FA kernel reads scale_inv from this tensor, never directly from
   `g_innerq_scale_inv_host`.

The publish path runs from the GPU side after a `set_rows` invocation
finishes the InnerQ stage, in `set-rows.cu`. There's no explicit
barrier. The next launch in the same stream observes the host-side
state via the implicit `cudaMemcpy` ordering.

---

## 7. Useful debugging commands

```cmd
REM list registered ggml types and their numeric ids:
build-tq-merged\bin\llama-cli.exe --help | findstr -i turbo

REM verify a turbo build runs at all:
build-tq-merged\bin\llama-server.exe --version

REM bench TG at empty cache (5 reps, no-mmap):
build-tq-merged\bin\llama-bench.exe -m model.gguf -fa 1 -ctk turbo4 -ctv turbo4 ^
   -ngl 99 -mg 0 -dev CUDA0 -ts 1 -mmp 0 -p 0 -n 128 -r 5

REM bench at 8k/32k/64k/128k context fills (separate PP and TG):
build-tq-merged\bin\llama-batched-bench.exe -m model.gguf -c 131072 ^
   -b 2048 -ub 1024 -fa on -ctk turbo4 -ctv turbo4 ^
   -ngl 99 -mg 0 -dev CUDA0 -ts 1 --no-mmap ^
   -npp 8000,32000,64000,128000 -ntg 128 -npl 1
```

```python
# Needle-in-a-haystack at 131k with turbo4 (3 depths, 1 needle each):
python scripts/utilities/needle_test_v2_turbo.py
```

---

## 8. Where to add new turbo types

If you ever want to add a `GGML_TYPE_TURBO5_0`, in order:

1. `ggml/include/ggml.h`: append to `enum ggml_type`, bump
   `GGML_TYPE_COUNT`.
2. `ggml-common.h`: add `block_turbo5_0` layout.
3. `ggml/src/ggml-turbo-quant.c`: add `quantize_row_turbo5_0_ref` and
   `dequantize_row_turbo5_0`. Register in the type traits table.
4. `ggml/src/ggml-cpu/ops.cpp`: add the type to the `set_rows` turbo
   dispatch (look for the existing `GGML_TYPE_TURBO{2,3,4}_0` if-tree).
5. `ggml/src/ggml-cuda/turbo-quant.cuh`: centroid tables, helpers.
6. `ggml/src/ggml-cuda/set-rows.cu`: quantize-on-write kernel.
7. `ggml/src/ggml-cuda/fattn-common.cuh`: `vec_dot_fattn_vec_KQ_turbo5_0`
   and the V dequant counterpart.
8. `ggml/src/ggml-cuda/fattn-vec.cuh`: extend the `type_K_is_turbo` /
   `type_V_is_turbo` constants.
9. `ggml/src/ggml-cuda/template-instances/`: add the cross-product
   `fattn-vec-instance-turbo5_0-*` files matching the pattern of the
   existing turbo instances.
10. `src/llama-kv-cache.cpp`: add the new ftype to the parsing of
    `--cache-type-k` and `--cache-type-v`.
11. `common/arg.cpp`: add the string token `"turbo5"` to the parser.

Build, run the needle test at 131k to validate correctness end-to-end,
then update the type-id table at the top of this doc.
