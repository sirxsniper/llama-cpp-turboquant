# Benchmarks: Qwen3.6-27B-Q6_K_XL @ turbo4 KV, single RTX 5090

Reference numbers for this fork's TurboQuant turbo4 KV cache, captured on
2026-05-05. Measured end-to-end with `llama-batched-bench` (throughput) and
`llama-cli` (needle-in-a-haystack recall) at seven context fills from 4k
to 256k tokens.

**Headline result: 21/21 needles recovered (100% accuracy) across 4k -> 256k contexts.
Generation throughput drops from 51.6 t/s at 4k to 31.1 t/s at 256k**, while keeping
the 256k KV cache inside 4.4 GiB of VRAM.

---

## 1. Hardware

| Component | Spec |
|---|---|
| GPU | **NVIDIA GeForce RTX 5090**, Blackwell (sm_120, compute capability 12.0) |
| VRAM | 32 GiB GDDR7 (32606 MiB reported by `nvidia-smi`) |
| GPU memory bandwidth | 1.79 TB/s (theoretical peak) |
| GPU TDP | 575 W |
| NVIDIA driver | 596.36 |
| Host OS | Windows 11 Pro (build 26200) |
| Power plan | High Performance |
| NVIDIA Control Panel power management | Prefer maximum performance |

The 256k pass uses the 5090 only. No multi-GPU, no CPU offload.

## 2. Build

| Component | Version |
|---|---|
| llama-cpp-turboquant | this fork at `8680495a8` (head of `main`, b9033 + May 2026 perf rework) |
| CUDA Toolkit | 13.1.80 |
| MSVC compiler | 14.44.35207 (VS 2022 BuildTools, 17.x) |
| CMake | 4.3.2 |
| Ninja | 1.13.2 |
| `CMAKE_CUDA_ARCHITECTURES` | `120a` (Blackwell consumer) |
| Build flags | `GGML_CUDA=ON`, `GGML_CUDA_FA=ON`, `GGML_CUDA_FA_ALL_QUANTS=ON`, `GGML_CUDA_GRAPHS=ON`, `GGML_NATIVE=ON`, `Release` |

Full build recipe in [`docs/BUILD-WINDOWS.md`](BUILD-WINDOWS.md).

## 3. Model

| Property | Value |
|---|---|
| Name | `Qwen3.6-27B-UD-Q6_K_XL` (Unsloth dynamic Q6) |
| Source | <https://huggingface.co/unsloth/Qwen3.6-27B-GGUF> |
| File | `Qwen3.6-27B-UD-Q6_K_XL.gguf` |
| Size on disk | 23.87 GiB (7.62 bits per weight) |
| Parameters | 26.90 B |
| Architecture | qwen35 (64 layers, head_count 24, head_count_kv 4, GQA-6) |
| Embedding dim | 5120 |
| Feed-forward dim | 17408 |
| Head dim K / V | 256 / 256 |
| Native context length | 262144 (256k) |

## 4. Inference settings

```
--cache-type-k turbo4 --cache-type-v turbo4
--flash-attn on
-ngl 99
-mg 0 -dev CUDA0 -ts 1
--no-mmap
--batch-size 2048 --ubatch-size 1024
--threads 16 --threads-batch 16
--ctx-size 262144
```

KV-cache footprint at turbo4 (1.0 byte per token per dim per layer for K + same for V; the
fork allocates `INNERQ_MAX_CHANNELS` scale_inv on top, ~1 MiB total):

| Context | KV cache | Total VRAM (model + KV + compute buffer) |
|---|---|---|
| 4k    | 68 MiB    | ~25.0 GiB |
| 8k    | 136 MiB   | ~25.1 GiB |
| 16k   | 272 MiB   | ~25.2 GiB |
| 32k   | 544 MiB   | ~25.5 GiB |
| 64k   | 1088 MiB  | ~26.1 GiB |
| 128k  | 2176 MiB  | ~27.1 GiB |
| 256k  | 4352 MiB  | ~29.3 GiB |

All seven fits comfortably in the 5090's 32 GiB.

---

## 5. Throughput (`llama-batched-bench`)

Single sequence (`-npl 1`), 128-token generation per run (`-ntg 128`), measured separately for
prompt processing (PP) and token generation (TG).

| Context fill | Generation t/s | Prompt-process t/s | TTFT (PP time) | 128-tok gen time |
|---:|---:|---:|---:|---:|
|   4k | **51.61** | 3258.74 |    1.23 s | 2.48 s |
|   8k | **51.73** | 3255.96 |    2.46 s | 2.47 s |
|  16k | **50.74** | 3155.47 |    5.07 s | 2.52 s |
|  32k | **48.98** | 2921.86 |   10.95 s | 2.61 s |
|  64k | **45.36** | 2455.34 |   26.07 s | 2.82 s |
| 128k | **39.51** | 1750.36 |   73.13 s | 3.24 s |
| 256k | **31.12** | 1095.35 |  233.71 s | 4.11 s |

Generation throughput drops 39.7% from 4k to 256k context (51.6 -> 31.1 t/s). Prompt
processing drops 66.4% over the same range (3259 -> 1095 t/s) which is normal scaling
for flash-attention kernels at long context.

Reproduction:

```cmd
llama-batched-bench.exe ^
  -m Qwen3.6-27B-UD-Q6_K_XL.gguf ^
  -c 262144 -b 2048 -ub 1024 ^
  -fa on -ctk turbo4 -ctv turbo4 ^
  -ngl 99 -mg 0 -dev CUDA0 -ts 1 --no-mmap ^
  -npp 4000,8000,16000,32000,64000,128000,256000 ^
  -ntg 128 -npl 1 -t 16 -tb 16
```

Total wall time for the seven-point sweep: 376 s (model load + 7 PP+TG rounds).

---

## 6. Needle-in-a-haystack accuracy

Three needles rotate per (context, depth) pair via a deterministic hash:

| Needle | Subtle fact | Expected exact-token answer |
|---|---|---|
| 0 | "The 1837 Treaty of Helsterveen ... reduced the export tariff on Bavarian copper to 4.7 percent" | `4.7` |
| 1 | "Marina Olbrecht ... bronze figure titled 'The Woolwasher' ... after working on it for thirty-one months" | `thirty-one` |
| 2 | "The submarine USS Calderwood was laid down ... commissioned with hull number SS-509" | `SS-509` |

Each fact is inserted at the configured depth (25 / 50 / 75 % of the document) inside
WikiText-2 padding sized to fill the target context. The model is given a paraphrased
question and instructed to quote the number or name exactly. Greedy decoding (temp = 0)
via `llama-cli`. A run is a hit if the expected exact-token string appears anywhere in
the model's reply.

### Per-(ctx, depth) results

| Context | depth 25% | depth 50% | depth 75% | hit rate |
|---:|:---:|:---:|:---:|:---:|
|   4k | OK 17.5s | OK 16.8s | OK 16.7s | **3/3** |
|   8k | OK 18.0s | OK 17.9s | OK 18.0s | **3/3** |
|  16k | OK 20.8s | OK 20.9s | OK 20.9s | **3/3** |
|  32k | OK 26.9s | OK 27.0s | OK 26.8s | **3/3** |
|  64k | OK 41.6s | OK 41.8s | OK 42.0s | **3/3** |
| 128k | OK 89.7s | OK 89.9s | OK 89.9s | **3/3** |
| 256k | OK 233.0s | OK 234.3s | OK 234.8s | **3/3** |

**Overall: 21/21 = 100%.** Time per cell is total wall clock for the run (model load +
prompt processing + 300 generated tokens).

Each elapsed time is dominated by prompt processing at the configured context fill:
4k = ~17 s, 8k = ~18 s, 16k = ~21 s, 32k = ~27 s, 64k = ~42 s, 128k = ~90 s, 256k = ~234 s.

Reproduction:

```cmd
python scripts\utilities\needle_multictx_turbo4_qwen36.py
```

(That runner sources `scripts/utilities/needle_test_v2.py`, which uses
`llama-cli.exe` from `build-tq-merged\bin\` and reads padding text from
`%LOCALAPPDATA%\Temp\turboq\wikitext-2-raw\`. Download
`wikitext-2-raw-v1.zip` from <https://wikitext.smerity.com/wikitext-2-raw-v1.zip>
and extract there if you don't have it.)

---

## 7. Methodology notes

* **Single-GPU isolation.** All runs use `-dev CUDA0 -ts 1` so the work stays on one
  device. Reported numbers are not multi-GPU-split.
* **No mmap.** `--no-mmap` is required for accurate TG numbers; mmap-on can cause
  occasional page-fault stalls during generation that show up as a roughly 1-3 t/s drop.
* **Power state.** Both the Windows power plan (High Performance) and the NVIDIA
  Control Panel "Power management mode" (Prefer maximum performance) must be set
  before the bench. With the default Balanced plan plus Optimal Power, the GPU sits in
  P1 with graphics clock ~2780 MHz instead of P0 with 3090 MHz, costing roughly 10% TG
  throughput. This was empirically confirmed during the May 2026 perf-regression hunt.
* **Cold cache for each PP measurement.** `llama-batched-bench` resets the KV cache
  before each `-npp` value, so each PP throughput is from an empty cache to the target
  fill. The TG number measured immediately after is steady-state generation with the
  cache filled.
* **Deterministic needle assignment.** Needle index = `hash((ctx, depth)) % 3`. This
  gives the same needle for a given `(ctx, depth)` pair across repeated runs, so
  results are directly comparable.
* **Greedy decoding.** Needle test uses `--temp 0` for reproducibility. Probabilistic
  decoding might score slightly differently but is not what we want to measure here.

---

## 8. What this demonstrates

* **turbo4 KV cache holds at 256k.** No quality collapse - 21 of 21 needles recovered,
  including all three at the maximum-supported 262144-token context.
* **Generation stays usable at long context.** 31 t/s at 256k context is well within
  interactive-chat territory (faster than human reading speed).
* **The 5090's 32 GiB is enough headroom for 256k turbo4 inference of a 27B Q6 model.**
  Total VRAM used is ~29.3 GiB at full 256k fill, leaving ~3 GiB headroom for compute
  buffers and OS overhead.
* **The May 2026 perf rework paid off.** The turbo K vec_dot, V dequant, and Walsh-
  Hadamard barrier-split optimizations (commit `6df078f`) closed the gap with q8_0 KV
  while keeping turbo4's 4x compression advantage. See
  [`docs/CHANGES-2026-05.md`](CHANGES-2026-05.md) section 4 for the per-optimization
  breakdown.

---

## 9. Raw logs

* Throughput: `D:\Projects\LocalAI\_downloads\bench-256k-7points.log`
* Needle: `D:\Projects\LocalAI\_downloads\needle-multictx-turbo4.log`
* Per-prompt artifacts: `%LOCALAPPDATA%\Temp\turboq\needle_multictx\`
* Raw JSON: `%LOCALAPPDATA%\Temp\turboq\needle_multictx\results.json`
