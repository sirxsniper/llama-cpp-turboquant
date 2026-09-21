# turbot: test sequence

`[TAG_TURBOT]`. This is the ordered, GPU-safe test plan for the turbot tiered KV cache (contract: `docs/turbot/SPEC.md`, section 11 for the tests). Each gate has commands and a pass criterion. Run the gates in order. A failed gate stops the sequence until it is understood. The B0 speed gate can stop the whole integration.

Paths below use the worktree `D:\Projects\LocalAI\source-build\turboquant-turbot` (called `WT`) and the build directory `WT\build-turbot` (called `BIN` for its `bin` folder).

---

## 0. Rules for every GPU step

1. **One GPU process at a time.** Before every launch, check for *any* llama, ggml or test-backend process (a pattern match, never a list of names):
   ```powershell
   Get-Process | Where-Object { $_.ProcessName -match 'llama|ggml|test-backend' }
   ```
   It must print nothing. The Jarvis server counts as one: stop it first.
2. **A GPU fault stops everything.** After any `CUDA error`, `illegal memory access`, or an `nvlddmkm` event, do not launch anything else. Check the event log first and reboot if there is an event:
   ```powershell
   Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='nvlddmkm'; StartTime=(Get-Date).AddHours(-2)} -ErrorAction SilentlyContinue
   ```
   After an illegal access the driver returns garbage until the reboot, so any numbers produced after it are invalid.
3. **Test servers use port 8091, never 8080.** Jarvis records whatever serves on 8080 as real usage.
4. **Rebuild tools before use.** Deploys copy only the server and DLLs, and a stale `llama-perplexity` or `llama-bench` crashes against new DLLs (0xC0000005, empty log). Build them from the same tree as the DLLs.
5. **Check output before quoting speed.** Throughput looks normal on a build that emits garbage, so run the correctness gates before any benchmark.
6. **Memory.** C: has little free space and the pagefile grows there. Never run a numpy simulation next to a KL run, and write large outputs to E:. `llama-perplexity` at 131K buffers about 80 GB of logits.

Build (refuses while a GPU process runs or commit is high):
```powershell
powershell -File C:\Users\xSniper\AppData\Local\Temp\claude\D--Projects\8efd145b-de13-4771-ae4d-110a53af2fef\scratchpad\turbot_build.ps1 `
    -Targets "ggml llama llama-server llama-perplexity llama-bench test-backend-ops test-turbot test-turbot-backend"
```

---

## 1. CPU gates (no GPU)

### 1a. Tables and plan tools

| Command | Pass |
|---|---|
| `python docs\turbot\gen_turbot_tables.py --check` | `ggml-turbot-tables.h matches` |
| `python tools\turbot\check_tables_vs_study.py` | `RESULT: PASS`. Structural checks, then the shipped tables against `kv_nested_study.build_designs(..., ['a_lloyd'])`: exact for b = 4, 5; within 2e-7 for b = 2, 3; within 6e-5 for b = 6 |
| `python tools\turbot\plan_vram.py docs\turbot\plans\turbot-default.plan --layers` | base `4520.00`, young pool `726.00`, total `5246.00` MiB, per-layer bytes equal SPEC appendix A, hash `0x56c3503c949a7749` |

To make a plan from another allocation:
```powershell
python tools\turbot\turbot_plan.py convert E:\kv-s3\plans\fq_3t0_w256_m16384_7b.txt -o my.plan
python tools\turbot\turbot_plan.py convert E:\kvdump\alloc3_plans.json --entry 0 -o my.plan
python tools\turbot\turbot_plan.py convert E:\kvdump\alloc3_plans.json --entry 1 --young 7 -o my.plan   # entry 1 has b2 = 6 < 6-bit heads
```

### 1b. `test-turbot` (SPEC 11.1)

- **Build.** `BIN\test-turbot.exe` runs all seven items. Item 7 links the exported `llama_kv_tier` and plan parser from `llama.dll`. The test uses no GPU.
- **While a GPU job is busy.** Use a CPU-only static build of the same worktree instead: it never loads CUDA, so it may run as long as there is RAM headroom.
  ```bat
  call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
  cd /d D:\Projects\LocalAI\source-build\turboquant-turbot
  cmake -S . -B build-turbot-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF -DGGML_NATIVE=ON ^
        -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF
  cmake --build build-turbot-cpu --target test-turbot -j 8
  build-turbot-cpu\bin\test-turbot.exe
  ```

Pass: exit 0 and the last line is `OK`. What it checks:

| Item | Check |
|---|---|
| 1 tables | Old levels bit-equal to kvfq `get_levels` (the Lloyd iteration is copied into the test). Midpoints, antisymmetry, zero middle threshold. Young thresholds sorted, with old thresholds embedded bit for bit. Every young level inside its own cell. Fill codes are the nearest levels. Run offsets equal the closed forms. `a_lloyd` recomputed in float64 within 4 float32 ULP. `ggml_turbot_old_level_i8(4 \| 5)` equal to `TURBO_C4_I8_LIST` / `TURBO_C5_I8_LIST`, and the scales equal `TURBO_INT8_4BIT/5BIT_SCALE_REVERSE` |
| 2 nesting | 10^6 N(0, 1/128) values plus every threshold and level, both float neighbours of each, ±0, ±1, ±FLT_MAX, ±inf, NaN: `young >> r == old` and `refine == young & (2^r - 1)` for every (b, y) |
| 3 planes | set/get roundtrip for w 1..6 over all 256 elements; neighbours and guard bytes untouched; every run byte covered |
| 4 coder | old tier bit-identical to a copy of `llama-kvfq.cpp fq_group` (codes and reconstructed values, b 2..6, 3,200 groups of assorted rows). Sphere Monte Carlo (16,000 groups): old nmse / D_b in [0.93, 1.03]; young nmse / independent y-bit Lloyd <= 1.18 at y = 7, 8. Zero row decodes to exact zeros. Encoding does not depend on the previous destination bytes |
| 5 fill | every (b, y): MSE(fill, old) / D_b <= 0.30 and MSE(fill, true) / D_b <= 1.30 (record the printed table in section 7) |
| 6 params and layout | op params roundtrip, FA bytes 0..15 untouched, bad magic, version, side or granule refused. Default plan bytes per layer equal appendix A (18,080 base, 11,616 pool per cell, 5,500,829,696 B total). Illegal widths refused. Type family traits (`blck 1024`, `type_size 32S+16`, `row_size(256) = 8S+4`) |
| 7a plan parser | the default plan parses with the expected hash. kvfq keys ignored. Y, POOL 0 and a trailing comment accepted. Refused: unknown tag, malformed line, duplicates, widths out of range, a missing L line, a layer the cache does not hold, POOL not a multiple of 64 or above kv_size, negative CAP |
| 7b tier | scenarios of SPEC 11.1 item 7 driving a real `llama_kv_cells`: single-sequence band 16,384; 4 sequences at quota 16,256 with 0 evictions after warm-up; eviction order (smallest margin, never the previous ubatch's granule); empty owned slot first; release then prefill or restore with 0 evictions; 10,000-step draft loop (counter == live rows, band intact); seq_cp adoption; restore into a larger counter; stale bits; trim then fill entry; failure then abort, re-queued fill and counter rollback; `abort_restore`; POOL 0; seq id 255; determinism |

`--quick` cuts the nesting and Monte Carlo sizes for a smoke run.

---

## 2. CUDA correctness (A + B, then C)

Gates 2 and 3 are also run for the `GGML_CUDA_TURBOT_OLD_I8 = 0` build (SPEC 7.4). Configure that build as a second build directory with `-DCMAKE_CUDA_FLAGS=-DGGML_CUDA_TURBOT_OLD_I8=0`, or rebuild after flipping the define in `fattn-turbot.cuh`. Use a different directory name, and never mix its DLLs with the first build's.

Leave `FA_NCOLS128` unset (default on). The nb 512 / 1280 cases must run the `<128,1> <64,2> <32,4> <16,8>` instances.

### 2a. Memory check on small sizes first

```powershell
$cs = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\compute-sanitizer.bat"
& $cs --tool memcheck --error-exitcode 99 BIN\test-backend-ops.exe test -b CUDA0 -o FLASH_ATTN_EXT -p "^turbot=[a-z0-9]+,kv=(96|100),nb=(1|2|4),"
& $cs --tool memcheck --error-exitcode 99 BIN\test-backend-ops.exe test -b CUDA0 -o TURBOT_SET_ROWS -p "rows=(1|4|16),"
& $cs --tool memcheck --error-exitcode 99 BIN\test-turbot-backend.exe --quick
```
Pass: `ERROR SUMMARY: 0 errors` for each and no FAIL line. A memcheck error is a stop (rule 0.2 applies if the driver logged an event).

### 2b. FA and writer against the CPU reference (SPEC 11.2)

```powershell
BIN\test-backend-ops.exe test -b CUDA0 -o FLASH_ATTN_EXT -p "^turbot=" > E:\kv-turbot\tbo_fa_i8.txt
BIN\test-backend-ops.exe test -b CUDA0 -o TURBOT_SET_ROWS -p "^turbot=" > E:\kv-turbot\tbo_writer_i8.txt
```

**Pass:**
- every case prints OK;
- no `not supported`: every turbot case must run on both CUDA0 and CPU;
- the exit code is 0;
- max nmse 5e-4 per case.

**FA coverage:**

| Parameter | Values |
|---|---|
| kv | 96, 100, 1000, 1024, 4096, 16384 (100 and 1000 end mid tile) |
| nb | 1, 2, 4, 8, 16, 512, 1280 |
| granule patterns | old, young, alternating, tail |
| mask | explicit, positional (odd start, holes), none |
| extras | sinks; softcap 30 |
| widths | mixed (K != V, b 2 and 6, y 8) and default layer 23 |

**Writer coverage:**

| Parameter | Values |
|---|---|
| rows | 1, 4, 16, 1280 |
| cell index type | I64, I32 |
| young rows | none, all, mixed |
| fill entries | present, absent |

### 2c. Bytes and the int8 old read (SPEC 11.3)

```powershell
BIN\test-turbot-backend.exe                    # OLD_I8 1 build
BIN_F\test-turbot-backend.exe --old-read float # OLD_I8 0 build
```

**Pass:** `OK (0 failures)`, which requires all of the following:
- writer codes agree on >= 99.99% of elements (base, refinement and fill), and gains are within one f16 step;
- the CPU op equals ggml-turbot.h bit for bit;
- no byte outside the written rows and pool parts changes on CUDA;
- every V element of the OLD read is within one half step of `old_level_i8 · scale · gain` (b <= 5), or of `C_b · gain` (b = 6 or `--old-read float`).

### 2d. Existing types unchanged

- `powershell -File D:\Projects\LocalAI\source-build\validate.ps1 -BuildDir WT\build-turbot -Tag turbot`: must print `GATE PASSED`. It runs the full `test-backend-ops` suite, including the f16, q8_0, turbo4p and turbo5p cases, plus the production end-to-end generation on turbo5p.
- **Codegen identity** (SPEC 12.3 item 7). Every object of the existing FA template instances (`ggml-cuda\template-instances\fattn-*-instance-*.cu`, excluding `fattn-mma-turbot-*`) must be byte-identical, relocations masked, to a build of the base commit (`60e5e979e`). Its constant-memory map must also be unchanged. `fattn.cu` and `ggml-cuda.cu` may differ in host code only.

---

## 3. Gate B0: speed of the read path (SPEC 7.8)

This gate runs at integration after A and B build, before the C and D results are used. Section 2b must pass first.

```powershell
BIN\test-backend-ops.exe perf -b CUDA0 -o FLASH_ATTN_EXT -p "turbot_perf=|turbot_ref=" > E:\kv-turbot\b0_i8.log
BIN_F\test-backend-ops.exe perf -b CUDA0 -o FLASH_ATTN_EXT -p "turbot_perf=|turbot_ref=" > E:\kv-turbot\b0_float.log
python tools\turbot\b0_gate.py E:\kv-turbot\b0_i8.log --alt E:\kv-turbot\b0_float.log --md
```

**Cases.**
- turbot FA with default-plan layer 23, against turbo5p FA at the same shape (head 256, GQA 6, explicit mask).
- kv 32768 / 131072 / 245760, nb 1 / 4 / 512.
- Tier mixes old, band16k and band64k; young is measured only at kv 32768.
- The cache bytes are host-encoded (512 distinct cells tiled), so no writer is needed.

**GO iff all of these hold** (ms per op, turbot / turbo5p):

| kv | nb | mix | limit |
|---|---|---|---|
| 131072 | 4 | band16k | <= 1.00 |
| 245760 | 4 | band16k | <= 1.00 |
| 131072 | 1 | band16k | <= 1.00 |
| 131072 | 4 | band64k | <= 1.05 |
| 131072 | 512 | band16k | <= 1.10 |

`b0_gate.py` also applies the `GGML_CUDA_TURBOT_OLD_I8` rule: keep the faster build that passes, and if both pass within 2% of each other, keep 0 (float reads).

**NO-GO:** stop the integration and paste both tables into section 7. The design is reopened, not tuned around.

Writer speed (SPEC 8.2 alternatives, reported only):
```powershell
BIN\test-backend-ops.exe perf -b CUDA0 -o TURBOT_SET_ROWS,SET_ROWS -p "turbot_perf=writer_|type_dst=turbo5p"
```

---

## 4. Full-stack correctness (A-D integrated)

Every server below runs `-ctk turbot -ctv turbot --kv-tier-plan WT\docs\turbot\plans\turbot-default.plan` on port 8091, one at a time. The scratchpad harnesses (`acceptab.py`, `overcommit_accept.py`, `pool_exact.py`, `crosstalk.py`) build their command from the Jarvis production profile. They take the cache type from that profile, so each needs a turbot arm with these arguments appended (the last `-ctk` wins). `LLAMA_TURBOT_PLAN` can carry the plan instead of the flag.

| # | Test | Command / procedure | Pass |
|---|---|---|---|
| 4.1 | tier unit tests | section 1b item 7 on the integrated tree | `OK` |
| 4.2 | refusals | start `llama-server` with `-ctk turbot` alone; `-ctk turbot -ctv turbot` without a plan; `-np 4` without `--kv-unified`; `-fa off`; `TURBO_KV_CPU_LAYERS=1`; `TURBO_LAYER_ADAPTIVE=1`; `TURBO_INNERQ=1` | each exits with a message starting `turbot:` (or the plan message of SPEC 9.1). No crash, no CUDA error |
| 4.3 | kill switch | `LLAMA_TURBOT=0` with `-ctk turbot -ctv turbot` | warning `LLAMA_TURBOT=0: turbot disabled, using turbo5p`, size line equals turbo5p |
| 4.4 | size line | server log | `turbot plan ...: 16 layers, old bits 4.289 (sum 549), young pool 65536 cells (1024 granules), cap 16384, hash 0x56c3503c949a7749` and `size = 5246.00 MiB ... young pool:  726.00 MiB`; no F16 scratch buffer |
| 4.5 | crosstalk | `crosstalk.py` with the turbot arm, cold and warm | every slot returns its own passphrase |
| 4.6 | overcommit | `overcommit_accept.py 62000 6000` with the turbot arm (4 × 62K + 6,000) | as for turbo5p: all agents finish, each follow-up holds its own passphrase and no other |
| 4.7 | pool exactness | `pool_exact.py <exe>` with the turbot arm | every arm identical to its reference, parks and resumes as listed |
| 4.8 | prefill tails | prompts of 512·k + n tokens, n = 2..5, at `-ub 512`, twice each | no CUDA error; greedy output coherent and identical across the two runs |
| 4.9 | state blobs | `python tools\turbot\blob_roundtrip.py --exe BIN\llama-server.exe` | `RESULT: 5/5 arms passed` (save and restore into another slot, prompt-cache restore, park and resume, turbo5p to turbot refused, turbot to turbo5p refused) |
| 4.10 | trim | two requests on one slot, the second a prefix of the first plus new text (`cache_prompt`), under `LLAMA_TURBOT_DEBUG=2` | no invariant assertion; the DEBUG=1 line shows fills > 0; the answer holds the passphrase |
| 4.11 | failure injection | `LLAMA_TURBOT_FAIL_UBATCH=<n>` for n in {1, 17, 500} with 4 slots and unique passphrases (crosstalk prompts), `LLAMA_TURBOT_DEBUG=2` | the failing request errors or retries per server policy; every other slot answers with its own passphrase; no assertion |
| 4.12 | hybrid prepare order | DFlash2 speculative decode with `LLAMA_TURBOT_DEBUG=2` (the `prepare()` apply path) and a trim that forces a fill on a reused hybrid graph | no invariant assertion; the fill runs (DEBUG=1 fill count > 0) and the output is coherent |

---

## 5. Quality

Model `D:\Projects\LocalAI\models\Qwen3.8-27B-UD-Q5_K_XL.gguf`. The reference is the S0 f16 base files in `E:\kv-bar-s0`, with the same corpora, chunking and batch as S0 and S3, so the numbers are directly comparable.

```powershell
$env:LLAMA_TURBOT_PLAN = "WT\docs\turbot\plans\turbot-default.plan"
$ppl  = "BIN\llama-perplexity.exe"
$args = "-m D:\Projects\LocalAI\models\Qwen3.8-27B-UD-Q5_K_XL.gguf -c 32768 --chunks 16 -ngl 999 -fa on --load-mode none -b 2048 --threads 16 -ctk turbot -ctv turbot"
& $ppl $args.Split(" ") -ub 512  -f E:\kv-bar-s0\code_corpus.txt  --kl-divergence-base E:\kv-bar-s0\base_code.dat  --kl-divergence *> E:\kv-s3\code_turbot.log
& $ppl $args.Split(" ") -ub 512  -f E:\kv-bar-s0\prose_corpus.txt --kl-divergence-base E:\kv-bar-s0\base_prose.dat --kl-divergence *> E:\kv-s3\prose_turbot.log
& $ppl $args.Split(" ") -ub 1280 -f E:\kv-bar-s0\code_corpus.txt  --kl-divergence-base E:\kv-bar-s0\base_code.dat  --kl-divergence *> E:\kv-s3\code_turbot_ub1280.log
& $ppl $args.Split(" ") -ub 1280 -f E:\kv-bar-s0\prose_corpus.txt --kl-divergence-base E:\kv-bar-s0\base_prose.dat --kl-divergence *> E:\kv-s3\prose_turbot_ub1280.log
python <scratchpad>\kv_gate_stats.py --arms turbot,turbot_ub1280 --corpora code,prose --refs floor_ub1280,q8_0
```

**Pass:** for each corpus and each statistic (mean KLD, same-top), the paired chunk-bootstrap 95% CI of both (arm − floor) and (arm − q8_0) contains 0. A failure caused by a single outlier chunk is labelled by `kv_gate_stats.py`.

Reference points (KLD / same-top):

| Arm | code | prose |
|---|---|---|
| f16 floor | 0.001041 / 99.306% | 0.001699 / 98.230% |
| q8_0 | 0.001025 | 0.001610 |
| turbo5p | 0.001631 | 0.002550 |
| fake-quant turbot design (7-bit young 16K, fq_3t0 old) | 0.001102 / 99.273% | 0.001719 / 98.206% |

Further quality arms, each against the same bar:

| Arm | Procedure | Pass |
|---|---|---|
| upstream Hadamard | `LLAMA_TURBOT_ATTN_ROT=1`, logs `*_turbot_rot.log` | reported; the default changes only if this arm is at least as good |
| 131K depth | wikitext-2 prose at `-c 131072`, PPL | within noise of f16 5.5193 / q8_0 5.5191 / turbo5p 5.5217 (2026-09-03) |
| needles | `niah.py` at 32K / 131K / 200K / 250K, 1 and 4 agents. `niah.py` hardcodes port 8080: run a copy with `U = "http://127.0.0.1:8091"` | retrieval equal to turbo5p |
| DFlash2 acceptance | `acceptab.py` turbot arm against turbo5p (drafter cache must log turbo5p) | acceptance >= turbo5p − 0.02 |
| vision | mmproj image prompts from the vision checks | answers equal to turbo5p in content |

---

## 6. Speed and VRAM

Rebuild `llama-bench` first (rule 0.4). llama-bench has no `--kv-tier-plan` flag: the plan comes from `LLAMA_TURBOT_PLAN`.

```powershell
$env:LLAMA_TURBOT_PLAN = "WT\docs\turbot\plans\turbot-default.plan"
BIN\llama-bench.exe -m D:\Projects\LocalAI\models\Qwen3.8-27B-UD-Q5_K_XL.gguf -ngl 99 -fa 1 -ctk turbot  -ctv turbot  -b 2048 -ub 1280 -t 16 -r 2 -p 2048 -n 32 -d 0,16384,65536,131072,245760 -o csv > E:\kv-turbot\bench_turbot.csv
BIN\llama-bench.exe -m D:\Projects\LocalAI\models\Qwen3.8-27B-UD-Q5_K_XL.gguf -ngl 99 -fa 1 -ctk turbo5p -ctv turbo5p -b 2048 -ub 1280 -t 16 -r 2 -p 2048 -n 32 -d 0,16384,65536,131072,245760 -o csv > E:\kv-turbot\bench_turbo5p.csv
```
Alternate turbot and turbo5p runs in pairs, two rounds each.

**Pass** (the plan's speed targets, SPEC 7.8):
- tg and pp at 131K and 245K within noise of turbo5p or better;
- a small loss at 0 and 16K is accepted (every cell is young below 65K, which reads 1.42× the bytes).

Then the server numbers:
- `acceptab.py` / `bench_ab.py` with DFlash2 on port 8091, turbot arm against turbo5p.
- Quote realistic t/s (about 110 in real use, not the greedy 227).
- Compare ms per step when outputs differ.

**VRAM:**
- the log `size = 5246.00 MiB` (turbo5p 5248.00 MiB);
- `nvidia-smi` used memory at 262,144 cells, 4 slots, no more than the turbo5p server's;
- no F16 FA scratch allocation in the log.

---

## 7. Results

### Recorded on CPU, 2026-09-15 (implementer E)

**`check_tables_vs_study.py`: PASS.**
- The header matches a fresh generation.
- All structural checks pass.
- Against the study: b = 4 and 5 identical bit for bit. Worst absolute difference b = 2: 1.19e-7, b = 3: 1.94e-7, b = 6: 4.97e-5.

**`plan_vram.py` on the default plan:**

| Quantity | Value |
|---|---|
| base | 4520.00 MiB |
| young pool | 726.00 MiB |
| total | 5246.00 MiB |
| margin to turbo5p | 2.00 MiB |
| old bits | mean 4.289, sum 549 |
| hash | `0x56c3503c949a7749` |

**`turbot_plan.py convert`:**

| Source | Result |
|---|---|
| `fq_3t0_w256_m16384_7b.txt` | identical to the default plan (same hash) |
| alloc3 entry 0 | identical to the default plan (same hash) |
| alloc3 entry 1 | refused: `b2 = 6` is not above its 6-bit heads |
| alloc3 entry 1 `--young 7` | 5378.00 MiB (above turbo5p by 130 MiB) |
| alloc3 entry 2 | 5138.00 MiB |
| alloc3 entry 3 | 5868.00 MiB, CAP 4096 |

### To fill at integration

| Gate | Result |
|---|---|
| 1b test-turbot (fill table: worst fill-vs-old, fill-vs-true) | |
| 2a memcheck | |
| 2b FA / writer cases, OLD_I8 1 and 0 | |
| 2c test-turbot-backend, OLD_I8 1 and 0 | |
| 2d validate.ps1, codegen identity | |
| 3 B0 table and decision | |
| 4.1-4.12 | |
| 5 KLD / same-top table, CI verdicts | |
| 6 llama-bench table, server t/s, VRAM | |

SPEC 4.4 expectation for the fill table (sphere Monte Carlo): worst MSE(fill, old)/D_b 0.200 and MSE(fill, true)/D_b 1.198, both at b2 y3.

### Integrated results, 2026-09-15

| Gate | Result |
|---|---|
| 1b test-turbot | 15,288 checks, 0 failed |
| 2a memcheck | FA, writer and bytes: 0 errors |
| 2b FA / writer | 195/195 (also balance seeds 1/2/3, STRIPE=0, BALANCE=0), 28/28 |
| 2c test-turbot-backend | OK (0 failures); gain check skips groups whose codes differ on a float threshold tie |
| 2d existing types | 0 new failures vs a fresh build of the base commit (both fail the same 64 hsk=40 f16 cases, a pre-existing issue in main) |
| 3 B0 (turbot / turbo5p per op) | G1 1.159, G2 1.125, G3 0.872, G4 1.242, G5 1.158 |
| 5 KLD / same-top, 16 x 32K | code 0.001137 / 99.267%, prose 0.001848 / 98.174% |
| 5 DFlash2 acceptance, matched prompts | single prose 0.451, code 0.764, multi 0.629, fanout 0.693, 131K 0.788 (turbo5p 0.440 / 0.771 / 0.638 / 0.666 / 0.783) |
| 6 llama-bench tg64 / pp512 | 0: 61.39 / 3437, 131K: 50.07 / 1502, 245K: 43.78 / 1003 (turbo5p 57.77 / 3463, 47.17 / 1605, 40.37 / 1091) |
| 6 DFlash2 server ms/step | 512: 23.07, 32K: 24.46, 131K: 27.04, 200K: 28.72 (turbo5p 22.52 / 23.55 / 26.20 / 28.20) |
| 6 VRAM | 5,246.00 MiB at 262,144 cells (turbo5p 5,248.00) |

