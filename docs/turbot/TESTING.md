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
| `python docs\turbot\gen_turbot_default_plan.py --check` ([TAG_TURBOT_EMBED_PLAN]) | prints `llama-turbot-default-plan.h matches` and exits 0: `src\llama-turbot-default-plan.h` holds the current plan. After an edit to the plan, run it without `--check` to regenerate the header. CMake configure stops on the same difference |

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
| 7a' built-in plan ([TAG_TURBOT_EMBED_PLAN]) | the built-in text is LF with hash `0x56c3503c949a7749`. `llama_turbot_plan_matches` is true for the Qwen3.8 layers (also at kv 4096). It is false for Spark's 9 layers (the reason names layer 39), a shifted list, 17 layers and an empty list. `default` resolves to the built-in plan and `./no-such-dir/default` to a file |
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

nb 1 and nb 2 run on the <4,8> instance (SPEC 7.6, Q ≤ 2 route). Run the FA command a second time with `$env:TURBOT_Q2_ROUTE = "0"`, which puts nb 2 back on <2,8>, and then remove the variable. Both runs must pass. With `LLAMA_TURBOT_FA_DEBUG=1` the nb 2 shapes print `ncols1=4 ncols2=8` and, with an explicit mask at kv >= 4096, `kv_scan=turbot<ncols1> wrap`.

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

- `powershell -File D:\Projects\LocalAI\source-build\validate.ps1 -BuildDir WT\build-turbot -Tag turbot`: must print `GATE PASSED` (exit 0; exit 2 means the preflight refused and nothing ran).
  - **Preflight.** It refuses while any llama, ggml, test-backend, test-turbot or compute-sanitizer process is alive, or after an nvlddmkm event since boot. It never stops a process it did not start.
  - **Part 1.** The full `test-backend-ops` suite, including the f16, q8_0, turbo4p and turbo5p cases. FAIL lines for cases in `-KnownFails` (default: the saved base-commit log with the 64 hsk=40 cases) are reported without failing the gate.
  - **Part 2, end-to-end smoke test.** One server per KV arm. The arms are **turbo4, turbo5p and turbot** (`-SmokeKv`, default `turbo4,turbo5p,turbot`). Each runs on port 8091 with Qwen3.8-27B-UD-Q5_K_XL, `-c 32768 --kv-unified -fa on`, and is probed by `thresh.py` (20 exact prompt lengths, number-sequence continuation) and `difflen.py` (10 prefill lengths up to 28,000 tokens, needle recall).
  - **turbot plan.** The turbot arm gets `--kv-tier-plan`: `-TurbotPlan`, else the first `turbot-default.plan` next to the build, else the deployed one.
  - **Environment.** Every child runs with `GGML_DISABLE_VULKAN=1` and `--load-mode none`. Pass `-LoadModeFlag '--no-mmap'` only for a build older than `--load-mode`.
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

Q ≤ 2 route (SPEC 7.6), an A/B on one binary. Route off (nb 2 on <2,8>), then the default (nb 2 on <4,8>):
```powershell
$p = "(turbot_perf=l23|turbot_ref=turbo5p),kv=(131072|245760),nb=2"
$env:TURBOT_Q2_ROUTE = "0"; BIN\test-backend-ops.exe perf -b CUDA0 -o FLASH_ATTN_EXT -p $p > E:\kv-turbot\q2_off.log
Remove-Item Env:TURBOT_Q2_ROUTE; BIN\test-backend-ops.exe perf -b CUDA0 -o FLASH_ATTN_EXT -p $p > E:\kv-turbot\q2_on.log
```
Keep the route only if every nb 2 turbot cell is at most as slow with it on. Confirm end to end with `llama-bench -ctk turbot -ctv turbot -p 2 -n 0 -d 131072,245760 -r 5` under both settings.

---

## 4. Full-stack correctness (A-D integrated)

Every server below runs `-ctk turbot -ctv turbot --kv-tier-plan WT\docs\turbot\plans\turbot-default.plan` on port 8091, one at a time. On the `upstream-sync` branch the plan flag is optional: without it the built-in plan is used, and it has the same hash.

REVISED 2026-09-21 ([TAG_HARNESS_SRVCMD]): the scratchpad harnesses (`acceptab.py`, `overcommit_accept.py`, `pool_exact.py`, `crosstalk.py`, `niah.py` and others) no longer read Jarvis. They build their command with `<scratchpad>\srvcmd.py`.
- The default preset, `qwen38-prod`, is the production command with turbot K and V.
- The build is chosen with `BENCH_SERVER_EXE=old|new|<path>` or `--exe`. `new` is `WT\build-sync\bin`: it gets `--load-mode none`, `--device CUDA0` and `--spec-draft-device CUDA0`, and uses the built-in plan.
- `BENCH_KV` or `--kv` sets another cache type, and `BENCH_TURBOT_PLAN` or `--plan` passes a plan file.
- srvcmd refuses to start while any llama, ggml or test-backend process runs, and it refuses port 8080.
- `<scratchpad>\README_harness.md` has the command for each harness on each build. `LLAMA_TURBOT_PLAN` can still carry the plan instead of the flag.

| # | Test | Command / procedure | Pass |
|---|---|---|---|
| 4.1 | tier unit tests | section 1b item 7 on the integrated tree | `OK` |
| 4.2 | refusals | start `llama-server` with `-ctk turbot` alone; `-np 4` without `--kv-unified`; `-fa off`; `TURBO_KV_CPU_LAYERS=1`; `TURBO_LAYER_ADAPTIVE=1`; `TURBO_INNERQ=1`; `--kv-tier-plan` with a plan missing one `L` line | REVISED 2026-09-21 ([TAG_KV_RESOLVE], `upstream-sync`): each one starts. It logs one `KV cache type for ...: turbot -> ... (reason)` warning, and the size line shows the fallback: turbo5p in most cases; q8_0 K with the default f16 V for `-ctk turbot` alone; K q8_0 and V f16 with `-fa off`. Greedy output is coherent. Run the same list again with `LLAMA_KV_RESOLVE=0`: each exits with a message starting `turbot:` (or the plan message of SPEC 9.1). Pre-sync builds: the `LLAMA_KV_RESOLVE=0` behaviour. No crash, no CUDA error |
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
| 4.13 | built-in plan ([TAG_TURBOT_EMBED_PLAN], `upstream-sync`) | start with no `--kv-tier-plan` and no `LLAMA_TURBOT_PLAN`; then with `--kv-tier-plan default`; then with `LLAMA_TURBOT_PLAN=default` | the first logs `turbot: no --kv-tier-plan or LLAMA_TURBOT_PLAN given, using the built-in default plan (docs/turbot/plans/turbot-default.plan, calibrated on Qwen3.8-27B)`, the others `turbot: --kv-tier-plan default: ...` / `LLAMA_TURBOT_PLAN=default: ...`. All three show `turbot plan <built-in default>: 16 layers, ... hash 0x56c3503c949a7749` and `size = 5246.00 MiB`, as in 4.4 |

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
| needles | `niah.py` at 32K / 131K / 200K / 250K, 1 and 4 agents. `niah.py` now talks to port 8091. Pass `--start` (or a build option such as `--exe new`) so that it starts its own server through srvcmd. Without it, niah.py sends its requests to whatever already listens on 8091, which can be a running gate's server | retrieval equal to turbo5p |
| DFlash2 acceptance | `acceptab.py` turbot arm against turbo5p (drafter cache must log turbo5p) | acceptance >= turbo5p − 0.02 |
| vision | mmproj image prompts from the vision checks | answers equal to turbo5p in content |

---

## 6. Speed and VRAM

Rebuild `llama-bench` first (rule 0.4). llama-bench has no `--kv-tier-plan` flag: the plan comes from `LLAMA_TURBOT_PLAN`, and without it from the built-in default plan ([TAG_TURBOT_EMBED_PLAN]).

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

Known limits, 2026-09-15 build:
- Prefill is 4-8% slower than turbo5p at 131K-245K.
- A cached long prompt can decode slightly differently from the same prompt sent cold.
- Two-token verify batches ran the <2,8> instance. On the upstream-sync branch they run <4,8> (SPEC 7.6, `TURBOT_Q2_ROUTE=0` to compare). That route is not measured yet; see the Q ≤ 2 route A/B in section 3.

---

## 8. upstream-sync branch and other models

Section 8 covers the branch `upstream-sync` in the worktree `D:\Projects\LocalAI\source-build\turboquant-sync` (called `WS`), built in `WS\build-sync` (its `bin` folder is `BIN` in this section). The branch adds:
- upstream `fb34fc262` (`b11093`), merged;
- the KV type resolver ([TAG_KV_RESOLVE]);
- the built-in turbot plan ([TAG_TURBOT_EMBED_PLAN]);
- the turbo5p512 D = 256 MMA kernel ([TAG_TURBO5P512_MMA]);
- the vision device choice ([TAG_MMDEV_TYPE]);
- the new switches of the README section "Using the fork with any model".

**None of it has been built or run yet.** Every result in 8.7 is still to be filled in. The rules of section 0 apply to every step.

### 8.1 Build

- Use a fresh build directory. CUDA objects have no depfiles, so a reused directory can keep stale kernels.
- Use the `build-turbot` options, with two changes:
  - drop `GGML_CUDA_FA_ALL_QUANTS`, which was removed upstream;
  - add `-DGGML_VULKAN=ON` only for the iGPU vision device. That needs the Vulkan SDK, and `GGML_BACKEND_DL` stays `OFF`.
- Targets: `ggml llama llama-server llama-perplexity llama-bench llama-mtmd-cli test-backend-ops test-turbot test-turbot-backend test-kv-resolve test-chat test-llama-archs`.
- A Vulkan build lists the 5090 twice (`CUDA0` and `Vulkan0`). Run every step below with `GGML_DISABLE_VULKAN=1` unless it is an iGPU step (8.5). validate.ps1 sets it for its own children.
- Model loading is `--load-mode none` everywhere. The synced build rejects `--no-mmap`, and llama-bench no longer has `-mmp`.

### 8.2 CPU gates

| Command | Pass |
|---|---|
| `python docs\turbot\gen_turbot_default_plan.py --check` | `llama-turbot-default-plan.h matches` |
| `BIN\test-turbot.exe` | `OK`, item 7a' included (1b) |
| `BIN\test-kv-resolve.exe` | Exit 0, and the last line reads `<n> checks, 0 failed`. The test covers three things: the shared turbot refusal messages, the built-in plan against the Qwen3.8 and Spark layer lists, and the resolve table in the README section "KV cache type, resolved per model" (synthetic hparams). |
| `BIN\test-chat.exe`, run from `WS` | Exit 0. This includes the Muse Glimmer case where the reply starts with a tool call (#29242). |

### 8.3 CUDA correctness

```powershell
$cs = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\compute-sanitizer.bat"
& $cs --tool memcheck --error-exitcode 99 BIN\test-backend-ops.exe test -b CUDA0 -o FLASH_ATTN_EXT -p "^split_plane=turbo5p512,.*,kv=1000,"
BIN\test-backend-ops.exe test -b CUDA0 -o FLASH_ATTN_EXT -p "^split_plane=" > E:\kv-turbot\sync\split_plane.txt
BIN\test-backend-ops.exe test -b CUDA0 -o GDN_L2_NORM > E:\kv-turbot\sync\gdn_l2_norm.txt
BIN\test-backend-ops.exe test -b CUDA0 -o SSM_SCAN,SSM_SCAN_ROLLBACK > E:\kv-turbot\sync\ssm_scan.txt
```

**Pass:**
- Memcheck: `ERROR SUMMARY: 0 errors`.
- Every case prints OK and the exit code is 0.
- `split_plane` ([TAG_TURBO5P512_MMA]):
  - The turbo5p512 cases cover 2 × 256 heads (512-element rows) and 4 × 256 heads (1024-element rows, the case that used to produce garbage). All must pass at nb 1 to 512, with positional masks, GQA 8, softcap and kv 1000.
  - The turbo5p cases at the Qwen3.8 geometry are the control. If only the control fails, suspect the test or its 5e-4 NMSE bound, which is not yet measured, rather than the kernel.
  - The nb 512 cases take seconds each on the CPU reference.
- `GDN_L2_NORM`: the fused kernel ([TAG_RMSNORM_SCALE_FUSION]) must pass. Run it again with `$env:TURBO_RMSNORM_SCALE_FUSION = "0"`; both runs must pass.
- `SSM_SCAN` includes the state size 96 cases of #28717 (Nemotron 3 Puzzle).
- validate.ps1 (2d) on this build:
  - Must print `GATE PASSED`.
  - The 64 hsk=40 f16 cases in its known-fails baseline should now pass, because the fattn-tile KV_max pair index is fixed. Record whether they do.
  - The KV_min fixup `test_flash_attn_ext_pos` cases (hs 256, kv 4096, nb 1280/2048, f16, q8_0, turbo5p) are part of the full suite.

### 8.4 KV resolver on real models

Start one server per row, one at a time, through srvcmd:
```powershell
python <scratchpad>\srvcmd.py serve --exe new --preset generic --model D:\Projects\LocalAI\models\<file> --kv turbot --ctx 32768 --parallel 1
```
Then send one short greedy prompt. Repeat each row with `--env LLAMA_KV_RESOLVE=0`.

**Pass:**
- The server starts.
- The `llama_kv_cache: size` line shows the expected type.
- A downgrade logs one `KV cache type for ...` warning that names the reason.
- The greedy output is coherent, and there is no CUDA error.
- With `LLAMA_KV_RESOLVE=0`, every row except the first two exits with the old `turbot:` refusal.

| Model file | Arch, KV shape | Expected with `--kv turbot` (from the rules, not yet run) |
|---|---|---|
| Qwen3.8-27B-UD-Q5_K_XL | qwen35, 16 attention layers, 4 × 256 | turbot, no warning |
| TURBO-Qwen3.8-27B-NEO-CODER-MAX-Q5_K_M | qwen35 fine-tune, same layers | turbot with the built-in plan. Its quality with this plan is not measured: run a KLD against turbo5p before using it. |
| Spark-X2.5-4B-Q8_0 | spark2_5, iSWA, 4 × 256 | turbo5p (`SWA caches are unsupported`) |
| Ornith-1.5-9B-Q8_0 | qwen35, 32 layers, 4 × 256 | turbo5p: the built-in plan names attention layers that this model does not have |
| Ornith-1.5-35B-Q4_K_M | qwen35moe, 2 × 256 | turbo5p512 (512-element rows), now read natively by the MMA kernel. Compare with `TURBO_MMA_NATIVE=0` |
| MiniCPM5-2B-Q8_0 | 2 × 128 (256-element rows) | turbo4 |
| Muse-Glimmer-30B-KQuant-17GB-Q4_K_M | SWA, 2 × 128 | turbo4 |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0 | nemotron_h MoE, 2 × 128 attention | turbo4 |

Also run each row with `--kv auto` (no cache type) as a baseline: it must start with f16 and no resolver line. Then run `-np 4` without `--kv-unified` on Qwen3.8: it must log turbot → turbo5p (`single KV stream`).

New architectures without a model on disk (maple, hy_v4, hrm_text):
```powershell
BIN\test-llama-archs.exe -a "spark2_5|maple|hy_v4|hrm_text"
```
This test builds a small random-weight model for each architecture and runs it on the backends, so it uses the GPU (rule 0.1). Pass: every listed architecture passes. The DeepSeek-V4 vision projector (`deepseek4v`) and Nemotron 3 Puzzle have no local check.

### 8.5 Vision device

Pinning rule (README, "Vision: pick the device"): `--device CUDA0 --spec-draft-device CUDA0 -mmdev cpu|gpu|igpu`.
- `cpu` or `gpu`: set `GGML_DISABLE_VULKAN=1`.
- `igpu`: set `VK_LOADER_DRIVERS_SELECT=*amd-vulkan64*`, with `GGML_DISABLE_VULKAN` unset.

srvcmd emits both for the new build (`--vision cpu|gpu|igpu`).

| # | Check | Pass |
|---|---|---|
| 8.5.1 | `BIN\llama-server.exe --list-devices`, with `GGML_DISABLE_VULKAN=1` and then with the iGPU loader filter | first: CUDA0 only; second: CUDA0 and one Vulkan device, the AMD iGPU |
| 8.5.2 | production preset (`qwen38-prod`, 4 slots, turbot) with `--vision cpu`, `gpu`, `igpu`, same image prompts as the vision checks of section 5 | the log puts the vision encoder on the chosen device and the model and drafter on CUDA0 only; answers match the cpu arm in content; 5090 VRAM with `cpu` or `igpu` equals the server without vision; the encode time per image is recorded |
| 8.5.3 | a corrupt image in one request while other slots generate | that request fails, the server keeps serving the others ([TAG_MTMD_ENCODE_CATCH]) |
| 8.5.4 | `-mmdev igpu` with `GGML_DISABLE_VULKAN=1` | the server starts and logs one warning naming the reason (`no integrated GPU device found`, [TAG_MMDEV_FALLBACK]); the vision encoder runs on the CPU and the image checks of 8.5.2 pass as in the cpu arm |
| 8.5.5 | iGPU ops, with the loader filter set: `BIN\test-backend-ops.exe test -b Vulkan0 -o <op>`, one op at a time, for MUL_MAT, FLASH_ATTN_EXT, ROPE, IM2COL, UPSCALE, NORM, ADD, MUL, UNARY, CPY, CONT, SOFT_MAX | 0 FAIL; F16 flash attention at head size 72 supported; turbo FA cases report `not supported` ([TAG_VK_NO_TURBO]) |
| 8.5.6 | `--mmproj-threads 8` against `0` (= `-t`) with `-mmdev cpu` | encode time recorded; answers unchanged |
| 8.5.7 | production preset with `--vision cpu`: one 4000-token image request while 3 slots stream text, then the same with `MTMD_ASYNC_ENCODE=0` ([TAG_MTMD_ASYNC_ENCODE]) | the log says `media is encoded on its own thread`; the 3 streams keep producing tokens during the encode (record their t/s and the longest gap between two tokens), while with `=0` they stall for the whole encode; `/slots` shows `waiting_media: true` for the image slot during the encode; the image answer matches the `=0` arm in content |
| 8.5.8 | cancel the image request of 8.5.7 during its encode, then send a text request and a second image request; stop the server during a third encode | the slot is free at once and the text request is served during the encode; the log shows `request gone, result dropped`; the second image is encoded after the first encode ends; the stop waits for the running encode and exits with no crash |

### 8.6 Switch A/Bs (one binary, default against the switch)

| Switch | A/B | Keep the default when |
|---|---|---|
| `TURBOT_Q2_ROUTE=0` | 2b second run and the section 3 nb 2 A/B | nb 2 is at most as slow with the route on |
| `TURBO_RMSNORM_SCALE_FUSION=0` | `BIN\llama-bench.exe -m D:\Projects\LocalAI\models\Qwen3.8-27B-UD-Q5_K_XL.gguf -ngl 99 -fa 1 -ctk turbot -ctv turbot -lm none -p 512 -n 64 -d 0,131072 -r 3`, and a greedy server prompt | greedy text byte-identical (the fusion is meant to be bit-exact) and tg not slower |
| `GGML_CPU_FA_DV_PAD=0` | `-mmdev cpu`, same images | answers equal in content; encode time lower with the default |
| `MTMD_CPU_KV_F32=0` | `-mmdev cpu`, same images | answers equal in content; encode time not higher with the default |
| `LLAMA_CTX_CHECKPOINT_MIN_STEP_ALWAYS=1` | `acceptab.py` multi-agent scenario with DFlash2 | same text and acceptance; with the default there are no extra full re-prefills (count them in the log; `superseding context checkpoint` lines are expected) |
| `TURBO_MMA_NATIVE=0` | Ornith-1.5-35B turbo5p512, `server_depth_bench.py` | output coherent in both; prefill and decode recorded |
| `LLAMA_KV_RESOLVE=0` | 4.2 and 8.4 | the pre-resolver refusals, word for word |
| `SPEC_DFT_DUMP=<file>` | the same greedy DFlash2 prompt on two builds that carry [TAG_SPEC_DFT_DUMP] | matched on (seq_id, pos0, id_last, i), the lattices agree up to the first differing accepted token, and every id is an integer |

### 8.7 Results (to fill)

| Step | Result |
|---|---|
| 8.2 gen --check, test-turbot, test-kv-resolve, test-chat | not yet run |
| 8.3 split_plane, GDN_L2_NORM, SSM_SCAN, memcheck, validate.ps1 (hsk=40 cases) | not yet run |
| 8.4 real models, resolved types and warnings, `LLAMA_KV_RESOLVE=0` | not yet run |
| 8.4 test-llama-archs for the new architectures | not yet run |
| 8.5 vision device, VRAM, encode time per device | not yet measured |
| 8.6 switch A/Bs | not yet measured |
| NEO-CODER-MAX turbot KLD against turbo5p | not yet measured |

---

## 9. Any-model gates (`[TAG_TURBOT_ANY_*]`, SPEC 14)

Section 9 covers turbot on other shapes (SPEC section 14):
- the geometry flags, the S2..S7 types, the NR-run coder and the CPU reference, with `test-turbot-geom` (WP1);
- the CUDA reader and writer for the new geometries, with their `test-backend-ops` / `test-turbot-backend` cases, `b0_gate.py` and `sass_diff.py` (WP2);
- the host cache and tier: NR-run plans, the automatic plan, the sidecar and per-stream tiers, with `test-turbot`, `turbot_plan.py`, `blob_roundtrip.py` and `turbot_guard.py` (WP3);
- the resolver, the iSWA split and the graph, with `test-kv-resolve` (WP4).

It runs on the `upstream-sync` worktree `WS` after the orchestrator's single integration build (the packages merge in the order WP1, WP2, WP3, WP4), in the build directory of section 8.1 (`BIN`).

**None of it has run yet.** Run the gates in order. A failed gate stops the sequence until it is understood. G2 is a hard gate: nothing after it counts if Qwen3.8-27B changed. The rules of section 0 apply to every step.

### 9.0 G0: preflight

1. **One GPU process** (rule 0.1). Any llama, ggml or test-backend process means wait:
   ```powershell
   Get-Process | Where-Object { $_.ProcessName -match 'llama|ggml|test-backend' }
   ```
2. **Driver faults first** (rule 0.2). Look for `nvlddmkm` event 153 before anything else. If there is one, reboot before any GPU step:
   ```powershell
   Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='nvlddmkm'; Id=153; StartTime=(Get-Date).AddHours(-24)} -ErrorAction SilentlyContinue
   ```
3. `$env:GGML_DISABLE_VULKAN = "1"` for every step.
4. **Rebuild every tool exe** with the new DLLs (rule 0.4): `test-backend-ops`, `llama-perplexity`, `llama-bench` and `llama-server`, plus `test-turbot`, `test-turbot-geom`, `test-kv-resolve` and `test-turbot-backend`.
5. Servers on port **8091** only. Outputs go to `E:` (for example `E:\kv-turbot\any`).

### 9.1 G1: CPU tests

| Command | Pass |
|---|---|
| `BIN\test-turbot.exe` | `OK` |
| `BIN\test-turbot-geom.exe` | `OK`. Parts (g) CPU flash attention and (h) CPU writer ran, not `SKIPPED`. |
| `BIN\test-kv-resolve.exe` | exit 0, last line `<n> checks, 0 failed` |

### 9.2 G2: Qwen3.8-27B bit-identity (hard gate)

The baseline is a build of commit `80f44b5d8` (the committed tree under the WP changes: `0fc83cc8d` plus `[TAG_FA_SINK_CLAMP]` and the seeded turbot test inputs `[TAG_TURBOT_TEST_SEEDED]`), from a fresh build directory `BASE`. Every item must hold:

| # | Check | Pass |
|---|---|---|
| 2.1 | `python tools\turbot\sass_diff.py BASE\bin\ggml-cuda.dll BIN\ggml-cuda.dll --arch sm_120a` (CPU only, cuobjdump) | exit 0: `IDENTICAL` for the 40 `flash_attn_ext_turbot<256,256,...>` kernels (20 instances, both softcap variants), `k_turbot_set_rows<int>` and `<int64_t>`, `k_turbot_fill` and `flash_attn_turbot_balance_bounds<4>`; `--scope turbot` for the full turbot list |
| 2.2 | `BIN\test-backend-ops.exe test -b CUDA0 -o FLASH_ATTN_EXT -p "^turbot=[a-z0-9]+,kv="`, then `-o TURBOT_SET_ROWS -p "^turbot=[a-z0-9]+,rows="` | 195/195, with case names identical to the baseline's `-p "^turbot="` run; the Qwen writer cases pass. The plain `^turbot=` filter of sections 2b and 8 now also selects the new geometry cases, whose names carry `d=..,hkv=..,hq=..` after the widths. |
| 2.3 | `BIN\test-turbot-backend.exe` on both builds | bytes identical |
| 2.4 | `validate.ps1` (section 2d) | `GATE PASSED` |
| 2.5 | production server: 262K, 4 slots, `--kv-unified`, turbot, DFlash2 n_max 3, port 8091 | the log shows `turbot plan <built-in default>`, hash `0x56c3503c949a7749`, and the same size lines as the baseline (`size = 5246.00 MiB ... young pool:  726.00 MiB`) |
| 2.6 | greedy transcripts: code, prose, a tool call, and 4 concurrent unique-value prompts | identical to the baseline build |
| 2.7 | KLD 16 × 32K (section 5 commands, `-ub 512`) | code **0.001139** and prose **0.001856**, exactly |
| 2.8 | `llama-bench` tg64 at `-d 0,131072,245760` | within noise of the baseline |

### 9.3 G3: new-kernel correctness

- Every new `test-backend-ops` case passes against the CPU reference:
  - FA at D 256 with GQA 4, GQA 1, 2 heads and 1 head;
  - FA at D 128 with 8, 4 and 2 heads, including GQA 7 and 16;
  - the NR 1 and NR 2 writers, and the NR 4 writer at D 128.
  ```powershell
  BIN\test-backend-ops.exe test -b CUDA0 -o FLASH_ATTN_EXT -p "^turbot=[a-z0-9]+,d=" > E:\kv-turbot\any\g3_fa.txt
  BIN\test-backend-ops.exe test -b CUDA0 -o TURBOT_SET_ROWS -p "^turbot=[a-z0-9]+,d=" > E:\kv-turbot\any\g3_writer.txt
  ```
- `compute-sanitizer` memcheck and synccheck on the small cases (the command pattern of 8.3, with `--tool memcheck` and `--tool synccheck`), for example `-p "^turbot=[a-z0-9]+,d=.*,kv=(96|100),nb=(1|2|4),"` for FA and `-p "^turbot=[a-z0-9]+,d=.*,rows=(1|4|16),"` for the writer: `ERROR SUMMARY: 0 errors` for both.

### 9.4 G4: kernel speed (B0 style)

`turbot_perf` at nb 1 and nb 512, each geometry against its reference:

| Geometry | Reference |
|---|---|
| D 128, 8 heads | turbo5p |
| D 256, 4 heads, GQA 4 | turbo5p |
| D 256, 2 heads | turbo5p512 (turbo5p cannot hold a 512-value row) |
| D 128, 2 heads, GQA 8 and GQA 16 | turbo4 |

```powershell
BIN\test-backend-ops.exe perf -b CUDA0 -o FLASH_ATTN_EXT -p "turbot_perf=[a-z0-9]+,d=|turbot_ref=[a-z0-9]+,d=" > E:\kv-turbot\any\g4.log
python tools\turbot\b0_gate.py E:\kv-turbot\any\g4.log
```

**Pass:** turbot is no slower than the reference at both nb (`b0_gate.py` prints `G4 <geometry>: VALIDATED`; `--g4-limit` changes the 1.00 bar). A geometry that fails leaves the VALIDATED list (SPEC 14.11).

### 9.5 G5: per-model quality

Run with `turbot_guard.py`, one model at a time. It refuses to start while a GPU process runs, writes under `E:\turbot-guard\<model>` and deletes the `.dat` logits at the end unless `--keep`:
```powershell
python tools\turbot\turbot_guard.py --selftest
python tools\turbot\turbot_guard.py --exe-dir BIN --model D:\Projects\LocalAI\models\Ornith-1.5-9B-Q8_0.gguf
python tools\turbot\turbot_guard.py --exe-dir BIN --model D:\Projects\LocalAI\models\MiniCPM5-2B-Q8_0.gguf --budget turbo5p --allow-larger
```
Without `--plan` it measures the automatic plan of the production shape (`--plan-ctx 262144 --plan-np 1`; add `--plan-unified` for a `--kv-unified` server). `--sidecar` also writes the stamped `<model>.turbot.plan` next to the model after a pass. The NR 1 opt-in arms need `--budget turbo5p --allow-larger`, because that plan is larger than turbo4.

**Setup.** f16 base logits, 32K × 8 chunks, code and prose corpora, `-ub 512`.

**Disk space.** Check `E:` first. The base files take about 65 GB per corpus for the ~248K vocabulary of the Ornith models (qwen35 family), and about 34 GB for 131K vocabularies. Delete them after each model.

| Model | turbot arm | Against |
|---|---|---|
| Ornith-1.5-9B-Q8_0 | turbot (automatic plan) | turbo5p |
| Spark-X2.5-4B-Q8_0 | turbot (automatic plan on the full-attention layers, SWA on turbo5p) | turbo5p |
| Ornith-1.5-35B-Q4_K_M | turbot (automatic plan, NR 2) | turbo5p512 |
| MiniCPM5-2B-Q8_0 | turbot with `LLAMA_TURBOT_AUTO_BUDGET=turbo5p` | turbo4 |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0 | turbot with `LLAMA_TURBOT_AUTO_BUDGET=turbo5p` | turbo4 |
| Muse-Glimmer-30B-KQuant-17GB-Q4_K_M | turbot with `LLAMA_TURBOT_AUTO_BUDGET=turbo5p` | turbo4 |

turbo4 is the reference for the three 2 × 128 models because turbo5p cannot hold a 256-value row.

**Pass, per corpus:**
- the paired chunk-bootstrap 95% CI of mean KLD (turbot − fallback) lies entirely below 0;
- the same-top lower bound is ≥ −0.05 points;
- p99.9 KLD is no worse.

If the CI crosses 0, go to 16 chunks.

**Depth arm.** 131K single-chunk PPL for Ornith-1.5-9B, Ornith-1.5-35B and Spark-X2.5-4B (PPL rather than KLD, because of the RAM spike of rule 0.6). Pass: turbot no worse than the fallback within noise.

Every arm must log its plan and hash lines. An arm whose cache silently resolved to another type fails.

### 9.6 G6: per-model speed and VRAM

For each model of G5, turbot against its fallback, three repetitions:
```powershell
BIN\llama-bench.exe -m D:\Projects\LocalAI\models\<file> -ngl 99 -fa 1 -lm none -ctk turbot -ctv turbot -p 2048 -n 64 -d 0,32768,131072 -r 3 -o csv > E:\kv-turbot\any\<model>_turbot.csv
BIN\llama-bench.exe -m D:\Projects\LocalAI\models\<file> -ngl 99 -fa 1 -lm none -ctk <fallback> -ctv <fallback> -p 2048 -n 64 -d 0,32768,131072 -r 3 -o csv > E:\kv-turbot\any\<model>_fallback.csv
```
- `<fallback>` is `turbo5p` (the cache takes turbo5p512 by itself for 512-value rows) or `turbo4`.
- The turbot arm of the NR 1 models runs with `$env:LLAMA_TURBOT_AUTO_BUDGET = "turbo5p"`.

**Pass:**
- tg64 and pp2048 of turbot ≥ the fallback at every depth;
- for automatic plans, the `llama_kv_cache: size` line shows turbot ≤ the fallback;
- for the NR 1 opt-in, record the extra MiB (sizing at 262K: +316 MiB MiniCPM5, +98 MiB Muse, +45 MiB Nemotron).

### 9.7 G7: server smoke

For each promoted model, on port 8091:
- facts and a tool call;
- vision with the Ornith and Muse mmproj files;
- 4 concurrent unique-value prompts with no cross-slot leak;
- `blob_roundtrip.py` slot save and restore.

Qwen3.8-27B with `-np 4` and no `--kv-unified`:
- the log shows turbot on 4 streams;
- each prompt's greedy output equals its single-stream output;
- `blob_roundtrip.py` passes, every arm with one KV stream per slot, including `streams_equal` (unified slot 0 against stream 2):
  ```powershell
  python tools\turbot\blob_roundtrip.py --exe BIN\llama-server.exe --np 4 --non-unified
  ```

Context checkpoints in slot files (`[TAG_SLOT_FILE_CKPT]`, `tools/server/server-context.cpp`), on Qwen3.8-27B:
- `python tools\turbot\blob_roundtrip.py --exe BIN\llama-server.exe --only save_restore`: `PASS`, the detail line shows `prompt_n` about 4 after the restore (was the full prompt) and identical tokens. The server log shows `[TAG_SLOT_FILE_CKPT] saved K context checkpoints` and `restored K of K context checkpoints ... draft state restored`.
- the same arm with `LLAMA_SLOT_FILE_CKPT=0` in the server environment: `prompt_n` is the full prompt again (the arm fails, as before this change), and the file size equals the old `n_written`.
- a file saved with `LLAMA_SLOT_FILE_CKPT=0`, restored on a default server: no warning, full prompt.
- the last byte of a saved file changed: the restore returns 200, the log says `checkpoint section ignored (... fails its checksum)`, the next request processes the full prompt, `/health` stays OK.
- a file saved with `LLAMA_CTX_CHECKPOINT_BUDGET_MIB=0`, restored on a default server: the log says `restored 13 of N`.
- a full run without `--only`: `RESULT: 6/6 arms passed` (the five arms of 4.9 plus `streams_equal`).

### 9.8 G8: set the defaults from the data

- The VALIDATED list becomes the geometries whose models passed G4-G7. Expected: {256 × 4, 256 × 2}.
- NR 1 becomes a default only if turbot beats turbo4 on all three NR 1 models in G5 and loses no decode speed in G6. It costs about 11% more KV VRAM than turbo4 at 262K cells (4-bit base rows alone are 6% larger, the young pool adds the rest). Otherwise it stays opt-in.
- Multi-stream and iSWA stay on only if G7 passes.
- A failure flips that switch's default in code (a one-line change). The env switches remain.

### 9.9 Results (to fill)

| Gate | Result |
|---|---|
| G0 preflight, rebuilt tools | not yet run |
| G1 test-turbot, test-turbot-geom, test-kv-resolve | not yet run |
| G2 Qwen3.8-27B bit-identity (SASS, test-backend-ops, bytes, validate.ps1, server log, transcripts, KLD, tg64) | not yet run |
| G3 new-kernel correctness, memcheck, synccheck | not yet run |
| G4 kernel speed per geometry | not yet measured |
| G5 quality per model (KLD CI, same-top, p99.9, 131K PPL) | not yet measured |
| G6 speed and VRAM per model | not yet measured |
| G7 server smoke per model, Qwen `-np 4` streams | not yet run |
| G7 slot-file checkpoints (`save_restore` arm, kill switch, old file, corrupt file, budget) | not yet run |
| G8 VALIDATED list and switch defaults | not yet decided |

