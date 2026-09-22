# turbot: tiered KV cache implementation contract

`[TAG_TURBOT]` is the grep tag for every change. This document is the contract for five parallel implementers (A-E). None of them compiles while implementing, so every name, signature, byte offset and formula below is normative. Whenever the prose and `ggml/include/ggml-turbot.h` disagree, the header wins, and the architect fixes the prose.

Architect-owned files, finished and checked in with this document:

| File | What it is |
|---|---|
| `docs/turbot/SPEC.md` | this contract |
| `ggml/include/ggml-turbot.h` | constants, layouts, op-params packing, the complete reference coder (pure inline C) |
| `ggml/include/ggml-turbot-tables.h` | GENERATED codebook tables (float32 lists as macros) |
| `ggml/src/ggml-cuda/turbot-tables.cuh` | `__constant__` device copies of the same lists, shared by readers and writers |
| `docs/turbot/gen_turbot_tables.py` | deterministic table generator (`--check` verifies the checked-in header) |

Reading order for implementers:
1. Everyone reads sections 1-5.
2. Then each implementer reads their own section:

| Implementer | Section |
|---|---|
| A | 6 |
| B | 7 |
| C | 8 |
| D | 9-10 |
| E | 11 |

3. Everyone reads section 12 (ownership and shared symbols).

---

## 1. Fixed design decisions

These are not reopened. Numbering follows the coordinator brief.

1. **No exact f16 tier.** There are two tiers: OLD (base code) and YOUNG (base code plus refinement).
2. **Base code for every cell.**
   - Width b ∈ [2..6] bits per (layer, KV head, side).
   - Lloyd-Max levels in the WHT-128 domain.
   - One f16 gain per 128-group, corrected for the base reconstruction.
   - Base row bytes per layer-side = 32·Σ_h b + 16.
3. **Young refinement.** CORRECTED 2026-09-15, see 1.1 below. The original "uniform subdivision, mul/add decode" text is withdrawn.
   - Per head young width y ∈ [b+1..8], default 7.
   - The refinement is the index of a Lloyd sub-cell inside the base cell.
   - Young read is ONE 2^y-entry LUT indexed by `(base << r) | refinement`, with r = y − b.
   - One f16 gain per 128-group, corrected for the young reconstruction.
   - Quantised once, from exact values, at write time.
   - Refinement bytes per layer-side = 32·Σ_h (y−b) + 16.
   - Aging is metadata only. There is never a transcode.
4. **Granules.**
   - A granule is a fixed range of 64 contiguous cell indices.
   - find_slot placement does not change.
   - A granule is young iff it owns a young slot.
   - Young pool default: 65,536 cells (1,024 granules).
5. **Policy.**
   - Before compute, every row written by the current ubatch lands in a young granule: allocate a slot, and if the pool is empty, first demote the least-recently-wanted young granule that is not in flight.
   - After a SUCCESSFUL ubatch only, per sequence s: the newest Y_s cells are wanted young.
     - Age is measured in per-sequence write-row stamps, not positions. Removing a sequence's newest rows (rejected drafts) rolls its counter back (9.7).
     - Y_s = min(CAP, ⌊N_eff · n_s / Σn⌋), N_eff = max(0, N_R − 128·n_active). REFINED 2026-09-15: 2 granules of boundary slack per active sequence (9.6). One sequence still gets 16,384.
     - "Least recently wanted" is implemented as: empty owned granules first, then the smallest stamp margin over the owner sequences' cuts (9.6).
   - A granule stays young iff any live cell in it is wanted by any owner sequence, or it is in flight.
   - A failed compute demotes nothing.
   - Draft rows are ordinary rows.
6. **Writer.**
   - One new op per layer-side writes the base code always, and the refinement when the row's granule has a slot.
   - Per-row input: young pool row, or −1 for none.
   - Base tier math is the kvfq coder.
   - The young code is nested and must be at least as good as an independent y-bit Lloyd code, within the study tolerance (section 4.6).
7. **Reader.**
   - Native CUDA FA through the MMA kernels for D=256, at every query count the model uses (Q=1 routed to MMA, verify Q≈4, prefill 512/1280/2048).
   - FA_POS_MASK, kv_pos and q_pos are kept.
   - A lane-uniform OLD/YOUNG branch per granule.
   - size_t for pool addresses.
   - The `shared_memory_limit_raised` size bug is fixed first.
   - Speed target: parity with turbo5p at long context, not a gain (read volume, 7.8). Gate B0 decides before the rest of the integration.
   - Turbot kernels live in their own template-instance TUs. No existing instance TU changes (7.1).
8. **Rotations.** The Q-side WHT and the V inverse rotation are exactly turbo5p's. REFINED 2026-09-15: the upstream Hadamard (`attn_rot_k/v`) is OFF for turbot by default, because every quality number and the per-head widths were measured without it (section 10.3).
9. **Selection.**
   - `-ctk turbot -ctv turbot` (both required).
   - Plan from `--kv-tier-plan <file>` or env `LLAMA_TURBOT_PLAN`. REVISED 2026-09-21 [TAG_TURBOT_EMBED_PLAN]: with neither, the built-in default plan is used when it fits the model; `default` selects it explicitly (section 9.1).
   - Kill switch: `LLAMA_TURBOT=0` falls back to turbo5p with a warning.
10. **VRAM.**
    - base + pool ≤ 5,248.00 MiB at 262,144 cells for the default plan.
    - A size log line.
    - No F16 scratch buffers.
11. **State.**
    - Versioned turbot blob: cell metadata + stamps, base bytes, young pool bytes per cell.
    - Restore keeps young cells young when the pool allows.
    - turbo5p ↔ turbot blobs are refused.
12. **Refusals.** n_stream > 1, K/V shift, seq_add/seq_div, non-CUDA attention layers, TURBO_KV_CPU_LAYERS, TURBO_LAYER_ADAPTIVE, unsupported head dim. Section 9.3 adds TURBO_INNERQ and a few structural ones.
13. **Drafter.** When the target uses turbot, the DFlash2 drafter cache uses turbo5p explicitly.

### 1.1 Correction to decision 3 (coordinator, 2026-09-15)

**The Phase 0b nested coder study measured the old decision-3 scheme ("a_mid") as a failure.**
- Study: `scratchpad/kv_nested_study.py`, log `kv_nested_selftest.log`.
- "a_mid" is a uniform subdivision of the base Lloyd cell with midpoint decode.
- At b=2 its error was 1.65-1.75× that of an independent 7-bit Lloyd code.
- It must NOT be implemented.

**The implemented scheme is "a_lloyd":**
1. **Base.** C_b = kvfq `get_levels(b)`:
   - ggml C4/C5 literal tables for 4 and 5 bits;
   - Gaussian Lloyd-Max / √128 otherwise.
   - Base thresholds are float32 midpoints of adjacent C_b entries.
   - base index j = number of thresholds ≤ u.
2. **Young partition per (b, r).**
   - Work in z = √128·u.
   - For every base cell, run a 2^r-level Lloyd-Max of N(0,1) truncated to the cell.
   - Initialise uniformly. Outer cells use [t₀−2w, t₀] and [t_{L−2}, t_{L−2}+2w], with w = t₁−t₀.
   - Iterate thresholds = level midpoints, levels = truncated-Gaussian centroids, until the maximum level change is < 1e-11 or 20,000 iterations, identical to kv_nested_study.
   - Every (b, y) converges except b=2, y=8, which stops at the cap with change 3.8e-10. The generator prints iterations and final change per (b, y) into the header comment and asserts change < 1e-9.
3. **Young index** = (j << r) | s, where s is the sub-cell index. So base = young >> r by construction.
4. **Young read** = one 2^y LUT (all sub-cell centroids / √128, flattened in cell order). There is no mul/add residual decode. Old read = C_b.
5. **Gains.** Each tier has its own f16 gain, norm / |recon|.
6. **Tables.**
   - Data independent.
   - Generated once in float64, cast to float32.
   - The SAME float32 tables are used by the CPU reference, the CUDA writer and the CUDA reader.
   - A unit test compares them against the study's `build_designs(..., ['a_lloyd'])`.
   - Coverage: b 2..6, y b+1..8.
7. **Young width.** y = 6 failed for every scheme, so the default stays 7.

Measured (study, code131k, layers 23 and 55), young error / independent y-bit Lloyd:

| | b=2 | b=3 | b=4 | b=5 |
|---|---|---|---|---|
| y=7 real data, worst of nmse and gout | ×1.046 | ×1.036 | ×1.011 | ×1.074 |
| y=7 Gaussian analytic | ×1.054 | ×1.054 | ×1.033 | ×1.122 |

Header math, sphere Monte Carlo (16,000 unit-norm Gaussian groups through the WHT, f16 gains; `scratchpad/turbot_review_indep.py`), young error / independent y-bit Lloyd with the same gains:

| | b=2 | b=3 | b=4 | b=5 | b=6 |
|---|---|---|---|---|---|
| y=7 | ×1.044 | ×1.038 | ×1.016 | ×1.082 | ×0.994 |
| y=8 | ×1.050 | ×1.041 | ×1.016 | ×1.089 | ×0.994 |

Old tier ×1.000: the base codes are identical to kvfq. The kill check passes.

---

## 2. Storage representation

### 2.1 Decision

- **Base cache:** a ggml type family `GGML_TYPE_TURBOT_S8 .. GGML_TYPE_TURBOT_S24` (17 types).
  - S = Σ of the 4 old widths of that layer-side.
  - blck_size 1024, type_size 32·S + 16.
  - One K tensor and one V tensor per attention layer; K and V may differ in S.
- **Young pool:** one `GGML_TYPE_I8` tensor per attention layer, shape `[pool_row_bytes, max(N_R, 64)]`. It exists even for POOL 0 (9.4).
  - A pool row is `[K part][V part]`.
- **Per-head widths** travel in 24 bytes of op_params (section 5.3). Kernels never derive head addresses from ggml strides.

### 2.2 Why

**Views and ggml_row_size stay correct.**
- `ggml_row_size(TURBOT_S, 1024) = 32S+16`, exactly the cell row.
- The FA view `[256, n_kv, 4, 1]` uses `row_size(type, 256) = 8S+4` for the head stride. That is the same fractional-block convention turbo5p already uses: `llama-kv-cache.cpp get_k`, `[TAG_TURBO4P_NC_PER_STREAM]`.
- So `ggml_can_mul_mat(k, q)` holds (ne0 256 == 256), and `ggml_new_tensor_impl` size checks pass.
- With raw I8 base tensors, no FA view with ne0 = 256 exists (rows are ~336-688 B, not 1024 B). The FA op would need a new op and new graph plumbing.

**State I/O stays correct.**
- The existing per-layer loop writes `type` + `row_size` + raw rows.
- A turbo5p blob into a turbot cache (or the reverse) fails at the existing "mismatched key type" check.
- Plan differences with identical S are caught by the plan hash in the v2 section (9.10).

**Existing checks keep working.**
- `ggml_is_quantized` is true, so flash attention is forced.
- `get_can_shift` is refused explicitly.
- Nothing in the turbo predicates (`llama_type_is_turbo`, `ggml_cuda_fattn_turbo_reads_native`, VEC routing) matches the new types. Every existing type keeps its behaviour and codegen.

**Why the pool is I8.** The pool is not cell-indexed by ggml (it is slot-indexed) and needs no element semantics. `row_size(I8, n) = n` keeps views and raw I/O trivial.

**Accepted costs.**
- `to_float` / `from_float_ref` are NULL for the family: a row cannot decode without the plan.
- Every path that would call them is refused:
  - CUDA supports_op;
  - the MMA F16 conversion;
  - get_rows / cpy;
  - the CPU FA fast paths.
- The CPU reference decodes with `ggml-turbot.h` instead.
- `GGML_TYPE_COUNT` grows by 17. That only resizes arrays.

---

## 3. Byte layout

### 3.1 Units

- One layer-side row = 1024 values = 4 KV heads × 256 = 8 WHT groups.
- Head h owns values [256h, 256h+256).
- Group g of head h owns values [256h+128g, 256h+128g+128).
- Element index inside a head: e ∈ [0, 256).

> `[TAG_TURBOT_ANY_GEOM]` This is geometry flags 0. Other shapes (2 × 256, 1 × 256, 8 × 128, 4 × 128, 2 × 128 KV heads) use rows of NR = 1, 2 or 4 runs of 256 values, one width per run, in the same 1024-wide container tensor. Section 14.1-14.3 has the flags, the run mapping and the row sizes.

### 3.2 Base row (one cell, one layer-side), `struct ggml_turbot_side`

| Bytes | Content |
|---|---|
| `[base_off[h], base_off[h] + 32·b[h])` | head h run, planes per 3.4; `base_off[0] = 0`, `base_off[h+1] = base_off[h] + 32·b[h]` |
| `[32·S, 32·S + 16)` | 8 × `ggml_fp16_t` old gains, gain of (h, g) at `32·S + 2·(2h+g)` |

- Row size = 32·S + 16 = 16·(2S+1). Every row base is 16-byte aligned (128-byte buffer alignment, 16 | row size).
- Every run and every plane starts at a multiple of 32 bytes from the row base, so it is 16-byte aligned.
- A 32-element chunk starting at an element multiple of 32 maps to one 16 B (P4), 8 B (P2) or 4 B (P1) aligned load.

### 3.3 Young pool row (one granule cell, one layer)

| Bytes | Content |
|---|---|
| `[0, K.young_bytes)` | K part: head runs of 32·(y−b) at `K.young_off[h]`, then 8 young gains at `32·Σ(y−b)` |
| `[K.young_bytes, K.young_bytes + V.young_bytes)` | V part, same layout (`pool_v_off = K.young_bytes`) |

- `young_bytes` = 32·Σ_h (y−b) + 16.
- pool_row_bytes = K.young_bytes + V.young_bytes, a multiple of 16.
- Pool tensor per attention layer: `GGML_TYPE_I8 [pool_row_bytes, max(N_R, GGML_TURBOT_POOL_MIN_ROWS)]`, N_R = POOL cells (multiple of 64). With POOL 0 the tensor keeps one dummy granule of rows that no gtab entry ever points to (9.4).
- Pool row of cell c in a granule owning slot k: `k·64 + (c & 63)` (`ggml_turbot_pool_row`).
- Byte address: `(size_t) row · pool->nb[1]`. All pool arithmetic is size_t or int64.

### 3.4 Planes and bit order (`ggml_turbot_planes_of`, `ggml_turbot_get_code`, `ggml_turbot_set_code`)

A run of width w has planes in this order, each only if needed:

| w | planes | run bytes |
|---|---|---|
| 1 | P1 | 32 |
| 2 | P2 | 64 |
| 3 | P2 P1 | 96 |
| 4 | P4 | 128 |
| 5 | P4 P1 | 160 |
| 6 | P4 P2 | 192 |

- **P4 (128 B):** element e in nibble e%2 of byte e/2, low nibble for even e (turbo4p rule). Holds code bits 0..3.
- **P2 (64 B):** element e at bits 2·(e%4)..2·(e%4)+1 of byte e/4. Holds the next 2 code bits (bits 0-1 when there is no P4).
- **P1 (32 B):** element e at bit e%8 of byte e/8. Holds the next code bit.
- **Base code** of width b is the old index j.
- **Refinement code** of width r = y−b is s. Refinement runs use the same plane rule for width r (r ≤ 6).
- **Young index** = (j << r) | s.

### 3.5 Granule table

- `I32 [n_granules]`, n_granules = kv_size / 64 (kv_size must be a multiple of 64).
- Entry g = young slot of granule g, or −1.
- One table per ubatch graph, shared by all layers.

### 3.6 VRAM

For kv_size cells, attention layers ℓ, and N_R pool cells:

```
bytes = kv_size · Σ_ℓ (32·S_K(ℓ) + 16 + 32·S_V(ℓ) + 16)  +  N_R · Σ_ℓ (32·R_K(ℓ) + 16 + 32·R_V(ℓ) + 16)
```

with R = Σ_h (y−b). turbo5p reference: kv_size · n_layers · 2 · 656.

Default plan (appendix A: fq_3t0 widths, y = 7 everywhere, POOL 65,536, CAP 16,384), 262,144 cells, 16 layers:

| Quantity | Value |
|---|---|
| Σ old widths over 128 (layer, side, head) | 549 (4.289 bits mean) |
| Base bytes per cell, all layer-sides | 18,080 |
| Pool bytes per pool cell, all layers | 11,616 |
| Base | 4,739,563,520 B = **4,520.00 MiB** |
| Young pool | 761,266,176 B = **726.00 MiB** |
| Total | 5,500,829,696 B = **5,246.00 MiB** |
| turbo5p | 5,502,926,848 B = 5,248.00 MiB (margin 2.00 MiB) |

Graph inputs:
- granule table: 16 KiB;
- young rows: 4 B per ubatch row;
- fill list: 16 B per entry.

No F16 scratch is reserved (section 7.6).

---

## 4. Coder math

All formulas are implemented in `ggml-turbot.h`. That file is the oracle for both CUDA kernels. Units: u is one coordinate of a unit-norm rotated 128-group ("r units"), z = √128·u.

### 4.1 Codebooks (generated, `ggml-turbot-tables.h`)

**Old codebook C_b** (2^b float32, ascending, exactly antisymmetric):
- b = 4: `C4N` mirrored; b = 5: `C5N` mirrored. These are the turbo4/turbo5p literals, cast to float32.
- b = 2, 3, 6: `llama-kvfq.cpp lloyd_max_gauss(b)`, same iteration (exact erf, bisection init, Jacobi Lloyd, stop at max Δ < 1e-12 or 20,000 iterations), divided by √128 in float64, then cast.
  - Convergence confirmed: 51 / 188 / 9,054 iterations for b = 2 / 3 / 6.

**Old thresholds:** `T_b[i] = (float32) 0.5f·(C_b[i] + C_b[i+1])`, i = 0..2^b−2.
- `T_b[2^{b−1}−1] == 0`.
- These are NOT the turbo `TURBO_MID_5BIT` literals. turbot always uses computed mids, exactly like kvfq.

**Young codebook (b, y), r = y−b:** section 1.1 point 2, the generator's `a_lloyd()`.
- `LUT_{b,y}[(j<<r)|s]` = level / √128 (float32).
- `TY_{b,y}` = 2^y−1 ascending thresholds.
  - Inner entries: float32(midpoint / √128).
  - Entry ((j+1)<<r)−1 is `T_b[j]` **bit for bit**.
- The generator asserts sortedness, the embedding, and `count(TY, u) >> r == count(T_b, u)` on a dense sweep.

**Fill code** `F_{b,y}[j]` = argmin_s |LUT[(j<<r)|s] − C_b[j]| (float64 of the float32 values, ties to the lower s).

**Offsets.** Old runs start at `2^b − 4`. Young runs start at `(b−2)·512 + 8 + 2^y − 2^{b+2}`. Fill runs start at `{0,24,64,128,224}[b−2] + (y−b−1)·2^b`. These are closed forms of the generated `GGML_TURBOT_*_OFF_LIST`.

Sizes:
- old: 124 entries;
- young LUT and thresholds: 2,312 each (the last threshold slot of each run is padding);
- fill: 352 entries.

### 4.2 Encode one group (`ggml_turbot_quantize_group`)

Inputs: x[128] (float32, the post-RoPE K or V values; turbot turns the upstream Hadamard off by default, 10.3), widths b, y.

```
norm_sq = Σ_i x_i²            (float32, i ascending)
norm    = sqrtf(norm_sq)
inv     = norm > 1e-10f ? 1/norm : 0
u_i     = x_i · inv
u      := S2 ⊙ (INV_SQRT_128 · H(S1 ⊙ u))     i.e. u *= S1; butterfly h = 1,2,..,64; u_i *= INV_SQRT_128·S2_i
j_i     = #{ k : u_i >= T_b[k] }
s_i     = #{ k < 2^r−1 : u_i >= TY_{b,y}[(j_i<<r) + k] }          (== (#{k : u_i >= TY[k]}) & (2^r−1))
gain_b  = f16( recon_b > 1e-10f ? norm/recon_b : norm ),  recon_b = sqrtf(Σ C_b[j_i]²)
gain_y  = f16( recon_y > 1e-10f ? norm/recon_y : norm ),  recon_y = sqrtf(Σ LUT[(j_i<<r)|s_i]²)
```

- S1/S2 are the seed-42 signs (`ggml_turbot_wht_s1/s2` == `turbo_cpu_s1/s2` == `TURBO_WHT_SIGNS1/2`).
- f16 means `ggml_fp32_to_fp16` on the host and `__float2half` on the device.
- The old tier is bit-identical to `llama-kvfq.cpp fq_group` codes. Stored gain = f16(g), and kvfq decodes with fp16(fp32(g)).
- **Index rule:** "count of thresholds with u ≥ t", equal to `std::upper_bound`. Any equivalent search is allowed (linear count, branchless binary search over a sorted run) provided it returns this count, including for u exactly equal to a threshold and for duplicate thresholds.
- `ggml_turbot_young_index(b, y, u)` is the writer's form (8.2): one count over all 2^y−1 young thresholds, then j = young >> r and s = young & (2^r−1). It equals the two-step rule because the old thresholds are embedded bit for bit (4.1).
- **Zero row:** inv = 0, so u = 0, j = 2^{b−1} (the smallest positive level), gains = 0, and it decodes to exact zeros.
- **NaN input:** every compare is false, so j = 0 and s = 0. This is not guarded (same as turbo5p).

### 4.3 Decode (`ggml_turbot_decode_side`)

```
old:    v = C_b[j] · f16→f32(gain_b)
young:  v = LUT_{b,y}[(j<<r)|s] · f16→f32(gain_y)
```

- The result is in the rotated domain. Q is rotated in the graph and the attention output is inverse-rotated for V, exactly as turbo5p.
- A reader may compute `LUT·gain` in float and store half, as the MMA tile already does.
- A reader must NOT quantise the young LUT to int8, and must not take the young read's base level from an int8 table. Measured with the header math (7.4): a direct int8 young LUT costs ×1.49-1.64 young error at y = 7, and an int8 base level adds the old tier's int8 rounding error, about 20% of the young error at b = 5.
- The old read MAY use an int8 register LUT for b ≤ 5 (`ggml_turbot_old_level_i8`), the trade turbo4p and turbo5p already ship: ×1.000 / ×1.001 / ×1.003 / ×1.014 old error for b = 2..5. b = 6 stays float or half (×1.124).

### 4.4 Center fill (`ggml_turbot_fill_side`)

Used when a granule gains a slot while it already holds live cells that have only old codes.

```
s_i    = F_{b,y}[j_i]
gain_y = gain_b_f32 · ( sqrtf(Σ C_b[j_i]²) / sqrtf(Σ LUT[(j_i<<r)|s_i]²) )    (gain_b_f32 if the young norm ≤ 1e-10f)
```

- The young read then approximates the old read, and the cell becomes readable through the young path.
- The whole refinement run and the gain of the cell's side part are overwritten.
- CORRECTED 2026-09-15. Measured with the header math: 16,000 unit-norm Gaussian groups through the WHT, f16 gains, and the fill gain above (`scratchpad/turbot_review_mc.py`). MSE relative to the analytic Gaussian D_b of the old codebook:

| (b, y) | fill read vs old read | fill read vs true value | old read vs true value (same MC) |
|---|---|---|---|
| b2 y3 (worst of all pairs) | 0.200 | 1.198 | 1.012 |
| b3 y4 | 0.166 | 1.142 | 0.978 |
| b4 y5 | 0.137 | 1.102 | 0.965 |
| b5 y6 | 0.055 | 1.005 | 0.945 |
| b6 y7 | 0.117 | 1.075 | 0.959 |
| y = 7, b = 2 / 3 / 4 / 5 | 0.0003 / 0.0040 / 0.0102 / 0.0289 | 1.012 / 0.982 / 0.975 / 0.977 | 1.012 / 0.978 / 0.965 / 0.945 |

- The gain rescale removes the component of the fill offset that lies along the codeword, so a filled cell reads at about the old tier's own error.
- Default plan (y = 7): b = 6 heads read at ×1.12 the measured old-tier error, b = 5 at ×1.03, every narrower head within ×1.01.
- The earlier ×1.68 / ×1.60 / ×1.13 table was an analytic Gaussian calculation without the gain and is withdrawn.

### 4.5 Summation order and bit exactness

- The host reference sums sequentially. CUDA reduces as a lane tree, the same as turbo5p (`set-rows.cu`, `[TAG_TURBO4P]` note).
- Index codes are exact except when a value lies within one ULP of a threshold after a different sum order changed the norm by one ULP.
- Gains may differ by one f16 step.
- Tests use the tolerances in section 11.

### 4.6 Tables vs study (checked 2026-09-15)

`turbot_tables_vs_study.py` in the scratchpad compared the generator with `build_designs(..., ['a_lloyd'])`:

| b | Result |
|---|---|
| 4, 5 | old and young tables identical bit for bit |
| 2, 3 | max abs difference 6.7e-8 and 1.6e-7 (r units) |
| 6 | max 5.0e-5 |

- Cause: the study's `kv_cand.lloyd_max` uses a linearly interpolated CDF. The generator uses kvfq's exact erf iteration, which converged (9,054 iterations for b = 6).
- The generator is authoritative, because kvfq's base codes produced the measured quality.
- The unit test uses these tolerances (section 11.1).

---

## 5. ggml API (owner A)

### 5.1 Types (`ggml/include/ggml.h`)

```c
GGML_TYPE_TURBOT_S8  = 49, GGML_TYPE_TURBOT_S9  = 50, ... GGML_TYPE_TURBOT_S24 = 65,   // [TAG_TURBOT]
GGML_TYPE_COUNT      = 66,
```

type_traits (`ggml.c`) for S in 8..24:

| Field | Value |
|---|---|
| `.type_name` | `"turbot_s<S>"` |
| `.blck_size` | 1024 |
| `.type_size` | 32·S+16 |
| `.is_quantized` | true |
| `.to_float` | NULL |
| `.from_float_ref` | NULL |

- `GGML_TYPE_TURBOT_S8` is also the "turbot requested" sentinel in `llama_context_params.type_k/type_v`. `common` maps `-ctk turbot` to it.
- `ggml_type_name(GGML_TYPE_TURBOT_S8)` stays "turbot_s8"; logs print "turbot" explicitly.
- If `ggml_validate_row_data` rejects unknown types in its switch, add the family as "no validation".
- `[TAG_TURBOT_ANY_TYPES]` `GGML_TYPE_TURBOT_S2` … `GGML_TYPE_TURBOT_S7` (66..71) are appended for rows with 1 or 2 runs, and `GGML_TYPE_COUNT` is 72 (14.3).

### 5.2 Ops

**`GGML_OP_TURBOT_SET_ROWS`** is appended immediately before `GGML_OP_COUNT`, after `GGML_OP_GLU`, so that no existing op value moves.
- `GGML_OP_COUNT` = 103.
- Name `"TURBOT_SET_ROWS"`, symbol `"turbot_set_rows(x)"`.
- Update both static_asserts.

```c
struct ggml_turbot_op_params;   // ggml-turbot.h

// [TAG_TURBOT] Write K or V rows into a turbot base cache and, for rows with a young pool row, into the young pool.
//   a     : base cache, turbot type, [1024, kv_size, 1, 1] (contiguous rows)
//   b     : F32 [1024, n_rows, 1, 1], ggml_is_contiguous_rows
//   c     : I64 or I32 [n_rows], destination cell of each row
//   pool  : GGML_TYPE_I8 [pool_row_bytes, n_pool_rows, 1, 1] (the layer's young pool)
//   young : I32 [n_rows], young pool row of each row, or -1
//   fill  : I32 [4, n_fill] of (granule, slot, mask_lo, mask_hi), or NULL
//   params: side GGML_TURBOT_SIDE_K or _V; ggml_turbot_type_of_s(side.s) must equal a->type
// returns a view of a (like ggml_set_rows). Fill entries are applied before the rows.
GGML_API struct ggml_tensor * ggml_turbot_set_rows(
        struct ggml_context                * ctx,
        struct ggml_tensor                 * a,
        struct ggml_tensor                 * b,
        struct ggml_tensor                 * c,
        struct ggml_tensor                 * pool,
        struct ggml_tensor                 * young,
        struct ggml_tensor                 * fill,
        const struct ggml_turbot_op_params * params);

// [TAG_TURBOT] Mark a GGML_OP_FLASH_ATTN_EXT whose K and V are turbot views. Sets src[7] = pool, src[8] = gtab and
// the turbot op params (side GGML_TURBOT_SIDE_BOTH). gtab: I32 [n_granules], contiguous, n_granules*64 >= K->ne[1].
GGML_API void ggml_flash_attn_ext_set_turbot(
        struct ggml_tensor                 * a,
        struct ggml_tensor                 * pool,
        struct ggml_tensor                 * gtab,
        const struct ggml_turbot_op_params * params);
```

**Source slot order**

| Op | src[0] | src[1] | src[2] | src[3] | src[4] | src[5] | src[6] | src[7] | src[8] | src[9] |
|---|---|---|---|---|---|---|---|---|---|---|
| TURBOT_SET_ROWS | b (rows) | c (cells) | a (base) | pool | young | fill or NULL | – | – | – | – |
| FLASH_ATTN_EXT (turbot) | q | k view | v view | mask or NULL | sinks or NULL | kv_pos or NULL | q_pos or NULL | pool | gtab | – |

**`ggml_turbot_set_rows` asserts:**
- `ggml_turbot_is_type(a->type)`;
- `b->type == F32`;
- `b->ne[0] == 1024` (REVISED `[TAG_TURBOT_ANY_GEOM]`: `ggml_turbot_geom_row_elems(flags)`, which is 1024 at flags 0; `a->ne[0]` stays 1024, 14.5);
- `b->ne[2] == b->ne[3] == 1`;
- `c->ne[0] == b->ne[1]`, type I64 or I32;
- `pool->type == I8` (never NULL: a POOL 0 plan still has 64 pool rows, 9.4);
- `young->type == I32`, `young->ne[0] == b->ne[1]`;
- `fill == NULL` or (`fill->type == I32`, `fill->ne[0] == 4`);
- the params pass `ggml_turbot_op_params_get`, and the side's S matches `a->type`.

Result: `ggml_view_tensor(ctx, a)`, op, sources as in the table. Bytes 0..23 of op_params are zero; turbot params go at byte 24.

**`ggml_flash_attn_ext_set_turbot` asserts:**
- `a->op == FLASH_ATTN_EXT`;
- src[1] and src[2] are turbot types matching the params' K and V sums;
- `src[1]->ne[0] == src[2]->ne[0] == 256`;
- `src[1]->ne[2] == src[2]->ne[2] == 4`;
- REVISED `[TAG_TURBOT_ANY_GEOM]`: the two lines above are the flags-0 case of `ne[0] == ggml_turbot_geom_head_dim(flags)` and `ne[2] == ggml_turbot_geom_n_head(flags)` (14.5);
- `src[1]->ne[3] == 1`;
- pool I8; gtab I32, 1-D, contiguous, `gtab->ne[0]*64 >= src[1]->ne[1]`;
- `src[7] == src[8] == NULL` before the call.

Bytes 0..19 (scale, max_bias, softcap, prec, n_kv_max) are untouched.

### 5.3 Op params (`struct ggml_turbot_op_params`, 24 bytes at op_params byte 24)

| Byte | Field |
|---|---|
| 24-27 | magic `0x54425254` |
| 28 | version 1 |
| 29 | side (0 K, 1 V, 2 both) |
| 30 | log2_granule = 6 |
| 31 | flags = geometry, 0 for 4 × 256 (`[TAG_TURBOT_ANY_GEOM]`, 14.1) |
| 32-35 | bk[4] |
| 36-39 | bv[4] |
| 40-43 | yk[4] |
| 44-47 | yv[4] |

The offset was 16 before the upstream sync; upstream `ggml_flash_attn_ext_set_n_kv_max` now owns int32 slot 4 (bytes 16-19) of every FLASH_ATTN_EXT.

Use `ggml_turbot_op_params_make/set/get` and `ggml_turbot_layer_from_op_params`. Never write the bytes by hand.

`[TAG_TURBOT_ANY_GEOM]` The version stays 1, because flags 0 reads exactly as before. `ggml_turbot_op_params_get` rejects a flags byte with a bit above 0x07 or an invalid geometry (3, 7). Width bytes of runs ≥ NR are 0, and `ggml_turbot_layer_from_op_params` rejects them otherwise (14.4).

### 5.4 supports_op rules

**CPU (owner A)**
- TURBOT_SET_ROWS: all asserts of 5.2 hold, the tensors are host-accessible, and src0 has ne[2] == ne[3] == 1.
- FLASH_ATTN_EXT with a turbot K or V:
  - both turbot, params valid with side BOTH;
  - src[7] and src[8] present;
  - D = 256, 4 KV heads. REVISED `[TAG_TURBOT_ANY_GEOM]`: D and the KV head count must match the op-params geometry (`ggml_turbot_geom_head_dim`, `ggml_turbot_geom_n_head`), which is D = 256 and 4 heads at flags 0. The CPU takes every geometry of 14.1, and a TURBOT_SET_ROWS source row must have `ggml_turbot_geom_row_elems(flags)` values;
  - `q->ne[3] == 1`;
  - mask ne[2] == 1 (or no mask).
- FLASH_ATTN_EXT with a turbot type but no src[7]/src[8]: false.

**CUDA (owner C for SET_ROWS, owner B for FA through `ggml_cuda_flash_attn_ext_supported`)**
- TURBOT_SET_ROWS: same conditions as the CPU, plus `pool` and `a` in CUDA buffers.
- FA: section 7.6 (`ggml_cuda_get_best_fattn_kernel` and `ggml_cuda_flash_attn_ext_supported`).

---

## 6. CPU reference implementation (owner A)

The CPU paths are the test oracle, not a product path. Correctness and clarity beat speed. Both run with `n_tasks = 1`.

### 6.1 `ggml_compute_forward_turbot_set_rows(const ggml_compute_params *, ggml_tensor * dst)` (`ggml-cpu/ops.cpp`, declared in `ops.h`, dispatched in `ggml-cpu.c`)

1. `ggml_turbot_op_params_get`, `ggml_turbot_layer_from_op_params`. Choose `sd = side == K ? &l.k : &l.v` and `part = side == K ? 0 : l.pool_v_off`.
2. For each fill entry (granule G, slot k, mask_lo, mask_hi), and each c ∈ [0,64) whose bit is set (c < 32: bit c of (uint32) mask_lo, else bit c−32 of mask_hi):
   - `cell = G·64 + c`;
   - `ggml_turbot_fill_side(base(cell), sd, pool_row(k·64 + c) + part)`.
3. For each row i:
   - `cell = c[i]`;
   - `ggml_turbot_encode_side(b_row(i), sd, base(cell), young[i] >= 0 ? pool_row(young[i]) + part : NULL)`.
   - base(cell) = `(uint8_t*) a->data + (size_t) cell·a->nb[1]`.
   - pool_row(p) = `(uint8_t*) pool->data + (size_t) p·pool->nb[1]`.
   - b_row(i) = `(const float*) b->data + i·b->nb[1]`.
4. Assert `cell < a->ne[1]`, `young[i] < pool->ne[1]`, `k·64+63 < pool->ne[1]`.

### 6.2 FA (`ggml_compute_forward_flash_attn_ext`)

At the top of `ggml_compute_forward_flash_attn_ext`, before the prec switch: if `ggml_turbot_is_type(dst->src[1]->type)`, call a new static `ggml_compute_forward_flash_attn_ext_turbot(params, dst)` and return.

Recommended implementation, which inherits exact FA semantics (mask, pos mask, sinks, softcap, GQA broadcast):
1. Let n_kv = k->ne[1]. Decode every cell i < n_kv of K and V:
   - `slot = gtab[i >> 6]`;
   - `young = slot >= 0 ? pool_row(ggml_turbot_pool_row(slot, i)) : NULL`;
   - `ggml_turbot_decode_side(k->data + i·k->nb[1], young ? young + 0 : NULL, &l.k, Kf + 1024·i)`;
   - same for V with `+ l.pool_v_off` and `&l.v`.
   - Buffers: `std::vector<float>` Kf, Vf of 1024·n_kv.
2. Build stack `ggml_tensor` copies `kf = *k`, `vf = *v`:
   - type `GGML_TYPE_F32`, data the buffers, same ne;
   - `nb = {4, 4·1024, 4·256, 4·1024·n_kv}`.
   - Element (cell i, head h, e) is at `4·(1024·i + 256·h + e)`, which is what `ik1·nb1 + ik2·nb2` gives.
3. `ggml_tensor d = *dst; d.src[1] = &kf; d.src[2] = &vf; d.src[7] = d.src[8] = NULL;` then call the existing `ggml_compute_forward_flash_attn_ext_f16(params, &d)`.
   - With F32 K/V it takes the existing F32 paths. The work size for FA already covers them; n_tasks is 1.

### 6.3 Graph plumbing (owner A)

`ggml-cpu.c`:
- compute_forward case for the new op;
- `ggml_get_n_tasks`: `GGML_OP_TURBOT_SET_ROWS` → 1, and FLASH_ATTN_EXT with a turbot src[1] → 1;
- work size 0 for the new op.

`ggml-cpu.cpp`: supports_op per 5.4.

---

## 7. CUDA read path (owner B)

### 7.1 Principle

Every existing kernel, loader, template and template-instance TU keeps byte-identical source, and no existing instance TU includes a turbot header. That guarantees identical codegen for every existing type. The allowed edits to existing files are host-side only:

1. `fattn-mma-f16.cuh`, as its own first change and nothing else in this file:
   - add `constexpr int FATTN_MMA_TURBO_MODE_COUNT = 5;` directly after the `FATTN_MMA_TURBO_*` enum;
   - size BOTH `shared_memory_limit_raised` arrays `[GGML_CUDA_MAX_DEVICES][FATTN_MMA_TURBO_MODE_COUNT]`. This fixes the existing out-of-bounds bug: `[4]` (fattn-mma-f16.cuh:2674) and `[3]` (2693) are indexed by `turbo_mode` up to 4 (`FATTN_MMA_TURBO5P512`).
   - No turbot enum value, no include and no turbot branch go into this file.
2. `fattn.cu`: routing, alloc size, supports and the case dispatch helper (7.6).

New files (all B):

| File | Contents |
|---|---|
| `ggml/src/ggml-cuda/fattn-turbot.cuh` | all turbot device code and the host case template. Includes `fattn-mma-f16.cuh`, `turbot-tables.cuh`, `ggml-turbot.h`. |
| `ggml/src/ggml-cuda/fattn-turbot-decl.cuh` | host-only declarations for `fattn.cu`: the `ggml_cuda_flash_attn_ext_turbot_case` template declaration, the `DECL_FATTN_TURBOT_CASE` macro and the 20 `extern DECL_FATTN_TURBOT_CASE(...)` lines. Includes `common.cuh` only, never `turbot-tables.cuh`. |
| `ggml/src/ggml-cuda/template-instances/fattn-mma-turbot-instance-ncols1_<n1>-ncols2_<n2>.cu` | one file per pair in 7.6: `#include "../fattn-turbot.cuh"` and `DECL_FATTN_TURBOT_CASE(n1, n2);`. Hand-written, with a header comment in the style of the `[TAG_FA_NCOLS_128]` files saying they are not produced by generate_cu_files.py. |

```cpp
#define DECL_FATTN_TURBOT_CASE(ncols1, ncols2) \
    template void ggml_cuda_flash_attn_ext_turbot_case<256, 256, ncols1, ncols2>(ggml_backend_cuda_context & ctx, ggml_tensor * dst)
```

- The existing GLOB `template-instances/fattn-mma*.cu` compiles the new instance files, and the `*.cu` / `*.cuh` GLOBs cover the rest, so `ggml-cuda/CMakeLists.txt` needs no change. B confirms the GLOB is unconditional for this build and edits CMake only if it is not.
- `turbot-tables.cuh` (about 20 KB of `__constant__` tables) is compiled only into the 20 turbot instance TUs and `turbot-set-rows.cu`.
- `fattn.cu` includes `ggml-turbot.h` for the predicates. Its `static const` tables are host data with internal linkage, unreferenced there, and never reach device constant memory.

### 7.2 New functions in `fattn-turbot.cuh`

**Kernel entry**

```cpp
template<int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap>
__launch_bounds__(ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols1*ncols2), ggml_cuda_fattn_mma_get_occupancy(DKQ, DV, ncols1*ncols2))
static __global__ void flash_attn_ext_turbot(/* every parameter of fattn_kernel_t, in the same order, then: */
        const char    * pool_ptr,        // young pool data (layer), never nullptr (9.4 keeps >= 64 rows)
        const int32_t * gtab,            // granule table, gtab[cell >> 6] = slot or -1
        const int64_t   nb_pool,         // pool->nb[1]
        const int64_t   turbot_desc0,    // packed layout, see below
        const int64_t   turbot_desc1,
        const int64_t   turbot_desc2);
```

- The layout is passed as scalar kernel arguments (register values) or a pool-allocated device array.
- **Never** use `__device__` globals: host writes to them race with kernels still queued for earlier layers.
- B chooses the packing of `turbot_desc*`. Contents: per side and head, b, y, `base_off`, `young_off`; per side, `base_gain_off` and `young_gain_off`; and `pool_v_off`.

**Per-head state helper (required)**

```cpp
struct turbot_head_state;   // per side (K, V) of head z_KV: b, r, run offset in the base row, run offset in the side part
                            // of a pool row, gain offsets, LUT run selectors; plus pool_v_off
static __device__ __forceinline__ turbot_head_state turbot_head_state_of(int z_KV, int64_t desc0, int64_t desc1, int64_t desc2);
```

- `flash_attn_ext_f16` computes head addressing in two places: inside the while loop (fattn-mma-f16.cuh:2407-2418) and again for the final `is_fixup` block (2478-2486). The two copies have already drifted (section 13).
- The turbot kernel calls `turbot_head_state_of` at BOTH sites and derives no per-head value anywhere else. Both call sites carry `[TAG_TURBOT_HEAD_STATE]`, and the helper's comment says there are exactly two.
- Base pointers are row bases: `K_row = K + nb13*sequence`, `V_row = V + nb23*sequence`. `nb12` / `nb22` (the 8S+4 view stride) are never used, because heads have different widths.

**Copies of the tile machinery**

- `flash_attn_ext_turbot_process_tile` and `flash_attn_ext_turbot_iter`: copies of `flash_attn_ext_f16_process_tile` / `_iter`, specialised as follows:
  - `nstages = 0`, `V_is_K_view = false`, `turbo_KV` removed, `static_assert(Q_in_reg)` (every D=256 config has it, fattn-mma-f16.cuh:69-254);
  - all turbo/f16 load branches replaced by the turbot loaders;
  - extra trailing parameters for the turbot state.
  - Softmax, mask, sinks, softcap, combine and fixup code is copied verbatim.
- `launch_fattn_turbot<DV, ncols1, ncols2>(ctx, dst, kernel, nwarps, nbytes_shared, nbatch_fa, warp_size)`: a copy of `launch_fattn` without the F16 conversion blocks.
  - It keeps the KV_max computations: explicit-mask path and `[TAG_FA_KVMAX_POS]` pos path, same gates.
  - It keeps the stream_k block layout and the fixup launches. It reuses the existing `flash_attn_pos_to_KV_max`, `flash_attn_mask_to_KV_max`, `flash_attn_stream_k_fixup_*` and `flash_attn_combine_results` kernels unchanged.
  - **Deviation (read-speed plan item 2):** layouts that need a fixup now use work-balanced seams and turbot copies of the fixup kernels by default. See section 13, `[TAG_TURBOT_FA_BALANCE]`.
- `template <int DKQ, int DV, int ncols1, int ncols2> void ggml_cuda_flash_attn_ext_turbot_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst)`: host.
  - Reads params, builds `ggml_turbot_layer`, sizes shared memory (7.5), and launches.
  - Raises the shared-memory limit of the turbot kernel once per device and softcap variant, with its own `static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES][2]`.

### 7.3 Tile loop integration

`k_VKQ_0 = kb0·nbatch_fa` is the first cell of the KV tile. All D=256 configs have nbatch_fa ∈ {32, 64}, so a tile lies inside one granule. Required: `static_assert(GGML_TURBOT_GRANULE % nbatch_fa == 0)` in the turbot iter.

At the top of every turbot iter, before the K loads:

```
g        = k_VKQ_0 >> 6                       // lane-uniform: every lane of the block works on the same tile
slot     = gtab[g]                            // one load per tile
young    = slot >= 0
base0    = K_row + (size_t) k_VKQ_0 * nb11     // base row of the tile's first cell (K); V: V_row + k_VKQ_0 * nb21
prow0    = pool + (size_t)(slot*64 + (k_VKQ_0 & 63)) * nb_pool      // only if young
```

- Rows inside the tile: `base0 + i·nb11` and `prow0 + i·nb_pool` (i < nbatch_fa, consecutive cells of one granule).
- Head `z_KV` selects its `turbot_head_state` (7.2). It is uniform for the whole CUDA block.
- K reads use side K offsets. V reads use side V offsets, plus `pool_v_off` into the pool row.
- `if (young) { YOUNG loader } else { OLD loader }` is a lane-uniform runtime branch. Width-dependent plane presence (`w & 2`, `w >= 4`, `w & 1`) is also a uniform runtime branch.
- OOB rows (`oob_check && i >= i_sup`) produce zeros without loading, like the existing loaders. The granule lookup itself stays in range: `g <= (ne11−1) >> 6 < n_granules`.

### 7.4 Loaders

32 elements = 16 half2 per chunk, like `flash_attn_ext_turbo5p_load_tile`. The chunk element offset is `e0 = k0_start·2 + k·32` inside the head. It is a multiple of 32, so a chunk never crosses a WHT group; g = e0 / 128.

**OLD**

```
run   = row + base_off[h]
bits  = (p4 >= 0 ? 16-byte load at run + p4 + e0/2 : 0)       // 32 nibbles
      | (p2 >= 0 ? 8-byte  load at run + p2 + e0/4 : 0)        // 32 two-bit fields
      | (p1 >= 0 ? 4-byte  load at run + p1 + e0/8 : 0)        // 32 bits
gain  = __half2float(row[base_gain_off + 2*(2h+g)])
for each element: j = assembled code;  v = level_old(j) * gain;  store half
```

**YOUNG**
- Same unpack for the base code (width b) from the base row, and for the refinement (width r) from `prow + side_off + young_off[h]`.
- `idx = (j << r) | s`, `v = level_young(idx) · gain_y`, with gain_y from the pool row at `young_gain_off + 2·(2h+g)`.

**Value representation.** Measured with the header math (sphere Monte Carlo, f16 gains, `scratchpad/turbot_review_mc.py`). Each entry is the error ratio against a float read of the same codes:

| Read | Representation | Error vs float | Rule |
|---|---|---|---|
| OLD b = 2, 3 | int8 register LUT, one `__byte_perm` over a 4- or 8-byte table; scale `ggml_turbot_old_i8_scale(b)` | ×1.000 / ×1.001 | allowed |
| OLD b = 4, 5 | the existing `turbo4_int8_lut` / `turbo5_int8_lut` (fattn-common.cuh:512, 558), unchanged; same C4N/C5N literals and scales | ×1.003 / ×1.014 | allowed (the trade turbo5p ships, `[TAG_TURBO4P_CENT]`) |
| OLD b = 6 | int8 | ×1.124 | forbidden: float or half |
| YOUNG | direct int8 LUT, one scale per (b, y) | ×1.49-1.64 at y = 7, ×3.2-3.6 at y = 8 | forbidden |
| YOUNG | int8 base level + anything | adds the old int8 rounding, about +20% young error at b = 5 | forbidden |
| YOUNG | float base level + int8 offset, one scale per (b, y) | ×1.02-1.20 at y = 7 | forbidden |
| YOUNG | float base level + int8 offset `LUT[(j<<r)|s] − C_b[j]`, one float scale per base cell | ≤ ×1.009 at y = 7 for b ≥ 3, ×1.070 at b = 2 | allowed only if B0 shows it faster than the float LUT |
| YOUNG | float (or half) LUT runs in shared memory | ×1 | default |

- The b = 5 turbot run is P4 then P1, so one 32-element chunk is one 16-byte P4 load plus one 4-byte P1 load. That is exactly the (qs, qh) pair `turbo5_int8_lut::gather4` consumes in `flash_attn_ext_turbo5p_load_tile`; b = 4 is P4 only, the `turbo4_int8_lut` input.
- B verifies in a comment tagged `[TAG_TURBOT_I8]` that entry k of each reused table is `ggml_turbot_old_level_i8(b, k)` for the turbot index order (ascending levels, j = count of thresholds ≤ u). E's backend test checks it numerically (11.3).
- `GGML_CUDA_TURBOT_OLD_I8` (a `#define` at the top of `fattn-turbot.cuh`, 0 or 1, default 1) selects the OLD representation for b ≤ 5 at compile time, so gate B0 can build and measure both. The CPU reference always reads float; the 11.2 FA tolerance covers the int8 difference.
- The young variants need two data-dependent float gathers (base level and scale) plus the int8 gather, against one float gather for the LUT. So the float LUT is expected to win, and it is the default.

**Float LUT runs in shared memory.** These cover YOUNG always, OLD b = 6, and OLD b ≤ 5 when `GGML_CUDA_TURBOT_OLD_I8` is 0.
- Area: `lut_off = GGML_PAD(nbytes_shared_KV_1stage + nbytes_shared_mask, 16)` bytes from the start of `tile_Q`, which is `tile_K` because every D=256 config has `Q_in_reg`. That is directly after the mask tile.
- Copy point: after the SECOND `__syncthreads()` that follows the `Q_B` register load (the position of fattn-mma-f16.cuh:1836 in the copied process_tile) and before the first iter. Nowhere earlier.
  - With `Q_in_reg`, the KV, mask and LUT areas all overlap the Q tile. For ncols = 128 the Q tile uses 128·132·4 = 67,584 B, while the LUT area starts near byte 27,000. tile_Q is filled before the first `__syncthreads` and copied to registers only after it. A copy right after the Q fill overwrites Q columns from about 50 onward before they reach `Q_B`, and prefill attention is silently wrong.
- Contents: the runs the call needs for its head. Per side, the OLD run (≤ 64 floats) and the YOUNG run (≤ 256 floats), plus ≤ 64 per-cell scales if that variant is chosen. `nbytes_shared_lut = GGML_PAD(2·(64 + 256 + 64)·sizeof(float), 16) = 3,072 B`.
- The run indices are runtime-uniform per call, so the `__constant__` loads broadcast. Per-element gathers then read shared memory, not `__constant__` memory with data-dependent addresses.

**Registers.** The existing kernels sit at 255 registers. The first build must check register use of the turbot kernels with `cuobjdump` for `<256,256,1,8>`, `<4,8>`, `<32,2>`, `<64,2>` and `<128,1>`. If a kernel spills, split the OLD/YOUNG loaders or reduce chunking inside the turbot kernel only.

### 7.5 Shared memory sizing

```
nbytes_shared_KV_1stage = nbatch_fa · max(nbatch_K2 + 4, nbatch_V2 + 4) · sizeof(half2)
nbytes_shared_mask      = ncols1 · (nbatch_fa/2 + 4) · sizeof(half2)
lut_off                 = GGML_PAD(nbytes_shared_KV_1stage + nbytes_shared_mask, 16)
nbytes_shared_Q         = ncols · (DKQ/2 + 4) · sizeof(half2)
nbytes_shared_combine   = nwarps·cols_per_warp · (nbatch_combine + 4) · sizeof(half2)
nbytes_shared_total     = max(nbytes_shared_combine, max(nbytes_shared_Q, lut_off + nbytes_shared_lut))
```

- Assert that `nbytes_shared_total` does not exceed what the existing f16 case function requests for the same config plus `nbytes_shared_lut`.
- ncols = 128 is the largest, where Q (67,584 B) dominates.

### 7.6 Routing, dispatch and scratch (`fattn.cu`)

**`ggml_cuda_get_best_fattn_kernel`.** Immediately after the `switch (K->ne[0])` head-dim check:

```cpp
if (ggml_turbot_is_type(K->type) || ggml_turbot_is_type(V->type)) {   // [TAG_TURBOT]
    const bool ok = ggml_turbot_is_type(K->type) && ggml_turbot_is_type(V->type) &&
                    K->ne[0] == 256 && V->ne[0] == 256 && K->ne[2] == 4 && V->ne[2] == 4 &&
                    dst->src[7] != nullptr && dst->src[8] != nullptr &&
                    (!mask || mask->ne[2] == 1) && Q->ne[3] == 1 && turing_mma_available(cc);
    return ok ? BEST_FATTN_KERNEL_MMA_F16 : BEST_FATTN_KERNEL_NONE;
}
```

- `cc` must be computed before this block; move nothing else.
- Every Q width goes to MMA, including Q = 1.
- Turbot has no VEC or TILE instance and never routes there. On GPUs without Turing-class MMA, supports_op is false.

**Case dispatch.** Add above `ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1`:

```cpp
// [TAG_TURBOT] every D=256 case call passes through here so a turbot K/V reaches its own kernel instances
template <int DKQ, int DV, int ncols1, int ncols2>
static void ggml_cuda_fattn_mma_case_dispatch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if constexpr (DKQ == 256 && DV == 256 && ncols2 <= 8) {
        if (ggml_turbot_is_type(dst->src[1]->type)) {
            ggml_cuda_flash_attn_ext_turbot_case<DKQ, DV, ncols1, ncols2>(ctx, dst);
            return;
        }
    }
    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, ncols1, ncols2>(ctx, dst);
}
```

- Replace the five `ggml_cuda_flash_attn_ext_mma_f16_case<` calls inside `ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1` (fattn.cu:17, 24, 31, 45, 50) with `ggml_cuda_fattn_mma_case_dispatch<`. No other call site changes.
- The D=256 path reaches only these five: `case 256` → `switch_ncols2<256, 256>` → `switch_ncols1<256, 256, n2>` with n2 ∈ {1, 2, 4, 8} (fattn.cu:87-223).
- Instance pairs (ncols1, ncols2), exactly the 20 the ladder can select:

| ncols | pairs |
|---|---|
| 8 | (8,1) (4,2) (2,4) (1,8) |
| 16 | (16,1) (8,2) (4,4) (2,8) |
| 32 | (32,1) (16,2) (8,4) (4,8) |
| 64 | (64,1) (32,2) (16,4) (8,8) |
| 128 | (128,1) (64,2) (32,4) (16,8) |

- At Q = 1, <1,8> / <8,1> use nbatch 64. At Q≈4, <4,8> uses 32. Prefill uses <16..128, *> at 32. All of them satisfy 64 % nbatch_fa == 0.
- **Q ≤ 2 route** (`[TAG_TURBOT_Q1_ROUTE]`, `[TAG_TURBOT_Q2_ROUTE]`, added after this spec; fattn.cu `ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1`). With ncols2 = 8, Q = 1 and Q = 2 both run on <4,8> instead of <1,8> and <2,8>.
  - Q = 2 is one slot verifying a 1-token draft, or two slots decoding one token each (one FA call with ne[1] = 2 on the unified cache).
  - Nothing else changes for the narrower Q. The kernel zero-fills the missing Q columns and skips their output, and the fixup kernels skip rows past ne01.
  - `launch_fattn_turbot` bounds KV with the row-wrapping mask scan (`flash_attn_turbot_mask_to_KV_max`: `turbot<1>` for Q = 1, `turbot<ncols1> wrap` for Q % ncols1 != 0), so the mask is never read past its last row. The positional scan wraps with fastmodulo.
  - `TURBOT_Q2_ROUTE=0` puts Q = 2 back on <2,8> for the A/B on one binary; Q = 1 stays on <4,8> either way. `LLAMA_TURBOT_FA_DEBUG=1` prints the instance (`ncols1=4 ncols2=8`) and `kv_scan` per shape.

**`ggml_cuda_flash_attn_ext_get_alloc_size`.** Before the kernel switch: `if (ggml_turbot_is_type(K->type)) { need_f16_K = need_f16_V = false; }` and skip the switch. No F16 scratch is reserved (decision 10).

**`ggml_cuda_flash_attn_ext_supported`.** Unchanged, apart from the rule above, which lives in `get_best_fattn_kernel`. Also check:
- `src[8]->type == I32 && ggml_is_contiguous(src[8]) && src[8]->ne[0]*64 >= K->ne[1]`;
- `src[7]->type == I8`;
- the params pass `ggml_turbot_op_params_get` with side BOTH.

### 7.7 What VEC does for turbot

Nothing. There is no VEC instance, no VEC type case and no F16 conversion. `ggml_cuda_fattn_kv_type_supported`, the mixed-type `is_turbo` lambda and the split-plane geometry checks are never reached for turbot, because of the early return in 7.6.

### 7.8 Read cost and speed gate B0

**Read volume.** Bytes per cell over the 32 layer-sides of the default plan:

| Case | bytes per cell | vs turbo5p (20,992 B) |
|---|---|---|
| old cell | 18,080 | 0.861× |
| young cell | 29,696 | 1.415× |
| 262,144 cells, 1 sequence (16,384 young) | 18,806 mean | 0.896× |
| 262,144 cells, 4 sequences at quota (65,024 young) | 20,961 mean | 0.999× |
| context ≤ 65K (every cell young) | 29,696 | 1.415× |

- Loads per 32-element chunk: turbo5p reads one 16 B load, one 4 B load and a norm. A turbot OLD chunk reads up to 2 base-plane loads and a gain. A YOUNG chunk adds up to 2 refinement-plane loads from a second memory region (the pool) and a second gain.
- The repo's own measurement (`[TAG_TURBO4P_CENT]`, fattn-mma-f16.cuh:789-798) says the turbo5p win came from load width and the int8 register gather, not from bytes read.
- **The speed target is therefore parity with turbo5p at long context, not a gain.** Short context may lose.

**Gate B0.** Runs at integration after A and B build, before C and D results are used, under the one-GPU-process rule and never on port 8080.
- **Builds:** A + B with `GGML_CUDA_TURBOT_OLD_I8` 1, then again with 0.
- **Measure:** `test-backend-ops perf -o FLASH_ATTN_EXT` with E's `turbot_perf` cases (11.2), turbot against turbo5p on each type's own route. Same D, GQA 6 and mask; kv 32768 / 131072 / 245760; nb 1 / 4 / 512 (ncols128 on).
- **Mixes:** `old` (gtab all −1), `band16k` (newest 16,384 cells young), `band64k` (newest 65,024 cells young), `young` (all young).
- **Data:** cache bytes are encoded on the host with `ggml-turbot.h` and uploaded, so B0 needs no writer.
- **The ncols128 FA correctness cases (11.2) must pass first.**
- **GO iff all hold** (ms per op, turbot / turbo5p):
  1. kv 131072, nb 4, band16k: ≤ 1.00;
  2. kv 245760, nb 4, band16k: ≤ 1.00;
  3. kv 131072, nb 1, band16k: ≤ 1.00;
  4. kv 131072, nb 4, band64k: ≤ 1.05;
  5. kv 131072, nb 512, band16k: ≤ 1.10.
  - kv 32768 `young` is reported, not gated.
- **NO-GO:** stop the integration, record the table in TESTING.md and return to the coordinator. The design is reopened, not tuned around.
- **Choosing `GGML_CUDA_TURBOT_OLD_I8`:** keep the faster build that passes. If both pass within 2% of each other, keep 0 (float reads, the measured quality).

---

## 8. CUDA writer (owner C)

### 8.1 Files

- New `ggml/src/ggml-cuda/turbot-set-rows.cu` and `turbot-set-rows.cuh`:

```cpp
void ggml_cuda_op_turbot_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_turbot_set_rows_supported(const ggml_tensor * op);
```

- `ggml-cuda.cu`:
  - `#include "turbot-set-rows.cuh"`;
  - `case GGML_OP_TURBOT_SET_ROWS: ggml_cuda_op_turbot_set_rows(ctx, dst); break;` in `ggml_cuda_compute_forward`;
  - `case GGML_OP_TURBOT_SET_ROWS: return ggml_cuda_turbot_set_rows_supported(op);` in `ggml_backend_cuda_device_supports_op`.
  - Nothing else in `ggml-cuda.cu` changes. The rope/set_rows fusions key on `GGML_OP_SET_ROWS` and do not match the new op.
- Tables: `#include "turbot-tables.cuh"`. WHT signs: `TURBO_WHT_SIGNS1/2` from `turbo-quant.cuh` (read-only use).

### 8.2 Row kernel `k_turbot_set_rows<idx_t>`

`__launch_bounds__(128)`: one CUDA block per (row, group), n_rows × 8 blocks, 4 warps × 32 lanes. It is modelled on `k_set_rows_turbo5p` and keeps its proven structure.

1. **Decode and addressing.** blockIdx → (row i, group index ig ∈ 0..7), h = ig/2, g = ig%2. The 32-bit division rule of `[TAG_TURBO5P]` applies.
   - `cell = src1[i]`, `yrow = young[i]`.
   - `base = dst + (size_t) cell·nb1`, `prow = pool + (size_t) yrow·nb_pool + part_off`.
   - b, y, r, `base_off[h]`, `young_off[h]` and the gain offsets arrive as kernel arguments (block-uniform).
2. **Load, norm and rotate.** Load x, then the norm tree, normalise, S1, butterfly (lane stages h < 32 in registers, then **the mandatory `__syncthreads()` barrier**, then the shared stages 32 and 64), then `x *= inv_sqrt_128·S2`. This is the turbo5p sequence verbatim. No InnerQ (turbot refuses `TURBO_INNERQ`).
3. **Index.** The branch on `yrow >= 0` is block-uniform.
   - Old-only row: `j = Σ_{k < 2^b−1} (u >= TURBOT_D_OLD_THR[old_off(b) + k])`.
   - Young row (default): one uniform count over all young thresholds, `iy = Σ_{k < 2^y−1} (u >= TURBOT_D_YOUNG_THR[young_off(b,y) + k])`, then `j = iy >> r` and `s = iy & (2^r − 1)` (`ggml_turbot_young_index`).
   - Every lane reads the same addresses, so the loads broadcast. `iy >> r` equals the old count by the bit-exact embedding (4.1). A fixed-length loop with `k < n` masking is allowed.
   - Alternative, only if C's writer perf case (11.2) measures it faster: the old count, then the branchless binary search inside the base cell, `s = 0; for (k = r−1; k >= 0; --k) s += (u >= THR[base + (j<<r) + s + (1<<k) − 1]) << k;`. It costs r data-dependent `__constant__` loads per element, which replay per distinct address.
4. **Pack the base code** into the planes of `base + base_off[h]`, element `128g + lane`:
   - P4: partner nibble via `__shfl_sync`, even lane stores (turbo5p).
   - P2: two `__ballot_sync` masks (field bit 0 and bit 1); lanes with `(lane & 3) == 0` store byte e/4 with bits 2k, 2k+1 from lane+k.
   - P1: one ballot; `(lane & 7) == 0` stores byte e/8 (turbo5p).
   - Refinement s is packed the same way into `prow + young_off[h]`.
   - Every byte of a run is written, so no pre-clear is needed. Both planes of each group are fully covered by its 128 lanes.
5. **Gain tables.** At block start, after the rotation barrier, copy `C_b` (2^b floats) and, for young rows, `LUT_{b,y}` (2^y floats) from `turbot-tables.cuh` into a `__shared__` float array. The run offsets are lane-uniform, so these loads broadcast. Size and raise the shared limit the way `k_set_rows_turbo5p` does.
6. **Gains.** Per lane: `cb = shared_C[j]`, `cy = shared_LUT[(j<<r)|s]`. Tree sums `recon_b² = Σ cb²` and `recon_y² = Σ cy²`. `gain = recon > 1e-10f ? norm/recon : norm`. Stores: `base[base_gain_off + 2(2h+g)] = __float2half(gain_b)`; if young, `prow[young_gain_off + 2(2h+g)] = __float2half(gain_y)`.
   - C may instead read the `__constant__` tables per element (one data-dependent read per element, what turbo5p's writer pays) if that measures faster.

### 8.3 Fill kernel `k_turbot_fill`

- One block per (fill entry, cell c ∈ 0..63, group ig): n_fill × 64 × 8 blocks, launched **before** the row kernel on the same stream.
- `if (!(mask bit c)) return;` is block-uniform.
- `cell = G·64 + c`, `prow = pool + (size_t)(slot·64 + c)·nb_pool + part_off`.
- Unpack j per element from the base planes. `s = TURBOT_D_FILL_CODE[fill_off(b,y) + j]`, then pack s. The fill-code run may be copied to shared memory like step 5.
- Gain: tree sums `rb² = Σ C_b[j]²`, `ry² = Σ LUT[(j<<r)|s]²` from the same shared copies. `gain_y = __half2float(base gain)·(sqrt(rb²)/sqrt(ry²))`, or the base gain if `sqrt(ry²) ≤ 1e-10f`; store half.

### 8.4 Launcher

- `ggml_cuda_op_turbot_set_rows` reads params, builds the layout, chooses the side and `part_off` (0 or `pool_v_off`), and launches fill (if `src[5]`) then rows.
- Both index types (I64 / I32) are instantiated.
- Assert `ne00 == 1024` and the row count. `n_rows == 0` launches nothing.

### 8.5 supports

`ggml_cuda_turbot_set_rows_supported` checks exactly the 5.2 asserts, returning false instead of asserting.

---

## 9. Host policy (owner D)

### 9.1 Selection and plan

**`common/arg.cpp`**
- `-ctk turbot` / `-ctv turbot`: special-case the string "turbot" to `GGML_TYPE_TURBOT_S8` before `kv_cache_type_from_str`. Mention "turbot" in the help list. Do not add the 17 family names to `kv_cache_types`.
- `--kv-tier-plan <file>` (env `LLAMA_ARG_KV_TIER_PLAN`) → `common_params::kv_tier_plan` (std::string).
- **Plan path.** The FIRST statement of `common_init_result::common_init_result` (common.cpp:1290) calls `llama_turbot_set_plan_path(params.kv_tier_plan.c_str())` when the string is non-empty. It must run before the `params.fit_params` block (common.cpp:1295), because `common_fit_params` creates probe contexts whose cache constructor needs the plan. Otherwise fit reads the "needs a plan" refusal as a memory failure. (Since [TAG_TURBOT_EMBED_PLAN] there is no such refusal: a late call would make the probes size the cache with the built-in plan instead of the given one.)
- **Drafter swap** in `common_base_params_to_speculative` (common/speculative.cpp:2908), right after `result.cache_type_k/v` are copied from the draft params:
  - each of the two fields that is `GGML_TYPE_TURBOT_S8` becomes `GGML_TYPE_TURBO5P_0`;
  - warn once when the user asked for it explicitly (`params_spec.cache_type_k_set` / `cache_type_v_set`).
  - This one function feeds the fit probe (common.cpp:1304), the server drafter (server-context.cpp:1287) and speculative-simple, so fit and the real drafter always agree. `[TAG_SPEC_KV_INHERIT]` in arg.cpp stays unchanged; it still copies `turbot` into the draft params, and the swap removes it.
- kv_size is always a multiple of 256, and so of 64: `llama_context` pads n_ctx to 256 (llama-context.cpp:295), fit's probe contexts included. No turbot rounding is needed.

**`src/llama-ext.h`**

```cpp
// [TAG_TURBOT] Process-wide turbot plan path; takes precedence over env LLAMA_TURBOT_PLAN. nullptr or "" clears it.
LLAMA_API void llama_turbot_set_plan_path(const char * path);
```

**Plan file grammar.** One item per line; `#` starts a comment; blank lines are ignored.

```
L <il> K <b0> <b1> <b2> <b3> V <b0> <b1> <b2> <b3>      required for every attention layer the cache holds
Y <il> K <y0> <y1> <y2> <y3> V <y0> <y1> <y2> <y3>      optional, default y = 7 for every head of that layer
POOL <cells>                                            optional, default 65536
CAP <cells>                                             optional, default 16384
W <n> | W2 <n> | M <bits>                               kvfq bench keys: accepted, ignored, one warning
```

**Validation** refuses with a message naming the line:
- an unknown tag;
- a malformed line, or a duplicate L/Y/POOL/CAP;
- b ∉ [2,6] or y ∉ [b+1, 8];
- a missing L line for an attention layer, or an L/Y line for a layer the cache does not hold;
- POOL not a multiple of 64 (a POOL above kv_size, explicit or the implicit default, is not refused: it is clamped to kv_size in whole granules with a warning, so llama-bench at low depth, perplexity at 32K and fit probes run; the 262144-cell server keeps POOL 65536; coordinator change 2026-09-15);
- CAP < 0.

POOL 0 is allowed as a diagnostic arm:
- every row is written old-only (young rows −1, gtab all −1, no fill);
- the pool tensor still has `GGML_TURBOT_POOL_MIN_ROWS` (64) rows, so every op keeps a non-NULL pool (9.4).

**Plan source precedence** (REVISED 2026-09-21 [TAG_TURBOT_EMBED_PLAN]; `llama_turbot_plan_get_source`, src/llama-kv-tier.cpp)
1. `llama_turbot_set_plan_path`
2. `LLAMA_TURBOT_PLAN`
3. Otherwise the built-in default plan (origin `LLAMA_TURBOT_PLAN_BUILTIN_AUTO`). There is no longer a "needs a plan" refusal.

An empty value counts as unset. The value `default` in 1 or 2 selects the built-in plan (`LLAMA_TURBOT_PLAN_BUILTIN_FORCED`); `./default` names a file.

**Built-in default plan**
- `docs/turbot/plans/turbot-default.plan` is compiled into libllama as a string. The checked-in `src/llama-turbot-default-plan.h` holds it, generated by `docs/turbot/gen_turbot_default_plan.py`.
- It is a checked-in header, not generated at configure time, so builds that compile `src/` without our CMake still get the plan.
- Drift guard:
  - `src/CMakeLists.txt` compares the header's `LLAMA_TURBOT_DEFAULT_PLAN_SHA256` with the SHA-256 of the plan file (LF line endings) and stops the configure on a difference. Both files are configure dependencies.
  - `gen_turbot_default_plan.py --check` exits 1 on a difference.
  - test-turbot checks that the built-in plan parses to hash `0x56c3503c949a7749`.
- `llama_turbot_plan_matches(text, attn_layers, kv_size, why)` is pure (no log line). It says whether a plan text names exactly the given attention layers. `kv_size` only clamps POOL. A context-level KV resolver calls it to fall back before the cache is built when the plan does not fit. The cache constructor itself still refuses a plan that does not fit, with the reason and a hint (`--kv-tier-plan <file>` or `-ctk turbo5p -ctv turbo5p`).
- With the built-in plan, `llama_turbot_plan::path` and the log lines say `<built-in default>`.

**Plan hash** (for blobs): `h = GGML_TURBOT_FNV_OFFSET`; for each attention layer in ascending il: `h = ggml_turbot_plan_hash_layer(h, il, &layer)`; then `h = ggml_turbot_plan_hash_finish(h, pool_cells, cap_cells)`. pool_cells is the value after validation.

### 9.2 Kill switch and context validation (`llama-context.cpp`)

- If `type_k` or `type_v` is `GGML_TYPE_TURBOT_S8`:
  - if env `LLAMA_TURBOT` is `"0"`: set both to `GGML_TYPE_TURBO5P_0`, warn `"LLAMA_TURBOT=0: turbot disabled, using turbo5p"`, and continue with the existing turbo5p checks;
  - otherwise both must be turbot, or refuse: `"turbot needs -ctk turbot and -ctv turbot together"`.
- Flash attention: AUTO → ENABLED; DISABLED → refuse (same pattern as the turbo K check).
- The existing quantized-K/V head-divisibility loops (`[TAG_MTP_KV_VALIDATE]`) skip turbot. Turbot validates its own geometry in the cache constructor (9.3).
- The `[TAG_TURBO5P512]` substitution never touches turbot.
- REVISED 2026-09-21 [TAG_KV_RESOLVE]: right after the `LLAMA_TURBOT=0` swap, `llama_kv_resolve` (src/llama-kv-cache-resolve.h) runs the 9.3 refusals and the plan match (`llama_turbot_plan_matches`) without building a cache. When turbot would be refused it falls back turbot → turbo5p (turbo5p512 for 512-element rows) → turbo4 → q8_0 → f16, with one `LLAMA_LOG_WARN` per downgrade that names every reason, so none of the refusals above or in 9.3 fails context creation. Should the cache constructor still refuse turbot, the `llama_context` constructor resolves again without it and rebuilds the memory. The other KV types go through the same resolver (head size, row and flash-attention rules). Env `LLAMA_KV_RESOLVE=0` turns it off and restores the refusals.
  - The refusal checks are shared between the resolver and the cache constructor: `llama_turbot_cache_refusal`, `llama_turbot_env_refusal`, `llama_turbot_layer_refusal` and `llama_turbot_layer_device_refusal` (src/llama-kv-cache-resolve.h), with the constructor's messages unchanged.
  - The attention layers the plan must name are derived with the same filter as `llama_model::create_memory` (`llama_kv_resolve_attn_layer`, a copy). A disagreement between the two is caught by the constructor retry above.
  - Plan check: `llama_turbot_plan_get_source` → `llama_turbot_plan_read` → `llama_turbot_plan_matches`, for every origin (9.1). A plan file that does not fit also falls back.
  - What a turbot request becomes (tests/test-kv-resolve.cpp, synthetic hparams):

    | Cause | Result |
    |---|---|
    | n_stream > 1, SWA, MLA or shared cells, a TURBO_* env switch, a layer off CUDA, a plan that does not fit | turbo5p (turbo5p512 when a row is 512 elements) if the model takes it, else further down the chain |
    | head geometry other than 4 × 256 | turbo5p if heads are 128/256 and rows a multiple of 512, else turbo4 for 128/256 heads, else q8_0, else f16 |
    | only one of -ctk/-ctv is turbot | the turbot side → turbo5p, then the pair rules apply (`-ctk turbot -ctv turbo5p` → turbo5p/turbo5p; a split-plane K against f16 V → q8_0/f16) |
    | FA explicitly off | K q8_0, V f16 |
    | an arch whose attention has no turbo query rotation (DeepSeek 3.2/V4, GLM DSA, Hy V4, dots3-note, MiniMax-M3, Qwen4exp QSA) | q8_0 (f16 where the head is not a whole q8_0 block) |

  - Qwen3.8-27B with the built-in or the default plan file keeps turbot with no new log line. Not yet built or run.

### 9.3 Refusals (thrown as `std::runtime_error` from the `llama_kv_cache` constructor unless noted)

Each message starts with `"turbot: "`.

REVISED 2026-09-21 [TAG_KV_RESOLVE]: with the resolver on (the default), every condition below is checked before the cache is built and leads to a fallback with a warning (9.2), not to a failed context. The constructor still throws when it is reached directly, or with `LLAMA_KV_RESOLVE=0`.

REVISED 2026-09-22 `[TAG_TURBOT_ANY_*]`: n_stream > 1, the head geometry and SWA are no longer refusals on their own. Section 14.8 lists what SUPPORTED means now, and 14.10 lists the switches that restore the rows of this table (`LLAMA_TURBOT_ANY=0` restores all of them).

| Condition | Message gist |
|---|---|
| n_stream > 1 (`!unified && n_seq_max > 1`) | needs `--kv-unified` or `-np 1` |
| v_trans (FA off) | needs flash attention |
| an attention layer with n_embd_head_k ≠ 256, n_embd_head_v ≠ 256 or n_head_kv ≠ 4 | unsupported head geometry |
| any attention layer whose buffer type is not a CUDA device, or `offload == false` | attention KV must be on CUDA |
| env `TURBO_KV_CPU_LAYERS` > 0 | incompatible |
| env `TURBO_LAYER_ADAPTIVE` > 0 | incompatible |
| env `TURBO_INNERQ` > 0 | incompatible |
| kv_size % 64 ≠ 0 | kv_size must be a multiple of 64 |
| SWA model (`swa_type != NONE` or n_swa > 0) | unsupported |
| shared cells (`mem_other != nullptr`) or MLA | unsupported |
| TriAttention enabled on this cache | unsupported |
| plan unreadable, invalid, or not naming exactly the attention layers (a missing plan is no longer possible: the built-in plan is used, 9.1) | 9.1 messages |

- `get_can_shift()` returns false when any layer is turbot.
- `seq_add` and `seq_div` on a turbot cache: `GGML_ABORT("turbot: seq_add/seq_div are not supported (no K shift)")`.

### 9.4 Cache construction (`llama-kv-cache.cpp`)

- Per attention layer: `S_K = Σ bk`, `S_V = Σ bv`, `k = ggml_new_tensor_3d(ctx, ggml_turbot_type_of_s(S_K), 1024, kv_size, 1)`, same for v.
- `pool = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, layer.pool_row_bytes, std::max(N_R, GGML_TURBOT_POOL_MIN_ROWS))`, named `cache_%sturbot_young_l%d`. With POOL 0 the 64 rows are never addressed; the tier has 0 slots.
- `attn_rot_k = attn_rot_v = false` unless `LLAMA_TURBOT_ATTN_ROT=1` (10.3).
- Add the pool to the ctx mem_size count (+1 tensor per layer).
- No turbo rotation matrices, no InnerQ tensor: the Q WHT takes `innerq_scale = nullptr`.
- Buffers are cleared to 0 as today.
- VRAM formula (3.6). If the total exceeds the turbo5p size for the same kv_size and layer count, warn. Do not refuse.
- Logs:
  - `llama_kv_cache: turbot plan <path>: <n> layers, old bits <mean> (sum <S>), young pool <N_R> cells (<N_R/64> granules), cap <CAP>, hash 0x<16 hex>`
  - `llama_kv_cache: size = %7.2f MiB (%6u cells, %3d layers, %2u/%u seqs), K (turbot): %7.2f MiB, V (turbot): %7.2f MiB, young pool: %7.2f MiB` — K and V are base bytes only; the size is base + pool.
  - Default plan: `size = 5246.00 MiB ... young pool:  726.00 MiB`.

### 9.5 Tier state (`src/llama-kv-tier.h`, `src/llama-kv-tier.cpp`, owner D; E unit-tests the public interface)

```cpp
struct llama_turbot_plan {
    std::string                               path;
    std::map<int32_t, ggml_turbot_layer>      layers;      // by model layer il
    uint32_t                                  pool_cells = GGML_TURBOT_POOL_DEFAULT;
    uint32_t                                  cap_cells  = GGML_TURBOT_CAP_DEFAULT;
    uint64_t                                  hash       = 0;
};

// attn_layers: the il values the cache holds (every one needs an L line). Returns false and sets err on refusal.
bool llama_turbot_plan_parse_text(const std::string & text, const std::vector<int32_t> & attn_layers, uint32_t kv_size,
                                  llama_turbot_plan & plan, std::string & err);
bool llama_turbot_plan_parse_file(const std::string & path, const std::vector<int32_t> & attn_layers, uint32_t kv_size,
                                  llama_turbot_plan & plan, std::string & err);

class llama_kv_tier {
public:
    llama_kv_tier(uint32_t kv_size, uint32_t pool_cells, uint32_t cap_cells);

    uint32_t n_granules() const;                          // kv_size / 64
    uint32_t n_slots() const;                             // pool_cells / 64 (0 for POOL 0)
    const std::vector<int32_t> & granule_slots() const;   // [n_granules], slot or -1 (granule table contents)
    bool     cell_young(uint32_t cell) const;             // gslot >= 0 && ref_valid bit && !pending bit

    // before compute (9.6). cells[i]: destination cell of ubatch row i; the ubatch is already applied to kvc
    void begin_ubatch(const llama_ubatch & ubatch, const std::vector<uint32_t> & cells, const llama_kv_cells & kvc);
    const std::vector<int32_t> & young_rows() const;      // [n_rows], pool row or -1
    const std::vector<int32_t> & fill_entries() const;    // 4*n_fill: granule, slot, (int32) mask_lo, (int32) mask_hi

    void commit_ubatch(const llama_kv_cells & kvc);       // after a successful compute
    void abort_ubatch();                                  // after a failed compute, before the failure seq_rm

    // sequence hooks, called by llama_kv_cache for turbot caches (9.7)
    void on_cell_emptied(uint32_t cell);
    void on_seq_tail_removed(llama_seq_id seq, uint64_t min_removed_stamp);
    void on_seq_cp(llama_seq_id src, llama_seq_id dst, bool dst_was_empty);
    void clear();

    // state restore (9.10). counters: destination seq ids with their saved values. Returns the pool row of each
    // restored cell or -1 (refinement dropped). May reclaim or evict slots (9.6 alloc_slot).
    std::vector<int32_t> restore_cells(const std::vector<uint32_t> & cells, const std::vector<uint64_t> & stamps,
                                       const std::vector<uint8_t> & young, const std::vector<std::pair<llama_seq_id, uint64_t>> & counters,
                                       const llama_kv_cells & kvc);
    void abort_restore();                                 // the pool bytes of the last restore_cells could not be read

    uint64_t stamp(uint32_t cell) const;
    uint64_t row_counter(llama_seq_id seq) const;
    uint64_t n_evictions() const;                         // granules with live cells evicted by alloc_slot (tests, DEBUG=1)
};
```

**`llama_kv_cells` (llama-kv-cells.h)**
- Gets `uint32_t seq_n_cells(llama_seq_id s) const`.
  - It counts (cell, s) pairs, maintained in `seq_pos_inc` (+1) and `seq_pos_dec` (−1), and zeroed in `reset()`.
  - Those two functions are the only choke points for (cell, seq) membership, so every mutation path (including `prepare()`'s speculative apply and restore through `set()`) stays exact.
- Gets a `const std::bitset<LLAMA_MAX_SEQ> & seq_bits(uint32_t i) const` accessor if no equivalent exists.
- This is host-only bookkeeping and does not change any existing behaviour.

**State held by the tier**

| Member | Contents |
|---|---|
| `gslot` | [n_granules] i32 |
| `owner` | [n_slots] i32 granule or −1 |
| free slots | `std::set<int32_t>`, lowest first |
| `last_touch` | [n_slots] u64: `touch_serial` of the last begin_ubatch or restore_cells that allocated the slot or wrote a row into its granule |
| `touch_serial` | u64, incremented at the start of every begin_ubatch and restore_cells |
| `ref_valid` | [n_granules] u64 bitmask, bit c = refinement of cell G·64+c is valid |
| `stamp` | [kv_size] u64 |
| `row_ctr` | [LLAMA_MAX_SEQ] u64. With kv_unified, decode accepts every seq id below LLAMA_MAX_SEQ (llama-context.cpp:1753), matching `seq_to_stream.size()`. |
| `cut`, `has_cut` | [LLAMA_MAX_SEQ] i64 and a bitset, from the last commit |
| `in_flight` | granules touched by the current ubatch: vector in first-touch order plus an [n_granules] u8 flag |
| `restored` | slots allocated by the last restore_cells; cleared by begin_ubatch, commit_ubatch and abort_restore |
| `pending` | `std::map<uint32_t, uint64_t>` granule → cells that still need fill |
| `victims` | eviction candidates, built lazily inside one begin_ubatch or restore_cells call (9.6), discarded at its end |

### 9.6 Ubatch lifecycle

**Quota**, computed in commit_ubatch:

```
n[s]     = kvc.seq_n_cells(s);   live = { s : n[s] > 0 };   n_active = |live|;   sum = Σ n[s]
N_eff    = max(0, pool_cells − 64·GGML_TURBOT_QUOTA_SLACK_GRANULES·n_active)
Y[s]     = sum ? min(cap, floor((uint64) N_eff · n[s] / sum)) : 0
cut[s]   = (int64) row_ctr[s] − (int64) Y[s];   has_cut = live
```

- **Why the slack.** Granules are fixed 64-cell ranges that find_slot does not align. A band of Y cells that starts off a granule boundary touches up to Y/64 + 2 granules. Without slack, 4 sequences at 16,384 each need about 1,028 granules against 1,024 slots and would evict a wanted granule on every step. Refinement cannot come back (no transcode).
- Default plan: one sequence keeps Y = 16,384 (the measured configuration); four equal sequences get 16,256. VRAM is unchanged.

**Slot allocation `alloc_slot(G)`**, used by begin_ubatch and restore_cells:
1. The lowest free slot.
2. Otherwise build `victims`, on the first need within this call. Candidates are every owned slot whose granule is not in `in_flight` and, during restore, whose slot is not in `restored`. Order ascending by:
   - class: 0 if all 64 cells of the granule are empty (`kvc.is_empty`), else 1;
   - margin (class 1 only): max over live cells c of the granule and s ∈ `seq_bits(c)` of `(int64) stamp[c] − cut[s]`; INT64_MAX if some such s has no cut yet (it appeared after the last commit);
   - `last_touch[slot]`, then slot.
   Take the first entry whose slot is still owned and unprotected.
3. Free the victim: `gslot[G'] = −1`, `ref_valid[G'] = 0`, `pending.erase(G')`, `owner = −1`. Class 1 increments `n_evictions`.
4. Nothing left → −1.

- **Why empties first.** A slot whose granule lost all its cells (server release `seq_rm(id, −1, −1)`, trims, park) is otherwise returned only at the next commit. A first prefill ubatch (up to 32 granules per 2,048 rows), a prompt-cache restore after a release (server-task.cpp:1981), or a resume right after a park (up to 256 granules) would evict live granules of other slots while reusable slots exist.
- **Why a margin.** After any commit every kept slot was "wanted at this epoch", so an epoch key ties and degenerates to the lowest slot id, which can be the granule written one ubatch ago. The margin orders each sequence's band newest-first relative to its own cut, so each sequence loses its oldest-in-band granule first.
- **Cost.** The victim build walks owned slots × 64 cells and runs only when the free set is empty. With the quota slack and commit's reclamation that is rare; `LLAMA_TURBOT_DEBUG=1` logs how often.

**`begin_ubatch`** is called from `llama_kv_cache_context::apply()`, right after `kv->apply_ubatch(...)`, only for turbot caches.
- `prepare()` and `state_read_meta()` call `apply_ubatch` directly and must NOT call `begin_ubatch`.

```
if in_flight not empty: abort_ubatch()                           // defensive
restored.clear(); touch_serial++
for each row i:  for k < ubatch.n_seq_id[i]: row_ctr[ubatch.seq_id[i][k]]++
                 stamp[cells[i]] = row_ctr[ubatch.seq_id[i][0]]
touched = granules of cells, first-touch order; in_flight = touched
written[G] = OR of bits (cell & 63) of this ubatch's cells in G
for G in touched (first-touch order), then G in keys(pending) not in touched (ascending granule id):
    if gslot[G] < 0 and G in touched:
        slot = alloc_slot(G)
        if slot < 0: continue                                      // G stays old; its rows get young -1 (log once per 1000)
        gslot[G] = slot; owner[slot] = G; ref_valid[G] = 0
    if gslot[G] < 0: pending.erase(G); continue
    if G in touched: last_touch[gslot[G]] = touch_serial
    live          = bitmask of non-empty cells of G (kvc.is_empty)
    ref_valid[G] &= live                                           // stale bits of emptied cells never survive (9.8)
    need          = live & ~ref_valid[G] & ~written[G]
    if need: fill += {G, gslot[G], (int32)(need & 0xffffffff), (int32)(need >> 32)}; ref_valid[G] |= need
    pending.erase(G)
young_rows[i] = gslot[G_i] >= 0 ? ggml_turbot_pool_row(gslot[G_i], cells[i]) : -1; if >= 0: ref_valid[G_i] |= bit
```

- Pending bits are always a subset of `~ref_valid`, because abort and restore clear the valid bit whenever they set pending. `need` therefore covers them.
- A row listed in several sequences advances every listed counter and takes its stamp from the first. The server never builds such rows; it shares prefixes with seq_cp.

**`commit_ubatch`** is called from `llama_context::decode` right after `process_ubatch` returns non-null (after the `if (!res) { ... }` block), for the memory's attention cache. Find it with the same `dynamic_cast` pattern as `llama_memory_attn_n_free_ext`: `llama_kv_cache` or `llama_memory_hybrid::get_mem_attn()`.

```
compute n, Y, cut, has_cut (quota above); live_mask = bitset of live
for slot ascending with owner G >= 0:
    wanted = false; any_live = false
    for c in the 64 cells of G:
        if kvc.is_empty(c): continue
        any_live = true
        for s in kvc.seq_bits(c) & live_mask:
            if (int64) stamp[c] > cut[s]: wanted = true; break
        if wanted: break
    if !any_live: free(slot); continue
    if !wanted and G not in in_flight: free(slot)
in_flight.clear(); restored.clear()
```

- free(slot): `gslot[owner] = −1`, `ref_valid[owner] = 0`, `pending.erase(owner)`, `owner = −1`, and the slot returns to the free set.
- Cost: a wanted granule stops at its first wanted cell, so steady state costs about one or two cells per owned slot plus 64 per freed granule. Each cell costs one bitset AND against at most the live sequences, never a loop over LLAMA_MAX_SEQ.

**`abort_ubatch`** is called in the failure branch of `llama_context::decode` BEFORE the existing `memory->seq_rm` loop.
- For every fill entry of the current ubatch: `ref_valid[G] &= ~mask`, `pending[G] |= mask`. The fill may not have executed.
- Then `in_flight.clear()`.
- Slots allocated by this ubatch stay; the next commit reclaims them. No demotion.
- The failure `seq_rm(s, pos_min, −1)` that follows rolls the counters back (9.7).

**Encoder, graph reserve and update contexts** never call begin/commit.

**Debug env (D)**
- `LLAMA_TURBOT_DEBUG=1`: one log line per commit: slots used, freed, empty slots reclaimed by alloc_slot, evictions (class 1), fills, young cells per sequence.
- `LLAMA_TURBOT_DEBUG=2`: the full 9.8 check after every begin_ubatch and restore_cells.
- `LLAMA_TURBOT_FAIL_UBATCH=<n>`: treat the n-th `process_ubatch` (1-based, process-wide) as failed with `GGML_STATUS_FAILED` after it returns (failure-injection test).

### 9.7 Sequence operations

`llama_kv_cache` calls the hooks for turbot caches only (and never when `other` is set; turbot refuses shared cells).

| Operation | Tier action |
|---|---|
| `seq_rm(s ≥ 0, p0, p1)` | `tail = (p1 < 0 or p1 == max)`, taken before p1 is normalised. In the existing loop (llama-kv-cache.cpp:683-694), before `cells.seq_rm(i, s)`: `min_st = min(min_st, tier->stamp(i))`. If the call empties the cell: `tier->on_cell_emptied(i)`. After the loop, if `tail` and something was removed: `tier->on_seq_tail_removed(s, min_st)`. The early return (`seq_pos_max(s) < p0`) stays. |
| `seq_rm(−1, p0, p1)` | Before `cells.rm(i)`: for every s in `seq_bits(i)`, `min_st[s] = min(min_st[s], stamp(i))`; then `on_cell_emptied(i)`. After the loop, if `tail`: `on_seq_tail_removed(s, min_st[s])` for every s seen. |
| `seq_cp(src, dst)` (same stream, metadata only) | `dst_was_empty = cells.seq_n_cells(dst) == 0` BEFORE the copy loop; after it `tier->on_seq_cp(src, dst, dst_was_empty)`. |
| `seq_keep(s)` | `on_cell_emptied(i)` for every cell the call empties. No counter change. |
| `clear(data)` | `tier->clear()`: all slots free; stamps, counters, cuts, touch serial, ref_valid, pending and restored zeroed. |

```
on_cell_emptied(c):        G = c >> 6; ref_valid[G] &= ~bit(c); if G in pending: pending[G] &= ~bit(c), erase when 0
on_seq_tail_removed(s, m): row_ctr[s] = m > 0 ? min(row_ctr[s], m − 1) : 0
on_seq_cp(src, dst, e):    row_ctr[dst] = e ? row_ctr[src] : max(row_ctr[dst], row_ctr[src])
```

- **Why the rollback.** The server removes rejected draft rows with `seq_rm(id, pos_next, −1)` (server-context.cpp:5289, also 4511 and 5235). Without a rollback, every rejected row leaves a gap in stamp space. "stamp > row_ctr − Y_s" then covers only the accepted fraction: about 60-75% of the band at DFlash2 n_max 3.
- A removal that is not a stamp tail (rare) can give a later row the stamp of a live cell. That only makes one more cell look wanted.
- **Why adoption on seq_cp.** Server `copy_state_to` does `seq_rm(other, −1, −1)` then `seq_cp(id, other)` (server-context.cpp:833-834). With max(), a follower whose stale counter is far ahead sees the copied prefix as ancient and demotes it as soon as the leader releases.
- Stamps and counters live in per-sequence spaces. Shared cells carry the source's stamp, and after adoption both sequences start from the same counter.

### 9.8 Invariant

For every granule G with `gslot[G] >= 0`:
1. every live cell of G whose pending bit is clear has its `ref_valid` bit set;
2. no empty cell of G has a `ref_valid` or pending bit;
3. after begin_ubatch, no young granule has pending bits.

Readers of a young granule therefore never see a refinement that was not written, and `cell_young` never reports an empty or unfilled cell.
- Graph inputs are uploaded after begin.
- Fill runs in each layer's writer before that layer's FA.
- `LLAMA_TURBOT_DEBUG=2` asserts 1-3 after every begin_ubatch, and 1-2 after every restore_cells.

### 9.9 Free-count API

`llama_memory_attn_n_free_ext` is unchanged. find_slot placement is unchanged (decision 4), and the young pool is best effort, so park admission is unaffected.

### 9.10 State blob v2 (`state_write` / `state_read`)

The existing stream section (cell_count, meta, data) is unchanged. For turbot caches the data section is followed by:

```
u32  magic         GGML_TURBOT_BLOB_MAGIC (0x32544254)
u32  version       GGML_TURBOT_BLOB_VERSION (2)
u64  plan_hash
u32  cell_count    == this stream's cell_count
u32  n_counters
     n_counters × { i32 seq_id; u64 row_ctr }     seq blob: exactly one, the saved seq; full blob: every seq with seq_n_cells > 0
     cell_count × u64 stamp                        blob cell order (ascending cell ranges, as meta)
     cell_count × u8  young                        1 iff tier.cell_young(cell) at save time
u32  n_young       number of young == 1
u32  n_layer
     n_layer × { u32 il; u32 pool_row_bytes }      layer table, ascending il
     n_layer × { n_young × pool_row_bytes raw bytes }   same layer order, young cells in blob order
```

**Write rules**
- The layer table precedes every pool byte.
- Pool bytes are written with at most one `io.write_tensor` per run of consecutive young cells that are consecutive pool rows (same granule, consecutive cells).

**Read rules (`state_read_turbot`, after `state_read_data`, same stream)**
1. **Validate before touching the tier.** Read and check everything up to the raw bytes:
   - magic, version, plan_hash;
   - cell_count;
   - counters: seq ids < LLAMA_MAX_SEQ, and exactly one for a seq blob;
   - stamps and flags, with n_young == the number of flags set;
   - n_layer and every (il, pool_row_bytes) against this cache.
   Any mismatch → return false with the tier untouched. The existing failure path clears or `seq_rm`s and throws.
2. **`tier->restore_cells(sinfo.idxs[0], stamps, young, counters, kvc)`:**
   - Counters: `row_ctr[s] = saved`, an assignment, never max().
     - For a seq blob, s is `dest_seq_id`, which is empty here because `state_read_meta` ran `seq_rm(dest_seq_id, −1, −1)` first (llama-kv-cache.cpp:2763). For a full blob the cache was cleared.
     - The server prompt cache restores into any slot id (server-task.cpp:1981). With max(), a slot whose old counter is far larger would see the whole restored band as ancient and demote it at the next commit.
   - `restored.clear()`, `touch_serial++`, `stamp[cell] = saved` for every restored cell.
   - For each destination granule holding at least one young restored cell, ascending: if it has no slot, `alloc_slot(G)` (protected: in_flight and restored), push the slot to `restored`, set `last_touch`.
   - For each restored cell c of granule G:
     - young flag and `gslot[G] >= 0` → pool row, set the ref_valid bit, clear the pending bit;
     - otherwise → −1, clear the ref_valid bit, and set the pending bit if `gslot[G] >= 0`.
   - For each granule newly allocated by this call: every other live cell gets its ref_valid bit cleared and its pending bit set.
3. **Stream the pool bytes** per layer: runs of consecutive young cells with consecutive pool rows ≥ 0 go straight into the pool with `io.read_tensor(pool, (size_t) row·nb1, run·pool_row_bytes)`. Bytes of dropped cells are read into a bounded scratch buffer (64 cells at a time).
4. **On failure in step 3** (short blob, io exception): `tier->abort_restore()`, then rethrow or return false. The failure path `seq_rm`s the restored cells afterwards.
   - `abort_restore` frees every slot in `restored` (gslot −1, ref_valid 0, pending erased) and clears the ref_valid bits of restored cells in granules that were young before the call.
   - Evictions made in step 2 are not undone. A truncated blob is logged as such.
   - Why the bytes are not read first: a 4-slot full blob carries up to 761 MB of pool bytes (65,536 cells × 11,616 B). `llama_io_read_i` offers only copying reads, so reading them before restore_cells would double that in host RAM on the park/resume path.

**Refusals**
- A turbo5p blob into turbot, or turbot into turbo5p: the existing per-layer type check fails first. Log it once as `"turbot: state blob type mismatch (turbo5p vs turbot)"` from the turbot side.
- A turbot blob from another plan: hash mismatch.

### 9.11 Hybrid memory

`llama_memory_hybrid` needs no change: attention restore still runs before recurrent restore, and the commit hook reaches the attention cache through `get_mem_attn()`. D verifies `llama-memory-hybrid.cpp` compiles unchanged, and only edits it if a turbot accessor is required.

---

## 10. Graph (owner D, `src/llama-graph.{h,cpp}`, `src/llama-kv-cache.{h,cpp}`)

### 10.1 Inputs (`llm_graph_input_attn_kv`)

```cpp
ggml_tensor * self_turbot_gtab  = nullptr;   // I32 [n_granules]            slot or -1
ggml_tensor * self_turbot_young = nullptr;   // I32 [n_tokens]              pool row or -1
ggml_tensor * self_turbot_fill  = nullptr;   // I32 [4, n_fill], only when n_fill > 0
```

- **Created** in `build_attn_inp_kv_impl` when the cache is turbot. The pos-mask decision is unchanged.
- **Filled** in BOTH `llm_graph_input_attn_kv::set_input` and `llm_graph_input_mem_hybrid::set_input`:
  - gtab = `tier.granule_slots()`;
  - young = `tier.young_rows()`;
  - fill = `tier.fill_entries()`;
  - via a new `llama_kv_cache_context::set_input_turbot(gtab, young, fill)`, host buffers asserted like `set_input_k_idxs`.
- **Uploaded** every ubatch (16 KiB).
- **`can_reuse`** additionally requires the conditions below. They go into BOTH `llm_graph_input_attn_kv::can_reuse` (llama-graph.cpp:496) AND `llm_graph_input_mem_hybrid::can_reuse` (llama-graph.cpp:1132), and into any other `can_reuse` that owns an `llm_graph_input_attn_kv`. The hybrid one re-implements the attention checks inline instead of delegating, and it is the one Qwen3.8 uses.
  - `self_turbot_gtab->ne[0] == n_granules`;
  - `self_turbot_young->ne[0] == n_tokens`;
  - `(self_turbot_fill != nullptr) == (n_fill > 0)`;
  - if present, `self_turbot_fill->ne[1] == n_fill`.

  A ubatch with fills rebuilds the graph. That is rare, and it happens after begin, which `process_ubatch` runs first via `mctx->apply()`. Patching only the attn_kv version would let the hybrid path reuse a graph without the fill tensor: fills would never run, and young granules would read unwritten refinement.

### 10.2 `build_attn` (the `llm_graph_input_attn_kv` overload; the hybrid qwen35 path uses it through `inp->get_attn()`)

When the cache is turbot:

```
k_row = ggml_view_2d(k_cur as [1024, n_tokens])      // 4 heads x 256, no padding
v_row = same for v_cur
expand ggml_turbot_set_rows(ctx0, cache_k(il), k_row, k_idxs, pool(il), young, fill, &params_K)
expand ggml_turbot_set_rows(ctx0, cache_v(il), v_row, v_idxs, pool(il), young, fill, &params_V)
k = get_k(il), v = get_v(il)                          // same views as turbo5p (head 256)
q = ggml_turbo_wht(ctx0, cont(q), 0, 0, nullptr)      // forward WHT-128, as turbo5p
cur = build_attn_mha(..., kv_pos, q_pos, pool(il), gtab, &params_BOTH)
```

- `build_attn_mha` gains trailing params `ggml_tensor * turbot_pool = nullptr, ggml_tensor * turbot_gtab = nullptr, const ggml_turbot_op_params * turbot_params = nullptr`.
- After `ggml_flash_attn_ext_set_pos` it calls `ggml_flash_attn_ext_set_turbot` when `turbot_params != nullptr`.
- The V inverse-WHT type lists (FA and non-FA branches) and the Q-WHT list in `build_attn` add `ggml_turbot_is_type(...)`. The group size is 128.
- `llama_type_is_turbo()` in `llama-kv-cache.cpp` is NOT extended. Turbot sites check `ggml_turbot_is_type` explicitly.

### 10.3 Rotations

- **Default off.** `attn_rot_k` and `attn_rot_v` are forced to false in the `llama_kv_cache` constructor when type_k or type_v is turbot. This goes in the `!other` branch (llama-kv-cache.cpp:555-580), after the existing computation, with the log line `turbot: upstream Hadamard off (quality was measured without it; LLAMA_TURBOT_ATTN_ROT=1 enables it)`.
- **What upstream would do.** It computes the flags as `ggml_is_quantized(type) && head % 64 == 0`, which is true for every turbot type. llama-graph.cpp:2970-2976 would then apply `llama_mul_mat_hadamard` (K nrot 256, V nrot 64) before the writer.
- **Why off.** The kvfq bench behind every quality number and the per-head fq_3t0 widths wrote into an F16 cache, where attn_rot is off. With it on, turbot would code H256·k and H64·v through WHT-128, which changes both the per-group norm split and the coordinate distribution the widths were fitted to.
- WHT-128 is already the incoherence rotation. The Q-side WHT and V inverse WHT of 10.2 are unchanged.
- **Opt-in.** Env `LLAMA_TURBOT_ATTN_ROT=1` keeps the upstream rule. `LLAMA_ATTN_ROT_DISABLE` keeps its meaning.
- Phase 6 runs the `LLAMA_TURBOT_ATTN_ROT=1` arm before anyone changes the default.
- turbo5p and every other type are unaffected: the override sits in a turbot-only branch.

---

## 11. Test plan (owner E)

No test or tool runs during implementation. These are written now and executed at the integration build under the one-GPU-process rule, never on port 8080.

### 11.1 `tests/test-turbot.cpp` (CPU only, links ggml; registered in `tests/CMakeLists.txt`)

1. **Tables**
   - Recompute kvfq `lloyd_max_gauss(b)` in C++ (copy the function) for b = 2, 3, 6. Old levels must equal the header bit for bit.
   - C4/C5 literals; mids = `0.5f·(c+c)`; antisymmetry; `T_b[2^{b−1}−1] == 0`.
   - Young thresholds sorted, with bit-exact embedding at `((j+1)<<r)−1`.
   - `ggml_turbot_young_off` / `fill_off` closed forms equal the generated OFF lists.
   - `ggml_turbot_old_level_i8(4|5, k)` equals the int8 tables of `turbo-quant.cuh` (copy the two literal lists into the test with a pointer to their source).
2. **Nesting.** Inputs: 10⁶ values from N(0, 1/128), plus every threshold, `nextafter` below and above each, ±0, ±1, ±inf and NaN. For all b 2..6 and y b+1..8, require:
   - `refine_index == young_index & (2^r−1)`;
   - `young_index >> r == old_index`.
3. **Planes.** Set/get roundtrip for w 1..6, all 256 elements, random codes, neighbours untouched.
4. **Coder**
   - Primary gate: the old tier equals a C++ copy of `llama-kvfq.cpp fq_group` output bit for bit (b 2..6).
   - Gaussian rows (16,000 unit-norm groups through the WHT, f16 gains): old nmse / analytic Gaussian D_b within [0.93, 1.03].
     - Measured 1.012 / 0.978 / 0.965 / 0.945 / 0.959 for b = 2..6.
     - The sphere marginal and the gain correction put b ≥ 3 below the Gaussian integral, so a ±3% band fails on a correct coder.
   - Young nmse ≤ 1.18 × an independent y-bit Lloyd code (local copy of `lloyd_max_gauss(y)`, same sphere MC and f16 gains). Measured worst ×1.089 (b5 y8) and ×1.082 (b5 y7).
   - Zero row → exact zeros.
5. **Fill.** For every (b, y): MSE(fill read, old read) ≤ 0.30·D_b and MSE(fill read, true value) ≤ 1.30·D_b, with D_b the analytic Gaussian distortion.
   - Measured worsts: 0.200 and 1.198, both at b2 y3 (table in 4.4).
   - Record the measured values in TESTING.md.
6. **Op params / layout**
   - Pack → set → get → layer roundtrip.
   - Default-plan layout: base row bytes per layer match 3.2.
   - Pool row bytes sum 11,616 per pool cell, base 18,080 per cell.
   - Illegal widths are rejected.
7. **Plan parser and tier** (links llama; tier scenarios drive a real `llama_kv_cells`). `llama_turbot_plan_parse_text` accept/refuse cases, then `llama_kv_tier` scripted scenarios:
   - single sequence decode → young band = Y_s live cells (16,384 with the default plan);
   - 4 interleaved sequences at quota: after warm-up `n_evictions()` stays 0, and every sequence's newest 16,256 cells are young;
   - eviction order with the pool forced full: an empty owned slot is taken first, then the smallest margin; a granule written in the previous ubatch is never taken while a smaller-margin granule exists;
   - release then new prefill (`seq_rm(s, −1, −1)` then begin): empty owned slots are reused and nothing live is evicted; same for restore right after a release;
   - draft loop: write 4 rows, remove the last 2 with `seq_rm(s, p, −1)`, repeat 10,000 times. The young band equals Y_s live cells, and `row_counter(s)` equals the live row count;
   - seq_cp into an empty dst adopts the counter: after the leader releases, the follower's copied prefix stays young at the next commit;
   - seq restore into a slot id whose counter is larger than the saved one: the restored band stays young after the next commit;
   - stale bits: in a young granule, `seq_rm` a cell, restore a cell into that position, then `state_write` before any begin. `cell_young` is false for the new cell, and no ref_valid or pending bit sits on an empty cell;
   - trim then rewrite into an old granule → fill entry;
   - failure → abort keeps state and re-queues the fill; the failure `seq_rm` rolls the counters back;
   - restore failure after restore_cells → `abort_restore` frees the slots it allocated;
   - POOL 0: all young rows −1, gtab all −1, 0 slots;
   - seq id LLAMA_MAX_SEQ − 1 with kv_unified;
   - determinism (identical scripts → identical states).

### 11.2 `tests/test-backend-ops.cpp` additions

- **`test_turbot_set_rows`**
  - Output comparison: build writer → FA (turbot) over the written cells, so the harness compares F32 attention outputs.
  - Cases: rows {1, 4, 16, 1280}; I64 and I32 idx; young rows all −1 / all valid / mixed; fill present / absent; K widths ≠ V widths; heads with b = 2 and b = 6.
- **`test_flash_attn_ext_turbot`** (mask and pos-mask variants), max nmse 5e-4:
  - hs 256, 4 KV heads, GQA 6;
  - kv {96, 100, 1000, 1024, 4096, 16384}. 100 and 1000 are not multiples of the tile span, so the last `is_fixup` block runs; these use non-uniform widths (K ≠ V, b = 2 and b = 6 heads in one layer, one head at y = 8);
  - nb {1, 2, 4, 8, 16, 512, 1280}. nb 512 and 1280 run with `FA_NCOLS128` on (the default), so the <128,1> / <64,2> / <32,4> / <16,8> instances execute. These MUST pass before any perf run (7.4 LUT placement);
  - granule patterns: all old / all young / alternating / young only for the last partial tile;
  - pos-mask sequence starting at a cell not a multiple of 64, with holes;
  - one sinks case, one softcap case;
  - both `GGML_CUDA_TURBOT_OLD_I8` builds.
- **Hybrid graph reuse** (llama-level test in `tests/test-turbot.cpp` or a small driver): a reused hybrid graph followed by a ubatch that needs a fill must rebuild and run the fill (10.1).
- **Perf cases (gate B0, 7.8)**, not correctness: `turbot_perf` FA cases against turbo5p.
  - kv 32768 / 131072 / 245760; nb 1 / 4 / 512; mixes `old`, `band16k`, `band64k`, `young`.
  - Cache bytes are encoded on the host with `ggml-turbot.h`.
  - Plus a writer perf case: turbot set_rows at 2,048 rows against turbo5p set_rows (8.2 alternatives).

### 11.3 `tests/test-turbot-backend.cpp` (CPU vs CUDA bytes)

- Writer base and pool bytes, CUDA vs CPU reference: codes agree on ≥ 99.99% of elements, gains within one f16 step.
- The fill kernel likewise.
- Old-tier int8 read (`GGML_CUDA_TURBOT_OLD_I8` 1): FA values decoded through the CUDA loaders for b = 2..5 equal `ggml_turbot_old_level_i8 · scale · gain` within one half step.
- Run under `compute-sanitizer memcheck` first (the `cand_safe.py` flow).

### 11.4 Tools (`tools/turbot/`)

| Tool | Purpose |
|---|---|
| `check_tables_vs_study.py` | runs `gen_turbot_tables.py --check`, then compares with `kv_nested_study.build_designs(..., ['a_lloyd'])`. Tolerances: exact for b = 4, 5; ≤ 2e-7 for b = 2, 3; ≤ 6e-5 for b = 6 (4.6). |
| `plan_vram.py <plan> [--kv 262144]` | prints base / pool / total MiB and the plan hash. Must print 4520.00 / 726.00 / 5246.00 for the default plan. |
| `b0_gate.py <perf log turbot> <perf log turbo5p>` | prints the 7.8 ratio table and GO / NO-GO |
| `blob_roundtrip.py` | server park/resume and prompt-cache restore, including turbo5p ↔ turbot refusal and restore into a different slot id |

### 11.5 `docs/turbot/TESTING.md` (E)

Gates in order, each with a command and pass criterion:
1. **CPU unit tests** (11.1 items 1-6).
2. **FA correctness** (11.2 FA cases, ncols128 included) and 11.3.
3. **Gate B0** (7.8). NO-GO stops here.
4. **Unit and correctness, full stack**
   - `validate.ps1`
   - tier tests (11.1 item 7), writer tests
   - `crosstalk.py` cold/warm
   - `overcommit_accept.py` (4×62K + 6,000)
   - `pool_exact.py`
   - prefill tails n = 2-5
   - save/restore round trip, restore into a different slot
   - trim test
   - failure injection (`LLAMA_TURBOT_FAIL_UBATCH`) with 4 slots and a unique-value check
   - hybrid prepare-order check
5. **Quality**
   - KLD / same-top gates (16/32 × 32K, code and prose, ub 512 and 1280, against floor and q8_0, paired chunk bootstrap)
   - the `LLAMA_TURBOT_ATTN_ROT=1` arm (10.3)
   - 131K depth
   - needles 32K/131K/200K/250K at 1 and 4 agents
   - DFlash2 acceptance ≥ turbo5p within ±0.02
   - vision checks
6. **Speed and VRAM**
   - end-to-end speed (plan section 6), quoted as realistic t/s
   - VRAM log line `size = 5246.00 MiB`, no F16 scratch

### 11.6 Default plan file

`docs/turbot/plans/turbot-default.plan` (E): the appendix A contents verbatim.

---

## 12. File ownership and shared symbols

### 12.1 Ownership (disjoint)

| Owner | Files (create or edit) |
|---|---|
| **Architect** | `docs/turbot/SPEC.md`, `docs/turbot/gen_turbot_tables.py`, `ggml/include/ggml-turbot.h`, `ggml/include/ggml-turbot-tables.h` (generated), `ggml/src/ggml-cuda/turbot-tables.cuh` |
| **A** ggml core + CPU reference | `ggml/include/ggml.h`, `ggml/src/ggml.c`, `ggml/src/ggml-quants.c` (validate_row_data only, if needed), `ggml/src/CMakeLists.txt` (list the two new headers in ggml-base), `ggml/src/ggml-cpu/ggml-cpu.c`, `ggml/src/ggml-cpu/ggml-cpu.cpp`, `ggml/src/ggml-cpu/ops.cpp`, `ggml/src/ggml-cpu/ops.h` |
| **B** CUDA FA read path | `ggml/src/ggml-cuda/fattn-mma-f16.cuh` (the array-size fix only), `ggml/src/ggml-cuda/fattn.cu`, `ggml/src/ggml-cuda/fattn-turbot.cuh` (new), `ggml/src/ggml-cuda/fattn-turbot-decl.cuh` (new), `ggml/src/ggml-cuda/template-instances/fattn-mma-turbot-instance-ncols1_*-ncols2_*.cu` (20 new), `ggml/src/ggml-cuda/CMakeLists.txt` (no change expected). `fattn-common.cuh` is read-only for B (the int8 LUTs are reused, not edited). |
| **C** CUDA writer + dispatch | `ggml/src/ggml-cuda/turbot-set-rows.cu` (new), `ggml/src/ggml-cuda/turbot-set-rows.cuh` (new), `ggml/src/ggml-cuda/ggml-cuda.cu` |
| **D** llama host side | `src/llama-kv-cache.h`, `src/llama-kv-cache.cpp`, `src/llama-kv-cells.h`, `src/llama-kv-tier.h` (new), `src/llama-kv-tier.cpp` (new), `src/CMakeLists.txt`, `src/llama-graph.h`, `src/llama-graph.cpp`, `src/llama-context.h`, `src/llama-context.cpp`, `src/llama-memory-hybrid.h`, `src/llama-memory-hybrid.cpp`, `src/llama-ext.h`, `common/arg.cpp`, `common/common.h`, `common/common.cpp`, `common/speculative.cpp`, `tools/server/*` (only if required; none expected) |
| **E** tests, tools, docs | `tests/test-turbot.cpp` (new), `tests/test-turbot-backend.cpp` (new), `tests/test-backend-ops.cpp`, `tests/CMakeLists.txt`, `tools/turbot/*` (new), `docs/turbot/TESTING.md` (new), `docs/turbot/plans/turbot-default.plan` (new) |

- Line endings: the working tree is CRLF (`core.autocrlf=true`; `fattn-common.cuh` is `-text` with CRLF). Every edit keeps the file's existing endings, and new files use CRLF like their neighbours.
- `tools/perplexity/perplexity.cpp` has an unrelated uncommitted change in this worktree. Nobody touches it.

### 12.2 Shared symbols (exact names; the owner defines, others consume)

| Symbol | Signature / value | Owner | Consumers |
|---|---|---|---|
| everything in `ggml-turbot.h` | as written, including `ggml_turbot_young_index`, `ggml_turbot_old_level_i8`, `ggml_turbot_old_i8_scale`, `GGML_TURBOT_POOL_MIN_ROWS`, `GGML_TURBOT_QUOTA_SLACK_GRANULES` | Architect | A B C D E |
| `GGML_TURBOT_*_LIST`, `GGML_TURBOT_*_TOTAL` | generated macros | Architect | A (via header) B C E |
| `TURBOT_D_OLD_LEVELS`, `TURBOT_D_OLD_THR`, `TURBOT_D_YOUNG_LUT`, `TURBOT_D_YOUNG_THR`, `TURBOT_D_FILL_CODE`, `turbot_d_old_off`, `turbot_d_young_off`, `turbot_d_fill_off` | `turbot-tables.cuh` (turbot instance TUs and turbot-set-rows.cu only) | Architect | B C |
| `GGML_TYPE_TURBOT_S8` … `GGML_TYPE_TURBOT_S24` | enum 49 … 65, `GGML_TYPE_COUNT = 66` | A | all |
| `GGML_TYPE_TURBOT_S2` … `GGML_TYPE_TURBOT_S7` | enum 66 … 71, `GGML_TYPE_COUNT = 72` (`[TAG_TURBOT_ANY_TYPES]`, 14.3) | WP1 | all |
| geometry symbols of 14.1 and 14.4 (`ggml_turbot_geom_*`, `ggml_turbot_side_init_nr`, `ggml_turbot_layer_init_geom`, `side.nr`, `layer.flags`) | `ggml-turbot.h` (`[TAG_TURBOT_ANY_GEOM]`) | WP1 | WP2 WP3 WP4 |
| type names | `"turbot_s8"` … `"turbot_s24"` | A | E (logs) |
| `GGML_OP_TURBOT_SET_ROWS` | enum, directly before `GGML_OP_COUNT` (= 103) | A | C D E |
| `ggml_turbot_set_rows` | `struct ggml_tensor * (struct ggml_context *, struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * c, struct ggml_tensor * pool, struct ggml_tensor * young, struct ggml_tensor * fill, const struct ggml_turbot_op_params *)` | A | D E |
| `ggml_flash_attn_ext_set_turbot` | `void (struct ggml_tensor * a, struct ggml_tensor * pool, struct ggml_tensor * gtab, const struct ggml_turbot_op_params *)` | A | D E |
| FA src slots | src[7] = pool (I8, never NULL), src[8] = gtab (I32) | A | B D E |
| `ggml_compute_forward_turbot_set_rows` | `void (const struct ggml_compute_params *, struct ggml_tensor *)` in `ops.h` | A | A |
| `FATTN_MMA_TURBO_MODE_COUNT` | `constexpr int = 5` in `fattn-mma-f16.cuh` | B | B |
| `ggml_cuda_flash_attn_ext_turbot_case` | `template <int DKQ, int DV, int ncols1, int ncols2> void (ggml_backend_cuda_context &, ggml_tensor *)`, 20 instances (7.6) | B | B |
| `DECL_FATTN_TURBOT_CASE`, `ggml_cuda_fattn_mma_case_dispatch`, `turbot_head_state_of`, `GGML_CUDA_TURBOT_OLD_I8` | 7.1, 7.2, 7.4, 7.6 | B | B E (B0 builds) |
| `ggml_cuda_op_turbot_set_rows` | `void (ggml_backend_cuda_context & ctx, ggml_tensor * dst)` | C | C |
| `ggml_cuda_turbot_set_rows_supported` | `bool (const ggml_tensor * op)` | C | C |
| `llama_turbot_set_plan_path` | `LLAMA_API void (const char * path)` in `src/llama-ext.h` | D | D (common) |
| `llama_kv_cells::seq_n_cells`, `llama_kv_cells::seq_bits` | `uint32_t (llama_seq_id) const`, `const std::bitset<LLAMA_MAX_SEQ> & (uint32_t) const` | D | D E |
| `llama_turbot_plan`, `llama_turbot_plan_parse_text`, `llama_turbot_plan_parse_file`, `llama_kv_tier` | section 9.5 | D | E |
| `llama_turbot_default_plan_text`, `llama_turbot_default_plan_hash`, `llama_turbot_plan_get_source`, `llama_turbot_plan_read`, `llama_turbot_plan_matches` | `src/llama-kv-tier.h`, 9.1 ([TAG_TURBOT_EMBED_PLAN]) | D | D (resolver, cache constructor) E |
| `llama_kv_resolve`, `llama_kv_resolve_type_name`, `llama_turbot_cache_refusal`, `llama_turbot_env_refusal`, `llama_turbot_layer_refusal`, `llama_turbot_layer_device_refusal` | `src/llama-kv-cache-resolve.h`, 9.2 ([TAG_KV_RESOLVE]) | D | D E (test-kv-resolve) |
| `common_params::kv_tier_plan` | `std::string` | D | D |
| env | `LLAMA_TURBOT`, `LLAMA_TURBOT_PLAN` (a path, or `default`), `LLAMA_TURBOT_DEBUG`, `LLAMA_TURBOT_FAIL_UBATCH`, `LLAMA_TURBOT_ATTN_ROT`, `LLAMA_ARG_KV_TIER_PLAN`, `LLAMA_KV_RESOLVE` ([TAG_KV_RESOLVE]), `TURBOT_Q2_ROUTE` (7.6) | D | E |
| CLI | `-ctk turbot`, `-ctv turbot`, `--kv-tier-plan <file>` or `--kv-tier-plan default` | D | E |
| plan file | `docs/turbot/plans/turbot-default.plan` | E | D (docs), E |

### 12.3 Integration order

1. The architect's files are already in place.
2. A lands types, ops and the CPU reference. Until then, nothing including `ggml-turbot.h` compiles, because the header names `GGML_TYPE_TURBOT_S8`.
3. B lands the `shared_memory_limit_raised` fix as its own change, then the turbot kernel. E's CPU unit tests build after A.
4. First build of A + B + E's FA and perf cases: `turbot_build.ps1` (scratchpad), the register check (7.4), FA correctness (11.2), then gate B0 (7.8). NO-GO stops the integration.
5. C and D are integrated against A's API. They may be written in parallel with B from the start.
6. E's backend tests after B and C; tier tests after D.
7. Codegen check at the first full build:
   - every existing template-instance object must be byte-identical before and after, relocations masked;
   - the constant memory map of the existing instance TUs must be unchanged. They include no turbot header, so any difference is a defect;
   - `fattn.cu` and `ggml-cuda.cu` change host code only.

---

## 13. Notes for reviewers

- **Existing bug, fixed by B first:** `shared_memory_limit_raised` is `[4]` and `[3]` but indexed by turbo_mode up to 4. The softcap array overflows for turbo5p512.
- **Existing bugs, reported and not fixed** (turbo5p512 only, whose model files were deleted):
  1. `ggml_cuda_flash_attn_ext_mma_f16_case` maps turbo_mode `TURBO5P512` to the `FATTN_MMA_TURBO_NONE` kernel while passing `need_f16 = false`.
  2. The last-iteration `[TAG_TURBO4P_HEAD]` block in `flash_attn_ext_f16` omits `TURBO5P512`.

  Fixing either would change existing codegen, so both are out of scope. The second is why turbot requires one head-state helper at both sites (7.2).

  REVISED 2026-09-21 [TAG_TURBO5P512_MMA]: both are fixed on the `upstream-sync` branch.
  - `ggml_cuda_fattn_mma_f16_select_kernel` is an exhaustive switch over the turbo modes, so turbo5p512 gets its own D = 256 MMA kernels (with and without softcap). An unknown mode aborts instead of running the f16 kernel.
  - The last-iteration block includes turbo5p512.
  - The native-read predicate checks turbo5p512 against its 512-element block, so 512-element rows read natively too.
  - turbot, turbo4, turbo4p and turbo5p keep their kernels. `TURBO_MMA_NATIVE=0` restores the F16 conversion path.
  - Test: `test-backend-ops -o FLASH_ATTN_EXT -p split_plane`. Not yet built or run.
- **Speed:** turbot reads 0.90× turbo5p's bytes at 262K with one sequence, 1.00× with four at quota, and 1.42× below 65K context (7.8). The target is parity at long context, decided by gate B0 before C and D are integrated.
- **Known limits** (as of the upstream sync):
  - Prefill is 4-8% slower than turbo5p at 131K-245K.
  - A cached long prompt can decode slightly differently from the same prompt sent cold.
  - Two-token verify batches no longer run the <2,8> instance: they take <4,8>, as Q = 1 does (7.6, `[TAG_TURBOT_Q2_ROUTE]`). That route is on by default and `TURBOT_Q2_ROUTE=0` restores <2,8>. It stays on only if the nb 2 A/B at kv 131072 / 245760 (TESTING.md 3) shows it neutral or faster.
- **Deviation from 7.2, work-balanced stream_k blocks (`[TAG_TURBOT_FA_BALANCE]`, read-speed plan item 2).** 7.2 says `launch_fattn_turbot` keeps the stream_k block layout and reuses the fixup kernels unchanged. That no longer holds for layouts that already need a fixup (`ntiles_dst % nblocks != 0`, which covers every decode and prefill shape at depth).
  - **Reason:** every stream_k block launches at once, so the op waits for its slowest block. At 131K nb 4 there are 340 blocks, 85 per output tile, and each lies inside one KV head. The young band is the newest cells of every head, so a uniform slice puts the all-young tail blocks last, and those gate the op: a young tile costs 1.22× an old one at ncols 32 and about 1.10× at ncols ≥ 64 (B0, after item 4). Balancing the work across blocks is expected to move G1 from 1.35 toward about 1.14.
  - **What changed:**
    - Uniform-fixup layouts, which include every decode shape (`nblocks` a multiple of `ntiles_dst`), use striped blocks (`[TAG_TURBOT_FA_STRIPE]`):
      - Block `tile·stripe + j` processes KV tiles `j, j + stripe, j + 2·stripe, …` of its output tile, instead of a contiguous run.
      - Every block therefore sees the same old/young mix, with no pass over gtab. The KV_min/KV_max skips stay on the stripe.
      - Block `j = stripe − 1` writes dst with needs_fixup and the others write is_fixup parts, so the common `flash_attn_stream_k_fixup_uniform` is launched unchanged.
      - The seam-table design below was measured first on these layouts. Its two passes cost more than the young penalty they remove (131K nb 4: prefix +32 us, seams +16 us, moved seams +26 us, against a 36 us penalty), which is why striping replaced it there.
    - General-fixup layouts (prefill): two small turbot kernels run before the FA kernel. `flash_attn_turbot_young_prefix` counts young KV tiles over gtab. `flash_attn_turbot_balance_bounds` computes the seams of equal estimated work, with integer weights per head and tier from the B0 ratios.
    - The kernel entry reads its `[kbc, kbc_stop)` from the seam table.
    - The fixup runs through `flash_attn_turbot_stream_k_fixup_general` (a copy of the common general fixup that reads the same table; the default) or `flash_attn_turbot_stream_k_fixup_tile` (one launch block per output tile, `LLAMA_TURBOT_FA_BALANCE_FIXUP=tile`). The combine order and arithmetic are unchanged.
    - Layouts whose blocks cover whole output tiles keep the uniform slice.
  - **Measured** (B0, same session, turbot / turbo5p, `E:/kv-turbot/fix5/r5/b0_table.txt`; balancing off → default):
    - G1 1.307 → 1.159; G2 1.321 → 1.125; G3 0.986 → 0.872; G4 1.285 → 1.242.
    - G5 1.151 → 1.158, unchanged within noise: the seam table at nb 512 is neutral.
    - The nb 2 band16k cell goes from 2.007 to 1.511. All-old cells are unchanged.
  - **What it costs:**
    - Every unit is still processed exactly once, so values are unchanged apart from the float bits at the moved seams.
    - Weights affect speed only.
    - Stored bytes and quality are unchanged.
    - Existing kernels and their codegen are untouched: only the turbot header changes.
  - **Switches:** `LLAMA_TURBOT_FA_BALANCE=0` restores the uniform slice and the common fixup kernels. `LLAMA_TURBOT_FA_BALANCE_STRIPE=0` sends uniform-fixup layouts through the seam table as well. `LLAMA_TURBOT_FA_BALANCE_DIAG=1|2` runs the passes without using the seams, to measure their cost. `LLAMA_TURBOT_FA_BALANCE_SEED=<n>` replaces the weights with per-shape random ones so the seams land anywhere, including empty blocks, for correctness runs.
- **Fill quality (corrected):** a filled cell reads at about the old tier's own error: ×1.12 for b = 6 heads under y = 7, ×1.03 for b = 5, within ×1.01 otherwise (4.4). New rows are always refined. The Phase 6 trim arm and the 4-agent interleave runs measure the end-to-end effect.
- **Pool exhaustion:** with 2 granules of quota slack per sequence, a wanted granule is evicted only under very scattered placement. Example: a sequence decoding interleaved with another's long prefill gets one cell per granule. The margin key then drops each sequence's oldest-in-band granules first. `LLAMA_TURBOT_DEBUG=1` counts evictions. If every slot is in flight (a ubatch touching more than 1,024 granules), the extra rows are written old-only and logged.
- **Truncated state blob:** evictions made by `restore_cells` before the byte read fails are not undone (9.10).
- **Upstream Hadamard:** off by default for turbot, an opt-in arm (10.3).
- **Study codebook deviation:** 4.6. The generator is authoritative.
- **Other models, 2026-09-21 (`upstream-sync` branch).** turbot stays a Qwen3.8-27B-shaped cache: 4 KV heads × 256, no SWA, one stream, and a plan that names exactly the attention layers. Every other model is served by the resolver fallback (9.2), not by turbot.
  - The README section "Using the fork with any model" is the user-facing summary. It covers the resolve rules, the built-in plan, the vision device (`-mmdev cpu|gpu|igpu` with the `--device CUDA0 --spec-draft-device CUDA0` pinning rule), `--load-mode none`, the new switches and the architectures that came with the sync.
  - TESTING.md section 8 lists the checks. None of them has run yet.
  - Two turbot-relevant switches came with the sync, both default on:
    - `TURBOT_Q2_ROUTE` (7.6).
    - `TURBO_RMSNORM_SCALE_FUSION`, the CUDA GDN q/k norm fusion. It is on the Qwen3.8 decode path and meant to be bit-exact; that is not yet checked.
  - REVISED 2026-09-22 `[TAG_TURBOT_ANY_*]`: turbot now also serves other shapes, iSWA models and several streams. Section 14 is the contract.

---

## 14. turbot on other shapes (`[TAG_TURBOT_ANY_*]`, 2026-09-22)

This section extends turbot from the one Qwen3.8-27B shape (4 KV heads × 256) to every model whose attention layers fit one of six geometries. It is the contract of four work packages with disjoint files, merged in this order:

| Package | Scope |
|---|---|
| WP1 | ggml core (types, ops, `ggml-turbot.h`), CPU reference, `tests/test-turbot-geom.cpp`, this section, TESTING.md 9, README |
| WP2 | CUDA reader and writer for the new geometries, their routing and `GGML_TURBOT_ANY`; the new `test-backend-ops` and `test-turbot-backend` cases; `tools/turbot/b0_gate.py` (G4) and `tools/turbot/sass_diff.py` (G2) |
| WP3 | host cache and tier: NR-run plan lines, plan precedence, automatic plan, sidecar, per-stream tiers and the `LLAMA_TURBOT_*` switches (`src/llama-kv-tier.*`, `src/llama-kv-cache.*`, `common`); `tests/test-turbot.cpp`; `tools/turbot/turbot_plan.py`, `plan_vram.py`, `blob_roundtrip.py` and `turbot_guard.py` (G5) |
| WP4 | host resolver (`[TAG_KV_RESOLVE]`), iSWA split, graph and context (`src/llama-kv-cache-resolve.h`, `src/llama-context.cpp`, `src/llama-graph.*`, the iSWA caches, `src/llama-model.cpp`); `tests/test-kv-resolve.cpp` |

Tags: `[TAG_TURBOT_ANY_GEOM]` (geometry), `[TAG_TURBOT_ANY_TYPES]` (S2..S7) and the other `[TAG_TURBOT_ANY_*]` tags of each package.

**Status: implemented, not yet built or run.** The sizing numbers in 14.7 come from a CPU-only script over the GGUF headers (`scratchpad/autoplan_models.py`), not from a measurement. The gates in TESTING.md section 9 decide the defaults (14.11).

**Hard requirement.** Qwen3.8-27B (hybrid GDN, 16 attention layers 3, 7, …, 63, 4 KV heads × 256, built-in plan hash `0x56c3503c949a7749`) keeps exactly today's turbot behaviour, kernels and numerics. Every code path at geometry flags 0 is either unchanged or constant-folds to today's code. Gate G2 checks this bit for bit.

### 14.1 Geometry flags

A layer's geometry travels as one byte, `flags`, in `struct ggml_turbot_layer` and at op-params byte 31 (5.3). Bits 0-1 hold log2(4/NR), where NR is the number of 256-value runs in a row. Bit 2 (`GGML_TURBOT_GEOM_D128`) says the head dim is 128.

| Head dim × KV heads | flags | NR | Values per row | WHT groups per row | Model on disk |
|---|---|---|---|---|---|
| 256 × 4 | 0 | 4 | 1024 | 8 | Qwen3.8-27B (sections 1-13), Ornith-1.5-9B, Spark-X2.5-4B |
| 256 × 2 | 1 | 2 | 512 | 4 | Ornith-1.5-35B |
| 256 × 1 | 2 | 1 | 256 | 2 | none |
| 128 × 8 | 4 | 4 | 1024 | 8 | none |
| 128 × 4 | 5 | 2 | 512 | 4 | none |
| 128 × 2 | 6 | 1 | 256 | 2 | MiniCPM5-2B, Muse Glimmer 30B, Nemotron 3.5 30B |

- flags 3 and 7 (log2 field 3) are invalid, and so is any value with a bit above `GGML_TURBOT_GEOM_MASK` (0x07).
- `ggml_turbot_geom_flags(head_dim, n_head_kv)` returns the flags of a shape, or −1 when turbot has no layout for it. That covers head dims other than 128 and 256, 128 × 1 (half a run), and rows above 1024 values (256 × 8, 128 × 16).
- Accessors in `ggml-turbot.h`:
  - `ggml_turbot_geometry_supported(head_dim, n_head_kv)`;
  - `ggml_turbot_geom_valid(flags)`;
  - `ggml_turbot_geom_nr(flags)` = 4 >> (flags & 3);
  - `ggml_turbot_geom_head_dim(flags)` = 128 or 256;
  - `ggml_turbot_geom_n_head(flags)` = NR·256 / head dim;
  - `ggml_turbot_geom_row_elems(flags)` = NR·256;
  - `ggml_turbot_geom_n_groups(flags)` = 2·NR (the writer's NG).
- New constants: `GGML_TURBOT_RUN_ELEMS` 256, `GGML_TURBOT_MAX_RUNS` 4, `GGML_TURBOT_S_MIN_ANY` 2, `GGML_TURBOT_GEOM_LOG2_MASK` 0x03, `GGML_TURBOT_GEOM_D128` 0x04, `GGML_TURBOT_GEOM_MASK` 0x07.
- `GGML_TURBOT_HEAD_DIM` (256), `GGML_TURBOT_N_HEAD` (4) and `GGML_TURBOT_ROW_ELEMS` (1024) keep their values and their meaning at flags 0. `GGML_TURBOT_ROW_ELEMS` is now documented as the container width (14.3).

### 14.2 Run mapping

A run is 256 values with one old width `b[r]` and one young width `y[r]`, and it is two WHT-128 groups. In every geometry:
- run r holds values [256r, 256r + 256);
- group g of run r holds values [256r + 128g, 256r + 128g + 128);
- the old gain of (run r, group g) sits at byte `32·S + 2·(2r + g)` of the base row, and the young gain at byte `32·R + 2·(2r + g)` of the side's pool part.

KV head z of a cell:
- **Head dim 256:** head z is run z. At flags 0 this is 3.1 unchanged.
- **Head dim 128:** head z is run z >> 1 at element base 128·(z & 1), which is group z & 1 of run z >> 1. The two heads of a run share its widths.

In both cases head z covers values [D·z, D·z + D) of the decoded row. That is why the CPU reference's F32 stand-in uses a cell stride of NR·256 values and a head stride of D (14.5).

### 14.3 Rows, container tensors and types

- **Base row** of a layer-side: the NR runs (planes per 3.4), then 8 `ggml_fp16_t` gain slots, of which slots 2·NR … 7 stay zero. The size is 32·S + 16 bytes for any NR, with S = Σ b[r] over the NR runs.
- **Young part:** the NR refinement runs, then 8 gain slots: 32·R + 16 bytes, R = Σ (y[r] − b[r]). Pool rows keep the 3.3 layout, `[K part][V part]`.
- **Container tensor.** The base cache stays `[1024, kv_size]` of type `turbot_s<S>` for every geometry. 1024 is the block size of every turbot type, so one cell is one block of 32·S + 16 bytes whatever NR is. The K and V views are `[D, n_kv, H, 1]` with nb[1] = 32·S + 16, built as for Qwen3.8 (10.2).
- **Types.** S ranges over 2..6 at NR 1, 4..12 at NR 2 and 8..24 at NR 4.
  - S8..S24 keep ids 49..65.
  - `GGML_TYPE_TURBOT_S2` … `GGML_TYPE_TURBOT_S7` are appended as ids 66..71, and `GGML_TYPE_COUNT` becomes 72. No existing id moves, so state files and the `GGML_TYPE_TURBOT_S8` sentinel stay valid.
  - Traits as in 5.1: names `turbot_s2` … `turbot_s7`, blck_size 1024, type_size 80 / 112 / 144 / 176 / 208 / 240, quantized, no to_float.
  - `ggml_validate_row_data` accepts them without validation. Vulkan refuses ids 43..71 (`[TAG_VK_NO_TURBO]`).
  - `ggml_turbot_is_type` covers 49..65 and 66..71. `ggml_turbot_type_of_s` and `ggml_turbot_s_of_type` handle S 2..24 over both ranges. `GGML_TURBOT_S_MIN` stays 8, the first S of the 49..65 block; `GGML_TURBOT_S_MIN_ANY` is 2.
- **Examples.**
  - NR 1, b 3, y 7: S 3, base row 112 B, young part 144 B.
  - NR 2, b {2, 6}, y {8, 7}: runs at bytes 0 and 64, base row 272 B; young runs at 0 and 192, young part 240 B.
- **VRAM:** the 3.6 formula holds, with the sums taken over the NR runs of each layer-side.

### 14.4 Header contract C1 (`ggml/include/ggml-turbot.h`, WP1)

| Symbol | Contract |
|---|---|
| `struct ggml_turbot_side` | New LAST member `uint8_t nr`. Runs r ≥ nr have b = y = 0 and offsets 0. |
| `struct ggml_turbot_layer` | New LAST member `uint8_t flags`. |
| `ggml_turbot_side_init_nr(sd, b, y, nr)` | nr ∈ {1, 2, 4}. Reads b[0..nr) and y[0..nr) only. Returns false on an illegal nr or width. The old `ggml_turbot_side_init(sd, b, y)` is nr = 4. |
| `ggml_turbot_layer_init_geom(l, bk, bv, yk, yv, flags)` | Returns false on invalid flags. Zeroes the whole struct first, so two equal layers compare equal with memcmp. The old `ggml_turbot_layer_init(...)` is flags = 0. |
| `ggml_turbot_encode_side`, `ggml_turbot_decode_side`, `ggml_turbot_fill_side` | Loop over `sd->nr` runs instead of 4, gains at 2r + g. A row takes or gives NR·256 values. For nr = 4 the arithmetic and its order are unchanged. |
| op params | Byte 31 `flags` is the geometry; the version stays 1. `ggml_turbot_op_params_make` copies `l->flags`. `ggml_turbot_op_params_get` rejects `flags & ~0x07` and an invalid geometry. `ggml_turbot_layer_from_op_params` passes the flags, and it also rejects a width in a run ≥ nr. |
| `ggml_turbot_plan_hash_layer` | Folds `{'G','E','O','1', flags}` after the widths, only when flags ≠ 0. Qwen3.8-27B's hash `0x56c3503c949a7749` does not change. |

Every existing inline function gives the same result for nr = 4 and flags = 0. `tests/test-turbot-geom.cpp` (c), (d) and (f) compare them with copies of the pre-change code.

### 14.5 ggml ops

- `ggml_turbot_set_rows`: `a->ne[0] == 1024` (the container), `b->ne[0] == ggml_turbot_geom_row_elems(params->flags)` and `a->type == ggml_turbot_type_of_s(side.s)`. The rest of 5.2 is unchanged.
- `ggml_flash_attn_ext_set_turbot`: the K and V views have `ne[0] == ggml_turbot_geom_head_dim(flags)` and `ne[2] == ggml_turbot_geom_n_head(flags)`. `ne[3] == 1` stays; a cache with several streams passes one view per stream.
- **CPU reference** (6.1, 6.2):
  - The writer takes NR·256-value rows and runs the header coder over the side's nr runs.
  - The FA decodes every cell to NR·256 floats and runs the unchanged scalar kernel on F32 stand-ins with cell stride NR·256 and head stride D, GQA broadcast included. At flags 0 this is the old code.
  - The CPU supports every geometry. It is the test oracle, so it has no switch.
- **CUDA** (WP2): readers and writers for the geometries of 14.1. `GGML_TURBOT_ANY=0` accepts only flags 0 and NG 8, which is today's kernel routing.
  - FA: the Qwen predicate (D 256, 4 KV heads) is tested first and unchanged; any other geometry is admitted only when it fails. D 128 head z is run z>>1 at element 128·(z&1); a D 128 tile holds at most 64 cells (one granule). The 16 D 128 instances (`fattn-mma-turbot-d128-instance-*.cu`) are dropped by the CMake option `GGML_CUDA_FA_TURBOT_D128=OFF`, which also refuses D 128 turbot in the routing.
  - Writer: NG = 2·NR WHT groups per row (`k_turbot_set_rows<idx, LOG2_NG>`); NG 8 compiles the old kernels.
  - ggml-cuda exports `ggml_backend_turbot_supports_geometry`, which the resolver asks (14.8): a geometry of 14.1, Turing+ MMA, `GGML_TURBOT_ANY` on for anything but 4 × 256, and the D 128 instances for head dim 128.

### 14.6 Plan text and precedence

- An `L` or `Y` line for a layer with nr runs has exactly 4 + 2·nr tokens: `L <il> K <b0> … <b(nr−1)> V <b0> … <b(nr−1)>`. nr = 4 is the old 12-token form. A line with the wrong count is refused with its line number.
  - NR 2: `L 3 K 5 4 V 5 4`
  - NR 1: `L 0 K 5 V 4`
- The plan gives the widths. The geometry comes from the model (`ggml_turbot_geom_flags` of each layer's head dim and KV heads), and the plan hash folds it (14.4).

**Plan precedence.** The first match wins:
1. `--kv-tier-plan` or `LLAMA_TURBOT_PLAN`:
   - `<file>` uses only that file. A mismatch falls back down the type chain and never becomes an automatic plan.
   - `default` uses the built-in plan.
   - `auto` uses the automatic plan.
2. A sidecar `<model>.turbot.plan`, used only when it has a `# verified:` stamp and a matching `# model:` fingerprint.
3. The built-in plan, when it names exactly the model's attention layers and geometry. Qwen3.8-27B and fine-tunes with the same layers stop here, bit-identical to today.
4. The automatic plan (14.7), only for the main context (ctx_type DEFAULT) with the resolver on. Every layer geometry must be in the VALIDATED list (14.11), or be allowed by `LLAMA_TURBOT_AUTO_PLAN=all` (any geometry) or `LLAMA_TURBOT_AUTO_BUDGET=turbo5p` (NR 1 layers), and the plan must fit the budget (14.7) with old widths 4 or 5.

### 14.7 Automatic plan v1 (WP3)

A deterministic integer model. It reads no calibration data.
1. kvt = kv_size · n_stream.
2. Budget B = kvt · Σ over layers of 2 · `ggml_row_size(budget_type, row_elems)`. `budget_type` is one type for the whole cache, the one the resolver falls back to (`llama_turbot_budget_type`):
   - `TURBO5P_0` (656 B per 1024 values) when every layer's row_elems % 1024 == 0;
   - else `TURBO5P512_0` (336 B per 512 values) when every row_elems % 512 == 0;
   - else `TURBO4_0` (136 B per 256 values).

   With `LLAMA_TURBOT_AUTO_BUDGET=turbo5p` every row is budgeted at turbo5p's rate, 656 B per 1024 values.
3. POOL P = min(roundup64(n_seq_max · (16384 + 128)), rounddown64(kvt)), then rounded down to a multiple of 64 · n_stream.
4. Old widths are 4 or 5 only. For m from 2·nr down to 0: K gets ceil(m/2) runs at width 5 and V gets floor(m/2), lowest run first, and the other runs get 4. Every layer uses the same pattern, within its own nr. Take the largest m with base + P · young ≤ B (bytes as in 3.6, with y = 7).
5. If even m = 0 does not fit, shrink P. Refuse (fall back down the type chain) if P < n_seq_max · (1024 + 128).
6. y = 7 everywhere, CAP 16384.
7. Plan text, in this order: the line `# turbot auto plan v1`; comment lines with the shape, the budget type and `X MiB (<type> Y MiB)`; the L lines; POOL; CAP. `LLAMA_TURBOT_AUTO_PLAN_DUMP=<file>` writes it to a file.

Sizing. The budget is the bytes of the type the model would otherwise fall back to. 262,144 cells and one sequence unless noted. Not measured.

| Model | Attention layers, shape | Mean old width | Plan | Fallback |
|---|---|---|---|---|
| Ornith-1.5-9B | 8, 4 × 256 | 4.75 (POOL 16512) | 2572.6 MiB | turbo5p 2624 MiB |
| Ornith-1.5-9B, 32K | 8, 4 × 256 | 4.0 | 327.7 MiB | turbo5p 328 MiB |
| Spark-X2.5-4B, full-attention layers | 9, 4 × 256 | 4.75 | 2894.2 MiB | turbo5p 2952 MiB |
| Ornith-1.5-35B | 10, 2 × 256 | 4.75 (4.0 at 4 sequences, 1622.0 MiB) | 1650.4 MiB | turbo5p512 1680 MiB |
| MiniCPM5-2B, Muse Glimmer 30B, Nemotron 3.5 30B | 42 / 13 / 6, 2 × 128 | refused: 4-bit old rows alone cost 144 B per 256 values, turbo4 136 B | — | turbo4 |
| the same, `LLAMA_TURBOT_AUTO_BUDGET=turbo5p` | 2 × 128 | 4 | +316 / +98 / +45 MiB of KV over turbo4 | turbo4 |

### 14.8 Default policy

`-ctk turbot -ctv turbot` keeps turbot for a model only when the model is SUPPORTED and has a PLAN (14.6). Everything else steps down the existing chain turbot → turbo5p / turbo5p512 → turbo4 → q8_0 → f16, with one WARN line per step (9.2).

SUPPORTED means all of these hold:
- flash attention is on;
- no MLA and no shared cells;
- no `TURBO_*` environment switch (9.3);
- the arch has the turbo query rotation;
- kv_size is a multiple of 64;
- K and V are both turbot;
- every turbot layer has head_k == head_v, a geometry in {256 × 4, 256 × 2, 256 × 1, 128 × 8, 128 × 4, 128 × 2}, and its KV on CUDA with Turing+ MMA. The resolver checks the CUDA device, `GGML_TURBOT_ANY=0` (other geometries step down), and the backend's answer through the optional proc `ggml_backend_turbot_supports_geometry` when the backend exports it. ggml-cuda exports it (WP2, 14.5): the geometry, Turing+ MMA, `GGML_TURBOT_ANY` and the D 128 instances, so a layer the build or device cannot run steps down here instead of reaching a refused FA. Under `LLAMA_TURBOT_ANY=0` the resolver does not ask it, as before;
- SWA appears only as the SWA half of an iSWA split (14.9); all-SWA models are refused;
- n_stream > 1 is allowed, with a tier per stream (14.9).

Expected defaults on the models on disk, before the gates:
- Ornith-1.5-9B: turbot.
- Spark-X2.5-4B: turbot on the full-attention layers, turbo5p on the SWA layers.
- Ornith-1.5-35B: turbot.
- MiniCPM5, Muse Glimmer and Nemotron: turbo4 unless opted in.
- The DFlash2 and MTP draft contexts never use turbot (unchanged, 9.1).

### 14.9 iSWA split and streams

- **iSWA.** An iSWA model has two caches. turbot goes on the full-attention child. The SWA child resolves its own type through its own chain and gets turbo5p (turbo5p512 for 512-value rows).
  - `LLAMA_TURBOT_SWA_TYPE=<type>` sets the SWA child's type, for A/B runs.
  - `LLAMA_TURBOT_ISWA=0` refuses turbot on SWA models, which then fall back to turbo5p / turbo5p as today.
- **Streams.** Without `--kv-unified`, `-np N` (N > 1) gives the cache N streams. Each stream has its own tier state and POOL_s = POOL / n_stream young rows, rounded down to whole granules, with the same CAP. The automatic plan makes POOL a multiple of 64 · n_stream, so nothing is lost there; a plan file's POOL can lose up to one granule per stream to the rounding. The base cache is one `[1024, kv_size, n_stream]` container: the writer sees it flattened to `[1024, kv_size · n_stream]`, and the graph runs one turbot FA per stream (`ne[3] == 1`).
  - `LLAMA_TURBOT_MULTI_STREAM=0` refuses n_stream > 1, which then falls back to turbo5p as today.

### 14.10 Switches

| Variable | Scope | Effect |
|---|---|---|
| `LLAMA_TURBOT_ANY=0` | master switch, host | Restores today's behaviour exactly: only the 4 × 256 geometry, no automatic plan, no sidecar, no iSWA split, no multi-stream. Every other switch below reads false when this is 0. |
| `GGML_TURBOT_ANY=0` | ggml CUDA routing and writer supports | Only flags 0 (4 × 256) and NG 8 are accepted: today's kernel routing. The resolver reads it too, so other geometries step down before a graph is built. |
| `LLAMA_TURBOT_AUTO_PLAN=0\|1\|all` | automatic plan | `0`: fall back to turbo5p when no plan matches (today). Unset or `1`: VALIDATED geometries only. `all`: every supported geometry. |
| `LLAMA_TURBOT_AUTO_BUDGET=turbo5p` | automatic plan, opt-in | Budget at turbo5p's rate. Unset: the budget is the fallback type's bytes, never more VRAM than today. |
| `LLAMA_TURBOT_AUTO_PLAN_DUMP=<file>` | diagnostic | Writes the generated plan text. |
| `LLAMA_TURBOT_SIDECAR=0` | sidecar lookup | The sidecar is ignored. |
| `LLAMA_TURBOT_ISWA=0` | iSWA split | SWA models refuse turbot and fall back to turbo5p / turbo5p (today). |
| `LLAMA_TURBOT_SWA_TYPE=<type>` | A/B arm | Unset: the SWA child resolves turbo5p through its own chain. |
| `LLAMA_TURBOT_MULTI_STREAM=0` | non-unified `-np N` | n_stream > 1 is refused and falls back to turbo5p (today). |
| `LLAMA_KV_RESOLVE=0`, `LLAMA_TURBOT=0` | existing | Unchanged meaning. With the resolver off there is no plan scope, so the cache constructor uses the old precedence (9.1) plus only the explicit `auto` keyword. |

### 14.11 VALIDATED list and the models on disk

- **VALIDATED** is the set of geometries the automatic plan may use by default. It is expected to be {256 × 4, 256 × 2} after the gates.
  - 128 × 8 and 128 × 4 have kernels but no model on disk to measure them, so they are opt-in with `LLAMA_TURBOT_AUTO_PLAN=all`.
  - The NR = 1 shapes (2 × 128, 1 × 256) cannot fit the turbo4 budget and are opt-in with `LLAMA_TURBOT_AUTO_BUDGET=turbo5p`. NR = 1 becomes a default only if turbot beats turbo4 on all three NR = 1 models in G5 and loses no decode speed in G6; it costs about 11% more KV VRAM than turbo4 at 262K cells (+316 / +98 / +45 MiB, 14.7).
- Multi-stream and iSWA stay on only if G7 passes.
- A failed gate flips that switch's default in code (a one-line change); the env switches remain.

| Model file | Arch | Attention layers | KV shape | GQA | Fallback | turbot path |
|---|---|---|---|---|---|---|
| Qwen3.8-27B (reference) | qwen35 hybrid | 3, 7, …, 63 (16) | 4 × 256 | 6 | turbo5p | built-in plan, unchanged |
| Ornith-1.5-9B-Q8_0 | qwen35 hybrid, MTP layer 32 excluded | 3, 7, …, 31 (8) | 4 × 256 | 4 | turbo5p | automatic plan. Same kernel shape as Qwen; the ncols2 = 4 instances run for the first time. |
| Spark-X2.5-4B-Q8_0 | spark2_5 iSWA, window 512 | full 3, 7, …, 35 (9), 27 SWA | 4 × 256 | 4 | turbo5p | iSWA split: turbot on the 9 full layers, turbo5p on the SWA layers; automatic plan |
| Ornith-1.5-35B-Q4_K_M | qwen35moe hybrid, MTP layer 40 excluded | 3, …, 39 (10) | 2 × 256 (NR 2) | 8 | turbo5p512 | new head count, 512-value rows; automatic plan |
| MiniCPM5-2B-Q8_0 | llama | 0..41 (42) | 2 × 128 (NR 1) | 8 | turbo4 (68 B per 128) | D = 128 kernels and S < 8 types; opt-in only (budget rule) |
| Muse-Glimmer-30B-KQuant-17GB-Q4_K_M | muse-glimmer iSWA, window 2048 | 13 full of 52 | 2 × 128 (NR 1) | 16 | turbo4 | iSWA and NR 1; opt-in only |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0 | nemotron_h_moe hybrid | 5, 12, 19, 26, 33, 42 (6) | 2 × 128 (NR 1) | 16 | turbo4 | NR 1; opt-in only |

### 14.12 Not covered

These stay on the fallback chain (9.2), with the reason in the WARN line:
- head dims other than 128 and 256 (64, 80, 96, 192, 512, …), 128 × 1 (half a run), and rows above 1024 values (256 × 8, 128 × 16, …): turbot has no layout for them;
- K and V head sizes that differ, MLA, shared cells, and archs without the turbo query rotation: refused as before;
- all-SWA models: there is no full-attention child to hold turbot;
- KV off CUDA, or a GPU without Turing+ MMA;
- NR = 1 shapes at the default budget: they cannot fit turbo4's bytes (14.7), so they need the opt-in;
- the DFlash2 and MTP draft contexts.

K shift (`seq_add` / `seq_div`) stays unsupported on every geometry (9.3).

### 14.13 Tests

- `tests/test-turbot-geom.cpp` (CPU only; registered like test-turbot):
  - (a) geometry flags of the six shapes, the accessors, the run mapping, and −1 for (64, 8), (128, 16), (256, 8), (512, 1) and others;
  - (b) side and layer init for nr 1, 2 and 4: sums, offsets, row and young bytes, zero runs beyond nr, illegal input;
  - (c) encode, decode and fill per nr at old widths 2..6 and young 7 and 8, against `ggml_turbot_quantize_group`. An NR-run row equals the first NR runs of a 4-run row. nr 4 (flags 0) is byte-identical to a copy of the pre-change loops;
  - (d) hash: flags 0 equals a copy of the old fold (default plan `0x56c3503c949a7749`), and flags ≠ 0 differs;
  - (e) the type family: ids, `ggml_type_size` = 32·S + 16, blck 1024, names, `is_type` false at 48 and 72;
  - (f) op params: make / get / layer round trip carries flags; reserved bits, invalid geometries and widths beyond nr are rejected;
  - (g) CPU FA reference at D 128 NR 4 (GQA 4 and 7), D 256 NR 2 (GQA 8) and the other geometries, equal to dense attention over the decoded K/V; CPU supports_op follows the op-params geometry;
  - (h) the CPU writer op at every geometry, fill entries included, byte-identical to the header coder.
  - (g) and (h) call the CPU backend directly, so they are compiled only without `GGML_BACKEND_DL` and print SKIPPED otherwise.
- The GPU gates G0-G8 are in TESTING.md section 9.

---

## Review log (2026-09-15)

Each review issue was checked against the code or reproduced on CPU (`scratchpad/turbot_review_mc.py`, `scratchpad/turbot_review_indep.py`, and the a_lloyd convergence probe).

| # | Severity, lens | Issue | Verdict | Where |
|---|---|---|---|---|
| 1 | major, kernel | Reader per-element float gathers; no read-cost model for "long context gains" | Accepted. Read volume confirmed (0.861× old, 1.415× young). Target changed to parity; gate B0 added. Register gathers measured: old int8 is allowed for b ≤ 5 (≤ ×1.014) and forbidden for b = 6 (×1.124). Young (c) with one scale per (b, y) measured ×1.02-1.20, so only the per-base-cell-scale form is allowed, and only if B0 shows a win. | 4.3, 7.4, 7.8, 11.2 |
| 2 | major, kernel | LUT copy after the Q load overwrites tile_Q | Accepted, verified: `tile_K == tile_Q` for every D=256 config (`Q_in_reg`), and Q reaches `Q_B` between the `__syncthreads` at fattn-mma-f16.cuh:1825 and 1836. Exact offset and copy point stated; ncols128 FA cases gate the perf runs. | 7.4, 7.5, 11.2 |
| 3 | minor, kernel | Writer refinement search and gain sums replay `__constant__` reads | Accepted: uniform young count is the default, gain tables go to shared memory, binary search only if measured faster. `ggml_turbot_young_index` added to the header. | 8.2, 8.3, 4.2 |
| 4 | minor, kernel | POOL 0 has no pool tensor but ops assert a pool | Accepted: pool rows = max(N_R, 64), never NULL, gtab all −1. | 2.1, 3.3, 5.2, 9.1, 9.4 |
| 5 | minor, kernel | Including fattn-turbot.cuh into every MMA TU risks the codegen identity | Accepted: separate turbot instance TUs, a dispatch helper in fattn.cu, and fattn-mma-f16.cuh keeps only the array-size fix. `FATTN_MMA_TURBOT` dropped. | 7.1, 7.6, 12.3 |
| 6 | minor, kernel | Head addressing computed at two drifting sites | Accepted: `turbot_head_state_of` required at both sites; `is_fixup` FA cases with non-uniform widths. | 7.2, 11.2 |
| 7 | major, host | Eviction key ties after every commit | Accepted, verified from the spec text: `last_wanted` removed; victims ordered by (empty, margin, last_touch, slot); quota slack. | 9.6 |
| 8 | major, host | Empty owned slots returned only at commit; begin and restore evict live granules first | Accepted: `alloc_slot` reclaims empty owned slots before any live eviction. | 9.6, 9.10 |
| 9 | major, host | Counter spaces merged with max() on restore and seq_cp | Accepted, verified (`state_read_meta` seq_rm's dest first, llama-kv-cache.cpp:2763; server copy_state_to does seq_rm then seq_cp): restore assigns counters, seq_cp into an empty dst adopts the source counter. | 9.7, 9.10 |
| 10 | major, host | Rejected draft rows leave stamp gaps | Accepted, verified (server-context.cpp:5289): tail `seq_rm` rolls the counter back. | 9.7, 11.1 |
| 11 | major, host | Stale ref_valid bits after seq_rm reach a later restore and state_write | Accepted: emptied cells clear their bits, begin masks with live, restore clears bits of cells without a pool row, `cell_young` excludes pending; DEBUG=2 check extended. | 9.6-9.8, 9.10 |
| 12 | major, host | Hybrid can_reuse re-implements the checks and would skip the fill | Accepted, verified (llama-graph.cpp:1132 does not delegate). | 10.1, 11.2 |
| 13 | minor, host | row_ctr sized n_seq_max | Accepted: LLAMA_MAX_SEQ (llama-context.cpp:1753, llama-kv-cache.cpp:172). | 9.5 |
| 14 | minor, host | commit walks every sequence per cell | Accepted: live-sequence mask, early break; margins only on the eviction path. | 9.6 |
| 15 | minor, host | Plan path set after fit; drafter swap not seen by fit | Partly accepted: plan path set as the first statement of common_init_result; drafter swap moved into common_base_params_to_speculative. Rejected the n_ctx rounding: llama_context already pads n_ctx to 256 (llama-context.cpp:295), fit probes included. | 9.1 |
| 16 | minor, host | restore_cells evicts before a short blob fails | Partly accepted: everything but the raw pool bytes is read and validated before restore_cells, and `abort_restore` frees the slots a failed restore allocated. Reading the bytes first is rejected: up to 761 MB of extra host RAM with copying io reads. Evictions already made are not undone. | 9.10 |
| 17 | major, math | Quality was measured without attn_rot, turbot would ship with it | Accepted, verified (llama-kv-cache.cpp:562-579, llama-graph.cpp:2970-2976; kvfq ran on F16): attn_rot forced off for turbot, `LLAMA_TURBOT_ATTN_ROT=1` opt-in, Phase 6 arm. | 1 (8), 4.2, 9.4, 10.3 |
| 18 | major, math | Old nmse ±3% gate fails on a correct coder | Accepted, reproduced (1.012 / 0.978 / 0.965 / 0.945 / 0.959): band [0.93, 1.03] plus the bit-exact fq_group gate. | 11.1 |
| 19 | minor, math | "Converges to 1e-11" is false for b2 y8 | Accepted, reproduced (20,000 iterations, 3.81e-10): the generator records and prints the stats and asserts < 1e-9. The tables are unchanged bit for bit (only header comment lines differ). | 1.1, gen_turbot_tables.py, ggml-turbot-tables.h |
| 20 | minor, math | Fill fidelity figures ignore the gain rescale | Accepted, reproduced: 4.4 recomputed; bounds 0.30 / 1.30; section 13 corrected. | 4.4, 11.1, 13 |
| 21 | minor, math | Pool has zero slack against the quota | Accepted: N_eff = N_R − 128·n_active; CAP stays 16,384, so one sequence is unchanged, and VRAM is unchanged. | 1 (5), 9.6 |

---

## Appendix A: default plan (`docs/turbot/plans/turbot-default.plan`)

```
# turbot default plan: fq_3t0 old widths (E:\kv-s3\plans\fq_3t0_w256_m16384_7b.txt), young width 7 everywhere
# base 4520.00 MiB + young pool 726.00 MiB = 5246.00 MiB at 262144 cells (turbo5p 5248.00 MiB)
POOL 65536
CAP 16384
L 3 K 2 2 2 4 V 2 2 2 4
L 7 K 2 5 3 2 V 2 5 2 2
L 11 K 4 5 4 2 V 3 4 5 2
L 15 K 4 4 3 4 V 5 3 3 4
L 19 K 4 4 3 4 V 3 4 2 5
L 23 K 5 4 4 6 V 5 4 4 5
L 27 K 5 5 6 5 V 5 5 6 5
L 31 K 6 5 5 5 V 5 4 5 4
L 35 K 4 5 5 5 V 4 4 4 5
L 39 K 5 5 4 6 V 4 4 4 5
L 43 K 4 4 6 5 V 4 4 5 4
L 47 K 4 5 5 5 V 4 4 5 5
L 51 K 5 5 5 6 V 5 4 4 5
L 55 K 5 5 5 5 V 5 5 5 5
L 59 K 4 5 5 5 V 5 6 5 5
L 63 K 4 4 4 4 V 5 4 5 5
```

Per-layer bytes (base K, base V, young K part, young V part):

| il | 3 | 7 | 11 | 15 | 19 | 23 | 27 | 31 | 35 | 39 | 43 | 47 | 51 | 55 | 59 | 63 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base K | 336 | 400 | 496 | 496 | 496 | 624 | 688 | 688 | 624 | 656 | 624 | 624 | 688 | 656 | 624 | 528 |
| base V | 336 | 368 | 464 | 496 | 464 | 592 | 688 | 592 | 560 | 560 | 560 | 592 | 592 | 656 | 688 | 624 |
| young K | 592 | 528 | 432 | 432 | 432 | 304 | 240 | 240 | 304 | 272 | 304 | 304 | 240 | 272 | 304 | 400 |
| young V | 592 | 560 | 464 | 432 | 464 | 336 | 240 | 336 | 368 | 368 | 368 | 336 | 336 | 272 | 240 | 304 |
