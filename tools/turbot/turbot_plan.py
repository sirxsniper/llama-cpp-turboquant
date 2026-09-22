# [TAG_TURBOT] Plan file helpers for the turbot tiered KV cache (docs/turbot/SPEC.md 3.6, 9.1). CPU only, stdlib only
# (the --gguf options import gguf-py from this repository).
#
# Library used by plan_vram.py, turbot_guard.py and the commands below. The grammar and the validation mirror
# llama_turbot_plan_parse_shape (src/llama-kv-tier.cpp); the byte layout and the plan hash mirror ggml/include/ggml-turbot.h.
#
#   python tools/turbot/turbot_plan.py convert E:/kv-s3/plans/fq_3t0_w256_m16384_7b.txt -o my.plan
#   python tools/turbot/turbot_plan.py convert E:/kvdump/alloc3_plans.json --entry 0 -o my.plan
#   python tools/turbot/turbot_plan.py convert E:/kvdump/alloc3_plans.json --entry 1 --young 7 -o my.plan
#
# [TAG_TURBOT_ANY_PLAN] the automatic plan of llama_turbot_plan_auto_text, the same text and hash byte for byte:
#   python tools/turbot/turbot_plan.py auto --layers 3,7,11,15,19,23,27,31 --head-dim 256 --n-head-kv 4 -c 262144
#   python tools/turbot/turbot_plan.py auto --gguf D:/Projects/LocalAI/models/Ornith-1.5-35B-Q4_K_M.gguf -c 262144 -np 4
#   python tools/turbot/turbot_plan.py auto --gguf <model.gguf> -c 262144 --budget turbo5p -o <model.gguf>.turbot.plan
#     -c is n_ctx, -np n_seq_max, --unified one KV stream for all sequences (else -np streams of n_ctx/np cells, as
#     llama_context), --budget turbo5p = LLAMA_TURBOT_AUTO_BUDGET=turbo5p. The text goes to stdout (or -o), the hash to stderr.
# [TAG_TURBOT_ANY_SIDECAR] the '# model:' line of a verified plan (llama_turbot_fingerprint_text):
#   python tools/turbot/turbot_plan.py fingerprint --gguf <model.gguf>
#     The attention layers are read from the GGUF metadata (full_attention_interval, sliding_window_pattern,
#     nextn_predict_layers). libllama logs the authoritative line ("turbot: model fingerprint: ...") for every file,
#     sidecar and automatic plan; turbot_guard.py takes it from there.
#
# Conversion rules:
#   kvfq text plans ("L il K b b b b V b b b b", "W", "W2", "M"): old widths from L. Young width = --young if given, else M
#   when present, else 7. CAP = --cap if given, else W2 when present, else 16384. W is dropped (turbot has no exact tier).
#   alloc3_plans.json entries (bK, bV per attention layer, W1, W2, b2): the same with b2 in the place of M. The entries
#   list the 16 attention layers of Qwen3.8-27B in order, il = 3, 7, ..., 63.
#   POOL = --pool if given, else 65536. A young width that is not above an old width is an error unless --young fixes it.
#
# Geometry ([TAG_TURBOT_ANY_GEOM], ggml-turbot.h): a layer-side row is NR runs of 256 values, one width per run. flags by
# (head dim, KV heads): (256,4)=0 (256,2)=1 (256,1)=2 (128,8)=4 (128,4)=5 (128,2)=6, NR = 4 >> (flags & 3). An L or Y line
# of a layer with NR runs has 4+2*NR tokens. A layer with flags != 0 also folds 'GEO1' + flags into the plan hash.
import argparse
import io
import json
import os
import struct
import sys

B_MIN, B_MAX, Y_MAX, Y_DEFAULT = 2, 6, 8, 7
POOL_DEFAULT, CAP_DEFAULT = 65536, 16384
N_HEAD = 4
MAX_RUNS = 4
RUN_ELEMS = 256
GRANULE = 64
QUOTA_SLACK_GRANULES = 2
AUTO_POOL_MIN_SEQ = 1024               # young pool floor per sequence of an automatic plan, plus the quota slack
QWEN38_ATTN_LAYERS = list(range(3, 64, 4))
KV_DEFAULT = 262144
TURBO5P_ROW = 656                      # bytes per 1024 values (turbo5p_0)
FNV_OFFSET, FNV_PRIME = 0xcbf29ce484222325, 0x100000001b3
MIB = 1024.0 * 1024.0
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))

# (head dim, KV heads) -> flags
GEOM_FLAGS = {(256, 4): 0, (256, 2): 1, (256, 1): 2, (128, 8): 4, (128, 4): 5, (128, 2): 6}
# validated for an automatic plan without LLAMA_TURBOT_AUTO_PLAN=all (llama_turbot_auto_geom_validated)
AUTO_VALIDATED = {(256, 4), (256, 2)}
# fallback KV types: (bytes per block, values per block)
BUDGET_TYPES = {"turbo5p": (656, 1024), "turbo5p512": (336, 512), "turbo4": (68, 128)}   # ggml type_size / blck_size


class PlanError(Exception):
    pass


def geom_flags(head_dim, n_head_kv):
    return GEOM_FLAGS.get((int(head_dim), int(n_head_kv)), -1)


def geom_nr(flags):
    return MAX_RUNS >> (flags & 3)


def geom_row_elems(flags):
    return geom_nr(flags) * RUN_ELEMS


def budget_type(row_elems):
    """llama_turbot_budget_type: the fallback KV type of a row of row_elems values."""
    if row_elems % 1024 == 0:
        return "turbo5p"
    if row_elems % 512 == 0:
        return "turbo5p512"
    return "turbo4"


def type_row_bytes(t, row_elems):
    size, blck = BUDGET_TYPES[t]
    if row_elems % blck:
        raise PlanError("%s cannot hold a row of %d values" % (t, row_elems))
    return size * row_elems // blck


def cache_fallback(rows):
    """The one fallback type of a cache whose rows are rows (the resolver's turbot -> turbo5p -> turbo4 step)."""
    if all(r % 1024 == 0 for r in rows):
        return "turbo5p"
    if all(r % 512 == 0 for r in rows):
        return "turbo5p512"
    return "turbo4"


def make_shape(layers, head_dim=256, n_head_kv=4, kv_size=KV_DEFAULT, n_stream=1, n_seq_max=1):
    """llama_turbot_cache_shape. layers: list of il (all head_dim x n_head_kv), or dict il -> (head dim, KV heads)."""
    if isinstance(layers, dict):
        geo = {int(il): (int(d), int(h)) for il, (d, h) in layers.items()}
    else:
        geo = {int(il): (int(head_dim), int(n_head_kv)) for il in layers}
    return dict(layers=geo, kv_size=int(kv_size), n_stream=int(n_stream), n_seq_max=int(n_seq_max))


def ctx_shape(layers, n_ctx, n_seq_max=1, unified=False):
    """The attention cache of llama_context for -c n_ctx -np n_seq_max [--kv-unified]: kv_size and n_stream."""
    pad = lambda x: (x + 255) // 256 * 256
    n_ctx = pad(n_ctx)
    n_seq_max = max(1, n_seq_max)
    if unified or n_seq_max == 1:
        return dict(layers=layers, kv_size=n_ctx, n_stream=1, n_seq_max=n_seq_max)
    return dict(layers=layers, kv_size=pad(n_ctx // n_seq_max), n_stream=n_seq_max, n_seq_max=n_seq_max)


def shape_flags(shape):
    """il -> flags, ascending il; PlanError when turbot has no layout for a layer."""
    out = {}
    for il in sorted(shape["layers"]):
        d, h = shape["layers"][il]
        f = geom_flags(d, h)
        if f < 0:
            raise PlanError("attention layer %d: turbot has no layout for %d KV heads x %d" % (il, h, d))
        out[il] = f
    return out


def side_bytes(b, y):
    """(base row bytes, young part bytes) of one layer-side (SPEC 3.2, 3.3), any number of runs."""
    s = sum(b)
    r = sum(yy - bb for bb, yy in zip(b, y))
    return 32 * s + 16, 32 * r + 16


def check_widths(b, y, where):
    for bb, yy in zip(b, y):
        if not (B_MIN <= bb <= B_MAX):
            raise PlanError("%s: old width %d outside [%d, %d]" % (where, bb, B_MIN, B_MAX))
        if not (bb < yy <= Y_MAX):
            raise PlanError("%s: young width %d outside [%d, %d]" % (where, yy, bb + 1, Y_MAX))


def _int(s):
    try:
        return int(s)
    except ValueError:
        return None


def parse_text(text, attn_layers=None, kv_size=None, shape=None):
    """-> dict(layers={il: dict(bk, bv, yk, yv, flags)}, pool, cap, ignored) or raises PlanError naming the line.
    shape (make_shape / ctx_shape) gives every layer its runs and the cells (kv_size*n_stream) POOL is clamped to;
    without it every layer has 4 runs (4 KV heads x 256) and attn_layers / kv_size work as before."""
    flags = None
    kv_cells = kv_size
    if shape is not None:
        flags = shape_flags(shape)
        attn_layers = sorted(flags)
        kv_cells = shape["kv_size"] * max(1, shape.get("n_stream", 1))
    elif attn_layers is not None:
        flags = {int(il): 0 for il in attn_layers}

    def nr_of(il):
        return geom_nr(flags[il]) if flags is not None and il in flags else N_HEAD

    L, Y = {}, {}
    pool = cap = None
    ignored = []
    for ln, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        tok = line.split()
        tag = tok[0]
        where = "line %d (%s)" % (ln, raw.strip())
        if tag in ("L", "Y"):
            il_peek = _int(tok[1]) if len(tok) >= 2 else None
            nr = nr_of(il_peek)
            if len(tok) != 4 + 2 * nr or tok[2] != "K" or tok[3 + nr] != "V":
                w = " ".join("w" for _ in range(nr))
                raise PlanError("%s: malformed, expected '%s il K %s V %s'" % (where, tag, w, w))
            try:
                il = int(tok[1])
                k = [int(t) for t in tok[3:3 + nr]]
                v = [int(t) for t in tok[4 + nr:4 + 2 * nr]]
            except ValueError:
                raise PlanError("%s: malformed number" % where)
            dst = L if tag == "L" else Y
            if il in dst:
                raise PlanError("%s: duplicate %s line for layer %d" % (where, tag, il))
            dst[il] = (k, v, where)
        elif tag in ("POOL", "CAP"):
            if len(tok) != 2:
                raise PlanError("%s: malformed" % where)
            try:
                val = int(tok[1])
            except ValueError:
                raise PlanError("%s: malformed number" % where)
            if tag == "POOL":
                if pool is not None:
                    raise PlanError("%s: duplicate POOL" % where)
                if val < 0 or val % GRANULE != 0:
                    raise PlanError("%s: POOL must be a non-negative multiple of 64" % where)
                pool = val    # a pool above the cache is clamped after the loop, like llama-kv-tier.cpp
            else:
                if cap is not None:
                    raise PlanError("%s: duplicate CAP" % where)
                if val < 0:
                    raise PlanError("%s: CAP must not be negative" % where)
                cap = val
        elif tag in ("W", "W2", "M"):
            ignored.append(tag)
        else:
            raise PlanError("%s: unknown tag '%s'" % (where, tag))
    layers = {}
    for il, (bk, bv, where) in L.items():
        nr = len(bk)
        yk, yv = ([Y_DEFAULT] * nr, [Y_DEFAULT] * nr)
        if il in Y:
            yk, yv = Y[il][0], Y[il][1]
            if len(yk) != nr:
                raise PlanError("%s: Y line for layer %d has %d widths, the L line %d" % (Y[il][2], il, len(yk), nr))
        check_widths(bk, yk, where + " K")
        check_widths(bv, yv, where + " V")
        layers[il] = dict(bk=bk, bv=bv, yk=yk, yv=yv, flags=flags.get(il, 0) if flags is not None else 0)
    for il, (_, _, where) in Y.items():
        if il not in L:
            raise PlanError("%s: Y line for layer %d without an L line" % (where, il))
    if attn_layers is not None:
        for il in attn_layers:
            if il not in layers:
                raise PlanError("missing L line for attention layer %d" % il)
        for il in layers:
            if il not in attn_layers:
                raise PlanError("L line for layer %d, which the cache does not hold" % il)
    if not layers:
        raise PlanError("no L lines")
    if pool is None:
        pool = POOL_DEFAULT
    if kv_cells is not None and pool > kv_cells:
        pool = (kv_cells // GRANULE) * GRANULE    # clamped to the cache in whole granules, as the C++ parser does
    return dict(layers=layers, pool=pool, cap=CAP_DEFAULT if cap is None else cap, ignored=sorted(set(ignored)))


def parse_file(path, attn_layers=None, kv_size=None, shape=None):
    with io.open(path, encoding="utf-8") as f:
        return parse_text(f.read(), attn_layers, kv_size, shape)


def _pad4(w):
    return list(w) + [0] * (MAX_RUNS - len(w))


def plan_hash(plan):
    """ggml_turbot_plan_hash_layer over ascending il, then ggml_turbot_plan_hash_finish (FNV-1a 64). Widths of runs
    r >= nr are 0; a layer with flags != 0 also folds 'GEO1' + flags."""
    def fnv(h, data):
        for c in data:
            h ^= c
            h = (h * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
        return h
    h = FNV_OFFSET
    for il in sorted(plan["layers"]):
        l = plan["layers"][il]
        h = fnv(h, b"LAY1")
        h = fnv(h, struct.pack("<i", il))
        for key in ("bk", "bv", "yk", "yv"):
            h = fnv(h, bytes(_pad4(l[key])))
        if l.get("flags", 0):
            h = fnv(h, b"GEO1" + bytes([l["flags"]]))
    h = fnv(h, struct.pack("<I", plan["pool"]))
    h = fnv(h, struct.pack("<I", plan["cap"]))
    return h


def vram(plan, kv_size=KV_DEFAULT, n_stream=1):
    """-> dict of per-cell and total byte counts (SPEC 3.6) and the fallback type's bytes of the same cache."""
    base_cell = pool_cell = s_sum = n_runs = 0
    rows = []
    row_elems = []
    for il in sorted(plan["layers"]):
        l = plan["layers"][il]
        bk, yk = side_bytes(l["bk"], l["yk"])
        bv, yv = side_bytes(l["bv"], l["yv"])
        base_cell += bk + bv
        pool_cell += yk + yv
        s_sum += sum(l["bk"]) + sum(l["bv"])
        n_runs += len(l["bk"]) + len(l["bv"])
        rows.append((il, bk, bv, yk, yv))
        row_elems.append(geom_row_elems(l.get("flags", 0)))
    n_layers = len(plan["layers"])
    kv_cells = kv_size * n_stream
    base = kv_cells * base_cell
    pool = plan["pool"] * pool_cell
    fb = cache_fallback(row_elems)
    fallback = kv_cells * sum(2 * type_row_bytes(fb, r) for r in row_elems)
    turbo5p = kv_cells * sum(2 * (r * TURBO5P_ROW // 1024) for r in row_elems)
    return dict(rows=rows, base_cell=base_cell, pool_cell=pool_cell, base=base, pool=pool, total=base + pool,
                turbo5p=turbo5p, fallback=fallback, fallback_type=fb, s_sum=s_sum, n_heads=n_runs, n_runs=n_runs,
                n_layers=n_layers, kv_size=kv_size, n_stream=n_stream)


def render(plan, header_lines=()):
    out = ["# " + h for h in header_lines]
    out.append("POOL %d" % plan["pool"])
    out.append("CAP %d" % plan["cap"])
    for il in sorted(plan["layers"]):
        l = plan["layers"][il]
        out.append("L %d K %s V %s" % (il, " ".join(map(str, l["bk"])), " ".join(map(str, l["bv"]))))
    for il in sorted(plan["layers"]):
        l = plan["layers"][il]
        if any(y != Y_DEFAULT for y in l["yk"] + l["yv"]):
            out.append("Y %d K %s V %s" % (il, " ".join(map(str, l["yk"])), " ".join(map(str, l["yv"]))))
    return "\n".join(out) + "\n"


def summary_lines(plan, kv_size=KV_DEFAULT, n_stream=1):
    v = vram(plan, kv_size, n_stream)
    return [
        "base %.2f MiB + young pool %.2f MiB = %.2f MiB at %d cells (%s %.2f MiB, margin %.2f MiB)"
        % (v["base"] / MIB, v["pool"] / MIB, v["total"] / MIB, kv_size * n_stream, v["fallback_type"],
           v["fallback"] / MIB, (v["fallback"] - v["total"]) / MIB),
        "%d layers, old bits mean %.3f (sum %d over %d runs), base %d B/cell, pool %d B/pool cell, POOL %d (%d granules), CAP %d, hash 0x%016x"
        % (v["n_layers"], v["s_sum"] / float(v["n_runs"]), v["s_sum"], v["n_runs"], v["base_cell"], v["pool_cell"],
           plan["pool"], plan["pool"] // 64, plan["cap"], plan_hash(plan)),
    ]


# ---------------------------------------------------------------------------------------------------------------
# [TAG_TURBOT_ANY_PLAN] automatic plan: llama_turbot_auto_impl (src/llama-kv-tier.cpp), line for line
# ---------------------------------------------------------------------------------------------------------------

def auto_plan(shape, budget_turbo5p=False):
    """The text of llama_turbot_plan_auto_text for shape; PlanError with the C++ reason when it refuses."""
    flags = shape_flags(shape)
    if not flags:
        raise PlanError("the cache holds no attention layers")
    kv_size = shape["kv_size"]
    if kv_size <= 0 or kv_size % GRANULE:
        raise PlanError("kv_size %d is not a positive multiple of %d" % (kv_size, GRANULE))

    n_stream = max(1, shape.get("n_stream", 1))
    n_seq = max(1, shape.get("n_seq_max", 1))
    kv_cells = kv_size * n_stream
    step = GRANULE * n_stream
    cap = CAP_DEFAULT
    slack = GRANULE * QUOTA_SLACK_GRANULES
    y = Y_DEFAULT

    rows = [geom_row_elems(f) for f in flags.values()]
    fallback = cache_fallback(rows)
    name = "turbo5p" if budget_turbo5p else fallback
    budget = kv_cells * sum(2 * (r * TURBO5P_ROW // 1024 if budget_turbo5p else type_row_bytes(fallback, r)) for r in rows)

    nr_max = max(geom_nr(f) for f in flags.values())

    def widths(m, nr, side):
        n5 = min((m + 1) // 2 if side == 0 else m // 2, nr)
        return [5 if r < n5 else 4 for r in range(nr)]

    def per_cell(m):
        base = young = 0
        for f in flags.values():
            nr = geom_nr(f)
            for side in (0, 1):
                b = widths(m, nr, side)
                base += 32 * sum(b) + 16
                young += 32 * sum(y - x for x in b) + 16
        return base, young

    pool = min((n_seq * (cap + slack) + GRANULE - 1) // GRANULE * GRANULE, kv_cells // GRANULE * GRANULE)
    pool = pool // step * step

    m_best = None
    for m in range(2 * nr_max, -1, -1):
        base, young = per_cell(m)
        if kv_cells * base + pool * young <= budget:
            m_best = m
            break

    if m_best is None:
        base, young = per_cell(0)
        if kv_cells * base > budget:
            raise PlanError("4-bit old rows alone need %.2f MiB, above the %.2f MiB of %s"
                            % (kv_cells * base / MIB, budget / MIB, name))
        pool = (budget - kv_cells * base) // young
        pool = pool // step * step
        m_best = 0

    pool_min = n_seq * (AUTO_POOL_MIN_SEQ + slack)
    if pool < pool_min:
        raise PlanError("the young pool would be %d cells in the %.2f MiB of %s, below %d (%d sequences x %d)"
                        % (pool, budget / MIB, name, pool_min, n_seq, AUTO_POOL_MIN_SEQ + slack))

    base, young = per_cell(m_best)
    total = kv_cells * base + pool * young

    sum_w = n_runs = 0
    for f in flags.values():
        nr = geom_nr(f)
        for side in (0, 1):
            sum_w += sum(widths(m_best, nr, side))
            n_runs += nr

    groups = []   # [(KV heads, head dim), count] in order of first appearance by il
    for il in sorted(flags):
        d, h = shape["layers"][il]
        for g in groups:
            if g[0] == (h, d):
                g[1] += 1
                break
        else:
            groups.append([(h, d), 1])
    geo = ", ".join("%d x %dx%d" % (c, hd[0], hd[1]) for hd, c in groups)

    out = ["# turbot auto plan v1",
           "# shape: %d attention layers: %s (KV heads x head dim); kv_size %d, n_stream %d, n_seq_max %d"
           % (len(flags), geo, kv_size, n_stream, n_seq)]
    if budget_turbo5p:
        out.append("# budget: turbo5p rate, 656 B per 1024 values (LLAMA_TURBOT_AUTO_BUDGET=turbo5p); the fallback type is %s"
                   % fallback)
    else:
        out.append("# budget: the bytes of the fallback type %s" % name)
    out.append("# size: %.2f MiB (%s %.2f MiB); old widths 4/5 mean %.3f, young %d, uncalibrated"
               % (total / MIB, name, budget / MIB, sum_w / float(n_runs), y))
    for il in sorted(flags):
        nr = geom_nr(flags[il])
        out.append("L %d K %s V %s" % (il, " ".join(map(str, widths(m_best, nr, 0))), " ".join(map(str, widths(m_best, nr, 1)))))
    out.append("POOL %d" % pool)
    out.append("CAP %d" % cap)
    return "\n".join(out) + "\n"


def auto_allowed(shape, auto_all=False, budget_turbo5p=False):
    """llama_turbot_auto_allowed: may the chooser's step 4 give shape an automatic plan? -> (bool, why)."""
    for il, f in shape_flags(shape).items():
        d, h = shape["layers"][il]
        if (d, h) in AUTO_VALIDATED or auto_all or (budget_turbo5p and geom_nr(f) == 1):
            continue
        return False, "layer %d: %d KV heads x %d is not validated for an automatic plan" % (il, h, d)
    return True, ""


# ---------------------------------------------------------------------------------------------------------------
# [TAG_TURBOT_ANY_SIDECAR] model fingerprint and GGUF shape
# ---------------------------------------------------------------------------------------------------------------

def fingerprint_text(arch, basename, size, shape):
    """llama_turbot_fingerprint_text: 'arch=.. basename=.. size=.. layers=il,.. geom=DxH' (one geom when uniform)."""
    ils = sorted(shape["layers"])
    geos = ["%dx%d" % shape["layers"][il] for il in ils]
    geo = "" if not ils else (geos[0] if all(g == geos[0] for g in geos) else ",".join(geos))
    return "arch=%s basename=%s size=%s layers=%s geom=%s" % (arch, basename, size, ",".join(map(str, ils)), geo)


def _gguf_fields(path):
    sys.path.insert(0, os.path.join(REPO, "gguf-py"))
    from gguf import GGUFReader  # noqa: E402

    r = GGUFReader(path)

    def val(f):
        t0 = f.types[0].name
        if t0 == "STRING":
            return bytes(f.parts[f.data[0]]).decode("utf-8", "replace")
        if t0 == "ARRAY":
            if f.types[-1].name == "STRING":
                return [bytes(f.parts[i]).decode("utf-8", "replace") for i in f.data]
            return [f.parts[i].tolist()[0] for i in f.data]
        return f.parts[f.data[0]].tolist()[0]

    out = {}
    for k, f in r.fields.items():
        if k.startswith("tokenizer."):
            continue
        try:
            out[k] = val(f)
        except Exception:
            pass
    return out


def gguf_info(path):
    """-> dict(arch, basename, size, layers={il: (head dim, KV heads)}) of the attention layers a turbot cache of the main
    context would hold: every block below nextn_predict_layers that is a full-attention layer (full_attention_interval,
    not a sliding_window_pattern layer) with KV heads. The same selection as llama for the models of this fork; the
    'turbot: model fingerprint:' log line is the authority."""
    m = _gguf_fields(path)
    arch = m.get("general.architecture", "")
    g = lambda k, d=None: m.get("%s.%s" % (arch, k), d)
    n_layer = int(g("block_count", 0))
    n_main = n_layer - int(g("nextn_predict_layers", 0) or 0)
    head_kv = g("attention.head_count_kv")
    n_head = g("attention.head_count")
    key_len = g("attention.key_length")
    if key_len is None and n_head:
        n_embd = g("embedding_length")
        key_len = n_embd // (n_head[0] if isinstance(n_head, list) else n_head)
    interval = g("full_attention_interval")
    swa = g("attention.sliding_window_pattern")
    layers = {}
    for il in range(n_main):
        if interval and (il + 1) % int(interval) != 0:
            continue
        if isinstance(swa, list) and il < len(swa) and swa[il]:
            continue
        hkv = head_kv[il] if isinstance(head_kv, list) else head_kv
        if not hkv:
            continue
        layers[il] = (int(key_len), int(hkv))
    return dict(arch=arch, basename=m.get("general.basename", ""), size=m.get("general.size_label", ""), layers=layers)


def _young_list(young, fallback, b):
    y = young if young is not None else (fallback if fallback is not None else Y_DEFAULT)
    return [y] * len(b)


def convert(src, entry=0, young=None, pool=None, cap=None):
    """kvfq text plan or alloc3_plans.json entry -> plan dict (validated)."""
    layers = {}
    if src.lower().endswith(".json"):
        with io.open(src, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list) or not (0 <= entry < len(data)):
            raise PlanError("%s: entry %d not found (%d entries)" % (src, entry, len(data) if isinstance(data, list) else 0))
        e = data[entry]
        bK, bV = e["bK"], e["bV"]
        if len(bK) != len(QWEN38_ATTN_LAYERS) or len(bV) != len(QWEN38_ATTN_LAYERS):
            raise PlanError("%s: entry %d has %d/%d layers, expected %d" % (src, entry, len(bK), len(bV), len(QWEN38_ATTN_LAYERS)))
        m = e.get("b2")
        w2 = e.get("W2")
        for i, il in enumerate(QWEN38_ATTN_LAYERS):
            bk, bv = [int(x) for x in bK[i]], [int(x) for x in bV[i]]
            layers[il] = dict(bk=bk, bv=bv, yk=_young_list(young, m, bk), yv=_young_list(young, m, bv))
        origin = "%s entry %d (W1 %s, W2 %s, b2 %s)" % (src, entry, e.get("W1"), w2, m)
    else:
        m = w2 = None
        with io.open(src, encoding="utf-8") as f:
            for ln, raw in enumerate(f, 1):
                tok = raw.split("#", 1)[0].split()
                if not tok:
                    continue
                if tok[0] == "L" and len(tok) == 12:
                    il = int(tok[1])
                    layers[il] = dict(bk=[int(t) for t in tok[3:7]], bv=[int(t) for t in tok[8:12]])
                elif tok[0] == "M" and len(tok) == 2:
                    m = int(tok[1])
                elif tok[0] == "W2" and len(tok) == 2:
                    w2 = int(tok[1])
                elif tok[0] in ("W",):
                    pass
                else:
                    raise PlanError("%s line %d: unexpected '%s'" % (src, ln, raw.strip()))
        for il, l in layers.items():
            l["yk"] = _young_list(young, m, l["bk"])
            l["yv"] = _young_list(young, m, l["bv"])
        origin = "%s (W2 %s, M %s)" % (src, w2, m)
    plan = dict(layers=layers,
                pool=POOL_DEFAULT if pool is None else pool,
                cap=cap if cap is not None else (w2 if w2 else CAP_DEFAULT),
                ignored=[])
    text = render(plan)
    return parse_text(text), origin


def _cli_shape(args):
    """make_shape of the auto / fingerprint options, and the GGUF info (or None)."""
    info = None
    if args.gguf:
        info = gguf_info(args.gguf)
        layers = info["layers"]
        if args.layers:
            keep = {int(x) for x in args.layers.split(",") if x.strip()}
            layers = {il: g for il, g in layers.items() if il in keep}
    else:
        if not args.layers:
            raise PlanError("give --gguf or --layers")
        layers = {int(x): (args.head_dim, args.n_head_kv) for x in args.layers.split(",") if x.strip()}
    return ctx_shape(layers, args.ctx, args.n_seq_max, args.unified), info


def main():
    ap = argparse.ArgumentParser(description="turbot plan tools")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert", help="kvfq .txt plan or alloc3_plans.json entry -> turbot plan")
    c.add_argument("src")
    c.add_argument("--entry", type=int, default=0, help="entry index in a .json plan list")
    c.add_argument("--young", type=int, default=None, help="young width for every head (default: M / b2, else 7)")
    c.add_argument("--pool", type=int, default=None, help="young pool cells (default 65536)")
    c.add_argument("--cap", type=int, default=None, help="per-sequence young cap (default: W2, else 16384)")
    c.add_argument("--kv", type=int, default=KV_DEFAULT, help="cells for the VRAM line (default 262144)")
    c.add_argument("-o", "--out", default=None, help="output plan file (default: print)")
    for name, hlp in (("auto", "the automatic plan of llama_turbot_plan_auto_text (same text and hash)"),
                      ("fingerprint", "the '# model:' fingerprint of a model")):
        a = sub.add_parser(name, help=hlp)
        a.add_argument("--gguf", default=None, help="read the attention layers and their geometry from this model")
        a.add_argument("--layers", default=None, help="comma list of attention layers (with --gguf: keep only these)")
        a.add_argument("--head-dim", type=int, default=256)
        a.add_argument("--n-head-kv", type=int, default=4)
        a.add_argument("-c", "--ctx", type=int, default=KV_DEFAULT, help="n_ctx (default 262144)")
        a.add_argument("-np", "--n-seq-max", type=int, default=1, help="n_seq_max (default 1)")
        a.add_argument("--unified", action="store_true", help="--kv-unified: one stream for all sequences")
        if name == "auto":
            a.add_argument("--budget", choices=("fallback", "turbo5p"), default="fallback",
                           help="turbo5p = LLAMA_TURBOT_AUTO_BUDGET=turbo5p")
            a.add_argument("-o", "--out", default=None, help="output plan file (default: print)")
        else:
            a.add_argument("--arch", default=None)
            a.add_argument("--basename", default=None)
            a.add_argument("--size", default=None)
    args = ap.parse_args()

    if args.cmd == "auto":
        try:
            shape, _ = _cli_shape(args)
            text = auto_plan(shape, args.budget == "turbo5p")
            plan = parse_text(text, shape=shape)
        except (PlanError, OSError, KeyError, ValueError) as e:
            print("REFUSED: %s" % e)
            return 1
        if args.out:
            with io.open(args.out, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
            print("wrote %s" % args.out)
        else:
            sys.stdout.write(text)
            sys.stdout.flush()
        sys.stderr.write("hash 0x%016x\n" % plan_hash(plan))
        return 0

    if args.cmd == "fingerprint":
        try:
            shape, info = _cli_shape(args)
        except (PlanError, OSError, KeyError, ValueError) as e:
            print("ERROR: %s" % e)
            return 1
        info = info or {}
        print(fingerprint_text(args.arch if args.arch is not None else info.get("arch", ""),
                               args.basename if args.basename is not None else info.get("basename", ""),
                               args.size if args.size is not None else info.get("size", ""), shape))
        return 0

    try:
        plan, origin = convert(args.src, args.entry, args.young, args.pool, args.cap)
        parse_text(render(plan), QWEN38_ATTN_LAYERS, args.kv)
    except (PlanError, KeyError, ValueError, OSError) as e:
        print("ERROR: %s" % e)
        return 1
    lines = ["turbot plan converted from " + origin] + summary_lines(plan, args.kv)
    text = render(plan, lines)
    if args.out:
        with io.open(args.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        print("wrote %s" % args.out)
    else:
        sys.stdout.write(text)
    for l in lines[1:]:
        print(l)
    return 0


if __name__ == "__main__":
    sys.exit(main())
