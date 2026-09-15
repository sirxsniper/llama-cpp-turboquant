# [TAG_TURBOT] Plan file helpers for the turbot tiered KV cache (docs/turbot/SPEC.md 3.6, 9.1). CPU only, stdlib only.
#
# Library used by plan_vram.py and the "convert" command below. The grammar and the validation mirror
# llama_turbot_plan_parse_text (src/llama-kv-tier.cpp); the byte layout and the plan hash mirror ggml/include/ggml-turbot.h.
#
#   python tools/turbot/turbot_plan.py convert E:/kv-s3/plans/fq_3t0_w256_m16384_7b.txt -o my.plan
#   python tools/turbot/turbot_plan.py convert E:/kvdump/alloc3_plans.json --entry 0 -o my.plan
#   python tools/turbot/turbot_plan.py convert E:/kvdump/alloc3_plans.json --entry 1 --young 7 -o my.plan
#
# Conversion rules:
#   kvfq text plans ("L il K b b b b V b b b b", "W", "W2", "M"): old widths from L. Young width = --young if given, else M
#   when present, else 7. CAP = --cap if given, else W2 when present, else 16384. W is dropped (turbot has no exact tier).
#   alloc3_plans.json entries (bK, bV per attention layer, W1, W2, b2): the same with b2 in the place of M. The entries
#   list the 16 attention layers of Qwen3.8-27B in order, il = 3, 7, ..., 63.
#   POOL = --pool if given, else 65536. A young width that is not above an old width is an error unless --young fixes it.
import argparse
import io
import json
import os
import struct
import sys

B_MIN, B_MAX, Y_MAX, Y_DEFAULT = 2, 6, 8, 7
POOL_DEFAULT, CAP_DEFAULT = 65536, 16384
N_HEAD = 4
QWEN38_ATTN_LAYERS = list(range(3, 64, 4))
KV_DEFAULT = 262144
TURBO5P_ROW = 656                      # bytes per 1024 values (turbo5p_0)
FNV_OFFSET, FNV_PRIME = 0xcbf29ce484222325, 0x100000001b3
MIB = 1024.0 * 1024.0


class PlanError(Exception):
    pass


def side_bytes(b, y):
    """(base row bytes, young part bytes) of one layer-side (SPEC 3.2, 3.3)."""
    s = sum(b)
    r = sum(yy - bb for bb, yy in zip(b, y))
    return 32 * s + 16, 32 * r + 16


def check_widths(b, y, where):
    for bb, yy in zip(b, y):
        if not (B_MIN <= bb <= B_MAX):
            raise PlanError("%s: old width %d outside [%d, %d]" % (where, bb, B_MIN, B_MAX))
        if not (bb < yy <= Y_MAX):
            raise PlanError("%s: young width %d outside [%d, %d]" % (where, yy, bb + 1, Y_MAX))


def parse_text(text, attn_layers=None, kv_size=None):
    """-> dict(layers={il: dict(bk, bv, yk, yv)}, pool, cap, ignored) or raises PlanError naming the line."""
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
            if len(tok) != 12 or tok[2] != "K" or tok[7] != "V":
                raise PlanError("%s: malformed, expected '%s il K w w w w V w w w w'" % (where, tag))
            try:
                il = int(tok[1])
                k = [int(t) for t in tok[3:7]]
                v = [int(t) for t in tok[8:12]]
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
                if val < 0 or val % 64 != 0:
                    raise PlanError("%s: POOL must be a non-negative multiple of 64" % where)
                pool = val    # a pool above kv_size is clamped after the loop, like llama-kv-tier.cpp
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
        yk, yv = ([Y_DEFAULT] * N_HEAD, [Y_DEFAULT] * N_HEAD)
        if il in Y:
            yk, yv = Y[il][0], Y[il][1]
        check_widths(bk, yk, where + " K")
        check_widths(bv, yv, where + " V")
        layers[il] = dict(bk=bk, bv=bv, yk=yk, yv=yv)
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
    if kv_size is not None and pool > kv_size:
        pool = (kv_size // 64) * 64    # clamped to the cache in whole granules, as the C++ parser does
    return dict(layers=layers, pool=POOL_DEFAULT if pool is None else pool, cap=CAP_DEFAULT if cap is None else cap,
                ignored=sorted(set(ignored)))


def parse_file(path, attn_layers=None, kv_size=None):
    with io.open(path, encoding="utf-8") as f:
        return parse_text(f.read(), attn_layers, kv_size)


def plan_hash(plan):
    """ggml_turbot_plan_hash_layer over ascending il, then ggml_turbot_plan_hash_finish (FNV-1a 64)."""
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
            h = fnv(h, bytes(l[key]))
    h = fnv(h, struct.pack("<I", plan["pool"]))
    h = fnv(h, struct.pack("<I", plan["cap"]))
    return h


def vram(plan, kv_size=KV_DEFAULT):
    """-> dict of per-cell and total byte counts (SPEC 3.6)."""
    base_cell = pool_cell = s_sum = 0
    rows = []
    for il in sorted(plan["layers"]):
        l = plan["layers"][il]
        bk, yk = side_bytes(l["bk"], l["yk"])
        bv, yv = side_bytes(l["bv"], l["yv"])
        base_cell += bk + bv
        pool_cell += yk + yv
        s_sum += sum(l["bk"]) + sum(l["bv"])
        rows.append((il, bk, bv, yk, yv))
    n_layers = len(plan["layers"])
    base = kv_size * base_cell
    pool = plan["pool"] * pool_cell
    turbo5p = kv_size * n_layers * 2 * TURBO5P_ROW
    return dict(rows=rows, base_cell=base_cell, pool_cell=pool_cell, base=base, pool=pool, total=base + pool,
                turbo5p=turbo5p, s_sum=s_sum, n_heads=n_layers * 2 * N_HEAD, n_layers=n_layers, kv_size=kv_size)


def render(plan, header_lines=()):
    out = ["# " + h for h in header_lines]
    out.append("POOL %d" % plan["pool"])
    out.append("CAP %d" % plan["cap"])
    for il in sorted(plan["layers"]):
        l = plan["layers"][il]
        out.append("L %d K %s V %s" % (il, " ".join(map(str, l["bk"])), " ".join(map(str, l["bv"]))))
    for il in sorted(plan["layers"]):
        l = plan["layers"][il]
        if l["yk"] != [Y_DEFAULT] * N_HEAD or l["yv"] != [Y_DEFAULT] * N_HEAD:
            out.append("Y %d K %s V %s" % (il, " ".join(map(str, l["yk"])), " ".join(map(str, l["yv"]))))
    return "\n".join(out) + "\n"


def summary_lines(plan, kv_size=KV_DEFAULT):
    v = vram(plan, kv_size)
    return [
        "base %.2f MiB + young pool %.2f MiB = %.2f MiB at %d cells (turbo5p %.2f MiB, margin %.2f MiB)"
        % (v["base"] / MIB, v["pool"] / MIB, v["total"] / MIB, kv_size, v["turbo5p"] / MIB, (v["turbo5p"] - v["total"]) / MIB),
        "%d layers, old bits mean %.3f (sum %d over %d heads), base %d B/cell, pool %d B/pool cell, POOL %d (%d granules), CAP %d, hash 0x%016x"
        % (v["n_layers"], v["s_sum"] / float(v["n_heads"]), v["s_sum"], v["n_heads"], v["base_cell"], v["pool_cell"],
           plan["pool"], plan["pool"] // 64, plan["cap"], plan_hash(plan)),
    ]


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


def main():
    ap = argparse.ArgumentParser(description="turbot plan converter")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert", help="kvfq .txt plan or alloc3_plans.json entry -> turbot plan")
    c.add_argument("src")
    c.add_argument("--entry", type=int, default=0, help="entry index in a .json plan list")
    c.add_argument("--young", type=int, default=None, help="young width for every head (default: M / b2, else 7)")
    c.add_argument("--pool", type=int, default=None, help="young pool cells (default 65536)")
    c.add_argument("--cap", type=int, default=None, help="per-sequence young cap (default: W2, else 16384)")
    c.add_argument("--kv", type=int, default=KV_DEFAULT, help="cells for the VRAM line (default 262144)")
    c.add_argument("-o", "--out", default=None, help="output plan file (default: print)")
    args = ap.parse_args()
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
