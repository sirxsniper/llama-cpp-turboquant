#!/usr/bin/env python3
# [TAG_FN_SP0] Header-only type report of a Flash-Next split set (reads no tensor data).
#
#   python tools/qwen4exp/fn_types.py <shard 00001 path> [--expect A|B|C] [--json out.json]
#
# --expect checks the recipe files of E:/turbot-gates/flashnext/PLAN.md 7.3:
#   A: exact UD-Q4_K_XL + shared Q8_0 head: ffn_down_exps q5_1 on 43 layers, q8_0 on 2,4,30,46,47; blk.48 all q8_0
#   B: A with ffn_down_exps iq4_nl on those 43 layers; 2,4,30,46,47 still q8_0; blk.48 unchanged
#   C: A with blk.48 experts gate/up q4_K and down q5_1
# --expect-mtp gate,up,down ([TAG_FN_L3_MTP_Q2]): A's trunk with the MTP block's experts at that mix, e.g. q2_0,q2_0,q2_0
# It also prints the expert bytes per layer and per routed token, which the perf model and the hot-set budget use.
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from merge_mtp import Header, nbytes  # noqa: E402

TYPE_NAMES = {0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1", 6: "q5_0", 7: "q5_1", 8: "q8_0", 10: "q2_K", 11: "q3_K",
              12: "q4_K", 13: "q5_K", 14: "q6_K", 16: "iq2_xxs", 17: "iq2_xs", 18: "iq3_xxs", 19: "iq1_s", 20: "iq4_nl",
              21: "iq3_s", 22: "iq2_s", 23: "iq4_xs", 30: "bf16", 42: "q2_0"}
Q8_LAYERS = {2, 4, 30, 46, 47}


def shard_paths(first):
    m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", first)
    if not m:
        return [first]
    n = int(m.group(2))
    return [first[:m.start()] + "-%05d-of-%05d.gguf" % (i, n) for i in range(1, n + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--expect", choices=["A", "B", "C"])
    ap.add_argument("--expect-mtp", default="", help="gate,up,down types of blk.48 on A's trunk (implies --expect A)")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    mtp_mix = None
    if a.expect_mtp:  # [TAG_FN_L3_MTP_Q2]
        mtp_mix = tuple(x.strip() for x in a.expect_mtp.split(","))
        if len(mtp_mix) != 3:
            raise SystemExit("--expect-mtp needs gate,up,down")
        a.expect = "A"

    exps = {}   # (layer, kind) -> (type, bytes, n_expert)
    n_tensors = 0
    for p in shard_paths(a.model):
        h = Header(p)
        n_tensors += len(h.tensors)
        for name, dims, tt, _off in h.tensors:
            m = re.match(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$", name)
            if m:
                exps[(int(m.group(1)), m.group(2))] = (tt, nbytes(dims, tt), dims[2])
    layers = sorted({k[0] for k in exps})
    rows, errors = [], []
    per_token = 0.0
    for il in layers:
        t = {k: exps.get((il, k)) for k in ("gate", "up", "down")}
        if None in t.values():
            errors.append("layer %d: missing expert tensor" % il)
            continue
        n_exp = t["gate"][2]
        per_exp = sum(v[1] for v in t.values()) / n_exp
        rows.append({"layer": il, "gate": TYPE_NAMES.get(t["gate"][0], t["gate"][0]),
                     "up": TYPE_NAMES.get(t["up"][0], t["up"][0]), "down": TYPE_NAMES.get(t["down"][0], t["down"][0]),
                     "expert_bytes": int(per_exp), "layer_bytes": int(per_exp * n_exp)})
        if il < 48:
            per_token += per_exp * 10
    for r in rows:
        print("blk.%-2d gate %-6s up %-6s down %-6s  %9d B/expert  %7.1f MiB/layer" %
              (r["layer"], r["gate"], r["up"], r["down"], r["expert_bytes"], r["layer_bytes"] / 2**20))
    print("%d tensors, %d expert layers, routed expert bytes per token (trunk, 10 used): %.1f MB" %
          (n_tensors, len(rows), per_token / 1e6))

    if a.expect:
        want_down = {"A": "q5_1", "B": "iq4_nl", "C": "q5_1"}[a.expect]
        for r in rows:
            il = r["layer"]
            if il == 48:
                want = ("q4_K", "q4_K", "q5_1") if a.expect == "C" else ("q8_0", "q8_0", "q8_0")
                if mtp_mix:
                    want = mtp_mix
                if (r["gate"], r["up"], r["down"]) != want:
                    errors.append("blk.48 is %s/%s/%s, expected %s" % (r["gate"], r["up"], r["down"], "/".join(want)))
                continue
            wd = "q8_0" if il in Q8_LAYERS else want_down
            if r["down"] != wd:
                errors.append("blk.%d ffn_down_exps is %s, expected %s" % (il, r["down"], wd))
        if 48 not in layers:
            errors.append("no MTP block (blk.48) in the set")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"model": a.model, "layers": rows, "routed_bytes_per_token": per_token}, f, indent=1)
    for e in errors:
        print("ERROR: " + e)
    if a.expect:
        print("EXPECT %s: %s" % (a.expect, "OK" if not errors else "FAILED"))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
