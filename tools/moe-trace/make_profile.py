#!/usr/bin/env python3
# [TAG_FN_MOE_TRACE] Build a hot-set profile (moeprof v1) from MOET traces and/or merge existing profiles, and print the
# hot plan the engine would pick for a VRAM budget (the same greedy as llama_moe_hot_init in src/llama-moecache.cpp).
#
#   python tools/moe-trace/make_profile.py --trace code.moet --trace prose.moet --out flashnext-q4.moeprof
#   python tools/moe-trace/make_profile.py --trace pf_code.moet --prefill-as-decode 3 --out code_pf.moeprof
#   python tools/moe-trace/make_profile.py --profile a.moeprof --profile b.moeprof --out merged.moeprof
#   python tools/moe-trace/make_profile.py --profile flashnext-q4.moeprof --plan-mib 14300 --types types_A.json --host-layers 0-47
#
# --prefill-as-decode k: prefill records (teacher forcing) are cut into windows of k consecutive tokens, and each window
# counts as one decode step (its union), which approximates MTP verify batches of width k from prompt data.
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moet import read_profile, read_trace, write_profile  # noqa: E402

SECTIONS = ("decode_union", "decode_tokens", "prefill_tokens", "decode_weight_u")
# relative CPU seconds per byte by type; must match LLAMA_MOE_HOT_COST's defaults in src/llama-moecache.cpp
DEFAULT_COST = {"q5_1": 1.3}
BPE = {"q4_K": 144 / 256, "q5_K": 176 / 256, "q6_K": 210 / 256, "q5_1": 24 / 32, "q8_0": 34 / 32, "iq4_nl": 18 / 32,
       "q4_0": 18 / 32, "iq4_xs": 136 / 256, "iq3_s": 110 / 256, "q3_K": 110 / 256}


def parse_layers(s, n):
    if not s:
        return list(range(n))
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def from_traces(paths, k_prefill):
    n_expert = None
    acc = {s: {} for s in SECTIONS}
    stats = {"steps": 0, "decode_tokens": 0, "prefill_tokens": 0, "draft_ubatches": 0}

    def add(sec, il, idx, val=1.0):
        a = acc[sec].setdefault(il, np.zeros(n_expert, dtype=np.float64))
        np.add.at(a, idx, val)

    for p in paths:
        for ne, rec in read_trace(p):
            n_expert = ne
            if rec.draft:
                stats["draft_ubatches"] += 1
            elif rec.prefill:
                stats["prefill_tokens"] += rec.n_tokens
            else:
                stats["steps"] += 1
                stats["decode_tokens"] += rec.n_tokens
            for L in rec.layers:
                if rec.prefill:
                    add("prefill_tokens", L.il, L.ids.ravel())
                    if k_prefill > 0:
                        for t0 in range(0, L.ids.shape[0], k_prefill):
                            add("decode_union", L.il, np.unique(L.ids[t0:t0 + k_prefill]))
                        if L.il == rec.layers[0].il:
                            stats["steps"] += (L.ids.shape[0] + k_prefill - 1) // k_prefill
                else:
                    add("decode_tokens", L.il, L.ids.ravel())
                    add("decode_union", L.il, np.unique(L.ids))
                    add("decode_weight_u", L.il, L.ids.ravel(), (L.w.ravel() * 1e6).astype(np.float64))
    return n_expert, acc, stats


def plan(sec, n_expert, budget_mib, types, host_layers, cost):
    """greedy by count x cost-per-byte; returns {il: sorted expert ids}, bytes used"""
    layer_bytes, layer_cost = {}, {}
    for row in types["layers"]:
        il = row["layer"]
        parts = [BPE.get(row[k], 1.0) for k in ("gate", "up", "down")]   # the three matrices have equal element counts
        layer_bytes[il] = row["expert_bytes"]
        layer_cost[il] = sum(p * cost.get(row[k], 1.0) for p, k in zip(parts, ("gate", "up", "down"))) / sum(parts)
    cand = []
    for il in host_layers:
        if il not in sec or il not in layer_bytes:
            continue
        v = sec[il] * layer_cost[il]
        for e in np.nonzero(sec[il])[0]:
            cand.append((v[e], il, int(e)))
    cand.sort(reverse=True)
    budget = budget_mib * 2**20
    used, chosen = 0, {}
    for v, il, e in cand:
        b = layer_bytes[il]
        if used + b > budget:
            continue
        used += b
        chosen.setdefault(il, []).append(e)
    return {il: sorted(v) for il, v in chosen.items()}, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", action="append", default=[])
    ap.add_argument("--profile", action="append", default=[])
    ap.add_argument("--prefill-as-decode", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--source", default="")
    ap.add_argument("--plan-mib", type=float, default=0)
    ap.add_argument("--types", default="", help="fn_types.py --json output (expert bytes and types per layer)")
    ap.add_argument("--host-layers", default="0-47")
    ap.add_argument("--section", default="decode_union")
    ap.add_argument("--cost", default="", help="type=factor,... relative CPU cost per byte (default q5_1=1.3)")
    a = ap.parse_args()

    n_expert, acc, hdr = None, {s: {} for s in SECTIONS}, {}
    if a.trace:
        n_expert, acc, st = from_traces(a.trace, a.prefill_as_decode)
        hdr.update(st)
    for p in a.profile:
        h, secs = read_profile(p)
        n_expert = int(h["n_expert"])
        for s, layers in secs.items():
            for il, arr in layers.items():
                cur = acc.setdefault(s, {}).get(il)
                acc[s][il] = arr if cur is None else cur + arr
        for k in ("steps", "decode_tokens", "prefill_tokens", "draft_ubatches"):
            if k in h:
                hdr[k] = hdr.get(k, 0) + int(h[k])
    if n_expert is None:
        raise SystemExit("no input")
    acc = {s: v for s, v in acc.items() if v}
    hdr["source"] = (a.source or "+".join(os.path.basename(p) for p in a.trace + a.profile)).replace(" ", "_")

    for s in ("decode_union", "decode_tokens", "prefill_tokens"):
        if s in acc:
            tot = sum(v.sum() for v in acc[s].values())
            print("%-15s layers %2d total %12.0f" % (s, len(acc[s]), tot))
    if "decode_union" in acc and acc["decode_union"]:
        # skew: share of the top 20% experts in each layer, averaged
        shares = []
        for il, v in sorted(acc["decode_union"].items()):
            if v.sum() > 0:
                s = np.sort(v)[::-1]
                shares.append(s[: max(1, len(s) // 5)].sum() / s.sum())
        print("top-20%% experts carry %.3f of decode routings (mean over %d layers)" % (np.mean(shares), len(shares)))

    if a.out:
        write_profile(a.out, n_expert, acc, hdr)
        print("wrote %s" % a.out)

    if a.plan_mib:
        if not a.types:
            raise SystemExit("--plan-mib needs --types")
        with open(a.types, encoding="utf-8") as f:
            types = json.load(f)
        cost = dict(DEFAULT_COST)
        for kv in filter(None, a.cost.split(",")):
            k, v = kv.split("=")
            cost[k] = float(v)
        sec = acc.get(a.section)
        if not sec:
            raise SystemExit("profile has no %s section" % a.section)
        chosen, used = plan(sec, n_expert, a.plan_mib, types, parse_layers(a.host_layers, 48), cost)
        tot_hit, tot = 0.0, 0.0
        for il in parse_layers(a.host_layers, 48):
            if il not in sec:
                continue
            v = sec[il]
            h = v[chosen.get(il, [])].sum()
            tot_hit += h
            tot += v.sum()
            print("blk.%-2d n_hot %3d  in-sample hit %.3f" % (il, len(chosen.get(il, [])), h / v.sum() if v.sum() else 0))
        print("plan: %.0f MiB of %.0f, %d experts, in-sample hit rate %.3f (routed units, not bytes)" %
              (used / 2**20, a.plan_mib, sum(len(v) for v in chosen.values()), tot_hit / tot if tot else 0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
