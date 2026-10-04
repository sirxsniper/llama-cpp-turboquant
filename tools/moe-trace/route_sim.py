#!/usr/bin/env python3
# [TAG_FN_MOE_TRACE] Offline replay of MoE routing traces (MOET v1) against hot-set policies. CPU only, no model needed.
# Decides D3 of E:/turbot-gates/flashnext/PLAN.md: static vs adaptive hit rate at the 131K / 262K budgets, the routing
# skew alpha (h = f^alpha), the verify-window union U(k), and the next-layer router recall (prefetch, F7c).
#
#   python tools/moe-trace/route_sim.py --trace code=E:/.../code.moet --trace prose=E:/.../prose.moet \
#       --types E:/turbot-gates/flashnext/recipe/types_A.json --budget-mib 14300,16000 --out E:/turbot-gates/flashnext/route
#   python tools/moe-trace/route_sim.py --selftest
#
# Units: a "step" is one decode ubatch (or, with --prefill-as-decode k, a window of k teacher-forced prompt tokens);
# per step and layer the traffic is the union of the experts its tokens use. Hit rate = resident bytes of that union /
# all bytes of that union, over the host layers (--host-layers). Policies:
#   static   top experts by train-split counts x CPU cost per byte (the SP-3 greedy), fixed
#   lru      one global LRU over (layer, expert) with a byte budget; every miss is admitted (upload bytes reported)
#   wlfu     windowed LFU (SP-4): starts from static; admits an expert seen >= N times in the last W steps when it beats
#            the coldest resident by more than the hysteresis; at most --admit-mib per step; the copy lands next step
#   belady   optimal replacement with unit sizes (upper bound; --no-belady to skip)
#   decay    [TAG_FN_R4_ADAPT_DECAY] the decayed-count policy of LLAMA_MOE_HOT_DECAY (src/llama-moe-decay.h): every step
#            adds 1 per sighting, every --decay-every steps one pass pairs the best non-residents with the weakest
#            residents (admit >= --decay-admit, > --decay-ratio x and > --decay-hyst + the victim) under --admit-mib, then
#            all counts x --decay (Strata #407: long memory 31-38% fewer misses than short windows); --decay 0 skips it
from __future__ import annotations

import argparse
import collections
import heapq
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_profile import BPE, DEFAULT_COST, parse_layers  # noqa: E402
from moet import read_trace  # noqa: E402

N_LAYER_MAX = 64


def load_events(path, host, k_prefill, use_prefill_steps):
    """steps: list of int arrays of keys (il*n_expert + e), unique per layer; plus U(k), recall and prediction rows"""
    steps, n_expert = [], None
    ustats = collections.defaultdict(list)
    rec_hit = rec_tot = 0
    pred_rows = []
    for ne, rec in read_trace(path):
        n_expert = ne
        if rec.draft:
            continue
        for L in rec.layers:
            if L.pred is not None and L.pred.shape[0] == L.ids.shape[0]:
                for t in range(L.ids.shape[0]):
                    rec_hit += len(np.intersect1d(L.ids[t], L.pred[t]))
                    rec_tot += L.ids.shape[1]
                if len(pred_rows) < 100000 and L.il in host:
                    pred_rows.append((L.il, L.ids, L.pred))
        if rec.prefill:
            wins = collections.defaultdict(list)
            for L in rec.layers:
                for k in (1, 2, 3, 4):
                    for t0 in range(0, L.ids.shape[0] - k + 1, k):
                        ustats[str(k)].append(len(np.unique(L.ids[t0:t0 + k])))
                if L.il in host and k_prefill > 0:
                    for w0 in range(0, L.ids.shape[0], k_prefill):
                        wins[w0].append(L.il * ne + np.unique(L.ids[w0:w0 + k_prefill]))
            if use_prefill_steps:
                for w0 in sorted(wins):
                    steps.append(np.concatenate(wins[w0]))
        else:
            parts = []
            for L in rec.layers:
                ustats["T%d" % L.ids.shape[0]].append(len(np.unique(L.ids)))
                if L.il in host:
                    parts.append(L.il * ne + np.unique(L.ids))
            if parts:
                steps.append(np.concatenate(parts))
    return n_expert, steps, ustats, (rec_hit / rec_tot if rec_tot else None), pred_rows


def key_arrays(types, n_expert):
    kb = np.zeros(N_LAYER_MAX * n_expert)
    kc = np.ones(N_LAYER_MAX * n_expert)
    for row in types["layers"]:
        il = row["layer"]
        parts = [BPE.get(row[k], 1.0) for k in ("gate", "up", "down")]
        c = sum(p * DEFAULT_COST.get(row[k], 1.0) for p, k in zip(parts, ("gate", "up", "down"))) / sum(parts)
        kb[il * n_expert:(il + 1) * n_expert] = row["expert_bytes"]
        kc[il * n_expert:(il + 1) * n_expert] = c
    return kb, kc


def static_mask(steps, kb, kc, budget):
    cnt = np.zeros(len(kb))
    for s in steps:
        cnt[s] += 1
    order = np.argsort(-(cnt * kc), kind="stable")
    order = order[cnt[order] > 0]
    csum = np.cumsum(kb[order])
    mask = np.zeros(len(kb), dtype=bool)
    mask[order[csum <= budget]] = True
    return mask


def eval_mask(steps, mask, kb):
    hit = tot = 0.0
    for s in steps:
        b = kb[s]
        tot += b.sum()
        hit += b[mask[s]].sum()
    return hit / tot if tot else 0.0


def sim_lru(steps, kb, budget):
    lru, used = collections.OrderedDict(), 0.0
    hit = tot = upl = 0.0
    for s in steps:
        for key in s.tolist():
            b = kb[key]
            tot += b
            if key in lru:
                hit += b
                lru.move_to_end(key)
            else:
                lru[key] = b
                used += b
                upl += b
                while used > budget:
                    _, vb = lru.popitem(last=False)
                    used -= vb
    return hit / tot if tot else 0.0, upl / max(1, len(steps))


def sim_wlfu(steps, kb, budget, start_mask, W, N, hyst, admit_bytes):
    res = start_mask.copy()
    used = kb[res].sum()
    wc = np.zeros(len(kb), dtype=np.int32)
    win = collections.deque()
    hit = tot = upl = 0.0
    pending = np.zeros(0, dtype=np.int64)
    for s in steps:
        res[pending] = True              # copies queued last step have landed
        b = kb[s]
        tot += b.sum()
        hit += b[res[s]].sum()
        win.append(s)
        wc[s] += 1
        if len(win) > W:
            wc[win.popleft()] -= 1
        cand = s[(~res[s]) & (wc[s] >= N)]
        pending = []
        if len(cand):
            cand = cand[np.argsort(-wc[cand], kind="stable")]
            res_idx = np.nonzero(res)[0]
            vict = res_idx[np.argsort(wc[res_idx], kind="stable")] if len(res_idx) else res_idx
            vi, left = 0, admit_bytes
            for key in cand.tolist():
                bk = kb[key]
                if bk > left:
                    break
                ok = True
                while used + bk > budget:
                    while vi < len(vict) and not res[vict[vi]]:
                        vi += 1
                    if vi >= len(vict) or wc[vict[vi]] + hyst >= wc[key]:
                        ok = False
                        break
                    res[vict[vi]] = False
                    used -= kb[vict[vi]]
                    vi += 1
                if not ok:
                    break
                used += bk               # evict first, reserve the slot, publish when the copy has landed
                left -= bk
                upl += bk
                pending.append(key)
        pending = np.array(pending, dtype=np.int64)
    return hit / tot if tot else 0.0, upl / max(1, len(steps))


def sim_decay(steps, kb, budget, start_mask, decay, every, admit, ratio, hyst, admit_bytes):
    """[TAG_FN_R4_ADAPT_DECAY] the LLAMA_MOE_HOT_DECAY policy on one global byte budget (the real set has per-layer slots;
    the budget is spread the same way by the starting mask). Copies land at the next step."""
    res = start_mask.copy()
    used = kb[res].sum()
    cnt = np.zeros(len(kb))
    cnt[res] = admit
    hit = tot = upl = 0.0
    pending = np.zeros(0, dtype=np.int64)
    busy = np.zeros(len(kb), dtype=bool)
    for i, s in enumerate(steps):
        res[pending] = True
        busy[pending] = False
        b = kb[s]
        tot += b.sum()
        hit += b[res[s]].sum()
        cnt[s] += 1.0
        pend = []
        if (i + 1) % every == 0:
            cand = np.nonzero((~res) & (~busy) & (cnt >= admit))[0]
            cand = cand[np.argsort(-cnt[cand], kind="stable")]
            res_idx = np.nonzero(res)[0]
            vict = res_idx[np.argsort(cnt[res_idx], kind="stable")]
            vi, left = 0, admit_bytes
            for key in cand.tolist():
                bk = kb[key]
                if bk > left:
                    break
                ok = True
                while used + bk > budget:
                    if vi >= len(vict):
                        ok = False
                        break
                    v = vict[vi]
                    if not (cnt[key] > ratio * cnt[v] and cnt[key] > cnt[v] + hyst):
                        ok = False
                        break
                    res[v] = False
                    used -= kb[v]
                    vi += 1
                if not ok:
                    break
                used += bk
                left -= bk
                upl += bk
                busy[key] = True
                pend.append(key)
            cnt *= decay
        pending = np.array(pend, dtype=np.int64)
    return hit / tot if tot else 0.0, upl / max(1, len(steps))


def sim_belady(steps, kb, budget):
    cap = int(budget // np.mean(kb[kb > 0]))
    seq = np.concatenate(steps).tolist() if steps else []
    nxt = [0] * len(seq)
    last = {}
    for i in range(len(seq) - 1, -1, -1):
        nxt[i] = last.get(seq[i], math.inf)
        last[seq[i]] = i
    res, heap, cur = set(), [], {}
    hit = 0
    for i, key in enumerate(seq):
        if key in res:
            hit += 1
        else:
            if len(res) >= cap:
                while heap:
                    negn, k = heapq.heappop(heap)
                    if k in res and cur.get(k) == -negn:
                        res.discard(k)
                        break
            res.add(key)
        cur[key] = nxt[i]
        heapq.heappush(heap, (-nxt[i], key))
    return hit / len(seq) if seq else 0.0


def selftest():
    # Zipf-skewed synthetic routing: static and adaptive must beat uniform, belady must bound everything
    rng = np.random.default_rng(1)
    n_expert, n_layer = 64, 4
    types = {"layers": [{"layer": il, "gate": "q4_K", "up": "q4_K", "down": "q5_1", "expert_bytes": 3072000}
                        for il in range(n_layer)]}
    kb, kc = key_arrays(types, n_expert)
    p = 1.0 / np.arange(1, n_expert + 1) ** 1.0
    p /= p.sum()
    steps = [np.concatenate([il * n_expert + np.unique(rng.choice(n_expert, 20, p=p)) for il in range(n_layer)])
             for _ in range(600)]
    budget = 0.25 * n_layer * n_expert * 3072000
    m = static_mask(steps[:300], kb, kc, budget)
    hs = eval_mask(steps[300:], m, kb)
    hl, _ = sim_lru(steps[300:], kb, budget)
    hw, _ = sim_wlfu(steps[300:], kb, budget, m, 16, 3, 1, 64 * 2**20)
    hd, _ = sim_decay(steps[300:], kb, budget, m, 0.92, 2, 2.0, 1.2, 0.5, 64 * 2**20)  # [TAG_FN_R4_ADAPT_DECAY]
    hb = sim_belady(steps[300:], kb, budget)
    ok = 0.3 < hs <= hb + 1e-9 and hw <= hb + 1e-9 and hl <= hb + 1e-9 and hd <= hb + 1e-9 and hd > 0.3 and \
        m.sum() == int(budget // 3072000)
    print("selftest static %.3f lru %.3f wlfu %.3f decay %.3f belady %.3f -> %s" % (hs, hl, hw, hd, hb, "OK" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", action="append", default=[], help="[domain=]path.moet")
    ap.add_argument("--types", default="")
    ap.add_argument("--budget-mib", default="14300,16000")
    ap.add_argument("--host-layers", default="0-47")
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--prefill-as-decode", type=int, default=3)
    ap.add_argument("--no-prefill-steps", action="store_true", help="use only real decode records as steps")
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--admit-n", type=int, default=3)
    ap.add_argument("--hyst", type=int, default=1)
    ap.add_argument("--admit-mib", type=float, default=64)
    # [TAG_FN_R4_ADAPT_DECAY] the decayed policy (0 = skip)
    ap.add_argument("--decay", type=float, default=0.92)
    ap.add_argument("--decay-every", type=int, default=2)
    ap.add_argument("--decay-admit", type=float, default=2.0)
    ap.add_argument("--decay-ratio", type=float, default=1.2)
    ap.add_argument("--decay-hyst", type=float, default=0.5)
    ap.add_argument("--no-belady", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.trace or not a.types:
        raise SystemExit("--trace and --types are required")

    with open(a.types, encoding="utf-8") as f:
        types = json.load(f)
    host = set(parse_layers(a.host_layers, 48))
    budgets = [float(x) for x in a.budget_mib.split(",") if x]
    summary = {"budgets_mib": budgets, "host_layers": sorted(host), "domains": {}, "cross": {}}
    domains = {}
    kb = kc = None
    for spec in a.trace:
        dom, path = spec.split("=", 1) if "=" in spec else (os.path.basename(spec), spec)
        n_expert, steps, ust, recall, pred_rows = load_events(path, host, a.prefill_as_decode, not a.no_prefill_steps)
        if kb is None:
            kb, kc = key_arrays(types, n_expert)
            kb_host = np.zeros_like(kb)
            for il in host:
                kb_host[il * n_expert:(il + 1) * n_expert] = kb[il * n_expert:(il + 1) * n_expert]
            total_bytes = kb_host.sum()
            summary["total_host_expert_mib"] = total_bytes / 2**20
        domains[dom] = steps
        d = {"steps": len(steps), "U": {k: float(np.mean(v)) for k, v in sorted(ust.items()) if v}, "recall_at_k": recall,
             "policies": {}}
        n_tr = int(len(steps) * a.train_frac)
        train, test = steps[:n_tr], steps[n_tr:]
        for B in budgets:
            budget = B * 2**20
            f = budget / total_bytes
            m = static_mask(train, kb, kc, budget)
            row = {"f": f, "static": eval_mask(test, m, kb)}
            row["lru"], up = sim_lru(test, kb, budget)
            row["lru_upload_mib_per_step"] = up / 2**20
            row["wlfu"], up = sim_wlfu(test, kb, budget, m, a.window, a.admit_n, a.hyst, a.admit_mib * 2**20)
            row["wlfu_upload_mib_per_step"] = up / 2**20
            if a.decay > 0:  # [TAG_FN_R4_ADAPT_DECAY]
                row["decay"], up = sim_decay(test, kb, budget, m, a.decay, a.decay_every, a.decay_admit, a.decay_ratio,
                                             a.decay_hyst, a.admit_mib * 2**20)
                row["decay_upload_mib_per_step"] = up / 2**20
            if not a.no_belady:
                row["belady"] = sim_belady(test, kb, budget)
            for k in ("static", "lru", "wlfu", "decay", "belady"):
                if k in row and 0 < row[k] < 1 and 0 < f < 1:
                    row["alpha_" + k] = math.log(row[k]) / math.log(f)
            hits = tot = 0
            for il, ids, pred in pred_rows:
                for t in range(ids.shape[0]):
                    miss = ids[t][~m[il * n_expert + ids[t]]]
                    tot += len(miss)
                    hits += len(np.intersect1d(miss, pred[t]))
            row["pred_recall_nonhot_misses"] = hits / tot if tot else None
            d["policies"][str(int(B))] = row
            print("%-8s %6.0f MiB (f %.3f): static %.3f lru %.3f (+%.1f MiB/step) wlfu %.3f (+%.1f)%s%s alpha_static %s"
                  " nonhot-recall %s" % (dom, B, f, row["static"], row["lru"], row["lru_upload_mib_per_step"], row["wlfu"],
                                         row["wlfu_upload_mib_per_step"],
                                         (" decay %.3f (+%.1f)" % (row["decay"], row["decay_upload_mib_per_step"]))
                                         if "decay" in row else "",
                                         (" belady %.3f" % row["belady"]) if "belady" in row else "",
                                         ("%.3f" % row["alpha_static"]) if "alpha_static" in row else "-",
                                         ("%.3f" % row["pred_recall_nonhot_misses"]) if row["pred_recall_nonhot_misses"] is not None else "-"))
        print("%-8s steps %d U(k) %s recall@k %s" % (dom, len(steps), {k: round(v, 2) for k, v in d["U"].items()}, recall))
        summary["domains"][dom] = d

    for tr, s_tr in domains.items():
        for te, s_te in domains.items():
            if tr == te:
                continue
            for B in budgets:
                h = eval_mask(s_te, static_mask(s_tr, kb, kc, B * 2**20), kb)
                summary["cross"]["%s->%s@%d" % (tr, te, B)] = h
                print("cross static train %-8s test %-8s %6.0f MiB: %.3f" % (tr, te, B, h))

    b0 = str(int(budgets[0]))
    rows = [d["policies"][b0] for d in summary["domains"].values()]
    summary["alpha_static"] = float(np.nanmean([r.get("alpha_static", np.nan) for r in rows]))
    summary["alpha_adaptive"] = float(np.nanmean([r.get("alpha_wlfu", np.nan) for r in rows]))
    summary["hit_static"] = float(np.mean([r["static"] for r in rows]))
    summary["hit_adaptive"] = float(np.mean([r["wlfu"] for r in rows]))
    print("at %s MiB: static %.3f adaptive %.3f alpha_static %.3f alpha_adaptive %.3f  (D3: adaptive < 0.60 -> 65 needs F7;"
          " SP-4 only if adaptive - static >= 0.04)" % (b0, summary["hit_static"], summary["hit_adaptive"],
                                                        summary["alpha_static"], summary["alpha_adaptive"]))
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1)
        print("wrote %s" % os.path.join(a.out, "summary.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
