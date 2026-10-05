#!/usr/bin/env python3
# [TAG_FN_L3_POLICY_SIM] Offline replay of decode routing traces (MOET v1, LLAMA_MOE_TRACE) against the hot-set placement
# policies of src/llama-moecache.cpp. CPU only, no model. Score: cold expert bytes (the experts the CPU must read from DRAM)
# per decode step and per generated token, upload bytes, and the byte hit rate.
#
#   python tools/moe-trace/route_policy.py --trace E:/.../real.moet[:answers.json] --types types_A.json \
#       --budget-mib 10300,14300 --grid default --out E:/.../policy
#   python tools/moe-trace/route_policy.py --selftest
#
# The replay follows the real set step by step:
#   - a decode step (graph of <= 8 tokens, not the MTP draft) is scored against the PUBLISHED resident set: per host layer
#     the union of the experts its tokens route to costs the bytes of its non-resident members;
#   - then the step is counted (once per expert and step = "union", or once per token = "token");
#   - every `every` decode steps a pass pairs candidates with victims (llama_moe_decay_pairs: free slots first, admit, ratio,
#     hysteresis), orders all layers' pairs by gain and queues them under the upload budget; the victim leaves the set at
#     once (evict first) and its slot is busy until the upload has landed;
#   - the upload worker moves `up_bw` bytes per decode step, first in first out; an upload that lands during step i is
#     published at the end of step i (the real worker reports done, the next step boundary publishes): usable from i + 1;
#   - a prefill record (more than 8 tokens: a prompt or a new user turn) adds its token-level routing x `seed` to the
#     counts when it ends ("the prompt-seeded warm-up").
# Allocation of the budget to layers: "even" (llama_fn_even_slots: the same slot count in every host layer, the shipped
# default), "var" (per-layer slot counts by count x CPU cost per byte from a profile: the greedy of llama_moe_hot_init),
# "pool" (one dynamic slot pool per expert size class: any layer of the class can take any slot of the class).
from __future__ import annotations

import argparse
import collections
import copy
import heapq
import itertools
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moet import read_trace  # noqa: E402

BPE = {"q4_K": 4.5, "q5_K": 5.5, "q5_1": 6.0, "q8_0": 8.5, "q6_K": 6.5625, "iq4_nl": 4.5, "q4_0": 4.5, "mxfp4": 4.25}
DEFAULT_COST = {"q5_1": 1.3}
DECODE_MAX_T = 8


# ---------------------------------------------------------------------------------------------------------------------
# trace

class Trace:
    """decode steps of the host layers as flat keys (li * n_expert + e), prompts as token-level count vectors"""

    def __init__(self, name):
        self.name = name
        self.n_expert = 0
        self.host = []            # il of every host layer, li = index
        self.steps = []           # [n_steps] np.int32 unique keys of the step (per layer unique)
        self.tok = []             # [n_steps] np.int32 token-level keys (repeats kept)
        self.T = []               # [n_steps] tokens of the step
        self.prompt_at = {}       # step index -> np.float32 [L*E] token counts of the prefill records before that step
        self.answers = []         # [(first step, n steps, generated tokens or None)]

    @property
    def L(self):
        return len(self.host)

    def n_keys(self):
        return self.L * self.n_expert


def load_trace(spec, host_layers, prefill_as_decode=0):
    """spec: path.moet[:answers.json]. answers.json (optional): {"answers": [{"predicted_n": n}, ...]} in trace order.
    prefill_as_decode k > 0: teacher-forced traces (llama-perplexity) - every prefill record becomes decode steps of k tokens"""
    path, ans_path = spec, None
    if ".moet:" in spec:
        path, ans_path = spec.split(".moet:", 1)
        path += ".moet"
    tr = Trace(os.path.basename(path))
    li_of = {}
    pend_prompt = None
    new_answer = True
    for ne, rec in read_trace(path):
        if tr.n_expert == 0:
            tr.n_expert = ne
            tr.host = sorted(host_layers)
            li_of = {il: i for i, il in enumerate(tr.host)}
        if rec.draft:
            continue
        E = tr.n_expert
        if prefill_as_decode > 0 and (rec.n_tokens > DECODE_MAX_T or rec.prefill):
            k = prefill_as_decode
            T = min(Ly.ids.shape[0] for Ly in rec.layers)
            if new_answer:
                tr.answers.append([len(tr.steps), 0, None])
                new_answer = False
            for t0 in range(0, T - k + 1, k):
                parts, toks = [], []
                for Ly in rec.layers:
                    li = li_of.get(Ly.il)
                    if li is None:
                        continue
                    ids = Ly.ids[t0:t0 + k].reshape(-1)
                    ids = ids[(ids >= 0) & (ids < E)]
                    parts.append(li * E + np.unique(ids))
                    toks.append(li * E + ids)
                tr.steps.append(np.concatenate(parts).astype(np.int32))
                tr.tok.append(np.concatenate(toks).astype(np.int32))
                tr.T.append(k)
                tr.answers[-1][1] += 1
            continue
        if rec.n_tokens > DECODE_MAX_T or rec.prefill:
            if pend_prompt is None:
                pend_prompt = np.zeros(len(tr.host) * E, dtype=np.float32)
            for Ly in rec.layers:
                li = li_of.get(Ly.il)
                if li is None:
                    continue
                ids = Ly.ids.reshape(-1)
                ids = ids[(ids >= 0) & (ids < E)]
                pend_prompt += np.bincount(li * E + ids, minlength=len(tr.host) * E).astype(np.float32)
            new_answer = True
            continue
        parts, toks = [], []
        for Ly in rec.layers:
            li = li_of.get(Ly.il)
            if li is None:
                continue
            ids = Ly.ids.reshape(-1)
            ids = ids[(ids >= 0) & (ids < E)]
            parts.append(li * E + np.unique(ids))
            toks.append(li * E + ids)
        if not parts:
            continue
        si = len(tr.steps)
        if pend_prompt is not None:
            tr.prompt_at[si] = pend_prompt
            pend_prompt = None
        if new_answer:
            tr.answers.append([si, 0, None])
            new_answer = False
        tr.answers[-1][1] += 1
        tr.steps.append(np.concatenate(parts).astype(np.int32))
        tr.tok.append(np.concatenate(toks).astype(np.int32))
        tr.T.append(rec.n_tokens)
    if ans_path:
        with open(ans_path, encoding="utf-8") as f:
            meta = json.load(f)
        gen = [a.get("predicted_n") for a in meta.get("answers", [])]
        if len(gen) == len(tr.answers):
            for a, g in zip(tr.answers, gen):
                a[2] = g
        else:
            print("warning: %s has %d answers, the trace %d - generated tokens unknown" % (ans_path, len(gen), len(tr.answers)))
    return tr


def subtrace(tr, a0, a1, name=None):
    """answers [a0, a1) as a trace of their own"""
    out = Trace(name or "%s[%d:%d]" % (tr.name, a0, a1))
    out.n_expert, out.host = tr.n_expert, tr.host
    s0 = tr.answers[a0][0]
    s1 = tr.answers[a1 - 1][0] + tr.answers[a1 - 1][1]
    out.steps, out.tok, out.T = tr.steps[s0:s1], tr.tok[s0:s1], tr.T[s0:s1]
    out.prompt_at = {k - s0: v for k, v in tr.prompt_at.items() if s0 <= k < s1}
    out.answers = [[a[0] - s0, a[1], a[2]] for a in tr.answers[a0:a1]]
    return out


def concat_traces(trs, name):
    out = Trace(name)
    out.n_expert, out.host = trs[0].n_expert, trs[0].host
    for t in trs:
        s0 = len(out.steps)
        out.steps += t.steps
        out.tok += t.tok
        out.T += t.T
        out.prompt_at.update({k + s0: v for k, v in t.prompt_at.items()})
        out.answers += [[a[0] + s0, a[1], a[2]] for a in t.answers]
    return out


def union_counts(tr):
    c = np.zeros(tr.n_keys())
    for s in tr.steps:
        c[s] += 1.0
    return c


# ---------------------------------------------------------------------------------------------------------------------
# model geometry

class Geo:
    def __init__(self, types, host, n_expert):
        rows = {r["layer"]: r for r in types["layers"]}
        self.L, self.E = len(host), n_expert
        self.bytes = np.zeros(self.L)     # per expert, up + gate + down
        self.cost = np.ones(self.L)       # CPU cost per byte (LLAMA_MOE_HOT_COST: q5_1 = 1.3)
        self.cls = []                     # size class: (gate, up, down, bytes)
        for li, il in enumerate(host):
            r = rows[il]
            parts = [BPE.get(r[k], 1.0) for k in ("gate", "up", "down")]
            self.bytes[li] = r["expert_bytes"]
            self.cost[li] = sum(p * DEFAULT_COST.get(r[k], 1.0) for p, k in zip(parts, ("gate", "up", "down"))) / sum(parts)
            self.cls.append((r["gate"], r["up"], r["down"], r["expert_bytes"]))
        self.kb = np.repeat(self.bytes, n_expert)  # per key
        self.kc = np.repeat(self.cost, n_expert)

    def even_slots(self, budget):
        """llama_fn_even_slots: rows of all layers incl. the zero slot; min(n_expert, rows - 1)"""
        rows = int(budget // self.bytes.sum())
        n = min(self.E, rows - 1) if rows > 1 else 0
        return np.full(self.L, n, dtype=np.int64)

    def var_slots(self, budget, prof):
        """per-layer counts by value = count x cost per byte (llama_moe_hot_init's greedy); every layer with a slot pays its
        zero slot. prof: [L*E] counts"""
        val = (prof * self.kc).reshape(self.L, self.E)
        order = np.argsort(-val.reshape(-1), kind="stable")
        order = order[val.reshape(-1)[order] > 0]
        n = np.zeros(self.L, dtype=np.int64)
        used = 0.0
        for k in order.tolist():
            li = k // self.E
            b = self.bytes[li] * (2 if n[li] == 0 else 1)  # the first slot of a layer brings its zero slot
            if used + b > budget:
                continue
            used += b
            n[li] += 1
        return n

    def classes(self):
        cl = collections.OrderedDict()
        for li, c in enumerate(self.cls):
            cl.setdefault(c, []).append(li)
        return list(cl.values())


# ---------------------------------------------------------------------------------------------------------------------
# the decayed policy (llama-moe-decay.h) on the real set's step order

DEF = dict(decay=0.92, every=2, admit=2.0, ratio=1.2, hyst=0.5, up_mib=128.0, up_n=0, up_bw_mib=256.0, seed=0.0,
           count="union", alloc="even", pool=False, value="count", start="empty", lat=0, token_w=0.0, fold="add",
           prof=None, admit_free=None, seed_norm=0.0, pace=False, burst_mib=0.0, burst_ratio=2.0)


def sim_policy(tr, geo, budget, cfg, state=None, collect_answers=True):
    """returns a result dict; state: a dict from a previous run (persistence: residents and counts) or None"""
    c = dict(DEF)
    c.update(cfg)
    L, E = geo.L, geo.E
    NK = L * E
    kb = geo.kb

    # slots per layer (or per pool)
    if c["alloc"] == "even":
        n_slots = geo.even_slots(budget)
    elif c["alloc"] == "var":
        n_slots = geo.var_slots(budget, c["prof"])
    else:
        raise SystemExit("alloc %s" % c["alloc"])
    pools = geo.classes() if c["pool"] else [[li] for li in range(L)]
    pool_of = np.zeros(L, dtype=np.int64)
    for pi, p in enumerate(pools):
        for li in p:
            pool_of[li] = pi
    # a pool of k layers shares one zero slot: k*(n + 1) - 1 slots in the same bytes (llama_moe_hot_init)
    pool_slots = np.array([int(n_slots[p].sum()) + (len(p) - 1 if len(p) > 1 and n_slots[p].min() > 0 else 0)
                           for p in pools], dtype=np.int64)
    pool_keys = [np.concatenate([np.arange(li * E, (li + 1) * E) for li in p]) for p in pools]
    pool_bytes = np.array([geo.bytes[p[0]] for p in pools])

    cnt = np.zeros(NK)
    res = np.zeros(NK, dtype=bool)
    busy = np.zeros(NK, dtype=bool)  # an upload of this expert is queued or running (its slot is taken)
    used = np.zeros(len(pools), dtype=np.int64)  # resident + in flight per pool

    if state is not None:
        # persistence: the saved residents (best first) up to the slots, and the counts
        cnt[:] = state["cnt"]
        for pi, keys in enumerate(pool_keys):
            r = keys[state["res"][keys]]
            if r.size > pool_slots[pi]:
                r = r[np.argsort(-state["cnt"][r], kind="stable")[:pool_slots[pi]]]
            res[r] = True
            used[pi] = r.size
    elif c["start"] == "static" and c["prof"] is not None:
        prof = c["prof"]
        for pi, keys in enumerate(pool_keys):
            k = keys[np.argsort(-prof[keys], kind="stable")]
            k = k[prof[k] > 0][:pool_slots[pi]]
            res[k] = True
            used[pi] = k.size
            cnt[k] = c["admit"]

    queue = collections.deque()  # [key, bytes left]
    up_bw = c["up_bw_mib"] * 2**20
    up_budget = c["up_mib"] * 2**20
    up_n = int(c["up_n"])

    hit_b = tot_b = up_b = 0.0
    n_steps = 0
    per_ans = []
    ans_i = 0
    ans_hit = ans_tot = 0.0
    warm_hit = warm_tot = 0.0  # the first 64 steps of every answer
    ad_steps = 0
    seed = c["seed"]
    token_w = c["token_w"]
    ans_first = {a[0]: i for i, a in enumerate(tr.answers)}
    ans_step = 0

    for si, s in enumerate(tr.steps):
        if si in ans_first:
            if collect_answers and si > 0 and ans_tot > 0:
                per_ans.append(ans_hit / ans_tot)
            ans_hit = ans_tot = 0.0
            ans_step = 0
        # the prompt (or new turn) before this step: its routing x seed into the counts
        if seed > 0 and si in tr.prompt_at:
            pc = tr.prompt_at[si]
            if c["seed_norm"] > 0:
                # [TAG_FN_L3_POLICY_SEED] LLAMA_MOE_HOT_SEED_NORM: the prompt as seed_norm decode steps of its own routing
                # mix (per layer: the share of the prompt's tokens that route to each expert), whatever its length
                n_tok = pc.reshape(L, E).sum(axis=1, keepdims=True) / 10.0
                cnt += seed * c["seed_norm"] * (pc.reshape(L, E) / np.maximum(n_tok, 1.0)).reshape(-1)
            else:
                cnt += seed * pc
        # score against the published set
        b = kb[s]
        h = b[res[s]].sum()
        t = b.sum()
        hit_b += h
        tot_b += t
        ans_hit += h
        ans_tot += t
        if ans_step < 64:
            warm_hit += h
            warm_tot += t
        ans_step += 1
        n_steps += 1
        # count
        if c["count"] == "union":
            cnt[s] += 1.0
        elif c["count"] == "token":
            np.add.at(cnt, tr.tok[si], 1.0)
        elif c["count"] == "mix":  # once per step, plus token_w per further token that uses it
            cnt[s] += 1.0
            extra = np.bincount(tr.tok[si], minlength=NK)[s] - 1
            cnt[s] += token_w * extra
        elif c["count"] == "cost":  # once per step, x the CPU cost per byte (pools mix no types, so only the order moves)
            cnt[s] += geo.kc[s]
        ad_steps += 1
        # the upload worker during this step: FIFO at up_bw bytes per step; landed = published at this step's end
        left = up_bw
        while queue and left > 0:
            q = queue[0]
            take = min(left, q[1])
            q[1] -= take
            left -= take
            up_b += take
            if q[1] <= 0:
                queue.popleft()
                k = q[0]
                busy[k] = False
                res[k] = True
        # the pass
        if ad_steps % c["every"] == 0:
            swaps = []  # (gain, order, pool, key, victim key or -1)
            order_i = 0
            for pi, keys in enumerate(pool_keys):
                if pool_slots[pi] == 0:
                    continue
                ck = cnt[keys]
                cand = keys[(~res[keys]) & (~busy[keys]) & (ck >= c["admit"])]
                if cand.size == 0:
                    continue
                cand = cand[np.argsort(-cnt[cand], kind="stable")]
                n_free = int(pool_slots[pi] - used[pi])
                nf = min(n_free, cand.size)
                for k in cand[:nf].tolist():
                    swaps.append((cnt[k] + 1.0, order_i, pi, k, -1))
                    order_i += 1
                rest = cand[nf:]
                if rest.size == 0:
                    continue
                vict = keys[res[keys]]
                if vict.size == 0:
                    continue
                vict = vict[np.argsort(cnt[vict], kind="stable")]
                m = min(rest.size, vict.size)
                ce = cnt[rest[:m]]
                cv = cnt[vict[:m]]
                ok = (ce > c["ratio"] * cv) & (ce > cv + c["hyst"])
                kk = m if ok.all() else int(np.argmin(ok))
                for j in range(kk):
                    swaps.append((ce[j] - cv[j], order_i, pi, int(rest[j]), int(vict[j])))
                    order_i += 1
            swaps.sort(key=lambda x: (-x[0], x[1]))
            bytes_q = 0.0
            n_q = 0
            budget_q = up_budget
            total_q = up_budget + c["burst_mib"] * 2**20
            if c["pace"]:
                # [TAG_FN_L3_POLICY_UPLOAD] LLAMA_MOE_HOT_UP_MIB_STEP: at most what the worker moves until the next pass,
                # minus the queued bytes (hot_adapt_decay's cap)
                pace = max(0.0, up_bw * c["every"] - sum(q[1] for q in queue))
                budget_q = min(budget_q, pace)
                total_q = min(total_q, pace)
            for g, _, pi, k, v in swaps:
                bk = pool_bytes[pi]
                if up_n > 0:
                    if n_q >= up_n:
                        break
                elif bytes_q + bk > budget_q:
                    # [TAG_FN_L3_POLICY_BURST] LLAMA_MOE_HOT_BURST_MIB: past the pass budget only strong pairs (a free slot,
                    # or a candidate over burst_ratio x its victim) take the burst budget, so a shifted working set
                    # refills fast while the steady state keeps the small budget
                    if bytes_q + bk > total_q:
                        break
                    if not (v < 0 or cnt[k] > c["burst_ratio"] * cnt[v]):
                        continue
                if v >= 0:
                    res[v] = False  # evict first: the slot is busy until the new expert has landed
                else:
                    used[pi] += 1
                busy[k] = True
                queue.append([k, bk])
                bytes_q += bk
                n_q += 1
            cnt *= c["decay"]
    if collect_answers and ans_tot > 0:
        per_ans.append(ans_hit / ans_tot)
    gen = sum(a[2] for a in tr.answers) if tr.answers and all(a[2] for a in tr.answers) else None
    cold = tot_b - hit_b
    out = {
        "hit": hit_b / tot_b if tot_b else 0.0,
        "hit_warm64": warm_hit / warm_tot if warm_tot else 0.0,
        "cold_mib_step": cold / max(1, n_steps) / 2**20,
        "up_mib_step": up_b / max(1, n_steps) / 2**20,
        "steps": n_steps,
        "slots": int(n_slots.sum()),
        "vram_mib": float(((n_slots + (n_slots > 0)) * geo.bytes).sum() / 2**20),
        "per_answer_hit": [round(x, 4) for x in per_ans],
    }
    if gen:
        out["cold_mib_tok"] = cold / gen / 2**20
        out["up_mib_tok"] = up_b / gen / 2**20
        out["tok_step"] = gen / max(1, n_steps)
    out["_state"] = {"cnt": cnt.copy(), "res": res.copy()}
    return out


# ---------------------------------------------------------------------------------------------------------------------
# references

def sim_static(tr, geo, n_slots, prof):
    """a fixed set: per layer the top n_slots experts of prof (an oracle when prof is the trace's own counts)"""
    L, E = geo.L, geo.E
    res = np.zeros(L * E, dtype=bool)
    for li in range(L):
        keys = np.arange(li * E, (li + 1) * E)
        k = keys[np.argsort(-prof[keys], kind="stable")][:n_slots[li]]
        res[k[prof[k] > 0]] = True
    hit = tot = 0.0
    for s in tr.steps:
        b = geo.kb[s]
        hit += b[res[s]].sum()
        tot += b.sum()
    return hit / tot if tot else 0.0, tot - hit


def _belady_hits(seq, cap):
    """optimal replacement with bypass over one access sequence (keys of one cache): a miss is admitted only when its next
    use comes before the furthest next use among the residents (then that one is evicted). Instant free uploads."""
    if cap <= 0 or not seq:
        return 0
    nxt = [0] * len(seq)
    last = {}
    for i in range(len(seq) - 1, -1, -1):
        nxt[i] = last.get(seq[i], math.inf)
        last[seq[i]] = i
    resd, heap, cur = set(), [], {}
    hits = 0
    for i, k in enumerate(seq):
        if k in resd:
            hits += 1
        elif len(resd) >= cap:
            # the resident with the furthest next use (lazy heap: skip stale entries)
            while heap and not (heap[0][1] in resd and cur.get(heap[0][1]) == -heap[0][0]):
                heapq.heappop(heap)
            if nxt[i] >= -heap[0][0]:
                continue  # bypass: k is needed later than every resident
            resd.discard(heapq.heappop(heap)[1])
            resd.add(k)
        else:
            resd.add(k)
        cur[k] = nxt[i]
        heapq.heappush(heap, (-nxt[i], k))
    return hits


def sim_belady(tr, geo, n_slots, pools=None, pool_slots=None):
    """optimal replacement with bypass (furthest next use, instant free uploads): the bound for the given slots. pools:
    lists of layer indices that share one cache of pool_slots[i] slots (the shape-class pools); default one per layer"""
    L, E = geo.L, geo.E
    if pools is None:
        pools = [[li] for li in range(L)]
        pool_slots = [int(n_slots[li]) for li in range(L)]
    pool_of = {}
    for pi, p in enumerate(pools):
        for li in p:
            pool_of[li] = pi
    seqs = [[] for _ in pools]
    for s in tr.steps:
        li = s // E
        for l, k in zip(li.tolist(), s.tolist()):
            seqs[pool_of[l]].append(k)
    hit_b = tot_b = 0.0
    for pi, p in enumerate(pools):
        seq = seqs[pi]
        bl = geo.bytes[p[0]]  # a pool holds one size class
        tot_b += bl * len(seq)
        hit_b += _belady_hits(seq, int(pool_slots[pi])) * bl
    return hit_b / tot_b if tot_b else 0.0, tot_b - hit_b


def pool_layout(geo, n_slots):
    """the shape-class pools of llama_moe_hot_init (LLAMA_MOE_HOT_POOL=1): k layers x (n + 1) - 1 slots, a class of one
    keeps its n slots"""
    pools = geo.classes()
    slots = [int(n_slots[p].sum()) + (len(p) - 1 if len(p) > 1 and n_slots[p].min() > 0 else 0) for p in pools]
    return pools, slots


# ---------------------------------------------------------------------------------------------------------------------
# grids

def grid_default():
    """the shipped default and the candidates of the l3 policy round"""
    g = collections.OrderedDict()
    g["ship"] = {}                                        # LLAMA_MOE_HOT_DECAY 0.92/2, admit 2, 1.2x/+0.5, 128 MiB/pass
    g["ship_seed03"] = {"seed": 0.03}                     # the seed as configured (the prefill stream hides it today)
    g["ship_seed01"] = {"seed": 0.01}
    g["ship_seed10"] = {"seed": 0.10}
    g["ship_seednorm8"] = {"seed": 1.0, "seed_norm": 8.0}
    g["ship_seednorm32"] = {"seed": 1.0, "seed_norm": 32.0}
    g["pool95_u256"] = {"pool": True, "decay": 0.95, "up_mib": 256.0}
    g["pool95_u256_seed03"] = {"pool": True, "decay": 0.95, "up_mib": 256.0, "seed": 0.03}
    g["pool95_u256_seednorm16"] = {"pool": True, "decay": 0.95, "up_mib": 256.0, "seed": 1.0, "seed_norm": 16.0}
    g["pool97_u256"] = {"pool": True, "decay": 0.97, "up_mib": 256.0}
    g["strata_tuned"] = {"admit": 2.0, "ratio": 1.0, "hyst": 1.5, "up_n": 192, "up_mib": 1e9}
    g["flashrt"] = {"every": 1, "admit": 1.0, "ratio": 1.2, "hyst": 0.0, "up_n": 64, "up_mib": 1e9, "seed": 0.03}
    g["ratio15"] = {"ratio": 1.5}
    g["hyst1"] = {"hyst": 1.0}
    g["admit1"] = {"admit": 1.0}
    g["decay090"] = {"decay": 0.90}
    g["decay095"] = {"decay": 0.95}
    g["decay097"] = {"decay": 0.97}
    g["every1"] = {"every": 1, "up_mib": 64.0}
    g["every4"] = {"every": 4, "up_mib": 256.0}
    g["up64"] = {"up_mib": 64.0}
    g["up256"] = {"up_mib": 256.0}
    g["up512"] = {"up_mib": 512.0}
    g["count_token"] = {"count": "token"}
    g["count_mix05"] = {"count": "mix", "token_w": 0.5}
    g["pool"] = {"pool": True}
    g["pool_up256"] = {"pool": True, "up_mib": 256.0}
    return g


def grid_real():
    """[TAG_FN_L3_POLICY_SIM] the arms for the real-use traces (prompts + answers): the shipped policy, the prompt seed
    (LLAMA_MOE_HOT_SEED_NODE, raw x seed or LLAMA_MOE_HOT_SEED_NORM), the pools, longer memory with fewer uploads, and
    the paced worker (LLAMA_MOE_HOT_UP_MIB_STEP)"""
    g = collections.OrderedDict()
    g["ship"] = {}
    for s in (0.01, 0.03, 0.10):
        g["seed%03d" % round(s * 100)] = {"seed": s}
    for n in (8.0, 16.0, 32.0, 64.0):
        g["seednorm%d" % n] = {"seed": 1.0, "seed_norm": n}
    g["pool"] = {"pool": True}
    g["pool_seed03"] = {"pool": True, "seed": 0.03}
    g["pool_seednorm16"] = {"pool": True, "seed": 1.0, "seed_norm": 16.0}
    for d in (0.95, 0.97, 0.98):
        g["d%d_r15h10" % round(d * 100)] = {"decay": d, "ratio": 1.5, "hyst": 1.0}
        g["pool_d%d_r15h10" % round(d * 100)] = {"pool": True, "decay": d, "ratio": 1.5, "hyst": 1.0}
    for d in (0.95, 0.97):
        g["pool_d%d_r15h10_seed03" % round(d * 100)] = {"pool": True, "decay": d, "ratio": 1.5, "hyst": 1.0, "seed": 0.03}
        g["pool_d%d_r15h10_seednorm16" % round(d * 100)] = {"pool": True, "decay": d, "ratio": 1.5, "hyst": 1.0,
                                                           "seed": 1.0, "seed_norm": 16.0}
    for d, bm, br in ((0.95, 256.0, 2.0), (0.97, 256.0, 2.0), (0.95, 512.0, 3.0)):
        g["pool_d%d_r15h10_b%d_r%g" % (round(d * 100), bm, br)] = {"pool": True, "decay": d, "ratio": 1.5, "hyst": 1.0,
                                                                  "burst_mib": bm, "burst_ratio": br}
        g["pool_d%d_r15h10_b%d_r%g_seed03" % (round(d * 100), bm, br)] = {"pool": True, "decay": d, "ratio": 1.5,
                                                                         "hyst": 1.0, "burst_mib": bm, "burst_ratio": br,
                                                                         "seed": 0.03}
    g["pool_d95_u256"] = {"pool": True, "decay": 0.95, "up_mib": 256.0}
    # Strata's long-memory decayed LFU: x0.92 per 2-step pass, admit over 1.2-1.5x the weakest resident, 64-192 uploads
    g["strata_tuned"] = {"admit": 2.0, "ratio": 1.0, "hyst": 1.5, "up_n": 192, "up_mib": 1e9}
    for n in (64, 128, 192):
        g["strata_r15_n%d" % n] = {"admit": 2.0, "ratio": 1.5, "hyst": 0.0, "up_n": n, "up_mib": 1e9}
    # counting: once per verify step (the union the step pays for, shipped), per token, or the union + 0.5 per extra token
    g["count_token"] = {"count": "token"}
    g["count_mix05"] = {"count": "mix", "token_w": 0.5}
    for bw in (32.0, 64.0, 96.0):
        g["pool_pace%d" % bw] = {"pool": True, "pace": True, "up_bw_mib": bw}
    return g


def grid_wide():
    """[TAG_FN_L3_POLICY_SIM] decay x every x upload budget x pool x ratio/hyst"""
    g = collections.OrderedDict()
    for pool in (False, True):
        for decay in (0.92, 0.95, 0.97, 0.98):
            for every in (2, 4):
                for up in (64.0, 128.0, 256.0):
                    for rh in ((1.2, 0.5), (1.5, 1.0)):
                        name = "%s_d%.2f_e%d_u%d_r%.1fh%.1f" % ("pool" if pool else "even", decay, every, up, rh[0], rh[1])
                        g[name] = {"pool": pool, "decay": decay, "every": every, "up_mib": up * every / 2.0,
                                   "ratio": rh[0], "hyst": rh[1]}
    return g


def eff(r, k):
    """DRAM-equivalent MiB per step: the CPU reads every cold byte once; an upload byte costs k (memcpy from the page cache
    into the pinned staging and the DMA read: 2-3 when it competes with the CPU job, ~0 when it runs in its gaps)"""
    return r["cold_mib_step"] + k * r["up_mib_step"]


def fmt(r, keys=("hit", "hit_warm64", "cold_mib_step", "up_mib_step")):
    out = []
    for k in keys:
        if k in r:
            out.append("%s %.4f" % (k, r[k]) if "hit" in k else "%s %.1f" % (k, r[k]))
    if "cold_mib_tok" in r:
        out.append("cold_mib_tok %.1f up_mib_tok %.1f tok/step %.2f" % (r["cold_mib_tok"], r["up_mib_tok"], r["tok_step"]))
    if "cold_mib_step" in r and "up_mib_step" in r:
        out.append("eff1 %.0f eff3 %.0f" % (eff(r, 1), eff(r, 3)))
    return "  ".join(out)


# ---------------------------------------------------------------------------------------------------------------------
# selftest

def _synthetic(n_layer=4, n_expert=64, n_steps=800, seed=1, shift_at=None):
    rng = np.random.default_rng(seed)
    tr = Trace("synthetic")
    tr.n_expert, tr.host = n_expert, list(range(n_layer))
    p = 1.0 / np.arange(1, n_expert + 1) ** 1.1
    p /= p.sum()
    perm = [rng.permutation(n_expert) for _ in range(n_layer)]
    tr.answers.append([0, 0, None])
    for i in range(n_steps):
        if shift_at and i == shift_at:
            perm = [rng.permutation(n_expert) for _ in range(n_layer)]
            pr = np.zeros(n_layer * n_expert, dtype=np.float32)
            for l in range(n_layer):
                ids = perm[l][rng.choice(n_expert, 400, p=p)]
                pr += np.bincount(l * n_expert + ids, minlength=n_layer * n_expert).astype(np.float32)
            tr.prompt_at[i] = pr
            tr.answers.append([i, 0, None])
        parts, toks = [], []
        T = 3
        for l in range(n_layer):
            ids = np.concatenate([perm[l][rng.choice(n_expert, 4, replace=False, p=p)] for _ in range(T)])
            parts.append(l * n_expert + np.unique(ids))
            toks.append(l * n_expert + ids)
        tr.steps.append(np.concatenate(parts).astype(np.int32))
        tr.tok.append(np.concatenate(toks).astype(np.int32))
        tr.T.append(T)
        tr.answers[-1][1] += 1
    return tr


def selftest():
    ok = True
    n_layer, n_expert = 4, 64
    types = {"layers": [{"layer": l, "gate": "q4_K", "up": "q4_K", "down": "q5_1", "expert_bytes": 1 << 20} for l in range(n_layer)]}
    geo = Geo(types, list(range(n_layer)), n_expert)
    tr = _synthetic(n_layer, n_expert, shift_at=400)
    budget = (16 + 1) * n_layer * (1 << 20)  # 16 slots per layer + the zero slots
    n_even = geo.even_slots(budget)
    ok &= bool((n_even == 16).all())
    base = {"up_mib": 16.0, "up_bw_mib": 64.0}
    r0 = sim_policy(tr, geo, budget, base)
    rs = sim_policy(tr, geo, budget, dict(base, seed=0.03))
    rp = sim_policy(tr, geo, budget, dict(base, pool=True))
    hb, _ = sim_belady(tr, geo, n_even)
    hbp, _ = sim_belady(tr, geo, n_even, *pool_layout(geo, n_even))
    hs, _ = sim_static(tr, geo, n_even, union_counts(tr))
    print("selftest: decay %.3f  +seed %.3f (warm %.3f vs %.3f)  pool %.3f  static-oracle %.3f  belady %.3f (pooled %.3f)" %
          (r0["hit"], rs["hit"], rs["hit_warm64"], r0["hit_warm64"], rp["hit"], hs, hb, hbp))
    ok &= 0.3 < r0["hit"] <= hb + 1e-9 and rs["hit"] <= hb + 1e-9 and hs <= hb + 1e-9
    ok &= rp["hit"] <= hbp + 1e-9 and hb <= hbp + 1e-9  # one pool of 4 x 17 - 1 slots holds what 4 x 16 hold
    ok &= rs["hit_warm64"] > r0["hit_warm64"]  # the seed warms a shifted working set faster
    ok &= r0["up_mib_step"] <= 64.0 + 1e-9  # the worker's rate limit holds
    # the even rule of llama_fn_even_slots: rows = budget / sum(bytes), n = rows - 1
    ok &= int(geo.even_slots(10 * n_layer * (1 << 20) + 5)[0]) == 9
    # var with the same skewed profile in every layer gives every layer the same count (zero slots included)
    nv = geo.var_slots(budget, np.tile(1.0 / np.arange(1, n_expert + 1), n_layer))
    ok &= int(nv.sum()) == 64 and bool((nv == 16).all())
    print("selftest: even %s var %s" % (n_even.tolist(), nv.tolist()))
    # persistence: a restart from the saved state is warmer than an empty start
    st = r0["_state"]
    rw = sim_policy(subtrace(tr, 1, 2), geo, budget, base, state=st)
    rc = sim_policy(subtrace(tr, 1, 2), geo, budget, base)
    ok &= rw["hit_warm64"] >= rc["hit_warm64"]
    print("selftest: restart warm64 saved %.3f vs empty %.3f -> %s" % (rw["hit_warm64"], rc["hit_warm64"], "OK" if ok else "FAIL"))
    return 0 if ok else 1


# ---------------------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", action="append", default=[], help="[name=]path.moet[:answers.json]")
    ap.add_argument("--types", default="E:/turbot-gates/flashnext/recipe/types_A.json")
    ap.add_argument("--host-layers", default="0-47")
    ap.add_argument("--budget-mib", default="10300,14300")
    ap.add_argument("--grid", default="default", help="default | wide | real | none")
    ap.add_argument("--top", type=int, default=0, help="print only the best N arms per trace and budget (by eff2), plus ship")
    ap.add_argument("--set", action="append", default=[], help="extra arm: name=key:val,key:val")
    ap.add_argument("--no-belady", action="store_true")
    ap.add_argument("--var-from", default="", help="trace name whose union counts size the layers (var alloc); 'self' = oracle")
    ap.add_argument("--prefill-as-decode", type=int, default=0, help="teacher-forced traces: prefill windows of k tokens as steps")
    ap.add_argument("--restart", action="store_true", help="persistence: replay every answer after a restart, empty vs saved state")
    ap.add_argument("--out", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.trace:
        raise SystemExit("--trace is required")
    with open(a.types, encoding="utf-8") as f:
        types = json.load(f)
    lo, hi = a.host_layers.split("-")
    host = list(range(int(lo), int(hi) + 1))
    traces = collections.OrderedDict()
    for spec in a.trace:
        name, path = spec.split("=", 1) if "=" in spec and not spec.startswith(("E:", "D:", "C:")) else (None, spec)
        t0 = time.time()
        tr = load_trace(path, host, a.prefill_as_decode)
        tr.name = name or tr.name
        traces[tr.name] = tr
        gen = [x[2] for x in tr.answers]
        print("trace %s: %d decode steps, %d answers (steps %s, generated %s), %d prompts, mean T %.2f, %.1f s" %
              (tr.name, len(tr.steps), len(tr.answers), [x[1] for x in tr.answers], gen, len(tr.prompt_at),
               float(np.mean(tr.T)) if tr.T else 0.0, time.time() - t0))
    any_tr = next(iter(traces.values()))
    geo = Geo(types, any_tr.host, any_tr.n_expert)
    budgets = [float(x) * 2**20 for x in a.budget_mib.split(",") if x]
    grid = {"default": grid_default, "wide": grid_wide, "real": grid_real}.get(a.grid, collections.OrderedDict)()
    for s in a.set:
        name, kv = s.split("=", 1)
        d = {}
        for item in kv.split(","):
            k, v = item.split(":", 1)
            d[k] = v if k in ("count", "alloc", "start", "fold") else (v == "1" if k in ("pool", "pace") else float(v))
        grid[name] = d
    summary = {"budgets_mib": [b / 2**20 for b in budgets], "traces": {}, "geo": {"bytes_sum": float(geo.bytes.sum())}}
    counts = {n: union_counts(t) for n, t in traces.items()}
    for name, tr in traces.items():
        dsum = summary["traces"][name] = {"steps": len(tr.steps), "answers": [x[1:] for x in tr.answers], "budgets": {}}
        for B in budgets:
            n_even = geo.even_slots(B)
            row = dsum["budgets"]["%d" % (B / 2**20)] = {"even_slots": int(n_even[0])}
            hs, _ = sim_static(tr, geo, n_even, counts[name])
            row["static_oracle"] = hs
            if not a.no_belady:
                row["belady_even"], _ = sim_belady(tr, geo, n_even)
                pl, ps = pool_layout(geo, n_even)
                row["belady_pool"], _ = sim_belady(tr, geo, n_even, pl, ps)
            line = "%-10s %6.0f MiB even %d slots: static-oracle %.4f%s" % (
                name, B / 2**20, n_even[0], hs, (" belady %.4f (pools %.4f)" % (row["belady_even"], row["belady_pool"]))
                if "belady_even" in row else "")
            print(line)
            for arm, cfg in grid.items():
                t0 = time.time()
                r = sim_policy(tr, geo, B, cfg)
                r.pop("_state")
                row[arm] = r
                if not a.top:
                    print("  %-14s %s  (%.1f s)" % (arm, fmt(r), time.time() - t0))
            if a.top:
                ship = sim_policy(tr, geo, B, {})
                ship.pop("_state")
                row["ship"] = ship
                arms = [k for k in grid if k in row]
                for kk in (0, 1, 3):
                    best = sorted(arms, key=lambda k: eff(row[k], kk))[:a.top]
                    print("  best by eff%d (ship %.0f, hit %.4f):" % (kk, eff(ship, kk), ship["hit"]))
                    for k in best:
                        print("    %-34s %s" % (k, fmt(row[k])))
            # variable per-layer slots: sized from another trace's counts (or the oracle)
            srcs = [n for n in traces if n != name] if a.var_from in ("", "others") else [a.var_from]
            for src in list(dict.fromkeys(srcs + ["self"])):
                prof = counts[name] if src == "self" else counts.get(src)
                if prof is None:
                    continue
                nv = geo.var_slots(B, prof)
                r = sim_policy(tr, geo, B, {"alloc": "var", "prof": prof})
                r.pop("_state")
                r["slots_min_max"] = [int(nv.min()), int(nv.max())]
                row["var_from_" + src] = r
                if not a.no_belady:
                    r["belady_var"], _ = sim_belady(tr, geo, nv)
                print("  %-14s %s  slots %d..%d%s" % ("var<-" + src, fmt(r), nv.min(), nv.max(),
                                                       (" belady %.4f" % r["belady_var"]) if "belady_var" in r else ""))
            if a.restart and len(tr.answers) > 1:
                # every answer after the first: started cold (empty, + the seed) vs from the state saved after the answers
                # before it (the restart persistence of LLAMA_MOE_HOT_STATE)
                res_cold, res_warm = [], []
                for ai in range(1, len(tr.answers)):
                    prev = sim_policy(subtrace(tr, 0, ai), geo, B, {})
                    t = subtrace(tr, ai, ai + 1)
                    rc = sim_policy(t, geo, B, {})
                    rw = sim_policy(t, geo, B, {}, state=prev["_state"])
                    res_cold.append((rc["hit"], rc["hit_warm64"]))
                    res_warm.append((rw["hit"], rw["hit_warm64"]))
                row["restart"] = {"cold": res_cold, "saved": res_warm}
                print("  restart: cold hit %s warm64 %s | saved hit %s warm64 %s" % (
                    [round(x[0], 3) for x in res_cold], [round(x[1], 3) for x in res_cold],
                    [round(x[0], 3) for x in res_warm], [round(x[1], 3) for x in res_warm]))
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        p = os.path.join(a.out, "policy_summary.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
        print("wrote %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
