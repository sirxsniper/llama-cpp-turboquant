#!/usr/bin/env python3
# [TAG_FN_L3_MTP_HEADIDS] Ranked draft vocabulary for LLAMA_MTP_HEAD_IDS (qwen4exp MTP) from real outputs.
#
# Rank: 1) ids the model wrote in real answers (token ids saved by the test harness, e.g. test/r2/eos/e*.jsonl), most
# frequent first; 2) ids of tokenized prompt text (e.g. test/r2/slots2/fn2_d245760.json), most frequent first; 3) the
# rest by id (BPE merge order). Control, user-defined and unused tokens are left out (the loader adds the control and
# user-defined ones itself). Writes the first N ids of the rank for each --n, one id per line with a '#' header.
#
# Coverage check: the answers are split in two folds by answer; each fold is scored against a rank built without it.
# Printed per N: the share of held-out output tokens inside the set, alone and with the prompt's own tokens added
# (LLAMA_MTP_HEAD_PROMPT), next to LLAMA_MTP_HEAD_ROWS=98304 (ids below 98304).
#
#   python tools/qwen4exp/mtp_head_rank.py --vocab <shard 1 of the model> --answers e1.jsonl e4.jsonl \
#       --prompt-ids fn2_d245760.json --n 32768 40960 --out-dir E:/turbot-gates/flashnext/test/l3/mtp
# CPU only; reads the GGUF header (vocabulary) and the json files.
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from merge_mtp import Header  # noqa: E402

TT_NORMAL, TT_UNKNOWN, TT_CONTROL, TT_USER, TT_UNUSED, TT_BYTE = 1, 2, 3, 4, 5, 6


def load_answers(paths):
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                toks = r.get("tokens")
                if isinstance(toks, list) and toks:
                    out.append([int(t) for t in toks])
    return out


def rank(out_cnt, prm_cnt, n_vocab, skip):
    seen = set()
    order = []
    for t, _c in sorted(out_cnt.items(), key=lambda kv: (-kv[1], -prm_cnt.get(kv[0], 0), kv[0])):
        if 0 <= t < n_vocab and t not in skip:
            order.append(t)
            seen.add(t)
    for t, _c in sorted(prm_cnt.items(), key=lambda kv: (-kv[1], kv[0])):
        if 0 <= t < n_vocab and t not in skip and t not in seen:
            order.append(t)
            seen.add(t)
    for t in range(n_vocab):
        if t not in skip and t not in seen:
            order.append(t)
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", required=True, help="a GGUF with the tokenizer (shard 1 of the model)")
    ap.add_argument("--answers", nargs="+", required=True)
    ap.add_argument("--prompt-ids", nargs="*", default=[])
    ap.add_argument("--n", nargs="+", type=int, default=[32768, 40960])
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--rows", type=int, default=98304, help="the LLAMA_MTP_HEAD_ROWS value to compare with")
    a = ap.parse_args()

    h = Header(a.vocab)
    # arrays come back as (element type, values)
    toks = (h.get("tokenizer.ggml.tokens") or (0, []))[1]
    types = (h.get("tokenizer.ggml.token_type") or (0, []))[1]
    n_vocab = len(toks)
    if n_vocab < 1000 or len(types) != n_vocab:
        raise SystemExit("%s: no usable tokenizer arrays (%d tokens, %d types)" % (a.vocab, n_vocab, len(types)))
    skip = {i for i, t in enumerate(types) if t in (TT_CONTROL, TT_USER, TT_UNUSED)}
    print("vocabulary: %d tokens, %d control / user-defined / unused left out of the rank" % (n_vocab, len(skip)))

    answers = load_answers(a.answers)
    prm = collections.Counter()
    for p in a.prompt_ids:
        with open(p, encoding="utf-8") as f:
            prm.update(int(t) for t in json.load(f))
    n_out = sum(len(x) for x in answers)
    print("answers: %d (%d tokens, %d distinct); prompt ids: %d (%d distinct)" % (len(answers), n_out,
          len({t for x in answers for t in x}), sum(prm.values()), len(prm)))

    # held-out coverage, 2 folds by answer
    ns = sorted(set(a.n + [16384, 24576, 32768, 40960, 49152, 65536]))
    cov = {n: [0, 0] for n in ns}
    cov_p = {n: [0, 0] for n in ns}
    cov_rows = [0, 0]
    prm_set = set(prm)
    for fold in (0, 1):
        train = collections.Counter(t for i, x in enumerate(answers) if i % 2 != fold for t in x)
        test = [t for i, x in enumerate(answers) if i % 2 == fold for t in x]
        order = rank(train, prm, n_vocab, skip)
        pos = {t: k for k, t in enumerate(order)}
        for t in test:
            k = pos.get(t, -1)  # -1: a skipped (control) id - the loader keeps those
            for n in ns:
                inside = k < 0 or k < n
                cov[n][0] += inside
                cov[n][1] += 1
                cov_p[n][0] += inside or t in prm_set
                cov_p[n][1] += 1
            cov_rows[0] += t < a.rows or t in skip
            cov_rows[1] += 1
    print("held-out output tokens inside the draft vocabulary (2 folds):")
    print("  LLAMA_MTP_HEAD_ROWS=%d (ids < %d + control): %.4f%%" % (a.rows, a.rows, 100.0 * cov_rows[0] / max(1, cov_rows[1])))
    for n in ns:
        print("  ranked top %6d: %.4f%%   + the prompt's tokens: %.4f%%" % (n, 100.0 * cov[n][0] / max(1, cov[n][1]),
                                                                          100.0 * cov_p[n][0] / max(1, cov_p[n][1])))

    if a.out_dir:
        full = rank(collections.Counter(t for x in answers for t in x), prm, n_vocab, skip)
        os.makedirs(a.out_dir, exist_ok=True)
        for n in a.n:
            ids = sorted(full[:n])
            path = os.path.join(a.out_dir, "fn_mtp_head_ids_%dk.txt" % (n // 1024))
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write("# [TAG_FN_L3_MTP_HEADIDS] qwen4exp MTP draft vocabulary: top %d of a rank by real output frequency "
                        "(%d answer tokens), then prompt-text frequency (%d tokens), then id; control / user-defined / "
                        "unused ids excluded (the loader adds control and user-defined ids). Sources: %s %s\n"
                        % (n, n_out, sum(prm.values()), " ".join(a.answers), " ".join(a.prompt_ids)))
                for t in ids:
                    f.write("%d\n" % t)
            print("wrote %s (%d ids, max id %d, %d ids >= %d)" % (path, len(ids), ids[-1], sum(1 for t in ids if t >= a.rows), a.rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
