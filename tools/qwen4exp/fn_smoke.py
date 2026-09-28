#!/usr/bin/env python3
# [TAG_FN_SP0] Flash-Next smoke test on port 8091 (E:/turbot-gates/flashnext/PLAN.md 7.2 / F0). Same preflight as fn_bench.py.
#
#   python tools/qwen4exp/fn_smoke.py [--bin ...] [--model ...] [--base-args "..."] [--env K=V ...]
#
# Checks, in one server run each:
#   1. number sequence: greedy continuation of "1, 2, ..., 40," must contain "41, 42, 43"
#   2. needle at ~16K tokens: a passphrase at 40% depth must be recalled
#   3. MTP acceptance on greedy code (reported; below 0.80 is flagged, the F0 gate is 0.90 on the full bench)
#   4. TURBO_NAN_SCAN=1 short run: no "[NAN]" line in the server log
# Exit 0 only when every check passes.
from __future__ import annotations

import argparse
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fn_bench as fb  # noqa: E402

FILLER = ("The river valley was quiet in the late afternoon. Farmers moved slowly between the rows, checking the soil "
          "and the young plants. A dog barked somewhere near the old mill, and the wind carried the smell of cut hay. ")


def needle_prompt(n_words, depth, secret):
    words = (FILLER * (n_words // len(FILLER.split()) + 2)).split()[:n_words]
    k = int(len(words) * depth)
    words.insert(k, "\nThe secret passphrase for the archive is %s.\n" % secret)
    return ("Read the following notes carefully.\n\n" + " ".join(words) +
            "\n\nQuestion: what is the secret passphrase for the archive? Answer with the passphrase only.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=fb.MODEL_A)
    ap.add_argument("--bin", default=fb.BIN)
    ap.add_argument("--base-args", default=fb.BASE_131K)
    ap.add_argument("--env", action="append", default=[], help="K=V passed to the server (repeatable)")
    ap.add_argument("--out", default=fb.ROOT + "/flashnext/smoke")
    ap.add_argument("--max-commit-gb", type=int, default=50)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    errs = fb.preflight(a.max_commit_gb)
    if errs:
        for e in errs:
            fb.log("REFUSED: " + e)
        return 2
    os.makedirs(a.out, exist_ok=True)
    env = dict(fb.BASE_ENV)
    env.update(dict(kv.split("=", 1) for kv in a.env))
    args = fb.split_args(a.base_args)
    results = {}

    srv = fb.Server(a.bin, a.model, args, env, os.path.join(a.out, "smoke_main.log"))
    try:
        fb.log("load %.0f s" % srv.start())
        r = fb.complete(", ".join(str(i) for i in range(1, 41)) + ",", 24, greedy=True)
        results["sequence"] = "41, 42, 43" in r["content"].replace("\n", " ")
        fb.log("sequence: %r -> %s" % (r["content"][:60], results["sequence"]))

        secret = "amber-%d" % random.Random(7).randint(1000, 9999)
        r = fb.complete(fb.chat_prompt(needle_prompt(11500, 0.4, secret)), 32, greedy=True)
        results["needle16k"] = secret in r["content"]
        fb.log("needle ~16K (prompt %d tokens, %.0f t/s prefill): %r -> %s" % (
            r["prompt_n"], r["prompt_n"] / max(r["prompt_ms"], 1) * 1000, r["content"][:60], results["needle16k"]))

        r = fb.complete(fb.chat_prompt(fb.CODE_PROMPT), 256, greedy=True)
        acc = r["accept"]
        results["mtp_accept"] = acc is None or acc >= 0.80
        fb.log("code greedy %.2f t/s, draft acceptance %s" % (r["tps"], acc))
    finally:
        srv.stop()
    txt = srv.text()
    results["no_fatal"] = not re.search(r"CUDA error|illegal memory access|GGML_ASSERT|ggml_abort", txt)

    env_nan = dict(env)
    env_nan["TURBO_NAN_SCAN"] = "1"
    srv = fb.Server(a.bin, a.model, args, env_nan, os.path.join(a.out, "smoke_nanscan.log"))
    try:
        srv.start()
        fb.complete(fb.chat_prompt("Name three prime numbers above 100."), 24, greedy=True)
    finally:
        srv.stop()
    nan_lines = re.findall(r"\[NAN\][^\n]*", srv.text())
    results["nan_scan"] = not nan_lines
    fb.log("nan scan: %s" % (nan_lines[0] if nan_lines else "clean"))

    ok = all(results.values())
    for k, v in results.items():
        print("%-12s %s" % (k, "PASS" if v else "FAIL"))
    print("SMOKE %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
