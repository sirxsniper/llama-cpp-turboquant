#!/usr/bin/env python3
# [TAG_MOE_BRIDGE] [TAG_FN_MERGE] Stall and recovery test of the MoE host bridge through a real llama-server (port 8091,
# own server only, same preflight as fn_bench.py). E:/turbot-gates/flashnext/r2/TEST_PLAN.md step R1.
#
#   python tools/qwen4exp/fn_bridge_stall.py --bin <bin> [--model <gguf>] --base-args "..." [--env K=V ...]
#                                            [--stall-ms 100] [--every 0] [--requests 6] [--n 256]
#
# The server runs with LLAMA_MOE_BRIDGE=1, LLAMA_MOE_BRIDGE_STATS=1 and LLAMA_MOE_BRIDGE_TEST_STALL=<stall-ms> (plus
# LLAMA_MOE_BRIDGE_TEST_STALL_EVERY=<every> when set): the host executor sleeps after job 199 (and every <every> jobs
# after it), so the device wait of that job times out. MTP flags are removed (--spec-*), so every request is plain
# greedy decode of the same prompt and the texts can be compared.
#
# Pass, single stall (--every 0):
#   - exactly 1 "wait timeout" (or "host job failed") line and at least 1 "re-armed" line in the server log
#   - exactly 1 request fails with a clean error; the request right after it succeeds (its first ~15 steps run while
#     the bridge pauses, on the CPU chain, whose summation order may flip a greedy near-tie, so its text is not
#     compared); every other request returns the same text (sha1). Needs --requests >= 4.
#   - no CUDA error / illegal memory access / GGML_ASSERT, the server is still healthy after the last request
# Pass, repeated stalls (--every N):
#   - exactly 3 error lines, then "3 errors, the bridge stays off for this context"; 1..3 requests fail
#   - the requests after the bridge turned off succeed (their text may differ from the bridged text by the CPU-chain
#     summation order, so only success is checked), no fatal message, the server healthy at the end
# The GPU-side watchdog (nvlddmkm / TDR) is checked by gpucorr_run.sh's post step, not here.
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fn_bench as fb  # noqa: E402

PROMPT = ("Write a detailed, well structured essay about the history of lighthouses: their construction, the lenses, "
          "the keepers and how automation changed the job.")
FATAL = r"CUDA error|illegal memory access|GGML_ASSERT|ggml_abort|unspecified launch failure"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", default=fb.MODEL_A)
    ap.add_argument("--base-args", required=True, help="server flags without --host/--port/-m")
    ap.add_argument("--env", action="append", default=[], help="K=V for the server (repeatable)")
    ap.add_argument("--stall-ms", type=int, default=100)
    ap.add_argument("--every", type=int, default=0, help="also stall every N jobs after the first (0 = once)")
    ap.add_argument("--requests", type=int, default=6, help=">= 4 for the single-stall text check")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--out", default="E:/turbot-gates/flashnext/r2/gpu")
    ap.add_argument("--tag", default="bridge_stall")
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
    env.update({"GGML_DISABLE_VULKAN": "1", "LLAMA_MOE_BRIDGE": "1", "LLAMA_MOE_BRIDGE_STATS": "1",
                "LLAMA_MOE_BRIDGE_TEST_STALL": str(a.stall_ms)})
    if a.every > 0:
        env["LLAMA_MOE_BRIDGE_TEST_STALL_EVERY"] = str(a.every)
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v
    args = fb.strip_spec(fb.split_args(a.base_args))
    log_path = os.path.join(a.out, "%s_server.log" % a.tag)
    srv = fb.Server(a.bin, a.model, args, env, log_path)
    results = []
    try:
        fb.log("load %.0f s" % srv.start())
        prompt = fb.chat_prompt(PROMPT)
        for i in range(a.requests):
            try:
                r = fb.complete(prompt, a.n, greedy=True)
                sha = hashlib.sha1(r["content"].encode("utf-8", "replace")).hexdigest()[:12]
                results.append({"i": i, "ok": r["n"] > 0, "sha": sha, "n": r["n"], "tps": r["tps"]})
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")[:200]
                results.append({"i": i, "ok": False, "err": "HTTP %d %s" % (e.code, body)})
            except (urllib.error.URLError, OSError) as e:
                results.append({"i": i, "ok": False, "err": repr(e)[:200]})
            fb.log("request %d: %s" % (i, results[-1]))
        healthy = True
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/health" % fb.PORT, timeout=10) as r:
                healthy = r.status == 200
        except Exception:  # noqa: BLE001
            healthy = False
    finally:
        srv.stop()

    txt = srv.text()
    n_err_lines = len(re.findall(r"MoE bridge \d+: (?:wait timeout|host job failed) at layer", txt))
    n_rearm = len(re.findall(r"MoE bridge \d+ re-armed", txt))
    off = bool(re.search(r"3 errors, the bridge stays off", txt))
    enabled = bool(re.search(r"MoE bridge \d+ on \S+: \d+ host expert layers", txt))
    fatal = re.search(FATAL, txt)
    failed = [r for r in results if not r["ok"]]
    good = [r for r in results if r["ok"]]
    shas = {r["sha"] for r in good}
    # [TAG_MOE_BRIDGE] the request after the failed one runs its first steps while the bridge pauses (CPU chain): only
    # its success counts; the fully bridged requests before and after it must give one text
    i_fail = failed[0]["i"] if len(failed) == 1 else -1
    after = next((r for r in results if r["i"] == i_fail + 1), None) if i_fail >= 0 else None
    bridged = [r for r in good if i_fail >= 0 and r["i"] != i_fail + 1]
    shas_bridged = {r["sha"] for r in bridged}

    checks = [("bridge enabled", enabled), ("no fatal message", not fatal), ("server healthy at the end", healthy)]
    if a.every <= 0:
        checks += [("exactly 1 stall error line", n_err_lines == 1), ("re-armed after it", n_rearm >= 1),
                   ("exactly 1 failed request", len(failed) == 1),
                   ("the request after it succeeds", after is not None and after["ok"]),
                   ("the bridged requests the same text", len(bridged) >= 2 and len(shas_bridged) == 1)]
    else:
        checks += [("exactly 3 stall error lines", n_err_lines == 3), ("bridge off after 3 errors", off),
                   ("1..3 failed requests", 1 <= len(failed) <= 3),
                   ("the last request succeeds", bool(results) and results[-1]["ok"])]
    ok = all(c[1] for c in checks)
    for name, c in checks:
        print("  %-36s %s" % (name, "ok" if c else "FAIL"))
    print("stall lines %d, re-armed %d, off %s, failed requests %d, distinct texts %d (bridged %d)%s" % (
        n_err_lines, n_rearm, off, len(failed), len(shas), len(shas_bridged), ("  fatal: " + fatal.group(0)) if fatal else ""))
    print("BRIDGE STALL TEST %s (log %s)" % ("PASS" if ok else "FAIL", log_path))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
