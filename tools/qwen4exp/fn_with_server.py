#!/usr/bin/env python3
# [TAG_FN_MERGE] Run any client command against ONE Flash-Next llama-server of a given configuration, then stop it.
# Port 8091, own server only, same preflight as fn_bench.py (SIX_DONE, no STOP_GPU, ONE-process check, commit, port,
# nvlddmkm). For clients that expect a server already on 127.0.0.1:8091, e.g. the needle harness:
#
#   python tools/qwen4exp/fn_with_server.py --bin E:/turbot-gates/flashnext/r2/bin-r2 --base-args "-c 131072 ..." \
#          --arms E:/turbot-gates/flashnext/r2/arms/arms_hot.json --arm hot48ad --tag needles_hot -- \
#          python C:/.../scratchpad/niah.py 131072 20,50,80 3
#
# The server gets --base-args + the arm's "args", fn_bench.BASE_ENV + GGML_DISABLE_VULKAN=1 + the arm's "env" + --env K=V.
# Exit code: the client's, or 3 when the server log shows a fatal message (CUDA error, illegal access, GGML_ASSERT).
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fn_bench as fb  # noqa: E402

FATAL = r"CUDA error|illegal memory access|GGML_ASSERT|ggml_abort|unspecified launch failure"


def main():
    if "--" not in sys.argv:
        sys.exit("usage: fn_with_server.py [options] -- <client command ...>")
    cut = sys.argv.index("--")
    client = sys.argv[cut + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", default=fb.MODEL_A)
    ap.add_argument("--base-args", required=True, help="server flags without --host/--port/-m")
    ap.add_argument("--arms", default="", help="arms JSON ({name, args, env, bin, model}); with --arm")
    ap.add_argument("--arm", default="")
    ap.add_argument("--env", action="append", default=[], help="K=V for the server (repeatable)")
    ap.add_argument("--out", default="E:/turbot-gates/flashnext/r2/gpu")
    ap.add_argument("--tag", default="with_server")
    ap.add_argument("--timeout", type=int, default=6 * 3600, help="client timeout in seconds")
    ap.add_argument("--max-commit-gb", type=int, default=50)
    a = ap.parse_args(sys.argv[1:cut])
    if not client:
        sys.exit("no client command after --")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    errs = fb.preflight(a.max_commit_gb)
    if errs:
        for e in errs:
            fb.log("REFUSED: " + e)
        return 2
    os.makedirs(a.out, exist_ok=True)

    arm = {}
    if a.arms:
        with open(a.arms, encoding="utf-8") as f:
            arms = json.load(f)
        found = [x for x in arms if x.get("name") == a.arm]
        if not found:
            sys.exit("arm %r not in %s" % (a.arm, a.arms))
        arm = found[0]
    env = dict(fb.BASE_ENV)
    env["GGML_DISABLE_VULKAN"] = "1"
    env.update(arm.get("env", {}))
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v
    args = fb.split_args(a.base_args) + arm.get("args", [])
    log_path = os.path.join(a.out, "%s_server.log" % a.tag)
    srv = fb.Server(arm.get("bin", a.bin), arm.get("model", a.model), args, env, log_path)
    rc = 1
    try:
        fb.log("server ready after %.0f s (%s)" % (srv.start(), log_path))
        fb.log("client: " + " ".join(client))
        try:
            rc = subprocess.run(client, timeout=a.timeout).returncode
        except subprocess.TimeoutExpired:
            fb.log("client timed out after %d s" % a.timeout)
            rc = 124
    finally:
        srv.stop()
    fatal = re.search(FATAL, srv.text())
    if fatal:
        fb.log("FATAL in the server log: " + fatal.group(0))
        return 3
    fb.log("client rc %d" % rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
