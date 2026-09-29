#!/usr/bin/env python3
# [TAG_FN_MOE_HOT] Hot set on vs off on a synthetic qwen4exp model (no 100 GB model needed). GPU: same preflight as
# fn_bench.py (SIX_DONE, ONE-process check, commit, no nvlddmkm events).
#
#   python tools/qwen4exp/fn_synth_check.py [--bin build-fn/bin] [--dir E:/synth/fnhot]
#
# 1. writes E:/synth/fnhot/q4x-hot.gguf with synth_gguf.py (8 layers, 64 experts, the real tokenizer from shard 1 of the
#    downloaded split, no MTP) and a Zipf-skewed random profile, if they are missing (CPU and disk only)
# 2. for each ubatch width T = 1, 2, 3, 4 (the MTP verify widths) and 2 sequences per batch: llama-perplexity with all
#    experts on the CPU (--n-cpu-moe 8) writes a sparse base with the hot set OFF, then runs again with it ON
#    (LLAMA_MOE_HOT_PROFILE, half of the expert bytes resident) against that base
# 3. pass: mean KLD <= 1e-3, same top >= 99 %, the log shows the hot set and its hit statistics, no fatal message
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fn_bench as fb  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SHARD1 = fb.M + "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
CORPUS = "E:/kv-bar-s0/code_corpus.txt"


def run(argv, env, log):
    e = dict(os.environ)
    e.update(env)
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        f.write(" ".join(argv) + "\n" + repr(env) + "\n")
        f.flush()
        r = subprocess.run(argv, env=e, stdout=f, stderr=subprocess.STDOUT)
    with open(log, encoding="utf-8", errors="replace") as f:
        return r.returncode, f.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=fb.BIN)
    ap.add_argument("--dir", default="E:/synth/fnhot")
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--hot-mib", type=float, default=0, help="hot budget (default: half of the expert bytes)")
    ap.add_argument("--max-commit-gb", type=int, default=50)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    errs = fb.preflight(a.max_commit_gb)
    if errs:
        for e in errs:
            fb.log("REFUSED: " + e)
        return 2
    os.makedirs(a.dir, exist_ok=True)
    model = os.path.join(a.dir, "q4x-hot.gguf")
    prof = os.path.join(a.dir, "q4x-hot.moeprof")
    if not (os.path.exists(model) and os.path.exists(prof)):
        cmd = [sys.executable, os.path.join(HERE, "synth_gguf.py"), model, "--layers", str(a.layers), "--experts",
               str(a.experts), "--no-mtp", "--hot-profile-out", prof]
        if os.path.exists(SHARD1):
            cmd += ["--meta", SHARD1]
        fb.log("writing the synthetic model: " + " ".join(cmd))
        if subprocess.run(cmd).returncode != 0:
            fb.log("synth_gguf.py failed")
            return 1

    # expert bytes per layer: 3 matrices of n_embd x 640 at q8_0 (synth default) x n_expert
    hot_mib = a.hot_mib or (a.layers * a.experts * 3 * 2560 * 640 * 34 / 32 / 2**20) / 2
    common = [os.path.join(a.bin, "llama-perplexity.exe"), "-m", model, "-f", CORPUS, "-c", "64", "-b", "128",
              "--chunks", "4", "-ngl", "99", "--n-cpu-moe", str(a.layers), "-fit", "off", "-fa", "on", "-t", "8"]
    ok_all = True
    for T in (1, 2, 3, 4):
        base = os.path.join(a.dir, "base_ub%d.sparse" % T)
        if os.path.exists(base):
            os.remove(base)
        rc, txt = run(common + ["-ub", str(T), "--kl-divergence-base", base], {"LLAMA_PPL_SPARSE_K": "64"},
                      os.path.join(a.dir, "base_ub%d.log" % T))
        if rc != 0 or not os.path.exists(base):
            fb.log("T=%d base run failed (rc %d)" % (T, rc))
            ok_all = False
            continue
        env = {"LLAMA_MOE_HOT_PROFILE": prof, "LLAMA_MOE_HOT_MIB": "%.1f" % hot_mib, "LLAMA_MOE_HOT_STATS": "1"}
        rc, txt = run(common + ["-ub", str(T), "--kl-divergence-base", base, "--kl-divergence"], env,
                      os.path.join(a.dir, "hot_ub%d.log" % T))
        kld = re.search(r"Mean\s+KLD:\s*([0-9.eE+-]+)", txt)
        top = re.search(r"Same top p:\s*([0-9.]+)", txt)
        hot = re.search(r"moe-hot: (\d+) layers, ([0-9.]+) MiB", txt)
        fatal = re.search(r"CUDA error|illegal memory access|GGML_ASSERT|ggml_abort", txt)
        ok = rc == 0 and kld and top and hot and not fatal and float(kld.group(1)) <= 1e-3 and float(top.group(1)) >= 99.0
        ok_all = ok_all and bool(ok)
        print("T=%d  hot %s  KLD %s  same-top %s  -> %s" % (T, hot.group(0) if hot else "NOT ENABLED",
              kld.group(1) if kld else "-", top.group(1) if top else "-", "PASS" if ok else "FAIL"))
    print("SYNTH HOT CHECK %s" % ("PASS" if ok_all else "FAIL"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
