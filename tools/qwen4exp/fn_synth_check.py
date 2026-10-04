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
#
# [TAG_FN_MERGE] placement band (default on, --no-band = the absolute rule above). The synthetic model has random weights:
# its logits are nearly uniform (PPL ~ 3e5 on a 248K vocabulary), so top-1 near-ties are everywhere and CPU-vs-GPU
# expert arithmetic alone moves same-top far below 99 % (r1_synth_check 2026-09-29: hot set KLD 1e-6, same-top 80-89 %).
# Each width therefore also runs a band arm, every expert on the GPU (--n-cpu-moe 0), against the same all-CPU base: the
# largest difference an exact path that moves experts between CPU and GPU arithmetic can show. A candidate passes with
# KLD <= min(1e-3, max(2 x band KLD, 2e-6)) (the log prints KLD to 1e-6) and same-top >= min(99, band same-top -
# max(2, 3 x SE)), SE = the combined standard error llama-perplexity prints for the two same-top values (~124 scored
# tokens per width, SE ~3.5 points). The KLD bound is the sharp one: a wrong, lost or double-counted expert moves KLD
# by orders of magnitude past the band.
#
# [TAG_MOE_BRIDGE] --mode bridge: the candidate runs with the MoE host bridge instead (LLAMA_MOE_BRIDGE=1, one device
# graph per ubatch, the experts on the CPU pool); --mode bridge-hot: bridge and hot set together. --wait hostfunc
# selects the host-function wait. The log must show the bridge and no bridge timeout.
from __future__ import annotations

import argparse
import math
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


def same_top(txt):
    """[TAG_FN_MERGE] (same-top %, its standard error %) from a llama-perplexity KLD log, or None."""
    m = re.search(r"Same top p:\s*([0-9.]+)\s*(?:\u00b1|\+/-|\+-)?\s*([0-9.]+)?", txt)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2)) if m.group(2) else 0.0


def band_ref(common, T, base, d, layers):
    """[TAG_FN_MERGE] (KLD, same-top, same-top SE) of every expert on the GPU (--n-cpu-moe 0) against the all-CPU
    base, or None."""
    argv = list(common)
    i = argv.index("--n-cpu-moe")
    argv[i + 1] = "0"
    rc, txt = run(argv + ["-ub", str(T), "--kl-divergence-base", base, "--kl-divergence"], {},
                  os.path.join(d, "band_ub%d.log" % T))
    kld = re.search(r"Mean\s+KLD:\s*([0-9.eE+-]+)", txt)
    top = same_top(txt)
    if rc != 0 or not kld or not top:
        return None
    return float(kld.group(1)), top[0], top[1]


def limits(band, max_kld=1e-3, min_top=99.0, cand_se=0.0):
    """[TAG_FN_MERGE] the pass limits (max KLD, min same-top) for a band, or the absolute ones without a band."""
    if band is None:
        return max_kld, min_top
    margin = max(2.0, 3.0 * math.hypot(band[2], cand_se))
    return min(max_kld, max(2.0 * band[0], 2e-6)), min(min_top, band[1] - margin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=fb.BIN)
    ap.add_argument("--dir", default="E:/synth/fnhot")
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--hot-mib", type=float, default=0, help="hot budget (default: half of the expert bytes)")
    ap.add_argument("--max-commit-gb", type=int, default=50)
    ap.add_argument("--mode", choices=["hot", "bridge", "bridge-hot"], default="hot")  # [TAG_MOE_BRIDGE]
    ap.add_argument("--wait", choices=["spin", "hostfunc"], default="spin")
    ap.add_argument("--no-band", action="store_true", help="[TAG_FN_MERGE] absolute rule (KLD <= 1e-3, same-top >= 99)")
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
        band = None
        if not a.no_band:  # [TAG_FN_MERGE]
            band = band_ref(common, T, base, a.dir, a.layers)
            if band is None:
                fb.log("T=%d band run failed" % T)
                ok_all = False
                continue
        env = {}
        if a.mode in ("hot", "bridge-hot"):
            env.update({"LLAMA_MOE_HOT_PROFILE": prof, "LLAMA_MOE_HOT_MIB": "%.1f" % hot_mib, "LLAMA_MOE_HOT_STATS": "1"})
        if a.mode in ("bridge", "bridge-hot"):  # [TAG_MOE_BRIDGE]
            env.update({"LLAMA_MOE_BRIDGE": "1", "LLAMA_MOE_BRIDGE_STATS": "1", "LLAMA_MOE_BRIDGE_WAIT": a.wait})
        rc, txt = run(common + ["-ub", str(T), "--kl-divergence-base", base, "--kl-divergence"], env,
                      os.path.join(a.dir, "%s_ub%d.log" % (a.mode, T)))
        kld = re.search(r"Mean\s+KLD:\s*([0-9.eE+-]+)", txt)
        top = re.search(r"Same top p:\s*([0-9.]+)", txt)
        hot = re.search(r"moe-hot: (\d+) layers, ([0-9.]+) MiB", txt) if a.mode != "bridge" else re.search(r"MoE bridge \d+ on \S+: (\d+) host expert layers", txt)
        if hot and a.mode == "bridge-hot" and not re.search(r"MoE bridge \d+ on \S+: (\d+) host expert layers", txt):
            hot = None
        fatal = re.search(r"CUDA error|illegal memory access|GGML_ASSERT|ggml_abort|wait timeout|host job failed", txt)
        st = same_top(txt)
        lim_kld, lim_top = limits(band, cand_se=st[1] if st else 0.0)
        ok = (rc == 0 and kld and top and hot and not fatal and float(kld.group(1)) <= lim_kld and
              float(top.group(1)) >= lim_top)
        ok_all = ok_all and bool(ok)
        print("T=%d  hot %s  KLD %s  same-top %s  (band %s; limits KLD <= %.3g, same-top >= %.2f)  -> %s" % (
              T, hot.group(0) if hot else "NOT ENABLED", kld.group(1) if kld else "-", top.group(1) if top else "-",
              "KLD %.3g same-top %.2f +- %.2f" % band if band else "off", lim_kld, lim_top, "PASS" if ok else "FAIL"))
    print("SYNTH %s CHECK %s" % (a.mode.upper(), "PASS" if ok_all else "FAIL"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
