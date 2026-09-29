#!/usr/bin/env python3
# [TAG_MOE_DMA_SHARE] [TAG_MOE_PREFETCH] [TAG_FN_PREFILL_STREAM] The gen5 paths on vs off on the synthetic qwen4exp model of
# fn_synth_check.py (no 100 GB model needed). GPU: same preflight as fn_bench.py (SIX_DONE, ONE-process check, commit).
#
#   python tools/qwen4exp/fn_gen5_check.py [--bin build-gen5/bin] [--dir E:/synth/fnhot]
#
# All experts are host-resident (--n-cpu-moe = layers). llama-perplexity writes a sparse base with every switch off, then
# each case runs against it:
#   prefill stream  -ub 64 (>= 32 tokens: the scheduler's op-offload path in the base), LLAMA_PREFILL_STREAM=1 with 1, 2
#                   and 3 banks, a 4 MiB chunk ring: the same kernels on the same weights, so KLD must be 0 and same-top
#                   100 %
#   DMA share       -ub 1..4 (the MTP verify widths), LLAMA_MOE_DMA_SHARE=1 / auto, LLAMA_MOE_PREFETCH=1, with and
#                   without the hot set: every expert runs on the CPU or the GPU, so KLD <= 1e-3 and same-top >= 99 %
# Each log must show the path enabled and no fatal message.
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fn_bench as fb  # noqa: E402
import fn_synth_check as fs  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=fb.BIN)
    ap.add_argument("--dir", default="E:/synth/fnhot")
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--layers", type=int, default=8)
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
        if os.path.exists(fs.SHARD1):
            cmd += ["--meta", fs.SHARD1]
        fb.log("writing the synthetic model: " + " ".join(cmd))
        if subprocess.run(cmd).returncode != 0:
            fb.log("synth_gguf.py failed")
            return 1
    hot_mib = (a.layers * a.experts * 3 * 2560 * 640 * 34 / 32 / 2**20) / 2
    common = [os.path.join(a.bin, "llama-perplexity.exe"), "-m", model, "-f", fs.CORPUS, "-c", "64", "-b", "128",
              "--chunks", "4", "-ngl", "99", "--n-cpu-moe", str(a.layers), "-fit", "off", "-fa", "on", "-t", "8"]
    fatal_re = re.compile(r"CUDA error|illegal memory access|GGML_ASSERT|ggml_abort|outlived")
    ok_all = True

    def case(tag, ub, env, want_log, max_kld, min_top):
        nonlocal ok_all
        base = os.path.join(a.dir, "gen5_base_ub%d.sparse" % ub)
        if not os.path.exists(base):
            rc, _ = fs.run(common + ["-ub", str(ub), "--kl-divergence-base", base], {"LLAMA_PPL_SPARSE_K": "64"},
                           os.path.join(a.dir, "gen5_base_ub%d.log" % ub))
            if rc != 0 or not os.path.exists(base):
                fb.log("ub=%d base run failed (rc %d)" % (ub, rc))
                ok_all = False
                return
        rc, txt = fs.run(common + ["-ub", str(ub), "--kl-divergence-base", base, "--kl-divergence"], env,
                         os.path.join(a.dir, "gen5_%s.log" % tag))
        kld = re.search(r"Mean\s+KLD:\s*([0-9.eE+-]+)", txt)
        top = re.search(r"Same top p:\s*([0-9.]+)", txt)
        on = re.search(want_log, txt)
        fatal = fatal_re.search(txt)
        ok = (rc == 0 and kld and top and on and not fatal and float(kld.group(1)) <= max_kld and
              float(top.group(1)) >= min_top)
        ok_all = ok_all and bool(ok)
        print("%-28s ub %d  %s  KLD %s  same-top %s  -> %s" % (tag, ub, "on" if on else "NOT ENABLED",
              kld.group(1) if kld else "-", top.group(1) if top else "-", "PASS" if ok else "FAIL"))

    for bufs in (1, 2, 3):
        case("pfs_bufs%d" % bufs, 64,
             {"LLAMA_PREFILL_STREAM": "1", "LLAMA_PREFILL_STREAM_BUFS": str(bufs), "LLAMA_PREFILL_STREAM_CHUNK_MIB": "4",
              "LLAMA_PREFILL_STREAM_RING_MIB": "16", "LLAMA_PREFILL_STREAM_STATS": "1"},
             r"prefill-stream: ubatch \d+", 1e-9, 100.0)
    dma = {"LLAMA_MOE_DMA_SHARE": "1", "LLAMA_MOE_PREFETCH": "1", "LLAMA_MOE_DMA_STATS": "1",
           "LLAMA_MOE_DMA_RING_MIB": "64", "LLAMA_MOE_DMA_ADMIT": "1/8"}
    for T in (1, 2, 3, 4):
        case("dma_prefetch_ub%d" % T, T, dma, r"moe-dma: \d+ host expert layers", 1e-3, 99.0)
        case("dma_auto_hot_ub%d" % T, T,
             dict(dma, LLAMA_MOE_DMA_SHARE="auto", LLAMA_MOE_HOT_PROFILE=prof, LLAMA_MOE_HOT_MIB="%.1f" % (hot_mib / 2)),
             r"moe-dma: \d+ host expert layers", 1e-3, 99.0)
    case("dma_sync_inline_ub3", 3, dict(dma, LLAMA_MOE_DMA_SYNC="1", LLAMA_MOE_DMA_INLINE="1"),
         r"moe-dma: \d+ host expert layers", 1e-3, 99.0)
    case("all_gen5_ub64", 64, dict(dma, LLAMA_PREFILL_STREAM="1"), r"prefill-stream: \d+ host expert layers", 1e-9, 100.0)
    print("GEN5 SYNTH CHECK %s" % ("PASS" if ok_all else "FAIL"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
