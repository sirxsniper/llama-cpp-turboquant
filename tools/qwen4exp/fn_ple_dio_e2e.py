#!/usr/bin/env python3
# [TAG_FN_PLE_DIRECT_IO] CPU-only end-to-end check of LLAMA_PLE_DIRECT_IO on the synthetic qwen4exp model: llama-perplexity
# with the PLE table read through the mapping (sparse KLD base) vs direct reads at ubatch 1, 4 and 64; LLAMA_PLE_DIO_FILE
# twice (the first load makes the copy, the second reuses it); a wrong copy of the right size (must be detected and made
# again); and LLAMA_PLE_DIO_TEST_FAIL=3 (every 3rd read fails and comes from the mapping). GPU hidden
# (CUDA_VISIBLE_DEVICES=-1, GGML_VK_VISIBLE_DEVICES=none, -dev none, -ngl 0). Pass: every chunk has the base PPL, KLD 0
# and same top 100 %, plus the expected log lines.
#
#   python tools/qwen4exp/fn_ple_dio_e2e.py [--beside-kld]
import os
import re
import subprocess
import sys

FORCE = "--beside-kld" in sys.argv  # the caller checked: only quality-only KLD runs are active (build_any.ps1 rule)
BIN = "D:/Projects/LocalAI/source-build/wt-fnple/build-ple/bin"
MODEL = "E:/synth/fnhot/q4x-hot.gguf"
CORPUS = "E:/kv-bar-s0/code_corpus.txt"
OUT = "E:/turbot-gates/flashnext/ple/e2e"
FATAL = re.compile(r"GGML_ASSERT|ggml_abort|exception|failed to load|warning: read of row|is read through the mapping|"
                   r"not usable|does not match|copy failed", re.I)


def busy():
    out = subprocess.run(["tasklist"], capture_output=True, text=True).stdout
    return [l.split()[0] for l in out.splitlines()
            if re.match(r"(llama|ggml|test-|compute-sanitizer|ninja|cmake|nvcc|cl\.exe|link\.exe)", l, re.I)]


def run(argv, env, log):
    e = dict(os.environ)
    e.update({"CUDA_VISIBLE_DEVICES": "-1", "GGML_VK_VISIBLE_DEVICES": "none", "LLAMA_PPL_SPARSE_K": "64"})
    e.update(env)
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        f.write(" ".join(argv) + "\n" + repr(env) + "\n")
        f.flush()
        r = subprocess.run(argv, env=e, stdout=f, stderr=subprocess.STDOUT)
    with open(log, encoding="utf-8", errors="replace") as f:
        return r.returncode, f.read()


def kld_rows(txt):
    # chunk  PPL ± e  ln-ratio ± e  KLD ± e  dp ± e %  same ± e %
    rows = []
    for line in txt.splitlines():
        f = line.split()
        if len(f) >= 17 and f[0].isdigit() and f[2] == "±":
            rows.append((f[1], float(f[7]), float(f[14])))
    return rows


def check(txt, ppl_base):
    rows = kld_rows(txt)
    ok = bool(rows) and [r[0] for r in rows] == ppl_base and all(r[1] == 0.0 and r[2] == 100.0 for r in rows)
    return ok, rows


def main():
    b = busy()
    if b and not FORCE:
        print("REFUSED: busy: " + ", ".join(b))
        return 2
    os.makedirs(OUT, exist_ok=True)
    common = [os.path.join(BIN, "llama-perplexity.exe"), "-m", MODEL, "-f", CORPUS, "-c", "64", "-b", "64",
              "--chunks", "3", "-dev", "none", "-ngl", "0", "-fit", "off", "-t", "8", "-lv", "4"]
    ok_all = True
    ppl_base = {}
    for T in (1, 4, 64):
        base = os.path.join(OUT, "base_ub%d.sparse" % T)
        if os.path.exists(base):
            os.remove(base)
        rc0, t0 = run(common + ["-ub", str(T), "--kl-divergence-base", base], {}, os.path.join(OUT, "map_ub%d.log" % T))
        rc1, t1 = run(common + ["-ub", str(T), "--kl-divergence-base", base, "--kl-divergence"],
                      {"LLAMA_PLE_DIRECT_IO": "1", "LLAMA_PLE_DIO_STATS": "1"}, os.path.join(OUT, "dio_ub%d.log" % T))
        ppl_base[T] = re.findall(r"\[\d+\]([0-9.]+)", t0)
        same, rows = check(t1, ppl_base[T])
        loader = "rows read from the file, never mapped in" in t1 and "rows read from the file" not in t0
        opened = "per_layer_token_embd.weight: rows read unbuffered from " + MODEL in t1
        stats = re.search(r"\[TAG_FN_PLE_DIRECT_IO\] final: (.*)", t1)
        fatal = FATAL.search(t1)
        ok = rc0 == 0 and rc1 == 0 and same and loader and opened and stats and not fatal
        ok_all = ok_all and bool(ok)
        print("ub=%-3d rc %d/%d  PPL base %s  dio rows %s  loader %s  open %s  %s -> %s" % (
            T, rc0, rc1, ",".join(ppl_base[T]), ["%s/%g/%g" % r for r in rows], loader, opened,
            ("fatal: " + fatal.group(0)) if fatal else "", "PASS" if ok else "FAIL"))
        if stats:
            print("        stats: " + stats.group(1))

    # LLAMA_PLE_DIO_FILE: the first load makes the copy, the second reuses it
    T = 4
    base = os.path.join(OUT, "base_ub%d.sparse" % T)
    copy = OUT + "/synth-ple.bin"
    for f in (copy, copy + ".tmp"):
        if os.path.exists(f):
            os.remove(f)
    for run_i, want_copy in ((0, True), (1, False)):
        rc, t = run(common + ["-ub", str(T), "--kl-divergence-base", base, "--kl-divergence"],
                    {"LLAMA_PLE_DIRECT_IO": "1", "LLAMA_PLE_DIO_STATS": "1", "LLAMA_PLE_DIO_FILE": copy},
                    os.path.join(OUT, "dio_copy%d_ub%d.log" % (run_i, T)))
        same, rows = check(t, ppl_base[T])
        copied = "copying the PLE table" in t
        opened = "per_layer_token_embd.weight: rows read unbuffered from " + copy in t
        fatal = FATAL.search(t)
        ok = rc == 0 and same and copied == want_copy and opened and not fatal
        ok_all = ok_all and bool(ok)
        print("copy run %d ub=%d rc %d  rows %s  copied %s (want %s)  reads the copy %s  %s -> %s" % (
            run_i, T, rc, ["%s/%g/%g" % r for r in rows], copied, want_copy, opened,
            ("fatal: " + fatal.group(0)) if fatal else "", "PASS" if ok else "FAIL"))
    # a wrong copy of the right size (every row shifted by one) must be detected and made again
    if os.path.exists(copy):
        with open(copy, "rb") as f:
            data = f.read()
        rb = 90  # any shift that is not a multiple of the row size makes every row differ
        with open(copy, "wb") as f:
            f.write(data[rb:] + data[:rb])
        del data
        rc, t = run(common + ["-ub", str(T), "--kl-divergence-base", base, "--kl-divergence"],
                    {"LLAMA_PLE_DIRECT_IO": "1", "LLAMA_PLE_DIO_STATS": "1", "LLAMA_PLE_DIO_FILE": copy},
                    os.path.join(OUT, "dio_copybad_ub%d.log" % T))
        same, rows = check(t, ppl_base[T])
        detected = "does not match the table in the model file" in t and "copying the PLE table" in t
        opened = "per_layer_token_embd.weight: rows read unbuffered from " + copy in t
        fatal = re.search(r"GGML_ASSERT|ggml_abort|exception|failed to load|warning: read of row|copy failed", t, re.I)
        ok = rc == 0 and same and detected and opened and not fatal
        ok_all = ok_all and bool(ok)
        print("wrong copy ub=%d rc %d  rows %s  detected + made again %s  reads the copy %s  %s -> %s" % (
            T, rc, ["%s/%g/%g" % r for r in rows], detected, opened,
            ("fatal: " + fatal.group(0)) if fatal else "", "PASS" if ok else "FAIL"))
    else:
        ok_all = False
        print("wrong copy: FAIL (no copy to corrupt)")
    if os.path.exists(copy):
        os.remove(copy)

    # failed reads (every 3rd, injected) come from the mapped table: the values must not change
    for T in (1, 64):
        base = os.path.join(OUT, "base_ub%d.sparse" % T)
        rc, t = run(common + ["-ub", str(T), "--kl-divergence-base", base, "--kl-divergence"],
                    {"LLAMA_PLE_DIRECT_IO": "1", "LLAMA_PLE_DIO_STATS": "1", "LLAMA_PLE_DIO_TEST_FAIL": "3"},
                    os.path.join(OUT, "dio_fail3_ub%d.log" % T))
        same, rows = check(t, ppl_base[T])
        warned = "warning: read of row" in t and "TEST: injected read failures" in t
        stats = re.search(r"\[TAG_FN_PLE_DIRECT_IO\] final: (.*)", t)
        fb = re.search(r"(\d+) fallbacks", stats.group(1)) if stats else None
        fatal = re.search(r"GGML_ASSERT|ggml_abort|exception|failed to load|is read through the mapping", t, re.I)
        ok = rc == 0 and same and warned and fb and int(fb.group(1)) > 0 and not fatal
        ok_all = ok_all and bool(ok)
        print("fail 1/3 ub=%-3d rc %d  rows %s  warned %s  fallbacks %s  %s -> %s" % (
            T, rc, ["%s/%g/%g" % r for r in rows], warned, fb.group(1) if fb else None,
            ("fatal: " + fatal.group(0)) if fatal else "", "PASS" if ok else "FAIL"))
    print("PLE DIO E2E %s" % ("PASS" if ok_all else "FAIL"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
