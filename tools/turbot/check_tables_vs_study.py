# [TAG_TURBOT] Check the SHIPPED turbot codebook tables against the Phase 0b nested coder study (docs/turbot/SPEC.md
# 1.1, 4.6, 11.4). CPU only (numpy), a few seconds, well under 1 GB.
#
#   python tools/turbot/check_tables_vs_study.py [--study-dir <dir with kv_nested_study.py, kv_sim.py, kv_cand.py>]
#
# 1. runs docs/turbot/gen_turbot_tables.py --check (the checked-in header equals a fresh generation)
# 2. parses ggml/include/ggml-turbot-tables.h (the lists the CPU reference, CUDA writer and CUDA reader expand)
# 3. compares them with kv_nested_study.build_designs(b 2..6, y 3..8, ["a_lloyd"]):
#      old levels, old thresholds, young LUT, young thresholds per (b, y)
#    tolerances (SPEC 4.6): exact for b = 4, 5; <= 2e-7 for b = 2, 3; <= 6e-5 for b = 6. The study's lloyd_max uses an
#    interpolated CDF for b = 2, 3, 6, the generator the exact kvfq erf iteration, which is authoritative.
# 4. structural checks on the shipped tables that need no study: run offsets, sortedness, bit-exact embedding of the old
#    thresholds in the young thresholds, nesting (young index >> r == old index) on a dense sweep, fill codes.
# Exit 0 only when everything passes.
import argparse
import math
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
HDR = os.path.join(ROOT, "ggml", "include", "ggml-turbot-tables.h")
GEN = os.path.join(ROOT, "docs", "turbot", "gen_turbot_tables.py")
STUDY_DEFAULT = r"C:\Users\xSniper\AppData\Local\Temp\claude\D--Projects\8efd145b-de13-4771-ae4d-110a53af2fef\scratchpad"
TOL = {2: 2e-7, 3: 2e-7, 4: 0.0, 5: 0.0, 6: 6e-5}

old_off = lambda b: (1 << b) - 4
young_off = lambda b, y: (b - 2) * 512 + 8 + (1 << y) - (1 << (b + 2))
fill_off = lambda b, y: {2: 0, 3: 24, 4: 64, 5: 128, 6: 224}[b] + (y - b - 1) * (1 << b)

FAILS = []


def fail(msg):
    FAILS.append(msg)
    print("  FAIL " + msg)


def macro(txt, name):
    m = re.search(r"#define %s \\\n(.*?)(?:\n\n|\n?\Z)" % name, txt, re.S)
    if not m:
        raise SystemExit("macro %s not found in %s" % (name, HDR))
    body = m.group(1).replace("\\", " ")
    return [t.strip() for t in body.replace("\n", " ").split(",") if t.strip()]


def load_header():
    txt = open(HDR, "rb").read().decode("utf-8").replace("\r", "")
    f32 = lambda name: np.array([float(t.rstrip("f")) for t in macro(txt, name)], dtype=np.float32)
    ints = lambda name: np.array([int(t) for t in macro(txt, name)], dtype=np.int64)
    return dict(old_lev=f32("GGML_TURBOT_OLD_LEVELS_LIST"), old_thr=f32("GGML_TURBOT_OLD_THR_LIST"),
                ylut=f32("GGML_TURBOT_YOUNG_LUT_LIST"), ythr=f32("GGML_TURBOT_YOUNG_THR_LIST"),
                fill=ints("GGML_TURBOT_FILL_LIST"), yoff=ints("GGML_TURBOT_YOUNG_OFF_LIST").reshape(7, 9),
                foff=ints("GGML_TURBOT_FILL_OFF_LIST").reshape(7, 9))


def maxdiff(a, b):
    return float(np.abs(a.astype(np.float64) - np.asarray(b, dtype=np.float64)).max()) if a.size else 0.0


def structural(t):
    print("structural checks on the shipped tables")
    if t["old_lev"].size != 124 or t["ylut"].size != 2312 or t["ythr"].size != 2312 or t["fill"].size != 352:
        fail("table sizes %d %d %d %d" % (t["old_lev"].size, t["ylut"].size, t["ythr"].size, t["fill"].size))
    xs = np.linspace(-0.6, 0.6, 200001, dtype=np.float32)
    for b in range(2, 7):
        L = 1 << b
        C = t["old_lev"][old_off(b):old_off(b) + L]
        T = t["old_thr"][old_off(b):old_off(b) + L - 1]
        if not np.all(np.diff(C) > 0):
            fail("b%d old levels not ascending" % b)
        if not np.array_equal(C, -C[::-1]):
            fail("b%d old levels not antisymmetric" % b)
        if not np.array_equal(T, (np.float32(0.5) * (C[:-1] + C[1:])).astype(np.float32)):
            fail("b%d old thresholds are not float32 midpoints" % b)
        if T[L // 2 - 1] != 0.0 or t["old_thr"][old_off(b) + L - 1] != 0.0:
            fail("b%d middle threshold or padding not 0" % b)
        jb = np.searchsorted(T, xs, side="right")
        for y in range(b + 1, 9):
            r = y - b
            n = 1 << y
            o = young_off(b, y)
            if t["yoff"][b, y] != o or t["foff"][b, y] != fill_off(b, y):
                fail("b%d y%d run offsets %d/%d, closed form %d/%d" % (b, y, t["yoff"][b, y], t["foff"][b, y], o, fill_off(b, y)))
            lut = t["ylut"][o:o + n]
            thr = t["ythr"][o:o + n - 1]
            if not np.all(np.diff(thr) >= 0):
                fail("b%d y%d young thresholds not sorted" % (b, y))
            if not np.all(np.diff(lut) > 0):
                fail("b%d y%d young LUT not ascending" % (b, y))
            pos = (np.arange(1, L) << r) - 1
            if thr[pos].tobytes() != T.tobytes():
                fail("b%d y%d old thresholds not embedded bit for bit" % (b, y))
            if not np.array_equal(np.searchsorted(thr, lut, side="right"), np.arange(n)):
                fail("b%d y%d a young level lies outside its own cell" % (b, y))
            iy = np.searchsorted(thr, xs, side="right")
            if not np.array_equal(iy >> r, jb):
                fail("b%d y%d nesting broken: young >> r != old index" % (b, y))
            fo = fill_off(b, y)
            for j in range(L):
                d = np.abs(lut[(j << r):((j + 1) << r)].astype(np.float64) - np.float64(C[j]))
                if int(np.argmin(d)) != int(t["fill"][fo + j]):
                    fail("b%d y%d fill code of j=%d is %d, nearest is %d" % (b, y, j, t["fill"][fo + j], int(np.argmin(d))))
    for b in range(0, 7):
        for y in range(0, 9):
            legal = b >= 2 and b < y <= 8
            if not legal and (t["yoff"][b, y] != -1 or t["foff"][b, y] != -1):
                fail("offset table entry (b%d, y%d) should be -1" % (b, y))


def vs_study(t, study_dir):
    print("shipped tables vs kv_nested_study.build_designs(a_lloyd), study dir %s" % study_dir)
    sys.path.insert(0, study_dir)
    import kv_nested_study as K  # noqa: E402
    refs, designs = K.build_designs([2, 3, 4, 5, 6], [3, 4, 5, 6, 7, 8], ["a_lloyd"], "minimax")
    worst = {}
    for b in range(2, 7):
        L = 1 << b
        tol = TOL[b]
        C = t["old_lev"][old_off(b):old_off(b) + L]
        T = t["old_thr"][old_off(b):old_off(b) + L - 1]
        dc, dm = maxdiff(C, refs[b]["C"]), maxdiff(T, refs[b]["mids"])
        ok = dc <= tol and dm <= tol
        print("  b=%d old levels max|d| %.3g, thresholds %.3g (tol %.0e) %s" % (b, dc, dm, tol, "ok" if ok else "FAIL"))
        if not ok:
            fail("b%d old tables differ from the study beyond %.0e" % (b, tol))
        worst[b] = max(dc, dm)
        for y in range(b + 1, 9):
            ds = [d for d in designs[(b, y)] if d["name"] == "a_lloyd"]
            if not ds:
                fail("study has no a_lloyd design for b%d y%d" % (b, y))
                continue
            d = ds[0]
            o = young_off(b, y)
            n = 1 << y
            dl = maxdiff(t["ylut"][o:o + n], d["lut_y"])
            dt = maxdiff(t["ythr"][o:o + n - 1], d["thr_r"])
            nequal = int(np.sum(t["ylut"][o:o + n] == d["lut_y"]))
            ok = dl <= tol and dt <= tol
            print("    y=%d LUT max|d| %.3g (%d/%d identical), thresholds %.3g %s" % (y, dl, nequal, n, dt, "ok" if ok else "FAIL"))
            if not ok:
                fail("b%d y%d young tables differ from the study beyond %.0e" % (b, y, tol))
            worst[b] = max(worst[b], dl, dt)
    print("  worst per b: " + ", ".join("b%d %.3g" % (b, worst[b]) for b in sorted(worst)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study-dir", default=STUDY_DEFAULT)
    ap.add_argument("--skip-gen-check", action="store_true")
    args = ap.parse_args()
    if not args.skip_gen_check:
        r = subprocess.run([sys.executable, GEN, "--check"], capture_output=True, text=True)
        print("gen_turbot_tables.py --check: " + (r.stdout.strip() or r.stderr.strip()))
        if r.returncode != 0:
            fail("checked-in ggml-turbot-tables.h differs from a fresh generation")
    t = load_header()
    structural(t)
    if os.path.exists(os.path.join(args.study_dir, "kv_nested_study.py")):
        vs_study(t, args.study_dir)
    else:
        fail("kv_nested_study.py not found in %s" % args.study_dir)
    print("RESULT: %s" % ("PASS" if not FAILS else "FAIL (%d)" % len(FAILS)))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
