# [TAG_TURBOT] Generator for ggml/include/ggml-turbot-tables.h (the codebook tables of the turbot KV format).
#
# CPU only (python + numpy), deterministic, data independent. Writes the header with CRLF line endings like the
# rest of the working tree.
#
#   python docs/turbot/gen_turbot_tables.py            # (re)write ggml/include/ggml-turbot-tables.h
#   python docs/turbot/gen_turbot_tables.py --check    # exit 1 if the checked-in header differs from a fresh run
#
# Math, docs/turbot/SPEC.md section 4 (all in float64, cast to float32 at the end):
#   old (base) codebook C_b, b = 2..6, "r units" (one coordinate of a unit 128-vector):
#     b = 4, 5: the turbo4 / turbo5p literal tables (C4N / C5N mirrored), cast to float32
#     b = 2, 3, 6: llama-kvfq.cpp lloyd_max_gauss(b) (exact erf, same iteration) / sqrt(128), cast to float32
#   old thresholds: float32 0.5f * (C_b[i] + C_b[i+1]), i = 0..2^b-2   (llama-kvfq.cpp get_levels().mids)
#   young partition for (b, y), r = y - b ("a_lloyd", kv_nested_study.py build_designs):
#     z units: t = float64(old thresholds) * sqrt(128); base cell j = (t[j-1], t[j]) with t[-1] = -inf, t[L-1] = +inf
#     per cell: 2^r level Lloyd-Max of N(0,1) truncated to the cell. Init levels uniform over [lo, hi] where the
#     unbounded outer cells use [t0 - 2w, t0] and [t_{L-2}, t_{L-2} + 2w], w = t[1] - t[0]. Iterate
#     sub-thresholds = midpoints of levels, levels = truncated-Gaussian centroids of the sub-cells, until the max
#     level change < 1e-11 (at most 20000 iterations).
#     young LUT  = levels / sqrt(128), float32, flattened in cell order (index (j << r) | s)
#     young thr  = the 2^y - 1 thresholds in ascending order: inner sub-thresholds / sqrt(128) cast to float32, and
#                  at positions ((j + 1) << r) - 1 the float32 old thresholds verbatim (bit-exact nesting)
#   fill code for (b, y, j): the refinement s in 0..2^r-1 whose young level is nearest to C_b[j]
#     (smallest |lut - C_b[j]| in float64 of the float32 values, ties to the lower s)
import math
import os
import sys

import numpy as np

B_MIN, B_MAX, Y_MAX = 2, 6, 8
SQ128 = math.sqrt(128.0)
F64 = np.float64

C5N = [-0.271948, -0.222223, -0.189260, -0.163683, -0.142366, -0.123814, -0.107237, -0.092232,
       -0.078519, -0.065814, -0.053979, -0.042868, -0.032338, -0.022390, -0.013026, -0.004245]
C4N = [-0.241530, -0.182875, -0.143021, -0.111033, -0.083297, -0.058053, -0.034304, -0.011349]


# ---------------------------------------------------------------- exact kvfq base codebook
def _Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _phi(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * 3.14159265358979323846)


def lloyd_max_gauss(bits):
    """llama-kvfq.cpp lloyd_max_gauss(), step for step (float64)."""
    L = 1 << bits
    c = [0.0] * L
    for i in range(L):
        target = (i + 0.5) / L
        lo, hi = -12.0, 12.0
        for _ in range(100):
            m = 0.5 * (lo + hi)
            if _Phi(m) < target:
                lo = m
            else:
                hi = m
        c[i] = 0.5 * (lo + hi)
    for _ in range(20000):
        delta = 0.0
        cn = [0.0] * L
        for i in range(L):
            has_lo, has_hi = i > 0, i < L - 1
            tlo = 0.5 * (c[i - 1] + c[i]) if has_lo else 0.0
            thi = 0.5 * (c[i] + c[i + 1]) if has_hi else 0.0
            pl = _phi(tlo) if has_lo else 0.0
            ph = _phi(thi) if has_hi else 0.0
            Pl = _Phi(tlo) if has_lo else 0.0
            Ph = _Phi(thi) if has_hi else 1.0
            cn[i] = (pl - ph) / max(Ph - Pl, 1e-300)
            delta = max(delta, abs(cn[i] - c[i]))
        c = cn
        if delta < 1e-12:
            break
    return c


def old_levels(b):
    if b == 5:
        v = np.array(C5N + [-x for x in reversed(C5N)], dtype=np.float32)
    elif b == 4:
        v = np.array(C4N + [-x for x in reversed(C4N)], dtype=np.float32)
    else:
        v = np.array([x / math.sqrt(128.0) for x in lloyd_max_gauss(b)], dtype=F64).astype(np.float32)
    return v


def old_mids(c):
    return (np.float32(0.5) * (c[:-1] + c[1:])).astype(np.float32)


# ---------------------------------------------------------------- truncated Gaussian moments (kv_nested_study.py)
_ERF = np.frompyfunc(math.erf, 1, 1)


def Phi(x):
    x = np.asarray(x, dtype=F64)
    return 0.5 * (1.0 + _ERF(x / math.sqrt(2.0)).astype(F64))


def phi(x):
    x = np.asarray(x, dtype=F64)
    return np.exp(-0.5 * np.minimum(x * x, 1e6)) / math.sqrt(2.0 * math.pi)


def moments(edges):
    e = np.asarray(edges, dtype=F64)
    fin = np.isfinite(e)
    ef = np.where(fin, e, 0.0)
    P, ph = Phi(e), np.where(fin, phi(ef), 0.0)
    xph = np.where(fin, ef * phi(ef), 0.0)
    return np.diff(P, axis=-1), -np.diff(ph, axis=-1), np.diff(P - xph, axis=-1)


def centroids(edges):
    m0, m1, _ = moments(edges)
    e = np.clip(np.asarray(edges, dtype=F64), -12.0, 12.0)
    mid = 0.5 * (e[..., :-1] + e[..., 1:])
    return np.where(m0 > 1e-300, m1 / np.maximum(m0, 1e-300), mid)


def interleave(inner, base):
    L = inner.shape[0]
    out = []
    for j in range(L):
        out.extend(inner[j].tolist())
        if j < L - 1:
            out.append(float(base[j]))
    return np.array(out, dtype=F64)


A_LLOYD_TOL = 1e-11      # stop when the max level change is below this (kv_nested_study.py)
A_LLOYD_MAX_ITERS = 20000
A_LLOYD_ASSERT = 1e-9    # b = 2, y = 8 stops at the iteration cap with delta 3.8e-10; anything worse is a regression


def a_lloyd(mids_b, r):
    """-> lut_y float32 [2^y], thr_y float32 [2^y - 1], iterations, final max level change
    (kv_nested_study.py a_lloyd + finish)."""
    t = mids_b.astype(F64) * SQ128
    wadj = t[1] - t[0]
    sub = 1 << r
    e_lo = np.concatenate(([-np.inf], t))
    e_hi = np.concatenate((t, [np.inf]))
    with np.errstate(invalid="ignore"):
        lo = np.where(np.isfinite(e_lo), e_lo, e_hi - 2.0 * wadj)
        hi = np.where(np.isfinite(e_hi), e_hi, e_lo + 2.0 * wadj)
    lev = lo[:, None] + (hi - lo)[:, None] * (np.arange(sub) + 0.5)[None, :] / sub
    iters, dlt = 0, float("inf")
    for iters in range(1, A_LLOYD_MAX_ITERS + 1):
        th = 0.5 * (lev[:, :-1] + lev[:, 1:])
        E = np.concatenate([e_lo[:, None], th, e_hi[:, None]], axis=1)
        new = centroids(E)
        dlt = float(np.abs(new - lev).max())
        lev = new
        if dlt < A_LLOYD_TOL:
            break
    assert dlt < A_LLOYD_ASSERT, "a_lloyd did not converge: r=%d delta %.3g after %d iterations" % (r, dlt, iters)
    th = 0.5 * (lev[:, :-1] + lev[:, 1:])
    thr = (interleave(th, t) / SQ128).astype(np.float32)
    L = t.size + 1
    pos = (np.arange(1, L) << r) - 1
    thr[pos] = mids_b
    assert thr.size == (1 << (L.bit_length() - 1 + r)) - 1
    assert np.all(np.diff(thr) >= 0), "young thresholds not sorted"
    lut = (lev.ravel() / SQ128).astype(np.float32)
    return lut, thr, iters, dlt


# ---------------------------------------------------------------- emit
def fmt_floats(vals, per=6):
    out = []
    for i in range(0, len(vals), per):
        out.append("    " + " ".join("%.9g," % float(v) if float(v) != 0.0 else "0.0," for v in vals[i:i + per]))
    return out


def fmt_ints(vals, per=16):
    return ["    " + " ".join("%d," % int(v) for v in vals[i:i + per]) for i in range(0, len(vals), per)]


def fixfloat(line):
    # '%.9g' may print integers or exponents; make every token a float32 literal
    toks = []
    for tok in line.split():
        body = tok.rstrip(",")
        if "." not in body and "e" not in body:
            body += ".0"
        toks.append(body + "f,")
    return "    " + " ".join(toks)


def macro(name, lines, is_float):
    res = ["#define %s \\" % name]
    for k, l in enumerate(lines):
        l = fixfloat(l) if is_float else l
        res.append(l + ("" if k + 1 == len(lines) else " \\"))
    res.append("")
    return res


def generate():
    old_lev, old_thr = [], []
    young_lut, young_thr, fill = [], [], []
    young_off = [[-1] * (Y_MAX + 1) for _ in range(B_MAX + 1)]
    fill_off = [[-1] * (Y_MAX + 1) for _ in range(B_MAX + 1)]
    stats = []
    for b in range(B_MIN, B_MAX + 1):
        c = old_levels(b)
        m = old_mids(c)
        L = 1 << b
        assert np.all(np.diff(c) > 0) and np.all(np.diff(m) > 0)
        assert np.array_equal(c, -c[::-1]), "old levels must be antisymmetric"
        assert m[L // 2 - 1] == 0.0
        old_lev += c.tolist()
        old_thr += m.tolist() + [0.0]
        for y in range(b + 1, Y_MAX + 1):
            r = y - b
            lut, thr, iters, dlt = a_lloyd(m, r)
            young_off[b][y] = len(young_lut)
            young_lut += lut.tolist()
            young_thr += thr.tolist() + [0.0]
            fill_off[b][y] = len(fill)
            for j in range(L):
                d = np.abs(lut[(j << r):((j + 1) << r)].astype(F64) - F64(c[j]))
                fill.append(int(np.argmin(d)))
            # nesting self check on a dense sweep: base index == young index >> r
            xs = np.linspace(-0.6, 0.6, 20001, dtype=np.float32)
            jb = np.searchsorted(m, xs, side="right")
            iy = np.searchsorted(thr, xs, side="right")
            assert np.array_equal(iy >> r, jb), (b, y)
            stats.append((b, y, iters, dlt))
    return old_lev, old_thr, young_lut, young_thr, fill, young_off, fill_off, stats


def render():
    old_lev, old_thr, young_lut, young_thr, fill, young_off, fill_off, stats = generate()
    o = []
    o.append("// [TAG_TURBOT] GENERATED by docs/turbot/gen_turbot_tables.py - do not edit by hand, re-run the script.")
    o.append("// Codebook tables of the turbot KV format, see docs/turbot/SPEC.md section 4. Values are float32 in r units")
    o.append("// (one coordinate of a unit-norm WHT-128 group). Included by ggml-turbot.h (host) and")
    o.append("// ggml-cuda/turbot-tables.cuh (device); both expand the SAME lists, so CPU and GPU agree bit for bit.")
    o.append("#pragma once")
    o.append("")
    o.append("#define GGML_TURBOT_OLD_TOTAL   %d   // sum over b = 2..6 of 2^b" % len(old_lev))
    o.append("#define GGML_TURBOT_YOUNG_TOTAL %d  // sum over b = 2..6, y = b+1..8 of 2^y" % len(young_lut))
    o.append("#define GGML_TURBOT_FILL_TOTAL  %d   // sum over b = 2..6, y = b+1..8 of 2^b" % len(fill))
    o.append("")
    o.append("// a_lloyd convergence per (b, y): iterations, final max level change (z units). Stop rule: change < %g or %d"
             % (A_LLOYD_TOL, A_LLOYD_MAX_ITERS))
    o.append("// iterations, the same rule as kv_nested_study.py; the generator asserts change < %g." % A_LLOYD_ASSERT)
    for k in range(0, len(stats), 4):
        o.append("//  " + "   ".join("b%d y%d %5d %.2e" % s for s in stats[k:k + 4]))
    o.append("")
    o.append("// old codebook C_b[j], entry ggml_turbot_old_off(b) + j, j = 0..2^b-1, ascending, antisymmetric")
    o += macro("GGML_TURBOT_OLD_LEVELS_LIST", fmt_floats(old_lev), True)
    o.append("// old thresholds, entry ggml_turbot_old_off(b) + i, i = 0..2^b-2; slot 2^b-1 of each run is 0 padding")
    o += macro("GGML_TURBOT_OLD_THR_LIST", fmt_floats(old_thr), True)
    o.append("// young LUT, entry GGML_TURBOT_YOUNG_OFF(b, y) + ((j << (y-b)) | s), j = base index, s = refinement")
    o += macro("GGML_TURBOT_YOUNG_LUT_LIST", fmt_floats(young_lut), True)
    o.append("// young thresholds, entry GGML_TURBOT_YOUNG_OFF(b, y) + i, i = 0..2^y-2, ascending; slot 2^y-1 is 0 padding.")
    o.append("// Entry ((j+1) << (y-b)) - 1 is bit-identical to old threshold j.")
    o += macro("GGML_TURBOT_YOUNG_THR_LIST", fmt_floats(young_thr), True)
    o.append("// center-fill refinement code s for base index j, entry GGML_TURBOT_FILL_OFF(b, y) + j")
    o += macro("GGML_TURBOT_FILL_LIST", fmt_ints(fill, 32), False)
    o.append("// run offsets, row b (0..6), column y (0..8), -1 where (b, y) is not a legal pair")
    o += macro("GGML_TURBOT_YOUNG_OFF_LIST", ["    " + " ".join("%d," % v for v in young_off[b]) for b in range(B_MAX + 1)], False)
    o += macro("GGML_TURBOT_FILL_OFF_LIST", ["    " + " ".join("%d," % v for v in fill_off[b]) for b in range(B_MAX + 1)], False)
    return "\n".join(o)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.normpath(os.path.join(here, "..", "..", "ggml", "include", "ggml-turbot-tables.h"))
    text = render().replace("\n", "\r\n")
    if "--check" in sys.argv:
        cur = open(path, "rb").read().decode("utf-8") if os.path.exists(path) else ""
        same = cur == text
        print("ggml-turbot-tables.h %s" % ("matches" if same else "DIFFERS from a fresh generation"))
        return 0 if same else 1
    with open(path, "wb") as f:
        f.write(text.encode("utf-8"))
    print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
