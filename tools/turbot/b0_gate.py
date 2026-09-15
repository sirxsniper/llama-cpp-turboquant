# [TAG_TURBOT] Speed gate B0 (docs/turbot/SPEC.md 7.8): turbot FA op time against turbo5p at the same shapes.
# CPU only: reads test-backend-ops perf console logs.
#
#   test-backend-ops perf -b CUDA0 -o FLASH_ATTN_EXT -p "turbot_perf=|turbot_ref=" > b0_i8.log        (OLD_I8 1 build)
#   test-backend-ops perf -b CUDA0 -o FLASH_ATTN_EXT -p "turbot_perf=|turbot_ref=" > b0_float.log     (OLD_I8 0 build)
#   python tools/turbot/b0_gate.py b0_i8.log [--ref b0_i8.log] [--alt b0_float.log] [--md]
#
# The turbot cases print "turbot_perf=<widths>,kv=..,nb=..,mix=<old|band16k|band64k|young>,..." and the turbo5p
# reference cases "turbot_ref=turbo5p,kv=..,nb=..". Both may come from one log (the default) or from separate logs
# (--ref). Repeated runs of a case are reduced to the median us/run.
#
# GO iff all hold (turbot / turbo5p, ms per op):
#   kv 131072 nb 4   band16k <= 1.00
#   kv 245760 nb 4   band16k <= 1.00
#   kv 131072 nb 1   band16k <= 1.00
#   kv 131072 nb 4   band64k <= 1.05
#   kv 131072 nb 512 band16k <= 1.10
# kv 32768 young is reported, not gated. With --alt, the rule for GGML_CUDA_TURBOT_OLD_I8 is applied: keep the faster
# build that passes; if both pass within 2% of each other keep 0 (float reads).
import argparse
import re
import statistics
import sys

GATES = [(131072, 4, "band16k", 1.00), (245760, 4, "band16k", 1.00), (131072, 1, "band16k", 1.00),
         (131072, 4, "band64k", 1.05), (131072, 512, "band16k", 1.10)]
ANSI = re.compile(r"\x1b\[[0-9;]*m")
LINE = re.compile(r"^\s*FLASH_ATTN_EXT\((?P<vars>[^)]*)\):\s+(?P<runs>\d+) runs - +(?P<us>[0-9.]+) us/run")


GATED_WIDTHS = "l23"   # the gates apply to the default-plan layer 23 cases; other widths (uniform5, uniform6) are reported


def read_log_text(path):
    """A perf log as text, UTF-16 (PowerShell *> / Out-File redirect, with or without BOM) or UTF-8."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16", errors="replace")
    head = data[:4096]
    if head and head.count(b"\x00") * 4 >= len(head):   # UTF-16 without BOM: every other byte of ASCII text is NUL
        return data.decode("utf-16-le" if head[1:2] == b"\x00" else "utf-16-be", errors="replace")
    return data.decode("utf-8-sig", errors="replace")


def parse(path):
    """-> (turbot {(widths, kv, nb, mix): us}, ref {(kv, nb): us})"""
    tb, rf = {}, {}
    for raw in read_log_text(path).splitlines():
        m = LINE.match(ANSI.sub("", raw))
        if not m:
            continue
        kv = dict(p.split("=", 1) for p in m.group("vars").split(",") if "=" in p)
        us = float(m.group("us"))
        try:
            key_kv, key_nb = int(kv["kv"]), int(kv["nb"])
        except (KeyError, ValueError):
            continue
        if "turbot_perf" in kv:
            tb.setdefault((kv["turbot_perf"], key_kv, key_nb, kv.get("mix", "?")), []).append(us)
        elif kv.get("turbot_ref") == "turbo5p":
            rf.setdefault((key_kv, key_nb), []).append(us)
    med = lambda d: {k: statistics.median(v) for k, v in d.items()}
    return med(tb), med(rf)


def evaluate(name, tb, rf, md):
    print("\n%s" % name)
    hdr = "| kv | nb | mix | turbot ms | turbo5p ms | ratio | limit | verdict |"
    hdr = "| widths | kv | nb | mix | turbot ms | turbo5p ms | ratio | limit | verdict |"
    print(hdr if md else "%-9s %8s %5s %-8s %11s %11s %7s %6s  %s" % ("widths", "kv", "nb", "mix", "turbot ms", "turbo5p ms", "ratio", "limit", "verdict"))
    if md:
        print("|---|---|---|---|---|---|---|---|---|")
    limits = {(GATED_WIDTHS, k, n, m): lim for k, n, m, lim in GATES}
    go, missing, worst = True, [], {}
    for key in sorted(tb, key=lambda k: (k[0] != GATED_WIDTHS, k[0], k[1], k[2], k[3])):
        w, kv, nb, mix = key
        ref = rf.get((kv, nb))
        ratio = tb[key] / ref if ref else float("nan")
        lim = limits.get(key)
        if lim is None:
            verdict = "report"
        elif ref is None:
            verdict = "NO REF"
        else:
            verdict = "pass" if ratio <= lim else "FAIL"
        if lim is not None:
            worst[key] = ratio
        row = (w, kv, nb, mix, tb[key] / 1000.0, (ref or float("nan")) / 1000.0, ratio, ("%.2f" % lim) if lim else "-", verdict)
        print(("| %s | %d | %d | %s | %.3f | %.3f | %.3f | %s | %s |" if md else "%-9s %8d %5d %-8s %11.3f %11.3f %7.3f %6s  %s") % row)
    for kv, nb, mix, lim in GATES:
        key = (GATED_WIDTHS, kv, nb, mix)
        if key not in tb or (kv, nb) not in rf:
            missing.append("kv %d nb %d %s" % key[1:])
            go = False
        elif tb[key] / rf[(kv, nb)] > lim:
            go = False
    if missing:
        print("missing gate cases: " + "; ".join(missing))
    print("B0 %s: %s" % (name, "GO" if go else "NO-GO"))
    return go, worst


def main():
    ap = argparse.ArgumentParser(description="turbot gate B0")
    ap.add_argument("log", help="perf log of the build under test (turbot cases, and turbo5p refs unless --ref)")
    ap.add_argument("--ref", default=None, help="perf log holding the turbo5p reference cases")
    ap.add_argument("--alt", default=None, help="perf log of the other GGML_CUDA_TURBOT_OLD_I8 build")
    ap.add_argument("--alt-is-i8", action="store_true", help="the --alt build is OLD_I8 1 (default: --alt is the float build)")
    ap.add_argument("--md", action="store_true", help="markdown tables for docs/turbot/TESTING.md")
    args = ap.parse_args()
    tb, rf = parse(args.log)
    if args.ref:
        rf = parse(args.ref)[1]
    if not tb:
        print("no turbot_perf cases in %s" % args.log)
        return 2
    if not rf:
        print("no turbot_ref=turbo5p cases found")
        return 2
    go, worst = evaluate(args.log, tb, rf, args.md)
    if not args.alt:
        return 0 if go else 1
    tb2, rf2 = parse(args.alt)
    if not args.ref and rf2:
        rf_alt = rf2
    else:
        rf_alt = rf
    go2, worst2 = evaluate(args.alt, tb2, rf_alt, args.md)
    main_is_i8 = not args.alt_is_i8
    name_main = "OLD_I8 %d" % (1 if main_is_i8 else 0)
    name_alt = "OLD_I8 %d" % (0 if main_is_i8 else 1)
    if not go and not go2:
        print("\nDECISION: NO-GO in both builds. Stop the integration and record the tables in docs/turbot/TESTING.md.")
        return 1
    if go != go2:
        print("\nDECISION: GO with %s (the only build that passes)." % (name_main if go else name_alt))
        return 0
    # both pass: compare the geometric mean of the gated ratios
    gm = lambda w: statistics.geometric_mean(list(w.values())) if w else float("nan")
    a, b = gm(worst), gm(worst2)
    float_name = name_alt if main_is_i8 else name_main
    faster = name_main if a < b else name_alt
    if abs(a - b) / max(a, b) <= 0.02:
        print("\nDECISION: GO. Both builds pass within 2%% (gated geomean %.3f vs %.3f): keep %s (float reads)." % (a, b, float_name))
    else:
        print("\nDECISION: GO with %s (gated geomean %.3f vs %.3f)." % (faster, a, b))
    return 0


if __name__ == "__main__":
    sys.exit(main())
