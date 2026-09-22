# [TAG_TURBOT] Print the VRAM and plan hash of a turbot plan (docs/turbot/SPEC.md 3.6, 9.1, 11.4). CPU only.
#
#   python tools/turbot/plan_vram.py docs/turbot/plans/turbot-default.plan [--kv 262144] [--layers] [--attn-layers 3,7,...]
#   [TAG_TURBOT_ANY_GEOM] other shapes: --geom <head dim>x<KV heads> for every layer, or --shape il:DxH,il:DxH,...
#   python tools/turbot/plan_vram.py ornith35.plan --attn-layers 3,7,11,15,19,23,27,31,35,39 --geom 256x2
#   [TAG_TURBOT_ANY_STREAMS] --n-stream N: N streams of --kv cells each (-np N without --kv-unified), one pool
#
# Must print base 4520.00 / young pool 726.00 / total 5246.00 MiB for the default plan. The hash is the value
# llama_kv_cache logs ("hash 0x...") and the state blob v2 carries; tests/test-turbot.cpp checks the same number.
# The reference is the cache's fallback type (turbo5p for rows of 1024 values, turbo5p512 for 512, turbo4 for 256),
# the size llama_kv_cache warns against.
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import turbot_plan as P  # noqa: E402


def parse_geom(s):
    d, h = s.lower().split("x")
    return int(d), int(h)


def main():
    ap = argparse.ArgumentParser(description="turbot plan VRAM")
    ap.add_argument("plan")
    ap.add_argument("--kv", type=int, default=P.KV_DEFAULT, help="cells per stream (default 262144)")
    ap.add_argument("--n-stream", type=int, default=1, help="KV streams (default 1)")
    ap.add_argument("--layers", action="store_true", help="print the per-layer byte table")
    ap.add_argument("--attn-layers", default=",".join(map(str, P.QWEN38_ATTN_LAYERS)),
                    help="attention layers the cache holds (default Qwen3.8-27B: 3,7,...,63); '' skips the check")
    ap.add_argument("--geom", default="256x4", help="<head dim>x<KV heads> of every attention layer (default 256x4)")
    ap.add_argument("--shape", default="", help="il:DxH,... per layer; replaces --attn-layers and --geom")
    args = ap.parse_args()

    shape = None
    try:
        if args.shape:
            layers = {}
            for item in args.shape.split(","):
                il, g = item.split(":")
                layers[int(il)] = parse_geom(g)
            shape = P.make_shape(layers, kv_size=args.kv, n_stream=args.n_stream)
        else:
            attn = [int(x) for x in args.attn_layers.split(",") if x.strip()]
            d, h = parse_geom(args.geom)
            if attn and ((d, h) != (256, 4) or args.n_stream != 1):
                shape = P.make_shape(attn, d, h, kv_size=args.kv, n_stream=args.n_stream)
        if shape is not None:
            plan = P.parse_file(args.plan, shape=shape)
        else:
            attn = [int(x) for x in args.attn_layers.split(",") if x.strip()] or None
            plan = P.parse_file(args.plan, attn, args.kv * args.n_stream)
    except (P.PlanError, OSError, ValueError) as e:
        print("REFUSED: %s" % e)
        return 1

    v = P.vram(plan, args.kv, args.n_stream)
    cells = args.kv * args.n_stream
    if plan["ignored"]:
        print("note: kvfq keys ignored: %s" % " ".join(plan["ignored"]))
    if args.layers:
        print("   il  runs  base K  base V  young K  young V")
        for il, bk, bv, yk, yv in v["rows"]:
            print("%5d %5d %7d %7d %8d %8d" % (il, len(plan["layers"][il]["bk"]), bk, bv, yk, yv))
        print("  sum %13d B/cell base, %d B/pool cell young" % (v["base_cell"], v["pool_cell"]))
    print("cells       %d%s" % (cells, "" if args.n_stream == 1 else " (%d streams x %d)" % (args.n_stream, args.kv)))
    print("base        %10.2f MiB  (%d B)" % (v["base"] / P.MIB, v["base"]))
    pool_s = ""
    if args.n_stream > 1:
        pool_s = ", %d cells per stream" % (plan["pool"] // args.n_stream // P.GRANULE * P.GRANULE)
    print("young pool  %10.2f MiB  (%d B, POOL %d cells = %d granules%s, CAP %d)"
          % (v["pool"] / P.MIB, v["pool"], plan["pool"], plan["pool"] // 64, pool_s, plan["cap"]))
    print("total       %10.2f MiB  (%d B)" % (v["total"] / P.MIB, v["total"]))
    print("%-11s %10.2f MiB  (margin %.2f MiB)%s" % (v["fallback_type"], v["fallback"] / P.MIB, (v["fallback"] - v["total"]) / P.MIB,
                                                   "" if v["total"] <= v["fallback"] else "  WARNING: above %s" % v["fallback_type"]))
    if v["fallback_type"] != "turbo5p":
        print("turbo5p rate %9.2f MiB  (LLAMA_TURBOT_AUTO_BUDGET=turbo5p budget)" % (v["turbo5p"] / P.MIB))
    print("old bits    mean %.3f, sum %d over %d runs" % (v["s_sum"] / float(v["n_runs"]), v["s_sum"], v["n_runs"]))
    print("hash        0x%016x" % P.plan_hash(plan))
    return 0


if __name__ == "__main__":
    sys.exit(main())
