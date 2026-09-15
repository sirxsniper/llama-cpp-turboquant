# [TAG_TURBOT] Print the VRAM and plan hash of a turbot plan (docs/turbot/SPEC.md 3.6, 9.1, 11.4). CPU only.
#
#   python tools/turbot/plan_vram.py docs/turbot/plans/turbot-default.plan [--kv 262144] [--layers] [--attn-layers 3,7,...]
#
# Must print base 4520.00 / young pool 726.00 / total 5246.00 MiB for the default plan. The hash is the value
# llama_kv_cache logs ("hash 0x...") and the state blob v2 carries; tests/test-turbot.cpp checks the same number.
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import turbot_plan as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="turbot plan VRAM")
    ap.add_argument("plan")
    ap.add_argument("--kv", type=int, default=P.KV_DEFAULT, help="cells (default 262144)")
    ap.add_argument("--layers", action="store_true", help="print the per-layer byte table")
    ap.add_argument("--attn-layers", default=",".join(map(str, P.QWEN38_ATTN_LAYERS)),
                    help="attention layers the cache holds (default Qwen3.8-27B: 3,7,...,63); '' skips the check")
    args = ap.parse_args()
    attn = [int(x) for x in args.attn_layers.split(",") if x.strip()] or None
    try:
        plan = P.parse_file(args.plan, attn, args.kv)
    except (P.PlanError, OSError) as e:
        print("REFUSED: %s" % e)
        return 1
    v = P.vram(plan, args.kv)
    if plan["ignored"]:
        print("note: kvfq keys ignored: %s" % " ".join(plan["ignored"]))
    if args.layers:
        print("   il  base K  base V  young K  young V")
        for il, bk, bv, yk, yv in v["rows"]:
            print("%5d %7d %7d %8d %8d" % (il, bk, bv, yk, yv))
        print("  sum %7d B/cell base, %d B/pool cell young" % (v["base_cell"], v["pool_cell"]))
    print("cells       %d" % args.kv)
    print("base        %10.2f MiB  (%d B)" % (v["base"] / P.MIB, v["base"]))
    print("young pool  %10.2f MiB  (%d B, POOL %d cells = %d granules, CAP %d)"
          % (v["pool"] / P.MIB, v["pool"], plan["pool"], plan["pool"] // 64, plan["cap"]))
    print("total       %10.2f MiB  (%d B)" % (v["total"] / P.MIB, v["total"]))
    print("turbo5p     %10.2f MiB  (margin %.2f MiB)%s" % (v["turbo5p"] / P.MIB, (v["turbo5p"] - v["total"]) / P.MIB,
                                                       "" if v["total"] <= v["turbo5p"] else "  WARNING: above turbo5p"))
    print("old bits    mean %.3f, sum %d over %d heads" % (v["s_sum"] / float(v["n_heads"]), v["s_sum"], v["n_heads"]))
    print("hash        0x%016x" % P.plan_hash(plan))
    return 0


if __name__ == "__main__":
    sys.exit(main())
