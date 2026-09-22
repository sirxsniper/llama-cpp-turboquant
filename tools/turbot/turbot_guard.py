# [TAG_TURBOT_ANY_GUARD] Quality guard for a turbot plan on a model other than Qwen3.8-27B (GPU gate G5 of the
# "turbot on other models" plan). It measures the plan against the KV type turbot would otherwise fall back to and, only
# when turbot is better, stamps the plan file with the '# model:' fingerprint and a '# verified:' line. libllama uses a
# <model>.turbot.plan sidecar only with both (llama_turbot_plan_choose step 2, llama_turbot_sidecar_accepts).
#
#   python tools/turbot/turbot_guard.py --exe-dir <build>/bin --model D:/Projects/LocalAI/models/Ornith-1.5-9B-Q8_0.gguf
#          [--plan <file>] [--plan-ctx 262144 --plan-np 1 [--plan-unified] --budget fallback|turbo5p]
#          [--fallback turbo5p] [--sidecar] [--force] [--keep] [--out E:/turbot-guard] [--chunks 8]
#   python tools/turbot/turbot_guard.py --selftest        (CPU only: parser and decision on synthetic logs)
#
# Plan: --plan <file>, or the automatic plan of the production shape (turbot_plan.py auto --gguf <model> -c --plan-ctx -np
# --plan-np, written to <out>/<model>/<model>.turbot.plan). The arms load it with --kv-tier-plan <file>, so the widths that
# are measured are the widths the server would use; POOL is clamped to the 32K perplexity cache as llama does.
#
# Arms, per corpus (E:/kv-bar-s0/code_corpus.txt and prose_corpus.txt), llama-perplexity -c 32768 --chunks 8 -ub 512
# -b 2048 -fa on -ngl 999, logs and logits under <out>/<model>/:
#   base      -ctk f16 -ctv f16 --kl-divergence-base <corpus>_base.dat
#   fallback  -ctk <fallback> -ctv <fallback> --kl-divergence (turbo5p for rows of 1024 / 512 values, turbo4 for 256:
#             the type the resolver falls back to; turbo5p runs as turbo5p512 on 512-value rows by itself)
#   turbot    -ctk turbot -ctv turbot --kv-tier-plan <plan> --kl-divergence
# Every arm must log the KV type it asked for (the "llama_kv_cache: size = ..." line) and no "KV cache type for" downgrade;
# the turbot arm must also log "turbot plan <plan>" with the hash of the plan at this cache size and the model fingerprint.
# A silent downgrade fails the arm, and with it the guard.
#
# Pass rule (G5), per corpus, turbot - fallback on the paired chunks (paired chunk bootstrap, ported from
# kv_gate_stats.py, 4000 resamples, seed 12345):
#   - the 95% CI of the mean KLD difference is entirely below 0
#   - the lower bound of the 95% CI of the same-top difference is >= -0.05 percentage points
#   - the 99.9% KLD of turbot is no worse than the fallback's
# A KLD CI that crosses 0 at --chunks 8 re-runs the three arms of that corpus at 16 chunks and decides on those.
#
# On a pass: the plan file gets '# model: <fingerprint>' and '# verified: <date> <build> <per-corpus deltas>' (old stamp
# lines replaced; comments only, the plan hash does not change). A model whose attention layers the built-in plan already
# names (Qwen3.8-27B and its fine-tunes) is refused unless --force: a verified sidecar would replace the calibrated
# built-in plan. --sidecar also copies the stamped plan to <model>.turbot.plan (an existing one is kept as .bak).
# The .dat logits files are deleted at the end unless --keep.
#
# ONE GPU process: refuses to start, and to start each arm, while any llama / ggml / test-backend process runs.
import argparse
import datetime
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import turbot_plan as P  # noqa: E402

try:
    import numpy as np
except ImportError:      # the statistics need numpy; --help and the plan checks do not
    np = None

CORPUS_DIR = "E:/kv-bar-s0"
OUT_DEFAULT = "E:/turbot-guard"
BUILTIN_PLAN = os.path.normpath(os.path.join(HERE, "..", "..", "docs", "turbot", "plans", "turbot-default.plan"))
DEF_BOOT, DEF_SEED = 4000, 12345
TOP_FLOOR = -0.05        # same-top CI lower bound, percentage points
KLD_Q, TOP_Q = 1e-5, 1e-3
P_SUFFIX = ".turbot.plan"   # the sidecar next to a model (LLAMA_TURBOT_SIDECAR_SUFFIX)


# ---------------------------------------------------------------------------------------------------------------
# processes
# ---------------------------------------------------------------------------------------------------------------

def busy():
    out = subprocess.run(["tasklist"], capture_output=True, text=True).stdout.lower()
    return [l.split()[0] for l in out.splitlines() if any(k in l for k in ("llama", "ggml", "test-backend"))]


def run_arm(exe, argv, env_extra, log_path, timeout_min):
    if busy():
        raise SystemExit("REFUSED: a llama/ggml process is running: %s" % busy())
    env = dict(os.environ)
    env.update(env_extra)
    print("  run %s" % " ".join([exe] + argv))
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        p = subprocess.Popen([exe] + argv, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            rc = p.wait(timeout=timeout_min * 60)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
            rc = -9
    print("  rc %s in %.0f s, log %s" % (rc, time.time() - t0, log_path))
    for _ in range(60):
        if not busy():
            break
        time.sleep(1)
    with open(log_path, encoding="utf-8", errors="replace") as f:
        return rc, f.read()


# ---------------------------------------------------------------------------------------------------------------
# log checks
# ---------------------------------------------------------------------------------------------------------------

SIZE_LINE = re.compile(r"llama_kv_cache: size = +([0-9.]+) MiB .*?K \((\w+)\): +[0-9.]+ MiB, V \((\w+)\)")
PLAN_LINE = re.compile(r"turbot plan (.+?): (\d+) layers, old bits ([0-9.]+) .*hash 0x([0-9a-f]{16})")
FP_LINE = re.compile(r"turbot: model fingerprint: (.+)$", re.M)
DOWNGRADE = re.compile(r"KV cache type for [^\n]*")
BUILD_LINE = re.compile(r"build: (\d+) \(([0-9a-f]+)\)")


def check_arm_log(text, want_type, plan_path=None, plan_hash=None):
    """-> (ok, notes, facts). want_type: the KV type the size lines must show (turbo5p also accepts turbo5p512). With
    turbot, the SWA child of an iSWA cache keeps its own type (one size line must be turbot, a downgrade line that names
    the SWA layers is allowed); every other arm needs its type on every size line."""
    notes, facts = [], {}
    sizes = SIZE_LINE.findall(text)
    has_turbot = any(tk == "turbot" for _, tk, _ in sizes)
    for d in DOWNGRADE.findall(text):
        if want_type == "turbot" and has_turbot and "SWA" in d:
            continue
        notes.append("downgraded: %s" % d)
        break
    if not sizes:
        notes.append("no 'llama_kv_cache: size' line")
    else:
        facts["size_mib"] = sum(float(mib) for mib, _, _ in sizes)
        facts["type"] = ",".join(tk for _, tk, _ in sizes)
        ok_types = {want_type} | ({"turbo5p512"} if want_type == "turbo5p" else set())
        if want_type == "turbot":
            if not has_turbot:
                notes.append("no turbot cache (KV types %s)" % facts["type"])
        else:
            for _, tk, tv in sizes:
                if tk not in ok_types or tv not in ok_types:
                    notes.append("KV types K %s / V %s, asked for %s" % (tk, tv, want_type))
                    break
    if want_type == "turbot":
        m = PLAN_LINE.search(text)
        if not m:
            notes.append("no 'turbot plan' line")
        else:
            facts["plan"], facts["hash"] = m.group(1), m.group(4)
            if plan_path is not None and os.path.normcase(os.path.abspath(m.group(1))) != os.path.normcase(os.path.abspath(plan_path)):
                notes.append("plan %s, asked for %s" % (m.group(1), plan_path))
            if plan_hash is not None and int(m.group(4), 16) != plan_hash:
                notes.append("plan hash 0x%s, expected 0x%016x" % (m.group(4), plan_hash))
        f = FP_LINE.search(text)
        if f:
            facts["fingerprint"] = f.group(1).strip()
        else:
            notes.append("no 'turbot: model fingerprint' line")
    b = BUILD_LINE.search(text)
    if b:
        facts["build"] = "b%s-%s" % (b.group(1), b.group(2))
    return not notes, notes, facts


def shape_of_fingerprint(fp, kv_size):
    """make_shape from 'arch=.. layers=il,.. geom=DxH[,DxH..]'."""
    kv = dict(re.findall(r"(\w+)=(\S*)", fp))
    ils = [int(x) for x in kv.get("layers", "").split(",") if x]
    geos = [tuple(int(v) for v in g.split("x")) for g in kv.get("geom", "").split(",") if g]
    if len(geos) == 1:
        geos = geos * len(ils)
    if len(geos) != len(ils) or not ils:
        raise ValueError("fingerprint '%s' has %d layers and %d geometries" % (fp, len(ils), len(geos)))
    return P.make_shape(dict(zip(ils, geos)), kv_size=kv_size)


# ---------------------------------------------------------------------------------------------------------------
# statistics (kv_gate_stats.py: rounded running means -> per-chunk values, paired chunk bootstrap)
# ---------------------------------------------------------------------------------------------------------------

_F = r"(\S+)"
ROW = re.compile(r"^\s*(\d+)\s+" + _F + r"\s+±\s+" + _F + r"\s+" + _F + r"\s+±\s+" + _F + r"\s+" + _F + r"\s+±\s+" + _F
                 + r"\s+" + _F + r"\s+±\s+" + _F + r"\s*%\s+" + _F + r"\s+±\s+" + _F + r"\s*%\s*$", re.M)
MEAN_KLD = re.compile(r"^Mean\s+KLD:\s*([-+]?\d+\.\d+(?:e[-+]?\d+)?)", re.M)
SAME_TOP = re.compile(r"^Same top p:\s*([-+]?\d+\.\d+)", re.M)
P999_KLD = re.compile(r"^99\.9%\s+KLD:\s*([-+]?\d+\.\d+)", re.M)


def fnum(s):
    try:
        return float(s)
    except ValueError:
        return float("nan")


def parse_kl(text):
    """llama-perplexity --kl-divergence log -> dict(n, chunk_kld, chunk_top, final_kld, final_top, p999, complete)."""
    rows = {}
    for m in ROW.finditer(text):
        rows[int(m.group(1))] = (fnum(m.group(6)), fnum(m.group(10)))
    n = 0
    while (n + 1) in rows:
        n += 1
    ck = np.array([rows[k][0] for k in range(1, n + 1)], dtype=np.float64)
    ct = np.array([rows[k][1] for k in range(1, n + 1)], dtype=np.float64)
    mk, mt, mp = MEAN_KLD.search(text), SAME_TOP.search(text), P999_KLD.search(text)
    g = dict(n=n, final_kld=float(mk.group(1)) if mk else None, final_top=float(mt.group(1)) if mt else None,
             p999=float(mp.group(1)) if mp else None, complete=mk is not None and n > 0)
    if g["complete"] and abs(g["final_kld"] - ck[-1]) <= 0.5 * KLD_Q + 1e-9:
        ck = ck.copy()
        ck[-1] = g["final_kld"]          # the 6-decimal summary pins the last running mean
    k = np.arange(1, n + 1, dtype=np.float64)
    prev_k = np.concatenate([[0.0], ck[:-1]]) if n else ck
    prev_t = np.concatenate([[0.0], ct[:-1]]) if n else ct
    g["chunk_kld"] = k * ck - (k - 1) * prev_k
    g["chunk_top"] = k * ct - (k - 1) * prev_t
    return g


def paired_bootstrap(d, idx):
    d = np.asarray(d, dtype=np.float64)
    bm = d[idx].mean(axis=1)
    lo, hi = np.percentile(bm, [2.5, 97.5])
    return dict(mean=float(d.mean()), lo=float(lo), hi=float(hi))


def gate(turbot, fallback, n_boot=DEF_BOOT, seed=DEF_SEED):
    """G5 on two parsed logs of one corpus -> dict(verdict PASS / FAIL / RETRY / INCOMPLETE, kld, top, p999, why)."""
    r = dict(verdict="INCOMPLETE", why=[])
    if not turbot["complete"] or not fallback["complete"] or turbot["n"] != fallback["n"] or turbot["n"] < 4:
        r["why"].append("logs incomplete or with different chunk counts (%d / %d)" % (turbot["n"], fallback["n"]))
        return r
    n = turbot["n"]
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    r["n"] = n
    r["kld"] = paired_bootstrap(turbot["chunk_kld"] - fallback["chunk_kld"], idx)
    r["top"] = paired_bootstrap(turbot["chunk_top"] - fallback["chunk_top"], idx)
    r["p999"] = (turbot["p999"], fallback["p999"])
    kld_below = r["kld"]["hi"] < 0.0
    kld_cross = r["kld"]["lo"] < 0.0 <= r["kld"]["hi"]
    top_ok = r["top"]["lo"] >= TOP_FLOOR
    p999_ok = turbot["p999"] is not None and fallback["p999"] is not None and turbot["p999"] <= fallback["p999"]
    if not kld_below:
        r["why"].append("KLD CI [%+.2e, %+.2e] not below 0" % (r["kld"]["lo"], r["kld"]["hi"]))
    if not top_ok:
        r["why"].append("same-top CI lower bound %+.3f pp below %.2f" % (r["top"]["lo"], TOP_FLOOR))
    if not p999_ok:
        r["why"].append("99.9%% KLD %s vs fallback %s" % r["p999"])
    if kld_below and top_ok and p999_ok:
        r["verdict"] = "PASS"
    elif kld_cross and top_ok and p999_ok:
        r["verdict"] = "RETRY"          # the CI crosses 0: more chunks decide
    else:
        r["verdict"] = "FAIL"
    return r


def delta_text(corpus, g):
    return "%s dKLD %+.2e [%+.2e, %+.2e] dtop %+.3f [%+.3f, %+.3f] p99.9 %.5f/%.5f n%d" % (
        corpus, g["kld"]["mean"], g["kld"]["lo"], g["kld"]["hi"], g["top"]["mean"], g["top"]["lo"], g["top"]["hi"],
        g["p999"][0], g["p999"][1], g["n"])


# ---------------------------------------------------------------------------------------------------------------
# plan stamp
# ---------------------------------------------------------------------------------------------------------------

def stamp(text, fingerprint, verified):
    """The plan text with its '# model:' / '# verified:' lines replaced (inserted after the leading comment block)."""
    lines = [l for l in text.splitlines() if not l.strip().startswith(("# model:", "# verified:"))]
    at = 0
    while at < len(lines) and lines[at].strip().startswith("#"):
        at += 1
    lines[at:at] = ["# model: %s" % fingerprint, "# verified: %s" % verified]
    return "\n".join(lines) + "\n"


def matches_builtin(shape):
    """True when the built-in plan names exactly this model's attention layers and geometry (llama_turbot_plan_choose
    step 3 would take it)."""
    try:
        with io.open(BUILTIN_PLAN, encoding="utf-8") as f:
            P.parse_text(f.read(), shape=shape)
        return True
    except (P.PlanError, OSError):
        return False


# ---------------------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------------------

def arm_argv(args, model, corpus_path, kv, dat, kl, plan=None):
    argv = ["-m", model, "-f", corpus_path, "-c", str(args.ctx), "--chunks", str(args.cur_chunks), "-ngl", "999",
            "-fa", "on", "-b", "2048", "-ub", str(args.ub), "--threads", str(args.threads), "-ctk", kv, "-ctv", kv,
            "--kl-divergence-base", dat, "-lv", "4"]   # libllama info (size, plan, fingerprint lines) shows from -lv 4
    if kl:
        argv.append("--kl-divergence")
    if plan:
        argv += ["--kv-tier-plan", plan]
    return argv


def run_corpus(args, exe, model, out, corpus, plan_path, fallback):
    corpus_path = "%s/%s_corpus.txt" % (args.corpus_dir, corpus)
    if not os.path.isfile(corpus_path):
        raise SystemExit("REFUSED: corpus %s missing" % corpus_path)
    tag = "%s_c%d" % (corpus, args.cur_chunks)
    dat = os.path.join(out, "%s_base.dat" % tag)
    env = {"GGML_DISABLE_VULKAN": "1"}
    res = {}
    for arm, kv, kl, plan in (("base", "f16", False, None), ("fallback", fallback, True, None), ("turbot", "turbot", True, plan_path)):
        log_path = os.path.join(out, "%s_%s.log" % (tag, arm))
        rc, text = run_arm(exe, arm_argv(args, model, corpus_path, kv, dat, kl, plan), env, log_path, args.timeout_min)
        expected_hash = None
        if arm == "turbot":
            fp = FP_LINE.search(text)
            if fp:
                with io.open(plan_path, encoding="utf-8") as f:
                    expected_hash = P.plan_hash(P.parse_text(f.read(), shape=shape_of_fingerprint(fp.group(1).strip(), args.ctx)))
        ok, notes, facts = check_arm_log(text, kv, plan_path if arm == "turbot" else None, expected_hash)
        if rc != 0:
            ok = False
            notes.append("exit code %s" % rc)
        facts.update(ok=ok, notes=notes, log=log_path)
        if kl:
            facts["kl"] = parse_kl(text)
        res[arm] = facts
        print("  %s %s/%s: %s" % ("OK  " if ok else "FAIL", corpus, arm, "; ".join(notes) or "types and plan as asked"))
        if not ok:
            break
    return res, dat


def main(argv=None):
    ap = argparse.ArgumentParser(description="turbot plan quality guard (G5)")
    ap.add_argument("--exe-dir", default=None, help="bin directory of the build (llama-perplexity.exe)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--plan", default=None, help="plan file (default: the automatic plan of --plan-ctx / --plan-np)")
    ap.add_argument("--plan-ctx", type=int, default=262144, help="n_ctx of the production server, for the automatic plan")
    ap.add_argument("--plan-np", type=int, default=1, help="slots of the production server, for the automatic plan")
    ap.add_argument("--plan-unified", action="store_true", help="the production server runs --kv-unified")
    ap.add_argument("--budget", choices=("fallback", "turbo5p"), default="fallback", help="budget of the automatic plan")
    ap.add_argument("--fallback", default=None, help="fallback KV type (default: turbo5p, or turbo4 for 256-value rows)")
    ap.add_argument("--allow-larger", action="store_true", help="accept a turbot cache larger than the fallback's")
    ap.add_argument("--corpora", default="code,prose")
    ap.add_argument("--corpus-dir", default=CORPUS_DIR)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--retry-chunks", type=int, default=16)
    ap.add_argument("--ub", type=int, default=512)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--timeout-min", type=int, default=60)
    ap.add_argument("--boot", type=int, default=DEF_BOOT)
    ap.add_argument("--seed", type=int, default=DEF_SEED)
    ap.add_argument("--sidecar", action="store_true", help="also copy the stamped plan to <model>.turbot.plan")
    ap.add_argument("--force", action="store_true", help="stamp even when the built-in plan already names the model")
    ap.add_argument("--keep", action="store_true", help="keep the .dat logits files")
    ap.add_argument("--selftest", action="store_true", help="CPU only: parser and decision on synthetic logs")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    if np is None:
        print("REFUSED: numpy is needed for the statistics")
        return 2
    if not args.exe_dir or not args.model:
        ap.error("--exe-dir and --model are required")
    if busy():
        print("REFUSED: a llama/ggml process is already running: %s" % busy())
        return 2

    exe = os.path.join(args.exe_dir, "llama-perplexity.exe" if os.name == "nt" else "llama-perplexity")
    if not os.path.isfile(exe):
        print("REFUSED: %s not found" % exe)
        return 2
    name = os.path.splitext(os.path.basename(args.model))[0]
    out = os.path.join(args.out, name)
    os.makedirs(out, exist_ok=True)

    info = P.gguf_info(args.model)
    shape_prod = P.ctx_shape(info["layers"], args.plan_ctx, args.plan_np, args.plan_unified)
    if args.plan is None:
        args.plan = os.path.join(out, name + P_SUFFIX)
        try:
            text = P.auto_plan(shape_prod, args.budget == "turbo5p")
        except P.PlanError as e:
            print("REFUSED: no automatic plan for %s: %s" % (name, e))
            return 1
        with io.open(args.plan, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        print("automatic plan for %s written to %s" % (name, args.plan))
    args.plan = os.path.abspath(args.plan)

    rows = [P.geom_row_elems(P.geom_flags(d, h)) for d, h in info["layers"].values()]
    fallback = args.fallback or ("turbo4" if P.cache_fallback(rows) == "turbo4" else "turbo5p")

    builtin = matches_builtin(P.make_shape(info["layers"]))
    if builtin and not args.force:
        print("note: the built-in plan names this model's attention layers; the plan will be measured but not stamped (--force)")

    results, dats, fails = {}, [], []
    for corpus in [c for c in args.corpora.split(",") if c]:
        print("[%s]" % corpus)
        args.cur_chunks = args.chunks
        while True:
            res, dat = run_corpus(args, exe, args.model, out, corpus, args.plan, fallback)
            dats.append(dat)
            # the base logits of one corpus run are tens of GB and no later arm reads them: free E: before the next run
            if not args.keep and os.path.isfile(dat):
                os.remove(dat)
            if not all(res.get(a, {}).get("ok") for a in ("base", "fallback", "turbot")):
                g = dict(verdict="FAIL", why=["an arm failed its log check (see above)"])
            else:
                g = gate(res["turbot"]["kl"], res["fallback"]["kl"], args.boot, args.seed)
                t_mib, f_mib = res["turbot"].get("size_mib"), res["fallback"].get("size_mib")
                if t_mib is not None and f_mib is not None and t_mib > f_mib and not args.allow_larger:
                    g["verdict"] = "FAIL"
                    g["why"].append("turbot cache %.2f MiB above the fallback's %.2f MiB (--allow-larger)" % (t_mib, f_mib))
            if g["verdict"] == "RETRY" and args.cur_chunks < args.retry_chunks:
                print("  KLD CI crosses 0 at %d chunks: again at %d" % (args.cur_chunks, args.retry_chunks))
                args.cur_chunks = args.retry_chunks
                continue
            if g["verdict"] == "RETRY":
                g["verdict"] = "FAIL"
            break
        results[corpus] = dict(gate=g, arms={a: {k: v for k, v in f.items() if k != "kl"} for a, f in res.items()})
        print("  %s %s: %s" % (g["verdict"], corpus, delta_text(corpus, g) if "kld" in g else "; ".join(g["why"])))
        if g["verdict"] != "PASS":
            fails.append(corpus)
            print("    why: %s" % "; ".join(g["why"]))

    verdict = "PASS" if not fails and results else "FAIL"
    summary = dict(model=args.model, plan=args.plan, fallback=fallback, verdict=verdict, builtin=builtin,
                   results={c: dict(gate={k: v for k, v in r["gate"].items()}, arms=r["arms"]) for c, r in results.items()})

    stamped = False
    if verdict == "PASS":
        fps = {r["arms"]["turbot"].get("fingerprint") for r in results.values()}
        builds = {r["arms"]["turbot"].get("build") for r in results.values()}
        if len(fps) != 1 or None in fps:
            print("not stamped: the turbot arms logged %s as the model fingerprint" % sorted(map(str, fps)))
        elif builtin and not args.force:
            print("not stamped: the built-in plan names this model (a sidecar would replace it); --force stamps anyway")
        else:
            fp = fps.pop()
            verified = "%s %s %s" % (datetime.date.today().isoformat(), "/".join(sorted(map(str, builds))),
                                     "; ".join(delta_text(c, r["gate"]) for c, r in results.items()))
            with io.open(args.plan, encoding="utf-8") as f:
                text = f.read()
            text = stamp(text, fp, verified)
            with io.open(args.plan, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
            stamped = True
            print("stamped %s\n  # model: %s\n  # verified: %s" % (args.plan, fp, verified))
            if args.sidecar:
                side = args.model + P_SUFFIX
                if os.path.isfile(side):
                    shutil.copyfile(side, side + ".bak")
                shutil.copyfile(args.plan, side)
                print("sidecar %s written" % side)
    summary["stamped"] = stamped

    if not args.keep:
        for d in dats:
            if os.path.isfile(d):
                os.remove(d)
    with io.open(os.path.join(out, "guard_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=str)
    print("\nRESULT %s (%s)" % (verdict, ", ".join("%s %s" % (c, r["gate"]["verdict"]) for c, r in results.items())))
    return 0 if verdict == "PASS" else 1



# ---------------------------------------------------------------------------------------------------------------
# self-test (CPU only)
# ---------------------------------------------------------------------------------------------------------------

def _synth_kl(kld, top, p999):
    lines, ck, ct = ["chunk  PPL  ln(PPL(Q)/PPL(base))  KL Divergence  dp RMS  Same top p"], 0.0, 0.0
    for k, (x, s) in enumerate(zip(kld, top), 1):
        ck += (x - ck) / k
        ct += (s - ct) / k
        lines.append("%4d       1.5000 ±    0.0080      0.00010 ±    0.00030      %8.5f ±    0.00004     1.100 ±  0.040 %%    %6.3f ±  0.040 %%" % (k, ck, ct))
    lines += ["Mean    KLD:   %.6f ±   0.000012" % ck, "99.9%%   KLD:   %.6f" % p999, "Same top p: %.3f ± 0.016 %%" % ct]
    return "\n".join(lines) + "\n"


def selftest():
    if np is None:
        print("selftest needs numpy")
        return 2
    fails = []

    def check(cond, what):
        print("  %s %s" % ("ok  " if cond else "FAIL", what))
        if not cond:
            fails.append(what)

    rng = np.random.default_rng(3)
    n = 8
    fb = 0.0016 + rng.normal(0, 5e-5, n)
    tops = 99.0 + rng.normal(0, 0.03, n)
    good = parse_kl(_synth_kl(fb - 3e-4 + rng.normal(0, 2e-5, n), tops + 0.02, 0.05))
    base = parse_kl(_synth_kl(fb, tops, 0.06))
    check(good["n"] == n and good["complete"] and good["p999"] == 0.05, "KL log parse")
    check(gate(good, base)["verdict"] == "PASS", "turbot below the fallback passes")
    worse = parse_kl(_synth_kl(fb + 3e-4, tops, 0.05))
    check(gate(worse, base)["verdict"] == "FAIL", "turbot above the fallback fails")
    cross = parse_kl(_synth_kl(fb + rng.normal(0, 2e-4, n), tops, 0.05))
    check(gate(cross, base)["verdict"] in ("RETRY", "FAIL"), "a CI across 0 asks for more chunks or fails")
    tail = parse_kl(_synth_kl(fb - 3e-4, tops, 0.07))
    check(gate(tail, base)["verdict"] == "FAIL", "a worse 99.9% KLD fails")
    toplow = parse_kl(_synth_kl(fb - 3e-4, tops - 0.2, 0.05))
    check(gate(toplow, base)["verdict"] == "FAIL", "a same-top loss beyond 0.05 pp fails")
    short = parse_kl(_synth_kl(fb[:5], tops[:5], 0.05))
    check(gate(short, base)["verdict"] == "INCOMPLETE", "different chunk counts are incomplete")

    log = ("build: 11093 (0fc83cc8d) with MSVC\n"
           "llama_turbot_plan_load: turbot: model fingerprint: arch=qwen35 basename=ornith-1.5 size=9B layers=3,7,11,15,19,23,27,31 geom=256x4\n"
           "llama_kv_cache: turbot plan E:/x.plan: 8 layers, old bits 4.750 (sum 304), young pool 16512 cells (258 granules), cap 16384, hash 0x7de0f0bf9a53a865\n"
           "llama_kv_cache: size = 1286.30 MiB ( 32768 cells,   8 layers,  1/1 seqs), K (turbot):  600.00 MiB, V (turbot):  600.00 MiB, young pool:   86.30 MiB\n")
    ok, notes, facts = check_arm_log(log, "turbot", "E:/x.plan", 0x7de0f0bf9a53a865)
    check(ok and facts["fingerprint"].startswith("arch=qwen35") and facts["build"] == "b11093-0fc83cc8d", "turbot arm log check")
    ok, notes, _ = check_arm_log(log.replace("hash 0x7de0f0bf9a53a865", "hash 0x0000000000000001"), "turbot", "E:/x.plan", 0x7de0f0bf9a53a865)
    check(not ok, "a wrong plan hash fails")
    down = ("llama_init_from_model: KV cache type for K and V: turbot -> turbo5p (turbot: SWA caches are unsupported)\n"
            "llama_kv_cache: size =  656.00 MiB ( 32768 cells,  16 layers,  1/1 seqs), K (turbo5p):  328.00 MiB, V (turbo5p):  328.00 MiB\n")
    ok, notes, _ = check_arm_log(down, "turbot")
    check(not ok and any("downgraded" in x for x in notes), "a silent downgrade fails the arm")
    ok, _, _ = check_arm_log("llama_kv_cache: size =  328.00 MiB ( 32768 cells,  10 layers,  1/1 seqs), K (turbo5p512):  164.00 MiB, V (turbo5p512):  164.00 MiB\n", "turbo5p")
    check(ok, "turbo5p512 counts as the turbo5p fallback")

    shape = shape_of_fingerprint("arch=qwen35 basename=ornith-1.5 size=9B layers=3,7,11,15,19,23,27,31 geom=256x4", 32768)
    check(sorted(shape["layers"]) == list(range(3, 32, 4)) and shape["layers"][3] == (256, 4), "shape from the fingerprint")
    text = P.auto_plan(P.make_shape(list(range(3, 32, 4)), kv_size=262144))
    st = stamp(text, "arch=a basename=b size=c layers=3 geom=256x4", "2026-09-22 b1 code ok")
    st2 = stamp(st, "arch=a basename=b size=c layers=3 geom=256x4", "2026-09-23 b2 code ok")
    check(st2.count("# model:") == 1 and st2.count("# verified:") == 1 and "2026-09-23" in st2, "stamp replaces old stamp lines")
    shape9 = P.make_shape(list(range(3, 32, 4)), kv_size=262144)
    check(P.plan_hash(P.parse_text(st2, shape=shape9)) == P.plan_hash(P.parse_text(text, shape=shape9)), "stamp keeps the plan hash")
    check(matches_builtin(P.make_shape(P.QWEN38_ATTN_LAYERS)) and not matches_builtin(shape9), "built-in plan match")

    print("[selftest] %s" % ("PASS" if not fails else "FAIL: " + "; ".join(fails)))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
