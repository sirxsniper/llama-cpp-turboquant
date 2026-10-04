#!/usr/bin/env python3
# [TAG_FN_SP0] Paired A/B/A/B llama-server runner for Flash-Next (E:/turbot-gates/flashnext/PLAN.md 7.5).
#
# Rules it enforces (it refuses to start otherwise):
#   - E:/turbot-gates/imp/SIX_DONE exists, E:/turbot-gates/STOP_GPU and E:/turbot-gates/REBOOT_NEEDED.txt do not
#   - ONE-process check: no process matching llama|ggml|test-backend|test-turbot|compute-sanitizer is alive
#   - commit charge at most --max-commit-gb (50), port 8091 free, no nvlddmkm event since boot
#   - it starts and stops only its own server (never Stop-Process by name), always on port 8091
#
# Arms file (JSON): [{"name": "off", "env": {}, "args": []}, {"name": "async", "env": {"GGML_SCHED_SPLIT_ASYNC": "1"}}]
# Every arm runs with --base-args (default: the 131K baseline of PLAN 7.4) plus its own args, in the order
# rep0: A B ..., rep1: A B ..., so drift hits every arm equally. Only these within-run comparisons count.
#
#   python tools/qwen4exp/fn_bench.py --arms arms.json --reps 3 --tag sp1a
#   python tools/qwen4exp/fn_bench.py --arms arms.json --identity --tag sp1a_id     (sha1 of 4 x 512 greedy tokens, MTP off)
#   python tools/qwen4exp/fn_bench.py --arms arms.json --streams 2 --tag sp1a_2s
#
# Per request it records t/s, ms/step, tau (tokens per target step), draft acceptance and the nvidia-smi peak. Results go
# to <out>/<tag>.jsonl and a summary to <out>/<tag>.txt.
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = "E:/turbot-gates"
PORT = 8091
M = "D:/Projects/LocalAI/models/"
MODEL_A = M + "Qwen3.8-Flash-Next-UD-Q4_K_XL-MTP-00001-of-00005.gguf"
BIN = "D:/Projects/LocalAI/source-build/wt-fsync/build-fsync/bin"  # [TAG_SYNC_1004] the synced branch (flashnext/synced)

# [TAG_FN_MERGE] turbot KV only on Flash-Next (owner, 2026-09-28: no q4 / turbo4 KV); ncmoe 39 = the F0 fit (32K, MTP on)
BASE_131K = ("-c 131072 -ngl 99 --n-cpu-moe 39 -fit off -fa on -ctk turbot -ctv turbot -b 2048 -ub 512 -t 16 "
             "--cpu-mask 55555555 --cpu-strict 1 --prio 2 --parallel 1 "
             "--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.5")
BASE_ENV = {"SPEC_DFT_UBATCH": "128"}

CODE_PROMPT = ("Write a complete, well commented Python module that implements a persistent key-value store with a "
               "write-ahead log, crash recovery, compaction and an LRU read cache. Include type hints, a command line "
               "interface and unit tests.")
PROSE_PROMPT = ("Write a long, detailed story about a lighthouse keeper on a remote northern island who finds a sealed "
                "letter washed ashore, and how it changes the lives of three families over one winter.")
IDENTITY_PROMPTS = [
    CODE_PROMPT,
    PROSE_PROMPT,
    "Explain, step by step, how a CPU executes a branch misprediction recovery, then summarise in a table.",
    "Translate the following into French, German and Spanish, then explain the grammar differences: "
    "'The committee postponed the decision because the report arrived late.'",
]

SPEC_FLAGS = {"--spec-type", "--spec-draft-n-max", "--spec-draft-p-min", "--spec-draft-n-min"}

# --depth: every prompt starts with the same ~N-token prefix (prose corpus text), cached by the server after the warm-up
DEPTH_PREFIX = ""
CACHE_PROMPT = False


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def ps(cmd):
    r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True)
    return r.stdout.strip()


def preflight(max_commit_gb, allow_no_six=False):
    errs = []
    if not allow_no_six and not os.path.exists(ROOT + "/imp/SIX_DONE"):
        errs.append("E:/turbot-gates/imp/SIX_DONE is missing (GPU not released)")
    for f in ("/STOP_GPU", "/REBOOT_NEEDED.txt"):
        if os.path.exists(ROOT + f):
            errs.append("%s exists" % (ROOT + f))
    busy = ps("Get-Process | Where-Object { $_.ProcessName -match 'llama|ggml|test-backend|test-turbot|compute-sanitizer' }"
              " | ForEach-Object { \"$($_.ProcessName) $($_.Id)\" }")
    if busy:
        errs.append("ONE-process check failed: " + busy.replace("\n", ", "))
    commit = ps("$o = Get-CimInstance Win32_OperatingSystem; [math]::Round(($o.TotalVirtualMemorySize - $o.FreeVirtualMemory)/1MB, 1)")
    try:
        if float(commit) > max_commit_gb:
            errs.append("commit %s GB > %d GB" % (commit, max_commit_gb))
    except ValueError:
        errs.append("could not read the commit charge (%r)" % commit)
    # [TAG_FN_MERGE] as in the speed stage's copy (2026-09-29): the gpucorr_guard.ps1 -Mode pre rule. The boot time is
    # the newer of LastBootUpTime and the latest Kernel-Boot 27 event (Fast Startup), and events at or before the
    # newest FAULT line of E:/turbot-gates/imp/gpu_faults.txt are acknowledged (recorded crashes, handled by the
    # FAULT/FIXED protocol). Without it every event since a Fast Startup boot refused the run.
    ev = ps("$b = (Get-CimInstance Win32_OperatingSystem).LastBootUpTime; "
            "$kb = Get-WinEvent -FilterHashtable @{LogName='System';ProviderName='Microsoft-Windows-Kernel-Boot';Id=27} -MaxEvents 1 -ErrorAction SilentlyContinue; "
            "if ($kb -and $kb.TimeCreated -gt $b) { $b = $kb.TimeCreated }; "
            "$f = 'E:/turbot-gates/imp/gpu_faults.txt'; $ack = $null; "
            "if (Test-Path $f) { foreach ($l in Get-Content $f) { if ($l -match '^FAULT\\|([^|]+)\\|') { $d = [datetime]::Parse($Matches[1]); if (-not $ack -or $d -gt $ack) { $ack = $d } } } }; "
            "if ($ack -and $ack.AddSeconds(1) -gt $b) { $b = $ack.AddSeconds(1) }; "
            "@(Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='nvlddmkm'; StartTime=$b} -ErrorAction SilentlyContinue).Count")
    if ev not in ("", "0"):
        errs.append("%s nvlddmkm events since boot: check event 153 and reboot first" % ev)
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", PORT)) == 0:
            errs.append("port %d is in use" % PORT)
    return errs


class VramSampler(threading.Thread):
    def __init__(self, period=0.5):
        super().__init__(daemon=True)
        self.period, self.peak, self.stop_flag = period, 0, False

    def run(self):
        while not self.stop_flag:
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout
                self.peak = max(self.peak, max(int(x) for x in out.split() if x.strip().isdigit()))
            except Exception:  # noqa: BLE001
                pass
            time.sleep(self.period)


class Server:
    def __init__(self, bin_dir, model, args, env, log_path):
        self.bin_dir, self.model, self.args, self.log_path = bin_dir, model, args, log_path
        self.env = dict(os.environ)
        self.env.update(env)
        self.proc = None

    def start(self, timeout=1800):
        exe = os.path.join(self.bin_dir, "llama-server.exe")
        argv = [exe, "-m", self.model, "--host", "127.0.0.1", "--port", str(PORT), "--no-webui"] + self.args
        self.logf = open(self.log_path, "w", encoding="utf-8", errors="replace")
        self.logf.write(" ".join(argv) + "\n" + json.dumps({k: v for k, v in self.env.items() if k.startswith(
            ("LLAMA_", "GGML_", "TURBO_", "SPEC_", "CUDA_"))}) + "\n")
        self.logf.flush()
        self.proc = subprocess.Popen(argv, env=self.env, stdout=self.logf, stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError("server exited with %s, see %s" % (self.proc.returncode, self.log_path))
            try:
                with urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=5) as r:
                    if r.status == 200:
                        return time.time() - t0
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            time.sleep(2)
        raise RuntimeError("server not healthy after %d s" % timeout)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(30)
        if getattr(self, "logf", None):
            self.logf.close()

    def text(self):
        try:
            with open(self.log_path, encoding="utf-8", errors="replace") as f:
                return f.read()
        except OSError:
            return ""


def post(path, body, timeout=3600):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (PORT, path), data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def chat_prompt(user):
    if DEPTH_PREFIX:
        user = "Reference material:" + DEPTH_PREFIX + "Task: " + user
    r = post("/apply-template", {"messages": [{"role": "user", "content": user}],
                                 "chat_template_kwargs": {"enable_thinking": False}})
    return r["prompt"]


def complete(prompt, n, greedy, seed=1234, tokens=False):
    body = {"prompt": prompt, "n_predict": n, "ignore_eos": True, "cache_prompt": CACHE_PROMPT, "seed": seed,
            "return_tokens": tokens}
    if greedy:
        body.update({"temperature": 0.0, "top_k": 1})
    else:
        body.update({"temperature": 1.0, "top_p": 0.95, "top_k": 20})
    t0 = time.time()
    r = post("/completion", body)
    wall = time.time() - t0
    t = r.get("timings", {})
    n_pred = t.get("predicted_n", 0)
    d_n, d_acc = t.get("draft_n", 0) or 0, t.get("draft_n_accepted", 0) or 0
    steps = max(1, n_pred - d_acc)
    return {"tps": t.get("predicted_per_second", 0.0), "n": n_pred, "ms": t.get("predicted_ms", 0.0),
            "ms_step": t.get("predicted_ms", 0.0) / steps, "tau": n_pred / steps,
            "accept": (d_acc / d_n) if d_n else None, "draft_n": d_n, "prompt_ms": t.get("prompt_ms", 0.0),
            "prompt_n": t.get("prompt_n", 0), "wall": wall, "tokens": r.get("tokens", []), "content": r.get("content", "")}


def split_args(s):
    return [a for a in re.split(r"\s+", s.strip()) if a]


def strip_spec(args):
    out, skip = [], 0
    for a in args:
        if skip:
            skip -= 1
            continue
        if a in SPEC_FLAGS:
            skip = 1
            continue
        out.append(a)
    return out


def run_arm(arm, a, rep, out):
    args = split_args(a.base_args) + arm.get("args", [])
    if a.identity and not a.identity_keep_spec:
        args = strip_spec(args)
    env = dict(BASE_ENV)
    env.update(arm.get("env", {}))
    logp = os.path.join(a.out, "%s_%s_r%d.log" % (a.tag, arm["name"], rep))
    srv = Server(a.bin, a.model, args, env, logp)
    vs = VramSampler()
    rec = {"arm": arm["name"], "rep": rep, "args": args, "env": env}
    try:
        rec["load_s"] = srv.start()
        vs.start()
        code_p, prose_p = chat_prompt(CODE_PROMPT), chat_prompt(PROSE_PROMPT)
        if a.identity:
            digests = []
            for i, p in enumerate(IDENTITY_PROMPTS):
                r = complete(chat_prompt(p), a.n, greedy=True, tokens=True)
                digests.append(hashlib.sha1(json.dumps(r["tokens"]).encode()).hexdigest())
                rec.setdefault("tokens", []).append(r["tokens"])
            rec["sha1"] = digests
        else:
            complete(code_p, a.warmup, greedy=True)          # warm expert pages and CUDA graphs
            for kind, prompt, greedy in (("code", code_p, True), ("prose", prose_p, False)):
                if a.streams <= 1:
                    rec[kind] = complete(prompt, a.n, greedy, seed=1234 + rep)
                else:
                    res = [None] * a.streams
                    def one(i):
                        res[i] = complete(prompt, a.n, greedy, seed=1234 + rep * 16 + i)
                    t0 = time.time()
                    th = [threading.Thread(target=one, args=(i,)) for i in range(a.streams)]
                    for t in th:
                        t.start()
                    for t in th:
                        t.join()
                    wall = time.time() - t0
                    rec[kind] = {"tps": sum(r["n"] for r in res) / wall, "per_stream": res,
                                 "accept": statistics.mean([r["accept"] for r in res if r["accept"] is not None] or [0])}
                for r in ([rec[kind]] + rec[kind].get("per_stream", [])):
                    r.pop("tokens", None)
                    r.pop("content", None)
        txt = srv.text()
        rec["reused"] = re.findall(r"graphs reused[^\n]*", txt)[-1:] if "graphs reused" in txt else []
        rec["moe_hot"] = re.findall(r"moe-hot[^\n]*hit[^\n]*", txt)[-3:]
    finally:
        vs.stop_flag = True
        srv.stop()
    rec["vram_peak"] = vs.peak
    txt = srv.text()
    if re.search(r"CUDA error|illegal memory access|GGML_ASSERT|ggml_abort", txt):
        rec["error"] = "fatal message in " + logp
    out.write(json.dumps(rec) + "\n")
    out.flush()
    return rec


def summarise(recs, arms, a):
    lines = []
    if a.identity:
        ref = None
        for r in recs:
            if "sha1" not in r:
                continue
            if ref is None:
                ref = r
            same = r["sha1"] == ref["sha1"]
            detail = ""
            if not same:
                for i, (x, y) in enumerate(zip(r["tokens"], ref["tokens"])):
                    k = next((j for j, (p, q) in enumerate(zip(x, y)) if p != q), None)
                    if k is not None:
                        detail = " (prompt %d differs from token %d)" % (i, k)
                        break
            lines.append("%-16s rep %d %s%s" % (r["arm"], r["rep"], "IDENTICAL" if same else "DIFFERENT", detail))
        lines.append("IDENTITY %s" % ("PASS" if all(r.get("sha1") == ref["sha1"] for r in recs if ref) else "FAIL"))
        return lines
    for kind in ("code", "prose"):
        base = None
        for arm in arms:
            rs = [r for r in recs if r["arm"] == arm["name"] and kind in r]
            if not rs:
                continue
            tps = [r[kind]["tps"] for r in rs]
            med = statistics.median(tps)
            acc = [r[kind].get("accept") for r in rs if r[kind].get("accept") is not None]
            ms = [r[kind].get("ms_step") for r in rs if r[kind].get("ms_step")]
            line = "%-5s %-16s t/s median %7.2f  [%s]  ms/step %s  accept %s  vram %d" % (
                kind, arm["name"], med, " ".join("%.2f" % x for x in tps),
                ("%.2f" % statistics.median(ms)) if ms else "-", ("%.3f" % statistics.median(acc)) if acc else "-",
                max(r["vram_peak"] for r in rs))
            if base is None:
                base = rs
            else:
                ratios = [r[kind]["tps"] / b[kind]["tps"] for r, b in zip(rs, base) if b[kind]["tps"]]
                if ratios:
                    line += "  paired vs %s: %+.1f%% (per rep %s)" % (arms[0]["name"], 100 * (statistics.median(ratios) - 1),
                                                                   " ".join("%+.1f" % (100 * (x - 1)) for x in ratios))
            lines.append(line)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True, help="JSON file: list of {name, env, args}")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=800)
    ap.add_argument("--depth", type=int, default=0, help="prefix every prompt with ~N tokens of corpus text (context depth)")
    ap.add_argument("--depth-file", default="E:/kv-bar-s0/prose_corpus.txt")
    ap.add_argument("--streams", type=int, default=1)
    ap.add_argument("--identity", action="store_true", help="sha1 of 4 x --n greedy tokens per arm; spec flags removed")
    ap.add_argument("--identity-keep-spec", action="store_true")
    ap.add_argument("--model", default=MODEL_A)
    ap.add_argument("--bin", default=BIN)
    ap.add_argument("--base-args", default=BASE_131K)
    ap.add_argument("--out", default=ROOT + "/flashnext/bench")
    ap.add_argument("--max-commit-gb", type=int, default=50)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if a.depth > 0:
        global DEPTH_PREFIX, CACHE_PROMPT
        with open(a.depth_file, encoding="utf-8", errors="replace") as f:
            text = f.read(a.depth * 4)   # ~4 characters per token
        DEPTH_PREFIX = chr(10) + text + chr(10) + chr(10)
        CACHE_PROMPT = True
    errs = preflight(a.max_commit_gb)
    if errs:
        for e in errs:
            log("REFUSED: " + e)
        return 2
    with open(a.arms, encoding="utf-8") as f:
        arms = json.load(f)
    os.makedirs(a.out, exist_ok=True)
    reps = 1 if a.identity else a.reps
    recs = []
    with open(os.path.join(a.out, a.tag + ".jsonl"), "a", encoding="utf-8") as out:
        for rep in range(reps):
            for arm in arms:
                errs = preflight(a.max_commit_gb)
                if errs:
                    log("STOP before %s rep %d: %s" % (arm["name"], rep, "; ".join(errs)))
                    return 2
                log("rep %d arm %s" % (rep, arm["name"]))
                r = run_arm(arm, a, rep, out)
                recs.append(r)
                if "error" in r:
                    log("STOP: " + r["error"])
                    return 3
                if not a.identity:
                    log("  code %.2f t/s (accept %s)  prose %.2f t/s  vram %d" % (
                        r["code"]["tps"], r["code"].get("accept"), r["prose"]["tps"], r["vram_peak"]))
    lines = summarise(recs, arms, a)
    with open(os.path.join(a.out, a.tag + ".txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
