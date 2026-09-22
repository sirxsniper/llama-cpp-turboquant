# [TAG_TURBOT] State blob round trips on a real llama-server (docs/turbot/SPEC.md 9.10, 11.4).
#
# Port 8091, never 8080 (Jarvis records whatever serves on 8080 as real usage). ONE GPU process: refuses to start while
# any llama/ggml/test-backend process runs, and stops its own server before the next arm starts.
#
#   python tools/turbot/blob_roundtrip.py --exe <build>/bin/llama-server.exe [--plan docs/turbot/plans/turbot-default.plan]
#          [--model D:/Projects/LocalAI/models/Qwen3.8-27B-UD-Q5_K_XL.gguf] [--prompt-tokens 20000] [--only a,b,...]
#
# Arms (each PASS/FAIL, exit 0 only if all pass):
#   save_restore      turbot: greedy reference on slot 0, save slot 0, erase it, restore the file into slot 2 (a different
#                     slot id, whose counter differs), rerun the prompt on slot 2: prompt_n small (cache hit) and tokens
#                     identical to the reference
#   prompt_cache      turbot: reference on slot 1, push it out with another long prompt on slot 1, rerun on slot 3: the
#                     server prompt cache restores the state into another slot; tokens identical, prompt_n small
#   park_resume       turbot: TURBO_POOL_DEBUG_FAIL=gen@17 forces the pool-full park and resume ([TAG_POOL_PREEMPT]); tokens
#                     identical to a run without the hook, the log shows a park and a resume
#   refuse_t5_into_t  save a slot on a turbo5p server, restore it on a turbot server: the restore fails cleanly, the server
#                     stays healthy and still answers, the log names the type mismatch
#   refuse_t_into_t5  the reverse
#   streams_equal     [TAG_TURBOT_ANY_STREAMS] the same greedy prompt on slot 0 of a --kv-unified server and on slot 2 of a
#                     server with one KV stream per slot (-np N --no-kv-unified): tokens identical (below the per-stream
#                     young quota every cell is young in both, and stream 2 exercises every per-stream offset)
#
# The server command comes from the Jarvis production profile through the scratchpad harness (acceptab.build_cmd_and_env,
# port 8091) when --harness-dir holds it, otherwise from a minimal command (-np 4 --kv-unified -c 262144 -fa on). The cache
# type and plan are forced here; the last -ctk/-ctv on the command line wins.
#
# [TAG_TURBOT_ANY_STREAMS] --np N sets the slot count, --non-unified runs every arm with one KV stream per slot
# (--no-kv-unified; turbot keeps one tier per stream unless LLAMA_TURBOT_MULTI_STREAM=0):
#   python tools/turbot/blob_roundtrip.py --exe <build>/bin/llama-server.exe --np 4 --non-unified
# Every turbot server must log its "turbot plan" line: a server that fell back to another KV type fails the arm instead
# of testing that type.
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

PORT = 8091
HARNESS_DEFAULT = r"C:\Users\xSniper\AppData\Local\Temp\claude\D--Projects\8efd145b-de13-4771-ae4d-110a53af2fef\scratchpad"
MODEL_DEFAULT = "D:/Projects/LocalAI/models/Qwen3.8-27B-UD-Q5_K_XL.gguf"
HERE = os.path.dirname(os.path.abspath(__file__))
PLAN_DEFAULT = os.path.normpath(os.path.join(HERE, "..", "..", "docs", "turbot", "plans", "turbot-default.plan"))
BASE = "http://127.0.0.1:%d" % PORT
assert PORT != 8080


def busy():
    out = subprocess.run(["tasklist"], capture_output=True, text=True).stdout.lower()
    return [l.split()[0] for l in out.splitlines() if any(k in l for k in ("llama", "ggml", "test-backend"))]


def http(path, body=None, timeout=3600):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode() or "{}")
        except Exception:
            return e.code, {}


def base_cmd(args, unified=None):
    """unified: None keeps the profile's KV layout (or --non-unified), True / False force --kv-unified / --no-kv-unified."""
    try:
        sys.path.insert(0, args.harness_dir)
        import acceptab as A  # noqa: E402
        cmd, env = A.build_cmd_and_env({"name": "blob", "env": {}, "args": []})
        cmd[0] = args.exe
    except Exception as e:
        print("note: production profile unavailable (%s), using a minimal command" % e)
        cmd = [args.exe, "-m", args.model, "-ngl", "999", "-fa", "on", "-c", "262144", "-np", "4", "--kv-unified",
               "-b", "2048", "-ub", "512", "--host", "127.0.0.1"]
        env = dict(os.environ)
    if unified is None and args.non_unified:
        unified = False
    drop2 = ["--port", "-ctk", "-ctv", "--cache-type-k", "--cache-type-v", "--kv-tier-plan", "--slot-save-path"]
    drop1 = []
    if args.np is not None:
        drop2 += ["-np", "--parallel"]
    if unified is not None:
        drop1 += ["-kvu", "--kv-unified", "-no-kvu", "--no-kv-unified"]
    out, i = [], 0
    while i < len(cmd):                       # drop any port / cache type / plan / layout the profile sets
        if cmd[i] in drop2:
            i += 2
            continue
        if cmd[i] in drop1:
            i += 1
            continue
        out.append(cmd[i])
        i += 1
    if args.np is not None:
        out += ["-np", str(args.np)]
    if unified is not None:
        out += ["--kv-unified" if unified else "--no-kv-unified"]
        env = dict(env)
        env.pop("LLAMA_ARG_KV_UNIFIED", None)
    return out + ["-lv", "4", "--port", str(PORT)], env   # libllama info (check_kv's "turbot plan" line) shows from -lv 4


class Server:
    def __init__(self, args, kv, extra_env=None, tag="srv", unified=None):
        self.args, self.kv, self.tag = args, kv, tag
        self.extra_env = extra_env or {}
        self.unified = unified
        self.streams_line = ""
        self.p = None
        self.log_path = os.path.join(args.work, "%s_%s.log" % (tag, kv))

    def __enter__(self):
        if busy():
            raise SystemExit("REFUSED: a llama/ggml process is already running: %s" % busy())
        cmd, env = base_cmd(self.args, self.unified)
        cmd += ["-ctk", self.kv, "-ctv", self.kv, "--slot-save-path", self.args.work + os.sep]
        if self.kv == "turbot":
            cmd += ["--kv-tier-plan", self.args.plan]
        env = dict(env)
        env.update(self.extra_env)
        self.log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        print("  start %s: %s" % (self.tag, " ".join(cmd)))
        self.p = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, env=env)
        t0 = time.time()
        while time.time() - t0 < 900:
            if self.p.poll() is not None:
                raise RuntimeError("server exited with %s, see %s" % (self.p.returncode, self.log_path))
            ready = False
            try:
                st, js = http("/health", timeout=3)
                ready = st == 200
            except Exception:
                pass
            if ready:
                try:
                    self.check_kv()
                except RuntimeError:
                    self.__exit__()
                    raise
                return self
            time.sleep(3)
        raise RuntimeError("server not ready after 900 s")

    def check_kv(self):
        """[TAG_TURBOT_ANY_STREAMS] a turbot server that silently runs another KV type would test that type"""
        if self.kv != "turbot":
            return
        log = self.log_text()
        if "turbot plan " not in log:
            down = re.findall(r"KV cache type for [^\n]*", log)
            raise RuntimeError("turbot server has no 'turbot plan' line (downgraded: %s), see %s" % (down[:2], self.log_path))
        m = re.search(r"turbot plan .*young pool \d+ cells \(\d+ granules\)(, \d+ streams x \d+ cells)?", log)
        self.streams_line = m.group(1) if m and m.group(1) else ""

    def __exit__(self, *exc):
        if self.p and self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(60)
            except subprocess.TimeoutExpired:
                self.p.kill()
        for _ in range(120):
            if not [x for x in busy() if "server" in x]:
                break
            time.sleep(1)
        time.sleep(2)
        self.log.close()

    def log_text(self):
        self.log.flush()
        with open(self.log_path, encoding="utf-8", errors="replace") as f:
            return f.read()


def filler(seed, n_words):
    import random
    rnd = random.Random(seed)
    syll = ["ka", "lo", "mer", "tan", "vi", "sor", "del", "pra", "nu", "gim", "ro", "shi", "bel", "tor", "an", "que"]
    vocab = sorted({"".join(rnd.choice(syll) for _ in range(rnd.randint(1, 3))) for _ in range(3000)})
    out = []
    for i in range(n_words):
        out.append(rnd.choice(vocab))
        if i % 17 == 16:
            out.append(".")
    return " ".join(out)


def prompt(seed, n_tokens, secret):
    head = "Document %d. Remember this: the code word is %s.\n\n" % (seed, secret)
    tail = "\n\nWhat is the code word? Then continue the document.\n"
    words = filler(seed, int(n_tokens * 0.9))
    text = head + words + tail
    for _ in range(6):
        st, js = http("/tokenize", {"content": text})
        n = len(js.get("tokens", []))
        if n == 0 or abs(n - n_tokens) <= 64:
            break
        w = words.split(" ")
        words = " ".join(w[: max(10, int(len(w) * n_tokens / n))])
        text = head + words + tail
    return text


def complete(text, slot, n_predict=96):
    st, js = http("/completion", {"prompt": text, "n_predict": n_predict, "temperature": 0.0, "top_k": 1, "seed": 1,
                                  "id_slot": slot, "cache_prompt": True, "return_tokens": True})
    if st != 200:
        raise RuntimeError("completion on slot %d failed: %s %s" % (slot, st, js))
    return js.get("tokens", []), js.get("timings", {}).get("prompt_n", -1), js.get("content", "")


RESULTS = []


def record(name, ok, detail):
    RESULTS.append((name, ok, detail))
    print("  %s %s: %s" % ("PASS" if ok else "FAIL", name, detail))


def arm_save_restore(args):
    with Server(args, "turbot", tag="save_restore") as srv:
        text = prompt(11, args.prompt_tokens, "COBALT-HERON-4817")
        ref, pn0, _ = complete(text, 0)
        st, js = http("/slots/0?action=save", {"filename": "turbot_slot0.bin"})
        if st != 200:
            return record("save_restore", False, "save failed: %s %s" % (st, js))
        http("/slots/0?action=erase", {})
        st, js = http("/slots/2?action=restore", {"filename": "turbot_slot0.bin"})
        if st != 200:
            return record("save_restore", False, "restore into slot 2 failed: %s %s" % (st, js))
        toks, pn, _ = complete(text, 2)
        ok = toks == ref and 0 <= pn < 256
        record("save_restore", ok, "prompt_n %d -> %d, tokens %s, blob restore log: %s" % (
            pn0, pn, "identical" if toks == ref else "DIFFER", "yes" if re.search(r"turbot", srv.log_text()) else "no turbot line"))


def arm_prompt_cache(args):
    with Server(args, "turbot", tag="prompt_cache"):
        text = prompt(21, args.prompt_tokens, "AMBER-LYNX-2093")
        ref, _, _ = complete(text, 1)
        complete(prompt(22, args.prompt_tokens, "VIOLET-OTTER-6652"), 1)
        toks, pn, _ = complete(text, 3)
        record("prompt_cache", toks == ref and 0 <= pn < 256, "prompt_n %d on slot 3, tokens %s" % (pn, "identical" if toks == ref else "DIFFER"))


def arm_park_resume(args):
    text_seed, secret = 31, "SILVER-FALCON-3308"
    with Server(args, "turbot", tag="park_ref"):
        ref, _, _ = complete(prompt(text_seed, args.prompt_tokens, secret), 0, n_predict=400)
    with Server(args, "turbot", extra_env={"TURBO_POOL_DEBUG_FAIL": "gen@17"}, tag="park_hook") as srv:
        toks, _, _ = complete(prompt(text_seed, args.prompt_tokens, secret), 0, n_predict=400)
        log = srv.log_text()
    parks, resumes = log.count("__TEST_TAG_POOL_PARK__"), log.count("__TEST_TAG_POOL_RESUME__")
    record("park_resume", toks == ref and parks >= 1 and resumes >= 1,
           "tokens %s, parks %d, resumes %d" % ("identical" if toks == ref else "DIFFER", parks, resumes))


def arm_streams_equal(args):
    n_tok = min(args.prompt_tokens, 12000)    # below the per-stream quota of POOL / n_stream - 128 cells
    text = prompt(51, n_tok, "INDIGO-MARTEN-7710")
    with Server(args, "turbot", tag="streams_unified", unified=True):
        ref, _, _ = complete(text, 0, n_predict=128)
    with Server(args, "turbot", tag="streams_split", unified=False) as srv:
        toks, _, _ = complete(text, 2, n_predict=128)
        streams = srv.streams_line
    record("streams_equal", toks == ref and streams != "",
           "%d prompt tokens, unified slot 0 vs per-stream slot 2: tokens %s, stream split logged: %s" % (
               n_tok, "identical" if toks == ref else "DIFFER", streams.strip(", ") or "NO (not multi-stream)"))


def arm_refuse(args, src_kv, dst_kv, name):
    fname = "blob_%s.bin" % src_kv
    with Server(args, src_kv, tag=name + "_src"):
        complete(prompt(41, 2000, "COPPER-LANTERN-3092"), 0, n_predict=8)
        st, js = http("/slots/0?action=save", {"filename": fname})
        if st != 200:
            return record(name, False, "save on %s failed: %s %s" % (src_kv, st, js))
    with Server(args, dst_kv, tag=name + "_dst") as srv:
        st, js = http("/slots/1?action=restore", {"filename": fname})
        alive = http("/health", timeout=10)[0] == 200
        answered = False
        try:
            toks, _, _ = complete("Say hello.", 1, n_predict=8)
            answered = len(toks) > 0
        except Exception:
            pass
        log = srv.log_text()
        mismatch = "turbot: state blob type mismatch" in log or "mismatched key type" in log
        record(name, st != 200 and alive and answered and mismatch,
               "restore status %s, healthy %s, answers after %s, mismatch logged %s" % (st, alive, answered, mismatch))


def main():
    ap = argparse.ArgumentParser(description="turbot state blob round trips")
    ap.add_argument("--exe", required=True, help="llama-server.exe of the turbot build")
    ap.add_argument("--plan", default=PLAN_DEFAULT)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--harness-dir", default=HARNESS_DEFAULT)
    ap.add_argument("--prompt-tokens", type=int, default=20000)
    ap.add_argument("--work", default=None, help="directory for slot files and server logs (default: a temp dir on E: if present)")
    ap.add_argument("--only", default="", help="comma list of arms")
    ap.add_argument("--np", type=int, default=None, help="slots (-np); default: the profile's (4)")
    ap.add_argument("--non-unified", action="store_true", help="one KV stream per slot (--no-kv-unified) in every arm")
    args = ap.parse_args()
    if args.work is None:
        root = "E:/" if os.path.isdir("E:/") else None
        args.work = tempfile.mkdtemp(prefix="turbot_blob_", dir=root)
    os.makedirs(args.work, exist_ok=True)
    args.work = os.path.abspath(args.work)
    print("work dir %s" % args.work)
    if busy():
        print("REFUSED: a llama/ggml process is already running: %s" % busy())
        return 2
    arms = {
        "save_restore": lambda: arm_save_restore(args),
        "prompt_cache": lambda: arm_prompt_cache(args),
        "park_resume": lambda: arm_park_resume(args),
        "refuse_t5_into_t": lambda: arm_refuse(args, "turbo5p", "turbot", "refuse_t5_into_t"),
        "refuse_t_into_t5": lambda: arm_refuse(args, "turbot", "turbo5p", "refuse_t_into_t5"),
        "streams_equal": lambda: arm_streams_equal(args),
    }
    only = [a for a in args.only.split(",") if a]
    for name, fn in arms.items():
        if only and name not in only:
            continue
        print("[%s]" % name)
        try:
            fn()
        except Exception as e:
            record(name, False, "error: %s" % e)
    print("\nRESULT: %d/%d arms passed" % (sum(1 for r in RESULTS if r[1]), len(RESULTS)))
    return 0 if RESULTS and all(r[1] for r in RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
