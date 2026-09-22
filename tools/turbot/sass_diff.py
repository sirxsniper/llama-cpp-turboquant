# [TAG_TURBOT_ANY_TEST] Gate G2 SASS identity check (turbot on other models, WP2): the Qwen3.8-27B turbot kernels of two
# ggml-cuda builds must compile to the same machine code. CPU only: runs cuobjdump -sass on the two binaries, never a GPU.
#
#   python tools/turbot/sass_diff.py BASE NEW [--scope gate|turbot] [--arch sm_120a] [--dump-dir DIR] [--cuda-bin DIR]
#
# BASE and NEW are ggml-cuda.dll / libggml-cuda.so files (any host binary or fatbin cuobjdump reads), or text files that
# already hold `cuobjdump -sass` output (.txt / .sass). BASE is the HEAD + [TAG_FA_SINK_CLAMP] build, NEW the build with
# the WP2 kernels.
#
# Kernels compared (--scope gate, the default):
#   - every flash_attn_ext_turbot<256, 256, ncols1, ncols2, softcap> (the 20 D = 256 instances, both softcap variants)
#   - k_turbot_set_rows<int> and k_turbot_set_rows<int64_t>, k_turbot_fill (the writer)
#   - flash_attn_turbot_balance_bounds<4> (the seam kernel)
# --scope turbot compares every kernel with "turbot" in its name that BASE has (fixups, prefix, KV bounds scans too).
#
# Kernels are matched by name and template arguments, read from the Itanium-mangled symbol itself (_Z / _ZL / _ZN..E
# free-function templates with integer, bool and builtin-type arguments; cu++filt cannot demangle the _ZL symbols of
# static kernels). The parameter list is not part of the key: it is fixed per template. The WP2 template arguments are
# normalised before names are matched: k_turbot_set_rows<int, 3> -> k_turbot_set_rows<int>, k_turbot_fill<3> ->
# k_turbot_fill, flash_attn_turbot_balance_bounds<4, 1> -> flash_attn_turbot_balance_bounds<4>. The new instances (NG 2 / 4
# writers, other balance head counts, D = 128 kernels) have no BASE twin and are counted as NEW-only.
#
# A kernel is compared per SM architecture as the multiset of its copies (a static kernel is compiled into every TU that
# uses it, e.g. the balance kernel into each of the 20 instance TUs). SASS bodies are compared after dropping
# .headerflags and renumbering the .L_x_N branch labels in order of appearance inside the function (cuobjdump numbers
# them per ELF, so an unrelated new kernel in the same TU would shift them); instruction words, their encodings and
# offsets are compared as printed.
#
# Output: one line per kernel, IDENTICAL or DIFF (or MISSING when NEW lacks it), then a summary. Exit code 0 when every
# compared kernel is IDENTICAL, 1 otherwise, 2 on a usage or tool error (including "no gated kernel found in BASE").
import argparse
import glob
import hashlib
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict

RE_ARCH = re.compile(r"^\s*(?:arch\s*=|code for)\s*(sm_\w+)")
RE_FUNC = re.compile(r"^\s*Function\s*:\s*(\S+)")
RE_END = re.compile(r"^\s*\.{5,}\s*$")
RE_FATBIN = re.compile(r"^\s*Fatbin\b")
RE_LABEL = re.compile(r"\.L_x_\d+")
RE_NUM = re.compile(r"\d+")

# WP2 template-argument normalisation, applied to the canonical "name<arg,arg>" keys
NORMALIZE = [
    (re.compile(r"^k_turbot_set_rows<([^<>,]+),3>$"), r"k_turbot_set_rows<\1>"),
    (re.compile(r"^k_turbot_fill<3>$"), r"k_turbot_fill"),
    (re.compile(r"^flash_attn_turbot_balance_bounds<4,1>$"), r"flash_attn_turbot_balance_bounds<4>"),
]

GATE = [
    re.compile(r"^flash_attn_ext_turbot<256,256,\d+,\d+,(true|false)>$"),
    re.compile(r"^k_turbot_set_rows<(int|long|long long)>$"),
    re.compile(r"^k_turbot_fill$"),
    re.compile(r"^flash_attn_turbot_balance_bounds<4>$"),
]

BUILTIN = {"v": "void", "b": "bool", "c": "char", "a": "signed char", "h": "unsigned char", "s": "short",
           "t": "unsigned short", "i": "int", "j": "unsigned int", "l": "long", "m": "unsigned long",
           "x": "long long", "y": "unsigned long long", "f": "float", "d": "double"}

EXPECTED_FA = 40   # 20 D = 256 instances x 2 softcap variants, per architecture


def find_tool(name, cuda_bin):
    """cuobjdump: --cuda-bin, $CUDA_PATH/bin, PATH, then the default Windows toolkit folders (newest first)."""
    exe = name + (".exe" if os.name == "nt" else "")
    dirs = []
    if cuda_bin:
        dirs.append(cuda_bin)
    for env in ("CUDA_PATH", "CUDA_HOME"):
        if os.environ.get(env):
            dirs.append(os.path.join(os.environ[env], "bin"))
    for d in dirs:
        p = os.path.join(d, exe)
        if os.path.isfile(p):
            return p
    p = shutil.which(name)
    if p:
        return p
    for d in sorted(glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin"), reverse=True):
        p = os.path.join(d, exe)
        if os.path.isfile(p):
            return p
    return None


def is_text_dump(path):
    if path.lower().endswith((".txt", ".sass")):
        return True
    with open(path, "rb") as f:
        head = f.read(4096)
    return b"Fatbin" in head and b"\x00" not in head


def sass_lines(path, cuobjdump):
    """lines of `cuobjdump -sass path`, streamed (a whole ggml-cuda dump is large), or of a saved dump"""
    if is_text_dump(path):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                yield line
        return
    if not cuobjdump:
        raise SystemExit("sass_diff: cuobjdump not found (use --cuda-bin)")
    proc = subprocess.Popen([cuobjdump, "-sass", path], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True, encoding="utf-8", errors="replace")
    for line in proc.stdout:
        yield line
    err = proc.stderr.read()
    if proc.wait() != 0:
        raise SystemExit("sass_diff: cuobjdump failed on %s: %s" % (path, err.strip()))


def normalize_body(lines):
    labels = {}

    def relabel(m):
        k = m.group(0)
        if k not in labels:
            labels[k] = ".L_x_%d" % len(labels)
        return labels[k]

    out = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith(".headerflags"):
            continue
        out.append(RE_LABEL.sub(relabel, s))
    return "\n".join(out) + "\n"


def collect(path, cuobjdump, arch_filter, dump_dir, tag):
    """-> {(arch, mangled): [(sha256, n_lines, dump_file or None), ...]} for every function whose name mentions turbot"""
    funcs = defaultdict(list)
    arch, name, body = "?", None, None

    def flush():
        nonlocal name, body
        if name is not None and body is not None:
            text = normalize_body(body)
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            dump = None
            if dump_dir:
                copies = funcs[(arch, name)]
                dump = os.path.join(dump_dir, tag, arch,
                                    "%s.%d.sass" % (hashlib.sha1(name.encode("utf-8")).hexdigest()[:16], len(copies)))
                os.makedirs(os.path.dirname(dump), exist_ok=True)
                with open(dump, "w", encoding="utf-8", newline="\n") as f:
                    f.write("// " + name + "\n" + text)
            funcs[(arch, name)].append((digest, text.count("\n"), dump))
        name, body = None, None

    for line in sass_lines(path, cuobjdump):
        m = RE_FUNC.match(line)
        if m:
            flush()
            mangled = m.group(1)
            if "turbot" in mangled and (arch_filter is None or arch == arch_filter):
                name, body = mangled, []
            continue
        if RE_END.match(line) or RE_FATBIN.match(line):
            flush()
            continue
        m = RE_ARCH.match(line)
        if m:
            flush()
            arch = m.group(1)
            continue
        if body is not None:
            body.append(line)
    flush()
    return funcs


def parse_source_name(m, pos):
    """<length><identifier> at pos -> (identifier, next pos), or (None, pos)"""
    mm = RE_NUM.match(m, pos)
    if not mm:
        return None, pos
    n = int(mm.group(0))
    return m[mm.end():mm.end() + n], mm.end() + n


def parse_template_args(m, pos):
    """I <args> E at pos, args being integer / bool literals or builtin types -> ([args], next pos), or (None, pos)"""
    if pos >= len(m) or m[pos] != "I":
        return [], pos
    pos += 1
    args = []
    while pos < len(m) and m[pos] != "E":
        c = m[pos]
        if c == "L":                                   # literal: L <builtin type> <value> E
            end = m.find("E", pos + 2)
            if end < 0 or m[pos + 1] not in BUILTIN:
                return None, pos
            t, val = m[pos + 1], m[pos + 2:end].replace("n", "-")
            args.append(("true" if val == "1" else "false") if t == "b" else val)
            pos = end + 1
        elif c in BUILTIN:
            args.append(BUILTIN[c])
            pos += 1
        else:
            return None, pos
    return (args, pos + 1) if pos < len(m) else (None, pos)


def canonical(mangled):
    """Itanium-mangled kernel symbol -> "name<arg,arg>" (the key BASE and NEW are matched on, WP2 arguments dropped).
    Free functions only: _Z<name>, _ZL<name> (internal linkage) and _ZN..<name>E (the last component of a nested name,
    e.g. an _INTERNAL_ or anonymous-namespace prefix). Anything else is returned unchanged."""
    m = mangled
    nested = False
    if m.startswith("_ZL"):
        name, pos = parse_source_name(m, 3)
    elif m.startswith("_ZN"):
        pos, name, nested = 3, None, True
        while True:
            comp, npos = parse_source_name(m, pos)
            if comp is None:
                break
            name, pos = comp, npos
    elif m.startswith("_Z"):
        name, pos = parse_source_name(m, 2)
    else:
        return m
    if not name:
        return m
    args, pos = parse_template_args(m, pos)
    if args is None or (nested and (pos >= len(m) or m[pos] != "E")):
        return m
    s = name + ("<" + ",".join(args) + ">" if args else "")
    for rx, rep in NORMALIZE:
        s = rx.sub(rep, s)
    return s


def in_scope(name, scope):
    if scope == "turbot":
        return "turbot" in name
    return any(rx.search(name) for rx in GATE)


def main():
    ap = argparse.ArgumentParser(description="turbot SASS identity (gate G2)")
    ap.add_argument("base", help="baseline ggml-cuda binary (HEAD + [TAG_FA_SINK_CLAMP]) or its cuobjdump -sass text")
    ap.add_argument("new", help="new ggml-cuda binary or its cuobjdump -sass text")
    ap.add_argument("--scope", choices=("gate", "turbot"), default="gate", help="gate: the G2 kernel list; turbot: every turbot kernel")
    ap.add_argument("--arch", default=None, help="compare one SM architecture only, e.g. sm_120a (default: every one with SASS)")
    ap.add_argument("--dump-dir", default=None, help="write every compared body (normalised) under DIR/base and DIR/new")
    ap.add_argument("--cuda-bin", default=None, help="folder holding cuobjdump")
    ap.add_argument("--list", action="store_true", help="also list the NEW-only kernels")
    args = ap.parse_args()

    cuobjdump = find_tool("cuobjdump", args.cuda_bin)

    base = collect(args.base, cuobjdump, args.arch, args.dump_dir, "base")
    new = collect(args.new, cuobjdump, args.arch, args.dump_dir, "new")

    def by_canon(funcs):
        out = defaultdict(list)
        for (arch, mangled), copies in funcs.items():
            out[(arch, canonical(mangled))].extend(copies)
        return out

    cb, cn = by_canon(base), by_canon(new)
    keys = sorted(k for k in cb if in_scope(k[1], args.scope))
    if not keys:
        print("sass_diff: no %s kernel found in %s" % (args.scope, args.base))
        return 2

    n_ident = n_diff = n_miss = 0
    for arch, name in keys:
        a = sorted(cb[(arch, name)])
        b = sorted(cn.get((arch, name), []))
        if not b:
            verdict = "MISSING"
            n_miss += 1
        elif [x[0] for x in a] == [x[0] for x in b]:
            verdict = "IDENTICAL"
            n_ident += 1
        else:
            verdict = "DIFF"
            n_diff += 1
        detail = "%d cop%s, %d lines" % (len(a), "y" if len(a) == 1 else "ies", a[0][1])
        if verdict == "DIFF":
            detail += " | new: %d cop%s, %s lines" % (len(b), "y" if len(b) == 1 else "ies", "/".join(sorted({str(x[1]) for x in b})))
            if args.dump_dir:
                detail += " | e.g. %s vs %s" % (a[0][2], b[0][2])
        print("%-9s %-8s %s  (%s)" % (verdict, arch, name, detail))

    # per-architecture sanity: the 20 D = 256 FA instances (x 2 softcap variants) must all have been compared
    if args.scope == "gate":
        for arch in sorted({k[0] for k in keys}):
            n_fa = sum(1 for k in keys if k[0] == arch and k[1].startswith("flash_attn_ext_turbot<256,256,"))
            if n_fa != EXPECTED_FA:
                print("note: %s has %d flash_attn_ext_turbot<256,256,...> kernels in BASE (expected %d)" % (arch, n_fa, EXPECTED_FA))

    new_only = sorted(k for k in cn if k not in cb and "turbot" in k[1])
    if args.list:
        for arch, name in new_only:
            print("NEW-ONLY  %-8s %s" % (arch, name))
    unparsed = sorted({k[1] for k in list(cb) + list(cn) if k[1].startswith("_Z")})
    if unparsed:
        print("note: %d turbot symbols kept their mangled name (not a free-function template this tool parses), e.g. %s"
              % (len(unparsed), unparsed[0]))
    print("\nsass_diff (%s): %d IDENTICAL, %d DIFF, %d MISSING; %d NEW-only turbot kernels" % (args.scope, n_ident, n_diff, n_miss, len(new_only)))
    print("G2 SASS: %s" % ("IDENTICAL" if n_diff == 0 and n_miss == 0 else "DIFF"))
    return 0 if n_diff == 0 and n_miss == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
