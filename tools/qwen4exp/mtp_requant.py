#!/usr/bin/env python3
# [TAG_FN_R4_MTP_MIX] Requantize the routed experts of a Qwen3.8-Flash-Next MTP head file to the trunk's own expert mix,
# every other tensor byte for byte. The Unsloth shared head carries its 512 experts as q8_0 (3 x 850 MiB = 2.49 GiB);
# the trunk's mix (UD-Q4_K_XL: gate/up q4_K, down q5_1) holds them in 1.46 GiB, so ~1.0 GiB of VRAM moves to the hot
# set. Only the drafts can change: the target verifies every token with its own (unchanged) weights.
#
#   python tools/qwen4exp/mtp_requant.py --head D:/Projects/LocalAI/models/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
#       --trunk D:/Projects/LocalAI/models/Qwen3.8-Flash-Next-UD-Q4_K_XL-MTP-00001-of-00005.gguf \
#       --out D:/Projects/LocalAI/models/mtp-Qwen3.8-Flash-Next-shared-trunkmix.gguf [--dry-run]
#
# - The mix: --mix gate,up,down (type names), or the most common (gate, up, down) of the trunk's expert layers (--trunk,
#   a header read of every shard; the MTP block blk.<n_layer> is left out of the count).
# - The quantizer is ggml's own (ggml_quantize_chunk from ggml-base, --ggml <bin dir>), one expert matrix at a time on
#   --threads threads, without an importance matrix (the Unsloth imatrix has no entries for the MTP block).
# - Sources q8_0 / f16 / bf16 / f32 are dequantized here (q8_0: d x q, exact).
# - After writing, the output header is read back: the same tensors and shapes, the planned types, aligned offsets, the
#   untouched tensors sha1-equal to the head file, and the requantized experts dequantized again (gguf-py) for their
#   relative RMS error against the source on a sample of experts. Any mismatch exits non-zero.
# - --dry-run prints the plan (types, bytes, VRAM freed) and writes --tt-out (a --tensor-type-file for llama-quantize,
#   the other route: llama-quantize --allow-requantize --tensor-type-file <it> <head> <out> COPY).
#
# Then merge it like file A (a dense nextn block): merge_mtp.py --mtp-ratio 0 --head <out> ... (recipe_mtp_mix.ps1).
# CPU and disk only (reads ~2.6 GB, writes ~1.6 GB). Never run beside a GPU speed measurement.
from __future__ import annotations

import argparse
import collections
import ctypes
import hashlib
import io
import os
import re
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from merge_mtp import GGUF_MAGIC, Header, nbytes, wstr, wval  # noqa: E402

TYPE_IDS = {"f32": 0, "f16": 1, "q4_0": 2, "q4_1": 3, "q5_0": 6, "q5_1": 7, "q8_0": 8, "q2_K": 10, "q3_K": 11,
            "q4_K": 12, "q5_K": 13, "q6_K": 14, "iq4_nl": 20, "iq4_xs": 23, "bf16": 30}
TYPE_NAMES = {v: k for k, v in TYPE_IDS.items()}
EXP_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")


def shard_paths(first):
    m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", first)
    if not m:
        return [first]
    n = int(m.group(2))
    return [first[:m.start()] + "-%05d-of-%05d.gguf" % (i, n) for i in range(1, n + 1)]


def trunk_mix(first, mtp_layer):
    """the most common (gate, up, down) type triple of the trunk's expert layers"""
    per = collections.defaultdict(dict)
    for p in shard_paths(first):
        for name, _dims, tt, _off in Header(p).tensors:
            m = EXP_RE.match(name)
            if m and int(m.group(1)) != mtp_layer:
                per[int(m.group(1))][m.group(2)] = tt
    cnt = collections.Counter(tuple(d.get(k) for k in ("gate", "up", "down")) for d in per.values())
    if not cnt:
        raise SystemExit("no expert tensors in %s" % first)
    (mix, n), = cnt.most_common(1)
    print("trunk expert mixes: %s" % ", ".join("%s/%s/%s x%d" % (TYPE_NAMES.get(a, a), TYPE_NAMES.get(b, b),
                                                                 TYPE_NAMES.get(c, c), k) for (a, b, c), k in cnt.most_common()))
    return mix


def load_ggml(bin_dir):
    names = ["ggml-base.dll", "libggml-base.so", "libggml-base.dylib"]
    for nm in names:
        p = os.path.join(bin_dir, nm)
        if os.path.exists(p):
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(os.path.abspath(bin_dir))
            lib = ctypes.CDLL(os.path.abspath(p))
            lib.ggml_quantize_chunk.restype = ctypes.c_size_t
            lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                                ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
            lib.ggml_quantize_init.restype = None
            lib.ggml_quantize_init.argtypes = [ctypes.c_int]
            lib.ggml_quantize_requires_imatrix.restype = ctypes.c_bool
            lib.ggml_quantize_requires_imatrix.argtypes = [ctypes.c_int]
            return lib, p
    raise SystemExit("no ggml-base library in %s (--ggml)" % bin_dir)


def to_f32(raw, tt, n):
    """raw bytes of n elements of type tt as f32"""
    if tt == TYPE_IDS["q8_0"]:
        blk = np.frombuffer(raw, dtype=np.dtype([("d", "<f2"), ("qs", "i1", 32)]))
        return (blk["qs"].astype(np.float32) * blk["d"].astype(np.float32)[:, None]).reshape(n)
    if tt == TYPE_IDS["f16"]:
        return np.frombuffer(raw, dtype="<f2").astype(np.float32)
    if tt == TYPE_IDS["bf16"]:
        u = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        return u.view(np.float32)
    if tt == TYPE_IDS["f32"]:
        return np.frombuffer(raw, dtype="<f4").copy()
    raise SystemExit("source type %s is not supported here (q8_0, f16, bf16, f32)" % TYPE_NAMES.get(tt, tt))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--head", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--trunk", default="", help="shard 1 of the trunk set: its most common expert mix is the target")
    ap.add_argument("--mix", default="", help="gate,up,down type names, e.g. q4_K,q4_K,q5_1 (instead of --trunk)")
    ap.add_argument("--ggml", default=r"D:/Projects/LocalAI/source-build/wt-fsync/build-fsync/bin")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tt-out", default="", help="write a llama-quantize --tensor-type-file for the same plan")
    ap.add_argument("--check-experts", type=int, default=6, help="experts per tensor dequantized again for the error")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    hh = Header(a.head)
    arch = hh.get("general.architecture")
    exps = {}
    for name, dims, tt, off in hh.tensors:
        m = EXP_RE.match(name)
        if m:
            exps[m.group(2)] = (name, dims, tt, off, int(m.group(1)))
    if sorted(exps) != ["down", "gate", "up"]:
        raise SystemExit("%s: expected one ffn_{gate,up,down}_exps each, found %s" % (a.head, sorted(exps)))
    mtp_layer = exps["gate"][4]

    if a.mix:
        parts = [x.strip() for x in a.mix.split(",")]
        if len(parts) != 3 or any(p not in TYPE_IDS for p in parts):
            raise SystemExit("--mix needs three of %s" % sorted(TYPE_IDS))
        mix = tuple(TYPE_IDS[p] for p in parts)
    elif a.trunk:
        mix = trunk_mix(a.trunk, mtp_layer)
    else:
        raise SystemExit("--trunk or --mix is required")
    target = dict(zip(("gate", "up", "down"), mix))

    print("head %s: %s, MTP block blk.%d, %d tensors" % (a.head, arch, mtp_layer, len(hh.tensors)))
    before = after = 0
    for k in ("gate", "up", "down"):
        name, dims, tt, _off, _ = exps[k]
        nb0, nb1 = nbytes(dims, tt), nbytes(dims, target[k])
        before += nb0
        after += nb1
        print("  %-30s %-18s %-6s -> %-6s  %8.1f -> %8.1f MiB" % (name, dims, TYPE_NAMES.get(tt, tt),
                                                                   TYPE_NAMES.get(target[k], target[k]), nb0 / 2**20, nb1 / 2**20))
    print("experts: %.3f GiB -> %.3f GiB, %.3f GiB of VRAM freed for the hot set" % (before / 2**30, after / 2**30,
                                                                                       (before - after) / 2**30))
    if a.tt_out:
        with open(a.tt_out, "w", encoding="utf-8") as f:
            for k in ("gate", "up", "down"):
                f.write("%s=%s\n" % (exps[k][0], TYPE_NAMES[target[k]]))
        print("wrote %s" % a.tt_out)
    if a.dry_run:
        return 0
    if not a.out:
        raise SystemExit("--out is required")
    if os.path.abspath(a.out) == os.path.abspath(a.head):
        raise SystemExit("--out would overwrite the head")
    if os.path.exists(a.out) and not a.force:
        raise SystemExit("%s exists (use --force)" % a.out)

    lib, libpath = load_ggml(a.ggml)
    print("quantizer: %s" % libpath)
    for k in ("gate", "up", "down"):
        lib.ggml_quantize_init(target[k])
        if lib.ggml_quantize_requires_imatrix(target[k]):
            raise SystemExit("%s needs an importance matrix" % TYPE_NAMES[target[k]])

    # requantize, one expert matrix per task
    t0 = time.time()
    new_data = {}
    with io.open(a.head, "rb") as f:
        for k in ("gate", "up", "down"):
            name, dims, tt, off, _ = exps[k]
            ne0, ne1, n_exp = dims
            src_e = nbytes([ne0, ne1], tt)
            dst_e = nbytes([ne0, ne1], target[k])
            out = bytearray(dst_e * n_exp)
            f.seek(hh.data_start + off)
            raw_all = f.read(src_e * n_exp)
            if len(raw_all) != src_e * n_exp:
                raise SystemExit("short read on %s" % name)
            ob = (ctypes.c_char * len(out)).from_buffer(out)

            def work(e):
                x = np.ascontiguousarray(to_f32(raw_all[e * src_e:(e + 1) * src_e], tt, ne0 * ne1))
                n = lib.ggml_quantize_chunk(target[k], x.ctypes.data, ctypes.addressof(ob) + e * dst_e, 0, ne1, ne0, None)
                if n != dst_e:
                    raise RuntimeError("expert %d: %d bytes, expected %d" % (e, n, dst_e))
                return e

            with ThreadPoolExecutor(max_workers=max(1, a.threads)) as ex:
                list(ex.map(work, range(n_exp)))
            del ob
            new_data[name] = (bytes(out), target[k])
            print("  %s: %d experts requantized (%.0f s)" % (name, n_exp, time.time() - t0))

    # write: the head's metadata, the same tensor order, aligned offsets
    align = hh.align
    tensors, off = [], 0
    for name, dims, tt, src_off in hh.tensors:
        ntt = new_data[name][1] if name in new_data else tt
        n = nbytes(dims, ntt)
        off = (off + align - 1) // align * align
        tensors.append((name, dims, tt, ntt, src_off, off, n))
        off += n
    tmp = a.out + ".part"
    with io.open(tmp, "wb") as out, io.open(a.head, "rb") as fin:
        out.write(GGUF_MAGIC)
        out.write(struct.pack("<IQQ", hh.ver, len(tensors), len(hh.kv)))
        for k, t, v in hh.kv:
            wstr(out, k)
            out.write(struct.pack("<I", t))
            wval(out, t, v)
        for name, dims, _tt, ntt, _s, noff, _n in tensors:
            wstr(out, name)
            out.write(struct.pack("<I", len(dims)))
            for d in dims:
                out.write(struct.pack("<Q", d))
            out.write(struct.pack("<I", ntt))
            out.write(struct.pack("<Q", noff))
        out.write(b"\0" * ((align - out.tell() % align) % align))
        base = out.tell()
        for name, _dims, tt, _ntt, s, noff, n in tensors:
            out.seek(base + noff)
            if name in new_data:
                out.write(new_data[name][0])
                continue
            fin.seek(hh.data_start + s)
            left = n
            while left:
                chunk = fin.read(min(left, 16 << 20))
                if not chunk:
                    raise SystemExit("short read on %s" % name)
                out.write(chunk)
                left -= len(chunk)
        end = out.seek(0, 2)
        out.write(b"\0" * ((align - end % align) % align))
    os.replace(tmp, a.out)
    print("wrote %s (%.2f GB)" % (a.out, os.path.getsize(a.out) / 1e9))
    return verify(a, hh, exps, target, new_data)


def sha1_range(path, start, n):
    h = hashlib.sha1()
    with io.open(path, "rb") as f:
        f.seek(start)
        left = n
        while left:
            chunk = f.read(min(left, 16 << 20))
            if not chunk:
                raise SystemExit("short read in %s" % path)
            h.update(chunk)
            left -= len(chunk)
    return h.hexdigest()


def verify(a, hh, exps, target, new_data):
    errors = []
    ho = Header(a.out)
    if [t[0] for t in ho.tensors] != [t[0] for t in hh.tensors]:
        errors.append("tensor list differs from the head")
    by_name = {t[0]: t for t in hh.tensors}
    for name, dims, tt, off in ho.tensors:
        src = by_name.get(name)
        if src is None:
            continue
        if src[1] != dims:
            errors.append("%s: shape %s != %s" % (name, dims, src[1]))
        want_tt = new_data[name][1] if name in new_data else src[2]
        if tt != want_tt:
            errors.append("%s: type %s, planned %s" % (name, TYPE_NAMES.get(tt, tt), TYPE_NAMES.get(want_tt, want_tt)))
        n = nbytes(dims, tt)
        if off % ho.align or ho.data_start + off + n > ho.size:
            errors.append("%s: offset %d not aligned or past the end" % (name, off))
        if name not in new_data and not errors:
            if sha1_range(a.out, ho.data_start + off, n) != sha1_range(a.head, hh.data_start + src[3], n):
                errors.append("%s: bytes differ from the head" % name)
    # the requantized experts against the source, dequantized again with gguf-py
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"))
        from gguf.quants import dequantize  # type: ignore
        from gguf.constants import GGMLQuantizationType  # type: ignore
        by_out = {t[0]: t for t in ho.tensors}
        with io.open(a.head, "rb") as fs, io.open(a.out, "rb") as fo:
            for k in ("gate", "up", "down"):
                name, dims, tt, off, _ = exps[k]
                ne0, ne1, n_exp = dims
                src_e = nbytes([ne0, ne1], tt)
                dst_e = nbytes([ne0, ne1], target[k])
                worst = 0.0
                for e in np.linspace(0, n_exp - 1, num=max(1, a.check_experts)).astype(int).tolist():
                    fs.seek(hh.data_start + off + e * src_e)
                    x = to_f32(fs.read(src_e), tt, ne0 * ne1)
                    fo.seek(ho.data_start + by_out[name][3] + e * dst_e)
                    q = np.frombuffer(fo.read(dst_e), dtype=np.uint8)
                    y = dequantize(q, GGMLQuantizationType(target[k])).reshape(-1).astype(np.float32)
                    rel = float(np.sqrt(np.mean((x - y) ** 2)) / max(1e-12, np.sqrt(np.mean(x ** 2))))
                    worst = max(worst, rel)
                print("  %s: relative RMS error vs the source <= %.4f over %d experts" % (name, worst, a.check_experts))
                if not worst < 0.25:
                    errors.append("%s: relative RMS error %.3f (a broken requantization)" % (name, worst))
    except ImportError as ex:
        print("  (gguf-py not importable, the error check is skipped: %s)" % ex)
    if errors:
        for e in errors:
            print("ERROR: " + e)
        print("VERIFY FAILED")
        return 1
    print("VERIFY OK: %d tensors, %d requantized, the rest byte-equal" % (len(ho.tensors), len(new_data)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
