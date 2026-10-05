#!/usr/bin/env python3
# [TAG_FN_SP0] Merge a Qwen3.8-Flash-Next MTP head file into a split trunk set, without rewriting the trunk.
#
# --spec-type draft-mtp builds its draft context from the TARGET model, so the nextn block must be inside the target's
# split set. This is cheap:
#   - shard 1 of the Unsloth split holds metadata only (0 tensors), so it is rewritten with block_count +1,
#     nextn_predict_layers 1, compress_ratios +[r] and the new split.count / split.tensors.count
#   - r is the MTP block's compress ratio (--mtp-ratio). [TAG_SYNC_1004] The default is the trunk's QSA ratio (4 for
#     Flash-Next): upstream #29761 runs the MTP block as a QSA layer, and its converter writes that ratio for it (the
#     head file carries the block's indexer tensors). --mtp-ratio 0 makes the old layout: a dense MTP block, as the
#     2026-09-29 file A and the Unsloth head file itself have it (the loader runs a ratio 0 block dense, without the
#     indexer cache, and only a dense block takes LLAMA_MTP_ATTN_WINDOW).
#   - the tensor shards 2..N are hardlinked under the new names (no copy; the loader checks only split.no there)
#   - one new shard at split.no = N holds the head's tensors (the "shared" head has no token_embd / output)
#
# Safety:
#   - the tensor data of a GGUF starts at align(header_end, general.alignment). The 2026-09-10 merge ignored the padding
#     and shifted every head tensor by 16 bytes (eh_proj gave 100% NaN). This script computes it, and re-reads the output.
#   - after writing, every output header is read back: tensor count, split.no/split.count, aligned in-bounds offsets,
#     and the sha1 of every head tensor against the head file. Any mismatch exits non-zero.
#   - source files are never modified. Existing outputs are replaced only with --force.
#
# Example (file A of E:/turbot-gates/flashnext/PLAN.md 7.3; this exact set exists with --mtp-ratio 0):
#   python tools/qwen4exp/merge_mtp.py \
#       --src-pattern D:/Projects/LocalAI/models/Qwen3.8-Flash-Next-UD-Q4_K_XL-%05d-of-00004.gguf --n-src 4 \
#       --head D:/Projects/LocalAI/models/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
#       --out-pattern D:/Projects/LocalAI/models/Qwen3.8-Flash-Next-UD-Q4_K_XL-MTP-%05d-of-00005.gguf --mtp-ratio 0
# The same trunk with the MTP block as a QSA layer (upstream's layout) next to it, for the dense/QSA A/B (only shard 1
# differs; shards 2-4 are hardlinks again, shard 5 is a 2.8 GB copy of the head):
#   python tools/qwen4exp/merge_mtp.py ... --out-pattern D:/Projects/LocalAI/models/Qwen3.8-Flash-Next-UD-Q4_K_XL-MTPQ-%05d-of-00005.gguf
#   python tools/qwen4exp/merge_mtp.py --verify-only --out-pattern ... --n-out 5 --head ...
from __future__ import annotations

import argparse
import hashlib
import io
import os
import struct
import sys

GGUF_MAGIC = b"GGUF"
T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 = range(13)

# ggml type id -> (block elements, block bytes); used when gguf-py cannot be imported
_SIZES = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 36),
          10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210), 15: (256, 292),
          16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110), 22: (256, 82),
          23: (256, 136), 24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8), 29: (256, 56), 30: (1, 2),
          42: (64, 18)}  # [TAG_FN_L3_MTP_Q2] q2_0
try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"))
    from gguf.constants import GGML_QUANT_SIZES  # type: ignore
    for _k, _v in GGML_QUANT_SIZES.items():
        _SIZES[int(_k)] = (int(_v[0]), int(_v[1]))
except Exception:  # noqa: BLE001 - the built-in table covers every type these files use
    pass


def rd(f, fmt):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))


def rstr(f):
    (n,) = rd(f, "<Q")
    return f.read(n).decode("utf-8", "surrogatepass")


def rval(f, t):
    simple = {T_U8: "<B", T_I8: "<b", T_U16: "<H", T_I16: "<h", T_U32: "<I", T_I32: "<i", T_F32: "<f",
              T_BOOL: "<B", T_U64: "<Q", T_I64: "<q", T_F64: "<d"}
    if t in simple:
        return rd(f, simple[t])[0]
    if t == T_STR:
        return rstr(f)
    if t == T_ARR:
        (et,) = rd(f, "<I")
        (n,) = rd(f, "<Q")
        return (et, [rval(f, et) for _ in range(n)])
    raise ValueError("gguf kv type %d" % t)


def wstr(out, s):
    b = s.encode("utf-8", "surrogatepass")
    out.write(struct.pack("<Q", len(b)))
    out.write(b)


def wval(out, t, v):
    simple = {T_U8: "<B", T_I8: "<b", T_U16: "<H", T_I16: "<h", T_U32: "<I", T_I32: "<i", T_F32: "<f",
              T_BOOL: "<B", T_U64: "<Q", T_I64: "<q", T_F64: "<d"}
    if t in simple:
        out.write(struct.pack(simple[t], v))
    elif t == T_STR:
        wstr(out, v)
    elif t == T_ARR:
        et, items = v
        out.write(struct.pack("<I", et))
        out.write(struct.pack("<Q", len(items)))
        for it in items:
            wval(out, et, it)
    else:
        raise ValueError(t)


class Header:
    def __init__(self, path):
        self.path = path
        with io.open(path, "rb") as f:
            if f.read(4) != GGUF_MAGIC:
                raise SystemExit("%s: not a GGUF file" % path)
            self.ver, n_tensors, n_kv = rd(f, "<IQQ")
            self.kv = []  # (key, type, value)
            for _ in range(n_kv):
                k = rstr(f)
                (t,) = rd(f, "<I")
                self.kv.append((k, t, rval(f, t)))
            self.tensors = []  # [name, dims, type, offset]
            for _ in range(n_tensors):
                name = rstr(f)
                (nd,) = rd(f, "<I")
                dims = list(rd(f, "<" + "Q" * nd))
                (tt,) = rd(f, "<I")
                (off,) = rd(f, "<Q")
                self.tensors.append([name, dims, tt, off])
            self.align = int(self.get("general.alignment", 32))
            self.header_end = f.tell()
        self.data_start = (self.header_end + self.align - 1) // self.align * self.align
        self.size = os.path.getsize(path)

    def get(self, key, default=None):
        for k, _t, v in self.kv:
            if k == key:
                return v
        return default

    def set(self, key, t, v):
        for i, (k, _t, _v) in enumerate(self.kv):
            if k == key:
                self.kv[i] = (key, t, v)
                return
        self.kv.append((key, t, v))


def nbytes(dims, tt):
    if tt not in _SIZES:
        raise SystemExit("unknown ggml type %d" % tt)
    blk, bpb = _SIZES[tt]
    n = 1
    for d in dims:
        n *= d
    if n % blk:
        raise SystemExit("tensor size %d is not a multiple of the block size %d (type %d)" % (n, blk, tt))
    return n // blk * bpb


def write_gguf(path, ver, kv, tensors, src=None, src_data_start=0, align=32):
    """tensors: [name, dims, type, new_offset, src_offset, nbytes]"""
    tmp = path + ".part"
    with io.open(tmp, "wb") as out:
        out.write(GGUF_MAGIC)
        out.write(struct.pack("<IQQ", ver, len(tensors), len(kv)))
        for k, t, v in kv:
            wstr(out, k)
            out.write(struct.pack("<I", t))
            wval(out, t, v)
        for name, dims, tt, off, _s, _n in tensors:
            wstr(out, name)
            out.write(struct.pack("<I", len(dims)))
            for d in dims:
                out.write(struct.pack("<Q", d))
            out.write(struct.pack("<I", tt))
            out.write(struct.pack("<Q", off))
        out.write(b"\0" * ((align - out.tell() % align) % align))
        base = out.tell()
        if tensors:
            with io.open(src, "rb") as fin:
                for name, _dims, _tt, off, s, n in tensors:
                    out.seek(base + off)
                    fin.seek(src_data_start + s)
                    left = n
                    while left:
                        chunk = fin.read(min(left, 16 << 20))
                        if not chunk:
                            raise SystemExit("short read on %s" % name)
                        out.write(chunk)
                        left -= len(chunk)
            # pad the tail so the file size is a multiple of the alignment (as gguf-py writes it)
            end = out.seek(0, 2)
            out.write(b"\0" * ((align - end % align) % align))
    os.replace(tmp, path)
    return os.path.getsize(path)


def sha1_of(path, data_start, off, n):
    h = hashlib.sha1()
    with io.open(path, "rb") as f:
        f.seek(data_start + off)
        left = n
        while left:
            chunk = f.read(min(left, 16 << 20))
            if not chunk:
                raise SystemExit("short read in %s" % path)
            h.update(chunk)
            left -= len(chunk)
    return h.hexdigest()


def verify(out_pattern, n_out, head_path):
    print("verifying %s (%d shards)" % (out_pattern % 1, n_out))
    errors = []
    hdrs = [Header(out_pattern % i) for i in range(1, n_out + 1)]
    total = sum(len(h.tensors) for h in hdrs)
    want_total = hdrs[0].get("split.tensors.count")
    if want_total != total:
        errors.append("split.tensors.count %s != %d tensors found" % (want_total, total))
    if hdrs[0].get("split.count") != n_out or hdrs[-1].get("split.count") != n_out:
        errors.append("split.count is not %d in shard 1 or the head shard" % n_out)
    for i, h in enumerate(hdrs):
        if h.get("split.no") != i:
            errors.append("%s: split.no %s != %d" % (h.path, h.get("split.no"), i))
        for name, dims, tt, off in h.tensors:
            n = nbytes(dims, tt)
            if off % h.align:
                errors.append("%s: %s offset %d not aligned to %d" % (h.path, name, off, h.align))
            if h.data_start + off + n > h.size:
                errors.append("%s: %s ends past the end of the file" % (h.path, name))
    n_layer = hdrs[0].get("qwen4exp.block_count")
    ratios = hdrs[0].get("qwen4exp.attention.compress_ratios")
    if ratios is None or len(ratios[1]) != n_layer:
        errors.append("compress_ratios has %s entries, block_count %s" % (None if ratios is None else len(ratios[1]), n_layer))
    if hdrs[0].get("qwen4exp.nextn_predict_layers") != 1:
        errors.append("nextn_predict_layers is not 1")
    names = set()
    for h in hdrs:
        for t in h.tensors:
            if t[0] in names:
                errors.append("duplicate tensor %s" % t[0])
            names.add(t[0])
    if head_path:
        hh = Header(head_path)
        last = hdrs[-1]
        by_name = {t[0]: t for t in last.tensors}
        for name, dims, tt, off in hh.tensors:
            if name not in by_name:
                errors.append("head tensor %s missing from %s" % (name, last.path))
                continue
            o = by_name[name]
            if o[1] != dims or o[2] != tt:
                errors.append("head tensor %s: shape/type differ" % name)
                continue
            n = nbytes(dims, tt)
            a = sha1_of(hh.path, hh.data_start, off, n)
            b = sha1_of(last.path, last.data_start, o[3], n)
            if a != b:
                errors.append("head tensor %s: sha1 %s != %s" % (name, b, a))
        print("  %d head tensors compared by sha1" % len(hh.tensors))
    print("  %d tensors in %d shards, block_count %s, MTP block compress ratio %s" %
          (total, n_out, n_layer, ratios[1][-1] if ratios is not None and ratios[1] else None))
    if errors:
        for e in errors:
            print("  ERROR: " + e)
        print("VERIFY FAILED")
        return 1
    print("VERIFY OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-pattern", help="printf pattern of the trunk shards, 1-based")
    ap.add_argument("--n-src", type=int, default=4)
    ap.add_argument("--head", help="MTP head GGUF (blk.<n_layer>.* tensors only); with --verify-only optional (no sha1 check)")
    ap.add_argument("--out-pattern", required=True, help="printf pattern of the output shards, 1-based")
    ap.add_argument("--n-out", type=int, default=0, help="with --verify-only: shard count of the output")
    ap.add_argument("--force", action="store_true", help="replace existing outputs")
    ap.add_argument("--mtp-ratio", default="trunk",
                    help="compress ratio of the MTP block: 'trunk' (default, the trunk's QSA ratio, as upstream's converter "
                         "writes it) or a number (0 = a dense MTP block, the old layout)")
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if a.verify_only:
        return verify(a.out_pattern, a.n_out, a.head)
    if not a.src_pattern or not a.head:
        raise SystemExit("--src-pattern and --head are required")

    n_out = a.n_src + 1
    outs = [a.out_pattern % i for i in range(1, n_out + 1)]
    srcs = [a.src_pattern % i for i in range(1, a.n_src + 1)]
    for s in srcs + [a.head]:
        if not os.path.exists(s):
            raise SystemExit("missing input %s" % s)
    for o in outs:
        if os.path.abspath(o) in [os.path.abspath(s) for s in srcs + [a.head]]:
            raise SystemExit("output %s would overwrite an input" % o)
        if os.path.exists(o) and not a.force:
            raise SystemExit("output %s exists (use --force)" % o)

    h1 = Header(srcs[0])
    if h1.tensors:
        raise SystemExit("shard 1 holds %d tensors; this script needs a metadata-only first shard" % len(h1.tensors))
    if h1.get("split.count") != a.n_src:
        raise SystemExit("shard 1 says split.count %s, --n-src is %d" % (h1.get("split.count"), a.n_src))
    arch = h1.get("general.architecture")
    n_layer = h1.get("%s.block_count" % arch)
    hh = Header(a.head)
    if hh.get("general.architecture") != arch:
        raise SystemExit("head arch %s != trunk arch %s" % (hh.get("general.architecture"), arch))
    bad = [t[0] for t in hh.tensors if not t[0].startswith("blk.%d." % n_layer)]
    if bad:
        raise SystemExit("head tensors outside blk.%d: %s" % (n_layer, bad[:4]))
    trunk_names = set()
    for s in srcs[1:]:
        trunk_names.update(t[0] for t in Header(s).tensors)
    clash = [t[0] for t in hh.tensors if t[0] in trunk_names]
    if clash:
        raise SystemExit("head tensors already in the trunk: %s" % clash[:4])

    key_r = "%s.attention.compress_ratios" % arch
    ratios = h1.get(key_r)
    if ratios is None or len(ratios[1]) != n_layer:
        raise SystemExit("trunk %s missing or not %s entries" % (key_r, n_layer))
    old_total = h1.get("split.tensors.count")
    new_total = old_total + len(hh.tensors)
    print("trunk: %s, %d layers, %d tensors; head: %d tensors (align %d, data at %d)" %
          (arch, n_layer, old_total, len(hh.tensors), hh.align, hh.data_start))

    h1.set("%s.block_count" % arch, T_U32, n_layer + 1)
    h1.set("%s.nextn_predict_layers" % arch, T_U32, 1)
    trunk_r = sorted({int(r) for r in ratios[1] if int(r) > 0})
    if a.mtp_ratio == "trunk":
        if len(trunk_r) != 1:
            raise SystemExit("the trunk has compress ratios %s; give --mtp-ratio" % trunk_r)
        mtp_r = trunk_r[0]
    else:
        mtp_r = int(a.mtp_ratio)
        if mtp_r < 0 or (mtp_r > 0 and trunk_r and mtp_r not in trunk_r):
            raise SystemExit("--mtp-ratio %d: QSA layers must share one ratio (trunk %s)" % (mtp_r, trunk_r))
    if mtp_r > 0:
        idx = [t[0] for t in hh.tensors if ".indexer." in t[0]]
        if not idx:
            raise SystemExit("--mtp-ratio %d needs the head's indexer tensors, the head has none" % mtp_r)
    print("MTP block compress ratio %d (%s)" % (mtp_r, "QSA, upstream's layout" if mtp_r > 0 else "dense"))
    h1.set(key_r, T_ARR, (ratios[0], list(ratios[1]) + [mtp_r]))
    h1.set("split.count", T_U16, n_out)
    h1.set("split.no", T_U16, 0)
    h1.set("split.tensors.count", T_I32, new_total)
    sz = write_gguf(outs[0], h1.ver, h1.kv, [], align=h1.align)
    print("wrote %s (%.1f MB)" % (outs[0], sz / 1e6))

    for i in range(1, a.n_src):
        if os.path.exists(outs[i]):
            os.remove(outs[i])
        os.link(srcs[i], outs[i])
        print("hardlinked %s -> %s (%.1f GB, no copy)" % (outs[i], srcs[i], os.path.getsize(outs[i]) / 1e9))

    align = hh.align
    tensors, off = [], 0
    for name, dims, tt, src_off in hh.tensors:
        n = nbytes(dims, tt)
        off = (off + align - 1) // align * align
        tensors.append([name, dims, tt, off, src_off, n])
        off += n
    kv_last = [("split.no", T_U16, a.n_src), ("split.count", T_U16, n_out), ("split.tensors.count", T_I32, new_total),
               ("general.alignment", T_U32, align)]
    sz = write_gguf(outs[-1], hh.ver, kv_last, tensors, src=a.head, src_data_start=hh.data_start, align=align)
    print("wrote %s (%.2f GB)" % (outs[-1], sz / 1e9))
    return verify(a.out_pattern, n_out, a.head)


if __name__ == "__main__":
    sys.exit(main())
