#!/usr/bin/env python3
# [TAG_FN_MOE_TRACE] Readers for the MoE routing trace ("MOET" v1, LLAMA_MOE_TRACE) and profile ("moeprof v1",
# LLAMA_MOE_PROFILE) written by src/llama-moetrace.cpp. Pure Python + numpy, CPU only.
#
# MOET v1 (little endian):
#   file:   char[4] "MOET", u32 version = 1, u32 n_expert
#   record: u32 len, then len bytes:
#           u32 step, u16 n_tokens, u8 flags (1 prefill, 2 draft graph, 4 multi-token decode), u8 n_layers,
#           u16 seq_id[n_tokens],
#           per layer: u16 il, u16 T, u8 n_used, u8 pred_k, u16 T_pred, u16 0,
#                      u16 ids[T*n_used], f16 w[T*n_used], u16 pred[T_pred*pred_k]
#   (T of the last trunk layer can be n_outputs instead of n_tokens: qwen4exp crops the rows before the last FFN)
from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

FLAG_PREFILL, FLAG_DRAFT, FLAG_MULTI = 1, 2, 4


@dataclass
class Layer:
    il: int
    ids: np.ndarray             # [T, n_used] int
    w: np.ndarray               # [T, n_used] float32
    pred: np.ndarray | None     # [T_pred, pred_k] int or None


@dataclass
class Record:
    step: int
    n_tokens: int
    flags: int
    seq: np.ndarray
    layers: list = field(default_factory=list)

    @property
    def prefill(self):
        return bool(self.flags & FLAG_PREFILL)

    @property
    def draft(self):
        return bool(self.flags & FLAG_DRAFT)


def read_trace(path):
    """yields (n_expert, Record)"""
    with open(path, "rb") as f:
        hdr = f.read(12)
        if len(hdr) < 12 or hdr[:4] != b"MOET":
            raise SystemExit("%s: not a MOET trace" % path)
        ver, n_expert = struct.unpack("<II", hdr[4:])
        if ver != 1:
            raise SystemExit("%s: MOET version %d not supported" % (path, ver))
        while True:
            lb = f.read(4)
            if len(lb) < 4:
                return
            (n,) = struct.unpack("<I", lb)
            buf = f.read(n)
            if len(buf) < n:
                return  # truncated tail (process killed while writing)
            step, T, flags, n_l = struct.unpack_from("<IHBB", buf, 0)
            off = 8
            seq = np.frombuffer(buf, dtype="<u2", count=T, offset=off).astype(np.int32)
            off += 2 * T
            rec = Record(step, T, flags, seq)
            for _ in range(n_l):
                il, Tl, n_used, pk, Tp, _pad = struct.unpack_from("<HHBBHH", buf, off)
                off += 10
                ids = np.frombuffer(buf, dtype="<u2", count=Tl * n_used, offset=off).astype(np.int32).reshape(Tl, n_used)
                off += 2 * Tl * n_used
                w = np.frombuffer(buf, dtype="<f2", count=Tl * n_used, offset=off).astype(np.float32).reshape(Tl, n_used)
                off += 2 * Tl * n_used
                pred = None
                if pk and Tp:
                    pred = np.frombuffer(buf, dtype="<u2", count=Tp * pk, offset=off).astype(np.int32).reshape(Tp, pk)
                    off += 2 * Tp * pk
                rec.layers.append(Layer(il, ids, w, pred))
            yield n_expert, rec


def read_profile(path):
    """returns (header dict, {section: {il: np.ndarray[n_expert] float64}})"""
    with open(path, encoding="utf-8") as f:
        first = f.readline().split()
        if first[:2] != ["moeprof", "v1"]:
            raise SystemExit("%s: not a moeprof v1 file" % path)
        hdr = {}
        for tok in first[2:]:
            if "=" in tok:
                k, v = tok.split("=", 1)
                hdr[k] = v
        sections = {}
        for line in f:
            parts = line.split()
            if len(parts) < 3 or line.startswith("#"):
                continue
            sec, il = parts[0], int(parts[1])
            sections.setdefault(sec, {})[il] = np.array([float(x) for x in parts[2:]], dtype=np.float64)
    return hdr, sections


def write_profile(path, n_expert, sections, header):
    """sections: {name: {il: array}}; header: dict of extra header fields"""
    n_layer = 1 + max(il for s in sections.values() for il in s)
    extra = " ".join("%s=%s" % (k, v) for k, v in header.items())
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("moeprof v1 n_layer=%d n_expert=%d %s\n" % (n_layer, n_expert, extra))
        for name, layers in sections.items():
            for il in sorted(layers):
                f.write("%s %d %s\n" % (name, il, " ".join(str(int(round(x))) for x in layers[il])))
