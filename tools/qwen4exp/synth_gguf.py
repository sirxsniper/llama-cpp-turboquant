#!/usr/bin/env python3
# [TAG_QWEN4EXP_MTP] Write a qwen4exp (Qwen3.8-Flash-Next) GGUF with random weights and the real per-layer shapes, so
# graph build, the MTP head and the CPU/GPU split can be tested without the 100 GB model.
#
# What is real and what is not:
#   - every per-layer shape (n_embd 2560, hc 4, 2 KV heads x 256, 640-wide experts, PLE rows of 160, vocab 248320)
#     and the per-token work (n_expert_used experts per layer) are the real ones
#   - the layer count, the expert count and the PLE table rows are cut down (flags), so the file stays small
#   - the MTP head is the 32-tensor "shared" head of the merged Unsloth file (no embed_tokens / shared_head_head):
#     it uses the model's token_embd and output
#   - the values are random, so the text is noise and MTP acceptance is ~0; the checks are "loads, runs, finite,
#     deterministic, CPU == GPU, MTP on == MTP off at temp 0"
#
# Real metadata: pass --meta with the metadata-only first shard of the Unsloth split (Qwen3.8-Flash-Next-*-00001-of-*.gguf,
# ~11 MB, 0 tensors; needs the owner's OK to download). Without it the built-in DEFAULTS are used; the values marked
# "guess" there are not known for this model.
#
# Examples:
#   python tools/qwen4exp/synth_gguf.py E:/synth/q4x-smoke.gguf                         # 8 layers + MTP, 32 experts, ~3.3 GB
#   python tools/qwen4exp/synth_gguf.py E:/synth/q4x-48.gguf --layers 48 --no-mtp       # the 48-layer split structure
#   python tools/qwen4exp/synth_gguf.py E:/synth/q4x-smoke.gguf --meta shard1.gguf      # real metadata and tokenizer

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf  # noqa: E402
from gguf import GGMLQuantizationType as QT  # noqa: E402
from gguf import GGUFValueType as VT  # noqa: E402

ARCH = "qwen4exp"

# the real model where known (memory notes 2026-08-27 .. 2026-09-26); "guess" = not known, override with --meta
DEFAULTS = {
    "n_vocab":        248320,
    "n_ctx_train":    262144,
    "n_embd":         2560,
    "n_layer":        48,
    "full_attn_int":  4,          # 12 of 48 layers are full attention
    "n_head":         16,         # guess
    "n_head_kv":      2,
    "head_dim":       256,
    "n_rot":          64,         # guess (partial rotary 0.25)
    "rope_sections":  [11, 11, 10, 0],  # guess
    "rope_base":      10000000.0, # guess
    "rms_eps":        1e-6,
    "n_expert":       512,
    "n_expert_used":  10,
    "n_ff_exp":       640,
    "n_ff_shexp":     640,        # guess
    "ssm_d_conv":     4,          # guess
    "ssm_d_state":    128,        # guess
    "ssm_n_group":    16,         # guess
    "ssm_dt_rank":    32,         # guess
    "hc":             4,
    "hc_low_rank":    32,         # guess
    "idx_n_head":     16,         # guess
    "idx_head_dim":   128,        # guess
    "idx_top_k":      2048,
    "compress_ratio": 4,
    "ple_layer":      0,          # guess (must be a linear attention layer)
    "ple_ngram":      3,          # guess
    "ple_heads_per":  4,          # guess
    "ple_conv":       4,          # guess
    "ple_head_dim":   160,
}


def k(name: str) -> str:
    return f"{ARCH}.{name}"


def load_meta(path: str) -> dict[str, tuple]:
    r = gguf.GGUFReader(path)
    meta = {}
    for name, field in r.fields.items():
        if name.startswith("GGUF.") or name.startswith("split."):
            continue
        vtype = field.types[0]
        sub = field.types[-1] if vtype == VT.ARRAY else None
        meta[name] = (field.contents(), vtype, sub)
    return meta


def default_meta(d: dict) -> dict[str, tuple]:
    n_heads = (d["ple_ngram"] - 1) * d["ple_heads_per"]
    rng = np.random.default_rng(1234)
    mult = [int(x) | 1 for x in rng.integers(1 << 40, 1 << 45, size=d["ple_ngram"])]
    m = {
        "general.architecture":              (ARCH, VT.STRING, None),
        "general.name":                      ("qwen4exp-synthetic", VT.STRING, None),
        k("vocab_size"):                     (d["n_vocab"], VT.UINT32, None),
        k("context_length"):                 (d["n_ctx_train"], VT.UINT32, None),
        k("embedding_length"):               (d["n_embd"], VT.UINT32, None),
        k("block_count"):                    (d["n_layer"], VT.UINT32, None),
        k("feed_forward_length"):            (d["n_ff_exp"], VT.UINT32, None),
        k("expert_feed_forward_length"):     (d["n_ff_exp"], VT.UINT32, None),
        k("expert_shared_feed_forward_length"): (d["n_ff_shexp"], VT.UINT32, None),
        k("expert_count"):                   (d["n_expert"], VT.UINT32, None),
        k("expert_used_count"):              (d["n_expert_used"], VT.UINT32, None),
        k("attention.head_count"):           (d["n_head"], VT.UINT32, None),
        k("attention.head_count_kv"):        (d["n_head_kv"], VT.UINT32, None),
        k("attention.key_length"):           (d["head_dim"], VT.UINT32, None),
        k("attention.value_length"):         (d["head_dim"], VT.UINT32, None),
        k("attention.layer_norm_rms_epsilon"): (d["rms_eps"], VT.FLOAT32, None),
        k("rope.dimension_count"):           (d["n_rot"], VT.UINT32, None),
        k("rope.dimension_sections"):        (d["rope_sections"], VT.ARRAY, VT.INT32),
        k("rope.freq_base"):                 (d["rope_base"], VT.FLOAT32, None),
        k("full_attention_interval"):        (d["full_attn_int"], VT.UINT32, None),
        k("ssm.conv_kernel"):                (d["ssm_d_conv"], VT.UINT32, None),
        k("ssm.state_size"):                 (d["ssm_d_state"], VT.UINT32, None),
        k("ssm.group_count"):                (d["ssm_n_group"], VT.UINT32, None),
        k("ssm.time_step_rank"):             (d["ssm_dt_rank"], VT.UINT32, None),
        k("ssm.inner_size"):                 (d["ssm_d_state"] * d["ssm_dt_rank"], VT.UINT32, None),
        k("hyper_connection.count"):         (d["hc"], VT.UINT32, None),
        k("hyper_connection.low_rank"):      (d["hc_low_rank"], VT.UINT32, None),
        k("attention.indexer.head_count"):   (d["idx_n_head"], VT.UINT32, None),
        k("attention.indexer.key_length"):   (d["idx_head_dim"], VT.UINT32, None),
        k("attention.indexer.top_k"):        (d["idx_top_k"], VT.UINT32, None),
        k("attention.compress_ratios"):      ([d["compress_ratio"] if (i + 1) % d["full_attn_int"] == 0 else 0
                                               for i in range(d["n_layer"])], VT.ARRAY, VT.INT32),
        k("ple.layers"):                     ([d["ple_layer"]], VT.ARRAY, VT.INT32),
        k("ple.ngram_size"):                 (d["ple_ngram"], VT.UINT32, None),
        k("ple.heads_per_ngram"):            (d["ple_heads_per"], VT.UINT32, None),
        k("ple.conv_kernel"):                (d["ple_conv"], VT.UINT32, None),
        k("ple.eos_token_id"):               (d["n_vocab"] - 1, VT.UINT32, None),
        k("embedding_length_per_layer_input"): (d["ple_head_dim"], VT.UINT32, None),
        k("ple.layer_multipliers"):          (mult, VT.ARRAY, VT.UINT64),
        k("ple.head_offsets"):               ([0] * n_heads, VT.ARRAY, VT.UINT64),   # set by the row cut below
        k("ple.head_vocab_sizes"):           ([1] * n_heads, VT.ARRAY, VT.UINT64),
        # dummy tokenizer, as tests/test-llama-archs.cpp uses: no BOS/EOS, generation runs to n_predict
        "tokenizer.ggml.model":              ("test", VT.STRING, None),
        "tokenizer.ggml.tokens":             ([f"tok_{i}" for i in range(d["n_vocab"])], VT.ARRAY, VT.STRING),
        "tokenizer.ggml.scores":             ([0.0] * d["n_vocab"], VT.ARRAY, VT.FLOAT32),
    }
    return m


def get(meta: dict, key: str, default=None):
    return meta[key][0] if key in meta else default


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="output .gguf (write it to E:, not C:)")
    ap.add_argument("--meta", help="metadata-only first shard of the real split; its KV and tokenizer are copied")
    ap.add_argument("--layers", type=int, default=8, help="trunk layers (default 8: 2 QSA + 6 GDN at interval 4)")
    ap.add_argument("--experts", type=int, default=32, help="experts per layer (the real model has 512; the per-token "
                    "work depends only on the used count)")
    ap.add_argument("--ple-rows", type=int, default=65536, help="PLE table rows per head (the real table is ~40M)")
    ap.add_argument("--no-mtp", action="store_true", help="no nextn block")
    ap.add_argument("--mtp-ratio", type=int, default=0, help="[TAG_SYNC_1004] compress ratio of the nextn block: 0 = dense "
                    "(as the Unsloth head and file A), the trunk ratio = a QSA block (upstream #29761's layout)")
    ap.add_argument("--no-output", action="store_true", help="tie output to token_embd (saves ~0.7 GB)")
    ap.add_argument("--type", choices=["q8_0", "q4_0", "f16"], default="q8_0", help="type of the large matrices")
    ap.add_argument("--sigma", type=float, default=0.02, help="std of the random matrices")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hot-profile-out", default="", help="[TAG_FN_MOE_HOT] also write a random moeprof v1 profile "
                    "for LLAMA_MOE_HOT_PROFILE (Zipf-skewed counts per trunk layer)")
    args = ap.parse_args()

    meta = load_meta(args.meta) if args.meta else default_meta(DEFAULTS)
    if get(meta, "general.architecture") != ARCH:
        print(f"--meta is not a {ARCH} file", file=sys.stderr)
        return 1

    d = dict(DEFAULTS)
    n_src_all   = get(meta, k("block_count"))
    n_src_nextn = get(meta, k("nextn_predict_layers"), 0)
    n_layer_src = n_src_all - n_src_nextn
    tokens = get(meta, "tokenizer.ggml.tokens")
    d["n_vocab"]       = len(tokens) if tokens is not None else get(meta, k("vocab_size"), d["n_vocab"])
    d["n_embd"]        = get(meta, k("embedding_length"), d["n_embd"])
    d["n_head"]        = get(meta, k("attention.head_count"), d["n_head"])
    d["n_head_kv"]     = get(meta, k("attention.head_count_kv"), d["n_head_kv"])
    d["head_dim"]      = get(meta, k("attention.key_length"), d["n_embd"] // d["n_head"])
    d["n_ff_exp"]      = get(meta, k("expert_feed_forward_length"), d["n_ff_exp"])
    d["n_ff_shexp"]    = get(meta, k("expert_shared_feed_forward_length"), d["n_ff_exp"])
    d["n_expert_used"] = get(meta, k("expert_used_count"), d["n_expert_used"])
    d["ssm_d_conv"]    = get(meta, k("ssm.conv_kernel"), d["ssm_d_conv"])
    d["ssm_d_state"]   = get(meta, k("ssm.state_size"), d["ssm_d_state"])
    d["ssm_n_group"]   = get(meta, k("ssm.group_count"), d["ssm_n_group"])
    d["ssm_dt_rank"]   = get(meta, k("ssm.time_step_rank"), d["ssm_dt_rank"])
    d["hc"]            = get(meta, k("hyper_connection.count"), d["hc"])
    d["hc_low_rank"]   = get(meta, k("hyper_connection.low_rank"), d["hc_low_rank"])
    d["idx_n_head"]    = get(meta, k("attention.indexer.head_count"), d["idx_n_head"])
    d["idx_head_dim"]  = get(meta, k("attention.indexer.key_length"), d["idx_head_dim"])
    d["full_attn_int"] = get(meta, k("full_attention_interval"), d["full_attn_int"])

    for kk in ("n_head", "n_head_kv", "n_ff_exp", "n_expert_used"):
        if isinstance(d[kk], list):
            print(f"per-layer {kk} arrays are not handled by this script", file=sys.stderr)
            return 1

    n_trunk = args.layers
    mtp     = not args.no_mtp
    n_all   = n_trunk + (1 if mtp else 0)
    n_expert = args.experts
    if n_expert < d["n_expert_used"]:
        print(f"--experts must be >= expert_used_count ({d['n_expert_used']})", file=sys.stderr)
        return 1

    # which trunk layers are linear attention: the file's list when it has one, else the interval rule
    recr_src = get(meta, k("attention.recurrent_layers"))
    if recr_src is not None:
        recr = [bool(x) for x in recr_src[:n_trunk]]
    else:
        recr = [(i + 1) % d["full_attn_int"] != 0 for i in range(n_trunk)]
    if all(recr):
        print("the cut keeps no full attention layer; use --layers >= the full attention interval", file=sys.stderr)
        return 1

    ratios_src = get(meta, k("attention.compress_ratios"))
    ratio_full = max(ratios_src) if ratios_src else DEFAULTS["compress_ratio"]
    ratios = [0 if recr[i] else ratio_full for i in range(n_trunk)]

    # the cut: layer and expert counts, per-layer arrays, the nextn block
    def put(key, val, vtype, sub=None):
        meta[key] = (val, vtype, sub)

    # any other per-layer array keeps the trunk entries it has; the nextn entry is the file's own nextn entry, or
    # else that of the first full attention layer
    full0 = next(i for i in range(n_trunk) if not recr[i])
    handled = (k("attention.compress_ratios"), k("attention.recurrent_layers"))
    for key, (val, vtype, sub) in list(meta.items()):
        if (not key.startswith(ARCH + ".") or key.startswith(k("ple.")) or key in handled or vtype != VT.ARRAY
                or not isinstance(val, list) or len(val) != n_src_all):
            continue
        tail = val[n_layer_src] if n_src_nextn > 0 else val[full0]
        put(key, val[:n_trunk] + [tail] * (n_all - n_trunk), vtype, sub)
        print(f"note: per-layer array {key} cut to {n_all} entries")

    put(k("block_count"),  n_all, VT.UINT32)
    put(k("expert_count"), n_expert, VT.UINT32)
    # the MTP block is dense unless --mtp-ratio makes it a QSA layer; QSA layers share one ratio
    if args.mtp_ratio not in (0, ratio_full):
        print(f"--mtp-ratio must be 0 or the trunk ratio {ratio_full}", file=sys.stderr)
        return 1
    put(k("attention.compress_ratios"), ratios + [args.mtp_ratio] * (n_all - n_trunk), VT.ARRAY, VT.INT32)
    if recr_src is not None:
        put(k("attention.recurrent_layers"), recr + [False] * (n_all - n_trunk), VT.ARRAY, VT.BOOL)
    if mtp:
        put(k("nextn_predict_layers"), 1, VT.UINT32)
    else:
        meta.pop(k("nextn_predict_layers"), None)

    # PLE: keep the hash constants, cut the rows; the layer must be a kept linear attention layer
    ple = get(meta, k("ple.layers"))
    if ple:
        d["ple_ngram"]     = get(meta, k("ple.ngram_size"))
        d["ple_heads_per"] = get(meta, k("ple.heads_per_ngram"))
        d["ple_conv"]      = get(meta, k("ple.conv_kernel"))
        d["ple_head_dim"]  = get(meta, k("embedding_length_per_layer_input"))
        ple_layer = ple[0] if ple[0] < n_trunk and recr[ple[0]] else next(i for i in range(n_trunk) if recr[i])
        if ple_layer != ple[0]:
            print(f"note: PLE layer {ple[0]} moved to {ple_layer}")
        put(k("ple.layers"), [ple_layer], VT.ARRAY, VT.INT32)
        n_ple_heads = (d["ple_ngram"] - 1) * d["ple_heads_per"]
        put(k("ple.head_vocab_sizes"), [args.ple_rows] * n_ple_heads, VT.ARRAY, VT.UINT64)
        put(k("ple.head_offsets"), [h * args.ple_rows for h in range(n_ple_heads)], VT.ARRAY, VT.UINT64)
    else:
        ple_layer = -1

    # tensors, in ggml order (ne0 first), exactly as llama_model_qwen4exp::load_arch_tensors creates them
    n_embd = d["n_embd"]
    hc     = d["hc"]
    hc_dim = hc * n_embd
    hc_lr  = d["hc_low_rank"]
    n_vocab = d["n_vocab"]
    hd     = d["head_dim"]
    n_head, n_head_kv = d["n_head"], d["n_head_kv"]
    key_dim   = d["ssm_d_state"] * d["ssm_n_group"]
    value_dim = d["ssm_d_state"] * d["ssm_dt_rank"]
    conv_dim  = key_dim * 2 + value_dim
    n_ff, n_ff_sh = d["n_ff_exp"], d["n_ff_shexp"]
    idx_nh, idx_hd = d["idx_n_head"], d["idx_head_dim"]

    # kind: "mat" = random matrix, "norm" = 1 + noise, "zero" = zeros, "neg" = negative (ssm_a), "small" = small f32
    tensors: list[tuple[str, list[int], str]] = []

    def t(name, ne, kind="mat"):
        tensors.append((name, list(ne), kind))

    t("token_embd.weight", [n_embd, n_vocab])
    if not args.no_output:
        t("output.weight", [n_embd, n_vocab])
    t("output_hc_norm.weight", [hc_dim], "norm")
    t("output_hc_down.weight", [hc_dim, hc_lr])
    t("output_hc_up.weight",   [hc_lr, hc_dim])
    if ple_layer >= 0:
        n_ple_heads = (d["ple_ngram"] - 1) * d["ple_heads_per"]
        t("per_layer_token_embd.weight", [d["ple_head_dim"], n_ple_heads * args.ple_rows])

    def block(il, full):
        b = f"blk.{il}."
        for m in ("attn", "ffn"):
            t(b + f"hc_{m}_norm.weight", [hc_dim], "norm")
            t(b + f"hc_{m}_down.weight", [hc_dim, hc_lr])
            t(b + f"hc_{m}_up.weight",   [hc_lr, hc_dim])
            t(b + f"hc_{m}_inject.weight", [hc_dim, hc])
        if full:
            t(b + "attn_q.weight",      [n_embd, hd * n_head * 2])
            t(b + "attn_k.weight",      [n_embd, hd * n_head_kv])
            t(b + "attn_v.weight",      [n_embd, hd * n_head_kv])
            t(b + "attn_output.weight", [hd * n_head, n_embd])
            t(b + "attn_q_norm.weight", [hd], "norm")
            t(b + "attn_k_norm.weight", [hd], "norm")
            t(b + "indexer.q_proj.weight", [n_embd, idx_nh * idx_hd])
            t(b + "indexer.k_proj.weight", [n_embd, idx_hd])
            t(b + "indexer.q_norm.weight", [idx_hd], "norm")
            t(b + "indexer.k_norm.weight", [idx_hd], "norm")
        else:
            t(b + "attn_qkv.weight",   [n_embd, conv_dim])
            t(b + "attn_gate.weight",  [n_embd, value_dim])
            t(b + "ssm_conv1d.weight", [d["ssm_d_conv"], conv_dim], "small")
            t(b + "ssm_dt.bias",       [d["ssm_dt_rank"]], "zero")
            t(b + "ssm_a",             [d["ssm_dt_rank"]], "neg")
            t(b + "ssm_beta.weight",   [n_embd, d["ssm_dt_rank"]])
            t(b + "ssm_alpha.weight",  [n_embd, d["ssm_dt_rank"]])
            t(b + "ssm_norm.weight",   [d["ssm_d_state"]], "norm")
            t(b + "ssm_out.weight",    [value_dim, n_embd])
        if il == ple_layer:
            t(b + "ple_key.weight",        [n_embd, hc_dim])
            t(b + "ple_value.weight",      [n_embd, n_embd])
            t(b + "ple_norm_key.weight",   [hc_dim], "norm")
            t(b + "ple_norm_query.weight", [hc_dim], "norm")
            t(b + "ple_norm_conv.weight",  [hc_dim], "norm")
            t(b + "ple_conv1d.weight",     [d["ple_conv"], hc_dim], "small")
        t(b + "ffn_gate_inp.weight",  [n_embd, n_expert], "small")
        t(b + "ffn_down_exps.weight", [n_ff, n_embd, n_expert])
        t(b + "ffn_gate_exps.weight", [n_embd, n_ff, n_expert])
        t(b + "ffn_up_exps.weight",   [n_embd, n_ff, n_expert])
        t(b + "ffn_gate_inp_shexp.weight", [n_embd], "small")
        t(b + "ffn_gate_shexp.weight", [n_embd, n_ff_sh])
        t(b + "ffn_up_shexp.weight",   [n_embd, n_ff_sh])
        t(b + "ffn_down_shexp.weight", [n_ff_sh, n_embd])

    for il in range(n_trunk):
        block(il, not recr[il])
    if mtp:
        il = n_trunk
        block(il, True)
        b = f"blk.{il}.nextn."
        t(b + "eh_proj.weight",      [2 * n_embd, n_embd])
        t(b + "enorm.weight",        [n_embd], "norm")
        t(b + "hnorm.weight",        [hc_dim], "norm")
        t(b + "hc_head_norm.weight", [hc_dim], "norm")
        t(b + "hc_head_down.weight", [hc_dim, hc_lr])
        t(b + "hc_head_up.weight",   [hc_lr, hc_dim])

    big = {"q8_0": QT.Q8_0, "q4_0": QT.Q4_0, "f16": QT.F16}[args.type]

    def qtype(ne, kind):
        if kind != "mat":
            return QT.F32
        n = int(np.prod(ne))
        if len(ne) >= 2 and ne[0] % 32 == 0 and n >= (1 << 20):
            return big
        return QT.F16

    def nbytes(ne, q):
        bs, ts = gguf.GGML_QUANT_SIZES[q]
        return int(np.prod(ne)) // bs * ts

    rng = np.random.default_rng(args.seed)

    def make(ne, kind, q):
        n = int(np.prod(ne))
        if kind == "norm":
            return (1.0 + 0.02 * rng.standard_normal(n)).astype(np.float32)
        if kind == "zero":
            return np.zeros(n, dtype=np.float32)
        if kind == "neg":
            return (-rng.uniform(0.5, 1.5, n)).astype(np.float32)
        if kind == "small":
            return (0.02 * rng.standard_normal(n)).astype(np.float32)
        if q == QT.F16:
            return (args.sigma * rng.standard_normal(n)).astype(np.float16)
        nb = n // 32
        if q == QT.Q8_0:
            # uniform int8 in [-127, 127] has std ~73.3, so d = sigma/73.3 gives the requested std
            dd = np.full((nb, 1), args.sigma / 73.3, dtype=np.float16).view(np.uint8)
            qs = rng.integers(-127, 128, size=(nb, 32), dtype=np.int8).view(np.uint8)
            return np.concatenate([dd, qs], axis=1).reshape(-1)
        if q == QT.Q4_0:
            # uniform nibbles minus 8 have std ~4.61
            dd = np.full((nb, 1), args.sigma / 4.61, dtype=np.float16).view(np.uint8)
            qs = rng.integers(0, 256, size=(nb, 16), dtype=np.uint8)
            return np.concatenate([dd, qs], axis=1).reshape(-1)
        raise ValueError(q)

    w = gguf.GGUFWriter(None, ARCH)
    if "general.alignment" in meta:
        w.data_alignment = int(get(meta, "general.alignment"))
    for key, (val, vtype, sub) in meta.items():
        w.add_key_value(key, val, vtype, sub_type=sub)

    total = 0
    for name, ne, kind in tensors:
        q = qtype(ne, kind)
        nb = nbytes(ne, q)
        total += nb
        # numpy order is the reverse of ggml's ne
        w.add_tensor_info(name, list(reversed(ne)), np.dtype(np.float32), nb, raw_dtype=q)

    print(f"{len(tensors)} tensors, {total / 2**30:.2f} GiB, {n_trunk} trunk layers "
          f"({sum(1 for r in recr if not r)} full attention), mtp={mtp}, {n_expert} experts, vocab {n_vocab}")

    out = Path(args.out)
    w.write_header_to_file(out)
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for name, ne, kind in tensors:
        q = qtype(ne, kind)
        w.write_tensor_data(make(ne, kind, q))
    w.close()
    print(f"wrote {out}")
    if args.hot_profile_out:
        # [TAG_FN_MOE_HOT] a skewed random profile: tools/qwen4exp/fn_synth_check.py runs hot on / off against it
        prng = np.random.default_rng(args.seed + 1)
        zipf = 1.0 / np.arange(1, n_expert + 1) ** 0.9
        lines = [f"moeprof v1 n_layer={n_trunk} n_expert={n_expert} steps=1000 source=synth_gguf"]
        for il in range(n_trunk):
            counts = (zipf[prng.permutation(n_expert)] * 1000).astype(np.int64) + 1
            lines.append("decode_union %d %s" % (il, " ".join(str(int(c)) for c in counts)))
        Path(args.hot_profile_out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {args.hot_profile_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
