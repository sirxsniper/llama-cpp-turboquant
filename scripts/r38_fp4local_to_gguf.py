# [TAG_R38_E2] nvidia/Qwen3.8-Flash-Next-FP4-Local (ModelOpt QAD, safetensors) -> one GGUF for the fork.
#
# The tensor list, shapes, metadata (tokenizer, chat template, hparams) come from a TEMPLATE GGUF of the same model
# (the UD-Q4_K_XL MTPq3 shards); the VALUES come from the FP4-Local checkpoint:
#   routed experts (trunk + MTP)  NVFP4, kept exact: E2M1 codes + E4M3 block scales (CSF streams decoded, vLLM
#                                 nvfp4_utils._decode_csf), the FP32 per-expert global scale -> blk.N.ffn_*_exps.scale
#   MXFP8 / BF16 dense tensors    decoded to f32 exactly, then written in the template's type (Q8_0 / BF16 / F32)
#   PLE n-gram table              NVFP4 rows of 160 values (not a multiple of the 64-value ggml NVFP4 block): decoded
#                                 exactly (x global scale) and written as Q8_0 (160 = 5 blocks of 32)
# Every non-expert tensor is checked against the template (cosine / max diff) so a wrong name map or transform fails.
# numpy + gguf-py only (no torch).  Usage:
#   python fp4local_to_gguf.py --src <hf dir> --tmpl <template shard 1> --out <file.gguf> [--check-only] [--only REGEX]
import sys, os, json, struct, re, time, argparse, glob

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gguf-py'))
import numpy as np
import gguf
from gguf import GGMLQuantizationType as QT

# ---------------------------------------------------------------- decoders
def _e4m3_lut():
    lut = np.zeros(256, np.float32)
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        e = (b >> 3) & 0xF
        m = b & 7
        if e == 0xF and m == 7:
            v = np.nan
        elif e == 0:
            v = m * 2.0 ** -9
        else:
            v = (1 + m / 8.0) * 2.0 ** (e - 7)
        lut[b] = s * v
    return lut
E4M3 = _e4m3_lut()
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)

def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)

def decode_csf(fixed, exc):
    # vLLM nvfp4_utils._decode_csf (Apache-2.0): per 16-row slab 16 base bytes then 4-bit codes (low nibble first);
    # scale byte = base + code; exceptions: low 24 bits flat position, high 8 bits the byte
    assert fixed.dtype == np.uint8 and fixed.ndim == 2 and fixed.shape[1] % 16 == 0
    rows = fixed.shape[0] * 16
    cols = (fixed.shape[1] // 16 - 1) * 2
    assert rows * cols <= 1 << 24
    bases = fixed[:, :16].reshape(rows, 1)
    assert (bases <= 240).all()
    packed = fixed[:, 16:].reshape(rows, cols // 2)
    codes = np.stack((packed & 0xF, packed >> 4), axis=-1).reshape(rows, cols)
    out = (bases + codes).astype(np.uint8).reshape(-1)
    words = exc.reshape(-1).astype(np.uint32)
    pos = words & 0xFFFFFF
    if pos.size:
        assert pos[-1] < rows * cols and (pos.size < 2 or (pos[1:] > pos[:-1]).all())
        out[pos] = (words >> 24).astype(np.uint8)
    return out.reshape(rows, cols)

def mxfp8_to_f32(w_u8, s_u8):
    r, c = w_u8.shape
    assert s_u8.shape == (r, c // 32), (w_u8.shape, s_u8.shape)
    assert (s_u8 != 255).all()
    sc = np.ldexp(np.float32(1.0), s_u8.astype(np.int32) - 127).astype(np.float32)
    return (E4M3[w_u8].reshape(r, c // 32, 32) * sc[:, :, None]).reshape(r, c)

def nvfp4_to_f32(w_u8, s_u8, g=1.0):
    r, cb = w_u8.shape
    c = cb * 2
    codes = np.stack((w_u8 & 0xF, w_u8 >> 4), axis=-1).reshape(r, c)
    sc = E4M3[s_u8 & 0x7F]
    return (E2M1[codes].reshape(r, c // 16, 16) * sc[:, :, None] * np.float32(g)).reshape(r, c)

def nvfp4_pack(w_u8, s_u8):
    # upstream conversion/base.py _nvfp4_pack: ggml super-block [4 UE4M3 scales, 32 qs] per 64 values; within a 16-value
    # sub-block qs[j] = v[j] | v[j+8] << 4
    r, cb = w_u8.shape
    nb = s_u8.shape[1]
    assert nb * 8 == cb and nb % 4 == 0
    v = np.stack((w_u8 & 0xF, w_u8 >> 4), axis=-1).reshape(r, nb, 16)
    qs = (v[:, :, :8] | (v[:, :, 8:] << 4)).astype(np.uint8)              # [r, nb, 8]
    d = (s_u8 & 0x7F).reshape(r, nb // 4, 4)
    q = qs.reshape(r, nb // 4, 32)
    return np.concatenate([d, q], axis=-1).reshape(r, (nb // 4) * 36)

# ---------------------------------------------------------------- safetensors
class ST:
    def __init__(self, d):
        self.d = d
        self.map = json.load(open(os.path.join(d, 'model.safetensors.index.json')))['weight_map']
        self.hdr = {}
        self.mm = {}

    def _h(self, f):
        if f not in self.hdr:
            p = os.path.join(self.d, f)
            with open(p, 'rb') as fh:
                n = struct.unpack('<Q', fh.read(8))[0]
                self.hdr[f] = (json.loads(fh.read(n)), 8 + n)
            self.mm[f] = np.memmap(p, dtype=np.uint8, mode='r')
        return self.hdr[f]

    def has(self, name):
        return name in self.map

    def get(self, name):
        f = self.map[name]
        h, base = self._h(f)
        info = h[name]
        a, b = info['data_offsets']
        raw = self.mm[f][base + a: base + b]
        dt = {'U8': np.uint8, 'F8_E4M3': np.uint8, 'BF16': np.uint16, 'F32': np.float32, 'U32': np.uint32,
              'I64': np.int64, 'F16': np.float16}[info['dtype']]
        return np.array(raw).view(dt).reshape(info['shape']), info['dtype']

    def f32(self, name):
        x, dt = self.get(name)
        if dt == 'BF16':
            return bf16_to_f32(x)
        if dt == 'F32':
            return x.astype(np.float32)
        if dt == 'F16':
            return x.astype(np.float32)
        if dt == 'F8_E4M3':
            s, sdt = self.get(name.replace('.weight', '.weight_scale'))
            assert sdt == 'U8', sdt
            return mxfp8_to_f32(x, s)
        if dt == 'U8':  # NVFP4 dense (scale per 16 + scale_2)
            s, _ = self.get(name.replace('.weight', '.weight_scale'))
            g, _ = self.get(name.replace('.weight', '.weight_scale_2'))
            return nvfp4_to_f32(x, s, float(g.reshape(-1)[0]))
        raise ValueError((name, dt))

# ---------------------------------------------------------------- name map
NK, NVPK, HD = 16, 3, 128   # linear attention: key heads, v heads per key head, head dim

def reorder_v(x, axis, hd, nk=NK, nvpk=NVPK):
    sh = list(x.shape)
    if axis < 0:
        axis += len(sh)
    ns = sh[:axis] + [nk, nvpk, hd] + sh[axis + 1:]
    y = x.reshape(ns)
    perm = list(range(len(ns)))
    perm[axis], perm[axis + 1] = perm[axis + 1], perm[axis]
    return np.ascontiguousarray(y.transpose(perm)).reshape(sh)

def hf_prefix(bid):
    return 'mtp.layers.0' if bid == 48 else f'model.language_model.layers.{bid}'

def source(st, gname):
    x = _source(st, gname)
    # the inherited norm.weight rule: zero-centred gammas (+1), except the gated linear-attention norm and the ones
    # _source already shifted
    if x is not None and gname.endswith('norm.weight') and not gname.endswith('ssm_norm.weight') and             not re.search(r'(indexer\.[qk]_norm|ple_norm_(key|query|conv))\.weight$', gname):
        x = x + 1.0
    return x

def _source(st, gname):
    """f32 numpy array (numpy shape = reversed ggml ne) for a dense template tensor, or None if unmapped."""
    m = re.match(r'blk\.(\d+)\.(.+)$', gname)
    if not m:
        G = {
            'token_embd.weight': 'model.language_model.embed_tokens.weight',
            'output.weight': 'lm_head.weight',
            'output_hc_norm.weight': 'model.language_model.hyper_connection_mixer.hc_norm.weight',
            'output_hc_down.weight': 'model.language_model.hyper_connection_mixer.input_mix_weight_down.weight',
            'output_hc_up.weight': 'model.language_model.hyper_connection_mixer.input_mix_weight_up.weight',
        }
        return st.f32(G[gname]) if gname in G else None
    bid, t = int(m.group(1)), m.group(2)
    P = hf_prefix(bid)
    if bid == 48 and t.startswith('nextn.'):
        n = t[6:]
        if n == 'eh_proj.weight':
            return np.concatenate([st.f32('mtp.fc_embedding.weight'), st.f32('mtp.fc_hidden.weight')], axis=1)
        M = {'enorm.weight': 'mtp.pre_fc_norm_embedding.weight', 'hnorm.weight': 'mtp.pre_fc_norm_hidden.weight',
             'hc_head_norm.weight': 'mtp.hyper_connection_mixer.hc_norm.weight',
             'hc_head_down.weight': 'mtp.hyper_connection_mixer.input_mix_weight_down.weight',
             'hc_head_up.weight': 'mtp.hyper_connection_mixer.input_mix_weight_up.weight'}
        return st.f32(M[n]) if n in M else None
    S = {
        'hc_attn_norm.weight': 'attn_hyper_connection.hc_norm.weight',
        'hc_attn_down.weight': 'attn_hyper_connection.input_mix_weight_down.weight',
        'hc_attn_up.weight': 'attn_hyper_connection.input_mix_weight_up.weight',
        'hc_attn_inject.weight': 'attn_hyper_connection.block_inject_weight.weight',
        'hc_ffn_norm.weight': 'mlp_hyper_connection.hc_norm.weight',
        'hc_ffn_down.weight': 'mlp_hyper_connection.input_mix_weight_down.weight',
        'hc_ffn_up.weight': 'mlp_hyper_connection.input_mix_weight_up.weight',
        'hc_ffn_inject.weight': 'mlp_hyper_connection.block_inject_weight.weight',
        'attn_q.weight': 'self_attn.q_proj.weight', 'attn_k.weight': 'self_attn.k_proj.weight',
        'attn_v.weight': 'self_attn.v_proj.weight', 'attn_output.weight': 'self_attn.o_proj.weight',
        'attn_q_norm.weight': 'self_attn.q_norm.weight', 'attn_k_norm.weight': 'self_attn.k_norm.weight',
        'ffn_gate_inp.weight': 'mlp.gate.weight',
        'ffn_gate_shexp.weight': 'mlp.shared_expert.gate_proj.weight',
        'ffn_up_shexp.weight': 'mlp.shared_expert.up_proj.weight',
        'ffn_down_shexp.weight': 'mlp.shared_expert.down_proj.weight',
        'ple_key.weight': 'ple.key_proj.weight', 'ple_value.weight': 'ple.value_proj.weight',
        'ssm_norm.weight': 'linear_attn.norm.weight',
    }
    if t in S:
        return st.f32(f'{P}.{S[t]}')
    if t == 'ffn_gate_inp_shexp.weight':
        return st.f32(f'{P}.mlp.shared_expert_gate.weight').reshape(-1)
    if t in ('indexer.q_proj.weight', 'indexer.k_proj.weight'):
        qk = st.f32(f'{P}.self_attn.indexer.index_qk_proj.weight')
        nq = 4 * 128
        return qk[:nq] if t.startswith('indexer.q') else qk[nq:]
    # zero-centred gammas (+1), as conversion/qwen4exp.py
    Z = {'indexer.q_norm.weight': 'self_attn.indexer.q_layernorm.weight',
         'indexer.k_norm.weight': 'self_attn.indexer.k_layernorm.weight',
         'ple_norm_key.weight': 'ple.norm_key.weight', 'ple_norm_query.weight': 'ple.norm_query.weight',
         'ple_norm_conv.weight': 'ple.norm_conv.weight'}
    if t in Z:
        return st.f32(f'{P}.{Z[t]}') + 1.0
    if t == 'ple_conv1d.weight':
        return st.f32(f'{P}.ple.conv1d.weight').reshape(st.get(f'{P}.ple.conv1d.weight')[0].shape[0], -1)
    # linear attention (V heads grouped -> tiled, _LinearAttentionVReorderBase)
    LA = f'{P}.linear_attn'
    if t == 'attn_qkv.weight':
        x = st.f32(f'{LA}.in_proj_qkv.weight')
        nqk = 2 * NK * 128
        return np.concatenate([x[:nqk], reorder_v(x[nqk:], 0, HD)], axis=0)
    if t == 'attn_gate.weight':
        return reorder_v(st.f32(f'{LA}.in_proj_z.weight'), 0, HD)
    if t == 'ssm_alpha.weight':
        return reorder_v(st.f32(f'{LA}.in_proj_a.weight'), 0, 1)
    if t == 'ssm_beta.weight':
        return reorder_v(st.f32(f'{LA}.in_proj_b.weight'), 0, 1)
    if t == 'ssm_out.weight':
        return reorder_v(st.f32(f'{LA}.out_proj.weight'), 1, HD)
    if t == 'ssm_a':
        return reorder_v(-np.exp(st.f32(f'{LA}.A_log')), 0, 1)
    if t == 'ssm_dt.bias':
        return reorder_v(st.f32(f'{LA}.dt_bias'), 0, 1)
    if t == 'ssm_conv1d.weight':
        raw = st.get(f'{LA}.conv1d.weight')[0]
        x = st.f32(f'{LA}.conv1d.weight').reshape(raw.shape[0], -1)   # [C, K]
        nqk = 2 * NK * 128
        return np.concatenate([x[:nqk], reorder_v(x[nqk:], 0, HD)], axis=0)
    return None

# ---------------------------------------------------------------- template
def load_template(path1):
    files = sorted(glob.glob(re.sub(r'-00001-of-(\d+)\.gguf$', r'-*-of-\1.gguf', path1)))
    readers = [gguf.GGUFReader(f) for f in files]
    tensors = []
    for r in readers:
        for t in r.tensors:
            tensors.append(t)
    return readers[0], tensors

def tmpl_f32(t):
    return gguf.quants.dequantize(np.asarray(t.data), t.tensor_type).astype(np.float32).reshape([int(x) for x in reversed(t.shape)])

def cos(a, b):
    a = a.reshape(-1).astype(np.float64); b = b.reshape(-1).astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))

def to_type(x, qt):
    x = np.ascontiguousarray(x, dtype=np.float32)
    if qt == QT.F32:
        return x, None
    if qt == QT.BF16:
        u = x.view(np.uint32)
        r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)   # round to nearest even (exact for MXFP8 values)
        return r.view(np.uint8).reshape(x.shape[:-1] + (x.shape[-1] * 2,)), QT.BF16
    q = gguf.quants.quantize(x, qt)
    return q, qt

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--tmpl', required=True)
    ap.add_argument('--out')
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--only', default=None)
    ap.add_argument('--log', default=None)
    a = ap.parse_args()
    logf = open(a.log, 'a') if a.log else None
    def log(*x):
        s = ' '.join(str(v) for v in x)
        print(s, flush=True)
        if logf:
            logf.write(s + '\n'); logf.flush()

    st = ST(a.src)
    r0, tensors = load_template(a.tmpl)
    only = re.compile(a.only) if a.only else None
    t0 = time.time()

    # ---- plan
    plan = []   # (name, kind, ggml_type, numpy byte shape / f32 shape, extra)
    expert_names = []
    for t in tensors:
        n = t.name
        if only and not only.search(n):
            continue
        ne = [int(x) for x in t.shape]
        m = re.match(r'blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$', n)
        if m:
            bid, proj = int(m.group(1)), m.group(2)
            ne0, ne1, ne2 = ne
            plan.append((n, 'exps', QT.NVFP4, (ne2, ne1, ne0 // 64 * 36), (bid, proj)))
            plan.append((n.replace('.weight', '.scale'), 'exps_s', QT.F32, (ne2,), (bid, proj)))
            expert_names.append(n)
        elif n == 'per_layer_token_embd.weight':
            ne0, ne1 = ne
            plan.append((n, 'ple', QT.Q8_0, (ne1, ne0 // 32 * 34), (ne0, ne1)))
        else:
            qt = t.tensor_type
            shp = list(reversed(ne))
            if qt in (QT.F32,):
                bshape = tuple(shp)
            elif qt == QT.BF16:
                bshape = tuple(shp[:-1] + [shp[-1] * 2])
            else:
                bs, ts = gguf.GGML_QUANT_SIZES[qt]
                bshape = tuple(shp[:-1] + [shp[-1] // bs * ts])
            plan.append((n, 'dense', qt, bshape, t))
    log(f'plan: {len(plan)} tensors ({len(expert_names)} expert weights) template {len(tensors)}')

    # ---- writer
    w = None
    if not a.check_only:
        w = gguf.GGUFWriter(a.out, 'qwen4exp', use_temp_file=False)
        for f in r0.fields.values():
            if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith('GGUF.') or f.name.startswith('split.'):
                continue
            vt = f.types[0]
            sub = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
            val = f.contents()
            if f.name == 'general.name':
                val = 'Qwen3.8-Flash-Next-FP4-Local (NVIDIA QAD NVFP4 experts) r38-E2'
            w.add_key_value(f.name, val, vt, sub_type=sub)
        w.add_tensor_extra_prec_a4(sorted(expert_names), [False] * len(expert_names))
        for (n, kind, qt, bshape, ex) in plan:
            if qt == QT.F32:
                nbytes = int(np.prod(bshape)) * 4
                w.add_tensor_info(n, bshape, np.float32, nbytes)
            else:
                nbytes = int(np.prod(bshape))
                w.add_tensor_info(n, bshape, np.uint8, nbytes, raw_dtype=qt)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_ti_data_to_file()

    def emit(arr):
        if w is not None:
            w.write_tensor_data(arr)

    def emit_chunks(gen, nbytes):
        # write_tensor_data for a tensor produced in pieces (the PLE table)
        if w is None:
            for _ in gen:
                pass
            return
        fout = w.fout[0]
        name = next(iter(w.tensors[0]))
        ti = w.tensors[0].pop(name)
        assert ti.nbytes == nbytes, (ti.nbytes, nbytes)
        w.write_padding(fout, fout.tell())
        done = 0
        for c in gen:
            c.tofile(fout); done += c.nbytes
        assert done == nbytes, (done, nbytes)
        w.write_padding(fout, nbytes)
        w.state = gguf.gguf_writer.WriterState.WEIGHTS

    tbyname = {t.name: t for t in tensors}
    worst = []
    gu_equal = [0, 0]
    for (n, kind, qt, bshape, ex) in plan:
        ts = time.time()
        if kind == 'exps':
            bid, proj = ex
            P = hf_prefix(bid)
            ne2 = bshape[0]
            out = np.empty(bshape, np.uint8)
            chk = []
            for e in range(ne2):
                base = f'{P}.mlp.experts.{e}.{proj}_proj'
                wq, _ = st.get(base + '.weight')
                if st.has(base + '.weight_scale.nvfp4_csf_fixed'):
                    s = decode_csf(st.get(base + '.weight_scale.nvfp4_csf_fixed')[0], st.get(base + '.weight_scale.nvfp4_csf_exceptions')[0])
                else:
                    s, sdt = st.get(base + '.weight_scale')
                    assert sdt == 'F8_E4M3'
                assert (s & 0x80).sum() == 0, 'negative block scale'
                out[e] = nvfp4_pack(wq, s)
                if e in (0, 1, ne2 // 2, ne2 - 1):
                    chk.append((e, wq, s))
            # packing check: gguf-py's ggml-convention dequant of the packed bytes == the direct E2M1 x E4M3 decode
            t = tbyname[n]
            tm = None
            for (e, wq, s) in chk:
                ref = nvfp4_to_f32(wq, s)
                got = gguf.quants.dequantize(out[e], QT.NVFP4).reshape(ref.shape)
                assert np.array_equal(ref, got), f'{n} expert {e}: NVFP4 packing mismatch'
                if tm is None:
                    tm = np.asarray(t.data)
                g = float(st.get(f'{P}.mlp.experts.{e}.{proj}_proj.weight_scale_2')[0].reshape(-1)[0])
                tref = gguf.quants.dequantize(tm[e], t.tensor_type).reshape(ref.shape)
                c = cos(ref * g, tref)
                worst.append((c, n + f'[{e}]'))
                if c < 0.95:
                    raise SystemExit(f'{n} expert {e}: cosine vs template {c:.4f} < 0.95 (wrong map?)')
            emit(out)
            log(f'{n} NVFP4 {bshape} cos(template) {min(c for c, nn in worst if nn.startswith(n + "[")):.4f} {time.time()-ts:.1f}s')
        elif kind == 'exps_s':
            bid, proj = ex
            P = hf_prefix(bid)
            ne2 = bshape[0]
            sc = np.array([st.get(f'{P}.mlp.experts.{e}.{proj}_proj.weight_scale_2')[0].reshape(-1)[0] for e in range(ne2)], np.float32)
            assert np.isfinite(sc).all() and (sc > 0).all()
            if proj == 'gate':
                su = np.array([st.get(f'{P}.mlp.experts.{e}.up_proj.weight_scale_2')[0].reshape(-1)[0] for e in range(ne2)], np.float32)
                gu_equal[0] += int((su == sc).sum()); gu_equal[1] += ne2
            emit(sc)
        elif kind == 'ple':
            ne0, ne1 = ex
            P = hf_prefix(1)
            E = f'{P}.ple.ple_embedding.ngram_embedding'
            g = float(st.get(E + '.weight_scale_2')[0].reshape(-1)[0])
            nsh = len([k for k in st.map if k.startswith(E + '.shard_') and k.endswith('.weight')])
            rows = 0
            for i in range(nsh):
                rows += st.get(f'{E}.shard_{i}.weight')[0].shape[0]
            assert rows == ne1, (rows, ne1)
            tpl = tbyname[n]
            def gen():
                r_at = 0
                for i in range(nsh):
                    wq, _ = st.get(f'{E}.shard_{i}.weight')
                    s, sdt = st.get(f'{E}.shard_{i}.weight_scale')
                    assert wq.shape[1] * 2 == ne0 and sdt == 'F8_E4M3'
                    x = nvfp4_to_f32(wq, s, g)
                    if i in (0, nsh // 2, nsh - 1):
                        tr = gguf.quants.dequantize(np.asarray(tpl.data[r_at:r_at + 4096]), tpl.tensor_type).reshape(-1, ne0)
                        c = cos(x[:4096], tr)
                        worst.append((c, f'{n}[shard {i}]'))
                        if c < 0.95:
                            raise SystemExit(f'PLE shard {i}: cosine vs template {c:.4f}')
                    q = gguf.quants.quantize(x, QT.Q8_0)
                    r_at += wq.shape[0]
                    if i % 16 == 0:
                        log(f'  PLE shard {i}/{nsh} rows {r_at} {time.time()-ts:.0f}s')
                    yield q
            emit_chunks(gen(), int(np.prod(bshape)))
            log(f'{n} Q8_0 {bshape} (global scale {g:.6g}) {time.time()-ts:.1f}s')
        else:
            t = ex
            x = source(st, n)
            if x is None:
                raise SystemExit(f'no source for {n}')
            ref = tmpl_f32(t)
            if x.size != ref.size:
                raise SystemExit(f'{n}: size {x.shape} vs template {ref.shape}')
            x = x.reshape(ref.shape)
            c = cos(x, ref)
            md = float(np.max(np.abs(x - ref)))
            worst.append((c, n))
            if c < 0.98:
                raise SystemExit(f'{n}: cosine vs template {c:.4f} (max diff {md:.4g}) - wrong map / transform')
            q, qt2 = to_type(x, t.tensor_type)
            if q.nbytes != int(np.prod(bshape)) * (4 if t.tensor_type == QT.F32 else 1):
                raise SystemExit(f'{n}: {q.nbytes} bytes vs plan {bshape}')
            emit(q.reshape(bshape) if t.tensor_type != QT.F32 else q.reshape(bshape))
            if c < 0.999 or 'ssm_a' in n or 'norm' in n:
                log(f'{n} {t.tensor_type.name} cos {c:.5f} maxdiff {md:.4g}')
    if w is not None:
        w.close()
    worst.sort()
    log('worst cosines:', [(round(c, 4), nn) for c, nn in worst[:12]])
    log(f'gate scale == up scale for {gu_equal[0]} of {gu_equal[1]} experts')
    log(f'DONE {time.time()-t0:.0f}s')

if __name__ == '__main__':
    main()
