#!/usr/bin/env python3
# [TAG_DFL_HEAD] Draft vocabulary for the DFlash2 drafter head, from token counts of local model output.
#
# The drafter scores every target token on every draft position through the target's lm_head (Qwen3.8-27B: 248,320
# rows of Q6_K, 1.04 GB per drafter step). A draft vocabulary reads fewer rows. It changes which tokens the drafter
# can propose, so it changes acceptance, never the output: the target verifies every draft with its full head.
#
#   stats  count target token ids and print the coverage of a head prefix (LLAMA_DFLASH_HEAD_ROWS=N plus the rows
#          the loader adds on its own) and of a calibrated list of each size (5-fold cross-validated)
#   extra  write a LLAMA_DFLASH_HEAD_EXTRA file: the ids >= --rows seen at least --min-count times
#   build  write a DFlash drafter GGUF that carries its own head: the target lm_head rows of a calibrated list, byte
#          for byte, plus an i32 d2t (run it with LLAMA_DFLASH_D2T_COMPACT=1)
#   ids    [TAG_FN_MTP_HEAD_IDS] write a calibrated list as an id file for LLAMA_MTP_HEAD_IDS (qwen4exp MTP drafts;
#          the loader copies those head rows once and adds the control tokens itself)
#
# Token ids come from JSON/JSONL files (every "tokens" list of ints, e.g. a harness that saved the server's tokens;
# identical sequences count once) or from plain text tokenized with --tokenizer (a byte-level BPE tokenizer.json).
#
#   python scripts/dflash-draft-vocab.py stats --target Qwen3.8-27B-UD-Q5_K_XL.gguf --ids E:/runs --rows 94208,98304
#   python scripts/dflash-draft-vocab.py build --target T.gguf --draft D.gguf --ids E:/runs --size 65536 -o D-v64k.gguf

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import unicodedata
from pathlib import Path

import numpy as np

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent.parent / 'gguf-py'))
import gguf  # noqa: E402

TOKEN_TYPE_NORMAL       = 1
TOKEN_TYPE_CONTROL      = 3
TOKEN_TYPE_USER_DEFINED = 4


def bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('\xa1'), ord('\xac') + 1)) + list(range(ord('\xae'), ord('\xff') + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


class Vocab:
    # the target vocabulary: byte strings and token types, read from the target GGUF
    def __init__(self, reader: gguf.GGUFReader):
        f = reader.get_field('tokenizer.ggml.tokens')
        t = reader.get_field('tokenizer.ggml.token_type')
        if f is None or t is None:
            raise SystemExit('target GGUF has no tokenizer.ggml.tokens / token_type')
        u2b = {v: k for k, v in bytes_to_unicode().items()}
        self.types = np.array([int(t.parts[i][0]) for i in t.data], dtype=np.int32)
        self.pieces: list[bytes] = []
        for i in f.data:
            s = bytes(f.parts[i]).decode('utf-8', errors='replace')
            try:
                self.pieces.append(bytes(u2b[ch] for ch in s))
            except KeyError:
                self.pieces.append(s.encode('utf-8'))
        self.n = len(self.pieces)

    def special(self, i: int) -> bool:
        return int(self.types[i]) in (TOKEN_TYPE_CONTROL, TOKEN_TYPE_USER_DEFINED)

    def punct(self, i: int) -> bool:
        # same rule as dfl_head_extra_token() in src/models/dflash.cpp
        if int(self.types[i]) != TOKEN_TYPE_NORMAL:
            return False
        b = self.pieces[i]
        return len(b) > 0 and all((0x20 <= c < 0x7f or c in (9, 10, 13)) and not chr(c).isalnum() for c in b)

    def loader_extra(self, rows: int) -> list[int]:
        # the rows the loader adds past LLAMA_DFLASH_HEAD_ROWS on its own
        return [i for i in range(rows, self.n) if self.special(i) or self.punct(i)]


class BPE:
    # byte-level BPE from a HF tokenizer.json (Qwen, GPT-2 style), only to count tokens of plain text
    def __init__(self, path: Path):
        import regex
        t = json.loads(path.read_text(encoding='utf-8'))
        self.vocab = t['model']['vocab']
        self.ranks = {}
        for i, m in enumerate(t['model']['merges']):
            a, b = m.split(' ', 1) if isinstance(m, str) else m
            self.ranks[(a, b)] = i
        pats = [p['pattern']['Regex'] for p in t['pre_tokenizer'].get('pretokenizers', [t['pre_tokenizer']]) if p.get('type') == 'Split']
        if not pats:
            raise SystemExit('tokenizer.json: no Split pre-tokenizer')
        self.pat = regex.compile(pats[0])
        self.added = {a['content']: a['id'] for a in t.get('added_tokens', [])}
        self.added_re = regex.compile('|'.join(regex.escape(k) for k in sorted(self.added, key=len, reverse=True))) if self.added else None
        self.b2u = bytes_to_unicode()
        self.cache: dict[str, list[int]] = {}

    def _bpe(self, word: str) -> list[int]:
        ids = self.cache.get(word)
        if ids is not None:
            return ids
        parts = list(word)
        while len(parts) > 1:
            best = None
            for i in range(len(parts) - 1):
                r = self.ranks.get((parts[i], parts[i + 1]))
                if r is not None and (best is None or r < best[0]):
                    best = (r, i)
            if best is None:
                break
            i = best[1]
            parts = parts[:i] + [parts[i] + parts[i + 1]] + parts[i + 2:]
        ids = [self.vocab[p] for p in parts]
        self.cache[word] = ids
        return ids

    def _plain(self, text: str) -> list[int]:
        out: list[int] = []
        for m in self.pat.findall(unicodedata.normalize('NFC', text)):
            out.extend(self._bpe(''.join(self.b2u[b] for b in m.encode('utf-8'))))
        return out

    def encode(self, text: str) -> list[int]:
        if self.added_re is None:
            return self._plain(text)
        out: list[int] = []
        pos = 0
        for m in self.added_re.finditer(text):
            out.extend(self._plain(text[pos:m.start()]))
            out.append(self.added[m.group(0)])
            pos = m.end()
        out.extend(self._plain(text[pos:]))
        return out


def iter_files(paths: list[str], exts: tuple[str, ...]):
    for p in paths:
        q = Path(p)
        if q.is_dir():
            for f in sorted(q.rglob('*')):
                if f.is_file() and f.suffix.lower() in exts:
                    yield f
        elif q.is_file():
            yield q
        else:
            print(f'warning: {p} not found', file=sys.stderr)


def collect_token_lists(x, out: list[tuple[str, list[int]]]):
    # (group, tokens): runs of one prompt are near copies, so they share a fold. The group is the record's prompt name,
    # or else its first 12 tokens (runs of one prompt mostly open the same way)
    if isinstance(x, dict):
        t = x.get('tokens')
        if isinstance(t, list) and len(t) >= 8 and all(isinstance(v, int) for v in t[:8]):
            out.append((str(x.get('prompt') or '') or 'head:' + ','.join(map(str, t[:12])), t))
        for v in x.values():
            if isinstance(v, (dict, list)):
                collect_token_lists(v, out)
    elif isinstance(x, list):
        for v in x:
            if isinstance(v, (dict, list)):
                collect_token_lists(v, out)


def load_sequences(args, n_vocab: int) -> tuple[list[np.ndarray], list[str]]:
    seqs: dict[tuple, tuple[str, np.ndarray]] = {}
    for f in iter_files(args.ids or [], ('.json', '.jsonl')):
        if f.stat().st_size > args.max_file_mb * 1e6:
            continue
        lists: list[tuple[str, list[int]]] = []
        try:
            if f.suffix.lower() == '.jsonl':
                for line in f.open(encoding='utf-8'):
                    line = line.strip()
                    if line:
                        collect_token_lists(json.loads(line), lists)
            else:
                collect_token_lists(json.loads(f.read_text(encoding='utf-8')), lists)
        except (ValueError, UnicodeDecodeError) as e:
            print(f'warning: {f}: {e}', file=sys.stderr)
            continue
        for n, (name, t) in enumerate(lists):
            k = tuple(t)
            if k not in seqs and max(t) < n_vocab and min(t) >= 0:
                seqs[k] = (name or f'{f}:{n}', np.array(t, dtype=np.int64))
    if args.text:
        if not args.tokenizer:
            raise SystemExit('--text needs --tokenizer tokenizer.json')
        bpe = BPE(Path(args.tokenizer))
        for f in iter_files(args.text, ('.txt', '.md')):
            ids = bpe.encode(f.read_text(encoding='utf-8', errors='replace'))
            if ids:
                seqs[tuple(ids)] = (str(f), np.array(ids, dtype=np.int64))
    if not seqs:
        raise SystemExit('no token ids found (--ids / --text)')
    return [v[1] for v in seqs.values()], [v[0] for v in seqs.values()]


def counts_of(seqs: list[np.ndarray], n_vocab: int) -> np.ndarray:
    c = np.zeros(n_vocab, dtype=np.int64)
    for s in seqs:
        np.add.at(c, s, 1)
    return c


def calibrated_list(counts: np.ndarray, vocab: Vocab, size: int) -> np.ndarray:
    # control / user-defined tokens, then every seen token by count, then the lowest unseen ids (BPE merge order is a
    # frequency prior: Qwen3.8 keeps its common Latin and code tokens at the low ids)
    chosen = np.zeros(vocab.n, dtype=bool)
    for i in range(vocab.n):
        if vocab.special(i):
            chosen[i] = True
    order = np.lexsort((np.arange(vocab.n), -counts[:vocab.n]))
    for i in order:
        if chosen.sum() >= size or counts[i] <= 0:
            break
        chosen[i] = True
    need = size - int(chosen.sum())
    if need > 0:
        free = np.nonzero(~chosen & (vocab.types == TOKEN_TYPE_NORMAL))[0][:need]
        chosen[free] = True
    ids = np.nonzero(chosen)[0]
    return ids[:size] if len(ids) > size else ids


def cmd_stats(args, vocab: Vocab):
    seqs, groups = load_sequences(args, vocab.n)
    counts = counts_of(seqs, vocab.n)
    total = int(counts.sum())
    print(f'{len(seqs)} sequences ({len(set(groups))} prompts or files), {total} tokens, {int((counts > 0).sum())} distinct ids')
    if args.save_counts:
        nz = np.nonzero(counts)[0]
        Path(args.save_counts).write_text(json.dumps({'n_vocab': vocab.n, 'total': total,
            'counts': {str(int(i)): int(counts[i]) for i in nz}}), encoding='utf-8')
    print('\nhead prefix (LLAMA_DFLASH_HEAD_ROWS) plus the rows the loader adds:')
    print(f'{"rows":>8} {"extra":>6} {"head %":>7} {"coverage":>9} {"no extra":>9}')
    for rows in [int(r) for r in args.rows.split(',') if r]:
        extra = vocab.loader_extra(rows)
        cov_plain = counts[:rows].sum() / total
        cov = (counts[:rows].sum() + counts[extra].sum()) / total
        print(f'{rows:8d} {len(extra):6d} {100.0*(rows + len(extra))/vocab.n:7.2f} {100.0*cov:8.3f}% {100.0*cov_plain:8.3f}%')
    print('\ncalibrated list (d2t), 5-fold cross-validated coverage on held-out prompts:')
    names = sorted(set(groups))
    random.Random(args.seed).shuffle(names)
    fold_of = {g: k % 5 for k, g in enumerate(names)}
    for size in [int(s) for s in args.sizes.split(',') if s]:
        hit = tot = 0
        for k in range(5):
            test = [i for i in range(len(seqs)) if fold_of[groups[i]] == k]
            c = counts_of([seqs[i] for i in range(len(seqs)) if fold_of[groups[i]] != k], vocab.n)
            m = np.zeros(vocab.n, dtype=bool)
            m[calibrated_list(c, vocab, size)] = True
            for i in test:
                hit += int(m[seqs[i]].sum())
                tot += len(seqs[i])
        print(f'{size:8d} {100.0*size/vocab.n:7.2f}% of the head  coverage {100.0*hit/tot:8.3f}%')


def cmd_extra(args, vocab: Vocab):
    counts = counts_of(load_sequences(args, vocab.n)[0], vocab.n)
    rows = int(args.rows)
    auto = set(vocab.loader_extra(rows))
    ids = [i for i in range(rows, vocab.n) if counts[i] >= args.min_count and i not in auto]
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(f'# LLAMA_DFLASH_HEAD_EXTRA for LLAMA_DFLASH_HEAD_ROWS={rows}: ids seen >= {args.min_count} times\n')
        for i in ids:
            f.write(f'{i}  # {vocab.pieces[i].decode("utf-8", errors="replace")!r} x{int(counts[i])}\n')
    print(f'{len(ids)} ids -> {args.output} (the loader adds {len(auto)} more on its own)')


def cmd_build(args, vocab: Vocab, target: gguf.GGUFReader):
    if args.list:
        ids = np.array(sorted({int(t) for t in Path(args.list).read_text(encoding='utf-8').replace(',', ' ').split()}), dtype=np.int64)
    else:
        counts = counts_of(load_sequences(args, vocab.n)[0], vocab.n)
        ids = calibrated_list(counts, vocab, args.size)
    ids = np.sort(ids)
    if len(ids) == 0 or ids[-1] >= vocab.n:
        raise SystemExit('bad id list')

    head = next((t for t in target.tensors if t.name == 'output.weight'), None)
    if head is None:
        raise SystemExit('target GGUF has no output.weight (tied embeddings are not handled)')
    if head.data.shape[0] != vocab.n and head.data.shape[0] < ids[-1] + 1:
        raise SystemExit(f'target head has {head.data.shape[0]} rows, ids go to {ids[-1]}')

    draft = gguf.GGUFReader(args.draft)
    arch = draft.get_field('general.architecture').contents()
    if arch != 'dflash':
        raise SystemExit(f'draft architecture is {arch}, expected dflash')
    names = {t.name for t in draft.tensors}
    if 'd2t' in names or 'output.weight' in names:
        raise SystemExit('draft already has d2t or output.weight')

    writer = gguf.GGUFWriter(args.output, arch)
    align = draft.get_field('general.alignment')
    if align is not None:
        writer.data_alignment = int(align.contents())
    for field in draft.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.'):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

    rows = np.ascontiguousarray(head.data[ids])  # quantized rows are independent: a byte copy of each row
    d2t = ids.astype(np.int32)
    tensors = [(t.name, t.data, t.tensor_type) for t in draft.tensors]
    tensors.append(('output.weight', rows, head.tensor_type))
    tensors.append(('d2t', d2t, gguf.GGMLQuantizationType.I32))
    for name, data, ttype in tensors:
        writer.add_tensor_info(name, data.shape, data.dtype, data.nbytes, ttype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for name, data, ttype in tensors:
        writer.write_tensor_data(data, tensor_endianess=draft.endianess)
    writer.close()
    print(f'{args.output}: {len(ids)} head rows ({100.0*len(ids)/vocab.n:.1f}% of {vocab.n}, {rows.nbytes/1e6:.1f} MB '
          f'{head.tensor_type.name}) + i32 d2t; run with LLAMA_DFLASH_D2T_COMPACT=1')


def cmd_ids(args, vocab: Vocab):
    # [TAG_FN_MTP_HEAD_IDS] the calibrated list of cmd_build as a text id file, one id per line with its piece and count
    counts = counts_of(load_sequences(args, vocab.n)[0], vocab.n)
    ids = np.sort(calibrated_list(counts, vocab, args.size))
    total = int(counts.sum())
    cov = counts[ids].sum() / max(1, total)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(f'# LLAMA_MTP_HEAD_IDS: {len(ids)} ids of {vocab.n} ({100.0*len(ids)/vocab.n:.1f}% of the head), '
                f'{100.0*cov:.3f}% of {total} calibration tokens\n')
        for i in ids:
            f.write(f'{int(i)}  # {vocab.pieces[i].decode("utf-8", errors="replace")!r} x{int(counts[i])}\n')
    print(f'{len(ids)} ids -> {args.output}, in-sample coverage {100.0*cov:.3f}% (stats --sizes {args.size} gives the held-out one)')


def main():
    ap = argparse.ArgumentParser(description='DFlash2 draft vocabulary from local token counts')
    sub = ap.add_subparsers(dest='cmd', required=True)

    def inputs(p):
        p.add_argument('--target', required=True, help='target GGUF (vocabulary, and the lm_head for build)')
        p.add_argument('--ids', action='append', help='JSON/JSONL file or directory with "tokens" lists (repeatable)')
        p.add_argument('--text', action='append', help='plain text file or directory, needs --tokenizer (repeatable)')
        p.add_argument('--tokenizer', help='HF tokenizer.json of the target, for --text')
        p.add_argument('--max-file-mb', type=float, default=50.0, help='skip larger JSON files')

    p = sub.add_parser('stats', help='coverage of head prefixes and calibrated lists')
    inputs(p)
    p.add_argument('--rows', default='65536,81920,90112,94208,96000,98304,131072')
    p.add_argument('--sizes', default='32768,40960,49152,65536,98304')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--save-counts', help='write the token counts as JSON')

    p = sub.add_parser('extra', help='write a LLAMA_DFLASH_HEAD_EXTRA file')
    inputs(p)
    p.add_argument('--rows', required=True)
    p.add_argument('--min-count', type=int, default=3)
    p.add_argument('-o', '--output', required=True)

    p = sub.add_parser('build', help='write a drafter GGUF with its own head rows and an i32 d2t')
    inputs(p)
    p.add_argument('--draft', required=True, help='DFlash drafter GGUF without its own head')
    p.add_argument('--size', type=int, default=65536, help='draft vocabulary size (calibrated list)')
    p.add_argument('--list', help='use these target ids instead (whitespace or comma separated)')
    p.add_argument('-o', '--output', required=True)

    p = sub.add_parser('ids', help='write a calibrated id file for LLAMA_MTP_HEAD_IDS (qwen4exp MTP drafts)')
    inputs(p)
    p.add_argument('--size', type=int, default=40960, help='draft vocabulary size (calibrated list)')
    p.add_argument('-o', '--output', required=True)

    args = ap.parse_args()
    target = gguf.GGUFReader(args.target)
    vocab = Vocab(target)
    if args.cmd == 'stats':
        cmd_stats(args, vocab)
    elif args.cmd == 'extra':
        cmd_extra(args, vocab)
    elif args.cmd == 'ids':
        cmd_ids(args, vocab)
    else:
        cmd_build(args, vocab, target)


if __name__ == '__main__':
    main()
