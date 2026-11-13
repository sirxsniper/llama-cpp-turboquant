import sys, os
# extract.py <llama-kv-cache.cpp> <outdir>: the lend code and state_clear, verbatim
src, out = sys.argv[1], sys.argv[2]
L = open(src).read().split('\n')
def idx(pred, start=0):
    for i in range(start, len(L)):
        if pred(L[i]):
            return i
    raise SystemExit('not found')
tag = idx(lambda s: 'TAG_FN_L8_KVLEND] ----' in s)
ns0 = idx(lambda s: s == 'namespace {', tag)
ns1 = idx(lambda s: s == '} // namespace', ns0)
api0 = idx(lambda s: s.startswith('size_t llama_kv_lend_unmapped_bytes()'))
api1 = idx(lambda s: s.startswith('llama_kv_cache::llama_kv_cache('))
sc0 = idx(lambda s: s.startswith('void llama_kv_cache::state_clear(llama_seq_id seq_id, uint32_t strm, const slot_info & '))
sc1 = idx(lambda s: s == '}', sc0)
open(os.path.join(out, 'ns.inc'), 'w').write('\n'.join(L[ns0 + 1:ns1]) + '\n')
open(os.path.join(out, 'api.inc'), 'w').write('\n'.join(L[api0:api1]) + '\n')
open(os.path.join(out, 'state_clear.inc'), 'w').write('\n'.join(L[sc0:sc1 + 1]) + '\n')
print('ns %d-%d api %d-%d state_clear %d-%d' % (ns0 + 2, ns1, api0 + 1, api1, sc0 + 1, sc1 + 1))
