// [TAG_R34_D] debug tool (r34 item D, Flint batch-size sensitivity): the same tokens through two batch shapes, every
// captured graph tensor compared token by token. Usual common args (-m -ngl -fa -ctk -ctv -c -b -ub -t -dev) plus env:
//   BD_TEXT    text file (tokenized, add_bos per vocab)       BD_OFFSET  first token (default 0)
//   BD_PREFIX  context tokens decoded first (-b batches)      BD_NTAIL   tail tokens captured (default 32)
//   BD_TA      tail batch of run A (default 1)                BD_TB      tail batch of run B (default 8)
//   BD_FILTER  regex of the captured tensor names (default ^(l_out|ffn_moe_topk|result_output))
//   BD_SAVE    write run A's captures to this file (run B skipped)
//   BD_REF     compare run B against this file instead of an in-process run A
// Output: per tensor (graph order) mean / max relative L2 error over the tail tokens, I32 tensors: tokens whose value
// set differs; then the first tensor above 1e-3 / 1e-2 / 1e-1.
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <regex>
#include <set>
#include <string>
#include <vector>

static int env_int(const char * k, int d) { const char * e = getenv(k); return e && e[0] ? atoi(e) : d; }
static std::string env_str(const char * k, const char * d) { const char * e = getenv(k); return e && e[0] ? e : d; }

struct cap_entry {
    int type = 0;                              // 0 float, 1 int
    std::vector<std::vector<float>> tok;       // [tail token] -> values
};

struct stat_entry {
    int    n = 0, n_set_diff = 0, n_size_diff = 0;
    double sum_rel = 0.0, max_rel = 0.0;
    int    max_tok = -1;
    double amax_a = 0.0, amax_b = 0.0;   // largest |value| seen in run A / run B
};

struct bd_state {
    std::regex                         filter;
    bool                               capture = false;
    int                                t0 = 0;       // tail index of the ubatch's first token
    int                                n_tok = 0;    // tokens of the current batch
    int                                mode = 0;     // 0 = store (run A), 1 = compare (run B)
    std::map<std::string, cap_entry>   ref;
    std::vector<std::string>           order;        // graph order of first capture
    std::map<std::string, stat_entry>  st;
    std::vector<float>                 buf;
    std::map<std::string, int>         occ;          // uses of a name in the current decode (repeated cb names)
};

static bool tensor_to_float(const ggml_tensor * t, std::vector<float> & out, int & type) {
    if (!ggml_is_contiguous(t)) {
        return false;
    }
    const int64_t n = ggml_nelements(t);
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    out.resize(n);
    type = 0;
    switch (t->type) {
        case GGML_TYPE_F32: memcpy(out.data(), raw.data(), n*sizeof(float)); break;
        case GGML_TYPE_F16: for (int64_t i = 0; i < n; ++i) out[i] = ggml_fp16_to_fp32(((const ggml_fp16_t *) raw.data())[i]); break;
        case GGML_TYPE_BF16: for (int64_t i = 0; i < n; ++i) out[i] = ggml_bf16_to_fp32(((const ggml_bf16_t *) raw.data())[i]); break;
        case GGML_TYPE_I32: type = 1; for (int64_t i = 0; i < n; ++i) out[i] = (float) ((const int32_t *) raw.data())[i]; break;
        default: return false;
    }
    return true;
}

// token slice j of a tensor whose highest dim of size n_tok is the token dim
static void token_slice(const ggml_tensor * t, const std::vector<float> & v, int n_tok, int j, std::vector<float> & out) {
    int d = -1;
    for (int k = 3; k >= 0; --k) { if (t->ne[k] == n_tok) { d = k; break; } }
    out.clear();
    if (d < 0) { if (j == 0) out = v; return; }
    const int64_t * ne = t->ne;
    for (int64_t i3 = 0; i3 < ne[3]; ++i3) for (int64_t i2 = 0; i2 < ne[2]; ++i2)
    for (int64_t i1 = 0; i1 < ne[1]; ++i1) for (int64_t i0 = 0; i0 < ne[0]; ++i0) {
        const int64_t id[4] = { i0, i1, i2, i3 };
        if (id[d] != j) continue;
        out.push_back(v[((i3*ne[2] + i2)*ne[1] + i1)*ne[0] + i0]);
    }
}

static bool bd_cb(struct ggml_tensor * t, bool ask, void * ud) {
    bd_state * s = (bd_state *) ud;
    if (ask) {
        return s->capture && std::regex_search(t->name, s->filter);
    }
    if (!s->capture || !std::regex_search(t->name, s->filter)) {
        return true;
    }
    int type = 0;
    if (!tensor_to_float(t, s->buf, type)) {
        return true;
    }
    // a name the graph uses more than once (Qcur before / after the norm, ...) gets #k for its k-th use in a decode
    const int k_occ = s->occ[t->name]++;
    const std::string name = k_occ == 0 ? std::string(t->name) : std::string(t->name) + "#" + std::to_string(k_occ);
    std::vector<float> sl;
    for (int j = 0; j < s->n_tok; ++j) {
        token_slice(t, s->buf, s->n_tok, j, sl);
        if (sl.empty()) continue;
        const int ti = s->t0 + j;
        if (s->mode == 0) {
            auto & e = s->ref[name];
            if (e.tok.empty()) s->order.push_back(name);
            e.type = type;
            if ((int) e.tok.size() <= ti) e.tok.resize(ti + 1);
            e.tok[ti] = sl;
        } else {
            auto it = s->ref.find(name);
            if (it == s->ref.end() || (int) it->second.tok.size() <= ti) continue;
            const auto & a = it->second.tok[ti];
            auto & st = s->st[name];
            if (a.size() != sl.size()) { st.n_size_diff++; continue; }
            st.n++;
            for (float x : a)  st.amax_a = std::max(st.amax_a, (double) fabsf(x));
            for (float x : sl) st.amax_b = std::max(st.amax_b, (double) fabsf(x));
            if (type == 1) {
                std::multiset<float> sa(a.begin(), a.end()), sb(sl.begin(), sl.end());
                if (sa != sb) st.n_set_diff++;
            } else {
                double num = 0.0, den = 0.0;
                for (size_t i = 0; i < a.size(); ++i) { const double df = (double) sl[i] - a[i]; num += df*df; den += (double) a[i]*a[i]; }
                const double rel = den > 0 ? sqrt(num/den) : sqrt(num);
                st.sum_rel += rel;
                if (rel > st.max_rel || std::isnan(rel)) { st.max_rel = rel; st.max_tok = ti; }
            }
        }
    }
    return true;
}

static bool run_tail(llama_context * ctx, bd_state & s, const std::vector<llama_token> & toks, int n_prefix, int n_tail, int tb, int n_batch) {
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_batch batch = llama_batch_init(std::max(n_batch, tb), 0, 1);
    s.capture = false;
    for (int j = 0; j < n_prefix; ) {
        const int n = std::min(n_batch, n_prefix - j);
        common_batch_clear(batch);
        for (int i = 0; i < n; ++i) common_batch_add(batch, toks[j + i], j + i, { 0 }, false);
        if (llama_decode(ctx, batch) != 0) { LOG_ERR("prefix decode failed at %d\n", j); llama_batch_free(batch); return false; }
        j += n;
    }
    for (int j = 0; j < n_tail; ) {
        const int n = std::min(tb, n_tail - j);
        common_batch_clear(batch);
        for (int i = 0; i < n; ++i) common_batch_add(batch, toks[n_prefix + j + i], n_prefix + j + i, { 0 }, true);
        s.capture = true; s.t0 = j; s.n_tok = n; s.occ.clear();
        if (llama_decode(ctx, batch) != 0) { LOG_ERR("tail decode failed at %d\n", j); llama_batch_free(batch); return false; }
        llama_synchronize(ctx);
        s.capture = false;
        j += n;
    }
    llama_batch_free(batch);
    return true;
}

static void save_ref(const bd_state & s, const std::string & path) {
    std::ofstream f(path, std::ios::binary);
    const int32_t nn = (int32_t) s.order.size();
    f.write((const char *) &nn, 4);
    for (const auto & name : s.order) {
        const auto & e = s.ref.at(name);
        const int32_t ln = (int32_t) name.size(), nt = (int32_t) e.tok.size(), ty = e.type;
        f.write((const char *) &ln, 4); f.write(name.data(), ln); f.write((const char *) &ty, 4); f.write((const char *) &nt, 4);
        for (const auto & v : e.tok) { const int32_t sz = (int32_t) v.size(); f.write((const char *) &sz, 4); f.write((const char *) v.data(), sz*4); }
    }
}

static bool load_ref(bd_state & s, const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    int32_t nn = 0; f.read((char *) &nn, 4);
    for (int k = 0; k < nn; ++k) {
        int32_t ln, ty, nt; f.read((char *) &ln, 4); std::string name(ln, '\0'); f.read(&name[0], ln);
        f.read((char *) &ty, 4); f.read((char *) &nt, 4);
        auto & e = s.ref[name]; e.type = ty; e.tok.resize(nt); s.order.push_back(name);
        for (auto & v : e.tok) { int32_t sz; f.read((char *) &sz, 4); v.resize(sz); f.read((char *) v.data(), sz*4); }
    }
    return (bool) f;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    llama_backend_init();
    llama_numa_init(params.numa);

    bd_state s;
    s.filter = std::regex(env_str("BD_FILTER", "^(l_out|ffn_moe_topk|result_output)"));
    params.cb_eval = bd_cb;
    params.cb_eval_user_data = &s;
    params.warmup = false;

    auto init = common_init_from_params(params);
    llama_model * model = init->model();
    llama_context * ctx = init->context();
    if (!model || !ctx) { LOG_ERR("init failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const std::string text_path = env_str("BD_TEXT", "");
    std::ifstream tf(text_path, std::ios::binary);
    if (!tf) { LOG_ERR("BD_TEXT %s not readable\n", text_path.c_str()); return 1; }
    std::string text((std::istreambuf_iterator<char>(tf)), std::istreambuf_iterator<char>());
    std::vector<llama_token> all = common_tokenize(ctx, text, llama_vocab_get_add_bos(vocab), false);
    const int off = env_int("BD_OFFSET", 0), n_prefix = env_int("BD_PREFIX", 1024), n_tail = env_int("BD_NTAIL", 32);
    const int ta = env_int("BD_TA", 1), tb = env_int("BD_TB", 8);
    if ((int) all.size() < off + n_prefix + n_tail) { LOG_ERR("text too short\n"); return 1; }
    std::vector<llama_token> toks(all.begin() + off, all.begin() + off + n_prefix + n_tail);
    if (off > 0 && llama_vocab_get_add_bos(vocab)) toks[0] = llama_vocab_bos(vocab);
    const int n_batch = (int) params.n_batch;

    const std::string save = env_str("BD_SAVE", ""), refp = env_str("BD_REF", "");
    if (!refp.empty()) {
        if (!load_ref(s, refp)) { LOG_ERR("BD_REF %s not readable\n", refp.c_str()); return 1; }
        LOG_INF("bd: reference %s: %zu tensors\n", refp.c_str(), s.order.size());
    } else {
        s.mode = 0;
        if (!run_tail(ctx, s, toks, n_prefix, n_tail, ta, n_batch)) return 1;
        LOG_INF("bd: run A (tail batch %d) captured %zu tensors\n", ta, s.order.size());
        if (!save.empty()) { save_ref(s, save); LOG_INF("bd: saved %s\n", save.c_str()); llama_backend_free(); return 0; }
    }
    s.mode = 1;
    if (!run_tail(ctx, s, toks, n_prefix, n_tail, tb, n_batch)) return 1;

    printf("bd: prefix %d tail %d offset %d | A %s | B tail batch %d\n", n_prefix, n_tail, off,
           refp.empty() ? ("tail batch " + std::to_string(ta)).c_str() : refp.c_str(), tb);
    printf("%-34s %5s %12s %12s %5s %8s %11s %11s\n", "tensor", "n", "mean_rel", "max_rel", "tok", "set_diff", "amax_A", "amax_B");
    const double thr[3] = { 1e-3, 1e-2, 1e-1 };
    std::string first[3];
    for (const auto & name : s.order) {
        auto it = s.st.find(name);
        if (it == s.st.end()) { printf("%-34s (not captured in run B)\n", name.c_str()); continue; }
        const auto & st = it->second;
        const bool is_int = s.ref[name].type == 1;
        if (is_int) {
            printf("%-34s %5d %12s %12s %5s %8d %11s %11s%s\n", name.c_str(), st.n, "-", "-", "-", st.n_set_diff, "-", "-", st.n_size_diff ? " SIZE" : "");
        } else {
            const double mean = st.n ? st.sum_rel/st.n : 0.0;
            printf("%-34s %5d %12.3e %12.3e %5d %8s %11.4g %11.4g%s\n", name.c_str(), st.n, mean, st.max_rel, st.max_tok, "-", st.amax_a, st.amax_b, st.n_size_diff ? " SIZE" : "");
            for (int k = 0; k < 3; ++k) if (first[k].empty() && (st.max_rel > thr[k] || std::isnan(st.max_rel))) first[k] = name;
        }
    }
    for (int k = 0; k < 3; ++k) printf("bd: first tensor with max_rel > %.0e: %s\n", thr[k], first[k].empty() ? "(none)" : first[k].c_str());
    fflush(stdout);
    llama_backend_free();
    return 0;
}
