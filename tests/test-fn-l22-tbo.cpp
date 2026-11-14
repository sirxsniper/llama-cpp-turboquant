// [TAG_FN_L22_TBO] round 22 D5: CPU unit test of the two-chain decode graph (src/llama-fn-tbo.h). The pure parts: the
// switch, the gate, the walk order, the bridge's channel layout. The row views the qwen4exp graph builds for a chain (the
// same fn_tbo functions), checked op by op on the CPU backend: a chain's rows equal the matching rows of the one-chain op,
// bit for bit - flash attention with the positional mask (two sequences, sequence sets) and with the explicit mask, the
// QSA mask, M-RoPE, KV row writes, the gated delta net with its snapshot writes at cell head + chain, the conv history and
// its ring writes. No GPU, no model. Exit code 0 iff every check passes.

#include "llama-fn-tbo.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <utility>
#include <vector>

static int g_fail   = 0;
static int g_checks = 0;

#define TCHECK(cond, ...)                                        \
    do {                                                         \
        ++g_checks;                                              \
        if (!(cond)) {                                           \
            ++g_fail;                                            \
            fprintf(stderr, "FAIL %s:%d: ", __func__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                        \
            fprintf(stderr, "\n");                               \
        }                                                        \
    } while (0)

namespace {

int g_threads = 4;

// ---------------------------------------------------------------------------------------------------------------------
// pure parts

void test_parse_mode() {
    TCHECK(fn_tbo::parse_mode(nullptr) == 0, "null");
    TCHECK(fn_tbo::parse_mode("") == 0, "empty");
    TCHECK(fn_tbo::parse_mode("0") == 0, "0");
    TCHECK(fn_tbo::parse_mode("1") == 1, "1");
    TCHECK(fn_tbo::parse_mode("2") == 2, "2");
    TCHECK(fn_tbo::parse_mode(" 1") == 1, "space 1");
    TCHECK(fn_tbo::parse_mode("on") == 1, "on");
    TCHECK(fn_tbo::parse_mode("ON") == 1, "ON");
    TCHECK(fn_tbo::parse_mode("off") == 0, "off");
    TCHECK(fn_tbo::parse_mode("true") == 1, "true");
    TCHECK(fn_tbo::parse_mode("yes") == 1, "yes");
    TCHECK(fn_tbo::parse_mode("no") == 0, "no");
    TCHECK(fn_tbo::parse_mode("false") == 0, "false");
    TCHECK(fn_tbo::post_key(7, 0) == 7, "chain 0 keeps the layer key");
    TCHECK(fn_tbo::post_key(7, 1) == 7 + fn_tbo::KEY_STRIDE, "chain 1 key");
    {
        // every (layer, chain) of a model below the stride has its own key
        std::vector<int> keys;
        for (int c = 0; c < fn_tbo::MAX_CHAINS; ++c) {
            for (int il = 0; il < 512; ++il) {
                keys.push_back(fn_tbo::post_key(il, c));
            }
        }
        std::sort(keys.begin(), keys.end());
        TCHECK(std::adjacent_find(keys.begin(), keys.end()) == keys.end(), "post keys collide");
    }
}

fn_tbo::gate_in good_gate(uint32_t n_seq_tokens) {
    fn_tbo::gate_in g;
    g.mode         = 1;
    g.mtp          = false;
    g.bridge       = true;
    g.n_chains_br  = 2;
    g.spin         = true;
    g.equal_seqs   = true;
    g.n_seqs       = 2;
    g.n_seqs_unq   = 2;
    g.n_seq_tokens = n_seq_tokens;
    g.n_tokens     = 2*n_seq_tokens;
    g.n_outputs    = 2*n_seq_tokens;
    g.token        = true;
    g.embd         = false;
    g.kv_unified   = true;
    g.flash_attn   = true;
    g.fits         = true;
    g.layer_inp    = false;
    g.trace        = false;
    g.inputs_ok    = true;
    return g;
}

void test_decide() {
    for (uint32_t T : { 1u, 2u, 3u, 4u }) {
        const fn_tbo::plan p = fn_tbo::decide(good_gate(T));
        TCHECK(p.on && p.n_ch == 2 && p.T == (int64_t) T && p.order == 1, "2 x %u all-output unified: on (%s)", T, p.why);
    }
    {
        fn_tbo::gate_in g = good_gate(3);
        g.mode = 2;
        const fn_tbo::plan p = fn_tbo::decide(g);
        TCHECK(p.on && p.order == 2, "mode 2: sequential order");
    }
    // every single condition turns it off
    struct flip {
        const char * name;
        void (*apply)(fn_tbo::gate_in &);
    };
    const flip flips[] = {
        { "switch off",      [](fn_tbo::gate_in & g) { g.mode = 0; } },
        { "MTP graph",       [](fn_tbo::gate_in & g) { g.mtp = true; } },
        { "no bridge",       [](fn_tbo::gate_in & g) { g.bridge = false; } },
        { "one channel set", [](fn_tbo::gate_in & g) { g.n_chains_br = 1; } },
        { "no bridge chains", [](fn_tbo::gate_in & g) { g.n_chains_br = 0; } },
        { "hostfunc waits",  [](fn_tbo::gate_in & g) { g.spin = false; } },
        { "unequal",         [](fn_tbo::gate_in & g) { g.equal_seqs = false; } },
        { "1 sequence",      [](fn_tbo::gate_in & g) { g.n_seqs = 1; g.n_seqs_unq = 1; g.n_tokens = g.n_seq_tokens; g.n_outputs = g.n_tokens; } },
        { "3 sequences",     [](fn_tbo::gate_in & g) { g.n_seqs = 3; g.n_seqs_unq = 3; g.n_tokens = 3*g.n_seq_tokens; g.n_outputs = g.n_tokens; } },
        { "shared ids",      [](fn_tbo::gate_in & g) { g.n_seqs_unq = 3; } },
        { "row count",       [](fn_tbo::gate_in & g) { g.n_tokens = 5; g.n_outputs = 5; } },
        { "zero rows",       [](fn_tbo::gate_in & g) { g.n_seq_tokens = 0; g.n_tokens = 0; g.n_outputs = 0; } },
        { "a row not output", [](fn_tbo::gate_in & g) { g.n_outputs = g.n_tokens - 1; } },
        { "embd rows",       [](fn_tbo::gate_in & g) { g.embd = true; } },
        { "no tokens",       [](fn_tbo::gate_in & g) { g.token = false; } },
        { "not unified",     [](fn_tbo::gate_in & g) { g.kv_unified = false; } },
        { "no FA",           [](fn_tbo::gate_in & g) { g.flash_attn = false; } },
        { "does not fit",    [](fn_tbo::gate_in & g) { g.fits = false; } },
        { "layer input",     [](fn_tbo::gate_in & g) { g.layer_inp = true; } },
        { "trace",           [](fn_tbo::gate_in & g) { g.trace = true; } },
        { "input layout",    [](fn_tbo::gate_in & g) { g.inputs_ok = false; } },
    };
    for (const flip & f : flips) {
        fn_tbo::gate_in g = good_gate(3);
        f.apply(g);
        const fn_tbo::plan p = fn_tbo::decide(g);
        TCHECK(!p.on && p.order == 0 && p.why != nullptr && strcmp(p.why, "on") != 0, "%s: must be off (%s)", f.name, p.why);
    }
}

void test_walk() {
    for (int order : { 1, 2 }) {
        for (int n_ch = 1; n_ch <= 3; ++n_ch) {
            for (int n_layer = 1; n_layer <= 6; ++n_layer) {
                std::vector<std::pair<int, std::pair<int, int>>> seq; // (kind, (chain, layer))
                fn_tbo::walk(n_layer, n_ch, order, [&](int kind, int c, int il) { seq.push_back({ kind, { c, il } }); });
                TCHECK((int) seq.size() == 2*n_ch*n_layer, "order %d n_ch %d n_layer %d: %zu steps", order, n_ch, n_layer, seq.size());
                // per chain: begin(0) finish(0) begin(1) finish(1) ... in this order
                for (int c = 0; c < n_ch; ++c) {
                    std::vector<std::pair<int, int>> own;
                    for (const auto & s : seq) {
                        if (s.second.first == c) {
                            own.push_back({ s.first, s.second.second });
                        }
                    }
                    bool ok = (int) own.size() == 2*n_layer;
                    for (int i = 0; ok && i < (int) own.size(); ++i) {
                        ok = own[i].first == (i % 2) && own[i].second == i/2;
                    }
                    TCHECK(ok, "order %d n_ch %d n_layer %d chain %d: its steps are not begin/finish per layer in order", order, n_ch, n_layer, c);
                }
                if (order == 1) {
                    // the expected interleave, literally
                    std::vector<std::pair<int, std::pair<int, int>>> want;
                    for (int c = 0; c < n_ch; ++c) {
                        want.push_back({ 0, { c, 0 } });
                    }
                    for (int il = 0; il + 1 < n_layer; ++il) {
                        for (int c = 0; c < n_ch; ++c) {
                            want.push_back({ 1, { c, il } });
                            want.push_back({ 0, { c, il + 1 } });
                        }
                    }
                    for (int c = 0; c < n_ch; ++c) {
                        want.push_back({ 1, { c, n_layer - 1 } });
                    }
                    TCHECK(seq == want, "order 1 n_ch %d n_layer %d: not the interleave", n_ch, n_layer);
                    // finish(L) is followed by the same chain's begin(L + 1) (the combine fuses into the next mixer), and
                    // with two chains or more another chain's step sits between a chain's begin (post) and its finish (wait)
                    for (size_t i = 0; i < seq.size(); ++i) {
                        const int c = seq[i].second.first;
                        const int il = seq[i].second.second;
                        if (seq[i].first == 1 && il + 1 < n_layer) {
                            TCHECK(i + 1 < seq.size() && seq[i + 1].first == 0 && seq[i + 1].second.first == c &&
                                    seq[i + 1].second.second == il + 1, "order 1 n_ch %d n_layer %d: finish(%d, %d) not followed by begin(%d, %d)",
                                    n_ch, n_layer, c, il, c, il + 1);
                        }
                        if (seq[i].first == 0 && n_ch >= 2) {
                            size_t j = i + 1;
                            while (j < seq.size() && !(seq[j].first == 1 && seq[j].second.first == c)) {
                                ++j;
                            }
                            bool other = false;
                            for (size_t k = i + 1; k < j; ++k) {
                                other = other || seq[k].second.first != c;
                            }
                            TCHECK(j < seq.size() && other, "order 1 n_ch %d n_layer %d: nothing of another chain between begin(%d, %d) and its finish",
                                    n_ch, n_layer, c, il);
                        }
                    }
                    // the host serves jobs in post order: every wait waits for a job posted before it, and the first job
                    // not yet waited for is always the oldest posted one (no wait can block on a later post)
                    std::vector<std::pair<int, int>> posted;
                    size_t served = 0;
                    bool fifo_ok = true;
                    for (const auto & s : seq) {
                        if (s.first == 0) {
                            posted.push_back(s.second);
                        } else {
                            fifo_ok = fifo_ok && served < posted.size() && posted[served] == s.second;
                            ++served;
                        }
                    }
                    TCHECK(fifo_ok, "order 1 n_ch %d n_layer %d: waits are not in post order", n_ch, n_layer);
                }
            }
        }
    }
    int n = 0;
    fn_tbo::walk(0, 2, 1, [&](int, int, int) { ++n; });
    fn_tbo::walk(3, 0, 1, [&](int, int, int) { ++n; });
    TCHECK(n == 0, "empty walks");
}

void test_layout() {
    // one chain: today's numbers (layer channels 0..n0-1, PLE n0, defer channels n0 + 1 ..)
    {
        fn_tbo::chan_layout l;
        l.n0 = 48; l.n_ch = 1; l.ple = true; l.defer = true;
        TCHECK(l.layer_chan(0, 5) == 5 && l.layer_chan(1, 5) == -1, "one chain layer channels");
        TCHECK(l.ple_chan() == 48 && l.defer_base() == 49 && l.defer_chan(5) == 54 && l.n_chan() == 97, "one chain: %d %d %d %d",
                l.ple_chan(), l.defer_base(), l.defer_chan(5), l.n_chan());
        l.ple = false;
        TCHECK(l.ple_chan() == -1 && l.defer_base() == 48 && l.n_chan() == 96, "one chain, no PLE");
        l.defer = false;
        TCHECK(l.defer_base() == -1 && l.defer_chan(3) == -1 && l.n_chan() == 48, "one chain, nothing else");
    }
    // two chains: chain 1 right after chain 0, PLE and the defer channels (chain 0 only) after both
    {
        fn_tbo::chan_layout l;
        l.n0 = 48; l.n_ch = 2; l.ple = true; l.defer = true;
        TCHECK(l.layer_chan(0, 0) == 0 && l.layer_chan(1, 0) == 48 && l.layer_chan(1, 47) == 95 && l.layer_chan(2, 0) == -1 &&
               l.layer_chan(1, 48) == -1, "two chains layer channels");
        TCHECK(l.n_layer_chans() == 96 && l.ple_chan() == 96 && l.defer_base() == 97, "two chains: PLE %d defer %d", l.ple_chan(), l.defer_base());
        TCHECK(l.defer_chan(0) == 97 && l.defer_chan(47) == 144 && l.defer_chan(48) == -1 && l.defer_chan(-1) == -1,
                "defer channels of chain 0 only");
        TCHECK(l.n_chan() == 145, "two chains n_chan %d", l.n_chan());
        TCHECK(l.chain_of(5) == 0 && l.chain_of(53) == 1 && l.chain_of(96) == -1 && l.chain_of(-1) == -1, "chain_of");
        // every channel number is used once
        std::vector<int> used((size_t) l.n_chan(), 0);
        for (int c = 0; c < l.n_ch; ++c) {
            for (int i = 0; i < l.n0; ++i) {
                used[(size_t) l.layer_chan(c, i)]++;
            }
        }
        used[(size_t) l.ple_chan()]++;
        for (int i = 0; i < l.n0; ++i) {
            used[(size_t) l.defer_chan(i)]++;
        }
        bool once = true;
        for (int u : used) {
            once = once && u == 1;
        }
        TCHECK(once, "two chains: every channel exactly once");
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// ggml ops on the CPU backend

struct ctx_guard {
    ggml_context * ctx = nullptr;
    explicit ctx_guard(size_t mib) {
        ggml_init_params ip = { mib*1024*1024, nullptr, false };
        ctx = ggml_init(ip);
    }
    ~ctx_guard() {
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

void fill_f32(ggml_tensor * t, std::mt19937 & rng, float scale = 1.0f) {
    std::normal_distribution<float> nd(0.0f, scale);
    float * d = (float *) t->data;
    for (int64_t i = 0; i < ggml_nelements(t); ++i) {
        d[i] = nd(rng);
    }
}

void fill_f16(ggml_tensor * t, std::mt19937 & rng, float scale = 1.0f) {
    std::normal_distribution<float> nd(0.0f, scale);
    ggml_fp16_t * d = (ggml_fp16_t *) t->data;
    for (int64_t i = 0; i < ggml_nelements(t); ++i) {
        d[i] = ggml_fp32_to_fp16(nd(rng));
    }
}

// a contiguous copy of rows [r0, r0 + T) along dim 2 (the chain's own tensor in the graph)
ggml_tensor * rows_copy3(ggml_context * ctx, ggml_tensor * t, int64_t r0, int64_t T) {
    ggml_tensor * c = ggml_new_tensor_4d(ctx, t->type, t->ne[0], t->ne[1], T, t->ne[3]);
    GGML_ASSERT(ggml_is_contiguous(t) && t->ne[3] == 1);
    memcpy(c->data, (const char *) t->data + r0*t->nb[2], (size_t) T*t->nb[2]);
    return c;
}

bool same_bytes(const void * a, const void * b, size_t n) {
    return memcmp(a, b, n) == 0;
}

void compute(ggml_context * ctx, std::initializer_list<ggml_tensor *> outs) {
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    for (ggml_tensor * t : outs) {
        ggml_build_forward_expand(gf, t);
    }
    ggml_graph_compute_with_ctx(ctx, gf, g_threads);
}

// the positions of a unified cache with two sequences: a shared prefix (set 3), then cells of either sequence, some empty;
// the last T cells of each sequence are the ubatch's own (positions last - T + 1 .. last)
struct pos_data {
    std::vector<int32_t> kv;  // [n_kv] positions then [n_kv] sets
    std::vector<int32_t> q;   // [N] positions then [N] bits
};

pos_data make_pos(int64_t n_kv, int64_t T) {
    pos_data d;
    d.kv.assign((size_t) 2*n_kv, 0);
    int32_t p[2] = { 0, 0 };
    for (int64_t j = 0; j < n_kv; ++j) {
        int32_t pos = -1;
        int32_t set = 0;
        if (j < 4) {
            pos = p[0];
            p[0]++;
            p[1] = p[0];
            set = 3;
        } else if (j % 11 != 7) {
            const int s = (int) ((j/5) % 2);
            pos = p[s]++;
            set = 1 << s;
        }
        d.kv[(size_t) j]        = pos;
        d.kv[(size_t) (n_kv + j)] = set;
    }
    const int64_t N = 2*T;
    d.q.assign((size_t) 2*N, 0);
    for (int s = 0; s < 2; ++s) {
        for (int64_t t = 0; t < T; ++t) {
            const int64_t i = s*T + t;
            d.q[(size_t) i]       = std::max<int32_t>(0, p[s] - 1 - (int32_t) (T - 1 - t));
            d.q[(size_t) (N + i)] = 1 << s;
        }
    }
    return d;
}

void test_fa(int64_t T, bool explicit_mask) {
    ctx_guard g(64);
    ggml_context * ctx = g.ctx;
    std::mt19937 rng((unsigned) (17 + T + (explicit_mask ? 100 : 0)));

    const int64_t D = 64, H = 4, H_kv = 2, n_kv = 96, N = 2*T;
    ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, n_kv, H_kv);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, n_kv, H_kv);
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, H, N); // the layout of a projected, roped Q: [D, H, rows]
    fill_f16(k, rng);
    fill_f16(v, rng);
    fill_f32(q, rng);

    const pos_data pd = make_pos(n_kv, T);
    ggml_tensor * kv_pos = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_kv, 2);
    ggml_tensor * q_pos  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N, 2);
    memcpy(kv_pos->data, pd.kv.data(), pd.kv.size()*sizeof(int32_t));
    memcpy(q_pos->data, pd.q.data(), pd.q.size()*sizeof(int32_t));

    // the explicit mask the same positions give (the KQ mask rule of a unified cache)
    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, N, 1, 1);
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < n_kv; ++j) {
            const bool vis = pd.kv[(size_t) j] >= 0 && pd.kv[(size_t) j] <= pd.q[(size_t) i] &&
                             (pd.kv[(size_t) (n_kv + j)] & pd.q[(size_t) (N + i)]) != 0;
            ((ggml_fp16_t *) mask->data)[i*n_kv + j] = ggml_fp32_to_fp16(vis ? 0.0f : -INFINITY);
        }
    }

    const float scale = 1.0f/sqrtf((float) D);
    auto fa = [&](ggml_tensor * qq, ggml_tensor * m, ggml_tensor * qp) {
        ggml_tensor * r = ggml_flash_attn_ext(ctx, ggml_permute(ctx, qq, 0, 2, 1, 3), k, v, m, scale, 0.0f, 0.0f);
        if (qp) {
            ggml_flash_attn_ext_set_pos(r, kv_pos, qp);
        }
        ggml_prec_set_acc(r, GGML_PREC_F32);
        return r;
    };

    ggml_tensor * full = explicit_mask ? fa(q, mask, nullptr) : fa(q, nullptr, q_pos);
    ggml_tensor * part[2];
    for (int c = 0; c < 2; ++c) {
        const int64_t r0 = c*T;
        ggml_tensor * qc = rows_copy3(ctx, q, r0, T);
        part[c] = explicit_mask ? fa(qc, fn_tbo::mask_rows(ctx, mask, r0, T), nullptr)
                                : fa(qc, nullptr, fn_tbo::qpos_rows(ctx, q_pos, r0, T));
    }
    compute(ctx, { full, part[0], part[1] });

    for (int c = 0; c < 2; ++c) {
        TCHECK(part[c]->ne[0] == D && part[c]->ne[1] == H && part[c]->ne[2] == T, "fa chain shape");
        const bool eq = same_bytes((const char *) full->data + c*T*full->nb[2], part[c]->data, (size_t) T*full->nb[2]);
        TCHECK(eq, "flash attention T %lld %s: chain %d rows differ from the one-chain rows", (long long) T,
                explicit_mask ? "explicit mask" : "positional mask", c);
    }
}

void test_qsa_mask(int64_t T) {
    ctx_guard g(32);
    ggml_context * ctx = g.ctx;
    std::mt19937 rng((unsigned) (31 + T));

    const int64_t n_kv = 96, N = 2*T, group = 4, n_top = 8, n_sel = n_top*group + group - 1;
    const pos_data pd = make_pos(n_kv, T);
    ggml_tensor * kv_pos = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_kv, 2);
    ggml_tensor * q_pos  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N, 2);
    memcpy(kv_pos->data, pd.kv.data(), pd.kv.size()*sizeof(int32_t));
    memcpy(q_pos->data, pd.q.data(), pd.q.size()*sizeof(int32_t));

    ggml_tensor * sel  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_sel, N);
    ggml_tensor * live = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_top, N);
    std::uniform_int_distribution<int> cell(-1, (int) n_kv + 1);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    for (int64_t i = 0; i < n_sel*N; ++i) {
        ((int32_t *) sel->data)[i] = cell(rng);
    }
    for (int64_t i = 0; i < n_top*N; ++i) {
        const float r = u(rng);
        ((float *) live->data)[i] = r < 0.2f ? -INFINITY : r < 0.3f ? 0.0f : r;
    }

    ggml_tensor * full = ggml_qsa_mask(ctx, kv_pos, q_pos, sel, live, (int32_t) group);
    ggml_tensor * part[2];
    for (int c = 0; c < 2; ++c) {
        const int64_t r0 = c*T;
        part[c] = ggml_qsa_mask(ctx, kv_pos, fn_tbo::qpos_rows(ctx, q_pos, r0, T), fn_tbo::rows2(ctx, sel, r0, T),
                fn_tbo::rows2(ctx, live, r0, T), (int32_t) group);
    }
    compute(ctx, { full, part[0], part[1] });
    for (int c = 0; c < 2; ++c) {
        TCHECK(same_bytes((const char *) full->data + c*T*full->nb[1], part[c]->data, (size_t) T*full->nb[1]),
                "QSA mask T %lld chain %d rows differ", (long long) T, c);
    }
}

void test_rope(int64_t T) {
    ctx_guard g(32);
    ggml_context * ctx = g.ctx;
    std::mt19937 rng((unsigned) (43 + T));

    const int64_t D = 64, H = 4, N = 2*T;
    ggml_tensor * a   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, H, N);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4*N);
    fill_f32(a, rng);
    for (int64_t s = 0; s < 4; ++s) {
        for (int64_t i = 0; i < N; ++i) {
            ((int32_t *) pos->data)[s*N + i] = (int32_t) (1000*s + 37*i + (i >= T ? 500 : 0)); // distinct per row and section
        }
    }
    int sections[4] = { 12, 10, 10, 0 };
    auto rope = [&](ggml_tensor * x, ggml_tensor * p) {
        return ggml_rope_multi(ctx, x, p, nullptr, (int) D, sections, GGML_ROPE_TYPE_IMROPE, 4096, 10000.0f, 1.0f, 0.0f, 1.0f,
                32.0f, 1.0f);
    };
    ggml_tensor * full = rope(a, pos);
    ggml_tensor * part[2];
    ggml_tensor * pr[2];
    for (int c = 0; c < 2; ++c) {
        const int64_t r0 = c*T;
        pr[c]   = fn_tbo::pos_rows(ctx, pos, N, r0, T);
        part[c] = rope(rows_copy3(ctx, a, r0, T), pr[c]);
    }
    compute(ctx, { full, part[0], part[1], pr[0], pr[1] });
    for (int c = 0; c < 2; ++c) {
        // the chain's positions are the matching entries of every section
        bool pos_ok = pr[c]->ne[0] == 4*T;
        for (int64_t s = 0; pos_ok && s < 4; ++s) {
            for (int64_t t = 0; pos_ok && t < T; ++t) {
                pos_ok = ((const int32_t *) pr[c]->data)[s*T + t] == ((const int32_t *) pos->data)[s*N + c*T + t];
            }
        }
        TCHECK(pos_ok, "M-RoPE positions of chain %d", c);
        TCHECK(same_bytes((const char *) full->data + c*T*full->nb[2], part[c]->data, (size_t) T*full->nb[2]),
                "M-RoPE T %lld chain %d rows differ", (long long) T, c);
    }
}

void test_set_rows(int64_t T) {
    ctx_guard g(32);
    ggml_context * ctx = g.ctx;
    std::mt19937 rng((unsigned) (59 + T));

    const int64_t E = 32, n_cells = 40, N = 2*T;
    ggml_tensor * cache_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, n_cells);
    ggml_tensor * cache_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, n_cells);
    fill_f32(cache_a, rng);
    memcpy(cache_b->data, cache_a->data, ggml_nbytes(cache_a));
    ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, N);
    fill_f32(src, rng);
    ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, N);
    std::vector<int64_t> cells((size_t) n_cells);
    for (int64_t i = 0; i < n_cells; ++i) {
        cells[(size_t) i] = i;
    }
    std::shuffle(cells.begin(), cells.end(), rng);
    for (int64_t i = 0; i < N; ++i) {
        ((int64_t *) idx->data)[i] = cells[(size_t) i];
    }
    ggml_tensor * full = ggml_set_rows(ctx, cache_a, src, idx);
    ggml_tensor * w = cache_b;
    for (int c = 0; c < 2; ++c) {
        w = ggml_set_rows(ctx, w, fn_tbo::rows2(ctx, src, c*T, T), fn_tbo::rows1(ctx, idx, c*T, T));
    }
    compute(ctx, { full, w });
    TCHECK(same_bytes(cache_a->data, cache_b->data, ggml_nbytes(cache_a)), "KV row writes T %lld: caches differ", (long long) T);
}

// the snapshot writes of llm_build_delta_net_base::build_recurrent_attn (no replay ring) for n_seqs sequences from cell
// kv_head (a chain: n_seqs 1, kv_head = head + chain)
void gdn_writes(ggml_context * ctx, std::vector<ggml_tensor *> & outs, ggml_tensor * gdn_out, ggml_tensor * s, ggml_tensor * cache,
        int64_t D, int64_t S_v, int64_t H, int64_t n_seq_tokens, int64_t n_seqs, int64_t K, int64_t kv_head, int64_t mem_size) {
    const int64_t attn_score_elems    = S_v*H*n_seq_tokens*n_seqs;
    const int64_t state_size_per_snap = S_v*S_v*H*n_seqs;
    const size_t  row_size            = (size_t) D*sizeof(float);
    const int64_t n_written           = std::min(n_seq_tokens, K);
    ggml_tensor * src = ggml_view_3d(ctx, gdn_out, D, n_seqs, n_written, ggml_row_size(gdn_out->type, D),
            ggml_row_size(gdn_out->type, state_size_per_snap), ggml_row_size(gdn_out->type, attn_score_elems));
    ggml_tensor * dst = ggml_view_3d(ctx, cache, D, n_seqs, n_written, cache->nb[1], (size_t) mem_size*row_size,
            (size_t) kv_head*row_size);
    outs.push_back(ggml_cpy(ctx, src, dst));
    if (n_seq_tokens < K) {
        ggml_tensor * s_in  = ggml_reshape_2d(ctx, ggml_is_contiguous(s) ? s : ggml_cont(ctx, s), D, n_seqs);
        ggml_tensor * s_dst = ggml_view_2d(ctx, cache, D, n_seqs, cache->nb[1], ((size_t) n_seq_tokens*mem_size + kv_head)*row_size);
        outs.push_back(ggml_cpy(ctx, s_in, s_dst));
    }
}

void test_gdn(int64_t T, int64_t K) {
    ctx_guard g(64);
    ggml_context * ctx = g.ctx;
    std::mt19937 rng((unsigned) (71 + T*7 + K));

    const int64_t S = 16, H = 4, B = 2, D = S*S*H, mem_size = 8, head = 2;
    ggml_tensor * q    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, T, B);
    ggml_tensor * k    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, T, B);
    ggml_tensor * v    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, T, B);
    ggml_tensor * gate = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, T, B);
    ggml_tensor * beta = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, T, B);
    ggml_tensor * rows = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, B); // the one gather of both sequences' states
    fill_f32(q, rng, 0.3f);
    fill_f32(k, rng, 0.3f);
    fill_f32(v, rng, 0.5f);
    fill_f32(gate, rng, 0.2f);
    fill_f32(beta, rng, 0.5f);
    fill_f32(rows, rng, 0.1f);
    for (int64_t i = 0; i < ggml_nelements(gate); ++i) {
        ((float *) gate->data)[i] = -fabsf(((float *) gate->data)[i]); // a log decay
    }
    for (int64_t i = 0; i < ggml_nelements(beta); ++i) {
        ((float *) beta->data)[i] = 1.0f/(1.0f + expf(-((float *) beta->data)[i]));
    }
    ggml_tensor * cache_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, (K + 1)*mem_size);
    ggml_tensor * cache_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, (K + 1)*mem_size);
    fill_f32(cache_a, rng);
    memcpy(cache_b->data, cache_a->data, ggml_nbytes(cache_a));

    std::vector<ggml_tensor *> outs;
    ggml_tensor * s_full = ggml_reshape_4d(ctx, rows, S, S, H, B);
    ggml_tensor * full = ggml_gated_delta_net(ctx, q, k, v, gate, beta, s_full, K);
    outs.push_back(full);
    gdn_writes(ctx, outs, full, s_full, cache_a, D, S, H, T, B, K, head, mem_size);

    auto seq_of = [&](ggml_tensor * t, int64_t c) {
        return ggml_cont(ctx, ggml_view_4d(ctx, t, t->ne[0], t->ne[1], t->ne[2], 1, t->nb[1], t->nb[2], t->nb[3], (size_t) c*t->nb[3]));
    };
    ggml_tensor * part[2];
    for (int c = 0; c < 2; ++c) {
        ggml_tensor * s_c = ggml_reshape_4d(ctx, fn_tbo::col(ctx, rows, c), S, S, H, 1);
        part[c] = ggml_gated_delta_net(ctx, seq_of(q, c), seq_of(k, c), seq_of(v, c), seq_of(gate, c), seq_of(beta, c), s_c, K);
        outs.push_back(part[c]);
        gdn_writes(ctx, outs, part[c], s_c, cache_b, D, S, H, T, 1, K, head + c, mem_size);
    }
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    for (ggml_tensor * t : outs) {
        ggml_build_forward_expand(gf, t);
    }
    ggml_graph_compute_with_ctx(ctx, gf, g_threads);

    const size_t attn_seq = (size_t) S*H*T*sizeof(float);
    for (int c = 0; c < 2; ++c) {
        TCHECK(same_bytes((const char *) full->data + c*attn_seq, part[c]->data, attn_seq), "GDN T %lld K %lld chain %d output differs",
                (long long) T, (long long) K, c);
    }
    TCHECK(same_bytes(cache_a->data, cache_b->data, ggml_nbytes(cache_a)), "GDN T %lld K %lld: snapshot caches differ", (long long) T,
            (long long) K);
}

// build_conv_state_at: the history of the one gather, the chain's rows after it, the convolution, the ring writes
void conv_part(ggml_context * ctx, std::vector<ggml_tensor *> & outs, ggml_tensor * rows_c, ggml_tensor * x_c, ggml_tensor * kern,
        ggml_tensor * cache, int64_t state_cols, int64_t channels, int64_t n_seqs, int64_t n_slots, int64_t kv_head, int64_t mem_size,
        ggml_tensor ** conv_out) {
    ggml_tensor * state      = ggml_reshape_3d(ctx, rows_c, state_cols, channels, n_seqs);
    ggml_tensor * conv_input = ggml_concat(ctx, state, ggml_transpose(ctx, x_c), 0);
    *conv_out = ggml_ssm_conv(ctx, conv_input, kern);
    outs.push_back(*conv_out);
    const size_t row_size = ggml_row_size(cache->type, state_cols*channels);
    for (int64_t slot = 0; slot < n_slots; ++slot) {
        const int64_t s_idx = std::max<int64_t>(0, conv_input->ne[0] - state_cols - slot);
        ggml_tensor * tail = ggml_view_3d(ctx, conv_input, state_cols, channels, n_seqs, conv_input->nb[1], conv_input->nb[2],
                ggml_row_size(conv_input->type, s_idx));
        ggml_tensor * dst = ggml_view_2d(ctx, cache, state_cols*channels, n_seqs, cache->nb[1], (size_t) (slot*mem_size + kv_head)*row_size);
        outs.push_back(ggml_cpy(ctx, tail, dst));
    }
}

void test_conv(int64_t T) {
    ctx_guard g(32);
    ggml_context * ctx = g.ctx;
    std::mt19937 rng((unsigned) (97 + T));

    const int64_t state_cols = 3, channels = 24, B = 2, mem_size = 8, head = 3, n_slots = 4;
    ggml_tensor * rows = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, state_cols*channels, B);
    ggml_tensor * x    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, channels, T, B);
    ggml_tensor * kern = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, state_cols + 1, channels);
    fill_f32(rows, rng);
    fill_f32(x, rng);
    fill_f32(kern, rng);
    ggml_tensor * cache_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, state_cols*channels, n_slots*mem_size);
    ggml_tensor * cache_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, state_cols*channels, n_slots*mem_size);
    fill_f32(cache_a, rng);
    memcpy(cache_b->data, cache_a->data, ggml_nbytes(cache_a));

    std::vector<ggml_tensor *> outs;
    ggml_tensor * full = nullptr;
    conv_part(ctx, outs, rows, x, kern, cache_a, state_cols, channels, B, n_slots, head, mem_size, &full);
    ggml_tensor * part[2];
    for (int c = 0; c < 2; ++c) {
        ggml_tensor * x_c = ggml_cont(ctx, ggml_view_3d(ctx, x, channels, T, 1, x->nb[1], x->nb[2], (size_t) c*x->nb[2]));
        conv_part(ctx, outs, fn_tbo::col(ctx, rows, c), x_c, kern, cache_b, state_cols, channels, 1, n_slots, head + c, mem_size, &part[c]);
    }
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    for (ggml_tensor * t : outs) {
        ggml_build_forward_expand(gf, t);
    }
    ggml_graph_compute_with_ctx(ctx, gf, g_threads);
    for (int c = 0; c < 2; ++c) {
        TCHECK(same_bytes((const char *) full->data + c*full->nb[2], part[c]->data, (size_t) full->nb[2]),
                "conv T %lld chain %d output differs", (long long) T, c);
    }
    TCHECK(same_bytes(cache_a->data, cache_b->data, ggml_nbytes(cache_a)), "conv T %lld: ring caches differ", (long long) T);
}

void test_rows_views() {
    ctx_guard g(8);
    ggml_context * ctx = g.ctx;
    const int64_t T = 3, N = 6;
    ggml_tensor * res = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 4, N);
    for (int64_t i = 0; i < ggml_nelements(res); ++i) {
        ((float *) res->data)[i] = (float) i;
    }
    ggml_tensor * r1 = fn_tbo::rows3(ctx, res, T, T);
    TCHECK(ggml_is_contiguous(r1) && r1->ne[2] == T && ((float *) r1->data)[0] == (float) (T*8*4), "residual rows view");
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 16, N, 1, 1);
    ggml_tensor * mr = fn_tbo::mask_rows(ctx, m, T, T);
    TCHECK(ggml_is_contiguous(mr) && mr->ne[0] == 16 && mr->ne[1] == T && mr->ne[2] == 1 && mr->ne[3] == 1 &&
            (char *) mr->data == (char *) m->data + T*m->nb[1], "mask rows view");
    TCHECK(fn_tbo::mask_rows(ctx, nullptr, 0, T) == nullptr && fn_tbo::rows1(ctx, nullptr, 0, T) == nullptr &&
            fn_tbo::rows2(ctx, nullptr, 0, T) == nullptr && fn_tbo::qpos_rows(ctx, nullptr, 0, T) == nullptr, "null inputs");
    ggml_tensor * st = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 12, 2);
    ggml_tensor * c1 = fn_tbo::col(ctx, st, 1);
    TCHECK(ggml_is_contiguous(c1) && c1->ne[0] == 12 && c1->ne[1] == 1 && (char *) c1->data == (char *) st->data + st->nb[1], "state column");
    ggml_tensor * q1 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    ggml_tensor * qv = fn_tbo::qpos_rows(ctx, q1, T, T);
    TCHECK(qv->op == GGML_OP_VIEW && qv->ne[0] == T, "one-row query positions: a view");
    ggml_tensor * p1 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    ggml_tensor * pv = fn_tbo::pos_rows(ctx, p1, N, T, T);
    TCHECK(pv->op == GGML_OP_VIEW && pv->ne[0] == T, "one-section positions: a view");
}

} // namespace

int main(int argc, char ** argv) {
    g_threads = argc > 1 ? std::max(1, atoi(argv[1])) : 4;
    ggml_cpu_init();

    test_parse_mode();
    test_decide();
    test_walk();
    test_layout();
    test_rows_views();
    for (int64_t T : { 1, 2, 3, 4 }) {
        test_fa(T, false);
        test_fa(T, true);
        test_qsa_mask(T);
        test_rope(T);
        test_set_rows(T);
        test_conv(T);
        test_gdn(T, 4);
        test_gdn(T, 2);
    }
    printf("test-fn-l22-tbo: %d checks, %d failed (%d threads)\n", g_checks, g_fail, g_threads);
    return g_fail == 0 ? 0 : 1;
}
