// [TAG_PFS_LEND_RESERVE] CPU test of the lending prefill stream's reserve views (src/llama-prefill-stream.cpp,
// llama_prefill_stream_reserve_begin / _end). The scheduler has two CPU backends: the first plays the device (the stream's
// compute backend, every CPU buffer maps to it), the second the host that the gate and release nodes are pinned to, so a
// streamed graph splits as on a GPU box. A small MoE graph is built per layer the way build_moe_ffn uses the stream
// (lookup, gate, the router on a device weight, the expert chain on the view's tensors, release). It checks:
//  - a lending stream outside a reserve still builds the op-offload form (no gate or release node), as v0.32.0;
//  - between begin and end its graph has the streamed form, node for node (op, type, shape, backend) the graph of a stream
//    with its own banks (LLAMA_PREFILL_STREAM_LEND=0), with the same splits and the same compute buffer size, and the
//    reserve views' tensors take no compute memory (their buffer has none);
//  - begin refuses without a lending stream, for another owner, below the stream's minimum, and with
//    LLAMA_PREFILL_STREAM_LEND_RESERVE=0;
//  - the scheduler reserving the op-offload graph and then the streamed one keeps the larger buffer and reports the
//    streamed graph's splits; a later smaller reserve does not shrink it (what llama_context::graph_reserve relies on).
// It does not run llama_context::graph_reserve, build_moe_ffn or a CUDA backend: the sizes here are CPU sizes of a toy
// graph, not those of the GPU box. No model, no GPU. Exit code 0 iff every check passes.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llama.h"

#include "../ggml/src/ggml-backend-impl.h"
#include "../src/llama-moe-gen5.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
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

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

constexpr int64_t N_EMBD   = 256;
constexpr int64_t N_FF     = 128;
constexpr int64_t N_EXPERT = 16;
constexpr int64_t N_USED   = 4;
constexpr int     N_LAYER  = 4;
constexpr int64_t MIN_T    = 16;

struct t_model {
    ggml_backend_t        dev  = nullptr; // the "device": the stream's compute backend
    ggml_backend_t        host = nullptr; // the CPU backend the gen5 host nodes are pinned to
    ggml_context *        ctx  = nullptr;
    ggml_backend_buffer_t buf  = nullptr;
    ggml_tensor *         w[N_LAYER][3] = {};
    ggml_tensor *         router[N_LAYER] = {}; // as ffn_gate_inp: a device weight, so the routing runs on the device
};

static bool build_model(t_model & m) {
    m.dev  = ggml_backend_cpu_init();
    m.host = ggml_backend_cpu_init();
    if (!m.dev || !m.host) {
        return false;
    }
    const ggml_type types[N_LAYER][3] = {
        { GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, GGML_TYPE_Q8_0 },
        { GGML_TYPE_Q5_K, GGML_TYPE_Q5_K, GGML_TYPE_Q8_0 },
        { GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_1 },
        { GGML_TYPE_Q6_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0 },
    };
    ggml_init_params ip = { ggml_tensor_overhead()*64, nullptr, true };
    m.ctx = ggml_init(ip);
    for (int l = 0; l < N_LAYER; ++l) {
        m.w[l][0] = ggml_new_tensor_3d(m.ctx, types[l][0], N_EMBD, N_FF, N_EXPERT);
        m.w[l][1] = ggml_new_tensor_3d(m.ctx, types[l][1], N_EMBD, N_FF, N_EXPERT);
        m.w[l][2] = ggml_new_tensor_3d(m.ctx, types[l][2], N_FF, N_EMBD, N_EXPERT);
        m.router[l] = ggml_new_tensor_2d(m.ctx, GGML_TYPE_F32, N_EMBD, N_EXPERT);
    }
    m.buf = ggml_backend_alloc_ctx_tensors(m.ctx, m.dev);
    if (!m.buf) {
        return false;
    }
    ggml_backend_buffer_clear(m.buf, 0);
    ggml_backend_buffer_set_usage(m.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS); // host experts, as the model loads them
    return true;
}

static void free_model(t_model & m) {
    if (m.buf)  { ggml_backend_buffer_free(m.buf); }
    if (m.ctx)  { ggml_free(m.ctx); }
    if (m.dev)  { ggml_backend_free(m.dev); }
    if (m.host) { ggml_backend_free(m.host); }
}

static bool stream_init(const t_model & m, const void * owner) {
    std::vector<llama_moe_gen5_layer_desc> d;
    for (int l = 0; l < N_LAYER; ++l) {
        llama_moe_gen5_layer_desc x;
        x.il   = l;
        x.up   = m.w[l][0];
        x.gate = m.w[l][1];
        x.down = m.w[l][2];
        d.push_back(x);
    }
    llama_moe_gen5_device dv;
    dv.dev     = ggml_backend_get_device(m.dev);
    dv.buft    = ggml_backend_cpu_buffer_type();
    dv.compute = m.dev;
    return llama_prefill_stream_init_layers(d, dv, owner);
}

struct t_graph {
    ggml_context * ctx = nullptr;
    ggml_cgraph *  gf  = nullptr;
    int n_gate    = 0;
    int n_release = 0;
    std::vector<const llama_pfs_view *> views;
    ~t_graph() {
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

// the stream part of build_moe_ffn (src/llama-graph.cpp): lookup, gate pinned to the host, the experts from the view,
// release pinned to the host
static void build(t_graph & g, const t_model & m, ggml_backend_sched_t sched, int64_t T) {
    ggml_init_params ip = { ggml_tensor_overhead()*2048 + ggml_graph_overhead_custom(2048, false), nullptr, true };
    g.ctx = ggml_init(ip);
    g.gf  = ggml_new_graph_custom(g.ctx, 2048, false);
    ggml_tensor * x = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, N_EMBD, T);
    ggml_set_input(x);
    ggml_tensor * cur = x;
    for (int l = 0; l < N_LAYER; ++l) {
        ggml_tensor * w[3] = { m.w[l][0], m.w[l][1], m.w[l][2] };
        const llama_pfs_view * v = llama_prefill_stream_lookup(sched, m.w[l][0], T);
        g.views.push_back(v);
        if (v) {
            ggml_backend_sched_set_tensor_backend(sched, llama_prefill_stream_build_gate(g.ctx, g.gf, v, cur), m.host);
            g.n_gate++;
            w[0] = v->up;
            w[1] = v->gate;
            w[2] = v->down;
        }
        ggml_tensor * logits = ggml_mul_mat(g.ctx, m.router[l], cur); // [N_EXPERT, T]
        ggml_tensor * probs  = ggml_soft_max(g.ctx, logits);
        ggml_tensor * ids    = ggml_argsort_top_k(g.ctx, probs, N_USED); // [N_USED, T]
        ggml_tensor * wts    = ggml_get_rows(g.ctx, ggml_reshape_3d(g.ctx, probs, 1, N_EXPERT, T), ids); // [1, N_USED, T]
        ggml_tensor * xin  = ggml_reshape_3d(g.ctx, cur, N_EMBD, 1, T);
        ggml_tensor * up   = ggml_mul_mat_id(g.ctx, w[0], xin, ids);
        ggml_tensor * gt   = ggml_mul_mat_id(g.ctx, w[1], xin, ids);
        ggml_tensor * act  = ggml_swiglu_split(g.ctx, gt, up);
        ggml_tensor * down = ggml_mul_mat_id(g.ctx, w[2], act, ids);
        if (v) {
            ggml_backend_sched_set_tensor_backend(sched, llama_prefill_stream_build_release(g.ctx, g.gf, v, down), m.host);
            g.n_release++;
        }
        ggml_tensor * ex  = ggml_mul(g.ctx, down, wts);
        ggml_tensor * sum = nullptr;
        for (int s = 0; s < N_USED; ++s) {
            ggml_tensor * vs = ggml_view_2d(g.ctx, ex, N_EMBD, T, ex->nb[2], s*ex->nb[1]);
            sum = sum ? ggml_add(g.ctx, sum, vs) : vs;
        }
        cur = ggml_add(g.ctx, cur, sum);
    }
    ggml_set_output(cur);
    ggml_build_forward_expand(g.gf, cur);
}

struct t_node {
    ggml_op   op;
    ggml_type type;
    int64_t   ne[4];
    int       backend;
};

struct t_shape {
    int    n_nodes  = 0;
    int    n_splits = 0;
    int    n_gate   = 0;
    int    n_rel    = 0;
    size_t buf      = 0; // the device backend's compute buffer
    size_t hbuf     = 0; // the host backend's
    std::vector<t_node> nodes;
};

// the host backend's compute buffers come from a copy of the CPU buffer type: with the same type for both backends the
// scheduler moves every node of the host backend up to the device one (its pass 3), and nothing would split
static ggml_backend_buffer_type g_host_buft;

static ggml_backend_sched_t new_sched(const t_model & m) {
    g_host_buft = *ggml_backend_cpu_buffer_type();
    ggml_backend_t             be[2] = { m.dev, m.host };
    ggml_backend_buffer_type_t bt[2] = { ggml_backend_cpu_buffer_type(), &g_host_buft };
    return ggml_backend_sched_new(be, bt, 2, 4096, false, true);
}

// the graph's structure (split only, per node backend) and its compute buffer (a reserve on a fresh scheduler)
// rsv: build it between llama_prefill_stream_reserve_begin and _end
static t_shape measure(const t_model & m, const void * owner, int64_t T, bool rsv) {
    t_shape r;
    for (int pass = 0; pass < 2; ++pass) {
        ggml_backend_sched_t sched = new_sched(m);
        bool began = false;
        if (rsv) {
            began = llama_prefill_stream_reserve_begin(owner, T);
            TCHECK(began, "reserve begin (T %lld)", (long long) T);
        }
        t_graph g;
        build(g, m, sched, T);
        if (began) {
            llama_prefill_stream_reserve_end(owner);
        }
        if (pass == 0) {
            ggml_backend_sched_split_graph(sched, g.gf);
            r.n_nodes  = ggml_graph_n_nodes(g.gf);
            r.n_splits = ggml_backend_sched_get_n_splits(sched);
            r.n_gate   = g.n_gate;
            r.n_rel    = g.n_release;
            for (int i = 0; i < r.n_nodes; ++i) {
                ggml_tensor * n = ggml_graph_node(g.gf, i);
                ggml_backend_t b = ggml_backend_sched_get_tensor_backend(sched, n);
                t_node tn = { n->op, n->type, { n->ne[0], n->ne[1], n->ne[2], n->ne[3] }, b == m.dev ? 0 : (b == m.host ? 1 : -1) };
                r.nodes.push_back(tn);
            }
            if (rsv) {
                // the reserve views: the aliases' types and shapes, a buffer of the bank type with no memory, data set
                bool ok = true;
                for (int l = 0; l < N_LAYER; ++l) {
                    const llama_pfs_view * v = g.views[l];
                    if (!v) {
                        ok = false;
                        break;
                    }
                    const ggml_tensor * vt[3] = { v->up, v->gate, v->down };
                    for (int k = 0; k < 3; ++k) {
                        const ggml_tensor * t = vt[k];
                        const ggml_tensor * s = m.w[l][k];
                        ok = ok && t != s && t->type == s->type && ggml_are_same_shape(t, s) && t->buffer != nullptr &&
                             t->data != nullptr && ggml_backend_buffer_get_size(t->buffer) == 0 &&
                             ggml_backend_buffer_get_type(t->buffer) == ggml_backend_cpu_buffer_type();
                    }
                }
                TCHECK(ok, "reserve views: same types and shapes, a buffer without memory");
            }
        } else {
            TCHECK(ggml_backend_sched_reserve(sched, g.gf), "reserve");
            r.buf  = ggml_backend_sched_get_buffer_size(sched, m.dev);
            r.hbuf = ggml_backend_sched_get_buffer_size(sched, m.host);
        }
        ggml_backend_sched_free(sched);
    }
    return r;
}

static bool same_nodes(const t_shape & a, const t_shape & b, std::string & why) {
    if (a.nodes.size() != b.nodes.size()) {
        why = "node count " + std::to_string(a.nodes.size()) + " vs " + std::to_string(b.nodes.size());
        return false;
    }
    for (size_t i = 0; i < a.nodes.size(); ++i) {
        const t_node & x = a.nodes[i];
        const t_node & y = b.nodes[i];
        if (x.op != y.op || x.type != y.type || memcmp(x.ne, y.ne, sizeof(x.ne)) != 0 || x.backend != y.backend) {
            why = "node " + std::to_string(i) + ": " + ggml_op_name(x.op) + " on " + std::to_string(x.backend) + " vs " +
                  ggml_op_name(y.op) + " on " + std::to_string(y.backend);
            return false;
        }
    }
    return true;
}

static void print_shape(const char * name, const t_shape & s) {
    printf("  %-34s nodes %4d, splits %2d, gates %d, releases %d, compute buffers: device %8.1f KiB, host %6.1f KiB\n", name,
           s.n_nodes, s.n_splits, s.n_gate, s.n_rel, s.buf/1024.0, s.hbuf/1024.0);
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const char * k_env[] = { "LLAMA_PREFILL_STREAM_MIN", "LLAMA_PREFILL_STREAM_BUFS", "LLAMA_PREFILL_STREAM_LEND",
        "LLAMA_PREFILL_STREAM_LEND_RESERVE", "LLAMA_PREFILL_STREAM_CHUNK_MIB", "LLAMA_PREFILL_STREAM_RING_MIB",
        "LLAMA_PREFILL_STREAM_THREADS", "LLAMA_PREFILL_STREAM_D2D", "LLAMA_PREFILL_STREAM_AHEAD", "LLAMA_PREFILL_STREAM_CACHED" };
    for (const char * n : k_env) {
        set_env(n, nullptr);
    }
    set_env("LLAMA_PREFILL_STREAM_MIN", std::to_string(MIN_T).c_str());
    set_env("LLAMA_PREFILL_STREAM_CHUNK_MIB", "1");
    set_env("LLAMA_PREFILL_STREAM_RING_MIB", "2");
    set_env("LLAMA_PREFILL_STREAM_THREADS", "2");

    t_model m;
    if (!build_model(m)) {
        fprintf(stderr, "cannot build the test model\n");
        return 1;
    }
    static int owner_tag = 0;
    static int other_tag = 0;
    const void * owner = &owner_tag;
    const void * other = &other_tag;
    const int64_t Ts[] = { MIN_T, 64, 192 };

    // the reference: a stream with its own banks (LLAMA_PREFILL_STREAM_LEND=0) builds the streamed form at every width
    printf("test-fn-pfs-reserve: own banks (LEND=0)\n");
    set_env("LLAMA_PREFILL_STREAM_LEND", "0");
    std::vector<t_shape> own;
    if (!stream_init(m, owner)) {
        TCHECK(false, "stream init (LEND=0)");
    } else {
        TCHECK(!llama_prefill_stream_reserve_begin(owner, 64), "no reserve views without a lending stream");
        for (int64_t T : Ts) {
            own.push_back(measure(m, owner, T, false));
            print_shape(("T " + std::to_string(T) + ", streamed").c_str(), own.back());
            TCHECK(own.back().n_gate == N_LAYER && own.back().n_rel == N_LAYER, "own banks: a gate and a release per layer");
        }
        llama_prefill_stream_free(owner);
    }

    printf("test-fn-pfs-reserve: lending stream (LEND=1)\n");
    set_env("LLAMA_PREFILL_STREAM_LEND", "1");
    if (!stream_init(m, owner)) {
        TCHECK(false, "stream init (LEND=1)");
    } else {
        TCHECK(llama_prefill_stream_lookup(nullptr, m.w[0][0], 64) == nullptr, "not borrowed: no view outside a reserve");
        TCHECK(!llama_prefill_stream_reserve_begin(other, 64), "another owner: refused");
        TCHECK(!llama_prefill_stream_reserve_begin(owner, MIN_T - 1), "below the minimum: refused");
        TCHECK(llama_prefill_stream_lookup(nullptr, m.w[0][0], 64) == nullptr, "a refused begin changes nothing");
        TCHECK(llama_prefill_stream_reserve_begin(owner, MIN_T), "at the minimum: begins");
        const llama_pfs_view * v = llama_prefill_stream_lookup(nullptr, m.w[0][0], MIN_T);
        TCHECK(v != nullptr && v->up && v->handle, "a reserve view between begin and end");
        TCHECK(llama_prefill_stream_lookup(nullptr, m.w[0][0], MIN_T - 1) == nullptr, "below the minimum: no view in a reserve");
        llama_prefill_stream_reserve_end(other);
        TCHECK(llama_prefill_stream_lookup(nullptr, m.w[0][0], MIN_T) != nullptr, "another owner's end changes nothing");
        llama_prefill_stream_reserve_end(owner);
        TCHECK(llama_prefill_stream_lookup(nullptr, m.w[0][0], MIN_T) == nullptr, "after end: no view");

        for (size_t i = 0; i < sizeof(Ts)/sizeof(Ts[0]); ++i) {
            const int64_t T = Ts[i];
            const t_shape off = measure(m, owner, T, false);
            const t_shape rsv = measure(m, owner, T, true);
            print_shape(("T " + std::to_string(T) + ", reserve without views").c_str(), off);
            print_shape(("T " + std::to_string(T) + ", reserve with views").c_str(), rsv);
            TCHECK(off.n_gate == 0 && off.n_rel == 0, "T %lld: without the views the reserve has no stream nodes (v0.32.0)", (long long) T);
            TCHECK(rsv.n_gate == N_LAYER && rsv.n_rel == N_LAYER, "T %lld: with the views a gate and a release per layer", (long long) T);
            TCHECK(off.n_splits < rsv.n_splits, "T %lld: the streamed form has more splits (%d vs %d)", (long long) T, rsv.n_splits, off.n_splits);
            if (i < own.size()) {
                std::string why;
                TCHECK(same_nodes(rsv, own[i], why), "T %lld: the reserve graph is the streamed graph node for node (%s)", (long long) T, why.c_str());
                TCHECK(rsv.n_splits == own[i].n_splits, "T %lld: same splits (%d vs %d)", (long long) T, rsv.n_splits, own[i].n_splits);
                TCHECK(rsv.buf == own[i].buf && rsv.hbuf == own[i].hbuf, "T %lld: same compute buffers (%zu / %zu vs %zu / %zu): the "
                       "views take no compute memory", (long long) T, rsv.buf, rsv.hbuf, own[i].buf, own[i].hbuf);
            }

            // graph_reserve's two passes on one scheduler: op-offload graph, then streamed: the larger buffer, the
            // streamed splits; a smaller reserve after them does not shrink the buffer
            ggml_backend_sched_t sched = new_sched(m);
            size_t b1 = 0, b2 = 0, b3 = 0, h2 = 0;
            int    s2 = 0;
            {
                t_graph g;
                build(g, m, sched, T);
                TCHECK(ggml_backend_sched_reserve(sched, g.gf), "two passes: op-offload reserve");
                b1 = ggml_backend_sched_get_buffer_size(sched, m.dev);
            }
            {
                TCHECK(llama_prefill_stream_reserve_begin(owner, T), "two passes: begin");
                t_graph g;
                build(g, m, sched, T);
                llama_prefill_stream_reserve_end(owner);
                TCHECK(ggml_backend_sched_reserve(sched, g.gf), "two passes: streamed reserve");
                b2 = ggml_backend_sched_get_buffer_size(sched, m.dev);
                h2 = ggml_backend_sched_get_buffer_size(sched, m.host);
                s2 = ggml_backend_sched_get_n_splits(sched);
            }
            {
                t_graph g;
                build(g, m, sched, MIN_T);
                TCHECK(ggml_backend_sched_reserve(sched, g.gf), "two passes: a narrow reserve after them");
                b3 = ggml_backend_sched_get_buffer_size(sched, m.dev);
            }
            ggml_backend_sched_free(sched);
            printf("  T %3lld two passes: device %.1f KiB then %.1f KiB (alone %.1f / %.1f), then narrow %.1f KiB; host %.1f KiB "
                   "(alone %.1f / %.1f); splits %d\n", (long long) T, b1/1024.0, b2/1024.0, off.buf/1024.0, rsv.buf/1024.0,
                   b3/1024.0, h2/1024.0, off.hbuf/1024.0, rsv.hbuf/1024.0, s2);
            TCHECK(b1 == off.buf, "T %lld: the first pass is the op-offload reserve", (long long) T);
            TCHECK(b2 == std::max(off.buf, rsv.buf), "T %lld: the buffer ends at the larger need (%zu, max %zu)", (long long) T, b2,
                   std::max(off.buf, rsv.buf));
            TCHECK(h2 == std::max(off.hbuf, rsv.hbuf), "T %lld: the host buffer too (%zu, max %zu)", (long long) T, h2,
                   std::max(off.hbuf, rsv.hbuf));
            TCHECK(s2 == rsv.n_splits, "T %lld: the splits read after the passes are the streamed graph's", (long long) T);
            TCHECK(b3 == b2, "T %lld: a smaller reserve keeps the buffer", (long long) T);
        }
        llama_prefill_stream_free(owner);
    }

    printf("test-fn-pfs-reserve: LLAMA_PREFILL_STREAM_LEND_RESERVE=0\n");
    set_env("LLAMA_PREFILL_STREAM_LEND_RESERVE", "0");
    if (!stream_init(m, owner)) {
        TCHECK(false, "stream init (LEND_RESERVE=0)");
    } else {
        TCHECK(!llama_prefill_stream_reserve_begin(owner, 64), "switch off: begin refused");
        TCHECK(llama_prefill_stream_lookup(nullptr, m.w[0][0], 64) == nullptr, "switch off: no view");
        llama_prefill_stream_free(owner);
    }

    for (const char * n : k_env) {
        set_env(n, nullptr);
    }
    free_model(m);
    printf("test-fn-pfs-reserve: %d checks, %d errors%s\n", g_checks, g_fail, g_fail ? "" : " - PASSED");
    return g_fail == 0 ? 0 : 1;
}
