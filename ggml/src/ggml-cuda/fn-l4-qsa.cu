// [TAG_FN_L4_QSA] Flash-Next (qwen4exp) lever round 4, the QSA layers: see fn-l4-qsa.cuh and ggml-fn-l4-qsa.h.

#include "fn-l4-qsa.cuh"
#include "fn-l3.cuh"
#include "turbot-set-rows.cuh"
#include "ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <unordered_map>
#include <unordered_set>
#include <vector>

bool ggml_cuda_fn_l4_qsa_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_FN_L4_QSA");
        return !(e && e[0] == '0');
    }();
    return on;
}

enum {
    FN_L4_NOTE_IDXDEP = 0,
    FN_L4_NOTE_SEL,
    FN_L4_NOTE_KVW,
    FN_L4_NOTE_POOL,
    FN_L4_NOTE_STREAMS,
    FN_L4_NOTE_STREAMS_SKIP,
    FN_L4_NOTE_COUNT,
};

static void fn_l4_note(int what, const char * msg) {
    static std::atomic<bool> seen[FN_L4_NOTE_COUNT];
    if (what < 0 || what >= FN_L4_NOTE_COUNT || seen[what].exchange(true)) {
        return;
    }
    GGML_LOG_INFO("ggml_cuda: [TAG_FN_L4_QSA] %s\n", msg);
}

static bool fn_l4_noop(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_NONE || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE;
}

static ggml_tensor * fn_l4_base(ggml_tensor * t) {
    return t->view_src ? t->view_src : t;
}

// an op that writes into another tensor's memory (SET_ROWS, TURBOT_SET_ROWS, CPY into a view, ...)
static bool fn_l4_writer(const ggml_tensor * t) {
    return t->view_src != nullptr && !fn_l4_noop(t);
}

// ---------------------------------------------------------------------------------------------------------------------
// [TAG_FN_L4_QSA_IDXDEP] the fix of FAULT 2026-10-05 22:06:26 (LLAMA_FN_GPU_IDXQ8, round 3). The fused indexer reads the
// row indices of the q8_0 gather it replaces (gather->src[1], an I32 graph input), but the indexer node does not list
// them as a source: the allocator frees them after the gather, its last listed reader, and a node between the gather and
// the indexer (the indexer query projection, its norm, rope, the head weights) may take their memory. memcheck saw the
// fused kernel read q8_0 rows 119 GB past the cache with such indices. graph_optimize now keeps the indices allocated
// until the indexer ran and marks the indexer (IDXDEP); fn-l3.cu takes the fused path only with that mark.
// ---------------------------------------------------------------------------------------------------------------------

static ggml_tensor * fn_l4_idxq8_gather(ggml_tensor * ix) {
    if (ix->op != GGML_OP_LIGHTNING_INDEXER || ggml_fn_l3_get(ix) != GGML_FN_L3_IDXQ8) {
        return nullptr;
    }
    ggml_tensor * g = ix->src[1];
    for (int i = 0; i < 4 && g != nullptr && (g->op == GGML_OP_RESHAPE || (g->op == GGML_OP_VIEW && g->view_offs == 0)); ++i) {
        g = g->src[0];
    }
    if (g == nullptr || g->op != GGML_OP_GET_ROWS || ggml_fn_l3_get(g) != GGML_FN_L3_IDXQ8 || g->src[0] == nullptr ||
            g->src[1] == nullptr) {
        return nullptr;
    }
    return g;
}

// ---------------------------------------------------------------------------------------------------------------------
// [TAG_FN_L4_QSA_SEL] the nodes that read the QSA top-k, in the order the qwen4exp builder puts them with SEL on:
//   t  = TOP_K(score)                                   I32 [k, nt]
//   ts = GET_ROWS(reshape(score), t)                    F32 [1, k, nt]   the picked scores
//   gs = GET_ROWS(pool_idxs, reshape(t))                I32 [kpool, k*nt] the member cells of the picks
//   cc = CONCAT(reshape(gs), tail_idxs, 0)              I32 [kpool*k + kpool - 1, nt]
// with only no-op views between them.
// ---------------------------------------------------------------------------------------------------------------------

struct fn_l4_sel_nodes {
    ggml_tensor * t  = nullptr;
    ggml_tensor * ts = nullptr;
    ggml_tensor * gs = nullptr;
    ggml_tensor * cc = nullptr;
    int           i_cc = -1;
};

static bool fn_l4_sel_match(const ggml_cgraph * cg, const int i, fn_l4_sel_nodes & m) {
    ggml_tensor * t = cg->nodes[i];
    if (t->op != GGML_OP_TOP_K || t->type != GGML_TYPE_I32 || t->src[0] == nullptr) {
        return false;
    }
    ggml_tensor * score = t->src[0];
    if (score->op != GGML_OP_LIGHTNING_INDEXER || !(ggml_fn_l4_qsa_get(score) & GGML_FN_L4_QSA_SEL)) {
        return false;
    }
    int k = i + 1;
    auto next = [&]() -> ggml_tensor * {
        while (k < cg->n_nodes && fn_l4_noop(cg->nodes[k])) {
            ++k;
        }
        return k < cg->n_nodes ? cg->nodes[k++] : nullptr;
    };
    ggml_tensor * ts = next();
    if (ts == nullptr || ts->op != GGML_OP_GET_ROWS || ts->type != GGML_TYPE_F32 || ts->src[1] != t || ts->src[0] == nullptr ||
            ts->src[0]->op != GGML_OP_RESHAPE || ts->src[0]->src[0] != score) {
        return false;
    }
    ggml_tensor * gs = next();
    if (gs == nullptr || gs->op != GGML_OP_GET_ROWS || gs->type != GGML_TYPE_I32 || gs->src[0] == nullptr ||
            gs->src[0]->type != GGML_TYPE_I32 || gs->src[1] == nullptr || gs->src[1]->op != GGML_OP_RESHAPE ||
            gs->src[1]->src[0] != t) {
        return false;
    }
    ggml_tensor * cc = next();
    if (cc == nullptr || cc->op != GGML_OP_CONCAT || cc->type != GGML_TYPE_I32 || ggml_get_op_params_i32(cc, 0) != 0 ||
            cc->src[0] == nullptr || cc->src[0]->op != GGML_OP_RESHAPE || cc->src[0]->src[0] != gs || cc->src[1] == nullptr ||
            cc->src[1]->type != GGML_TYPE_I32) {
        return false;
    }
    m.t    = t;
    m.ts   = ts;
    m.gs   = gs;
    m.cc   = cc;
    m.i_cc = k - 1;
    return true;
}

int ggml_cuda_fn_l4_qsa_try_sel(ggml_backend_cuda_context & ctx, ggml_cgraph * cg, int i) {
    if (!ggml_cuda_fn_l4_qsa_enabled()) {
        return 0;
    }
    fn_l4_sel_nodes m;
    if (!fn_l4_sel_match(cg, i, m) || !(ggml_fn_l4_qsa_get(m.t->src[0]) & GGML_FN_L4_QSA_SELDEP) ||
            !ggml_cuda_top_k_takes_fn_l3(m.t)) {
        return 0;
    }
    const ggml_tensor * score = m.t->src[0];
    const ggml_tensor * pidx  = m.gs->src[0];
    const ggml_tensor * tail  = m.cc->src[1];
    const int64_t n_pool = score->ne[0];
    const int64_t n_rows = ggml_nrows(score);
    const int64_t k      = m.t->ne[0];
    const int64_t kpool  = pidx->ne[0];

    // the shapes the unfused nodes have in the qwen4exp graph, every tensor with contiguous rows
    if (score->ne[1] != n_rows || m.t->ne[1] != n_rows || ggml_nrows(m.t) != n_rows || !ggml_is_contiguous(m.t) ||
            m.ts->ne[0] != 1 || m.ts->ne[1] != k || m.ts->ne[2] != n_rows || !ggml_is_contiguous(m.ts) ||
            pidx->ne[1] != n_pool || pidx->ne[2] != 1 || pidx->ne[3] != 1 || pidx->nb[0] != sizeof(int32_t) ||
            pidx->nb[1] % sizeof(int32_t) != 0 ||
            m.gs->ne[0] != kpool || m.gs->ne[1] != k*n_rows || !ggml_is_contiguous(m.gs) ||
            tail->ne[1] != n_rows || tail->ne[2] != 1 || tail->ne[3] != 1 || tail->nb[0] != sizeof(int32_t) ||
            tail->nb[1] % sizeof(int32_t) != 0 ||
            m.cc->ne[0] != kpool*k + tail->ne[0] || m.cc->ne[1] != n_rows || m.cc->ne[2] != 1 || m.cc->ne[3] != 1 ||
            m.cc->nb[0] != sizeof(int32_t) || m.cc->nb[1] % sizeof(int32_t) != 0 || kpool > 64) {
        return 0;
    }

    ggml_cuda_fn_l4_sel_args a;
    a.score     = (const float *) score->data;
    a.n_pool    = n_pool;
    a.n_rows    = n_rows;
    a.k         = k;
    a.pool_idxs = (const int32_t *) pidx->data;
    a.pool_ld   = (int64_t) (pidx->nb[1] / sizeof(int32_t));
    a.kpool     = (int32_t) kpool;
    a.tail      = (const int32_t *) tail->data;
    a.tail_ld   = (int64_t) (tail->nb[1] / sizeof(int32_t));
    a.n_tail    = (int32_t) tail->ne[0];
    a.topk      = (int32_t *) m.t->data;
    a.ts        = (float *) m.ts->data;
    a.gs        = (int32_t *) m.gs->data;
    a.cc        = (int32_t *) m.cc->data;
    a.cc_ld     = (int64_t) (m.cc->nb[1] / sizeof(int32_t));
    if (!ggml_cuda_fn_l4_qsa_sel_launch(ctx.pool(), ctx.stream(), a)) {
        return 0;
    }
    fn_l4_note(FN_L4_NOTE_SEL, "SEL: QSA top-k merge writes the picked scores, the selected cells and the tail");
    return m.i_cc - i;
}

// ---------------------------------------------------------------------------------------------------------------------
// [TAG_FN_L4_QSA_KVW] the K TURBOT_SET_ROWS the builder marked and the V one right after it (no-op views between)
// ---------------------------------------------------------------------------------------------------------------------

int ggml_cuda_fn_l4_qsa_try_kvw(ggml_backend_cuda_context & ctx, ggml_cgraph * cg, int i) {
    ggml_tensor * dk = cg->nodes[i];
    if (!ggml_cuda_fn_l4_qsa_enabled() || dk->op != GGML_OP_TURBOT_SET_ROWS || !(ggml_fn_l4_qsa_get(dk) & GGML_FN_L4_QSA_KVW)) {
        return 0;
    }
    int j = i + 1;
    while (j < cg->n_nodes && fn_l4_noop(cg->nodes[j])) {
        ++j;
    }
    if (j >= cg->n_nodes) {
        return 0;
    }
    ggml_tensor * dv = cg->nodes[j];
    if (dv->op != GGML_OP_TURBOT_SET_ROWS || !(ggml_fn_l4_qsa_get(dv) & GGML_FN_L4_QSA_KVW) ||
            (dv->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        return 0;
    }
    // V must not read what K writes: its rows, indices and young rows are other tensors than K's caches
    for (int s = 0; s < GGML_MAX_SRC; ++s) {
        if (dv->src[s] != nullptr && s != 3 && fn_l4_base(dv->src[s]) == fn_l4_base(dk)) {
            return 0;
        }
    }
    if (!ggml_cuda_turbot_set_rows_kv(ctx, dk, dv)) {
        return 0;
    }
    fn_l4_note(FN_L4_NOTE_KVW, "KVW: the turbot K and V rows of a QSA layer in one launch");
    return j - i;
}

// ---------------------------------------------------------------------------------------------------------------------
// [TAG_FN_L4_QSA_POOL] the k-pool update chain of a QSA layer (build_qsa_sel, cache_safe), from the FILL the builder
// marked, nodes in this order with only no-op views between them:
//   fill = FILL(0)                                   [d, nt]
//   cat  = CONCAT(k_raw, fill, 0)                    [2d, nt]
//   w1   = SET_ROWS(cache, view(cat), k_idxs)        raw key rows of the ubatch (the pooled half zeroed)
//   rows = GET_ROWS(key gate view of cache, idxs)    [d, kpool*n_new] the members of the pools to re-pool
//   s    = CONT(slice 0), ADD(s, slice m) for m = 1 .. kpool-1
//   sc   = SCALE(s), rn = RMS_NORM(sc), mu = MUL(rn, w), ro = ROPE(view(mu), pos)
//   w2   = SET_ROWS(pooled view of cache, view(ro), rep)
// Only the cache rows are written by the fused kernel: every node between fill and w2 must have no reader outside
// the chain (use counts, no output flag).
// ---------------------------------------------------------------------------------------------------------------------

int ggml_cuda_fn_l4_qsa_try_pool(ggml_backend_cuda_context & ctx, ggml_cgraph * cg, int i) {
    ggml_tensor * fill = cg->nodes[i];
    if (!ggml_cuda_fn_l4_qsa_enabled() || fill->op != GGML_OP_FILL || !(ggml_fn_l4_qsa_get(fill) & GGML_FN_L4_QSA_POOL) ||
            fill->type != GGML_TYPE_F32 || ggml_get_op_params_f32(fill, 0) != 0.0f || fill->view_src != nullptr) {
        return 0;
    }
    int k = i + 1;
    int idx = -1;
    auto next = [&]() -> ggml_tensor * {
        while (k < cg->n_nodes && fn_l4_noop(cg->nodes[k])) {
            ++k;
        }
        idx = k;
        return k < cg->n_nodes ? cg->nodes[k++] : nullptr;
    };
    std::vector<int> mids; // non-view intermediates: one reader each, no output flag
    mids.push_back(i);

    const int64_t d  = fill->ne[0];
    const int64_t nt = fill->ne[1];

    ggml_tensor * cat = next();
    if (cat == nullptr || cat->op != GGML_OP_CONCAT || ggml_get_op_params_i32(cat, 0) != 0 || cat->src[1] != fill ||
            cat->src[0] == nullptr || cat->src[0]->type != GGML_TYPE_F32 || cat->src[0]->ne[0] != d || cat->src[0]->ne[1] != nt ||
            cat->src[0]->ne[2] != 1 || cat->src[0]->ne[3] != 1 || cat->src[0]->nb[0] != sizeof(float) || fill->ne[2] != 1 ||
            fill->ne[3] != 1) {
        return 0;
    }
    mids.push_back(idx);

    ggml_tensor * w1 = next();
    if (w1 == nullptr || w1->op != GGML_OP_SET_ROWS || w1->type != GGML_TYPE_Q8_0 || w1->view_src == nullptr ||
            w1->src[0] == nullptr || fn_l4_base(w1->src[0]) != cat || w1->src[0]->ne[0] != 2*d || w1->src[0]->ne[1] != nt ||
            w1->src[1] == nullptr || (w1->src[1]->type != GGML_TYPE_I64 && w1->src[1]->type != GGML_TYPE_I32) ||
            w1->src[1]->ne[0] != nt || w1->ne[0] != 2*d || w1->view_offs != 0) {
        return 0;
    }
    ggml_tensor * cache = w1->view_src;
    const size_t row_bytes = w1->nb[1];

    ggml_tensor * rows = next();
    if (rows == nullptr || rows->op != GGML_OP_GET_ROWS || rows->type != GGML_TYPE_F32 || rows->src[0] == nullptr ||
            rows->src[0]->type != GGML_TYPE_Q8_0 || fn_l4_base(rows->src[0]) != cache || rows->src[0]->view_offs != 0 ||
            rows->src[0]->ne[0] != d || rows->src[0]->nb[1] != row_bytes || rows->src[1] == nullptr ||
            rows->src[1]->type != GGML_TYPE_I32 || rows->ne[0] != d || !ggml_is_contiguous(rows)) {
        return 0;
    }
    ggml_tensor * nidx = fn_l4_base(rows->src[1]);
    if (nidx->type != GGML_TYPE_I32 || nidx->ne[2] != 1 || nidx->ne[3] != 1 || !ggml_is_contiguous(nidx) ||
            rows->ne[1] != nidx->ne[0]*nidx->ne[1] || rows->src[1]->view_offs != 0) {
        return 0;
    }
    const int64_t kpool = nidx->ne[0];
    const int64_t n_new = nidx->ne[1];
    mids.push_back(idx);

    // the member sum: CONT of slice 0, then one ADD per further member, each slice a view of rows at m*d floats
    auto is_slice = [&](const ggml_tensor * v, int64_t m) {
        return v != nullptr && v->view_src == rows && v->view_offs == (size_t) (m*d)*sizeof(float) && v->ne[0] == d &&
               v->ne[1] == n_new && v->nb[1] == (size_t) (kpool*d)*sizeof(float);
    };
    ggml_tensor * s = next();
    if (s == nullptr || s->op != GGML_OP_CONT || !is_slice(s->src[0], 0) || s->type != GGML_TYPE_F32) {
        return 0;
    }
    mids.push_back(idx);
    for (int64_t m = 1; m < kpool; ++m) {
        ggml_tensor * add = next();
        if (add == nullptr || add->op != GGML_OP_ADD || add->src[0] != s || !is_slice(add->src[1], m) ||
                add->type != GGML_TYPE_F32 || add->ne[0] != d || add->ne[1] != n_new) {
            return 0;
        }
        mids.push_back(idx);
        s = add;
    }

    ggml_tensor * sc = next();
    if (sc == nullptr || sc->op != GGML_OP_SCALE || sc->src[0] != s) {
        return 0;
    }
    mids.push_back(idx);
    ggml_tensor * rn = next();
    if (rn == nullptr || rn->op != GGML_OP_RMS_NORM || rn->src[0] != sc) {
        return 0;
    }
    mids.push_back(idx);
    ggml_tensor * mu = next();
    if (mu == nullptr || mu->op != GGML_OP_MUL || mu->src[0] != rn || mu->src[1] == nullptr || mu->src[1]->type != GGML_TYPE_F32 ||
            mu->src[1]->ne[0] != d || ggml_nelements(mu->src[1]) != d || !ggml_is_contiguous(mu->src[1])) {
        return 0;
    }
    mids.push_back(idx);
    ggml_tensor * ro = next();
    if (ro == nullptr || ro->op != GGML_OP_ROPE || ro->src[0] == nullptr || fn_l4_base(ro->src[0]) != mu ||
            ro->src[0]->ne[0] != d || ro->src[0]->ne[1] != 1 || ro->src[0]->ne[2] != n_new || ro->src[1] == nullptr ||
            ro->src[1]->type != GGML_TYPE_I32 || ro->src[1]->ne[0] != 4*n_new || !ggml_is_contiguous(ro->src[1]) ||
            !ggml_is_contiguous(ro) || ro->type != GGML_TYPE_F32) {
        return 0;
    }
    mids.push_back(idx);
    ggml_tensor * w2 = next();
    if (w2 == nullptr || w2->op != GGML_OP_SET_ROWS || w2->type != GGML_TYPE_Q8_0 || w2->view_src != cache ||
            w2->view_offs != ggml_row_size(GGML_TYPE_Q8_0, d) || w2->nb[1] != row_bytes || w2->ne[0] != d ||
            w2->src[0] == nullptr || fn_l4_base(w2->src[0]) != ro || w2->src[1] == nullptr ||
            (w2->src[1]->type != GGML_TYPE_I64 && w2->src[1]->type != GGML_TYPE_I32) || w2->src[1]->ne[0] != n_new) {
        return 0;
    }
    const int i_w2 = idx;

    for (int m : mids) {
        const ggml_tensor * t = cg->nodes[m];
        if ((t->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 || ggml_node_get_use_count(cg, m) != 1) {
            return 0;
        }
    }
    if (d % QK8_0 != 0 || d > 128 || 2*d != (int64_t) (row_bytes / sizeof(block_q8_0))*QK8_0 || nt > 64 || n_new > 64 ||
            kpool > 16 || kpool < 1) {
        return 0;
    }

    ggml_cuda_fn_l4_pool_args a;
    a.k_raw           = (const float *) cat->src[0]->data;
    a.k_raw_ld        = (int64_t) (cat->src[0]->nb[1] / sizeof(float));
    a.k_idxs          = w1->src[1]->data;
    a.k_idxs_i64      = w1->src[1]->type == GGML_TYPE_I64;
    a.k_idxs_ld       = (int64_t) (w1->src[1]->nb[0] / ggml_type_size(w1->src[1]->type));
    a.cache           = (char *) cache->data;
    a.cache_row_bytes = (int64_t) row_bytes;
    a.pooled_off      = (int64_t) ggml_row_size(GGML_TYPE_Q8_0, d);
    a.w1_row_elems    = (int32_t) (2*d);
    a.new_idxs        = (const int32_t *) nidx->data;
    a.rep             = w2->src[1]->data;
    a.rep_i64         = w2->src[1]->type == GGML_TYPE_I64;
    a.rep_ld          = (int64_t) (w2->src[1]->nb[0] / ggml_type_size(w2->src[1]->type));
    a.pos             = (const int32_t *) ro->src[1]->data;
    a.norm_w          = (const float *) mu->src[1]->data;
    a.d               = (int32_t) d;
    a.n_tok           = (int32_t) nt;
    a.n_new           = (int32_t) n_new;
    a.kpool           = (int32_t) kpool;
    a.scale           = ggml_get_op_params_f32(sc, 0);
    a.scale_bias      = ggml_get_op_params_f32(sc, 1);
    a.eps             = ggml_get_op_params_f32(rn, 0);
    if (!ggml_cuda_fn_l4_qsa_pool_launch(ctx, a, ro)) {
        return 0;
    }
    fn_l4_note(FN_L4_NOTE_POOL, "POOL: the k-pool update of a QSA layer (raw key rows, pooled keys) in one launch");
    return i_w2 - i;
}

// ---------------------------------------------------------------------------------------------------------------------
// graph_optimize: the allocation dependencies of IDXQ8 and SEL
// ---------------------------------------------------------------------------------------------------------------------

void ggml_cuda_fn_l4_qsa_deps(ggml_cgraph * cg, ggml_backend_graph_optimize_params * params) {
    if (params == nullptr || params->add_alloc_dep == nullptr) {
        return;
    }
    for (int i = 0; i < cg->n_nodes; ++i) {
        ggml_tensor * node = cg->nodes[i];
        // [TAG_FN_L4_QSA_IDXDEP] (not under GGML_CUDA_FN_L4_QSA: it makes the round-3 lever safe, it does not add one)
        if (node->op == GGML_OP_LIGHTNING_INDEXER && ggml_cuda_fn_l3_enabled()) {
            ggml_tensor * g = fn_l4_idxq8_gather(node);
            if (g != nullptr) {
                params->add_alloc_dep(params->user_data, g->src[1], node);
                ggml_fn_l4_qsa_add(node, GGML_FN_L4_QSA_IDXDEP);
                fn_l4_note(FN_L4_NOTE_IDXDEP, "IDXQ8: the gather's indices stay allocated until the fused indexer ran");
            }
            continue;
        }
        // [TAG_FN_L4_QSA_SEL] everything the fused merge reads or writes stays allocated until the CONCAT
        if (node->op == GGML_OP_TOP_K && ggml_cuda_fn_l4_qsa_enabled()) {
            fn_l4_sel_nodes m;
            if (fn_l4_sel_match(cg, i, m)) {
                ggml_tensor * keep[] = { m.t->src[0], m.t, m.ts, m.gs, m.gs->src[0], m.cc->src[1] };
                for (ggml_tensor * t : keep) {
                    params->add_alloc_dep(params->user_data, t, m.cc);
                }
                ggml_fn_l4_qsa_add(m.t->src[0], GGML_FN_L4_QSA_SELDEP);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// [TAG_FN_L4_QSA_STREAMS] the indexer chain of a QSA layer (k-pool update, indexer query, lightning indexer, top-k, the
// selection and its mask: everything the FLASH_ATTN_EXT reads through its mask) on stream 1, and the q / k / v chain
// (projections, norms, rope, turbot K / V writes, the Q rotation: everything else the FLASH_ATTN_EXT reads) on stream
// 0, both after the last node they both read (the fork) and joined at the FLASH_ATTN_EXT, through the concurrent-event
// machinery of GGML_CUDA_GRAPH_OPT (ggml_cuda_concurrent_event). The node order does not change.
//   - the sides: a reverse pass over the 512 nodes before the attention marks every ancestor of the mask (1) and of
//     its other sources (2), through the sources and through the memory a writer node (SET_ROWS into a cache) fills
//     for a later reader of the same tensor; the fork is the last computed node with both marks. A computed node
//     between fork and join with no mark, a tensor one side writes and the other side reads or writes, or no marked
//     indexer on side 1: the layer stays on one stream.
//   - memory: every tensor a node between fork and join reads or writes stays allocated until the join (scheduler
//     allocation dependencies), so no two of them share memory and neither side can overwrite what the other reads.
//   - the same kernels with the same inputs on another stream: the same bits.
// ---------------------------------------------------------------------------------------------------------------------

bool ggml_cuda_fn_l4_qsa_streams_wanted(const ggml_cgraph * cg) {
    if (!ggml_cuda_fn_l4_qsa_enabled()) {
        return false;
    }
    for (int i = 0; i < cg->n_nodes; ++i) {
        const ggml_tensor * n = cg->nodes[i];
        if (n->op == GGML_OP_LIGHTNING_INDEXER && (ggml_fn_l4_qsa_get(n) & GGML_FN_L4_QSA_STREAMS)) {
            return true;
        }
    }
    return false;
}

static constexpr int FN_L4_STREAMS_WINDOW = 512;

// the graph's node positions and, per tensor written in place, the nodes that write it (built once per graph)
struct fn_l4_graph_index {
    std::unordered_map<const ggml_tensor *, int>              pos;
    std::unordered_map<const ggml_tensor *, std::vector<int>> writers;
};

// side[k - f - 1] for the nodes between fork f and join j: 0 = no-op, not mapped; 1 = stream 1; 2 = stream 2
static bool fn_l4_streams_region(ggml_cgraph * cg, const fn_l4_graph_index & gi, const int j, int & f_out, std::vector<uint8_t> & side) {
    ggml_tensor * J = cg->nodes[j];
    if (J->op != GGML_OP_FLASH_ATTN_EXT || J->src[3] == nullptr || J->src[0] == nullptr) {
        return false;
    }
    const int w0 = std::max(0, j - FN_L4_STREAMS_WINDOW);
    const int n  = j - w0;

    std::vector<uint8_t> anc(n, 0);
    auto writes_before = [&](const ggml_tensor * base, const int below, const uint8_t bit) {
        auto wt = gi.writers.find(base);
        if (wt == gi.writers.end()) {
            return;
        }
        for (int g : wt->second) {
            if (g >= w0 && g - w0 < below) {
                anc[g - w0] |= bit;
            }
        }
    };
    auto mark_src = [&](ggml_tensor * s, const int below, const uint8_t bit) {
        if (s == nullptr) {
            return;
        }
        auto it = gi.pos.find(s);
        if (it != gi.pos.end() && it->second >= w0 && it->second - w0 < below) {
            anc[it->second - w0] |= bit;
        }
        writes_before(fn_l4_base(s), below, bit);
    };

    for (int s = 0; s < GGML_MAX_SRC; ++s) {
        mark_src(J->src[s], n, s == 3 ? 1 : 2);
    }
    for (int k = n - 1; k >= 0; --k) {
        const uint8_t a = anc[k];
        if (a == 0) {
            continue;
        }
        ggml_tensor * x = cg->nodes[w0 + k];
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            mark_src(x->src[s], k, a);
        }
        if (x->view_src != nullptr) {
            writes_before(x->view_src, k, a);
        }
    }

    int f = -1;
    for (int k = n - 1; k >= 0; --k) {
        if (anc[k] == 3 && !fn_l4_noop(cg->nodes[w0 + k])) {
            f = k;
            break;
        }
    }
    if (f < 0 || n - f - 1 < 8) {
        return false;
    }

    side.assign(n - f - 1, 0);
    int n_side[3] = { 0, 0, 0 };
    bool indexer = false;
    for (int k = f + 1; k < n; ++k) {
        const ggml_tensor * x = cg->nodes[w0 + k];
        if (fn_l4_noop(x)) {
            continue;
        }
        if (anc[k] != 1 && anc[k] != 2) {
            return false;
        }
        if ((x->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
            return false;
        }
        side[k - f - 1] = anc[k];
        n_side[anc[k]]++;
        indexer = indexer || (anc[k] == 1 && x->op == GGML_OP_LIGHTNING_INDEXER && (ggml_fn_l4_qsa_get(x) & GGML_FN_L4_QSA_STREAMS));
    }
    if (!indexer || n_side[1] < 2 || n_side[2] < 2) {
        return false;
    }

    // a tensor one side writes into must be neither read nor written by the other side
    std::unordered_map<const ggml_tensor *, uint8_t> wr;
    std::unordered_map<const ggml_tensor *, uint8_t> rd;
    for (int k = f + 1; k < n; ++k) {
        const uint8_t a = side[k - f - 1];
        if (a == 0) {
            continue;
        }
        ggml_tensor * x = cg->nodes[w0 + k];
        if (fn_l4_writer(x)) {
            wr[x->view_src] |= a;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (x->src[s] != nullptr) {
                rd[fn_l4_base(x->src[s])] |= a;
            }
        }
    }
    for (const auto & it : wr) {
        const auto r = rd.find(it.first);
        const uint8_t both = it.second | (r != rd.end() ? r->second : 0);
        if (both == 3) {
            return false;
        }
    }

    f_out = w0 + f;
    return true;
}

void ggml_cuda_fn_l4_qsa_streams(ggml_backend_cuda_context * ctx, ggml_cgraph * cg, ggml_backend_graph_optimize_params * params) {
    if (params == nullptr || params->add_alloc_dep == nullptr || !ggml_cuda_fn_l4_qsa_enabled()) {
        return;
    }

    // the last node index that reads each tensor's memory
    std::unordered_map<const ggml_tensor *, int> last_use;
    last_use.reserve(4*cg->n_nodes);
    for (int i = 0; i < cg->n_nodes; ++i) {
        ggml_tensor * x = cg->nodes[i];
        if (x->view_src != nullptr) {
            last_use[x->view_src] = i;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (x->src[s] != nullptr) {
                last_use[fn_l4_base(x->src[s])] = i;
            }
        }
    }

    fn_l4_graph_index gi;
    gi.pos.reserve(2*cg->n_nodes);
    for (int i = 0; i < cg->n_nodes; ++i) {
        ggml_tensor * t = cg->nodes[i];
        gi.pos[t] = i;
        if (fn_l4_writer(t)) {
            gi.writers[t->view_src].push_back(i);
        }
    }

    auto & events = ctx->stream_context().concurrent_events;
    int n_events      = 0;
    int last_join     = -1;
    int last_indexer  = -1;
    std::vector<uint8_t> side;
    for (int j = 0; j < cg->n_nodes; ++j) {
        ggml_tensor * J = cg->nodes[j];
        if (J->op == GGML_OP_LIGHTNING_INDEXER && (ggml_fn_l4_qsa_get(J) & GGML_FN_L4_QSA_STREAMS)) {
            last_indexer = j;
            continue;
        }
        if (J->op != GGML_OP_FLASH_ATTN_EXT || last_indexer < 0 || j - last_indexer > FN_L4_STREAMS_WINDOW) {
            continue;
        }
        int f = -1;
        if (!fn_l4_streams_region(cg, gi, j, f, side) || f <= last_join) {
            if (f >= 0 && f <= last_join) {
                fn_l4_note(FN_L4_NOTE_STREAMS_SKIP, "STREAMS: a QSA layer overlaps the previous region, it stays on one stream");
            }
            continue;
        }
        ggml_tensor * fork = cg->nodes[f];
        if (events.find(fork) != events.end()) {
            continue;
        }

        // the indexer chain on stream 1; the q / k / v chain stays on stream 0, where its mat-vecs keep the q8_1 reuse
        // cache of the layer input (side streams quantize it again)
        ggml_cuda_concurrent_event ev(1);
        ev.join_node = J;
        for (int k = f + 1; k < j; ++k) {
            ev.original_order.push_back(cg->nodes[k]);
            if (side[k - f - 1] != 0) {
                ev.stream_mapping[cg->nodes[k]] = side[k - f - 1] == 1 ? 1 : 0;
            }
        }

        std::vector<ggml_tensor *>        keep; // in node order, so every build of the graph gets the same dependencies
        std::unordered_set<ggml_tensor *> kept;
        auto keep_base = [&](ggml_tensor * t) {
            if (t == nullptr) {
                return;
            }
            ggml_tensor * b = fn_l4_base(t);
            if (b->data != nullptr || b->buffer != nullptr) {
                return; // weights, caches: not in the compute buffer
            }
            const auto it = last_use.find(b);
            if (it != last_use.end() && it->second >= j) {
                return; // read by the join or later: allocated until then anyway
            }
            if (kept.insert(b).second) {
                keep.push_back(b);
            }
        };
        for (int k = f; k < j; ++k) {
            ggml_tensor * x = cg->nodes[k];
            keep_base(x);
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                keep_base(x->src[s]);
            }
        }
        for (ggml_tensor * t : keep) {
            params->add_alloc_dep(params->user_data, t, J);
        }

        events.emplace(fork, std::move(ev));
        last_join = j;
        n_events++;
    }
    if (n_events > 0) {
        fn_l4_note(FN_L4_NOTE_STREAMS, "STREAMS: the indexer chain and the q/k/v chain of each QSA layer on two streams");
    }
}
