// [TAG_FN_L22_CPUSHARE] round 22, package P16 (M11): the CPU share of streamed prompt ubatches of <= 3072 rows. See
// llama-fn-l22-p16.h. With LLAMA_FN_L22_CPUSHARE unset nothing here runs: the hook is not called (llama-graph.cpp) and the
// stream allocates no zero expert.

#include "llama-fn-l22.h"
#include "llama-fn-l22-p16.h"
#include "llama-adapter.h"
#include "llama-fn-auto.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-moe-gen5.h"
#include "llama-moecache.h"
#include "llama-moe-bridge.h" // [TAG_R38_E2] llama_r38_fp4s

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#if defined(_MSC_VER)
#    include <intrin.h>
#endif

namespace {

using cps_mr_t = void (*)(int n, float * s, size_t bs, const void * vx, size_t bx, int nr, const void * const * vy, int nc);

constexpr int CPS_NC  = 4;  // columns per multi-row call (GGML_CPU_FN_MR_MAX_NC)
constexpr int CPS_RA  = 32; // rows of up / gate per piece
constexpr int CPS_RB  = 64; // rows of down per piece
constexpr int CPS_MAX_USED = 16;

// the CPU kernels of one weight type
struct cps_kern {
    ggml_type         type     = GGML_TYPE_COUNT;
    ggml_vec_dot_t    vec_dot  = nullptr;
    ggml_type         vdt      = GGML_TYPE_COUNT; // the type the columns are quantized to
    ggml_from_float_t from_vdt = nullptr;
    cps_mr_t          mr       = nullptr;         // or null: vec_dot per (row, column)
};

struct cps_slice {
    const uint8_t * p[3] = { nullptr, nullptr, nullptr }; // up, gate, down bytes of one CPU expert
};

struct cps_layer {
    int                    il       = -1;
    int                    pos      = -1;
    const llama_pfs_view * view     = nullptr;
    const ggml_tensor *    host[3]  = { nullptr, nullptr, nullptr };
    int64_t                n_expert = 0;
    cps_kern               kern[3];
    bool                   kern_ok  = false;

    std::vector<float>    share;      // EWMA of rows / token per expert
    bool                  share_init = false;
    std::vector<uint64_t> bytes;      // host bytes per expert (up + gate + down)
    std::vector<uint8_t>  hot;        // in a hot slot at the ubatch begin
    std::vector<uint8_t>  late;       // copied after the plan in this ubatch

    // the plan (plan op -> share op)
    std::vector<int32_t>   cpu;       // CPU experts, ascending
    std::vector<cps_slice> slice;     // per CPU expert
    std::vector<int32_t>   e_pair0;   // [n_cpu + 1] pairs of each CPU expert
    std::vector<int32_t>   pair_j;    // the pair's token index into tok
    std::vector<float>     pair_w;    // the pair's routing weight
    std::vector<int32_t>   tok;       // distinct CPU tokens, ascending
    std::vector<int32_t>   j_pair0;   // [n_tok + 1] pairs of each token, in slot order (indices into j_pairs)
    std::vector<int32_t>   j_pairs;
    std::vector<int32_t>   tickets;   // pin tickets held while the share op reads the slices
    std::atomic<int>       bar[4];

    // the graph being built (build -> finish of the same build_moe_ffn call)
    ggml_tensor * g_plan  = nullptr;
    ggml_tensor * g_share = nullptr;
    int64_t       g_kT    = 0;

    cps_layer() {
        for (auto & b : bar) {
            b.store(0);
        }
    }
};

struct cps_state {
    const void *        owner = nullptr;
    const llama_model * model = nullptr;
    llama_fn_cps_cfg    cfg;

    std::vector<std::unique_ptr<cps_layer>> layers; // by stream position; sized once, at the first build

    const ggml_type_traits_cpu * (*traits)(ggml_type) = nullptr;
    void * (*mr_get)(ggml_type) = nullptr;

    // scratch of the share op (sized by the plan op; grow only)
    std::vector<uint8_t> xq[2]; // quantized token rows for up's / gate's column type
    std::vector<float>   act;   // [pairs][n_ff]
    std::vector<uint8_t> actq;  // [pairs][row size of down's column type]
    std::vector<float>   y;     // [pairs][n_embd]

    llama_fn_cps_rls rls;

    std::mutex        sums_mtx;
    llama_fn_cps_sums sums;
    std::vector<uint8_t> late_all; // [n_pos][n_e] the share_begin masks
};

cps_state * g_cps = nullptr;
std::mutex  g_cps_mtx;

// test pins live outside the state, so a test may set them before the configure
int                        g_pin_active_mock = -1;
llama_fn_cps_pin_query_t   g_pin_query_mock  = nullptr;
llama_fn_cps_pin_release_t g_pin_rel_mock    = nullptr;

bool cps_pin_active() {
    return g_pin_active_mock >= 0 ? g_pin_active_mock != 0 : llama_fn_pin_active();
}

bool cps_pin_query(const void * p, size_t n, const void ** src, size_t * len, int32_t * ticket) {
    if (g_pin_query_mock) {
        return g_pin_query_mock(p, n, src, len, ticket);
    }
    llama_fn_pin_seg seg = {};
    if (!llama_fn_pin_query(p, n, &seg)) {
        return false;
    }
    *src    = seg.src;
    *len    = seg.len;
    *ticket = seg.ticket;
    return true;
}

void cps_pin_release(int32_t t) {
    if (t < 0) {
        return;
    }
    if (g_pin_rel_mock) {
        g_pin_rel_mock(t);
        return;
    }
    llama_fn_pin_release(t);
}

void cps_release_tickets(cps_layer & L) {
    for (int32_t t : L.tickets) {
        cps_pin_release(t);
    }
    L.tickets.clear();
}

inline void cps_pause() {
#if defined(_MSC_VER)
    _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}

// an op-local barrier of the share op's nth threads (reset by the plan op before each run)
void cps_barrier(std::atomic<int> & a, int nth) {
    a.fetch_add(1, std::memory_order_acq_rel);
    int spins = 0;
    while (a.load(std::memory_order_acquire) < nth) {
        if (++spins < 4096) {
            cps_pause();
        } else {
            std::this_thread::yield();
        }
    }
}

// the CPU backend's kernel lookups (procs, so a build with loadable backends works too)
void cps_lookup(cps_state * g) {
    if (ggml_backend_reg_t reg = ggml_backend_reg_by_name("CPU")) {
        g->traits = (const ggml_type_traits_cpu * (*)(ggml_type)) ggml_backend_reg_get_proc_address(reg, "ggml_get_type_traits_cpu");
        g->mr_get = (void * (*)(ggml_type)) ggml_backend_reg_get_proc_address(reg, "ggml_cpu_fn_mr_kernel_get");
    }
}

bool cps_kern_init(const cps_state * g, ggml_type t, cps_kern & k) {
    if (!g->traits) {
        return false;
    }
    const ggml_type_traits_cpu * tt = g->traits(t);
    if (!tt || !tt->vec_dot || tt->vec_dot_type >= GGML_TYPE_COUNT) {
        return false;
    }
    const ggml_type_traits_cpu * tv = g->traits(tt->vec_dot_type);
    if (!tv || !tv->from_float) {
        return false;
    }
    k.type     = t;
    k.vec_dot  = tt->vec_dot;
    k.vdt      = tt->vec_dot_type;
    k.from_vdt = tv->from_float;
    k.mr       = g->mr_get ? (cps_mr_t) g->mr_get(t) : nullptr;
    return true;
}

// s[c*bs + r] = dot(row r of vx, column vy[c]) for r < nr, c < nc
inline void cps_dot(const cps_kern & k, int n, float * s, size_t bs, const uint8_t * vx, size_t bx, int nr,
        const void * const * vy, int nc) {
    if (k.mr) {
        k.mr(n, s, bs, vx, bx, nr, vy, nc);
        return;
    }
    for (int c = 0; c < nc; ++c) {
        for (int r = 0; r < nr; ++r) {
            k.vec_dot(n, s + c*bs + r, 0, vx + (size_t) r*bx, 0, vy[c], 0, 1);
        }
    }
}

inline float cps_silu(float x) {
    return x/(1.0f + expf(-x));
}

void cps_sums_add(cps_state * g, const llama_fn_cps_sums & d) {
    std::lock_guard<std::mutex> lk(g->sums_mtx);
    llama_fn_cps_sums & s = g->sums;
    s.ubatches    += d.ubatches;
    s.layers      += d.layers;
    s.cpu_experts += d.cpu_experts;
    s.cpu_pairs   += d.cpu_pairs;
    s.cpu_tokens  += d.cpu_tokens;
    s.skip_cpu    += d.skip_cpu;
    s.skip_unr    += d.skip_unr;
    s.late_bytes  += d.late_bytes;
    s.dropped     += d.dropped;
    s.cut_cap     += d.cut_cap;
    s.cpu_ms      += d.cpu_ms;
    s.budget_k    += d.budget_k;
    s.pins_off    += d.pins_off;
    if (d.rows > 0) {
        s.rows = d.rows;
    }
}

// the first share layer of a ubatch: late sets of every position, then the stream's new-ubatch work
void cps_ubatch_begin(cps_state * g, int n_used, int n_tokens) {
    const llama_pfs_share_counters sc = llama_prefill_stream_share_counters();
    const double dma  = sc.dma_bpu_ewma > 0 ? std::min(60000.0, std::max(5000.0, sc.dma_bpu_ewma)) : 40000.0;
    const double plan = sc.plan_us_ewma >= 0 ? std::min(200000.0, sc.plan_us_ewma) : 10000.0;
    int64_t n_e = 0;
    for (const auto & up : g->layers) {
        if (up) {
            n_e = std::max(n_e, up->n_expert);
        }
    }
    const size_t n_pos = g->layers.size();
    g->late_all.assign(n_pos*(size_t) n_e, 0);
    llama_fn_cps_sums d;
    d.ubatches = 1;
    d.rows     = n_tokens;
    std::vector<int32_t> slot;
    std::vector<float>   cnt;
    std::vector<uint8_t> vol;
    for (size_t p = 0; p < n_pos; ++p) {
        cps_layer * L = g->layers[p].get();
        if (!L || L->n_expert <= 0) {
            continue;
        }
        const int64_t ne = L->n_expert;
        L->hot.assign((size_t) ne, 0);
        L->late.assign((size_t) ne, 0);
        slot.assign((size_t) ne, -1);
        const ggml_tensor * s3[3] = { nullptr, nullptr, nullptr };
        if (llama_moe_hot_slots(L->host[0], slot.data(), ne, s3)) {
            for (int64_t e = 0; e < ne; ++e) {
                L->hot[e] = slot[e] >= 0;
            }
        }
        if (!L->share_init) {
            // the prior: the hot set's decayed counts when it has them, else every expert alike
            L->share.assign((size_t) ne, (float) n_used/(float) ne);
            cnt.assign((size_t) ne, 0.0f);
            vol.assign((size_t) ne, 0);
            slot.assign((size_t) ne, -1);
            if (llama_moe_hot_tier_view(L->host[0], slot.data(), cnt.data(), vol.data(), ne) == 1) {
                double sum = 0.0;
                for (float c : cnt) {
                    sum += std::max(0.0f, c);
                }
                if (sum > 0.0) {
                    for (int64_t e = 0; e < ne; ++e) {
                        L->share[e] = (float) (n_used*std::max(0.0f, cnt[e])/sum);
                    }
                }
            }
            L->share_init = true;
        }
        if (p == 0) {
            continue; // the stream never copies layer 0 late
        }
        uint64_t host = 0;
        for (int64_t e = 0; e < ne; ++e) {
            host += L->hot[e] ? 0 : L->bytes[e];
        }
        // the copy engine has this much early work while the plan forms
        const double early_min = g->cfg.early_kib >= 0 ? 1024.0*g->cfg.early_kib : plan*dma;
        llama_fn_cps_late_in in;
        in.n_expert  = (int) ne;
        in.n_tokens  = n_tokens;
        in.share     = L->share.data();
        in.hot       = L->hot.data();
        in.bytes     = L->bytes.data();
        in.late_rows = g->cfg.late_rows;
        in.byte_cap  = (double) host > early_min ? host - (uint64_t) early_min : 0;
        d.late_bytes += llama_fn_cps_late(in, L->late.data());
        memcpy(g->late_all.data() + p*(size_t) n_e, L->late.data(), (size_t) ne);
    }
    cps_sums_add(g, d);
    llama_prefill_stream_share_begin(g->late_all.data(), (int) n_e);
}

// the pairs of a plan's CPU experts (by expert, tokens ascending) and of each CPU token (slots ascending), the slices,
// and the share op's scratch
void cps_plan_pairs(cps_state * g, cps_layer & Lr, const int32_t * ids, const float * w, int k, int T,
        const std::vector<int32_t> & cpu, const std::vector<int32_t> & tok, const std::vector<cps_slice> & slices) {
    cps_layer * L = &Lr;
    const int64_t ne    = L->n_expert;
    const int64_t kT    = (int64_t) k*T;
    const int     n_tok = (int) tok.size();
    std::vector<int32_t> jt((size_t) T, -1);
    for (int j = 0; j < n_tok; ++j) {
        jt[tok[j]] = j;
    }
    std::vector<int32_t> cpos((size_t) ne, -1);
    L->cpu = cpu;
    L->slice.resize(L->cpu.size());
    for (size_t c = 0; c < L->cpu.size(); ++c) {
        cpos[L->cpu[c]] = (int32_t) c;
        L->slice[c]     = slices[L->cpu[c]];
    }
    const int n_cpu = (int) L->cpu.size();
    L->e_pair0.assign((size_t) n_cpu + 1, 0);
    for (int64_t i = 0; i < kT; ++i) {
        const int32_t e = ids[i];
        if (e >= 0 && e < ne && cpos[e] >= 0) {
            L->e_pair0[cpos[e] + 1]++;
        }
    }
    for (int c = 0; c < n_cpu; ++c) {
        L->e_pair0[c + 1] += L->e_pair0[c];
    }
    const int n_pairs = L->e_pair0[n_cpu];
    L->pair_j.assign((size_t) n_pairs, 0);
    L->pair_w.assign((size_t) n_pairs, 0.0f);
    std::vector<int32_t> fill(L->e_pair0.begin(), L->e_pair0.end() - 1);
    L->j_pair0.assign((size_t) n_tok + 1, 0);
    std::vector<int32_t> pair_s((size_t) n_pairs, 0);
    for (int t = 0; t < T; ++t) {
        for (int s = 0; s < k; ++s) {
            const int32_t e = ids[(size_t) t*k + s];
            if (e < 0 || e >= ne || cpos[e] < 0) {
                continue;
            }
            const int q = fill[cpos[e]]++;
            L->pair_j[q] = jt[t];
            L->pair_w[q] = w[(size_t) t*k + s];
            pair_s[q]    = s;
            L->j_pair0[jt[t] + 1]++;
        }
    }
    for (int j = 0; j < n_tok; ++j) {
        L->j_pair0[j + 1] += L->j_pair0[j];
    }
    L->j_pairs.assign((size_t) n_pairs, 0);
    std::vector<int32_t> jf(L->j_pair0.begin(), L->j_pair0.end() - 1);
    for (int q = 0; q < n_pairs; ++q) {
        L->j_pairs[jf[L->pair_j[q]]++] = q;
    }
    for (int j = 0; j < n_tok; ++j) { // slot order within a token (k <= 16: insertion sort)
        int32_t * a = L->j_pairs.data() + L->j_pair0[j];
        const int n = L->j_pair0[j + 1] - L->j_pair0[j];
        for (int x = 1; x < n; ++x) {
            const int32_t v = a[x];
            int y = x - 1;
            while (y >= 0 && pair_s[a[y]] > pair_s[v]) {
                a[y + 1] = a[y];
                --y;
            }
            a[y + 1] = v;
        }
    }
    L->tok = tok;
    if (n_cpu > 0) {
        const int64_t n_embd = L->host[0]->ne[0];
        const int64_t n_ff   = L->host[0]->ne[1];
        const size_t rs0 = ggml_row_size(L->kern[0].vdt, n_embd);
        const size_t rs1 = ggml_row_size(L->kern[1].vdt, n_embd);
        const size_t rsd = ggml_row_size(L->kern[2].vdt, n_ff);
        if (g->xq[0].size() < (size_t) n_tok*rs0) {
            g->xq[0].resize((size_t) n_tok*rs0);
        }
        if (L->kern[1].vdt != L->kern[0].vdt && g->xq[1].size() < (size_t) n_tok*rs1) {
            g->xq[1].resize((size_t) n_tok*rs1);
        }
        if (g->act.size() < (size_t) n_pairs*n_ff) {
            g->act.resize((size_t) n_pairs*n_ff);
        }
        if (g->actq.size() < (size_t) n_pairs*rsd) {
            g->actq.resize((size_t) n_pairs*rsd);
        }
        if (g->y.size() < (size_t) n_pairs*n_embd) {
            g->y.resize((size_t) n_pairs*n_embd);
        }
    }
}


void cps_plan_op(ggml_tensor * dst, int ith, int nth, void * ud) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }
    cps_state * g = g_cps;
    cps_layer * L = (cps_layer *) ud;
    if (!g || L->pos < 0 || L->pos >= (int) g->layers.size() || g->layers[L->pos].get() != L) {
        GGML_ABORT("cpushare: a graph outlived its state");
    }
    const ggml_tensor * ids_t = dst->src[0];
    const ggml_tensor * w_t   = dst->src[1];
    const int      k     = (int) ids_t->ne[0];
    const int      T     = (int) ids_t->ne[1];
    const int64_t  ne    = L->n_expert;
    const bool     mode1 = g->cfg.mode == 1 && w_t != nullptr;
    const int32_t * ids  = (const int32_t *) ids_t->data;
    const float   * w    = mode1 ? (const float *) w_t->data : nullptr;

    cps_release_tickets(*L); // a graph cut after an earlier plan of this layer left them
    for (auto & b : L->bar) {
        b.store(0, std::memory_order_relaxed);
    }
    L->cpu.clear();
    L->slice.clear();
    L->tok.clear();

    if (L->pos == 0) {
        cps_ubatch_begin(g, k, T);
    }
    const bool share = llama_prefill_stream_share_active();
    if (!share || L->late.size() != (size_t) ne) {
        L->late.assign((size_t) ne, 0); // no late phase (an aborted begin, or the plain gate began this ubatch)
    }
    if (L->hot.size() != (size_t) ne) {
        L->hot.assign((size_t) ne, 0);
    }

    // the plan
    const bool pins = cps_pin_active();
    const bool full = share && llama_prefill_stream_layer_full(L->view);
    const bool cpu_on = mode1 && share && pins && !full && L->kern_ok && g->cfg.k_fixed != 0;
    llama_fn_cps_pick_in in;
    in.n_expert  = (int) ne;
    in.n_used    = k;
    in.n_tokens  = T;
    in.ids       = ids;
    in.late      = share ? L->late.data() : nullptr;
    in.cand      = nullptr;
    in.bytes     = L->bytes.data();
    in.rows_max  = g->cfg.rows;
    in.k_fixed   = cpu_on ? g->cfg.k_fixed : 0;
    in.cap       = g->cfg.cap;
    in.pairs_max = 256*g->cfg.rows;
    in.c_e       = g->rls.th[0];
    in.c_r       = g->rls.th[1];
    {
        const llama_pfs_share_counters sc = llama_prefill_stream_share_counters();
        in.dma_bpu = sc.dma_bpu_ewma > 0 ? std::min(60000.0, std::max(5000.0, sc.dma_bpu_ewma)) : 40000.0;
    }
    in.frac = 0.8;
    // [TAG_R35_ROUTED] the no-wait rule: the stream may skip any expert it has not copied yet that no row routes to
    in.skip_all = g->cfg.mode == 2 && g->cfg.nowait != 0;
    std::vector<cps_slice> slices((size_t) ne);
    uint64_t dropped = 0;
    if (cpu_on) {
        in.accept = [&](int e) -> bool {
            if (L->hot[e]) {
                return false;
            }
            const uint8_t cls = llama_fn_res_class(g->model, L->il, e);
            if (cls != LLAMA_FN_RES_PIN && cls != LLAMA_FN_RES_RAM) {
                return false;
            }
            int32_t tk[3] = { -1, -1, -1 };
            for (int part = 0; part < 3; ++part) {
                const ggml_tensor * t = L->host[part];
                const uint8_t * p = (const uint8_t *) t->data + (size_t) e*t->nb[2];
                const size_t    n = t->nb[2];
                const void * src = p;
                size_t  len = n;
                int32_t tkt = -1;
                if (!cps_pin_query(p, n, &src, &len, &tkt)) {
                    src = p;
                    len = n;
                    tkt = -1;
                }
                if (len >= n && src != nullptr && src != (const void *) p) {
                    slices[e].p[part] = (const uint8_t *) src; // a pinned copy: held by its ticket while the op reads it
                    tk[part] = tkt;
                } else if (src == (const void *) p) {
                    slices[e].p[part] = p; // the mapped bytes themselves
                    cps_pin_release(tkt);
                } else {
                    cps_pin_release(tkt); // a copy that does not hold the whole expert: the device computes it
                    for (int q = 0; q < part; ++q) {
                        cps_pin_release(tk[q]);
                    }
                    dropped++;
                    return false;
                }
            }
            for (int q = 0; q < 3; ++q) {
                if (tk[q] >= 0) {
                    L->tickets.push_back(tk[q]);
                }
            }
            return true;
        };
    }
    llama_fn_cps_pick_out out;
    llama_fn_cps_pick(in, out);
    if (!cpu_on) {
        out.cpu.clear(); // also when k_fixed made it empty: no CPU set
    }

    // the stream may skip the late experts of the plan now
    if (share) {
        llama_prefill_stream_share_publish(L->view, out.skip.data(), (int) ne);
    }

    // routing share EWMA for the next ubatch's late sets
    if (L->share.size() == (size_t) ne && T > 0) {
        for (int64_t e = 0; e < ne; ++e) {
            L->share[e] = 0.7f*L->share[e] + 0.3f*(float) out.rows[e]/(float) T;
        }
    }

    // the plan for the share op and the device ids
    if (mode1) {
        int32_t * P = (int32_t *) dst->data;
        const int64_t kT  = (int64_t) k*T;
        const int64_t cap = dst->ne[0] - kT;
        for (int64_t i = 0; i < kT; ++i) {
            const int32_t e = ids[i];
            P[i] = e >= 0 && e < ne ? out.table[e] : e;
        }
        const int n_tok = (int) out.tok.size();
        for (int64_t j = 0; j < cap; ++j) {
            P[kT + j] = j < n_tok ? out.tok[j] : (n_tok > 0 ? out.tok[0] : 0);
        }
        cps_plan_pairs(g, *L, ids, w, k, T, out.cpu, out.tok, slices);
    } else if (dst->data != nullptr) {
        memset(dst->data, 0, ggml_nbytes(dst));
    }
    if (L->cpu.empty()) {
        cps_release_tickets(*L); // nothing reads them
    }

    llama_fn_cps_sums d;
    d.layers      = 1;
    d.cpu_experts = L->cpu.size();
    d.cpu_pairs   = mode1 && !L->e_pair0.empty() ? (uint64_t) L->e_pair0.back() : 0;
    d.cpu_tokens  = L->tok.size();
    d.skip_cpu    = L->cpu.empty() ? 0 : out.skip_cpu;
    d.skip_unr    = out.skip_unr;
    d.dropped     = dropped;
    d.cut_cap     = (uint64_t) out.cut_cap;
    d.budget_k    = (double) L->cpu.size();
    d.pins_off    = mode1 && share && !pins ? 1 : 0;
    cps_sums_add(g, d);
}

// the share op's work for thread ith of nth: C [cap][n_embd] from the plan in L and the token rows x (row stride x_nb1)
void cps_share_run(cps_state * g, cps_layer * L, const char * x_data, size_t x_nb1, float * C, int64_t n_embd, int64_t cap,
        int ith, int nth) {
    const int     n_cpu  = (int) L->cpu.size();
    const int     n_tok  = (int) L->tok.size();
    if (n_cpu == 0 || n_tok == 0) {
        for (int64_t j = ith; j < cap; j += nth) {
            memset(C + j*n_embd, 0, (size_t) n_embd*sizeof(float)); // x + 0 at token 0 for every column
        }
        return;
    }
    const int64_t t0 = ith == 0 ? ggml_time_us() : 0;
    const cps_kern & ku = L->kern[0];
    const cps_kern & kg = L->kern[1];
    const cps_kern & kd = L->kern[2];
    const int64_t n_ff    = L->host[0]->ne[1];
    const size_t  bx_u    = L->host[0]->nb[1];
    const size_t  bx_g    = L->host[1]->nb[1];
    const size_t  bx_d    = L->host[2]->nb[1];
    const size_t  rs0     = ggml_row_size(ku.vdt, n_embd);
    const size_t  rs1     = ggml_row_size(kg.vdt, n_embd);
    const size_t  rsd     = ggml_row_size(kd.vdt, n_ff);
    const bool    two     = kg.vdt != ku.vdt;
    uint8_t *     xq0     = g->xq[0].data();
    uint8_t *     xq1     = two ? g->xq[1].data() : xq0;
    const size_t  rsg     = two ? rs1 : rs0;
    float *       act     = g->act.data();
    uint8_t *     actq    = g->actq.data();
    float *       y       = g->y.data();
    const int     n_pairs = L->e_pair0[n_cpu];

    // Q: the CPU tokens' rows in up's (and gate's) column type
    for (int j = ith; j < n_tok; j += nth) {
        const float * x = (const float *) (x_data + (size_t) L->tok[j]*x_nb1);
        ku.from_vdt(x, xq0 + (size_t) j*rs0, n_embd);
        if (two) {
            kg.from_vdt(x, xq1 + (size_t) j*rs1, n_embd);
        }
    }
    cps_barrier(L->bar[0], nth);

    // A: up and gate rows of each CPU expert for its pairs, act = silu(gate) x up
    {
        const int     n_rb = (int) ((n_ff + CPS_RA - 1)/CPS_RA);
        const int64_t n_pc = (int64_t) n_cpu*n_rb;
        float u[CPS_NC*CPS_RA];
        float v[CPS_NC*CPS_RA];
        for (int64_t pc = ith; pc < n_pc; pc += nth) {
            const int c  = (int) (pc/n_rb);
            const int r0 = (int) (pc % n_rb)*CPS_RA;
            const int nr = (int) std::min<int64_t>(CPS_RA, n_ff - r0);
            const uint8_t * wu = L->slice[c].p[0] + (size_t) r0*bx_u;
            const uint8_t * wg = L->slice[c].p[1] + (size_t) r0*bx_g;
            for (int q = L->e_pair0[c]; q < L->e_pair0[c + 1]; q += CPS_NC) {
                const int nc = std::min(CPS_NC, L->e_pair0[c + 1] - q);
                const void * cu[CPS_NC];
                const void * cg[CPS_NC];
                for (int i = 0; i < nc; ++i) {
                    cu[i] = xq0 + (size_t) L->pair_j[q + i]*rs0;
                    cg[i] = xq1 + (size_t) L->pair_j[q + i]*rsg;
                }
                cps_dot(ku, (int) n_embd, u, CPS_RA, wu, bx_u, nr, cu, nc);
                cps_dot(kg, (int) n_embd, v, CPS_RA, wg, bx_g, nr, cg, nc);
                for (int i = 0; i < nc; ++i) {
                    float * a = act + (size_t) (q + i)*n_ff + r0;
                    for (int r = 0; r < nr; ++r) {
                        a[r] = cps_silu(v[i*CPS_RA + r])*u[i*CPS_RA + r];
                    }
                }
            }
        }
    }
    cps_barrier(L->bar[1], nth);

    // A2: act rows in down's column type
    for (int q = ith; q < n_pairs; q += nth) {
        kd.from_vdt(act + (size_t) q*n_ff, actq + (size_t) q*rsd, n_ff);
    }
    cps_barrier(L->bar[2], nth);

    // B: down rows of each CPU expert for its pairs
    {
        const int     n_rb = (int) ((n_embd + CPS_RB - 1)/CPS_RB);
        const int64_t n_pc = (int64_t) n_cpu*n_rb;
        float o[CPS_NC*CPS_RB];
        for (int64_t pc = ith; pc < n_pc; pc += nth) {
            const int c  = (int) (pc/n_rb);
            const int r0 = (int) (pc % n_rb)*CPS_RB;
            const int nr = (int) std::min<int64_t>(CPS_RB, n_embd - r0);
            const uint8_t * wd = L->slice[c].p[2] + (size_t) r0*bx_d;
            for (int q = L->e_pair0[c]; q < L->e_pair0[c + 1]; q += CPS_NC) {
                const int nc = std::min(CPS_NC, L->e_pair0[c + 1] - q);
                const void * ca[CPS_NC];
                for (int i = 0; i < nc; ++i) {
                    ca[i] = actq + (size_t) (q + i)*rsd;
                }
                cps_dot(kd, (int) n_ff, o, CPS_RB, wd, bx_d, nr, ca, nc);
                for (int i = 0; i < nc; ++i) {
                    memcpy(y + (size_t) (q + i)*n_embd + r0, o + i*CPS_RB, (size_t) nr*sizeof(float));
                }
            }
        }
    }
    cps_barrier(L->bar[3], nth);
    if (ith == 0) {
        cps_release_tickets(*L); // every thread is past its last weight read
    }

    // C: per token the weighted sum of its CPU pairs in slot order; padding columns repeat token 0's column
    for (int64_t j = ith; j < cap; j += nth) {
        const int jj = j < n_tok ? (int) j : 0;
        float * cj = C + j*n_embd;
        memset(cj, 0, (size_t) n_embd*sizeof(float));
        for (int x = L->j_pair0[jj]; x < L->j_pair0[jj + 1]; ++x) {
            const int     q  = L->j_pairs[x];
            const float   wq = L->pair_w[q];
            const float * yq = y + (size_t) q*n_embd;
            for (int64_t i = 0; i < n_embd; ++i) {
                cj[i] += wq*yq[i];
            }
        }
    }
    if (ith == 0) {
        const double us = (double) (ggml_time_us() - t0);
        llama_fn_cps_rls_update(g->rls, (double) n_cpu, (double) n_pairs, us);
        llama_fn_cps_sums d;
        d.cpu_ms = us/1e3;
        cps_sums_add(g, d);
    }
}

void cps_share_op(ggml_tensor * dst, int ith, int nth, void * ud) {
    cps_state * g = g_cps;
    cps_layer * L = (cps_layer *) ud;
    if (!g || L->pos < 0 || L->pos >= (int) g->layers.size() || g->layers[L->pos].get() != L) {
        GGML_ABORT("cpushare: a graph outlived its state");
    }
    const ggml_tensor * xt = dst->src[1];
    cps_share_run(g, L, (const char *) xt->data, xt->nb[1], (float *) dst->data, dst->ne[0], dst->ne[1], ith, nth);
}

cps_layer * cps_layer_get(cps_state * g, int pos, int n_pos) {
    if (g->layers.empty()) {
        g->layers.resize((size_t) n_pos);
    }
    if (pos < 0 || pos >= (int) g->layers.size()) {
        return nullptr;
    }
    if (!g->layers[pos]) {
        g->layers[pos] = std::make_unique<cps_layer>();
        g->layers[pos]->pos = pos;
    }
    return g->layers[pos].get();
}

} // namespace

// ------------------------------------------------------------------------------------------------------------------------
// pure helpers
// ------------------------------------------------------------------------------------------------------------------------

uint64_t llama_fn_cps_late(const llama_fn_cps_late_in & in, uint8_t * late) {
    if (in.n_expert <= 0 || !late) {
        return 0;
    }
    memset(late, 0, (size_t) in.n_expert);
    if (!in.share || !in.bytes || in.late_rows <= 0) {
        return 0;
    }
    std::vector<std::pair<float, int>> c;
    for (int e = 0; e < in.n_expert; ++e) {
        if (in.hot && in.hot[e]) {
            continue;
        }
        const float pred = in.share[e]*(float) in.n_tokens;
        if (pred <= (float) in.late_rows) {
            c.push_back({ pred, e });
        }
    }
    std::sort(c.begin(), c.end());
    uint64_t sum = 0;
    for (const auto & x : c) {
        const uint64_t b = in.bytes[x.second];
        if (sum + b > in.byte_cap) {
            break;
        }
        late[x.second] = 1;
        sum += b;
    }
    return sum;
}

void llama_fn_cps_pick(const llama_fn_cps_pick_in & in, llama_fn_cps_pick_out & out) {
    const int ne = in.n_expert;
    out = llama_fn_cps_pick_out();
    out.rows.assign((size_t) std::max(0, ne), 0);
    out.skip.assign((size_t) std::max(0, ne), 0);
    out.cpu_m.assign((size_t) std::max(0, ne), 0);
    out.table.resize((size_t) std::max(0, ne));
    for (int e = 0; e < ne; ++e) {
        out.table[e] = e;
    }
    if (ne <= 0 || in.n_tokens <= 0 || in.n_used <= 0 || !in.ids) {
        return;
    }
    const int T = in.n_tokens;
    const int k = in.n_used;
    for (int64_t i = 0; i < (int64_t) T*k; ++i) {
        const int32_t e = in.ids[i];
        if (e >= 0 && e < ne) {
            out.rows[e]++;
        }
    }
    if (!in.late) {
        return; // no late phase: nothing to skip, nothing for the CPU
    }
    for (int e = 0; e < ne; ++e) {
        if (in.late[e] && out.rows[e] > 0) {
            out.late_left += in.bytes ? in.bytes[e] : 0;
        }
    }
    // candidates
    std::vector<int> cand;
    if (in.k_fixed != 0) {
        for (int e = 0; e < ne; ++e) {
            if (in.late[e] && out.rows[e] >= 1 && out.rows[e] <= in.rows_max && (!in.cand || in.cand[e])) {
                cand.push_back(e);
            }
        }
    }
    std::sort(cand.begin(), cand.end(), [&](int a, int b) {
        if (out.rows[a] != out.rows[b]) {
            return out.rows[a] < out.rows[b];
        }
        const uint64_t ba = in.bytes ? in.bytes[a] : 0;
        const uint64_t bb = in.bytes ? in.bytes[b] : 0;
        if (ba != bb) {
            return ba > bb;
        }
        return a < b;
    });
    // the tokens of each candidate
    std::vector<int32_t> off((size_t) ne + 1, 0);
    for (int e = 0; e < ne; ++e) {
        off[e + 1] = off[e] + out.rows[e];
    }
    std::vector<int32_t> toks((size_t) off[ne]);
    {
        std::vector<int32_t> f(off.begin(), off.end() - 1);
        for (int t = 0; t < T; ++t) {
            for (int s = 0; s < k; ++s) {
                const int32_t e = in.ids[(size_t) t*k + s];
                if (e >= 0 && e < ne) {
                    toks[f[e]++] = t;
                }
            }
        }
    }
    std::vector<uint8_t> mark((size_t) T, 0);
    int    n_tok = 0;
    int    n_cpu = 0;
    double t_sum = 0.0;
    for (int e : cand) {
        if (in.k_fixed >= 0 && n_cpu >= in.k_fixed) {
            break;
        }
        const uint64_t be = in.bytes ? in.bytes[e] : 0;
        const double   te = in.c_e + in.c_r*out.rows[e];
        if (in.k_fixed < 0) {
            const uint64_t left = out.late_left > be ? out.late_left - be : 0;
            if (t_sum + te > in.frac*(double) left/std::max(1.0, in.dma_bpu)) {
                continue;
            }
        }
        int add = 0;
        for (int32_t x = off[e]; x < off[e + 1]; ++x) {
            add += mark[toks[x]] ? 0 : 1;
        }
        if (n_tok + add > in.cap || out.n_pairs + out.rows[e] > in.pairs_max) {
            out.cut_cap++;
            continue;
        }
        if (in.accept && !in.accept(e)) {
            out.rejected++;
            continue;
        }
        for (int32_t x = off[e]; x < off[e + 1]; ++x) {
            if (!mark[toks[x]]) {
                mark[toks[x]] = 1;
                n_tok++;
            }
        }
        out.cpu_m[e] = 1;
        out.n_pairs += out.rows[e];
        out.late_left = out.late_left > be ? out.late_left - be : 0;
        t_sum += te;
        n_cpu++;
    }
    out.t_cpu = t_sum;
    for (int e = 0; e < ne; ++e) {
        if (out.cpu_m[e]) {
            out.cpu.push_back(e);
            out.table[e] = ne;
            out.skip[e]  = 1;
            out.skip_cpu += in.bytes ? in.bytes[e] : 0;
        } else if ((in.late[e] || in.skip_all) && out.rows[e] == 0) { // [TAG_R35_ROUTED] skip_all: late or not
            out.skip[e]  = 1;
            out.skip_unr += in.bytes ? in.bytes[e] : 0;
        }
    }
    for (int t = 0; t < T; ++t) {
        if (mark[t]) {
            out.tok.push_back(t);
        }
    }
}

void llama_fn_cps_rls::update(double x0, double x1, double yv) {
    const double px0 = P[0][0]*x0 + P[0][1]*x1;
    const double px1 = P[1][0]*x0 + P[1][1]*x1;
    const double den = lambda + x0*px0 + x1*px1;
    if (!(den > 1e-12) || !std::isfinite(yv)) {
        return;
    }
    const double k0  = px0/den;
    const double k1  = px1/den;
    const double err = yv - (th[0]*x0 + th[1]*x1);
    th[0] += k0*err;
    th[1] += k1*err;
    const double xp0 = x0*P[0][0] + x1*P[1][0];
    const double xp1 = x0*P[0][1] + x1*P[1][1];
    double n00 = (P[0][0] - k0*xp0)/lambda;
    double n01 = (P[0][1] - k0*xp1)/lambda;
    double n10 = (P[1][0] - k1*xp0)/lambda;
    double n11 = (P[1][1] - k1*xp1)/lambda;
    const double sym = 0.5*(n01 + n10);
    // bounded: no wind-up while the inputs do not change
    n00 = std::min(1e6, std::max(1e-6, n00));
    n11 = std::min(1e6, std::max(1e-6, n11));
    P[0][0] = n00;
    P[1][1] = n11;
    P[0][1] = P[1][0] = std::max(-std::sqrt(n00*n11), std::min(std::sqrt(n00*n11), sym));
    th[0] = std::min(5000.0, std::max(1.0, th[0]));
    th[1] = std::min(500.0, std::max(0.05, th[1]));
    n++;
}

void llama_fn_cps_rls_update(llama_fn_cps_rls & r, double x0, double x1, double y) {
    r.update(x0, x1, y);
}

bool llama_fn_cps_reference(const llama_fn_cps_ref_in & in, const std::vector<uint8_t> & cpu_m, const std::vector<int32_t> & tok,
        std::vector<float> & c) {
    cps_state tmp;
    cps_lookup(&tmp);
    tmp.mr_get = nullptr; // vec_dot per (row, column): the reference of the multi-row kernels
    cps_kern k[3];
    const ggml_tensor * w[3] = { in.up, in.gate, in.down };
    for (int i = 0; i < 3; ++i) {
        if (!cps_kern_init(&tmp, w[i]->type, k[i])) {
            return false;
        }
    }
    const int64_t n_embd = in.up->ne[0];
    const int64_t n_ff   = in.up->ne[1];
    c.assign((size_t) tok.size()*n_embd, 0.0f);
    std::vector<uint8_t> xu(ggml_row_size(k[0].vdt, n_embd));
    std::vector<uint8_t> xg(ggml_row_size(k[1].vdt, n_embd));
    std::vector<float>   a((size_t) n_ff);
    std::vector<uint8_t> aq(ggml_row_size(k[2].vdt, n_ff));
    std::vector<float>   yv((size_t) n_embd);
    for (size_t j = 0; j < tok.size(); ++j) {
        const int t = tok[j];
        const float * x = in.x + (size_t) t*n_embd;
        k[0].from_vdt(x, xu.data(), n_embd);
        k[1].from_vdt(x, xg.data(), n_embd);
        for (int s = 0; s < in.n_used; ++s) {
            const int32_t e = in.ids[(size_t) t*in.n_used + s];
            if (e < 0 || e >= (int32_t) cpu_m.size() || !cpu_m[e]) {
                continue;
            }
            const uint8_t * wu = (const uint8_t *) in.up->data   + (size_t) e*in.up->nb[2];
            const uint8_t * wg = (const uint8_t *) in.gate->data + (size_t) e*in.gate->nb[2];
            const uint8_t * wd = (const uint8_t *) in.down->data + (size_t) e*in.down->nb[2];
            for (int64_t r = 0; r < n_ff; ++r) {
                float u = 0.0f;
                float v = 0.0f;
                k[0].vec_dot((int) n_embd, &u, 0, wu + (size_t) r*in.up->nb[1], 0, xu.data(), 0, 1);
                k[1].vec_dot((int) n_embd, &v, 0, wg + (size_t) r*in.gate->nb[1], 0, xg.data(), 0, 1);
                a[r] = cps_silu(v)*u;
            }
            k[2].from_vdt(a.data(), aq.data(), n_ff);
            for (int64_t r = 0; r < n_embd; ++r) {
                k[2].vec_dot((int) n_ff, &yv[r], 0, wd + (size_t) r*in.down->nb[1], 0, aq.data(), 0, 1);
            }
            const float wq = in.w[(size_t) t*in.n_used + s];
            float * cj = c.data() + j*n_embd;
            for (int64_t r = 0; r < n_embd; ++r) {
                cj[r] += wq*yv[r];
            }
        }
    }
    return true;
}

bool llama_fn_cps_share_test(const llama_fn_cps_ref_in & in, const std::vector<uint8_t> & cpu_m, int nth, int cap, bool use_mr,
        std::vector<int32_t> & tok, std::vector<float> & c) {
    auto g = std::make_unique<cps_state>();
    cps_lookup(g.get());
    if (!use_mr) {
        g->mr_get = nullptr;
    }
    auto L = std::make_unique<cps_layer>();
    L->host[0]  = in.up;
    L->host[1]  = in.gate;
    L->host[2]  = in.down;
    L->n_expert = in.up->ne[2];
    for (int i = 0; i < 3; ++i) {
        if (!cps_kern_init(g.get(), L->host[i]->type, L->kern[i])) {
            return false;
        }
    }
    const int64_t ne = L->n_expert;
    std::vector<int32_t> cpu;
    for (int64_t e = 0; e < ne && e < (int64_t) cpu_m.size(); ++e) {
        if (cpu_m[e]) {
            cpu.push_back((int32_t) e);
        }
    }
    std::vector<uint8_t> mark((size_t) in.n_tokens, 0);
    for (int t = 0; t < in.n_tokens; ++t) {
        for (int s = 0; s < in.n_used; ++s) {
            const int32_t e = in.ids[(size_t) t*in.n_used + s];
            if (e >= 0 && e < ne && e < (int32_t) cpu_m.size() && cpu_m[e]) {
                mark[t] = 1;
            }
        }
    }
    tok.clear();
    for (int t = 0; t < in.n_tokens; ++t) {
        if (mark[t]) {
            tok.push_back(t);
        }
    }
    if ((int) tok.size() > cap || cap < 1) {
        return false;
    }
    std::vector<cps_slice> slices((size_t) ne);
    for (int32_t e : cpu) {
        for (int part = 0; part < 3; ++part) {
            slices[e].p[part] = (const uint8_t *) L->host[part]->data + (size_t) e*L->host[part]->nb[2];
        }
    }
    cps_plan_pairs(g.get(), *L, in.ids, in.w, in.n_used, in.n_tokens, cpu, tok, slices);
    const int64_t n_embd = in.up->ne[0];
    c.assign((size_t) cap*n_embd, NAN);
    for (auto & b : L->bar) {
        b.store(0);
    }
    nth = std::max(1, nth);
    std::vector<std::thread> th;
    for (int i = 1; i < nth; ++i) {
        th.emplace_back([&, i]() {
            cps_share_run(g.get(), L.get(), (const char *) in.x, (size_t) n_embd*sizeof(float), c.data(), n_embd, cap, i, nth);
        });
    }
    cps_share_run(g.get(), L.get(), (const char *) in.x, (size_t) n_embd*sizeof(float), c.data(), n_embd, cap, 0, nth);
    for (auto & t : th) {
        t.join();
    }
    return true;
}

// ------------------------------------------------------------------------------------------------------------------------
// configuration
// ------------------------------------------------------------------------------------------------------------------------

void llama_fn_cpushare_test_pins(int active, llama_fn_cps_pin_query_t q, llama_fn_cps_pin_release_t r) {
    g_pin_active_mock = active;
    g_pin_query_mock  = q;
    g_pin_rel_mock    = r;
}

int llama_fn_cpushare_configure_cfg(const llama_fn_cps_cfg & cfg, const llama_model * model, const void * owner) {
    std::lock_guard<std::mutex> lk(g_cps_mtx);
    if (cfg.mode <= 0) {
        return 0;
    }
    if (g_cps && g_cps->owner != owner) {
        LLAMA_LOG_WARN("%s: [TAG_FN_L22_CPUSHARE] another context owns the CPU share - off for this one\n", __func__);
        return 0;
    }
    if (!g_cps) {
        g_cps = new cps_state();
    }
    cps_state * g = g_cps;
    g->owner = owner;
    g->model = model;
    g->cfg   = cfg;
    cps_lookup(g);
    if (g->cfg.mode == 1 && !g->traits) {
        LLAMA_LOG_WARN("%s: [TAG_FN_L22_CPUSHARE] the CPU backend has no kernel lookup: mode 2 (skip only)\n", __func__);
        g->cfg.mode = 2;
    }
    const bool nowait = g->cfg.mode == 2 && g->cfg.nowait != 0; // [TAG_R35_ROUTED]
    LLAMA_LOG_INFO("%s: [TAG_FN_L22_CPUSHARE] mode %d (%s): ubatches <= %d rows, late experts <= %d predicted rows, CPU experts "
            "1..%d rows, %d tokens, k %d, plan wait <= %d ms, multi-row kernels %s; [TAG_R35_CPSGUARD] guard %d KiB\n", __func__,
            g->cfg.mode, g->cfg.mode == 1 ? "CPU share while pins are active + skip of unrouted late experts" :
            (nowait ? "[TAG_R35_ROUTED] no wait: skip of every expert not copied yet that no row routes to" :
                      "skip of unrouted late experts"),
            g->cfg.max_t, g->cfg.late_rows, g->cfg.rows, g->cfg.cap, g->cfg.k_fixed, nowait ? 0 : g->cfg.wait_ms,
            g->mr_get ? "on" : "off", g->cfg.guard_kib);
    llama_fn_l22_log("switch", "sub=cps mode=%d max=%d rows=%d late=%d cap=%d k=%d wait_ms=%d nowait=%d phases=%d guard_kib=%d",
            g->cfg.mode, g->cfg.max_t, g->cfg.rows, g->cfg.late_rows, g->cfg.cap, g->cfg.k_fixed, g->cfg.wait_ms, nowait ? 1 : 0,
            g->cfg.phases, g->cfg.guard_kib);
    return g->cfg.mode;
}

int llama_fn_cpushare_configure(const llama_model & model, const void * owner) {
    if (model.arch != LLM_ARCH_QWEN4EXP || model.no_alloc()) {
        return 0;
    }
    const char * v = llama_fn_env(model, "LLAMA_FN_L22_CPUSHARE");
    if (!llama_fn_l22_switch_on(v)) {
        return 0;
    }
    auto num = [&](const char * name, int def, int lo, int hi) {
        const char * s = llama_fn_env(model, name);
        const int x = s && s[0] ? atoi(s) : def;
        return std::max(lo, std::min(hi, x));
    };
    llama_fn_cps_cfg c;
    c.mode      = atoi(v) == 2 ? 2 : 1;
    c.rows      = num("LLAMA_FN_L22_CPUSHARE_ROWS", 8, 1, 64);
    c.max_t     = num("LLAMA_FN_L22_CPUSHARE_MAX", 3072, 32, 16384);
    c.cap       = num("LLAMA_FN_L22_CPUSHARE_CAP", 512, 16, 4096);
    c.k_fixed   = num("LLAMA_FN_L22_CPUSHARE_K", -1, -1, 4096);
    c.wait_ms   = num("LLAMA_FN_L22_CPUSHARE_WAIT_MS", 100, 1, 10000);
    c.late_rows = num("LLAMA_FN_L22_CPUSHARE_LATE", 16, 0, 4096);
    c.early_kib = num("LLAMA_FN_L22_CPUSHARE_EARLY_KIB", -1, -1, 1 << 22);
    c.nowait    = num("LLAMA_R35_CPS_NOWAIT", 1, 0, 1);       // [TAG_R35_ROUTED] 0: the v0.27.0 rule
    c.phases    = num("LLAMA_R35_CPS_PHASES", 0, 0, 1);       // [TAG_R35_ROUTED] 1: early then late experts
    c.guard_kib = num("LLAMA_R35_CPS_GUARD_KIB", 4, 0, 1024); // [TAG_R35_CPSGUARD] 0: the v0.27.0 copies
    return llama_fn_cpushare_configure_cfg(c, &model, owner);
}

void llama_fn_cpushare_free(const void * owner) {
    std::lock_guard<std::mutex> lk(g_cps_mtx);
    if (!g_cps || g_cps->owner != owner) {
        return;
    }
    for (auto & L : g_cps->layers) {
        if (L) {
            cps_release_tickets(*L);
        }
    }
    delete g_cps;
    g_cps = nullptr;
}

const llama_fn_cps_cfg * llama_fn_cpushare_cfg() {
    const cps_state * g = g_cps;
    return g && g->cfg.mode > 0 ? &g->cfg : nullptr;
}

bool llama_fn_cpushare_zero_expert() {
    const cps_state * g = g_cps;
    return g && g->cfg.mode == 1;
}

size_t llama_fn_cpushare_reserve_extra(int64_t n_embd, int64_t n_used, int64_t n_tokens) {
    const cps_state * g = g_cps;
    if (!g || g->cfg.mode != 1 || n_embd < 1 || n_used < 1 || n_tokens < 1 || n_tokens > g->cfg.max_t) {
        return 0;
    }
    const int64_t cap = std::min<int64_t>(g->cfg.cap, n_tokens);
    // share copy, rows and sum [n_embd, cap] F32; ids, w and the plan copy [k, T] (+ cap) 4-byte; 1 MiB for alignment
    return (size_t) (3*n_embd*cap*4 + (3*n_used*n_tokens + cap)*4) + ((size_t) 1 << 20);
}

llama_fn_cps_sums llama_fn_cpushare_take() {
    cps_state * g = g_cps;
    if (!g) {
        return {};
    }
    std::lock_guard<std::mutex> lk(g->sums_mtx);
    llama_fn_cps_sums r = g->sums;
    g->sums = {};
    r.rls_e = g->rls.th[0]; // written by the share op's thread 0; this read runs after the graph
    r.rls_r = g->rls.th[1];
    return r;
}

// ------------------------------------------------------------------------------------------------------------------------
// graph
// ------------------------------------------------------------------------------------------------------------------------

bool llama_fn_cpushare_build(ggml_context * ctx, ggml_cgraph * gf, ggml_backend_sched_t sched, ggml_backend_t cpu, int il,
        llama_fn_cpushare_io * io, bool chain_ok) {
    cps_state * g = g_cps;
    if (!g || g->cfg.mode <= 0 || !io || !io->pfs || !chain_ok) {
        return false;
    }
    const llama_pfs_view * v = (const llama_pfs_view *) io->pfs;
    const int pos   = llama_prefill_stream_pos(v);
    const int n_pos = llama_prefill_stream_n_pos();
    if (pos < 0 || pos >= n_pos) {
        return false;
    }
    ggml_tensor * cur = io->cur;
    ggml_tensor * ids = io->selected_experts;
    ggml_tensor * w   = io->weights;
    const ggml_tensor * hu = io->up_host;
    const ggml_tensor * hg = io->gate_host;
    const ggml_tensor * hd = io->down_host;
    if (!cur || !ids || !w || !hu || !hg || !hd) {
        return false;
    }
    const int64_t T      = cur->ne[1];
    const int64_t k      = ids->ne[0];
    const int64_t n_embd = cur->ne[0];
    const int64_t ne     = hu->ne[2];
    if (T < 1 || T > g->cfg.max_t || k < 1 || k > CPS_MAX_USED || ids->type != GGML_TYPE_I32 || ids->ne[1] != T ||
        ggml_nelements(w) != k*T || ne < 1 || hg->ne[2] != ne || hd->ne[2] != ne || hu->ne[0] != n_embd ||
        hg->ne[0] != n_embd || hd->ne[1] != n_embd || hd->ne[0] != hu->ne[1] || hg->ne[1] != hu->ne[1] ||
        !hu->data || !hg->data || !hd->data) {
        return false;
    }
    // without active pins a mode-1 layer is built as mode 2 (skip only): no token rows go to the host for a share that
    // cannot run (a graph kept until the pins come shares nothing; one built with pins shares nothing once they go)
    const bool mode1 = g->cfg.mode == 1 && cps_pin_active();
    ggml_tensor * z[3] = { nullptr, nullptr, nullptr };
    if (mode1) {
        if (cur->type != GGML_TYPE_F32 || cur->nb[0] != sizeof(float) || w->type != GGML_TYPE_F32 ||
            !llama_prefill_stream_view_z(v, z)) {
            return false;
        }
    }
    cps_layer * L = cps_layer_get(g, pos, n_pos);
    if (!L) {
        return false;
    }
    if (L->il != il || L->host[0] != hu || L->n_expert != ne) {
        L->il       = il;
        L->host[0]  = hu;
        L->host[1]  = hg;
        L->host[2]  = hd;
        L->n_expert = ne;
        L->share.clear();
        L->share_init = false;
        L->bytes.assign((size_t) ne, (uint64_t) (hu->nb[2] + hg->nb[2] + hd->nb[2]));
        L->kern_ok = cps_kern_init(g, hu->type, L->kern[0]) && cps_kern_init(g, hg->type, L->kern[1]) &&
                     cps_kern_init(g, hd->type, L->kern[2]);
        if (L->kern_ok) {
            const int64_t n_ff = hu->ne[1];
            L->kern_ok = n_embd % ggml_blck_size(L->kern[0].vdt) == 0 && n_embd % ggml_blck_size(L->kern[1].vdt) == 0 &&
                         n_ff % ggml_blck_size(L->kern[2].vdt) == 0 && n_embd % ggml_blck_size(hu->type) == 0 &&
                         n_ff % ggml_blck_size(hd->type) == 0;
        }
    }
    L->view = v;

    ggml_backend_t dev = llama_prefill_stream_device();
    auto pin = [&](ggml_tensor * t, ggml_backend_t b) {
        if (sched && b) {
            ggml_backend_sched_set_tensor_backend(sched, t, b);
        }
    };
    ggml_tensor * ids_c = ggml_cont(ctx, ids); // [k, T] next to the router, copied to the host once
    ggml_format_name(ids_c, "cps_ids-%d", il);
    pin(ids_c, dev);
    ggml_tensor * w_c = nullptr;
    if (mode1) {
        w_c = ggml_reshape_2d(ctx, ggml_cont(ctx, w), k, T);
        ggml_format_name(w_c, "cps_w-%d", il);
        pin(w_c->view_src ? w_c->view_src : w_c, dev);
    }
    const int64_t cap = mode1 ? std::min<int64_t>(g->cfg.cap, T) : 0;
    ggml_tensor * pargs[2] = { ids_c, w_c };
    ggml_tensor * plan = ggml_custom_4d(ctx, GGML_TYPE_I32, mode1 ? k*T + cap : 1, 1, 1, 1, pargs, mode1 ? 2 : 1, cps_plan_op, 1, L);
    ggml_format_name(plan, "cps_plan-%d", il);
    pin(plan, cpu);
    ggml_build_forward_expand(gf, plan);
    ggml_tensor * dep = plan;
    ggml_tensor * share = nullptr;
    if (mode1) {
        ggml_tensor * sargs[2] = { plan, cur };
        share = ggml_custom_4d(ctx, GGML_TYPE_F32, n_embd, cap, 1, 1, sargs, 2, cps_share_op, GGML_N_TASKS_MAX, L);
        ggml_format_name(share, "cps_share-%d", il);
        pin(share, cpu);
        ggml_build_forward_expand(gf, share);
        dep = share;
    }
    ggml_tensor * gate = llama_prefill_stream_build_gate_share(ctx, gf, v, dep);
    pin(gate, cpu);
    io->gate_built = true;
    if (mode1) {
        ggml_tensor * ids_dev = ggml_view_2d(ctx, plan, k, T, k*sizeof(int32_t), 0);
        ggml_format_name(ids_dev, "cps_ids_dev-%d", il);
        io->ids     = ids_dev;
        io->up_mm   = z[0];
        io->gate_mm = z[1];
        io->down_mm = z[2];
        L->g_plan   = plan;
        L->g_share  = share;
        L->g_kT     = k*T;
    } else {
        L->g_plan  = nullptr;
        L->g_share = nullptr;
    }
    io->user = L;
    return true;
}

ggml_tensor * llama_fn_cpushare_build_finish(ggml_context * ctx, ggml_backend_sched_t sched, llama_fn_cpushare_io * io,
        ggml_tensor * moe_out) {
    cps_state * g = g_cps;
    cps_layer * L = io ? (cps_layer *) io->user : nullptr;
    if (!g || !L || g->cfg.mode != 1 || !L->g_plan || !L->g_share || !moe_out || moe_out->type != GGML_TYPE_F32 ||
        moe_out->ne[0] != L->g_share->ne[0] || moe_out->ne[2] != 1 || moe_out->ne[3] != 1) {
        return moe_out;
    }
    ggml_backend_t dev = llama_prefill_stream_device();
    auto pin = [&](ggml_tensor * t) {
        if (sched && dev) {
            ggml_backend_sched_set_tensor_backend(sched, t, dev);
        }
    };
    const int64_t cap = L->g_share->ne[1];
    ggml_tensor * idx  = ggml_view_1d(ctx, L->g_plan, cap, L->g_kT*sizeof(int32_t));
    ggml_tensor * rows = ggml_get_rows(ctx, moe_out, idx);  // [n_embd, cap]
    pin(rows);
    ggml_tensor * sum  = ggml_add(ctx, rows, L->g_share);
    pin(sum);
    ggml_tensor * out  = ggml_set_rows(ctx, moe_out, sum, idx);
    pin(out);
    ggml_format_name(out, "cps_out-%d", L->il);
    L->g_plan  = nullptr;
    L->g_share = nullptr;
    return out;
}

// the hook (R-P16-1): qwen4exp's streamed layers, separate up / gate, SILU, no bias / scale / clamp
bool llama_fn_cpushare_plan(llm_graph_context * g, const llama_ubatch & ubatch, int il, void * plan) {
    GGML_UNUSED(ubatch);
    cps_state * st = g_cps;
    auto * io = (llama_fn_cpushare_io *) plan;
    if (!st || !g || !io || st->cfg.mode <= 0 || !st->model || g->arch != LLM_ARCH_QWEN4EXP || il < 0 ||
        il >= (int) st->model->layers.size()) {
        return false;
    }
    const auto & layer = st->model->layers[il];
    // [TAG_R38_E2] per-expert scales are fine for the skip-only mode (2: no expert runs on the host, the device matmuls
    // keep their scales); the host share (mode 1) has no scales, so a scaled layer stays out of it
    const bool no_scales = !layer.ffn_up_exps_s && !layer.ffn_gate_exps_s && !layer.ffn_down_exps_s;
    const bool r38_skip_only = llama_r38_fp4s() && st->cfg.mode == 2 && layer.ffn_up_exps_s && layer.ffn_gate_exps_s &&
        layer.ffn_down_exps_s;
    const bool chain_ok = layer.ffn_up_exps == io->up_host && layer.ffn_gate_exps == io->gate_host &&
        layer.ffn_down_exps == io->down_host && !layer.ffn_gate_up_exps && !layer.ffn_up_exps_b && !layer.ffn_gate_exps_b &&
        !layer.ffn_down_exps_b && (no_scales || r38_skip_only) &&
        !(g->hparams.swiglu_clamp_exp[il] > 1e-6f) && (!g->loras || g->loras->empty());
    return llama_fn_cpushare_build(g->ctx0, g->gf, g->sched, g->backend_cpu, il, io, chain_ok);
}

ggml_tensor * llama_fn_cpushare_finish(llm_graph_context * g, void * plan, ggml_tensor * moe_out) {
    if (!g) {
        return moe_out;
    }
    return llama_fn_cpushare_build_finish(g->ctx0, g->sched, (llama_fn_cpushare_io *) plan, moe_out);
}
