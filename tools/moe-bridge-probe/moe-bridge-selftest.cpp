// [TAG_MOE_BRIDGE] moe-bridge-probe --selftest: the MoE host bridge (GGML_OP_MOE_HOST_POST / WAIT on the GPU backend,
// the host job on the CPU MoE pool) against the CPU MUL_MAT_ID chain (up, gate, swiglu, down, weight, sum) on random
// weights, T = 1..8, spin and hostfunc waits, with and without a hot-expert table, plus a stall that must time out,
// report the error, write zeros, and recover after a reset. No model. Exit code 0 only if every case passes.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-moe-bridge.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

struct st_api {
    ggml_backend_moe_bridge_new_t        br_new        = nullptr;
    ggml_backend_moe_bridge_free_t       br_free       = nullptr;
    ggml_backend_moe_bridge_id_t         br_id         = nullptr;
    ggml_backend_moe_bridge_set_runner_t br_set_runner = nullptr;
    ggml_backend_moe_bridge_poll_t       br_poll       = nullptr;
    ggml_backend_moe_bridge_complete_t   br_complete   = nullptr;
    ggml_backend_moe_bridge_error_t      br_error      = nullptr;
    ggml_backend_moe_bridge_reset_t      br_reset      = nullptr;
    ggml_backend_moe_bridge_get_stats_t  br_stats      = nullptr;

    ggml_cpu_moe_pool_params (*pool_params_default)(int)                                         = nullptr;
    ggml_cpu_moe_pool *      (*pool_new)(const ggml_cpu_moe_pool_params *)                       = nullptr;
    void                     (*pool_free)(ggml_cpu_moe_pool *)                                   = nullptr;
    ggml_status              (*pool_run)(ggml_cpu_moe_pool *, const ggml_cpu_moe_job *)          = nullptr;
};

struct st_weights {
    ggml_context * ctx  = nullptr;
    ggml_tensor  * up   = nullptr;
    ggml_tensor  * gate = nullptr;
    ggml_tensor  * down = nullptr;
    ggml_tensor  * table = nullptr; // i32 [1, n_expert]: value n_hot_slots = on the host, else "hot" (served elsewhere)
    int32_t        table_miss = 0;
};

struct st_runner {
    const st_api *     api  = nullptr;
    ggml_cpu_moe_pool * pool = nullptr;
    ggml_cpu_moe_layer  layer     = {};
    ggml_cpu_moe_layer  layer_tbl = {};
    std::atomic<uint64_t> jobs{0};
};

bool st_run(const ggml_moe_bridge_job * j, void * ud) {
    auto * r = (st_runner *) ud;
    const ggml_cpu_moe_job job = { (j->flags & GGML_MOE_BRIDGE_JOB_TABLE) ? &r->layer_tbl : &r->layer,
                                   j->n_tokens, j->n_used, j->x, j->ids, j->w, j->out };
    r->jobs.fetch_add(1);
    return r->api->pool_run(r->pool, &job) == GGML_STATUS_SUCCESS;
}

struct st_executor {
    const st_api *    api = nullptr;
    ggml_moe_bridge * br  = nullptr;
    st_runner *       run = nullptr;
    std::atomic<bool> stop{false};
    std::atomic<bool> paused{false};
    std::thread       th;

    void start() {
        th = std::thread([this]() {
            while (!stop.load()) {
                if (paused.load()) {
                    std::this_thread::sleep_for(std::chrono::microseconds(200));
                    continue;
                }
                ggml_moe_bridge_job job;
                if (api->br_poll(br, &job)) {
                    const bool ok = st_run(&job, run);
                    api->br_complete(br, &job, ok);
                }
            }
        });
    }
    void join() {
        stop = true;
        if (th.joinable()) {
            th.join();
        }
    }
};

void fill_random(std::vector<float> & v, std::mt19937 & rng, float a) {
    std::uniform_real_distribution<float> d(-a, a);
    for (auto & x : v) {
        x = d(rng);
    }
}

ggml_tensor * new_quant(ggml_context * ctx, ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, std::mt19937 & rng, float a) {
    ggml_tensor * t = ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2);
    std::vector<float> f((size_t) ne0*ne1);
    for (int64_t e = 0; e < ne2; ++e) {
        fill_random(f, rng, a);
        ggml_quantize_chunk(type, f.data(), (char *) t->data + (size_t) e*t->nb[2], 0, ne1, ne0, nullptr);
    }
    return t;
}

bool make_weights(st_weights & w, ggml_type t_up, ggml_type t_gate, ggml_type t_down, int64_t n_embd, int64_t n_ff, int64_t n_exp,
                  std::mt19937 & rng) {
    const size_t bytes = ggml_row_size(t_up, n_embd)*n_ff*n_exp + ggml_row_size(t_gate, n_embd)*n_ff*n_exp +
                         ggml_row_size(t_down, n_ff)*n_embd*n_exp + n_exp*sizeof(int32_t) + 16*ggml_tensor_overhead() + (1 << 20);
    ggml_init_params ip = { bytes, nullptr, false };
    w.ctx = ggml_init(ip);
    if (!w.ctx) {
        return false;
    }
    w.up   = new_quant(w.ctx, t_up,   n_embd, n_ff, n_exp, rng, 0.08f);
    w.gate = new_quant(w.ctx, t_gate, n_embd, n_ff, n_exp, rng, 0.08f);
    w.down = new_quant(w.ctx, t_down, n_ff, n_embd, n_exp, rng, 0.08f);
    ggml_set_name(w.up,   "blk.0.ffn_up_exps.weight");
    ggml_set_name(w.gate, "blk.0.ffn_gate_exps.weight");
    ggml_set_name(w.down, "blk.0.ffn_down_exps.weight");
    // about a third of the experts are "hot" (served by a device chain in a real graph, so skipped on the host)
    w.table = ggml_new_tensor_2d(w.ctx, GGML_TYPE_I32, 1, n_exp);
    w.table_miss = (int32_t) (n_exp/3);
    int32_t * tb = (int32_t *) w.table->data;
    int32_t slot = 0;
    for (int64_t e = 0; e < n_exp; ++e) {
        tb[e] = (e % 3 == 1 && slot < w.table_miss) ? slot++ : w.table_miss;
    }
    return true;
}

// the reference: the CPU MUL_MAT_ID chain as build_moe_ffn builds it, with the table skip of the hot set
bool reference(ggml_backend_t cpu, const st_weights & wts, const std::vector<float> & x, const std::vector<int32_t> & ids,
               const std::vector<float> & w, int64_t n_embd, int n_used, int T, bool table, std::vector<float> & out) {
    const size_t mem = 64*ggml_tensor_overhead() + ggml_graph_overhead() +
        (size_t) T*n_used*(wts.up->ne[1]*3 + n_embd*3)*sizeof(float) + (size_t) T*(n_embd*4 + n_used*8)*sizeof(float) + (1 << 20);
    ggml_init_params ip = { mem, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        return false;
    }
    ggml_tensor * tx  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, T);
    ggml_tensor * tid = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, T);
    ggml_tensor * tw  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_used, T);
    memcpy(tx->data,  x.data(),   x.size()*sizeof(float));
    memcpy(tid->data, ids.data(), ids.size()*sizeof(int32_t));
    memcpy(tw->data,  w.data(),   w.size()*sizeof(float));

    ggml_tensor * up   = ggml_mul_mat_id(ctx, wts.up,   tx, tid);
    ggml_tensor * gate = ggml_mul_mat_id(ctx, wts.gate, tx, tid);
    ggml_tensor * act  = ggml_swiglu_split(ctx, gate, up);
    ggml_tensor * down = ggml_mul_mat_id(ctx, wts.down, act, tid);
    if (table) {
        for (ggml_tensor * t : { up, gate, down }) {
            t->src[3] = wts.table;
            t->op_params[0] = wts.table_miss;
        }
    }
    ggml_tensor * ex = ggml_mul(ctx, down, tw);
    ggml_tensor * sum = nullptr;
    for (int i = 0; i < n_used; ++i) {
        ggml_tensor * v = ggml_view_2d(ctx, ex, n_embd, T, ex->nb[2], i*ex->nb[1]);
        sum = sum ? ggml_add(ctx, sum, v) : ggml_cont(ctx, v);
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, sum);
    const bool ok = ggml_backend_graph_compute(cpu, gf) == GGML_STATUS_SUCCESS;
    if (ok) {
        out.assign((const float *) sum->data, (const float *) sum->data + (size_t) n_embd*T);
    }
    ggml_free(ctx);
    return ok;
}

// the device graph: post, some device work (the "shared expert"), wait on it
struct st_graph {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_cgraph *         gf  = nullptr;
    ggml_tensor * x = nullptr;
    ggml_tensor * ids_full = nullptr;
    ggml_tensor * w = nullptr;
    ggml_tensor * out = nullptr;

    bool build(ggml_backend_t gpu, int32_t bid, int32_t chan, int64_t n_embd, int n_used, int T, bool table) {
        ggml_init_params ip = { 32*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ctx = ggml_init(ip);
        x        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, T);
        ids_full = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used + 6, T); // a strided view, as the argsort top-k is
        w        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_used, T);
        ggml_set_input(x);
        ggml_set_input(ids_full);
        ggml_set_input(w);
        ggml_tensor * ids = ggml_view_2d(ctx, ids_full, n_used, T, ids_full->nb[1], 0);
        ggml_tensor * ticket = ggml_moe_host_post(ctx, x, ids, w, bid, chan, table ? GGML_MOE_BRIDGE_JOB_TABLE : 0);
        ggml_tensor * dep    = ggml_scale(ctx, ggml_sqr(ctx, x), 0.5f);
        out = ggml_moe_host_wait(ctx, ticket, dep, n_embd, T, bid, chan);
        ggml_set_output(out);
        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        if (!ggml_backend_supports_op(gpu, ticket) || !ggml_backend_supports_op(gpu, out)) {
            fprintf(stderr, "  the GPU backend does not support the bridge ops\n");
            return false;
        }
        buf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
        return buf != nullptr;
    }
    void free() {
        if (buf) {
            ggml_backend_buffer_free(buf);
        }
        if (ctx) {
            ggml_free(ctx);
        }
        buf = nullptr;
        ctx = nullptr;
    }
};

struct st_diff {
    double max_abs = 0.0;
    double ref_max = 0.0;
    double nmse    = 0.0;
    bool   bitwise = true;
};

st_diff compare(const std::vector<float> & a, const std::vector<float> & ref) {
    st_diff d;
    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double e = (double) a[i] - (double) ref[i];
        d.max_abs = std::max(d.max_abs, std::fabs(e));
        d.ref_max = std::max(d.ref_max, (double) std::fabs(ref[i]));
        num += e*e;
        den += (double) ref[i]*ref[i];
        if (memcmp(&a[i], &ref[i], sizeof(float)) != 0) {
            d.bitwise = false;
        }
    }
    d.nmse = den > 0.0 ? num/den : num;
    return d;
}

void random_job(std::mt19937 & rng, int64_t n_embd, int n_used, int n_exp, int T, std::vector<float> & x,
                std::vector<int32_t> & ids_full, std::vector<int32_t> & ids, std::vector<float> & w) {
    x.resize((size_t) n_embd*T);
    fill_random(x, rng, 1.0f);
    ids_full.assign((size_t) (n_used + 6)*T, -7);
    ids.resize((size_t) n_used*T);
    w.resize((size_t) n_used*T);
    std::uniform_real_distribution<float> du(0.01f, 1.0f);
    for (int t = 0; t < T; ++t) {
        std::vector<int32_t> perm(n_exp);
        for (int e = 0; e < n_exp; ++e) {
            perm[e] = e;
        }
        std::shuffle(perm.begin(), perm.end(), rng);
        float s = 0.0f;
        for (int k = 0; k < n_used; ++k) {
            ids[(size_t) t*n_used + k]            = perm[k];
            ids_full[(size_t) t*(n_used + 6) + k] = perm[k];
            w[(size_t) t*n_used + k] = du(rng);
            s += w[(size_t) t*n_used + k];
        }
        for (int k = 0; k < n_used; ++k) {
            w[(size_t) t*n_used + k] /= s;
        }
    }
}

bool run_graph(ggml_backend_t gpu, st_graph & g, const std::vector<float> & x, const std::vector<int32_t> & ids_full,
               const std::vector<float> & w, std::vector<float> & out) {
    ggml_backend_tensor_set(g.x, x.data(), 0, x.size()*sizeof(float));
    ggml_backend_tensor_set(g.ids_full, ids_full.data(), 0, ids_full.size()*sizeof(int32_t));
    ggml_backend_tensor_set(g.w, w.data(), 0, w.size()*sizeof(float));
    if (ggml_backend_graph_compute(gpu, g.gf) != GGML_STATUS_SUCCESS) {
        return false;
    }
    out.resize((size_t) ggml_nelements(g.out));
    ggml_backend_tensor_get(g.out, out.data(), 0, out.size()*sizeof(float));
    return true;
}

} // namespace

int moe_bridge_selftest(int argc, char ** argv) {
    int n_threads = 8;
    int n_exp     = 64;
    int reps      = 4;
    bool quick    = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--threads" && i + 1 < argc) { n_threads = std::max(1, atoi(argv[++i])); }
        else if (a == "--experts" && i + 1 < argc) { n_exp = std::max(16, atoi(argv[++i])); }
        else if (a == "--reps" && i + 1 < argc) { reps = std::max(1, atoi(argv[++i])); }
        else if (a == "--quick") { quick = true; }
    }

    printf("\nmoe-bridge-probe --selftest [TAG_MOE_BRIDGE]: %d experts, %d pool threads, %d reps per case\n", n_exp, n_threads, reps);

    ggml_backend_load_all();
    ggml_backend_dev_t gdev = nullptr;
    if (ggml_backend_reg_t r = ggml_backend_reg_by_name("CUDA"); r && ggml_backend_reg_dev_count(r) > 0) {
        gdev = ggml_backend_reg_dev_get(r, 0);
    } else {
        gdev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    }
    if (gdev == nullptr) {
        printf("  no GPU device\n");
        return 2;
    }
    ggml_backend_reg_t greg = ggml_backend_dev_backend_reg(gdev);
    ggml_backend_reg_t creg = ggml_backend_reg_by_name("CPU");
    st_api api;
    api.br_new        = (ggml_backend_moe_bridge_new_t)        ggml_backend_reg_get_proc_address(greg, "ggml_backend_moe_bridge_new");
    api.br_free       = (ggml_backend_moe_bridge_free_t)       ggml_backend_reg_get_proc_address(greg, "ggml_backend_moe_bridge_free");
    api.br_id         = (ggml_backend_moe_bridge_id_t)         ggml_backend_reg_get_proc_address(greg, "ggml_backend_moe_bridge_id");
    api.br_set_runner = (ggml_backend_moe_bridge_set_runner_t) ggml_backend_reg_get_proc_address(greg, "ggml_backend_moe_bridge_set_runner");
    api.br_poll       = (ggml_backend_moe_bridge_poll_t)       ggml_backend_reg_get_proc_address(greg, "ggml_backend_moe_bridge_poll");
    api.br_complete   = (ggml_backend_moe_bridge_complete_t)   ggml_backend_reg_get_proc_address(greg, "ggml_backend_moe_bridge_complete");
    api.br_error      = (ggml_backend_moe_bridge_error_t)      ggml_backend_reg_get_proc_address(greg, "ggml_backend_moe_bridge_error");
    api.br_reset      = (ggml_backend_moe_bridge_reset_t)      ggml_backend_reg_get_proc_address(greg, "ggml_backend_moe_bridge_reset");
    api.br_stats      = (ggml_backend_moe_bridge_get_stats_t)  ggml_backend_reg_get_proc_address(greg, "ggml_backend_moe_bridge_get_stats");
    api.pool_params_default = (decltype(api.pool_params_default)) ggml_backend_reg_get_proc_address(creg, "ggml_cpu_moe_pool_params_default");
    api.pool_new            = (decltype(api.pool_new))            ggml_backend_reg_get_proc_address(creg, "ggml_cpu_moe_pool_new");
    api.pool_free           = (decltype(api.pool_free))           ggml_backend_reg_get_proc_address(creg, "ggml_cpu_moe_pool_free");
    api.pool_run            = (decltype(api.pool_run))            ggml_backend_reg_get_proc_address(creg, "ggml_cpu_moe_run");
    if (!api.br_new || !api.br_free || !api.br_id || !api.br_set_runner || !api.br_poll || !api.br_complete || !api.br_error ||
        !api.br_reset || !api.br_stats || !api.pool_params_default || !api.pool_new || !api.pool_free || !api.pool_run) {
        printf("  the GPU backend (%s) or the CPU backend lacks the bridge / pool functions\n", ggml_backend_reg_name(greg));
        return 2;
    }
    int dev_index = 0;
    for (size_t i = 0; i < ggml_backend_reg_dev_count(greg); ++i) {
        if (ggml_backend_reg_dev_get(greg, i) == gdev) {
            dev_index = (int) i;
        }
    }

    ggml_backend_t gpu = ggml_backend_dev_init(gdev, nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!gpu || !cpu) {
        printf("  backend init failed\n");
        return 2;
    }
    if (auto set_n_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(creg, "ggml_backend_set_n_threads")) {
        set_n_threads(cpu, n_threads);
    }

    const int64_t n_embd = 2560;
    const int64_t n_ff   = 640;
    const int     n_used = 10;
    std::mt19937 rng(1234);

    struct type_set { ggml_type up, gate, down; };
    std::vector<type_set> sets = { { GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_1 } };
    if (!quick) {
        sets.push_back({ GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, GGML_TYPE_Q8_0 });
        sets.push_back({ GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, GGML_TYPE_IQ4_NL });
    }

    int n_pass = 0;
    int n_fail = 0;
    int n_bitwise = 0;

    for (const auto & ts : sets) {
        st_weights wts;
        if (!make_weights(wts, ts.up, ts.gate, ts.down, n_embd, n_ff, n_exp, rng)) {
            printf("  weight allocation failed\n");
            return 2;
        }
        printf("\n  types up %s, gate %s, down %s\n", ggml_type_name(ts.up), ggml_type_name(ts.gate), ggml_type_name(ts.down));

        for (int mode : { (int) GGML_MOE_BRIDGE_WAIT_SPIN, (int) GGML_MOE_BRIDGE_WAIT_HOSTFUNC }) {
            if (mode == GGML_MOE_BRIDGE_WAIT_HOSTFUNC && ts.down != GGML_TYPE_Q5_1) {
                continue;
            }
            ggml_moe_bridge_params bp = {};
            bp.device     = dev_index;
            bp.n_chan     = 2;
            bp.n_embd     = n_embd;
            bp.n_used     = n_used;
            bp.max_tokens = 8;
            bp.wait_mode  = mode;
            bp.timeout_ms = 200;
            bp.job_max_ms = 1000;
            bp.stats      = true;
            ggml_moe_bridge * br = api.br_new(&bp);
            if (br == nullptr) {
                printf("  bridge creation failed (%s mode)\n", mode == GGML_MOE_BRIDGE_WAIT_SPIN ? "spin" : "hostfunc");
                n_fail++;
                continue;
            }
            const int32_t bid = api.br_id(br);

            st_runner run;
            run.api = &api;
            ggml_cpu_moe_pool_params pp = api.pool_params_default(n_threads);
            pp.spin_us = 500;
            if (mode == GGML_MOE_BRIDGE_WAIT_SPIN) {
                pp.pin_caller = false;
            }
            run.pool = api.pool_new(&pp);
            run.layer = { wts.up, wts.gate, wts.down, nullptr, 0 };
            run.layer_tbl = { wts.up, wts.gate, wts.down, (const int32_t *) wts.table->data, wts.table_miss };

            st_executor ex;
            if (mode == GGML_MOE_BRIDGE_WAIT_SPIN) {
                ex.api = &api;
                ex.br  = br;
                ex.run = &run;
                ex.start();
            } else {
                api.br_set_runner(br, st_run, &run);
            }

            for (int T = 1; T <= 8; ++T) {
                for (int table = 0; table <= 1; ++table) {
                    st_graph g;
                    const int32_t chan = (T + table) % 2;
                    if (!g.build(gpu, bid, chan, n_embd, n_used, T, table != 0)) {
                        n_fail++;
                        g.free();
                        continue;
                    }
                    st_diff worst;
                    bool ok = true;
                    for (int r = 0; r < reps && ok; ++r) {
                        std::vector<float> x, w, out, ref;
                        std::vector<int32_t> ids_full, ids;
                        random_job(rng, n_embd, n_used, n_exp, T, x, ids_full, ids, w);
                        ok = run_graph(gpu, g, x, ids_full, w, out) &&
                             reference(cpu, wts, x, ids, w, n_embd, n_used, T, table != 0, ref);
                        if (!ok) {
                            break;
                        }
                        const st_diff d = compare(out, ref);
                        worst.max_abs = std::max(worst.max_abs, d.max_abs);
                        worst.ref_max = std::max(worst.ref_max, d.ref_max);
                        worst.nmse    = std::max(worst.nmse, d.nmse);
                        worst.bitwise = worst.bitwise && d.bitwise;
                    }
                    const uint32_t err = api.br_error(br);
                    const bool pass = ok && err == GGML_MOE_BRIDGE_ERR_NONE && worst.nmse < 1e-10 && worst.ref_max > 0.0;
                    printf("    %-8s T=%d table=%d: %s  max|d| %.3g of %.3g, nmse %.3g%s%s\n",
                            mode == GGML_MOE_BRIDGE_WAIT_SPIN ? "spin" : "hostfunc", T, table, pass ? "OK  " : "FAIL",
                            worst.max_abs, worst.ref_max, worst.nmse, worst.bitwise ? ", bitwise" : "",
                            err ? ", bridge error set" : "");
                    pass ? n_pass++ : n_fail++;
                    n_bitwise += pass && worst.bitwise ? 1 : 0;
                    g.free();
                }
            }

            // stall: the executor pauses (spin) / the runner sleeps (hostfunc) past the timeout
            if (mode == GGML_MOE_BRIDGE_WAIT_SPIN) {
                st_graph g;
                bool pass = g.build(gpu, bid, 0, n_embd, n_used, 3, false);
                std::vector<float> x, w, out, ref;
                std::vector<int32_t> ids_full, ids;
                random_job(rng, n_embd, n_used, n_exp, 3, x, ids_full, ids, w);
                ex.paused = true;
                const auto t0 = std::chrono::steady_clock::now();
                pass = pass && run_graph(gpu, g, x, ids_full, w, out);
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                bool zeros = true;
                for (float v : out) {
                    zeros = zeros && v == 0.0f;
                }
                const uint32_t err = api.br_error(br);
                const bool refused = !api.br_reset(br); // the executor still owes the job
                ex.paused = false;
                bool reset_ok = false;
                for (int k = 0; k < 500 && !reset_ok; ++k) {
                    reset_ok = api.br_reset(br);
                    if (!reset_ok) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(2));
                    }
                }
                random_job(rng, n_embd, n_used, n_exp, 3, x, ids_full, ids, w);
                const bool again = reset_ok && run_graph(gpu, g, x, ids_full, w, out) &&
                                   reference(cpu, wts, x, ids, w, n_embd, n_used, 3, false, ref) &&
                                   compare(out, ref).nmse < 1e-10 && api.br_error(br) == GGML_MOE_BRIDGE_ERR_NONE;
                pass = pass && zeros && err == GGML_MOE_BRIDGE_ERR_TIMEOUT && refused && again && ms >= 150.0 && ms < 2000.0;
                printf("    spin     stall: %s  returned after %.0f ms (timeout 200), error %u, zeros %d, reset refused while owed %d, "
                        "recovered %d\n", pass ? "OK  " : "FAIL", ms, err, (int) zeros, (int) refused, (int) again);
                pass ? n_pass++ : n_fail++;
                g.free();
            }

            // round trip of a real job through the bridge graph (informational, decides nothing)
            if (mode == GGML_MOE_BRIDGE_WAIT_SPIN || !quick) {
                for (int T : { 1, 3 }) {
                    st_graph g;
                    if (!g.build(gpu, bid, 1, n_embd, n_used, T, false)) {
                        g.free();
                        continue;
                    }
                    std::vector<float> x, w, out;
                    std::vector<int32_t> ids_full, ids;
                    random_job(rng, n_embd, n_used, n_exp, T, x, ids_full, ids, w);
                    for (int k = 0; k < 20; ++k) {
                        run_graph(gpu, g, x, ids_full, w, out);
                    }
                    const int iters = quick ? 100 : 400;
                    const auto t0 = std::chrono::steady_clock::now();
                    for (int k = 0; k < iters; ++k) {
                        ggml_backend_graph_compute(gpu, g.gf);
                    }
                    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count()/iters;
                    printf("    %-8s T=%d: %.1f us per graph (post + device work + host job of %d experts x %d tokens + wait)\n",
                            mode == GGML_MOE_BRIDGE_WAIT_SPIN ? "spin" : "hostfunc", T, us, n_used, T);
                    g.free();
                }
            }

            ggml_moe_bridge_stats st;
            api.br_stats(br, &st);
            printf("    stats: posts %llu, completed %llu, waits %llu (%.1f%% ready at once), %.1f us waited on average\n",
                    (unsigned long long) st.posts, (unsigned long long) st.completed, (unsigned long long) st.waits,
                    st.waits ? 100.0*st.waits_ready/st.waits : 0.0, st.waits ? st.wait_ns/1e3/st.waits : 0.0);

            ggml_backend_synchronize(gpu);
            ex.join();
            api.br_free(br);
            api.pool_free(run.pool);
        }
        ggml_free(wts.ctx);
    }

    ggml_backend_free(gpu);
    ggml_backend_free(cpu);

    printf("\nselftest: %d passed, %d failed (%d of the passed bitwise equal to the CPU chain)\n", n_pass, n_fail, n_bitwise);
    return n_fail == 0 ? 0 : 1;
}
