// [TAG_QSA_POOL_RANGE] the QSA pool mask as one range per token (LLAMA_QSA_POOL_RANGE=1, ggml_kpool_mask) against the
// explicit [n_pool, n_tokens] f16 mask input, on the CPU backend, bit for bit.
//
// Part 1 (always): ggml_kpool_mask against the rows the explicit form writes (-INF, 0 over [first, first + count), -INF),
// whole and in row chunks (row0 / n_rows, as TURBO_QSA_CHUNK builds them), and through a view of the range rows. Ranges:
// empty, the whole row, at both edges, and values outside the row (the op clamps them).
//
// Part 2 (with a model file): whole decodes of a tiny qwen4exp model with the switch on against off.
// Usage: test-fn-qsa-pool-range <qwen4exp-moe.gguf> [threads]   (make the file with: test-llama-archs -a qwen4exp -o <dir>)
// Two sequences: two long prompts (several ubatches), a batch with prompt rows of both, joint decode steps with a
// rollback, a removed conversation and a new one in its cells, a last joint step. Every eighth prompt row and every
// decode row is an output, and all of their logits must have the same bits with the switch on as with it off, for
// every combination of: positional QSA masks or the explicit mask, n_rs_seq 0 or 3, the incremental fill or the full
// rebuild (LLAMA_FN_L10_KPOOLINC), listed sequence pools off / on / checked (LLAMA_FN_L22_QSA_SEQPOOLS 0 / 1 / 2), and the
// width bound (LLAMA_QSA_POOL_RANGE_MIN 0, 16 and unset = 128); then with one KV stream per sequence (kv_unified off),
// and with two-chain decode graphs (LLAMA_FN_L22_TBO=1 and the bound at 0, so the chains take row views of the ranges).
// Each run also counts its GGML_OP_KPOOL_MASK nodes: some with the switch on, none with it off.
// TURBO_QSA_CHUNK and LLAMA_KPOOL_VERIFY are read once per process: run the test once with TURBO_QSA_CHUNK unset (one
// mask per layer), once with e.g. TURBO_QSA_CHUNK=24 (chunks), and again with LLAMA_KPOOL_VERIFY=1.
// No GPU (run with CUDA_VISIBLE_DEVICES=-1).

#include "llama.h"
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
#include <vector>

namespace {

int g_fail   = 0;
int g_checks = 0;

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

void set_env(const char * name, const char * value) {
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

// ---- part 1: the op ----

// the rows of the explicit form (fill_mask of llama_memory_hybrid_idx_context::set_input_kpool), ranges clamped to the row
std::vector<ggml_fp16_t> ref_rows(const std::vector<int32_t> & range, int64_t n_pool, int64_t row0, int64_t n_rows) {
    const ggml_fp16_t keep = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t drop = ggml_fp32_to_fp16(-INFINITY);
    std::vector<ggml_fp16_t> m((size_t) (n_pool*n_rows), drop);
    for (int64_t r = 0; r < n_rows; ++r) {
        const int64_t p0 = std::min<int64_t>(std::max<int64_t>(range[2*(size_t) (row0 + r)], 0), n_pool);
        const int64_t nv = std::min<int64_t>(std::max<int64_t>(range[2*(size_t) (row0 + r) + 1], 0), n_pool - p0);
        std::fill(m.begin() + r*n_pool + p0, m.begin() + r*n_pool + p0 + nv, keep);
    }
    return m;
}

void op_case(int64_t n_pool, int64_t n_batch, int64_t chunk, bool wild, int seed, int n_threads) {
    std::mt19937 rng((uint32_t) seed);
    std::vector<int32_t> range((size_t) (2*n_batch));
    for (int64_t i = 0; i < n_batch; ++i) {
        int32_t p0 = (int32_t) (rng() % (uint64_t) (n_pool + 1));
        int32_t nv = (int32_t) (rng() % (uint64_t) (n_pool - p0 + 1));
        switch (rng() % 8) {
            case 0: nv = 0; break;                        // nothing visible
            case 1: p0 = 0; nv = (int32_t) n_pool; break; // every pool
            case 2: p0 = 0; break;                        // from the first pool (a listed sequence)
            case 3: nv = (int32_t) n_pool - p0; break;    // up to the last pool
            default: break;
        }
        if (wild && i % 5 == 0) {
            p0 = (int32_t) (rng() % 3 == 0 ? -7 : n_pool + 3);
            nv = (int32_t) (rng() % 2 == 0 ? -1 : 2*n_pool);
        } else if (wild && i % 5 == 1) {
            nv = (int32_t) (2*n_pool); // a count past the end of the row
        }
        range[2*(size_t) i]     = p0;
        range[2*(size_t) i + 1] = nv;
    }

    const size_t mem = (size_t) (n_pool*n_batch)*2*sizeof(ggml_fp16_t) + ((size_t) 64 << 20);
    ggml_init_params ip = { mem, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    TCHECK(ctx != nullptr, "ggml_init failed");
    if (ctx == nullptr) {
        return;
    }
    ggml_tensor * rg = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, n_batch);
    memcpy(rg->data, range.data(), range.size()*sizeof(int32_t));

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    // the whole mask, its chunks, and the same chunks through a view of the range rows
    ggml_tensor * whole = ggml_kpool_mask(ctx, rg, n_pool, 0, n_batch);
    ggml_build_forward_expand(gf, whole);
    std::vector<ggml_tensor *> parts;
    std::vector<ggml_tensor *> parts_v;
    std::vector<int64_t>       part_r0;
    for (int64_t c0 = 0; c0 < n_batch; c0 += chunk) {
        const int64_t nc = std::min(chunk, n_batch - c0);
        parts.push_back(ggml_kpool_mask(ctx, rg, n_pool, c0, nc));
        ggml_build_forward_expand(gf, parts.back());
        ggml_tensor * rv = ggml_view_2d(ctx, rg, 2, nc, rg->nb[1], (size_t) c0*rg->nb[1]);
        parts_v.push_back(ggml_kpool_mask(ctx, rv, n_pool, 0, nc));
        ggml_build_forward_expand(gf, parts_v.back());
        part_r0.push_back(c0);
    }
    const bool ran = ggml_graph_compute_with_ctx(ctx, gf, n_threads) == GGML_STATUS_SUCCESS;
    TCHECK(ran, "graph compute failed");
    if (ran) {
        const auto ref = ref_rows(range, n_pool, 0, n_batch);
        TCHECK(whole->type == GGML_TYPE_F16 && whole->ne[0] == n_pool && whole->ne[1] == n_batch, "shape of the whole mask");
        TCHECK(memcmp(whole->data, ref.data(), ref.size()*sizeof(ggml_fp16_t)) == 0,
                "whole mask differs (n_pool %lld, n_batch %lld, wild %d)", (long long) n_pool, (long long) n_batch, (int) wild);
        for (size_t k = 0; k < parts.size(); ++k) {
            const int64_t nc = parts[k]->ne[1];
            const auto refc = ref_rows(range, n_pool, part_r0[k], nc);
            TCHECK(memcmp(parts[k]->data, refc.data(), refc.size()*sizeof(ggml_fp16_t)) == 0,
                    "chunk at row %lld differs (n_pool %lld, chunk %lld)", (long long) part_r0[k], (long long) n_pool, (long long) chunk);
            TCHECK(memcmp(parts_v[k]->data, refc.data(), refc.size()*sizeof(ggml_fp16_t)) == 0,
                    "chunk through a view at row %lld differs (n_pool %lld, chunk %lld)", (long long) part_r0[k], (long long) n_pool,
                    (long long) chunk);
        }
    }
    ggml_free(ctx);
}

// ---- part 2: whole decodes ----

struct run_cfg {
    bool range;    // LLAMA_QSA_POOL_RANGE
    int  min_rows; // LLAMA_QSA_POOL_RANGE_MIN (-1: unset)
    bool pos_mask; // LLAMA_QSA_POS_MASK
    int  n_rs;     // n_rs_seq
    bool inc;      // LLAMA_FN_L10_KPOOLINC
    int  seqpools; // LLAMA_FN_L22_QSA_SEQPOOLS
    bool unified;  // kv_unified
    bool tbo;      // LLAMA_FN_L22_TBO=1 (with LLAMA_FN_L22_TBO_CPUTEST=1: two-chain decode graphs without a bridge)
};

struct run_out {
    std::vector<float> logits;
    int  n_kpool = 0; // GGML_OP_KPOOL_MASK nodes computed
    bool ok = false;
};

bool count_kpool(struct ggml_tensor * t, bool ask, void * ud) {
    if (ask && t->op == GGML_OP_KPOOL_MASK) {
        ++*(int *) ud;
    }
    return false;
}

uint32_t g_n_vocab = 0;
int      g_verbose = 0;

void log_cb(ggml_log_level level, const char * text, void * /*ud*/) {
    if (g_verbose > 0 || level == GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

bool decode(llama_context * ctx, const std::vector<llama_token> & tok, const std::vector<llama_pos> & pos,
        const std::vector<llama_seq_id> & seq, const std::vector<int8_t> & out, std::vector<float> & logits) {
    const int n = (int) tok.size();
    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        b.token[i]     = tok[(size_t) i];
        b.pos[i]       = pos[(size_t) i];
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = seq[(size_t) i];
        b.logits[i]    = out[(size_t) i];
    }
    b.n_tokens = n;
    const int rc = llama_decode(ctx, b);
    if (rc == 0) {
        for (int i = 0; i < n; ++i) {
            if (!out[(size_t) i]) {
                continue;
            }
            const float * l = llama_get_logits_ith(ctx, i);
            logits.insert(logits.end(), l, l + g_n_vocab);
        }
    }
    llama_batch_free(b);
    return rc == 0;
}

run_out run(const char * path, const run_cfg & cfg, int n_threads) {
    run_out r;
    set_env("LLAMA_QSA_POOL_RANGE", cfg.range ? "1" : "0");
    set_env("LLAMA_QSA_POOL_RANGE_MIN", cfg.min_rows < 0 ? nullptr : std::to_string(cfg.min_rows).c_str());
    set_env("LLAMA_QSA_POS_MASK", cfg.pos_mask ? "1" : "0");
    set_env("LLAMA_FN_L10_KPOOLINC", cfg.inc ? "1" : "0");
    set_env("LLAMA_FN_L22_QSA_SEQPOOLS", std::to_string(cfg.seqpools).c_str());
    set_env("LLAMA_FN_L22_TBO", cfg.tbo ? "1" : "0");
    set_env("LLAMA_FN_L22_TBO_CPUTEST", cfg.tbo ? "1" : nullptr);

    llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t no_dev[1] = { nullptr };
    mp.devices = no_dev; // CPU only
    llama_model * model = llama_model_load_from_file(path, mp);
    if (model == nullptr) {
        fprintf(stderr, "cannot load %s\n", path);
        return r;
    }
    g_n_vocab = (uint32_t) llama_vocab_n_tokens(llama_model_get_vocab(model));

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = 1024;
    cp.n_batch         = 512;
    cp.n_ubatch        = 256;
    cp.n_seq_max       = 2;
    cp.kv_unified      = cfg.unified;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.n_threads       = n_threads;
    cp.n_threads_batch = n_threads;
    cp.n_rs_seq        = (uint32_t) cfg.n_rs;
    cp.cb_eval           = count_kpool;
    cp.cb_eval_user_data = &r.n_kpool;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) {
        fprintf(stderr, "cannot create a context\n");
        llama_model_free(model);
        return r;
    }

    const llama_token V = (llama_token) g_n_vocab;
    auto tk = [V](int s, int i) { return (llama_token) (1 + (s*37 + i*11 + (i*i) % 7) % (V - 1)); };

    bool ok = true;
    llama_pos p[2] = { 0, 0 };
    int       n[2] = { 0, 0 }; // tokens given so far, for the token ids
    // rows of one or two sequences in one batch; every eighth row and the last one of each sequence is an output
    auto rows = [&](int n0, int n1, bool all_out) {
        std::vector<llama_token> t;
        std::vector<llama_pos> ps;
        std::vector<llama_seq_id> sq;
        std::vector<int8_t> o;
        const int cnt[2] = { n0, n1 };
        for (int s = 0; s < 2; ++s) {
            for (int i = 0; i < cnt[s]; ++i) {
                t.push_back(tk(s, n[s]++));
                ps.push_back(p[s]++);
                sq.push_back(s);
                o.push_back(all_out || i % 8 == 7 || i == cnt[s] - 1);
            }
        }
        return decode(ctx, t, ps, sq, o, r.logits);
    };

    ok = ok && rows(300, 0, false);  // a long prompt: ubatches of 256 and 44 rows
    ok = ok && rows(0, 140, false);  // the other sequence
    ok = ok && rows(40, 40, false);  // prompt rows of both sequences in one batch
    const int steps_T[] = { 3, 1, 3, 2 };
    int step = 0;
    for (int T : steps_T) {
        ok = ok && rows(T, T, true); // joint decode / verify steps
        if (ok && step == 2 && cfg.n_rs >= 2) {
            llama_memory_t mem = llama_get_memory(ctx);
            for (int s = 0; s < 2 && ok; ++s) {
                ok = llama_memory_seq_rm(mem, s, p[s] - 2, -1); // a draft rejected after its first token
                p[s] -= 2;
            }
            TCHECK(ok, "rollback refused (n_rs_seq %d)", cfg.n_rs);
        }
        ++step;
    }
    ok = ok && rows(0, 200, false);  // the second sequence goes deeper
    if (ok) {
        // the first conversation ends, a new one takes its cells
        ok = llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
        TCHECK(ok, "removing sequence 0 refused");
        p[0] = 0;
    }
    ok = ok && rows(150, 0, false);
    ok = ok && rows(2, 2, true);

    llama_free(ctx);
    llama_model_free(model);
    r.ok = ok;
    return r;
}

void compare(const run_cfg & c, const run_out & on, const run_out & off) {
    TCHECK(on.ok && off.ok, "a run failed (range on ok %d, off ok %d)", (int) on.ok, (int) off.ok);
    if (!on.ok || !off.ok) {
        return;
    }
    TCHECK(on.logits.size() == off.logits.size() && !on.logits.empty(), "logit counts %zu vs %zu", on.logits.size(), off.logits.size());
    if (on.logits.size() != off.logits.size()) {
        return;
    }
    bool finite = true;
    for (float v : off.logits) {
        finite = finite && std::isfinite(v);
    }
    const bool same = memcmp(on.logits.data(), off.logits.data(), on.logits.size()*sizeof(float)) == 0;
    if (g_verbose > 0 || !same) {
        printf("  min %4d, pos mask %d, n_rs_seq %d, inc %d, seqpools %d, unified %d, tbo %d: %zu logits, %s, %d mask ops\n",
                c.min_rows, (int) c.pos_mask, c.n_rs, (int) c.inc, c.seqpools, (int) c.unified, (int) c.tbo, on.logits.size(),
                same ? "same bits" : "DIFFER", on.n_kpool);
    }
    TCHECK(finite, "non-finite logits");
    TCHECK(same, "ranges vs the mask input must give the same bits (min %d, pos mask %d, n_rs_seq %d, inc %d, seqpools %d, "
            "unified %d, tbo %d)", c.min_rows, (int) c.pos_mask, c.n_rs, (int) c.inc, c.seqpools, (int) c.unified, (int) c.tbo);
    // the switch must take the range path, and only the switch
    TCHECK(on.n_kpool > 0, "no GGML_OP_KPOOL_MASK node ran with the switch on (min %d, unified %d, tbo %d)", c.min_rows,
            (int) c.unified, (int) c.tbo);
    TCHECK(off.n_kpool == 0, "%d GGML_OP_KPOOL_MASK nodes ran with the switch off", off.n_kpool);
}

} // namespace

int main(int argc, char ** argv) {
    const int n_threads = argc > 2 ? std::max(1, atoi(argv[2])) : 4;
    g_verbose = getenv("POOL_RANGE_TEST_VERBOSE") ? atoi(getenv("POOL_RANGE_TEST_VERBOSE")) : 0;

    ggml_cpu_init();
    int seed = 4100;
    for (int64_t n_pool : { 1, 2, 64, 511, 512, 4097 }) {
        for (int64_t n_batch : { 1, 3, 30, 129, 700 }) {
            for (int64_t chunk : { 1, 24, 512 }) {
                if (chunk > n_batch && chunk != 512) {
                    continue;
                }
                op_case(n_pool, n_batch, chunk, false, seed++, n_threads);
                op_case(n_pool, n_batch, chunk, true,  seed++, n_threads);
            }
        }
    }
    op_case(131584, 64, 24, false, seed++, n_threads); // the pool count of a 262K, 2-sequence reserve
    printf("test-fn-qsa-pool-range: the op: %d checks, %d failed\n", g_checks, g_fail);

    if (argc < 2) {
        printf("test-fn-qsa-pool-range: whole decodes SKIPPED (no model file; make one with test-llama-archs -a qwen4exp -o <dir>)\n");
        return g_fail == 0 ? 0 : 1;
    }
    const char * path = argv[1];
    llama_log_set(log_cb, nullptr);
    llama_backend_init();

    int n_pairs = 0;
    for (bool pos_mask : { true, false }) {
        for (int n_rs : { 0, 3 }) {
            for (bool inc : { true, false }) {
                for (int seqpools : { 0, 1, 2 }) {
                    const run_out off = run(path, { false, -1, pos_mask, n_rs, inc, seqpools, true, false }, n_threads);
                    for (int min_rows : { 0, 16, -1 }) {
                        const run_cfg c = { true, min_rows, pos_mask, n_rs, inc, seqpools, true, false };
                        compare(c, run(path, c, n_threads), off);
                        ++n_pairs;
                    }
                }
                // one KV stream per sequence (no listed pools there)
                const run_out off2 = run(path, { false, -1, pos_mask, n_rs, inc, 0, false, false }, n_threads);
                for (int min_rows : { 0, -1 }) {
                    const run_cfg c = { true, min_rows, pos_mask, n_rs, inc, 0, false, false };
                    compare(c, run(path, c, n_threads), off2);
                    ++n_pairs;
                }
            }
            // two-chain decode graphs: with the bound at 0 the chains read row views of the range input
            const run_out off3 = run(path, { false, -1, pos_mask, n_rs, true, 1, true, true }, n_threads);
            const run_cfg c3 = { true, 0, pos_mask, n_rs, true, 1, true, true };
            compare(c3, run(path, c3, n_threads), off3);
            ++n_pairs;
        }
    }

    llama_backend_free();
    printf("test-fn-qsa-pool-range: %d whole-decode comparisons, %d checks, %d failed\n", n_pairs, g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
