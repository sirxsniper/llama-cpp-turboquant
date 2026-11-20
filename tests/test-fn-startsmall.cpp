// [TAG_FN_STARTSMALL] the start-up compute reserve (LLAMA_FN_STARTSMALL): the constructor measures its worst-case reserve,
// releases it and holds one of the SMALL width until the hot set fit or a wider batch.
//
// The pure rules (src/llama-fn-auto.h) are checked by enumeration in test-fn-auto (built on every platform).
//
// Here (with a model file): whole decodes of a tiny qwen4exp model with LLAMA_FN_STARTSMALL=2 against unset (a CPU build
// has no deferred hot set fit, so mode 2 is the one that runs here), 2 and 4 sequences: narrow batches (under the start-up
// reserve), then a wide prompt (it ends the start-up reserve), then narrow and wide batches again. Checked:
//   - all logits have the same bits;
//   - the memory breakdown reports the same compute bytes with the switch as without it, before and after the wide batch
//     (the server's RAM fit reads it during the load);
//   - the log says the reserve was released and held smaller, and that the wide batch ended it, once each;
//   - no more graph reallocations than without the switch (ctest also runs it with GGML_SCHED_DEBUG_REALLOC=1).
// Usage: test-fn-startsmall <qwen4exp-moe.gguf> [threads]   (make the file with: test-llama-archs -a qwen4exp -o <dir>)
// No GPU (run with CUDA_VISIBLE_DEVICES=-1).

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include "../src/llama-ext.h" // llama_get_memory_breakdown

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

uint32_t g_n_vocab = 0;
int      g_verbose = 0;
int      g_n_released = 0; // "[TAG_FN_STARTSMALL] the ... compute reserve ... is released"
int      g_n_end_wide = 0; // "[TAG_FN_STARTSMALL] the start-up reserve ends (a wider batch)"
int      g_n_realloc  = 0; // "graph reallocation:" (llama_context::sched_graph_realloc)
double   g_held_mib   = -1.0;
double   g_meas_mib   = -1.0;

void log_cb(ggml_log_level level, const char * text, void * /*ud*/) {
    if (g_verbose > 0 || level == GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
    if (strstr(text, "[TAG_FN_STARTSMALL]") && strstr(text, "is released")) {
        ++g_n_released;
        // "... (devices %.2f MiB, host %.2f MiB) is released until ...; held now: %u tokens (devices %.2f MiB, host %.2f MiB)"
        const char * a = strstr(text, "host ");
        const char * b = strstr(text, "held now:");
        const char * c = b ? strstr(b, "host ") : nullptr;
        if (a && c) {
            g_meas_mib = atof(a + 5);
            g_held_mib = atof(c + 5);
        }
    }
    if (strstr(text, "[TAG_FN_STARTSMALL] the start-up reserve ends (a wider batch)")) {
        ++g_n_end_wide;
    }
    if (strstr(text, "graph reallocation:")) {
        ++g_n_realloc;
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

size_t compute_bytes(const llama_context * ctx) {
    size_t b = 0;
    for (const auto & [buft, mb] : llama_get_memory_breakdown(ctx)) {
        b += mb.compute;
    }
    return b;
}

struct run_out {
    std::vector<float> logits;
    size_t compute_load = 0; // breakdown right after the context exists
    size_t compute_end  = 0; // after the last batch
    int    n_released   = 0;
    int    n_end_wide   = 0;
    int    n_end_early  = 0; // end lines before the wide prompt
    int    n_realloc    = 0;
    double held_mib     = -1.0;
    double meas_mib     = -1.0;
    bool   ok           = false;
};

run_out run(const char * path, const char * mode, int n_seq_max, int n_threads) {
    run_out r;
    set_env("LLAMA_FN_STARTSMALL", mode);
    g_n_released = g_n_end_wide = g_n_realloc = 0;
    g_held_mib = g_meas_mib = -1.0;

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
    cp.n_seq_max       = (uint32_t) n_seq_max;
    cp.kv_unified      = true;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.n_threads       = n_threads;
    cp.n_threads_batch = n_threads;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) {
        fprintf(stderr, "cannot create a context\n");
        llama_model_free(model);
        return r;
    }
    r.compute_load = compute_bytes(ctx);

    const llama_token V = (llama_token) g_n_vocab;
    auto tk = [V](int s, int i) { return (llama_token) (1 + (s*37 + i*11 + (i*i) % 7) % (V - 1)); };

    bool ok = true;
    llama_pos p[4] = { 0, 0, 0, 0 };
    int       n[4] = { 0, 0, 0, 0 };
    auto rows = [&](std::vector<int> cnt, bool all_out) {
        std::vector<llama_token> t;
        std::vector<llama_pos> ps;
        std::vector<llama_seq_id> sq;
        std::vector<int8_t> o;
        for (int s = 0; s < (int) cnt.size(); ++s) {
            for (int i = 0; i < cnt[(size_t) s]; ++i) {
                t.push_back(tk(s, n[s]++));
                ps.push_back(p[s]++);
                sq.push_back(s);
                o.push_back(all_out || i % 8 == 7 || i == cnt[(size_t) s] - 1);
            }
        }
        return decode(ctx, t, ps, sq, o, r.logits);
    };

    std::vector<int> one(n_seq_max, 1), three(n_seq_max, 3);
    ok = ok && rows({ 2 }, true);       // the warm-up's two tokens
    ok = ok && rows(one, true);         // a decode step of every sequence
    ok = ok && rows(three, true);       // a verify step of every sequence (n_max 2)
    ok = ok && rows({ 0, 9 }, false);   // a short prompt within the start-up width
    r.n_end_early = g_n_end_wide;
    ok = ok && rows({ 300 }, false);    // a long prompt: ends the start-up reserve
    ok = ok && rows(three, true);
    ok = ok && rows({ 0, 140 }, false);
    ok = ok && rows(one, true);

    r.compute_end = compute_bytes(ctx);
    r.n_released  = g_n_released;
    r.n_end_wide  = g_n_end_wide;
    r.n_realloc   = g_n_realloc;
    r.held_mib    = g_held_mib;
    r.meas_mib    = g_meas_mib;

    llama_free(ctx);
    llama_model_free(model);
    r.ok = ok;
    return r;
}

void compare(int n_seq_max, const run_out & on, const run_out & off) {
    TCHECK(on.ok && off.ok, "a run failed (on ok %d, off ok %d, %d sequences)", (int) on.ok, (int) off.ok, n_seq_max);
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
    printf("  %d sequences: %zu logits, %s; breakdown compute at load %.2f / %.2f MiB, at the end %.2f / %.2f MiB (on / off); "
            "start-up reserve: measured host %.2f MiB, held %.2f MiB; reallocations %d / %d\n", n_seq_max, on.logits.size(),
            same ? "same bits" : "DIFFER", on.compute_load/1048576.0, off.compute_load/1048576.0, on.compute_end/1048576.0,
            off.compute_end/1048576.0, on.meas_mib, on.held_mib, on.n_realloc, off.n_realloc);
    TCHECK(finite, "non-finite logits");
    TCHECK(same, "the start-up reserve must not change the logits (%d sequences)", n_seq_max);
    TCHECK(on.compute_load == off.compute_load, "the breakdown at load must report the worst case: %zu vs %zu", on.compute_load,
            off.compute_load);
    TCHECK(on.compute_end == off.compute_end, "the breakdown after the wide batch: %zu vs %zu", on.compute_end, off.compute_end);
    TCHECK(on.n_released == 1 && off.n_released == 0, "released lines %d / %d", on.n_released, off.n_released);
    TCHECK(on.n_end_wide == 1 && off.n_end_wide == 0, "end lines %d / %d", on.n_end_wide, off.n_end_wide);
    TCHECK(on.n_end_early == 0, "the start-up reserve ended before the wide prompt (%d end lines)", on.n_end_early);
    TCHECK(on.held_mib >= 0.0 && on.meas_mib > 0.0 && on.held_mib < on.meas_mib, "held %.2f MiB must be below the measured %.2f MiB",
            on.held_mib, on.meas_mib);
    TCHECK(on.n_realloc <= off.n_realloc, "reallocations %d with the switch, %d without", on.n_realloc, off.n_realloc);
}

} // namespace

int main(int argc, char ** argv) {
    if (const char * v = getenv("TEST_VERBOSE"); v && atoi(v) > 0) {
        g_verbose = atoi(v);
    }
    if (argc < 2) {
        printf("test-fn-startsmall: SKIPPED (no model file; make one with test-llama-archs -a qwen4exp -o <dir>)\n");
        return 0;
    }
    const char * path = argv[1];
    const int n_threads = argc > 2 ? std::max(1, atoi(argv[2])) : 4;
    llama_log_set(log_cb, nullptr);
    llama_backend_init();

    int n_pairs = 0;
    for (int n_seq_max : { 2, 4 }) {
        const run_out off = run(path, nullptr, n_seq_max, n_threads);
        const run_out on  = run(path, "2", n_seq_max, n_threads);
        compare(n_seq_max, on, off);
        ++n_pairs;
        // mode 1 needs the deferred hot set fit, which a CPU build never has: nothing changes
        const run_out on1 = run(path, "1", n_seq_max, n_threads);
        TCHECK(on1.ok && on1.logits == off.logits && on1.n_released == 0 && on1.compute_load == off.compute_load,
                "LLAMA_FN_STARTSMALL=1 without the deferred fit must change nothing (%d sequences)", n_seq_max);
        ++n_pairs;
    }

    llama_backend_free();
    printf("test-fn-startsmall: %d whole-decode comparisons, %d checks, %d failed\n", n_pairs, g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
