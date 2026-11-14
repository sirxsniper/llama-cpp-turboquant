// [TAG_FN_GDN_REPLAY] whole-decode CPU test of the GDN replay layout on a tiny qwen4exp model (the test-llama-archs fixture:
// GDN layers with the PLE block, a QSA attention layer, shared + routed experts, hyper-connections).
// Usage: test-fn-gdn-replay <qwen4exp-moe.gguf> [threads]   (make the file with: test-llama-archs -a qwen4exp -o <dir>)
// Without a file it prints SKIP and passes.
//
// Two contexts on one model: LLAMA_FN_GDN_REPLAY=0 (snapshot groups, v0.31.0) and =1 (one committed state per cell and a
// ring of token inputs). Both get the same calls and every output row is compared bit for bit, for n_rs_seq 0 and 3 and
// for one and two sequences:
//   - prefill ubatches of 1, 3, 30, 70 and 300 tokens per sequence (70 and 300 take the chunked prefill split, which runs
//     sequential kernels on the CPU);
//   - decode and verify steps with speculative rollbacks of 1, 2 and 3 tokens, and steps of unequal length;
//   - a rollback of a whole ubatch (the MoE bridge retry): taken only where the ring keeps the whole ubatch (GDN_NW_CAP=0
//     gives the bridge's ring); both layouts must accept or refuse it alike;
//   - state saves and restores in the middle: llama_state_seq_get_data / set_data (with a pending rollback), the
//     PARTIAL_ONLY flag of the server checkpoints and the ATTN_ONLY_EXT flag of the idle-slot park;
//   - seq_cp of a sequence and its continuation (two sequences);
//   - a snapshot-layout state blob loaded into the replay layout (continues bit for bit) and the other way (refused);
//     a replay-layout blob is refused at another n_rs_seq, a snapshot-layout blob loads there.
// It also prints the recurrent memory lines of both layouts. No GPU (run with CUDA_VISIBLE_DEVICES=-1).

#include "llama.h"
#include "llama-ext.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
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

int g_verbose = 0;

// the memory lines of the last context created
std::vector<std::string> g_mem_lines;

void log_cb(ggml_log_level level, const char * text, void * /*ud*/) {
    if (strstr(text, "llama_memory_recurrent") != nullptr &&
            (strstr(text, "size =") != nullptr || strstr(text, "RS buffer size") != nullptr || strstr(text, "GDN replay") != nullptr)) {
        g_mem_lines.push_back(text);
    }
    if (g_verbose > 0 || level == GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

int g_n_vocab = 0;

struct pair_ctx {
    llama_context * c[2] = { nullptr, nullptr }; // 0: LLAMA_FN_GDN_REPLAY=0, 1: =1
    std::vector<std::string> mem[2];
    llama_pos pos[2] = { 0, 0 };
    int n_rs  = 0;
    int n_seq = 1;
    size_t rows_cmp = 0; // output rows compared
};

llama_context * make_ctx(llama_model * model, const char * replay, int n_rs, int n_seq, int n_threads, std::vector<std::string> & mem) {
    set_env("LLAMA_FN_GDN_REPLAY", replay);
    g_mem_lines.clear();
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = 2048;
    cp.n_batch         = 1024;
    cp.n_ubatch        = 1024;
    cp.n_seq_max       = (uint32_t) n_seq;
    cp.kv_unified      = true;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.n_threads       = n_threads;
    cp.n_threads_batch = n_threads;
    cp.n_rs_seq        = (uint32_t) n_rs;
    llama_context * ctx = llama_init_from_model(model, cp);
    mem = g_mem_lines;
    set_env("LLAMA_FN_GDN_REPLAY", nullptr);
    return ctx;
}

llama_token tk(int s, llama_pos p, int salt) {
    return (llama_token) (1 + ((uint32_t) (s*37 + p*11 + (p*p) % 7 + salt*101)) % (uint32_t) (g_n_vocab - 1));
}

// one decode in both contexts: n[s] tokens of sequence s, rows with out_every (every k-th row and the last) as outputs;
// every output row compared bit for bit
bool step(pair_ctx & pc, const char * what, const std::vector<int> & n, int out_every, int salt) {
    int n_tok = 0;
    for (int x : n) {
        n_tok += x;
    }
    if (n_tok == 0) {
        return true;
    }
    llama_batch b = llama_batch_init(n_tok, 0, 1);
    std::vector<int> outs;
    int j = 0;
    for (int s = 0; s < (int) n.size(); ++s) {
        for (int i = 0; i < n[s]; ++i) {
            const llama_pos p = pc.pos[s] + i;
            b.token[j]     = tk(s, p, salt);
            b.pos[j]       = p;
            b.n_seq_id[j]  = 1;
            b.seq_id[j][0] = s;
            b.logits[j]    = (i + 1 == n[s] || (out_every > 0 && i % out_every == 0)) ? 1 : 0;
            if (b.logits[j]) {
                outs.push_back(j);
            }
            ++j;
        }
        pc.pos[s] += n[s];
    }
    b.n_tokens = n_tok;
    const int rc0 = llama_decode(pc.c[0], b);
    const int rc1 = llama_decode(pc.c[1], b);
    llama_batch_free(b);
    TCHECK(rc0 == 0 && rc1 == 0, "%s: decode rc %d / %d", what, rc0, rc1);
    if (rc0 != 0 || rc1 != 0) {
        return false;
    }
    size_t n_diff = 0;
    double max_abs = 0.0;
    for (int r : outs) {
        const float * l0 = llama_get_logits_ith(pc.c[0], r);
        const float * l1 = llama_get_logits_ith(pc.c[1], r);
        bool row_diff = false;
        for (int t = 0; t < g_n_vocab; ++t) {
            if (memcmp(&l0[t], &l1[t], sizeof(float)) != 0) {
                row_diff = true;
                max_abs = std::max(max_abs, (double) std::fabs(l0[t] - l1[t]));
            }
        }
        n_diff += row_diff ? 1 : 0;
    }
    pc.rows_cmp += outs.size();
    if (g_verbose > 0 || n_diff > 0) {
        printf("    %-34s %4zu rows, %zu differ (max |d| %.3g)\n", what, outs.size(), n_diff, max_abs);
    }
    TCHECK(n_diff == 0, "%s: %zu of %zu output rows differ, max |d| %.3g (n_rs_seq %d, %d seqs)", what, n_diff, outs.size(),
            max_abs, pc.n_rs, pc.n_seq);
    return n_diff == 0;
}

// a speculative rollback of r tokens of sequence s in both contexts; both must give the same answer. expect: -1 either,
// 0 refused, 1 taken. Returns whether it was taken.
bool rollback(pair_ctx & pc, int s, int r, int expect, const char * what) {
    if (r == 0) {
        return true;
    }
    const llama_pos p0 = pc.pos[s] - r;
    const bool ok0 = llama_memory_seq_rm(llama_get_memory(pc.c[0]), s, p0, -1);
    const bool ok1 = llama_memory_seq_rm(llama_get_memory(pc.c[1]), s, p0, -1);
    TCHECK(ok0 == ok1, "%s: rollback of %d on seq %d: snapshot %d, replay %d", what, r, s, (int) ok0, (int) ok1);
    if (expect >= 0) {
        TCHECK((int) ok0 == expect, "%s: rollback of %d on seq %d: %d, expected %d", what, r, s, (int) ok0, expect);
    }
    if (ok0 && ok1) {
        pc.pos[s] = p0;
    }
    return ok0 && ok1;
}

// save and restore sequence s in both contexts with the given flags
bool save_restore(pair_ctx & pc, int s, llama_state_seq_flags flags, const char * what, size_t * sizes) {
    for (int k = 0; k < 2; ++k) {
        llama_context * ctx = pc.c[k];
        const size_t n = llama_state_seq_get_size_ext(ctx, s, flags);
        std::vector<uint8_t> buf(n);
        const size_t got = llama_state_seq_get_data_ext(ctx, buf.data(), n, s, flags);
        TCHECK(n > 0 && got == n, "%s: save of seq %d (layout %d): %zu of %zu bytes", what, s, k, got, n);
        if (got != n) {
            return false;
        }
        sizes[k] = n;
        if (flags & LLAMA_STATE_SEQ_FLAGS_ATTN_ONLY_EXT) {
            const bool rm = llama_memory_attn_seq_rm_ext(ctx, s);
            TCHECK(rm, "%s: attention removal of seq %d (layout %d)", what, s, k);
        } else if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            llama_memory_seq_rm(llama_get_memory(ctx), s, -1, -1);
        }
        const size_t put = llama_state_seq_set_data_ext(ctx, buf.data(), n, s, flags);
        TCHECK(put == n, "%s: restore of seq %d (layout %d): %zu of %zu bytes", what, s, k, put, n);
        if (put != n) {
            return false;
        }
    }
    return true;
}

void print_mem(const pair_ctx & pc) {
    for (int k = 0; k < 2; ++k) {
        for (const auto & l : pc.mem[k]) {
            printf("    LLAMA_FN_GDN_REPLAY=%d | %s", k, l.c_str());
        }
    }
}

void run_case(llama_model * model, int n_rs, int n_seq, int n_threads, bool nw_full) {
    pair_ctx pc;
    pc.n_rs  = n_rs;
    pc.n_seq = n_seq;
    pc.c[0] = make_ctx(model, "0", n_rs, n_seq, n_threads, pc.mem[0]);
    pc.c[1] = make_ctx(model, "1", n_rs, n_seq, n_threads, pc.mem[1]);
    TCHECK(pc.c[0] && pc.c[1], "context creation (n_rs_seq %d, %d seqs)", n_rs, n_seq);
    if (!pc.c[0] || !pc.c[1]) {
        llama_free(pc.c[0]);
        llama_free(pc.c[1]);
        return;
    }
    const int n_rs_ctx = (int) llama_n_rs_seq(pc.c[1]);
    bool replay_on = false;
    for (const auto & l : pc.mem[1]) {
        replay_on = replay_on || l.find("GDN replay") != std::string::npos;
    }
    bool replay_off = true;
    for (const auto & l : pc.mem[0]) {
        replay_off = replay_off && l.find("GDN replay") == std::string::npos;
    }
    printf("  n_rs_seq %d, %d seq%s: replay layout %s in the =1 context, %s in the =0 context\n", n_rs_ctx, n_seq,
            n_seq > 1 ? "s" : "", replay_on ? "on" : "off", replay_off ? "off" : "ON");
    TCHECK(replay_off, "LLAMA_FN_GDN_REPLAY=0 built the replay layout");
    TCHECK(replay_on == (n_rs_ctx > 0), "LLAMA_FN_GDN_REPLAY=1: replay %d with n_rs_seq %d", (int) replay_on, n_rs_ctx);
    print_mem(pc);

    auto all = [&](int t) { return std::vector<int>((size_t) n_seq, t); };
    const int rb_ok = n_rs_ctx >= 3 ? 1 : 0; // speculative rollbacks of up to 3 tokens: taken with 3 ring tokens

    bool ok = true;
    // prefill ubatches (one decode each)
    int salt = 0;
    for (int T : { 1, 3, 30, 70, 300 }) {
        char what[64];
        snprintf(what, sizeof(what), "prefill %d", T);
        ok = ok && step(pc, what, all(T), 1, salt++); // every row an output
    }
    // verify steps with rollbacks of 1, 2, 3 tokens (seq 1 rolls back one token more, mod 4), decode steps between
    const int verify_rb[][2] = { { 3, 4 }, { 2, 4 }, { 1, 4 }, { 0, 1 }, { 0, 4 }, { 1, 2 }, { 2, 3 }, { 0, 1 } }; // { r, T }
    for (const auto & v : verify_rb) {
        const int r = v[0];
        const int T = v[1];
        char what[64];
        snprintf(what, sizeof(what), "verify T=%d", T);
        ok = ok && step(pc, what, all(T), 1, salt++);
        if (!ok || n_rs_ctx == 0) {
            continue;
        }
        for (int s = 0; s < n_seq; ++s) {
            const int rs = s == 0 ? r : (r + 1) % 4;
            if (rs < T) {
                rollback(pc, s, rs, rs <= std::min(T - 1, 3) ? rb_ok : -1, what);
            }
        }
    }
    // steps of unequal length in one batch (the split puts them in separate ubatches; an extra cell can move)
    if (ok && n_seq == 2) {
        ok = ok && step(pc, "unequal 4 + 2", { 4, 2 }, 1, salt++);
        if (ok && n_rs_ctx > 0) {
            rollback(pc, 0, 2, rb_ok, "unequal");
            rollback(pc, 1, 1, rb_ok, "unequal");
        }
        ok = ok && step(pc, "unequal 1 + 4", { 1, 4 }, 1, salt++);
        if (ok && n_rs_ctx > 0) {
            rollback(pc, 1, 3, rb_ok, "unequal");
        }
        ok = ok && step(pc, "after unequal", all(2), 1, salt++);
    }
    // a whole ubatch rolled back (the MoE bridge retry), then computed again: a 1-token decode always, a 3-token ubatch only
    // when the ring keeps every token of it (GDN_NW_CAP=0, the bridge's llama_rs_set_nw_full)
    if (ok && n_rs_ctx > 0) {
        for (int T : { 1, 3 }) {
            const llama_pos p_before = pc.pos[0];
            char what[64];
            snprintf(what, sizeof(what), "whole ubatch T=%d", T);
            ok = ok && step(pc, what, all(T), 1, salt);
            if (!ok) {
                break;
            }
            const int expect = T == 1 || nw_full ? 1 : 0;
            bool taken = true;
            for (int s = 0; s < n_seq; ++s) {
                taken = rollback(pc, s, T, expect, what) && taken;
            }
            if (taken) {
                TCHECK(pc.pos[0] == p_before, "whole ubatch: position %d after the rollback, %d before", pc.pos[0], p_before);
                ok = ok && step(pc, "whole ubatch again", all(T), 1, salt); // the same tokens
            } else {
                // the refused rollback changed nothing; drop the tokens the way the failure path does is not possible here,
                // so the sequences go on from where they are
                ok = ok && step(pc, "after a refused rollback", all(1), 1, salt + 1);
            }
            ++salt;
        }
    }
    // state saves and restores in the middle
    size_t sz[2] = { 0, 0 };
    if (ok) {
        ok = ok && step(pc, "before save", all(4), 1, salt++);
        if (ok && n_rs_ctx > 0) {
            rollback(pc, 0, 2, rb_ok, "pending before save"); // a pending rollback goes into the saved state
        }
        ok = ok && save_restore(pc, 0, LLAMA_STATE_SEQ_FLAGS_NONE, "full save", sz);
        printf("    full save of seq 0: %zu bytes (=0), %zu bytes (=1)\n", sz[0], sz[1]);
        ok = ok && step(pc, "after full restore", all(4), 1, salt++);
        if (ok && n_rs_ctx > 0) {
            rollback(pc, 0, 3, rb_ok, "after full restore");
        }
        ok = ok && step(pc, "decode after restore", all(1), 1, salt++);
    }
    if (ok) {
        ok = ok && save_restore(pc, n_seq - 1, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY, "partial save", sz);
        printf("    PARTIAL_ONLY save of seq %d: %zu bytes (=0), %zu bytes (=1)\n", n_seq - 1, sz[0], sz[1]);
        if (ok && n_rs_ctx > 0) {
            // a restored state has no conv groups for a rollback: both layouts refuse one
            rollback(pc, n_seq - 1, 1, 0, "right after partial restore");
        }
        ok = ok && step(pc, "after partial restore", all(4), 1, salt++);
        if (ok && n_rs_ctx > 0) {
            rollback(pc, n_seq - 1, 2, rb_ok, "after partial restore");
        }
        ok = ok && step(pc, "decode after partial", all(1), 1, salt++);
    }
    if (ok && llama_memory_attn_park_ok_ext(pc.c[0]) && llama_memory_attn_park_ok_ext(pc.c[1])) {
        ok = ok && save_restore(pc, 0, LLAMA_STATE_SEQ_FLAGS_ATTN_ONLY_EXT, "attn-only park", sz);
        printf("    ATTN_ONLY_EXT park of seq 0: %zu bytes (=0), %zu bytes (=1)\n", sz[0], sz[1]);
        ok = ok && step(pc, "after attn-only unpark", all(4), 1, salt++);
        if (ok && n_rs_ctx > 0) {
            rollback(pc, 0, 1, rb_ok, "after attn-only unpark");
        }
        ok = ok && step(pc, "decode after unpark", all(1), 1, salt++);
    }
    // seq_cp: sequence 1 becomes a copy of sequence 0, then both go on in turn and together
    if (ok && n_seq == 2) {
        for (int k = 0; k < 2; ++k) {
            llama_memory_t mem = llama_get_memory(pc.c[k]);
            llama_memory_seq_rm(mem, 1, -1, -1);
            llama_memory_seq_cp(mem, 0, 1, -1, -1);
        }
        pc.pos[1] = pc.pos[0];
        if (n_rs_ctx > 0) {
            rollback(pc, 1, 1, 0, "shared cell"); // [TAG_FN_L21_RS_SHARED_RM] a shared cell does not move back
        }
        ok = ok && step(pc, "seq 1 after seq_cp", { 0, 3 }, 1, salt++);
        ok = ok && step(pc, "seq 0 after seq_cp", { 3, 0 }, 1, salt++);
        if (ok && n_rs_ctx > 0) {
            rollback(pc, 0, 2, rb_ok, "after seq_cp");
            rollback(pc, 1, 1, rb_ok, "after seq_cp");
        }
        ok = ok && step(pc, "both after seq_cp", all(4), 1, salt++);
    }
    // blobs across the layouts (slot files, the conversation store): a snapshot-layout blob loads into the replay layout
    // as a committed state with an empty ring and continues bit for bit; a replay-layout blob is refused by the snapshot
    // layout (the caller processes the prompt again)
    if (ok && n_rs_ctx > 0) {
        ok = ok && step(pc, "before cross load", all(4), 1, salt++);
        if (ok) {
            rollback(pc, 0, 1, rb_ok, "pending before cross load");
        }
        std::vector<uint8_t> blob[2];
        for (int k = 0; k < 2 && ok; ++k) {
            blob[k].resize(llama_state_seq_get_size(pc.c[k], 0));
            ok = llama_state_seq_get_data(pc.c[k], blob[k].data(), blob[k].size(), 0) == blob[k].size();
            TCHECK(ok, "cross load: save of layout %d", k);
        }
        for (int k = 0; k < 2 && ok; ++k) {
            llama_memory_seq_rm(llama_get_memory(pc.c[k]), 0, -1, -1);
            const size_t put = llama_state_seq_set_data(pc.c[k], blob[0].data(), blob[0].size(), 0);
            TCHECK(put == blob[0].size(), "cross load: the snapshot-layout blob into layout %d: %zu of %zu bytes", k, put,
                    blob[0].size());
            ok = put == blob[0].size();
        }
        ok = ok && step(pc, "after snapshot blob in both", all(4), 1, salt++);
        if (ok) {
            rollback(pc, 0, 2, rb_ok, "after snapshot blob in both");
        }
        ok = ok && step(pc, "decode after snapshot blob", all(1), 1, salt++);
        if (ok) {
            const size_t put = llama_state_seq_set_data(pc.c[0], blob[1].data(), blob[1].size(), 0);
            printf("    replay-layout blob (%zu bytes) into the snapshot layout: %zu bytes read (0 = refused)\n", blob[1].size(), put);
            TCHECK(put == 0, "cross load: the snapshot layout read a replay-layout blob (%zu bytes)", put);
        }
        // a replay-layout blob loads only where the ring has the same n_rs_seq (state_read_data, "mismatched GDN replay
        // ring"); a snapshot-layout blob loads at any n_rs_seq. Checked on contexts with a ring of one token less
        if (ok) {
            std::vector<std::string> mem_x[2];
            llama_context * cx[2] = { make_ctx(model, "0", n_rs_ctx - 1, n_seq, n_threads, mem_x[0]),
                                      make_ctx(model, "1", n_rs_ctx - 1, n_seq, n_threads, mem_x[1]) };
            TCHECK(cx[0] && cx[1], "context creation (n_rs_seq %d, %d seqs)", n_rs_ctx - 1, n_seq);
            if (cx[0] && cx[1]) {
                const size_t put0 = llama_state_seq_set_data(cx[0], blob[0].data(), blob[0].size(), 0);
                const size_t put1 = llama_state_seq_set_data(cx[1], blob[1].data(), blob[1].size(), 0);
                printf("    blobs saved at n_rs_seq %d into n_rs_seq %d: snapshot layout %s, replay layout %s\n", n_rs_ctx,
                        n_rs_ctx - 1, put0 == blob[0].size() ? "loaded" : "refused", put1 == 0 ? "refused" : "loaded");
                TCHECK(put0 == blob[0].size(), "the snapshot-layout blob at n_rs_seq %d: %zu of %zu bytes", n_rs_ctx - 1,
                        put0, blob[0].size());
                TCHECK(put1 == 0, "the replay-layout blob at n_rs_seq %d: %zu bytes read", n_rs_ctx - 1, put1);
            }
            llama_free(cx[0]);
            llama_free(cx[1]);
        }
    }
    printf("  n_rs_seq %d, %d seq%s: %zu output rows compared, %s\n", n_rs_ctx, n_seq, n_seq > 1 ? "s" : "", pc.rows_cmp,
            ok ? "all bitwise equal" : "FAILED");

    llama_free(pc.c[0]);
    llama_free(pc.c[1]);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        printf("test-fn-gdn-replay: SKIP (no model file; make one with test-llama-archs -a qwen4exp -o <dir>)\n");
        return 0;
    }
    const char * path = argv[1];
    const int n_threads = argc > 2 ? std::max(1, atoi(argv[2])) : 4;
    g_verbose = getenv("GDN_REPLAY_TEST_VERBOSE") ? atoi(getenv("GDN_REPLAY_TEST_VERBOSE")) : 0;
    const char * cap = getenv("GDN_NW_CAP");
    const bool nw_full = cap && cap[0] == '0' && cap[1] == '\0'; // the ring of the MoE bridge (llama_rs_set_nw_full)
    // the test compares LLAMA_FN_GDN_REPLAY=0 against =1; GDN_REPLAY=0 from the caller would turn both into snapshot groups
    set_env("GDN_REPLAY", nullptr);
    llama_log_set(log_cb, nullptr);
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t no_dev[1] = { nullptr };
    mp.devices = no_dev; // CPU only
    llama_model * model = llama_model_load_from_file(path, mp);
    if (model == nullptr) {
        fprintf(stderr, "cannot load %s\n", path);
        return 1;
    }
    char arch[64] = {};
    llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
    if (strcmp(arch, "qwen4exp") != 0) {
        printf("test-fn-gdn-replay: SKIP (arch %s, not qwen4exp)\n", arch);
        llama_model_free(model);
        return 0;
    }
    g_n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    printf("test-fn-gdn-replay: %s, vocab %d, ring %s\n", path, g_n_vocab,
            nw_full ? "keeps whole ubatches (GDN_NW_CAP=0)" : "default (GDN_NW_CAP unset)");

    for (int n_rs : { 0, 3 }) {
        for (int n_seq : { 1, 2 }) {
            run_case(model, n_rs, n_seq, n_threads, nw_full);
        }
    }

    llama_model_free(model);
    llama_backend_free();
    printf("test-fn-gdn-replay: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
