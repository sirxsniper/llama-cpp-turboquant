#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama.h"

#include "../src/llama-io.h"
#include "../src/llama-memory.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <set>
#include <string>
#include <utility>
#include <vector>

static bool decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens, uint32_t count) {
    llama_batch batch = llama_batch_init(count, 0, 1);
    for (uint32_t pos = 0; pos < count; ++pos) {
        common_batch_add(batch, tokens[pos], pos, { 0 }, pos + 1 == count);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool decode_one(llama_context * ctx, llama_token tok, llama_pos pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, tok, pos, { 0 }, true);
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

struct cache_buffer_collector : llama_io_write_i {
    std::set<ggml_backend_buffer_t> buffers;
    size_t size = 0;

    void write(const void *, size_t n) override {
        size += n;
    }

    void write_tensor(ggml_tensor * tensor, size_t, size_t n) override {
        buffers.insert(tensor->buffer);
        size += n;
    }

    size_t n_bytes() override {
        return size;
    }
};

static llama_context * init_ctx(llama_model * model, llama_context_params cparams, uint8_t fill) {
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr || fill == 0) {
        return ctx;
    }

    // Use a full ubatch so buffer discovery preserves prefill allocation sizes.
    const uint32_t n_tokens = llama_n_ubatch(ctx);
    if (!decode_tokens(ctx, std::vector<llama_token>(n_tokens, 0), n_tokens)) {
        llama_free(ctx);
        return nullptr;
    }
    llama_synchronize(ctx);
    cache_buffer_collector collector;
    llama_get_memory(ctx)->state_write(collector);
    llama_memory_clear(llama_get_memory(ctx), true);
    if (collector.buffers.empty()) {
        fprintf(stderr, "%s : no cache buffers found\n", __func__);
        llama_free(ctx);
        return nullptr;
    }
    for (auto * buffer : collector.buffers) {
        ggml_backend_buffer_clear(buffer, fill);
    }
    return ctx;
}

static llama_context * make_ctx(const common_params & params, llama_model * model, uint8_t fill) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    cparams.n_rs_seq  = 8;
    cparams.n_batch   = std::max(cparams.n_batch,  (uint32_t) (cparams.n_rs_seq + 1));
    cparams.n_ubatch  = std::max(cparams.n_ubatch, (uint32_t) (cparams.n_rs_seq + 1));
    return init_ctx(model, cparams, fill);
}

static float logit_diff(float a, float b) {
    return std::isfinite(a) && std::isfinite(b) ? std::fabs(a - b) : std::numeric_limits<float>::infinity();
}

static double nmse(const float * a, const float * b, int n) {
    double mse_ab = 0.0;
    double mse_a0 = 0.0;
    for (int i = 0; i < n; i++) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            return std::numeric_limits<double>::infinity();
        }
        const double diff = (double) a[i] - b[i];
        mse_ab += diff*diff;
        mse_a0 += (double) a[i]*a[i];
    }
    return mse_a0 == 0.0 ? (mse_ab == 0.0 ? 0.0 : std::numeric_limits<double>::infinity()) : mse_ab/mse_a0;
}

// Roll back multiple sequences, then replay them in a single batch whose
// per-seq token count exceeds n_ubatch: each seq's replay spans several
// ubatches while its rollback restore is still pending. Compared against a
// reference context that never advanced past the rollback point and decodes
// the identical replay batch.
static bool test_multi_seq_split_replay(const common_params & params, llama_model * model, const int n_vocab, uint8_t fill) {
    constexpr uint32_t  n_seqs     = 2;
    constexpr uint32_t  n_ubatch   = 16;
    constexpr uint32_t  n_prompt   = 19;
    constexpr uint32_t  n_rollback = 3;
    constexpr uint32_t  n_replay   = 40; // > n_ubatch so each seq spans multiple ubatches
    constexpr llama_pos p0         = n_prompt - n_rollback;

    const auto make_ctx_multi = [&]() {
        auto cparams = common_context_params_to_llama(params);
        cparams.n_seq_max  = n_seqs;
        cparams.n_rs_seq   = 8;
        cparams.n_ctx      = 256;
        cparams.n_batch    = 256;
        cparams.n_ubatch   = n_ubatch;
        cparams.kv_unified = false;
        return init_ctx(model, cparams, fill);
    };

    llama_context * ctx_roll = make_ctx_multi();
    llama_context * ctx_ref  = make_ctx_multi();
    if (ctx_roll == nullptr || ctx_ref == nullptr) {
        fprintf(stderr, "%s : failed to init multi-seq contexts\n", __func__);
        return false;
    }

    const auto cleanup = [&]() {
        llama_free(ctx_roll);
        llama_free(ctx_ref);
    };

    if (llama_n_rs_seq(ctx_roll) < n_rollback) {
        fprintf(stderr, "%s : skipping because n_rs_seq is too small\n", __func__);
        cleanup();
        return true;
    }

    const auto tok = [&](uint32_t seq, llama_pos pos) {
        return (llama_token) ((7*(uint32_t) pos + 31*seq + 1) % (uint32_t) n_vocab);
    };

    bool ok = true;

    // both contexts decode the identical [0, p0) prefill; only ctx_roll decodes
    // the tail, which is then rolled back so its restore is pending at replay
    for (uint32_t s = 0; s < n_seqs && ok; ++s) {
        llama_batch batch = llama_batch_init(n_prompt, 0, 1);
        for (llama_pos pos = 0; pos < (llama_pos) p0; ++pos) {
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, false);
        }
        ok = ok && llama_decode(ctx_roll, batch) == 0;
        ok = ok && llama_decode(ctx_ref,  batch) == 0;

        common_batch_clear(batch);
        for (llama_pos pos = p0; pos < (llama_pos) n_prompt; ++pos) {
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, false);
        }
        ok = ok && llama_decode(ctx_roll, batch) == 0;
        llama_batch_free(batch);

        ok = ok && llama_memory_seq_rm(llama_get_memory(ctx_roll), (llama_seq_id) s, p0, -1);

        // a second partial removal while one is pending must be refused
        ok = ok && !llama_memory_seq_rm(llama_get_memory(ctx_roll), (llama_seq_id) s, p0 - 1, -1);
    }
    if (!ok) {
        fprintf(stderr, "%s : multi-seq prefill/rollback failed\n", __func__);
        cleanup();
        return false;
    }

    llama_batch batch = llama_batch_init(n_seqs*n_replay, 0, 1);
    for (uint32_t s = 0; s < n_seqs; ++s) {
        for (uint32_t i = 0; i < n_replay; ++i) {
            const llama_pos pos = p0 + (llama_pos) i;
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, true);
        }
    }
    ok = llama_decode(ctx_roll, batch) == 0;
    ok = ok && llama_decode(ctx_ref, batch) == 0;
    llama_batch_free(batch);
    if (!ok) {
        fprintf(stderr, "%s : multi-seq replay decode failed\n", __func__);
        cleanup();
        return false;
    }

    // identical ubatch shapes should produce identical states, but the larger
    // stdev makes the model sensitive to backend scheduling/rounding noise
    constexpr float nmse_eps = 1e-5f;

    float    diff_max  = 0.0f;
    uint32_t seq_first = 0;
    int32_t  pos_first = -1;
    double   nmse_ab   = 0.0;
    double   nmse_a0   = 0.0;
    for (uint32_t i = 0; i < n_seqs*n_replay; ++i) {
        const float * l_roll = llama_get_logits_ith(ctx_roll, i);
        const float * l_ref  = llama_get_logits_ith(ctx_ref,  i);
        if (l_roll == nullptr || l_ref == nullptr) {
            fprintf(stderr, "%s : missing multi-seq logits at index %u\n", __func__, i);
            cleanup();
            return false;
        }
        for (int t = 0; t < n_vocab; ++t) {
            const float r = l_roll[t];
            const float f = l_ref[t];
            const float diff = logit_diff(r, f);
            if (diff > 0.0f && pos_first < 0) {
                seq_first = i/n_replay;
                pos_first = p0 + (int32_t) (i%n_replay);
            }
            diff_max = std::max(diff_max, diff);
            if (std::isfinite(r) && std::isfinite(f)) {
                const double d = (double) r - f;
                nmse_ab += d*d;
                nmse_a0 += (double) r*r;
            } else {
                nmse_ab = std::numeric_limits<double>::infinity();
                nmse_a0 = 1.0;
            }
        }
    }
    const double nmse_val = nmse_a0 == 0.0 ? (nmse_ab == 0.0 ? 0.0 : std::numeric_limits<double>::infinity()) : nmse_ab/nmse_a0;

    if (nmse_val > nmse_eps) {
        fprintf(stderr, "%s : multi-seq split replay logits mismatch (max diff %g, nmse %g, first at seq %u pos %d)\n",
                __func__, (double) diff_max, nmse_val, seq_first, pos_first);
        cleanup();
        return false;
    }

    fprintf(stderr, "%s : multi-seq split replay matched (max diff %g, nmse %g)\n", __func__, (double) diff_max, nmse_val);

    // seq-1-only decodes must be independent of seq 0's content: diverge seq 0
    // in ctx_ref only, then compare identical seq-1-only continuations bitwise
    constexpr uint32_t n_tail = 4;

    {
        llama_batch batch_tail = llama_batch_init(n_tail, 0, 1);
        for (uint32_t i = 0; i < n_tail; ++i) {
            const llama_pos pos = p0 + (llama_pos) (n_replay + i);
            common_batch_add(batch_tail, tok(0, pos + 7), pos, { 0 }, false);
        }
        ok = llama_decode(ctx_ref, batch_tail) == 0;
        llama_batch_free(batch_tail);
    }

    float diff_tail = 0.0f;
    double nmse_tail_ab = 0.0;
    double nmse_tail_a0 = 0.0;
    for (uint32_t i = 0; i < n_tail && ok; ++i) {
        const llama_pos pos = p0 + (llama_pos) (n_replay + i);
        llama_batch batch_one = llama_batch_init(1, 0, 1);
        common_batch_add(batch_one, tok(1, pos), pos, { 1 }, true);
        ok = llama_decode(ctx_roll, batch_one) == 0;
        ok = ok && llama_decode(ctx_ref, batch_one) == 0;
        llama_batch_free(batch_one);
        if (!ok) {
            break;
        }

        const float * l_roll = llama_get_logits_ith(ctx_roll, 0);
        const float * l_ref  = llama_get_logits_ith(ctx_ref,  0);
        ok = l_roll != nullptr && l_ref != nullptr;
        for (int t = 0; ok && t < n_vocab; ++t) {
            const float r = l_roll[t];
            const float f = l_ref[t];
            diff_tail = std::max(diff_tail, logit_diff(r, f));
            if (std::isfinite(r) && std::isfinite(f)) {
                const double d = (double) r - f;
                nmse_tail_ab += d*d;
                nmse_tail_a0 += (double) r*r;
            } else {
                nmse_tail_ab = std::numeric_limits<double>::infinity();
                nmse_tail_a0 = 1.0;
            }
        }
    }
    const double nmse_tail = nmse_tail_a0 == 0.0 ? (nmse_tail_ab == 0.0 ? 0.0 : std::numeric_limits<double>::infinity()) : nmse_tail_ab/nmse_tail_a0;

    if (!ok || nmse_tail > nmse_eps) {
        fprintf(stderr, "%s : seq-1-only decode leaked seq 0 state (ok=%d, max diff %g, nmse %g)\n",
                __func__, ok ? 1 : 0, (double) diff_tail, nmse_tail);
        cleanup();
        return false;
    }

    fprintf(stderr, "%s : seq-1-only decode independent of seq 0 (max diff %g, nmse %g)\n", __func__, (double) diff_tail, nmse_tail);
    cleanup();
    return true;
}

static int test_rollback(const common_params & params, llama_model * model, uint8_t fill) {
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    // TODO: use smart pointers
    llama_context * ctx_src = make_ctx(params, model, fill);
    llama_context * ctx_dst = make_ctx(params, model, fill);
    if (ctx_src == nullptr || ctx_dst == nullptr) {
        fprintf(stderr, "%s : failed to init contexts\n", __func__);
        return 1;
    }

    if (llama_n_rs_seq(ctx_src) == 0) {
        fprintf(stderr, "%s : skipping because n_rs_seq is disabled\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return 0;
    }

    std::vector<llama_token> tokens;
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        tokens = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    } else {
        tokens = common_tokenize(ctx_src, "The quick brown fox jumps over the lazy dog", true);
    }
    const uint32_t n_rs_seq = llama_n_rs_seq(ctx_src);
    constexpr uint32_t n_rollback = 3;
    if (n_rs_seq < n_rollback) {
        fprintf(stderr, "%s : skipping because n_rs_seq is too small\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return 0;
    }
    if (tokens.empty()) {
        fprintf(stderr, "%s : not enough prompt tokens\n", __func__);
        return 1;
    }
    tokens.resize(n_rs_seq + 1, tokens.back());

    const uint32_t  n_tokens     = tokens.size();
    const llama_pos rollback_pos = (llama_pos) n_tokens - n_rollback;

    // Decode the full prompt on the source, then roll back three positions.
    // Replaying them crosses DSV4's ratio-4 compressor boundary.
    // Rollback leaves the recurrent memory in a snapshot state (rs_idx != 0).
    if (!decode_tokens(ctx_src, tokens, n_tokens)) {
        fprintf(stderr, "%s : failed to decode prompt\n", __func__);
        return 1;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_src), 0, rollback_pos, -1)) {
        fprintf(stderr, "%s : rollback failed\n", __func__);
        return 1;
    }

    // Save the rolled-back state and restore it into a fresh context.
    common_prompt_checkpoint ckpt;
    ckpt.update_tgt(ctx_src, 0, 0);
    ckpt.load_tgt(ctx_dst, 0, 0);

    constexpr float nmse_eps = 0.0;
    std::vector<std::vector<float>> logits_src_replay(n_rollback);
    const auto replay_and_compare = [&](const char * mode) {
        for (uint32_t i = 0; i < n_rollback; ++i) {
            const llama_pos pos = rollback_pos + i;
            if (!decode_one(ctx_src, tokens[pos], pos) ||
                !decode_one(ctx_dst, tokens[pos], pos)) {
                fprintf(stderr, "%s : %s replay failed at position %d\n", __func__, mode, pos);
                return false;
            }

            const float * logits_src = llama_get_logits_ith(ctx_src, 0);
            const float * logits_dst = llama_get_logits_ith(ctx_dst, 0);
            if (logits_src == nullptr || logits_dst == nullptr) {
                fprintf(stderr, "%s : missing %s logits at position %d\n", __func__, mode, pos);
                return false;
            }

            logits_src_replay[i].assign(logits_src, logits_src + n_vocab);
            const double nmse_val = nmse(logits_src, logits_dst, n_vocab);
            int token_first = -1;
            for (int token = 0; token < n_vocab; ++token) {
                if (logit_diff(logits_src[token], logits_dst[token]) > 0.0f && token_first < 0) {
                    token_first = token;
                }
            }
            if (nmse_val > nmse_eps) {
                fprintf(stderr, "%s : %s logits mismatch at position %d, first token %d, nmse %g\n",
                        __func__, mode, pos, token_first, nmse_val);
                return false;
            }
        }
        return true;
    };
    if (!replay_and_compare("full")) {
        return 1;
    }

    // TODO: this test is invalid because RS rollback is only correct once after a ubatch with more than n_rs_seq tokens
    //       this is not the case here. add asserts and guardrails to prevent such attempts
    //if (!llama_memory_seq_rm(llama_get_memory(ctx_src), 0, rollback_pos, -1) ||
    //    !llama_memory_seq_rm(llama_get_memory(ctx_dst), 0, rollback_pos, -1)) {
    //    fprintf(stderr, "%s : partial rollback failed\n", __func__);
    //    return 1;
    //}

    //constexpr llama_state_seq_flags partial_flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    //common_prompt_checkpoint ckpt_partial;
    //ckpt_partial.update_tgt(ctx_src, 0, partial_flags);
    //ckpt_partial.load_tgt(ctx_dst, 0, partial_flags);

    //if (!replay_and_compare("partial")) {
    //    return 1;
    //}

    // Repeat the load into a context that already has its own rollback state:
    // groups 1..n_rs_seq hold a different prompt's history, and rs_idx[0] is
    // non-zero at load time. The restore must wipe that state and still match.
    llama_context * ctx_dirty = make_ctx(params, model, fill);
    if (ctx_dirty == nullptr) {
        fprintf(stderr, "%s : failed to init dirty ctx\n", __func__);
        return 1;
    }

    std::vector<llama_token> noise = tokens;
    for (auto & t : noise) {
        t = (t + 1) % n_vocab;
        if (t < 0) {
            t = 0;
        }
    }
    if (!decode_tokens(ctx_dirty, noise, n_tokens)) {
        fprintf(stderr, "%s : dirty prompt decode failed\n", __func__);
        return 1;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_dirty), 0, rollback_pos, -1)) {
        fprintf(stderr, "%s : dirty rollback failed\n", __func__);
        return 1;
    }

    ckpt.load_tgt(ctx_dirty, 0, 0);

    for (uint32_t i = 0; i < n_rollback; ++i) {
        const llama_pos pos = rollback_pos + i;
        if (!decode_one(ctx_dirty, tokens[pos], pos)) {
            fprintf(stderr, "%s : dirty replay failed at position %d\n", __func__, pos);
            return 1;
        }

        const float * logits_dirty = llama_get_logits_ith(ctx_dirty, 0);
        if (logits_dirty == nullptr) {
            fprintf(stderr, "%s : missing dirty logits at position %d\n", __func__, pos);
            return 1;
        }

        const double nmse_dirty = nmse(logits_src_replay[i].data(), logits_dirty, n_vocab);
        int token_first = -1;
        for (int token = 0; token < n_vocab; ++token) {
            if (logit_diff(logits_src_replay[i][token], logits_dirty[token]) > 0.0f && token_first < 0) {
                token_first = token;
            }
        }
        if (nmse_dirty > nmse_eps) {
            fprintf(stderr, "%s : dirty-ctx logits mismatch at position %d, first token %d, nmse %g\n",
                    __func__, pos, token_first, nmse_dirty);
            return 1;
        }
    }

    fprintf(stderr, "%s : recurrent rollback checkpoint restored successfully\n", __func__);
    llama_free(ctx_src);
    llama_free(ctx_dst);
    llama_free(ctx_dirty);

    if (!test_multi_seq_split_replay(params, model, n_vocab, fill)) {
        return 1;
    }

    return 0;
}

// [TAG_XSEQ_PLANES] A sequence verified in an earlier ubatch of a batch and then moved as an "extra" cell by a later
// multi-seq ubatch of the same batch must keep its rollback snapshots.
//   prefill [B x8][A x8][C x8]           -> one ubatch, recurrent rows [B, A, C]
//   verify  [A x4][B x6][C x6]           -> split_equal (n_keep_tail 4) emits [A x4] alone (B, C would keep 2 < 4),
//                                           then [B x6, C x6], whose gather moves A from row 1 to row 2
//   seq_rm(A, 9, -1)                     -> rollback 3, pending on the snapshot group 3 of A's (new) row
//   replay  [A x4]                       -> must match a context where A only ever decoded the accepted token
// Before the fix the snapshot groups stayed in row 1 (then overwritten by C) and the replay read C's old snapshot.
static bool test_displaced_extra_rollback(const common_params & params, llama_model * model, uint8_t fill) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    constexpr uint32_t     n_rs_seq = 3;
    constexpr uint32_t     n_prompt = 8;
    constexpr uint32_t     n_verify = n_rs_seq + 1; // sampled token + 3 drafts
    constexpr uint32_t     n_other  = 6;            // 2 left after the first 4 -> A is emitted alone
    constexpr llama_seq_id seq_a    = 0;
    constexpr llama_seq_id seq_b    = 1;
    constexpr llama_seq_id seq_c    = 2;

    const auto make_ctx_xseq = [&]() {
        auto cparams = common_context_params_to_llama(params);
        cparams.n_seq_max  = 3;
        cparams.n_rs_seq   = n_rs_seq;
        cparams.n_ctx      = 256;
        cparams.n_batch    = 256;
        cparams.n_ubatch   = 64;
        cparams.kv_unified = true;
        return init_ctx(model, cparams, fill);
    };

    llama_context * ctx_roll = make_ctx_xseq();
    llama_context * ctx_ref  = make_ctx_xseq();
    if (ctx_roll == nullptr || ctx_ref == nullptr) {
        fprintf(stderr, "%s : failed to init contexts\n", __func__);
        llama_free(ctx_roll);
        llama_free(ctx_ref);
        return false;
    }

    const auto cleanup = [&]() {
        llama_free(ctx_roll);
        llama_free(ctx_ref);
    };

    if (llama_n_rs_seq(ctx_roll) < n_rs_seq) {
        fprintf(stderr, "%s : skipping because n_rs_seq is too small\n", __func__);
        cleanup();
        return true;
    }

    const auto tok = [&](llama_seq_id seq, llama_pos pos) {
        return (llama_token) ((7*(uint32_t) pos + 31*(uint32_t) seq + 1) % (uint32_t) n_vocab);
    };
    // rejected drafts: tokens that differ from the replay tokens
    const auto tok_bad = [&](llama_seq_id seq, llama_pos pos) {
        return (llama_token) ((13*(uint32_t) pos + 5*(uint32_t) seq + 3) % (uint32_t) n_vocab);
    };

    bool ok = true;

    // prefill, B first so that A ends up between B and C
    {
        llama_batch batch = llama_batch_init(3*n_prompt, 0, 1);
        for (llama_seq_id s : { seq_b, seq_a, seq_c }) {
            for (llama_pos pos = 0; pos < (llama_pos) n_prompt; ++pos) {
                common_batch_add(batch, tok(s, pos), pos, { s }, false);
            }
        }
        ok = ok && llama_decode(ctx_roll, batch) == 0;
        ok = ok && llama_decode(ctx_ref,  batch) == 0;
        llama_batch_free(batch);
    }

    // verify step: ctx_roll decodes A's sampled token plus 3 drafts, ctx_ref only the sampled token
    const llama_pos p_a = n_prompt;
    for (int which = 0; which < 2 && ok; ++which) {
        llama_context * ctx = which == 0 ? ctx_roll : ctx_ref;
        const uint32_t n_a = which == 0 ? n_verify : 1;

        llama_batch batch = llama_batch_init(n_verify + 2*n_other, 0, 1);
        for (uint32_t i = 0; i < n_a; ++i) {
            const llama_pos pos = p_a + (llama_pos) i;
            common_batch_add(batch, i == 0 ? tok(seq_a, pos) : tok_bad(seq_a, pos), pos, { seq_a }, true);
        }
        for (llama_seq_id s : { seq_b, seq_c }) {
            for (uint32_t i = 0; i < n_other; ++i) {
                const llama_pos pos = n_prompt + (llama_pos) i;
                common_batch_add(batch, tok(s, pos), pos, { s }, i + 1 == n_other);
            }
        }
        ok = llama_decode(ctx, batch) == 0;
        llama_batch_free(batch);
    }

    // reject all 3 drafts of A
    ok = ok && llama_memory_seq_rm(llama_get_memory(ctx_roll), seq_a, p_a + 1, -1);

    if (!ok) {
        fprintf(stderr, "%s : prefill/verify/rollback failed\n", __func__);
        cleanup();
        return false;
    }

    // replay A from the accepted position in both contexts
    constexpr uint32_t n_replay = 4;
    {
        llama_batch batch = llama_batch_init(n_replay, 0, 1);
        for (uint32_t i = 0; i < n_replay; ++i) {
            const llama_pos pos = p_a + 1 + (llama_pos) i;
            common_batch_add(batch, tok(seq_a, pos), pos, { seq_a }, true);
        }
        ok = llama_decode(ctx_roll, batch) == 0;
        ok = ok && llama_decode(ctx_ref, batch) == 0;
        llama_batch_free(batch);
    }
    if (!ok) {
        fprintf(stderr, "%s : replay decode failed\n", __func__);
        cleanup();
        return false;
    }

    // the verify step used different ubatch shapes in the two contexts (A x4 alone vs A x1 in a 3-seq ubatch), so
    // allow rounding noise; a foreign recurrent state gives an nmse of order 1e-1
    constexpr double nmse_eps = 1e-4;

    double nmse_max = 0.0;
    for (uint32_t i = 0; i < n_replay; ++i) {
        const float * l_roll = llama_get_logits_ith(ctx_roll, i);
        const float * l_ref  = llama_get_logits_ith(ctx_ref,  i);
        if (l_roll == nullptr || l_ref == nullptr) {
            fprintf(stderr, "%s : missing logits at replay index %u\n", __func__, i);
            cleanup();
            return false;
        }
        nmse_max = std::max(nmse_max, nmse(l_ref, l_roll, n_vocab));
    }

    if (!(nmse_max <= nmse_eps)) {
        fprintf(stderr, "%s : rollback of a sequence moved as an extra cell read a foreign snapshot (nmse %g)\n",
                __func__, nmse_max);
        cleanup();
        return false;
    }

    fprintf(stderr, "%s : moved extra cell kept its rollback snapshots (nmse %g)\n", __func__, nmse_max);
    cleanup();
    return true;
}

// [TAG_4C_GDN_REPLAY] GDN_REPLAY is read when a context creates its recurrent memory
static void set_env_gdn_replay(const char * value) {
#ifdef _WIN32
    _putenv_s("GDN_REPLAY", value == nullptr ? "" : value);
#else
    if (value == nullptr) {
        unsetenv("GDN_REPLAY");
    } else {
        setenv("GDN_REPLAY", value, 1);
    }
#endif
}

// [TAG_4C_GDN_REPLAY] The replay layout (one committed state and a ring of token inputs) must give the logits of the
// snapshot layout (GDN_REPLAY=0) bit for bit: a prefill, verify steps on two sequences with rollbacks of 0..3 tokens, a
// single-token decode, and a save/restore of a sequence with a pending rollback. Models without the replay layout
// compare two identical contexts.
static bool test_replay_vs_snapshot(const common_params & params, llama_model * model, uint8_t fill) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    // only these archs have the replay layout; elsewhere both contexts would be the same
    char arch[64] = {};
    llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
    if (strcmp(arch, "qwen35") != 0 && strcmp(arch, "qwen35moe") != 0) {
        fprintf(stderr, "%s : skipping for arch %s\n", __func__, arch);
        return true;
    }

    constexpr uint32_t n_rs_seq = 3;

    const char * env_old = getenv("GDN_REPLAY");
    const std::string env_old_s = env_old ? env_old : "";

    const auto make = [&](const char * replay) {
        set_env_gdn_replay(replay);
        auto cparams = common_context_params_to_llama(params);
        cparams.n_seq_max  = 2;
        cparams.n_rs_seq   = n_rs_seq;
        cparams.n_ctx      = 256;
        cparams.n_batch    = 256;
        cparams.n_ubatch   = 64;
        cparams.kv_unified = true;
        return init_ctx(model, cparams, fill);
    };

    llama_context * ctx_snap = make("0");
    llama_context * ctx_rpl  = make("1");
    set_env_gdn_replay(env_old ? env_old_s.c_str() : nullptr);

    const auto cleanup = [&]() {
        llama_free(ctx_snap);
        llama_free(ctx_rpl);
    };

    if (ctx_snap == nullptr || ctx_rpl == nullptr) {
        fprintf(stderr, "%s : failed to init contexts\n", __func__);
        cleanup();
        return false;
    }

    if (llama_n_rs_seq(ctx_snap) < n_rs_seq) {
        fprintf(stderr, "%s : skipping because n_rs_seq is too small\n", __func__);
        cleanup();
        return true;
    }

    const auto tok = [&](llama_seq_id seq, llama_pos pos) {
        return (llama_token) ((7*(uint32_t) pos + 31*(uint32_t) seq + 1) % (uint32_t) n_vocab);
    };
    const auto tok_bad = [&](llama_seq_id seq, llama_pos pos) {
        return (llama_token) ((13*(uint32_t) pos + 5*(uint32_t) seq + 3) % (uint32_t) n_vocab);
    };

    llama_pos pos_next[2] = { 0, 0 };

    // decode the same rows in both contexts and compare every output row bitwise; n_bad trailing rows are rejected drafts
    const auto step = [&](const char * what, std::vector<std::pair<llama_seq_id, std::pair<uint32_t, uint32_t>>> rows) {
        uint32_t n_tokens = 0;
        for (const auto & r : rows) {
            n_tokens += r.second.first;
        }
        llama_batch batch = llama_batch_init(n_tokens, 0, 1);
        for (const auto & r : rows) {
            const llama_seq_id seq   = r.first;
            const uint32_t     n     = r.second.first;
            const uint32_t     n_bad = r.second.second;
            for (uint32_t i = 0; i < n; ++i) {
                const llama_pos pos = pos_next[seq] + (llama_pos) i;
                common_batch_add(batch, i + n_bad >= n && i > 0 ? tok_bad(seq, pos) : tok(seq, pos), pos, { seq }, true);
            }
            pos_next[seq] += (llama_pos) n;
        }
        bool ok = llama_decode(ctx_snap, batch) == 0;
        ok = ok && llama_decode(ctx_rpl, batch) == 0;
        llama_batch_free(batch);
        if (!ok) {
            fprintf(stderr, "%s : %s: decode failed\n", __func__, what);
            return false;
        }
        for (uint32_t i = 0; i < n_tokens; ++i) {
            const float * l_snap = llama_get_logits_ith(ctx_snap, i);
            const float * l_rpl  = llama_get_logits_ith(ctx_rpl,  i);
            if (l_snap == nullptr || l_rpl == nullptr) {
                fprintf(stderr, "%s : %s: missing logits at row %u\n", __func__, what, i);
                return false;
            }
            for (int t = 0; t < n_vocab; ++t) {
                if (logit_diff(l_snap[t], l_rpl[t]) > 0.0f) {
                    fprintf(stderr, "%s : %s: logits differ at row %u token %d (%g vs %g), nmse %g\n", __func__, what, i, t,
                            (double) l_snap[t], (double) l_rpl[t], nmse(l_snap, l_rpl, n_vocab));
                    return false;
                }
            }
        }
        return true;
    };

    const auto rollback = [&](llama_seq_id seq, llama_pos n) {
        if (n == 0) {
            return true;
        }
        pos_next[seq] -= n;
        const bool ok_snap = llama_memory_seq_rm(llama_get_memory(ctx_snap), seq, pos_next[seq], -1);
        const bool ok_rpl  = llama_memory_seq_rm(llama_get_memory(ctx_rpl),  seq, pos_next[seq], -1);
        if (!ok_snap || !ok_rpl) {
            fprintf(stderr, "%s : rollback of %d on seq %d failed (snapshot %d, replay %d)\n", __func__, n, seq, ok_snap, ok_rpl);
            return false;
        }
        return true;
    };

    const auto save_restore = [&](llama_seq_id seq) {
        for (llama_context * ctx : { ctx_snap, ctx_rpl }) {
            std::vector<uint8_t> buf(llama_state_seq_get_size(ctx, seq));
            if (llama_state_seq_get_data(ctx, buf.data(), buf.size(), seq) != buf.size()) {
                fprintf(stderr, "%s : save of seq %d failed\n", __func__, seq);
                return false;
            }
            llama_memory_seq_rm(llama_get_memory(ctx), seq, -1, -1);
            if (llama_state_seq_set_data(ctx, buf.data(), buf.size(), seq) != buf.size()) {
                fprintf(stderr, "%s : restore of seq %d failed\n", __func__, seq);
                return false;
            }
        }
        return true;
    };

    bool ok = true;
    ok = ok && step("prefill",  { { 0, { 10, 0 } }, { 1, {  7, 0 } } });
    ok = ok && step("verify 1", { { 0, {  4, 3 } }, { 1, {  4, 2 } } });
    ok = ok && rollback(0, 3) && rollback(1, 2);
    ok = ok && step("verify 2", { { 0, {  4, 1 } }, { 1, {  4, 0 } } });
    ok = ok && rollback(0, 1);
    ok = ok && step("decode",   { { 0, {  1, 0 } } });
    ok = ok && step("verify 3", { { 1, {  4, 3 } } });
    ok = ok && rollback(1, 3);
    ok = ok && save_restore(1);
    ok = ok && step("verify 4", { { 0, {  4, 2 } }, { 1, {  4, 0 } } });
    ok = ok && rollback(0, 2);
    ok = ok && step("decode 2", { { 0, {  1, 0 } }, { 1, {  1, 0 } } });

    if (ok) {
        fprintf(stderr, "%s : replay and snapshot layouts gave identical logits\n", __func__);
    }
    cleanup();
    return ok;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict = 1;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to init model\n", __func__);
        return 1;
    }

    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        fprintf(stderr, "%s : skipping for non-recurrent model\n", __func__);
        return 0;
    }

    for (uint8_t fill : { 0, 0x3e }) {
        fprintf(stderr, "%s : testing with cache fill 0x%02x\n", __func__, fill);
        if (test_rollback(params, model, fill) != 0) {
            return 1;
        }
        if (!test_displaced_extra_rollback(params, model, fill)) {
            return 1;
        }
        if (!test_replay_vs_snapshot(params, model, fill)) { // [TAG_4C_GDN_REPLAY]
            return 1;
        }
    }

    return 0;
}
