#include "speculative.h"

#include <cstdlib>

#include "common.h"
#include "ggml.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "log.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "sampling.h"

#include "../src/llama-ext.h" // staging API: llama_set_embeddings_nextn / llama_get_embeddings_nextn_ith (used by MTP)

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <map>
#include <mutex>
#include <random>

#define SPC_DBG(fmt, ...) LOG_DBG("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_TRC(fmt, ...) LOG_TRC("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_INF(fmt, ...) LOG_INF("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_WRN(fmt, ...) LOG_WRN("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_ERR(fmt, ...) LOG_ERR("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_CNT(fmt, ...) LOG_CNT(""              fmt,               __VA_ARGS__)

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

const std::map<std::string, common_speculative_type> common_speculative_type_from_name_map = {
    {"none",          COMMON_SPECULATIVE_TYPE_NONE},
    {"draft-simple",  COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE},
    {"draft-eagle3",  COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3},
    {"draft-mtp",     COMMON_SPECULATIVE_TYPE_DRAFT_MTP},
    {"draft-dflash",  COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH},
    {"draft-dspark",  COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK},
    {"ngram-simple",  COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE},
    {"ngram-map-k",   COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K},
    {"ngram-map-k4v", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V},
    {"ngram-mod",     COMMON_SPECULATIVE_TYPE_NGRAM_MOD},
    {"ngram-cache",   COMMON_SPECULATIVE_TYPE_NGRAM_CACHE}
};

static std::string common_speculative_get_devices_str(const std::vector<ggml_backend_dev_t> & devices) {
    std::string result;
    for (size_t i = 0; i < devices.size(); i++) {
        if (devices[i] == nullptr) {
            continue;
        }
        if (!result.empty()) result += ", ";
        result += ggml_backend_dev_name(devices[i]);
    }
    return result.empty() ? "default" : result;
}

struct common_speculative_config {
    common_speculative_type type;
    common_params_speculative params;

    common_speculative_config(common_speculative_type t,
            const common_params_speculative & p = common_params_speculative{}) : type(t), params(p) {}
};

static bool common_speculative_are_compatible(
    const llama_model * model_tgt,
    const llama_model * model_dft) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const auto vocab_type_tgt = llama_vocab_type(vocab_tgt);
    SPC_DBG("vocab_type tgt: %d\n", vocab_type_tgt);

    const auto vocab_type_dft = llama_vocab_type(vocab_dft);
    SPC_DBG("vocab_type dft: %d\n", vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        SPC_WRN("draft model vocab type must match target model to use speculation but "
                "vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        (llama_vocab_get_add_bos(vocab_tgt) && llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft))) {
        SPC_WRN("draft model bos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                llama_vocab_get_add_bos(vocab_tgt), llama_vocab_get_add_bos(vocab_dft),
                llama_vocab_bos(vocab_tgt), llama_vocab_bos(vocab_dft));
        return false;
    }

    if (llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        (llama_vocab_get_add_eos(vocab_tgt) && llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft))) {
        SPC_WRN("draft model eos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                llama_vocab_get_add_eos(vocab_tgt), llama_vocab_get_add_eos(vocab_dft),
                llama_vocab_eos(vocab_tgt), llama_vocab_eos(vocab_dft));
        return false;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            SPC_DBG("draft model vocab must closely match target model to use speculation but "
                    "target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                SPC_DBG("draft model vocab must match target model to use speculation but "
                        "token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(vocab_tgt, i).c_str(),
                        common_token_to_piece(vocab_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

using common_speculative_draft_params_vec = std::vector<common_speculative_draft_params>;

// state of an implementation of speculative decoding
//
// each implementation has a unique type and a state that is implementation-specific
// in a subclass of common_speculative_impl
struct common_speculative_impl {
    const common_speculative_type type;

    uint32_t n_seq;
    int32_t n_max; // maximum draft length after implementation-specific limits
    int32_t n_max_ext = 0; // [TAG_DFL_LABD] longest draft with lookup rows past n_max (0 = n_max)

    size_t n_call_begin  = 0; // number of times this implementation was called for refresh.
    size_t n_call_draft  = 0; // number of times this implementation was called for generation.
    size_t n_call_accept = 0; // number of times this implementation was called for accumulation.

    size_t n_gen_drafts = 0; // number of times a draft or part was generated by this implementation.
    size_t n_acc_drafts = 0; // number of times a draft or part was accepted by the target model.
    size_t n_gen_tokens = 0; // number of tokens generated by this implementation.
    size_t n_acc_tokens = 0; // number of tokens accepted by the target model.

    std::vector<size_t> n_acc_tokens_per_pos; // number of tokens accepted per draft position.

    // TODO: track performance of most recent calls
    const bool gen_perf = true; // whether to generate performance stats.

    int64_t t_begin_us  = 0; // total time spent in refresh of this implementation in microseconds.
    int64_t t_draft_us  = 0; // total time spent in generating drafts in this implementation in microseconds.
    int64_t t_accept_us = 0; // total time spent in accumulation of this implementation in microseconds.

    common_speculative_impl(common_speculative_type type, uint32_t n_seq, int32_t n_max) : type(type), n_seq(n_seq), n_max(n_max) {}

    virtual ~common_speculative_impl() = default;

    virtual void begin(llama_seq_id seq_id, const llama_tokens & prompt) = 0;

    virtual bool process(const llama_batch & batch) = 0;

    // [TAG_SPEC_PREFILL_TAIL_EXTRACT] true when this implementation will not read the target's
    // per-layer inputs (llama_get_embeddings_layer_inp) for this batch, so the target may skip
    // extracting them. Only implementations that read them need to override: dflash answers with
    // its sliding-window skip decision, eagle3 always reads them. The rest never touch them.
    virtual bool prefill_skips_layer_inputs(const llama_batch & batch) const {
        (void) batch;
        return true;
    }

    // [TAG_SPEC_PREFILL_TAIL]
    // How many prompt tokens still follow the ubatch currently being processed.
    // Set by the server before each common_speculative_process() call during prompt
    // processing; 0 during generation and whenever the caller does not know.
    //
    // A drafter whose attention is a sliding window only needs a warm KV for the last
    // n_swa positions - everything earlier is evicted before it is ever drafted from -
    // so an implementation may use this to skip work on early prefill ubatches.
    int32_t n_prefill_after = 0;

    // [TAG_SPEC_PREFILL_TAIL_PER_SEQ] The scalar above is the MAX over every slot still
    // processing a prompt, which is the right input for the is_prefill heuristic but the WRONG
    // one for the skip decision. The skip wipes the drafter KV of every sequence in the batch,
    // and with -np >= 2 and continuous batching a GENERATING slot is routinely co-batched with a
    // prefilling one. It would then have its drafter cache cleared on every batch for the whole
    // of the other slot's prefill - no crash and no wrong output, but acceptance collapses to
    // ~0 and speculation becomes pure overhead. Indexed by llama_seq_id, which the server sets
    // to the slot id. -1 means "not prefilling", i.e. never skip this sequence.
    // MEASURED A/B, Qwen3.8-27B-UD-Q5_K_XL, -np 2 --kv-unified, one slot generating 700 tokens
    // while the other prefills 60000, acceptance of the GENERATING slot:
    //   scalar (pre-fix)   40.2% solo -> 32.5% co-batched   80.7% retained
    //   per-seq (this)     40.2% solo -> 40.0% co-batched   99.5% retained
    // The loss is a 19% relative dent, not the total collapse the mechanism suggests, because
    // the drafter re-warms between wipes. Still worth having, and it grows with prompt length.
    std::vector<int32_t> n_prefill_after_seq;

    int32_t prefill_after_for(llama_seq_id s) const {
        if (s < 0 || (size_t) s >= n_prefill_after_seq.size()) {
            return 0;
        }
        return n_prefill_after_seq[s];
    }

    virtual void draft(common_speculative_draft_params_vec & dparams) = 0;

    virtual void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) = 0;

    // (optional) serialize/restore per-seq internal state (e.g. eagle3's deferred boundary).
    virtual bool get_state(llama_seq_id /*seq_id*/, std::vector<uint8_t> & /*data*/) const { return false; }
    virtual void set_state(llama_seq_id /*seq_id*/, const std::vector<uint8_t> & /*data*/) {}

    // (optional) more statistics at the end of a generation, from common_speculative_print_stats()
    virtual void print_stats_extra() {}
};

struct common_speculative_impl_draft_simple : public common_speculative_impl {
    common_params_speculative_draft params;

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    common_speculative_impl_draft_simple(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE, n_seq, params.draft.n_max)
        , params(params.draft)
    {
        auto * ctx_dft = this->params.ctx_dft;
        auto * ctx_tgt = this->params.ctx_tgt;

        if (!ctx_dft) {
            throw std::runtime_error("draft-simple requires a draft context");
        }

        SPC_TRC("%s", "adding speculative implementation 'draft-simple'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%f\n", this->params.n_max, this->params.n_min, this->params.p_min);
        SPC_TRC("- gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n",
                this->params.n_gpu_layers,
                ggml_type_name(this->params.cache_type_k),
                ggml_type_name(this->params.cache_type_v),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no",
                common_speculative_get_devices_str(this->params.devices).c_str());

        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);

        // TODO: optimize or pass from outside?
        // {
        //     common_params_sampling params;
        //     params.no_perf = false;
        //
        //     params.top_k = 40;
        //     params.top_p = 0.9;
        //
        //     params.samplers = {
        //         COMMON_SAMPLER_TYPE_TOP_K,
        //         COMMON_SAMPLER_TYPE_TOP_P,
        //         COMMON_SAMPLER_TYPE_INFILL,
        //     };
        //
        //     result->smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        // }

        smpls.resize(n_seq);
        for (auto & smpl : smpls) {
            common_params_sampling params;
            params.no_perf = false;
            params.top_k = 10;
            params.samplers.assign(1, COMMON_SAMPLER_TYPE_TOP_K);

            smpl.reset(common_sampler_init(llama_get_model(ctx_dft), params));
        }

        const bool vocab_cmpt = common_speculative_are_compatible(llama_get_model(ctx_tgt), llama_get_model(ctx_dft));
        SPC_DBG("vocab_cmpt = %d\n", vocab_cmpt);

        if (!vocab_cmpt) {
            SPC_ERR("%s", "the target and draft vocabs are not compatible\n");

            throw std::runtime_error("draft model vocab type must match target model to use speculation");
        }

        if (n_seq != llama_n_seq_max(ctx_dft)) {
            SPC_ERR("n_seq mismatch: %d != %d\n", n_seq, llama_n_seq_max(ctx_dft));

            throw std::runtime_error("the draft model number of sequences is incompatible with the speculative n_seq");
        }
    }

    ~common_speculative_impl_draft_simple() override {
        llama_batch_free(batch);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & batch) override {
        auto * ctx_dft = params.ctx_dft;

        llama_batch batch_dft = batch;
        batch_dft.logits = nullptr;

        const int ret = llama_decode(ctx_dft, batch_dft);

        if (ret != 0) {
            SPC_ERR("failed to decode draft batch, ret = %d\n", ret);

            return false;
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            common_batch_add(batch, dp.id_last, dp.pos0, { seq_id }, true);
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            SPC_ERR("llama_decode returned %d\n", ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                // add drafted token for each sequence
                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if ((params.n_max <= (int) result.size()) ||
                    (dp.n_max > 0 && dp.n_max <= (int) result.size())) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, dp.pos0 + i + 1, { seq_id }, true);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                SPC_ERR("llama_decode[%d] returned %d\n", i, ret);
                break;
            }

            ++i;
        }

        for (auto & dp : dparams) {
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};


// EAGLE3 speculative decoding state
//
// Input of draft decoder: (This is different compared to MTP)
//   At "pos P", the decoder takes input pair (t_{P+1}, g_P), with RoPE at P.
//     - t_{P+1} = token at sequence pos P+1 (the *next* token after P)
//     - g_P     = encoder output = projection of target's extracted hidden states at P
//
// Deferred boundary (MTP doesn't have this issue):
//   Within a single process() call with n_tokens, we can only write decoder KV for
//   training pos 0..n_tokens-2. The last training pos (n_tokens-1) needs t_{n_tokens}
//   which lies *outside* this batch — it is the token target will sample next or the first token from next ubatch.
//   So the last training pos of each process() call is *deferred* to whichever next call has
//   the missing token in hand:
//     - multi-ubatch prefill: the next process()'s first token completes the pair
//                              (handled by the per-seq "cross-ubatch bridge")
//     - single-ubatch prefill / after verify: draft()'s seed step uses "dp.id_last"
//                              (target's freshest sample) to complete the pair
//
// Per-seq carry-over state:
//   pending_g_last    [n_embd_dec]  ┐  the deferred boundary's (g, pos). Set by
//   pending_pos_last  llama_pos     ┘  process() at end of ubatch (= last row);
//                                       rebased by accept() to first-non-accepted pos.
//   verify_g          [N × n_embd_dec] snapshot of process()'s encoder output;
//   verify_pos_first  llama_pos         consumed by accept() to recover the right
//   verify_g_rows     int32_t           pending_g_last row for any n_accepted value.
//
// Performance is overall good but there is waste in verify cycle:
//   process() runs encoder + decoder on the *full* verify batch including rows for
//   rejected drafts. The KV at those positions is then dropped.
//
// TODO: Not sure if we need optimization for this waste?
// If so we may need hybrid stash:
//      in verify mode, have process() only stash features and let draft() seed run
//      encoder+decoder on n_accepted+1 rows).
struct common_speculative_impl_draft_eagle3 : public common_speculative_impl {
    // [TAG_SPEC_PREFILL_TAIL_EXTRACT] eagle3 reads the target's layer inputs on every batch
    bool prefill_skips_layer_inputs(const llama_batch & batch) const override {
        (void) batch;
        return false;
    }

    common_params_speculative_draft params;
    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd_dec = 0;       // draft hidden size
    int32_t n_embd_enc = 0;       // target_layer_ids_n * target_hidden_size
    int32_t n_embd_tgt = 0;       // target model hidden size
    int32_t n_layer_tgt = 0;      // target model layer count

    const int32_t * target_layer_ids   = nullptr; // model_dft's extract layer indices
    uint32_t        target_layer_ids_n = 0;

    // [per-seq] deferred boundary state
    std::vector<std::vector<float>> pending_g_last;
    std::vector<llama_pos>          pending_pos_last;

    // [per-seq] snapshot of the most recent process()'s encoder output
    std::vector<std::vector<float>> verify_g;         // [n_seq][n_rows * n_embd_dec]
    std::vector<llama_pos>          verify_pos_first; // [n_seq] — pos of verify_g[seq][0]
    std::vector<int32_t>            verify_g_rows;    // [n_seq] — number of rows

    // scratch buffer for concatenated target features [n_tokens, n_embd_enc]
    std::vector<float> features_buf;
    std::vector<float> g_embd_buf;

    common_speculative_impl_draft_eagle3(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, n_seq, params.draft.n_max)
        , params(params.draft)
    {
        SPC_TRC("%s", "adding speculative implementation 'draft-eagle3'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%f, backend_sampling=%d\n", params.draft.n_max, params.draft.n_min, params.draft.p_min, (int) params.draft.backend_sampling);

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "EAGLE3 requires ctx_tgt and ctx_dft to be set");

        const llama_model * model_dft = llama_get_model(ctx_dft);
        const llama_model * model_tgt = llama_get_model(ctx_tgt);

        target_layer_ids   = llama_model_target_layer_ids  (model_dft);
        target_layer_ids_n = llama_model_target_layer_ids_n(model_dft);
        if (target_layer_ids_n != 3) {
            throw std::runtime_error("draft model is not eagle3 (expected 3 extract layers, got " +
                                     std::to_string(target_layer_ids_n) + ")");
        }

        n_embd_tgt = llama_model_n_embd(model_tgt);
        n_embd_dec = llama_model_n_embd(model_dft);
        n_embd_enc = (int32_t) target_layer_ids_n * n_embd_tgt;
        n_layer_tgt = llama_model_n_layer(model_tgt);

        // [TAG_SPEC_BATCH_FROM_TGT] This batch receives tokens copied out of the TARGET's
        // batch in process(), so it must be sized by what the target can hand us, not by
        // the draft context's own n_batch. SPEC_DFT_UBATCH deliberately clamps the draft
        // context's n_batch/n_ubatch (to shrink its logits buffer, which reserves
        // n_ubatch x n_vocab floats), and that clamp was silently shrinking this
        // allocation too: with SPEC_DFT_UBATCH=256 and a 1024-token prefill ubatch,
        // common_batch_add walked off the end and tripped
        //   GGML_ASSERT(batch.seq_id[batch.n_tokens] && "llama_batch size exceeded")
        // on the first prompt longer than 256 tokens. Short prompts fit, which is why
        // this only showed up once a real prompt was used.
        const uint32_t n_b_dft = llama_n_batch(ctx_dft);
        const uint32_t n_b_tgt = ctx_tgt ? llama_n_batch(ctx_tgt) : 0u;
        const int32_t  n_b     = (int32_t) std::max(n_b_dft, n_b_tgt);
        batch = llama_batch_init(/*n_tokens=*/ n_b, /*embd=*/ n_embd_dec, /*n_seq_max=*/ 1);
        // llama_batch_init allocates only one of token/embd; eagle3 decoder needs both.
        // TODO: fix, how to call without malloc
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_b);

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        // turn on extraction of the target layers' hidden states
        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            if (target_layer_ids[k] < n_layer_tgt) {
                llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k], true);
            } else if (target_layer_ids[k] == n_layer_tgt) {
                llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
            } else {
                GGML_ABORT("EAGLE3: target layer id %d exceeds target n_layer %d", target_layer_ids[k], n_layer_tgt);
            }
        }

        // turn on extraction of the draft model's pre-norm hidden state
        // (used both for the encoder output g_embd and the decoder pre-norm output).
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);

        pending_g_last.assign(n_seq, std::vector<float>(n_embd_dec, 0.0f));
        pending_pos_last.assign(n_seq, -1);

        verify_g.assign(n_seq, std::vector<float>());
        verify_pos_first.assign(n_seq, -1);
        verify_g_rows.assign(n_seq, 0);
    }

    ~common_speculative_impl_draft_eagle3() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        if (batch.token != nullptr) {
            free(batch.token);
            batch.token = nullptr;
        }
        llama_batch_free(batch);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }
        // expected state after prefill: ctx_dft has pos 0..N-2 (last position is deferred to
        // draft()'s seed step). Warn only if more than one position is missing.
        auto * ctx_dft = this->params.ctx_dft;
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
        if (pos_max < N - 2) {
            SPC_WRN("ctx_dft pos_max=%d < N-2=%d — process() did not run on every prefill ubatch. "
                    "Drafts may degrade.\n",
                    (int) pos_max, N - 2);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        if (batch_in.token == nullptr || batch_in.embd != nullptr) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // i_batch_beg[seq] / i_batch_end[seq]: inclusive batch indices of this seq's
        // first/last token in batch_in. Assumes per-seq tokens are contiguous within
        // the ubatch (server's default ordering).
        std::vector<int32_t> i_batch_beg(n_seq, -1);
        std::vector<int32_t> i_batch_end(n_seq, -1);
        for (int k = 0; k < n_tokens; ++k) {
            GGML_ASSERT(batch_in.n_seq_id[k] == 1);
            const llama_seq_id seq_id = batch_in.seq_id[k][0];
            if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
                continue;
            }
            i_batch_end[seq_id] = k;
            if (i_batch_beg[seq_id] < 0) {
                i_batch_beg[seq_id] = k;
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        // Interleave each extract_layer's hidden state into a contiguous buffer of
        // shape [n_tokens, target_layer_ids_n * n_embd_tgt]. Then run EAGLE3 encoder
        // to get one g_embd row per token.
        features_buf.resize((size_t) n_tokens * n_embd_enc, 0.0f);

        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            const float * layer = target_layer_ids[k] < n_layer_tgt
                ? llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k])
                : llama_get_embeddings_nextn(ctx_tgt);
            if (!layer) {
                GGML_ABORT("EAGLE3: target layer %d input not extracted.", target_layer_ids[k]);
            }
            for (int32_t i = 0; i < n_tokens; ++i) {
                float * dst = features_buf.data() + (size_t) i * n_embd_enc + k * (size_t) n_embd_tgt;
                const float * src = layer + (size_t) i * n_embd_tgt;
                std::memcpy(dst, src, (size_t) n_embd_tgt * sizeof(float));
            }
        }

        g_embd_buf.resize((size_t) n_tokens * n_embd_dec);

        // llama_encode() requires the full encoder batch to fit in n_ubatch.
        // Allow batch > ubatch: eagle3's per-token encoder can be chunked safely.
        const int32_t n_ubatch_dft = (int32_t) llama_n_ubatch(ctx_dft);
        for (int32_t i = 0; i < n_tokens; i += n_ubatch_dft) {
            const int32_t n_chunk = std::min(n_ubatch_dft, n_tokens - i);

            llama_batch enc_batch = {
                /*.n_tokens =*/ n_chunk,
                /*.token    =*/ nullptr,
                /*.embd     =*/ features_buf.data() + (size_t) i * n_embd_enc,
                /*.pos      =*/ nullptr,
                /*.n_seq_id =*/ nullptr,
                /*.seq_id   =*/ nullptr,
                /*.logits   =*/ nullptr,
            };
            const int32_t rc = llama_encode(ctx_dft, enc_batch);
            if (rc != 0) {
                SPC_ERR("llama_encode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                        rc, (int) n_chunk, (int) i);
                return false;
            }

            // g_embd has shape [n_chunk, n_embd_dec] in ctx_dft's pre-norm embeddings buffer.
            const float * g_embd_chunk = llama_get_embeddings_nextn(ctx_dft);
            GGML_ASSERT(g_embd_chunk && "EAGLE3 encoder produced no output.");
            std::memcpy(g_embd_buf.data() + (size_t) i * n_embd_dec,
                        g_embd_chunk,
                        (size_t) n_chunk * n_embd_dec * sizeof(float));
        }

        const float * g_embd = g_embd_buf.data();

        const size_t row_bytes = (size_t) n_embd_dec * sizeof(float);

        // EAGLE3 decoder input convention: at memory pos P the input pair is
        // (token[P+1], g_embd[P]). This shifts the token index "left by one" relative to g_embd.
        //
        // Per seq, in order:
        //   (a) cross-ubatch bridge — when applicable, write the previously-deferred
        //       pos using this ubatch's first token + pending_g_last.
        //   (b) main write loop — for k in [beg, end-1], write (token[k+1], g_embd[k])
        //       at pos[k]. The last training pos (k=end) is left unwritten = new
        //       deferred boundary, completed by the next process() or draft() call.
        //   (c) refresh deferred state — stash this ubatch's full g_embd into verify_g,
        //       update pending_g_last / pending_pos_last to the last row.
        common_batch_clear(batch);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            const int32_t beg = i_batch_beg[seq_id];
            const int32_t end = i_batch_end[seq_id];
            if (beg < 0 || end < 0) {
                continue;
            }

            // cross-ubatch bridge — complete the prior ubatch's deferred boundary.
            // Fires iff all three preconditions hold:
            //   1) pending_pos_last >= 0
            //   2) pending_pos_last + 1 == pos[beg]
            //   3) pending_pos_last > dft_pos_max // TODO: is this check needed?
            const llama_pos pending_pos = pending_pos_last[seq_id];
            if (pending_pos >= 0 && pending_pos + 1 == batch_in.pos[beg]) {
                const llama_pos dft_pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
                if (pending_pos > dft_pos_max) {
                    common_batch_add(batch, batch_in.token[beg], pending_pos, { seq_id }, /*logits=*/ false);
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                                pending_g_last[seq_id].data(), row_bytes);
                }
            }

            for (int32_t k = beg; k < end; ++k) {
                common_batch_add(batch, batch_in.token[k + 1], batch_in.pos[k], { seq_id }, /*logits=*/ false);
                std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                            g_embd + (size_t) k * n_embd_dec, row_bytes);
            }

            // refresh deferred state
            const int32_t n_rows = end - beg + 1;
            verify_pos_first[seq_id] = batch_in.pos[beg];
            pending_pos_last[seq_id] = batch_in.pos[end];
            verify_g_rows[seq_id]    = n_rows;
            verify_g[seq_id].resize((size_t) n_rows * n_embd_dec, 0.0f);
            std::memcpy(verify_g[seq_id].data(),       g_embd + (size_t) beg * n_embd_dec, row_bytes * n_rows);
            std::memcpy(pending_g_last[seq_id].data(), g_embd + (size_t) end * n_embd_dec, row_bytes);
        }

        if (batch.n_tokens > 0) {
            const int32_t rc = llama_decode(ctx_dft, batch);
            if (rc != 0) {
                SPC_ERR("llama_decode(ctx_dft) failed rc=%d (n_tokens=%d, ubatch_pos[0]=%d)\n",
                        rc, (int) batch.n_tokens, (int) batch_in.pos[0]);
                return false;
            }
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        const size_t row_bytes = (size_t) n_embd_dec * sizeof(float);

        // Complete the deferred boundary pair (dp.id_last, pending_g_last) at memory
        // pos pending_pos_last. dp.id_last is target's freshest sample (= corrected
        // token after verify, or first generated token after prefill), matching the
        // EAGLE3 input convention (token[P+1], g_embd[P]) at pos P.
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }
            if (pending_pos_last[seq_id] < 0) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, pending_pos_last[seq_id], -1);

            common_batch_add(batch, dp.id_last, pending_pos_last[seq_id], { seq_id }, true);
            std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                        pending_g_last[seq_id].data(),
                        row_bytes);
        }

        if (batch.n_tokens == 0) {
            return;
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            SPC_ERR("llama_decode returned %d\n", ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                // pre-norm hidden state of this position becomes g_embd for the next step
                const float * prenorm = llama_get_embeddings_nextn_ith(ctx_dft, i_batch);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                // (configurable via --spec-draft-p-min, set to 0.0 to disable early-stop)
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if (params.n_max <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, pending_pos_last[seq_id] + (i + 1), { seq_id }, true);
                std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec, prenorm, row_bytes);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                SPC_ERR("llama_decode[%d] returned %d\n", i, ret);
                break;
            }

            ++i;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t n_rows = verify_g_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_g = std::min<int32_t>(n_accepted, n_rows - 1);
        pending_pos_last[seq_id] = verify_pos_first[seq_id] + i_g;
        std::memcpy(pending_g_last[seq_id].data(),
                    verify_g[seq_id].data() + (size_t) i_g * n_embd_dec,
                    (size_t) n_embd_dec * sizeof(float));
    }

    // we only need to stash the deferred boundary's g_embd row for recurrent/hybrid targets:
    // their single-position checkpoints drop it on restore
    bool need_boundary_stash() const {
        const llama_model * model_tgt = llama_get_model(params.ctx_tgt);
        return llama_model_is_recurrent(model_tgt) || llama_model_is_hybrid(model_tgt);
    }

    bool get_state(llama_seq_id seq_id, std::vector<uint8_t> & data) const override {
        if (!need_boundary_stash()) {
            return false;
        }
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq || pending_pos_last[seq_id] < 0) {
            return false;
        }

        const llama_pos          pos = pending_pos_last[seq_id];
        const std::vector<float> & g = pending_g_last[seq_id];

        data.resize(sizeof(llama_pos) + g.size() * sizeof(float));
        std::memcpy(data.data(),                     &pos,     sizeof(llama_pos));
        std::memcpy(data.data() + sizeof(llama_pos), g.data(), g.size() * sizeof(float));
        return true;
    }

    void set_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) override {
        if (!need_boundary_stash()) {
            return;
        }
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }
        if (data.size() != sizeof(llama_pos) + (size_t) n_embd_dec * sizeof(float)) {
            return;
        }

        llama_pos pos = -1;
        std::memcpy(&pos, data.data(), sizeof(llama_pos));

        pending_pos_last[seq_id] = pos;
        pending_g_last[seq_id].resize(n_embd_dec);
        std::memcpy(pending_g_last[seq_id].data(), data.data() + sizeof(llama_pos), (size_t) n_embd_dec * sizeof(float));
    }
};

// DFlash: block-diffusion drafting with a draft-side KV cache injection
// [TAG_SPEC_PREFILL_TAIL] Default 2048 = one DFlash2 sliding window. SPEC_PREFILL_TAIL=0 disables
// the skip and restores the old behaviour; a larger value is more conservative (keeps more of
// the prompt warm in the draft KV).
static int32_t spec_prefill_tail() {
    static const int32_t tail = [] {
        const char * e = getenv("SPEC_PREFILL_TAIL");
        if (e == nullptr) {
            return (int32_t) 2048;
        }
        const int v = atoi(e);
        return (int32_t) (v > 0 ? v : 0);
    }();
    return tail;
}

// [TAG_SPEC_DFT_DUMP] SPEC_DFT_DUMP=<path> writes every DFlash2 selector lattice row that draft() reads
// from llama_get_embeddings_nextn, so an old and a new build can be compared offline (sync plan B4: the
// upstream in-graph selector must reproduce the layout draft() reads; a mismatch would otherwise only
// show up as lower acceptance). Unset (the default) costs one cached-pointer test per draft() call.
//
// Text, '\n' endings, the file is truncated when the process opens it. One header line, then one line
// per drafted block position:
//   # spec_dft_dump v1 top_k=<K> n_embd=<N>
//   <call> <seq_id> <pos0> <id_last> <i> <K candidate ids> <K*K scores>
//   <call>    draft() invocation counter in this process, 1-based
//   <pos0>    position of id_last, the block anchor; the line is block position <i> = 1..n_block-1
//             (row 0 of a block is the anchor and holds no candidates, so it is not written)
//   ids       the lattice's f32 values printed as they are; a non-integer means a broken layout
//   scores    predecessor-major: K groups of K successor scores, group p conditioned on candidate p
//             at position i-1 (at i=1 all groups are conditioned on the anchor and are identical)
// Floats use %.9g, which round-trips an f32 exactly. Match old and new runs on
// (seq_id, pos0, id_last, i), not on <call>: once two builds accept different tokens the steps diverge.
static FILE * spec_dft_dump_file() {
    static FILE * const f = [] () -> FILE * {
        const char * path = getenv("SPEC_DFT_DUMP");
        if (path == nullptr || path[0] == '\0') {
            return nullptr;
        }
        FILE * fp = fopen(path, "wb");
        if (fp == nullptr) {
            LOG_WRN("spec_dft_dump: cannot open SPEC_DFT_DUMP='%s' for writing, dump disabled\n", path);
        } else {
            LOG_INF("spec_dft_dump: writing the DFlash2 selector lattice to '%s'\n", path);
        }
        return fp;
    }();
    return f;
}

static void spec_dft_dump_rows(FILE * f, const float * lattice, int32_t n_embd, int32_t top_k, uint64_t n_call,
        llama_seq_id seq_id, llama_pos pos0, llama_token id_last, int32_t beg, int32_t n_block_tokens) {
    static std::mutex mtx;
    static bool header_done = false;

    std::lock_guard<std::mutex> lock(mtx);
    if (!header_done) {
        fprintf(f, "# spec_dft_dump v1 top_k=%d n_embd=%d\n", (int) top_k, (int) n_embd);
        header_done = true;
    }

    const int32_t n_used = top_k + top_k * top_k;
    for (int32_t i = 1; i < n_block_tokens; ++i) {
        const float * row = lattice + (size_t) (beg + i) * n_embd;
        fprintf(f, "%" PRIu64 " %d %d %d %d", n_call, (int) seq_id, (int) pos0, (int) id_last, (int) i);
        for (int32_t k = 0; k < n_used; ++k) {
            fprintf(f, " %.9g", (double) row[k]);
        }
        fputc('\n', f);
    }
    fflush(f);
}

// [TAG_4C_PROBE] process() calls of <= 64 rows, and how many of them synchronized the drafter (SPEC_PHASE_PROBE=1 only)
static uint64_t g_spec_dft_sync_calls  = 0;
static uint64_t g_spec_dft_sync_synced = 0;

// [TAG_DFL_LABD] lookup-augmented DFlash drafting, SPEC_DFT_LABD=1 (default off). Read once per process.
static int32_t spec_labd_env_i(const char * name, int32_t def) {
    const char * e = getenv(name);
    return (e && e[0]) ? (int32_t) atoi(e) : def;
}

static bool spec_labd_on() {
    static const bool on = spec_labd_env_i("SPEC_DFT_LABD", 0) == 1;
    return on;
}

// longest draft of a lookup copy run (SPEC_DFT_LABD_MAX, default 15 = 16 verify rows)
static int32_t spec_labd_max() {
    static const int32_t mx = std::min(31, std::max(1, spec_labd_env_i("SPEC_DFT_LABD_MAX", 15)));
    return mx;
}

int32_t common_speculative_labd_n_max(int32_t n_max) {
    return spec_labd_on() ? std::max(n_max, spec_labd_max()) : 0;
}

struct common_speculative_impl_draft_dflash : public common_speculative_impl {
    // [TAG_SPEC_PREFILL_TAIL_EXTRACT] The per-sequence skip decision, shared by process() and by the
    // server, which asks BEFORE the target decode so that llama can skip extracting the five layer
    // inputs this batch would never read: 5 x n_ubatch x n_embd x 4 B, 131 MB per 1280-token ubatch,
    // copied GPU to host on the compute stream, 13 GB over a 131k prompt. Media batches never skip.
    bool prefill_batch_wants_skip(const llama_batch & batch_in) const {
        const int32_t tail = spec_prefill_tail();
        if (tail <= 0 || batch_in.token == nullptr || batch_in.embd != nullptr) {
            return false;
        }

        // Every sequence in the batch must want the skip. The wipe in process() is unconditional
        // over the batch, so skipping on behalf of a sequence that is generating (or is near the
        // end of its own prompt) would destroy its drafter cache. Conservative by construction.
        llama_seq_id prev_s = -1;
        int n_seqs_seen = 0;
        for (int32_t i = 0; i < batch_in.n_tokens; ++i) {
            if (batch_in.n_seq_id == nullptr || batch_in.n_seq_id[i] <= 0) {
                continue;
            }
            const llama_seq_id s = batch_in.seq_id[i][0];
            if (s == prev_s) {
                continue;
            }
            prev_s = s;
            ++n_seqs_seen;
            if (prefill_after_for(s) < tail) {
                return false;
            }
        }

        return n_seqs_seen > 0;
    }

    bool prefill_skips_layer_inputs(const llama_batch & batch) const override {
        return prefill_batch_wants_skip(batch);
    }

    common_params_speculative_draft params;

    llama_batch batch;        // noise tokens
    llama_batch batch_inject; // target features for KV cache injection

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd_dec = 0;  // draft hidden size
    int32_t n_embd_enc = 0;  // target_layer_ids_n * target_hidden_size
    int32_t n_embd_tgt = 0;  // target model hidden size

    int32_t     block_size    = 0;
    int32_t     trained_max   = 0;   // [TAG_DFL_BLOCK_EXT] longest draft of the trained block
    llama_token mask_token_id = 0;

    bool    is_dflash2     = false;
    bool    is_mrope       = false;
    int32_t selector_top_k = 0;
    std::vector<std::mt19937> selector_rng;
    std::vector<bool> selector_reset;

    uint64_t n_dump_call = 0; // [TAG_SPEC_DFT_DUMP] draft() calls written so far

    // draft-dspark: the draft carries a Markov head and uses an anchor-first block layout
    const bool is_dspark;

    // dspark speculators
    bool sample_from_anchor = true;

    // block-internal attention
    bool causal_attn = false;

    const int32_t * target_layer_ids   = nullptr; // model_dft's extract layer indices
    uint32_t        target_layer_ids_n = 0;

    std::vector<int32_t> sync_rows_seq; // [TAG_4C_DFT_SYNC] rows per sequence in the batch, reused

    // [TAG_DFL_ADAPT] content-adaptive draft length (--spec-draft-adapt on, or SPEC_DFT_ADAPT=1), for n_max up to the
    // trained block (DFlash2: 7). Chosen per step, BEFORE the drafter decode, as ONE length k for all drafting sequences:
    // on the hybrid target a batch of different lengths is split (split_equal) and every distinct verify shape resets the
    // CUDA graphs, so ragged lengths cost far more than they save. Candidates are the short draft (3, the old fixed
    // default, drafted with the short 4-token block, so a k = 3 step is the n_max 3 step) and the full block (n_max).
    // Per sequence and draft position j the conditional acceptance alpha_j (P(j accepted | j - 1 accepted)) is tracked
    // with decayed counts, shrunk towards the same counts pooled over all sequences (the position profile is not flat).
    // k maximises the geometric mean over the drafting sequences of E_s(k) / C(k) (proportional fairness: a length that
    // speeds up the batch while one stream slows down must win clearly), with
    //     E_s(k) = 1 + sum_{j < k} prod_{i <= j} alpha_i        expected tokens per step, bonus token included
    //     C(k)   = 1 + seq*n + cost*R + step*[R > rows]         relative step time; n = drafting sequences, R = n*(k + 1)
    // seq, cost and step grow per 32768 tokens of mean depth; the step term is the jump when the verify batch leaves the
    // kernels for <= 16 columns. Defaults: least-squares fit of the fixed n_max 3 / 7 sweep of Qwen3.8-27B-UD-Q5_K_XL +
    // DFlash2 on the RTX 5090 at 1/2/4 streams (ms/step = 21.0 + 0.53 n + 0.43 R + 3.3 [R > 16] at depth 0,
    // 18.7 + 3.3 n + 0.62 R + 3.6 [R > 16] at 32K). A hysteresis keeps the verify shape (and its graphs) stable, and
    // SPEC_DFT_ADAPT_KBYN caps k by the number of drafting sequences. Default cap: n_max with 1 or 2 drafting sequences,
    // short_k with 3 or more. Measured (24 prompts, 4 streams): the model above picked k = 7 on most steps and lost 7-11 %
    // against a fixed 3, because it prices the extra verify rows but not the lower acceptance of the long block's first
    // positions at this batch width. With the cap, 3-4 streams run the n_max 3 step exactly.
    // On by default when n_max > short_k (with n_max <= short_k it is the fixed draft, bit for bit).
    struct adapt_cfg {
        bool  on      = false;
        float cost    = 0.021f;  // SPEC_DFT_ADAPT_COST: relative step cost of one more verify row
        float step    = 0.158f;  // SPEC_DFT_ADAPT_STEP: one-off relative cost when R exceeds `rows`
        float over    = 0.0f;    // SPEC_DFT_ADAPT_OVER: extra relative cost of each row above `rows`
        float seq_d   = 0.153f;  // SPEC_DFT_ADAPT_SEQ_D:  seq  added per 32768 tokens of mean depth
        float cost_d  = 0.012f;  // SPEC_DFT_ADAPT_COST_D: cost added per 32768 tokens of mean depth
        float step_d  = 0.033f;  // SPEC_DFT_ADAPT_STEP_D: step added per 32768 tokens of mean depth
        float seq     = 0.025f;  // SPEC_DFT_ADAPT_SEQ: relative cost of one more drafting sequence
        int   rows    = 16;      // SPEC_DFT_ADAPT_ROWS
        int   k_min   = 1;       // SPEC_DFT_ADAPT_KMIN
        int   short_k = 3;       // SPEC_DFT_ADAPT_SHORT: the short candidate
        std::vector<int> kbyn;   // SPEC_DFT_ADAPT_KBYN=k1,k2,...: cap on k with 1, 2, ... drafting sequences (the last
                                 //   entry covers more); a cap of short_k or less fixes k (and the short block).
                                 //   Unset: n_max,n_max,short_k. 0: no cap
        bool  all_k   = false;   // SPEC_DFT_ADAPT_ALLK=1: every length from k_min to n_max is a candidate
        float decay   = 0.95f;   // SPEC_DFT_ADAPT_DECAY: weight of the past per accepted step
        float prior   = 0.6f;    // SPEC_DFT_ADAPT_PRIOR: alpha of a new sequence ...
        float prior_w = 2.0f;    // ... worth this many trials (pooled estimate)
        float shrink  = 3.0f;    // SPEC_DFT_ADAPT_SHRINK: the pooled estimate is worth this many of a sequence's trials
        float gdecay  = 0.995f;  // SPEC_DFT_ADAPT_GDECAY: decay of the pooled counts per verified position
        float hyst    = 0.02f;   // SPEC_DFT_ADAPT_HYST: a new k must gain this fraction, plus ...
        float hyst_dn = 0.06f;   // SPEC_DFT_ADAPT_HYST_DOWN: ... this one for a shorter k with ONE stream: its extra verify
                                 //   rows are nearly free, and a short draft loses far more once the text turns
                                 //   predictable again (the estimate lags) than a long one loses in a hard stretch
        float hyst_rows = 0.06f; // SPEC_DFT_ADAPT_HYST_ROWS: ... and this one when the switch crosses `rows`
        int   small   = 3;       // SPEC_DFT_ADAPT_SMALL: drafter block of small + 1 while k <= small (0 = always full).
                                 //   Measured: the full block cut to 3 accepts the same as the short block (2.95 vs 2.96
                                 //   tokens/step greedy, 24 prompts) and only costs drafter time (-3 % at 1 stream)
        bool  fair    = true;    // SPEC_DFT_ADAPT_FAIR: 1 = proportional-fair share of the streams, 0 = batch total
        float pareto  = 0.01f;   // SPEC_DFT_ADAPT_PARETO: largest predicted slow-down of any stream for a longer k (< 0: off)
        bool  probe   = false;   // SPEC_DFT_ADAPT_PROBE=1: a k histogram every 256 draft calls
        bool  trace   = false;   // SPEC_DFT_ADAPT_PROBE=2: also one line per decision
        float rdecay  = 0.95f;   // SPEC_DFT_ADAPT_RDECAY: weight of a sequence's past realized tokens/step per step
        float rw0     = 2.0f;    // SPEC_DFT_ADAPT_RW0: the model estimate is worth this many realized steps (0: model only)
        bool  rel     = true;    // SPEC_DFT_ADAPT_REL: a position's prior is the sequence's previous position times the pooled
                                 //   ratio of the two (0: the pooled value itself)
        bool  solo    = true;    // SPEC_DFT_ADAPT_SOLO: one drafting sequence drafts the cap (at most the trained block)
        bool  duo     = true;    // SPEC_DFT_ADAPT_DUO: two drafting sequences follow the rule below, not the model
        float duo_deep = 8192.0f; // SPEC_DFT_ADAPT_DUO_DEEP: mean depth from which two sequences need ...
        float duo_acc  = 0.80f;  // SPEC_DFT_ADAPT_DUO_ACC: ... this first-position acceptance each for the long draft
        float duo_band = 0.05f;  // SPEC_DFT_ADAPT_DUO_BAND: +- band around duo_acc (hysteresis)
    };
    adapt_cfg ad;

    // [TAG_DFL_QTRUNC] SPEC_DFT_QTRUNC=1: a sampled DFlash2 draft picks from its proposal cut the way the request cuts the
    // target (top-k, top-p, min-p over the selector candidates, renormalised). The cut proposal is the one handed to
    // the verifier, so the output distribution stays exact; the drafter just stops proposing tail tokens the target's
    // own cut would reject. NEGATIVE RESULT (24 prompts, 1 stream, temp 1 / top-p 0.95 / top-k 20): acceptance
    // 0.41 -> 0.35, -11 % decode. The proposal is not too flat; cutting it makes the accepted ratio p/q smaller.
    bool dft_qtrunc = false;
    // [TAG_DFL_QTEMP] SPEC_DFT_QTEMP=x: the sampled DFlash2 proposal uses temperature x * the request's (default 1). The
    // verifier gets the same proposal, so the output stays exact; only the acceptance changes. NEGATIVE RESULT (same
    // set): x 1.25 -9 %, x 0.8 -14 %; the selector's own temperature-1 proposal is the best of the three.
    float dft_qtemp = 1.0f;

    static void dfl_qtrunc(std::vector<float> & probs, int32_t top_k, float top_p, float min_p) {
        const int32_t n = (int32_t) probs.size();
        std::vector<int32_t> ord(n);
        for (int32_t i = 0; i < n; ++i) {
            ord[i] = i;
        }
        std::stable_sort(ord.begin(), ord.end(), [&](int32_t a, int32_t b) { return probs[a] > probs[b]; });
        int32_t keep = n;
        if (top_k > 0 && top_k < keep) {
            keep = top_k;
        }
        if (top_p < 1.0f) {
            float cum = 0.0f;
            for (int32_t i = 0; i < keep; ++i) {
                cum += probs[ord[i]];
                if (cum >= top_p) {
                    keep = i + 1;
                    break;
                }
            }
        }
        if (min_p > 0.0f && n > 0) {
            const float thr = min_p*probs[ord[0]];
            int32_t m = 1;
            while (m < keep && probs[ord[m]] >= thr) {
                ++m;
            }
            keep = m;
        }
        if (keep >= n || keep <= 0) {
            return;
        }
        float sum = 0.0f;
        for (int32_t i = 0; i < keep; ++i) {
            sum += probs[ord[i]];
        }
        for (int32_t i = keep; i < n; ++i) {
            probs[ord[i]] = 0.0f;
        }
        if (sum > 0.0f) {
            for (int32_t i = 0; i < keep; ++i) {
                probs[ord[i]] /= sum;
            }
        }
    }

    // per sequence and draft position j (0-based): decayed trials (drafts that reached j: every earlier position was
    // accepted) and successes (j accepted), [s*n_max + j]; the same pooled over all sequences with a slower decay. A
    // position a sequence has not drafted lately falls back to the pooled one: the conditional acceptance is NOT flat
    // (measured: after three accepted positions the next ones are accepted more often, most of all when sampling)
    std::vector<float>   ad_tr;
    std::vector<float>   ad_ok;
    std::vector<float>   ad_gtr;
    std::vector<float>   ad_gok;
    // [TAG_DFL_ADAPT_REAL] per sequence and draft length: decayed realized tokens per step and step count, [s*(n_max+1) + k]
    std::vector<float>   ad_rt;
    std::vector<float>   ad_rn;
    std::vector<int32_t> ad_last;  // draft length handed out in the last draft() per sequence (0 = none pending)
    int32_t  ad_k       = 0;       // the current common length (0 = not chosen yet)
    uint64_t ad_calls   = 0;
    std::vector<uint64_t> ad_hist; // probe: steps per chosen k

    // [TAG_DFL_ADAPT_CONTENT] content-aware length for 1-2 drafting sequences, SPEC_DFT_ADAPT_CONTENT=1 (default off).
    // Today's rule (solo: long, duo: long unless deep and the first position is weak) stays an upper bound; the policy
    // only replaces its long draft by the short one when that gives more tokens per unit of step time. Per sequence:
    //     E_short = 1 + E[min(a, s)]      one sample per realized step of either length (the long block cut to s
    //                                     accepts as much as the short block: 2.95 vs 2.96 tokens/step, 24 prompts)
    //     E_long  = E_short + F*T         F = P(a >= s), T = E[a - s | a >= s] from long steps only, kept per sample
    // each shrunk towards the position model (adapt_alpha), value = E / C(k) with the cost model of adapt_choose(),
    // combined over the sequences like the model (SPEC_DFT_ADAPT_FAIR). The split keeps the long estimate fresh on the
    // short draft: F moves every step, only T needs long steps. A hysteresis, a dwell and a periodic long probe step
    // keep the verify shape stable. Exact: only the draft length changes.
    struct adapt_content_cfg {
        bool  on     = false;
        int   est    = 1;       // SPEC_DFT_ADAPT_CONTENT_EST: 1 = the split above, 0 = adapt_eblend() per length ([TAG_DFL_ADAPT_REAL])
        int   warm   = 16;      // SPEC_DFT_ADAPT_CONTENT_WARM: realized steps per sequence before the policy acts (rule until then)
        int   dwell  = 16;      // SPEC_DFT_ADAPT_CONTENT_DWELL: policy steps at a length before the next switch
        int   probe  = 32;      // SPEC_DFT_ADAPT_CONTENT_PROBE: one long step after this many short steps (0 = never)
        float up     = 0.03f;   // SPEC_DFT_ADAPT_CONTENT_UP: short -> long needs this relative gain
        float down   = 0.05f;   // SPEC_DFT_ADAPT_CONTENT_DOWN: long -> short needs this relative gain
        float w0     = 2.0f;    // SPEC_DFT_ADAPT_CONTENT_W0: the position model is worth this many samples
        float decay  = 0.95f;   // SPEC_DFT_ADAPT_CONTENT_DECAY: weight of the past per realized step (E_short, F)
        float tdecay = 0.9f;    // SPEC_DFT_ADAPT_CONTENT_TDECAY: weight of the past per tail sample (T)
        float tage   = 0.995f;  // SPEC_DFT_ADAPT_CONTENT_TAGE: weight of the tail samples per realized step
        int   verbose = 0;      // SPEC_DFT_ADAPT_CONTENT_LOG: 1 = step counters at generation end, 2 = also one line per step
    };
    adapt_content_cfg cx;
    std::vector<float>   cx_n, cx_e, cx_f;   // per sequence: decayed steps, sum of min(a, s) + 1, steps with a >= s
    std::vector<float>   cx_tn, cx_tt;       // per sequence: decayed tail samples and their sum of a - s
    std::vector<int32_t> cx_seen;            // per sequence: realized steps since begin()
    int32_t  cx_k     = 0;   // the policy's length (a probe does not change it; 0 = none yet)
    int32_t  cx_since = 0;   // policy steps at cx_k
    int32_t  cx_short = 0;   // short steps since the last probe
    int32_t  cx_klong = 0;   // the long length of the last step
    uint64_t cx_n_warm = 0, cx_n_rule = 0, cx_n_short = 0, cx_n_long = 0, cx_n_probe = 0, cx_n_switch = 0;   // since the last report
    double   cx_r_tok[2] = { 0.0, 0.0 };     // realized tokens at the short / long length since the last report
    uint64_t cx_r_n[2]   = { 0, 0 };

    static float adapt_env_f(const char * name, float def) {
        const char * e = getenv(name);
        return (e && e[0]) ? (float) atof(e) : def;
    }

    void adapt_init() {
        // --spec-draft-adapt on|off wins; otherwise SPEC_DFT_ADAPT=0|1; otherwise on when n_max > short_k
        ad.short_k = std::max(1,    (int) adapt_env_f("SPEC_DFT_ADAPT_SHORT", (float) ad.short_k));
        const char * e = getenv("SPEC_DFT_ADAPT");
        ad.on      = params.adapt >= 0 ? params.adapt == 1 : ((e && e[0]) ? e[0] == '1' : n_max > ad.short_k);
        ad.cost    = std::max(0.0f, adapt_env_f("SPEC_DFT_ADAPT_COST",  ad.cost));
        ad.step    = std::max(0.0f, adapt_env_f("SPEC_DFT_ADAPT_STEP",  ad.step));
        ad.over    = std::max(0.0f, adapt_env_f("SPEC_DFT_ADAPT_OVER",  ad.over));
        ad.seq_d   = std::max(0.0f, adapt_env_f("SPEC_DFT_ADAPT_SEQ_D",  ad.seq_d));
        ad.cost_d  = std::max(0.0f, adapt_env_f("SPEC_DFT_ADAPT_COST_D", ad.cost_d));
        ad.step_d  = std::max(0.0f, adapt_env_f("SPEC_DFT_ADAPT_STEP_D", ad.step_d));
        ad.seq     = std::max(0.0f, adapt_env_f("SPEC_DFT_ADAPT_SEQ",   ad.seq));
        ad.rows    = std::max(1,    (int) adapt_env_f("SPEC_DFT_ADAPT_ROWS", (float) ad.rows));
        ad.k_min   = std::max(1,    (int) adapt_env_f("SPEC_DFT_ADAPT_KMIN", (float) ad.k_min));
        ad.all_k   = adapt_env_f("SPEC_DFT_ADAPT_ALLK", 0.0f) != 0.0f;
        ad.kbyn.clear();
        const char * kb = getenv("SPEC_DFT_ADAPT_KBYN");
        if (kb && kb[0]) {
            for (const auto & t : string_split<std::string>(kb, ',')) {
                if (t.empty()) {
                    continue;
                }
                const int v = atoi(t.c_str());
                if (v <= 0) {   // 0: no cap
                    ad.kbyn.clear();
                    break;
                }
                ad.kbyn.push_back(v);
            }
        } else {
            // [TAG_DFL_BLOCK_EXT] past the trained block the 2-stream cap is the trained block
            ad.kbyn = { n_max, (trained_max > 0 ? std::min(n_max, trained_max) : n_max), std::min(ad.short_k, n_max) };
        }
        {
            const char * q = getenv("SPEC_DFT_QTRUNC");
            dft_qtrunc = (q && q[0]) ? q[0] == '1' : dft_qtrunc;
            dft_qtemp  = std::min(4.0f, std::max(0.25f, adapt_env_f("SPEC_DFT_QTEMP", dft_qtemp)));
        }
        ad.decay   = std::min(0.999f, std::max(0.0f, adapt_env_f("SPEC_DFT_ADAPT_DECAY", ad.decay)));
        ad.prior   = std::min(0.99f,  std::max(0.01f, adapt_env_f("SPEC_DFT_ADAPT_PRIOR", ad.prior)));
        ad.prior_w = std::max(0.0f, adapt_env_f("SPEC_DFT_ADAPT_PRIOR_W", ad.prior_w));
        ad.shrink  = std::max(0.01f, adapt_env_f("SPEC_DFT_ADAPT_SHRINK", ad.shrink));
        ad.gdecay  = std::min(0.9999f, std::max(0.0f, adapt_env_f("SPEC_DFT_ADAPT_GDECAY", ad.gdecay)));
        ad.hyst    = std::max(0.0f, adapt_env_f("SPEC_DFT_ADAPT_HYST",  ad.hyst));
        ad.hyst_rows = std::max(0.0f, adapt_env_f("SPEC_DFT_ADAPT_HYST_ROWS", ad.hyst_rows));
        ad.hyst_dn = std::max(0.0f, adapt_env_f("SPEC_DFT_ADAPT_HYST_DOWN", ad.hyst_dn));
        ad.small   = std::max(0,    (int) adapt_env_f("SPEC_DFT_ADAPT_SMALL", (float) ad.small));
        ad.fair    = adapt_env_f("SPEC_DFT_ADAPT_FAIR", 1.0f) != 0.0f;
        ad.pareto  = adapt_env_f("SPEC_DFT_ADAPT_PARETO", ad.pareto);
        const char * p = getenv("SPEC_DFT_ADAPT_PROBE");
        ad.probe   = p && (p[0] == '1' || p[0] == '2');
        ad.trace   = p && p[0] == '2';
        ad_tr .assign((size_t) n_seq*std::max(1, n_max), 0.0f);
        ad_ok .assign((size_t) n_seq*std::max(1, n_max), 0.0f);
        ad_gtr.assign((size_t) std::max(1, n_max), 0.0f);
        ad_gok.assign((size_t) std::max(1, n_max), 0.0f);
        ad_last.assign(n_seq, 0);
        ad.rdecay  = std::min(0.999f, std::max(0.0f, adapt_env_f("SPEC_DFT_ADAPT_RDECAY", ad.rdecay)));
        ad.rw0     = std::max(0.0f, adapt_env_f("SPEC_DFT_ADAPT_RW0", ad.rw0));
        ad.rel     = adapt_env_f("SPEC_DFT_ADAPT_REL", 1.0f) != 0.0f;
        ad.solo    = adapt_env_f("SPEC_DFT_ADAPT_SOLO", 1.0f) != 0.0f;
        ad.duo     = adapt_env_f("SPEC_DFT_ADAPT_DUO", 1.0f) != 0.0f;
        ad.duo_deep = std::max(0.0f, adapt_env_f("SPEC_DFT_ADAPT_DUO_DEEP", ad.duo_deep));
        ad.duo_acc  = std::min(1.0f, std::max(0.0f, adapt_env_f("SPEC_DFT_ADAPT_DUO_ACC", ad.duo_acc)));
        ad.duo_band = std::max(0.0f, adapt_env_f("SPEC_DFT_ADAPT_DUO_BAND", ad.duo_band));
        ad_rt.assign((size_t) n_seq*(std::max(1, n_max) + 1), 0.0f);
        ad_rn.assign((size_t) n_seq*(std::max(1, n_max) + 1), 0.0f);
        ad_hist.assign(n_max + 1, 0);
        if (ad.on) {
            std::string kcap;
            for (size_t i = 0; i < ad.kbyn.size(); ++i) {
                kcap += string_format("%s%d", i ? "," : "", ad.kbyn[i]);
            }
            LOG_INF("%s: [TAG_DFL_ADAPT] adaptive draft length on: k in [%d, %d], cap by drafting sequences {%s}, cost %.3f/seq %.3f/row, %.3f + %.3f/row "
                    "above %d rows, +%.3f/%.3f/%.3f per 32K of depth, decay %.2f, prior %.2f x %.1f, hysteresis up %.3f down %.3f "
                    "rows +%.3f, %s, block %s\n",
                    __func__, std::min(ad.k_min, n_max), n_max, kcap.empty() ? "none" : kcap.c_str(), ad.seq, ad.cost, ad.step, ad.over, ad.rows, ad.seq_d, ad.cost_d,
                    ad.step_d, ad.decay, ad.prior, ad.prior_w, ad.hyst, ad.hyst_dn, ad.hyst_rows,
                    ad.fair ? "proportional-fair" : "batch total", ad.small > 0 ? "short while k <= small" : "full");
        }
        {
            // [TAG_DFL_ADAPT_CONTENT]
            const char * c = getenv("SPEC_DFT_ADAPT_CONTENT");
            const bool want = c && c[0] == '1';
            // the long draft of the rule must be longer than the short one, else there is nothing to choose
            const int32_t k_long = trained_max > 0 ? std::min(n_max, trained_max) : n_max;
            cx.on     = want && ad.on && k_long > std::min(ad.short_k, n_max);
            cx.est    = adapt_env_f("SPEC_DFT_ADAPT_CONTENT_EST", 1.0f) != 0.0f ? 1 : 0;
            cx.warm   = std::max(0, (int) adapt_env_f("SPEC_DFT_ADAPT_CONTENT_WARM",  (float) cx.warm));
            cx.dwell  = std::max(1, (int) adapt_env_f("SPEC_DFT_ADAPT_CONTENT_DWELL", (float) cx.dwell));
            cx.probe  = std::max(0, (int) adapt_env_f("SPEC_DFT_ADAPT_CONTENT_PROBE", (float) cx.probe));
            cx.up     = std::max(0.0f, adapt_env_f("SPEC_DFT_ADAPT_CONTENT_UP",   cx.up));
            cx.down   = std::max(0.0f, adapt_env_f("SPEC_DFT_ADAPT_CONTENT_DOWN", cx.down));
            cx.w0     = std::max(0.01f, adapt_env_f("SPEC_DFT_ADAPT_CONTENT_W0",  cx.w0));
            cx.decay  = std::min(0.999f, std::max(0.0f, adapt_env_f("SPEC_DFT_ADAPT_CONTENT_DECAY",  cx.decay)));
            cx.tdecay = std::min(0.999f, std::max(0.0f, adapt_env_f("SPEC_DFT_ADAPT_CONTENT_TDECAY", cx.tdecay)));
            cx.tage   = std::min(1.0f,   std::max(0.0f, adapt_env_f("SPEC_DFT_ADAPT_CONTENT_TAGE",   cx.tage)));
            cx.verbose = std::max(0, (int) adapt_env_f("SPEC_DFT_ADAPT_CONTENT_LOG", 0.0f));
            cx_n   .assign(n_seq, 0.0f);
            cx_e   .assign(n_seq, 0.0f);
            cx_f   .assign(n_seq, 0.0f);
            cx_tn  .assign(n_seq, 0.0f);
            cx_tt  .assign(n_seq, 0.0f);
            cx_seen.assign(n_seq, 0);
            if (cx.on) {
                LOG_INF("%s: [TAG_DFL_ADAPT_CONTENT] content-aware draft length on for 1-2 drafting sequences: short %d, %s estimate, "
                        "rule for the first %d steps, dwell %d, probe every %d short steps, margin up %.3f down %.3f, model weight %.1f, "
                        "decay %.3f, tail %.3f x %.4f per step, log %d\n",
                        __func__, std::min(ad.short_k, n_max), cx.est ? "split" : "per-length", cx.warm, cx.dwell, cx.probe, cx.up,
                        cx.down, cx.w0, cx.decay, cx.tdecay, cx.tage, cx.verbose);
            } else if (want) {
                LOG_WRN("%s: [TAG_DFL_ADAPT_CONTENT] needs the adaptive draft length (--spec-draft-adapt not off) and a long draft above %d "
                        "(--spec-draft-n-max above %d): off\n", __func__, ad.short_k, ad.short_k);
            }
        }
        LOG_INF("%s: [TAG_DFL_QTRUNC] sampled drafts cut to the request's top-k/top-p/min-p: %s, proposal temperature x%.2f\n", __func__, dft_qtrunc ? "on" : "off", dft_qtemp);
    }

    // conditional acceptance of position j: the sequence's own counts, shrunk towards the pooled estimate (which is
    // shrunk towards the prior)
    float adapt_alpha(llama_seq_id s, int j) const {
        const size_t i = (size_t) s*n_max + j;
        const int    t = trained_max - 1;
        if (j > t && t >= 0 && trained_max < n_max) {
            // [TAG_DFL_BLOCK_EXT] a position past the trained block: the sequence's own last trained position times the
            // pooled ratio of this position to that one (prior 1), so a sequence tries the long block while its trained
            // positions are accepted and stops when the pooled data for the extra positions says they are not
            const float g_t = (ad_gok[t] + ad.prior*ad.prior_w) / (ad_gtr[t] + ad.prior_w);
            const float g_j = (ad_gok[j] + g_t*ad.prior_w) / (ad_gtr[j] + ad.prior_w);
            const float r   = std::min(1.1f, std::max(0.1f, g_j / std::max(1e-3f, g_t)));
            const float tgt = std::min(0.999f, adapt_alpha(s, t)*r);
            return (ad_ok[i] + tgt*ad.shrink) / (ad_tr[i] + ad.shrink);
        }
        const float g = (ad_gok[j] + ad.prior*ad.prior_w) / (ad_gtr[j] + ad.prior_w);
        float tgt = g;
        if (j > 0 && ad.rel) {
            // [TAG_DFL_ADAPT_REL] the deep positions of a sequence have few trials of their own (a trial needs every
            // earlier position accepted); the pooled value there mostly reflects other content (code at 0.9 made prose
            // at 32K look like a k = 7 case). Scale the sequence's own previous position by the pooled step instead.
            const float gp = (ad_gok[j - 1] + ad.prior*ad.prior_w) / (ad_gtr[j - 1] + ad.prior_w);
            const float r  = std::min(1.25f, std::max(0.5f, g / std::max(1e-3f, gp)));
            tgt = std::min(0.999f, adapt_alpha(s, j - 1)*r);
        }
        return (ad_ok[i] + tgt*ad.shrink) / (ad_tr[i] + ad.shrink);
    }

    // expected tokens per step for a draft of k tokens (the bonus token included): 1 + sum_j prod_{i <= j} alpha_i
    double adapt_expect(llama_seq_id s, int k) const {
        double e = 1.0, p = 1.0;
        for (int j = 0; j < k; ++j) {
            p *= adapt_alpha(s, j);
            e += p;
        }
        return e;
    }

    // [TAG_DFL_ADAPT_REAL] tokens per step for length k: the sequence's recent realized value at k, with the model
    // estimate as a prior worth rw0 steps. The model alone over-rated the long draft where the deep positions had few
    // trials of their own (prose at 32K, 2 streams: it kept k = 7 and lost 16 % against the n_max 3 step); a length
    // not used lately fades back to the model, which retries it now and then.
    double adapt_eblend(llama_seq_id s, int k) const {
        const double em = adapt_expect(s, k);
        if (ad.rw0 <= 0.0f || k < 0 || k > n_max) {
            return em;
        }
        const size_t i = (size_t) s*(n_max + 1) + k;
        return (ad_rt[i] + em*ad.rw0) / (ad_rn[i] + ad.rw0);
    }


    // [TAG_DFL_ADAPT] candidate lengths: the short draft, the cap, and (with [TAG_DFL_BLOCK_EXT]) the trained block
    bool adapt_cand(int32_t k, int32_t k_short, int32_t k_hi) const {
        return k == k_short || k == k_hi || (trained_max > k_short && trained_max < k_hi && k == trained_max);
    }

    // [TAG_DFL_ADAPT] the common draft length for this step, chosen BEFORE the drafter decode (it depends only on the
    // acceptance history, the number of drafting sequences and their depth), so the drafter block can follow it
    int32_t adapt_choose(const common_speculative_draft_params_vec & dparams) {
        std::vector<llama_seq_id> ds;
        double depth = 0.0;
        for (llama_seq_id s = 0; s < (llama_seq_id) n_seq; ++s) {
            if (!dparams[s].drafting) {
                continue;
            }
            ds.push_back(s);
            depth += (double) dparams[s].pos0;
        }
        if (ds.empty() || n_max <= 0) {
            return n_max;
        }
        // the per-concurrency cap
        const int32_t k_hi = ad.kbyn.empty() ? n_max :
            std::min<int32_t>(n_max, ad.kbyn[std::min(ds.size(), ad.kbyn.size()) - 1]);
        const int32_t k_lo = std::min(ad.k_min, k_hi);
        // [TAG_DFL_ADAPT] one drafting sequence: the long draft, no model. Its extra verify rows cost ~8 % per step and it
        // won every measured one-stream case (24 prompts: +19 % greedy, +35 % sampled; fourconn code/prose at 0 and 32K:
        // +61/+13/+58/+2 %), while the model sometimes settled on 3 from a few bad early steps
        if (ad.solo && ds.size() == 1 && k_hi > std::min(ad.short_k, n_max)) {
            const int32_t k1 = trained_max > 0 ? std::min(k_hi, trained_max) : k_hi;
            ad_k = k1;
            if (ad.probe) {
                ad_hist[std::min<size_t>(k1, ad_hist.size() - 1)]++;
            }
            return k1;
        }
        // [TAG_DFL_ADAPT] two drafting sequences: the long draft, except deep in the context, where the long draft pays only
        // when both sequences accept their first draft token often. Measured at 2 streams, long vs short: code +62 % /
        // prose 0 % at depth 0, code +29 % / prose -16 % at 32K (first-position acceptance code 0.92, prose 0.65-0.71).
        // The model (SPEC_DFT_ADAPT_DUO=0) switched back and forth on these and lost up to 9 %: each switch changes the
        // verify and drafter shapes, and its deep-position estimates rest on few trials.
        if (ad.duo && ds.size() == 2 && k_hi > std::min(ad.short_k, n_max)) {
            const int32_t k2  = trained_max > 0 ? std::min(k_hi, trained_max) : k_hi;
            const int32_t k_s = std::min(ad.short_k, n_max);
            int32_t k = k2;
            if (depth / 2.0 >= ad.duo_deep) {
                const float thr = ad.duo_acc + (ad_k == k2 ? -ad.duo_band : ad.duo_band);
                for (llama_seq_id s : ds) {
                    if (adapt_alpha(s, 0) < thr) {
                        k = k_s;
                    }
                }
            }
            ad_k = k;
            if (ad.probe) {
                ad_hist[std::min<size_t>(k, ad_hist.size() - 1)]++;
            }
            return k;
        }
        if (k_hi <= std::min(ad.short_k, n_max)) {
            ad_k = k_hi;
            if (ad.probe) {
                ad_hist[std::min<size_t>(k_hi, ad_hist.size() - 1)]++;
            }
            return k_hi;
        }
        const double  n    = (double) ds.size();
        const double  dz   = std::min(8.0, depth / n / 32768.0);   // mean depth in units of 32K
        const double  c_seq  = ad.seq  + ad.seq_d *dz;
        const double  c_row  = ad.cost + ad.cost_d*dz;
        const double  c_step = ad.step + ad.step_d*dz;
        // fair (default): sum_s log(E_s(k) / C(k)), the proportional-fair share: a length that speeds up the whole batch
        // but slows one stream down must win clearly on the others. Not fair: sum_s E_s(k) / C(k), the batch total.
        auto cost_of = [&](int32_t k) {
            const double rows = n*(k + 1);
            return 1.0 + c_seq*n + c_row*rows + (rows > ad.rows ? c_step : 0.0) + ad.over*std::max(0.0, rows - ad.rows);
        };
        auto through = [&](int32_t k) {
            const double c = cost_of(k);
            double v = 0.0;
            for (llama_seq_id s : ds) {
                const double e = adapt_eblend(s, k);
                v += ad.fair ? std::log(e / c) : e / c;
            }
            return ad.fair ? std::exp(v / n) : v;   // fair: the geometric mean of the per-stream rates
        };
        // Pareto guard: a length longer than the short candidate is allowed only when no stream is predicted to get
        // slower than with the short one by more than `pareto`. Measured without it, 2 code + 2 prose streams at k = 7:
        // the code requests finished 13 % sooner, the prose requests 7 % (greedy) to 17 % (sampled) later.
        const int32_t k_short = std::min(ad.short_k, n_max);
        auto safe = [&](int32_t k) {
            if (k <= k_short || ad.pareto < 0.0f) {
                return true;
            }
            const double cr = cost_of(k_short) / cost_of(k);
            for (llama_seq_id s : ds) {
                if (adapt_eblend(s, k) / adapt_eblend(s, k_short) * cr < 1.0 - ad.pareto) {
                    return false;
                }
            }
            return true;
        };
        // candidates: the short draft (`short_k`, the old fixed default) and the full block (n_max), or every length from
        // k_min with SPEC_DFT_ADAPT_ALLK=1. Measured: the lengths in between cost the per-row steps of the verify kernels
        // (MMVQ/MMQ-SN widths, FA rows per sequence) without the full block's acceptance.
        int32_t k_best = k_lo;
        double  t_best = -1.0;
        for (int32_t k = k_lo; k <= k_hi; ++k) {
            if ((!ad.all_k && !adapt_cand(k, k_short, k_hi)) || !safe(k)) {
                continue;
            }
            const double t = through(k);
            if (t > t_best) {
                t_best = t;
                k_best = k;
            }
        }
        // near the break-even point the estimate wanders across it: a new verify shape must pay for itself, a shorter
        // one more than a longer one, and crossing the row budget (the kernels for <= `rows` columns) more again
        const bool   crosses = ad_k > 0 && ((n*(ad_k + 1) > ad.rows) != (n*(k_best + 1) > ad.rows));
        // [TAG_DFL_BLOCK_EXT] the one-stream bias against a shorter draft holds inside the trained block only: past it
        // the extra verify rows cost 10-13 % per step, not "nearly free"
        const double margin  = ad.hyst + (k_best < ad_k && ds.size() == 1 && ad_k <= trained_max ? ad.hyst_dn : 0.0) +
                               (crosses ? ad.hyst_rows : 0.0);
        const bool   ad_k_ok = ad_k >= k_lo && ad_k <= k_hi && (ad.all_k || adapt_cand(ad_k, k_short, k_hi)) && safe(ad_k);
        if (ad_k_ok && ad_k != k_best && through(ad_k)*(1.0 + margin) >= t_best) {
            k_best = ad_k;   // not worth a new verify shape
        }
        if (ad.trace) {
            std::string tr;
            for (int32_t k = k_lo; k <= k_hi; ++k) {
                if (adapt_cand(k, k_short, k_hi)) {
                    tr += string_format(" k%d:E=%.2f,C=%.3f,t=%.3f", k, adapt_eblend(ds[0], k), cost_of(k), through(k));
                }
            }
            fprintf(stderr, "turbo-probe: dft-adapt-trace n=%d depth=%.0f prev=%d k=%d%s\n", (int) ds.size(), depth / n, ad_k, k_best, tr.c_str());
        }
        ad_k = k_best;
        if (ad.probe) {
            ad_hist[std::min<size_t>(k_best, ad_hist.size() - 1)]++;
            if (++ad_calls % 256 == 0) {
                std::string h;
                for (size_t k = 0; k < ad_hist.size(); ++k) {
                    h += string_format("%s%zu:%llu", k ? " " : "", k, (unsigned long long) ad_hist[k]);
                }
                std::string al;
                for (llama_seq_id s : ds) {
                    al += string_format("%s%d:%.2f/%.2f/%.2f", al.empty() ? "" : " ", (int) s, adapt_alpha(s, 0),
                                        adapt_alpha(s, std::min(2, n_max - 1)), adapt_alpha(s, n_max - 1));
                }
                fprintf(stderr, "turbo-probe: dft-adapt calls=%llu k now %d (n=%d, depth %.0f) hist {%s} alpha {%s}\n",
                        (unsigned long long) ad_calls, k_best, (int) ds.size(), depth / n, h.c_str(), al.c_str());
                fflush(stderr);
            }
        }
        return k_best;
    }

    // [TAG_DFL_ADAPT] drafter block for a chosen length k: the short block (anchor + `small` masks, bit-identical to a
    // fixed --spec-draft-n-max `small` drafter) while k fits in it, else the full block (the trained layout when
    // n_max = block_size - 1)
    int32_t adapt_block(int32_t k) const {
        if (ad.small > 0 && k <= ad.small) {
            return std::min(ad.small, n_max);
        }
        // [TAG_DFL_BLOCK_EXT] a draft that fits the trained block uses the trained block
        return (trained_max > 0 && trained_max < n_max && k <= trained_max) ? trained_max : n_max;
    }

    // [TAG_DFL_ADAPT] cut every drafted sequence's draft (and its distributions) to the chosen common length
    static void adapt_cut(common_speculative_draft_params_vec & dparams, const std::vector<int32_t> & i_block_beg, int32_t k) {
        for (size_t s = 0; s < dparams.size(); ++s) {
            if (i_block_beg[s] < 0) {
                continue;
            }
            auto & result = *dparams[s].result;
            if ((int32_t) result.size() > k) {
                result.resize(k);
                // [TAG_SPEC_DISTS_TRUNC] the per-draft distributions go with the draft
                if (dparams[s].dists && dparams[s].dists->size() > result.size()) {
                    dparams[s].dists->resize(result.size());
                }
            }
        }
    }

    // [TAG_DFL_ADAPT_CONTENT] relative step time C(k), the cost model of adapt_choose()
    double adapt_cost(double n, double dz, int32_t k) const {
        const double rows = n*(k + 1);
        return 1.0 + (ad.seq + ad.seq_d*dz)*n + (ad.cost + ad.cost_d*dz)*rows + (rows > ad.rows ? ad.step + ad.step_d*dz : 0.0) +
               ad.over*std::max(0.0, rows - ad.rows);
    }

    // [TAG_DFL_ADAPT_CONTENT] expected tokens per step of sequence s with the short (k_s) and the long (k_l) draft
    void adapt_content_est(llama_seq_id s, int32_t k_s, int32_t k_l, double & e_s, double & e_l) const {
        if (cx.est == 0) {
            e_s = adapt_eblend(s, k_s);
            e_l = adapt_eblend(s, k_l);
            return;
        }
        // the position model: E(k_s), P(the first k_s accepted) and the expected tail after them
        double em = 1.0, f0 = 1.0, t0 = 0.0, q = 1.0;
        for (int32_t j = 0; j < k_l; ++j) {
            const double al = adapt_alpha(s, j);
            if (j < k_s) {
                f0 *= al;
                em += f0;
            } else {
                q  *= al;
                t0 += q;
            }
        }
        const double w = cx.w0;
        const double f = (cx_f[s]  + w*f0) / (cx_n[s]  + w);
        const double t = (cx_tt[s] + w*t0) / (cx_tn[s] + w);
        e_s = (cx_e[s] + w*em) / (cx_n[s] + w);
        e_l = e_s + f*t;
    }

    void adapt_content_note(int32_t k) {
        if (k != cx_k) {
            cx_n_switch += cx_k > 0 ? 1 : 0;
            cx_k     = k;
            cx_since = 0;
            cx_short = 0;
        }
        if (cx_since < (1 << 30)) {
            cx_since++;   // saturate: cx_k is kept across generations
        }
    }

    // [TAG_DFL_ADAPT_CONTENT] the common length: today's rule first (it keeps its own state in ad_k), then with 1-2
    // drafting sequences and the rule at the long draft, the policy's choice (or a probe step)
    int32_t adapt_content(const common_speculative_draft_params_vec & dparams) {
        const int32_t k_rule = adapt_choose(dparams);
        std::vector<llama_seq_id> ds;
        double depth = 0.0;
        for (llama_seq_id s = 0; s < (llama_seq_id) n_seq; ++s) {
            if (dparams[s].drafting) {
                ds.push_back(s);
                depth += (double) dparams[s].pos0;
            }
        }
        if (ds.empty() || ds.size() > 2 || n_max <= 0) {
            return k_rule;
        }
        const int32_t k_hi = ad.kbyn.empty() ? n_max :
            std::min<int32_t>(n_max, ad.kbyn[std::min(ds.size(), ad.kbyn.size()) - 1]);
        const int32_t k_s  = std::min(ad.short_k, n_max);
        const int32_t k_l  = trained_max > 0 ? std::min(k_hi, trained_max) : k_hi;   // the long draft of the rule
        if (k_l <= k_s) {
            return k_rule;
        }
        cx_klong = k_l;
        bool warm = false;
        for (llama_seq_id s : ds) {
            warm = warm || cx_seen[s] < cx.warm;
        }
        if (warm) {
            adapt_content_note(k_rule);   // the policy starts from the rule
            cx_n_warm++;
            return k_rule;
        }
        if (k_rule != k_l) {
            cx_n_rule++;                  // the rule is an upper bound (duo: deep and a weak first position)
            return k_rule;
        }
        const double n   = (double) ds.size();
        const double dz  = std::min(8.0, depth / n / 32768.0);
        const double c_s = adapt_cost(n, dz, k_s);
        const double c_l = adapt_cost(n, dz, k_l);
        double v_s = 0.0, v_l = 0.0, e0_s = 0.0, e0_l = 0.0;
        for (llama_seq_id s : ds) {
            double e_s = 0.0, e_l = 0.0;
            adapt_content_est(s, k_s, k_l, e_s, e_l);
            if (s == ds[0]) {
                e0_s = e_s;
                e0_l = e_l;
            }
            v_s += ad.fair ? std::log(e_s / c_s) : e_s / c_s;
            v_l += ad.fair ? std::log(e_l / c_l) : e_l / c_l;
        }
        if (ad.fair) {
            v_s = std::exp(v_s / n);
            v_l = std::exp(v_l / n);
        }
        int32_t k = cx_k;
        if (k != k_s && k != k_l) {
            k = v_l >= v_s ? k_l : k_s;
        } else if (cx_since >= cx.dwell) {
            if (k == k_s && v_l > v_s*(1.0 + cx.up)) {
                k = k_l;
            } else if (k == k_l && v_s > v_l*(1.0 + cx.down)) {
                k = k_s;
            }
        }
        adapt_content_note(k);
        bool probe = false;
        if (k == k_s && cx.probe > 0) {
            // `probe` short steps, then one long step
            if (cx_short >= cx.probe) {
                cx_short = 0;
                probe    = true;
            } else {
                cx_short++;
            }
        }
        const int32_t k_out = probe ? k_l : k;
        if (probe) {
            cx_n_probe++;
        } else if (k == k_s) {
            cx_n_short++;
        } else {
            cx_n_long++;
        }
        if (ad.probe && k_out != k_rule) {
            // the rule counted its own length
            ad_hist[std::min<size_t>(k_rule, ad_hist.size() - 1)]--;
            ad_hist[std::min<size_t>(k_out,  ad_hist.size() - 1)]++;
        }
        if (cx.verbose >= 2) {
            fprintf(stderr, "turbo-probe: dft-adapt-content n=%d depth=%.0f k=%d%s since=%d v%d=%.3f v%d=%.3f seq%d E%d=%.2f E%d=%.2f\n",
                    (int) ds.size(), depth / n, k_out, probe ? " (probe)" : "", cx_since, k_s, v_s, k_l, v_l, (int) ds[0],
                    k_s, e0_s, k_l, e0_l);
        }
        return k_out;
    }

    // [TAG_DFL_ADAPT_CONTENT] one realized step (no lookup position) of sequence s: k drafted, a accepted
    void adapt_content_accept(llama_seq_id s, int32_t k, int32_t a) {
        const int32_t k_s = std::min(ad.short_k, n_max);
        if (k < k_s) {
            return;   // a draft cut below k_s: the short draft's outcome is unknown
        }
        cx_n[s]  = cx.decay*cx_n[s] + 1.0f;
        cx_e[s]  = cx.decay*cx_e[s] + (float) (std::min(a, k_s) + 1);
        cx_f[s]  = cx.decay*cx_f[s] + (a >= k_s ? 1.0f : 0.0f);
        cx_tn[s] *= cx.tage;
        cx_tt[s] *= cx.tage;
        if (k > k_s && k == cx_klong && a >= k_s) {
            cx_tn[s] = cx.tdecay*cx_tn[s] + 1.0f;
            cx_tt[s] = cx.tdecay*cx_tt[s] + (float) (a - k_s);
        }
        cx_seen[s]++;
        if (k == k_s || k == cx_klong) {
            const int i = k == k_s ? 0 : 1;
            cx_r_tok[i] += (double) (a + 1);
            cx_r_n[i]++;
        }
    }

    // [TAG_DFL_ADAPT_CONTENT] SPEC_DFT_ADAPT_CONTENT_LOG=1: the step counters since the last report, at generation end
    void print_stats_extra() override {
        if (!cx.on || cx.verbose <= 0) {
            return;
        }
        LOG_INF("%s: [TAG_DFL_ADAPT_CONTENT] steps: warm-up %llu, rule %llu, short %llu, long %llu, probe %llu, switches %llu; "
                "realized tokens/step: short %.2f (%llu steps), long %.2f (%llu steps)\n", __func__,
                (unsigned long long) cx_n_warm, (unsigned long long) cx_n_rule, (unsigned long long) cx_n_short,
                (unsigned long long) cx_n_long, (unsigned long long) cx_n_probe, (unsigned long long) cx_n_switch,
                cx_r_n[0] ? cx_r_tok[0] / (double) cx_r_n[0] : 0.0, (unsigned long long) cx_r_n[0],
                cx_r_n[1] ? cx_r_tok[1] / (double) cx_r_n[1] : 0.0, (unsigned long long) cx_r_n[1]);
        cx_n_warm = cx_n_rule = cx_n_short = cx_n_long = cx_n_probe = cx_n_switch = 0;
        cx_r_tok[0] = cx_r_tok[1] = 0.0;
        cx_r_n[0]   = cx_r_n[1]   = 0;
    }

    // [TAG_4C_DFT_SYNC] a prefill batch: prompt rows still follow, or one sequence has more rows than one
    // verify block (n_max + 1). A sequence id out of range counts as prefill, so the sync stays.
    bool dft_sync_is_prefill(const llama_batch & batch_in) {
        if (n_prefill_after > 0) {
            return true;
        }

        const int32_t n_rows_max = std::max(n_max, n_max_ext) + 1;   // [TAG_DFL_LABD] a lookup tail verifies more rows

        sync_rows_seq.assign(n_seq, 0);
        for (int32_t i = 0; i < batch_in.n_tokens; ++i) {
            const int32_t n_ids = batch_in.n_seq_id ? batch_in.n_seq_id[i] : 1;
            for (int32_t k = 0; k < n_ids; ++k) {
                const llama_seq_id s = batch_in.seq_id ? batch_in.seq_id[i][k] : 0;
                if (s < 0 || s >= (llama_seq_id) n_seq || ++sync_rows_seq[s] > n_rows_max) {
                    return true;
                }
            }
        }

        return false;
    }

    // [TAG_DFL_LABD] Lookup-augmented drafting (HyperQwen "LABD"), SPEC_DFT_LABD=1, default off.
    // The drafter sees a 2048-token window, so when the model copies text that sits in the context (reproduce or edit a
    // file, quote a document) it guesses what the context already holds. Per sequence, a suffix-match index over the
    // prompt and the output (every token the target decoded, by position) finds the longest earlier occurrence of the
    // text that ends at id_last (up to nmax tokens, newest on a tie); its continuation is the lookup draft.
    //   head (the common k positions): the lookup takes over when the match is long (>= nstrong, decided on the history
    //     alone, so the drafter rows of that sequence are skipped) or when it is >= nmin and the drafter's first `agree`
    //     tokens are the lookup's (those positions stay the drafter's, with its own q). Otherwise the drafter's draft.
    //   tail (positions >= k, up to SPEC_DFT_LABD_MAX = 15 drafts = 16 verify rows): lookup tokens only, when exactly ONE
    //     sequence drafts this step (several keep one common k: ragged verify lengths measured 48-97 ms/step), the head
    //     is the lookup's or the drafter agreed on all k, and a copy run is on: the last two steps each accepted a full
    //     short block (>= `full` draft tokens), held for `sticky` steps after that drops.
    // Exact at temp > 0: a lookup position carries the point mass q = 1 on its token, a legal proposal for the residual
    // sampler (accept with p(x), else sample p without x). A position keeps the drafter's q only when the choice to keep
    // it depends on the history and on EARLIER drafter tokens alone, never on its own drafter token; that is why the
    // agreement prefix stays the drafter's. Greedy never reads q. The output never depends on the history being right:
    // a wrong history only gives a worse draft.
    // History: stacking the ngram drafters on DFlash2 (whole drafts replaced, up to 48 tokens) was +52 % on code edits
    // but -5 % on code generation. Here the head changes only on a long or confirmed match, the tail only in a copy run,
    // and `guard` stops history-only take-overs for a sequence whose lookup drafts keep failing at their first token.
    // Cost: rows past the drafter's k need --spec-rs-seq >= SPEC_DFT_LABD_MAX on a hybrid target (automatic unless
    // --spec-rs-seq is given; a lower value caps the tail). Qwen3.8-27B (48 GDN layers), per slot and extra rollback
    // row: 7.9 MiB of VRAM (conv group 5.6 + GDN replay ring 2.3), so 4 slots at 7 -> 15 = +253 MiB; a saved sequence
    // state (context checkpoint, prompt cache, park) carries the whole ring, +18 MiB of host RAM each. The pinned host
    // output buffer grows to the widest verify decode only: 16 rows at one slot (+8 rows x 3.8 MiB with -bs), no change
    // at 4 slots (4 x 8 rows is wider). Host RAM for the index: ~20 B per position and 2 MiB of hash heads per slot.
    struct labd_cfg {
        bool    on        = false;
        int32_t max       = 15;    // SPEC_DFT_LABD_MAX: longest draft (head + lookup tail)
        int32_t nmin      = 6;     // SPEC_DFT_LABD_NMIN: shortest match that takes the head with drafter agreement
        int32_t nmax      = 12;    // SPEC_DFT_LABD_NMAX: matches compare up to this length (longer = equal, newest wins)
        int32_t nstrong   = 8;     // SPEC_DFT_LABD_NSTRONG: a match this long takes the head on the history alone
        int32_t agree     = 2;     // SPEC_DFT_LABD_AGREE: drafter tokens that must agree for nmin <= match < nstrong
                                   //   (0: every match >= nmin takes the head on the history alone)
        int32_t nmin_tail = 4;     // SPEC_DFT_LABD_NMIN_TAIL: shortest match that fills the tail of a running long block
        int32_t longmin   = 6;     // SPEC_DFT_LABD_LONGMIN: shortest match that starts a long block
        bool    adaptive  = true;  // SPEC_DFT_LABD_ADAPTIVE: 1 = long block only in a copy run, 0 = whenever a tail matches
        int32_t sticky    = 3;     // SPEC_DFT_LABD_STICKY: steps the long block is held after the copy-run flag drops
        int32_t full      = 0;     // SPEC_DFT_LABD_FULL: accepted draft tokens of a "full" step (0 = the trained block)
        bool    skip      = true;  // SPEC_DFT_LABD_SKIP: no drafter rows for a sequence whose head the history takes
        float   guard     = 0.5f;  // SPEC_DFT_LABD_GUARD: history-only take-overs need this first-token acceptance (0: off)
        int32_t chain     = 64;    // SPEC_DFT_LABD_CHAIN: candidates checked per hash chain
        bool    probe     = false; // SPEC_DFT_LABD_PROBE=1: counters every 256 draft calls
    };
    labd_cfg lk;

    static constexpr int LABD_HB = 18;   // hash heads per key length: 2^18

    // per sequence: the token history by position (LLAMA_TOKEN_NULL where unknown, e.g. media rows) and a hash-chain
    // index of the n-grams ending at every position < n_idx. Positions are added in order and undone newest first, so
    // every chain is exact: strictly decreasing positions with their current tokens.
    struct labd_seq {
        std::vector<llama_token> tok;
        std::vector<int32_t>     prev_s, prev_l;   // previous position with the same short / long key hash
        std::vector<int32_t>     hash_s, hash_l;   // key hash of each position (-1 = not indexed)
        std::vector<int32_t>     head_s, head_l;   // newest position per key hash
        int32_t n_idx = 0;
        bool    bad   = false;   // rewound below the index outside a new request (context shift): off until begin()
        int32_t run   = 0;       // consecutive full steps
        int32_t hold  = 0;       // long-block steps left after the run drops
        bool    tail  = false;   // the last draft had a lookup tail
        float   g_tr  = 0.0f;    // decayed trials / successes of the first lookup-owned position (guard)
        float   g_ok  = 0.0f;
        int32_t own   = -1;      // first lookup-owned position of the pending draft (-1 = none)
        int32_t mlen  = 0;       // match length of this step
        std::vector<llama_token> cont;   // lookup continuation of this step
    };
    std::vector<labd_seq> lks;
    std::vector<int32_t>  lk_pmin;   // labd_track(): lowest position per sequence in the batch
    int32_t  lk_ks = 0;              // short key: min(nmin, nmin_tail)
    int32_t  lk_kl = 0;              // long key: nstrong (0 = no long table)
    uint64_t lk_n_calls = 0, lk_n_hist = 0, lk_n_agree = 0, lk_n_tail = 0, lk_n_skip = 0, lk_n_tok = 0, lk_n_acc = 0;

    static uint32_t labd_hash(const llama_token * t, int32_t n) {
        uint64_t h = 0;
        for (int32_t i = 0; i < n; ++i) {
            h = (h + (uint32_t) t[i] + 1) * 0x9E3779B97F4A7C15ull;
            h ^= h >> 31;
        }
        return (uint32_t) (h >> (64 - LABD_HB));
    }

    void labd_init() {
        lk.on = spec_labd_on();
        if (!lk.on) {
            return;
        }
        lk.nmin      = std::max(1, spec_labd_env_i("SPEC_DFT_LABD_NMIN",      lk.nmin));
        lk.nmax      = std::min(64, std::max(1, spec_labd_env_i("SPEC_DFT_LABD_NMAX", lk.nmax)));
        lk.nstrong   = std::max(1, spec_labd_env_i("SPEC_DFT_LABD_NSTRONG",   lk.nstrong));
        lk.agree     = std::max(0, spec_labd_env_i("SPEC_DFT_LABD_AGREE",     lk.agree));
        lk.nmin_tail = std::max(1, spec_labd_env_i("SPEC_DFT_LABD_NMIN_TAIL", lk.nmin_tail));
        lk.longmin   = std::max(1, spec_labd_env_i("SPEC_DFT_LABD_LONGMIN",   lk.longmin));
        lk.adaptive  = spec_labd_env_i("SPEC_DFT_LABD_ADAPTIVE", 1) != 0;
        lk.sticky    = std::max(0, spec_labd_env_i("SPEC_DFT_LABD_STICKY",    lk.sticky));
        lk.full      = std::max(0, spec_labd_env_i("SPEC_DFT_LABD_FULL",      lk.full));
        lk.skip      = spec_labd_env_i("SPEC_DFT_LABD_SKIP", 1) != 0;
        lk.guard     = adapt_env_f("SPEC_DFT_LABD_GUARD", lk.guard);
        lk.chain     = std::max(1, spec_labd_env_i("SPEC_DFT_LABD_CHAIN",     lk.chain));
        lk.probe     = spec_labd_env_i("SPEC_DFT_LABD_PROBE", 0) != 0;
        if (lk.full <= 0) {
            lk.full = trained_max > 0 ? trained_max : std::max(1, n_max);
        }
        // matches longer than nmax compare equal, so a threshold above it could never be reached
        lk.nstrong = std::min(lk.nstrong, lk.nmax);
        lk.nmin    = std::min(lk.nmin,    lk.nmax);
        lk.longmin = std::min(lk.longmin, lk.nmax);
        lk_ks = std::min(lk.nmin, lk.nmin_tail);
        lk_kl = lk.nstrong > lk_ks ? lk.nstrong : 0;

        // the tail must fit the target's rollback: a longer draft takes the checkpoint path (a state save every step)
        int32_t mx = common_speculative_labd_n_max(n_max);
        const llama_model * model_tgt = llama_get_model(params.ctx_tgt);
        const uint32_t      n_rs      = llama_n_rs_seq(params.ctx_tgt);
        if (n_rs > 0 && (uint32_t) mx > n_rs) {
            LOG_WRN("%s: [TAG_DFL_LABD] the target rolls back %u tokens (--spec-rs-seq): lookup drafts capped at %d, not %d\n",
                    __func__, n_rs, std::max<int32_t>(n_max, (int32_t) n_rs), mx);
            mx = std::max<int32_t>(n_max, (int32_t) n_rs);
        } else if (n_rs == 0 && (llama_model_is_recurrent(model_tgt) || llama_model_is_hybrid(model_tgt))) {
            LOG_WRN("%s: [TAG_DFL_LABD] recurrent target without partial rollback (--spec-rs-seq 0): no lookup tail\n", __func__);
            mx = n_max;
        }
        lk.max    = std::max(n_max, mx);
        n_max_ext = lk.max > n_max ? lk.max : 0;

        lks.assign(n_seq, labd_seq{});
        LOG_INF("%s: [TAG_DFL_LABD] lookup-augmented drafting on: drafts up to %d (drafter %d), match %d..%d, strong %d, "
                "agree %d, tail %d/%d, %s, sticky %d, full %d, skip %d, guard %.2f, chain %d\n",
                __func__, lk.max, n_max, lk.nmin, lk.nmax, lk.nstrong, lk.agree, lk.nmin_tail, lk.longmin,
                lk.adaptive ? "long block in copy runs" : "long block always", lk.sticky, lk.full, (int) lk.skip,
                lk.guard, lk.chain);
    }

    // the n-gram of length K that ends at e: add to its chain
    static void labd_put(labd_seq & q, int32_t e, int32_t K, std::vector<int32_t> & head, std::vector<int32_t> & prev,
            std::vector<int32_t> & hv) {
        hv[e]   = -1;
        prev[e] = -1;
        if (K <= 0 || e + 1 < K) {
            return;
        }
        for (int32_t i = e - K + 1; i <= e; ++i) {
            if (q.tok[i] == LLAMA_TOKEN_NULL) {
                return;
            }
        }
        const uint32_t h = labd_hash(q.tok.data() + e - K + 1, K);
        hv[e]   = (int32_t) h;
        prev[e] = head[h];
        head[h] = e;
    }

    static void labd_pop(int32_t e, std::vector<int32_t> & head, std::vector<int32_t> & prev, std::vector<int32_t> & hv) {
        if (hv[e] >= 0 && head[hv[e]] == e) {
            head[hv[e]] = prev[e];
        }
        hv[e] = -1;
    }

    // index the n-grams that end at the positions [n_idx, upto): their tokens are final
    void labd_index(labd_seq & q, int32_t upto) {
        upto = std::min<int32_t>(upto, (int32_t) q.tok.size());
        if (upto <= q.n_idx) {
            return;
        }
        if (q.head_s.empty()) {
            q.head_s.assign((size_t) 1 << LABD_HB, -1);
            if (lk_kl > 0) {
                q.head_l.assign((size_t) 1 << LABD_HB, -1);
            }
        }
        if ((int32_t) q.prev_s.size() < upto) {
            const size_t n = std::max<size_t>((size_t) upto, q.tok.capacity());
            q.prev_s.resize(n, -1);
            q.hash_s.resize(n, -1);
            if (lk_kl > 0) {
                q.prev_l.resize(n, -1);
                q.hash_l.resize(n, -1);
            }
        }
        for (int32_t e = q.n_idx; e < upto; ++e) {
            labd_put(q, e, lk_ks, q.head_s, q.prev_s, q.hash_s);
            if (lk_kl > 0) {
                labd_put(q, e, lk_kl, q.head_l, q.prev_l, q.hash_l);
            }
        }
        q.n_idx = upto;
    }

    // undo the index from position m on, newest first
    void labd_rewind(labd_seq & q, int32_t m) {
        m = std::max(0, m);
        for (int32_t e = q.n_idx - 1; e >= m; --e) {
            labd_pop(e, q.head_s, q.prev_s, q.hash_s);
            if (lk_kl > 0) {
                labd_pop(e, q.head_l, q.prev_l, q.hash_l);
            }
        }
        q.n_idx = std::min(q.n_idx, m);
    }

    // [TAG_DFL_LABD] every target batch (prompt ubatches, verify rows) writes its tokens at their positions; a batch
    // that starts at p makes the positions below p final
    void labd_track(const llama_batch & b) {
        if (b.pos == nullptr || b.n_seq_id == nullptr || b.seq_id == nullptr) {
            return;
        }
        lk_pmin.assign(n_seq, INT32_MAX);
        for (int32_t i = 0; i < b.n_tokens; ++i) {
            for (int32_t k = 0; k < b.n_seq_id[i]; ++k) {
                const llama_seq_id s = b.seq_id[i][k];
                if (s >= 0 && s < (llama_seq_id) n_seq && b.pos[i] >= 0) {
                    lk_pmin[s] = std::min<int32_t>(lk_pmin[s], b.pos[i]);
                }
            }
        }
        for (llama_seq_id s = 0; s < (llama_seq_id) n_seq; ++s) {
            if (lk_pmin[s] == INT32_MAX) {
                continue;
            }
            auto & q = lks[s];
            const int32_t p = lk_pmin[s];
            if (p < q.n_idx) {
                // below the verified text: a new request (begin() follows and checks) or a context shift (stays off)
                labd_rewind(q, p);
                q.bad = true;
            }
            q.tok.resize(p, LLAMA_TOKEN_NULL);
            labd_index(q, p);
        }
        for (int32_t i = 0; i < b.n_tokens; ++i) {
            const llama_pos p = b.pos[i];
            if (p < 0) {
                continue;
            }
            for (int32_t k = 0; k < b.n_seq_id[i]; ++k) {
                const llama_seq_id s = b.seq_id[i][k];
                if (s < 0 || s >= (llama_seq_id) n_seq) {
                    continue;
                }
                auto & tok = lks[s].tok;
                if ((int32_t) tok.size() <= p) {
                    tok.resize(p + 1, LLAMA_TOKEN_NULL);
                }
                tok[p] = b.token ? b.token[i] : LLAMA_TOKEN_NULL;   // media rows: unknown
            }
        }
    }

    // [TAG_DFL_LABD] a new generation. The prompt (text tokens) is the reference for a text-only history; this also
    // repairs positions restored from a cache without a target batch.
    void labd_begin(llama_seq_id s, const llama_tokens & prompt) {
        auto & q = lks[s];
        q.run  = 0;
        q.hold = 0;
        q.tail = false;
        q.g_tr = 0.0f;
        q.g_ok = 0.0f;
        q.own  = -1;
        const int32_t n = (int32_t) prompt.size();
        if ((int32_t) q.tok.size() == n) {
            int32_t m = 0;
            while (m < n && q.tok[m] == prompt[m]) {
                ++m;
            }
            if (m < n) {
                labd_rewind(q, m);
                std::copy(prompt.begin() + m, prompt.end(), q.tok.begin() + m);
            }
        } else if ((int32_t) q.tok.size() < n) {
            labd_rewind(q, 0);
            q.tok = prompt;
        }
        // more positions than text tokens: media rows; the positions the target batches wrote are kept
        q.bad = false;
    }

    // length of the common suffix of the text ending at e and the text ending at n - 1 (e < n - 1), at most cap
    static int32_t labd_match(const labd_seq & q, int32_t e, int32_t n, int32_t cap) {
        const llama_token * t = q.tok.data();
        int32_t j = 0;
        while (j < cap && e - j >= 0 && t[e - j] == t[n - 1 - j] && t[e - j] != LLAMA_TOKEN_NULL) {
            ++j;
        }
        return j;
    }

    void labd_walk(const labd_seq & q, const std::vector<int32_t> & head, const std::vector<int32_t> & prev, int32_t K,
            int32_t n, int32_t & best_len, int32_t & best_e) const {
        if (K <= 0 || n - 1 < K || head.empty()) {
            return;
        }
        for (int32_t i = n - K; i < n; ++i) {
            if (q.tok[i] == LLAMA_TOKEN_NULL) {
                return;
            }
        }
        int32_t e     = head[labd_hash(q.tok.data() + n - K, K)];
        int32_t bound = n - 1;
        for (int32_t c = 0; c < lk.chain && e >= 0 && e < bound; ++c) {
            const int32_t len = labd_match(q, e, n, lk.nmax);
            if (len >= K && len > best_len) {   // newest first: a tie keeps the newer one
                best_len = len;
                best_e   = e;
                if (len >= lk.nmax) {
                    break;
                }
            }
            bound = e;
            e     = prev[e];
        }
    }

    // [TAG_DFL_LABD] the lookup of one drafting sequence, before the drafter decode. 0 = no match, 1 = a match the
    // drafter has to confirm (or that can only fill a tail), 2 = the history takes the head
    int8_t labd_find(llama_seq_id s, const common_speculative_draft_params & dp, int32_t k_head) {
        auto & q = lks[s];
        q.mlen = 0;
        q.cont.clear();
        if (q.bad || dp.pos0 < 0 || k_head <= 0) {
            return 0;
        }
        const int32_t p0 = (int32_t) dp.pos0;
        if (p0 < q.n_idx) {
            labd_rewind(q, p0);   // not expected: a batch that rewinds went through labd_track() first
        }
        q.tok.resize(p0, LLAMA_TOKEN_NULL);
        q.tok.push_back(dp.id_last);
        labd_index(q, p0);

        const int32_t n = p0 + 1;
        int32_t best_len = 0;
        int32_t best_e   = -1;
        labd_walk(q, q.head_l, q.prev_l, lk_kl, n, best_len, best_e);
        if (lk_kl == 0 || best_len < lk_kl) {
            labd_walk(q, q.head_s, q.prev_s, lk_ks, n, best_len, best_e);
        }
        if (best_e < 0) {
            return 0;
        }
        for (int32_t j = best_e + 1; j < n && (int32_t) q.cont.size() < lk.max; ++j) {
            if (q.tok[j] == LLAMA_TOKEN_NULL) {
                break;
            }
            q.cont.push_back(q.tok[j]);
        }
        q.mlen = best_len;
        if (q.cont.empty()) {
            return 0;
        }
        const bool hist  = q.mlen >= lk.nstrong || (lk.agree == 0 && q.mlen >= lk.nmin);
        const bool trust = lk.guard <= 0.0f || (q.g_ok + 0.8f*2.0f) / (q.g_tr + 2.0f) >= lk.guard;
        return hist && trust && (int32_t) q.cont.size() >= k_head ? 2 : 1;
    }

    // [TAG_DFL_LABD] the long block for the one drafting sequence: in a copy run (two full steps in a row), then held
    // for `sticky` steps after the run drops
    bool labd_long(labd_seq & q) {
        if (!lk.adaptive) {
            return true;
        }
        if (q.run >= 2) {
            q.hold = lk.sticky;
            return true;
        }
        if (q.hold > 0) {
            q.hold--;
            return true;
        }
        return false;
    }

    // [TAG_DFL_LABD] merge the lookup continuation into one sequence's draft (after the drafter and the common cut)
    void labd_fuse(common_speculative_draft_params & dp, labd_seq & q, int8_t mode, int32_t k_head, bool long_ok) {
        auto & result = *dp.result;
        const bool    want_q = dp.temperature > 0.0f && dp.dists != nullptr;
        const int32_t valid  = (int32_t) q.cont.size();
        q.own = -1;
        if (mode == 0 || valid == 0) {
            q.tail = false;
            return;
        }

        bool    take = false;
        int32_t beg  = 0;
        if (mode == 2) {
            // decided on the history alone: every head position is the lookup's
            result.clear();
            if (dp.dists) {
                dp.dists->clear();
            }
            take = true;
        } else if (lk.agree > 0 && q.mlen >= lk.nmin && valid >= k_head && (int32_t) result.size() >= lk.agree) {
            take = std::equal(result.begin(), result.begin() + lk.agree, q.cont.begin());
            beg  = lk.agree;
        }
        // the drafter's own q covers the draft so far (a sampled DFlash2 draft); otherwise the verifier compares tokens
        const bool q_ok = want_q && dp.dists->size() == result.size();
        auto push = [&](llama_token id) {
            result.push_back(id);
            if (q_ok) {
                common_speculative_token_dist d;
                d.ids   = { id };
                d.probs = { 1.0f };
                dp.dists->push_back(std::move(d));
            }
        };

        if (take && beg < k_head) {
            result.resize(beg);
            if (q_ok) {
                dp.dists->resize(beg);
            }
            for (int32_t j = beg; j < k_head; ++j) {
                push(q.cont[j]);
            }
            q.own = beg;
            if (mode == 2) {
                lk_n_hist++;
            } else {
                lk_n_agree++;
            }
        }

        // the tail: positions the drafter never proposed
        bool agree_all = false;
        if (!take && (int32_t) result.size() == k_head && valid > k_head) {
            agree_all = std::equal(result.begin(), result.end(), q.cont.begin());
        }
        const int32_t n_end = std::min(std::min(lk.max, valid), dp.n_max > 0 ? dp.n_max : lk.max);
        const bool    tail  = long_ok && (take || agree_all) && (int32_t) result.size() == k_head && n_end > k_head &&
                              q.mlen >= lk.nmin_tail && (q.mlen >= lk.longmin || q.tail);
        if (tail) {
            if (q.own < 0) {
                q.own = k_head;
            }
            for (int32_t j = k_head; j < n_end; ++j) {
                push(q.cont[j]);
            }
            lk_n_tail++;
        }
        q.tail = tail;
        if (q.own >= 0) {
            lk_n_tok += (uint64_t) ((int32_t) result.size() - q.own);
        }
    }

    // [TAG_DFL_LABD] after the verify: copy-run state, guard and counters. Returns the drafter's own positions of the
    // step, the only ones its acceptance statistics may count.
    int32_t labd_accept(labd_seq & q, int32_t k, int32_t n_acc) {
        q.run = n_acc >= std::min(k, lk.full) ? q.run + 1 : 0;
        int32_t k_dr = k;
        if (q.own >= 0 && q.own < k) {
            k_dr = q.own;
            if (n_acc >= q.own) {   // the first lookup position was verified
                q.g_tr = 0.9f*q.g_tr + 1.0f;
                q.g_ok = 0.9f*q.g_ok + (n_acc > q.own ? 1.0f : 0.0f);
            }
            lk_n_acc += (uint64_t) std::max(0, std::min(n_acc, k) - q.own);
        }
        q.own = -1;
        return k_dr;
    }

    // [TAG_DFL_ADAPT] [TAG_DFL_LABD] the end of draft(): the common cut, the lookup, what each sequence was given
    void draft_finish(common_speculative_draft_params_vec & dparams, const std::vector<int32_t> & i_block_beg,
            int32_t k_adapt, const std::vector<int8_t> & lk_mode, int32_t n_drafting) {
        if (ad.on) {
            adapt_cut(dparams, i_block_beg, k_adapt);
        }
        if (lk.on) {
            for (llama_seq_id s = 0; s < (llama_seq_id) n_seq; ++s) {
                auto & dp = dparams[s];
                if (!dp.drafting) {
                    continue;
                }
                auto & q = lks[s];
                // only one drafting sequence may run a long block; with several there is no hold (one k per step)
                bool long_ok = false;
                if (n_drafting == 1) {
                    long_ok = labd_long(q);
                } else {
                    q.hold = 0;
                }
                if (i_block_beg[s] < 0 && lk_mode[s] != 2) {
                    continue;
                }
                labd_fuse(dp, q, lk_mode[s], k_adapt, long_ok);
            }
            if (lk.probe && ++lk_n_calls % 256 == 0) {
                fprintf(stderr, "turbo-probe: dft-labd calls=%llu head hist=%llu agree=%llu tail=%llu drafter blocks skipped=%llu "
                        "lookup tokens %llu accepted %llu (%.3f)\n",
                        (unsigned long long) lk_n_calls, (unsigned long long) lk_n_hist, (unsigned long long) lk_n_agree,
                        (unsigned long long) lk_n_tail, (unsigned long long) lk_n_skip, (unsigned long long) lk_n_tok,
                        (unsigned long long) lk_n_acc, lk_n_tok ? (double) lk_n_acc / (double) lk_n_tok : 0.0);
                fflush(stderr);
            }
        }
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_block_beg[seq_id] >= 0 || (lk.on && lk_mode[seq_id] == 2 && dparams[seq_id].drafting)) {
                ad_last[seq_id] = (int32_t) dparams[seq_id].result->size();
            }
        }
    }

    common_speculative_impl_draft_dflash(const common_params_speculative & params, uint32_t n_seq,
            common_speculative_type type = COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH)
        : common_speculative_impl(type, n_seq, params.draft.n_max)
        , params(params.draft)
        , is_dspark(type == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK)
    {
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "DFlash requires ctx_tgt and ctx_dft to be set");

        const llama_model * model_dft = llama_get_model(ctx_dft);
        const llama_model * model_tgt = llama_get_model(ctx_tgt);

        target_layer_ids   = llama_model_target_layer_ids  (model_dft);
        target_layer_ids_n = llama_model_target_layer_ids_n(model_dft);
        GGML_ASSERT(target_layer_ids_n > 0 && "DFlash model has no target_layer_ids");

        n_embd_tgt    = llama_model_n_embd(model_tgt);
        n_embd_dec    = llama_model_n_embd(model_dft);
        n_embd_enc    = (int32_t) target_layer_ids_n * n_embd_tgt;

        // read the trained block size from the dflash.block_size metadata key
        block_size = 16;
        {
            char buf[32] = {};
            if (llama_model_meta_val_str(model_dft, "dflash.block_size", buf, sizeof(buf)) >= 0) {
                block_size = std::atoi(buf);
            }
            if (llama_model_meta_val_str(model_dft, "dflash.sample_from_anchor", buf, sizeof(buf)) >= 0) {
                sample_from_anchor = std::strcmp(buf, "true") == 0;
            }
            if (llama_model_meta_val_str(model_dft, "dflash.attention.causal", buf, sizeof(buf)) >= 0) {
                causal_attn = std::strcmp(buf, "true") == 0;
            }
        }

        selector_top_k = llama_model_dflash_selector_top_k(model_dft);
        is_dflash2     = selector_top_k > 0;
        mask_token_id = llama_vocab_mask(llama_model_get_vocab(model_dft));

        if (is_dspark && this->params.p_min > 0.0f) {
            char buf[16] = {};
            const bool has_conf =
                llama_model_meta_val_str(model_dft, "dflash.has_confidence_head", buf, sizeof(buf)) < 0 ||
                std::strcmp(buf, "true") == 0;
            if (!has_conf) {
                throw std::runtime_error("DSpark draft has no confidence head: please set --spec-draft-p-min 0");
            }
        }

        LOG_INF("%s: adding speculative implementation '%s'\n", __func__, common_speculative_type_to_str(type).c_str());
        LOG_INF("%s: - n_max=%d, n_min=%d, p_min=%.2f\n", __func__, this->params.n_max, this->params.n_min, this->params.p_min);
        LOG_INF("%s: - block_size=%d, mask_token_id=%d, n_extract=%u, sample_from_anchor=%s\n", __func__,
                block_size, mask_token_id, target_layer_ids_n, sample_from_anchor ? "true" : "false");

        // DFlash input is [id_last, <mask> * (block_size-1)]: in-place denoising yields at most
        // block_size-1 draft tokens, anchor-first DSpark yields a full block_size draft tokens
        int32_t n_draft_max = is_dspark && sample_from_anchor ? block_size : block_size - 1;
        // [TAG_DFL_BLOCK_EXT] NEGATIVE RESULT, kept as an opt-in. A DFlash drafter may decode a block longer than it was
        // trained for (up to 2*block_size - 1 draft tokens); the positions past the trained block are extrapolated.
        // Qwen3.8-27B + DFlash2 (block 8), 24 prompts at 1 stream against the adaptive n_max 7: fixed 15 -4.6 % greedy,
        // -22 % sampled; adaptive 3/7/15 +2.9 % greedy (+1.3 % without looping texts), -7.5 % sampled. The single-prompt
        // "+43 %" was a repetitive text that the long block copies.
        trained_max = n_draft_max;
        if (!is_dspark) {
            // opt-in only (SPEC_DFT_BLOCK_EXT=N, then --spec-draft-n-max up to N): on 24 prompts at 1 stream it lost
            const char * e = getenv("SPEC_DFT_BLOCK_EXT");
            const int32_t ext = (e && e[0]) ? std::min(atoi(e), 2*block_size - 1) : 0;
            if (ext > n_draft_max) {
                LOG_WRN("%s: [TAG_DFL_BLOCK_EXT] drafts up to %d tokens, trained block %d\n", __func__, ext, block_size);
                n_draft_max = ext;
            }
        }
        if (this->params.n_max > n_draft_max || this->params.n_min > n_draft_max) {
            LOG_WRN("%s: requested draft size (n_max=%d, n_min=%d) exceeds the trained block size %d -- clamping to %d\n",
                    __func__, this->params.n_max, this->params.n_min, block_size, n_draft_max);
            this->params.n_max = std::min(this->params.n_max, n_draft_max);
            this->params.n_min = std::min(this->params.n_min, n_draft_max);
        }
        this->n_max = this->params.n_max;

        adapt_init(); // [TAG_DFL_ADAPT]
        labd_init();  // [TAG_DFL_LABD]

        batch        = llama_batch_init(llama_n_batch(ctx_dft), 0,          n_seq);
        batch_inject = llama_batch_init(llama_n_ubatch(ctx_dft), n_embd_enc, n_seq);

        // embd batches on an M-RoPE draft need 4 position rows per token
        is_mrope = llama_model_rope_type(model_dft) == LLAMA_ROPE_TYPE_MROPE;
        if (is_mrope) {
            free(batch_inject.pos);
            batch_inject.pos = (llama_pos *) malloc(sizeof(llama_pos) * 4 * llama_n_batch(ctx_dft));
        }

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = is_dflash2 ? selector_top_k : 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(model_dft, sparams));
        }

        selector_rng.resize(n_seq);
        selector_reset.assign(n_seq, true);

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling && !is_dflash2) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        // turn on extraction of the target layers' input embeddings
        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k], true);
        }

        // DFlash2 reads its selector lattice from h_nextn and never consumes raw logits.
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ !is_dflash2);
        llama_set_causal_attn(ctx_dft, causal_attn); // DFlash needs non-causal attention unless the model says otherwise
    }

    ~common_speculative_impl_draft_dflash() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        llama_batch_free(batch);
        llama_batch_free(batch_inject);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }

        selector_reset[seq_id] = true;

        // [TAG_DFL_ADAPT] a new generation starts from the prior
        std::fill(ad_tr.begin() + (size_t) seq_id*n_max, ad_tr.begin() + (size_t) (seq_id + 1)*n_max, 0.0f);
        std::fill(ad_ok.begin() + (size_t) seq_id*n_max, ad_ok.begin() + (size_t) (seq_id + 1)*n_max, 0.0f);
        std::fill(ad_rt.begin() + (size_t) seq_id*(n_max + 1), ad_rt.begin() + (size_t) (seq_id + 1)*(n_max + 1), 0.0f);
        std::fill(ad_rn.begin() + (size_t) seq_id*(n_max + 1), ad_rn.begin() + (size_t) (seq_id + 1)*(n_max + 1), 0.0f);
        ad_last[seq_id] = 0;
        if (cx.on) {
            // [TAG_DFL_ADAPT_CONTENT] the same: the rule decides again until this sequence has `warm` realized steps
            cx_n[seq_id] = cx_e[seq_id] = cx_f[seq_id] = cx_tn[seq_id] = cx_tt[seq_id] = 0.0f;
            cx_seen[seq_id] = 0;
        }

        if (lk.on) {
            labd_begin(seq_id, prompt); // [TAG_DFL_LABD]
        }

        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(params.ctx_dft), seq_id);
        if (pos_max < N - 1) {
            LOG_WRN("%s: ctx_dft pos_max=%d < N-1=%d - process() did not run on every prefill ubatch. "
                    "Drafts may degrade.\n",
                    __func__, (int) pos_max, N - 1);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        // [TAG_DFL_LABD] before any early return: every target batch updates the lookup history
        if (lk.on) {
            labd_track(batch_in);
        }

        // Target prefill may contain token IDs or multimodal embeddings. Both
        // produce the target-layer features used to seed the draft KV cache, so
        // embeddings are injected too, except the pinned ones skipped below.
        // TODO: revisit after https://github.com/ggml-org/llama.cpp/pull/24669 is merged
        // [TAG_SPEC_MEDIA_POS] Media rows are injected only where the drafter can store their
        // positions; the per-sequence skip sits after the [TAG_SPEC_TAIL_PER_ROW] filter below.
        // (Until the upstream sync this rejected EVERY media batch, like the eagle3 and mtp impls.)
        //
        // An embedding batch from the target is always multimodal, and for an M-RoPE target its
        // pos[] is four concatenated planes. Plane 0, the only one read below, is the temporal
        // index, which mtmd sets to the CONSTANT pos_0 for every image row. Injecting those rows
        // writes n_tokens draft KV cells all at pos_0 while the target's sequence advances by
        // mtmd_input_chunk_get_n_pos() = max(nx, ny). Measured on a 20x13 image: the callback
        // delivered 260 rows with pos0 = posN = 59, the drafter's pos_max stopped at 59 while the
        // next text batch arrived at 79, and llama_batch_allocr::init rejected it with
        //   "the last position stored ... is X = 59 ... starting position of Y = 79"
        // so llama_decode returned -1 and the request died with HTTP 500.
        //
        // Only the DRAFTER fails, which is why vision works fine without one: the qwen35 target
        // is IMROPE so n_pos_per_embd() == 4 and it takes the LENIENT position check, while a
        // DFlash drafter is NEOX with n_pos_per_embd() == 1 and takes the STRICT branch that
        // demands Y == X + 1. The message quoted above exists only in that strict branch, which
        // is independent proof the failing context was ctx_dft.
        //
        // The alternative considered was remapping the rows onto the real span
        // (pos_0 + r*n_pos/n_rows) so the drafter still sees image-derived features. It is
        // workable but needs a per-seq side channel, a running row counter across sub-batches,
        // and clamps at both chunk boundaries. Skipping is simpler and measured well: acceptance
        // 51.7% on the image turn and 160.2 t/s on the text turn straight after it.
        //
        // Injecting cannot be made to work here: 260 rows do not fit in 20 positions, and
        // llama_batch_allocr::init requires the distinct positions to be dense. What this leaves
        // behind is a POSITION hole, not a content hole, and it is repaired once per media chunk
        // by the caller - see [TAG_SPEC_MEDIA_POS] in server-context.cpp process_mtmd_chunk().
        const bool has_tokens     = batch_in.token != nullptr;
        const bool has_embeddings = batch_in.embd  != nullptr;
        if (has_tokens == has_embeddings) {
            return true;
        }

        // [TAG_SPEC_PREFILL_TAIL]
        // Skip prompt ubatches that end more than the drafter's sliding window from the
        // end of the prompt: their KV entries are evicted before any draft reads them.
        //
        // This is not free work being skipped - per prompt ubatch this hook runs a full
        // encode AND decode over the drafter (1.9B x 2 flops x n_ubatch x 2 passes, ~49 ms
        // against a ~620 ms ubatch at 124K), pulls the target's hidden states at every
        // dflash.target_layers entry (5 layers x n_ubatch x n_embd x 4 B = ~131 MB per
        // ubatch, GPU->host), and memcpys ~33 MB on the CPU. Measured cost of attaching a
        // drafter is ~13% of prefill throughput (2043 -> 1785 t/s at 124K).
        //
        // Measured on Qwen3.8-27B-UD-Q4_K_XL at ~124K, two independent runs:
        //   prefill 1772.2 -> 1948.3 t/s  (+9.9%)
        //   prefill 1749.4 -> 1924.0 t/s  (+10.0%)
        // i.e. it recovers ~70% of what attaching a drafter costs prefill. Draft
        // acceptance does NOT degrade (25.8% -> 30.2% in one run, 28.6% -> 26.8% in
        // the other) - a drafter reading a holed KV would collapse toward zero, so
        // this also confirms the skip is functionally sound.
        {
            const int32_t tail = spec_prefill_tail();
            (void) tail;
            // >= not >: is_masked_swa masks when p1 - p0 >= n_swa, so a query at P sees
            // keys at P-(n_swa-1)..P. With n_ubatch == n_swa the second-to-last ubatch has
            // n_prefill_after == tail exactly, and every one of its rows is already outside
            // the window by the time any draft reads it - so it was being processed for
            // nothing, doubling the drafter's prefill work at -ub 2048.
            // [TAG_SPEC_PREFILL_TAIL_PER_SEQ] Every sequence in this batch must want the skip, see
            // prefill_batch_wants_skip(). The server asks the same question before the target decode.
            const bool all_want_skip = prefill_batch_wants_skip(batch_in);
            const int  n_seqs_seen   = all_want_skip ? 1 : 0;

            if (all_want_skip && n_seqs_seen > 0) {
                // Skipping is only safe if it cannot leave a HOLE in the draft KV.
                //
                // On a cold prompt the draft cache is empty, so skipping the early ubatches
                // and then feeding only the tail simply starts the draft KV at the tail's
                // first position, which works. But when the server reuses a cached prompt
                // prefix - every follow-up turn in a conversation - the draft cache still
                // holds positions from the PREVIOUS request. Skipping ahead then asks the
                // drafter to decode at a position far past its last one, and the batch is
                // rejected:
                //
                //   the last position stored in the memory module ... for sequence 0 is X = 31120
                //   srv update_slots: decode() failed: failed to process speculative batch
                //
                // which surfaces as HTTP 500 on the request. Dropping the stale draft KV for
                // the sequences in this ubatch makes the reuse case behave exactly like the
                // cold case. Nothing is lost: everything before the tail is outside the
                // drafter's sliding window and would have been evicted before any draft read
                // it, which is the premise this skip rests on to begin with.
                auto * mem_dft = llama_get_memory(this->params.ctx_dft);

                llama_seq_id prev = -1;
                for (int32_t i = 0; i < batch_in.n_tokens; ++i) {
                    if (batch_in.n_seq_id == nullptr || batch_in.n_seq_id[i] <= 0) {
                        continue;
                    }
                    const llama_seq_id s = batch_in.seq_id[i][0];
                    if (s == prev) {
                        continue;
                    }
                    prev = s;
                    if (llama_memory_seq_pos_max(mem_dft, s) >= 0) {
                        llama_memory_seq_rm(mem_dft, s, -1, -1);
                    }
                }

                return true;
            }
        }

        const int32_t n_tokens = batch_in.n_tokens;

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        const int32_t n_ubatch = (int32_t) llama_n_ubatch(ctx_dft);

        // [TAG_SPEC_TAIL_PER_ROW] The skip above fires only when EVERY sequence in the batch is still far
        // from its prompt end. With several agents that almost never holds: a generating slot's verify
        // tokens share the server batch with another agent's prompt chunk, its prefill_after is 0, and
        // the whole chunk - up to n_batch rows - was copied out of the target's layer inputs and injected
        // in n_ubatch pieces, each followed by a synchronize. That is the ~13%-of-prefill drafter cost
        // the tail skip exists to remove, returning in exactly the multi-agent overlap.
        // Decide per sequence instead: sequences that want the skip have their drafter KV wiped once
        // (the state the solo skip leaves) and their rows dropped; every other row is injected exactly
        // as before. The target is untouched, so greedy output cannot change.
        // TURBO_SPEC_TAIL_PER_ROW=0 restores the whole-batch veto.
        static const bool tail_per_row = [] {
            const char * e = getenv("TURBO_SPEC_TAIL_PER_ROW");
            return e == nullptr || atoi(e) != 0;
        }();
        std::vector<int32_t> keep_rows;   // batch indices to inject; empty = every row
        {
            const int32_t tail = spec_prefill_tail();
            if (tail_per_row && tail > 0 && batch_in.token != nullptr && batch_in.embd == nullptr &&
                batch_in.n_seq_id != nullptr) {
                std::vector<int8_t> skip_seq(n_seq, -1);   // -1 not in batch, 0 inject, 1 skip
                bool any_skip = false;
                bool any_keep = false;
                for (int32_t i = 0; i < n_tokens; ++i) {
                    if (batch_in.n_seq_id[i] <= 0) {
                        continue;
                    }
                    const llama_seq_id s = batch_in.seq_id[i][0];
                    if (s < 0 || s >= (llama_seq_id) n_seq || skip_seq[s] >= 0) {
                        continue;
                    }
                    skip_seq[s] = prefill_after_for(s) >= tail ? 1 : 0;
                    any_skip    = any_skip || skip_seq[s] == 1;
                    any_keep    = any_keep || skip_seq[s] == 0;
                }
                if (any_skip && any_keep) {
                    auto * mem_dft = llama_get_memory(ctx_dft);
                    for (llama_seq_id s = 0; s < (llama_seq_id) n_seq; ++s) {
                        if (skip_seq[s] == 1 && llama_memory_seq_pos_max(mem_dft, s) >= 0) {
                            llama_memory_seq_rm(mem_dft, s, -1, -1);
                        }
                    }
                    keep_rows.reserve(n_tokens);
                    for (int32_t i = 0; i < n_tokens; ++i) {
                        if (batch_in.n_seq_id[i] > 0) {
                            const llama_seq_id s = batch_in.seq_id[i][0];
                            if (s >= 0 && s < (llama_seq_id) n_seq && skip_seq[s] == 1) {
                                continue;
                            }
                        }
                        keep_rows.push_back(i);
                    }
                }
            }
        }

        // [TAG_SYNC_SPEC_MEDIA_POS] Upstream injects media and skips only a sequence whose rows are all pinned
        // to one position (an M-RoPE image: a windowed draft cache cannot free cells for it). Kept for M-RoPE
        // drafters. A NEOX drafter takes the strict Y == X + 1 check described at [TAG_SPEC_MEDIA_POS], so it
        // keeps the fork rule for M-RoPE targets (skip every media row; the server repairs the position hole)
        // and injects only dense runs otherwise. Qwen3.8 target + NEOX DFlash2 drafter: unchanged, all skipped.
        if (has_embeddings) {
            const int32_t rope_tgt  = llama_model_rope_type(llama_get_model(ctx_tgt));
            const bool    tgt_mrope = rope_tgt == LLAMA_ROPE_TYPE_MROPE || rope_tgt == LLAMA_ROPE_TYPE_IMROPE;
            std::vector<int32_t> i_seq_beg(n_seq, -1);
            std::vector<int32_t> i_seq_end(n_seq, -1);
            std::vector<int32_t> n_seq_rows(n_seq, 0);
            for (int32_t k = 0; k < n_tokens; ++k) {
                if (batch_in.n_seq_id == nullptr || batch_in.n_seq_id[k] != 1) {
                    continue;
                }
                const llama_seq_id s = batch_in.seq_id[k][0];
                if (s < 0 || s >= (llama_seq_id) n_seq) {
                    continue;
                }
                i_seq_end[s] = k;
                if (i_seq_beg[s] < 0) {
                    i_seq_beg[s] = k;
                }
                n_seq_rows[s]++;
            }
            std::vector<bool> inject_seq(n_seq, false);
            bool any_inject = false;
            for (llama_seq_id s = 0; s < (llama_seq_id) n_seq; ++s) {
                if (i_seq_beg[s] < 0) {
                    continue;
                }
                const llama_pos p_beg      = batch_in.pos[i_seq_beg[s]];
                const llama_pos p_end      = batch_in.pos[i_seq_end[s]];
                const bool      pos_pinned = p_beg == p_end;
                const bool      pos_dense  = p_end - p_beg == n_seq_rows[s] - 1;
                inject_seq[s] = is_mrope ? (n_seq_rows[s] <= 1 || !pos_pinned)
                                         : (!tgt_mrope && pos_dense);
                any_inject    = any_inject || inject_seq[s];
            }
            if (!any_inject) {
                return true;
            }
            keep_rows.clear();
            keep_rows.reserve(n_tokens);
            for (int32_t k = 0; k < n_tokens; ++k) {
                if (batch_in.n_seq_id == nullptr || batch_in.n_seq_id[k] != 1) {
                    continue;
                }
                const llama_seq_id s = batch_in.seq_id[k][0];
                if (s >= 0 && s < (llama_seq_id) n_seq && inject_seq[s]) {
                    keep_rows.push_back(k);
                }
            }
        }

        const int32_t n_rows = keep_rows.empty() ? n_tokens : (int32_t) keep_rows.size();
        const auto row_at = [&keep_rows](int32_t r) -> int32_t { return keep_rows.empty() ? r : keep_rows[r]; };

        // Flatten token-wise encoder work into shared chunks while preserving each row's position and sequence.
        // Upstream 662a0b012 folded the DFlash encoder (fc + norm) into the decoder's embd branch, so the
        // target features go straight into the inject batch and one llama_decode does both jobs. That
        // removes a llama_encode, a device-to-host round trip of its output and a graph build per chunk.
        for (int32_t offset = 0; offset < n_rows; offset += n_ubatch) {
            const int32_t n_chunk = std::min(n_ubatch, n_rows - offset);

            batch_inject.n_tokens = n_chunk;
            for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
                const float * layer = llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k]);
                if (!layer) {
                    GGML_ABORT("DFlash: target layer %d input not extracted.", target_layer_ids[k]);
                }
                for (int32_t i = 0; i < n_chunk; ++i) {
                    float       * dst = batch_inject.embd + (size_t) i * n_embd_enc + k * (size_t) n_embd_tgt;
                    const float * src = layer + (size_t) row_at(offset + i) * n_embd_tgt;
                    std::memcpy(dst, src, (size_t) n_embd_tgt * sizeof(float));
                }
            }

            for (int32_t i = 0; i < n_chunk; ++i) {
                const int32_t j = row_at(offset + i);
                GGML_ASSERT(batch_in.n_seq_id[j] == 1);
                const llama_seq_id seq_id = batch_in.seq_id[j][0];
                GGML_ASSERT(seq_id >= 0 && seq_id < (llama_seq_id) n_seq);
                const llama_pos p = batch_in.pos[j];
                batch_inject.pos[i] = p;
                if (is_mrope) {
                    // embd batches on an M-RoPE draft need 4 position rows per token
                    batch_inject.pos[1 * n_chunk + i] = p;
                    batch_inject.pos[2 * n_chunk + i] = p;
                    batch_inject.pos[3 * n_chunk + i] = 0;
                }
                batch_inject.n_seq_id[i]  = 1;
                batch_inject.seq_id[i][0] = seq_id;
                batch_inject.logits[i]    = false;
            }

            int32_t rc = llama_decode(ctx_dft, batch_inject);
            if (rc != 0) {
                LOG_ERR("%s: llama_decode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                        __func__, rc, (int) n_chunk, (int) offset);
                return false;
            }
            // [TAG_SPEC_DFT_SYNC] The server may switch contexts before the next draft
            // decode, so this drains the drafter queue to be safe.
            //
            // It is also a full pipeline stall, paid once per chunk - and during GENERATION
            // that is once per token step, where a chunk is only the 1-8 accepted tokens.
            // Measured on a real session at ~91K context: a speculative step costs 45.2 ms
            // against ~19 ms for the target forward, so ~26 ms goes to the drafter path -
            // about 12x what streaming the drafter's 1.06 GiB three times would cost. The
            // work is not compute-bound or bandwidth-bound; it is overhead, and a per-step
            // synchronize is the largest single candidate.
            //
            // Gated for measurement rather than removed: the sync is a correctness guard and
            // is only obviously redundant when nothing switches contexts between here and
            // the draft decode, which reads drafter output and synchronizes anyway.
            // SPEC_DFT_SYNC=0 skips it during generation only; prefill always syncs.
            // Default is now AUTO: the sync is only needed when something can switch the
            // drafter context between here and the draft decode, which cannot happen with a
            // single sequence. Prefill always syncs. Measured at d32768, median of three
            // 1500-token samples: 34.00 ms/step with the sync against 33.22 without, -2.3%.
            //   SPEC_DFT_SYNC=1 forces the old always-sync behaviour.
            //   SPEC_DFT_SYNC=0 forces it off even for multiple sequences (measurement only).
            // [TAG_4C_DFT_SYNC] Auto now skips the sync with several live sequences as well. Until the next
            // draft decode, the server touches the drafter context only through host metadata (seq_rm,
            // seq_add, seq_cp, seq_pos_min/max), through state get/set, which synchronize first
            // (llama_state_seq_*_data_ext), and through the next llama_decode(ctx_dft). That decode is queued on
            // the same drafter stream, so it runs after this one, and its host inputs are uploaded before it
            // returns ([TAG_SCHED_INPUT_BATCH]); every output getter synchronizes.
            // Prefill is now decided per sequence (dft_sync_is_prefill): "n_tokens > 8" made every step with
            // 3 or more generating slots a prefill, so 4 slots synced every step whatever SPEC_DFT_SYNC said.
            //   SPEC_DFT_SYNC=0          the pre-4C 0 rule, exactly (still syncs every step at 3+ slots)
            //   SPEC_DFT_SYNC=2          the pre-4C auto rule, exactly (kill switch)
            //   SPEC_DFT_SYNC_PREFILL=0  also skip the sync after prefill chunks (auto only)
            static const int dft_sync_mode = [] {
                const char * e = getenv("SPEC_DFT_SYNC");
                return (e && e[0]) ? atoi(e) : -1;   // -1 = auto
            }();
            static const bool dft_sync_prefill = [] {
                const char * e = getenv("SPEC_DFT_SYNC_PREFILL");
                return e == nullptr || e[0] == '\0' || atoi(e) != 0;
            }();
            bool sync_needed = false;
            if (dft_sync_mode == 1) {
                sync_needed = true;
            } else if (dft_sync_mode == 0) {
                sync_needed = n_prefill_after > 0 || n_tokens > 8;
            } else if (dft_sync_mode == 2) {
                const bool is_prefill = n_prefill_after > 0 || n_tokens > 8;
                // [TAG_SPEC_DFT_SYNC_LIVE] "a single sequence" means one sequence IN THIS BATCH, not a
                // context built for one. n_seq is n_seq_max (= --parallel), so under -np 4 auto mode
                // synced on every step even with one agent generating - the measured -2.3% was never
                // delivered to the deployed config. Count the sequences actually present (a verify
                // batch is at most n_parallel * (1 + n_draft) tokens, so this is a few iterations).
                // Several live sequences keep the conservative sync, exactly as before.
                int n_live_seq = 0;
                {
                    llama_seq_id first = -1;
                    for (int32_t i = 0; i < batch_in.n_tokens && n_live_seq < 2; ++i) {
                        if (batch_in.n_seq_id == nullptr || batch_in.n_seq_id[i] <= 0) {
                            continue;
                        }
                        const llama_seq_id s = batch_in.seq_id[i][0];
                        if (n_live_seq == 0) {
                            first      = s;
                            n_live_seq = 1;
                        } else if (s != first) {
                            n_live_seq = 2;
                        }
                    }
                }
                sync_needed = is_prefill || n_live_seq > 1;
            } else {
                sync_needed = dft_sync_prefill && dft_sync_is_prefill(batch_in);
            }
            // [TAG_4C_PROBE] one count per process() call of <= 64 rows
            if (offset == 0 && n_tokens <= 64 && common_speculative_probe_enabled()) {
                g_spec_dft_sync_calls++;
                g_spec_dft_sync_synced += sync_needed ? 1 : 0;
            }
            if (sync_needed) {
                llama_synchronize(ctx_dft);
            }
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // build one batch holding every drafting sequence's noise block into a single decode)
        // record where each block starts and its size
        std::vector<int32_t> i_block_beg(n_seq, -1);
        std::vector<int32_t> n_block    (n_seq,  0);

        // [TAG_DFL_ADAPT] common length k for this step and the drafter block that serves it
        const int32_t k_adapt = ad.on ? (cx.on ? adapt_content(dparams) : adapt_choose(dparams)) : params.n_max;   // [TAG_DFL_ADAPT_CONTENT]
        const int32_t n_draft = ad.on ? adapt_block(k_adapt)  : params.n_max;

        // [TAG_DFL_LABD] the lookup of every drafting sequence first: a head the history takes needs no drafter rows
        std::vector<int8_t> lk_mode;
        int32_t n_drafting = 0;
        if (lk.on) {
            lk_mode.assign(n_seq, 0);
            for (llama_seq_id s = 0; s < (llama_seq_id) n_seq; ++s) {
                if (dparams[s].drafting) {
                    n_drafting++;
                    lk_mode[s] = labd_find(s, dparams[s], k_adapt);
                }
            }
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }
            if (lk.on && lk.skip && lk_mode[seq_id] == 2) {
                lk_n_skip++;
                continue;
            }

            common_sampler_reset(smpls[seq_id].get());

            const int32_t n = (int32_t) dp.pos0;

            const int32_t n_block_tokens = n_draft + (is_dspark && sample_from_anchor ? 0 : 1);
            i_block_beg[seq_id] = batch.n_tokens;
            n_block    [seq_id] = n_block_tokens;
            for (int32_t i = 0; i < n_block_tokens; ++i) {
                common_batch_add(batch, i == 0 ? dp.id_last : mask_token_id, n + i, { seq_id }, !is_dflash2);
            }
        }

        if (batch.n_tokens == 0) {
            // [TAG_DFL_LABD] every drafting sequence takes its draft from the lookup: no drafter decode
            draft_finish(dparams, i_block_beg, k_adapt, lk_mode, n_drafting);
            return;
        }

        // decode all sequence's noise block in a single batch
        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            LOG_WRN("%s: llama_decode returned %d\n", __func__, ret);
            return;
        }

        // [TAG_SPEC_DFT_DUMP] nullptr unless SPEC_DFT_DUMP is set
        FILE * dump = is_dflash2 ? spec_dft_dump_file() : nullptr;
        if (dump) {
            ++n_dump_call;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_block_beg[seq_id] < 0) {
                continue;
            }
            auto & dp = dparams[seq_id];

            const int32_t beg            = i_block_beg[seq_id];
            const int32_t n_block_tokens = n_block[seq_id];

            auto * smpl = smpls[seq_id].get();

            auto & result = *dp.result;

            if (dp.dists) {
                dp.dists->clear();
            }

            if (is_dflash2) {
                GGML_ASSERT(dp.temperature <= 0.0f || dp.dists);
                const float * lattice = llama_get_embeddings_nextn(ctx_dft);
                GGML_ASSERT(lattice && "DFlash2 selector produced no lattice");

                if (dump) {
                    spec_dft_dump_rows(dump, lattice, n_embd_dec, selector_top_k, n_dump_call,
                            seq_id, dp.pos0, dp.id_last, beg, n_block_tokens);
                }

                if (selector_reset[seq_id]) {
                    uint32_t seed = dp.seed;
                    if (seed == LLAMA_DEFAULT_SEED) {
                        seed = (uint32_t) std::chrono::high_resolution_clock::now().time_since_epoch().count();
                    }
                    selector_rng[seq_id].seed(seed ^ 0x85ebca6bU);
                    selector_reset[seq_id] = false;
                }

                int32_t predecessor = 0;
                for (int32_t i = 1; i < n_block_tokens; ++i) {
                    const float * row = lattice + (size_t) (beg + i) * n_embd_dec;
                    const float * scores = row + selector_top_k + (size_t) predecessor * selector_top_k;

                    if (dp.temperature > 0.0f) {
                        common_speculative_token_dist dist;
                        dist.ids.resize(selector_top_k);
                        dist.probs.resize(selector_top_k);
                        const float max_score = *std::max_element(scores, scores + selector_top_k);
                        float sum = 0.0f;
                        for (int32_t k = 0; k < selector_top_k; ++k) {
                            dist.ids[k] = (llama_token) row[k];
                            dist.probs[k] = std::exp((scores[k] - max_score) / (dp.temperature*dft_qtemp));
                            sum += dist.probs[k];
                        }
                        for (float & p : dist.probs) {
                            p /= sum;
                        }
                        if (dft_qtrunc) {
                            dfl_qtrunc(dist.probs, dp.top_k, dp.top_p, dp.min_p);
                        }
                        std::discrete_distribution<int32_t> sample(dist.probs.begin(), dist.probs.end());
                        predecessor = sample(selector_rng[seq_id]);
                        if (dist.probs[predecessor] < params.p_min) {
                            break;
                        }
                        result.push_back(dist.ids[predecessor]);
                        dp.dists->push_back(std::move(dist));
                    } else {
                        predecessor = (int32_t) std::distance(scores,
                                std::max_element(scores, scores + selector_top_k));
                        // [TAG_SPEC_PMIN] NEGATIVE RESULT. Truncating a low-confidence
                        // draft here sounds like it should pay for itself twice - fewer
                        // tokens to verify, and a narrower batch through attention. It
                        // does not. Measured, Qwen3.8-27B-UD-Q5_K_XL, turbo4 KV, DFlash2
                        // n_max 7, d131072, median of four greedy samples:
                        //
                        //     p_min   ms/step   tok/s
                        //      0.0     43.21    79.7 .. 105.4
                        //      0.4     47.50    98.5
                        //      0.6     49.47    84.0
                        //
                        // Steps get MORE expensive, not less, so leave p_min at 0.
                        // (An early reading of 169 t/s at 0.6 was the model looping:
                        // repeated text drafts at 100% acceptance. Always check the
                        // generated text before believing a speculative benchmark.)
                        if (params.p_min > 0.0f) {
                            // softmax(scores) at the argmax, i.e. 1 / sum(exp(s_k - s_max))
                            float sum = 0.0f;
                            for (int32_t k = 0; k < selector_top_k; ++k) {
                                sum += std::exp(scores[k] - scores[predecessor]);
                            }
                            if (1.0f / sum < params.p_min) {
                                break;
                            }
                        }
                        result.push_back((llama_token) row[predecessor]);
                    }
                }

                if (result.size() < (size_t) params.n_min) {
                    result.clear();
                    if (dp.dists) {
                        dp.dists->clear();
                    }
                }
                continue;
            }

            if (is_dspark) {
                // DSpark: read from the first draft slot, truncate below the confidence threshold
                const float * conf = params.p_min > 0.0f ? llama_get_embeddings_nextn(ctx_dft) : nullptr;
                // bonus-anchor drafts read the mask positions only, like DFlash
                const int32_t i_draft_beg = sample_from_anchor ? 0 : 1;
                for (int32_t i = i_draft_beg; i < n_block_tokens; ++i) {
                    const int32_t idx = beg + i;

                    if (conf && conf[(size_t) idx * n_embd_dec] < params.p_min) {
                        break;
                    }

                    common_sampler_sample(smpl, ctx_dft, idx, true);

                    const auto * cur_p = common_sampler_get_candidates(smpl, true);

                    for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                        LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                                seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                                common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                    }

                    const llama_token id = cur_p->data[0].id;

                    common_sampler_accept(smpl, id, true);

                    result.push_back(id);
                }
            } else {
                // greedily read the predicted block at this sequence's noise positions 1..n_block_tokens-1
                for (int32_t i = 1; i < n_block_tokens; ++i) {
                    common_sampler_sample(smpl, ctx_dft, beg + i, true);

                    const auto * cur_p = common_sampler_get_candidates(smpl, true);

                    for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                        LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                                seq_id, k, i - 1, cur_p->data[k].id, cur_p->data[k].p,
                                common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                    }

                    const llama_token id = cur_p->data[0].id;

                    if (cur_p->data[0].p < params.p_min) {
                        break;
                    }

                    common_sampler_accept(smpl, id, true);

                    result.push_back(id);
                }
            }

            if (result.size() < (size_t) params.n_min) {
                result.clear();
            }
        }

        // [TAG_DFL_ADAPT] one common draft length for this step, then remember what each sequence was given
        draft_finish(dparams, i_block_beg, k_adapt, lk_mode, n_drafting);
    }

    // [TAG_DFL_ADAPT] one Bernoulli trial per drafted position up to the first rejection: n_accepted successes, plus
    // one failure when the draft was cut short. The server may still shorten a draft after draft() (the slot's
    // remaining budget), so the trial count is capped by what was handed out.
    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        if (is_other || seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }
        const int32_t k = ad_last[seq_id];
        ad_last[seq_id] = 0;
        if (k <= 0) {
            return;
        }
        // [TAG_DFL_LABD] a step with lookup positions: the drafter's statistics take its own positions only (k_dr)
        const int32_t k_dr = lk.on ? labd_accept(lks[seq_id], k, n_accepted) : k;
        const int32_t a = std::min<int32_t>(n_accepted, k_dr);
        // [TAG_DFL_ADAPT_REAL] realized tokens of this step (bonus token included) under length k; not for a lookup step
        if (k_dr == k) {
            for (int32_t kk = 0; kk <= n_max; ++kk) {
                const size_t i = (size_t) seq_id*(n_max + 1) + kk;
                ad_rt[i] *= ad.rdecay;
                ad_rn[i] *= ad.rdecay;
            }
            if (k <= n_max) {
                const size_t i = (size_t) seq_id*(n_max + 1) + k;
                ad_rt[i] += (float) (a + 1);
                ad_rn[i] += 1.0f;
            }
            if (cx.on) {
                adapt_content_accept(seq_id, k, a); // [TAG_DFL_ADAPT_CONTENT] the same steps
            }
        }
        for (int32_t j = 0; j < n_max; ++j) {
            const size_t i = (size_t) seq_id*n_max + j;
            const float  t = j < k_dr && j <= a ? 1.0f : 0.0f;   // position j was verified
            const float  o = j < a ? 1.0f : 0.0f;             // and accepted
            ad_tr[i]  = ad.decay*ad_tr[i] + t;
            ad_ok[i]  = ad.decay*ad_ok[i] + o;
            if (t > 0.0f) {
                ad_gtr[j] = ad.gdecay*ad_gtr[j] + t;
                ad_gok[j] = ad.gdecay*ad_gok[j] + o;
            }
        }
    }
};

struct common_speculative_impl_draft_mtp : public common_speculative_impl {
    common_params_speculative_draft params; // reuses the draft-model params slot (ctx_tgt/ctx_dft)

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd = 0;

    // One MTP draft driver, three modes (set once in the ctor):
    //   is_mem_shared (gemma4): shares the target KV, runs all heads in one graph.
    //   chain_heads (step35): n_mtp_layers trained heads, one per draft step.
    //   neither (qwen35 / qwen35moe): a single trained MTP head.
    int32_t n_mtp_layers  = 1;
    bool    is_mem_shared = false;   // gemma4
    bool    chain_heads   = false;   // derived in the ctor: n_mtp_layers > 1 && !is_mem_shared

    // Per-sequence cross-batch carryover: pair (h_p, x_{p+1}) at MTP pos p+1.
    // The last h-row of one process() call needs the first token of the NEXT
    // call to pair with, so it's stashed here until that next call fires.
    std::vector<std::vector<float>> pending_h;   // [n_seq][n_embd]

    std::vector<int32_t> i_batch_beg;
    std::vector<int32_t> i_batch_end;

    // Hidden rows from the most recent target verification batch, grouped by seq.
    // Row 0 corresponds to the sampled token, row N to the Nth accepted draft token.
    std::vector<std::vector<float>> verify_h;
    std::vector<int32_t> verify_h_rows;

    std::vector<int>                i_last;
    std::vector<std::vector<float>> chain_h;

    common_speculative_impl_draft_mtp(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, n_seq, params.draft.n_max)
        , params(params.draft)
    {
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "MTP requires ctx_tgt and ctx_dft to be set");

        n_embd = llama_model_n_embd_out(llama_get_model(ctx_dft));
        GGML_ASSERT(n_embd == llama_model_n_embd_out(llama_get_model(ctx_tgt)) &&
                "MTP input row width must match the target h_nextn width");
        n_mtp_layers = std::max(1, (int) llama_model_n_layer_nextn(llama_get_model(ctx_dft)));

        SPC_TRC("%s", "adding speculative implementation 'draft-mtp'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%.2f, n_embd=%d, backend_sampling=%d\n", this->params.n_max, this->params.n_min, this->params.p_min, n_embd, (int) this->params.backend_sampling);
        SPC_TRC("- gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n",
                this->params.n_gpu_layers,
                ggml_type_name(this->params.cache_type_k),
                ggml_type_name(this->params.cache_type_v),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no",
                common_speculative_get_devices_str(this->params.devices).c_str());

        // [TAG_SPEC_BATCH_FROM_TGT] This batch receives tokens copied out of the TARGET's
        // batch in process(), so it must be sized by what the target can hand us, not by
        // the draft context's own n_batch. SPEC_DFT_UBATCH deliberately clamps the draft
        // context's n_batch/n_ubatch (to shrink its logits buffer, which reserves
        // n_ubatch x n_vocab floats), and that clamp was silently shrinking this
        // allocation too: with SPEC_DFT_UBATCH=256 and a 1024-token prefill ubatch,
        // common_batch_add walked off the end and tripped
        //   GGML_ASSERT(batch.seq_id[batch.n_tokens] && "llama_batch size exceeded")
        // on the first prompt longer than 256 tokens. Short prompts fit, which is why
        // this only showed up once a real prompt was used.
        const uint32_t n_b_dft = llama_n_batch(ctx_dft);
        const uint32_t n_b_tgt = ctx_tgt ? llama_n_batch(ctx_tgt) : 0u;
        const int32_t  n_b     = (int32_t) std::max(n_b_dft, n_b_tgt);
        batch = llama_batch_init(/*n_tokens=*/ n_b, /*embd=*/ n_embd, /*n_seq_max=*/ 1);
        // llama_batch_init allocates only one of token/embd; MTP needs both.
        // TODO: fix, how to call without malloc
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_b);

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);

        is_mem_shared = llama_get_ctx_other(ctx_dft) == ctx_tgt;
        chain_heads   = n_mtp_layers > 1 && !is_mem_shared;

        if (chain_heads) {
            this->params.n_max = std::min(this->params.n_max, n_mtp_layers);

            chain_h.assign(n_seq, {});
            for (auto & c : chain_h) {
                c.reserve((size_t) (this->params.n_max + 1) * n_embd);
            }
        }
        this->n_max = this->params.n_max;

        pending_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));

        i_last.assign(n_seq, -1);
        i_batch_beg.assign(n_seq, -1);
        i_batch_end.assign(n_seq, -1);

        verify_h.assign(n_seq, {});
        verify_h_rows.assign(n_seq, 0);
    }

    ~common_speculative_impl_draft_mtp() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        if (batch.token != nullptr) {
            free(batch.token);
            batch.token = nullptr;
        }
        llama_batch_free(batch);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }

        auto * ctx_dft = this->params.ctx_dft;
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);

        if (pos_max < N - 1 && !is_mem_shared) {
            SPC_WRN("ctx_dft pos_max=%d < N-1=%d - "
                    "process() hook may not have run on every prefill ubatch "
                    "(need_embd / logits=1 on every prompt position?). "
                    "Drafts may degrade.\n",
                    (int) pos_max, N - 1);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        // TODO: how to make it work with vision tokens?
        if (batch_in.token == nullptr || batch_in.embd != nullptr) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // remember the first and last batch index for each sequence
        std::fill(i_batch_beg.begin(), i_batch_beg.end(), -1);
        std::fill(i_batch_end.begin(), i_batch_end.end(), -1);

        for (int k = 0; k < n_tokens; ++k) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                GGML_ASSERT(batch_in.n_seq_id[k] == 1);

                if (batch_in.seq_id[k][0] == seq_id) {
                    i_batch_end[seq_id] = k;
                    if (i_batch_beg[seq_id] < 0) {
                        i_batch_beg[seq_id] = k;
                    }
                }
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        // if kv is shared with target (e.g Gemma4), then we can skip this catch-up decode
        if (!is_mem_shared) {
            // [TAG_MTP_CHUNK_DECODE] This batch is built from the TARGET's ubatch, which can
            // be n_batch(ctx_tgt) wide, but it is decoded on the DRAFT context whose n_batch
            // SPEC_DFT_UBATCH deliberately clamps. Sizing the allocation from the target was
            // only half the fix: llama_decode then asserts n_tokens_all <= cparams.n_batch,
            // so the abort merely moved from common_batch_add to llama_context::decode, and
            // the only way to run was to widen the draft context. That is expensive: its
            // compute buffer reserves a KQ mask of n_kv * n_ubatch f16, which at n_kv 262144
            // is 128 MiB at ubatch 256 but 1024 MiB at 2048.
            //
            // Chunk the decode instead. Positions are strictly increasing and contiguous per
            // sequence, so each chunk's min pos still exceeds the draft KV's pos_max, and
            // the h-shift is expressed in GLOBAL token index so splitting it is transparent.
            auto * mem_dft = llama_get_memory(ctx_dft);
            const float * h_tgt = llama_get_embeddings_nextn(ctx_tgt);
            const int32_t n_b_max = std::max<int32_t>(1, (int32_t) llama_n_batch(ctx_dft));

            bool ok = true;
            for (int head = 0; head < n_mtp_layers && ok; ++head) {
                if (chain_heads) {
                    // ref: https://github.com/ggml-org/llama.cpp/pull/24340/changes#r3413498544
                    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                        if (i_batch_beg[seq_id] < 0) {
                            continue;
                        }
                        llama_memory_seq_rm(mem_dft, seq_id, batch_in.pos[i_batch_beg[seq_id]], -1);
                    }
                    llama_set_nextn_layer_offset(ctx_dft, head);
                }

                for (int32_t off = 0; off < n_tokens && ok; off += n_b_max) {
                    const int32_t n_chunk = std::min<int32_t>(n_b_max, n_tokens - off);

                    common_batch_clear(batch);
                    for (int32_t l = 0; l < n_chunk; ++l) {
                        const int k = off + l;
                        common_batch_add(batch, batch_in.token[k], batch_in.pos[k], { batch_in.seq_id[k][0] }, 0);
                    }

                    // tgt embeddings shifted right by one, in global index space
                    for (int32_t l = 0; l < n_chunk; ++l) {
                        const int k = off + l;
                        if (k >= 1) {
                            std::memcpy(batch.embd + (size_t) l*n_embd, h_tgt + (size_t) (k-1)*n_embd, row_bytes);
                        } else {
                            // row 0 has no predecessor; overwritten below when a sequence starts here
                            std::memset(batch.embd, 0, row_bytes);
                        }
                    }

                    // fill the pending embeddings from a previous run
                    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                        const int idx = i_batch_beg[seq_id];
                        if (idx < 0 || idx < off || idx >= off + n_chunk) {
                            continue;
                        }
                        std::memcpy(batch.embd + (size_t) (idx - off)*n_embd, pending_h[seq_id].data(), row_bytes);
                    }

                    const int32_t rc = llama_decode(ctx_dft, batch);
                    if (rc != 0) {
                        SPC_ERR("llama_decode(ctx_dft) head=%d failed rc=%d (pos=%d off=%d n=%d)\n",
                                head, (int) rc, (int) batch_in.pos[0], (int) off, (int) n_chunk);
                        ok = false;
                        break;
                    }
                }
            }
            if (chain_heads) {
                llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes
            }
            if (!ok) {
                return false;
            }
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_end[seq_id] < 0) {
                continue;
            }

            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;
            verify_h_rows[seq_id] = n_rows;
            verify_h[seq_id].resize((size_t) n_rows * n_embd);

            for (int32_t i = 0; i < n_rows; ++i) {
                const float * h = llama_get_embeddings_nextn_ith(ctx_tgt, i_batch_beg[seq_id] + i);
                std::memcpy(verify_h[seq_id].data() + (size_t) i * n_embd, h, row_bytes);
            }

            std::memcpy(pending_h[seq_id].data(),
                    verify_h[seq_id].data() + (size_t) (n_rows - 1) * n_embd, row_bytes);
        }

        return true;
    }

    // [TAG_MTP_DISTS] one RNG per sequence for sampling the draft from the drafter's own
    // distribution. Seeded deterministically so a run is reproducible; the draft is a PROPOSAL,
    // so this choice cannot affect the output distribution, only which tokens get proposed.
    std::vector<std::mt19937> mtp_dist_rng;

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // [TAG_MTP_DISTS] q is recorded in lockstep with the draft; the server requires
        // dists.size() == draft.size() or it silently uses the weaker exact-match accept rule.
        if (mtp_dist_rng.size() != n_seq) {
            mtp_dist_rng.clear();
            for (uint32_t s = 0; s < n_seq; ++s) {
                mtp_dist_rng.emplace_back(0x9E3779B9u + s);
            }
        }
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (dp.drafting && dp.dists) {
                dp.dists->clear();
            }
        }

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            common_batch_add(batch, dp.id_last, dp.pos0, { seq_id }, true);
            std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, pending_h[seq_id].data(), row_bytes);

            i_last[seq_id] = batch.n_tokens - 1;

            if (chain_heads) {
                chain_h[seq_id].assign(pending_h[seq_id].begin(), pending_h[seq_id].end());
            }
        }

        int i = 0;

        while (n_drafting > 0) {
            // each step decodes under a different head, i.e. a different decoder layer, and
            // KV is per layer. process() filled this layer's KV only for positions < pos0
            // (prompt + accepted prefix) — nothing in the draft region yet. so reset the
            // draft region (the seq_rm lower bound is pos0, leaving the prompt KV intact)
            // and select head i so it rebuilds its own layer's KV there; decoding just the
            // latest token would leave its attention reading cells only another head wrote.
            if (chain_heads) {
                auto * mem_dft = llama_get_memory(ctx_dft);
                for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                    if (drafting[seq_id]) {
                        llama_memory_seq_rm(mem_dft, seq_id, dparams[seq_id].pos0, -1);
                    }
                }
                llama_set_nextn_layer_offset(ctx_dft, i);
            }

            int ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                SPC_ERR("llama_decode[%d] returned %d\n", i, ret);
                break;
            }

            // rebuild the batch for the next step: the growing-KV paths re-add only the
            // new token (the KV already holds the prefix), while chained heads re-add the
            // whole prefix at the next head. dropped sequences are simply not re-added.
            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_last[seq_id], true);
                const float * h_row = llama_get_embeddings_nextn_ith(ctx_dft, i_last[seq_id]);

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                // [TAG_MTP_DISTS] At temp > 0 the target can use rejection sampling - accept with
                // probability p/q - but ONLY if we hand it q, the distribution this draft token was
                // actually sampled from (server-context.cpp:4232 checks dists.size() ==
                // draft.size()). Without it the target silently falls back to exact-match
                // (sampling.cpp:803), which accepts with probability sum p*q instead of
                // sum min(p,q) - far worse on a flat prose distribution.
                //
                // Both halves are required together: recording q while proposing its ARGMAX would
                // make the p/q ratio wrong and silently bias the output. So this branch samples.
                static const bool mtp_dists_on = [] {
                    const char * e = getenv("TURBO_MTP_DISTS");
                    return !(e && e[0] == '0');
                }();

                llama_token id;
                float       id_p;

                if (mtp_dists_on && dp.dists && dp.temperature > 0.0f && cur_p->size > 0) {
                    common_speculative_token_dist dist;
                    const size_t n_cand = cur_p->size;

                    dist.ids.resize(n_cand);
                    dist.probs.resize(n_cand);

                    float sum = 0.0f;
                    for (size_t k = 0; k < n_cand; ++k) {
                        dist.ids[k]   = cur_p->data[k].id;
                        dist.probs[k] = cur_p->data[k].p;
                        sum          += cur_p->data[k].p;
                    }
                    if (sum > 0.0f) {
                        for (float & pr : dist.probs) {
                            pr /= sum;
                        }
                    }

                    std::discrete_distribution<size_t> pick(dist.probs.begin(), dist.probs.end());
                    const size_t sel = pick(mtp_dist_rng[seq_id]);

                    id   = dist.ids[sel];
                    id_p = dist.probs[sel];

                    // [TAG_MTP_DISTS] p_min keeps its ORIGINAL meaning - "is the drafter confident"
                    // - which is a property of the distribution, not of which token we drew from it.
                    // Gating on the sampled token instead would silently disable drafting: the
                    // shipped Flash-Next profile uses mtp_p_min 0.5, and on prose most sampled
                    // tokens sit below that, so MTP would stop after the first draft every time.
                    if (cur_p->data[0].p < params.p_min) {
                        drafting[seq_id] = false;
                        n_drafting--;

                        continue;
                    }

                    dp.dists->push_back(std::move(dist));
                } else {
                    // temp 0, or no dists requested: exact-match is optimal at temp 0 and the
                    // argmax is the right proposal for it.
                    id   = cur_p->data[0].id;
                    id_p = cur_p->data[0].p;

                    // only collect very high-confidence draft tokens
                    if (id_p < params.p_min) {
                        drafting[seq_id] = false;
                        n_drafting--;

                        continue;
                    }
                }

                common_sampler_accept(smpl, id, true);

                result.push_back(id);

                if (params.n_max <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                if (chain_heads) {
                    // ref: https://github.com/ggml-org/llama.cpp/pull/24340#discussion_r3448031546
                    chain_h[seq_id].insert(chain_h[seq_id].end(), h_row, h_row + n_embd);

                    const int n_rows = (int) result.size() + 1; // id_last + tokens drafted so far
                    for (int t = 0; t < n_rows; ++t) {
                        const llama_token tok = (t == 0) ? dp.id_last : result[t - 1];
                        common_batch_add(batch, tok, dp.pos0 + t, { seq_id }, t == n_rows - 1);
                        std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd,
                                    chain_h[seq_id].data() + (size_t) t * n_embd, row_bytes);
                    }
                } else if (is_mem_shared) {
                    // note: with shared memory (e.g. Gemma4 assistants) we use the same position for all draft tokens
                    // ref: https://github.com/huggingface/transformers/blob/effde20942e3f82a1b97449f60b3a48c5ff96145/docs/source/en/model_doc/gemma4_assistant.md?plain=1#L36-L37
                    common_batch_add(batch, id, dp.pos0, { seq_id }, true);
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, h_row, row_bytes);
                } else {
                    common_batch_add(batch, id, dp.pos0 + i + 1, { seq_id }, true);
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, h_row, row_bytes);
                }

                i_last[seq_id] = batch.n_tokens - 1;
            }

            if (batch.n_tokens == 0) {
                break;
            }

            ++i;
        }

        if (chain_heads) {
            llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t n_rows = verify_h_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_h = std::min<int32_t>(n_accepted, n_rows - 1);
        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        std::memcpy(pending_h[seq_id].data(), verify_h[seq_id].data() + (size_t) i_h * n_embd, row_bytes);
    }
};

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_impl_ngram_simple : public common_speculative_impl {
    common_params_speculative_ngram_map params;

    // shared across all sequences
    common_ngram_simple_config config;

    common_speculative_impl_ngram_simple(
            const common_params_speculative & params, uint32_t n_seq,
            common_ngram_simple_config config)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, n_seq, params.ngram_simple.size_m)
        , params(params.ngram_simple)
        , config(config)
    {
        SPC_TRC("%s", "adding speculative implementation 'ngram-simple'\n");
        SPC_TRC("- size_n=%d, size_m=%d, min_hits=%d\n",
                this->params.size_n, this->params.size_m, this->params.min_hits);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            *dp.result = common_ngram_simple_draft(config, *dp.prompt, dp.id_last);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};

struct common_speculative_impl_ngram_map_k : public common_speculative_impl {
    // n_seq configs
    std::vector<common_ngram_map> config;

    common_speculative_impl_ngram_map_k(
            const common_ngram_map & config,
            uint32_t n_seq)
        : common_speculative_impl(config.key_only ? COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K
            : COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, n_seq, config.size_value)
    {
        for (uint32_t i = 0; i < n_seq; i++) {
            this->config.push_back(config);
        }

        SPC_TRC("adding speculative implementation '%s'\n", common_speculative_type_to_str(this->type).c_str());
        SPC_TRC("- size_key=%d, size_value=%d, key_only=%d, min_hits=%d\n",
                config.size_key, config.size_value, config.key_only, config.min_hits);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        GGML_ASSERT(seq_id < (llama_seq_id) n_seq);

        common_ngram_map_begin(config[seq_id], prompt);
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            common_ngram_map_draft(config[seq_id], *dp.prompt, dp.id_last, *dp.result);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        GGML_ASSERT((seq_id < (llama_seq_id) config.size()));

        if (is_other) {
            return;
        }

        common_ngram_map_accept(config[seq_id], n_accepted);
    }
};

struct common_speculative_impl_ngram_mod : public common_speculative_impl {
    common_params_speculative_ngram_mod params;

    // shared across all sequences
    common_ngram_mod mod;

    // enable trace logging if LLAMA_TRACE is set
    const bool verbose;

    struct seq_info {
        // the last position in the prompt that was added to the ngram container
        size_t i_last = 0;

        // length of the last drafted n-gram (number of tokens returned by draft)
        size_t n_draft_last = 0;

        // consecutive accept rounds with low acceptance fraction (< 0.5)
        int n_low = 0;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_mod(
            const common_params_speculative & params,
            uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, n_seq, params.ngram_mod.n_max)
        , params(params.ngram_mod)
        , mod(params.ngram_mod.n_match, 4*1024*1024)
        , verbose(std::getenv("LLAMA_TRACE") != nullptr) {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));

        SPC_TRC("%s", "adding speculative implementation 'ngram-mod'\n");
        SPC_TRC("- n_match=%d, n_max=%d, n_min=%d\n",
                this->params.n_match, this->params.n_max, this->params.n_min);
        SPC_TRC("- mod size=%zu (%.3f MB)\n",
                mod.size(), (float)(mod.size_bytes())/1024/1024);

        if (this->params.n_match < 16) {
            SPC_WRN("ngram_mod n_match=%d is too small - poor quality is possible, "
                    "see: https://github.com/ggml-org/llama.cpp/pull/19164\n", this->params.n_match);
        }

        sinfos.resize(n_seq);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        auto & sinfo = sinfos[seq_id];

        sinfo.i_last = 0;
        sinfo.n_draft_last = 0;

        const size_t n = mod.get_n();
        if (prompt.size() < n) {
            return;
        }

        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }

        sinfo.i_last = prompt.size() - n;

        const double f = (double)mod.get_used() / (double)mod.size();
        SPC_TRC("ngram_mod occupancy = %zu/%zu (%.2f)\n", mod.get_used(), mod.size(), f);

        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            SPC_WRN("ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", f, f_thold);

            mod.reset();
        }
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        sinfo.n_draft_last = 0;

        const size_t cur_len = prompt.size();
        if (cur_len < mod.get_n()) {
            return;
        }

        const size_t n = mod.get_n();

        // add new ngrams in chunks
        if (sinfo.i_last + 32 < cur_len) {
            for (size_t i = sinfo.i_last; i < cur_len - n; ++i) {
                mod.add(prompt.data() + i);
            }

            sinfo.i_last = cur_len - n;
        }

        result.resize(n + params.n_max);
        for (size_t i = 0; i < n - 1; ++i) {
            result[i] = prompt.at(cur_len - n + 1 + i);
        }
        result[n - 1] = dparams.id_last;

        for (int i = 0; i < params.n_max; ++i) {
            const llama_token token = mod.get(result.data() + i);
            if (token == common_ngram_mod::EMPTY) {
                if (i < params.n_min) {
                    result.clear();
                    return;
                }

                result.resize(n + i);
                break;
            }
            result[n + i] = token;
        }

        // only return the m tokens that were drafted
        for (size_t i = 0; n + i < result.size(); ++i) {
            result[i] = result[n + i];
        }
        result.resize(result.size() - n);

        // store length of drafted n-gram for later acceptance analysis
        sinfo.n_draft_last = result.size();
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        if (is_other) {
            return;
        }

        auto & sinfo = sinfos[seq_id];

        // compute acceptance fraction if we have a recorded draft length
        if (sinfo.n_draft_last > 0) {
            const double f_acc = (double)n_accepted / (double)sinfo.n_draft_last;
            if (f_acc < 0.25) {
                sinfo.n_low++;
                if (sinfo.n_low >= 5) {
                    if (verbose) {
                        SPC_TRC("low acceptance streak (%d) - resetting ngram_mod\n", sinfo.n_low);
                    }

                    mod.reset();
                    sinfo.n_low = 0;
                    sinfo.i_last = 0;
                }
            } else {
                sinfo.n_low = 0;
            }
        }
    }
};

struct common_speculative_impl_ngram_cache : public common_speculative_impl {
    common_params_speculative_ngram_cache params;

    uint16_t n_draft;

    bool save_dynamic;
    bool save_static;

    struct seq_info {
        size_t cache_size = 0; // number of tokens in n-gram cache

        common_ngram_cache ngram_cache_context;
        common_ngram_cache ngram_cache_dynamic;
        common_ngram_cache ngram_cache_static;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_cache(
            const common_params_speculative & params,
            uint32_t n_seq,
            uint16_t n_draft,
            const std::string & path_static,
            const std::string & path_dynamic,
            bool save_dynamic,
            bool save_static)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, n_seq, n_draft)
        , params(params.ngram_cache)
        , n_draft(n_draft)
        , save_dynamic(save_dynamic)
        , save_static(save_static)
    {
        SPC_TRC("%s", "adding speculative implementation 'ngram-cache'\n");
        SPC_TRC("- n_draft=%d, cache_static=%s, cache_dynamic=%s\n",
                n_draft,
                path_static.empty() ? "none" : path_static.c_str(),
                path_dynamic.empty() ? "none" : path_dynamic.c_str());

        sinfos.resize(n_seq);

        if (!path_static.empty()) {
            try {
                auto ngram_cache_static = common_ngram_cache_load(path_static);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_static = ngram_cache_static;
                }
            } catch (...) {
                SPC_ERR("failed to open static lookup cache: %s", path_static.c_str());
                GGML_ABORT("Couldn't read static lookup cache");
            }
        }

        if (!path_dynamic.empty()) {
            try {
                auto ngram_cache_dynamic = common_ngram_cache_load(path_dynamic);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_dynamic = ngram_cache_dynamic;
                }
            } catch (...) {
                SPC_ERR("failed to open dynamic lookup cache: %s", path_dynamic.c_str());
                GGML_ABORT("Couldn't read dynamic lookup cache");
            }
        }
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        if (sinfo.cache_size < prompt.size() + 1) {
            llama_tokens tokens_new;
            tokens_new.reserve(prompt.size() + 1 - sinfo.cache_size);
            for (size_t j = sinfo.cache_size; j < prompt.size(); ++j) {
                tokens_new.push_back(prompt[j]);
            }
            tokens_new.push_back(dparams.id_last); // add the last token

            // Update context ngram cache with new dparams.prompt:
            common_ngram_cache_update(
                    sinfo.ngram_cache_context,
                    LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                    tokens_new, tokens_new.size(), false);
            sinfo.cache_size = prompt.size() + 1;
        }

        llama_tokens inp;
        inp.reserve(prompt.size() + 1);
        for (size_t j = 0; j < prompt.size(); ++j) {
            inp.push_back(prompt[j]);
        }
        inp.push_back(dparams.id_last);

        result.push_back(dparams.id_last);

        common_ngram_cache_draft(
                inp, result, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                sinfo.ngram_cache_context,
                sinfo.ngram_cache_dynamic,
                sinfo.ngram_cache_static);

        if (result.size() > 0) {
            // delete first token in result (which is the id_last token)
            result.erase(result.begin());
        }
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};

struct common_speculative {
    common_speculative_draft_params_vec dparams;

    // list of implementations to use and their states
    std::vector<std::unique_ptr<common_speculative_impl>> impls;

    // which implementaion was used for a given seq_id
    std::vector<common_speculative_impl *> impl_last;

    std::vector<double> synth_probs;
};

static common_ngram_map get_common_ngram_map(
        common_speculative_type type,
        const common_params_speculative_ngram_map & config) {
    uint16_t size_key   = config.size_n;
    uint16_t size_value = config.size_m;
    bool     key_only   = type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K;
    uint16_t min_hits   = config.min_hits;

    return common_ngram_map(size_key, size_value, key_only, min_hits);
}

static common_speculative_impl_ngram_cache create_state_ngram_cache(
        const common_speculative_config & config,
        uint32_t n_seq,
        const std::string & path_static,
        const std::string & path_dynamic) {
    uint16_t n_draft = 8; // TODO get from config?

    // TODO bool param in common/common.h to set save_static/save_dynamic?
    bool save_static = false;
    bool save_dynamic = false;

    common_speculative_impl_ngram_cache state(config.params, n_seq, n_draft, path_static, path_dynamic, save_static, save_dynamic);

    return state;
}

std::string common_speculative_type_name_str(const std::vector<common_speculative_type> & types) {
    std::string result;

    for (size_t i = 0; i < types.size(); i++) {
        if (i > 0) {
            result += ",";
        }
        result += common_speculative_type_to_str(types[i]);
    }
    return result;
}

const char * common_speculative_all_types_str() {
    static std::string all_types_str = []() {
        std::vector<common_speculative_type> types;
        types.reserve(COMMON_SPECULATIVE_TYPE_COUNT);
        for (int i = 0; i < COMMON_SPECULATIVE_TYPE_COUNT; i++) {
            types.push_back((common_speculative_type) i);
        }
        return common_speculative_type_name_str(types);
    }();
    return all_types_str.c_str();
}

std::string common_speculative_type_to_str(common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_NONE:          return "none";
        case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:  return "draft-simple";
        case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:  return "draft-eagle3";
        case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:     return "draft-mtp";
        case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH:  return "draft-dflash";
        case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK:  return "draft-dspark";
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:  return "ngram-simple";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:   return "ngram-map-k";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: return "ngram-map-k4v";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:     return "ngram-mod";
        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:   return "ngram-cache";
        default:                                    return "unknown";
    }
}

std::vector<common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names) {
    std::vector<common_speculative_type> types;
    types.reserve(names.size());

    for (const auto & name : names) {
        auto type = common_speculative_type_from_name_map.find(name);
        if (type != common_speculative_type_from_name_map.end()) {
            if (type->second == COMMON_SPECULATIVE_TYPE_NONE) {
                return std::vector<common_speculative_type> { COMMON_SPECULATIVE_TYPE_NONE };
            }
            types.push_back(type->second);
            continue;
        }
        throw std::invalid_argument("unknown speculative type: " + name);
    }

    return types;
}

common_speculative_type common_speculative_type_from_name(const std::string & name) {
    const auto it = common_speculative_type_from_name_map.find(name);
    if (it == common_speculative_type_from_name_map.end()) {
        return COMMON_SPECULATIVE_TYPE_COUNT;
    }
    return it->second;
}

std::vector<common_speculative_type> common_speculative_types_from_gguf(const std::string & path) {
    struct gguf_init_params gguf_params = {
        /* .no_alloc = */ true,
        /* .ctx      = */ nullptr,
    };

    gguf_context_ptr gguf_ctx(gguf_init_from_file(path.c_str(), gguf_params));
    if (!gguf_ctx) {
        return {};
    }

    const int64_t arch_id = gguf_find_key(gguf_ctx.get(), "general.architecture");
    if (arch_id < 0 || gguf_get_kv_type(gguf_ctx.get(), arch_id) != GGUF_TYPE_STRING) {
        return {};
    }

    const std::string arch = gguf_get_val_str(gguf_ctx.get(), arch_id);
    if (arch != "dflash") {
        const uint32_t block_count = gguf_get_val_u32(gguf_ctx.get(), gguf_find_key(gguf_ctx.get(), (arch + ".block_count").c_str()));

        if (gguf_find_tensor(gguf_ctx.get(), ("blk." + std::to_string(block_count - 1) + ".nextn.eh_proj.weight").c_str()) >= 0) {
            return { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        }

        return {};
    }

    // the Markov head distinguishes draft-dspark from draft-dflash
    const auto type = gguf_find_tensor(gguf_ctx.get(), "markov_w1.weight") >= 0
                    ? COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK
                    : COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH;

    SPC_INF("auto-detected speculative type '%s' from the draft model metadata\n", common_speculative_type_to_str(type).c_str());

    return { type };
}

static uint32_t common_get_enabled_speculative_configs(const std::vector<common_speculative_type> & configs) {
    uint32_t result = 0;
    for (size_t i = 0; i < configs.size(); i++) {
        result |= (1u << configs[i]);
    }
    return result;
}

int32_t common_speculative_n_max(const common_params_speculative * spec) {
    int32_t n_max = 0;

    for (const auto type : spec->types) {
        switch (type) {
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:
                n_max = std::max(n_max, std::max(0, spec->draft.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH:
            case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK:
                n_max = std::max(n_max, std::max(0, spec->draft.n_max));
                // [TAG_DFL_LABD] lookup rows past the drafter's block (0 when off)
                n_max = std::max(n_max, common_speculative_labd_n_max(spec->draft.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:
                n_max = std::max(n_max, (int32_t) spec->ngram_simple.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
                n_max = std::max(n_max, (int32_t) spec->ngram_map_k.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V:
                n_max = std::max(n_max, (int32_t) spec->ngram_map_k4v.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:
                n_max = std::max(n_max, std::max(0, spec->ngram_mod.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:
                n_max = std::max(n_max, (int32_t) 8);
                break;
            case COMMON_SPECULATIVE_TYPE_NONE:
            case COMMON_SPECULATIVE_TYPE_COUNT:
                break;
        }
    }

    return n_max;
}

int32_t common_speculative_n_max(const common_speculative * spec) {
    int32_t n_max = 0;

    if (spec == nullptr) {
        return n_max;
    }

    for (const auto & impl : spec->impls) {
        n_max = std::max(n_max, std::max(0, impl->n_max));
        n_max = std::max(n_max, impl->n_max_ext); // [TAG_DFL_LABD]
    }

    return n_max;
}

std::vector<double> common_speculative_synth_rates_resolve(const common_params_speculative * spec, int32_t n_max) {
    const bool has_length = spec->synth_len != -1.0;
    const bool has_rates  = !spec->synth_rates.empty();

    if (!has_length && !has_rates) {
        return {};
    }
    if (has_length && has_rates) {
        throw std::invalid_argument("synthetic acceptance length and rates are mutually exclusive");
    }

    if (n_max <= 0) {
        throw std::invalid_argument("synthetic acceptance requires at least one speculative token");
    }

    if (has_rates) {
        const auto & rates = spec->synth_rates;
        if (rates.size() != (size_t) n_max) {
            throw std::invalid_argument(string_format(
                    "synthetic acceptance rates must contain %d values, got %zu", n_max, rates.size()));
        }

        for (size_t i = 0; i < rates.size(); ++i) {
            if (!std::isfinite(rates[i]) || rates[i] < 0.0 || rates[i] > 1.0) {
                throw std::invalid_argument("synthetic acceptance rates must be finite and within [0, 1]");
            }
            if (i > 0 && rates[i] > rates[i - 1]) {
                throw std::invalid_argument("synthetic acceptance rates must be monotonically non-increasing");
            }
        }

        return rates;
    }

    const double length = spec->synth_len;
    const double length_max = (double) n_max + 1.0;
    if (!std::isfinite(length) || length < 1.0 || length > length_max) {
        throw std::invalid_argument(string_format(
                "synthetic acceptance length must be finite and within [1, %.0f]", length_max));
    }

    double p = 0.0;
    if (length == length_max) {
        p = 1.0;
    } else if (length > 1.0) {
        double p_min = 0.0;
        double p_max = 1.0;
        for (int i = 0; i < 32; ++i) {
            const double p_mid = 0.5 * (p_min + p_max);
            double sum = 0.0;
            double term = p_mid;
            for (int32_t j = 0; j < n_max; ++j) {
                sum += term;
                term *= p_mid;
            }

            if (sum < length - 1.0) {
                p_min = p_mid;
            } else {
                p_max = p_mid;
            }
        }
        p = 0.5 * (p_min + p_max);
    }

    std::vector<double> rates;
    rates.reserve(n_max);
    double rate = p;
    for (int32_t i = 0; i < n_max; ++i) {
        rates.push_back(rate);
        rate *= p;
    }

    return rates;
}

const std::vector<double> & common_speculative_get_synth_probs(const common_speculative * spec) {
    GGML_ASSERT(spec);
    return spec->synth_probs;
}

// [TAG_FN_TURBOT_MTP] general.architecture of a GGUF file (header only, cached per path), "" when it cannot be read
static std::string common_spec_gguf_arch(const std::string & path) {
    static std::mutex                         mutex;
    static std::map<std::string, std::string> cache;

    std::lock_guard<std::mutex> lock(mutex);
    const auto it = cache.find(path);
    if (it != cache.end()) {
        return it->second;
    }

    std::string arch;
    std::error_code ec;
    if (!path.empty() && std::filesystem::is_regular_file(std::filesystem::path(path), ec)) {
        struct gguf_init_params gguf_params = {
            /* .no_alloc = */ true,
            /* .ctx      = */ nullptr,
        };
        gguf_context_ptr gguf_ctx(gguf_init_from_file(path.c_str(), gguf_params));
        if (gguf_ctx) {
            const int64_t id = gguf_find_key(gguf_ctx.get(), "general.architecture");
            if (id >= 0 && gguf_get_kv_type(gguf_ctx.get(), id) == GGUF_TYPE_STRING) {
                arch = gguf_get_val_str(gguf_ctx.get(), id);
            }
        }
    }

    cache[path] = arch;
    return arch;
}

common_params common_base_params_to_speculative(const common_params & params) {
    const bool has_draft = params.speculative.has_dft();

    const auto & params_spec = params.speculative.draft;
    common_params result = params;

    result.embedding    = false;
    result.pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED;

    if (has_draft) {
        // default to global devices value
        if (!params_spec.devices.empty()) {
            result.devices           = params_spec.devices;
        }
        result.model                 = params_spec.mparams;
        result.n_gpu_layers          = params_spec.n_gpu_layers;
        result.tensor_buft_overrides = params_spec.tensor_buft_overrides;

        // a draft pinned to a single device doesn't need the meta wrapper an inherited -sm tensor would give it
        // (the device list is null-terminated, so a single device means size 2)
        const size_t n_devs = std::count_if(params_spec.devices.begin(), params_spec.devices.end(),
                [](ggml_backend_dev_t d) { return d != nullptr; });
        if (n_devs == 1) {
            result.split_mode = LLAMA_SPLIT_MODE_LAYER;
        }

        if (params_spec.cpuparams.n_threads > 0) {
            result.cpuparams.n_threads       = params_spec.cpuparams.n_threads;
            result.cpuparams_batch.n_threads = params_spec.cpuparams_batch.n_threads;
        }
    }

    result.cache_type_k  = params_spec.cache_type_k;
    result.cache_type_v  = params_spec.cache_type_v;

    // [TAG_TURBOT] the draft cache never uses turbot: its plan names the target's attention layers, and DFlash2
    // acceptance was measured with a turbo5p drafter. [TAG_SPEC_KV_INHERIT] copies -ctk turbot into the draft params;
    // this swap removes it again. Every draft context (fit probe, server drafter, speculative-simple) comes through here.
    if (result.cache_type_k == GGML_TYPE_TURBOT_S8 || result.cache_type_v == GGML_TYPE_TURBOT_S8) {
        const bool asked = (params_spec.cache_type_k_set && params_spec.cache_type_k == GGML_TYPE_TURBOT_S8) ||
                           (params_spec.cache_type_v_set && params_spec.cache_type_v == GGML_TYPE_TURBOT_S8);
        // [TAG_FN_TURBOT_MTP] Qwen3.8-Flash-Next (qwen4exp, its MTP head) keeps its KV at q8_0 or better: q8_0 for the
        // draft cache, not turbo5p (turbo5p512 on its 512-value rows). +104 MiB at 262144 cells; it changes only the
        // draft acceptance, never the output. LLAMA_KV_HQ_FALLBACK=0 restores turbo5p, as in llama_context.
        const char * hq_env = getenv("LLAMA_KV_HQ_FALLBACK");
        const bool   hq     = !(hq_env != nullptr && strcmp(hq_env, "0") == 0) &&
                              ((hq_env != nullptr && strcmp(hq_env, "1") == 0) || common_spec_gguf_arch(params.model.path) == "qwen4exp");
        const ggml_type swap = hq ? GGML_TYPE_Q8_0 : GGML_TYPE_TURBO5P_0;
        if (result.cache_type_k == GGML_TYPE_TURBOT_S8) {
            result.cache_type_k = swap;
        }
        if (result.cache_type_v == GGML_TYPE_TURBOT_S8) {
            result.cache_type_v = swap;
        }
        static std::atomic<bool> warned{false};
        if ((asked || hq) && !warned.exchange(true)) {
            if (asked) {
                LOG_WRN("%s: the draft KV cache does not support turbot, using %s for it\n", __func__, ggml_type_name(swap));
            } else {
                LOG_INF("%s: the draft KV cache does not support turbot; this model keeps q8-level KV, using q8_0 for it\n", __func__);
            }
        }
    }

    result.n_outputs_max = params.n_parallel;
    result.n_outputs_max_per_seq = 1;

    // dflash/dspark decode every sequence's full noise block in one pass
    // TODO: refactor such properties to be announced by the speculative types
    //       something like `struct common_speculative_type_props common_speculative_type_get_props(...);`
    const bool has_block_draft = std::any_of(
        params.speculative.types.begin(), params.speculative.types.end(),
        [](common_speculative_type t) {
            return t == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH || t == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK;
        });
    if (has_block_draft) {
        // per-seq output positions: DFlash decodes anchor + n_max masks (n_max + 1); DSpark n_max -> +1 covers both
        const int32_t per_seq = std::max(1, params_spec.n_max + 1);
        result.n_outputs_max = params.n_parallel * per_seq;
        result.n_batch  = std::max(result.n_batch,  result.n_outputs_max);
        result.n_ubatch = std::max(result.n_ubatch, result.n_outputs_max);
        if (params_spec.backend_sampling) {
            result.n_outputs_max_per_seq = per_seq;
        }
    }

    return result;
}

struct common_speculative_init_result::impl {
    impl() = default;
    ~impl() = default;

    // note: the order in which model, context, etc. are declared matters because their destructors will be called bottom-to-top
    llama_model_ptr   model;
    llama_context_ptr context;
};

common_speculative_init_result::common_speculative_init_result(
    common_params & params,
      llama_model * model_tgt,
    llama_context * ctx_tgt) :
    pimpl(new impl{}) {
    const bool has_draft = params.speculative.has_dft();
    const bool spec_mtp = std::find(params.speculative.types.begin(),
                                    params.speculative.types.end(),
                                    COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();

    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);

    // [TAG_SPEC_MTP_SIDECAR] draft-mtp + a draft model is upstream's way to run an MTP head
    // shipped as its own mtp-*.gguf (the ggml-org Nemotron 3.5 repos): -hf <repo> --spec-type
    // draft-mtp wires the sidecar in as the draft model, and -md mtp-*.gguf alone auto-detects
    // draft-mtp (common_speculative_types_from_gguf). The draft context is then an MTP context
    // on the sidecar's weights, which the has_draft branch below builds. What still cannot work
    // is a SECOND drafter beside it: this struct owns one ctx_dft, and common_memory mirrors
    // every sequence operation (seq_rm/seq_cp/seq_add) plus the speculative checkpoint to
    // exactly that one. A draft file that is not an MTP head is refused after loading it.
    if (spec_mtp && has_draft) {
        const bool other_model_draft = std::any_of(
            params.speculative.types.begin(), params.speculative.types.end(),
            [](common_speculative_type t) {
                return t == COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE || t == COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3 ||
                       t == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH || t == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK;
            });
        if (other_model_draft) {
            LOG_ERR("%s: draft-mtp cannot be combined with another draft-model type in --spec-type. "
                    "Both need the single draft context. Pick one: draft-mtp (the target's own MTP head, "
                    "or an mtp-*.gguf head via --spec-draft-model) or the other drafter.\n", __func__);
            return;
        }
    }

    if (spec_mtp) {
        cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    }

    // the draft context holds as many tokens per sequence as the target context
    cparams.n_ctx = llama_n_ctx(ctx_tgt);

    // note: for small models maybe we can set this to the maximum possible draft from all speculative types
    //       the extra memory for small models is likely negligible?
    cparams.n_rs_seq  = 0;
    cparams.ctx_other = ctx_tgt;

    // [TAG_SPEC_DFT_UBATCH] The draft context inherits the target's n_batch/n_ubatch, so its
    // compute buffer reserves logits for a full ubatch: 1024 x 248320 vocab x 4 B is ~970 MiB
    // on this model, for a drafter that only ever needs logits for its 8-token draft block.
    // The drafter DOES encode the prompt in n_ubatch chunks during prefill, so shrinking this
    // trades prefill throughput for VRAM. SPEC_DFT_UBATCH sets it; unset keeps the old
    // behaviour exactly.
    {
        const char * e_du = getenv("SPEC_DFT_UBATCH");
        const int v_du = e_du ? atoi(e_du) : 0;
        if (v_du > 0) {
            if ((uint32_t) v_du < cparams.n_ubatch) { cparams.n_ubatch = (uint32_t) v_du; }
            if ((uint32_t) v_du < cparams.n_batch ) { cparams.n_batch  = (uint32_t) v_du; }
        }
    }

    std::string model_path;
    if (has_draft) {
        model_path = params.speculative.draft.mparams.path;
        LOG_INF("%s: loading draft model '%s'\n", __func__, model_path.c_str());

        llama_model * model_dft = llama_model_load_from_file(params.model.path.c_str(), mparams);
        if (model_dft == NULL) {
            LOG_ERR("%s: failed to load draft model, '%s'\n", __func__, model_path.c_str());
            return;
        }

        pimpl->model.reset(model_dft);

        // [TAG_SPEC_MTP_SIDECAR] with draft-mtp the draft file has to BE an MTP head (mtp-*.gguf);
        // a DFlash / EAGLE3 / plain drafter would only fail inside llama_init_from_model with a
        // generic "doesn't contain MTP layers" warning
        if (spec_mtp && llama_model_n_layer_nextn(model_dft) == 0) {
            LOG_ERR("%s: draft-mtp with --spec-draft-model needs an MTP head file (mtp-*.gguf), but '%s' "
                    "has no MTP layers. Drop --spec-type draft-mtp to use it as its own drafter type, or "
                    "drop --spec-draft-model to use the target's in-model MTP head.\n", __func__, model_path.c_str());
            return;
        }

        llama_context * ctx_dft = llama_init_from_model(model_dft, cparams);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s: failed to create %s context\n", __func__, spec_mtp ? "MTP" : "draft");
            return;
        }

        pimpl->context.reset(ctx_dft);
    } else if (spec_mtp) {
        model_path = params.model.path;

        LOG_INF("%s: creating MTP draft context against the target model '%s'\n", __func__, model_path.c_str());

        llama_context * ctx_dft = llama_init_from_model(model_tgt, cparams);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s: failed to create MTP context\n", __func__);
            return;
        }

        pimpl->context.reset(ctx_dft);
    }
}

common_speculative_init_result::~common_speculative_init_result() = default;

llama_model * common_speculative_init_result::model() {
    return pimpl->model.get();
}

llama_context * common_speculative_init_result::context() {
    return pimpl->context.get();
}

common_speculative_init_result_ptr common_speculative_init_from_params(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt) {
    return std::make_unique<common_speculative_init_result>(params, model_tgt, ctx_tgt);
}

common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft) {
    const int64_t per_seq = 1 + (int64_t) std::max(0, n_draft);
    const int64_t total   = (int64_t) n_parallel * per_seq;

    return {
        /* .total   = */ (int32_t) std::min<int64_t>(n_batch, total),
        /* .per_seq = */ (int32_t) std::min<int64_t>(n_batch, per_seq),
    };
}

// initialization of the speculative decoding system
//
common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq) {
    // Compute the implementations to use based on the config and their order of preference
    std::vector<common_speculative_config> configs = {}; // list of speculative configs to try
    {
        uint32_t enabled_configs = common_get_enabled_speculative_configs(params.types);

        auto add_config_if_enabled = [&](common_speculative_type type, bool available = true) {
            if (available && (enabled_configs & (1u << type))) {
                configs.emplace_back(type, params);
            }
        };

        // when adding a new type - update here the logic above
        static_assert(COMMON_SPECULATIVE_TYPE_COUNT == 11);

        // this list here defines the priority of the speculators
        // the one with highest priority are listed first
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MOD);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE);

        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, params.draft.ctx_dft != nullptr);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_MTP,    params.draft.ctx_dft != nullptr);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH, params.draft.ctx_dft != nullptr);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK, params.draft.ctx_dft != nullptr);
    }

    std::vector<std::unique_ptr<common_speculative_impl>> impls = {};

    for (const common_speculative_config & config : configs) {
        switch (config.type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_simple>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_eagle3>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_mtp>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_dflash>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_dflash>(
                        config.params, n_seq, COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE: {
                common_ngram_map ngram_map = get_common_ngram_map(config.type, config.params.ngram_simple);

                uint16_t ngram_size_key   = ngram_map.size_key;
                uint16_t mgram_size_value = ngram_map.size_value;

                auto config_simple = common_ngram_simple_config {
                    /* .size_ngram = */ ngram_size_key,
                    /* .size_mgram = */ mgram_size_value
                };
                auto state = std::make_unique<common_speculative_impl_ngram_simple>(
                    /* .params = */ config.params,
                    /* .n_seq  = */ n_seq,
                    /* .state  = */ config_simple
                );
                impls.push_back(std::move(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            get_common_ngram_map(config.type, config.params.ngram_map_k), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            get_common_ngram_map(config.type, config.params.ngram_map_k4v), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_mod>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                auto state = create_state_ngram_cache(
                        config, n_seq,
                        params.ngram_cache.lookup_cache_static,
                        params.ngram_cache.lookup_cache_dynamic);
                impls.push_back(std::make_unique<common_speculative_impl_ngram_cache>(state));
                break;
            }
            default:
                break;
        }
    }

    if (impls.empty()) {
        SPC_TRC("%s", "no implementations specified for speculative decoding\n");
        return nullptr;
    }

    common_speculative_ptr result(new common_speculative {
        /* .dparams     = */ common_speculative_draft_params_vec(n_seq),
        /* .impls       = */ std::move(impls),
        /* .impl_last   = */ std::vector<common_speculative_impl *>(n_seq, nullptr),
        /* .synth_probs = */ {},
    });

    const int32_t n_max_configured = common_speculative_n_max(&params);
    const int32_t n_max_effective  = common_speculative_n_max(result.get());
    const auto rates = common_speculative_synth_rates_resolve(&params, n_max_effective);

    std::vector<std::string> rates_str;
    rates_str.reserve(rates.size());
    result->synth_probs.reserve(rates.size());
    double rate_prev = 1.0;
    double acceptance_length = 1.0;
    for (const double rate : rates) {
        result->synth_probs.push_back(rate_prev > 0.0 ? rate / rate_prev : 0.0);
        rates_str.push_back(string_format("%.6g", rate));
        rate_prev = rate;
        acceptance_length += rate;
    }
    if (!result->synth_probs.empty()) {
        SPC_WRN("%s", "synthetic speculative acceptance is enabled for benchmarking; generated output is not valid\n");
        if (n_max_effective != n_max_configured) {
            SPC_WRN("synthetic acceptance draft limit was reduced from %d to %d by the initialized speculative implementations\n",
                    n_max_configured, n_max_effective);
        }
        SPC_INF("synthetic acceptance: n_max = %zu, mean length = %.6f, rates = [%s]\n",
                rates.size(), acceptance_length, string_join(rates_str, ", ").c_str());
    }

    return result.release();
}

void common_speculative_free(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    delete spec;
}

common_speculative_draft_params & common_speculative_get_draft_params(
        common_speculative * spec,
        llama_seq_id seq_id) {
    GGML_ASSERT(spec);
    GGML_ASSERT(seq_id < (llama_seq_id) spec->dparams.size());

    return spec->dparams[seq_id];
}

void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        impl->begin(seq_id, prompt);
        impl->n_call_begin++;
    }
}

void common_speculative_set_prefill_after(common_speculative * spec, int32_t n_after) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        impl->n_prefill_after = n_after;
    }
}

// [TAG_SPEC_PREFILL_TAIL_PER_SEQ] Clear every sequence back to "not prefilling" before the
// server republishes this batch's values. Without the reset a slot that finished its prompt
// would keep a stale positive count and keep being skipped while it generates.
void common_speculative_clear_prefill_after_seq(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        std::fill(impl->n_prefill_after_seq.begin(), impl->n_prefill_after_seq.end(), 0);
    }
}

void common_speculative_set_prefill_after_seq(common_speculative * spec, llama_seq_id seq_id, int32_t n_after) {
    if (spec == nullptr || seq_id < 0) {
        return;
    }

    for (auto & impl : spec->impls) {
        if ((size_t) seq_id >= impl->n_prefill_after_seq.size()) {
            impl->n_prefill_after_seq.resize(seq_id + 1, 0);
        }
        impl->n_prefill_after_seq[seq_id] = n_after;
    }
}

bool common_speculative_prefill_will_skip(const common_speculative * spec, const llama_batch & batch) {
    if (spec == nullptr || spec->impls.empty()) {
        return false;
    }

    for (const auto & impl : spec->impls) {
        if (!impl->prefill_skips_layer_inputs(batch)) {
            return false;
        }
    }

    return true;
}

// [TAG_SPEC_PHASE_PROBE] see speculative.h
static double g_spec_phase_ms[COMMON_SPEC_PHASE_COUNT] = {0};
static unsigned g_spec_phase_steps = 0;

bool common_speculative_probe_enabled() {
    static const bool v = [] {
        const char * e = getenv("SPEC_PHASE_PROBE");
        return e && e[0] == '1';
    }();
    return v;
}

// [TAG_4C_PROBE] Server step accounting, see common_speculative_probe_step_begin() in speculative.h.
// Phases added inside an open step wait in g_spec_step_ms and count only when the step does.
static double   g_spec_step_ms[COMMON_SPEC_PHASE_COUNT] = {0};
static bool     g_spec_step_open    = false;
static bool     g_spec_prev_counted = false;
static double   g_spec_step_gap     = -1.0;   // loop gap before the open step, -1 = none
static double   g_spec_wall_ms      = 0.0;    // wall time of the counted steps
static double   g_spec_gap_ms       = 0.0;    // end of a counted step -> begin of the next counted step
static unsigned g_spec_gap_n        = 0;
static double   g_spec_win_ms[COMMON_SPEC_PHASE_COUNT] = {0};
static double   g_spec_win_wall     = 0.0;
static double   g_spec_win_gap      = 0.0;
static unsigned g_spec_win_gap_n    = 0;
static unsigned g_spec_win_steps    = 0;
static std::chrono::steady_clock::time_point g_spec_step_t0;
static std::chrono::steady_clock::time_point g_spec_step_t1;   // end of the last step

static double spec_probe_ms(std::chrono::steady_clock::duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
}

// SPEC_PHASE_PROBE_EVERY=N prints a report every N counted steps (default 128)
static unsigned spec_probe_every() {
    static const unsigned v = [] {
        const char * e = getenv("SPEC_PHASE_PROBE_EVERY");
        const int n = e ? atoi(e) : 0;
        return n > 0 ? (unsigned) n : 128u;
    }();
    return v;
}

void common_speculative_probe_add(int phase, double ms) {
    if (phase >= 0 && phase < COMMON_SPEC_PHASE_COUNT) {
        (g_spec_step_open ? g_spec_step_ms : g_spec_phase_ms)[phase] += ms;
    }
}

void common_speculative_probe_step_begin() {
    if (!common_speculative_probe_enabled()) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (g_spec_step_open) {
        g_spec_prev_counted = false;   // the last step never ended: no gap across it
    }
    std::fill(g_spec_step_ms, g_spec_step_ms + COMMON_SPEC_PHASE_COUNT, 0.0);
    g_spec_step_gap  = g_spec_prev_counted ? spec_probe_ms(now - g_spec_step_t1) : -1.0;
    g_spec_step_t0   = now;
    g_spec_step_open = true;
}

void common_speculative_probe_step_end(bool gen) {
    if (!common_speculative_probe_enabled() || !g_spec_step_open) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    g_spec_step_open    = false;
    g_spec_step_t1      = now;
    g_spec_prev_counted = gen;
    if (!gen) {
        return;
    }

    const double wall = spec_probe_ms(now - g_spec_step_t0);
    for (int i = 0; i < COMMON_SPEC_PHASE_COUNT; ++i) {
        g_spec_phase_ms[i] += g_spec_step_ms[i];
        g_spec_win_ms[i]   += g_spec_step_ms[i];
    }
    g_spec_wall_ms  += wall;
    g_spec_win_wall += wall;
    if (g_spec_step_gap >= 0.0) {
        g_spec_gap_ms  += g_spec_step_gap;
        g_spec_win_gap += g_spec_step_gap;
        g_spec_gap_n++;
        g_spec_win_gap_n++;
    }
    g_spec_phase_steps++;
    if (++g_spec_win_steps < spec_probe_every()) {
        return;
    }

    // the first six lines keep the old format: cumulative phases, % of their sum
    const char * nm[COMMON_SPEC_PHASE_COUNT] = {"tgt_decode","spec_process","spec_draft","spec_accept","sample"};
    const double n = (double) g_spec_phase_steps;
    double tot = 0.0;
    for (int i = 0; i < COMMON_SPEC_PHASE_COUNT; ++i) tot += g_spec_phase_ms[i];
    fprintf(stderr, "turbo-probe: spec-phase over %u steps (%.2f ms/step total)\n", g_spec_phase_steps, tot / n);
    for (int i = 0; i < COMMON_SPEC_PHASE_COUNT; ++i) {
        fprintf(stderr, "turbo-probe:   %-12s %8.3f ms/step  %5.1f%%\n",
                nm[i], g_spec_phase_ms[i] / n, tot > 0.0 ? 100.0 * g_spec_phase_ms[i] / tot : 0.0);
    }
    fprintf(stderr, "turbo-probe:   %-12s %8.3f ms/step  %5.1f%% of step wall %.3f ms/step, loop_gap %.3f ms/step\n",
            "host_other", (g_spec_wall_ms - tot) / n,
            g_spec_wall_ms > 0.0 ? 100.0 * (g_spec_wall_ms - tot) / g_spec_wall_ms : 0.0,
            g_spec_wall_ms / n, g_spec_gap_n ? g_spec_gap_ms / g_spec_gap_n : 0.0);

    // the last window alone, so a run that changes the load between reports can be split
    const double w = (double) g_spec_win_steps;
    double win_tot = 0.0;
    for (int i = 0; i < COMMON_SPEC_PHASE_COUNT; ++i) win_tot += g_spec_win_ms[i];
    fprintf(stderr, "turbo-probe:   window %u steps: wall %.3f tgt %.3f proc %.3f draft %.3f accept %.3f smp %.3f "
            "host_other %.3f loop_gap %.3f ms/step\n", g_spec_win_steps, g_spec_win_wall / w,
            g_spec_win_ms[COMMON_SPEC_PHASE_TGT_DECODE] / w, g_spec_win_ms[COMMON_SPEC_PHASE_PROCESS] / w,
            g_spec_win_ms[COMMON_SPEC_PHASE_DRAFT] / w, g_spec_win_ms[COMMON_SPEC_PHASE_ACCEPT] / w,
            g_spec_win_ms[COMMON_SPEC_PHASE_SAMPLE] / w, (g_spec_win_wall - win_tot) / w,
            g_spec_win_gap_n ? g_spec_win_gap / g_spec_win_gap_n : 0.0);
    fprintf(stderr, "turbo-probe:   dft_sync %" PRIu64 " of %" PRIu64 " process calls of <= 64 rows synced\n",
            g_spec_dft_sync_synced, g_spec_dft_sync_calls);
    fflush(stderr);

    std::fill(g_spec_win_ms, g_spec_win_ms + COMMON_SPEC_PHASE_COUNT, 0.0);
    g_spec_win_wall  = 0.0;
    g_spec_win_gap   = 0.0;
    g_spec_win_gap_n = 0;
    g_spec_win_steps = 0;
}

void common_speculative_probe_step() {
    if (!common_speculative_probe_enabled()) {
        return;
    }
    if (++g_spec_phase_steps % 128u != 0u) {
        return;
    }
    const char * nm[COMMON_SPEC_PHASE_COUNT] = {"tgt_decode","spec_process","spec_draft","spec_accept","sample"};
    double tot = 0.0;
    for (int i = 0; i < COMMON_SPEC_PHASE_COUNT; ++i) tot += g_spec_phase_ms[i];
    fprintf(stderr, "turbo-probe: spec-phase over %u steps (%.2f ms/step total)\n",
            g_spec_phase_steps, tot / g_spec_phase_steps);
    for (int i = 0; i < COMMON_SPEC_PHASE_COUNT; ++i) {
        fprintf(stderr, "turbo-probe:   %-12s %8.3f ms/step  %5.1f%%\n",
                nm[i], g_spec_phase_ms[i] / g_spec_phase_steps,
                tot > 0.0 ? 100.0 * g_spec_phase_ms[i] / tot : 0.0);
    }
    fflush(stderr);
}

namespace {
struct spec_phase_timer {
    int phase; std::chrono::steady_clock::time_point t0; bool on;
    explicit spec_phase_timer(int p) : phase(p), on(common_speculative_probe_enabled()) {
        if (on) t0 = std::chrono::steady_clock::now();
    }
    ~spec_phase_timer() {
        if (on) common_speculative_probe_add(phase,
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
};
}

bool common_speculative_process(common_speculative * spec, const llama_batch & batch) {
    bool result = true;

    if (spec == nullptr) {
        return result;
    }

    // Count only generation-sized batches: this runs after every llama_decode,
    // prefill ubatches included, which would otherwise be averaged into the
    // per-step figure and made this phase look like it scales with context.
    spec_phase_timer tm_probe(batch.n_tokens <= 64 ? COMMON_SPEC_PHASE_PROCESS : -1);

    for (auto & impl : spec->impls) {
        result = result && impl->process(batch);
    }

    return result;
}

bool common_speculative_wants_prompt(const common_speculative * spec) {
    if (!spec) {
        return false;
    }
    for (const auto & impl : spec->impls) {
        switch (impl->type) {
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V:
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:
                return true;
            default:
                break;
        }
    }
    return false;
}

void common_speculative_draft(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    spec_phase_timer tm_probe(COMMON_SPEC_PHASE_DRAFT);

    auto & dparams = spec->dparams;

    {
        int n_drafting = 0;

        for (auto & dp : dparams) {
            GGML_ASSERT(!dp.drafting || dp.result->empty());

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            return;
        }
    }

    for (auto & impl : spec->impls) {
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(dparams);
            impl->n_call_draft++;
        }

        int n_drafting = 0;

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            auto & result = *dp.result;

            // a new draft has been sampled
            if (dp.drafting && !result.empty()) {
                dp.drafting = false;

                if (dp.n_max > 0) {
                    if (!result.empty() && (int) result.size() > dp.n_max) {
                        SPC_DBG("truncating draft to %d tokens\n", dp.n_max);
                        result.resize(dp.n_max);

                        // [TAG_SPEC_DISTS_TRUNC] dists is filled in lockstep with the draft, so it
                        // must be truncated with it. The server gates the residual rejection sampler
                        // on dists.size() == draft.size() and silently falls back to plain exact-match
                        // verification when they disagree, which is a different and strictly worse
                        // acceptance rule at temperature > 0. Fires whenever get_n_draft_max() drops
                        // below n_max: the last few tokens of any context-full or capped generation.
                        if (dp.dists && dp.dists->size() > result.size()) {
                            dp.dists->resize(result.size());
                        }
                    }
                }

                if (!result.empty()) {
                    SPC_DBG("called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n",
                            common_speculative_type_to_str(impl.get()->type).c_str(), dp.prompt->size(),
                            impl.get()->n_call_draft, result.size());

                    // remember which implementation was used
                    spec->impl_last[seq_id] = impl.get();

                    impl->n_gen_drafts++;
                    impl->n_gen_tokens += result.size();
                }
            }

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            break;
        }
    }

    // these sequences failed to generate a draft
    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
        auto & dp = dparams[seq_id];

        if (dp.drafting) {
            dp.drafting = false;
        }
    }
}

void common_speculative_accept(common_speculative * spec, llama_seq_id seq_id, uint16_t n_accepted) {
    common_speculative_impl * impl = spec->impl_last[seq_id];

    if (impl == nullptr) {
        GGML_ASSERT(n_accepted == 0);
        return;
    }

    {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);

        if (impl->n_acc_tokens_per_pos.size() < n_accepted) {
            impl->n_acc_tokens_per_pos.resize(n_accepted, 0);
        }

        for (size_t i = 0; i < n_accepted; ++i) {
            impl->n_acc_tokens_per_pos[i]++;
        }

        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;
        }

        impl->accept(seq_id, n_accepted, false);
        impl->n_call_accept++;
    }

    // accept with the rest of the implementations, using is_other == true
    for (auto & impl_other : spec->impls) {
        if (impl_other.get() != impl) {
            impl_other->accept(seq_id, n_accepted, true);
        }
    }
}

// TODO: support the case of more than one speculative implementations having a state
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data) {
    if (spec == nullptr) {
        return false;
    }

    for (auto & impl : spec->impls) {
        if (impl->get_state(seq_id, data)) {
            return true;
        }
    }

    return false;
}

void common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        impl->set_state(seq_id, data);
    }
}

void common_speculative_print_stats(const common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    for (const auto & impl : spec->impls) {
        std::string str_perf;
        if (impl->gen_perf) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << impl->t_begin_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_draft_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_accept_us / 1000.0;
            str_perf = ", dur(b,g,a) = " + oss.str() + " ms";
        } else {
            str_perf = "";
        }

        std::string str_stats;
        if (impl->n_call_accept > 0) {
            const double mean =
                1.0 + (double) impl->n_acc_tokens / (double) impl->n_call_accept;
            std::ostringstream tmp;
            tmp << std::fixed << std::setprecision(3);
            for (size_t i = 0; i < impl->n_acc_tokens_per_pos.size(); ++i) {
                if (i > 0) {
                    tmp << ", ";
                }
                tmp << (double) impl->n_acc_tokens_per_pos[i] / (double) impl->n_call_accept;
            }
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << mean;
            str_stats = ", #mean acc len = " + oss.str() + ", #acc rate/pos = (" + tmp.str() + ")";
        }

        SPC_TRC("statistics %16s: #calls(b,g,a) = %4zu %6zu %6zu, #gen drafts = %6zu, #acc drafts = %5zu, #gen tokens = %6zu, #acc tokens = %5zu%s%s\n",
                common_speculative_type_to_str(impl->type).c_str(),
                impl->n_call_begin, impl->n_call_draft, impl->n_call_accept,
                impl->n_gen_drafts,
                impl->n_acc_drafts,
                impl->n_gen_tokens,
                impl->n_acc_tokens,
                str_stats.c_str(),
                str_perf.c_str());

        impl->print_stats_extra(); // [TAG_DFL_ADAPT_CONTENT]
    }
}
