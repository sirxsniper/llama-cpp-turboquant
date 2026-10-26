#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"

#include <cstdint>
#include <map>
#include <vector>

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

// the backend buffers that a state write of this memory would touch - empty caches report none
LLAMA_API std::vector<ggml_backend_buffer_t> llama_memory_get_buffers(const llama_memory_t mem);

// Turn the KV cache attention rotation on or off for the contexts created from this model.
// Useful for testing purposes.
LLAMA_API void llama_model_set_attn_rot(struct llama_model * model, bool value);

// Set whether the context outputs nextn embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);

// Select which appended NextN block the DECODER_MTP graph runs (offset past
// the trunk: il = n_layer() + offset). Used by the speculative NextN driver to
// chain multiple trained NextN heads. Default 0 (first head).
LLAMA_API void llama_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset);

// Marks the entries that a decision head (clef, vllm-decision2) reads, the default is 0
// See https://github.com/ggml-org/llama.cpp/pull/29831 for details
// A run of entries with the same value is one span, spans must be separated by entries with value 0
// An option belongs to the last question before it
enum llama_decision_order {
    LLAMA_DECISION_ORDER_NONE            = 0, // not read by the head
    LLAMA_DECISION_ORDER_QUESTION_NOUL   = 1, // text of a question
    LLAMA_DECISION_ORDER_QUESTION_CHOICE = 2,
    LLAMA_DECISION_ORDER_QUESTION_SCORE  = 3,
    LLAMA_DECISION_ORDER_OPTION          = 4, // text of an option
};
// The embeddings output has one value per entry: row i is the score of option i
LLAMA_API bool llama_batch_ext_set_decision_order(struct llama_batch_ext * batch, int32_t idx, enum llama_decision_order order);

// Label of option i of a decision model ("<arch>.decision.labels"), nullptr past the last one
LLAMA_API const char * llama_model_decision_label(const struct llama_model * model, uint32_t i);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);

// Set whether the context outputs the input embeddings of a specific layer
LLAMA_API void llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool value);

// [TAG_SPEC_PREFILL_TAIL_EXTRACT] Enable or disable the per-decode copy of the enabled layer inputs
// to the host. The layers stay enabled (the graph does not change), only the GPU-to-host copy is
// skipped while this is false. Default true.
LLAMA_API void llama_set_layer_inp_extract(struct llama_context * ctx, bool enable);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);

//
// model/context data extraction
//

LLAMA_API int32_t llama_model_dflash_selector_top_k(const struct llama_model * model);

// returns pointer to the target-model layer indices
LLAMA_API const int32_t * llama_model_target_layer_ids  (const struct llama_model * model);
// returns the number of extracted layers from target model
LLAMA_API uint32_t        llama_model_target_layer_ids_n(const struct llama_model * model);

// [TAG_BS_LAZY_GRAMMAR] true while `smpl` is a lazy grammar sampler that has not yet seen its trigger. Until then its
// apply() is a no-op (llama_grammar_apply_impl returns immediately), so sampling that ignores it is exact.
// false for any other sampler, for non-lazy grammars, for llguidance grammars, and after the trigger.
LLAMA_API bool llama_sampler_grammar_awaiting_trigger(const struct llama_sampler * smpl);

// [TAG_BS_LAZY_GRAMMAR] Undo the offload state that llama_set_sampler() put on a sampler chain, for a chain the context
// no longer holds. A chain's CPU apply() skips its leading samplers whenever the chain was backend-initialized, because
// in that mode the backend already ran them. Dropping the chain with llama_set_sampler(ctx, seq, nullptr) left that
// state set, so every later CPU sample skipped top-k, top-p, temperature and dist and never selected a token. After this
// call the chain samples on the CPU exactly as a chain that was never offloaded, down to one RNG draw per token (a
// multi-output dist ends its backend draw transaction, so a seeded request stays reproducible against pure CPU sampling
// after the switch). Call it only after the context dropped
// the chain, and do not offload the same chain again. No-op for anything that is not a sampler chain.
LLAMA_API void llama_sampler_chain_backend_detach(struct llama_sampler * smpl);

// [TAG_POOL_PREEMPT] Empty attention KV cells in the stream that seq_id writes to (the whole pool with --kv-unified).
// Exact for llama_kv_cache and for the attention half of llama_memory_hybrid on models without SWA, because prepare()
// places every ubatch row in any empty cell of the stream. -1 for any other memory type.
LLAMA_API int32_t llama_memory_attn_n_free_ext(struct llama_context * ctx, llama_seq_id seq_id);

// [TAG_V0262_IDLELAZY] What llama_memory_attn_n_free_ext would return after removing every sequence in seqs (n_seqs ids,
// all writing to one stream, as with --kv-unified): the empty cells plus the cells that only those sequences hold. A cell
// another sequence also holds (seq_cp: a parent and its children, an attach donor, a shared prefix) stays used. One pass
// over the stream's cells. -1 for any other memory type or an invalid id.
LLAMA_API int32_t llama_memory_attn_n_free_without_ext(struct llama_context * ctx, const llama_seq_id * seqs, int32_t n_seqs);

// [TAG_FN_MTP_ATTN_WINDOW] The sliding window of the context's attention cache (llama_kv_cache, or the attention half of
// llama_memory_hybrid), 0 for none or for any other memory type. A qwen4exp MTP draft context gets one from
// LLAMA_MTP_ATTN_WINDOW (llama-model.cpp).
LLAMA_API uint32_t llama_memory_attn_swa_ext(struct llama_context * ctx);

// [TAG_FN_L3_MTP_COST2] Cold-expert counts of the MoE bridge (LLAMA_MOE_BRIDGE=1) of ctx, for the MTP draft-length
// policy. llama_moe_bridge_track_ext(ctx, true): every bridged graph then records, per token prefix t, the sum over its
// bridged layers of the distinct experts of tokens 0..t that are not in the VRAM hot set; false when ctx owns no bridge.
// llama_moe_bridge_last_ext: the last bridged graph that completed without error - its width (*n_tokens), cold[t] for
// t < min(width, n) and its serial number (one per bridged graph); false when there is none. The context's thread only.
LLAMA_API bool llama_moe_bridge_track_ext(struct llama_context * ctx, bool on);
LLAMA_API bool llama_moe_bridge_last_ext(const struct llama_context * ctx, int32_t * n_tokens, uint32_t * cold, int32_t n, uint64_t * serial);

// [TAG_FN_L4_EXIT] The caller has no more work for ctx for now (a server whose slots all went idle): the context's graphs
// (also one whose outputs nobody reads, e.g. a draft context's update), its host step and the MoE hot set's upload batch in
// flight end before this returns, and the hot set's upload queue waits for the context's next graph. Outputs never change.
LLAMA_API void llama_quiesce_ext(struct llama_context * ctx);

// [TAG_FN_L18_FIRSTANS_FIT] The caller has loaded the model and every context of it (a server about to listen) and no
// request has come yet: qwen4exp with LLAMA_FN_L18_FIRSTANS_FIT sizes its deferred VRAM hot set now instead of in the first
// prompt, and with LLAMA_FN_L18_FIRSTANS_WARMLATE the experts' warm pass starts now. A no-op otherwise. Outputs never change.
LLAMA_API void llama_fn_ready_ext(struct llama_context * ctx);

// [TAG_FN_L20_BIGPROMPT_UB] Before a decode: how many prompt tokens come after this batch (the most of any sequence in it;
// 0 = this batch ends its prompts, -1 = not known). Read once by the next decode of ctx; it only picks qwen4exp's compute
// reserve width with LLAMA_FN_L20_BIGPROMPT_UB. Outputs never change.
LLAMA_API void llama_fn_prompt_after_ext(struct llama_context * ctx, int32_t n_after);

// [TAG_FN_L24_ENDOVL1] Before a decode: how many sequences the caller has busy (in a prompt or an answer; -1 = not known). Kept
// until set again; with one busy sequence a qwen4exp context of n_seq_max > 1 keeps FULL at the prompt's end for the answer's
// first step (LLAMA_FN_L22_ENDOVERLAP), as with n_seq_max = 1. Outputs never change.
LLAMA_API void llama_fn_seqs_busy_ext(struct llama_context * ctx, int32_t n_busy);

// [TAG_FN_L20_BIGPROMPT_PLEPRE] Before a decode: the NEXT prompt batch of seq_id, toks[i_first..n) at positions pos0 + i_first..
// (toks[0..i_first) the tokens before it, up to 16, for the n-gram context; pos0 = the position of toks[0]). qwen4exp reads
// their PLE rows ahead (llama_ple_prefetch_ext) once this decode's first ubatch has set its inputs. Outputs never change.
LLAMA_API void llama_fn_ple_next_ext(struct llama_context * ctx, llama_seq_id seq_id, llama_pos pos0, const llama_token * toks,
        int32_t n, int32_t i_first);

// [TAG_FN_L3_MTP_HEADPROMPT] qwen4exp with a MTP draft vocabulary and LLAMA_MTP_HEAD_PROMPT=<cap>: the drafts also score
// these prompt tokens where the vocabulary leaves them out (the first cap such ids in the given order replace the last
// set). Returns how many were taken, -1 when the model has no such list. Only drafts change; verify uses the full head.
LLAMA_API int32_t llama_model_mtp_head_set_prompt(const struct llama_model * model, const llama_token * ids, int32_t n);

// [TAG_FN_L18_KEEPCTX] The bytes of llama_state_seq_get_data_ext / llama_state_seq_set_data_ext (the same format), passed
// through a callback in pieces of at most 32 MiB instead of one host buffer of the whole state. A callback returns false to
// stop. Returns the bytes written / read, 0 on error; after a failed read the sequence is cleared, as after a failed
// llama_state_seq_set_data_ext. Tensor bytes are read from the device piece by piece, so no host copy of the whole state
// is ever made.
typedef bool (*llama_state_stream_write_fn)(const void * src, size_t size, void * user_data);
typedef bool (*llama_state_stream_read_fn)(void * dst, size_t size, void * user_data);
LLAMA_API size_t llama_state_seq_write_stream_ext(struct llama_context * ctx, llama_seq_id seq_id, llama_state_seq_flags flags,
                                                  llama_state_stream_write_fn fn, void * user_data);
LLAMA_API size_t llama_state_seq_read_stream_ext(struct llama_context * ctx, llama_seq_id seq_id, llama_state_seq_flags flags,
                                                 llama_state_stream_read_fn fn, void * user_data);

// [TAG_FN_L18_KEEPCTX_HOT] The decayed counts of the MoE hot set ctx owns (qwen4exp adaptive set): get copies them, set
// puts them back (the same process and model). false when ctx owns no decayed adaptive set or the layout differs.
LLAMA_API bool llama_moe_hot_counts_get_ext(struct llama_context * ctx, std::vector<float> & out);
LLAMA_API bool llama_moe_hot_counts_set_ext(struct llama_context * ctx, const std::vector<float> & in);

// [TAG_TURBOT] Process-wide turbot plan path; takes precedence over env LLAMA_TURBOT_PLAN. nullptr or "" clears it.
// [TAG_TURBOT_EMBED_PLAN] "default" selects the built-in plan; with neither set, the built-in plan is used when it fits.
LLAMA_API void llama_turbot_set_plan_path(const char * path);

// [TAG_TURBOT_ANY_SIDECAR] Process-wide path of the model's sidecar plan (<model>.turbot.plan), set by common when the file
// exists and neither --kv-tier-plan nor LLAMA_TURBOT_PLAN is given. It is used only when it carries a '# verified:' stamp
// and a '# model:' fingerprint equal to the model's (tools/turbot/turbot_guard.py writes both), and loses to an explicit
// plan. LLAMA_TURBOT_SIDECAR=0 (or LLAMA_TURBOT_ANY=0) ignores it. nullptr or "" clears it.
LLAMA_API void llama_turbot_set_sidecar_path(const char * path);

// [TAG_FN_AUTO] getenv(name), else the automatic value of this model (LLAMA_FLASHNEXT_PROFILE on qwen4exp with host
// experts, src/llama-fn-auto.h), else nullptr. model may be nullptr (then plain getenv).
LLAMA_API const char * llama_model_fn_env(const struct llama_model * model, const char * name);
// [TAG_FN_AUTO] "safe", "fast", "trial" or "trial-dma" while the automatic defaults are active for this model, else nullptr
LLAMA_API const char * llama_model_fn_profile(const struct llama_model * model);

// [TAG_FN_RAM_FIT] host memory plan for a model bigger than RAM (mmap): when the host-resident model bytes, the host
// buffers, the prompt cache and the context checkpoints need more than frac of physical RAM, the checkpoint budget is
// lowered (never below ckpt_min_mib per slot) and then the prompt cache (0 below 256 MiB), so the mapped expert pages
// stay in the page cache. Values the user set are kept. Nothing is raised.
struct llama_ram_fit_in {
    uint64_t ram_total     = 0;     // physical RAM, bytes
    uint64_t model_host    = 0;     // model bytes on host buffers (mapped or not)
    uint64_t host_buffers  = 0;     // host KV / recurrent / compute buffers of the contexts, bytes
    int64_t  cache_ram_mib = 0;     // prompt cache (--cache-ram), -1 = no limit, 0 = off
    bool     cache_ram_set = false; // set by the user
    int64_t  ckpt_mib      = 0;     // context checkpoint byte budget per slot, 0 = none or no byte bound
    bool     ckpt_set      = false; // set by the user (--ctx-checkpoints or LLAMA_CTX_CHECKPOINT_BUDGET_MIB)
    int32_t  n_slots       = 1;
    double   frac          = 0.85;
    int64_t  ckpt_min_mib  = 512;
};

struct llama_ram_fit_out {
    bool     over          = false; // the request passed frac x RAM
    bool     changed       = false; // a value below differs from the request
    uint64_t limit         = 0;     // frac x RAM, bytes
    uint64_t need          = 0;     // bytes of the request
    int64_t  cache_ram_mib = 0;     // prompt cache to use
    int64_t  ckpt_mib      = 0;     // checkpoint budget per slot to use
};

LLAMA_API llama_ram_fit_out llama_ram_fit_plan(const llama_ram_fit_in & in);

// retrieves the whole token embedding matrix in F32 format (n_embd * n_vocab)
// returns total number of elements or 0 on error
// if out is nullptr, returns the number of tokens without writing to out
// caller must allocate enough memory for out before calling
LLAMA_API uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out);

// [TAG_FN_L4_HOST] LLAMA_FN_L4_HOSTPROF=1|N (qwen4exp, src/llama-fn-hostprof.cpp): exclusive host time per named segment
// of the decode loop, a report every N (1: 256) steps. Off: every call returns at its first branch (-1 / nothing).
enum llama_hp_seg {
    LLAMA_HP_OTHER = 0,
    LLAMA_HP_SRV_PRE,     // server: pre_decode before the drafts
    LLAMA_HP_SRV_CKPT,    // server: checkpoints and draft-context seq_rm after the drafts
    LLAMA_HP_SRV_BATCH,   // server: the verify batch until llama_process(target)
    LLAMA_HP_SRV_DPOST,   // server: decode() after the target decode
    LLAMA_HP_SRV_SAMPLE,  // server: verify sampling and accept
    LLAMA_HP_SRV_BOOK,    // server: accept bookkeeping
    LLAMA_HP_SRV_TOKEN,   // server: process_token, responses
    LLAMA_HP_MTP_PROC,    // MTP process(): host part
    LLAMA_HP_MTP_DRAFT,   // MTP draft(): host part (incl. draft sampling)
    LLAMA_HP_MTP_ACCEPT,  // MTP accept()
    LLAMA_HP_DEC_BASE,    // + class * LLAMA_HP_DEC_N + sub: inside llama_decode (class 0 target, 1 MTP process, 2 MTP draft)
};
enum llama_hp_dec {
    LLAMA_HP_DEC_PREP = 0, LLAMA_HP_DEC_APPLY, LLAMA_HP_DEC_GRAPH, LLAMA_HP_DEC_INPUTS, LLAMA_HP_DEC_LAUNCH,
    LLAMA_HP_DEC_WAIT, LLAMA_HP_DEC_BEND, LLAMA_HP_DEC_OUT, LLAMA_HP_DEC_POST, LLAMA_HP_DEC_SYNC, LLAMA_HP_DEC_RESET,
    LLAMA_HP_DEC_ALLOC,
    // inside the launch (scheduler hook): split input copies, waits, a host split's compute, a device split's compute
    LLAMA_HP_DEC_SCPY, LLAMA_HP_DEC_SSYNC, LLAMA_HP_DEC_SCPU, LLAMA_HP_DEC_SGPU,
    LLAMA_HP_DEC_N,
};
#define LLAMA_HP_N_SEG ((int) LLAMA_HP_DEC_BASE + 3*(int) LLAMA_HP_DEC_N)

LLAMA_API void llama_hp_enable(int every);
LLAMA_API bool llama_hp_on(void);
LLAMA_API int  llama_hp_switch(int seg);             // returns the previous segment, -1 when off or another thread
LLAMA_API int  llama_hp_current(void);
LLAMA_API void llama_hp_ctx_class_set(const void * ctx, int cls);
LLAMA_API int  llama_hp_ctx_class_get(const void * ctx);
LLAMA_API int  llama_hp_class_of_caller(void);       // the decode class the current segment implies
LLAMA_API void llama_hp_step(int n_tokens);          // a target graph of n_tokens was launched

// [TAG_FN_L4_HOST] a lever of the l4 host family: true only for a qwen4exp model and when the switch name (else the
// umbrella LLAMA_FN_L4_HOST) is set to a non-zero value (environment or the model's automatic profile)
LLAMA_API bool llama_fn_l4_host_flag(const struct llama_model * model, const char * name);

// [TAG_FN_L4_HOST] the value of one switch only (no umbrella): def unless the model is qwen4exp and the switch is set
LLAMA_API int llama_fn_l4_int(const struct llama_model * model, const char * name, int def);

// [TAG_R28_PENFAST] the window of a penalties sampler: the distinct tokens its next apply moves, i.e. every token with a
// non-zero count among the last penalty_last_n accepted ones. Writes at most cap of them to out (out may be null when cap
// is 0) and returns how many there are; -1 when smpl is not an active penalties sampler.
LLAMA_API int32_t llama_sampler_penalties_window(const struct llama_sampler * smpl, llama_token * out, int32_t cap);

// [TAG_R28_PENFAST] llama_sampler_apply(chain, cur_p) restricted to the samplers [i0, i1) of the chain, with the chain's own
// rule for samplers the backend already ran (skipped while every sampler before them was one too): applying [0, i) and
// then [i, n) is one apply of the chain.
LLAMA_API void llama_sampler_chain_apply_range(struct llama_sampler * chain, llama_token_data_array * cur_p, int32_t i0, int32_t i1);

// [TAG_FN_L4_HOST_PLEPRE] LLAMA_FN_L4_HOST_PLEPRE (qwen4exp with PLE direct I/O): a helper thread reads the PLE rows of
// toks[i_first..n), which a coming decode of seq_id holds at positions pos0 + i, into the row cache (predecessors: the
// earlier toks, else the KV cells). A hint only: the decode computes and reads its own rows as before. No-op otherwise.
LLAMA_API void llama_ple_prefetch_ext(struct llama_context * ctx, llama_seq_id seq_id, llama_pos pos0, const llama_token * toks,
        int32_t n, int32_t i_first);

// [TAG_FN_L17_VISION_LEND] LLAMA_FN_L17_VISION=1 (qwen4exp, a hot set fitted to VRAM on dev with virtual memory): the
// projector may park its device buffers between encodes. _register: the device bytes it freed (before the fit, which
// lends them to the hot set). Around each encode (the owner's thread): _take(bytes) gives VRAM back from the hot set
// (returns the bytes, 0: none lent), _give returns it.
LLAMA_API bool   llama_fn_vision_lend_wanted(const struct llama_model * model, ggml_backend_dev_t dev);
LLAMA_API void   llama_fn_vision_lend_register(const struct llama_model * model, size_t bytes);
LLAMA_API size_t llama_fn_vision_take(const struct llama_model * model, size_t bytes);
LLAMA_API void   llama_fn_vision_give(const struct llama_model * model);

//
// [TAG_FN_L22] round 22 interfaces (E:/turbot-gates/flashnext/test/r22/PLAN.md section 4.2), written by P0 and frozen.
// Owner and stub behaviour per entry; the internal ones are in src/llama-fn-l22.h.
//

// [TAG_FN_L22_IMGPASS] P1: the token id an embedding entry stands for in the PLE / KV ids (a text row of a merged image
// turn: its token; an image row: llama_model_media_token). LLAMA_TOKEN_NULL = none. P0 stores it per entry (get returns
// it, LLAMA_TOKEN_NULL for none or a bad idx); P1 carries it into the ubatch.
LLAMA_API void        llama_batch_ext_set_token_aux(struct llama_batch_ext * batch, int32_t idx, llama_token tok);
LLAMA_API llama_token llama_batch_ext_get_token_aux(const struct llama_batch_ext * batch, int32_t idx);
// P1: the trunk's input embedding rows of n tokens (n x n_embd_inp floats, bit for bit the host gather's); stub false
LLAMA_API bool        llama_model_token_embd_rows(const struct llama_model * model, const llama_token * toks, int32_t n, float * out);
// P1: the token an image row stands for (qwen4exp: hparams.ple_image_token_id); stub LLAMA_TOKEN_NULL
LLAMA_API llama_token llama_model_media_token(const struct llama_model * model);
// [TAG_FN_L22_IMGPASS_MASK] P1: 0 = keyed positional masks for image ubatches, 1 = explicit masks; stub 1
LLAMA_API int32_t     llama_fn_imgpass_mask_mode(const struct llama_model * model);

// [TAG_FN_L22_GENSHARE2] P4: the width of the context's next prompt ubatch (read after llama_fn_prompt_after_ext); stub
// n_ubatch
LLAMA_API int32_t llama_fn_prompt_ubatch_width_ext(struct llama_context * ctx);

// [TAG_V0261_RING_ACCESSOR] true when the context raised n_rs_seq by one for the MoE bridge's ring ([TAG_FN_R1_BRIDGE_RB]):
// a speculative caller can roll back n_rs_seq - 1 drafts
LLAMA_API bool llama_fn_bridge_rb_ext(const struct llama_context * ctx);

// [TAG_FN_L22_PHASE] P0: the decode-phase flag (plan 3.8). The caller sets GEN before a decode whose batch holds only
// generation / verify rows; every other decode runs as PROMPT. The flag goes back to PROMPT when the decode returns.
// Only the answer-changing levers (LLAMA_FN_L22_X_*) read it, and only in ubatches whose rows are all output rows.
enum {
    LLAMA_FN_PHASE_PROMPT = 0,
    LLAMA_FN_PHASE_GEN    = 1,
};
LLAMA_API void    llama_set_decode_phase_ext(struct llama_context * ctx, int32_t phase);
LLAMA_API int32_t llama_get_decode_phase_ext(const struct llama_context * ctx);

// [TAG_FN_L22_OOMHOOK] P0: llama_fn_pin_on_oom (P6) for code outside libllama (the vision lend, the server): a device
// allocation of need_bytes failed or will fail; true = pins were released, retry once
LLAMA_API bool llama_fn_pin_on_oom_ext(const struct llama_model * model, size_t need_bytes);
// [TAG_FN_L23_PINFIX] R-L23-8: the server marks a request (+1 / -1; 0 = time stamp only): pins register only while idle
LLAMA_API void llama_fn_pin_server_busy(int delta);
// [TAG_FN_V0254_STORE] see llama-fn-l22.h
LLAMA_API uint64_t llama_fn_res_touched_ext(void);
// [TAG_FN_L22_PIN] (P2b request 3) llama_fn_pin_active (P6) for code outside libllama
LLAMA_API bool llama_fn_pin_active_ext(void);

// [TAG_FN_L22] test helper: a llama_batch_ext without a context (no memory, n_vocab ids accepted)
LLAMA_API struct llama_batch_ext * llama_batch_ext_init_test(int32_t n_tokens_max, int32_t n_embd, int32_t n_seq_max,
        int32_t n_vocab, int32_t n_pos_per_embd);

// RAII: switch to seg, back to the previous segment at scope exit
struct llama_hp_scope {
    int prev;
    explicit llama_hp_scope(int seg) : prev(llama_hp_switch(seg)) {}
    ~llama_hp_scope() { if (prev >= 0) { llama_hp_switch(prev); } }
    llama_hp_scope(const llama_hp_scope &) = delete;
    llama_hp_scope & operator=(const llama_hp_scope &) = delete;
};
// same as above, but only the rows of the given tokens, in order (n_tokens * n_embd elements)
// returns the number of elements written or 0 on error
LLAMA_API uint32_t llama_model_get_tok_embd_rows(const struct llama_model * model, const llama_token * tokens, uint32_t n_tokens, float * out);

// realtime models: number of tokens the output lags behind the audio input, used by mtmd; 0 if not applicable
LLAMA_API uint32_t llama_model_n_tokens_delay(const struct llama_model * model);
