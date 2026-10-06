#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"

#include <cstdint>
#include <map>

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

// Set whether the context outputs nextn embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);

// Select which appended NextN block the DECODER_MTP graph runs (offset past
// the trunk: il = n_layer() + offset). Used by the speculative NextN driver to
// chain multiple trained NextN heads. Default 0 (first head).
LLAMA_API void llama_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset);

// Marks the entries that a joint decision head (clef) reads, the default is 0
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

// [TAG_FN_L3_MTP_HEADPROMPT] qwen4exp with a MTP draft vocabulary and LLAMA_MTP_HEAD_PROMPT=<cap>: the drafts also score
// these prompt tokens where the vocabulary leaves them out (the first cap such ids in the given order replace the last
// set). Returns how many were taken, -1 when the model has no such list. Only drafts change; verify uses the full head.
LLAMA_API int32_t llama_model_mtp_head_set_prompt(const struct llama_model * model, const llama_token * ids, int32_t n);

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

// [TAG_FN_L4_HOST_PLEPRE] LLAMA_FN_L4_HOST_PLEPRE (qwen4exp with PLE direct I/O): a helper thread reads the PLE rows of
// toks[i_first..n), which a coming decode of seq_id holds at positions pos0 + i, into the row cache (predecessors: the
// earlier toks, else the KV cells). A hint only: the decode computes and reads its own rows as before. No-op otherwise.
LLAMA_API void llama_ple_prefetch_ext(struct llama_context * ctx, llama_seq_id seq_id, llama_pos pos0, const llama_token * toks,
        int32_t n, int32_t i_first);

// RAII: switch to seg, back to the previous segment at scope exit
struct llama_hp_scope {
    int prev;
    explicit llama_hp_scope(int seg) : prev(llama_hp_switch(seg)) {}
    ~llama_hp_scope() { if (prev >= 0) { llama_hp_switch(prev); } }
    llama_hp_scope(const llama_hp_scope &) = delete;
    llama_hp_scope & operator=(const llama_hp_scope &) = delete;
};
