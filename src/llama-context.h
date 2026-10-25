#pragma once

#include "llama.h"
#include "llama-ext.h"
#include "llama-fn-l22.h" // [TAG_FN_L22_PHASE]
#include "llama-fn-l22-p10.h" // [TAG_FN_L22_P10] llama_fn_l22_cost
#include "llama-fn-l22-p16.h" // [TAG_FN_L22_CPUSHARE] llama_pfs_share_counters
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-adapter.h"
#include "llama-impl.h"
#include "llama-memory.h"
#include "llama-moe-cache.h"

#include "ggml-cpp.h"
#include "ggml-opt.h"

#include <array>
#include <functional>
#include <map>
#include <vector>

struct llama_model;
class llama_batch_allocr;
struct llama_ubatch;
class llama_moe_cache;

class llama_io_read_i;
class llama_io_write_i;

// "memory" as in abstract memory for the context
struct llama_memory_i;
struct llama_memory_context_i;

// stores copy of the memory in device buffer. used for fast state save/load
struct llama_memory_buffer {
    int n_tensors = 0;
    size_t total_size = 0;

    ggml_backend_buffer_ptr buf;

    ggml_context_ptr ctx;

    std::vector<ggml_tensor *> org;
    std::vector<ggml_tensor *> cpy;
};

using llama_memory_buffers = std::map<ggml_backend_buffer_type_t, llama_memory_buffer>;

struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    llama_context(
            const llama_model & model,
                  llama_context_params params);

    ~llama_context();

    // reserve a new backend scheduler (if needed)
    // for example, when:
    //   - changing loras
    //   - changing samplers
    //   - changing attention type
    //   - etc.
    void sched_reserve();

    void synchronize();

    // [TAG_FN_L4_EXIT] the caller has no work for now: the graphs, the host step after the last one and the hot set's
    // upload batch in flight end before this returns, and the hot set's queue waits for the next graph
    void quiesce();

    // [TAG_FN_L18_FIRSTANS_FIT] the caller's model and contexts are loaded and no request has come yet (a server that is
    // about to listen): with LLAMA_FN_L18_FIRSTANS_FIT the deferred hot set is sized now instead of in the first prompt
    void fn_ready();
    // [TAG_FN_L20_BIGPROMPT_UB] the caller's prompt tokens after the next decode's batch (-1: not known), read once by it
    void fn_prompt_after(int32_t n) { l20_prompt_after = n; }
    // [TAG_FN_L24_ENDOVL1] the caller's busy sequences (in a prompt or an answer; -1: not known), kept until set again
    void fn_seqs_busy(int32_t n) { l24_seqs_busy = n; }
    // [TAG_FN_L22_PHASE] the decode-phase flag (llama_set_decode_phase_ext): read by the next decode's graphs, PROMPT again
    // when that decode returns
    void    fn_set_phase(int32_t phase) { fn_phase = phase == LLAMA_FN_PHASE_GEN ? LLAMA_FN_PHASE_GEN : LLAMA_FN_PHASE_PROMPT; }
    int32_t fn_get_phase() const { return fn_phase; }
    // [TAG_FN_L22_GENSHARE2] the width of the next prompt ubatch given the current hint (llama_fn_prompt_ubatch_width_ext)
    uint32_t fn_prompt_ubatch_width() const;
    // [TAG_FN_L20_BIGPROMPT_PLEPRE] the next prompt batch's tokens (with up to 16 before them for the n-gram context): their
    // PLE rows are read ahead once this decode's first ubatch has its inputs
    void fn_ple_next(llama_seq_id seq, llama_pos pos0, const llama_token * toks, int32_t n, int32_t i_first);

    void set_layer_inp_extract(bool enable);

    const llama_model   & get_model()   const;
    const llama_cparams & get_cparams() const;

    ggml_backend_sched_t get_sched() const;

    uint32_t n_ctx()     const;
    uint32_t n_ctx_seq() const;
    uint32_t n_batch()   const;
    uint32_t n_ubatch()  const;
    uint32_t n_seq_max() const;

    uint32_t n_threads()       const;
    uint32_t n_threads_batch() const;

    llama_memory_t get_memory() const;

    // return true if the memory was updated
    bool memory_update(bool optimize);

    enum llama_pooling_type pooling_type() const;

    float * get_logits();
    float * get_logits_ith(int32_t i);

    float * get_embeddings();
    float * get_embeddings_ith(int32_t i);
    float * get_embeddings_seq(llama_seq_id seq_id);

    float * get_embeddings_nextn();
    float * get_embeddings_nextn_ith(int32_t i);

    float * get_embeddings_layer_inp(uint32_t lid);

    llama_token * get_sampled_tokens() const;
    llama_token   get_sampled_token_ith(int32_t idx);

    float * get_sampled_logits_ith(int32_t idx);
    size_t  get_sampled_logits_count(int32_t idx);

    float * get_sampled_probs_ith(int32_t idx);
    size_t  get_sampled_probs_count(int32_t idx);

    const llama_token * get_sampled_candidates_ith(int32_t idx);
    size_t get_sampled_candidates_count(int32_t idx);

    bool get_causal_attn() const;

    void attach_threadpool(
            ggml_threadpool_t threadpool,
            ggml_threadpool_t threadpool_batch);

    void detach_threadpool();

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    void set_embeddings (bool value);
    void set_embeddings_nextn(bool value, bool masked);
    void set_embeddings_layer_inp(uint32_t lid, bool enable);
    void set_nextn_layer_offset(int32_t offset);
    void set_causal_attn(bool value);
    void set_warmup(bool value);

    void set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end);

    // process a single ubatch with a specific graph type
    // if memory_context is provided, it will be applied first to the context's memory
    // ret contains the status of the graph computation
    // returns nullptr only if ret != GGML_STATUS_SUCCESS
    llm_graph_result * process_ubatch(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
                       ggml_status & ret);

    int encode(const llama_batch_ext & batch_inp);
    int decode(const llama_batch_ext & batch_inp);

    // compat version
    int encode(const llama_batch & batch_inp);
    int decode(const llama_batch & batch_inp);

    //
    // state save/load
    //

    size_t state_get_size();
    size_t state_get_data(      uint8_t * dst, size_t size);
    size_t state_set_data(const uint8_t * src, size_t size);

    size_t state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags);

    size_t state_seq_get_data(llama_seq_id seq_id,       uint8_t * dst, size_t size, llama_state_seq_flags flags);
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags);

    bool state_load_file(
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    bool state_save_file(
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    size_t state_seq_load_file(
          llama_seq_id   seq_id,
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    size_t state_seq_save_file(
          llama_seq_id   seq_id,
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    // [TAG_FN_L18_KEEPCTX] the bytes of state_seq_get_data / state_seq_set_data, passed through a callback in pieces
    size_t state_seq_write_stream(llama_seq_id seq_id, llama_state_seq_flags flags, const std::function<bool(const void *, size_t)> & fn);
    size_t state_seq_read_stream (llama_seq_id seq_id, llama_state_seq_flags flags, const std::function<bool(void *, size_t)> & fn);

    //
    // perf
    //

    llama_perf_context_data perf_get_data() const;
    void perf_reset();

    llama_memory_breakdown memory_breakdown() const;

    //
    // training
    //

    void opt_init(struct llama_model * model, struct llama_opt_params lopt_params);

    // TODO: more flexible combinations of logical/physical batch size and context size
    void opt_epoch(
            ggml_opt_dataset_t      dataset,
            ggml_opt_result_t       result_train,
            ggml_opt_result_t       result_eval,
            int64_t                 idata_split,
            ggml_opt_epoch_callback callback_train,
            ggml_opt_epoch_callback callback_eval);

    void opt_epoch_iter(
            ggml_opt_dataset_t               dataset,
            ggml_opt_result_t                result,
            const std::vector<llama_token> & tokens,
            const std::vector<llama_token> & labels_sparse,
            llama_batch                    & batch,
            ggml_opt_epoch_callback          callback,
            bool                             train,
            int64_t                          idata_in_loop,
            int64_t                          ndata_in_loop,
            int64_t                          t_loop_start);

private:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    uint32_t output_reserve(int32_t n_outputs);

    void output_reorder();

    // map the output row index `i` to batch index
    int64_t output_resolve_row(int32_t i) const;

    // async-copy enabled layer-input tensors (per cparams.output_layer_inp)
    // from backend into host-side embd_layer_inp buffers, each row at its batch index (scatter) or as one
    // block whose rows output_reorder() permutes into batch order; returns true if anything was copied
    bool extract_layer_inputs(const llm_graph_result * res, size_t token_offset, const llama_ubatch & ubatch, bool scatter);

    // [TAG_LAYER_INP_SCATTER] async-copy a token-indexed [row_floats, n_tokens] tensor so that ubatch
    // row r lands at host row ubatch.data->idx_batch[r] (runs of consecutive indices are one copy);
    // without scatter the tensor is copied as one block at token_offset
    void tensor_get_rows_batch_order(ggml_backend_t backend, ggml_tensor * t, float * dst, size_t dst_floats,
            size_t row_floats, size_t token_offset, const llama_ubatch & ubatch, bool scatter);

    // [TAG_LAYER_INP_SCATTER] [TAG_EXTRACT_TARGET_EMBEDDINGS] write token-indexed rows at their batch index
    // (false: block copies + output_reorder() permutation, for tensor split or LLAMA_LAYER_INP_SCATTER=0)
    bool layer_inp_scatter() const;

    //
    // graph
    //

public:
    uint32_t graph_max_nodes(uint32_t n_tokens) const;

    // can reuse the llm_graph_result instance of the context (for example to update a memory module)
    llm_graph_result * get_gf_res_reserve() const;

    // returns the result of ggml_backend_sched_graph_compute_async execution
    ggml_status graph_compute(ggml_cgraph * gf, bool batched);

    // reserve a graph with a dummy ubatch of the specified size
    ggml_cgraph * graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only = false, size_t * sizes = nullptr);

    bool set_sampler(llama_seq_id seq_id, llama_sampler * sampler);

    // [TAG_FN_L3_MTP_COST2] the MoE bridge this context owns (nullptr: none)
    struct llama_moe_bridge * get_moe_bridge() const { return moe_bridge; }

private:
    llm_graph_result * get_gf_res_prev();

    llm_graph_params graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const;

    llm_graph_cb graph_get_cb() const;

    // create a new backend scheduler for a graph of the specified size and register the callbacks below
    void sched_new(size_t max_nodes, bool parallel);

    // ggml_backend_sched copy callback, copies only the experts used by MUL_MAT_ID and updates the MoE cache
    static bool sched_copy_experts(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph, void * user_data);

    // ggml_backend_sched realloc callback, counts and traces the graph reallocations of this context
    static void sched_graph_realloc(ggml_backend_sched_t sched, ggml_cgraph * gf, bool backend_ids_changed, bool graph_size_changed, void * user_data);

    // disable auto fused ops (Flash Attention, Gated Delta Net) whose op lands on a device
    // that differs from the layer it belongs to (usually due to missing backend support)
    void resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs);

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(llama_io_write_i & io);
    size_t state_read_data (llama_io_read_i  & io);

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_read_data (llama_io_read_i  & io, llama_seq_id seq_id, llama_state_seq_flags flags);

    //
    // members
    //

    const llama_model & model;

    llama_cparams cparams;

    llama_adapter_cvec_ptr  cvec;
    llama_adapter_loras_ptr loras;

    llama_cross cross; // TODO: tmp for handling cross-attention - need something better probably

    llama_memory_ptr memory;
    llama_moe_cache_ptr moe_cache;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    buffer_view<float> logits = {nullptr, 0};

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    buffer_view<float> embd = {nullptr, 0};

    // hidden state required by the nextn layers (2-dimensional array: [n_outputs][n_embd])
    // populated only when cparams.embeddings_nextn is enabled and the model graph
    // sets llm_graph_result::t_h_nextn
    buffer_view<float> embd_nextn = {nullptr, 0};

    // host buffers for output layer input embeddings, per layer
    // populated when cparams.output_layer_inp[il] is true
    std::vector<buffer_view<float>> embd_layer_inp;
    std::vector<int32_t> embd_batch_idxs; // extracted index -> original batch index

    struct sampling_info {
        // !samplers.empty() to check if any samplers are active
        std::map<llama_seq_id, llama_sampler *> samplers;

        buffer_view<float>       logits     = {nullptr, 0};
        buffer_view<llama_token> sampled    = {nullptr, 0};
        buffer_view<float>       probs      = {nullptr, 0};
        buffer_view<llama_token> candidates = {nullptr, 0};

        std::vector<uint32_t> logits_count;
        std::vector<uint32_t> probs_count;
        std::vector<uint32_t> candidates_count;

        // optimization
        std::vector<llama_token> token_ids_full_vocab;
    };

    sampling_info sampling;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // reuse the batch_allocr to avoid unnecessary memory allocations
    std::unique_ptr<llama_batch_allocr> balloc;

    // [TAG_SYNC_BATCH_EXT_COMPAT] reused llama_batch -> llama_batch_ext conversion for encode/decode(const llama_batch &).
    // A fresh llama_batch_ext per call regrew its embd vector row by row and, above ~512 KiB on Windows, faulted in
    // new pages every time; the DFlash2 inject rows (100 KiB each) made that a per-step and per-prefill-chunk cost.
    // Created on first use; cleared (capacity kept) on every call.
    std::unique_ptr<llama_batch_ext> batch_compat;

    llama_batch_ext & batch_compat_get();

    uint32_t n_input_tensors = 0; // number of tensors marked as input during the last graph reserve
    uint32_t n_outputs = 0; // number of actually-used outputs in the current ubatch or last logical batch

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers

    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };

    std::vector<swap_info> output_swaps;

    ggml_backend_sched_ptr sched;

    bool sched_need_reserve = true;

    // whether the reserve graph has a logits tensor, set in sched_reserve()
    // true until then, because the first output_reserve() runs before any graph is reserved
    bool graph_has_logits = true;

    // state of sched_copy_experts, reset before each graph compute
    struct copy_experts_info {
        const ggml_tensor *  ids = nullptr;
        std::vector<int32_t> ids_data;
        std::vector<bool>    used;

        void reset() {
            ids = nullptr;
            ids_data.clear();
            used.clear();
        }
    };

    copy_experts_info copy_experts;

    // graph reallocations during inference where the backend assignments did not change, split by whether the
    // graph size is the same as the previously scheduled one (unexpected) or different
    int32_t n_reallocs_same = 0;
    int32_t n_reallocs_diff = 0;

    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    // training
    ggml_opt_context_t opt_ctx = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // pointers and buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    std::vector<size_t>                     backend_buf_exp_size; // expected buffer sizes

    // Separate arenas give batches with and without outputs distinct CUDA graph cache keys.
    std::array<llm_graph_result_ptr, 2> gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;

    llm_graph_result * gf_res_prev_active = nullptr;

    // [TAG_FN_GRAPH_PER_WIDTH] LLAMA_GRAPH_PER_WIDTH=1: one graph per decode width 1..4 (with outputs), so MTP verify
    // widths that alternate 1/2/3 switch graphs with a scheduler re-split instead of a full rebuild
    bool graph_per_width = false;
    std::array<llm_graph_result_ptr, 4> gf_res_width;
    uint64_t hot_gen_seen = 0; // [TAG_FN_L8_KVLEND] llama_moe_hot_generation() of the graphs built
    uint64_t n_width_switch = 0;

    // [TAG_FN_L4_HOST_SNAP] LLAMA_FN_L4_HOST_SNAP (qwen4exp, with LLAMA_GRAPH_PER_WIDTH): a width switch puts back the
    // scheduler state of that width's last split instead of splitting and allocating again; decode widths 1..4 without
    // outputs (the MTP verify catch-up) keep their own graph as well
    bool     snap_on = false;
    uint64_t n_snap_restore = 0;
    uint64_t n_snap_save    = 0;
    std::array<llm_graph_result_ptr, 4> gf_res_width_noout;
    void snap_take(llm_graph_result * res, bool launch2);

    // [TAG_FN_L4_HOST_EARLYOUT] LLAMA_FN_L4_HOST_EARLYOUT (qwen4exp): decode() hands process_ubatch the output copies of
    // the ubatch, which queues them behind a bridged graph before that graph's synchronize
    bool   early_out_on = false;

    int    batch_inputs_mode = 0;     // [TAG_FN_L4_HOST_BATCHCPY] LLAMA_FN_L4_HOST_BATCHCPY (qwen4exp): 0 off, 1 batch, 2 packed
    bool   upload_verify     = false; // [TAG_FN_L16_UPLOAD] LLAMA_FN_UPLOAD_VERIFY (qwen4exp, debug)
    bool   mtp_async_on      = false; // [TAG_FN_L16_MTP_ASYNC] LLAMA_FN_L16_MTP_ASYNC (qwen4exp MTP draft context)
    bool   mtp_tail_live     = false; // [TAG_FN_L16_MTP_ASYNC] the last decode returned before its graph ended; synchronize() clears it
    void   sched_hooks();             // [TAG_FN_L4_HOST_BATCHCPY] [TAG_FN_L4_HOST] set on every new scheduler
    void (*early_extract_fn)(void * ud, const llm_graph_result * res) = nullptr;
    void * early_extract_ud = nullptr;

    // [TAG_MOE_BRIDGE] LLAMA_MOE_BRIDGE=1: the host doorbell of the host-resident expert layers (llama-moe-bridge.h)
    struct llama_moe_bridge * moe_bridge = nullptr;
    bool     moe_bridge_failed = false; // [TAG_FN_R1_BRIDGE_RETRY] the last process_ubatch failed in the bridge
    uint64_t n_bridge_retry    = 0;     // [TAG_FN_R1_BRIDGE_RETRY] ubatches computed again after a bridge failure

    // [TAG_FN_VRAM_FIT] the hot set waits for the first decode after every context of the model exists (the MTP draft
    // context is created after this one), then takes what the model, KV, compute buffers and draft context left
    bool     moe_hot_pending = false;
    uint32_t moe_hot_waits   = 0;
    // [TAG_FN_L18_FIRSTANS_*] qwen4exp switches of the first answer after a start, read when the context starts
    bool     fa18_once = false; // LLAMA_FN_L18_FIRSTANS_ONCE: the prompt-end pass and the eager SMALL switch once per prompt
    bool     fa18_fit  = false; // LLAMA_FN_L18_FIRSTANS_FIT: the deferred hot set fit when the caller is ready, not in a prompt
    bool     fa18_ovl  = false; // LLAMA_FN_L18_FIRSTANS_OVERLAP: the prompt-end pass's copies overlap the host work after it
    bool     fa18_warm = false; // LLAMA_FN_L18_FIRSTANS_WARM: only answer steps make the experts' warm pass wait
    // [TAG_FN_L19_TTFT_HUSH] LLAMA_FN_L19_TTFT_HUSH: bit 1 = the warm pass and the saved state's read-ahead pause while a FULL
    // compute reserve is made, bit 2 = while a prompt batch (wider than an answer step) runs; 0 = off
    int      l19_hush  = 0;
    // [TAG_FN_L19_TTFT_EARLY] LLAMA_FN_L19_TTFT_EARLY=<n>: while the experts' warm pass runs (the first minute or so after a
    // start), a prompt batch of up to n tokens stays SMALL (the LLAMA_FN_L4_PROMPT path: no FULL reserve, whose allocation
    // sat 7.4 s in the driver beside the pass); 0 = off
    uint32_t l19_early = 0;
    uint64_t cbuf_n_early = 0;
    // [TAG_FN_L19_TTFT_TAILFULL] LLAMA_FN_L19_TTFT_TAILFULL=<n>: the narrow last part of a split prompt (it asks for outputs, the
    // batch before it was wider than SMALL, had none and fewer than n tokens) stays FULL; the answer's first step returns to
    // SMALL, as after an unsplit prompt. With LLAMA_FN_L19_TTFT_CKPT_TAIL the server splits the last 4 tokens of short turns
    // too, and their SMALL switch (~0.25 s) would otherwise move in front of the first token. 0 = off
    uint32_t l19_tailfull  = 0;
    // [TAG_FN_L20_CONVFIX_IO] LLAMA_FN_L20_CONVFIX_IO=1: the pinned piece of the streamed state reads / writes (the
    // conversation store) is allocated once per context instead of once per call (-1 = not read yet)
    int                     l20_io = -1;
    ggml_backend_buffer_ptr io_stream_pinned;
    uint32_t cbuf_prev_n   = 0;     // the previous batch of this context: tokens, and whether it asked for outputs
    bool     cbuf_prev_out = true;
    uint64_t cbuf_n_tailfull = 0;
    void     moe_hot_fit_try();

    // [TAG_FN_L3_HOST_STEP] LLAMA_FN_HOST_STEP=1 (qwen4exp): the owner's moe-cache / DMA step after a decode runs on a
    // helper thread; step_join() waits for it before anything that may read or change that state again
    struct step_worker;
    bool                         step_async_want = false;
    std::unique_ptr<step_worker> step_async;
    void                         step_join();

    // [TAG_FN_L3_VRAM_CBUF] LLAMA_FN_CBUF=1 (qwen4exp, the hot set's owner; the [TAG_FN_R3_CBUF_LEND] design of round 2):
    // between prompts the scheduler holds a SMALL reserve (graphs of <= cbuf_small_t tokens), and the hot set keeps the
    // difference to the FULL reserve as a tail of its device buffer; a wider batch first gives the tail's VRAM back and
    // reserves FULL, the next narrow one returns
    enum { CBUF_OFF = 0, CBUF_FULL = 1, CBUF_SMALL = 2 };
    int      cbuf_state     = CBUF_OFF;
    uint32_t cbuf_small_t   = 31;
    size_t   cbuf_full_b    = 0;   // device bytes of the trunk's compute buffers at the FULL reserve
    size_t   cbuf_small_b   = 0;
    size_t   cbuf_slack_b   = 0;   // extra tail for a FULL reserve that grows once the hot set exists
    bool     cbuf_grew      = false;
    size_t   cbuf_pool_mib  = 0;   // LLAMA_FN_CBUF_POOL_MIB: extra tail for the pool growth of a FULL prompt
    bool     cbuf_trim      = true;
    bool     cbuf_eager     = true;  // LLAMA_FN_CBUF_EAGER: return to SMALL at the end of a prompt's last (partial) batch
    uint64_t cbuf_n_full    = 0;
    uint64_t cbuf_n_small   = 0;
    double   cbuf_ms_full   = 0.0;
    double   cbuf_ms_small  = 0.0;
    size_t   cbuf_peak      = 0;   // the most device use (all processes) seen at a switch
    uint64_t cbuf_n_retry   = 0;   // narrow batches since the last return to SMALL left the tail out
    // LLAMA_FN_CBUF_THRASH: that many FULL switches within 64 batches (steps wider than the SMALL reserve, e.g. several
    // streams drafting 8 tokens each) pin the FULL state: the tail stays out, no switch per step; 256 narrow batches in a
    // row unpin it
    int      cbuf_thrash    = 8;
    bool     cbuf_pinned    = false;
    uint64_t cbuf_n_batch   = 0;
    uint64_t cbuf_narrow_run = 0;  // narrow batches in a row while pinned (256 unpin)
    std::vector<uint64_t> cbuf_full_at; // the batch numbers of the recent FULL switches
    // [TAG_FN_L4_MEM_PROMPT] LLAMA_FN_L4_PROMPT=<n>: a batch of cbuf_small_t + 1 .. n tokens that comes in the SMALL state
    // stays SMALL: ubatches of <= cbuf_small_t tokens with the host experts on the CPU (the path of a prompt below the
    // stream's minimum), no FULL reserve, no prefill stream and no tail refill afterwards
    uint32_t cbuf_short_max = 0;
    bool     cbuf_short     = false; // the running batch takes that path
    uint64_t cbuf_n_short   = 0;
    double   cbuf_ms_short  = 0.0;
    bool     cbuf_wanted() const;
    void     cbuf_note_full();
    size_t   cbuf_prepare(ggml_backend_dev_t dev);
    void     cbuf_off();
    bool     cbuf_set(bool full, uint32_t width = 0, uint32_t cls = 0); // [TAG_FN_L22_LENDCLASS] cls: the FULL class (0: none)
    // [TAG_FN_L20_BIGPROMPT_UB] LLAMA_FN_L20_BIGPROMPT_UB=<n> (qwen4exp profile lookup, 0 = off): with the compute-buffer lend,
    // a target context whose -b and -ub are equal (>= 4096) takes n for both. A prompt whose batch or rest (the server's
    // hint) is wider than the command's ubatch runs FULL at n tokens per ubatch (the hot set's whole tail lent); any other
    // wide batch runs FULL at the command's width (only the tail's top part lent, l20_base_need bytes). The width is chosen
    // at the switch to FULL and kept until SMALL. A narrow part of a prompt that goes on (hint > 0) keeps the FULL state.
    uint32_t l20_base_ub      = 0;  // the command's n_ubatch (0: the switch is off or did not apply)
    uint32_t cbuf_full_t      = 0;  // the width of the FULL reserve now (0: n_ubatch)
    size_t   l20_base_need    = 0;  // tail bytes a FULL reserve at l20_base_ub needs given back
    size_t   l20_full_base_b  = 0;
    int32_t  l20_prompt_after = -1;
    int32_t  l24_seqs_busy    = -1; // [TAG_FN_L24_ENDOVL1]
    // [TAG_FN_L22_PHASE] the decode-phase flag and the model's X levers (read at construction); graph_params folds both into
    // fn_gen / fn_x (plan 3.8)
    int32_t            fn_phase = LLAMA_FN_PHASE_PROMPT;
    llama_fn_l22_x_cfg fn_l22_x;
    uint8_t            fn_l22_hooks = 0; // [TAG_FN_L22] llama_fn_l22_hook bits, read at construction
    // [TAG_FN_L20_BIGPROMPT_PLEPRE] the pending read-ahead (empty toks: none)
    llama_seq_id             l20_ple_seq   = -1;
    llama_pos                l20_ple_pos0  = 0;
    int32_t                  l20_ple_first = 0;
    std::vector<llama_token> l20_ple_toks;
    uint64_t                 l20_ple_n     = 0;
    uint64_t l20_n_big = 0, l20_n_base = 0, l20_n_keep = 0;
    // [TAG_FN_L20_BIGPROMPT_DFTUB] the MTP draft context with LLAMA_FN_L20_BIGPROMPT_DFTUB: n_ubatch = the wide width, but its
    // compute reserve (and its ubatches) stay at dft_narrow (SPEC_DFT_UBATCH) except while the target runs a prompt at its
    // big ubatch: the target's tail then also holds the draft's extra (l20_dft_extra, measured at the target's fit)
    uint32_t dft_narrow    = 0;     // 0: no two widths
    bool     dft_wide      = false;
    uint64_t dft_n_wide    = 0;
    size_t   l20_dft_extra = 0;     // the target: device bytes the draft's wide reserve adds
    // [TAG_FN_L21_BIGUB] LLAMA_FN_L21_BIGUB=1 (qwen4exp profile lookup) with LLAMA_FN_L20_BIGPROMPT_UB: the tail is sized for
    // the command's width (the hot set's layout and its KV slot lend as without the 16K switch); the 16K reserve's extra
    // (l21_extra) comes from borrow segments in the KV lend layers' base slots, out only while a 16K FULL runs; the 16K
    // width only for prompts above n_ubatch tokens (llama_fn_l21_full_width). No borrow placed: the command's width always
    bool     l21_bigub     = false;
    bool     l21_big_off   = false;
    size_t   l21_extra     = 0;
    uint64_t l21_n_borrow  = 0, l21_n_fallback = 0;

    // [TAG_FN_L22_P4] round 22, package P4 (prompt stream state machine); every switch off unless set (plan 3.4)
    struct l22_p4_cfg {
        bool lendclass  = false; // LLAMA_FN_L22_LENDCLASS: FULL reserve classes 1K / 2K / 4K below the command's width
        bool keeplend   = false; // LLAMA_FN_L22_KEEPLEND: narrow ubatches with prompt rows after them keep the stream's banks
        bool endoverlap = false; // LLAMA_FN_L22_ENDOVERLAP: the prompt's end stays FULL, the answer's first step goes SMALL async
        bool refill     = false; // LLAMA_FN_L22_REFILL: pinned refill sources, the refill set's prefetch, bank 0 pre-staged
        bool readahead  = false; // LLAMA_FN_L22_READAHEAD
        int  ssdtier    = 0;     // LLAMA_FN_L22_SSDTIER (0/1/2)
        bool pin        = false; // LLAMA_FN_L22_PIN (the stream's pinned segments; the OOM callers act whenever pins exist)
        bool graphkey   = false; // LLAMA_FN_L22_GRAPHKEY (G4 key mode on the CUDA backends)
        bool schedbuf   = false; // LLAMA_FN_L22_SCHEDBUF (graph node budget from the measured reserve, qwen4exp)
        bool sampstatic = false; // LLAMA_FN_L22_SAMPSTATIC (sampling nodes counted for static chains)
        bool graphkeep  = false; // GGML_CUDA_FN_L22_GRAPHKEEP >= 1: the pool marks before FULL / at the draft's wide batch
        bool e6         = true;  // [TAG_FN_L22_E6] merged small state reads (exact; LLAMA_FN_L22_STATEMERGE=0 turns it off)
        int  passlog    = 0;
    };
    l22_p4_cfg l22;
    int64_t  l22_after      = -1;    // the hint of the running batch (read once at its start)
    bool     l22_tail       = false; // [TAG_FN_L22_ENDOVERLAP] the running batch is a prompt's narrow last part kept FULL
    bool     l22_last_ub    = false; // [TAG_FN_L22_REFILL] the running ubatch is the prompt's last wide one
    // [TAG_FN_L22_LENDCLASS]
    uint32_t l22_class      = 0;     // the class of the FULL state now (0: none)
    uint32_t l22_w_old      = 0;     // the width the old rule chose at the switch (an upgrade goes there)
    std::map<uint32_t, size_t> l22_class_need; // measured tail need per class width
    uint64_t l22_n_class    = 0, l22_n_upgrade = 0;
    // [TAG_FN_L22_SCHEDBUF] node budget per reserve width (measured)
    mutable std::map<uint32_t, uint32_t> l22_nodes;
    // [TAG_FN_L22_RWIDTH] a prompt that starts while FULL stays (the thrash pin, or a prompt end without an answer step)
    // and plans a wider width than FULL has: FULL goes back to SMALL first, then the batch makes the switch from SMALL
    bool     l22_rw_planned = false; // the running prompt's width was planned (reset by a batch with outputs)
    uint64_t l22_rw_n       = 0, l22_rw_n_nopin = 0;
    int      l22_rw_press   = -1; // [TAG_FN_L22_RWIDTH] pressure pin on (1) / off (0) / not read yet (-1)
    uint64_t l22_rw_n_press = 0;
    bool     l24_pin_press  = false; // [TAG_FN_L24_PRESSLIFT] the pin is a pressure pin
    int      l24_press_lift = -1;    // narrow batches that lift a pressure pin (LLAMA_FN_L24_PRESSLIFT; -1: not read yet)
    uint64_t l24_n_lift     = 0;
    bool     l24_endovl1    = false; // [TAG_FN_L24_ENDOVL1] ENDOVERLAP with one busy sequence of n_seq_max > 1
    // [TAG_FN_L22_ENDOVERLAP] a prompt ended in FULL without the eager switch: the next narrow batch goes SMALL, async
    bool     l22_end_pending = false;
    int      l22_first_n     = 0;    // decode calls still counted after a prompt's switch (first 8)
    int64_t  l22_first_t0    = 0;
    // [TAG_FN_L22_PASSLOG] one record per prompt (FULL period, or a short-path batch); P10 reads l22_rec / l22_rec_seq
    struct l22_pass_rec {
        uint64_t rows = 0, batches = 0, ubatches = 0;
        int      path = 0;           // 0 cpu path (FULL, no stream), 1 stream, 2 short (SMALL)
        uint32_t width = 0, cls = 0; // the FULL width and class
        double   wall_ms = 0.0;      // from the FULL switch's end to the end of the last batch (switches excluded)
        double   full_ms = 0.0, small_ms = 0.0; // the two switches
        double   gate_wait_ms = 0.0;
        uint64_t b_pinned = 0, b_staged = 0, b_ssd = 0, b_d2d = 0, b_ra = 0;
        double   miss_share = 0.0;
        uint32_t miss_layers = 0;
        uint64_t keeps = 0;
        uint64_t faults = 0, page_reads = 0;
        uint32_t n_kv = 0;           // the KV cells used at the end (depth)
    };
    l22_pass_rec l22_rec_cur, l22_rec;
    uint64_t     l22_rec_seq   = 0;
    bool         l22_rec_on    = false;
    int64_t      l22_rec_t0    = 0;
    llama_fn_l22_counters l22_rec_c0 = {};
    void     l22_init();
    void     l22_oom_pre(size_t need);
    bool     l22_oom_retry(size_t need);
    bool     l22_class_upgrade();
    void     l22_graph_procs(bool mark, int window);
    void     l22_rec_begin();
    void     l22_rec_end();
    uint32_t graph_max_nodes_base(uint32_t n_tokens) const;

    // [TAG_FN_L22_P10] round 22, package P10 (stream tuning + G3); every switch off unless set (plan 3.4), qwen4exp target only
    struct l22_p10_cfg {
        int  stagers   = 0;     // LLAMA_FN_L22_STAGERS: 0 off, -1 auto, n >= 2 fixed copy threads of the stream
        int  xover     = 0;     // LLAMA_FN_L22_XOVER: 0 off, 1 the measured crossover, n >= 2 a fixed one
        bool ubplan    = false; // LLAMA_FN_L22_UBPLAN: the FULL width of a prompt from the cost model
        int  bigub2    = 0;     // LLAMA_FN_L22_BIGUB2: the 16K FULL reserve for one prompt sequence (2-slot servers)
        int  smallkeep = 0;     // LLAMA_FN_L22_SMALLKEEP: 1 the SMALL scheduler kept through FULL periods, 2 its bytes in the tail
        bool observe() const { return xover > 0 || ubplan; }
    };
    l22_p10_cfg       l22p;
    llama_fn_l22_cost l22p_cost;            // [TAG_FN_L22_XOVER] [TAG_FN_L22_UBPLAN]
    int32_t  l22p_xover_t     = 0;          // the crossover XOVER set (0: none, the stream's own minimum)
    int64_t  l22p_xover_env   = 0;          // the stream's minimum at the start (0: no lending stream)
    int      l22p_bucket      = 0;          // depth bucket of the running prompt
    int64_t  l22p_depth       = 0;          // its first position
    uint32_t l22p_prompt_seqs = 0;          // [TAG_FN_L22_BIGUB2] sequences of the running batch with more than cbuf_small_t rows
    uint32_t l22p_nseq        = 0;          // sequences of the running batch
    uint32_t l22p_passes      = 0;          // stream passes seen in the FULL period
    bool     l22p_big         = false;      // the FULL period runs at the big width with the borrow out
    double   l22p_big_ms      = 0.0;        // its borrow + unborrow host ms
    int64_t  l22p_rows_plan   = 0;          // [TAG_FN_L22_UBPLAN] the rows the switch planned for (batch + hint)
    double   l22p_est_base    = 0.0, l22p_est_big = 0.0;
    const char * l22p_why     = "";
    const char * l22p_src     = "prior";    // the crossover's source: prior / meas / fixed
    int      l22p_stagers_seen = 0;         // [TAG_FN_L22_STAGERS] copy threads of the last pass
    std::map<uint32_t, size_t> l22p_class_full; // [TAG_FN_L22_SMALLKEEP] measured FULL bytes per class width
    uint64_t l22p_n_xover = 0, l22p_n_big = 0, l22p_n_base = 0, l22p_n_bg2_split = 0;
    bool     l22p_bg2_on      = false;      // [TAG_FN_L22_BIGUB2] the big FULL reserve is made for one sequence
    bool     l22p_bg2_force   = false;      // cbuf_prepare's 1-sequence probe
    // [TAG_FN_L23_TWOSLOT] LLAMA_FN_L23_BG2POOLS: the 1-sequence reserve sizes the k-pool lists for one sequence as well; a
    // batch whose pools could pass that (a cache shared by both sequences) takes the base width
    bool     l23_bg2pools     = false;
    uint64_t l23_n_pools_base = 0;
    bool     l23_pools_fit(uint32_t n_add) const;
    bool     l22p_bg2_base(uint32_t n_tokens); // a batch under the 1-sequence reserve that takes the base width
    // [TAG_FN_L22_SMALLKEEP] the SMALL scheduler, its graphs and its expected sizes while a FULL period runs (parked)
    ggml_backend_sched_ptr              l22_sk_sched;
    std::array<llm_graph_result_ptr, 2> l22_sk_prev;
    std::array<llm_graph_result_ptr, 4> l22_sk_width;
    std::array<llm_graph_result_ptr, 4> l22_sk_noout;
    llm_graph_result_ptr                l22_sk_reserve;
    llm_graph_result *                  l22_sk_active  = nullptr;
    std::vector<size_t>                 l22_sk_exp;
    uint64_t                            l22_sk_hot_gen = 0;
    size_t                              l22_sk_bytes   = 0;
    bool     l22_sk_parked  = false;
    bool     l22_sk_stale   = false; // a reserve ran outside the cbuf code while parked (adapters, embeddings, ...)
    bool     l22_sk_in_cbuf = false;
    int      l22_sk_last    = 0;     // the last FULL switch: 1 parked, 0 not (l22_sk_why)
    const char * l22_sk_why = "";
    uint64_t l22_sk_n_park = 0, l22_sk_n_swap = 0, l22_sk_n_drop = 0, l22_sk_n_nofit = 0;
    void     l22p_init();
    void     l22p_batch_scan();
    void     l22p_period_begin();
    void     l22p_xover_apply();
    uint32_t l22p_plan_width(uint32_t n_tokens, int64_t after, bool vram_check);
    void     l22p_observe(uint32_t rows, bool one_seq, double ms);
    void     l22p_period_end();
    void     l22p_offload_proc(int v);
    int32_t  l22p_stream_min() const;
    bool     l22_sk_park(ggml_backend_dev_t dev, size_t expect_full);
    void     l22_sk_free_full();
    void     l22_sk_swap_in(bool restored);
    void     l22_sk_drop();

    // [TAG_FN_L22_CPUSHARE] round 22, package P16: the CPU share mode this context owns (0: none)
    int      l22_cps_mode   = 0;
    uint64_t l22_cps_n_grow = 0; // compute-buffer growths seen after a share decode
    llama_pfs_share_counters l22_cps_sc0; // the stream's share counters at the last PASSLOG line
    void     l22_cps_log(ggml_backend_sched_t s0, const std::vector<size_t> & b0);

    bool     dft_set_wide(bool wide);
    llama_context * l20_draft() const;
    bool     cbuf_to_small(ggml_backend_dev_t dev, size_t & trimmed, bool & restored, bool async_refill = false); // [TAG_FN_L22_ENDOVERLAP]
    void     cbuf_retry_restore();
    void     cbuf_free_compute();
    size_t   cbuf_compute_bytes(ggml_backend_dev_t dev) const;
    uint32_t reserve_n_tokens() const;

    // [TAG_FN_L3_VRAM_ACCOUNT] LLAMA_FN_VRAM_ACCOUNT=1 (qwen4exp): device use at each stage, the model's device tensors by
    // kind, and the temporary-memory pools of this context's backends
    bool     vram_acc_on   = false;
    size_t   vram_acc_last = 0;
    bool     vram_acc_wide = false; // the last batch was wider than 31 tokens (a prompt)
    size_t   vram_acc_stack = 0;    // the CUDA stack limit last printed
    uint32_t vram_acc_n    = 0;
    void     vram_account(const char * stage);

    // [TAG_FN_L3_VRAM_TRIM] LLAMA_FN_CBUF_DRAFT_TRIM (the MTP draft context of qwen4exp with LLAMA_FN_CBUF=1): the first
    // narrow batch after a prompt (a batch of more than 64 tokens) gives the pools' prompt growth back to the driver
    int      dtrim_on      = -1;    // -1: not decided yet
    bool     dtrim_wide    = false;
    uint64_t dtrim_n       = 0;
    size_t   dtrim_b       = 0;
    void     draft_trim(uint32_t n_tokens_all);

    llm_graph_result * get_gf_res_prev(const llama_ubatch & ubatch);
    void gf_res_prev_reset_all();

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    // keep copies of the per-sequence memory on the device
    std::map<llama_seq_id, llama_memory_buffers> mem_storage;

    bool has_evaluated_once = false;

    // [TAG_SYNC_EARLY_OUT] set by graph_compute() when work is queued on the backends, cleared by synchronize()
    bool sched_pending = false;

    // [TAG_SPEC_PREFILL_TAIL_EXTRACT] when false, decode() does not copy the per-layer inputs to the host
    bool layer_inp_extract = true;

    // env: LLAMA_GRAPH_REUSE_DISABLE
    bool graph_reuse_disable = false;

    // true after the prism.hadamard coverage check passes on a reserved graph
    bool hadamard_verified = false;

    // perf
    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0; // number of eval calls

    mutable int32_t n_reused = 0; // number of times the previous graph was reused
};
