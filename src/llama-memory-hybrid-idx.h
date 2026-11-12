#pragma once

#include "llama-memory-hybrid.h"

#include <array>
#include <limits>
#include <memory>
#include <string>
#include <vector>

//
// llama_memory_hybrid_idx
//

// llama_memory_hybrid plus a third cache with one indexer key per token, for block-sparse attention (qwen4exp QSA)
// the indexer is a side buffer over the attention cells: same size, padding, streams and slots, so cell j is one token in both

class llama_memory_hybrid_idx : public llama_memory_hybrid {
public:
    llama_memory_hybrid_idx(
        const llama_model & model,
                     bool   no_alloc,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
                            /* the indexer cache exists only if this is given */
    const layer_filter_cb & filter_idx);

    // Defined out of line because kpool_layout is incomplete here.
    ~llama_memory_hybrid_idx();

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    //
    // llama_memory_hybrid_idx specific API
    //

    // [TAG_R40_NP2KVPARK] the attention and indexer cells of seq_id go, the recurrent cell stays
    bool seq_rm_attn_only(llama_seq_id seq_id) override;

    llama_kv_cache * get_mem_idx() const;   // nullptr when the model carries no indexer

    // The model's indexer pool size.
    uint32_t get_kpool() const { return hparams_idx.indexer_kpool; }

    // Whether pools are kpool consecutive cells in sequence order (qwen4exp) instead of kpool consecutive positions.
    bool get_kpool_by_order() const { return hparams_idx.indexer_kpool_by_order; }

    // Which cells of a sequence make up which pool of kpool consecutive positions (or cells, in order mode).
    // It is kept here because it outlives the batch: pools are fixed by the positions relative to the
    // sequence's first one, so a ubatch only ever appends to it. Sequence edits drop it, see mem_idx_stale.
    struct kpool_layout;
    struct kpool_layout_seq; // [TAG_FN_R1_KPOOL_TAIL] one sequence of it

    const kpool_layout & kpool_layout_update();
    const kpool_layout & kpool_layout_get() const;

    // The pooled keys persist in the idx cache across batches. A sequence edit can regroup the pools
    // from some position on, which stales every pooled key at or after it. POS_CLEAN means none.
    using stale_pos_t = std::array<llama_pos, LLAMA_MAX_SEQ>;

    static constexpr llama_pos POS_CLEAN = std::numeric_limits<llama_pos>::max();

    static stale_pos_t stale_pos_clean() {
        stale_pos_t res;
        res.fill(POS_CLEAN);
        return res;
    }

    const stale_pos_t & mem_idx_stale_get() const { return mem_idx_stale; }
    void mem_idx_stale_clear() { mem_idx_stale.fill(POS_CLEAN); }

    // [TAG_FN_L3_HOST_DIAG] LLAMA_FN_HOST_KPOOL_PROBE=1 on a qwen4exp model (llama-fn-auto.h): time the k-pool host work
    bool kpool_probe_on() const { return kpool_probe; }

    // [TAG_FN_L10_KPOOLINC] qwen4exp (LLAMA_FN_L10_KPOOLINC, default 1): the k-pool input arrays kept per sequence and
    // updated at the layout's tail instead of rebuilt over every pool on each ubatch
    bool kpool_inc_on() const { return kpool_inc; }

    // [TAG_FN_L22_QSA_SEQPOOLS] qwen4exp (LLAMA_FN_L22_QSA_SEQPOOLS, 0 off, 1 on, 2 on + check against the full lists): a
    // single-sequence ubatch lists only its sequence's pools (see llama_fn_qsa_seqpools_width below)
    int  qsa_seqpools() const { return seqpools; }
    // the top-k picks the same set from a shorter list (unordered radix select on CUDA, ties keep the lowest index)
    bool qsa_seqpools_ok() const { return seqpools_ok; }
    // pools the indexer picks per query (indexer_top_k / kpool)
    uint32_t qsa_n_top() const { return n_top_pool; }
    // [TAG_FN_L22_QSA_SHAREPFX] qwen4exp (LLAMA_FN_L22_QSA_SHAREPFX): an aligned shared prefix keeps the pooled-key cache
    bool qsa_sharepfx() const { return sharepfx; }

    // [TAG_FN_L22_QSA_SEQPOOLS] PASSLOG level-2 counters (the inference thread)
    struct qsa_stats {
        uint64_t ubatches  = 0; // k-pool states built
        uint64_t eligible  = 0; // single-sequence, cache-safe, one stream
        uint64_t filtered  = 0; // listed one sequence
        uint64_t pools_lst = 0; // sum of listed widths
        uint64_t pools_all = 0; // sum of full widths of the filtered ubatches
        uint64_t pfx_kept  = 0; // states whose cache stayed safe through SHAREPFX
        uint64_t checked   = 0; // ubatches checked (QSA_SEQPOOLS=2)
    };
    qsa_stats & qsa_stats_mut() const { return qsa_st; }

    // [TAG_FN_L23_TWOSLOT] the pool count a graph reserve sizes for n_seqs sequences (0: n_seq_max), and an upper bound of
    // the pools once n_add more (token, sequence) pairs are in: (every sequence's cells + n_add) / kpool; 0: no indexer
    uint64_t kpool_reserve_pools(uint32_t n_seqs) const;
    uint64_t kpool_bound(uint32_t n_add) const;

private:
    // forget seq_id (all of it if seq_id < 0) in every cache at once, so a failed restore cannot leave the caches out of step
    // seq_id < 0 drops the whole context, as the caches themselves do on a failed restore
    void state_drop(llama_seq_id seq_id);

    // the indexer cache holds one key head per layer, so it needs its own hparams:
    // llama_kv_cache keeps a reference to what it is given
    llama_hparams hparams_idx;

    const std::unique_ptr<llama_kv_cache> mem_idx;

    // unique_ptr because kpool_layout is incomplete here
    std::unique_ptr<kpool_layout> kpool_lay;

    bool kpool_probe = false; // [TAG_FN_L3_HOST_DIAG]
    bool kpool_inc   = false; // [TAG_FN_L10_KPOOLINC]

    int      seqpools    = 0;     // [TAG_FN_L22_QSA_SEQPOOLS]
    bool     seqpools_ok = false;
    uint32_t n_top_pool  = 0;
    bool     sharepfx    = false; // [TAG_FN_L22_QSA_SHAREPFX]
    mutable qsa_stats qsa_st;

    // whether the current layout has cells shared between sequences (kpool_layout is incomplete here, so out of line)
    bool kpool_layout_shared() const;

    // [TAG_FN_L22_QSA_SHAREPFX] whether seq_rm(seq_id, p0, p1) would take a shared cell away (before the removal)
    bool kpool_rm_takes_shared(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const;

    // seq_id < 0 stales every sequence, p0 < 0 stales the sequence from its first position
    void mem_idx_stale_set(llama_seq_id seq_id, llama_pos p0);

    // the position an edit at p0 stales the sequence from
    llama_pos mem_idx_stale_pos(llama_seq_id seq_id, llama_pos p0) const;

    stale_pos_t mem_idx_stale = stale_pos_clean();
};

class llama_memory_hybrid_idx_context : public llama_memory_hybrid_context {
public:
    class kpool_access {
    public:
        ggml_tensor * gather_key_gate(ggml_tensor * idxs) const;
        ggml_tensor * scatter_pooled(ggml_tensor * values, ggml_tensor * idxs) const;
        ggml_tensor * gather_pooled(ggml_tensor * idxs) const;

    private:
        friend class llama_memory_hybrid_idx_context;

        kpool_access(ggml_context * ctx, ggml_tensor * k, int64_t n_embd);

        ggml_context * ctx;
        ggml_tensor  * key_gate;
        ggml_tensor  * pooled;
    };

    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // used for errors
    explicit llama_memory_hybrid_idx_context(llama_memory_status status);

    // used to create a full-cache context
    explicit llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem);

    // used to create an update context
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                      llama_context * lctx,
                               bool   optimize);

    // used to create a batch processing context from a batch
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                    slot_info_vec_t   sinfos_attn,
                    slot_info_vec_t   sinfos_idx,
          std::vector<llama_ubatch>   ubatches);

    ~llama_memory_hybrid_idx_context(); // Defined out of line because kpool_state is incomplete here.

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    //
    // llama_memory_hybrid_idx_context specific API
    //

    // nullptr with no indexer
    const llama_kv_cache_context * get_idx() const;

    // streams in the current slot info, the `ns` of get_k/get_v; 1 if unified
    uint32_t get_n_stream() const;

    // glm5-next and qwen4exp, complete pools of kpool cells per sequence, scored as whole pools.
    uint32_t get_n_kpool    () const; // Padded pool count, where the last pool is always unused.
    uint32_t get_n_kpool_new() const; // Pools to re-pool this ubatch, padded to a stable bound, never below 1.
    bool get_kpool_cache_safe() const;
    kpool_access get_kpool_access(ggml_context * ctx, int32_t il, int64_t n_embd) const;
    ggml_tensor * gather_mla_rows(ggml_context * ctx, ggml_tensor * idxs, int64_t n_rows, int64_t n_embd, int32_t il) const;
    // new_pool_pos (I32 [4*n_new]): M-RoPE position of each new pool's first member, for pooled keys rotated at pooling time
    void set_input_kpool(ggml_tensor * pool_cells, ggml_tensor * pool_idxs, ggml_tensor * pool_mask, ggml_tensor * tail_idxs,
                         ggml_tensor * gather_mask, bool gather, ggml_tensor * new_pool_idxs, ggml_tensor * new_pool_rep,
                         const llama_ubatch * ubatch, ggml_tensor * new_pool_pos = nullptr) const;

private:
    llama_memory_hybrid_idx * mem = nullptr;

    // streams per ubatch, read from the slot infos before ctx_idx takes them
    // declared first, so it is initialised while sinfos_idx is still intact
    const std::vector<uint32_t> ns_ubatch;

    // the indexer cells of each ubatch, kept for pools in cache order (qwen4exp): token s*n + i of ubatch u
    // sits in cell idxs[s][i] of stream strm[s] of sinfos_kpool[u], and several cells can share a position
    const slot_info_vec_t sinfos_kpool;

    // null unless the model has an indexer
    const llama_memory_context_ptr ctx_idx;

    // mirrors the base class's ubatch cursor, which is private there
    size_t i_cur = 0;

    // Which pools of the layout this ubatch must re-pool. The layout itself belongs to the memory.
    struct kpool_state;
    kpool_state kpool_build_sizes() const;
    void kpool_build_state(const llama_ubatch & ubatch);
    const kpool_state & kpool_cur() const;

    // unique_ptr because kpool_state is incomplete here.
    std::unique_ptr<kpool_state> kpool_st;

    // The ubatch kpool_st was built for, guards against reads before apply.
    size_t i_kpool = SIZE_MAX;

    // Whether this context tracks k-pool states.
    bool kpool_track() const;

    // Positions each sequence must re-pool from, cleared only after the first ubatch succeeds
    llama_memory_hybrid_idx::stale_pos_t mem_idx_stale_batch = llama_memory_hybrid_idx::stale_pos_clean();
};

// [TAG_FN_L23_TWOSLOT] the full contexts (graph reserve) this thread makes next size the k-pool lists for n_seqs sequences
// (0: n_seq_max). The 1-sequence 16K FULL reserve of a 2-slot server sets 1 (its decode keeps the bound, kpool_bound)
void llama_fn_kpool_reserve_seqs(uint32_t n_seqs);

//
// [TAG_FN_L22_QSA_SEQPOOLS] [TAG_FN_L22_QSA_SHAREPFX] round 22 P13: pure pieces of the per-sequence pool lists, exported
// for tests/test-fn-l22-kvrows.cpp
//

// the padded pool count of llama_memory_hybrid_idx (kpool_pad) for a step of pad_g pools: max(pad_g, PAD(n + 1, pad_g))
LLAMA_API uint32_t llama_fn_qsa_kpool_pad(uint32_t n_pool, uint32_t pad_g);

// The list width of a single-sequence ubatch that lists only its sequence's n_seq pools instead of every sequence's
// n_all (the full width is kpool_pad(n_all)), or 0 to keep the full list. The picked set must stay the same:
//   - the number of picks n_top_pool = min(width, n_top) stays n_top (so n_sel, the FA budget, is unchanged): width >= n_top
//   - the top-k ties (many visible pools score exactly 0) resolve by index in the CUDA radix select only for ncols >= 2048
//     (ggml-cuda/top-k.cu, [TAG_TOPK_SELECT] / [TAG_FN_L3_GPU_TOPK]); below that the CUB / bitonic sort resolves them by
//     position. So the full width must be >= 2048, and the listed width >= 2048 or exactly n_top (k == ncols: all picked)
//   - the sequence's pools keep their relative order (they are the same columns, shifted), foreign pools score -INF
LLAMA_API uint32_t llama_fn_qsa_seqpools_width(uint32_t n_all, uint32_t n_seq, uint32_t n_top, uint32_t pad_g);

// QSA_SEQPOOLS=2: the listed inputs against the full ones (the same ubatch, no listing). pcell / pidx / mask of the
// listed width n_pool_l (n_list real entries) vs the full width n_pool_f where the sequence's pools start at gstart;
// masks as float [n_pool, n_tokens]; the padded entries hold dummy / sentinel. "" when they agree, else the first
// difference.
struct llama_fn_qsa_lists {
    const int32_t * pcell = nullptr; // [n_pool]
    const int32_t * pidx  = nullptr; // [kpool, n_pool]
    const float   * mask  = nullptr; // [n_pool, n_tokens]
    uint32_t        n_pool = 0;
};
LLAMA_API std::string llama_fn_qsa_seqpools_check(const llama_fn_qsa_lists & lst, uint32_t n_list, const llama_fn_qsa_lists & full,
                                                  uint32_t gstart, uint32_t kpool, uint32_t n_tokens, int32_t dummy, int32_t sentinel);

// [TAG_FN_L22_QSA_SHAREPFX] every rep cell names one member set: pools that share a rep (a shared prefix pool, listed
// once per sharing sequence) have the same members. "" when that holds for the n pools, else the first offender.
LLAMA_API std::string llama_fn_qsa_rep_members_check(const int32_t * pcell, const int32_t * pidx, uint32_t n, uint32_t kpool);

// [TAG_FN_L22_QSA_SHAREPFX] the shared-cell shape of one sequence, kept while its layout appends: the leading run of
// shared cells (pfx_n cells, seq_count between cmin and cmax), and whether a shared cell sits after that run
struct llama_fn_qsa_pfx {
    bool     shared = false;
    bool     irr    = false;
    uint32_t pfx_n  = 0;
    int32_t  cmin   = 0;
    int32_t  cmax   = 0;
};

// cell j of the sequence (cells in position order, j counting from 0) has seq_count cnt; false once the shape is irregular
LLAMA_API bool llama_fn_qsa_pfx_step(llama_fn_qsa_pfx & st, size_t j, int32_t cnt);

// whether the pooled-key cache stays exact with the shared cells of these sequences (one entry per sequence that holds
// cells; pos_min the first position of each): every sequence holding a shared cell has a regular shape, the same run
// length, the same first position, and every cell of the run belongs to exactly the |S| sharing sequences. Then those
// runs are the same cells in the same order (a shared cell's position is the cell's), so every pool lies inside the
// prefix (the same members and rep in every sharing sequence) or ends in a sequence's own cells (its own rep).
LLAMA_API bool llama_fn_qsa_sharepfx_safe(const llama_fn_qsa_pfx * seqs, const int32_t * pos_min, int n);
