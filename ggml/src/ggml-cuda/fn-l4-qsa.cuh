#pragma once

// [TAG_FN_L4_QSA] Flash-Next (qwen4exp) lever round 4, the QSA layers: the CUDA side of the marks in ggml-fn-l4-qsa.h.
// Nothing here runs for a graph without them. Every path gives the same bits as the unmarked graph (same kernels or the
// same arithmetic, other launch order or stream only). GGML_CUDA_FN_L4_QSA=0 turns all of them off in one binary.

#include "common.cuh"
#include "ggml-fn-l4-qsa.h"
#include "ggml-backend-impl.h"

// false when GGML_CUDA_FN_L4_QSA=0
bool ggml_cuda_fn_l4_qsa_enabled();

// graph_optimize, before allocation: the allocation dependencies of IDXQ8 (IDXDEP) and SEL (SELDEP) for every graph
void ggml_cuda_fn_l4_qsa_deps(ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params);

// graph_optimize: true when a QSA layer of the graph asks for STREAMS
bool ggml_cuda_fn_l4_qsa_streams_wanted(const ggml_cgraph * cgraph);

// graph_optimize: one concurrent event per marked QSA layer (indexer chain on stream 1, q / k / v chain on stream 0,
// joined at the layer's FLASH_ATTN_EXT), with the allocation dependencies that keep every tensor the two branches touch
// allocated until the join. The caller reset the stream context.
void ggml_cuda_fn_l4_qsa_streams(ggml_backend_cuda_context * ctx, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params);

// eval: the SEL fusion at TOP_K node i (the picks, their scores, the selected cells and the tail in the top-k's merge).
// Returns the number of following nodes it computed (0: not taken, nothing launched).
int ggml_cuda_fn_l4_qsa_try_sel(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i);

// eval: the KVW fusion at turbot SET_ROWS node i (K) and the V node after it in one launch. Returns the nodes skipped.
int ggml_cuda_fn_l4_qsa_try_kvw(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i);

// eval: the POOL fusion at the marked FILL node i (the k-pool update chain). Returns the nodes skipped.
int ggml_cuda_fn_l4_qsa_try_pool(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i);

// [TAG_FN_L4_QSA_POOL] rope.cu: the k-pool update of a QSA layer in one launch (fn-l4-qsa.cu fills the arguments from the
// nodes; rope is the ROPE node, whose parameters the launcher reads as ggml_cuda_op_rope_impl does). false: not taken.
struct ggml_cuda_fn_l4_pool_args {
    const float   * k_raw           = nullptr; // [d, n_tok], row stride k_raw_ld floats
    int64_t         k_raw_ld        = 0;
    const void    * k_idxs          = nullptr; // [n_tok] cache cell of each token's row
    int64_t         k_idxs_ld       = 0;
    bool            k_idxs_i64      = true;
    char          * cache           = nullptr; // q8_0 rows of w1_row_elems (raw key | pooled key)
    int64_t         cache_row_bytes = 0;
    int64_t         pooled_off      = 0;       // byte offset of the pooled key in a row
    int32_t         w1_row_elems    = 0;       // 2*d
    const int32_t * new_idxs        = nullptr; // [kpool, n_new]: member cells of each pool to re-pool
    const void    * rep             = nullptr; // [n_new] cell that caches each new pooled key
    int64_t         rep_ld          = 0;
    bool            rep_i64         = true;
    const int32_t * pos             = nullptr; // [4*n_new] M-RoPE positions
    const float   * norm_w          = nullptr; // [d]
    int32_t         d               = 0;
    int32_t         n_tok           = 0;
    int32_t         n_new           = 0;
    int32_t         kpool           = 0;
    float           scale           = 1.0f;    // the SCALE node: scale*x + bias
    float           scale_bias      = 0.0f;
    float           eps             = 0.0f;    // the RMS_NORM node
    // the ROPE node's, set by the launcher
    int32_t         n_dims          = 0;
    int32_t         n_offs          = 0;
    bool            is_imrope       = false;
    float           theta_scale     = 0.0f;
    float           freq_scale      = 1.0f;
    float           ext_factor      = 0.0f;
    float           attn_factor     = 1.0f;
};

bool ggml_cuda_fn_l4_qsa_pool_launch(ggml_backend_cuda_context & ctx, ggml_cuda_fn_l4_pool_args a, const ggml_tensor * rope);

// top-k.cu: the unordered top k that ggml_cuda_op_top_k computes for dst with the [TAG_FN_L3_GPU_TOPK] path, plus the
// outputs of the gathers / concat that read it. Same stage 1, same select, same order as k_top_k_fn_l3_merge.
struct ggml_cuda_fn_l4_sel_args {
    const float   * score     = nullptr; // [n_pool, n_rows] contiguous
    int64_t         n_pool    = 0;
    int64_t         n_rows    = 0;
    int64_t         k         = 0;
    const int32_t * pool_idxs = nullptr; // [kpool, n_pool]: the member cells of each pool
    int64_t         pool_ld   = 0;       // int32 per pool row
    int32_t         kpool     = 0;
    const int32_t * tail      = nullptr; // [n_tail, n_rows]
    int64_t         tail_ld   = 0;
    int32_t         n_tail    = 0;
    int32_t       * topk      = nullptr; // [k, n_rows]
    float         * ts        = nullptr; // [1, k, n_rows]: score of each pick
    int32_t       * gs        = nullptr; // [kpool, k*n_rows]: member cells of each pick
    int32_t       * cc        = nullptr; // [kpool*k + n_tail, n_rows]: gs rows, then the tail
    int64_t         cc_ld     = 0;
};

// true when ggml_cuda_op_top_k takes the [TAG_FN_L3_GPU_TOPK] two-stage path for dst
bool ggml_cuda_top_k_takes_fn_l3(const ggml_tensor * dst);

// false: not launched (the shape is not one the [TAG_FN_L3_GPU_TOPK] path takes)
bool ggml_cuda_fn_l4_qsa_sel_launch(ggml_cuda_pool & pool, cudaStream_t stream, const ggml_cuda_fn_l4_sel_args & a);
