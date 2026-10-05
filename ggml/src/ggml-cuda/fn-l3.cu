// [TAG_FN_L3_GPU] Flash-Next (qwen4exp) device-time levers of lever round 3, see fn-l3.cuh and ggml-fn-l3.h.

#include "fn-l3.cuh"
#include "ggml-impl.h"

#include <atomic>
#include <cmath>

bool ggml_cuda_fn_l3_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_FN_L3");
        return !(e && e[0] == '0');
    }();
    return on;
}

void ggml_cuda_fn_l3_note(int path, const char * what) {
    static std::atomic<bool> seen[GGML_CUDA_FN_L3_PATH_COUNT];
    if (path < 0 || path >= GGML_CUDA_FN_L3_PATH_COUNT || seen[path].exchange(true)) {
        return;
    }
    GGML_LOG_INFO("ggml_cuda: [TAG_FN_L3_GPU] %s\n", what);
}

// ---------------------------------------------------------------------------------------------------------------------
// [TAG_FN_L3_GPU_HCFUSE] the hc chains of the qwen4exp graph in one launch each. The arithmetic is the unfused ops',
// one op after the other in f32: scale_f32 (scale*x + bias), op_sigmoid (1/(1 + exp(-x))), op_silu (x/(1 + exp(-x)))
// and dsv4_hc_post_f32 without comb (x*post, then + residual).
// ---------------------------------------------------------------------------------------------------------------------

// scale -> sigmoid -> scale
static __global__ void k_fn_l3_hcw3(const float * x, float * dst, const float s1, const float b1, const float s2,
        const float b2, const int64_t n) {
    ggml_cuda_pdl_lc();
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    ggml_cuda_pdl_sync();
    const float t  = s1 * x[i] + b1;
    const float sg = 1.0f / (1.0f + expf(-t));
    dst[i] = s2 * sg + b2;
}

// scale -> sigmoid -> scale -> dsv4_hc_post (no comb): the post weight of (idst, it) comes from the raw inject
static __global__ void k_fn_l3_hcw4(
        const float * inject, const float * x, const float * residual, float * dst,
        const float s1, const float b1, const float s2, const float b2,
        const int64_t n_embd, const int64_t hc, const int64_t n_tokens,
        const int64_t si0, const int64_t si1,
        const int64_t sx0, const int64_t sx1,
        const int64_t sr0, const int64_t sr1, const int64_t sr2,
        const int64_t sd0, const int64_t sd1, const int64_t sd2) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * hc * n_tokens;
    if (ir >= nr) {
        return;
    }
    ggml_cuda_pdl_sync();

    const int64_t i0   = ir % n_embd;
    const int64_t idst = (ir / n_embd) % hc;
    const int64_t it   = ir / (n_embd * hc);

    const float t    = s1 * inject[idst*si0 + it*si1] + b1;
    const float sg   = 1.0f / (1.0f + expf(-t));
    const float post = s2 * sg + b2;

    float sum = x[i0*sx0 + it*sx1] * post;
    sum += residual[i0*sr0 + idst*sr1 + it*sr2];

    dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
}

// scale -> silu
static __global__ void k_fn_l3_hclo(const float * x, float * dst, const float s, const float b, const int64_t n) {
    ggml_cuda_pdl_lc();
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    ggml_cuda_pdl_sync();
    const float t = s * x[i] + b;
    dst[i] = t / (1.0f + expf(-t));
}

// [TAG_FN_L3_GPU_Q8F] quantize_q8_1 of one q8_1 block held by a warp: lane l holds element l. The same operations in the
// same order as quantize_q8_1 (quantize.cu), so the same bytes. Every lane of the warp must call it.
static __device__ __forceinline__ void fn_l3_q8_1_store(const float xi, block_q8_1 * yb, const int lane) {
    float amax = fabsf(xi);
    float sum  = xi;

    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    yb->qs[lane] = q;

    if (lane > 0) {
        return;
    }

    yb->ds = make_half2(d, sum);
}

// [TAG_FN_L3_GPU_Q8F] scale -> silu with the q8_1 copy: thread (i0, r) of the padded rows as in quantize_q8_1; element
// (i0, r) is flat element r*ne10 + i0 of x and of the output (both contiguous), the padding quantizes 0
static __global__ void k_fn_l3_hclo_q8(const float * x, float * dst, const float s, const float b, const int64_t ne10,
        block_q8_1 * q8, const int64_t bpr) {
    ggml_cuda_pdl_lc();
    const int64_t i0 = (int64_t) blockDim.x*blockIdx.x + threadIdx.x;
    if (i0 >= bpr*QK8_1) {
        return;
    }
    const int64_t r = blockIdx.y;
    ggml_cuda_pdl_sync();
    float v = 0.0f;
    if (i0 < ne10) {
        const float t = s * x[r*ne10 + i0] + b;
        v = t / (1.0f + expf(-t));
        dst[r*ne10 + i0] = v;
    }
    fn_l3_q8_1_store(v, q8 + r*bpr + i0/QK8_1, threadIdx.x % WARP_SIZE);
}

static float fn_l3_op_f32(const ggml_tensor * t, int slot) {
    float v;
    memcpy(&v, (const float *) t->op_params + slot, sizeof(float));
    return v;
}

void ggml_cuda_fn_l3_hcw3(ggml_backend_cuda_context & ctx, const ggml_tensor * s1, ggml_tensor * s2) {
    const ggml_tensor * x = s1->src[0];
    const int64_t n = ggml_nelements(x);
    const int bs = 256;
    const ggml_cuda_kernel_launch_params lp((dim3) ((n + bs - 1)/bs), bs, 0, ctx.stream());
    ggml_cuda_kernel_launch(k_fn_l3_hcw3, lp, (const float *) x->data, (float *) s2->data,
            fn_l3_op_f32(s1, 0), fn_l3_op_f32(s1, 1), fn_l3_op_f32(s2, 0), fn_l3_op_f32(s2, 1), n);
    ggml_cuda_fn_l3_note(GGML_CUDA_FN_L3_PATH_HCW3, "HCFUSE: hc combine weights (scale -> sigmoid -> scale) in one launch");
}

void ggml_cuda_fn_l3_hcw4(ggml_backend_cuda_context & ctx, const ggml_tensor * s1, const ggml_tensor * s2, ggml_tensor * post) {
    const ggml_tensor * inject   = s1->src[0];
    const ggml_tensor * x        = post->src[0];
    const ggml_tensor * residual = post->src[1];

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t hc       = residual->ne[1];

    const int bs = 256;
    const int64_t nr = n_embd * hc * n_tokens;
    const ggml_cuda_kernel_launch_params lp((dim3) ((nr + bs - 1)/bs), bs, 0, ctx.stream());
    ggml_cuda_kernel_launch(k_fn_l3_hcw4, lp,
            (const float *) inject->data, (const float *) x->data, (const float *) residual->data, (float *) post->data,
            fn_l3_op_f32(s1, 0), fn_l3_op_f32(s1, 1), fn_l3_op_f32(s2, 0), fn_l3_op_f32(s2, 1),
            n_embd, hc, n_tokens,
            (int64_t) (inject->nb[0]/sizeof(float)), (int64_t) (inject->nb[1]/sizeof(float)),
            (int64_t) (x->nb[0]/sizeof(float)), (int64_t) (x->nb[1]/sizeof(float)),
            (int64_t) (residual->nb[0]/sizeof(float)), (int64_t) (residual->nb[1]/sizeof(float)), (int64_t) (residual->nb[2]/sizeof(float)),
            (int64_t) (post->nb[0]/sizeof(float)), (int64_t) (post->nb[1]/sizeof(float)), (int64_t) (post->nb[2]/sizeof(float)));
    ggml_cuda_fn_l3_note(GGML_CUDA_FN_L3_PATH_HCW4, "HCFUSE: hc combine (scale -> sigmoid -> scale -> hc post) in one launch");
}

void ggml_cuda_fn_l3_hclo(ggml_backend_cuda_context & ctx, const ggml_tensor * s, ggml_tensor * silu) {
    const ggml_tensor * x = s->src[0];
    const int64_t n = ggml_nelements(x);
    const int bs = 256;

    // [TAG_FN_L3_GPU_Q8F] the silu marked Q8OUT: also the q8_1 copy for the mat-vec that reads it
    ggml_cuda_fn_l3_q8_dst q8;
    if (ggml_fn_l3_get(silu) == GGML_FN_L3_Q8OUT && ggml_nelements(silu) == n && ggml_cuda_mmvq_q8_produce(ctx, silu, q8, /*padding_ok =*/ true)) {
        GGML_ASSERT(q8.ne10*q8.nrows == n);
        const ggml_cuda_kernel_launch_params lpq(dim3((unsigned) ((q8.bpr*QK8_1 + bs - 1)/bs), (unsigned) q8.nrows, 1), bs, 0,
                ctx.stream());
        ggml_cuda_kernel_launch(k_fn_l3_hclo_q8, lpq, (const float *) x->data, (float *) silu->data,
                fn_l3_op_f32(s, 0), fn_l3_op_f32(s, 1), q8.ne10, q8.y, q8.bpr);
        ggml_cuda_fn_l3_note(GGML_CUDA_FN_L3_PATH_Q8F_HCLO, "Q8F: hc mixer low rank (scale -> silu) writes its q8_1 copy");
        return;
    }

    const ggml_cuda_kernel_launch_params lp((dim3) ((n + bs - 1)/bs), bs, 0, ctx.stream());
    ggml_cuda_kernel_launch(k_fn_l3_hclo, lp, (const float *) x->data, (float *) silu->data,
            fn_l3_op_f32(s, 0), fn_l3_op_f32(s, 1), n);
    ggml_cuda_fn_l3_note(GGML_CUDA_FN_L3_PATH_HCLO, "HCFUSE: hc mixer low rank (scale -> silu) in one launch");
}

// ---------------------------------------------------------------------------------------------------------------------
// [TAG_FN_L3_GPU_COMPACT] the sparse-index compaction of fattn.cu on many blocks. flash_attn_mask_to_sparse_indices
// walks a whole row in one block, chunk after chunk, four barriers per chunk (75 us per call at 246K cells). Here:
//   count: block c counts the selected columns of chunk c (FN_L3_COMPACT_CHUNK columns);
//   write: block c writes its columns at the sum of the counts before it, in column order; block 0 writes the -1 tail
//          and the list's count.
// A column is selected as in the original (any query of the group sees it), so the lists and counts are the same.
// ---------------------------------------------------------------------------------------------------------------------

#define FN_L3_COMPACT_THREADS 256
#define FN_L3_COMPACT_VPL     8
#define FN_L3_COMPACT_CHUNK   (FN_L3_COMPACT_THREADS*FN_L3_COMPACT_VPL)

template <int ncols1, bool oob>
static __device__ __forceinline__ bool fn_l3_compact_sel(const half * mask, const int i, const int ne30, const int nq,
        const int64_t s31) {
    bool selected = false;
    if (i < ne30) {
#pragma unroll
        for (int q = 0; q < ncols1; ++q) {
            selected |= (!oob || q < nq) && isfinite(__half2float(mask[q*s31 + i]));
        }
    }
    return selected;
}

// grid (n_chunks, n_groups, n_seq)
template <int ncols1, bool oob>
__launch_bounds__(FN_L3_COMPACT_THREADS, 1)
static __global__ void k_fn_l3_compact_count(const half * mask_ptr, int32_t * chunk_counts, const int ne30,
        const int n_queries, const int64_t s31, const int64_t s33) {
    const int chunk    = blockIdx.x;
    const int group    = blockIdx.y;
    const int sequence = blockIdx.z;
    const int tid      = threadIdx.x;
    const int warp     = tid / WARP_SIZE;
    const int lane     = tid % WARP_SIZE;

    const int q0 = group*ncols1;
    const int nq = min(q0 + ncols1, n_queries) - q0;
    const half * mask = mask_ptr + sequence*s33 + q0*s31;

    const int i0 = chunk*FN_L3_COMPACT_CHUNK;
    int warp_count = 0;
#pragma unroll
    for (int item = 0; item < FN_L3_COMPACT_VPL; ++item) {
        const int i = i0 + (warp*FN_L3_COMPACT_VPL + item)*WARP_SIZE + lane;
        warp_count += __popc(__ballot_sync(0xFFFFFFFF, fn_l3_compact_sel<ncols1, oob>(mask, i, ne30, nq, s31)));
    }

    __shared__ int s_warp[FN_L3_COMPACT_THREADS/WARP_SIZE];
    if (lane == 0) {
        s_warp[warp] = warp_count;
    }
    __syncthreads();
    if (tid == 0) {
        int sum = 0;
#pragma unroll
        for (int w = 0; w < FN_L3_COMPACT_THREADS/WARP_SIZE; ++w) {
            sum += s_warp[w];
        }
        chunk_counts[(int64_t(sequence)*gridDim.y + group)*gridDim.x + chunk] = sum;
    }
}

template <int ncols1, bool oob>
__launch_bounds__(FN_L3_COMPACT_THREADS, 1)
static __global__ void k_fn_l3_compact_write(const half * mask_ptr, const int32_t * chunk_counts, int32_t * indices_ptr,
        int32_t * counts_ptr, const int ne30, const int n_queries, const int n_kv_max, const int64_t s31, const int64_t s33) {
    const int chunk    = blockIdx.x;
    const int group    = blockIdx.y;
    const int sequence = blockIdx.z;
    const int n_chunks = gridDim.x;
    const int tid      = threadIdx.x;
    const int warp     = tid / WARP_SIZE;
    const int lane     = tid % WARP_SIZE;

    const int q0 = group*ncols1;
    const int nq = min(q0 + ncols1, n_queries) - q0;
    const half * mask = mask_ptr + sequence*s33 + q0*s31;

    const int64_t list = int64_t(sequence)*gridDim.y + group;
    const int32_t * cc = chunk_counts + list*n_chunks;
    int32_t * indices  = indices_ptr + list*n_kv_max;

    __shared__ int s_warp[FN_L3_COMPACT_THREADS/WARP_SIZE];
    __shared__ int s_base;
    __shared__ int s_total;

    // the counts before this chunk and of the whole row: one warp, lanes stride over the chunks
    if (warp == 0) {
        int before = 0;
        int total  = 0;
        for (int c = lane; c < n_chunks; c += WARP_SIZE) {
            const int v = cc[c];
            before += c < chunk ? v : 0;
            total  += v;
        }
        before = warp_reduce_sum<WARP_SIZE>(before);
        total  = warp_reduce_sum<WARP_SIZE>(total);
        if (lane == 0) {
            s_base  = before;
            s_total = total;
        }
    }

    const int i0 = chunk*FN_L3_COMPACT_CHUNK;
    uint32_t selected_warp[FN_L3_COMPACT_VPL];
    int warp_count = 0;
#pragma unroll
    for (int item = 0; item < FN_L3_COMPACT_VPL; ++item) {
        const int i = i0 + (warp*FN_L3_COMPACT_VPL + item)*WARP_SIZE + lane;
        selected_warp[item] = __ballot_sync(0xFFFFFFFF, fn_l3_compact_sel<ncols1, oob>(mask, i, ne30, nq, s31));
        warp_count += __popc(selected_warp[item]);
    }
    if (lane == 0) {
        s_warp[warp] = warp_count;
    }
    __syncthreads();

    int warp_offset = s_base;
    for (int w = 0; w < warp; ++w) {
        warp_offset += s_warp[w];
    }

    const uint32_t lane_mask = lane == 0 ? 0 : (1u << lane) - 1;
    int item_offset = 0;
#pragma unroll
    for (int item = 0; item < FN_L3_COMPACT_VPL; ++item) {
        const int i   = i0 + (warp*FN_L3_COMPACT_VPL + item)*WARP_SIZE + lane;
        const int dst = warp_offset + item_offset + __popc(selected_warp[item] & lane_mask);
        if ((selected_warp[item] & (uint32_t(1) << lane)) && dst < n_kv_max) {
            indices[dst] = i;
        }
        item_offset += __popc(selected_warp[item]);
    }

    if (chunk == 0) {
        const int count = min(s_total, n_kv_max);
        for (int i = count + tid; i < n_kv_max; i += FN_L3_COMPACT_THREADS) {
            indices[i] = -1;
        }
        if (tid == 0) {
            counts_ptr[list] = count;
        }
    }
}

void ggml_cuda_fn_l3_compact_mask(ggml_cuda_pool & pool, const ggml_tensor * mask, int32_t * indices, int32_t * counts,
        int32_t n_queries, int32_t ncols1, int32_t n_kv_max, cudaStream_t stream) {
    GGML_ASSERT(ncols1 == 1 || ncols1 == 4 || ncols1 == 8);
    const int     ne30     = (int) mask->ne[0];
    const int64_t s31      = mask->nb[1] / sizeof(half);
    const int64_t s33      = mask->nb[3] / sizeof(half);
    const int     n_groups = (n_queries + ncols1 - 1) / ncols1;
    const int     n_seq    = (int) mask->ne[3];
    const int     n_chunks = (ne30 + FN_L3_COMPACT_CHUNK - 1) / FN_L3_COMPACT_CHUNK;
    GGML_ASSERT(n_chunks >= 1);

    ggml_cuda_pool_alloc<int32_t> chunk_counts(pool, (size_t) n_chunks*n_groups*n_seq);

    const dim3 grid(n_chunks, n_groups, n_seq);
    const dim3 block(FN_L3_COMPACT_THREADS, 1, 1);
    const bool oob = n_queries % ncols1 != 0;

#define FN_L3_COMPACT_LAUNCH(nc, ob)                                                                                 \
    k_fn_l3_compact_count<nc, ob><<<grid, block, 0, stream>>>((const half *) mask->data, chunk_counts.get(), ne30,   \
            n_queries, s31, s33);                                                                                     \
    k_fn_l3_compact_write<nc, ob><<<grid, block, 0, stream>>>((const half *) mask->data, chunk_counts.get(), indices, \
            counts, ne30, n_queries, n_kv_max, s31, s33)

    if (ncols1 == 1) {
        FN_L3_COMPACT_LAUNCH(1, false);
    } else if (ncols1 == 4) {
        if (oob) { FN_L3_COMPACT_LAUNCH(4, true); } else { FN_L3_COMPACT_LAUNCH(4, false); }
    } else {
        if (oob) { FN_L3_COMPACT_LAUNCH(8, true); } else { FN_L3_COMPACT_LAUNCH(8, false); }
    }
#undef FN_L3_COMPACT_LAUNCH
    CUDA_CHECK(cudaGetLastError());
    ggml_cuda_fn_l3_note(GGML_CUDA_FN_L3_PATH_COMPACT, "COMPACT: QSA sparse-index compaction on many blocks");
}

// ---------------------------------------------------------------------------------------------------------------------
// [TAG_FN_L3_GPU_IDXQ8] the lightning indexer reads its keys from the q8_0 rows the gather would have copied
// ---------------------------------------------------------------------------------------------------------------------

static bool fn_l3_is_noop_view(const ggml_tensor * t) {
    return t->op == GGML_OP_RESHAPE || (t->op == GGML_OP_VIEW && t->view_offs == 0);
}

const ggml_tensor * ggml_cuda_fn_l3_idxq8_gather(const ggml_tensor * indexer) {
    if (!ggml_cuda_fn_l3_enabled() || indexer == nullptr || indexer->op != GGML_OP_LIGHTNING_INDEXER ||
            ggml_fn_l3_get(indexer) != GGML_FN_L3_IDXQ8) {
        return nullptr;
    }
    const ggml_tensor * q = indexer->src[0];
    const ggml_tensor * k = indexer->src[1];
    // the instance that exists: 4 heads of 128, one stream
    if (q == nullptr || k == nullptr || q->ne[0] != 128 || q->ne[1] != 4 || q->ne[3] != 1 ||
            k->type != GGML_TYPE_F32 || k->ne[0] != 128 || k->ne[1] != 1 || k->ne[3] != 1 || !ggml_is_contiguous(k)) {
        return nullptr;
    }
    const ggml_tensor * g = k;
    for (int i = 0; i < 4 && g != nullptr && fn_l3_is_noop_view(g); ++i) {
        g = g->src[0];
    }
    if (g == nullptr || g->op != GGML_OP_GET_ROWS || g->type != GGML_TYPE_F32 || !ggml_is_contiguous(g) ||
            ggml_nelements(g) != ggml_nelements(k) || g->data != k->data) {
        return nullptr;
    }
    const ggml_tensor * src = g->src[0];
    const ggml_tensor * idx = g->src[1];
    if (src == nullptr || idx == nullptr || src->type != GGML_TYPE_Q8_0 || idx->type != GGML_TYPE_I32 ||
            src->ne[0] != 128 || src->ne[2] != 1 || src->ne[3] != 1 || src->nb[0] != ggml_type_size(GGML_TYPE_Q8_0) ||
            idx->ne[1] != 1 || idx->ne[2] != 1 || idx->ne[3] != 1 || !ggml_is_contiguous(idx) ||
            idx->ne[0] != k->ne[2] || g->ne[1] != k->ne[2] || src->nb[1] % 2 != 0) {
        return nullptr;
    }
    return g;
}

bool ggml_cuda_fn_l3_idxq8_skip(const ggml_cgraph * cgraph, int i) {
    const ggml_tensor * g = cgraph->nodes[i];
    if (g->op != GGML_OP_GET_ROWS || !ggml_cuda_fn_l3_enabled() || (g->flags & GGML_TENSOR_FLAG_OUTPUT) ||
            g->type != GGML_TYPE_F32 || g->src[0] == nullptr || g->src[0]->type != GGML_TYPE_Q8_0) {
        return false;
    }

    // every reader of g (through no-op views) is a marked indexer that reads through g, and they all come soon
    constexpr int window = 64;
    const ggml_tensor * via[4] = { g, nullptr, nullptr, nullptr };
    int n_via = 1;
    int32_t pending = ggml_node_get_use_count(cgraph, i);
    if (pending <= 0) {
        return false;
    }
    int n_readers = 0;
    for (int j = i + 1; j < cgraph->n_nodes && j <= i + window && pending > 0; ++j) {
        const ggml_tensor * t = cgraph->nodes[j];
        int uses = 0;
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            for (int v = 0; v < n_via; ++v) {
                uses += t->src[s] != nullptr && t->src[s] == via[v] ? 1 : 0;
            }
        }
        if (uses == 0) {
            continue;
        }
        pending -= uses;
        if (fn_l3_is_noop_view(t) && t->view_src == g && !(t->flags & GGML_TENSOR_FLAG_OUTPUT) && n_via < 4) {
            via[n_via++] = t;
            pending += ggml_node_get_use_count(cgraph, j);
            continue;
        }
        if (ggml_cuda_fn_l3_idxq8_gather(t) != g || uses != 1) {
            return false;
        }
        ++n_readers;
    }
    if (pending != 0 || n_readers == 0) {
        return false;
    }
    ggml_cuda_fn_l3_note(GGML_CUDA_FN_L3_PATH_IDXQ8_SKIP, "IDXQ8: the f32 copy of the pooled indexer keys is not made");
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------
// [TAG_FN_L3_GPU_Q8F] producers that also write the q8_1 copy of their output for the mat-vec that reads it next.
// Each kernel is the kernel of the unmarked op with the copy added: rms_norm_f32<block_size, true> (norm.cu) and
// dsv4_hc_pre_f32<true> (dsv4-hc.cu), the same operations in the same order, then fn_l3_q8_1_store on the value it stores.
// Element e of the (contiguous) output is element e % ne10 of row e / ne10 of the src1 that reads it; a warp holds 32
// consecutive elements of one q8_1 block, so ncols (n_embd) and ne10 must be multiples of 32 and the rows unpadded.
// ---------------------------------------------------------------------------------------------------------------------

template <int block_size>
static __global__ void k_fn_l3_rms_norm_mul_q8(
        const float * x, float * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps, const float * mul, const int64_t mul_stride_row,
        const int64_t mul_stride_channel, const int64_t mul_stride_sample, const uint3 mul_ncols_packed,
        const uint3 mul_nrows_packed, const uint3 mul_nchannels_packed, const uint3 mul_nsamples_packed,
        block_q8_1 * q8, const int64_t q8_ne10, const int64_t q8_bpr) {
    ggml_cuda_pdl_lc();
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    const int64_t e0 = int64_t((sample*nchannels + channel)*nrows + row)*ncols;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
    const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
    const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
    mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;

    float tmp = 0.0f; // partial sum for thread in warp

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        const int mul_col = fastmodulo(col, mul_ncols_packed);
        const float v = scale * x[col] * mul[mul_col];
        dst[col] = v;
        const int64_t e = e0 + col;
        fn_l3_q8_1_store(v, q8 + (e / q8_ne10)*q8_bpr + (e % q8_ne10)/QK8_1, tid % WARP_SIZE);
    }
}

bool ggml_cuda_fn_l3_rms_norm_mul_q8(ggml_backend_cuda_context & ctx, ggml_tensor * rms_norm, ggml_tensor * mul_tensor) {
    if (!ggml_cuda_fn_l3_enabled() || ggml_fn_l3_get(mul_tensor) != GGML_FN_L3_Q8OUT) {
        return false;
    }
    const ggml_tensor * x = rms_norm->src[0];
    const ggml_tensor * w = mul_tensor->src[0] == rms_norm ? mul_tensor->src[1] : mul_tensor->src[0];
    if ((mul_tensor->src[0] != rms_norm && mul_tensor->src[1] != rms_norm) || x == nullptr || w == nullptr ||
            x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || rms_norm->type != GGML_TYPE_F32 ||
            mul_tensor->type != GGML_TYPE_F32 || x->nb[0] != sizeof(float) || w->nb[0] != sizeof(float) ||
            !ggml_is_contiguous(mul_tensor) || !ggml_are_same_shape(x, mul_tensor) || x->ne[0] % QK8_1 != 0 ||
            x->ne[0] > INT32_MAX || ggml_nrows(x) > INT32_MAX) {
        return false;
    }
    float eps = 0.0f;
    memcpy(&eps, rms_norm->op_params, sizeof(float));

    ggml_cuda_fn_l3_q8_dst q8;
    if (!ggml_cuda_mmvq_q8_produce(ctx, mul_tensor, q8, /*padding_ok =*/ false)) {
        return false;
    }

    // the launch rms_norm_mul_f32_cuda makes
    const int64_t s01 = x->nb[1]/sizeof(float);
    const int64_t s02 = x->nb[2]/sizeof(float);
    const int64_t s03 = x->nb[3]/sizeof(float);
    const dim3 blocks_num((unsigned) x->ne[1], (unsigned) x->ne[2], (unsigned) x->ne[3]);
    const uint3 mul_ncols_packed     = init_fastdiv_values((uint32_t) w->ne[0]);
    const uint3 mul_nrows_packed     = init_fastdiv_values((uint32_t) w->ne[1]);
    const uint3 mul_nchannels_packed = init_fastdiv_values((uint32_t) w->ne[2]);
    const uint3 mul_nsamples_packed  = init_fastdiv_values((uint32_t) w->ne[3]);
    const int ncols = (int) x->ne[0];
    if (ncols < 1024) {
        const dim3 block_dims(256, 1, 1);
        const ggml_cuda_kernel_launch_params lp(blocks_num, block_dims, 32*sizeof(float), ctx.stream());
        ggml_cuda_kernel_launch(k_fn_l3_rms_norm_mul_q8<256>, lp, (const float *) x->data, (float *) mul_tensor->data,
                ncols, s01, s02, s03, eps, (const float *) w->data, (int64_t) (w->nb[1]/sizeof(float)),
                (int64_t) (w->nb[2]/sizeof(float)), (int64_t) (w->nb[3]/sizeof(float)), mul_ncols_packed, mul_nrows_packed,
                mul_nchannels_packed, mul_nsamples_packed, q8.y, q8.ne10, q8.bpr);
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params lp(blocks_num, block_dims, 32*sizeof(float), ctx.stream());
        ggml_cuda_kernel_launch(k_fn_l3_rms_norm_mul_q8<1024>, lp, (const float *) x->data, (float *) mul_tensor->data,
                ncols, s01, s02, s03, eps, (const float *) w->data, (int64_t) (w->nb[1]/sizeof(float)),
                (int64_t) (w->nb[2]/sizeof(float)), (int64_t) (w->nb[3]/sizeof(float)), mul_ncols_packed, mul_nrows_packed,
                mul_nchannels_packed, mul_nsamples_packed, q8.y, q8.ne10, q8.bpr);
    }
    ggml_cuda_fn_l3_note(GGML_CUDA_FN_L3_PATH_Q8F_NORM, "Q8F: hc mixer norm (rms_norm * w) writes its q8_1 copy");
    return true;
}

static __global__ void k_fn_l3_hc_pre_q8(
        const float * x, const float * weights, float * dst, int64_t n_embd, int64_t hc, int64_t n_tokens,
        int64_t sx0, int64_t sx1, int64_t sx2, int64_t sw0, int64_t sw1, int64_t sw2, int64_t sd0, int64_t sd1,
        float scale, block_q8_1 * q8, const int64_t q8_ne10, const int64_t q8_bpr) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0 = ir % n_embd;
    const int64_t it = ir / n_embd;

    float sum = 0.0f;
    for (int64_t ih = 0; ih < hc; ++ih) {
        const float xv = x[i0*sx0 + ih*sx1 + it*sx2];
        float wv;
        wv = 1.0f / (1.0f + expf(-weights[i0*sw0 + ih*sw1 + it*sw2]));
        sum += xv * wv;
    }

    const float v = scale * sum;
    dst[i0*sd0 + it*sd1] = v;
    const int64_t e = it*n_embd + i0;
    fn_l3_q8_1_store(v, q8 + (e / q8_ne10)*q8_bpr + (e % q8_ne10)/QK8_1, threadIdx.x % WARP_SIZE);
}

bool ggml_cuda_fn_l3_hc_pre_q8(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (!ggml_cuda_fn_l3_enabled() || ggml_fn_l3_get(dst) != GGML_FN_L3_Q8OUT) {
        return false;
    }
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];
    const bool gated = ggml_get_op_params_i32(dst, 1) != 0;
    if (!gated || x == nullptr || weights == nullptr || x->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 ||
            dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) || x->ne[0] % QK8_1 != 0 ||
            dst->ne[0] != x->ne[0] || dst->ne[1] != x->ne[2]) {
        return false;
    }

    ggml_cuda_fn_l3_q8_dst q8;
    if (!ggml_cuda_mmvq_q8_produce(ctx, dst, q8, /*padding_ok =*/ false)) {
        return false;
    }

    // the launch ggml_cuda_op_dsv4_hc_pre makes
    const int64_t n_embd   = x->ne[0];
    const int64_t hc       = x->ne[1];
    const int64_t n_tokens = x->ne[2];
    const float   scale    = ggml_get_op_params_f32(dst, 0);

    const int block_size = 256;
    const int64_t nr = n_embd * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((unsigned) ((nr + block_size - 1) / block_size), 1, 1);
    const ggml_cuda_kernel_launch_params lp(grid_dims, block_dims, 0, ctx.stream());
    ggml_cuda_kernel_launch(k_fn_l3_hc_pre_q8, lp,
            (const float *) x->data, (const float *) weights->data, (float *) dst->data,
            n_embd, hc, n_tokens,
            (int64_t) (x->nb[0]/sizeof(float)), (int64_t) (x->nb[1]/sizeof(float)), (int64_t) (x->nb[2]/sizeof(float)),
            (int64_t) (weights->nb[0]/sizeof(float)), (int64_t) (weights->nb[1]/sizeof(float)), (int64_t) (weights->nb[2]/sizeof(float)),
            (int64_t) (dst->nb[0]/sizeof(float)), (int64_t) (dst->nb[1]/sizeof(float)),
            scale, q8.y, q8.ne10, q8.bpr);
    ggml_cuda_fn_l3_note(GGML_CUDA_FN_L3_PATH_Q8F_HCPRE, "Q8F: hc_pre (gated mean of the streams) writes its q8_1 copy");
    return true;
}
