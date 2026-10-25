#include "llama-memory-hybrid-idx.h"

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"

#include "ggml-turbot.h"   // [TAG_FN_TURBOT_IDX] ggml_turbot_is_type
#include "llama-fn-auto.h"  // [TAG_FN_L3_HOST_DIAG] llama_fn_l3_flag
#include "llama-fn-l22.h"   // [TAG_FN_L22_QSA_SEQPOOLS] PASSLOG

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <stdexcept>

//
// llama_memory_hybrid_idx
//

// [TAG_FN_TURBOT_IDX] K type of the indexer key cache. The indexer reads its keys with get_rows (raw, before norm and
// RoPE), so a turbo or turbot type cannot hold them, and top-k selection must not get a low-bit key. Rule:
//   - env LLAMA_QSA_IDX_TYPE = f32, f16, bf16 or q8_0: that type (the A/B switch for the top-k agreement gate)
//   - the attention type when it is f32, f16, bf16 or q8_0 (unchanged from before)
//   - else q8_0: turbot, the turbo types and every other quantized type (never below q8_0)
static ggml_type llama_qsa_idx_type(ggml_type type_attn) {
    static const ggml_type ok[] = { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0 };

    const char * env = getenv("LLAMA_QSA_IDX_TYPE");
    if (env != nullptr && env[0] != '\0') {
        for (const ggml_type t : ok) {
            if (strcmp(env, ggml_type_name(t)) == 0) {
                return t;
            }
        }
        LLAMA_LOG_WARN("%s: LLAMA_QSA_IDX_TYPE=%s is not f32, f16, bf16 or q8_0, ignored\n", __func__, env);
    }

    for (const ggml_type t : ok) {
        if (type_attn == t) {
            return t;
        }
    }

    return GGML_TYPE_Q8_0;
}

llama_memory_hybrid_idx::llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
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
    const layer_filter_cb & filter_idx) :
    llama_memory_hybrid(
        model,
        type_k, type_v, v_trans, kv_size, n_pad, n_swa, swa_type,
        type_r, type_s, rs_size,
        n_seq_max, n_rs_seq, offload, unified,
        filter_attn, filter_recr),
    hparams_idx(model.hparams),
    mem_idx(filter_idx == nullptr ? nullptr : [&] {
        // MQA with a single key head of indexer_head_size, as llama_kv_cache_dsa shapes its own
        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
        // a k-pool indexer caches its per-token rows and the pooled key side by side
        // (glm5-next: key | gate | pooled, qwen4exp: key | pooled)
        hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size * (model.hparams.indexer_kpool > 0 ? model.hparams.indexer_kpool_row : 1);

        // the cached indexer keys are raw, rotation happens after pooling at read time, so a
        // K-shift must not rotate them while the stream copies in the same update still apply
        hparams_idx.rope_type = LLAMA_ROPE_TYPE_NONE;

        // fool llama_kv_cache into thinking this is a MLA cache, so it won't cache V tensors
        hparams_idx.n_embd_head_k_mla_impl = model.hparams.indexer_head_size;
        hparams_idx.n_embd_head_v_mla_impl = model.hparams.indexer_head_size;

        // [TAG_FN_TURBOT_IDX] its own K type; V is never cached (MLA fake above), so it takes the same type
        const ggml_type type_idx = llama_qsa_idx_type(type_k);

        if (type_idx == type_k) {
            LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, kv_size);
        } else {
            LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells, K type %s (attention cache %s; the indexer keys stay at q8_0 or better)\n",
                    __func__, kv_size, ggml_type_name(type_idx), ggml_turbot_is_type(type_k) ? "turbot" : ggml_type_name(type_k));
        }

        return new llama_kv_cache(
            model, hparams_idx, type_idx, type_idx, v_trans, offload, unified,
            kv_size, n_seq_max, n_pad, n_swa, swa_type,
            nullptr, filter_idx, nullptr, nullptr, "idx_");
    }()) {
    kpool_probe = llama_fn_l3_flag(model, "LLAMA_FN_HOST_KPOOL_PROBE"); // [TAG_FN_L3_HOST_DIAG]
    kpool_inc   = llama_fn_l3_int(model, "LLAMA_FN_L10_KPOOLINC", 1) != 0; // [TAG_FN_L10_KPOOLINC] qwen4exp only

    // [TAG_FN_L22_QSA_SEQPOOLS] [TAG_FN_L22_QSA_SHAREPFX] qwen4exp only (llama_fn_l3_int), off when unset
    seqpools = std::min(std::max(llama_fn_l3_int(model, "LLAMA_FN_L22_QSA_SEQPOOLS", 0), 0), 2);
    sharepfx = llama_fn_l3_int(model, "LLAMA_FN_L22_QSA_SHAREPFX", 0) != 0;
    if (mem_idx && model.hparams.indexer_kpool > 0 && (seqpools != 0 || sharepfx)) {
        n_top_pool = model.hparams.indexer_top_k / model.hparams.indexer_kpool;
        // the picked set of a shorter list is the same only with the unordered radix select (qwen4exp reads the same
        // switch; TURBO_TOPK_ORDERED forces CUDA's sort) and with every layer's indexer on a CUDA device
        const char * tu = getenv("TURBO_QSA_TOPK_UNORDERED");
        const bool unordered = tu != nullptr && tu[0] == '1' && getenv("TURBO_TOPK_ORDERED") == nullptr;
        const uint32_t n_layer = model.hparams.n_layer();
        bool cuda = n_layer > 0;
        for (uint32_t il = 0; il < n_layer && cuda; ++il) {
            ggml_backend_dev_t dev = model.dev_layer((int) il);
            ggml_backend_reg_t reg = dev != nullptr ? ggml_backend_dev_backend_reg(dev) : nullptr;
            cuda = reg != nullptr && strcmp(ggml_backend_reg_name(reg), "CUDA") == 0;
        }
        seqpools_ok = seqpools != 0 && unordered && cuda && n_top_pool > 0;
        llama_fn_l22_log("switch", "[TAG_FN_L22_QSA_SEQPOOLS] seqpools=%d%s sharepfx=%d n_top=%u topk_unordered=%d "
                "indexer_on_cuda=%d%s", seqpools, seqpools == 2 ? " (check)" : "", (int) sharepfx, n_top_pool,
                (int) unordered, (int) cuda, seqpools != 0 && !seqpools_ok ? " -> no per-sequence lists" : "");
    }

    // [TAG_FN_L16_SEQRM] qwen4exp (the trunk and the MTP draft context): seq_rm of one sequence walks only its own cells
    if (llama_fn_l3_flag(model, "LLAMA_FN_L16_SEQRM")) {
        const bool verify = llama_fn_l3_flag(model, "LLAMA_SEQ_RM_VERIFY");
        get_mem_attn()->set_seq_rm_indexed(true, verify);
        if (mem_idx) {
            mem_idx->set_seq_rm_indexed(true, verify);
        }
        LLAMA_LOG_INFO("%s: [TAG_FN_L16_SEQRM] seq_rm from the sequence index (attention%s)%s\n", __func__,
                mem_idx ? " and indexer caches" : " cache", verify ? ", checked against the scan (LLAMA_SEQ_RM_VERIFY)" : "");
    }
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    // note: repeats llama_memory_hybrid::init_batch, as the indexer needs the attention slot infos that the base context hides
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (get_mem_attn()->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = get_mem_recr()->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!get_mem_recr()->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = get_mem_attn()->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // the indexer uses the attention cache's slot layout; a separate one can drift from it
        llama_kv_cache::slot_info_vec_t heads_idx;
        if (mem_idx) {
            heads_idx = heads_attn;
        }

        return std::make_unique<llama_memory_hybrid_idx_context>(
                this, std::move(heads_attn), std::move(heads_idx), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_full() {
    return std::make_unique<llama_memory_hybrid_idx_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_idx_context>(this, lctx, optimize);
}

void llama_memory_hybrid_idx::clear(bool data) {
    llama_memory_hybrid::clear(data);

    if (mem_idx) {
        mem_idx->clear(data);
        mem_idx_stale_set(-1, 0);
    }
}

// A pooled key is only valid while the grouping that produced it holds. Grouping is sequence relative,
// so an edit at p0 leaves every pool that ends before p0 alone.
void llama_memory_hybrid_idx::mem_idx_stale_set(llama_seq_id seq_id, llama_pos p0) {
    p0 = std::max<llama_pos>(p0, 0);

    if (seq_id < 0) {
        for (auto & p : mem_idx_stale) {
            p = std::min(p, p0);
        }

        return;
    }

    GGML_ASSERT(seq_id < (llama_seq_id) LLAMA_MAX_SEQ);

    mem_idx_stale[seq_id] = std::min(mem_idx_stale[seq_id], p0);
}

// An edit at or below the first position moves pos_min, which regroups the whole sequence.
llama_pos llama_memory_hybrid_idx::mem_idx_stale_pos(llama_seq_id seq_id, llama_pos p0) const {
    if (seq_id < 0 || p0 <= mem_idx->seq_pos_min(seq_id)) {
        return 0;
    }

    return p0;
}

bool llama_memory_hybrid_idx::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // same order as llama_memory_hybrid::seq_rm: the recurrent cache can refuse, so try it first
    if (!get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        return false;
    }

    if (mem_idx) {
        const llama_pos stale = mem_idx_stale_pos(seq_id, p0);
        // [TAG_FN_L22_QSA_SHAREPFX] a removal that takes no shared cell changes no other sequence's cells: the sharing
        // stays as it is (the speculative rollback past a shared prefix, every step)
        const bool rm_shared = !sharepfx || !kpool_layout_shared() || kpool_rm_takes_shared(seq_id, p0, p1);
        mem_idx->seq_rm(seq_id, p0, p1);
        mem_idx_stale_set(seq_id, stale);

        // removing a sequence can free cells another sequence shared, but only this one is marked stale, so the
        // survivor would keep shared = true and pin cache_safe off forever; stale every sequence to re-derive it
        if (kpool_layout_shared() && rm_shared) {
            mem_idx_stale_set(-1, 0);
        }
    }

    return get_mem_attn()->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid_idx::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);

    if (mem_idx) {
        mem_idx->seq_cp(seq_id_src, seq_id_dst, p0, p1);
        // the copy shares cells, which cannot hold two groupings, so both sides drop their cached keys
        mem_idx_stale_set(seq_id_src, 0);
        mem_idx_stale_set(seq_id_dst, 0);
        // [TAG_FN_L22_QSA_SHAREPFX] the copy also adds a holder to cells a third sequence may share with the source:
        // re-derive every sequence's shared run
        if (sharepfx && kpool_layout_shared()) {
            mem_idx_stale_set(-1, 0);
        }
    }
}

void llama_memory_hybrid_idx::seq_keep(llama_seq_id seq_id) {
    llama_memory_hybrid::seq_keep(seq_id);

    if (mem_idx) {
        mem_idx->seq_keep(seq_id);
        // cells shared with the dropped sequences become exclusive again, their keys were never cached
        mem_idx_stale_set(-1, 0);
    }
}

void llama_memory_hybrid_idx::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);

    if (mem_idx) {
        // a negative shift moves the cells below p0, so they regroup as well
        const llama_pos stale = mem_idx_stale_pos(seq_id, shift < 0 ? p0 + shift : p0);
        mem_idx->seq_add(seq_id, p0, p1, shift);
        mem_idx_stale_set(seq_id, stale);
        // [TAG_FN_L22_QSA_SHAREPFX] a shared cell's position is the cell's: a shift moves it for every holder
        if (sharepfx && kpool_layout_shared()) {
            mem_idx_stale_set(-1, 0);
        }
    }
}

void llama_memory_hybrid_idx::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    llama_memory_hybrid::seq_div(seq_id, p0, p1, d);

    if (mem_idx) {
        mem_idx->seq_div(seq_id, p0, p1, d);
        mem_idx_stale_set(seq_id, 0);
        // [TAG_FN_L22_QSA_SHAREPFX] as seq_add
        if (sharepfx && kpool_layout_shared()) {
            mem_idx_stale_set(-1, 0);
        }
    }
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_idx::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = llama_memory_hybrid::memory_breakdown();

    if (mem_idx) {
        for (const auto & buft_size : mem_idx->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    return mb;
}

void llama_memory_hybrid_idx::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    llama_memory_hybrid::state_write(io, seq_id, flags);

    // [TAG_HYBRID_IDX_STATE] the indexer section goes last, so it is a pure suffix: an old reader stops early instead of misparsing it
    // The indexer mirrors the attention cache, so it uses the same PARTIAL_ONLY gate.
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        if (mem_idx) {
            mem_idx->state_write(io, seq_id, flags);
        }
    }

}

void llama_memory_hybrid_idx::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // note: repeats llama_memory_hybrid::state_read
    // the indexer needs the attention cache's cells, and a half-failed restore must leave all three caches alike

    // [TAG_HYBRID_IDX_SINFO]
    // the indexer restore adopts the attention cache's layout instead of searching for cells of its own
    // two find_slot calls agree only while both caches see the same occupancy, which a restore cannot promise
    llama_kv_cache::slot_info_vec_t sinfos_attn;

    try {
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            get_mem_attn()->state_read_sinfo(io, seq_id, flags, mem_idx ? &sinfos_attn : nullptr, nullptr);
        }

        get_mem_recr()->state_read(io, seq_id, flags);

        // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            if (mem_idx) {
                mem_idx->state_read_sinfo(io, seq_id, flags, nullptr, &sinfos_attn);
                // the restore rewrites the cells behind the pool layout's back
                mem_idx_stale_set(seq_id, 0);
                // it can also change which cells are shared; re-derive sharing for every sequence, as seq_rm does
                if (kpool_layout_shared()) {
                    mem_idx_stale_set(-1, 0);
                }
            }
        }

    } catch (...) {
        // a half-restored context is the one state the indexer cannot fix by itself: attention holds new cells, the indexer old ones
        // drop what was being restored from all of them, which is a state they do agree on.
        state_drop(seq_id);

        throw;
    }
}

void llama_memory_hybrid_idx::state_drop(llama_seq_id seq_id) {
    // dropped directly, not via seq_rm: the recurrent cache may refuse it and then only the other two get cleared
    if (seq_id < 0) {
        clear(true);

        return;
    }

    get_mem_attn()->state_clear(seq_id);
    get_mem_recr()->seq_rm(seq_id, -1, -1);

    if (mem_idx) {
        mem_idx->state_clear(seq_id);
        mem_idx_stale_set(seq_id, 0);
        // clearing this sequence can end a sharing the survivor would otherwise keep flagged (see seq_rm)
        if (kpool_layout_shared()) {
            mem_idx_stale_set(-1, 0);
        }
    }
}

llama_kv_cache * llama_memory_hybrid_idx::get_mem_idx() const {
    return mem_idx.get();
}

//
// llama_memory_hybrid_idx_context
//

// streams in each ubatch's slot info, matching get_k/get_v's `ns`
static std::vector<uint32_t> llama_memory_hybrid_idx_ns(const llama_kv_cache::slot_info_vec_t & sinfos) {
    std::vector<uint32_t> res;
    res.reserve(sinfos.size());

    for (const auto & sinfo : sinfos) {
        res.push_back(sinfo.s1 - sinfo.s0 + 1);
    }

    return res;
}

// Which cells of a sequence make up which pool, for the whole cache.
struct llama_memory_hybrid_idx::kpool_layout_seq {
    llama_pos pos_min = 0;
    uint32_t  strm    = 0; // Stream holding this sequence's cells
    std::vector<std::pair<llama_pos, uint32_t>> cells; // Position and stream local cell pairs, sorted by position.
    std::vector<uint32_t> pools;

    // Where the pool scan stopped, so an append resumes instead of starting over.
    size_t j_next = 0;

    // Whether any cell also carries another sequence, which rules out caching this sequence's pooled keys.
    bool shared = false;

    // [TAG_FN_L10_KPOOLINC] how many leading pools are unchanged (same start, same cells) since set_input_kpool last
    // brought the input cache below up to date; only ever lowered here, raised by the consumer
    mutable size_t pools_keep = 0;

    // [TAG_FN_L22_QSA_SHAREPFX] the shared-cell shape (leading run of shared cells), kept with LLAMA_FN_L22_QSA_SHAREPFX
    llama_fn_qsa_pfx pfx;
};

struct llama_memory_hybrid_idx::kpool_layout {
    using seq = kpool_layout_seq; // [TAG_FN_R1_KPOOL_TAIL] named, so the pool scan helper can take one

    std::array<seq, LLAMA_MAX_SEQ> seqs;

    uint32_t n_pool_real = 0;
    bool cache_safe      = true;
    bool has_shared      = false; // [TAG_FN_L22_QSA_SHAREPFX] some sequence holds a shared cell
    bool pfx_safe        = false; // [TAG_FN_L22_QSA_SHAREPFX] cache_safe despite shared cells (aligned prefix)

    // [TAG_FN_L10_KPOOLINC] the input values of each sequence's pools (one stream: the cell is the K row), valid for the
    // first pools_keep pools, and how many of those pools use each cell as their rep (the pad search's "is a rep")
    struct inc_seq {
        std::vector<int32_t>   pcell; // rep cell per pool
        std::vector<int32_t>   pidx;  // kpool member cells per pool
        std::vector<llama_pos> end;   // position of the last member per pool
    };
    mutable std::array<inc_seq, LLAMA_MAX_SEQ> inc;
    mutable std::vector<uint16_t> inc_rep_cnt;
};

// Which pools of the layout the current ubatch must re-pool, in the layout's pool order.
struct llama_memory_hybrid_idx_context::kpool_state {
    std::vector<uint32_t> is_new;
    std::vector<uint32_t> new_list; // [TAG_FN_L10_KPOOLINC] the marked pools (cache-safe layouts), sorted before use
    uint32_t generation = 0;

    uint32_t n_pool_real = 0;
    uint32_t n_new       = 0;
    uint32_t n_new_g     = 1; // graph size of the new pool list, stable across decode steps
    bool     cache_safe  = true;

    // [TAG_FN_L22_QSA_SEQPOOLS] the graph's pool count (get_n_kpool): kpool_pad(n_pool_real), or the width of the one
    // sequence a single-sequence ubatch lists (seq_only >= 0, its n_pool_list pools first)
    uint32_t     n_pool_g    = 0;
    llama_seq_id seq_only    = -1;
    uint32_t     n_pool_list = 0;
};

namespace {

// The last padded pool is always unused.
// [TAG_FN_L8_GRAPHPAD] LLAMA_KPOOL_PAD=<pools> (a power of two >= 64, default 64): the pool count in coarser steps. Every
// step changes the shapes of the decode graphs, so the shape-keyed CUDA graph of each verify width (1-3 with MTP) is made
// again (one uncaptured run of ~8500 nodes + a capture): at 64 pools x kpool 4 that was every 256 tokens, ~5 % of the
// target graphs (TURBO_GRAPH_DIFF, l8/diag). The padded pools point at a dummy cell and are masked. The qwen4exp profile
// sets 512 (2048 tokens) together with LLAMA_KV_PAD_INDEXER=2048.
uint32_t kpool_pad_g() {
    static const uint32_t g = [] {
        const char * e = getenv("LLAMA_KPOOL_PAD");
        const int    v = e ? atoi(e) : 64;
        uint32_t     p = 64;
        while (p < (uint32_t) std::min(v, 65536)) {
            p *= 2;
        }
        return p;
    }();
    return g;
}

uint32_t kpool_pad(uint32_t n_pool) {
    return llama_fn_qsa_kpool_pad(n_pool, kpool_pad_g());
}

// [TAG_FN_L3_HOST_DIAG] LLAMA_FN_HOST_KPOOL_PROBE=1 (qwen4exp): host time of the k-pool work per ubatch (the layout
// update and re-pool marks in apply(), the input arrays in set_input_kpool), a line every 64 fills. Diagnostic only.
struct kpool_probe {
    double   apply_us = 0.0;
    double   input_us = 0.0;
    uint64_t n_apply  = 0;
    uint64_t n_input  = 0;
    uint64_t pools    = 0;
};

kpool_probe & kpool_probe_get() {
    static kpool_probe p; // the inference thread only (apply and set_inputs run on it)
    return p;
}

void kpool_probe_add(bool input, double us, uint32_t n_pool) {
    kpool_probe & p = kpool_probe_get();
    if (input) {
        p.input_us += us;
        p.pools    += n_pool;
        if (++p.n_input % 64 == 0) {
            LLAMA_LOG_INFO("kpool-probe: [TAG_FN_L3_HOST_DIAG] %llu ubatches: layout update + re-pool marks %.1f us, input "
                    "arrays %.1f us per ubatch (%.0f pools)\n", (unsigned long long) p.n_input,
                    p.n_apply ? p.apply_us / p.n_apply : 0.0, p.input_us / 64.0, (double) p.pools / 64.0);
            p.apply_us = p.input_us = 0.0;
            p.n_apply  = 0;
            p.pools    = 0;
        }
    } else {
        p.apply_us += us;
        p.n_apply++;
    }
}

// Rank of (pos, cell) in a sequence's cells sorted by position then cell, or -1 when absent.
// In order mode the rank alone places a token: cells sharing a position (M-RoPE images) have distinct ranks.
int64_t kpool_rank(const std::vector<std::pair<llama_pos, uint32_t>> & cells, llama_pos pos, uint32_t cell) {
    auto it = std::lower_bound(cells.begin(), cells.end(), std::make_pair(pos, cell));
    return it != cells.end() && it->second == cell && it->first == pos ? it - cells.begin() : -1;
}

}

llama_memory_hybrid_idx::~llama_memory_hybrid_idx() = default;

// [TAG_FN_L23_TWOSLOT]
static thread_local uint32_t g_kpool_reserve_seqs = 0;

void llama_fn_kpool_reserve_seqs(uint32_t n_seqs) {
    g_kpool_reserve_seqs = n_seqs;
}

uint64_t llama_memory_hybrid_idx::kpool_reserve_pools(uint32_t n_seqs) const {
    if (mem_idx == nullptr || get_kpool() == 0) {
        return 0;
    }
    const uint32_t n_max = mem_idx->get_n_seq_max();
    return uint64_t(mem_idx->get_size() / get_kpool()) * (n_seqs > 0 ? std::min(n_seqs, n_max) : n_max);
}

uint64_t llama_memory_hybrid_idx::kpool_bound(uint32_t n_add) const {
    if (mem_idx == nullptr || get_kpool() == 0) {
        return 0;
    }
    // a sequence's pools are whole groups of kpool of its cells (kpool_scan_pools): sum over s of floor(cells_s / kpool)
    // <= floor((sum of cells_s) / kpool), with the n_add (token, sequence) pairs of the batch counted in
    const uint32_t n_stream = mem_idx->get_n_stream();
    uint64_t n = n_add;
    for (llama_seq_id s = 0; s < (llama_seq_id) LLAMA_MAX_SEQ; ++s) {
        if (n_stream > 1 && s >= (llama_seq_id) n_stream) {
            break;
        }
        n += mem_idx->get_cells(n_stream == 1 ? 0 : s).seq_pos_get(s).size();
    }
    return n / get_kpool();
}

const llama_memory_hybrid_idx::kpool_layout & llama_memory_hybrid_idx::kpool_layout_get() const {
    GGML_ASSERT(kpool_lay != nullptr);

    return *kpool_lay;
}

bool llama_memory_hybrid_idx::kpool_layout_shared() const {
    // [TAG_FN_L22_QSA_SHAREPFX] with an aligned shared prefix the cache stays safe while cells are shared
    return kpool_lay && (sharepfx ? kpool_lay->has_shared : !kpool_lay->cache_safe);
}

// [TAG_FN_L22_QSA_SHAREPFX]
bool llama_memory_hybrid_idx::kpool_rm_takes_shared(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const {
    if (mem_idx == nullptr || seq_id < 0 || seq_id >= (llama_seq_id) LLAMA_MAX_SEQ) {
        return true;
    }
    if (p0 < 0) {
        p0 = 0;
    }
    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }
    const auto & cells = mem_idx->get_cells(seq_id);
    const auto & sp    = cells.seq_pos_get(seq_id);
    for (auto it = sp.lower_bound(std::make_pair(p0, (uint32_t) 0)); it != sp.end() && it->first < p1; ++it) {
        if (cells.seq_count(it->second) > 1) {
            return true;
        }
    }
    return false;
}

// [TAG_FN_R1_KPOOL_TAIL] the pool scan from sq.j_next over sq.cells (pools start at the first valid token); the update
// and its verification share it
static void kpool_scan_pools(llama_memory_hybrid_idx::kpool_layout_seq & sq, uint32_t kpool, bool by_order) {
    size_t j = sq.j_next;
    if (by_order) {
        // consecutive cells in sequence order, whatever their positions
        for (; j + kpool <= sq.cells.size(); j += kpool) {
            sq.pools.push_back((uint32_t) j);
        }
    } else {
        while (j + kpool <= sq.cells.size()) {
            const llama_pos p0 = sq.cells[j].first;
            if ((p0 - sq.pos_min) % (llama_pos) kpool != 0) {
                ++j;
                continue;
            }
            bool ok = true;
            for (uint32_t k = 1; k < kpool; ++k) {
                if (sq.cells[j + k].first != p0 + (llama_pos) k) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                sq.pools.push_back((uint32_t) j);
                j += kpool;
            } else {
                ++j;
            }
        }
    }
    sq.j_next = j;
}

// [TAG_FN_R1_KPOOL_TAIL] the tail path is on by default (Flash-Next at 262K, one slot: +5% at 32K, +20% at 131K, +30%
// at 246K real-use decode, the same layout as a rebuild under LLAMA_KPOOL_VERIFY=1); LLAMA_KPOOL_TAIL=0 rebuilds the
// sequence's layout on every edit, as before. LLAMA_KPOOL_VERIFY=1 rebuilds every updated sequence from scratch as well
// and aborts on any difference
static int kpool_tail_mode() {
    static const int m = [] {
        const char * e = getenv("LLAMA_KPOOL_TAIL");
        const char * v = getenv("LLAMA_KPOOL_VERIFY");
        const int tail   = e != nullptr && atoi(e) == 0 ? 0 : 1;
        const int verify = v != nullptr && atoi(v) != 0 ? 2 : 0;
        return tail | verify;
    }();
    return m;
}

// Pools are fixed by the positions relative to the sequence's first one, so the layout survives a plain
// append. A sequence edit can regroup them, and mem_idx_stale tells us it happened.
// [TAG_FN_R1_KPOOL_TAIL] An edit from p_st on, p_st after the first position (the speculative rollback of rejected
// drafts on nearly every decode step), regroups only the pools that reach p_st: the cells before it and the pools that
// end before it are kept, the cells from p_st on are appended again from the position set, and the scan resumes after
// the last kept pool. That is the layout a rebuild derives (the scan is a pure function of the cells, and the cells
// before p_st do not change), without walking every cell of the position set again (a 246K-node std::set per step).
const llama_memory_hybrid_idx::kpool_layout & llama_memory_hybrid_idx::kpool_layout_update() {
    GGML_ASSERT(mem_idx != nullptr);

    if (!kpool_lay) {
        kpool_lay = std::make_unique<kpool_layout>();
    }

    auto & lay = *kpool_lay;

    const uint32_t kpool       = get_kpool();
    const uint32_t n_stream_kv = mem_idx->get_n_stream();
    const bool     unified     = n_stream_kv == 1;

    lay.n_pool_real = 0;
    lay.cache_safe  = true;
    lay.has_shared  = false;
    lay.pfx_safe    = false;

    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        auto & sq = lay.seqs[s];

        // a non unified cache gives each sequence its own stream, with stream local cell indices
        if (!unified && s >= (llama_seq_id) n_stream_kv) {
            sq = kpool_layout::seq();
            continue;
        }

        const auto & cells = mem_idx->get_cells(unified ? 0 : s);
        const auto & sp    = cells.seq_pos_get(s);

        sq.strm = unified ? 0 : mem_idx->get_stream(s);

        size_t n_kept = 0;
        bool   tail   = false;
        const llama_pos p_st = mem_idx_stale[s];
        if (mem_idx_stale[s] == POS_CLEAN && !sq.cells.empty() && !sp.empty() &&
                sq.pos_min == sp.begin()->first) {
            n_kept = sq.cells.size();
            for (auto it = sp.upper_bound(sq.cells.back()); it != sp.end(); ++it) {
                sq.cells.push_back(*it);
            }
        } else if ((kpool_tail_mode() & 1) && p_st != POS_CLEAN && p_st > sq.pos_min &&
                (!sq.shared || (sharepfx && unified && !sq.pfx.irr && sq.pfx.pfx_n > 0 && sq.pfx.pfx_n <= sq.cells.size() &&
                                p_st > sq.cells[sq.pfx.pfx_n - 1].first)) &&   // [TAG_FN_L22_QSA_SHAREPFX] a cut past the shared run
                !sq.cells.empty() && !sp.empty() && sq.pos_min == sp.begin()->first) {
            // [TAG_FN_R1_KPOOL_TAIL]
            const size_t n_cut = (size_t) (std::lower_bound(sq.cells.begin(), sq.cells.end(),
                    std::make_pair(p_st, (uint32_t) 0)) - sq.cells.begin());
            sq.cells.resize(n_cut);
            while (!sq.pools.empty() && (size_t) sq.pools.back() + kpool > n_cut) {
                sq.pools.pop_back();
            }
            sq.pools_keep = std::min(sq.pools_keep, sq.pools.size()); // [TAG_FN_L10_KPOOLINC]
            sq.j_next = sq.pools.empty() ? 0 : (size_t) sq.pools.back() + kpool;
            for (auto it = sp.lower_bound(std::make_pair(p_st, (uint32_t) 0)); it != sp.end(); ++it) {
                sq.cells.push_back(*it);
            }
            n_kept = n_cut;
            tail   = sq.cells.size() == sp.size(); // else a cell before p_st went away: rebuild below
        }

        // the appended tail accounts for every cell only if nothing before it was dropped, but an edit can
        // regroup a sequence without changing its cell count, so a stale sequence must rebuild regardless
        // (unless the tail path above handled it)
        if (sq.cells.size() != sp.size() || (mem_idx_stale[s] != POS_CLEAN && !tail)) {
            sq.cells.assign(sp.begin(), sp.end());
            sq.pools.clear();
            sq.pools_keep = 0; // [TAG_FN_L10_KPOOLINC]
            sq.j_next  = 0;
            sq.shared  = false;
            sq.pos_min = sp.empty() ? 0 : sp.begin()->first;
            sq.pfx     = llama_fn_qsa_pfx(); // [TAG_FN_L22_QSA_SHAREPFX]
            n_kept     = 0;
        }

        // sharing starts with a seq_cp; it ends with an edit, or a seq_rm/state_drop/state_read that frees the
        // shared cells - each stales every sequence so the rebuild above re-derives it, so once set it holds
        // until then and the rescan can be skipped
        if (unified && sharepfx) {
            // [TAG_FN_L22_QSA_SHAREPFX] the appended cells extend the leading shared run or end the regular shape; a
            // removal that takes a shared cell, a copy, a shift or a restore while cells are shared stales every
            // sequence (seq_rm / seq_cp / seq_add / seq_div / state_read), so the kept cells' counts still hold
            for (size_t j = n_kept; j < sq.cells.size() && !sq.pfx.irr; ++j) {
                llama_fn_qsa_pfx_step(sq.pfx, j, cells.seq_count(sq.cells[j].second));
            }
            sq.shared = sq.pfx.shared;
        } else if (unified && !sq.shared) {
            for (size_t j = n_kept; j < sq.cells.size(); ++j) {
                if (cells.seq_count(sq.cells[j].second) > 1) {
                    sq.shared = true;
                    break;
                }
            }
        }

        // Pools start at the first valid token
        kpool_scan_pools(sq, kpool, hparams_idx.indexer_kpool_by_order);

        // [TAG_FN_R1_KPOOL_TAIL] LLAMA_KPOOL_VERIFY=1: the same sequence rebuilt from scratch must match
        if ((kpool_tail_mode() & 2) && !sp.empty()) {
            kpool_layout_seq ref;
            ref.strm    = sq.strm;
            ref.cells.assign(sp.begin(), sp.end());
            ref.pos_min = sp.begin()->first;
            kpool_scan_pools(ref, kpool, hparams_idx.indexer_kpool_by_order);
            if (ref.cells != sq.cells || ref.pools != sq.pools || ref.j_next != sq.j_next || ref.pos_min != sq.pos_min) {
                GGML_ABORT("kpool layout: seq %d differs from a rebuild (tail %d, stale %d, cells %zu/%zu, pools %zu/%zu)",
                        (int) s, (int) tail, (int) p_st, sq.cells.size(), ref.cells.size(), sq.pools.size(), ref.pools.size());
            }
        }

        lay.n_pool_real += (uint32_t) sq.pools.size();
        lay.cache_safe   = lay.cache_safe && !sq.shared;
        lay.has_shared   = lay.has_shared || sq.shared;
    }

    // [TAG_FN_L22_QSA_SHAREPFX] shared cells that form one aligned prefix of every sharing sequence keep the pooled-key
    // cache exact: every pool lies inside the prefix (same members and rep in each sequence) or ends in the sequence's
    // own cells (its own rep), see llama_fn_qsa_sharepfx_safe
    if (sharepfx && unified && lay.has_shared) {
        llama_fn_qsa_pfx shapes[LLAMA_MAX_SEQ];
        int32_t          pmin[LLAMA_MAX_SEQ];
        int n = 0;
        for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
            const auto & sq = lay.seqs[s];
            if (sq.cells.empty()) {
                continue;
            }
            shapes[n] = sq.pfx;
            pmin[n]   = sq.cells.front().first;
            ++n;
        }
        if (llama_fn_qsa_sharepfx_safe(shapes, pmin, n)) {
            lay.cache_safe = true;
            lay.pfx_safe   = true;
        }
    }

    return lay;
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_status status) :
    llama_memory_hybrid_context(status) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem) :
    llama_memory_hybrid_context(mem),
    mem(mem),
    // graph reservation walks a full context, and qwen4exp builds the sparse attention only when this is set
    // without it the reserved worst case is the dense graph, so ggml-alloc must grow the buffer on the first decode
    ns_ubatch(mem->get_mem_idx() == nullptr ?
        std::vector<uint32_t>() : std::vector<uint32_t>{ mem->get_mem_idx()->get_n_stream() }),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx())) {
    if (kpool_track()) {
        mem->kpool_layout_update();
        auto st = kpool_build_sizes();
        const uint64_t n_pool_max = mem->kpool_reserve_pools(g_kpool_reserve_seqs); // [TAG_FN_L23_TWOSLOT]
        GGML_ASSERT(n_pool_max <= UINT32_MAX - 64);
        st.n_pool_real = std::max(st.n_pool_real, uint32_t(n_pool_max));
        st.n_new   = st.n_pool_real;
        st.n_new_g = std::max(st.n_new, 1u);
        st.n_pool_g = kpool_pad(st.n_pool_real); // [TAG_FN_L22_QSA_SEQPOOLS] the reserve keeps the worst case
        kpool_st = std::make_unique<kpool_state>(std::move(st));
        i_kpool  = 0;
    }
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                  llama_context * lctx,
                           bool   optimize) :
    llama_memory_hybrid_context(mem, lctx, optimize),
    mem(mem),
    // update() applies a pending cross-stream seq_cp, else the copy keeps stale indexer keys
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        mem->get_mem_idx()->init_update(lctx, optimize)) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                slot_info_vec_t   sinfos_attn,
                slot_info_vec_t   sinfos_idx,
      std::vector<llama_ubatch>   ubatches) :
    // note: the base copies the ubatches; ctx_idx gets a copy of its own
    llama_memory_hybrid_context(mem, std::move(sinfos_attn), ubatches),
    mem(mem),
    ns_ubatch(llama_memory_hybrid_idx_ns(sinfos_idx)),
    sinfos_kpool(mem->get_mem_idx() != nullptr && mem->get_kpool() > 0 && mem->get_kpool_by_order() ? sinfos_idx : slot_info_vec_t()),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx(), std::move(sinfos_idx), ubatches)) {
    // Sequence edits force the touched positions to re-pool.
    mem_idx_stale_batch = mem->mem_idx_stale_get();
}

llama_memory_hybrid_idx_context::~llama_memory_hybrid_idx_context() = default;

bool llama_memory_hybrid_idx_context::next() {
    // Clear only after a successful ubatch.
    if (i_cur == 0 && mem != nullptr) {
        mem->mem_idx_stale_clear();
    }

    if (ctx_idx) {
        ctx_idx->next();
    }

    ++i_cur;

    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_idx_context::apply() {
    bool res = llama_memory_hybrid_context::apply();

    if (ctx_idx) {
        res = res & ctx_idx->apply();
    }

    // Extend the pool layout with this ubatch's cells, then pick what it must re-pool.
    if (res && kpool_track()) {
        const int64_t t_kp = mem->kpool_probe_on() ? ggml_time_us() : 0; // [TAG_FN_L3_HOST_DIAG]
        mem->kpool_layout_update();
        if (!kpool_st) {
            kpool_st = std::make_unique<kpool_state>();
        }
        kpool_build_state(get_ubatch());
        i_kpool  = i_cur;
        if (mem->kpool_probe_on()) {
            kpool_probe_add(false, (double) (ggml_time_us() - t_kp), 0);
        }
    }

    return res;
}

bool llama_memory_hybrid_idx_context::kpool_track() const {
    // Derived from mem instead of being cached.
    return mem != nullptr && mem->get_mem_idx() != nullptr && mem->get_kpool() > 0 && !ns_ubatch.empty();
}

const llama_kv_cache_context * llama_memory_hybrid_idx_context::get_idx() const {
    return static_cast<const llama_kv_cache_context *>(ctx_idx.get());
}

uint32_t llama_memory_hybrid_idx_context::get_n_stream() const {
    GGML_ASSERT(i_cur < ns_ubatch.size());

    return ns_ubatch[i_cur];
}

llama_memory_hybrid_idx_context::kpool_access::kpool_access(ggml_context * ctx, ggml_tensor * k, int64_t n_embd) : ctx(ctx) {
    // rows are the per-token part (glm5-next: key | gate, qwen4exp: key), then the pooled key
    const int64_t n_tok = k->ne[0] - n_embd;
    GGML_ASSERT(n_tok > 0 && n_tok % n_embd == 0);

    const int64_t n_cells = k->ne[1]*k->ne[2];

    // Pool indices can refer to other streams. Revisit these full-storage views if that changes:
    // https://github.com/ggml-org/llama.cpp/pull/27773#discussion_r4130905603
    key_gate = ggml_view_2d(ctx, k, n_tok,  n_cells, k->nb[1], 0);
    pooled   = ggml_view_2d(ctx, k, n_embd, n_cells, k->nb[1], ggml_row_size(k->type, n_tok));
}

ggml_tensor * llama_memory_hybrid_idx_context::kpool_access::gather_key_gate(ggml_tensor * idxs) const {
    return ggml_get_rows(ctx, key_gate, idxs);
}

ggml_tensor * llama_memory_hybrid_idx_context::kpool_access::scatter_pooled(ggml_tensor * values, ggml_tensor * idxs) const {
    return ggml_set_rows(ctx, pooled, values, idxs);
}

ggml_tensor * llama_memory_hybrid_idx_context::kpool_access::gather_pooled(ggml_tensor * idxs) const {
    return ggml_get_rows(ctx, pooled, idxs);
}

llama_memory_hybrid_idx_context::kpool_access llama_memory_hybrid_idx_context::get_kpool_access(
        ggml_context * ctx, int32_t il, int64_t n_embd) const {
    GGML_ASSERT(mem != nullptr && mem->get_mem_idx() != nullptr);

    return kpool_access(ctx, mem->get_mem_idx()->get_k_storage(il), n_embd);
}

ggml_tensor * llama_memory_hybrid_idx_context::gather_mla_rows(
        ggml_context * ctx, ggml_tensor * idxs, int64_t n_rows, int64_t n_embd, int32_t il) const {
    GGML_ASSERT(mem != nullptr);
    ggml_tensor * k = mem->get_mem_attn()->get_k_storage(il);
    GGML_ASSERT(k->ne[0] == n_embd);

    ggml_tensor * rows = ggml_view_2d(ctx, k, k->ne[0], k->ne[1]*k->ne[2], k->nb[1], 0);
    return ggml_get_rows(ctx, rows, ggml_reshape_1d(ctx, idxs, n_rows));
}

// k-pool DSA indexer (glm5-next, qwen4exp QSA)

// Sizes only, used by the full cache context so get_n_kpool() works during graph reserve.
llama_memory_hybrid_idx_context::kpool_state llama_memory_hybrid_idx_context::kpool_build_sizes() const {
    const auto & lay = mem->kpool_layout_get();

    kpool_state st;
    st.n_pool_real = lay.n_pool_real;
    st.cache_safe  = lay.cache_safe;
    st.n_pool_g    = kpool_pad(st.n_pool_real); // [TAG_FN_L22_QSA_SEQPOOLS]

    return st;
}

// Which pools this ubatch must re-pool.
// Pool cache lifecycle:
// 1. cpy_k writes each token's key | gate into its idx cache row, pooled slot are zeroed.
// 2. This marks the pools the ubatch touches or completes as new, during decode that's one pool every kpool tokens, zero elsewise.
// 3. The graph pools only the new pools and set_rows each result into the pooled slot of the pool's last member row.
// 4. All pools are gathered in one get_rows via pool_cells, fresh ones just written, older ones from whatever batch last wrote them.
// A seq_* edit regroups the pools from the edited position on, so it stales them and the first ubatch of the next batch
// rebuilds them from the still-valid key | gate rows, rewriting the (possibly different) rep rows.
// Orphaned pooled slots are never cleared, a slot is only ever read through pool_cells, which follows the current grouping.
void llama_memory_hybrid_idx_context::kpool_build_state(const llama_ubatch & ubatch) {
    const auto & lay = mem->kpool_layout_get();
    auto & st = *kpool_st;

    st.n_pool_real = lay.n_pool_real;
    st.cache_safe  = lay.cache_safe;
    st.n_new       = 0;
    st.new_list.clear(); // [TAG_FN_L10_KPOOLINC]
    // [TAG_FN_L22_QSA_SEQPOOLS] every sequence's pools unless the end of this function lists one
    st.n_pool_g    = kpool_pad(st.n_pool_real);
    st.seq_only    = -1;
    st.n_pool_list = 0;
    if (++st.generation == 0) {
        std::fill(st.is_new.begin(), st.is_new.end(), 0);
        st.generation = 1;
    }
    st.is_new.resize(lay.n_pool_real, 0);

    auto mark = [&](uint32_t ip) {
        if (st.is_new[ip] != st.generation) {
            st.is_new[ip] = st.generation;
            ++st.n_new;
            st.new_list.push_back(ip); // [TAG_FN_L10_KPOOLINC]
        }
    };

    const uint32_t kpool = mem->get_kpool();
    std::array<uint32_t, LLAMA_MAX_SEQ> pool_start;
    uint32_t ip = 0;
    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        const auto & sq = lay.seqs[s];
        pool_start[s] = ip;
        ip += (uint32_t) sq.pools.size();

        if (!st.cache_safe) {
            continue;
        }

        // A sequence edit invalidates only pools ending after the edited position.
        const llama_pos stale_from = i_cur == 0 ?
            mem_idx_stale_batch[s] : llama_memory_hybrid_idx::POS_CLEAN;
        if (stale_from == llama_memory_hybrid_idx::POS_CLEAN) {
            continue;
        }

        auto first = std::lower_bound(sq.pools.begin(), sq.pools.end(), stale_from,
                [&](uint32_t j, llama_pos p) { return sq.cells[j + kpool - 1].first < p; });
        for (auto it = first; it != sq.pools.end(); ++it) {
            mark(pool_start[s] + (uint32_t) (it - sq.pools.begin()));
        }
    }
    GGML_ASSERT(ip == st.is_new.size());

    if (!st.cache_safe) {
        std::fill(st.is_new.begin(), st.is_new.end(), st.generation);
        st.n_new   = st.n_pool_real;
        st.n_new_g = std::max(st.n_new, 1u);
        return;
    }

    // in order mode a token's cell gives its rank, and the rank its pool: positions cannot, as an image shares one
    const bool by_order = mem->get_kpool_by_order();
    const auto *   sinfo = by_order ? &sinfos_kpool[i_cur] : nullptr;
    const uint32_t n_tps = by_order ? (uint32_t) sinfo->size() : 0;

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const llama_pos p = ubatch.pos[i];
        for (int32_t k = 0; k < ubatch.n_seq_id[i]; ++k) {
            const llama_seq_id s = ubatch.seq_id[i][k];
            const auto & sq = lay.seqs[s];
            if (by_order) {
                const int64_t r = kpool_rank(sq.cells, p, sinfo->idxs[i / n_tps][i % n_tps]);
                GGML_ASSERT(r >= 0);
                if ((size_t) r / kpool < sq.pools.size()) {
                    mark(pool_start[s] + (uint32_t) (r / kpool));
                }
                continue;
            }
            auto it = std::upper_bound(sq.pools.begin(), sq.pools.end(), p,
                    [&](llama_pos pos, uint32_t j) { return pos < sq.cells[j].first; });
            if (it == sq.pools.begin()) {
                continue;
            }
            --it;
            if (p <= sq.cells[*it + kpool - 1].first) {
                mark(pool_start[s] + (uint32_t) (it - sq.pools.begin()));
            }
        }
    }

    // [TAG_FN_L22_QSA_SHAREPFX] a shared prefix pool sits in the layout of every sharing sequence (same members, same
    // rep): re-pool it once, from its first place in the layout's order (both fills follow is_new / new_list)
    if (lay.pfx_safe && st.n_new > 1) {
        std::vector<std::pair<uint32_t, uint32_t>> reps; // (rep cell, pool)
        reps.reserve(st.n_new);
        for (const uint32_t ipn : st.new_list) {
            const llama_seq_id s = (llama_seq_id) (std::upper_bound(pool_start.begin(), pool_start.end(), ipn) - pool_start.begin()) - 1;
            GGML_ASSERT(s >= 0 && s < (llama_seq_id) LLAMA_MAX_SEQ);
            const auto &   sq = lay.seqs[s];
            const uint32_t j  = sq.pools[ipn - pool_start[s]];
            reps.emplace_back(sq.cells[j + kpool - 1].second, ipn);
        }
        std::sort(reps.begin(), reps.end());
        bool dup = false;
        for (size_t i = 1; i < reps.size(); ++i) {
            if (reps[i].first == reps[i - 1].first) {
                st.is_new[reps[i].second] = 0;   // generations start at 1
                dup = true;
            }
        }
        if (dup) {
            st.new_list.erase(std::remove_if(st.new_list.begin(), st.new_list.end(),
                    [&](uint32_t ipn) { return st.is_new[ipn] != st.generation; }), st.new_list.end());
            st.n_new = (uint32_t) st.new_list.size();
        }
    }

    // a ubatch touches at most t_s/kpool + 1 pools per sequence, pad to that bound so the graph keeps its shape
    // as the count moves; reserve sizes the list for every pool the cache can hold, so never pad past n_pool_max
    const auto *   idx        = mem->get_mem_idx();
    const uint32_t n_pool_max = idx->get_size() / kpool * idx->get_n_seq_max();
    const uint32_t bound = ubatch.n_tokens/kpool + ubatch.n_seqs_unq;
    st.n_new_g = std::max({st.n_new, 1u, std::min({bound, kpool_pad(st.n_pool_real) - 1, n_pool_max})});

    // [TAG_FN_L22_QSA_SEQPOOLS] a single-sequence ubatch lists only its sequence's pools (D12): the other sequences'
    // pools score -INF for its tokens, so the indexer and the top-k skip them; the new-pool list and the pad cells stay
    // global (set_input_kpool). Not for a non-unified cache, a shared layout or a top-k that is not index-stable.
    auto & qs = mem->qsa_stats_mut();
    ++qs.ubatches;
    qs.pfx_kept += lay.pfx_safe ? 1 : 0;
    // [TAG_FN_L24_SEQPOOLS_IMG] not for an M-RoPE embedding ubatch (a picture, or a merged picture turn: keyed positional
    // vectors): listed, a 2-slot server's picture turns scored KLD 0.057-0.059 vs 0.0057 on 1 slot (img12, 1,024 forced tokens,
    // v0.25.1 per-chunk and the merged pass alike); the full pool list (QSA_SEQPOOLS=0 for that ubatch) gave 0.0059
    if (mem->qsa_seqpools() != 0 && mem->qsa_seqpools_ok() && idx->get_n_stream() == 1 && ubatch.n_seqs_unq == 1 &&
            ubatch.n_tokens > 0 && (ubatch.token != nullptr || !ubatch.is_pos_2d())) {
        const llama_seq_id s0 = ubatch.seq_id[0][0];
        bool one = s0 >= 0 && s0 < (llama_seq_id) LLAMA_MAX_SEQ;
        for (uint32_t i = 0; i < ubatch.n_tokens && one; ++i) {
            one = ubatch.n_seq_id[i] == 1 && ubatch.seq_id[i][0] == s0;
        }
        if (one) {
            ++qs.eligible;
            const uint32_t n_s = (uint32_t) lay.seqs[s0].pools.size();
            const uint32_t w   = llama_fn_qsa_seqpools_width(st.n_pool_real, n_s, mem->qsa_n_top(), kpool_pad_g());
            if (w > 0) {
                st.n_pool_g    = w;
                st.seq_only    = s0;
                st.n_pool_list = n_s;
                ++qs.filtered;
                qs.pools_lst += w;
                qs.pools_all += kpool_pad(st.n_pool_real);
            }
        }
    }
    if (qs.ubatches % 4096 == 0 && (mem->qsa_seqpools() != 0 || mem->qsa_sharepfx()) && llama_fn_l22_passlog_level() >= 2) {
        llama_fn_l22_log("graph", "[TAG_FN_L22_QSA_SEQPOOLS] %llu k-pool states: %llu eligible, %llu listed one sequence "
                "(mean %.0f of %.0f pools), %llu kept the cache through a shared prefix, %llu checked",
                (unsigned long long) qs.ubatches, (unsigned long long) qs.eligible, (unsigned long long) qs.filtered,
                qs.filtered ? (double) qs.pools_lst / qs.filtered : 0.0, qs.filtered ? (double) qs.pools_all / qs.filtered : 0.0,
                (unsigned long long) qs.pfx_kept, (unsigned long long) qs.checked);
    }
}

const llama_memory_hybrid_idx_context::kpool_state & llama_memory_hybrid_idx_context::kpool_cur() const {
    GGML_ASSERT(kpool_st != nullptr && i_kpool == i_cur && "k-pool state read before apply()");

    return *kpool_st;
}

uint32_t llama_memory_hybrid_idx_context::get_n_kpool() const {
    return kpool_cur().n_pool_g; // [TAG_FN_L22_QSA_SEQPOOLS] kpool_pad(n_pool_real) unless one sequence is listed
}

uint32_t llama_memory_hybrid_idx_context::get_n_kpool_new() const {
    return kpool_cur().n_new_g;
}

bool llama_memory_hybrid_idx_context::get_kpool_cache_safe() const {
    return kpool_cur().cache_safe;
}

void llama_memory_hybrid_idx_context::set_input_kpool(ggml_tensor * pool_cells, ggml_tensor * pool_idxs, ggml_tensor * pool_mask, ggml_tensor * tail_idxs,
        ggml_tensor * gather_mask, bool gather, ggml_tensor * new_pool_idxs, ggml_tensor * new_pool_rep,
        const llama_ubatch * ubatch, ggml_tensor * new_pool_pos) const {
    // [TAG_FN_L3_HOST_DIAG] LLAMA_FN_HOST_KPOOL_PROBE=1: the host time of this fill
    struct probe_scope {
        int64_t  t0;
        uint32_t n_pool;
        ~probe_scope() {
            if (t0 > 0) {
                kpool_probe_add(true, (double) (ggml_time_us() - t0), n_pool);
            }
        }
    } probe { mem != nullptr && mem->kpool_probe_on() ? ggml_time_us() : 0, (uint32_t) pool_cells->ne[0] };

    GGML_ASSERT(mem != nullptr && mem->get_mem_idx() != nullptr);
    GGML_ASSERT(ggml_backend_buffer_is_host(pool_cells->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(pool_idxs->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(pool_mask->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(tail_idxs->buffer));

    const uint32_t kpool = mem->get_kpool();
    const uint32_t n_kv  = get_idx()->get_n_kv();

    const auto & st  = kpool_cur();
    const auto & lay = mem->kpool_layout_get();

    const uint32_t n_tokens = ubatch->n_tokens;
    const uint32_t n_pool   = (uint32_t) pool_cells->ne[0];
    const uint32_t n_new    = st.n_new;
    // the graph always pools at least one entry, padded to a stable bound, see kpool_build_state
    const uint32_t n_new_g  = st.n_new_g;

    const bool by_order = mem->get_kpool_by_order();

    GGML_ASSERT(n_pool == st.n_pool_g); // [TAG_FN_L22_QSA_SEQPOOLS] kpool_pad(n_pool_real) unless one sequence is listed
    GGML_ASSERT(st.is_new.size() == st.n_pool_real);
    GGML_ASSERT(pool_mask->ne[0] == (int64_t) n_pool && pool_mask->ne[1] == (int64_t) n_tokens);
    GGML_ASSERT(tail_idxs->ne[0] == (int64_t) kpool - 1 && tail_idxs->ne[1] == (int64_t) n_tokens);
    GGML_ASSERT(pool_idxs->ne[0] == (int64_t) kpool && pool_idxs->ne[1] == (int64_t) n_pool);
    GGML_ASSERT(st.cache_safe == (new_pool_rep != nullptr));
    GGML_ASSERT(ggml_backend_buffer_is_host(new_pool_idxs->buffer));
    GGML_ASSERT(new_pool_idxs->ne[0] == (int64_t) kpool && new_pool_idxs->ne[1] == (int64_t) n_new_g);
    if (new_pool_rep != nullptr) {
        GGML_ASSERT(ggml_backend_buffer_is_host(new_pool_rep->buffer));
        GGML_ASSERT(new_pool_rep->ne[0] == (int64_t) n_new_g);
    }
    if (new_pool_pos != nullptr) {
        GGML_ASSERT(ggml_backend_buffer_is_host(new_pool_pos->buffer));
        GGML_ASSERT(new_pool_pos->ne[0] == 4*(int64_t) n_new_g);
    }

    const uint32_t kv_size = mem->get_mem_idx()->get_size();
    const uint32_t n_stream_kv = mem->get_mem_idx()->get_n_stream();

    // [TAG_FN_L22_QSA_SEQPOOLS] a listed sequence (kpool_build_state): its pools first, then padding; the new-pool list,
    // the pad cells and the per-sequence input cache stay global
    const llama_seq_id s_list = st.seq_only;
    if (s_list >= 0) {
        GGML_ASSERT(!gather && n_stream_kv == 1 && st.cache_safe);
        GGML_ASSERT(st.n_pool_list == (uint32_t) lay.seqs[s_list].pools.size() && st.n_pool_list < n_pool);
        for (uint32_t i = 0; i < n_tokens; ++i) {
            GGML_ASSERT(ubatch->n_seq_id[i] == 1 && ubatch->seq_id[i][0] == s_list && "QSA_SEQPOOLS: a listed ubatch holds one sequence");
        }
    }

    auto gcell = [&](const llama_memory_hybrid_idx::kpool_layout::seq & sq, uint32_t cell) {
        return (int64_t) sq.strm*kv_size + cell;
    };

    // Sequences present in this ubatch, pools of absent sequences must fall on the scatter sentinel row.
    std::vector<uint8_t> seq_in_ub(LLAMA_MAX_SEQ, 0);
    for (uint32_t i = 0; i < n_tokens; ++i) {
        for (int32_t k = 0; k < ubatch->n_seq_id[i]; ++k) {
            seq_in_ub[ubatch->seq_id[i][k]] = 1;
        }
    }

    // Use the first ubatch cell for padded gathers.
    int64_t dummy_cell = 0;
    {
        const llama_seq_id s = ubatch->seq_id[0][0];
        const auto & sq = lay.seqs[s];
        auto it = std::lower_bound(sq.cells.begin(), sq.cells.end(), std::make_pair(ubatch->pos[0], 0u));
        GGML_ASSERT(it != sq.cells.end() && it->first == ubatch->pos[0]);
        dummy_cell = gcell(sq, it->second);
    }

    // in order mode a token sees the pools and the tail up to its own rank in the sequence, which its cell pins down
    std::vector<int64_t> rank;
    if (by_order) {
        const auto &   sinfo = sinfos_kpool[i_cur];
        const uint32_t n_tps = (uint32_t) sinfo.size();

        rank.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; ++i) {
            rank[i] = kpool_rank(lay.seqs[ubatch->seq_id[i][0]].cells, ubatch->pos[i], sinfo.idxs[i / n_tps][i % n_tps]);
            GGML_ASSERT(rank[i] >= 0);
        }
    }

    // Gather maps padding to a real cell and masks it separately.
    const int32_t sentinel = gather ? (int32_t) dummy_cell : (int32_t) n_kv;

    float *  gm    = nullptr;
    uint32_t n_sel = 0;
    uint32_t n_top = 0; // Pools per token in the selection.
    if (gather_mask != nullptr) {
        GGML_ASSERT(ggml_backend_buffer_is_host(gather_mask->buffer));
        GGML_ASSERT(gather_mask->type == GGML_TYPE_F32);
        GGML_ASSERT(gather_mask->ne[3] == (int64_t) n_tokens && gather_mask->ne[1] == 1 && gather_mask->ne[2] == 1);
        n_sel = (uint32_t) gather_mask->ne[0];
        // The tail slots, when selected, are the n_sel % kpool != 0 remainder.
        n_top = n_sel / kpool;
        GGML_ASSERT(n_sel % kpool == 0 || n_sel % kpool == kpool - 1);
        gm = (float *) gather_mask->data;
    }

    // pools are laid out per sequence
    std::vector<uint32_t>  seq_pool_start(LLAMA_MAX_SEQ, 0);
    std::vector<llama_pos> pool_end; // the full rebuild's

    int32_t * pcell = (int32_t *) pool_cells->data;
    int32_t * pidx  = (int32_t *) pool_idxs->data;
    int32_t * nidx  = (int32_t *) new_pool_idxs->data;
    int64_t * nrep  = new_pool_rep != nullptr ? (int64_t *) new_pool_rep->data : nullptr;
    int32_t * npos  = new_pool_pos != nullptr ? (int32_t *) new_pool_pos->data : nullptr;

    // [TAG_FN_L22_QSA_SEQPOOLS] what the fills write: the graph's tensors, or the full lists of the check below
    uint32_t     n_pool_w = n_pool;
    llama_seq_id s_list_w = s_list;
    void *       mask_w   = pool_mask->data;

    if (npos != nullptr) {
        std::fill(npos, npos + 4*n_new_g, 0);
    }

    // the full rebuild: every pool's entries, the pad search over a fresh bitmap, the masks
    auto fill_full = [&]() {
    std::fill(seq_pool_start.begin(), seq_pool_start.end(), 0);
    pool_end.clear();
    pool_end.reserve(n_pool);

    uint32_t i_new = 0;
    const bool listed = s_list_w >= 0;   // [TAG_FN_L22_QSA_SEQPOOLS]
    uint32_t   lp     = 0;               // listed entries written
    std::vector<int64_t> reps_all;       // every pool's rep (the pad search) when the arrays hold one sequence
    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        const auto & sq = lay.seqs[s];
        seq_pool_start[s] = (uint32_t) pool_end.size();

        const bool inert = !gather && n_stream_kv > 1 && !seq_in_ub[s];
        const bool write = !listed || s == s_list_w;

        for (size_t pi = 0; pi < sq.pools.size(); ++pi) {
            const uint32_t j  = sq.pools[pi];
            const uint32_t ip = (uint32_t) pool_end.size();
            const uint32_t wp = listed ? lp : ip;   // [TAG_FN_L22_QSA_SEQPOOLS] where its entries go

            // The pooled key lives in the last member's row.
            const uint32_t rep = sq.cells[j + kpool - 1].second;
            if (listed) {
                reps_all.push_back(gcell(sq, rep));
            }
            if (write) {
                GGML_ASSERT(wp + 1 < n_pool_w);
                pcell[wp] = (int32_t) gcell(sq, rep);

                for (uint32_t k = 0; k < kpool; ++k) {
                    pidx[(size_t) wp*kpool + k] = inert ? sentinel :
                        (int32_t) (gather ? gcell(sq, sq.cells[j + k].second) : (int64_t) sq.cells[j + k].second);
                }
                lp += listed ? 1 : 0;
            }

            if (st.is_new[ip] == st.generation) {
                GGML_ASSERT(i_new < n_new);
                for (uint32_t k = 0; k < kpool; ++k) {
                    nidx[(size_t) i_new*kpool + k] = (int32_t) gcell(sq, sq.cells[j + k].second);
                }
                if (nrep != nullptr) {
                    nrep[i_new] = gcell(sq, rep);
                }
                if (npos != nullptr) {
                    // a pooled key is rotated to the M-RoPE position of its first member
                    const uint32_t c = sq.cells[j].second;
                    const auto &   e = mem->get_mem_idx()->get_cells(s).ext_get(c);
                    npos[0*n_new_g + i_new] = sq.cells[j].first;
                    npos[1*n_new_g + i_new] = e.y;
                    npos[2*n_new_g + i_new] = e.x;
                    npos[3*n_new_g + i_new] = sq.cells[j].first;
                }
                ++i_new;
            }

            pool_end.push_back(sq.cells[j + kpool - 1].first);
        }
    }
    GGML_ASSERT(i_new == n_new);

    // Padded entries re-pool cells whose pooled slot is never read: only the reps of complete pools are read.
    // Each entry takes its own cell, entries sharing one would write it from several threads in the scatter.
    if (n_new_g > n_new) {
        // [TAG_FN_R1_KPOOL_PAD] the smallest cells that are no pool's rep, from a bitmap over the K storage: O(n_pool),
        // where sorting every rep cost O(n_pool log n_pool) on most decode steps (a 3-token step pads its one entry
        // whenever no pool completes; 61K pools at 246K). The same cells as the sorted search.
        const int64_t n_cells = (int64_t) kv_size*n_stream_kv;
        std::vector<uint64_t> rep_bits((size_t) (n_cells + 63)/64, 0);
        const size_t n_reps = listed ? reps_all.size() : pool_end.size();
        for (size_t ip = 0; ip < n_reps; ++ip) {
            const int64_t c = listed ? reps_all[ip] : pcell[ip];
            if (c >= 0 && c < n_cells) {
                rep_bits[(size_t) c >> 6] |= 1ull << (c & 63);
            }
        }
        auto is_rep = [&](int64_t c) {
            return c < n_cells && ((rep_bits[(size_t) c >> 6] >> (c & 63)) & 1ull) != 0;
        };

        int64_t pad_cell = 0;
        for (uint32_t i = n_new; i < n_new_g; ++i, ++pad_cell) {
            while (is_rep(pad_cell)) {
                ++pad_cell;
            }
            GGML_ASSERT(pad_cell < n_cells);
            for (uint32_t k = 0; k < kpool; ++k) {
                nidx[(size_t) i*kpool + k] = (int32_t) pad_cell;
            }
            if (nrep != nullptr) {
                nrep[i] = pad_cell;
            }
        }
    }

    const uint32_t n_pool_real = listed ? lp : (uint32_t) pool_end.size();
    for (uint32_t ip = n_pool_real; ip < n_pool_w; ++ip) {
        pcell[ip] = (int32_t) dummy_cell; // pool_cells always addresses the K storage
        for (uint32_t k = 0; k < kpool; ++k) {
            pidx[(size_t) ip*kpool + k] = sentinel;
        }
    }

    // a pool is visible when it belongs to the token's sequence and ends at or before it
    auto fill_mask = [&](auto * data) {
        using T = std::remove_pointer_t<decltype(data)>;
        const T keep = llama_cast<T>(0.0f);
        const T drop = llama_cast<T>(-INFINITY);

        for (uint32_t i = 0; i < n_tokens; ++i) {
            const llama_seq_id s = ubatch->seq_id[i][0];
            const llama_pos    p = ubatch->pos[i];

            T * row = data + (size_t) i*n_pool_w;
            std::fill(row, row + n_pool_w, drop);

            const uint32_t p0 = seq_pool_start[s];
            const uint32_t p1 = p0 + (uint32_t) lay.seqs[s].pools.size();
            const uint32_t nv = by_order ? std::min(p1 - p0, (uint32_t) ((rank[i] + 1)/kpool)) :
                (uint32_t) (std::upper_bound(pool_end.begin() + p0, pool_end.begin() + p1, p) - (pool_end.begin() + p0));
            const uint32_t r0 = listed ? 0 : p0;   // [TAG_FN_L22_QSA_SEQPOOLS] the listed sequence's pools come first
            std::fill(row + r0, row + r0 + nv, keep);

            // Finite visible pools occupy the first min(nv, n_top) ranked slots.
            if (gm != nullptr) {
                const uint32_t nvc = std::min(nv, n_top);
                float * grow = gm + (size_t) i*n_sel;
                std::fill(grow,                        grow + (size_t) nvc*kpool,  0.0f);
                std::fill(grow + (size_t) nvc*kpool,   grow + (size_t) n_top*kpool, -INFINITY);
            }
        }
    };
    if (pool_mask->type == GGML_TYPE_F16) {
        fill_mask((ggml_fp16_t *) mask_w);
    } else {
        fill_mask((float *) mask_w);
    }
    };

    // [TAG_FN_L10_KPOOLINC] one stream (the cell is the K row, no inert sequences) and no shared cells: the per-pool
    // entries come from the layout's input cache, brought up to date at each sequence's tail (the pools after pools_keep
    // dropped, the new ones appended), and are copied in; the new-pool list comes from the marks, the pad cells from the
    // rep counts. The same values as the full rebuild (LLAMA_KPOOL_VERIFY=1 compares them on every fill); the full
    // rebuild cost ~6 ns per pool each ubatch (0.05 ms at 32K, 0.35 ms at 245K with 62K pools)
    auto fill_inc = [&]() {
        const bool listed = s_list_w >= 0;   // [TAG_FN_L22_QSA_SEQPOOLS]
        const int64_t n_cells = (int64_t) kv_size;
        if ((int64_t) lay.inc_rep_cnt.size() != n_cells) {
            lay.inc_rep_cnt.assign((size_t) n_cells, 0);
            for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
                lay.inc[s] = {};
                lay.seqs[s].pools_keep = 0;
            }
        }
        uint16_t * rc = lay.inc_rep_cnt.data();
        // drop every sequence's changed tail first, then append: a cell can move from one sequence to another
        for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
            auto &       c    = lay.inc[s];
            const size_t keep = std::min(lay.seqs[s].pools_keep, c.end.size());
            for (size_t k = keep; k < c.end.size(); ++k) {
                GGML_ASSERT(rc[c.pcell[k]] > 0);
                --rc[c.pcell[k]];
            }
            c.pcell.resize(keep);
            c.pidx.resize(keep*kpool);
            c.end.resize(keep);
        }
        uint32_t ip0 = 0;
        for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
            const auto & sq = lay.seqs[s];
            auto &       c  = lay.inc[s];
            for (size_t pi = c.end.size(); pi < sq.pools.size(); ++pi) {
                const uint32_t j   = sq.pools[pi];
                const uint32_t rep = sq.cells[j + kpool - 1].second;
                GGML_ASSERT((int64_t) rep < n_cells);
                c.pcell.push_back((int32_t) rep);
                ++rc[rep];
                for (uint32_t k = 0; k < kpool; ++k) {
                    c.pidx.push_back((int32_t) sq.cells[j + k].second);
                }
                c.end.push_back(sq.cells[j + kpool - 1].first);
            }
            sq.pools_keep = sq.pools.size();

            const uint32_t n = (uint32_t) c.end.size();
            seq_pool_start[s] = ip0;
            // [TAG_FN_L22_QSA_SEQPOOLS] a listed ubatch copies only its sequence, to the front
            if (n > 0 && (!listed || s == s_list_w)) {
                const uint32_t wp0 = listed ? 0 : ip0;
                GGML_ASSERT(wp0 + n < n_pool_w);
                memcpy(pcell + wp0,                c.pcell.data(), (size_t) n*sizeof(int32_t));
                memcpy(pidx + (size_t) wp0*kpool, c.pidx.data(),  (size_t) n*kpool*sizeof(int32_t));
            }
            ip0 += n;
        }
        GGML_ASSERT(ip0 == st.n_pool_real);

        // the new pools in the layout's order
        GGML_ASSERT(st.new_list.size() == n_new);
        std::vector<uint32_t> nl = st.new_list;
        std::sort(nl.begin(), nl.end());
        llama_seq_id s = 0;
        for (uint32_t i_new = 0; i_new < n_new; ++i_new) {
            const uint32_t ip = nl[i_new];
            while (ip >= seq_pool_start[s] + (uint32_t) lay.seqs[s].pools.size()) {
                ++s;
                GGML_ASSERT(s < LLAMA_MAX_SEQ);
            }
            const auto &   sq = lay.seqs[s];
            const uint32_t j  = sq.pools[ip - seq_pool_start[s]];
            for (uint32_t k = 0; k < kpool; ++k) {
                nidx[(size_t) i_new*kpool + k] = (int32_t) gcell(sq, sq.cells[j + k].second);
            }
            if (nrep != nullptr) {
                nrep[i_new] = gcell(sq, sq.cells[j + kpool - 1].second);
            }
            if (npos != nullptr) {
                const uint32_t c = sq.cells[j].second;
                const auto &   e = mem->get_mem_idx()->get_cells(s).ext_get(c);
                npos[0*n_new_g + i_new] = sq.cells[j].first;
                npos[1*n_new_g + i_new] = e.y;
                npos[2*n_new_g + i_new] = e.x;
                npos[3*n_new_g + i_new] = sq.cells[j].first;
            }
        }

        // padded entries: the smallest cells that are no pool's rep
        int64_t pad_cell = 0;
        for (uint32_t i = n_new; i < n_new_g; ++i, ++pad_cell) {
            while (pad_cell < n_cells && rc[pad_cell] != 0) {
                ++pad_cell;
            }
            GGML_ASSERT(pad_cell < n_cells);
            for (uint32_t k = 0; k < kpool; ++k) {
                nidx[(size_t) i*kpool + k] = (int32_t) pad_cell;
            }
            if (nrep != nullptr) {
                nrep[i] = pad_cell;
            }
        }

        const uint32_t ipw = listed ? st.n_pool_list : ip0;
        for (uint32_t ip = ipw; ip < n_pool_w; ++ip) {
            pcell[ip] = (int32_t) dummy_cell;
        }
        std::fill(pidx + (size_t) ipw*kpool, pidx + (size_t) n_pool_w*kpool, sentinel);

        // the masks: each row written once (dropped, the visible pools of the token's sequence, dropped)
        auto fill_mask_inc = [&](auto * data) {
            using T = std::remove_pointer_t<decltype(data)>;
            const T keep = llama_cast<T>(0.0f);
            const T drop = llama_cast<T>(-INFINITY);

            for (uint32_t i = 0; i < n_tokens; ++i) {
                const llama_seq_id si = ubatch->seq_id[i][0];
                const llama_pos    p  = ubatch->pos[i];
                const auto &       ce = lay.inc[si].end;

                const uint32_t p0 = listed ? 0 : seq_pool_start[si];   // [TAG_FN_L22_QSA_SEQPOOLS]
                const uint32_t nv = by_order ? std::min((uint32_t) ce.size(), (uint32_t) ((rank[i] + 1)/kpool)) :
                    (uint32_t) (std::upper_bound(ce.begin(), ce.end(), p) - ce.begin());

                T * row = data + (size_t) i*n_pool_w;
                std::fill(row,           row + p0,     drop);
                std::fill(row + p0,      row + p0 + nv, keep);
                std::fill(row + p0 + nv, row + n_pool_w, drop);

                if (gm != nullptr) {
                    const uint32_t nvc = std::min(nv, n_top);
                    float * grow = gm + (size_t) i*n_sel;
                    std::fill(grow,                        grow + (size_t) nvc*kpool,  0.0f);
                    std::fill(grow + (size_t) nvc*kpool,   grow + (size_t) n_top*kpool, -INFINITY);
                }
            }
        };
        if (pool_mask->type == GGML_TYPE_F16) {
            fill_mask_inc((ggml_fp16_t *) mask_w);
        } else {
            fill_mask_inc((float *) mask_w);
        }
    };

    if (mem->kpool_inc_on() && st.cache_safe && n_stream_kv == 1) {
        fill_inc();
        if (kpool_tail_mode() & 2) {
            // LLAMA_KPOOL_VERIFY=1: the full rebuild must give the same bytes
            auto snap = [](const ggml_tensor * t) {
                const uint8_t * d = (const uint8_t *) t->data;
                return std::vector<uint8_t>(d, d + ggml_nbytes(t));
            };
            const auto v_pcell = snap(pool_cells);
            const auto v_pidx  = snap(pool_idxs);
            const auto v_mask  = snap(pool_mask);
            const auto v_nidx  = snap(new_pool_idxs);
            const auto v_nrep  = new_pool_rep ? snap(new_pool_rep) : std::vector<uint8_t>();
            const auto v_npos  = new_pool_pos ? snap(new_pool_pos) : std::vector<uint8_t>();
            const auto v_gm    = gather_mask ? snap(gather_mask) : std::vector<uint8_t>();
            fill_full();
            const bool same = v_pcell == snap(pool_cells) && v_pidx == snap(pool_idxs) && v_mask == snap(pool_mask) &&
                v_nidx == snap(new_pool_idxs) && (!new_pool_rep || v_nrep == snap(new_pool_rep)) &&
                (!new_pool_pos || v_npos == snap(new_pool_pos)) && (!gather_mask || v_gm == snap(gather_mask));
            if (!same) {
                GGML_ABORT("[TAG_FN_L10_KPOOLINC] the incremental k-pool inputs differ from a rebuild (%u pools, %u new, %u tokens)",
                        st.n_pool_real, n_new, n_tokens);
            }
        }
    } else {
        fill_full();
    }

    // [TAG_FN_L22_QSA_SEQPOOLS] LLAMA_FN_L22_QSA_SEQPOOLS=2: the listed inputs against the full lists of the same ubatch
    // (a full rebuild without the listing into scratch arrays), and with a shared prefix kept safe, every rep against one
    // member set. Aborts at the first difference.
    if (mem->qsa_seqpools() == 2 && (s_list >= 0 || lay.pfx_safe)) {
        const uint32_t n_full = kpool_pad(st.n_pool_real);
        const size_t   esz    = ggml_type_size(pool_mask->type);
        std::vector<int32_t> f_pcell(n_full), f_pidx((size_t) n_full*kpool), f_nidx((size_t) kpool*n_new_g);
        std::vector<int64_t> f_nrep(n_new_g);
        std::vector<int32_t> f_npos((size_t) 4*n_new_g, 0);
        std::vector<uint8_t> f_mask((size_t) n_full*n_tokens*esz);

        int32_t * t_pcell = pcell;
        int32_t * t_pidx  = pidx;
        int32_t * t_nidx  = nidx;
        int64_t * t_nrep  = nrep;
        int32_t * t_npos  = npos;
        float   * t_gm    = gm;
        pcell = f_pcell.data();
        pidx  = f_pidx.data();
        nidx  = f_nidx.data();
        nrep  = nrep != nullptr ? f_nrep.data() : nullptr;
        npos  = npos != nullptr ? f_npos.data() : nullptr;
        gm    = nullptr;
        n_pool_w = n_full;
        s_list_w = -1;
        mask_w   = f_mask.data();
        fill_full();
        pcell = t_pcell;
        pidx  = t_pidx;
        nidx  = t_nidx;
        nrep  = t_nrep;
        npos  = t_npos;
        gm    = t_gm;
        n_pool_w = n_pool;
        s_list_w = s_list;
        mask_w   = pool_mask->data;

        std::string err;
        if (s_list >= 0) {
            auto to_f32 = [&](const void * src, size_t n) {
                std::vector<float> v(n);
                if (pool_mask->type == GGML_TYPE_F16) {
                    const ggml_fp16_t * h = (const ggml_fp16_t *) src;
                    for (size_t i = 0; i < n; ++i) {
                        v[i] = ggml_fp16_to_fp32(h[i]);
                    }
                } else {
                    memcpy(v.data(), src, n*sizeof(float));
                }
                return v;
            };
            const std::vector<float> m_l = to_f32(pool_mask->data, (size_t) n_pool*n_tokens);
            const std::vector<float> m_f = to_f32(f_mask.data(),   (size_t) n_full*n_tokens);

            llama_fn_qsa_lists lst;
            lst.pcell  = pcell;
            lst.pidx   = pidx;
            lst.mask   = m_l.data();
            lst.n_pool = n_pool;
            llama_fn_qsa_lists full;
            full.pcell  = f_pcell.data();
            full.pidx   = f_pidx.data();
            full.mask   = m_f.data();
            full.n_pool = n_full;
            err = llama_fn_qsa_seqpools_check(lst, st.n_pool_list, full, seq_pool_start[s_list], kpool, n_tokens,
                                              (int32_t) dummy_cell, sentinel);
            if (err.empty() && memcmp(nidx, f_nidx.data(), f_nidx.size()*sizeof(int32_t)) != 0) {
                err = "the new-pool members differ";
            }
            if (err.empty() && nrep != nullptr && memcmp(nrep, f_nrep.data(), f_nrep.size()*sizeof(int64_t)) != 0) {
                err = "the new-pool reps differ";
            }
            if (err.empty() && npos != nullptr && memcmp(npos, f_npos.data(), f_npos.size()*sizeof(int32_t)) != 0) {
                err = "the new-pool positions differ";
            }
        }
        if (err.empty() && lay.pfx_safe) {
            err = llama_fn_qsa_rep_members_check(f_pcell.data(), f_pidx.data(), st.n_pool_real, kpool);
        }
        if (!err.empty()) {
            GGML_ABORT("[TAG_FN_L22_QSA_SEQPOOLS] check failed: %s (seq %d, %u listed of %u pools, %u tokens, shared prefix %d)",
                    err.c_str(), (int) s_list, st.n_pool_list, st.n_pool_real, n_tokens, (int) lay.pfx_safe);
        }
        auto & qs = mem->qsa_stats_mut();
        if (++qs.checked % 4096 == 1) {
            llama_fn_l22_log("graph", "[TAG_FN_L22_QSA_SEQPOOLS] check: %llu ubatches agree with the full lists",
                    (unsigned long long) qs.checked);
        }
    }

    int32_t * tidx = (int32_t *) tail_idxs->data;
    for (uint32_t i = 0; i < n_tokens; ++i) {
        const llama_seq_id s = ubatch->seq_id[i][0];
        const llama_pos    p = ubatch->pos[i];
        const auto & sq = lay.seqs[s];

        const uint32_t n_tail = by_order ?
            (uint32_t) ((rank[i] + 1) % kpool) :
            (uint32_t) ((p - sq.pos_min + 1) % (llama_pos) kpool);

        for (uint32_t k = 0; k < kpool - 1; ++k) {
            int32_t cell = sentinel;
            bool    real = false;
            if (k < n_tail && by_order) {
                const uint32_t c = sq.cells[rank[i] - k].second;
                cell = (int32_t) (gather ? gcell(sq, c) : (int64_t) c);
                real = true;
            } else if (k < n_tail) {
                const llama_pos pt = p - (llama_pos) k;
                auto it = std::lower_bound(sq.cells.begin(), sq.cells.end(), std::make_pair(pt, 0u));
                if (it != sq.cells.end() && it->first == pt) {
                    cell = (int32_t) (gather ? gcell(sq, it->second) : (int64_t) it->second);
                    real = true;
                }
            }
            tidx[(size_t) i*(kpool - 1) + k] = cell;

            if (gm != nullptr && n_sel % kpool != 0) {
                gm[(size_t) i*n_sel + (size_t) n_top*kpool + k] = real ? 0.0f : -INFINITY;
            }
        }
    }
}

//
// [TAG_FN_L22_QSA_SEQPOOLS] [TAG_FN_L22_QSA_SHAREPFX] round 22 P13: pure pieces (llama-memory-hybrid-idx.h)
//

uint32_t llama_fn_qsa_kpool_pad(uint32_t n_pool, uint32_t pad_g) {
    return std::max<uint32_t>(pad_g, GGML_PAD(n_pool + 1, pad_g));
}

uint32_t llama_fn_qsa_seqpools_width(uint32_t n_all, uint32_t n_seq, uint32_t n_top, uint32_t pad_g) {
    // the CUDA top-k keeps the lowest indices on ties only in its radix select, taken for ncols >= 2048
    // (ggml-cuda/top-k.cu: [TAG_TOPK_SELECT], the [TAG_FN_L3_GPU_TOPK] two-stage form gives the same set)
    constexpr uint32_t stable_cols = 2048;

    if (n_top == 0 || pad_g == 0 || n_seq >= n_all) {
        return 0;
    }
    const uint32_t full = llama_fn_qsa_kpool_pad(n_all, pad_g);
    if (full < stable_cols) {
        return 0;
    }
    // at least n_top columns, so the picks stay n_top (n_sel unchanged)
    uint32_t cand = std::max(llama_fn_qsa_kpool_pad(n_seq, pad_g), llama_fn_qsa_kpool_pad(n_top - 1, pad_g));
    if (cand != n_top && cand < stable_cols) {
        cand = llama_fn_qsa_kpool_pad(stable_cols - 1, pad_g);
    }
    return cand < full ? cand : 0;
}

static bool qsa_same_bits(float a, float b) {
    uint32_t ua;
    uint32_t ub;
    memcpy(&ua, &a, sizeof(ua));
    memcpy(&ub, &b, sizeof(ub));
    return ua == ub;
}

std::string llama_fn_qsa_seqpools_check(const llama_fn_qsa_lists & lst, uint32_t n_list, const llama_fn_qsa_lists & full,
                                        uint32_t gstart, uint32_t kpool, uint32_t n_tokens, int32_t dummy, int32_t sentinel) {
    char buf[256];
    if (lst.pcell == nullptr || lst.pidx == nullptr || lst.mask == nullptr || full.pcell == nullptr || full.pidx == nullptr ||
            full.mask == nullptr || kpool == 0 || n_list >= lst.n_pool || (uint64_t) gstart + n_list > full.n_pool) {
        snprintf(buf, sizeof(buf), "shapes: %u listed of width %u, start %u of width %u", n_list, lst.n_pool, gstart, full.n_pool);
        return buf;
    }
    for (uint32_t i = 0; i < n_list; ++i) {
        if (lst.pcell[i] != full.pcell[gstart + i]) {
            snprintf(buf, sizeof(buf), "pool %u: rep cell %d, full list %d", i, lst.pcell[i], full.pcell[gstart + i]);
            return buf;
        }
        if (memcmp(lst.pidx + (size_t) i*kpool, full.pidx + (size_t) (gstart + i)*kpool, kpool*sizeof(int32_t)) != 0) {
            snprintf(buf, sizeof(buf), "pool %u: members differ from the full list", i);
            return buf;
        }
    }
    for (uint32_t i = n_list; i < lst.n_pool; ++i) {
        if (lst.pcell[i] != dummy) {
            snprintf(buf, sizeof(buf), "padded pool %u: rep cell %d, not the dummy %d", i, lst.pcell[i], dummy);
            return buf;
        }
        for (uint32_t k = 0; k < kpool; ++k) {
            if (lst.pidx[(size_t) i*kpool + k] != sentinel) {
                snprintf(buf, sizeof(buf), "padded pool %u: member %u is %d, not the sentinel %d", i, k,
                         lst.pidx[(size_t) i*kpool + k], sentinel);
                return buf;
            }
        }
    }
    for (uint32_t t = 0; t < n_tokens; ++t) {
        const float * ml = lst.mask  + (size_t) t*lst.n_pool;
        const float * mf = full.mask + (size_t) t*full.n_pool;
        for (uint32_t i = 0; i < n_list; ++i) {
            if (!qsa_same_bits(ml[i], mf[gstart + i])) {
                snprintf(buf, sizeof(buf), "token %u, pool %u: mask %g, full list %g", t, i, ml[i], mf[gstart + i]);
                return buf;
            }
        }
        for (uint32_t i = n_list; i < lst.n_pool; ++i) {
            if (!(std::isinf(ml[i]) && ml[i] < 0.0f)) {
                snprintf(buf, sizeof(buf), "token %u, padded pool %u: mask %g, not -inf", t, i, ml[i]);
                return buf;
            }
        }
        for (uint32_t j = 0; j < full.n_pool; ++j) {
            if (j >= gstart && j < gstart + n_list) {
                continue;
            }
            if (!(std::isinf(mf[j]) && mf[j] < 0.0f)) {
                snprintf(buf, sizeof(buf), "token %u sees pool %u of the full list outside its sequence (%g)", t, j, mf[j]);
                return buf;
            }
        }
    }
    return std::string();
}

std::string llama_fn_qsa_rep_members_check(const int32_t * pcell, const int32_t * pidx, uint32_t n, uint32_t kpool) {
    if (n == 0) {
        return std::string();
    }
    std::vector<uint32_t> ord(n);
    for (uint32_t i = 0; i < n; ++i) {
        ord[i] = i;
    }
    std::sort(ord.begin(), ord.end(), [&](uint32_t a, uint32_t b) {
        return pcell[a] != pcell[b] ? pcell[a] < pcell[b] : a < b;
    });
    for (uint32_t i = 1; i < n; ++i) {
        const uint32_t a = ord[i - 1];
        const uint32_t b = ord[i];
        if (pcell[a] == pcell[b] &&
                memcmp(pidx + (size_t) a*kpool, pidx + (size_t) b*kpool, kpool*sizeof(int32_t)) != 0) {
            char buf[160];
            snprintf(buf, sizeof(buf), "pools %u and %u share rep cell %d with different members", a, b, pcell[a]);
            return buf;
        }
    }
    return std::string();
}

bool llama_fn_qsa_pfx_step(llama_fn_qsa_pfx & st, size_t j, int32_t cnt) {
    if (st.irr) {
        return false;
    }
    if (cnt > 1) {
        st.shared = true;
        if (j == st.pfx_n) {
            st.cmin = st.pfx_n == 0 ? cnt : std::min(st.cmin, cnt);
            st.cmax = st.pfx_n == 0 ? cnt : std::max(st.cmax, cnt);
            ++st.pfx_n;
        } else {
            st.irr = true;   // a shared cell after an own cell
            return false;
        }
    }
    return true;
}

bool llama_fn_qsa_sharepfx_safe(const llama_fn_qsa_pfx * seqs, const int32_t * pos_min, int n) {
    int n_s   = 0;
    int first = -1;
    for (int i = 0; i < n; ++i) {
        if (!seqs[i].shared) {
            continue;
        }
        if (seqs[i].irr || seqs[i].pfx_n == 0) {
            return false;
        }
        if (first < 0) {
            first = i;
        } else if (seqs[i].pfx_n != seqs[first].pfx_n || pos_min[i] != pos_min[first]) {
            return false;
        }
        ++n_s;
    }
    if (n_s < 2) {
        return false;
    }
    for (int i = 0; i < n; ++i) {
        if (seqs[i].shared && (seqs[i].cmin != n_s || seqs[i].cmax != n_s)) {
            return false;
        }
    }
    return true;
}
