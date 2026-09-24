#include "llama-memory-recurrent.h"

#include "ggml-backend.h"
#include "llama-impl.h"
#include "llama-io.h"
#include "llama-batch.h"
#include "llama-model.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <cstdlib>

// [TAG_XSEQ_PLANES] LLAMA_XSEQ_FIX=0 restores the old behaviour (snapshot rows of moved extra cells stay behind)
static bool llama_xseq_fix_enabled() {
    static const bool enabled = [] {
        const char * e = getenv("LLAMA_XSEQ_FIX");
        return !(e && e[0] == '0' && e[1] == '\0');
    }();
    return enabled;
}

// [TAG_4C_GDN_REPLAY] Gated DeltaNet state replay instead of snapshot groups. Design note:
//
// Old layout: s_l has 1 + n_rs_seq row groups per cell. A ubatch writes the state after each of its last K tokens,
//   and a rollback of r tokens (seq_rm, rs_idx = r) reads group r. Qwen3.8-27B, 4 slots, n_rs_seq 3: 2,304 MiB f32.
// Replay layout (default; GDN_REPLAY=0 when the memory is created keeps the old layout and kernels):
//   - s_l has one row per cell: the committed state C.
//   - ring_l has, per cell and layer, the inputs (k, v, g, beta) of the n_ring <= n_rs_seq tokens that follow C.
//     The state of the sequence is C with those tokens applied. cell.n_ring counts them.
//   - A ubatch of T tokens (GGML_OP_GATED_DELTA_NET_REPLAY) first applies the live ring tokens of the cell to C
//     without output, then runs the T new tokens. It commits the state before its last n_w = min(T, n_rs_seq) new
//     tokens as the new C and stores those n_w tokens as the new ring.
//   - Rollback of r tokens stays metadata: seq_rm sets rs_idx = r as before (refused when r > n_rb, where the old
//     groups gave a stale state), and the next ubatch replays n_ring - r ring tokens (s_copy consumes rs_idx).
//   - The conv state keeps its 1 + n_rs_seq groups (5.6 MiB per cell per group at Qwen3.8-27B), unchanged. cell.n_rb
//     (<= n_ring) counts the tokens whose conv groups are valid: n_w after a ubatch, 0 after a restore (only group 0
//     is saved) or after an extra folded a rollback (its groups are not shifted).
// State readers:
//   - graph: s_copy_r gives the group-0 source rows of C and the ring; build_rs moves every extra cell's row, so a
//     cell that find_slot relocates carries C, ring and n_ring (swapped with pos/src). A pending rollback of such an
//     extra folds into its n_ring. [TAG_XSEQ_PLANES] still moves the conv groups.
//   - seq_cp shares the cell, with C, ring and n_ring; find_slot copies n_ring when it splits a shared cell.
//   - state_write/state_read (slot save/restore, prompt cache, park/resume, context checkpoints) store C, the ring and
//     the live ring count instead of a materialized state: a host-side replay would round differently from the GPU
//     kernel, and the restored sequence would not continue bit-identically. Old-format data loads as C with an empty
//     ring. Ring-format data needs GDN_REPLAY on (a clear load error otherwise, and the caller re-processes).
// Why decode stays bit-identical: each token applies the same f32 update to the same f32 inputs in the same order as
//   in the snapshot kernel. C and the ring are exact copies, so the state after the replay has the bits of the group
//   the old path would read, and the new tokens then run the old per-token code. The old kernel is not changed.
// Prefill: [TAG_GDN_CHUNKED_PF] a long ubatch runs its first T - n_w tokens on the chunked kernel and the last n_w on the
//   replay op, with the same committed state and ring layout (design note in src/models/delta-net-base.cpp).
// Cost: the kernel runs up to n_rs_seq extra state-only token steps and writes 1 state per cell instead of K.
static bool llama_gdn_replay_env() {
    // read at every memory creation, so a test can compare both layouts in one process
    const char * e = getenv("GDN_REPLAY");
    return !(e && e[0] == '0' && e[1] == '\0');
}

// archs whose recurrent layers are scalar-gate gated delta nets built by llm_build_delta_net_base
static bool llama_gdn_replay_arch(llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_QWEN35:
        case LLM_ARCH_QWEN35MOE:
            return true;
        default:
            return false;
    }
}

// one ring token: k [S_k*H_v] (H_v heads, the graph repeats k when the fused op is off) | v [S_v*H_v] | g [H_v] |
// beta [H_v], padded to 4 floats for the float4 loads of the CUDA kernel
// [TAG_RS_SNAP_DEPTH] graphs that write the state before the ubatch into group n_seq_tokens (delta-net-base conv + GDN)
static bool llama_rs_pre_state_arch(llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_QWEN35:
        case LLM_ARCH_QWEN35MOE:
        case LLM_ARCH_QWEN3NEXT:
            return true;
        default:
            return false;
    }
}

static int64_t llama_gdn_ring_slot(const llama_hparams & hparams) {
    const int64_t H_v = hparams.ssm_dt_rank;
    const int64_t S_k = hparams.ssm_d_state;
    const int64_t S_v = hparams.ssm_d_inner / H_v;
    return GGML_PAD(S_k*H_v + S_v*H_v + 2*H_v, 4);
}

static bool llama_gdn_replay_shape_ok(const llama_hparams & hparams) {
    const int64_t H_v = hparams.ssm_dt_rank;
    const int64_t H_k = hparams.ssm_n_group;
    if (H_v <= 0 || H_k <= 0 || H_v % H_k != 0 || hparams.ssm_d_inner % H_v != 0) {
        return false;
    }
    const int64_t S_v = hparams.ssm_d_inner / H_v;
    return (int64_t) hparams.ssm_d_state == S_v && (int64_t) hparams.n_embd_s() == S_v*S_v*H_v;
}

// the replay op has to run on the device of each recurrent layer; a device without it keeps the old layout
static bool llama_gdn_replay_dev_ok(ggml_backend_dev_t dev, const llama_hparams & hparams, uint32_t n_ring) {
    if (dev == nullptr) {
        return true;
    }

    const int64_t H_v = hparams.ssm_dt_rank;
    const int64_t H_k = hparams.ssm_n_group;
    const int64_t S_v = hparams.ssm_d_inner / H_v;

    ggml_init_params params = {
        /*.mem_size   =*/ 16*ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx { ggml_init(params) };
    if (!ctx) {
        return false;
    }

    ggml_tensor * q    = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S_v, H_k, 1, 1);
    ggml_tensor * k    = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S_v, H_k, 1, 1);
    ggml_tensor * v    = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S_v, H_v, 1, 1);
    ggml_tensor * g    = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 1, H_v, 1, 1);
    ggml_tensor * b    = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 1, H_v, 1, 1);
    ggml_tensor * s    = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S_v, S_v, H_v, 1);
    ggml_tensor * ring = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_ring*llama_gdn_ring_slot(hparams), 1);
    ggml_tensor * rn   = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);

    ggml_tensor * op = ggml_gated_delta_net_replay(ctx.get(), q, k, v, g, b, s, ring, rn, (int32_t) n_ring);

    return ggml_backend_dev_supports_op(dev, op);
}

// the ring format of state_write_data reuses its s_trans word (0 = old format), so an old reader rejects it
static constexpr uint32_t LLAMA_RS_STATE_FMT_RING = 2;

//
// llama_memory_recurrent
//

llama_memory_recurrent::llama_memory_recurrent(
        const llama_model & model,
                ggml_type   type_r,
                ggml_type   type_s,
                     bool   offload,
                 uint32_t   mem_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
    const layer_filter_cb & filter) : hparams(model.hparams), n_seq_max(n_seq_max) {
    const int32_t n_layer = hparams.n_layer();

    head = 0;
    size = mem_size;
    used = 0;

    this->n_rs_seq = n_rs_seq;
    rs_idx.assign(n_seq_max, 0);

    // [TAG_4C_GDN_REPLAY] decided before any tensor is created, because it sets the s_l row count
    rs_pre_state = llama_rs_pre_state_arch(model.arch);

    replay = n_rs_seq > 0 && llama_gdn_replay_env() && llama_gdn_replay_arch(model.arch) &&
        type_s == GGML_TYPE_F32 && model.split_mode() != LLAMA_SPLIT_MODE_TENSOR && llama_gdn_replay_shape_ok(hparams);
    if (replay) {
        for (int i = 0; i < n_layer && replay; i++) {
            if (filter && !filter(i)) {
                continue;
            }
            // the CPU backend always has the op, so only the layer device is checked (also with offload off)
            if (!llama_gdn_replay_dev_ok(model.dev_layer(i), hparams, n_rs_seq)) {
                LLAMA_LOG_WARN("%s: layer %d: device has no GATED_DELTA_NET_REPLAY, keeping %u recurrent snapshot groups\n",
                        __func__, i, 1 + n_rs_seq);
                replay = false;
            }
        }
    }
    const int64_t ring_slot = replay ? llama_gdn_ring_slot(hparams) : 0;

    cells.clear();
    cells.resize(mem_size);

    // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    // create a context for each buffer type
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                // r and s per layer, plus the separate PLE conv row where the model has one, plus the replay ring
                /*.mem_size   =*/ size_t(((hparams.ple_conv_state() > 0 ? 3u : 2u) + (replay ? 1u : 0u))*n_layer*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map.emplace(buft, ctx);

            return ctx;
        }

        return it->second.get();
    };

    r_l.resize(n_layer);
    s_l.resize(n_layer);
    p_l.resize(n_layer);
    ring_l.resize(n_layer);

    for (int i = 0; i < n_layer; i++) {
        if (filter && !filter(i)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: skipped\n", __func__, i);
            continue;
        }

        const char * dev_name = "CPU";

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (offload) {
            auto * dev = model.dev_layer(i);
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);
        }

        LLAMA_LOG_DEBUG("%s, layer %3d: dev = %s\n", __func__, i, dev_name);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for rs cache");
        }

        const uint32_t n_rows = mem_size * (1 + n_rs_seq);
        ggml_tensor * r = ggml_new_tensor_2d(ctx, type_r, hparams.n_embd_r(), n_rows);
        ggml_tensor * s = ggml_new_tensor_2d(ctx, type_s, hparams.n_embd_s(), replay ? mem_size : n_rows); // [TAG_4C_GDN_REPLAY]
        ggml_format_name(r, "cache_r_l%d", i);
        ggml_format_name(s, "cache_s_l%d", i);
        r_l[i] = r;
        s_l[i] = s;

        // [TAG_4C_GDN_REPLAY] token inputs of up to n_rs_seq tokens after the committed state
        if (replay) {
            ggml_tensor * ring = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rs_seq*ring_slot, mem_size);
            ggml_format_name(ring, "cache_ring_l%d", i);
            ring_l[i] = ring;
        }

        // the PLE history needs its own row: Meta must mirror it while the delta-net conv state next door stays split
        if (hparams.ple_conv_state() > 0 && hparams.is_ple(i)) {
            ggml_tensor * p = ggml_new_tensor_2d(ctx, type_r, hparams.ple_conv_state(), n_rows);
            ggml_format_name(p, "cache_ple_r_l%d", i);
            p_l[i] = p;
        }
    }

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for rs cache");
        }
        ggml_backend_buffer_clear(buf, 0);
        LLAMA_LOG_INFO("%s: %10s RS buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);
        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    {
        const size_t memory_size_r    = size_r_bytes();
        const size_t memory_size_s    = size_s_bytes();
        const size_t memory_size_p    = size_p_bytes();
        const size_t memory_size_ring = size_ring_bytes();

        LLAMA_LOG_INFO("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u seqs %2u rs_seq), R (%s): %7.2f MiB, S (%s): %7.2f MiB, P (%s): %7.2f MiB\n", __func__,
                (float)(memory_size_r + memory_size_s + memory_size_p + memory_size_ring) / (1024.0f * 1024.0f), mem_size, n_layer, n_seq_max, n_rs_seq,
                ggml_type_name(type_r), (float)memory_size_r / (1024.0f * 1024.0f),
                ggml_type_name(type_s), (float)memory_size_s / (1024.0f * 1024.0f),
                ggml_type_name(type_r), (float)memory_size_p / (1024.0f * 1024.0f));

        if (replay) {
            LLAMA_LOG_INFO("%s: [TAG_4C_GDN_REPLAY] GDN replay: 1 committed state per cell, ring of %u tokens (%.2f MiB), GDN_REPLAY=0 keeps %u snapshot groups\n",
                    __func__, n_rs_seq, (float)memory_size_ring / (1024.0f * 1024.0f), 1 + n_rs_seq);
        }
    }
}

void llama_memory_recurrent::clear(bool data) {
    for (int32_t i = 0; i < (int32_t) size; ++i) {
        cells[i].pos = -1;
        cells[i].seq_id.clear();
        cells[i].src = -1;
        cells[i].tail = -1;
        cells[i].n_ring = 0; // [TAG_4C_GDN_REPLAY]
        cells[i].n_rb   = 0;
    }

    head = 0;
    used = 0;

    if (data) {
        for (auto & [_, buf] : ctxs_bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
    }

    std::fill(rs_idx.begin(), rs_idx.end(), 0);
}

bool llama_memory_recurrent::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    uint32_t new_head = size;

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    if ((uint32_t) seq_id >= this->n_seq_max) {
        LLAMA_LOG_ERROR("%s: invalid seq_id (%d) - larger than n_seq_max (%d)\n", __func__, seq_id, this->n_seq_max);
        return false;
    }

    const bool rm_all = p0 == 0 && p1 == std::numeric_limits<llama_pos>::max();
    if (rm_all) {
        set_rs_idx(seq_id, 0);
    }

    // models like Mamba or RWKV can't have a state partially erased at the end
    // of the sequence because their state isn't preserved for previous tokens
    if (seq_id >= (int64_t) size) {
        // could be fatal
        return false;
    }
    if (0 <= seq_id) {
        int32_t & tail_id = cells[seq_id].tail;
        if (tail_id >= 0) {
            auto & cell = cells[tail_id];

            // partial rollback via per-token snapshot index (bounded by n_rs_seq)
            if (0 < p0 && p0 <= cell.pos && p1 > cell.pos) {
                const llama_pos rollback = cell.pos - (p0 - 1);
                // pending rollback is single-use
                const bool pending = rs_idx[seq_id] != 0;
                // [TAG_4C_GDN_REPLAY] only the last n_rb tokens have a ring entry and a valid conv group
                // [TAG_RS_SNAP_DEPTH] the snapshot layout has valid groups for the last n_rb tokens too
                const bool in_ring = rollback <= (llama_pos) cell.n_rb;
                if (!pending && in_ring && rollback >= 1 && rollback <= (llama_pos) n_rs_seq) {
                    set_rs_idx(seq_id, (uint32_t) rollback);
                    cell.pos = p0 - 1;
                    return true;
                }
                return false;
            }
            // invalidate tails which will be cleared
            if (p0 <= cell.pos && cell.pos < p1) {
                tail_id = -1;
            }
        }
    } else {
        // seq_id is negative, then the range should include everything or nothing
        if (p0 != p1 && (p0 != 0 || p1 != std::numeric_limits<llama_pos>::max())) {
            //printf("[DEBUG] inside `llama_memory_recurrent::seq_rm`: `seq_id` is negative, so returning false\n");
            return false;
        }
    }

    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].pos >= p0 && cells[i].pos < p1) {
            if (seq_id < 0) {
                cells[i].seq_id.clear();
            } else if (cells[i].has_seq_id(seq_id)) {
                cells[i].seq_id.erase(seq_id);
            } else {
                continue;
            }
            if (cells[i].is_empty()) {
                // keep count of the number of used cells
                if (cells[i].pos >= 0) {
                    used--;
                }
                cells[i].pos = -1;
                cells[i].src = -1;
                cells[i].n_ring = 0; // [TAG_4C_GDN_REPLAY]
                cells[i].n_rb   = 0;
                if (new_head == size) {
                    new_head = i;
                }
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != size && new_head < head) {
        head = new_head;
    }

    return true;
}

void llama_memory_recurrent::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    if (seq_id_src == seq_id_dst) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    if ((uint32_t) seq_id_dst < size && (uint32_t) seq_id_src < size) {
        auto & tail_src = cells[seq_id_src];
        auto & tail_dst = cells[seq_id_dst];
        if (tail_dst.tail >= 0) {
            // clear destination seq_id if it wasn't empty
            auto & cell_dst = cells[tail_dst.tail];

            cell_dst.seq_id.erase(seq_id_dst);
            tail_dst.tail = -1;
            if (cell_dst.seq_id.empty()) {
                cell_dst.pos = -1;
                cell_dst.src = -1;
                cell_dst.n_ring = 0; // [TAG_4C_GDN_REPLAY]
                cell_dst.n_rb   = 0;
                used -= 1;
            }
        }
        if (tail_src.tail >= 0) {
            auto & cell_src = cells[tail_src.tail];

            cell_src.seq_id.insert(seq_id_dst);
            tail_dst.tail = tail_src.tail;
        }
    }
}

void llama_memory_recurrent::seq_keep(llama_seq_id seq_id) {
    uint32_t new_head = size;

    for (uint32_t i = 0; i < size; ++i) {
        if ((llama_seq_id) i != seq_id) {
            cells[i].tail = -1;
        }

        if (!cells[i].has_seq_id(seq_id)) {
            if (cells[i].pos >= 0) {
                used--;
            }

            cells[i].pos = -1;
            cells[i].src = -1;
            cells[i].n_ring = 0; // [TAG_4C_GDN_REPLAY]
            cells[i].n_rb   = 0;
            cells[i].seq_id.clear();

            if (new_head == size){
                new_head = i;
            }
        } else {
            cells[i].seq_id.clear();
            cells[i].seq_id.insert(seq_id);
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != size && new_head < head) {
        head = new_head;
    }
}

void llama_memory_recurrent::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    if (shift == 0) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the
    if (p0 == p1) {
        return;
    }

    // for Mamba-like or RWKV models, only the pos needs to be shifted
    if (0 <= seq_id && seq_id < (int64_t) size) {
        const int32_t tail_id = cells[seq_id].tail;
        if (tail_id >= 0) {
            auto & cell = cells[tail_id];
            if (cell.has_seq_id(seq_id) && p0 <= cell.pos && cell.pos < p1) {
                cell.pos += shift;
            }
        }
    }
}

void llama_memory_recurrent::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    if (d == 1) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the cache.
    if (p0 == p1) {
        return;
    }

    // for Mamba-like or RWKV models, only the pos needs to be changed
    if (0 <= seq_id && seq_id < (int64_t) size) {
        const int32_t tail_id = cells[seq_id].tail;
        if (tail_id >= 0) {
            auto & cell = cells[tail_id];
            if (cell.has_seq_id(seq_id) && p0 <= cell.pos && cell.pos < p1) {
                cell.pos /= d;
            }
        }
    }
}

llama_pos llama_memory_recurrent::seq_pos_min(llama_seq_id seq_id) const {
    llama_pos result = std::numeric_limits<llama_pos>::max();

    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].has_seq_id(seq_id)) {
            result = std::min(result, cells[i].pos);
        }
    }

    if (result == std::numeric_limits<llama_pos>::max()) {
        result = -1;
    }

    return result;
}

llama_pos llama_memory_recurrent::seq_pos_max(llama_seq_id seq_id) const {
    llama_pos result = -1;

    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].has_seq_id(seq_id)) {
            result = std::max(result, cells[i].pos);
        }
    }

    return result;
}

void llama_memory_recurrent::set_rs_idx(llama_seq_id seq_id, uint32_t idx) {
    if (seq_id < 0) {
        std::fill(rs_idx.begin(), rs_idx.end(), 0);
        return;
    }

    assert(n_seq_max == rs_idx.size());

    GGML_ASSERT((uint32_t) seq_id < n_seq_max);
    GGML_ASSERT(idx <= n_rs_seq);

    rs_idx[seq_id] = idx;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_recurrent::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [_, buf] : ctxs_bufs) {
        ret[ggml_backend_buffer_get_type(buf.get())] += ggml_backend_buffer_get_size(buf.get());
    }
    return ret;
}

llama_memory_context_ptr llama_memory_recurrent::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // TODO: non-sequential equal split can be done if using unified KV cache
                //       for simplicity, we always use sequential equal split for now
                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                ubatch = balloc.split_equal(n_ubatch, true, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
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

        if (!prepare(ubatches)) {
            break;
        }

        return std::make_unique<llama_memory_recurrent_context>(this, std::move(ubatches));
    } while (false);

    return std::make_unique<llama_memory_recurrent_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_recurrent::init_full() {
    return std::make_unique<llama_memory_recurrent_context>(this);
}

llama_memory_context_ptr llama_memory_recurrent::init_update(llama_context * lctx, bool optimize) {
    GGML_UNUSED(lctx);
    GGML_UNUSED(optimize);

    return std::make_unique<llama_memory_recurrent_context>(LLAMA_MEMORY_STATUS_NO_UPDATE);
}

bool llama_memory_recurrent::prepare(const std::vector<llama_ubatch> & ubatches) {
    // simply remember the full state because it is very small for this type of cache
    // TODO: optimize
    auto org_cells = cells;
    auto org_used = used;
    auto org_head = head;

    bool success = true;

    for (const auto & ubatch : ubatches) {
        if (!find_slot(ubatch)) {
            success = false;
            break;
        }
    }

    // restore the original state
    cells = std::move(org_cells);
    used = org_used;
    head = org_head;

    return success;
}

bool llama_memory_recurrent::find_slot(const llama_ubatch & ubatch) {
    const uint32_t n_seq_tokens = ubatch.n_seq_tokens;
    const uint32_t n_seqs       = ubatch.n_seqs;

    // if we have enough unused cells before the current head ->
    //   better to start searching from the beginning of the cache, hoping to fill it
    if (head > used + 2*n_seqs) {
        head = 0;
    }

    // For recurrent state architectures (like Mamba or RWKV),
    // each cache cell can store the state for a whole sequence.
    // A slot should be always be contiguous.

    // can only process batches with an equal number of new tokens in each sequence
    GGML_ASSERT(ubatch.equal_seqs());

    int32_t min = size - 1;
    int32_t max = 0;

    // everything should fit if all seq_ids are smaller than the max
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t i = s*n_seq_tokens; // first token of sequence set s
        const uint32_t n_seq_id = ubatch.n_seq_id[i];

        for (uint32_t j = 0; j < n_seq_id; ++j) {
            const llama_seq_id seq_id = ubatch.seq_id[i][j];

            if (seq_id < 0 || (uint32_t) seq_id >= size) {
                // too big seq_id
                // TODO: would it be possible to resize the cache instead?
                LLAMA_LOG_ERROR("%s: seq_id=%d >= n_seq_max=%u Try using a bigger --parallel value\n", __func__, seq_id, n_seq_max);
                return false;
            }
            if (j > 0) {
                auto & seq = cells[seq_id];
                if (seq.tail >= 0) {
                    auto & cell = cells[seq.tail];
                    // clear cells from seq_ids that become shared
                    // (should not normally happen, but let's handle it anyway)
                    cell.seq_id.erase(seq_id);
                    seq.tail = -1;
                    if (cell.seq_id.empty()) {
                        cell.pos = -1;
                        cell.src = -1;
                        cell.n_ring = 0; // [TAG_4C_GDN_REPLAY]
                        cell.n_rb   = 0;
                        used -= 1;
                    }
                }
            }
        }
    }

#ifndef NDEBUG
    {
        std::vector<int32_t> tails_verif;
        tails_verif.assign(size, -1);
        for (uint32_t i = 0; i < size; ++i) {
            auto & cell = cells[i];
            for (llama_seq_id seq_id : cell.seq_id) {
                if (tails_verif[seq_id] != -1) {
                    LLAMA_LOG_ERROR("%s: duplicate tail for seq_id %d in cell %d and %d\n", __func__, seq_id, i, tails_verif[seq_id]);
                }
                tails_verif[seq_id] = i;
            }
        }
        for (uint32_t i = 0; i < size; ++i) {
            if (tails_verif[i] != cells[i].tail) {
                LLAMA_LOG_ERROR("%s: wrong tail for seq_id %d, (%d instead of %d)\n", __func__, i, cells[i].tail, tails_verif[i]);
            }
        }
    }
#endif

    // find next empty cell
    uint32_t next_empty_cell = head;

    for (uint32_t i = 0; i < size; ++i) {
        if (next_empty_cell >= size) { next_empty_cell -= size; }
        auto & cell = cells[next_empty_cell];
        if (cell.is_empty()) { break; }
        next_empty_cell += 1;
    }

    // find usable cell range
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t i = s*n_seq_tokens;
        const llama_seq_id seq_id = ubatch.seq_id[i][0];
        auto & seq_meta = cells[seq_id];
        bool has_cell = false;
        if (seq_meta.tail >= 0) {
            auto & cell = cells[seq_meta.tail];
            GGML_ASSERT(cell.has_seq_id(seq_id));
            // does this seq_id "own" the cell?
            if (cell.seq_id.size() == 1) { has_cell = true; }
        }
        if (!has_cell) {
            auto & empty_cell = cells[next_empty_cell];
            GGML_ASSERT(empty_cell.is_empty());
            // copy old tail into the empty cell
            if (seq_meta.tail >= 0) {
                auto & orig_cell = cells[seq_meta.tail];
                empty_cell.pos = orig_cell.pos;
                empty_cell.src = orig_cell.src;
                empty_cell.n_ring = orig_cell.n_ring; // [TAG_4C_GDN_REPLAY] the ring is copied with the state
                empty_cell.n_rb   = orig_cell.n_rb;
                orig_cell.seq_id.erase(seq_id);
                empty_cell.seq_id.insert(seq_id); // will be overwritten
                GGML_ASSERT(!orig_cell.is_empty()); // has at least one remaining seq_id
            }
            seq_meta.tail = next_empty_cell;
            // find next empty cell
            if (s + 1 < n_seqs) {
                for (uint32_t j = 0; j < size; ++j) {
                    next_empty_cell += 1;
                    if (next_empty_cell >= size) { next_empty_cell -= size; }
                    auto & cell = cells[next_empty_cell];
                    if (cell.is_empty()) { break; }
                }
            }
        }
        if (min > seq_meta.tail) { min = seq_meta.tail; }
        if (max < seq_meta.tail) { max = seq_meta.tail; }
    }

    // gather and re-order
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t i = s*n_seq_tokens;
        const int32_t dst_id = s + min;
        const int32_t src_id = cells[ubatch.seq_id[i][0]].tail;
        if (dst_id != src_id) {
            auto & dst_cell = cells[dst_id];
            auto & src_cell = cells[src_id];

            std::swap(dst_cell.pos, src_cell.pos);
            std::swap(dst_cell.src, src_cell.src);
            std::swap(dst_cell.seq_id, src_cell.seq_id);
            std::swap(dst_cell.n_ring, src_cell.n_ring); // [TAG_4C_GDN_REPLAY] moves with the data like src
            std::swap(dst_cell.n_rb,   src_cell.n_rb);

            // swap tails
            for (uint32_t j = 0; j < size; ++j) {
                int32_t & tail = cells[j].tail;
                if (tail == src_id) {
                    tail = dst_id;
                } else if (tail == dst_id) {
                    tail = src_id;
                }
            }
        }
    }

    // update the pos of the used seqs
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t i = s*n_seq_tokens;
        const llama_pos last_pos = ubatch.pos[i + n_seq_tokens - 1];
        const int32_t cell_id = s + min;
        auto & cell = cells[cell_id];

        if (cell.pos >= 0 && last_pos != cell.pos + (llama_pos) n_seq_tokens) {
            // What should happen when the pos backtracks or skips a value?
            // Clearing the state mid-batch would require special-casing which isn't done.
            LLAMA_LOG_WARN("%s: non-consecutive token position %d after %d for sequence %d with %u new tokens\n",
                __func__, last_pos, cell.pos, ubatch.seq_id[i][0], n_seq_tokens);
        }
        cell.pos = last_pos;
        cell.seq_id.clear();
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            const llama_seq_id seq_id = ubatch.seq_id[i][j];
            cell.seq_id.insert(seq_id);
            cells[seq_id].tail = cell_id;
        }
    }

    // Find first cell without src refs, to use as the zero-ed state
    {
        // TODO: bake-in src refcounts in the cell metadata
        std::vector<int32_t> refcounts(size, 0);
        for (size_t i = 0; i < size; ++i) {
            const int32_t src = cells[i].src;
            if (src >= 0) {
                refcounts[src] += 1;
            }
        }

        rs_z = -1;
        for (int i = min; i <= max; ++i) {
            if (refcounts[i] == 0) {
                rs_z = i;
                break;
            }
        }

        for (int i = min; i <= max; ++i) {
            if (cells[i].src < 0) {
                GGML_ASSERT(rs_z >= 0);
                cells[i].src0 = rs_z;
                cells[i].n_ring = 0; // [TAG_4C_GDN_REPLAY] a zeroed state has no ring tokens
                cells[i].n_rb   = 0;
            } else {
                // Stage the source ids for all used cells to allow correct seq_* behavior
                // and still make these values available when setting the inputs
                cells[i].src0 = cells[i].src;
            }
            cells[i].src = i; // avoid moving or clearing twice
        }
    }

    // [TAG_XSEQ_PLANES] the gather above can move a cell that is not part of this ubatch (an "extra") to another row.
    //   The graph copies only its main row (group 0). Its rollback snapshots (groups 1..n_rs_seq) stay in the old
    //   row, which this ubatch then overwrites with its own snapshots. That is harmless unless the moved sequence
    //   still needs them: a sequence whose verify tokens went into an EARLIER ubatch of the same batch (split_equal
    //   emits it alone when the draft sizes differ, or defers others for n_keep_tail) gets moved here, and its
    //   rollback after the verify then reads another sequence's snapshot. Count such moves so the graph also moves
    //   the snapshot groups of the extra cells.
    rs_n_mv = 0;
    if (n_rs_seq > 0 && llama_xseq_fix_enabled()) {
        for (int i = min + (int) n_seqs; i <= max; ++i) {
            if (!cells[i].seq_id.empty() && cells[i].src0 >= 0 && cells[i].src0 != i) {
                rs_n_mv++;
            }
        }
    }

    // [TAG_4C_GDN_REPLAY] s_copy updates the ring counts when it consumes rs_idx: the first n_seqs cells then keep the
    //   last n_w tokens of this ubatch, the extras keep their ring minus a pending rollback
    rs_n_main = n_seqs;
    rs_n_w    = std::min(n_seq_tokens, n_rs_seq);
    rs_n_tok  = n_seq_tokens;

    // allow getting the range of used cells, from head to head + n
    head = min;
    n    = max - min + 1;
    used = std::count_if(cells.begin(), cells.end(),
        [](const mem_cell & cell){ return !cell.is_empty(); });

    // sanity check
    return n >= n_seqs;
}

bool llama_memory_recurrent::get_can_shift() const {
    // shifting the pos is trivial for recurrent models
    return true;
}

size_t llama_memory_recurrent::total_size() const {
    size_t size = 0;
    for (const auto & [_, buf] : ctxs_bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

size_t llama_memory_recurrent::size_r_bytes() const {
    size_t size_r_bytes = 0;

    for (const auto & r : r_l) {
        if (r != nullptr) {
            size_r_bytes += ggml_nbytes(r);
        }
    }

    return size_r_bytes;
}

size_t llama_memory_recurrent::size_s_bytes() const {
    size_t size_s_bytes = 0;

    for (const auto & s : s_l) {
        if (s != nullptr) {
            size_s_bytes += ggml_nbytes(s);
        }
    }

    return size_s_bytes;
}

size_t llama_memory_recurrent::size_p_bytes() const {
    size_t size_p_bytes = 0;

    for (const auto & p : p_l) {
        if (p != nullptr) {
            size_p_bytes += ggml_nbytes(p);
        }
    }

    return size_p_bytes;
}

size_t llama_memory_recurrent::size_ring_bytes() const {
    size_t size_ring_bytes = 0;

    for (const auto & ring : ring_l) {
        if (ring != nullptr) {
            size_ring_bytes += ggml_nbytes(ring);
        }
    }

    return size_ring_bytes;
}

void llama_memory_recurrent::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    GGML_UNUSED(flags);

    std::vector<std::pair<uint32_t, uint32_t>> cell_ranges; // ranges, from inclusive, to exclusive
    std::vector<std::pair<uint32_t, uint32_t>> cell_ranges_data; // logical source row ranges
    std::vector<std::pair<uint32_t, uint32_t>> cell_ranges_data0; // [TAG_4C_GDN_REPLAY] group-0 rows of C and the ring
    std::vector<uint32_t> ring_live;                              // [TAG_4C_GDN_REPLAY] live ring tokens per cell
    uint32_t cell_count = 0;

    // Count the number of cells with the specified seq_id
    // Find all the ranges of cells with this seq id (or all, when -1)
    uint32_t cell_range_begin = size;
    for (uint32_t i = 0; i < size; ++i) {
        const auto & cell = cells[i];
        // TODO: fix incosistent handling of `seq_id < 0` and `seq_id == -1` in the codebase [TAG_LLAMA_SEQ_ID_NEG]
        if ((seq_id == -1 && !cell.is_empty()) || cell.has_seq_id(seq_id)) {
            ++cell_count;
            uint32_t rs_idx_cur = 0;

            if (n_rs_seq != 0) {
                if (seq_id != -1) {
                    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < rs_idx.size());
                    rs_idx_cur = rs_idx[seq_id];
                } else {
                    bool has_rs_idx = false;
                    for (const llama_seq_id cell_seq_id : cell.seq_id) {
                        GGML_ASSERT(cell_seq_id >= 0 && (size_t) cell_seq_id < rs_idx.size());

                        const uint32_t seq_rs_idx = rs_idx[cell_seq_id];
                        if (!has_rs_idx) {
                            rs_idx_cur = seq_rs_idx;
                            has_rs_idx = true;
                        } else if (rs_idx_cur != seq_rs_idx) {
                            GGML_ABORT("cannot write shared recurrent state with different rollback indices");
                        }
                    }
                }
            }

            const uint32_t cell_id = rs_idx_cur * size + (cell.src >= 0 ? cell.src : (int32_t) i);
            if (cell_ranges_data.empty() || cell_ranges_data.back().second != cell_id) {
                cell_ranges_data.emplace_back(cell_id, cell_id + 1);
            } else {
                cell_ranges_data.back().second++;
            }

            // [TAG_4C_GDN_REPLAY] the conv rows above include the pending rollback; C and the ring are group 0, and the
            //   rollback shortens the live ring instead
            if (replay) {
                const uint32_t cell_id0 = cell.src >= 0 ? cell.src : (int32_t) i;
                if (cell_ranges_data0.empty() || cell_ranges_data0.back().second != cell_id0) {
                    cell_ranges_data0.emplace_back(cell_id0, cell_id0 + 1);
                } else {
                    cell_ranges_data0.back().second++;
                }
                ring_live.push_back(cell.n_ring >= rs_idx_cur ? cell.n_ring - rs_idx_cur : 0);
            }

            if (cell_range_begin == size) {
                cell_range_begin = i;
            }
        } else {
            if (cell_range_begin != size) {
                cell_ranges.emplace_back(cell_range_begin, i);
                cell_range_begin = size;
            }
        }
    }
    if (cell_range_begin != size) {
        cell_ranges.emplace_back(cell_range_begin, size);
    }

    if ((flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) && cell_ranges.size() > 1) {
        GGML_ABORT("cannot save/load multiple ranges of cells to/from device memory\n");
    }

    // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
    uint32_t cell_count_check = 0;
    for (const auto & range : cell_ranges) {
        cell_count_check += range.second - range.first;
    }
    GGML_ASSERT(cell_count == cell_count_check);

    cell_count_check = 0;
    for (const auto & range : cell_ranges_data) {
        cell_count_check += range.second - range.first;
    }
    GGML_ASSERT(cell_count == cell_count_check);

    io.write(&cell_count, sizeof(cell_count));

    state_write_meta(io, cell_ranges, seq_id);
    state_write_data(io, cell_ranges_data, cell_ranges_data0, ring_live);
}

void llama_memory_recurrent::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(flags);

    uint32_t cell_count;
    io.read(&cell_count, sizeof(cell_count));

    bool res = true;

    res = res && state_read_meta(io, cell_count, seq_id);

    try {
        res = res && state_read_data(io, cell_count);
    } catch (...) {
        res = false;
    }

    if (!res) {
        // TODO: fix incosistent handling of `seq_id < 0` and `seq_id == -1` in the codebase [TAG_LLAMA_SEQ_ID_NEG]
        if (seq_id == -1) {
            clear(true);
        } else {
            seq_rm(seq_id, -1, -1);
        }
        throw std::runtime_error("failed to restore kv cache");
    }

    if (n_rs_seq != 0) {
        set_rs_idx(seq_id, 0);
    }
}

void llama_memory_recurrent::state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id) const {
    for (const auto & range : cell_ranges) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            const auto & cell = cells[i];
            const llama_pos pos      = cell.pos;
            const uint32_t  n_seq_id = seq_id == -1 ? cell.seq_id.size() : 0;

            io.write(&pos,      sizeof(pos));
            io.write(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id) {
                for (auto seq_id : cell.seq_id) {
                    io.write(&seq_id, sizeof(seq_id));
                }
            }
        }
    }
}

void llama_memory_recurrent::state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges,
        const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges0, const std::vector<uint32_t> & ring_live) const {
    const uint32_t s_trans = 0;
    const uint32_t n_layer = hparams.n_layer();

    // [TAG_4C_GDN_REPLAY] the ring format takes the s_trans word, so a reader without replay rejects it
    const uint32_t s_fmt = replay ? LLAMA_RS_STATE_FMT_RING : s_trans;

    io.write(&s_fmt,   sizeof(s_fmt));
    io.write(&n_layer, sizeof(n_layer));

    if (replay) {
        const uint32_t n_ring = n_rs_seq;
        io.write(&n_ring, sizeof(n_ring));
        if (!ring_live.empty()) {
            io.write(ring_live.data(), ring_live.size()*sizeof(uint32_t));
        }
    }

    // Iterate and write all the R tensors first, each row is a cell
    // Get whole range at a time
    for (uint32_t il = 0; il < n_layer; ++il) {
        // skip null layers (read_data will handle this by checking "r_l" and "s_l" for null)
        if (r_l[il] == nullptr) continue;

        // Write R tensor type
        const int32_t r_type_i = (int32_t)r_l[il]->type;
        io.write(&r_type_i, sizeof(r_type_i));

        // Write row size of R tensor
        const uint64_t r_size_row = ggml_row_size(r_l[il]->type, hparams.n_embd_r());
        io.write(&r_size_row, sizeof(r_size_row));

        // Write each logical cell row range. With pending recurrent rollback,
        // the logical current state may live in a rollback snapshot plane.
        for (const auto & range : cell_ranges) {
            const size_t range_size = range.second - range.first;
            const size_t buf_size = range_size * r_size_row;
            io.write_tensor(r_l[il], range.first * r_size_row, buf_size);
        }

        // the PLE conv history is a second recurrent row, so it has to travel with the first
        if (p_l[il] != nullptr) {
            const uint64_t p_size_row = ggml_row_size(p_l[il]->type, hparams.ple_conv_state());
            io.write(&p_size_row, sizeof(p_size_row));

            for (const auto & range : cell_ranges) {
                const size_t range_size = range.second - range.first;
                io.write_tensor(p_l[il], range.first * p_size_row, range_size * p_size_row);
            }
        }
    }

    if (!s_trans) {
        for (uint32_t il = 0; il < n_layer; ++il) {
            // skip null layers (read_data will handle this by checking "r_l" and "s_l" for null)
            if (s_l[il] == nullptr) continue;

            // Write S tensor type
            const int32_t s_type_i = (int32_t)s_l[il]->type;
            io.write(&s_type_i, sizeof(s_type_i));

            // Write row size of S tensor
            const uint64_t s_size_row = ggml_row_size(s_l[il]->type, hparams.n_embd_s());
            io.write(&s_size_row, sizeof(s_size_row));

            // Write each logical cell row range. With pending recurrent rollback,
            // the logical current state may live in a rollback snapshot plane.
            // [TAG_4C_GDN_REPLAY] with replay it is the committed state in group 0
            for (const auto & range : (replay ? cell_ranges0 : cell_ranges)) {
                const size_t range_size = range.second - range.first;
                const size_t buf_size = range_size * s_size_row;
                io.write_tensor(s_l[il], range.first * s_size_row, buf_size);
            }
        }

        // [TAG_4C_GDN_REPLAY] the rings, after all S rows
        for (uint32_t il = 0; replay && il < n_layer; ++il) {
            if (ring_l[il] == nullptr) continue;

            const int32_t ring_type_i = (int32_t) ring_l[il]->type;
            io.write(&ring_type_i, sizeof(ring_type_i));

            const uint64_t ring_size_row = ggml_row_size(ring_l[il]->type, ring_l[il]->ne[0]);
            io.write(&ring_size_row, sizeof(ring_size_row));

            for (const auto & range : cell_ranges0) {
                const size_t range_size = range.second - range.first;
                io.write_tensor(ring_l[il], range.first * ring_size_row, range_size * ring_size_row);
            }
        }
    } else {
        // When S tensor is transposed, we also need the element size and get the element ranges from each row
        const uint32_t mem_size = size;
        for (uint32_t il = 0; il < n_layer; ++il) {
            // skip null layers (read_data will handle this by checking "r_l" and "s_l" for null)
            if (s_l[il] == nullptr) continue;

            const uint32_t n_embd_s = hparams.n_embd_s();

            // Write S tensor type
            const int32_t s_type_i = (int32_t)s_l[il]->type;
            io.write(&s_type_i, sizeof(s_type_i));

            // Write element size
            const uint32_t s_size_el = ggml_type_size(s_l[il]->type);
            io.write(&s_size_el, sizeof(s_size_el));

            // Write GQA embedding size
            io.write(&n_embd_s, sizeof(n_embd_s));

            // For each row, we get the element values of each logical cell
            for (uint32_t j = 0; j < n_embd_s; ++j) {
                for (const auto & range : cell_ranges) {
                    const size_t range_size = range.second - range.first;
                    const size_t src_offset = (range.first + j * mem_size) * s_size_el;
                    const size_t buf_size = range_size * s_size_el;
                    io.write_tensor(s_l[il], src_offset, buf_size);
                }
            }
        }
    }
}

bool llama_memory_recurrent::state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id) {
    if (dest_seq_id != -1) {
        // single sequence
        seq_rm(dest_seq_id, -1, -1);

        if (cell_count == 0) {
            return true;
        }

        llama_batch_allocr balloc(hparams.n_pos_per_embd());

        llama_ubatch ubatch = balloc.ubatch_reserve(cell_count, 1);

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id != 0) {
                LLAMA_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
                return false;
            }

            ubatch.pos[i] = pos;
        }
        ubatch.n_seq_id[0] = 1;
        ubatch.seq_id[0] = &dest_seq_id;

        if (!find_slot(ubatch)) {
            LLAMA_LOG_ERROR("%s: failed to find available cells in kv cache\n", __func__);
            return false;
        }

        // DEBUG CHECK: kv.head should be our first cell, kv.head + cell_count - 1 should be our last cell (verify seq_id and pos values)
        // Assume that this is one contiguous block of cells
        GGML_ASSERT(head + cell_count <= size);
        GGML_ASSERT(cells[head].pos == ubatch.pos[0]);
        GGML_ASSERT(cells[head + cell_count - 1].pos == ubatch.pos[cell_count - 1]);
        GGML_ASSERT(cells[head].has_seq_id(dest_seq_id));
        GGML_ASSERT(cells[head + cell_count - 1].has_seq_id(dest_seq_id));
    } else {
        // whole KV cache restore

        if (cell_count > size) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        clear(true);

        for (uint32_t i = 0; i < cell_count; ++i) {
            auto & cell = cells[i];

            llama_pos pos;
            uint32_t  n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            cell.pos = pos;

            for (uint32_t j = 0; j < n_seq_id; ++j) {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));

                if (seq_id < 0 || (uint32_t) seq_id >= this->n_seq_max) {
                    LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, seq_id, this->n_seq_max);
                    return false;
                }

                cell.seq_id.insert(seq_id);

                int32_t & tail = cells[seq_id].tail;
                if (tail != -1) {
                    LLAMA_LOG_ERROR("%s: duplicate tail for seq_id %d in cell %d and %d\n", __func__, seq_id, i, tail);
                    return false;
                }
                tail = i;
            }
        }

        head = 0;
        used = cell_count;
    }

    for (uint32_t i = 0; i < cell_count; ++i) {
        uint32_t cell_id = head + i;
        // make sure the recurrent states will keep their restored state
        cells[cell_id].src = cell_id;
    }

    return true;
}

bool llama_memory_recurrent::state_read_data(llama_io_read_i & io, uint32_t cell_count) {
    uint32_t s_trans;
    uint32_t n_layer;
    io.read(&s_trans, sizeof(s_trans));
    io.read(&n_layer, sizeof(n_layer));

    if (n_layer != hparams.n_layer()) {
        LLAMA_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, n_layer, hparams.n_layer());
        return false;
    }
    if (cell_count > size) {
        LLAMA_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, cell_count, size);
        return false;
    }

    // [TAG_4C_GDN_REPLAY] ring format: live ring counts here, the rings after the S rows
    const bool fmt_ring = s_trans == LLAMA_RS_STATE_FMT_RING;
    std::vector<uint32_t> ring_live;
    if (fmt_ring) {
        if (!replay) {
            LLAMA_LOG_ERROR("%s: the state was saved with the GDN replay layout, which this context does not use (GDN_REPLAY=0?)\n", __func__);
            return false;
        }
        uint32_t n_ring;
        io.read(&n_ring, sizeof(n_ring));
        if (n_ring != n_rs_seq) {
            LLAMA_LOG_ERROR("%s: mismatched GDN replay ring (%u tokens instead of %u)\n", __func__, n_ring, n_rs_seq);
            return false;
        }
        ring_live.resize(cell_count);
        if (cell_count) {
            io.read(ring_live.data(), cell_count*sizeof(uint32_t));
        }
        for (const uint32_t n : ring_live) {
            if (n > n_rs_seq) {
                LLAMA_LOG_ERROR("%s: invalid GDN replay ring count %u\n", __func__, n);
                return false;
            }
        }
        s_trans = 0;
    }

    if (false != (bool) s_trans) {
        LLAMA_LOG_ERROR("%s: incompatible s transposition\n", __func__);
        return false;
    }

    // For each layer, read the keys for each cell, one row is one cell, read as one contiguous block
    for (uint32_t il = 0; il < n_layer; ++il) {
        // skip null layers
        if (r_l[il] == nullptr) continue;

        // Read type of key
        int32_t r_type_i_ref;
        io.read(&r_type_i_ref, sizeof(r_type_i_ref));
        const int32_t r_type_i = (int32_t) r_l[il]->type;
        if (r_type_i != r_type_i_ref) {
            LLAMA_LOG_ERROR("%s: mismatched r type (%d != %d, layer %d)\n", __func__, r_type_i, r_type_i_ref, il);
            return false;
        }

        // Read row size of key
        uint64_t r_size_row_ref;
        io.read(&r_size_row_ref, sizeof(r_size_row_ref));
        const size_t r_size_row = ggml_row_size(r_l[il]->type, hparams.n_embd_r());
        if (r_size_row != r_size_row_ref) {
            LLAMA_LOG_ERROR("%s: mismatched r row size (%zu != %zu, layer %d)\n", __func__, r_size_row, (size_t) r_size_row_ref, il);
            return false;
        }

        if (cell_count) {
            // Read and set the keys for the whole cell range
            io.read_tensor(r_l[il], head * r_size_row, cell_count * r_size_row);
        }

        if (p_l[il] != nullptr) {
            uint64_t p_size_row_ref;
            io.read(&p_size_row_ref, sizeof(p_size_row_ref));
            const size_t p_size_row = ggml_row_size(p_l[il]->type, hparams.ple_conv_state());
            if (p_size_row != p_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched ple row size (%zu != %zu, layer %d)\n", __func__, p_size_row, (size_t) p_size_row_ref, il);
                return false;
            }

            if (cell_count) {
                io.read_tensor(p_l[il], head * p_size_row, cell_count * p_size_row);
            }
        }
    }

    if (!s_trans) {
        for (uint32_t il = 0; il < n_layer; ++il) {
            // skip null layers
            if (s_l[il] == nullptr) continue;

            // Read type of value
            int32_t s_type_i_ref;
            io.read(&s_type_i_ref, sizeof(s_type_i_ref));
            const int32_t s_type_i = (int32_t)s_l[il]->type;

            if (s_type_i != s_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched s type (%d != %d, layer %d)\n", __func__, s_type_i, s_type_i_ref, il);
                return false;
            }

            // Read row size of value
            uint64_t s_size_row_ref;
            io.read(&s_size_row_ref, sizeof(s_size_row_ref));
            const size_t s_size_row = ggml_row_size(s_l[il]->type, hparams.n_embd_s());
            if (s_size_row != s_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched s row size (%zu != %zu, layer %d)\n", __func__, s_size_row, (size_t) s_size_row_ref, il);
                return false;
            }

            if (cell_count) {
                // Read and set the values for the whole cell range
                io.read_tensor(s_l[il], head * s_size_row, cell_count * s_size_row);
            }
        }

        // [TAG_4C_GDN_REPLAY] the rings, after all S rows
        for (uint32_t il = 0; fmt_ring && il < n_layer; ++il) {
            if (ring_l[il] == nullptr) continue;

            int32_t ring_type_i_ref;
            io.read(&ring_type_i_ref, sizeof(ring_type_i_ref));
            if ((int32_t) ring_l[il]->type != ring_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched ring type (%d != %d, layer %d)\n", __func__, (int32_t) ring_l[il]->type, ring_type_i_ref, il);
                return false;
            }

            uint64_t ring_size_row_ref;
            io.read(&ring_size_row_ref, sizeof(ring_size_row_ref));
            const size_t ring_size_row = ggml_row_size(ring_l[il]->type, ring_l[il]->ne[0]);
            if (ring_size_row != ring_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched ring row size (%zu != %zu, layer %d)\n", __func__, ring_size_row, (size_t) ring_size_row_ref, il);
                return false;
            }

            if (cell_count) {
                io.read_tensor(ring_l[il], head * ring_size_row, cell_count * ring_size_row);
            }
        }
    } else {
        // For each layer, read the values for each cell (transposed)
        for (uint32_t il = 0; il < n_layer; ++il) {
            // skip null layers
            if (s_l[il] == nullptr) continue;

            const uint32_t n_embd_s = hparams.n_embd_s();

            // Read type of value
            int32_t s_type_i_ref;
            io.read(&s_type_i_ref, sizeof(s_type_i_ref));
            const int32_t s_type_i = (int32_t)s_l[il]->type;
            if (s_type_i != s_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched s type (%d != %d, layer %d)\n", __func__, s_type_i, s_type_i_ref, il);
                return false;
            }

            // Read element size of value
            uint32_t s_size_el_ref;
            io.read(&s_size_el_ref, sizeof(s_size_el_ref));
            const size_t s_size_el = ggml_type_size(s_l[il]->type);
            if (s_size_el != s_size_el_ref) {
                LLAMA_LOG_ERROR("%s: mismatched s element size (%zu != %zu, layer %d)\n", __func__, s_size_el, (size_t) s_size_el_ref, il);
                return false;
            }

            // Read state embedding size
            uint32_t n_embd_s_ref;
            io.read(&n_embd_s_ref, sizeof(n_embd_s_ref));
            if (n_embd_s != n_embd_s_ref) {
                LLAMA_LOG_ERROR("%s: mismatched s embedding size (%u != %u, layer %d)\n", __func__, n_embd_s, n_embd_s_ref, il);
                return false;
            }

            if (cell_count) {
                // For each row in the transposed matrix, read the values for the whole cell range
                for (uint32_t j = 0; j < n_embd_s; ++j) {
                    const size_t dst_offset = (head + j * size) * s_size_el;
                    io.read_tensor(s_l[il], dst_offset, cell_count * s_size_el);
                }
            }
        }
    }

    // [TAG_4C_GDN_REPLAY] old-format data is a materialized state: the committed state with an empty ring. Only conv
    //   group 0 is restored, so no rollback before the next ubatch.
    for (uint32_t i = 0; i < cell_count; ++i) {
        cells[head + i].n_ring = replay && fmt_ring ? ring_live[i] : 0;
        cells[head + i].n_rb   = 0;
    }

    return true;
}

//
// llama_memory_recurrent_context
//

llama_memory_recurrent_context::llama_memory_recurrent_context(llama_memory_status status) : status(status) {}

llama_memory_recurrent_context::llama_memory_recurrent_context(
        llama_memory_recurrent * mem) : status(LLAMA_MEMORY_STATUS_SUCCESS), mem(mem), is_full(true) {
}

llama_memory_recurrent_context::llama_memory_recurrent_context(
        llama_memory_recurrent * mem,
        std::vector<llama_ubatch> ubatches) : status(LLAMA_MEMORY_STATUS_SUCCESS), mem(mem), ubatches(std::move(ubatches)) {}

llama_memory_recurrent_context::~llama_memory_recurrent_context() = default;

bool llama_memory_recurrent_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_memory_recurrent_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    // no ubatches -> this is an update
    if (ubatches.empty()) {
        // recurrent cache never performs updates
        assert(status == LLAMA_MEMORY_STATUS_NO_UPDATE);

        return true;
    }

    mem->find_slot(ubatches[i_next]);

    return true;
}

llama_memory_status llama_memory_recurrent_context::get_status() const {
    return status;
}

const llama_ubatch & llama_memory_recurrent_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_next];
}

uint32_t llama_memory_recurrent_context::get_n_rs() const {
    return is_full ? mem->size : mem->n;
}

uint32_t llama_memory_recurrent_context::get_head() const {
    return is_full ? 0 : mem->head;
}

int32_t llama_memory_recurrent_context::get_rs_z() const {
    return is_full ? 0 : mem->rs_z;
}

uint32_t llama_memory_recurrent_context::get_size() const {
    return mem->size;
}

ggml_tensor * llama_memory_recurrent_context::get_r_l(int32_t il) const {
    return mem->r_l[il];
}

ggml_tensor * llama_memory_recurrent_context::get_s_l(int32_t il) const {
    return mem->s_l[il];
}

ggml_tensor * llama_memory_recurrent_context::get_p_l(int32_t il) const {
    return mem->p_l[il];
}

int32_t llama_memory_recurrent_context::s_copy(int i) const {
    const uint32_t cell_idx = i + mem->head;
    const int32_t  src0     = mem->cells[cell_idx].src0;

    if (mem->n_rs_seq == 0) {
        return src0;
    }

    uint32_t idx = 0;
    if (!mem->cells[cell_idx].seq_id.empty()) {
        const llama_seq_id seq = *mem->cells[cell_idx].seq_id.begin();
        if (seq >= 0 && (size_t) seq < mem->rs_idx.size()) {
            idx = mem->rs_idx[seq];
            // reset rollback idx
            mem->rs_idx[seq] = 0;
        }
    }

    // [TAG_4C_GDN_REPLAY] the consumed rollback shortens the ring: the ubatch cells replay the rest and then keep the
    //   ring of their last tokens, an extra keeps the rest (its C and ring rows move unchanged). An extra that folds a
    //   rollback keeps its conv groups unshifted, so it allows no further rollback until its next ubatch.
    if (mem->replay) {
        auto & cell = mem->cells[cell_idx];
        const bool is_main = (uint32_t) i < mem->rs_n_main;
        cell.n_rpl  = cell.n_ring >= idx ? cell.n_ring - idx : 0;
        cell.n_ring = is_main ? mem->rs_n_w : cell.n_rpl;
        cell.n_rb   = is_main ? mem->rs_n_w : (idx > 0 ? 0 : std::min(cell.n_rb, cell.n_ring));
    } else {
        // [TAG_RS_SNAP_DEPTH] a main cell has groups for the last rs_n_w tokens (without the pre-ubatch state: not the
        //   whole ubatch), an extra that folds a rollback has none
        auto & cell = mem->cells[cell_idx];
        const bool     is_main = (uint32_t) i < mem->rs_n_main;
        const uint32_t n_main  = mem->rs_pre_state ? mem->rs_n_w : std::min(mem->rs_n_w, mem->rs_n_tok > 0 ? mem->rs_n_tok - 1 : 0);
        cell.n_rb = is_main ? n_main : (idx > 0 ? 0 : cell.n_rb);
    }

    return (int32_t)(idx * mem->size) + src0;
}

ggml_tensor * llama_memory_recurrent_context::get_ring_l(int32_t il) const {
    return mem->ring_l[il];
}

bool llama_memory_recurrent_context::get_replay() const {
    return mem->replay;
}

int32_t llama_memory_recurrent_context::s_copy_r(int i) const {
    return mem->cells[i + mem->head].src0;
}

int32_t llama_memory_recurrent_context::ring_n(int i) const {
    return (int32_t) mem->cells[i + mem->head].n_rpl;
}

uint32_t llama_memory_recurrent_context::get_n_mv() const {
    return is_full ? 0 : mem->rs_n_mv;
}

uint32_t llama_memory_recurrent_context::get_n_rs_seq() const {
    return mem->n_rs_seq;
}

int32_t llama_memory_recurrent_context::s_copy_plane(int i, uint32_t plane) const {
    const uint32_t cell_idx = i + mem->head;
    const int32_t  src0     = mem->cells[cell_idx].src0;

    GGML_ASSERT(plane >= 1 && plane <= mem->n_rs_seq);
    GGML_ASSERT(src0 >= 0 && (uint32_t) src0 < mem->size);

    return (int32_t)(plane * mem->size) + src0;
}
