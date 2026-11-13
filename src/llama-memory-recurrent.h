#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-io.h"
#include "llama-memory.h"

#include <algorithm>
#include <map>
#include <set>
#include <vector>

// [TAG_GDN_NW_CAP] new tokens of a ubatch that stay rollback-able (ring tokens and conv groups): a 2..n_rs_seq-token
// ubatch keeps T - 1 (a speculative step rolls back at most its draft, T - 1 tokens), a longer one n_rs_seq, one token 1.
// A 4-token verify step with n_rs_seq 7 then does the n_rs_seq 3 work. GDN_NW_CAP=0: min(T, n_rs_seq) (old).
uint32_t llama_rs_n_w(uint32_t n_seq_tokens, uint32_t n_rs_seq);

// [TAG_FN_R1_BRIDGE_RB] process-wide: the ring keeps every token of a ubatch (n_w = min(T, n_rs_seq), no [TAG_GDN_NW_CAP]
// T - 1), so a ubatch of up to n_rs_seq tokens can be rolled back whole (the MoE bridge's retry after a failed ubatch).
// Set by the llama_context constructor before any memory or graph exists; GDN_NW_CAP=0 has the same effect.
void llama_rs_set_nw_full(bool full);

// [TAG_NP2PARK_HOSTREAD] host bytes that stand for tensor bytes [off, off + n)
struct llama_rs_host_span {
    size_t          off;
    size_t          n;
    const uint8_t * data;
};

// [TAG_NP2PARK_HOSTREAD] io.write_tensor of tensor bytes [off, off + n), but the parts inside `host` (sorted by off, not
// overlapping) come from their host bytes: the same bytes in the same order as one io.write_tensor
inline void llama_rs_write_tensor_host(llama_io_write_i & io, ggml_tensor * t, size_t off, size_t n,
        const std::vector<llama_rs_host_span> & host) {
    const size_t end = off + n;
    size_t cur = off;
    for (const auto & h : host) {
        const size_t a = std::max(h.off, cur);
        const size_t b = std::min(h.off + h.n, end);
        if (a >= b) {
            continue;
        }
        if (a > cur) {
            io.write_tensor(t, cur, a - cur);
        }
        io.write(h.data + (a - h.off), b - a);
        cur = b;
    }
    if (end > cur) {
        io.write_tensor(t, cur, end - cur);
    }
}

// [TAG_R35_NP2PARK] a whole granule that holds rows of one cell only, and a cell's parking state (r35_alloc builds them)
struct llama_rs_park_gran {
    ggml_tensor * t     = nullptr; // the tensor whose row holds the granule
    size_t        off_t = 0;       // the granule's offset in that tensor
    size_t        off_b = 0;       // and in the buffer
};

struct llama_rs_park_cell {
    std::vector<llama_rs_park_gran> grans;          // the whole granules that hold this cell's rows only
    bool                            parked = false;
    bool                            kept   = false; // the owner's last batch start kept it (a busy sequence's): no swap parks it
    std::vector<uint8_t>            host;           // their bytes while parked (granule k at k * granule)
};

// [TAG_NP2PARK_HOSTREAD] io.write_tensor for a state save: the granules of a parked cell come from its host copy. Row r of
// t belongs to cell r % cells.size() (row = group*size + cell), and only a row of a granule or more holds whole granules
inline void llama_rs_write_tensor_parked(llama_io_write_i & io, ggml_tensor * t, size_t off, size_t n,
        const std::vector<llama_rs_park_cell> & cells, size_t gran) {
    std::vector<llama_rs_host_span> host;
    const size_t row = t->nb[1];
    if (!cells.empty() && gran > 0 && row >= gran) {
        for (size_t r = off/row; r*row < off + n; ++r) {
            const llama_rs_park_cell & pc = cells[r % cells.size()];
            if (!pc.parked) {
                continue;
            }
            for (size_t k = 0; k < pc.grans.size(); ++k) {
                const llama_rs_park_gran & gr = pc.grans[k];
                if (gr.t == t && gr.off_t >= r*row && gr.off_t < (r + 1)*row) {
                    host.push_back({ gr.off_t, gran, pc.host.data() + k*gran });
                }
            }
        }
    }
    llama_rs_write_tensor_host(io, t, off, n, host);
}

// [TAG_NP2PARK_HOSTREAD] the cells of the rows in `ranges` (row = group*size + cell), appended to `cells`
inline void llama_rs_cells_of_rows(std::vector<uint32_t> & cells, const std::vector<std::pair<uint32_t, uint32_t>> & ranges,
        uint32_t size) {
    for (const auto & range : ranges) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            cells.push_back(i % size);
        }
    }
}

//
// llama_memory_recurrent
//

// TODO: extract the cache state used for graph computation into llama_memory_recurrent_context_i
//       see the implementation of llama_kv_cache_context_i for an example how to do it
class llama_memory_recurrent : public llama_memory_i {
public:
    llama_memory_recurrent(
            const llama_model & model,
                         bool   no_alloc,
                    ggml_type   type_r,
                    ggml_type   type_s,
                         bool   offload,
                     uint32_t   mem_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_rs_seq,
        const layer_filter_cb & filter);

    ~llama_memory_recurrent();

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

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    bool prepare(const std::vector<llama_ubatch> & ubatches);

    // find a contiguous slot of memory cells and emplace the ubatch there
    bool find_slot(const llama_ubatch & ubatch);

    bool get_can_shift() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    uint32_t head = 0; // the location where the batch will be placed in the cache (see find_slot())
    uint32_t size = 0; // total number of cells, shared across all sequences
    uint32_t used = 0; // used cells (i.e. at least one seq_id)

    // number of recurrent-state snapshots per seq for rollback; tensors are widened to (1 + n_rs_seq) groups
    uint32_t n_rs_seq = 0;

    // per-seq rollback index
    // [TAG_RS_IDX_SHARE] rs_idx[s] != 0 means: the cell of s still holds the state before the rollback in group 0.
    //   seq_cp gives dst the index of src (0 when src has no cell), seq_rm of the whole sequence and seq_keep reset it when
    //   they drop the tail of s, and s_copy consumes it for every sharer of the cell. find_slot's branch for tokens with
    //   several sequence ids drops the tail of the other ids without a reset; s_copy clears their index in that ubatch,
    //   since they then share the cell of the first id.
    std::vector<uint32_t> rs_idx;

    void set_rs_idx(llama_seq_id seq_id, uint32_t idx);

    // computed before each graph build
    uint32_t n = 0;

    // first zero-ed state
    int32_t rs_z = -1;

    // [TAG_XSEQ_PLANES] number of non-empty extra cells of the current ubatch (inside [head, head + n) but not part
    //   of the ubatch) whose data row was moved by find_slot. The graph moves only their main row, so their
    //   rollback snapshot rows (groups 1..n_rs_seq) must be moved as well, otherwise a pending rollback of that
    //   sequence reads the snapshot another sequence wrote into the new row. 0 when LLAMA_XSEQ_FIX=0.
    uint32_t rs_n_mv = 0;

    // [TAG_4C_GDN_REPLAY] s_l holds one committed state per cell and ring_l the inputs of up to n_rs_seq later tokens
    //   (design note in llama-memory-recurrent.cpp). false: s_l has 1 + n_rs_seq snapshot groups as before.
    bool replay = false;

    // [TAG_4C_GDN_REPLAY] of the current ubatch: its sequence count, and the ring tokens each of them keeps
    uint32_t rs_n_main = 0;
    uint32_t rs_n_w    = 0;

    // [TAG_RS_SNAP_DEPTH] snapshot layout: rollback depth after a ubatch of rs_n_tok tokens. rs_pre_state: the graph also
    //   writes the state before the ubatch (group rs_n_tok), so a rollback of the whole ubatch is exact
    uint32_t rs_n_tok      = 0;
    bool     rs_pre_state  = false;

    // TODO: optimize for recurrent state needs
    struct mem_cell {
        llama_pos pos  = -1;
        int32_t   src  = -1; // used to know where states should be copied from
        int32_t   src0 = -1; // like src, but only used when setting the inputs (allowing to copy once)
        int32_t   tail = -1;

        // [TAG_4C_GDN_REPLAY] ring tokens that follow the committed state of this cell's data, and the part of them
        //   the current ubatch replays (set by s_copy for the graph inputs)
        uint32_t n_ring = 0;
        uint32_t n_rpl  = 0;
        // [TAG_4C_GDN_REPLAY] rollback limit (<= n_ring): tokens whose conv snapshot groups are also valid
        uint32_t n_rb   = 0;

        std::set<llama_seq_id> seq_id;

        bool has_seq_id(const llama_seq_id & id) const {
            return seq_id.find(id) != seq_id.end();
        }

        bool is_empty() const {
            return seq_id.empty();
        }

        bool is_same_seq(const mem_cell & other) const {
            return seq_id == other.seq_id;
        }
    };

    std::vector<mem_cell> cells;

    // per layer
    std::vector<ggml_tensor *> r_l;
    std::vector<ggml_tensor *> s_l;
    // a second conv history that must stay replicated across devices, so it cannot share the r row
    std::vector<ggml_tensor *> p_l;
    // [TAG_4C_GDN_REPLAY] per layer ring of token inputs, F32 [n_rs_seq*slot, mem_size], only when replay
    std::vector<ggml_tensor *> ring_l;

    // [TAG_R35_NP2PARK] a multi-slot qwen4exp context keeps its recurrent rows on CUDA virtual memory: the whole granules
    // that hold one cell's rows only can leave the device (their bytes wait in host RAM, the addresses stay valid on a
    // shared alias granule) while that cell's sequence is idle, and the hot expert set holds the VRAM meanwhile (design
    // note in the .cpp). The owner context drives it; every read or write of a cell's data maps the cell first, except a
    // state save, which reads a parked cell's granules from host RAM ([TAG_NP2PARK_HOSTREAD]).
    struct r35_hooks {
        size_t (*take)(void * user, size_t bytes) = nullptr; // `bytes` more VRAM from the hot set before a map
        void   (*give)(void * user, size_t bytes) = nullptr; // settle the hot set's side to the cells parked now
        void   (*sync)(void * user)               = nullptr; // the owner's graphs are done (before an unmap)
        bool   (*idle)(void * user)               = nullptr; // no other sequence is busy (a kept cell may swap out)
        void *  user                              = nullptr;
    };
    bool     r35_parkable() const { return r35_vmm; }      // the layout is there (switch on, VMM, >= 2 cells)
    bool     r35_active()   const { return r35_on; }       // parking runs (after r35_start)
    size_t   r35_cell_bytes(uint32_t c) const;              // the bytes a parked cell gives back
    size_t   r35_parked_bytes() const;                      // the bytes all parked cells give back now
    uint32_t r35_n_parked() const;
    uint64_t r38_n_moves() const { return r35_n_park + r35_n_unpark; } // [TAG_R38_E3] parks + maps so far
    // the fit: park every cell but `keep` (they are empty there; their VRAM is free for the fit, the hot set lends it), then
    // later parks / maps take and give through `h`. false: nothing parked (the layout cannot)
    bool     r35_start(const r35_hooks & h, uint32_t keep);
    void     r35_stop();                                    // every cell mapped again, parking off
    // before a batch: the cells of `keep_seqs` stay (a sequence without a cell: nothing parks now), every other cell parks
    void     r35_park_except(const std::vector<llama_seq_id> & keep_seqs);
    void     r35_unpark_all();
    // the guard: every cell in `need` mapped before its data is read or written (a parked one maps, VRAM from the hot set);
    // false: a map failed (the cell stays on its alias granule)
    bool     r35_ensure(const std::vector<uint32_t> & need) const;
    // a reserve's full context covers this many cells (0: all): the owner's one-sequence SMALL reserve sets 1 while it
    // reserves (a one-sequence ubatch uses one cell; the n_seq_max-cell reserve gathered an extra cell's states)
    uint32_t r35_full_n = 0;
    // the owner holds one busy sequence (a cell mapped on demand then swaps with a mapped cell nothing needs)
    void     r35_set_one(bool one);

private:
    //const llama_model & model;
    const llama_hparams & hparams;

    const uint32_t n_seq_max = 1;

    // ggml contexts for the RS cache along with the allocated backend buffers:
    llama_memory_bufs ctxs_bufs;

    // true if no layers - can happen if the layer filter removes all layers
    bool is_empty() const;

    size_t total_size() const;

    size_t size_r_bytes() const;
    size_t size_s_bytes() const;
    size_t size_p_bytes() const;
    size_t size_ring_bytes() const;

    void state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id = -1) const;
    // [TAG_4C_GDN_REPLAY] cell_ranges0 (group-0 rows) and ring_live are used only with replay
    void state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges,
            const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges0, const std::vector<uint32_t> & ring_live,
            llama_state_seq_flags flags) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t cell_count);

    void state_clear(llama_seq_id seq_id, uint32_t cell_head, uint32_t cell_count);

    // [TAG_R35_NP2PARK] the parking layout and state (mutable: the const state_write maps a parked cell for an on-device save)
    using r35_gran = llama_rs_park_gran;
    using r35_cell = llama_rs_park_cell;
    ggml_backend_buffer_t        r35_stage = nullptr;      // pinned host staging for the copies (r35_stage_n granules)
    size_t                       r35_stage_n = 0;
    void r35_copy_out(const r35_cell & pc) const;          // the cell's granules -> pc.host (through the staging)
    void r35_copy_in(const r35_cell & pc) const;           // pc.host -> the cell's granules
    bool                         r35_vmm = false;
    mutable bool                 r35_on  = false;
    bool                         r35_one = false;
    ggml_backend_buffer_t        r35_buf = nullptr;
    size_t                       r35_g   = 0;
    bool (*r35_fmap)(ggml_backend_buffer_t, size_t, size_t)   = nullptr;
    bool (*r35_funmap)(ggml_backend_buffer_t, size_t, size_t) = nullptr;
    bool (*r35_falias)(ggml_backend_buffer_t, size_t, size_t) = nullptr;
    mutable std::vector<r35_cell> r35_cells;
    r35_hooks                    r35_h;
    mutable uint64_t             r35_n_park = 0, r35_n_unpark = 0;
    mutable double               r35_park_ms = 0.0, r35_unpark_ms = 0.0;

    ggml_backend_buffer_t r35_alloc(const llama_model & model, ggml_context * ctx, ggml_backend_buffer_type_t buft);
    bool r35_park_cell(uint32_t c, bool give) const;
    bool r35_unpark_cell(uint32_t c, bool take) const;
};

class llama_memory_recurrent_context : public llama_memory_context_i {
public:
    // used for errors
    llama_memory_recurrent_context(llama_memory_status status);

    // used to create a full-cache or update context
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem);

    // used to create a batch processing context from a batch
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_memory_recurrent_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_memory_recurrent_context specific API
    //

    uint32_t get_n_rs() const;
    uint32_t get_head() const;
    int32_t  get_rs_z() const;
    uint32_t get_size() const;

    ggml_tensor * get_r_l(int32_t il) const;
    ggml_tensor * get_s_l(int32_t il) const;
    ggml_tensor * get_p_l(int32_t il) const;

    int32_t s_copy(int i) const;

    // [TAG_XSEQ_PLANES] extra cells whose data row moved in the current ubatch (0 -> no snapshot move needed)
    uint32_t get_n_mv() const;
    uint32_t get_n_rs_seq() const;

    // source row of snapshot group `plane` (1..n_rs_seq) for cell i (relative to head); does not consume rs_idx
    int32_t s_copy_plane(int i, uint32_t plane) const;

    // [TAG_4C_GDN_REPLAY] nullptr when the memory keeps snapshot groups instead of rings
    ggml_tensor * get_ring_l(int32_t il) const;
    bool get_replay() const;

    // source row of the one-group tensors (committed state, ring) for cell i (relative to head); no plane offset
    int32_t s_copy_r(int i) const;

    // [TAG_UPSTREAM_30139] the committed states of the ubatch can be read in place (each cell reads its own row)
    bool get_rs_r_in_place(uint32_t n_seqs) const;

    // ring tokens that cell i (relative to head) replays in the current ubatch
    int32_t ring_n(int i) const;

private:
    const llama_memory_status status;

    llama_memory_recurrent * mem;

    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    // TODO: extract all the state like `head` and `n` here
    //

    const bool is_full = false;
};
