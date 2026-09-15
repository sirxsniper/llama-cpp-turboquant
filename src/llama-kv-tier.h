#pragma once

// [TAG_TURBOT] Host policy of the turbot tiered KV cache: plan file, granule table, young slot allocator, per-sequence
// write-row stamps and the ubatch lifecycle. docs/turbot/SPEC.md sections 9.1 and 9.5-9.10 are the contract.
//
// Terms:
//   granule   64 contiguous cell indices, G = cell >> 6. Placement (find_slot) does not know about granules.
//   slot      64 rows of the young pool. A granule is young iff it owns a slot (gslot[G] >= 0).
//   stamp     per cell, the value of its first sequence's write-row counter when the row was written.
//   cut       per sequence, row_ctr - Y_s at the last successful commit: stamps above it are "wanted young".
//   ref_valid per granule, bit c = the refinement of cell G*64 + c was written (or is queued as a fill).
//   pending   per granule, cells whose refinement must still be center-filled from their base code.
//
// The class is exported so that tests/test-turbot.cpp can drive it against a real llama_kv_cells with a shared llama.

#include "llama.h"
#include "llama-kv-cells.h"

#include "ggml-turbot.h"

#include <bitset>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

struct llama_ubatch;

struct llama_turbot_plan {
    std::string                               path;
    std::map<int32_t, ggml_turbot_layer>      layers;      // by model layer il
    uint32_t                                  pool_cells = GGML_TURBOT_POOL_DEFAULT;
    uint32_t                                  cap_cells  = GGML_TURBOT_CAP_DEFAULT;
    uint64_t                                  hash       = 0;
};

// attn_layers: the il values the cache holds (every one needs an L line). Returns false and sets err on refusal.
LLAMA_API bool llama_turbot_plan_parse_text(const std::string & text, const std::vector<int32_t> & attn_layers, uint32_t kv_size,
                                            llama_turbot_plan & plan, std::string & err);
LLAMA_API bool llama_turbot_plan_parse_file(const std::string & path, const std::vector<int32_t> & attn_layers, uint32_t kv_size,
                                            llama_turbot_plan & plan, std::string & err);

// the path stored by llama_turbot_set_plan_path() (llama-ext.h), "" when none was set
std::string llama_turbot_get_plan_path();

// f(s) for every sequence id in seqs, ascending. A cell carries a handful of sequences at most, so the walk stops at the
// last set bit instead of visiting all LLAMA_MAX_SEQ ids.
template<typename F>
inline void llama_turbot_for_each_seq(const std::bitset<LLAMA_MAX_SEQ> & seqs, F && f) {
    size_t left = seqs.count();
    for (llama_seq_id s = 0; left > 0 && s < (llama_seq_id) LLAMA_MAX_SEQ; ++s) {
        if (seqs.test(s)) {
            f(s);
            --left;
        }
    }
}

class LLAMA_API llama_kv_tier {
public:
    llama_kv_tier(uint32_t kv_size, uint32_t pool_cells, uint32_t cap_cells);

    uint32_t n_granules() const;                          // kv_size / 64
    uint32_t n_slots() const;                             // pool_cells / 64 (0 for POOL 0)
    const std::vector<int32_t> & granule_slots() const;   // [n_granules], slot or -1 (granule table contents)
    bool     cell_young(uint32_t cell) const;             // gslot >= 0 && ref_valid bit && !pending bit

    // before compute (9.6). cells[i]: destination cell of ubatch row i; the ubatch is already applied to kvc
    void begin_ubatch(const llama_ubatch & ubatch, const std::vector<uint32_t> & cells, const llama_kv_cells & kvc);
    const std::vector<int32_t> & young_rows() const;      // [n_rows], pool row or -1
    const std::vector<int32_t> & fill_entries() const;    // 4*n_fill: granule, slot, (int32) mask_lo, (int32) mask_hi

    void commit_ubatch(const llama_kv_cells & kvc);       // after a successful compute
    void abort_ubatch();                                  // after a failed compute, before the failure seq_rm

    // sequence hooks, called by llama_kv_cache for turbot caches (9.7)
    void on_cell_emptied(uint32_t cell);
    void on_seq_tail_removed(llama_seq_id seq, uint64_t min_removed_stamp);
    void on_seq_cp(llama_seq_id src, llama_seq_id dst, bool dst_was_empty);
    void clear();

    // state restore (9.10). counters: destination seq ids with their saved values. Returns the pool row of each
    // restored cell or -1 (refinement dropped). May reclaim or evict slots (9.6 alloc_slot).
    std::vector<int32_t> restore_cells(const std::vector<uint32_t> & cells, const std::vector<uint64_t> & stamps,
                                       const std::vector<uint8_t> & young, const std::vector<std::pair<llama_seq_id, uint64_t>> & counters,
                                       const llama_kv_cells & kvc);
    void abort_restore();                                 // the pool bytes of the last restore_cells could not be read

    uint64_t stamp(uint32_t cell) const;
    uint64_t row_counter(llama_seq_id seq) const;
    uint64_t n_evictions() const;                         // granules with live cells evicted by alloc_slot (tests, DEBUG=1)

private:
    struct victim {
        int32_t  cls;      // 0: every cell of the granule is empty, 1: live cells
        int64_t  margin;   // class 1: max over live (cell, owner seq) of stamp - cut, INT64_MAX if an owner has no cut
        uint64_t touch;    // last_touch of the slot
        int32_t  slot;
    };

    uint32_t kv_size;
    uint32_t pool_cells;
    uint32_t cap_cells;

    uint32_t n_gran;
    uint32_t n_slot;

    std::vector<int32_t>  gslot;           // [n_granules] slot or -1
    std::vector<int32_t>  owner;           // [n_slots] granule or -1
    std::set<int32_t>     free_slots;      // lowest first
    std::vector<uint64_t> last_touch;      // [n_slots]
    uint64_t              touch_serial = 0;
    std::vector<uint64_t> ref_valid;       // [n_granules]
    std::vector<uint64_t> stamps;          // [kv_size]
    std::vector<uint64_t> row_ctr;         // [LLAMA_MAX_SEQ]
    std::vector<int64_t>  cut;             // [LLAMA_MAX_SEQ]
    std::bitset<LLAMA_MAX_SEQ> has_cut;

    std::vector<uint32_t> in_flight;       // granules touched by the current ubatch, first-touch order
    std::vector<uint32_t> in_flight_pos;   // [n_granules] index in in_flight + 1, 0 when not in flight

    std::vector<int32_t>  restored;        // slots allocated by the last restore_cells
    std::vector<uint32_t> restored_cells;  // cells of the last restore_cells (abort_restore)

    std::map<uint32_t, uint64_t> pending;  // granule -> cells that still need a fill

    std::vector<victim>   victims;         // built lazily inside one begin_ubatch or restore_cells call
    bool                  victims_built = false;
    size_t                victims_next  = 0;

    std::vector<int32_t>  young;           // [n_rows] of the last begin_ubatch
    std::vector<int32_t>  fill;            // 4*n_fill of the last begin_ubatch

    uint64_t n_evict = 0;

    // env LLAMA_TURBOT_DEBUG: 1 = one line per commit, 2 = also the full invariant check (9.8)
    int debug = 0;

    // DEBUG=1 counters since the last commit
    uint64_t dbg_reclaimed  = 0;
    uint64_t dbg_evicted    = 0;
    uint64_t dbg_fill_cells = 0;

    // lifetime count of touched granules written old-only because every slot was in flight (logged once per 1000)
    uint64_t dbg_no_slot    = 0;

    uint64_t live_mask(const llama_kv_cells & kvc, uint32_t g) const;

    void    free_slot(int32_t slot, bool to_free_set);
    void    build_victims(const llama_kv_cells & kvc, bool during_restore);
    int32_t alloc_slot(const llama_kv_cells & kvc, bool during_restore);

    void pending_set  (uint32_t g, uint64_t mask);
    void pending_clear(uint32_t g, uint64_t mask);
    uint64_t pending_of(uint32_t g) const;

    void clear_in_flight();

    void check_invariant(const llama_kv_cells & kvc, bool after_begin, const char * where) const;
};
