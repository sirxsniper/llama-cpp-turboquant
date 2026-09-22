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
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

struct llama_ubatch;
struct llama_model;

struct llama_turbot_plan {
    std::string                               path;
    std::map<int32_t, ggml_turbot_layer>      layers;      // by model layer il
    uint32_t                                  pool_cells = GGML_TURBOT_POOL_DEFAULT;
    uint32_t                                  cap_cells  = GGML_TURBOT_CAP_DEFAULT;
    uint64_t                                  hash       = 0;
};

// attn_layers: the il values the cache holds (every one needs an L line). Returns false and sets err on refusal.
// [TAG_TURBOT_ANY_PLAN] these overloads parse against a shape of 4 KV heads x 256 on every layer with one stream (the
// Qwen3.8-27B geometry), exactly as before; llama_turbot_plan_parse_shape below takes any shape.
LLAMA_API bool llama_turbot_plan_parse_text(const std::string & text, const std::vector<int32_t> & attn_layers, uint32_t kv_size,
                                            llama_turbot_plan & plan, std::string & err);
LLAMA_API bool llama_turbot_plan_parse_file(const std::string & path, const std::vector<int32_t> & attn_layers, uint32_t kv_size,
                                            llama_turbot_plan & plan, std::string & err);

//
// [TAG_TURBOT_ANY_PLAN] turbot on other models (contract C2). The cache shape, the plan chooser, the automatic plan, the
// plan scope that carries the chosen plan from llama_context into the cache constructor, and the env switches.
//
// Plan precedence (llama_turbot_plan_choose, first match wins):
//   1. --kv-tier-plan or LLAMA_TURBOT_PLAN: a file is used alone (a mismatch fails, it never becomes an automatic plan);
//      "default" is the built-in plan; "auto" is the automatic plan of the shape.
//   2. the sidecar <model>.turbot.plan (llama_turbot_set_sidecar_path), only with a '# verified:' line and a '# model:'
//      fingerprint equal to the model's (llama_turbot_model_fingerprint).
//   3. the built-in plan, when it names exactly the shape's layers with their geometry. Qwen3.8-27B and fine-tunes with
//      the same attention layers stop here, bit-identical to the plan before this change.
//   4. the automatic plan (llama_turbot_plan_auto_text), only with shape.auto_ok (the main context, resolver on), the
//      switch LLAMA_TURBOT_AUTO_PLAN on, and every layer geometry allowed: the validated list
//      (llama_turbot_auto_geom_validated: 256x4, 256x2 as head dim x KV heads), every supported geometry with
//      LLAMA_TURBOT_AUTO_PLAN=all, and the one-run geometries (128x2, 256x1) with LLAMA_TURBOT_AUTO_BUDGET=turbo5p.
//

// one attention layer of the cache: model layer il, head dim (K = V) and KV heads
struct llama_turbot_layer_geom {
    int32_t  il;
    uint16_t head_dim;
    uint8_t  n_head_kv;
};

struct llama_turbot_cache_shape {
    std::vector<llama_turbot_layer_geom> layers;          // the attention layers the turbot cache holds, any order
    uint32_t                             kv_size   = 0;   // cells per stream
    uint32_t                             n_stream  = 1;   // KV streams (1 with --kv-unified or one sequence)
    uint32_t                             n_seq_max = 1;
    bool                                 auto_ok   = false;     // step 4 may run (main context, resolver on)
    const llama_model *                  model     = nullptr;   // sidecar fingerprint; nullptr disables the sidecar
};

enum llama_turbot_plan_kind {
    LLAMA_TURBOT_PLAN_KIND_FILE,
    LLAMA_TURBOT_PLAN_KIND_BUILTIN,
    LLAMA_TURBOT_PLAN_KIND_SIDECAR,
    LLAMA_TURBOT_PLAN_KIND_AUTO,
};

struct llama_turbot_plan_choice {
    llama_turbot_plan_kind kind = LLAMA_TURBOT_PLAN_KIND_BUILTIN;
    std::string            text;   // the plan text (LF or CRLF)
    std::string            name;   // file path, LLAMA_TURBOT_PLAN_BUILTIN_NAME or LLAMA_TURBOT_PLAN_AUTO_NAME
};

// the name of an automatic plan in log lines, errors and llama_turbot_plan::path
#define LLAMA_TURBOT_PLAN_AUTO_NAME       "<automatic>"
// the keyword that selects the automatic plan in --kv-tier-plan / LLAMA_TURBOT_PLAN (use "./auto" for a file of that name)
#define LLAMA_TURBOT_PLAN_KEYWORD_AUTO    "auto"
// the sidecar next to a model file: <model path> LLAMA_TURBOT_SIDECAR_SUFFIX
#define LLAMA_TURBOT_SIDECAR_SUFFIX       ".turbot.plan"

// Which plan a turbot cache of this shape would use (precedence above). Quiet (no log line) and pure except for getenv
// and reading the plan / sidecar files. true: choice holds the text, why is cleared. false: no plan fits, why says why.
LLAMA_API bool llama_turbot_plan_choose(const llama_turbot_cache_shape & shape, llama_turbot_plan_choice & choice, std::string & why);

// The automatic, uncalibrated plan of shape (deterministic integer model, tools/turbot/turbot_plan.py 'auto' prints the
// same text and hash): old widths 4 or 5 per run (K gets the 5s first, lowest run first, the same pattern on every
// layer), young width 7, CAP 16384, POOL = min(n_seq_max*(16384+128), kv_size*n_stream) in whole granules and a multiple
// of 64*n_stream. It takes the largest number of 5-bit runs whose base + young pool bytes fit the bytes of the fallback
// type (turbo5p, turbo5p512 or turbo4 by the rows; the turbo5p rate with LLAMA_TURBOT_AUTO_BUDGET=turbo5p), then shrinks
// POOL when even all-4 does not fit. false and why when POOL would drop below n_seq_max*(1024+128) cells.
LLAMA_API bool llama_turbot_plan_auto_text(const llama_turbot_cache_shape & shape, std::string & text, std::string & why);

// Parse text against shape: every layer of the shape needs an L line with 4+2*nr tokens (nr = runs of its geometry,
// 12 tokens for 4 KV heads x 256), POOL is clamped to kv_size*n_stream. quiet: no warning lines.
LLAMA_API bool llama_turbot_plan_parse_shape(const std::string & text, const std::string & source, const llama_turbot_cache_shape & shape,
                                             llama_turbot_plan & plan, std::string & err, bool quiet = false);

// The plan chosen by llama_context for the caches it builds next on this thread (thread_local, nests). The turbot cache
// constructor parses the current scope's text against its own shape; without a scope it uses the old precedence
// (llama_turbot_plan_get_source) plus the "auto" keyword.
struct LLAMA_API llama_turbot_plan_scope {
    explicit llama_turbot_plan_scope(const llama_turbot_plan_choice & choice);
    ~llama_turbot_plan_scope();

    llama_turbot_plan_scope(const llama_turbot_plan_scope &)             = delete;
    llama_turbot_plan_scope & operator=(const llama_turbot_plan_scope &) = delete;

private:
    llama_turbot_plan_choice         choice;
    const llama_turbot_plan_choice * prev = nullptr;
};

// the innermost live scope of this thread, nullptr when none
LLAMA_API const llama_turbot_plan_choice * llama_turbot_plan_scope_current();

// Env switches, read with getenv on every call (tests toggle them). With LLAMA_TURBOT_ANY=0 every other field is false
// or empty: only 4 KV heads x 256, no automatic plan, no sidecar, no iSWA split, one stream (the behaviour before
// [TAG_TURBOT_ANY_*]).
//   any                  LLAMA_TURBOT_ANY != 0
//   auto_plan            LLAMA_TURBOT_AUTO_PLAN unset, 1 or all (0: no automatic plan, fall back to turbo5p)
//   auto_all             LLAMA_TURBOT_AUTO_PLAN=all: every supported geometry may get an automatic plan
//   auto_budget_turbo5p  LLAMA_TURBOT_AUTO_BUDGET=turbo5p: the automatic plan may use the turbo5p rate (656 B per 1024
//                        values) instead of the fallback type's bytes, and the one-run geometries may get one
//   sidecar              LLAMA_TURBOT_SIDECAR != 0
//   iswa                 LLAMA_TURBOT_ISWA != 0 (llama_context: turbot on the base child of an iSWA cache)
//   multi_stream         LLAMA_TURBOT_MULTI_STREAM != 0 (one tier per KV stream)
//   swa_type             LLAMA_TURBOT_SWA_TYPE, "" when unset (llama_context: type of the SWA child)
struct llama_turbot_switches {
    bool        any                 = true;
    bool        auto_plan           = true;
    bool        auto_all            = false;
    bool        auto_budget_turbo5p = false;
    bool        sidecar             = true;
    bool        iswa                = true;
    bool        multi_stream        = true;
    std::string swa_type;
};

LLAMA_API llama_turbot_switches llama_turbot_read_switches();

// the geometries an automatic plan is validated on without LLAMA_TURBOT_AUTO_PLAN=all: 256x4 and 256x2 (head dim x KV
// heads). A constexpr table.
LLAMA_API bool llama_turbot_auto_geom_validated(int head_dim, int n_head_kv);

// the fallback type whose bytes an automatic plan must fit, by row values: TURBO5P_0 if row % 1024 == 0, TURBO5P512_0 if
// row % 512 == 0, else TURBO4_0 (the turbot -> turbo5p -> turbo4 chain of the resolver)
LLAMA_API ggml_type llama_turbot_budget_type(uint32_t row_elems);

// '# model:' fingerprint of the model and shape, the text after "# model: ":
//   arch=<general.architecture> basename=<general.basename> size=<general.size_label> layers=<il,...> geom=<D>x<H>
// il ascending; geom is one <head dim>x<KV heads> when every layer has it, else one per layer, comma separated. A missing
// metadata key gives an empty value. "" when shape.model is nullptr.
LLAMA_API std::string llama_turbot_model_fingerprint(const llama_turbot_cache_shape & shape);

// the same from the metadata values (pure; tools/turbot/turbot_plan.py 'fingerprint' prints it too)
LLAMA_API std::string llama_turbot_fingerprint_text(const std::string & arch, const std::string & basename, const std::string & size,
                                                    const llama_turbot_cache_shape & shape);

// A sidecar text is usable when it has a '# verified:' line with a stamp and a '# model:' line whose text equals
// fingerprint (whitespace at both ends ignored). Pure. false and why otherwise.
LLAMA_API bool llama_turbot_sidecar_accepts(const std::string & text, const std::string & fingerprint, std::string & why);

// tests only: when non-null and not empty, llama_turbot_model_fingerprint returns this for any non-null shape.model
// (tests/test-turbot.cpp has no model to read metadata from). nullptr clears it.
LLAMA_API void llama_turbot_test_set_model_fingerprint(const char * fingerprint);

// the path stored by llama_turbot_set_sidecar_path() (llama-ext.h), "" when none was set
LLAMA_API std::string llama_turbot_get_sidecar_path();

// the path stored by llama_turbot_set_plan_path() (llama-ext.h), "" when none was set
std::string llama_turbot_get_plan_path();

//
// [TAG_TURBOT_EMBED_PLAN] built-in default plan and plan source
//
// docs/turbot/plans/turbot-default.plan is compiled into libllama through the checked-in, generated
// src/llama-turbot-default-plan.h (docs/turbot/gen_turbot_default_plan.py; src/CMakeLists.txt stops the configure when
// the two drift apart). It was calibrated on Qwen3.8-27B: 16 attention layers il = 3, 7, ..., 63, 4 KV heads x 256.
//

// the keyword that selects the built-in plan in --kv-tier-plan / LLAMA_TURBOT_PLAN (use "./default" for a file of that name)
#define LLAMA_TURBOT_PLAN_KEYWORD_DEFAULT "default"
// the name of the built-in plan in log lines and errors, and its llama_turbot_plan::path
#define LLAMA_TURBOT_PLAN_BUILTIN_NAME    "<built-in default>"

// the built-in plan text: NUL-terminated, LF line endings, never nullptr
LLAMA_API const char * llama_turbot_default_plan_text();
// its plan hash when the cache has at least POOL cells (0x56c3503c949a7749); a smaller cache clamps POOL, which changes it
LLAMA_API uint64_t     llama_turbot_default_plan_hash();

// Pure (no log line, no global state): does the plan text parse and name exactly the attention layers attn_layers?
// Every attention layer needs an L line and every L/Y line must be one of them. kv_size only clamps POOL, so it never
// causes a mismatch. On false, why says what is wrong and names source, e.g. "turbot: plan <built-in default> line 14:
// layer 39 is not an attention layer of this cache"; on true, why is cleared. It does NOT check head geometry, SWA,
// streams, flash attention or devices: the cache constructor (llama-kv-cache.cpp) still refuses those.
LLAMA_API bool llama_turbot_plan_matches(const std::string & text, const std::vector<int32_t> & attn_layers, uint32_t kv_size,
                                         std::string & why, const std::string & source = LLAMA_TURBOT_PLAN_BUILTIN_NAME);

enum llama_turbot_plan_origin {
    LLAMA_TURBOT_PLAN_BUILTIN_AUTO,     // nothing set: the built-in plan, meant to be used only when it matches the model
    LLAMA_TURBOT_PLAN_BUILTIN_FORCED,   // --kv-tier-plan default or LLAMA_TURBOT_PLAN=default
    LLAMA_TURBOT_PLAN_FILE,             // a plan file
    LLAMA_TURBOT_PLAN_AUTO_FORCED,      // [TAG_TURBOT_ANY_PLAN] --kv-tier-plan auto or LLAMA_TURBOT_PLAN=auto
    LLAMA_TURBOT_PLAN_SIDECAR,          // [TAG_TURBOT_ANY_SIDECAR] <model>.turbot.plan (llama_turbot_plan_sidecar_source)
};

struct llama_turbot_plan_source {
    llama_turbot_plan_origin origin = LLAMA_TURBOT_PLAN_BUILTIN_AUTO;
    std::string              path;      // LLAMA_TURBOT_PLAN_FILE, _SIDECAR: the path as given, else ""
    std::string              name;      // the path, LLAMA_TURBOT_PLAN_BUILTIN_NAME or LLAMA_TURBOT_PLAN_AUTO_NAME
    std::string              set_by;    // "--kv-tier-plan", "LLAMA_TURBOT_PLAN", or "" for BUILTIN_AUTO and SIDECAR
};

// Where the plan of a turbot cache comes from, in this order: llama_turbot_set_plan_path() (--kv-tier-plan, or env
// LLAMA_ARG_KV_TIER_PLAN, set by common), then env LLAMA_TURBOT_PLAN, then the built-in plan. An empty value counts as
// unset; the value "default" in either place selects the built-in plan, and [TAG_TURBOT_ANY_PLAN] "auto" the automatic
// plan (origin LLAMA_TURBOT_PLAN_AUTO_FORCED; with LLAMA_TURBOT_ANY=0 "auto" is a file name, as before). The sidecar
// is not a source here: llama_turbot_plan_choose checks it (step 2).
LLAMA_API llama_turbot_plan_source llama_turbot_plan_get_source();

// [TAG_TURBOT_ANY_SIDECAR] origin LLAMA_TURBOT_PLAN_SIDECAR with the path of llama_turbot_set_sidecar_path(); origin
// LLAMA_TURBOT_PLAN_BUILTIN_AUTO (no sidecar) when none is set or LLAMA_TURBOT_SIDECAR=0 / LLAMA_TURBOT_ANY=0.
LLAMA_API llama_turbot_plan_source llama_turbot_plan_sidecar_source();

// The plan text of src: the built-in text, or the file contents (FILE, SIDECAR). false and err ("turbot: cannot open
// plan file ...") when the file cannot be read. AUTO_FORCED has no text without a shape: false, err names
// llama_turbot_plan_auto_text.
LLAMA_API bool llama_turbot_plan_read(const llama_turbot_plan_source & src, std::string & text, std::string & err);

// The turbot cache constructor's plan loader: llama_turbot_plan_get_source + llama_turbot_plan_read + parse, with
// plan.path = src.name. When the built-in plan does not fit, err also says how to pick another plan or KV type.
// [TAG_TURBOT_ANY_PLAN] a shape of 4 KV heads x 256 on every layer, one stream: the shape overload below (exported for
// tests/test-turbot.cpp).
LLAMA_API bool llama_turbot_plan_load(const std::vector<int32_t> & attn_layers, uint32_t kv_size, llama_turbot_plan & plan, std::string & err);

// [TAG_TURBOT_ANY_PLAN] the constructor path for any shape:
//   - a live llama_turbot_plan_scope: its text parsed against shape (plan.path = its name); an error is returned, the
//     llama_context retry without turbot is the safety net
//   - no scope (LLAMA_KV_RESOLVE=0, or a cache built outside llama_context): the old precedence above, plus the "auto"
//     keyword, whose plan is generated from shape (no validated-list or auto_ok check: it was asked for)
// Logs: the built-in plan lines of before; one INFO line for an automatic plan (widths, POOL, CAP, size against the
// fallback type, hash) and LLAMA_TURBOT_AUTO_PLAN_DUMP=<file> writes its text; one INFO line for a sidecar, and one
// (once per path) when a sidecar was set but not used.
LLAMA_API bool llama_turbot_plan_load(const llama_turbot_cache_shape & shape, llama_turbot_plan & plan, std::string & err);

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

    // [TAG_TURBOT_ANY_STREAMS] a cross-stream seq_cp (llama_kv_cache::seq_cp): this tier becomes a copy of src (granule
    // table, slots, stamps, refinement bits, pending fills), which is what the pool-slice copy in llama_kv_cache::update
    // puts in its pool rows. seq_dst takes seq_src's write-row counter and cut; every other sequence of this tier starts
    // over (the destination stream was reset). Both tiers must have the same kv_size and POOL.
    void copy_stream_from(const llama_kv_tier & src, llama_seq_id seq_src, llama_seq_id seq_dst);

    // [TAG_TURBOT_ANY_STREAMS] drop the refinement and pending bits of cells that kvc says are empty (after
    // copy_stream_from, whose source may hold cells the destination did not copy)
    void drop_empty_cells(const llama_kv_cells & kvc);

    // state restore (9.10). counters: destination seq ids with their saved values. Returns the pool row of each
    // restored cell or -1 (refinement dropped). May reclaim or evict slots (9.6 alloc_slot).
    std::vector<int32_t> restore_cells(const std::vector<uint32_t> & cells, const std::vector<uint64_t> & stamps,
                                       const std::vector<uint8_t> & young, const std::vector<std::pair<llama_seq_id, uint64_t>> & counters,
                                       const llama_kv_cells & kvc);
    void abort_restore();                                 // the pool bytes of the last restore_cells could not be read

    uint64_t stamp(uint32_t cell) const;
    uint64_t row_counter(llama_seq_id seq) const;
    uint64_t n_evictions() const;                         // granules with live cells evicted by alloc_slot (tests, DEBUG=1)

    // [TAG_TURBOT_ANY_STREAMS] the cut of seq at the last commit: false when seq had no cells then (tests)
    bool     seq_cut(llama_seq_id seq, int64_t & cut_out) const;

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

//
// [TAG_TURBOT_ANY_STREAMS] the tiers of a turbot cache with n_stream KV streams (llama_kv_cache::turbot_tier): tier s
// owns the cells of stream s and slots [s*n_slot_s, (s+1)*n_slot_s) of the one pool tensor per layer, n_slot_s =
// tiers[0]->n_slots() (POOL_s = POOL / n_stream rows each, same CAP). Graph inputs hold global values:
//   gtab  stream-major over the view streams s0..s1: gtab[j*n_gran + g] = slot + (s0+j)*n_slot_s, or -1
//   young per ubatch row: pool row + s*POOL_s, or -1
//   fill  per entry: (s*n_gran + g, slot + s*n_slot_s, mask_lo, mask_hi), g and slot of tier s
// so a pool row is always slot*64 + (cell & 63) and a base row s*kv_size + cell (the writer's dst is flattened to
// [1024, kv_size*n_stream]). With one stream every offset is 0 and the values are the single tier's, byte for byte.
// strm: slot_info::strm of the ubatch, rows [k*n, (k+1)*n) of the ubatch belong to stream strm[k], n = idxs[k].size().
//

using llama_kv_tier_ptrs = std::vector<std::unique_ptr<llama_kv_tier>>;

// begin_ubatch of every stream of the ubatch on its slice of rows, against cells[strm[k]]
LLAMA_API void llama_turbot_streams_begin(llama_kv_tier_ptrs & tiers, const llama_ubatch & ubatch, const std::vector<llama_seq_id> & strm,
                                          const std::vector<std::vector<uint32_t>> & idxs, const std::vector<llama_kv_cells> & cells);
// commit_ubatch / abort_ubatch of the streams in strm only (each against its own cells)
LLAMA_API void llama_turbot_streams_commit(llama_kv_tier_ptrs & tiers, const std::vector<llama_seq_id> & strm, const std::vector<llama_kv_cells> & cells);
LLAMA_API void llama_turbot_streams_abort (llama_kv_tier_ptrs & tiers, const std::vector<llama_seq_id> & strm);

// fill entries of the streams in strm (their last begin_ubatch)
LLAMA_API int64_t llama_turbot_streams_n_fill(const llama_kv_tier_ptrs & tiers, const std::vector<llama_seq_id> & strm);
// dst: [(s1 - s0 + 1)*n_gran]
LLAMA_API void llama_turbot_streams_gtab (const llama_kv_tier_ptrs & tiers, uint32_t s0, uint32_t s1, int32_t * dst);
// dst: [sum of the young_rows() sizes of strm], in strm order
LLAMA_API void llama_turbot_streams_young(const llama_kv_tier_ptrs & tiers, const std::vector<llama_seq_id> & strm, int32_t * dst);
// dst: [4*llama_turbot_streams_n_fill], in strm order
LLAMA_API void llama_turbot_streams_fill (const llama_kv_tier_ptrs & tiers, const std::vector<llama_seq_id> & strm, int32_t * dst);
