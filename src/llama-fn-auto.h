#pragma once

// [TAG_FN_AUTO] automatic Flash-Next (qwen4exp) defaults behind ONE switch, LLAMA_FLASHNEXT_PROFILE:
//   off        nothing automatic (the switch-by-switch behaviour of flashnext/all)
//   safe       default: the measured winners (k_items, each with its A/B evidence next to it): every trunk layer's
//              routed experts on the host, the adaptive decayed per-expert VRAM hot set sized by the VRAM fit, the RAM
//              fit, and [TAG_FN_SHIP1] lever round 1's set at -c 262144: the host bridge with the DMA share inside it,
//              the CPU expert kernels, host trims, PLE direct I/O from an unmapped copy, the positional QSA mask, sparse
//              turbot FA and unordered top-k, the MTP cost / window / head rows / draft ubatch, the prefill stream with
//              lend (real use +65.9 % at 262K vs the round's start, E:/turbot-gates/flashnext/test/history.md)
//   fast       safe + levers that won only a narrower A/B (k_items, P_FAST). None: every measured lever is in safe
//   trial      safe + every built lever that is NOT measured yet. None today, so trial == safe. For A/B runs only
//   trial-dma  trial without the host bridge: the gen5 DMA share in its own CPU split (the pre-round-4 exclusive mode)
// LLAMA_FLASHNEXT_FAST=<NAME=VALUE,...> replaces the levers of fast / trial / trial-dma that are not in safe (none today)
// and adds or changes options (one lever per A/B arm); safe ignores it.
// Active only for arch qwen4exp whose routed experts the tensor overrides (--n-cpu-moe, --cpu-moe, -ot, -fit) put on
// the host, with the weights loaded (not the -fit memory probes). Every other model is unchanged. A variable that is
// set in the environment always wins over the profile.
//
// The lookup items (LLAMA_MOE_HOT_*, LLAMA_FN_*, LLAMA_RAM_FIT, LLAMA_PLE_DIO_FILE) are read through llama_fn_env() by
// the code that owns them (placement, hot set, VRAM fit, RAM fit, the PLE copy), so they never leave the model. The other
// levers are read by getenv() deep in other code, so while the model is loaded the profile puts them into the process
// environment (GGML_CPU_* through ggml_cpu_fn_set_switch) and takes them out again when the model is freed. Switches that
// latch on first use (a static in the code that reads them) keep their value for the process.

#include "llama.h"
#include "llama-ext.h"
#include "llama-fn-l22-p10.h" // [TAG_FN_L22_P10] llama_fn_l22_cost

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

struct llama_model;
struct llama_model_loader;
struct llama_context;

enum llama_fn_profile {
    LLAMA_FN_PROFILE_OFF       = 0,
    LLAMA_FN_PROFILE_SAFE      = 1,
    LLAMA_FN_PROFILE_FAST      = 2,
    LLAMA_FN_PROFILE_TRIAL     = 3,
    LLAMA_FN_PROFILE_TRIAL_DMA = 4,
};

struct llama_fn_opt {
    std::string name;
    std::string value;
    bool        inject = false; // read by getenv() outside the fn-auto lookup: set in the environment while the model lives
};

struct llama_fn_auto_state {
    bool             active      = false;
    llama_fn_profile profile     = LLAMA_FN_PROFILE_OFF;
    bool             promote     = false; // LLAMA_FN_PLACEMENT=auto: the routed experts of every trunk layer on the host
    int              n_layer     = 0;     // trunk layers
    int              n_host_user = 0;     // trunk layers with routed experts the overrides put on the host
    int              n_host      = 0;     // after load: trunk layers with host experts and a device router
    bool             mtp_loaded  = false; // the MTP block is loaded, so an MTP draft context will follow
    std::string      model_path;

    std::vector<llama_fn_opt> opts;     // the profile's values (explicit environment values are not copied here)
    std::vector<std::string>  injected; // names this state put into the environment
    std::vector<std::string>  cpu_set;  // GGML_CPU_* switches this state set
};

//
// pure pieces (tests/test-fn-auto.cpp)
//

// "off"/"0", "safe"/"1", "fast", "trial", "trial-dma"; nullptr or "" = safe. *ok = false for an unknown name (then safe)
LLAMA_API llama_fn_profile llama_fn_profile_parse(const char * s, bool * ok);
LLAMA_API const char *     llama_fn_profile_name(llama_fn_profile p);

// "NAME=VALUE,NAME=VALUE" (spaces around items allowed); false if an item has no '=' or an empty name
LLAMA_API bool llama_fn_parse_opt_list(const char * s, std::vector<std::pair<std::string, std::string>> & out);

// the options of a profile; fast_list (LLAMA_FLASHNEXT_FAST) replaces the built-in levers of fast / trial / trial-dma
// that are not in safe and adds or changes options ([TAG_FN_SHIP1] every built-in lever is in safe today)
LLAMA_API std::vector<llama_fn_opt> llama_fn_profile_opts(llama_fn_profile p, const char * fast_list);

// the built-in levers of fast (the measured winners) beyond safe; 0 today
LLAMA_API int llama_fn_fast_lever_count();

// trunk layers 0..n_layer-1 with at least one routed expert tensor that the first matching override puts on a host
// buffer type; overrides = (pattern, buft is host), in order
LLAMA_API int llama_fn_count_host_expert_layers(const std::vector<std::pair<std::string, bool>> & overrides, int n_layer);

// the environment value, else the state's profile value (when active), else nullptr
LLAMA_API const char * llama_fn_state_env(const llama_fn_auto_state * st, const char * name);

// put the inject options into the environment (skipping names set there by the user) and set the GGML_CPU_* switches;
// undo takes back only what inject set. Reference counted per name across models.
LLAMA_API void llama_fn_state_inject(llama_fn_auto_state & st);
LLAMA_API void llama_fn_state_undo  (llama_fn_auto_state & st);

// slots per layer when the budget is spread evenly (no routing profile): every layer also holds its zero slot, so
// budget / sum of the layers' bytes per expert - 1, at most n_expert
LLAMA_API int32_t llama_fn_even_slots(const std::vector<size_t> & bytes_per_expert, size_t budget, int32_t n_expert);

// [TAG_FN_L18_CHURN_LAYERS] slots per layer in proportion to the weights (one per layer, > 0): n_l ~ lam * w_l, at most
// n_expert, with every layer's zero slot inside the budget and the rest given out one slot at a time; wrong weights (count,
// value) give the even count everywhere
LLAMA_API std::vector<int32_t> llama_fn_weighted_slots(const std::vector<size_t> & bytes_per_expert, size_t budget, int32_t n_expert,
        const std::vector<float> & weights);
// "w0,w1,..." -> the weights; empty on any bad item (not a number, <= 0)
LLAMA_API std::vector<float> llama_fn_parse_weights(const char * s);

// [TAG_FN_VRAM_FIT] default device-use ceiling: total - max(1536 MiB, total/8), rounded down to 256 MiB;
// 28416 MiB on a 32607 MiB card (the 5090 limit is 28500 MiB)
LLAMA_API size_t llama_fn_vram_ceiling_default(size_t total);
// hot-set budget: min(free - margin, ceiling - (total - free) - margin), 0 when negative
LLAMA_API size_t llama_fn_vram_fit_budget(size_t total, size_t free, size_t ceiling, size_t margin);

// [TAG_FN_L3_VRAM_CBUF] the compute-buffer lend (LLAMA_FN_CBUF):
// the widest graph the SMALL reserve serves: below the op-offload minimum (a graph that wide copies host experts into the
// compute buffer), a multiple of n_seqs (the reserve rounds up to one), at least 1. Below 8 (the hot chain's widest graph)
// the graphs between it and 8 tokens run with the FULL reserve and without the tail's hot chain (correct, a switch each)
LLAMA_API uint32_t llama_fn_cbuf_small_t(uint32_t requested, int op_offload_min, uint32_t n_seqs);
// the tail the hot set keeps: the FULL - SMALL reserve difference plus pool_extra, rounded up to the granularity; 0 when
// FULL is not at least min_gain larger than SMALL
LLAMA_API size_t   llama_fn_cbuf_tail(size_t full, size_t small, size_t pool_extra, size_t gran, size_t min_gain);
// the first layer of the tail: whole layers from the top (layer_bytes in buffer order) until they hold tail_bytes; -1 when
// that needs every layer (one must stay below the tail) or tail_bytes is 0
LLAMA_API int      llama_fn_cbuf_first_tail_layer(const std::vector<size_t> & layer_bytes, size_t tail_bytes);
// the budget holds the tail, the prefill stream's lend range and about one layer (budget / 32) between them, plus slack
LLAMA_API bool     llama_fn_cbuf_fits(size_t budget, size_t tail, size_t stream_lend);
// [TAG_FN_L19_TTFT_EARLY] the widest batch the short SMALL path takes now: short_max (LLAMA_FN_L4_PROMPT), raised to early
// (LLAMA_FN_L19_TTFT_EARLY) for a wider batch of at most early tokens while the experts' warm pass runs
LLAMA_API uint32_t llama_fn_cbuf_short_max(uint32_t short_max, uint32_t early, bool warm_running, uint32_t n_tokens, uint32_t small_t);
// [TAG_FN_L19_TTFT_TAILFULL] the narrow last part of a split prompt keeps FULL: tailfull (LLAMA_FN_L19_TTFT_TAILFULL) > 0, the
// state is FULL, the batch is not on the short path, is no wider than small_t, asks for outputs, and the batch before it
// was wider than small_t, asked for none and had fewer than tailfull tokens
LLAMA_API bool     llama_fn_cbuf_tail_full(uint32_t tailfull, bool state_full, bool short_path, uint32_t n_tokens, uint32_t n_outputs,
        uint32_t prev_n, bool prev_out, uint32_t small_t);
// [TAG_FN_L20_BIGPROMPT_TAIL] the lowest tail offset a release of need bytes from the top of [lo, hi) gives back, with the
// tail mapped in segments of seg bytes from lo (an unmap takes whole segments): the highest segment start a with hi - a >=
// need; lo (the whole tail) for need 0, seg 0, or need beyond the segments above lo
LLAMA_API size_t   llama_fn_cbuf_part_lo(size_t lo, size_t hi, size_t seg, size_t need);
// [TAG_FN_L20_BIGPROMPT_UB] the FULL reserve's width for a wide batch of n_tokens with `after` prompt tokens after it (-1:
// not known): the big ubatch when the batch or the batch plus the rest is wider than the command's ubatch base_ub
LLAMA_API uint32_t llama_fn_l20_full_width(uint32_t base_ub, uint32_t n_ubatch, uint32_t n_tokens, int64_t after);
// [TAG_FN_L21_BIGUB] the same with LLAMA_FN_L21_BIGUB: the big ubatch only when the prompt is wider than it (the batch plus
// the rest above n_ubatch; with no hint, a batch of the full n_ubatch), else the command's width
LLAMA_API uint32_t llama_fn_l21_full_width(uint32_t base_ub, uint32_t n_ubatch, uint32_t n_tokens, int64_t after);
// [TAG_FN_L20_BIGPROMPT_UB] a narrow batch (no wider than the SMALL reserve) keeps the FULL state when the switch is on
// (base_ub > 0), the state is FULL, it is not on the short path and its prompt goes on after it (after > 0)
LLAMA_API bool     llama_fn_l20_keep_full(uint32_t base_ub, bool wide, bool short_path, bool state_full, int64_t after);
// [TAG_FN_V0254_NARROWFULL] a narrow batch of a multi-sequence context starts the FULL period early: it is no wider than
// small_t, has prompt rows (n_outputs < n_tokens), its prompt goes on (after > 0), and the rest's next batch (min(after,
// n_batch) rows; n_batch 0: after) would itself take the FULL path: wider than small_t and not on the short path
// (llama_fn_cbuf_short_max(short_max, early, warm_running) for those rows). A rest that stays SMALL keeps this part SMALL
// too (20 rows + 150 rows with short_max 160: both SMALL, as v0.25.3), so the guard judges the rest batch alone, as the
// short path does, not the whole prompt
LLAMA_API bool     llama_fn_v0254_narrow_full(uint32_t n_tokens, uint32_t n_outputs, int64_t after, uint32_t n_batch,
        uint32_t small_t, uint32_t short_max, uint32_t early, bool warm_running);
// the hot-set budget with the lend: the fit's budget plus pool_extra (LLAMA_FN_CBUF_POOL_MIB, also in the tail), never
// above budget_max nor above free - 256 MiB (unless the fit's budget already is)
LLAMA_API size_t   llama_fn_cbuf_budget(size_t budget, size_t pool_extra, size_t free, size_t budget_max);
// [TAG_FN_L22_LENDCLASS] the FULL reserve class of a prompt below the command's width: the smallest of 1024 / 2048 / 4096
// tokens that holds the whole prompt (the batch plus `after` tokens after it; after < 0: unknown -> 0) and is narrower
// than base_ub; 0 when none (the old width). The ubatches stay the same (the class holds every one of them).
LLAMA_API uint32_t llama_fn_l22_lend_class(uint32_t base_ub, uint32_t n_tokens, int64_t after);
// the tail bytes a class's FULL reserve needs given back: full - small + pool + max(128 MiB, full/16), at most cap (cap 0:
// no cap); 0 when full <= small
LLAMA_API size_t   llama_fn_l22_class_need(size_t full_b, size_t small_b, size_t pool_b, size_t cap);
// [TAG_FN_L22_SCHEDBUF] the graph node budget of a width whose reserve graphs were measured: nodes x 3/2 + 8192, never
// above the formula's
LLAMA_API uint32_t llama_fn_l22_sched_nodes(uint32_t formula, uint32_t nodes_measured);
// [TAG_FN_L22_SAMPSTATIC] the sampling nodes graph_max_nodes counts when every sampler builds n_out_per_seq chains:
// max(old count, sum of the samplers' nodes x n_out_per_seq)
LLAMA_API uint32_t llama_fn_l22_sampling_nodes(uint32_t old_count, uint32_t sum_nodes, uint32_t n_out_per_seq);
// [TAG_FN_L22_E6] (#30694) the state writer's device reads: consecutive ranges of the same tensor, each < 64 KiB, at
// increasing offsets, merged while the span stays <= 8 x their payload + 1 MiB. groups: [first, last) index pairs
struct llama_fn_l22_io_range {
    const void * tensor;
    size_t       offset;
    size_t       size;
};
LLAMA_API void     llama_fn_l22_io_merge(const std::vector<llama_fn_l22_io_range> & r, std::vector<std::pair<size_t, size_t>> & groups);
// [TAG_FN_L22_GETTER] the width of the next prompt ubatch (llama_fn_prompt_ubatch_width_ext) for a context in the SMALL
// state, given the hint `after` (the prompt tokens to come): the rule the next switch to FULL applies (l21 or l20), then
// the class
LLAMA_API uint32_t llama_fn_l22_next_width(uint32_t base_ub, uint32_t n_ubatch, bool l21_bigub, bool l21_big_off, bool lendclass,
        int64_t after);

// [TAG_FN_L22_P10] round 22, package P10 (stream tuning + G3); llama_fn_l22_cost is in llama-fn-l22-p10.h
// [TAG_FN_L22_STAGERS] LLAMA_FN_L22_STAGERS: unset / "" / 0 / off -> 0 (pool untouched), "auto" / 1 -> -1 (the formula),
// n >= 2 -> n (fixed)
LLAMA_API int      llama_fn_l22_stagers_mode(const char * v);
// copy threads of a pass whose host bytes are `host`, `staged` of them memcpy'd into the ring by the pool:
// round(n_max x staged / host) in [n_min, n_max]; host 0 -> n_max
LLAMA_API int      llama_fn_l22_stagers(uint64_t staged, uint64_t host, int n_max, int n_min);
// [TAG_FN_L22_XOVER] the depth bucket of a prompt that starts at n_kv cells: < 16K, < 64K, < 160K, >= 160K
LLAMA_API int      llama_fn_l22_depth_bucket(int64_t n_kv);
// the priors (EST): pass 3400 ms (pinned 1700), 0.42 GPU ms per row, 4.55 CPU-path ms per row, fix 240 ms, 16K extra 300 ms
LLAMA_API void     llama_fn_l22_cost_init(llama_fn_l22_cost & c, bool pinned);
// the pins came or went: the copy cost of every bucket starts again from its prior (the row costs stay)
LLAMA_API void     llama_fn_l22_cost_repin(llama_fn_l22_cost & c, bool pinned);
// samples (EWMA alpha 0.25; from the 4th sample of a value on, a sample is clamped to [0.5x, 2x] of its estimate). A pass
// bound by its copies (gate wait >= 10 % of the wall) gives its wall as the copy cost; a compute-bound pass only lowers a
// larger copy estimate to its wall. Passes of >= 2048 rows give the GPU row cost (wall - wait) / rows
LLAMA_API void     llama_fn_l22_cost_pass(llama_fn_l22_cost & c, int b, uint32_t rows, double wall_ms, double wait_ms);
LLAMA_API void     llama_fn_l22_cost_cpu (llama_fn_l22_cost & c, int b, uint32_t rows, double wall_ms);
// fix_ms < 0: no sample; big_ms < 0: no 16K period
LLAMA_API void     llama_fn_l22_cost_fix (llama_fn_l22_cost & c, double fix_ms, double big_ms);
// one stream pass of `rows`: max(copy cost, rows x GPU row cost); the CPU path: rows x CPU row cost
LLAMA_API double   llama_fn_l22_est_pass(const llama_fn_l22_cost & c, int b, uint32_t rows);
LLAMA_API double   llama_fn_l22_est_cpu (const llama_fn_l22_cost & c, int b, uint32_t rows);
// the crossover: the fewest rows in [lo, hi] whose CPU-path cost reaches one stream pass + the fixed cost. cur until the
// bucket has >= 3 stream passes and >= 3 CPU-path samples; cur when the new value is within 10 % of it
LLAMA_API int32_t  llama_fn_l22_xover(const llama_fn_l22_cost & c, int b, int32_t cur, int32_t lo, int32_t hi);
// [TAG_FN_L22_UBPLAN] the FULL width for a prompt of n_tokens + after rows (after < 0: unknown -> llama_fn_l21_full_width):
// base_ub when the whole prompt fits it or !big_ok; else the cost of the ubatch split [w, w, .., rest] (rows below xover
// on the CPU path, a 16K period adds its extra) for w = base_ub and n_ubatch, and the other width than today's rule only when
// it is at least 3 % cheaper. est_base / est_big (may be null): the two costs (0 when not priced)
LLAMA_API uint32_t llama_fn_l22_ubplan(const llama_fn_l22_cost & c, int b, uint32_t base_ub, uint32_t n_ubatch, uint32_t n_tokens,
        int64_t after, int32_t xover, bool big_ok, double * est_base, double * est_big);
// [TAG_FN_L22_BIGUB2] the 16K FULL reserve for one prompt sequence is used: two or more sequences and the 1-sequence 16K
// device bytes are at least the n_seq_max-sequence base-width bytes (a 2-prompt batch then splits at the base width and
// never grows the buffers)
LLAMA_API bool     llama_fn_l22_bigub2_ok(uint32_t n_seq_max, size_t full_big_1seq, size_t full_base_nseq);
// the sequences of a batch with more than small_t rows (prompt rows; decode / verify rows stay below)
LLAMA_API uint32_t llama_fn_l22_prompt_seqs(const uint32_t * rows_per_seq, int n_seq, uint32_t small_t);
// [TAG_FN_L22_SMALLKEEP] the kept SMALL reserve fits beside FULL: free_now >= expect_full + pool + margin (no overflow);
// false for an unknown FULL size (0)
LLAMA_API bool     llama_fn_l22_smallkeep_fits(size_t free_now, size_t expect_full, size_t pool, size_t margin);
// [TAG_FN_L22_XOVER] the crossover the PHOT rule reads through llama_fn_l22_offload_min (llama-fn-l22.h); 0 clears it
LLAMA_API void     llama_fn_l22_offload_min_set(int32_t v);

//
// model hooks (src/llama.cpp) and lookups
//

// before the tensors are created: decide the profile, the placement and the options; inject the levers
void llama_fn_auto_on_load(llama_model & model, llama_model_loader & ml, const std::string & fname, const llama_model_params & params);
// after the tensors are created: confirm host experts with a device router, else turn everything back off
void llama_fn_auto_after_load(llama_model & model);
// the model is being freed
void llama_fn_auto_on_free(llama_model & model);

// [TAG_FN_SHIP1] src/llama.cpp: LLAMA_NO_ECOQOS=1 opts the process out of EcoQoS (power throttling). llama_backend_init()
// calls it before any model exists; the qwen4exp profile calls it again after it set the variable.
void llama_ecoqos_opt_out(void);

// getenv(name), else this model's automatic value, else nullptr
const char * llama_fn_env(const llama_model & model, const char * name);
bool         llama_fn_active(const llama_model & model);

// [TAG_FN_L3_HOST] lever round 3, host / WDDM / sync family. Every switch is off unless set, and read for arch qwen4exp
// only (every other model runs none of this code), through llama_fn_env:
//   LLAMA_FN_HOST_EXEC=1     [TAG_FN_L3_HOST_EXEC] with LLAMA_MOE_BRIDGE_PF=1 (spin): the bridge's executor computes
//                            its share of every job (thread 0 of the pool, PF_SOLO off) and pulls that share into its
//                            own caches between two jobs, until the next job is posted (it watches the next ring word)
//   LLAMA_FN_HOST_STEP=1     [TAG_FN_L3_HOST_STEP] the hot set's and the DMA ring's step after each decode of their
//                            owner run on a helper thread, joined at that context's next decode / reserve / free
//                            (inline while the hot set holds an MTP block layer: the draft context reads those)
//   LLAMA_FN_HOST_LAUNCH2=N  [TAG_FN_L3_HOST_LAUNCH2] the scheduler cuts a bridged decode graph after layer N's output
//                            (l_last-N, 0 <= N < n_layer - 1): two device graph launches per step, the first runs while
//                            the host launches the second
//   LLAMA_FN_HOST_LAUNCH_EVERY=K  [TAG_FN_L3_HOST_LAUNCH2] a cut after every K layers (l_last-(K-1), l_last-(2K-1), ...):
//                            one launch per K layers, the device runs a piece while the host launches the next (with
//                            LAUNCH2 the cuts of both). The graph inputs of all pieces are copied before the first one
//   LLAMA_FN_HOST_POKE=1     [TAG_FN_L3_HOST_POKE] flush WDDM's batched submission after an eager (non-graph) CUDA
//                            split and after the prefill stream's / DMA issuer's queued copies
//   LLAMA_FN_HOST_QOS=1      [TAG_FN_L3_HOST_QOS] Windows: 1 ms timer resolution while the model lives, and the process
//                            keeps it (and full execution speed) when minimized or occluded
//   LLAMA_FN_HOST_QUIET=1    [TAG_FN_L3_HOST_QUIET] the hot set's upload worker and the DMA ring's fill threads wait
//                            while a bridged graph runs (its CPU experts are DRAM bound) and copy between graphs
//   LLAMA_FN_HOST_DMAOFF=1   [TAG_FN_L3_HOST_DMAOFF] no DMA share at all: the bridge makes no fetch side and no context
//                            makes a DMA state (LLAMA_MOE_BRIDGE_DMA=0 alone let the MTP draft context make one: 148.5 MiB
//                            of banks, a 1 GiB pinned ring and fill threads that no graph of it uses)
//   LLAMA_FN_HOST_REARM=1|N  [TAG_FN_L3_HOST_REARM] a bridge that 3 errors turned off comes back after a cool-down (1: 256
//                            steps or 30 s, N > 1: N steps or N/8 s, whichever ends first), doubled at every further
//                            turn-off up to 64x; 65536 clean graphs start the back-off over. Off: off for good (as before)
//   LLAMA_FN_HOST_DIAG=1|N   [TAG_FN_L3_HOST_DIAG] the bridge's per-graph host timing every N (1: 64) bridged graphs
//   LLAMA_FN_HOST_SYNC_TRACE=<us>  [TAG_FN_L3_HOST_DIAG] every synchronize / blocking device copy of >= us is counted by
//                            its call stack (ggml_backend_sync_trace_set), a report every 256 such waits
//   LLAMA_FN_HOST_KPOOL_PROBE=1  [TAG_FN_L3_HOST_DIAG] host time of the k-pool layout update and of its input arrays per
//                            ubatch, a line every 64 fills (read with getenv by llama-memory-hybrid-idx.cpp)
bool llama_fn_l3_flag(const llama_model & model, const char * name);
int  llama_fn_l3_int (const llama_model & model, const char * name, int def);
// [TAG_FN_L21] the round-21 stability switches cover every model: qwen4exp reads them through llama_fn_env (environment,
// then the profile), every other model from the environment only; unset = def
int  llama_fn_l21_int(const llama_model & model, const char * name, int def);
// [TAG_FN_L3_HOST_QOS] [TAG_FN_L3_HOST_POKE] model hooks (load: at the end of llama_fn_auto_on_load, after the profile
// state exists; free)
void llama_fn_l3_host_on_load(const llama_model & model);
void llama_fn_l3_host_on_free(const llama_model & model);

// the contexts of a model (the VRAM fit breakdown, and whether the MTP draft context exists yet)
void llama_fn_ctx_add   (const llama_model & model, llama_context * ctx, int ctx_type);
void llama_fn_ctx_remove(const llama_model & model, llama_context * ctx);
std::vector<std::pair<llama_context *, int>> llama_fn_ctx_list(const llama_model & model);
