#pragma once

// [TAG_FN_L22_CPUSHARE] round 22, package P16 (M11): the CPU share of streamed prompt ubatches of <= 3072 rows. Shared by
// llama-fn-cpushare.cpp, llama-prefill-stream.cpp, llama-context.cpp and tests/test-fn-l22-cpushare.cpp only.
//
// LLAMA_FN_L22_CPUSHARE (qwen4exp, owner context; unset / 0 = off):
//   1  experts with 1.._ROWS rows of a layer whose bytes the stream copies late run on the CPU (graph threads, during the
//      gate wait) while pins are active; the device reads a zero expert for them; the stream also skips late experts that
//      no row routes to (exact)
//   2  only the skip of the late experts that no row routes to (exact, bit-identical outputs). [TAG_R35_ROUTED] with
//      LLAMA_R35_CPS_NOWAIT (default 1) the stream never waits for a plan and skips any expert it has not copied yet that
//      no row routes to (late or not); LLAMA_R35_CPS_NOWAIT=0 is the v0.27.0 rule (late experts wait <= _WAIT_MS)
// A share layer: plan op (CPU: late set at the first layer, row counts, CPU set, skip publish, remapped ids + token list),
// share op (CPU, all threads: the CPU experts' rows, per token sums C), share gate (waits for the bank), the device
// MUL_MAT_IDs on the zero-expert aliases with the remapped ids, then moe_out rows of the CPU tokens += C (set_rows).

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

struct llama_model;
struct llama_pfs_view;
struct llama_fn_cpushare_io;

//
// configuration (llama-fn-cpushare.cpp)
//

struct llama_fn_cps_cfg {
    int mode      = 0;    // LLAMA_FN_L22_CPUSHARE: 0 off, 1 CPU share + skip, 2 skip only
    int rows      = 8;    // _ROWS: experts with 1..rows rows may go to the CPU
    int max_t     = 3072; // _MAX: ubatches of at most this many rows
    int cap       = 512;  // _CAP: distinct CPU tokens per layer (columns of the share op's output)
    int k_fixed   = -1;   // _K: >= 0 fixes the most CPU experts per layer (KLD runs); -1 = the measured budget
    int wait_ms   = 100;  // _WAIT_MS: the streamer waits at most this long for a layer's plan, then copies everything
    int late_rows = 16;   // _LATE: experts predicted to get at most this many rows are copied after the layer's plan
    int early_kib = -1;   // _EARLY_KIB: early bytes kept per layer; -1 = measured plan delay x copy rate (so the copy
                          // engine has early work while the plan forms)
    // [TAG_R35_ROUTED] LLAMA_R35_CPS_NOWAIT (mode 2 only; 0 = the v0.27.0 rule): the streamer never waits for a layer's
    // plan. It copies the layer's experts in pieces of about one ring chunk (the early ones, then the late ones) and looks
    // the plan up before each piece; once it is published, every expert not copied yet that no row of the ubatch routes to
    // is skipped (Strata #4283's rule: never wait, skip what the routing shows is unused)
    int nowait    = 1;
    // [TAG_R35_ROUTED] LLAMA_R35_CPS_PHASES (no-wait rule only): 0 (default) = one pass over the layer's host experts in id
    // order; 1 = the early experts, then the late ones. The split interleaves by id, so both phases came in runs of 1-2
    // experts (~257 runs per tensor and layer): 24 % fewer bytes but only 3 % less pass time (r35 port turn 1)
    int phases    = 0;
    // [TAG_R35_CPSGUARD] LLAMA_R35_CPS_GUARD_KIB (0 = the v0.27.0 copies): a share job that leaves an expert out still
    // writes the first KiB of it (its own host bytes) after every expert it does write. The batched kernels (MMQ) read
    // past the last row of an expert into the next one (the last K iteration loads ITER_K/qk blocks against zero-padded
    // activations); a skipped expert's slot holds an older layer's bytes, and a NaN / Inf scale there times 0 made the
    // routed expert's last output feature NaN (r34 port: CPUSHARE=2, 1511 rows, 50 GiB skipped, then the NaN guard)
    int guard_kib = 4;
};

// the owner context (qwen4exp, not the MTP draft) before the stream exists; reads the switches through the profile.
// Returns the mode this owner got (0: none)
LLAMA_API int  llama_fn_cpushare_configure(const llama_model & model, const void * owner);
// tests: a given configuration (model may be null: classes RAM, host bytes from the host tensors)
LLAMA_API int  llama_fn_cpushare_configure_cfg(const llama_fn_cps_cfg & cfg, const llama_model * model, const void * owner);
LLAMA_API void llama_fn_cpushare_free(const void * owner);
// nullptr when no share is configured
LLAMA_API const llama_fn_cps_cfg * llama_fn_cpushare_cfg();
// mode 1 is configured: the stream's banks carry a zero expert behind every layer
LLAMA_API bool llama_fn_cpushare_zero_expert();
// R-REV-7: the device bytes one share layer adds to a streamed ubatch of n_tokens rows beyond the reserve graph (ids / w
// copies, the plan and share copies, rows / sum of the finish), 0 unless mode 1 is configured and n_tokens <= _MAX
LLAMA_API size_t llama_fn_cpushare_reserve_extra(int64_t n_embd, int64_t n_used, int64_t n_tokens);

// the share of one streamed layer (the hook body, also called by the CPU test): builds the plan / share / gate nodes and
// fills io (gate_built, ids, *_mm, user). false: no share for this layer (the caller builds the plain gate unless
// io->gate_built). chain_ok: the caller checked the layer's FFN form (separate up / gate, SILU, no bias / scale / clamp)
LLAMA_API bool llama_fn_cpushare_build(ggml_context * ctx, ggml_cgraph * gf, ggml_backend_sched_t sched, ggml_backend_t cpu,
        int il, llama_fn_cpushare_io * io, bool chain_ok);
// after a true build: moe_out with the CPU tokens' rows plus their share (mode 1), else moe_out
LLAMA_API ggml_tensor * llama_fn_cpushare_build_finish(ggml_context * ctx, ggml_backend_sched_t sched, llama_fn_cpushare_io * io,
        ggml_tensor * moe_out);

// per-pass sums for PASSLOG (taken and reset)
struct llama_fn_cps_sums {
    uint64_t ubatches    = 0; // share ubatches
    uint64_t layers      = 0; // share layer plans
    uint64_t cpu_experts = 0;
    uint64_t cpu_pairs   = 0;
    uint64_t cpu_tokens  = 0;
    uint64_t skip_cpu    = 0; // bytes the plans skipped because the CPU computes them
    uint64_t skip_unr    = 0; // bytes of late experts no row routes to
    uint64_t late_bytes  = 0; // late bytes at the ubatch begins
    uint64_t dropped     = 0; // CPU candidates dropped (bytes not readable in one piece)
    uint64_t cut_cap     = 0; // CPU candidates cut by the token / pair capacity
    double   cpu_ms      = 0.0;
    double   budget_k    = 0.0; // sum of CPU experts per layer (k mean = budget_k / layers)
    uint64_t pins_off    = 0;   // plans without active pins (no CPU set)
    int64_t  rows        = 0;   // rows of the last share ubatch
    double   rls_e       = 0.0; // the CPU time model now: us per expert
    double   rls_r       = 0.0; // and us per row
};
LLAMA_API llama_fn_cps_sums llama_fn_cpushare_take();

//
// pure helpers (tested by test-fn-l22-cpushare)
//

// late set of one layer: experts predicted to get <= late_rows rows (share[e] x n_tokens), not hot, by ascending predicted
// rows (ties: lower id) while the late bytes stay <= byte_cap
struct llama_fn_cps_late_in {
    int              n_expert  = 0;
    int              n_tokens  = 0;
    const float    * share     = nullptr; // [n_expert] rows / token
    const uint8_t  * hot       = nullptr; // [n_expert] or null
    const uint64_t * bytes     = nullptr; // [n_expert] host bytes (up + gate + down)
    int              late_rows = 16;
    uint64_t         byte_cap  = UINT64_MAX;
};
LLAMA_API uint64_t llama_fn_cps_late(const llama_fn_cps_late_in & in, uint8_t * late /*[n_expert]*/);

// the plan of one layer from its routing. CPU set = late, rows in [1, rows_max], cand, accepted; ordered by rows asc,
// bytes desc, id asc; cut by k_fixed (>= 0) or the budget (k_fixed < 0: add e while
// sum t + t(e) <= frac x (late routed bytes left on the copy - bytes(e)) / dma_bpu, t(e) = c_e + c_r x rows(e) us), and by
// cap distinct tokens / pairs_max pairs. skip = late and (rows 0 or CPU)
struct llama_fn_cps_pick_in {
    int              n_expert  = 0;
    int              n_used    = 0;
    int              n_tokens  = 0;
    const int32_t  * ids       = nullptr; // [n_tokens][n_used]
    const uint8_t  * late      = nullptr; // [n_expert] or null: no late phase (no skip, no CPU set)
    const uint8_t  * cand      = nullptr; // [n_expert] or null: no CPU set
    const uint64_t * bytes     = nullptr; // [n_expert]
    int              rows_max  = 8;
    int              k_fixed   = -1;
    int              cap       = 512;
    int              pairs_max = 2048;
    double           c_e       = 90.0;    // us per expert
    double           c_r       = 3.0;     // us per row
    double           dma_bpu   = 40000.0; // late copy bytes per us
    double           frac      = 0.8;
    std::function<bool(int)> accept;      // or empty: called once per expert before it joins the CPU set
    bool             skip_all  = false;   // [TAG_R35_ROUTED] skip every expert no row routes to, not only the late ones
};

struct llama_fn_cps_pick_out {
    std::vector<int32_t> rows;  // [n_expert] routed rows
    std::vector<uint8_t> skip;  // [n_expert]
    std::vector<uint8_t> cpu_m; // [n_expert] CPU set mask
    std::vector<int32_t> cpu;   // CPU experts, ascending id
    std::vector<int32_t> table; // [n_expert]: e, or n_expert for a CPU expert
    std::vector<int32_t> tok;   // distinct CPU tokens, ascending
    int      n_pairs   = 0;
    uint64_t skip_cpu  = 0;
    uint64_t skip_unr  = 0;
    uint64_t late_left = 0;     // late routed bytes the stream still copies
    double   t_cpu     = 0.0;   // predicted CPU us
    int      cut_cap   = 0;     // candidates cut by cap / pairs_max
    int      rejected  = 0;     // candidates accept() refused
};
LLAMA_API void llama_fn_cps_pick(const llama_fn_cps_pick_in & in, llama_fn_cps_pick_out & out);

// the CPU time model t = th0 x experts + th1 x pairs (us): recursive least squares with forgetting, bounded
struct llama_fn_cps_rls {
    double th[2]   = { 90.0, 3.0 };
    double P[2][2] = { { 400.0, 0.0 }, { 0.0, 4.0 } };
    double lambda  = 0.95;
    int    n       = 0;
    void   update(double x0, double x1, double y);
    double pred(double x0, double x1) const { return th[0]*x0 + th[1]*x1; }
};
LLAMA_API void llama_fn_cps_rls_update(llama_fn_cps_rls & r, double x0, double x1, double y);

// the share op's math for tests: C[:, j] for the CPU tokens of a plan, single thread, scalar order of the op
struct llama_fn_cps_ref_in {
    const ggml_tensor * up;   // host [n_embd, n_ff, n_expert]
    const ggml_tensor * gate;
    const ggml_tensor * down; // [n_ff, n_embd, n_expert]
    const float *       x;    // [n_tokens][n_embd]
    const float *       w;    // [n_tokens][n_used]
    const int32_t *     ids;  // [n_tokens][n_used]
    int                 n_used;
    int                 n_tokens;
};
// C as [n_tok][n_embd] for tok (ascending CPU tokens) and the CPU set mask; vec_dot per (row, column)
LLAMA_API bool llama_fn_cps_reference(const llama_fn_cps_ref_in & in, const std::vector<uint8_t> & cpu_m,
        const std::vector<int32_t> & tok, std::vector<float> & c);
// the share op itself on the plan of a CPU set (tok: the tokens routed to it, ascending), nth threads, with the
// multi-row kernels or not; c as [cap][n_embd] (columns from tok.size() on repeat token tok[0]'s). false: no kernels,
// or more tokens than cap
LLAMA_API bool llama_fn_cps_share_test(const llama_fn_cps_ref_in & in, const std::vector<uint8_t> & cpu_m, int nth, int cap,
        bool use_mr, std::vector<int32_t> & tok, std::vector<float> & c);

// tests: a mock pin provider (active, query, release) instead of P6's; null functions restore P6's
using llama_fn_cps_pin_query_t   = bool (*)(const void * p, size_t n, const void ** src, size_t * len, int32_t * ticket);
using llama_fn_cps_pin_release_t = void (*)(int32_t ticket);
LLAMA_API void llama_fn_cpushare_test_pins(int active, llama_fn_cps_pin_query_t q, llama_fn_cps_pin_release_t r);

//
// prefill stream side (llama-prefill-stream.cpp)
//

// the stream position of a layer view (-1: none)
LLAMA_API int  llama_prefill_stream_pos(const llama_pfs_view * v);
// stream positions (0: no stream)
LLAMA_API int  llama_prefill_stream_n_pos();
// the zero-expert aliases [ne0, ne1, n_expert + 1] of a layer (mode 1 at the stream's init); false: none
LLAMA_API bool llama_prefill_stream_view_z(const llama_pfs_view * v, ggml_tensor * out[3]);
// the backend that runs the stream's layers (the device; the CPU backend in tests)
LLAMA_API ggml_backend_t llama_prefill_stream_device();
// the first share layer of a ubatch: the new-ubatch work of gate 0 (queue every layer) with late masks [n_pos][n_expert]
// (null: no late phase this ubatch). Position 0 is never late.
LLAMA_API void llama_prefill_stream_share_begin(const uint8_t * late, int n_expert);
// the running ubatch began with late masks
LLAMA_API bool llama_prefill_stream_share_active();
// a layer's plan: the late experts the stream skips (skip[e] != 0; null = none); the streamer's wait for it ends
LLAMA_API void llama_prefill_stream_share_publish(const llama_pfs_view * v, const uint8_t * skip, int n_expert);
// the layer's bank is complete for this ubatch already (a share would save no copy)
LLAMA_API bool llama_prefill_stream_layer_full(const llama_pfs_view * v);
// the share gate (waits for the bank, no new-ubatch work): a CPU custom op after dep, expanded into gf
LLAMA_API ggml_tensor * llama_prefill_stream_build_gate_share(ggml_context * ctx, ggml_cgraph * gf, const llama_pfs_view * v,
        ggml_tensor * dep);

struct llama_pfs_share_counters {
    uint64_t jobs         = 0;   // jobs with a late phase
    uint64_t skip_bytes   = 0;   // late bytes the plans let the stream skip
    uint64_t late_bytes   = 0;   // late bytes copied after a plan
    uint64_t timeouts     = 0;   // plan waits that ran out (everything copied)
    uint64_t invalidated  = 0;   // partial banks dropped at a new ubatch
    double   plan_wait_ms = 0.0; // streamer time spent waiting for plans
    double   plan_us_ewma = -1.0; // job start -> plan arrival (EWMA)
    double   dma_bpu_ewma = -1.0; // bytes per us of share passes (EWMA)
    // [TAG_R35_ROUTED] [TAG_R35_CPSGUARD]
    uint64_t skip_early   = 0;    // early (not late) bytes the published plan let the stream skip (no-wait rule)
    uint64_t noplan_late  = 0;    // jobs whose late experts began before the plan was there (no-wait rule)
    uint64_t noplan       = 0;    // jobs that ended before the plan was there (no-wait rule)
    uint64_t guard_bytes  = 0;    // head bytes written after the written experts (guard)
};
LLAMA_API llama_pfs_share_counters llama_prefill_stream_share_counters();
