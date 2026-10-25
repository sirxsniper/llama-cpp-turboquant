#pragma once

// [TAG_FN_L22] round 22 shared interfaces (E:/turbot-gates/flashnext/test/r22/PLAN.md section 4). Written once by P0 and
// frozen: only the "requests" section at the end may grow (the steward merges r22/impl/requests.md into it). Each group
// names its owner; P0 put a stub in the owner's file that keeps today's behaviour (no pins, no SSD tier, no lists ...).
// Every function is exported (LLAMA_API) so the server, the tools and the CPU unit tests can call it.

#include "llama.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <functional>

struct llama_model;
struct llama_context;
struct llama_ubatch;
struct llama_moe_bridge;
struct llama_moe_dma_view;
struct llm_graph_context;
struct ggml_moe_bridge_job;
struct ggml_turbot_op_params;

//
// switches (P0)
//

// a switch value turns its lever on: non-empty and not 0 / 0.0 / off / false / no (case-insensitive)
LLAMA_API bool llama_fn_l22_switch_on(const char * v);

// [TAG_FN_L22_PHASE] the answer-changing levers (plan 3.8). Read once per context from the model's environment / profile
// (llama_model_fn_env; a null model reads the plain environment).
enum llama_fn_l22_xbit : uint8_t {
    LLAMA_FN_L22_XB_TAIL     = 1, // LLAMA_FN_L22_X_TAILSKIP > 0 and (one sequence or LLAMA_FN_L22_X_TAILSKIP_MULTI)
    LLAMA_FN_L22_XB_PRIOR    = 2, // LLAMA_FN_L22_X_PRIOR
    LLAMA_FN_L22_XB_DEFER    = 4, // LLAMA_FN_L22_X_DEFER (P11 checks one sequence itself)
    LLAMA_FN_L22_XB_RESIDENT = 8, // LLAMA_FN_L22_X_RESIDENT
};

struct llama_fn_l22_x_cfg {
    int32_t tailskip   = 0;     // LLAMA_FN_L22_X_TAILSKIP (the rank R, 0 = off)
    bool    tail_multi = false; // LLAMA_FN_L22_X_TAILSKIP_MULTI
    bool    prior      = false; // LLAMA_FN_L22_X_PRIOR
    bool    defer      = false; // LLAMA_FN_L22_X_DEFER (any value but off, "check" too)
    bool    resident   = false; // LLAMA_FN_L22_X_RESIDENT

    bool any() const { return tailskip > 0 || prior || defer || resident; }
};

LLAMA_API llama_fn_l22_x_cfg llama_fn_l22_x_read(const llama_model * model);

// graph hooks a context turns on (llm_graph_params::fn_hooks), read once per context like the X levers
enum llama_fn_l22_hook : uint8_t {
    LLAMA_FN_L22_HOOK_CPUSHARE = 1, // LLAMA_FN_L22_CPUSHARE: build_moe_ffn asks llama_fn_cpushare_plan on streamed layers
    LLAMA_FN_L22_HOOK_SAMPSTATIC = 2, // R-P1-1 LLAMA_FN_L22_SAMPSTATIC: build_sampling builds n_outputs_max_per_seq chains (E7)
    LLAMA_FN_L22_HOOK_PRIOR      = 4, // R-P12-2 LLAMA_FN_L22_X_PRIOR or _X_RESIDENT set: build_moe_ffn calls llama_fn_route_prior
                                      // in every graph; the hook acts by g->fn_gen / g->fn_x (or _X_PRIOR_PROMPT)
};

LLAMA_API uint8_t llama_fn_l22_hooks_read(const llama_model * model);

// [TAG_FN_L22_XOVER] P10 R1: the op-offload minimum the PHOT rule of build_moe_ffn uses (> 0), 0 = GGML_OP_OFFLOAD_MIN_BATCH
// as today. Defined in src/llama-fn-auto.cpp (P0 stub 0; P10 in phase 2)
LLAMA_API int32_t llama_fn_l22_offload_min(void);

// fn_gen (plan 3.8): 1 only when the phase is LLAMA_FN_PHASE_GEN, every row of the ubatch is an output row (n_outputs ==
// n_tokens > 0) and an X lever is set for the model; else 0
LLAMA_API uint8_t llama_fn_l22_gen_rule(int32_t phase, uint32_t n_tokens, uint32_t n_outputs, bool x_any);

// the levers that act in a graph (llama_fn_l22_xbit), 0 unless fn_gen
LLAMA_API uint8_t llama_fn_l22_x_bits(const llama_fn_l22_x_cfg & cfg, uint8_t fn_gen, uint32_t n_seqs_unq);

//
// residency (P5, src/llama-fn-resident.cpp; stub: RAM / 0 / no-op / 0)
//

enum llama_fn_res {
    LLAMA_FN_RES_RAM = 0,
    LLAMA_FN_RES_VRAM,
    LLAMA_FN_RES_PIN,
    LLAMA_FN_RES_SSD,
};

LLAMA_API uint8_t  llama_fn_res_class(const llama_model * model, int il, int e);
LLAMA_API uint64_t llama_fn_res_gen(const llama_model * model);
LLAMA_API void     llama_fn_res_mark_pin(const llama_model * model, int il, int e, bool on);
LLAMA_API uint64_t llama_fn_res_stream_bytes(const llama_model * model, uint8_t cls);
// [TAG_FN_L22_SSDFIX] the classes of experts [0, n) of layer il in one call (the stream's samples and read-ahead); false
// (out untouched) when there is no residency table or the layer is not in it
LLAMA_API bool     llama_fn_res_classes(const llama_model * model, int il, uint8_t * out, int n);

//
// expert ranges (P0, src/llama-fn-l22-log.cpp)
//

// the host bytes of expert e of layer il, part 0 up, 1 gate, 2 down: data + e x nb[2], nb[2] bytes of the layer's
// ffn_*_exps tensor. false if the layer has no such tensor, e is out of range or the tensor is not in host memory
LLAMA_API bool llama_fn_expert_host_range(const llama_model * model, int il, int e, int part /*0 up, 1 gate, 2 down*/,
        const uint8_t ** p, size_t * n);

//
// direct I/O (P5, src/llama-fn-dio.cpp; stub: submit -1, the caller uses the mapped read)
//

struct llama_fn_dio_req {
    const void * src; // the mapped address of the bytes (the reader maps it to the file offset)
    size_t       len;
    void *       dst;
};

// a ticket >= 0, or -1 when direct I/O cannot take the requests (read the mapped bytes instead)
LLAMA_API int  llama_fn_dio_submit(const llama_model * model, const llama_fn_dio_req * req, int n);
// true when every request of the ticket was read
LLAMA_API bool llama_fn_dio_wait(int ticket);

//
// pins (P6, src/llama-fn-pin.cpp; stub: no pins)
//

LLAMA_API void llama_fn_pin_on_ready(const llama_model * model);

struct llama_fn_pin_seg {
    const void * src;    // the first byte of the segment
    size_t       len;
    bool         pinned; // registered with the device (DMA-able)
    int32_t      ticket; // release with llama_fn_pin_release when >= 0 (holds the range registered)
};

// the first segment of [p, p + n): its length and whether it is pinned. Stub: one unpinned segment of length n
LLAMA_API bool llama_fn_pin_query(const void * p, size_t n, llama_fn_pin_seg * seg);
LLAMA_API void llama_fn_pin_release(int32_t ticket);
// a device allocation of need_bytes is about to fail: unpin enough (synchronized) and return true to retry. Stub: false
LLAMA_API bool llama_fn_pin_on_oom(size_t need_bytes);
// [TAG_FN_L23_PINFIX] R-L23-5: an allocation failed while pins were held: every pin goes (true: some did), the caller retries once
LLAMA_API bool llama_fn_pin_on_alloc_fail(size_t need_bytes);
LLAMA_API bool llama_fn_pin_active(void);
// DXGI QueryVideoMemoryInfo of the adapter with this LUID (non_local 0: LOCAL, 1: NON_LOCAL). Stub: false
LLAMA_API bool llama_fn_dxgi_query(uint64_t luid, int non_local, uint64_t * budget, uint64_t * usage);
// P6 R1: llama_context::decode marks its begin (true) and end (false) on every context (stub: no-op)
LLAMA_API void llama_fn_pin_decode_mark(bool in_decode);
// [TAG_FN_L23_PINFIX] R-L23-8: the server marks a request (+1 / -1; 0 = time stamp only): pins register only while idle
LLAMA_API void llama_fn_pin_server_busy(int delta);
// P6 R2: a device reserve / lend / grow size a package may need after ready, for the watcher's LOCAL reserve (stub: no-op)
enum llama_fn_pin_need {
    LLAMA_FN_PIN_NEED_FULL_EXTRA = 0, // the FULL compute reserve above SMALL (P4, P10)
    LLAMA_FN_PIN_NEED_BORROW,         // the 16K borrow (P4, P10)
    LLAMA_FN_PIN_NEED_VISION_LEND,    // the projector's lend (P2)
    LLAMA_FN_PIN_NEED_KV_GROW,        // one KV lend grow step (P1)
    LLAMA_FN_PIN_NEED_GRAPH_INST,     // CUDA graph instances (P7)
};
LLAMA_API void llama_fn_pin_note_need(const llama_model * model, int kind, size_t bytes);
// P6 R3: stop the pin thread, wait for tickets (bounded), unregister everything; the model is going away (P5's tier
// deleter and P6's arena deleter call it first; stub: no-op)
LLAMA_API void llama_fn_pin_shutdown(const llama_model * model);

//
// control vector, project mode (P12c, src/llama-fn-cvec.cpp; stub returns h). P12c is skipped in round 22 (owner
// 10-18): the qwen4exp call is not placed, the declaration stays for round 23
//

LLAMA_API ggml_tensor * llama_fn_cvec_project(ggml_context * ctx, const llama_model * model, ggml_tensor * h, int il);

//
// route prior (P12, src/llama-fn-prior.cpp; stub returns probs). build_moe_ffn calls it on the selection probabilities
// (before the top-k; the weights are still gathered from the unscaled probabilities) in every graph of a context with
// LLAMA_FN_L22_HOOK_PRIOR (R-P12-2); it must return probs unchanged unless g->fn_x has LLAMA_FN_L22_XB_PRIOR /
// LLAMA_FN_L22_XB_RESIDENT (decode-phase GEN) or its own prompt rule (_X_PRIOR_PROMPT) applies
//

LLAMA_API ggml_tensor * llama_fn_route_prior(llm_graph_context * g, ggml_tensor * probs, int il);

//
// CPU share of streamed prompt ubatches (P16, src/llama-fn-cpushare.cpp; stub false). With LLAMA_FN_L22_HOOK_CPUSHARE
// (R-P16-1) build_moe_ffn defers a streamed layer's stream gate to after the routing and calls the plan there (after the
// routing trace); the plan may build the gate itself (gate_built), else build_moe_ffn builds it right after the call. A
// plan that returns true may replace the ids and the bank aliases of the three expert MUL_MAT_IDs, give a src[3] skip
// table and a host sum added to the output, and gets llama_fn_cpushare_finish(moe_out) before "ffn_moe_out"
//

struct llama_fn_cpushare_io {
    // in
    ggml_tensor * cur;              // [n_embd, n_tokens] f32, the layer's FFN input
    ggml_tensor * selected_experts; // [n_used, n_tokens] i32
    ggml_tensor * weights;          // [1, n_used, n_tokens] f32
    ggml_tensor * up_host;          // the layer's host expert tensors
    ggml_tensor * gate_host;
    ggml_tensor * down_host;
    const void  * pfs;              // const llama_pfs_view * of the layer (llama-moe-gen5.h)
    ggml_tensor * up_dev;           // the stream bank aliases the expert MUL_MAT_IDs read
    ggml_tensor * gate_dev;
    ggml_tensor * down_dev;
    // out
    bool          gate_built = false;   // the plan built the layer's stream gate (also when it returns false)
    ggml_tensor * ids       = nullptr;  // or null: the ids the three expert MUL_MAT_IDs read instead of selected_experts
    ggml_tensor * up_mm     = nullptr;  // or null: replacements of up_dev / gate_dev / down_dev in those MUL_MAT_IDs
    ggml_tensor * gate_mm   = nullptr;
    ggml_tensor * down_mm   = nullptr;
    ggml_tensor * dev_table = nullptr;  // or null: i32 [n_expert] skip table of the expert MUL_MAT_IDs (src[3])
    int32_t       dev_miss  = 0;        // the value of a computed (not skipped) expert in dev_table
    ggml_tensor * host_out  = nullptr;  // or null: f32 [n_embd, n_tokens], the CPU share weighted sum, added to the output
    void *        user      = nullptr;  // the plan state for llama_fn_cpushare_finish
};

// plan: llama_fn_cpushare_io *
LLAMA_API bool          llama_fn_cpushare_plan(llm_graph_context * g, const llama_ubatch & ubatch, int il, void * plan);
// after a plan that returned true: the layer output before "ffn_moe_out" (stub: moe_out)
LLAMA_API ggml_tensor * llama_fn_cpushare_finish(llm_graph_context * g, void * plan, ggml_tensor * moe_out);

//
// per-sequence K/V rows of a unified turbot cache (P13, src/llama-fn-kvrows.cpp; stub nullptr). build_attn_mha calls it
// for a turbot FA over one stream with the positional mask, only when LLAMA_KV_ROWS is 1 or 2; a non-null result is the
// FA output (in token order, the layout of the one FA), null keeps the one FA. The hook may add its own graph input
// (g->res->add_input) for the lists.
//

struct llama_fn_kvrows_args {
    ggml_tensor * q;                     // permuted as the FA takes it
    ggml_tensor * k;
    ggml_tensor * v;
    ggml_tensor * kq_mask;
    ggml_tensor * sinks;
    ggml_tensor * v_mla;
    ggml_tensor * kv_pos;
    ggml_tensor * q_pos;
    ggml_tensor * turbot_pool;
    ggml_tensor * turbot_gtab;
    const ggml_turbot_op_params * turbot_params;
    int64_t       n_kv_max;
    float         kq_scale;
    int           il;
    const void *  mctx;                  // const llama_kv_cache_context * of the layer (fn_cells, get_n_kv, ...)
    // one FA node exactly as build_attn_mha builds it (sinks, n_kv_max, positional mask with q_pos_fa, turbot, F32 acc)
    std::function<ggml_tensor *(ggml_tensor * q_fa, ggml_tensor * k_fa, ggml_tensor * v_fa, ggml_tensor * mask_fa,
                                ggml_tensor * gtab_fa, ggml_tensor * q_pos_fa)> build_fa;
};

// args: const llama_fn_kvrows_args *
LLAMA_API ggml_tensor * llama_fn_kvrows_attn(llm_graph_context * g, const void * args);

//
// decode DMA share from pinned pages (P14, src/llama-fn-dmashare.cpp and ggml/src/ggml-cuda/fn-l22-dmafetch.cu; stub
// false / nullptr). The bridge asks _wanted once at its creation (spin mode, fetch side available): true gives it the
// device fetch side (plan areas) and turns its DMA flag on. The graph asks _view for a layer's banks when the gen5 DMA
// state has none. br_run calls _plan for a job that carries GGML_MOE_BRIDGE_JOB_DMA while no gen5 DMA state serves the
// layer: the plan writes job->plan (n_copy, off, slot, slot_ids) and the skip table (skip[e] != 0: the CPU skips
// expert e; the hot experts are already marked), then the bridge publishes the plan. false fails the job.
//

struct llama_fn_dmashare_job {
    const llama_moe_bridge *    br;
    const ggml_moe_bridge_job * job;       // ids, w, n_tokens, n_used, plan
    int32_t                     chan;
    int32_t                     il;        // the model layer, -1 unknown
    const ggml_tensor *         up;        // the layer's host expert tensors
    const ggml_tensor *         gate;
    const ggml_tensor *         down;
    const int32_t *             hot_table; // or null: expert -> hot slot, hot_miss when not hot
    int32_t                     hot_miss;
    int32_t                     max_fetch; // experts the device fetch can copy per job
};

LLAMA_API bool llama_fn_dmashare_wanted(const llama_model * model, const llama_moe_bridge * br);
LLAMA_API const llama_moe_dma_view * llama_fn_dmashare_view(const llama_moe_bridge * br, const ggml_tensor * up_exps, int64_t n_tokens);
LLAMA_API bool llama_fn_dmashare_plan(llama_fn_dmashare_job * jv, int32_t * skip_table);
// R-P14-3: the bridge's job / graph / life events while l22_dmashare (the job's CPU time and CPU experts after the pool
// ran it; the graph synced, with its bridge error and the device wait per layer in us or -1; the bridge goes away after
// its executor stopped)
LLAMA_API void llama_fn_dmashare_job_done(const llama_moe_bridge * br, int32_t chan, uint32_t seq, uint64_t cpu_ns, uint32_t n_cpu);
LLAMA_API void llama_fn_dmashare_graph_end(const llama_moe_bridge * br, uint32_t err, double dev_wait_us_per_layer);
LLAMA_API void llama_fn_dmashare_free(const llama_moe_bridge * br);

//
// PASSLOG (P0, src/llama-fn-l22-log.cpp)
//

// LLAMA_FN_L22_PASSLOG: 0 off, 1 lines, 2 verbose lines (environment, else the profile of a loaded qwen4exp model)
LLAMA_API int llama_fn_l22_passlog_level();

struct llama_fn_l22_counters {
    uint64_t faults;         // page faults of this process (GetProcessMemoryInfo)
    uint64_t sys_page_reads; // system page reads (SystemPerformanceInformation.PageReadCount), 0 when not readable
    uint64_t avail_ram;      // available physical bytes (GlobalMemoryStatusEx)
    uint64_t standby;        // standby list bytes (SystemMemoryListInformation) when allowed, else 0
};

LLAMA_API llama_fn_l22_counters llama_fn_l22_counters_now();

// one WARN line "[TAG_FN_L22_PASSLOG] kind=<kind> <text>" when the level is >= 1 (kind: pass, switch, res, pin, media,
// turn, graph, dma)
LLAMA_API void llama_fn_l22_log(const char * kind, const char * fmt, ...);

// the profile's LLAMA_FN_L22_PASSLOG of a loaded model (llama-fn-auto calls them at load and free)
LLAMA_API void llama_fn_l22_passlog_model(const llama_model * model, bool loaded);

//
// requests (append-only, the steward; see E:/turbot-gates/flashnext/test/r22/impl/requests.md)
//

// R-P6-1 / R-P14-5 (P6, src/llama-fn-pin.cpp; no stub): llama_fn_pin_query plus the device address of seg->src for a
// pinned segment (0 when unknown or unpinned; set only with LLAMA_FN_L22_DMASHARE). Valid while the ticket is held
LLAMA_API bool llama_fn_pin_query_dev(const void * p, size_t n, llama_fn_pin_seg * seg, uint64_t * dev);

// R-REV-2 (review fixes, P6, src/llama-fn-pin.cpp): the headroom every pin pre-check keeps on top of an allocation's bytes:
// 512 MiB, at most the watcher threshold that the pin budget set at ready (512 MiB without pins)
LLAMA_API uint64_t llama_fn_pin_precheck_margin(void);
// R-REV-12 (review fixes, P6): 0 llama_fn_pin_on_ready not called yet, 1 planning / registering, 2 registration ended
// (pins may be active or all gone), 3 no pins in this run (no thread, or shut down)
LLAMA_API int llama_fn_pin_phase(void);

// R-REV-6 (review fixes, P0, src/llama-fn-l22-log.cpp): the plan 3.8 gates of the answer-changing levers as pure functions
// (build_moe_ffn, fn_x_defer_mode and llama_fn_route_prior call them; test-fn-l22-gate covers them). The bridge job flags
// a graph's levers add to its posts and sync ops: GGML_MOE_BRIDGE_JOB_TAIL iff fn_x has LLAMA_FN_L22_XB_TAIL
LLAMA_API int32_t llama_fn_l22_job_flags(uint8_t fn_x);
// DEFER acts in a graph: fn_x has LLAMA_FN_L22_XB_DEFER, one sequence, a bridge (or its sync op), not a two-chain graph
LLAMA_API bool llama_fn_l22_defer_acts(uint8_t fn_x, uint32_t n_seqs_unq, bool have_bridge, bool tbo);
// the route prior / resident swap act in a graph: fn_x has XB_PRIOR or XB_RESIDENT, or (_X_PRIOR_PROMPT) fn_gen is 0
LLAMA_API bool llama_fn_l22_prior_acts(uint8_t fn_x, uint8_t fn_gen, bool prompt_rule);
