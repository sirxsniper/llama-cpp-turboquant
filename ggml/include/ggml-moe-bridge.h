#pragma once

// [TAG_MOE_BRIDGE] GPU <-> host doorbell for the routed experts of host-resident MoE layers.
//
// Graph side (ggml.h): ggml_moe_host_post writes x, ids and w of one MoE layer to a channel in mapped pinned host memory
// and returns a ticket; ggml_moe_host_wait waits (bounded) until the host has answered that ticket and returns the
// host's weighted sum [n_embd, T]. Between the two, the device runs whatever the graph puts there (the hot experts, the
// shared expert), so the host work overlaps it and a decode step stays one device graph.
//
// Host side: an executor takes the posted jobs (poll/complete, spin mode) or is called by the device stream itself
// (runner, hostfunc mode), computes sum over its experts of w * down(swiglu(gate(x), up(x))), writes out, then done.
//
// The CUDA backend implements it; the functions are returned by ggml_backend_reg_get_proc_address under the names in
// the comments below (ggml-cuda.h also declares them).

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum ggml_moe_bridge_wait_mode {
    GGML_MOE_BRIDGE_WAIT_SPIN     = 0, // a bounded device spin on a mapped done flag; a host executor polls the posts
    GGML_MOE_BRIDGE_WAIT_HOSTFUNC = 1, // the device stream calls the runner through a host function node
};

enum ggml_moe_bridge_error {
    GGML_MOE_BRIDGE_ERR_NONE    = 0,
    GGML_MOE_BRIDGE_ERR_TIMEOUT = 1, // a wait gave up; its output and every later one of the graph are zero
    GGML_MOE_BRIDGE_ERR_RUNNER  = 2, // the host could not run a job; its output is zero
};

// job flags (the post op's flags, handed to the host)
#define GGML_MOE_BRIDGE_JOB_TABLE 1 // the graph serves the hot experts of the layer's table on the device: skip them
#define GGML_MOE_BRIDGE_JOB_DMA   2 // [TAG_FN_R4_BRIDGE_DMA] the graph fetches a PCIe share of the job's experts
                                    // (GGML_OP_MOE_HOST_FETCH): the host must publish a plan (job->plan) before it
                                    // computes, and skip the planned experts
#define GGML_MOE_BRIDGE_JOB_HINT  4 // [TAG_FN_L3_CPU_DEVPRED] a hint (ggml_moe_host_hint) follows the post: the predicted
                                    // experts of the next layer arrive in the channel's hint area

// [TAG_FN_R4_BRIDGE_DMA] the largest PCIe share of one job, in experts
#define GGML_MOE_BRIDGE_MAX_FETCH 32

// [TAG_FN_L3_CPU_DEVPRED] the most predicted experts per token a hint carries (ggml_moe_host_hint)
#define GGML_MOE_BRIDGE_MAX_HINT_K 32

// [TAG_FN_R4_BRIDGE_DMA] the PCIe share of one job, in mapped memory: the host writes it, then publishes it
// (ggml_backend_moe_bridge_publish_plan); the device fetch reads it, copies the experts from the registered ring into
// the bank slots and hands slot_ids to the bank chain
struct ggml_moe_bridge_plan {
    int32_t  n_copy;                              // experts to copy, <= params.max_fetch
    int32_t  pad0;
    uint64_t off [GGML_MOE_BRIDGE_MAX_FETCH];     // byte offset of each expert in the ring: up | gate | down, packed
    int32_t  slot[GGML_MOE_BRIDGE_MAX_FETCH];     // its bank slot
    int32_t  slot_ids[1];                         // [n_used, n_tokens]: the bank slot of every routed (slot, token),
                                                  // the zero slot (n_slots) where the expert is not in the bank
};

struct ggml_moe_bridge_params {
    int     device;      // backend device index
    int     n_chan;      // channels, one per bridged MoE layer
    int64_t n_embd;
    int     n_used;      // largest experts per token of a job
    int     max_tokens;  // largest T of a job
    int     wait_mode;   // enum ggml_moe_bridge_wait_mode
    int     timeout_ms;  // a wait gives up after this long if the host has not taken the job
    int     job_max_ms;  // ... or after this long once it has (a slow job: page faults); <= 0: max(timeout_ms, 1000).
                         // Both are clamped to 1500 ms, below the ~2 s driver watchdog.
    bool    stats;       // keep device wait statistics
    int     max_fetch;   // [TAG_FN_R4_BRIDGE_DMA] experts per job a fetch may copy (0: no fetch, <= GGML_MOE_BRIDGE_MAX_FETCH)
    int     hint_k;      // [TAG_FN_L3_CPU_DEVPRED] predicted experts per token a hint may carry (0: no hints,
                         // <= GGML_MOE_BRIDGE_MAX_HINT_K)
};

// one posted job; the pointers stay valid until it is completed
struct ggml_moe_bridge_job {
    int32_t         chan;
    uint32_t        seq;
    int32_t         n_tokens;
    int32_t         n_used;
    int64_t         n_embd;
    int32_t         flags;   // GGML_MOE_BRIDGE_JOB_*
    const float   * x;       // [n_embd, n_tokens]
    const int32_t * ids;     // [n_used, n_tokens]
    const float   * w;       // [n_used, n_tokens]
    float         * out;     // [n_embd, n_tokens], written by the host
    struct ggml_moe_bridge_plan * plan; // [TAG_FN_R4_BRIDGE_DMA] the job's plan area (mapped), NULL without a fetch
};

// [TAG_FN_R4_BRIDGE_DMA] what the device measured for the last job of a channel (mapped memory, written by the device)
struct ggml_moe_bridge_chan_times {
    uint64_t wait_ns;    // the wait for the host's result (0: ready at the first look)
    uint64_t fetch_ns;   // the fetch: from its plan to the last copied byte (0: no fetch, or nothing copied)
    uint32_t n_fetch;    // experts the fetch copied
    uint32_t seq;        // the job these belong to
    uint64_t t_post;     // [TAG_FN_L14_PROBE] device clock (ns): the job's post, the start and the end of the wait
    uint64_t t_wstart;
    uint64_t t_wend;
    uint64_t t_pub;      // [TAG_FN_L14_PROBE] device clock: the post's stamp published
};

struct ggml_moe_bridge_stats {
    uint64_t posts;          // jobs posted by the device (spin mode)
    uint64_t completed;      // jobs completed by the host
    uint64_t waits;          // device waits (stats on)
    uint64_t waits_ready;    // waits that found the result ready at the first look
    uint64_t wait_ns;        // device time spent waiting
    uint32_t error;          // enum ggml_moe_bridge_error
    int32_t  error_chan;
};

struct ggml_moe_bridge;

// returns false when the job failed (its output is then zeroed and the bridge error is set)
typedef bool (*ggml_moe_bridge_runner_t)(const struct ggml_moe_bridge_job * job, void * user_data);

// "ggml_backend_moe_bridge_new":    NULL when the backend cannot build one (no mapped memory, unsupported device)
typedef struct ggml_moe_bridge * (*ggml_backend_moe_bridge_new_t)(const struct ggml_moe_bridge_params * params);
// "ggml_backend_moe_bridge_free":   only when no graph that uses it runs
typedef void     (*ggml_backend_moe_bridge_free_t)(struct ggml_moe_bridge * bridge);
// "ggml_backend_moe_bridge_id":     the value for the ops' bridge argument
typedef int32_t  (*ggml_backend_moe_bridge_id_t)(const struct ggml_moe_bridge * bridge);
// "ggml_backend_moe_bridge_set_runner": hostfunc mode (the device stream calls it); set before the first graph
typedef void     (*ggml_backend_moe_bridge_set_runner_t)(struct ggml_moe_bridge * bridge, ggml_moe_bridge_runner_t runner, void * user_data);
// "ggml_backend_moe_bridge_poll":   spin mode, one executor thread: take the next posted job; false when none is posted
typedef bool     (*ggml_backend_moe_bridge_poll_t)(struct ggml_moe_bridge * bridge, struct ggml_moe_bridge_job * job);
// "ggml_backend_moe_bridge_complete": publish job->out (ok) or zeros and the runner error (!ok), then the done flag
typedef void     (*ggml_backend_moe_bridge_complete_t)(struct ggml_moe_bridge * bridge, const struct ggml_moe_bridge_job * job, bool ok);
// "ggml_backend_moe_bridge_error":  enum ggml_moe_bridge_error, sticky until reset
typedef uint32_t (*ggml_backend_moe_bridge_error_t)(const struct ggml_moe_bridge * bridge);
// "ggml_backend_moe_bridge_reset":  clear the error; only when no graph that uses it runs; false while the host still
//                                   owes a posted job (try again later)
typedef bool     (*ggml_backend_moe_bridge_reset_t)(struct ggml_moe_bridge * bridge);
// "ggml_backend_moe_bridge_get_stats"
typedef void     (*ggml_backend_moe_bridge_get_stats_t)(const struct ggml_moe_bridge * bridge, struct ggml_moe_bridge_stats * stats);

// [TAG_FN_R4_BRIDGE_DMA]
// "ggml_backend_moe_bridge_set_ring": the pinned host ring the fetches copy from (the gen5 DMA ring: host memory of the
//                                     device's host buffer type); false when the device cannot read it (no fetch then).
//                                     Only when no graph that uses the bridge runs.
typedef bool     (*ggml_backend_moe_bridge_set_ring_t)(struct ggml_moe_bridge * bridge, void * host_ptr, size_t size);
// "ggml_backend_moe_bridge_publish_plan": job->plan is written: let the device fetch go (a release store, no CUDA call)
typedef void     (*ggml_backend_moe_bridge_publish_plan_t)(struct ggml_moe_bridge * bridge, const struct ggml_moe_bridge_job * job);
// "ggml_backend_moe_bridge_chan_times": the device's times of the last job of a channel
typedef void     (*ggml_backend_moe_bridge_chan_times_t)(const struct ggml_moe_bridge * bridge, int32_t chan, struct ggml_moe_bridge_chan_times * t);
// [TAG_FN_L14_PROBE] "ggml_backend_moe_bridge_clock_offset": the device clock minus the host's steady_clock, in ns
typedef bool     (*ggml_backend_moe_bridge_clock_offset_t)(const struct ggml_moe_bridge * bridge, int64_t * off_ns);
// "ggml_backend_moe_bridge_release": make every device wait and fetch of this bridge give up at once (exit and crash
//                                     paths: no kernel keeps spinning on a flag nobody will raise); host memory only
typedef void     (*ggml_backend_moe_bridge_release_t)(struct ggml_moe_bridge * bridge);

// [TAG_FN_L3_HOST_EXEC]
// "ggml_backend_moe_bridge_set_watch": spin mode. on: poll zeroes the stamp of every ring entry it takes, so the word of
//                                      the next entry reads 0 until the device posts there. Before the first graph.
typedef void     (*ggml_backend_moe_bridge_set_watch_t)(struct ggml_moe_bridge * bridge, bool on);
// "ggml_backend_moe_bridge_next_post_word": with watch on, a host word that turns nonzero when the next job is posted
//                                           (no CUDA call to read it); valid until the next poll. Executor thread only.
typedef const volatile int32_t * (*ggml_backend_moe_bridge_next_post_word_t)(struct ggml_moe_bridge * bridge);
// "ggml_backend_moe_bridge_test_seed": test hook, spin mode, no graph running and no job owed: the ring counter restarts
//                                      at g (nonzero), so a test crosses the 2^32 wrap of the ring stamps in a few posts
typedef bool     (*ggml_backend_moe_bridge_test_seed_t)(struct ggml_moe_bridge * bridge, uint32_t g);

// [TAG_FN_L16_PLEGATE]
// "ggml_backend_moe_bridge_host_take": a job of chan that no device post makes (the graph only waits for ticket seq, e.g.
//                                      the PLE gate): job gets the channel's out area for n_tokens rows, the job is owed
//                                      (reset waits for it) and its wait now allows job_max_ms. Host memory only, no CUDA
//                                      call, any thread; answer it with ggml_backend_moe_bridge_complete. false: bad chan,
//                                      seq (0) or n_tokens, nothing changed
typedef bool     (*ggml_backend_moe_bridge_host_take_t)(struct ggml_moe_bridge * bridge, int32_t chan, uint32_t seq,
                                                        int32_t n_tokens, struct ggml_moe_bridge_job * job);

// [TAG_FN_L3_CPU_DEVPRED]
// "ggml_backend_moe_bridge_read_hint": the predicted ids the graph wrote after the post of job seq of chan
//                                      (ggml_moe_host_hint): ids [n_tokens][k] by rank; false if that job's hint is not
//                                      (or no longer) there, or does not fit max_ids. Host memory only, no CUDA call.
typedef bool     (*ggml_backend_moe_bridge_read_hint_t)(const struct ggml_moe_bridge * bridge, int32_t chan, uint32_t seq,
                                                        int32_t * ids, int max_ids, int * k, int * n_tokens);

#ifdef __cplusplus
}
#endif
