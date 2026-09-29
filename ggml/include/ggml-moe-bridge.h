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

struct ggml_moe_bridge_params {
    int     device;      // backend device index
    int     n_chan;      // channels, one per bridged MoE layer
    int64_t n_embd;
    int     n_used;      // largest experts per token of a job
    int     max_tokens;  // largest T of a job
    int     wait_mode;   // enum ggml_moe_bridge_wait_mode
    int     timeout_ms;  // a wait gives up after this long if the host has not taken the job
    int     job_max_ms;  // ... or after this long once it has (a slow job: page faults); <= 0: max(timeout_ms, 1000).
                         // Both stay far below the ~2 s driver watchdog.
    bool    stats;       // keep device wait statistics
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

#ifdef __cplusplus
}
#endif
