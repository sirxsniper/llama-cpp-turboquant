#pragma once

// [TAG_FN_MOE_TRACE] MoE routing profiler and trace. Off unless an env var is set; when off, graphs are unchanged.
//
//   LLAMA_MOE_PROFILE=<file>        per-layer expert activation counts, written at exit and every
//                                   LLAMA_MOE_PROFILE_EVERY ubatches (default 256), text "moeprof v1"
//   LLAMA_MOE_TRACE=<file>          binary record per ubatch: routed ids + weights per layer ("MOET" v1)
//   LLAMA_MOE_TRACE_PRED=1          qwen4exp: also record the next layer's router applied one layer early (top 16)
//   LLAMA_MOE_TRACE_MAX_MB=<n>      stop writing trace records after n MiB (default 4096)
//   LLAMA_MOE_TRACE_SOURCE=<text>   free text for the profile header
//
// Capture: build_moe_ffn marks a contiguous copy of the top-k ids and of the final weights as graph outputs;
// process_ubatch reads them back after compute (async gets + one synchronize). Offline tools: tools/moe-trace/.

#include <cstdint>

struct ggml_tensor;
struct ggml_backend_sched;
struct llama_ubatch;
class  llm_graph_result;

// true when LLAMA_MOE_PROFILE, LLAMA_MOE_TRACE or LLAMA_MOE_HOT_STATS is set (read once)
bool llama_moe_trace_active();

// top-k of the next-layer router prediction, 0 when LLAMA_MOE_TRACE_PRED is not set
int llama_moe_trace_pred_k();

// read the routing tensors of a computed graph and account them
void llama_moe_trace_collect(ggml_backend_sched * sched, const llm_graph_result * res, const llama_ubatch & ubatch, bool is_draft);

// write the profile and flush the trace now (also done at exit)
void llama_moe_trace_flush();

// [TAG_FN_MOE_HOT] hit accounting for the hot set: the hot module registers its host tables here
using llama_moe_trace_hit_fn = void (*)(int il, const int32_t * ids, int64_t n_used, int64_t n_tokens, void * ud);
void llama_moe_trace_set_hit_cb(llama_moe_trace_hit_fn fn, void * ud);
