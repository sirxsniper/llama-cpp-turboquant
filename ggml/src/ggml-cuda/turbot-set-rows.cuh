#pragma once

#include "common.cuh"

// [TAG_TURBOT] CUDA writer of the turbot tiered KV cache, GGML_OP_TURBOT_SET_ROWS (docs/turbot/SPEC.md section 8).
// Host declarations only: this header must not pull turbot-tables.cuh into ggml-cuda.cu (SPEC 7.1).
void ggml_cuda_op_turbot_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_turbot_set_rows_supported(const ggml_tensor * op);

// [TAG_TURBOT_ANY_ROUTE] Kill switch of turbot on other KV geometries in the CUDA backend. GGML_TURBOT_ANY=0 accepts
// only the 4 x 256 geometry (op-params flags 0, writer NG = 8): the FA routing (fattn.cu) and the writer supports
// (turbot-set-rows.cu) are then exactly the pre-change ones. Default on. Read once per translation unit.
static inline bool ggml_cuda_turbot_any_on() {
    static const bool on = [] {
        const char * e = getenv("GGML_TURBOT_ANY");
        return !(e != nullptr && e[0] == '0');
    }();
    return on;
}

// [TAG_TURBOT_ANY_RESOLVE] Whether this build runs turbot for a (head dim, KV heads) layer on CUDA device `device`: a
// turbot geometry (ggml_turbot_geom_flags), Turing+ MMA, GGML_TURBOT_ANY on for anything but 4 x 256 and, at head dim 128,
// the D = 128 instances. Defined in fattn.cu. ggml-cuda.cu exports it to the KV resolver as the registry proc
// "ggml_backend_turbot_supports_geometry" (LLAMA_TURBOT_DEV_GEOM_PROC in src/llama-context.cpp), so a geometry this build
// cannot run steps down to the next KV type instead of reaching a flash attention the backend refuses.
bool ggml_cuda_turbot_geometry_supported(int device, int head_dim, int n_head_kv);
