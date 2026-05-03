#pragma once

// TurboQuant InnerQ host-side shared state.
//
// These symbols live in ggml-base.dll (NOT in ggml-cuda.dll) so they remain
// resolvable even when GGML_BACKEND_DL=ON makes ggml-cuda.dll a runtime-loaded
// plugin. The InnerQ data is plain CPU floats and bool flags — no CUDA needed.
//
// Consumers:
//   - ggml-cuda.dll  (set-rows.cu / turbo-quant.cuh) — calls publish() after
//     CUDA-side calibration completes
//   - llama.dll      (llama-kv-cache.cpp)            — checks the update flag
//     each token and uploads the updated tensor when needed

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

#define INNERQ_MAX_CHANNELS 128

GGML_API bool  g_innerq_finalized;
GGML_API float g_innerq_scale_inv_host[INNERQ_MAX_CHANNELS];

// Called from CUDA code after InnerQ calibration finishes.
GGML_API void turbo_innerq_publish(const float * scale_inv, int group_size);

// Called from the consumer (llama-kv-cache) to check if the device-side scale
// tensor needs to be re-uploaded with new values.
GGML_API bool turbo_innerq_needs_tensor_update(void);

// Called by the consumer after it uploads the updated tensor, to clear the flag.
GGML_API void turbo_innerq_mark_tensor_updated(void);

#ifdef __cplusplus
}
#endif
