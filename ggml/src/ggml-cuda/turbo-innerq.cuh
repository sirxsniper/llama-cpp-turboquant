#pragma once

// TurboQuant InnerQ per-channel equalization — cross-TU shared state
// The host-side state lives in turbo-innerq.cu; device-side state is per-TU
// in turbo-quant.cuh (only set-rows.cu needs device access).

#define INNERQ_MAX_CHANNELS 128

#if defined(_WIN32)
#  ifdef ggml_cuda_EXPORTS
#    define TURBO_INNERQ_API __declspec(dllexport)
#  else
#    define TURBO_INNERQ_API __declspec(dllimport)
#  endif
#else
#  define TURBO_INNERQ_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Host-side shared state (defined in turbo-innerq.cu)
extern TURBO_INNERQ_API bool  g_innerq_finalized;
extern TURBO_INNERQ_API float g_innerq_scale_inv_host[INNERQ_MAX_CHANNELS];

// Called from set-rows.cu after InnerQ finalization to publish scale_inv
void turbo_innerq_publish(const float * scale_inv, int group_size);

// Called from llama-kv-cache.cpp (or equivalent) to check if tensor needs update
// Returns true if there are new scale_inv values to upload
TURBO_INNERQ_API bool turbo_innerq_needs_tensor_update(void);

// Called after tensor update to clear the flag
TURBO_INNERQ_API void turbo_innerq_mark_tensor_updated(void);

#ifdef __cplusplus
}
#endif
