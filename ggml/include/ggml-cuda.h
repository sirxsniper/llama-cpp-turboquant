#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-moe-bridge.h" // [TAG_MOE_BRIDGE]

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

// ---- TriAttention GPU scoring ----
// Opaque handle for GPU-resident scoring state
typedef struct triattention_gpu_state triattention_gpu_state;

struct triattention_gpu_head_calib {
    const float * q_mean_real;    // [freq_count]
    const float * q_mean_imag;
    const float * q_mean_abs;
    const float * extra_weight;
};

struct triattention_gpu_config {
    uint32_t head_dim;
    uint32_t freq_count;
    uint32_t n_kv_heads;
    uint32_t n_sampled;
    uint32_t n_offsets;
    enum ggml_type k_type;
    bool     need_wht_inv;
    bool     disable_trig;
};

GGML_BACKEND_API triattention_gpu_state * triattention_gpu_init(
    const struct triattention_gpu_config * config,
    const struct triattention_gpu_head_calib * head_calibs,
    const float * omega,
    const float * freq_scale_sq,
    const float * offsets,
    void * stream);

GGML_BACKEND_API void triattention_gpu_score_head(
    triattention_gpu_state * state,
    const void   * k_data_dev,
    uint64_t       n_embd_k_gqa,
    size_t         row_bytes,
    uint32_t       kv_head_idx,
    uint32_t       head_calib_idx,
    const uint32_t * cell_indices_dev,
    const int32_t  * positions_dev,
    uint32_t       n_cells,
    int64_t        round_start,
    int            agg_mode,
    float        * scores_dev,
    void * stream);

GGML_BACKEND_API void triattention_gpu_scores_to_host(
    float * scores_host,
    const float * scores_dev,
    uint32_t n_cells,
    void * stream);

GGML_BACKEND_API void triattention_gpu_upload_cells(
    uint32_t     ** cell_indices_dev,
    int32_t      ** positions_dev,
    const uint32_t * cell_indices_host,
    const int32_t  * positions_host,
    uint32_t        n_cells,
    void * stream);

GGML_BACKEND_API float * triattention_gpu_alloc_scores(uint32_t n_cells, void * stream);
GGML_BACKEND_API void    triattention_gpu_free_dev(void * ptr);
GGML_BACKEND_API void    triattention_gpu_free(triattention_gpu_state * state);

// [TAG_MOE_BRIDGE] MoE host doorbell, see ggml-moe-bridge.h (also returned by get_proc_address as ggml_backend_moe_bridge_*)
GGML_BACKEND_API struct ggml_moe_bridge * ggml_backend_cuda_moe_bridge_new(const struct ggml_moe_bridge_params * params);
GGML_BACKEND_API void     ggml_backend_cuda_moe_bridge_free(struct ggml_moe_bridge * bridge);
GGML_BACKEND_API int32_t  ggml_backend_cuda_moe_bridge_id(const struct ggml_moe_bridge * bridge);
GGML_BACKEND_API void     ggml_backend_cuda_moe_bridge_set_runner(struct ggml_moe_bridge * bridge, ggml_moe_bridge_runner_t runner, void * user_data);
GGML_BACKEND_API bool     ggml_backend_cuda_moe_bridge_poll(struct ggml_moe_bridge * bridge, struct ggml_moe_bridge_job * job);
GGML_BACKEND_API void     ggml_backend_cuda_moe_bridge_complete(struct ggml_moe_bridge * bridge, const struct ggml_moe_bridge_job * job, bool ok);
GGML_BACKEND_API uint32_t ggml_backend_cuda_moe_bridge_error(const struct ggml_moe_bridge * bridge);
GGML_BACKEND_API bool     ggml_backend_cuda_moe_bridge_reset(struct ggml_moe_bridge * bridge);
GGML_BACKEND_API void     ggml_backend_cuda_moe_bridge_get_stats(const struct ggml_moe_bridge * bridge, struct ggml_moe_bridge_stats * stats);
// [TAG_FN_R4_BRIDGE_DMA]
GGML_BACKEND_API bool     ggml_backend_cuda_moe_bridge_set_ring(struct ggml_moe_bridge * bridge, void * host_ptr, size_t size);
GGML_BACKEND_API void     ggml_backend_cuda_moe_bridge_publish_plan(struct ggml_moe_bridge * bridge, const struct ggml_moe_bridge_job * job);
GGML_BACKEND_API void     ggml_backend_cuda_moe_bridge_chan_times(const struct ggml_moe_bridge * bridge, int32_t chan, struct ggml_moe_bridge_chan_times * t);
GGML_BACKEND_API void     ggml_backend_cuda_moe_bridge_release(struct ggml_moe_bridge * bridge);
// [TAG_FN_L3_HOST_EXEC]
GGML_BACKEND_API void     ggml_backend_cuda_moe_bridge_set_watch(struct ggml_moe_bridge * bridge, bool on);
GGML_BACKEND_API const volatile int32_t * ggml_backend_cuda_moe_bridge_next_post_word(struct ggml_moe_bridge * bridge);

// [TAG_FN_L3_HOST_POKE] on: an eager (non-graph) evaluation ends with a stream query, as a graph launch does with
// GGML_CUDA_GRAPH_POKE, and ggml_backend_cuda_stream_poke flushes a backend's stream (helper threads that queue copies
// and go on with other work); off (default): ggml_backend_cuda_stream_poke does nothing
GGML_BACKEND_API void     ggml_backend_cuda_fn_set_poke(int on);
GGML_BACKEND_API void     ggml_backend_cuda_stream_poke(ggml_backend_t backend);

#ifdef  __cplusplus
}
#endif
