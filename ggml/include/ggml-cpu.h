#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

    // the compute plan that needs to be prepared for ggml_graph_compute()
    // since https://github.com/ggml-org/ggml/issues/287
    struct ggml_cplan {
        size_t    work_size; // size of work buffer, calculated by `ggml_graph_plan()`
        uint8_t * work_data; // work buffer, to be allocated by caller before calling to `ggml_graph_compute()`

        int n_threads;
        struct ggml_threadpool * threadpool;

        // abort ggml_graph_compute when true
        ggml_abort_callback abort_callback;
        void *              abort_callback_data;

        // use only reference implementations
        bool use_ref;
    };

    // numa strategies
    enum ggml_numa_strategy {
        GGML_NUMA_STRATEGY_DISABLED   = 0,
        GGML_NUMA_STRATEGY_DISTRIBUTE = 1,
        GGML_NUMA_STRATEGY_ISOLATE    = 2,
        GGML_NUMA_STRATEGY_NUMACTL    = 3,
        GGML_NUMA_STRATEGY_MIRROR     = 4,
        GGML_NUMA_STRATEGY_COUNT
    };

    GGML_BACKEND_API void    ggml_numa_init(enum ggml_numa_strategy numa); // call once for better performance on NUMA systems
    GGML_BACKEND_API bool    ggml_is_numa(void); // true if init detected that system has >1 NUMA node

    GGML_BACKEND_API struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value);
    GGML_BACKEND_API struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value);

    GGML_BACKEND_API struct ggml_tensor * ggml_set_i32 (struct ggml_tensor * tensor, int32_t value);
    GGML_BACKEND_API struct ggml_tensor * ggml_set_f32 (struct ggml_tensor * tensor, float value);

    GGML_BACKEND_API int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i);
    GGML_BACKEND_API void    ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value);

    GGML_BACKEND_API int32_t ggml_get_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
    GGML_BACKEND_API void    ggml_set_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value);

    GGML_BACKEND_API float   ggml_get_f32_1d(const struct ggml_tensor * tensor, int i);
    GGML_BACKEND_API void    ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value);

    GGML_BACKEND_API float   ggml_get_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
    GGML_BACKEND_API void    ggml_set_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value);

    GGML_BACKEND_API struct ggml_threadpool *      ggml_threadpool_new           (struct ggml_threadpool_params  * params);
    GGML_BACKEND_API void                          ggml_threadpool_free          (struct ggml_threadpool * threadpool);
    GGML_BACKEND_API int                           ggml_threadpool_get_n_threads (struct ggml_threadpool * threadpool);
    GGML_BACKEND_API void                          ggml_threadpool_pause         (struct ggml_threadpool * threadpool);
    GGML_BACKEND_API void                          ggml_threadpool_resume        (struct ggml_threadpool * threadpool);

    // ggml_graph_plan() has to be called before ggml_graph_compute()
    // when plan.work_size > 0, caller must allocate memory for plan.work_data
    GGML_BACKEND_API struct ggml_cplan ggml_graph_plan(
                  const struct ggml_cgraph * cgraph,
                                       int   n_threads, /* = GGML_DEFAULT_N_THREADS */
                    struct ggml_threadpool * threadpool /* = NULL */ );
    GGML_BACKEND_API enum ggml_status  ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);

    // same as ggml_graph_compute() but the work data is allocated as a part of the context
    // note: the drawback of this API is that you must have ensured that the context has enough memory for the work data
    GGML_BACKEND_API enum ggml_status  ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads);

    //
    // system info
    //

    // x86
    GGML_BACKEND_API int ggml_cpu_has_sse3       (void);
    GGML_BACKEND_API int ggml_cpu_has_ssse3      (void);
    GGML_BACKEND_API int ggml_cpu_has_avx        (void);
    GGML_BACKEND_API int ggml_cpu_has_avx_vnni   (void);
    GGML_BACKEND_API int ggml_cpu_has_avx2       (void);
    GGML_BACKEND_API int ggml_cpu_has_bmi2       (void);
    GGML_BACKEND_API int ggml_cpu_has_f16c       (void);
    GGML_BACKEND_API int ggml_cpu_has_fma        (void);
    GGML_BACKEND_API int ggml_cpu_has_avx512     (void);
    GGML_BACKEND_API int ggml_cpu_has_avx512_vbmi(void);
    GGML_BACKEND_API int ggml_cpu_has_avx512_vnni(void);
    GGML_BACKEND_API int ggml_cpu_has_avx512_bf16(void);
    GGML_BACKEND_API int ggml_cpu_has_amx_int8   (void);
    // ARM
    GGML_BACKEND_API int ggml_cpu_has_neon       (void);
    GGML_BACKEND_API int ggml_cpu_has_arm_fma    (void);
    GGML_BACKEND_API int ggml_cpu_has_fp16_va    (void);
    GGML_BACKEND_API int ggml_cpu_has_dotprod    (void);
    GGML_BACKEND_API int ggml_cpu_has_matmul_int8(void);
    GGML_BACKEND_API int ggml_cpu_has_sve        (void);
    GGML_BACKEND_API int ggml_cpu_get_sve_cnt    (void);  // sve vector length in bytes
    GGML_BACKEND_API int ggml_cpu_has_sme        (void);
    GGML_BACKEND_API int ggml_cpu_has_sme2       (void);
    // other
    GGML_BACKEND_API int ggml_cpu_has_riscv_v    (void);
    GGML_BACKEND_API int ggml_cpu_get_rvv_vlen   (void);  // risc-v vector length in bytes
    GGML_BACKEND_API int ggml_cpu_has_vsx        (void);
    GGML_BACKEND_API int ggml_cpu_has_vxe        (void);
    GGML_BACKEND_API int ggml_cpu_has_wasm_simd  (void);
    GGML_BACKEND_API int ggml_cpu_has_llamafile  (void);

    // Internal types and functions exposed for tests and benchmarks

    typedef void (*ggml_vec_dot_t)  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT x, size_t bx,
                                       const void * GGML_RESTRICT y, size_t by, int nrc);

    struct ggml_type_traits_cpu {
        ggml_from_float_t        from_float;
        ggml_vec_dot_t           vec_dot;
        enum ggml_type           vec_dot_type;
        int64_t                  nrows; // number of rows to process simultaneously
    };

    GGML_BACKEND_API const struct ggml_type_traits_cpu * ggml_get_type_traits_cpu(enum ggml_type type);

    GGML_BACKEND_API void ggml_cpu_init(void);

    // [TAG_FN_CPU_SWITCHES] switches of the CPU expert path (Flash-Next work package WP-CPU), all off by default.
    // ggml_cpu_init() reads each one once from its environment variable (ggml_cpu_fn_switch_env). Tests and
    // benchmarks may change them with ggml_cpu_fn_set_switch(), but only while no graph is being computed.
    enum ggml_cpu_fn_switch {
        GGML_CPU_FN_APPLY_ONCE = 0, // GGML_CPU_APPLY_ONCE=1: skip re-applying an unchanged thread priority / affinity
        GGML_CPU_FN_Q5_1_AVX512,    // GGML_CPU_Q5_1_AVX512=1: AVX-512 q5_1 x q8_1 dot product (bitwise equal to AVX2)
        GGML_CPU_FN_MMID_MR,        // GGML_CPU_MMID_MR=1: MUL_MAT_ID dots up to 4 tokens per decoded weight row
                                    //   (bitwise equal); =2: the same with the 256-bit bodies only
        GGML_CPU_FN_MOE_FUSE,       // GGML_CPU_MOE_FUSE=1: the MoE split up / gate / swiglu / down as one op with one
                                    //   barrier, for up to 16 tokens (bitwise equal to the unfused nodes)
        GGML_CPU_FN_VNNI,           // [TAG_FN_R4_VNNI] GGML_CPU_VNNI=1: the 512-bit multi-row bodies of q4_K, q5_1 and
                                    //   iq4_nl use AVX512-VNNI (vpdpwssd / vpdpbusd) when the CPU has it, also in a build
                                    //   that was not compiled for it (MSVC: no /arch flag defines __AVX512VNNI__); it turns
                                    //   the multi-row path on as GGML_CPU_MMID_MR=1 does (bitwise equal to the AVX2 dots)
        GGML_CPU_FN_SWITCH_COUNT,
    };

    GGML_BACKEND_API int          ggml_cpu_fn_get_switch(enum ggml_cpu_fn_switch sw);
    // [TAG_FN_R4_VNNI] true if this build has the VNNI bodies and the CPU runs AVX512-VNNI (GGML_CPU_VNNI can act)
    GGML_BACKEND_API bool         ggml_cpu_fn_vnni_available(void);
    GGML_BACKEND_API void         ggml_cpu_fn_set_switch(enum ggml_cpu_fn_switch sw, int value);
    GGML_BACKEND_API const char * ggml_cpu_fn_switch_env(enum ggml_cpu_fn_switch sw); // environment variable name

    // [TAG_FN_CPU_MMID_MR] test / benchmark hook: s[c*bs + r] = dot(row r of vx (row stride bx), column vy[c]) for
    // r < nr and c < nc <= 4, the columns already in the type's vec_dot_type, with the multi-row x multi-token kernel of
    // this build (GGML_CPU_MMID_MR picks its body; the kernel runs whatever the switch's on/off state). Returns false
    // when this build has none for the type.
    GGML_BACKEND_API bool ggml_cpu_fn_vec_dot_mr(enum ggml_type type, int n, float * s, size_t bs, const void * vx, size_t bx,
                                                 int nr, const void * const * vy, int nc);

    // [TAG_FN_CPU_MOE_FUSE] a persistent CPU worker pool that runs the host experts of one MoE layer for a few tokens,
    // outside any graph (the CPU side of a GPU/CPU doorbell). It computes what the fused graph op computes: for every
    // (slot, token) whose expert passes the table, down(swiglu(gate(x), up(x))) with the CPU kernels of this build.
    struct ggml_cpu_moe_layer {              // one MoE layer with host-resident experts, fixed for the model's life
        const struct ggml_tensor * up;       // [n_embd, n_ff, n_expert]
        const struct ggml_tensor * gate;     // [n_embd, n_ff, n_expert]
        const struct ggml_tensor * down;     // [n_ff, n_embd, n_expert]
        const int32_t * table;               // [n_expert] or NULL; expert e is computed only if table[e] == table_miss
        int32_t         table_miss;          // the moe-cache host_table uses n_slots
    };

    struct ggml_cpu_moe_job {
        const struct ggml_cpu_moe_layer * layer;
        int32_t n_tokens;                    // 1..16
        int32_t n_used;                      // experts per token
        const float   * x;                   // [n_embd, n_tokens] f32
        const int32_t * ids;                 // [n_used, n_tokens]
        const float   * w;                   // [n_used, n_tokens] or NULL
        float         * out;                 // w != NULL: [n_embd, n_tokens] = sum over the computed slots, in slot order, of w * expert(x)
                                             // w == NULL: [n_embd, n_used, n_tokens], rows of slots not computed here zeroed
    };

    struct ggml_cpu_moe_pool_params {
        int  n_threads;                      // workers including the calling thread (worker 0)
        bool cpumask[GGML_MAX_N_THREADS];    // worker k >= 1 is pinned to the k-th CPU of the mask; all false: the
                                             // k-th physical core (SMT siblings skipped); the caller keeps its affinity
        int  prio;                           // enum ggml_sched_priority
        int  spin_us;                        // idle workers spin this long, then sleep until the next job
        bool pin_caller;                     // [TAG_MOE_BRIDGE] also pin the calling thread (worker 0) to the first CPU
                                             // of the list, with prio; create the pool on the thread that runs the jobs
        bool skip_first_core;                // [TAG_MOE_BRIDGE] leave the first CPU of the list free (the main thread
                                             // of a ggml threadpool sits there) and use at most the rest, one thread each
    };

    struct ggml_cpu_moe_pool;

    GGML_BACKEND_API struct ggml_cpu_moe_pool_params ggml_cpu_moe_pool_params_default(int n_threads);
    GGML_BACKEND_API struct ggml_cpu_moe_pool *      ggml_cpu_moe_pool_new (const struct ggml_cpu_moe_pool_params * p);
    GGML_BACKEND_API void                           ggml_cpu_moe_pool_free(struct ggml_cpu_moe_pool * pool);
    // blocking; one job at a time per pool. GGML_STATUS_FAILED for a job the fused kernel does not take (types,
    // shapes, n_tokens > 16): the caller then runs the layer another way
    GGML_BACKEND_API enum ggml_status               ggml_cpu_moe_run      (struct ggml_cpu_moe_pool * pool, const struct ggml_cpu_moe_job * job);

    // [TAG_MOE_BRIDGE] park: idle workers sleep now instead of spinning (other CPU work is about to run on their
    // cores); wake: sleeping workers spin again for spin_us (a job is about to come). A job un-parks the pool.
    GGML_BACKEND_API void                           ggml_cpu_moe_pool_park(struct ggml_cpu_moe_pool * pool);
    GGML_BACKEND_API void                           ggml_cpu_moe_pool_wake(struct ggml_cpu_moe_pool * pool);
    // [TAG_MOE_BRIDGE] true if ggml_cpu_moe_run takes jobs of this layer (weight types and shapes)
    GGML_BACKEND_API bool                           ggml_cpu_moe_layer_supported(const struct ggml_cpu_moe_layer * layer);

    // [TAG_FN_R2_BRIDGE_PF] next-layer prefetch, for the idle time between two jobs of a GPU/CPU doorbell: the workers
    // predict the experts of the next MoE layer (top-k of its router logits on the current layer's input, per token),
    // drop the ones its table serves elsewhere (the hot set), and pull the rest's weights into the CPU caches, every
    // worker the pieces it computes in a job of those experts. Asynchronous: returns once the workers have the job; the
    // caller is not one of them (it keeps polling for the next job). They stop at their next piece after
    // ggml_cpu_moe_prefetch_stop (any thread); ggml_cpu_moe_run, the next prefetch and ggml_cpu_moe_pool_free stop and
    // wait for them first. Values are never touched. Needs >= 2 pool threads.
    struct ggml_cpu_moe_prefetch_job {
        const struct ggml_cpu_moe_layer * layer;  // the predicted layer; its table marks the experts not to fetch
        const ggml_fp16_t * router;               // [n_expert][n_embd]: the layer's router rows in f16
        int32_t       n_tokens;                   // 1..16
        const float * x;                          // [n_tokens][n_embd] f32, copied by the call
        int32_t       k;                          // experts predicted per token
        int32_t       mode;                       // 0: real loads (they wait for the data), 1: software prefetches
    };
    GGML_BACKEND_API enum ggml_status ggml_cpu_moe_prefetch     (struct ggml_cpu_moe_pool * pool, const struct ggml_cpu_moe_prefetch_job * job);
    GGML_BACKEND_API void             ggml_cpu_moe_prefetch_stop(struct ggml_cpu_moe_pool * pool);
    // [TAG_FN_R2_BRIDGE_PF] solo: the caller stays out of ggml_cpu_moe_run's compute (it waits) and the workers split the
    // job, as they split a prefetch, so every worker computes exactly the pieces it pulled into its own caches. The values
    // are the same (the kernel's do not depend on the thread count). Needs >= 2 pool threads; caller thread only.
    GGML_BACKEND_API void             ggml_cpu_moe_pool_set_solo(struct ggml_cpu_moe_pool * pool, bool solo);
    // totals since the pool was made: prefetch jobs, jobs stopped before their end, bytes covered, experts predicted
    // (not resident)
    GGML_BACKEND_API void             ggml_cpu_moe_prefetch_stats(struct ggml_cpu_moe_pool * pool, uint64_t * jobs,
                                                                  uint64_t * stopped, uint64_t * bytes, uint64_t * experts);
    // [TAG_FN_L3_HOST_EXEC] the caller's own part of the posted prefetch (not solo): once the workers have the predicted
    // experts, pull the pieces that the caller (thread 0) computes in a job of them into its own caches. Returns at the
    // end, or as soon as *stop != 0 (checked while it waits for the list and every 16 KiB). Caller thread only, after
    // ggml_cpu_moe_prefetch and before the next job. Returns the bytes covered.
    GGML_BACKEND_API size_t           ggml_cpu_moe_prefetch_caller(struct ggml_cpu_moe_pool * pool, const volatile int32_t * stop);
    // [TAG_FN_L3_HOST_EXEC] totals: caller prefetches started, stopped before their end, bytes covered
    GGML_BACKEND_API void             ggml_cpu_moe_prefetch_caller_stats(struct ggml_cpu_moe_pool * pool, uint64_t * calls,
                                                                         uint64_t * stopped, uint64_t * bytes);

    // [TAG_FN_CPU_MOE_FUSE] test / benchmark hook: how many fused MoE graph ops have run in this process
    GGML_BACKEND_API uint64_t ggml_cpu_fn_moe_fused_calls(void);

    //
    // CPU backend
    //

    GGML_BACKEND_API ggml_backend_t ggml_backend_cpu_init(void);

    GGML_BACKEND_API bool ggml_backend_is_cpu                (ggml_backend_t backend);
    GGML_BACKEND_API void ggml_backend_cpu_set_n_threads     (ggml_backend_t backend_cpu, int n_threads);
    GGML_BACKEND_API void ggml_backend_cpu_set_threadpool    (ggml_backend_t backend_cpu, ggml_threadpool_t threadpool);
    GGML_BACKEND_API void ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, void * abort_callback_data);

    GGML_BACKEND_API void ggml_backend_cpu_set_use_ref(ggml_backend_t backend_cpu, bool use_ref);

    GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cpu_reg(void);

    GGML_BACKEND_API void ggml_cpu_fp32_to_fp32(const float *,       float *, int64_t);
    GGML_BACKEND_API void ggml_cpu_fp32_to_i32 (const float *,     int32_t *, int64_t);
    GGML_BACKEND_API void ggml_cpu_fp32_to_fp16(const float *, ggml_fp16_t *, int64_t);
    GGML_BACKEND_API void ggml_cpu_fp16_to_fp32(const ggml_fp16_t *, float *, int64_t);
    GGML_BACKEND_API void ggml_cpu_fp32_to_bf16(const float *, ggml_bf16_t *, int64_t);
    GGML_BACKEND_API void ggml_cpu_bf16_to_fp32(const ggml_bf16_t *, float *, int64_t);

#ifdef __cplusplus
}
#endif
