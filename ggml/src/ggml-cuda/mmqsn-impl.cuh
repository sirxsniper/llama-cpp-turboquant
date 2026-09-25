#pragma once

// [TAG_MMQSN] MMQ with a cp.async weight ring, for 5..16 src1 columns of Q4_K / Q5_K / Q6_K weights.
//
// Why: at 8-16 columns MMQ reads the weights at only 1.12-1.18 TB/s from DRAM although its compute runs at 1.7-2.1
// TB/s-equivalent from L2: its K-loop (mmq.cuh, mul_mat_q_process_tile) has nothing in flight while it computes. This
// kernel keeps MMQ's tile (I = 128, J = 8/16, 8 warps), unpack (load_tiles), vec_dot, write_back, stream-k partition
// and fixup arithmetic, and replaces only the K-loop:
//   - the raw weight bytes of the next 2 steps are copied to shared memory with cp.async (2-slot ring),
//   - the q8_1 tile of the next step is loaded into registers while vec_dot runs,
//   - 2 barriers per 256-value step instead of 4,
//   - PDL launches; the weight copies are issued before the grid-dependency wait (weights buffers only),
//   - optional L2 prefetch pf_dist steps ahead (pf_run steps per run) for more bytes in flight.
// The output is bit-identical to MMQ: same q8_1 bytes, same x tile, same accumulation order, same stream-k partition
// and fixup order (GGML_CUDA_MMQSN_CHECK=1 compares in-process, mmqsn.cu).
//
// Loop modes: MMQSN_RING (default), MMQSN_PF (stock load_tiles from global memory plus L2 prefetch, no ring) and
// MMQSN_STREAM (the ring without unpack/vec_dot/write_back: measures the ceiling of this access pattern, output is
// garbage).

#include "common.cuh"
#include "cp-async.cuh"
#include "mmq.cuh"

#include <cstdint>

enum mmqsn_mode {
    MMQSN_RING   = 0,
    MMQSN_PF     = 1,
    MMQSN_STREAM = 2,
};

#define MMQSN_I          128     // rows per tile (MMQ's I on NVIDIA for Q4_K/Q5_K/Q6_K)
#define MMQSN_NTHREADS   256     // threads per block (MMQ's nthreads)
#define MMQSN_X_STRIDE    76     // ints per tile_x row (Q8_1 and Q6_K MMA layouts)
#define MMQSN_SMEM_MAX   101376  // opt-in shared memory per block on sm_120 (99 KB)
#define MMQSN_Y_PAD      16      // extra block_q8_1_mmq after the q8_1 data: J-wide y reads of the last K step

// Raw weight bytes per row per 256-value step (bs) and the slot pitch in shared memory (pitch). Q6_K blocks are 210 B
// and only 2-byte aligned, so a Q6_K slot row holds the covering 16-byte window [p & ~15, (p & ~15) + 224); the block
// starts at (p & 15) in it and (p & 15) + 210 <= 224 always holds.
template <ggml_type type> struct mmqsn_raw;
template <> struct mmqsn_raw<GGML_TYPE_Q4_K> { static constexpr int bs = 144; static constexpr int pitch = 144; };
template <> struct mmqsn_raw<GGML_TYPE_Q5_K> { static constexpr int bs = 176; static constexpr int pitch = 176; };
template <> struct mmqsn_raw<GGML_TYPE_Q6_K> { static constexpr int bs = 210; static constexpr int pitch = 224; };

static_assert(sizeof(block_q4_K) == 144, "mmqsn: unexpected block_q4_K size");
static_assert(sizeof(block_q5_K) == 176, "mmqsn: unexpected block_q5_K size");
static_assert(sizeof(block_q6_K) == 210, "mmqsn: unexpected block_q6_K size");
static_assert(sizeof(block_q8_1_mmq) == 4*MMQ_TILE_Y_K, "mmqsn: unexpected block_q8_1_mmq size");

// Dynamic shared memory, in this order (all offsets multiples of 16 bytes):
//   ids[J] | tile_y[2*J*MMQ_TILE_Y_K] (both y halves) | tile_x[MMQSN_I*MMQSN_X_STRIDE] | RING/STREAM: 2 slots of I*pitch
template <ggml_type type, int J, int mode>
static constexpr __host__ __device__ size_t mmqsn_nbytes_shared() {
    return (size_t) J*sizeof(int) + (size_t) 2*J*MMQ_TILE_Y_K*sizeof(int) + (size_t) MMQSN_I*MMQSN_X_STRIDE*sizeof(int) +
        (mode == MMQSN_PF ? (size_t) 0 : (size_t) 2*MMQSN_I*mmqsn_raw<type>::pitch);
}

static_assert(mmqsn_nbytes_shared<GGML_TYPE_Q4_K, 16, MMQSN_RING>() ==  80448, "mmqsn: shared memory layout");
static_assert(mmqsn_nbytes_shared<GGML_TYPE_Q5_K, 16, MMQSN_RING>() ==  88640, "mmqsn: shared memory layout");
static_assert(mmqsn_nbytes_shared<GGML_TYPE_Q6_K, 16, MMQSN_RING>() == 100928, "mmqsn: shared memory layout");
static_assert(mmqsn_nbytes_shared<GGML_TYPE_Q6_K,  8, MMQSN_RING>() ==  98592, "mmqsn: shared memory layout");
static_assert(mmqsn_nbytes_shared<GGML_TYPE_Q6_K, 16, MMQSN_PF>()   ==  43584, "mmqsn: shared memory layout");
static_assert(mmqsn_nbytes_shared<GGML_TYPE_Q6_K, 16, MMQSN_STREAM>() <= MMQSN_SMEM_MAX, "mmqsn: shared memory");
static_assert(mmqsn_nbytes_shared<GGML_TYPE_Q6_K,  8, MMQSN_PF>() % 16 == 0, "mmqsn: shared memory alignment");
static_assert((8*sizeof(int) + 2*8*MMQ_TILE_Y_K*sizeof(int)) % 16 == 0, "mmqsn: tile_x alignment (J = 8)");

// Runtime form of mmqsn_nbytes_shared for the host (routing check).
static size_t mmqsn_nbytes_shared_rt(const ggml_type type, const int J, const int mode) {
    const int pitch = type == GGML_TYPE_Q4_K ? mmqsn_raw<GGML_TYPE_Q4_K>::pitch :
                      type == GGML_TYPE_Q5_K ? mmqsn_raw<GGML_TYPE_Q5_K>::pitch : mmqsn_raw<GGML_TYPE_Q6_K>::pitch;
    return (size_t) J*sizeof(int) + (size_t) 2*J*MMQ_TILE_Y_K*sizeof(int) + (size_t) MMQSN_I*MMQSN_X_STRIDE*sizeof(int) +
        (mode == MMQSN_PF ? (size_t) 0 : (size_t) 2*MMQSN_I*pitch);
}

// ------------------------------------------------------------------------------------------------------------------
// Device helpers
// ------------------------------------------------------------------------------------------------------------------

// Per-block step list: step g of this block is K step kb of output tile `tile`.
struct mmqsn_steps {
    int  kbc0;      // stream-k: first continuous k-block index of this block
    int  nsteps;    // steps of this block
    int  persist;   // 1: persistent tiling (tiles blockIdx.x + q*gridDim.x, each summed start to end)
    uint3 nkb_fd;   // fastdiv values of nkb = ne00/256
};

static __device__ __forceinline__ int2 mmqsn_step(const mmqsn_steps & s, const int g) {
    if (s.persist) {
        const int q  = fastdiv((uint32_t) g, s.nkb_fd);
        return make_int2(blockIdx.x + q*gridDim.x, g - q*(int) s.nkb_fd.z);
    }
    const int t    = s.kbc0 + g;
    const int tile = fastdiv((uint32_t) t, s.nkb_fd);
    return make_int2(tile, t - tile*(int) s.nkb_fd.z);
}

// Issues the cp.async copies of step g into shared-memory slot s0 (a 32-bit shared address): 128 rows x pitch bytes,
// 16 bytes per copy, spread over the 256 threads.
template <ggml_type type>
static __device__ __forceinline__ void mmqsn_issue(
        const char * x, const mmqsn_steps & s, const int g, const int64_t row_bytes, const unsigned int s0,
        const int l2hint, const int tid) {
    constexpr int bs    = mmqsn_raw<type>::bs;
    constexpr int pitch = mmqsn_raw<type>::pitch;
    constexpr int nch   = pitch/16;
    constexpr int nc    = MMQSN_I*nch;

    const int2 tk = mmqsn_step(s, g);
    const char * xt = x + (int64_t) tk.x*MMQSN_I*row_bytes + (int64_t) tk.y*bs;   // row 0 of the tile, block kb

#pragma unroll
    for (int c0 = 0; c0 < nc; c0 += MMQSN_NTHREADS) {
        const int c = c0 + tid;
        if (c0 + MMQSN_NTHREADS > nc && c >= nc) {
            break;
        }
        const int i  = c / nch;
        const int ch = c - i*nch;
        const char * p = xt + i*row_bytes;
        const char * src;
        if constexpr (type == GGML_TYPE_Q6_K) {
            src = (const char *) ((uintptr_t) p & ~(uintptr_t) 15) + 16*ch;
        } else {
            src = p + 16*ch;
        }
        const unsigned int dst = s0 + i*pitch + 16*ch;
        if (l2hint) {
            cp_async_cg_16<256>(dst, src);
        } else {
            cp_async_cg_16<0>(dst, src);
        }
    }
}

// L2 prefetch of target step t: thread i < 128 covers row i. A run starts where kb % pf_run == 0 (or at the first target
// t_first) and covers up to pf_run steps of the row, clipped at the tile end and at the block's last step.
template <ggml_type type>
static __device__ __forceinline__ void mmqsn_prefetch(
        const char * x, const mmqsn_steps & s, const int t, const int t_first, const int pf_run, const int64_t row_bytes,
        const int tid) {
    constexpr int bs = mmqsn_raw<type>::bs;
    if (t >= s.nsteps || tid >= MMQSN_I) {
        return;
    }
    const int2 tk = mmqsn_step(s, t);
    const int  kr = tk.y % pf_run;
    if (kr != 0 && t != t_first) {
        return;
    }
    int run = pf_run - kr;
    run = min(run, (int) s.nkb_fd.z - tk.y);
    run = min(run, s.nsteps - t);
    const char * p  = x + ((int64_t) tk.x*MMQSN_I + tid)*row_bytes + (int64_t) tk.y*bs;
    const char * a0 = (const char *) ((uintptr_t) p & ~(uintptr_t) 15);
    const uint32_t n = ((uint32_t) (p + run*bs - a0) + 15u) & ~15u;
    ggml_cuda_prefetch_l2(a0, n);
}

// q8_1 tile of step (kb) into registers: half h is J*MMQ_TILE_Y_K contiguous ints at y + ne11*(2*kb + h)*MMQ_TILE_Y_K
// (MMQ's addressing, mmq.cuh mul_mat_q_process_tile). Columns >= ne11 read the next block or the MMQSN_Y_PAD padding,
// exactly as MMQ does; their results are never stored.
template <int J>
static __device__ __forceinline__ void mmqsn_load_y(
        const int * y, const int ne11, const int kb, int (&yr)[2][(J*MMQ_TILE_Y_K + MMQSN_NTHREADS - 1)/MMQSN_NTHREADS],
        const int tid) {
    constexpr int ny  = J*MMQ_TILE_Y_K;
    constexpr int nyr = (ny + MMQSN_NTHREADS - 1)/MMQSN_NTHREADS;
    const int * by0 = y + (int64_t) ne11*(2*kb)*MMQ_TILE_Y_K;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int * by = by0 + (int64_t) h*ne11*MMQ_TILE_Y_K;
#pragma unroll
        for (int r = 0; r < nyr; ++r) {
            const int l = tid + r*MMQSN_NTHREADS;
            if (r < nyr - 1 || ny % MMQSN_NTHREADS == 0 || l < ny) {
                yr[h][r] = by[l];
            }
        }
    }
}

template <int J>
static __device__ __forceinline__ void mmqsn_store_y(
        int * tile_y, int (&yr)[2][(J*MMQ_TILE_Y_K + MMQSN_NTHREADS - 1)/MMQSN_NTHREADS], const int tid) {
    constexpr int ny  = J*MMQ_TILE_Y_K;
    constexpr int nyr = (ny + MMQSN_NTHREADS - 1)/MMQSN_NTHREADS;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
#pragma unroll
        for (int r = 0; r < nyr; ++r) {
            const int l = tid + r*MMQSN_NTHREADS;
            if (r < nyr - 1 || ny % MMQSN_NTHREADS == 0 || l < ny) {
                tile_y[h*ny + l] = yr[h][r];
            }
        }
    }
}

// Q6_K block of tile row i in a staged slot: the window of row i starts 16-byte aligned, the block at
// (row_lo + i*row_step) & 15 in it (row_lo = block address of row 0 mod 16, row_step = row stride in bytes mod 16).
static __device__ __forceinline__ const block_q6_K * mmqsn_q6_K_row(
        const char * slot, const int i, const uint32_t row_lo, const uint32_t row_step) {
    return (const block_q6_K *) (slot + i*mmqsn_raw<GGML_TYPE_Q6_K>::pitch + ((row_lo + (uint32_t) i*row_step) & 15u));
}

// Copy of the MMA branch of ggml_cuda_mmq_load_tiles_q6_K<GGML_TYPE_Q6_K, J, false, false> (mmq-load-tiles.cuh) reading
// a staged slot. Only the three bxi lines differ (mmqsn_q6_K_row); the (threadIdx.x % 4)/4 term of the scales loop is
// always 0. The x tile is identical to MMQ's.
template <int J>
static __device__ __forceinline__ void mmqsn_load_tiles_q6_K_staged(
        const char * __restrict__ x, int * __restrict__ x_tile, const uint32_t row_lo, const uint32_t row_step) {
    constexpr ggml_type type     = GGML_TYPE_Q6_K;
    constexpr bool      fallback = false;
    constexpr bool      has_ids  = false;
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback, has_ids) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback, has_ids);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback, has_ids);

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
    int   * x_sc = (int   *) (x_df + MMQ_TILE_NE_K/QI6_K);

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR6_K);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        const block_q6_K * bxi = mmqsn_q6_K_row(x, i, row_lo, row_step);

        const int ql = get_int_b2(bxi->ql, txi);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_b2(bxi->qh, (QI6_K/4) * (txi / (QI6_K/2)) + txi % (QI6_K/4));
        const int qh0 = ((qh >> ((txi & 0x08) >> 2)) << 4) & 0x30303030;
        const int qh1 =  (qh >> ((txi & 0x08) >> 2))       & 0x30303030;

        const int kq0 = 2*txi - txi % (QI6_K/2) + 0;
        const int kq1 = 2*txi - txi % (QI6_K/2) + QI6_K/2;

        x_qs[i*sram_stride + kq0] = __vsubss4(ql0 | qh0, 0x20202020);
        x_qs[i*sram_stride + kq1] = __vsubss4(ql1 | qh1, 0x20202020);
    }

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps*warp_size) {
        int i = (i0 + threadIdx.y*warp_size + threadIdx.x) % I;

        const block_q6_K * bxi = mmqsn_q6_K_row(x, i, row_lo, row_step);

        x_df[i*sram_stride]                     = bxi->d;
    }

    constexpr int rows_per_warp = warp_size / 4;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps*rows_per_warp) {
        int i = (i0 + threadIdx.y*rows_per_warp + threadIdx.x/(MMQ_TILE_NE_K/8)) % I;

        const block_q6_K * bxi = mmqsn_q6_K_row(x, i, row_lo, row_step);

        x_sc[i*sram_stride + threadIdx.x%4] = get_int_b2(bxi->scales, threadIdx.x % (MMQ_TILE_NE_K/8));
    }
}

// Unpacks the staged step (tile tk.x, K step tk.y) from slot into tile_x, MMQ's layout.
template <ggml_type type, int J>
static __device__ __forceinline__ void mmqsn_unpack(
        const char * slot, int * tile_x, const char * x, const int2 tk, const int64_t row_bytes) {
    if constexpr (type == GGML_TYPE_Q6_K) {
        const uint32_t row_lo   = (uint32_t) (uintptr_t) (x + (int64_t) tk.x*MMQSN_I*row_bytes + (int64_t) tk.y*mmqsn_raw<type>::bs) & 15u;
        const uint32_t row_step = (uint32_t) row_bytes & 15u;
        mmqsn_load_tiles_q6_K_staged<J>(slot, tile_x, row_lo, row_step);
    } else if constexpr (type == GGML_TYPE_Q5_K) {
        GGML_UNUSED(x); GGML_UNUSED(tk); GGML_UNUSED(row_bytes);
        ggml_cuda_mmq_load_tiles_q5_K<type, J, false, false>(slot, tile_x, 0, MMQSN_I - 1, 1);
    } else {
        static_assert(type == GGML_TYPE_Q4_K, "mmqsn: unsupported type");
        GGML_UNUSED(x); GGML_UNUSED(tk); GGML_UNUSED(row_bytes);
        ggml_cuda_mmq_load_tiles_q4_K<type, J, false, false>(slot, tile_x, 0, MMQSN_I - 1, 1);
    }
}

// ------------------------------------------------------------------------------------------------------------------
// Kernels
// ------------------------------------------------------------------------------------------------------------------

// x: weights; y: q8_1 in MMQ's block layout; tmp: stream-k fixup buffer (grid*J*I floats, only when a fixup runs).
// persist = 1 only where MMQ would use its tiling grid (then grid <= ntiles). x_pf = 1: the weight copies (or the PF
// mode prefetches) of the first steps are issued before the PDL wait; only valid for weights buffers.
template <ggml_type type, int J, int mode>
__launch_bounds__(MMQSN_NTHREADS, 1)
static __global__ void mul_mat_qsn(
        const char * x_ptr, const int * y_ptr, float * dst_ptr, float * tmp_ptr, const uint3 nkb_fd,
        const int nrows_x, const int ne11, const int stride_row_x, const int stride_col_dst, const int ntiles,
        const int persist, const int x_pf, const int pf_dist, const int pf_run, const int l2hint) {
#if defined(TURING_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    // PDL and __restrict__ are mutually exclusive (common.cuh): plain or GGML_CUDA_RESTRICT locals only.
    const char * GGML_CUDA_RESTRICT x   = x_ptr;
    const int  * GGML_CUDA_RESTRICT y   = y_ptr;
    float      * GGML_CUDA_RESTRICT dst = dst_ptr;
    float      * GGML_CUDA_RESTRICT tmp = tmp_ptr;

    constexpr int qk              = ggml_cuda_type_traits<type>::qk;
    constexpr int I               = MMQSN_I;
    constexpr int blocks_per_iter = MMQ_ITER_K / qk;
    constexpr int ny              = J*MMQ_TILE_Y_K;
    constexpr int nyr             = (ny + MMQSN_NTHREADS - 1)/MMQSN_NTHREADS;
    constexpr int bs              = mmqsn_raw<type>::bs;
    static_assert(ggml_cuda_mmq_get_I(type, J, false, false) == I, "mmqsn: MMQ tile height must be 128");
    static_assert(ggml_cuda_mmq_get_nthreads(type, J, false, false) == MMQSN_NTHREADS, "mmqsn: MMQ must use 256 threads");
    static_assert(ggml_cuda_mmq_get_K_vram(type, J, false, false) == MMQ_ITER_K, "mmqsn: one 256-value step per iteration");
    static_assert(ggml_cuda_mmq_get_stream_k(type, J, false, false), "mmqsn: MMQ must use stream-k");
    static_assert(ggml_cuda_mmq_get_sram_stride(type, J, false, false) == MMQSN_X_STRIDE, "mmqsn: tile_x stride");
    static_assert(qk == MMQ_ITER_K && blocks_per_iter == 1, "mmqsn: K-quants only");
    static_assert(mmqsn_nbytes_shared<type, J, mode>() <= MMQSN_SMEM_MAX, "mmqsn: shared memory");
    static_assert(mode == MMQSN_RING || mode == MMQSN_PF || mode == MMQSN_STREAM, "mmqsn: bad mode");

    constexpr ggml_cuda_mmq_vec_dot_t    vec_dot    = ggml_cuda_mmq_get_vec_dot<type, J, false, false>();
    constexpr ggml_cuda_mmq_write_back_t write_back = ggml_cuda_mmq_get_write_back<type, J, false, false>();

    extern __shared__ __align__(16) int data_mul_mat_qsn[];
    int  * ids    = data_mul_mat_qsn;
    int  * tile_y = ids + J;
    int  * tile_x = tile_y + 2*ny;
    char * stage  = (char *) (tile_x + I*MMQSN_X_STRIDE);   // RING / STREAM only

    const int tid = threadIdx.y*WARP_SIZE + threadIdx.x;
    const int64_t row_bytes = (int64_t) stride_row_x*bs;

    // Identity ids for write_back (dense matmul only). Visible to all threads after the first barrier of the loop.
    if (tid < J) {
        ids[tid] = tid;
    }

    // Step list. Stream-k: MMQ's partition exactly (mmq.cuh mul_mat_q); blocks_per_iter is 1, so the rounding lines do
    // nothing for K-quants (kept for a later Q8_0 variant).
    mmqsn_steps s;
    s.nkb_fd  = nkb_fd;
    s.persist = persist;
    if (persist) {
        s.kbc0   = 0;
        s.nsteps = ((ntiles - (int) blockIdx.x + (int) gridDim.x - 1) / (int) gridDim.x) * (int) nkb_fd.z;
    } else {
        int kbc      = int64_t(blockIdx.x)    *(ntiles*(int) nkb_fd.z) / gridDim.x;
        int kbc_stop = int64_t(blockIdx.x + 1)*(ntiles*(int) nkb_fd.z) / gridDim.x;
        kbc      -= fastmodulo(kbc,      nkb_fd) % blocks_per_iter;
        kbc_stop -= fastmodulo(kbc_stop, nkb_fd) % blocks_per_iter;
        s.kbc0   = kbc;
        s.nsteps = kbc_stop - kbc;
    }
    const int nsteps = s.nsteps;

    float sum[J*I / MMQSN_NTHREADS] = {0.0f};
    int yr[2][nyr] = {{0}};

    if constexpr (mode == MMQSN_PF) {
        // ---- PF: stock load_tiles from global memory, L2 prefetch pf_dist steps ahead, y one step ahead. ----
        constexpr ggml_cuda_mmq_load_tiles_t load_tiles = ggml_cuda_mmq_get_load_tiles<type, J, false, false>();
        GGML_UNUSED(stage);
        GGML_UNUSED(l2hint);

        if (x_pf) {
            for (int t = 0; t < pf_dist; ++t) {
                mmqsn_prefetch<type>(x, s, t, 0, pf_run, row_bytes, tid);
            }
        }
        ggml_cuda_pdl_sync();   // nothing reads y, dst or tmp before this point
        if (!x_pf) {
            for (int t = 0; t < pf_dist; ++t) {
                mmqsn_prefetch<type>(x, s, t, 0, pf_run, row_bytes, tid);
            }
        }
        if (nsteps == 0) {
            return;
        }

        mmqsn_load_y<J>(y, ne11, mmqsn_step(s, 0).y, yr, tid);

#pragma unroll 1
        for (int g = 0; g < nsteps; ++g) {
            if (pf_dist > 0) {
                mmqsn_prefetch<type>(x, s, g + pf_dist, 0, pf_run, row_bytes, tid);
            }
            const int2 tk = mmqsn_step(s, g);
            load_tiles(x, tile_x, tk.x*I*stride_row_x + tk.y, I - 1, stride_row_x);
            mmqsn_store_y<J>(tile_y, yr, tid);

            __syncthreads();

            if (g + 1 < nsteps) {
                mmqsn_load_y<J>(y, ne11, mmqsn_step(s, g + 1).y, yr, tid);
            }

            vec_dot(tile_x, tile_y,      sum, 0);
            vec_dot(tile_x, tile_y + ny, sum, MMQ_TILE_NE_K);

            const bool tile_end = tk.y == (int) nkb_fd.z - 1;
            if (tile_end || g == nsteps - 1) {
                if (tile_end) {
                    write_back(sum, ids, dst + tk.x*I, nullptr, stride_col_dst, nrows_x - tk.x*I - 1, ne11 - 1);
                } else {
                    write_back(sum, ids, tmp + blockIdx.x*(J*I), nullptr, I, I, J);
                }
#pragma unroll
                for (int l = 0; l < J*I / MMQSN_NTHREADS; ++l) {
                    sum[l] = 0.0f;
                }
            }

            __syncthreads();
        }
    } else {
        // ---- RING (and STREAM): 2-slot cp.async ring of raw weight bytes, y one step ahead in registers. ----
        constexpr int pitch = mmqsn_raw<type>::pitch;
        const unsigned int stage_s = ggml_cuda_cvta_generic_to_shared(stage);

        // Prologue: steps 0 and 1. Always 2 commits per thread, so the wait_group<1> count below stays right.
        if (x_pf) {
            if (nsteps > 0) {
                mmqsn_issue<type>(x, s, 0, row_bytes, stage_s, l2hint, tid);
            }
            cp_async_commit_group();
            if (nsteps > 1) {
                mmqsn_issue<type>(x, s, 1, row_bytes, stage_s + I*pitch, l2hint, tid);
            }
            cp_async_commit_group();
        }
        ggml_cuda_pdl_sync();   // every thread; nothing reads y, dst or tmp before this point
        if (!x_pf) {
            if (nsteps > 0) {
                mmqsn_issue<type>(x, s, 0, row_bytes, stage_s, l2hint, tid);
            }
            cp_async_commit_group();
            if (nsteps > 1) {
                mmqsn_issue<type>(x, s, 1, row_bytes, stage_s + I*pitch, l2hint, tid);
            }
            cp_async_commit_group();
        }
        if (nsteps == 0) {
            return;
        }
        if (pf_dist > 0) {
            for (int t = 2; t < 2 + pf_dist; ++t) {
                mmqsn_prefetch<type>(x, s, t, 2, pf_run, row_bytes, tid);
            }
        }

        mmqsn_load_y<J>(y, ne11, mmqsn_step(s, 0).y, yr, tid);
        int sink = 0;   // STREAM: keeps the y loads alive

#pragma unroll 1
        for (int g = 0; g < nsteps; ++g) {
            const int slot = g & 1;
            const int2 tk  = mmqsn_step(s, g);

            cp_async_wait_group<1>();   // this thread's copies of step g landed (groups 0..g complete)
            __syncthreads();            // A: slot g visible to all; every warp finished vec_dot(g-1)

            if constexpr (mode == MMQSN_RING) {
                mmqsn_unpack<type, J>(stage + slot*(I*pitch), tile_x, x, tk, row_bytes);
                mmqsn_store_y<J>(tile_y, yr, tid);
            } else {
                GGML_UNUSED(tk);
#pragma unroll
                for (int h = 0; h < 2; ++h) {
#pragma unroll
                    for (int r = 0; r < nyr; ++r) {
                        sink ^= yr[h][r];
                    }
                }
            }

            __syncthreads();            // B: tile_x/tile_y ready; slot g&1 free again

            if (g + 2 < nsteps) {
                mmqsn_issue<type>(x, s, g + 2, row_bytes, stage_s + slot*(I*pitch), l2hint, tid);
            }
            cp_async_commit_group();    // always (empty past the end), so the wait count stays right
            if (pf_dist > 0) {
                mmqsn_prefetch<type>(x, s, g + 2 + pf_dist, 2, pf_run, row_bytes, tid);
            }
            if (g + 1 < nsteps) {
                mmqsn_load_y<J>(y, ne11, mmqsn_step(s, g + 1).y, yr, tid);   // L2 hit, hidden behind vec_dot
            }

            if constexpr (mode == MMQSN_RING) {
                vec_dot(tile_x, tile_y,      sum, 0);
                vec_dot(tile_x, tile_y + ny, sum, MMQ_TILE_NE_K);

                const bool tile_end = tk.y == (int) nkb_fd.z - 1;
                if (tile_end || g == nsteps - 1) {
                    if (tile_end) {
                        write_back(sum, ids, dst + tk.x*I, nullptr, stride_col_dst, nrows_x - tk.x*I - 1, ne11 - 1);
                    } else {
                        write_back(sum, ids, tmp + blockIdx.x*(J*I), nullptr, I, I, J);
                    }
#pragma unroll
                    for (int l = 0; l < J*I / MMQSN_NTHREADS; ++l) {
                        sum[l] = 0.0f;
                    }
                }
            }
        }

        if constexpr (mode == MMQSN_STREAM) {
            if (sink == 0x7F3A91C5) {   // practically never true; keeps the loads without a real store
                dst[threadIdx.x] = (float) sink;
            }
            GGML_UNUSED(sum);
            GGML_UNUSED(ids);
            GGML_UNUSED(tile_x);
            GGML_UNUSED(tile_y);
            GGML_UNUSED(tmp);
            GGML_UNUSED(vec_dot);
            GGML_UNUSED(write_back);
        }
    }

    ggml_cuda_pdl_lc();
#else
    GGML_UNUSED_VARS(x_ptr, y_ptr, dst_ptr, tmp_ptr, nkb_fd, nrows_x, ne11, stride_row_x, stride_col_dst, ntiles,
        persist, x_pf, pf_dist, pf_run, l2hint);
    NO_DEVICE_CODE;
#endif // defined(TURING_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

// Copy of the !ids_dst path of mul_mat_q_stream_k_fixup (mmq.cuh) with nsamples = nchannels = ntx = 1 and a PDL
// launch: same early exits, same summation order (walk down from bidx0 - 1, then dst += sum).
template <ggml_type type, int J>
__launch_bounds__(MMQSN_NTHREADS/2, 1)
static __global__ void mul_mat_qsn_fixup(
        float * dst_ptr, const float * tmp_ptr, const uint3 nkb_fd, const int ntiles, const int ncols_dst,
        const int stride_col_dst) {
#if defined(TURING_MMA_AVAILABLE) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    ggml_cuda_pdl_lc();
    // Wait before the early exits: if no block of this grid waited, it could finish before mul_mat_qsn, and a PDL
    // kernel after it would then only wait for this grid (and could reuse the q8_1/tmp pool memory too early).
    ggml_cuda_pdl_sync();
    float       * GGML_CUDA_RESTRICT dst           = dst_ptr;
    const float * GGML_CUDA_RESTRICT tmp_last_tile = tmp_ptr;

    constexpr int warp_size       = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps          = (MMQSN_NTHREADS/2) / warp_size;
    constexpr int I               = MMQSN_I;
    constexpr int qk              = ggml_cuda_type_traits<type>::qk;
    constexpr int blocks_per_iter = MMQ_ITER_K / qk;
    static_assert(ggml_cuda_mmq_get_I(type, J, false, false) == I, "mmqsn: MMQ tile height must be 128");

    const uint3 blocks_per_ne00 = nkb_fd;

    float sum[J / nwarps] = {0.0f};
    const int i = blockIdx.y*warp_size + threadIdx.x;

    const int bidx0 = blockIdx.x;

    // kbc == k block continuous, current index in continuous ijk space.
    int kbc0      = int64_t(blockIdx.x)    *(ntiles*(int) blocks_per_ne00.z) / gridDim.x;
    int kbc0_stop = int64_t(blockIdx.x + 1)*(ntiles*(int) blocks_per_ne00.z) / gridDim.x;

    kbc0      -= fastmodulo(kbc0,      blocks_per_ne00) % blocks_per_iter;
    kbc0_stop -= fastmodulo(kbc0_stop, blocks_per_ne00) % blocks_per_iter;

    const bool did_not_have_any_data   = kbc0 == kbc0_stop;
    const bool wrote_beginning_of_tile = fastmodulo(kbc0, blocks_per_ne00) == 0;
    const bool did_not_write_last      = fastdiv(kbc0, blocks_per_ne00) == fastdiv(kbc0_stop, blocks_per_ne00) && fastmodulo(kbc0_stop, blocks_per_ne00) != 0;
    if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        return;
    }

    bool any_fixup = false;

    // Iterate over previous blocks and sum up partial sums written to fixup buffer.
    // All CUDA blocks that get here must have a previous block that needs a fixup.
    int bidx = bidx0 - 1;
    int kbc_stop = kbc0;
    while(true) {
        int kbc = int64_t(bidx)*(ntiles*(int) blocks_per_ne00.z) / gridDim.x;
        kbc -= fastmodulo(kbc, blocks_per_ne00) % blocks_per_iter;

        if (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            continue;
        }

        any_fixup = true;

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            sum[j0/nwarps] += tmp_last_tile[bidx*(J*I) + j*I + i];
        }

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        if (fastmodulo(kbc, blocks_per_ne00) == 0 || fastdiv(kbc, blocks_per_ne00) < fastdiv(kbc0, blocks_per_ne00)) {
            break;
        }
        bidx--;
        kbc_stop = kbc;
    }

    if (!any_fixup) {
        return;
    }

    const int it = fastdiv(kbc0, blocks_per_ne00);

    dst += it*I;

    const int j_max = ncols_dst - 1;

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

        if (j > j_max) {
            return;
        }

        dst[j*stride_col_dst + i] += sum[j0/nwarps];
    }
#else
    GGML_UNUSED_VARS(dst_ptr, tmp_ptr, nkb_fd, ntiles, ncols_dst, stride_col_dst);
    NO_DEVICE_CODE;
#endif // defined(TURING_MMA_AVAILABLE) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

// ------------------------------------------------------------------------------------------------------------------
// Launch
// ------------------------------------------------------------------------------------------------------------------

struct mmqsn_args {
    const char * x;             // src0->data
    const int  * y;             // q8_1, MMQ block layout, + MMQSN_Y_PAD blocks
    float      * dst;
    float      * tmp;           // fixup buffer (grid*J*I floats) or nullptr
    int ncols_x;                // ne00 (multiple of 256)
    int nrows_x;                // ne01 (multiple of 128)
    int ne11;                   // src1 columns = dst columns
    int stride_row_x;           // nb01 in blocks
    int stride_col_dst;         // nb1 in floats
    int J;                      // 8 or 16
    int mode;                   // mmqsn_mode
    int grid;
    int ntiles;
    int persist;
    int fixup;
    int x_pf;
    int pf_dist;
    int pf_run;
    int l2hint;
};

template <ggml_type type, int J, int mode>
static void mmqsn_launch(const mmqsn_args & a, cudaStream_t stream) {
    constexpr size_t nbytes_shared = mmqsn_nbytes_shared<type, J, mode>();
    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_qsn<type, J, mode>), nbytes_shared);

    const uint3 nkb_fd = init_fastdiv_values(a.ncols_x / QK_K);

    const dim3 block_nums(a.grid, 1, 1);
    const dim3 block_dims(WARP_SIZE, MMQSN_NTHREADS/WARP_SIZE, 1);
    ggml_cuda_kernel_launch(mul_mat_qsn<type, J, mode>,
        ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream),
        a.x, a.y, a.dst, a.tmp, nkb_fd, a.nrows_x, a.ne11, a.stride_row_x, a.stride_col_dst, a.ntiles,
        a.persist, a.x_pf, a.pf_dist, a.pf_run, a.l2hint);

    if (!a.fixup) {
        return;
    }
    const dim3 block_nums_fixup(a.grid, MMQSN_I/WARP_SIZE, 1);
    const dim3 block_dims_fixup(WARP_SIZE, MMQSN_NTHREADS/(2*WARP_SIZE), 1);
    ggml_cuda_kernel_launch(mul_mat_qsn_fixup<type, J>,
        ggml_cuda_kernel_launch_params(block_nums_fixup, block_dims_fixup, 0, stream),
        a.dst, (const float *) a.tmp, nkb_fd, a.ntiles, a.ne11, a.stride_col_dst);
}

// Instantiates RING and PF for J = 8 and 16 and STREAM for J = 16 (5 kernels + 2 fixups per type).
template <ggml_type type>
void mmqsn_case(const mmqsn_args & a, cudaStream_t stream) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (a.mode == MMQSN_STREAM) {
        mmqsn_launch<type, 16, MMQSN_STREAM>(a, stream);
        return;
    }
    if (a.J == 8) {
        if (a.mode == MMQSN_PF) {
            mmqsn_launch<type,  8, MMQSN_PF>(a, stream);
        } else {
            mmqsn_launch<type,  8, MMQSN_RING>(a, stream);
        }
    } else {
        GGML_ASSERT(a.J == 16);
        if (a.mode == MMQSN_PF) {
            mmqsn_launch<type, 16, MMQSN_PF>(a, stream);
        } else {
            mmqsn_launch<type, 16, MMQSN_RING>(a, stream);
        }
    }
#else
    GGML_UNUSED(a);
    GGML_UNUSED(stream);
    GGML_ABORT("mmqsn: CUDA only");
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

#define DECL_MMQSN_CASE(type)                                                        \
    template void mmqsn_case<type>(const mmqsn_args & a, cudaStream_t stream) \

extern DECL_MMQSN_CASE(GGML_TYPE_Q4_K);
extern DECL_MMQSN_CASE(GGML_TYPE_Q5_K);
extern DECL_MMQSN_CASE(GGML_TYPE_Q6_K);
