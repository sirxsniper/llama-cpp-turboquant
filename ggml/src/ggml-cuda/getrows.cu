#include "getrows.cuh"

#include <set>
#include "dequantize.cuh"
#include "convert.cuh"

template<int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static __global__ void k_get_rows(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00, /*const int64_t ne01, const int64_t ne02, const int64_t ne03,*/
        /*const int64_t ne10,*/ const int64_t ne11, const uint3 ne12_fdv, /*const int64_t ne13,*/
        /*const size_t s0,*/ const size_t s1, const size_t s2, const size_t s3,
        /*const size_t nb00,*/ const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12/*, const size_t s13*/) {

    ggml_cuda_pdl_sync();
    for (int64_t z = blockIdx.z; z < ne11*(int64_t)ne12_fdv.z; z += gridDim.z) {
        for (int64_t i00 = 2*(blockIdx.y*blockDim.x + threadIdx.x); i00 < ne00; i00 += gridDim.y*blockDim.x) {
            // The x and y dimensions of the grid are swapped because the maximum allowed grid size for x is higher.
            const int i10 =  blockIdx.x;
            const uint2 dm  = fast_div_modulo((uint32_t)z, ne12_fdv);
            const int i11 =  dm.x;
            const int i12 =  dm.y;

            const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

            dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
            const void * src0_row = (const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03;

            const int ib   =  i00/qk;      // block index
            const int iqs  = (i00%qk)/qr;  // quant index
            const int iybs = i00 - i00%qk; // dst block start index
            const int y_offset = qr == 1 ? 1 : qk/2;

            // dequantize
            float2 v;
            dequantize_kernel(src0_row, ib, iqs, v);

            dst_row[iybs + iqs + 0]        = ggml_cuda_cast<dst_t>(v.x);
            dst_row[iybs + iqs + y_offset] = ggml_cuda_cast<dst_t>(v.y);
        }
    }
}

template<typename dst_t, dequantize_kq_t<dst_t> dequantize_kq>
static __global__ void k_get_rows_kq(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00, /*const int64_t ne01, const int64_t ne02, const int64_t ne03,*/
        /*const int64_t ne10,*/ const int64_t ne11, const uint3 ne12_fdv, /*const int64_t ne13,*/
        /*const size_t s0,*/ const size_t s1, const size_t s2, const size_t s3,
        /*const size_t nb00,*/ const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12/*, const size_t s13*/) {

    ggml_cuda_pdl_sync();
    const int64_t nsb = ne00/QK_K; // super-blocks per row
    for (int64_t z = blockIdx.z; z < ne11*(int64_t)ne12_fdv.z; z += gridDim.z) {
        // The x and y dimensions of the grid are swapped because the maximum allowed grid size for x is higher.
        const int i10 = blockIdx.x;
        const uint2 dm  = fast_div_modulo((uint32_t)z, ne12_fdv);
        const int i11 = dm.x;
        const int i12 = dm.y;

        const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

        dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
        const void * src0_row = (const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03;

        for (int64_t ib = blockIdx.y; ib < nsb; ib += gridDim.y) {
            dequantize_kq(src0_row, ib, dst_row + ib*QK_K, threadIdx.x);
        }
    }
}

template<typename src0_t, typename dst_t>
static __global__ void k_get_rows_float(
        const src0_t * src0_ptr, const int32_t * src1_ptr, dst_t * dst_ptr,
        const int64_t ne00, /*const int64_t ne01, const int64_t ne02, const int64_t ne03,*/
        /*const int64_t ne10,*/ const int64_t ne11, const uint3 ne12_fdv, /*const int64_t ne13,*/
        /*const size_t s0,*/ const size_t s1, const size_t s2, const size_t s3,
        /*const size_t nb00,*/ const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12/*, const size_t s13*/) {

    ggml_cuda_pdl_lc();
    const src0_t  * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const int32_t * GGML_CUDA_RESTRICT src1 = src1_ptr;
    dst_t         * GGML_CUDA_RESTRICT dst  = dst_ptr;
    ggml_cuda_pdl_sync();
    for (int64_t z = blockIdx.z; z < ne11*(int64_t)ne12_fdv.z; z += gridDim.z) {
        // The x and y dimensions of the grid are swapped because the maximum allowed grid size for x is higher.
        const int i10 = blockIdx.x;
        const uint2 dm = fast_div_modulo((uint32_t)z, ne12_fdv);
        const int i11 = dm.x;
        const int i12 = dm.y;

        const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

        dst_t * GGML_CUDA_RESTRICT dst_row = dst + i10*s1 + i11*s2 + i12*s3;
        const src0_t * GGML_CUDA_RESTRICT src0_row = (const src0_t *)((const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03);

        for (int64_t i00 = blockIdx.y*blockDim.x + threadIdx.x; i00 < ne00; i00 += gridDim.y*blockDim.x) {
            dst_row[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);
        }
    }
}

template<typename dst_t>
static __global__ void k_get_rows_float_vec(
        const dst_t * src0_ptr, const int32_t * src1_ptr, dst_t * dst_ptr,
        const int64_t ne00v,
        const int64_t ne11, const uint3 ne12_fdv,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {

    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    for (int64_t z = blockIdx.z; z < ne11*(int64_t)ne12_fdv.z; z += gridDim.z) {
        const int i10 = blockIdx.x;
        const uint2 dm = fast_div_modulo((uint32_t)z, ne12_fdv);
        const int i11 = dm.x;
        const int i12 = dm.y;

        const int i01 = src1_ptr[i10*s10 + i11*s11 + i12*s12];

        int4       * GGML_CUDA_RESTRICT dst_row  = (int4 *)      (dst_ptr + i10*s1 + i11*s2 + i12*s3);
        const int4 * GGML_CUDA_RESTRICT src0_row = (const int4 *)((const char *) src0_ptr + i01*nb01 + i11*nb02 + i12*nb03);

        for (int64_t i = blockIdx.y*blockDim.x + threadIdx.x; i < ne00v; i += gridDim.y*blockDim.x) {
            dst_row[i] = src0_row[i];
        }
    }
}

template<typename grad_t, typename dst_t>
static __global__ void k_get_rows_back_float(
        const grad_t * __restrict__ grad, const int32_t * __restrict__ rows, dst_t * __restrict__ dst,
        const int64_t ncols, const int64_t nrows_grad, const int64_t nrows_dst) {
    const int col = blockIdx.x*blockDim.x + threadIdx.x;

    if (col >= ncols) {
        return;
    }

    ggml_cuda_pdl_sync();

    // grid.y is clamped to the CUDA grid limit, so stride over the destination rows
    for (int64_t dst_row = blockIdx.y; dst_row < nrows_dst; dst_row += gridDim.y) {
        float sum = 0.0f;

        for (int64_t i = 0; i < nrows_grad; ++i) {
            if (rows[i] != dst_row) {
                continue;
            }
            sum += grad[i*ncols + col];
        }

        dst[dst_row*ncols + col] = sum;
    }
}

// ---- turbo4 get_rows: warp-cooperative centroid broadcast ----
//
// qwen4exp's sparse attention gathers the ENTIRE indexer cache every decode token
// (models/qwen4exp.cpp, build_qsa_top_k: ggml_get_rows(ctx0, k_all, inp->blk_cells)),
// so this kernel runs O(n_kv) per token and its per-element cost compounds with context.
//
// The generic get_rows_cuda_q path calls dequantize_turbo4_0 per element, which reads
// TURBO_CENTROIDS_4BIT[idx] with a data-dependent index. Constant memory broadcasts only
// when every lane reads the SAME address, so a 4-bit index means up to 16-way replay.
// Measured on Qwen3.8-Flash-Next (turbo4, ncmoe 36, 262144 ctx): decode fell 23 -> 4
// tok/s from 0.5K to 80K, while the same run with an F16 indexer stayed at 20 -> 16.7.
//
// Cure is the one that worked in convert.cu (+57% there): each lane holds ONE centroid
// pre-scaled by the block norm and __shfl_sync broadcasts it from registers, which has
// no divergence penalty. One warp per turbo4 block (QK_TURBO4 = 128 elements).
template <typename dst_t>
static __global__ void k_get_rows_turbo4(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00, const int64_t ne11, const uint3 ne12_fdv,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {

    const int64_t blocks_per_row = ne00 / QK_TURBO4;

    const int     i10 = blockIdx.x;
    const int64_t z   = blockIdx.z;
    const uint2   dm  = fast_div_modulo((uint32_t) z, ne12_fdv);
    const int     i11 = dm.x;
    const int     i12 = dm.y;

    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int64_t ib = (int64_t) blockIdx.y * warps_per_block + (threadIdx.x / WARP_SIZE);
    if (ib >= blocks_per_row) {
        return;
    }
    const int lane = threadIdx.x % WARP_SIZE;

    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

    const block_turbo4_0 * __restrict__ x =
        (const block_turbo4_0 *) ((const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03) + ib;
    dst_t * __restrict__ y = dst + i10*s1 + i11*s2 + i12*s3 + ib*QK_TURBO4;

    const float norm = __half2float(x->norm);

    // lane L < 16 holds centroid L already scaled by this block's norm
    const float my_scaled = (lane < 16) ? TURBO_CENTROIDS_4BIT[lane] * norm : 0.0f;

    constexpr int per_lane = QK_TURBO4 / WARP_SIZE;   // 4 elements per lane
#pragma unroll
    for (int e = 0; e < per_lane; e += 2) {
        const int j = lane * per_lane + e;
        const uint8_t qb = x->qs[j >> 1];
        const float v0 = __shfl_sync(0xFFFFFFFFu, my_scaled, (qb >> 0) & 0xFu, WARP_SIZE);
        const float v1 = __shfl_sync(0xFFFFFFFFu, my_scaled, (qb >> 4) & 0xFu, WARP_SIZE);
        y[j + 0] = ggml_cuda_cast<dst_t>(v0);
        y[j + 1] = ggml_cuda_cast<dst_t>(v1);
    }
}

template<typename dst_t>
static void get_rows_cuda_turbo4(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_TURBO4 == 0 && "turbo4 get_rows needs block-aligned rows");

    constexpr int warps_per_block = 4;
    const int64_t blocks_per_row  = ne00 / QK_TURBO4;

    const dim3 block_dims(warps_per_block*WARP_SIZE, 1, 1);
    const dim3 block_nums((unsigned) ne10,
                          (unsigned) MIN((blocks_per_row + warps_per_block - 1) / warps_per_block, (int64_t) UINT16_MAX),
                          (unsigned) MIN(ne11*ne12, (int64_t) UINT16_MAX));

    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);
    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);

    GGML_ASSERT(ne12 > 0);
    const uint3 ne12_fdv = init_fastdiv_values(ne12);

    k_get_rows_turbo4<dst_t><<<block_nums, block_dims, 0, stream>>>(
        src0_d, src1_d, dst_d, ne00, ne11, ne12_fdv,
        s1, s2, s3, nb01, nb02, nb03, s10, s11, s12);
}

// ---- turbo4p get_rows: same centroid broadcast, split-plane shape ----
//
// turbo4p_0 quantizes identically to turbo4_0, so the reason the kernel above exists is
// unchanged - a data-dependent index into __constant__ TURBO_CENTROIDS_4BIT replays up to
// 16 times per warp, and holding one centroid per lane in a register fixes it.
//
// Two differences, both inside turbo4p_dequant_lane (turbo-quant.cuh):
//   - a lane covers 16/sizeof(dst_t) elements and stores them with ONE 16-byte instruction.
//     k_get_rows_turbo4 above writes its four elements one scalar store at a time.
//   - the qs read is 4 bytes for an f16 destination, up from turbo4_0's two byte loads,
//     because a turbo4p block base is 16-byte aligned and jb/2 is a multiple of 4.
//
// [TAG_TURBO4_NC_GRID_STRIDE] grid.y and grid.z are clamped to UINT16_MAX by the launcher,
// so both are grid-strided here. That clamp with no stride is what silently dropped whole
// slices in the turbo4_0 dequant path, and there is no reason to leave the same trap armed.
template <typename dst_t>
static __global__ void k_get_rows_turbo4p(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00, const int64_t nz, const uint3 ne12_fdv,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {

    constexpr int per_lane = turbo4p_lane_shape<dst_t>::elems;
    constexpr int per_warp = turbo4p_lane_shape<dst_t>::per_warp;

    const int64_t warps_per_row = ne00 / per_warp;

    const int i10 = blockIdx.x;

    const int     warps_per_block = blockDim.x / WARP_SIZE;
    const int     warp_in_block   = threadIdx.x / WARP_SIZE;
    const int     lane            = threadIdx.x % WARP_SIZE;

    const float raw_centroid = (lane < 16) ? TURBO_CENTROIDS_4BIT[lane] : 0.0f;

    for (int64_t z = blockIdx.z; z < nz; z += gridDim.z) {
        const uint2 dm  = fast_div_modulo((uint32_t) z, ne12_fdv);
        const int   i11 = dm.x;
        const int   i12 = dm.y;

        const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

        const char * __restrict__ xrow = (const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03;
        dst_t * __restrict__      yrow = dst + i10*s1 + i11*s2 + i12*s3;

        // Warp-uniform loop bound, so a whole warp always runs the same iterations and the
        // full-mask __shfl_sync inside turbo4p_dequant_lane stays legal.
        for (int64_t iw = (int64_t) blockIdx.y * warps_per_block + warp_in_block;
             iw < warps_per_row;
             iw += (int64_t) gridDim.y * warps_per_block) {

            const int64_t j  = iw * per_warp + lane * per_lane;
            const int64_t ib = j / QK_TURBO4P;
            const int     jb = (int) (j % QK_TURBO4P);

            turbo4p_dequant_lane<dst_t>((const block_turbo4p_0 *) xrow + ib, jb,
                                        raw_centroid, yrow + j);
        }
    }
}

template<typename dst_t>
static void get_rows_cuda_turbo4p(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_TURBO4P == 0 && "turbo4p get_rows needs block-aligned rows");

    constexpr int warps_per_block = 4;
    constexpr int per_warp        = turbo4p_lane_shape<dst_t>::per_warp;
    const int64_t warps_per_row   = ne00 / per_warp;

    const dim3 block_dims(warps_per_block*WARP_SIZE, 1, 1);
    const dim3 block_nums((unsigned) ne10,
                          (unsigned) MIN((warps_per_row + warps_per_block - 1) / warps_per_block, (int64_t) UINT16_MAX),
                          (unsigned) MIN(ne11*ne12, (int64_t) UINT16_MAX));

    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);
    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);

    GGML_ASSERT(ne12 > 0);
    const uint3 ne12_fdv = init_fastdiv_values(ne12);

    k_get_rows_turbo4p<dst_t><<<block_nums, block_dims, 0, stream>>>(
        src0_d, src1_d, dst_d, ne00, ne11*ne12, ne12_fdv,
        s1, s2, s3, nb01, nb02, nb03, s10, s11, s12);
}


// ---- [TAG_TURBO5P] turbo5p get_rows: turbo4p's over the 5-bit lane helper ----
template <typename dst_t>
static __global__ void k_get_rows_turbo5p(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00, const int64_t nz, const uint3 ne12_fdv,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {

    constexpr int per_lane = turbo5p_lane_shape<dst_t>::elems;
    constexpr int per_warp = turbo5p_lane_shape<dst_t>::per_warp;

    const int64_t warps_per_row = ne00 / per_warp;

    const int i10 = blockIdx.x;

    const int     warps_per_block = blockDim.x / WARP_SIZE;
    const int     warp_in_block   = threadIdx.x / WARP_SIZE;
    const int     lane            = threadIdx.x % WARP_SIZE;

    const float raw_centroid = TURBO_CENTROIDS_5BIT[lane];

    for (int64_t z = blockIdx.z; z < nz; z += gridDim.z) {
        const uint2 dm  = fast_div_modulo((uint32_t) z, ne12_fdv);
        const int   i11 = dm.x;
        const int   i12 = dm.y;

        const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

        const char * __restrict__ xrow = (const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03;
        dst_t * __restrict__      yrow = dst + i10*s1 + i11*s2 + i12*s3;

        // Warp-uniform loop bound, so a whole warp always runs the same iterations and the
        // full-mask __shfl_sync inside turbo5p_dequant_lane stays legal.
        for (int64_t iw = (int64_t) blockIdx.y * warps_per_block + warp_in_block;
             iw < warps_per_row;
             iw += (int64_t) gridDim.y * warps_per_block) {

            const int64_t j  = iw * per_warp + lane * per_lane;
            const int64_t ib = j / QK_TURBO5P;
            const int     jb = (int) (j % QK_TURBO5P);

            turbo5p_dequant_lane<dst_t>((const block_turbo5p_0 *) xrow + ib, jb,
                                        raw_centroid, yrow + j);
        }
    }
}

template<typename dst_t>
static void get_rows_cuda_turbo5p(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_TURBO5P == 0 && "turbo5p get_rows needs block-aligned rows");

    constexpr int warps_per_block = 4;
    constexpr int per_warp        = turbo5p_lane_shape<dst_t>::per_warp;
    const int64_t warps_per_row   = ne00 / per_warp;

    const dim3 block_dims(warps_per_block*WARP_SIZE, 1, 1);
    const dim3 block_nums((unsigned) ne10,
                          (unsigned) MIN((warps_per_row + warps_per_block - 1) / warps_per_block, (int64_t) UINT16_MAX),
                          (unsigned) MIN(ne11*ne12, (int64_t) UINT16_MAX));

    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);
    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);

    GGML_ASSERT(ne12 > 0);
    const uint3 ne12_fdv = init_fastdiv_values(ne12);

    k_get_rows_turbo5p<dst_t><<<block_nums, block_dims, 0, stream>>>(
        src0_d, src1_d, dst_d, ne00, ne11*ne12, ne12_fdv,
        s1, s2, s3, nb01, nb02, nb03, s10, s11, s12);
}

template<int qk, int qr, dequantize_kernel_t dq, typename dst_t>
static void get_rows_cuda_q(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);
    const int block_num_y = (ne00 + 2*CUDA_GET_ROWS_BLOCK_SIZE - 1) / (2*CUDA_GET_ROWS_BLOCK_SIZE);
    const dim3 block_nums(ne10, MIN(block_num_y, UINT16_MAX), MIN(ne11*ne12, UINT16_MAX));

    // strides in elements
    // const size_t s0 = nb0 / sizeof(dst_t);
    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);

    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);
    // const size_t s13 = nb13 / sizeof(int32_t);

    GGML_ASSERT(ne00 % 2 == 0);

    GGML_ASSERT(ne12 > 0);
    GGML_ASSERT(ne11 <= std::numeric_limits<uint32_t>::max() / ne12);
    const uint3 ne12_fdv = init_fastdiv_values(ne12);

    k_get_rows<qk, qr, dq><<<block_nums, block_dims, 0, stream>>>(
        src0_d, src1_d, dst_d,
        ne00, /*ne01, ne02, ne03,*/
        /*ne10,*/ ne11, ne12_fdv, /*ne13,*/
        /* s0,*/ s1, s2, s3,
        /* nb00,*/ nb01, nb02, nb03,
        s10, s11, s12/*, s13*/);
}

template<int block_dim, typename dst_t, dequantize_kq_t<dst_t> dequantize_kq>
static void get_rows_cuda_kq(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_K == 0);
    const int64_t nsb = ne00/QK_K;

    const dim3 block_dims(block_dim, 1, 1);
    const dim3 block_nums(ne10, MIN(nsb, UINT16_MAX), MIN(ne11*ne12, UINT16_MAX));

    // strides in elements
    // const size_t s0 = nb0 / sizeof(dst_t);
    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);

    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);
    // const size_t s13 = nb13 / sizeof(int32_t);

    GGML_ASSERT(ne12 > 0);
    GGML_ASSERT(ne11 <= std::numeric_limits<uint32_t>::max() / ne12);
    const uint3 ne12_fdv = init_fastdiv_values(ne12);

    k_get_rows_kq<dst_t, dequantize_kq><<<block_nums, block_dims, 0, stream>>>(
        src0_d, src1_d, dst_d,
        ne00, /*ne01, ne02, ne03,*/
        /*ne10,*/ ne11, ne12_fdv, /*ne13,*/
        /* s0,*/ s1, s2, s3,
        /* nb00,*/ nb01, nb02, nb03,
        s10, s11, s12/*, s13*/);
}

template<typename src0_t, typename dst_t>
static void get_rows_cuda_float(
        const src0_t * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);

    // strides in elements
    // const size_t s0 = nb0 / sizeof(dst_t);
    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);

    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);
    // const size_t s13 = nb13 / sizeof(int32_t);

    GGML_ASSERT(ne12 > 0);
    GGML_ASSERT(ne11 <= std::numeric_limits<uint32_t>::max() / ne12);
    const uint3 ne12_fdv = init_fastdiv_values(ne12);

    if constexpr (std::is_same<src0_t, dst_t>::value) {
        constexpr int VEC = 16 / sizeof(dst_t);
        const int64_t ne00v = ne00 / VEC;
        const int64_t vec_block_num_y = (ne00v + CUDA_GET_ROWS_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BLOCK_SIZE;
        const bool enough_blocks = vec_block_num_y * ne10 * ne11 * ne12 >= 128;
        const bool can_vec = VEC > 1 && enough_blocks &&
            (ne00 % VEC == 0) &&
            (nb01 % 16 == 0) && (nb02 % 16 == 0) && (nb03 % 16 == 0) &&
            (nb1  % 16 == 0) && (nb2  % 16 == 0) && (nb3  % 16 == 0) &&
            (((uintptr_t) src0_d) % 16 == 0) && (((uintptr_t) dst_d) % 16 == 0);

        if (can_vec) {
            const int block_num_y = vec_block_num_y;
            const dim3 block_nums(ne10, MIN(block_num_y, UINT16_MAX), MIN(ne11*ne12, UINT16_MAX));
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{block_nums, block_dims, 0, stream};
            ggml_cuda_kernel_launch(k_get_rows_float_vec<dst_t>, launch_params,
                (const dst_t *) src0_d, src1_d, dst_d,
                ne00v, ne11, ne12_fdv,
                s1, s2, s3,
                nb01, nb02, nb03,
                s10, s11, s12);
            return;
        }
    }

    const int block_num_y = (ne00 + CUDA_GET_ROWS_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BLOCK_SIZE;
    const dim3 block_nums(ne10, MIN(block_num_y, UINT16_MAX), MIN(ne11*ne12, UINT16_MAX));

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{block_nums, block_dims, 0, stream};
    ggml_cuda_kernel_launch(k_get_rows_float<src0_t, dst_t>, launch_params,
        src0_d, src1_d, dst_d,
        ne00, /*ne01, ne02, ne03,*/
        /*ne10,*/ ne11, ne12_fdv, /*ne13,*/
        /* s0,*/ s1, s2, s3,
        /* nb00,*/ nb01, nb02, nb03,
        s10, s11, s12/*, s13*/);
}

template <typename dst_t>
static void ggml_cuda_get_rows_switch_src0_type(
        const void * src0_d, const ggml_type src0_type, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    switch (src0_type) {
        case GGML_TYPE_F16:
            get_rows_cuda_float((const half *) src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_F32:
            get_rows_cuda_float((const float *) src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_I32:
            get_rows_cuda_float((const int32_t *) src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_BF16:
            get_rows_cuda_float((const nv_bfloat16 *) src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q1_0:
            get_rows_cuda_q<QK1_0, QR1_0, dequantize_q1_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q2_0:
            get_rows_cuda_q<QK2_0, QR2_0, dequantize_q2_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q4_0:
            get_rows_cuda_q<QK4_0, QR4_0, dequantize_q4_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q4_1:
            get_rows_cuda_q<QK4_1, QR4_1, dequantize_q4_1>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q5_0:
            get_rows_cuda_q<QK5_0, QR5_0, dequantize_q5_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q5_1:
            get_rows_cuda_q<QK5_1, QR5_1, dequantize_q5_1>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q8_0:
            get_rows_cuda_q<QK8_0, QR8_0, dequantize_q8_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        // TurboQuant KV types. qwen4exp's sparse attention gathers K-cache blocks with
        // ggml_get_rows (models/qwen4exp.cpp: ggml_get_rows(ctx0, k_all, inp->blk_cells)),
        // so -ctk turbo4 lands here on the live decode path. The switch below has no
        // default arm, which meant an unhandled type silently ran NO kernel and left the
        // destination buffer as-is - the model answered the previous question, or emitted
        // an empty string, with no error anywhere. Both the cases and the abort matter.
        //
        // Like q8_0 these use qr == 1: one call yields two CONSECUTIVE elements, which is
        // what the turbo packing gives (element j lives in byte j/2, nibble j%2).
        case GGML_TYPE_TURBO4_0:
            get_rows_cuda_turbo4(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_TURBO4P_0:
            get_rows_cuda_turbo4p(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_TURBO5P_0:
            get_rows_cuda_turbo5p(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_TURBO3_0:
            get_rows_cuda_q<QK_TURBO3, QR_TURBO3, dequantize_turbo3_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_TURBO2_0:
            get_rows_cuda_q<QK_TURBO2, QR_TURBO2, dequantize_turbo2_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q2_K:
            get_rows_cuda_kq<64, dst_t, dequantize_q2_K<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q3_K:
            get_rows_cuda_kq<64, dst_t, dequantize_q3_K<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q4_K:
            get_rows_cuda_kq<32, dst_t, dequantize_q4_K<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q5_K:
            get_rows_cuda_kq<64, dst_t, dequantize_q5_K<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q6_K:
            get_rows_cuda_kq<64, dst_t, dequantize_q6_K<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            get_rows_cuda_kq<32, dst_t, dequantize_iq2_xxs<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            get_rows_cuda_kq<32, dst_t, dequantize_iq2_xs<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ2_S:
            get_rows_cuda_kq<32, dst_t, dequantize_iq2_s<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            get_rows_cuda_kq<32, dst_t, dequantize_iq3_xxs<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ3_S:
            get_rows_cuda_kq<32, dst_t, dequantize_iq3_s<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ1_S:
            get_rows_cuda_kq<32, dst_t, dequantize_iq1_s<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ1_M:
            get_rows_cuda_kq<32, dst_t, dequantize_iq1_m<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            get_rows_cuda_kq<32, dst_t, dequantize_iq4_nl<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            get_rows_cuda_kq<32, dst_t, dequantize_iq4_xs<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_MXFP4:
            get_rows_cuda_kq<32, dst_t, dequantize_mxfp4<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        default:
            GGML_ABORT("%s: unsupported src0 type: %s\n", __func__, ggml_type_name(src0_type));
            break;
    }
}

void get_rows_cuda(
        const void * src0_d, ggml_type src0_type, const int32_t * src1_d, void * dst_d, ggml_type dst_type,
        int64_t ne00, size_t nb01, size_t nb02, size_t nb03,
        int64_t ne10, int64_t ne11, int64_t ne12, size_t nb10, size_t nb11, size_t nb12,
        size_t nb1, size_t nb2, size_t nb3,
        cudaStream_t stream) {
    switch (dst_type) {
        case GGML_TYPE_F32:
            ggml_cuda_get_rows_switch_src0_type(src0_d, src0_type, src1_d, (float *) dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_I32:
            ggml_cuda_get_rows_switch_src0_type(src0_d, src0_type, src1_d, (int32_t *) dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_F16:
            ggml_cuda_get_rows_switch_src0_type(src0_d, src0_type, src1_d, (half *) dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_BF16:
            ggml_cuda_get_rows_switch_src0_type(src0_d, src0_type, src1_d, (nv_bfloat16 *) dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        default:
            GGML_ABORT("%s: unsupported dst type: %s\n", __func__, ggml_type_name(dst_type));
            break;
    }
}

void ggml_cuda_op_get_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    // one-shot: which src0 types actually flow through get_rows, and how big.
    {
        static std::set<int> seen;
        const ggml_tensor * s0 = dst->src[0];
        // Opt-IN: this ran a getenv plus a std::set lookup on EVERY get_rows.
        static const bool probe_on = [] {
            const char * e = getenv("TURBO_PATH_PROBE");
            return e && e[0] == '1';
        }();
        if (probe_on && seen.insert((int) s0->type).second) {
            fprintf(stderr, "turbo-probe: get_rows src0=%s ne=[%d,%d] dst=%s\n",
                    ggml_type_name(s0->type), (int) s0->ne[0], (int) s0->ne[1],
                    ggml_type_name(dst->type));
            fflush(stderr);
        }
    }
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    cudaStream_t stream = ctx.stream();

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(ne13 == 1);

    GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));
    GGML_ASSERT(src1->nb[0] == ggml_type_size(src1->type));
    GGML_ASSERT(dst->nb[0]  == ggml_type_size(dst->type));

    get_rows_cuda(src0->data, src0->type, (const int32_t *) src1->data, dst->data, dst->type,
        ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
}

void ggml_cuda_op_get_rows_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // gradients of forward pass output
    const ggml_tensor * src1 = dst->src[1]; // src1 in forward pass

    GGML_TENSOR_BINARY_OP_LOCALS

    const float   * src0_d = (const float   *) src0->data;
    const int32_t * src1_d = (const int32_t *) src1->data;
    float         * dst_d  = (float         *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(ne02*ne03 == 1);
    GGML_ASSERT(ne12*ne13 == 1);
    GGML_ASSERT(ne2*ne3 == 1);

    const dim3 block_dims(CUDA_GET_ROWS_BACK_BLOCK_SIZE, 1, 1);
    const int block_num_x = (ne00 + CUDA_GET_ROWS_BACK_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BACK_BLOCK_SIZE;
    const dim3 block_nums(block_num_x, MIN(ne1, (int64_t)UINT16_MAX), 1);

    k_get_rows_back_float<<<block_nums, block_dims, 0, stream>>>(src0_d, src1_d, dst_d, ne00, ne10, ne1);
}
