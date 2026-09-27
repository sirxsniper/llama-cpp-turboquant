// [TAG_MMQSN] Host side of the cp.async-ring MMQ (kernels in mmqsn-impl.cuh, instances in template-instances/mmqsn-*).
//
// The routing, the q8_1 quantization, the J choice and the grid rule reproduce MMQ (mmq.cu, mmq.cuh launch_mul_mat_q)
// so the result is bit-identical; GGML_CUDA_MMQSN_CHECK=1 verifies that in-process against stock MMQ.

#include "mmqsn.cuh"
#include "mmqsn-impl.cuh"
#include "mmq.cuh"
#include "mmvq.cuh"       // ggml_cuda_should_use_mmvq (phase gate)
#include "quantize.cuh"   // quantize_mmq_q8_1_pdl_cuda

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>

// ------------------------------------------------------------------------------------------------------------------
// Environment
// ------------------------------------------------------------------------------------------------------------------

struct mmqsn_env_t {
    int     mode;       // 0 = off (unset), 1 = cc 1200 only, 2 = any Ampere+ NVIDIA GPU
    int     loop;       // mmqsn_mode
    int     pf_dist;    // L2 prefetch distance in steps
    int     pf_run;     // steps per prefetch run: 1, 2 or 4
    int     l2hint;     // 1 = cp.async .L2::256B
    int     min_n;      // 0 = only the widths MMQ takes today; n = also MMVQ's widths >= n
    int     max_n;      // widest src1 batch, 2..16 (2..32 with wide)
    int64_t min_rows;   // smallest weight row count
    int64_t max_rows;   // largest weight row count, 0 = no limit
    int     types;      // 1 = Q4_K, 2 = Q5_K, 4 = Q6_K
    int     persist;    // persistent grid for tiling shapes
    bool    xpf;        // test only: weight copies before the PDL wait for any src0 buffer
    bool    check;      // compare against stock MMQ
    bool    probe;      // TURBO_PATH_PROBE=1
    bool    wide;       // [TAG_MMQSN_WIDE] 17..32 columns (J = 24/32, RING mode)
    bool    fusefix;    // [TAG_MMQSN_FUSEFIX] stream-k fixup inside the RING kernel
};

static int mmqsn_env_int(const char * name, const int def) {
    const char * e = getenv(name);
    return (e && e[0]) ? atoi(e) : def;
}

static const char * mmqsn_mode_name(const int loop) {
    return loop == MMQSN_PF ? "pf" : (loop == MMQSN_STREAM ? "stream" : "ring");
}

static const mmqsn_env_t & mmqsn_env() {
    static const mmqsn_env_t env = [] {
        mmqsn_env_t v;
        // [TAG_MMQSN_DEFAULT] on by default where it was measured (cc 1200): server ms/step -5 % at 4 streams, -6 % at 2,
        // bit-identical output. GGML_CUDA_MMQSN=0 is the kill switch, =2 enables it on any Ampere+ NVIDIA GPU (tests).
        const int m = mmqsn_env_int("GGML_CUDA_MMQSN", 1);
        v.mode = (m == 1 || m == 2) ? m : 0;
        const char * lm = getenv("GGML_CUDA_MMQSN_MODE");
        v.loop = MMQSN_RING;
        if (lm && strcmp(lm, "pf") == 0) {
            v.loop = MMQSN_PF;
        } else if (lm && strcmp(lm, "stream") == 0) {
            v.loop = MMQSN_STREAM;
        }
        v.pf_dist  = std::max(0, mmqsn_env_int("GGML_CUDA_MMQSN_PF", v.loop == MMQSN_PF ? 2 : 0));
        const int run = mmqsn_env_int("GGML_CUDA_MMQSN_PF_RUN", 1);
        v.pf_run   = (run == 2 || run == 4) ? run : 1;
        v.l2hint   = mmqsn_env_int("GGML_CUDA_MMQSN_L2HINT", 1) != 0 ? 1 : 0;   // [TAG_MMQSN_DEFAULT] best in G0/G2
        v.min_n    = std::max(0, mmqsn_env_int("GGML_CUDA_MMQSN_MIN", 0));
        // [TAG_MMQSN_WIDE] off by default until measured: without it the widest batch stays 16 (today's routing).
        v.wide     = mmqsn_env_int("GGML_CUDA_MMQSN_WIDE", 0) != 0;
        const int max_cap = v.wide ? 32 : 16;
        v.max_n    = std::min(max_cap, std::max(2, mmqsn_env_int("GGML_CUDA_MMQSN_MAX", max_cap)));
        // [TAG_MMQSN_FUSEFIX] off by default until measured.
        v.fusefix  = mmqsn_env_int("GGML_CUDA_MMQSN_FUSEFIX", 0) != 0;
        const char * r0 = getenv("GGML_CUDA_MMQSN_MIN_ROWS");
        v.min_rows = (r0 && r0[0]) ? std::max<int64_t>(0, (int64_t) atoll(r0)) : 2048;
        const char * r1 = getenv("GGML_CUDA_MMQSN_MAX_ROWS");
        v.max_rows = (r1 && r1[0]) ? std::max<int64_t>(0, (int64_t) atoll(r1)) : 0;
        v.types    = mmqsn_env_int("GGML_CUDA_MMQSN_TYPES", 7) & 7;
        v.persist  = mmqsn_env_int("GGML_CUDA_MMQSN_PERSIST", 1) != 0 ? 1 : 0;
        v.xpf      = mmqsn_env_int("GGML_CUDA_MMQSN_XPF", 0) != 0;
        v.check    = mmqsn_env_int("GGML_CUDA_MMQSN_CHECK", 0) != 0;
        const char * p = getenv("TURBO_PATH_PROBE");
        v.probe = p && p[0] == '1';
        if (v.probe) {
            fprintf(stderr, "turbo-probe: mmqsn env on=%d mode=%s pf=%d pf_run=%d l2hint=%d min=%d max=%d min_rows=%lld "
                    "max_rows=%lld types=%d persist=%d xpf=%d check=%d wide=%d fusefix=%d\n",
                    v.mode, mmqsn_mode_name(v.loop), v.pf_dist, v.pf_run, v.l2hint, v.min_n, v.max_n,
                    (long long) v.min_rows, (long long) v.max_rows, v.types, v.persist, (int) v.xpf, (int) v.check,
                    (int) v.wide, (int) v.fusefix);
            fflush(stderr);
        }
        if (v.mode != 0 && v.loop == MMQSN_STREAM) {
            fprintf(stderr, "WARNING: GGML_CUDA_MMQSN_MODE=stream measures the weight-stream ceiling only: matmul results "
                    "are WRONG\n");
            fflush(stderr);
        }
        return v;
    }();
    return env;
}

static int mmqsn_type_bit(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_K: return 1;
        case GGML_TYPE_Q5_K: return 2;
        case GGML_TYPE_Q6_K: return 4;
        default:             return 0;
    }
}

// MMQ's J for 2..32 columns on NVIDIA (mul_mat_q_switch_J: the smallest J with one column tile), which the result
// depends on. STREAM always runs J = 16. [TAG_MMQSN_WIDE] 24 and 32 exist in RING mode only.
static int mmqsn_pick_J(const int64_t ne11, const int loop) {
    if (loop == MMQSN_STREAM) {
        return 16;
    }
    return ne11 <= 8 ? 8 : (ne11 <= 16 ? 16 : (ne11 <= 24 ? 24 : 32));
}

// ------------------------------------------------------------------------------------------------------------------
// Routing
// ------------------------------------------------------------------------------------------------------------------

bool ggml_cuda_should_use_mmqsn(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, const int cc) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA) || defined(GGML_CUDA_FORCE_CUBLAS)
    GGML_UNUSED_VARS(src0, src1, dst, cc);
    return false;
#else
    const mmqsn_env_t & env = mmqsn_env();
    if (env.mode == 0) {
        return false;                                   // default: today's routing exactly
    }
    if (!ampere_mma_available(cc)) {
        return false;
    }
    if (env.mode == 1 && cc != GGML_CUDA_CC_BLACKWELL) {
        return false;                                   // only where it is measured
    }

    const ggml_type type = src0->type;
    const int bit = mmqsn_type_bit(type);
    if (bit == 0 || (env.types & bit) == 0) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne11 = src1->ne[1];
    if (src1->ne[0] != ne00 || ne00 % QK_K != 0 || ne00 <= 0 || src0->nb[0] != ggml_type_size(type)) {
        return false;
    }
    if (ne01 % MMQSN_I != 0 || ne01 < env.min_rows || (env.max_rows > 0 && ne01 > env.max_rows) || ne01 >= INT_MAX/2) {
        return false;
    }
    if ((ne01/MMQSN_I)*(ne00/QK_K) >= (int64_t(1) << 30)) {
        return false;                                   // kbc must not overflow (as MMQ asserts)
    }
    // cp.async needs 16-byte aligned sources: Q4_K/Q5_K rows are, Q6_K rows (210 B blocks) only need 2 bytes because
    // the kernel copies the covering 16-byte window.
    const size_t align_row = type == GGML_TYPE_Q6_K ? 2 : 16;
    if ((uintptr_t) src0->data % 16 != 0 || src0->nb[1] % align_row != 0 || src0->nb[1]/ggml_type_size(type) >= (size_t) INT_MAX) {
        return false;
    }
    if (type == GGML_TYPE_Q6_K) {
        // The 224-byte window of a row's last block reads up to 14 bytes past it: stay inside the buffer.
        if (src0->buffer == nullptr) {
            return false;
        }
        const uintptr_t base = (uintptr_t) ggml_backend_buffer_get_base(src0->buffer);
        const uintptr_t end  = (((uintptr_t) src0->data + ggml_nbytes(src0)) + 15) & ~(uintptr_t) 15;
        if (base == 0 || (uintptr_t) src0->data < base || end > base + ggml_backend_buffer_get_size(src0->buffer)) {
            return false;
        }
    }
    if (src1->nb[0] != sizeof(float) || (uintptr_t) src1->data % 16 != 0 || src1->nb[1] % 16 != 0) {
        return false;
    }
    if (dst->nb[0] != sizeof(float) || dst->nb[1] % sizeof(float) != 0 || dst->nb[1]/sizeof(float) >= (size_t) INT_MAX) {
        return false;
    }
    if (ne11 < 2 || ne11 > env.max_n) {
        return false;                                   // ne11 == 1 stays on MMVQ (fused gate/up/GLU)
    }
    if (ne11 > 16 && env.loop != MMQSN_RING) {
        return false;                                   // [TAG_MMQSN_WIDE] J = 24/32 are RING kernels only
    }
    // Phase gate: by default MMVQ keeps every width it takes today (Blackwell: Q4_K/Q5_K <= 5, Q6_K <= 7).
    // GGML_CUDA_MMQSN_MIN=n also takes those widths from n up (not bit-identical: MMVQ quantizes differently).
    if (ne11 < (env.min_n ? env.min_n : INT_MAX) && ggml_cuda_should_use_mmvq(type, cc, ne11, ne01)) {
        return false;
    }
    if (!ggml_cuda_should_use_mmq(type, cc, ne11, /*n_experts =*/ 0)) {
        return false;
    }
    const int J = mmqsn_pick_J(ne11, env.loop);
    const size_t smpbo = ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo;
    if (mmqsn_nbytes_shared_rt(type, J, env.loop) > smpbo) {
        return false;
    }
    return true;
#endif // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA) || defined(GGML_CUDA_FORCE_CUBLAS)
}

// ------------------------------------------------------------------------------------------------------------------
// CHECK mode: bitwise compare against stock MMQ
// ------------------------------------------------------------------------------------------------------------------

// cnt[0] = number of bitwise mismatches, cnt[1] = largest |a - b| as float bits (non-negative floats order like uints).
static __global__ void mmqsn_check_compare(
        const float * a, const float * b, const int64_t ne0, const int64_t ne1, const int64_t s1a, const int64_t s1b,
        unsigned int * cnt) {
    const int64_t n = ne0*ne1;
    for (int64_t idx = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; idx < n; idx += (int64_t) gridDim.x*blockDim.x) {
        const int64_t j = idx / ne0;
        const int64_t i = idx - j*ne0;
        const float va = a[j*s1a + i];
        const float vb = b[j*s1b + i];
        if (__float_as_uint(va) != __float_as_uint(vb)) {
            atomicAdd(&cnt[0], 1u);
            const float d = fabsf(va - vb);
            if (d == d) {
                atomicMax(&cnt[1], __float_as_uint(d));
            }
        }
    }
}

static void mmqsn_check(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * dst, const mmqsn_env_t & env) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    cudaStream_t stream = ctx.stream();
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr, "turbo-probe: mmqsn check skipped under graph capture, set GGML_CUDA_DISABLE_GRAPHS=1\n");
            fflush(stderr);
        }
        return;
    }

    ggml_cuda_pool_alloc<char> ref_buf(ctx.pool(), ggml_nbytes(dst));
    ggml_tensor ref = *dst;
    ref.data = ref_buf.get();
    ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, &ref);

    ggml_cuda_pool_alloc<unsigned int> cnt(ctx.pool(), 2);
    CUDA_CHECK(cudaMemsetAsync(cnt.get(), 0, 2*sizeof(unsigned int), stream));
    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];
    const int64_t n   = ne0*ne1;
    const int nblocks = (int) std::min<int64_t>(1024, (n + 255)/256);
    mmqsn_check_compare<<<nblocks, 256, 0, stream>>>((const float *) dst->data, (const float *) ref.data, ne0, ne1,
        dst->nb[1]/sizeof(float), ref.nb[1]/sizeof(float), cnt.get());
    CUDA_CHECK(cudaGetLastError());
    unsigned int h[2] = {0, 0};
    CUDA_CHECK(cudaMemcpyAsync(h, cnt.get(), sizeof(h), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    static std::mutex check_mutex;
    static std::set<std::tuple<int, int64_t, int64_t, int64_t>> check_seen;
    std::lock_guard<std::mutex> lock(check_mutex);
    const bool first = check_seen.insert(std::make_tuple((int) src0->type, src0->ne[1], src0->ne[0], src1->ne[1])).second;
    if (first || h[0] != 0) {
        float maxabs;
        memcpy(&maxabs, &h[1], sizeof(maxabs));
        fprintf(stderr, "turbo-probe: mmqsn check type=%s m=%lld k=%lld n=%lld mode=%s mismatches=%u/%lld maxabs=%g\n",
                ggml_type_name(src0->type), (long long) src0->ne[1], (long long) src0->ne[0], (long long) src1->ne[1],
                mmqsn_mode_name(env.loop), h[0], (long long) n, (double) maxabs);
        fflush(stderr);
    }
#else
    GGML_UNUSED_VARS(ctx, src0, src1, dst, env);
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

// ------------------------------------------------------------------------------------------------------------------
// [TAG_MMQSN_FUSEFIX] Tile counters for the in-kernel fixup
// ------------------------------------------------------------------------------------------------------------------

// One zeroed buffer of MMQSN_FIXUP_CNT_MAX counters (16 KB) per stream: kernels of one stream run in order (a PDL
// kernel reads the counters only after its grid-dependency wait), and every launch leaves its counters at 0. Allocated
// on the first use outside graph capture; until then, for more tiles, or if the allocation fails, the caller launches
// the fixup kernel instead. The buffers live until the process ends.
#define MMQSN_FIXUP_CNT_MAX 4096

static int * mmqsn_fixup_counters(cudaStream_t stream, const int ntiles) {
    if (ntiles > MMQSN_FIXUP_CNT_MAX) {
        return nullptr;
    }
    static std::mutex cnt_mutex;
    static std::unordered_map<cudaStream_t, int *> cnt_bufs;
    std::lock_guard<std::mutex> lock(cnt_mutex);
    const auto it = cnt_bufs.find(stream);
    if (it != cnt_bufs.end()) {
        return it->second;
    }
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone) {
        return nullptr;
    }
    int * buf = nullptr;
    if (cudaMalloc((void **) &buf, (size_t) MMQSN_FIXUP_CNT_MAX*sizeof(int)) != cudaSuccess) {
        (void) cudaGetLastError();
        cnt_bufs[stream] = nullptr;   // do not retry: this stream keeps the fixup kernel
        return nullptr;
    }
    CUDA_CHECK(cudaMemsetAsync(buf, 0, (size_t) MMQSN_FIXUP_CNT_MAX*sizeof(int), stream));
    cnt_bufs[stream] = buf;
    return buf;
}

// ------------------------------------------------------------------------------------------------------------------
// Host entry
// ------------------------------------------------------------------------------------------------------------------

void ggml_cuda_mul_mat_qsn(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    GGML_TENSOR_BINARY_OP_LOCALS;

    GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ne00 % QK_K == 0 && ne01 % MMQSN_I == 0);
    GGML_ASSERT(ne11 >= 2 && ne11 <= 32);
    GGML_ASSERT(nb10 == sizeof(float) && nb0 == sizeof(float));

    const mmqsn_env_t & env = mmqsn_env();
    cudaStream_t stream = ctx.stream();
    const int id  = ggml_cuda_get_device();
    const int nsm = ggml_cuda_info().devices[id].nsm;
    const ggml_type type = src0->type;

    // 1. q8_1 activations exactly as MMQ quantizes them (mmq.cu ggml_cuda_mul_mat_q): padded K, MMQ block layout
    //    (unit u of column c at block u*ne11 + c), + MMQSN_Y_PAD blocks for the J-wide y reads of the last step. The PDL
    //    copy of the quantizer writes the same bytes and lets the matmul start its weight copies early.
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> q8(ctx.pool(),
        (size_t) (ne11*ne10_padded/QK8_1_MMQ)*sizeof(block_q8_1_mmq) + (size_t) MMQSN_Y_PAD*sizeof(block_q8_1_mmq));
    quantize_mmq_q8_1_pdl_cuda((const float *) src1->data, nullptr, q8.get(), type, ne10,
        nb11/sizeof(float), nb12/sizeof(float), nb13/sizeof(float), ne10_padded, ne11, 1, 1, stream);

    // 2. MMQ's J (smallest tile count: 8 up to 8 columns, else 16; [TAG_MMQSN_WIDE] 24 / 32 up to 24 / 32) and grid
    //    rule: tiling (one block per tile) when its efficiency is >= 90 %, else nsm stream-k blocks plus a fixup when
    //    ntiles % nsm != 0. Persistent tiling runs min(ntiles, nsm) blocks over the same tiles, each summed start to
    //    end, so the result does not change.
    const int J      = mmqsn_pick_J(ne11, env.loop);
    const int ntiles = (int) (ne01/MMQSN_I);
    const int nwaves = (ntiles + nsm - 1)/nsm;
    const bool tiling = 100*ntiles/(nsm*nwaves) >= 90;
    const int  grid   = tiling ? (env.persist ? std::min(ntiles, nsm) : ntiles) : nsm;
    const bool persist      = tiling && grid < ntiles;
    const bool fixup_needed = !tiling && ntiles % nsm != 0;

    ggml_cuda_pool_alloc<float> tmp(ctx.pool());
    if (fixup_needed) {
        tmp.alloc((size_t) grid*J*MMQSN_I);
    }

    // [TAG_MMQSN_FUSEFIX] the RING kernel does the fixup itself when it has zeroed tile counters for this stream.
    int * cnt = nullptr;
    if (fixup_needed && env.fusefix && env.loop == MMQSN_RING) {
        cnt = mmqsn_fixup_counters(stream, ntiles);
    }

    const int x_pf = (env.xpf || (src0->buffer && ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS)) ? 1 : 0;

    if (env.probe) {
        static std::mutex    probe_mutex;
        static std::set<int> probe_seen;
        const int key = ((int) type << 16) | (J << 8) | (env.loop << 4) | ((cnt != nullptr) << 3) | ((int) persist << 2) |
            ((int) fixup_needed << 1) | (grid == nsm);
        std::lock_guard<std::mutex> lock(probe_mutex);
        if (probe_seen.insert(key).second) {
            fprintf(stderr, "turbo-probe: mmqsn type=%s J=%d mode=%s grid=%d persist=%d fixup=%d fused=%d (first: ne01=%lld "
                    "ne00=%lld ne11=%lld ntiles=%d x_pf=%d pf=%d pf_run=%d l2hint=%d yh=%d)\n",
                    ggml_type_name(type), J, mmqsn_mode_name(env.loop), grid, (int) persist, (int) fixup_needed,
                    (int) (cnt != nullptr), (long long) ne01, (long long) ne00, (long long) ne11, ntiles, x_pf, env.pf_dist,
                    env.pf_run, env.l2hint, mmqsn_y_halves_rt(type, J, env.loop));
            fflush(stderr);
        }
    }

    mmqsn_args a;
    a.x              = (const char *) src0->data;
    a.y              = (const int *) q8.get();
    a.dst            = (float *) dst->data;
    a.tmp            = fixup_needed ? tmp.get() : nullptr;
    a.ncols_x        = (int) ne00;
    a.nrows_x        = (int) ne01;
    a.ne11           = (int) ne11;
    a.stride_row_x   = (int) (nb01/ggml_type_size(type));
    a.stride_col_dst = (int) (nb1/sizeof(float));
    a.J              = J;
    a.mode           = env.loop;
    a.grid           = grid;
    a.ntiles         = ntiles;
    a.persist        = persist ? 1 : 0;
    a.fixup          = fixup_needed && cnt == nullptr ? 1 : 0;
    a.x_pf           = x_pf;
    a.pf_dist        = env.pf_dist;
    a.pf_run         = env.pf_run;
    a.l2hint         = env.l2hint;
    a.cnt            = cnt;

    switch (type) {
        case GGML_TYPE_Q4_K:
            mmqsn_case<GGML_TYPE_Q4_K>(a, stream);
            break;
        case GGML_TYPE_Q5_K:
            mmqsn_case<GGML_TYPE_Q5_K>(a, stream);
            break;
        case GGML_TYPE_Q6_K:
            mmqsn_case<GGML_TYPE_Q6_K>(a, stream);
            break;
        default:
            GGML_ABORT("mmqsn: unsupported type %s", ggml_type_name(type));
    }

    if (env.check) {
        mmqsn_check(ctx, src0, src1, dst, env);
    }
#else
    GGML_UNUSED_VARS(ctx, src0, src1, dst);
    GGML_ABORT("mmqsn: CUDA only");
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}
