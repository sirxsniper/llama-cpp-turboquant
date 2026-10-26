#include "ggml-cuda.h"

#include <set>
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-cuda/allreduce.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/acc.cuh"
#include "ggml-cuda/add-id.cuh"
#include "ggml-cuda/arange.cuh"
#include "ggml-cuda/argmax.cuh"
#include "ggml-cuda/argsort.cuh"
#include "ggml-cuda/binbcast.cuh"
#include "ggml-cuda/clamp.cuh"
#include "ggml-cuda/col2im-1d.cuh"
#include "ggml-cuda/concat.cuh"
#include "ggml-cuda/conv-transpose-1d.cuh"
#include "ggml-cuda/conv2d.cuh"
#include "ggml-cuda/conv2d-dw.cuh"
#include "ggml-cuda/conv2d-transpose.cuh"
#include "ggml-cuda/conv3d.cuh"
#include "ggml-cuda/convert.cuh"
#include "ggml-cuda/count-equal.cuh"
#include "ggml-cuda/cpy.cuh"
#include "ggml-cuda/cross-entropy-loss.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/diagmask.cuh"
#include "ggml-cuda/diag.cuh"
#include "ggml-cuda/fattn.cuh"
#include "ggml-cuda/fwht.cuh"
#include "ggml-cuda/getrows.cuh"
#include "ggml-cuda/im2col.cuh"
#include "ggml-cuda/mmf.cuh"
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/mmsb.cuh"   // [TAG_MMSB] [TAG_SMALLB]
#include "ggml-cuda/moe-bridge.cuh" // [TAG_MOE_BRIDGE]
#include "ggml-fn-l4-gpumoe.h"          // [TAG_FN_L4_POST]
#include "ggml-cuda/qsa-mask.cuh"   // [TAG_FN_R4_QSA_POS]
#include "ggml-cuda/mmqsn.cuh"  // [TAG_MMQSN]
#include "ggml-cuda/mmvf.cuh"
#include "ggml-cuda/mmvq.cuh"
#include "ggml-cuda/moe-weighted-reduction.cuh"
#include "ggml-cuda/norm.cuh"
#include "ggml-cuda/opt-step-adamw.cuh"
#include "ggml-cuda/opt-step-sgd.cuh"
#include "ggml-cuda/out-prod.cuh"
#include "ggml-cuda/pad.cuh"
#include "ggml-cuda/pool2d.cuh"
#include "ggml-cuda/pool1d.cuh"
#include "ggml-cuda/quantize.cuh"
#include "ggml-cuda/rope.cuh"
#include "ggml-cuda/roll.cuh"
#include "ggml-cuda/scale.cuh"
#include "ggml-cuda/snake.cuh"
#include "ggml-cuda/softcap.cuh"
#include "ggml-cuda/softmax.cuh"
#include "ggml-cuda/ssm-conv.cuh"
#include "ggml-cuda/ssm-scan.cuh"
#include "ggml-cuda/sum.cuh"
#include "ggml-cuda/sumrows.cuh"
#include "ggml-cuda/top-k.cuh"
#include "ggml-cuda/mean.cuh"
#include "ggml-cuda/tsembd.cuh"
#include "ggml-cuda/topk-moe.cuh"
#include "ggml-cuda/unary.cuh"
#include "ggml-cuda/upscale.cuh"
#include "ggml-cuda/wkv.cuh"
#include "ggml-cuda/gla.cuh"
#include "ggml-cuda/gated_delta_net.cuh"
#include "ggml-cuda/dsv4-hc.cuh"
#include "ggml-cuda/set.cuh"
#include "ggml-cuda/set-rows.cuh"
#include "ggml-cuda/turbot-set-rows.cuh"   // [TAG_TURBOT]
#include "ggml-cuda/turbo-wht.cuh"
#include "ggml-cuda/pad_reflect_1d.cuh"
#include "ggml-cuda/solve_tri.cuh"
#include "ggml-cuda/tri.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/fill.cuh"
#include "ggml-cuda/lightning-indexer.cuh"
#include "ggml-cuda/fn-l3.cuh"            // [TAG_FN_L3_GPU]
#include "ggml-cuda/fn-l4-qsa.cuh"        // [TAG_FN_L4_QSA]
#include "ggml-cuda/fn-l4-hc.cuh"         // [TAG_FN_L4_HC]
#include "ggml-cuda/fn-l22-dmafetch.cuh"   // [TAG_FN_L22_DMASHARE] R-P14-4
#include "ggml-cuda/top-k-mask.cuh"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cfloat>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

static_assert(sizeof(half) == sizeof(ggml_fp16_t), "wrong fp16 size");

#define GGML_LOG_WARN_ONCE(str) \
    { static std::once_flag warn_flag; std::call_once(warn_flag, []() { GGML_LOG_WARN(str); }); }

[[noreturn]]
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    int id = -1; // in case cudaGetDevice fails
    (void)cudaGetDevice(&id);

    GGML_LOG_ERROR(GGML_CUDA_NAME " error: %s\n", msg);
    GGML_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", id, func, file, line);
    GGML_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ABORT to get a stack trace
    GGML_ABORT(GGML_CUDA_NAME " error");
}

// map a (possibly virtual) device id to the physical CUDA device that backs it
static int ggml_cuda_get_physical_device(int device) {
    const ggml_cuda_device_info & info = ggml_cuda_info();
    GGML_ASSERT(device >= 0 && device < info.device_count);
    return info.devices[device].physical_device;
}

// this is faster on Windows
// probably because the Windows CUDA libraries forget to make this check before invoking the drivers
void ggml_cuda_set_device(int device) {
    // translate the (possibly virtual) device id to the physical CUDA device that backs it
    const int physical_device = ggml_cuda_get_physical_device(device);

    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));

    if (physical_device == current_device) {
        return;
    }

    CUDA_CHECK(cudaSetDevice(physical_device));
}

int ggml_cuda_get_device() {
    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

static bool ggml_cuda_env_flag(const char * name) {
    const char * env = getenv(name);
    return env != nullptr && std::atoi(env) != 0;
}

static int ggml_cuda_env_int(const char * name, int default_value) {
    const char * env = getenv(name);
    return env != nullptr ? std::atoi(env) : default_value;
}

static bool ggml_cuda_env_set(const char * name) {
    return getenv(name) != nullptr;
}

static bool ggml_cuda_unified_memory_enabled() {
    static const bool enabled = ggml_cuda_env_flag("GGML_CUDA_ENABLE_UNIFIED_MEMORY");
    return enabled;
}

static bool ggml_cuda_pinned_memory_disabled() {
    static const bool disabled = ggml_cuda_env_flag("GGML_CUDA_NO_PINNED");
    return disabled;
}

static bool ggml_cuda_register_host_enabled() {
    static const bool enabled = ggml_cuda_env_flag("GGML_CUDA_REGISTER_HOST");
    return enabled;
}

static bool ggml_cuda_fusion_disabled() {
    static const bool disabled = ggml_cuda_env_flag("GGML_CUDA_DISABLE_FUSION");
    return disabled;
}

static bool ggml_cuda_p2p_enabled() {
    static const bool enabled = ggml_cuda_env_set("GGML_CUDA_P2P");
    return enabled;
}

static int ggml_cuda_op_offload_min_batch() {
    static const int min_batch = ggml_cuda_env_int("GGML_OP_OFFLOAD_MIN_BATCH", 32);
    return min_batch;
}

static cudaError_t ggml_cuda_device_malloc(void ** ptr, size_t size, int device) {
    ggml_cuda_set_device(device);
    cudaError_t err;
    if (ggml_cuda_unified_memory_enabled()) {
        err = cudaMallocManaged(ptr, size);
#if defined(GGML_USE_HIP)
        if (err == hipSuccess) {
            // hipMemAdviseSetCoarseGrain is an optional performance hint;
            // ignore errors (e.g. hipErrorInvalidValue on some APU/iGPU configs).
            (void)cudaMemAdvise(*ptr, size, hipMemAdviseSetCoarseGrain, device);
            (void)hipGetLastError(); // clear any error
        }

        // fall back to cudaMalloc if not supported (e.g. on Windows)
        if (err == hipErrorNotSupported) {
            static bool warned_unsupported = false;
            if (!warned_unsupported) {
                GGML_LOG_WARN("hipMallocManaged unsupported, falling back to hipMalloc.\n");
                warned_unsupported = true;
            }

            err = cudaMalloc(ptr, size);
        }
#endif // defined(GGML_USE_HIP)
    } else {
        err = cudaMalloc(ptr, size);
    }
    return err;
}

#if defined(GGML_USE_HIP)
static int ggml_cuda_parse_id(char devName[]) {
    // A list of possible Target IDs can be found under the rocclr/clr repo in device.cpp
    // these values are not stable so this is susceptible to breakage
    // https://github.com/ROCm/clr/blob/amd-staging/rocclr/device/device.cpp
    int archMajor = 0x0;
    int archMinor = 0x0;
    int archNum = GGML_CUDA_CC_OFFSET_AMD;
    int archLen = strlen(devName);
    char archName[archLen + 1];

    // strip leading 'gfx' while copying into our buffer
    if (archLen > 3) {
        strcpy(archName, &devName[3]);
        archLen -= 3;
    }

    // trim trailing :xnack- or :sramecc- statuses
    archLen = strcspn(archName, ":");
    archName[archLen] = '\0';

    // tease out the version information
    if (archLen > 8) {
        // versions labeled generic use '-' as delimiter
        // strip the trailing "-generic" then iterate through what remains
        if ((strstr(archName, "-generic"))) {
            archName[archLen - 8] = '\0';
            char * pch;
            if ((pch = strtok(archName, "-"))) {
                archMajor = (int)strtoul(pch, 0, 16);
                if ((pch = strtok(NULL, "-"))) {
                    archMinor = 0x10 * (int)strtoul(pch, 0, 16);
                }
            }
        }
    } else if (archLen >= 3) {
        // last two digits should be the minor * 0x10 + stepping
        archMinor = (int)strtoul(&archName[archLen - 2], 0, 16);
        archName[archLen - 2] = '\0';

        // only the major version remains
        archMajor = (int)strtoul(archName, 0, 16);
    }
    archNum += archMajor * 0x100;
    archNum += archMinor;

    return archNum;
}
#endif // defined(GGML_USE_HIP)

static ggml_cuda_device_info ggml_cuda_init() {
    ggml_cuda_device_info info = {};

    cudaError_t err = cudaGetDeviceCount(&info.physical_device_count);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: failed to initialize " GGML_CUDA_NAME ": %s\n", __func__, cudaGetErrorString(err));
        return info;
    }

    GGML_ASSERT(info.physical_device_count <= GGML_CUDA_MAX_DEVICES);

    // by default expose exactly the physical devices; GGML_CUDA_DEVICES can request a different
    // number of (virtual) devices to emulate multi-GPU systems on a machine with fewer GPUs
    info.device_count = info.physical_device_count;

    const char * devices_env = getenv("GGML_CUDA_DEVICES");
    if (devices_env != nullptr && info.physical_device_count > 0) {
        const int requested = atoi(devices_env);
        if (requested > 0) {
            info.device_count = requested;
        } else {
            GGML_LOG_WARN("%s: ignoring invalid GGML_CUDA_DEVICES=\"%s\"\n", __func__, devices_env);
        }
    }

    if (info.device_count > GGML_CUDA_MAX_DEVICES) {
        GGML_LOG_WARN("%s: requested %d devices, clamping to GGML_CUDA_MAX_DEVICES=%d\n",
                      __func__, info.device_count, GGML_CUDA_MAX_DEVICES);
        info.device_count = GGML_CUDA_MAX_DEVICES;
    }

    // map each (virtual) device to a backing physical device (round-robin), assign each its index
    // among the (virtual) devices sharing that physical GPU, and store the per-physical share count
    int physical_share_count[GGML_CUDA_MAX_DEVICES] = {};
    GGML_ASSERT(info.device_count == 0 || info.physical_device_count > 0);
    for (int id = 0; id < info.device_count; ++id) {
        info.devices[id].physical_device = id % info.physical_device_count;
        info.devices[id].virtual_index  = physical_share_count[info.devices[id].physical_device]++;
    }

    int64_t total_vram = 0;
    for (int id = 0; id < info.physical_device_count; ++id) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
        total_vram += prop.totalGlobalMem;
    }
    GGML_LOG_INFO("%s: found %d " GGML_CUDA_NAME " devices (Total VRAM: %zu MiB):\n",
                  __func__, info.physical_device_count, (size_t)(total_vram / (1024 * 1024)));
    if (info.device_count != info.physical_device_count) {
        GGML_LOG_INFO("%s: emulating %d virtual device(s) on %d physical device(s) (GGML_CUDA_DEVICES)\n",
                      __func__, info.device_count, info.physical_device_count);
    }
    total_vram = 0;

    std::vector<std::pair<int, std::string>> turing_devices_without_mma;
    for (int id = 0; id < info.device_count; ++id) {
        const int physical_id = info.devices[id].physical_device;

        int device_vmm = 0;

#if defined(GGML_USE_VMM)
        CUdevice device;
        CU_CHECK(cuDeviceGet(&device, physical_id));
        CU_CHECK(cuDeviceGetAttribute(&device_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device));

        if (device_vmm) {
            CUmemAllocationProp alloc_prop = {};
            alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            alloc_prop.location.id = physical_id;
            CU_CHECK(cuMemGetAllocationGranularity(&info.devices[id].vmm_granularity, &alloc_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        }
#endif // defined(GGML_USE_VMM)
        info.devices[id].vmm = !!device_vmm;

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, physical_id));

        // a virtual device owns only a share of its physical GPU's memory; report that share so the
        // logged per-device VRAM sums to the physical total above.
        GGML_ASSERT(physical_share_count[physical_id] > 0);
        info.devices[id].physical_share_count = physical_share_count[physical_id];
        const size_t device_vram = prop.totalGlobalMem / info.devices[id].physical_share_count;
        const size_t device_vram_mib = device_vram / (1024 * 1024);

        info.default_tensor_split[id] = total_vram;
        total_vram += device_vram;
        info.devices[id].integrated = false; // Temporarily disabled due to issues with corrupted output (e.g. #15034)
        info.devices[id].nsm        = prop.multiProcessorCount;
        info.devices[id].smpb       = prop.sharedMemPerBlock;
        info.devices[id].warp_size  = prop.warpSize;

        int supports_coop_launch = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&supports_coop_launch, cudaDevAttrCooperativeLaunch, physical_id));
        info.devices[id].supports_cooperative_launch = !!supports_coop_launch;

#if defined(GGML_USE_HIP)
        info.devices[id].smpbo = prop.sharedMemPerBlock;

        info.devices[id].cc = ggml_cuda_parse_id(prop.gcnArchName);
        if ((info.devices[id].cc & 0xff00) == 0x0) {
            GGML_LOG_WARN("invalid architecture ID received for device %d %s: %s  cc %d.%d\n",
                            id, prop.name, prop.gcnArchName, prop.major, prop.minor);

            // Fallback to prop.major and prop.minor
            if (prop.major > 0) {
                info.devices[id].cc = GGML_CUDA_CC_OFFSET_AMD + prop.major * 0x100;
                info.devices[id].cc += prop.minor * 0x10;
            }
        }
        GGML_LOG_INFO("  Device %d: %s, %s (0x%x), VMM: %s, Wave Size: %d, VRAM: %zu MiB\n",
                      id, prop.name, prop.gcnArchName, info.devices[id].cc & 0xffff,
                      device_vmm ? "yes" : "no", prop.warpSize,
                      device_vram_mib);
#elif defined(GGML_USE_MUSA)
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = GGML_CUDA_CC_OFFSET_MTHREADS + prop.major * 0x100;
        info.devices[id].cc += prop.minor * 0x10;
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n",
                      id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                      device_vram_mib);
#else
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = 100*prop.major + 10*prop.minor;
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n",
                      id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                      device_vram_mib);
        std::string device_name(prop.name);
        if (device_name == "NVIDIA GeForce MX450") {
            turing_devices_without_mma.push_back({ id, device_name });
        } else if (device_name == "NVIDIA GeForce MX550") {
            turing_devices_without_mma.push_back({ id, device_name });
        } else if (device_name.substr(0, 21) == "NVIDIA GeForce GTX 16") {
            turing_devices_without_mma.push_back({ id, device_name });
        }

        // Temporary performance fix:
        // Setting device scheduling strategy for iGPUs with cc121 to "spinning" to avoid delays in cuda synchronize calls.
        // TODO: Check for future drivers the default scheduling strategy and
        // remove this call again when cudaDeviceScheduleSpin is default.
        if (prop.major == 12 && prop.minor == 1) {
            CUDA_CHECK(cudaSetDevice(physical_id));
            CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
        }

        // [TAG_CUDA_SCHED] TURBO_CUDA_SCHED=spin|yield|blocking pins the host wait policy used by every
        // synchronize on this device (the driver's "auto" picks per process, which is one candidate
        // for the decode step costing 25.3 ms in one server launch and 28.0 ms in the next).
        {
            static const int sched_flag = [] {
                const char * e = getenv("TURBO_CUDA_SCHED");
                if (e == nullptr) {
                    return -1;
                }
                if (strcmp(e, "spin") == 0)     { return (int) cudaDeviceScheduleSpin; }
                if (strcmp(e, "yield") == 0)    { return (int) cudaDeviceScheduleYield; }
                if (strcmp(e, "blocking") == 0) { return (int) cudaDeviceScheduleBlockingSync; }
                return -1;
            }();
            if (sched_flag >= 0) {
                CUDA_CHECK(cudaSetDevice(physical_id));
                CUDA_CHECK(cudaSetDeviceFlags((unsigned) sched_flag));
                GGML_LOG_INFO("%s: device %d host wait policy set from TURBO_CUDA_SCHED (%s)\n", __func__, id, getenv("TURBO_CUDA_SCHED"));
            }
        }

#endif  // defined(GGML_USE_HIP)
    }

    if (ggml_cuda_highest_compiled_arch(GGML_CUDA_CC_TURING) >= GGML_CUDA_CC_TURING && !turing_devices_without_mma.empty()) {
        GGML_LOG_INFO("The following devices will have suboptimal performance due to a lack of tensor cores:\n");
        for (size_t device_pos = 0; device_pos < turing_devices_without_mma.size(); device_pos++) {
            GGML_LOG_INFO(
                "  Device %d: %s\n", turing_devices_without_mma[device_pos].first, turing_devices_without_mma[device_pos].second.c_str());
        }
        GGML_LOG_INFO(
            "Consider compiling with CMAKE_CUDA_ARCHITECTURES=61-virtual;80-virtual and DGGML_CUDA_FORCE_MMQ to force the use of the Pascal code for Turing.\n");
    }

    for (int id = 0; id < info.device_count; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }

    // configure logging to stdout
    // CUBLAS_CHECK(cublasLoggerConfigure(1, 1, 0, nullptr));

    if (ggml_cuda_p2p_enabled()) {
        for (int id = 0; id < info.physical_device_count; ++id) {
            CUDA_CHECK(cudaSetDevice(id));
            for (int id_other = 0; id_other < info.physical_device_count; ++id_other) {
                if (id == id_other) {
                    continue;
                }
                int can_access_peer;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id, id_other));
                if (can_access_peer) {
                    CUDA_CHECK(cudaDeviceEnablePeerAccess(id_other, 0));
                }
            }
        }
    }

    return info;
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = ggml_cuda_init();
    return info;
}

// #define DEBUG_CUDA_MALLOC

// [TAG_FN_L22_OOMHOOK] defined after ggml_cuda_lock_counter: the libllama OOM hook (pins), outside a capture
static void ggml_cuda_fn_l22_oom_precheck(size_t need);
static bool ggml_cuda_fn_l22_oom_retry(size_t need, int code = 1);

// buffer pool for cuda (legacy)
struct ggml_cuda_pool_leg : public ggml_cuda_pool {
    static const int MAX_BUFFERS = 256;

    int device;
    struct ggml_cuda_buffer {
        void * ptr = nullptr;
        size_t size = 0;
    };

    ggml_cuda_buffer buffer_pool[MAX_BUFFERS] = {};
    size_t pool_size = 0;
    size_t live      = 0; // [TAG_FN_L3_VRAM_TRIM] bytes handed out and not returned
    size_t live_hwm  = 0;

    explicit ggml_cuda_pool_leg(int device) :
        device(device) {
    }

    // [TAG_FN_L3_VRAM_TRIM] the cached buffers go back to the driver; handed-out ones stay (the pool does not track them)
    size_t reserved() const override { return pool_size; }
    size_t used_hwm() const override { return live_hwm; }
    void   reset_hwm() override { live_hwm = live; }
    size_t trim(size_t keep) override {
        GGML_UNUSED(keep);
        const size_t before = pool_size;
        clear_pool();
        return before - pool_size;
    }

    uint64_t n_flush = 0; // [TAG_UPSTREAM_26771]
    uint64_t flush_count() const override { return n_flush; }

    ~ggml_cuda_pool_leg() {
        clear_pool();
        GGML_ASSERT(pool_size == 0);
    }

    void clear_pool() {
        ggml_cuda_set_device(device);
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                CUDA_CHECK(cudaFree(b.ptr));
                pool_size -= b.size;
                b.ptr  = nullptr;
                b.size = 0;
            }
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
        void * ptr = alloc_raw(size, actual_size);
        live    += *actual_size; // [TAG_FN_L3_VRAM_TRIM]
        live_hwm = std::max(live_hwm, live);
        return ptr;
    }

    void * alloc_raw(size_t size, size_t * actual_size) {
#ifdef DEBUG_CUDA_MALLOC
        int nnz = 0;
        size_t max_size = 0;
#endif
        size_t best_diff = 1ull << 36;
        int ibest = -1;
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
#ifdef DEBUG_CUDA_MALLOC
                ++nnz;
                if (b.size > max_size) max_size = b.size;
#endif
                if (b.size >= size) {
                    size_t diff = b.size - size;
                    if (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        if (!best_diff) {
                            void * ptr = b.ptr;
                            *actual_size = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            return ptr;
                        }
                    }
                }
            }
        }
        if (ibest >= 0) {
            ggml_cuda_buffer& b = buffer_pool[ibest];
            void * ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
        void * ptr;
        size_t look_ahead_size = (size_t) (1.05 * size);
        look_ahead_size = 256 * ((look_ahead_size + 255)/256);
        ggml_cuda_set_device(device);
        cudaError_t err = ggml_cuda_device_malloc(&ptr, look_ahead_size, device);
        if (err == cudaErrorMemoryAllocation) {
            (void)cudaGetLastError();
            const size_t cached_bytes = pool_size;
            GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: alloc of %.2f MiB failed, flushing %.2f MiB of cached buffers and retrying\n",
                           device, look_ahead_size/1024.0/1024.0, cached_bytes/1024.0/1024.0);
            CUDA_CHECK(cudaDeviceSynchronize());
            clear_pool();
            // [TAG_UPSTREAM_26771] captured graphs may hold the freed buffers: the next graph_compute drops them
            n_flush += cached_bytes > 0 ? 1 : 0;
            err = ggml_cuda_device_malloc(&ptr, look_ahead_size, device);
            if (err == cudaSuccess) {
                GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: retry succeeded\n", device);
            }
        }
        CUDA_CHECK(err);
        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;
#ifdef DEBUG_CUDA_MALLOC
        GGML_LOG_INFO("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, device, nnz,
                           (uint32_t)(max_size / 1024 / 1024), (uint32_t)(pool_size / 1024 / 1024), (uint32_t)(size / 1024 / 1024));
#endif
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        live -= std::min(live, size); // [TAG_FN_L3_VRAM_TRIM]
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                return;
            }
        }
        GGML_LOG_DEBUG(GGML_CUDA_NAME " buffer pool full, increase MAX_CUDA_BUFFERS\n");
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaFree(ptr));
        pool_size -= size;
    }
};

// pool with virtual memory
#if defined(GGML_USE_VMM)
struct ggml_cuda_pool_vmm : public ggml_cuda_pool {
    static const size_t CUDA_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB

    int device;
    int physical_device;
    CUdeviceptr pool_addr = 0;
    size_t pool_used = 0;
    size_t pool_size = 0;
    size_t pool_hwm  = 0; // [TAG_FN_L3_VRAM_TRIM]
    size_t cap_top   = 0; // [TAG_FN_L22_GRAPHKEEP] highest pool_used since cap_begin()
    size_t granularity;
    // [TAG_FN_L3_VRAM_TRIM] every mapping in address order (HIP unmaps them one by one, trim() unmaps from the top)
    std::vector<std::pair<CUdeviceptr, size_t>> mappings;

    explicit ggml_cuda_pool_vmm(int device) :
        device(device),
        physical_device(ggml_cuda_get_physical_device(device)),
        granularity(ggml_cuda_info().devices[device].vmm_granularity) {
    }

    ~ggml_cuda_pool_vmm() {
        if (pool_addr != 0) {
#if defined(GGML_USE_HIP)
            // Workaround for https://github.com/ROCm/ROCR-Runtime/issues/285
            for (std::pair<CUdeviceptr, size_t> & mapping : mappings) {
                CU_CHECK(cuMemUnmap(mapping.first, mapping.second));
            }
#else
            if (pool_size > 0) { // [TAG_FN_L3_VRAM_TRIM] a trimmed pool may hold nothing
                CU_CHECK(cuMemUnmap(pool_addr, pool_size));
            }
#endif
            CU_CHECK(cuMemAddressFree(pool_addr, CUDA_POOL_VMM_MAX_SIZE));
        }
    }

    // [TAG_FN_L3_VRAM_TRIM]
    size_t reserved() const override { return pool_size; }
    size_t used_hwm() const override { return pool_hwm; }
    void   reset_hwm() override { pool_hwm = pool_used; }
    void   cap_begin() override { cap_top = pool_used; }
    size_t cap_hwm() const override { return cap_top; }
    size_t trim(size_t keep) override {
        if (pool_used != 0) {
            return 0;
        }
        size_t freed = 0;
        while (!mappings.empty() && pool_size - mappings.back().second >= keep) {
            const std::pair<CUdeviceptr, size_t> m = mappings.back();
            GGML_ASSERT(m.first + m.second == pool_addr + pool_size);
            CU_CHECK(cuMemUnmap(m.first, m.second));
            pool_size -= m.second;
            freed     += m.second;
            mappings.pop_back();
        }
        return freed;
    }

    void * alloc(size_t size, size_t * actual_size) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        const size_t alignment = 128;
        size = alignment * ((size + alignment - 1) / alignment);

        size_t avail = pool_size - pool_used;

        if (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_size = size - avail;
            reserve_size = granularity * ((reserve_size + granularity - 1) / granularity);

            GGML_ASSERT(pool_size + reserve_size <= CUDA_POOL_VMM_MAX_SIZE);

            // allocate more physical memory
            CUmemAllocationProp prop = {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = physical_device;
            CUmemGenericAllocationHandle handle;
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
            ggml_cuda_fn_l22_oom_precheck(reserve_size); // [TAG_FN_L22_OOMHOOK]
            CUresult cr = cuMemCreate(&handle, reserve_size, &prop, 0);
            // [TAG_FN_L23_PINFIX] R-L23-6: a failed retry aborts here, so the hook gets two attempts (code 3, then 4)
            for (int k = 0; k < 2 && cr == CUDA_ERROR_OUT_OF_MEMORY && ggml_cuda_fn_l22_oom_retry(reserve_size, 3 + k); ++k) { // [TAG_FN_L22_OOMHOOK]
                cr = cuMemCreate(&handle, reserve_size, &prop, 0);
            }
            CU_CHECK(cr);
#else
            CU_CHECK(cuMemCreate(&handle, reserve_size, &prop, 0));
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

            // reserve virtual address space (if not already reserved)
            if (pool_addr == 0) {
                CU_CHECK(cuMemAddressReserve(&pool_addr, CUDA_POOL_VMM_MAX_SIZE, 0, 0, 0));
            }

            // map at the end of the pool
            CUdeviceptr start_ptr = (CUdeviceptr)((char *)(pool_addr) + pool_size);
            CU_CHECK(cuMemMap(start_ptr, reserve_size, 0, handle, 0));
            mappings.push_back({start_ptr, reserve_size});

            // the memory allocation handle is no longer needed after mapping
            CU_CHECK(cuMemRelease(handle));

            // VMM Bug fix for P2P access if GGML_CUDA_P2P is set, or if NCCL build
            bool use_peer_access = ggml_cuda_p2p_enabled();
#if defined(GGML_USE_NCCL)
            use_peer_access = true;
#endif // defined(GGML_USE_NCCL)

            if (use_peer_access) {
                // NCCL implicitly enables peer access (cudaDeviceEnablePeerAccess), and
                // GGML_CUDA_P2P enables it explicitly. Unlike cudaMalloc buffers, VMM
                // allocations do not become peer-accessible from that alone, so access
                // must be granted explicitly here. With virtual devices, grant access
                // on the backing *physical* devices (deduplicated, since several
                // virtual devices can map to the same physical GPU).
                std::vector<CUmemAccessDesc> access_descs;
                bool physical_seen[GGML_CUDA_MAX_DEVICES] = {};
                const int device_count = ggml_cuda_info().device_count;
                for (int id = 0; id < device_count; ++id) {
                    const int id_physical = ggml_cuda_get_physical_device(id);
                    if (id_physical != physical_device) {
                        int can_access_peer = 0;
                        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id_physical, physical_device));
                        if (!can_access_peer) {
                            continue;
                        }
                    }
                    if (physical_seen[id_physical]) {
                        continue;
                    }
                    physical_seen[id_physical] = true;
                    CUmemAccessDesc access = {};
                    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                    access.location.id = id_physical;
                    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                    access_descs.push_back(access);
                }
                CU_CHECK(cuMemSetAccess(start_ptr, reserve_size, access_descs.data(), access_descs.size()));
            } else {
                // set access for non P2P
                CUmemAccessDesc access = {};
                access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                access.location.id = physical_device;
                access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                CU_CHECK(cuMemSetAccess(start_ptr, reserve_size, &access, 1));
            }

            // add to the pool
            pool_size += reserve_size;

            //printf("cuda pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
            //       device, (unsigned long long) (pool_size/1024/1024),
            //       (unsigned long long) (reserve_size/1024/1024));
        }

        GGML_ASSERT(pool_addr != 0);

        void * ptr = (void *) ((CUdeviceptr)((char *)(pool_addr) + pool_used));
        *actual_size = size;
        pool_used += size;
        pool_hwm   = std::max(pool_hwm, pool_used); // [TAG_FN_L3_VRAM_TRIM]
        cap_top    = std::max(cap_top, pool_used);  // [TAG_FN_L22_GRAPHKEEP]

#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: allocated %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        return ptr;
    }

    void free(void * ptr, size_t size) override {
#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: freed %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == (void *) ((char *)(pool_addr) + pool_used));
    }
};
#endif // defined(GGML_USE_VMM)

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int                  device,
                                                                               [[maybe_unused]] int stream_no) {
#if defined(GGML_USE_VMM)
    if (ggml_cuda_info().devices[device].vmm) {
        return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_vmm(device));
    }
#endif // defined(GGML_USE_VMM)
    return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_leg(device));
}

// destroying a cuBLAS handle while a graph is being captured in a different thread can result in a CUDA error
// this lock is used to ensure that no cuBLAS handle is destroyed while a graph is being captured

static std::mutex ggml_cuda_lock;
static std::condition_variable ggml_cuda_lock_cv;
static std::atomic<int> ggml_cuda_lock_counter;

// [TAG_FN_L22_OOMHOOK] plan 3.6: a device allocation after ready asks the owner of pinned host memory (libllama, set
// through the proc "ggml_backend_cuda_set_oom_hook") first to check the device headroom (after_failure 0: give memory
// back only if the headroom is short) and, if it still fails, to give some back once (after_failure 1). Never inside a
// capture: the hook unregisters host memory, which a capturing stream must not see.
typedef bool (*ggml_cuda_fn_l22_oom_fn_t)(size_t need, int after_failure, void * ud);
static std::atomic<ggml_cuda_fn_l22_oom_fn_t> ggml_cuda_fn_l22_oom_fn{nullptr};
static std::atomic<void *>                    ggml_cuda_fn_l22_oom_ud{nullptr};

static void ggml_cuda_fn_l22_oom_precheck(size_t need) {
    const ggml_cuda_fn_l22_oom_fn_t fn = ggml_cuda_fn_l22_oom_fn.load();
    if (fn != nullptr && ggml_cuda_lock_counter.load(std::memory_order_relaxed) == 0) {
        fn(need, 0, ggml_cuda_fn_l22_oom_ud.load());
    }
}

// code ([TAG_FN_L23_PINFIX] R-L23-6): 1 = device allocation, 2 = pinned host allocation, 3 / 4 = device allocation whose failed retry aborts (first / second attempt)
static bool ggml_cuda_fn_l22_oom_retry(size_t need, int code) {
    const ggml_cuda_fn_l22_oom_fn_t fn = ggml_cuda_fn_l22_oom_fn.load();
    if (fn == nullptr || ggml_cuda_lock_counter.load(std::memory_order_relaxed) != 0) {
        return false;
    }
    const bool freed = fn(need, code, ggml_cuda_fn_l22_oom_ud.load());
    GGML_LOG_WARN("ggml_cuda: [TAG_FN_L22_OOMHOOK] %s allocation of %.1f MiB failed (code %d): the OOM hook %s\n", code == 2 ? "pinned host" : "device",
            need/1048576.0, code, freed ? "gave memory back, retry" : "had nothing to give");
    return freed;
}

static void ggml_backend_cuda_set_oom_hook(ggml_cuda_fn_l22_oom_fn_t fn, void * ud) {
    ggml_cuda_fn_l22_oom_ud.store(ud);
    ggml_cuda_fn_l22_oom_fn.store(fn);
}

// [TAG_FN_L22_GRAPHKEEP] [TAG_FN_L22_GRAPH2] switches read once, at the first graph (after a model profile set them)
int ggml_cuda_fn_l22_graph_keep(void) {
    static const int v = [] {
        const char * e = getenv("GGML_CUDA_FN_L22_GRAPHKEEP");
        return e ? atoi(e) : 0;
    }();
    return v;
}

static bool ggml_cuda_fn_l22_graph2() {
    static const bool v = [] {
        const char * e = getenv("GGML_CUDA_FN_L22_GRAPH2");
        return e != nullptr && atoi(e) != 0;
    }();
    return v;
}

static bool ggml_cuda_fn_l22_graph_diag() {
    static const bool v = [] {
        const char * e = getenv("GGML_CUDA_FN_L22_GRAPHDIAG");
        return e != nullptr && atoi(e) != 0;
    }();
    return v;
}

ggml_backend_cuda_context::~ggml_backend_cuda_context() {
    std::unique_lock<std::mutex> lock(ggml_cuda_lock);
    ggml_cuda_lock_cv.wait(lock, []{ return ggml_cuda_lock_counter.load(std::memory_order_relaxed) == 0; });

    if (copy_event != nullptr) {
        CUDA_CHECK(cudaEventDestroy(copy_event));
    }
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        for (int j = 0; j < GGML_CUDA_MAX_STREAMS; ++j) {
            if (streams[i][j] != nullptr) {
                CUDA_CHECK(cudaStreamDestroy(streams[i][j]));
            }
            if (cublas_handles[i][j] != nullptr) {
                CUBLAS_CHECK(cublasDestroy(cublas_handles[i][j]));
            }
            if (cublas_workspaces[i][j] != nullptr) {
                CUDA_CHECK(cudaFree(cublas_workspaces[i][j]));
            }
        }
    }
    if (mmvq_q8.buf != nullptr) { // [TAG_MMVQ_Q8_REUSE]
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaFree(mmvq_q8.buf));
        mmvq_q8.buf = nullptr;
    }
    if (fn_l4_list_buf != nullptr) { // [TAG_FN_L4_QSA_LIST]
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaFree(fn_l4_list_buf));
        fn_l4_list_buf = nullptr;
    }
    for (ggml_cuda_upload_slot & s : upload_slot) { // [TAG_FN_L16_UPLOAD]
        if (s.host == nullptr) {
            continue; // never allocated (a slot has all three or none)
        }
        ggml_cuda_set_device(device);
        if (s.done != nullptr) {
            CUDA_CHECK(cudaEventSynchronize(s.done));
            CUDA_CHECK(cudaEventDestroy(s.done));
        }
        if (s.dev != nullptr) {
            CUDA_CHECK(cudaFree(s.dev));
        }
        if (s.host != nullptr) {
            CUDA_CHECK(cudaFreeHost(s.host));
        }
        s = ggml_cuda_upload_slot();
    }
}


// cuda buffer

#if defined(GGML_USE_VMM) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#define GGML_CUDA_VMM_BUFFERS // [TAG_FN_L3_VRAM_CBUF]
#endif

struct ggml_backend_cuda_buffer_context {
    int device;
    void * dev_ptr = nullptr;
    std::string name;

    // [TAG_FN_L3_VRAM_CBUF] vmm_size > 0: dev_ptr starts a reserved address range of vmm_size bytes, and only the
    // vmm_segs (offset, size) ranges have physical memory (ggml_backend_cuda_vmm_buffer_*); else a cudaMalloc buffer
    size_t vmm_size = 0;
    size_t vmm_gran = 0;
    std::vector<std::pair<size_t, size_t>> vmm_segs;

    ggml_backend_cuda_buffer_context(int device, void * dev_ptr) :
        device(device), dev_ptr(dev_ptr),
        name(GGML_CUDA_NAME + std::to_string(device)) {
    }

    ~ggml_backend_cuda_buffer_context() {
        if (vmm_size == 0) {
            CUDA_CHECK(cudaFree(dev_ptr));
            return;
        }
#ifdef GGML_CUDA_VMM_BUFFERS
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaDeviceSynchronize()); // cudaFree would wait for the device too
        for (const auto & s : vmm_segs) {
            CU_CHECK(cuMemUnmap((CUdeviceptr) dev_ptr + s.first, s.second));
        }
        CU_CHECK(cuMemAddressFree((CUdeviceptr) dev_ptr, vmm_size));
#endif
    }
};

static void ggml_backend_cuda_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    delete ctx;
}

static bool ggml_backend_buffer_is_cuda(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_cuda_buffer_free_buffer;
}

static void * ggml_backend_cuda_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    return ctx->dev_ptr;
}

static enum ggml_status ggml_backend_cuda_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        return GGML_STATUS_SUCCESS;
    }

    if (ggml_is_quantized(tensor->type) && tensor->view_src == nullptr && ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        // initialize padding to 0 to avoid possible NaN values
        const size_t original_size = ggml_nbytes(tensor);
        const size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size) {
            ggml_cuda_set_device(ctx->device);
            // [TAG_FN_L21_PADMEMSET] LLAMA_FN_L21_PADMEMSET (default 1; upstream #28782): the pad is cleared on the per-thread
            // stream and waited for there. cudaMemset runs on the legacy default stream, which syncs with every blocking stream
            // of the device; the bridge, warm and upload threads init tensors beside graph capture. 0 = cudaMemset as before.
            static const bool l21_padmemset = [] { const char * e = getenv("LLAMA_FN_L21_PADMEMSET"); return !(e && e[0] && atoi(e) == 0); }();
            if (l21_padmemset) {
                CUDA_CHECK(cudaMemsetAsync((char *)tensor->data + original_size, 0, padded_size - original_size, cudaStreamPerThread));
                CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
            } else {
                CUDA_CHECK(cudaMemset((char *)tensor->data + original_size, 0, padded_size - original_size));
            }
        }
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync((char *) tensor->data + offset, value, size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

// [TAG_FN_L23_PINFIX] a host -> device copy whose source starts inside a range registered with cudaHostRegister but runs
// past its end is refused by CUDA (cudaErrorInvalidValue: nothing is queued, the error is not sticky). libllama registers
// per-expert ranges of the mapped model ([TAG_FN_L22_PIN]) and now keeps them, so a caller that does not split its copy at
// the registered edges (FAULT 2026-10-21 10:10: a hot-set refill slice of a lend-layer resident through the synchronous
// set_tensor, LLAMA_FN_L22_REFILL=0) hit it. Such a copy goes again through a pinned bounce buffer (CPU copy, then DMA,
// 32 MiB pieces, synchronous on the stream); any other error stays fatal. true: the bounce ran (the stream is idle)
// [TAG_FN_L23_PINFIX] R-L23-8: one pinned bounce buffer for the whole process, allocated before the first pin (the first
// registration asks for it), so a copy never needs new pinned memory when the page-lock budget is short
static std::mutex g_fn_bounce_mtx;
static void *     g_fn_bounce = nullptr;
static bool       g_fn_bounce_tried = false;
static constexpr size_t FN_BOUNCE_B = (size_t) 32 << 20;

static void * ggml_cuda_fn_bounce_locked() {
    if (g_fn_bounce == nullptr && !g_fn_bounce_tried) {
        g_fn_bounce_tried = true;
        if (cudaMallocHost(&g_fn_bounce, FN_BOUNCE_B) != cudaSuccess) {
            (void) cudaGetLastError();
            g_fn_bounce = nullptr;
            GGML_LOG_WARN("ggml_cuda: [TAG_FN_L23_PINFIX] no pinned bounce buffer (%.0f MiB)\n", FN_BOUNCE_B/1048576.0);
        }
    }
    return g_fn_bounce;
}

// a host copy through the bounce buffer, synchronous on the stream (CPU copy + DMA in 32 MiB pieces)
static cudaError_t ggml_cuda_fn_copy_bounced(void * dst, const void * src, size_t size, cudaMemcpyKind kind, cudaStream_t stream) {
    std::lock_guard<std::mutex> lk(g_fn_bounce_mtx);
    if (g_fn_bounce == nullptr) {
        g_fn_bounce_tried = false; // one more try: pins may have gone
    }
    void * bb = ggml_cuda_fn_bounce_locked();
    if (bb == nullptr) {
        return cudaErrorMemoryAllocation;
    }
    cudaError_t e = cudaStreamSynchronize(stream);
    for (size_t o = 0; e == cudaSuccess && o < size; o += FN_BOUNCE_B) {
        const size_t k = std::min(FN_BOUNCE_B, size - o);
        if (kind == cudaMemcpyHostToDevice) {
            memcpy(bb, (const char *) src + o, k);
            e = cudaMemcpyAsync((char *) dst + o, bb, k, cudaMemcpyHostToDevice, stream);
        } else {
            e = cudaMemcpyAsync(bb, (const char *) src + o, k, cudaMemcpyDeviceToHost, stream);
        }
        if (e == cudaSuccess) {
            e = cudaStreamSynchronize(stream);
        }
        if (e == cudaSuccess && kind == cudaMemcpyDeviceToHost) {
            memcpy((char *) dst + o, bb, k);
        }
    }
    return e;
}

// [TAG_FN_L23_PINFIX] R-L23-7 / R-L23-8: a copy with pageable host memory needs the driver's pinned staging, which fails with
// out of memory when the page-lock budget is short (FAULTs 2026-10-21 15:20 and 15:58: a D2H get_tensor, the abort, the TDR).
// It never aborts for that: the bounce buffer first, then the pins drop (hook code 2), then direct / bounce again while the
// driver frees the pages (up to ~2 s)
static cudaError_t ggml_cuda_memcpy_oom_retry(void * dst, const void * src, size_t size, cudaMemcpyKind kind, cudaStream_t stream) {
    cudaError_t e = cudaMemcpyAsync(dst, src, size, kind, stream);
    if (e != cudaErrorMemoryAllocation || size == 0) {
        return e;
    }
    (void) cudaGetLastError();
    static std::atomic<int> n_notes{0};
    const int note = n_notes.fetch_add(1) + 1;
    e = ggml_cuda_fn_copy_bounced(dst, src, size, kind, stream);
    const char * how = "bounce";
    if (e != cudaSuccess) {
        (void) cudaGetLastError();
        const bool freed = ggml_cuda_fn_l22_oom_retry(size, 2);
        how = freed ? "pins dropped" : "retry";
        for (int k = 0; k < 100; ++k) {
            e = cudaMemcpyAsync(dst, src, size, kind, stream);
            if (e == cudaSuccess) {
                break;
            }
            (void) cudaGetLastError();
            if (e != cudaErrorMemoryAllocation) {
                return e;
            }
            e = ggml_cuda_fn_copy_bounced(dst, src, size, kind, stream);
            if (e == cudaSuccess) {
                break;
            }
            (void) cudaGetLastError();
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }
    if (note <= 3 || note % 256 == 0) {
        GGML_LOG_WARN("ggml_cuda: [TAG_FN_L23_PINFIX] a %s copy of %.2f MiB ran out of pinned staging: %s (%s, %d so far)\n",
                kind == cudaMemcpyHostToDevice ? "H2D" : "D2H", size/1048576.0, e == cudaSuccess ? "done" : "FAILED", how, note);
    }
    return e;
}

static bool ggml_cuda_h2d_or_bounce(void * dst, const void * src, size_t size, cudaStream_t stream) {
    const cudaError_t e = ggml_cuda_memcpy_oom_retry(dst, src, size, cudaMemcpyHostToDevice, stream);
    if (e != cudaErrorInvalidValue || size == 0) {
        CUDA_CHECK(e);
        return false;
    }
    (void) cudaGetLastError();
    static std::atomic<int> n_notes{0};
    CUDA_CHECK(ggml_cuda_fn_copy_bounced(dst, src, size, cudaMemcpyHostToDevice, stream));
    const int k = n_notes.fetch_add(1) + 1;
    if (k <= 3 || k % 1024 == 0) {
        GGML_LOG_WARN("ggml_cuda: [TAG_FN_L23_PINFIX] a host copy of %.2f MiB at %p crossed a registered host range's edge: sent "
                "through the pinned bounce buffer (%d so far)\n", size/1048576.0, src, k);
    }
    return true;
}

static void ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    ggml_cuda_h2d_or_bounce((char *) tensor->data + offset, data, size, cudaStreamPerThread); // [TAG_FN_L23_PINFIX]
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(ggml_cuda_memcpy_oom_retry(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cudaStreamPerThread)); // [TAG_FN_L23_PINFIX] R-L23-7
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpy2DAsync(
        (char *) tensor->data + offset, stride_tensor, data, stride_data, size, n_copies, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static bool ggml_backend_cuda_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_cuda(src->buffer)) {
        ggml_backend_cuda_buffer_context * src_ctx = (ggml_backend_cuda_buffer_context *)src->buffer->context;
        ggml_backend_cuda_buffer_context * dst_ctx = (ggml_backend_cuda_buffer_context *)dst->buffer->context;
        // compare the backing physical devices: distinct virtual devices may share one physical GPU,
        // in which case a same-device copy (not a peer copy) is required
        const int src_physical = ggml_cuda_get_physical_device(src_ctx->device);
        const int dst_physical = ggml_cuda_get_physical_device(dst_ctx->device);
        if (src_physical == dst_physical) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(src), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, ggml_nbytes(src), cudaStreamPerThread));
#endif
        }
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cuda_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    if (ctx->vmm_size > 0) {
        // [TAG_FN_L3_VRAM_CBUF] only the mapped ranges
        for (const auto & s : ctx->vmm_segs) {
            CUDA_CHECK(cudaMemsetAsync((char *) ctx->dev_ptr + s.first, value, s.second, cudaStreamPerThread));
        }
    } else {
        CUDA_CHECK(cudaMemsetAsync(ctx->dev_ptr, value, buffer->size, cudaStreamPerThread));
    }
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static const ggml_backend_buffer_i ggml_backend_cuda_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_cuda_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_cuda_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cuda_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_cuda_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_cuda_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_cuda_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cuda_buffer_clear,
    /* .reset           = */ NULL,
};

// cuda buffer type
struct ggml_backend_cuda_buffer_type_context {
    int device;
    std::string name;
};

static const char * ggml_backend_cuda_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_buffer_type_context * ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static bool ggml_backend_buft_is_cuda(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_buffer_type_get_name;
}

static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    ggml_cuda_set_device(buft_ctx->device);

    void * dev_ptr;
    // [TAG_FN_L22_OOMHOOK] R-REV-7: a device buffer after ready with pins active (a compute buffer that grows inside a
    // decode, a KV or bank allocation): DXGI LOCAL headroom first, one unpin and retry on failure (no hook: unchanged)
    ggml_cuda_fn_l22_oom_precheck(size);
    cudaError_t err = ggml_cuda_device_malloc(&dev_ptr, size, buft_ctx->device);
    if (err == cudaErrorMemoryAllocation && ggml_cuda_fn_l22_oom_retry(size)) {
        (void)cudaGetLastError();
        err = ggml_cuda_device_malloc(&dev_ptr, size, buft_ctx->device);
    }
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: cudaMalloc failed: %s\n", __func__, size / 1024.0 / 1024.0, buft_ctx->device, cudaGetErrorString(err));
        return nullptr;
    }

    ggml_backend_cuda_buffer_context * ctx = new ggml_backend_cuda_buffer_context(buft_ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, size);
}

static size_t ggml_backend_cuda_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_cuda_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *) buft->context;

    size_t size = tensor->op == GGML_OP_FLASH_ATTN_EXT
        ? ggml_cuda_flash_attn_ext_get_alloc_size(buft_ctx->device, tensor)
        : ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    // [TAG_ALLOC_SIZE_EXPAND]
    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            GGML_ASSERT(tensor->nb[0] == ggml_element_size(tensor));
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;
}

static const ggml_backend_buffer_type_i ggml_backend_cuda_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_cuda_buffer_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_cuda_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ NULL,
    /* .get_alignment       = */ ggml_backend_cuda_buffer_type_get_alignment,
    /* .get_max_size        = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size      = */ ggml_backend_cuda_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ NULL,
    /* .is_host             = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_cuda_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type ggml_backend_cuda_buffer_types[GGML_CUDA_MAX_DEVICES];

    static bool ggml_backend_cuda_buffer_type_initialized = false;

    if (!ggml_backend_cuda_buffer_type_initialized) {
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); i++) {
            ggml_backend_cuda_buffer_types[i] = {
                /* .iface    = */ ggml_backend_cuda_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), i),
                /* .context  = */ new ggml_backend_cuda_buffer_type_context{i, GGML_CUDA_NAME + std::to_string(i)},
            };
        }
        ggml_backend_cuda_buffer_type_initialized = true;
    }

    return &ggml_backend_cuda_buffer_types[device];
}

// [TAG_FN_L3_VRAM_CBUF] device buffers on reserved virtual memory --------------------------------------------------------
// _alloc reserves the addresses of a buffer of buft's device and maps nothing; _map backs [offset, offset + size) with new
// physical memory (its content is undefined); _unmap gives mapped segments back to the driver at once. Offsets and sizes
// are multiples of the granularity, and an unmap covers whole segments of earlier maps. The buffer has the CUDA buffer
// interface: a tensor in it works like in any CUDA buffer while its bytes are mapped. The caller makes sure that nothing
// reads or writes a range while it is not mapped (synchronize before an unmap).

size_t ggml_backend_cuda_vmm_granularity(ggml_backend_buffer_type_t buft) {
#ifdef GGML_CUDA_VMM_BUFFERS
    if (!buft || !ggml_backend_buft_is_cuda(buft)) {
        return 0;
    }
    const int device = ((ggml_backend_cuda_buffer_type_context *) buft->context)->device;
    const auto & info = ggml_cuda_info().devices[device];
    return info.vmm ? info.vmm_granularity : 0;
#else
    GGML_UNUSED(buft);
    return 0;
#endif
}

ggml_backend_buffer_t ggml_backend_cuda_vmm_buffer_alloc(ggml_backend_buffer_type_t buft, size_t size) {
#ifdef GGML_CUDA_VMM_BUFFERS
    const size_t gran = ggml_backend_cuda_vmm_granularity(buft);
    if (gran == 0 || size == 0) {
        return nullptr;
    }
    const int device = ((ggml_backend_cuda_buffer_type_context *) buft->context)->device;
    size = (size + gran - 1) / gran * gran;
    ggml_cuda_set_device(device);
    CUdeviceptr ptr = 0;
    const CUresult r = cuMemAddressReserve(&ptr, size, 0, 0, 0);
    if (r != CUDA_SUCCESS) {
        const char * s = nullptr;
        cuGetErrorString(r, &s);
        GGML_LOG_WARN("%s: reserving %.1f MiB of addresses on device %d failed: %s\n", __func__, size/1048576.0, device, s ? s : "?");
        return nullptr;
    }
    auto * ctx = new ggml_backend_cuda_buffer_context(device, (void *) ptr);
    ctx->vmm_size = size;
    ctx->vmm_gran = gran;
    return ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, size);
#else
    GGML_UNUSED(buft);
    GGML_UNUSED(size);
    return nullptr;
#endif
}

bool ggml_backend_cuda_vmm_buffer_map(ggml_backend_buffer_t buffer, size_t offset, size_t size) {
#ifdef GGML_CUDA_VMM_BUFFERS
    if (!buffer || !ggml_backend_buffer_is_cuda(buffer)) {
        return false;
    }
    auto * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;
    const size_t gran = ctx->vmm_gran;
    if (ctx->vmm_size == 0 || size == 0 || offset % gran != 0 || size % gran != 0 || offset + size > ctx->vmm_size) {
        return false;
    }
    for (const auto & s : ctx->vmm_segs) {
        if (s.first < offset + size && offset < s.first + s.second) {
            return false; // a part is mapped already
        }
    }
    ggml_cuda_set_device(ctx->device);
    const int phys = ggml_cuda_get_physical_device(ctx->device);
    CUmemAllocationProp prop = {};
    prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id   = phys;
    CUmemAccessDesc acc = {};
    acc.location.type  = CU_MEM_LOCATION_TYPE_DEVICE;
    acc.location.id    = phys;
    acc.flags          = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    // physical allocations of at most 1 GiB: a failed one leaves less to undo
    const size_t seg_max = std::max(gran, ((size_t) 1 << 30) / gran * gran);
    std::vector<std::pair<size_t, size_t>> done;
    CUresult    r    = CUDA_SUCCESS;
    const char * what = nullptr;
    for (size_t o = offset; o < offset + size; ) {
        const size_t      n = std::min(seg_max, offset + size - o);
        const CUdeviceptr p = (CUdeviceptr) ctx->dev_ptr + o;
        CUmemGenericAllocationHandle h;
        r = cuMemCreate(&h, n, &prop, 0);
        if (r != CUDA_SUCCESS) {
            what = "cuMemCreate";
            break;
        }
        r = cuMemMap(p, n, 0, h, 0);
        (void) cuMemRelease(h); // the mapping holds the memory from here on (nothing does after a failed map)
        if (r != CUDA_SUCCESS) {
            what = "cuMemMap";
            break;
        }
        r = cuMemSetAccess(p, n, &acc, 1);
        if (r != CUDA_SUCCESS) {
            (void) cuMemUnmap(p, n);
            what = "cuMemSetAccess";
            break;
        }
        done.push_back({ o, n });
        o += n;
    }
    if (what) {
        const char * s = nullptr;
        cuGetErrorString(r, &s);
        GGML_LOG_WARN("%s: mapping %.1f MiB at +%.1f MiB on device %d failed (%s: %s)\n", __func__, size/1048576.0,
                offset/1048576.0, ctx->device, what, s ? s : "?");
        for (const auto & d : done) {
            (void) cuMemUnmap((CUdeviceptr) ctx->dev_ptr + d.first, d.second);
        }
        return false;
    }
    ctx->vmm_segs.insert(ctx->vmm_segs.end(), done.begin(), done.end());
    std::sort(ctx->vmm_segs.begin(), ctx->vmm_segs.end());
    return true;
#else
    GGML_UNUSED(buffer);
    GGML_UNUSED(offset);
    GGML_UNUSED(size);
    return false;
#endif
}

bool ggml_backend_cuda_vmm_buffer_unmap(ggml_backend_buffer_t buffer, size_t offset, size_t size) {
#ifdef GGML_CUDA_VMM_BUFFERS
    if (!buffer || !ggml_backend_buffer_is_cuda(buffer)) {
        return false;
    }
    auto * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;
    if (ctx->vmm_size == 0 || offset + size > ctx->vmm_size) {
        return false;
    }
    for (const auto & s : ctx->vmm_segs) {
        const bool touch = s.first < offset + size && offset < s.first + s.second;
        if (touch && (s.first < offset || s.first + s.second > offset + size)) {
            return false; // a segment reaches out of the range: it cannot be unmapped in part
        }
    }
    ggml_cuda_set_device(ctx->device);
    bool ok = true;
    for (auto it = ctx->vmm_segs.begin(); it != ctx->vmm_segs.end(); ) {
        if (it->first >= offset && it->first + it->second <= offset + size) {
            const CUresult r = cuMemUnmap((CUdeviceptr) ctx->dev_ptr + it->first, it->second);
            if (r != CUDA_SUCCESS) {
                const char * s = nullptr;
                cuGetErrorString(r, &s);
                GGML_LOG_WARN("%s: unmapping %.1f MiB at +%.1f MiB failed: %s\n", __func__, it->second/1048576.0,
                        it->first/1048576.0, s ? s : "?");
                ok = false;
                ++it;
                continue;
            }
            it = ctx->vmm_segs.erase(it);
        } else {
            ++it;
        }
    }
    return ok;
#else
    GGML_UNUSED(buffer);
    GGML_UNUSED(offset);
    GGML_UNUSED(size);
    return false;
#endif
}

// mapped bytes of a buffer from ggml_backend_cuda_vmm_buffer_alloc (the whole size of any other CUDA buffer)
size_t ggml_backend_cuda_vmm_buffer_mapped(ggml_backend_buffer_t buffer) {
    if (!buffer || !ggml_backend_buffer_is_cuda(buffer)) {
        return 0;
    }
    const auto * ctx = (const ggml_backend_cuda_buffer_context *) buffer->context;
    if (ctx->vmm_size == 0) {
        return ggml_backend_buffer_get_size(buffer);
    }
    size_t n = 0;
    for (const auto & s : ctx->vmm_segs) {
        n += s.second;
    }
    return n;
}

// Communication context for multi-GPU AllReduce during tensor parallelism.
//
// Created once per meta backend instance.  Resources for the selected mode
// (NCCL communicators or the internal AllReduce pipeline) are initialised
// eagerly during comm_init so any init failure surfaces at startup rather
// than mid-run.
struct ggml_backend_cuda_comm_context {
    using try_allreduce_fn = bool(*)(ggml_backend_cuda_comm_context *, struct ggml_tensor **);

    std::vector<ggml_backend_t> backends;
    std::vector<int>            dev_ids;

    // Set by the init chain (comm_init_{nccl, internal, none}) to one of
    // try_allreduce_{nccl, internal, butterfly}.  nccl needs `comms`,
    // internal needs `ar_pipeline`, butterfly needs nothing.  Per-call
    // failures return false; the meta backend's generic implementation then
    // handles that call.
    try_allreduce_fn            try_allreduce = nullptr;

    ggml_cuda_ar_pipeline *     ar_pipeline = nullptr;

#ifdef GGML_USE_NCCL
    std::vector<ncclComm_t>     comms;
#endif // GGML_USE_NCCL

    ~ggml_backend_cuda_comm_context() {
#ifdef GGML_USE_NCCL
        for (ncclComm_t comm : comms) {
            NCCL_CHECK(ncclCommDestroy(comm));
        }
#endif // GGML_USE_NCCL
        ggml_cuda_ar_pipeline_free(ar_pipeline);
    }
};

#ifdef GGML_USE_NCCL
// AllReduce via NCCL. Reduces as FP32 for small tensors and BF16 for large
// tensors (bandwidth-bound), then converts back to FP32.
static bool ggml_backend_cuda_comm_allreduce_nccl(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    const int64_t ne = ggml_nelements(tensors[0]);
    // FIXME the input of llm_graph_context::build_in_out_ids can produce a tensor with 0 elements if n_outputs == 0
    // This then causes a crash in this function
    if (ne == 0) {
        return true;
    }

    const size_t n_backends = comm_ctx->backends.size();

    for (size_t i = 0; i < n_backends; ++i) {
        GGML_ASSERT(tensors[i] != nullptr);
        GGML_ASSERT(ggml_nelements(tensors[i]) == ne);
        GGML_ASSERT(ggml_is_contiguously_allocated(tensors[i]));
    }

    // For small tensors, simply reduce them as FP32.
    // The following heuristic for how "small" a tensor should be is based on RTX 4090s connected via 16x PCIe 4.0.
    if ((n_backends <= 2 && ne < 32768) || (n_backends == 3 && ne < 131072) || (n_backends >= 4 && ne < 262144)) {
        for (size_t i = 0; i < n_backends; ++i) {
            if ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
                ggml_cuda_set_device(cuda_ctx->device);
                CUDA_CHECK(cudaMemsetAsync(tensors[i]->data, 0, ggml_nbytes(tensors[i]), cuda_ctx->stream()));
            }
        }
        NCCL_CHECK(ncclGroupStart());
        for (size_t i = 0; i < n_backends; ++i) {
            ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
            NCCL_CHECK(ncclAllReduce(tensors[i]->data, tensors[i]->data, ne, ncclFloat, ncclSum, comm_ctx->comms[i], cuda_ctx->stream()));
        }
        NCCL_CHECK(ncclGroupEnd());
        return true;
    }

    // For large tensors it's faster to compress them to BF16 for the reduction:
    to_bf16_cuda_t to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);
    to_fp32_cuda_t to_fp32 = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);

    ggml_cuda_pool_alloc<nv_bfloat16> tmp[GGML_CUDA_MAX_DEVICES];
    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        tmp[i].pool = &cuda_ctx->pool();
        tmp[i].alloc(ne);

        ggml_cuda_set_device(cuda_ctx->device);
        if (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) {
            to_bf16(tensors[i]->data, tmp[i].get(), ne, cuda_ctx->stream());
        } else {
            CUDA_CHECK(cudaMemsetAsync(tmp[i].get(), 0, ne * sizeof(nv_bfloat16), cuda_ctx->stream()));
        }
        CUDA_CHECK(cudaGetLastError());
    }

    NCCL_CHECK(ncclGroupStart());
    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        NCCL_CHECK(ncclAllReduce(tmp[i].get(), tmp[i].get(), ne, ncclBfloat16, ncclSum, comm_ctx->comms[i], cuda_ctx->stream()));
    }
    NCCL_CHECK(ncclGroupEnd());

    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;

        ggml_cuda_set_device(cuda_ctx->device);
        to_fp32(tmp[i].get(), (float *) tensors[i]->data, ne, cuda_ctx->stream());
        CUDA_CHECK(cudaGetLastError());
    }

    return true;
}
#endif // GGML_USE_NCCL

// Run the internal AR pipeline.  Returns false on unsupported / failed input
// -- the caller decides whether to abort (env-forced) or fall back silently.
static bool ggml_backend_cuda_comm_allreduce_internal(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    GGML_ASSERT(comm_ctx->ar_pipeline != nullptr);

    const size_t n_backends = comm_ctx->backends.size();
    GGML_ASSERT(n_backends == 2);
    GGML_ASSERT(tensors[0] != nullptr);

    const int64_t   ne   = ggml_nelements(tensors[0]);
    const ggml_type type = tensors[0]->type;

    if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_BF16) {
        GGML_LOG_DEBUG("%s: internal unsupported: type=%d\n", __func__, (int) type);
        return false;
    }

    if (ne == 0) {
        return true;
    }

    for (size_t i = 0; i < n_backends; ++i) {
        if (tensors[i] == nullptr) {
            GGML_LOG_ERROR("%s: internal failed: tensor[%zu] is null\n", __func__, i);
            return false;
        }
        if (ggml_nelements(tensors[i]) != ne || tensors[i]->type != type) {
            GGML_LOG_ERROR("%s: internal failed: tensor[%zu] ne=%" PRId64 " type=%d expected ne=%" PRId64 " type=%d\n",
                           __func__, i, ggml_nelements(tensors[i]), (int) tensors[i]->type, ne, (int) type);
            return false;
        }
        if (!ggml_is_contiguously_allocated(tensors[i])) {
            GGML_LOG_DEBUG("%s: internal unsupported: tensor[%zu] is not contiguously allocated: ne=%" PRId64 " nbytes=%zu packed=%zu type=%d\n",
                           __func__, i, ne, ggml_nbytes(tensors[i]),
                           (size_t) ne * ggml_type_size(type) / ggml_blck_size(type), (int) type);
            return false;
        }
        if (((uintptr_t) tensors[i]->data & 0xF) != 0) {
            GGML_LOG_DEBUG("%s: internal unsupported: tensor[%zu] data pointer is not 16-byte aligned: %p type=%d ne=%" PRId64 "\n",
                           __func__, i, tensors[i]->data, (int) type, ne);
            return false;
        }
        GGML_ASSERT((ggml_nbytes(tensors[i]) & 0xF) == 0);
    }

    return ggml_cuda_ar_allreduce(comm_ctx->ar_pipeline, comm_ctx->backends.data(), tensors);
}

// ---------------------------------------------------------------------------
// Per-call dispatch -- three variants, one per backend.  Each is set as
// comm_ctx->try_allreduce by the matching init step.  Per-call failure
// returns false; the meta backend's generic implementation handles that call.
// ---------------------------------------------------------------------------

#ifdef GGML_USE_NCCL
static bool ggml_backend_cuda_comm_try_allreduce_nccl(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    return ggml_backend_cuda_comm_allreduce_nccl(comm_ctx, tensors);
}
#endif // GGML_USE_NCCL

static bool ggml_backend_cuda_comm_try_allreduce_internal(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    return ggml_backend_cuda_comm_allreduce_internal(comm_ctx, tensors);
}

static bool ggml_backend_cuda_comm_try_allreduce_butterfly(
        ggml_backend_cuda_comm_context *, struct ggml_tensor **) {
    return false;
}

static void ggml_backend_cuda_comm_free(void * comm_ctx_v) {
    if (comm_ctx_v == nullptr) {
        return;
    }
    delete static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
}

// ---------------------------------------------------------------------------
// Init -- chained nccl -> internal -> none.  Each step tries to bring up its
// resource; on failure it warns and recurses into the next step.
// ---------------------------------------------------------------------------
static void ggml_backend_cuda_comm_init_none(ggml_backend_cuda_comm_context * ret) {
    ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_butterfly;
}

static void ggml_backend_cuda_comm_init_internal(ggml_backend_cuda_comm_context * ret) {
    ret->ar_pipeline = ggml_cuda_ar_pipeline_init(ret->dev_ids.data(), ret->dev_ids.size());
    if (ret->ar_pipeline) {
        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_internal;
        return;
    }

    // Clear sticky CUDA error from the failed init.
    (void) cudaGetLastError();
    GGML_LOG_WARN("internal AllReduce init failed (n_devices != 2?); "
                  "falling back to meta-backend butterfly\n");
    ggml_backend_cuda_comm_init_none(ret);
}

static void ggml_backend_cuda_comm_init_nccl(ggml_backend_cuda_comm_context * ret) {
#ifdef GGML_USE_NCCL
    // Disabling NCCL path when CUDA virtual devices are in use since NCCL requires one distinct physical GPU per rank.
    const ggml_cuda_device_info & info = ggml_cuda_info();
    if (info.device_count > info.physical_device_count) {
        GGML_LOG_WARN("NCCL disabled: virtual devices in use; "
                      "falling back to internal AllReduce\n");
        ggml_backend_cuda_comm_init_internal(ret);
        return;
    }

    const size_t n = ret->dev_ids.size();
    ret->comms.resize(n);
    ncclResult_t rc = ncclCommInitAll(ret->comms.data(), (int) n, ret->dev_ids.data());
    if (rc == ncclSuccess) {
        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_nccl;
        return;
    }

    ret->comms.clear();
    GGML_LOG_WARN("NCCL init failed (%s); falling back to internal AllReduce\n",
                  ncclGetErrorString(rc));
#else // GGML_USE_NCCL
#ifndef GGML_USE_HIP
    GGML_LOG_WARN("NCCL not compiled in; falling back to internal AllReduce.  "
                  "Recompile with -DGGML_CUDA_NCCL=ON for best multi-GPU performance.\n");
#endif // !GGML_USE_HIP
#endif // GGML_USE_NCCL

    ggml_backend_cuda_comm_init_internal(ret);
}

// Top-level init.  Picks one of the three init paths based on
// GGML_CUDA_ALLREDUCE (or the platform default) and lets the chain handle
// any fallback.  Unrecognised env values warn and fall through to the
// platform default.
static void * ggml_backend_cuda_comm_init(ggml_backend_t * backends, size_t n_backends) {
    for (size_t i = 0; i < n_backends; i++) {
        if (!ggml_backend_is_cuda(backends[i])) {
            return nullptr;
        }
    }

    auto * ret = new ggml_backend_cuda_comm_context;
    ret->backends.assign(backends, backends + n_backends);
    ret->dev_ids.reserve(n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        ret->dev_ids.push_back(static_cast<ggml_backend_cuda_context *>(backends[i]->context)->device);
    }

    const char * env = getenv("GGML_CUDA_ALLREDUCE");
    if (!env) {
        // Platform default: Linux uses NCCL, otherwise (generally Windows) internal
#if defined(__linux__)
        ggml_backend_cuda_comm_init_nccl(ret);
#else
        ggml_backend_cuda_comm_init_internal(ret);
#endif // defined(__linux__)
    } else {
        std::string env_str(env);
        if (env_str == "nccl") {
            ggml_backend_cuda_comm_init_nccl(ret);
        } else if (env_str == "internal") {
            ggml_backend_cuda_comm_init_internal(ret);
        } else if (env_str == "none") {
            ggml_backend_cuda_comm_init_none(ret);
        } else {
            GGML_LOG_WARN("unknown GGML_CUDA_ALLREDUCE value: %s\n", env);
            ggml_backend_cuda_comm_init_none(ret);
        }
    }

    return ret;
}

// Top-level dispatch -- calls the function pointer chosen by comm_init.
// Returns false to let the meta-backend's butterfly run.
static bool ggml_backend_cuda_comm_allreduce_tensor(void * comm_ctx_v, struct ggml_tensor ** tensors) {
    if (comm_ctx_v == nullptr) {
        return false;
    }
    auto * comm_ctx = static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
    return comm_ctx->try_allreduce(comm_ctx, tensors);
}

// host buffer type

static const char * ggml_backend_cuda_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_CUDA_NAME "_Host";

    GGML_UNUSED(buft);
}

static bool ggml_backend_buft_is_cuda_host(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
}

static void ggml_backend_cuda_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    CUDA_CHECK(cudaFreeHost(buffer->context));
}

static void * ggml_cuda_host_malloc(size_t size) {
    if (ggml_cuda_pinned_memory_disabled()) {
        return nullptr;
    }

    void * ptr = nullptr;
    cudaError_t err = cudaMallocHost((void **) &ptr, size);
    // [TAG_FN_L23_PINFIX] R-L23-5: pinned host memory short while libllama holds pins: they go first, one retry
    if (err == cudaErrorMemoryAllocation && ggml_cuda_fn_l22_oom_retry(size, 2)) {
        (void) cudaGetLastError();
        err = cudaMallocHost((void **) &ptr, size);
    }
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_DEBUG("%s: failed to allocate %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return nullptr;
    }

    return ptr;
}

static ggml_backend_buffer_t ggml_backend_cuda_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = ggml_cuda_host_malloc(size);

    if (ptr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_cuda_host_buffer_free_buffer;

    return buffer;
}

static size_t ggml_backend_cuda_host_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    // very large pinned allocations can fail (e.g. on Windows) and fall back to pageable memory, smaller ones succeed
    return 4ull*1024*1024*1024;
}

ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_cuda_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name            = */ ggml_backend_cuda_host_buffer_type_name,
            /* .alloc_buffer        = */ ggml_backend_cuda_host_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ NULL,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size        = */ ggml_backend_cuda_host_buffer_type_get_max_size,
            /* .get_alloc_size      = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .get_alloc_size_n    = */ NULL,
            /* .is_host             = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), 0),
        /* .context  = */ nullptr,
    };

    return &ggml_backend_cuda_buffer_type_host;
}

//static bool ggml_backend_buffer_is_cuda_host(ggml_backend_buffer_t buffer) {
//    return buffer->buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
//}

/// kernels

typedef void (*ggml_cuda_op_mul_mat_t)(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

static __global__ void k_compute_batched_ptrs(
        const void * src0_as_f16, const void * src1_as_f16, char * dst,
        const void ** ptrs_src, void ** ptrs_dst,
        int64_t ne12, int64_t ne13,
        int64_t ne23,
        size_t  nb02, size_t  nb03,
        size_t  nb12, size_t  nb13,
        size_t  nbd2, size_t  nbd3,
        int64_t r2,   int64_t r3) {
    const int64_t i13 = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t i12 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }

    const int64_t i03 = i13 / r3;
    const int64_t i02 = i12 / r2;

    ptrs_src[0*ne23 + i12 + i13*ne12] = (const char *) src0_as_f16 + i02*nb02 + i03*nb03;
    ptrs_src[1*ne23 + i12 + i13*ne12] = (const char *) src1_as_f16 + i12*nb12 + i13*nb13;
    ptrs_dst[0*ne23 + i12 + i13*ne12] = (      char *)         dst + i12*nbd2 + i13*nbd3;
}

// Type traits for mapping ggml types to CUDA/cuBLAS types
template<ggml_type T>
struct batched_mul_mat_traits;

template<>
struct batched_mul_mat_traits<GGML_TYPE_F32> {
    using cuda_type = float;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_32F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F32;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_fp32_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_fp32_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_BF16> {
    using cuda_type = nv_bfloat16;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_16BF;
    static inline const ggml_type ggml_type_val = GGML_TYPE_BF16;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_bf16_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_bf16_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_F16> {
    using cuda_type = half;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
    static inline const cudaDataType_t data_type = CUDA_R_16F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F16;
    static inline const half alpha = 1.0;
    static inline const half beta = 0.0;
    static inline const void* get_alpha() { static const half val = alpha; return &val; }
    static inline const void* get_beta() { static const half val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_fp16_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_fp16_nc_cuda(src_type); }
};

template<ggml_type compute_type>
static void ggml_cuda_mul_mat_cublas_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    using traits = batched_mul_mat_traits<compute_type>;
    using cuda_t = typename traits::cuda_type;

    // F32 chunks can have padding between rows to preserve the original destination stride.
    GGML_ASSERT(ggml_is_contiguous(dst) ||
        (compute_type == GGML_TYPE_F32 && ggml_is_contiguous_1(dst)));

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t ldc = nb1 / sizeof(float);
    const int64_t ne_dst = ggml_nelements(dst);
    cudaStream_t main_stream = ctx.stream();
    cublasHandle_t cublas_h = ctx.cublas_handle();

    const size_t src0_ts = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == src0_ts);
    int64_t s01 = nb01 / src0_ts;
    int64_t s02 = nb02 / src0_ts;
    int64_t s03 = nb03 / src0_ts;

    const size_t src1_ts = ggml_type_size(src1->type);
    GGML_ASSERT(nb10 == src1_ts);
    int64_t s11 = nb11 / src1_ts;
    int64_t s12 = nb12 / src1_ts;
    int64_t s13 = nb13 / src1_ts;

    float * dst_ddf = (float *) dst->data;

    const cuda_t * src0_ptr = nullptr;
    const cuda_t * src1_ptr = nullptr;

    ggml_cuda_pool_alloc<cuda_t> src0_alloc(ctx.pool());
    ggml_cuda_pool_alloc<cuda_t> src1_alloc(ctx.pool());

    bool is_src0_cont_2 = ggml_is_contiguous_2(src0);
    bool is_src1_cont_2 = ggml_is_contiguous_2(src1);

    if (src0->type == compute_type) {
        src0_ptr = (const cuda_t *) src0->data;
    } else {
        src0_alloc.alloc(ggml_nelements(src0));

        if (ggml_is_contiguously_allocated(src0)) {
            const auto convert_func = traits::convert(src0->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src0->data, src0_alloc.get(), ggml_nelements(src0), main_stream);
            const size_t src0_bs = ggml_blck_size(src0->type);
            s01 *= src0_bs;
            s02 *= src0_bs;
            s03 *= src0_bs;
        } else {
            const auto convert_func = traits::convert_nc(src0->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src0->data, src0_alloc.get(), ne00, ne01, ne02, ne03, s01, s02, s03, main_stream);
            s01 = ne00;
            s02 = ne01*s01;
            s03 = ne02*s02;
            is_src0_cont_2 = true;
        }
        src0_ptr = src0_alloc.get();
    }

    if (src1->type == compute_type) {
        src1_ptr = (const cuda_t *) src1->data;
    } else {
        src1_alloc.alloc(ggml_nelements(src1));

        if (ggml_is_contiguously_allocated(src1)) {
            const auto convert_func = traits::convert(src1->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src1->data, src1_alloc.get(), ggml_nelements(src1), main_stream);
            const size_t src1_bs = ggml_blck_size(src1->type);
            s11 *= src1_bs;
            s12 *= src1_bs;
            s13 *= src1_bs;
        } else {
            const auto convert_func = traits::convert_nc(src1->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src1->data, src1_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, main_stream);
            s11 = ne10;
            s12 = ne11*s11;
            s13 = ne12*s12;
            is_src1_cont_2 = true;
        }
        src1_ptr = src1_alloc.get();
    }

    ggml_cuda_pool_alloc<cuda_t> dst_temp(ctx.pool());
    char * dst_ptr;
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    cublasComputeType_t cu_compute_type = traits::compute_type;
    cudaDataType_t cu_data_type = traits::data_type;
    cudaDataType_t cu_data_type_a = traits::data_type;
    cudaDataType_t cu_data_type_b = traits::data_type;
    const void * alpha = traits::get_alpha();
    const void * beta = traits::get_beta();

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    bool prefer_f32_output = false;
    if (compute_type == GGML_TYPE_F16) {
        prefer_f32_output = cc == GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_CDNA(cc);
    } else if (compute_type == GGML_TYPE_BF16) {
        prefer_f32_output = !GGML_CUDA_CC_IS_RDNA3(cc) && !GGML_CUDA_CC_IS_CDNA(cc);
    }

    if (prefer_f32_output) {
        dst_ptr = (char *) dst_ddf;
        cu_compute_type = batched_mul_mat_traits<GGML_TYPE_F32>::compute_type;
        cu_data_type = batched_mul_mat_traits<GGML_TYPE_F32>::data_type;
        alpha = batched_mul_mat_traits<GGML_TYPE_F32>::get_alpha();
        beta = batched_mul_mat_traits<GGML_TYPE_F32>::get_beta();
    } else {
        if constexpr (compute_type == GGML_TYPE_F32) {
            dst_ptr = (char *) dst_ddf;  // Direct F32 output
        } else {
            dst_ptr = (char *) dst_temp.alloc(ne_dst);
            nbd2 /= sizeof(float) / sizeof(cuda_t);
            nbd3 /= sizeof(float) / sizeof(cuda_t);
        }
    }

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

    // Theoretically cublasGemmStridedBatchedEx would always work, even for a single matrix.
    // However, for some old NVIDIA and AMD GPUs the strided/Ex GEMM is much slower,
    //     probably because the internal kernel selection logic is suboptimal.
    if (compute_type == GGML_TYPE_F32 && ne12 == 1 && ne13 == 1) {
        CUBLAS_CHECK(
            cublasSgemm(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                    ne01, ne11, ne10,
                    (const float *) alpha, (const float *) src0_ptr, s01,
                                           (const float *) src1_ptr, s11,
                    (const float *) beta,  (float       *)  dst_ptr, ldc));
    } else if (ne12 == 1 && ne13 == 1) {
        CUBLAS_CHECK(
            cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                    ne01, ne11, ne10,
                    alpha, src0_ptr, cu_data_type_a, s01,
                           src1_ptr, cu_data_type_b, s11,
                    beta,   dst_ptr, cu_data_type,   ldc,
                    cu_compute_type,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else if (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
        // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
        const int64_t sma = ne02 == 1 ? s03 : s02;
        const int64_t smb = ne12 == 1 ? s13 : s12;

        // there is no broadcast and src0, src1 are contiguous across dims 2, 3
        // use cublasGemmStridedBatchedEx
        CUBLAS_CHECK(
        cublasGemmStridedBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, src0_ptr, cu_data_type_a, s01, sma,     // strideA
                       src1_ptr, cu_data_type_b, s11, smb,     // strideB
                beta,   dst_ptr, cu_data_type,   ldc, ne1*ldc, // strideC
                ne12*ne13,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        // use cublasGemmBatchedEx
        const int64_t ne23 = ne12*ne13;

        ggml_cuda_pool_alloc<const void *> ptrs_src(ctx.pool(), 2*ne23);
        ggml_cuda_pool_alloc<      void *> ptrs_dst(ctx.pool(), 1*ne23);

        const size_t src_type_size = sizeof(cuda_t);

        const int threads_x = 16;
        const int threads_y = 16;
        const dim3 block_dims(threads_x, threads_y);

        const dim3 grid_dims(
            (ne13 + threads_x - 1) / threads_x,
            (ne12 + threads_y - 1) / threads_y
        );
        k_compute_batched_ptrs<<<grid_dims, block_dims, 0, main_stream>>>(
                src0_ptr, src1_ptr, dst_ptr,
                ptrs_src.get(), ptrs_dst.get(),
                ne12, ne13,
                ne23,
                s02*src_type_size, s03*src_type_size,
                s12*src_type_size, s13*src_type_size,
                nbd2, nbd3,
                r2, r3);

        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(
        cublasGemmBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, (const void **) (ptrs_src.get() + 0*ne23), cu_data_type_a, s01,
                       (const void **) (ptrs_src.get() + 1*ne23), cu_data_type_b, s11,
                beta,  (      void **) (ptrs_dst.get() + 0*ne23), cu_data_type,   ldc,
                ne23,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // Convert output back to F32 if needed
    if (cu_data_type != CUDA_R_32F) {
        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(traits::ggml_type_val);
        to_fp32_cuda(dst_temp.get(), dst_ddf, ne_dst, main_stream);
    }
}

static void ggml_cuda_mul_mat_cublas(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const ggml_prec prec = (ggml_prec) ggml_get_op_params_i32(dst, 0);
    ggml_type compute_type = src0->type;
    if (ggml_is_quantized(compute_type)) {
        compute_type = fast_fp16_hardware_available(cc) ? GGML_TYPE_F16 : GGML_TYPE_F32;
    } else if (compute_type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc)) {
        compute_type = GGML_TYPE_F32;
    } else if (compute_type == GGML_TYPE_BF16 && !fast_bf16_hardware_available(cc)) {
        if (GGML_CUDA_CC_IS_AMD(cc) && src1->ne[1] > 32) {
            compute_type = GGML_TYPE_F32;
        }
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && src1->ne[1] > (cc >= GGML_CUDA_CC_VOLTA ? 8 : 128)) {
            compute_type = GGML_TYPE_F32;
        }
    }
    // F16 is the only compute type that can not satisfy a request for BF16
    if (prec == GGML_PREC_BF16 && compute_type == GGML_TYPE_F16) {
        compute_type = fast_bf16_hardware_available(cc) ? GGML_TYPE_BF16 : GGML_TYPE_F32;
    } else if (prec == GGML_PREC_F32) {
        compute_type = GGML_TYPE_F32;
    }

    const char * env_c = getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE");
    if (env_c != nullptr) {
        std::string env_cpp = env_c;
        for (char & c : env_cpp) {
            c = std::tolower(c);
        }
        if (env_cpp == "f32" || env_cpp == "fp32") {
            compute_type = GGML_TYPE_F32;
        } else if (env_cpp == "f16" || env_cpp == "fp16") {
            compute_type = GGML_TYPE_F16;
        } else if (env_cpp == "bf16") {
            compute_type = GGML_TYPE_BF16;
        } else if (env_cpp != "auto") {
            GGML_LOG_WARN("%s: unknown value for GGML_CUDA_CUBLAS_COMPUTE_TYPE: %s", __func__, env_cpp.c_str());
        }
    }

    constexpr size_t max_src0_convert_size = 512ull * 1024 * 1024;
    const size_t src0_f32_size = ggml_nelements(src0) * sizeof(float);

    if (compute_type == GGML_TYPE_F32 &&
            (src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16) &&
            src0_f32_size > max_src0_convert_size) {
        const size_t f32_row_size = src0_f32_size / src0->ne[1];
        const int64_t rows_per_chunk = std::max<int64_t>(1, (int64_t) (max_src0_convert_size / f32_row_size));

        if (rows_per_chunk < src0->ne[1]) {
            ggml_tensor src0_chunk = *src0;
            ggml_tensor dst_chunk = *dst;

            for (int64_t i01 = 0; i01 < src0->ne[1]; i01 += rows_per_chunk) {
                src0_chunk.ne[1] = std::min(rows_per_chunk, src0->ne[1] - i01);
                src0_chunk.data = (char *) src0->data + i01*src0->nb[1];

                dst_chunk.ne[0] = src0_chunk.ne[1];
                dst_chunk.data = (char *) dst->data + i01*dst->nb[0];

                ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F32>(ctx, &src0_chunk, src1, &dst_chunk);
            }
            return;
        }
    }

    switch (compute_type) {
        case GGML_TYPE_F32:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F32>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_BF16:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_BF16>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_F16:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F16>(ctx, src0, src1, dst);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

static bool ggml_cuda_should_fuse_mul_mat(const ggml_tensor * ffn_up,
                                          const ggml_tensor * ffn_gate,
                                          const ggml_tensor * glu,
                                          const ggml_tensor * ffn_up_bias = nullptr,
                                          const ggml_tensor * ffn_gate_bias = nullptr,
                                          const ggml_tensor * ffn_up_scale = nullptr,
                                          const ggml_tensor * ffn_gate_scale = nullptr) {
    const bool has_bias = ffn_up_bias != nullptr || ffn_gate_bias != nullptr;
    const bool has_scale = ffn_up_scale != nullptr || ffn_gate_scale != nullptr;

    if (has_bias && (!ffn_up_bias || !ffn_gate_bias)) {
        return false;
    }
    if (has_scale && (!ffn_up_scale || !ffn_gate_scale)) {
        return false;
    }

    const bool is_mul_mat     = ffn_up->op == GGML_OP_MUL_MAT     && ffn_gate->op == GGML_OP_MUL_MAT     && glu->op == GGML_OP_GLU;
    const bool is_mul_mat_id  = ffn_up->op == GGML_OP_MUL_MAT_ID  && ffn_gate->op == GGML_OP_MUL_MAT_ID  && glu->op == GGML_OP_GLU;

    GGML_ASSERT(ffn_up && ffn_gate && glu);

    if (!is_mul_mat && !is_mul_mat_id) {
        return false;
    }

    const ggml_op expected_bias_op = is_mul_mat ? GGML_OP_ADD : GGML_OP_ADD_ID;
    const ggml_tensor * ffn_up_bias_src   = has_scale ? ffn_up_scale   : ffn_up;
    const ggml_tensor * ffn_gate_bias_src = has_scale ? ffn_gate_scale : ffn_gate;
    const ggml_tensor * ffn_up_out        = has_bias ? ffn_up_bias     : ffn_up_bias_src;
    const ggml_tensor * ffn_gate_out      = has_bias ? ffn_gate_bias   : ffn_gate_bias_src;

    if (glu->src[0] != ffn_gate_out || glu->src[1] != ffn_up_out) {
        return false;
    }

    if (has_scale) {
        if (ffn_up_scale->op != GGML_OP_MUL || ffn_gate_scale->op != GGML_OP_MUL) {
            return false;
        }
        const bool up_has_mm   = ffn_up_scale->src[0] == ffn_up || ffn_up_scale->src[1] == ffn_up;
        const bool gate_has_mm = ffn_gate_scale->src[0] == ffn_gate || ffn_gate_scale->src[1] == ffn_gate;
        if (!up_has_mm || !gate_has_mm) {
            return false;
        }
    }

    if (has_bias) {
        if (ffn_up_bias->op != expected_bias_op || ffn_gate_bias->op != expected_bias_op) {
            return false;
        }

        if (expected_bias_op == GGML_OP_ADD) {
            const bool up_has_mul   = ffn_up_bias->src[0] == ffn_up_bias_src || ffn_up_bias->src[1] == ffn_up_bias_src;
            const bool gate_has_mul = ffn_gate_bias->src[0] == ffn_gate_bias_src || ffn_gate_bias->src[1] == ffn_gate_bias_src;
            if (!up_has_mul || !gate_has_mul) {
                return false;
            }
        } else { // GGML_OP_ADD_ID
            if (ffn_up_bias->src[0] != ffn_up_bias_src || ffn_gate_bias->src[0] != ffn_gate_bias_src) {
                return false;
            }
            if (ffn_up_bias->src[2] != ffn_up->src[2] || ffn_gate_bias->src[2] != ffn_gate->src[2]) {
                return false;
            }
        }
    }

    if (ffn_up->src[0]->type != ffn_gate->src[0]->type || !ggml_are_same_shape(ffn_up->src[0], ffn_gate->src[0]) ||
        !ggml_are_same_stride(ffn_up->src[0], ffn_gate->src[0])) {
        return false;
    }

    if (ffn_up->src[1] != ffn_gate->src[1]) {
        return false;
    }

    if (is_mul_mat_id && ffn_up->src[2] != ffn_gate->src[2]) {
        return false;
    }

    static constexpr std::array<ggml_glu_op, 4> valid_glu_ops = { GGML_GLU_OP_SWIGLU, GGML_GLU_OP_GEGLU, GGML_GLU_OP_SWIGLU_OAI, GGML_GLU_OP_SWIGLU_CLAMP };

    if (std::find(valid_glu_ops.begin(), valid_glu_ops.end(), ggml_get_glu_op(glu)) == valid_glu_ops.end()) {
        return false;
    }

    if (const bool swapped = ggml_get_op_params_i32(glu, 1); swapped) {
        return false;
    }

    return true;
}

static bool ggml_cuda_should_fuse_mul_mat_vec_f(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool is_mul_mat_id = tensor->op == GGML_OP_MUL_MAT_ID;

    bool use_mul_mat_vec_f =
        (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16) &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    const int cc        = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    use_mul_mat_vec_f = use_mul_mat_vec_f && ggml_cuda_should_use_mmvf(src0->type, cc, warp_size, src0->ne, src0->nb, is_mul_mat_id ? src1->ne[2] : src1->ne[1]);

    //we only support fusion for ncols_dst = 1
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] != 1) {
        return false;
    }


    return use_mul_mat_vec_f;
}

static bool ggml_cuda_should_fuse_mul_mat_vec_q(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
                                   ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) &&
                                   src0->view_src;

    bool use_mul_mat_vec_q = ggml_is_quantized(src0->type) && !bad_padding_clear && src1->type == GGML_TYPE_F32 &&
                             dst->type == GGML_TYPE_F32 && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;

    // fusion is not universally faster on Pascal
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (cc <= GGML_CUDA_CC_PASCAL) {
        return false;
    }
    //we only support fusion for ncols_dst = 1
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] > get_mmvq_mmid_max_batch(src0->type, cc)) {
        return false;
    }

    return use_mul_mat_vec_q;
}

// [TAG_FN_L21_SHEXP] GGML_CUDA_FN_L21_SHEXP=1 (the qwen4exp profile sets it; unset = as before): upstream #29986, the
// shared-expert match is batch independent, so graph_optimize gives every ubatch size the same node order and alloc deps
// (no compute-buffer re-reserve and backend sync on each prompt <-> decode switch); the MMVQ batch test moves to the fusion
static bool ggml_cuda_fn_l21_shexp() {
    static const bool on = [] { const char * e = getenv("GGML_CUDA_FN_L21_SHEXP"); return e && atoi(e) > 0; }();
    return on;
}

static bool ggml_cuda_match_shared_expert(const ggml_cgraph * graph, int routed_idx, int shared_idx) {
    if (routed_idx + 2 >= graph->n_nodes || shared_idx + 2 >= graph->n_nodes || shared_idx < routed_idx + 3) {
        return false;
    }
    const int nodes[] = { routed_idx, routed_idx + 1, routed_idx + 2, shared_idx, shared_idx + 1, shared_idx + 2 };
    const ggml_op ops[] = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU,
                           GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU };
    const int outputs[] = { routed_idx + 2, shared_idx + 2 };
    if (!ggml_can_fuse_subgraph_ext(graph, nodes, 6, ops, outputs, 2)) {
        return false;
    }

    const ggml_tensor * routed = graph->nodes[routed_idx + 2];
    const ggml_tensor * shared = graph->nodes[shared_idx + 2];
    const ggml_tensor * gate = routed->src[0];
    const ggml_tensor * up = routed->src[1];
    const ggml_tensor * shared_gate = shared->src[0];
    const ggml_tensor * shared_up = shared->src[1];
    const auto is_pair = [&](const ggml_tensor * a, const ggml_tensor * b, int idx) {
        return (a == graph->nodes[idx] && b == graph->nodes[idx + 1]) ||
               (b == graph->nodes[idx] && a == graph->nodes[idx + 1]);
    };
    // only batch-size independent checks here: graph_optimize must produce the same graph topology for every ubatch
    // size, otherwise ggml-alloc has to re-reserve (and the scheduler to synchronize) at runtime.
    // the MMVQ batch size check is done in ggml_cuda_try_fuse
    if (!is_pair(gate, up, routed_idx) || !is_pair(shared_gate, shared_up, shared_idx) ||
            !ggml_cuda_should_fuse_mul_mat(up, gate, routed) ||
            !ggml_cuda_should_fuse_mul_mat(shared_up, shared_gate, shared) ||
            !up->src[0]->buffer ||
            (ggml_cuda_fn_l21_shexp() ? !ggml_is_quantized(up->src[0]->type) : !ggml_cuda_should_fuse_mul_mat_vec_q(up))) {
        return false;
    }
    const ggml_tensor * input = up->src[1];
    const ggml_tensor * weight = up->src[0];
    const ggml_tensor * shared_weight = shared_up->src[0];
    if (input->op != GGML_OP_RESHAPE || input->src[0] != shared_up->src[1] ||
            input->ne[1] != 1 || input->ne[3] != 1 || !ggml_is_contiguous(input) ||
            !ggml_is_contiguous(shared_up->src[1]) || !ggml_is_matrix(shared_up->src[1]) ||
            weight->type != shared_weight->type || weight->ne[0] != shared_weight->ne[0] ||
            weight->ne[1] != shared_weight->ne[1] || weight->nb[1] != shared_weight->nb[1] || weight->ne[3] != 1 ||
            !ggml_is_matrix(shared_weight) || !ggml_is_contiguous(shared_weight) ||
            !ggml_is_contiguous(shared_gate->src[0]) || !ggml_is_contiguous(routed) || !ggml_is_contiguous(shared)) {
        return false;
    }
    if (shared_weight->op != GGML_OP_NONE || shared_gate->src[0]->op != GGML_OP_NONE ||
            ggml_get_glu_op(routed) != ggml_get_glu_op(shared) ||
            ggml_get_op_params_f32(routed, 3) != ggml_get_op_params_f32(shared, 3)) {
        return false;
    }
    return true;
}

// [TAG_UPSTREAM_29948] Fused gate/up matmuls + GLU in one MMQ kernel (upstream #29948, prefill widths only).
// GGML_CUDA_MMQ_GLU_FUSION: 0 = off (default, the old path); 1 = only where unfused MMQ computes each tile over all of K,
// so the result bits do not change; 2 = also where unfused MMQ splits K with stream-k (last-bit differences, as upstream).
// [TAG_FN_L22_GLUFUSE] round 22 refresh to the PR head b6e0f0334: the matcher is batch-size independent (graph_optimize
// adds the same alloc deps for every ubatch size, #29986), the hint and src1-precision checks follow the PR, and both
// modes fuse only where the fused kernel (no stream-k) tiles the GPU efficiently.
static int ggml_cuda_mmq_glu_fusion_mode() {
    static const int mode = [] {
        const char * e = getenv("GGML_CUDA_MMQ_GLU_FUSION");
        return e != nullptr ? std::atoi(e) : 0;
    }();
    return mode;
}

// Buffer- and batch-size independent, so graph_optimize produces the same graph topology for every ubatch size (#29986).
static bool ggml_cuda_mul_mat_q_fusion_matches(const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * glu, const int device) {
    if (ggml_cuda_mmq_glu_fusion_mode() == 0) {
        return false;
    }

    const ggml_tensor * x_up = up->src[0];
    const ggml_tensor * y    = up->src[1];

    if (up->op != GGML_OP_MUL_MAT || !ggml_cuda_mmq_fusion_supported(x_up->type) || y->type != GGML_TYPE_F32) {
        return false;
    }

    if (!ggml_cuda_should_fuse_mul_mat(up, gate, glu)) {
        return false;
    }

    // The fused kernel quantizes src1 to Q8_1, only fuse if the src1 precision of up and gate allows it.
    for (const ggml_tensor * mm : { up, gate }) {
        const ggml_prec prec_src1 = (ggml_prec) ggml_get_op_params_i32(mm, 3);
        if (prec_src1 != GGML_PREC_UNDEFINED && prec_src1 != GGML_PREC_Q8 && prec_src1 != GGML_PREC_Q4) {
            return false;
        }
    }

    // The fused write-back applies SWIGLU_OAI with the default alpha and limit of ggml_cuda_op_swiglu_oai_single.
    if (ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU_OAI && (ggml_get_op_params_f32(glu, 2) != 1.702f || ggml_get_op_params_f32(glu, 3) != 7.0f)) {
        return false;
    }

    // The fused kernels have no row bounds checks: whole 128-row MMQ tiles only (the 27B 17408 and Flash-Next 640 rows are).
    if (x_up->ne[1] % 128 != 0) {
        return false;
    }

    // The fused launch has the row tiles (64 output rows each) in gridDim.y and channels * samples in gridDim.z.
    if ((x_up->ne[1] + 63)/64 > 65535 || y->ne[2]*y->ne[3] > 65535) {
        return false;
    }

    // NVIDIA only: the fused write-back relies on the MMA accumulator layout.
    return turing_mma_available(ggml_cuda_info().devices[device].cc);
}

static bool ggml_cuda_should_fuse_mul_mat_q(const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * glu) {
    // A hinted matmul can be fused, but its unfused path may be faster (e.g. FWHT for GGML_HINT_SRC0_IS_HADAMARD).
    if (ggml_get_op_params_i32(up, 1) != GGML_HINT_NONE || ggml_get_op_params_i32(gate, 1) != GGML_HINT_NONE) {
        return false;
    }

    // The padding of a compute-buffer weight that is a view cannot be cleared, see ggml_cuda_mul_mat.
    const auto bad_padding_clear = [](const ggml_tensor * x) {
        return ggml_backend_buffer_get_usage(x->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
            ggml_nbytes(x) != ggml_backend_buffer_get_alloc_size(x->buffer, x) && x->view_src;
    };
    if (!ggml_cuda_mul_mat_q_fusion_matches(up, gate, glu, ggml_cuda_get_device()) || bad_padding_clear(up->src[0]) || bad_padding_clear(gate->src[0])) {
        return false;
    }

    // The batch size checks are done here, not in ggml_cuda_mul_mat_q_fusion_matches.
    const int           device = ggml_cuda_get_device();
    const int           cc     = ggml_cuda_info().devices[device].cc;
    const ggml_tensor * x      = up->src[0];
    const ggml_tensor * y      = up->src[1];
    if (ggml_cuda_should_use_mmvq(x->type, cc, y->ne[1], /*ne01 =*/ x->ne[1]) || !ggml_cuda_should_use_mmq(x->type, cc, y->ne[1], /*n_experts =*/ 0)) {
        return false;
    }

    // MMSB and MMQ-SN keep the widths they take from MMQ (multi-stream verify), as in ggml_cuda_mul_mat.
    for (const ggml_tensor * mm : { up, gate }) {
        if (ggml_cuda_should_use_mmsb(mm->src[0], mm->src[1], mm, cc) || ggml_cuda_should_use_mmqsn(mm->src[0], mm->src[1], mm, cc)) {
            return false;
        }
    }

    // The fused kernel cannot use stream-k, so only fuse if its tiling is efficient.
    if (!ggml_cuda_mmq_fusion_is_efficient(x->type, x->ne[1], y->ne[1], y->ne[2]*y->ne[3], device)) {
        return false;
    }

    if (ggml_cuda_mmq_glu_fusion_mode() == 1) {
        return ggml_cuda_mmq_tiles_whole_k(x->type, cc, x->ne[1], y->ne[1], y->ne[2]*y->ne[3]);
    }
    return true;
}

static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS

    if (ggml_cuda_op_mul_mat_use_fwht(dst) && ggml_cuda_op_fwht(ctx, src1, dst)) {
        return;
    }

    // If src0 is a temporary compute buffer it may have some padding that needs to be cleared for mul_mat_vec_q or mul_mat_q.
    // But if src0 is also a view of another tensor then this cannot be done safely because it may overwrite valid tensor data.
    // Therefore, in such cases use cuBLAS.
    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE
        && ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) && src0->view_src;
    if (bad_padding_clear || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
        return;
    }

    const int cc        = ggml_cuda_info().devices[ctx.device].cc;
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;

    // [TAG_FN_L3_GPU_MMV] a mat-vec the qwen4exp graph marked: the same sums with the loads issued ahead
    if (ggml_fn_l3_get(dst) == GGML_FN_L3_MMV) {
        if (src0->type == GGML_TYPE_F32 && ggml_cuda_should_use_mmvf(src0->type, cc, ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size, src0->ne, src0->nb, ne11) &&
                ggml_cuda_fn_l3_mul_mat_vec_f(ctx, src0, src1, dst)) {
            return;
        }
        if (src0->type == GGML_TYPE_Q8_0 && !ggml_cuda_should_use_mmvf(src0->type, cc, ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size, src0->ne, src0->nb, ne11) &&
                !ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11, /*mul_mat_id =*/ false) &&
                ne01 != 1 && !ggml_cuda_should_use_mmsb(src0, src1, dst, cc) &&
                !ggml_cuda_should_use_mmqsn(src0, src1, dst, cc) &&
                ggml_cuda_should_use_mmvq(src0->type, cc, ne11, /*ne01 =*/ src0->ne[1]) &&
                ggml_cuda_fn_l3_mul_mat_vec_q(ctx, src0, src1, dst)) {
            return;
        }
    }

    if (ggml_cuda_should_use_mmvf(src0->type, cc, ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size, src0->ne, src0->nb, ne11)) {
        // The custom F16 vector kernel can be used over batched cuBLAS GEMM.
        // But this is only faster for GPUs without tensor cores or with a thin src0 matrix (particularly KQV in attention)
        ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, dst);
        return;
    }
    // A transposed vector can still use MMVQ (i.e. ne01 == 1)
    if (ne01 == 1 && ne11 > MMVF_MAX_BATCH_SIZE && ne2 == 1 && ne3 == 1
            && src0->type == GGML_TYPE_F32
            && ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)
            && ggml_cuda_should_use_mmvf(src1->type, cc, warp_size, src1->ne, src1->nb, /*ne11 =*/ 1)) {
        ggml_tensor dst_vec = *dst;
        dst_vec.ne[0] = ne11;
        dst_vec.ne[1] = 1;
        dst_vec.nb[1] = dst_vec.nb[0]*ne11;
        dst_vec.nb[2] = dst_vec.nb[1];
        dst_vec.nb[3] = dst_vec.nb[1];
        ggml_cuda_mul_mat_vec_f(ctx, src1, src0, nullptr, &dst_vec);
        return;
    }
    if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11, /*mul_mat_id =*/ false)) {
        ggml_cuda_mul_mat_f(ctx, src0, src1, nullptr, dst);
        return;
    }
    // [TAG_MMSB] [TAG_SMALLB] 2..16-column Q4_K/Q5_K/Q6_K/Q8_0 matmuls on the int8 tensor cores (mmsb.cu), ahead of
    // MMVQ/MMQ. By default MMVQ keeps every width it takes today, so this only replaces MMQ at the multi-stream verify
    // widths; MMQ itself is unchanged (prefill). GGML_CUDA_SMALLB=0 restores today's dispatch exactly.
    if (ggml_cuda_should_use_mmsb(src0, src1, dst, cc)) {
        ggml_cuda_mul_mat_sb(ctx, src0, src1, dst);
        return;
    }
    // [TAG_MMQSN] MMQ with a cp.async weight ring (mmqsn.cu) for the 5..16-column widths MMQ takes today on weights with
    // >= 2048 rows; bit-identical to MMQ. [TAG_MMQSN_DEFAULT] On by default on cc 1200, GGML_CUDA_MMQSN=0 turns it off.
    // [TAG_MMQSN_WIDE] also 17..32 columns (on by default since 2026-09-28, GGML_CUDA_MMQSN_WIDE=0 turns it off).
    if (ggml_cuda_should_use_mmqsn(src0, src1, dst, cc)) {
        ggml_cuda_mul_mat_qsn(ctx, src0, src1, dst);
        return;
    }
    if (ggml_cuda_should_use_mmvq(src0->type, cc, ne11, /*ne01 =*/ src0->ne[1])) {
        ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, dst);
        return;
    }
    if (ggml_cuda_should_use_mmq(src0->type, cc, ne11, /*n_experts =*/ 0)) {
        ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, dst);
        return;
    }
    ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
}

// returns true when ggml_cuda_mul_mat_id takes the fallback path that requires stream synchronization
// [TAG_MUL_MAT_ID_CUDA_GRAPHS]
static bool ggml_cuda_mul_mat_id_needs_sync(const ggml_tensor * dst, const int cc) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;

    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return true;
    }

    if (dst->ne[2] <= MMVQ_MAX_BATCH_SIZE) {
        if (ggml_is_quantized(src0->type)) {
            if (dst->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cc)) {
                return false;
            }
        } else if (GGML_CUDA_CC_IS_AMD(cc)) {
            return false;
        }
    }

    if (ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[2], /*n_experts=*/src0->ne[2])) {
        return false;
    }

    if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
        return false;
    }

    return true;
}

static void ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;

    // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
    if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        static_assert(MMVQ_MAX_BATCH_SIZE == MMVF_MAX_BATCH_SIZE);
        if (ne2 <= MMVQ_MAX_BATCH_SIZE) {
            if (ggml_is_quantized(src0->type)) {
                const int mmvq_mmid_max = get_mmvq_mmid_max_batch(src0->type, cc);
                if (ne2 <= mmvq_mmid_max) {
                    ggml_cuda_mul_mat_vec_q(ctx, src0, src1, ids, dst);
                    return;
                }
            } else {
                if (GGML_CUDA_CC_IS_AMD(cc)) {
                    ggml_cuda_mul_mat_vec_f(ctx, src0, src1, ids, dst);
                    return;
                }
            }
        }

        if (ggml_cuda_should_use_mmq(src0->type, cc, ne12, /*n_experts=*/ne02)) {
            ggml_cuda_mul_mat_q(ctx, src0, src1, ids, dst);
            return;
        }

        if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
            ggml_cuda_mul_mat_f(ctx, src0, src1, ids, dst);
            return;
        }
    }

    // note: this path should not be reached when recording CUDA graphs, because it requires stream synchronization
    GGML_ASSERT(ggml_cuda_mul_mat_id_needs_sync(dst, cc));
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const ggml_type type_src1_sorted = (src0->type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc))
        || ggml_is_quantized(src0->type) ? GGML_TYPE_F32 : src0->type;
    const ggml_type type_dst_sorted  = GGML_TYPE_F32;
    const size_t ts_src1_sorted = ggml_type_size(type_src1_sorted);
    const size_t ts_dst_sorted  = ggml_type_size(type_dst_sorted);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;

    std::vector<int32_t> ids_to_sorted_host;
    ids_to_sorted_host.reserve(2*ne_get_rows);
    std::vector<int32_t> ids_from_sorted_host(ne_get_rows);

    ggml_cuda_pool_alloc<int32_t> ids_buf_dev(ctx.pool(), 2*ne_get_rows);

    std::vector<int32_t> tokens_per_expert(ne02);

    ggml_cuda_pool_alloc<char> src1_sorted(ctx.pool(), ne12*n_expert_used*ne10*ts_src1_sorted);
    ggml_cuda_pool_alloc<char>  dst_sorted(ctx.pool(), ne2 *n_expert_used* ne0*ts_dst_sorted);

    std::vector<char> ids_host(ggml_nbytes(ids));
    CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int64_t i02 = 0; i02 < ne02; ++i02) { // expert matrices
        for (int64_t i12 = 0; i12 < ne12; ++i12) { // tokens
            for (int64_t iex = 0; iex < n_expert_used; ++iex) {
                const int32_t expert_to_use = *(const int32_t *)(ids_host.data() + i12*ids->nb[1] + iex*ids->nb[0]);
                assert(expert_to_use >= 0 && expert_to_use < ne02);
                if (expert_to_use == i02) {
                    ids_from_sorted_host[i12*n_expert_used + iex] = ids_to_sorted_host.size();
                    ids_to_sorted_host.push_back(i12*ne11 + iex % ne11);
                    tokens_per_expert[i02]++;
                    break;
                }
            }
        }
    }
    GGML_ASSERT(ids_to_sorted_host.size() == size_t(ne_get_rows));

    ids_to_sorted_host.insert(ids_to_sorted_host.end(), ids_from_sorted_host.begin(), ids_from_sorted_host.end());

    CUDA_CHECK(cudaMemcpyAsync(ids_buf_dev.ptr, ids_to_sorted_host.data(), 2*ne_get_rows*sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const int32_t * ids_to_sorted   = ids_buf_dev.ptr + 0*ne_get_rows;
    const int32_t * ids_from_sorted = ids_buf_dev.ptr + 1*ne_get_rows;

    get_rows_cuda(src1->data, src1->type, ids_to_sorted, src1_sorted.ptr, type_src1_sorted,
        ne10, nb11, nb12, nb13,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, stream);
    CUDA_CHECK(cudaGetLastError());

    char * src1_data_cur = (char *) src1_sorted.ptr;
    char *  dst_data_cur = (char *)  dst_sorted.ptr;
    for (int64_t i02 = 0; i02 < ne02; ++i02) {
        if (tokens_per_expert[i02] == 0) {
            continue;
        }

        ggml_tensor src0_slice = *src0;
        src0_slice.ne[2]    = 1;
        src0_slice.nb[3]    = src0_slice.nb[2];
        src0_slice.op       = GGML_OP_VIEW;
        src0_slice.view_src = dst->src[0]; // non-const pointer to src0
        src0_slice.data     = (char *) src0->data + i02*nb02;

        ggml_tensor src1_slice;
        memset(&src1_slice, 0, sizeof(src1_slice));
        src1_slice.buffer = src1->buffer;
        src1_slice.type   = type_src1_sorted;
        src1_slice.ne[0]  = ne10;
        src1_slice.ne[1]  = tokens_per_expert[i02];
        src1_slice.ne[2]  = 1;
        src1_slice.ne[3]  = 1;
        src1_slice.nb[0]  = ts_src1_sorted;
        src1_slice.nb[1]  = src1_slice.ne[0] * src1_slice.nb[0];
        src1_slice.nb[2]  = src1_slice.ne[1] * src1_slice.nb[1];
        src1_slice.nb[3]  = src1_slice.ne[2] * src1_slice.nb[2];
        src1_slice.data   = src1_data_cur;

        ggml_tensor dst_slice;
        memset(&dst_slice, 0, sizeof(dst_slice));
        memcpy(dst_slice.op_params, dst->op_params, sizeof(dst_slice.op_params));
        dst_slice.buffer = dst->buffer;
        dst_slice.type   = type_dst_sorted;
        dst_slice.ne[0]  = ne0;
        dst_slice.ne[1]  = tokens_per_expert[i02];
        dst_slice.ne[2]  = 1;
        dst_slice.ne[3]  = 1;
        dst_slice.nb[0]  = ts_dst_sorted;
        dst_slice.nb[1]  = dst_slice.ne[0] * dst_slice.nb[0];
        dst_slice.nb[2]  = dst_slice.ne[1] * dst_slice.nb[1];
        dst_slice.nb[3]  = dst_slice.ne[2] * dst_slice.nb[2];
        dst_slice.data   = dst_data_cur;

        ggml_cuda_mul_mat(ctx, &src0_slice, &src1_slice, &dst_slice);
        CUDA_CHECK(cudaGetLastError());

        src1_data_cur += src1_slice.nb[2];
        dst_data_cur  +=  dst_slice.nb[2];
    }

    get_rows_cuda(dst_sorted.ptr, type_dst_sorted, ids_from_sorted, dst->data, dst->type,
        ne0, ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        nb1, nb2, nb3, stream);
}

static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    // turbo-op-probe: report ONCE per (op, src0 type) every op that consumes a turbo
    // tensor, so we can see which kernel actually touches the KV/indexer caches instead
    // of inferring it. TURBO_OP_PROBE=0 silences.
    {
        static std::set<int> seen;
        const ggml_tensor * a = dst->src[0];
        const ggml_tensor * b = dst->src[1];
        auto is_turbo = [](const ggml_tensor * t) {
            return t && (t->type == GGML_TYPE_TURBO2_0 || t->type == GGML_TYPE_TURBO3_0 ||
                         t->type == GGML_TYPE_TURBO4_0 || t->type == GGML_TYPE_TURBO4P_0 || t->type == GGML_TYPE_TURBO5P_0 ||
                         t->type == GGML_TYPE_TURBO5P512_0);
        };
        // read once: getenv takes the CRT environment lock and walks the environment, and this runs for every node of
        // every eager or captured evaluation
        static const bool op_probe = [] {
            const char * e = getenv("TURBO_OP_PROBE");
            return e != nullptr && e[0] == '1';
        }();
        if (op_probe && (is_turbo(a) || is_turbo(b))) {
            const int key = ((int) dst->op << 8) | (int) (a ? a->type : GGML_TYPE_COUNT);
            if (seen.insert(key).second) {
                fprintf(stderr, "turbo-op: %s src0=%s[%d,%d] src1=%s dst=%s\n",
                        ggml_op_name(dst->op),
                        a ? ggml_type_name(a->type) : "-", a ? (int) a->ne[0] : 0, a ? (int) a->ne[1] : 0,
                        b ? ggml_type_name(b->type) : "-", ggml_type_name(dst->type));
                fflush(stderr);
            }
        }
    }
    switch (dst->op) {
        case GGML_OP_ARGMAX:
            ggml_cuda_argmax(ctx, dst);
            break;
        case GGML_OP_COUNT_EQUAL:
            ggml_cuda_count_equal(ctx, dst);
            break;
        case GGML_OP_REPEAT:
            ggml_cuda_op_repeat(ctx, dst);
            break;
        case GGML_OP_REPEAT_BACK:
            ggml_cuda_op_repeat_back(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_cuda_op_get_rows(ctx, dst);
            break;
        case GGML_OP_GET_ROWS_BACK:
            ggml_cuda_op_get_rows_back(ctx, dst);
            break;
        case GGML_OP_SET_ROWS:
            ggml_cuda_op_set_rows(ctx, dst);
            break;
        case GGML_OP_TURBOT_SET_ROWS:   // [TAG_TURBOT]
            ggml_cuda_op_turbot_set_rows(ctx, dst);
            break;
        case GGML_OP_TURBO_WHT:
            ggml_cuda_turbo_wht(ctx, dst);
            break;
        case GGML_OP_SET:
            ggml_cuda_op_set(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggml_cuda_cpy(ctx, dst->src[0], dst->src[1]);
            break;
        case GGML_OP_CONT:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
        case GGML_OP_ADD1: // TODO: more efficient implementation
            ggml_cuda_op_add(ctx, dst);
            break;
        case GGML_OP_ADD_ID:
            ggml_cuda_op_add_id(ctx, dst);
            break;
        case GGML_OP_SUB:
            ggml_cuda_op_sub(ctx, dst);
            break;
        case GGML_OP_ACC:
            ggml_cuda_op_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_cuda_op_mul(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_cuda_op_div(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_ABS:
                    ggml_cuda_op_abs(ctx, dst);
                    break;
                case GGML_UNARY_OP_SGN:
                    ggml_cuda_op_sgn(ctx, dst);
                    break;
                case GGML_UNARY_OP_NEG:
                    ggml_cuda_op_neg(ctx, dst);
                    break;
                case GGML_UNARY_OP_STEP:
                    ggml_cuda_op_step(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU:
                    ggml_cuda_op_gelu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SILU:
                    ggml_cuda_op_silu(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_ERF:
                    ggml_cuda_op_gelu_erf(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    ggml_cuda_op_gelu_quick(ctx, dst);
                    break;
                case GGML_UNARY_OP_TANH:
                    ggml_cuda_op_tanh(ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    ggml_cuda_op_relu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    ggml_cuda_op_sigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    ggml_cuda_op_hardsigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    ggml_cuda_op_hardswish(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXP:
                    ggml_cuda_op_exp(ctx, dst);
                    break;
                case GGML_UNARY_OP_ELU:
                    ggml_cuda_op_elu(ctx, dst);
                    break;
                case GGML_UNARY_OP_XIELU:
                    ggml_cuda_op_xielu(ctx, dst);
                    break;
                case GGML_UNARY_OP_FLOOR:
                    ggml_cuda_op_floor(ctx, dst);
                    break;
                case GGML_UNARY_OP_CEIL:
                    ggml_cuda_op_ceil(ctx, dst);
                    break;
                case GGML_UNARY_OP_ROUND:
                    ggml_cuda_op_round(ctx, dst);
                    break;
                case GGML_UNARY_OP_TRUNC:
                    ggml_cuda_op_trunc(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXPM1:
                    ggml_cuda_op_expm1(ctx, dst);
                    break;
                case GGML_UNARY_OP_SOFTPLUS:
                    ggml_cuda_op_softplus(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(dst)) {
                case GGML_GLU_OP_REGLU:
                    ggml_cuda_op_reglu(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU:
                    ggml_cuda_op_geglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU:
                    ggml_cuda_op_swiglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    ggml_cuda_op_swiglu_oai(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_ERF:
                    ggml_cuda_op_geglu_erf(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_QUICK:
                    ggml_cuda_op_geglu_quick(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    ggml_cuda_op_swiglu_clamp(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_cuda_op_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_cuda_op_group_norm(ctx, dst);
            break;
        case GGML_OP_L2_NORM:
            ggml_cuda_op_l2_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            ggml_cuda_op_concat(ctx, dst);
            break;
        case GGML_OP_UPSCALE:
            ggml_cuda_op_upscale(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_cuda_op_pad(ctx, dst);
            break;
        case GGML_OP_PAD_REFLECT_1D:
            ggml_cuda_op_pad_reflect_1d(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_cuda_op_arange(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_cuda_op_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_cuda_op_leaky_relu(ctx, dst);
            break;
        case GGML_OP_SILU_BACK:
            ggml_cuda_op_silu_back(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_cuda_op_rms_norm(ctx, dst);
            break;
        case GGML_OP_RMS_NORM_BACK:
            ggml_cuda_op_rms_norm_back(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            ggml_cuda_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            break;
        case GGML_OP_MUL_MAT_ID:
            ggml_cuda_mul_mat_id(ctx, dst);
            break;
        case GGML_OP_OUT_PROD:
            ggml_cuda_out_prod(ctx, dst);
            break;
        case GGML_OP_SCALE:
            ggml_cuda_op_scale(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggml_cuda_op_sqr(ctx, dst);
            break;
        case GGML_OP_SQRT:
            ggml_cuda_op_sqrt(ctx, dst);
            break;
        case GGML_OP_SIN:
            ggml_cuda_op_sin(ctx, dst);
            break;
        case GGML_OP_COS:
            ggml_cuda_op_cos(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_cuda_op_clamp(ctx, dst);
            break;
        case GGML_OP_LOG:
            ggml_cuda_op_log(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
                break;
        case GGML_OP_DIAG:
            ggml_cuda_op_diag(ctx, dst);
            break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_cuda_op_diag_mask_inf(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_cuda_op_soft_max(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX_BACK:
            ggml_cuda_op_soft_max_back(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_cuda_op_rope(ctx, dst);
            break;
        case GGML_OP_ROPE_BACK:
            ggml_cuda_op_rope_back(ctx, dst);
            break;
        case GGML_OP_ROLL:
            ggml_cuda_op_roll(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_cuda_op_im2col(ctx, dst);
            break;
        case GGML_OP_IM2COL_3D:
            ggml_cuda_op_im2col_3d(ctx, dst);
            break;
        case GGML_OP_CONV_2D:
            ggml_cuda_op_conv2d(ctx, dst);
            break;
        case GGML_OP_CONV_3D:
            ggml_cuda_op_conv3d(ctx, dst);
            break;
        case GGML_OP_CONV_2D_DW:
            ggml_cuda_op_conv2d_dw(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            ggml_cuda_conv_2d_transpose_p0(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_cuda_op_conv_transpose_1d(ctx,dst);
            break;
        case GGML_OP_COL2IM_1D:
            ggml_cuda_op_col2im_1d(ctx, dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_cuda_op_pool2d(ctx, dst);
            break;
        case GGML_OP_POOL_1D:
            ggml_cuda_op_pool1d(ctx, dst);
            break;
        case GGML_OP_SUM:
            ggml_cuda_op_sum(ctx, dst);
            break;
        case GGML_OP_CUMSUM:
            ggml_cuda_op_cumsum(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            ggml_cuda_op_sum_rows(ctx, dst);
            break;
        case GGML_OP_MEAN:
            ggml_cuda_op_mean(ctx, dst);
            break;
        case GGML_OP_SSM_CONV:
            ggml_cuda_op_ssm_conv(ctx, dst);
            break;
        case GGML_OP_SSM_SCAN:
            ggml_cuda_op_ssm_scan(ctx, dst);
            break;
        case GGML_OP_TOP_K:
            ggml_cuda_op_top_k(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            ggml_cuda_op_argsort(ctx, dst);
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_cuda_flash_attn_ext(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_cuda_cross_entropy_loss(ctx, dst);
            break;
        case GGML_OP_TRI:
            ggml_cuda_op_tri(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV6:
            ggml_cuda_op_rwkv_wkv6(ctx, dst);
            break;
        case GGML_OP_GATED_LINEAR_ATTN:
            ggml_cuda_op_gated_linear_attn(ctx, dst);
            break;
        case GGML_OP_GATED_DELTA_NET:
            ggml_cuda_op_gated_delta_net(ctx, dst);
            break;
        case GGML_OP_GATED_DELTA_NET_REPLAY: // [TAG_4C_GDN_REPLAY]
            ggml_cuda_op_gated_delta_net_replay(ctx, dst, nullptr);
            break;
        case GGML_OP_MOE_HOST_POST: // [TAG_MOE_BRIDGE]
            ggml_cuda_op_moe_host_post(ctx, dst);
            break;
        case GGML_OP_MOE_HOST_WAIT: // [TAG_MOE_BRIDGE]
            ggml_cuda_op_moe_host_wait(ctx, dst);
            break;
        case GGML_OP_MOE_HOST_FETCH: // [TAG_FN_R4_BRIDGE_DMA]
            ggml_cuda_op_moe_host_fetch(ctx, dst);
            break;
        case GGML_OP_QSA_MASK: // [TAG_FN_R4_QSA_POS]
            if (!ggml_cuda_fn_l4_qsa_list_mask(ctx, dst)) { // [TAG_FN_L4_QSA_LIST] the mask at the selection + the FA's lists
                ggml_cuda_op_qsa_mask(ctx, dst);
            }
            break;
        case GGML_OP_DSV4_HC_COMB:
            ggml_cuda_op_dsv4_hc_comb(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_PRE:
            if (!ggml_cuda_fn_l3_hc_pre_q8(ctx, dst)) { // [TAG_FN_L3_GPU_Q8F] the same op, plus the q8_1 copy when marked
                ggml_cuda_op_dsv4_hc_pre(ctx, dst);
            }
            break;
        case GGML_OP_DSV4_HC_POST:
            ggml_cuda_op_dsv4_hc_post(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV7:
            ggml_cuda_op_rwkv_wkv7(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            ggml_cuda_cross_entropy_loss_back(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            ggml_cuda_opt_step_adamw(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_SGD:
            ggml_cuda_opt_step_sgd(ctx, dst);
            break;
        case GGML_OP_SOLVE_TRI:
            ggml_cuda_op_solve_tri(ctx, dst);
            break;
        case GGML_OP_FILL:
            ggml_cuda_op_fill(ctx, dst);
            break;
        case GGML_OP_LIGHTNING_INDEXER:
            ggml_cuda_lightning_indexer(ctx, dst);
            break;
        case GGML_OP_TOP_K_MASK:
            ggml_cuda_op_top_k_mask(ctx, dst);
            break;
        default:
            return false;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: %s failed\n", __func__, ggml_op_desc(dst));
        CUDA_CHECK(err);
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////

// backend

static const char * ggml_backend_cuda_get_name(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    return cuda_ctx->name.c_str();
}

static void ggml_backend_cuda_free(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    delete cuda_ctx;
    delete backend;
}

static void ggml_backend_cuda_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    ggml_cuda_h2d_or_bounce((char *) tensor->data + offset, data, size, cuda_ctx->stream()); // [TAG_FN_L23_PINFIX]
}

// [TAG_FN_L4_HOST_BATCHCPY] host -> device copies into whole device tensors as one cudaMemcpyBatchAsync on the backend's
// stream (source read in stream order, as cudaMemcpyAsync does); false (nothing queued) where the API is missing or
// refuses the batch, so the caller copies one by one
// [TAG_FN_L14_PFSD2D] device-to-device copies on the backend's stream, one batch (the prefill stream's banks from the hot
// set's slots); one copy at a time when the batch call is not there or fails
static bool ggml_backend_cuda_copy_d2d_batch_async(ggml_backend_t backend, int n, void * const * dst, const void * const * src,
        const size_t * sizes) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    if (n <= 0) {
        return true;
    }
    ggml_cuda_set_device(cuda_ctx->device);
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000 && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) // [TAG_FN_L15_CUDA12] the CUDA 13 signature (12.8/12.9 have a fail index)
    {
        static thread_local std::vector<void *>       d;
        static thread_local std::vector<const void *> s;
        static thread_local std::vector<size_t>       len;
        d.assign(dst, dst + n);
        s.assign(src, src + n);
        len.assign(sizes, sizes + n);
        cudaMemcpyAttributes attr = {};
        attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
        size_t attr_idx = 0;
        if (cudaMemcpyBatchAsync(d.data(), s.data(), len.data(), (size_t) n, &attr, &attr_idx, 1, cuda_ctx->stream()) == cudaSuccess) {
            return true;
        }
        (void) cudaGetLastError();
    }
#endif
    for (int i = 0; i < n; ++i) {
        if (cudaMemcpyAsync(dst[i], src[i], sizes[i], cudaMemcpyDeviceToDevice, cuda_ctx->stream()) != cudaSuccess) {
            (void) cudaGetLastError();
            return false;
        }
    }
    return true;
}

static bool ggml_backend_cuda_set_tensors_batch_async(ggml_backend_t backend, int n, ggml_tensor * const * tensors,
        const void * const * data, const size_t * sizes) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000 && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) // [TAG_FN_L15_CUDA12] the CUDA 13 signature (12.8/12.9 have a fail index)
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    if (n <= 0) {
        return true;
    }
    static thread_local std::vector<void *>       dst;
    static thread_local std::vector<const void *> src;
    static thread_local std::vector<size_t>       len;
    dst.resize(n);
    src.resize(n);
    len.resize(n);
    for (int i = 0; i < n; ++i) {
        const ggml_tensor * t = tensors[i];
        ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
        if (buf == nullptr || buf->buft != ggml_backend_cuda_buffer_type(cuda_ctx->device) || sizes[i] > ggml_nbytes(t)) {
            return false;
        }
        dst[i] = t->data;
        src[i] = data[i];
        len[i] = sizes[i];
    }
    ggml_cuda_set_device(cuda_ctx->device);
    cudaMemcpyAttributes attr = {};
    attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    size_t attr_idx = 0;
    const cudaError_t e = cudaMemcpyBatchAsync(dst.data(), src.data(), len.data(), (size_t) n, &attr, &attr_idx, 1, cuda_ctx->stream());
    if (e != cudaSuccess) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            GGML_LOG_WARN("%s: [TAG_FN_L4_HOST_BATCHCPY] cudaMemcpyBatchAsync failed (%s): copies go one by one\n", __func__,
                    cudaGetErrorString(e));
        }
        (void) cudaGetLastError();
        return false;
    }
    return true;
#else
    GGML_UNUSED(backend); GGML_UNUSED(n); GGML_UNUSED(tensors); GGML_UNUSED(data); GGML_UNUSED(sizes);
    return false;
#endif
}

// [TAG_FN_L16_UPLOAD] one table entry of a packed upload: the payload at slot + off goes to dst
struct ggml_cuda_upload_item {
    uint64_t dst;
    uint32_t off;
    uint32_t size;
};
static_assert(sizeof(ggml_cuda_upload_item) == 16, "the payload after the table must stay 16-byte aligned");

static constexpr size_t GGML_CUDA_UPLOAD_SLOT_BYTES = 1024*1024; // [TAG_FN_L16_UPLOAD] per slot, host and device
// [TAG_FN_L16_UPLOAD] larger copies go plain. 96 KiB, not 64: the qwen4exp MTP draft's one-row KQ mask (33,024 window cells
// x f16 = 66,048 B) and its h rows of a two-token decode (81,920 B) were just over 64 KiB, so every draft decode still
// sent one or two copies plain; packing a copy of this size costs about what its own cudaMemcpyAsync call costs
static constexpr size_t GGML_CUDA_UPLOAD_BIG        = 96*1024;

// [TAG_FN_L16_UPLOAD] bytes of an entry one block copies: a 96 KiB entry is 12 blocks, not one block that waits for its
// loads 24 times in a row
static constexpr uint32_t GGML_CUDA_UPLOAD_CHUNK = 8*1024;

// [TAG_FN_L16_UPLOAD] block (x, y): chunk x of table entry y, copied to its device address (16-byte words when dst allows;
// the payloads and the chunk starts are 16-byte aligned)
static __global__ void k_unpack_inputs(const char * __restrict__ slot) {
    const ggml_cuda_upload_item it = ((const ggml_cuda_upload_item *) slot)[blockIdx.y];
    const uint32_t c0 = blockIdx.x*GGML_CUDA_UPLOAD_CHUNK;
    if (c0 >= it.size) {
        return;
    }
    const uint32_t c1  = min(it.size, c0 + GGML_CUDA_UPLOAD_CHUNK);
    const char *   src = slot + it.off;
    char *         dst = (char *) it.dst;
    uint32_t       i0  = c0;
    if ((it.dst & 15) == 0) {
        const uint32_t e16 = c0 + ((c1 - c0) & ~15u);
        for (uint32_t i = c0 + 16*threadIdx.x; i < e16; i += 16*blockDim.x) {
            *(int4 *) (dst + i) = *(const int4 *) (src + i);
        }
        i0 = e16;
    }
    for (uint32_t i = i0 + threadIdx.x; i < c1; i += blockDim.x) {
        dst[i] = src[i];
    }
}

static bool ggml_cuda_upload_slot_alloc(ggml_cuda_upload_slot & s, size_t cap) {
    s.host = (char *) ggml_cuda_host_malloc(cap);
    if (s.host == nullptr) {
        return false;
    }
    if (cudaMalloc((void **) &s.dev, cap) != cudaSuccess ||
        cudaEventCreateWithFlags(&s.done, cudaEventDisableTiming) != cudaSuccess) {
        (void) cudaGetLastError();
        if (s.dev != nullptr) {
            (void) cudaFree(s.dev);
        }
        (void) cudaFreeHost(s.host);
        s = ggml_cuda_upload_slot();
        return false;
    }
    s.cap = cap;
    return true;
}

// [TAG_FN_L16_UPLOAD] host -> device copies into whole device tensors (LLAMA_FN_L4_HOST_BATCHCPY=2): copies over
// GGML_CUDA_UPLOAD_BIG go first as plain cudaMemcpyAsync; the rest are packed into a pinned slot (a table of {dst, off,
// size}, then the 16-byte aligned payloads), sent as one copy to the slot's device twin, and written to their tensors by
// one unpack kernel. All on the backend's stream, before the split's graph. A slot is reused only after the event behind
// its last unpack. The same bytes reach the same tensors. Without a slot: the cudaMemcpyBatchAsync batch (mode 1), else
// false (nothing queued).
static bool ggml_backend_cuda_set_tensors_pack_async(ggml_backend_t backend, int n, ggml_tensor * const * tensors,
        const void * const * data, const size_t * sizes) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    if (n <= 0) {
        return true;
    }
    size_t n_small = 0;
    for (int i = 0; i < n; ++i) {
        const ggml_tensor * t = tensors[i];
        ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
        if (buf == nullptr || buf->buft != ggml_backend_cuda_buffer_type(cuda_ctx->device) || sizes[i] > ggml_nbytes(t)) {
            return false;
        }
        n_small += sizes[i] > 0 && sizes[i] <= GGML_CUDA_UPLOAD_BIG ? 1 : 0;
    }
    if (cuda_ctx->upload_off) {
        return ggml_backend_cuda_set_tensors_batch_async(backend, n, tensors, data, sizes);
    }
    ggml_cuda_set_device(cuda_ctx->device);
    ggml_cuda_upload_slot & s = cuda_ctx->upload_slot[cuda_ctx->upload_next];
    if (s.host == nullptr && !ggml_cuda_upload_slot_alloc(s, GGML_CUDA_UPLOAD_SLOT_BYTES)) {
        cuda_ctx->upload_off = true;
        GGML_LOG_WARN("%s: [TAG_FN_L16_UPLOAD] no staging slot (%zu bytes pinned + device): batched copies (mode 1) from now on\n",
                __func__, GGML_CUDA_UPLOAD_SLOT_BYTES);
        return ggml_backend_cuda_set_tensors_batch_async(backend, n, tensors, data, sizes);
    }
    cudaStream_t stream = cuda_ctx->stream();

    // [TAG_FN_L16_UPLOAD] which backend packs what (the target and the MTP draft context each have their own): a line at
    // the first upload and every 4096th
    cuda_ctx->upload_n_calls++;
    const bool note = cuda_ctx->upload_n_calls == 1 || cuda_ctx->upload_n_calls % 4096 == 0;
    auto note_line = [&]() {
        const double nc = (double) cuda_ctx->upload_n_calls;
        GGML_LOG_INFO("%s: [TAG_FN_L16_UPLOAD] %s backend %p: %" PRIu64 " uploads, per upload %.2f copies packed (%.1f KiB) "
                "and %.2f sent plain\n", __func__, cuda_ctx->name.c_str(), (void *) cuda_ctx, cuda_ctx->upload_n_calls,
                cuda_ctx->upload_n_packed/nc, cuda_ctx->upload_b_packed/1024.0/nc, cuda_ctx->upload_n_plain/nc);
    };

    for (int i = 0; i < n; ++i) {
        if (sizes[i] > GGML_CUDA_UPLOAD_BIG) {
            CUDA_CHECK(cudaMemcpyAsync(tensors[i]->data, data[i], sizes[i], cudaMemcpyHostToDevice, stream));
            cuda_ctx->upload_n_plain++;
        }
    }
    if (n_small == 0) {
        if (note) {
            note_line();
        }
        return true;
    }

    // the last unpack that read this slot (and the copy before it) is done
    if (s.busy) {
        CUDA_CHECK(cudaEventSynchronize(s.done));
        s.busy = false;
    }

    ggml_cuda_upload_item * tab = (ggml_cuda_upload_item *) s.host;
    size_t   pos    = n_small*sizeof(ggml_cuda_upload_item);
    uint32_t n_pack = 0;
    size_t   sz_max = 0;
    for (int i = 0; i < n; ++i) {
        const size_t sz = sizes[i];
        if (sz == 0 || sz > GGML_CUDA_UPLOAD_BIG) {
            continue;
        }
        if (pos + sz > s.cap) {
            CUDA_CHECK(cudaMemcpyAsync(tensors[i]->data, data[i], sz, cudaMemcpyHostToDevice, stream)); // the slot is full
            cuda_ctx->upload_n_plain++;
            continue;
        }
        memcpy(s.host + pos, data[i], sz);
        tab[n_pack].dst  = (uint64_t) (uintptr_t) tensors[i]->data;
        tab[n_pack].off  = (uint32_t) pos;
        tab[n_pack].size = (uint32_t) sz;
        n_pack++;
        pos = GGML_PAD(pos + sz, 16);
        sz_max = std::max(sz_max, sz);
    }
    if (n_pack == 0) {
        if (note) {
            note_line();
        }
        return true;
    }

    CUDA_CHECK(cudaMemcpyAsync(s.dev, s.host, pos, cudaMemcpyHostToDevice, stream));
    const dim3 grid((unsigned) ((sz_max + GGML_CUDA_UPLOAD_CHUNK - 1)/GGML_CUDA_UPLOAD_CHUNK), n_pack, 1);
    k_unpack_inputs<<<grid, 256, 0, stream>>>(s.dev);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(s.done, stream));
    s.busy = true;
    cuda_ctx->upload_next ^= 1;
    cuda_ctx->upload_n_packed += n_pack;
    cuda_ctx->upload_b_packed += pos;
    if (note) {
        note_line();
    }
    return true;
}

static void ggml_backend_cuda_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(ggml_cuda_memcpy_oom_retry(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cuda_ctx->stream())); // [TAG_FN_L23_PINFIX] R-L23-7
}

static void ggml_backend_cuda_set_tensor_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpy2DAsync(
        (char *) tensor->data + offset, stride_tensor, data, stride_data, size, n_copies, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

static void ggml_backend_cuda_get_tensor_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

static bool ggml_backend_cuda_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    if (!ggml_backend_is_cuda(backend_src) || !ggml_backend_is_cuda(backend_dst)) {
        return false;
    }

    if (!ggml_backend_buffer_is_cuda(buf_src) || !ggml_backend_buffer_is_cuda(buf_dst)) {
        return false;
    }

    // device -> device copy
    ggml_backend_cuda_context * cuda_ctx_src = (ggml_backend_cuda_context *) backend_src->context;
    ggml_backend_cuda_context * cuda_ctx_dst = (ggml_backend_cuda_context *) backend_dst->context;

    ggml_backend_cuda_buffer_context * buf_ctx_src = (ggml_backend_cuda_buffer_context *) buf_src->context;
    ggml_backend_cuda_buffer_context * buf_ctx_dst = (ggml_backend_cuda_buffer_context *) buf_dst->context;

    if (cuda_ctx_src->device != buf_ctx_src->device || cuda_ctx_dst->device != buf_ctx_dst->device) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: backend and buffer devices do not match\n", __func__);
#endif // NDEBUG
        return false;
    }

    if (backend_src != backend_dst) {
        // copy on src stream
        // compare the backing physical devices: distinct virtual devices may share one physical GPU,
        // in which case a same-device copy (not a peer copy) is required
        const int src_physical = ggml_cuda_get_physical_device(cuda_ctx_src->device);
        const int dst_physical = ggml_cuda_get_physical_device(cuda_ctx_dst->device);
        if (src_physical == dst_physical) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, ggml_nbytes(dst), cuda_ctx_src->stream()));
#endif // GGML_CUDA_NO_PEER_COPY
        }

        // record event on src stream after the copy
        if (!cuda_ctx_src->copy_event) {
            ggml_cuda_set_device(cuda_ctx_src->device);
            CUDA_CHECK(cudaEventCreateWithFlags(&cuda_ctx_src->copy_event, cudaEventDisableTiming));
        }

        CUDA_CHECK(cudaEventRecord(cuda_ctx_src->copy_event, cuda_ctx_src->stream()));

        // wait on dst stream for the copy to complete
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx_dst->stream(), cuda_ctx_src->copy_event, 0));
    } else {
        // src and dst are on the same backend
        CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
    }
    return true;
}

static void ggml_backend_cuda_synchronize(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));

    GGML_UNUSED(backend);
}

static bool ggml_cuda_is_view_or_noop(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}

#ifdef USE_CUDA_GRAPH
static bool ggml_cuda_graph_debug_enabled() {
    static const bool enabled = ggml_cuda_env_flag("GGML_CUDA_GRAPH_DEBUG");
    return enabled;
}

static bool ggml_cuda_graph_check_compability(ggml_cgraph * cgraph) {

    bool use_cuda_graph = true;
    // Loop over nodes in GGML graph to obtain info needed for CUDA graph

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (ggml_cuda_is_view_or_noop(node)) {
            continue;
        }

        // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
        if (node->op == GGML_OP_MUL_MAT_ID) {
            const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
            if (ggml_cuda_mul_mat_id_needs_sync(node, cc)) {
                // the mul_mat_id fallback path synchronizes the stream, so we cannot use CUDA graphs
                // ref: https://github.com/ggml-org/llama.cpp/pull/18958
                use_cuda_graph = false;
                if (ggml_cuda_graph_debug_enabled()) {
                    GGML_LOG_DEBUG("%s: disabling CUDA graphs due to unsupported node type\n", __func__);
                }
            }
        }

        // Chunked GDN allocates scratch via ggml_cuda_pool_alloc; on pool_leg those addresses can
        // be freed/reused between capture and replay, corrupting the baked-in kernel pointers.
        // Decode is unaffected (T=1 is ineligible)
        if (ggml_cuda_gdn_op_is_chunked(node)) {
            use_cuda_graph = false;
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: disabling CUDA graphs due to chunked GATED_DELTA_NET\n", __func__);
#endif
        }

        if (!use_cuda_graph) {
            break;
        }
    }

    return use_cuda_graph;
}

// [TAG_CUDA_GRAPH_STRUCT_KEY] Key the per-context CUDA graphs by the graph STRUCTURE (op, type, shape
// of every node), not by the address of the first node. llama rebuilds the ggml graph whenever its
// reuse check fails, and the DFlash drafter fails it on every decode (two batch shapes per step, one
// of variable length), so with a pointer key every draft decode started a fresh warmup and never
// captured: ~150 raw launches per draft decode at ~7 us each on WDDM. With a structural key a rebuilt
// graph of the same shape lands on the same object; the node-property comparison in
// ggml_cuda_graph_update_required still decides whether it can be replayed, so a shape whose
// allocation moved simply re-warms as before. Distinct shapes get distinct objects, and the 10 s
// eviction sweep bounds the map.
// [TAG_FN_L22_GRAPHKEY] big > 0: every dimension >= big is hashed as one value, so the decode graphs of a new KV bucket land
// on the old object and take the RECAP path (capture + cudaGraphExecUpdate); the property compare is unchanged.
// [TAG_FN_L22_GRAPHKEEP] rows: the token rows of the graph, from the weight matmuls (src1 rows; MUL_MAT_ID src1->ne[2]).
// Not the widest node: GDN decode graphs have conv / concat nodes with ne[1] = d_inner. Only a src0 in a weights buffer
// counts. No GET_ROWS: its id count is not a token count in decode graphs (Flash-Next QSA GET_ROWS(pool_idxs, top_k) =
// 512 ids per row, the DFlash2 head / selector gathers of weight rows), and those graphs ran eagerly without a CUDA graph
// (r22 end test T2: FN 57.7 vs 20.2 ms per step at 18K, 27B +6 % ms per step)
static bool ggml_cuda_fn_l22_is_weight(const ggml_tensor * a) {
    return a->op == GGML_OP_NONE && a->view_src == nullptr && a->buffer != nullptr &&
           ggml_backend_buffer_get_usage(a->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

static int64_t ggml_cuda_fn_l22_node_rows(const ggml_tensor * t) {
    const ggml_tensor * a = t->src[0];
    const ggml_tensor * b = t->src[1];
    if (a == nullptr || b == nullptr) {
        return 0;
    }
    if (t->op == GGML_OP_MUL_MAT && ggml_cuda_fn_l22_is_weight(a)) {
        return ggml_nrows(b) / std::max<int64_t>(1, a->ne[2]*a->ne[3]);
    }
    if (t->op == GGML_OP_MUL_MAT_ID) {
        return b->ne[2];
    }
    return 0;
}

static const void * ggml_cuda_graph_get_key_hash(ggml_cgraph * cgraph, const uint32_t big = 0, int32_t * rows = nullptr) {
    uint64_t h = 1469598103934665603ull;   // FNV-1a
    auto mix = [&h](uint64_t v) { h ^= v; h *= 1099511628211ull; };
    // [TAG_FN_L22_GRAPHKEY]
    auto dim = [big](int64_t v) { return big > 0 && v >= (int64_t) big ? (uint64_t) 0x7fffffffffffull : (uint64_t) v; };
    int64_t n_rows = 0;
    mix((uint64_t) cgraph->n_nodes);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * t = cgraph->nodes[i];
        // [TAG_CUDA_GRAPH_KEY_IDENTITY] the name distinguishes one layer's split from the next.
        // Without it every structurally identical per-layer split shares one graph object, their
        // node_props overwrite each other and CUDA graph warmup can never complete. Names survive a
        // graph rebuild, so this does not reintroduce the churn a pointer-based key caused.
        for (const char * c = t->name; *c; ++c) { mix((uint64_t) (unsigned char) *c); }
        mix((uint64_t) t->op); mix((uint64_t) t->type);
        mix(dim(t->ne[0])); mix(dim(t->ne[1])); mix(dim(t->ne[2])); mix(dim(t->ne[3]));
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (t->src[j]) { mix((uint64_t) t->src[j]->type); mix(dim(t->src[j]->ne[0])); mix(dim(t->src[j]->ne[1])); }
        }
        if (rows) {
            n_rows = std::max(n_rows, ggml_cuda_fn_l22_node_rows(t));
        }
    }
    if (h == 0) { h = 1; }
    if (h == (uint64_t) (uintptr_t) GGML_CUDA_FN_L22_EAGER_KEY) { h = 3; } // [TAG_FN_L22_GRAPHKEEP] reserved key
    if (rows) {
        *rows = (int32_t) std::min<int64_t>(n_rows, INT32_MAX);
    }
    return reinterpret_cast<const void *>((uintptr_t) h);
}

// [TAG_GRAPH_KEY_MEMO] The structural key above walks every node and every character of its name, and one
// graph_compute asks for it twice (here and in ggml_cuda_graph_update_required): about 3900 nodes per Qwen3.8-27B
// decode step, all on the host while the GPU waits for the next launch. The scheduler gives each split a new uid when
// it splits a new graph, and llama replays the same splits while its graph reuse holds, so with
// the memo on, the key of a (uid, cgraph, n_nodes) is computed once and then read back. uid 0 (graphs
// that did not come from the scheduler, and the split views passed to graph_optimize) always hashes. The key only
// picks the ggml_cuda_graph object; ggml_cuda_graph_update_required still compares every node property before a
// replay, so a stale key could only cost a re-warm, never a wrong result.
// [TAG_GRAPH_KEY_MEMO_DEFAULT] on by default since 2026-09-28 (host time per graph_compute 449 -> 105 us at 1-token
// decode; with the q8_1 reuse -3.1 % ms/step at 1 stream, -2.2 % at 4, identical KLD and texts).
// GGML_CUDA_GRAPH_KEY_MEMO=0 is the kill switch (hash every time, as before).
static bool ggml_cuda_graph_key_memo_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_GRAPH_KEY_MEMO");
        return e == nullptr || e[0] != '0';
    }();
    return on;
}

// [TAG_FN_GRAPH_KEY_MEMO] 4-way set lookup by uid: O(1) per call with 256 entries (uids are consecutive per graph split)
static constexpr int GGML_CUDA_GRAPH_KEY_MEMO_WAYS = 4;

static ggml_cuda_graph_key_memo * ggml_cuda_graph_key_memo_find(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph) {
    constexpr int n_memo = ggml_backend_cuda_context::GRAPH_KEY_MEMO_N;
    const int base = (int) (cgraph->uid % n_memo);
    for (int w = 0; w < GGML_CUDA_GRAPH_KEY_MEMO_WAYS; ++w) {
        ggml_cuda_graph_key_memo & m = cuda_ctx->graph_key_memo[(base + w) % n_memo];
        if (m.uid == cgraph->uid && m.cgraph == cgraph && m.n_nodes == cgraph->n_nodes && m.key != nullptr) {
            return &m;
        }
    }
    return nullptr;
}

static const void * ggml_cuda_graph_get_key(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int32_t * rows = nullptr) {
    const uint32_t big = cuda_ctx != nullptr ? cuda_ctx->fn_l22_key_big : 0; // [TAG_FN_L22_GRAPHKEY]
    if (cuda_ctx == nullptr || cgraph->uid == 0 || !ggml_cuda_graph_key_memo_enabled()) {
        return ggml_cuda_graph_get_key_hash(cgraph, big, rows);
    }

    if (const ggml_cuda_graph_key_memo * m = ggml_cuda_graph_key_memo_find(cuda_ctx, cgraph)) {
        if (rows) {
            *rows = m->rows;
        }
        return m->key;
    }

    int32_t n_rows = 0;
    const void * key = ggml_cuda_graph_get_key_hash(cgraph, big, &n_rows);
    if (rows) {
        *rows = n_rows;
    }

    // [TAG_FN_GRAPH_KEY_MEMO] replace an empty way, else the oldest (smallest uid) of the set
    constexpr int n_memo = ggml_backend_cuda_context::GRAPH_KEY_MEMO_N;
    const int base = (int) (cgraph->uid % n_memo);
    int victim = base;
    for (int w = 0; w < GGML_CUDA_GRAPH_KEY_MEMO_WAYS; ++w) {
        const int i = (base + w) % n_memo;
        if (cuda_ctx->graph_key_memo[i].key == nullptr) {
            victim = i;
            break;
        }
        if (cuda_ctx->graph_key_memo[i].uid < cuda_ctx->graph_key_memo[victim].uid) {
            victim = i;
        }
    }
    ggml_cuda_graph_key_memo & m = cuda_ctx->graph_key_memo[victim];
    m.uid     = cgraph->uid;
    m.cgraph  = cgraph;
    m.n_nodes = cgraph->n_nodes;
    m.key     = key;
    m.compat  = -1;
    m.rows    = n_rows;

    return key;
}

// [TAG_GRAPH_KEY_MEMO] the compatibility walk only looks at node ops and shapes, which a uid does not change either
static bool ggml_cuda_graph_check_compability_memo(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph) {
    ggml_cuda_graph_key_memo * m = cgraph->uid != 0 && ggml_cuda_graph_key_memo_enabled()
        ? ggml_cuda_graph_key_memo_find(cuda_ctx, cgraph) : nullptr;
    if (m != nullptr && m->compat >= 0) {
        return m->compat != 0;
    }
    const bool compat = ggml_cuda_graph_check_compability(cgraph);
    if (m != nullptr) {
        m->compat = compat ? 1 : 0;
    }
    return compat;
}

// [TAG_FN_L22_GRAPHKEEP] ------------------------------------------------------------------------------------------------
// GGML_CUDA_FN_L22_GRAPHKEEP >= 1: no 10 s sweep; graphs wider than GGML_CUDA_FN_L22_GRAPH_MAXROWS (128) token rows run
// eagerly without a map entry (prompt graphs are one-off shapes: the same kernels, only not captured); an LRU cap on
// objects (GGML_CUDA_FN_L22_GRAPH_CAP "objects,instances", 512,64) never evicts a replayed graph (>= 2 replays) for a
// new key, and the instance cap drops cold instances first. Unset: the sweep and every graph mapped, as before.

static void ggml_cuda_fn_l22_graph_cap(size_t * cap_n, int * cap_i) {
    static const std::pair<size_t, int> caps = [] {
        size_t n = 512;
        int    i = 64;
        if (const char * e = getenv("GGML_CUDA_FN_L22_GRAPH_CAP")) {
            long long a = 0, b = 0;
            const int got = sscanf(e, "%lld,%lld", &a, &b);
            if (got >= 1 && a > 0) {
                n = (size_t) a;
            }
            if (got >= 2 && b > 0) {
                i = (int) b;
            }
        }
        return std::make_pair(std::max<size_t>(n, 8), std::max(i, 2));
    }();
    *cap_n = caps.first;
    *cap_i = caps.second;
}

static int32_t ggml_cuda_fn_l22_graph_maxrows() {
    static const int32_t v = [] {
        const char * e = getenv("GGML_CUDA_FN_L22_GRAPH_MAXROWS");
        const int r = e ? atoi(e) : 128;
        return r > 0 ? (int32_t) r : 128;
    }();
    return v;
}

static bool ggml_cuda_fn_l22_hot(const ggml_cuda_graph & g) {
    return g.has_instance() && g.n_replay >= 2;
}

void ggml_cuda_fn_l22_evict_for_new(ggml_backend_cuda_context * ctx, const void * key) {
    size_t cap_n = 0;
    int    cap_i = 0;
    ggml_cuda_fn_l22_graph_cap(&cap_n, &cap_i);
    while (ctx->cuda_graphs.size() >= cap_n) {
        auto victim = ctx->cuda_graphs.end();
        for (auto it = ctx->cuda_graphs.begin(); it != ctx->cuda_graphs.end(); ++it) {
            if (it->first == key || ggml_cuda_fn_l22_hot(*it->second)) {
                continue;
            }
            if (victim == ctx->cuda_graphs.end() || it->second->last_used_time < victim->second->last_used_time) {
                victim = it;
            }
        }
        if (victim == ctx->cuda_graphs.end()) {
            break; // only replayed graphs: the instance cap bounds them
        }
        ctx->cuda_graphs.erase(victim);
        ctx->fn_l22_cnt[GGML_CUDA_FN_L22_CNT_EVICT_LRU]++;
    }
}

static int ggml_cuda_fn_l22_count_instances(const ggml_backend_cuda_context * ctx) {
    int n = 0;
    for (const auto & [k, g] : ctx->cuda_graphs) {
        n += g->has_instance() ? 1 : 0; // [TAG_UPSTREAM_30804] one per chunk list
        for (const auto & v : g->fn_l22_alt) {
            n += v.has_instance() ? 1 : 0;
        }
    }
    return n;
}

// drop the LRU instance (cold ones first: objects with < 2 replays), never one of keep; false: nothing to drop
static bool ggml_cuda_fn_l22_drop_one_instance(ggml_backend_cuda_context * ctx, const ggml_cuda_graph * keep, bool cold_only) {
    ggml_cuda_graph * best_g = nullptr;
    int               best_v = -1;
    int64_t           best_t = 0;
    bool              best_hot = true;
    for (auto & [k, gp] : ctx->cuda_graphs) {
        ggml_cuda_graph * g = gp.get();
        if (g == keep) {
            continue;
        }
        const auto consider = [&](int v, int64_t t, uint64_t n_replay) {
            const bool hot = n_replay >= 2;
            if (cold_only && hot) {
                return;
            }
            if (best_g == nullptr || (best_hot && !hot) || (best_hot == hot && t < best_t)) {
                best_g = g; best_v = v; best_t = t; best_hot = hot;
            }
        };
        if (g->has_instance()) {
            consider(-1, g->last_used_time, g->n_replay);
        }
        for (int v = 0; v < (int) g->fn_l22_alt.size(); ++v) {
            if (g->fn_l22_alt[v].has_instance()) {
                consider(v, g->fn_l22_alt[v].last_used, g->fn_l22_alt[v].n_replay);
            }
        }
    }
    if (best_g == nullptr) {
        return false;
    }
    if (best_v < 0) {
        best_g->fn_l22_drop_instance();
    } else {
        best_g->fn_l22_alt[best_v].destroy();
        best_g->fn_l22_alt.erase(best_g->fn_l22_alt.begin() + best_v);
    }
    ctx->fn_l22_cnt[best_hot ? GGML_CUDA_FN_L22_CNT_EVICT_INST_HOT : GGML_CUDA_FN_L22_CNT_EVICT_INST_COLD]++;
    return true;
}

// before an instantiate for graph: keep at most cap_i instances
static void ggml_cuda_fn_l22_instance_cap(ggml_backend_cuda_context * ctx, const ggml_cuda_graph * graph) {
    if (ggml_cuda_fn_l22_graph_keep() < 1) {
        return;
    }
    size_t cap_n = 0;
    int    cap_i = 0;
    ggml_cuda_fn_l22_graph_cap(&cap_n, &cap_i);
    int n = ggml_cuda_fn_l22_count_instances(ctx);
    while (n >= cap_i && ggml_cuda_fn_l22_drop_one_instance(ctx, graph, false)) {
        n--;
    }
}

// cudaGraphInstantiate of one chunk of graph [TAG_UPSTREAM_30804] with the instance cap first (for the first chunk of an
// object without instances: the cap counts chunk lists) and, on an out-of-memory error, the cold instances dropped and the
// OOM hook asked once before one retry [TAG_FN_L22_OOMHOOK]. The OOM hook runs only outside a capture (the last chunk).
static void ggml_cuda_fn_l22_instantiate(ggml_backend_cuda_context * ctx, ggml_cuda_graph * graph, cudaGraphExec_t * instance,
                                         cudaGraph_t cuda_graph) {
    if (!graph->has_instance()) {
        ggml_cuda_fn_l22_instance_cap(ctx, graph);
    }
    ggml_cuda_fn_l22_oom_precheck(64u*1024*1024); // [TAG_FN_L22_OOMHOOK] an instance of a large graph: tens of MiB
    const auto inst = [&]() {
        if (ctx->fn_l4_prio) { // [TAG_FN_L4_QSA_STREAMS]
            return cudaGraphInstantiateWithFlags(instance, cuda_graph, cudaGraphInstantiateFlagUseNodePriority);
        }
        return cudaGraphInstantiate(instance, cuda_graph, NULL, NULL, 0);
    };
    cudaError_t err = inst();
    if (err == cudaErrorMemoryAllocation) {
        (void) cudaGetLastError();
        *instance = nullptr;
        int n_dropped = 0;
        while (ggml_cuda_fn_l22_drop_one_instance(ctx, graph, true)) {
            n_dropped++;
        }
        const bool hook = ggml_cuda_fn_l22_oom_retry(256u*1024*1024, 3);
        GGML_LOG_WARN("ggml_cuda: [TAG_FN_L22_OOMHOOK] graph instantiate out of memory (%d nodes): %d cold instances dropped, "
                "hook %d, one retry\n", (int) graph->node_props.size(), n_dropped, hook ? 1 : 0);
        err = inst();
        // [TAG_FN_L23_PINFIX] R-L23-6: a second failure aborts below: the hook's second attempt first
        if (err == cudaErrorMemoryAllocation && ggml_cuda_fn_l22_oom_retry(256u*1024*1024, 4)) {
            (void) cudaGetLastError();
            *instance = nullptr; // [TAG_FN_V0252_MERGE] the chunk's instance ([TAG_UPSTREAM_30804] has no graph->instance)
            err = inst();
        }
    }
    CUDA_CHECK(err);
}

static void ggml_cuda_fn_l22_cap_begin(ggml_backend_cuda_context * ctx) {
    for (int s = 0; s < GGML_CUDA_MAX_STREAMS; ++s) {
        if (ctx->pools[ctx->device][s]) {
            ctx->pools[ctx->device][s]->cap_begin();
        }
    }
}

static void ggml_cuda_fn_l22_cap_end(ggml_backend_cuda_context * ctx, ggml_cuda_graph * graph) {
    bool known = true;
    for (int s = 0; s < GGML_CUDA_MAX_STREAMS; ++s) {
        size_t top = 0;
        if (ctx->pools[ctx->device][s]) {
            top = ctx->pools[ctx->device][s]->cap_hwm();
            known = known && top != (size_t) -1;
        }
        graph->pool_top[s] = top;
    }
    for (int d = 0; d < GGML_CUDA_MAX_DEVICES && known; ++d) {
        for (int s = 0; s < GGML_CUDA_MAX_STREAMS && d != ctx->device; ++s) {
            known = known && ctx->pools[d][s] == nullptr; // another device's pool: not tracked
        }
    }
    graph->pool_top_known = known;
    graph->n_replay_cap   = 0;
}

// one "[TAG_FN_L22_PASSLOG] kind=graph" line per window (PASSLOG format; ggml cannot call libllama's helper)
// [TAG_FN_L22_PASSLOG] R-REV-11: libllama's resolved LLAMA_FN_L22_PASSLOG (proc "ggml_backend_cuda_set_passlog_level");
// -1 = never set (the lines print as before)
static std::atomic<int> ggml_cuda_fn_l22_passlog{-1};

static void ggml_backend_cuda_set_passlog_level(int level) {
    ggml_cuda_fn_l22_passlog.store(level, std::memory_order_relaxed);
}

static bool ggml_cuda_fn_l22_passlog_on(void) {
    const int v = ggml_cuda_fn_l22_passlog.load(std::memory_order_relaxed);
    return v < 0 || v >= 1;
}

static void ggml_cuda_fn_l22_window_start(ggml_backend_cuda_context * ctx, int n) {
    ctx->fn_l22_win_left = n;
    ctx->fn_l22_win_len  = n;
    memcpy(ctx->fn_l22_win_base, ctx->fn_l22_cnt, sizeof(ctx->fn_l22_cnt));
}

static void ggml_cuda_fn_l22_window_tick(ggml_backend_cuda_context * ctx) {
    if (ctx->fn_l22_win_left <= 0 || --ctx->fn_l22_win_left > 0) {
        return;
    }
    if (!ggml_cuda_fn_l22_passlog_on()) {
        return;
    }
    const uint64_t * c = ctx->fn_l22_cnt;
    const uint64_t * b = ctx->fn_l22_win_base;
    GGML_LOG_WARN("[TAG_FN_L22_PASSLOG] kind=graph ctx=%s %p first %d calls: replay %" PRIu64 " recap %" PRIu64 " capture %" PRIu64
            " warmup %" PRIu64 " eager %" PRIu64 " host %.1f ms; map %zu objects, %d instances; evict lru %" PRIu64 " inst cold %" PRIu64
            " hot %" PRIu64 ", variant swaps %" PRIu64 "\n",
            ctx->name.c_str(), (void *) ctx, ctx->fn_l22_win_len,
            c[GGML_CUDA_FN_L22_CNT_REPLAY] - b[GGML_CUDA_FN_L22_CNT_REPLAY], c[GGML_CUDA_FN_L22_CNT_RECAP] - b[GGML_CUDA_FN_L22_CNT_RECAP],
            c[GGML_CUDA_FN_L22_CNT_CAPTURE] - b[GGML_CUDA_FN_L22_CNT_CAPTURE], c[GGML_CUDA_FN_L22_CNT_WARMUP] - b[GGML_CUDA_FN_L22_CNT_WARMUP],
            c[GGML_CUDA_FN_L22_CNT_EAGER_ROWS] - b[GGML_CUDA_FN_L22_CNT_EAGER_ROWS],
            (c[GGML_CUDA_FN_L22_CNT_HOST_US] - b[GGML_CUDA_FN_L22_CNT_HOST_US])/1000.0,
            ctx->cuda_graphs.size(), ggml_cuda_fn_l22_count_instances(ctx),
            c[GGML_CUDA_FN_L22_CNT_EVICT_LRU], c[GGML_CUDA_FN_L22_CNT_EVICT_INST_COLD], c[GGML_CUDA_FN_L22_CNT_EVICT_INST_HOT],
            c[GGML_CUDA_FN_L22_CNT_VARIANT_SWAP]);
}

static bool ggml_cuda_fn_l22_props_equal(const std::vector<ggml_cuda_graph::node_properties> & a,
                                         const std::vector<ggml_cuda_graph::node_properties> & b) {
    return a.size() == b.size() && (a.empty() || memcmp(a.data(), b.data(), a.size()*sizeof(a[0])) == 0);
}

static bool ggml_cuda_graph_update_required(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const void * graph_key, bool * nodes_changed) {
    bool res = false;
    *nodes_changed = false; // [TAG_FN_L21_RECAP]

    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

    if (cgraph->uid != 0 &&
        cgraph->uid == graph->uid) {
        if (ggml_cuda_graph_debug_enabled()) {
            GGML_LOG_DEBUG("CUDA Graph id %zu reused\n", cgraph->uid);
        }
        GGML_ASSERT((int)graph->node_props.size() == cgraph->n_nodes);
        return false;
    }

    graph->uid = cgraph->uid;

    // [TAG_FN_L22_GRAPH2] an instance that replayed a few times and is about to change goes aside with its properties: a key
    // that alternates between two buffer layouts then swaps instances instead of re-warming at every switch. A graph
    // that replayed long (a KV bucket change) keeps the RECAP update in place.
    const bool g2 = ggml_cuda_fn_l22_graph2() && graph->has_instance() && graph->n_replay_cap >= 1 &&
        graph->n_replay_cap <= 32 && (int) graph->node_props.size() == cgraph->n_nodes;
    std::vector<ggml_cuda_graph::node_properties> g2_old;

    // Check if the graph size has changed
    if ((int)graph->node_props.size() != cgraph->n_nodes) {
        res = true;
        *nodes_changed = true; // [TAG_FN_L21_RECAP]
        graph->node_props.resize(cgraph->n_nodes);
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_cuda_graph::node_properties prop = {};
        memcpy(&prop.node, cgraph->nodes[i], sizeof(ggml_tensor));

        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (cgraph->nodes[i]->src[j]) {
                prop.node_src_data_ptrs[j] = cgraph->nodes[i]->src[j]->data;
                memcpy(prop.node_src_ne[j], cgraph->nodes[i]->src[j]->ne, sizeof(prop.node_src_ne[j]));
                memcpy(prop.node_src_nb[j], cgraph->nodes[i]->src[j]->nb, sizeof(prop.node_src_nb[j]));
            }
        }

        if (res || memcmp(&graph->node_props[i], &prop, sizeof(prop)) != 0) {
            if (g2 && !res) {
                g2_old = graph->node_props; // [TAG_FN_L22_GRAPH2] the layout the instance was captured for
            }
            // [TAG_FN_L7_GDIFF] TURBO_GRAPH_DIFF=1 (or GGML_CUDA_FN_L22_GRAPHDIAG=1): the first node whose properties
            // changed, and what changed (diagnostic)
            static const bool gdiff = [] {
                const char * e = getenv("TURBO_GRAPH_DIFF");
                return (e != nullptr && e[0] == '1') || ggml_cuda_fn_l22_graph_diag();
            }();
            if (gdiff && !res && (int) graph->node_props.size() == cgraph->n_nodes) {
                static std::atomic<int> n_logged{0};
                if (n_logged.fetch_add(1) < 400) {
                    const ggml_tensor & a = graph->node_props[i].node;
                    const ggml_tensor & b = prop.node;
                    char what[256];
                    int  w = 0;
                    what[0] = 0;
                    if (memcmp(a.ne, b.ne, sizeof(a.ne)) != 0) { w += snprintf(what + w, sizeof(what) - w, " ne(%lld,%lld,%lld,%lld->%lld,%lld,%lld,%lld)", (long long) a.ne[0], (long long) a.ne[1], (long long) a.ne[2], (long long) a.ne[3], (long long) b.ne[0], (long long) b.ne[1], (long long) b.ne[2], (long long) b.ne[3]); }
                    if (memcmp(a.nb, b.nb, sizeof(a.nb)) != 0) { w += snprintf(what + w, sizeof(what) - w, " nb"); }
                    if (a.data != b.data)                       { w += snprintf(what + w, sizeof(what) - w, " data"); }
                    if (memcmp(a.op_params, b.op_params, sizeof(a.op_params)) != 0) { w += snprintf(what + w, sizeof(what) - w, " op_params"); }
                    for (int j = 0; j < GGML_MAX_SRC && w < (int) sizeof(what) - 32; ++j) {
                        if (graph->node_props[i].node_src_data_ptrs[j] != prop.node_src_data_ptrs[j]) { w += snprintf(what + w, sizeof(what) - w, " src%d.data(%s)", j, b.src[j] ? b.src[j]->name : "-"); }
                        if (memcmp(graph->node_props[i].node_src_ne[j], prop.node_src_ne[j], sizeof(prop.node_src_ne[j])) != 0) { w += snprintf(what + w, sizeof(what) - w, " src%d.ne", j); }
                    }
                    if (w == 0) { snprintf(what, sizeof(what), " other fields"); }
                    fprintf(stderr, "turbo-probe: graph-diff ctx %p key %p nodes %d: node %d '%s' %s:%s\n", (void *) cuda_ctx,
                            graph_key, cgraph->n_nodes, i, b.name, ggml_op_desc(&b), what);
                }
            }
            // [TAG_FN_L21_RECAP] a new op, or a node that starts or stops running a kernel, changes the CUDA graph's topology
            *nodes_changed = *nodes_changed || graph->node_props[i].node.op != prop.node.op ||
                ggml_cuda_is_view_or_noop(&graph->node_props[i].node) != ggml_cuda_is_view_or_noop(&prop.node);
            graph->node_props[i] = prop;
            res = true;
        }
    }

    // [TAG_FN_L22_GRAPH2] swap in a kept instance of this layout, else put the current one aside (at most 2 kept)
    if (res && !g2_old.empty()) {
        ggml_cuda_graph::fn_l22_variant cur;
        cur.instances      = std::move(graph->instances); // [TAG_UPSTREAM_30804] the chunk list
        graph->instances.clear();
        cur.node_props     = std::move(g2_old);
        cur.pool_top_known = graph->pool_top_known;
        memcpy(cur.pool_top, graph->pool_top, sizeof(cur.pool_top));
        cur.n_replay       = graph->n_replay;
        cur.last_used      = ggml_time_us();

        for (ggml_cuda_graph::fn_l22_variant & v : graph->fn_l22_alt) {
            if (ggml_cuda_fn_l22_props_equal(v.node_props, graph->node_props)) {
                graph->instances      = std::move(v.instances);
                graph->pool_top_known = v.pool_top_known;
                memcpy(graph->pool_top, v.pool_top, sizeof(graph->pool_top));
                graph->n_replay_cap    = 1;
                graph->warmup_complete = true;
                v = std::move(cur);
                *nodes_changed = false;
                cuda_ctx->fn_l22_cnt[GGML_CUDA_FN_L22_CNT_VARIANT_SWAP]++;
                return false; // the swapped-in instance was captured from exactly these properties
            }
        }
        graph->fn_l22_alt.push_back(std::move(cur));
        if (graph->fn_l22_alt.size() > 2) {
            size_t oldest = 0;
            for (size_t v = 1; v < graph->fn_l22_alt.size(); ++v) {
                if (graph->fn_l22_alt[v].last_used < graph->fn_l22_alt[oldest].last_used) {
                    oldest = v;
                }
            }
            graph->fn_l22_alt[oldest].destroy();
            graph->fn_l22_alt.erase(graph->fn_l22_alt.begin() + oldest);
        }
        graph->instances.clear();
        graph->replayed_since_capture = false;
        graph->n_replay_cap           = 0;
        graph->pool_top_known         = false;
    }

    return res;
}

// [TAG_UPSTREAM_30804] update one chunk's executable from its new capture (re-instantiate when the topology changed)
static void ggml_cuda_graph_update_executable(ggml_backend_cuda_context * cuda_ctx, ggml_cuda_graph * graph,
                                              cudaGraphExec_t & instance, cudaGraph_t cuda_graph) {
#if CUDART_VERSION >= 12000
    cudaGraphExecUpdateResultInfo result_info;
    cudaError_t stat = cudaGraphExecUpdate(instance, cuda_graph, &result_info);
#else
    cudaGraphNode_t errorNode;
    cudaGraphExecUpdateResult result_info;
    cudaError_t stat = cudaGraphExecUpdate(instance, cuda_graph, &errorNode, &result_info);
#endif // CUDART_VERSION >= 12000

    if (stat == cudaErrorGraphExecUpdateFailure) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: CUDA graph update failed\n", __func__);
#endif

        // The pre-existing graph exec cannot be updated due to violated constraints
        // so instead clear error and re-instantiate
        (void)cudaGetLastError();
        CUDA_CHECK(cudaGraphExecDestroy(instance));
        instance = nullptr;
        ggml_cuda_fn_l22_instantiate(cuda_ctx, graph, &instance, cuda_graph); // [TAG_FN_L22_GRAPHKEEP] OOM hook
    } else {
        GGML_ASSERT(stat == cudaSuccess);
    }
}

// [TAG_UPSTREAM_30804] (#30804 + #31053) ggml nodes per CUDA graph chunk; 0 captures the whole ggml graph as one CUDA graph
static int ggml_cuda_graph_chunk_size() {
    static const int chunk_size = [] {
        const char * env = getenv("GGML_CUDA_GRAPH_CHUNK_SIZE");
        return env != nullptr ? atoi(env) : 512;
    }();
    return chunk_size;
}

// [TAG_UPSTREAM_30804] chunk idx of graph was captured into cuda_graph: update its executable (or create it) and launch it
static void ggml_cuda_graph_exec_chunk(ggml_backend_cuda_context * cuda_ctx, ggml_cuda_graph * graph, size_t idx,
                                       cudaGraph_t cuda_graph) {
    if (idx < graph->instances.size()) {
        ggml_cuda_graph_update_executable(cuda_ctx, graph, graph->instances[idx], cuda_graph);
    } else {
        GGML_ASSERT(idx == graph->instances.size());
        cudaGraphExec_t instance = nullptr;
        ggml_cuda_fn_l22_instantiate(cuda_ctx, graph, &instance, cuda_graph); // [TAG_FN_L22_GRAPHKEEP] cap, OOM hook
        graph->instances.push_back(instance);
    }
    CUDA_CHECK(cudaGraphDestroy(cuda_graph));
    CUDA_CHECK(cudaGraphLaunch(graph->instances[idx], cuda_ctx->stream()));
}

// [TAG_FN_R4_POKE] GGML_CUDA_GRAPH_POKE=1: one stream query right after a launch flushes WDDM's batched submission, so the
// GPU starts the graph now rather than at the next driver call, and host work that should overlap it (a CPU split, the
// MoE bridge's executor, the capture of the next chunk) does. Strata R4.2d measured 17 of 10,562 layers overlapped
// without it and 9,190 with it. The query changes no result; "not ready" is its normal answer here. Never inside a capture.
static void ggml_cuda_graph_poke(ggml_backend_cuda_context * cuda_ctx) {
    static const bool graph_poke = [] {
        const char * e = getenv("GGML_CUDA_GRAPH_POKE");
        return e != nullptr && atoi(e) != 0;
    }();
    if (graph_poke) {
        const cudaError_t q = cudaStreamQuery(cuda_ctx->stream());
        if (q == cudaErrorNotReady) {
            (void) cudaGetLastError(); // not an error: clear it, as torch's stream query does
        } else if (q != cudaSuccess) {
            CUDA_CHECK(q);
        }
    }
}
#endif // USE_CUDA_GRAPH

static bool ggml_cuda_should_fuse_rope_set_rows(const ggml_tensor * rope,
                                                const ggml_tensor * view,
                                                const ggml_tensor * set_rows) {

    if (rope->op != GGML_OP_ROPE || view->op != GGML_OP_VIEW || set_rows->op != GGML_OP_SET_ROWS) {
        return false;
    }
    // ne3 not tested
    if (rope->src[0]->ne[3] != 1) {
        return false;
    }

    if (set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16) {
        return false;
    }

    if (set_rows->src[1]->type != GGML_TYPE_I64) {
        return false;
    }

    // The view should flatten two dims of rope into one dim
    if (!ggml_is_contiguous(view) || view->ne[0] != rope->ne[0] * rope->ne[1]) {
        return false;
    }

    // Only norm/neox/multi shaders have the fusion code
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX &&
        mode != GGML_ROPE_TYPE_MROPE && mode != GGML_ROPE_TYPE_IMROPE) {
        return false;
    }

    return true;
}

static bool ggml_cuda_should_fuse_rms_norm_mul_rope(const ggml_tensor * rms_norm,
                                                    const ggml_tensor * mul,
                                                    const ggml_tensor * rope) {
    if (rms_norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || rope->op != GGML_OP_ROPE) {
        return false;
    }

    if (rms_norm->src[0]->type != GGML_TYPE_F32 || rms_norm->type != GGML_TYPE_F32 ||
        mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32 ||
        mul->type != GGML_TYPE_F32 || rope->type != GGML_TYPE_F32) {
        return false;
    }

    if (rope->src[0] != mul) {
        return false;
    }

    //if rms norm is the B operand, then we don't handle broadcast
    if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
        return false;
    }

    if (!ggml_are_same_shape(rms_norm, mul)) {
        return false;
    }

    //rms_norm kernel assumes contiguous rows
    if (!ggml_is_contiguous_rows(rms_norm->src[0]) ||
        !ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
        return false;
    }

    // the fused kernel handles the norm/neox rope modes only
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
        return false;
    }

    const int n_dims = ((const int32_t *) rope->op_params)[1];
    if (n_dims % 2 != 0 || rope->src[0]->ne[0] % 2 != 0) {
        return false;
    }

    // ggml_rope_set_offset is not yet supported in the fused kernel
    const int n_offs = ((const int32_t *) rope->op_params)[15];
    if (n_offs != 0) {
        return false;
    }

    return true;
}

// match gated_delta_net + the strided cpy that scatters its state snapshots into the cache
// (slot i -> rollback group i, slot 0 newest), so the kernel can write them and skip the cpy.
static int ggml_cuda_try_gdn_cache_fusion(
        const ggml_cgraph * cgraph, int node_idx, ggml_cuda_gated_delta_net_fused_cache & fused_state_cpy) {
    const ggml_tensor * gdn = cgraph->nodes[node_idx];
    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    if (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 ||
        (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }

    // Chunked prefill writes state to dst; it cannot scatter into the snapshot cache, so skip fusion.
    // Decode and fallback recurrent (T < 128) do fuse.
    if (ggml_cuda_gdn_op_is_chunked(gdn)) {
        return 0;
    }

    const ggml_tensor * src_v     = gdn->src[2];
    const int64_t       S_v       = src_v->ne[0];
    const int64_t       H         = src_v->ne[1];
    const int64_t       n_tokens  = src_v->ne[2];
    const int64_t       n_seqs    = src_v->ne[3];
    const int64_t       D         = S_v * S_v * H;
    const int64_t       K         = ggml_get_op_params_i32(gdn, 0); // snapshot slot count
    const int64_t       n_written = std::min<int64_t>(n_tokens, K); // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    const size_t tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    // snapshot cpy is the first real node after the gdn (skip views/no-ops)
    const ggml_tensor * cpy  = nullptr;
    int                 skip = 0;
    for (int j = node_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        cpy  = n;
        skip = j - node_idx;
    }
    if (cpy == nullptr) {
        return 0;
    }

    const ggml_tensor * src = cpy->src[0]; // view of the gdn snapshot tail
    const ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != tail_off ||
        !ggml_is_contiguous(src)) {
        return 0;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D (the per-seq stride the kernel
    // assumes). ggml_cpy pins src to the same element count.
    const std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    if (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) || dst->nb[1] != (size_t) ggml_row_size(GGML_TYPE_F32, D)) {
        return 0;
    }

    fused_state_cpy.data        = (float *) dst->data; // rollback group 0 (newest)
    fused_state_cpy.slot_stride = K > 1 ? (int64_t) (dst->nb[2] / sizeof(float)) : 0;
    return skip;
}

// [TAG_4C_GDN_REPLAY] match gated_delta_net_replay + the cpy of its committed states into the cache + the cpy of its new
// rings into the ring cache, so the kernel writes both and the two cpys are skipped
static int ggml_cuda_try_gdn_replay_fusion(const ggml_cgraph * cgraph, int node_idx, ggml_cuda_gdn_replay_out & out) {
    const ggml_tensor * gdn = cgraph->nodes[node_idx];
    // the kernel skips the state and ring tails, so the op output must not be a graph output
    if (gdn->op != GGML_OP_GATED_DELTA_NET_REPLAY || gdn->type != GGML_TYPE_F32 ||
        (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }

    const ggml_tensor * src_v    = gdn->src[2];
    const int64_t       S_v      = src_v->ne[0];
    const int64_t       H        = src_v->ne[1];
    const int64_t       n_tokens = src_v->ne[2];
    const int64_t       n_seqs   = src_v->ne[3];
    const int64_t       D        = S_v * S_v * H;
    const int64_t       ring_row = gdn->src[6]->ne[0];

    const size_t  offs[2] = {
        ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs),
        ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs + D * n_seqs),
    };
    const int64_t rows[2] = { D, ring_row };

    // the state cpy and then the ring cpy are the next two real nodes (skip views/no-ops)
    const ggml_tensor * cpys[2] = { nullptr, nullptr };
    int n_cpy = 0;
    int skip  = 0;
    for (int j = node_idx + 1; j < cgraph->n_nodes && n_cpy < 2; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        cpys[n_cpy++] = n;
        skip = j - node_idx;
    }
    if (n_cpy < 2) {
        return 0;
    }

    float * ptrs[2] = { nullptr, nullptr };
    for (int c = 0; c < 2; ++c) {
        const ggml_tensor * src = cpys[c]->src[0]; // view of the op output tail
        const ggml_tensor * dst = cpys[c]->src[1]; // cache rows [head, head + n_seqs)
        if (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != offs[c] || !ggml_is_contiguous(src) ||
            src->ne[0] != rows[c] || src->ne[1] != n_seqs) {
            return 0;
        }
        if (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
            dst->ne[0] != rows[c] || dst->ne[1] != n_seqs || dst->ne[2] != 1 || dst->ne[3] != 1 ||
            dst->nb[0] != ggml_type_size(GGML_TYPE_F32) || dst->nb[1] != (size_t) ggml_row_size(GGML_TYPE_F32, rows[c])) {
            return 0;
        }
        ptrs[c] = (float *) dst->data;
    }

    out.state = ptrs[0];
    out.ring  = ptrs[1];
    return skip;
}

// match ssm_scan + the strided cpy that scatters its state snapshots into the cache, so the kernel writes them and skips the cpy
static int ggml_cuda_try_ssm_scan_cache_fusion(
        const ggml_cgraph * cgraph, int node_idx, ggml_cuda_ssm_scan_fused_cache & fused_state_cpy) {
    const ggml_tensor * ssm = cgraph->nodes[node_idx];
    // the kernel skips the snapshot tail, so the scan output must not be a graph output
    if (ssm->op != GGML_OP_SSM_SCAN || ssm->type != GGML_TYPE_F32 || (ssm->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }

    const int64_t K = ggml_get_op_params_i32(ssm, 0); // snapshot slot count

    const ggml_tensor * s = ssm->src[0];
    const ggml_tensor * x = ssm->src[1];
    const ggml_tensor * A = ssm->src[3];

    const int64_t d_state = s->ne[0];
    const int64_t D       = d_state * s->ne[1] * x->ne[1]; // d_state * head_dim * n_head
    const int64_t n_tok   = x->ne[2];
    const int64_t n_seqs  = x->ne[3];

    // only the mamba-2 kernels (group scan and SSD) write to the cache; mamba-1 still uses the cpy
    if (A->nb[1] != sizeof(float) || (d_state != 96 && d_state != 128 && d_state != 256)) {
        return 0;
    }

    // the scan reads its input rows from the cache (picked by ids), so with more than one seq a seq can read a row that another seq writes in the same launch
    if (n_seqs != 1) {
        return 0;
    }

    const int64_t n_written = std::min<int64_t>(n_tok, K);
    const size_t  tail_off  = ggml_row_size(GGML_TYPE_F32, ggml_nelements(x));

    // snapshot cpy is the first real node after the scan (skip views/no-ops)
    const ggml_tensor * cpy  = nullptr;
    int                 skip = 0;
    for (int j = node_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        cpy  = n;
        skip = j - node_idx;
    }
    if (cpy == nullptr) {
        return 0;
    }

    const ggml_tensor * src = cpy->src[0]; // view of the scan snapshot tail
    const ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this scan's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != ssm || src->view_offs != tail_off ||
        !ggml_is_contiguous(src)) {
        return 0;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D, the per-seq stride the kernel takes from src0->nb[3]
    const std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    if (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) || dst->nb[1] != (size_t) ggml_row_size(GGML_TYPE_F32, D)) {
        return 0;
    }

    fused_state_cpy.data        = (float *) dst->data; // rollback slot 0 (newest)
    fused_state_cpy.slot_stride = K > 1 ? (int64_t) (dst->nb[2] / sizeof(float)) : 0;
    return skip;
}

// expert group selection from build_moe_ffn (n_expert_groups > 1)
// src0/src1 index into this table, -1 is the (biased) probs, -2 is not checked
static const struct {
    ggml_op op;
    int     src0;
    int     src1;
} topk_moe_group_pattern[] = {
    { GGML_OP_RESHAPE,  -1, -2 }, //  0: [n_exp_per_group, n_expert_groups, n_tokens]
    { GGML_OP_RESHAPE,   0, -2 }, //  1: [1, n_exp_per_group, n_expert_groups, n_tokens]
    { GGML_OP_ARGSORT,   0, -2 }, //  2
    { GGML_OP_VIEW,      2, -2 }, //  3: top-2 per group
    { GGML_OP_GET_ROWS,  1,  3 }, //  4
    { GGML_OP_RESHAPE,   4, -2 }, //  5
    { GGML_OP_SUM_ROWS,  5, -2 }, //  6: group scores
    { GGML_OP_RESHAPE,   6, -2 }, //  7
    { GGML_OP_ARGSORT,   7, -2 }, //  8
    { GGML_OP_VIEW,      8, -2 }, //  9: top n_group_used groups
    { GGML_OP_GET_ROWS,  0,  9 }, // 10
    { GGML_OP_FILL,      0, -2 }, // 11
    { GGML_OP_SET_ROWS, 10,  9 }, // 12: unused groups -> -inf
    { GGML_OP_RESHAPE,  12, -2 }, // 13: [n_expert, n_tokens]
};

static bool ggml_cuda_topk_moe_group_fusion(const struct ggml_cgraph * cgraph, int node_idx, const ggml_tensor * selection, ggml_cuda_topk_moe_args & args) {
#if defined(GGML_USE_HIP) && HIP_VERSION < 60300000
    GGML_UNUSED_VARS(cgraph, node_idx, selection, args);
    return false;
#else
    const int n_ops = (int) std::size(topk_moe_group_pattern);
    if (node_idx + n_ops > cgraph->n_nodes) {
        return false;
    }

    ggml_tensor ** nodes = cgraph->nodes + node_idx;
    auto src_ok = [&](const ggml_tensor * src, int idx) {
        return idx == -2 || src == (idx == -1 ? selection : nodes[idx]);
    };

    for (int i = 0; i < n_ops; i++) {
        const auto & p = topk_moe_group_pattern[i];
        if (nodes[i]->op != p.op || !src_ok(nodes[i]->src[0], p.src0) || !src_ok(nodes[i]->src[1], p.src1)) {
            return false;
        }
    }

    const int n_groups     = nodes[0]->ne[1];
    const int n_group_used = nodes[9]->ne[0];

    if (nodes[3]->ne[0] != 2 || nodes[12]->src[2] != nodes[11] || ggml_get_op_params_f32(nodes[11], 0) != -INFINITY ||
            n_groups > WARP_SIZE) {
        return false;
    }

    args.n_groups     = n_groups;
    args.n_group_used = n_group_used;
    return true;
#endif // defined(GGML_USE_HIP) && HIP_VERSION < 60300000
}

static bool ggml_cuda_topk_moe_fusion(const struct ggml_cgraph * cgraph, int node_idx, ggml_cuda_topk_moe_args & args) {
    args.sigmoid         = false;
    args.sqrt_softplus   = false;
    args.softmax         = false;
    args.delayed_softmax = false;
    args.prob_bias       = false;
    args.norm            = false;
    args.n_groups        = 0;
    args.n_group_used    = 0;

    const int      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    if (nodes[node_idx]->op == GGML_OP_SOFT_MAX) {
        args.softmax = true;
    }

    if (nodes[node_idx]->op == GGML_OP_UNARY) {
        const ggml_unary_op unary_op = ggml_get_unary_op(nodes[node_idx]);
        if (unary_op == GGML_UNARY_OP_SIGMOID) {
            args.sigmoid = true;
        } else if (unary_op == GGML_UNARY_OP_SOFTPLUS && node_idx + 1 < n_nodes &&
                   nodes[node_idx + 1]->op == GGML_OP_SQRT && nodes[node_idx + 1]->src[0] == nodes[node_idx]) {
            // sqrt(softplus(x)) scoring (DeepSeek-V4)
            args.sqrt_softplus = true;
            node_idx++;
        } else {
            return false;
        }
    }

    if (nodes[node_idx]->op == GGML_OP_ARGSORT) {
        args.delayed_softmax = true;
    }

    node_idx++;

    if (args.sigmoid || args.sqrt_softplus || args.softmax) {
        // SOFTMAX -> RESHAPE
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_RESHAPE ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx];
        node_idx++;

        if (node_idx >= n_nodes) {
            return false;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        const ggml_tensor * selection = nodes[node_idx - 2];
        if (nodes[node_idx]->op == GGML_OP_ADD && nodes[node_idx]->src[0] == selection) {
            args.prob_bias = true;
            selection = nodes[node_idx];
            node_idx++;
        }

        if (ggml_cuda_topk_moe_group_fusion(cgraph, node_idx, selection, args)) {
            node_idx += (int) std::size(topk_moe_group_pattern);
            selection = nodes[node_idx - 1];
        }

        // RESHAPE/ADD/group mask -> ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_ARGSORT || nodes[node_idx]->src[0] != selection) {
            return false;
        }

        node_idx++;

        // ARGSORT-> VIEW
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }

        // kept groups must hold all used experts
        if (args.n_groups > 0 && nodes[node_idx]->ne[0] > args.n_group_used * (probs_reshaped->ne[1] / args.n_groups)) {
            return false;
        }
        node_idx++;

        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_GET_ROWS) {
            return false;
        }

        // GET_ROWS
        if (nodes[node_idx]->src[0] != probs_reshaped || nodes[node_idx]->src[1] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;
    } else if (args.delayed_softmax) {
        if (node_idx - 2 < 0) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx - 2];

        // VIEW->ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
            nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;

        // GET_ROWS
        if (node_idx >= n_nodes || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
                nodes[node_idx]->src[0] != probs_reshaped) {
            return false;
        }
        node_idx++;

        static const std::vector<ggml_op> remaining_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        for (const ggml_op op : remaining_ops) {
            if (node_idx >= n_nodes || nodes[node_idx]->op != op || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
                return false;
            }
            node_idx++;
        }
    }

    // At this point we can check for norm + scale. Everything is now at least valid till the norm
    if (node_idx >= n_nodes) {
        return true;
    }

    if (nodes[node_idx]->op == GGML_OP_RESHAPE) {
        //check RESHAPE->SUM_ROWS->CLAMP->DIV->RESHAPE
        static const std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP };

        args.norm = true;
        for (const ggml_op op : norm_ops) {
            if (nodes[node_idx]->op == op && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
                node_idx++;
            } else {
                args.norm = false;
                return true;
            }
        }

        // DIV <- CLAMP, RESHAPE
        if (nodes[node_idx]->op != GGML_OP_DIV || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
            nodes[node_idx]->src[0] != nodes[node_idx - 3]) {
            args.norm = false;
            return true;
        }
        node_idx++;

        if (nodes[node_idx]->op != GGML_OP_RESHAPE || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            args.norm = false;
            return true;
        }

        node_idx++;
    }

    if (nodes[node_idx]->op == GGML_OP_SCALE && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
        args.scale = true;
    }

    return true;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
static bool ggml_cuda_check_fusion_memory_ranges(const ggml_cgraph * cgraph,
                                                 const int           node_idx,
                                                 const int           node_count,
                                                 const int *         out_nodes,
                                                 const int           out_count,
                                                 const bool          is_topk_moe = false) {
    auto nodes_overlap = [&](const ggml_tensor * a, const ggml_tensor * b) {
        const int64_t a_start = (int64_t) a->data;
        const int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);

        const int64_t b_start = (int64_t) b->data;
        const int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        if ((b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end)) {
            return true;
        }

        return false;
    };

    bool is_ok = true;
    // one block reads all logits before it writes, so logits may alias the out nodes
    const ggml_tensor * logits_may_alias = nullptr;
    if (is_topk_moe && ggml_nrows(cgraph->nodes[node_idx]) <= TOPK_MOE_ROWS_PER_BLOCK) {
        logits_may_alias = cgraph->nodes[node_idx]->src[0];
    }

    for (int i = 0; i < out_count; ++i) {
        const ggml_tensor * dst = cgraph->nodes[out_nodes[i]];

        for (int j = node_idx; j < node_idx + node_count; ++j) {
            // Loop over all srcs of all nodes in the fusion. If the src overlaps
            // the destination and the src is not an intermediate node that's being
            // elided, then disable fusion.

            for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                const ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                if (!src || src->op == GGML_OP_NONE || src == logits_may_alias) {
                    continue;
                }

                if (nodes_overlap(dst, src)) {
                    bool found = false;

                    for (int k = node_idx; k < j; ++k) {
                        if (cgraph->nodes[k] == src) {
                            found = true;
                            break;
                        }
                    }

                    if (!found) {
                        is_ok = false;
                        break;
                    }
                }
            }
        }
    }

    return is_ok;
}

// The long form spans 2*k + 1 nodes. ggml_can_fuse_subgraph() accepts at most
// 31 nodes, so k <= 15; larger values use the per-operation path.
static constexpr int MOE_WEIGHTED_REDUCTION_MAX_EXPERTS = 15;

struct ggml_cuda_moe_weighted_reduction_match {
    const ggml_tensor * experts      = nullptr;
    const ggml_tensor * expert_scale = nullptr;
    const ggml_tensor * weights      = nullptr;
    ggml_tensor *       dst          = nullptr;
    int                 node_count   = 0;
};

static bool ggml_cuda_match_moe_weighted_reduction(
        const ggml_cgraph * cgraph,
        int node_idx,
        ggml_cuda_moe_weighted_reduction_match & match) {
    const ggml_tensor * first = cgraph->nodes[node_idx];
    if (first->op != GGML_OP_MUL || first->type != GGML_TYPE_F32 || !ggml_is_contiguous(first)) {
        return false;
    }

    auto split_mul = [](const ggml_tensor * mul, const ggml_tensor *& full, const ggml_tensor *& broadcast) {
        auto is_weights = [mul](const ggml_tensor * tensor) {
            return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) && tensor->ne[0] == 1 &&
                tensor->ne[1] == mul->ne[1] && tensor->ne[2] == mul->ne[2] && tensor->ne[3] == mul->ne[3];
        };
        auto is_experts = [mul](const ggml_tensor * tensor) {
            return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) &&
                ggml_are_same_shape(tensor, mul);
        };

        if (is_experts(mul->src[0]) && is_weights(mul->src[1])) {
            full      = mul->src[0];
            broadcast = mul->src[1];
            return true;
        }
        if (is_experts(mul->src[1]) && is_weights(mul->src[0])) {
            full      = mul->src[1];
            broadcast = mul->src[0];
            return true;
        }
        return false;
    };

    const ggml_tensor * weighted     = first;
    const ggml_tensor * experts      = nullptr;
    const ggml_tensor * expert_scale = nullptr;
    const ggml_tensor * weights      = nullptr;
    int                 mul_count    = 1;

    // Match both structural forms:
    //   (experts * expert_scale) * router_weight
    //   experts * router_weight
    // The matcher does not depend on the model or quantization type.
    if (node_idx + 1 < cgraph->n_nodes) {
        const ggml_tensor * second = cgraph->nodes[node_idx + 1];
        const ggml_tensor * scaled = nullptr;
        const ggml_tensor * route  = nullptr;
        const ggml_tensor * raw    = nullptr;
        const ggml_tensor * scale  = nullptr;
        if (second->op == GGML_OP_MUL && second->type == GGML_TYPE_F32 && ggml_is_contiguous(second) &&
                split_mul(second, scaled, route) && scaled == first && split_mul(first, raw, scale)) {
            weighted     = second;
            experts      = raw;
            expert_scale = scale;
            weights      = route;
            mul_count    = 2;
        }
    }

    if (experts == nullptr && !split_mul(first, experts, weights)) {
        return false;
    }

    const int     n_expert_used = (int) weighted->ne[1];
    const int64_t n_tokens      = weighted->ne[2] * weighted->ne[3];
    if (n_expert_used < 2 || n_expert_used > MOE_WEIGHTED_REDUCTION_MAX_EXPERTS) {
        return false;
    }

    const int node_count = 2 * n_expert_used + mul_count - 1;
    if (node_idx + node_count > cgraph->n_nodes) {
        return false;
    }

    std::vector<ggml_op> ops(node_count, GGML_OP_VIEW);
    ops[0] = GGML_OP_MUL;
    if (mul_count == 2) {
        ops[1] = GGML_OP_MUL;
    }
    std::vector<const ggml_tensor *> views;
    views.reserve(n_expert_used);
    const ggml_tensor * previous = nullptr;
    int n_adds = 0;
    for (int offset = mul_count; offset < node_count; ++offset) {
        const ggml_tensor * candidate = cgraph->nodes[node_idx + offset];
        ops[offset] = candidate->op;

        if (candidate->op == GGML_OP_VIEW) {
            const int expert = (int) views.size();
            if (expert >= n_expert_used || candidate->src[0] != weighted || candidate->view_src != weighted ||
                    candidate->type != GGML_TYPE_F32 || candidate->ne[0] != weighted->ne[0] ||
                    candidate->ne[1] != n_tokens || candidate->ne[2] != 1 || candidate->ne[3] != 1 ||
                    candidate->nb[0] != weighted->nb[0] || candidate->nb[1] != weighted->nb[2] ||
                    candidate->view_offs != (size_t) expert * weighted->nb[1]) {
                return false;
            }
            views.push_back(candidate);
            continue;
        }

        if (candidate->op != GGML_OP_ADD || views.size() < 2 || n_adds + 1 >= (int) views.size()) {
            return false;
        }
        const ggml_tensor * lhs = n_adds == 0 ? views[0] : previous;
        const ggml_tensor * rhs = views[n_adds + 1];
        if (candidate->src[0] != lhs || candidate->src[1] != rhs || candidate->type != GGML_TYPE_F32) {
            return false;
        }
        previous = candidate;
        ++n_adds;
    }

    if ((int) views.size() != n_expert_used || n_adds != n_expert_used - 1 || previous == nullptr) {
        return false;
    }
    if (!ggml_is_contiguous(previous) || previous->ne[0] != weighted->ne[0] ||
            previous->ne[1] != n_tokens || previous->ne[2] != 1 || previous->ne[3] != 1) {
        return false;
    }

    const int output_idx = node_idx + node_count - 1;
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, node_count, ops.data(), &output_idx, 1)) {
        return false;
    }

    match.experts      = experts;
    match.expert_scale = expert_scale;
    match.weights      = weights;
    match.dst          = cgraph->nodes[output_idx];
    match.node_count   = node_count;
    return true;
}


static bool ggml_cuda_can_fuse(const struct ggml_cgraph *                cgraph,
                               int                                       node_idx,
                               std::initializer_list<enum ggml_op>       ops,
                               std::initializer_list<enum ggml_unary_op> unary_ops) {
#ifndef NDEBUG
    const size_t num_unary = std::count(ops.begin(), ops.end(), GGML_OP_UNARY);
    GGML_ASSERT(unary_ops.size() == num_unary);
#endif

    const auto is_equal = [](const std::initializer_list<enum ggml_op> & list1,
                             const std::initializer_list<enum ggml_op> & list2) {
        return std::equal(list1.begin(), list1.end(), list2.begin(), list2.end());
    };

    std::initializer_list<enum ggml_op> mul_mat_bias_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_id_bias_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_GLU };

    std::initializer_list<enum ggml_op> mul_mat_id_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_MUL_MAT,    GGML_OP_GLU };

    if ((is_equal(mul_mat_bias_glu_ops, ops) || is_equal(mul_mat_id_bias_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * ffn_gate      = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_gate_bias = cgraph->nodes[node_idx + 1];
        const ggml_tensor * ffn_up        = cgraph->nodes[node_idx + 2];
        const ggml_tensor * ffn_up_bias   = cgraph->nodes[node_idx + 3];
        const ggml_tensor * glu           = cgraph->nodes[node_idx + 4];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu, ffn_up_bias, ffn_gate_bias)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if ((is_equal(mul_mat_id_glu_ops, ops) || is_equal(mul_mat_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * ffn_gate = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_up   = cgraph->nodes[node_idx + 1];
        const ggml_tensor * glu      = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    std::initializer_list<enum ggml_op> rms_norm_mul_rope_ops          = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE };
    std::initializer_list<enum ggml_op> rms_norm_mul_rope_set_rows_ops = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rms_norm_mul_rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 3];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 4];

        if (ggml_check_edges(cgraph, node_idx, {{1, 0, 0}, {2, 0, 1}, {3, 0, 2}, {4, 0, 3}}) &&
            ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope) &&
            ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if (is_equal(rms_norm_mul_rope_ops, ops) && ggml_can_fuse(cgraph, node_idx, ops)) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
        return false;
    }

    std::initializer_list<enum ggml_op> rope_set_rows_ops = { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * rope     = cgraph->nodes[node_idx];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 1];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    // scale_out*sigmoid(scale_in*post) as the hc_post weights (qwen4exp hc_combine)
    std::initializer_list<enum ggml_op> hc_post_gate_ops = { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_DSV4_HC_POST };

    if (is_equal(hc_post_gate_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 3 })) {
        const ggml_tensor * scale_in  = cgraph->nodes[node_idx];
        const ggml_tensor * sigmoid   = cgraph->nodes[node_idx + 1];
        const ggml_tensor * scale_out = cgraph->nodes[node_idx + 2];

        // the chain must feed hc_post->src[2]; a bias on either scale is not handled
        if (ggml_check_edges(cgraph, node_idx, {{1, 0, 0}, {2, 0, 1}, {3, 2, 2}}) &&
            ggml_get_unary_op(sigmoid) == GGML_UNARY_OP_SIGMOID &&
            ggml_get_op_params_f32(scale_in, 1) == 0.0f && ggml_get_op_params_f32(scale_out, 1) == 0.0f &&
            scale_in->src[0]->type == GGML_TYPE_F32 && scale_out->type == GGML_TYPE_F32 &&
            ggml_are_same_shape(scale_in->src[0], scale_out)) {
            int out_nodes[] = { node_idx + 3 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
        return false;
    }

    if (!ggml_can_fuse(cgraph, node_idx, ops)) {
        return false;
    }

    if ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        const ggml_tensor *rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor *mul      = cgraph->nodes[node_idx+1];
        const ggml_tensor *add      = nullptr;

        if (ops.size() == 3 && ops.begin()[2] == GGML_OP_ADD) {
            add = cgraph->nodes[node_idx+2];
        }

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        //rms norm only supports F32
        if (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            return false;
        }

        if (add && (add->src[0]->type != GGML_TYPE_F32 ||
            add->src[1]->type != GGML_TYPE_F32 ||
            add->type != GGML_TYPE_F32) ) {
            return false;
        }

        //if rms norm is the B operand, then we don't handle broadcast
        if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
            return false;
        }

        //rms_norm kernel assumes contiguous rows
        if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            return false;
        }

        if (add && (!ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous_rows(add->src[1]))) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_SCALE) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * scale    = cgraph->nodes[node_idx+1];

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        float bias;
        memcpy(&bias, (const float *) scale->op_params + 1, sizeof(float));

        return bias == 0.0f && scale->type == GGML_TYPE_F32;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_UNARY
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * silu     = cgraph->nodes[node_idx+1];
        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }

        if (ssm_conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_ADD
     && ops.begin()[2] == GGML_OP_UNARY && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * add      = cgraph->nodes[node_idx+1];
        const ggml_tensor * silu     = cgraph->nodes[node_idx+2];
        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }

        if (ssm_conv->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }

        // ADD must consume ssm_conv's output and broadcast a 1-D channel-wise bias.
        const ggml_tensor * bias = (add->src[0] == ssm_conv) ? add->src[1] : add->src[0];
        if (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
            return false;
        }
        if (ggml_nelements(bias) != ssm_conv->ne[0] || bias->ne[0] != ssm_conv->ne[0]) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL
     && unary_ops.size() == 1 && (unary_ops.begin()[0] == GGML_UNARY_OP_SILU || unary_ops.begin()[0] == GGML_UNARY_OP_SIGMOID || unary_ops.begin()[0] == GGML_UNARY_OP_SOFTPLUS)) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * mul   = cgraph->nodes[node_idx+1];

        if (ggml_get_unary_op(unary) != unary_ops.begin()[0]) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16 && unary->type != GGML_TYPE_BF16) {
            return false;
        }

        if (unary->type != mul->type) {
            return false;
        }

        const ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
        if (other->type != unary->type) {
            return false;
        }
        if (!ggml_is_contiguous_1(other) || !ggml_is_contiguous_1(unary->src[0]) || !ggml_are_same_shape(other, unary)) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_SQR
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_RELU) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * sqr   = cgraph->nodes[node_idx+1];

        if (ggml_get_unary_op(unary) != GGML_UNARY_OP_RELU) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            return false;
        }

        if (unary->type != sqr->type) {
            return false;
        }

        if (!ggml_is_contiguous(unary->src[0])) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SCALE && ops.begin()[1] == GGML_OP_UNARY && ops.begin()[2] == GGML_OP_SCALE
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_TANH) {
        const ggml_tensor *scale  = cgraph->nodes[node_idx];
        const ggml_tensor *tanh   = cgraph->nodes[node_idx+1];
        const ggml_tensor *scale2 = cgraph->nodes[node_idx+2];

        // the fused softcap kernel is F32 only
        if (scale->src[0]->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32) {
            return false;
        }

        if (ggml_get_unary_op(tanh) != GGML_UNARY_OP_TANH) {
            return false;
        }

        // Check for bias
        if (ggml_get_op_params_f32(scale, 1) != 0.0f || ggml_get_op_params_f32(scale2, 1) != 0.0f) {
            return false;
        }

        return true;
    }

    return false;
}

// try and fuse nodes and return the number of nodes to skip
// [TAG_CPY_CHAIN_FUSION] Finds n >= 2 consecutive CPY nodes (only view/no-op nodes between them) that
// copy the same shape with the same strides and whose src/dst pointers advance by constant byte
// deltas. The chain is safe to run as one launch when no copy reads what any copy writes and no two
// destinations overlap; the element order inside each copy is unchanged, so the result is the same
// bytes. TURBO_CPY_CHAIN=0 disables it. Returns the number of graph nodes to skip after node i.
static int ggml_cuda_try_cpy_chain_fusion(const ggml_cgraph * cgraph, int i, int & n_out, int64_t & dsrc_out, int64_t & ddst_out) {
    static const bool on = [] {
        const char * e = getenv("TURBO_CPY_CHAIN");
        return !(e && e[0] == '0');
    }();
    if (!on) {
        return 0;
    }
    const ggml_tensor * first = cgraph->nodes[i];
    if (first->op != GGML_OP_CPY || (first->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }
    const ggml_tensor * s0 = first->src[0];
    const ggml_tensor * s1 = first->src[1];
    if (!s0 || !s1 || s0->type != GGML_TYPE_F32 || s1->type != GGML_TYPE_F32 || first->type != GGML_TYPE_F32 ||
        s0->ne[3] != 1 || s1->ne[3] != 1 || first->data != s1->data || ggml_is_empty(s0)) {
        return 0;
    }
    const auto same_layout = [](const ggml_tensor * a, const ggml_tensor * b) {
        return a->type == b->type && std::equal(a->ne, a->ne + GGML_MAX_DIMS, b->ne) &&
               std::equal(a->nb, a->nb + GGML_MAX_DIMS, b->nb);
    };
    int     n    = 1;
    int     last = i;
    int64_t dsrc = 0;
    int64_t ddst = 0;
    const ggml_tensor * prev = first;
    for (int j = i + 1; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * t = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(t)) {
            continue;
        }
        if (t->op != GGML_OP_CPY || (t->flags & GGML_TENSOR_FLAG_OUTPUT) || !t->src[0] || !t->src[1] ||
            t->data != t->src[1]->data || t->type != GGML_TYPE_F32 ||
            !same_layout(t->src[0], s0) || !same_layout(t->src[1], s1)) {
            break;
        }
        const int64_t d0 = (const char *) t->src[0]->data - (const char *) prev->src[0]->data;
        const int64_t d1 = (const char *) t->src[1]->data - (const char *) prev->src[1]->data;
        if (n == 1) {
            dsrc = d0;
            ddst = d1;
        } else if (d0 != dsrc || d1 != ddst) {
            break;
        }
        ++n;
        last = j;
        prev = t;
    }
    if (n < 2) {
        return 0;
    }
    // byte spans touched by the whole chain (nbytes covers the strided extent of one copy)
    const int64_t ext0 = (int64_t) ggml_nbytes(s0);
    const int64_t ext1 = (int64_t) ggml_nbytes(s1);
    const int64_t s0_lo = (int64_t) s0->data + (dsrc < 0 ? dsrc * (n - 1) : 0);
    const int64_t s0_hi = (int64_t) s0->data + (dsrc > 0 ? dsrc * (n - 1) : 0) + ext0;
    const int64_t s1_lo = (int64_t) s1->data + (ddst < 0 ? ddst * (n - 1) : 0);
    const int64_t s1_hi = (int64_t) s1->data + (ddst > 0 ? ddst * (n - 1) : 0) + ext1;
    const bool reads_own_writes = s0_lo < s1_hi && s1_lo < s0_hi;
    const bool dst_overlap      = ddst == 0 || (ddst > 0 ? ddst : -ddst) < ext1;
    if (reads_own_writes || dst_overlap) {
        return 0;
    }
    n_out    = n;
    dsrc_out = dsrc;
    ddst_out = ddst;
    return last - i;
}

// [TAG_GDN_PREP_FUSION] add(x, dt) -> softplus -> mul(., a) with dt and a plain F32 vectors along dim 0:
// Qwen3.8's GDN decay gate does this three times per recurrent layer per step. Bit-exact single
// launch; TURBO_GDN_PREP=0 disables it.
static bool ggml_cuda_try_gdn_prep_fusion(const ggml_cgraph * cgraph, int i) {
    static const bool on = [] {
        const char * e = getenv("TURBO_GDN_PREP");
        return !(e && e[0] == '0');
    }();
    if (!on || i + 2 >= cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * add = cgraph->nodes[i];
    const ggml_tensor * sp  = cgraph->nodes[i + 1];
    const ggml_tensor * mul = cgraph->nodes[i + 2];
    if (add->op != GGML_OP_ADD || sp->op != GGML_OP_UNARY || ggml_get_unary_op(sp) != GGML_UNARY_OP_SOFTPLUS || mul->op != GGML_OP_MUL) {
        return false;
    }
    if (!ggml_can_fuse(cgraph, i, { GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_MUL }) ||
        !ggml_check_edges(cgraph, i, { { 1, 0, 0 }, { 2, 0, 1 } })) {
        return false;
    }
    const ggml_tensor * x  = add->src[0];
    const ggml_tensor * dt = add->src[1];
    const ggml_tensor * a  = mul->src[1];
    const auto vec_like = [&](const ggml_tensor * v) {
        return v && v->type == GGML_TYPE_F32 && ggml_is_contiguous(v) && v->ne[0] == x->ne[0] &&
               v->ne[1] == 1 && v->ne[2] == 1 && v->ne[3] == 1;
    };
    if (!x || x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || !vec_like(dt) || !vec_like(a)) {
        return false;
    }
    if (add->type != GGML_TYPE_F32 || sp->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(mul) || !ggml_are_same_shape(x, mul) || !ggml_are_same_shape(x, add) || !ggml_are_same_shape(x, sp)) {
        return false;
    }
    return true;
}

// [TAG_RESID_NORM_FUSION] add(x, y) -> rms_norm -> mul(., w) where the add is a plain same-shape F32 sum
// (a residual connection). The add's result keeps being written, so its other consumers are fine.
// Bit-exact single launch; TURBO_RESID_NORM=0 disables it.
static bool ggml_cuda_try_resid_norm_fusion(const ggml_cgraph * cgraph, int i) {
    static const bool on = [] {
        const char * e = getenv("TURBO_RESID_NORM");
        return !(e && e[0] == '0');
    }();
    if (!on || i + 2 >= cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * add  = cgraph->nodes[i];
    const ggml_tensor * norm = cgraph->nodes[i + 1];
    const ggml_tensor * mul  = cgraph->nodes[i + 2];
    if (add->op != GGML_OP_ADD || norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || norm->src[0] != add) {
        return false;
    }
    const ggml_tensor * x = add->src[0];
    const ggml_tensor * y = add->src[1];
    if (!x || !y || add->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32 || y->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(add) || !ggml_is_contiguous(x) || !ggml_is_contiguous(y) ||
        !ggml_are_same_shape(add, x) || !ggml_are_same_shape(add, y) || ggml_is_empty(add)) {
        return false;
    }
    // the norm -> mul pair must be fusable on its own (single use of the norm output, F32 everywhere)
    if (!ggml_cuda_can_fuse(cgraph, i + 1, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
        return false;
    }
    const ggml_tensor * w = mul->src[0] == norm ? mul->src[1] : mul->src[0];
    if (w->type != GGML_TYPE_F32 || w->nb[0] != ggml_type_size(GGML_TYPE_F32) || !ggml_is_contiguous(mul)) {
        return false;
    }
    return true;
}

// [TAG_RMSNORM_SCALE_FUSION] rms_norm(x, eps) -> scale(., s) with no bias. Upstream 5fdfa6282 builds the GDN q/k
// l2 norm this way (build_gdn_l2_norm in src/models/models.h: rms_norm(x, eps/n) * 1/sqrt(n)), twice per recurrent
// layer per step on Qwen3.8. The graph keeps both ops, so other backends run them plain; CUDA runs one launch that
// never writes the norm's output. Bit-exact; TURBO_RMSNORM_SCALE_FUSION=0 disables it.
static bool ggml_cuda_try_rms_norm_scale_fusion(const ggml_cgraph * cgraph, int i) {
    static const bool on = [] {
        const char * e = getenv("TURBO_RMSNORM_SCALE_FUSION");
        return !(e && e[0] == '0');
    }();
    if (!on || i + 1 >= cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * norm  = cgraph->nodes[i];
    const ggml_tensor * scale = cgraph->nodes[i + 1];
    if (norm->op != GGML_OP_RMS_NORM || scale->op != GGML_OP_SCALE || scale->src[0] != norm) {
        return false;
    }
    // the norm output has the scale as its only reader, is not a graph output or a view, same shape
    if (!ggml_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_SCALE })) {
        return false;
    }
    const ggml_tensor * x = norm->src[0];
    if (!x || x->type != GGML_TYPE_F32 || norm->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32 ||
        x->nb[0] != ggml_type_size(GGML_TYPE_F32) || !ggml_is_contiguous(norm) || !ggml_is_contiguous(scale) ||
        ggml_is_empty(norm)) {
        return false;
    }
    if (ggml_get_op_params_f32(scale, 1) != 0.0f) { // scale bias
        return false;
    }
    // the fused kernel reads x after other rows may have been written: the output must not alias it
    const int out_node = i + 1;
    return ggml_cuda_check_fusion_memory_ranges(cgraph, i, 2, &out_node, 1);
}

// [TAG_FN_L3_GPU_HCFUSE] at a SCALE node that the qwen4exp graph marked (ggml-fn-l3.h): the hc chains in one launch.
//   HCLO: scale -> silu;  HCW: scale -> sigmoid -> scale, and with the dsv4_hc_post (no comb) that reads it, that too.
// Returns the nodes after i that the launch covered (0: none). Same arithmetic as the unfused ops (fn-l3.cu).
static int ggml_cuda_fn_l3_try_hc(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    const ggml_tensor * s1   = cgraph->nodes[i];
    const int32_t       mark = ggml_fn_l3_get(s1);
    if (s1->op != GGML_OP_SCALE || (mark != GGML_FN_L3_HCW && mark != GGML_FN_L3_HCLO) || !ggml_cuda_fn_l3_enabled()) {
        return 0;
    }
    const ggml_tensor * x = s1->src[0];
    if (x == nullptr || x->type != GGML_TYPE_F32 || s1->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) ||
            !ggml_is_contiguous(s1) || ggml_is_empty(s1)) {
        return 0;
    }
    const auto is_unary = [&](int j, ggml_unary_op uop) {
        return j < cgraph->n_nodes && cgraph->nodes[j]->op == GGML_OP_UNARY && ggml_get_unary_op(cgraph->nodes[j]) == uop &&
               cgraph->nodes[j]->type == GGML_TYPE_F32 && ggml_is_contiguous(cgraph->nodes[j]);
    };

    if (mark == GGML_FN_L3_HCLO) {
        if (!is_unary(i + 1, GGML_UNARY_OP_SILU) || !ggml_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY })) {
            return 0;
        }
        const int out = i + 1;
        if (!ggml_cuda_check_fusion_memory_ranges(cgraph, i, 2, &out, 1)) {
            return 0;
        }
        ggml_cuda_fn_l3_hclo(*cuda_ctx, s1, cgraph->nodes[i + 1]);
        return 1;
    }

    if (!is_unary(i + 1, GGML_UNARY_OP_SIGMOID) ||
            !ggml_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE }) ||
            cgraph->nodes[i + 2]->type != GGML_TYPE_F32 || !ggml_is_contiguous(cgraph->nodes[i + 2])) {
        return 0;
    }
    ggml_tensor * s2 = cgraph->nodes[i + 2];

    // the combine that reads s2 right after it (not when the graph deferred the weights: then s2 is the output)
    if (i + 3 < cgraph->n_nodes) {
        ggml_tensor * post = cgraph->nodes[i + 3];
        const ggml_tensor * xb  = post->src[0];
        const ggml_tensor * res = post->src[1];
        if (post->op == GGML_OP_DSV4_HC_POST && (post->flags & GGML_TENSOR_FLAG_COMPUTE) && post->src[2] == s2 &&
                post->src[3] == nullptr && ggml_node_has_n_uses(cgraph, i + 2, 1) &&
                xb && res && xb->type == GGML_TYPE_F32 && res->type == GGML_TYPE_F32 && post->type == GGML_TYPE_F32 &&
                s2->ne[0] == res->ne[1] && s2->ne[1] == xb->ne[1] && s2->ne[2] == 1 && s2->ne[3] == 1) {
            const int out = i + 3;
            if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, 4, &out, 1)) {
                ggml_cuda_fn_l3_hcw4(*cuda_ctx, s1, s2, post);
                return 3;
            }
        }
    }

    const int out = i + 2;
    if (!ggml_cuda_check_fusion_memory_ranges(cgraph, i, 3, &out, 1)) {
        return 0;
    }
    ggml_cuda_fn_l3_hcw3(*cuda_ctx, s1, s2);
    return 2;
}

// [TAG_FN_L3_GPU_GDNAB] at the GDN beta MUL_MAT the qwen4exp graph marked: beta -> sigmoid, alpha (the other marked MUL_MAT
// of the same input) -> add dt -> softplus -> mul a, in this order with only views between them; one launch writes the
// sigmoid and the mul (fn-l3.cu). Returns the nodes after i that the launch covered (0: none).
static int ggml_cuda_fn_l3_try_gdnab(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    ggml_tensor * mb = cgraph->nodes[i];
    if (mb->op != GGML_OP_MUL_MAT || ggml_fn_l3_get(mb) != GGML_FN_L3_GDNAB || !ggml_cuda_fn_l3_enabled()) {
        return 0;
    }
    // the next five nodes that are not views
    int idx[6] = { i, -1, -1, -1, -1, -1 };
    int n = 1;
    for (int j = i + 1; j < cgraph->n_nodes && n < 6; ++j) {
        if (ggml_cuda_is_view_or_noop(cgraph->nodes[j])) {
            // only the views of this chain may sit between its nodes
            if (ggml_node_get_use_count(cgraph, j) != 1 || (cgraph->nodes[j]->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                return 0;
            }
            continue;
        }
        idx[n++] = j;
    }
    if (n < 6) {
        return 0;
    }
    ggml_tensor * sig  = cgraph->nodes[idx[1]];
    ggml_tensor * ma   = cgraph->nodes[idx[2]];
    ggml_tensor * add  = cgraph->nodes[idx[3]];
    ggml_tensor * sp   = cgraph->nodes[idx[4]];
    ggml_tensor * gate = cgraph->nodes[idx[5]];
    const auto base = [](const ggml_tensor * t) {
        for (int k = 0; k < 4 && t != nullptr && (t->op == GGML_OP_RESHAPE || (t->op == GGML_OP_VIEW && t->view_offs == 0)); ++k) {
            t = t->src[0];
        }
        return t;
    };
    const auto vec_of = [](const ggml_tensor * v, int64_t ne0) {
        return v != nullptr && v->op == GGML_OP_NONE && v->type == GGML_TYPE_F32 && ggml_is_contiguous(v) &&
               v->ne[0] == ne0 && v->ne[1] == 1 && v->ne[2] == 1 && v->ne[3] == 1;
    };
    const ggml_tensor * wb = mb->src[0];
    const ggml_tensor * y  = mb->src[1];
    if (sig->op != GGML_OP_UNARY || ggml_get_unary_op(sig) != GGML_UNARY_OP_SIGMOID || base(sig->src[0]) != mb ||
            ma->op != GGML_OP_MUL_MAT || ggml_fn_l3_get(ma) != GGML_FN_L3_GDNAB || ma->src[1] != y ||
            add->op != GGML_OP_ADD || base(add->src[0]) != ma ||
            sp->op != GGML_OP_UNARY || ggml_get_unary_op(sp) != GGML_UNARY_OP_SOFTPLUS || sp->src[0] != add ||
            gate->op != GGML_OP_MUL || gate->src[0] != sp) {
        return 0;
    }
    const ggml_tensor * wa = ma->src[0];
    // plain f32 [K, rows] x [K, T] products of one input, T <= 8, the rows of each output contiguous and token-major
    if (wb->type != GGML_TYPE_F32 || wa->type != GGML_TYPE_F32 || y->type != GGML_TYPE_F32 || mb->type != GGML_TYPE_F32 ||
            ma->type != GGML_TYPE_F32 || wb->ne[0] != y->ne[0] || wa->ne[0] != y->ne[0] || y->ne[0] % 2 != 0 ||
            wb->ne[2] != 1 || wb->ne[3] != 1 || wa->ne[2] != 1 || wa->ne[3] != 1 || y->ne[2] != 1 || y->ne[3] != 1 ||
            y->ne[1] < 1 || y->ne[1] > MMVF_MAX_BATCH_SIZE || wb->nb[0] != sizeof(float) || wa->nb[0] != sizeof(float) ||
            y->nb[0] != sizeof(float) || (wb->nb[1]/sizeof(float)) % 2 != 0 || (wa->nb[1]/sizeof(float)) % 2 != 0 ||
            (y->nb[1]/sizeof(float)) % 2 != 0 || wb->ne[0] > INT32_MAX || (wb->nb[1]/sizeof(float)) > INT32_MAX ||
            (wa->nb[1]/sizeof(float)) > INT32_MAX || (y->nb[1]/sizeof(float)) > INT32_MAX ||
            !vec_of(add->src[1], wa->ne[1]) || !vec_of(gate->src[1], wa->ne[1]) ||
            sig->type != GGML_TYPE_F32 || gate->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || sp->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(sig) || !ggml_is_contiguous(gate) || !ggml_is_contiguous(mb) || !ggml_is_contiguous(ma) ||
            ggml_nelements(sig) != wb->ne[1]*y->ne[1] || ggml_nelements(gate) != wa->ne[1]*y->ne[1] ||
            ggml_nelements(add) != wa->ne[1]*y->ne[1] || ggml_nelements(sp) != wa->ne[1]*y->ne[1]) {
        return 0;
    }
    // only where ggml_cuda_mul_mat computes both unmarked products with mul_mat_vec_f (f32 at T <= 3 on a GPU with fp32
    // MMA: wider products go to MMF / cuBLAS, whose sums round differently), and the sigmoid / add read the products in
    // their own element order (contiguous views at offset 0)
    const int cc = ggml_cuda_info().devices[cuda_ctx->device].cc;
    const auto mmvf_path = [cc](const ggml_tensor * mm) {
        const ggml_tensor * w = mm->src[0];
        const bool bad_padding_clear = w->buffer != nullptr &&
            ggml_backend_buffer_get_usage(w->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
            ggml_nbytes(w) != ggml_backend_buffer_get_alloc_size(w->buffer, w) && w->view_src;
        return !bad_padding_clear && !ggml_cuda_op_mul_mat_use_fwht(mm) &&
               ggml_cuda_should_use_mmvf(w->type, cc, ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size, w->ne, w->nb, mm->src[1]->ne[1]);
    };
    if (!mmvf_path(mb) || !mmvf_path(ma) || !ggml_is_contiguous(sig->src[0]) || !ggml_is_contiguous(add->src[0])) {
        return 0;
    }
    // every node but the two outputs has this chain as its only reader
    for (int k : { idx[0], idx[2], idx[3], idx[4] }) {
        if (ggml_node_get_use_count(cgraph, k) != 1 || (cgraph->nodes[k]->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
    }
    const int outs[2] = { idx[1], idx[5] };
    if (!ggml_cuda_check_fusion_memory_ranges(cgraph, i, idx[5] - i + 1, outs, 2)) {
        return 0;
    }
    ggml_cuda_fn_l3_gdnab(*cuda_ctx, mb, ma, add->src[1], gate->src[1], sig, gate);
    return idx[5] - i;
}

// [TAG_FN_L4_POST] a bridge post marked GGML_FN_L4_POST right behind the fused topk-moe nodes (only views between them):
// the top-k and the post in one launch. Returns the nodes to skip after i_last (0: not taken, nothing launched).
static int ggml_cuda_fn_l4_try_topk_post(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i_last,
        const ggml_tensor * logits, ggml_tensor * weights, ggml_tensor * ids, const ggml_tensor * clamp,
        const ggml_tensor * scale, const ggml_tensor * bias, const ggml_cuda_topk_moe_args & args) {
    for (int j = i_last + 1; j < cgraph->n_nodes; ++j) {
        ggml_tensor * t = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(t)) {
            continue;
        }
        if (t->op != GGML_OP_MOE_HOST_POST || ggml_fn_l3_get(t) != GGML_FN_L4_POST || (t->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            return 0;
        }
        return ggml_cuda_fn_l4_topk_post(ctx, logits, weights, ids, clamp, scale, bias, args, t) ? j - i_last : 0;
    }
    return 0;
}

// [TAG_FN_L4_QSA_QKV] a MUL_MAT that ggml_cuda_mul_mat sends to the run-ahead kernel of ggml_cuda_fn_l3_mul_mat_vec_q
static bool ggml_cuda_fn_l4_qkv_node_ok(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    if (dst->op != GGML_OP_MUL_MAT || src0 == nullptr || src1 == nullptr || src0->buffer == nullptr ||
            (dst->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || ggml_fn_l3_get(dst) != GGML_FN_L3_MMV ||
            src0->type != GGML_TYPE_Q8_0 || ggml_cuda_op_mul_mat_use_fwht(dst)) {
        return false;
    }
    const int     cc        = ggml_cuda_info().devices[ctx.device].cc;
    const int     warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    const int64_t ne11      = src1->ne[1];
    return !ggml_cuda_should_use_mmvf(src0->type, cc, ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size, src0->ne, src0->nb, ne11) &&
           !ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11, /*mul_mat_id =*/ false) &&
           src0->ne[1] != 1 && !ggml_cuda_should_use_mmsb(src0, src1, dst, cc) &&
           !ggml_cuda_should_use_mmqsn(src0, src1, dst, cc) &&
           ggml_cuda_should_use_mmvq(src0->type, cc, ne11, /*ne01 =*/ src0->ne[1]) &&
           ggml_cuda_fn_l4_mmvq_ra_ok(ctx, src0, src1, dst);
}

// [TAG_FN_L4_QSA_QKV] the marked q projection at node i and the two products after it (no-op views between) that read the
// same src1, each one a run-ahead mat-vec: one launch. Returns the nodes skipped, 0 when not taken (nothing launched).
static int ggml_cuda_fn_l4_try_qkv(ggml_backend_cuda_context * ctx, ggml_cgraph * cg, int i) {
    ggml_tensor * mm[3] = { cg->nodes[i], nullptr, nullptr };
    int j = i + 1;
    for (int s = 1; s < 3; ++s) {
        while (j < cg->n_nodes && ggml_cuda_is_view_or_noop(cg->nodes[j])) {
            ++j;
        }
        if (j >= cg->n_nodes) {
            return 0;
        }
        mm[s] = cg->nodes[j++];
    }
    for (ggml_tensor * t : mm) {
        if (!ggml_cuda_fn_l4_qkv_node_ok(*ctx, t) || t->src[1] != mm[0]->src[1] || t->src[0]->ne[0] != mm[0]->src[0]->ne[0]) {
            return 0;
        }
    }
    const ggml_tensor * src0[3] = { mm[0]->src[0], mm[1]->src[0], mm[2]->src[0] };
    ggml_cuda_fn_l4_mul_mat_vec_q_multi(*ctx, src0, mm[0]->src[1], mm, 3);
    static std::atomic<bool> told{false};
    if (!told.exchange(true)) {
        GGML_LOG_INFO("ggml_cuda: [TAG_FN_L4_QSA] QKV: the q, k and v projections of a QSA layer in one launch\n");
    }
    return (j - 1) - i;
}

// [TAG_FN_L18_GPUFLOOR_HCADD] GGML_CUDA_FN_L18_HCADD=1: the hc norm launch (K1) also folds in the ADD of the bridged FFN
// output (hot chain + host experts) that comes before the ADD of the shared expert: (hot + host) + shared, as the two
// adds compute it, one launch less per layer
static bool fn_l18_hcadd_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_FN_L18_HCADD");
        const bool v = e != nullptr && atoi(e) != 0;
        if (v) {
            GGML_LOG_INFO("ggml_cuda: [TAG_FN_L18_GPUFLOOR_HCADD] the hc norm launch also adds the bridged FFN output\n");
        }
        return v;
    }();
    return on;
}

// [TAG_FN_L18_GPUFLOOR_QKVZ] the marked GDN qkv projection at node i and the z projection after it (no-op views between),
// both reading the same src1, both run-ahead mat-vecs: one launch of the run-ahead body (the warp-per-row one with
// GGML_CUDA_FN_L18_MMV=1), the same bits per row. Returns the nodes
// skipped, 0 when not taken (nothing launched).
static int ggml_cuda_fn_l18_try_qkvz(ggml_backend_cuda_context * ctx, ggml_cgraph * cg, int i) {
    ggml_tensor * mm[2] = { cg->nodes[i], nullptr };
    int j = i + 1;
    while (j < cg->n_nodes && ggml_cuda_is_view_or_noop(cg->nodes[j])) {
        ++j;
    }
    if (j >= cg->n_nodes) {
        return 0;
    }
    mm[1] = cg->nodes[j++];
    for (ggml_tensor * t : mm) {
        if (!ggml_cuda_fn_l4_qkv_node_ok(*ctx, t) || t->src[1] != mm[0]->src[1] || t->src[0]->ne[0] != mm[0]->src[0]->ne[0]) {
            return 0;
        }
    }
    const ggml_tensor * src0[2] = { mm[0]->src[0], mm[1]->src[0] };
    ggml_cuda_fn_l4_mul_mat_vec_q_multi(*ctx, src0, mm[0]->src[1], mm, 2);
    ggml_cuda_fn_l3_note(GGML_CUDA_FN_L18_PATH_QKVZ, "[TAG_FN_L18_GPUFLOOR_QKVZ] the GDN qkv and z projections in one launch");
    return (j - 1) - i;
}

// [TAG_FN_L18_GPUFLOOR_GDNOUT] the marked RMS_NORM at node i -> MUL (gamma) -> UNARY SIGMOID (z) -> MUL, the inner nodes
// read only by the next one: one launch, plus the q8_1 copy for the MUL_MAT that reads the result next. Returns the
// nodes skipped, 0 when not taken (nothing launched).
static int32_t ggml_cuda_fn_l4_uses(const ggml_cgraph * cgraph, const ggml_tensor * t);
static int ggml_cuda_fn_l18_try_gdnout(ggml_backend_cuda_context * ctx, ggml_cgraph * cg, int i) {
    const auto next = [cg](int k) {
        for (++k; k < cg->n_nodes && ggml_cuda_is_view_or_noop(cg->nodes[k]); ++k) {
        }
        return k < cg->n_nodes ? k : -1;
    };
    const auto f32c = [](const ggml_tensor * t) {
        return t != nullptr && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t);
    };
    const auto inner = [cg](const ggml_tensor * t) {
        return ggml_cuda_fn_l4_uses(cg, t) == 1 && !(t->flags & GGML_TENSOR_FLAG_OUTPUT) && t->view_src == nullptr;
    };
    ggml_tensor * rms = cg->nodes[i];
    const int i_mw = next(i);
    const int i_sg = i_mw < 0 ? -1 : next(i_mw);
    if (i_sg < 0 || i_sg + 1 >= cg->n_nodes) {
        return 0;
    }
    ggml_tensor * mw = cg->nodes[i_mw];
    ggml_tensor * sg = cg->nodes[i_sg];
    ggml_tensor * m  = cg->nodes[i_sg + 1];
    const ggml_tensor * x = rms->src[0];
    if (rms->op != GGML_OP_RMS_NORM || x == nullptr || x->type != GGML_TYPE_F32 || x->nb[0] != sizeof(float) ||
            !f32c(rms) || !inner(rms) || rms->ne[3] != 1 || rms->ne[0] % 32 != 0 || rms->ne[0] > 256 ||
            !ggml_are_same_shape(x, rms) || x->ne[3] != 1 || (x->nb[1] % sizeof(float)) != 0 || (x->nb[2] % sizeof(float)) != 0) {
        return 0;
    }
    const ggml_tensor * w = mw->op == GGML_OP_MUL ? (mw->src[0] == rms ? mw->src[1] : mw->src[1] == rms ? mw->src[0] : nullptr) : nullptr;
    if (w == nullptr || !f32c(w) || ggml_nelements(w) != rms->ne[0] || w->ne[0] != rms->ne[0] || !f32c(mw) ||
            !ggml_are_same_shape(mw, rms) || !inner(mw)) {
        return 0;
    }
    const ggml_tensor * z = sg->src[0];
    if (sg->op != GGML_OP_UNARY || ggml_get_unary_op(sg) != GGML_UNARY_OP_SIGMOID || !f32c(z) || !f32c(sg) || !inner(sg) ||
            !ggml_are_same_shape(z, rms)) {
        return 0;
    }
    if (m->op != GGML_OP_MUL || !((m->src[0] == mw && m->src[1] == sg) || (m->src[0] == sg && m->src[1] == mw)) || !f32c(m) ||
            !ggml_are_same_shape(m, rms)) {
        return 0;
    }
    // the outputs do not overlap the inputs the launch reads
    const auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        const char * a0 = (const char *) a->data;
        const char * b0 = (const char *) b->data;
        return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
    };
    if (overlap(m, x) || overlap(m, w) || overlap(m, z)) {
        return 0;
    }
    ggml_cuda_fn_l3_q8_dst q8;
    block_q8_1 * yq = nullptr;
    int64_t bpr = 0;
    if (ggml_cuda_mmvq_q8_produce(*ctx, m, q8, /*padding_ok =*/ false)) {
        if (q8.ne10 != rms->ne[0]*rms->ne[1] || q8.nrows != rms->ne[2] || q8.bpr*QK8_1 != q8.ne10) {
            ctx->mmvq_q8.reset();
        } else {
            yq  = q8.y;
            bpr = q8.bpr;
            ctx->mmvq_q8.group   = cg->nodes + i;
            ctx->mmvq_q8.n_group = i_sg + 1 - i + 1;
        }
    }
    ggml_cuda_fn_l18_gdnout_launch(*ctx, rms, w, z, m, yq, bpr);
    ggml_cuda_fn_l3_note(GGML_CUDA_FN_L18_PATH_GDNOUT, "[TAG_FN_L18_GPUFLOOR_GDNOUT] GDN output norm, gate and q8_1 copy in one launch");
    return (i_sg + 1) - i;
}

// [TAG_FN_L4_HC] graph uses of t (0: not a node of the graph)
static int32_t ggml_cuda_fn_l4_uses(const ggml_cgraph * cgraph, const ggml_tensor * t) {
    const size_t pos = ggml_hash_find(&cgraph->visited_hash_set, t);
    if (pos == GGML_HASHSET_FULL || !ggml_bitset_get(cgraph->visited_hash_set.used, pos)) {
        return 0;
    }
    return cgraph->use_counts[pos];
}

// [TAG_FN_L4_HC] the tensor a chain of reshapes / offset-0 views starts from
static const ggml_tensor * ggml_cuda_fn_l4_base(const ggml_tensor * t) {
    for (int k = 0; k < 4 && t != nullptr && (t->op == GGML_OP_RESHAPE || (t->op == GGML_OP_VIEW && t->view_offs == 0)); ++k) {
        t = t->src[0];
    }
    return t;
}

// [TAG_FN_L4_HC] true when ggml_cuda_mul_mat computes this q8_0 MUL_MAT (1..4 columns) with mul_mat_vec_q's generic launch
static bool ggml_cuda_fn_l4_mmvq_node(const ggml_tensor * mm, const int cc, const int warp_size) {
    const ggml_tensor * src0 = mm->src[0];
    const ggml_tensor * src1 = mm->src[1];
    if (src0 == nullptr || src1 == nullptr || src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 ||
            mm->type != GGML_TYPE_F32 || src0->buffer == nullptr || src0->data == nullptr ||
            ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE ||
            ggml_cuda_op_mul_mat_use_fwht(mm) || warp_size != 32 || !ggml_cuda_fn_l4_mmvq_generic(cc)) {
        return false;
    }
    const int64_t ne11 = src1->ne[1];
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || ne11 < 1 || ne11 > 4 ||
            src0->nb[0] != ggml_type_size(src0->type) || src0->ne[0] % QK8_0 != 0 || src0->ne[0] != src1->ne[0] ||
            src0->ne[0] > INT32_MAX/64 || src0->ne[1] > INT32_MAX/64 || !ggml_is_contiguous(src1) || !ggml_is_contiguous(mm)) {
        return false;
    }
    return !ggml_cuda_should_use_mmvf(src0->type, cc, ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size, src0->ne, src0->nb, ne11) &&
           !ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11, /*mul_mat_id =*/ false) &&
           !ggml_cuda_should_use_mmsb(src0, src1, mm, cc) && !ggml_cuda_should_use_mmqsn(src0, src1, mm, cc) &&
           ggml_cuda_should_use_mmvq(src0->type, cc, ne11, /*ne01 =*/ src0->ne[1]);
}

// [TAG_FN_L4_HC] at a SCALE marked GGML_FN_L4_HCPOST (the combine, then the mixer) or at an RMS_NORM whose MUL is marked
// GGML_FN_L4_HC: the whole hc mixer in three launches (fn-l4-hc.cu). Returns the nodes after i that they cover (0: none).
static int ggml_cuda_fn_l4_try_hc(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    ggml_tensor * n0 = cgraph->nodes[i];
    if ((n0->op != GGML_OP_SCALE && n0->op != GGML_OP_RMS_NORM && n0->op != GGML_OP_ADD) || cuda_ctx->curr_stream_no != 0) {
        return 0;
    }

    // the next node that computes (views and no-ops between the chain's nodes are skipped), -1 when none
    const auto next = [cgraph](int j) -> int {
        if (j < 0) {
            return -1;
        }
        for (++j; j < cgraph->n_nodes; ++j) {
            const ggml_tensor * t = cgraph->nodes[j];
            if (ggml_cuda_is_view_or_noop(t)) {
                continue;
            }
            return (t->flags & GGML_TENSOR_FLAG_COMPUTE) ? j : -1;
        }
        return -1;
    };

    // at an ADD: the block output of a combine that starts here (the FFN half's routed + shared experts)
    ggml_tensor * add = nullptr;
    ggml_tensor * add0 = nullptr; // [TAG_FN_L18_GPUFLOOR_HCADD] the ADD before it (hot chain + host experts), folded in too
    int i_s1 = i;
    if (n0->op == GGML_OP_ADD) {
        i_s1 = next(i);
        if (i_s1 >= 0 && cgraph->nodes[i_s1]->op == GGML_OP_ADD && fn_l18_hcadd_enabled()) {
            ggml_tensor * a2 = cgraph->nodes[i_s1];
            const int i_s2 = next(i_s1);
            if ((a2->src[0] == n0 || a2->src[1] == n0) && i_s2 >= 0 && cgraph->nodes[i_s2]->op == GGML_OP_SCALE) {
                add0 = n0;
                n0   = a2;
                i_s1 = i_s2;
            }
        }
        if (i_s1 < 0 || cgraph->nodes[i_s1]->op != GGML_OP_SCALE) {
            return 0;
        }
        add = n0;
    }
    const bool at_post = cgraph->nodes[i_s1]->op == GGML_OP_SCALE && ggml_fn_l3_get(cgraph->nodes[i_s1]) == GGML_FN_L4_HCPOST;
    if (!at_post && n0->op != GGML_OP_RMS_NORM) {
        return 0;
    }
    // [TAG_FN_L6] GGML_CUDA_FN_L6_DIAG=1: the line of the first check that stops a combine (at_post) match
    const auto fn_l6_fail = [&](int line) {
        static std::atomic<int> shown{0};
        static const bool diag = getenv("GGML_CUDA_FN_L6_DIAG") != nullptr;
        // decode widths only (the s1 / ADD output has T columns), with the ops of the next computing nodes
        if (diag && at_post && cgraph->nodes[i_s1]->ne[1] <= 4 && shown.fetch_add(1) < 64) {
            std::string ops;
            for (int j = i_s1, k = 0; j >= 0 && k < 8; j = next(j), ++k) {
                ops += std::string(" ") + ggml_op_desc(cgraph->nodes[j]) + "(" + cgraph->nodes[j]->name + ")";
            }
            GGML_LOG_INFO("ggml_cuda: [TAG_FN_L6] hc combine not matched at node %d (%s) T %lld: ggml-cuda.cu line %d:%s\n", i,
                    n0->name, (long long) cgraph->nodes[i_s1]->ne[1], line, ops.c_str());
        }
        return 0;
    };

    int i_rms = i;
    if (at_post) {
        i_rms = next(next(next(next(i_s1))));
    }
    const int i_mul = next(i_rms);
    if (i_mul < 0 || cgraph->nodes[i_rms]->op != GGML_OP_RMS_NORM || cgraph->nodes[i_mul]->op != GGML_OP_MUL ||
            ggml_fn_l3_get(cgraph->nodes[i_mul]) != GGML_FN_L4_HC || !ggml_cuda_fn_l4_enabled()) {
        return fn_l6_fail(__LINE__);
    }
    static const bool trace = getenv("GGML_CUDA_FN_L4_TRACE") != nullptr;
    if (trace) {
        fprintf(stderr, "fn_l4 trace: match at %d (%s), rms %d mul %d\n", i, ggml_op_name(n0->op), i_rms, i_mul);
        fflush(stderr);
    }

    const auto f32c = [](const ggml_tensor * t) {
        return t != nullptr && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t);
    };
    // a node no launch writes: its only reader is the next node of the chain
    const auto inner = [cgraph](const ggml_tensor * t) {
        return ggml_cuda_fn_l4_uses(cgraph, t) == 1 && !(t->flags & GGML_TENSOR_FLAG_OUTPUT) && t->view_src == nullptr;
    };

    ggml_cuda_fn_l4_hc_chain c;
    ggml_tensor * rms = cgraph->nodes[i_rms];
    ggml_tensor * mul = cgraph->nodes[i_mul];
    const int64_t n_embd = mul->ne[0];
    const int64_t hc     = mul->ne[1];
    const int64_t T      = mul->ne[2];
    if (!f32c(mul) || mul->ne[3] != 1 || hc != 4 || T < 1 || T > 4 || n_embd % 32 != 0 || n_embd < 32 ||
            n_embd > (n_embd < 1024 ? 4*256 : 4*1024) || !ggml_are_same_shape(rms, mul) || rms->type != GGML_TYPE_F32 ||
            !inner(rms) || (mul->src[0] != rms && mul->src[1] != rms)) {
        return fn_l6_fail(__LINE__);
    }
    const ggml_tensor * w_norm = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    if (w_norm == nullptr || w_norm->type != GGML_TYPE_F32 || w_norm->nb[0] != sizeof(float) || w_norm->ne[0] != n_embd ||
            w_norm->ne[1] != hc || w_norm->ne[2] != 1 || w_norm->ne[3] != 1) {
        return fn_l6_fail(__LINE__);
    }

    const ggml_tensor * x_in = rms->src[0]; // what the norm reads without a combine
    if (at_post) {
        const ggml_tensor * s1   = cgraph->nodes[i_s1];
        const ggml_tensor * sig  = cgraph->nodes[next(i_s1)];
        const ggml_tensor * s2   = cgraph->nodes[next(next(i_s1))];
        ggml_tensor *       post = cgraph->nodes[next(next(next(i_s1)))];
        if (sig->op != GGML_OP_UNARY || ggml_get_unary_op(sig) != GGML_UNARY_OP_SIGMOID || sig->src[0] != s1 ||
                s2->op != GGML_OP_SCALE || s2->src[0] != sig || post->op != GGML_OP_DSV4_HC_POST || post->src[2] != s2 ||
                post->src[3] != nullptr || rms->src[0] != post || !f32c(s1) || !f32c(sig) || !f32c(s2) ||
                !f32c(s1->src[0]) || !inner(s1) || !inner(sig) || !inner(s2) || s1->ne[0] != hc || s1->ne[1] != T ||
                ggml_nelements(s1) != hc*T || !ggml_are_same_shape(s1, s1->src[0])) {
            return fn_l6_fail(__LINE__);
        }
        const ggml_tensor * bo  = post->src[0];
        const ggml_tensor * res = post->src[1];
        if (bo == nullptr || res == nullptr || bo->type != GGML_TYPE_F32 || res->type != GGML_TYPE_F32 ||
                post->type != GGML_TYPE_F32 || bo->ne[0] != n_embd || bo->ne[1] != T || bo->ne[2] != 1 || bo->ne[3] != 1 ||
                !ggml_are_same_shape(res, mul) || !ggml_are_same_shape(post, mul) || bo->nb[0] % sizeof(float) != 0 ||
                res->nb[0] % sizeof(float) != 0 || post->nb[0] % sizeof(float) != 0) {
            return fn_l6_fail(__LINE__);
        }
        // the ADD: two same-shaped f32 [n_embd, T] operands, its only reader the combine (its output is not written)
        if (add != nullptr) {
            const ggml_tensor * a0 = add->src[0];
            const ggml_tensor * a1 = add->src[1];
            if (bo != add || !f32c(add) || !inner(add) || a0 == nullptr || a1 == nullptr || a0->type != GGML_TYPE_F32 ||
                    a1->type != GGML_TYPE_F32 || !ggml_are_same_shape(a0, add) || !ggml_are_same_shape(a1, add) ||
                    a0->nb[0] != sizeof(float) || a1->nb[0] != sizeof(float)) {
                return fn_l6_fail(__LINE__);
            }
        }
        // [TAG_FN_L18_GPUFLOOR_HCADD] add0: two same-shaped f32 [n_embd, T] operands, its only reader add (not written)
        if (add0 != nullptr) {
            const ggml_tensor * b0 = add0->src[0];
            const ggml_tensor * b1 = add0->src[1];
            if (add == nullptr || !f32c(add0) || !inner(add0) || b0 == nullptr || b1 == nullptr || b0->type != GGML_TYPE_F32 ||
                    b1->type != GGML_TYPE_F32 || !ggml_are_same_shape(b0, add0) || !ggml_are_same_shape(b1, add0) ||
                    !ggml_are_same_shape(add0, add) || b0->nb[0] != sizeof(float) || b1->nb[0] != sizeof(float)) {
                return fn_l6_fail(__LINE__);
            }
        }
        c.add0 = add0;
        c.add  = add;
        c.s1   = s1;
        c.s2   = s2;
        c.post = post;
        x_in   = nullptr;
    } else if (x_in == nullptr || x_in->type != GGML_TYPE_F32 || x_in->nb[0] != sizeof(float) ||
            !ggml_are_same_shape(x_in, mul)) {
        return fn_l6_fail(__LINE__);
    }
    c.rms = rms;
    c.mul = mul;

    const int cc        = ggml_cuda_info().devices[cuda_ctx->device].cc;
    const int warp_size = ggml_cuda_info().devices[cuda_ctx->device].warp_size;
    const int64_t hc_dim = n_embd*hc;

    // down, then the inject when it reads the same norm
    const int i_down = next(i_mul);
    if (i_down < 0) {
        return fn_l6_fail(__LINE__);
    }
    ggml_tensor * down = cgraph->nodes[i_down];
    if (down->op != GGML_OP_MUL_MAT || ggml_cuda_fn_l4_base(down->src[1]) != mul || down->src[1]->ne[0] != hc_dim ||
            down->src[1]->ne[1] != T || !ggml_cuda_fn_l4_mmvq_node(down, cc, warp_size) || !inner(down) ||
            down->src[0]->ne[1] % 32 != 0 || down->src[0]->ne[1] > 1024) {
        return fn_l6_fail(__LINE__);
    }
    const int64_t lr = down->src[0]->ne[1];
    c.down = down;

    int i_ls = next(i_down);
    if (i_ls < 0) {
        return fn_l6_fail(__LINE__);
    }
    ggml_tensor * x = cgraph->nodes[i_ls];
    if (x->op == GGML_OP_MUL_MAT && x->src[0] != nullptr && x->src[0]->type == GGML_TYPE_F32 &&
            ggml_cuda_fn_l4_base(x->src[1]) == mul) {
        const ggml_tensor * wi = x->src[0];
        const ggml_tensor * yi = x->src[1];
        if (x->type != GGML_TYPE_F32 || x->nb[0] != sizeof(float) || wi->ne[0] != hc_dim || wi->ne[2] != 1 ||
                wi->ne[3] != 1 || wi->ne[1] < 1 || wi->ne[1] > 64 || !f32c(yi) || yi->ne[0] != hc_dim || yi->ne[1] != T ||
                x->ne[0] != wi->ne[1] || x->ne[1] != T || x->ne[2] != 1 || x->ne[3] != 1) {
            return fn_l6_fail(__LINE__);
        }
        c.inject = x;
        // only where ggml_cuda_mul_mat computes it with mul_mat_vec_f (f32 at T <= 3 on a GPU with fp32 MMA), else the op runs
        c.inject_fused = warp_size == 32 && wi->nb[0] == sizeof(float) && hc_dim % 2 == 0 &&
            (wi->nb[1]/sizeof(float)) % 2 == 0 && (yi->nb[1]/sizeof(float)) % 2 == 0 && hc_dim <= INT32_MAX/4 &&
            wi->buffer != nullptr && ggml_backend_buffer_get_usage(wi->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
            !ggml_cuda_op_mul_mat_use_fwht(x) && ggml_cuda_should_use_mmvf(wi->type, cc, ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size, wi->ne, wi->nb, T);
        if (c.inject_fused) {
            // the block size launch_mul_mat_vec_f_cuda picks
            int64_t bs_best    = warp_size;
            int64_t niter_best = (hc_dim + 2*warp_size - 1) / (2*warp_size);
            for (int64_t bs = 2*warp_size; bs <= 256; bs += warp_size) {
                const int64_t niter = (hc_dim + 2*bs - 1) / (2*bs);
                if (niter < niter_best) {
                    niter_best = niter;
                    bs_best    = bs;
                }
            }
            c.inject_bs = (int) bs_best;
        }
        i_ls = next(i_ls);
        if (i_ls < 0) {
            return fn_l6_fail(__LINE__);
        }
    }

    const int i_silu = next(i_ls);
    const int i_up   = next(i_silu);
    const int i_pre  = next(i_up);
    if (i_pre < 0) {
        return fn_l6_fail(__LINE__);
    }
    ggml_tensor * lo_s = cgraph->nodes[i_ls];
    ggml_tensor * silu = cgraph->nodes[i_silu];
    ggml_tensor * up   = cgraph->nodes[i_up];
    ggml_tensor * pre  = cgraph->nodes[i_pre];
    if (lo_s->op != GGML_OP_SCALE || lo_s->src[0] != down || !f32c(lo_s) || !f32c(down) || !inner(lo_s) ||
            silu->op != GGML_OP_UNARY || ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU || silu->src[0] != lo_s ||
            !f32c(silu) || !inner(silu) ||
            up->op != GGML_OP_MUL_MAT || up->src[1] != silu || !ggml_cuda_fn_l4_mmvq_node(up, cc, warp_size) ||
            up->src[0]->ne[0] != lr || up->src[0]->ne[1] != hc_dim || !inner(up) ||
            pre->op != GGML_OP_DSV4_HC_PRE || ggml_get_op_params_i32(pre, 1) == 0 || pre->type != GGML_TYPE_F32 ||
            pre->nb[0] % sizeof(float) != 0 || pre->ne[0] != n_embd || pre->ne[1] != T || pre->ne[2] != 1 ||
            pre->ne[3] != 1) {
        return fn_l6_fail(__LINE__);
    }
    // hc_pre reads the norm and the up output as [n_embd, hc, T], the latter through reshapes that only it reads
    const ggml_tensor * px = pre->src[0];
    const ggml_tensor * pw = pre->src[1];
    if (!f32c(px) || !f32c(pw) || !ggml_are_same_shape(px, mul) || !ggml_are_same_shape(pw, mul) ||
            ggml_cuda_fn_l4_base(px) != mul || ggml_cuda_fn_l4_base(pw) != up) {
        return fn_l6_fail(__LINE__);
    }
    for (const ggml_tensor * v = pw; v != up; v = v->src[0]) {
        if (ggml_cuda_fn_l4_uses(cgraph, v) != 1 || (v->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return fn_l6_fail(__LINE__);
        }
    }
    c.lo_s = lo_s;
    c.up   = up;
    c.pre  = pre;
    c.q8   = ggml_fn_l3_get(pre) == GGML_FN_L4_HCQ8;
    c.presync = (ggml_fn_l6_get(mul) & GGML_FN_L6_PRESYNC) != 0; // [TAG_FN_L6_PRESYNC]

    // the written tensors overlap neither each other nor what a launch reads after another one wrote
    const ggml_tensor * outs[4] = { c.post, mul, c.inject, pre };
    const ggml_tensor * ins[11] = { c.post ? c.post->src[0] : nullptr, c.post ? c.post->src[1] : nullptr,
                                    c.s1 ? c.s1->src[0] : nullptr, x_in, mul, w_norm, c.inject ? c.inject->src[0] : nullptr,
                                    c.add ? c.add->src[0] : nullptr, c.add ? c.add->src[1] : nullptr,
                                    c.add0 ? c.add0->src[0] : nullptr, c.add0 ? c.add0->src[1] : nullptr };
    const auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        const char * a0 = (const char *) a->data;
        const char * b0 = (const char *) b->data;
        return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
    };
    for (int o = 0; o < 4; ++o) {
        if (outs[o] == nullptr) {
            continue;
        }
        for (int o2 = o + 1; o2 < 4; ++o2) {
            if (outs[o2] != nullptr && overlap(outs[o], outs[o2])) {
                return fn_l6_fail(__LINE__);
            }
        }
        for (const ggml_tensor * in : ins) {
            if (in != nullptr && in != outs[o] && overlap(outs[o], in)) {
                return fn_l6_fail(__LINE__);
            }
        }
    }

    if (trace) {
        fprintf(stderr, "fn_l4 trace: matched %d..%d\n", i, i_pre);
        fflush(stderr);
    }
    ggml_cuda_fn_l4_hc_run(*cuda_ctx, c, cgraph, i, i_pre, [](ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
        ggml_cuda_mul_mat(ctx, dst->src[0], dst->src[1], dst);
    });
    static std::atomic<bool> noted{false};
    if (!noted.exchange(true)) {
        GGML_LOG_INFO("ggml_cuda: [TAG_FN_L4_HC] hc mixer in three launches (combine + norm + q8_1, down + inject, up + hc_pre)\n");
    }
    return i_pre - i;
}

// [TAG_FN_L6_HCCOMB] the hc mixers marked GGML_FN_L6_HCCOMB: the combine before the mixer (DSV4_HC_POST -> RMS_NORM ->
// MUL) reads the block output, the residual and the raw inject; the fused norm launch (fn-l4-hc.cu K1) reads them while
// it writes the combine and the norm, and the later launches write the inject and hc_pre. They stay allocated until
// the mixer's DSV4_HC_PRE, so that no output of the three launches can take their memory (ggml_cuda_fn_l4_try_hc
// refuses any overlap).
static void ggml_cuda_fn_l6_hc_deps(ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    if (params == nullptr || params->add_alloc_dep == nullptr) {
        return;
    }
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * mul = cgraph->nodes[i];
        if (mul->op != GGML_OP_MUL || ggml_fn_l3_get(mul) != GGML_FN_L4_HC || !(ggml_fn_l6_get(mul) & GGML_FN_L6_HCCOMB)) {
            continue;
        }
        ggml_tensor * rms = mul->src[0] != nullptr && mul->src[0]->op == GGML_OP_RMS_NORM ? mul->src[0] : mul->src[1];
        if (rms == nullptr || rms->op != GGML_OP_RMS_NORM) {
            continue;
        }
        ggml_tensor * post = rms->src[0];
        if (post == nullptr || post->op != GGML_OP_DSV4_HC_POST || post->src[2] == nullptr || post->src[2]->op != GGML_OP_SCALE) {
            continue;
        }
        ggml_tensor * sig = post->src[2]->src[0];
        ggml_tensor * s1  = sig != nullptr ? sig->src[0] : nullptr;
        if (s1 == nullptr || s1->op != GGML_OP_SCALE || ggml_fn_l3_get(s1) != GGML_FN_L4_HCPOST) {
            continue;
        }
        // the mixer's hc_pre: the first DSV4_HC_PRE after the MUL that reads it
        ggml_tensor * pre = nullptr;
        for (int j = i + 1; j < cgraph->n_nodes && j < i + 32 && pre == nullptr; ++j) {
            ggml_tensor * t = cgraph->nodes[j];
            if (t->op == GGML_OP_DSV4_HC_PRE && ggml_cuda_fn_l4_base(t->src[0]) == mul) {
                pre = t;
            }
        }
        if (pre == nullptr) {
            continue;
        }
        ggml_tensor * bo = post->src[0];
        ggml_tensor * b0 = bo != nullptr && bo->op == GGML_OP_ADD ? bo->src[0] : nullptr;
        ggml_tensor * b1 = bo != nullptr && bo->op == GGML_OP_ADD ? bo->src[1] : nullptr;
        // [TAG_FN_L18_GPUFLOOR_HCADD] an ADD operand of that ADD: its operands as well (K1 reads them when it folds it in);
        // only in the decode graphs the fused mixer takes (T <= 4): a prompt graph would keep two [n_embd, ubatch] tensors
        // longer for nothing (+126 MiB of the -ub 8192 compute buffer)
        ggml_tensor * bb = b0 != nullptr && b0->op == GGML_OP_ADD ? b0 : b1 != nullptr && b1->op == GGML_OP_ADD ? b1 : nullptr;
        if (mul->ne[2] > 4) {
            bb = nullptr;
        }
        ggml_tensor * keep[] = { bo, post->src[1], s1->src[0], b0, b1,
                                 fn_l18_hcadd_enabled() && bb != nullptr ? bb->src[0] : nullptr,
                                 fn_l18_hcadd_enabled() && bb != nullptr ? bb->src[1] : nullptr };
        for (ggml_tensor * t : keep) {
            if (t != nullptr) {
                params->add_alloc_dep(params->user_data, t, pre);
            }
        }
    }
}

// [TAG_FN_L18_GPUFLOOR_PFWAIT] GGML_CUDA_FN_L18_PFWAIT=1: the marked bridge waits load the L2 prefetch ranges themselves
bool ggml_cuda_fn_l18_pfwait_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_FN_L18_PFWAIT");
        const bool v = e != nullptr && atoi(e) != 0;
        if (v) {
            GGML_LOG_INFO("ggml_cuda: [TAG_FN_L18_GPUFLOOR_PFWAIT] bridge waits load the next layer's weights into L2 while they wait\n");
        }
        return v;
    }();
    return on;
}

// [TAG_FN_L6_L2PF] at a bridge wait marked GGML_FN_L6_L2PF: prefetch into L2 the weights in this device's buffers that
// the nodes up to the next MOE_HOST_POST read, in graph order, up to a budget (GGML_CUDA_FN_L6_L2PF_MB, default 3/4 of
// the L2). The device is idle while it waits for the host, and the next layer's chain then reads them from L2.
static void ggml_cuda_fn_l6_l2pf(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, int i) {
    static size_t budget[GGML_CUDA_MAX_DEVICES] = {};
    if (budget[cuda_ctx->device] == 0) {
        int l2 = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, cuda_ctx->device));
        const char * e = getenv("GGML_CUDA_FN_L6_L2PF_MB");
        budget[cuda_ctx->device] = e && atoi(e) > 0 ? (size_t) atoi(e) << 20 : (size_t) l2/4*3;
        GGML_LOG_INFO("ggml_cuda: [TAG_FN_L6_L2PF] bridge waits prefetch the next layer's weights into L2 (L2 %d MiB, budget %zu MiB)\n",
                l2 >> 20, budget[cuda_ctx->device] >> 20);
    }
    ggml_cuda_fn_l6_pf_ranges r;
    size_t total = 0;
    for (int j = i + 1; j < cgraph->n_nodes && r.nr < FN_L6_PF_MAX && total < budget[cuda_ctx->device]; ++j) {
        const ggml_tensor * node = cgraph->nodes[j];
        if (node->op == GGML_OP_MOE_HOST_POST) {
            break;
        }
        if (ggml_cuda_is_view_or_noop(node) || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        for (int s = 0; s < GGML_MAX_SRC && r.nr < FN_L6_PF_MAX; ++s) {
            const ggml_tensor * t = node->src[s];
            if (t == nullptr || t->data == nullptr || t->buffer == nullptr || !ggml_backend_buffer_is_cuda(t->buffer) ||
                    ggml_backend_buffer_get_usage(t->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
                    ((ggml_backend_cuda_buffer_context *) t->buffer->context)->device != cuda_ctx->device ||
                    ((uintptr_t) t->data & 15) != 0) {
                continue;
            }
            const char * p = (const char *) t->data;
            size_t       n = std::min(ggml_nbytes(t), budget[cuda_ctx->device] - total);
            bool seen = false;
            for (int k = 0; k < r.nr; ++k) {
                seen = seen || (p < r.p[k] + r.n[k] && r.p[k] < p + n);
            }
            if (seen || n < 4096) {
                continue;
            }
            r.p[r.nr] = p;
            r.n[r.nr] = n;
            r.nr++;
            total += n;
        }
    }
    static std::atomic<int> shown{0};
    if (getenv("GGML_CUDA_FN_L6_DIAG") && shown.fetch_add(1) < 4) {
        GGML_LOG_INFO("ggml_cuda: [TAG_FN_L6_L2PF] wait %d (%s): %d ranges, %.1f MiB\n", i, cgraph->nodes[i]->name, r.nr,
                total/1048576.0);
    }
    // [TAG_FN_L18_GPUFLOOR_PFWAIT] the wait launch loads them (real loads, until the host's result is there)
    if (ggml_cuda_fn_l18_pfwait_enabled()) {
        ggml_cuda_fn_l18_pf_set(cuda_ctx->device, cgraph->nodes[i], r.p, r.n, r.nr);
        return;
    }
    ggml_cuda_fn_l6_l2pf_launch(*cuda_ctx, r);
}

// [TAG_FN_L22_GLUQ8] (#30489) GGML_CUDA_FN_L22_GLUQ8=1: a GLU whose only consumer is an MMQ matmul (dense or MUL_MAT_ID)
// a few nodes ahead is computed inside that matmul's Q8_1 activation quantization; the GLU output is never written.
// Exact (the same silu / gelu expression in registers). #29948's gate/up/GLU fusion consumes its GLU first.
static bool ggml_cuda_fn_l22_gluq8() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_FN_L22_GLUQ8");
        return e != nullptr && atoi(e) != 0;
    }();
    return on;
}

static bool ggml_cuda_fn_l22_tensors_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    if (a->buffer == nullptr || b->buffer == nullptr || a->data == nullptr || b->data == nullptr) {
        return a == b;
    }
    const int64_t a_start = (int64_t) a->data;
    const int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);
    const int64_t b_start = (int64_t) b->data;
    const int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);
    return (b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end);
}

// the fork's own dispatch (ggml_cuda_mul_mat / ggml_cuda_mul_mat_id) would run dst on MMQ with src1 = glu
static bool ggml_cuda_fn_l22_glu_mmq_ok(const ggml_tensor * glu, const ggml_tensor * dst, const int cc, const int warp_size) {
    const ggml_tensor * weight = dst->src[0];
    const ggml_tensor * gate   = glu->src[0];
    const ggml_tensor * up     = glu->src[1] != nullptr ? glu->src[1] : gate;

    static constexpr std::array<ggml_glu_op, 4> valid_glu_ops = { GGML_GLU_OP_SWIGLU, GGML_GLU_OP_GEGLU, GGML_GLU_OP_SWIGLU_OAI, GGML_GLU_OP_SWIGLU_CLAMP };
    if (std::find(valid_glu_ops.begin(), valid_glu_ops.end(), ggml_get_glu_op(glu)) == valid_glu_ops.end()) {
        return false;
    }
    if (!ggml_is_quantized(weight->type) || dst->type != GGML_TYPE_F32) {
        return false;
    }
    // native FP4 quantizes the activation its own way
    if (blackwell_mma_available(cc) && (weight->type == GGML_TYPE_MXFP4 || weight->type == GGML_TYPE_NVFP4)) {
        return false;
    }
    // a padded weight buffer gets cleared on first use; the fused path must not skip that
    if (weight->buffer == nullptr || (ggml_backend_buffer_get_usage(weight->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
            ggml_nbytes(weight) != ggml_backend_buffer_get_alloc_size(weight->buffer, weight) && weight->view_src)) {
        return false;
    }
    // the fused quantizer reads gate / up with float4 loads at the layout of the GLU output
    const bool layout_ok =
        gate->type == GGML_TYPE_F32 && up->type == GGML_TYPE_F32 && glu->type == GGML_TYPE_F32 &&
        gate->nb[0] == sizeof(float) && up->nb[0] == sizeof(float) && glu->ne[0] % 4 == 0 &&
        ggml_is_contiguous(glu) && ggml_is_contiguous(gate) && ggml_is_contiguous(up) &&
        ggml_cuda_is_aligned(gate, sizeof(float4)) && ggml_cuda_is_aligned(up, sizeof(float4));
    if (!layout_ok) {
        return false;
    }
    if (dst->op == GGML_OP_MUL_MAT) {
        const int64_t ne11 = glu->ne[1];
        if (ggml_cuda_op_mul_mat_use_fwht(dst) || ggml_fn_l3_get(dst) == GGML_FN_L3_MMV || weight->ne[1] == 1 ||
                ggml_cuda_should_use_mmvf(weight->type, cc, ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size, weight->ne, weight->nb, ne11) ||
                ggml_cuda_should_use_mmf(weight->type, cc, warp_size, weight->ne, weight->nb, ne11, /*mul_mat_id =*/ false) ||
                ggml_cuda_should_use_mmsb(weight, glu, dst, cc) || ggml_cuda_should_use_mmqsn(weight, glu, dst, cc) ||
                ggml_cuda_should_use_mmvq(weight->type, cc, ne11, /*ne01 =*/ weight->ne[1])) {
            return false;
        }
        return ggml_cuda_should_use_mmq(weight->type, cc, ne11, /*n_experts =*/ 0);
    }
    // MUL_MAT_ID: one activation row per used expert (a broadcast activation takes the scatter quantizer)
    const ggml_tensor * ids = dst->src[2];
    if (ids == nullptr || glu->ne[1] != ids->ne[0] || glu->ne[1] == 1) {
        return false;
    }
    if (dst->ne[2] <= MMVQ_MAX_BATCH_SIZE && dst->ne[2] <= get_mmvq_mmid_max_batch(weight->type, cc)) {
        return false;
    }
    return ggml_cuda_should_use_mmq(weight->type, cc, glu->ne[2], /*n_experts =*/ weight->ne[2]);
}

// at node i (a GLU): compute its consumer now with the GLU fused in; true marks both nodes done
static bool ggml_cuda_fn_l22_try_gluq8(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const int i, std::vector<uint8_t> & fused) {
    ggml_tensor * glu = cgraph->nodes[i];
    const int cc        = ggml_cuda_info().devices[cuda_ctx->device].cc;
    const int warp_size = ggml_cuda_info().devices[cuda_ctx->device].warp_size;
    const int j_end     = std::min(i + 8, cgraph->n_nodes);
    for (int j = i + 1; j < j_end; ++j) {
        ggml_tensor * dst = cgraph->nodes[j];
        if ((dst->op != GGML_OP_MUL_MAT && dst->op != GGML_OP_MUL_MAT_ID) || dst->src[1] != glu) {
            continue;
        }
        // nothing between them may touch dst's memory or produce one of dst's other inputs
        bool safe = true;
        for (int k = i + 1; k < j && safe; ++k) {
            const ggml_tensor * n = cgraph->nodes[k];
            safe = !ggml_cuda_fn_l22_tensors_overlap(dst, n);
            for (int sidx = 0; sidx < GGML_MAX_SRC && safe; ++sidx) {
                const ggml_tensor * src = n->src[sidx];
                safe = src == nullptr || !ggml_cuda_fn_l22_tensors_overlap(dst, src);
            }
            for (int sidx = 0; sidx < GGML_MAX_SRC && safe; ++sidx) {
                for (const ggml_tensor * v = dst->src[sidx]; v != nullptr && safe; v = v->view_src) {
                    safe = v != n;
                }
            }
        }
        const int           idxs[2]      = { i, j };
        const ggml_op       ops[2]       = { GGML_OP_GLU, dst->op };
        const int           out_nodes[1] = { j };
        if (safe && ggml_can_fuse_subgraph_ext(cgraph, idxs, 2, ops, out_nodes, 1) && ggml_cuda_fn_l22_glu_mmq_ok(glu, dst, cc, warp_size)) {
            ggml_cuda_mul_mat_q(*cuda_ctx, dst->src[0], glu, dst->op == GGML_OP_MUL_MAT_ID ? dst->src[2] : nullptr, dst, nullptr, glu);
            if (ggml_cuda_mmvq_q8_reuse_enabled()) {
                ggml_cuda_mmvq_q8_cache_note_write(*cuda_ctx, dst); // [TAG_MMVQ_Q8_REUSE] dst is written now, not at j
            }
            fused[i] = 1;
            fused[j] = 1;
            return true;
        }
        break;
    }
    return false;
}

// [TAG_FLINT_AMAX_FUSION] build_amax_src1_f16 (llama-graph.cpp, the MoE down-projection rescale of Flint / Mistral4 /
// Mellum): ABS, POOL_1D, SCALE, LOG, SCALE, CEIL, CLAMP, SCALE, EXP, DIV(x, e) are 10 launches per MoE layer; they go
// into one (ggml_cuda_op_amax_src1_fused) when the 10 nodes follow each other and only e (EXP) and the DIV leave the
// group. Same float ops, so the outputs keep their bits. GGML_CUDA_AMAX_FUSION=0 turns it off.
static bool ggml_cuda_try_amax_src1_fusion(const ggml_cgraph * cgraph, int i, ggml_cuda_amax_src1_params & p) {
    static const bool on = [] { const char * e = getenv("GGML_CUDA_AMAX_FUSION"); return e == nullptr || atoi(e) != 0; }();
    static const ggml_op ops[] = { GGML_OP_UNARY, GGML_OP_POOL_1D, GGML_OP_SCALE, GGML_OP_LOG, GGML_OP_SCALE,
                                   GGML_OP_UNARY, GGML_OP_CLAMP, GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_DIV };
    constexpr int n = 10;
    if (!on || i + n > cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * const * nd = cgraph->nodes + i;
    for (int k = 0; k < n; ++k) {
        if (nd[k]->op != ops[k] || nd[k]->type != GGML_TYPE_F32) {
            return false;
        }
        if (k > 0 && k < n - 1 && nd[k]->src[0] != nd[k - 1]) {
            return false;
        }
    }
    if (ggml_get_unary_op(nd[0]) != GGML_UNARY_OP_ABS || ggml_get_unary_op(nd[5]) != GGML_UNARY_OP_CEIL ||
            ggml_get_unary_op(nd[8]) != GGML_UNARY_OP_EXP) {
        return false;
    }
    const ggml_tensor * x   = nd[0]->src[0];
    const ggml_tensor * e   = nd[8];
    const ggml_tensor * dst = nd[9];
    if (x->type != GGML_TYPE_F32 || dst->src[0] != x || dst->src[1] != e || !ggml_is_contiguous(x) ||
            !ggml_is_contiguous(e) || !ggml_is_contiguous(dst) || !ggml_are_same_shape(x, dst) || e->ne[0] != 1 ||
            e->ne[1] != x->ne[1] || e->ne[2] != x->ne[2] || e->ne[3] != x->ne[3]) {
        return false;
    }
    const int32_t * pp = (const int32_t *) nd[1]->op_params; // POOL_1D: op, k0, s0, p0
    if (pp[0] != GGML_OP_POOL_MAX || pp[1] != x->ne[0] || pp[2] != x->ne[0] || pp[3] != 0) {
        return false;
    }
    const int outs[2] = { i + 8, i + 9 };
    if (!ggml_can_fuse_subgraph(cgraph, i, n, ops, outs, 2)) {
        return false;
    }
    // e must not share memory with x or the DIV; the DIV may run in place on x (the kernel reads a row before writing it)
    auto range = [](const ggml_tensor * t) {
        const int64_t a = (int64_t) t->data;
        return std::make_pair(a, a + (int64_t) ggml_backend_buft_get_alloc_size(t->buffer->buft, t));
    };
    auto overlap = [](std::pair<int64_t, int64_t> a, std::pair<int64_t, int64_t> b) {
        return a.first < b.second && b.first < a.second;
    };
    const auto rx = range(x), re = range(e), rd = range(dst);
    if (overlap(re, rx) || overlap(re, rd) || (overlap(rd, rx) && dst->data != x->data)) {
        return false;
    }
    p.s0   = ggml_get_op_params_f32(nd[2], 0); p.b0   = ggml_get_op_params_f32(nd[2], 1);
    p.s1   = ggml_get_op_params_f32(nd[4], 0); p.b1   = ggml_get_op_params_f32(nd[4], 1);
    p.cmin = ggml_get_op_params_f32(nd[6], 0); p.cmax = ggml_get_op_params_f32(nd[6], 1);
    p.s2   = ggml_get_op_params_f32(nd[7], 0); p.b2   = ggml_get_op_params_f32(nd[7], 1);
    return true;
}

static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i, std::vector<uint8_t> * fn_l22_fused = nullptr) {

    if (ggml_cuda_fusion_disabled()) {
        return 0;
    }

    ggml_tensor * node = cgraph->nodes[i];

    // [TAG_FN_L4_QSA_QKV]
    if (node->op == GGML_OP_MUL_MAT && (ggml_fn_l4_qsa_get(node) & GGML_FN_L4_QSA_QKV) && ggml_cuda_fn_l4_qsa_enabled()) {
        const int n = ggml_cuda_fn_l4_try_qkv(cuda_ctx, cgraph, i);
        if (n > 0) {
            return n;
        }
    }
    // [TAG_FN_L18_GPUFLOOR_GDNOUT]
    if (node->op == GGML_OP_RMS_NORM && ggml_fn_l3_get(node) == GGML_FN_L18_GDNOUT && ggml_cuda_fn_l3_enabled()) {
        const int n = ggml_cuda_fn_l18_try_gdnout(cuda_ctx, cgraph, i);
        if (n > 0) {
            return n;
        }
    }
    // [TAG_FN_L18_GPUFLOOR_QKVZ]
    if (node->op == GGML_OP_MUL_MAT && (ggml_fn_l4_qsa_get(node) & GGML_FN_L18_QKVZ) && ggml_cuda_fn_l3_enabled()) {
        const int n = ggml_cuda_fn_l18_try_qkvz(cuda_ctx, cgraph, i);
        if (n > 0) {
            return n;
        }
    }
    // [TAG_FN_L4_QSA_SEL] [TAG_FN_L4_QSA_KVW]
    if (node->op == GGML_OP_TOP_K) {
        const int n = ggml_cuda_fn_l4_qsa_try_sel(*cuda_ctx, cgraph, i);
        if (n > 0) {
            return n;
        }
    }
    if (node->op == GGML_OP_TURBOT_SET_ROWS) {
        const int n = ggml_cuda_fn_l4_qsa_try_kvw(*cuda_ctx, cgraph, i);
        if (n > 0) {
            return n;
        }
    }
    // [TAG_FN_L4_QSA_POOL]
    if (node->op == GGML_OP_FILL) {
        const int n = ggml_cuda_fn_l4_qsa_try_pool(*cuda_ctx, cgraph, i);
        if (n > 0) {
            return n;
        }
    }
    // [TAG_FN_L4_HC] before the round-3 hc paths: the marked mixer as a whole
    if ((node->op == GGML_OP_SCALE && ggml_fn_l3_get(node) == GGML_FN_L4_HCPOST) || node->op == GGML_OP_RMS_NORM ||
            node->op == GGML_OP_ADD) {
        const int n = ggml_cuda_fn_l4_try_hc(cuda_ctx, cgraph, i);
        if (n > 0) {
            return n;
        }
    }

    // [TAG_FN_L3_GPU_HCFUSE]
    if (node->op == GGML_OP_SCALE && ggml_fn_l3_get(node) != GGML_FN_L3_NONE) {
        const int n = ggml_cuda_fn_l3_try_hc(cuda_ctx, cgraph, i);
        if (n > 0) {
            return n;
        }
    }

    // [TAG_FN_L3_GPU_GDNAB]
    if (node->op == GGML_OP_MUL_MAT && ggml_fn_l3_get(node) == GGML_FN_L3_GDNAB) {
        const int n = ggml_cuda_fn_l3_try_gdnab(cuda_ctx, cgraph, i);
        if (n > 0) {
            return n;
        }
    }

    // [TAG_FLINT_AMAX_FUSION] the MoE down-projection rescale (10 nodes) -> one launch
    if (node->op == GGML_OP_UNARY) {
        ggml_cuda_amax_src1_params ap;
        if (ggml_cuda_try_amax_src1_fusion(cgraph, i, ap)) {
            ggml_cuda_op_amax_src1_fused(*cuda_ctx, node->src[0], cgraph->nodes[i + 8], cgraph->nodes[i + 9], ap);
            return 9;
        }
    }

    // [TAG_GDN_PREP_FUSION] add -> softplus -> mul -> one launch
    if (node->op == GGML_OP_ADD && ggml_cuda_try_gdn_prep_fusion(cgraph, i)) {
        ggml_cuda_op_gdn_gate_prep(*cuda_ctx, node, cgraph->nodes[i + 2]);
        return 2;
    }

    // [TAG_RESID_NORM_FUSION] add -> rms_norm -> mul -> one launch (the add result is still written)
    if (node->op == GGML_OP_ADD && ggml_cuda_try_resid_norm_fusion(cgraph, i)) {
        ggml_cuda_op_rms_norm_resid_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    // [TAG_RMSNORM_SCALE_FUSION] rms_norm -> scale (GDN q/k l2 norm) -> one launch
    if (node->op == GGML_OP_RMS_NORM && ggml_cuda_try_rms_norm_scale_fusion(cgraph, i)) {
        ggml_cuda_op_rms_norm_scale_fused(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    // [TAG_CPY_CHAIN_FUSION] cpy, cpy, cpy ... with constant pointer deltas -> one launch
    if (node->op == GGML_OP_CPY) {
        int     n_chain = 0;
        int64_t dsrc = 0, ddst = 0;
        const int nodes_to_skip = ggml_cuda_try_cpy_chain_fusion(cgraph, i, n_chain, dsrc, ddst);
        if (nodes_to_skip > 0) {
            ggml_cuda_cpy_chain(*cuda_ctx, node->src[0], node->src[1], n_chain, dsrc, ddst);
            return nodes_to_skip;
        }
    }

    // [TAG_SYNC_1004] upstream #29184: shared experts fused into the routed MMVQ
    // [TAG_FN_L4_QSA_STREAMS] the QSA streams keep the node order: only GGML_CUDA_GRAPH_OPT's regions (which interleave
    // the branches) stop this fusion
    static const bool graph_opt_env = [] {
        const char * env = getenv("GGML_CUDA_GRAPH_OPT");
        return env != nullptr && atoi(env) == 1;
    }();
    if (node->op == GGML_OP_MUL_MAT_ID && (cuda_ctx->stream_context().concurrent_events.empty() || !graph_opt_env) &&
            ggml_cuda_match_shared_expert(cgraph, i, i + 3) &&
            (!ggml_cuda_fn_l21_shexp() || ggml_cuda_should_fuse_mul_mat_vec_q(cgraph->nodes[i + 2]->src[1]))) { // [TAG_FN_L21_SHEXP]
        const int outputs[] = { i + 2, i + 5 };
        if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, 6, outputs, 2)) {
            ggml_tensor * routed = cgraph->nodes[i + 2];
            ggml_tensor * shared = cgraph->nodes[i + 5];
            const ggml_tensor * up = routed->src[1];
            ggml_cuda_mm_fusion_args_host fusion{};
            fusion.gate = routed->src[0]->src[0];
            fusion.glu_op = ggml_get_glu_op(routed);
            fusion.glu_limit = ggml_get_op_params_f32(routed, 3);
            fusion.shared_up = shared->src[1]->src[0];
            fusion.shared_gate = shared->src[0]->src[0];
            fusion.shared_dst = shared;
            ggml_cuda_mul_mat_vec_q(*cuda_ctx, up->src[0], up->src[1], up->src[2], routed, &fusion);
            return 5;
        }
    }

    if (node->op == GGML_OP_MUL) {
        ggml_cuda_moe_weighted_reduction_match match;
        if (ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {
            const int output_idx = i + match.node_count - 1;
            if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, match.node_count, &output_idx, 1)) {
                ggml_cuda_op_moe_weighted_reduction(
                    *cuda_ctx, match.experts, match.expert_scale, match.weights, match.dst);
                return match.node_count - 1;
            }
        }
    }

    // gated_delta_net -> cpy: scatter recurrent-state snapshots into the cache
    if (node->op == GGML_OP_GATED_DELTA_NET) {
        ggml_cuda_gated_delta_net_fused_cache fused_state_cpy;
        const int nodes_to_skip = ggml_cuda_try_gdn_cache_fusion(cgraph, i, fused_state_cpy);
        if (nodes_to_skip > 0) {
#ifdef GGML_CUDA_DEBUG
            GGML_LOG_INFO("%s: fused gated_delta_net snapshot copies for %s (skipped %d nodes)\n",
                          __func__, node->name, nodes_to_skip);
#endif
            ggml_cuda_op_gated_delta_net_fused_cache(*cuda_ctx, node, fused_state_cpy);
            return nodes_to_skip;
        }
    }

    // [TAG_4C_GDN_REPLAY] gated_delta_net_replay -> cpy (state) -> cpy (ring): write both straight into the caches
    if (node->op == GGML_OP_GATED_DELTA_NET_REPLAY) {
        ggml_cuda_gdn_replay_out fused_out;
        const int nodes_to_skip = ggml_cuda_try_gdn_replay_fusion(cgraph, i, fused_out);
        if (nodes_to_skip > 0) {
            ggml_cuda_op_gated_delta_net_replay(*cuda_ctx, node, &fused_out);
            return nodes_to_skip;
        }
    }

    // [TAG_FN_L22_X_PRIOR] R-P12-1: the route prior's pattern (a marked MUL between the router's RESHAPE and ARGSORT) goes to
    // the bridge code: the prior / resident top-k, with the bridge post behind it in the same launch (moe-bridge.cu)
    if (node->op == GGML_OP_SOFT_MAX && i + 2 < cgraph->n_nodes && ggml_fn_l3_get(cgraph->nodes[i + 2]) == GGML_FN_L22_PRIOR) {
        const int n = ggml_cuda_fn_l22_try_prior(*cuda_ctx, cgraph, i);
        if (n > 0) {
            return n;
        }
    }

    // ssm_scan -> cpy: scatter recurrent-state snapshots into the cache
    if (node->op == GGML_OP_SSM_SCAN) {
        ggml_cuda_ssm_scan_fused_cache fused_state_cpy;
        const int nodes_to_skip = ggml_cuda_try_ssm_scan_cache_fusion(cgraph, i, fused_state_cpy);
        if (nodes_to_skip > 0) {
#ifdef GGML_CUDA_DEBUG
            GGML_LOG_INFO("%s: fused ssm_scan snapshot copies for %s (skipped %d nodes)\n",
                          __func__, node->name, nodes_to_skip);
#endif
            ggml_cuda_op_ssm_scan_fused_cache(*cuda_ctx, node, fused_state_cpy);
            return nodes_to_skip;
        }
    }

    //topk-moe
    if (cgraph->nodes[i]->op == GGML_OP_UNARY || cgraph->nodes[i]->op == GGML_OP_SOFT_MAX ||
            cgraph->nodes[i]->op == GGML_OP_ARGSORT) {
        ggml_cuda_topk_moe_args args;
        const bool              can_fuse = ggml_cuda_topk_moe_fusion(cgraph, i, args);
        std::vector<ggml_op>    ops;
        ops.reserve(13);  // max ops; avoids gcc -Wstringop-overflow false positive

        if (can_fuse) {
            const ggml_tensor * logits  = node->src[0];
            ggml_tensor *       weights = nullptr;
            ggml_tensor *       ids     = nullptr;
            const ggml_tensor * bias    = nullptr;
            const ggml_tensor * clamp   = nullptr;
            const ggml_tensor * scale   = nullptr;

            if (!args.delayed_softmax) {
                int out_nodes[2];  // nodes which can't be elided

                if (args.sigmoid) {
                    ops.insert(ops.end(), { GGML_OP_UNARY });
                } else if (args.sqrt_softplus) {
                    ops.insert(ops.end(), { GGML_OP_UNARY, GGML_OP_SQRT });
                } else {
                    ops.insert(ops.end(), { GGML_OP_SOFT_MAX });
                }
                ops.push_back(GGML_OP_RESHAPE);
                if (args.prob_bias) {
                    ops.push_back(GGML_OP_ADD);
                    bias = cgraph->nodes[i + ops.size() - 1]->src[1];
                }
                if (args.n_groups > 0) {
                    for (const auto & p : topk_moe_group_pattern) {
                        ops.push_back(p.op);
                    }
                }
                ops.insert(ops.end(), { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
                out_nodes[0] = i + ops.size() - 2;
                ids = cgraph->nodes[out_nodes[0]];

                if (args.norm) {
                    ops.insert(ops.end(),
                               { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
                    clamp = cgraph->nodes[i + ops.size() - 3];
                }
                if (args.scale) {
                    ops.insert(ops.end(), { GGML_OP_SCALE });
                    scale = cgraph->nodes[i + ops.size() - 1];
                }

                weights      = cgraph->nodes[i + ops.size() - 1];
                out_nodes[1] = i + ops.size() - 1;

                if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                        ggml_cuda_should_use_topk_moe(node, logits, weights, ids) &&
                        ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
                    const int n_post = ggml_cuda_fn_l4_try_topk_post(*cuda_ctx, cgraph, i + (int) ops.size() - 1, logits, weights,
                            ids, clamp, scale, bias, args); // [TAG_FN_L4_POST]
                    if (n_post > 0) {
                        return ops.size() - 1 + n_post;
                    }
                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args);
                    return ops.size() - 1;
                }
            } else if (!args.norm && !args.prob_bias) {
                //special case gpt-oss, no norm, no bias.
                ops.insert(ops.end(), { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_RESHAPE,
                                        GGML_OP_SOFT_MAX, GGML_OP_RESHAPE });
                weights                     = cgraph->nodes[i + 5];
                ids                         = cgraph->nodes[i + 1];
                const ggml_tensor * softmax = cgraph->nodes[i + 4];

                int out_nodes[2] = { i + 1, i + 5 };
                if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                        ggml_cuda_should_use_topk_moe(softmax, logits, weights, ids) &&
                        ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args);
                    return ops.size() - 1;
                }
            }
        }
    }

    //RoPE + view + set-rows
    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_tensor * rope     = cgraph->nodes[i];
        ggml_tensor * set_rows = cgraph->nodes[i + 2];

        ggml_cuda_op_rope_fused(*cuda_ctx, rope, set_rows);
        return 2;
    }

    // Snake activation: y = x + sin(a*x)^2 * inv_b
    // Naive 5-op decomposition emitted by frontends: mul -> sin -> sqr -> mul -> add
    if (ggml_can_fuse_subgraph(cgraph, i,
            { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD },
            { i + 4 })) {
        const ggml_tensor * mul0 = cgraph->nodes[i];
        const ggml_tensor * sqr  = cgraph->nodes[i + 2];
        const ggml_tensor * mul1 = cgraph->nodes[i + 3];
        ggml_tensor *       add  = cgraph->nodes[i + 4];

        // x carries the full activation shape, a is the broadcast operand
        const ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
        const ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

        // mul1 reads sqr and inv_b in either operand order
        const ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

        // closure check: the trailing add must read the same x as the leading mul
        const ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

        // Kernel iterates over total = T * C, so x and add must be 2D and
        // a / inv_b must collapse to [1, C, 1, 1]. Higher dims are not handled.
        const bool dim_ok   = (x->ne[2]   == 1 && x->ne[3]   == 1) &&
                              (add->ne[2] == 1 && add->ne[3] == 1) &&
                              (a->ne[2]   == 1 && a->ne[3]   == 1);
        const bool shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];

        // x is in the supported whitelist and every chain intermediate shares
        // x's type. launch_snake reads a and inv_b as const float *, so they
        // stay F32.
        const ggml_tensor * sin1 = cgraph->nodes[i + 1];
        const bool types_ok = (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
                              (a->type    == GGML_TYPE_F32) && (inv_b->type == GGML_TYPE_F32) &&
                              (mul0->type == x->type) && (sin1->type  == x->type) &&
                              (sqr->type  == x->type) && (mul1->type  == x->type) &&
                              (add->type  == x->type);

        // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
        const bool contig_ok = ggml_is_contiguous(x) && ggml_is_contiguous(add) &&
                               ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

        if (types_ok && shape_ok && dim_ok && contig_ok && x_in_add == x) {
            ggml_cuda_op_snake_fused(*cuda_ctx, x, a, inv_b, add);
            return 4;
        }
    }

    // multi-(add or mul)
    if (node->op == GGML_OP_ADD || node->op == GGML_OP_MUL) {
        int     n_fuse = 0;
        ggml_op ops[8];
        std::fill(ops, ops + 8, node->op);

        for (; n_fuse <= 6; ++n_fuse) {
            if (!ggml_can_fuse(cgraph, i + n_fuse, ops + n_fuse, 2)) {
                break;
            }
            if (cgraph->nodes[i + n_fuse] != cgraph->nodes[i + n_fuse + 1]->src[0]) {
                break;
            }
            if (!ggml_are_same_layout(cgraph->nodes[i + n_fuse]->src[1], cgraph->nodes[i + n_fuse + 1]->src[1])) {
                break;
            }
        }

        n_fuse++;

        // [TAG_FUSED_BINBCAST_ALIAS] the fused kernel reads every src while it writes the output of the last node. The
        // allocator only saw the unfused chain, so it can place that output over a src the chain no longer needed (a
        // broadcast src1 freed after its node, e.g. the MoE amax rescale factor between the down projection's two MULs:
        // Flint with one expert layer split by -fit, garbage at -ub 128). Shorten the chain until no src overlaps the
        // output; src0 of the first node may be the output itself (in place: each element is read before it is written).
        auto chain_alias_free = [&](int n) {
            const ggml_tensor * out = cgraph->nodes[i + n - 1];
            const char * o0 = (const char *) out->data;
            const char * o1 = o0 + ggml_nbytes(out);
            for (int j = 0; j < n; ++j) {
                for (int k = j == 0 ? 0 : 1; k < 2; ++k) {
                    const ggml_tensor * src = cgraph->nodes[i + j]->src[k];
                    const char * s0 = (const char *) src->data;
                    const char * s1 = s0 + ggml_nbytes(src);
                    if (s1 <= o0 || o1 <= s0) {
                        continue;
                    }
                    if (j == 0 && k == 0 && s0 == o0 && ggml_are_same_layout(src, out)) {
                        continue;
                    }
                    return false;
                }
            }
            return true;
        };
        while (n_fuse > 1 && !chain_alias_free(n_fuse)) {
            n_fuse--;
        }

        if (n_fuse > 1) {
            ggml_tensor fused_node;
            memcpy(&fused_node, node, sizeof(ggml_tensor));
            for (int j = 0; j < n_fuse - 1; ++j) {
                fused_node.src[j + 2] = cgraph->nodes[i + j + 1]->src[1];
            }
            fused_node.data = cgraph->nodes[i + n_fuse - 1]->data;
            if (node->op == GGML_OP_ADD) {
                ggml_cuda_op_fused_add(*cuda_ctx, &fused_node, n_fuse);
            } else {
                ggml_cuda_op_fused_mul(*cuda_ctx, &fused_node, n_fuse);
            }
            return n_fuse - 1;
        }
    }

    bool fused_mul_mat_vec = false;
    int  fused_node_count  = 0;

    auto get_mul_mat_scale = [](const ggml_tensor * scale_node, const ggml_tensor * mm_node) -> const ggml_tensor * {
        const bool scale_lhs_mm = scale_node->src[0] == mm_node;
        const bool scale_rhs_mm = scale_node->src[1] == mm_node;
        if (!scale_lhs_mm && !scale_rhs_mm) {
            return nullptr;
        }

        const ggml_tensor * scale = scale_lhs_mm ? scale_node->src[1] : scale_node->src[0];
        if (mm_node->src[0]->type != GGML_TYPE_NVFP4 || scale_node->type != GGML_TYPE_F32 ||
                scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != 1 ||
                !ggml_are_same_shape(scale_node, mm_node)) {
            return nullptr;
        }

        return scale;
    };

    auto get_mul_mat_id_scale = [](const ggml_tensor * reshape, const ggml_tensor * repeat, const ggml_tensor * getrows,
            const ggml_tensor * scale_node, const ggml_tensor * mm_node) -> const ggml_tensor * {
        if (repeat->src[0] != reshape || getrows->src[0] != repeat || getrows->src[1] != mm_node->src[2]) {
            return nullptr;
        }
        if (!((scale_node->src[0] == mm_node && scale_node->src[1] == getrows) ||
                (scale_node->src[0] == getrows && scale_node->src[1] == mm_node))) {
            return nullptr;
        }

        const ggml_tensor * scale = reshape->src[0];
        if (mm_node->src[0]->type != GGML_TYPE_NVFP4 || scale_node->type != GGML_TYPE_F32 ||
                scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != mm_node->src[0]->ne[2] ||
                !ggml_are_same_shape(scale_node, mm_node)) {
            return nullptr;
        }

        return scale;
    };

    auto get_bias_tensor = [](const ggml_tensor * bias_node, const ggml_tensor * mul_node, ggml_op op_bias) -> const ggml_tensor * {
        if (op_bias == GGML_OP_ADD) {
            if (bias_node->src[0] == mul_node) {
                return bias_node->src[1];
            }
            if (bias_node->src[1] == mul_node) {
                return bias_node->src[0];
            }
            return nullptr;
        }
        GGML_ASSERT(op_bias == GGML_OP_ADD_ID);
        GGML_ASSERT(bias_node->src[0] == mul_node);
        return bias_node->src[1];
    };

    // gate + glu + up, with optional scale/bias on both lanes.
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        if (op == GGML_OP_MUL_MAT) {
            for (const bool with_bias : { false, true }) {
                const int gate_idx       = i;
                const int gate_scale_idx = i + 1;
                const int gate_bias_idx  = with_bias ? i + 2 : -1;
                const int up_idx         = with_bias ? i + 3 : i + 2;
                const int up_scale_idx   = up_idx + 1;
                const int up_bias_idx    = with_bias ? up_idx + 2 : -1;
                const int glu_idx        = with_bias ? up_idx + 3 : up_idx + 2;

                const int out_nodes[] = { glu_idx };
                ggml_op ops[7];
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = bias_op;
                    ops[3] = op;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                    ops[6] = GGML_OP_GLU;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = op;
                    ops[3] = GGML_OP_MUL;
                    ops[4] = GGML_OP_GLU;
                }
                const int n_ops = with_bias ? 7 : 5;

                if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                        !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                    continue;
                }

                ggml_tensor * gate_n       = cgraph->nodes[gate_idx];
                ggml_tensor * gate_scale_n = cgraph->nodes[gate_scale_idx];
                ggml_tensor * gate_out_n   = with_bias ? cgraph->nodes[gate_bias_idx] : gate_scale_n;
                ggml_tensor * up_n         = cgraph->nodes[up_idx];
                ggml_tensor * up_scale_n   = cgraph->nodes[up_scale_idx];
                ggml_tensor * up_out_n     = with_bias ? cgraph->nodes[up_bias_idx] : up_scale_n;
                const ggml_tensor * glu = cgraph->nodes[glu_idx];

                if (!ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                        with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr, up_scale_n, gate_scale_n)) {
                    continue;
                }

                const ggml_tensor * gate_scale = get_mul_mat_scale(gate_scale_n, gate_n);
                const ggml_tensor * up_scale   = get_mul_mat_scale(up_scale_n, up_n);
                if (!gate_scale || !up_scale) {
                    continue;
                }

                const ggml_tensor * up_bias   = with_bias ? get_bias_tensor(up_out_n, up_scale_n, bias_op) : nullptr;
                const ggml_tensor * gate_bias = with_bias ? get_bias_tensor(gate_out_n, gate_scale_n, bias_op) : nullptr;
                if (with_bias && (!ggml_are_same_shape(gate_out_n->src[0], gate_out_n->src[1]) ||
                        !ggml_are_same_shape(up_out_n->src[0], up_out_n->src[1]))) {
                    continue;
                }

                const ggml_tensor * src0 = up_n->src[0];
                const ggml_tensor * src1 = up_n->src[1];
                const ggml_tensor * ids  = up_n->src[2];

                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate       = gate_n->src[0];
                fusion_data.x_bias     = up_bias;
                fusion_data.gate_bias  = gate_bias;
                fusion_data.x_scale    = up_scale;
                fusion_data.gate_scale = gate_scale;
                fusion_data.glu_op     = ggml_get_glu_op(glu);
                fusion_data.glu_limit  = ggml_get_op_params_f32(glu, 3);

                if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) {
                break;
            }
        } else {
            for (const bool with_bias : { false, true }) {
                const int gate_idx       = i;
                const int gate_scale_idx = i + 4;
                const int gate_bias_idx  = with_bias ? i + 5 : -1;
                const int up_idx         = with_bias ? i + 6 : i + 5;
                const int up_scale_idx   = up_idx + 4;
                const int up_bias_idx    = with_bias ? up_idx + 5 : -1;
                const int glu_idx        = with_bias ? up_idx + 6 : up_idx + 5;

                const int out_nodes[] = { glu_idx };
                ggml_op ops[13];
                if (with_bias) {
                    ops[0]  = op;
                    ops[1]  = GGML_OP_RESHAPE;
                    ops[2]  = GGML_OP_REPEAT;
                    ops[3]  = GGML_OP_GET_ROWS;
                    ops[4]  = GGML_OP_MUL;
                    ops[5]  = bias_op;
                    ops[6]  = op;
                    ops[7]  = GGML_OP_RESHAPE;
                    ops[8]  = GGML_OP_REPEAT;
                    ops[9]  = GGML_OP_GET_ROWS;
                    ops[10] = GGML_OP_MUL;
                    ops[11] = bias_op;
                    ops[12] = GGML_OP_GLU;
                } else {
                    ops[0]  = op;
                    ops[1]  = GGML_OP_RESHAPE;
                    ops[2]  = GGML_OP_REPEAT;
                    ops[3]  = GGML_OP_GET_ROWS;
                    ops[4]  = GGML_OP_MUL;
                    ops[5]  = op;
                    ops[6]  = GGML_OP_RESHAPE;
                    ops[7]  = GGML_OP_REPEAT;
                    ops[8]  = GGML_OP_GET_ROWS;
                    ops[9]  = GGML_OP_MUL;
                    ops[10] = GGML_OP_GLU;
                }
                const int n_ops = with_bias ? 13 : 11;

                if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                        !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                    continue;
                }

                ggml_tensor * gate_n       = cgraph->nodes[gate_idx];
                ggml_tensor * gate_scale_n = cgraph->nodes[gate_scale_idx];
                ggml_tensor * gate_out_n   = with_bias ? cgraph->nodes[gate_bias_idx] : gate_scale_n;
                ggml_tensor * up_n         = cgraph->nodes[up_idx];
                ggml_tensor * up_scale_n   = cgraph->nodes[up_scale_idx];
                ggml_tensor * up_out_n     = with_bias ? cgraph->nodes[up_bias_idx] : up_scale_n;
                const ggml_tensor * glu = cgraph->nodes[glu_idx];

                if (!ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                        with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr, up_scale_n, gate_scale_n)) {
                    continue;
                }

                const ggml_tensor * gate_scale = get_mul_mat_id_scale(cgraph->nodes[gate_idx + 1], cgraph->nodes[gate_idx + 2],
                        cgraph->nodes[gate_idx + 3], gate_scale_n, gate_n);
                const ggml_tensor * up_scale = get_mul_mat_id_scale(cgraph->nodes[up_idx + 1], cgraph->nodes[up_idx + 2],
                        cgraph->nodes[up_idx + 3], up_scale_n, up_n);
                if (!gate_scale || !up_scale) {
                    continue;
                }

                const ggml_tensor * up_bias   = with_bias ? get_bias_tensor(up_out_n, up_scale_n, bias_op) : nullptr;
                const ggml_tensor * gate_bias = with_bias ? get_bias_tensor(gate_out_n, gate_scale_n, bias_op) : nullptr;

                const ggml_tensor * src0 = up_n->src[0];
                const ggml_tensor * src1 = up_n->src[1];
                const ggml_tensor * ids  = up_n->src[2];

                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate       = gate_n->src[0];
                fusion_data.x_bias     = up_bias;
                fusion_data.gate_bias  = gate_bias;
                fusion_data.x_scale    = up_scale;
                fusion_data.gate_scale = gate_scale;
                fusion_data.glu_op     = ggml_get_glu_op(glu);
                fusion_data.glu_limit  = ggml_get_op_params_f32(glu, 3);

                if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) {
                break;
            }
        }

        if (ggml_cuda_can_fuse(cgraph, i, { op, bias_op, op, bias_op, GGML_OP_GLU }, {})) {
            ggml_tensor * glu         = cgraph->nodes[i + 4];
            ggml_tensor * gate_bias_n = glu->src[0];
            ggml_tensor * up_bias_n   = glu->src[1];

            //we don't assume the order for {gate, up}. Instead infer it from the bias tensor
            ggml_tensor * gate_n = nullptr;
            ggml_tensor * up_n   = nullptr;

            if (gate_bias_n->src[0] == cgraph->nodes[i] || gate_bias_n->src[1] == cgraph->nodes[i]) {
                gate_n = cgraph->nodes[i];
                up_n   = cgraph->nodes[i + 2];
            } else if (gate_bias_n->src[0] == cgraph->nodes[i + 2] || gate_bias_n->src[1] == cgraph->nodes[i + 2]) {
                gate_n = cgraph->nodes[i + 2];
                up_n   = cgraph->nodes[i];
            } else {
                continue;
            }

            const ggml_tensor * up_bias_tensor   = get_bias_tensor(up_bias_n, up_n, bias_op);
            const ggml_tensor * gate_bias_tensor = get_bias_tensor(gate_bias_n, gate_n, bias_op);

            if (!up_bias_tensor || !gate_bias_tensor) {
                continue;
            }

            // we don't support repeating adds
            if (bias_op == GGML_OP_ADD && (!ggml_are_same_shape(gate_bias_n->src[0], gate_bias_n->src[1]) ||
                                           !ggml_are_same_shape(up_bias_n->src[0], up_bias_n->src[1]))) {
                continue;
            }

            const ggml_tensor * src0 = up_n->src[0];
            const ggml_tensor * src1 = up_n->src[1];
            const ggml_tensor * ids  = up_n->src[2];

            if (ggml_cuda_should_fuse_mul_mat_vec_f(up_n)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate_n->src[0];
                fusion_data.x_bias    = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 5;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate_n->src[0];
                fusion_data.x_bias    = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 5;
                break;
            }
        } else if (ggml_cuda_can_fuse(cgraph, i, { op, op, GGML_OP_GLU }, {})) {
            ggml_tensor * glu  = cgraph->nodes[i + 2];
            ggml_tensor * gate = glu->src[0];
            ggml_tensor * up   = glu->src[1];

            bool ok = (gate == cgraph->nodes[i] && up == cgraph->nodes[i + 1]) ||
                      (gate == cgraph->nodes[i + 1] && up == cgraph->nodes[i]);

            if (!ok) {
                continue;
            }

            const ggml_tensor * src0 = up->src[0];
            const ggml_tensor * src1 = up->src[1];
            const ggml_tensor * ids  = up->src[2];

            if (ggml_cuda_should_fuse_mul_mat_vec_f(up)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_q(up)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }

            // [TAG_UPSTREAM_29948] prefill widths: gate/up matmuls + GLU in one MMQ kernel (GGML_CUDA_MMQ_GLU_FUSION)
            if (ggml_cuda_should_fuse_mul_mat_q(up, gate, glu)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    fused_mul_mat_vec = false;
    fused_node_count  = 0;

    // mul_mat + scale + optional bias
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        for (const bool with_bias : { false, true }) {
            const int n_ops = op == GGML_OP_MUL_MAT ? (with_bias ? 3 : 2) : (with_bias ? 6 : 5);
            const int out_nodes[] = { i + n_ops - 1 };
            ggml_op ops[6];
            if (op == GGML_OP_MUL_MAT) {
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = bias_op;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                }
            } else {
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                }
            }

            if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                    !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                continue;
            }

            ggml_tensor * mm_node    = cgraph->nodes[i];
            ggml_tensor * scale_node = op == GGML_OP_MUL_MAT ? cgraph->nodes[i + 1] : cgraph->nodes[i + 4];
            ggml_tensor * out_node   = with_bias ? cgraph->nodes[i + n_ops - 1] : scale_node;

            const ggml_tensor * scale = nullptr;
            if (op == GGML_OP_MUL_MAT) {
                scale = get_mul_mat_scale(scale_node, mm_node);
            } else {
                scale = get_mul_mat_id_scale(cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 3], scale_node, mm_node);
            }
            if (!scale) {
                continue;
            }

            const ggml_tensor * bias = with_bias ? get_bias_tensor(out_node, scale_node, bias_op) : nullptr;
            if (with_bias && !bias) {
                continue;
            }
            if (with_bias && bias_op == GGML_OP_ADD && !ggml_are_same_shape(out_node->src[0], out_node->src[1])) {
                continue;
            }
            if (with_bias && bias_op == GGML_OP_ADD_ID && out_node->src[2] != mm_node->src[2]) {
                continue;
            }

            const ggml_tensor * src0 = mm_node->src[0];
            const ggml_tensor * src1 = mm_node->src[1];
            const ggml_tensor * ids  = mm_node->src[2];

            ggml_cuda_mm_fusion_args_host fusion_data{};
            fusion_data.x_bias  = bias;
            fusion_data.x_scale = scale;

            if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, out_node, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = n_ops;
                break;
            }
        }
        if (fused_mul_mat_vec) {
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    // mul_mat + add
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        if (!ggml_can_fuse(cgraph, i, { op, bias_op })) {
            continue;
        }

        ggml_tensor * mm_node   = cgraph->nodes[i];
        ggml_tensor * bias_node = cgraph->nodes[i + 1];

        ggml_tensor * bias_tensor = nullptr;
        if (bias_op == GGML_OP_ADD) {
            if (bias_node->src[0] == mm_node) {
                bias_tensor = bias_node->src[1];
            } else if (bias_node->src[1] == mm_node) {
                bias_tensor = bias_node->src[0];
            } else {
                continue;
            }
        } else {
            if (bias_node->src[0] != mm_node) {
                continue;
            }
            bias_tensor = bias_node->src[1];
        }

        const ggml_tensor * src0 = mm_node->src[0];
        const ggml_tensor * src1 = mm_node->src[1];
        const ggml_tensor * ids  = mm_node->src[2];

        if (bias_op == GGML_OP_ADD_ID && bias_node->src[2] != ids) {
            continue;
        }

        if (bias_op == GGML_OP_ADD && !ggml_are_same_shape(bias_node->src[0], bias_node->src[1])) {
            continue;
        }

        ggml_cuda_mm_fusion_args_host fusion_data{};
        fusion_data.x_bias = bias_tensor;

        if (ggml_cuda_should_fuse_mul_mat_vec_f(mm_node)) {
            ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count  = 2;
            break;
        }

        if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
            ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count  = 2;
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 4]);
        return 4;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], nullptr);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, {})) {
        ggml_cuda_op_rms_norm_fused_add(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    // [TAG_FN_L3_GPU_Q8F] a MUL the qwen4exp graph marked (the hc norm): the same kernel, plus the q8_1 copy of its output
    if (node->op == GGML_OP_RMS_NORM && i + 1 < cgraph->n_nodes && ggml_fn_l3_get(cgraph->nodes[i + 1]) == GGML_FN_L3_Q8OUT &&
            ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {}) &&
            ggml_cuda_fn_l3_rms_norm_mul_q8(*cuda_ctx, node, cgraph->nodes[i + 1])) {
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
        ggml_cuda_op_rms_norm_fused(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    // [TAG_SYNC_1004] RMS_NORM + SCALE (upstream #29393) is dispatched by ggml_cuda_try_rms_norm_scale_fusion above
    // [TAG_RMSNORM_SCALE_FUSION]: same kernel (upstream's do_scale rms_norm_f32), plus the fork's alias check and its
    // TURBO_RMSNORM_SCALE_FUSION=0 switch, which a second dispatch here would bypass.

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        ggml_cuda_op_ssm_conv(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        ggml_cuda_op_ssm_conv(*cuda_ctx, node, /*bias_add_node=*/ nullptr, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SILU }) ||
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SIGMOID }) ||
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SOFTPLUS })) {
        ggml_cuda_op_unary_mul(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_SQR }, { GGML_UNARY_OP_RELU })) {
        ggml_cuda_op_relu_sqr(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_DSV4_HC_POST }, { GGML_UNARY_OP_SIGMOID })) {
        ggml_cuda_op_dsv4_hc_post(*cuda_ctx, cgraph->nodes[i + 3], node, cgraph->nodes[i + 2]);
        return 3;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE }, { GGML_UNARY_OP_TANH })) {
        ggml_cuda_op_softcap(*cuda_ctx, cgraph->nodes[i + 2], node);
        return 2;
    }

    // [TAG_FN_L22_GLUQ8] last: every other fusion that starts at this GLU keeps its place
    if (fn_l22_fused != nullptr && node->op == GGML_OP_GLU) {
        ggml_cuda_fn_l22_try_gluq8(cuda_ctx, cgraph, i, *fn_l22_fused);
    }

    return 0;
}

// [TAG_FN_L3_HOST_POKE] llama turns it on for qwen4exp (LLAMA_FN_HOST_POKE): an eager evaluation (no CUDA graph) ends
// with a stream query too, and helper threads may flush their copy streams (ggml_backend_cuda_stream_poke)
static std::atomic<int> g_ggml_cuda_fn_poke{0};

static bool ggml_cuda_fn_poke_on() {
    return g_ggml_cuda_fn_poke.load(std::memory_order_relaxed) != 0;
}

// one stream query: WDDM submits the stream's batched work now. "Not ready" is its normal answer here.
static void ggml_cuda_stream_query_flush(cudaStream_t stream) {
    const cudaError_t q = cudaStreamQuery(stream);
    if (q == cudaErrorNotReady) {
        (void) cudaGetLastError();
    } else if (q != cudaSuccess) {
        CUDA_CHECK(q);
    }
}

void ggml_backend_cuda_fn_set_poke(int on) {
    g_ggml_cuda_fn_poke.store(on != 0 ? 1 : 0, std::memory_order_relaxed);
}

void ggml_backend_cuda_stream_poke(ggml_backend_t backend) {
    if (!ggml_cuda_fn_poke_on() || backend == nullptr || !ggml_backend_is_cuda(backend)) {
        return;
    }
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_set_device(cuda_ctx->device);
    ggml_cuda_stream_query_flush(cuda_ctx->stream());
}

static void ggml_cuda_graph_evaluate_and_capture(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const bool use_cuda_graph, const bool cuda_graph_update_required, const void * graph_key) {
    bool graph_evaluated_or_captured = false;

    // flag used to determine whether it is an integrated_gpu
    const bool integrated            = ggml_cuda_info().devices[cuda_ctx->device].integrated;

    ggml_cuda_stream_context & stream_ctx = cuda_ctx->stream_context();
    bool                         is_concurrent_event_active = false;
    ggml_cuda_concurrent_event * concurrent_event           = nullptr;
    bool                         should_launch_concurrent_events = false;

#ifdef USE_CUDA_GRAPH
    // [TAG_UPSTREAM_30804] a capture ends a chunk every chunk_size nodes and launches it, so the GPU runs the first chunks
    // while the rest is captured; a replay launches the chunks one after the other (same kernels, same order, one stream).
    // Not with a hostfunc-mode MoE bridge: its post forks an aux stream that only a later wait node joins.
    ggml_cuda_graph * chunk_graph = use_cuda_graph && cuda_graph_update_required ? cuda_ctx->cuda_graph(graph_key) : nullptr;
    const int         chunk_size  = chunk_graph != nullptr && !ggml_cuda_moe_bridge_hostfunc_any() ? ggml_cuda_graph_chunk_size() : 0;
    int               chunk_end   = chunk_size;
    size_t            n_chunks    = 0;
#endif // USE_CUDA_GRAPH

    const auto try_launch_concurrent_event = [&](const ggml_tensor * node) {
        if (stream_ctx.concurrent_events.find(node) != stream_ctx.concurrent_events.end()) {
            concurrent_event = &stream_ctx.concurrent_events[node];

            is_concurrent_event_active = true;

            GGML_LOG_DEBUG("Launching %d streams at %s\n", concurrent_event->n_streams, node->name);

            cudaStream_t main_stream = cuda_ctx->stream();  // this should be stream 0
            GGML_ASSERT(cuda_ctx->curr_stream_no == 0);
            CUDA_CHECK(cudaEventRecord(concurrent_event->fork_event, main_stream));

            for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                cudaStream_t stream = cuda_ctx->stream(cuda_ctx->device, i);
                CUDA_CHECK(cudaStreamWaitEvent(stream, concurrent_event->fork_event));
            }
        }
    };

    while (!graph_evaluated_or_captured) {
        // Only perform the graph execution if CUDA graphs are not enabled, or we are capturing the graph.
        // With the use of CUDA graphs, the execution will be performed by the graph launch.
        if (!use_cuda_graph || cuda_graph_update_required) {
            [[maybe_unused]] int prev_i = 0;

            // [TAG_MMVQ_Q8_REUSE] a q8_1 copy never outlives the evaluation that made it
            const bool mmvq_q8_reuse = ggml_cuda_mmvq_q8_reuse_enabled();
            if (mmvq_q8_reuse) {
                ggml_cuda_mmvq_q8_cache_reset(*cuda_ctx);
            }

            if (stream_ctx.concurrent_events.size() > 0) {
                should_launch_concurrent_events = true;
                for (const auto & [tensor, event] : stream_ctx.concurrent_events) {
                    should_launch_concurrent_events = should_launch_concurrent_events && event.is_valid();
                }
            }

            if (should_launch_concurrent_events) {
                // Restore original node order within each concurrent region to enable fusion within streams

                std::unordered_map<const ggml_tensor *, int> node_to_idx;
                node_to_idx.reserve(cgraph->n_nodes);
                for (int i = 0; i < cgraph->n_nodes; ++i) {
                    node_to_idx[cgraph->nodes[i]] = i;
                }

                for (auto & [fork_node, event] : stream_ctx.concurrent_events) {
                    // Find positions of all nodes from this event in the current graph
                    std::vector<int> positions;
                    positions.reserve(event.original_order.size());

                    bool all_found = true;
                    for (const ggml_tensor * orig_node : event.original_order) {
                        auto it = node_to_idx.find(orig_node);
                        if (it != node_to_idx.end()) {
                            positions.push_back(it->second);
                        } else {
                            all_found = false;
                            break;
                        }
                    }

                    if (!all_found || positions.size() != event.original_order.size()) {
                        continue;
                    }

                    // Sort positions to get contiguous range
                    std::vector<int> sorted_positions = positions;
                    std::sort(sorted_positions.begin(), sorted_positions.end());

                    bool is_contiguous = true;
                    for (size_t i = 1; i < sorted_positions.size(); ++i) {
                        if (sorted_positions[i] != sorted_positions[i-1] + 1) {
                            is_contiguous = false;
                            break;
                        }
                    }

                    if (!is_contiguous) {
                        continue;
                    }

                    // Restore original order at the sorted positions
                    int start_pos = sorted_positions[0];
                    for (size_t i = 0; i < event.original_order.size(); ++i) {
                        cgraph->nodes[start_pos + i] = const_cast<ggml_tensor *>(event.original_order[i]);
                    }
                }
            } else {
                // [TAG_FN_L4_QSA_STREAMS] say once that a graph's regions were dropped (a write overlap or a cross-stream source)
                if (!stream_ctx.concurrent_events.empty()) {
                    static std::atomic<bool> told{false};
                    if (!told.exchange(true)) {
                        GGML_LOG_INFO("ggml_cuda: [TAG_FN_L4_QSA] STREAMS: %zu concurrent regions failed the overlap check, the graph runs on one stream\n",
                                stream_ctx.concurrent_events.size());
                    }
                }
                stream_ctx.concurrent_events.clear();
            }

            // [TAG_FN_L22_GLUQ8] nodes a fusion computed ahead of their place (only with one stream)
            std::vector<uint8_t> fn_l22_fused;
            if (ggml_cuda_fn_l22_gluq8() && stream_ctx.concurrent_events.empty()) {
                fn_l22_fused.assign(cgraph->n_nodes, 0);
            }

            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
#ifdef USE_CUDA_GRAPH
                // [TAG_UPSTREAM_30804] fused nodes are skipped together, so a cut never splits a fusion; a concurrent region
                // must join before its capture can end
                if (chunk_size > 0 && i >= chunk_end && !is_concurrent_event_active) {
                    cudaGraph_t cuda_graph = nullptr;
                    CUDA_CHECK(cudaStreamEndCapture(cuda_ctx->stream(), &cuda_graph));
                    ggml_cuda_graph_exec_chunk(cuda_ctx, chunk_graph, n_chunks++, cuda_graph);
                    ggml_cuda_graph_poke(cuda_ctx); // [TAG_FN_R4_POKE] the chunk starts while the next one is captured
                    CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
                    chunk_end = i + chunk_size;
                }
#endif // USE_CUDA_GRAPH
                if (!fn_l22_fused.empty() && fn_l22_fused[i]) { // [TAG_FN_L22_GLUQ8]
                    prev_i = i;
                    continue;
                }
                if (is_concurrent_event_active) {
                    GGML_ASSERT(concurrent_event);

                    if (node == concurrent_event->join_node) {
                        cuda_ctx->curr_stream_no = 0;
                        for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                            // Wait on join events of forked streams in the main stream
                            CUDA_CHECK(cudaEventRecord(concurrent_event->join_events[i - 1],
                                                       cuda_ctx->stream(cuda_ctx->device, i)));
                            CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), concurrent_event->join_events[i - 1]));
                        }

                        is_concurrent_event_active = false;
                        concurrent_event           = nullptr;
                    } else {
                        // [TAG_FN_L4_QSA_STREAMS] the QSA regions map the computed nodes only: a no-op between fork and
                        // join (a view, the scheduler's allocation dependency nodes) computes nothing on any stream
                        const auto it_map = concurrent_event->stream_mapping.find(node);
                        GGML_ASSERT(it_map != concurrent_event->stream_mapping.end() || ggml_cuda_is_view_or_noop(node));
                        if (it_map != concurrent_event->stream_mapping.end()) {
                            cuda_ctx->curr_stream_no = it_map->second;
                        }
                        GGML_LOG_DEBUG("Setting stream no to %d for node %s\n", cuda_ctx->curr_stream_no, node->name);
                    }
                } else if (i - prev_i > 1) {
                    //the previous node was fused
                    const ggml_tensor * prev_node = cgraph->nodes[i - 1];
                    try_launch_concurrent_event(prev_node);

                    if (is_concurrent_event_active) {
                        // [TAG_FN_L4_QSA_STREAMS] find, not operator[]: an unmapped no-op must not enter the mapping
                        const auto it_map = concurrent_event->stream_mapping.find(node);
                        GGML_ASSERT(it_map != concurrent_event->stream_mapping.end() || ggml_cuda_is_view_or_noop(node));
                        if (it_map != concurrent_event->stream_mapping.end()) {
                            cuda_ctx->curr_stream_no = it_map->second;
                        }
                        GGML_LOG_DEBUG("Setting stream no to %d for node %s\n", cuda_ctx->curr_stream_no, node->name);
                    }
                }

                prev_i = i;

                if (ggml_cuda_is_view_or_noop(node)) {
                    continue;
                }

                if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                    continue;
                }

                // [TAG_MMVQ_Q8_REUSE] drop the q8_1 copy before a node writes the bytes it was made from
                if (mmvq_q8_reuse) {
                    ggml_cuda_mmvq_q8_cache_note_write(*cuda_ctx, node);
                    cuda_ctx->mmvq_q8.cgraph   = cgraph;
                    cuda_ctx->mmvq_q8.node_idx = i;
                }

                // [TAG_FN_L3_GPU_IDXQ8] a q8_0 gather that only marked lightning indexers read: they read through it
                if (node->op == GGML_OP_GET_ROWS && ggml_cuda_fn_l3_idxq8_skip(cgraph, i)) {
                    continue;
                }

                // [TAG_FN_L6_L2PF] the prefetch goes in front of the wait (the wait itself is computed as before)
                if (node->op == GGML_OP_MOE_HOST_WAIT && ggml_fn_l3_get(node) == GGML_FN_L6_L2PF && cuda_ctx->curr_stream_no == 0) {
                    ggml_cuda_fn_l6_l2pf(cuda_ctx, cgraph, i);
                }

                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i, fn_l22_fused.empty() ? nullptr : &fn_l22_fused);
                if (!fn_l22_fused.empty() && fn_l22_fused[i]) {
                    continue; // [TAG_FN_L22_GLUQ8] the GLU went into its consumer's quantization
                }

                if (nodes_to_skip != 0) {
                    // [TAG_MMVQ_Q8_REUSE] the fused nodes wrote their outputs too
                    if (mmvq_q8_reuse) {
                        for (int k = 1; k <= nodes_to_skip && i + k < cgraph->n_nodes; ++k) {
                            ggml_cuda_mmvq_q8_cache_note_write(*cuda_ctx, cgraph->nodes[i + k]);
                        }
                    }
#ifdef GGML_CUDA_DEBUG
                    const int last_fused = i + nodes_to_skip;
                    GGML_LOG_INFO("nodes_fused: %d, first: %s (%s), last: %s (%s)\n",
                            nodes_to_skip + 1, ggml_op_name(node->op), node->name,
                            ggml_op_name(cgraph->nodes[last_fused]->op), cgraph->nodes[last_fused]->name);
#endif
                    i += nodes_to_skip;
                    continue;
                }
#ifndef NDEBUG
                // On integrated GPUs (APUs, e.g. RDNA3.5) the scheduler may place a
                // node's output on the host-visible buffer, which the compute path
                // handles. Allow that here, mirroring the src-tensor check below.
                assert(node->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                       (integrated && ggml_backend_buft_is_cuda_host(node->buffer->buft)));
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    if (node->src[j] != nullptr) {
                        assert(node->src[j]->buffer);
                        assert(node->src[j]->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                               (integrated && ggml_backend_buft_is_cuda_host(node->src[j]->buffer->buft)));
                    }
                }
#else
                GGML_UNUSED(integrated);
#endif  // NDEBUG

                bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
                if (!ok) {
                    GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
                }
                GGML_ASSERT(ok);

                if (!is_concurrent_event_active) {
                    try_launch_concurrent_event(node);
               }
            }

            // [TAG_MMVQ_Q8_REUSE] no graph node is being computed any more
            cuda_ctx->mmvq_q8.cgraph   = nullptr;
            cuda_ctx->mmvq_q8.node_idx = -1;
        }

#ifdef USE_CUDA_GRAPH
        ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
        if (use_cuda_graph && cuda_graph_update_required) { // End CUDA graph capture
            cudaGraph_t cuda_graph = nullptr;
            CUDA_CHECK(cudaStreamEndCapture(cuda_ctx->stream(), &cuda_graph));
            graph_evaluated_or_captured = true; // CUDA graph has been captured
            ggml_cuda_fn_l22_cap_end(cuda_ctx, graph); // [TAG_FN_L22_GRAPHKEEP] the pool offsets this capture uses

            {
                std::lock_guard<std::mutex> lock(ggml_cuda_lock);
                if (ggml_cuda_lock_counter.fetch_sub(1, std::memory_order_relaxed) == 1) {
                    ggml_cuda_lock_cv.notify_all();
                }
            }

            // [TAG_UPSTREAM_30804] the last chunk (the only one when the graph has <= chunk_size nodes): update or create its
            // executable outside the capture lock, as before (instance cap, OOM hook), and launch it; the earlier chunks
            // were launched during the capture. Chunks left over from a capture with more chunks are dropped.
            ggml_cuda_graph_exec_chunk(cuda_ctx, graph, n_chunks++, cuda_graph);
            for (size_t c = n_chunks; c < graph->instances.size(); ++c) {
                CUDA_CHECK(cudaGraphExecDestroy(graph->instances[c]));
            }
            graph->instances.resize(n_chunks);
        } else {
            graph_evaluated_or_captured = true; // ggml graph has been directly evaluated
        }
    }

    // [TAG_FN_L3_HOST_POKE] the kernels of an eager evaluation start now, not at the next driver call
    if (!use_cuda_graph && ggml_cuda_fn_poke_on()) {
        ggml_cuda_stream_query_flush(cuda_ctx->stream());
    }

#ifdef USE_CUDA_GRAPH
    // [TAG_FN_L21_RECAP] this call replays the instance as it is (the next call may capture again at once)
    cuda_ctx->cuda_graph(graph_key)->replayed_since_capture = use_cuda_graph && !cuda_graph_update_required;
#endif  // USE_CUDA_GRAPH

    if (use_cuda_graph) {
        ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
        // [TAG_UPSTREAM_30804] a capture launched its chunks already; a replay launches them here, in order
        // ([TAG_FN_L4_QSA_STREAMS] the instantiate flags keep the QSA side stream's priority; [TAG_FN_L22_GRAPHKEEP] the
        // instance cap and the OOM retry are in ggml_cuda_fn_l22_instantiate)
        if (!cuda_graph_update_required) {
            GGML_ASSERT(graph->has_instance());
            for (cudaGraphExec_t instance : graph->instances) {
                CUDA_CHECK(cudaGraphLaunch(instance, cuda_ctx->stream()));
            }
        }
        ggml_cuda_graph_poke(cuda_ctx); // [TAG_FN_R4_POKE]
#else
        GGML_UNUSED(graph_key);
        graph_evaluated_or_captured = true;
#endif  // USE_CUDA_GRAPH
    }
}

#ifdef USE_CUDA_GRAPH
static bool ggml_cuda_graph_set_enabled(ggml_backend_cuda_context * cuda_ctx, const void * graph_key) {
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

    if (!graph->has_instance()) {
        if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_VOLTA) {
            if (!graph->disable_due_to_gpu_arch) {
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to GPU architecture\n", __func__);
            }
            graph->disable_due_to_gpu_arch = true;
        }
    }

    return graph->is_enabled();
}
#endif // USE_CUDA_GRAPH

// [TAG_GRAPH_DUMP] TURBO_GRAPH_DUMP=1 prints every node of the first four distinct graph sizes this
// backend computes, once each, to stderr: op, name, type, shape, strides, contiguity and sources.
// Diagnostic only; a single static bool when the env is unset.
static void ggml_cuda_graph_dump_once(const ggml_cgraph * cgraph) {
    static const bool on = [] {
        const char * e = getenv("TURBO_GRAPH_DUMP");
        return e && e[0] == '1';
    }();
    if (!on) {
        return;
    }
    static std::mutex mtx;
    static std::set<int> seen;
    std::lock_guard<std::mutex> lock(mtx);
    if (seen.size() >= 4 || !seen.insert(cgraph->n_nodes).second) {
        return;
    }
    fprintf(stderr, "turbo-dump: graph n_nodes=%d\n", cgraph->n_nodes);
    auto desc = [](const ggml_tensor * t) {
        char buf[256];
        snprintf(buf, sizeof(buf), "%s %s [%lld,%lld,%lld,%lld] nb[%zu,%zu,%zu,%zu] c%d",
                 t->name, ggml_type_name(t->type),
                 (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
                 t->nb[0], t->nb[1], t->nb[2], t->nb[3], ggml_is_contiguous(t) ? 1 : 0);
        return std::string(buf);
    };
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * n = cgraph->nodes[i];
        std::string line = "turbo-dump: #" + std::to_string(i) + " " + ggml_op_desc(n) + " " + desc(n);
        for (int j = 0; j < GGML_MAX_SRC && n->src[j]; ++j) {
            line += " | src" + std::to_string(j) + "=" + desc(n->src[j]);
        }
        fprintf(stderr, "%s\n", line.c_str());
    }
    fflush(stderr);
}

static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    ggml_cuda_set_device(cuda_ctx->device);
    ggml_cuda_graph_dump_once(cgraph);

    bool use_cuda_graph             = false;
    bool cuda_graph_update_required = false;
    const void * graph_key = nullptr;

    // [TAG_GRAPH_HOST_PROBE] TURBO_GRAPH_HOST_PROBE=1: host time of this call per context (the key, the key plus the
    // compatibility and property checks, and the whole call including the launch or the eager evaluation), mean over
    // 512 calls. In a serial decode loop the GPU waits for all of it. Diagnostic only.
    static const bool host_probe = [] {
        const char * e = getenv("TURBO_GRAPH_HOST_PROBE");
        return e != nullptr && e[0] == '1';
    }();
    using host_probe_clock = std::chrono::steady_clock;
    host_probe_clock::time_point hp_t0, hp_t1, hp_t2;
    if (host_probe) {
        hp_t0 = host_probe_clock::now();
    }

#ifdef USE_CUDA_GRAPH
    // [TAG_FN_L22_GRAPHKEEP] host time of the calls inside a log window
    const bool    fn_l22_win = cuda_ctx->fn_l22_win_left > 0;
    const int64_t fn_l22_t0  = fn_l22_win ? ggml_time_us() : 0;

    int32_t graph_rows = 0;
    graph_key = ggml_cuda_graph_get_key(cuda_ctx, cgraph, &graph_rows); // [TAG_GRAPH_KEY_MEMO]
    if (host_probe) {
        hp_t1 = host_probe_clock::now();
    }

    // [TAG_FN_L22_GRAPHKEEP] a graph of more token rows than GGML_CUDA_FN_L22_GRAPH_MAXROWS runs eagerly, with no map entry
    const bool fn_l22_eager = ggml_cuda_fn_l22_graph_keep() >= 1 && graph_rows > ggml_cuda_fn_l22_graph_maxrows();
    if (fn_l22_eager) {
        graph_key = GGML_CUDA_FN_L22_EAGER_KEY;
    }

    // [TAG_UPSTREAM_26771] a legacy-pool flush freed buffers that captured graphs may write: capture them again
    {
        uint64_t n_flush = 0;
        for (int d = 0; d < GGML_CUDA_MAX_DEVICES; ++d) {
            for (int s = 0; s < GGML_CUDA_MAX_STREAMS; ++s) {
                n_flush += cuda_ctx->pools[d][s] ? cuda_ctx->pools[d][s]->flush_count() : 0;
            }
        }
        if (n_flush != cuda_ctx->pool_flush_seen) {
            cuda_ctx->pool_flush_seen = n_flush;
            if (!cuda_ctx->cuda_graphs.empty()) {
                GGML_LOG_WARN("%s: [TAG_UPSTREAM_26771] the CUDA pool gave buffers back to the driver: %zu CUDA graphs captured again\n",
                        __func__, cuda_ctx->cuda_graphs.size());
                cuda_ctx->cuda_graphs.clear();
            }
        }
    }

    // [TAG_FN_BRIDGE_GRAPHGEN] a MoE bridge was freed or changed its device view: a captured graph would replay the old
    // bridge's posts and waits (a new bridge with the same id and the same tensors passes the property check), so capture
    // them again
    {
        const uint32_t mb_gen = ggml_cuda_moe_bridge_generation();
        if (mb_gen != cuda_ctx->mb_gen_seen) {
            cuda_ctx->mb_gen_seen = mb_gen;
            if (!cuda_ctx->cuda_graphs.empty()) {
                GGML_LOG_INFO("%s: [TAG_FN_BRIDGE_GRAPHGEN] a MoE bridge was freed or changed: %zu CUDA graphs captured again\n",
                        __func__, cuda_ctx->cuda_graphs.size());
                cuda_ctx->cuda_graphs.clear();
            }
        }
    }

    ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);

    const bool gd_new_key = cuda_ctx->cuda_graphs.find(graph_key) == cuda_ctx->cuda_graphs.end(); // [TAG_FN_L7_GDIFF]
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
    // [TAG_CUDA_GRAPH_PROBE] why a call did or did not run as a CUDA graph.
    // Windows WDDM charges ~7us of CPU per kernel launch, and a decode step issues
    // thousands of them, so whether graphs engage is the single biggest lever on
    // decode speed. Any change in a node property (a tensor ne/nb or a src data
    // pointer) resets warmup and drops the step back to raw launches.
    // TURBO_GRAPH_PROBE=0 silences; it prints every 512 calls.
    int gp_reason = 0; // 0=graph used, 1=disabled, 2=incompatible, 3=props changed, 4=warmup, 5=eager rows [TAG_FN_L22_GRAPHKEEP]
    if (fn_l22_eager) {
        gp_reason = 5;
    } else if (graph->is_enabled()) {
        const bool graph_compatible = ggml_cuda_graph_check_compability_memo(cuda_ctx, cgraph); // [TAG_GRAPH_KEY_MEMO]
        if (graph_compatible) {
            bool nodes_changed = false; // [TAG_FN_L21_RECAP]
            const bool properties_changed = ggml_cuda_graph_update_required(cuda_ctx, cgraph, graph_key, &nodes_changed);

            if (!graph->warmup_complete) {
                // Warmup: need at least 2 calls with no property change on the 2nd call
                if (!properties_changed) {
                    graph->warmup_complete = true;
                    GGML_LOG_DEBUG("%s: CUDA graph warmup complete\n", __func__);
                    use_cuda_graph = true;
                    cuda_graph_update_required = true;
                } else {
                    gp_reason = 4;
                }
            } else {
                // Post-warmup: normal CUDA graph operation
                // [TAG_FN_L21_RECAP] (upstream #29768) LLAMA_FN_L21_RECAP (default on, 0 = off): after a replay of an unchanged
                // instance, changed properties with no node changing its op or starting / stopping a kernel are captured
                // again at once instead of two direct runs of warmup. Legacy pools can synchronize on OOM inside a
                // capture, so only with virtual-memory pools on every device. The same kernels either way.
                static const bool recap_on = [] {
                    const char * e = getenv("LLAMA_FN_L21_RECAP");
                    return !(e && e[0] == '0');
                }();
                bool recapture_safe = false;
#if defined(GGML_USE_VMM)
                recapture_safe = recap_on;
                for (int device = 0; device < ggml_cuda_info().device_count && recapture_safe; ++device) {
                    recapture_safe = ggml_cuda_info().devices[device].vmm;
                }
#endif // defined(GGML_USE_VMM)
                if (properties_changed && recapture_safe && graph->has_instance() && graph->replayed_since_capture &&
                        !nodes_changed) {
                    use_cuda_graph = true;
                    cuda_graph_update_required = true;
                } else if (properties_changed) {
                    // Properties changed - reset warmup, execute directly until stable again
                    graph->warmup_complete = false;
                    GGML_LOG_DEBUG("%s: CUDA graph warmup reset\n", __func__);
                    gp_reason = 3;
                } else {
                    use_cuda_graph = true;
                    cuda_graph_update_required = !graph->has_instance();
                }
            }
        } else {
            gp_reason = 2;
        }
    } else {
        gp_reason = 1;
    }
    if (host_probe) {
        hp_t2 = host_probe_clock::now();
    }
    // [TAG_FN_L7_GDIFF] TURBO_GRAPH_DIFF=1: a graph of > 1000 nodes that does not replay: its first node that differs (name,
    // op, shape) from the previous graph of > 1000 nodes this context ran, or the node count change (diagnostic)
    {
        static const bool gdiff2 = [] {
            const char * e = getenv("TURBO_GRAPH_DIFF");
            return e != nullptr && e[0] == '1';
        }();
        if (gdiff2 && cgraph->n_nodes > 1000) {
            struct sig { std::string name; int op; int64_t ne[4]; };
            static std::unordered_map<const void *, std::vector<sig>> last; // per context
            static std::atomic<int> n_logged{0};
            std::vector<sig> cur((size_t) cgraph->n_nodes);
            for (int i = 0; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * t = cgraph->nodes[i];
                cur[i].name = t->name;
                cur[i].op   = (int) t->op;
                memcpy(cur[i].ne, t->ne, sizeof(cur[i].ne));
            }
            // the previous graph of the same width (the input embedding's token count) in this context
            int64_t width = -1;
            for (const auto & g : cur) {
                if (g.name.rfind("model.input_embed", 0) == 0) {
                    width = g.ne[2];
                    break;
                }
            }
            auto & prev = last[(const void *) ((uintptr_t) cuda_ctx + (uintptr_t) (width & 0xff))];
            if (gp_reason != 0 && n_logged.load() < 600) {
                n_logged.fetch_add(1);
                int d = -1;
                const size_t n = std::min(prev.size(), cur.size());
                for (size_t i = 0; i < n; ++i) {
                    if (prev[i].name != cur[i].name || prev[i].op != cur[i].op || memcmp(prev[i].ne, cur[i].ne, sizeof(prev[i].ne)) != 0) {
                        d = (int) i;
                        break;
                    }
                }
                if (prev.empty() || d < 0) {
                    fprintf(stderr, "turbo-probe: graph-warm width %lld reason %d new key %d: nodes %zu -> %zu, %s\n", (long long) width,
                            gp_reason, gd_new_key ? 1 : 0, prev.size(), cur.size(), prev.empty() ? "first of its width" : "same shapes as before");
                } else {
                    fprintf(stderr, "turbo-probe: graph-warm width %lld reason %d new key %d: node %d '%s' op %d ne %lld,%lld,%lld,%lld -> '%s' ne %lld,%lld,%lld,%lld\n",
                            (long long) width, gp_reason, gd_new_key ? 1 : 0, d, prev[d].name.c_str(), prev[d].op,
                            (long long) prev[d].ne[0], (long long) prev[d].ne[1], (long long) prev[d].ne[2], (long long) prev[d].ne[3],
                            cur[d].name.c_str(), (long long) cur[d].ne[0], (long long) cur[d].ne[1], (long long) cur[d].ne[2], (long long) cur[d].ne[3]);
                }
            }
            prev.swap(cur);
        }
    }
    {
        static const bool probe_on = [] {
            const char * e = getenv("TURBO_GRAPH_PROBE");
            return !(e && e[0] == '0');
        }();
        if (probe_on) {
            static std::atomic<unsigned> n_call{0}, n_reason[6]{};
            n_reason[gp_reason].fetch_add(1, std::memory_order_relaxed);
            const unsigned n = n_call.fetch_add(1, std::memory_order_relaxed) + 1;
            if ((n & 511u) == 0u) {
                if (n_reason[5].load() == 0) {
                    fprintf(stderr,
                            "turbo-probe: cuda-graph calls=%u used=%u disabled=%u incompatible=%u props-changed=%u warmup=%u nodes=%d\n",
                            n, n_reason[0].load(), n_reason[1].load(), n_reason[2].load(),
                            n_reason[3].load(), n_reason[4].load(), cgraph->n_nodes);
                } else { // [TAG_FN_L22_GRAPHKEEP]
                    fprintf(stderr,
                            "turbo-probe: cuda-graph calls=%u used=%u disabled=%u incompatible=%u props-changed=%u warmup=%u eager-rows=%u nodes=%d\n",
                            n, n_reason[0].load(), n_reason[1].load(), n_reason[2].load(),
                            n_reason[3].load(), n_reason[4].load(), n_reason[5].load(), cgraph->n_nodes);
                }
                fflush(stderr);
            }
        }
        // [TAG_FN_L22_GRAPH2] GGML_CUDA_FN_L22_GRAPHDIAG=1: calls that did not replay, by node count and reason
        if (ggml_cuda_fn_l22_graph_diag()) {
            static std::atomic<unsigned> d_call{0}, d_raw[4][6]{};
            const int bucket = cgraph->n_nodes < 256 ? 0 : cgraph->n_nodes < 1024 ? 1 : cgraph->n_nodes < 4096 ? 2 : 3;
            if (gp_reason != 0) {
                d_raw[bucket][gp_reason].fetch_add(1, std::memory_order_relaxed);
            }
            if (((d_call.fetch_add(1, std::memory_order_relaxed) + 1) & 1023u) == 0u && ggml_cuda_fn_l22_passlog_on()) {
                std::string line;
                const char * bname[4] = { "<256", "<1024", "<4096", ">=4096" };
                for (int b = 0; b < 4; ++b) {
                    char buf[160];
                    snprintf(buf, sizeof(buf), " nodes%s: disabled %u incompat %u props %u warmup %u eager %u;", bname[b],
                            d_raw[b][1].load(), d_raw[b][2].load(), d_raw[b][3].load(), d_raw[b][4].load(), d_raw[b][5].load());
                    line += buf;
                }
                GGML_LOG_WARN("[TAG_FN_L22_GRAPH2] kind=graph raw calls by size and reason after %u calls:%s\n", d_call.load(), line.c_str());
            }
        }
    }

    // [TAG_FN_L22_GRAPHKEEP] counters (ggml_backend_cuda_graph_counters, the log window)
    {
        uint64_t * c = cuda_ctx->fn_l22_cnt;
        c[GGML_CUDA_FN_L22_CNT_CALLS]++;
        if (use_cuda_graph && !cuda_graph_update_required) {
            c[GGML_CUDA_FN_L22_CNT_REPLAY]++;
            graph->n_replay++;
            graph->n_replay_cap++;
        } else if (use_cuda_graph) {
            c[graph->has_instance() ? GGML_CUDA_FN_L22_CNT_RECAP : GGML_CUDA_FN_L22_CNT_CAPTURE]++;
        } else if (gp_reason == 5) {
            c[GGML_CUDA_FN_L22_CNT_EAGER_ROWS]++;
        } else if (gp_reason == 3 || gp_reason == 4) {
            c[GGML_CUDA_FN_L22_CNT_WARMUP]++;
        }
    }
#endif // USE_CUDA_GRAPH

    if (use_cuda_graph && cuda_graph_update_required) {
        // Start CUDA graph capture
        {
            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            ggml_cuda_lock_counter.fetch_add(1, std::memory_order_relaxed);
        }

#ifdef USE_CUDA_GRAPH
        ggml_cuda_fn_l22_cap_begin(cuda_ctx); // [TAG_FN_L22_GRAPHKEEP]
#endif // USE_CUDA_GRAPH
        CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
    }

    ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required, graph_key);

#ifdef USE_CUDA_GRAPH
    if (fn_l22_win) { // [TAG_FN_L22_GRAPHKEEP]
        cuda_ctx->fn_l22_cnt[GGML_CUDA_FN_L22_CNT_HOST_US] += (uint64_t) (ggml_time_us() - fn_l22_t0);
        ggml_cuda_fn_l22_window_tick(cuda_ctx);
    }
#endif // USE_CUDA_GRAPH

    if (host_probe) {
        const auto us = [](host_probe_clock::duration d) {
            return std::chrono::duration<double, std::micro>(d).count();
        };
        const host_probe_clock::time_point hp_t3 = host_probe_clock::now();
#ifndef USE_CUDA_GRAPH
        hp_t1 = hp_t0;
        hp_t2 = hp_t0;
#endif // USE_CUDA_GRAPH
        ggml_cuda_graph_host_probe & p = cuda_ctx->graph_host_probe;
        p.n++;
        p.key_us  += us(hp_t1 - hp_t0);
        p.chk_us  += us(hp_t2 - hp_t0);
        p.tot_us  += us(hp_t3 - hp_t0);
        p.n_graph += use_cuda_graph && !cuda_graph_update_required ? 1 : 0;
        if (p.n >= 512) {
#ifdef USE_CUDA_GRAPH
            const int memo_on = ggml_cuda_graph_key_memo_enabled() ? 1 : 0;
#else
            const int memo_on = 0;
#endif // USE_CUDA_GRAPH
            fprintf(stderr, "turbo-probe: graph-host %s ctx %p: %" PRIu64 " calls, nodes %d, key %.1f us, key+checks %.1f us, "
                    "call %.1f us, graph replays %" PRIu64 ", mmvq q8 reuse hit %" PRIu64 " miss %" PRIu64 " (memo %d, reuse %d)\n",
                    cuda_ctx->name.c_str(), (void *) cuda_ctx, p.n, cgraph->n_nodes, p.key_us / p.n, p.chk_us / p.n,
                    p.tot_us / p.n, p.n_graph, cuda_ctx->mmvq_q8.n_hit, cuda_ctx->mmvq_q8.n_miss,
                    memo_on, ggml_cuda_mmvq_q8_reuse_enabled() ? 1 : 0);
            fflush(stderr);
            p = {};
            cuda_ctx->mmvq_q8.n_hit  = 0;
            cuda_ctx->mmvq_q8.n_miss = 0;
        }
    }

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    CUDA_CHECK(cudaEventRecord((cudaEvent_t)event->context, cuda_ctx->stream()));
}

static void ggml_backend_cuda_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    if (ggml_backend_is_cuda(backend)) {
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), (cudaEvent_t)event->context, 0));
    } else {
#if 0
        // untested
        auto wait_fn = [](void * user_data) {
            ggml_backend_event_t event = (ggml_backend_event_t)user_data;
            ggml_backend_event_synchronize(event);
        };

        CUDA_CHECK(cudaLaunchHostFunc(cuda_ctx->stream(), wait_fn, event));
#endif
        GGML_ABORT("fatal error");
    }
}

static void ggml_backend_cuda_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    const bool disable_fusion = ggml_cuda_fusion_disabled();

    auto add_alloc_deps = [&](size_t start, size_t last_node) {

        for (size_t i = start; i < last_node; ++i) {
            params->add_alloc_dep(params->user_data, cgraph->nodes[i], cgraph->nodes[last_node]);

            for (int j = 0; j < GGML_MAX_SRC; ++j) {
                if (cgraph->nodes[i]->src[j]) {
                    params->add_alloc_dep(params->user_data, cgraph->nodes[i]->src[j], cgraph->nodes[last_node]);
                }
            }
        }
    };

    if (!disable_fusion) {
        // add alloc deps for performance positive fusions. This may increase the overall compute buffer size.
        // TODO: consolidate fusion paths in graph_optimize and graph_compute
        ggml_cuda_set_device(cuda_ctx->device);
        for (int i = 0; i + 5 < cgraph->n_nodes; ++i) {
            if (cgraph->nodes[i]->op != GGML_OP_MUL_MAT_ID) {
                continue;
            }
            for (int j = i + 3; j + 2 < cgraph->n_nodes; ++j) {
                if (cgraph->nodes[j]->op == GGML_OP_MUL_MAT_ID && cgraph->nodes[j + 1]->op == GGML_OP_MUL_MAT_ID) {
                    break;
                }
                if (cgraph->nodes[j]->op != GGML_OP_MUL_MAT || !ggml_cuda_match_shared_expert(cgraph, i, j)) {
                    continue;
                }
                // Group both outputs before allocation so the shared result cannot alias intervening nodes.
                std::rotate(cgraph->nodes + i + 3, cgraph->nodes + j, cgraph->nodes + j + 3);
                ggml_tensor * up = cgraph->nodes[i + 2]->src[1];
                params->add_alloc_dep(params->user_data, up->src[1], cgraph->nodes[i + 5]);
                params->add_alloc_dep(params->user_data, up->src[2], cgraph->nodes[i + 5]);
                i += 5;
                break;
            }
        }
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            // [TAG_UPSTREAM_29948] Keep the inputs of the fused gate/up/GLU MMQ kernel alive until the GLU so that they cannot alias its output (under op offload the weights are buffer copies).
            if (i + 2 < cgraph->n_nodes && cgraph->nodes[i + 2]->op == GGML_OP_GLU) {
                ggml_tensor * glu  = cgraph->nodes[i + 2];
                ggml_tensor * gate = glu->src[0];
                ggml_tensor * up   = glu->src[1];
                const bool ok = (gate == cgraph->nodes[i] && up == cgraph->nodes[i + 1]) || (gate == cgraph->nodes[i + 1] && up == cgraph->nodes[i]);
                if (ok && ggml_cuda_mul_mat_q_fusion_matches(up, gate, glu, cuda_ctx->device)) {
                    add_alloc_deps(i, i + 2);
                    i += 2;
                    continue;
                }
            }

            ggml_cuda_moe_weighted_reduction_match match;
            if (ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {
                params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.experts), match.dst);
                params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.weights), match.dst);
                if (match.expert_scale != nullptr) {
                    params->add_alloc_dep(
                        params->user_data, const_cast<ggml_tensor *>(match.expert_scale), match.dst);
                }
                i += match.node_count - 1;
            }

            // [TAG_FN_L22_X_PRIOR] R-P12-1: the route prior's pattern keeps its nodes allocated until its weights, as topk-moe's
            if (cgraph->nodes[i]->op == GGML_OP_SOFT_MAX && i + 2 < cgraph->n_nodes &&
                    ggml_fn_l3_get(cgraph->nodes[i + 2]) == GGML_FN_L22_PRIOR) {
                const int n = ggml_cuda_fn_l22_prior_nodes(cgraph, i);
                if (n > 0 && i + n < cgraph->n_nodes) {
                    add_alloc_deps(i, i + n);
                    i += n - 1;
                    continue;
                }
            }

            if (cgraph->nodes[i]->op == GGML_OP_UNARY || cgraph->nodes[i]->op == GGML_OP_SOFT_MAX ||
                    cgraph->nodes[i]->op == GGML_OP_ARGSORT) {
                ggml_cuda_topk_moe_args args;
                const bool              can_fuse = ggml_cuda_topk_moe_fusion(cgraph, i, args);
                std::vector<ggml_op>    ops;
                ops.reserve(13);  // max ops; avoids gcc -Wstringop-overflow false positive

                const ggml_tensor * node = cgraph->nodes[i];

                if (can_fuse) {
                    const ggml_tensor * logits  = node->src[0];
                    ggml_tensor *       weights = nullptr;
                    ggml_tensor *       ids     = nullptr;

                    if (!args.delayed_softmax) {
                        int out_nodes[2];  // nodes which can't be elided

                        if (args.sigmoid) {
                            ops.insert(ops.end(), { GGML_OP_UNARY });
                        } else if (args.sqrt_softplus) {
                            ops.insert(ops.end(), { GGML_OP_UNARY, GGML_OP_SQRT });
                        } else {
                            ops.insert(ops.end(), { GGML_OP_SOFT_MAX });
                        }
                        const int i_probs = i + (int) ops.size() - 1;  // last node of the gating activation

                        if (args.prob_bias) {
                            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW,
                                                    GGML_OP_GET_ROWS });
                            out_nodes[0] = i_probs + 4;
                        } else {
                            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
                            out_nodes[0] = i_probs + 3;
                        }
                        ids = cgraph->nodes[out_nodes[0]];

                        if (args.norm) {
                            ops.insert(ops.end(),
                                       { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
                        }
                        if (args.scale) {
                            ops.insert(ops.end(), { GGML_OP_SCALE });
                        }

                        weights      = cgraph->nodes[i + ops.size() - 1];
                        out_nodes[1] = i + ops.size() - 1;

                        if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                                ggml_cuda_should_use_topk_moe(node, logits, weights, ids)) {

                            add_alloc_deps(i, i + ops.size());
                            i += ops.size() - 1;
                        }
                    }
                }
            }
        }
    }

#ifdef USE_CUDA_GRAPH
    int32_t graph_rows = 0;
    const void * graph_key = ggml_cuda_graph_get_key(cuda_ctx, cgraph, &graph_rows); // uid 0 here: always hashed
    // [TAG_FN_L22_GRAPHKEEP] a graph that will run eagerly gets no map entry here either
    const bool use_cuda_graph = ggml_cuda_graph_set_enabled(cuda_ctx,
        ggml_cuda_fn_l22_graph_keep() >= 1 && graph_rows > ggml_cuda_fn_l22_graph_maxrows() ? GGML_CUDA_FN_L22_EAGER_KEY : graph_key);
#else
    const bool use_cuda_graph = false;
    GGML_UNUSED(cuda_ctx);
    GGML_UNUSED(cgraph);
#endif

    static bool enable_graph_optimization = [] {
        const char * env     = getenv("GGML_CUDA_GRAPH_OPT");
        return env != nullptr && atoi(env) == 1;
    }();

    // [TAG_FN_L4_QSA] the allocation dependencies of the qwen4exp IDXQ8 / SEL paths, and the two streams of its QSA
    // layers (STREAMS); a graph without the marks gets nothing
    ggml_cuda_fn_l4_qsa_deps(cgraph, params);
    ggml_cuda_fn_l6_hc_deps(cgraph, params); // [TAG_FN_L6_HCCOMB]
    ggml_cuda_fn_l4_qsa_list_prepare(cuda_ctx, cgraph); // [TAG_FN_L4_QSA_LIST]
    const bool fn_l4_streams = ggml_cuda_fn_l4_qsa_streams_wanted(cgraph);

    if (!enable_graph_optimization && !fn_l4_streams) {
        // [TAG_FN_L4_QSA_STREAMS] the concurrent events of an earlier graph must not outlive it
        if (!cuda_ctx->stream_context().concurrent_events.empty()) {
            cuda_ctx->stream_context().reset();
        }
        return;
    }

    ggml_cuda_stream_context & stream_context = cuda_ctx->stream_context();
    stream_context.reset();

    if (!use_cuda_graph) {
        return;
    }

    if (fn_l4_streams) {
        ggml_cuda_set_device(cuda_ctx->device);
        ggml_cuda_fn_l4_qsa_streams(cuda_ctx, cgraph, params);
        if (!stream_context.concurrent_events.empty() || !enable_graph_optimization) {
            return; // GGML_CUDA_GRAPH_OPT's own regions only for a graph without QSA streams
        }
    }

    ggml_cuda_set_device(cuda_ctx->device);

    // number of out-degrees for a particular node
    std::unordered_map<const ggml_tensor *, int> fan_out;
    // reverse mapping of node to index in the cgraph
    std::unordered_map<const ggml_tensor *, int> node_indices;

    const auto & is_noop = [](const ggml_tensor * node) -> bool {
        return ggml_is_empty(node) || node->op == GGML_OP_NONE || node->op == GGML_OP_RESHAPE ||
               node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE;
    };

    const auto & depends_on = [](const ggml_tensor * dst, const ggml_tensor * src) -> bool {
        for (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
            if (dst->src[s] == src) {
                return true;
            }
        }
        // implicit dependency if they view the same tensor
        const ggml_tensor * dst2 = dst->view_src ? dst->view_src : dst;
        const ggml_tensor * src2 = src->view_src ? src->view_src : src;
        if (dst2 == src2) {
            return true;
        }
        return false;
    };

    for (int node_idx = 0; node_idx < cgraph->n_nodes; node_idx++) {
        const ggml_tensor * node = cgraph->nodes[node_idx];
        node_indices[node]       = node_idx;

        if (is_noop(node)) {
            continue;
        }
        for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
            const ggml_tensor * src = cgraph->nodes[node_idx]->src[src_idx];
            //TODO: check why nrows > 1 fails
            if (node && !is_noop(node) && ggml_nrows(node) <= 1) {
                fan_out[src] += 1;
            }
        }
    }

    // Target Q, K, V for concurrency
    // this is a more general way to find nodes which can be candidates for concurrency (although it has not been tested for anything else):
    // 1. find fan-out (fork) nodes where the same input is used at least N times (in QKV, it would be "attn-norm")
    // 2. find the join node, where 2 or more of the outputs are required (in QKV, this would "KQ" or "flash-attn")
    // 3. account for all branches from the fork to the join
    // 4. To extend lifetimes of the tensors, we interleave the branches (see below for more details)
    // 5. save the original cgraph and restore it in graph_compute, to enable fusion within streams
    // See discussion: https://github.com/ggml-org/llama.cpp/pull/16991#issuecomment-3522620030

    const int min_fan_out = 3;
    const int max_fan_out = 3;

    // store {fork_idx, join_idx}
    std::vector<std::pair<int, int>> concurrent_node_ranges;

    for (const auto & [root_node, count] : fan_out) {
        if (count >= min_fan_out && count <= max_fan_out) {
            const int root_node_idx = node_indices[root_node];

            // only optimize for attn_norm
            // TODO: make this more generic
            if (!strstr(root_node->name, "attn_norm")) {
                continue;
            }

            bool is_part_of_event = false;
            for (const auto & [start, end] : concurrent_node_ranges) {
                if (root_node_idx >= start && root_node_idx <= end) {
                    is_part_of_event = true;
                }
            }

            if (is_part_of_event) {
                continue;
            }

            std::vector<std::vector<const ggml_tensor *>> nodes_per_branch;
            for (int i = root_node_idx + 1; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * node = cgraph->nodes[i];
                if (!is_noop(node) && depends_on(node, root_node)) {
                    nodes_per_branch.push_back({ node });
                }
            }

            GGML_ASSERT(nodes_per_branch.size() == (size_t) count);

            //find the join point
            const ggml_tensor * join_node = nullptr;

            const auto & belongs_to_branch = [&](const ggml_tensor *                      node,
                                                 const std::vector<const ggml_tensor *> & branch) -> bool {
                for (const ggml_tensor * n : branch) {
                    if (depends_on(node, n)) {
                        return true;
                    }
                }
                return false;
            };

            for (int i = root_node_idx + 1; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * curr_node = cgraph->nodes[i];

                int num_joins = 0;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    if (belongs_to_branch(curr_node, nodes_per_branch[branch_idx])) {
                        num_joins++;
                    }
                }

                if (num_joins >= 2) {
                    join_node = curr_node;
                    break;
                }

                bool found_branch = false;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    std::vector<const ggml_tensor *> & branch_vec = nodes_per_branch[branch_idx];
                    if (belongs_to_branch(curr_node, branch_vec)) {
                        //continue accumulating
                        if (std::find(branch_vec.begin(), branch_vec.end(), curr_node) == branch_vec.end()) {
                            branch_vec.push_back(curr_node);
                        }
                        found_branch = true;
                    }
                }

                if (!found_branch && is_noop(curr_node)) {
                    // we can put it in any branch because it will be ignored
                    nodes_per_branch[0].push_back({ curr_node });
                }
            }

            if (join_node) {
                //Create ggml_cuda_concurrent_event
                ggml_cuda_concurrent_event concurrent_event(nodes_per_branch.size());
                concurrent_event.join_node = join_node;

                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    for (const ggml_tensor * n : nodes_per_branch[branch_idx]) {
                        concurrent_event.stream_mapping[n] = branch_idx + 1;
                    }
                }

                int fork_node_idx = node_indices[root_node];
                int join_node_idx = node_indices[join_node];

                int       current_branch_idx = 0;
                int       current_node_idx   = fork_node_idx + 1;
                const int n_branches         = nodes_per_branch.size();

                int total_branch_nodes = 0;
                for (std::vector<const ggml_tensor *> branch_nodes : nodes_per_branch) {
                    total_branch_nodes += branch_nodes.size();
                }

                // there are other nodes in the middle which are unaccounted for
                // usually (cpy) nodes, then ignore this fork
                if (join_node_idx - fork_node_idx - 1 != total_branch_nodes) {
                    GGML_LOG_DEBUG(
                        "Skipping %s because the number of nodes in the middle is not equal to the total number of "
                        "branch nodes %d != %d\n",
                        root_node->name, join_node_idx - fork_node_idx - 1, total_branch_nodes);
                    continue;
                }

                // Save the original order of nodes in this region before interleaving
                // This is used later to restore grouping for fusion within streams
                concurrent_event.original_order.reserve(total_branch_nodes);
                for (int i = fork_node_idx + 1; i < join_node_idx; ++i) {
                    concurrent_event.original_order.push_back(cgraph->nodes[i]);
                }

                std::unordered_map<const ggml_tensor *, ggml_cuda_concurrent_event> & concurrent_events = cuda_ctx->stream_context().concurrent_events;
                GGML_ASSERT(concurrent_events.find(root_node) == concurrent_events.end());
                concurrent_events.emplace(root_node, std::move(concurrent_event));
                GGML_LOG_DEBUG("Adding stream at node %s %p\n", root_node->name, root_node);
                concurrent_node_ranges.emplace_back(fork_node_idx, join_node_idx);

                // interleave tensors to extend lifetimes so that ggml graph doesn't recycle them
                // example transformation:
                // [attn-norm, QMul, QNorm, QRope, KMul, KNorm, KRope, VMul, attn] ->
                // [attn-norm, QMul, KMul, VMul, QNorm, VNorm, QRope, KRope, attn]
                while (current_node_idx < join_node_idx) {
                    std::vector<const ggml_tensor *> & branch_nodes = nodes_per_branch[current_branch_idx];

                    bool has_node = false;
                    for (std::vector<const ggml_tensor *> branch_node : nodes_per_branch) {
                        has_node |= branch_node.size() > 0;
                    }

                    GGML_ASSERT(has_node);

                    if (branch_nodes.empty()) {
                        current_branch_idx = (current_branch_idx + 1) % n_branches;
                        continue;
                    }

                    cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                    current_node_idx++;
                    branch_nodes.erase(branch_nodes.begin());

                    // append all empty nodes
                    while (!branch_nodes.empty() && is_noop(branch_nodes.front())) {
                        cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                        current_node_idx++;
                        branch_nodes.erase(branch_nodes.begin());
                    }

                    current_branch_idx = (current_branch_idx + 1) % n_branches;
                }
            }
        }
    }
}

static const ggml_backend_i ggml_backend_cuda_interface = {
    /* .get_name                = */ ggml_backend_cuda_get_name,
    /* .free                    = */ ggml_backend_cuda_free,
    /* .set_tensor_async        = */ ggml_backend_cuda_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_cuda_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_cuda_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_cuda_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_cuda_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_cuda_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_cuda_graph_compute,
    /* .event_record            = */ ggml_backend_cuda_event_record,
    /* .event_wait              = */ ggml_backend_cuda_event_wait,
    /* .graph_optimize          = */ ggml_backend_cuda_graph_optimize,
};

static ggml_guid_t ggml_backend_cuda_guid() {
    static ggml_guid guid = { 0x2c, 0xdd, 0xe8, 0x1c, 0x65, 0xb3, 0x65, 0x73, 0x6a, 0x12, 0x88, 0x61, 0x1c, 0xc9, 0xdc, 0x25 };
    return &guid;
}

bool ggml_backend_is_cuda(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cuda_guid());
}

// [TAG_FN_L3_VRAM_TRIM] the temporary-memory pools of one CUDA backend: device bytes they hold, and the most they handed
// out at once since the last trim (or since they were created)
void ggml_backend_cuda_pool_stats(ggml_backend_t backend, size_t * reserved, size_t * hwm) {
    size_t r = 0;
    size_t h = 0;
    if (ggml_backend_is_cuda(backend)) {
        auto * ctx = (ggml_backend_cuda_context *) backend->context;
        for (int d = 0; d < GGML_CUDA_MAX_DEVICES; ++d) {
            for (int s = 0; s < GGML_CUDA_MAX_STREAMS; ++s) {
                if (ctx->pools[d][s]) {
                    r += ctx->pools[d][s]->reserved();
                    h  = std::max(h, ctx->pools[d][s]->used_hwm());
                }
            }
        }
    }
    if (reserved) {
        *reserved = r;
    }
    if (hwm) {
        *hwm = h;
    }
}

// [TAG_FN_L3_VRAM_TRIM] give the pools' device memory above keep back to the driver. Waits for the backend's streams; the
// captured CUDA graphs of the backend are dropped, because their kernels keep the pool addresses of the capture (the next
// runs capture again). Returns the bytes given back.
size_t ggml_backend_cuda_pool_trim(ggml_backend_t backend, size_t keep) {
    if (!ggml_backend_is_cuda(backend)) {
        return 0;
    }
    auto * ctx = (ggml_backend_cuda_context *) backend->context;
    for (int d = 0; d < GGML_CUDA_MAX_DEVICES; ++d) {
        for (int s = 0; s < GGML_CUDA_MAX_STREAMS; ++s) {
            if (ctx->streams[d][s]) {
                ggml_cuda_set_device(d);
                CUDA_CHECK(cudaStreamSynchronize(ctx->streams[d][s]));
            }
        }
    }
    ggml_cuda_set_device(ctx->device);
#ifdef USE_CUDA_GRAPH
    // [TAG_FN_L22_GRAPHKEEP] G2: with GGML_CUDA_FN_L22_GRAPHKEEP, a mark (ggml_backend_pool_mark) and pools that tracked every
    // capture, each pool keeps max(keep, mark) and only the instances whose capture used pool memory above what stays
    // mapped are dropped; otherwise every graph goes, as before
    bool g2 = ggml_cuda_fn_l22_graph_keep() >= 1 && ctx->fn_l22_mark_valid;
    for (int d = 0; d < GGML_CUDA_MAX_DEVICES && g2; ++d) {
        for (int s = 0; s < GGML_CUDA_MAX_STREAMS && g2; ++s) {
            g2 = ctx->pools[d][s] == nullptr || (d == ctx->device && ctx->pools[d][s]->cap_hwm() != (size_t) -1);
        }
    }
    ctx->fn_l22_mark_valid = false;
    const int n_before = ggml_cuda_fn_l22_graph_keep() >= 1 && !g2 ? ggml_cuda_fn_l22_count_instances(ctx) : 0;
    if (!g2) {
        ctx->cuda_graphs.clear();
    }
    for (auto & m : ctx->graph_key_memo) {
        m = ggml_cuda_graph_key_memo();
    }
#endif
    size_t freed = 0;
    for (int d = 0; d < GGML_CUDA_MAX_DEVICES; ++d) {
        for (int s = 0; s < GGML_CUDA_MAX_STREAMS; ++s) {
            if (ctx->pools[d][s]) {
                ggml_cuda_set_device(d);
                size_t keep_s = keep;
#ifdef USE_CUDA_GRAPH
                if (g2) {
                    keep_s = std::max(keep, ctx->fn_l22_keep_mark[s]);
                }
#endif
                freed += ctx->pools[d][s]->trim(keep_s);
                ctx->pools[d][s]->reset_hwm();
            }
        }
    }
    ggml_cuda_set_device(ctx->device);
#ifdef USE_CUDA_GRAPH
    if (ggml_cuda_fn_l22_graph_keep() >= 1) {
        int kept = 0, dropped = 0;
        if (g2) {
            const auto above = [ctx](bool known, const size_t * top) {
                if (!known) {
                    return true;
                }
                for (int s = 0; s < GGML_CUDA_MAX_STREAMS; ++s) {
                    const size_t mapped = ctx->pools[ctx->device][s] ? ctx->pools[ctx->device][s]->reserved() : 0;
                    if (top[s] > mapped) {
                        return true;
                    }
                }
                return false;
            };
            for (auto & [k, g] : ctx->cuda_graphs) {
                for (size_t v = 0; v < g->fn_l22_alt.size(); ) {
                    if (above(g->fn_l22_alt[v].pool_top_known, g->fn_l22_alt[v].pool_top)) {
                        g->fn_l22_alt[v].destroy();
                        g->fn_l22_alt.erase(g->fn_l22_alt.begin() + v);
                        dropped++;
                    } else {
                        ++v;
                        kept++;
                    }
                }
                if (!g->has_instance()) {
                    continue;
                }
                if (above(g->pool_top_known, g->pool_top)) {
                    g->fn_l22_drop_instance();
                    dropped++;
                } else {
                    kept++;
                }
            }
        } else {
            dropped = n_before;
        }
        ctx->fn_l22_cnt[GGML_CUDA_FN_L22_CNT_TRIM_KEPT]    += kept;
        ctx->fn_l22_cnt[GGML_CUDA_FN_L22_CNT_TRIM_DROPPED] += dropped;
        if (ggml_cuda_fn_l22_passlog_on()) {
            GGML_LOG_WARN("[TAG_FN_L22_PASSLOG] kind=graph ctx=%s %p trim: %.1f MiB given back, instances kept %d dropped %d%s\n",
                    ctx->name.c_str(), (void *) ctx, freed/1048576.0, kept, dropped, g2 ? "" : " (no mark or untracked pools: all graphs cleared)");
        }
        ggml_cuda_fn_l22_window_start(ctx, 16);
    }
#endif
    return freed;
}

// [TAG_FN_L22_GRAPHKEEP] proc "ggml_backend_pool_mark": the next trim keeps, per pool, what the period since the last mark
// handed out at most and what the captured instances use
static void ggml_backend_cuda_pool_mark(ggml_backend_t backend) {
    if (!ggml_backend_is_cuda(backend)) {
        return;
    }
    auto * ctx = (ggml_backend_cuda_context *) backend->context;
#ifdef USE_CUDA_GRAPH
    for (int s = 0; s < GGML_CUDA_MAX_STREAMS; ++s) {
        ggml_cuda_pool * p = ctx->pools[ctx->device][s].get();
        size_t m = p ? p->used_hwm() : 0;
        for (const auto & [k, g] : ctx->cuda_graphs) {
            if (g->has_instance() && g->pool_top_known) {
                m = std::max(m, g->pool_top[s]);
            }
            for (const auto & v : g->fn_l22_alt) {
                if (v.has_instance() && v.pool_top_known) {
                    m = std::max(m, v.pool_top[s]);
                }
            }
        }
        ctx->fn_l22_keep_mark[s] = m;
        if (p) {
            p->reset_hwm();
        }
    }
    ctx->fn_l22_mark_valid = true;
#else
    GGML_UNUSED(ctx);
#endif
}

// [TAG_FN_L22_GRAPHKEY] proc "ggml_backend_cuda_set_graph_key_mode": big > 0 hashes every dimension >= big as one value
static void ggml_backend_cuda_set_graph_key_mode(ggml_backend_t backend, uint32_t big) {
    if (!ggml_backend_is_cuda(backend)) {
        return;
    }
#ifdef USE_CUDA_GRAPH
    auto * ctx = (ggml_backend_cuda_context *) backend->context;
    ctx->fn_l22_key_big = big;
    for (auto & m : ctx->graph_key_memo) {
        m = ggml_cuda_graph_key_memo();
    }
#else
    GGML_UNUSED(big);
#endif
}

// [TAG_FN_L22_GRAPHKEEP] proc "ggml_backend_cuda_graph_counters": cumulative counters (ggml_cuda_fn_l22_cnt order)
static void ggml_backend_cuda_graph_counters(ggml_backend_t backend, uint64_t * out, int n) {
    if (out == nullptr || n <= 0) {
        return;
    }
    memset(out, 0, n*sizeof(uint64_t));
    if (!ggml_backend_is_cuda(backend)) {
        return;
    }
#ifdef USE_CUDA_GRAPH
    auto * ctx = (ggml_backend_cuda_context *) backend->context;
    ctx->fn_l22_cnt[GGML_CUDA_FN_L22_CNT_OBJECTS]   = ctx->cuda_graphs.size();
    ctx->fn_l22_cnt[GGML_CUDA_FN_L22_CNT_INSTANCES] = ggml_cuda_fn_l22_count_instances(ctx);
    memcpy(out, ctx->fn_l22_cnt, std::min(n, (int) GGML_CUDA_FN_L22_CNT_COUNT)*sizeof(uint64_t));
#endif
}

// [TAG_FN_L22_GRAPHKEEP] proc "ggml_backend_cuda_graph_window": one kind=graph log line after the next n calls
static void ggml_backend_cuda_graph_window(ggml_backend_t backend, int n_calls) {
    if (!ggml_backend_is_cuda(backend) || n_calls <= 0) {
        return;
    }
#ifdef USE_CUDA_GRAPH
    ggml_cuda_fn_l22_window_start((ggml_backend_cuda_context *) backend->context, n_calls);
#endif
}

// [TAG_FN_L3_VRAM_ACCOUNT] the context's per-thread stack limit (the driver grows the local memory to it for every
// resident thread and keeps it), the device-side malloc heap and the printf FIFO, and the resident threads of the device
void ggml_backend_cuda_limits(ggml_backend_t backend, size_t * stack, size_t * heap, size_t * fifo, size_t * threads) {
    size_t s = 0, h = 0, f = 0, t = 0;
    if (ggml_backend_is_cuda(backend)) {
        const int device = ((ggml_backend_cuda_context *) backend->context)->device;
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaDeviceGetLimit(&s, cudaLimitStackSize));
        CUDA_CHECK(cudaDeviceGetLimit(&h, cudaLimitMallocHeapSize));
        CUDA_CHECK(cudaDeviceGetLimit(&f, cudaLimitPrintfFifoSize));
        const int phys = ggml_cuda_get_physical_device(device);
        int n_sm = 0, n_thr = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&n_sm,  cudaDevAttrMultiProcessorCount, phys));
        CUDA_CHECK(cudaDeviceGetAttribute(&n_thr, cudaDevAttrMaxThreadsPerMultiProcessor, phys));
        t = (size_t) n_sm * (size_t) n_thr;
    }
    if (stack)   { *stack   = s; }
    if (heap)    { *heap    = h; }
    if (fifo)    { *fifo    = f; }
    if (threads) { *threads = t; }
}

int ggml_backend_cuda_get_device_count() {
    return ggml_cuda_info().device_count;
}

static std::string ggml_cuda_device_description(int device) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(device)));

    const ggml_cuda_device_info & info = ggml_cuda_info();
    std::string description = prop.name;
    if (info.device_count > info.physical_device_count) {
        description += " (dev p" + std::to_string(info.devices[device].physical_device) +
                       "/v" + std::to_string(info.devices[device].virtual_index) + ")";
    }
    return description;
}

void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size) {
    snprintf(description, description_size, "%s", ggml_cuda_device_description(device).c_str());
}

static int ggml_cuda_physical_device_share_count(int device) {
    const ggml_cuda_device_info & info = ggml_cuda_info();
    GGML_ASSERT(device >= 0 && device < info.device_count);
    return info.devices[device].physical_share_count;
}

void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total) {
    ggml_cuda_set_device(device);

    CUDA_CHECK(cudaMemGetInfo(free, total));

    // virtual devices sharing one physical GPU share its memory pool; split it between them
    const int share_count = ggml_cuda_physical_device_share_count(device);
    *free  /= share_count;
    *total /= share_count;
}

bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size) {
    if (!ggml_cuda_register_host_enabled()) {
        return false;
    }

#if CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA) || defined(GGML_USE_HIP)
    cudaError_t err = cudaHostRegister(buffer, size, cudaHostRegisterPortable | cudaHostRegisterReadOnly);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();

        GGML_LOG_DEBUG("%s: failed to register %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return false;
    }
    return true;
#else
    GGML_UNUSED(buffer);
    GGML_UNUSED(size);
    return false;
#endif // CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA)
}

void ggml_backend_cuda_unregister_host_buffer(void * buffer) {
    if (!ggml_cuda_register_host_enabled()) {
        return;
    }

    cudaError_t err = cudaHostUnregister(buffer);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
    }
}


// backend device

struct ggml_backend_cuda_device_context {
    int device;
    std::string name;
    std::string description;
    std::string pci_bus_id;
    int op_offload_min_batch_size;
};

static const char * ggml_backend_cuda_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_cuda_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->description.c_str();
}

#if defined(__linux__)
// Helper function to get available memory from /proc/meminfo for UMA systems
static bool ggml_backend_cuda_get_available_uma_memory(long * available_memory_kb, long * free_swap_kb) {
    FILE * meminfo_file = nullptr;
    // 2KB buffer for reading /proc/meminfo since it does not report size info, should be enough
    const size_t BUFFER_SIZE = 2048;
    auto file_buffer = std::make_unique<char[]>(BUFFER_SIZE);
    size_t bytes_read = 0;
    long huge_tlb_total_pages = -1;
    long huge_tlb_free_pages = -1;
    long huge_tlb_page_size = -1;

    if (available_memory_kb == nullptr || free_swap_kb == nullptr) {
        return false;
    }

    meminfo_file = fopen("/proc/meminfo", "r");
    if (meminfo_file == nullptr) {
        GGML_LOG_ERROR("%s: failed to open /proc/meminfo\n", __func__);
        return false;
    }

    // Read file into buffer
    bytes_read = fread(file_buffer.get(), 1, BUFFER_SIZE - 1, meminfo_file);
    fclose(meminfo_file);

    if (bytes_read == 0) {
        GGML_LOG_ERROR("%s: failed to read from /proc/meminfo\n", __func__);
        return false;
    }
    file_buffer[bytes_read] = '\0';

    *available_memory_kb = -1;
    *free_swap_kb = -1;

    // Parse the file buffer line by line
    char * line = file_buffer.get();
    char * line_next;
    while (line < file_buffer.get() + bytes_read) {
        // Find the end of the current line
        line_next = strchr(line, '\n');
        if (line_next != nullptr) {
            *line_next = '\0';
            line_next++;
        } else {
            line_next = file_buffer.get() + bytes_read;
        }

        long value;
        if (sscanf(line, "MemAvailable: %ld kB", &value) == 1) {
            *available_memory_kb = value;
        } else if (sscanf(line, "SwapFree: %ld kB", &value) == 1) {
            *free_swap_kb = value;
        } else if (sscanf(line, "HugePages_Total: %ld", &value) == 1) {
            huge_tlb_total_pages = value;
        } else if (sscanf(line, "HugePages_Free: %ld", &value) == 1) {
            huge_tlb_free_pages = value;
        } else if (sscanf(line, "Hugepagesize: %ld kB", &value) == 1) {
            huge_tlb_page_size = value;
        }

        line = line_next;
    }

    if (huge_tlb_total_pages != 0 && huge_tlb_total_pages != -1) {
        *available_memory_kb = huge_tlb_free_pages * huge_tlb_page_size;

        // Hugetlbfs pages are not swappable.
        *free_swap_kb = 0;
    }

    GGML_LOG_DEBUG("%s: final available_memory_kb: %ld\n", __func__, *available_memory_kb);
    return true;
}
#endif // defined(__linux__)

static void ggml_backend_cuda_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    ggml_cuda_set_device(ctx->device);
    cudaError_t err = cudaMemGetInfo(free, total);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        GGML_LOG_WARN("%s: cudaMemGetInfo failed (%s), returning 0/0\n", __func__, cudaGetErrorString(err));
        *free = 0;
        *total = 0;
        return;
    }

// ref: https://github.com/ggml-org/llama.cpp/pull/17368
#if defined(__linux__) && !defined(GGML_USE_HIP)
    // Check if this is a UMA (Unified Memory Architecture) system
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(ctx->device)));

    // Check if UMA is explicitly enabled via environment variable
    bool is_uma = prop.integrated > 0 || ggml_cuda_unified_memory_enabled();

    if (is_uma) {
        // For UMA systems (like DGX Spark), use system memory info
        long available_memory_kb = 0;
        long free_swap_kb = 0;

        if (ggml_backend_cuda_get_available_uma_memory(&available_memory_kb, &free_swap_kb) && available_memory_kb > 0) {
            *free = (size_t)available_memory_kb * 1024;
        } else {
            GGML_LOG_ERROR("%s: /proc/meminfo reading failed, using cudaMemGetInfo\n", __func__);
        }
    }
#endif // defined(__linux__) && !defined(GGML_USE_HIP)

    // virtual devices sharing one physical GPU share its memory pool; split it between them
    const int share_count = ggml_cuda_physical_device_share_count(ctx->device);
    *free  /= share_count;
    *total /= share_count;
}

static enum ggml_backend_dev_type ggml_backend_cuda_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *) dev->context;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(ctx->device)));

    return prop.integrated
        ? GGML_BACKEND_DEVICE_TYPE_IGPU
        : GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_cuda_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;

    props->name        = ggml_backend_cuda_device_get_name(dev);
    props->description = ggml_backend_cuda_device_get_description(dev);
    props->type        = ggml_backend_cuda_device_get_type(dev);
    props->device_id   = ctx->pci_bus_id.empty() ? nullptr : ctx->pci_bus_id.c_str();
    ggml_backend_cuda_device_get_memory(dev, &props->memory_free, &props->memory_total);

    const bool host_buffer = !ggml_cuda_pinned_memory_disabled();
#ifdef GGML_CUDA_NO_PEER_COPY
    bool events = false;
#else
    bool events = true;
#endif

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ events,
        /* .mmap_support          = */ props->type != GGML_BACKEND_DEVICE_TYPE_IGPU,
    };
}

static ggml_backend_t ggml_backend_cuda_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_init(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cuda_host_buffer_type();
}

// TODO: move these functions here
static bool ggml_backend_cuda_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    // check if all the sources are allocated on this device
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] && op->src[i]->buffer && ggml_backend_buft_is_cuda(op->src[i]->buffer->buft)) {
            ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)op->src[i]->buffer->buft->context;
            if (buft_ctx->device != dev_ctx->device) {
                return false;
            }
        }
    }

    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_XIELU:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    if (op->src[0]->type == GGML_TYPE_BF16 && ggml_get_unary_op(op) == GGML_UNARY_OP_XIELU) {
                        return false;
                    }
                    return op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_BF16;
                default:
                    return false;
            }
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    if (op->src[0]->type == GGML_TYPE_BF16 &&
                            (ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU_OAI || ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU_CLAMP)) {
                        return false;
                    }
                    if (op->src[0]->type == GGML_TYPE_F16 && ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU_OAI) {
                        return false;
                    }
                    return ggml_is_contiguous_1(op->src[0]);
                default:
                    return false;
            }
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            {
                struct ggml_tensor * a = op->src[0];
                struct ggml_tensor * b = op->src[1];
                if (a->nb[0] != ggml_element_size(a) || b->nb[0] != ggml_element_size(b)) {
                    return false; // TODO this could in principle be implemented though currently there is no use case.
                }
                if (b->type == GGML_TYPE_F16 && a->type != GGML_TYPE_F16 && !ggml_cuda_op_mul_mat_use_fwht(op)) {
                    return false;
                }
                if (op->op == GGML_OP_MUL_MAT_ID && ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
                    return false;
                }
#ifdef GGML_USE_MUSA
                const int cc = ggml_cuda_info().devices[dev_ctx->device].cc;
                if (b->ne[2]*b->ne[3] > 1 && !ggml_is_transposed(a) && !ggml_is_transposed(b)) {
                    if (GGML_CUDA_CC_IS_QY1(cc) && op->op == GGML_OP_MUL_MAT &&
                            a->type == GGML_TYPE_F16 && b->type == GGML_TYPE_F16) {
                        return false;
                    }
                    if (GGML_CUDA_CC_IS_QY2(cc) && op->op == GGML_OP_MUL_MAT_ID &&
                            a->type == GGML_TYPE_Q2_K && b->type == GGML_TYPE_F32) {
                        return false;
                    }
                }
#endif // GGML_USE_MUSA
                switch (a->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_MXFP4:
                    case GGML_TYPE_NVFP4:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_K:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_IQ4_XS:
                    case GGML_TYPE_BF16:
                        return true;
                    default:
                        return false;
                }
            }
        case GGML_OP_OUT_PROD:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_GET_ROWS:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_F32:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_I32:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ4_XS:
                        return true;
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_MXFP4:
                        // 32-value sub-blocks, the row size does not guarantee
                        // the QK_K super-blocks the get_rows kernel iterates on
                        return op->src[0]->ne[0] % QK_K == 0;
                    default:
                        return false;
                }
            }
        case GGML_OP_GET_ROWS_BACK:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->ne[2] == 1 && op->ne[3] == 1;
        case GGML_OP_SET_ROWS:
            {
                // turbo types require head_dim divisible by appropriate group size
                if ((op->type == GGML_TYPE_TURBO3_0 || op->type == GGML_TYPE_TURBO2_0) && op->src[0]->ne[0] % 64 != 0) {
                    return false;
                }
                // turbo4 block size is 128, so head_dim must be divisible by 128
                if (op->type == GGML_TYPE_TURBO4_0 && op->src[0]->ne[0] % 128 != 0) {
                    return false;
                }
                // [TAG_TURBO4P] turbo4p packs 8 WHT groups into one 1024-element block, so
                // the row must be a whole number of blocks, not merely group-aligned.
                // [TAG_TURBO5P512] each split-plane type against its own block length.
                if ((op->type == GGML_TYPE_TURBO4P_0 || op->type == GGML_TYPE_TURBO5P_0) && op->src[0]->ne[0] % 1024 != 0) {
                    return false;
                }
                if (op->type == GGML_TYPE_TURBO5P512_0 && op->src[0]->ne[0] % 512 != 0) {
                    return false;
                }
                return (
                           (
                               (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16 ||
                               op->type == GGML_TYPE_Q4_0 || op->type == GGML_TYPE_Q4_1 || op->type == GGML_TYPE_Q5_0 ||
                               op->type == GGML_TYPE_Q5_1 || op->type == GGML_TYPE_Q8_0 || op->type == GGML_TYPE_IQ4_NL ||
                               op->type == GGML_TYPE_TURBO3_0 || op->type == GGML_TYPE_TURBO2_0 ||
                               op->type == GGML_TYPE_TURBO4_0 || op->type == GGML_TYPE_TURBO4P_0 ||
                               op->type == GGML_TYPE_TURBO5P_0 || op->type == GGML_TYPE_TURBO5P512_0) &&
                               op->src[0]->type == GGML_TYPE_F32
                           ) || (
                               op->type == GGML_TYPE_F16 && op->src[0]->type == GGML_TYPE_F16
                           )
                       ) &&
                       (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
            } break;
        case GGML_OP_TURBOT_SET_ROWS:
            // [TAG_TURBOT] SPEC 5.4: the op checks, plus the base cache (src[2]) and the young pool (src[3]) written in
            // place in CUDA buffers. A tensor without a buffer yet is placed on this device by the scheduler.
            return ggml_cuda_turbot_set_rows_supported(op) &&
                   (op->src[2]->buffer == nullptr || ggml_backend_buft_is_cuda(op->src[2]->buffer->buft)) &&
                   (op->src[3]->buffer == nullptr || ggml_backend_buft_is_cuda(op->src[3]->buffer->buft));
        case GGML_OP_SET:
            {
                const ggml_type t = op->type;
                return (t == GGML_TYPE_F32 || t == GGML_TYPE_I32) &&
                    t == op->src[0]->type &&
                    t == op->src[1]->type;
            }
        case GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if ((src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_BF16 || src0_type == GGML_TYPE_F16) &&
                    (src1_type == GGML_TYPE_F32 || src1_type == GGML_TYPE_BF16 || src1_type == GGML_TYPE_F16)
                ) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q8_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q8_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_IQ4_NL) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == src1_type && ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                    return true;
                }
                return false;
            }
        case GGML_OP_DUP:
                return true;
        case GGML_OP_ARGMAX:
        case GGML_OP_COUNT_EQUAL:
            return true;
        case GGML_OP_REPEAT:
            {
                // the CUDA REPEAT path only implements F32/F16; other types assert at runtime
                ggml_type src0_type = op->src[0]->type;
                return src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16;
            }
        case GGML_OP_REPEAT_BACK:
                return op->type == GGML_TYPE_F32 && (op->src[0]->ne[2]*op->src[0]->ne[3]) <= (1 << 15);
        case GGML_OP_CONCAT:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                const int32_t dim = op->op_params[0];
                return src0_type == src1_type &&
                       src0_type == op->type &&
                       (
                           (
                               ggml_is_quantized(src0_type) &&
                               (
                                   (
                                       dim == 3 &&
                                       ggml_is_contiguous(op->src[0]) &&
                                       ggml_is_contiguous(op->src[1])
                                   ) || (
                                       dim != 3 &&
                                       ggml_is_contiguous_to_3(op->src[0]) &&
                                       ggml_is_contiguous_to_3(op->src[1])
                                   )
                               ) &&
                               op->src[0]->ne[0] % ggml_blck_size(src0_type) == 0 &&
                               op->src[1]->ne[0] % ggml_blck_size(src0_type) == 0
                           ) || (
                               !ggml_is_quantized(src0_type) &&
                               ggml_blck_size(src0_type) == 1 &&
                               (
                                   ggml_type_size(src0_type) == 1 ||
                                   ggml_type_size(src0_type) == 2 ||
                                   ggml_type_size(src0_type) == 4 ||
                                   ggml_type_size(src0_type) == 8
                               )
                           )
                       );
            }
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                return false;
            }
        case GGML_OP_COL2IM_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                return (src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16 || src0_type == GGML_TYPE_BF16) &&
                    op->type == src0_type &&
                    ggml_is_contiguous(op->src[0]) &&
                    ggml_is_contiguous(op);
            }
        case GGML_OP_SILU_BACK:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_RMS_NORM_BACK:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_ADD_ID:
        case GGML_OP_ADD1:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_LOG:
            return true;
        case GGML_OP_TURBO_WHT:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   op->src[0]->ne[0] % 64 == 0;  // head dim must be divisible by 64 (supports 64 and 128 WHT groups)
        case GGML_OP_SCALE:
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_BF16) && op->type == op->src[0]->type;
        case GGML_OP_CLAMP:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            if (op->src[0]->type == GGML_TYPE_BF16 || op->src[1]->type == GGML_TYPE_BF16 || op->type == GGML_TYPE_BF16) {
                return op->src[0]->type == GGML_TYPE_BF16 && op->type == GGML_TYPE_BF16 &&
                    (op->src[1]->type == GGML_TYPE_BF16 || op->src[1]->type == GGML_TYPE_F32);
            }
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) &&
                   (op->type         == GGML_TYPE_F32 || op->type         == GGML_TYPE_F16);
        case GGML_OP_SSM_SCAN: {
            const int32_t K = ggml_get_op_params_i32(op, 0);

            if (op->src[3]->ne[0] == 1) {
                // Mamba2
                // (kernel only supports (d_state == 96 || d_state == 128 || d_state == 256) && d_head % 16 == 0)
                const int64_t d_state = op->src[0]->ne[0];
                return (d_state == 96 || d_state == 128 || d_state == 256) && op->src[0]->ne[1] % 16 == 0;
            } else {
                if (K > 1) {
                    return false;
                }

                // Mamba
                // (kernel only supports d_state == 16, d_head == 1, n_head % 128 == 0, n_group == 1)
                return op->src[0]->ne[0] == 16 && op->src[0]->ne[1] == 1 && op->src[0]->ne[2] % 128 == 0 && op->src[4]->ne[1] == 1;
            }
        }
        case GGML_OP_SSM_CONV: {
            // assumes d_inner % threads == 0
            return op->src[0]->ne[1] % 128 == 0;
        }
        case GGML_OP_CONT:
            return true;
        case GGML_OP_DIAG_MASK_INF:
            return true;
        case GGML_OP_SOFT_MAX:
            return true;
        case GGML_OP_SOFT_MAX_BACK: {
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
            return max_bias == 0.0f;
        }
        case GGML_OP_ROLL:
            if(op->src[0]->type == GGML_TYPE_F32) {
                return true;
            }
            return false;
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK: {
            return op->src[0]->nb[0] == ggml_type_size(op->src[0]->type) && ggml_is_contiguous_2(op->src[0]);
        }
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_3D:
        case GGML_OP_CONV_2D:
            return (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]));
        case GGML_OP_CONV_3D:
            return (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                   op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && ggml_is_contiguous(op);
        case GGML_OP_CONV_2D_DW:
            return (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                   op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_CONV_TRANSPOSE_2D:
        case GGML_OP_POOL_1D:
        case GGML_OP_POOL_2D:
            return true;
        case GGML_OP_ACC:
            // TODO: extend support like so:
            //return ggml_is_contiguous_rows(op->src[0]) && ggml_is_contiguous_rows(op->src[1]);
            return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
        case GGML_OP_SUM:
            return ggml_is_contiguously_allocated(op->src[0]);
        case GGML_OP_TOP_K:
            return op->src[0]->ne[0] <= INT_MAX;
        case GGML_OP_ARGSORT:
#ifndef GGML_CUDA_USE_CUB
            {
                // bitonic path: the padded row must fit in shared memory
                int64_t ncols_pad = 1;
                while (ncols_pad < op->src[0]->ne[0]) {
                    ncols_pad *= 2;
                }
                return ncols_pad * sizeof(int) <= ggml_cuda_info().devices[dev_ctx->device].smpb;
            }
#else
            return op->src[0]->ne[0] <= INT_MAX;
#endif
        case GGML_OP_SUM_ROWS:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_MEAN:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_GROUP_NORM:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_PAD:
            return true;
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_RWKV_WKV7:
            return true;
        case GGML_OP_GATED_DELTA_NET:
            return true;
        case GGML_OP_GATED_DELTA_NET_REPLAY: // [TAG_4C_GDN_REPLAY] kept off MUSA (untested there; upstream f9af9be21 enabled only GATED_DELTA_NET)
#ifdef GGML_USE_MUSA
            return false;
#else
            return ggml_cuda_gdn_replay_supported(op);
#endif // GGML_USE_MUSA
        case GGML_OP_MOE_HOST_POST: // [TAG_MOE_BRIDGE] only for a bridge registered on this device
        case GGML_OP_MOE_HOST_WAIT:
        case GGML_OP_MOE_HOST_FETCH: // [TAG_FN_R4_BRIDGE_DMA]
            return ggml_cuda_moe_bridge_supports_op(dev_ctx->device, op);
        case GGML_OP_QSA_MASK: // [TAG_FN_R4_QSA_POS]
            return ggml_cuda_qsa_mask_supported(op);
        case GGML_OP_DSV4_HC_COMB:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_PRE:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_POST:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && (op->src[3] == nullptr || op->src[3]->type == GGML_TYPE_F32) &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_cuda_flash_attn_ext_supported(dev_ctx->device, op);
        case GGML_OP_FILL:
            // [TAG_FN_L21_RS_FILL] ggml_cuda_op_fill writes F32 and F16 only (it aborts on any other type), and a
            // contiguous tensor
            return (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && ggml_is_contiguous(op);
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
        case GGML_OP_CUMSUM:
        case GGML_OP_TRI:
        case GGML_OP_DIAG:
        case GGML_OP_SOLVE_TRI:
            return true;
        case GGML_OP_LIGHTNING_INDEXER:
            return ggml_cuda_lightning_indexer_supported(dev_ctx->device, op);
        case GGML_OP_TOP_K_MASK:
            return op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32;

        default:
            return false;
    }
}

static bool ggml_backend_cuda_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;
    const bool integrated = ggml_cuda_info().devices[dev_ctx->device].integrated;
    return (ggml_backend_buft_is_cuda(buft) && buft->device == dev) || (integrated && ggml_backend_buft_is_cuda_host(buft));
}

static int64_t get_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_GET_ROWS:
            return 0;
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}

// [TAG_FN_L22_XOVER] proc "ggml_backend_fn_set_offload_min_batch": an override of the op-offload minimum batch (P10's
// measured crossover); < 0 clears it
static std::atomic<int> ggml_cuda_fn_l22_offload_min{-1};

static void ggml_backend_cuda_fn_set_offload_min_batch(int min_batch) {
    ggml_cuda_fn_l22_offload_min.store(min_batch < 0 ? -1 : min_batch, std::memory_order_relaxed);
}

static bool ggml_backend_cuda_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    const int fn_min = ggml_cuda_fn_l22_offload_min.load(std::memory_order_relaxed); // [TAG_FN_L22_XOVER]
    if (fn_min >= 0) {
        return get_op_batch_size(op) >= fn_min;
    }

    // [TAG_FN_L12_PROMPTMIN] GGML_OP_OFFLOAD_MIN_BATCH as it is now: a model's automatic profile sets it at load, after the
    // backend registered (the value read at registration stays the default)
    const char * e = getenv("GGML_OP_OFFLOAD_MIN_BATCH");
    const int min_batch = e ? atoi(e) : dev_ctx->op_offload_min_batch_size;
    return get_op_batch_size(op) >= min_batch;
}

static ggml_backend_event_t ggml_backend_cuda_device_event_new(ggml_backend_dev_t dev) {
#ifdef GGML_CUDA_NO_PEER_COPY
    GGML_UNUSED(dev);
    return nullptr;
#else
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *)dev->context;

    ggml_cuda_set_device(dev_ctx->device);

    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

    return new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };
#endif
}

// [TAG_FN_L22_PIN] R-REV-3: true once the event has completed (non-blocking; false while pending or on an error), for the
// prefill stream's pin tickets (proc "ggml_backend_fn_event_done")
static bool ggml_backend_fn_event_done(ggml_backend_event_t event) {
#ifdef GGML_CUDA_NO_PEER_COPY
    GGML_UNUSED(event);
    return false;
#else
    if (event == nullptr || event->context == nullptr) {
        return false;
    }
    const cudaError_t err = cudaEventQuery((cudaEvent_t) event->context);
    if (err == cudaSuccess) {
        return true;
    }
    if (err != cudaErrorNotReady) {
        (void) cudaGetLastError(); // the release then waits on the event (the error shows there)
    }
    return false;
#endif
}

static void ggml_backend_cuda_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);

    CUDA_CHECK(cudaEventDestroy((cudaEvent_t)event->context));
    delete event;
}

static void ggml_backend_cuda_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    CUDA_CHECK(cudaEventSynchronize((cudaEvent_t)event->context));
}

static const ggml_backend_device_i ggml_backend_cuda_device_interface = {
    /* .get_name                = */ ggml_backend_cuda_device_get_name,
    /* .get_description         = */ ggml_backend_cuda_device_get_description,
    /* .get_memory              = */ ggml_backend_cuda_device_get_memory,
    /* .get_type                = */ ggml_backend_cuda_device_get_type,
    /* .get_props               = */ ggml_backend_cuda_device_get_props,
    /* .init_backend            = */ ggml_backend_cuda_device_init_backend,
    /* .get_buffer_type         = */ ggml_backend_cuda_device_get_buffer_type,
    /* .get_host_buffer_type    = */ ggml_backend_cuda_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ NULL,
    /* .supports_op             = */ ggml_backend_cuda_device_supports_op,
    /* .supports_buft           = */ ggml_backend_cuda_device_supports_buft,
    /* .offload_op              = */ ggml_backend_cuda_device_offload_op,
    /* .event_new               = */ ggml_backend_cuda_device_event_new,
    /* .event_free              = */ ggml_backend_cuda_device_event_free,
    /* .event_synchronize       = */ ggml_backend_cuda_device_event_synchronize,
};

// backend reg

struct ggml_backend_cuda_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_cuda_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_CUDA_NAME;
}

static size_t ggml_backend_cuda_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_cuda_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static ggml_backend_feature * ggml_backend_cuda_get_features(ggml_backend_reg_t reg) {
    static std::vector<ggml_backend_feature> features = []() {
        std::vector<ggml_backend_feature> features;
    #define _STRINGIFY(...) #__VA_ARGS__
    #define STRINGIFY(...) _STRINGIFY(__VA_ARGS__)

    #ifdef __CUDA_ARCH_LIST__
        features.push_back({ "ARCHS", STRINGIFY(__CUDA_ARCH_LIST__) });
    #endif

    #ifdef GGML_CUDA_FORCE_MMQ
        features.push_back({ "FORCE_MMQ", "1" });
    #endif

    #ifdef GGML_CUDA_FORCE_CUBLAS
        features.push_back({ "FORCE_CUBLAS", "1" });
    #endif

    #ifndef GGML_USE_VMM
        features.push_back({ "NO_VMM", "1" });
    #endif

    #ifdef GGML_CUDA_NO_PEER_COPY
        features.push_back({ "NO_PEER_COPY", "1" });
    #endif

    #ifdef GGML_CUDA_USE_GRAPHS
        features.push_back({ "USE_GRAPHS", "1" });
    #endif

    #ifdef GGML_CUDA_FA_QUANTS
        features.push_back({ "FA_QUANTS", GGML_CUDA_FA_QUANTS });
    #endif

    {
        const auto & info = ggml_cuda_info();
        for (int id = 0; id < info.device_count; ++id) {
            if (blackwell_mma_available(info.devices[id].cc)) {
                features.push_back({ "BLACKWELL_NATIVE_FP4", "1"});
                break;
            }
        }
    }

    #undef _STRINGIFY
    #undef STRINGIFY

        features.push_back({ nullptr, nullptr });

        return features;
    }();

    return features.data();

    GGML_UNUSED(reg);
}

// [TAG_TURBOT_ANY_RESOLVE] "ggml_backend_turbot_supports_geometry", bool (ggml_backend_dev_t, int head_dim, int n_head_kv):
// the KV resolver (LLAMA_TURBOT_DEV_GEOM_PROC, src/llama-context.cpp) asks it per attention layer, so a layer this build
// or this device cannot run as turbot steps down instead of reaching a refused flash attention. See fattn.cu.
static bool ggml_backend_cuda_turbot_supports_geometry(ggml_backend_dev_t dev, int head_dim, int n_head_kv) {
    const ggml_backend_cuda_device_context * dev_ctx = (const ggml_backend_cuda_device_context *) dev->context;
    return ggml_cuda_turbot_geometry_supported(dev_ctx->device, head_dim, n_head_kv);
}

// [TAG_FN_L23_PINFIX] R-L23-8 backstop for our registrations (upstream #31122: the WDDM page-lock budget is about half the RAM
// and unstable past it): the total stays under the cap libllama sets (0.4 x RAM), and the first refusal by the driver stops
// every later registration of the process (no retry until a restart)
static std::mutex                         g_fn_reg_mtx;
static std::unordered_map<void *, size_t> g_fn_reg_map;
static size_t                             g_fn_reg_total   = 0;
static size_t                             g_fn_reg_cap     = 0;
static bool                               g_fn_reg_refused = false;

static void ggml_backend_cuda_fn_set_reg_cap(size_t cap) {
    std::lock_guard<std::mutex> lk(g_fn_reg_mtx);
    g_fn_reg_cap = cap;
}

static bool ggml_cuda_fn_reg_admit(size_t size) {
    {
        std::lock_guard<std::mutex> lk(g_fn_bounce_mtx);
        (void) ggml_cuda_fn_bounce_locked(); // the bounce buffer exists before the first pin
    }
    std::lock_guard<std::mutex> lk(g_fn_reg_mtx);
    if (g_fn_reg_refused) {
        return false;
    }
    if (g_fn_reg_cap > 0 && g_fn_reg_total + size > g_fn_reg_cap) {
        GGML_LOG_WARN("ggml_cuda: [TAG_FN_L23_PINFIX] %.0f MiB not registered: %.2f GiB registered, cap %.2f GiB\n", size/1048576.0,
                g_fn_reg_total/1073741824.0, g_fn_reg_cap/1073741824.0);
        return false;
    }
    return true;
}

static void ggml_cuda_fn_reg_done(void * buffer, size_t size, cudaError_t err) {
    std::lock_guard<std::mutex> lk(g_fn_reg_mtx);
    if (err != cudaSuccess) {
        if (!g_fn_reg_refused) {
            GGML_LOG_WARN("ggml_cuda: [TAG_FN_L23_PINFIX] the driver refused a registration (%.0f MiB, %.2f GiB registered: %s): no more "
                    "registrations in this process\n", size/1048576.0, g_fn_reg_total/1073741824.0, cudaGetErrorString(err));
        }
        g_fn_reg_refused = true;
        return;
    }
    g_fn_reg_map[buffer] = size;
    g_fn_reg_total += size;
}

// [TAG_FN_L14_ARENA] the expert arena (locked host memory) registered for direct DMA, without the GGML_CUDA_REGISTER_HOST
// switch the generic pair needs; false when the driver refuses (the caller keeps its staged copies then)
static bool ggml_backend_cuda_fn_register_host(void * buffer, size_t size) {
    if (!ggml_cuda_fn_reg_admit(size)) {
        return false;
    }
    cudaError_t err = cudaHostRegister(buffer, size, cudaHostRegisterPortable | cudaHostRegisterReadOnly);
    ggml_cuda_fn_reg_done(buffer, size, err);
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        GGML_LOG_WARN("%s: %.0f MiB not registered: %s\n", __func__, size/1048576.0, cudaGetErrorString(err));
        return false;
    }
    return true;
}

static void ggml_backend_cuda_fn_unregister_host(void * buffer) {
    if (cudaHostUnregister(buffer) != cudaSuccess) {
        (void) cudaGetLastError();
    }
    std::lock_guard<std::mutex> lk(g_fn_reg_mtx);
    auto it = g_fn_reg_map.find(buffer);
    if (it != g_fn_reg_map.end()) {
        g_fn_reg_total -= std::min(g_fn_reg_total, it->second);
        g_fn_reg_map.erase(it);
    }
}

// [TAG_FN_L22_PIN] R-P14-5: a host range registered with a device mapping (pinned DMA share reads it from kernels), and
// the device address of a registered host pointer (null if the driver has none)
static bool ggml_backend_cuda_fn_register_host_mapped(void * buffer, size_t size, void ** dev) {
    if (dev) {
        *dev = nullptr;
    }
    if (!ggml_cuda_fn_reg_admit(size)) {
        return false;
    }
    cudaError_t err = cudaHostRegister(buffer, size, cudaHostRegisterPortable | cudaHostRegisterReadOnly | cudaHostRegisterMapped);
    ggml_cuda_fn_reg_done(buffer, size, err);
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        GGML_LOG_WARN("%s: %.0f MiB not registered: %s\n", __func__, size/1048576.0, cudaGetErrorString(err));
        return false;
    }
    if (dev && cudaHostGetDevicePointer(dev, buffer, 0) != cudaSuccess) {
        (void) cudaGetLastError();
        *dev = nullptr;
    }
    return true;
}

static void * ggml_backend_cuda_fn_host_dev_ptr(const void * host) {
    void * dev = nullptr;
    if (host == nullptr || cudaHostGetDevicePointer(&dev, const_cast<void *>(host), 0) != cudaSuccess) {
        (void) cudaGetLastError();
        return nullptr;
    }
    return dev;
}

// [TAG_FN_L22_PIN] proc "ggml_backend_fn_device_luid" (P6 DXGI budget): the adapter LUID of a CUDA device, as the 8 bytes
// of cudaDeviceProp::luid (LowPart, HighPart of the Windows LUID); false off Windows or when the driver gives none
static bool ggml_backend_cuda_fn_device_luid(ggml_backend_dev_t dev, uint64_t * luid) {
    if (dev == nullptr || luid == nullptr || ggml_backend_dev_backend_reg(dev) != ggml_backend_cuda_reg()) {
        return false;
    }
#if defined(_WIN32) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    const ggml_backend_cuda_device_context * dev_ctx = (const ggml_backend_cuda_device_context *) dev->context;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(dev_ctx->device)) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    if (prop.luidDeviceNodeMask == 0) {
        return false;
    }
    uint64_t v = 0;
    memcpy(&v, prop.luid, sizeof(v));
    *luid = v;
    return v != 0;
#else
    return false;
#endif
}

static void * ggml_backend_cuda_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (void * p = ggml_cuda_fn_l22_dmafetch_proc(name)) { // [TAG_FN_L22_DMASHARE] R-P14-4 (stub: nullptr)
        return p;
    }
    if (strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *)ggml_backend_cuda_comm_init;
    }
    if (strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *)ggml_backend_cuda_comm_free;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *)ggml_backend_cuda_comm_allreduce_tensor;
    }
    if (strcmp(name, "ggml_backend_register_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_register_host_buffer;
    }
    if (strcmp(name, "ggml_backend_unregister_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_unregister_host_buffer;
    }
    if (strcmp(name, "ggml_backend_fn_register_host") == 0) { // [TAG_FN_L14_ARENA]
        return (void *)ggml_backend_cuda_fn_register_host;
    }
    if (strcmp(name, "ggml_backend_fn_unregister_host") == 0) {
        return (void *)ggml_backend_cuda_fn_unregister_host;
    }
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_cuda_get_features;
    }
    if (strcmp(name, "ggml_backend_set_tensors_batch_async") == 0) { // [TAG_FN_L4_HOST_BATCHCPY]
        return (void *)ggml_backend_cuda_set_tensors_batch_async;
    }
    if (strcmp(name, "ggml_backend_set_tensors_pack_async") == 0) { // [TAG_FN_L16_UPLOAD]
        return (void *)ggml_backend_cuda_set_tensors_pack_async;
    }
    if (strcmp(name, "ggml_backend_copy_d2d_batch_async") == 0) { // [TAG_FN_L14_PFSD2D]
        return (void *)ggml_backend_cuda_copy_d2d_batch_async;
    }
    if (strcmp(name, "ggml_backend_turbot_supports_geometry") == 0) {   // [TAG_TURBOT_ANY_RESOLVE]
        return (void *)ggml_backend_cuda_turbot_supports_geometry;
    }
    // [TAG_MOE_BRIDGE] ggml-moe-bridge.h
    if (strcmp(name, "ggml_backend_moe_bridge_new") == 0) {
        return (void *)ggml_backend_cuda_moe_bridge_new;
    }
    if (strcmp(name, "ggml_backend_moe_bridge_free") == 0) {
        return (void *)ggml_backend_cuda_moe_bridge_free;
    }
    if (strcmp(name, "ggml_backend_moe_bridge_id") == 0) {
        return (void *)ggml_backend_cuda_moe_bridge_id;
    }
    if (strcmp(name, "ggml_backend_moe_bridge_set_runner") == 0) {
        return (void *)ggml_backend_cuda_moe_bridge_set_runner;
    }
    if (strcmp(name, "ggml_backend_moe_bridge_poll") == 0) {
        return (void *)ggml_backend_cuda_moe_bridge_poll;
    }
    if (strcmp(name, "ggml_backend_moe_bridge_complete") == 0) {
        return (void *)ggml_backend_cuda_moe_bridge_complete;
    }
    if (strcmp(name, "ggml_backend_moe_bridge_error") == 0) {
        return (void *)ggml_backend_cuda_moe_bridge_error;
    }
    if (strcmp(name, "ggml_backend_moe_bridge_reset") == 0) {
        return (void *)ggml_backend_cuda_moe_bridge_reset;
    }
    if (strcmp(name, "ggml_backend_moe_bridge_get_stats") == 0) {
        return (void *)ggml_backend_cuda_moe_bridge_get_stats;
    }
    // [TAG_FN_R4_BRIDGE_DMA]
    if (strcmp(name, "ggml_backend_moe_bridge_set_ring") == 0) {
        return (void *)ggml_backend_cuda_moe_bridge_set_ring;
    }
    if (strcmp(name, "ggml_backend_moe_bridge_publish_plan") == 0) {
        return (void *)ggml_backend_cuda_moe_bridge_publish_plan;
    }
    if (strcmp(name, "ggml_backend_moe_bridge_chan_times") == 0) {
        return (void *)ggml_backend_cuda_moe_bridge_chan_times;
    }
    if (strcmp(name, "ggml_backend_moe_bridge_clock_offset") == 0) { // [TAG_FN_L14_PROBE]
        return (void *)ggml_backend_cuda_moe_bridge_clock_offset;
    }
    if (strcmp(name, "ggml_backend_moe_bridge_release") == 0) {
        return (void *)ggml_backend_cuda_moe_bridge_release;
    }
    // [TAG_FN_L3_HOST_EXEC]
    if (strcmp(name, "ggml_backend_moe_bridge_set_watch") == 0) {
        return (void *)ggml_backend_cuda_moe_bridge_set_watch;
    }
    if (strcmp(name, "ggml_backend_moe_bridge_next_post_word") == 0) {
        return (void *)ggml_backend_cuda_moe_bridge_next_post_word;
    }
    if (strcmp(name, "ggml_backend_moe_bridge_test_seed") == 0) {
        return (void *)ggml_backend_cuda_moe_bridge_test_seed;
    }
    if (strcmp(name, "ggml_backend_moe_bridge_host_take") == 0) { // [TAG_FN_L16_PLEGATE]
        return (void *)ggml_backend_cuda_moe_bridge_host_take;
    }
    // [TAG_FN_L3_HOST_POKE]
    if (strcmp(name, "ggml_backend_cuda_fn_set_poke") == 0) {
        return (void *)ggml_backend_cuda_fn_set_poke;
    }
    if (strcmp(name, "ggml_backend_cuda_stream_poke") == 0) {
        return (void *)ggml_backend_cuda_stream_poke;
    }
    // [TAG_FN_L3_VRAM_CBUF] [TAG_FN_L3_VRAM_TRIM]
    if (strcmp(name, "ggml_backend_vmm_granularity") == 0) {
        return (void *)ggml_backend_cuda_vmm_granularity;
    }
    if (strcmp(name, "ggml_backend_vmm_buffer_alloc") == 0) {
        return (void *)ggml_backend_cuda_vmm_buffer_alloc;
    }
    if (strcmp(name, "ggml_backend_vmm_buffer_map") == 0) {
        return (void *)ggml_backend_cuda_vmm_buffer_map;
    }
    if (strcmp(name, "ggml_backend_vmm_buffer_unmap") == 0) {
        return (void *)ggml_backend_cuda_vmm_buffer_unmap;
    }
    if (strcmp(name, "ggml_backend_vmm_buffer_mapped") == 0) {
        return (void *)ggml_backend_cuda_vmm_buffer_mapped;
    }
    if (strcmp(name, "ggml_backend_pool_stats") == 0) {
        return (void *)ggml_backend_cuda_pool_stats;
    }
    if (strcmp(name, "ggml_backend_pool_trim") == 0) {
        return (void *)ggml_backend_cuda_pool_trim;
    }
    // [TAG_FN_L22_GRAPHKEEP] [TAG_FN_L22_GRAPHKEY] [TAG_FN_L22_PIN] [TAG_FN_L22_XOVER] [TAG_FN_L22_OOMHOOK] round 22 P7 procs
    if (strcmp(name, "ggml_backend_pool_mark") == 0) {
        return (void *)ggml_backend_cuda_pool_mark;
    }
    if (strcmp(name, "ggml_backend_cuda_set_graph_key_mode") == 0) {
        return (void *)ggml_backend_cuda_set_graph_key_mode;
    }
    if (strcmp(name, "ggml_backend_cuda_graph_counters") == 0) {
        return (void *)ggml_backend_cuda_graph_counters;
    }
    if (strcmp(name, "ggml_backend_cuda_graph_window") == 0) {
        return (void *)ggml_backend_cuda_graph_window;
    }
    if (strcmp(name, "ggml_backend_fn_device_luid") == 0) {
        return (void *)ggml_backend_cuda_fn_device_luid;
    }
    if (strcmp(name, "ggml_backend_fn_set_offload_min_batch") == 0) {
        return (void *)ggml_backend_cuda_fn_set_offload_min_batch;
    }
    if (strcmp(name, "ggml_backend_fn_host_dev_ptr") == 0) {
        return (void *)ggml_backend_cuda_fn_host_dev_ptr;
    }
    if (strcmp(name, "ggml_backend_fn_register_host_mapped") == 0) {
        return (void *)ggml_backend_cuda_fn_register_host_mapped;
    }
    if (strcmp(name, "ggml_backend_fn_event_done") == 0) { // [TAG_FN_L22_PIN] R-REV-3
        return (void *)ggml_backend_fn_event_done;
    }
    if (strcmp(name, "ggml_backend_cuda_set_passlog_level") == 0) { // [TAG_FN_L22_PASSLOG] R-REV-11
        return (void *)ggml_backend_cuda_set_passlog_level;
    }
    if (strcmp(name, "ggml_backend_cuda_set_oom_hook") == 0) {
        return (void *)ggml_backend_cuda_set_oom_hook;
    }
    if (strcmp(name, "ggml_backend_cuda_fn_set_reg_cap") == 0) { // [TAG_FN_L23_PINFIX] R-L23-8
        return (void *)ggml_backend_cuda_fn_set_reg_cap;
    }
    if (strcmp(name, "ggml_backend_cuda_limits") == 0) { // [TAG_FN_L3_VRAM_ACCOUNT]
        return (void *)ggml_backend_cuda_limits;
    }
    if (strcmp(name, "ggml_backend_moe_bridge_read_hint") == 0) { // [TAG_FN_L3_CPU_DEVPRED]
        return (void *)ggml_backend_cuda_moe_bridge_read_hint;
    }
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_cuda_reg_interface = {
    /* .get_name          = */ ggml_backend_cuda_reg_get_name,
    /* .get_device_count  = */ ggml_backend_cuda_reg_get_device_count,
    /* .get_device        = */ ggml_backend_cuda_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_cuda_reg_get_proc_address,
};

// backend registry
ggml_backend_reg_t ggml_backend_cuda_reg() {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_backend_cuda_reg_context * ctx = new ggml_backend_cuda_reg_context;
            const int min_batch_size = ggml_cuda_op_offload_min_batch();

            const ggml_cuda_device_info & info = ggml_cuda_info();
            const bool virtual_devices = info.device_count > info.physical_device_count;

            for (int i = 0; i < info.device_count; i++) {
                const int physical_id = info.devices[i].physical_device;

                ggml_backend_cuda_device_context * dev_ctx = new ggml_backend_cuda_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_CUDA_NAME + std::to_string(i);
                dev_ctx->description = ggml_cuda_device_description(i);

                char pci_bus_id[32] = {};
                CUDA_CHECK(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), physical_id));
                dev_ctx->pci_bus_id = pci_bus_id;
                if (virtual_devices) {
                    // make the pci bus id unique for virtual devices
                    dev_ctx->pci_bus_id += "-v" + std::to_string(i);
                }
                for (char & c : dev_ctx->pci_bus_id) {
                    c = std::tolower(c);
                }
                dev_ctx->op_offload_min_batch_size = min_batch_size;

                ggml_backend_dev_t dev = new ggml_backend_device {
                    /* .iface   = */ ggml_backend_cuda_device_interface,
                    /* .reg     = */ &reg,
                    /* .context = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_cuda_reg_interface,
                /* .context     = */ ctx
            };
        }

        initialized = true;
    }

    return &reg;
}

ggml_backend_t ggml_backend_cuda_init(int device) {
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) {
        GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_cuda_context * ctx = new ggml_backend_cuda_context(device);
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: failed to allocate context\n", __func__);
        return nullptr;
    }

    ggml_backend_t cuda_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_cuda_guid(),
        /* .iface   = */ ggml_backend_cuda_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), device),
        /* .context = */ ctx,
    };

    return cuda_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cuda_reg)
