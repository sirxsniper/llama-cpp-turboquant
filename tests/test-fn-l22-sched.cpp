// [TAG_FN_L22_SCHEDBUF] GGML_SCHED_FN_L22_SPLITS (round 22, P7 M1): the scheduler's backend id arrays and context buffer
// start at that many splits and grow when a graph needs more (grow, never clamp). A chain of N nodes pinned to two
// backends in turn (a fake device backend whose host memory the CPU backend does not take, and the CPU backend) makes N
// splits with one input copy each, far more than SPLITS = 8 covers. The result must equal the host reference and the run
// without the switch, bit for bit. CPU only, no model.

#include "../ggml/src/ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// [TAG_FN_L22_SCHEDBUF] debug lines of the scheduler's growth paths, counted per run
static int g_grow_ctx = 0;
static int g_grow_ids = 0;

static void log_cb(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    if (strstr(text, "[TAG_FN_L22_SCHEDBUF] context buffer") != nullptr) {
        g_grow_ctx++;
    }
    if (strstr(text, "[TAG_FN_L22_SCHEDBUF] backend id arrays") != nullptr) {
        g_grow_ids++;
    }
}

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

// ---------------------------------------------------------------------------------------------------------------------
// fake device: host memory marked as not host, compute through an inner CPU backend

struct fake_buf {
    void * data;
    size_t size;
};

static void fake_buf_free(ggml_backend_buffer_t buffer) {
    fake_buf * b = (fake_buf *) buffer->context;
    free(b->data);
    delete b;
}

static void * fake_buf_base(ggml_backend_buffer_t buffer) {
    return ((fake_buf *) buffer->context)->data;
}

static void fake_buf_memset(ggml_backend_buffer_t, ggml_tensor * t, uint8_t value, size_t offset, size_t size) {
    memset((char *) t->data + offset, value, size);
}

static void fake_buf_set(ggml_backend_buffer_t, ggml_tensor * t, const void * data, size_t offset, size_t size) {
    memcpy((char *) t->data + offset, data, size);
}

static void fake_buf_get(ggml_backend_buffer_t, const ggml_tensor * t, void * data, size_t offset, size_t size) {
    memcpy(data, (const char *) t->data + offset, size);
}

static bool fake_buf_cpy(ggml_backend_buffer_t, const ggml_tensor * src, ggml_tensor * dst) {
    if (src->buffer && ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;
}

static void fake_buf_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    fake_buf * b = (fake_buf *) buffer->context;
    memset(b->data, value, b->size);
}

static const ggml_backend_buffer_i fake_buf_iface = {
    /* .free_buffer   = */ fake_buf_free,
    /* .get_base      = */ fake_buf_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ fake_buf_memset,
    /* .set_tensor    = */ fake_buf_set,
    /* .get_tensor    = */ fake_buf_get,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ fake_buf_cpy,
    /* .clear         = */ fake_buf_clear,
    /* .reset         = */ nullptr,
    /* .set_tensor_n  = */ nullptr,
    /* .get_tensor_n  = */ nullptr,
    /* .cpy_tensor_n  = */ nullptr,
};

static const char * fake_buft_name(ggml_backend_buffer_type_t) {
    return "FAKE";
}

static ggml_backend_buffer_t fake_buft_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    fake_buf * b = new fake_buf;
    b->size = size;
    b->data = malloc(size + 64);
    GGML_ASSERT(b->data != nullptr);
    return ggml_backend_buffer_init(buft, fake_buf_iface, b, size);
}

static size_t fake_buft_alignment(ggml_backend_buffer_type_t) {
    return 32;
}

static bool fake_buft_is_host(ggml_backend_buffer_type_t) {
    return false; // the CPU backend must copy what it reads from here
}

static ggml_backend_buffer_type fake_buft = {
    /* .iface   = */ {
        /* .get_name         = */ fake_buft_name,
        /* .alloc_buffer     = */ fake_buft_alloc,
        /* .alloc_buffer_n   = */ nullptr,
        /* .get_alignment    = */ fake_buft_alignment,
        /* .get_max_size     = */ nullptr,
        /* .get_alloc_size   = */ nullptr,
        /* .get_alloc_size_n = */ nullptr,
        /* .is_host          = */ fake_buft_is_host,
    },
    /* .device  = */ nullptr,
    /* .context = */ nullptr,
};

static const char * fake_dev_name(ggml_backend_dev_t) {
    return "FAKE";
}

static void fake_dev_memory(ggml_backend_dev_t, size_t * free, size_t * total) {
    *free  = (size_t) 1 << 30;
    *total = (size_t) 1 << 30;
}

static enum ggml_backend_dev_type fake_dev_type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void fake_dev_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    memset(props, 0, sizeof(*props));
    props->name        = fake_dev_name(dev);
    props->description = fake_dev_name(dev);
    props->type        = GGML_BACKEND_DEVICE_TYPE_GPU;
    fake_dev_memory(dev, &props->memory_free, &props->memory_total);
}

static ggml_backend_buffer_type_t fake_dev_buft(ggml_backend_dev_t) {
    return &fake_buft;
}

static bool fake_dev_supports_op(ggml_backend_dev_t, const ggml_tensor *) {
    return true;
}

static bool fake_dev_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) {
    return buft == &fake_buft;
}

static bool fake_dev_offload_op(ggml_backend_dev_t, const ggml_tensor *) {
    return false;
}

static ggml_backend_device fake_dev = {
    /* .iface   = */ {
        /* .get_name             = */ fake_dev_name,
        /* .get_description      = */ fake_dev_name,
        /* .get_memory           = */ fake_dev_memory,
        /* .get_type             = */ fake_dev_type,
        /* .get_props            = */ fake_dev_props,
        /* .init_backend         = */ nullptr,
        /* .get_buffer_type      = */ fake_dev_buft,
        /* .get_host_buffer_type = */ nullptr,
        /* .buffer_from_host_ptr = */ nullptr,
        /* .supports_op          = */ fake_dev_supports_op,
        /* .supports_buft        = */ fake_dev_supports_buft,
        /* .offload_op           = */ fake_dev_offload_op,
        /* .event_new            = */ nullptr,
        /* .event_free           = */ nullptr,
        /* .event_synchronize    = */ nullptr,
    },
    /* .reg     = */ nullptr,
    /* .context = */ nullptr,
};

static const char * fake_backend_name(ggml_backend_t) {
    return "FAKE";
}

static void fake_backend_free(ggml_backend_t backend) {
    delete backend;
}

static void fake_backend_sync(ggml_backend_t) {
}

static enum ggml_status fake_backend_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    // the tensors are plain host memory: the CPU backend computes them in place
    return ggml_backend_graph_compute((ggml_backend_t) backend->context, cgraph);
}

static ggml_guid fake_guid = { 0x5a, 0x3e, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0x01 };

static ggml_backend_t fake_backend_new(ggml_backend_t cpu) {
    fake_buft.device = &fake_dev;
    ggml_backend_i iface = {};
    iface.get_name      = fake_backend_name;
    iface.free          = fake_backend_free;
    iface.synchronize   = fake_backend_sync;
    iface.graph_compute = fake_backend_compute;
    return new ggml_backend { &fake_guid, iface, &fake_dev, cpu };
}

// ---------------------------------------------------------------------------------------------------------------------

static const int     N_NODES = 3000;
static const int64_t N_ELEM  = 64;

// x0 -> (scale, add c) alternately; node i on the fake backend for even i, on the CPU for odd i
static bool run_chain(const char * splits_env, std::vector<float> & out, int * n_splits) {
    set_env("GGML_SCHED_FN_L22_SPLITS", splits_env);

    ggml_backend_t cpu   = ggml_backend_cpu_init();
    ggml_backend_t inner = ggml_backend_cpu_init();
    ggml_backend_t fake  = fake_backend_new(inner);

    const size_t graph_size = 8192;
    ggml_init_params ip = { ggml_tensor_overhead()*(N_NODES + 16) + ggml_graph_overhead_custom(graph_size, false), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N_ELEM);
    ggml_tensor * c = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N_ELEM);
    ggml_set_input(x);
    ggml_set_input(c);
    std::vector<ggml_tensor *> nodes;
    ggml_tensor * cur = x;
    for (int i = 0; i < N_NODES; ++i) {
        cur = (i % 2) ? ggml_add(ctx, cur, c) : ggml_scale(ctx, cur, 0.999f);
        nodes.push_back(cur);
    }
    ggml_set_output(cur);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, graph_size, false);
    ggml_build_forward_expand(gf, cur);

    ggml_backend_t backends[2] = { fake, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, graph_size, false, false);
    for (int i = 0; i < N_NODES; ++i) {
        ggml_backend_sched_set_tensor_backend(sched, nodes[i], (i % 2) ? cpu : fake);
    }
    ggml_backend_sched_set_tensor_backend(sched, x, fake);
    ggml_backend_sched_set_tensor_backend(sched, c, cpu);

    bool ok = ggml_backend_sched_alloc_graph(sched, gf);
    if (ok) {
        std::vector<float> xv(N_ELEM), cv(N_ELEM);
        for (int64_t k = 0; k < N_ELEM; ++k) {
            xv[k] = 1.0f + 0.01f*(float) k;
            cv[k] = 0.001f*(float) (k % 7);
        }
        ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
        ggml_backend_tensor_set(c, cv.data(), 0, ggml_nbytes(c));
        ok = ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
        ggml_backend_sched_synchronize(sched);
        out.resize(N_ELEM);
        ggml_backend_tensor_get(cur, out.data(), 0, ggml_nbytes(cur));
        *n_splits = ggml_backend_sched_get_n_splits(sched);
    }

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_free(fake);
    ggml_backend_free(inner);
    ggml_backend_free(cpu);
    return ok;
}

int main() {
    int n_fail = 0;
    ggml_log_set(log_cb, nullptr);

    std::vector<float> ref(N_ELEM);
    for (int64_t k = 0; k < N_ELEM; ++k) {
        float v = 1.0f + 0.01f*(float) k;
        const float cv = 0.001f*(float) (k % 7);
        for (int i = 0; i < N_NODES; ++i) {
            v = (i % 2) ? v + cv : v*0.999f;
        }
        ref[k] = v;
    }

    std::vector<float> out_off, out_on, out_big;
    int splits_off = 0, splits_on = 0, splits_big = 0;
    const bool ok_off = run_chain(nullptr, out_off, &splits_off);
    const int  grow_off = g_grow_ctx + g_grow_ids;
    const bool ok_on  = run_chain("8", out_on, &splits_on);
    const int  grow_ctx_on = g_grow_ctx, grow_ids_on = g_grow_ids;
    const bool ok_big = run_chain("4096", out_big, &splits_big);

    const auto check = [&](const char * what, bool cond) {
        printf("%-60s %s\n", what, cond ? "ok" : "FAIL");
        n_fail += cond ? 0 : 1;
    };
    check("chain runs without the switch", ok_off);
    check("chain runs with GGML_SCHED_FN_L22_SPLITS=8 (grows)", ok_on);
    check("chain runs with GGML_SCHED_FN_L22_SPLITS=4096", ok_big);
    check("thousands of splits", splits_off >= N_NODES/2 && splits_on == splits_off && splits_big == splits_off);
    check("no growth without the switch", grow_off == 0);
    check("SPLITS=8 grew the context buffer and the id arrays", grow_ctx_on >= 1 && grow_ids_on >= 1);
    if (ok_off && ok_on && ok_big) {
        check("SPLITS=8 equals the run without the switch (bits)", memcmp(out_on.data(), out_off.data(), N_ELEM*sizeof(float)) == 0);
        check("SPLITS=4096 equals the run without the switch (bits)", memcmp(out_big.data(), out_off.data(), N_ELEM*sizeof(float)) == 0);
        float max_rel = 0.0f;
        for (int64_t k = 0; k < N_ELEM; ++k) {
            max_rel = std::fmax(max_rel, std::fabs(out_off[k] - ref[k]) / std::fmax(1e-6f, std::fabs(ref[k])));
        }
        check("result equals the host reference (rel 1e-5)", max_rel < 1e-5f);
    }
    printf("splits %d / %d / %d\n", splits_off, splits_on, splits_big);
    set_env("GGML_SCHED_FN_L22_SPLITS", nullptr);

    printf("%s: %d failed\n", n_fail ? "FAIL" : "OK", n_fail);
    return n_fail ? 1 : 0;
}
