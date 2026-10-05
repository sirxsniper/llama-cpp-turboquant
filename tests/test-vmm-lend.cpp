// [TAG_FN_L3_VRAM_CBUF] [TAG_FN_L3_VRAM_TRIM] GPU test of the CUDA virtual-memory buffers and the pool trim, the paths
// under the compute-buffer lend (src/llama-moecache.cpp, src/llama-context.cpp cbuf_set):
//   1. hot-set-like slot tensors (q4_K up / gate, q5_1 down, the zero slot last) in a VMM buffer whose top layers form a
//      separately mapped tail: the MUL_MAT_ID chain gives the same bytes as from a cudaMalloc buffer;
//   2. the tail unmapped: a graph over the base layers runs and gives the same bytes;
//   3. the tail mapped again, cleared, refilled: the whole chain gives the same bytes again; 8 cycles, and the device's
//      free memory moves by the tail each time (the VRAM really goes back);
//   4. the pool trim between CUDA-graph runs of one graph: the pool gives its memory back, the captured graphs are dropped,
//      the next runs give the same bytes.
// Run it under compute-sanitizer memcheck first (a read of an unmapped range is reported there). Built, not registered with
// ctest: it needs the GPU. Exit code 0 iff every check passes; prints SKIPPED without a CUDA device with these functions.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <random>
#include <string>
#include <vector>

static int g_fail   = 0;
static int g_checks = 0;

#define TCHECK(cond, ...)                                        \
    do {                                                         \
        ++g_checks;                                              \
        if (!(cond)) {                                           \
            ++g_fail;                                            \
            fprintf(stderr, "FAIL %s:%d: ", __func__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                        \
            fprintf(stderr, "\n");                               \
        }                                                        \
    } while (0)

typedef size_t                (*vmm_gran_t)(ggml_backend_buffer_type_t);
typedef ggml_backend_buffer_t (*vmm_alloc_t)(ggml_backend_buffer_type_t, size_t);
typedef bool                  (*vmm_map_t)(ggml_backend_buffer_t, size_t, size_t);
typedef size_t                (*vmm_mapped_t)(ggml_backend_buffer_t);
typedef void                  (*pool_stats_t)(ggml_backend_t, size_t *, size_t *);
typedef size_t                (*pool_trim_t)(ggml_backend_t, size_t);

static constexpr int64_t N_EMBD  = 512;
static constexpr int64_t N_FF    = 256;
static constexpr int     N_SLOTS = 16;  // + the zero slot
static constexpr int     N_LAYER = 8;
static constexpr int     N_USED  = 4;
static constexpr int     N_TOK   = 3;   // an MTP verify graph

struct layer_w {
    ggml_tensor * up   = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * down = nullptr;
};

// slot weights (host bytes per tensor kind and layer), the zero slot all zeros
struct weights {
    std::vector<std::vector<uint8_t>> up, gate, down;
};

static std::vector<uint8_t> quantize_slots(ggml_type type, int64_t ne0, int64_t ne1, std::mt19937 & rng) {
    std::normal_distribution<float> nd(0.0f, 0.5f);
    const size_t row  = ggml_row_size(type, ne0);
    const size_t slot = row * ne1;
    std::vector<uint8_t> out(slot * (N_SLOTS + 1), 0);
    std::vector<float> f((size_t) (ne0 * ne1));
    for (int s = 0; s < N_SLOTS; ++s) {
        for (float & x : f) {
            x = nd(rng);
        }
        ggml_quantize_chunk(type, f.data(), out.data() + (size_t) s * slot, 0, ne1, ne0, nullptr);
    }
    return out;
}

static void make_layers(ggml_context * ctx, std::vector<layer_w> & L) {
    L.resize(N_LAYER);
    for (int il = 0; il < N_LAYER; ++il) {
        L[il].up   = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, N_EMBD, N_FF,   N_SLOTS + 1);
        L[il].gate = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, N_EMBD, N_FF,   N_SLOTS + 1);
        L[il].down = ggml_new_tensor_3d(ctx, GGML_TYPE_Q5_1, N_FF,   N_EMBD, N_SLOTS + 1);
        ggml_format_name(L[il].up,   "up.%d",   il);
        ggml_format_name(L[il].gate, "gate.%d", il);
        ggml_format_name(L[il].down, "down.%d", il);
    }
}

static void upload_layer(const std::vector<layer_w> & L, const weights & W, int il) {
    ggml_backend_tensor_set(L[il].up,   W.up[il].data(),   0, W.up[il].size());
    ggml_backend_tensor_set(L[il].gate, W.gate[il].data(), 0, W.gate[il].size());
    ggml_backend_tensor_set(L[il].down, W.down[il].data(), 0, W.down[il].size());
}

// the hot chain of llama-graph.cpp over layers [0, n): out_l = down(swiglu(gate(x), up(x))) with slot ids
struct chain_graph {
    ggml_context *       ctx  = nullptr;
    ggml_cgraph *        gf   = nullptr;
    ggml_gallocr_t       ga   = nullptr;
    ggml_tensor *        x    = nullptr;
    ggml_tensor *        ids  = nullptr;
    std::vector<ggml_tensor *> outs;

    void build(const std::vector<layer_w> & L, int n, ggml_backend_buffer_type_t buft) {
        ggml_init_params ip = { ggml_tensor_overhead() * 256 + ggml_graph_overhead(), nullptr, true };
        ctx = ggml_init(ip);
        x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, N_EMBD, 1, N_TOK);
        ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_USED, N_TOK);
        ggml_set_input(x);
        ggml_set_input(ids);
        gf = ggml_new_graph(ctx);
        for (int il = 0; il < n; ++il) {
            ggml_tensor * up   = ggml_mul_mat_id(ctx, L[il].up,   x, ids);
            ggml_tensor * gate = ggml_mul_mat_id(ctx, L[il].gate, x, ids);
            ggml_tensor * act  = ggml_swiglu_split(ctx, gate, up);
            ggml_tensor * down = ggml_mul_mat_id(ctx, L[il].down, act, ids);
            ggml_set_output(down);
            ggml_build_forward_expand(gf, down);
            outs.push_back(down);
        }
        ga = ggml_gallocr_new(buft);
        ggml_gallocr_alloc_graph(ga, gf);
    }

    std::vector<float> run(ggml_backend_t backend, const std::vector<float> & xv, const std::vector<int32_t> & iv) {
        ggml_backend_tensor_set(x,   xv.data(), 0, xv.size() * sizeof(float));
        ggml_backend_tensor_set(ids, iv.data(), 0, iv.size() * sizeof(int32_t));
        const ggml_status st = ggml_backend_graph_compute(backend, gf);
        TCHECK(st == GGML_STATUS_SUCCESS, "graph compute status %d", (int) st);
        std::vector<float> r;
        for (ggml_tensor * t : outs) {
            const size_t n0 = r.size();
            r.resize(n0 + (size_t) ggml_nelements(t));
            ggml_backend_tensor_get(t, r.data() + n0, 0, ggml_nbytes(t));
        }
        return r;
    }

    ~chain_graph() {
        if (ga) {
            ggml_gallocr_free(ga);
        }
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

static bool same_bytes(const std::vector<float> & a, const std::vector<float> & b, size_t n) {
    return a.size() >= n && b.size() >= n && memcmp(a.data(), b.data(), n * sizeof(float)) == 0;
}

static size_t dev_free(ggml_backend_dev_t dev) {
    size_t f = 0, t = 0;
    ggml_backend_dev_memory(dev, &f, &t);
    return f;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    auto gran_f   = reg ? (vmm_gran_t)   ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_granularity")   : nullptr;
    auto alloc_f  = reg ? (vmm_alloc_t)  ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_buffer_alloc")  : nullptr;
    auto map_f    = reg ? (vmm_map_t)    ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_buffer_map")    : nullptr;
    auto unmap_f  = reg ? (vmm_map_t)    ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_buffer_unmap")  : nullptr;
    auto mapped_f = reg ? (vmm_mapped_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_buffer_mapped") : nullptr;
    auto stats_f  = reg ? (pool_stats_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_pool_stats")        : nullptr;
    auto trim_f   = reg ? (pool_trim_t)  ggml_backend_reg_get_proc_address(reg, "ggml_backend_pool_trim")         : nullptr;
    if (!gran_f || !alloc_f || !map_f || !unmap_f || !mapped_f || !stats_f || !trim_f) {
        printf("test-vmm-lend: SKIPPED (no GPU backend with the virtual-memory buffer functions)\n");
        return 0;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    const size_t gran = gran_f(buft);
    if (gran == 0) {
        printf("test-vmm-lend: SKIPPED (%s has no virtual memory management)\n", ggml_backend_dev_name(dev));
        return 0;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    printf("test-vmm-lend: %s, granularity %zu KiB\n", ggml_backend_dev_name(dev), gran >> 10);

    // the weights, shared by every placement
    std::mt19937 rng(42);
    weights W;
    for (int il = 0; il < N_LAYER; ++il) {
        W.up.push_back(quantize_slots(GGML_TYPE_Q4_K, N_EMBD, N_FF, rng));
        W.gate.push_back(quantize_slots(GGML_TYPE_Q4_K, N_EMBD, N_FF, rng));
        W.down.push_back(quantize_slots(GGML_TYPE_Q5_1, N_FF, N_EMBD, rng));
    }
    std::vector<float> xv((size_t) (N_EMBD * N_TOK));
    {
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (float & v : xv) {
            v = nd(rng);
        }
    }
    // slot ids as the hot chain gets them: resident slots and the zero slot (an expert that is not hot)
    std::vector<int32_t> iv = { 0, 3, N_SLOTS, 15,   7, N_SLOTS, 1, 2,   N_SLOTS, 9, 12, 4 };

    // (0) reference: the slots in a cudaMalloc buffer
    ggml_init_params ip = { ggml_tensor_overhead() * (3 * N_LAYER + 8), nullptr, true };
    ggml_context * ctx_ref = ggml_init(ip);
    std::vector<layer_w> Lref;
    make_layers(ctx_ref, Lref);
    ggml_backend_buffer_t buf_ref = ggml_backend_alloc_ctx_tensors_from_buft(ctx_ref, buft);
    TCHECK(buf_ref != nullptr, "reference buffer");
    for (int il = 0; il < N_LAYER; ++il) {
        upload_layer(Lref, W, il);
    }
    std::vector<float> r_ref;
    {
        chain_graph g;
        g.build(Lref, N_LAYER, buft);
        r_ref = g.run(backend, xv, iv);
    }

    // (1) the same tensors on reserved virtual memory, the top 3 layers mapped apart from a granule boundary on
    ggml_context * ctx_v = ggml_init(ip);
    std::vector<layer_w> L;
    make_layers(ctx_v, L);
    const size_t align = ggml_backend_buft_get_alignment(buft);
    auto tsize = [&](const ggml_tensor * t) { return (ggml_backend_buft_get_alloc_size(buft, t) + align - 1) / align * align; };
    const int first_tail = N_LAYER - 3;
    std::vector<size_t> off;
    size_t o  = 0;
    size_t lo = 0;
    for (int il = 0; il < N_LAYER; ++il) {
        if (il == first_tail) {
            o  = (o + gran - 1) / gran * gran;
            lo = o;
        }
        for (const ggml_tensor * t : { L[il].up, L[il].gate, L[il].down }) {
            off.push_back(o);
            o += tsize(t);
        }
    }
    const size_t total = (o + gran - 1) / gran * gran;
    size_t layer_bytes = tsize(L[0].up) + tsize(L[0].gate) + tsize(L[0].down);
    printf("test-vmm-lend: %d layers of %.2f MiB, buffer %.1f MiB, tail %.1f MiB at +%.1f MiB\n", N_LAYER,
            layer_bytes / 1048576.0, total / 1048576.0, (total - lo) / 1048576.0, lo / 1048576.0);
    TCHECK(total - lo >= gran, "the tail spans at least a granule");

    const size_t free0 = dev_free(dev);
    ggml_backend_buffer_t vb = alloc_f(buft, total);
    TCHECK(vb != nullptr, "virtual-memory buffer");
    if (!vb) {
        return 1;
    }
    TCHECK(mapped_f(vb) == 0, "nothing mapped at first");
    TCHECK(!map_f(vb, 1, gran), "an offset off the granularity is refused");
    TCHECK(map_f(vb, 0, lo), "map the base");
    TCHECK(!map_f(vb, 0, gran), "a mapped range is not mapped twice");
    TCHECK(map_f(vb, lo, total - lo), "map the tail");
    TCHECK(mapped_f(vb) == total, "all mapped");
    TCHECK(!unmap_f(vb, lo + gran, total - lo - gran), "a part of a mapped segment cannot be unmapped");
    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(vb);
    {
        size_t i = 0;
        for (int il = 0; il < N_LAYER; ++il) {
            for (ggml_tensor * t : { L[il].up, L[il].gate, L[il].down }) {
                TCHECK(ggml_backend_tensor_alloc(vb, t, base + off[i++]) == GGML_STATUS_SUCCESS, "place %s", t->name);
            }
        }
    }
    ggml_backend_buffer_clear(vb, 0);
    // a raw view of the tail, as the hot set clears it after a remap
    ggml_init_params ipr = { ggml_tensor_overhead() * 2, nullptr, true };
    ggml_context * ctx_raw = ggml_init(ipr);
    ggml_tensor * raw = ggml_new_tensor_1d(ctx_raw, GGML_TYPE_I8, (int64_t) (total - lo));
    TCHECK(ggml_backend_tensor_alloc(vb, raw, base + lo) == GGML_STATUS_SUCCESS, "raw tail tensor");
    for (int il = 0; il < N_LAYER; ++il) {
        upload_layer(L, W, il);
    }
    const size_t free1 = dev_free(dev);
    printf("test-vmm-lend: free %.1f -> %.1f MiB after mapping %.1f MiB\n", free0 / 1048576.0, free1 / 1048576.0, total / 1048576.0);

    chain_graph g_all;
    g_all.build(L, N_LAYER, buft);
    chain_graph g_base;
    g_base.build(L, first_tail, buft);
    const size_t n_base = (size_t) (N_EMBD * N_USED * N_TOK) * first_tail;

    std::vector<float> r1 = g_all.run(backend, xv, iv);
    TCHECK(r1.size() == r_ref.size() && same_bytes(r1, r_ref, r_ref.size()), "(1) virtual memory gives the reference bytes");

    // (2)/(3) the tail out and back, 8 times
    int n_cycle_ok = 0;
    for (int cycle = 0; cycle < 8; ++cycle) {
        ggml_backend_synchronize(backend);
        const size_t f_in = dev_free(dev);
        const bool un = unmap_f(vb, lo, total - lo);
        const size_t f_out = dev_free(dev);
        TCHECK(un, "cycle %d: unmap the tail", cycle);
        TCHECK(mapped_f(vb) == lo, "cycle %d: only the base mapped", cycle);
        // the driver gives back at least the tail's size less one granule of rounding
        TCHECK(f_out + gran >= f_in + (total - lo), "cycle %d: free %.1f -> %.1f MiB, the tail is %.1f MiB", cycle,
                f_in / 1048576.0, f_out / 1048576.0, (total - lo) / 1048576.0);

        std::vector<float> rb = g_base.run(backend, xv, iv);
        TCHECK(same_bytes(rb, r_ref, n_base), "cycle %d: (2) base layers with the tail out", cycle);

        TCHECK(map_f(vb, lo, total - lo), "cycle %d: map the tail again", cycle);
        ggml_backend_tensor_memset(raw, 0, 0, ggml_nbytes(raw));
        for (int il = first_tail; il < N_LAYER; ++il) {
            upload_layer(L, W, il);
        }
        std::vector<float> r2 = g_all.run(backend, xv, iv);
        const bool ok = same_bytes(r2, r_ref, r_ref.size());
        TCHECK(ok, "cycle %d: (3) every layer after the refill", cycle);
        n_cycle_ok += ok ? 1 : 0;
    }
    printf("test-vmm-lend: %d of 8 out/back cycles gave the reference bytes\n", n_cycle_ok);
    // the zero slot (the last one) of a tail layer reads 0 after the remap and the clear
    {
        const ggml_tensor * d = L[N_LAYER - 1].down;
        std::vector<uint8_t> z(d->nb[2], 0xff);
        ggml_backend_tensor_get(d, z.data(), (size_t) N_SLOTS * d->nb[2], d->nb[2]);
        bool zero = true;
        for (uint8_t b : z) {
            zero = zero && b == 0;
        }
        TCHECK(zero, "the zero slot of a tail layer reads 0 after the remap");
    }

    // (4) the pool trim between CUDA-graph runs of one graph
    {
        size_t res0 = 0, hwm0 = 0;
        stats_f(backend, &res0, &hwm0);
        for (int i = 0; i < 6; ++i) {
            std::vector<float> r = g_all.run(backend, xv, iv); // repeated: the CUDA backend captures it
            TCHECK(same_bytes(r, r_ref, r_ref.size()), "run %d before the trim", i);
        }
        // a wide product grows the pool (quantized activations, MMQ scratch)
        ggml_init_params ipw = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
        ggml_context * cw = ggml_init(ipw);
        ggml_tensor * w  = ggml_new_tensor_2d(cw, GGML_TYPE_Q4_K, N_EMBD, 1024);
        ggml_tensor * xw = ggml_new_tensor_2d(cw, GGML_TYPE_F32, N_EMBD, 2048);
        ggml_tensor * yw = ggml_mul_mat(cw, w, xw);
        ggml_cgraph * gw = ggml_new_graph(cw);
        ggml_build_forward_expand(gw, yw);
        ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors_from_buft(cw, buft);
        std::vector<uint8_t> wq(ggml_nbytes(w));
        {
            std::vector<float> f((size_t) (N_EMBD * 1024));
            std::normal_distribution<float> nd(0.0f, 0.5f);
            for (float & v : f) {
                v = nd(rng);
            }
            ggml_quantize_chunk(GGML_TYPE_Q4_K, f.data(), wq.data(), 0, 1024, N_EMBD, nullptr);
            std::vector<float> xf((size_t) (N_EMBD * 2048));
            for (float & v : xf) {
                v = nd(rng);
            }
            ggml_backend_tensor_set(w,  wq.data(), 0, wq.size());
            ggml_backend_tensor_set(xw, xf.data(), 0, xf.size() * sizeof(float));
        }
        TCHECK(ggml_backend_graph_compute(backend, gw) == GGML_STATUS_SUCCESS, "wide product");
        std::vector<float> yw0((size_t) ggml_nelements(yw));
        ggml_backend_tensor_get(yw, yw0.data(), 0, ggml_nbytes(yw));
        size_t res1 = 0, hwm1 = 0;
        stats_f(backend, &res1, &hwm1);
        const size_t f_before = dev_free(dev);
        const size_t freed = trim_f(backend, 0);
        const size_t f_after = dev_free(dev);
        size_t res2 = 0, hwm2 = 0;
        stats_f(backend, &res2, &hwm2);
        printf("test-vmm-lend: pool %.1f MiB (high-water %.1f) before the wide product, %.1f (%.1f) after it; trim gave back "
                "%.1f MiB (free %.1f -> %.1f MiB), %.1f left\n", res0 / 1048576.0, hwm0 / 1048576.0, res1 / 1048576.0,
                hwm1 / 1048576.0, freed / 1048576.0, f_before / 1048576.0, f_after / 1048576.0, res2 / 1048576.0);
        TCHECK(res1 > 0 && hwm1 > 0, "the wide product used the pool");
        TCHECK(freed == res1 - res2 && res2 == 0, "trim(0) gives everything back");
        TCHECK(f_after + gran >= f_before + freed, "the driver has the trimmed memory back");
        for (int i = 0; i < 6; ++i) {
            std::vector<float> r = g_all.run(backend, xv, iv);
            TCHECK(same_bytes(r, r_ref, r_ref.size()), "run %d after the trim", i);
        }
        TCHECK(ggml_backend_graph_compute(backend, gw) == GGML_STATUS_SUCCESS, "wide product after the trim");
        std::vector<float> yw1((size_t) ggml_nelements(yw));
        ggml_backend_tensor_get(yw, yw1.data(), 0, ggml_nbytes(yw));
        TCHECK(same_bytes(yw0, yw1, yw0.size()), "the wide product gives the same bytes after the trim");
        ggml_backend_buffer_free(bw);
        ggml_free(cw);
    }

    // the buffer goes, and its VRAM with it
    ggml_backend_synchronize(backend);
    const size_t f_pre = dev_free(dev);
    ggml_backend_buffer_free(vb);
    const size_t f_post = dev_free(dev);
    TCHECK(f_post + gran >= f_pre + total, "freeing the buffer gives its %.1f MiB back (free %.1f -> %.1f MiB)",
            total / 1048576.0, f_pre / 1048576.0, f_post / 1048576.0);

    ggml_free(ctx_raw);
    ggml_free(ctx_v);
    ggml_backend_buffer_free(buf_ref);
    ggml_free(ctx_ref);
    ggml_backend_free(backend);

    printf("test-vmm-lend: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
