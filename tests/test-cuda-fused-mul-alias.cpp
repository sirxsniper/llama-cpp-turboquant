// [TAG_FUSED_BINBCAST_ALIAS] GPU test of the CUDA multi-mul fusion (ggml_cuda_try_fuse, ADD / MUL chains): the fused
// kernel reads every src of the chain while it writes the output of the last node, but the graph allocator only knows the
// unfused chain and may place that output over a broadcast src1 that an earlier node of the chain was the last to read.
// The MoE down projection with the amax rescale (#30861) has such a chain: MUL by the expert scale, MUL by the rescale
// factor e, MUL by the router weights; on Flint with an expert layer split by -fit the output landed on e and the prompt
// ubatches came out as garbage (PPL 6 -> 90).
// Here the tensors are placed by hand in one device buffer: the output of the last MUL starts where e lives, every other
// tensor has its own range. One node at a time this is well defined (e is read by the second MUL, then overwritten by
// the third), so the device result must equal x * s * e * w computed on the host.
// Built, not registered with ctest: it needs the GPU. Exit code 0 iff every check passes; prints SKIPPED without a GPU.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static bool run_case(ggml_backend_t backend, int64_t n, int64_t k, int64_t t) {
    ggml_init_params ip = { 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * x  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n, k, t);
    ggml_tensor * s  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, k, t);
    ggml_tensor * e  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, k, t);
    ggml_tensor * w  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, k, t);
    ggml_tensor * m0 = ggml_mul(ctx, x, s);
    ggml_tensor * m1 = ggml_mul(ctx, m0, e);
    ggml_tensor * m2 = ggml_mul(ctx, m1, w);
    ggml_set_output(m2);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, m2);

    // x, s, w, m0, m1 each in their own range; m2 starts at e (e is dead once m1 is computed)
    const size_t align = 256;
    auto pad = [&](size_t v) { return (v + align - 1) / align * align; };
    const size_t big = pad(ggml_nbytes(x));
    const size_t sml = pad(ggml_nbytes(s));
    ggml_backend_buffer_t buf = ggml_backend_alloc_buffer(backend, 4*big + 3*sml + align);
    char * base = (char *) ggml_backend_buffer_get_base(buf);
    size_t off = 0;
    auto place = [&](ggml_tensor * tt, size_t size) {
        ggml_backend_tensor_alloc(buf, tt, base + off);
        off += size;
    };
    place(x,  big);
    place(m0, big);
    place(m1, big);
    place(s,  sml);
    place(w,  sml);
    ggml_backend_tensor_alloc(buf, e,  base + off);
    ggml_backend_tensor_alloc(buf, m2, base + off);

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> u(0.5f, 2.0f);
    std::vector<float> xv(ggml_nelements(x)), sv(ggml_nelements(s)), ev(ggml_nelements(e)), wv(ggml_nelements(w));
    for (auto * v : { &xv, &sv, &ev, &wv }) {
        for (float & f : *v) {
            f = u(rng);
        }
    }
    ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(s, sv.data(), 0, ggml_nbytes(s));
    ggml_backend_tensor_set(e, ev.data(), 0, ggml_nbytes(e));
    ggml_backend_tensor_set(w, wv.data(), 0, ggml_nbytes(w));

    bool ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;

    std::vector<float> out(ggml_nelements(m2));
    ggml_backend_tensor_get(m2, out.data(), 0, ggml_nbytes(m2));

    double max_rel = 0.0;
    for (int64_t it = 0; it < t; ++it) {
        for (int64_t ik = 0; ik < k; ++ik) {
            const int64_t r = it*k + ik;
            for (int64_t in = 0; in < n; ++in) {
                const float ref = ((xv[r*n + in] * sv[r]) * ev[r]) * wv[r];
                max_rel = std::fmax(max_rel, std::fabs(out[r*n + in] - ref) / std::fabs(ref));
            }
        }
    }
    ok = ok && max_rel < 1e-6;
    printf("  n=%5lld k=%lld t=%3lld: max rel err %.3g %s\n", (long long) n, (long long) k, (long long) t, max_rel, ok ? "OK" : "FAIL");

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        printf("test-cuda-fused-mul-alias: SKIPPED (no GPU backend)\n");
        return 0;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    printf("test-cuda-fused-mul-alias: %s\n", ggml_backend_dev_name(dev));

    bool ok = true;
    // decode and prompt shapes of the Flint down projection (n_embd 2048, 6 experts), and a small one
    for (int rep = 0; rep < 3; ++rep) {
        ok = run_case(backend, 64,   6, 4)   && ok;
        ok = run_case(backend, 2048, 6, 1)   && ok;
        ok = run_case(backend, 2048, 6, 128) && ok;
    }

    ggml_backend_free(backend);
    printf("test-cuda-fused-mul-alias: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
