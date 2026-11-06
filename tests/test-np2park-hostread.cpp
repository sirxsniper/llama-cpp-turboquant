// [TAG_NP2PARK_HOSTREAD] a state save of a parked recurrent cell (llama_rs_write_tensor_host, src/llama-memory-recurrent.h):
// the tensor bytes come from the device and the whole granules of a parked cell from its host copy, and the result has the
// bytes and the order of one io.write_tensor. The layout is the NP2PARK one (rows plane-major: row = group * size + cell;
// a whole granule inside a row belongs to that row's cell), so the rows of a snapshot group (a pending rollback) are
// covered too. On the "device" a parked granule holds other bytes, as the shared alias granule would. CPU only, no model.

#include "../src/llama-memory-recurrent.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <utility>
#include <vector>

static int g_fail   = 0;
static int g_checks = 0;

#define TCHECK(cond, ...)                      \
    do {                                       \
        g_checks++;                            \
        if (!(cond)) {                         \
            g_fail++;                          \
            fprintf(stderr, "FAIL: ");         \
            fprintf(stderr, __VA_ARGS__);      \
            fprintf(stderr, "\n");             \
        }                                      \
    } while (0)

// copies at once: write() appends the bytes, write_tensor() the tensor bytes; remembers the tensor ranges it read
struct copy_writer : llama_io_write_i {
    std::vector<uint8_t>                   out;
    std::vector<std::pair<size_t, size_t>> reads;
    size_t                                 n_zero = 0;

    void write(const void * src, size_t n) override {
        n_zero += n == 0;
        out.insert(out.end(), (const uint8_t *) src, (const uint8_t *) src + n);
    }

    void write_tensor(ggml_tensor * t, size_t off, size_t n) override {
        n_zero += n == 0;
        reads.push_back({ off, n });
        const size_t o = out.size();
        out.resize(o + n);
        ggml_backend_tensor_get(t, out.data() + o, off, n);
    }

    size_t n_bytes() override {
        return out.size();
    }
};

struct layout {
    size_t   row;    // bytes per row
    size_t   gran;   // granule bytes
    uint32_t n_cell; // cells (size)
    uint32_t n_grp;  // 1 + n_rs_seq groups
};

static void run_layout(const layout & L, std::mt19937 & rng) {
    const size_t rows  = (size_t) L.n_cell * L.n_grp;
    const size_t bytes = rows * L.row;

    ggml_init_params ip = { 2 * ggml_tensor_overhead(), nullptr, true };
    ggml_context_ptr ctx { ggml_init(ip) };
    ggml_tensor * t = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I8, (int64_t) L.row, (int64_t) rows);
    ggml_backend_buffer_ptr buf { ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type()) };
    TCHECK(buf != nullptr, "buffer");
    if (!buf) {
        return;
    }

    std::vector<uint8_t> truth(bytes);
    for (auto & b : truth) {
        b = (uint8_t) (rng() & 0xff);
    }
    ggml_backend_tensor_set(t, truth.data(), 0, bytes);

    // the granules: whole ones inside a row belong to cell row % n_cell (the tensor starts on a granule)
    struct gran { size_t off; uint32_t cell; };
    std::vector<gran> grans;
    for (size_t r = 0; r < rows; ++r) {
        for (size_t g = (r * L.row + L.gran - 1) / L.gran; (g + 1) * L.gran <= (r + 1) * L.row; ++g) {
            grans.push_back({ g * L.gran, (uint32_t) (r % L.n_cell) });
        }
    }

    for (uint32_t parked_mask = 0; parked_mask < (1u << L.n_cell); ++parked_mask) {
        // park: the host copy gets the bytes, the device granule other bytes (the alias)
        std::map<size_t, std::vector<uint8_t>> host;
        ggml_backend_tensor_set(t, truth.data(), 0, bytes);
        for (const auto & g : grans) {
            if (parked_mask & (1u << g.cell)) {
                host[g.off].assign(truth.begin() + g.off, truth.begin() + g.off + L.gran);
                std::vector<uint8_t> alias(L.gran, 0xee);
                ggml_backend_tensor_set(t, alias.data(), g.off, L.gran);
            }
        }

        auto check = [&](size_t off, size_t n, const char * what) {
            // the spans as the memory gathers them: rows in order, each row's parked granules in order
            std::vector<llama_rs_host_span> spans;
            for (size_t r = off / L.row; r * L.row < off + n; ++r) {
                for (const auto & g : grans) {
                    if (g.off >= r * L.row && g.off < (r + 1) * L.row && (parked_mask & (1u << g.cell))) {
                        spans.push_back({ g.off, L.gran, host[g.off].data() });
                    }
                }
            }
            copy_writer w;
            llama_rs_write_tensor_host(w, t, off, n, spans);
            const bool same = w.out.size() == n && memcmp(w.out.data(), truth.data() + off, n) == 0;
            TCHECK(same, "row %zu gran %zu cells %u groups %u parked 0x%x %s [%zu, %zu): bytes differ", L.row, L.gran,
                    L.n_cell, L.n_grp, parked_mask, what, off, off + n);
            TCHECK(w.n_zero == 0, "%s [%zu, %zu): an empty write", what, off, off + n);
            bool alias_read = false;
            for (const auto & rd : w.reads) {
                for (const auto & g : grans) {
                    const bool parked = parked_mask & (1u << g.cell);
                    alias_read = alias_read || (parked && rd.first < g.off + L.gran && g.off < rd.first + rd.second);
                }
            }
            TCHECK(!alias_read, "parked 0x%x %s [%zu, %zu): a tensor read reached a parked granule", parked_mask, what, off,
                    off + n);
            if (spans.empty()) {
                TCHECK(w.reads.size() == 1, "%s [%zu, %zu): not one tensor read without parked granules", what, off, off + n);
            }
        };

        // what state_write_data reads: a range of cells in one group (a pending rollback reads group > 0)
        for (uint32_t grp = 0; grp < L.n_grp; ++grp) {
            for (uint32_t c0 = 0; c0 < L.n_cell; ++c0) {
                for (uint32_t c1 = c0 + 1; c1 <= L.n_cell; ++c1) {
                    const size_t r0 = (size_t) grp * L.n_cell + c0;
                    check(r0 * L.row, (size_t) (c1 - c0) * L.row, "cells");
                }
            }
        }
        // any byte range, also cutting through granules
        for (int i = 0; i < 200; ++i) {
            const size_t a = rng() % bytes;
            const size_t b = a + 1 + rng() % (bytes - a);
            check(a, b - a, "bytes");
        }
        check(0, bytes, "all");
    }
}

int main() {
    std::mt19937 rng(1234);
    // Flash-Next: a 3 MiB S row holds one 2 MiB granule (at its start or its end); also two per row, exact fits, small rows
    const layout layouts[] = {
        {  96, 64, 2, 4 },
        {  96, 64, 3, 1 },
        { 160, 64, 2, 2 },
        {  64, 64, 2, 3 },
        {  48, 64, 2, 4 },
        { 200, 64, 3, 2 },
    };
    for (const auto & L : layouts) {
        run_layout(L, rng);
    }
    printf("test-np2park-hostread: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
