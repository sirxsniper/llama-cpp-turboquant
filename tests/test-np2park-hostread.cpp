// [TAG_NP2PARK_HOSTREAD] a state save of a parked recurrent cell, with the code state_write_data runs (src/llama-memory-recurrent.h):
//  - llama_rs_write_tensor_parked reads the whole granules of a parked cell from its host copy and the rest from the device,
//    with the bytes and the order of one io.write_tensor
//  - llama_rs_cells_of_rows gives the cells an on-device save maps first
// The layout is the NP2PARK one (r35_alloc): rows plane-major (row = group * size + cell), a tensor whose rows reach a
// granule starts on a granule, a whole granule inside a row belongs to that row's cell, and a cell's list holds its
// granules of every tensor (two here, as the S tensors of two layers). Ranges of every group are read, so the rows of a
// pending rollback (group > 0) are covered. On the "device" a parked granule holds other bytes, as the shared alias
// granule would. CPU only, no model.

#include "../src/llama-memory-recurrent.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstdio>
#include <cstring>
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

static constexpr int N_T = 2;

static void run_layout(const layout & L, std::mt19937 & rng) {
    const size_t rows  = (size_t) L.n_cell * L.n_grp;
    const size_t bytes = rows * L.row;

    ggml_init_params ip = { N_T * ggml_tensor_overhead(), nullptr, true };
    ggml_context_ptr ctx { ggml_init(ip) };
    ggml_tensor * ts[N_T];
    for (int ti = 0; ti < N_T; ++ti) {
        ts[ti] = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I8, (int64_t) L.row, (int64_t) rows);
    }
    ggml_backend_buffer_ptr buf { ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type()) };
    TCHECK(buf != nullptr, "buffer");
    if (!buf) {
        return;
    }

    std::vector<uint8_t> truth[N_T];
    for (int ti = 0; ti < N_T; ++ti) {
        truth[ti].resize(bytes);
        for (auto & b : truth[ti]) {
            b = (uint8_t) (rng() & 0xff);
        }
    }

    // the cells' granule lists as r35_alloc builds them: tensor by tensor, row by row (offsets relative to the tensor)
    std::vector<llama_rs_park_cell> cells(L.n_cell);
    for (int ti = 0; ti < N_T; ++ti) {
        for (size_t r = 0; r < rows; ++r) {
            for (size_t g = (r * L.row + L.gran - 1) / L.gran; (g + 1) * L.gran <= (r + 1) * L.row; ++g) {
                cells[r % L.n_cell].grans.push_back({ ts[ti], g * L.gran, g * L.gran });
            }
        }
    }
    auto tensor_index = [&](const ggml_tensor * t) {
        return t == ts[0] ? 0 : 1;
    };

    for (uint32_t parked_mask = 0; parked_mask < (1u << L.n_cell); ++parked_mask) {
        // park: the host copy gets the bytes (granule k at k * gran), the device granule other bytes (the alias)
        for (int ti = 0; ti < N_T; ++ti) {
            ggml_backend_tensor_set(ts[ti], truth[ti].data(), 0, bytes);
        }
        const std::vector<uint8_t> alias(L.gran, 0xee);
        for (uint32_t c = 0; c < L.n_cell; ++c) {
            llama_rs_park_cell & pc = cells[c];
            pc.parked = (parked_mask >> c) & 1;
            pc.host.clear();
            if (!pc.parked) {
                continue;
            }
            pc.host.resize(pc.grans.size() * L.gran);
            for (size_t k = 0; k < pc.grans.size(); ++k) {
                const llama_rs_park_gran & gr = pc.grans[k];
                memcpy(pc.host.data() + k * L.gran, truth[tensor_index(gr.t)].data() + gr.off_t, L.gran);
                ggml_backend_tensor_set(gr.t, alias.data(), gr.off_t, L.gran);
            }
        }

        auto check = [&](int ti, size_t off, size_t n, const char * what) {
            copy_writer w;
            llama_rs_write_tensor_parked(w, ts[ti], off, n, cells, L.gran);
            const bool same = w.out.size() == n && memcmp(w.out.data(), truth[ti].data() + off, n) == 0;
            TCHECK(same, "row %zu gran %zu cells %u groups %u parked 0x%x tensor %d %s [%zu, %zu): bytes differ", L.row,
                    L.gran, L.n_cell, L.n_grp, parked_mask, ti, what, off, off + n);
            TCHECK(w.n_zero == 0, "tensor %d %s [%zu, %zu): an empty write", ti, what, off, off + n);
            bool alias_read  = false;
            bool parked_in_n = false;
            for (const auto & pc : cells) {
                for (const auto & gr : pc.grans) {
                    if (!pc.parked || gr.t != ts[ti]) {
                        continue;
                    }
                    parked_in_n = parked_in_n || (off < gr.off_t + L.gran && gr.off_t < off + n);
                    for (const auto & rd : w.reads) {
                        alias_read = alias_read || (rd.first < gr.off_t + L.gran && gr.off_t < rd.first + rd.second);
                    }
                }
            }
            TCHECK(!alias_read, "parked 0x%x tensor %d %s [%zu, %zu): a tensor read reached a parked granule", parked_mask, ti,
                    what, off, off + n);
            if (!parked_in_n) {
                TCHECK(w.reads.size() == 1, "tensor %d %s [%zu, %zu): not one tensor read without parked granules", ti, what,
                        off, off + n);
            }
        };

        // what state_write_data reads: a range of cells in one group (a pending rollback reads group > 0), and the cells an
        // on-device save maps for it
        for (int ti = 0; ti < N_T; ++ti) {
            for (uint32_t grp = 0; grp < L.n_grp; ++grp) {
                for (uint32_t c0 = 0; c0 < L.n_cell; ++c0) {
                    for (uint32_t c1 = c0 + 1; c1 <= L.n_cell; ++c1) {
                        const uint32_t r0 = grp * L.n_cell + c0;
                        check(ti, (size_t) r0 * L.row, (size_t) (c1 - c0) * L.row, "cells");

                        std::vector<uint32_t> need;
                        llama_rs_cells_of_rows(need, { { r0, r0 + (c1 - c0) } }, L.n_cell);
                        bool need_ok = need.size() == c1 - c0;
                        for (size_t i = 0; need_ok && i < need.size(); ++i) {
                            need_ok = need[i] == c0 + i;
                        }
                        TCHECK(need_ok, "cells %u groups %u: rows [%u, %u) do not map cells [%u, %u)", L.n_cell, L.n_grp, r0,
                                r0 + (c1 - c0), c0, c1);
                    }
                }
            }
            // any byte range, also cutting through granules
            for (int i = 0; i < 100; ++i) {
                const size_t a = rng() % bytes;
                const size_t b = a + 1 + rng() % (bytes - a);
                check(ti, a, b - a, "bytes");
            }
            check(ti, 0, bytes, "all");
        }
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
