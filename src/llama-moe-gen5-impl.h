#pragma once

// [TAG_MOE_DMA_SHARE] [TAG_FN_PREFILL_STREAM] helpers shared by llama-moe-dma.cpp and llama-prefill-stream.cpp

#include "ggml-backend.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace gen5 {

// host memory for copies to the device: pinned (the device's host buffer type, counted against the gen5 cap of
// LLAMA_GEN5_PIN_MAX_MIB <= 4096) or, for a CPU device (tests), plain memory
struct host_mem {
    ggml_backend_buffer_t buf    = nullptr;
    std::vector<uint8_t>  plain;
    uint8_t *             ptr    = nullptr;
    size_t                size   = 0;
    bool                  pinned = false;

    host_mem() = default;
    host_mem(const host_mem &) = delete;
    host_mem & operator=(const host_mem &) = delete;
    host_mem(host_mem && o) noexcept { *this = std::move(o); }
    host_mem & operator=(host_mem && o) noexcept {
        if (this != &o) {
            release();
            buf    = o.buf;
            plain  = std::move(o.plain);
            ptr    = o.ptr;
            size   = o.size;
            pinned = o.pinned;
            o.buf  = nullptr;
            o.ptr  = nullptr;
            o.size = 0;
        }
        return *this;
    }
    ~host_mem();

    bool alloc(ggml_backend_dev_t dev, size_t n, const char * what);
    void release();
};

size_t pin_cap();

bool env_flag(const char * name);
int  env_int(const char * name, int def, int lo, int hi);

// true when sched is null or runs backend b (the owner context's graphs)
bool sched_has(ggml_backend_sched_t sched, ggml_backend_t b);

// events that may be null (CPU device): record/wait/sync fall back to a synchronize of src
ggml_backend_event_t ev_new(ggml_backend_dev_t dev);
void ev_record(ggml_backend_event_t ev, ggml_backend_t b);
void ev_wait(ggml_backend_t b, ggml_backend_event_t ev, ggml_backend_t src);
void ev_sync(ggml_backend_event_t ev, ggml_backend_t src);

// llama-moe-dma.cpp: LLAMA_MOE_DMA_SHARE or LLAMA_MOE_PREFETCH is set
bool dma_requested();

// [TAG_FN_R25_PIN] an affinity: a processor group and a mask of its logical CPUs
struct cpu_mask {
    int      group = 0;
    uint64_t mask  = 0;
};
// [TAG_FN_R25_PIN] the copy workers' affinities, taken in turn by the workers. mode 1: one logical CPU per physical core (the
// core's last one: the SMT sibling a --cpu-mask 0x5555... leaves free), the cores in turn from each L3 domain; mode 2: the
// whole L3 domains (the CCDs) in turn, so that both CCDs' fabric links carry the copies while the scheduler still moves a
// worker inside its CCD. Empty when the topology cannot be read (or not on Windows) or mode 0
std::vector<cpu_mask> copy_affinities(int mode);

// a memcpy split over worker threads; copy() blocks until every part is done
struct copy_seg {
    uint8_t *       dst;
    const uint8_t * src;
    size_t          n;
};

class copy_pool {
public:
    ~copy_pool() { stop(); }
    // [TAG_FN_R25_STAGE] call before start(). temporal: plain (cached) stores instead of non-temporal ones, for a ring small
    // enough to stay in the L3 (the DMA then reads the lines from the cache: the ring costs no DRAM write and no DRAM read);
    // prefetch: the source lines with the NTA hint (they pass the caches without pushing the ring out); spin_us: the workers
    // and the caller poll this long for the next job / the last part before they sleep (0: sleep at once, as before)
    void set_mode(bool temporal, bool prefetch, int spin_us);
    // [TAG_FN_R25_PIN] call before start(): worker i runs on affs[(i - 1) % size]; the caller keeps its affinity. Empty: none
    void set_affinities(const std::vector<cpu_mask> & affs);
    // [TAG_FN_R25_PIN] call before start(): > 0 the threads take units of this many bytes from a shared counter instead of a
    // fixed part each (a worker that the scheduler stops holds one unit, not 1/n of the chunk); 0: fixed parts (as before)
    void set_grain(size_t bytes);
    void start(int n_threads);
    void stop();
    void copy(void * dst, const void * src, size_t n);
    // [TAG_FN_L14_PFSD2D] the segments as one byte range split over the threads (one wake for many small pieces)
    void copy_list(const copy_seg * segs, int n);
    // [TAG_FN_R25_STAGE] the same copy by the workers alone: submit() returns at once (the caller dispatches the chunk before
    // meanwhile), wait() blocks until it is done. The segments must stay unchanged until wait() returns. A pool without
    // workers, or less than 1 MiB, copies inside submit()
    void submit(const copy_seg * segs, int n);
    void wait();
    int  size() const { return (int) workers.size() + 1; }

private:
    void run(int idx);

    std::vector<std::thread> workers;
    std::mutex               mtx;
    std::condition_variable  cv_go;
    std::condition_variable  cv_done;
    // [TAG_FN_R25_STAGE] gen / pending / quit are written under mtx (pending also by the workers' decrement) and polled
    // without it while a thread spins
    std::atomic<uint64_t>    gen{0};
    std::atomic<int>         pending{0};
    std::atomic<bool>        quit{false};
    const copy_seg *         segs_p  = nullptr;
    int                      n_segs  = 0;
    size_t                   n_bytes = 0;
    int                      job_base  = 0; // [TAG_FN_R25_STAGE] worker idx - job_base = its part of job_parts (copy_list: 0 /
    int                      job_parts = 1; // size(), the caller copies part 0; submit: 1 / size() - 1)
    bool                     temporal = false; // [TAG_FN_R25_STAGE]
    bool                     prefetch = false;
    int                      spin_us  = 0;
    std::vector<cpu_mask>    affs;             // [TAG_FN_R25_PIN]
    size_t                   grain = 0;        // [TAG_FN_R25_PIN] 0: fixed parts
    std::atomic<size_t>      next_unit{0};     // the next unit of the job (grain > 0)
    void copy_units(const copy_seg * segs, int n_sg, size_t n);
};

} // namespace gen5
