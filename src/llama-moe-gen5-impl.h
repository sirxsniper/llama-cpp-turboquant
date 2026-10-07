#pragma once

// [TAG_MOE_DMA_SHARE] [TAG_FN_PREFILL_STREAM] helpers shared by llama-moe-dma.cpp and llama-prefill-stream.cpp

#include "ggml-backend.h"

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

// a memcpy split over worker threads; copy() blocks until every part is done
struct copy_seg {
    uint8_t *       dst;
    const uint8_t * src;
    size_t          n;
};

class copy_pool {
public:
    ~copy_pool() { stop(); }
    void start(int n_threads);
    void stop();
    void copy(void * dst, const void * src, size_t n);
    // [TAG_FN_L14_PFSD2D] the segments as one byte range split over the threads (one wake for many small pieces)
    void copy_list(const copy_seg * segs, int n);
    int  size() const { return (int) workers.size() + 1; }

private:
    void run(int idx);

    std::vector<std::thread> workers;
    std::mutex               mtx;
    std::condition_variable  cv_go;
    std::condition_variable  cv_done;
    uint64_t                 gen     = 0;
    int                      pending = 0;
    bool                     quit    = false;
    const copy_seg *         segs_p  = nullptr;
    int                      n_segs  = 0;
    size_t                   n_bytes = 0;
};

} // namespace gen5
