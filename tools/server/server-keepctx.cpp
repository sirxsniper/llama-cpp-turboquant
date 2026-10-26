#include "server-keepctx.h"
#include "server-common.h"
#include "server-convstore.h" // [TAG_FN_L19_CONVSTORE]

#include "llama.h"
#include "log.h"
#include "speculative.h"
#include "../../src/llama-ext.h" // [TAG_FN_L18_KEEPCTX] llama_state_seq_write_stream_ext

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <mutex>
#include <system_error>
#include <thread>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h> // [TAG_FN_V0254_STORE] page faults, working set (the K32 functions of kernel32)
#include <winioctl.h> // [TAG_FN_V0254_STORE] the disk of a volume, its seek penalty
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

// unbuffered I/O moves whole sectors: every block is a multiple of this, the last one is padded with zeros
static constexpr size_t KC_ALIGN = 4096;
static constexpr size_t KC_BLOCK = 16u*1024*1024;
static constexpr int    KC_NBUF  = 3;

static size_t kc_round_up(size_t n) {
    return (n + KC_ALIGN - 1) / KC_ALIGN * KC_ALIGN;
}

static double kc_ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// FNV-1a, for params.verify only
static uint64_t kc_hash(uint64_t h, const void * data, size_t n) {
    const uint8_t * p = (const uint8_t *) data;
    for (size_t i = 0; i < n; ++i) {
        h = (h ^ p[i]) * 0x100000001b3ull;
    }
    return h;
}

static constexpr uint64_t KC_HASH_SEED = 0xcbf29ce484222325ull;

// [TAG_FN_V0254_STORE] one moment of the server thread for the trace: wall clock, the thread's CPU time (user, kernel)
// and the process's page faults. A part whose wall time is far above its CPU time waited (disk, paging, device).
struct kc_mark {
    std::chrono::steady_clock::time_point t;
    uint64_t user_100ns = 0;
    uint64_t kern_100ns = 0;
    uint64_t pf         = 0;

    static kc_mark now() {
        kc_mark m;
        m.t = std::chrono::steady_clock::now();
#if defined(_WIN32)
        FILETIME c, e, k, u;
        if (GetThreadTimes(GetCurrentThread(), &c, &e, &k, &u)) {
            m.user_100ns = ((uint64_t) u.dwHighDateTime << 32) | u.dwLowDateTime;
            m.kern_100ns = ((uint64_t) k.dwHighDateTime << 32) | k.dwLowDateTime;
        }
        PROCESS_MEMORY_COUNTERS pmc = {};
        if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
            m.pf = pmc.PageFaultCount;
        }
#endif
        return m;
    }
};

// "wall/user/kernel ms, faults" of the part from a to b
static std::string kc_part(const kc_mark & a, const kc_mark & b) {
    return string_format("%.0f/%.0f/%.0f ms %llu pf", std::chrono::duration<double, std::milli>(b.t - a.t).count(),
                         (b.user_100ns - a.user_100ns)/1e4, (b.kern_100ns - a.kern_100ns)/1e4, (unsigned long long) (b.pf - a.pf));
}

// [TAG_FN_V0254_STORE] the bytes of [p, p + n) outside the process's working set (sampled per page)
static uint64_t kc_nonresident(const void * p, size_t n) {
#if defined(_WIN32)
    if (p == nullptr || n == 0) {
        return 0;
    }
    const uintptr_t lo = (uintptr_t) p & ~(uintptr_t) (KC_ALIGN - 1);
    const uintptr_t hi = (uintptr_t) p + n;
    std::vector<PSAPI_WORKING_SET_EX_INFORMATION> ws;
    for (uintptr_t a = lo; a < hi; a += KC_ALIGN) {
        PSAPI_WORKING_SET_EX_INFORMATION e = {};
        e.VirtualAddress = (PVOID) a;
        ws.push_back(e);
    }
    if (!K32QueryWorkingSetEx(GetCurrentProcess(), ws.data(), (DWORD) (ws.size()*sizeof(ws[0])))) {
        return 0;
    }
    uint64_t out = 0;
    for (const auto & e : ws) {
        out += e.VirtualAttributes.Valid ? 0 : KC_ALIGN;
    }
    return out;
#else
    (void) p; (void) n;
    return 0;
#endif
}

static unsigned long kc_pid() {
#if defined(_WIN32)
    return (unsigned long) GetCurrentProcessId();
#else
    return (unsigned long) getpid();
#endif
}

//
// one store file: no OS file cache, deleted when closed
//

class server_keepctx_file {
public:
    static std::shared_ptr<server_keepctx_file> create(const std::filesystem::path & path, std::string & err) {
        auto f = std::shared_ptr<server_keepctx_file>(new server_keepctx_file());
#if defined(_WIN32)
        f->h = CreateFileW(path.wstring().c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                           FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE | FILE_FLAG_NO_BUFFERING, nullptr);
        if (f->h == INVALID_HANDLE_VALUE) {
            err = "CreateFileW failed, error " + std::to_string((unsigned long) GetLastError());
            return nullptr;
        }
#else
        // always a new file: an existing name (a planted file or link) fails instead of being reused
        int flags = O_RDWR | O_CREAT | O_EXCL;
#ifdef O_NOFOLLOW
        flags |= O_NOFOLLOW;
#endif
#ifdef O_CLOEXEC
        flags |= O_CLOEXEC;
#endif
        f->fd = open(path.string().c_str(), flags, 0600);
        if (f->fd < 0) {
            err = std::string("open failed: ") + strerror(errno);
            return nullptr;
        }
        unlink(path.string().c_str()); // the file lives until the descriptor closes
#endif
        return f;
    }

    ~server_keepctx_file() {
#if defined(_WIN32)
        if (h != INVALID_HANDLE_VALUE) {
            CloseHandle(h);
        }
#else
        if (fd >= 0) {
            close(fd);
        }
#endif
    }

    // n and off are multiples of KC_ALIGN, buf is KC_ALIGN aligned
    bool pwrite(const void * buf, size_t n, uint64_t off) {
#if defined(_WIN32)
        OVERLAPPED ov = {};
        ov.Offset     = (DWORD) (off & 0xffffffffu);
        ov.OffsetHigh = (DWORD) (off >> 32);
        DWORD done = 0;
        return WriteFile(h, buf, (DWORD) n, &done, &ov) && done == n;
#else
        const uint8_t * p = (const uint8_t *) buf;
        while (n > 0) {
            const ssize_t r = ::pwrite(fd, p, n, (off_t) off);
            if (r <= 0) {
                return false;
            }
            p += r; n -= (size_t) r; off += (uint64_t) r;
        }
        return true;
#endif
    }

    bool pread(void * buf, size_t n, uint64_t off) {
#if defined(_WIN32)
        OVERLAPPED ov = {};
        ov.Offset     = (DWORD) (off & 0xffffffffu);
        ov.OffsetHigh = (DWORD) (off >> 32);
        DWORD done = 0;
        return ReadFile(h, buf, (DWORD) n, &done, &ov) && done == n;
#else
        uint8_t * p = (uint8_t *) buf;
        while (n > 0) {
            const ssize_t r = ::pread(fd, p, n, (off_t) off);
            if (r <= 0) {
                return false;
            }
            p += r; n -= (size_t) r; off += (uint64_t) r;
        }
        return true;
#endif
    }

private:
    server_keepctx_file() = default;
#if defined(_WIN32)
    HANDLE h = INVALID_HANDLE_VALUE;
#else
    int fd = -1;
#endif
};

// the KC_NBUF blocks of a pipe (sector aligned for unbuffered I/O)
static uint8_t * kc_alloc_blocks() {
#if defined(_WIN32)
    return (uint8_t *) VirtualAlloc(nullptr, KC_NBUF*KC_BLOCK, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
    return (uint8_t *) std::aligned_alloc(KC_ALIGN, KC_NBUF*KC_BLOCK);
#endif
}

static void kc_free_blocks(uint8_t * mem) {
#if defined(_WIN32)
    VirtualFree(mem, 0, MEM_RELEASE);
#else
    std::free(mem);
#endif
}

//
// a thread moves blocks between a few aligned buffers and the file, so the device copies of the caller and the disk
// overlap. Write: put() fills blocks, the thread writes them in order. Read: the thread reads ahead, get() drains.
//

class kc_pipe {
public:
    // [TAG_FN_L19_CONVSTORE_SPILL] base: where in the file the pipe starts (a multiple of KC_ALIGN)
    // [TAG_FN_L20_CONVFIX_IO] ext_mem: KC_NBUF blocks the caller owns (kc_alloc_blocks), else the pipe allocates its own
    kc_pipe(server_keepctx_file & file, bool writing, uint64_t n_read, uint64_t base = 0, uint8_t * ext_mem = nullptr,
            bool trace = false) :
            trace(trace), file(file), writing(writing), n_read(kc_round_up(n_read)), base(base) {
        llama_fn_pin_server_busy(1); // [TAG_FN_L23_PINFIX] R-L23-8: a keepctx copy is busy (the destructor ends it)
        if (ext_mem) {
            mem     = ext_mem;
            own_mem = false;
        } else {
            mem = kc_alloc_blocks();
        }
        if (mem == nullptr) {
            err = true;
            return;
        }
        for (int b = 0; b < KC_NBUF; ++b) {
            q_free.push_back(b);
        }
        th = std::thread([this]() { this->writing ? run_write() : run_read(); });
    }

    ~kc_pipe() {
        llama_fn_pin_server_busy(-1); // [TAG_FN_L23_PINFIX] R-L23-8
        {
            std::lock_guard<std::mutex> lk(mu);
            stop = true;
        }
        cv.notify_all();
        if (th.joinable()) {
            th.join();
        }
        if (mem && own_mem) {
            kc_free_blocks(mem);
        }
    }

    bool     hashing = false;
    uint64_t hash    = KC_HASH_SEED;

    // [TAG_FN_V0254_STORE] trace: the caller's wait for a block (write: the disk is behind; read: not read yet), its
    // copies of big pieces, and the thread's file I/O (sum, longest, count), in us
    const bool           trace;
    int64_t              us_wait = 0;
    int64_t              us_copy = 0;
    std::atomic<int64_t> us_io{0};
    std::atomic<int64_t> us_io_max{0};
    std::atomic<int64_t> n_io{0};
    std::string stats() const {
        return string_format("pipe wait %.1f, copy %.1f, file I/O %lld x in %.1f (longest %.1f) ms", us_wait/1000.0, us_copy/1000.0,
                             (long long) n_io.load(), us_io.load()/1000.0, us_io_max.load()/1000.0);
    }

    // [TAG_FN_L20_CONVFIX_LAZY] the logical bytes so far (the file offset of the next byte, from base)
    uint64_t logical() const { return n_logical; }

    // writing: zeros up to the next sector boundary (the next put starts at an offset a lone read can use)
    bool align() {
        static const uint8_t zeros[KC_ALIGN] = {};
        const size_t n = (size_t) (kc_round_up((size_t) n_logical) - n_logical);
        if (n == 0) {
            return !failed();
        }
        const bool h = hashing;
        hashing = false;
        const bool ok = put(zeros, n);
        hashing = h;
        return ok;
    }

    // reading: drop the bytes up to the logical offset off (padding)
    bool skip_to(uint64_t off) {
        uint8_t scratch[KC_ALIGN];
        while (n_logical < off) {
            const size_t k = (size_t) std::min<uint64_t>(sizeof(scratch), off - n_logical);
            if (!get(scratch, k)) {
                return false;
            }
        }
        return n_logical == off;
    }

    bool put(const void * src, size_t n) {
        if (hashing) {
            hash = kc_hash(hash, src, n);
        }
        const uint8_t * p = (const uint8_t *) src;
        while (n > 0) {
            if (cur < 0 && !take(q_free)) {
                return false;
            }
            const size_t k = std::min(n, KC_BLOCK - cur_off);
            const auto t0 = trace && k >= 65536 ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
            std::memcpy(mem + (size_t) cur*KC_BLOCK + cur_off, p, k);
            if (trace && k >= 65536) {
                us_copy += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
            }
            cur_off += k; p += k; n -= k; n_logical += k;
            if (cur_off == KC_BLOCK) {
                give(q_full, KC_BLOCK);
            }
        }
        return !failed();
    }

    // write the last block (padded) and wait for the thread; the logical bytes written
    bool finish(uint64_t & n_out) {
        if (cur >= 0 && cur_off > 0) {
            const size_t n = kc_round_up(cur_off);
            std::memset(mem + (size_t) cur*KC_BLOCK + cur_off, 0, n - cur_off);
            give(q_full, n);
        }
        {
            std::lock_guard<std::mutex> lk(mu);
            done = true;
        }
        cv.notify_all();
        if (th.joinable()) {
            th.join();
        }
        n_out = n_logical;
        return !err;
    }

    bool get(void * dst, size_t n) {
        uint8_t * p = (uint8_t *) dst;
        while (n > 0) {
            if (cur < 0 && !take(q_full)) {
                return false;
            }
            const size_t k = std::min(n, len[cur] - cur_off);
            const auto t0 = trace && k >= 65536 ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
            std::memcpy(p, mem + (size_t) cur*KC_BLOCK + cur_off, k);
            if (trace && k >= 65536) {
                us_copy += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
            }
            cur_off += k; p += k; n -= k; n_logical += k;
            if (cur_off == len[cur]) {
                give(q_free, 0);
            }
        }
        return true;
    }

private:
    bool failed() {
        std::lock_guard<std::mutex> lk(mu);
        return err;
    }

    // the caller waits for a block of q
    bool take(std::deque<int> & q) {
        std::unique_lock<std::mutex> lk(mu);
        if (trace && q.empty() && !err) {
            const auto t0 = std::chrono::steady_clock::now();
            cv.wait(lk, [&]() { return !q.empty() || err; });
            us_wait += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
        }
        cv.wait(lk, [&]() { return !q.empty() || err; });
        if (q.empty()) {
            return false;
        }
        cur     = q.front();
        cur_off = 0;
        q.pop_front();
        return true;
    }

    void io_note(std::chrono::steady_clock::time_point t0) {
        const int64_t us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
        us_io += us;
        n_io++;
        if (us > us_io_max.load()) {
            us_io_max = us;
        }
    }

    void give(std::deque<int> & q, size_t n) {
        {
            std::lock_guard<std::mutex> lk(mu);
            if (&q == &q_full) {
                len[cur] = n;
            }
            q.push_back(cur);
        }
        cv.notify_all();
        cur     = -1;
        cur_off = 0;
    }

    void run_write() {
        uint64_t off = base;
        while (true) {
            int b = -1;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&]() { return !q_full.empty() || done || stop; });
                if (q_full.empty() || stop) {
                    return;
                }
                b = q_full.front();
                q_full.pop_front();
            }
            const auto t0 = std::chrono::steady_clock::now();
            const bool ok = file.pwrite(mem + (size_t) b*KC_BLOCK, len[b], off);
            if (trace) {
                io_note(t0);
            }
            off += len[b];
            {
                std::lock_guard<std::mutex> lk(mu);
                if (!ok) {
                    err = true;
                }
                q_free.push_back(b);
            }
            cv.notify_all();
            if (!ok) {
                return;
            }
        }
    }

    void run_read() {
        uint64_t off = 0;
        while (off < n_read) {
            int b = -1;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&]() { return !q_free.empty() || stop; });
                if (stop) {
                    return;
                }
                b = q_free.front();
                q_free.pop_front();
            }
            const size_t n  = (size_t) std::min<uint64_t>(KC_BLOCK, n_read - off);
            const auto   t0 = std::chrono::steady_clock::now();
            const bool   ok = file.pread(mem + (size_t) b*KC_BLOCK, n, base + off);
            if (trace) {
                io_note(t0);
            }
            off += n;
            {
                std::lock_guard<std::mutex> lk(mu);
                if (!ok) {
                    err = true;
                } else {
                    len[b] = n;
                    q_full.push_back(b);
                }
            }
            cv.notify_all();
            if (!ok) {
                return;
            }
        }
        // past the end: get() asking for more bytes than were written fails instead of waiting
        {
            std::lock_guard<std::mutex> lk(mu);
            err = true;
        }
        cv.notify_all();
    }

    server_keepctx_file & file;
    const bool            writing;
    const uint64_t        n_read;
    const uint64_t        base;

    uint8_t * mem = nullptr;
    bool      own_mem = true; // [TAG_FN_L20_CONVFIX_IO]
    size_t    len[KC_NBUF] = {};

    std::mutex              mu;
    std::condition_variable cv;
    std::deque<int>         q_free;
    std::deque<int>         q_full;
    bool                    err  = false;
    bool                    done = false;
    bool                    stop = false;
    std::thread             th;

    // caller side
    int      cur       = -1;
    size_t   cur_off   = 0;
    uint64_t n_logical = 0;
};

static bool kc_write_cb(const void * src, size_t size, void * ud) {
    return ((kc_pipe *) ud)->put(src, size);
}

static bool kc_read_cb(void * dst, size_t size, void * ud) {
    return ((kc_pipe *) ud)->get(dst, size);
}

//
// server_keepctx
//

server_keepctx::server_keepctx(const server_keepctx_params & params) : params_(params) {
    std::error_code ec;
#if !defined(_WIN32)
    mkdir(params_.dir.c_str(), 0700); // a store folder made here (its parent exists) is accessible to this user only
#endif
    std::filesystem::create_directories(params_.dir, ec);
    if (!std::filesystem::is_directory(params_.dir, ec)) {
        SRV_ERR("[TAG_FN_L18_KEEPCTX] the store folder '%s' cannot be created\n", params_.dir.c_str());
        return;
    }
#if !defined(_WIN32)
    // the file names are predictable: another user must not be able to put files or links in the folder
    struct stat st = {};
    if (lstat(params_.dir.c_str(), &st) != 0 || S_ISLNK(st.st_mode) || st.st_uid != geteuid()) {
        SRV_ERR("[TAG_FN_L18_KEEPCTX] the store folder '%s' is a symbolic link or belongs to another user\n", params_.dir.c_str());
        return;
    }
#endif
    // files of a process that ended without closing them (power loss): a file still open elsewhere cannot be removed
    for (const auto & de : std::filesystem::directory_iterator(params_.dir, ec)) {
        const std::string name = de.path().filename().string();
        if (name.rfind("keepctx-", 0) == 0 && de.path().extension() == ".kc") {
            std::error_code ec_rm;
            std::filesystem::remove(de.path(), ec_rm);
        }
    }
    ok_ = true;

    // [TAG_FN_L20_CONVFIX_IO]
    if (params_.io_reuse) {
        io_mem_ = kc_alloc_blocks();
    }

    if (params_.verify) {
        std::string err;
        const bool pass = selftest(err);
        SRV_INF("[TAG_FN_L18_KEEPCTX] store I/O self-test %s%s\n", pass ? "PASS" : "FAIL: ", err.c_str());
        ok_ = pass;
    }
}

server_keepctx::~server_keepctx() {
    // [TAG_FN_L19_CONVSTORE_SPILL] the queued writes finish first
    {
        std::lock_guard<std::mutex> lk(wr_mu_);
        wr_stop_ = true;
    }
    wr_cv_.notify_all();
    if (wr_th_.joinable()) {
        wr_th_.join();
    }
    if (io_mem_) {
        kc_free_blocks(io_mem_); // [TAG_FN_L20_CONVFIX_IO]
    }
}

void server_keepctx::wr_run() {
    while (true) {
        std::unique_ptr<spill_job> job;
        {
            std::unique_lock<std::mutex> lk(wr_mu_);
            wr_cv_.wait(lk, [&]() { return wr_stop_ || !wr_q_.empty(); });
            if (wr_q_.empty()) {
                return;
            }
            job = std::move(wr_q_.front());
            wr_q_.pop_front();
        }
        wr_job(*job);
        // the target bytes' buffer serves the next checkpoint (take_spare); one at most, it is RAM beside the model
        std::lock_guard<std::mutex> lk(wr_mu_);
        if (spare_.empty() && !job->tgt.empty()) {
            spare_.push_back(std::move(job->tgt));
        }
    }
}

void server_keepctx::wr_job(spill_job & job) {
    const auto t0 = std::chrono::steady_clock::now();
    uint64_t n_written = 0;
    uint64_t hash      = 0;
    bool     ok        = true;
    {
        kc_pipe pipe(*job.file, true, 0, 0);
        pipe.hashing = job.verify;
        ok = pipe.put(job.tgt.data(), job.tgt.size()) && pipe.put(job.dft.data(), job.dft.size()) &&
             pipe.put(job.spec.data(), job.spec.size());
        ok = pipe.finish(n_written) && ok;
        hash = pipe.hash;
    }
    ok = ok && n_written == job.tgt.size() + job.dft.size() + job.spec.size();
    {
        std::lock_guard<std::mutex> lk(job.st->mu);
        job.st->done      = ok ? 1 : -1;
        job.st->hash_data = hash;
        job.st->ms        = kc_ms_since(t0);
    }
    job.st->cv.notify_all();
}

// write a pattern in pieces of awkward sizes through the pipe, read it back in other pieces, compare
bool server_keepctx::selftest(std::string & err) {
    const auto path = std::filesystem::path(params_.dir) / string_format("keepctx-%lu-selftest.kc", kc_pid());
    auto file = server_keepctx_file::create(path, err);
    if (!file) {
        return false;
    }
    const size_t wsz[] = { 1, 4095, 4097, KC_BLOCK + 3, 7, 2*KC_BLOCK - 5, 33, 12345 };
    const size_t rsz[] = { 5, 4096, KC_BLOCK - 1, 17, 3*KC_BLOCK, 1 };
    size_t total = 0;
    for (size_t n : wsz) {
        total += n;
    }
    std::vector<uint8_t> ref(total);
    for (size_t i = 0; i < total; ++i) {
        ref[i] = (uint8_t) ((i * 2654435761u) >> 13);
    }
    uint64_t n_written = 0;
    {
        kc_pipe pipe(*file, true, 0);
        size_t off = 0;
        for (size_t n : wsz) {
            if (!pipe.put(ref.data() + off, n)) {
                err = "write";
                return false;
            }
            off += n;
        }
        if (!pipe.finish(n_written) || n_written != total) {
            err = "finish";
            return false;
        }
    }
    std::vector<uint8_t> got(total);
    {
        kc_pipe pipe(*file, false, total);
        size_t off = 0;
        for (size_t k = 0; off < total; ++k) {
            const size_t n = std::min(rsz[k % (sizeof(rsz)/sizeof(rsz[0]))], total - off);
            if (!pipe.get(got.data() + off, n)) {
                err = "read";
                return false;
            }
            off += n;
        }
    }
    if (got != ref) {
        err = "data differs";
        return false;
    }

    // [TAG_FN_L19_CONVSTORE_SPILL] a checkpoint through the writer thread and back, then written at once and back
    for (int async = 1; async >= 0; --async) {
        common_prompt_checkpoint c;
        c.n_tokens = 1234;
        c.pos_min  = 1233;
        c.pos_max  = 1233;
        c.data_tgt .resize(KC_BLOCK + 12345);
        c.data_dft .resize(4097);
        c.data_spec.resize(17);
        for (size_t i = 0; i < c.data_tgt.size(); ++i) {
            c.data_tgt[i] = (uint8_t) ((i * 2654435761u) >> 11);
        }
        for (size_t i = 0; i < c.data_dft.size(); ++i) {
            c.data_dft[i] = (uint8_t) (i * 7 + async);
        }
        for (size_t i = 0; i < c.data_spec.size(); ++i) {
            c.data_spec[i] = (uint8_t) (255 - i);
        }
        const auto rt = c.data_tgt;
        const auto rd = c.data_dft;
        const auto rs = c.data_spec;
        server_keepctx_spill sp;
        std::string info;
        if (!spill_write(sp, c, 42, async != 0, info) || sp.list.size() != 1) {
            err = "spill write: " + info;
            return false;
        }
        common_prompt_checkpoint r;
        if (!spill_read(sp.list[0], r, info)) {
            err = "spill read: " + info;
            return false;
        }
        if (r.data_tgt != rt || r.data_dft != rd || r.data_spec != rs || r.n_tokens != 1234 || r.pos_max != 1233 ||
            (!async && c.data_tgt != rt)) {
            err = async ? "spilled checkpoint differs (writer thread)" : "spilled checkpoint differs (at once)";
            return false;
        }
    }
    return true;
}

uint64_t server_keepctx::n_bytes() const {
    uint64_t n = 0;
    for (const auto & e : entries) {
        n += e.n_bytes_all();
    }
    return n;
}

uint64_t server_keepctx_entry::n_bytes_all() const {
    uint64_t n = n_bytes;
    for (const auto & s : spilled) {
        n += s.bytes();
    }
    return n;
}

// [TAG_FN_L19_CONVSTORE_SPILL] a running hash of the tokens (media chunks by their id and size). at: positions in
// ascending order; out[k] = the hash of tokens [0, at[k]), 0 when it cannot be computed (past the end, a position inside
// a media chunk)
// The tokens are read one by one (operator[]): get_tokens() asserts on a server with a multimodal projector (vision on,
// the owner's command), FAULT 2026-10-12 19:10:14
static void kc_prefix_hashes(const server_tokens & tokens, const std::vector<int64_t> & at, std::vector<uint64_t> & out) {
    const size_t n_tok = tokens.size();
    out.assign(at.size(), 0);
    uint64_t h = KC_HASH_SEED;
    size_t   i = 0;
    size_t   k = 0;
    try {
        while (k < at.size()) {
            if (at[k] < 0 || (size_t) at[k] > n_tok) {
                k++; // stays 0
                continue;
            }
            if ((size_t) at[k] == i) {
                out[k++] = h == 0 ? 1 : h;
                continue;
            }
            if ((size_t) at[k] < i) {
                k++; // inside a media chunk
                continue;
            }
            if (tokens[i] != LLAMA_TOKEN_NULL) {
                h = (h ^ (uint32_t) tokens[i]) * 0x100000001b3ull;
                ++i;
                continue;
            }
            const auto & chunk = tokens.find_chunk(i);
            if (!chunk) {
                return;
            }
            const char * id = mtmd_input_chunk_get_id(chunk.get());
            const size_t nt = mtmd_input_chunk_get_n_tokens(chunk.get());
            if (nt == 0) {
                return;
            }
            h = kc_hash(h, id ? id : "", id ? strlen(id) : 0);
            h = (h ^ (uint64_t) nt) * 0x100000001b3ull;
            i += nt;
        }
    } catch (const std::exception &) {
        for (; k < at.size(); ++k) {
            out[k] = 0;
        }
    }
}

uint64_t keepctx_prefix_hash(const server_tokens & tokens, size_t n) {
    std::vector<uint64_t> out;
    kc_prefix_hashes(tokens, { (int64_t) n }, out);
    return out[0];
}

int keepctx_spill_validate(const server_tokens & tokens, std::vector<server_keepctx_spilled> & list) {
    if (list.empty()) {
        return 0;
    }
    std::vector<int64_t> at;
    for (const auto & s : list) {
        at.push_back(s.n_tokens);
    }
    std::vector<uint64_t> h;
    kc_prefix_hashes(tokens, at, h);
    int n_drop = 0;
    std::vector<server_keepctx_spilled> keep;
    for (size_t k = 0; k < list.size(); ++k) {
        if (h[k] != 0 && h[k] == list[k].hash) {
            keep.push_back(std::move(list[k]));
        } else {
            n_drop++;
        }
    }
    list = std::move(keep);
    return n_drop;
}

void server_keepctx_spill::keep_first(int64_t n_keep) {
    while (!list.empty() && list.back().n_tokens > n_keep) {
        list.pop_back();
    }
}

bool server_keepctx::spill_write(server_keepctx_spill & spill, common_prompt_checkpoint & c, uint64_t prefix_hash,
                                 bool async, std::string & info) {
    const auto t0 = std::chrono::steady_clock::now();
    static std::atomic<uint64_t> n_spill_files{0};
    const auto path = std::filesystem::path(params_.dir) /
            string_format("keepctx-%lu-spill%llu.kc", kc_pid(), (unsigned long long) ++n_spill_files);
    std::string err;
    auto file = server_keepctx_file::create(path, err);
    if (!file) {
        info = "cannot create '" + path.string() + "': " + err;
        return false;
    }
    server_keepctx_spilled s;
    s.n_tokens  = c.n_tokens;
    s.id_task   = c.id_task;
    s.pos_min   = c.pos_min;
    s.pos_max   = c.pos_max;
    s.size_tgt  = c.data_tgt.size();
    s.size_dft  = c.data_dft.size();
    s.size_spec = c.data_spec.size();
    s.off       = 0;
    s.hash      = prefix_hash;
    s.file      = file;
    s.st        = std::make_shared<server_keepctx_spill_state>();
    const uint64_t nb = s.bytes();
    auto job = std::make_unique<spill_job>();
    job->file   = file;
    job->st     = s.st;
    job->tgt    = std::move(c.data_tgt);
    job->dft    = std::move(c.data_dft);
    job->spec   = std::move(c.data_spec);
    job->verify = params_.verify;
    bool queued = false;
    if (async) {
        std::lock_guard<std::mutex> lk(wr_mu_);
        if (wr_q_.size() < 4) { // more pending: the disk is behind, write this one here
            if (!wr_th_.joinable()) {
                wr_th_ = std::thread([this]() { wr_run(); });
            }
            wr_q_.push_back(std::move(job));
            queued = true;
        }
    }
    if (queued) {
        wr_cv_.notify_all();
    } else {
        wr_job(*job);
        c.data_tgt  = std::move(job->tgt);
        c.data_dft  = std::move(job->dft);
        c.data_spec = std::move(job->spec);
        if (s.st->done < 0) {
            info = string_format("writing a checkpoint at n_tokens = %lld failed", (long long) c.n_tokens);
            return false;
        }
    }
    spill.n_written++;
    spill.bytes_written += s.bytes();
    auto pos = std::upper_bound(spill.list.begin(), spill.list.end(), s.n_tokens,
                                [](int64_t n, const server_keepctx_spilled & x) { return n < x.n_tokens; });
    spill.list.insert(pos, std::move(s));
    const double ms = kc_ms_since(t0);
    info = string_format("%.1f MiB %s in %.1f ms", nb/1048576.0, queued ? "queued" : "written", ms);
    return true;
}

void server_keepctx::take_spare(std::vector<uint8_t> & buf) {
    std::lock_guard<std::mutex> lk(wr_mu_);
    if (!spare_.empty()) {
        buf = std::move(spare_.back());
        spare_.pop_back();
    }
}

bool server_keepctx::spill_read(const server_keepctx_spilled & s, common_prompt_checkpoint & c, std::string & info) {
    if (!s.file || !s.st) {
        info = "no file";
        return false;
    }
    uint64_t hash_data = 0;
    {
        std::unique_lock<std::mutex> lk(s.st->mu);
        s.st->cv.wait(lk, [&]() { return s.st->done != 0; });
        if (s.st->done < 0) {
            info = "its write failed";
            return false;
        }
        hash_data = s.st->hash_data;
    }
    c.n_tokens = s.n_tokens;
    c.id_task  = s.id_task;
    c.pos_min  = s.pos_min;
    c.pos_max  = s.pos_max;
    c.data_tgt .resize(s.size_tgt);
    c.data_dft .resize(s.size_dft);
    c.data_spec.resize(s.size_spec);
    kc_pipe pipe(*s.file, false, s.bytes(), s.off, io_mem_);
    if (!(pipe.get(c.data_tgt.data(), c.data_tgt.size()) && pipe.get(c.data_dft.data(), c.data_dft.size()) &&
          pipe.get(c.data_spec.data(), c.data_spec.size()))) {
        info = "read failed";
        return false;
    }
    if (params_.verify) {
        uint64_t h = KC_HASH_SEED;
        h = kc_hash(h, c.data_tgt.data(), c.data_tgt.size());
        h = kc_hash(h, c.data_dft.data(), c.data_dft.size());
        h = kc_hash(h, c.data_spec.data(), c.data_spec.size());
        info = h == hash_data ? "verify: identical" : "verify: DIFFERS";
    }
    return true;
}

// [TAG_FN_L24_KEEPVAL] steady-clock seconds
static double kc_now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// [TAG_FN_L19_CONVSTORE_EVICT]
bool server_keepctx::evict_one(const server_keepctx_entry * keep, bool budget, uint64_t * freed) {
    // [TAG_FN_L24_KEEPVAL]
    if (params_.val) {
        const double now = kc_now_s();
        std::vector<convstore_val_entry> es;
        std::vector<std::list<server_keepctx_entry>::iterator> its;
        for (auto it = entries.begin(); it != entries.end(); ++it) {
            convstore_val_entry v;
            v.n_tokens = it->prompt.n_tokens();
            v.hits     = it->conv.hits;
            v.age_s    = now - it->t_use;
            v.keep     = &*it == keep;
            es.push_back(v);
            its.push_back(it);
        }
        const int vi = convstore_val_victim(es, params_.fixed_tokens, params_.val_half_s, params_.val_active_s, budget);
        if (vi < 0) {
            return false;
        }
        const int prot = convstore_val_protect(es, params_.val_active_s);
        LOG_INF("[TAG_FN_L24_KEEPVAL] evicted a stored conversation of %lld tokens (value %.0f, %.0f s since used, %lld restores) "
                "for the %s%s\n", (long long) es[vi].n_tokens,
                convstore_val(es[vi].n_tokens, es[vi].hits, es[vi].age_s, params_.fixed_tokens, params_.val_half_s), es[vi].age_s,
                (long long) es[vi].hits, budget ? "disk budget" : "count limit",
                vi == prot ? " (the protected one: nothing else left)" :
                (prot >= 0 ? string_format(", kept the biggest active one (%lld tokens)", (long long) es[prot].n_tokens).c_str() : ""));
        if (freed) {
            *freed = its[vi]->n_bytes_all();
        }
        n_evicted_gdsf++;
        entries.erase(its[vi]);
        return true;
    }
    auto victim = entries.end();
    for (auto it = entries.begin(); it != entries.end(); ++it) {
        if (&*it == keep) {
            continue;
        }
        if (!params_.gdsf) {
            victim = it; // least recently stored first
            break;
        }
        if (victim == entries.end() || it->value < victim->value) {
            victim = it;
        }
    }
    if (victim == entries.end()) {
        return false;
    }
    if (params_.gdsf) {
        gdsf_L_ = std::max(gdsf_L_, victim->value);
        n_evicted_gdsf++;
    }
    if (freed) {
        *freed = victim->n_bytes_all();
    }
    entries.erase(victim);
    return true;
}

int server_keepctx::drop_prefixes_of(const server_tokens & tokens, const server_keepctx_entry * keep, int32_t tol) {
    int n = 0;
    for (auto it = entries.begin(); it != entries.end();) {
        const size_t n_e = it->prompt.tokens.size();
        // [TAG_FN_L19_CONVSTORE] superseded: shares all but its last tol tokens
        const bool sup = tol > 0 && n_e > (size_t) tol && it->prompt.tokens.get_common_prefix(tokens) + (size_t) tol > n_e;
        if (&*it != keep && (sup || (n_e <= tokens.size() && it->prompt.tokens.get_common_prefix(tokens) == n_e))) {
            it = entries.erase(it);
            n++;
        } else {
            ++it;
        }
    }
    return n;
}

bool server_keepctx::save(const server_prompt & prompt, llama_context * ctx_tgt, llama_context * ctx_dft,
                          common_speculative * spec, llama_seq_id seq_id, const server_keepctx_entry * keep,
                          std::string & info, const server_keepctx_spill * spill, const server_keepctx_conv * conv,
                          int32_t supersede_tol) {
    const auto t0 = std::chrono::steady_clock::now();
    const kc_mark mk_s = params_.trace ? kc_mark::now() : kc_mark(); // [TAG_FN_V0254_STORE]

    server_keepctx_entry e;
    e.size_tgt = llama_state_seq_get_size_ext(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE);
    e.size_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;
    if (e.size_tgt == 0) {
        info = "the target state is empty";
        return false;
    }

    uint64_t need = e.size_tgt + e.size_dft;
    for (const auto & c : prompt.checkpoints) {
        if (params_.lazy) {
            need = kc_round_up((size_t) need); // [TAG_FN_L20_CONVFIX_LAZY] each checkpoint at a sector offset
        }
        need += c.data_tgt.size() + c.data_dft.size() + c.data_spec.size();
    }
    // [TAG_FN_L19_CONVSTORE_SPILL] the conversation's checkpoints on disk go with it (the same file, nothing written),
    // except the ones the ring holds as well
    uint64_t need_spilled = 0;
    if (spill) {
        for (const auto & sp : spill->list) {
            if (sp.n_tokens > prompt.n_tokens()) {
                continue;
            }
            bool in_ring = false;
            for (const auto & c : prompt.checkpoints) {
                in_ring = in_ring || c.n_tokens == sp.n_tokens;
            }
            if (!in_ring) {
                e.spilled.push_back(sp);
            }
        }
        keepctx_spill_validate(prompt.tokens, e.spilled);
        for (const auto & sp : e.spilled) {
            need_spilled += sp.bytes();
        }
    }
    if (conv) {
        e.conv = *conv;
    }
    if (need + need_spilled > params_.budget) {
        e.spilled.clear(); // the state and the ring's checkpoints first
        need_spilled = 0;
    }
    if (need > params_.budget) {
        info = string_format("%.1f MiB is more than the store budget of %.1f MiB", need/1048576.0, params_.budget/1048576.0);
        return false;
    }

    // room: older states of this conversation, then the oldest entries (or [TAG_FN_L19_CONVSTORE_EVICT] the lowest
    // value per byte), never `keep`
    int n_evicted = drop_prefixes_of(prompt.tokens, keep, supersede_tol);
    if (params_.val) {
        // [TAG_FN_L24_KEEPVAL] the disk first (budget and free space, the protected entry last), then the count (never the
        // protected entry: the count may then pass n_max by the protected and the keep entry)
        // the margin only drives evictions; the store fails on the disk only when the bytes do not fit at all
        int64_t disk_short = 0;
        int64_t disk_full  = 0;
        if (params_.disk_margin > 0) {
            std::error_code ec;
            const auto sp = std::filesystem::space(params_.dir, ec);
            if (!ec) {
                disk_full  = (int64_t) (need + need_spilled) - (int64_t) sp.available;
                disk_short = disk_full + (int64_t) params_.disk_margin;
            }
        }
        while (n_bytes() + need + need_spilled > params_.budget || disk_short > 0) {
            uint64_t freed = 0;
            if (!evict_one(keep, true, &freed)) {
                break;
            }
            disk_short -= (int64_t) freed;
            disk_full  -= (int64_t) freed;
            n_evicted++;
        }
        while ((int32_t) entries.size() >= params_.n_max && evict_one(keep, false)) {
            n_evicted++;
        }
        if (n_bytes() + need + need_spilled > params_.budget || disk_full > 0 || (int32_t) entries.size() >= params_.n_max + 2) {
            info = string_format("no room beside the conversation the next prompt restores (%.1f MiB needed%s)", need/1048576.0,
                                 disk_full > 0 ? ", the disk is full" : "");
            return false;
        }
    } else {
        while ((n_bytes() + need + need_spilled > params_.budget || (int32_t) entries.size() >= params_.n_max) && evict_one(keep)) {
            n_evicted++;
        }
        if (n_bytes() + need + need_spilled > params_.budget || (int32_t) entries.size() >= params_.n_max) {
            info = string_format("no room beside the conversation the next prompt restores (%.1f MiB needed)", need/1048576.0);
            return false;
        }
    }

    e.id = ++n_next_;
    const auto path = std::filesystem::path(params_.dir) /
            string_format("keepctx-%lu-%llu.kc", kc_pid(), (unsigned long long) e.id);
    std::string err;
    e.file = server_keepctx_file::create(path, err);
    if (!e.file) {
        info = "cannot create '" + path.string() + "': " + err;
        return false;
    }

    bool ok = true;
    // [TAG_FN_L20_CONVFIX_TRACE] the parts: before the writes (eviction, file), target state, draft state, checkpoints,
    // the last block
    double tr_pre = 0.0, tr_tgt = 0.0, tr_dft = 0.0, tr_ckpt = 0.0, tr_fin = 0.0;
    auto   tr_t   = std::chrono::steady_clock::now();
    auto   tr_lap = [&](double & acc) {
        const auto now = std::chrono::steady_clock::now();
        acc += std::chrono::duration<double, std::milli>(now - tr_t).count();
        tr_t = now;
    };
    tr_pre = kc_ms_since(t0);
    // [TAG_FN_V0254_STORE] trace: CPU time and page faults of each part, the checkpoint bytes outside the working set,
    // the residency service's touches meanwhile
    kc_mark mk[5];
    mk[0] = params_.trace ? kc_mark::now() : kc_mark();
    const kc_mark mk_p = mk[0];
    uint64_t ck_bytes = 0, ck_out = 0;
    double   ck_q_ms  = 0.0;
    const uint64_t res0 = params_.trace ? llama_fn_res_touched_ext() : 0;
    if (params_.trace) {
        const auto q0 = std::chrono::steady_clock::now();
        for (const auto & c : prompt.checkpoints) {
            ck_bytes += c.data_tgt.size() + c.data_dft.size() + c.data_spec.size();
            ck_out   += kc_nonresident(c.data_tgt.data(), c.data_tgt.size()) + kc_nonresident(c.data_dft.data(), c.data_dft.size()) +
                        kc_nonresident(c.data_spec.data(), c.data_spec.size());
        }
        ck_q_ms = kc_ms_since(q0);
    }
    if (params_.trace) {
        mk[0] = kc_mark::now();
        tr_t  = mk[0].t;
    }
    std::string pipe_st;
    {
        kc_pipe pipe(*e.file, true, 0, 0, io_mem_, params_.trace);
        pipe.hashing = params_.verify;
        ok = ok && llama_state_seq_write_stream_ext(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE, kc_write_cb, &pipe) == e.size_tgt;
        tr_lap(tr_tgt);
        mk[1] = params_.trace ? kc_mark::now() : kc_mark();
        e.hash_tgt = pipe.hash;
        pipe.hash  = KC_HASH_SEED;
        if (ok && ctx_dft) {
            ok = llama_state_seq_write_stream_ext(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE, kc_write_cb, &pipe) == e.size_dft;
        }
        tr_lap(tr_dft);
        mk[2] = params_.trace ? kc_mark::now() : kc_mark();
        e.hash_dft   = pipe.hash;
        pipe.hashing = false;
        for (const auto & c : prompt.checkpoints) {
            if (!ok) {
                break;
            }
            // [TAG_FN_L20_CONVFIX_LAZY] a sector offset, so a restore can leave it on disk and read it alone later
            if (params_.lazy) {
                ok = pipe.align();
                e.ckpt_off.push_back(pipe.logical());
                if (params_.verify) {
                    uint64_t h = KC_HASH_SEED;
                    h = kc_hash(h, c.data_tgt.data(), c.data_tgt.size());
                    h = kc_hash(h, c.data_dft.data(), c.data_dft.size());
                    h = kc_hash(h, c.data_spec.data(), c.data_spec.size());
                    e.ckpt_hash.push_back(h);
                }
            }
            ok = ok && pipe.put(c.data_tgt.data(), c.data_tgt.size()) && pipe.put(c.data_dft.data(), c.data_dft.size()) &&
                 pipe.put(c.data_spec.data(), c.data_spec.size());
            e.ckpt_sizes.push_back(c.data_tgt.size());
            e.ckpt_sizes.push_back(c.data_dft.size());
            e.ckpt_sizes.push_back(c.data_spec.size());
        }
        tr_lap(tr_ckpt);
        mk[3] = params_.trace ? kc_mark::now() : kc_mark();
        uint64_t n_written = 0;
        ok = pipe.finish(n_written) && ok;
        e.n_bytes = n_written;
        if (params_.trace) {
            pipe_st = pipe.stats();
        }
    }
    tr_lap(tr_fin);
    mk[4] = params_.trace ? kc_mark::now() : kc_mark();
    if (!ok || e.n_bytes != need) {
        info = string_format("writing '%s' failed (%.1f of %.1f MiB)", path.string().c_str(), e.n_bytes/1048576.0, need/1048576.0);
        return false;
    }

    common_speculative_get_carry(spec, seq_id, e.carry);

    // [TAG_FN_L18_KEEPCTX_HOT] the counts before the new prompt fades them
    if (params_.hot) {
        llama_moe_hot_counts_get_ext(ctx_tgt, e.hot);
    }

    e.prompt.tokens = prompt.tokens.clone();
    for (const auto & c : prompt.checkpoints) {
        common_prompt_checkpoint m;
        m.n_tokens = c.n_tokens;
        m.id_task  = c.id_task;
        m.pos_min  = c.pos_min;
        m.pos_max  = c.pos_max;
        e.prompt.checkpoints.push_back(std::move(m));
    }

    // [TAG_FN_L19_CONVSTORE_EVICT]
    e.value = convstore_gdsf(gdsf_L_, e.conv.hits, prompt.n_tokens(), e.n_bytes_all(), params_.fixed_tokens,
                             params_.budget / (uint64_t) std::max(1, params_.n_max));
    // [TAG_FN_L24_KEEPVAL]
    e.t_use = kc_now_s();
    if (params_.val) {
        e.value = convstore_val(prompt.n_tokens(), e.conv.hits, 0.0, params_.fixed_tokens, params_.val_half_s);
    }

    const double ms = kc_ms_since(t0);
    info = string_format("%d tokens, %u checkpoints, %.1f MiB (target %.1f, draft %.1f) in %.0f ms = %.2f GB/s, %d older evicted, "
                         "store %zu conversations / %.1f MiB", prompt.n_tokens(), (unsigned) prompt.checkpoints.size(),
                         need/1048576.0, e.size_tgt/1048576.0, e.size_dft/1048576.0, ms, need/(ms*1e6), n_evicted,
                         entries.size() + 1, (n_bytes() + e.n_bytes_all())/1048576.0);
    if (!e.spilled.empty() || params_.gdsf || params_.val) {
        info += string_format(" [TAG_FN_L19_CONVSTORE] %zu checkpoints on disk (%.1f MiB), restores %lld, value %.0f, system block %d",
                              e.spilled.size(), need_spilled/1048576.0, (long long) e.conv.hits, e.value, (int) e.conv.sys_end);
    }
    if (params_.trace) {
        info += string_format(" [TAG_FN_L20_CONVFIX_TRACE] pre %.1f, target %.1f, draft %.1f, checkpoints %.1f, finish %.1f ms",
                              tr_pre, tr_tgt, tr_dft, tr_ckpt, tr_fin);
        // [TAG_FN_V0254_STORE]
        info += string_format(" [TAG_FN_V0254_STORE] wall/user/kernel, faults: pre %s, target %s, draft %s, checkpoints %s, finish %s; "
                              "checkpoint bytes out of the working set %.1f of %.1f MiB (%.1f ms to look); %s; residency service "
                              "touched %.1f MiB", kc_part(mk_s, mk_p).c_str(), kc_part(mk[0], mk[1]).c_str(), kc_part(mk[1], mk[2]).c_str(),
                              kc_part(mk[2], mk[3]).c_str(), kc_part(mk[3], mk[4]).c_str(), ck_out/1048576.0, ck_bytes/1048576.0,
                              ck_q_ms, pipe_st.c_str(), (llama_fn_res_touched_ext() - res0)/1048576.0);
    }
    entries.push_back(std::move(e));
    return true;
}

bool server_keepctx::load(std::list<server_keepctx_entry>::iterator it, server_prompt & prompt, llama_context * ctx_tgt,
                          llama_context * ctx_dft, common_speculative * spec, llama_seq_id seq_id, bool keep, std::string & info,
                          server_keepctx_spill * spill, server_keepctx_conv * conv,
                          std::list<common_prompt_checkpoint> * reuse) {
    const auto t0 = std::chrono::steady_clock::now();

    server_keepctx_entry e_own;
    if (!keep) {
        e_own = std::move(*it);
        entries.erase(it);
    }
    server_keepctx_entry & e = keep ? *it : e_own;

    // a kept entry that failed to restore is not offered again
    auto fail = [&](const char * why) {
        info = why;
        if (keep) {
            entries.erase(it);
        }
        return false;
    };

    if ((e.size_dft > 0) != (ctx_dft != nullptr)) {
        return fail("the draft context does not match the stored state");
    }

    std::list<common_prompt_checkpoint> ckpts;
    bool ok = true;
    // [TAG_FN_L20_CONVFIX_TRACE]
    double tr_tgt = 0.0, tr_dft = 0.0, tr_alloc = 0.0, tr_ckpt = 0.0;
    auto   tr_t   = std::chrono::steady_clock::now();
    auto   tr_lap = [&](double & acc) {
        const auto now = std::chrono::steady_clock::now();
        acc += std::chrono::duration<double, std::milli>(now - tr_t).count();
        tr_t = now;
    };
    int n_reused = 0;

    // [TAG_FN_L20_CONVFIX_LAZY] the checkpoints stay in the file: each one becomes a checkpoint on disk of the slot (its
    // offset, sizes and the hash of the tokens in front of it), unless one cannot be described that way
    std::vector<server_keepctx_spilled> lazy_refs;
    bool lazy = params_.lazy && spill != nullptr && !e.prompt.checkpoints.empty() &&
                e.ckpt_off.size() == e.prompt.checkpoints.size() && e.ckpt_sizes.size() == 3*e.prompt.checkpoints.size();
    if (lazy) {
        std::vector<int64_t> at;
        for (const auto & m : e.prompt.checkpoints) {
            lazy = lazy && (at.empty() || m.n_tokens > at.back());
            at.push_back(m.n_tokens);
        }
        std::vector<uint64_t> h;
        if (lazy) {
            kc_prefix_hashes(e.prompt.tokens, at, h);
        }
        size_t i = 0;
        for (const auto & m : e.prompt.checkpoints) {
            if (!lazy) {
                break;
            }
            if (h[i] == 0) {
                lazy = false;
                break;
            }
            server_keepctx_spilled s;
            s.n_tokens  = m.n_tokens;
            s.id_task   = m.id_task;
            s.pos_min   = m.pos_min;
            s.pos_max   = m.pos_max;
            s.size_tgt  = e.ckpt_sizes[3*i + 0];
            s.size_dft  = e.ckpt_sizes[3*i + 1];
            s.size_spec = e.ckpt_sizes[3*i + 2];
            s.off       = e.ckpt_off[i];
            s.hash      = h[i];
            s.file      = e.file;
            s.st        = std::make_shared<server_keepctx_spill_state>();
            s.st->done      = 1;
            s.st->hash_data = i < e.ckpt_hash.size() ? e.ckpt_hash[i] : 0;
            lazy_refs.push_back(std::move(s));
            i++;
        }
        if (!lazy) {
            lazy_refs.clear();
        }
    }
    // [TAG_FN_V0254_STORE] trace: CPU time and page faults of each part
    kc_mark mk[4];
    std::string pipe_st;
    const uint64_t res0 = params_.trace ? llama_fn_res_touched_ext() : 0;
    if (params_.trace) {
        mk[0] = kc_mark::now();
        tr_t  = mk[0].t;
    }
    {
        kc_pipe pipe(*e.file, false, lazy ? e.size_tgt + e.size_dft : e.n_bytes, 0, io_mem_, params_.trace);
        ok = ok && llama_state_seq_read_stream_ext(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE, kc_read_cb, &pipe) == e.size_tgt;
        tr_lap(tr_tgt);
        mk[1] = params_.trace ? kc_mark::now() : kc_mark();
        if (ok && ctx_dft) {
            ok = llama_state_seq_read_stream_ext(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE, kc_read_cb, &pipe) == e.size_dft;
        }
        tr_lap(tr_dft);
        mk[2] = params_.trace ? kc_mark::now() : kc_mark();
        size_t k = 0;
        size_t i = 0;
        for (const auto & m : e.prompt.checkpoints) {
            if (!ok || lazy) {
                break;
            }
            // [TAG_FN_L20_CONVFIX_LAZY] a file of the sector-offset layout read in full
            if (i < e.ckpt_off.size()) {
                ok = pipe.skip_to(e.ckpt_off[i]);
            }
            i++;
            common_prompt_checkpoint c;
            // [TAG_FN_L20_CONVFIX_IO] the buffers of an outgoing checkpoint: a resize to the same size neither allocates
            // nor zero-fills (every byte is read over below)
            if (reuse && !reuse->empty()) {
                common_prompt_checkpoint & r = reuse->front();
                c.data_tgt  = std::move(r.data_tgt);
                c.data_dft  = std::move(r.data_dft);
                c.data_spec = std::move(r.data_spec);
                reuse->pop_front();
                n_reused++;
            }
            c.n_tokens = m.n_tokens;
            c.id_task  = m.id_task;
            c.pos_min  = m.pos_min;
            c.pos_max  = m.pos_max;
            c.data_tgt .resize(e.ckpt_sizes[k + 0]);
            c.data_dft .resize(e.ckpt_sizes[k + 1]);
            c.data_spec.resize(e.ckpt_sizes[k + 2]);
            k += 3;
            tr_lap(tr_alloc);
            ok = pipe.get(c.data_tgt.data(), c.data_tgt.size()) && pipe.get(c.data_dft.data(), c.data_dft.size()) &&
                 pipe.get(c.data_spec.data(), c.data_spec.size());
            tr_lap(tr_ckpt);
            ckpts.push_back(std::move(c));
        }
        if (params_.trace) {
            pipe_st = pipe.stats();
        }
    }
    mk[3] = params_.trace ? kc_mark::now() : kc_mark();
    if (!ok) {
        return fail("reading the stored state failed");
    }
    if (!common_speculative_set_carry(spec, seq_id, e.carry)) {
        return fail("the drafter carry does not match this drafter");
    }

    prompt.tokens      = keep ? e.prompt.tokens.clone() : std::move(e.prompt.tokens);
    prompt.checkpoints = std::move(ckpts);

    // [TAG_FN_L19_CONVSTORE] the checkpoints on disk go to the slot (the files are shared), the restore counts
    if (spill) {
        spill->list = e.spilled;
        // [TAG_FN_L20_CONVFIX_LAZY] and the ring's, which stayed in the file (ascending n_tokens, one per position)
        for (auto & s : lazy_refs) {
            auto pos = std::lower_bound(spill->list.begin(), spill->list.end(), s.n_tokens,
                                        [](const server_keepctx_spilled & x, int64_t n) { return x.n_tokens < n; });
            if (pos != spill->list.end() && pos->n_tokens == s.n_tokens) {
                *pos = std::move(s);
            } else {
                spill->list.insert(pos, std::move(s));
            }
        }
    }
    if (conv) {
        *conv = e.conv;
        conv->hits++;
    }
    if (keep) {
        e.conv.hits++;
        e.value = convstore_gdsf(gdsf_L_, e.conv.hits, e.prompt.n_tokens(), e.n_bytes_all(), params_.fixed_tokens,
                                 params_.budget / (uint64_t) std::max(1, params_.n_max));
        e.t_use = kc_now_s(); // [TAG_FN_L24_KEEPVAL]
    }

    const double ms = kc_ms_since(t0);

    // [TAG_FN_L18_KEEPCTX_HOT] the next pass moves the set back toward this conversation's experts
    std::string hot;
    if (!e.hot.empty()) {
        hot = llama_moe_hot_counts_set_ext(ctx_tgt, e.hot) ? ", hot-set counts back" : ", hot-set counts NOT set";
    }

    // the states as the contexts hold them now must be the bytes that were stored
    std::string ver;
    if (params_.verify) {
        auto state_hash = [&](llama_context * ctx, uint64_t & h) {
            std::vector<uint8_t> buf(llama_state_seq_get_size_ext(ctx, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE));
            const bool got = llama_state_seq_get_data_ext(ctx, buf.data(), buf.size(), seq_id, LLAMA_STATE_SEQ_FLAGS_NONE) == buf.size();
            h = kc_hash(KC_HASH_SEED, buf.data(), buf.size());
            return got;
        };
        uint64_t ht = 0;
        uint64_t hd = 0;
        const bool vt = state_hash(ctx_tgt, ht) && ht == e.hash_tgt;
        const bool vd = !ctx_dft || (state_hash(ctx_dft, hd) && hd == e.hash_dft);
        ver = string_format(", verify: target %s, draft %s", vt ? "identical" : "DIFFERS", ctx_dft ? (vd ? "identical" : "DIFFERS") : "none");
    }
    // [TAG_FN_L20_CONVFIX_LAZY] the bytes read
    const uint64_t n_read_bytes = lazy ? e.size_tgt + e.size_dft : e.n_bytes;
    info = string_format("%d tokens, %zu checkpoints, %.1f MiB in %.0f ms = %.2f GB/s, %s, store %zu conversations / %.1f MiB%s%s",
                         prompt.n_tokens(), prompt.checkpoints.size(), n_read_bytes/1048576.0, ms, n_read_bytes/(ms*1e6),
                         keep ? "kept in the store (the prompt continues a part of it)" : "taken from the store",
                         entries.size(), n_bytes()/1048576.0, hot.c_str(), ver.c_str());
    if (!e.spilled.empty()) {
        info += string_format(" [TAG_FN_L19_CONVSTORE] %zu checkpoints on disk", e.spilled.size());
    }
    if (lazy) {
        info += string_format(" [TAG_FN_L20_CONVFIX_LAZY] %zu checkpoints left on disk", lazy_refs.size());
    }
    if (params_.trace) {
        info += string_format(" [TAG_FN_L20_CONVFIX_TRACE] target %.1f, draft %.1f, checkpoint buffers %.1f (%d reused), "
                              "checkpoints %.1f ms", tr_tgt, tr_dft, tr_alloc, n_reused, tr_ckpt);
        // [TAG_FN_V0254_STORE]
        info += string_format(" [TAG_FN_V0254_STORE] wall/user/kernel, faults: target %s, draft %s, checkpoints %s; %s; residency "
                              "service touched %.1f MiB", kc_part(mk[0], mk[1]).c_str(), kc_part(mk[1], mk[2]).c_str(),
                              kc_part(mk[2], mk[3]).c_str(), pipe_st.c_str(), (llama_fn_res_touched_ext() - res0)/1048576.0);
    }
    if (keep) {
        entries.splice(entries.end(), entries, it); // the most recently used
    }
    return true;
}

// [TAG_FN_V0254_STORE] the volume of a path ("D:\"), its physical disk number, its drive type and whether it is an SSD (no
// seek penalty); false when any of it is unknown (a mount folder, a spanned volume, a network path)
#if defined(_WIN32)
static bool kc_volume_info(const std::filesystem::path & p, std::wstring & vol, DWORD & disk, UINT & type, bool & ssd) {
    wchar_t v[MAX_PATH + 1] = {};
    if (!GetVolumePathNameW(p.wstring().c_str(), v, MAX_PATH)) {
        return false;
    }
    vol = v;
    if (vol.size() != 3 || vol[1] != L':') {
        return false;
    }
    type = GetDriveTypeW(vol.c_str());
    const std::wstring dev = L"\\\\.\\" + vol.substr(0, 2); // "\\.\D:"
    HANDLE h = CreateFileW(dev.c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        return false;
    }
    DWORD got = 0;
    STORAGE_DEVICE_NUMBER dn = {};
    const bool ok_dn = DeviceIoControl(h, IOCTL_STORAGE_GET_DEVICE_NUMBER, nullptr, 0, &dn, sizeof(dn), &got, nullptr) != 0;
    STORAGE_PROPERTY_QUERY q = {};
    q.PropertyId = StorageDeviceSeekPenaltyProperty;
    q.QueryType  = PropertyStandardQuery;
    DEVICE_SEEK_PENALTY_DESCRIPTOR sp = {};
    const bool ok_sp = DeviceIoControl(h, IOCTL_STORAGE_QUERY_PROPERTY, &q, sizeof(q), &sp, sizeof(sp), &got, nullptr) != 0;
    CloseHandle(h);
    disk = dn.DeviceNumber;
    ssd  = ok_sp && !sp.IncursSeekPenalty;
    return ok_dn && ok_sp;
}
#endif

std::string keepctx_default_dir(const std::string & model_path, uint64_t need_free, std::string & why) {
    std::error_code ec;
    const auto tmp = std::filesystem::temp_directory_path(ec);
    const auto tmp_dir = (ec ? std::filesystem::path(".") : tmp) / "llama-keepctx";
#if defined(_WIN32)
    if (ec || model_path.empty()) {
        why = "the TEMP folder (no model path)";
        return tmp_dir.string();
    }
    const auto mdir = std::filesystem::absolute(std::filesystem::u8path(model_path), ec).parent_path();
    std::wstring vm, vt;
    DWORD dm = 0, dt = 0;
    UINT  tm = 0, tt = 0;
    bool  sm = false, st = false;
    if (ec || !kc_volume_info(mdir, vm, dm, tm, sm) || !kc_volume_info(tmp, vt, dt, tt, st)) {
        why = "the TEMP folder (the drives are not known)";
        return tmp_dir.string();
    }
    if (dm == dt) {
        why = "the TEMP folder (the model is on the same disk)";
        return tmp_dir.string();
    }
    if (tm != DRIVE_FIXED || !sm) {
        why = "the TEMP folder (the model's drive is not a fixed SSD)";
        return tmp_dir.string();
    }
    const auto space = std::filesystem::space(mdir, ec);
    if (ec || space.available < need_free) {
        why = string_format("the TEMP folder (the model's drive has %.1f GiB free, the store wants %.1f GiB)",
                            ec ? 0.0 : space.available/1073741824.0, need_free/1073741824.0);
        return tmp_dir.string();
    }
    const auto dir = mdir / "llama-keepctx";
    std::filesystem::create_directories(dir, ec);
    HANDLE h = INVALID_HANDLE_VALUE;
    if (std::filesystem::is_directory(dir, ec)) {
        const auto probe = dir / string_format("keepctx-%lu-probe.kc", kc_pid());
        h = CreateFileW(probe.wstring().c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                        FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE | FILE_FLAG_NO_BUFFERING, nullptr);
    }
    if (h == INVALID_HANDLE_VALUE) {
        why = "the TEMP folder (the model's folder is not writable)";
        return tmp_dir.string();
    }
    CloseHandle(h);
    why = string_format("beside the model (disk %lu, a fixed SSD; TEMP is on disk %lu)", (unsigned long) dm, (unsigned long) dt);
    return dir.string();
#else
    (void) model_path;
    (void) need_free;
    why = "the TEMP folder";
    // one folder per user: the store refuses a folder another user owns (a shared /tmp)
    return tmp_dir.string() + string_format("-%lu", (unsigned long) geteuid());
#endif
}
