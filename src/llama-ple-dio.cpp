// [TAG_FN_PLE_DIRECT_IO] unbuffered on-demand row reads, see llama-ple-dio.h

#include "llama-ple-dio.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(_WIN32)
// FILE_STORAGE_INFO (sector sizes) needs Windows 8 headers
#    if !defined(_WIN32_WINNT) || _WIN32_WINNT < 0x0602
#        undef _WIN32_WINNT
#        define _WIN32_WINNT 0x0602
#    endif
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <cerrno>
#    include <fcntl.h>
#    include <sys/stat.h>
#    include <sys/types.h>
#    include <unistd.h>
#endif

using dio_clock = std::chrono::steady_clock;

static double dio_us(dio_clock::duration d) {
    return std::chrono::duration<double, std::micro>(d).count();
}

bool llama_ple_dio_requested() {
    const char * e = getenv("LLAMA_PLE_DIRECT_IO");
    return e != nullptr && atoi(e) != 0;
}

#if defined(_WIN32)
static std::wstring dio_wpath(const std::string & path) {
    const int wn = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    if (wn <= 0) {
        return std::wstring();
    }
    std::wstring w(wn, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, &w[0], wn);
    w.resize(wn - 1);
    return w;
}

static std::string dio_win_err(DWORD e) {
    LPSTR buf = nullptr;
    const DWORD n = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
            nullptr, e, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR) &buf, 0, nullptr);
    std::string s = n ? std::string(buf, n) : "error " + std::to_string(e);
    if (buf) {
        LocalFree(buf);
    }
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' || s.back() == '.')) {
        s.pop_back();
    }
    return s;
}
#endif

namespace {

// exact LRU of rows: slot arrays + an open-addressing row -> slot table (backward-shift deletion), no per-entry heap
struct row_lru {
    size_t  rb   = 0;
    int32_t cap  = 0;
    int32_t n    = 0;
    int32_t head = -1; // most recent
    int32_t tail = -1; // least recent

    std::vector<uint8_t> data;
    std::vector<int32_t> key;
    std::vector<int32_t> prv;
    std::vector<int32_t> nxt;

    std::vector<int32_t> tk; // table keys, -1 = empty (row ids are >= 0)
    std::vector<int32_t> tv; // table values (slots)
    uint32_t mask = 0;
    int      bits = 0;

    void init(size_t bytes, size_t row_bytes) {
        rb = row_bytes;
        const size_t c = row_bytes ? bytes / row_bytes : 0;
        cap = (int32_t) std::min<size_t>(c, (size_t) 1 << 28);
        if (cap <= 0) {
            cap = 0;
            return;
        }
        data.resize((size_t) cap * rb);
        key.resize(cap);
        prv.resize(cap);
        nxt.resize(cap);
        bits = 4;
        while (((size_t) 1 << bits) < 2 * (size_t) cap) {
            ++bits;
        }
        tk.assign((size_t) 1 << bits, -1);
        tv.assign((size_t) 1 << bits, -1);
        mask = (uint32_t) (((size_t) 1 << bits) - 1);
    }

    uint32_t home(int32_t k) const {
        return (uint32_t) (((uint64_t) (uint32_t) k * 0x9E3779B97F4A7C15ull) >> (64 - bits));
    }

    int32_t find(int32_t k) const {
        for (uint32_t i = home(k);; i = (i + 1) & mask) {
            if (tk[i] == k) {
                return tv[i];
            }
            if (tk[i] < 0) {
                return -1;
            }
        }
    }

    void tab_insert(int32_t k, int32_t v) {
        uint32_t i = home(k);
        while (tk[i] >= 0) {
            i = (i + 1) & mask;
        }
        tk[i] = k;
        tv[i] = v;
    }

    void tab_erase(int32_t k) {
        uint32_t i = home(k);
        while (tk[i] != k) {
            i = (i + 1) & mask;
        }
        for (uint32_t j = i;;) {
            j = (j + 1) & mask;
            if (tk[j] < 0) {
                break;
            }
            const uint32_t h = home(tk[j]);
            // an entry may fill the hole at i unless its home lies cyclically in (i, j]
            const bool stays = i <= j ? (i < h && h <= j) : (i < h || h <= j);
            if (!stays) {
                tk[i] = tk[j];
                tv[i] = tv[j];
                i = j;
            }
        }
        tk[i] = -1;
    }

    void unlink(int32_t s) {
        if (prv[s] >= 0) { nxt[prv[s]] = nxt[s]; } else { head = nxt[s]; }
        if (nxt[s] >= 0) { prv[nxt[s]] = prv[s]; } else { tail = prv[s]; }
    }

    void push_front(int32_t s) {
        prv[s] = -1;
        nxt[s] = head;
        if (head >= 0) {
            prv[head] = s;
        }
        head = s;
        if (tail < 0) {
            tail = s;
        }
    }

    const uint8_t * get(int32_t k) {
        if (cap == 0) {
            return nullptr;
        }
        const int32_t s = find(k);
        if (s < 0) {
            return nullptr;
        }
        if (s != head) {
            unlink(s);
            push_front(s);
        }
        return data.data() + (size_t) s * rb;
    }

    // k must not be present
    void put(int32_t k, const uint8_t * src) {
        if (cap == 0) {
            return;
        }
        int32_t s;
        if (n < cap) {
            s = n++;
        } else {
            s = tail;
            unlink(s);
            tab_erase(key[s]);
        }
        key[s] = k;
        memcpy(data.data() + (size_t) s * rb, src, rb);
        tab_insert(k, s);
        push_front(s);
    }
};

struct read_span {
    uint64_t start; // aligned file offset
    size_t   len;   // aligned length
    size_t   skip;  // row bytes start this far into the read
};

#if !defined(_WIN32)
// persistent workers for the POSIX path; the caller is worker 0
struct dio_pool {
    int n = 0; // extra threads
    std::vector<std::thread> th;
    std::mutex              m;
    std::condition_variable cv;
    std::condition_variable cv_done;
    uint64_t gen     = 0;
    int      pending = 0;
    bool     stop    = false;
    std::function<void(int)> fn;

    void start(int n_extra) {
        n = n_extra;
        for (int i = 0; i < n; ++i) {
            th.emplace_back([this, i] { loop(i + 1); });
        }
    }

    void loop(int tid) {
        uint64_t seen = 0;
        for (;;) {
            std::function<void(int)> f;
            {
                std::unique_lock<std::mutex> l(m);
                cv.wait(l, [&] { return stop || gen != seen; });
                if (stop) {
                    return;
                }
                seen = gen;
                f    = fn;
            }
            f(tid);
            {
                std::lock_guard<std::mutex> l(m);
                if (--pending == 0) {
                    cv_done.notify_one();
                }
            }
        }
    }

    void run(const std::function<void(int)> & f, bool inline_only) {
        if (n == 0 || inline_only) {
            f(0);
            return;
        }
        {
            std::lock_guard<std::mutex> l(m);
            fn      = f;
            pending = n;
            ++gen;
        }
        cv.notify_all();
        f(0);
        std::unique_lock<std::mutex> l(m);
        cv_done.wait(l, [&] { return pending == 0; });
    }

    ~dio_pool() {
        {
            std::lock_guard<std::mutex> l(m);
            stop = true;
        }
        cv.notify_all();
        for (auto & t : th) {
            t.join();
        }
    }
};
#endif

} // namespace

struct llama_ple_dio::impl {
    llama_ple_dio_params p;

    size_t   A          = 4096; // alignment of offsets, lengths and buffers
    size_t   slot_bytes = 0;    // largest aligned span of one row
    uint64_t file_size  = 0;
    bool     is_direct  = true;

    mutable std::mutex mtx;
    row_lru             lru;
    llama_ple_dio_stats st;
    bool                warned_fail = false;

    // per-call scratch
    std::vector<uint64_t> order;
    std::vector<int32_t>  urow;
    std::vector<uint32_t> upos;
    std::vector<uint8_t>  ubuf;
    std::vector<int32_t>  miss;

#if defined(_WIN32)
    struct win_slot {
        OVERLAPPED ov; // first member: the completion hands back &ov
        int32_t    u;
        size_t     skip;
        dio_clock::time_point t0;
    };

    HANDLE h    = INVALID_HANDLE_VALUE;
    HANDLE iocp = nullptr;
    uint8_t * bufs = nullptr; // queue_depth * slot_bytes, VirtualAlloc (page aligned)
    std::vector<win_slot>          slots;
    std::vector<int>               free_slots;
    std::vector<OVERLAPPED_ENTRY>  ents;
#else
    int fd = -1;
    std::vector<uint8_t *> tbufs; // one aligned buffer per thread
    dio_pool pool;
#endif

    void emit(const char * fmt, ...) const {
        char buf[1024];
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(buf, sizeof(buf), fmt, ap);
        va_end(ap);
        if (p.log) {
            p.log(buf);
        } else {
            fputs(buf, stderr);
        }
    }

    std::string line() const {
        const llama_ple_dio_stats & s = st;
        const double tok = s.tokens ? (double) s.tokens : 1.0;
        const double cal = s.calls  ? (double) s.calls  : 1.0;
        char buf[512];
        snprintf(buf, sizeof(buf),
                "%llu ubatches, %llu tokens, %.2f rows/token (%.2f distinct), LRU hits %.1f%%, %llu reads, %.2f MiB read "
                "(%.0f B/token), %.1f us/ubatch avg (max %.1f), %.1f us/read avg (%.1f us to issue, %llu done inline), "
                "%llu fallbacks",
                (unsigned long long) s.calls, (unsigned long long) s.tokens, s.rows/tok, s.unique/tok,
                s.unique ? 100.0*s.hits/s.unique : 0.0, (unsigned long long) s.reads, s.bytes/1048576.0, s.bytes/tok,
                s.call_us/cal, s.call_us_max, s.reads ? s.read_us/s.reads : 0.0, s.reads ? s.issue_us/s.reads : 0.0,
                (unsigned long long) s.inline_done, (unsigned long long) s.fallbacks);
        return buf;
    }

    read_span span_of(int32_t row) const {
        const uint64_t off   = p.offset + (uint64_t) row * p.row_bytes;
        const uint64_t start = off & ~(uint64_t) (A - 1);
        const uint64_t end   = (off + p.row_bytes + A - 1) & ~(uint64_t) (A - 1);
        return { start, (size_t) (end - start), (size_t) (off - start) };
    }

    void fail_row(int32_t u, const uint8_t * fallback, unsigned long code) {
        const size_t rb = p.row_bytes;
        if (!fallback) {
            throw std::runtime_error("PLE direct I/O: read of row " + std::to_string(urow[u]) + " failed (error " +
                                     std::to_string(code) + ") and there is no fallback");
        }
        if (!warned_fail) {
            warned_fail = true;
            emit("%s: [TAG_FN_PLE_DIRECT_IO] warning: read of row %d failed (error %lu); such rows are copied from the "
                 "mapped table\n", __func__, urow[u], code);
        }
        memcpy(ubuf.data() + (size_t) u * rb, fallback + (size_t) urow[u] * rb, rb);
        st.fallbacks++;
    }

#if defined(_WIN32)
    static std::string win_err(DWORD e) {
        return dio_win_err(e);
    }

    bool open_file(std::string & err) {
        const std::wstring wpath = dio_wpath(p.path);
        if (wpath.empty()) {
            err = "cannot convert the path to UTF-16";
            return false;
        }

        h = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
        if (h == INVALID_HANDLE_VALUE) {
            err = "CreateFileW(FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED) failed: " + win_err(GetLastError());
            return false;
        }

        LARGE_INTEGER sz;
        if (!GetFileSizeEx(h, &sz)) {
            err = "GetFileSizeEx failed: " + win_err(GetLastError());
            return false;
        }
        file_size = (uint64_t) sz.QuadPart;

        // unbuffered I/O wants sector-aligned offsets, lengths and buffers; 4096 covers 512e and 4Kn volumes
        A = 4096;
        FILE_STORAGE_INFO si = {};
        if (GetFileInformationByHandleEx(h, FileStorageInfo, &si, sizeof(si))) {
            for (const ULONG s : { si.LogicalBytesPerSector, si.PhysicalBytesPerSectorForAtomicity }) {
                if (s > A && s <= 65536 && (s & (s - 1)) == 0) {
                    A = s;
                }
            }
        }

        iocp = CreateIoCompletionPort(h, nullptr, 1, 1);
        if (iocp == nullptr) {
            err = "CreateIoCompletionPort failed: " + win_err(GetLastError());
            return false;
        }
        return true;
    }

    bool alloc_buffers(std::string & err) {
        const int qd = p.queue_depth;
        bufs = (uint8_t *) VirtualAlloc(nullptr, (size_t) qd * slot_bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (!bufs) {
            err = "VirtualAlloc failed: " + win_err(GetLastError());
            return false;
        }
        slots.resize(qd);
        ents.resize(qd);
        free_slots.clear();
        for (int k = qd - 1; k >= 0; --k) {
            free_slots.push_back(k);
        }
        return true;
    }

    void close_all() {
        if (iocp) {
            CloseHandle(iocp);
            iocp = nullptr;
        }
        if (h != INVALID_HANDLE_VALUE) {
            CloseHandle(h);
            h = INVALID_HANDLE_VALUE;
        }
        if (bufs) {
            VirtualFree(bufs, 0, MEM_RELEASE);
            bufs = nullptr;
        }
    }

    // all misses in flight at once (up to queue_depth), completions reaped from the port in any order
    void read_misses(const uint8_t * fallback) {
        const size_t rb     = p.row_bytes;
        const int    n_miss = (int) miss.size();
        const int    qd     = p.queue_depth;

        int next     = 0;
        int inflight = 0;
        while (next < n_miss || inflight > 0) {
            while (next < n_miss && inflight < qd) {
                const int32_t   u = miss[next++];
                const read_span s = span_of(urow[u]);
                const int       k = free_slots.back();
                free_slots.pop_back();

                win_slot & sl = slots[k];
                memset(&sl.ov, 0, sizeof(sl.ov));
                sl.ov.Offset     = (DWORD) (s.start & 0xffffffffu);
                sl.ov.OffsetHigh = (DWORD) (s.start >> 32);
                sl.u    = u;
                sl.skip = s.skip;
                sl.t0   = dio_clock::now();

                const BOOL  done = ReadFile(h, bufs + (size_t) k * slot_bytes, (DWORD) s.len, nullptr, &sl.ov);
                const DWORD e    = done ? ERROR_SUCCESS : GetLastError();
                st.issue_us += dio_us(dio_clock::now() - sl.t0);
                if (!done && e != ERROR_IO_PENDING) {
                    // no completion packet is queued for a read that failed to start
                    fail_row(u, fallback, e);
                    free_slots.push_back(k);
                    continue;
                }
                st.inline_done += done ? 1 : 0;
                ++inflight;
                st.reads++;
            }
            if (inflight == 0) {
                break;
            }

            ULONG got = 0;
            if (!GetQueuedCompletionStatusEx(iocp, ents.data(), (ULONG) std::min<int>(inflight, (int) ents.size()), &got,
                        10000, FALSE)) {
                const DWORD e = GetLastError();
                if (e == WAIT_TIMEOUT) {
                    emit("%s: [TAG_FN_PLE_DIRECT_IO] warning: %d reads still pending after 10 s\n", __func__, inflight);
                    continue;
                }
                // the buffers stay alive with this object, so reads still in flight land in valid memory
                throw std::runtime_error("PLE direct I/O: GetQueuedCompletionStatusEx failed: " + win_err(e));
            }
            const auto t1 = dio_clock::now();
            for (ULONG j = 0; j < got; ++j) {
                win_slot * sl = reinterpret_cast<win_slot *>(ents[j].lpOverlapped);
                const int  k  = (int) (sl - slots.data());
                const DWORD nb = ents[j].dwNumberOfBytesTransferred;
                st.read_us += dio_us(t1 - sl->t0);
                if (sl->ov.Internal == 0 && (size_t) nb >= sl->skip + rb) {
                    memcpy(ubuf.data() + (size_t) sl->u * rb, bufs + (size_t) k * slot_bytes + sl->skip, rb);
                    st.bytes += nb;
                } else {
                    fail_row(sl->u, fallback, (unsigned long) sl->ov.Internal);
                }
                free_slots.push_back(k);
                --inflight;
            }
        }
    }
#else
    bool open_file(std::string & err) {
        int flags = O_RDONLY;
#ifdef O_CLOEXEC
        flags |= O_CLOEXEC;
#endif
        is_direct = false;
#ifdef O_DIRECT
        fd = ::open(p.path.c_str(), flags | O_DIRECT);
        if (fd >= 0) {
            is_direct = true;
        }
#endif
        if (fd < 0) {
            fd = ::open(p.path.c_str(), flags);
            if (fd < 0) {
                err = std::string("open failed: ") + strerror(errno);
                return false;
            }
#if defined(__APPLE__) && defined(F_NOCACHE)
            if (fcntl(fd, F_NOCACHE, 1) != -1) {
                is_direct = true;
            }
#endif
        }
        struct stat sb {};
        if (fstat(fd, &sb) != 0) {
            err = std::string("fstat failed: ") + strerror(errno);
            return false;
        }
        file_size = (uint64_t) sb.st_size;
        A = 4096;
        if (sb.st_blksize > (blksize_t) A && sb.st_blksize <= 65536 && (sb.st_blksize & (sb.st_blksize - 1)) == 0) {
            A = (size_t) sb.st_blksize;
        }
        return true;
    }

    bool alloc_buffers(std::string & err) {
        const int n_threads = std::max(1, std::min(p.queue_depth, 8));
        for (int t = 0; t < n_threads; ++t) {
            void * b = nullptr;
            if (posix_memalign(&b, A, slot_bytes) != 0) {
                err = "posix_memalign failed";
                return false;
            }
            tbufs.push_back((uint8_t *) b);
        }
        pool.start(n_threads - 1);
        return true;
    }

    void close_all() {
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
        for (uint8_t * b : tbufs) {
            free(b);
        }
        tbufs.clear();
    }

    void read_misses(const uint8_t * fallback) {
        const size_t rb     = p.row_bytes;
        const int    n_miss = (int) miss.size();

        struct acc_t { uint64_t reads = 0, bytes = 0; double us = 0; };
        std::vector<acc_t>   acc(tbufs.size());
        std::vector<int32_t> err_of(n_miss, 0);
        std::atomic<int>     next{0};

        auto work = [&](int tid) {
            uint8_t * buf = tbufs[tid];
            for (int i; (i = next.fetch_add(1)) < n_miss;) {
                const int32_t   u = miss[i];
                const read_span s = span_of(urow[u]);
                const auto      t0 = dio_clock::now();
                size_t got = 0;
                int    e   = 0;
                while (got < s.len) {
                    const ssize_t r = pread(fd, buf + got, s.len - got, (off_t) (s.start + got));
                    if (r < 0) {
                        if (errno == EINTR) {
                            continue;
                        }
                        e = errno;
                        break;
                    }
                    if (r == 0) {
                        break; // end of file: the row may still be complete
                    }
                    got += (size_t) r;
                }
#ifdef POSIX_FADV_DONTNEED
                if (!is_direct) {
                    posix_fadvise(fd, (off_t) s.start, (off_t) s.len, POSIX_FADV_DONTNEED);
                }
#endif
                acc[tid].reads++;
                acc[tid].bytes += got;
                acc[tid].us    += dio_us(dio_clock::now() - t0);
                if (e == 0 && got >= s.skip + rb) {
                    memcpy(ubuf.data() + (size_t) u * rb, buf + s.skip, rb);
                } else {
                    err_of[i] = e ? e : -1;
                }
            }
        };
        pool.run(work, n_miss <= 2);

        for (const auto & a : acc) {
            st.reads   += a.reads;
            st.bytes   += a.bytes;
            st.read_us += a.us;
        }
        for (int i = 0; i < n_miss; ++i) {
            if (err_of[i] != 0) {
                fail_row(miss[i], fallback, (unsigned long) err_of[i]);
            }
        }
    }
#endif
};

llama_ple_dio::llama_ple_dio(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

std::unique_ptr<llama_ple_dio> llama_ple_dio::open(const llama_ple_dio_params & params, std::string & err) {
    if (params.path.empty() || params.row_bytes == 0 || params.n_rows <= 0 || params.n_rows > INT32_MAX) {
        err = "bad parameters";
        return nullptr;
    }
    auto d = std::make_unique<impl>();
    d->p = params;
    d->p.queue_depth = std::max(1, std::min(params.queue_depth, 1024));

    if (!d->open_file(err)) {
        d->close_all();
        return nullptr;
    }
    const uint64_t end = params.offset + (uint64_t) params.n_rows * params.row_bytes;
    if (end > d->file_size) {
        err = "the table (ends at " + std::to_string(end) + ") lies outside the file (" + std::to_string(d->file_size) + " bytes)";
        d->close_all();
        return nullptr;
    }
    // largest aligned span of one row: it starts up to A-1 bytes into its first sector
    d->slot_bytes = (d->A - 1 + params.row_bytes + d->A - 1) / d->A * d->A;
    if (!d->alloc_buffers(err)) {
        d->close_all();
        return nullptr;
    }
    d->lru.init(params.cache_bytes, params.row_bytes);
    return std::unique_ptr<llama_ple_dio>(new llama_ple_dio(std::move(d)));
}

llama_ple_dio::~llama_ple_dio() {
    if (pimpl->p.stats_every > 0 && pimpl->st.calls > 0) {
        pimpl->emit("%s: [TAG_FN_PLE_DIRECT_IO] final: %s\n", __func__, pimpl->line().c_str());
    }
    // no read is in flight between calls, and idle POSIX workers never touch the buffers
    pimpl->close_all();
}

const llama_ple_dio_params & llama_ple_dio::params() const { return pimpl->p; }
size_t llama_ple_dio::align() const { return pimpl->A; }
bool llama_ple_dio::direct() const { return pimpl->is_direct; }

llama_ple_dio_stats llama_ple_dio::stats() const {
    std::lock_guard<std::mutex> lock(pimpl->mtx);
    return pimpl->st;
}

std::string llama_ple_dio::stats_line() const {
    std::lock_guard<std::mutex> lock(pimpl->mtx);
    return pimpl->line();
}

void llama_ple_dio::reset_stats() {
    std::lock_guard<std::mutex> lock(pimpl->mtx);
    pimpl->st = llama_ple_dio_stats();
}

void llama_ple_dio::read_rows(const int32_t * rows, int64_t n, uint8_t * dst, int64_t n_tokens, const uint8_t * fallback) {
    impl & d = *pimpl;
    std::lock_guard<std::mutex> lock(d.mtx);

    if (n <= 0) {
        return;
    }
    if (n > (int64_t) UINT32_MAX) {
        throw std::runtime_error("PLE direct I/O: too many rows in one call");
    }

    const auto   t0 = dio_clock::now();
    const size_t rb = d.p.row_bytes;

    // distinct rows in file order; upos maps each requested position to its distinct row
    d.order.resize(n);
    for (int64_t i = 0; i < n; ++i) {
        const int32_t r = rows[i];
        if (r < 0 || r >= d.p.n_rows) {
            throw std::runtime_error("PLE direct I/O: row " + std::to_string(r) + " out of range");
        }
        d.order[i] = ((uint64_t) (uint32_t) r << 32) | (uint64_t) i;
    }
    std::sort(d.order.begin(), d.order.end());
    d.urow.clear();
    d.upos.resize(n);
    for (int64_t i = 0; i < n; ++i) {
        const int32_t r = (int32_t) (d.order[i] >> 32);
        if (d.urow.empty() || d.urow.back() != r) {
            d.urow.push_back(r);
        }
        d.upos[(size_t) (d.order[i] & 0xffffffffu)] = (uint32_t) (d.urow.size() - 1);
    }
    const size_t nu = d.urow.size();
    d.ubuf.resize(nu * rb);

    d.miss.clear();
    for (size_t u = 0; u < nu; ++u) {
        if (const uint8_t * c = d.lru.get(d.urow[u])) {
            memcpy(d.ubuf.data() + u * rb, c, rb);
            d.st.hits++;
        } else {
            d.miss.push_back((int32_t) u);
        }
    }

    if (!d.miss.empty()) {
        d.read_misses(fallback);
        for (const int32_t u : d.miss) {
            d.lru.put(d.urow[u], d.ubuf.data() + (size_t) u * rb);
        }
    }

    for (int64_t i = 0; i < n; ++i) {
        memcpy(dst + (size_t) i * rb, d.ubuf.data() + (size_t) d.upos[i] * rb, rb);
    }

    const double us = dio_us(dio_clock::now() - t0);
    d.st.calls++;
    d.st.tokens  += (uint64_t) std::max<int64_t>(n_tokens, 0);
    d.st.rows    += (uint64_t) n;
    d.st.unique  += nu;
    d.st.call_us += us;
    d.st.call_us_max = std::max(d.st.call_us_max, us);

    const int every = d.p.stats_every;
    if ((every == 1 && d.st.calls == 256) || (every > 1 && d.st.calls % (uint64_t) every == 0)) {
        d.emit("%s: [TAG_FN_PLE_DIRECT_IO] %s\n", __func__, d.line().c_str());
    }
}

int64_t llama_ple_dio_file_size(const std::string & path) {
#if defined(_WIN32)
    const std::wstring w = dio_wpath(path);
    WIN32_FILE_ATTRIBUTE_DATA a;
    if (w.empty() || !GetFileAttributesExW(w.c_str(), GetFileExInfoStandard, &a) || (a.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
        return -1;
    }
    return (int64_t) (((uint64_t) a.nFileSizeHigh << 32) | a.nFileSizeLow);
#else
    struct stat sb {};
    if (stat(path.c_str(), &sb) != 0 || S_ISDIR(sb.st_mode)) {
        return -1;
    }
    return (int64_t) sb.st_size;
#endif
}

static void dio_copy_log(void (*log)(const char *), const char * fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (log) {
        log(buf);
    } else {
        fputs(buf, stderr);
    }
}

bool llama_ple_dio_copy(const std::string & src, uint64_t offset, uint64_t size, const std::string & dst, size_t chunk,
        std::string & err, void (*log)(const char * line)) {
    const std::string tmp = dst + ".tmp";
    chunk = std::min<size_t>(std::max<size_t>(chunk, 1 << 20), 64u << 20);

    const auto t0 = dio_clock::now();
    uint64_t pos = 0;
    int      last_tenth = 0;
    auto progress = [&]() {
        const int tenth = (int) (pos * 10 / std::max<uint64_t>(size, 1));
        if (tenth > last_tenth && tenth < 10) {
            last_tenth = tenth;
            const double s = std::chrono::duration<double>(dio_clock::now() - t0).count();
            dio_copy_log(log, "llama_ple_dio_copy: [TAG_FN_PLE_DIRECT_IO] copy %d%% (%.0f of %.0f MiB, %.2f GB/s)\n",
                    tenth * 10, pos / 1048576.0, size / 1048576.0, pos / std::max(s, 1e-9) / 1e9);
        }
    };

#if defined(_WIN32)
    const std::wstring ws = dio_wpath(src);
    const std::wstring wt = dio_wpath(tmp);
    const std::wstring wd = dio_wpath(dst);
    if (ws.empty() || wt.empty() || wd.empty()) {
        err = "cannot convert a path to UTF-16";
        return false;
    }

    // room for the copy plus 1 GiB
    const size_t   slash = wd.find_last_of(L"/\\");
    const std::wstring dir = slash == std::wstring::npos ? std::wstring(L".") : wd.substr(0, slash + 1);
    ULARGE_INTEGER free_bytes;
    if (GetDiskFreeSpaceExW(dir.c_str(), &free_bytes, nullptr, nullptr) && free_bytes.QuadPart < size + (1ull << 30)) {
        err = "not enough free space for " + std::to_string(size) + " bytes in " + dst;
        return false;
    }

    HANDLE hs = CreateFileW(ws.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
            OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (hs == INVALID_HANDLE_VALUE) {
        err = "cannot open " + src + ": " + dio_win_err(GetLastError());
        return false;
    }
    HANDLE hd = CreateFileW(wt.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH, nullptr);
    if (hd == INVALID_HANDLE_VALUE) {
        err = "cannot create " + tmp + ": " + dio_win_err(GetLastError());
        CloseHandle(hs);
        return false;
    }

    // unbuffered I/O: sector-aligned offsets, lengths and buffers on both volumes
    size_t A = 4096;
    for (HANDLE hh : { hs, hd }) {
        FILE_STORAGE_INFO si = {};
        if (GetFileInformationByHandleEx(hh, FileStorageInfo, &si, sizeof(si))) {
            for (const ULONG s : { si.LogicalBytesPerSector, si.PhysicalBytesPerSectorForAtomicity }) {
                if (s > A && s <= 65536 && (s & (s - 1)) == 0) {
                    A = s;
                }
            }
        }
    }
    chunk = chunk / A * A;

    FILE_ALLOCATION_INFO ai;
    ai.AllocationSize.QuadPart = (LONGLONG) size;
    SetFileInformationByHandle(hd, FileAllocationInfo, &ai, sizeof(ai)); // best effort: one extent run

    uint8_t * rbuf = (uint8_t *) VirtualAlloc(nullptr, chunk + A, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    uint8_t * wbuf = (uint8_t *) VirtualAlloc(nullptr, chunk,     MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    bool ok = rbuf && wbuf;
    if (!ok) {
        err = "VirtualAlloc failed";
    }

    // the table starts skip bytes into a sector; every source read starts at that same phase
    const size_t skip = (size_t) (offset % A);
    while (ok && pos < size) {
        const size_t   n    = (size_t) std::min<uint64_t>(chunk, size - pos);
        const uint64_t rs   = offset + pos - skip;
        const size_t   rlen = (skip + n + A - 1) / A * A;
        OVERLAPPED ov = {};
        ov.Offset     = (DWORD) (rs & 0xffffffffu);
        ov.OffsetHigh = (DWORD) (rs >> 32);
        DWORD got = 0;
        if (!ReadFile(hs, rbuf, (DWORD) rlen, &got, &ov) || got < skip + n) {
            err = "read of " + src + " at " + std::to_string(rs) + " failed: " + dio_win_err(GetLastError());
            ok = false;
            break;
        }
        memcpy(wbuf, rbuf + skip, n);
        const size_t wlen = (n + A - 1) / A * A;
        if (wlen > n) {
            memset(wbuf + n, 0, wlen - n);
        }
        OVERLAPPED ow = {};
        ow.Offset     = (DWORD) (pos & 0xffffffffu);
        ow.OffsetHigh = (DWORD) (pos >> 32);
        DWORD put = 0;
        if (!WriteFile(hd, wbuf, (DWORD) wlen, &put, &ow) || put != wlen) {
            err = "write of " + tmp + " at " + std::to_string(pos) + " failed: " + dio_win_err(GetLastError());
            ok = false;
            break;
        }
        pos += n;
        progress();
    }
    if (ok) {
        // the last write was padded to a whole sector
        FILE_END_OF_FILE_INFO eof;
        eof.EndOfFile.QuadPart = (LONGLONG) size;
        if (!SetFileInformationByHandle(hd, FileEndOfFileInfo, &eof, sizeof(eof))) {
            err = "cannot set the size of " + tmp + ": " + dio_win_err(GetLastError());
            ok = false;
        }
    }
    if (rbuf) {
        VirtualFree(rbuf, 0, MEM_RELEASE);
    }
    if (wbuf) {
        VirtualFree(wbuf, 0, MEM_RELEASE);
    }
    CloseHandle(hs);
    CloseHandle(hd);
    if (ok && !MoveFileExW(wt.c_str(), wd.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        err = "cannot rename " + tmp + " to " + dst + ": " + dio_win_err(GetLastError());
        ok = false;
    }
    if (!ok) {
        DeleteFileW(wt.c_str());
    }
#else
    int fs = ::open(src.c_str(), O_RDONLY);
    if (fs < 0) {
        err = "cannot open " + src + ": " + strerror(errno);
        return false;
    }
    int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        err = "cannot create " + tmp + ": " + strerror(errno);
        ::close(fs);
        return false;
    }
    std::vector<uint8_t> buf(chunk);
    bool ok = true;
    while (ok && pos < size) {
        const size_t n = (size_t) std::min<uint64_t>(chunk, size - pos);
        for (size_t done = 0; ok && done < n;) {
            const ssize_t r = pread(fs, buf.data() + done, n - done, (off_t) (offset + pos + done));
            if (r < 0 && errno == EINTR) {
                continue;
            }
            if (r <= 0) {
                err = "read of " + src + " failed: " + (r < 0 ? strerror(errno) : "end of file");
                ok = false;
                break;
            }
            done += (size_t) r;
        }
        for (size_t done = 0; ok && done < n;) {
            const ssize_t r = pwrite(fd, buf.data() + done, n - done, (off_t) (pos + done));
            if (r < 0 && errno == EINTR) {
                continue;
            }
            if (r <= 0) {
                err = "write of " + tmp + " failed: " + strerror(errno);
                ok = false;
                break;
            }
            done += (size_t) r;
        }
        // keep both files out of the page cache (dirty pages must reach the disk before they can be dropped)
        if (ok && fdatasync(fd) != 0) {
            err = "fdatasync of " + tmp + " failed: " + strerror(errno);
            ok = false;
        }
#ifdef POSIX_FADV_DONTNEED
        posix_fadvise(fs, (off_t) (offset + pos), (off_t) n, POSIX_FADV_DONTNEED);
        posix_fadvise(fd, (off_t) pos, (off_t) n, POSIX_FADV_DONTNEED);
#endif
        pos += ok ? n : 0;
        progress();
    }
    ::close(fs);
    if (::close(fd) != 0 && ok) {
        err = "close of " + tmp + " failed: " + strerror(errno);
        ok = false;
    }
    if (ok && rename(tmp.c_str(), dst.c_str()) != 0) {
        err = "cannot rename " + tmp + " to " + dst + ": " + strerror(errno);
        ok = false;
    }
    if (!ok) {
        unlink(tmp.c_str());
    }
#endif
    if (ok) {
        const double s = std::chrono::duration<double>(dio_clock::now() - t0).count();
        dio_copy_log(log, "%s: [TAG_FN_PLE_DIRECT_IO] copied %.0f MiB to %s in %.2f s\n", __func__, size / 1048576.0,
                dst.c_str(), s);
    }
    return ok;
}
