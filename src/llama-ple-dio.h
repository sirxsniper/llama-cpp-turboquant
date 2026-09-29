#pragma once

// [TAG_FN_PLE_DIRECT_IO] rows of a large on-disk table read straight from the file with unbuffered I/O.
//
// LLAMA_PLE_DIRECT_IO=1 (default off, qwen4exp only): the PLE n-gram table (per_layer_token_embd, 28.8 GB iq4_nl in
// Flash-Next) is never mapped in or paged: each ubatch's rows are read from the GGUF shard with unbuffered I/O
// (Windows: FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED on an I/O completion port, all rows in flight at once;
// POSIX: O_DIRECT, or pread + posix_fadvise(DONTNEED), on a small thread pool), so random row reads no longer leave
// 4 KB pages in RAM that displace the mapped expert pages. The caller dequantizes the raw rows with the same to_float
// the CPU GET_ROWS uses, so the values are bit-identical to the mapped path.
//
// Windows serves unbuffered reads of a file one at a time while the file has a data section, i.e. while any process
// maps it (measured: ~10K reads/s at any queue depth vs ~130-170K unmapped). The shard that holds the table is mapped
// for its other tensors, so LLAMA_PLE_DIO_FILE points the reads at a copy of the table that nothing maps.
//
// This file depends on nothing but the C++ library and the OS, so tests can compile it directly.
//
// Environment (read by the model, see qwen4exp.cpp):
//   LLAMA_PLE_DIRECT_IO=1        switch it on
//   LLAMA_PLE_DIO_FILE=<path>    read an unmapped copy of the table (raw rows, row 0 at offset 0); a missing or wrong
//                                copy is (re)made at load with unbuffered I/O (28.8 GB for Flash-Next); every load
//                                checks 32 sample rows against the model file
//   LLAMA_PLE_DIO_CACHE_MB=64    LRU of recently read rows (0 = off)
//   LLAMA_PLE_DIO_QD=64          reads in flight at most (1..1024)
//   LLAMA_PLE_DIO_STATS=1        one statistics line after 256 ubatches and one when the model is freed;
//                                N > 1 = a line every N ubatches

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

struct llama_ple_dio_params {
    std::string path;               // file that holds the table (UTF-8)
    uint64_t    offset      = 0;    // file offset of row 0
    size_t      row_bytes   = 0;    // bytes per row
    int64_t     n_rows      = 0;    // rows in the table
    size_t      cache_bytes = 0;    // LRU budget for recent rows, 0 = no cache
    int         queue_depth = 64;   // reads in flight at most
    int         stats_every = 0;    // 0 = no statistics, 1 = after 256 ubatches + at close, N > 1 = every N ubatches
    void     (* log)(const char * line) = nullptr; // one line per call, ends with '\n'; nullptr = stderr
};

struct llama_ple_dio_stats {
    uint64_t calls      = 0; // read_rows() calls (ubatches)
    uint64_t tokens     = 0; // tokens the calls were made for
    uint64_t rows       = 0; // rows requested
    uint64_t unique     = 0; // distinct rows per call, summed
    uint64_t hits       = 0; // distinct rows served from the LRU
    uint64_t reads      = 0; // unbuffered reads issued
    uint64_t bytes      = 0; // bytes those reads transferred (sector aligned)
    uint64_t fallbacks  = 0; // rows copied from the fallback memory after a failed read
    double   call_us    = 0; // wall time of all calls
    double   call_us_max = 0;
    double   read_us    = 0; // issue-to-completion time of all reads
    double   issue_us   = 0; // time spent inside the read calls (Windows: ReadFile; ~0 when the reads really overlap)
    uint64_t inline_done = 0; // Windows: reads that ReadFile completed before returning (not overlapped)
};

struct llama_ple_dio {
    // nullptr, with the reason in err, when the file cannot be opened for unbuffered reads or the parameters are bad
    static std::unique_ptr<llama_ple_dio> open(const llama_ple_dio_params & params, std::string & err);

    ~llama_ple_dio();

    // copy the raw bytes of rows[0..n) into dst (n * row_bytes bytes, in the order given; repeated rows are read once).
    // fallback (may be null) is the same table in memory: a row whose read fails is copied from it (logged once);
    // without it a failed read throws. n_tokens only feeds the statistics. Thread-safe (serialised).
    void read_rows(const int32_t * rows, int64_t n, uint8_t * dst, int64_t n_tokens, const uint8_t * fallback);

    const llama_ple_dio_params & params() const;
    size_t              align() const;  // read alignment in bytes (offsets, sizes and buffers)
    bool                direct() const; // false: POSIX fallback without O_DIRECT (pread + fadvise DONTNEED)
    llama_ple_dio_stats stats() const;
    std::string         stats_line() const;

    struct impl;

private:
    explicit llama_ple_dio(std::unique_ptr<impl> p);

    std::unique_ptr<impl> pimpl;
};

// LLAMA_PLE_DIRECT_IO is set to a non-zero value
bool llama_ple_dio_requested();

// size of a file in bytes, -1 when it does not exist
int64_t llama_ple_dio_file_size(const std::string & path);

// write bytes [offset, offset + size) of src to a new file dst (via dst + ".tmp", then a rename), with unbuffered reads
// and writes so neither file goes through the page cache. chunk: bytes per read (rounded to the sector size).
// false + err on failure (dst is left as it was). log (may be null) gets progress lines.
bool llama_ple_dio_copy(const std::string & src, uint64_t offset, uint64_t size, const std::string & dst, size_t chunk,
        std::string & err, void (*log)(const char * line));
