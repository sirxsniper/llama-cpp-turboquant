#pragma once

// [TAG_FN_L18_KEEPCTX] Conversation store of the slots (LLAMA_FN_L18_KEEPCTX=1).
//
// A request with another prompt prefix (a chat title, a second conversation) used to take the slot and drop the
// conversation in it: the next message of that conversation processed the whole prompt again (45,404 tokens, 23.6 s
// before the first token on Flash-Next). Before the slot's state is replaced, the store writes it to a file: the
// target and draft sequence states (llama_state_seq_write_stream_ext: the exact bytes llama_state_seq_get_data_ext
// gives), the slot's context checkpoints and the drafter's carry. A later prompt that continues a stored conversation
// further than the slot content gets it back, and only its new tokens are processed.
//
// The files bypass the OS file cache (Windows: FILE_FLAG_NO_BUFFERING), so a model bigger than RAM keeps its cached
// pages, and they are deleted when closed (FILE_FLAG_DELETE_ON_CLOSE), also when the process ends abnormally.

#include "server-task.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct llama_context;
struct common_speculative;

struct server_keepctx_params {
    std::string dir;          // folder of the store files
    uint64_t    budget = 0;   // bytes on disk for all conversations
    int32_t     n_max  = 4;   // conversations kept
    bool        verify = false; // after a restore, compare a hash of the restored states with the stored bytes (tests)
    bool        hot    = false; // [TAG_FN_L18_KEEPCTX_HOT] the MoE hot set's counts go with the conversation

    // [TAG_FN_L19_CONVSTORE_EVICT] evict by value per byte (convstore_gdsf) instead of least recently stored
    bool        gdsf         = false;
    double      fixed_tokens = 0.0;   // the fixed prefill cost of a restore's alternative, in tokens

    // [TAG_FN_L24_KEEPVAL] evict by re-read cost x recency (convstore_val, before gdsf) and keep the biggest active
    // conversation against the count limit; the disk budget and the free disk space (disk_margin kept) still bind
    bool        val          = false;
    double      val_half_s   = 900.0;  // recency: an entry this old counts half
    double      val_active_s = 3600.0; // the biggest entry used within this is protected
    uint64_t    disk_margin  = 0;      // free bytes kept on the store's disk (0: not checked)

    // [TAG_FN_L20_CONVFIX_IO] a store and a restore reuse their memory: the pipe's blocks are allocated once, and a
    // restore puts the stored checkpoints into the outgoing ring's buffers (no fresh, zero-filled 112.6 MiB allocation
    // per checkpoint and no free of the old ones while the next prompt streams the experts). The bytes are the same.
    bool        io_reuse = false;
    // [TAG_FN_L20_CONVFIX_TRACE] the time of each part of a store / restore in its log line
    bool        trace    = false;
    // [TAG_FN_L20_CONVFIX_LAZY] a stored conversation's checkpoints start at sector offsets in its file, and a restore reads
    // only the target and draft states: the checkpoints stay on disk as the slot's checkpoints on disk (the
    // [TAG_FN_L19_CONVSTORE_SPILL] list: read back when a prompt resumes at one of them, stored with the conversation
    // again without a write). A restore moves ~half the bytes (4 checkpoints of 112.6 MiB beside a ~420 MiB state).
    bool        lazy     = false;
};

class server_keepctx_file;

// [TAG_FN_L19_CONVSTORE_SPILL] the write of a checkpoint to disk, done by the store's writer thread
struct server_keepctx_spill_state {
    std::mutex              mu;
    std::condition_variable cv;
    int                     done      = 0;   // 0 being written, 1 written, -1 failed
    uint64_t                hash_data = 0;   // params.verify: a hash of the bytes written
    double                  ms        = 0.0; // the write
};

// [TAG_FN_L19_CONVSTORE_SPILL] a context checkpoint kept on disk: its position, where its bytes are, and a hash of the
// conversation's tokens before it (it is valid only while the tokens in front of it are the same)
struct server_keepctx_spilled {
    int64_t   n_tokens = 0;
    int       id_task  = -1;
    int32_t   pos_min  = 0;
    int32_t   pos_max  = 0;
    uint64_t  size_tgt  = 0;
    uint64_t  size_dft  = 0;
    uint64_t  size_spec = 0;
    uint64_t  off       = 0;   // in file, a multiple of the sector size
    uint64_t  hash      = 0;   // keepctx_prefix_hash of the tokens [0, n_tokens)

    std::shared_ptr<server_keepctx_file>        file;
    std::shared_ptr<server_keepctx_spill_state> st;   // the write (a reader waits for it)

    uint64_t bytes() const { return size_tgt + size_dft + size_spec; }
};

// [TAG_FN_L19_CONVSTORE] what the store keeps about a conversation besides its state
struct server_keepctx_conv {
    int64_t hits     = 0;     // restores it served
    int32_t sys_end  = -1;    // the end of its system + tools block (its first user message, -1 unknown)
    int32_t n_prompt = -1;    // the length of its last prompt (before the answer, -1 unknown)
    bool    rerender = false; // [TAG_FN_L19_CONVSTORE_OPENER] its client sent the last answer back re-rendered (the
                              // reasoning dropped): the next prompt left the previous one inside its last tokens
};

struct server_keepctx_entry {
    uint64_t id = 0;

    server_prompt prompt;               // tokens, and the checkpoint positions (their bytes are in the file)
    std::vector<uint64_t> ckpt_sizes;   // per checkpoint: target, draft and drafter bytes

    uint64_t size_tgt = 0;
    uint64_t size_dft = 0;
    uint64_t n_bytes  = 0;              // bytes in the file

    std::vector<uint8_t> carry;         // the drafter's carry (common_speculative_get_carry)

    uint64_t hash_tgt = 0;              // params.verify: hashes of the stored target and draft state bytes
    uint64_t hash_dft = 0;

    // [TAG_FN_L20_CONVFIX_LAZY] params.lazy: each checkpoint's offset in the file (a multiple of the sector size) and, with
    // params.verify, a hash of its bytes
    std::vector<uint64_t> ckpt_off;
    std::vector<uint64_t> ckpt_hash;

    std::vector<float> hot;             // params.hot: the MoE hot set's decayed counts when it left the slot

    std::shared_ptr<server_keepctx_file> file;

    // [TAG_FN_L19_CONVSTORE] the conversation's checkpoints on disk (beside the ring's, in file), what is known about it
    // and its eviction value
    std::vector<server_keepctx_spilled> spilled;
    server_keepctx_conv                 conv;
    double                              value = 0.0;
    double                              t_use = 0.0; // [TAG_FN_L24_KEEPVAL] steady-clock seconds of the store or the last restore

    uint64_t n_bytes_all() const;       // n_bytes + the spilled checkpoints' bytes
};

// [TAG_FN_L19_CONVSTORE_SPILL] a hash of tokens [0, n) (media chunks by their id): 0 when it cannot be computed
uint64_t keepctx_prefix_hash(const server_tokens & tokens, size_t n);

// drop the spilled checkpoints (ascending n_tokens) whose tokens in front differ from tokens [0, n_tokens); the number dropped
int keepctx_spill_validate(const server_tokens & tokens, std::vector<server_keepctx_spilled> & list);

// [TAG_FN_L19_CONVSTORE_SPILL] the checkpoints of a slot's conversation that its ring dropped, each in its own file (shared
// with the stored conversations that hold it, deleted with its last holder, so the disk holds only the ones kept). They
// are valid only while the slot holds the same tokens in front of them.
struct server_keepctx_spill {
    std::vector<server_keepctx_spilled> list;              // ascending n_tokens
    uint64_t                            n_written     = 0; // checkpoints written (statistics)
    uint64_t                            bytes_written = 0;

    void clear() { list.clear(); }
    // drop the ones beyond n_keep tokens (the prompt keeps only these)
    void keep_first(int64_t n_keep);
};

struct server_keepctx {
    explicit server_keepctx(const server_keepctx_params & params);
    ~server_keepctx();

    bool ok() const { return ok_; }

    const server_keepctx_params & params() const { return params_; }

    std::list<server_keepctx_entry> entries; // least recently stored first

    uint64_t n_bytes() const;

    // store the sequence seq_id with its prompt (tokens and checkpoints). Older states of the same conversation (entries
    // whose tokens are a prefix of it) and then the oldest entries are dropped to stay within the budget and the count,
    // never `keep`. false: nothing stored (too big, no room beside `keep`, or an I/O error).
    // [TAG_FN_L19_CONVSTORE] spill (may be null): the slot's checkpoints on disk, kept with the entry (shared files);
    // conv (may be null): what is known about the conversation; supersede_tol: see drop_prefixes_of
    bool save(const server_prompt & prompt, llama_context * ctx_tgt, llama_context * ctx_dft, common_speculative * spec,
              llama_seq_id seq_id, const server_keepctx_entry * keep, std::string & info,
              const server_keepctx_spill * spill = nullptr, const server_keepctx_conv * conv = nullptr, int32_t supersede_tol = 0);

    // restore the entry into seq_id and prompt. The entry is removed, unless keep (the prompt continues only a part of
    // it: the rest stays stored for a later continuation) and the restore worked. false: the sequence and prompt are in
    // an unknown state and the caller must clear them.
    // [TAG_FN_L19_CONVSTORE] spill / conv (may be null): receive the entry's checkpoints on disk and what is known about
    // it (its restores counting this one)
    // [TAG_FN_L20_CONVFIX_IO] reuse (may be null): checkpoints whose buffers the restored checkpoints take (the slot's
    // outgoing ring); the ones left over are given back in it
    bool load(std::list<server_keepctx_entry>::iterator it, server_prompt & prompt, llama_context * ctx_tgt,
              llama_context * ctx_dft, common_speculative * spec, llama_seq_id seq_id, bool keep, std::string & info,
              server_keepctx_spill * spill = nullptr, server_keepctx_conv * conv = nullptr,
              std::list<common_prompt_checkpoint> * reuse = nullptr);

    // remove the entries whose tokens are a prefix of tokens (an older state of the same conversation), except keep
    // [TAG_FN_L19_CONVSTORE] tol > 0: also the entries that leave tokens within their last tol tokens (a conversation
    // whose client re-rendered the last answer resumed from the entry before its end: the entry is superseded)
    int drop_prefixes_of(const server_tokens & tokens, const server_keepctx_entry * keep, int32_t tol = 0);

    // [TAG_FN_L19_CONVSTORE_SPILL] write a checkpoint to a new file and add it to the list (ascending). Its bytes are
    // moved out of c: async (the default), the writer thread writes them and keeps the buffer for take_spare(); else they
    // are written at once and given back to c. false: not written (I/O error), the checkpoint is lost as before.
    bool spill_write(server_keepctx_spill & spill, common_prompt_checkpoint & c, uint64_t prefix_hash, bool async,
                     std::string & info);

    // read a spilled checkpoint back into c (positions and bytes), after its write is done
    bool spill_read(const server_keepctx_spilled & s, common_prompt_checkpoint & c, std::string & info);

    // a buffer the writer thread is done with, for the next checkpoint (buf stays empty when there is none)
    void take_spare(std::vector<uint8_t> & buf);

    // [TAG_FN_L19_CONVSTORE_EVICT] statistics of the store
    uint64_t n_evicted_gdsf = 0;

private:
    bool selftest(std::string & err);
    // drop the entry of the lowest value (or the oldest), never keep; false when only keep is left
    // [TAG_FN_L24_KEEPVAL] budget: the room is for the disk (the protected entry may go, last), else for the count;
    // freed: the bytes of the evicted entry
    bool evict_one(const server_keepctx_entry * keep, bool budget = true, uint64_t * freed = nullptr);

    // [TAG_FN_L19_CONVSTORE_SPILL] the writer thread and a checkpoint's bytes on their way to disk
    struct spill_job {
        std::shared_ptr<server_keepctx_file>        file;
        std::shared_ptr<server_keepctx_spill_state> st;
        std::vector<uint8_t>                        tgt;
        std::vector<uint8_t>                        dft;
        std::vector<uint8_t>                        spec;
        bool                                        verify = false;
    };
    void wr_run();
    void wr_job(spill_job & job);
    std::thread                             wr_th_;
    std::mutex                              wr_mu_;
    std::condition_variable                 wr_cv_;
    std::deque<std::unique_ptr<spill_job>>  wr_q_;
    bool                                    wr_stop_ = false;
    std::vector<std::vector<uint8_t>>       spare_;   // buffers of written checkpoints (wr_mu_)

    // [TAG_FN_L20_CONVFIX_IO] the blocks of the pipes of save / load / spill_read (the server thread, one at a time)
    uint8_t *             io_mem_ = nullptr;

    server_keepctx_params params_;
    bool                  ok_     = false;
    uint64_t              n_next_ = 0;
    double                gdsf_L_ = 0.0; // [TAG_FN_L19_CONVSTORE_EVICT] the value of the last evicted entry
};
